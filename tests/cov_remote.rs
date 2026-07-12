//! Integration coverage for the iroh fleet — hub side.
//!
//! Targets branches the existing remote suite (`remote_hub_e2e`, `remote_pairing`,
//! `hub_server`, `chat_fleet`, `control_fleet_routes`) leaves uncovered:
//!
//! - The node CONTROL plane over a REAL `higgs --node`: every `higgs/node/*` dispatch
//!   arm (`dispatch_node_control`) — bad params, unknown worker, the reserved
//!   `update` refusal (HG026), unknown-method, plus the happy load/status/kill/unload
//!   Ok arms — driven through a raw `NodeTransport` (`node/control.rs`, `node/runtime.rs`
//!   unload/kill/status arms, `node/transport.rs` request path, `node/mod.rs`
//!   `handle_node_stream` unknown-method arm).
//! - The DATA plane: a remote chat with tools + chat-template kwargs (transport `chat`
//!   optional-field inserts), and a chat to an unknown worker (the relay's `chat_handle`
//!   error arm in `node/data.rs`).
//! - `HubFleet` admission control: a stale admission-generation refusal, `disconnect_all`
//!   draining a live transport, `seed_node`/`bus`, and the unknown-/disconnected-node
//!   error arms (`node/fleet.rs`).
//! - Node-side HELLO validation: a hub that closes before replying (EOF), a hub HELLO
//!   with the wrong role, and an unexpected reply frame (`node/mod.rs` `connect_node` +
//!   `validate_hub_hello`).
//! - Pure registry + identity units reachable from the crate API (`node/worker_id.rs`,
//!   `node/identity.rs`).
//!
//! All node-spawning work is consolidated into ONE test; the rest run in-process or are
//! pure, so the suite stays fast. Skips only the one node test when no tiny GGUF is present.

mod common;

use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use iroh_tickets::endpoint::EndpointTicket;
use serde_json::json;
use tokio::io::{AsyncBufReadExt, BufReader};

use higgs::auth::{Allowlist, PairingTokens};
use higgs::diagnostic::HiggsError;
use higgs::log_bus::LogBus;
use higgs::node::fleet::HubFleet;
use higgs::node::identity::load_or_create_secret;
use higgs::node::transport::{ChatDone, NodeTransport};
use higgs::node::worker_id::{WorkerId, WorkerRegistry};
use higgs::node::{
    connect_node, dial_and_hello, gate_connection, GateOutcome, HubIdentity, HELLO_DEADLINE,
};
use higgs::remote::{
    HelloResult, ALPN, M_NODE_KILL, M_NODE_LOAD, M_NODE_SCAN, M_NODE_STATUS, M_NODE_SYSINFO,
    M_NODE_UNLOAD, M_NODE_UPDATE,
};
use higgs::rpc::{self, RpcFrame, RpcNotification, RpcResponse};

use common::{stage_tiny_model, tiny_gguf_path, TINY_MODEL_ID};

// ── shared helpers (copied from the remote test harness; common/mod.rs is off-limits) ─────

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// A spawned `higgs --node` child. SIGTERM (graceful, flushes llvm-cov) + reap on drop.
struct NodeProc(Child);
impl Drop for NodeProc {
    fn drop(&mut self) {
        // SAFETY: a plain kill(2) on our own child's pid.
        unsafe { libc::kill(self.0.id() as libc::pid_t, libc::SIGTERM) };
        let _ = self.0.wait();
    }
}

/// Spawn a real `higgs --node <ticket> <token>` daemon dialing our in-process hub over
/// hermetic iroh, with its own isolated `HIGGS_HOME` and a staged model dir.
#[allow(clippy::zombie_processes)] // reaped by NodeProc::drop (SIGTERM + wait)
fn spawn_node(ticket: &str, token: &str, home: &Path, model_dir: &Path) -> NodeProc {
    NodeProc(
        Command::new(env!("CARGO_BIN_EXE_higgs"))
            .arg("--node")
            .arg(ticket)
            .arg(token)
            .env("HIGGS_HOME", home)
            .env("HIGGS_MODEL_DIR", model_dir)
            .env("HIGGS_IROH_LOCAL", "1")
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("spawn higgs --node"),
    )
}

/// A hermetic (relay-disabled) iroh endpoint on the higgs remote ALPN.
async fn minimal_ep() -> iroh::Endpoint {
    iroh::Endpoint::builder(iroh::endpoint::presets::Minimal)
        .alpns(vec![ALPN.to_vec()])
        .relay_mode(iroh::RelayMode::Disabled)
        .bind()
        .await
        .expect("bind endpoint")
}

/// Accept + gate the next inbound node connection, returning `(conn, peer_id)`.
async fn admit_node(
    hub: &iroh::Endpoint,
    allow: &mut Allowlist,
    tokens: &mut PairingTokens,
    hub_id: &str,
) -> (iroh::endpoint::Connection, String) {
    let incoming = tokio::time::timeout(Duration::from_secs(30), hub.accept())
        .await
        .expect("node dialed within 30s")
        .expect("incoming");
    let conn = incoming.await.expect("connection");
    let peer = conn.remote_id().to_string();
    let outcome = gate_connection(
        &conn,
        allow,
        tokens,
        now_ms(),
        &HubIdentity::new(hub_id),
        Some("cov".into()),
        HELLO_DEADLINE,
    )
    .await;
    assert!(
        matches!(outcome, GateOutcome::Admitted { .. }),
        "admitted: {outcome:?}"
    );
    (conn, peer)
}

/// An empty, uniquely-pathed allowlist under the OS temp dir (writable, so the gate's
/// post-pairing `save()` succeeds).
fn temp_allowlist() -> Allowlist {
    let n = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "higgs-cov-remote-{}-{n:x}.json",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&path);
    Allowlist::load(&path).expect("empty allowlist loads")
}

/// Keeps the two endpoints + the node-side connection alive so a hub-side `Connection`
/// obtained from [`admitted_pair`] stays open for the test's duration.
struct KeepAlive {
    _hub_ep: iroh::Endpoint,
    _node_ep: iroh::Endpoint,
    _node_conn: iroh::endpoint::Connection,
}

/// Establish ONE fully-admitted (HELLO-exchanged) iroh connection between two in-process
/// endpoints and return the HUB side. No node daemon, no serving loop — just a live
/// `Connection` the fleet can hold a transport over.
async fn admitted_pair() -> (iroh::endpoint::Connection, KeepAlive) {
    let hub = minimal_ep().await;
    let node = minimal_ep().await;
    let hub_addr = hub.addr();
    let hub_id = hub.id().to_string();
    let node_id = node.id().to_string();
    let mut tokens = PairingTokens::new();
    let tok = tokens.mint(now_ms(), 600_000);

    let hub_task = tokio::spawn(async move {
        let mut allow = temp_allowlist();
        let incoming = hub.accept().await.expect("incoming");
        let conn = incoming.await.expect("connection");
        let out = gate_connection(
            &conn,
            &mut allow,
            &mut tokens,
            now_ms(),
            &HubIdentity::new(hub_id),
            Some("cov".into()),
            HELLO_DEADLINE,
        )
        .await;
        assert!(
            matches!(out, GateOutcome::Admitted { .. }),
            "admitted: {out:?}"
        );
        (conn, hub)
    });

    let (node_conn, _res) = connect_node(&node, hub_addr, node_id, String::new(), Some(tok))
        .await
        .expect("node connects + completes HELLO");
    let (hub_conn, hub_ep) = hub_task.await.expect("hub gate task");
    (
        hub_conn,
        KeepAlive {
            _hub_ep: hub_ep,
            _node_ep: node,
            _node_conn: node_conn,
        },
    )
}

/// Drain a remote chat: count merged deltas AND resolve the final outcome future.
async fn drain_chat(
    rx: higgs::delta_queue::DeltaReceiver,
    fut: ChatDone,
) -> (usize, Result<serde_json::Value, HiggsError>) {
    let collector = tokio::spawn(async move {
        let mut rx = rx;
        let mut n = 0usize;
        while rx.recv().await.is_some() {
            n += 1;
        }
        n
    });
    let res = fut.await;
    let n = collector.await.expect("collector join");
    (n, res)
}

/// Reply one hand-built frame to a node's HELLO on `conn`, playing a MISBEHAVING hub: accept
/// the node's control stream, drain its HELLO request, then write `frame` (or, when `None`,
/// simply finish the stream WITHOUT replying → the node reads EOF).
async fn misbehaving_hub_reply(conn: &iroh::endpoint::Connection, frame: Option<RpcFrame>) {
    let (mut send, recv) = conn.accept_bi().await.expect("accept control stream");
    let mut lines = BufReader::new(recv).lines();
    let _hello_req = lines.next_line().await.expect("read node HELLO req");
    if let Some(frame) = frame {
        // `write_all`/`finish` are inherent iroh SendStream methods (no AsyncWriteExt).
        send.write_all(format!("{}\n", rpc::encode(&frame)).as_bytes())
            .await
            .expect("write reply frame");
    }
    send.finish().expect("finish");
    // Hold the connection open briefly so the node reads its reply / EOF before we drop it.
    tokio::time::sleep(Duration::from_millis(200)).await;
}

// ── the ONE node-spawning test: control + data plane RPC arms over a real node ────────────

/// Drive a REAL `higgs --node` over iroh with a raw `NodeTransport` and exercise every
/// `higgs/node/*` control arm + the chat data relay:
///
/// - `sysinfo`/`scan` happy replies (control Ok arms);
/// - bad-params `load`/`status` → INVALID_PARAMS (control parse-error arms);
/// - unknown-worker `unload`/`kill`/`status` → the runtime's `no_worker` error (control Err
///   arms + `runtime` Unload/Kill `None` arm);
/// - the reserved `update` method → typed HG026 refusal;
/// - an unknown `higgs/node/*` method (control dispatch fallthrough) AND a non-namespaced
///   method (`handle_node_stream`'s method-not-found arm);
/// - a real `load` → `status` → `kill` and a second `load` → chat (tools + kwargs) → `unload`
///   (control Ok arms + transport chat optional-field inserts + the data relay happy path);
/// - a chat to an unknown worker → the relay's `chat_handle` error arm.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn remote_node_control_and_chat_rpc_arms() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("SKIP remote_node_control_and_chat_rpc_arms: no tiny GGUF (set HIGGS_TEST_GGUF)");
        return;
    };
    let scan_root = stage_tiny_model(&gguf);
    let node_home = tempfile::tempdir().expect("node home");

    let hub = minimal_ep().await;
    let hub_id = hub.id().to_string();
    let ticket = EndpointTicket::new(hub.addr()).to_string();
    let mut allow = Allowlist::load(&node_home.path().join("hub-pairings.json")).unwrap();
    let mut tokens = PairingTokens::new();
    let token = tokens.mint(now_ms(), 600_000);

    let _node = spawn_node(&ticket, &token, node_home.path(), scan_root.path());

    let (conn, _peer) = admit_node(&hub, &mut allow, &mut tokens, &hub_id).await;
    let transport = NodeTransport::new(conn);

    // ── control-plane happy replies ──
    let sysinfo = transport
        .request(M_NODE_SYSINFO, json!({}))
        .await
        .expect("sysinfo ok");
    assert!(
        sysinfo.get("hardware").is_some() && sysinfo.get("runtime").is_some(),
        "sysinfo carries hardware+runtime: {sysinfo}"
    );
    let scan = transport
        .request(M_NODE_SCAN, json!({}))
        .await
        .expect("scan ok");
    assert!(
        scan["models"]
            .as_array()
            .is_some_and(|m| m.iter().any(|x| x["id"] == TINY_MODEL_ID)),
        "node catalog lists the staged model: {scan}"
    );

    // ── control-plane error arms ──
    // bad params (no `id`) → INVALID_PARAMS, relayed as a WorkerRpc error.
    let e = transport
        .request(M_NODE_LOAD, json!({}))
        .await
        .expect_err("load with no id fails");
    assert!(
        matches!(e, HiggsError::WorkerRpc { .. }),
        "bad load params → WorkerRpc, got {e:?}"
    );
    // bad params (no `worker_id`) on status → INVALID_PARAMS.
    assert!(
        transport.request(M_NODE_STATUS, json!({})).await.is_err(),
        "status with no worker_id fails"
    );
    // unknown worker → the runtime's `no_worker` error on unload/kill/status.
    assert!(
        transport
            .request(M_NODE_UNLOAD, json!({ "worker_id": 9999 }))
            .await
            .is_err(),
        "unloading an unknown worker fails"
    );
    assert!(
        transport
            .request(M_NODE_KILL, json!({ "worker_id": 9999 }))
            .await
            .is_err(),
        "killing an unknown worker fails"
    );
    assert!(
        transport
            .request(M_NODE_STATUS, json!({ "worker_id": 9999 }))
            .await
            .is_err(),
        "status of an unknown worker fails"
    );
    // the reserved update handshake → typed HG026 refusal.
    let e = transport
        .request(M_NODE_UPDATE, json!({}))
        .await
        .expect_err("update refused");
    assert!(
        matches!(&e, HiggsError::WorkerRpc { worker_code: Some(c), .. } if c == "HG026"),
        "update → HG026 refusal, got {e:?}"
    );
    // an unknown higgs/node/* method → the control dispatch method-not-found arm.
    assert!(
        transport
            .request("higgs/node/bogus", json!({}))
            .await
            .is_err(),
        "an unknown higgs/node/* method is rejected"
    );
    // a non-namespaced method → handle_node_stream's method-not-found arm.
    assert!(
        transport
            .request("garbage/method", json!({}))
            .await
            .is_err(),
        "a non-namespaced method is rejected"
    );

    // ── control-plane Ok arms: load → status → kill ──
    let loaded = transport
        .request(M_NODE_LOAD, json!({ "id": TINY_MODEL_ID }))
        .await
        .expect("real load ok");
    let w1 = loaded["worker_id"]
        .as_u64()
        .expect("worker id in load reply") as u32;
    let status = transport
        .request(M_NODE_STATUS, json!({ "worker_id": w1 }))
        .await
        .expect("status of a live worker ok");
    assert!(status.is_object(), "worker status is an object: {status}");
    transport
        .request(M_NODE_KILL, json!({ "worker_id": w1 }))
        .await
        .expect("kill of a live worker ok");

    // ── data-plane happy: load → chat (tools + kwargs) → unload ──
    let loaded2 = transport
        .request(M_NODE_LOAD, json!({ "id": TINY_MODEL_ID }))
        .await
        .expect("second load ok");
    let w2 = loaded2["worker_id"].as_u64().expect("worker id 2") as u32;

    let tools = r#"[{"type":"function","function":{"name":"noop","description":"n","parameters":{"type":"object","properties":{}}}}]"#;
    let (rx, fut) = transport
        .chat(
            w2,
            TINY_MODEL_ID.to_string(),
            "[{\"role\":\"user\",\"content\":\"Once upon a time\"}]".to_string(),
            8,
            0.0,
            Some(tools.to_string()),
            Some("{}".to_string()), // chat_template_kwargs (optional transport field)
        )
        .await
        .expect("remote chat opens");
    let (chunks, res) = drain_chat(rx, fut).await;
    let value = res.expect("remote chat (tools + kwargs) completed ok");
    let content = value.get("content").and_then(|c| c.as_str()).unwrap_or("");
    assert!(
        chunks > 0 || !content.is_empty(),
        "remote chat produced tokens (chunks={chunks}, content={content:?})"
    );
    transport
        .request(M_NODE_UNLOAD, json!({ "worker_id": w2 }))
        .await
        .expect("unload of the chatted worker ok");

    // ── data-plane error arm: chat to an unknown worker ──
    let (rx2, fut2) = transport
        .chat(
            9999,
            TINY_MODEL_ID.to_string(),
            "[{\"role\":\"user\",\"content\":\"hi\"}]".to_string(),
            4,
            0.0,
            None,
            None,
        )
        .await
        .expect("bad-worker chat opens the stream");
    let (_n, res2) = drain_chat(rx2, fut2).await;
    let err = res2.expect_err("chat to an unknown worker errors");
    assert!(
        matches!(err, HiggsError::WorkerRpc { .. }),
        "unknown-worker chat surfaces the relay's error, got {err:?}"
    );
}

// ── HubFleet admission control (in-process connections; no node daemon) ────────────────────

/// A `HubFleet::add_node` carrying a STALE admission generation is refused: nothing is
/// assigned, inserted, or seeded, and the refused transport is closed.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fleet_stale_admit_gen_is_refused() {
    let (conn, _keep) = admitted_pair().await;
    let peer = conn.remote_id().to_string();
    let fleet = Arc::new(HubFleet::new(Arc::new(LogBus::new())));

    // Advance the current admission generation to 1; then admit with the now-STALE gen 0.
    let current = fleet.bump_admit_gen().await;
    assert_eq!(current, 1, "first bump yields generation 1");
    fleet
        .add_node(
            peer.clone(),
            Arc::new(NodeTransport::new(conn)),
            Some(0),
            None,
            None,
            false,
        )
        .await;

    assert!(
        fleet.node_id(&peer).await.is_none(),
        "a stale-generation admission assigns no NodeId"
    );
    assert!(
        fleet.node_ids().await.is_empty(),
        "a stale-generation admission adds no connected node"
    );
    assert!(
        fleet.nodes_view().await.is_empty(),
        "a stale-generation admission seeds nothing into the fleet view"
    );
}

/// A live-transport node is admitted; `disconnect_all` closes its transport (kill switch)
/// leaving it listed-but-disconnected with its route/seed kept; `retire` then removes it
/// from the fleet view entirely.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fleet_admit_disconnect_all_then_retire() {
    let (conn, _keep) = admitted_pair().await;
    let peer = conn.remote_id().to_string();
    let fleet = Arc::new(HubFleet::new(Arc::new(LogBus::new())));

    fleet
        .add_node(
            peer.clone(),
            Arc::new(NodeTransport::new(conn)),
            None,
            None,
            None,
            false,
        )
        .await;
    assert_eq!(
        fleet.node_ids().await,
        vec![peer.clone()],
        "the admitted node is connected"
    );
    assert!(
        fleet
            .nodes_view()
            .await
            .iter()
            .any(|n| n.endpoint_id == peer && n.connected),
        "the admitted node shows connected in the fleet view"
    );

    fleet.disconnect_all().await;
    assert!(
        fleet.node_ids().await.is_empty(),
        "disconnect_all drops every node's transport"
    );
    assert!(
        fleet
            .nodes_view()
            .await
            .iter()
            .any(|n| n.endpoint_id == peer && !n.connected),
        "after disconnect_all the node is still listed but disconnected (route kept)"
    );

    fleet.retire(&peer).await;
    assert!(
        fleet.nodes_view().await.is_empty(),
        "retire removes the node from the fleet view entirely"
    );
}

/// A fresh fleet: `bus()` returns its log bus, `seed_node` lists a known node as
/// disconnected, and every op against an unknown/disconnected node or served id returns the
/// accurate typed error (HG027 node-unreachable / HG002 model-not-found).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fleet_seed_and_unknown_node_arms() {
    let bus = Arc::new(LogBus::new());
    let fleet = Arc::new(HubFleet::new(bus.clone()));
    assert!(
        Arc::ptr_eq(fleet.bus(), &bus),
        "bus() returns the fleet's own log bus"
    );

    let known = "1".repeat(64);
    fleet.seed_node(&known).await;
    assert!(
        fleet.node_id(&known).await.is_some(),
        "a seeded node is assigned a stable NodeId"
    );
    assert!(
        fleet
            .nodes_view()
            .await
            .iter()
            .any(|n| n.endpoint_id == known && !n.connected),
        "a seeded node is listed as disconnected before it reconnects"
    );
    assert!(
        fleet.node_ids().await.is_empty(),
        "seeding does not mark a node connected"
    );

    // Ops against a seeded-but-disconnected node → HG027 NodeUnreachable.
    assert!(
        matches!(
            fleet.load(&known, "some/model", None).await,
            Err(HiggsError::NodeUnreachable { .. })
        ),
        "load on a disconnected node → HG027"
    );
    assert!(
        matches!(
            fleet.scan_node(&known).await,
            Err(HiggsError::NodeUnreachable { .. })
        ),
        "scan of a disconnected node → HG027"
    );
    assert!(
        matches!(
            fleet.refresh_inventory(&known).await,
            Err(HiggsError::NodeUnreachable { .. })
        ),
        "inventory of a disconnected node → HG027"
    );

    // Ops against an unknown served id → HG002 ModelNotFound.
    assert!(
        matches!(
            fleet.unload("no/served").await,
            Err(HiggsError::ModelNotFound { .. })
        ),
        "unload of an unknown served id → HG002"
    );
    assert!(
        matches!(
            fleet.kill("no/served").await,
            Err(HiggsError::ModelNotFound { .. })
        ),
        "kill of an unknown served id → HG002"
    );
    assert!(
        !fleet.is_remote("no/served").await,
        "unknown id is not remote"
    );
    assert!(
        fleet.routed_models().await.is_empty(),
        "no served ids on an empty fleet"
    );
    assert!(
        fleet.resolve("no/served").await.is_none(),
        "an unknown served id resolves to nothing"
    );
}

// ── node-side HELLO validation (misbehaving hub, in-process) ───────────────────────────────

/// A hub that accepts the node's HELLO then closes the stream WITHOUT replying: the node's
/// `connect_node` read sees EOF and returns `UnexpectedEof`, never a hang.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn node_dial_hub_that_closes_before_reply_is_eof() {
    let hub = minimal_ep().await;
    let node = minimal_ep().await;
    let hub_addr = hub.addr();

    let hub_task = tokio::spawn(async move {
        let incoming = hub.accept().await.expect("incoming");
        let conn = incoming.await.expect("connection");
        misbehaving_hub_reply(&conn, None).await; // finish without replying
    });

    let node_id = node.id().to_string();
    let err = dial_and_hello(&node, hub_addr, node_id, String::new(), None)
        .await
        .expect_err("a hub that closes before replying must error");
    assert_eq!(
        err.kind(),
        std::io::ErrorKind::UnexpectedEof,
        "closing before the HELLO reply surfaces UnexpectedEof: {err}"
    );
    let _ = hub_task.await;
}

/// A hub whose HELLO reply carries the wrong role (`node`, not `hub`) is rejected by the
/// node's `validate_hub_hello` identity check (HG038), so a spoofed hub can't poison the
/// node's saved-hub id.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn node_rejects_hub_hello_with_wrong_role() {
    let hub = minimal_ep().await;
    let node = minimal_ep().await;
    let hub_addr = hub.addr();
    let hub_id = hub.id().to_string();

    let hub_task = tokio::spawn(async move {
        let incoming = hub.accept().await.expect("incoming");
        let conn = incoming.await.expect("connection");
        // role != "hub" (node_id DOES match the TLS peer, so ONLY the role check can reject).
        let bad = HelloResult {
            role: "node".into(),
            node_id: hub_id,
            hub_name: "imposter".into(),
            agreed_version: 1,
            software_version: "0.0.0".into(),
            assigned_label: None,
            capabilities: Default::default(),
        };
        let resp = RpcResponse {
            jsonrpc: "2.0".into(),
            id: 1,
            result: Some(serde_json::to_value(bad).unwrap()),
            error: None,
        };
        misbehaving_hub_reply(&conn, Some(RpcFrame::Response(resp))).await;
    });

    let node_id = node.id().to_string();
    let err = dial_and_hello(&node, hub_addr, node_id, String::new(), None)
        .await
        .expect_err("node must reject a hub HELLO with role != hub");
    assert!(
        err.to_string().contains("[HG038]"),
        "wrong-role hub HELLO → HG038 protocol violation, got: {err}"
    );
    let _ = hub_task.await;
}

/// A hub that replies with an unexpected frame kind (a notification, not a response to
/// HELLO) is rejected by `connect_node`'s protocol-violation arm (HG038).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn node_rejects_unexpected_hub_reply_frame() {
    let hub = minimal_ep().await;
    let node = minimal_ep().await;
    let hub_addr = hub.addr();

    let hub_task = tokio::spawn(async move {
        let incoming = hub.accept().await.expect("incoming");
        let conn = incoming.await.expect("connection");
        let note = RpcNotification {
            jsonrpc: "2.0".into(),
            method: "higgs/node/surprise".into(),
            params: json!({}),
        };
        misbehaving_hub_reply(&conn, Some(RpcFrame::Notification(note))).await;
    });

    let node_id = node.id().to_string();
    let err = dial_and_hello(&node, hub_addr, node_id, String::new(), None)
        .await
        .expect_err("node must reject an unexpected reply frame");
    assert!(
        err.to_string().contains("[HG038]"),
        "an unexpected reply frame → HG038 protocol violation, got: {err}"
    );
    let _ = hub_task.await;
}

// ── pure units reachable from the crate API ────────────────────────────────────────────────

/// The per-node worker registry: monotonic ids that are NEVER reused, plus
/// `insert`/`reserve`/`insert_reserved`/`get`/`remove`/`len`/`is_empty`/`ids`.
#[test]
fn worker_registry_semantics() {
    let mut reg: WorkerRegistry<String> = WorkerRegistry::default();
    assert!(reg.is_empty(), "a fresh registry is empty");
    assert_eq!(reg.len(), 0);

    let a = reg.insert("alpha".into());
    let b = reg.insert("beta".into());
    assert_eq!(a, WorkerId(1), "ids start at 1");
    assert_eq!(b, WorkerId(2), "ids are monotonic");
    assert_eq!(reg.len(), 2);
    assert!(!reg.is_empty());
    assert_eq!(reg.get(a).map(String::as_str), Some("alpha"));
    assert_eq!(
        reg.ids(),
        vec![WorkerId(1), WorkerId(2)],
        "ids listed ascending"
    );

    // reserve() hands out an id WITHOUT storing a value; insert_reserved commits it later.
    let r = reg.reserve();
    assert_eq!(r, WorkerId(3), "reserve advances the counter");
    assert_eq!(reg.get(r), None, "a reserved id holds no value yet");
    reg.insert_reserved(r, "gamma".into());
    assert_eq!(reg.get(r).map(String::as_str), Some("gamma"));

    // remove frees the value; the id is NEVER reused.
    assert_eq!(reg.remove(a), Some("alpha".to_string()));
    assert_eq!(reg.get(a), None);
    let d = reg.insert("delta".into());
    assert_eq!(d, WorkerId(4), "a removed id is never reused");
    assert_eq!(
        reg.remove(WorkerId(999)),
        None,
        "removing an absent id is None"
    );

    assert_eq!(WorkerId(7).to_string(), "w-7", "WorkerId renders as w-<n>");
}

/// The node identity key: `load_or_create_secret` creates+persists a stable 32-byte secret,
/// reloads it identically, and rejects a corrupt (non-32-byte) key file with `InvalidData`
/// rather than silently minting a new id.
#[test]
fn identity_secret_create_reload_and_corrupt() {
    let dir = tempfile::tempdir().expect("temp dir");
    let key_path = dir.path().join("endpoint.key");

    // First call CREATES + persists the key (the NotFound → create path).
    let sk1 = load_or_create_secret(&key_path).expect("create secret");
    assert!(key_path.is_file(), "the key file is persisted");
    assert_eq!(
        std::fs::read(&key_path).expect("read key").len(),
        32,
        "an ed25519 secret is 32 bytes"
    );

    // Second call RELOADS the same key — a stable identity across restarts.
    let sk2 = load_or_create_secret(&key_path).expect("reload secret");
    assert_eq!(
        sk1.to_bytes(),
        sk2.to_bytes(),
        "reloading yields the same secret (stable EndpointId)"
    );

    // A corrupt (wrong-length) key file fails loudly with InvalidData.
    let bad_path = dir.path().join("bad.key");
    std::fs::write(&bad_path, b"not-thirty-two-bytes").expect("write bad key");
    let err = load_or_create_secret(&bad_path).expect_err("corrupt key rejected");
    assert_eq!(
        err.kind(),
        std::io::ErrorKind::InvalidData,
        "a corrupt key file is InvalidData: {err}"
    );
    assert!(
        err.to_string().contains("32 bytes"),
        "the error names the 32-byte expectation: {err}"
    );
}
