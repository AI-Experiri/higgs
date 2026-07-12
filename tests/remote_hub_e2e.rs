//! Hub-seam end-to-end: prove the hub's `/v1` chat path (`Higgs::chat_stream`) routes a
//! remote-resident model through the `HubFleet` to a REAL spawned `higgs --node` process.
//!
//! Flow: build a hub `Higgs` with a `HubFleet` installed → spawn a real node → pair over
//! hermetic iroh → `fleet.load` a real GGUF on the node → `higgs.chat_stream(model)` routes
//! remotely and streams real tokens back. This exercises the full hub seam (chat_stream →
//! fleet routing → NodeTransport → node relay → worker → model), nothing faked.
//!
//! Skips when no tiny GGUF is available.

mod common;

use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use iroh_tickets::endpoint::EndpointTicket;

use higgs::auth::{Allowlist, PairingTokens};
use higgs::log_bus::{LogBus, LogSource};
use higgs::node::fleet::HubFleet;
use higgs::node::transport::NodeTransport;
use higgs::node::{gate_connection, GateOutcome, HubIdentity, HELLO_DEADLINE};
use higgs::remote::ALPN;
use higgs::{Higgs, HiggsConfig};

use common::{stage_tiny_model, tiny_gguf_path, TINY_MODEL_ID};

/// Deterministic (greedy) sampling umbrella for `chat_stream` — `temperature: 0`
/// makes the worker pick `argmax`, so these end-to-end streams are reproducible.
fn greedy() -> higgs::worker::engine::SamplingParams {
    higgs::worker::engine::SamplingParams::llamacpp(
        higgs::worker::engine::llamacpp::params::LlamaCppSamplingParams {
            temperature: Some(0.0),
            ..Default::default()
        },
    )
}

struct NodeProc(Child);
impl Drop for NodeProc {
    fn drop(&mut self) {
        // SIGTERM: graceful node shutdown flushes its llvm-cov profile (a hard kill drops it).
        unsafe {
            libc::kill(self.0.id() as libc::pid_t, libc::SIGTERM);
        }
        let _ = self.0.wait();
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

async fn hub_endpoint() -> iroh::Endpoint {
    iroh::Endpoint::builder(iroh::endpoint::presets::Minimal)
        .alpns(vec![ALPN.to_vec()])
        .relay_mode(iroh::RelayMode::Disabled)
        .bind()
        .await
        .expect("bind hub endpoint")
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
        Some("test".into()),
        HELLO_DEADLINE,
    )
    .await;
    assert!(
        matches!(outcome, GateOutcome::Admitted { .. }),
        "admitted: {outcome:?}"
    );
    (conn, peer)
}

/// Drain a chat delta stream + final outcome, returning the chunk count and whether any
/// tokens were produced (chunks or non-empty content).
async fn drain_chat(
    deltas: higgs::delta_queue::DeltaReceiver,
    handle: tokio::task::JoinHandle<Result<higgs::api::ChatOutcome, higgs::diagnostic::HiggsError>>,
) -> bool {
    let collector = tokio::spawn(async move {
        let mut deltas = deltas;
        let mut n = 0usize;
        while deltas.recv().await.is_some() {
            n += 1;
        }
        n
    });
    let outcome = handle.await.expect("join").expect("chat outcome");
    let chunks = collector.await.unwrap();
    chunks > 0 || !outcome.content.is_empty()
}

/// P5 (streaming sink handoff) + multi-instance served ids, end-to-end over REAL iroh:
/// TWO `higgs --node` processes on one machine both load the SAME model, so the fleet
/// exposes TWO collision-free served ids (`id`, `id-1`); a streamed `/v1` chat to EACH
/// served id routes to its own node and streams real tokens; retiring one node leaves the
/// fleet (the survivor renumbers to the lone served id) while it keeps streaming.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn two_nodes_same_model_two_served_ids_stream_each_then_retire_one() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("skipping two_nodes_same_model: no tiny GGUF (set HIGGS_TEST_GGUF)");
        return;
    };
    let scan_root = stage_tiny_model(&gguf);
    let home_a = tempfile::tempdir().expect("node home a");
    let home_b = tempfile::tempdir().expect("node home b");

    // Hub: a local Higgs (no local models) with a HubFleet installed.
    let bus = Arc::new(LogBus::new());
    let higgs = Arc::new(Higgs::with_log_bus(HiggsConfig::default(), bus.clone()));
    let fleet = Arc::new(HubFleet::new(bus.clone()));
    higgs.set_fleet(fleet.clone());

    let hub = hub_endpoint().await;
    let hub_id = hub.id().to_string();
    let ticket = EndpointTicket::new(hub.addr()).to_string();
    let mut allow = Allowlist::load(&home_a.path().join("hub-pairings.json")).unwrap();
    let mut tokens = PairingTokens::new();
    let token_a = tokens.mint(now_ms(), 600_000);
    let token_b = tokens.mint(now_ms(), 600_000);

    // Spawn TWO real node processes (own homes, shared read-only model dir, hermetic iroh).
    let spawn_node = |home: &std::path::Path, token: &str| {
        Command::new(env!("CARGO_BIN_EXE_higgs"))
            .arg("--node")
            .arg(&ticket)
            .arg(token)
            .env("HIGGS_HOME", home)
            .env("HIGGS_MODEL_DIR", scan_root.path())
            .env("HIGGS_IROH_LOCAL", "1")
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("spawn higgs --node")
    };
    let _node_a = NodeProc(spawn_node(home_a.path(), &token_a));
    let _node_b = NodeProc(spawn_node(home_b.path(), &token_b));

    // Admit both (dial order is nondeterministic).
    let (conn1, peer1) = admit_node(&hub, &mut allow, &mut tokens, &hub_id).await;
    fleet
        .add_node(
            peer1.clone(),
            Arc::new(NodeTransport::new(conn1)),
            None,
            None,
            None,
        )
        .await;
    let (conn2, peer2) = admit_node(&hub, &mut allow, &mut tokens, &hub_id).await;
    fleet
        .add_node(
            peer2.clone(),
            Arc::new(NodeTransport::new(conn2)),
            None,
            None,
            None,
        )
        .await;
    assert_ne!(peer1, peer2, "two distinct nodes paired");

    // Load the SAME model on BOTH nodes → two instances → two collision-free served ids.
    fleet
        .load(&peer1, TINY_MODEL_ID, None)
        .await
        .expect("load node1");
    fleet
        .load(&peer2, TINY_MODEL_ID, None)
        .await
        .expect("load node2");

    let mut served = fleet.routed_models().await;
    served.sort();
    let suffixed = format!("{TINY_MODEL_ID}-1");
    assert_eq!(
        served,
        vec![TINY_MODEL_ID.to_string(), suffixed.clone()],
        "same model on two nodes → two collision-free served ids"
    );

    // Each served id resolves to a DISTINCT (node, worker) — i.e. targets the right node.
    let r0 = fleet
        .resolve(TINY_MODEL_ID)
        .await
        .expect("served[0] routed");
    let r1 = fleet.resolve(&suffixed).await.expect("served[1] routed");
    assert_ne!(r0, r1, "the two served ids map to two distinct instances");

    // The fleet view (what `/api/higgs/nodes` serves the UI) tags each resident worker
    // with its hub-assigned served id, so the UI shows exactly what clients must call.
    let view = fleet.nodes_view().await;
    let mut view_served: Vec<String> = view
        .iter()
        .filter_map(|n| n.inventory.as_ref())
        .flat_map(|inv| inv.workers.iter().map(|w| w.served_id.clone()))
        .collect();
    view_served.sort();
    assert_eq!(
        view_served,
        vec![TINY_MODEL_ID.to_string(), suffixed.clone()],
        "fleet view tags each resident worker with its served id"
    );

    // Stream a chat to EACH served id; both must stream real tokens (sink handoff end-to-end).
    let prompt = "[{\"role\":\"user\",\"content\":\"Once upon a time\"}]";
    for sid in [TINY_MODEL_ID.to_string(), suffixed.clone()] {
        let (deltas, handle) = higgs
            .chat_stream(sid.clone(), prompt.to_string(), 8, greedy(), None, None)
            .await
            .unwrap_or_else(|e| panic!("chat_stream {sid}: {e}"));
        assert!(
            drain_chat(deltas, handle).await,
            "served id {sid} streamed real tokens"
        );
    }

    // Retire ONE node → it leaves the fleet; the survivor's instance renumbers to the lone
    // (unsuffixed) served id and KEEPS streaming.
    fleet.retire(&peer1).await;
    assert_eq!(
        fleet.node_ids().await.len(),
        1,
        "one node remains after retiring the other"
    );
    assert_eq!(
        fleet.routed_models().await,
        vec![TINY_MODEL_ID.to_string()],
        "the survivor renumbers to the lone served id"
    );
    let (deltas, handle) = higgs
        .chat_stream(
            TINY_MODEL_ID.to_string(),
            prompt.to_string(),
            8,
            greedy(),
            None,
            None,
        )
        .await
        .expect("survivor chat");
    assert!(
        drain_chat(deltas, handle).await,
        "the survivor keeps streaming after the peer retired"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn hub_v1_chat_routes_to_remote_node() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("skipping remote_hub_e2e: no tiny GGUF (set HIGGS_TEST_GGUF)");
        return;
    };
    let scan_root = stage_tiny_model(&gguf);
    let node_home = tempfile::tempdir().expect("node home");

    // Hub: a local Higgs (no local models) with a HubFleet installed, SHARING one LogBus so
    // relayed remote worker stderr lands in the same console the hub serves.
    let bus = Arc::new(LogBus::new());
    let higgs = Arc::new(Higgs::with_log_bus(HiggsConfig::default(), bus.clone()));
    let fleet = Arc::new(HubFleet::new(bus.clone()));
    higgs.set_fleet(fleet.clone());

    // Hub iroh endpoint + a pairing ticket/token.
    let hub = hub_endpoint().await;
    let hub_id = hub.id().to_string();
    let ticket = EndpointTicket::new(hub.addr()).to_string();
    let mut allow = Allowlist::load(&node_home.path().join("hub-pairings.json")).unwrap();
    let mut tokens = PairingTokens::new();
    let token = tokens.mint(now_ms(), 600_000);

    // Spawn the real node process (hermetic iroh, real model dir).
    let child = Command::new(env!("CARGO_BIN_EXE_higgs"))
        .arg("--node")
        .arg(&ticket)
        .arg(&token)
        .env("HIGGS_HOME", node_home.path())
        .env("HIGGS_MODEL_DIR", scan_root.path())
        .env("HIGGS_IROH_LOCAL", "1")
        .env("HIGGS_VERBOSE", "1") // keep worker stderr so the log relay carries lines
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("spawn higgs --node");
    let _node = NodeProc(child);

    // Accept the node's dial, gate it, and register it in the fleet.
    let incoming = tokio::time::timeout(Duration::from_secs(30), hub.accept())
        .await
        .expect("node dialed within 30s")
        .expect("incoming");
    let conn = incoming.await.expect("connection");
    let peer = conn.remote_id().to_string();
    let outcome = gate_connection(
        &conn,
        &mut allow,
        &mut tokens,
        now_ms(),
        &HubIdentity::new(hub_id),
        Some("test".into()),
        HELLO_DEADLINE,
    )
    .await;
    assert!(
        matches!(outcome, GateOutcome::Admitted { .. }),
        "node admitted: {outcome:?}"
    );
    fleet
        .add_node(
            peer.clone(),
            Arc::new(NodeTransport::new(conn)),
            None,
            None,
            None,
        )
        .await;

    // Load a real model on the node via the fleet → records the route.
    fleet
        .load(&peer, TINY_MODEL_ID, None)
        .await
        .expect("remote load");
    assert!(
        fleet.is_remote(TINY_MODEL_ID).await,
        "model is now remote-routable"
    );

    // The hub's /v1 chat path: chat_stream routes the remote-resident model through the
    // fleet to the node, streaming real tokens back.
    let (mut deltas, handle) = higgs
        .chat_stream(
            TINY_MODEL_ID.to_string(),
            "[{\"role\":\"user\",\"content\":\"Once upon a time\"}]".to_string(),
            8,
            greedy(),
            None,
            None,
        )
        .await
        .expect("chat_stream routed remotely");

    let collector = tokio::spawn(async move {
        let mut n = 0usize;
        while deltas.recv().await.is_some() {
            n += 1;
        }
        n
    });
    let outcome = handle.await.expect("join").expect("chat outcome");
    let chunk_count = collector.await.unwrap();

    assert!(
        chunk_count > 0 || !outcome.content.is_empty(),
        "remote chat generated tokens (chunks={chunk_count}, content={:?})",
        outcome.content
    );
    assert!(!outcome.finish_reason.is_empty(), "finish_reason set");

    // ── Remote log relay (P4): the node forwarded its real worker's stderr over iroh, and
    // the hub filed it under LogSource::RemoteWorker keyed by (node, worker). It appears in
    // the hub's own Developer-Logs console (the shared bus), filterable per remote worker.
    let node_id = fleet
        .node_id(&peer)
        .await
        .expect("node has an assigned NodeId");
    let worker = fleet.resolve(TINY_MODEL_ID).await.expect("model routed").1;
    let remote_src = LogSource::RemoteWorker {
        node: node_id,
        worker,
    };
    // Relay is async (uni stream + load dump); poll briefly for the first relayed line.
    let mut relayed = Vec::new();
    for _ in 0..50 {
        relayed = bus.snapshot(usize::MAX, Some(remote_src));
        if !relayed.is_empty() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert!(
        !relayed.is_empty(),
        "remote worker stderr was relayed to the hub bus (source=node:{}:{})",
        node_id.0,
        worker.0
    );

    // ── Inventory (P4): the hub fetches the node's self-description and exposes it in the
    // fleet view — host identity, real hardware, and the resident worker → model mapping.
    let inv = fleet
        .refresh_inventory(&peer)
        .await
        .expect("fetch inventory");
    assert!(
        inv.hardware.cpu_cores > 0,
        "inventory carries real cpu cores"
    );
    assert_eq!(inv.runtime.engine, "llama.cpp");
    assert!(
        inv.workers.iter().any(|w| w.model == TINY_MODEL_ID),
        "inventory lists the resident worker's model: {:?}",
        inv.workers
    );
    let views = fleet.nodes_view().await;
    assert!(
        views
            .iter()
            .any(|v| v.endpoint_id == peer && v.connected && v.inventory.is_some()),
        "fleet view shows the connected node with its inventory"
    );

    // ── Fleet lifecycle over the live link: the model is advertised, then unloaded, then
    // re-loaded + force-killed, then the node is retired — each transition reflected in the
    // hub's routing view, and the remote log ring reclaimed on teardown.
    assert_eq!(
        fleet.routed_models().await,
        vec![TINY_MODEL_ID.to_string()],
        "routed_models lists it"
    );
    fleet.unload(TINY_MODEL_ID).await.expect("remote unload");
    assert!(
        !fleet.is_remote(TINY_MODEL_ID).await,
        "unload drops the route"
    );
    assert!(
        bus.snapshot(usize::MAX, Some(remote_src)).is_empty(),
        "unload reclaims the log ring"
    );

    // Chat to the now-unrouted model errors (no route) rather than hanging.
    assert!(
        fleet
            .chat(TINY_MODEL_ID, "[]".into(), 8, 0.0, None, None)
            .await
            .is_err(),
        "chat to an unrouted model errors"
    );

    // Re-load, then force-kill the worker.
    fleet
        .load(&peer, TINY_MODEL_ID, None)
        .await
        .expect("remote re-load");
    assert!(
        fleet.is_remote(TINY_MODEL_ID).await,
        "re-loaded model is routable again"
    );
    fleet.kill(TINY_MODEL_ID).await.expect("remote kill");
    assert!(
        !fleet.is_remote(TINY_MODEL_ID).await,
        "kill drops the route"
    );

    // Retire the node: its routes + transport are gone, and ops now report unreachable.
    fleet
        .load(&peer, TINY_MODEL_ID, None)
        .await
        .expect("load before retire");
    fleet.retire(&peer).await;
    assert!(
        fleet.node_ids().await.is_empty(),
        "retired node removed from the fleet"
    );
    assert!(
        fleet.routed_models().await.is_empty(),
        "retire clears routes"
    );
}

/// A node survives a connection blip: when the hub link drops, the `--node` daemon redials
/// (its reconnect loop), and because the node reuses ONE `NodeRuntime` across reconnects,
/// its resident worker — and the hub's durable route to it — persist. The hub re-admits the
/// reconnecting (now allowlisted) node and chats again WITHOUT reloading.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn node_reconnects_and_route_survives() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("skipping node_reconnects_and_route_survives: no tiny GGUF");
        return;
    };
    let scan_root = stage_tiny_model(&gguf);
    let node_home = tempfile::tempdir().expect("node home");

    let bus = Arc::new(LogBus::new());
    let higgs = Arc::new(Higgs::with_log_bus(HiggsConfig::default(), bus.clone()));
    let fleet = Arc::new(HubFleet::new(bus.clone()));
    higgs.set_fleet(fleet.clone());

    let hub = hub_endpoint().await;
    let hub_id = hub.id().to_string();
    let ticket = EndpointTicket::new(hub.addr()).to_string();
    let mut allow = Allowlist::load(&node_home.path().join("hub-pairings.json")).unwrap();
    let mut tokens = PairingTokens::new();
    let token = tokens.mint(now_ms(), 600_000);

    let child = Command::new(env!("CARGO_BIN_EXE_higgs"))
        .arg("--node")
        .arg(&ticket)
        .arg(&token)
        .env("HIGGS_HOME", node_home.path())
        .env("HIGGS_MODEL_DIR", scan_root.path())
        .env("HIGGS_IROH_LOCAL", "1")
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("spawn higgs --node");
    let _node = NodeProc(child);

    // Helper: accept + gate the next inbound node connection, returning its peer id.
    async fn admit(
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
            Some("test".into()),
            HELLO_DEADLINE,
        )
        .await;
        assert!(
            matches!(outcome, GateOutcome::Admitted { .. }),
            "admitted: {outcome:?}"
        );
        (conn, peer)
    }

    // First session: admit, load, route established.
    let (conn1, peer) = admit(&hub, &mut allow, &mut tokens, &hub_id).await;
    fleet
        .add_node(
            peer.clone(),
            Arc::new(NodeTransport::new(conn1.clone())),
            None,
            None,
            None,
        )
        .await;
    fleet
        .load(&peer, TINY_MODEL_ID, None)
        .await
        .expect("remote load");
    let worker_before = fleet.resolve(TINY_MODEL_ID).await.expect("routed").1;

    // Blip: drop the hub side. The node's serve loop returns and it redials after backoff.
    conn1.close(0u32.into(), b"blip");

    // Second session: the node redials (allowlisted now — no token), we re-admit and install
    // the fresh transport. The durable route to the still-resident worker is unchanged.
    let (conn2, peer2) = admit(&hub, &mut allow, &mut tokens, &hub_id).await;
    assert_eq!(peer2, peer, "same node reconnects");
    fleet
        .add_node(
            peer2.clone(),
            Arc::new(NodeTransport::new(conn2)),
            None,
            None,
            None,
        )
        .await;
    assert_eq!(
        fleet.resolve(TINY_MODEL_ID).await.map(|r| r.1),
        Some(worker_before),
        "route + worker id survive the reconnect (durable routes)"
    );

    // Chat works again over the fresh connection without reloading the model.
    // Pass a tools array too, so the tools_json relay branch (hub transport → node relay →
    // worker) is exercised end-to-end over the reconnected link.
    let tools = r#"[{"type":"function","function":{"name":"get_time","description":"now","parameters":{"type":"object","properties":{}}}}]"#;
    let (mut deltas, handle) = higgs
        .chat_stream(
            TINY_MODEL_ID.to_string(),
            "[{\"role\":\"user\",\"content\":\"Hello again\"}]".to_string(),
            8,
            greedy(),
            Some(tools.to_string()),
            None,
        )
        .await
        .expect("chat after reconnect");
    let collector = tokio::spawn(async move {
        let mut n = 0usize;
        while deltas.recv().await.is_some() {
            n += 1;
        }
        n
    });
    let outcome = handle.await.expect("join").expect("chat outcome");
    let chunks = collector.await.unwrap();
    assert!(
        chunks > 0 || !outcome.content.is_empty(),
        "post-reconnect chat generated tokens"
    );
}
