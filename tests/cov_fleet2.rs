//! Round-2 integration coverage for the iroh fleet/hub wire — the ERROR/malformed-reply
//! arms that the existing remote suite (`remote_pairing`, `remote_cli`, `remote_hub_e2e`,
//! `remote_node_e2e`, `stream_remote_chat`, `chat_fleet`, `control_fleet_routes`,
//! `cov_remote`) leaves uncovered because those all drive a REAL, well-behaved node/hub.
//!
//! Every one of these paths only fires when a peer sends something the happy path never
//! sends. So each test here stands up an in-process iroh peer that hand-writes the exact
//! malformed / error / truncated reply and asserts the REAL outcome the code under test
//! produces (typed HiggsError variant / HG code / GateOutcome / route present-absent):
//!
//!  - `node/mod.rs` hub gate: the version-mismatch reject (HG023) reachable ONLY via a
//!    crafted HELLO whose `protocol_versions` don't intersect ours — `remote_pairing`
//!    documents this branch as un-producible through the real `dial_and_hello` (both
//!    endpoints speak `[1]`), so a hand-built HELLO is the only way in.
//!  - `node/transport.rs`: `request()`/`chat()` error mapping when the node returns an
//!    unexpected frame, closes the stream before replying, or streams undecodable / stray
//!    chunks with no final Response (the `transport_dead` / `[HG051]`-drop arms).
//!  - `node/fleet.rs`: `HubFleet` op error arms over a MOCK node — a junk inventory reply
//!    (ProtocolViolation), a `load` reply missing / with an out-of-range `worker_id`
//!    (`parse_worker_id`), a non-route-invalidating `unload` error (route kept), a
//!    route-invalidating `chat` error (route dropped), and a transport-dead `scan`
//!    (`handle_op_error` → drop transport → HG027).
//!  - `node/mod.rs` node side: `send_leave` error arms (EOF, an uncoded hub rejection →
//!    `hub_rejection`'s generic HG039 branch, an unexpected reply frame → HG038).
//!
//! All work is in-process (two hermetic iroh endpoints per test) — no spawned processes,
//! no GGUF — so the suite is fast and never skips.

use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use iroh::endpoint::Connection;
use iroh::Endpoint;
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, BufReader};

use higgs::auth::{Allowlist, PairingTokens};
use higgs::delta_queue::DeltaReceiver;
use higgs::diagnostic::HiggsError;
use higgs::log_bus::LogBus;
use higgs::node::fleet::HubFleet;
use higgs::node::transport::NodeTransport;
use higgs::node::{gate_connection, send_leave, GateOutcome, HubIdentity, HELLO_DEADLINE};
use higgs::remote::{
    node_capabilities, HelloParams, ALPN, M_HELLO, M_NODE_INVENTORY, M_NODE_LOAD, M_NODE_SCAN,
    M_NODE_UNLOAD,
};
use higgs::rpc::{self, RpcError, RpcFrame, RpcNotification, RpcRequest, RpcResponse};
use higgs::worker::{M_CHAT, N_CHAT_CHUNK};

// ── shared helpers (common/mod.rs is off-limits, so these are inlined) ─────────────────────

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// A hermetic (relay-disabled) iroh endpoint on the higgs remote ALPN — the same recipe the
/// remote suite uses for in-process links.
async fn minimal_ep() -> Endpoint {
    Endpoint::builder(iroh::endpoint::presets::Minimal)
        .alpns(vec![ALPN.to_vec()])
        .relay_mode(iroh::RelayMode::Disabled)
        .bind()
        .await
        .expect("bind endpoint")
}

/// An empty, uniquely-pathed, WRITABLE allowlist under the OS temp dir (so a gate that admits
/// can persist its pairing without a spurious HG040).
fn temp_allowlist(tag: &str) -> Allowlist {
    let n = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "higgs-cov-fleet2-{tag}-{}-{n:x}.json",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&path);
    Allowlist::load(&path).expect("empty allowlist loads")
}

/// Establish ONE raw ALPN connection between two hermetic endpoints (no HELLO — `NodeTransport`
/// and `send_leave` operate on a bare `Connection`). Returns `(dialer, acceptor, endpoints)`;
/// keep the endpoints in scope or the connections drop.
async fn connect_pair() -> (Connection, Connection, (Endpoint, Endpoint)) {
    let a = minimal_ep().await;
    let b = minimal_ep().await;
    let b_addr = b.addr();
    let accept = tokio::spawn(async move {
        let incoming = b.accept().await.expect("incoming");
        let conn = incoming.await.expect("accept conn");
        (conn, b)
    });
    let dialer = a.connect(b_addr, ALPN).await.expect("dial");
    let (acceptor, b_ep) = accept.await.expect("accept join");
    (dialer, acceptor, (a, b_ep))
}

fn resp_ok(id: u64, result: Value) -> RpcResponse {
    RpcResponse {
        jsonrpc: "2.0".into(),
        id,
        result: Some(result),
        error: None,
    }
}

/// A JSON-RPC error reply carrying `code` in `data.code` (so `extract_result` recovers the
/// worker origin code, exactly as a real node error does).
fn resp_err(id: u64, code: &str, message: &str) -> RpcResponse {
    RpcResponse {
        jsonrpc: "2.0".into(),
        id,
        result: None,
        error: Some(RpcError {
            code: -32000,
            message: message.into(),
            data: Some(json!({ "code": code })),
        }),
    }
}

fn notification(method: &str, params: Value) -> RpcNotification {
    RpcNotification {
        jsonrpc: "2.0".into(),
        method: method.into(),
        params,
    }
}

/// What a mock peer writes back on the accepted control/data stream.
enum Reply {
    /// One JSON-RPC Response frame, then finish.
    Response(RpcResponse),
    /// Several frames (e.g. chat chunks), then finish WITHOUT a final Response.
    Frames(Vec<RpcFrame>),
    /// One raw line (e.g. non-JSON garbage), then finish.
    RawLine(String),
    /// Finish the stream immediately WITHOUT replying (node closes before answering).
    Silent,
}

/// A mock node/hub serving loop: accept every bi stream, read the one request, and reply per
/// `handler(method, id, params)`. Drives the `NodeTransport`/`HubFleet` code under test with
/// exactly the wire behavior each error arm needs.
async fn serve_mock<F>(conn: Connection, handler: F)
where
    F: Fn(String, u64, Value) -> Reply + Send + Sync + 'static,
{
    let handler = Arc::new(handler);
    while let Ok((send, recv)) = conn.accept_bi().await {
        let handler = handler.clone();
        tokio::spawn(async move {
            let mut send = send;
            let mut lines = BufReader::new(recv).lines();
            let Ok(Some(line)) = lines.next_line().await else {
                return;
            };
            let Ok(RpcFrame::Request(req)) = rpc::decode(&line) else {
                return;
            };
            match handler(req.method.clone(), req.id, req.params.clone()) {
                Reply::Response(resp) => {
                    // `write_all`/`finish` are inherent iroh SendStream methods (no AsyncWriteExt).
                    let _ = send
                        .write_all(
                            format!("{}\n", rpc::encode(&RpcFrame::Response(resp))).as_bytes(),
                        )
                        .await;
                    let _ = send.finish();
                }
                Reply::Frames(frames) => {
                    for f in &frames {
                        let _ = send
                            .write_all(format!("{}\n", rpc::encode(f)).as_bytes())
                            .await;
                    }
                    let _ = send.finish();
                }
                Reply::RawLine(s) => {
                    let _ = send.write_all(format!("{s}\n").as_bytes()).await;
                    let _ = send.finish();
                }
                Reply::Silent => {
                    let _ = send.finish();
                }
            }
        });
    }
}

async fn count_deltas(mut rx: DeltaReceiver) -> usize {
    let mut n = 0usize;
    while rx.recv().await.is_some() {
        n += 1;
    }
    n
}

// ── 1. hub gate: crafted HELLO with an unsupported protocol version → HG023 ─────────────────

/// The hub-side version-negotiation reject (`gate_read_hello`'s HG023 branch) fires only when a
/// peer offers `protocol_versions` that don't intersect ours. The real `dial_and_hello` always
/// sends `[1]`, so `remote_pairing.rs` documents this branch as unreachable through it. Here a
/// hand-built HELLO carries a VALID identity (so only the version branch can reject) but offers
/// `[999]` — the gate must reply the typed HG023 close and return `Rejected { code: "HG023" }`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn hub_gate_rejects_crafted_version_mismatch() {
    let hub = minimal_ep().await;
    let node = minimal_ep().await;
    let hub_addr = hub.addr();
    let hub_id = hub.id().to_string();
    let node_id = node.id().to_string();

    let gate = tokio::spawn(async move {
        let mut allow = temp_allowlist("hg023");
        let mut tokens = PairingTokens::new();
        let incoming = tokio::time::timeout(Duration::from_secs(10), hub.accept())
            .await
            .expect("node dialed within 10s")
            .expect("incoming");
        let conn = incoming.await.expect("connection");
        gate_connection(
            &conn,
            &mut allow,
            &mut tokens,
            now_ms(),
            &HubIdentity::new(hub_id),
            Some("cov".into()),
            HELLO_DEADLINE,
        )
        .await
    });

    // Dial + hand-write a HELLO: node_id == our TLS id (identity passes), but an unsupported
    // version set (only the negotiate-version branch can reject).
    let conn = node.connect(hub_addr, ALPN).await.expect("connect");
    let (mut send, recv) = conn.open_bi().await.expect("open_bi");
    let params = HelloParams {
        role: "node".into(),
        node_id: node_id.clone(),
        name: String::new(),
        pairing_token: None,
        protocol_versions: vec![999],
        min_supported: 999,
        software_version: "9.9.9".into(),
        capabilities: node_capabilities(),
    };
    let req = RpcRequest {
        jsonrpc: "2.0".into(),
        id: 1,
        method: M_HELLO.into(),
        params: serde_json::to_value(params).expect("hello serializes"),
    };
    send.write_all(format!("{}\n", rpc::encode(&RpcFrame::Request(req))).as_bytes())
        .await
        .expect("write hello");
    let _ = send.finish();

    // Read the hub's typed reject so the node sees the HG023 code (not a bare EOF), then close
    // so the gate's grace-close resolves promptly.
    let mut lines = BufReader::new(recv).lines();
    let reply = tokio::time::timeout(Duration::from_secs(5), lines.next_line())
        .await
        .ok()
        .and_then(Result::ok)
        .flatten();
    assert!(
        reply.as_deref().is_some_and(|l| l.contains("HG023")),
        "the node receives a typed HG023 reject frame, not a bare EOF: {reply:?}"
    );
    conn.close(0u32.into(), b"done");

    let outcome = gate.await.expect("gate join");
    assert_eq!(
        outcome,
        GateOutcome::Rejected { code: "HG023" },
        "an unsupported protocol-version HELLO is rejected with HG023"
    );
}

// ── 2. NodeTransport: unexpected / truncated / undecodable replies ──────────────────────────

/// `NodeTransport::request` and `chat` must map every malformed node reply to a typed
/// `WorkerDead`/dropped-chunk outcome (never a hang or panic):
///  - an unexpected reply frame (a Notification where a Response is due) → `transport_dead`;
///  - the node closing the control stream before replying → `transport_dead`;
///  - a chat stream that sends an UNDECODABLE `N_CHAT_CHUNK` (no `delta`, dropped with an
///    `[HG051]` warn), then a STRAY non-chunk frame (ignored), then closes with NO final
///    Response → the chat future errors `transport_dead` and ZERO deltas reach the receiver.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn node_transport_surfaces_malformed_and_dead_replies() {
    let (dialer, acceptor, _eps) = connect_pair().await;
    let _mock = tokio::spawn(serve_mock(acceptor, |method, id, _params| {
        match method.as_str() {
            // A Notification where the caller awaits a Response → request()'s "unexpected reply
            // frame" arm.
            "higgs/node/probe-unexpected" => Reply::Frames(vec![RpcFrame::Notification(
                notification("higgs/surprise", json!({})),
            )]),
            // Read the request, then close WITHOUT replying → request()'s stream-closed arm.
            "higgs/node/probe-silent" => Reply::Silent,
            // Chat: an undecodable chunk (no `delta`), a stray frame, then EOF (no final).
            m if m == M_CHAT => Reply::Frames(vec![
                RpcFrame::Notification(notification(N_CHAT_CHUNK, json!({ "request_id": id }))),
                RpcFrame::Notification(notification("higgs/stray", json!({}))),
            ]),
            _ => Reply::Silent,
        }
    }));

    let transport = NodeTransport::new(dialer);

    let e = transport
        .request("higgs/node/probe-unexpected", json!({}))
        .await
        .expect_err("an unexpected reply frame surfaces an error");
    assert!(
        matches!(e, HiggsError::WorkerDead { .. }),
        "unexpected reply frame → WorkerDead, got {e:?}"
    );

    let e = transport
        .request("higgs/node/probe-silent", json!({}))
        .await
        .expect_err("a node that closes before replying surfaces an error");
    assert!(
        matches!(e, HiggsError::WorkerDead { .. }),
        "stream closed before reply → WorkerDead, got {e:?}"
    );

    let (rx, fut) = transport
        .chat(1, "m".into(), "[]".into(), 4, 0.0, None, None)
        .await
        .expect("chat opens the stream");
    let collector = tokio::spawn(count_deltas(rx));
    let res = fut.await;
    let deltas = collector.await.expect("collector join");
    let e = res.expect_err("a chat stream with no final Response errors");
    assert!(
        matches!(e, HiggsError::WorkerDead { .. }),
        "chat closed before final → WorkerDead, got {e:?}"
    );
    assert_eq!(
        deltas, 0,
        "the undecodable chunk is dropped, so no delta reaches the receiver"
    );
}

// ── 3. HubFleet op error arms over a mock node ──────────────────────────────────────────────

/// Drive `HubFleet`'s op error arms with a MOCK node whose replies are crafted per method — the
/// arms a real, well-behaved node never triggers:
///  - `refresh_inventory` on a junk inventory reply → `ProtocolViolation`;
///  - `load` whose reply is missing / has an out-of-range `worker_id` → `ProtocolViolation`
///    (`parse_worker_id` both arms);
///  - a real load records a route; a NON-route-invalidating `unload` error (HG005) is passed
///    through and the route is KEPT;
///  - a route-invalidating `chat` error (HG007) drops the instance (route gone after);
///  - a transport-dead `scan` (undecodable reply) → `handle_op_error` drops the transport and
///    remaps to HG027 `NodeUnreachable`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn hub_fleet_error_arms_over_mock_node() {
    let (dialer, acceptor, _eps) = connect_pair().await;
    let peer = dialer.remote_id().to_string();

    let _mock = tokio::spawn(serve_mock(acceptor, |method, id, params| {
        match method.as_str() {
            // Always a decodable Response, but NOT a valid NodeInventory → ProtocolViolation.
            m if m == M_NODE_INVENTORY => {
                Reply::Response(resp_ok(id, json!({ "not": "an inventory" })))
            }
            m if m == M_NODE_LOAD => {
                let model = params.get("id").and_then(Value::as_str).unwrap_or("");
                let result = match model {
                    "missing/wid" => json!({}),                            // no worker_id
                    "oor/wid" => json!({ "worker_id": 9_999_999_999u64 }), // > u32::MAX
                    _ => json!({ "worker_id": 1 }),                        // a normal load
                };
                Reply::Response(resp_ok(id, result))
            }
            // A client-error code that is NOT route-invalidating (route must survive).
            m if m == M_NODE_UNLOAD => Reply::Response(resp_err(id, "HG005", "client error")),
            // A worker-gone code that IS route-invalidating (route must be dropped).
            m if m == M_CHAT => Reply::Response(resp_err(id, "HG007", "worker gone")),
            // Non-JSON → the transport decode fails → WorkerDead → HG027 remap.
            m if m == M_NODE_SCAN => Reply::RawLine("this is not valid json".into()),
            _ => Reply::Response(resp_err(id, "HG037", "unknown method")),
        }
    }));

    let fleet = Arc::new(HubFleet::new(Arc::new(LogBus::new())));
    fleet
        .add_node(
            peer.clone(),
            Arc::new(NodeTransport::new(dialer)),
            None,
            None,
        )
        .await;

    // Junk inventory → ProtocolViolation (the reply decoded but isn't a NodeInventory).
    let e = fleet
        .refresh_inventory(&peer)
        .await
        .expect_err("a junk inventory reply is a protocol violation");
    assert!(
        matches!(e, HiggsError::ProtocolViolation { .. }),
        "junk inventory → ProtocolViolation, got {e:?}"
    );

    // load reply missing worker_id → ProtocolViolation (parse_worker_id "missing" arm).
    let e = fleet
        .load(&peer, "missing/wid", None)
        .await
        .expect_err("a load reply missing worker_id is a protocol violation");
    assert!(
        matches!(e, HiggsError::ProtocolViolation { .. }),
        "missing worker_id → ProtocolViolation, got {e:?}"
    );

    // load reply worker_id out of u32 range → ProtocolViolation (the "out of range" arm).
    let e = fleet
        .load(&peer, "oor/wid", None)
        .await
        .expect_err("an out-of-range worker_id is a protocol violation");
    assert!(
        matches!(e, HiggsError::ProtocolViolation { .. }),
        "out-of-range worker_id → ProtocolViolation, got {e:?}"
    );

    // A real load records a route (served id == the raw model, one instance).
    let worker = fleet.load(&peer, "ok/model", None).await.expect("load ok");
    assert_eq!(worker.0, 1, "the node-assigned worker id is threaded back");
    assert!(
        fleet.is_remote("ok/model").await,
        "the load records a route"
    );
    assert!(
        fleet
            .routed_models()
            .await
            .contains(&"ok/model".to_string()),
        "the served id is advertised while the node is connected"
    );

    // A NON-route-invalidating unload error is passed through, and the route is KEPT.
    let e = fleet
        .unload("ok/model")
        .await
        .expect_err("the node's unload error surfaces");
    assert!(
        matches!(&e, HiggsError::WorkerRpc { worker_code: Some(c), .. } if c == "HG005"),
        "a non-invalidating unload error is passed through verbatim, got {e:?}"
    );
    assert!(
        fleet.is_remote("ok/model").await,
        "a non-route-invalidating unload error keeps the route"
    );

    // A route-invalidating chat error drops the instance (route gone afterward).
    let (rx, fut) = fleet
        .chat("ok/model", "[]".into(), 4, 0.0, None, None)
        .await
        .expect("chat resolves + opens the relay");
    let collector = tokio::spawn(count_deltas(rx));
    let res = fut.await;
    let _ = collector.await;
    let e = res.expect_err("the route-invalidating chat error surfaces");
    assert!(
        matches!(&e, HiggsError::WorkerRpc { worker_code: Some(c), .. } if c == "HG007"),
        "a route-invalidating chat error is returned, got {e:?}"
    );
    assert!(
        !fleet.is_remote("ok/model").await,
        "a route-invalidating (HG007) chat error drops the instance route"
    );

    // A transport-dead scan (undecodable reply) → drop the transport → HG027 NodeUnreachable.
    let e = fleet
        .scan_node(&peer)
        .await
        .expect_err("an undecodable scan reply is a dead transport");
    assert!(
        matches!(e, HiggsError::NodeUnreachable { .. }),
        "transport-dead scan → HG027 NodeUnreachable, got {e:?}"
    );
}

// ── 4. node side: send_leave error arms ─────────────────────────────────────────────────────

/// How a mock hub answers a node's `M_NODE_LEAVE`.
enum LeaveReply {
    /// Read the leave, then close the stream with NO reply → the node reads EOF.
    Eof,
    /// Reply an error whose message carries NO `[HG` code → `hub_rejection`'s generic HG039.
    Uncoded,
    /// Reply an unexpected frame (a Notification) → the node's protocol-violation arm (HG038).
    Unexpected,
}

/// Run `send_leave` against a mock hub that answers with `reply`, returning the node-side result.
async fn leave_result(reply: LeaveReply) -> std::io::Result<()> {
    let (node_conn, hub_conn, eps) = connect_pair().await;
    let _keep = eps;
    let mock = tokio::spawn(async move {
        let (mut send, recv) = hub_conn.accept_bi().await.expect("accept leave stream");
        let mut lines = BufReader::new(recv).lines();
        let _ = lines.next_line().await; // drain the M_NODE_LEAVE request
        match reply {
            LeaveReply::Eof => {
                let _ = send.finish(); // no reply
            }
            LeaveReply::Uncoded => {
                let resp = RpcResponse {
                    jsonrpc: "2.0".into(),
                    id: 1,
                    result: None,
                    error: Some(RpcError {
                        code: -32000,
                        message: "plain rejection with no HG code".into(),
                        data: None,
                    }),
                };
                let _ = send
                    .write_all(format!("{}\n", rpc::encode(&RpcFrame::Response(resp))).as_bytes())
                    .await;
                let _ = send.finish();
            }
            LeaveReply::Unexpected => {
                let note = notification("higgs/node/surprise", json!({}));
                let _ = send
                    .write_all(
                        format!("{}\n", rpc::encode(&RpcFrame::Notification(note))).as_bytes(),
                    )
                    .await;
                let _ = send.finish();
            }
        }
        // Hold the connection open briefly so the node reads its reply / EOF before we drop it.
        tokio::time::sleep(Duration::from_millis(150)).await;
        hub_conn
    });
    let res = send_leave(&node_conn).await;
    let _ = mock.await;
    res
}

/// `send_leave` maps each way a hub can misbehave to a distinct, typed node-side error (never a
/// hang): a hub that closes before acking → `UnexpectedEof`; an uncoded rejection → the generic
/// `hub_rejection` HG039 (the node always shows SOME code); an unexpected reply frame → HG038.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn node_send_leave_error_arms() {
    let eof = leave_result(LeaveReply::Eof)
        .await
        .expect_err("a hub that closes before acking is an error");
    assert_eq!(
        eof.kind(),
        std::io::ErrorKind::UnexpectedEof,
        "closing before the leave ack surfaces UnexpectedEof: {eof}"
    );

    let uncoded = leave_result(LeaveReply::Uncoded)
        .await
        .expect_err("an uncoded hub rejection is an error");
    assert!(
        uncoded.to_string().contains("HG039"),
        "an uncoded rejection becomes the generic HG039 HubRequestRejected, got: {uncoded}"
    );

    let unexpected = leave_result(LeaveReply::Unexpected)
        .await
        .expect_err("an unexpected reply frame is an error");
    assert!(
        unexpected.to_string().contains("HG038"),
        "an unexpected reply frame → HG038 protocol violation, got: {unexpected}"
    );
}

// ── 4. T8: the M_NODE_LOAD payload carries params at protocol 2, and only then ─────────────

/// The WIRE pin for per-load params: against a MAJOR-2 admission a params-load
/// puts `ctx_len` (and friends) on the `M_NODE_LOAD` payload, while a bare load
/// still sends EXACTLY `{ "id" }` (no stray keys an old node could trip on).
/// Fail-on-revert: revert `HubFleet::load`'s params branch to the bare payload
/// and the ctx assert in the mock fails.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn params_load_payload_carries_ctx_at_protocol_two() {
    let (dialer, acceptor, _eps) = connect_pair().await;
    let peer = dialer.remote_id().to_string();

    let _mock = tokio::spawn(serve_mock(acceptor, |method, id, params| {
        match method.as_str() {
            m if m == M_NODE_LOAD => {
                let model = params.get("id").and_then(Value::as_str).unwrap_or("");
                match model {
                    "with/params" => {
                        assert_eq!(
                            params.get("ctx_len").and_then(Value::as_u64),
                            Some(2048),
                            "params-load payload must carry ctx_len: {params}"
                        );
                        assert!(
                            params.get("gpu_layers").is_some(),
                            "params-load payload must carry gpu_layers: {params}"
                        );
                    }
                    "only/ctx" => {
                        // The r1 HIGH regression pin: a PARTIAL params-load
                        // must NOT launder hub struct-defaults onto the wire —
                        // absent fields stay absent so the NODE's defaults
                        // (all-GPU, its own threads) genuinely apply.
                        assert_eq!(
                            params.get("ctx_len").and_then(Value::as_u64),
                            Some(256),
                            "partial params carry the one set field: {params}"
                        );
                        assert!(
                            params.get("gpu_layers").is_none() && params.get("threads").is_none(),
                            "unset fields stay ABSENT (no CPU-only/0-thread laundering): {params}"
                        );
                    }
                    "forced/id" => {
                        // The FLEET-level id-force in isolation (the facade has
                        // its own redundant force; this mock is reached via
                        // fleet.load DIRECTLY, so only the fleet force can have
                        // rewritten the divergent p.id below).
                        assert_eq!(
                            params.get("id").and_then(Value::as_str),
                            Some("forced/id"),
                            "the fleet rewrites a divergent p.id to model: {params}"
                        );
                    }
                    "bare/load" => {
                        let keys: Vec<_> = params
                            .as_object()
                            .map(|o| o.keys().cloned().collect())
                            .unwrap_or_default();
                        assert_eq!(keys, vec!["id"], "bare load sends ONLY the id: {params}");
                    }
                    other => panic!("unexpected model in mock: {other}"),
                }
                Reply::Response(resp_ok(id, json!({ "worker_id": 1 })))
            }
            // Post-load inventory refresh — an empty-but-valid inventory suffices.
            m if m == M_NODE_INVENTORY => Reply::Response(resp_ok(
                id,
                json!({ "hostname": "mock", "os": "mock", "workers": [],
                        "hardware": { "cpu_name": "m", "cpu_cores": 1, "ram_total_bytes": 1,
                                       "vram_total_bytes": 0, "gpus": [] },
                        "runtime": { "engine": "mock", "version": "0", "backend": "cpu" } }),
            )),
            _ => Reply::Response(resp_err(id, "HG037", "unknown method")),
        }
    }));

    let fleet = Arc::new(HubFleet::new(Arc::new(LogBus::new())));
    fleet
        .add_node(
            peer.clone(),
            Arc::new(NodeTransport::new(dialer)),
            None,
            Some(2),
        )
        .await;

    // Params-load: the mock's asserts fire inside the handler; a panic there
    // fails the RPC and thus this expect.
    let params = higgs::remote::NodeLoadParams {
        id: "with/params".into(),
        ctx_len: Some(2048),
        gpu_layers: Some(higgs::worker::engine::GpuLayers::Count { n: 7 }),
        threads: None,
        params: None,
    };
    fleet
        .load(&peer, "with/params", Some(params))
        .await
        .expect("params-load dispatches at protocol 2");

    // Bare load: unchanged classic payload.
    fleet
        .load(&peer, "bare/load", None)
        .await
        .expect("bare load still works");

    // Divergent p.id via the PUB fleet API: the fleet's own force wins.
    let divergent = higgs::remote::NodeLoadParams {
        id: "not/the-model".into(),
        ctx_len: Some(64),
        gpu_layers: None,
        threads: None,
        params: None,
    };
    fleet
        .load(&peer, "forced/id", Some(divergent))
        .await
        .expect("divergent-id params-load dispatches with the forced id");

    // ALL-None params are a BARE load: byte-identical payload, never
    // version-gated — this node is admitted at major 2, but the same call
    // against a floor-1 admission must also pass (pinned in fleet_tests).
    let empty = higgs::remote::NodeLoadParams {
        id: String::new(),
        ctx_len: None,
        gpu_layers: None,
        threads: None,
        params: None,
    };
    fleet
        .load(&peer, "bare/load", Some(empty))
        .await
        .expect("all-None params behave as a bare load");

    // Zero normalization: ctx 0 / threads 0 read as ABSENT hub-side, so this
    // params-load degrades to a BARE payload (the "bare/load" mock arm asserts
    // keys == ["id"]) — an older major-2 node can never see a zero to coerce.
    let zeros = higgs::remote::NodeLoadParams {
        id: String::new(),
        ctx_len: Some(0),
        gpu_layers: None,
        threads: Some(0),
        params: None,
    };
    fleet
        .load(&peer, "bare/load", Some(zeros))
        .await
        .expect("zero params normalize to a bare load");

    // Partial params: only ctx set — gpu_layers/threads stay off the wire.
    let partial = higgs::remote::NodeLoadParams {
        id: "only/ctx".into(),
        ctx_len: Some(256),
        gpu_layers: None,
        threads: None,
        params: None,
    };
    fleet
        .load(&peer, "only/ctx", Some(partial))
        .await
        .expect("partial params-load dispatches");
}
