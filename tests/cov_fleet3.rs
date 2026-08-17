//! Round-3 integration coverage for the iroh fleet wire — the arms `cov_fleet2` and the
//! remote suite leave uncovered because they need either (a) a peer that misbehaves in a
//! very specific way on a SPECIFIC method (`M_NODE_LOGS` streams, bounded status reads,
//! pull progress), (b) node-PUSHED `N_FLEET_EVENT` frames raced against the inventory
//! cache (retained-replay / fallback-pull / stamp-ordering arms), or (c) a REAL spawned
//! `higgs --node` child whose node-side download machinery is put under contention.
//!
//! Layout:
//!  - `transport_*`: drive `NodeTransport`'s pub methods directly over an in-process mock
//!    peer (`node_logs` snapshot/follow/EOF, `request_bounded` caps, `pull` demux edges,
//!    the chat payload's optional fields).
//!  - `fleet_*`: drive `HubFleet` over a mock node — seq-less stamp ordering, pushes that
//!    outrun the connect pull (retain + replay + coalesced fallback pull + bounded
//!    give-up), the version-push pin CAS, chat-refresh debounce vs retire, and the
//!    refcounted daemon-log watch.
//!  - `hub_fetch_bytes_*`: the HF card-fetch primary/fallback arms against a loopback
//!    fixture (`HIGGS_HF_ENDPOINT`).
//!  - `node_child_*` / `crafted_hub_*`: a REAL `higgs --node` child over hermetic iroh —
//!    machine-wide download-lock contention vs I/O fault, live pull-status announcement,
//!    hub-vanishes-mid-pull, and node-side wire validation a well-behaved hub never sends.

mod common;

use std::future::IntoFuture;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::response::IntoResponse;
use iroh::endpoint::Connection;
use iroh::Endpoint;
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, BufReader};

use common::higgs_local;
use higgs::auth::{Allowlist, PairingTokens};
use higgs::diagnostic::HiggsError;
use higgs::log_bus::{LogBus, LogSource};
use higgs::node::fleet::{HubFleet, PinnedPush};
use higgs::node::transport::NodeTransport;
use higgs::node::{gate_connection, GateOutcome, HubIdentity, HELLO_DEADLINE};
use higgs::remote::{
    ALPN, M_NODE_INVENTORY, M_NODE_LOAD, M_NODE_LOGS, M_NODE_PULL, M_NODE_UPDATE_VERSION,
    N_FLEET_EVENT, N_LOG_LINE, N_NODE_LOG, N_PROGRESS,
};
use higgs::rpc::{self, RpcError, RpcFrame, RpcNotification, RpcRequest, RpcResponse};
use higgs::worker::{M_CHAT, N_CHAT_CHUNK};

// ── shared helpers ──────────────────────────────────────────────────────────────────────

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// A hermetic (relay-disabled) iroh endpoint on the higgs remote ALPN.
async fn minimal_ep() -> Endpoint {
    Endpoint::builder(iroh::endpoint::presets::Minimal)
        .alpns(vec![ALPN.to_vec()])
        .relay_mode(iroh::RelayMode::Disabled)
        .bind()
        .await
        .expect("bind endpoint")
}

/// One raw ALPN connection between two hermetic endpoints; keep the endpoints alive.
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

fn note(method: &str, params: Value) -> String {
    rpc::encode(&RpcFrame::Notification(RpcNotification {
        jsonrpc: "2.0".into(),
        method: method.into(),
        params,
    }))
}

/// A minimal VALID `M_NODE_INVENTORY` reply (no `snapshot_seq` — the legacy/seq-less
/// shape the stamp-ordering arms need).
fn inventory_json(hostname: &str, workers: &Value) -> Value {
    json!({ "hostname": hostname, "os": "mock", "workers": workers,
            "hardware": { "cpu_name": "m", "arch": "test", "cpu_cores": 1,
                          "ram_total_bytes": 1, "ram_used_bytes": 0,
                          "cpu_usage_percent": 0.0, "gpus": [],
                          "vram_total_bytes": 0 },
            "runtime": { "engine": "mock", "backend": "cpu", "version": "0",
                         "binding": "0" } })
}

/// What the mock peer writes back on an accepted bi stream.
enum Reply {
    /// One Response frame, then finish.
    Response(RpcResponse),
    /// Raw pre-encoded lines, then finish.
    Lines(Vec<String>),
    /// Raw pre-encoded lines, then HOLD the stream open (a live follow stream).
    LinesHold(Vec<String>),
    /// One raw (usually undecodable) line, then finish.
    RawLine(String),
    /// Finish without replying.
    Silent,
}

/// Mock peer serve loop: accept every bi stream, read the one request, reply per handler.
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
                    // `write_all`/`finish` are inherent iroh SendStream methods.
                    let _ = send
                        .write_all(
                            format!("{}\n", rpc::encode(&RpcFrame::Response(resp))).as_bytes(),
                        )
                        .await;
                    let _ = send.finish();
                }
                Reply::Lines(ls) => {
                    for l in &ls {
                        let _ = send.write_all(format!("{l}\n").as_bytes()).await;
                    }
                    let _ = send.finish();
                }
                Reply::LinesHold(ls) => {
                    for l in &ls {
                        let _ = send.write_all(format!("{l}\n").as_bytes()).await;
                    }
                    // Hold open: a healthy idle follow stream. The task dies with the
                    // runtime at test end; the WATCHER side is what tears down first.
                    tokio::time::sleep(Duration::from_secs(60)).await;
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

/// Admit `transport` into `fleet` with per-capability toggles (the identity-carrying
/// admission the production accept loop uses).
#[allow(clippy::too_many_arguments)]
async fn admit(
    fleet: &Arc<HubFleet>,
    peer: &str,
    transport: Arc<NodeTransport>,
    fleet_events: bool,
    log_capable: bool,
    pull_capable: bool,
) {
    fleet
        .add_node_with_identity(
            peer.to_string(),
            transport,
            None,
            Some(2),
            None,
            fleet_events,
            None,
            true,
            None,
            None,
            false,
            false,
            log_capable,
            pull_capable,
            Vec::new(),
        )
        .await;
}

/// Open a uni stream from the mock node's side and push raw lines (the node→hub
/// notification channel `read_node_notifications` consumes).
async fn push_uni(conn: &Connection, lines: &[String]) {
    let mut uni = conn.open_uni().await.expect("open uni");
    for l in lines {
        uni.write_all(format!("{l}\n").as_bytes())
            .await
            .expect("write uni line");
    }
    uni.finish().expect("finish uni");
    // Give QUIC a moment to flush before the stream handle drops.
    tokio::time::sleep(Duration::from_millis(50)).await;
}

/// The serialized `kind` of a broadcast FleetEvent (the enum itself is pub(crate)).
fn ev_kind(ev: &impl serde::Serialize) -> String {
    serde_json::to_value(ev).expect("serialize event")["kind"]
        .as_str()
        .unwrap_or_default()
        .to_string()
}

/// Poll `f` every 100 ms until it returns true or `secs` elapse; false on timeout.
async fn wait_until<F, Fut>(secs: u64, mut f: F) -> bool
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = bool>,
{
    let deadline = tokio::time::Instant::now() + Duration::from_secs(secs);
    while tokio::time::Instant::now() < deadline {
        if f().await {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    false
}

struct Proc(Child);
impl Drop for Proc {
    fn drop(&mut self) {
        // SIGTERM (not kill) so the child's coverage profile flushes.
        // SAFETY: a plain kill(2) on our own child's pid.
        unsafe { libc::kill(self.0.id() as libc::pid_t, libc::SIGTERM) };
        let _ = self.0.wait();
    }
}

// ════════════════════════════════════════════════════════════════════════════════════════
// 1. NodeTransport direct — bounded reads, node_logs stream, pull demux, chat payload
// ════════════════════════════════════════════════════════════════════════════════════════

/// `request_bounded` must surface each malformed-peer behavior as a typed transport-dead
/// error (never a hang, never unbounded buffering): the node closing before replying, a
/// reply line LARGER than the caller's byte cap (the OOM bound the cap exists for), and
/// an unexpected reply frame where a Response is due.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn transport_request_bounded_maps_silent_oversize_and_unexpected_reply_frames() {
    let (dialer, acceptor, _eps) = connect_pair().await;
    let _mock = tokio::spawn(serve_mock(acceptor, |method, id, _p| {
        match method.as_str() {
            "higgs/probe-silent" => Reply::Silent,
            "higgs/probe-oversize" => Reply::RawLine("x".repeat(100_000)),
            "higgs/probe-unexpected" => Reply::Lines(vec![note("higgs/surprise", json!({}))]),
            _ => Reply::Response(resp_err(id, "HG037", "unknown")),
        }
    }));
    let transport = NodeTransport::new(dialer);

    let e = transport
        .request_bounded("higgs/probe-silent", json!({}), 1024)
        .await
        .expect_err("silent close errors");
    assert!(matches!(e, HiggsError::WorkerDead { .. }), "{e:?}");

    let e = transport
        .request_bounded("higgs/probe-oversize", json!({}), 1024)
        .await
        .expect_err("an oversize reply line is rejected at the cap");
    assert!(matches!(e, HiggsError::WorkerDead { .. }), "{e:?}");

    let e = transport
        .request_bounded("higgs/probe-unexpected", json!({}), 1024)
        .await
        .expect_err("a notification where a response is due errors");
    assert!(matches!(e, HiggsError::WorkerDead { .. }), "{e:?}");
}

/// `node_logs` snapshot mode: `N_NODE_LOG` lines reach `on_line` verbatim, a `lagged`
/// marker renders as a visible drop notice, junk/unknown frames are skipped (forward
/// compat), and the final Response completes the call.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn transport_node_logs_snapshot_delivers_lines_lagged_markers_and_skips_junk() {
    let (dialer, acceptor, _eps) = connect_pair().await;
    let _mock = tokio::spawn(serve_mock(acceptor, |method, id, _p| {
        if method == M_NODE_LOGS {
            Reply::Lines(vec![
                note(N_NODE_LOG, json!({ "line": "alpha" })),
                note(N_NODE_LOG, json!({ "lagged": 3 })),
                note(N_NODE_LOG, json!({ "neither": true })), // no line, no lagged → ignored
                "this is not a json frame".into(),            // undecodable → skipped
                note("higgs/other", json!({})),               // unknown method → skipped
                rpc::encode(&RpcFrame::Response(resp_ok(id, json!({})))),
            ])
        } else {
            Reply::Silent
        }
    }));
    let transport = NodeTransport::new(dialer);
    let mut lines = Vec::new();
    let mut on_line = |l: String| lines.push(l);
    let stop = std::future::pending::<()>();
    tokio::pin!(stop);
    transport
        .node_logs(10, false, &mut on_line, stop.as_mut())
        .await
        .expect("snapshot completes on the final response");
    assert_eq!(
        lines,
        vec![
            "alpha".to_string(),
            "… 3 lines dropped (stream lagging)".into()
        ],
        "lines + lagged marker delivered; junk frames skipped"
    );
}

/// `node_logs` termination edges: a bare EOF BEFORE the final Response is an error (a
/// truncated snapshot must not read as whole), and in follow mode the caller's `stop`
/// future ends the stream cleanly (watcher-left teardown, not an error).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn transport_node_logs_errors_on_bare_eof_and_follow_stop_ends_cleanly() {
    let (dialer, acceptor, _eps) = connect_pair().await;
    let follow_mode = Arc::new(AtomicBool::new(false));
    let fm = follow_mode.clone();
    let _mock = tokio::spawn(serve_mock(acceptor, move |method, _id, _p| {
        if method == M_NODE_LOGS {
            if fm.load(Ordering::SeqCst) {
                // Live follow: one line, then hold the stream open (healthy idle log).
                Reply::LinesHold(vec![note(N_NODE_LOG, json!({ "line": "live-1" }))])
            } else {
                // Snapshot cut short: a line, then EOF with NO final Response.
                Reply::Lines(vec![note(N_NODE_LOG, json!({ "line": "cut" }))])
            }
        } else {
            Reply::Silent
        }
    }));
    let transport = NodeTransport::new(dialer);

    let mut lines = Vec::new();
    let mut on_line = |l: String| lines.push(l);
    let stop = std::future::pending::<()>();
    tokio::pin!(stop);
    let e = transport
        .node_logs(10, false, &mut on_line, stop.as_mut())
        .await
        .expect_err("a stream closed before its final response is an error");
    assert!(matches!(e, HiggsError::WorkerDead { .. }), "{e:?}");
    assert_eq!(
        lines,
        vec!["cut".to_string()],
        "delivered lines still surfaced"
    );

    // Follow mode: stop resolves once the first live line lands → clean Ok.
    follow_mode.store(true, Ordering::SeqCst);
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    let mut on_line = move |l: String| {
        let _ = tx.send(l);
    };
    let stop = async move {
        let _ = rx.recv().await; // the first live line is the leave signal
    };
    tokio::pin!(stop);
    transport
        .node_logs(10, true, &mut on_line, stop.as_mut())
        .await
        .expect("a follow stream ended by the watcher is not an error");
}

/// The pull demux: an impossible node-supplied total (less than `downloaded`) degrades to
/// None (no >100% bars downstream), stray frames are skipped, the final Response resolves
/// the pull — and a stream closed before the final is a typed transport-dead error.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn transport_pull_normalizes_impossible_totals_and_errors_on_early_close() {
    let (dialer, acceptor, _eps) = connect_pair().await;
    let _mock = tokio::spawn(serve_mock(acceptor, |method, id, p| {
        if method == M_NODE_PULL {
            let file = p.get("file").and_then(Value::as_str).unwrap_or("");
            if file == "early.gguf" {
                // Progress, then EOF with no final Response.
                Reply::Lines(vec![note(
                    N_PROGRESS,
                    json!({ "request_id": id, "downloaded": 1, "total": 2 }),
                )])
            } else {
                Reply::Lines(vec![
                    // total < downloaded → normalized to None.
                    note(
                        N_PROGRESS,
                        json!({ "request_id": id, "downloaded": 10, "total": 5 }),
                    ),
                    note("higgs/stray", json!({})), // unknown frame → skipped
                    note(
                        N_PROGRESS,
                        json!({ "request_id": id, "downloaded": 20, "total": 40 }),
                    ),
                    rpc::encode(&RpcFrame::Response(resp_ok(id, json!({ "path": "/x" })))),
                ])
            }
        } else {
            Reply::Silent
        }
    }));
    let transport = NodeTransport::new(dialer);

    let mut seen: Vec<(u64, Option<u64>)> = Vec::new();
    let mut on_progress = |d: u64, t: Option<u64>| seen.push((d, t));
    let v = transport
        .pull("acme/m", "ok.gguf", None, &mut on_progress)
        .await
        .expect("pull resolves on the final response");
    assert_eq!(v["path"], "/x");
    assert_eq!(
        seen,
        vec![(10, None), (20, Some(40))],
        "impossible total degraded to unknown-length; valid total passed through"
    );

    let mut on_progress = |_d: u64, _t: Option<u64>| {};
    let e = transport
        .pull("acme/m", "early.gguf", None, &mut on_progress)
        .await
        .expect_err("a pull stream closed before the final is an error");
    assert!(matches!(e, HiggsError::WorkerDead { .. }), "{e:?}");
}

/// The chat payload carries BOTH optional fields when set — `tools` and
/// `chat_template_kwargs` — so a node applies the caller's template arguments (an old
/// node simply ignores unknown keys; that passthrough is the additive-field contract).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn transport_chat_payload_carries_tools_and_template_kwargs() {
    let (dialer, acceptor, _eps) = connect_pair().await;
    let _mock = tokio::spawn(serve_mock(acceptor, |method, id, p| {
        if method == M_CHAT {
            assert_eq!(
                p.get("tools").and_then(Value::as_str),
                Some("[{\"t\":1}]"),
                "tools string forwarded: {p}"
            );
            assert_eq!(
                p.get("chat_template_kwargs").and_then(Value::as_str),
                Some("{\"k\":true}"),
                "template kwargs forwarded: {p}"
            );
            Reply::Lines(vec![
                note(N_CHAT_CHUNK, json!({ "request_id": id, "delta": "hi" })),
                rpc::encode(&RpcFrame::Response(resp_ok(id, json!({ "content": "hi" })))),
            ])
        } else {
            Reply::Silent
        }
    }));
    let transport = NodeTransport::new(dialer);
    let (mut rx, fut) = transport
        .chat(
            1,
            "m/x".into(),
            "[]".into(),
            4,
            0.0,
            Some("[{\"t\":1}]".into()),
            Some("{\"k\":true}".into()),
        )
        .await
        .expect("chat opens");
    let final_v = fut
        .await
        .expect("final result (mock asserts run in-handler)");
    assert_eq!(final_v["content"], "hi");
    let mut got_delta = false;
    while let Some(d) = rx.recv().await {
        if d.text == "hi" {
            got_delta = true;
        }
    }
    assert!(got_delta, "the streamed delta reached the receiver");
}

// ════════════════════════════════════════════════════════════════════════════════════════
// 2. HubFleet over a mock node — ordering, retained pushes, pinned pushes, log watches
// ════════════════════════════════════════════════════════════════════════════════════════

/// A status poll for a node the fleet never admitted is HG027 unreachable — the actor
/// refuses before any capability check or lock allocation (caller-supplied strings must
/// not grow per-node state).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fleet_node_downloads_for_an_unknown_node_is_unreachable() {
    let fleet = Arc::new(HubFleet::new(Arc::new(LogBus::new())));
    let e = fleet
        .node_downloads("never-admitted")
        .await
        .expect_err("unknown node refused");
    assert!(matches!(e, HiggsError::NodeUnreachable { .. }), "{e:?}");
}

/// Seq-less ordering: a node that never reports `snapshot_seq` (legacy shape) commits
/// repeat pulls by the hub's monotonic stamp, and a push onto that seq-less cache takes
/// the stamp-fallback arm (applied, no route reconciliation). A push whose seq does NOT
/// advance the cache is dropped silent (no event), a hub-local kind in a push is a
/// protocol violation (dropped whole), an UNKNOWN future kind keeps its data but
/// generifies the broadcast to InventorySynced, and `N_LOG_LINE` frames file worker
/// stderr into the hub bus (an out-of-range worker id is skipped, not misfiled).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fleet_seqless_inventory_commits_by_stamp_and_pushes_take_the_stamp_fallback() {
    let (dialer, acceptor, _eps) = connect_pair().await;
    let peer = dialer.remote_id().to_string();
    let node_conn = acceptor.clone();
    let _mock = tokio::spawn(serve_mock(acceptor, move |method, id, _p| {
        if method == M_NODE_INVENTORY {
            Reply::Response(resp_ok(id, inventory_json("mock-stamp", &json!([]))))
        } else {
            Reply::Response(resp_err(id, "HG037", "unknown"))
        }
    }));

    let bus = Arc::new(LogBus::new());
    let fleet = Arc::new(HubFleet::new(bus.clone()));
    admit(
        &fleet,
        &peer,
        Arc::new(NodeTransport::new(dialer)),
        true,
        false,
        false,
    )
    .await;

    // Wait for the connect-time pull to commit (cache exists, seq-less).
    assert!(
        wait_until(10, || {
            let fleet = fleet.clone();
            let peer = peer.clone();
            async move {
                fleet
                    .nodes_view()
                    .await
                    .iter()
                    .any(|n| n.endpoint_id == peer && n.inventory.is_some())
            }
        })
        .await,
        "connect-time inventory pull committed"
    );
    // A SECOND seq-less pull commits via the stamp arm (both sides carry no seq).
    fleet
        .refresh_inventory(&peer)
        .await
        .expect("repeat seq-less refresh commits by stamp");

    let mut events = fleet.subscribe_fleet_events();
    // The node pushes its uni-stream notifications: junk, malformed, a spoofed hub-local
    // kind, an unknown future kind, a seq-stale push, the real ChatEnd, and log lines.
    push_uni(
        &node_conn,
        &[
            "not json at all".into(),
            note(N_FLEET_EVENT, json!({ "kind": "chat_end" })), // no seq/workers → dropped
            note(
                N_FLEET_EVENT,
                json!({ "kind": "node_dropped", "snapshot_seq": 100,
                        "workers": [{ "worker_id": 9, "model": "spoof/m" }] }),
            ),
            note(
                N_FLEET_EVENT,
                json!({ "kind": "kind_from_the_future", "snapshot_seq": 1,
                        "workers": [{ "worker_id": 1, "model": "m/a" }] }),
            ),
            note(
                N_FLEET_EVENT,
                json!({ "kind": "chat_end", "snapshot_seq": 1,
                        "workers": [{ "worker_id": 2, "model": "m/stale" }] }),
            ),
            note(
                N_FLEET_EVENT,
                json!({ "kind": "chat_end", "snapshot_seq": 2,
                        "workers": [{ "worker_id": 3, "model": "m/c" }] }),
            ),
            note(
                N_LOG_LINE,
                json!({ "worker_id": 3, "line": "worker says hi" }),
            ),
            note(
                N_LOG_LINE,
                json!({ "worker_id": u64::MAX, "line": "misfile me" }),
            ),
        ],
    )
    .await;

    // Events arrive in stream order: the unknown kind generified to inventory_synced,
    // then the applied chat_end. The spoofed node_dropped and the stale seq-1 push must
    // broadcast NOTHING — proven by order: chat_end arrives without either in between.
    let mut kinds = Vec::new();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    while tokio::time::Instant::now() < deadline {
        match tokio::time::timeout(Duration::from_secs(1), events.recv()).await {
            Ok(Ok(ev)) => {
                kinds.push(ev_kind(&ev));
                if kinds.last().is_some_and(|k| k == "chat_end") {
                    break;
                }
            }
            _ => break,
        }
    }
    assert_eq!(
        kinds,
        vec!["inventory_synced".to_string(), "chat_end".into()],
        "unknown kind generified; hub-local + stale pushes silent"
    );

    // The cache holds the seq-2 snapshot (the stale seq-1 worker never shows).
    let row = fleet
        .nodes_view()
        .await
        .into_iter()
        .find(|n| n.endpoint_id == peer)
        .expect("node row");
    let workers: Vec<String> = row
        .inventory
        .expect("inventory cached")
        .workers
        .iter()
        .map(|w| w.model.clone())
        .collect();
    assert_eq!(
        workers,
        vec!["m/c".to_string()],
        "the applied push replaced the snapshot"
    );

    // The valid worker log line landed in the bus ring; the out-of-range one did not.
    let logged = bus.snapshot(200, None).join("\n");
    assert!(
        logged.contains("worker says hi"),
        "worker line filed: {logged}"
    );
    assert!(
        !logged.contains("misfile me"),
        "out-of-range id skipped: {logged}"
    );
}

/// A push that OUTRUNS the connect-time pull (no cached inventory yet) is retained and
/// triggers ONE coalesced fallback pull; when that pull finally commits, the retained
/// push replays ON TOP of it (newer node truth) and its kind is re-broadcast. The
/// operator-visible outcome: the pushed worker appears in the fleet view even though
/// every inventory pull before it had failed.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fleet_pushes_before_any_snapshot_are_retained_and_replayed_via_the_fallback_pull() {
    let (dialer, acceptor, _eps) = connect_pair().await;
    let peer = dialer.remote_id().to_string();
    let node_conn = acceptor.clone();
    let inv_ok = Arc::new(AtomicBool::new(false));
    let flag = inv_ok.clone();
    let _mock = tokio::spawn(serve_mock(acceptor, move |method, id, _p| {
        if method == M_NODE_INVENTORY {
            if flag.load(Ordering::SeqCst) {
                Reply::Response(resp_ok(id, inventory_json("mock-late", &json!([]))))
            } else {
                // A decodable ERROR (not junk): the pull fails without the transport
                // being classed dead, so the node stays connected while cache-less.
                Reply::Response(resp_err(id, "HG037", "inventory not ready"))
            }
        } else {
            Reply::Response(resp_err(id, "HG037", "unknown"))
        }
    }));

    let fleet = Arc::new(HubFleet::new(Arc::new(LogBus::new())));
    let mut events = fleet.subscribe_fleet_events();
    admit(
        &fleet,
        &peer,
        Arc::new(NodeTransport::new(dialer)),
        true,
        false,
        false,
    )
    .await;

    // The connect pull fails → no cache. Push a worker snapshot the hub can't merge yet.
    tokio::time::sleep(Duration::from_millis(200)).await;
    push_uni(
        &node_conn,
        &[note(
            N_FLEET_EVENT,
            json!({ "kind": "worker_loaded", "snapshot_seq": 5,
                    "workers": [{ "worker_id": 7, "model": "m/pushed" }] }),
        )],
    )
    .await;

    // Let the fallback owner burn at least one failing attempt, then heal the node.
    tokio::time::sleep(Duration::from_millis(1200)).await;
    inv_ok.store(true, Ordering::SeqCst);

    // The retained push replays on the fallback pull's commit: worker visible.
    assert!(
        wait_until(15, || {
            let fleet = fleet.clone();
            let peer = peer.clone();
            async move {
                fleet.nodes_view().await.iter().any(|n| {
                    n.endpoint_id == peer
                        && n.inventory
                            .as_ref()
                            .is_some_and(|i| i.workers.iter().any(|w| w.model == "m/pushed"))
                })
            }
        })
        .await,
        "the retained push replayed onto the fallback pull's commit"
    );
    // The replayed push's kind was re-broadcast (after the commit's inventory_synced).
    let mut saw_kind = false;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(3);
    while tokio::time::Instant::now() < deadline && !saw_kind {
        match tokio::time::timeout(Duration::from_millis(500), events.recv()).await {
            Ok(Ok(ev)) => saw_kind = ev_kind(&ev) == "worker_loaded",
            Ok(Err(_)) | Err(_) => break,
        }
    }
    assert!(saw_kind, "the replayed push announces its own kind");
}

/// The coalesced fallback pull is BOUNDED: against a node whose inventory NEVER becomes
/// valid it retries a few paced attempts and then stands down — the node stays connected,
/// the view stays inventory-less, and the loop does not spin forever. A second push while
/// the owner runs is DEFERRED (no concurrent owner), and a stale-seq push does not
/// replace the retained one.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn fleet_fallback_pull_gives_up_after_bounded_retries_on_junk_inventory() {
    let (dialer, acceptor, _eps) = connect_pair().await;
    let peer = dialer.remote_id().to_string();
    let node_conn = acceptor.clone();
    let _mock = tokio::spawn(serve_mock(acceptor, move |method, id, _p| {
        if method == M_NODE_INVENTORY {
            Reply::Response(resp_err(id, "HG037", "inventory permanently broken"))
        } else {
            Reply::Response(resp_err(id, "HG037", "unknown"))
        }
    }));

    let fleet = Arc::new(HubFleet::new(Arc::new(LogBus::new())));
    admit(
        &fleet,
        &peer,
        Arc::new(NodeTransport::new(dialer)),
        true,
        false,
        false,
    )
    .await;
    tokio::time::sleep(Duration::from_millis(200)).await;

    // First push claims the fallback slot; the second (stale seq 0) is deferred AND
    // refused as a retention replacement — one owner, newest push retained.
    push_uni(
        &node_conn,
        &[
            note(
                N_FLEET_EVENT,
                json!({ "kind": "worker_loaded", "snapshot_seq": 1,
                        "workers": [{ "worker_id": 1, "model": "m/first" }] }),
            ),
            note(
                N_FLEET_EVENT,
                json!({ "kind": "worker_loaded", "snapshot_seq": 0,
                        "workers": [{ "worker_id": 2, "model": "m/stale" }] }),
            ),
        ],
    )
    .await;

    // The owner's 5 paced attempts (~1 s apart) all fail; it must then stand down for
    // good. Poll THROUGH the window asserting no inventory ever materializes.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(8);
    while tokio::time::Instant::now() < deadline {
        let row = fleet
            .nodes_view()
            .await
            .into_iter()
            .find(|n| n.endpoint_id == peer)
            .expect("node stays in the fleet view");
        assert!(
            row.inventory.is_none(),
            "junk inventory never commits a snapshot"
        );
        assert!(row.connected, "a broken inventory is not a disconnect");
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
}

/// The version-only pinned update push refuses on a CAS mismatch — an unknown node and a
/// stale (reconnected-since-snapshot) transport both read `Reconnected`, nothing sent —
/// and an undecodable node reply on the current transport is a dead transport (HG027,
/// node dropped), never a bogus "accepted".
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fleet_push_update_version_pinned_reconnected_and_dead_transport_arms() {
    // Node A answers M_NODE_UPDATE_VERSION with a valid acceptance.
    let (d1, a1, _e1) = connect_pair().await;
    let peer1 = d1.remote_id().to_string();
    let _mock1 = tokio::spawn(serve_mock(a1, |m, id, _p| {
        if m == M_NODE_UPDATE_VERSION {
            Reply::Response(resp_ok(id, json!({ "status": "accepted" })))
        } else {
            Reply::Response(resp_err(id, "HG037", "unknown"))
        }
    }));
    let fleet = Arc::new(HubFleet::new(Arc::new(LogBus::new())));
    let t1 = Arc::new(NodeTransport::new(d1));
    admit(&fleet, &peer1, t1.clone(), false, false, false).await;

    // A transport that was never admitted for any node (the CAS's stale side).
    let (d2, _a2, _e2) = connect_pair().await;
    let stale = Arc::new(NodeTransport::new(d2));

    // Unknown node → Reconnected (no current transport at all).
    assert!(matches!(
        fleet
            .push_update_version_pinned("ghost", &stale, "2.0.0")
            .await
            .expect("refusal, not error"),
        PinnedPush::Reconnected
    ));
    // Known node, stale transport → Reconnected (ptr CAS failed).
    assert!(matches!(
        fleet
            .push_update_version_pinned(&peer1, &stale, "2.0.0")
            .await
            .expect("refusal, not error"),
        PinnedPush::Reconnected
    ));
    // Current transport → Accepted.
    assert!(matches!(
        fleet
            .push_update_version_pinned(&peer1, &t1, "2.0.0")
            .await
            .expect("accepted"),
        PinnedPush::Accepted(_)
    ));

    // Node B replies garbage → transport-dead → HG027 through handle_op_error.
    let (d3, a3, _e3) = connect_pair().await;
    let peer3 = d3.remote_id().to_string();
    let _mock3 = tokio::spawn(serve_mock(a3, |m, _id, _p| {
        if m == M_NODE_UPDATE_VERSION {
            Reply::RawLine("not a frame".into())
        } else {
            Reply::Silent
        }
    }));
    let t3 = Arc::new(NodeTransport::new(d3));
    admit(&fleet, &peer3, t3.clone(), false, false, false).await;
    let e = match fleet.push_update_version_pinned(&peer3, &t3, "2.0.0").await {
        Err(e) => e,
        Ok(_) => panic!("undecodable reply must be a dead transport, not a push outcome"),
    };
    assert!(matches!(e, HiggsError::NodeUnreachable { .. }), "{e:?}");
}

/// A chat whose FINAL is a non-route-invalidating worker error (HG016 timeout class)
/// surfaces the error but KEEPS the route (the worker is alive — dropping it would
/// unroute a healthy instance). The chat-end debounced refresh then runs (legacy
/// pull-model node): back-to-back chats coalesce into one owner with a trailing re-run,
/// and a RETIRE during the settle window makes the woken owner stand down without
/// touching the successor's state.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fleet_chat_error_keeps_route_and_debounced_refresh_survives_retire() {
    let (dialer, acceptor, _eps) = connect_pair().await;
    let peer = dialer.remote_id().to_string();
    let _mock = tokio::spawn(serve_mock(acceptor, move |method, id, _p| {
        match method.as_str() {
            m if m == M_NODE_LOAD => Reply::Response(resp_ok(id, json!({ "worker_id": 1 }))),
            m if m == M_NODE_INVENTORY => {
                Reply::Response(resp_ok(id, inventory_json("mock-chat", &json!([]))))
            }
            // A worker error that is NOT route-invalidating (the generation timed out;
            // the worker itself is alive and stays routed).
            m if m == M_CHAT => Reply::Response(resp_err(id, "HG016", "chat timed out")),
            _ => Reply::Response(resp_err(id, "HG037", "unknown")),
        }
    }));

    let fleet = Arc::new(HubFleet::new(Arc::new(LogBus::new())));
    // Legacy admission (no fleet_events): the debounced chat-end re-pull is live.
    admit(
        &fleet,
        &peer,
        Arc::new(NodeTransport::new(dialer)),
        false,
        false,
        false,
    )
    .await;
    fleet
        .load(&peer, "ok/model", None)
        .await
        .expect("load routes");

    // Two back-to-back failing chats: the second coalesces into the first owner's
    // trailing re-run (one refresh owner at a time, then one more pull).
    for _ in 0..2 {
        let (rx, fut) = fleet
            .chat("ok/model", "[]".into(), 4, 0.0, None, None)
            .await
            .expect("chat dispatches");
        drop(rx);
        let e = fut.await.expect_err("the worker error surfaces");
        assert!(
            matches!(&e, HiggsError::WorkerRpc { worker_code: Some(c), .. } if c == "HG016"),
            "non-invalidating chat error passed through: {e:?}"
        );
        assert!(
            fleet.is_remote("ok/model").await,
            "a non-invalidating chat error keeps the route"
        );
    }
    // Let the debounced owner run its settle + pull + trailing re-run to completion.
    tokio::time::sleep(Duration::from_millis(1200)).await;
    assert!(
        fleet.is_remote("ok/model").await,
        "route survives the refresh cycle"
    );

    // One more chat, then retire INSIDE the 250 ms settle: the woken owner finds its
    // debounce slot cleared and stands down (no pull against the retired node).
    let (rx, fut) = fleet
        .chat("ok/model", "[]".into(), 4, 0.0, None, None)
        .await
        .expect("chat dispatches");
    drop(rx);
    let _ = fut.await;
    tokio::time::sleep(Duration::from_millis(50)).await; // let ChatRefreshBegin land
    fleet.retire(&peer).await;
    tokio::time::sleep(Duration::from_millis(400)).await; // owner wakes into a cleared slot
    assert!(
        !fleet
            .nodes_view()
            .await
            .iter()
            .any(|n| n.endpoint_id == peer),
        "retire removed the node while the refresh owner stood down"
    );
}

/// The refcounted per-node daemon-log watch: the first watcher opens the `M_NODE_LOGS`
/// follow stream (`created`), a joiner shares it, the last drop tears the stream down;
/// a node-side stream DEATH surfaces as a visible "log stream ended" line in the log
/// console and clears the registry so the next toggle-on starts fresh. A node that never
/// advertised `node_logs` gets the friendly capability refusal on BOTH the watch and the
/// one-shot snapshot (never a wire method-not-found).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fleet_watch_node_logs_refcounts_streams_and_surfaces_stream_end() {
    let (dialer, acceptor, _eps) = connect_pair().await;
    let peer = dialer.remote_id().to_string();
    let die_after_line = Arc::new(AtomicBool::new(false));
    let flag = die_after_line.clone();
    let _mock = tokio::spawn(serve_mock(acceptor, move |method, id, p| {
        if method == M_NODE_LOGS {
            let follow = p.get("follow").and_then(Value::as_bool).unwrap_or(false);
            if !follow {
                Reply::Lines(vec![
                    note(N_NODE_LOG, json!({ "line": "snap-1" })),
                    note(N_NODE_LOG, json!({ "line": "snap-2" })),
                    rpc::encode(&RpcFrame::Response(resp_ok(id, json!({})))),
                ])
            } else if flag.load(Ordering::SeqCst) {
                // A follow stream that dies: one line then EOF (no final Response).
                Reply::Lines(vec![note(N_NODE_LOG, json!({ "line": "doomed" }))])
            } else {
                Reply::LinesHold(vec![note(N_NODE_LOG, json!({ "line": "live-1" }))])
            }
        } else if method == M_NODE_INVENTORY {
            Reply::Response(resp_ok(id, inventory_json("mock-logs", &json!([]))))
        } else {
            Reply::Response(resp_err(id, "HG037", "unknown"))
        }
    }));

    let fleet = Arc::new(HubFleet::new(Arc::new(LogBus::new())));
    admit(
        &fleet,
        &peer,
        Arc::new(NodeTransport::new(dialer)),
        false,
        true,
        false,
    )
    .await;

    // One-shot snapshot (follow=false): the collected lines, bounded, no stream kept.
    let lines = fleet
        .node_logs_snapshot(&peer, 5)
        .await
        .expect("snapshot returns");
    assert_eq!(lines, vec!["snap-1".to_string(), "snap-2".into()]);

    // Live watch: first watcher creates the stream, a second joins it.
    let mut w1 = fleet.watch_node_logs(&peer, 5).await.expect("first watch");
    assert!(w1.created, "first watcher spawns the streaming task");
    let got_live = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let l = w1.rx.recv().await.expect("bus line");
            if matches!(l.source, LogSource::RemoteNode { .. }) && l.text.contains("live-1") {
                return true;
            }
        }
    })
    .await
    .unwrap_or(false);
    assert!(got_live, "the follow stream's line reached the hub bus");
    let w2 = fleet.watch_node_logs(&peer, 5).await.expect("second watch");
    assert!(!w2.created, "a joiner shares the existing stream");
    drop(w2); // count 2 → 1: stream stays up
    drop(w1); // count 1 → 0: stop fires, task exits, registry entry cleared

    // A fresh watch after teardown starts a NEW stream; this one DIES node-side and the
    // end reason must land in the console as a visible line.
    die_after_line.store(true, Ordering::SeqCst);
    let mut w3 = fleet.watch_node_logs(&peer, 5).await.expect("fresh watch");
    assert!(
        w3.created,
        "the registry was cleared, so this watcher re-creates"
    );
    let saw_end = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let l = w3.rx.recv().await.expect("bus line");
            if matches!(l.source, LogSource::RemoteNode { .. })
                && l.text.contains("log stream ended")
            {
                return true;
            }
        }
    })
    .await
    .unwrap_or(false);
    assert!(
        saw_end,
        "a dead follow stream surfaces its end reason in the console"
    );
    drop(w3);

    // Capability refusal: a second node admitted WITHOUT node_logs.
    let (d2, a2, _e2) = connect_pair().await;
    let peer2 = d2.remote_id().to_string();
    let _mock2 = tokio::spawn(serve_mock(a2, |_m, id, _p| {
        Reply::Response(resp_err(id, "HG037", "x"))
    }));
    admit(
        &fleet,
        &peer2,
        Arc::new(NodeTransport::new(d2)),
        false,
        false,
        false,
    )
    .await;
    let e = match fleet.watch_node_logs(&peer2, 5).await {
        Err(e) => e,
        Ok(_) => panic!("watch must be refused for a pre-capability node"),
    };
    assert!(e.to_string().contains("node_logs-capable"), "{e}");
    let e = fleet
        .node_logs_snapshot(&peer2, 5)
        .await
        .expect_err("snapshot refused for a pre-capability node");
    assert!(e.to_string().contains("node_logs-capable"), "{e}");
}

// ════════════════════════════════════════════════════════════════════════════════════════
// 3. src/hub.rs — card-fetch primary/fallback classification against a loopback fixture
// ════════════════════════════════════════════════════════════════════════════════════════

/// Loopback "HuggingFace" for `fetch_bytes`: behavior keyed by the repo NAME segment.
///  - `fx/rescue`: HEAD → 500 (the hub-client primary always HEADs first), GET → bytes —
///    so the primary fails and the reqwest fallback rescues.
///  - `fx/ratelimit`: 429 everywhere (primary classifies HG031; fallback reports HTTP 429).
///  - `fx/conflict`: 409 everywhere (an HFError variant higgs has no specific code for —
///    the classifier's catch-all).
///  - anything else: 404 (primary EntryNotFound; fallback HTTP 404).
async fn hf_fixture() -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = axum::Router::new().fallback(axum::routing::any(
        |method: axum::http::Method, uri: axum::http::Uri| async move {
            let p = uri.path();
            if p.contains("/rescue/") {
                if method == axum::http::Method::HEAD {
                    (axum::http::StatusCode::INTERNAL_SERVER_ERROR, "").into_response()
                } else {
                    b"card-bytes".to_vec().into_response()
                }
            } else if p.contains("/ratelimit/") {
                (axum::http::StatusCode::TOO_MANY_REQUESTS, "slow down").into_response()
            } else if p.contains("/conflict/") {
                (axum::http::StatusCode::CONFLICT, "conflict").into_response()
            } else {
                (axum::http::StatusCode::NOT_FOUND, "nope").into_response()
            }
        },
    ));
    tokio::spawn(axum::serve(listener, app).into_future());
    format!("http://{addr}")
}

/// `fetch_bytes` primary/fallback composition: the reqwest fallback RESCUES a failing
/// hub-client primary (bytes delivered, no error), and when both fail the exhausted
/// error carries BOTH diagnoses — the primary's classified code (rate-limit HG031, the
/// catch-all client error) and the fallback's HTTP status.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn hub_fetch_bytes_fallback_rescue_and_classified_exhaustion() {
    // Hold the harness home lock (via a LocalHiggs) so the process-global
    // HIGGS_HF_ENDPOINT override cannot race another harness test.
    let Some(local) = higgs_local(&[]).await else {
        eprintln!("skipping hub_fetch_bytes: no tiny GGUF (set HIGGS_TEST_GGUF)");
        return;
    };
    let endpoint = hf_fixture().await;
    // SAFETY: serialized by the held harness lock; LocalHiggs::drop restores the var.
    unsafe { std::env::set_var("HIGGS_HF_ENDPOINT", &endpoint) };

    // Primary HEAD 500 → classified; fallback GET succeeds → bytes, no error.
    let bytes = higgs::hub::fetch_bytes("fx/rescue", "README.md")
        .await
        .expect("the reqwest fallback rescues a failing primary");
    assert_eq!(bytes, b"card-bytes");

    // 429 everywhere: primary classifies rate-limited; fallback reports the status.
    let e = higgs::hub::fetch_bytes("fx/ratelimit", "README.md")
        .await
        .expect_err("both paths rate-limited");
    match &e {
        HiggsError::HubFetchExhausted {
            primary, fallback, ..
        } => {
            assert!(
                primary.contains("HG031"),
                "primary classified as rate-limited: {primary}"
            );
            assert!(
                fallback.contains("429"),
                "fallback carries the HTTP status: {fallback}"
            );
        }
        other => panic!("expected HubFetchExhausted, got {other:?}"),
    }

    // 409: an HFError variant with no specific higgs code → the catch-all client error.
    let e = higgs::hub::fetch_bytes("fx/conflict", "README.md")
        .await
        .expect_err("both paths fail on 409");
    match &e {
        HiggsError::HubFetchExhausted { primary, .. } => {
            assert!(
                primary.contains("HG035"),
                "conflict falls into the catch-all: {primary}"
            );
        }
        other => panic!("expected HubFetchExhausted, got {other:?}"),
    }

    // 404: primary not-found + fallback HTTP 404 (the exhausted pair remote_download
    // already relies on — pinned here at the fetch_bytes level).
    let e = higgs::hub::fetch_bytes("fx/gone", "README.md")
        .await
        .expect_err("both paths 404");
    match &e {
        HiggsError::HubFetchExhausted { fallback, .. } => {
            assert!(
                fallback.contains("404"),
                "fallback names the status: {fallback}"
            );
        }
        other => panic!("expected HubFetchExhausted, got {other:?}"),
    }

    local.shutdown().await;
}

// ════════════════════════════════════════════════════════════════════════════════════════
// 4. Real `higgs --node` child — download-slot contention, live status, hub-vanish
// ════════════════════════════════════════════════════════════════════════════════════════

/// Loopback "HuggingFace" for the NODE's pulls: `slow-*` holds the transfer open ~3 s
/// (a live in-flight window), `chunked-*` streams WITHOUT a content-length (the node
/// must report unknown-length progress), everything else returns bytes immediately.
async fn node_hub_fixture() -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = axum::Router::new().fallback(axum::routing::any(
        |method: axum::http::Method, uri: axum::http::Uri| async move {
            let p = uri.path().to_string();
            if method == axum::http::Method::HEAD {
                // Headerless OK: the hub-client primary proceeds to GET without a length.
                return axum::http::StatusCode::OK.into_response();
            }
            if p.contains("slow-") {
                tokio::time::sleep(Duration::from_secs(3)).await;
                return b"GGUF-slow-bytes".to_vec().into_response();
            }
            if p.contains("chunked-") {
                // A streamed body: no content-length → total unknown end-to-end.
                let chunks: Vec<Result<&'static [u8], std::io::Error>> =
                    vec![Ok(b"GGUF-"), Ok(b"chunked-"), Ok(b"bytes")];
                let stream = futures::stream::iter(chunks);
                return axum::body::Body::from_stream(stream).into_response();
            }
            b"GGUF-plain-bytes".to_vec().into_response()
        },
    ));
    tokio::spawn(axum::serve(listener, app).into_future());
    format!("http://{addr}")
}

/// The node's machine-wide download authority, live status announcement, and orphan
/// behavior, against a REAL spawned node:
///  - a `(repo,file)` slot flocked by ANOTHER process (this test) is refused [HG090];
///  - an UNWRITABLE locks dir is the distinct I/O fault [HG034], never "in flight";
///  - a length-less transfer completes with unknown-total progress end to end;
///  - while a slow pull runs, `node_downloads` announces the registry row
///    (cancellable) and a case-variant ledger row for the SAME key is suppressed;
///  - the hub disabling MID-pull leaves the node finishing in the background (the
///    hub-side op fails; the node connection is gone).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn node_child_download_slots_status_and_hub_vanish() {
    let Some(higgs) = higgs_local(&[]).await else {
        eprintln!("skipping node_child_download_slots: no tiny GGUF (set HIGGS_TEST_GGUF)");
        return;
    };
    higgs.hub_enable().await.expect("hub enable");
    let pair = higgs.pair().await.expect("mint pairing");

    let endpoint = node_hub_fixture().await;
    let node_home = tempfile::tempdir().unwrap();
    let _node = Proc(
        Command::new(env!("CARGO_BIN_EXE_higgs"))
            .arg("--node")
            .arg(&pair.ticket)
            .arg(&pair.token)
            .env("HIGGS_HOME", node_home.path())
            .env("HIGGS_HF_ENDPOINT", &endpoint)
            .env("HIGGS_IROH_LOCAL", "1")
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("spawn higgs --node"),
    );
    let mut node_id = String::new();
    for _ in 0..150 {
        if let Some(n) = higgs
            .nodes()
            .await
            .into_iter()
            .find(|n| n.connected && !n.is_local)
        {
            node_id = n.endpoint_id.clone();
            break;
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    assert!(!node_id.is_empty(), "remote node paired + connected");
    let node_models = node_home.path().join("models");
    std::fs::create_dir_all(&node_models).unwrap();

    // ── contention: THIS process flocks the slot; the node must refuse HG090 ────────
    {
        let _held = higgs::catalog::download_lock::DownloadLock::acquire(
            &node_models,
            "acme/tiny",
            "held.gguf",
        )
        .expect("test process claims the machine-wide slot");
        let err = higgs
            .model_download_on(&node_id, "acme/tiny", "held.gguf")
            .await
            .expect_err("the node refuses a slot another process holds");
        let code = match &err {
            HiggsError::WorkerRpc { worker_code, .. } => worker_code.clone(),
            other => miette::Diagnostic::code(other).map(|c| c.to_string()),
        };
        assert_eq!(code.as_deref(), Some("HG090"), "contention is HG090: {err}");
    } // lock released

    // ── I/O fault: locks dir unwritable → HG034, NOT "already in flight" ────────────
    {
        use std::os::unix::fs::PermissionsExt;
        let locks = higgs::catalog::download_lock::locks_dir(&node_models);
        std::fs::create_dir_all(&locks).unwrap();
        std::fs::set_permissions(&locks, std::fs::Permissions::from_mode(0o555)).unwrap();
        let err = higgs
            .model_download_on(&node_id, "acme/tiny", "faulty.gguf")
            .await
            .expect_err("an unwritable locks dir is a filesystem fault");
        std::fs::set_permissions(&locks, std::fs::Permissions::from_mode(0o755)).unwrap();
        let code = match &err {
            HiggsError::WorkerRpc { worker_code, .. } => worker_code.clone(),
            other => miette::Diagnostic::code(other).map(|c| c.to_string()),
        };
        assert_eq!(code.as_deref(), Some("HG034"), "I/O fault is HG034: {err}");
    }

    // ── unknown-length transfer completes (chunked body, no content-length) ─────────
    let path = higgs
        .model_download_on(&node_id, "acme/tiny", "chunked-a.gguf")
        .await
        .expect("length-less download lands");
    assert_eq!(
        std::fs::read(&path).unwrap(),
        b"GGUF-chunked-bytes",
        "streamed bytes intact"
    );

    // ── live status while a slow pull runs + case-variant ledger suppression ────────
    let slow = {
        let higgs = higgs.handle();
        let node_id = node_id.clone();
        tokio::spawn(async move {
            higgs
                .model_download_on(&node_id, "acme/tiny", "slow-a.gguf")
                .await
        })
    };
    // The node registers the pull before the fixture's 3 s hold releases it.
    let mut announced = Vec::new();
    for _ in 0..40 {
        let rows = higgs.node_downloads(&node_id).await.unwrap_or_default();
        if rows.iter().any(|d| d.file == "slow-a.gguf") {
            announced = rows;
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    let row = announced
        .iter()
        .find(|d| d.repo == "acme/tiny" && d.file == "slow-a.gguf")
        .expect("the in-flight pull is announced via live status");
    assert!(row.cancellable, "a registry-backed row is cancellable");

    // A case-variant LEDGER row for the SAME key (another process's stale echo on a
    // case-insensitive filesystem) must NOT announce beside the live registry row.
    {
        use higgs::catalog::ledger;
        use higgs::catalog::wire::{DownloadLedgerEntry, DownloadLedgerStatus};
        let echo = DownloadLedgerEntry {
            repo: "ACME/TINY".into(),
            file: "SLOW-A.GGUF".into(),
            pid: std::process::id(),
            pid_started_at: None,
            started_at_ms: now_ms(),
            downloaded: 1,
            total: Some(2),
            status: DownloadLedgerStatus::Downloading,
            ended_at_ms: None,
            path: None,
            detail: None,
        };
        std::fs::write(
            ledger::ledger_path(&node_models),
            serde_json::to_vec(&vec![echo]).unwrap(),
        )
        .unwrap();
        let rows = higgs.node_downloads(&node_id).await.expect("status poll");
        let matches: Vec<_> = rows
            .iter()
            .filter(|d| {
                d.repo.eq_ignore_ascii_case("acme/tiny")
                    && d.file.eq_ignore_ascii_case("slow-a.gguf")
            })
            .collect();
        assert_eq!(
            matches.len(),
            1,
            "one row per key — registry wins: {rows:?}"
        );
        assert!(
            matches[0].cancellable,
            "the surviving row is the live registry one"
        );
    }

    // ── the hub vanishes MID-pull: the op fails hub-side; nothing hangs ─────────────
    higgs.hub_disable().await;
    let res = tokio::time::timeout(Duration::from_secs(10), slow)
        .await
        .expect("the in-flight op resolves promptly when the hub drops")
        .expect("join");
    assert!(
        res.is_err(),
        "the hub-side op cannot succeed after the hub dropped"
    );

    higgs.shutdown().await;
}

/// Node-side WIRE validation a well-behaved hub never exercises: a crafted hub (this
/// test gates the node's dial itself) sends `M_NODE_PULL` frames the production sender
/// cannot produce — undecodable params, a non-GGUF file, a malformed revision — and the
/// node must refuse each with a typed error BEFORE any fetch or registry slot.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn crafted_hub_pull_requests_hit_node_side_validation() {
    let hub = minimal_ep().await;
    let hub_id = hub.id().to_string();
    let ticket = iroh_tickets::endpoint::EndpointTicket::new(hub.addr()).to_string();
    let mut allow = {
        // A writable, uniquely-pathed allowlist under the scratchpad temp dir.
        let path = std::env::temp_dir().join(format!(
            "higgs-cov-fleet3-allow-{}-{:x}.json",
            std::process::id(),
            now_ms()
        ));
        let _ = std::fs::remove_file(&path);
        Allowlist::load(&path).expect("empty allowlist loads")
    };
    let mut tokens = PairingTokens::new();
    let token = tokens.mint(now_ms(), 600_000);

    let node_home = tempfile::tempdir().unwrap();
    let _node = Proc(
        Command::new(env!("CARGO_BIN_EXE_higgs"))
            .arg("--node")
            .arg(&ticket)
            .arg(&token)
            .env("HIGGS_HOME", node_home.path())
            .env("HIGGS_IROH_LOCAL", "1")
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("spawn higgs --node"),
    );

    // Gate the node's dial exactly as the production accept loop would.
    let incoming = tokio::time::timeout(Duration::from_secs(30), hub.accept())
        .await
        .expect("node dialed within 30s")
        .expect("incoming");
    let conn = incoming.await.expect("connection");
    let outcome = gate_connection(
        &conn,
        &mut allow,
        &mut tokens,
        now_ms(),
        &HubIdentity::new(hub_id),
        Some("cov3".into()),
        HELLO_DEADLINE,
    )
    .await;
    assert!(
        matches!(outcome, GateOutcome::Admitted { .. }),
        "the real node's HELLO is admitted by the crafted gate: {outcome:?}"
    );

    /// One crafted request → the node's single reply line.
    async fn send_req(conn: &Connection, id: u64, method: &str, params: Value) -> RpcResponse {
        let (mut send, recv) = conn.open_bi().await.expect("open_bi");
        let req = RpcRequest {
            jsonrpc: "2.0".into(),
            id,
            method: method.into(),
            params,
        };
        send.write_all(format!("{}\n", rpc::encode(&RpcFrame::Request(req))).as_bytes())
            .await
            .expect("write request");
        let _ = send.finish();
        let mut lines = BufReader::new(recv).lines();
        loop {
            let line = tokio::time::timeout(Duration::from_secs(10), lines.next_line())
                .await
                .expect("reply within 10s")
                .expect("read ok")
                .expect("a reply line before EOF");
            if let Ok(RpcFrame::Response(r)) = rpc::decode(&line) {
                return r;
            }
            // progress notifications (none expected here) are skipped
        }
    }

    // Undecodable pull params → the -32602 invalid-params refusal.
    let r = send_req(&conn, 1, M_NODE_PULL, json!({ "bogus": true })).await;
    let err = r.error.expect("refused");
    assert_eq!(
        err.code, -32602,
        "undecodable params are invalid-params: {}",
        err.message
    );
    assert!(
        err.message.contains("invalid pull params"),
        "{}",
        err.message
    );

    // A non-GGUF file fails dest-path validation BEFORE any lock/registry/fetch.
    let r = send_req(
        &conn,
        2,
        M_NODE_PULL,
        json!({ "request_id": 2, "repo": "acme/tiny", "file": "evil.txt" }),
    )
    .await;
    let err = r.error.expect("refused");
    assert!(
        err.message.contains("[HG025]"),
        "non-gguf file refused with the typed download error: {}",
        err.message
    );

    // A malformed revision is refused up front (it must never occupy a cancel slot).
    let r = send_req(
        &conn,
        3,
        M_NODE_PULL,
        json!({ "request_id": 3, "repo": "acme/tiny", "file": "ok.gguf",
                "revision": "bad rev" }),
    )
    .await;
    let err = r.error.expect("refused");
    assert!(
        err.message.contains("invalid revision"),
        "malformed revision refused: {}",
        err.message
    );

    conn.close(0u32.into(), b"done");
}
