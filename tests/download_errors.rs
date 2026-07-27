//! Black-box: a real `higgs --node`, paired to an in-process hub, runs `M_NODE_PULL`
//! against a LOCAL axum HTTP server (via `HIGGS_HF_ENDPOINT`, so NO network). The local
//! server returns deliberate error responses (404 / 403 / 401 / a truncated body / a closed
//! connection) and we assert the final `RpcResponse.error` carries the expected classified
//! `HGxxx` from `src/download.rs` + `src/hub.rs`.
//!
//! The node downloads via `download_dual(HubFetcher primary, HttpFetcher fallback)`; when BOTH
//! transports exhaust against the local server the terminal code is `HG036`
//! (`HubFetchExhausted`), whose message carries BOTH diagnoses — and the `reqwest` FALLBACK's
//! portion is the DETERMINISTIC classification (`http_status_to_error`: 404→HG030, 401/403→HG029,
//! 429→HG031, else→HG032), so we assert on the fallback's `[HGxxx]` substring inside the HG036.
//! (The hub-client primary's own classification of a non-HF local server is implementation
//! -dependent, so we don't pin it.) Skips without a tiny GGUF.

mod common;

use std::process::{Child, Command, Stdio};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use iroh::endpoint::Connection;
use iroh_tickets::endpoint::EndpointTicket;
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, BufReader};

use higgs::auth::{Allowlist, PairingTokens};
use higgs::node::{gate_connection, GateOutcome, HubIdentity, HELLO_DEADLINE};
use higgs::remote::{ALPN, M_NODE_PULL, N_PROGRESS};
use higgs::rpc::{self, RpcFrame, RpcRequest, RpcResponse};

use common::tiny_gguf_path;

/// A spawned `higgs --node`, SIGTERMed on drop (graceful → coverage flush).
struct NodeProc(Child);
impl Drop for NodeProc {
    fn drop(&mut self) {
        // SAFETY: a plain kill(2) on our own child's pid.
        unsafe { libc::kill(self.0.id() as libc::pid_t, libc::SIGTERM) };
        let _ = self.0.wait();
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// A minimal, relay-disabled iroh endpoint speaking the hub ALPN — the in-process "hub".
async fn hub_endpoint() -> iroh::Endpoint {
    iroh::Endpoint::builder(iroh::endpoint::presets::Minimal)
        .alpns(vec![ALPN.to_vec()])
        .relay_mode(iroh::RelayMode::Disabled)
        .bind()
        .await
        .expect("bind hub endpoint")
}

/// One control RPC (fresh bi stream, single response). Unused by these error tests but kept
/// to mirror `pull.rs` exactly; allowed to be dead.
#[allow(dead_code)]
async fn node_rpc(conn: &Connection, id: u64, method: &str, params: Value) -> RpcResponse {
    let (mut send, recv) = conn.open_bi().await.expect("open control stream");
    let req = RpcRequest {
        jsonrpc: "2.0".into(),
        id,
        method: method.into(),
        params,
    };
    send.write_all(format!("{}\n", rpc::encode(&RpcFrame::Request(req))).as_bytes())
        .await
        .expect("write");
    send.finish().expect("finish");
    let mut lines = BufReader::new(recv).lines();
    let line = lines.next_line().await.expect("read").expect("a line");
    match rpc::decode(&line).expect("decode") {
        RpcFrame::Response(r) => r,
        other => panic!("expected response, got {other:?}"),
    }
}

/// Spawn a `higgs --node` paired to `hub`, with its downloader pointed at `endpoint`
/// (a local axum server), and complete the gate handshake. Returns the admitted connection
/// plus the live `NodeProc` (kept alive by the caller) and the node's temp HOME.
async fn paired_node(
    hub: &iroh::Endpoint,
    endpoint: &str,
) -> (Connection, NodeProc, tempfile::TempDir) {
    let node_home = tempfile::tempdir().expect("node home");
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
        .env("HIGGS_IROH_LOCAL", "1")
        .env("HIGGS_HF_ENDPOINT", endpoint)
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("spawn higgs --node");
    let node = NodeProc(child);

    let incoming = tokio::time::timeout(Duration::from_secs(30), hub.accept())
        .await
        .expect("dial within 30s")
        .expect("incoming");
    let conn = incoming.await.expect("connection");
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
        "admitted: {outcome:?}"
    );
    (conn, node, node_home)
}

/// Drive `M_NODE_PULL` over a fresh data stream and return the FINAL `RpcResponse`
/// (draining any `N_PROGRESS` notifications first, then dropping the stream so nothing
/// is left open). `repo` must be a valid `<org>/<model>` and `file` a `*.gguf`, else the
/// node short-circuits with HG025 before ever touching the server.
async fn pull(conn: &Connection, repo: &str, file: &str) -> RpcResponse {
    let (mut send, recv) = conn.open_bi().await.expect("data stream");
    let req = RpcRequest {
        jsonrpc: "2.0".into(),
        id: 1,
        method: M_NODE_PULL.into(),
        params: json!({ "request_id": 1, "repo": repo, "file": file }),
    };
    send.write_all(format!("{}\n", rpc::encode(&RpcFrame::Request(req))).as_bytes())
        .await
        .unwrap();
    send.finish().unwrap();

    let mut lines = BufReader::new(recv).lines();
    loop {
        let line = lines
            .next_line()
            .await
            .expect("read frame")
            .expect("a frame");
        match rpc::decode(&line).expect("decode") {
            RpcFrame::Notification(n) if n.method == N_PROGRESS => continue,
            RpcFrame::Response(r) => return r,
            other => panic!("unexpected pull frame: {other:?}"),
        }
    }
}

/// Spawn a local axum server on `127.0.0.1:0` that returns `status` (with a short body)
/// for EVERY path — so both the hub-client primary and the reqwest fallback see the same
/// non-success response. Returns its base URL (`http://127.0.0.1:PORT`).
async fn serve_status(status: axum::http::StatusCode) -> String {
    let http = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", http.local_addr().unwrap());
    let app = axum::Router::new().fallback(move || async move { (status, "nope") });
    tokio::spawn(async move {
        let _ = axum::serve(http, app).await;
    });
    base
}

/// The classified `[HGxxx]` code the `reqwest` FALLBACK assigns to `status`, per
/// `hub::http_status_to_error`. This is the deterministic portion of the terminal HG036.
fn fallback_code(status: u16) -> &'static str {
    match status {
        401 | 403 => "[HG029]",
        404 => "[HG030]",
        429 => "[HG031]",
        _ => "[HG032]",
    }
}

/// Shared driver: pull against a local server returning `status`, assert the final error is
/// the terminal `HG036` carrying the deterministic fallback code for that status.
async fn assert_pull_status_maps_to(port: u16, status: axum::http::StatusCode) {
    let _ = port; // ports for the iroh hub are local-discovery; the HTTP server binds :0.
    let base = serve_status(status).await;
    let hub = hub_endpoint().await;
    let (conn, _node, _home) = paired_node(&hub, &base).await;

    let resp = pull(&conn, "higgs-test/err", "model.gguf").await;
    let err = resp.error.expect("pull must fail").message;
    let want = fallback_code(status.as_u16());
    assert!(
        err.starts_with("[HG036]"),
        "both transports exhausted → terminal HG036, got: {err}"
    );
    // Isolate the FALLBACK diagnosis (the deterministic `reqwest`/`http_status_to_error` path —
    // the hub-client primary's own mapping of a non-HF local server is implementation-dependent,
    // so we pin only the fallback). The HG036 display is
    //   "... — primary: <primary>; fallback: <fallback>".
    let fb = err
        .split_once("; fallback: ")
        .map(|(_, fb)| fb)
        .unwrap_or_else(|| panic!("HG036 must carry a `; fallback:` segment: {err}"));
    // The fallback streamed the resolve URL and saw this exact HTTP status, then classified it.
    assert!(
        fb.contains(&format!("HTTP {}", status.as_u16())),
        "fallback diagnosis must record the HTTP {} it saw: {err}",
        status.as_u16()
    );
    assert!(
        fb.starts_with(want),
        "fallback must classify HTTP {} as {want} (http_status_to_error): {err}",
        status.as_u16()
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pull_404_classifies_not_found() {
    let Some(_gguf) = tiny_gguf_path() else {
        eprintln!("skipping pull_404: no tiny GGUF");
        return;
    };
    // 404 → not-found. Fallback maps it to HG030; final wire error is HG036 carrying it.
    assert_pull_status_maps_to(12300, axum::http::StatusCode::NOT_FOUND).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pull_403_classifies_auth() {
    let Some(_gguf) = tiny_gguf_path() else {
        eprintln!("skipping pull_403: no tiny GGUF");
        return;
    };
    // 403 Forbidden → auth (HG029) on the fallback path.
    assert_pull_status_maps_to(12301, axum::http::StatusCode::FORBIDDEN).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pull_401_classifies_auth() {
    let Some(_gguf) = tiny_gguf_path() else {
        eprintln!("skipping pull_401: no tiny GGUF");
        return;
    };
    // 401 Unauthorized → auth (HG029) on the fallback path (same arm as 403).
    assert_pull_status_maps_to(12302, axum::http::StatusCode::UNAUTHORIZED).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pull_500_classifies_http_status() {
    let Some(_gguf) = tiny_gguf_path() else {
        eprintln!("skipping pull_500: no tiny GGUF");
        return;
    };
    // 500 → the catch-all HubHttpStatus (HG032) on the fallback path.
    assert_pull_status_maps_to(12303, axum::http::StatusCode::INTERNAL_SERVER_ERROR).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pull_truncated_body_with_wrong_length_fails() {
    let Some(_gguf) = tiny_gguf_path() else {
        eprintln!("skipping pull_truncated: no tiny GGUF");
        return;
    };
    // A 200 whose advertised Content-Length far exceeds the bytes actually sent: the body is
    // truncated mid-stream. The reqwest fallback streams chunks then the connection ends short
    // of the promised length → a transport error (HG033). The hub-client primary likewise fails
    // (a short/garbage GGUF stream), so the terminal error is HG036 — and it is NOT a success.
    let big_len = 50_000_000u64; // promise ~50MB, send ~9 bytes
    let http = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", http.local_addr().unwrap());
    let app = axum::Router::new().fallback(move || async move {
        let mut headers = axum::http::HeaderMap::new();
        headers.insert(
            axum::http::header::CONTENT_LENGTH,
            axum::http::HeaderValue::from_str(&big_len.to_string()).unwrap(),
        );
        // Body shorter than Content-Length → the client sees a truncated stream.
        (headers, "truncated")
    });
    tokio::spawn(async move {
        let _ = axum::serve(http, app).await;
    });

    let hub = hub_endpoint().await;
    let (conn, _node, _home) = paired_node(&hub, &base).await;
    let resp = pull(&conn, "higgs-test/short", "model.gguf").await;
    let err = resp
        .error
        .expect("truncated body must fail the pull")
        .message;
    // Terminal dual-path exhaustion; the truncated stream cannot yield a valid model file.
    assert!(
        err.starts_with("[HG036]"),
        "truncated body → both transports exhaust → HG036: {err}"
    );
    // The on-disk dest must NOT exist (failed download leaves no file in the catalog).
    let dest = node_home_dest(&_home, "higgs-test/short", "model.gguf");
    assert!(
        !dest.exists(),
        "no partial GGUF left after a failed pull: {dest:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pull_connection_closed_fails_transport() {
    let Some(_gguf) = tiny_gguf_path() else {
        eprintln!("skipping pull_closed: no tiny GGUF");
        return;
    };
    // The server accepts the TCP connection then immediately drops it (no HTTP response at all):
    // the reqwest fallback gets a transport error (HG033); the hub-client primary likewise. The
    // terminal error is HG036. This exercises the HubTransport classification arm end-to-end.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move {
        // Accept then immediately drop each connection (no HTTP response written).
        while let Ok((stream, _)) = listener.accept().await {
            drop(stream);
        }
    });

    let hub = hub_endpoint().await;
    let (conn, _node, _home) = paired_node(&hub, &base).await;
    let resp = pull(&conn, "higgs-test/dead", "model.gguf").await;
    let err = resp
        .error
        .expect("closed connection must fail the pull")
        .message;
    assert!(
        err.starts_with("[HG036]"),
        "closed connection → both transports exhaust → HG036: {err}"
    );
    assert!(
        err.contains("[HG033]"),
        "HG036 must carry the fallback's transport code HG033 for a dropped connection: {err}"
    );
}

/// The on-disk destination a pull WOULD land at, under the node's isolated HOME models dir
/// (`<HIGGS_HOME>/models/<org>/<model>/<file>`).
fn node_home_dest(home: &tempfile::TempDir, repo: &str, file: &str) -> std::path::PathBuf {
    home.path().join("models").join(repo).join(file)
}
