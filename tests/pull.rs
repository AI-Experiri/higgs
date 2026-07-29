//! Black-box P4b: a real `higgs --node` downloads a GGUF over `M_NODE_PULL` from a LOCAL
//! HTTP server (via the `HIGGS_HF_ENDPOINT` override, so no network), streaming `N_PROGRESS`,
//! and a subsequent `M_NODE_SCAN` then lists the pulled model. Skips without a tiny GGUF.

mod common;

use std::process::{Child, Command, Stdio};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use iroh::endpoint::Connection;
use iroh_tickets::endpoint::EndpointTicket;
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, BufReader};

use higgs::auth::{Allowlist, PairingTokens};
use higgs::node::{gate_connection, GateOutcome, HubIdentity, HELLO_DEADLINE};
use higgs::remote::{ALPN, M_NODE_PULL, M_NODE_SCAN, N_PROGRESS};
use higgs::rpc::{self, RpcFrame, RpcRequest, RpcResponse};

use common::tiny_gguf_path;

struct NodeProc(Child);
impl Drop for NodeProc {
    fn drop(&mut self) {
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

async fn hub_endpoint() -> iroh::Endpoint {
    iroh::Endpoint::builder(iroh::endpoint::presets::Minimal)
        .alpns(vec![ALPN.to_vec()])
        .relay_mode(iroh::RelayMode::Disabled)
        .bind()
        .await
        .expect("bind hub endpoint")
}

/// One control RPC (fresh bi stream, single response).
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn node_pulls_a_model_over_http_and_scan_sees_it() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("skipping pull: no tiny GGUF");
        return;
    };
    let bytes = std::fs::read(&gguf).expect("read tiny gguf");
    let node_home = tempfile::tempdir().expect("node home");

    // Local "HuggingFace": serve the GGUF bytes for ANY resolve path.
    let http = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", http.local_addr().unwrap());
    let app = axum::Router::new().fallback(move || {
        let bytes = bytes.clone();
        async move { bytes }
    });
    tokio::spawn(async move { axum::serve(http, app).await.unwrap() });

    // Hub: bind + token + ticket.
    let hub = hub_endpoint().await;
    let hub_id = hub.id().to_string();
    let ticket = EndpointTicket::new(hub.addr()).to_string();
    let mut allow = Allowlist::load(&node_home.path().join("hub-pairings.json")).unwrap();
    let mut tokens = PairingTokens::new();
    let token = tokens.mint(now_ms(), 600_000);

    // Spawn the node, pointing its downloader at the local server.
    let child = Command::new(env!("CARGO_BIN_EXE_higgs"))
        .arg("--node")
        .arg(&ticket)
        .arg(&token)
        .env("HIGGS_HOME", node_home.path())
        .env("HIGGS_IROH_LOCAL", "1")
        .env("HIGGS_HF_ENDPOINT", &endpoint)
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("spawn higgs --node");
    let _node = NodeProc(child);

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

    // M_NODE_PULL on a data stream: collect N_PROGRESS + the final {path}.
    let (mut send, recv) = conn.open_bi().await.expect("data stream");
    let req = RpcRequest {
        jsonrpc: "2.0".into(),
        id: 1,
        method: M_NODE_PULL.into(),
        params: json!({ "request_id": 1, "repo": "higgs-test/pulled", "file": "model.gguf" }),
    };
    send.write_all(format!("{}\n", rpc::encode(&RpcFrame::Request(req))).as_bytes())
        .await
        .unwrap();
    send.finish().unwrap();

    let mut lines = BufReader::new(recv).lines();
    let mut progress_frames = 0;
    let final_resp = loop {
        let line = lines.next_line().await.expect("read").expect("a frame");
        match rpc::decode(&line).expect("decode") {
            RpcFrame::Notification(n) if n.method == N_PROGRESS => progress_frames += 1,
            RpcFrame::Response(r) => break r,
            other => panic!("unexpected pull frame: {other:?}"),
        }
    };
    assert!(final_resp.error.is_none(), "pull ok: {final_resp:?}");
    assert!(
        progress_frames > 0,
        "at least one N_PROGRESS frame streamed"
    );
    let path = final_resp.result.unwrap()["path"]
        .as_str()
        .unwrap()
        .to_string();
    assert!(
        std::path::Path::new(&path).exists(),
        "pulled file exists on disk: {path}"
    );

    // A subsequent scan lists the freshly pulled model.
    let scan = node_rpc(&conn, 2, M_NODE_SCAN, json!({})).await;
    assert!(scan.error.is_none(), "scan ok: {scan:?}");
    let listed = scan.result.unwrap()["models"].as_array().unwrap().clone();
    assert!(
        listed.iter().any(|m| m["id"] == "higgs-test/pulled"),
        "scan lists the pulled model: {listed:?}"
    );
}
