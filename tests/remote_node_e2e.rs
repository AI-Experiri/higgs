//! Remote-node end-to-end integration test over a REAL spawned `higgs` process.
//!
//! This is the full remote workflow, black-box: the test acts as the **hub** (in-process,
//! using higgs's library types) and spawns a separate **`higgs --node`** OS process that
//! loads a real GGUF in a real llama.cpp child. Over a real (relay-disabled, hermetic)
//! iroh link it then exercises: pair → load → sysinfo (real cpu/mem/gpu) → chat (streamed
//! tokens) → status. Nothing is faked — if iroh, the handshake, control dispatch, model
//! loading, or the chat relay regress, this test fails.
//!
//! Skips when no tiny GGUF is available (same convention as the other model-backed tests).

mod common;

use std::io::Write as _;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use iroh::endpoint::Connection;
use iroh_tickets::endpoint::EndpointTicket;
use serde_json::{json, Value};

use higgs::auth::{Allowlist, PairingTokens};
use higgs::node::{gate_connection, GateOutcome, HELLO_DEADLINE};
use higgs::remote::{ALPN, M_NODE_LOAD, M_NODE_STATUS, M_NODE_SYSINFO};
use higgs::rpc::{self, RpcFrame, RpcRequest, RpcResponse};
use higgs::worker::{M_CHAT, N_CHAT_CHUNK};

use common::{stage_tiny_model, tiny_gguf_path, TINY_MODEL_ID};

/// Stops the spawned node process on drop so a failed assertion never leaks it.
struct NodeProc(Child);
impl Drop for NodeProc {
    fn drop(&mut self) {
        // SIGTERM (not SIGKILL): the `--node` daemon shuts down gracefully — drains its
        // workers and exits — which is also what flushes the spawned process's coverage
        // profile under llvm-cov instrumentation (a hard kill would discard it).
        // SAFETY: a plain kill(2) on our own child's pid.
        unsafe {
            libc::kill(self.0.id() as libc::pid_t, libc::SIGTERM);
        }
        let _ = self.0.wait();
    }
}

fn now_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis() as u64
}

/// Bind the test's hub endpoint: relay-disabled + local, matching the node's
/// `HIGGS_IROH_LOCAL` mode so the link is hermetic (no public relay/DNS).
async fn hub_endpoint() -> iroh::Endpoint {
    iroh::Endpoint::builder(iroh::endpoint::presets::Minimal)
        .alpns(vec![ALPN.to_vec()])
        .relay_mode(iroh::RelayMode::Disabled)
        .bind()
        .await
        .expect("bind hub endpoint")
}

/// Open a fresh control stream, send one `higgs/node/*` request, read the single response.
async fn node_rpc(conn: &Connection, id: u64, method: &str, params: Value) -> RpcResponse {
    use tokio::io::{AsyncBufReadExt, BufReader};
    let (mut send, recv) = conn.open_bi().await.expect("open control stream");
    let req = RpcRequest { jsonrpc: "2.0".into(), id, method: method.into(), params };
    send.write_all(format!("{}\n", rpc::encode(&RpcFrame::Request(req))).as_bytes())
        .await
        .expect("write request");
    send.finish().expect("finish");
    let mut lines = BufReader::new(recv).lines();
    let line = lines.next_line().await.expect("read").expect("a response line");
    match rpc::decode(&line).expect("decode") {
        RpcFrame::Response(r) => r,
        other => panic!("expected response, got {other:?}"),
    }
}

/// Open a data stream, send `M_CHAT`, collect streamed `N_CHAT_CHUNK` deltas + final.
async fn node_chat(conn: &Connection, worker_id: u64, request_id: u64) -> (Vec<String>, RpcResponse) {
    use tokio::io::{AsyncBufReadExt, BufReader};
    let (mut send, recv) = conn.open_bi().await.expect("open data stream");
    let req = RpcRequest {
        jsonrpc: "2.0".into(),
        id: request_id,
        method: M_CHAT.into(),
        params: json!({
            "worker_id": worker_id,
            "request_id": request_id,
            "model": TINY_MODEL_ID,
            "messages_json": "[{\"role\":\"user\",\"content\":\"Once upon a time\"}]",
            "max_tokens": 8,
            "temperature": 0.0,
        }),
    };
    send.write_all(format!("{}\n", rpc::encode(&RpcFrame::Request(req))).as_bytes())
        .await
        .expect("write chat");
    send.finish().expect("finish");

    let mut lines = BufReader::new(recv).lines();
    let mut chunks = Vec::new();
    loop {
        let line = lines.next_line().await.expect("read").expect("a frame");
        match rpc::decode(&line).expect("decode") {
            RpcFrame::Notification(n) if n.method == N_CHAT_CHUNK => {
                if let Some(d) = n.params.get("delta").and_then(|v| v.as_str()) {
                    chunks.push(d.to_string());
                }
            }
            RpcFrame::Response(r) => return (chunks, r),
            other => panic!("unexpected chat frame: {other:?}"),
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn remote_node_full_workflow_over_iroh() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("skipping remote_node_e2e: no tiny GGUF (set HIGGS_TEST_GGUF)");
        return;
    };
    let scan_root = stage_tiny_model(&gguf);
    let node_home = tempfile::tempdir().expect("node home");

    // Hub: bind, mint a token, build a pairing ticket from our direct addresses.
    let hub = hub_endpoint().await;
    let hub_id = hub.id().to_string();
    let ticket = EndpointTicket::new(hub.addr()).to_string();
    let mut allow = Allowlist::load(&node_home.path().join("hub-pairings.json")).unwrap();
    let mut tokens = PairingTokens::new();
    let token = tokens.mint(now_ms(), 600_000);

    // Spawn a REAL `higgs --node` process (hermetic iroh, real model dir).
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

    // Hub: accept the node's dial and gate it (token admits + allowlists).
    let incoming = tokio::time::timeout(Duration::from_secs(30), hub.accept())
        .await
        .expect("node dialed within 30s")
        .expect("incoming");
    let conn = incoming.await.expect("connection");
    let outcome =
        gate_connection(&conn, &mut allow, &mut tokens, now_ms(), hub_id, Some("test".into()), HELLO_DEADLINE)
            .await;
    assert!(matches!(outcome, GateOutcome::Admitted { .. }), "node admitted: {outcome:?}");

    // 1. Load a real model on the node (real llama.cpp child).
    let load = node_rpc(&conn, 1, M_NODE_LOAD, json!({ "id": TINY_MODEL_ID })).await;
    assert!(load.error.is_none(), "load ok: {load:?}");
    let worker_id = load.result.unwrap()["worker_id"].as_u64().expect("worker_id");

    // 2. sysinfo: real hardware params (cpu cores, ram) extracted from the node.
    let sys = node_rpc(&conn, 2, M_NODE_SYSINFO, json!({})).await;
    assert!(sys.error.is_none(), "sysinfo ok: {sys:?}");
    let sys = sys.result.unwrap();
    assert!(sys["hardware"]["cpu_cores"].as_u64().unwrap() > 0, "real cpu cores");
    assert!(sys["hardware"]["ram_total_bytes"].as_u64().unwrap() > 0, "real ram");
    assert!(sys["hardware"].get("gpus").is_some(), "gpu list present");
    assert_eq!(sys["runtime"]["engine"], "llama.cpp");

    // 3. status: the worker reports the resident model.
    let status = node_rpc(&conn, 3, M_NODE_STATUS, json!({ "worker_id": worker_id })).await;
    assert!(status.error.is_none(), "status ok: {status:?}");

    // 4. chat session: real generation streamed back over iroh.
    let (chunks, final_resp) = node_chat(&conn, worker_id, 7).await;
    assert!(final_resp.error.is_none(), "chat ok: {final_resp:?}");
    let result = final_resp.result.unwrap();
    assert!(result.get("content").is_some(), "chat result has content");
    // Either streamed chunks arrived, or the content is non-empty (both prove generation).
    let content = result["content"].as_str().unwrap_or("");
    assert!(!chunks.is_empty() || !content.is_empty(), "tokens were generated");
    let _ = std::io::stderr().flush();
}
