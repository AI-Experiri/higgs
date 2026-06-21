//! Shared `#[cfg(test)]` helpers for node tests: a self-responding fake worker, a
//! fake-backed `NodeRuntime`, a dummy on-disk model, a relay-disabled local endpoint,
//! and a one-shot control-RPC client. Keeps runtime/control/e2e tests duplication-free.

use std::path::PathBuf;
use std::sync::Arc;

use serde_json::{json, Value};
use tempfile::TempDir;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

use crate::log_bus::LogBus;
use crate::node::runtime::{NodeConfig, NodeRuntime};
use crate::remote::ALPN;
use crate::rpc::{self, RpcFrame, RpcRequest, RpcResponse};
use crate::supervisor::{HalvesFactory, Supervisor, WorkerHalves};

/// A self-responding in-memory worker: reads NDJSON requests and replies with valid
/// responses, so a Supervisor built on it behaves like a (model-less) real worker.
pub(crate) fn fake_worker_factory() -> HalvesFactory {
    Box::new(|_bus, _model| {
        let (sup_end, worker_end) = tokio::io::duplex(64 * 1024);
        let (sup_r, sup_w) = tokio::io::split(sup_end);
        let (wr, mut ww) = tokio::io::split(worker_end);
        tokio::spawn(async move {
            let mut lines = BufReader::new(wr).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                let Ok(RpcFrame::Request(r)) = rpc::decode(&line) else {
                    continue;
                };
                // M_CHAT streams N_CHAT_CHUNK notifications (echoing request_id) then a
                // final response, mirroring the real worker — so the chat relay/transport
                // can be exercised without llama.cpp.
                if r.method == crate::worker::M_CHAT {
                    let request_id = r.params.get("request_id").cloned().unwrap_or(Value::Null);
                    for delta in ["he", "llo"] {
                        let note = crate::rpc::RpcNotification {
                            jsonrpc: "2.0".into(),
                            method: crate::worker::N_CHAT_CHUNK.into(),
                            params: json!({ "request_id": request_id, "delta": delta }),
                        };
                        let line = format!("{}\n", rpc::encode(&RpcFrame::Notification(note)));
                        if ww.write_all(line.as_bytes()).await.is_err() {
                            return;
                        }
                    }
                    let resp = RpcResponse {
                        jsonrpc: "2.0".into(),
                        id: r.id,
                        result: Some(json!({ "content": "hello", "finish_reason": "stop" })),
                        error: None,
                    };
                    let line = format!("{}\n", rpc::encode(&RpcFrame::Response(resp)));
                    if ww.write_all(line.as_bytes()).await.is_err() {
                        return;
                    }
                    continue;
                }
                let result = match r.method.as_str() {
                    crate::worker::M_LOAD => {
                        json!({ "id": r.params.get("id").cloned().unwrap_or(Value::Null) })
                    }
                    crate::worker::M_STATUS => json!({ "loaded": Value::Null }),
                    crate::worker::M_SYSINFO => json!({ "gpus": [] }),
                    crate::worker::M_SHUTDOWN => break,
                    _ => json!({}),
                };
                let resp = RpcResponse {
                    jsonrpc: "2.0".into(),
                    id: r.id,
                    result: Some(result),
                    error: None,
                };
                let line = format!("{}\n", rpc::encode(&RpcFrame::Response(resp)));
                if ww.write_all(line.as_bytes()).await.is_err() {
                    break;
                }
            }
        });
        Ok(WorkerHalves {
            write: Box::new(sup_w),
            read: Box::new(sup_r),
            proc: None,
        })
    })
}

/// A `NodeRuntime` whose workers are fakes (no llama.cpp), scanning `dirs` for models.
pub(crate) fn fake_runtime(lmstudio_dirs: Vec<PathBuf>) -> NodeRuntime {
    NodeRuntime::with_spawner(
        NodeConfig {
            bus: Arc::new(LogBus::new()),
            lmstudio_dirs,
            hf_dirs: vec![],
            ollama_dirs: vec![],
        },
        Box::new(|_bus| Supervisor::with_factory(fake_worker_factory())),
    )
}

/// Stage a dummy `<root>/<id>/m.gguf` so `ModelStore::scan` catalogs it (the GGUF header
/// is read best-effort, so a non-GGUF file still catalogs — enough for resolution tests).
/// Returns the temp scan root (keep it alive) and the model id.
pub(crate) fn stage_dummy_model(id: &str) -> (TempDir, String) {
    let dir = TempDir::new().expect("staging dir");
    let model_dir = dir.path().join(id);
    std::fs::create_dir_all(&model_dir).expect("model dir");
    std::fs::write(model_dir.join("m.gguf"), b"GGUF\x00 dummy").expect("write dummy gguf");
    (dir, id.to_string())
}

/// Bind a local-only endpoint (relay disabled) for in-process iroh tests.
pub(crate) async fn local_endpoint() -> iroh::Endpoint {
    iroh::Endpoint::builder(iroh::endpoint::presets::Minimal)
        .alpns(vec![ALPN.to_vec()])
        .relay_mode(iroh::RelayMode::Disabled)
        .bind()
        .await
        .expect("bind local endpoint")
}

/// Open a fresh control bi-stream on `conn`, send one `higgs/node/*` request, and read
/// the single response — the hub's view of a control RPC.
pub(crate) async fn node_rpc(
    conn: &iroh::endpoint::Connection,
    id: u64,
    method: &str,
    params: Value,
) -> RpcResponse {
    let (mut send, recv) = conn.open_bi().await.expect("open control stream");
    let req = RpcRequest {
        jsonrpc: "2.0".into(),
        id,
        method: method.into(),
        params,
    };
    send.write_all(format!("{}\n", rpc::encode(&RpcFrame::Request(req))).as_bytes())
        .await
        .expect("write request");
    send.finish().expect("finish");
    let mut lines = BufReader::new(recv).lines();
    let line = lines
        .next_line()
        .await
        .expect("read")
        .expect("a response line");
    match rpc::decode(&line).expect("decode response") {
        RpcFrame::Response(r) => r,
        other => panic!("expected response, got {other:?}"),
    }
}
