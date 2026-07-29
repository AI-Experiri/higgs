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

/// Like [`fake_worker_factory`] but STATEFUL: it remembers the model from the last `M_LOAD`
/// (with its `ctx_len`/`gpu_layers`/`threads`) and reports it back in `M_STATUS` under
/// `loaded`, so facade tests that assert the status-mapping path work without llama.cpp.
/// `M_UNLOAD` clears it; `M_CHAT` streams `he`/`llo` then a final response.
pub(crate) fn fake_worker_factory_stateful() -> HalvesFactory {
    Box::new(|_bus, _model| {
        let (sup_end, worker_end) = tokio::io::duplex(64 * 1024);
        let (sup_r, sup_w) = tokio::io::split(sup_end);
        let (wr, mut ww) = tokio::io::split(worker_end);
        let loaded: std::sync::Arc<parking_lot::Mutex<Value>> =
            std::sync::Arc::new(parking_lot::Mutex::new(Value::Null));
        tokio::spawn(async move {
            let mut lines = BufReader::new(wr).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                let Ok(RpcFrame::Request(r)) = rpc::decode(&line) else {
                    continue;
                };
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
                    // Include token counts so callers can verify usage propagation.
                    // LOAD-BEARING SHAPE: the counts here (present, 10/3) vs the plain
                    // fake's reply (absent → 0) are what let
                    // `embed_tests::node_chat_test_bypasses_the_local_first_dispatch`
                    // tell a LOCAL answer from a REMOTE relay. If you change either
                    // fake's usage fields, that test's self-guard fails first — keep
                    // the two shapes distinguishable.
                    let resp = RpcResponse {
                        jsonrpc: "2.0".into(),
                        id: r.id,
                        result: Some(json!({
                            "content": "hello",
                            "finish_reason": "stop",
                            "prompt_tokens": 10,
                            "completion_tokens": 3,
                        })),
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
                        // Record the loaded model so M_STATUS can report it.
                        *loaded.lock() = json!({
                            "id": r.params.get("id").cloned().unwrap_or(Value::Null),
                            "ctx_len": r.params.get("ctx_len").cloned().unwrap_or(Value::Null),
                            "gpu_layers": r.params.get("gpu_layers").cloned().unwrap_or(Value::Null),
                            "threads": r.params.get("threads").cloned().unwrap_or(Value::Null),
                        });
                        json!({ "id": r.params.get("id").cloned().unwrap_or(Value::Null) })
                    }
                    crate::worker::M_STATUS => json!({ "loaded": loaded.lock().clone() }),
                    crate::worker::M_UNLOAD => {
                        *loaded.lock() = Value::Null;
                        json!({})
                    }
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

/// A stateful fake whose first TWO `M_LOAD`s FAIL with an OOM-classified worker error
/// (`HG004` + "out of memory") and every later load SUCCEEDS. The G5 OOM degrade-retry
/// ladder therefore walks past attempt-0 (seed) and the plain-retry rung and loads on the
/// KV rung — exercising the DEGRADE path. `attempts` is SHARED across worker spawns (each
/// ladder attempt is a fresh spawn), so it counts TOTAL load attempts, not per-worker.
fn oom_twice_factory(attempts: Arc<std::sync::atomic::AtomicU32>) -> HalvesFactory {
    Box::new(move |_bus, _model| {
        let attempts = attempts.clone();
        let (sup_end, worker_end) = tokio::io::duplex(64 * 1024);
        let (sup_r, sup_w) = tokio::io::split(sup_end);
        let (wr, mut ww) = tokio::io::split(worker_end);
        let loaded: Arc<parking_lot::Mutex<Value>> = Arc::new(parking_lot::Mutex::new(Value::Null));
        tokio::spawn(async move {
            let mut lines = BufReader::new(wr).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                let Ok(RpcFrame::Request(r)) = rpc::decode(&line) else {
                    continue;
                };
                let resp = if r.method == crate::worker::M_LOAD {
                    let n = attempts.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
                    if n <= 2 {
                        // OOM-classified failure (worker_code HG004 + "out of memory") →
                        // the ladder degrades to the next rung.
                        RpcResponse {
                            jsonrpc: "2.0".into(),
                            id: r.id,
                            result: None,
                            error: Some(crate::rpc::RpcError {
                                code: -32000,
                                message: "ggml_backend cuda error: out of memory".into(),
                                data: Some(json!({ "code": "HG004" })),
                            }),
                        }
                    } else {
                        *loaded.lock() = json!({
                            "id": r.params.get("id").cloned().unwrap_or(Value::Null),
                            "ctx_len": r.params.get("ctx_len").cloned().unwrap_or(Value::Null),
                            "gpu_layers": r.params.get("gpu_layers").cloned().unwrap_or(Value::Null),
                            "threads": r.params.get("threads").cloned().unwrap_or(Value::Null),
                        });
                        RpcResponse {
                            jsonrpc: "2.0".into(),
                            id: r.id,
                            result: Some(
                                json!({ "id": r.params.get("id").cloned().unwrap_or(Value::Null) }),
                            ),
                            error: None,
                        }
                    }
                } else {
                    let result = match r.method.as_str() {
                        crate::worker::M_STATUS => json!({ "loaded": loaded.lock().clone() }),
                        crate::worker::M_UNLOAD => {
                            *loaded.lock() = Value::Null;
                            json!({})
                        }
                        crate::worker::M_SYSINFO => json!({ "gpus": [] }),
                        crate::worker::M_SHUTDOWN => break,
                        _ => json!({}),
                    };
                    RpcResponse {
                        jsonrpc: "2.0".into(),
                        id: r.id,
                        result: Some(result),
                        error: None,
                    }
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

/// A `NodeRuntime` whose loads OOM-then-degrade ([`oom_twice_factory`]) — the shared
/// attempt counter lives here so it survives across worker spawns AND spawner calls.
pub(crate) fn fake_runtime_oom_twice(lmstudio_dirs: Vec<PathBuf>) -> NodeRuntime {
    let attempts = Arc::new(std::sync::atomic::AtomicU32::new(0));
    NodeRuntime::with_spawner(
        NodeConfig {
            bus: Arc::new(LogBus::new()),
            lmstudio_dirs,
            hf_dirs: vec![],
            ollama_dirs: vec![],
            idle_ttl: crate::node::runtime::DEFAULT_IDLE_TTL,
        },
        Arc::new(move |_bus| Supervisor::with_factory(oom_twice_factory(attempts.clone()))),
    )
}

/// A `NodeRuntime` whose workers are fakes (no llama.cpp), scanning `dirs` for models. Uses
/// the default (60-min) idle TTL so fast tests never trip the reaper.
pub(crate) fn fake_runtime(lmstudio_dirs: Vec<PathBuf>) -> NodeRuntime {
    fake_runtime_with_idle_ttl(lmstudio_dirs, crate::node::runtime::DEFAULT_IDLE_TTL)
}

/// A `NodeRuntime` backed by the STATEFUL fake worker ([`fake_worker_factory_stateful`]):
/// M_LOAD records the model, M_STATUS reports it, M_UNLOAD clears it, and M_CHAT streams
/// `he`/`llo` plus token counts. Used by the engine (`Higgs`) facade + serve-layer tests to
/// exercise load/status/chat without llama.cpp.
pub(crate) fn fake_runtime_stateful(lmstudio_dirs: Vec<PathBuf>) -> NodeRuntime {
    NodeRuntime::with_spawner(
        NodeConfig {
            bus: Arc::new(LogBus::new()),
            lmstudio_dirs,
            hf_dirs: vec![],
            ollama_dirs: vec![],
            idle_ttl: crate::node::runtime::DEFAULT_IDLE_TTL,
        },
        Arc::new(|_bus| Supervisor::with_factory(fake_worker_factory_stateful())),
    )
}

/// Like [`fake_runtime`] but with an explicit idle TTL — lets a test drive the node's idle
/// reaper deterministically with a tiny TTL.
pub(crate) fn fake_runtime_with_idle_ttl(
    lmstudio_dirs: Vec<PathBuf>,
    idle_ttl: std::time::Duration,
) -> NodeRuntime {
    NodeRuntime::with_spawner(
        NodeConfig {
            bus: Arc::new(LogBus::new()),
            lmstudio_dirs,
            hf_dirs: vec![],
            ollama_dirs: vec![],
            idle_ttl,
        },
        Arc::new(|_bus| Supervisor::with_factory(fake_worker_factory())),
    )
}

/// A self-responding worker that ERRORS every `M_LOAD` — so a Supervisor built on it spawns
/// (start succeeds) but the load RPC fails, exercising the node's post-spawn-failure reap
/// path. All other methods behave like [`fake_worker_factory`].
pub(crate) fn fake_load_failing_factory() -> HalvesFactory {
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
                if r.method == crate::worker::M_SHUTDOWN {
                    break;
                }
                let resp = if r.method == crate::worker::M_LOAD {
                    RpcResponse {
                        jsonrpc: "2.0".into(),
                        id: r.id,
                        result: None,
                        error: Some(crate::rpc::RpcError {
                            code: -32000,
                            message: "[HG017] fake load failure".into(),
                            // A real worker carries its diagnostic code in JSON-RPC `data`
                            // (supervisor's send_request reads `data.code`), so the boundary maps
                            // HG017 → 503. Mirror that so the fake is faithful to a real load fault.
                            data: Some(json!({ "code": "HG017" })),
                        }),
                    }
                } else {
                    RpcResponse {
                        jsonrpc: "2.0".into(),
                        id: r.id,
                        result: Some(json!({})),
                        error: None,
                    }
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

/// A `NodeRuntime` whose workers spawn but fail every `M_LOAD` ([`fake_load_failing_factory`]).
pub(crate) fn fake_runtime_load_fails(lmstudio_dirs: Vec<PathBuf>) -> NodeRuntime {
    NodeRuntime::with_spawner(
        NodeConfig {
            bus: Arc::new(LogBus::new()),
            lmstudio_dirs,
            hf_dirs: vec![],
            ollama_dirs: vec![],
            idle_ttl: crate::node::runtime::DEFAULT_IDLE_TTL,
        },
        Arc::new(|_bus| Supervisor::with_factory(fake_load_failing_factory())),
    )
}

/// A self-responding worker that LOADS successfully (so the node lists the model as
/// resident) but ERRORS every `M_STATUS` — simulating a worker BUSY mid-generation that
/// cannot answer the status probe. Exercises `Higgs::local_loaded_info`'s resident-but-
/// unreachable fallback (a permissive stub, never "not served").
pub(crate) fn fake_status_failing_factory() -> HalvesFactory {
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
                if r.method == crate::worker::M_SHUTDOWN {
                    break;
                }
                let resp = if r.method == crate::worker::M_STATUS {
                    RpcResponse {
                        jsonrpc: "2.0".into(),
                        id: r.id,
                        result: None,
                        error: Some(crate::rpc::RpcError {
                            code: -32000,
                            message: "fake status failure (worker busy mid-generation)".into(),
                            data: None,
                        }),
                    }
                } else if r.method == crate::worker::M_LOAD {
                    // Echo the id so the node records the worker as resident.
                    RpcResponse {
                        jsonrpc: "2.0".into(),
                        id: r.id,
                        result: Some(json!({
                            "id": r.params.get("id").cloned().unwrap_or(Value::Null),
                        })),
                        error: None,
                    }
                } else {
                    RpcResponse {
                        jsonrpc: "2.0".into(),
                        id: r.id,
                        result: Some(json!({})),
                        error: None,
                    }
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

/// A `NodeRuntime` whose workers LOAD but fail every `M_STATUS` ([`fake_status_failing_factory`]).
pub(crate) fn fake_runtime_status_fails(lmstudio_dirs: Vec<PathBuf>) -> NodeRuntime {
    NodeRuntime::with_spawner(
        NodeConfig {
            bus: Arc::new(LogBus::new()),
            lmstudio_dirs,
            hf_dirs: vec![],
            ollama_dirs: vec![],
            idle_ttl: crate::node::runtime::DEFAULT_IDLE_TTL,
        },
        Arc::new(|_bus| Supervisor::with_factory(fake_status_failing_factory())),
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
