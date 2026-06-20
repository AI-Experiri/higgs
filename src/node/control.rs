//! Node-side CONTROL dispatch: map an inbound `higgs/node/*` RPC request to a
//! `NodeRuntime` op and build the reply (DESIGN-remote.md §5.4a). These never reach a
//! worker's `WorkerState` — they operate on the registry one layer up. The data plane
//! (`M_CHAT`) is dispatched separately (§5.4b, P2 Task 5).

use serde_json::{json, Value};

use crate::diagnostic::HiggsError;
use crate::node::runtime::NodeRuntime;
use crate::node::worker_id::WorkerId;
use crate::remote::{
    NodeLoadParams, NodeLoadResult, WorkerRef, M_NODE_KILL, M_NODE_LOAD, M_NODE_SCAN,
    M_NODE_STATUS, M_NODE_SYSINFO, M_NODE_UNLOAD,
};
use crate::rpc::{RpcError, RpcRequest, RpcResponse};

/// JSON-RPC error codes used by the control plane (mirrors the worker wire).
const METHOD_NOT_FOUND: i64 = -32601;
const INVALID_PARAMS: i64 = -32602;
const INTERNAL_ERROR: i64 = -32000;

/// Dispatch one `higgs/node/*` control request against the node runtime.
pub async fn dispatch_node_control(rt: &NodeRuntime, req: RpcRequest) -> RpcResponse {
    let id = req.id;
    match req.method.as_str() {
        M_NODE_LOAD => match parse::<NodeLoadParams>(&req) {
            Ok(params) => match rt.load(params).await {
                Ok((worker_id, loaded)) => ok(id, NodeLoadResult { worker_id: worker_id.0, loaded }),
                Err(e) => err_from(id, &e),
            },
            Err(resp) => resp(id),
        },
        M_NODE_UNLOAD => match parse::<WorkerRef>(&req) {
            Ok(w) => match rt.unload(WorkerId(w.worker_id)).await {
                Ok(()) => ok_value(id, json!({})),
                Err(e) => err_from(id, &e),
            },
            Err(resp) => resp(id),
        },
        M_NODE_KILL => match parse::<WorkerRef>(&req) {
            Ok(w) => match rt.kill(WorkerId(w.worker_id)).await {
                Ok(()) => ok_value(id, json!({})),
                Err(e) => err_from(id, &e),
            },
            Err(resp) => resp(id),
        },
        M_NODE_STATUS => match parse::<WorkerRef>(&req) {
            Ok(w) => match rt.status(WorkerId(w.worker_id)).await {
                Ok(v) => ok_value(id, v),
                Err(e) => err_from(id, &e),
            },
            Err(resp) => resp(id),
        },
        M_NODE_SYSINFO => match rt.sysinfo().await {
            Ok(v) => ok_value(id, v),
            Err(e) => err_from(id, &e),
        },
        M_NODE_SCAN => match rt.scan().await {
            Ok(v) => ok_value(id, v),
            Err(e) => err_from(id, &e),
        },
        other => err(id, METHOD_NOT_FOUND, format!("unknown control method {other}"), None),
    }
}

/// Parse params into `T`; on failure return a closure that builds the INVALID_PARAMS
/// response for a given request id (so the message names the parse error).
fn parse<T: serde::de::DeserializeOwned>(req: &RpcRequest) -> Result<T, impl FnOnce(u64) -> RpcResponse> {
    serde_json::from_value::<T>(req.params.clone()).map_err(|e| {
        let msg = format!("invalid params: {e}");
        move |id: u64| err(id, INVALID_PARAMS, msg, None)
    })
}

fn ok<T: serde::Serialize>(id: u64, value: T) -> RpcResponse {
    ok_value(id, serde_json::to_value(value).expect("control results serialize"))
}

fn ok_value(id: u64, result: Value) -> RpcResponse {
    RpcResponse { jsonrpc: "2.0".into(), id, result: Some(result), error: None }
}

fn err(id: u64, code: i64, message: String, data: Option<Value>) -> RpcResponse {
    RpcResponse {
        jsonrpc: "2.0".into(),
        id,
        result: None,
        error: Some(RpcError { code, message, data }),
    }
}

/// Map a `HiggsError` to an RpcError carrying its HG code in `data` (mirrors the
/// supervisor's worker-error mapping, so the hub can recover the origin status).
fn err_from(id: u64, e: &HiggsError) -> RpcResponse {
    use miette::Diagnostic;
    let code = e.code().map(|c| c.to_string());
    err(id, INTERNAL_ERROR, e.to_string(), code.map(|c| json!({ "code": c })))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::log_bus::LogBus;
    use crate::node::runtime::{NodeConfig, NodeRuntime};
    use crate::rpc::{self, RpcFrame};
    use crate::supervisor::{Supervisor, WorkerHalves};
    use crate::worker::{M_LOAD, M_STATUS, M_SYSINFO};
    use std::sync::Arc;
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

    fn fake_worker_factory() -> crate::supervisor::HalvesFactory {
        Box::new(|_bus, _model| {
            let (sup_end, worker_end) = tokio::io::duplex(64 * 1024);
            let (sup_r, sup_w) = tokio::io::split(sup_end);
            let (wr, mut ww) = tokio::io::split(worker_end);
            tokio::spawn(async move {
                let mut lines = BufReader::new(wr).lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    let Ok(RpcFrame::Request(r)) = rpc::decode(&line) else { continue };
                    let result = match r.method.as_str() {
                        M_LOAD => json!({ "id": r.params.get("id").cloned().unwrap_or(Value::Null) }),
                        M_STATUS => json!({ "loaded": Value::Null }),
                        M_SYSINFO => json!({ "gpus": [] }),
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
            Ok(WorkerHalves { write: Box::new(sup_w), read: Box::new(sup_r), proc: None })
        })
    }

    fn fake_runtime() -> NodeRuntime {
        NodeRuntime::with_spawner(
            NodeConfig {
                bus: Arc::new(LogBus::new()),
                lmstudio_dirs: vec![],
                hf_dirs: vec![],
                ollama_dirs: vec![],
            },
            Box::new(|_bus| Supervisor::with_factory(fake_worker_factory())),
        )
    }

    fn req(id: u64, method: &str, params: Value) -> RpcRequest {
        RpcRequest { jsonrpc: "2.0".into(), id, method: method.into(), params }
    }

    #[tokio::test]
    async fn unknown_method_is_method_not_found() {
        let rt = fake_runtime();
        let resp = dispatch_node_control(&rt, req(1, "higgs/node/bogus", json!({}))).await;
        assert_eq!(resp.error.unwrap().code, METHOD_NOT_FOUND);
    }

    #[tokio::test]
    async fn load_with_unhonorable_param_is_invalid_params() {
        let rt = fake_runtime();
        // idle_ttl_minutes is rejected by deny_unknown_fields → -32602.
        let resp = dispatch_node_control(
            &rt,
            req(1, M_NODE_LOAD, json!({ "id": "m", "idle_ttl_minutes": 5 })),
        )
        .await;
        assert_eq!(resp.error.unwrap().code, INVALID_PARAMS);
    }

    #[tokio::test]
    async fn load_missing_model_maps_hg002() {
        let rt = fake_runtime(); // no model roots → ModelNotFound
        let resp = dispatch_node_control(&rt, req(1, M_NODE_LOAD, json!({ "id": "missing" }))).await;
        let e = resp.error.unwrap();
        assert_eq!(e.code, INTERNAL_ERROR);
        assert_eq!(e.data.unwrap()["code"], "HG002");
    }

    #[tokio::test]
    async fn sysinfo_and_status_dispatch_ok() {
        let rt = fake_runtime();
        // sysinfo is node-level (no worker needed).
        let sysinfo = dispatch_node_control(&rt, req(1, M_NODE_SYSINFO, json!({}))).await;
        assert!(sysinfo.error.is_none());
        assert!(sysinfo.result.unwrap().get("gpus").is_some());

        // status for an unknown worker errors cleanly.
        let status = dispatch_node_control(&rt, req(2, M_NODE_STATUS, json!({ "worker_id": 999 }))).await;
        assert!(status.error.is_some());
    }
}
