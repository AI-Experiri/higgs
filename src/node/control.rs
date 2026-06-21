//! Node-side CONTROL dispatch: map an inbound `higgs/node/*` RPC request to a
//! `NodeRuntime` op and build the reply (DESIGN-remote.md §5.4a). These never reach a
//! worker's `WorkerState` — they operate on the registry one layer up. The data plane
//! (`M_CHAT`) is dispatched separately (§5.4b, P2 Task 5).

use serde_json::{json, Value};

use crate::diagnostic::HiggsError;
use crate::node::runtime::NodeRuntime;
use crate::node::worker_id::WorkerId;
use crate::remote::{
    NodeLoadParams, NodeLoadResult, WorkerRef, M_NODE_INVENTORY, M_NODE_KILL, M_NODE_LOAD,
    M_NODE_SCAN, M_NODE_STATUS, M_NODE_SYSINFO, M_NODE_UNLOAD, M_NODE_UPDATE,
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
                Ok((worker_id, loaded)) => ok(
                    id,
                    NodeLoadResult {
                        worker_id: worker_id.0,
                        loaded,
                    },
                ),
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
        M_NODE_INVENTORY => match rt.inventory().await {
            Ok(v) => ok_value(id, v),
            Err(e) => err_from(id, &e),
        },
        // #18: the update handshake is recognized but unimplemented — typed HG026 refusal
        // (the real signature-verified updater is a later task; `update` cap is `false`).
        M_NODE_UPDATE => err_from(
            id,
            &HiggsError::UpdateUnsupported {
                detail: "this build ships no updater".into(),
            },
        ),
        other => err(
            id,
            METHOD_NOT_FOUND,
            format!("unknown control method {other}"),
            None,
        ),
    }
}

/// Parse params into `T`; on failure return a closure that builds the INVALID_PARAMS
/// response for a given request id (so the message names the parse error).
fn parse<T: serde::de::DeserializeOwned>(
    req: &RpcRequest,
) -> Result<T, impl FnOnce(u64) -> RpcResponse> {
    serde_json::from_value::<T>(req.params.clone()).map_err(|e| {
        let msg = format!("invalid params: {e}");
        move |id: u64| err(id, INVALID_PARAMS, msg, None)
    })
}

fn ok<T: serde::Serialize>(id: u64, value: T) -> RpcResponse {
    ok_value(
        id,
        serde_json::to_value(value).expect("control results serialize"),
    )
}

fn ok_value(id: u64, result: Value) -> RpcResponse {
    RpcResponse {
        jsonrpc: "2.0".into(),
        id,
        result: Some(result),
        error: None,
    }
}

fn err(id: u64, code: i64, message: String, data: Option<Value>) -> RpcResponse {
    RpcResponse {
        jsonrpc: "2.0".into(),
        id,
        result: None,
        error: Some(RpcError {
            code,
            message,
            data,
        }),
    }
}

/// Map a `HiggsError` to an RpcError carrying its HG code in `data` (mirrors the
/// supervisor's worker-error mapping, so the hub can recover the origin status — the
/// worker's own code when the failure came from the worker).
fn err_from(id: u64, e: &HiggsError) -> RpcResponse {
    err(
        id,
        INTERNAL_ERROR,
        e.to_string(),
        crate::node::worker_origin_code_data(e),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::test_support::fake_runtime as fake_runtime_with_dirs;

    fn fake_runtime() -> NodeRuntime {
        fake_runtime_with_dirs(vec![])
    }

    fn req(id: u64, method: &str, params: Value) -> RpcRequest {
        RpcRequest {
            jsonrpc: "2.0".into(),
            id,
            method: method.into(),
            params,
        }
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
        let resp =
            dispatch_node_control(&rt, req(1, M_NODE_LOAD, json!({ "id": "missing" }))).await;
        let e = resp.error.unwrap();
        assert_eq!(e.code, INTERNAL_ERROR);
        assert_eq!(e.data.unwrap()["code"], "HG002");
    }

    #[tokio::test]
    async fn update_is_recognized_but_refused_hg026() {
        let rt = fake_runtime();
        let resp = dispatch_node_control(&rt, req(1, M_NODE_UPDATE, json!({}))).await;
        let e = resp.error.expect("update refused");
        assert_eq!(e.code, INTERNAL_ERROR);
        assert_eq!(e.data.unwrap()["code"], "HG026", "typed update-unsupported");
    }

    #[tokio::test]
    async fn inventory_dispatch_reports_host_and_no_workers() {
        let rt = fake_runtime(); // empty registry
        let resp = dispatch_node_control(&rt, req(1, M_NODE_INVENTORY, json!({}))).await;
        assert!(resp.error.is_none(), "inventory ok: {resp:?}");
        let inv = resp.result.unwrap();
        assert!(
            inv["hardware"]["cpu_cores"].as_u64().unwrap() > 0,
            "real hw"
        );
        assert!(!inv["os"].as_str().unwrap().is_empty(), "os present");
        assert!(
            inv["workers"].as_array().unwrap().is_empty(),
            "no workers loaded"
        );
    }

    #[tokio::test]
    async fn sysinfo_and_status_dispatch_ok() {
        let rt = fake_runtime();
        // sysinfo is node-level (no worker needed).
        let sysinfo = dispatch_node_control(&rt, req(1, M_NODE_SYSINFO, json!({}))).await;
        assert!(sysinfo.error.is_none());
        assert!(sysinfo.result.unwrap().get("hardware").is_some());

        // status for an unknown worker errors cleanly.
        let status =
            dispatch_node_control(&rt, req(2, M_NODE_STATUS, json!({ "worker_id": 999 }))).await;
        assert!(status.error.is_some());
    }
}
