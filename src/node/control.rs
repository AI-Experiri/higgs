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
/// `-32601` (method-not-found) is now produced by the shared `rpc::method_not_found`
/// helper (which also rides the HG037 code); the named constant remains only for the
/// test that asserts the wire code.
#[cfg(test)]
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
        // Unknown method = a protocol skew (HG037, → 501); reuse the shared helper's
        // message + data.code, keeping the -32601 numeric code.
        other => {
            let mnf = crate::rpc::method_not_found("node", other);
            err(id, mnf.code, mnf.message, mnf.data)
        }
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
#[path = "control_tests.rs"]
mod tests;
