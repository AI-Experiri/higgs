//! Newline-delimited JSON-RPC 2.0 frames for supervisor↔worker stdio (MCP wire).
//! One JSON object per line; requests carry ids, notifications do not.
//!
//! Public for the worker round-trip integration test; internal wire detail, not a stability surface.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::diagnostic::HiggsError;

/// A JSON-RPC request (has an id; expects a response).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RpcRequest {
    /// JSON-RPC protocol version — always `"2.0"`. Validated softly on decode
    /// (see [`decode`]): a mismatch is logged, not rejected, because both peers
    /// are the same binary and dropping the frame would wedge the RPC loop.
    pub jsonrpc: String,
    /// Request correlation id. The supervisor's reader matches a response's
    /// `id` back to the waiting caller (`pending[id]`); for M_CHAT the same id
    /// also flows in `params.request_id` so chat chunks route to the right sink.
    pub id: u64,
    pub method: String,
    #[serde(default, skip_serializing_if = "Value::is_null")]
    pub params: Value,
}

/// A JSON-RPC response carrying either `result` or `error`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RpcResponse {
    /// JSON-RPC protocol version — always `"2.0"` (softly validated; see
    /// [`RpcRequest::jsonrpc`]).
    pub jsonrpc: String,
    /// Correlation id echoed from the originating [`RpcRequest::id`].
    pub id: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<RpcError>,
}

/// A JSON-RPC error object.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RpcError {
    pub code: i64,
    pub message: String,
    /// Structured error payload (JSON-RPC `data`). The worker carries the
    /// origin diagnostic code here (`{"code":"HG005"}`) so the supervisor can
    /// map a worker error to the right HTTP status instead of collapsing every
    /// worker failure to a generic 500. `None` when no structured data applies.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

/// A JSON-RPC notification (no id; fire-and-forget — used for chat chunks).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RpcNotification {
    /// JSON-RPC protocol version — always `"2.0"` (softly validated; see
    /// [`RpcRequest::jsonrpc`]).
    pub jsonrpc: String,
    pub method: String,
    #[serde(default, skip_serializing_if = "Value::is_null")]
    pub params: Value,
}

/// Any inbound frame: discriminated by presence of `id` / `method`.
#[derive(Debug, Clone, PartialEq)]
pub enum RpcFrame {
    Request(RpcRequest),
    Response(RpcResponse),
    Notification(RpcNotification),
}

/// Build a JSON-RPC "method not found" (`-32601`) error that carries the `HG037`
/// origin code in `data.code`, so the receiving supervisor/hub/HTTP boundary can
/// classify it (→ 501) instead of treating it as a transport fault. Shared by the
/// worker, node-control, and hub dispatchers — `endpoint` names which side rejected
/// the call (`worker`/`node`/`hub`).
pub fn method_not_found(endpoint: &str, method: &str) -> RpcError {
    use miette::Diagnostic;
    let e = HiggsError::RpcMethodNotFound {
        endpoint: endpoint.to_owned(),
        method: method.to_owned(),
    };
    RpcError {
        code: -32601,
        message: e.to_string(),
        data: e
            .code()
            .map(|c| serde_json::json!({ "code": c.to_string() })),
    }
}

/// Encode one frame as a single NDJSON line (no trailing newline included).
pub fn encode(frame: &RpcFrame) -> String {
    match frame {
        RpcFrame::Request(r) => serde_json::to_string(r),
        RpcFrame::Response(r) => serde_json::to_string(r),
        RpcFrame::Notification(n) => serde_json::to_string(n),
    }
    .expect("rpc frames are always serializable")
}

/// Decode one NDJSON line into a frame.
///
/// Fails with [HG008] when the line is not valid JSON-RPC 2.0.
///
/// The `jsonrpc` version is validated **softly**: a value other than `"2.0"`
/// is logged at `warn` but the frame is still decoded. Both peers are the same
/// binary, so a version mismatch is a bug to surface, not a wire condition to
/// hard-reject — dropping the frame would wedge the RPC loop (no response ever
/// drains the caller's `pending` entry).
pub fn decode(line: &str) -> Result<RpcFrame, HiggsError> {
    let v: Value = serde_json::from_str(line).map_err(|e| HiggsError::RpcDecode {
        detail: format!("not json: {e}"),
    })?;
    if let Some(ver) = v.get("jsonrpc").and_then(Value::as_str) {
        if ver != "2.0" {
            tracing::warn!(
                jsonrpc = ver,
                "higgs: unexpected JSON-RPC version (expected 2.0)"
            );
        }
    }
    let has_id = v.get("id").is_some();
    let has_method = v.get("method").is_some();
    let parse = |detail: &str| HiggsError::RpcDecode {
        detail: detail.to_string(),
    };
    match (has_id, has_method) {
        (true, true) => serde_json::from_value(v)
            .map(RpcFrame::Request)
            .map_err(|e| parse(&e.to_string())),
        (true, false) => serde_json::from_value(v)
            .map(RpcFrame::Response)
            .map_err(|e| parse(&e.to_string())),
        (false, true) => serde_json::from_value(v)
            .map(RpcFrame::Notification)
            .map_err(|e| parse(&e.to_string())),
        (false, false) => Err(parse("neither id nor method present")),
    }
}

#[cfg(test)]
#[path = "rpc_tests.rs"]
mod tests;
