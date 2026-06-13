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
    pub jsonrpc: String,
    pub id: u64,
    pub method: String,
    #[serde(default, skip_serializing_if = "Value::is_null")]
    pub params: Value,
}

/// A JSON-RPC response carrying either `result` or `error`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RpcResponse {
    pub jsonrpc: String,
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
}

/// A JSON-RPC notification (no id; fire-and-forget — used for chat chunks).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RpcNotification {
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
pub fn decode(line: &str) -> Result<RpcFrame, HiggsError> {
    let v: Value = serde_json::from_str(line).map_err(|e| HiggsError::RpcDecode {
        detail: format!("not json: {e}"),
    })?;
    let has_id = v.get("id").is_some();
    let has_method = v.get("method").is_some();
    let parse = |detail: &str| HiggsError::RpcDecode { detail: detail.to_string() };
    match (has_id, has_method) {
        (true, true) => serde_json::from_value(v).map(RpcFrame::Request).map_err(|e| parse(&e.to_string())),
        (true, false) => serde_json::from_value(v).map(RpcFrame::Response).map_err(|e| parse(&e.to_string())),
        (false, true) => serde_json::from_value(v).map(RpcFrame::Notification).map_err(|e| parse(&e.to_string())),
        (false, false) => Err(parse("neither id nor method present")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn request_roundtrip() {
        let f = RpcFrame::Request(RpcRequest {
            jsonrpc: "2.0".into(),
            id: 7,
            method: "higgs/load".into(),
            params: json!({"id": "google/gemma-4-12b"}),
        });
        let line = encode(&f);
        assert!(!line.contains('\n'));
        assert_eq!(decode(&line).unwrap(), f);
    }

    #[test]
    fn notification_roundtrip() {
        let f = RpcFrame::Notification(RpcNotification {
            jsonrpc: "2.0".into(),
            method: "higgs/chat/chunk".into(),
            params: json!({"request_id": 3, "delta": "hel"}),
        });
        assert_eq!(decode(&encode(&f)).unwrap(), f);
    }

    #[test]
    fn response_roundtrip_both_arms() {
        let ok = RpcFrame::Response(RpcResponse {
            jsonrpc: "2.0".into(),
            id: 9,
            result: Some(json!({"loaded": true})),
            error: None,
        });
        assert_eq!(decode(&encode(&ok)).unwrap(), ok);

        let err = RpcFrame::Response(RpcResponse {
            jsonrpc: "2.0".into(),
            id: 10,
            result: None,
            error: Some(RpcError { code: -32601, message: "unknown method".into() }),
        });
        assert_eq!(decode(&encode(&err)).unwrap(), err);
    }

    #[test]
    fn garbage_is_hg008() {
        let err = decode("{not json").unwrap_err();
        assert!(err.to_string().starts_with("[HG008]"));
    }
}
