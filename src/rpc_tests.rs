
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
        error: Some(RpcError {
            code: -32601,
            message: "unknown method".into(),
            data: None,
        }),
    });
    assert_eq!(decode(&encode(&err)).unwrap(), err);
}

#[test]
fn garbage_is_hg008() {
    let err = decode("{not json").unwrap_err();
    assert!(err.to_string().starts_with("[HG008]"));
}

#[test]
fn wrong_jsonrpc_version_decodes_softly() {
    // A mismatched version is logged (soft check) but still decodes — a hard
    // reject would wedge the RPC loop since both peers are the same binary.
    let line = r#"{"jsonrpc":"1.0","id":1,"method":"higgs/ping","params":null}"#;
    let frame = decode(line).expect("soft version check must not reject the frame");
    assert!(matches!(frame, RpcFrame::Request(_)));
}
