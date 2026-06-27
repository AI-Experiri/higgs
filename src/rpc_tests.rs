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
fn wrong_jsonrpc_version_warns_but_decodes() {
    // The soft version check (the `if ver != "2.0"` branch + its `warn!`) fires on a
    // mismatched version, yet the frame still decodes — a hard reject would wedge the
    // RPC loop. A notification form (no id) exercises the (false,true) success arm.
    let line = r#"{"jsonrpc":"9.9","method":"higgs/chunk","params":null}"#;
    let frame = decode(line).expect("soft version check still decodes");
    assert!(matches!(frame, RpcFrame::Notification(_)));
}

#[test]
fn malformed_frames_are_hg008_decode_errors() {
    // The `parse` closure + each typed arm's error branch: a frame that is valid JSON
    // but does not deserialize into its discriminated RpcFrame variant is an HG008
    // RpcDecode, not a panic.
    for (line, why) in [
        // has id + method, but id is a string → RpcRequest parse fails (true,true arm).
        (
            r#"{"jsonrpc":"2.0","id":"not-a-number","method":"x"}"#,
            "non-numeric id",
        ),
        // has id, no method, but missing required `jsonrpc` → RpcResponse fails (true,false).
        (
            r#"{"id":5,"result":{"ok":true}}"#,
            "response missing jsonrpc",
        ),
        // no id, has method, missing `jsonrpc` → RpcNotification fails (false,true).
        (
            r#"{"method":"x","params":null}"#,
            "notification missing jsonrpc",
        ),
    ] {
        let err = decode(line).unwrap_err();
        assert!(
            err.to_string().starts_with("[HG008]"),
            "{why}: expected HG008, got {err}"
        );
    }
}

#[test]
fn neither_id_nor_method_is_hg008() {
    // The (false, false) arm: a JSON object with neither id nor method is undecodable.
    let err = decode(r#"{"jsonrpc":"2.0"}"#).unwrap_err();
    let msg = err.to_string();
    assert!(msg.starts_with("[HG008]"), "{msg}");
    assert!(
        msg.contains("neither id nor method"),
        "carries the discriminator detail: {msg}"
    );
}

#[test]
fn wrong_jsonrpc_version_decodes_softly() {
    // A mismatched version is logged (soft check) but still decodes — a hard
    // reject would wedge the RPC loop since both peers are the same binary.
    let line = r#"{"jsonrpc":"1.0","id":1,"method":"higgs/ping","params":null}"#;
    let frame = decode(line).expect("soft version check must not reject the frame");
    assert!(matches!(frame, RpcFrame::Request(_)));
}
