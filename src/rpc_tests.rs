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

#[tokio::test]
async fn read_bounded_frame_matches_lines_semantics() {
    // Sequential frames, a CRLF-terminated line (the `\r` stripped), a final line WITHOUT a
    // trailing newline, then a clean EOF → None. This is the `tokio::io::Lines` contract the
    // node's serve loop relied on before the cap was added.
    let data = b"first\nwith-crlf\r\nlast-no-newline";
    let mut r = tokio::io::BufReader::new(&data[..]);
    assert_eq!(
        read_bounded_frame(&mut r, 1024).await.unwrap().as_deref(),
        Some("first")
    );
    assert_eq!(
        read_bounded_frame(&mut r, 1024).await.unwrap().as_deref(),
        Some("with-crlf"),
        "a preceding CR is stripped like Lines"
    );
    assert_eq!(
        read_bounded_frame(&mut r, 1024).await.unwrap().as_deref(),
        Some("last-no-newline"),
        "a final newline-less line is returned"
    );
    assert_eq!(
        read_bounded_frame(&mut r, 1024).await.unwrap(),
        None,
        "a clean EOF is None"
    );
}

#[tokio::test]
async fn read_bounded_frame_preserves_a_bare_cr_on_an_unterminated_final_line() {
    // A CR is stripped ONLY as part of a CRLF terminator; a bare trailing CR on a final line
    // with NO newline (EOF) is preserved — exactly like `tokio::io::Lines::next_line`. (Benign
    // for JSON decode either way, but the reader's Lines-parity claim must hold precisely.)
    let data = b"tail\r";
    let mut r = tokio::io::BufReader::new(&data[..]);
    assert_eq!(
        read_bounded_frame(&mut r, 1024).await.unwrap().as_deref(),
        Some("tail\r"),
    );
}

#[tokio::test]
async fn read_bounded_frame_accepts_a_frame_at_exactly_the_cap() {
    // 8 content bytes with cap 8 is allowed (the boundary is inclusive); one more is not
    // (asserted separately) — so a legitimate max-size frame is never falsely rejected.
    let data = b"12345678\n";
    let mut r = tokio::io::BufReader::new(&data[..]);
    assert_eq!(
        read_bounded_frame(&mut r, 8).await.unwrap().as_deref(),
        Some("12345678")
    );
}

#[tokio::test]
async fn read_bounded_frame_rejects_a_frame_over_the_cap_before_its_newline() {
    // 20 bytes then a newline, cap 8 → the newline-branch cap check fires with InvalidData.
    // Reverting that check reads the whole line as Ok(Some(..)) → unwrap_err panics (mutant).
    let data = b"xxxxxxxxxxxxxxxxxxxx\n";
    let mut r = tokio::io::BufReader::new(&data[..]);
    let err = read_bounded_frame(&mut r, 8).await.unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
}

#[tokio::test]
async fn read_bounded_frame_rejects_a_long_newlineless_stream() {
    // 100 KiB with NO newline exceeds `BufReader`'s internal chunk, so the ACCUMULATE-branch
    // cap check (not the newline branch) must fire — proving a peer cannot grow the buffer
    // unbounded by simply never sending a newline. cap 32 KiB → InvalidData.
    let data = vec![b'x'; 100 * 1024];
    let mut r = tokio::io::BufReader::new(&data[..]);
    let err = read_bounded_frame(&mut r, 32 * 1024).await.unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
}
