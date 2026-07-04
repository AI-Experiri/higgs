use super::*;
use crate::node::serve_node;
use crate::node::test_support::{fake_runtime, local_endpoint, stage_dummy_model};
use crate::remote::{ALPN, M_NODE_LOAD, M_NODE_STATUS, M_NODE_SYSINFO};
use std::sync::Arc;

/// Stand up a node (serve_node + fake workers) and a hub `NodeTransport` over a real
/// in-process iroh connection. Returns the transport + the staged model id.
async fn paired_transport() -> (NodeTransport, String, tempfile::TempDir) {
    let (root, model_id) = stage_dummy_model("higgs-test/m");
    let hub = local_endpoint().await;
    let node = local_endpoint().await;
    let hub_addr = hub.addr();

    // connect() and accept() must run concurrently (the QUIC handshake needs both
    // sides). Node dials + serves in a task; the hub accepts and wraps the conn.
    let rt = Arc::new(fake_runtime(vec![root.path().to_path_buf()]));
    tokio::spawn(async move {
        let node_conn = node.connect(hub_addr, ALPN).await.expect("node connect");
        serve_node(node_conn, rt).await;
    });
    let conn = hub.accept().await.expect("incoming").await.expect("conn");
    // Keep the hub Endpoint alive for the connection's lifetime (test-only leak).
    std::mem::forget(hub);
    (NodeTransport::new(conn), model_id, root)
}

#[tokio::test]
async fn control_request_roundtrips() {
    let (t, model_id, _root) = paired_transport().await;
    let loaded = t
        .request(M_NODE_LOAD, json!({ "id": model_id }))
        .await
        .unwrap();
    let worker_id = loaded["worker_id"].as_u64().unwrap();
    let status = t
        .request(M_NODE_STATUS, json!({ "worker_id": worker_id }))
        .await
        .unwrap();
    assert!(status.get("loaded").is_some());
    let sys = t.request(M_NODE_SYSINFO, json!({})).await.unwrap();
    assert!(sys["hardware"]["cpu_cores"].as_u64().unwrap() > 0);
}

#[tokio::test]
async fn control_request_error_maps_worker_code() {
    // Loading an id with no on-disk model → node replies HG002, surfaced as WorkerRpc
    // carrying the origin code (extract_result mapping).
    let (t, _model_id, _root) = paired_transport().await;
    let err = t
        .request(M_NODE_LOAD, json!({ "id": "missing/model" }))
        .await
        .unwrap_err();
    match err {
        HiggsError::WorkerRpc { worker_code, .. } => {
            assert_eq!(worker_code.as_deref(), Some("HG002"))
        }
        other => panic!("expected WorkerRpc, got {other}"),
    }
}

#[tokio::test]
async fn chat_unknown_worker_errors() {
    // Chat to a worker id that doesn't exist → the node's relay reports it; the chat
    // future resolves to an error rather than streaming.
    let (t, _model_id, _root) = paired_transport().await;
    let (_rx, fut) = t
        .chat(999, "m".into(), "[]".into(), 8, 0.0, None, None)
        .await
        .unwrap();
    assert!(fut.await.is_err(), "unknown worker → chat error");
}

#[tokio::test]
async fn chat_streams_chunks_then_final() {
    let (t, model_id, _root) = paired_transport().await;
    let worker_id = t
        .request(M_NODE_LOAD, json!({ "id": model_id }))
        .await
        .unwrap()["worker_id"]
        .as_u64()
        .unwrap() as u32;
    let (mut rx, fut) = t
        .chat(
            worker_id,
            "higgs-test/m".into(),
            "[]".into(),
            8,
            0.0,
            None,
            None,
        )
        .await
        .unwrap();
    // Drain chunks concurrently with driving `fut` (the future IS the reader that
    // feeds `rx`); draining first would deadlock.
    let collector = tokio::spawn(async move {
        let mut got = Vec::new();
        while let Some(d) = rx.recv().await {
            got.push(d.text);
        }
        got
    });
    let final_res = fut.await.unwrap();
    let got = collector.await.unwrap();
    assert_eq!(got, vec!["he", "llo"], "streamed chunks");
    assert_eq!(final_res["content"], "hello");
}
