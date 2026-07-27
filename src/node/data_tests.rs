use super::*;
use crate::node::test_support::local_endpoint;
use crate::remote::{ALPN, M_NODE_PULL, N_PROGRESS};
use crate::rpc;
use tokio::io::{AsyncBufReadExt, BufReader};

/// A no-network fetcher: one chunk + one progress tick.
struct FakeFetcher;
impl crate::download::Fetcher for FakeFetcher {
    async fn fetch(
        &self,
        _target: &crate::download::PullTarget,
        on_chunk: &mut (dyn FnMut(&[u8]) + Send),
        progress: &mut (dyn FnMut(u64, Option<u64>) + Send),
    ) -> Result<(), HiggsError> {
        on_chunk(b"hello");
        progress(5, Some(5));
        Ok(())
    }
}

/// A fetcher that always fails (transport) — for exercising the dual-path fallback.
struct FailFetcher;
impl crate::download::Fetcher for FailFetcher {
    async fn fetch(
        &self,
        target: &crate::download::PullTarget,
        _on_chunk: &mut (dyn FnMut(&[u8]) + Send),
        _progress: &mut (dyn FnMut(u64, Option<u64>) + Send),
    ) -> Result<(), HiggsError> {
        Err(HiggsError::HubTransport {
            repo: target.repo.clone(),
            detail: "fake primary down".into(),
        })
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pull_stream_streams_progress_then_final_path() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().to_path_buf();
    let node = local_endpoint().await;
    let hub = local_endpoint().await;
    let node_addr = node.addr();

    // Node side: accept the bi stream, read the M_NODE_PULL request, run pull_stream.
    tokio::spawn(async move {
        let conn = node.accept().await.unwrap().await.unwrap();
        let (mut send, recv) = conn.accept_bi().await.unwrap();
        let mut lines = BufReader::new(recv).lines();
        let line = lines.next_line().await.unwrap().unwrap();
        let RpcFrame::Request(req) = rpc::decode(&line).unwrap() else {
            panic!("want request")
        };
        // Primary FAILS, fallback succeeds — exercises the dual-path fallback end-to-end:
        // the streamed bytes + final path must still come from the fallback.
        pull_stream(&conn, &mut send, req, FailFetcher, FakeFetcher, root).await;
        let _ = send.finish();
        // keep conn alive until the hub reads
        let _ = conn.closed().await;
    });

    let conn = hub.connect(node_addr, ALPN).await.unwrap();
    let (mut send, recv) = conn.open_bi().await.unwrap();
    let req = RpcRequest {
        jsonrpc: "2.0".into(),
        id: 1,
        method: M_NODE_PULL.into(),
        params: json!({ "request_id": 1, "repo": "org/m", "file": "x.gguf" }),
    };
    send.write_all(format!("{}\n", rpc::encode(&RpcFrame::Request(req))).as_bytes())
        .await
        .unwrap();
    send.finish().unwrap();

    let mut lines = BufReader::new(recv).lines();
    let mut progress = 0;
    let final_resp = loop {
        let line = lines.next_line().await.unwrap().expect("a frame");
        match rpc::decode(&line).unwrap() {
            RpcFrame::Notification(n) if n.method == N_PROGRESS => progress += 1,
            RpcFrame::Response(r) => break r,
            other => panic!("unexpected: {other:?}"),
        }
    };
    assert!(progress >= 1, "at least one N_PROGRESS");
    assert!(final_resp.error.is_none(), "pull ok: {final_resp:?}");
    let path = final_resp.result.unwrap()["path"]
        .as_str()
        .unwrap()
        .to_string();
    assert_eq!(
        std::fs::read(&path).unwrap(),
        b"hello",
        "downloaded bytes written"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pull_stream_rejects_bad_params() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().to_path_buf();
    let node = local_endpoint().await;
    let hub = local_endpoint().await;
    let node_addr = node.addr();
    tokio::spawn(async move {
        let conn = node.accept().await.unwrap().await.unwrap();
        let (mut send, recv) = conn.accept_bi().await.unwrap();
        let mut lines = BufReader::new(recv).lines();
        let line = lines.next_line().await.unwrap().unwrap();
        let RpcFrame::Request(req) = rpc::decode(&line).unwrap() else {
            panic!()
        };
        pull_stream(&conn, &mut send, req, FakeFetcher, FakeFetcher, root).await;
        let _ = send.finish();
        let _ = conn.closed().await;
    });
    let conn = hub.connect(node_addr, ALPN).await.unwrap();
    let (mut send, recv) = conn.open_bi().await.unwrap();
    // Missing required fields → invalid params.
    let req = RpcRequest {
        jsonrpc: "2.0".into(),
        id: 9,
        method: M_NODE_PULL.into(),
        params: json!({}),
    };
    send.write_all(format!("{}\n", rpc::encode(&RpcFrame::Request(req))).as_bytes())
        .await
        .unwrap();
    send.finish().unwrap();
    let mut lines = BufReader::new(recv).lines();
    let line = lines.next_line().await.unwrap().unwrap();
    let RpcFrame::Response(r) = rpc::decode(&line).unwrap() else {
        panic!()
    };
    assert_eq!(r.error.unwrap().code, -32602, "invalid params");
}
