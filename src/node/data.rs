//! Node-side DATA relay: bridge a hub-opened chat stream to the worker's `Supervisor`
//! (DESIGN-remote.md §4.3, §5.4b). The node receives `M_CHAT{worker_id,…}` over an iroh
//! data stream, drives the existing `Supervisor::chat()` (which already bridges to the
//! child's sync stdio), and relays `N_CHAT_CHUNK` notifications + the final response back.
//!
//! One writer owns the stream (chunks then final — never raced); the hub's `request_id`
//! is echoed in every chunk; the relay is cancelled if the connection or stream drops
//! (so a slow/abandoned chat doesn't pin the worker's writer indefinitely).

use std::sync::Arc;

use iroh::endpoint::{Connection, SendStream};
use serde_json::json;

use crate::diagnostic::HiggsError;
use crate::node::runtime::NodeRuntime;
use crate::node::worker_id::WorkerId;
use crate::node::write_frame;
use crate::remote::NodeChatParams;
use crate::rpc::{RpcError, RpcFrame, RpcNotification, RpcRequest, RpcResponse};
use crate::worker::N_CHAT_CHUNK;

/// Relay one `M_CHAT` request (already read off `send`'s paired recv) to its worker and
/// stream the result back on `send`. Writes everything itself (chunks + final).
pub(crate) async fn relay_chat(
    rt: &Arc<NodeRuntime>,
    conn: &Connection,
    send: &mut SendStream,
    req: RpcRequest,
) {
    let params: NodeChatParams = match serde_json::from_value(req.params) {
        Ok(p) => p,
        Err(e) => {
            reply_err(send, req.id, -32602, format!("invalid chat params: {e}"), None).await;
            return;
        }
    };
    let sup = match rt.chat_handle(WorkerId(params.worker_id)) {
        Ok(s) => s,
        Err(e) => {
            reply_err(send, req.id, -32000, e.to_string(), hg_data(&e)).await;
            return;
        }
    };

    // Apply the worker's own defaults for omitted optional params (1024 / 0.7), so a
    // remote chat with no max_tokens generates normally instead of zero tokens.
    let (mut chunks, fut) = sup.chat(
        params.model,
        params.messages_json,
        params.max_tokens.unwrap_or(1024),
        params.temperature.unwrap_or(0.7),
        params.tools_json,
    );
    // `Supervisor::chat`'s future is `'static` (owns its own Arc) and removes the chat
    // sink on ANY outcome. Drive it in its own task so that cleanup runs even if the hub
    // disconnects mid-chat — otherwise an early return here would drop the future and
    // leak the registered sink. Bounded by the supervisor's chat timeout.
    let (final_tx, final_rx) = tokio::sync::oneshot::channel();
    tokio::spawn(async move {
        let _ = final_tx.send(fut.await);
    });
    tokio::pin!(final_rx);

    // Single writer; chunks first, then the final response. `chunks_open` disables the
    // chunk arm once the sink closes so the select never busy-loops on `None`.
    let mut chunks_open = true;
    let final_res: Result<serde_json::Value, HiggsError> = loop {
        tokio::select! {
            maybe = chunks.recv(), if chunks_open => match maybe {
                Some(delta) => {
                    if write_chunk(send, params.request_id, &delta).await.is_err() {
                        return; // hub gone — the chat task still runs to completion + cleans up
                    }
                }
                None => chunks_open = false,
            },
            res = &mut final_rx => break res.unwrap_or_else(|_| {
                Err(HiggsError::WorkerDead { context: "chat task dropped".into() })
            }),
            _ = conn.closed() => return, // chat task keeps running → sink cleaned up
            _ = send.stopped() => return,
        }
    };

    // Deliver any chunks buffered before the final resolved.
    while let Ok(delta) = chunks.try_recv() {
        if write_chunk(send, params.request_id, &delta).await.is_err() {
            return;
        }
    }

    match final_res {
        Ok(value) => reply_ok(send, req.id, value).await,
        Err(e) => reply_err(send, req.id, -32000, e.to_string(), hg_data(&e)).await,
    }
}

/// Relay one `M_NODE_PULL` request: download the GGUF into `~/.higgs/models/`, streaming
/// `N_PROGRESS` notifications, then reply with the final `{ path }` (or an `HG025` error).
pub(crate) async fn relay_pull(conn: &Connection, send: &mut SendStream, req: RpcRequest) {
    let models_root = match crate::download::models_dir() {
        Ok(d) => d,
        Err(e) => {
            reply_err(send, req.id, -32000, format!("models dir: {e}"), None).await;
            return;
        }
    };
    pull_stream(conn, send, req, crate::download::HttpFetcher, models_root).await;
}

/// Generic core of [`relay_pull`]: download via `fetcher` into `models_root`, streaming
/// `N_PROGRESS` then the final `{ path }`. Parameterized over the fetcher so it's unit-tested
/// offline with a fake (production passes the real `HttpFetcher`).
async fn pull_stream<F: crate::download::Fetcher + Send + Sync + 'static>(
    conn: &Connection,
    send: &mut SendStream,
    req: RpcRequest,
    fetcher: F,
    models_root: std::path::PathBuf,
) {
    let params: crate::remote::NodePullParams = match serde_json::from_value(req.params) {
        Ok(p) => p,
        Err(e) => {
            reply_err(send, req.id, -32602, format!("invalid pull params: {e}"), None).await;
            return;
        }
    };
    let request_id = params.request_id;
    let target = crate::download::PullTarget {
        repo: params.repo,
        file: params.file,
        revision: params.revision.unwrap_or_else(|| "main".into()),
    };

    // Run the download in its own task so a hub disconnect doesn't abort a near-complete
    // pull. Progress flows over a BOUNDED channel; when the hub stream is back-pressured the
    // download drops surplus ticks (`try_send`) rather than buffering every chunk — progress
    // is lossy-tolerant, so memory stays bounded regardless of model size.
    let (prog_tx, mut prog_rx) = tokio::sync::mpsc::channel::<(u64, Option<u64>)>(64);
    let (final_tx, final_rx) = tokio::sync::oneshot::channel();
    tokio::spawn(async move {
        let mut cb = move |downloaded: u64, total: Option<u64>| {
            let _ = prog_tx.try_send((downloaded, total));
        };
        let res = crate::download::download(&target, &models_root, &fetcher, &mut cb).await;
        let _ = final_tx.send(res);
    });
    tokio::pin!(final_rx);

    let mut progress_open = true;
    let final_res: Result<std::path::PathBuf, HiggsError> = loop {
        tokio::select! {
            maybe = prog_rx.recv(), if progress_open => match maybe {
                Some((downloaded, total)) => {
                    if write_progress(send, request_id, downloaded, total).await.is_err() {
                        return; // hub gone — the download task still finishes
                    }
                }
                None => progress_open = false,
            },
            res = &mut final_rx => break res.unwrap_or_else(|_| {
                Err(HiggsError::DownloadFailed { repo: String::new(), file: String::new(), detail: "download task dropped".into() })
            }),
            _ = conn.closed() => return,
            _ = send.stopped() => return,
        }
    };
    // Flush any progress buffered before the final resolved.
    while let Ok((downloaded, total)) = prog_rx.try_recv() {
        if write_progress(send, request_id, downloaded, total).await.is_err() {
            return;
        }
    }

    match final_res {
        Ok(path) => reply_ok(send, req.id, json!({ "path": path.to_string_lossy() })).await,
        Err(e) => reply_err(send, req.id, -32000, e.to_string(), hg_data(&e)).await,
    }
}

/// Write one notification frame (`method` + `params`) — the shared body of the typed
/// `write_progress`/`write_chunk` wrappers below.
async fn write_notification(
    send: &mut SendStream,
    method: &str,
    params: serde_json::Value,
) -> std::io::Result<()> {
    let note = RpcNotification { jsonrpc: "2.0".into(), method: method.into(), params };
    write_frame(send, &RpcFrame::Notification(note)).await
}

/// Write one `N_PROGRESS` notification (hub `request_id` + byte counts).
async fn write_progress(
    send: &mut SendStream,
    request_id: u64,
    downloaded: u64,
    total: Option<u64>,
) -> std::io::Result<()> {
    let params = json!({ "request_id": request_id, "downloaded": downloaded, "total": total });
    write_notification(send, crate::remote::N_PROGRESS, params).await
}

/// Write one `N_CHAT_CHUNK` notification carrying the hub's `request_id` + delta.
async fn write_chunk(send: &mut SendStream, request_id: u64, delta: &str) -> std::io::Result<()> {
    write_notification(send, N_CHAT_CHUNK, json!({ "request_id": request_id, "delta": delta })).await
}

/// Write a successful `result` response for request `id`.
async fn reply_ok(send: &mut SendStream, id: u64, result: serde_json::Value) {
    let resp = RpcResponse { jsonrpc: "2.0".into(), id, result: Some(result), error: None };
    let _ = write_frame(send, &RpcFrame::Response(resp)).await;
}

async fn reply_err(
    send: &mut SendStream,
    id: u64,
    code: i64,
    message: String,
    data: Option<serde_json::Value>,
) {
    let resp = RpcResponse {
        jsonrpc: "2.0".into(),
        id,
        result: None,
        error: Some(RpcError { code, message, data }),
    };
    let _ = write_frame(send, &RpcFrame::Response(resp)).await;
}

/// The HG diagnostic code for the JSON-RPC `data` — the worker's origin code when the
/// failure came from the worker (so HG003/HG005/HG018 survive the relay), else the
/// boundary code.
fn hg_data(e: &HiggsError) -> Option<serde_json::Value> {
    crate::node::worker_origin_code_data(e)
}

#[cfg(test)]
mod tests {
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
            _url: &str,
            on_chunk: &mut (dyn FnMut(&[u8]) + Send),
            progress: &mut (dyn FnMut(u64, Option<u64>) + Send),
        ) -> Result<(), String> {
            on_chunk(b"hello");
            progress(5, Some(5));
            Ok(())
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
            let RpcFrame::Request(req) = rpc::decode(&line).unwrap() else { panic!("want request") };
            pull_stream(&conn, &mut send, req, FakeFetcher, root).await;
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
        send.write_all(format!("{}\n", rpc::encode(&RpcFrame::Request(req))).as_bytes()).await.unwrap();
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
        let path = final_resp.result.unwrap()["path"].as_str().unwrap().to_string();
        assert_eq!(std::fs::read(&path).unwrap(), b"hello", "downloaded bytes written");
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
            let RpcFrame::Request(req) = rpc::decode(&line).unwrap() else { panic!() };
            pull_stream(&conn, &mut send, req, FakeFetcher, root).await;
            let _ = send.finish();
            let _ = conn.closed().await;
        });
        let conn = hub.connect(node_addr, ALPN).await.unwrap();
        let (mut send, recv) = conn.open_bi().await.unwrap();
        // Missing required fields → invalid params.
        let req = RpcRequest { jsonrpc: "2.0".into(), id: 9, method: M_NODE_PULL.into(), params: json!({}) };
        send.write_all(format!("{}\n", rpc::encode(&RpcFrame::Request(req))).as_bytes()).await.unwrap();
        send.finish().unwrap();
        let mut lines = BufReader::new(recv).lines();
        let line = lines.next_line().await.unwrap().unwrap();
        let RpcFrame::Response(r) = rpc::decode(&line).unwrap() else { panic!() };
        assert_eq!(r.error.unwrap().code, -32602, "invalid params");
    }
}
