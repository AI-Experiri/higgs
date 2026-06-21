//! Hub-side per-node transport client (DESIGN-remote.md §2.3, P3 hub seam).
//!
//! The hub does NOT wrap remote workers in a `Supervisor` (codex-validated): it keeps its
//! local `Supervisor` for its own child and, per paired node, holds a `NodeTransport` over
//! the live iroh `Connection`. `request()` issues a `higgs/node/*` control RPC; `chat()`
//! relays a remote `M_CHAT` and streams `N_CHAT_CHUNK` + final back. One bidi stream per
//! call, so the per-stream reader IS the demux — no shared pending map needed, and the two
//! correlation domains (local Supervisor vs this transport) never merge.

use std::sync::atomic::{AtomicU64, Ordering};

use iroh::endpoint::Connection;
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::sync::mpsc;

use crate::diagnostic::HiggsError;
use crate::node::write_frame;
use crate::rpc::{self, RpcFrame, RpcRequest, RpcResponse};
use crate::worker::{M_CHAT, N_CHAT_CHUNK};

/// Bounds a control RPC over the transport (mirrors the local `CONTROL_RPC_TIMEOUT`): a
/// wedged node that accepts the stream but never replies must not hang the hub forever.
const CONTROL_RPC_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(120);
/// Bounds a remote chat (mirrors the local `CHAT_RPC_TIMEOUT`).
const CHAT_RPC_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(600);

/// A live transport to one paired node.
pub struct NodeTransport {
    conn: Connection,
    next_id: AtomicU64,
}

impl NodeTransport {
    pub fn new(conn: Connection) -> Self {
        Self {
            conn,
            next_id: AtomicU64::new(1),
        }
    }

    fn alloc_id(&self) -> u64 {
        self.next_id.fetch_add(1, Ordering::Relaxed)
    }

    /// A clone of the underlying connection, for the fleet's node→hub log-relay reader
    /// (which accepts the node's uni stream of `N_LOG_LINE` frames).
    pub(crate) fn connection(&self) -> Connection {
        self.conn.clone()
    }

    /// Resolves when the underlying connection closes (peer death, idle drop, transport
    /// error) — used by the fleet to retire a node and clear its routes.
    pub async fn closed(&self) {
        let _ = self.conn.closed().await;
    }

    /// Explicitly close the connection (on retire / session replace) — also wakes any
    /// `closed()` watcher so it can release its handle and free the connection.
    pub fn close(&self) {
        self.conn.close(0u32.into(), b"retired");
    }

    /// Issue one `higgs/node/*` control RPC on a fresh bidi stream and await its response.
    pub async fn request(&self, method: &str, params: Value) -> Result<Value, HiggsError> {
        let id = self.alloc_id();
        let (mut send, recv) = self.conn.open_bi().await.map_err(transport_dead)?;
        let req = RpcRequest {
            jsonrpc: "2.0".into(),
            id,
            method: method.into(),
            params,
        };
        write_frame(&mut send, &RpcFrame::Request(req))
            .await
            .map_err(transport_dead)?;
        let _ = send.finish();
        let mut lines = BufReader::new(recv).lines();
        let line = match tokio::time::timeout(CONTROL_RPC_TIMEOUT, lines.next_line()).await {
            Ok(Ok(Some(line))) => line,
            Ok(Ok(None)) => return Err(transport_dead("node closed the stream before replying")),
            Ok(Err(e)) => return Err(transport_dead(e)),
            Err(_) => return Err(transport_dead(format!("control RPC {method} timed out"))),
        };
        match rpc::decode(&line).map_err(|e| transport_dead(e.to_string()))? {
            RpcFrame::Response(resp) => extract_result(method, resp),
            other => Err(transport_dead(format!("unexpected reply frame: {other:?}"))),
        }
    }

    /// Relay a remote chat: open a data stream, send `M_CHAT`, and return the streamed
    /// deltas (`rx`) plus a future resolving to the final result. The hub uses the SAME
    /// value for the JSON-RPC `id` and `params.request_id` (codex), so the per-stream
    /// reader routes both by one id.
    #[allow(clippy::too_many_arguments)]
    pub async fn chat(
        &self,
        worker_id: u32,
        model: String,
        messages_json: String,
        max_tokens: usize,
        temperature: f32,
        tools_json: Option<String>,
    ) -> Result<(mpsc::UnboundedReceiver<String>, ChatDone), HiggsError> {
        let id = self.alloc_id();
        let (mut send, recv) = self.conn.open_bi().await.map_err(transport_dead)?;
        let mut params = json!({
            "request_id": id,
            "worker_id": worker_id,
            "model": model,
            "messages_json": messages_json,
            "max_tokens": max_tokens,
            "temperature": temperature,
        });
        if let (Some(obj), Some(t)) = (params.as_object_mut(), tools_json) {
            obj.insert("tools".into(), Value::String(t));
        }
        let req = RpcRequest {
            jsonrpc: "2.0".into(),
            id,
            method: M_CHAT.into(),
            params,
        };
        write_frame(&mut send, &RpcFrame::Request(req))
            .await
            .map_err(transport_dead)?;
        let _ = send.finish();

        let (tx, rx) = mpsc::unbounded_channel();
        // The per-stream reader IS the demux: chunks → tx, final Response → the result.
        let fut = ChatDone {
            inner: Box::pin(async move {
                let read = async move {
                    let mut lines = BufReader::new(recv).lines();
                    loop {
                        match lines.next_line().await.map_err(transport_dead)? {
                            Some(line) => match rpc::decode(&line)
                                .map_err(|e| transport_dead(e.to_string()))?
                            {
                                RpcFrame::Notification(n) if n.method == N_CHAT_CHUNK => {
                                    if let Some(d) = n.params.get("delta").and_then(|v| v.as_str())
                                    {
                                        let _ = tx.send(d.to_string());
                                    }
                                }
                                RpcFrame::Response(resp) => break extract_result(M_CHAT, resp),
                                _ => {}
                            },
                            None => {
                                break Err(transport_dead(
                                    "node closed the chat stream before final",
                                ))
                            }
                        }
                    }
                };
                // Bound a wedged remote inference so the caller's stream can't hang forever.
                // A timeout is a CHAT timeout (HG016/504), NOT a dead transport: the node and
                // connection may be perfectly healthy on a long generation, so it must not be
                // remapped to HG027 or tear down the node (that's `handle_op_error`'s job only
                // for `WorkerDead`). HG016 is also what the local chat path surfaces.
                match tokio::time::timeout(CHAT_RPC_TIMEOUT, read).await {
                    Ok(result) => result,
                    Err(_) => Err(HiggsError::ChatTimeout {
                        elapsed: CHAT_RPC_TIMEOUT,
                    }),
                }
            }),
        };
        Ok((rx, fut))
    }
}

/// The final-result future returned by [`NodeTransport::chat`].
pub struct ChatDone {
    inner: std::pin::Pin<Box<dyn std::future::Future<Output = Result<Value, HiggsError>> + Send>>,
}

impl std::future::Future for ChatDone {
    type Output = Result<Value, HiggsError>;
    fn poll(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        self.inner.as_mut().poll(cx)
    }
}

/// Extract `result` from a node reply, mapping a node/worker error to a `HiggsError` that
/// carries the origin diagnostic code (so the hub maps the true status).
fn extract_result(method: &str, resp: RpcResponse) -> Result<Value, HiggsError> {
    if let Some(err) = resp.error {
        let worker_code = err
            .data
            .as_ref()
            .and_then(|d| d.get("code"))
            .and_then(|c| c.as_str())
            .map(ToOwned::to_owned);
        return Err(HiggsError::WorkerRpc {
            method: method.to_string(),
            message: err.message,
            worker_code,
        });
    }
    Ok(resp.result.unwrap_or(Value::Null))
}

#[allow(clippy::needless_pass_by_value)] // by-value is ergonomic for the varied callers
fn transport_dead(detail: impl ToString) -> HiggsError {
    HiggsError::WorkerDead {
        context: format!("node transport: {}", detail.to_string()),
    }
}

#[cfg(test)]
mod tests {
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
            .chat(999, "m".into(), "[]".into(), 8, 0.0, None)
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
            .chat(worker_id, "higgs-test/m".into(), "[]".into(), 8, 0.0, None)
            .await
            .unwrap();
        // Drain chunks concurrently with driving `fut` (the future IS the reader that
        // feeds `rx`); draining first would deadlock.
        let collector = tokio::spawn(async move {
            let mut got = Vec::new();
            while let Some(d) = rx.recv().await {
                got.push(d);
            }
            got
        });
        let final_res = fut.await.unwrap();
        let got = collector.await.unwrap();
        assert_eq!(got, vec!["he", "llo"], "streamed chunks");
        assert_eq!(final_res["content"], "hello");
    }
}
