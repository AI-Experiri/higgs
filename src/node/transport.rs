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

    /// A clone of the underlying connection, for the fleet's node→hub notification
    /// reader (which accepts the node's uni streams of `N_LOG_LINE` and
    /// `N_FLEET_EVENT` frames).
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
        crate::node::stream_priority::apply_for(&send, method);
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

    /// [`request`](Self::request) with a BOUNDED reply read: the frame is
    /// rejected once it exceeds `max_bytes`, so a compromised/buggy node
    /// cannot make the hub buffer an arbitrarily large reply line before any
    /// consumer-side cap applies. Use for ops whose reply has a known small
    /// shape (e.g. `M_NODE_PULL_STATUS`).
    pub async fn request_bounded(
        &self,
        method: &str,
        params: Value,
        max_bytes: usize,
    ) -> Result<Value, HiggsError> {
        let id = self.alloc_id();
        // Bound the OPEN + WRITE phase too (same hazard `pull()` guards): a
        // wedged node or withheld QUIC stream credit would otherwise hang
        // BEFORE the reply-read timeout ever starts — and the caller's
        // handle_op_error (which drops the dead transport) would never run.
        let open_and_send = async {
            let (mut send, recv) = self.conn.open_bi().await.map_err(transport_dead)?;
            crate::node::stream_priority::apply_for(&send, method);
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
            Ok::<_, HiggsError>(recv)
        };
        let recv = match tokio::time::timeout(CONTROL_RPC_TIMEOUT, open_and_send).await {
            Ok(r) => r?,
            Err(_) => {
                return Err(transport_dead(format!(
                    "control RPC {method}: open/send timed out"
                )))
            }
        };
        let mut reader = BufReader::new(recv);
        let line = match tokio::time::timeout(
            CONTROL_RPC_TIMEOUT,
            rpc::read_bounded_frame(&mut reader, max_bytes),
        )
        .await
        {
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

    /// Trigger a catalog download ON the node (`M_NODE_PULL`): the node fetches
    /// `repo`/`file` from the Hub ITSELF (its own endpoint/fallback, atomic into
    /// its models dir) — no bytes cross this stream, only `N_PROGRESS`
    /// notifications and the final `{ path }` reply. Deliberately no TOTAL
    /// deadline (a multi-GB pull runs as long as it runs) but a STALL deadline
    /// ([`PULL_STALL_TIMEOUT`]): the node streams a progress notification per
    /// chunk, so a stream with no frame for that long is a wedged node/Hub —
    /// iroh keep-alives keep the CONNECTION alive, so QUIC liveness alone
    /// would never bound an application-silent peer. Mirrors the stall-based
    /// load timeout philosophy.
    pub async fn pull(
        &self,
        repo: &str,
        file: &str,
        revision: Option<&str>,
        on_progress: &mut (dyn FnMut(u64, Option<u64>) + Send),
    ) -> Result<Value, HiggsError> {
        let id = self.alloc_id();
        // Bound the OPEN + WRITE phase with the same stall deadline: a peer
        // that withholds stream credit (or stalls the request write) while
        // keeping the connection alive would otherwise hang before the read
        // loop's own stall clock ever starts — the same hazard the update
        // push bounds with its whole-push timeout.
        let open_and_send = async {
            let (mut send, recv) = self.conn.open_bi().await.map_err(transport_dead)?;
            crate::node::stream_priority::apply_for(&send, crate::remote::M_NODE_PULL);
            let params = crate::remote::NodePullParams {
                request_id: id,
                repo: repo.to_owned(),
                file: file.to_owned(),
                revision: revision.map(str::to_owned),
            };
            let req = RpcRequest {
                jsonrpc: "2.0".into(),
                id,
                method: crate::remote::M_NODE_PULL.into(),
                params: serde_json::to_value(&params)
                    .map_err(|e| transport_dead(format!("encode pull params: {e}")))?,
            };
            write_frame(&mut send, &RpcFrame::Request(req))
                .await
                .map_err(transport_dead)?;
            let _ = send.finish();
            Ok::<_, HiggsError>(recv)
        };
        let recv = match tokio::time::timeout(PULL_STALL_TIMEOUT, open_and_send).await {
            Ok(res) => res?,
            Err(_) => {
                return Err(transport_dead(format!(
                    "pull open/send stalled: no stream in {}s",
                    PULL_STALL_TIMEOUT.as_secs()
                )))
            }
        };
        pull_read_loop(BufReader::new(recv), on_progress, PULL_STALL_TIMEOUT).await
    }

    /// Fetch/stream the node's OWN DAEMON log (`M_NODE_LOGS`, the node.log lines —
    /// never worker/model output). Every line (and `lagged` marker, rendered as a
    /// visible `… <n> lines dropped` line) is handed to `on_line`; returns when the
    /// node completes the reply (snapshot mode), errors, or — in `follow` mode —
    /// when `stop` resolves (the watcher left: the recv stream is DROPPED, the node
    /// sees `send.stopped()` and stops sending — hard teardown, no idle traffic).
    /// No stall clock in follow mode: an idle log is healthy; connection death is
    /// surfaced by the read erroring when QUIC closes.
    pub async fn node_logs(
        &self,
        n: u64,
        follow: bool,
        on_line: &mut (dyn FnMut(String) + Send),
        stop: std::pin::Pin<&mut (dyn std::future::Future<Output = ()> + Send)>,
    ) -> Result<(), HiggsError> {
        let id = self.alloc_id();
        let open_and_send = async {
            let (mut send, recv) = self.conn.open_bi().await.map_err(transport_dead)?;
            crate::node::stream_priority::apply_for(&send, crate::remote::M_NODE_LOGS);
            let req = RpcRequest {
                jsonrpc: "2.0".into(),
                id,
                method: crate::remote::M_NODE_LOGS.into(),
                params: serde_json::json!({ "n": n, "follow": follow }),
            };
            write_frame(&mut send, &RpcFrame::Request(req))
                .await
                .map_err(transport_dead)?;
            let _ = send.finish();
            Ok::<_, HiggsError>(recv)
        };
        let recv = match tokio::time::timeout(CONTROL_RPC_TIMEOUT, open_and_send).await {
            Ok(res) => res?,
            Err(_) => {
                return Err(transport_dead(format!(
                    "node-logs open/send stalled: no stream in {}s",
                    CONTROL_RPC_TIMEOUT.as_secs()
                )))
            }
        };
        let mut reader = BufReader::new(recv);
        let read_loop = async {
            let mut line = String::new();
            loop {
                line.clear();
                let read = reader
                    .read_line(&mut line)
                    .await
                    .map_err(|e| transport_dead(format!("node-logs read: {e}")))?;
                if read == 0 {
                    // The happy path ALWAYS terminates on the final Response frame
                    // (snapshot mode) or the `stop` future (follow mode); a bare EOF
                    // is the stream closing BEFORE that completion. Treat it as an
                    // error so a truncated snapshot isn't reported as whole and a
                    // dropped follow surfaces its end reason to the console.
                    return Err(transport_dead(
                        "node-logs stream closed before its final response",
                    ));
                }
                match rpc::decode(line.trim_end()) {
                    Ok(RpcFrame::Notification(note))
                        if note.method == crate::remote::N_NODE_LOG =>
                    {
                        if let Some(l) = note.params.get("line").and_then(Value::as_str) {
                            on_line(l.to_string());
                        } else if let Some(k) = note.params.get("lagged").and_then(Value::as_u64) {
                            on_line(format!("… {k} lines dropped (stream lagging)"));
                        }
                    }
                    Ok(RpcFrame::Response(resp)) => {
                        return extract_result(crate::remote::M_NODE_LOGS, resp).map(|_| ());
                    }
                    _ => {} // unknown frames skipped (forward compat)
                }
            }
        };
        tokio::select! {
            res = read_loop => res,
            // Watcher gone: drop the stream (via return) — the node's
            // `send.stopped()` fires and it stops streaming.
            _ = stop => Ok(()),
        }
    }

    /// Relay a remote chat: open a data stream, send `M_CHAT`, and return the streamed
    /// deltas (`rx`) plus a future resolving to the final result. The hub uses the SAME
    /// value for the JSON-RPC `id` and `params.request_id` (codex), so the per-stream
    /// reader routes both by one id.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    pub async fn chat(
        &self,
        worker_id: u32,
        model: String,
        messages_json: String,
        max_tokens: usize,
        temperature: f32,
        tools_json: Option<String>,
        chat_template_kwargs: Option<String>,
    ) -> Result<(crate::delta_queue::DeltaReceiver, ChatDone), HiggsError> {
        let id = self.alloc_id();
        let (mut send, recv) = self.conn.open_bi().await.map_err(transport_dead)?;
        crate::node::stream_priority::apply_for(&send, M_CHAT);
        let mut params = json!({
            "request_id": id,
            "worker_id": worker_id,
            "model": model,
            "messages_json": messages_json,
            "max_tokens": max_tokens,
            "temperature": temperature,
        });
        if let Some(obj) = params.as_object_mut() {
            if let Some(t) = tools_json {
                obj.insert("tools".into(), Value::String(t));
            }
            // Additive optional field — NodeChatParams is not deny_unknown_fields,
            // so an old node simply ignores it (chat passthrough policy).
            if let Some(k) = chat_template_kwargs {
                obj.insert("chat_template_kwargs".into(), Value::String(k));
            }
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

        // Bounded MERGING queue (crate::delta_queue): a slow hub-side SSE
        // consumer coalesces same-kind deltas instead of growing an unbounded
        // per-token backlog while the node keeps streaming.
        let (tx, rx) = crate::delta_queue::delta_channel();
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
                                    // Tolerant additive decode: absent `kind` ⇒
                                    // content (chunks from an old node just work).
                                    match crate::worker::engine::ChatDelta::decode_chunk_params(
                                        &n.params,
                                    ) {
                                        Some(d) => tx.send(d),
                                        None => tracing::warn!(
                                            params = %n.params,
                                            "[HG051] undecodable remote chat chunk dropped"
                                        ),
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
/// Stall deadline for one pull frame: the node emits a progress notification
/// per received chunk, so a stream with NO frame for this long means the node
/// (or its Hub fetch) is wedged — fail the pull instead of holding the
/// in-flight slot forever. There is deliberately no TOTAL deadline.
pub(crate) const PULL_STALL_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(120);

/// The pull stream demux, split from [`NodeTransport::pull`] so it is
/// unit-testable over any reader: `N_PROGRESS` notifications → `on_progress`
/// (tolerant decode — a malformed one is dropped, never fails the pull), the
/// final `Response` → the result, early close → transport-dead, and a frame
/// gap longer than `stall` → transport-dead ("stalled").
async fn pull_read_loop<R>(
    reader: R,
    on_progress: &mut (dyn FnMut(u64, Option<u64>) + Send),
    stall: std::time::Duration,
) -> Result<Value, HiggsError>
where
    R: tokio::io::AsyncBufRead + Unpin,
{
    let mut lines = reader.lines();
    loop {
        let next = match tokio::time::timeout(stall, lines.next_line()).await {
            Ok(res) => res.map_err(transport_dead)?,
            Err(_) => {
                break Err(transport_dead(format!(
                    "pull stalled: no frame for {}s",
                    stall.as_secs()
                )))
            }
        };
        match next {
            Some(line) => match rpc::decode(&line).map_err(|e| transport_dead(e.to_string()))? {
                RpcFrame::Notification(n) if n.method == crate::remote::N_PROGRESS => {
                    let downloaded = n
                        .params
                        .get("downloaded")
                        .and_then(Value::as_u64)
                        .unwrap_or(0);
                    // Same counter normalization `accept_announced_downloads`
                    // applies to the HELLO/pull_status paths for identical
                    // node-supplied numbers: an impossible total (zero, or
                    // less than `downloaded`) degrades to None ("length
                    // unknown") so a skewed/faulty node cannot push a
                    // divide-by-zero or a >100% bar into UI percent math.
                    let total = n
                        .params
                        .get("total")
                        .and_then(Value::as_u64)
                        .filter(|&t| t > 0 && t >= downloaded);
                    on_progress(downloaded, total);
                }
                RpcFrame::Response(resp) => break extract_result(crate::remote::M_NODE_PULL, resp),
                _ => {}
            },
            None => break Err(transport_dead("node closed the pull stream before final")),
        }
    }
}

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
#[path = "transport_tests.rs"]
mod tests;
