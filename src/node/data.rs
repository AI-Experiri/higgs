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

    let resp = match final_res {
        Ok(value) => RpcResponse { jsonrpc: "2.0".into(), id: req.id, result: Some(value), error: None },
        Err(e) => RpcResponse {
            jsonrpc: "2.0".into(),
            id: req.id,
            result: None,
            error: Some(RpcError { code: -32000, message: e.to_string(), data: hg_data(&e) }),
        },
    };
    let _ = write_frame(send, &RpcFrame::Response(resp)).await;
}

/// Write one `N_CHAT_CHUNK` notification carrying the hub's `request_id` + delta.
async fn write_chunk(send: &mut SendStream, request_id: u64, delta: &str) -> std::io::Result<()> {
    let note = RpcNotification {
        jsonrpc: "2.0".into(),
        method: N_CHAT_CHUNK.into(),
        params: json!({ "request_id": request_id, "delta": delta }),
    };
    write_frame(send, &RpcFrame::Notification(note)).await
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
