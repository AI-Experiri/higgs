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
            reply_err(
                send,
                req.id,
                -32602,
                format!("invalid chat params: {e}"),
                None,
            )
            .await;
            return;
        }
    };
    let lease = match rt
        .chat_handle(WorkerId(params.worker_id), &params.model)
        .await
    {
        Ok(s) => s,
        Err(e) => {
            reply_err(send, req.id, -32000, e.to_string(), hg_data(&e)).await;
            return;
        }
    };

    // Apply the worker's own defaults for omitted optional params (1024 / 0.7), so a
    // remote chat with no max_tokens generates normally instead of zero tokens.
    // Remote sampling forwarding is DEFERRED: the hub→node wire carries only
    // `temperature` (the rest of the sampler set, and any local card-recommended
    // base, are not applied on the relay path — see DESIGN-autotune §9). Wrap the
    // forwarded temperature in the engine umbrella; `None` lets the worker default
    // (0.7) stand.
    let sampling = crate::worker::engine::SamplingParams::llamacpp(
        crate::worker::engine::llamacpp::params::LlamaCppSamplingParams {
            temperature: params.temperature,
            ..Default::default()
        },
    );
    let (mut chunks, fut) = lease.chat(
        params.model,
        params.messages_json,
        params.max_tokens.unwrap_or(1024),
        sampling,
        params.tools_json,
        params.chat_template_kwargs,
    );
    // `Supervisor::chat`'s future is `'static` (owns its own Arc) and removes the chat
    // sink on ANY outcome. Drive it in its own task so that cleanup runs even if the hub
    // disconnects mid-chat — otherwise an early return here would drop the future and
    // leak the registered sink. Bounded by the supervisor's chat timeout.
    //
    // Move the `ChatLease` INTO this task so the worker's in-flight hold (which keeps the
    // idle reaper from unloading it) lasts until the generation ACTUALLY finishes — not
    // until `relay_chat` returns. A hub disconnect returns from `relay_chat` early while the
    // generation keeps running here; dropping the lease only now posts `ChatEnd`, so the
    // worker can't be idle-reaped mid-generation.
    let (final_tx, final_rx) = tokio::sync::oneshot::channel();
    tokio::spawn(async move {
        let res = fut.await;
        // Drop the lease BEFORE the final result becomes visible to the relay
        // (T14 r3): the drop posts `ChatEnd` into the node actor's mailbox, so
        // it is enqueued strictly before the hub can receive the reply — and
        // therefore before any reply-triggered inventory snapshot request
        // reaches that same mailbox. A refresh can then never observe this
        // COMPLETED chat as still in flight. NB `fut` resolving means the chat
        // CALL is over (result or the supervisor's ChatTimeout) — a timed-out
        // generation may still be running inside the worker; releasing the
        // in-flight hold then is PRE-EXISTING supervisor-timeout behavior
        // (the lease always dropped when `fut` resolved), and the idle reaper
        // reclaiming such a worker is the designed recovery for it.
        drop(lease);
        let _ = final_tx.send(res);
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
    while let Some(delta) = chunks.try_recv() {
        if write_chunk(send, params.request_id, &delta).await.is_err() {
            return;
        }
    }

    // A tripped delta-buffer cap means this hub stalled long enough that its
    // undelivered backlog was dropped ([HG057]) — the relayed stream is
    // incomplete, so fail the RPC loudly instead of replying as if the full
    // stream was delivered (the generation itself finished on the worker).
    if chunks.overflowed() {
        let e = HiggsError::ChatStreamOverflow {
            buffered_bytes: chunks.buffered_bytes(),
        };
        tracing::warn!(error = %e, "higgs: relayed chat stream overflowed");
        reply_err(send, req.id, -32000, e.to_string(), hg_data(&e)).await;
        return;
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
    // Hub client is the PRIMARY download path; the hand-rolled `reqwest` `HttpFetcher` is the
    // fail-open FALLBACK. `download_dual` tries them in order and reports `HG036` only if both
    // exhaust (each carrying its own classified diagnosis).
    pull_stream(
        conn,
        send,
        req,
        crate::hub::HubFetcher,
        crate::download::HttpFetcher,
        models_root,
    )
    .await;
}

/// Generic core of [`relay_pull`]: download via `primary` (falling back to `fallback`) into
/// `models_root`, streaming `N_PROGRESS` then the final `{ path }`. Parameterized over both
/// fetchers so it's unit-tested offline with fakes (production passes the hub-client primary +
/// the `HttpFetcher` fallback).
async fn pull_stream<P, F>(
    conn: &Connection,
    send: &mut SendStream,
    req: RpcRequest,
    primary: P,
    fallback: F,
    models_root: std::path::PathBuf,
) where
    P: crate::download::Fetcher + Send + Sync + 'static,
    F: crate::download::Fetcher + Send + Sync + 'static,
{
    let params: crate::remote::NodePullParams = match serde_json::from_value(req.params) {
        Ok(p) => p,
        Err(e) => {
            reply_err(
                send,
                req.id,
                -32602,
                format!("invalid pull params: {e}"),
                None,
            )
            .await;
            return;
        }
    };
    let request_id = params.request_id;
    let target = crate::download::PullTarget {
        repo: params.repo,
        file: params.file,
        revision: params.revision.unwrap_or_else(|| "main".into()),
    };
    tracing::info!(
        repo = %target.repo,
        file = %target.file,
        revision = %target.revision,
        dest = %models_root.display(),
        "higgs node: pull requested by hub"
    );

    // Run the download in its own task so a hub disconnect doesn't abort a near-complete
    // pull. Progress flows over a BOUNDED channel; when the hub stream is back-pressured the
    // download drops surplus ticks (`try_send`) rather than buffering every chunk — progress
    // is lossy-tolerant, so memory stays bounded regardless of model size.
    // The task logs its OWN lifecycle (start/progress deciles/outcome): the log record must
    // survive even when the hub connection is long gone by the time the transfer resolves.
    // Owned copies for log lines: the target moves into the download task and
    // is mutably-borrow-blocked inside it, while the handler needs the same
    // context on its disconnect lines (concurrent pulls must never interleave
    // ambiguously in the log).
    let (log_repo, log_file) = (target.repo.clone(), target.file.clone());
    let (prog_tx, mut prog_rx) = tokio::sync::mpsc::channel::<(u64, Option<u64>)>(64);
    let (final_tx, final_rx) = tokio::sync::oneshot::channel();
    tokio::spawn(async move {
        let started = std::time::Instant::now();
        let (cb_repo, cb_file) = (target.repo.clone(), target.file.clone());
        let mut last_logged_step: u64 = 0;
        let mut last_downloaded: u64 = 0;
        let mut cb = |downloaded: u64, total: Option<u64>| {
            last_downloaded = downloaded;
            // Every 10% when the length is known, else every GiB — enough to
            // follow a transfer from the log without flooding it per-chunk.
            let step = match total {
                Some(t) if t > 0 => downloaded * 10 / t,
                _ => downloaded >> 30,
            };
            if step > last_logged_step {
                last_logged_step = step;
                match total {
                    Some(t) if t > 0 => tracing::info!(
                        repo = %cb_repo,
                        file = %cb_file,
                        downloaded,
                        total = t,
                        percent = downloaded * 100 / t,
                        "higgs node: pull progress"
                    ),
                    _ => tracing::info!(
                        repo = %cb_repo,
                        file = %cb_file,
                        downloaded,
                        "higgs node: pull progress (length unknown)"
                    ),
                }
            }
            let _ = prog_tx.try_send((downloaded, total));
        };
        tracing::info!(repo = %target.repo, file = %target.file, "higgs node: pull download starting");
        let res =
            crate::download::download_dual(&target, &models_root, &primary, &fallback, &mut cb)
                .await;
        match &res {
            Ok(path) => tracing::info!(
                repo = %target.repo,
                file = %target.file,
                path = %path.display(),
                bytes = last_downloaded,
                elapsed_secs = started.elapsed().as_secs(),
                "higgs node: pull done — file on disk"
            ),
            Err(e) => tracing::warn!(
                repo = %target.repo,
                file = %target.file,
                bytes = last_downloaded,
                elapsed_secs = started.elapsed().as_secs(),
                error = %e,
                "higgs node: pull FAILED"
            ),
        }
        let _ = final_tx.send(res);
    });
    tokio::pin!(final_rx);

    let mut progress_open = true;
    let final_res: Result<std::path::PathBuf, HiggsError> = loop {
        tokio::select! {
            maybe = prog_rx.recv(), if progress_open => match maybe {
                Some((downloaded, total)) => {
                    if write_progress(send, request_id, downloaded, total).await.is_err() {
                        // Hub gone — the download task still finishes (and logs).
                        tracing::info!(
                            repo = %log_repo,
                            file = %log_file,
                            "higgs node: hub stream write failed mid-pull; download continues in the background"
                        );
                        return;
                    }
                }
                None => progress_open = false,
            },
            res = &mut final_rx => break res.unwrap_or_else(|_| {
                Err(HiggsError::DownloadFailed { repo: String::new(), file: String::new(), detail: "download task dropped".into() })
            }),
            _ = conn.closed() => {
                tracing::info!(
                    repo = %log_repo,
                    file = %log_file,
                    "higgs node: hub connection closed mid-pull; download continues in the background"
                );
                return;
            }
            _ = send.stopped() => {
                tracing::info!(
                    repo = %log_repo,
                    file = %log_file,
                    "higgs node: hub stopped the pull stream; download continues in the background"
                );
                return;
            }
        }
    };
    // Flush any progress buffered before the final resolved.
    while let Ok((downloaded, total)) = prog_rx.try_recv() {
        if write_progress(send, request_id, downloaded, total)
            .await
            .is_err()
        {
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
    let note = RpcNotification {
        jsonrpc: "2.0".into(),
        method: method.into(),
        params,
    };
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

/// Write one `N_CHAT_CHUNK` notification carrying the hub's `request_id` +
/// the tagged delta (additive `kind`/`tool` wire shape — an old hub reading
/// only `delta` degrades reasoning to content and tool fragments to "").
async fn write_chunk(
    send: &mut SendStream,
    request_id: u64,
    delta: &crate::worker::engine::ChatDelta,
) -> std::io::Result<()> {
    write_notification(
        send,
        N_CHAT_CHUNK,
        crate::worker::engine::ChatDelta::encode_chunk_params(
            &serde_json::json!(request_id),
            delta.kind,
            &delta.text,
        ),
    )
    .await
}

/// Write a successful `result` response for request `id`.
async fn reply_ok(send: &mut SendStream, id: u64, result: serde_json::Value) {
    let resp = RpcResponse {
        jsonrpc: "2.0".into(),
        id,
        result: Some(result),
        error: None,
    };
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
        error: Some(RpcError {
            code,
            message,
            data,
        }),
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
#[path = "data_tests.rs"]
mod tests;
