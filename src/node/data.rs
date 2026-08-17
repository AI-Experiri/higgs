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
        crate::catalog::cancel::node_registry(),
    )
    .await;
}

/// True iff a `DownloadLock::acquire` refusal is CONTENTION — another live
/// holder of the machine-wide slot (`HG090`) — rather than a filesystem
/// failure (`HG034`: locks dir uncreatable, lock file unopenable, flock
/// I/O error). Discriminates the daemon log line in [`pull_stream`] so an
/// I/O fault is never logged as "already in flight".
fn acquire_refusal_is_contention(e: &crate::diagnostic::HiggsError) -> bool {
    matches!(e, crate::diagnostic::HiggsError::DownloadInFlight { .. })
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
    cancels: &'static crate::catalog::cancel::PullCancelRegistry,
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
    // Register as cancellable BEFORE spawning — a duplicate (the hub re-issuing
    // a pull that is still running after a reconnect) is REFUSED right here on
    // the wire: it is the node's job to say "a download is already in
    // progress", never to start a second copy of the same transfer. The RAII
    // guard rides the download task and deregisters on every exit path;
    // `None` node: on the node process itself every pull is local; the
    // FUTURE `M_NODE_PULL_CANCEL` dispatch fires this key.
    // MIXED-VERSION NOTE: an OLD hub (pre-pull_status) that reconnects and
    // blindly re-issues gets one honest [HG090] error it cannot classify —
    // acceptable: in this fleet the hub (jigglebot embedding higgs at HEAD)
    // always upgrades before nodes, and even the old-hub case self-heals when
    // the orphan resolves.
    // ACCEPTED RESIDUAL: an ORPHANED transfer (hub disconnected mid-pull; the
    // task keeps going by design) holds the key until it resolves, so a
    // re-issue is refused ([HG090]) for the remainder — and because the
    // orphan's progress/final channels died with the old stream, the hub can
    // observe NEITHER progress NOR the outcome of that in-flight transfer
    // until it lands on disk (the next M_SCAN shows the file) or the cancel
    // dispatch (companion slice) kills it. The wedge self-heals on guard drop
    // — but a STALLED orphan (half-open TCP; the download path has no read
    // timeout, a pre-existing gap) holds the key until the transfer dies on
    // its own; the cancel dispatch is the recovery for that case too.
    // FOLLOW-UP: a throughput-based stall guard on the download path (like
    // self-update's) would bound the wedge without any cancel.
    // Validate the wire values FIRST (same rules the download applies) so the
    // registry keyspace only ever holds real, safe targets — garbage is
    // refused before it can occupy a cancel slot.
    if let Err(e) = crate::download::dest_path(&models_root, &target.repo, &target.file) {
        reply_err(send, req.id, -32000, e.to_string(), hg_data(&e)).await;
        return;
    }
    // Revision too (same rule download_attempt enforces later): a malformed
    // revision must fail HERE, before it can occupy a cancel-registry slot
    // and refuse a legitimate same-key pull with [HG090] until the doomed
    // task runs and dies.
    if !target
        .revision
        .split('/')
        .all(crate::download::is_safe_segment)
    {
        let e = HiggsError::DownloadFailed {
            repo: target.repo.clone(),
            file: target.file.clone(),
            detail: format!(
                "invalid revision {:?} (segments must be [A-Za-z0-9._-])",
                target.revision
            ),
        };
        reply_err(send, req.id, -32000, e.to_string(), hg_data(&e)).await;
        return;
    }
    // MACHINE-WIDE DOWNLOAD AUTHORITY: acquire the download-lock BEFORE the
    // node-local cancel registry insert. Otherwise there is a brief window
    // (registry inserted but flock not yet acquired) where
    // `announced_downloads()` reports this key as `cancellable: true` with
    // zero progress, suppressing the REAL foreign owner's ledger row in the
    // fleet view. With the lock held first, an announced `cancellable: true`
    // always means we own the machine-wide slot for the key.
    // The lock guard is MOVED into the download task below and lives for
    // the transfer's entire lifetime; the kernel drops it on any exit.
    let dl_lock = match crate::catalog::download_lock::DownloadLock::acquire(
        &models_root,
        &target.repo,
        &target.file,
    ) {
        Ok(l) => l,
        Err(e) => {
            // Discriminate the daemon log by CAUSE: acquire also fails on
            // I/O faults (locks dir uncreatable, lock file unopenable →
            // HG034), and logging those as "already in flight" sends the
            // operator hunting a phantom duplicate transfer instead of the
            // filesystem fault. The wire reply already carries the real
            // structured error either way.
            if acquire_refusal_is_contention(&e) {
                tracing::warn!(
                    repo = %target.repo,
                    file = %target.file,
                    "higgs node: pull refused — already in flight (machine-wide download lock held)"
                );
            } else {
                tracing::warn!(
                    repo = %target.repo,
                    file = %target.file,
                    error = %e,
                    "higgs node: pull refused — download-lock acquire failed"
                );
            }
            reply_err(send, req.id, -32000, e.to_string(), hg_data(&e)).await;
            return;
        }
    };
    let (cancel_guard, cancelled, pull_progress) =
        match crate::catalog::cancel::PullCancelRegistry::register(
            cancels,
            None,
            &target.repo,
            &target.file,
        ) {
            Ok(x) => x,
            Err(e) => {
                tracing::warn!(
                    repo = %target.repo,
                    file = %target.file,
                    "higgs node: pull refused — cancel registry conflict"
                );
                // Structured HG payload like every other pull error — the hub
                // distinguishes "already downloading" from a genuine failure by
                // code, never by string-matching the message.
                reply_err(send, req.id, -32000, e.to_string(), hg_data(&e)).await;
                return;
            }
        };
    // Owned copies for log lines: the target moves into the download task and
    // is mutably-borrow-blocked inside it, while the handler needs the same
    // context on its disconnect lines (concurrent pulls must never interleave
    // ambiguously in the log).
    let (log_repo, log_file) = (target.repo.clone(), target.file.clone());
    let (prog_tx, mut prog_rx) = tokio::sync::mpsc::channel::<(u64, Option<u64>)>(64);
    let (final_tx, final_rx) = tokio::sync::oneshot::channel();
    tokio::spawn(async move {
        // ORDER MATTERS: in Rust, locals drop in reverse declaration order.
        // Declare the machine-wide flock FIRST so it drops LAST (after all
        // logging + the cancel-registry deregistration below), and the
        // cancel guard SECOND so it drops FIRST. This closes the race
        // where the flock could be released before the cancel-registry
        // row is torn down — during which `announced_downloads()` would
        // publish `cancellable: true` for a slot the process no longer
        // owns.
        let dl_lock = dl_lock;
        let _cancel_guard = cancel_guard;
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
            // Feed the registry's live counters too — what M_NODE_PULL_STATUS
            // reports to a (re)connecting hub ("a download is already going,
            // here is how far along it is").
            pull_progress.set(downloaded, total);
            let _ = prog_tx.try_send((downloaded, total));
        };
        tracing::info!(repo = %target.repo, file = %target.file, "higgs node: pull download starting");
        // IMMEDIATE registration tick: a zero-byte progress frame the moment
        // the transfer is registered + about to run, BEFORE the first HF
        // byte. The hub uses its FIRST progress frame as wire attestation
        // ("the node really started this transfer") for the drop-guard's
        // HG089-vs-HG090 choice — without this tick, a registered transfer
        // with a slow time-to-first-byte (cold HF connection) dropped early
        // would read as "never started" (HG089) despite running. Same
        // notification channel progress already rides; an old hub just sees
        // one extra 0-byte tick.
        pull_progress.set(0, None);
        let _ = prog_tx.try_send((0, None));
        let res = crate::catalog::cancel::cancellable_pull(
            crate::download::download_dual_locked(
                &target,
                &models_root,
                &primary,
                &fallback,
                &mut cb,
                &dl_lock,
            ),
            cancelled,
            &models_root,
            &target.repo,
            &target.file,
        )
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
            // Cancelled ≠ failed, in the node log too: an operator cancel is
            // an info event, never a warning that trips log-based alerts.
            Err(e @ HiggsError::DownloadCancelled { .. }) => tracing::info!(
                repo = %target.repo,
                file = %target.file,
                bytes = last_downloaded,
                elapsed_secs = started.elapsed().as_secs(),
                detail = %e,
                "higgs node: pull cancelled"
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
        // Free the cancel key BEFORE the result becomes observable on the
        // wire: the handler replies the moment final_rx resolves, and a hub
        // that SAW success may instantly re-issue — it must not be refused
        // "already in flight" by a key whose transfer it knows is done.
        drop(_cancel_guard);
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

/// Everything this MACHINE is downloading, for the node's announcement
/// surfaces (the HELLO `downloads` field and the `M_NODE_PULL_STATUS`
/// reply): this process's cancel registry (live byte counters) UNION the
/// machine ledger's live entries from OTHER processes — e.g. a
/// `higgs download` CLI run on this box, its progress as of its last
/// throttled ledger write — so the fleet sees MACHINE truth, not process
/// truth. Registry entries win a key collision (their counters are fresher
/// and lock-free); collision is CASE-FOLDED, matching the machine
/// download-lock's key fold — on the default case-insensitive APFS a
/// case-variant identity is the same on-disk file and the same lock slot,
/// so a stale case-variant ledger row must not announce beside the live
/// registry row. The list is bounded to 16, the HELLO-frame producer
/// bound. A missing/unreadable ledger degrades to registry-only (status is
/// best-effort, never a failure source).
pub(crate) fn announced_downloads() -> Vec<crate::remote::HelloDownload> {
    let mut out: Vec<crate::remote::HelloDownload> = crate::catalog::cancel::node_registry()
        .in_flight()
        .into_iter()
        .map(|p| crate::remote::HelloDownload {
            repo: p.repo,
            file: p.file,
            downloaded: p.downloaded,
            total: p.total,
            // Registry-backed: the node's OWN process has the cancel
            // channel via `catalog::cancel::node_registry`.
            cancellable: true,
        })
        .collect();
    if let Ok(root) = crate::download::models_dir() {
        for e in crate::catalog::ledger::read_live(&root) {
            if e.pid != std::process::id()
                && !out.iter().any(|d| {
                    d.repo.eq_ignore_ascii_case(&e.repo) && d.file.eq_ignore_ascii_case(&e.file)
                })
            {
                out.push(crate::remote::HelloDownload {
                    repo: e.repo,
                    file: e.file,
                    downloaded: e.downloaded,
                    total: e.total,
                    // LEDGER-only: another process on this box owns it;
                    // this node has no cancel channel into that process.
                    cancellable: false,
                });
            }
        }
    }
    // PRODUCER-side validate-or-drop (the same predicate the hub applies):
    // ledger rows are file content, and a semantically-bad row with a huge
    // repo/file would inflate our OWN HELLO/status frame past the 64 KiB
    // caps — a bad status file must degrade to a smaller announcement, never
    // cost this node its admission. Also enforces the 16-cap.
    crate::remote::accept_announced_downloads(&out)
}
