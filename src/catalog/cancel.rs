//! In-flight catalog-download CANCEL registry: the seam that lets an operator
//! abort a running pull and get the partial file deleted, on whichever machine
//! is downloading.
//!
//! ONE instance exists in a running system today: the process-global
//! [`node_registry`], which hub-triggered `M_NODE_PULL` transfers register
//! into (the FUTURE `M_NODE_PULL_CANCEL` dispatch fires it; today nothing
//! triggers node-side cancels). LOCAL `Higgs::model_download*` transfers
//! do NOT register here — their duplicate gate is the facade's
//! `downloads_in_flight` slot set plus the machine-wide download lock; the
//! FUTURE cancel-dispatch slice adds a facade-held registry instance and
//! the `Higgs::model_download_cancel` trigger when local cancels ship.
//!
//! The registry maps `(node, repo, file)` → a `watch` stop channel (the
//! `LogWatchShared` pattern — no `tokio_util` dependency). A pull REGISTERS on
//! start (RAII guard, deregistered on every exit path) and runs its download
//! future inside [`cancellable_pull`], which aborts + cleans up when the channel
//! fires. Cancelling a key nobody registered is a clean coded error, never a
//! panic — the transfer may have just finished, and the caller deserves to know.
//!
//! INVARIANT: [HG089] means NOTHING landed. [`cancellable_pull`]'s `select!`
//! is `biased` toward the download arm, so a completed transfer (whose rename
//! runs in the same no-await poll as its final `Ready`) always wins over a
//! same-cycle cancel — the cancel arm only ever fires while the future is
//! still pending, before any rename.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use tokio::sync::watch;

use crate::diagnostic::HiggsError;

/// One in-flight pull's identity: `(target node, repo, file)` — `None` node =
/// the local machine. Matches the facade's in-flight-download slot key.
pub type PullKey = (Option<String>, String, String);

/// Registry of cancellable in-flight pulls. Cheap and lock-only; safe to hold
/// on the facade or as a process-global.
#[derive(Default)]
pub struct PullCancelRegistry {
    inner: parking_lot::Mutex<HashMap<PullKey, Entry>>,
}

struct Entry {
    tx: watch::Sender<bool>,
    progress: std::sync::Arc<PullProgress>,
}

/// Live byte counters for one in-flight pull, updated lock-free from the
/// download's progress callback and read by [`PullCancelRegistry::in_flight`]
/// — the numbers a reconnecting hub sees when the node announces "a download
/// is already going, here is how far along it is".
#[derive(Default, Debug)]
pub struct PullProgress {
    downloaded: std::sync::atomic::AtomicU64,
    /// 0 = length unknown (the server sent no content length).
    total: std::sync::atomic::AtomicU64,
}

impl PullProgress {
    /// Record the latest `(downloaded, total)` tick.
    pub fn set(&self, downloaded: u64, total: Option<u64>) {
        use std::sync::atomic::Ordering::Relaxed;
        self.downloaded.store(downloaded, Relaxed);
        self.total.store(total.unwrap_or(0), Relaxed);
    }
}

/// One in-flight pull as reported to the hub: identity + live progress.
#[derive(Debug, Clone, PartialEq)]
pub struct InFlightPull {
    pub node: Option<String>,
    pub repo: String,
    pub file: String,
    pub downloaded: u64,
    pub total: Option<u64>,
}

impl PullCancelRegistry {
    /// An empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a starting pull as cancellable. Returns the RAII guard (drop =
    /// deregister — keep it alive for the whole transfer) and the receiver the
    /// pull's `select!` watches. A DUPLICATE key is REJECTED ([HG090] "already
    /// in flight"): one transfer per (node, repo, file) — a hub re-issue after
    /// a reconnect races the still-running original and must not start a
    /// second copy; the caller surfaces the refusal, the user waits or cancels
    /// the original.
    pub fn register<'a>(
        &'a self,
        node: Option<&str>,
        repo: &str,
        file: &str,
    ) -> Result<
        (
            PullCancelGuard<'a>,
            watch::Receiver<bool>,
            std::sync::Arc<PullProgress>,
        ),
        HiggsError,
    > {
        let key = (node.map(str::to_owned), repo.to_owned(), file.to_owned());
        let mut map = self.inner.lock();
        if map.contains_key(&key) {
            // [HG090], NOT the HG025 failure umbrella: the hub must be able to
            // classify "already downloading" by code alone (wait/info in the
            // UI, never an error toast).
            return Err(HiggsError::DownloadInFlight {
                repo: repo.to_owned(),
                file: file.to_owned(),
            });
        }
        let (tx, rx) = watch::channel(false);
        let progress = std::sync::Arc::new(PullProgress::default());
        map.insert(
            key.clone(),
            Entry {
                tx,
                progress: progress.clone(),
            },
        );
        drop(map);
        Ok((
            PullCancelGuard {
                registry: self,
                key,
            },
            rx,
            progress,
        ))
    }

    /// Every in-flight pull with its live progress — what the node ANNOUNCES
    /// to a (re)connecting hub so a running download is continued, not
    /// duplicated (`M_NODE_PULL_STATUS`). BOUNDED to 16 entries at the
    /// PRODUCER: the announcement rides HELLO, whose frame is capped at
    /// 64 KiB — an unbounded list could make the HELLO unparseable and lock
    /// the node out of reconnecting (the exact wedge this exists to prevent).
    /// 16 concurrent multi-GB pulls is already beyond any real fleet use.
    pub fn in_flight(&self) -> Vec<InFlightPull> {
        use std::sync::atomic::Ordering::Relaxed;
        self.inner
            .lock()
            .iter()
            .take(16)
            .map(|((node, repo, file), e)| InFlightPull {
                node: node.clone(),
                repo: repo.clone(),
                file: file.clone(),
                downloaded: e.progress.downloaded.load(Relaxed),
                total: match e.progress.total.load(Relaxed) {
                    0 => None,
                    t => Some(t),
                },
            })
            .collect()
    }

    /// Fire the cancel for a registered pull. `Err` ([HG025], "no in-flight
    /// download") when nothing is registered under the key — the transfer never
    /// started, already finished, or was already cancelled.
    ///
    /// DELIBERATE: the key carries no epoch. A cancel means "stop downloading
    /// this file on this node" — if the transfer the operator saw finished and
    /// a fresh one for the same file started, cancelling the current one IS
    /// the operator's intent (duplicates are refused at register, so at most
    /// one transfer per key exists).
    pub fn cancel(&self, node: Option<&str>, repo: &str, file: &str) -> Result<(), HiggsError> {
        let key = (node.map(str::to_owned), repo.to_owned(), file.to_owned());
        let no_in_flight = || {
            // Deliberate: a coded error (not silent Ok) — the caller must be
            // able to tell "cancelled a running transfer" from "nothing to
            // cancel" (it likely just finished). How the DISPATCH slice maps
            // this benign race for the UI (info vs error) is its call.
            Err(HiggsError::DownloadFailed {
                repo: repo.to_owned(),
                file: file.to_owned(),
                detail: "no in-flight download to cancel".to_owned(),
            })
        };
        match self.inner.lock().get(&key) {
            Some(entry) => {
                if entry.tx.send(true).is_ok() {
                    Ok(())
                } else {
                    // The receiver is gone — `cancellable_pull` already
                    // resolved and only the guard's drop hasn't landed yet.
                    // Reporting Ok would claim "cancel accepted" for a
                    // transfer whose terminal result is already decided, so
                    // answer honestly — but do NOT remove the entry: the live
                    // guard is its SOLE owner, and sweeping here would let a
                    // fresh same-key registration slip in before that guard's
                    // unconditional drop, which would then delete the NEW
                    // pull's sender (ABA). The stale entry lives only for the
                    // instants between the pull resolving and the guard drop.
                    no_in_flight()
                }
            }
            None => no_in_flight(),
        }
    }

    /// Whether `key` is currently registered (test/introspection helper).
    pub fn is_registered(&self, node: Option<&str>, repo: &str, file: &str) -> bool {
        let key = (node.map(str::to_owned), repo.to_owned(), file.to_owned());
        self.inner.lock().contains_key(&key)
    }
}

/// RAII registration of one pull; dropping deregisters, so a finished (or
/// panicked/dropped) transfer can no longer be "cancelled".
pub struct PullCancelGuard<'a> {
    registry: &'a PullCancelRegistry,
    key: PullKey,
}

impl Drop for PullCancelGuard<'_> {
    fn drop(&mut self) {
        // Duplicates are rejected at register, so this guard is the sole
        // owner of its key — unconditional removal is safe.
        self.registry.inner.lock().remove(&self.key);
    }
}

/// The NODE process's registry for hub-triggered `M_NODE_PULL` transfers — a
/// process-global because one node process serves one models dir, and the
/// FUTURE `M_NODE_PULL_CANCEL` control dispatch has no shared struct with the
/// data plane's pull relay. The hub facade does NOT use this (its duplicate
/// gate is the `downloads_in_flight` slot set; a facade registry arrives
/// with the cancel-dispatch slice). KEY TRANSLATION (contract for the future dispatch slice): node
/// pulls register here under `node = None` (on the node process every pull
/// is local), while the HUB tracks the same transfer under
/// `Some(endpoint_id)`. That dispatch must fire this registry with `None` —
/// forwarding the hub-side `Some(node)` key verbatim would never match.
pub fn node_registry() -> &'static PullCancelRegistry {
    static REGISTRY: std::sync::OnceLock<PullCancelRegistry> = std::sync::OnceLock::new();
    REGISTRY.get_or_init(PullCancelRegistry::new)
}

/// Run a download future to completion UNLESS `cancelled` fires first: on
/// cancel the future is dropped (aborting the transfer mid-stream) and
/// [HG089] `DownloadCancelled` is returned. Temp cleanup is the dropped
/// future's OWN per-attempt `TempGuard` (it unlinks exactly the tmp this
/// transfer created as the future drops) — never a blanket sweep here; an
/// unlink failure is visible only in tracing. The caller maps the error
/// onto the terminal `Cancelled` download-event phase.
pub async fn cancellable_pull<F>(
    fut: F,
    mut cancelled: watch::Receiver<bool>,
    models_root: &Path,
    repo: &str,
    file: &str,
) -> Result<PathBuf, HiggsError>
where
    F: std::future::Future<Output = Result<PathBuf, HiggsError>>,
{
    // Cancel handling has TWO regimes with different invariants:
    //
    // (a) A cancel accepted BEFORE the future's next poll is honored
    //     deterministically: nothing has run yet, so HG089's "nothing landed"
    //     invariant is trivially true. We check the cancel state before EVERY
    //     poll (fast path + inside the loop) so a signal arriving between the
    //     first check and `select!`'s first poll is still caught before the
    //     future runs.
    //
    // (b) Mid-flight photo-finish (cancel and completion race the same poll
    //     cycle) resolves in favor of COMPLETION via the biased `select!`:
    //     if the future already finished the atomic rename, HG089 would lie.
    //     Ok(path) wins ⇒ Done event, honest.
    // `partial_swept` documents cleanup done by the DROPPED future's own
    // per-transfer temp guard (`download`'s `TempGuard`) — not by a
    // blanket sweep here. A blanket `remove_partials` here would erase any
    // OTHER concurrent same-pid `.part.<pid>.<seq>` temps for this key
    // (e.g. a caller running `download` directly outside the cancellable
    // wrapper), which is exactly the r45 finding. The per-transfer drop
    // guard is precise: it only touches the tmp path THIS transfer created.
    let _ = models_root;
    // Attempt-start bound for the Err+cancel ledger reconciliation below:
    // only a Failed terminal ended AT/AFTER this instant can belong to THIS
    // attempt (the wrapped future hasn't run yet). Prior attempts' genuine
    // Failed history for the same key is never a rewrite candidate.
    let attempt_started_ms = crate::catalog::ledger::unix_now_ms();
    let sweep_and_hg089 = || {
        // (The dropped future's own ledger claim guard records its
        // `cancelled` terminal as it drops with the future; the temp guard
        // unlinks the specific tmp — nothing to sweep here.)
        HiggsError::DownloadCancelled {
            repo: repo.to_owned(),
            file: file.to_owned(),
            partial_swept: true,
        }
    };
    tokio::pin!(fut);
    // TWO REGIMES, both LOAD-BEARING:
    //
    // (a) Cancel observed BEFORE the future is polled → HG089 (deterministic).
    //     A pre-poll re-check catches a cancel that races into the window
    //     between the entry check and the first `poll_immediate`.
    //
    // (b) Mid-flight photo-finish (cancel and completion race the same poll
    //     cycle) → biased-toward-completion so HG089 never lies about
    //     "nothing landed" when the file is on disk.
    //
    // ACCEPTED RESIDUAL (documented in convergence, clustered-findings
    // rule fired r43): a cancel signaled after `poll_immediate` returns
    // None but before the biased select's first poll can lose to a future
    // that becomes Ready on that same poll (e.g. an in-tokio wakeup from
    // a background task's fut-internal work). No production fetcher is
    // synthetic-instant-Ready (real HF/reqwest streams have `.await`
    // points; the moment the fetcher awaits, cancel wins the biased
    // select's cancel arm — regime (b) preserved). Making cancel
    // deterministically win in this window would REQUIRE inverting the
    // bias, which would let completion-wins-tie regress into HG089
    // reporting "nothing landed" while the file IS on disk — a stronger
    // lie than the residual. The residual manifests only as "cancel
    // returned Ok but the download completed anyway" — the user gets
    // what they asked for BEFORE cancelling.
    #[allow(clippy::never_loop)]
    loop {
        if *cancelled.borrow_and_update() {
            return Err(sweep_and_hg089());
        }
        if let Some(res) = futures::future::poll_immediate(&mut fut).await {
            // The future was Ready in its first poll. Race semantics with
            // a cancel that arrived during that same poll:
            //   * `Ok` + cancel  → completion won the tie (file landed);
            //     surface Ok but log HG091 so the operator sees why cancel
            //     didn't "stop" it.
            //   * `Err` + cancel → the transfer was failing at the same
            //     moment the operator asked to cancel. No file landed
            //     (TempGuard cleaned the tmp on future-drop, and the
            //     download path returns Err only for local-fs / exhausted-
            //     fallback cases where no rename happened). The operator's
            //     intent takes precedence — surface HG089, log the
            //     underlying fetcher error for forensics. Otherwise a UI
            //     that displayed the raw error would tell the operator
            //     their cancel "failed" while also showing a scary Failed
            //     row for a transfer they explicitly stopped.
            if *cancelled.borrow() {
                if let Err(e) = &res {
                    tracing::info!(
                        repo, file,
                        underlying = %e,
                        "cancel honored while download was also erroring — reporting HG089 (nothing landed)"
                    );
                    // The inner download's LedgerClaim::end already wrote
                    // `Failed{detail}` before returning this Err. Overwrite
                    // to Cancelled so the ledger row matches the surface
                    // (HG089), else `downloads_status` shows Failed for a
                    // transfer the caller saw as cancelled.
                    crate::catalog::ledger::overwrite_last_failed_to_cancelled(
                        models_root,
                        repo,
                        file,
                        attempt_started_ms,
                    );
                    return Err(sweep_and_hg089());
                }
                tracing::warn!(
                    error = %HiggsError::CancelLostToCompletion {
                        repo: repo.to_owned(),
                        file: file.to_owned(),
                    },
                    "cancel outraced by download completion"
                );
            }
            return res;
        }
        let res = tokio::select! {
            biased;
            res = &mut fut => res,
            Ok(_) = cancelled.wait_for(|c| *c) => return Err(sweep_and_hg089()),
        };
        // Outrace check: if a cancel signal is now visible, translate the
        // outcome the same way as the poll_immediate path — Ok wins with
        // HG091 (file landed), Err collapses to HG089 (nothing landed;
        // operator asked to stop).
        if *cancelled.borrow() {
            if let Err(e) = &res {
                tracing::info!(
                    repo, file,
                    underlying = %e,
                    "cancel honored while download was also erroring — reporting HG089 (nothing landed)"
                );
                // Same ledger reconciliation as the poll_immediate path:
                // the inner download already recorded Failed{raw error};
                // flip it to Cancelled so status matches the surface.
                crate::catalog::ledger::overwrite_last_failed_to_cancelled(
                    models_root,
                    repo,
                    file,
                    attempt_started_ms,
                );
                return Err(sweep_and_hg089());
            }
            tracing::warn!(
                error = %HiggsError::CancelLostToCompletion {
                    repo: repo.to_owned(),
                    file: file.to_owned(),
                },
                "cancel outraced by download completion"
            );
        }
        return res;
    }
}

#[cfg(test)]
#[path = "cancel_tests.rs"]
mod tests;
