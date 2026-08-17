//! Model downloader (P4b, `M_PULL`): fetch a GGUF from HuggingFace into the higgs models dir
//! (`~/.higgs/models/`) ONLY — never a scanned LM-Studio/Ollama dir. The byte source is
//! abstracted behind [`Fetcher`] so the path/guard/atomic-write logic is unit-tested without
//! network; the production [`HttpFetcher`] streams from the HF `resolve` URL with progress.
//!
//! Writes are atomic: bytes go to a `.part` temp and are renamed onto the final path only on
//! success, so a `M_SCAN`/`M_LOAD` never sees a half-written GGUF, and a failed download
//! leaves no partial file in the catalog.

use std::path::{Component, Path, PathBuf};

use crate::diagnostic::HiggsError;

/// What to pull: a HuggingFace `repo` + `file`, at `revision` (default `"main"`).
#[derive(Debug, Clone)]
pub struct PullTarget {
    pub repo: String,
    pub file: String,
    pub revision: String,
}

impl PullTarget {
    /// A target on `main` for `repo`/`file`.
    pub fn new(repo: impl Into<String>, file: impl Into<String>) -> Self {
        Self {
            repo: repo.into(),
            file: file.into(),
            revision: "main".into(),
        }
    }
}

/// The direct-download URL for a repo file at a revision. The base is HuggingFace by default,
/// overridable via `HIGGS_HF_ENDPOINT` (a mirror, an enterprise proxy, or a test server) —
/// trailing slash optional.
pub fn hf_url(repo: &str, revision: &str, file: &str) -> String {
    let base = std::env::var("HIGGS_HF_ENDPOINT")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "https://huggingface.co".into());
    let base = base.trim_end_matches('/');
    format!("{base}/{repo}/resolve/{revision}/{file}")
}

/// A byte source for [`download`]. Streams the bytes of `target`, handing each chunk to
/// `on_chunk` and reporting `(downloaded, total_opt)` to `progress`. Returns a CLASSIFIED
/// [`HiggsError`] on failure (the specific `HG029`–`HG035` code), which [`download`]
/// propagates verbatim — so a 404 vs an auth refusal vs a network blip stays distinguishable
/// all the way up. Implementors resolve their own endpoint from `target` (the hub client from
/// the repo handle; [`HttpFetcher`] from the `resolve` URL).
pub trait Fetcher {
    fn fetch(
        &self,
        target: &PullTarget,
        on_chunk: &mut (dyn FnMut(&[u8]) + Send),
        progress: &mut (dyn FnMut(u64, Option<u64>) + Send),
    ) -> impl std::future::Future<Output = Result<(), HiggsError>> + Send;
}

/// Resolve the on-disk destination under `models_root`, enforcing the layout the model
/// scanner traverses (`<root>/<org>/<model>/<file>.gguf`) AND rejecting any path escape
/// (`..`, absolute, drive prefix) — the values come off the wire. A pull that doesn't fit
/// the scanned layout would download to a dir `M_SCAN`/`M_LOAD` never sees, so it's refused
/// up front rather than silently succeeding into a dead spot.
pub fn dest_path(models_root: &Path, repo: &str, file: &str) -> Result<PathBuf, HiggsError> {
    let bad = |detail: &str| HiggsError::DownloadFailed {
        repo: repo.to_string(),
        file: file.to_string(),
        detail: detail.to_string(),
    };
    // repo must be exactly "<org>/<model>", each a SAFE segment (HF's `[A-Za-z0-9._-]`
    // charset). `is_safe_segment` rejects empties, `.`/`..`, path separators, AND
    // URL-reserved characters (`#`, `?`, `%`, space, …) — so the segment is both a valid
    // local dir name and a literal URL path component needing no percent-encoding.
    // No segment may exceed what the filesystem can HOLD, refused UP FRONT with the typed
    // error rather than dying ENAMETOOLONG mid-transfer. Repo segments are directories:
    // plain NAME_MAX (255 bytes on the Unix/macOS targets). The FILE bound additionally
    // reserves the download's own temp suffix — bytes land in
    // `<file>.part.<u32 pid>.<u64 seq>` first (see the temp naming in `download`), whose
    // worst case is ".part." + 10 + "." + 20 = 37 bytes — so a name whose FINAL path is
    // legal can never be "accepted but uncreatable" at the temp create. This bound does
    // double duty: the hub's announcement validator
    // (`remote::accept_announced_downloads`) IS this function, so every pull a node can
    // actually register is also announceable in its HELLO/status (no
    // invisible-but-in-flight window from a node/hub rule drift).
    const NAME_MAX: usize = 255;
    const TEMP_SUFFIX_MAX: usize = ".part.".len() + 10 + ".".len() + 20;
    let segs: Vec<&str> = repo.split('/').collect();
    if segs.len() != 2
        || !segs
            .iter()
            .all(|s| is_safe_segment(s) && s.len() <= NAME_MAX)
    {
        return Err(bad(
            "repo must be '<org>/<model>' (each [A-Za-z0-9._-], ≤255 bytes, no '..'/reserved chars)",
        ));
    }
    // file must be a single safe `*.gguf` component (case-insensitive) — no subdirectory,
    // no escape, no URL-reserved chars.
    if !is_safe_segment(file)
        || file.len() > NAME_MAX - TEMP_SUFFIX_MAX
        || !file.to_ascii_lowercase().ends_with(".gguf")
    {
        return Err(bad(
            "file must be a single '*.gguf' name ([A-Za-z0-9._-], ≤218 bytes so its              '.part.<pid>.<seq>' temp fits NAME_MAX, no subdir)",
        ));
    }
    let rel = PathBuf::from(repo).join(file);
    if rel.components().any(|c| {
        matches!(
            c,
            Component::ParentDir | Component::RootDir | Component::Prefix(_)
        )
    }) {
        return Err(bad("path must be relative with no '..'"));
    }
    Ok(models_root.join(rel))
}

/// Sweep the `.part` temps of ONE download destination — the cancel path's
/// cleanup, where the aborted transfer's own error-path removal never ran
/// (its future was dropped mid-poll). Validates `repo`/`file` through
/// [`dest_path`] (same escape/`*.gguf` rules as the download itself), then
/// removes every `<stem>.part.<pid>.<n>` sibling of the destination. The
/// final file and other downloads' temps are untouched; a missing model dir
/// is a clean no-op (nothing was ever written). Temps embed the FULL file
/// name, so even case-variant twins (`m.gguf` vs `m.GGUF`) have disjoint
/// temp namespaces — no cross-sweep exists.
///
/// INVARIANT (load-bearing): at most ONE download registry per process. The
/// pid scoping isolates PROCESSES; within a process the registry's duplicate
/// refusal guarantees one transfer per file. That holds today because the
/// node data plane (`M_NODE_PULL` → `node_registry()`) only ever runs in the
/// standalone node daemon, never in the embedded hub (whose facade keeps its
/// own in-flight set for LOCAL downloads). If those ever co-reside in one
/// process, a cancel sweep here could unlink the facade's live temp for the
/// same file — re-key the temps by registry before allowing that shape.
pub fn remove_partials(models_root: &Path, repo: &str, file: &str) -> Result<(), HiggsError> {
    let dest = dest_path(models_root, repo, file)?;
    let Some(dir) = dest.parent() else {
        return Ok(()); // dest_path always yields a parent; defensive no-op
    };
    // Sweep ONLY this process's own temps: `<file>.part.<OUR pid>.<seq>`.
    // The pid in the temp name exists for cross-process uniqueness — a node
    // daemon, the hub facade, and a `higgs download` CLI can all pull the SAME
    // file into the same dir concurrently (separate processes, separate
    // registries, the duplicate refusal cannot see across them). A pid-blind
    // sweep would unlink a sibling process's LIVE temp, making its healthy
    // transfer fail at the final rename after moving the whole file. The exact
    // `<digits>` seq check also protects a REAL model named
    // `<file>.part.<x>.gguf` (repo/file are wire values; `.` is legal).
    let prefix = format!("{file}.part.{}.", std::process::id());
    let is_temp_suffix = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        // Missing dir = nothing was ever written; any OTHER error (perms,
        // I/O) must not be silent — a multi-GB .part could be left behind
        // while [HG089] claims the partial was swept.
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => {
            return Err(HiggsError::DownloadFailed {
                repo: repo.to_owned(),
                file: file.to_owned(),
                detail: format!("partial sweep: cannot read model dir: {e}"),
            });
        }
    };
    for entry in entries {
        // An iteration error means the scan is INCOMPLETE — surface it (the
        // caller then reports partial_swept=false) instead of claiming a
        // clean sweep over files we never saw.
        let entry = entry.map_err(|e| HiggsError::DownloadFailed {
            repo: repo.to_owned(),
            file: file.to_owned(),
            detail: format!("partial sweep: readdir failed mid-scan: {e}"),
        })?;
        let name = entry.file_name();
        if name
            .to_str()
            .and_then(|n| n.strip_prefix(&prefix))
            .is_some_and(is_temp_suffix)
        {
            if let Err(e) = std::fs::remove_file(entry.path()) {
                // First failure wins: the caller reports the sweep as failed
                // ([HG089] then says so) rather than silently leaving a
                // multi-GB temp behind under a "swept" claim.
                return Err(HiggsError::DownloadFailed {
                    repo: repo.to_owned(),
                    file: file.to_owned(),
                    detail: format!(
                        "partial sweep: could not remove {}: {e}",
                        entry.path().display()
                    ),
                });
            }
        }
    }
    Ok(())
}

/// A safe path/URL segment: non-empty, not `.`/`..`, and only `[A-Za-z0-9._-]` — which is
/// HuggingFace's repo/file charset. This deliberately excludes `/`, `\`, and every
/// URL-reserved character (`#`, `?`, `%`, `:`, space, …) so a segment is BOTH a valid local
/// filename and a literal URL path component (no percent-encoding needed, no fragment/query
/// injection into the resolve URL).
pub(crate) fn is_safe_segment(s: &str) -> bool {
    !s.is_empty()
        && s != "."
        && s != ".."
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
}

/// Where a [`download`] attempt failed, by SOURCE — so [`download_dual`] can decide whether a
/// fallback retry could possibly help. The DISTINCTION IS THE SOURCE, NOT THE ERROR VARIANT:
/// the `HG034` (`HubFileWrite`) code is overloaded — `download`'s OWN write into
/// `~/.higgs/models` is `Local` (deterministic; a retry fails identically), but the hub
/// client's cache I/O surfaces the SAME code through the fetcher and is `Fetcher` (the direct
/// `reqwest` fallback bypasses that cache and may still succeed). A variant-only heuristic
/// can't tell those apart; this enum carries the source down from where it's known.
enum AttemptError {
    /// `download`'s own pre/around-fetch failure (wire-validation `HG025`, or a local
    /// filesystem `HG034`). Deterministic — `download_dual` returns it verbatim, no retry.
    Local(HiggsError),
    /// The fetcher failed (classified `HG029`–`HG035`, incl. the hub client's own I/O).
    /// Eligible for the dual-path fallback.
    Fetcher(HiggsError),
}

impl AttemptError {
    /// The underlying error, discarding the source tag (for the single-fetcher `download`).
    fn into_inner(self) -> HiggsError {
        match self {
            AttemptError::Local(e) | AttemptError::Fetcher(e) => e,
        }
    }
}

/// Single-fetcher download primitive. Takes the machine flock AND records
/// a full ledger claim (Downloading → progress → terminal), exactly like
/// [`download_dual`], but without the fallback retry — the fetcher's
/// classified error propagates verbatim (`HG029`–`HG035`), never wrapped
/// in `HG036`.
///
/// **`pub(crate)`** — external callers should use [`download_dual`]. The
/// old public `download` was a footgun: acquiring the flock without a
/// ledger claim left a transfer visible to other acquirers via HG090 but
/// invisible in `downloads_status` / `announced_downloads`.
///
/// Production paths all route through `download_dual`; this single-fetcher
/// contract is exercised by the unit suite (verbatim-error propagation,
/// atomicity, TempGuard) — hence the not(test) allow.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) async fn download<F: Fetcher>(
    target: &PullTarget,
    models_root: &Path,
    fetcher: &F,
    progress: &mut (dyn FnMut(u64, Option<u64>) + Send),
) -> Result<PathBuf, HiggsError> {
    // Full wire-identity validation BEFORE any side effect (same order as
    // `download_dual`): a refused pull leaves no lock file and no ledger row.
    dest_path(models_root, &target.repo, &target.file)?;
    if !target.revision.split('/').all(is_safe_segment) {
        return Err(HiggsError::DownloadFailed {
            repo: target.repo.clone(),
            file: target.file.clone(),
            detail: format!(
                "invalid revision {:?} (segments must be [A-Za-z0-9._-])",
                target.revision
            ),
        });
    }
    let _dl_lock = crate::catalog::download_lock::DownloadLock::acquire(
        models_root,
        &target.repo,
        &target.file,
    )?;
    let ledger_claim =
        crate::catalog::ledger::claim_start(models_root, &target.repo, &target.file)?;
    // Throttle the mirror (same 8 MiB / regression / total-change rules
    // as `download_dual`, minus the fallback-restart branch since there's
    // no fallback here).
    let mut last_written: u64 = 0;
    let mut last_total: Option<u64> = None;
    let mut last_seen: (u64, Option<u64>) = (0, None);
    let mut ledger_progress = |downloaded: u64, total: Option<u64>| {
        last_seen = (downloaded, total);
        let regressed = downloaded < last_written;
        let total_changed = total != last_total;
        if regressed || total_changed || downloaded.saturating_sub(last_written) >= 8 * 1024 * 1024
        {
            last_written = downloaded;
            last_total = total;
            crate::catalog::ledger::record_progress(
                models_root,
                &target.repo,
                &target.file,
                downloaded,
                total,
            );
        }
        progress(downloaded, total);
    };
    let result = download_attempt(target, models_root, fetcher, &mut ledger_progress)
        .await
        .map_err(AttemptError::into_inner);
    match &result {
        Ok(path) => ledger_claim.end(
            crate::catalog::ledger::LedgerEnd::Done {
                path: path.display().to_string(),
            },
            Some(last_seen),
        ),
        Err(e) => ledger_claim.end(
            crate::catalog::ledger::LedgerEnd::Failed {
                detail: e.to_string(),
            },
            Some(last_seen),
        ),
    }
    result
}

/// [`download`] but tagging each failure with its [`AttemptError`] source, for [`download_dual`].
async fn download_attempt<F: Fetcher>(
    target: &PullTarget,
    models_root: &Path,
    fetcher: &F,
    progress: &mut (dyn FnMut(u64, Option<u64>) + Send),
) -> Result<PathBuf, AttemptError> {
    // Wire-validation failures (bad repo/file/revision shape) — HG025, always LOCAL.
    let bad_request = |detail: String| {
        AttemptError::Local(HiggsError::DownloadFailed {
            repo: target.repo.clone(),
            file: target.file.clone(),
            detail,
        })
    };
    // OUR OWN filesystem failures (temp / write / fsync / rename into ~/.higgs/models) — HG034,
    // always LOCAL (a fallback's write to the same dir fails identically).
    let fs_fail = |detail: String| {
        AttemptError::Local(HiggsError::HubFileWrite {
            repo: target.repo.clone(),
            file: target.file.clone(),
            detail,
        })
    };

    let dest = dest_path(models_root, &target.repo, &target.file).map_err(AttemptError::Local)?;
    // The revision rides the URL path too but isn't part of the local dest, so validate it
    // here: each `/`-separated segment must be safe (allows branch paths like `refs/pr/1`),
    // keeping reserved chars out of the resolve URL.
    if !target.revision.split('/').all(is_safe_segment) {
        return Err(bad_request(format!(
            "invalid revision {:?} (segments must be [A-Za-z0-9._-])",
            target.revision
        )));
    }

    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent).map_err(|e| fs_fail(e.to_string()))?;
    }
    // A UNIQUE temp per download (pid + process-global counter), so two concurrent pulls to
    // the same destination never share a `.part` and corrupt each other; the last atomic
    // rename wins and the file is always whole.
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    // The temp APPENDS to the FULL file name (`m.gguf.part.<pid>.<seq>`) —
    // never `with_extension`, which would strip `.gguf` and collide the
    // case-variant twins `m.gguf`/`m.GGUF` onto one `m.part.*` namespace
    // (distinct files on HF and on case-sensitive filesystems; a cancel sweep
    // scoped by that shared stem would unlink the sibling's live temp).
    let tmp = dest.with_file_name(format!("{}.part.{}.{n}", target.file, std::process::id()));
    // RAII cleanup for THIS transfer's temp: if `download_attempt`'s future
    // is dropped mid-poll (cancellable_pull cancel, task abort, caller
    // gone), this guard's drop unlinks our specific tmp. Load-bearing: the
    // cancel path used to blanket-sweep every same-pid temp for the key,
    // which would clip a concurrent caller's live temp; per-attempt drop
    // makes the cancel-path sweep unnecessary and safe.
    struct TempGuard {
        path: std::path::PathBuf,
        disarmed: bool,
    }
    impl Drop for TempGuard {
        fn drop(&mut self) {
            if !self.disarmed {
                match std::fs::remove_file(&self.path) {
                    Ok(()) => {}
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => {} // already gone
                    Err(e) => {
                        // HG089's message points operators to the node log
                        // on cleanup trouble — honor that contract here.
                        tracing::warn!(
                            path = %self.path.display(),
                            error = %e,
                            "download temp cleanup FAILED — the `.part` file may remain"
                        );
                    }
                }
            }
        }
    }
    let mut tmp_guard = TempGuard {
        path: tmp.clone(),
        disarmed: false,
    };

    let mut file = std::fs::File::create(&tmp).map_err(|e| fs_fail(e.to_string()))?;
    let mut write_err: Option<std::io::Error> = None;
    let fetch_res = {
        use std::io::Write;
        let mut on_chunk = |bytes: &[u8]| {
            if write_err.is_none() {
                if let Err(e) = file.write_all(bytes) {
                    write_err = Some(e);
                }
            }
        };
        fetcher.fetch(target, &mut on_chunk, progress).await
    };
    // A fetch failure (a classified HiggsError) is a FETCHER error — propagated verbatim and
    // eligible for the fallback. A local write failure is HG034 LOCAL. Either removes the temp
    // — never leaving a partial/empty `.part` behind.
    if let Err(e) = fetch_res {
        drop(file);
        let _ = std::fs::remove_file(&tmp);
        return Err(AttemptError::Fetcher(e));
    }
    if let Some(e) = write_err {
        drop(file);
        let _ = std::fs::remove_file(&tmp);
        return Err(fs_fail(format!("write failed: {e}")));
    }
    if let Err(e) = file.sync_all() {
        drop(file);
        let _ = std::fs::remove_file(&tmp);
        return Err(fs_fail(e.to_string()));
    }
    drop(file);
    // Atomic replace: `rename(2)` replaces the destination in place, so an existing model
    // stays intact until the instant it's swapped (no missing-window) and a FAILED rename
    // leaves the old model untouched — we just remove our temp. (higgs targets Unix/macOS;
    // the llama.cpp FFI build is not Windows-portable, so the non-atomic-rename platform is
    // out of scope.)
    if let Err(e) = std::fs::rename(&tmp, &dest) {
        let _ = std::fs::remove_file(&tmp);
        return Err(fs_fail(e.to_string()));
    }
    // Success: the file is at `dest`. Prevent the drop guard from unlinking
    // a nonexistent temp (or, worse, a same-name recycled path — the
    // rename moved the inode, but the guard has no way to know that).
    tmp_guard.disarmed = true;
    Ok(dest)
}

/// Download `target` trying `primary` first and `fallback` only if the primary fails with a
/// FETCHER error — the hub-client-primary / `reqwest`-fallback strategy. Each attempt is a
/// self-contained [`download_attempt`] (its own temp + atomic rename), so a failed primary
/// leaves NO partial file for the fallback to trip over.
///
/// A primary failure in `download`'s OWN code ([`AttemptError::Local`] — wire-validation
/// `HG025`, or a local filesystem `HG034` writing to `~/.higgs/models`) is deterministic: the
/// fallback would fail identically, so it is returned VERBATIM (preserving the actionable
/// `HG025`/`HG034` rather than a misleading `HG036`) and the fallback is NOT attempted — no
/// pointless re-download of a multi-GB model. A FETCHER failure ([`AttemptError::Fetcher`] —
/// network/auth/404/http, or the hub client's own cache I/O) DOES trigger the fallback; if
/// that also fails, returns [`HiggsError::HubFetchExhausted`] (`HG036`) carrying both diagnoses.
pub async fn download_dual<P: Fetcher, F: Fetcher>(
    target: &PullTarget,
    models_root: &Path,
    primary: &P,
    fallback: &F,
    progress: &mut (dyn FnMut(u64, Option<u64>) + Send),
) -> Result<PathBuf, HiggsError> {
    // Validate the WHOLE wire identity — `(repo, file)` via dest_path AND
    // the revision — before anything touches the filesystem: a refused pull
    // (HG025) must leave no trace (no lock file, no ledger row, no event
    // beyond what the caller emits). The revision check here mirrors
    // `download_attempt`'s (which stays as defense-in-depth for direct
    // `download` callers); hoisting it in front of the lock is what keeps
    // a malformed revision from creating a lock file + a Failed ledger row
    // as side effects of an input-validation refusal.
    dest_path(models_root, &target.repo, &target.file)?;
    if !target.revision.split('/').all(is_safe_segment) {
        return Err(HiggsError::DownloadFailed {
            repo: target.repo.clone(),
            file: target.file.clone(),
            detail: format!(
                "invalid revision {:?} (segments must be [A-Za-z0-9._-])",
                target.revision
            ),
        });
    }
    // MACHINE-WIDE download authority: per-key advisory lock via
    // `crate::catalog::download_lock` (`fs2::FileExt::try_lock_exclusive`).
    // This is the ONLY gate that decides "may this transfer start?" — if
    // another process on this box is already downloading the same key,
    // `acquire` returns [HG090] and no bytes move, no ledger row is
    // created, no `.part` temp is opened. The kernel drops the lock on ANY
    // exit (Done, Failed, Cancelled, SIGKILL, panic, power loss), so there
    // is no residue-detection heuristic: state is a kernel-managed lock.
    let dl_lock = crate::catalog::download_lock::DownloadLock::acquire(
        models_root,
        &target.repo,
        &target.file,
    )?;
    download_dual_locked(target, models_root, primary, fallback, progress, &dl_lock).await
    // `dl_lock` drops here — after the future resolves. For `download_dual`
    // this is fine (no separate cancel-registry guard to keep in sync).
}

/// [`download_dual`] but ASSUMES the caller already holds the machine-wide
/// `DownloadLock` for `(target.repo, target.file)`.
///
/// **Borrows the lock — does NOT take ownership.** This matters because the
/// machine flock and the node's cancel-registry row are separate guards; if
/// this fn owned the lock, it would drop when the download future resolved,
/// leaving a window where the cancel-registry row still advertised
/// `cancellable: true` for a slot the process no longer owned. The caller
/// (`pull_stream`) declares the lock in the outer spawned task, orders
/// bindings so the cancel guard drops FIRST, and the flock lives to
/// end-of-task — no asymmetric-lifetime race.
///
/// `pub(crate)` on purpose: the borrowed `DownloadLock` is bound to a
/// specific key by its filesystem path, but the type alone does not carry
/// that identity — a caller could hold a lock for one key and hand it in
/// for another, bypassing the flock for the OTHER key. To make that
/// mismatch impossible, we (a) restrict this fn to in-crate callers, and
/// (b) assert `dl_lock.protects(models_root, repo, file)` up front — a
/// caller that passes the wrong lock gets a coded error, never a silent
/// downgrade.
pub(crate) async fn download_dual_locked<P: Fetcher, F: Fetcher>(
    target: &PullTarget,
    models_root: &Path,
    primary: &P,
    fallback: &F,
    progress: &mut (dyn FnMut(u64, Option<u64>) + Send),
    dl_lock: &crate::catalog::download_lock::DownloadLock,
) -> Result<PathBuf, HiggsError> {
    // Identity check: the flock guard is a raw fd holding a lock on a
    // specific file path; if that path doesn't match this target's key,
    // the caller is protecting the wrong slot. Refuse rather than proceed.
    if !dl_lock.protects(models_root, &target.repo, &target.file) {
        return Err(HiggsError::DownloadFailed {
            repo: target.repo.clone(),
            file: target.file.clone(),
            detail: format!(
                "internal error: DownloadLock was acquired for a different key \
                 (guard path {}, expected {})",
                dl_lock.path().display(),
                crate::catalog::download_lock::lock_path(models_root, &target.repo, &target.file,)
                    .display(),
            ),
        });
    }
    // Deliberately no `let _dl_lock = dl_lock;` — the borrow is enough,
    // and the caller owns the guard's lifetime.
    // Status-only ledger claim: records the transfer, mirrors progress
    // (throttled), and records the terminal. A cancel drops this future
    // mid-poll — its terminal is recorded by the cancel path itself
    // (`catalog::cancel::cancellable_pull`).
    let ledger_claim =
        crate::catalog::ledger::claim_start(models_root, &target.repo, &target.file)?;
    // Throttle the mirror: a ledger write per chunk would thrash the file —
    // every ≥8 MiB of new bytes is plenty for status. Two conditions bypass
    // the delta: a PROGRESS REGRESSION (the fallback restarting from zero
    // after the primary died mid-flight — the mirror must drop to the truth
    // immediately, not keep announcing the primary's high-water mark) and a
    // TOTAL change (None→Some on the first tick, or a differing fallback
    // content length).
    let mut last_written: u64 = 0;
    let mut last_total: Option<u64> = None;
    let mut last_seen: (u64, Option<u64>) = (0, None);
    let mut ledger_progress = |downloaded: u64, total: Option<u64>| {
        last_seen = (downloaded, total);
        let regressed = downloaded < last_written;
        let total_changed = total != last_total;
        if regressed || total_changed || downloaded.saturating_sub(last_written) >= 8 * 1024 * 1024
        {
            last_written = downloaded;
            last_total = total;
            crate::catalog::ledger::record_progress(
                models_root,
                &target.repo,
                &target.file,
                downloaded,
                total,
            );
        }
        progress(downloaded, total);
    };
    let result = match download_attempt(target, models_root, primary, &mut ledger_progress).await {
        Ok(path) => Ok(path),
        // Deterministic local failure (bad input / our own fs write) — surface it verbatim.
        Err(AttemptError::Local(e)) => Err(e),
        // A fetcher failure could be transport-specific; the other transport may succeed.
        Err(AttemptError::Fetcher(primary_err)) => {
            tracing::warn!(
                repo = %target.repo,
                file = %target.file,
                error = %primary_err,
                "higgs: hub download failed; trying reqwest fallback"
            );
            match download_attempt(target, models_root, fallback, &mut ledger_progress).await {
                Ok(path) => Ok(path),
                Err(fallback_err) => Err(HiggsError::HubFetchExhausted {
                    repo: target.repo.clone(),
                    file: target.file.clone(),
                    primary: primary_err.to_string(),
                    fallback: fallback_err.into_inner().to_string(),
                }),
            }
        }
    };
    match &result {
        Ok(path) => ledger_claim.end(
            crate::catalog::ledger::LedgerEnd::Done {
                path: path.display().to_string(),
            },
            Some(last_seen),
        ),
        Err(e) => ledger_claim.end(
            crate::catalog::ledger::LedgerEnd::Failed {
                detail: e.to_string(),
            },
            Some(last_seen),
        ),
    }
    result
}

/// The higgs-owned models directory (`~/.higgs/models/`) — the ONLY place downloads land.
pub fn models_dir() -> std::io::Result<PathBuf> {
    let dir = crate::home::ensure_home()?.join("models");
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// The FALLBACK [`Fetcher`]: streams the `resolve` URL over HTTPS with `reqwest` (the
/// hand-rolled path, used when the hub client primary fails). Classifies its failures into
/// the same distinct codes the hub path uses — a non-success status routes through
/// [`crate::hub::http_status_to_error`] (404→`HG030`, 401/403→`HG029`, 429→`HG031`, else
/// `HG032`); a transport error is `HG033` (`HubTransport`).
pub struct HttpFetcher;

impl Fetcher for HttpFetcher {
    async fn fetch(
        &self,
        target: &PullTarget,
        on_chunk: &mut (dyn FnMut(&[u8]) + Send),
        progress: &mut (dyn FnMut(u64, Option<u64>) + Send),
    ) -> Result<(), HiggsError> {
        use futures::StreamExt;
        let transport = |detail: String| HiggsError::HubTransport {
            repo: target.repo.clone(),
            detail,
        };
        let url = hf_url(&target.repo, &target.revision, &target.file);
        let resp = reqwest::get(&url)
            .await
            .map_err(|e| transport(e.to_string()))?;
        if !resp.status().is_success() {
            let status = resp.status();
            return Err(crate::hub::http_status_to_error(
                &target.repo,
                &target.file,
                status.as_u16(),
                format!("HTTP {status}"),
            ));
        }
        let total = resp.content_length();
        let mut downloaded = 0u64;
        let mut stream = resp.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| transport(e.to_string()))?;
            on_chunk(&chunk);
            downloaded += chunk.len() as u64;
            progress(downloaded, total);
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "download_tests.rs"]
mod tests;
