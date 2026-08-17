//! The machine-local downloads LEDGER: `<models_root>/.downloads.json`
//! (`~/.higgs/models/.downloads.json` in production) — the durable,
//! cross-process record of what this machine is downloading and has
//! downloaded, referable by anything that wants download status (the node's
//! HELLO/`pull_status` announcement, the embed facade, the CLI).
//!
//! Every downloader on the machine funnels through
//! [`crate::download::download_dual`], which is the ledger's ONE writer path:
//! [`claim_start`] (STATUS RECORDING ONLY — the machine-wide duplicate gate
//! is [`crate::catalog::download_lock::DownloadLock`], acquired BEFORE the
//! claim; `claim_start` keeps only an in-process double-claim backstop),
//! throttled [`record_progress`], and a terminal [`record_end`]. The cancel path records its own terminal
//! ([`crate::catalog::cancel::cancellable_pull`] drops the download future
//! mid-poll, so `download_dual`'s own end-write never runs on a cancel).
//!
//! Robustness posture: the ledger is STATUS, not a lock the download's
//! correctness depends on (temps are pid-scoped, the rename is atomic — two
//! writers cannot corrupt a model file). So every ledger I/O failure FAILS
//! OPEN with a warning: a broken/readonly ledger degrades duplicate
//! protection to per-process and costs visibility, never a download. Only a
//! POSITIVE live-conflict refuses. A live entry whose `pid` is dead (crashed
//! downloader) is swept to `Failed("downloader process exited")` on the next
//! read-modify-write. Terminal history is pruned to the newest
//! [`MAX_TERMINAL_ENTRIES`].

use std::io::Write;
use std::path::{Path, PathBuf};

use fs2::FileExt;

use crate::catalog::wire::{DownloadLedgerEntry, DownloadLedgerStatus};
use crate::diagnostic::HiggsError;

/// Terminal history kept in the ledger (newest first); live entries are never
/// pruned.
pub const MAX_TERMINAL_ENTRIES: usize = 50;

/// Grace window for a `Downloading` row that has NO download-lock file.
/// Current-build writers acquire the lock BEFORE the ledger insert, so a
/// legitimate row's lock file always exists; a lockless row can only be a
/// legacy pre-lock-build entry or an interrupted writer. 24h is generous
/// against any same-build in-flight window while still reaping legacy rows
/// within a day of an upgrade.
pub(crate) const LOCKLESS_ROW_MAX_AGE_MS: u64 = 24 * 60 * 60 * 1000;

/// The ledger lives INSIDE the models root (`~/.higgs/models/.downloads.json`
/// in production) — derived from `models_root` rather than the global home so
/// every caller that can download (they all hold a models root) records into
/// the SAME file for that root, and every test root is isolated for free. The
/// leading dot keeps it out of the scanner's way (it walks `<org>/<model>/`
/// DIRECTORIES; a root-level file is never a model).
pub fn ledger_path(models_root: &Path) -> PathBuf {
    models_root.join(".downloads.json")
}

/// How a transfer ended, for [`record_end`].
#[derive(Debug, Clone)]
pub enum LedgerEnd {
    Done { path: String },
    Failed { detail: String },
    Cancelled,
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// Is `pid` a live process? `kill(pid, 0)` probes without signalling: `0` =
/// alive, `EPERM` = alive but not ours — both count.
fn pid_alive(pid: u32) -> bool {
    // A value that cannot be a real positive pid is DEAD, never probed:
    // `u32::MAX as pid_t` wraps to -1, and `kill(-1, 0)` is the BROADCAST
    // probe (every signalable process) — it "succeeds", which would make a
    // malformed ledger row immortal. Same for 0 (process group) and any
    // other non-positive wrap.
    let Ok(pid) = libc::pid_t::try_from(pid) else {
        return false;
    };
    if pid <= 0 {
        return false;
    }
    // SAFETY: signal 0 delivers nothing; it only error-checks the pid.
    let r = unsafe { libc::kill(pid, 0) };
    r == 0 || std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

/// The process's kernel START TIME (seconds precision, platform units) — the
/// pid-reuse disambiguator: `(pid, start_time)` identifies ONE process
/// incarnation, so a crashed downloader's pid recycled onto a long-lived
/// unrelated process no longer masquerades as a live transfer (which would
/// refuse its key forever and announce a download nobody owns). `None` when
/// the platform query fails — entries then degrade to pid-only liveness.
fn pid_start_time(pid: u32) -> Option<u64> {
    #[cfg(target_os = "macos")]
    {
        // SAFETY: proc_pidinfo with a properly-sized PROC_PIDTBSDINFO out-buffer.
        let mut info: libc::proc_bsdinfo = unsafe { std::mem::zeroed() };
        let size = std::mem::size_of::<libc::proc_bsdinfo>() as libc::c_int;
        let r = unsafe {
            libc::proc_pidinfo(
                pid as libc::c_int,
                libc::PROC_PIDTBSDINFO,
                0,
                std::ptr::addr_of_mut!(info).cast(),
                size,
            )
        };
        // Combine tv_sec + tv_usec into one u64 to close the same-second
        // PID reuse window: if a downloader crashes and macOS reuses its
        // PID for another process within the same wall-clock second, a
        // seconds-only stamp would false-positive `entry_process_alive`
        // (the crashed row would look immortal and refuse HG090 forever).
        (r == size).then(|| info.pbi_start_tvsec * 1_000_000 + info.pbi_start_tvusec as u64)
    }
    #[cfg(target_os = "linux")]
    {
        // /proc/<pid>/stat field 22 (starttime, clock ticks since boot). The
        // comm field (2) can contain spaces/parens — split AFTER the last ')'.
        let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
        let after = stat.rsplit_once(')')?.1;
        after.split_whitespace().nth(19)?.parse().ok()
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        let _ = pid;
        None
    }
}

/// Is this ledger entry's recorded process incarnation still alive? Pid
/// liveness, PLUS the start-time match when the entry recorded one — a
/// recycled pid (different start time) is DEAD for ledger purposes.
fn entry_process_alive(e: &DownloadLedgerEntry) -> bool {
    if !pid_alive(e.pid) {
        return false;
    }
    match (e.pid_started_at, pid_start_time(e.pid)) {
        (Some(recorded), Some(current)) => recorded == current,
        // Legacy entry or unqueryable platform: degrade to pid-only.
        _ => true,
    }
}

/// One exclusive read-modify-write pass over the ledger under `flock`.
/// `mutate` sees the swept entries and returns `Ok(true)` to persist /
/// `Ok(false)` to skip the write / `Err` to abort (nothing written).
fn with_ledger<T>(
    models_root: &Path,
    mutate: impl FnOnce(&mut Vec<DownloadLedgerEntry>) -> Result<(T, bool), HiggsError>,
) -> Result<T, LedgerIo> {
    let path = ledger_path(models_root);
    let lock_path = path.with_file_name(".downloads.json.lock");
    let io = |e: std::io::Error| LedgerIo::Io(e.to_string());
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(io)?;
    }
    let lock = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(&lock_path)
        .map_err(io)?;
    // NON-BLOCKING lock with a bounded retry, never lock_exclusive-blocking:
    // a paused/wedged process holding the lock must not wedge every other
    // downloader on the machine (the fail-open posture — and a blocking
    // acquire here would also sit BEFORE the cancel seam's select, making a
    // pull neither progress nor cancellable). Normal contention is a
    // microseconds-long write and resolves on the first retries; the
    // pathological holder costs a bounded ~500 ms, then fails open.
    let mut locked = false;
    for _ in 0..20 {
        match lock.try_lock_exclusive() {
            Ok(()) => {
                locked = true;
                break;
            }
            // fs2 maps EWOULDBLOCK to WouldBlock; the flock is released on
            // close (drop) — same kernel guarantee as the raw flock it
            // replaced.
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::sleep(std::time::Duration::from_millis(25));
            }
            Err(e) => return Err(io(e)),
        }
    }
    if !locked {
        return Err(LedgerIo::Io("ledger lock busy (wedged holder?)".into()));
    }
    // A missing file is an empty ledger; a CORRUPT file is treated the same
    // (warn) — status must never wedge downloads.
    let mut entries: Vec<DownloadLedgerEntry> = match std::fs::read(&path) {
        Ok(bytes) => serde_json::from_slice(&bytes).unwrap_or_else(|e| {
            tracing::warn!(path = %path.display(), error = %e, "downloads ledger corrupt — resetting");
            Vec::new()
        }),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Vec::new(),
        Err(e) => return Err(io(e)),
    };
    let mut swept = false;
    // DRAIN: replay any previously-queued terminals for this root that a
    // failed I/O left in memory (see `pending_terminals` + `record_end`).
    // Same-pid match: only THIS process's own terminals are ours to commit.
    // Two-phase: APPLY to `entries` in memory here, then REMOVE from the
    // queue only AFTER the file write succeeds (see the write block below).
    // Removing before write would silently lose the terminal on a
    // write-tmp/rename failure and leave the row `Downloading` under our
    // still-live pid.
    let canonical_for_pending = canonical_root(models_root);
    // APPLIED: found and mutated in `entries` — remove from pending ONLY
    // after the write commits (else a write failure loses the terminal).
    let mut drain_applied: Vec<(String, String, u64)> = Vec::new();
    // MISSING: no matching row in `entries` (already committed by a parallel
    // path, or the file was rewritten out from under us). Nothing to write
    // — remove from pending UNCONDITIONALLY so a read-only ledger touch
    // (e.g. `read_all`) still garbage-collects them. Without this split, a
    // long-lived node accumulates queued pendings whose rows are already
    // gone.
    let mut drain_missing: Vec<(String, String, u64)> = Vec::new();
    {
        let pending = pending_terminals().lock();
        let own = std::process::id();
        for p in pending.iter() {
            if p.models_root_canonical != canonical_for_pending || p.pid != own {
                continue;
            }
            if let Some(e) = entries.iter_mut().find(|e| {
                e.status == DownloadLedgerStatus::Downloading
                    && e.pid == own
                    && e.repo == p.repo
                    && e.file == p.file
            }) {
                e.ended_at_ms = Some(p.ended_at_ms);
                if let Some((d, t)) = p.final_counters {
                    e.downloaded = d;
                    e.total = t;
                }
                match &p.end {
                    LedgerEnd::Done { path } => {
                        e.status = DownloadLedgerStatus::Done;
                        e.path = Some(path.clone());
                    }
                    LedgerEnd::Failed { detail } => {
                        e.status = DownloadLedgerStatus::Failed;
                        e.detail = Some(detail.clone());
                    }
                    LedgerEnd::Cancelled => e.status = DownloadLedgerStatus::Cancelled,
                }
                drain_applied.push((p.repo.clone(), p.file.clone(), p.ended_at_ms));
                swept = true;
            } else {
                drain_missing.push((p.repo.clone(), p.file.clone(), p.ended_at_ms));
            }
        }
    }
    // Missing entries drop from the queue NOW — no write is required.
    if !drain_missing.is_empty() {
        let own = std::process::id();
        let mut pending = pending_terminals().lock();
        pending.retain(|p| {
            !(p.models_root_canonical == canonical_for_pending
                && p.pid == own
                && drain_missing
                    .iter()
                    .any(|(r, f, t)| *r == p.repo && *f == p.file && *t == p.ended_at_ms))
        });
    }
    // Sweep: a `Downloading` row is stale when any of:
    //  (a) the recorded process is dead (pid-liveness);
    //  (b) `download_lock::is_key_stale` proves the machine-wide lock file
    //      exists but nobody currently holds it (owner released the flock
    //      without recording a terminal — e.g. its terminal write failed
    //      and the queued retry lives only in its own memory);
    //  (c) the row has NO lock file at all AND is older than
    //      [`LOCKLESS_ROW_MAX_AGE_MS`] — a legacy row from a pre-
    //      download-lock build, or a writer that died between the ledger
    //      insert and the lock-file create. `is_key_stale` deliberately
    //      returns `false` on a missing lock file (a FRESH row in the
    //      bootstrap window must not be swept), so without the age bound
    //      these rows would survive forever and keep being announced.
    //      Every current-build writer acquires the lock BEFORE
    //      `claim_start`, so a legitimate row's lock file always exists —
    //      an old lockless row only needs to outlive the grace window
    //      once to be reaped.
    for entry in entries.iter_mut() {
        if entry.status != DownloadLedgerStatus::Downloading {
            continue;
        }
        let dead_pid = !entry_process_alive(entry);
        let unowned =
            crate::catalog::download_lock::is_key_stale(models_root, &entry.repo, &entry.file);
        let lockless_expired =
            !crate::catalog::download_lock::lock_file_exists(models_root, &entry.repo, &entry.file)
                && now_ms().saturating_sub(entry.started_at_ms) > LOCKLESS_ROW_MAX_AGE_MS;
        if dead_pid || unowned || lockless_expired {
            entry.status = DownloadLedgerStatus::Failed;
            entry.ended_at_ms = Some(now_ms());
            entry.detail = Some(if dead_pid {
                "downloader process exited".into()
            } else if unowned {
                "downloader released the machine lock without recording a terminal".into()
            } else {
                "stale lockless entry (pre-lock build or interrupted writer)".into()
            });
            swept = true;
        }
    }
    let (value, mutated) = mutate(&mut entries).map_err(LedgerIo::Refused)?;
    if mutated || swept {
        // Prune terminal history (newest ended first), keep every live entry.
        let mut terminal: Vec<&DownloadLedgerEntry> = entries
            .iter()
            .filter(|e| e.status != DownloadLedgerStatus::Downloading)
            .collect();
        if terminal.len() > MAX_TERMINAL_ENTRIES {
            terminal.sort_by_key(|e| std::cmp::Reverse(e.ended_at_ms.unwrap_or(0)));
            let cutoff: Vec<(String, String, u64)> = terminal[MAX_TERMINAL_ENTRIES..]
                .iter()
                .map(|e| (e.repo.clone(), e.file.clone(), e.started_at_ms))
                .collect();
            entries.retain(|e| {
                e.status == DownloadLedgerStatus::Downloading
                    || !cutoff
                        .iter()
                        .any(|(r, f, s)| *r == e.repo && *f == e.file && *s == e.started_at_ms)
            });
        }
        let bytes = serde_json::to_vec_pretty(&entries).map_err(|e| LedgerIo::Io(e.to_string()))?;
        let tmp = path.with_file_name(format!(".downloads.json.tmp.{}", std::process::id()));
        let write_tmp = || -> std::io::Result<()> {
            let mut f = std::fs::File::create(&tmp)?;
            f.write_all(&bytes)?;
            f.sync_all()
        };
        if let Err(e) = write_tmp() {
            let _ = std::fs::remove_file(&tmp);
            return Err(io(e));
        }
        std::fs::rename(&tmp, &path).map_err(|e| {
            let _ = std::fs::remove_file(&tmp);
            io(e)
        })?;
        // Write committed on disk → NOW safe to drop the drained pending
        // terminals. A write failure above returns early with them still
        // queued for the next attempt.
        if !drain_applied.is_empty() {
            let own = std::process::id();
            let mut pending = pending_terminals().lock();
            pending.retain(|p| {
                !(p.models_root_canonical == canonical_for_pending
                    && p.pid == own
                    && drain_applied
                        .iter()
                        .any(|(r, f, t)| *r == p.repo && *f == p.file && *t == p.ended_at_ms))
            });
        }
    }
    Ok(value)
}

/// Internal outcome of a ledger pass: a REFUSAL is authoritative (a live
/// conflict), an I/O failure is what callers fail OPEN on.
enum LedgerIo {
    Refused(HiggsError),
    Io(String),
}

/// The PROCESS-GLOBAL set of `(models_root, repo, file)` keys this process
/// holds live claims on — the
/// in-process half of the duplicate gate (per ROOT: the same repo/file into
/// two different roots is two different files). The machine-wide download
/// LOCK (`download_lock`) arbitrates across processes; this set arbitrates
/// across everything inside one process (two
/// `Higgs` instances, direct `download_dual` callers, the node data plane —
/// none of which share any other guard). It is also what makes the own-pid
/// RECLAIM sound: an own-pid ledger row whose key is NOT in this set cannot
/// belong to a live claim in this process — it is failed-terminal-write
/// residue, safe to replace. Every claim inserts here; end AND drop remove.
fn active_claims(
) -> &'static parking_lot::Mutex<std::collections::HashSet<(PathBuf, String, String)>> {
    static ACTIVE: std::sync::OnceLock<
        parking_lot::Mutex<std::collections::HashSet<(PathBuf, String, String)>>,
    > = std::sync::OnceLock::new();
    ACTIVE.get_or_init(|| parking_lot::Mutex::new(std::collections::HashSet::new()))
}

/// A terminal ledger write that could not commit (I/O failure, e.g. the
/// non-blocking lock's bounded retry timing out on a wedged holder). Held
/// in-memory and REPLAYED on the next successful `with_ledger` for the same
/// root — without this, a completed downloader whose end-write failed would
/// leave its `Downloading` row live under this process's still-alive pid,
/// misreporting status until a sweep catches it (its flock is already
/// released, so the `is_key_stale` sweep flips the row to `Failed` on the
/// next ledger read by anyone; cross-process claims are never blocked —
/// the flock, not the ledger, is the gate). The queue also drains on retire (drop
/// of the last outstanding claim in the process — unenforced today; the
/// next ledger touch is the recovery path).
#[derive(Debug, Clone)]
struct PendingTerminal {
    models_root_canonical: PathBuf,
    repo: String,
    file: String,
    end: LedgerEnd,
    final_counters: Option<(u64, Option<u64>)>,
    ended_at_ms: u64,
    pid: u32,
}

fn pending_terminals() -> &'static parking_lot::Mutex<Vec<PendingTerminal>> {
    static P: std::sync::OnceLock<parking_lot::Mutex<Vec<PendingTerminal>>> =
        std::sync::OnceLock::new();
    P.get_or_init(|| parking_lot::Mutex::new(Vec::new()))
}

/// CANONICALIZE a models root before it keys the process-global active claim
/// set: two aliased paths to the same directory (symlink; relative vs
/// absolute) MUST land on the same key, or the same-process duplicate gate
/// silently splits and two transfers race for the same on-disk file.
///
/// CREATES the models dir first (idempotent), so a claim on a not-yet-
/// created root canonicalizes to the SAME path a later same-process claim
/// would compute after `with_ledger` created the dir mid-transfer — without
/// this, the first claim would key the raw path and the second the
/// canonical alias, both bypassing the gate. Falls back to the raw path
/// only if creation itself fails (unwriteable parent) — the fail-open
/// posture; a hostile filesystem degrades the gate to per-process, never
/// wedges a download.
fn canonical_root(models_root: &Path) -> PathBuf {
    let _ = std::fs::create_dir_all(models_root);
    std::fs::canonicalize(models_root).unwrap_or_else(|_| models_root.to_path_buf())
}

/// This process's live hold on a ledger key — the RAII half of
/// [`claim_start`]. Ends explicitly via [`LedgerClaim::end`]
/// (`done`/`failed`); a DROP without an explicit end records `cancelled`, so
/// a download future dropped mid-poll (task abort, cancel seam, caller gone)
/// can never strand a live entry its own pid keeps un-sweepable.
#[derive(Debug)]
pub struct LedgerClaim {
    models_root: PathBuf,
    /// The canonical form of `models_root` captured at claim time. Load-
    /// bearing: `with_ledger` may CREATE the models dir mid-transfer, so
    /// re-canonicalizing on release would return a different path than the
    /// insert, leaving the active-claim key permanently held and letting a
    /// second same-process claim on the canonical path bypass the gate.
    active_key: (PathBuf, String, String),
    repo: String,
    file: String,
    ended: bool,
}

impl LedgerClaim {
    /// Record the transfer's terminal and defuse the drop-cancel.
    /// `final_counters` = the last observed `(downloaded, total)` — the
    /// throttled progress mirror can be megabytes behind (or never fired for
    /// a small file), so the terminal write carries the real final numbers.
    pub fn end(mut self, end: LedgerEnd, final_counters: Option<(u64, Option<u64>)>) {
        self.ended = true;
        // ORDER LOAD-BEARING: hold the active-claim gate until AFTER the
        // terminal file write. Releasing first would open a window in which a
        // second same-process claim on the same key enters `claim_start`,
        // sees no active claim, reclaims THIS transfer's own-pid
        // `Downloading` row as failed-terminal-write residue, and pushes its
        // own row — and then THIS `record_end` catches that fresh row (same
        // pid, same key) and wrongly terminalizes IT. Same reasoning applies
        // to `Drop` below.
        record_end(
            &self.models_root,
            &self.repo,
            &self.file,
            end,
            final_counters,
        );
        active_claims().lock().remove(&self.active_key);
    }
}

impl Drop for LedgerClaim {
    fn drop(&mut self) {
        if !self.ended {
            // An un-ended drop is the download future dropping mid-poll. The
            // per-transfer `TempGuard` in `crate::download::download` unlinks
            // THIS transfer's specific `.part.<pid>.<seq>` tmp precisely
            // (r45), so this ledger drop must NOT blanket-sweep — a
            // `remove_partials` here would clip any concurrent same-pid
            // caller's live tmp on the same key (r47 finding). Just record
            // the terminal and release the gate.
            //
            // ORDER LOAD-BEARING (see [`LedgerClaim::end`]): record FIRST,
            // release the active-claim gate SECOND. Releasing before
            // record_end would race a same-process reclaimer + our own
            // delayed write and terminalize the fresh row.
            record_end(
                &self.models_root,
                &self.repo,
                &self.file,
                LedgerEnd::Cancelled,
                None,
            );
            active_claims().lock().remove(&self.active_key);
        }
    }
}

/// Record a new `Downloading` row for `(repo, file)` and return the RAII
/// [`LedgerClaim`] that owns its progress/terminal writes.
///
/// **Not the machine-wide download gate.** Authority for "may this transfer
/// start?" lives in [`crate::catalog::download_lock::DownloadLock`] — a
/// per-key `fs2::try_lock_exclusive` the kernel drops on any exit. Callers
/// MUST hold that flock BEFORE calling `claim_start`; only `download_dual`
/// / `download_dual_locked` / `download` satisfy this invariant, so this
/// fn is `pub(crate)`. The ledger is pure status/history: this call
/// purges any stale `Downloading` row for the key and inserts ours, but
/// does NOT refuse a foreign live entry — the flock already guarantees
/// no other live owner exists.
///
/// Kept: the in-process active-claim gate (a second same-process
/// `claim_start` for the same key without an intervening drop is a
/// programming error, refused with HG090). Ledger I/O failures FAIL OPEN
/// (warn + `Ok`): status must never block a download.
pub(crate) fn claim_start(
    models_root: &Path,
    repo: &str,
    file: &str,
) -> Result<LedgerClaim, HiggsError> {
    // IN-PROCESS gate first: one live claim per key per process, no matter
    // how many `Higgs` instances or direct `download_dual` callers coexist.
    // Without this, two same-process transfers of one key would reclaim each
    // other's rows, terminalize each other, and (worse) a drop-sweep would
    // unlink the sibling's live same-pid `.part` temp.
    let active_key = (
        canonical_root(models_root),
        repo.to_owned(),
        file.to_owned(),
    );
    {
        let mut active = active_claims().lock();
        if !active.insert(active_key.clone()) {
            return Err(HiggsError::DownloadInFlight {
                repo: repo.to_owned(),
                file: file.to_owned(),
            });
        }
    }
    let release_key = active_key.clone();
    let release = |e: HiggsError| {
        // Balance the refcount from THIS claim's insert. Same helper is
        // called by end()/drop().
        active_claims().lock().remove(&release_key);
        e
    };
    let result = with_ledger(models_root, |entries| {
        // The ledger is PURE STATUS/HISTORY. Authority for "may this
        // transfer start?" lives in `crate::catalog::download_lock` — a
        // per-key `flock` the kernel drops on ANY process exit (Done,
        // Failed, Cancelled, SIGKILL, panic, power loss). By the time we
        // reach `claim_start`, `download_dual` already owns the
        // machine-wide slot for `(repo, file)` and no OTHER live process
        // can be transferring the same key.
        //
        // Purge only rows we can OWN or PROVE stale: our own pid's
        // leftovers (failed-terminal-write residue) and dead-pid rows.
        // A FOREIGN row with a live pid is kept — in the current build it
        // cannot exist (the flock refuses before claim_start), but during
        // a mixed-version window a pre-download-lock process may be
        // genuinely mid-transfer with a live row and no lock file; erasing
        // it would hide a real transfer from the fleet view. Two live rows
        // for one key in that window is the honest picture (two writers
        // genuinely exist — the old build allowed that); consumers dedup
        // by (repo, file) via accept_announced_downloads, and the standard
        // sweeps (dead-pid / stale-lock / lockless-age) retire the foreign
        // row when its owner goes away.
        let own = std::process::id();
        entries.retain(|e| {
            !(e.status == DownloadLedgerStatus::Downloading
                && e.repo == repo
                && e.file == file
                && (e.pid == own || !entry_process_alive(e)))
        });
        entries.push(DownloadLedgerEntry {
            repo: repo.to_owned(),
            file: file.to_owned(),
            pid: std::process::id(),
            pid_started_at: pid_start_time(std::process::id()),
            started_at_ms: now_ms(),
            downloaded: 0,
            total: None,
            status: DownloadLedgerStatus::Downloading,
            ended_at_ms: None,
            path: None,
            detail: None,
        });
        Ok(((), true))
    });
    let claim = || LedgerClaim {
        models_root: models_root.to_path_buf(),
        active_key: active_key.clone(),
        repo: repo.to_owned(),
        file: file.to_owned(),
        ended: false,
    };
    match result {
        Ok(()) => Ok(claim()),
        // Nothing was written on a refusal — no guard exists, nothing leaks,
        // and no drop can "cancel" a stranger's live entry.
        Err(LedgerIo::Refused(e)) => Err(release(e)),
        Err(LedgerIo::Io(detail)) => {
            tracing::warn!(
                repo,
                file,
                detail,
                "downloads ledger unavailable — claim fails open"
            );
            // Fail-open: the guard's eventual end/drop writes are themselves
            // best-effort no-ops when the entry never landed.
            Ok(claim())
        }
    }
}

/// Update this process's live entry's byte counters (throttled by the
/// caller). Missing entry / I/O failure: silent best-effort no-op.
pub fn record_progress(
    models_root: &Path,
    repo: &str,
    file: &str,
    downloaded: u64,
    total: Option<u64>,
) {
    let _ = with_ledger(models_root, |entries| {
        let own = entries.iter_mut().find(|e| {
            e.status == DownloadLedgerStatus::Downloading
                && e.pid == std::process::id()
                && e.repo == repo
                && e.file == file
        });
        match own {
            Some(e) => {
                e.downloaded = downloaded;
                e.total = total;
                Ok(((), true))
            }
            None => Ok(((), false)),
        }
    });
}

/// Flip this process's live entry terminal (via [`LedgerClaim`]). Missing
/// entry / I/O failure: best-effort no-op (warn on I/O).
fn record_end(
    models_root: &Path,
    repo: &str,
    file: &str,
    end: LedgerEnd,
    final_counters: Option<(u64, Option<u64>)>,
) {
    // Clones so we can BOTH consume in the closure AND queue on I/O failure.
    let end_for_write = end.clone();
    let result = with_ledger(models_root, |entries| {
        let own = entries.iter_mut().find(|e| {
            e.status == DownloadLedgerStatus::Downloading
                && e.pid == std::process::id()
                && e.repo == repo
                && e.file == file
        });
        let Some(e) = own else { return Ok(((), false)) };
        e.ended_at_ms = Some(now_ms());
        if let Some((downloaded, total)) = final_counters {
            e.downloaded = downloaded;
            e.total = total;
        }
        match end_for_write {
            LedgerEnd::Done { path } => {
                e.status = DownloadLedgerStatus::Done;
                e.path = Some(path);
            }
            LedgerEnd::Failed { detail } => {
                e.status = DownloadLedgerStatus::Failed;
                e.detail = Some(detail);
            }
            LedgerEnd::Cancelled => e.status = DownloadLedgerStatus::Cancelled,
        }
        Ok(((), true))
    });
    if let Err(LedgerIo::Io(detail)) = result {
        tracing::warn!(
            repo,
            file,
            detail,
            "downloads ledger terminal write failed — queued for retry on next ledger touch"
        );
        pending_terminals().lock().push(PendingTerminal {
            models_root_canonical: canonical_root(models_root),
            repo: repo.to_owned(),
            file: file.to_owned(),
            end,
            final_counters,
            ended_at_ms: now_ms(),
            pid: std::process::id(),
        });
    }
}

/// Overwrite THIS ATTEMPT's `Failed` terminal for `(repo, file)` under this
/// process's pid to `Cancelled`, clearing the detail. Called by
/// `cancellable_pull` when the download's Err was translated to HG089
/// because a cancel was signaled in the same poll — without this, the
/// event stream would say Cancelled/HG089 while the ledger row said
/// Failed with the raw fetcher error. Best-effort (ledger is
/// status/history — a wedged write costs visibility, never correctness).
///
/// `attempt_started_ms` BOUNDS the rewrite to the current attempt: only a
/// Failed terminal ended AT/AFTER the attempt began can be this attempt's
/// (its `LedgerClaim::end` ran inside the attempt's lifetime). Without the
/// bound, the flip could retarget a PRIOR attempt's genuine Failed history
/// row for the same key — e.g. when this attempt's terminal got QUEUED
/// (ledger wedged) and the drain already reconciled it, the unbounded
/// max_by_key would fall through to yesterday's real failure and rewrite
/// it as an operator cancel.
pub(crate) fn overwrite_last_failed_to_cancelled(
    models_root: &Path,
    repo: &str,
    file: &str,
    attempt_started_ms: u64,
) {
    // FIRST: the Failed terminal may never have reached the file — a
    // ledger I/O failure at `record_end` QUEUES it in `pending_terminals`
    // (this process's memory) for replay on the next ledger touch. If we
    // only rewrote the file, the queued Failed would drain LATER and
    // resurrect the raw fetcher error over our Cancelled. Flip ONLY the
    // NEWEST queued Failed for this key+pid WITHIN the attempt window.
    {
        let own = std::process::id();
        let canonical = canonical_root(models_root);
        let mut pending = pending_terminals().lock();
        let newest = pending
            .iter()
            .enumerate()
            .filter(|(_, p)| {
                p.pid == own
                    && p.models_root_canonical == canonical
                    && p.repo == repo
                    && p.file == file
                    && p.ended_at_ms >= attempt_started_ms
                    && matches!(p.end, LedgerEnd::Failed { .. })
            })
            .max_by_key(|(_, p)| p.ended_at_ms)
            .map(|(i, _)| i);
        if let Some(i) = newest {
            pending[i].end = LedgerEnd::Cancelled;
        }
    }
    let _ = with_ledger(models_root, |entries| {
        let own = std::process::id();
        // Find the most recent Failed row for this key + this pid ENDED
        // WITHIN the attempt window — the one the inner download's
        // `LedgerClaim::end(Failed{..})` just wrote. Rows ended before the
        // attempt began are prior attempts' genuine history and are never
        // candidates.
        let idx = entries
            .iter()
            .enumerate()
            .filter(|(_, e)| {
                e.pid == own
                    && e.repo == repo
                    && e.file == file
                    && e.status == DownloadLedgerStatus::Failed
                    && e.ended_at_ms.unwrap_or(0) >= attempt_started_ms
            })
            .max_by_key(|(_, e)| e.ended_at_ms.unwrap_or(0))
            .map(|(i, _)| i);
        if let Some(i) = idx {
            entries[i].status = DownloadLedgerStatus::Cancelled;
            entries[i].detail = None;
            return Ok(((), true));
        }
        Ok(((), false))
    });
}

/// Milliseconds since the Unix epoch — the same clock every ledger stamp
/// uses. Exposed for `cancellable_pull` to capture an attempt-start bound.
pub(crate) fn unix_now_ms() -> u64 {
    now_ms()
}

/// Every ledger entry, dead-pid entries swept (persisted best-effort). Live
/// entries first, then terminal newest-ended first. I/O failure → empty
/// (status is best-effort; callers still have the in-process registry).
pub fn read_all(models_root: &Path) -> Vec<DownloadLedgerEntry> {
    let result = with_ledger(models_root, |entries| Ok((entries.clone(), false)));
    match result {
        Ok(mut entries) => {
            entries.sort_by_key(|e| {
                (
                    e.status != DownloadLedgerStatus::Downloading,
                    std::cmp::Reverse(e.ended_at_ms.unwrap_or(u64::MAX)),
                )
            });
            entries
        }
        Err(_) => Vec::new(),
    }
}

/// Only the live (`downloading`, pid-alive) entries.
pub fn read_live(models_root: &Path) -> Vec<DownloadLedgerEntry> {
    read_all(models_root)
        .into_iter()
        .filter(|e| e.status == DownloadLedgerStatus::Downloading)
        .collect()
}

#[cfg(test)]
#[path = "ledger_tests.rs"]
mod tests;
