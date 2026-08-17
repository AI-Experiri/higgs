use super::*;
use std::os::unix::fs::PermissionsExt;

fn root() -> (tempfile::TempDir, PathBuf) {
    let home = tempfile::tempdir().expect("home");
    let models = home.path().join("models");
    std::fs::create_dir_all(&models).expect("models dir");
    (home, models)
}

/// Queued pending-terminals for THIS test's models root only. The queue is
/// process-global and tests run in parallel — asserting on the whole queue
/// (emptiness, raw length) races sibling tests' entries. Every assertion
/// must scope to its own canonical root.
fn pending_for(models: &std::path::Path) -> usize {
    let canonical = canonical_root(models);
    pending_terminals()
        .lock()
        .iter()
        .filter(|p| p.models_root_canonical == canonical)
        .count()
}

/// Queue a REAL pending terminal through the production seam: hold the
/// ledger flock so `end()`'s write fails and the terminal queues in-memory.
fn wedge_end(models: &std::path::Path, claim: LedgerClaim, end: LedgerEnd) {
    use std::os::fd::AsRawFd;
    let lock_path = models.join(".downloads.json.lock");
    let holder = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(&lock_path)
        .unwrap();
    assert_eq!(unsafe { libc::flock(holder.as_raw_fd(), libc::LOCK_EX) }, 0);
    claim.end(end, Some((7, Some(10))));
    drop(holder);
}

#[test]
fn queued_terminals_drain_onto_their_rows_on_the_next_ledger_touch() {
    // All three terminal shapes queue under a wedged flock, then the NEXT
    // ledger touch (a plain read) applies each to its still-Downloading row
    // and drains the queue. This is the recovery path for "terminal write
    // failed but the process lives" — without it, rows stay Downloading
    // under a live pid until process exit.
    let (_home, models) = root();
    let c1 = claim_start(&models, "acme/done", "a.gguf").expect("claim a");
    let c2 = claim_start(&models, "acme/failed", "b.gguf").expect("claim b");
    let c3 = claim_start(&models, "acme/cxl", "c.gguf").expect("claim c");
    wedge_end(
        &models,
        c1,
        LedgerEnd::Done {
            path: "/models/acme/done/a.gguf".into(),
        },
    );
    wedge_end(
        &models,
        c2,
        LedgerEnd::Failed {
            detail: "boom".into(),
        },
    );
    wedge_end(&models, c3, LedgerEnd::Cancelled);
    assert_eq!(pending_for(&models), 3, "all three terminals queued");

    let all = read_all(&models);
    let by_repo = |r: &str| {
        all.iter()
            .find(|e| e.repo == r)
            .unwrap_or_else(|| panic!("row {r} present: {all:?}"))
    };
    assert_eq!(by_repo("acme/done").status, DownloadLedgerStatus::Done);
    assert_eq!(
        by_repo("acme/done").path.as_deref(),
        Some("/models/acme/done/a.gguf")
    );
    assert_eq!(by_repo("acme/failed").status, DownloadLedgerStatus::Failed);
    assert_eq!(by_repo("acme/failed").detail.as_deref(), Some("boom"));
    assert_eq!(by_repo("acme/cxl").status, DownloadLedgerStatus::Cancelled);
    assert_eq!(pending_for(&models), 0, "queue drained after the write");
}

#[test]
fn a_queued_terminal_for_a_vanished_row_drops_from_the_queue() {
    // The drain's OTHER arm: the queued terminal's row no longer exists in
    // the file (ledger deleted/rewritten by another process). The entry
    // must drop from the queue — retrying forever against a missing row
    // would pin the queue for the process lifetime.
    let (_home, models) = root();
    let claim = claim_start(&models, "acme/gone", "g.gguf").expect("claim");
    wedge_end(&models, claim, LedgerEnd::Cancelled);
    assert_eq!(pending_for(&models), 1, "terminal queued");
    std::fs::remove_file(ledger_path(&models)).expect("vanish the ledger");

    let all = read_all(&models);
    assert!(
        !all.iter().any(|e| e.repo == "acme/gone"),
        "no row resurrected for the vanished entry: {all:?}"
    );
    assert_eq!(
        pending_for(&models),
        0,
        "missing-row terminal dropped from the queue, not retried forever"
    );
}

#[test]
fn the_err_cancel_flip_also_rewrites_a_queued_failed_terminal() {
    // The Err+cancel ledger reconciliation must reach a Failed terminal
    // that is still QUEUED (its write failed) — the flip's queue pass. A
    // later drain then lands Cancelled, matching the HG089 the caller saw.
    let (_home, models) = root();
    let attempt_started = unix_now_ms();
    let claim = claim_start(&models, "acme/q", "q.gguf").expect("claim");
    wedge_end(
        &models,
        claim,
        LedgerEnd::Failed {
            detail: "fetcher boom".into(),
        },
    );
    assert_eq!(pending_for(&models), 1);

    overwrite_last_failed_to_cancelled(&models, "acme/q", "q.gguf", attempt_started);
    let canonical = canonical_root(&models);
    let flipped = pending_terminals()
        .lock()
        .iter()
        .filter(|p| p.models_root_canonical == canonical && p.repo == "acme/q")
        .all(|p| matches!(p.end, LedgerEnd::Cancelled));
    assert!(flipped, "queued Failed terminal rewritten to Cancelled");

    let all = read_all(&models);
    let row = all.iter().find(|e| e.repo == "acme/q").expect("row");
    assert_eq!(
        row.status,
        DownloadLedgerStatus::Cancelled,
        "the drained row matches the HG089 the caller saw"
    );
}

#[test]
fn an_unreadable_ledger_path_reads_as_empty_and_refuses_no_claim() {
    // A DIRECTORY squatting the ledger file path: reads degrade to empty
    // (status is best-effort, never a failure source) and a claim still
    // proceeds on the I/O-error arm (the ledger is status only — refusing
    // a download because the STATUS file is broken would invert authority).
    let (_home, models) = root();
    std::fs::create_dir_all(ledger_path(&models)).expect("squat ledger path");
    assert!(read_all(&models).is_empty(), "unreadable file reads empty");
    assert!(read_live(&models).is_empty());
    let claim = claim_start(&models, "acme/io", "i.gguf")
        .expect("a broken STATUS file must not refuse a download");
    claim.end(LedgerEnd::Cancelled, None);
    // Scope cleanup: drop any terminals this test queued (the write can
    // never land while the path is squatted).
    let canonical = canonical_root(&models);
    pending_terminals()
        .lock()
        .retain(|p| p.models_root_canonical != canonical);
}

#[test]
fn claim_progress_end_round_trip_and_history() {
    let (_home, models) = root();
    let claim = claim_start(&models, "acme/m", "m.gguf").expect("first claim");
    let live = read_live(&models);
    assert_eq!(live.len(), 1);
    assert_eq!(live[0].pid, std::process::id());
    assert_eq!(live[0].status, DownloadLedgerStatus::Downloading);

    record_progress(&models, "acme/m", "m.gguf", 42, Some(100));
    let live = read_live(&models);
    assert_eq!((live[0].downloaded, live[0].total), (42, Some(100)));

    claim.end(
        LedgerEnd::Done {
            path: "/models/acme/m/m.gguf".into(),
        },
        Some((100, Some(100))),
    );
    assert!(read_live(&models).is_empty(), "terminal frees the key");
    let all = read_all(&models);
    assert_eq!(all.len(), 1, "history retained");
    assert_eq!(all[0].status, DownloadLedgerStatus::Done);
    assert_eq!(all[0].path.as_deref(), Some("/models/acme/m/m.gguf"));
    assert!(all[0].ended_at_ms.is_some());

    // The key freed — a fresh claim succeeds and coexists with history.
    let claim2 = claim_start(&models, "acme/m", "m.gguf").expect("re-claim after done");
    assert_eq!(read_live(&models).len(), 1);
    assert_eq!(read_all(&models).len(), 2);
    claim2.end(LedgerEnd::Cancelled, None);
}

#[test]
fn dropping_the_claim_records_cancelled_and_frees_the_key() {
    // The RAII half: a download future dropped mid-poll (task abort, cancel
    // seam, caller gone) drops its claim — which must not strand a live entry
    // this process's own pid keeps un-sweepable. The drop records
    // `cancelled` and frees the key.
    let (_home, models) = root();
    let claim = claim_start(&models, "acme/m", "m.gguf").expect("claim");
    drop(claim);
    assert!(read_live(&models).is_empty(), "drop freed the key");
    let all = read_all(&models);
    assert_eq!(all[0].status, DownloadLedgerStatus::Cancelled);
    let _re = claim_start(&models, "acme/m", "m.gguf").expect("key claimable again");
}

// The cross-process "live key refuses" test is now
// `catalog::download_lock::tests::a_second_acquire_of_a_held_key_refuses_with_hg090`
// — machine-wide download authority is the flock in `download_lock`, not
// the ledger. The ledger is pure status/history; a stale `Downloading` row
// left by a crashed process gets purged and rewritten by the next claim.

#[test]
fn a_dead_pid_live_entry_is_swept_to_failed_and_frees_the_key() {
    let (_home, models) = root();
    // Hand-write a ledger with a crashed downloader's stale live entry.
    let stale = DownloadLedgerEntry {
        repo: "acme/m".into(),
        file: "m.gguf".into(),
        pid: 4_000_000_000, // beyond any real pid space — never alive
        pid_started_at: None,
        started_at_ms: 1,
        downloaded: 10,
        total: Some(100),
        status: DownloadLedgerStatus::Downloading,
        ended_at_ms: None,
        path: None,
        detail: None,
    };
    std::fs::write(
        ledger_path(&models),
        serde_json::to_vec(&vec![stale]).unwrap(),
    )
    .unwrap();

    assert!(
        read_live(&models).is_empty(),
        "a dead downloader's entry is not live"
    );
    let all = read_all(&models);
    assert_eq!(all[0].status, DownloadLedgerStatus::Failed);
    assert_eq!(all[0].detail.as_deref(), Some("downloader process exited"));
    assert!(all[0].ended_at_ms.is_some(), "sweep stamps the end");
    // The swept key is claimable again.
    let _re = claim_start(&models, "acme/m", "m.gguf").expect("key freed by the sweep");
}

#[test]
fn a_corrupt_ledger_resets_instead_of_wedging_downloads() {
    let (_home, models) = root();
    std::fs::write(ledger_path(&models), b"{ not json").unwrap();
    assert!(read_all(&models).is_empty(), "corrupt reads as empty");
    let _claim =
        claim_start(&models, "acme/m", "m.gguf").expect("claim proceeds on a reset ledger");
    assert_eq!(read_live(&models).len(), 1);
}

#[test]
fn terminal_history_is_pruned_to_the_cap_live_entries_never() {
    let (_home, models) = root();
    for i in 0..(MAX_TERMINAL_ENTRIES + 10) {
        let file = format!("f{i}.gguf");
        let claim = claim_start(&models, "acme/m", &file).expect("claim");
        claim.end(
            LedgerEnd::Failed {
                detail: "test".into(),
            },
            None,
        );
    }
    let _live_claim = claim_start(&models, "acme/m", "live.gguf").expect("live claim");
    let all = read_all(&models);
    let live = all
        .iter()
        .filter(|e| e.status == DownloadLedgerStatus::Downloading)
        .count();
    let terminal = all.len() - live;
    assert_eq!(live, 1, "the live entry survives every prune");
    assert!(
        terminal <= MAX_TERMINAL_ENTRIES,
        "terminal history capped: {terminal}"
    );
}

#[test]
fn ledger_lives_inside_the_models_root_dot_hidden() {
    let (_home, models) = root();
    assert_eq!(ledger_path(&models), models.join(".downloads.json"));
}

#[test]
fn an_own_pid_live_entry_is_reclaimed_not_refused() {
    // A live own-pid row in the FILE with NO active claim in this process is
    // failed-terminal-write residue (a disk hiccup while ending a prior
    // transfer left the row live; the in-process active-claim set proves no
    // live owner exists). Refusing it would wedge the key for the process
    // lifetime (the dead-pid sweep cannot reap a live pid); reclaiming
    // self-heals. Cross-process refusal is untouched.
    let (_home, models) = root();
    let residue = DownloadLedgerEntry {
        repo: "acme/m".into(),
        file: "m.gguf".into(),
        pid: std::process::id(),
        pid_started_at: None,
        // Fresh: an ancient lockless row would be reaped by the age sweep
        // before the reclaim path this test pins is ever exercised.
        started_at_ms: now_ms(),
        downloaded: 10,
        total: Some(100),
        status: DownloadLedgerStatus::Downloading,
        ended_at_ms: None,
        path: None,
        detail: None,
    };
    std::fs::write(
        ledger_path(&models),
        serde_json::to_vec(&vec![residue]).unwrap(),
    )
    .unwrap();
    assert_eq!(read_live(&models).len(), 1, "stale own-pid entry is live");
    let re = claim_start(&models, "acme/m", "m.gguf")
        .expect("an own-pid live entry with NO active in-process claim is residue — reclaimed");
    assert_eq!(
        read_live(&models).len(),
        1,
        "reclaim replaces, never duplicates"
    );
    re.end(LedgerEnd::Cancelled, None);
}

#[test]
fn a_second_in_process_claim_of_a_held_key_is_refused() {
    // The IN-PROCESS gate: `download_dual` is public and `Higgs` instances
    // can coexist in one process with no shared guard — without this
    // refusal, two same-process transfers of one key would reclaim each
    // other's rows and a drop-sweep would unlink the sibling's live
    // same-pid temp. One live claim per (root, repo, file) per process.
    let (_home, models) = root();
    let held = claim_start(&models, "acme/m", "m.gguf").expect("first claim");
    let err = claim_start(&models, "acme/m", "m.gguf")
        .expect_err("the key is actively claimed in this process");
    assert!(
        matches!(err, HiggsError::DownloadInFlight { .. }),
        "same [HG090] as every duplicate: {err}"
    );
    held.end(LedgerEnd::Cancelled, None);
    let _re = claim_start(&models, "acme/m", "m.gguf").expect("key free after end");
}

#[test]
fn a_recycled_pid_with_a_different_start_time_is_swept() {
    // Pid liveness alone can't tell a crashed downloader from an unrelated
    // long-lived process that inherited its pid — that stale row would refuse
    // the key indefinitely and announce a transfer nobody owns. The recorded
    // (pid, start_time) pair disambiguates: OUR pid with a WRONG start time
    // is exactly a recycled pid, and must read as dead.
    let (_home, models) = root();
    let recycled = DownloadLedgerEntry {
        repo: "acme/m".into(),
        file: "m.gguf".into(),
        pid: std::process::id(), // alive — but the start time says another incarnation
        pid_started_at: Some(1), // nothing real started at epoch second 1
        started_at_ms: 1,
        downloaded: 10,
        total: Some(100),
        status: DownloadLedgerStatus::Downloading,
        ended_at_ms: None,
        path: None,
        detail: None,
    };
    std::fs::write(
        ledger_path(&models),
        serde_json::to_vec(&vec![recycled]).unwrap(),
    )
    .unwrap();
    assert!(
        read_live(&models).is_empty(),
        "a recycled-pid entry is not live"
    );
    let _re = claim_start(&models, "acme/m", "m.gguf").expect("key freed by the sweep");
}

#[test]
fn a_wedged_lock_holder_fails_open_instead_of_hanging() {
    // The fail-open posture end to end: a paused/wedged process holding the
    // ledger lock must cost a bounded wait, never a hang (a blocking acquire
    // would sit BEFORE the cancel seam — a pull that can neither progress
    // nor cancel). The claim comes back Ok (machine-wide protection degrades
    // to per-process, warned) within seconds.
    use std::os::fd::AsRawFd;
    let (_home, models) = root();
    std::fs::create_dir_all(&models).unwrap();
    let lock_path = models.join(".downloads.json.lock");
    let holder = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(&lock_path)
        .unwrap();
    assert_eq!(
        unsafe { libc::flock(holder.as_raw_fd(), libc::LOCK_EX) },
        0,
        "test holds the ledger lock"
    );
    let (tx, rx) = std::sync::mpsc::channel();
    let m = models.clone();
    std::thread::spawn(move || {
        let _ = tx.send(claim_start(&m, "acme/m", "m.gguf").map(std::mem::forget));
    });
    let outcome = rx
        .recv_timeout(std::time::Duration::from_secs(5))
        .expect("claim returns within the bounded retry window — no hang");
    assert!(outcome.is_ok(), "wedged lock fails OPEN: {outcome:?}");
}

#[test]
fn dropping_the_claim_records_cancelled_without_touching_other_temps() {
    // Post-r47: `LedgerClaim::drop` no longer blanket-sweeps the key. Each
    // `download` transfer owns a per-attempt `TempGuard` (`download.rs`)
    // that unlinks THIS transfer's specific `.part.<pid>.<seq>` tmp on
    // drop — precise cleanup, no cross-caller collateral damage. The
    // ledger drop is responsible ONLY for the terminal + gate release.
    // Test: hand-place decoy tmps under the SAME key (both same-pid AND
    // foreign) and assert both survive the drop; the terminal is
    // recorded as Cancelled.
    let (_home, models) = root();
    let dir = models.join("acme/m");
    std::fs::create_dir_all(&dir).unwrap();
    let own_same_key = dir.join(format!("m.gguf.part.{}.99", std::process::id()));
    let foreign = dir.join("m.gguf.part.999999.0");
    std::fs::write(&own_same_key, b"another same-pid caller's tmp").unwrap();
    std::fs::write(&foreign, b"another process's tmp").unwrap();

    let claim = claim_start(&models, "acme/m", "m.gguf").expect("claim");
    drop(claim);
    assert!(
        own_same_key.exists(),
        "another same-pid caller's tmp for the SAME key survives — no blanket sweep"
    );
    assert!(foreign.exists(), "another process's tmp survives");
    assert_eq!(read_all(&models)[0].status, DownloadLedgerStatus::Cancelled);
    // Clean up decoys so the tempdir teardown is clean.
    let _ = std::fs::remove_file(&own_same_key);
    let _ = std::fs::remove_file(&foreign);
}

#[test]
fn a_pid_that_cannot_be_a_process_is_dead_not_broadcast() {
    // `u32::MAX as pid_t` is -1 — and `kill(-1, 0)` is the BROADCAST probe
    // (signals every process we may signal), which "succeeds" and would make
    // a malformed ledger row immortal: never swept, refusing its key with
    // HG090 forever, announced as a download nobody owns. Any pid that
    // cannot be a real positive process id must read as dead.
    let (_home, models) = root();
    let malformed = DownloadLedgerEntry {
        repo: "acme/m".into(),
        file: "m.gguf".into(),
        pid: u32::MAX, // wraps to pid_t -1: the broadcast pid
        pid_started_at: None,
        started_at_ms: 1,
        downloaded: 0,
        total: None,
        status: DownloadLedgerStatus::Downloading,
        ended_at_ms: None,
        path: None,
        detail: None,
    };
    std::fs::write(
        ledger_path(&models),
        serde_json::to_vec(&vec![malformed]).unwrap(),
    )
    .unwrap();
    assert!(
        read_live(&models).is_empty(),
        "broadcast pid is not a live process"
    );
    let _re = claim_start(&models, "acme/m", "m.gguf").expect("key freed by the sweep");
}

#[test]
fn a_failed_terminal_write_is_queued_and_flushes_on_next_ledger_touch() {
    // The wedged-lock case: our terminal write fails I/O (bounded flock
    // retry timing out on a held lock). The row stays `Downloading` under
    // our still-alive pid, so other processes would refuse HG090 forever.
    // Fix: the terminal is QUEUED in memory and REPLAYED on the next
    // successful `with_ledger` — one further touch by any code path (a
    // status read, a fresh claim) commits it, so the wedge duration is
    // bounded by real ledger traffic, not our process lifetime.
    use std::os::fd::AsRawFd;
    let (_home, models) = root();
    let claim = claim_start(&models, "acme/m", "m.gguf").expect("claim");
    // Hold the ledger lock so the terminal write times out.
    std::fs::create_dir_all(&models).unwrap();
    let lock_path = models.join(".downloads.json.lock");
    let holder = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(&lock_path)
        .unwrap();
    assert_eq!(unsafe { libc::flock(holder.as_raw_fd(), libc::LOCK_EX) }, 0);
    // With the lock held, end() cannot commit — but the queue captures it.
    claim.end(
        LedgerEnd::Done {
            path: "/x.gguf".into(),
        },
        Some((100, Some(100))),
    );
    assert!(
        pending_for(&models) > 0,
        "the wedged-lock terminal is queued for retry"
    );
    // Release the lock and touch the ledger — the queued terminal commits.
    drop(holder);
    let all = read_all(&models);
    assert_eq!(
        all[0].status,
        DownloadLedgerStatus::Done,
        "next ledger touch drained the queue"
    );
    assert_eq!(all[0].path.as_deref(), Some("/x.gguf"));
    assert_eq!(
        pending_for(&models),
        0,
        "this root's queue is empty after a successful drain"
    );
}

#[test]
fn aliased_paths_to_the_same_root_share_the_in_process_gate() {
    // Two aliased paths to the same directory (e.g. symlink; or relative vs
    // absolute) MUST land on the same in-process gate key, or the same
    // (repo, file) can be claimed twice in one process, race for the same
    // on-disk `.part` temps, and drop-sweep each other's live temp.
    let (home, models) = root();
    let alias = home.path().join("alias-models");
    std::os::unix::fs::symlink(&models, &alias).unwrap();
    let held = claim_start(&models, "acme/m", "m.gguf").expect("held via real path");
    let err = claim_start(&alias, "acme/m", "m.gguf")
        .expect_err("second claim via symlink must hit the same gate");
    assert!(
        matches!(err, HiggsError::DownloadInFlight { .. }),
        "aliased path is still the same key: {err}"
    );
    held.end(LedgerEnd::Cancelled, None);
    let _re = claim_start(&alias, "acme/m", "m.gguf").expect("free after end");
}

#[test]
fn drained_pending_terminals_survive_a_write_failure() {
    // Two-phase drain load-bearing: the queue must NOT be emptied until the
    // ledger write commits, or a `write_tmp`/`rename` failure silently loses
    // the terminal (row stays `Downloading` under our still-live pid → other
    // processes wedge on HG090). Setup: queue a terminal, then make the next
    // ledger write fail via a read-only models dir. Assert: the queue still
    // contains the terminal after a failed `read_all` attempt; the row is
    // still `Downloading` on disk.
    let (_home, models) = root();
    let claim = claim_start(&models, "acme/m", "m.gguf").expect("claim");
    // Queue a pending terminal by holding the lock during end().
    use std::os::fd::AsRawFd;
    let lock_path = models.join(".downloads.json.lock");
    let holder = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(&lock_path)
        .unwrap();
    assert_eq!(unsafe { libc::flock(holder.as_raw_fd(), libc::LOCK_EX) }, 0);
    claim.end(
        LedgerEnd::Done {
            path: "/x.gguf".into(),
        },
        Some((100, Some(100))),
    );
    // Release the flock so the next with_ledger CAN acquire it, but flip the
    // models dir to read-only so `write_tmp` / `rename` fail after the drain
    // applied its in-memory update.
    drop(holder);
    let queue_len_before = pending_for(&models);
    assert!(queue_len_before > 0, "the wedged-lock terminal is queued");
    let mode = std::fs::metadata(&models).unwrap().permissions().mode();
    let mut ro = std::fs::metadata(&models).unwrap().permissions();
    ro.set_mode(0o500); // r-x, no write
    std::fs::set_permissions(&models, ro).unwrap();
    // Force a write attempt: claim_start writes the ledger. Fails on write,
    // pending stays populated (the drain didn't remove — write failed).
    let re_result = claim_start(&models, "acme/m", "m.gguf");
    // Restore perms so the tempdir can be cleaned up.
    let mut back = std::fs::metadata(&models).unwrap().permissions();
    back.set_mode(mode);
    std::fs::set_permissions(&models, back).unwrap();
    // The queue MUST retain the terminal — write failed, drain did not commit.
    assert_eq!(
        pending_for(&models),
        queue_len_before,
        "a failed write must NOT drop drained pending terminals: {re_result:?}"
    );
    // Drain remnants for later ledger touches (release perms first). Scope
    // the cleanup to THIS root — a global clear() would race sibling tests'
    // queued entries (the queue is process-global).
    if let Ok(re) = re_result {
        re.end(LedgerEnd::Cancelled, None);
    }
    let canonical = canonical_root(&models);
    pending_terminals()
        .lock()
        .retain(|p| p.models_root_canonical != canonical);
}

#[test]
fn active_claim_key_survives_dir_creation_mid_transfer() {
    // `claim_start` may see a NON-EXISTENT models_root — canonicalize would
    // fall back to the raw path. `with_ledger` then creates the dir. If
    // release re-canonicalized, it would compute a DIFFERENT path and never
    // find the insert to remove — the active-claim gate leaks the key for
    // the process lifetime. Fix: capture the key at claim time, use it
    // verbatim on release.
    let home = tempfile::tempdir().expect("home");
    let never_created = home.path().join("does-not-exist-yet/models");
    let claim =
        claim_start(&never_created, "acme/m", "m.gguf").expect("claim before the dir exists");
    // The dir now exists (with_ledger created it during the claim's write).
    assert!(never_created.exists(), "models dir was created mid-claim");
    claim.end(LedgerEnd::Cancelled, None);
    // A fresh claim on the same path succeeds — proving release freed the key.
    let _re = claim_start(&never_created, "acme/m", "m.gguf")
        .expect("release freed the key (no leak from re-canonicalization)");
}

#[test]
fn ledger_claim_drop_does_not_blanket_sweep_the_key() {
    // Post-r47: `LedgerClaim::drop` no longer calls `remove_partials` — the
    // per-transfer `TempGuard` in `download` already cleans this transfer's
    // specific tmp precisely, and a blanket sweep here would clip any
    // concurrent same-pid caller's live tmp on the same key. Test: hand-
    // place a decoy same-pid tmp for the key, drop the LedgerClaim, assert
    // the decoy survives.
    let (_home, models) = root();
    let dir = models.join("acme/m");
    std::fs::create_dir_all(&dir).expect("dir");
    let pid = std::process::id();
    let decoy_same_key = dir.join(format!("m.gguf.part.{pid}.99"));
    std::fs::write(&decoy_same_key, b"another caller's live tmp").expect("fixture");

    let claim = claim_start(&models, "acme/m", "m.gguf").expect("claim");
    drop(claim);
    assert!(
        decoy_same_key.exists(),
        "LedgerClaim::drop must NOT blanket-sweep the key — another caller's tmp survives"
    );
    // Sanity: the drop still recorded a Cancelled terminal.
    assert_eq!(read_all(&models)[0].status, DownloadLedgerStatus::Cancelled);
    // Clean up the decoy so the tempdir teardown doesn't complain.
    let _ = std::fs::remove_file(&decoy_same_key);
}

#[test]
fn pid_start_time_carries_sub_second_precision() {
    // Same-second PID reuse is a real macOS hazard: a downloader crashes,
    // its PID gets recycled to a long-lived unrelated process within the
    // same wall-clock second. A seconds-only `pid_started_at` would then
    // treat the stale row as live and refuse HG090 forever. Two calls to
    // `pid_start_time` for OUR pid must return the SAME value (invariant:
    // start time is stable for a live process), but the value must have
    // sub-second resolution — check that at least one significant digit
    // lies below the 1_000_000 boundary for a reasonable-uptime process,
    // AND that same-pid queries agree.
    let a = pid_start_time(std::process::id());
    let b = pid_start_time(std::process::id());
    assert_eq!(a, b, "start-time stamp is stable for a live pid");
    if let Some(v) = a {
        // Post-r48 macOS: sec * 1e6 + usec. If sub-second precision is
        // stripped (seconds-only), `v` is always divisible by 1_000_000.
        // Real usec is essentially never 0 — check both sides of the
        // boundary to detect a regression to seconds-only.
        //
        // On Linux the stamp is in clock ticks since boot, sub-second by
        // nature; on other unqueryable platforms this returns None and
        // we skip.
        #[cfg(target_os = "macos")]
        assert!(
            v % 1_000_000 != 0,
            "macOS start-time must include the microsecond component (got {v})"
        );
    }
}

// The cross-process residue/repair/overwrite tests are deleted: the ledger
// no longer has a repair predicate to attack. The download-lock flock owns
// authority; a stale `Downloading` row of any origin is purged by the next
// claim without ceremony, and a genuine live conflict is refused at the
// flock (`download_lock::acquire`), never here.

#[test]
fn a_downloading_row_whose_flock_is_unheld_is_swept_on_read() {
    // The stale-visibility fix: a foreign process finishes a download,
    // drops its `DownloadLock`, its terminal ledger write fails (queued
    // only in its own process memory — invisible to us). Every other
    // process's next `read_all` sees the stale `Downloading` row. With
    // the fs2-driven sweep, we notice the flock file exists but nobody
    // holds it → flip the row to `Failed` with a clear detail line, so
    // the fleet view stops lying.
    let (_home, models) = root();
    // Simulate the foreign process by acquiring + dropping the lock (the
    // lock file persists after drop) and leaving a stale `Downloading`
    // row for the same key.
    {
        let _one =
            crate::catalog::download_lock::DownloadLock::acquire(&models, "acme/m", "m.gguf")
                .expect("acquire");
    } // lock file persists; row will "look live" without this fix
    let stale = DownloadLedgerEntry {
        repo: "acme/m".into(),
        file: "m.gguf".into(),
        pid: std::process::id(), // pid alive → not caught by dead-pid sweep
        pid_started_at: pid_start_time(std::process::id()),
        started_at_ms: 1,
        downloaded: 42,
        total: Some(100),
        status: DownloadLedgerStatus::Downloading,
        ended_at_ms: None,
        path: None,
        detail: None,
    };
    std::fs::write(
        ledger_path(&models),
        serde_json::to_vec(&vec![stale]).unwrap(),
    )
    .unwrap();
    // Fresh read runs the flock-driven sweep → row flips to Failed.
    let all = read_all(&models);
    assert_eq!(all.len(), 1);
    assert_eq!(all[0].status, DownloadLedgerStatus::Failed);
    let detail = all[0].detail.as_deref().unwrap_or_default();
    assert!(
        detail.contains("released the machine lock"),
        "flock-driven sweep records its distinct reason: {detail}"
    );
}

#[test]
fn an_ancient_lockless_downloading_row_is_reaped_on_read() {
    // Legacy sweep: a `Downloading` row with NO download-lock file (a
    // pre-lock-build entry, or a writer that died between the ledger
    // insert and the lock create) has no owner the flock can attest to.
    // `is_key_stale` deliberately says "not stale" for a missing lock
    // file, so without the age bound the row would be announced forever.
    // After LOCKLESS_ROW_MAX_AGE_MS it is reaped with a distinct detail.
    let (_home, models) = root();
    let ancient = DownloadLedgerEntry {
        repo: "acme/m".into(),
        file: "m.gguf".into(),
        pid: std::process::id(), // alive — pid sweep must NOT be the reaper here
        pid_started_at: pid_start_time(std::process::id()),
        started_at_ms: 1, // epoch — way past any grace window
        downloaded: 10,
        total: Some(100),
        status: DownloadLedgerStatus::Downloading,
        ended_at_ms: None,
        path: None,
        detail: None,
    };
    std::fs::write(
        ledger_path(&models),
        serde_json::to_vec(&vec![ancient]).unwrap(),
    )
    .unwrap();
    let all = read_all(&models);
    assert_eq!(all.len(), 1);
    assert_eq!(all[0].status, DownloadLedgerStatus::Failed);
    assert!(
        all[0]
            .detail
            .as_deref()
            .unwrap_or_default()
            .contains("stale lockless entry"),
        "distinct legacy-reap reason: {:?}",
        all[0].detail
    );
    // A FRESH lockless row (bootstrap window) is NOT reaped.
    let fresh = DownloadLedgerEntry {
        started_at_ms: now_ms(),
        ..all[0].clone()
    };
    let fresh = DownloadLedgerEntry {
        status: DownloadLedgerStatus::Downloading,
        ended_at_ms: None,
        detail: None,
        ..fresh
    };
    std::fs::write(
        ledger_path(&models),
        serde_json::to_vec(&vec![fresh]).unwrap(),
    )
    .unwrap();
    let all = read_all(&models);
    assert_eq!(
        all[0].status,
        DownloadLedgerStatus::Downloading,
        "fresh lockless row survives the grace window"
    );
}

#[test]
fn the_err_cancel_flip_never_retargets_a_prior_attempts_failed_history() {
    // r71 finding: `overwrite_last_failed_to_cancelled`'s file pass had no
    // binding to the current attempt — when the current attempt's Failed
    // terminal was already reconciled (queued+drained), the unbounded
    // max_by_key fell through to a PRIOR attempt's genuine Failed row and
    // rewrote real history as an operator cancel. The `attempt_started_ms`
    // bound closes it: rows ended BEFORE the attempt began are never
    // candidates.
    let (_home, models) = root();
    // Yesterday's genuine failure for the same key (terminal history).
    let old_fail = DownloadLedgerEntry {
        repo: "acme/m".into(),
        file: "m.gguf".into(),
        pid: std::process::id(),
        pid_started_at: pid_start_time(std::process::id()),
        started_at_ms: now_ms().saturating_sub(60_000),
        downloaded: 5,
        total: Some(100),
        status: DownloadLedgerStatus::Failed,
        ended_at_ms: Some(now_ms().saturating_sub(50_000)),
        path: None,
        detail: Some("404 not found".into()),
    };
    std::fs::write(
        ledger_path(&models),
        serde_json::to_vec(&vec![old_fail]).unwrap(),
    )
    .unwrap();
    // Current attempt starts NOW; its own Failed terminal never landed
    // (queued-and-drained scenario) — the flip must find NOTHING to
    // rewrite, leaving yesterday's row intact.
    let attempt_started = now_ms();
    overwrite_last_failed_to_cancelled(&models, "acme/m", "m.gguf", attempt_started);
    let all = read_all(&models);
    assert_eq!(all.len(), 1);
    assert_eq!(
        all[0].status,
        DownloadLedgerStatus::Failed,
        "prior attempt's genuine failure is never rewritten"
    );
    assert_eq!(
        all[0].detail.as_deref(),
        Some("404 not found"),
        "its detail survives"
    );
    // Control: a Failed row ended WITHIN the attempt window IS flipped.
    let fresh_fail = DownloadLedgerEntry {
        started_at_ms: attempt_started,
        ended_at_ms: Some(now_ms()),
        detail: Some("transient".into()),
        ..all[0].clone()
    };
    std::fs::write(
        ledger_path(&models),
        serde_json::to_vec(&vec![fresh_fail]).unwrap(),
    )
    .unwrap();
    overwrite_last_failed_to_cancelled(&models, "acme/m", "m.gguf", attempt_started);
    let all = read_all(&models);
    assert_eq!(
        all[0].status,
        DownloadLedgerStatus::Cancelled,
        "the current attempt's own Failed is flipped"
    );
    assert!(all[0].detail.is_none(), "detail cleared on the flip");
}
