use super::*;

// ── registry semantics ─────────────────────────────────────────────────────

#[test]
fn cancelling_an_unregistered_pull_is_a_clean_coded_error() {
    let reg = PullCancelRegistry::new();
    let err = reg
        .cancel(None, "acme/m", "m.gguf")
        .expect_err("nothing in flight");
    assert!(matches!(err, HiggsError::DownloadFailed { .. }));
    assert!(err.to_string().contains("no in-flight download to cancel"));
}

#[test]
fn cancel_fires_the_registered_receiver_and_guard_drop_deregisters() {
    let reg = PullCancelRegistry::new();
    let (guard, rx, _prog) = reg.register(None, "acme/m", "m.gguf").expect("register");
    assert!(reg.is_registered(None, "acme/m", "m.gguf"));
    assert!(!*rx.borrow());

    reg.cancel(None, "acme/m", "m.gguf").expect("registered");
    assert!(*rx.borrow(), "receiver observed the cancel");

    // A different key is untouched.
    assert!(reg.cancel(None, "acme/m", "other.gguf").is_err());
    // Node-keyed and local-keyed entries are distinct slots.
    assert!(reg.cancel(Some("n1"), "acme/m", "m.gguf").is_err());

    drop(guard);
    assert!(!reg.is_registered(None, "acme/m", "m.gguf"));
    assert!(
        reg.cancel(None, "acme/m", "m.gguf").is_err(),
        "deregistered on guard drop — a finished pull cannot be 'cancelled'"
    );
}

// ── cancellable_pull ───────────────────────────────────────────────────────

#[tokio::test]
async fn cancellable_pull_returns_the_result_when_never_cancelled() {
    let reg = PullCancelRegistry::new();
    let (_guard, rx, _prog) = reg.register(None, "acme/m", "m.gguf").expect("register");
    let root = tempfile::tempdir().expect("root");
    let out = cancellable_pull(
        async { Ok(PathBuf::from("/done")) },
        rx,
        root.path(),
        "acme/m",
        "m.gguf",
    )
    .await
    .expect("uncancelled pull completes");
    assert_eq!(out, PathBuf::from("/done"));
}

#[tokio::test]
async fn cancel_aborts_the_pull_and_sweeps_only_this_files_partials() {
    let reg = PullCancelRegistry::new();
    let (_guard, rx, _prog) = reg.register(None, "acme/m", "m-Q4.gguf").expect("register");
    let root = tempfile::tempdir().expect("root");
    let dir = root.path().join("acme/m");
    std::fs::create_dir_all(&dir).expect("dir");
    // The download's own `.part.<OUR pid>.<n>` temp (what a dropped future
    // leaves), a decoy partial of ANOTHER file, a SIBLING PROCESS's live temp
    // for the SAME file (different pid — its transfer is healthy and must not
    // be touched), and a finished sibling model.
    let pid = std::process::id();
    let partial = dir.join(format!("m-Q4.gguf.part.{pid}.0"));
    let decoy = dir.join(format!("other.gguf.part.{pid}.0"));
    let foreign = dir.join("m-Q4.gguf.part.999999.0");
    let finished = dir.join("m-Q4.gguf");
    for p in [&partial, &decoy, &foreign, &finished] {
        std::fs::write(p, b"bytes").expect("fixture");
    }

    reg.cancel(None, "acme/m", "m-Q4.gguf").expect("registered");
    let err = cancellable_pull(
        std::future::pending::<Result<PathBuf, HiggsError>>(),
        rx,
        root.path(),
        "acme/m",
        "m-Q4.gguf",
    )
    .await
    .expect_err("cancelled");
    assert!(
        matches!(err, HiggsError::DownloadCancelled { ref repo, ref file, partial_swept: true }
            if repo == "acme/m" && file == "m-Q4.gguf"),
        "cancel returns HG089: {err:?}"
    );
    // The cancel path no longer sweeps: each `download` transfer owns a
    // per-attempt drop guard (`TempGuard`) that unlinks THAT transfer's
    // specific tmp on drop, so cancellation of a real download cleans
    // its own temp precisely. `cancellable_pull` blanket-sweeping same-
    // pid temps for the key would erase concurrent same-pid transfers'
    // live tmps (r45 finding), which is what this design change closes.
    // Fixtures are untouched by the cancel; they get cleaned by their
    // owning `download` future's own TempGuard when that fn is what
    // wrote them.
    assert!(partial.exists(), "cancel path no longer blanket-sweeps");
    assert!(decoy.exists(), "another file's partial untouched");
    assert!(
        foreign.exists(),
        "a sibling PROCESS's live temp for the same file must survive"
    );
    assert!(finished.exists(), "the final destination is never touched");
}

#[test]
fn cancel_after_the_receiver_died_is_no_in_flight_not_ok() {
    // The transfer resolved (its rx dropped) but the guard's drop hasn't
    // landed yet: a cancel in that window must NOT claim "cancel accepted" —
    // the terminal result is already decided. It answers the same coded
    // "no in-flight download" as a missing key. The entry is NOT swept here:
    // the live guard is its SOLE owner, and sweeping would let a fresh
    // same-key registration slip in before the guard's unconditional drop
    // deleted the NEW pull's sender (ABA).
    let reg = PullCancelRegistry::new();
    let (guard, rx, _prog) = reg.register(None, "acme/m", "m.gguf").expect("register");
    drop(rx); // cancellable_pull returned; only the guard remains
    let err = reg
        .cancel(None, "acme/m", "m.gguf")
        .expect_err("dead receiver must not be a successful cancel");
    assert!(
        err.to_string().contains("no in-flight download to cancel"),
        "{err}"
    );
    assert!(
        reg.is_registered(None, "acme/m", "m.gguf"),
        "the guard stays sole owner — no sweep in cancel()"
    );
    // A re-register in this microscopic window is refused, honestly.
    assert!(reg.register(None, "acme/m", "m.gguf").is_err());
    drop(guard); // the guard's drop is the ONE place the key is freed
    assert!(!reg.is_registered(None, "acme/m", "m.gguf"));
    let _ = reg
        .register(None, "acme/m", "m.gguf")
        .expect("key free after the guard dropped");
}

#[test]
fn node_registry_is_one_process_global() {
    assert!(std::ptr::eq(node_registry(), node_registry()));
}

#[test]
fn a_duplicate_registration_is_rejected_while_the_first_is_in_flight() {
    // One transfer per key: a hub re-issue after a reconnect races the
    // still-running original and must be REFUSED ("already in flight"),
    // never started as a second copy. The original stays cancellable, and
    // once it ends (guard drop) the key is free again.
    let reg = PullCancelRegistry::new();
    let (guard_1, mut rx_1, _prog_1) = reg
        .register(None, "acme/m", "m-Q4.gguf")
        .expect("first register");
    let Err(err) = reg.register(None, "acme/m", "m-Q4.gguf") else {
        panic!("duplicate must be refused");
    };
    assert!(
        matches!(err, HiggsError::DownloadInFlight { .. }),
        "duplicate refusal is [HG090] — code-classifiable, not the HG025 umbrella: {err}"
    );
    assert!(err.to_string().contains("already in flight"), "{err}");
    // The refusal did not disturb the original registration.
    reg.cancel(None, "acme/m", "m-Q4.gguf")
        .expect("original still cancellable");
    assert!(*rx_1.borrow_and_update(), "cancel reached the original");
    // A DIFFERENT file is not blocked.
    let (_g_other, _rx_other, _prog_other) = reg
        .register(None, "acme/m", "other.gguf")
        .expect("other file registers fine");
    // Original ends → key free → a fresh pull can start.
    drop(guard_1);
    let (_guard_2, _rx_2, _prog_2) = reg
        .register(None, "acme/m", "m-Q4.gguf")
        .expect("key free after the original ended");
}

#[tokio::test]
async fn a_cancel_signalled_before_the_first_poll_wins_over_an_instant_completion() {
    // The biased select resolves the MID-FLIGHT photo-finish in favor of
    // completion (HG089 must mean nothing landed). But a cancel that was
    // ACCEPTED before the transfer's first poll is a different case: nothing
    // has run yet, so honoring it is always sound — without the fast path, a
    // fetch fast enough to complete inside its FIRST poll would land as Done
    // after the operator's cancel was already accepted.
    let reg = PullCancelRegistry::new();
    let (_guard, rx, _prog) = reg.register(None, "acme/m", "m.gguf").expect("register");
    reg.cancel(None, "acme/m", "m.gguf")
        .expect("cancel accepted");
    let root = tempfile::tempdir().expect("root");
    let err = cancellable_pull(
        async { Ok(PathBuf::from("/instant")) },
        rx,
        root.path(),
        "acme/m",
        "m.gguf",
    )
    .await
    .expect_err("a pre-accepted cancel is honored, not raced");
    assert!(
        matches!(err, HiggsError::DownloadCancelled { .. }),
        "pre-signalled cancel → HG089: {err:?}"
    );
}

#[tokio::test]
async fn a_cancel_racing_an_instant_future_still_wins_the_cancel_regime() {
    // The r30 fast path handles cancel-before-called deterministically.
    // Post-r41 the pre-poll check runs INSIDE the loop too, so a cancel
    // signaled BETWEEN the fast-path check and the first poll of the future
    // is still caught (before the future runs). Test: pre-signal cancel,
    // pass a future that would resolve to Ok on its first poll — cancel
    // wins, HG089 returned, no Ok leaks.
    let reg = PullCancelRegistry::new();
    let (_guard, rx, _prog) = reg.register(None, "acme/m", "m.gguf").expect("register");
    reg.cancel(None, "acme/m", "m.gguf")
        .expect("cancel accepted");
    let root = tempfile::tempdir().expect("root");
    // A future that WOULD complete immediately if polled.
    let err = cancellable_pull(
        async { Ok(PathBuf::from("/instant")) },
        rx,
        root.path(),
        "acme/m",
        "m.gguf",
    )
    .await
    .expect_err("cancel wins over an instant future");
    assert!(
        matches!(err, HiggsError::DownloadCancelled { .. }),
        "HG089: {err:?}"
    );
}

#[test]
fn hg091_diagnostic_names_the_photo_finish_and_says_the_file_landed() {
    // Photo-finish outcome (operator ruling): cancel signal arrives but the
    // download had already completed. The diagnostic distinguishes "cancel
    // honored, nothing landed" (HG089) from "cancel outraced, file landed"
    // (HG091) — the "cancel failed with a signal on why it failed" API.
    // Emitted as a `warn!` by `cancellable_pull` when the biased select
    // returns Ok with cancel visible; deterministic reproduction of the
    // microsecond race inside a unit test is not achievable (that IS the
    // race), so this pin is the wire-visible surface: the code, severity,
    // and message.
    let e = HiggsError::CancelLostToCompletion {
        repo: "acme/m".into(),
        file: "m.gguf".into(),
    };
    let s = e.to_string();
    assert!(s.contains("[HG091]"), "diagnostic code present: {s}");
    assert!(
        s.contains("cancel requested but download") && s.contains("completed first"),
        "diagnostic names the race: {s}"
    );
    assert!(
        s.contains("file is on disk"),
        "diagnostic tells the honest outcome: {s}"
    );
    use miette::Diagnostic;
    assert_eq!(
        e.severity(),
        Some(miette::Severity::Warning),
        "HG091 is a signal, not an error"
    );
}

#[test]
fn hg089_message_is_honest_about_best_effort_cleanup() {
    // Post-r46 semantic: `partial_swept` is a wire back-compat field, not a
    // cleanup-verified guarantee. The per-transfer `TempGuard`'s drop is
    // best-effort (unlink failures visible only in tracing), so the message
    // must NOT claim "swept" — it says cleanup was ATTEMPTED and points to
    // the log if disk usage grows.
    let e = HiggsError::DownloadCancelled {
        repo: "acme/m".into(),
        file: "m.gguf".into(),
        partial_swept: true,
    };
    let s = e.to_string();
    assert!(s.contains("[HG089]"), "{s}");
    assert!(
        s.contains("cleanup attempted") && s.contains("best-effort"),
        "message reflects the best-effort reality: {s}"
    );
    assert!(
        s.contains("check the node log"),
        "message directs the operator to the log for the rare failure case: {s}"
    );
    // A `false` value renders the same — the field is historical, not
    // load-bearing on the message.
    let e2 = HiggsError::DownloadCancelled {
        repo: "acme/m".into(),
        file: "m.gguf".into(),
        partial_swept: false,
    };
    assert_eq!(
        e.to_string(),
        e2.to_string(),
        "the message no longer branches on the field"
    );
}

#[tokio::test]
async fn a_cancel_outraced_by_an_instant_ok_returns_the_path_with_hg091() {
    // Photo-finish, completion side: the future completes Ok in the SAME
    // poll where cancel arrives. Completion wins the tie (the file is on
    // disk — HG089 would lie), the caller gets Ok(path), and the HG091
    // outrace is only a log line. Reverting the biased-completion tie
    // rule to cancel-wins would turn this Ok into a false HG089.
    let reg = std::sync::Arc::new(PullCancelRegistry::new());
    let (_guard, rx, _prog) = reg.register(None, "acme/m", "m.gguf").expect("register");
    let root = tempfile::tempdir().expect("root");
    let landed = root.path().join("m.gguf");
    let cancel_then_ok = {
        let reg = reg.clone();
        let landed = landed.clone();
        async move {
            reg.cancel(None, "acme/m", "m.gguf")
                .expect("cancel accepted");
            Ok::<PathBuf, HiggsError>(landed)
        }
    };
    let got = cancellable_pull(cancel_then_ok, rx, root.path(), "acme/m", "m.gguf")
        .await
        .expect("completion wins the same-poll tie");
    assert_eq!(got, landed, "the landed path is surfaced, not HG089");
}

#[tokio::test]
async fn a_yielded_ok_outracing_a_cancel_returns_the_path_via_the_select_arm() {
    // Same completion-wins contract, but through the SELECT regime: the
    // future is Pending at `poll_immediate` (one yield), then completes
    // Ok in the select's first poll while signalling cancel from inside
    // it. The biased select polls the download arm first, so Ok wins and
    // the post-select outrace check logs HG091 instead of lying HG089.
    let reg = std::sync::Arc::new(PullCancelRegistry::new());
    let (_guard, rx, _prog) = reg.register(None, "acme/m", "m.gguf").expect("register");
    let root = tempfile::tempdir().expect("root");
    let landed = root.path().join("m.gguf");
    let yield_then_cancel_then_ok = {
        let reg = reg.clone();
        let landed = landed.clone();
        async move {
            tokio::task::yield_now().await;
            reg.cancel(None, "acme/m", "m.gguf")
                .expect("cancel accepted");
            Ok::<PathBuf, HiggsError>(landed)
        }
    };
    let got = cancellable_pull(
        yield_then_cancel_then_ok,
        rx,
        root.path(),
        "acme/m",
        "m.gguf",
    )
    .await
    .expect("completion wins the biased select tie");
    assert_eq!(got, landed);
}

#[tokio::test]
async fn a_yielded_err_racing_a_cancel_collapses_to_hg089_via_the_select_arm() {
    // The select-regime mirror of the instant-Err race: Pending at
    // `poll_immediate`, then the future errors in the same select poll
    // where cancel lands. The operator's stop-intent wins — HG089, not
    // the raw fetcher error — through the POST-SELECT outrace check
    // (the second of the two translation sites; reverting just that arm
    // would surface Failed{HG036} for an explicitly-stopped transfer).
    let reg = std::sync::Arc::new(PullCancelRegistry::new());
    let (_guard, rx, _prog) = reg.register(None, "acme/m", "m.gguf").expect("register");
    let root = tempfile::tempdir().expect("root");
    let yield_then_cancel_then_err = {
        let reg = reg.clone();
        async move {
            tokio::task::yield_now().await;
            reg.cancel(None, "acme/m", "m.gguf")
                .expect("cancel accepted");
            Err::<PathBuf, HiggsError>(HiggsError::HubFetchExhausted {
                repo: "acme/m".into(),
                file: "m.gguf".into(),
                primary: "primary boom".into(),
                fallback: "fallback boom".into(),
            })
        }
    };
    let err = cancellable_pull(
        yield_then_cancel_then_err,
        rx,
        root.path(),
        "acme/m",
        "m.gguf",
    )
    .await
    .expect_err("cancel + err collapses to HG089 in the select regime too");
    assert!(
        matches!(err, HiggsError::DownloadCancelled { .. }),
        "stop-intent wins over the raw fetcher error: {err:?}"
    );
}

#[test]
fn in_flight_announcement_is_bounded_to_sixteen_rows() {
    // The in_flight list rides HELLO (64 KiB frame cap): the PRODUCER
    // bound is 16 no matter how many transfers are registered. Guards
    // survive in a Vec so all 20 registrations stay live at read time.
    let reg = PullCancelRegistry::new();
    let mut guards = Vec::new();
    for i in 0..20 {
        let (g, _rx, _p) = reg
            .register(None, "acme/m", &format!("f{i}.gguf"))
            .expect("register");
        guards.push(g);
    }
    assert_eq!(
        reg.in_flight().len(),
        16,
        "producer-side HELLO-frame bound caps the announcement"
    );
}

#[tokio::test]
async fn a_cancel_racing_an_instant_err_collapses_to_hg089() {
    // Symmetric to the instant-Ok pre-signal test, but for the harder
    // Err+cancel race: a fetcher errors in the SAME poll where cancel
    // arrives. The operator's intent takes precedence — surface HG089
    // (nothing landed), not the raw fetcher error. Without this, the UI
    // shows Failed{HG036} for a transfer the operator explicitly stopped.
    //
    // We cannot simply pre-signal cancel (the pre-poll fast-path
    // short-circuits before the future runs). Instead we build a future
    // whose FIRST poll fires the cancel from inside it, then returns
    // Err. This forces the race through `poll_immediate`'s Err branch
    // AFTER cancel is visible, which is exactly where the fix lives.
    let reg = std::sync::Arc::new(PullCancelRegistry::new());
    let (_guard, rx, _prog) = reg.register(None, "acme/m", "m.gguf").expect("register");
    let root = tempfile::tempdir().expect("root");
    let cancel_from_inside = {
        let reg = reg.clone();
        async move {
            // Set cancel FROM INSIDE the future's first poll, then error.
            reg.cancel(None, "acme/m", "m.gguf")
                .expect("cancel accepted");
            Err::<PathBuf, HiggsError>(HiggsError::HubFetchExhausted {
                repo: "acme/m".into(),
                file: "m.gguf".into(),
                primary: "primary boom".into(),
                fallback: "fallback boom".into(),
            })
        }
    };
    let err = cancellable_pull(cancel_from_inside, rx, root.path(), "acme/m", "m.gguf")
        .await
        .expect_err("cancel + err collapses to HG089");
    assert!(
        matches!(err, HiggsError::DownloadCancelled { .. }),
        "cancel-honored intent wins over the raw fetcher error: {err:?}"
    );
}

#[test]
fn in_flight_reports_live_progress_and_maps_zero_total_to_unknown() {
    // The in_flight announcement is what a reconnecting hub sees: each row must
    // carry the pull's LIVE byte counters (fed lock-free via PullProgress::set)
    // and translate the stored 0 total back to `None` — a hub shown
    // `Some(0)` would render a 0-byte "total" progress bar for a server that
    // simply never sent a content length.
    let reg = PullCancelRegistry::new();
    let (_g_local, _rx1, prog_local) = reg
        .register(None, "acme/m", "a.gguf")
        .expect("local register");
    let (_g_node, _rx2, _prog_node) = reg
        .register(Some("n1"), "acme/m", "b.gguf")
        .expect("node register");
    // The download's progress callback ticked: 5 of 100 bytes.
    prog_local.set(5, Some(100));

    let rows = reg.in_flight();
    assert_eq!(rows.len(), 2);
    let a = rows.iter().find(|r| r.file == "a.gguf").expect("local row");
    assert_eq!(
        (a.downloaded, a.total),
        (5, Some(100)),
        "live counters reach the announcement"
    );
    assert_eq!(a.node, None, "local pulls announce node=None");
    let b = rows.iter().find(|r| r.file == "b.gguf").expect("node row");
    assert_eq!(
        (b.downloaded, b.total),
        (0, None),
        "an un-ticked pull reports 0 downloaded and UNKNOWN total, not Some(0)"
    );
    assert_eq!(b.node.as_deref(), Some("n1"));
}
