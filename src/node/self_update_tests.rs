use super::*;
use crate::update::UpdateManifest;

// ---- helpers -------------------------------------------------------------

fn manifest(version: &str, target: &str, variant: &str) -> UpdateManifest {
    UpdateManifest {
        schema: 1,
        version: version.to_string(),
        commit: "deadbeef".to_string(),
        file: format!("higgs-v{version}-{target}.tar.gz"),
        target: target.to_string(),
        variant: variant.to_string(),
        sha256: "00".to_string(),
    }
}

fn ident(version: &str, target: &str, variant: &str) -> BuildIdentity {
    BuildIdentity {
        version: version.to_string(),
        target: target.to_string(),
        variant: variant.to_string(),
    }
}

/// A gzip'd tar carrying a single REGULAR executable file `higgs` (mode 0755)
/// with the given bytes — the shape [`stage_and_flip`] expects to unpack.
fn tarball_with_higgs(bytes: &[u8], mode: u32) -> Vec<u8> {
    let mut out = Vec::new();
    {
        let enc = flate2::write::GzEncoder::new(&mut out, flate2::Compression::default());
        let mut b = tar::Builder::new(enc);
        let mut h = tar::Header::new_gnu();
        h.set_size(bytes.len() as u64);
        h.set_mode(mode);
        h.set_entry_type(tar::EntryType::Regular);
        h.set_cksum();
        b.append_data(&mut h, "higgs", bytes).unwrap();
        b.finish().unwrap();
    }
    out
}

/// A gzip'd tar with a regular `higgs` PLUS a NESTED `tools/helper` entry (which unpack
/// materializes as a `tools/` subdir) — the non-flat, potentially-setuid-smuggling layout
/// the publisher must refuse.
fn tarball_with_nested() -> Vec<u8> {
    let mut out = Vec::new();
    {
        let enc = flate2::write::GzEncoder::new(&mut out, flate2::Compression::default());
        let mut b = tar::Builder::new(enc);
        let mut h = tar::Header::new_gnu();
        h.set_size(3);
        h.set_mode(0o755);
        h.set_entry_type(tar::EntryType::Regular);
        h.set_cksum();
        b.append_data(&mut h, "higgs", &b"bin"[..]).unwrap();
        let mut h2 = tar::Header::new_gnu();
        h2.set_size(1);
        h2.set_mode(0o4755); // setuid, smuggled in a subdir
        h2.set_entry_type(tar::EntryType::Regular);
        h2.set_cksum();
        b.append_data(&mut h2, "tools/helper", &b"x"[..]).unwrap();
        b.finish().unwrap();
    }
    out
}

/// A gzip'd tar with a regular `higgs`, a `locked/file`, and THEN a `locked/` directory
/// entry with mode `0o000` (chmod'ing the already-created dir unreadable) — a non-flat
/// layout the publisher rejects, but whose restrictive dir mode would defeat a naive
/// `remove_dir_all` staging cleanup. The dir entry comes AFTER its file so the file
/// extracts into a still-writable dir (the mode is clamped last).
fn tarball_with_a_restrictive_dir() -> Vec<u8> {
    let mut out = Vec::new();
    {
        let enc = flate2::write::GzEncoder::new(&mut out, flate2::Compression::default());
        let mut b = tar::Builder::new(enc);
        let mut h = tar::Header::new_gnu();
        h.set_size(3);
        h.set_mode(0o755);
        h.set_entry_type(tar::EntryType::Regular);
        h.set_cksum();
        b.append_data(&mut h, "higgs", &b"bin"[..]).unwrap();
        let mut hf = tar::Header::new_gnu();
        hf.set_size(1);
        hf.set_mode(0o644);
        hf.set_entry_type(tar::EntryType::Regular);
        hf.set_cksum();
        b.append_data(&mut hf, "locked/file", &b"x"[..]).unwrap();
        let mut hd = tar::Header::new_gnu();
        hd.set_size(0);
        hd.set_mode(0o000); // clamp the dir unreadable/untraversable
        hd.set_entry_type(tar::EntryType::Directory);
        hd.set_cksum();
        b.append_data(&mut hd, "locked/", &[][..]).unwrap();
        b.finish().unwrap();
    }
    out
}

/// Write `bytes` to `path` as a runnable (0755) regular file — the shape a real
/// `v<ver>/higgs` has, which the rollback-target check now requires.
fn write_runnable(path: &std::path::Path, bytes: &[u8]) {
    use std::os::unix::fs::PermissionsExt;
    std::fs::write(path, bytes).unwrap();
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755)).unwrap();
}

/// Lay down a bin dir with a `v<ver>/higgs` version dir and `current -> v<ver>`,
/// as install.sh would. Returns the bin path.
fn install_layout(root: &std::path::Path, ver: &str) -> std::path::PathBuf {
    let bin = root.join("bin");
    let vdir = bin.join(format!("v{ver}"));
    std::fs::create_dir_all(&vdir).unwrap();
    write_runnable(&vdir.join("higgs"), b"#!/bin/sh\n");
    std::os::unix::fs::symlink(format!("v{ver}"), bin.join("current")).unwrap();
    bin
}

// ---- eligibility (pure) --------------------------------------------------

#[test]
fn eligibility_accepts_a_newer_matching_build() {
    let running = ident("1.2.3", "aarch64-apple-darwin", "metal");
    let m = manifest("1.2.4", "aarch64-apple-darwin", "metal");
    assert!(evaluate_eligibility(&running, &m, false).is_ok());
}

#[test]
fn eligibility_refuses_a_downgrade_without_the_flag() {
    let running = ident("1.2.3", "aarch64-apple-darwin", "metal");
    let m = manifest("1.2.2", "aarch64-apple-darwin", "metal");
    match evaluate_eligibility(&running, &m, false) {
        Err(HiggsError::UpdateNotNewer { from, to }) => {
            assert_eq!(from, "1.2.3");
            assert_eq!(to, "1.2.2");
        }
        other => panic!("expected HG085 downgrade refusal, got {other:?}"),
    }
}

#[test]
fn eligibility_refuses_an_equal_version() {
    let running = ident("1.2.3", "aarch64-apple-darwin", "metal");
    let m = manifest("1.2.3", "aarch64-apple-darwin", "metal");
    assert!(matches!(
        evaluate_eligibility(&running, &m, false),
        Err(HiggsError::UpdateNotNewer { .. })
    ));
}

#[test]
fn eligibility_allows_a_downgrade_with_the_flag() {
    let running = ident("1.2.3", "aarch64-apple-darwin", "metal");
    let m = manifest("1.0.0", "aarch64-apple-darwin", "metal");
    assert!(evaluate_eligibility(&running, &m, true).is_ok());
}

#[test]
fn eligibility_refuses_an_equal_version_even_with_the_downgrade_flag() {
    // A same-version "update" would overwrite the live version dir before smoke and
    // record prev == to, so it is refused ALWAYS — --allow-downgrade only relaxes
    // strictly-older. (Reinstalling the same version is install.sh's job.)
    let running = ident("1.2.3", "aarch64-apple-darwin", "metal");
    let m = manifest("1.2.3", "aarch64-apple-darwin", "metal");
    assert!(matches!(
        evaluate_eligibility(&running, &m, true),
        Err(HiggsError::UpdateNotNewer { .. })
    ));
}

#[test]
fn eligibility_refuses_a_target_mismatch() {
    let running = ident("1.2.3", "aarch64-apple-darwin", "metal");
    let m = manifest("2.0.0", "x86_64-unknown-linux-gnu", "cpu");
    match evaluate_eligibility(&running, &m, false) {
        Err(HiggsError::UpdateTargetMismatch { field, .. }) => assert_eq!(field, "target"),
        other => panic!("expected HG086 target mismatch, got {other:?}"),
    }
}

#[test]
fn eligibility_refuses_a_variant_mismatch() {
    let running = ident("1.2.3", "x86_64-unknown-linux-gnu", "cpu");
    let m = manifest("2.0.0", "x86_64-unknown-linux-gnu", "cuda");
    match evaluate_eligibility(&running, &m, false) {
        Err(HiggsError::UpdateTargetMismatch {
            field,
            manifest,
            running,
        }) => {
            assert_eq!(field, "variant");
            assert_eq!(manifest, "cuda");
            assert_eq!(running, "cpu");
        }
        other => panic!("expected HG086 variant mismatch, got {other:?}"),
    }
}

#[test]
fn eligibility_checks_target_before_version() {
    // A wrong-target OLDER manifest reports the TARGET mismatch (the fundamental
    // "wrong build" refusal), never a version quibble about an unrunnable artifact.
    let running = ident("5.0.0", "aarch64-apple-darwin", "metal");
    let m = manifest("1.0.0", "x86_64-unknown-linux-gnu", "cpu");
    assert!(matches!(
        evaluate_eligibility(&running, &m, false),
        Err(HiggsError::UpdateTargetMismatch { .. })
    ));
}

#[test]
fn eligibility_honours_semver_prerelease_precedence() {
    // 1.0.0-rc.1 < 1.0.0 (a prerelease is OLDER than its release) — so shipping
    // the prerelease over the release is a downgrade.
    let running = ident("1.0.0", "aarch64-apple-darwin", "metal");
    let m = manifest("1.0.0-rc.1", "aarch64-apple-darwin", "metal");
    assert!(matches!(
        evaluate_eligibility(&running, &m, false),
        Err(HiggsError::UpdateNotNewer { .. })
    ));
    // …and the release over the prerelease IS an upgrade.
    let running = ident("1.0.0-rc.1", "aarch64-apple-darwin", "metal");
    let m = manifest("1.0.0", "aarch64-apple-darwin", "metal");
    assert!(evaluate_eligibility(&running, &m, false).is_ok());
}

#[test]
fn eligibility_refuses_equal_precedence_build_metadata() {
    // Build metadata does NOT affect semver PRECEDENCE — `1.2.3+aaa` and `1.2.3+zzz`
    // rank equal, so one is not an upgrade over the other (must refuse, not accept via
    // the total order).
    let running = ident("1.2.3+aaa", "aarch64-apple-darwin", "metal");
    let m = manifest("1.2.3+zzz", "aarch64-apple-darwin", "metal");
    assert!(matches!(
        evaluate_eligibility(&running, &m, false),
        Err(HiggsError::UpdateNotNewer { .. })
    ));
}

#[test]
fn eligibility_refuses_a_case_only_version_difference() {
    // On a case-INSENSITIVE fs `v1.0.0-A` and `v1.0.0-a` are the SAME dir, so an
    // "upgrade" between case-only variants would overwrite the live version dir.
    let running = ident("1.0.0-A", "aarch64-apple-darwin", "metal");
    let m = manifest("1.0.0-a", "aarch64-apple-darwin", "metal");
    assert!(matches!(
        evaluate_eligibility(&running, &m, false),
        Err(HiggsError::UpdateNotNewer { .. })
    ));
}

#[test]
fn eligibility_rejects_a_non_semver_manifest_version() {
    let running = ident("1.2.3", "aarch64-apple-darwin", "metal");
    let m = manifest("not-a-version", "aarch64-apple-darwin", "metal");
    assert!(matches!(
        evaluate_eligibility(&running, &m, false),
        Err(HiggsError::UpdateManifestInvalid { .. })
    ));
}

// ---- rollback decision (pure) --------------------------------------------

#[test]
fn rollback_decision_none_without_a_trial() {
    assert_eq!(decide_rollback(None, 9, BOOT_FAIL_BUDGET), None);
}

#[test]
fn rollback_decision_none_under_budget() {
    let t = TrialMarker {
        to: "v2".into(),
        prev: Some("v1".into()),
    };
    assert_eq!(decide_rollback(Some(&t), 2, 3), None);
}

#[test]
fn rollback_decision_at_budget_returns_prev() {
    let t = TrialMarker {
        to: "v2".into(),
        prev: Some("v1".into()),
    };
    assert_eq!(decide_rollback(Some(&t), 3, 3), Some("v1".into()));
}

#[test]
fn rollback_decision_none_over_budget_without_prev() {
    // A first install (no prev) that crash-loops has nothing to roll back to.
    let t = TrialMarker {
        to: "v2".into(),
        prev: None,
    };
    assert_eq!(decide_rollback(Some(&t), 9, 3), None);
}

#[test]
fn installed_identity_uses_the_installed_current_version() {
    // Eligibility must judge against the version INSTALLED as `current`, not the
    // invoking process — so a stale old binary can't downgrade the live install.
    let tmp = tempfile::tempdir().unwrap();
    let bin = install_layout(tmp.path(), "3.0.0"); // current -> v3.0.0
    assert_eq!(installed_identity(&bin).version, "3.0.0");
}

// ---- version-dir naming + install guard ----------------------------------

#[test]
fn version_dir_name_requires_a_real_semver() {
    assert!(is_version_dir_name("v1.0.0"));
    assert!(is_version_dir_name("v2.3.4-rc.1"));
    assert!(!is_version_dir_name("valuable")); // starts with v but not v<semver>
    assert!(!is_version_dir_name("var"));
    assert!(!is_version_dir_name("v"));
    assert!(!is_version_dir_name("1.0.0")); // no leading v
}

// ---- current_target ------------------------------------------------------

#[test]
fn current_target_reads_the_symlink_name() {
    let tmp = tempfile::tempdir().unwrap();
    let bin = install_layout(tmp.path(), "1.0.0");
    assert_eq!(current_target(&bin), Some("v1.0.0".to_string()));
}

#[test]
fn current_target_is_none_without_a_current_link() {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(tmp.path().join("bin")).unwrap();
    assert_eq!(current_target(&tmp.path().join("bin")), None);
}

// ---- boot-guard file wrappers --------------------------------------------

fn write_trial(bin: &std::path::Path, to: &str, prev: Option<&str>) {
    let marker = TrialMarker {
        to: to.into(),
        prev: prev.map(str::to_string),
    };
    std::fs::write(
        bin.join(".update-trial"),
        serde_json::to_vec(&marker).unwrap(),
    )
    .unwrap();
}

#[test]
fn boot_fail_counter_is_written_owner_only_not_umask_dependent() {
    // The counter is written from the BOOT path (record_boot_attempt), whose umask is the
    // daemon's INHERITED one (only stage_and_flip clamps 022). write_atomic must fchmod it
    // to 0600 regardless: a umask-derived 000 counter reads back as 0 (budget never spends
    // → crash-loop, no rollback), and a 0666 one is peer-writable. Under the standard test
    // umask (022) the unfixed File::create yields 0644; the fchmod makes it exactly 0600,
    // so this asserts the fchmod ran. Reverting the fchmod leaves 0644 and fails.
    use std::os::unix::fs::PermissionsExt;
    let tmp = tempfile::tempdir().unwrap();
    let bin = install_layout(tmp.path(), "2.0.0");
    write_trial(&bin, "v2.0.0", Some("v1.0.0"));
    record_boot_attempt(&bin, "2.0.0").unwrap();
    let mode = std::fs::metadata(bin.join(".update-bootfails"))
        .unwrap()
        .permissions()
        .mode()
        & 0o777;
    assert_eq!(
        mode, 0o600,
        "the boot-fail counter must be forced owner-only (fchmod), not left umask-masked"
    );
}

#[test]
fn boot_rollback_none_and_record_clears_counter_without_a_trial() {
    let tmp = tempfile::tempdir().unwrap();
    let bin = install_layout(tmp.path(), "1.0.0");
    std::fs::write(bin.join(".update-bootfails"), "2").unwrap();
    // No trial → rollback is a no-op; record_boot_attempt clears the stale counter.
    assert_eq!(boot_rollback_if_spent(&bin).unwrap(), None);
    record_boot_attempt(&bin, "1.0.0").unwrap();
    assert!(
        !bin.join(".update-bootfails").exists(),
        "stale counter cleared"
    );
}

#[test]
fn record_boot_attempt_increments_only_for_the_trialed_version() {
    let tmp = tempfile::tempdir().unwrap();
    let bin = install_layout(tmp.path(), "2.0.0");
    write_trial(&bin, "v2.0.0", Some("v1.0.0"));
    // This process IS v2.0.0 → increment.
    record_boot_attempt(&bin, "2.0.0").unwrap();
    assert_eq!(
        std::fs::read_to_string(bin.join(".update-bootfails")).unwrap(),
        "1"
    );
    // An OLD v1.0.0 daemon booting must NOT touch v2.0.0's counter.
    record_boot_attempt(&bin, "1.0.0").unwrap();
    assert_eq!(
        std::fs::read_to_string(bin.join(".update-bootfails")).unwrap(),
        "1",
        "a different running version leaves the counter alone"
    );
    assert_eq!(current_target(&bin), Some("v2.0.0".to_string()));
}

#[test]
fn boot_rollback_rolls_back_at_budget() {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(tmp.path().join("bin/v1.0.0")).unwrap();
    write_runnable(&tmp.path().join("bin/v1.0.0/higgs"), b"old");
    let bin = install_layout(tmp.path(), "2.0.0");
    write_trial(&bin, "v2.0.0", Some("v1.0.0"));
    std::fs::write(bin.join(".update-bootfails"), BOOT_FAIL_BUDGET.to_string()).unwrap();
    assert_eq!(
        boot_rollback_if_spent(&bin).unwrap(),
        Some("v1.0.0".to_string())
    );
    assert_eq!(
        current_target(&bin),
        Some("v1.0.0".to_string()),
        "rolled back"
    );
    assert!(!bin.join(".update-trial").exists(), "trial cleared");
    assert!(!bin.join(".update-bootfails").exists(), "counter cleared");
}

/// End-to-end accrual: a trialed binary that RECORDS a boot attempt every start but never commits
/// (`confirm_alive`) — the exact shape of an EARLY-INIT crash-loop, now that `record_boot_attempt`
/// runs BEFORE the risky daemon init — accrues and rolls back after `BOOT_FAIL_BUDGET`. Models the
/// real per-boot order: preflight CHECK (`boot_rollback_if_spent`) THEN daemon-body RECORD. Guards
/// the fix: were the record placed after the crash point (the old serve-commit), or the budget check
/// disabled, the crash-loop would never roll back.
#[test]
fn early_crash_boots_without_confirm_accrue_and_roll_back() {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(tmp.path().join("bin/v1.0.0")).unwrap();
    write_runnable(&tmp.path().join("bin/v1.0.0/higgs"), b"old");
    let bin = install_layout(tmp.path(), "2.0.0");
    write_trial(&bin, "v2.0.0", Some("v1.0.0"));
    // BOOT_FAIL_BUDGET boots that each CHECK (not yet spent), RECORD, then "crash" before confirm.
    for boot in 1..=BOOT_FAIL_BUDGET {
        assert_eq!(
            boot_rollback_if_spent(&bin).unwrap(),
            None,
            "boot {boot}: below budget, stay on the trial binary"
        );
        assert_eq!(current_target(&bin), Some("v2.0.0".to_string()));
        record_boot_attempt(&bin, "2.0.0").unwrap();
    }
    // The NEXT start's preflight check now sees the spent budget → rollback to prev.
    assert_eq!(
        boot_rollback_if_spent(&bin).unwrap(),
        Some("v1.0.0".to_string()),
        "budget spent by early-crash boots → rolled back"
    );
    assert_eq!(current_target(&bin), Some("v1.0.0".to_string()));
    assert!(
        !bin.join(".update-bootfails").exists(),
        "counter cleared on rollback"
    );
}

#[test]
fn record_failed_target_recovers_from_a_directory_at_the_poison_path() {
    // If `.update-failed` is a stray DIRECTORY — even a NON-EMPTY, NON-WRITABLE (0555) one —
    // `write_atomic`'s rename cannot replace it, so the poison could never persist and a
    // crash-looping version would loop UNBOUNDEDLY. The record must `chmod` it writable and clear
    // it first. Reverting that leaves the write failing + the version un-poisoned.
    use std::os::unix::fs::PermissionsExt;
    let tmp = tempfile::tempdir().unwrap();
    let bin = install_layout(tmp.path(), "1.0.0");
    let poison = bin.join(".update-failed");
    std::fs::create_dir_all(poison.join("child")).unwrap(); // a non-empty dir
    std::fs::set_permissions(&poison, std::fs::Permissions::from_mode(0o555)).unwrap(); // non-writable
    record_failed_target(&bin, "v2.0.0");
    assert!(
        is_failed_target(&bin, "v2.0.0"),
        "the poison persists even when a non-writable directory blocked the path"
    );
    assert!(poison.is_file(), "the poison path is now a regular file");
}

#[test]
fn boot_rollback_poisons_the_failed_target() {
    // A crash-looping trial that gets rolled back must POISON the failed version so a hub
    // re-pushing it is refused (idempotent retry after a rollback — else apply→crash→rollback
    // loops forever). Reverting the `record_failed_target` in `perform_trial_rollback` leaves
    // `is_failed_target` false and this fails.
    let tmp = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(tmp.path().join("bin/v1.0.0")).unwrap();
    write_runnable(&tmp.path().join("bin/v1.0.0/higgs"), b"old");
    let bin = install_layout(tmp.path(), "2.0.0");
    write_trial(&bin, "v2.0.0", Some("v1.0.0"));
    std::fs::write(bin.join(".update-bootfails"), BOOT_FAIL_BUDGET.to_string()).unwrap();
    assert_eq!(
        boot_rollback_if_spent(&bin).unwrap(),
        Some("v1.0.0".to_string())
    );
    assert!(
        is_failed_target(&bin, "v2.0.0"),
        "the rolled-back version is poisoned"
    );
    assert!(
        !is_failed_target(&bin, "v3.0.0"),
        "an unrelated version is NOT poisoned"
    );
}

// ---- last-failure marker (P4b (d): reported to the hub on the next HELLO) ------------------

#[test]
fn record_peek_clear_update_failure_roundtrip() {
    // The `.update-lastfail` marker persists the last self-update failure so the node can report
    // it to the hub on its NEXT HELLO (the boot-guard rollback runs before any hub connection).
    // record → peek returns the details; clear removes them. Reverting `record_update_failure`'s
    // write leaves peek None and this fails.
    let tmp = tempfile::tempdir().unwrap();
    let bin = install_layout(tmp.path(), "2.0.0");
    assert!(
        peek_update_failure(&bin).is_none(),
        "no marker before any failure"
    );
    record_update_failure(&bin, "1.0.0", "2.0.0", "HG084 artifact sha256 mismatch");
    let got = peek_update_failure(&bin).expect("a marker was recorded");
    assert_eq!(got.from, "1.0.0");
    assert_eq!(got.to, "2.0.0");
    assert_eq!(got.reason, "HG084 artifact sha256 mismatch");
    clear_update_failure(&bin);
    assert!(
        peek_update_failure(&bin).is_none(),
        "clear removes the marker (report-once)"
    );
}

#[test]
fn record_update_failure_sanitizes_peer_controlled_fields() {
    // `from`/`to`/`reason` cross the wire to the hub and reach a terminal + the fleet view. A
    // spoofed version (spaces/escapes) or a reason with control bytes must be scrubbed AT RECORD
    // TIME. Reverting the `sanitize_*` calls lets the raw bytes survive and this fails.
    let tmp = tempfile::tempdir().unwrap();
    let bin = install_layout(tmp.path(), "2.0.0");
    record_update_failure(
        &bin,
        "1.0.0\n bad", // newline + space → dropped (semver chars only)
        "2.0.0",
        "HG084\nsha256\x07 mismatch", // newline + BEL control → dropped
    );
    let got = peek_update_failure(&bin).unwrap();
    assert_eq!(got.from, "1.0.0bad", "version keeps only semver-safe chars");
    assert!(
        !got.reason.chars().any(char::is_control),
        "reason drops control bytes: {:?}",
        got.reason
    );
    assert!(
        got.reason.starts_with("HG084"),
        "reason keeps the visible text: {:?}",
        got.reason
    );
}

#[test]
fn reportable_update_failure_reports_a_live_marker_without_clearing_it() {
    // The node re-reports the marker on EVERY HELLO (report-until-resolved) — it must NOT clear on
    // a report, because a valid HELLO reply does not prove the hub STORED the failure. A marker
    // whose `from` still equals the RUNNING build (`CARGO_PKG_VERSION`) is a live failure: report
    // it, repeatedly.
    let tmp = tempfile::tempdir().unwrap();
    let bin = install_layout(tmp.path(), "2.0.0");
    record_update_failure(
        &bin,
        env!("CARGO_PKG_VERSION"),
        "9.9.9",
        "HG084 sha mismatch",
    );
    assert_eq!(
        reportable_update_failure(&bin).0.map(|f| f.to),
        Some("9.9.9".to_string()),
        "a marker matching the running build is reported"
    );
    // A SECOND report still returns it — reporting is non-destructive (no clear-on-reply).
    assert!(
        reportable_update_failure(&bin).0.is_some(),
        "reporting does not clear the marker"
    );
}

#[test]
fn reportable_update_failure_filters_a_stale_marker_without_deleting_it() {
    // A marker whose `from` is not the RUNNING build (`CARGO_PKG_VERSION`) is STALE — the process
    // restarted onto a different build (a later success, or a manual reinstall). reportable FILTERS
    // it (returns None, never reported) but does NOT delete it: a read-then-delete would be a
    // non-atomic read-modify-write racing a detached apply's fresh record. The inert marker lingers
    // (self-corrects: a later apply overwrites/clears it). Reverting the `from == CARGO_PKG_VERSION`
    // filter reports the stale marker (first assert); ADDING a delete here (the racy behavior) makes
    // the second assert fail.
    let tmp = tempfile::tempdir().unwrap();
    let bin = install_layout(tmp.path(), "2.0.0");
    record_update_failure(
        &bin,
        "0.0.0-not-the-running-build",
        "1.5.0",
        "an old failure",
    );
    assert_eq!(
        reportable_update_failure(&bin).0,
        None,
        "a marker for a build we are not running is not reported"
    );
    assert!(
        peek_update_failure(&bin).is_some(),
        "reportable is a pure read — it filters the stale marker but does not delete it"
    );
}

#[test]
fn reportable_update_failure_reports_inconclusive_on_a_read_error() {
    // The `conclusive` flag (the 2nd tuple field) tells the caller whether a `None` is an
    // AUTHORITATIVE absence. A confidently-absent marker is conclusive; an UNREADABLE one (here a
    // DIRECTORY at the marker path makes the file read fail) is INCONCLUSIVE — the node cannot
    // claim "no failure", so the caller omits the `update_reporting` capability rather than let the
    // hub erase a valid stored failure. Reverting the `Err(_) => (None, false)` arm reports it as
    // conclusive.
    let tmp = tempfile::tempdir().unwrap();
    let bin = install_layout(tmp.path(), "2.0.0");
    // No marker → conclusively absent.
    assert_eq!(reportable_update_failure(&bin), (None, true));
    // A DIRECTORY at the marker path → the read errors (not NotFound) → inconclusive.
    std::fs::create_dir(bin.join(".update-lastfail")).unwrap();
    assert_eq!(
        reportable_update_failure(&bin),
        (None, false),
        "an unreadable marker is inconclusive, not an authoritative absence"
    );
}

#[test]
fn reportable_compares_the_sanitized_running_version() {
    // The marker's `from` is stored THROUGH sanitize_version (64-char cap), so the report
    // comparison must sanitize the running version the SAME way. With a valid semver longer than
    // the cap (SemVer has no size limit), a raw comparison classifies every fresh marker stale →
    // authoritative None → the hub's report is silently cleared. Driven through the injected-
    // version seam (the compiled CARGO_PKG_VERSION can't take such a value). Reverting the
    // `sanitize_version(running_version)` in `reportable_update_failure_for` fails this.
    let long_version = format!("1.0.0-{}", "a".repeat(70)); // > the 64-char sanitizer cap
    let tmp = tempfile::tempdir().unwrap();
    let bin = install_layout(tmp.path(), "2.0.0");
    record_update_failure(&bin, &long_version, "9.9.9", "HG084 sha mismatch");
    let (report, conclusive) = reportable_update_failure_for(&bin, &long_version);
    assert!(
        report.is_some(),
        "a marker recorded for the running build stays reportable even when the version \
         exceeds the sanitizer cap"
    );
    assert!(conclusive);
    // And the crate's own version must survive sanitization VERBATIM — the guard that keeps the
    // production comparison exact (a future version with chars/length the sanitizer alters would
    // break report matching; catch it at test time).
    assert_eq!(
        crate::remote::sanitize_version(env!("CARGO_PKG_VERSION")),
        env!("CARGO_PKG_VERSION"),
        "the crate version must survive sanitize_version unchanged"
    );
}

#[test]
fn trial_rollback_records_before_the_fallible_marker_cleanup() {
    // The failure must be recorded IMMEDIATELY after the flip, BEFORE removing `.update-trial` — so
    // a transient unlink failure after a successful rollback still leaves a reason to report. Force
    // the removal to fail (a DIRECTORY at the trial-marker path makes `remove_file` fail) and assert
    // the record landed even though `perform_trial_rollback` then returns Err. Reverting the reorder
    // (record AFTER the removal) skips the record here.
    let running = env!("CARGO_PKG_VERSION");
    let tmp = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(tmp.path().join(format!("bin/v{running}"))).unwrap();
    write_runnable(&tmp.path().join(format!("bin/v{running}/higgs")), b"prev");
    let bin = install_layout(tmp.path(), "9.9.9-trial"); // current → the trial being rolled back
                                                         // A DIRECTORY at the trial-marker path makes the post-flip `remove_if_present` fail.
    std::fs::create_dir(bin.join(".update-trial")).unwrap();
    let outcome = perform_trial_rollback(
        &bin,
        &format!("v{running}"),
        Some("v9.9.9-trial"),
        "crash-looped on boot — rolled back",
    );
    assert!(
        outcome.is_err(),
        "the trial-marker removal failed (a directory sits at the path)"
    );
    // ...but the failure was recorded BEFORE that fallible cleanup.
    let recorded = peek_update_failure(&bin).expect("recorded despite the cleanup failure");
    assert_eq!(recorded.from, running, "from = the rolled-back-to build");
    assert_eq!(recorded.to, "9.9.9-trial", "to = the failed target");
}

#[test]
fn boot_rollback_records_a_reportable_failure_for_the_next_hello() {
    // A crash-looping trial rolled back at boot must persist WHY *and* be REPORTABLE on the next
    // HELLO. The rollback records `from` = the version it rolled back TO — which must equal the
    // now-running build's `CARGO_PKG_VERSION` in PLAIN semver form, or `reportable_update_failure`
    // reconciles it away (the node would send `update_failed: None` after every crash-loop). We
    // therefore roll back TO the running build. Reverting EITHER the `strip_v` in
    // `perform_trial_rollback` OR the `record_update_failure` call makes the reportable assert fail.
    let running = env!("CARGO_PKG_VERSION");
    let tmp = tempfile::tempdir().unwrap();
    // The rollback TARGET dir is `v<running>`, so post-rollback `current` matches the running build.
    std::fs::create_dir_all(tmp.path().join(format!("bin/v{running}"))).unwrap();
    write_runnable(&tmp.path().join(format!("bin/v{running}/higgs")), b"old");
    let bin = install_layout(tmp.path(), "9.9.9-trial"); // current → the crash-looping trial
    write_trial(&bin, "v9.9.9-trial", Some(&format!("v{running}")));
    std::fs::write(bin.join(".update-bootfails"), BOOT_FAIL_BUDGET.to_string()).unwrap();
    assert_eq!(
        boot_rollback_if_spent(&bin).unwrap(),
        Some(format!("v{running}")),
        "rolls back to the previous (running-build) version"
    );
    // The raw marker carries the PLAIN-semver from/to (no `v` prefix).
    let raw = peek_update_failure(&bin).expect("the rollback recorded a failure");
    assert_eq!(
        raw.from, running,
        "from = the rolled-back-to build, plain semver"
    );
    assert_eq!(
        raw.to, "9.9.9-trial",
        "to = the failed target, plain semver"
    );
    assert!(
        raw.reason.contains("rolled back"),
        "reason names the rollback: {:?}",
        raw.reason
    );
    // The REPORT path returns it — the version reconcile does not drop it, because `from` matches
    // the running build.
    let reported = reportable_update_failure(&bin)
        .0
        .expect("the rollback failure is reportable, not silently reconciled away");
    assert_eq!(reported.from, running);
    assert_eq!(reported.to, "9.9.9-trial");
}

#[test]
fn boot_rollback_is_a_noop_while_the_update_lock_is_held() {
    // An apply holds the UpdateLock across stage->flip. A concurrent boot must NOT
    // read-modify-write the markers (it could clobber the apply's fresh flip), so the
    // boot-guard hooks are no-ops while the lock is held.
    let tmp = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(tmp.path().join("bin/v1.0.0")).unwrap();
    write_runnable(&tmp.path().join("bin/v1.0.0/higgs"), b"old");
    let bin = install_layout(tmp.path(), "2.0.0");
    write_trial(&bin, "v2.0.0", Some("v1.0.0"));
    std::fs::write(bin.join(".update-bootfails"), BOOT_FAIL_BUDGET.to_string()).unwrap();
    // Hold the lock (as an apply would across stage->flip); a concurrent boot must NOT
    // roll back despite the spent budget — it is a no-op while the lock is held. (The
    // roll-back-when-free path is covered by `boot_rollback_rolls_back_at_budget`; we
    // do not drop-then-reacquire here — that is timing-flaky under parallel tests.)
    let _held = UpdateLock::acquire(&bin).unwrap();
    assert_eq!(boot_rollback_if_spent(&bin).unwrap(), None);
    assert_eq!(
        current_target(&bin),
        Some("v2.0.0".to_string()),
        "not rolled back"
    );
}

#[test]
fn boot_rollback_refuses_to_dangle_current_when_prev_is_gone() {
    // A spent trial whose rollback target was pruned must NOT flip `current` to a
    // dangling link (nothing would run, not even the guard). Leave current + trial be.
    let tmp = tempfile::tempdir().unwrap();
    let bin = install_layout(tmp.path(), "2.0.0"); // current -> v2.0.0; NO v1.0.0 dir
    write_trial(&bin, "v2.0.0", Some("v1.0.0"));
    std::fs::write(bin.join(".update-bootfails"), BOOT_FAIL_BUDGET.to_string()).unwrap();
    assert_eq!(
        boot_rollback_if_spent(&bin).unwrap(),
        None,
        "no dangling flip"
    );
    assert_eq!(
        current_target(&bin),
        Some("v2.0.0".to_string()),
        "current untouched when prev is gone"
    );
}

#[test]
fn boot_rollback_still_recovers_when_the_lock_cannot_be_taken() {
    // A SYMLINK at the lock path is refused by `try_acquire`'s O_NOFOLLOW open — a HARD
    // (non-contention) lock failure, distinct from another updater holding it. The boot
    // guard must STILL roll a spent trial back: the apply/rollback/prune path fails closed
    // on the identical error, so nothing else is mutating the tree, and silently skipping
    // would strand the bad binary in an endless restart. (Reverting `with_boot_lock`'s
    // Failed→run-lock-free arm to the old `let Ok(_lock)=acquire else return` makes this
    // return None and leaves `current` on the bad v2 — the mutant this test kills.)
    let tmp = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(tmp.path().join("bin/v1.0.0")).unwrap();
    write_runnable(&tmp.path().join("bin/v1.0.0/higgs"), b"old");
    let bin = install_layout(tmp.path(), "2.0.0"); // current -> v2.0.0
    write_trial(&bin, "v2.0.0", Some("v1.0.0"));
    std::fs::write(bin.join(".update-bootfails"), BOOT_FAIL_BUDGET.to_string()).unwrap();
    // O_NOFOLLOW refuses to open a symlinked lock -> LockAttempt::Failed.
    std::fs::create_dir_all(lock_dir(&bin)).unwrap();
    std::os::unix::fs::symlink("../v1.0.0/higgs", lock_path(&bin)).unwrap();
    assert_eq!(
        boot_rollback_if_spent(&bin).unwrap(),
        Some("v1.0.0".to_string()),
        "rolls back despite the hard lock failure"
    );
    assert_eq!(
        current_target(&bin),
        Some("v1.0.0".to_string()),
        "current restored to the good binary"
    );
    assert!(!bin.join(".update-trial").exists(), "trial cleared");
}

#[test]
fn boot_rollback_leaves_a_stale_marker_intact_under_a_hard_lock_failure() {
    // A `to != current` marker LOOKS stale, but under a HARD lock failure we cannot rule
    // out that it is a concurrent apply's FRESH pre-flip marker (stage_and_flip writes the
    // marker, THEN flips `current`). So lock-free we must NOT delete it — that would strip
    // a just-installed binary's rollback state. We LEAVE it (it drives no rollback) for a
    // later HELD boot to clean up. Reverting the `if locked` gate around the stale-clear
    // deletes the marker here and fails this test — the mutant it kills.
    let tmp = tempfile::tempdir().unwrap();
    let bin = install_layout(tmp.path(), "1.0.0"); // current -> v1.0.0
    write_trial(&bin, "v2.0.0", Some("v1.0.0")); // to=v2.0.0 != current=v1.0.0 -> "stale"
    std::fs::write(bin.join(".update-bootfails"), "1").unwrap();
    // A symlinked lock -> LockAttempt::Failed -> the body runs lock-free (locked=false).
    std::fs::create_dir_all(lock_dir(&bin)).unwrap();
    std::os::unix::fs::symlink("../v1.0.0/higgs", lock_path(&bin)).unwrap();
    assert_eq!(
        boot_rollback_if_spent(&bin).unwrap(),
        None,
        "a stale marker drives no rollback"
    );
    assert!(
        bin.join(".update-trial").exists(),
        "the stale-looking marker is LEFT intact lock-free (not clobbered)"
    );
    assert!(
        bin.join(".update-bootfails").exists(),
        "the counter is left intact lock-free too"
    );
    // Under a genuinely HELD lock the same stale marker IS cleaned up (control).
    std::fs::remove_file(lock_path(&bin)).unwrap();
    assert_eq!(boot_rollback_if_spent(&bin).unwrap(), None);
    assert!(
        !bin.join(".update-trial").exists(),
        "a HELD boot does clear the stale marker"
    );
}

#[test]
fn record_boot_attempt_still_counts_when_the_lock_cannot_be_taken() {
    // Same hard-lock-failure path for the counter write: a symlinked lock must not make
    // the boot-fail counter silently stop incrementing (which would keep the rollback
    // budget from ever being spent). Reverting the Failed→lock-free arm leaves no counter
    // file and this unwrap panics.
    let tmp = tempfile::tempdir().unwrap();
    let bin = install_layout(tmp.path(), "2.0.0");
    write_trial(&bin, "v2.0.0", Some("v1.0.0"));
    std::fs::create_dir_all(lock_dir(&bin)).unwrap();
    std::os::unix::fs::symlink("../v2.0.0/higgs", lock_path(&bin)).unwrap();
    record_boot_attempt(&bin, "2.0.0").unwrap();
    assert_eq!(
        std::fs::read_to_string(bin.join(".update-bootfails")).unwrap(),
        "1",
        "counter incremented lock-free on a hard lock failure"
    );
}

#[test]
fn try_acquire_reports_contention_distinctly_from_a_hard_failure() {
    // A genuinely-held lock is CONTENDED (an apply is mid-flight); a non-regular lock
    // node is a hard FAILURE. The boot hooks rely on this split to skip the former but
    // recover through the latter.
    let tmp = tempfile::tempdir().unwrap();
    let bin = install_layout(tmp.path(), "1.0.0");
    let _held = UpdateLock::acquire(&bin).unwrap();
    assert!(
        matches!(UpdateLock::try_acquire(&bin), LockAttempt::Contended),
        "a held flock is contention"
    );
    // A DIRECTORY at the lock path (after dropping the real lock) is a hard failure.
    drop(_held);
    std::fs::remove_file(lock_path(&bin)).unwrap();
    std::fs::create_dir(lock_path(&bin)).unwrap();
    assert!(
        matches!(UpdateLock::try_acquire(&bin), LockAttempt::Failed(_)),
        "a non-regular lock node is a hard failure, not contention"
    );
}

#[test]
fn confirm_alive_clears_only_for_the_trialed_version() {
    let tmp = tempfile::tempdir().unwrap();
    let bin = install_layout(tmp.path(), "2.0.0");
    write_trial(&bin, "v2.0.0", Some("v1.0.0"));
    std::fs::write(bin.join(".update-bootfails"), "1").unwrap();
    // An OLD v1.0.0 daemon admitted while v2.0.0 is on trial must NOT clear v2's trial.
    confirm_alive(&bin, "1.0.0").unwrap();
    assert!(
        bin.join(".update-trial").exists(),
        "trial survives an old daemon's admission"
    );
    // The v2.0.0 binary admitting DOES commit the update.
    confirm_alive(&bin, "2.0.0").unwrap();
    assert!(!bin.join(".update-trial").exists());
    assert!(!bin.join(".update-bootfails").exists());
}

// ---- rollback ------------------------------------------------------------

#[test]
fn rollback_repoints_current_to_prev() {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(tmp.path().join("bin/v1.0.0")).unwrap();
    write_runnable(&tmp.path().join("bin/v1.0.0/higgs"), b"old");
    let bin = install_layout(tmp.path(), "2.0.0");
    let marker = TrialMarker {
        to: "v2.0.0".into(),
        prev: Some("v1.0.0".into()),
    };
    std::fs::write(
        bin.join(".update-trial"),
        serde_json::to_vec(&marker).unwrap(),
    )
    .unwrap();
    assert_eq!(rollback(&bin).unwrap(), "v1.0.0");
    assert_eq!(current_target(&bin), Some("v1.0.0".to_string()));
    assert!(!bin.join(".update-trial").exists());
    // The manually-rolled-back version is POISONED too (like the boot-guard auto-rollback), so a
    // hub re-push of it is refused. Dropping the poison in `rollback` leaves it un-refused.
    assert!(
        is_failed_target(&bin, "v2.0.0"),
        "a manual --rollback poisons the version it rolled back from"
    );
}

#[test]
fn rollback_refuses_a_non_runnable_or_escaping_target() {
    let tmp = tempfile::tempdir().unwrap();
    let bin = install_layout(tmp.path(), "2.0.0");
    // (a) prev dir exists but its higgs is 0644 (NOT executable) → not a runnable
    // rollback target; repointing current at it would leave the service unable to exec.
    std::fs::create_dir_all(bin.join("v1.0.0")).unwrap();
    std::fs::write(bin.join("v1.0.0/higgs"), b"x").unwrap(); // default 0644
    write_trial(&bin, "v2.0.0", Some("v1.0.0"));
    assert!(
        matches!(
            rollback_target(&bin),
            Err(HiggsError::UpdateApplyFailed { .. })
        ),
        "a non-executable rollback target must be refused"
    );
    // (b) an ESCAPING prev (`../outside`) — even with a real runnable binary outside
    // `bin` — is refused because it is not a clean `v<semver>` name (no `..`).
    std::fs::create_dir_all(tmp.path().join("outside")).unwrap();
    write_runnable(&tmp.path().join("outside/higgs"), b"x");
    write_trial(&bin, "v2.0.0", Some("../outside"));
    assert!(
        matches!(
            rollback_target(&bin),
            Err(HiggsError::UpdateApplyFailed { .. })
        ),
        "an escaping rollback target must be refused"
    );
}

#[test]
fn rollback_refuses_a_stale_trial_superseded_by_a_manual_install() {
    // v2's trial marker lingers after a manual install of v3 (current -> v3). A manual
    // `--rollback` must NOT undo that repair by flipping v3 back to v1.
    let tmp = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(tmp.path().join("bin/v1.0.0")).unwrap();
    write_runnable(&tmp.path().join("bin/v1.0.0/higgs"), b"old");
    let bin = install_layout(tmp.path(), "3.0.0"); // current -> v3.0.0
    write_trial(&bin, "v2.0.0", Some("v1.0.0")); // STALE: to=v2 != current=v3
    assert!(matches!(
        rollback_target(&bin),
        Err(HiggsError::UpdateApplyFailed { .. })
    ));
}

#[test]
fn force_rollback_reverts_our_live_trial_regardless_of_budget() {
    // The disk-full recovery path: force a rollback of the pending trial for our version
    // even with the counter at 0 (unwritable). Restores the good binary.
    let tmp = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(tmp.path().join("bin/v1.0.0")).unwrap();
    write_runnable(&tmp.path().join("bin/v1.0.0/higgs"), b"old");
    let bin = install_layout(tmp.path(), "2.0.0"); // current -> v2.0.0
    write_trial(&bin, "v2.0.0", Some("v1.0.0")); // live trial for us
    assert_eq!(
        force_rollback_trial(&bin, "2.0.0").unwrap(),
        Some("v1.0.0".to_string())
    );
    assert_eq!(current_target(&bin), Some("v1.0.0".to_string()));
    assert!(!bin.join(".update-trial").exists());
    // The recorded reason is the ACCURATE cause — this trial did NOT crash-loop; its boot health
    // could not be tracked. Reverting the per-caller `reason` (hard-coding "crash-looped") would
    // send the operator chasing a nonexistent crash; this pins the forced path's distinct text.
    let recorded = peek_update_failure(&bin).expect("the forced rollback recorded a failure");
    assert!(
        recorded.reason.contains("could not track") && !recorded.reason.contains("crash-looped"),
        "the forced path reports the tracking failure, not a crash-loop: {:?}",
        recorded.reason
    );
    assert_eq!(
        recorded.to, "2.0.0",
        "to = the rolled-back trial (plain semver)"
    );
}

#[test]
fn force_rollback_ignores_a_stale_or_foreign_trial() {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(tmp.path().join("bin/v1.0.0")).unwrap();
    write_runnable(&tmp.path().join("bin/v1.0.0/higgs"), b"old");
    let bin = install_layout(tmp.path(), "2.0.0"); // current -> v2.0.0
    write_trial(&bin, "v2.0.0", Some("v1.0.0"));
    // A DIFFERENT running version (an old daemon) must not force our trial back.
    assert_eq!(force_rollback_trial(&bin, "9.9.9").unwrap(), None);
    assert_eq!(current_target(&bin), Some("v2.0.0".to_string()));
}

#[test]
fn rollback_refuses_without_a_recorded_previous() {
    let tmp = tempfile::tempdir().unwrap();
    let bin = install_layout(tmp.path(), "2.0.0");
    assert!(matches!(
        rollback(&bin),
        Err(HiggsError::UpdateApplyFailed { .. })
    ));
}

#[test]
fn rollback_refuses_when_the_target_dir_is_missing() {
    let tmp = tempfile::tempdir().unwrap();
    let bin = install_layout(tmp.path(), "2.0.0"); // only v2.0.0 exists
    let marker = TrialMarker {
        to: "v2.0.0".into(),
        prev: Some("v1.0.0".into()),
    };
    std::fs::write(
        bin.join(".update-trial"),
        serde_json::to_vec(&marker).unwrap(),
    )
    .unwrap();
    assert!(matches!(
        rollback(&bin),
        Err(HiggsError::UpdateApplyFailed { .. })
    ));
}

// ---- prune ---------------------------------------------------------------

#[test]
fn prune_keeps_current_and_rollback_target() {
    let tmp = tempfile::tempdir().unwrap();
    for v in ["v0.9.0", "v1.0.0", "v3.0.0"] {
        std::fs::create_dir_all(tmp.path().join("bin").join(v)).unwrap();
    }
    let bin = install_layout(tmp.path(), "2.0.0"); // current -> v2.0.0
    let marker = TrialMarker {
        to: "v2.0.0".into(),
        prev: Some("v1.0.0".into()),
    };
    std::fs::write(
        bin.join(".update-trial"),
        serde_json::to_vec(&marker).unwrap(),
    )
    .unwrap();
    let pruned = prune(&bin).unwrap();
    assert_eq!(pruned, vec!["v0.9.0", "v3.0.0"]); // v1.0.0 (prev) + v2.0.0 (current) kept
    assert!(bin.join("v1.0.0").exists());
    assert!(bin.join("v2.0.0").exists());
    assert!(!bin.join("v0.9.0").exists());
    assert!(!bin.join("v3.0.0").exists());
    // current symlink (not a v-dir) untouched.
    assert_eq!(current_target(&bin), Some("v2.0.0".to_string()));
}

#[test]
fn prune_refuses_a_dir_that_is_not_a_higgs_install() {
    // A dir with no `current` symlink is not a managed install — prune must refuse
    // (guards against a mis-derived bin path recursing into unrelated `v…` dirs).
    let tmp = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(tmp.path().join("v1.2.3")).unwrap();
    std::fs::create_dir_all(tmp.path().join("valuable")).unwrap();
    assert!(matches!(
        prune_plan(tmp.path()),
        Err(HiggsError::UpdateApplyFailed { .. })
    ));
    // Nothing was selected/removed.
    assert!(tmp.path().join("valuable").exists());
}

#[test]
fn prune_never_selects_a_non_semver_v_dir() {
    let tmp = tempfile::tempdir().unwrap();
    let bin = install_layout(tmp.path(), "1.0.0");
    std::fs::create_dir_all(bin.join("valuable")).unwrap(); // starts with v, not v<semver>
    std::fs::create_dir_all(bin.join("v0.9.0")).unwrap();
    let plan = prune_plan(&bin).unwrap();
    assert_eq!(plan, vec!["v0.9.0"]); // valuable is NOT a version dir
}

#[test]
fn prune_ignores_non_version_dirs() {
    let tmp = tempfile::tempdir().unwrap();
    let bin = install_layout(tmp.path(), "1.0.0");
    std::fs::create_dir_all(bin.join("logs")).unwrap();
    std::fs::create_dir_all(lock_dir(&bin)).unwrap();
    let pruned = prune(&bin).unwrap();
    assert!(pruned.is_empty());
    assert!(bin.join("logs").exists(), "non-v dir kept");
    assert!(lock_dir(&bin).exists(), "lock dir kept");
}

// ---- update lock ---------------------------------------------------------

#[test]
fn update_lock_refuses_a_symlinked_lock_path() {
    use std::os::unix::fs::PermissionsExt;
    let tmp = tempfile::tempdir().unwrap();
    let bin = install_layout(tmp.path(), "1.0.0");
    // Plant the lock path as a symlink to the binary — O_NOFOLLOW must refuse it, and
    // the fchmod-through-fd must NOT chmod the binary down to 0600.
    let victim = bin.join("v1.0.0/higgs");
    std::fs::create_dir_all(lock_dir(&bin)).unwrap();
    std::os::unix::fs::symlink(&victim, lock_path(&bin)).unwrap();
    let before = std::fs::metadata(&victim).unwrap().permissions().mode() & 0o777;
    assert!(matches!(
        UpdateLock::acquire(&bin),
        Err(HiggsError::UpdateApplyFailed { .. })
    ));
    let after = std::fs::metadata(&victim).unwrap().permissions().mode() & 0o777;
    assert_eq!(before, after, "the symlink target must not be chmod'd");
}

#[test]
fn rollback_target_requires_executability_by_this_user() {
    use std::os::unix::fs::PermissionsExt;
    // access(X_OK) semantics differ for root (bypasses perms); skip when euid==0.
    if unsafe { libc::geteuid() } == 0 {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let bin = install_layout(tmp.path(), "2.0.0");
    std::fs::create_dir_all(bin.join("v1.0.0")).unwrap();
    let higgs = bin.join("v1.0.0/higgs");
    std::fs::write(&higgs, b"x").unwrap();
    // 0o055: some exec bit is set (group/other), but the OWNER (us) cannot exec it —
    // a bare `mode & 0o111` check would wrongly accept it; access(X_OK) refuses.
    std::fs::set_permissions(&higgs, std::fs::Permissions::from_mode(0o055)).unwrap();
    write_trial(&bin, "v2.0.0", Some("v1.0.0"));
    assert!(matches!(
        rollback_target(&bin),
        Err(HiggsError::UpdateApplyFailed { .. })
    ));
}

#[test]
fn boot_rollback_ignores_a_stale_trial_superseded_by_a_new_install() {
    // A manual `install.sh` of v3 leaves v2's old trial marker behind. The boot-guard
    // must NOT roll v3 back to v1 on that stale marker (`to=v2` != `current=v3`).
    let tmp = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(tmp.path().join("bin/v1.0.0")).unwrap();
    write_runnable(&tmp.path().join("bin/v1.0.0/higgs"), b"old");
    let bin = install_layout(tmp.path(), "3.0.0"); // current -> v3.0.0 (manual v3 install)
    write_trial(&bin, "v2.0.0", Some("v1.0.0")); // STALE: to=v2, current=v3
    std::fs::write(bin.join(".update-bootfails"), BOOT_FAIL_BUDGET.to_string()).unwrap();
    assert_eq!(
        boot_rollback_if_spent(&bin).unwrap(),
        None,
        "a stale trial must not roll back a manual repair install"
    );
    assert_eq!(
        current_target(&bin),
        Some("v3.0.0".to_string()),
        "v3 preserved"
    );
    assert!(!bin.join(".update-trial").exists(), "stale trial cleared");
}

#[test]
fn prune_keeps_a_case_differing_current_dir() {
    // `current -> v1.0.0-rc1` but the on-disk dir is `v1.0.0-RC1` (case-preserved). The
    // case-insensitive keep set must recognize it as live and NOT prune it.
    let tmp = tempfile::tempdir().unwrap();
    let bin = tmp.path().join("bin");
    std::fs::create_dir_all(bin.join("v1.0.0-RC1")).unwrap();
    write_runnable(&bin.join("v1.0.0-RC1/higgs"), b"x");
    std::fs::create_dir_all(bin.join("v0.9.0")).unwrap();
    std::os::unix::fs::symlink("v1.0.0-rc1", bin.join("current")).unwrap();
    assert_eq!(
        prune_plan(&bin).unwrap(),
        vec!["v0.9.0"],
        "the case-differing live dir must be kept"
    );
}

#[test]
fn update_lock_is_exclusive() {
    let tmp = tempfile::tempdir().unwrap();
    let bin = install_layout(tmp.path(), "1.0.0");
    let _held = UpdateLock::acquire(&bin).unwrap();
    // A second acquire while the first is held fails (non-blocking). (We do NOT test
    // drop-then-reacquire: the flock release-on-close is an OS/RAII guarantee, and an
    // immediate re-acquire is timing-flaky under the parallel test suite.)
    assert!(matches!(
        UpdateLock::acquire(&bin),
        Err(HiggsError::UpdateApplyFailed { .. })
    ));
}

// ---- stage_and_flip (filesystem + injected smoke) ------------------------

fn ok_smoke(reported: &'static str) -> Box<SmokeRunner<'static>> {
    Box::new(move |_p: &std::path::Path| Ok(reported.to_string()))
}

/// A manifest whose target + variant MATCH the running build, so `stage_and_flip`'s
/// under-lock eligibility recheck (which compares against this machine's target and
/// the installed variant) passes on any CI host — leaving the STEP under test (smoke,
/// mode, flip) as the thing that decides the outcome.
fn local_manifest(version: &str) -> UpdateManifest {
    manifest(version, env!("HIGGS_BUILD_TARGET"), CURRENT_VARIANT)
}

#[test]
fn stage_and_flip_publishes_smokes_and_flips() {
    let tmp = tempfile::tempdir().unwrap();
    let bin = install_layout(tmp.path(), "1.0.0"); // current -> v1.0.0
    let art = tarball_with_higgs(b"new-binary", 0o755);
    let m = local_manifest("2.0.0");
    let smoke = ok_smoke("higgs 2.0.0");
    stage_and_flip(&bin, &m, &art, false, smoke.as_ref()).unwrap();

    // Published v2.0.0/higgs (regular exec) + .variant marker (this build's variant).
    assert!(bin.join("v2.0.0/higgs").exists());
    assert_eq!(
        std::fs::read_to_string(bin.join("v2.0.0/.variant")).unwrap(),
        CURRENT_VARIANT
    );
    // current flipped to v2.0.0.
    assert_eq!(current_target(&bin), Some("v2.0.0".to_string()));
    // Trial marker records the outgoing current as the rollback target.
    let raw = std::fs::read(bin.join(".update-trial")).unwrap();
    let marker: TrialMarker = serde_json::from_slice(&raw).unwrap();
    assert_eq!(marker.to, "v2.0.0");
    assert_eq!(marker.prev, Some("v1.0.0".to_string()));
    // No staging litter left behind.
    let leftover: Vec<_> = std::fs::read_dir(&bin)
        .unwrap()
        .filter_map(Result::ok)
        .filter(|e| e.file_name().to_string_lossy().starts_with(".update-stage"))
        .collect();
    assert!(leftover.is_empty(), "staging dir removed");
}

#[test]
fn stage_and_flip_clears_the_last_failure_marker_on_success() {
    // A successful stage+flip means the update TOOK, so any prior self-update failure is moot — the
    // marker is cleared HERE, the single flip point BOTH the hub-push apply AND the manual CLI
    // self-update reach, so a success never leaves a stale marker to resurrect on a later downgrade
    // (the CLI path used to skip clearing). Reverting the clear at the end of stage_and_flip leaves
    // the marker and this fails.
    let tmp = tempfile::tempdir().unwrap();
    let bin = install_layout(tmp.path(), "1.0.0");
    record_update_failure(&bin, "1.0.0", "9.9.9", "HG084 an earlier failure");
    assert!(
        peek_update_failure(&bin).is_some(),
        "seeded a prior failure"
    );
    let art = tarball_with_higgs(b"new-binary", 0o755);
    let m = local_manifest("2.0.0");
    let smoke = ok_smoke("higgs 2.0.0");
    stage_and_flip(&bin, &m, &art, false, smoke.as_ref()).unwrap();
    assert!(
        peek_update_failure(&bin).is_none(),
        "a successful stage+flip clears the last-failure marker"
    );
}

#[test]
fn stage_and_flip_refuses_a_rolled_back_version() {
    // A version that FAILED its boot trial and was rolled back is POISONED; re-applying it must
    // be refused (a hub re-pushing a crash-looping release doesn't loop). Reverting the
    // `is_failed_target` guard in `stage_and_flip` lets the poisoned version publish.
    let tmp = tempfile::tempdir().unwrap();
    let bin = install_layout(tmp.path(), "1.0.0"); // current -> v1.0.0
    record_failed_target(&bin, "v2.0.0"); // v2.0.0 crash-looped earlier
    let art = tarball_with_higgs(b"new-binary", 0o755);
    let m = local_manifest("2.0.0");
    let smoke = ok_smoke("higgs 2.0.0");
    let err = stage_and_flip(&bin, &m, &art, false, smoke.as_ref()).unwrap_err();
    assert!(
        err.to_string().contains("previously failed its boot trial"),
        "a poisoned version is refused: {err}"
    );
    // current stayed on the last-good v1.0.0 (nothing published).
    assert_eq!(current_target(&bin), Some("v1.0.0".to_string()));
    assert!(
        !bin.join("v2.0.0").exists(),
        "the poisoned version was not staged"
    );
}

#[test]
fn stage_and_flip_refuses_when_the_smoke_reports_the_wrong_version() {
    let tmp = tempfile::tempdir().unwrap();
    let bin = install_layout(tmp.path(), "1.0.0");
    let art = tarball_with_higgs(b"new-binary", 0o755);
    let m = local_manifest("2.0.0");
    // The staged binary claims a DIFFERENT version → refuse, do NOT flip.
    let smoke = ok_smoke("higgs 6.6.6");
    assert!(matches!(
        stage_and_flip(&bin, &m, &art, false, smoke.as_ref()),
        Err(HiggsError::UpdateApplyFailed { .. })
    ));
    assert_eq!(
        current_target(&bin),
        Some("v1.0.0".to_string()),
        "current NOT flipped"
    );
    assert!(
        !bin.join(".update-trial").exists(),
        "no trial written on refusal"
    );
}

#[test]
fn stage_and_flip_refuses_a_tarball_without_a_regular_higgs() {
    let tmp = tempfile::tempdir().unwrap();
    let bin = install_layout(tmp.path(), "1.0.0");
    // A tarball whose `higgs` is NOT executable (mode 0644).
    let art = tarball_with_higgs(b"data", 0o644);
    let m = local_manifest("2.0.0");
    let smoke = ok_smoke("higgs 2.0.0");
    assert!(matches!(
        stage_and_flip(&bin, &m, &art, false, smoke.as_ref()),
        Err(HiggsError::UpdateApplyFailed { .. })
    ));
    assert_eq!(current_target(&bin), Some("v1.0.0".to_string()));
}

#[test]
fn stage_and_flip_refuses_a_sticky_world_writable_bin() {
    // A `01777` (sticky, world-writable) bin lets a peer pre-create/​swap the staged
    // `v<ver>/higgs` before the restart. The old sticky-exempt walk PASSED it; the
    // hardened managed-leaf walk must REFUSE it (bin is a managed dir, not a pure
    // ancestor, so sticky grants no exemption).
    use std::os::unix::fs::PermissionsExt;
    let tmp = tempfile::tempdir().unwrap();
    let bin = install_layout(tmp.path(), "1.0.0");
    std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o1777)).unwrap();
    let art = tarball_with_higgs(b"new", 0o755);
    let m = local_manifest("2.0.0");
    let smoke = ok_smoke("higgs 2.0.0");
    let got = stage_and_flip(&bin, &m, &art, false, smoke.as_ref());
    // Restore perms so TempDir cleanup can recurse.
    let _ = std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o755));
    assert!(
        matches!(got, Err(HiggsError::UpdateApplyFailed { .. })),
        "a sticky world-writable bin must be refused, got {got:?}"
    );
    assert_eq!(
        current_target(&bin),
        Some("v1.0.0".to_string()),
        "not flipped"
    );
}

#[test]
fn stage_and_flip_smokes_before_publishing_so_a_failed_apply_keeps_prev() {
    // A downgrade whose smoke FAILS must NOT overwrite the existing (rollback-target)
    // version dir before smoke — smoke runs on the STAGING copy, so v1.0.0 is intact.
    let tmp = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(tmp.path().join("bin/v1.0.0")).unwrap();
    write_runnable(&tmp.path().join("bin/v1.0.0/higgs"), b"old");
    let bin = install_layout(tmp.path(), "2.0.0"); // current -> v2.0.0
    std::fs::write(bin.join("v1.0.0/higgs"), b"OLD-GOOD").unwrap();
    let art = tarball_with_higgs(b"NEW-BAD", 0o755);
    let m = local_manifest("1.0.0"); // a downgrade target
    let smoke = ok_smoke("higgs 6.6.6"); // reports the WRONG version → smoke fails
                                         // allow_downgrade=true so eligibility passes; the SMOKE failure is what refuses.
    assert!(matches!(
        stage_and_flip(&bin, &m, &art, true, smoke.as_ref()),
        Err(HiggsError::UpdateApplyFailed { .. })
    ));
    // The known-good rollback copy is untouched, and current did not move.
    assert_eq!(
        std::fs::read(bin.join("v1.0.0/higgs")).unwrap(),
        b"OLD-GOOD"
    );
    assert_eq!(current_target(&bin), Some("v2.0.0".to_string()));
}

#[test]
fn stage_and_flip_normalizes_a_world_writable_binary() {
    // A signed tar with `higgs` mode 0777 must publish as 0755 (no group/other write,
    // no setuid) — the archive's mode bits are not a security policy.
    use std::os::unix::fs::PermissionsExt;
    let tmp = tempfile::tempdir().unwrap();
    let bin = install_layout(tmp.path(), "1.0.0");
    let art = tarball_with_higgs(b"bin", 0o4777); // setuid + world-write + exec
    let m = local_manifest("2.0.0");
    let smoke = ok_smoke("higgs 2.0.0");
    stage_and_flip(&bin, &m, &art, false, smoke.as_ref()).unwrap();
    let mode = std::fs::metadata(bin.join("v2.0.0/higgs"))
        .unwrap()
        .permissions()
        .mode()
        & 0o7777;
    assert_eq!(mode, 0o755, "published binary must be exactly rwxr-xr-x");
}

#[test]
fn smoke_run_does_not_hang_on_a_detached_descendant() {
    // Stronger than the same-group case: a `--version` handler that DETACHES a child
    // into its OWN session (via perl POSIX::setsid, portable to macOS+Linux) which
    // holds the stdout pipe cannot be reaped by the process-group kill. smoke_run must
    // still RETURN (bounded), not block forever on the pipe read.
    use std::sync::mpsc;
    let tmp = tempfile::tempdir().unwrap();
    let script = tmp.path().join("higgs");
    write_script(
        &script,
        "#!/bin/sh\nperl -e 'use POSIX; POSIX::setsid(); exec \"sleep\",\"60\"' &\n\
         echo \"higgs 9.9.9\"\nexit 0\n",
    );
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(smoke_run(&script));
    });
    // Must return within the hard bound (SMOKE_TIMEOUT + a couple seconds), NOT hang
    // for the detached child's 60s lifetime.
    assert!(
        rx.recv_timeout(std::time::Duration::from_secs(20)).is_ok(),
        "smoke_run hung on a detached descendant holding stdout"
    );
}

#[test]
fn stage_and_flip_refuses_stacking_on_an_unconfirmed_trial() {
    // A pending (unconfirmed) trial must block a NEW apply — else the second apply would
    // record the unconfirmed version as `prev` and discard the last known-good one.
    let tmp = tempfile::tempdir().unwrap();
    let bin = install_layout(tmp.path(), "2.0.0");
    write_trial(&bin, "v2.0.0", Some("v1.0.0")); // v2 still on trial
    let art = tarball_with_higgs(b"new", 0o755);
    let m = local_manifest("3.0.0");
    let smoke = ok_smoke("higgs 3.0.0");
    assert!(matches!(
        stage_and_flip(&bin, &m, &art, false, smoke.as_ref()),
        Err(HiggsError::UpdateApplyFailed { .. })
    ));
    assert_eq!(
        current_target(&bin),
        Some("v2.0.0".to_string()),
        "not flipped"
    );
}

#[test]
fn stage_and_flip_clears_a_stale_trial_and_proceeds() {
    // A trial whose `to` no longer matches `current` is a crashed-mid-apply orphan (the
    // marker was written but the flip never happened). It must NOT wedge future updates:
    // clear it and proceed. Here current=v1.0.0 but the trial names v2.0.0 (never flipped).
    let tmp = tempfile::tempdir().unwrap();
    let bin = install_layout(tmp.path(), "1.0.0"); // current -> v1.0.0
    write_trial(&bin, "v2.0.0", Some("v0.9.0")); // stale: to=v2 but current=v1
    let art = tarball_with_higgs(b"new", 0o755);
    let m = local_manifest("3.0.0");
    let smoke = ok_smoke("higgs 3.0.0");
    stage_and_flip(&bin, &m, &art, false, smoke.as_ref()).unwrap();
    // Proceeded: current flipped to v3.0.0, trial now records v3 over v1.
    assert_eq!(current_target(&bin), Some("v3.0.0".to_string()));
    let marker: TrialMarker =
        serde_json::from_slice(&std::fs::read(bin.join(".update-trial")).unwrap()).unwrap();
    assert_eq!(marker.to, "v3.0.0");
    assert_eq!(marker.prev, Some("v1.0.0".to_string()));
}

#[test]
fn stage_and_flip_rechecks_eligibility_under_lock() {
    // The under-lock recheck compares against the version CURRENTLY installed — so a
    // manifest that was eligible against an older `current` (before a concurrent flip)
    // is refused if `current` moved forward. Here current=v3, manifest=v2, no flag.
    let tmp = tempfile::tempdir().unwrap();
    let bin = install_layout(tmp.path(), "3.0.0"); // current -> v3.0.0
    let art = tarball_with_higgs(b"new", 0o755);
    let m = local_manifest("2.0.0"); // older than the installed v3
    let smoke = ok_smoke("higgs 2.0.0");
    assert!(matches!(
        stage_and_flip(&bin, &m, &art, false, smoke.as_ref()),
        Err(HiggsError::UpdateNotNewer { .. })
    ));
    assert_eq!(current_target(&bin), Some("v3.0.0".to_string()));
}

#[test]
fn installed_identity_reads_the_variant_marker() {
    // The installed variant comes from the current version's `.variant` marker, so a
    // stale binary of a different variant cannot silently switch the install.
    let tmp = tempfile::tempdir().unwrap();
    let bin = install_layout(tmp.path(), "2.0.0");
    std::fs::write(bin.join("v2.0.0/.variant"), b"cuda\n").unwrap();
    assert_eq!(installed_identity(&bin).variant, "cuda");
}

#[test]
fn stage_and_flip_tightens_a_pre_existing_world_writable_version_dir() {
    use std::os::unix::fs::PermissionsExt;
    let tmp = tempfile::tempdir().unwrap();
    let bin = install_layout(tmp.path(), "1.0.0");
    // A pre-existing 0777 v2.0.0 dir (a peer-plantable dir); create_dir_all is a no-op
    // on it, so the apply must chmod it down to 0755.
    let vdir = bin.join("v2.0.0");
    std::fs::create_dir_all(&vdir).unwrap();
    std::fs::set_permissions(&vdir, std::fs::Permissions::from_mode(0o777)).unwrap();
    let art = tarball_with_higgs(b"new", 0o755);
    let m = local_manifest("2.0.0");
    let smoke = ok_smoke("higgs 2.0.0");
    stage_and_flip(&bin, &m, &art, false, smoke.as_ref()).unwrap();
    let mode = std::fs::metadata(&vdir).unwrap().permissions().mode() & 0o7777;
    assert_eq!(mode, 0o755, "version dir tightened to 0755");
}

#[test]
fn update_lock_file_is_owner_only() {
    use std::os::unix::fs::PermissionsExt;
    let tmp = tempfile::tempdir().unwrap();
    let bin = install_layout(tmp.path(), "1.0.0");
    let _held = UpdateLock::acquire(&bin).unwrap();
    let mode = std::fs::metadata(lock_path(&bin))
        .unwrap()
        .permissions()
        .mode()
        & 0o777;
    assert_eq!(
        mode, 0o600,
        "lock must be owner-only so a peer can't hold it"
    );
    // …and its private parent dir is 0700 so a peer can't even traverse to it.
    let dmode = std::fs::metadata(lock_dir(&bin))
        .unwrap()
        .permissions()
        .mode()
        & 0o777;
    assert_eq!(dmode, 0o700, "lock dir must be owner-only");
}

#[test]
fn unpack_creates_the_staging_dir_owner_only() {
    // The staging dir must be 0700 FROM CREATION (before extraction), so no peer can
    // traverse it and exec a setuid binary extracted mid-unpack.
    use std::os::unix::fs::PermissionsExt;
    let tmp = tempfile::tempdir().unwrap();
    let dest = tmp.path().join("stage");
    unpack_tar_gz(&tarball_with_higgs(b"x", 0o755), &dest).unwrap();
    let mode = std::fs::metadata(&dest).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode, 0o700, "staging dir must be created owner-only");
}

#[cfg(target_os = "macos")]
#[test]
fn stage_and_flip_refuses_a_pre_existing_vdir_with_a_peer_write_acl() {
    // A pre-existing `v2.0.0` at mode 0755 but carrying an ACL granting `everyone`
    // add_file/delete_child lets a peer swap the published binary — chmod 0755 does not
    // strip the ACL, so publish must refuse it and never flip.
    let tmp = tempfile::tempdir().unwrap();
    let bin = install_layout(tmp.path(), "1.0.0");
    let vdir = bin.join("v2.0.0");
    std::fs::create_dir_all(&vdir).unwrap();
    let add = std::process::Command::new("/bin/chmod")
        .arg("+a")
        .arg("everyone allow add_file,delete_child")
        .arg(&vdir)
        .status()
        .expect("chmod +a");
    assert!(add.success(), "could not add the test ACL");
    let art = tarball_with_higgs(b"new", 0o755);
    let m = local_manifest("2.0.0");
    let smoke = ok_smoke("higgs 2.0.0");
    let got = stage_and_flip(&bin, &m, &art, false, smoke.as_ref());
    let _ = std::process::Command::new("/bin/chmod")
        .arg("-N")
        .arg(&vdir)
        .status();
    assert!(
        matches!(got, Err(HiggsError::UpdateApplyFailed { .. })),
        "a peer-write ACL on the version dir must be refused, got {got:?}"
    );
    assert_eq!(
        current_target(&bin),
        Some("v1.0.0".to_string()),
        "not flipped"
    );
}

#[cfg(target_os = "macos")]
#[test]
fn update_lock_lives_in_a_private_dir_immune_to_an_inherited_read_acl() {
    // `bin` carries an inheritable `everyone allow read,...` ACE. If the lock sat directly
    // in `bin` it would inherit read access, and macOS `flock(LOCK_EX)` works on a
    // read-only fd — so a peer could open it read-only and hold the update lock forever,
    // making every boot hook skip on false contention (a bad trial never rolls back). The
    // lock must live in a 0700 dir whose inherited ACL is stripped, so no peer can even
    // traverse to it. Reverting the private-lock-dir fix puts the lock's parent back at
    // `bin` (which still carries the `everyone` ACE) and fails this test.
    let tmp = tempfile::tempdir().unwrap();
    let bin = install_layout(tmp.path(), "1.0.0");
    let add = std::process::Command::new("/bin/chmod")
        .arg("+a")
        .arg("everyone allow read,list,search,file_inherit,directory_inherit")
        .arg(&bin)
        .status()
        .expect("chmod +a");
    assert!(add.success(), "could not add the inheritable test ACL");
    let _held = UpdateLock::acquire(&bin).unwrap();
    let lock_parent = lock_path(&bin).parent().unwrap().to_path_buf();
    let acl = std::process::Command::new("/bin/ls")
        .arg("-lde")
        .arg(&lock_parent)
        .output()
        .expect("ls -lde");
    let _ = std::process::Command::new("/bin/chmod")
        .arg("-N")
        .arg(&bin)
        .status();
    let listing = String::from_utf8_lossy(&acl.stdout);
    assert!(
        !listing.contains("everyone"),
        "the lock's parent dir must carry no peer-granting ACE, ls -lde:\n{listing}"
    );
    use std::os::unix::fs::PermissionsExt;
    let mode = std::fs::metadata(&lock_parent)
        .unwrap()
        .permissions()
        .mode()
        & 0o777;
    assert_eq!(
        mode, 0o700,
        "the lock's parent dir must be owner-only (0700)"
    );
}

#[cfg(target_os = "macos")]
#[test]
fn unpack_strips_an_inherited_acl_from_the_private_staging_dir() {
    // `bin` carries an INHERITABLE `everyone allow list,search` ACE. Without stripping,
    // macOS copies it onto the new `0700` staging dir, letting any user traverse the
    // supposedly-private dir (and exec a transient setuid `higgs` mid-unpack). unpack must
    // strip it. Reverting `strip_inherited_acls` leaves the ACE and fails this test.
    let tmp = tempfile::tempdir().unwrap();
    let bin = install_layout(tmp.path(), "1.0.0");
    let add = std::process::Command::new("/bin/chmod")
        .arg("+a")
        .arg("everyone allow list,search,file_inherit,directory_inherit")
        .arg(&bin)
        .status()
        .expect("chmod +a");
    assert!(add.success(), "could not add the inheritable test ACL");
    let stage = bin.join(".update-stage.acltest");
    let res = unpack_tar_gz(&tarball_with_higgs(b"x", 0o755), &stage);
    let acl = std::process::Command::new("/bin/ls")
        .arg("-lde")
        .arg(&stage)
        .output()
        .expect("ls -lde");
    // clean up the inheritable ACL on bin so TempDir teardown isn't affected
    let _ = std::process::Command::new("/bin/chmod")
        .arg("-N")
        .arg(&bin)
        .status();
    assert!(res.is_ok(), "unpack should succeed: {res:?}");
    let listing = String::from_utf8_lossy(&acl.stdout);
    assert!(
        !listing.contains("everyone"),
        "the inherited ACE must be stripped from the staging dir, ls -lde:\n{listing}"
    );
}

#[test]
fn stage_and_flip_refuses_a_non_flat_archive() {
    // A nested `tools/helper` (setuid) entry makes the archive non-flat — publish must
    // refuse it (the top-level-only normalize would otherwise let the nested setuid file
    // survive), and never flip.
    let tmp = tempfile::tempdir().unwrap();
    let bin = install_layout(tmp.path(), "1.0.0");
    let m = local_manifest("2.0.0");
    let smoke = ok_smoke("higgs 2.0.0");
    assert!(matches!(
        stage_and_flip(&bin, &m, &tarball_with_nested(), false, smoke.as_ref()),
        Err(HiggsError::UpdateApplyFailed { .. })
    ));
    assert_eq!(
        current_target(&bin),
        Some("v1.0.0".to_string()),
        "not flipped"
    );
    assert!(!bin.join("v2.0.0").exists(), "not published");
}

#[test]
fn stage_and_flip_cleans_up_after_a_failed_unpack() {
    // A mid/failed unpack (here: a garbage, non-gzip artifact) must leave NO
    // `.update-stage.*` litter — the cleanup guard is armed BEFORE the unpack.
    let tmp = tempfile::tempdir().unwrap();
    let bin = install_layout(tmp.path(), "1.0.0");
    let m = local_manifest("2.0.0");
    let smoke = ok_smoke("higgs 2.0.0");
    assert!(matches!(
        stage_and_flip(&bin, &m, b"not a gzip tarball", false, smoke.as_ref()),
        Err(HiggsError::UpdateApplyFailed { .. })
    ));
    let litter: Vec<_> = std::fs::read_dir(&bin)
        .unwrap()
        .filter_map(Result::ok)
        .filter(|e| e.file_name().to_string_lossy().starts_with(".update-stage"))
        .collect();
    assert!(
        litter.is_empty(),
        "a failed unpack must leave no staging dir"
    );
    assert_eq!(
        current_target(&bin),
        Some("v1.0.0".to_string()),
        "not flipped"
    );
}

#[test]
fn stage_and_flip_cleans_up_a_stage_tree_with_a_restrictive_dir_mode() {
    // A verified-but-malformed archive with a mode-0o000 directory entry: publish rejects
    // the non-flat layout, and the staging tree — whose `locked/` dir is untraversable —
    // must STILL be fully removed, not leaked. Reverting `force_dirs_owner_rwx` in
    // RemoveOnDrop leaves the `.update-stage.*` tree behind and fails this test.
    let tmp = tempfile::tempdir().unwrap();
    let bin = install_layout(tmp.path(), "1.0.0");
    let m = local_manifest("2.0.0");
    let smoke = ok_smoke("higgs 2.0.0");
    assert!(matches!(
        stage_and_flip(
            &bin,
            &m,
            &tarball_with_a_restrictive_dir(),
            false,
            smoke.as_ref()
        ),
        Err(HiggsError::UpdateApplyFailed { .. })
    ));
    let litter: Vec<_> = std::fs::read_dir(&bin)
        .unwrap()
        .filter_map(Result::ok)
        .filter(|e| e.file_name().to_string_lossy().starts_with(".update-stage"))
        .collect();
    assert!(
        litter.is_empty(),
        "a restrictive-mode staging dir must be cleaned up, not leaked: {litter:?}"
    );
    assert_eq!(
        current_target(&bin),
        Some("v1.0.0".to_string()),
        "not flipped"
    );
    assert!(!bin.join("v2.0.0").exists(), "not published");
}

#[test]
fn is_trial_pending_for_tracks_the_running_version() {
    let tmp = tempfile::tempdir().unwrap();
    let bin = install_layout(tmp.path(), "2.0.0");
    assert!(
        !is_trial_pending_for(&bin, "2.0.0"),
        "no trial → not pending"
    );
    write_trial(&bin, "v2.0.0", Some("v1.0.0"));
    assert!(
        is_trial_pending_for(&bin, "2.0.0"),
        "trial for us → pending"
    );
    assert!(
        !is_trial_pending_for(&bin, "9.9.9"),
        "trial for another version → not pending for us"
    );
}

#[test]
fn stage_and_flip_records_no_prev_on_a_first_install() {
    let tmp = tempfile::tempdir().unwrap();
    let bin = tmp.path().join("bin");
    std::fs::create_dir_all(&bin).unwrap(); // no current yet
    let art = tarball_with_higgs(b"first", 0o755);
    let m = local_manifest("1.0.0");
    let smoke = ok_smoke("higgs 1.0.0");
    stage_and_flip(&bin, &m, &art, false, smoke.as_ref()).unwrap();
    assert_eq!(current_target(&bin), Some("v1.0.0".to_string()));
    let raw = std::fs::read(bin.join(".update-trial")).unwrap();
    let marker: TrialMarker = serde_json::from_slice(&raw).unwrap();
    assert_eq!(
        marker.prev, None,
        "no prior current → nothing to roll back to"
    );
}

// ---- verify_and_check (version bind BEFORE the artifact download) --------

#[test]
fn verify_and_check_refuses_a_wrong_version_manifest_before_fetching_the_artifact() {
    // A version-only trigger names the release. If the origin serves a
    // VALIDLY-SIGNED manifest for a DIFFERENT version at that path, the bind
    // must refuse it before spending the artifact bandwidth — `artifact()`
    // here panics to prove the download is never attempted.
    use std::io::Cursor;
    let minisign::KeyPair { pk, sk } =
        minisign::KeyPair::generate_unencrypted_keypair().expect("keygen");
    let manifest_bytes = serde_json::to_vec(&serde_json::json!({
        "schema": 1,
        "version": "9.9.9",
        "commit": "c0ffee0000000000000000000000000000000000",
        "file": "higgs-v9.9.9-aarch64-apple-darwin.tar.gz",
        "target": "aarch64-apple-darwin",
        "variant": "metal",
        "sha256": "aa".repeat(32),
    }))
    .unwrap();
    let sig = minisign::sign(None, &sk, Cursor::new(&manifest_bytes), None, None)
        .expect("sign")
        .into_string();
    struct Signed(Vec<u8>, String);
    impl UpdateSource for Signed {
        fn manifest(&self) -> Result<(Vec<u8>, String), HiggsError> {
            Ok((self.0.clone(), self.1.clone()))
        }
        fn artifact(&self, _f: &str) -> Result<Vec<u8>, HiggsError> {
            panic!("must not fetch the artifact when the manifest is for a different version");
        }
    }
    let pk_b64 = pk.to_base64();
    let table: Vec<(&str, &str)> = vec![("test-key", pk_b64.as_str())];
    let running = BuildIdentity::current();
    let mut authenticated = None;
    let err = verify_and_check_with(
        &Signed(manifest_bytes, sig),
        &running,
        false,
        &mut authenticated,
        Some("8.8.8"),
        &table,
    )
    .unwrap_err();
    assert!(
        matches!(err, HiggsError::UpdateManifestInvalid { .. }),
        "expected the version bind to refuse, got {err:?}"
    );
    // The manifest DID authenticate — the failure report names the real target.
    assert_eq!(authenticated.as_deref(), Some("9.9.9"));
}

// ---- verify_and_check (fail-closed on an unverifiable signature) ---------

#[test]
fn verify_and_check_fails_closed_on_an_unverifiable_signature() {
    // A shipped build pins the real release key, so a manifest whose signature does
    // not verify under it is refused (HG082) before eligibility or download — the
    // artifact is never fetched. (The no-pins HG081 path is covered at the
    // verify_manifest_any level in update_tests.rs.)
    struct AnyBytes;
    impl UpdateSource for AnyBytes {
        fn manifest(&self) -> Result<(Vec<u8>, String), HiggsError> {
            Ok((
                br#"{"schema":1}"#.to_vec(),
                "untrusted comment: x\nAAAA\n".to_string(),
            ))
        }
        fn artifact(&self, _f: &str) -> Result<Vec<u8>, HiggsError> {
            panic!("must not fetch the artifact when the manifest fails to verify");
        }
    }
    let running = BuildIdentity::current();
    let err = verify_and_check(&AnyBytes, &running, false, &mut None, None).unwrap_err();
    assert!(
        matches!(err, HiggsError::UpdateSignatureInvalid { .. }),
        "expected HG082 fail-closed, got {err:?}"
    );
}

// ---- smoke_run (real subprocess against a temp script) -------------------

/// Write an executable shell script at `path` with the given body.
fn write_script(path: &std::path::Path, body: &str) {
    use std::os::unix::fs::PermissionsExt;
    std::fs::write(path, body).unwrap();
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755)).unwrap();
}

#[test]
fn smoke_run_returns_a_scripts_stdout() {
    let tmp = tempfile::tempdir().unwrap();
    let script = tmp.path().join("higgs");
    write_script(&script, "#!/bin/sh\necho \"higgs 9.9.9\"\n");
    let out = smoke_run(&script).unwrap();
    assert!(out.contains("9.9.9"), "got {out:?}");
}

#[test]
fn smoke_run_errors_on_a_nonzero_exit() {
    let tmp = tempfile::tempdir().unwrap();
    let script = tmp.path().join("higgs");
    write_script(&script, "#!/bin/sh\nexit 3\n");
    assert!(matches!(
        smoke_run(&script),
        Err(HiggsError::UpdateApplyFailed { .. })
    ));
}

#[test]
fn smoke_run_errors_when_the_path_is_not_runnable() {
    let tmp = tempfile::tempdir().unwrap();
    let missing = tmp.path().join("nope");
    assert!(matches!(
        smoke_run(&missing),
        Err(HiggsError::UpdateApplyFailed { .. })
    ));
}

#[test]
fn smoke_run_does_not_hang_on_a_descendant_holding_stdout() {
    // A `--version` handler that backgrounds a long-lived child which INHERITS the
    // stdout pipe, prints the version, and exits. smoke_run must reap the process
    // GROUP (killing that descendant → pipe EOF) BEFORE draining stdout, or it would
    // block until the descendant exits. Prove it returns well within the descendant's
    // 30s lifetime by running it on a worker thread with a 5s patience.
    use std::sync::mpsc;
    let tmp = tempfile::tempdir().unwrap();
    let script = tmp.path().join("higgs");
    write_script(
        &script,
        "#!/bin/sh\nsleep 30 &\necho \"higgs 9.9.9\"\nexit 0\n",
    );
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(smoke_run(&script));
    });
    match rx.recv_timeout(std::time::Duration::from_secs(5)) {
        Ok(res) => assert!(res.unwrap().contains("9.9.9")),
        Err(_) => panic!("smoke_run hung on a descendant holding the stdout pipe"),
    }
}

// ---- LocalSource (real file reads) ---------------------------------------

#[test]
fn local_source_reads_the_three_files() {
    let tmp = tempfile::tempdir().unwrap();
    let m = tmp.path().join("m.json");
    let s = tmp.path().join("m.minisig");
    let t = tmp.path().join("a.tar.gz");
    std::fs::write(&m, br#"{"schema":1}"#).unwrap();
    std::fs::write(&s, "sig-text").unwrap();
    std::fs::write(&t, b"tarbytes").unwrap();
    let src = LocalSource {
        manifest: m,
        manifest_sig: s,
        tarball: t,
    };
    let (mb, sig) = src.manifest().unwrap();
    assert_eq!(mb, br#"{"schema":1}"#);
    assert_eq!(sig, "sig-text");
    assert_eq!(src.artifact("ignored").unwrap(), b"tarbytes");
}

#[test]
fn local_source_errors_on_a_missing_manifest() {
    let tmp = tempfile::tempdir().unwrap();
    let src = LocalSource {
        manifest: tmp.path().join("absent.json"),
        manifest_sig: tmp.path().join("absent.sig"),
        tarball: tmp.path().join("absent.tar.gz"),
    };
    assert!(matches!(
        src.manifest(),
        Err(HiggsError::UpdateApplyFailed { .. })
    ));
    assert!(matches!(
        src.artifact("x"),
        Err(HiggsError::UpdateApplyFailed { .. })
    ));
}

// ---- PushSource (hub M_UPDATE: inline manifest+sig, fetched artifact) -----

/// A clearly-PUBLIC literal IPv4 URL — resolves to ITSELF with no DNS query (push tests never
/// touch the network) and passes the SSRF address vet. `PushSource::new` never fetches, so it
/// need not be a live server.
const PUSH_PUBLIC_URL: &str = "https://93.184.216.34/rel/higgs.tar.gz";

/// Construct a `PushSource` DIRECTLY over a LOOPBACK client, bypassing `new`'s https/SSRF vet
/// (which refuses loopback), so `artifact()` can be exercised against a loopback fixture server.
/// White-box: the test module is a child of `self_update`, so it reaches the private fields +
/// `build_fetch_client`.
fn push_source_over_loopback(artifact_url: &str) -> PushSource {
    let url = reqwest::Url::parse(artifact_url).unwrap();
    let client = build_fetch_client(&url).unwrap();
    PushSource {
        manifest: b"MANIFEST-JSON".to_vec(),
        sig: "SIG-TEXT".into(),
        artifact_url: url,
        client,
    }
}

#[test]
fn push_source_returns_the_inline_manifest_without_fetching() {
    // manifest() + the sig are INLINE (no network). new() accepts a public-IP https URL.
    let src = PushSource::new("MANIFEST-JSON", "SIG-TEXT", PUSH_PUBLIC_URL).unwrap();
    let (m, s) = src.manifest().unwrap();
    assert_eq!(m, b"MANIFEST-JSON");
    assert_eq!(s, "SIG-TEXT");
}

#[test]
fn push_source_artifact_fetches_when_the_url_names_the_manifest_file() {
    // A loopback fixture serves the .tar.gz; artifact() fetches it once its last path segment
    // matches the signed manifest file. (Loopback client — `new` refuses loopback under the push
    // SSRF policy; this covers the FETCH delegation.)
    let manifest_url = spawn_loopback_http(b"", b"", b"THE-ARTIFACT", 1);
    let artifact_url = manifest_url.replace("higgs.manifest", "higgs.tar.gz");
    let src = push_source_over_loopback(&artifact_url);
    assert_eq!(src.artifact("higgs.tar.gz").unwrap(), b"THE-ARTIFACT");
}

#[test]
fn is_ssrf_prone_ip_classifies_every_non_global_range() {
    use std::net::IpAddr;
    // One representative per non-global range the push SSRF guard must reject.
    for ip in [
        "127.0.0.1",         // v4 loopback
        "10.1.2.3",          // v4 private 10/8
        "172.16.9.9",        // v4 private 172.16/12
        "192.168.0.1",       // v4 private 192.168/16
        "169.254.169.254",   // v4 link-local (cloud metadata)
        "0.0.0.0",           // v4 unspecified
        "0.1.2.3",           // v4 "this network" 0/8
        "100.64.0.1",        // v4 CGNAT 100.64/10
        "255.255.255.255",   // v4 broadcast
        "224.0.0.1",         // v4 multicast
        "240.0.0.1",         // v4 reserved 240/4
        "192.0.2.5",         // v4 documentation TEST-NET-1
        "192.0.0.8",         // v4 IETF protocol assignments 192.0.0/24
        "192.88.99.2",       // v4 6to4 relay anycast 192.88.99/24
        "198.18.0.1",        // v4 benchmarking 198.18/15
        "198.19.5.5",        // v4 benchmarking (upper half of 198.18/15)
        "::1",               // v6 loopback
        "::",                // v6 unspecified
        "fc00::1",           // v6 unique-local
        "fe80::1",           // v6 link-local
        "ff02::1",           // v6 multicast
        "4000::1",           // v6 outside global-unicast 2000::/3
        "2001:100::1",       // v6 IETF protocol assignments 2001::/23
        "2001:2::1",         // v6 benchmarking (inside 2001::/23)
        "2001:20::1",        // v6 ORCHIDv2 (inside 2001::/23)
        "2001:db8::1",       // v6 documentation 2001:db8::/32
        "2002::1",           // v6 6to4 2002::/16 (deprecated)
        "3fff::1",           // v6 documentation 3fff::/20
        "64:ff9b::10.0.0.1", // v6 NAT64 WKP embedding a PRIVATE v4 (SSRF-via-NAT64)
        "::ffff:127.0.0.1",  // v4-mapped v6 of a loopback v4
    ] {
        assert!(
            is_ssrf_prone_ip(&ip.parse::<IpAddr>().unwrap()),
            "{ip} must be classified SSRF-prone"
        );
    }
    // Real globally-routable addresses must PASS — including a NAT64 WKP mapping of a PUBLIC v4
    // (a DNS64-only node's legitimate release host) and a v4-mapped public v4.
    for ip in [
        "8.8.8.8",
        "1.1.1.1",
        "93.184.216.34",
        "2606:4700:4700::1111",
        "64:ff9b::8.8.8.8", // NAT64 WKP of a PUBLIC v4 — must NOT be rejected
        "::ffff:8.8.8.8",   // v4-mapped public v4
    ] {
        assert!(
            !is_ssrf_prone_ip(&ip.parse::<IpAddr>().unwrap()),
            "{ip} must be classified global"
        );
    }
}

#[test]
fn push_source_new_rejects_ssrf_prone_and_non_https_urls() {
    // The PUSH path is STRICTER than the CLI `--url` path: https only, and no private/loopback/
    // link-local target (RESOLVED, so a domain that resolves to a private IP is refused too), so
    // a compromised hub cannot aim the pre-sha256 artifact GET at the node's own or its LAN's
    // services. Reverting `vet_and_resolve_pushed_url` lets these through.
    for bad in [
        "http://93.184.216.34/higgs.tar.gz", // plaintext http to a public host
        "https://127.0.0.1/higgs.tar.gz",    // loopback
        "https://192.168.1.10/higgs.tar.gz", // RFC1918 private
        "https://10.0.0.5/higgs.tar.gz",     // RFC1918 private
        "https://169.254.169.254/higgs.tar.gz", // link-local (cloud metadata)
        "https://[::1]/higgs.tar.gz",        // IPv6 loopback
    ] {
        assert!(
            matches!(
                PushSource::new("m", "s", bad),
                Err(HiggsError::UpdateFetchFailed { .. })
            ),
            "an SSRF-prone / non-https pushed URL must be refused: {bad}"
        );
    }
    // A public literal IP passes the vet (resolves to itself; `new` does not fetch).
    assert!(PushSource::new("m", "s", PUSH_PUBLIC_URL).is_ok());
    // A public IPv6-LITERAL passes too — the typed-host path takes the bare address, NOT the
    // bracketed `host_str()` that `ToSocketAddrs` would reject (would else fail HG088).
    assert!(
        PushSource::new("m", "s", "https://[2606:4700:4700::1111]/higgs.tar.gz").is_ok(),
        "a public IPv6-literal artifact URL must pass resolution"
    );
}

#[test]
fn push_source_refuses_a_url_that_does_not_name_the_manifest_file() {
    // Defense-in-depth ON TOP of the address vet: artifact() binds the URL to the signed `file`
    // (last path segment must match), refusing an arbitrary endpoint BEFORE any GET. (Loopback
    // client used only so the mismatch is what fails, not `new`'s address vet.)
    let src = push_source_over_loopback("http://127.0.0.1:9/state-changing-get");
    let err = src
        .artifact("higgs.tar.gz")
        .expect_err("a URL that does not name the manifest file is refused");
    assert!(
        err.to_string()
            .contains("does not name the manifest's file"),
        "the refusal is the filename bind, not a fetch error: {err}"
    );
}

#[test]
fn apply_pushed_update_fails_closed_on_an_unverifiable_signature() {
    // The hub-push apply reaches signature verification and — because the manifest's signature
    // does not verify under the pinned release key — refuses (HG082) BEFORE fetching the
    // artifact. Proves the full push apply wrapper (root-check → PushSource → installed_identity
    // → verify_and_check) is wired; the artifact URL is never fetched (verification fails first),
    // so it only needs to parse.
    if unsafe { libc::geteuid() } == 0 {
        return; // apply refuses as root; this asserts the non-root verify path
    }
    let tmp = tempfile::tempdir().unwrap();
    let bin = install_layout(tmp.path(), "1.0.0"); // bin/v1.0.0/higgs + current -> v1.0.0
    let err = apply_pushed_update(
        &bin,
        r#"{"schema":1,"version":"2.0.0","commit":"x","file":"higgs.tar.gz","target":"t","variant":"v","sha256":"00"}"#,
        "untrusted comment: x\nRWQAAA==\n",
        "https://93.184.216.34/rel/higgs.tar.gz",
        false,
        "2.0.0", // the push's declared target hint
    )
    .unwrap_err();
    assert!(
        err.to_string().contains("HG082"),
        "an unverifiable signature must fail closed at verification: {err}"
    );
    // A POST-lock failure (verify failed here) records the marker so the node reports WHY on its
    // next HELLO: from = the installed version, to = the hint (the signature failed before the
    // authenticated version was known). Reverting the record-on-Err arm in apply leaves no marker.
    let marker = peek_update_failure(&bin).expect("apply records the connected failure");
    assert_eq!(
        marker.from,
        env!("CARGO_PKG_VERSION"),
        "from = the RUNNING build (matches the HELLO software_version), not installed `current`"
    );
    assert_eq!(
        marker.to, "2.0.0",
        "to = the declared target hint (pre-authentication)"
    );
    assert!(
        marker.reason.contains("HG082"),
        "the reason carries the failure: {:?}",
        marker.reason
    );
}

#[test]
fn apply_records_a_source_validation_failure() {
    // A rejected/unreachable artifact URL fails at `PushSource::new` (under the lock) — but the node
    // has already replied "accepted", so it must STILL record WHY for the next HELLO (else the hub
    // only ever sees a version that never advances). A non-https URL is refused synchronously (no
    // network). Reverting the record-on-Err arm leaves no marker and this fails.
    if unsafe { libc::geteuid() } == 0 {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let bin = install_layout(tmp.path(), "1.0.0");
    let err = apply_pushed_update(
        &bin,
        r#"{"schema":1,"version":"2.0.0","commit":"x","file":"higgs.tar.gz","target":"t","variant":"v","sha256":"00"}"#,
        "untrusted comment: x\nRWQAAA==\n",
        "http://93.184.216.34/rel/higgs.tar.gz", // NON-https → PushSource::new refuses
        false,
        "2.0.0",
    )
    .unwrap_err();
    assert!(
        err.to_string().to_lowercase().contains("https"),
        "the refusal is the https/URL policy: {err}"
    );
    let marker = peek_update_failure(&bin).expect("a source-validation failure is still recorded");
    assert_eq!(
        marker.from,
        env!("CARGO_PKG_VERSION"),
        "from = the running build"
    );
    assert_eq!(
        marker.to, "2.0.0",
        "to = the declared hint (nothing is authenticated yet)"
    );
}

#[test]
fn a_contended_apply_records_nothing_even_with_a_bad_source() {
    // The overlapping-push race fix: ALL recordable validation runs UNDER the lock, so a push that
    // LOSES the lock to a concurrent apply records NOTHING — its failure can never race the
    // lock-holder's success-clear. Holding the lock + a BAD (non-https) URL: the race-prone
    // pre-lock ordering would validate the URL and record its failure; the under-lock ordering
    // instead yields Contended (HG087) with NO marker. Reverting the lock-first ordering (validating
    // the source before the lock) makes this record a marker + return the https error, failing both
    // asserts.
    if unsafe { libc::geteuid() } == 0 {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let bin = install_layout(tmp.path(), "1.0.0");
    let _held = UpdateLock::acquire(&bin).unwrap(); // a concurrent apply holds the lock
    let err = apply_pushed_update(
        &bin,
        r#"{"schema":1,"version":"2.0.0","commit":"x","file":"higgs.tar.gz","target":"t","variant":"v","sha256":"00"}"#,
        "untrusted comment: x\nRWQAAA==\n",
        "http://93.184.216.34/rel/higgs.tar.gz", // a BAD url that the race-prone ordering would record
        false,
        "2.0.0",
    )
    .unwrap_err();
    assert!(
        err.to_string().contains("HG087"),
        "a contended push is refused at the lock, not by (recordable) URL validation: {err}"
    );
    assert!(
        peek_update_failure(&bin).is_none(),
        "a contended push records nothing — its validation runs UNDER the lock, so it cannot race \
         the lock-holder's success-clear"
    );
}

#[test]
fn apply_pushed_update_refuses_a_concurrent_apply_before_fetching() {
    // The lock is acquired BEFORE the artifact fetch, so a CONCURRENT update (holding the lock)
    // refuses THIS push with HG087 IMMEDIATELY — never buffering the (up-to-256 MiB) artifact
    // (the OOM a hostile server could trigger with N concurrent pushes). Moving the lock back
    // to AFTER verify lets verify run first: in a keyless build that yields HG081 (verify
    // failed), not HG087 (lock contended) — the mutant this kills.
    if unsafe { libc::geteuid() } == 0 {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let bin = install_layout(tmp.path(), "1.0.0");
    let _held = UpdateLock::acquire(&bin).unwrap(); // a concurrent update holds the lock
    let err = apply_pushed_update(
        &bin,
        r#"{"schema":1,"version":"2.0.0","commit":"x","file":"higgs.tar.gz","target":"t","variant":"v","sha256":"00"}"#,
        "untrusted comment: x\nRWQAAA==\n",
        "https://93.184.216.34/rel/higgs.tar.gz",
        false,
        "2.0.0",
    )
    .unwrap_err();
    assert!(
        err.to_string().contains("HG087"),
        "a concurrent push must be refused at the lock BEFORE fetching (HG087), not after \
         verify (HG081): {err}"
    );
    // A lock contention is TRANSIENT (another apply holds it), not this target failing — so it
    // must NOT record a failure marker. Recording it would report a spurious failure the moment
    // the other apply succeeds. Reverting the pre-lock/post-lock split (recording on ALL errors)
    // leaves a marker here and this fails.
    assert!(
        peek_update_failure(&bin).is_none(),
        "a concurrent-lock refusal records no failure marker"
    );
}

#[test]
fn apply_records_a_hard_lock_failure() {
    // A HARD lock failure (`LockAttempt::Failed` — a non-regular/malformed lock node, NOT genuine
    // contention) is a definitive failure of THIS push: the node already replied "accepted" and
    // stays put, so it must record WHY for the next HELLO. Only transient CONTENTION is exempt
    // (covered above). Reverting the Contended/Failed split (treating both as unrecorded) leaves no
    // marker here.
    if unsafe { libc::geteuid() } == 0 {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let bin = install_layout(tmp.path(), "1.0.0");
    // Force `LockAttempt::Failed`: seed the lock dir/file via a real acquire, then replace the lock
    // node with a DIRECTORY (a non-regular node try_acquire refuses as a hard failure).
    let seed = UpdateLock::acquire(&bin).unwrap();
    drop(seed);
    std::fs::remove_file(lock_path(&bin)).unwrap();
    std::fs::create_dir(lock_path(&bin)).unwrap();
    let _err = apply_pushed_update(
        &bin,
        r#"{"schema":1,"version":"2.0.0","commit":"x","file":"higgs.tar.gz","target":"t","variant":"v","sha256":"00"}"#,
        "untrusted comment: x\nRWQAAA==\n",
        "https://93.184.216.34/rel/higgs.tar.gz",
        false,
        "2.0.0",
    )
    .unwrap_err();
    let marker = peek_update_failure(&bin).expect("a hard lock failure is recorded");
    assert_eq!(
        marker.from,
        env!("CARGO_PKG_VERSION"),
        "from = the running build"
    );
    assert_eq!(marker.to, "2.0.0", "to = the declared hint");
}

#[test]
fn check_pushed_sizes_caps_the_inline_manifest_and_signature() {
    // A hub-PUSHED manifest+sig are INLINE; a paired-but-compromised hub could send a huge
    // string to OOM the node. The cap must reject it BEFORE any copy.
    let big_manifest = "x".repeat(70 * 1024); // > 64 KiB
    assert!(matches!(
        check_pushed_sizes(&big_manifest, "sig"),
        Err(HiggsError::UpdateFetchFailed { .. })
    ));
    let big_sig = "x".repeat(5 * 1024); // > 4 KiB
    assert!(matches!(
        check_pushed_sizes("{}", &big_sig),
        Err(HiggsError::UpdateFetchFailed { .. })
    ));
    assert!(check_pushed_sizes("{}", "sig").is_ok());
    // PushSource::new enforces it too (belt), before the copy.
    assert!(matches!(
        PushSource::new(&big_manifest, "s", "https://h/a.tar.gz"),
        Err(HiggsError::UpdateFetchFailed { .. })
    ));
}

// ---- HttpSource URL vetting (pure) ---------------------------------------

#[test]
fn parse_fetch_url_allows_https_and_loopback_http_only() {
    assert!(parse_fetch_url("https://example.com/rel/higgs.manifest").is_ok());
    assert!(parse_fetch_url("http://127.0.0.1:8080/higgs.manifest").is_ok());
    assert!(parse_fetch_url("http://localhost/higgs.manifest").is_ok());
    assert!(parse_fetch_url("http://[::1]:9000/higgs.manifest").is_ok());
    // plaintext http to a NON-loopback host is refused (an operator almost surely meant https)
    assert!(matches!(
        parse_fetch_url("http://example.com/higgs.manifest"),
        Err(HiggsError::UpdateFetchFailed { .. })
    ));
    // a public IP over http is refused too
    assert!(matches!(
        parse_fetch_url("http://93.184.216.34/higgs.manifest"),
        Err(HiggsError::UpdateFetchFailed { .. })
    ));
    // non-http(s) schemes and garbage are refused
    for bad in [
        "ftp://example.com/x",
        "file:///etc/passwd",
        "not a url",
        "//example.com/x",
    ] {
        assert!(
            matches!(
                parse_fetch_url(bad),
                Err(HiggsError::UpdateFetchFailed { .. })
            ),
            "should refuse {bad:?}"
        );
    }
    // credentials / query / fragment are refused (they corrupt the derived .minisig +
    // artifact URLs, or could leak on a downgrade redirect)
    for bad in [
        "https://user:secret@example.com/higgs.manifest",
        "https://user@example.com/higgs.manifest",
        "https://example.com/higgs.manifest?token=x",
        "https://example.com/higgs.manifest#frag",
    ] {
        assert!(
            matches!(
                parse_fetch_url(bad),
                Err(HiggsError::UpdateFetchFailed { .. })
            ),
            "should refuse {bad:?}"
        );
    }
}

#[test]
fn parse_fetch_url_errors_never_leak_a_secret() {
    // Rejecting a credential- or query-bearing URL must NOT echo the secret into the error
    // (which lands in stderr / automation logs). Reverting `redact_url` to `raw` leaks it.
    for (url, secret) in [
        (
            "https://user:SUPERSECRET@example.com/higgs.manifest",
            "SUPERSECRET",
        ),
        (
            "https://example.com/higgs.manifest?X-Amz-Signature=TOPSECRET",
            "TOPSECRET",
        ),
        ("https://tok:PW@example.com/higgs.manifest#SECRETFRAG", "PW"),
    ] {
        let err = parse_fetch_url(url).unwrap_err();
        let msg = err.to_string();
        assert!(
            !msg.contains(secret),
            "error must not leak {secret:?}, got: {msg}"
        );
    }
    // The redacted form names NO URL component — not the scheme, host, path, or query
    // (each can carry a capability: a `cap<token>://` scheme, a `cap-<token>.host` subdomain,
    // a `/download/<token>/` path, a presigned query). A made-up scheme is refused too.
    for u in [
        "https://user:x@cap-secret.updates.example/rel/higgs.manifest?t=1",
        "capsecretdeadbeef://updates.example/higgs.manifest",
    ] {
        let msg = parse_fetch_url(u).unwrap_err().to_string();
        for leak in [
            "cap-secret",
            "updates.example",
            "capsecretdeadbeef",
            "rel/higgs",
        ] {
            assert!(
                !msg.contains(leak),
                "must not leak {leak:?} from {u:?}: {msg}"
            );
        }
    }
}

/// A loopback HTTP/1.1 fixture: replies `sig` for `.minisig`, `artifact` for `.tar.gz`,
/// else `manifest`. Serves `n` requests then exits. Returns the manifest URL.
fn spawn_loopback_http(manifest: &[u8], sig: &[u8], artifact: &[u8], n: usize) -> String {
    use std::io::{Read, Write};
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let (m, s, a) = (manifest.to_vec(), sig.to_vec(), artifact.to_vec());
    std::thread::spawn(move || {
        for _ in 0..n {
            let Ok((mut stream, _)) = listener.accept() else {
                return;
            };
            let mut buf = [0u8; 2048];
            let read = stream.read(&mut buf).unwrap_or(0);
            let req = String::from_utf8_lossy(&buf[..read]);
            let body: &[u8] = if req.contains(".minisig") {
                &s
            } else if req.contains(".tar.gz") {
                &a
            } else {
                &m
            };
            let head = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            let _ = stream.write_all(head.as_bytes());
            let _ = stream.write_all(body);
            let _ = stream.flush();
        }
    });
    format!("http://127.0.0.1:{port}/higgs.manifest")
}

#[test]
fn http_source_fetches_manifest_sig_and_artifact() {
    let url = spawn_loopback_http(b"THE-MANIFEST", b"THE-SIG", b"THE-ARTIFACT", 3);
    let src = HttpSource::new(&url).unwrap();
    let (m, sig) = src.manifest().unwrap();
    assert_eq!(m, b"THE-MANIFEST");
    assert_eq!(sig, "THE-SIG");
    // artifact() derives the sibling URL from the manifest's `file` field.
    let a = src.artifact("higgs.tar.gz").unwrap();
    assert_eq!(a, b"THE-ARTIFACT");
}

#[test]
fn http_source_bounds_an_oversize_body() {
    // A body over the cap (checked via Content-Length) is refused — the DoS guard.
    let url = spawn_loopback_http(b"MANIFEST-BODY-13", b"", b"", 1);
    let src = HttpSource::new(&url).unwrap();
    let u = reqwest::Url::parse(&url).unwrap();
    assert!(matches!(
        src.get_bounded(&u, 3, "manifest"),
        Err(HiggsError::UpdateFetchFailed { .. })
    ));
}

#[test]
fn get_bounded_caps_a_body_with_no_content_length() {
    use std::io::{Read, Write};
    // The ADVERSARIAL path: a server that sends NO Content-Length (so the header pre-check
    // is skipped) then streams a body over the cap. The read-loop append-cap is then the
    // SOLE size bound — it MUST fire. Removing it lets a hostile server stream unbounded
    // into RAM (the OOM DoS the cap exists to prevent); this test is what catches that.
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    std::thread::spawn(move || {
        if let Ok((mut stream, _)) = listener.accept() {
            let mut b = [0u8; 1024];
            let _ = stream.read(&mut b);
            let _ = stream.write_all(b"HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n");
            let _ = stream.write_all(&vec![b'x'; 10_000]); // 10 KB, NO Content-Length
        }
    });
    let u_str = format!("http://127.0.0.1:{port}/higgs.manifest");
    let src = HttpSource::new(&u_str).unwrap();
    let u = reqwest::Url::parse(&u_str).unwrap();
    assert!(
        matches!(
            src.get_bounded(&u, 4096, "manifest"),
            Err(HiggsError::UpdateFetchFailed { .. })
        ),
        "the streaming append-cap must bound a body with no Content-Length"
    );
}

#[test]
fn a_redirect_is_not_followed() {
    use std::io::{Read, Write};
    // Redirects are NOT followed (SSRF surface + sig/artifact-sibling ambiguity). A 302 —
    // even same-origin — surfaces as an error; the redirect TARGET (a reachable second
    // location returning a distinctive body) is NEVER fetched. Reverting to a follow policy
    // would return REDIRECT-TARGET-BODY and fail this.
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    std::thread::spawn(move || {
        for i in 0..2 {
            let Ok((mut stream, _)) = listener.accept() else {
                return;
            };
            let mut b = [0u8; 1024];
            let _ = stream.read(&mut b);
            if i == 0 {
                let loc = format!("http://127.0.0.1:{port}/redirected");
                let _ = stream.write_all(
                    format!("HTTP/1.1 302 Found\r\nLocation: {loc}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").as_bytes(),
                );
            } else {
                let body = b"REDIRECT-TARGET-BODY";
                let _ = stream.write_all(
                    format!(
                        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        body.len()
                    )
                    .as_bytes(),
                );
                let _ = stream.write_all(body);
            }
        }
    });
    let u_str = format!("http://127.0.0.1:{port}/higgs.manifest");
    let src = HttpSource::new(&u_str).unwrap();
    let u = reqwest::Url::parse(&u_str).unwrap();
    let got = src.get_bounded(&u, 4096, "manifest");
    match got {
        Err(HiggsError::UpdateFetchFailed { detail }) => {
            assert!(
                !detail.contains("REDIRECT-TARGET-BODY"),
                "must not have fetched the redirect target"
            );
            assert!(
                detail.contains("redirected") || detail.contains("302"),
                "should report the redirect: {detail}"
            );
        }
        other => panic!("a 302 must surface as an error, not be followed: {other:?}"),
    }
}

#[test]
fn fetch_errors_never_leak_a_path_secret() {
    use std::io::{Read, Write};
    // A URL PATH can carry a capability token. A 404 (or any fetch failure) must NOT echo
    // the path into the error. Reverting redact_url to keep the path (or dropping
    // `.without_url()`) leaks PATHSECRET and fails this.
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    std::thread::spawn(move || {
        if let Ok((mut stream, _)) = listener.accept() {
            let mut b = [0u8; 1024];
            let _ = stream.read(&mut b);
            // The REASON PHRASE is server-controlled — put a secret there too, to prove we
            // format only the numeric status, never the phrase.
            let _ = stream.write_all(
                b"HTTP/1.1 404 REASONSECRET\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
            );
            let _ = stream.flush();
        }
    });
    let url = format!("http://127.0.0.1:{port}/download/PATHSECRET/higgs.manifest");
    let src = HttpSource::new(&url).unwrap();
    let err = src.manifest().unwrap_err().to_string();
    assert!(
        !err.contains("PATHSECRET"),
        "a path capability token must not leak into the fetch error: {err}"
    );
    assert!(
        !err.contains("REASONSECRET"),
        "a server-controlled reason phrase must not leak into the fetch error: {err}"
    );
    // The host (itself a possible capability subdomain) is NOT named — no URL component is.
    assert!(
        !err.contains("127.0.0.1"),
        "the host must not be named: {err}"
    );
    // The error still identifies the fetch by its `what` label + HG088, without any URL.
    assert!(
        err.contains("manifest") && err.contains("HG088"),
        "should name the `what` + HG088: {err}"
    );
}

#[test]
fn oversize_content_length_is_not_reflected_into_the_error() {
    use std::io::{Read, Write};
    // A hostile server can set Content-Length to a numeric capability token to reflect it
    // into our logs. The over-cap error must name only our own cap, never the server length.
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    std::thread::spawn(move || {
        if let Ok((mut stream, _)) = listener.accept() {
            let mut b = [0u8; 1024];
            let _ = stream.read(&mut b);
            // claim a huge (over-cap) length carrying a "secret" number, short/no body
            let _ = stream.write_all(
                b"HTTP/1.1 200 OK\r\nContent-Length: 987654321\r\nConnection: close\r\n\r\n",
            );
            let _ = stream.flush();
        }
    });
    let src = HttpSource::new(&format!("http://127.0.0.1:{port}/higgs.manifest")).unwrap();
    let err = src.manifest().unwrap_err().to_string();
    assert!(
        !err.contains("987654321"),
        "the server Content-Length must not be reflected into the error: {err}"
    );
}

#[test]
fn http_source_errors_on_a_non_2xx_status() {
    use std::io::{Read, Write};
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    std::thread::spawn(move || {
        if let Ok((mut stream, _)) = listener.accept() {
            let mut buf = [0u8; 1024];
            let _ = stream.read(&mut buf);
            let _ = stream.write_all(
                b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
            );
            let _ = stream.flush();
        }
    });
    let src = HttpSource::new(&format!("http://127.0.0.1:{port}/higgs.manifest")).unwrap();
    assert!(matches!(
        src.manifest(),
        Err(HiggsError::UpdateFetchFailed { .. })
    ));
}

#[test]
fn artifact_url_is_a_vetted_bare_sibling_of_the_manifest() {
    let m = reqwest::Url::parse("https://host.example/rel/v0.5.0/higgs.manifest").unwrap();
    let a = artifact_url_from(&m, "higgs-v0.5.0-aarch64-apple-darwin.tar.gz").unwrap();
    assert_eq!(
        a.as_str(),
        "https://host.example/rel/v0.5.0/higgs-v0.5.0-aarch64-apple-darwin.tar.gz"
    );
    // a `file` field that is not a bare sibling filename is refused — no traversal, no
    // absolute path, no nested dir, no leading dot, no backslash.
    for bad in [
        "../evil.tar.gz",
        "a/b.tar.gz",
        "/etc/passwd",
        "",
        ".hidden.tar.gz",
        "x\\y.tar.gz",
        "weird..name.tar.gz",
    ] {
        assert!(
            matches!(
                artifact_url_from(&m, bad),
                Err(HiggsError::UpdateFetchFailed { .. })
            ),
            "should refuse file {bad:?}"
        );
    }
}

// ── P4b (c): the update-restart trigger + operator-precedence shutdown selector ──────────────

/// The dedicated update-restart trigger and the biased shutdown selector — the whole point being
/// that an operator stop can NEVER be reclassified as an update restart (the bug a global flag
/// sampled late in the teardown allowed). All three causes are exercised without a real signal
/// (the trigger is an in-process `Notify`, testable where a self-SIGTERM was not):
///   (a) a lone operator signal → `Operator` (prompt stop, no drain/re-exec);
///   (b) a lone restart request → `Update` (drain + re-exec);
///   (c) a restart request PENDING while an operator stop also fires → `Operator` wins (`biased`).
/// (c) is the race fix: swapping the selector's arms (or dropping `biased`) reclassifies the
/// operator stop as an update restart and fails this assert. This test is the sole toucher of the
/// process-global trigger, and it drains every permit it stores so none leaks to another test.
#[tokio::test]
async fn an_operator_stop_is_never_reclassified_as_an_update_restart() {
    use crate::node::self_update::{await_node_shutdown, request_self_restart, ShutdownCause};
    use std::future::{pending, ready};

    // (a) An operator signal with NO pending update → a prompt stop.
    assert_eq!(
        await_node_shutdown(ready(())).await,
        ShutdownCause::Operator,
        "an operator SIGTERM alone is a prompt stop"
    );

    // (b) A restart request with NO operator signal → drain + re-exec. `notify_one` stored a
    // permit; the selector's update arm consumes it (operator future never resolves).
    request_self_restart();
    assert_eq!(
        await_node_shutdown(pending::<()>()).await,
        ShutdownCause::Update,
        "a lone update request drains + re-execs"
    );

    // (c) THE RACE FIX: a request pending (an update flipped concurrently) WHILE an operator stop
    // fires → the operator wins. The staged, verified update simply activates on the next start.
    request_self_restart();
    assert_eq!(
        await_node_shutdown(ready(())).await,
        ShutdownCause::Operator,
        "a concurrent update never turns an operator stop into a re-exec"
    );
    // (c)'s biased operator win left its permit UNCONSUMED — drain it (operator pending, so the
    // update arm consumes it) so it cannot leak into a later test as a spurious `Update`.
    assert_eq!(
        await_node_shutdown(pending::<()>()).await,
        ShutdownCause::Update,
        "the leftover permit drains cleanly"
    );
}

/// The monotonic Update→Operator override during the update drain: once an update restart is
/// underway, the daemon keeps watching the operator signal so a stop arriving MID-drain still wins
/// (no re-exec). `Some(drained)` = the drain owned the outcome (re-exec); `None` = the operator
/// overrode (prompt stop). Case (c) — operator AND drain both ready — must yield `None` (biased);
/// swapping the selector's arms yields `Some` and re-execs against the operator's wishes, failing
/// this assert.
#[tokio::test]
async fn an_operator_stop_mid_drain_overrides_the_re_exec() {
    use crate::node::self_update::drain_with_operator_override;
    use std::future::{pending, ready};

    // (a) The drain finishes cleanly, no operator stop → re-exec, carrying the clean/deadline bit.
    assert_eq!(
        drain_with_operator_override(pending::<()>(), ready(true)).await,
        Some(true),
        "a clean drain re-execs (drained = true)"
    );
    assert_eq!(
        drain_with_operator_override(pending::<()>(), ready(false)).await,
        Some(false),
        "a deadlined drain still re-execs (drained = false)"
    );

    // (b) An operator stop with the drain still running → override, NO re-exec.
    assert_eq!(
        drain_with_operator_override(ready(()), pending::<bool>()).await,
        None,
        "an operator stop mid-drain abandons the re-exec"
    );

    // (c) Operator AND drain BOTH ready → the operator wins (biased). This is the override's
    // guarantee: a stop is never lost to a drain that also just finished.
    assert_eq!(
        drain_with_operator_override(ready(()), ready(true)).await,
        None,
        "an operator stop wins a tie with a just-finished drain"
    );
}

/// The FINAL operator gate before the re-exec: a non-blocking poll of the operator signal so a
/// SIGTERM buffered during the (unabortable) worker teardown still cancels the re-exec. `true` when
/// the operator has fired, `false` when it has not — and it never blocks (a pending operator
/// resolves to `false` immediately). Reverting the poll to always-`false` (or dropping the guard in
/// `cli.rs`) lets an operator's mid-teardown stop be overridden by a restart onto `current`.
#[tokio::test]
async fn a_buffered_operator_stop_is_consumed_at_the_final_re_exec_gate() {
    use crate::node::self_update::operator_stop_pending;
    use std::future::{pending, ready};

    assert!(
        operator_stop_pending(ready(())).await,
        "a fired operator signal is observed at the gate → cancel the re-exec"
    );
    assert!(
        !operator_stop_pending(pending::<()>()).await,
        "no operator stop → the gate does not block and the re-exec proceeds"
    );
}

// ── UPx: version-only update (node self-fetch) ────────────────────────────────

#[test]
fn release_asset_urls_derive_from_base_and_build() {
    let (m, s, a) = release_asset_urls("https://github.com/o/r/releases", "1.2.3").unwrap();
    let suffix =
        crate::node::release_courier::asset_suffix(env!("HIGGS_BUILD_TARGET"), CURRENT_VARIANT);
    assert_eq!(
        m.as_str(),
        format!("https://github.com/o/r/releases/download/v1.2.3/higgs-v1.2.3-{suffix}.manifest")
    );
    assert!(s
        .as_str()
        .ends_with(&format!("higgs-v1.2.3-{suffix}.manifest.minisig")));
    assert!(a
        .as_str()
        .ends_with(&format!("higgs-v1.2.3-{suffix}.tar.gz")));
    // Trailing slash on the base tolerated.
    release_asset_urls("https://github.com/o/r/releases/", "1.2.3").unwrap();
    // Loopback http base allowed (the shared on-box/test affordance).
    release_asset_urls("http://127.0.0.1:8080/rel", "1.2.3").unwrap();
}

#[test]
fn release_asset_urls_gate_version_and_base() {
    // A hub-supplied version lands in a URL path — every non-semver shape must die.
    for bad in [
        "",
        "v1.2.3",
        "1.2.3/../evil",
        "1.2.3?x=1",
        "latest",
        "1.2",
        "a.b.c",
    ] {
        assert!(
            release_asset_urls("https://github.com/o/r/releases", bad).is_err(),
            "{bad:?} must be refused"
        );
    }
    // Base URL policy comes from the shared courier vet.
    assert!(release_asset_urls("http://mirror.example/r", "1.2.3").is_err());
    assert!(release_asset_urls("https://u:p@github.com/o/r/releases", "1.2.3").is_err());
    assert!(release_asset_urls("https://github.com/o/r/releases?tok=1", "1.2.3").is_err());
}

#[test]
fn release_redirects_https_only() {
    let ok = reqwest::Url::parse("https://objects.githubusercontent.com/x?sig=1").unwrap();
    assert!(release_redirect_ok(&ok), "https + query is a valid hop");
    let http = reqwest::Url::parse("http://example.com/x").unwrap();
    assert!(!release_redirect_ok(&http), "plaintext hop refused");
    let cred = reqwest::Url::parse("https://u:p@example.com/x").unwrap();
    assert!(!release_redirect_ok(&cred), "credentialed hop refused");
}

// ── version-only update: URL derivation + fetch policy (UPx) ────────────────

#[test]
fn release_asset_urls_derives_the_trio_and_gates_the_version_syntax() {
    // Non-semver (separators/traversal would land in a URL path) dies first.
    for bad in ["v1.2.3", "1.2", "../../x", "1.2.3/evil"] {
        let err = release_asset_urls("https://github.com/o/r/releases", bad).unwrap_err();
        assert!(
            matches!(err, HiggsError::UpdateFetchFailed { .. }),
            "{bad:?} → {err:?}"
        );
    }
    // A clean base yields the three siblings under the tag dir for THIS build.
    let ident = BuildIdentity::current();
    let suffix = crate::node::release_courier::asset_suffix(&ident.target, &ident.variant);
    let (m, s, a) = release_asset_urls("https://github.com/o/r/releases", "1.2.3").unwrap();
    let dir = format!("https://github.com/o/r/releases/download/v1.2.3/higgs-v1.2.3-{suffix}");
    assert_eq!(m.as_str(), format!("{dir}.manifest"));
    assert_eq!(s.as_str(), format!("{dir}.manifest.minisig"));
    assert_eq!(a.as_str(), format!("{dir}.tar.gz"));
    // A base the courier URL policy refuses (query) propagates the refusal.
    assert!(release_asset_urls("https://github.com/o/r/releases?x=1", "1.2.3").is_err());
}

#[test]
fn release_redirect_targets_must_be_https_without_credentials() {
    let ok = |u: &str| release_redirect_ok(&reqwest::Url::parse(u).unwrap());
    assert!(ok("https://objects.example.com/a?sig=abc"));
    assert!(
        !ok("http://127.0.0.1/a"),
        "no plaintext hop, not even loopback"
    );
    assert!(
        !ok("https://user@objects.example.com/a"),
        "no credentialed hop"
    );
}

/// One-shot blocking HTTP server on loopback: answers every request with `status`
/// and `body` until dropped. Loopback http is the on-box test affordance the
/// release fetch policy explicitly allows.
fn serve_loopback(status: &'static str, body: &'static [u8]) -> (std::net::TcpListener, String) {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!(
        "http://127.0.0.1:{}/releases",
        listener.local_addr().unwrap().port()
    );
    let l2 = listener.try_clone().unwrap();
    std::thread::spawn(move || {
        for stream in l2.incoming() {
            let Ok(mut s) = stream else { break };
            use std::io::{Read, Write};
            let mut buf = [0u8; 2048];
            let _ = s.read(&mut buf);
            let _ = write!(
                s,
                "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            let _ = s.write_all(body);
        }
    });
    (listener, base)
}

#[test]
fn version_source_missing_release_maps_to_a_fetch_failure() {
    let (_l, base) = serve_loopback("404 Not Found", b"nope");
    let source = VersionSource::new(&base, "9.9.9").expect("loopback http base is allowed");
    let err = source.manifest().unwrap_err();
    assert!(
        matches!(err, HiggsError::UpdateFetchFailed { .. }),
        "got {err:?}"
    );
}

#[test]
fn version_source_fetches_but_garbage_fails_the_signature_closed() {
    // The fetch itself succeeds (manifest + sig both served) — trust comes ONLY
    // from the CI signature, and garbage does not verify under any pinned key.
    let (_l, base) = serve_loopback("200 OK", b"not a manifest");
    let source = VersionSource::new(&base, "9.9.9").expect("source");
    let running = BuildIdentity::current();
    let err = verify_and_check(&source, &running, false, &mut None, Some("9.9.9")).unwrap_err();
    assert!(
        matches!(err, HiggsError::UpdateSignatureInvalid { .. }),
        "got {err:?}"
    );
}

// ── apply_version_update (config-driven fetch, fail-closed, failure recorded) ─

#[test]
fn apply_version_update_fails_closed_and_records_from_the_configured_release_url() {
    // The node's OWN config names the release origin — point it at a loopback
    // server that serves garbage: the fetch succeeds, the CI signature fails
    // closed, and the failure is recorded for the next HELLO. Serialized with
    // the other HIGGS_HOME tests; the var is restored after.
    let _env = crate::TEST_ENV_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (_l, base) = serve_loopback("200 OK", b"not a manifest");
    let home = tempfile::tempdir().unwrap();
    let prev = std::env::var_os("HIGGS_HOME");
    // SAFETY: serialized by TEST_ENV_LOCK; restored below.
    unsafe { std::env::set_var("HIGGS_HOME", home.path()) };
    std::fs::write(
        home.path().join("config.json"),
        serde_json::to_vec(&serde_json::json!({ "release_url": base })).unwrap(),
    )
    .unwrap();
    let bin = tempfile::tempdir().unwrap();
    let err = apply_version_update(bin.path(), "9.9.9").unwrap_err();
    unsafe {
        match prev {
            Some(v) => std::env::set_var("HIGGS_HOME", v),
            None => std::env::remove_var("HIGGS_HOME"),
        }
    }
    assert!(
        matches!(err, HiggsError::UpdateSignatureInvalid { .. }),
        "garbage manifest fails the signature closed: {err:?}"
    );
    let failed = peek_update_failure(bin.path()).expect("failure recorded for the next HELLO");
    assert_eq!(
        failed.to, "9.9.9",
        "pre-verify failure reports the hub's named target"
    );
}

#[test]
fn apply_version_update_refuses_while_another_update_holds_the_lock() {
    let bin = tempfile::tempdir().unwrap();
    let held = match UpdateLock::try_acquire(bin.path()) {
        LockAttempt::Held(l) => l,
        _ => panic!("fresh dir lock must be acquirable"),
    };
    let err = apply_version_update(bin.path(), "1.2.3").unwrap_err();
    drop(held);
    assert!(
        format!("{err}").contains("another self-update is already running"),
        "got {err}"
    );
}
