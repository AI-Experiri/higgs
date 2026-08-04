//! Node self-update — verified, atomic binary swap with rollback (§9 P3).
//!
//! Builds on the P1 TRUST ANCHOR ([`crate::update`]: `verify_manifest_any`,
//! `verify_artifact_sha256`, error taxonomy HG081-084) and the P2 install layout
//! (`install.sh`: `<prefix>/bin/v<ver>/higgs`, atomic `current` symlink). Where
//! [`crate::update`] answers AUTHENTICITY only ("did release CI build this?"),
//! this module adds ELIGIBILITY ("is it an upgrade for THIS build?") and APPLY
//! (stage → smoke → atomic flip → trial marker → boot-guarded rollback).
//!
//! ## Trust model recap
//! A dev/default build pins NO release keys, so EVERY manifest verification fails
//! closed (HG081) and a signed self-update is simply impossible — the update is
//! not "unverified", it is refused. Only a build that pins a key (the one-time
//! operator step in [`crate::update::HIGGS_UPDATE_PUBKEYS`]) can self-update. The
//! no-signature maintenance ops (`--rollback`, `--prune`) operate on already
//! installed version dirs and need no key.
//!
//! ## Why in-process, not a shell-out
//! Unpacking and the `current` flip run IN-PROCESS (Rust `tar`/`flate2`, `rename(2)`)
//! rather than shelling to `tar`/`ln`/`mv` — a bare `tar` on `$PATH` is exactly the
//! planted-binary class the install path spent its hardening on. The one unavoidable
//! subprocess (the post-stage `--version` smoke test) runs the JUST-STAGED binary
//! with a scrubbed env, a pinned root-owned `PATH`, and a bounded, group-reaped
//! timeout — the same discipline as install-service's exec-preflight.
//!
//! ## Seams (so the security-critical logic is unit-testable off-box)
//! - [`evaluate_eligibility`] — PURE policy over an [`UpdateManifest`].
//! - [`decide_rollback`] — PURE crash-loop rollback state machine.
//! - staging/flip/rollback/prune take an injected `bin: &Path` and (for the smoke
//!   test) an injected runner, so they run against a temp dir with no real service.
//!
//! The genuinely untestable tail is the network fetch and the final re-exec/restart.

use std::io::Read;
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::diagnostic::HiggsError;
use crate::node::cli::{temp_name, TRUSTED_PATH};
use crate::update::{verify_artifact_sha256, UpdateManifest};

/// The acceleration variant THIS binary was compiled as, in the LOWERCASE
/// spelling the release manifest + `install.sh` use (`metal`/`cuda`/`cpu`) — NOT
/// `system.rs`'s human display casing (`Metal`/`CUDA`/`CPU`). The eligibility
/// check compares this against the manifest's `variant`, so the two conventions
/// must agree; a CUDA artifact on a CPU node fails to even load. `cfg!` folds to a
/// compile-time bool, so this is a `const`.
pub const CURRENT_VARIANT: &str = if cfg!(target_os = "macos") {
    "metal"
} else if cfg!(feature = "cuda") {
    "cuda"
} else {
    "cpu"
};

/// The maximum number of consecutive failed boots on a freshly-flipped binary
/// before the boot-guard rolls `current` back to the previous version. A trial
/// binary that crash-loops (each restart re-enters [`record_boot_attempt`], bumping the
/// counter) is rolled back once the budget is spent; a binary that starts cleanly
/// clears the trial via [`confirm_alive`] long before then.
pub const BOOT_FAIL_BUDGET: u32 = 3;

/// How long the post-stage `--version` smoke test may run before the staged binary
/// is declared unrunnable and the update aborts (mirrors install-service's
/// exec-preflight timeout). The whole process group is killed on expiry.
const SMOKE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

/// How long a freshly-flipped binary must STAY UP (without a hub admission) before the
/// trial is committed anyway ([`confirm_alive`]). This is the "or N s alive" arm of
/// the DESIGN §9 clear rule: a binary that starts cleanly but cannot reach the hub (a
/// hub OUTAGE, not a bad binary) must NOT be rolled back across restarts — surviving
/// [`ALIVE_GRACE`] proves it is not crash-looping regardless of hub reachability. The
/// boot-fail budget then only catches a binary that dies WITHIN this window.
pub const ALIVE_GRACE: std::time::Duration = std::time::Duration::from_secs(30);

/// The "stay-up" commit ([`confirm_alive`]) is best-effort under the update lock, so a
/// long-running apply/`--prune` holding the lock at the grace moment would make it a
/// silent no-op. The caller RETRIES on this interval, for [`CONFIRM_MAX_ATTEMPTS`], so a
/// healthy offline trial is always eventually committed once the lock frees.
pub const CONFIRM_RETRY_INTERVAL: std::time::Duration = std::time::Duration::from_secs(15);
pub const CONFIRM_MAX_ATTEMPTS: u32 = 40; // ~10 min of retries past a stuck lock

// ---------------------------------------------------------------------------
// Running-binary identity
// ---------------------------------------------------------------------------

/// What the RUNNING binary is — the identity a candidate manifest is judged
/// against. `version`/`target`/`variant` are all compile-time (`CARGO_PKG_VERSION`,
/// the build-script `HIGGS_BUILD_TARGET`, and [`CURRENT_VARIANT`]); nothing here is
/// read off disk, so a tampered install dir cannot lie about what is running.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuildIdentity {
    pub version: String,
    pub target: String,
    pub variant: String,
}

impl BuildIdentity {
    /// The identity of THIS binary.
    pub fn current() -> Self {
        Self {
            version: env!("CARGO_PKG_VERSION").to_string(),
            target: env!("HIGGS_BUILD_TARGET").to_string(),
            variant: CURRENT_VARIANT.to_string(),
        }
    }
}

/// The identity an update is judged against for the install at `bin`: the version
/// AND variant currently published as `current` (the install being REPLACED — NOT
/// necessarily the invoking process, which may be a STALE binary run directly out of
/// an old `v<ver>/` dir), plus THIS machine's build target. Reading `current`'s
/// version + `.variant` marker (not the invoker's compile-time values) is what stops
/// a stale binary from flipping `current` to an OLDER version without
/// `--allow-downgrade`, OR silently switching the install's variant (e.g. a stale CPU
/// binary replacing a CUDA install with a CPU artifact). Falls back to this build's
/// version/variant only when there is no `current`/marker (a pre-P3 or fresh install).
pub fn installed_identity(bin: &Path) -> BuildIdentity {
    let cur = current_target(bin);
    let ver = cur.as_deref().and_then(|v| v.strip_prefix('v'));
    let version = ver
        .map(str::to_string)
        .unwrap_or_else(|| env!("CARGO_PKG_VERSION").to_string());
    let variant = ver
        .and_then(|v| std::fs::read_to_string(variant_marker(bin, v)).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| CURRENT_VARIANT.to_string());
    BuildIdentity {
        version,
        target: env!("HIGGS_BUILD_TARGET").to_string(),
        variant,
    }
}

// ---------------------------------------------------------------------------
// Eligibility — PURE policy (the home for exhaustive unit tests)
// ---------------------------------------------------------------------------

/// Decides whether an AUTHENTICATED manifest is an eligible upgrade for the
/// `running` binary. Authenticity ([`crate::update`]) proves provenance only; this
/// enforces the three policy rules a courier-replayed but genuine manifest must
/// still pass:
/// - `target` must equal the running triple (HG086) — a different-arch artifact
///   would not execute;
/// - `variant` must equal [`CURRENT_VARIANT`] (HG086) — a CUDA build on a CPU box
///   fails to load;
/// - `version` must be strictly NEWER by semver precedence, unless `allow_downgrade`
///   (HG085) — a signed OLD release replayed by a courier is a downgrade, not an
///   upgrade.
///
/// Target/variant are checked BEFORE version so the operator sees the fundamental
/// "wrong build" refusal rather than a version quibble about an artifact that could
/// never run here. Both version strings are parsed as semver; a manifest version
/// that does not parse is an HG083-class manifest defect (it authenticated but is
/// unusable), while a non-semver RUNNING version is an internal invariant break
/// (HG087) — `CARGO_PKG_VERSION` is always valid semver.
///
/// An EQUAL version is refused ALWAYS, even with `allow_downgrade` — self-update
/// stages into `v<ver>/`, so a same-version "update" would rename the new binary
/// OVER the live `current` version dir BEFORE the smoke test (a failed smoke then
/// leaves `current` on a bad binary) and record `prev == to` (rollback could not
/// restore the original). Re-installing the same version is `install.sh`'s job, not
/// the self-updater's. `allow_downgrade` therefore only relaxes STRICTLY-older.
pub fn evaluate_eligibility(
    running: &BuildIdentity,
    m: &UpdateManifest,
    allow_downgrade: bool,
) -> Result<(), HiggsError> {
    if m.target != running.target {
        return Err(HiggsError::UpdateTargetMismatch {
            field: "target".into(),
            manifest: m.target.clone(),
            running: running.target.clone(),
        });
    }
    if m.variant != running.variant {
        return Err(HiggsError::UpdateTargetMismatch {
            field: "variant".into(),
            manifest: m.variant.clone(),
            running: running.variant.clone(),
        });
    }
    // A version that differs from the installed one ONLY in ASCII case is refused: on a
    // case-INSENSITIVE filesystem (default macOS) `v1.0.0-A` and `v1.0.0-a` name the SAME
    // dir, so "upgrading" between them would overwrite the live/rollback version dir even
    // though semver treats the prereleases as distinct. (On a case-sensitive FS this is
    // still a nonsensical version bump, so refusing is safe either way.)
    if m.version != running.version && m.version.eq_ignore_ascii_case(&running.version) {
        return Err(HiggsError::UpdateNotNewer {
            from: running.version.clone(),
            to: m.version.clone(),
        });
    }
    let to = semver::Version::parse(&m.version).map_err(|e| HiggsError::UpdateManifestInvalid {
        detail: format!("manifest version {:?} is not valid semver ({e})", m.version),
    })?;
    let from =
        semver::Version::parse(&running.version).map_err(|e| HiggsError::UpdateApplyFailed {
            detail: format!(
                "running version {:?} is not valid semver ({e}) — internal invariant break",
                running.version
            ),
        })?;
    // `cmp_precedence`, NOT `cmp`: semver PRECEDENCE ignores build metadata (`1.2.3+aaa`
    // and `1.2.3+zzz` are EQUAL), while `Ord` for `Version` totally-orders it. Using the
    // total order would accept `1.2.3+zzz` over `1.2.3+aaa` as an "upgrade" though they
    // rank equal — so an equal-precedence manifest would slip through as newer.
    let refuse = match to.cmp_precedence(&from) {
        std::cmp::Ordering::Greater => false,
        // Equal precedence: never a valid self-update (see the fn doc — it would overwrite
        // the live version dir before smoke and destroy the rollback copy). Always refuse.
        std::cmp::Ordering::Equal => true,
        // Strictly older: a downgrade, allowed only behind the explicit flag.
        std::cmp::Ordering::Less => !allow_downgrade,
    };
    if refuse {
        return Err(HiggsError::UpdateNotNewer {
            from: running.version.clone(),
            to: m.version.clone(),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Layout helpers — pure path builders over an injected bin dir
// ---------------------------------------------------------------------------

/// `<prefix>/bin` — the directory `install.sh` publishes versions and `current` into.
pub fn bin_dir(prefix: &Path) -> PathBuf {
    prefix.join("bin")
}
fn current_link(bin: &Path) -> PathBuf {
    bin.join("current")
}
/// A version's directory name, `v<ver>`. `ver` is only ever a semver string that
/// already parsed (semver's grammar has no `/` or empty `..` component), so this is
/// path-safe by construction — but callers still validate via [`evaluate_eligibility`]
/// before this string reaches a path.
fn version_name(ver: &str) -> String {
    format!("v{ver}")
}
/// Strip a `v<semver>` dir-name prefix down to the PLAIN semver (`v1.2.3` → `1.2.3`) — the form
/// `CARGO_PKG_VERSION` and the HELLO `software_version` use, and what `reportable_update_failure`
/// reconciles an `UpdateFailed.from` against. A name already without the prefix is returned as-is.
fn strip_v(name: &str) -> &str {
    name.strip_prefix('v').unwrap_or(name)
}

/// Whether `name` is a real version-dir name: `v` followed by a VALID semver (not
/// merely "starts with v"). Used so `--prune` only ever selects genuine `v<semver>`
/// dirs — never an unrelated `v…`-prefixed dir (e.g. `valuable/`) that a mis-derived
/// bin path might sit beside.
pub fn is_version_dir_name(name: &str) -> bool {
    name.strip_prefix('v')
        .is_some_and(|v| semver::Version::parse(v).is_ok())
}
/// The `<bin>/current` symlink must exist for `bin` to be a higgs-managed install —
/// the guard that stops a maintenance op (`--rollback`/`--prune`) from acting on an
/// arbitrary directory that merely happens to contain `v…` entries.
fn is_managed_install(bin: &Path) -> bool {
    std::fs::symlink_metadata(current_link(bin)).is_ok_and(|m| m.file_type().is_symlink())
}
fn version_dir(bin: &Path, ver: &str) -> PathBuf {
    bin.join(version_name(ver))
}
fn variant_marker(bin: &Path, ver: &str) -> PathBuf {
    version_dir(bin, ver).join(".variant")
}
fn trial_marker_path(bin: &Path) -> PathBuf {
    bin.join(".update-trial")
}
fn boot_fail_path(bin: &Path) -> PathBuf {
    bin.join(".update-bootfails")
}
fn failed_target_path(bin: &Path) -> PathBuf {
    bin.join(".update-failed")
}

/// The `v<ver>` names that FAILED their boot trial and were rolled back (one per line).
fn read_failed_targets(bin: &Path) -> Vec<String> {
    std::fs::read_to_string(failed_target_path(bin))
        .ok()
        .map(|s| {
            s.lines()
                .map(|l| l.trim().to_string())
                .filter(|l| !l.is_empty())
                .collect()
        })
        .unwrap_or_default()
}

/// Record `vname` as a target that FAILED its boot trial and was rolled back, so a hub RE-PUSHING
/// the same crash-looping release is REFUSED before re-staging it (see [`is_failed_target`]) —
/// else a lost `"accepted"` reply, or a fleet that keeps pushing "latest", would loop
/// apply→crash→rollback→apply forever instead of resting on the known-good previous version.
/// Append-once + deduped. BEST-EFFORT: a write failure here must never fail the (critical)
/// rollback it rides on. Cleared by a deliberate operator re-install (`install.sh` wipes it), so
/// a genuinely-fixed rebuild of the same version can still be re-applied.
fn record_failed_target(bin: &Path, vname: &str) {
    let path = failed_target_path(bin);
    // Clear a NON-REGULAR node (a directory/fifo/symlink someone left at the poison path):
    // `write_atomic`'s rename cannot replace a directory, so without this the poison could NEVER
    // persist and a crash-looping version would loop UNBOUNDEDLY (`read_failed_targets` reads such
    // a node as empty). `chmod 0700` first so a non-writable directory's children can still be
    // unlinked, then remove it. Best-effort like the rest of this fn.
    //
    // RESIDUAL (documented, out of this trust model): a node that stays UN-REMOVABLE even after the
    // chmod — a root-owned or filesystem-immutable (`chattr +i`) node, or a deeply-nested
    // non-writable tree — needs privileges the operator lacks, i.e. ROOT-level sabotage of the
    // operator's OWN `bin` dir. That is self-inflicted and NOT attacker-reachable (a peer/hub
    // cannot write to the 0755 operator-owned `bin`). The DEFINITIVE bound on a hub that keeps
    // re-pushing a failing release is the HUB-side lifecycle (§9 `update_failed` over HELLO — the
    // hub stops re-pushing a version whose HELLO never advances); this node-side poison is
    // defense-in-depth for the common + attacker-reachable cases.
    if std::fs::symlink_metadata(&path).is_ok_and(|m| !m.file_type().is_file()) {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o700));
        let _ = std::fs::remove_dir_all(&path).or_else(|_| std::fs::remove_file(&path));
    }
    let mut set = read_failed_targets(bin);
    if !set.iter().any(|v| v == vname) {
        set.push(vname.to_string());
        if write_atomic(&path, set.join("\n").as_bytes()).is_ok() {
            let _ = fsync_path(bin);
        }
    }
}

/// True iff `vname` previously failed its boot trial (recorded by [`record_failed_target`]).
fn is_failed_target(bin: &Path, vname: &str) -> bool {
    read_failed_targets(bin).iter().any(|v| v == vname)
}

/// The `.update-lastfail` marker: the DETAILS of the last self-update failure (JSON of
/// [`UpdateFailed`]), persisted so the node can report it to the hub on its NEXT HELLO — a
/// boot-guard rollback happens at boot BEFORE any hub connection, so the reconnect is the first
/// chance to explain WHY the update did not take. Distinct from the `.update-failed` poison LIST
/// (which prevents re-applying a crash-looping version): this holds the last failure's `{from, to,
/// reason}` for reporting. A successful apply CLEARS it; otherwise it is overwritten by the next
/// failure, and once it goes stale (the node advances onto a different build)
/// [`reportable_update_failure`] simply FILTERS it out of the report (it is not deleted here — that
/// would race a detached apply).
fn update_lastfail_path(bin: &Path) -> PathBuf {
    bin.join(".update-lastfail")
}

/// Persist the LAST update-failure details for the next HELLO ([`reportable_update_failure`]). The
/// `reason` is sanitized (an HG code + phase, never free-form) since it crosses the wire and is
/// displayed. BEST-EFFORT: a write failure here must never fail the rollback/apply path it rides
/// on — the hub still infers failure from `software_version` never advancing.
///
/// On a rare write failure (ENOSPC/EIO — plausible when the apply itself failed on a full disk), a
/// previously-recorded marker is DELIBERATELY LEFT in place rather than invalidated: it is the last
/// SUCCESSFULLY-recorded REAL failure (often the same target being retried, in which case it is the
/// correct reason), and retaining a real reason is more useful to the operator than deleting it to
/// report nothing — the newer failure is still signalled by `software_version` not advancing. This
/// is the accepted best-effort behaviour, not a fabricated report.
pub(crate) fn record_update_failure(bin: &Path, from: &str, to: &str, reason: &str) {
    let path = update_lastfail_path(bin);
    // Clear a non-regular node first (same reasoning as `record_failed_target`), so the write
    // reliably lands.
    if std::fs::symlink_metadata(&path).is_ok_and(|m| !m.file_type().is_file()) {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o700));
        let _ = std::fs::remove_dir_all(&path).or_else(|_| std::fs::remove_file(&path));
    }
    let failed = crate::remote::UpdateFailed {
        from: crate::remote::sanitize_version(from),
        to: crate::remote::sanitize_version(to),
        reason: crate::remote::sanitize_display(reason),
    };
    if let Ok(bytes) = serde_json::to_vec(&failed) {
        if write_atomic(&path, &bytes).is_ok() {
            let _ = fsync_path(bin);
        }
    }
}

/// TEST-ONLY raw read of the last-failure marker as an `Option` (the production HELLO report goes
/// through [`reportable_update_failure`], which distinguishes a conclusive absence from an
/// inconclusive read). Tests use this to assert the RAW on-disk marker state (present/absent),
/// unfiltered by the running-build version check. A malformed/absent/unreadable marker is `None`.
#[cfg(test)]
pub(crate) fn peek_update_failure(bin: &Path) -> Option<crate::remote::UpdateFailed> {
    let raw = std::fs::read(update_lastfail_path(bin)).ok()?;
    serde_json::from_slice::<crate::remote::UpdateFailed>(&raw).ok()
}

/// Clear the last update-failure marker — called on a SUCCESSFUL apply (a newer attempt supersedes
/// an older unreported failure). Best-effort + fsync (durable): a failed clear just re-reports on
/// the next reconnect.
pub(crate) fn clear_update_failure(bin: &Path) {
    let _ = remove_if_present(&update_lastfail_path(bin));
    let _ = fsync_path(bin);
}

/// The failure the node should REPORT in its next HELLO. The node re-reports the marker on EVERY
/// HELLO (report-until-resolved) — it does NOT clear on a HELLO reply, because a valid reply does
/// not prove the hub stored the failure (a stale-generation admission stores nothing; a one-shot
/// pairing hub discards it), and clearing then would silently lose the only copy.
///
/// This is a PURE READ: it reports a marker only when its `from` equals the RUNNING build
/// (`CARGO_PKG_VERSION`, the same value the HELLO sends as `software_version`), and returns `None`
/// otherwise. A marker whose `from` differs is STALE — the process restarted onto a different build
/// (a later success, or a manual reinstall) — so it is FILTERED (never reported) but deliberately
/// NOT deleted here: a read-then-delete would be a non-atomic read-modify-write that could race a
/// detached apply atomically recording a FRESH failure between the read and the delete. The stale
/// marker lingers inert (always filtered) and self-corrects — any later apply OVERWRITES it (on
/// failure) or CLEARS it (on success), and the hub drops its stored failure via the admission
/// version-gate once this node reports `None` while advancing off the failed build. `None` also
/// covers "no marker" (never failed / already resolved) — which is exactly what lets the hub
/// distinguish "resolved" from "still failing" (a repeated `Some`).
///
/// ACCEPTED RESIDUAL (version heuristic vs. per-attempt ordering). Keying the report to the running
/// build makes `from` consistent with `software_version` and suppresses a stale marker from a build
/// the node left — but a version string is NOT a per-attempt generation. A failure RECORDED against
/// a build the node then LEAVES is therefore filtered out: e.g. if the hub pushes CONCURRENT updates
/// and B succeeds while a later C fails in the window before the B-restart, C's marker `{from:A}` is
/// filtered after restarting onto B, so its reason is lost (the node is healthy on B; re-pushing C
/// re-surfaces it). Reporting it instead would violate the `from == software_version` invariant —
/// the two goals genuinely conflict, and only a monotonic attempt generation (a larger feature)
/// satisfies both. This degrades safely and is only reachable via CONCURRENT per-node pushes, which
/// the hub courier avoids by SERIALIZING pushes per node (push, observe the HELLO outcome —
/// `software_version` / `update_failed` — then push the next).
///
/// Returns `(report, conclusive)`. `conclusive` is `false` ONLY when the marker read was
/// INCONCLUSIVE (a transient read error, or — improbable under atomic writes — corrupt JSON): the
/// node then genuinely cannot say "no failure", so the caller must NOT advertise `update_reporting`
/// for that admission (an authoritative `None` would erase a valid stored failure the marker still
/// holds). A file that is confidently ABSENT (NotFound) or a STALE marker (different build) is
/// `conclusive = true` — there is no current failure to report.
pub(crate) fn reportable_update_failure(bin: &Path) -> (Option<crate::remote::UpdateFailed>, bool) {
    reportable_update_failure_for(bin, env!("CARGO_PKG_VERSION"))
}

/// [`reportable_update_failure`] against an EXPLICIT running version — the injected seam that lets
/// tests drive versions the compiled `CARGO_PKG_VERSION` can't take (e.g. one past the sanitizer's
/// 64-char cap). The marker's `from` was stored THROUGH [`crate::remote::sanitize_version`] (it
/// truncates at 64), so the comparison sanitizes the running version the SAME way — comparing the
/// raw version instead would classify every fresh marker stale whenever the build version doesn't
/// survive sanitization verbatim (e.g. longer than the cap), silently dropping real reports.
fn reportable_update_failure_for(
    bin: &Path,
    running_version: &str,
) -> (Option<crate::remote::UpdateFailed>, bool) {
    let running = crate::remote::sanitize_version(running_version);
    match std::fs::read(update_lastfail_path(bin)) {
        // No marker → confidently no failure.
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => (None, true),
        // A transient read error (EIO / permission) → inconclusive: do not claim an authoritative
        // absence; the marker may still hold a valid failure.
        Err(_) => (None, false),
        Ok(raw) => match serde_json::from_slice::<crate::remote::UpdateFailed>(&raw) {
            // A live failure for THIS build → report it.
            Ok(f) if f.from == running => (Some(f), true),
            // A stale marker (a build we left) → confidently no CURRENT failure (filtered, not
            // deleted — see the residual note above).
            Ok(_) => (None, true),
            // Corrupt JSON (should not occur with atomic temp-write+rename) → inconclusive.
            Err(_) => (None, false),
        },
    }
}

/// The lock lives inside a PRIVATE `0700` directory (not directly in `bin`) so that no
/// peer can open it and hold the update lock — see [`open_private_lock_dir`].
fn lock_dir(bin: &Path) -> PathBuf {
    bin.join(".update-lock.d")
}
fn lock_path(bin: &Path) -> PathBuf {
    lock_dir(bin).join("lock")
}

/// The `v<ver>` name `current` currently points at, if any. Reads the symlink
/// TARGET verbatim (`install.sh` writes a RELATIVE `v<ver>` target), never following
/// it — so a dangling `current` still reports its intended name for rollback bookkeeping.
pub fn current_target(bin: &Path) -> Option<String> {
    std::fs::read_link(current_link(bin))
        .ok()
        .and_then(|t| t.file_name().map(|n| n.to_string_lossy().into_owned()))
        .filter(|n| n.starts_with('v'))
}

// ---------------------------------------------------------------------------
// Trial marker — records an unconfirmed flip for the boot-guard
// ---------------------------------------------------------------------------

/// Persisted the instant `current` is flipped to a freshly-staged version and
/// cleared only once that version proves it can run ([`confirm_alive`]). While it
/// exists, the boot-guard treats the current binary as ON TRIAL: repeated boot failures
/// roll `current` back to `prev` (via [`boot_rollback_if_spent`]).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrialMarker {
    /// The `v<ver>` name `current` was flipped TO.
    pub to: String,
    /// The `v<ver>` name `current` pointed at BEFORE the flip — the rollback target.
    /// `None` when there was no prior `current` (a first install can't roll back).
    pub prev: Option<String>,
}

fn read_trial(bin: &Path) -> Option<TrialMarker> {
    let raw = std::fs::read(trial_marker_path(bin)).ok()?;
    serde_json::from_slice(&raw).ok()
}

fn read_boot_fails(bin: &Path) -> u32 {
    std::fs::read_to_string(boot_fail_path(bin))
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Boot guard — crash-loop rollback split into three version-aware steps
// ---------------------------------------------------------------------------
//
// The guard is three distinct hooks. The two that MUTATE trial state (record + confirm)
// are keyed on the RUNNING binary's version, so an OLD daemon that happens to be running
// while a NEWER version is on trial never touches the new binary's trial state; step 1's
// rollback is NOT version-gated (any boot may restore a good binary) but fires only once
// the version-gated counter has spent the budget:
//
//   1. `boot_rollback_if_spent` — runs at EVERY daemon boot (before serving): if a
//      trial exists whose failure budget is already spent, repoint `current` back to
//      `prev` and clear the trial. Not version-gated — any boot may restore the good
//      binary. Read-only otherwise.
//   2. `record_boot_attempt(running)` — runs EARLY at daemon boot, BEFORE the risky init
//      (bind/runtime/serve) and once past the `--list`/no-hub short-circuits: if the trial
//      is for the running version, increment its failure counter, so a crash DURING that
//      init (an update that passed the `--version` smoke test but dies before admitting)
//      accrues toward the budget. A clean boot / graceful stop clears it via step 3.
//   3. `confirm_alive(running)` — runs on the first successful hub admission: if the
//      trial is for the running version, clear it (commit the update).
//
// Splitting record (step 2) from the rollback check (step 1) is what keeps a healthy
// binary safe from an operator's repeated `--list` one-shots: only a real serve boot
// records a failure, and only the trialed version's own failures count.

/// Whether a spent trial should roll back, given the marker + failure count. PURE.
/// A trial with a recorded `prev` whose failures have reached `budget` rolls back;
/// anything else (no trial, under budget, or no rollback target) does not.
pub fn decide_rollback(trial: Option<&TrialMarker>, fails: u32, budget: u32) -> Option<String> {
    match trial {
        Some(t) if fails >= budget => t.prev.clone(),
        _ => None,
    }
}

/// True iff a pending trial names the RUNNING version (`trial.to == "v<running>"`) —
/// i.e. THIS process is the binary on trial, so its boot outcome is what the trial
/// tracks. An old daemon running alongside a newer on-trial binary returns false and
/// leaves the trial untouched.
fn trial_is_for(trial: &TrialMarker, running_version: &str) -> bool {
    trial.to == version_name(running_version)
}

/// Whether a trial for `running_version` is still pending (uncommitted). Lets the
/// grace-commit caller stop retrying once the trial is confirmed/cleared.
pub fn is_trial_pending_for(bin: &Path, running_version: &str) -> bool {
    read_trial(bin).is_some_and(|t| trial_is_for(&t, running_version))
}

// Every boot-guard hook takes the same non-blocking `UpdateLock` an apply/rollback/
// prune holds, so a boot never read-modify-writes markers or flips `current` while an
// update is mid-flight (which could clobber a newer flip or delete a fresh trial). If
// the lock is held by an in-progress apply, the hook is a NO-OP — the apply is setting
// consistent state, and the next boot re-checks.

/// Run a boot-guard mutation `body` under the update lock, distinguishing the two
/// non-acquired cases so a HARD lock failure never silently disables crash-loop
/// recovery:
/// - the lock is held by an in-flight apply (`Contended`) → return `skip` (the apply is
///   installing consistent state; the next boot re-checks);
/// - the lock SUBSYSTEM itself failed (`Failed` — a planted `.update.lock` symlink
///   refused by `O_NOFOLLOW`, a non-regular node, `EACCES`, or an FS whose `flock` is
///   broken/unsupported) → run `body` LOCK-FREE, passing `locked = false`. This is
///   race-free for every REACHABLE input: the apply/rollback/prune path takes the SAME
///   lock via [`UpdateLock::acquire`], which returns `Err` on the identical failure, so
///   no apply can proceed past its own acquire — under a broken `flock` it aborts, and a
///   swapped lock pathname (the only way an apply could HOLD the lock while a boot sees
///   `Failed`) needs write access to the operator-owned `bin` that
///   `refuse_unsafe_operator_bin_tree` denies. Silently skipping (the old `let Ok(_lock)
///   = acquire(..) else { return skip }`) would instead strand a crash-looping trial with
///   its rollback permanently disabled. `body` still receives `locked` so it can gate any
///   DESTRUCTIVE cleanup (e.g. deleting a marker that merely LOOKS stale) behind a real
///   held lock — defense-in-depth against the swapped-lock state we argue is unreachable.
fn with_boot_lock<T>(bin: &Path, hook: &str, skip: T, body: impl FnOnce(bool) -> T) -> T {
    match UpdateLock::try_acquire(bin) {
        LockAttempt::Held(_lock) => body(true),
        LockAttempt::Contended => skip,
        LockAttempt::Failed(e) => {
            tracing::error!(
                error = %e,
                hook,
                "higgs self-update: boot-guard update lock unavailable (non-contention) — \
                 running lock-free to preserve crash-loop recovery"
            );
            body(false)
        }
    }
}

/// Step 1 — at daemon boot, roll `current` back to `prev` if the trial spent its
/// budget, then clear the trial state. Returns `Ok(Some(prev))` on rollback (the
/// caller exits so the service manager re-execs the now-`current` OLD binary),
/// `Ok(None)` to proceed. Not version-gated: any boot may restore the good binary.
pub fn boot_rollback_if_spent(bin: &Path) -> Result<Option<String>, HiggsError> {
    // Held by an apply → skip; lock subsystem broken → run lock-free (no apply can be
    // concurrent, since its `acquire` fails closed the same way).
    with_boot_lock(bin, "boot_rollback_if_spent", Ok(None), |locked| {
        boot_rollback_body(bin, locked)
    })
}

/// The read-modify-write body of [`boot_rollback_if_spent`], run either under the lock
/// (`locked = true`) or — on a hard lock failure — lock-free (`locked = false`); see
/// [`with_boot_lock`].
fn boot_rollback_body(bin: &Path, locked: bool) -> Result<Option<String>, HiggsError> {
    let trial = read_trial(bin);
    // A trial whose `to` no longer matches `current` is STALE — superseded by a manual
    // `install.sh` (or `--rollback`) that repointed `current` without clearing the old
    // marker. It must NOT drive a rollback (that would undo the operator's repair, e.g.
    // flip a freshly-installed v3 back to v1). (The apply path applies the same
    // `to != current` staleness rule.)
    if let Some(t) = &trial {
        if Some(t.to.as_str()) != current_target(bin).as_deref() {
            // Clearing the stale marker is a non-safety-critical CLEANUP and a DESTRUCTIVE
            // write — gate it behind a genuinely HELD lock. Without the lock we cannot rule
            // out that this `to != current` marker is a concurrent apply's FRESH marker
            // written microseconds before its `current` flip (stage_and_flip writes the
            // marker, THEN flips); deleting it there would strip the new binary's rollback
            // state. That interleaving needs the lock pathname swapped mid-apply, which
            // needs write access to the operator-owned bin that `refuse_unsafe_operator_
            // bin_tree` denies — but we defend in depth: lock-free we LEAVE the marker (it
            // drives no rollback — we return None) and a later HELD boot cleans it up.
            if locked {
                remove_if_present(&trial_marker_path(bin))?;
                remove_if_present(&boot_fail_path(bin))?;
                let _ = fsync_path(bin);
            }
            return Ok(None);
        }
    }
    let fails = read_boot_fails(bin);
    match decide_rollback(trial.as_ref(), fails, BOOT_FAIL_BUDGET) {
        Some(prev) => {
            let failed_to = trial.as_ref().map(|t| t.to.clone());
            let rolled = perform_trial_rollback(
                bin,
                &prev,
                failed_to.as_deref(),
                "crash-looped on boot — rolled back",
            )?;
            if rolled {
                tracing::error!(
                    rolled_back_to = %prev,
                    "higgs self-update: trial binary failed to boot {BOOT_FAIL_BUDGET}x — rolled current back"
                );
                Ok(Some(prev))
            } else {
                Ok(None)
            }
        }
        None => Ok(None),
    }
}

/// Force a rollback of the pending trial for `running_version` REGARDLESS of the boot
/// budget — used when the boot-fail counter itself CANNOT be persisted (a full/`ENOSPC`
/// filesystem), which would otherwise leave a crash-looping trial un-counted and never
/// rolled back. A rollback is safe (restores the known-good previous binary) and FREES
/// disk space (removes the trial + counter). No-op (returns `Ok(None)`) if an apply
/// holds the lock, the trial is stale (`to != current`) or not for this version, or the
/// rollback target is missing. Returns `Ok(Some(prev))` when it rolled back.
pub fn force_rollback_trial(
    bin: &Path,
    running_version: &str,
) -> Result<Option<String>, HiggsError> {
    // Rolling a spent trial back is safety-critical — run it lock-free on a hard lock
    // failure too (the flip + marker-clear it does IS the recovery, not a clobber of a
    // concurrent apply, which cannot exist here — see `with_boot_lock`).
    with_boot_lock(bin, "force_rollback_trial", Ok(None), |_locked| {
        force_rollback_trial_body(bin, running_version)
    })
}

fn force_rollback_trial_body(
    bin: &Path,
    running_version: &str,
) -> Result<Option<String>, HiggsError> {
    let Some(trial) = read_trial(bin) else {
        return Ok(None);
    };
    // Only OUR live trial (to == current == this version).
    if !trial_is_for(&trial, running_version)
        || Some(trial.to.as_str()) != current_target(bin).as_deref()
    {
        return Ok(None);
    }
    let failed_to = trial.to.clone();
    let Some(prev) = trial.prev else {
        return Ok(None);
    };
    // The ACCURATE cause: this trial did not crash-loop — its boot health could not be TRACKED
    // (the counter write failed, e.g. disk full), so the safe move was rolling back. Reporting
    // "crash-looped" here would send the operator chasing a nonexistent crash.
    if perform_trial_rollback(
        bin,
        &prev,
        Some(&failed_to),
        "could not track the update's boot health (disk full?) — rolled back untested",
    )? {
        tracing::error!(
            rolled_back_to = %prev,
            "higgs self-update: cannot persist the boot-fail counter (disk full?) — force-rolled current back"
        );
        Ok(Some(prev))
    } else {
        Ok(None)
    }
}

/// Repoint `current` to `prev` and clear the trial state — the shared rollback body for
/// [`boot_rollback_if_spent`] and [`force_rollback_trial`]. Verifies `prev` is an
/// installed, runnable version FIRST (a pruned/removed target would otherwise leave
/// `current` DANGLING). Returns `Ok(true)` on rollback, `Ok(false)` when `prev` is not
/// installed (caller leaves `current`/the trial untouched). The caller holds the lock.
/// `reason` is the CALLER's actual rollback cause, recorded for the node's next HELLO —
/// the crash-loop and the couldn't-track-boot-health paths roll back for different
/// reasons, and explaining WHY is the report's whole purpose.
fn perform_trial_rollback(
    bin: &Path,
    prev: &str,
    failed_to: Option<&str>,
    reason: &str,
) -> Result<bool, HiggsError> {
    if !is_installed_version(bin, prev) {
        tracing::error!(
            missing_rollback_target = %prev,
            "higgs self-update: rollback target {prev} is not installed — cannot auto-recover; \
             reinstall via install.sh"
        );
        return Ok(false);
    }
    // Free the boot-fail counter's inode BEFORE the flip: `flip_current_to` creates a
    // temporary symlink (one inode), so on an INODE-exhausted filesystem — the very
    // condition the disk-full recovery must survive — removing the counter first gives
    // the flip an inode to use. The TRIAL marker is removed AFTER the flip so it survives
    // a flip failure (leaving the operator's `--rollback` still able to recover).
    remove_if_present(&boot_fail_path(bin))?;
    flip_current_to(bin, prev)?;
    // POISON the rolled-back version + RECORD the failure IMMEDIATELY after the successful flip —
    // BEFORE the fallible trial-marker cleanup below. A transient failure to unlink `.update-trial`
    // (EIO / a file-specific ACL) must NOT skip recording: the node HAS rolled back, so the next
    // healthy build still needs a reason to report and the crash-looping version must still be
    // poisoned. Both are best-effort and durable (each fsyncs).
    if let Some(failed) = failed_to {
        record_failed_target(bin, failed);
        // The failure DETAILS for the node's NEXT HELLO (this boot has no hub connection yet — the
        // rollback ran in the boot-guard preflight). `from` is the version that will be RUNNING once
        // the service restarts onto the rolled-back `prev` — recorded in PLAIN semver form (the `v`
        // dir-name prefix stripped) so it matches that binary's `CARGO_PKG_VERSION`, which is what
        // `reportable_update_failure` reconciles against and what the HELLO reports as
        // `software_version`. A `v`-prefixed `from` would never match, dropping the report.
        record_update_failure(bin, strip_v(prev), strip_v(failed), reason);
    }
    remove_if_present(&trial_marker_path(bin))?;
    // Flush the cleared trial marker so a crash right after cannot resurrect the spent trial
    // and roll back the (now restored, healthy) previous binary again.
    let _ = fsync_path(bin);
    Ok(true)
}

/// Whether `vname` names a rollback-safe installed version — the check shared by the
/// boot-guard and the manual `--rollback` before repointing `current` at it. Strict, so
/// a corrupt/hand-edited trial marker can't make `current` dangle or point outside `bin`:
/// - `vname` must be a clean `v<semver>` name (no `..`, absolute, or path separators —
///   `is_version_dir_name` parses the semver, whose grammar has none), so `bin.join`
///   stays a direct child of `bin`;
/// - the version dir must be a REAL directory (not a symlink — `symlink_metadata`, which
///   does not follow, so a planted `v… -> /elsewhere` link is rejected);
/// - `<dir>/higgs` must be a REGULAR, EXECUTABLE, non-symlink file (`require_regular_exec`),
///   so a rollback never repoints `current` at a non-runnable or redirected binary.
fn is_installed_version(bin: &Path, vname: &str) -> bool {
    if !is_version_dir_name(vname) {
        return false;
    }
    let dir = bin.join(vname);
    let dir_is_real = std::fs::symlink_metadata(&dir)
        .map(|m| m.file_type().is_dir())
        .unwrap_or(false);
    dir_is_real && require_regular_exec(&dir.join("higgs")).is_ok()
}

/// Step 2 — record a failed-until-proven boot ATTEMPT of the trialed binary. Only
/// increments when a trial for `running_version` is pending (this process IS the
/// binary on trial); with no trial it clears any stale counter; with a trial for a
/// DIFFERENT version it leaves the counter alone (that budget belongs to the other
/// binary). Call ONCE, EARLY at daemon boot — BEFORE the risky init (bind / runtime /
/// serve) — so an update that dies during that init (having passed the `--version`
/// smoke test) still accrues; a healthy boot or a graceful shutdown then clears it via
/// [`confirm_alive`]. A held lock (apply in progress) is a no-op.
pub fn record_boot_attempt(bin: &Path, running_version: &str) -> Result<(), HiggsError> {
    with_boot_lock(
        bin,
        "record_boot_attempt",
        Ok(()),
        |_locked| match read_trial(bin) {
            Some(t) if trial_is_for(&t, running_version) => {
                let next = read_boot_fails(bin).saturating_add(1);
                write_atomic(&boot_fail_path(bin), next.to_string().as_bytes())
            }
            Some(_) => Ok(()),
            None => remove_if_present(&boot_fail_path(bin)),
        },
    )
}

/// Step 3 — commit the update once the freshly-flipped binary has proven it can run
/// (a successful hub admission, or [`ALIVE_GRACE`] of uptime). Clears the trial marker
/// and the counter ONLY when the pending trial is for `running_version` — so an OLD
/// daemon admitted while a NEWER version sits on trial (e.g. the service has not yet
/// restarted onto it) does NOT clear the new binary's trial. Idempotent; a no-op when
/// nothing is pending or when an apply holds the lock.
pub fn confirm_alive(bin: &Path, running_version: &str) -> Result<(), HiggsError> {
    // Committing lock-free on a hard lock failure is acceptable: clearing the trial for
    // the RUNNING version only advances the state toward "healthy" (it can never cause a
    // wrong rollback). The residual — under a persistently-broken advisory lock (e.g.
    // NFS) with MULTIPLE concurrent daemon instances, this commit can race a sibling's
    // lock-free `boot_rollback_body` and produce a spurious rollback toward the
    // known-good previous binary — is strictly safer than the pre-fix "loops forever" and
    // needs a degenerate deployment; documented rather than gated (gating it would break
    // the common broken-flock single-instance case, where a healthy update must commit).
    with_boot_lock(bin, "confirm_alive", Ok(()), |_locked| {
        match read_trial(bin) {
            Some(t) if trial_is_for(&t, running_version) => {
                remove_if_present(&trial_marker_path(bin))?;
                remove_if_present(&boot_fail_path(bin))?;
                // Flush the unlinks so a crash right after committing does not let the
                // trial + spent counter reappear and roll back this (healthy) binary.
                let _ = fsync_path(bin);
                Ok(())
            }
            _ => Ok(()),
        }
    })
}

// ---------------------------------------------------------------------------
// Update source — the network/local fetch seam
// ---------------------------------------------------------------------------

/// Where the update bytes come from. The manifest + its `.minisig` are fetched
/// first and VERIFIED before the (large) artifact is fetched, so a courier can never
/// make the node download an unauthenticated tarball. Kept a trait so tests inject
/// in-memory bytes and the CLI can offer both a local-file source and a REST source.
pub trait UpdateSource {
    /// `(manifest_bytes, signature_text)` — the JSON manifest and the full text of
    /// its `.minisig`.
    fn manifest(&self) -> Result<(Vec<u8>, String), HiggsError>;
    /// The artifact tarball bytes for `file` (the manifest's `file` field, already
    /// authenticated). The impl decides how to locate it (a sibling file, a release
    /// asset URL); the caller re-hashes the bytes against the manifest regardless.
    fn artifact(&self, file: &str) -> Result<Vec<u8>, HiggsError>;
}

/// An update staged from LOCAL files the operator already fetched (scp, `curl`, a
/// shared mount) — the testable, network-free path, mirroring `install.sh --tarball`.
pub struct LocalSource {
    pub manifest: PathBuf,
    pub manifest_sig: PathBuf,
    pub tarball: PathBuf,
}

impl UpdateSource for LocalSource {
    fn manifest(&self) -> Result<(Vec<u8>, String), HiggsError> {
        let m = std::fs::read(&self.manifest).map_err(|e| HiggsError::UpdateApplyFailed {
            detail: format!("cannot read manifest {}: {e}", self.manifest.display()),
        })?;
        let sig = std::fs::read_to_string(&self.manifest_sig).map_err(|e| {
            HiggsError::UpdateApplyFailed {
                detail: format!(
                    "cannot read manifest signature {}: {e}",
                    self.manifest_sig.display()
                ),
            }
        })?;
        Ok((m, sig))
    }

    fn artifact(&self, _file: &str) -> Result<Vec<u8>, HiggsError> {
        // The tarball path is given explicitly; `file` is cross-checked against the
        // manifest by the caller (via the sha256), so we don't re-derive the path
        // from an untrusted-until-verified field.
        std::fs::read(&self.tarball).map_err(|e| HiggsError::UpdateApplyFailed {
            detail: format!("cannot read tarball {}: {e}", self.tarball.display()),
        })
    }
}

// ---------------------------------------------------------------------------
// HTTP update source (§9 P4) — fetch the manifest+sig+artifact over the network
// ---------------------------------------------------------------------------

// The manifest is fetched (and its `.minisig` alongside) BEFORE the artifact, and
// `verify_and_check` authenticates the manifest signature + eligibility BEFORE
// `HttpSource::artifact` is ever called — so an unauthenticated tarball is never
// downloaded, and the fetched bytes are always re-verified against the pinned key
// (HG081-084). The network is UNTRUSTED here; the signature is the only trust anchor.

/// A manifest JSON is tiny — cap the download so a hostile/broken server cannot stream
/// unbounded bytes at us before we even parse it.
const MAX_MANIFEST_BYTES: u64 = 64 * 1024;
/// A minisign signature is a couple of short base64 lines.
const MAX_SIG_BYTES: u64 = 4 * 1024;
/// Ceiling for the release tarball, held in memory before hashing. A real single-binary
/// higgs tarball is tens of MiB; 256 MiB is ~10× headroom yet bounds the transient
/// allocation so a malicious host cannot OOM a small (2–4 GiB Pi/Jetson) node by serving
/// a huge body before the `sha256` check (HG084) rejects it. RESIDUAL: the body is still
/// buffered in RAM up to this cap — a fully stream-to-disk fetch (hashing the file, not a
/// `Vec`) is a larger refactor of the shared verify/apply pipeline, deferred.
const MAX_ARTIFACT_BYTES: u64 = 256 * 1024 * 1024;
const HTTP_CONNECT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(15);
/// Passed to the reqwest blocking client's `timeout`, which it applies PER READ of the
/// streamed body (not to the whole body) — so it serves as a per-read stall guard: a read
/// making NO progress for this long aborts. Paired with [`HTTP_FETCH_DEADLINE`] below,
/// which bounds the TOTAL body read (a slow-DRIP server that dribbles a byte just under
/// this per-read timeout would otherwise run for months).
const HTTP_READ_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);
/// Wall-clock ceiling for a single GET's whole body — enforced by us in [`HttpSource::
/// get_bounded`] because reqwest's blocking client has no total-body deadline.
const HTTP_FETCH_DEADLINE: std::time::Duration = std::time::Duration::from_secs(600);

fn fetch_err(detail: String) -> HiggsError {
    HiggsError::UpdateFetchFailed { detail }
}

/// True iff `u` is a fetch target we permit: `https` to any host, or `http` to a LOOPBACK
/// host only. (Redirects are NOT followed at all — see [`HttpSource::new`] — so this gates
/// only the INITIAL, operator-supplied URL.)
fn is_allowed_fetch_url(u: &reqwest::Url) -> bool {
    match u.scheme() {
        "https" => true,
        "http" => is_loopback_host(u),
        _ => false,
    }
}

/// True iff `u`'s host is loopback (`127.0.0.0/8`, `::1`, or `localhost`) — the only host
/// for which a plaintext `http://` update URL is allowed (same-box / test fetches never
/// leave the machine). The signature still gates authenticity regardless of scheme.
/// `pub(crate)` so the hub-side release courier reuses this exact loopback rule for the
/// operator-supplied MANIFEST url it fetches (REL-P4e).
pub(crate) fn is_loopback_host(u: &reqwest::Url) -> bool {
    match u.host_str() {
        Some(h) => {
            let bare = h.trim_start_matches('[').trim_end_matches(']');
            bare.eq_ignore_ascii_case("localhost")
                || bare
                    .parse::<std::net::IpAddr>()
                    .map(|ip| ip.is_loopback())
                    .unwrap_or(false)
        }
        None => false,
    }
}

/// A URL rendered SAFE for an error message / log: a CONSTANT placeholder that names NO
/// component of the URL. EVERY component is user/attacker-controlled and can carry a
/// capability — the userinfo (`user:pass@`), the SCHEME (a made-up `cap<token>://`), the
/// HOST/subdomain (`cap-<token>.host`), the port, the path (`/download/<token>/`), the
/// query (`?sig=…`), the fragment. The `what` label ("manifest"/"signature"/"artifact") +
/// the HG088 reason already identify the failure, and the operator knows the URL they
/// passed, so naming nothing loses no actionable context while guaranteeing no
/// operator/hub-supplied secret reaches a log that might be shipped to a less-privileged
/// reader. (Takes the URL only so call sites read naturally.)
fn redact_url(_u: &reqwest::Url) -> &'static str {
    "the update URL"
}

/// Parse + vet an update URL: `https` to any host, or `http` ONLY to a loopback host
/// ([`is_allowed_fetch_url`]). Additionally REFUSE a URL that carries USER-INFO
/// (`user:pass@` — could leak on a downgrade redirect) or a QUERY/FRAGMENT (the sibling
/// `.minisig` and artifact URLs are derived by appending to / replacing the last PATH
/// segment, which a query/fragment would corrupt — a presigned/query-authed URL doesn't
/// fit the sibling model anyway; fetch those with `--tarball/--manifest/--manifest-sig`).
/// Error messages carry only the [`redact_url`] form, never `raw` (which may hold a secret).
fn parse_fetch_url(raw: &str) -> Result<reqwest::Url, HiggsError> {
    // A parse failure's `e` (url::ParseError) does NOT echo the input, so it is safe; `raw`
    // itself is NOT included (it could be a mistyped secret-bearing URL).
    let u = reqwest::Url::parse(raw).map_err(|e| fetch_err(format!("invalid update URL: {e}")))?;
    if !u.username().is_empty() || u.password().is_some() {
        return Err(fetch_err(format!(
            "update URL {} must not embed credentials (they could leak on a redirect)",
            redact_url(&u)
        )));
    }
    if u.query().is_some() || u.fragment().is_some() {
        return Err(fetch_err(format!(
            "update URL {} must not have a query or fragment — the .minisig + artifact URLs \
             are derived from its path",
            redact_url(&u)
        )));
    }
    if is_allowed_fetch_url(&u) {
        Ok(u)
    } else if u.scheme() == "http" {
        Err(fetch_err(format!(
            "refusing a plaintext http:// update URL to a non-loopback host ({}) — use https",
            redact_url(&u)
        )))
    } else {
        // Do NOT echo `u.scheme()` — the scheme is user-controlled (a made-up
        // `cap<token>://` could carry a capability). Name only the allowed schemes.
        Err(fetch_err(format!(
            "{} has an unsupported scheme — only https (or http to a loopback host) is allowed",
            redact_url(&u)
        )))
    }
}

/// Derive the artifact URL as a SIBLING of the manifest URL named by the manifest's
/// (authenticated) `file` field. `file` must be a BARE filename — no `/`, `\`, or `..` —
/// so it can only name a file in the manifest's own directory, never traverse or
/// re-root the URL. (`file` is already authenticated when this runs; this is defence in
/// depth against a compromised-key manifest.)
fn artifact_url_from(manifest_url: &reqwest::Url, file: &str) -> Result<reqwest::Url, HiggsError> {
    if file.is_empty()
        || file.contains('/')
        || file.contains('\\')
        || file.contains("..")
        || file.starts_with('.')
    {
        return Err(fetch_err(format!(
            "manifest `file` {file:?} is not a bare sibling filename — refusing to derive an artifact URL"
        )));
    }
    // `Url::join(bare)` replaces the manifest URL's last path segment with `file`.
    manifest_url
        .join(file)
        .map_err(|e| fetch_err(format!("cannot derive artifact URL for {file:?}: {e}")))
}

/// An update fetched over HTTPS from a manifest URL (DESIGN-remote §9 P4). Downloads the
/// manifest, its `.minisig` (sibling `<manifest>.minisig`), and — after the caller has
/// verified the manifest — the artifact named by the manifest's `file` field. Every body
/// is size-capped and time-bounded; nothing is trusted until [`verify_and_check`] checks
/// the signature + sha256.
pub struct HttpSource {
    manifest_url: reqwest::Url,
    client: reqwest::blocking::Client,
}

/// Build the blocking HTTP client shared by [`HttpSource`] and [`PushSource`], for a fetch
/// aimed at `target`. Blocking because self-update runs OUTSIDE any async runtime. The
/// policy is deliberately conservative:
/// - connect timeout + a per-read stall timeout (reqwest's blocking `timeout` is applied per
///   READ of the streamed body; the TOTAL body time is bounded by [`fetch_bounded`]'s
///   `HTTP_FETCH_DEADLINE`);
/// - NO redirects (`Policy::none()`) — following one opens an SSRF surface a compromised
///   origin could exploit, and (for --url) makes sibling `.minisig`/artifact URLs ambiguous;
///   a 3xx surfaces as a clear error;
/// - `no_proxy()` when `target` is loopback, so an `http://127.0.0.1` fetch never traverses a
///   configured `HTTP_PROXY` off the machine (breaking the "stays on-box" premise + failing).
fn build_fetch_client(target: &reqwest::Url) -> Result<reqwest::blocking::Client, HiggsError> {
    let mut builder = reqwest::blocking::Client::builder()
        .connect_timeout(HTTP_CONNECT_TIMEOUT)
        .timeout(HTTP_READ_TIMEOUT)
        .redirect(reqwest::redirect::Policy::none());
    if is_loopback_host(target) {
        builder = builder.no_proxy();
    }
    builder
        .build()
        .map_err(|e| fetch_err(format!("cannot build the HTTP client: {e}")))
}

/// True iff `ip` is NOT a public, globally-routable address a hub-pushed artifact URL may target
/// (an SSRF guard for the PUSH path; see [`vet_and_resolve_pushed_url`]). Rust's `IpAddr::is_global`
/// is unstable, so this classifies CONSERVATIVELY: it accepts only what a real release host could
/// plausibly be (normal global-unicast, or a DNS64 NAT64 mapping of a public v4) and rejects
/// everything else — every non-global range PLUS the reserved/documentation/protocol blocks that
/// sit inside global-unicast. Rejecting an obscure-but-technically-global block (AMT, AS112,
/// ORCHID) is harmless: a release is never hosted there, so there is no false-negative that matters
/// AND no way for a hub to reach an internal service through it. `pub(crate)` so the hub-side
/// release courier reuses this exact IP policy when it resolves + vets the operator-supplied
/// MANIFEST host it fetches (REL-P4e) — ONE classifier, no drift.
///
/// RESIDUAL — NETWORK-SPECIFIC NAT64: on a DNS64/NAT64 network using a NETWORK-SPECIFIC prefix (RFC
/// 6052 permits any /32../96 prefix, not just the well-known `64:ff9b::/96` handled below), the
/// resolver can synthesise `<operator-prefix>::<private-v4>` for a hub-controlled domain whose `A`
/// record is private. That address is indistinguishable from normal global-unicast here — the
/// operator's prefix is unknown to this process, so NO IP-layer classifier can detect it. Accepted
/// residual: it needs an IPv6-only DNS64 deployment (uncommon for fleet nodes), the GET is BLIND
/// (its body still fails sha256), and the DEFINITIVE closure is signing the artifact URL in the CI
/// manifest so the node never fetches a hub-chosen URL — a P1/CI follow-up, out of this diff.
pub(crate) fn is_ssrf_prone_ip(ip: &std::net::IpAddr) -> bool {
    match ip {
        std::net::IpAddr::V4(v4) => is_ssrf_prone_v4(v4),
        std::net::IpAddr::V6(v6) => {
            let s = v6.segments();
            // An IPv4-mapped address (`::ffff:a.b.c.d`) is classified by its embedded v4.
            if let Some(mapped) = v6.to_ipv4_mapped() {
                return is_ssrf_prone_v4(&mapped);
            }
            // NAT64 well-known prefix `64:ff9b::/96` (RFC 6052): a DNS64 node receives these for
            // LEGIT v4 hosts, so classify by the embedded v4 — a public v4 is a valid release host
            // (fixing a false-positive), a private one is SSRF-via-NAT64. (The LOCAL-use prefix
            // `64:ff9b:1::/48` is caught below as non-global.)
            if s[..6] == [0x64, 0xff9b, 0, 0, 0, 0] {
                let v4 = std::net::Ipv4Addr::new(
                    (s[6] >> 8) as u8,
                    s[6] as u8,
                    (s[7] >> 8) as u8,
                    s[7] as u8,
                );
                return is_ssrf_prone_v4(&v4);
            }
            // The ONLY globally-routable IPv6 block is global-unicast `2000::/3`, so anything
            // OUTSIDE it (loopback/unspecified/ULA `fc00::/7`/link-local `fe80::/10`/multicast/
            // discard `100::/64`/local NAT64/unassigned) is non-global. WITHIN `2000::/3`,
            // conservatively reject the non-release blocks: `2001::/23` (IETF protocol
            // assignments — incl. benchmarking `2001:2::/48`, ORCHIDv2 `2001:20::/28`), the
            // documentation blocks `2001:db8::/32` and `3fff::/20`, and deprecated 6to4
            // `2002::/16`.
            (s[0] & 0xe000) != 0x2000
                || (s[0] == 0x2001 && s[1] < 0x0200) // 2001::/23 IETF protocol assignments
                || (s[0] == 0x2001 && s[1] == 0x0db8) // 2001:db8::/32 documentation
                || s[0] == 0x2002 // 2002::/16 6to4 (deprecated)
                || (s[0] == 0x3fff && s[1] <= 0x0fff) // 3fff::/20 documentation
        }
    }
}

fn is_ssrf_prone_v4(v4: &std::net::Ipv4Addr) -> bool {
    let o = v4.octets();
    let (a, b, c) = (o[0], o[1], o[2]);
    v4.is_loopback()
        || v4.is_private()
        || v4.is_link_local()
        || v4.is_broadcast()
        || v4.is_unspecified()
        || v4.is_documentation()
        || v4.is_multicast()
        || a == 0 // 0.0.0.0/8 "this network"
        || (a == 100 && (b & 0xc0) == 64) // 100.64.0.0/10 CGNAT
        || (a == 192 && b == 0 && c == 0) // 192.0.0.0/24 IETF protocol assignments
        || (a == 192 && b == 88 && c == 99) // 192.88.99.0/24 6to4 relay anycast (deprecated)
        || (a == 198 && (b & 0xfe) == 18) // 198.18.0.0/15 benchmarking
        || a >= 240 // 240.0.0.0/4 reserved (incl. 255.255.255.255)
}

/// Vet a HUB-PUSHED artifact URL with a STRICTER policy than the CLI `--url` path
/// ([`parse_fetch_url`], which permits loopback for local/test servers): require `https`, then
/// RESOLVE the host and refuse if ANY resolved address is non-global ([`is_ssrf_prone_ip`]).
/// Returns the resolved addresses so the client can be PINNED to them — the actual fetch then
/// connects to an address validated HERE, never a re-resolution that could rebind to a private
/// IP between this check and the connect (DNS-rebinding defence). A literal-IP host resolves to
/// itself without a DNS query. The pushed manifest is signature-verified regardless, but the
/// artifact GET fires BEFORE the sha256 check, so a compromised paired hub must not be able to
/// aim that GET at the node's own or its LAN's services.
fn vet_and_resolve_pushed_url(url: &reqwest::Url) -> Result<Vec<std::net::SocketAddr>, HiggsError> {
    use std::net::ToSocketAddrs;
    if url.scheme() != "https" {
        return Err(fetch_err(format!(
            "a hub-pushed artifact URL ({}) must be https — plaintext/loopback is a CLI-only \
             affordance, never extended to an untrusted hub push",
            redact_url(url)
        )));
    }
    let port = url.port_or_known_default().unwrap_or(443);
    // Use the TYPED host: a literal Ipv4/Ipv6 is taken verbatim (no DNS, and — crucially — no
    // `[...]` brackets, which `host_str()` keeps and `ToSocketAddrs` then rejects); only a
    // DOMAIN goes through the resolver.
    let addrs: Vec<std::net::SocketAddr> = match url.host() {
        Some(url::Host::Ipv4(ip)) => vec![std::net::SocketAddr::new(ip.into(), port)],
        Some(url::Host::Ipv6(ip)) => vec![std::net::SocketAddr::new(ip.into(), port)],
        Some(url::Host::Domain(d)) => (d, port)
            .to_socket_addrs()
            .map_err(|e| fetch_err(format!("cannot resolve the pushed update host: {e}")))?
            .collect(),
        None => {
            return Err(fetch_err(format!(
                "a hub-pushed artifact URL ({}) has no host",
                redact_url(url)
            )))
        }
    };
    if addrs.is_empty() {
        return Err(fetch_err(
            "the pushed update host did not resolve to any address".into(),
        ));
    }
    for a in &addrs {
        if is_ssrf_prone_ip(&a.ip()) {
            return Err(fetch_err(format!(
                "a hub-pushed artifact URL ({}) resolves to a private/loopback/link-local \
                 address — refusing (SSRF guard)",
                redact_url(url)
            )));
        }
    }
    Ok(addrs)
}

/// Build the blocking client for a HUB-PUSHED artifact fetch, PINNED to the pre-validated
/// `addrs` (from [`vet_and_resolve_pushed_url`]) so the connect cannot re-resolve the host to a
/// private IP. `no_proxy()` because a configured proxy would connect to the PROXY (defeating the
/// pin + the SSRF check). TLS still validates the host's certificate for its SNI name.
fn build_pushed_fetch_client(
    url: &reqwest::Url,
    addrs: &[std::net::SocketAddr],
) -> Result<reqwest::blocking::Client, HiggsError> {
    let host = url.host_str().unwrap_or_default();
    reqwest::blocking::Client::builder()
        .connect_timeout(HTTP_CONNECT_TIMEOUT)
        .timeout(HTTP_READ_TIMEOUT)
        .redirect(reqwest::redirect::Policy::none())
        .no_proxy()
        .resolve_to_addrs(host, addrs)
        .build()
        .map_err(|e| fetch_err(format!("cannot build the HTTP client: {e}")))
}

impl HttpSource {
    /// Build a source for `manifest_url` (vetted by [`parse_fetch_url`]).
    pub fn new(manifest_url: &str) -> Result<Self, HiggsError> {
        let manifest_url = parse_fetch_url(manifest_url)?;
        let client = build_fetch_client(&manifest_url)?;
        Ok(Self {
            manifest_url,
            client,
        })
    }

    /// Delegates to the shared [`fetch_bounded`] over this source's client.
    fn get_bounded(&self, url: &reqwest::Url, cap: u64, what: &str) -> Result<Vec<u8>, HiggsError> {
        fetch_bounded(&self.client, url, cap, what)
    }
}

/// GET `url`, erroring on a non-2xx status, an over-cap `Content-Length`, a body that
/// exceeds `cap` bytes even if the header lied/was absent, or a body that takes longer than
/// [`HTTP_FETCH_DEADLINE`] to arrive (a slow-drip guard the per-read timeout alone cannot
/// provide). The body is read in bounded chunks so nothing over `cap` is ever buffered.
/// Shared by [`HttpSource`] (--url CLI) and [`PushSource`] (M_UPDATE hub-push artifact).
fn fetch_bounded(
    client: &reqwest::blocking::Client,
    url: &reqwest::Url,
    cap: u64,
    what: &str,
) -> Result<Vec<u8>, HiggsError> {
    // Any URL COMPONENT (host/subdomain, path, query) can carry a capability, so every
    // error names only the redacted `loc` + the `what` label — never the full URL. reqwest's
    // own errors embed the URL too, so strip it with `.without_url()` before formatting.
    {
        let loc = redact_url(url);
        let deadline = std::time::Instant::now() + HTTP_FETCH_DEADLINE;
        let resp = client
            .get(url.clone())
            .send()
            .map_err(|e| fetch_err(format!("GET {what} {loc}: {}", e.without_url())))?;
        if !resp.status().is_success() {
            // Format ONLY the numeric status — the HTTP REASON PHRASE is server-controlled and
            // an untrusted server can echo the request's path/capability token back in it
            // (`HTTP/1.1 404 <token>`), which `.without_url()` would still include. A 3xx means
            // the update URL redirected (we do NOT follow — see `HttpSource::new`); hint that
            // the operator must pass the FINAL, direct URL.
            let status = resp.status();
            if status.is_redirection() {
                return Err(fetch_err(format!(
                    "{what} {loc} redirected (HTTP {}); redirects are not followed — pass the \
                     final direct URL to --url (or use --tarball/--manifest/--manifest-sig)",
                    status.as_u16()
                )));
            }
            return Err(fetch_err(format!(
                "GET {what} {loc}: HTTP {}",
                status.as_u16()
            )));
        }
        if let Some(len) = resp.content_length() {
            if len > cap {
                // Do NOT echo `len`: the Content-Length is server-controlled and a hostile
                // server can set it to a numeric capability/token from the request to reflect
                // it into our logs. Only our own `cap` is named.
                return Err(fetch_err(format!(
                    "{what} at {loc} exceeds the {cap}-byte cap"
                )));
            }
        }
        // Read in chunks: enforce the wall-clock deadline (slow-drip guard) and the size
        // cap (nothing over `cap` is ever appended) as we go, regardless of the untrusted
        // Content-Length header.
        let mut body = resp;
        let mut buf: Vec<u8> = Vec::new();
        let mut chunk = [0u8; 64 * 1024];
        loop {
            if std::time::Instant::now() >= deadline {
                return Err(fetch_err(format!(
                    "{what} at {loc} exceeded the {HTTP_FETCH_DEADLINE:?} fetch deadline"
                )));
            }
            let n = std::io::Read::read(&mut body, &mut chunk)
                .map_err(|e| fetch_err(format!("reading {what} {loc}: {e}")))?;
            if n == 0 {
                break;
            }
            if buf.len() as u64 + n as u64 > cap {
                return Err(fetch_err(format!(
                    "{what} at {loc} exceeded the {cap}-byte cap"
                )));
            }
            buf.extend_from_slice(&chunk[..n]);
        }
        Ok(buf)
    }
}

impl UpdateSource for HttpSource {
    fn manifest(&self) -> Result<(Vec<u8>, String), HiggsError> {
        let manifest = self.get_bounded(&self.manifest_url, MAX_MANIFEST_BYTES, "manifest")?;
        // The `.minisig` sits next to the manifest — append to the PATH (not the whole URL
        // string), so it stays correct even if the URL grows a component. (`parse_fetch_url`
        // already refuses a query/fragment, so there is nothing after the path to corrupt.)
        let mut sig_url = self.manifest_url.clone();
        sig_url.set_path(&format!("{}.minisig", self.manifest_url.path()));
        let sig_bytes = self.get_bounded(&sig_url, MAX_SIG_BYTES, "signature")?;
        let sig = String::from_utf8(sig_bytes).map_err(|e| {
            fetch_err(format!(
                "signature at {} is not UTF-8: {e}",
                redact_url(&sig_url)
            ))
        })?;
        Ok((manifest, sig))
    }

    fn artifact(&self, file: &str) -> Result<Vec<u8>, HiggsError> {
        let url = artifact_url_from(&self.manifest_url, file)?;
        self.get_bounded(&url, MAX_ARTIFACT_BYTES, "artifact")
    }
}

/// An update PUSHED by the hub (DESIGN-remote §9 `M_UPDATE`): the (tiny) manifest + its
/// `.minisig` arrive INLINE in the RPC, and only the (large) artifact is fetched, from a
/// single DIRECT `artifact_url`. This mirrors the `M_UPDATE` params and avoids the sibling-
/// derivation the `--url` path needs. The inline bytes + the fetched artifact are trusted by
/// nothing until [`verify_and_check`] checks the signature + sha256 — exactly like every
/// other [`UpdateSource`].
pub struct PushSource {
    manifest: Vec<u8>,
    sig: String,
    artifact_url: reqwest::Url,
    client: reqwest::blocking::Client,
}

/// Reject an oversized INLINE manifest/signature BEFORE it is copied/verified. The HTTP
/// caps ([`MAX_MANIFEST_BYTES`]/[`MAX_SIG_BYTES`]) bound a fetched body; a hub-PUSHED update
/// carries these INLINE, so a paired-but-compromised hub could otherwise send a 100 MiB
/// `manifest` string and OOM the node (each copy — parse, `PushSource`, `manifest()` — before
/// the signature refusal). Call this SYNCHRONOUSLY (before spawning the detached apply) so an
/// oversized push fails fast and is never copied again. A real manifest is a small JSON; a
/// minisig is a couple of short lines.
pub fn check_pushed_sizes(manifest: &str, manifest_sig: &str) -> Result<(), HiggsError> {
    if manifest.len() as u64 > MAX_MANIFEST_BYTES {
        return Err(HiggsError::UpdateFetchFailed {
            detail: format!(
                "pushed manifest is {} bytes, over the {MAX_MANIFEST_BYTES}-byte cap",
                manifest.len()
            ),
        });
    }
    if manifest_sig.len() as u64 > MAX_SIG_BYTES {
        return Err(HiggsError::UpdateFetchFailed {
            detail: format!(
                "pushed signature is {} bytes, over the {MAX_SIG_BYTES}-byte cap",
                manifest_sig.len()
            ),
        });
    }
    Ok(())
}

impl PushSource {
    /// `manifest` = the manifest JSON text, `sig` = its `.minisig` text (both inline from the
    /// push), `artifact_url` = a DIRECT URL to the tarball. Vetted by [`parse_fetch_url`] AND the
    /// stricter push-only [`vet_and_resolve_pushed_url`] (https-only, no private/loopback address,
    /// DNS-rebinding-pinned), because the URL is hub-supplied and untrusted; redirects are not
    /// followed, so it must be the final URL.
    pub fn new(manifest: &str, sig: &str, artifact_url: &str) -> Result<Self, HiggsError> {
        // Cap the inline bytes BEFORE the copy (belt for the handler's synchronous precheck).
        check_pushed_sizes(manifest, sig)?;
        let artifact_url = parse_fetch_url(artifact_url)?;
        // SSRF guard (push path only): reject a non-https / private / loopback target and PIN the
        // client to the validated address so the fetch can't rebind. The CLI `--url` path keeps
        // its looser policy (it deliberately serves local/test tarballs); a hub push must not.
        let addrs = vet_and_resolve_pushed_url(&artifact_url)?;
        let client = build_pushed_fetch_client(&artifact_url, &addrs)?;
        Ok(Self {
            manifest: manifest.as_bytes().to_vec(),
            sig: sig.to_string(),
            artifact_url,
            client,
        })
    }
}

impl UpdateSource for PushSource {
    fn manifest(&self) -> Result<(Vec<u8>, String), HiggsError> {
        // Already in hand from the push — no fetch. Authenticity is the caller's signature check.
        Ok((self.manifest.clone(), self.sig.clone()))
    }

    fn artifact(&self, file: &str) -> Result<Vec<u8>, HiggsError> {
        // BIND the hub-supplied URL to the SIGNED manifest `file`: the URL's last path segment
        // MUST equal the authenticated release filename. The `artifact_url` is pushed by the hub
        // INDEPENDENTLY of the inline manifest, and the sha256 check (HG084) only runs AFTER the
        // GET — so without this bind a compromised paired hub could replay a valid signed manifest
        // and point `artifact_url` at an arbitrary endpoint (blind SSRF: the node issues a GET to
        // e.g. an internal service, whose side effect lands even though the bytes then fail sha256).
        // Requiring the URL to NAME the signed release file reduces that to, at worst, a GET whose
        // path ends in the exact release filename. RESIDUAL: the URL's HOST is still hub-chosen and
        // loopback/private hosts are permitted (the CLI `--url` path needs local/test servers); a
        // hub pointing at an internal host that answers a `<file>`-suffixed path is a narrow, sha-
        // rejected residual — full private-network blocking (with DNS-rebinding defence) is a
        // deferred follow-up.
        let named = self
            .artifact_url
            .path_segments()
            .and_then(|mut segs| segs.next_back())
            .unwrap_or("");
        if named != file {
            return Err(fetch_err(format!(
                "pushed artifact URL {} does not name the manifest's file {file:?} — refusing",
                redact_url(&self.artifact_url)
            )));
        }
        fetch_bounded(
            &self.client,
            &self.artifact_url,
            MAX_ARTIFACT_BYTES,
            "artifact",
        )
    }
}

// ---------------------------------------------------------------------------
// Verify pipeline — authenticity + eligibility, before any disk write
// ---------------------------------------------------------------------------

/// A verified, eligible update ready to apply: the authenticated manifest, the
/// verified artifact bytes, and the key id that verified it. Produced by
/// [`verify_and_check`]; consumed by [`stage_and_flip`].
#[derive(Debug)]
pub struct VerifiedUpdate {
    pub key_id: String,
    pub manifest: UpdateManifest,
    pub artifact: Vec<u8>,
}

/// Fetch → verify signature (HG081-083) → check eligibility (HG085-086) → fetch
/// artifact → verify sha256 (HG084). Returns only when the bytes are BOTH authentic
/// and an eligible upgrade for `running`; nothing has touched the install tree yet.
pub fn verify_and_check(
    source: &dyn UpdateSource,
    running: &BuildIdentity,
    allow_downgrade: bool,
    // OUT: the AUTHENTICATED target version, set the moment the manifest signature AND parse both
    // succeed (before eligibility/artifact/sha). A caller that records a post-verification failure
    // reads this to report the real target version rather than an unauthenticated hint. Left `None`
    // when the signature fails OR the manifest fails to parse — including a signed manifest with an
    // UNSUPPORTED SCHEMA (HG083): its bytes are authentic, but this binary deserialized them with
    // its OWN schema's struct, so a `version` extracted that way has no defined meaning under the
    // unknown schema and must not be presented as authenticated. The caller's hint fallback is the
    // honest value there, and the HG083 reason carries the actual schema number for the operator.
    authenticated_version: &mut Option<String>,
    // When set, the AUTHENTICATED manifest must be exactly this version — the caller named a
    // release (a version-only trigger) rather than accepting whatever the source serves. Checked
    // BEFORE the artifact download, so a validly-signed manifest for a different version cannot
    // force a large fetch before being refused.
    expected_version: Option<&str>,
) -> Result<VerifiedUpdate, HiggsError> {
    verify_and_check_with(
        source,
        running,
        allow_downgrade,
        authenticated_version,
        expected_version,
        &crate::update::HIGGS_UPDATE_PUBKEYS,
    )
}

/// [`verify_and_check`] with an injected key table — PRIVATE test seam, same
/// rationale (and same table plumbing) as `verify_manifest_any_with`.
fn verify_and_check_with(
    source: &dyn UpdateSource,
    running: &BuildIdentity,
    allow_downgrade: bool,
    authenticated_version: &mut Option<String>,
    expected_version: Option<&str>,
    pubkeys: &[(&str, &str)],
) -> Result<VerifiedUpdate, HiggsError> {
    let (manifest_bytes, sig_text) = source.manifest()?;
    // Signature BEFORE anything else reads the manifest fields (P1 invariant).
    let (key_id, manifest) =
        crate::update::verify_manifest_any_with(&manifest_bytes, &sig_text, pubkeys)?;
    *authenticated_version = Some(manifest.version.clone());
    if let Some(expected) = expected_version {
        if manifest.version != expected {
            return Err(HiggsError::UpdateManifestInvalid {
                detail: "the fetched manifest's version does not match the requested \
                         release — refusing"
                    .into(),
            });
        }
    }
    // Eligibility BEFORE downloading the (large) artifact — refuse a wrong-target or
    // downgrade artifact without spending the bandwidth to fetch it.
    evaluate_eligibility(running, &manifest, allow_downgrade)?;
    let artifact = source.artifact(&manifest.file)?;
    verify_artifact_sha256(&manifest, &artifact)?;
    Ok(VerifiedUpdate {
        key_id,
        manifest,
        artifact,
    })
}

/// Apply a hub-PUSHED update (`M_UPDATE`, §9): build a [`PushSource`] from the INLINE
/// manifest+sig + the direct `artifact_url`, then run the SAME verify → check → lock → stage
/// → flip pipeline the CLI update path uses. Returns `(from_version, to_version)` on success;
/// the caller (the control handler) runs this on a BLOCKING thread and, on success, calls
/// [`request_self_restart`] so the daemon drains + restarts onto the staged binary and the
/// boot-guard confirms the trial or rolls it back. Fails CLOSED (HG081) on a dev build
/// with no pinned key. Refuses to run as root — the node daemon runs as the UNPRIVILEGED
/// operator, and a root-owned staged binary / lock would break the operator-run boot hooks.
///
/// Records the failure for the next HELLO on EVERY definitive failure — a rejected/unreachable
/// artifact URL (HG088 SSRF/DNS/size), a hard lock-subsystem failure, or a verify/eligibility/sha/
/// stage failure. The node stays on its running version and has already replied `"accepted"`, so
/// version-stagnation is all the hub would otherwise see; the marker tells it WHY via
/// [`record_update_failure`]. `from` is the RUNNING process version (`CARGO_PKG_VERSION`, matching
/// the HELLO `software_version`), NOT the installed `current` eligibility judges against — the two
/// can differ if `current` was flipped without a restart, and `from` must agree with
/// `software_version`. `to` is the AUTHENTICATED manifest version once the signature verifies, else
/// `target_hint` (the push's declared target) before that point. ALL recordable validation and the
/// success-clear run UNDER the update lock, so a CONCURRENT push can never record a failure that
/// races another apply's clear; only genuine lock CONTENTION (HG087 — another apply is mid-flight,
/// this one simply could not run) is exempt, and the pre-lock root refusal (a misconfiguration).
/// On SUCCESS it CLEARS any stale marker durably (fsync) — a newer staged attempt supersedes an
/// older unreported failure.
pub fn apply_pushed_update(
    bin: &Path,
    manifest: &str,
    manifest_sig: &str,
    artifact_url: &str,
    allow_downgrade: bool,
    // The push's DECLARED target version (an unauthenticated hint) — reported as `to` only when a
    // failure precedes signature verification, so the authenticated version is not yet known.
    target_hint: &str,
) -> Result<(String, String), HiggsError> {
    // The RUNNING process version — what the HELLO reports as `software_version` and what
    // `UpdateFailed.from` must agree with. Distinct from the installed `current` identity used for
    // eligibility below.
    let running_version = env!("CARGO_PKG_VERSION");
    // Refuse root BEFORE touching the lock — a root-created `0600` lock would lock out the
    // unprivileged operator's later runs. NOT recorded (a deployment misconfiguration that never
    // arises for the normal unprivileged daemon, and recording it as root would leave a
    // root-owned marker) and it never touches shared state, so it needs no serialization.
    // SAFETY: geteuid has no preconditions and no failure mode.
    if unsafe { libc::geteuid() } == 0 {
        return Err(HiggsError::UpdateApplyFailed {
            detail: "refusing to apply a pushed update as root — the node runs as the \
                     unprivileged operator"
                .into(),
        });
    }
    // Acquire the single-update lock FIRST, then run ALL recordable validation + the success-clear
    // UNDER it. This is BOTH the OOM guard (only one push fetches the up-to-256 MiB artifact at a
    // time; a concurrent one is refused HG087 before spending memory/bandwidth) AND the fix for the
    // overlapping-push race: because every failure-record and the success-clear are mutually
    // exclusive under this exclusive lock, a concurrent push can never record a failure that races
    // another apply's clear. Only genuine CONTENTION (another apply mid-flight — it simply could not
    // run) is exempt from recording; a hard lock-subsystem FAILURE (a malformed/inaccessible lock
    // dir/node) is a definitive failure of THIS push and IS recorded. `flock` on the per-open lock
    // fd serializes even two threads in this daemon (separate opens ⇒ separate lock owners).
    let _lock = match UpdateLock::try_acquire(bin) {
        LockAttempt::Held(lock) => lock,
        LockAttempt::Contended => {
            return Err(HiggsError::UpdateApplyFailed {
                detail: format!(
                    "another self-update is already running (lock held on {})",
                    lock_path(bin).display()
                ),
            });
        }
        LockAttempt::Failed(e) => {
            record_update_failure(bin, running_version, target_hint, &e.to_string());
            return Err(e);
        }
    };
    // Everything below runs under the lock: build+vet the push source (https/SSRF/size), judge
    // eligibility against the INSTALLED `current` (not the running process — a stale in-memory
    // identity must not flip `current` backwards), fetch+verify, and stage+flip. Any failure here
    // is recorded (to = the AUTHENTICATED manifest version once the signature verifies, captured
    // the instant it does; else the unauthenticated `target_hint`); success clears the marker.
    let mut authenticated_to: Option<String> = None;
    let staged = (|| {
        let source = PushSource::new(manifest, manifest_sig, artifact_url)?;
        let running = installed_identity(bin);
        let verified = verify_and_check(
            &source,
            &running,
            allow_downgrade,
            &mut authenticated_to,
            None,
        )?;
        stage_and_flip(
            bin,
            &verified.manifest,
            &verified.artifact,
            allow_downgrade,
            &smoke_run,
        )?;
        Ok::<(String, String), HiggsError>((running.version, verified.manifest.version))
    })();
    match staged {
        // A pending update STAGED — `stage_and_flip` already CLEARED the last-failure marker
        // durably at the flip (the single point both this path and the CLI reach), so any earlier
        // failure is moot. If THIS staged binary crash-loops, the boot-guard re-records the failure.
        Ok(pair) => Ok(pair),
        Err(e) => {
            record_update_failure(
                bin,
                running_version,
                authenticated_to.as_deref().unwrap_or(target_hint),
                &e.to_string(),
            );
            Err(e)
        }
    }
}

// ---------------------------------------------------------------------------
// Version-only update (M_NODE_UPDATE_VERSION): the node fetches its OWN release
// ---------------------------------------------------------------------------

/// Max redirect hops the release fetch follows. GitHub serves `releases/download/...`
/// via ONE 302 to its storage host; a small budget covers a mirror in front of it
/// without letting a hostile chain wander.
const RELEASE_REDIRECT_HOPS: usize = 4;

/// Derive the three release-asset URLs for `version` and THIS build's target/variant
/// under the node's configured `release_url` base (`…/releases`):
/// `<base>/download/v<ver>/higgs-v<ver>-<suffix>{.manifest,.manifest.minisig,.tar.gz}`.
/// PURE — the fetch policy is [`build_release_client`]. The base must be `https` (or
/// `http` to a LITERAL loopback IP — the on-box test affordance every other release
/// path shares) with no credentials/query/fragment, since the three children are
/// derived by path edits.
pub fn release_asset_urls(
    base: &str,
    version: &str,
) -> Result<(reqwest::Url, reqwest::Url, reqwest::Url), HiggsError> {
    // Syntax gate FIRST: the version lands in a URL path — a separator or traversal
    // in a hub-supplied string must die here, and semver rejects all of them.
    semver::Version::parse(version).map_err(|_| HiggsError::UpdateFetchFailed {
        detail: "the requested update version is not a plain semver string".into(),
    })?;
    let base_url = crate::node::release_courier::parse_courier_url(base)?;
    let ident = BuildIdentity::current();
    let name = format!(
        "higgs-v{version}-{}",
        crate::node::release_courier::asset_suffix(&ident.target, &ident.variant)
    );
    let dir = {
        let mut b = base_url.clone();
        let path = b.path().trim_end_matches('/').to_string();
        b.set_path(&format!("{path}/download/v{version}/"));
        b
    };
    let child = |file: String| -> Result<reqwest::Url, HiggsError> {
        dir.join(&file).map_err(|e| HiggsError::UpdateFetchFailed {
            detail: format!("cannot derive a release asset URL: {e}"),
        })
    };
    Ok((
        child(format!("{name}.manifest"))?,
        child(format!("{name}.manifest.minisig"))?,
        child(format!("{name}.tar.gz"))?,
    ))
}

/// True iff `next` is an acceptable redirect TARGET for the release fetch: `https`
/// with no embedded credentials. (Unlike every other update fetch, a QUERY is
/// allowed — GitHub's storage URLs carry signed query strings — because nothing is
/// derived from a post-redirect URL: all three asset URLs were derived from the
/// configured base BEFORE any redirect, and the bytes are untrusted until the CI
/// signature + sha256 verify.) `http` is never an acceptable hop — not even back to
/// loopback: a downgrade mid-chain would move signed-release traffic to plaintext.
fn release_redirect_ok(next: &reqwest::Url) -> bool {
    next.scheme() == "https" && next.username().is_empty() && next.password().is_none()
}

/// The blocking client for the version-update fetch: bounded redirects where EVERY
/// hop must pass [`release_redirect_ok`]; same connect/read timeouts as every other
/// update fetch. The release base is OPERATOR CONFIG (`config.json` `release_url`,
/// default = this repo's GitHub releases) — trusted at the same level as the CLI
/// `--url` — and the fetched bytes are trusted by NOTHING until [`verify_and_check`]
/// proves the CI signature and the artifact sha256. `no_proxy` for a loopback base so
/// an on-box test fetch never leaves the machine.
fn build_release_client(base: &reqwest::Url) -> Result<reqwest::blocking::Client, HiggsError> {
    let mut builder = reqwest::blocking::Client::builder()
        .connect_timeout(HTTP_CONNECT_TIMEOUT)
        .timeout(HTTP_READ_TIMEOUT)
        // No Referer across hops: a pre-redirect URL (which may carry a
        // configured-path capability) must not leak to later redirect targets.
        .referer(false)
        .redirect(reqwest::redirect::Policy::custom(|attempt| {
            if attempt.previous().len() > RELEASE_REDIRECT_HOPS {
                return attempt.error("too many redirects for a release asset");
            }
            if release_redirect_ok(attempt.url()) {
                attempt.follow()
            } else {
                attempt
                    .error("release asset redirected to a non-https or credentialed URL — refusing")
            }
        }));
    if is_loopback_host(base) {
        builder = builder.no_proxy();
    }
    builder.build().map_err(|e| HiggsError::UpdateFetchFailed {
        detail: format!("cannot build the HTTP client: {e}"),
    })
}

/// [`UpdateSource`] for a version-only update: all three asset URLs pre-derived from
/// the node's OWN configured release base ([`release_asset_urls`]) — the hub named a
/// version and nothing else.
pub struct VersionSource {
    manifest_url: reqwest::Url,
    sig_url: reqwest::Url,
    artifact_url: reqwest::Url,
    client: reqwest::blocking::Client,
}

impl VersionSource {
    pub fn new(release_base: &str, version: &str) -> Result<Self, HiggsError> {
        let (manifest_url, sig_url, artifact_url) = release_asset_urls(release_base, version)?;
        let client = build_release_client(&manifest_url)?;
        Ok(Self {
            manifest_url,
            sig_url,
            artifact_url,
            client,
        })
    }
}

impl UpdateSource for VersionSource {
    fn manifest(&self) -> Result<(Vec<u8>, String), HiggsError> {
        let manifest = fetch_bounded(
            &self.client,
            &self.manifest_url,
            MAX_MANIFEST_BYTES,
            "manifest",
        )?;
        let sig_bytes = fetch_bounded(&self.client, &self.sig_url, MAX_SIG_BYTES, "signature")?;
        let sig = String::from_utf8(sig_bytes).map_err(|_| HiggsError::UpdateManifestInvalid {
            detail: "the fetched signature is not UTF-8".into(),
        })?;
        Ok((manifest, sig))
    }

    fn artifact(&self, _file: &str) -> Result<Vec<u8>, HiggsError> {
        // Deliberately IGNORES the manifest's `file`: the artifact URL was derived
        // from the configured base + version + THIS build's suffix before any fetch,
        // so a hostile manifest cannot steer the download anywhere. If the manifest
        // names a different file, the sha256 check fails loudly right after.
        fetch_bounded(
            &self.client,
            &self.artifact_url,
            MAX_ARTIFACT_BYTES,
            "artifact",
        )
    }
}

/// Apply a hub-TRIGGERED version-only update (`M_NODE_UPDATE_VERSION`): the node
/// reads its OWN `release_url` from `config.json`, fetches + verifies + applies
/// `version` through the SAME pipeline as every other source. Mirrors
/// [`apply_pushed_update`] exactly (root refusal, lock, record/clear rules,
/// UPGRADE-ONLY) — only the source differs.
pub fn apply_version_update(bin: &Path, version: &str) -> Result<(String, String), HiggsError> {
    let running_version = env!("CARGO_PKG_VERSION");
    // SAFETY: geteuid has no preconditions and no failure mode.
    if unsafe { libc::geteuid() } == 0 {
        return Err(HiggsError::UpdateApplyFailed {
            detail: "refusing to apply a pushed update as root — the node runs as the \
                     unprivileged operator"
                .into(),
        });
    }
    let _lock = match UpdateLock::try_acquire(bin) {
        LockAttempt::Held(lock) => lock,
        LockAttempt::Contended => {
            return Err(HiggsError::UpdateApplyFailed {
                detail: format!(
                    "another self-update is already running (lock held on {})",
                    lock_path(bin).display()
                ),
            });
        }
        LockAttempt::Failed(e) => {
            record_update_failure(bin, running_version, version, &e.to_string());
            return Err(e);
        }
    };
    let mut authenticated_to: Option<String> = None;
    let staged = (|| {
        // The node's OWN config decides WHERE releases come from — never the hub.
        let release_base = crate::config::config_path()
            .ok()
            .and_then(|p| crate::config::InstanceConfig::load(&p).ok())
            .map(|c| c.release_url())
            .unwrap_or_else(|| crate::config::DEFAULT_RELEASE_URL.to_string());
        let source = VersionSource::new(&release_base, version)?;
        let running = installed_identity(bin);
        // The version bind (AUTHENTICATED manifest == the version the hub named) runs
        // inside `verify_and_check` via `expected_version`, BEFORE the artifact download —
        // a release origin serving a validly-signed different manifest at that path is
        // refused without spending the artifact bandwidth.
        let verified = verify_and_check(
            &source,
            &running,
            false,
            &mut authenticated_to,
            Some(version),
        )?;
        stage_and_flip(
            bin,
            &verified.manifest,
            &verified.artifact,
            false,
            &smoke_run,
        )?;
        Ok::<(String, String), HiggsError>((running.version, verified.manifest.version))
    })();
    match staged {
        Ok(pair) => Ok(pair),
        Err(e) => {
            record_update_failure(
                bin,
                running_version,
                authenticated_to.as_deref().unwrap_or(version),
                &e.to_string(),
            );
            Err(e)
        }
    }
}

/// Request a GRACEFUL restart of THIS daemon so a just-staged update ACTIVATES. `stage_and_flip`
/// only flips the `current` symlink to the new version; the running process keeps executing the
/// OLD binary until it restarts, so a successful push is a no-op without this. Wakes the node's
/// shutdown selector via a DEDICATED in-process trigger ([`await_node_shutdown`]) → the serve loop
/// breaks with cause [`ShutdownCause::Update`] → resident workers are drained then shut down
/// (`shutdown_all`) → the process RE-EXECS through `current` = the new version (or, if that fails,
/// exits for the service manager to restart), which the boot-guard then confirms (on hub admission)
/// or rolls back (crash-loop). The hub-push handler calls this after a successful
/// [`apply_pushed_update`] (having already replied `"accepted"`); the CLI `--url` path returns to
/// its own caller instead.
///
/// WHY A DEDICATED TRIGGER, NOT A SELF-`SIGTERM`: an operator/service-manager `SIGTERM` and a
/// self-`SIGTERM` are the SAME OS signal — indistinguishable at the signal layer. Inferring "was
/// this an update restart?" from a side-channel flag sampled later in the teardown lets a detached
/// update that flips DURING an operator stop reclassify that stop as an update restart (drain +
/// re-exec instead of a prompt exit — the operator's pid stays alive; a service manager's stop is
/// prolonged). A distinct trigger the daemon selects on makes the two causes structurally distinct
/// AT THE SOURCE: whichever future resolves the loop IS the cause, latched at the break — no late
/// global read, no ambiguity. [`await_node_shutdown`] is `biased` toward the operator signal, so a
/// genuinely simultaneous operator stop wins (its update, already staged + verified, simply
/// activates on the next start).
///
/// IN-FLIGHT WORK (P4b (c)): an [`ShutdownCause::Update`] teardown DRAINS — it waits (bounded) for
/// in-flight generations to finish before `shutdown_all`, then RE-EXECS so the staged binary
/// activates immediately (re-entering `main()` → the boot-guard) even without a service manager. An
/// operator/service-manager stop keeps the prompt stop-then-truncate behavior service managers expect.
pub fn request_self_restart() {
    tracing::info!("self-update: signalling a graceful restart to activate the staged update");
    // `notify_one` stores a permit if the daemon is not yet awaiting, so a request that races
    // ahead of the shutdown selector's first poll is NOT lost — the next `notified().await`
    // consumes it. Same process by construction (a detached tokio task here, the serve loop there).
    RESTART_NOTIFY.notify_one();
}

/// The cause that ended the node serve loop — decides the teardown. [`ShutdownCause::Update`]
/// (a [`request_self_restart`]) drains in-flight generations then re-execs through `current`;
/// [`ShutdownCause::Operator`] (a `SIGTERM`/`SIGINT` from an operator or service manager) is a
/// prompt stop: no drain, plain exit.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ShutdownCause {
    Operator,
    Update,
}

/// Await the node daemon's shutdown and report its CAUSE. `operator` is the eagerly-installed
/// OS-signal listener (`shutdown_listener`, whose handlers register synchronously at call time);
/// the update trigger is [`request_self_restart`]. `biased` gives the operator signal PRECEDENCE:
/// if both are ready in the same poll, the operator stop wins and no re-exec happens — an operator
/// who asked to stop is honored, and the staged update activates on the next start instead of
/// keeping this pid alive. Because the cause is whichever future resolves (not a flag sampled
/// later), a detached update finishing anywhere in the teardown cannot reclassify an operator stop.
pub async fn await_node_shutdown(operator: impl std::future::Future<Output = ()>) -> ShutdownCause {
    tokio::select! {
        biased;
        _ = operator => ShutdownCause::Operator,
        _ = RESTART_NOTIFY.notified() => ShutdownCause::Update,
    }
}

/// Run the update `drain` while STILL watching the operator signal, so an operator stop arriving
/// DURING the (bounded) drain overrides the re-exec — a monotonic `Update → Operator` transition.
/// The `operator` future is the SAME one the serve loop borrowed (it never resolved, because the
/// update won there): keeping one continuous observation means a SIGTERM delivered anywhere in the
/// teardown is seen, not swallowed by tokio's still-installed handler. `biased` toward the operator
/// so its arrival wins the poll. Returns `Some(drained)` when the drain finished on its own
/// (`drained == false` = it hit the deadline; either way the staged update still activates, so the
/// caller re-execs), or `None` when the operator overrode (the caller stops promptly, no re-exec).
pub async fn drain_with_operator_override<D: std::future::Future<Output = bool>>(
    operator: impl std::future::Future<Output = ()>,
    drain: D,
) -> Option<bool> {
    tokio::select! {
        biased;
        _ = operator => None,
        drained = drain => Some(drained),
    }
}

/// A NON-BLOCKING check of whether the operator signal has fired by now — polls `operator` once,
/// with a `ready` fallback so it never waits. This is the FINAL gate right before the update
/// re-exec: worker teardown (`shutdown_all`) must run to completion for BOTH an operator stop and
/// an update restart (workers get reaped either way), so there is nothing to abort during it — but
/// a SIGTERM/SIGINT delivered while it ran is buffered by tokio's installed handler, and polling
/// `operator` here CONSUMES it, cancelling the re-exec so an operator's stop is honored rather than
/// overridden by a restart onto `current`. Only call this while `operator` is still PENDING (it is,
/// whenever the drain override did not already fire) — polling a resolved future would panic.
pub async fn operator_stop_pending(operator: impl std::future::Future<Output = ()>) -> bool {
    tokio::select! {
        biased;
        _ = operator => true,
        _ = std::future::ready(()) => false,
    }
}

/// The dedicated in-process update-restart trigger. Process-local by construction — the requester
/// ([`request_self_restart`]) and the waiter ([`await_node_shutdown`]) are tasks of the same daemon.
static RESTART_NOTIFY: tokio::sync::Notify = tokio::sync::Notify::const_new();

// ---------------------------------------------------------------------------
// Single-update lock — flock on a FILE (not a dir -> no NFS RO-fd regression)
// ---------------------------------------------------------------------------

/// Held for the stage->flip critical section so two concurrent self-updates can't
/// interleave their `current` flips. `flock` on a FILE fd (writable) — never a
/// read-only DIR fd, whose fcntl emulation over NFS wrongly fails (the regression
/// that got the install-service lock reverted). Released on drop (fd close).
pub struct UpdateLock {
    _file: std::fs::File,
}

/// Ensure the update lock lives in a PRIVATE, ACL-free `0700` directory under `bin`, so
/// no peer can even TRAVERSE to the lock file — let alone open it and hold the lock. The
/// `0600` mode on the lock FILE is NOT sufficient on macOS: an inheritable `read` ACE on
/// `bin` (which the write-only bin-tree ACL check permits) propagates read access to the
/// newly-created lock file, and `flock(LOCK_EX)` succeeds on a read-only descriptor — so a
/// peer could open the lock read-only and hold it forever, making every boot hook see
/// false `Contended` and skip, so a bad trial's boots are never counted and it crash-loops
/// with NO rollback. Nesting the lock in a `0700` dir whose ACL we strip closes that: the
/// peer cannot traverse the dir, so it never reaches the lock file (which — created under
/// the now-ACL-free dir — also inherits no ACE). Idempotent: re-hardens every acquire, and
/// uses a NEW path (`.update-lock.d/`), so no lock a peer may already hold under the old
/// `bin/.update.lock` path is trusted. Any failure is returned (→ `Failed`, fail-safe).
fn open_private_lock_dir(bin: &Path) -> Result<(), HiggsError> {
    use std::os::unix::fs::DirBuilderExt;
    let dir = lock_dir(bin);
    match std::fs::DirBuilder::new().mode(0o700).create(&dir) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(e) => {
            return Err(HiggsError::UpdateApplyFailed {
                detail: format!("cannot create lock dir {}: {e}", dir.display()),
            });
        }
    }
    // Refuse a symlink/non-directory at the lock-dir path (never traverse through one).
    let m = std::fs::symlink_metadata(&dir).map_err(|e| HiggsError::UpdateApplyFailed {
        detail: format!("cannot stat lock dir {}: {e}", dir.display()),
    })?;
    if !m.file_type().is_dir() {
        return Err(HiggsError::UpdateApplyFailed {
            detail: format!("lock dir {} is not a directory — refusing", dir.display()),
        });
    }
    // Re-assert 0700 (bounds a permissive umask / fixes a pre-existing mode) and strip any
    // ACL so the `0700` is the ONLY thing granting access — no peer can traverse in.
    set_dir_mode_0700(&dir)?;
    strip_inherited_acls(&dir)?;
    Ok(())
}

/// The outcome of one NON-BLOCKING attempt to take the update lock, split so a
/// boot-guard hook can tell benign CONTENTION (another self-update legitimately holds
/// the lock — skip this boot, re-check next) apart from a HARD FAILURE of the locking
/// mechanism itself (the lock path could not be opened — `O_NOFOLLOW` rejecting a
/// planted `.update.lock` symlink, `EACCES` — a non-regular lock node, or `flock`
/// erroring for a reason OTHER than contention, e.g. unsupported on a network FS).
/// Collapsing the two — as a bare `acquire().is_ok()` did — let a hard lock failure
/// SILENTLY disable crash-loop rollback, stranding a bad binary in an endless restart.
pub enum LockAttempt {
    /// Acquired — hold it until the guarded mutation completes.
    Held(UpdateLock),
    /// Another self-update holds the lock (`EWOULDBLOCK`/`EAGAIN`) — benign contention.
    Contended,
    /// The lock subsystem itself failed. NOT contention; the caller must NOT treat it
    /// as "an apply is running".
    Failed(HiggsError),
}

impl UpdateLock {
    /// Acquire the exclusive, NON-BLOCKING update lock, or fail — the apply/rollback/
    /// prune entry point, which must NOT proceed without the lock. Contention (HG087,
    /// another self-update is running) and a hard lock-subsystem failure both surface as
    /// an error: an apply refusing to run in EITHER case is correct, and — crucially —
    /// this fail-closed apply path is WHY a boot hook may safely run lock-free on a hard
    /// failure (no apply can be concurrently mutating the tree; see [`Self::try_acquire`]).
    pub fn acquire(bin: &Path) -> Result<Self, HiggsError> {
        match Self::try_acquire(bin) {
            LockAttempt::Held(lock) => Ok(lock),
            LockAttempt::Contended => Err(HiggsError::UpdateApplyFailed {
                detail: format!(
                    "another self-update is already running (lock held on {})",
                    lock_path(bin).display()
                ),
            }),
            LockAttempt::Failed(e) => Err(e),
        }
    }

    /// One non-blocking attempt, reporting CONTENTION vs a HARD FAILURE distinctly (see
    /// [`LockAttempt`]). This is the primitive; [`Self::acquire`] folds both non-`Held`
    /// arms into an error for the apply path.
    pub fn try_acquire(bin: &Path) -> LockAttempt {
        use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
        // Ensure the lock's PRIVATE parent dir before opening it — a chmod/ACL/create
        // failure surfaces as `Failed` (the apply refuses; a boot hook runs lock-free and
        // still recovers), never a silent skip.
        if let Err(e) = open_private_lock_dir(bin) {
            return LockAttempt::Failed(e);
        }
        let path = lock_path(bin);
        let file = match std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            // 0600 owner-only: on a NATIVE Linux filesystem `flock(LOCK_EX)` works on a
            // READ-only fd, so a world-readable lock would let ANY local user who can
            // traverse `bin` take the lock and both block updates AND suppress crash-loop
            // rollback. An owner-only file a peer cannot even open closes that DoS.
            .mode(0o600)
            // O_NOFOLLOW: a `.update.lock -> v1.0.0/higgs` symlink (planted, or a bug)
            // must NOT be followed — else `open` + the fchmod below would operate on the
            // rollback BINARY, chmod'ing it to 0600 and destroying the rollback target.
            // The refusal surfaces as `Failed`, NOT silent contention.
            .custom_flags(libc::O_NOFOLLOW)
            .open(&path)
        {
            Ok(f) => f,
            Err(e) => {
                return LockAttempt::Failed(HiggsError::UpdateApplyFailed {
                    detail: format!("cannot open update lock {}: {e}", path.display()),
                });
            }
        };
        // Reject a non-REGULAR lock node (a fifo/device/dir a peer planted) — O_NOFOLLOW
        // already refused a symlink; this refuses the rest. Also a hard `Failed`.
        let meta = match file.metadata() {
            Ok(m) => m,
            Err(e) => {
                return LockAttempt::Failed(HiggsError::UpdateApplyFailed {
                    detail: format!("cannot stat update lock {}: {e}", path.display()),
                });
            }
        };
        if !meta.file_type().is_file() {
            return LockAttempt::Failed(HiggsError::UpdateApplyFailed {
                detail: format!(
                    "update lock {} is not a regular file — refusing",
                    path.display()
                ),
            });
        }
        // Tighten a pre-existing operator-owned lock too — but THROUGH THE OPENED FD
        // (`File::set_permissions` = fchmod), never a path-based chmod that would follow
        // a symlink we just refused to open. Best-effort (a file we don't own can't be
        // chmod'd, which is itself fine).
        let _ = file.set_permissions(std::fs::Permissions::from_mode(0o600));
        // SAFETY: flock is a bare syscall on a valid fd we own; LOCK_NB makes it
        // return EWOULDBLOCK instead of blocking when another run holds it.
        let rc = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        if rc == 0 {
            return LockAttempt::Held(Self { _file: file });
        }
        let err = std::io::Error::last_os_error();
        // ONLY EWOULDBLOCK/EAGAIN is genuine contention (another run holds it). Any other
        // errno — EOPNOTSUPP/EINVAL on an FS whose flock is broken/unsupported, EINTR,
        // ENOLCK — is a hard failure: NOT "an apply is running", so a boot hook must not
        // silently no-op on it.
        match err.raw_os_error() {
            Some(code) if code == libc::EWOULDBLOCK || code == libc::EAGAIN => {
                LockAttempt::Contended
            }
            _ => LockAttempt::Failed(HiggsError::UpdateApplyFailed {
                detail: format!("cannot lock {} (flock failed): {err}", path.display()),
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// Stage + smoke + flip — the apply, over an injected bin dir + smoke runner
// ---------------------------------------------------------------------------

/// How the post-stage `--version` smoke test is run — injected so tests drive the
/// flip logic without spawning a real binary. Production is [`smoke_run`].
pub type SmokeRunner<'a> = dyn Fn(&Path) -> Result<String, HiggsError> + 'a;

/// Stage the verified artifact into `v<ver>/`, prove it runs, then atomically flip
/// `current` and record the trial. Steps, in order, each failing CLOSED:
/// 1. refuse a peer-subvertible `bin` tree (mode + macOS ACL, both chains, up to `/`);
///    refuse STACKING on an unconfirmed trial; RE-CHECK eligibility against the
///    currently-installed version/variant UNDER THE LOCK;
/// 2. unpack the tarball IN-PROCESS into a random staging dir under `bin`, forced 0700;
/// 3. require a REGULAR, executable `higgs` (`-f && !-h && -x`, mirroring install.sh)
///    and NORMALIZE the staged modes (binary 0755, no setuid/world-write);
/// 4. SMOKE the STAGED binary `--version` and require it reports `<ver>` — BEFORE any
///    publish, so a bad binary never overwrites an existing `v<ver>` dir;
/// 5. publish into `v<ver>/` (forced 0755) by per-file `rename(2)` after fsync'ing each
///    staged file; drop the `.variant` marker; fsync the version dir;
/// 6. write the trial marker (recording the outgoing `current` as the rollback target),
///    then atomically flip `current` to `v<ver>` and fsync the bin dir.
///
/// The lock is the caller's responsibility (held across this whole call).
pub fn stage_and_flip(
    bin: &Path,
    manifest: &UpdateManifest,
    artifact: &[u8],
    allow_downgrade: bool,
    smoke: &SmokeRunner,
) -> Result<(), HiggsError> {
    let ver = &manifest.version;
    // NOTE: we do NOT touch the process `umask` here. This runs both as the one-shot CLI AND
    // (via the M_UPDATE hub-push) INSIDE the long-lived node daemon, whose `umask` must not be
    // mutated (a leaked `umask 022` would make a later `M_NODE_PULL` create a private model
    // 0644 instead of 0600). Every dir/file created below is given an EXPLICIT mode via a
    // `chmod` that is NOT umask-masked (a bare `DirBuilder.mode`/`OpenOptions.mode` IS masked,
    // so each is followed by an explicit `chmod` to the exact mode): the staging dir `0700`
    // (`unpack_tar_gz`), the version dir `0755` (below), the binary `0755` + other files `0644`
    // (`normalize_published_modes`), and the markers `0600` (`write_atomic` fchmod) — so nothing
    // depends on the ambient umask, in EITHER direction (too-permissive OR too-restrictive).

    // Before writing anything: refuse a peer-subvertible bin tree (mode + macOS ACL,
    // lexical + resolved, up to `/`) — the SAME hardened walk the install-service exec
    // path uses. A peer-writable bin/ancestor could swap the staged binary or repoint
    // `current` before the operator restarts.
    crate::node::cli::refuse_unsafe_operator_bin_tree(bin).map_err(|e| {
        HiggsError::UpdateApplyFailed {
            detail: format!("unsafe install directory: {e}"),
        }
    })?;

    // Refuse to STACK an update on a genuinely-pending UNCONFIRMED trial: if a prior
    // self-update flipped `current` but has not yet been confirmed (restarted → admitted
    // / N s alive) or rolled back, applying another would record the UNCONFIRMED version
    // as the rollback target and DISCARD the last known-good one. A trial whose `to` no
    // longer matches `current`, however, is a STALE orphan (the apply crashed AFTER
    // writing the marker but BEFORE the flip), which must NOT wedge future updates — clear
    // it and proceed.
    if let Some(t) = read_trial(bin) {
        if Some(t.to.as_str()) == current_target(bin).as_deref() {
            return Err(HiggsError::UpdateApplyFailed {
                detail: "a previous self-update is still on trial (unconfirmed) — restart the \
                         service to confirm it, or run `higgs node self-update --rollback`, before \
                         applying another update"
                    .into(),
            });
        }
        // Stale marker from a crashed mid-apply (flip never happened): discard it.
        remove_if_present(&trial_marker_path(bin))?;
        remove_if_present(&boot_fail_path(bin))?;
    }

    // Re-check eligibility UNDER THE LOCK against the version/variant CURRENTLY installed
    // — the CLI's pre-lock check can go stale (a concurrent update flipped `current`
    // between that check and this lock), which would otherwise let a signed older
    // manifest flip `current` backwards without `--allow-downgrade`.
    evaluate_eligibility(&installed_identity(bin), manifest, allow_downgrade)?;

    // REFUSE a version that already FAILED its boot trial (crash-looped → rolled back). Without
    // this a hub re-pushing the same crash-looping release — e.g. after a lost `"accepted"`
    // reply, or a fleet re-pushing "latest" — would loop apply→crash→rollback→apply instead of
    // resting on the known-good previous version. The poison is BEST-EFFORT (recorded after the
    // rollback flip, to never jeopardise that critical recovery), so this is EVENTUALLY-consistent:
    // a re-push of a poisoned version is refused HERE; if the poison was ever lost (a crash in the
    // rollback's flip→poison window), the re-push re-applies and SELF-HEALS via the same rollback
    // rather than looping unboundedly. A deliberate re-install (`install.sh`) clears the poison so
    // a fixed rebuild of the same version can still be applied.
    if is_failed_target(bin, &version_name(&manifest.version)) {
        return Err(HiggsError::UpdateApplyFailed {
            detail: format!(
                "version {} previously failed its boot trial and was rolled back — refusing to \
                 re-apply it (clear via a fresh install.sh)",
                manifest.version
            ),
        });
    }

    // (2) Unpack into a fresh staging dir under bin, created 0700 (owner-only) and kept
    // that way for the WHOLE stage — so NO other user can traverse into it while the
    // archive is being extracted (a setuid-operator `higgs` extracted early, before the
    // normalize below, would otherwise be executable by a peer mid-unpack) NOR after. The
    // cleanup guard is armed BEFORE the unpack, so a mid-unpack failure (ENOSPC/quota)
    // still removes the partially-extracted `.update-stage.*` rather than leaving litter.
    let stage = bin.join(temp_name(".update-stage"));
    let _stage_guard = RemoveOnDrop(stage.clone());
    unpack_tar_gz(artifact, &stage)?;
    set_dir_mode_0700(&stage)?;

    // (3) Require a regular, executable `higgs`, then ENFORCE a flat archive + NORMALIZE
    // all staged modes (binary 0755, others 0644; no setuid/world-write; a subdir/symlink
    // entry is refused) — the archive is CI-signed but its modes/layout are not policy.
    let staged_bin = stage.join("higgs");
    require_regular_exec(&staged_bin)?;
    normalize_published_modes(&stage)?;

    // (4) SMOKE the staged binary BEFORE publishing — so a binary that fails to run is
    // never renamed over an existing `v<ver>` dir (which could be the live `current`
    // or the rollback `prev`). The published bytes are identical (a rename), so
    // smoking the staged copy proves the published one runs.
    let reported = smoke(&staged_bin)?;
    if !reported.split_whitespace().any(|tok| tok == ver.as_str()) {
        return Err(HiggsError::UpdateApplyFailed {
            detail: format!(
                "staged {} reported version {reported:?}, expected {ver:?} — refusing to publish",
                staged_bin.display()
            ),
        });
    }

    // (5) Publish into v<ver>/ by per-file rename (only now that smoke passed).
    let vdir = version_dir(bin, ver);
    if std::fs::symlink_metadata(&vdir).is_ok_and(|m| m.file_type().is_symlink()) {
        return Err(HiggsError::UpdateApplyFailed {
            detail: format!(
                "{} is a symlink — refusing to publish through it",
                vdir.display()
            ),
        });
    }
    // Create with an EXPLICIT 0755 (via `DirBuilder.mode`, NOT the ambient umask) so a NEW
    // version dir never has a create→chmod window at a permissive mode. `recursive(true)` is a
    // no-op on an existing dir (leaving its mode for the `set_dir_mode_0755` below to fix).
    {
        use std::os::unix::fs::DirBuilderExt;
        std::fs::DirBuilder::new()
            .mode(0o755)
            .recursive(true)
            .create(&vdir)
            .map_err(|e| HiggsError::UpdateApplyFailed {
                detail: format!("cannot create version dir {}: {e}", vdir.display()),
            })?;
    }
    // Force 0755 — fixes a PRE-EXISTING world/group-writable `v<ver>` dir (the create above is
    // a no-op on an existing dir, so its mode would otherwise stand, letting a peer swap the
    // published binary through it).
    set_dir_mode_0755(&vdir)?;
    // …AND refuse a macOS write-granting ACL on a PRE-EXISTING version dir — `chmod 0755`
    // above does NOT strip an extended ACL, so a peer granted `add_file,delete_child` on
    // it could still swap the published binary before restart.
    crate::node::cli::refuse_writable_acl_on(&vdir, "version dir").map_err(|e| {
        HiggsError::UpdateApplyFailed {
            detail: format!("unsafe version dir: {e}"),
        }
    })?;
    for entry in std::fs::read_dir(&stage).map_err(io_apply("reading staging dir"))? {
        let entry = entry.map_err(io_apply("reading staging entry"))?;
        let src = entry.path();
        // Flush each staged file's DATA before it goes live, so a crash right after
        // the flip can't leave `current/higgs` on an unflushed/corrupt inode.
        fsync_path(&src)?;
        let dest = vdir.join(entry.file_name());
        std::fs::rename(&src, &dest).map_err(|e| HiggsError::UpdateApplyFailed {
            detail: format!("publishing {} failed: {e}", dest.display()),
        })?;
    }
    // (6) Variant marker + flush the version dir's new entries to disk.
    write_atomic(&variant_marker(bin, ver), manifest.variant.as_bytes())?;
    fsync_path(&vdir)?;

    // (7) FINAL eligibility recheck against the CURRENT installed version, immediately
    // before recording `prev` and flipping — this narrows to a couple of instructions
    // the window in which a tool that does NOT share the update lock (a concurrent
    // `install.sh`, which has no cross-platform flock — macOS lacks `flock(1)`) could
    // have flipped `current` forward after the earlier under-lock check, letting us
    // flip it BACKWARD without `--allow-downgrade`. If `current` moved to make the
    // manifest ineligible now, abort rather than downgrade.
    evaluate_eligibility(&installed_identity(bin), manifest, allow_downgrade)?;

    // Trial marker (records the rollback target) then the atomic flip, each flushed so a
    // post-crash boot always finds a consistent `current` + marker.
    let prev = current_target(bin);
    let marker = TrialMarker {
        to: version_name(ver),
        prev,
    };
    let marker_json = serde_json::to_vec(&marker).map_err(|e| HiggsError::UpdateApplyFailed {
        detail: format!("cannot serialize trial marker: {e}"),
    })?;
    write_atomic(&trial_marker_path(bin), &marker_json)?;
    remove_if_present(&boot_fail_path(bin))?;
    // flip_current_to fsyncs the bin dir, so the flipped `current` + the trial marker
    // are durable together the moment this returns.
    flip_current_to(bin, &version_name(ver))?;
    // The update STAGED and `current` flipped — every prior self-update FAILURE is now moot. Clear
    // the last-failure marker durably here (BOTH the hub-push `apply_pushed_update` and the manual
    // `higgs node self-update` CLI reach the flip through this one function, so this is the single
    // point that guarantees a success leaves no stale marker to resurrect on a later downgrade).
    clear_update_failure(bin);
    Ok(())
}

/// Production smoke runner: run `<path> --version` the way the SERVICE will —
/// `env_clear` + pinned `TRUSTED_PATH` (strips loader vars like `LD_PRELOAD`/`DYLD_*`
/// and any transient shell credentials), in its OWN session so a `--version` handler
/// that forks descendants is reaped as a group, bounded by [`SMOKE_TIMEOUT`]. Returns
/// the trimmed stdout (`higgs <ver>`), or an error if it cannot run / times out.
///
/// stdout is drained on a SEPARATE THREAD so the read can never make `smoke_run`
/// outlast the timeout: a `--version` handler that DETACHES a child (`setsid …`) which
/// keeps the stdout pipe open cannot be reaped by the process-group kill, and would
/// block an inline `read_to_string` on that pipe forever. Instead the reader thread
/// (which may block on the detached holder — a leaked thread on a FAILED update, not a
/// hang of the updater) delivers the output over a channel, and we wait for it with a
/// small bounded grace; if it does not arrive, smoke FAILS (the version can't be
/// confirmed) rather than hanging while holding the update lock.
pub fn smoke_run(path: &Path) -> Result<String, HiggsError> {
    use std::os::unix::process::CommandExt;
    let mut cmd = std::process::Command::new(path);
    cmd.arg("--version")
        .env_clear()
        .env("PATH", TRUSTED_PATH)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null());
    // SAFETY: the closure runs in the forked child and calls only setsid, a bare
    // async-signal-safe syscall — no allocation, no lock inheritance risk.
    unsafe {
        cmd.pre_exec(|| {
            if libc::setsid() == -1 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
    let mut child = cmd.spawn().map_err(|e| HiggsError::UpdateApplyFailed {
        detail: format!("cannot run staged binary {} --version: {e}", path.display()),
    })?;
    let pid = child.id() as libc::pid_t;
    // Reap the whole group on EVERY exit path (a `--version` that forks a same-group
    // `sleep &` then exits would otherwise orphan it). SAFETY: bare kill syscall; a
    // negative pid targets the process group; ESRCH (already gone) is harmless.
    let reap_group = || unsafe {
        libc::kill(-pid, libc::SIGKILL);
    };
    // Drain stdout on a worker thread → channel, so the read cannot block smoke_run
    // past the deadline (see the fn doc: a detached descendant may keep the pipe open).
    let (tx, rx) = std::sync::mpsc::channel::<String>();
    if let Some(mut so) = child.stdout.take() {
        std::thread::spawn(move || {
            let mut out = String::new();
            let _ = so.read_to_string(&mut out);
            let _ = tx.send(out);
        });
    }

    // Wait for the child to exit, bounded by SMOKE_TIMEOUT.
    let deadline = std::time::Instant::now() + SMOKE_TIMEOUT;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => {
                if std::time::Instant::now() >= deadline {
                    reap_group();
                    let _ = child.wait();
                    return Err(HiggsError::UpdateApplyFailed {
                        detail: format!(
                            "staged binary {} --version did not finish within {}s",
                            path.display(),
                            SMOKE_TIMEOUT.as_secs()
                        ),
                    });
                }
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
            Err(e) => {
                reap_group();
                let _ = child.wait();
                return Err(HiggsError::UpdateApplyFailed {
                    detail: format!("waiting on staged --version failed: {e}"),
                });
            }
        }
    };
    // Child exited — reap any same-group descendants (closes inherited pipe ends so a
    // well-behaved reader EOFs promptly), then collect the output with a BOUNDED wait.
    reap_group();
    let _ = child.wait();
    if !status.success() {
        return Err(HiggsError::UpdateApplyFailed {
            detail: format!(
                "staged binary {} --version exited with {status}",
                path.display()
            ),
        });
    }
    match rx.recv_timeout(std::time::Duration::from_secs(2)) {
        Ok(out) => Ok(out.trim().to_string()),
        // The reader is still blocked (a detached descendant holds stdout) → we cannot
        // confirm the version. Fail rather than hang; the reader thread is abandoned.
        Err(_) => Err(HiggsError::UpdateApplyFailed {
            detail: format!(
                "could not read {} --version output within 2s (a detached child may be holding \
                 stdout) — refusing to trust the update",
                path.display()
            ),
        }),
    }
}

// ---------------------------------------------------------------------------
// Rollback + prune — no-signature maintenance ops
// ---------------------------------------------------------------------------

/// READ-ONLY: resolve the rollback target (`trial.prev`) and verify it is installed,
/// WITHOUT changing anything. Powers both `--dry-run --rollback` (report only) and
/// [`rollback`] (which then flips). Refuses when no previous is recorded or its dir
/// is missing.
pub fn rollback_target(bin: &Path) -> Result<String, HiggsError> {
    if !is_managed_install(bin) {
        return Err(HiggsError::UpdateApplyFailed {
            detail: format!(
                "{} is not a higgs install (no bin/current symlink) — nothing to roll back",
                bin.display()
            ),
        });
    }
    let trial = read_trial(bin).ok_or_else(|| HiggsError::UpdateApplyFailed {
        detail: "no update is on trial — nothing to roll back (re-install to change versions)"
            .into(),
    })?;
    // A trial whose `to` no longer matches `current` is STALE — a manual `install.sh`
    // (or a prior rollback) already superseded it. Undoing it would revert the
    // operator's repair, so refuse (the boot + apply paths reject it the same way).
    if Some(trial.to.as_str()) != current_target(bin).as_deref() {
        return Err(HiggsError::UpdateApplyFailed {
            detail: format!(
                "the recorded trial (for {}) is stale — `current` is now a different version, so \
                 a manual install/rollback already superseded it; nothing to roll back",
                trial.to
            ),
        });
    }
    let prev = trial.prev.ok_or_else(|| HiggsError::UpdateApplyFailed {
        detail: "no previous version recorded — nothing to roll back to (re-install to change \
                 versions)"
            .into(),
    })?;
    if !is_installed_version(bin, &prev) {
        return Err(HiggsError::UpdateApplyFailed {
            detail: format!(
                "rollback target {prev} is not installed under {} (no v-dir with a higgs binary)",
                bin.display()
            ),
        });
    }
    Ok(prev)
}

/// Manually repoint `current` back to the trial marker's `prev` (validated by
/// [`rollback_target`]) and clear the trial state. This is the operator's
/// `--rollback`; the boot-guard does the same automatically on a crash loop. The
/// caller holds the update lock across this call.
pub fn rollback(bin: &Path) -> Result<String, HiggsError> {
    // Capture the version being rolled back FROM before the trial is cleared, so it can be
    // POISONED like the boot-guard's auto-rollback (else the operator manually rolls back v2,
    // the trial is cleared, and a hub re-push of v2 would re-apply it).
    let failed_to = read_trial(bin).map(|t| t.to);
    let prev = rollback_target(bin)?;
    flip_current_to(bin, &prev)?;
    remove_if_present(&trial_marker_path(bin))?;
    remove_if_present(&boot_fail_path(bin))?;
    if let Some(failed) = failed_to {
        record_failed_target(bin, &failed); // best-effort, after the flip (see the fn doc)
    }
    Ok(prev)
}

/// READ-ONLY: the sorted `v*` version dirs [`prune`] WOULD remove — every real
/// version dir except the one `current` points at and the trial marker's `prev`
/// (the rollback target). Powers both `--dry-run --prune` and [`prune`]. Never a
/// symlink (that's `current`) and never a kept version.
pub fn prune_plan(bin: &Path) -> Result<Vec<String>, HiggsError> {
    if !is_managed_install(bin) {
        return Err(HiggsError::UpdateApplyFailed {
            detail: format!(
                "{} is not a higgs install (no bin/current symlink) — refusing to prune",
                bin.display()
            ),
        });
    }
    // Keep set, compared CASE-INSENSITIVELY: on a case-INSENSITIVE fs (default macOS
    // APFS) `current -> v1.0.0-rc1` can resolve to an on-disk dir named `v1.0.0-RC1`
    // (mkdir preserves the first-seen case). A case-sensitive keep check would not
    // recognize the live dir and would DELETE it. Store keeps lowercased.
    let mut keep = std::collections::HashSet::new();
    if let Some(cur) = current_target(bin) {
        keep.insert(cur.to_ascii_lowercase());
    }
    if let Some(prev) = read_trial(bin).and_then(|t| t.prev) {
        keep.insert(prev.to_ascii_lowercase());
    }
    let mut plan = Vec::new();
    for entry in std::fs::read_dir(bin).map_err(io_apply("reading bin dir"))? {
        let entry = entry.map_err(io_apply("reading bin entry"))?;
        let name = entry.file_name().to_string_lossy().into_owned();
        // STRICT: only genuine `v<semver>` version dirs, never an unrelated `v…` name;
        // never a symlink (that's `current`) and never a kept version (case-insensitive).
        if !is_version_dir_name(&name) || keep.contains(&name.to_ascii_lowercase()) {
            continue;
        }
        if !entry
            .file_type()
            .map_err(io_apply("stat bin entry"))?
            .is_dir()
        {
            continue;
        }
        plan.push(name);
    }
    plan.sort();
    Ok(plan)
}

/// Remove the version dirs [`prune_plan`] selects. Returns the names pruned. Never
/// touches `current`, the markers, or the lock file. The caller holds the update lock.
pub fn prune(bin: &Path) -> Result<Vec<String>, HiggsError> {
    let plan = prune_plan(bin)?;
    for name in &plan {
        let dir = bin.join(name);
        std::fs::remove_dir_all(&dir).map_err(|e| HiggsError::UpdateApplyFailed {
            detail: format!("cannot prune {}: {e}", dir.display()),
        })?;
    }
    Ok(plan)
}

// ---------------------------------------------------------------------------
// Low-level filesystem primitives (in-process, no shell-out)
// ---------------------------------------------------------------------------

/// Atomically flip `<bin>/current` to `v<ver>` — a temp symlink (RANDOM name, never
/// a predictable one an interrupted run could leave for reuse) built in the SAME dir,
/// verified to point where intended, then `rename(2)` over `current`. `rename` never
/// follows the destination symlink and either fully succeeds or changes nothing.
fn flip_current_to(bin: &Path, vname: &str) -> Result<(), HiggsError> {
    let tmp = bin.join(temp_name(".current.tmp"));
    // A stale temp name is unexpected (random), but never descend into one: remove
    // any pre-existing node first so symlink() can't fail-or-nest.
    let _ = std::fs::remove_file(&tmp);
    std::os::unix::fs::symlink(vname, &tmp).map_err(|e| HiggsError::UpdateApplyFailed {
        detail: format!("cannot stage current flip at {}: {e}", tmp.display()),
    })?;
    // Prove the staged link points where we intend before it replaces current.
    match std::fs::read_link(&tmp) {
        Ok(t) if t.as_os_str() == std::ffi::OsStr::new(vname) => {}
        other => {
            let _ = std::fs::remove_file(&tmp);
            return Err(HiggsError::UpdateApplyFailed {
                detail: format!(
                    "staged flip link {} does not point at {vname}: {other:?}",
                    tmp.display()
                ),
            });
        }
    }
    std::fs::rename(&tmp, current_link(bin)).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        HiggsError::UpdateApplyFailed {
            detail: format!("atomic flip of {} failed: {e}", current_link(bin).display()),
        }
    })?;
    // Flush the flipped `current` entry so a crash right after cannot resurrect the old
    // target. BEST-EFFORT (warn, don't fail): the rename is the point of no return —
    // `current` ALREADY names the new target, so returning an error here would falsely
    // report "flip failed / previous install still live" while the flip in fact took
    // effect, and a later restart would run the "failed" update (HG087 promises the
    // opposite). A fsync failure now is only a durability (not-flushed) concern.
    if let Err(e) = fsync_path(bin) {
        tracing::warn!(error = %e, "higgs self-update: current flipped but its dir fsync failed (not confirmed durable)");
    }
    Ok(())
}

/// Enforce a FLAT archive and normalize the modes of everything staged for publication:
/// the `higgs` binary to `0o755` (rwxr-xr-x), any other file to `0o644` (no setuid/setgid/
/// sticky, no group/other write). The release artifact is a flat set of regular files
/// (`install.sh` publishes `stage/*` the same way), so a SUBDIRECTORY or a SYMLINK entry
/// is unexpected and REFUSED — this closes the gap where a nested `tools/helper` `04755`
/// or a `tools/` `0777` would survive publication un-normalized (the top-level-only walk
/// missed them). The archive is CI-signed but its mode bits + layout are not trusted.
fn normalize_published_modes(stage: &Path) -> Result<(), HiggsError> {
    use std::os::unix::fs::PermissionsExt;
    for entry in std::fs::read_dir(stage).map_err(io_apply("reading staging dir"))? {
        let entry = entry.map_err(io_apply("reading staging entry"))?;
        let ft = entry.file_type().map_err(io_apply("stat staging entry"))?;
        if !ft.is_file() {
            return Err(HiggsError::UpdateApplyFailed {
                detail: format!(
                    "unexpected update archive layout: {} is not a regular file (the release \
                     artifact is a flat set of regular files) — refusing",
                    entry.path().display()
                ),
            });
        }
        let mode = if entry.file_name() == std::ffi::OsStr::new("higgs") {
            0o755
        } else {
            0o644
        };
        std::fs::set_permissions(entry.path(), std::fs::Permissions::from_mode(mode)).map_err(
            |e| HiggsError::UpdateApplyFailed {
                detail: format!("cannot normalize mode of {}: {e}", entry.path().display()),
            },
        )?;
    }
    Ok(())
}

/// Unpack a gzip'd tar into `dest`, which is created FRESH as `0700` (owner-only) so no
/// peer can traverse it WHILE extraction is in progress — a setuid `higgs` extracted
/// early (before the later normalize) is unreachable to anyone but the operator. Created
/// with `create_new` (fails if it already exists — the caller uses a random temp name).
/// `tar`'s unpack refuses any entry whose path escapes `dest` (`..`/absolute); modes are
/// re-normalized by [`normalize_published_modes`] before publication, so a hostile mode
/// in the archive cannot survive; we never follow the archive outside `dest`.
fn unpack_tar_gz(bytes: &[u8], dest: &Path) -> Result<(), HiggsError> {
    use std::os::unix::fs::DirBuilderExt;
    std::fs::DirBuilder::new()
        .mode(0o700)
        .create(dest)
        .map_err(io_apply("creating staging dir"))?;
    // Force EXACTLY 0700 immediately — `DirBuilder.mode(0700)` above is masked by the ambient
    // umask (`0700 & ~umask`), so a daemon running with a restrictive umask (e.g. 0700 → dir
    // created 000, or 0400 → 0300) would leave the dir unwritable/unreadable and the `ar.unpack`
    // below would fail EACCES. `chmod` is NOT umask-masked, so this pins owner-rwx regardless.
    // Do it BEFORE the ACL strip + extraction so the dir is genuinely usable from the first write.
    set_dir_mode_0700(dest)?;
    // The `0700` mode alone does NOT make the staging dir private on macOS: if `bin`
    // carries an INHERITABLE ACL (e.g. an admin granted a peer `list,search` with
    // `directory_inherit`), mkdir copies that ACE onto this new dir and macOS evaluates
    // ACLs BEFORE the mode bits — so the peer could traverse the "private" dir and
    // execute a setuid-operator `higgs` that `set_preserve_permissions` materialises early
    // in the unpack, BEFORE `normalize_published_modes` strips the setuid bit. Strip any
    // inherited ACL so `0700` genuinely means owner-only. Fail CLOSED — a staging dir we
    // cannot prove private must not receive a possibly-setuid extraction.
    strip_inherited_acls(dest)?;
    let gz = flate2::read::GzDecoder::new(bytes);
    let mut ar = tar::Archive::new(gz);
    ar.set_preserve_permissions(true);
    ar.set_overwrite(true);
    ar.unpack(dest).map_err(|e| HiggsError::UpdateApplyFailed {
        detail: format!("unpacking update tarball failed: {e}"),
    })?;
    Ok(())
}

/// Remove every extended ACL from `dir` (macOS `chmod -N`), so its `0o700` mode is the
/// ONLY thing granting access — no ACE inherited from `bin` can let a peer traverse the
/// private staging dir (see [`unpack_tar_gz`]). Fail CLOSED: a non-zero/failed `chmod`
/// means we could not prove the dir private, so the caller must abort rather than extract
/// a possibly-setuid archive into it. No-op off macOS — the operator-bin-tree ACL model
/// (and thus this mitigation) is macOS-only; POSIX default-ACL inheritance on Linux is a
/// separate, currently-unmodelled concern (documented residual).
fn strip_inherited_acls(dir: &Path) -> Result<(), HiggsError> {
    if !cfg!(target_os = "macos") {
        return Ok(());
    }
    // `dir` is our own just-created random `.update-stage.*` path — no shell is involved
    // (Command args are passed directly, no metacharacter/injection surface).
    let status = std::process::Command::new("/bin/chmod")
        .arg("-N")
        .arg(dir)
        .status()
        .map_err(|e| HiggsError::UpdateApplyFailed {
            detail: format!("cannot strip ACLs from staging dir {}: {e}", dir.display()),
        })?;
    if !status.success() {
        return Err(HiggsError::UpdateApplyFailed {
            detail: format!(
                "refusing to unpack into {}: could not strip inherited ACLs (chmod -N: {status})",
                dir.display()
            ),
        });
    }
    Ok(())
}

/// Force a directory's mode: `0o700` for the private staging dir (owner-only, so no
/// peer can traverse it during unpack), `0o755` for the published version dir (world/
/// group read+exec — the service must traverse it). Bounds a permissive caller umask
/// and fixes a pre-existing wrong mode.
fn set_dir_mode(dir: &Path, mode: u32) -> Result<(), HiggsError> {
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(dir, std::fs::Permissions::from_mode(mode)).map_err(|e| {
        HiggsError::UpdateApplyFailed {
            detail: format!("cannot set mode on {}: {e}", dir.display()),
        }
    })
}
fn set_dir_mode_0700(dir: &Path) -> Result<(), HiggsError> {
    set_dir_mode(dir, 0o700)
}
fn set_dir_mode_0755(dir: &Path) -> Result<(), HiggsError> {
    set_dir_mode(dir, 0o755)
}

/// Flush a path's data/entries to disk (fsync). Used for both files (the staged
/// binary) and directories (the version dir + bin dir, so a rename survives a crash).
/// A fsync FAILURE is propagated (HG087) — the updater must not report a durable
/// success when the backing store (a FUSE/NFS prefix returning `EIO`) could not
/// confirm the write. macOS `fsync` is not a full power-loss barrier (that needs
/// `F_FULLFSYNC`), matching install.sh's best-effort `sync`, but it still returns
/// success here, so propagating an ERROR only fires on a genuine flush failure.
fn fsync_path(path: &Path) -> Result<(), HiggsError> {
    let f = std::fs::File::open(path).map_err(|e| HiggsError::UpdateApplyFailed {
        detail: format!("cannot open {} to flush: {e}", path.display()),
    })?;
    f.sync_all().map_err(|e| HiggsError::UpdateApplyFailed {
        detail: format!(
            "cannot flush {} to disk ({e}) — refusing an un-durable update",
            path.display()
        ),
    })
}

/// Require `path` to be a REGULAR file (not a dir, not a symlink) that the CALLING
/// process — the operator running the node / self-update — can actually EXECUTE. The
/// shape a runnable `current/higgs` needs (mirrors install.sh's `-f && !-h && -x`).
///
/// Executability is checked with `access(X_OK)`, NOT merely "some `0o111` bit set": a
/// root-owned `0700` binary has an exec bit yet an unprivileged operator gets `EACCES`
/// — repointing `current` at it would wedge the service on `EACCES`. `access` answers
/// "can THIS caller exec it" against the file's owner/group/mode relative to the caller.
fn require_regular_exec(path: &Path) -> Result<(), HiggsError> {
    let m = std::fs::symlink_metadata(path).map_err(|e| HiggsError::UpdateApplyFailed {
        detail: format!(
            "update tarball did not contain a regular executable 'higgs' ({}): {e}",
            path.display()
        ),
    })?;
    if m.file_type().is_symlink() || !m.file_type().is_file() {
        return Err(HiggsError::UpdateApplyFailed {
            detail: format!("staged {} is not a regular file", path.display()),
        });
    }
    let cpath = std::ffi::CString::new(path.as_os_str().as_encoded_bytes()).map_err(|_| {
        HiggsError::UpdateApplyFailed {
            detail: format!("path {} has an interior NUL", path.display()),
        }
    })?;
    // SAFETY: access() reads a valid NUL-terminated path; no memory is retained.
    if unsafe { libc::access(cpath.as_ptr(), libc::X_OK) } != 0 {
        return Err(HiggsError::UpdateApplyFailed {
            detail: format!(
                "staged {} is not executable by this user ({})",
                path.display(),
                std::io::Error::last_os_error()
            ),
        });
    }
    Ok(())
}

/// Write `bytes` to `path` via a temp-then-rename so a reader never sees a partial
/// file. The temp lives in the same dir (same filesystem -> `rename(2)` is atomic).
/// The temp's DATA is fsync'd before the rename and the DIR after, so a marker that
/// reports "written" survives a crash (best-effort on macOS, durable on Linux).
///
/// The file is forced to `0600` THROUGH ITS FD (`fchmod`), NOT left to `File::create`'s
/// umask-masked default: this writes the boot-fail counter + trial marker from the BOOT
/// path, whose umask is the daemon's INHERITED one (nothing in this module clamps it —
/// every mode is set by an explicit umask-independent `chmod`/`fchmod`).
/// A daemon under `umask 0777` would otherwise create a mode-`000` counter that
/// `read_boot_fails` cannot read (it silently returns 0, so the crash-loop budget never
/// accumulates and a bad trial never rolls back); `umask 000` would create a `0666`
/// counter a local peer could rewrite to force/suppress a rollback. `0600` is deterministic
/// and owner-only regardless of umask.
fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), HiggsError> {
    use std::os::unix::fs::PermissionsExt;
    let dir = path.parent().unwrap_or(Path::new("."));
    let tmp = dir.join(temp_name(".tmp"));
    {
        let mut f = std::fs::File::create(&tmp).map_err(|e| {
            let _ = std::fs::remove_file(&tmp);
            HiggsError::UpdateApplyFailed {
                detail: format!("creating temp for {}: {e}", path.display()),
            }
        })?;
        // fchmod through the fd (owner-only, umask-independent) BEFORE writing the bytes.
        if let Err(e) = f.set_permissions(std::fs::Permissions::from_mode(0o600)) {
            let _ = std::fs::remove_file(&tmp);
            return Err(HiggsError::UpdateApplyFailed {
                detail: format!("setting mode on temp for {}: {e}", path.display()),
            });
        }
        use std::io::Write;
        if let Err(e) = f.write_all(bytes).and_then(|_| f.sync_all()) {
            let _ = std::fs::remove_file(&tmp);
            return Err(HiggsError::UpdateApplyFailed {
                detail: format!("writing {}: {e}", path.display()),
            });
        }
    }
    std::fs::rename(&tmp, path).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        HiggsError::UpdateApplyFailed {
            detail: format!("renaming into {}: {e}", path.display()),
        }
    })?;
    fsync_path(dir)?;
    Ok(())
}

/// Remove a path if it exists; a missing path is success (idempotent cleanup).
fn remove_if_present(path: &Path) -> Result<(), HiggsError> {
    match std::fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(HiggsError::UpdateApplyFailed {
            detail: format!("cannot remove {}: {e}", path.display()),
        }),
    }
}

/// Map an io::Error at an apply step to HG087 with a step label.
fn io_apply(step: &'static str) -> impl Fn(std::io::Error) -> HiggsError {
    move |e| HiggsError::UpdateApplyFailed {
        detail: format!("{step}: {e}"),
    }
}

/// Force every DIRECTORY under `root` (including `root`) to be owner-traversable
/// (`0o700`), top-down so each dir is chmod'd BEFORE it is read. A verified-but-malformed
/// archive can carry a directory entry with a restrictive mode (e.g. `0o000`), which
/// `remove_dir_all` then cannot enter — leaking the whole staging tree. We own every node
/// we just unpacked, so restoring owner rwx always succeeds and makes the tree removable.
/// Best-effort throughout (this is cleanup): a per-entry error must not abort the rest,
/// and symlinks are never followed (only real subdirectories are recursed into).
fn force_dirs_owner_rwx(root: &Path) {
    use std::os::unix::fs::PermissionsExt;
    // chmod THIS dir first so the read_dir below can list it.
    let _ = std::fs::set_permissions(root, std::fs::Permissions::from_mode(0o700));
    let Ok(rd) = std::fs::read_dir(root) else {
        return;
    };
    for entry in rd.flatten() {
        let p = entry.path();
        if let Ok(m) = std::fs::symlink_metadata(&p) {
            if m.file_type().is_dir() {
                force_dirs_owner_rwx(&p);
            }
        }
    }
}

/// Best-effort removal of a staging dir on scope exit (any early return from
/// [`stage_and_flip`] leaves no half-unpacked litter under `bin`).
struct RemoveOnDrop(PathBuf);
impl Drop for RemoveOnDrop {
    fn drop(&mut self) {
        // A restrictive-mode directory in a malformed archive would make `remove_dir_all`
        // fail to traverse and leak the whole tree (accumulating undeletable `.update-
        // stage.*` litter under `bin` across repeated failed updates). Restore owner
        // traversal first so the removal always succeeds.
        force_dirs_owner_rwx(&self.0);
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[cfg(test)]
#[path = "self_update_tests.rs"]
mod tests;
