//! Round-3 integration coverage for `node/cli.rs` — the argument/refusal surface.
//!
//! Every test here spawns the real `higgs` binary with bad or edge ARGUMENTS and asserts
//! the usage text / refusal phrasing / exit code an operator actually sees. Nothing ever
//! installs a service, pairs to a real hub, or self-updates: each scenario is steered into
//! a refusal (or a maintenance no-op) that returns BEFORE any destructive step, inside a
//! throwaway `HIGGS_HOME`/prefix. Targets (all currently uncovered):
//!
//! - `node self-update`: empty `--url` value; `--prune` over a single-version install
//!   ("nothing to prune", dry-run and real).
//! - `node install-service`: the HOME-unset state-dir refusal; the non-executable /
//!   non-regular / hanging service-binary preflight; FIFO + symlink `node.log` refusals;
//!   unwritable logs dir; `--system` logs-outside-prefix + dangling-logs-symlink; the
//!   macOS ACL walk (write-granting, deny/self-ignoring, unreadable-ACL-as-present);
//!   group-writable `bin` under `--system`.
//! - the self-update BOOT GUARD: a spent trial rolls `current` back before the daemon
//!   runs; a non-install layout (no `current` symlink) never triggers it.
//! - `higgs --node` daemon: corrupt saved ticket; the bare-wait loop detecting a pairing
//!   and reconnect diagnostics after the hub drops; corrupt config mid-wait; the pairing
//!   one-shot's fast-reject "pairing failed" and SIGTERM "pairing cancelled" arms.
//! - `link pair`: a leave whose durable allowlist removal cannot persist answers
//!   "leave failed" and keeps the node paired.
//!
//! All children get a TEMP `HIGGS_HOME` (never the real `~/.higgs`), loopback-only iroh
//! (`HIGGS_IROH_LOCAL=1`), SIGTERM teardown, and deadline-bounded reads — no bare sleeps
//! as synchronization. Env is set per-child only (no process-global `set_var`), so no
//! cross-test serialization lock is needed.

use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use iroh_tickets::endpoint::EndpointTicket;
use tokio::io::{AsyncBufReadExt, BufReader};

use higgs::auth::{Allowlist, PairingTokens};
use higgs::config::{InstanceConfig, SavedHub};
use higgs::node::{
    connect_node, gate_connection, send_leave, GateOutcome, HubIdentity, HELLO_DEADLINE,
};
use higgs::remote::ALPN;

// ── shared helpers ──────────────────────────────────────────────────────────────────────

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// Run `higgs <args>` to completion with an isolated `HIGGS_HOME` + hermetic iroh.
fn run_higgs(home: &Path, args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_higgs"))
        .args(args)
        .env("HIGGS_HOME", home)
        .env("HIGGS_IROH_LOCAL", "1")
        .output()
        .expect("spawn higgs")
}

fn stderr_of(out: &std::process::Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

fn stdout_of(out: &std::process::Output) -> String {
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// A spawned std child. SIGTERM + reap on drop so an assertion failure never leaks a
/// daemon (and the graceful signal lets llvm-cov flush its profile).
struct Proc(Child);
impl Drop for Proc {
    fn drop(&mut self) {
        unsafe { libc::kill(self.0.id() as libc::pid_t, libc::SIGTERM) };
        let _ = self.0.wait();
    }
}

/// SIGTERM a tokio child (graceful teardown; a hard kill would drop the coverage flush).
fn sigterm(child: &tokio::process::Child) {
    if let Some(pid) = child.id() {
        unsafe { libc::kill(pid as libc::pid_t, libc::SIGTERM) };
    }
}

/// Poll a std child for exit, bounded by `secs`. Returns its status, or None on timeout.
fn wait_exit(child: &mut Child, secs: u64) -> Option<std::process::ExitStatus> {
    let deadline = Instant::now() + Duration::from_secs(secs);
    loop {
        if let Ok(Some(st)) = child.try_wait() {
            return Some(st);
        }
        if Instant::now() >= deadline {
            return None;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

/// Read child lines until one contains `needle`, bounded by `secs`.
async fn read_until<R>(lines: &mut tokio::io::Lines<R>, needle: &str, secs: u64) -> Option<String>
where
    R: tokio::io::AsyncBufRead + Unpin,
{
    tokio::time::timeout(Duration::from_secs(secs), async {
        while let Ok(Some(l)) = lines.next_line().await {
            if l.contains(needle) {
                return Some(l);
            }
        }
        None
    })
    .await
    .ok()
    .flatten()
}

/// A hermetic (relay-disabled) iroh endpoint on the higgs remote ALPN.
async fn minimal_ep() -> iroh::Endpoint {
    iroh::Endpoint::builder(iroh::endpoint::presets::Minimal)
        .alpns(vec![ALPN.to_vec()])
        .relay_mode(iroh::RelayMode::Disabled)
        .bind()
        .await
        .expect("bind endpoint")
}

/// Write an executable `#!/bin/sh` script — a stand-in service binary whose `--version`
/// probe behavior we control exactly.
fn write_script(path: &Path, body: &str) {
    use std::os::unix::fs::PermissionsExt;
    std::fs::write(path, format!("#!/bin/sh\n{body}\n")).expect("write script");
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755)).expect("chmod script");
}

/// Build the `<prefix>/bin/current/higgs` tree install-service preflights, with `current`
/// as a REAL directory (the CLI never requires it be a symlink for the preflight walks).
fn service_prefix_with_binary(prefix: &Path, script_body: &str) -> PathBuf {
    let current = prefix.join("bin").join("current");
    std::fs::create_dir_all(&current).expect("mkdir bin/current");
    let exec = current.join("higgs");
    write_script(&exec, script_body);
    exec
}

/// Hard-link (same volume) or copy the real test binary to `dst` — the install-shaped
/// layouts below need the genuine `higgs` at `bin/v<ver>/higgs`.
fn place_real_higgs(dst: &Path) {
    let src = env!("CARGO_BIN_EXE_higgs");
    if std::fs::hard_link(src, dst).is_err() {
        std::fs::copy(src, dst).expect("copy higgs binary");
    }
}

/// An install-shaped `<root>/bin` tree: the REAL binary at `v9.9.9/higgs`, a shell-script
/// rollback target at `v0.0.1/higgs`, and (optionally) `current -> v9.9.9`.
fn install_layout(root: &Path, with_current: bool) -> PathBuf {
    let bin = root.join("bin");
    let v_new = bin.join("v9.9.9");
    let v_old = bin.join("v0.0.1");
    std::fs::create_dir_all(&v_new).expect("mkdir v9.9.9");
    std::fs::create_dir_all(&v_old).expect("mkdir v0.0.1");
    place_real_higgs(&v_new.join("higgs"));
    // The rollback target only needs to be a regular executable named `higgs` — it is
    // validated (`is_installed_version`), never run, so a tiny script suffices.
    write_script(&v_old.join("higgs"), "exit 0");
    if with_current {
        std::os::unix::fs::symlink("v9.9.9", bin.join("current")).expect("symlink current");
    }
    bin
}

// ── `node self-update` argument arms ────────────────────────────────────────────────────

/// An empty `--url` value is refused up front — before any network or prefix work — with
/// the flag named (the VALUE is never echoed: it could carry a capability).
#[test]
fn self_update_empty_url_value_is_refused() {
    let home = tempfile::tempdir().unwrap();
    let out = run_higgs(home.path(), &["node", "self-update", "--url", ""]);
    assert!(!out.status.success(), "empty --url exits non-zero");
    assert!(
        stderr_of(&out).contains("--url must not be empty"),
        "names the empty flag: {}",
        stderr_of(&out)
    );
}

/// `--prune` over a single-version install (only the live `current` target exists) is a
/// clean NO-OP in both forms: `--dry-run` reports "nothing to prune" without touching the
/// tree, and the real op takes the update lock, removes nothing, and says so.
#[test]
fn self_update_prune_reports_nothing_to_prune_on_single_version_install() {
    let home = tempfile::tempdir().unwrap();
    let root = tempfile::tempdir().unwrap();
    // Minimal managed install: bin/current -> v0.0.1 with a regular executable inside.
    let bin = root.path().join("bin");
    let vdir = bin.join("v0.0.1");
    std::fs::create_dir_all(&vdir).unwrap();
    write_script(&vdir.join("higgs"), "exit 0");
    std::os::unix::fs::symlink("v0.0.1", bin.join("current")).unwrap();
    let prefix = root.path().to_str().unwrap();

    let dry = run_higgs(
        home.path(),
        &[
            "node",
            "self-update",
            "--prune",
            "--dry-run",
            "--prefix",
            prefix,
        ],
    );
    assert!(
        dry.status.success(),
        "prune --dry-run exits 0: {}",
        stderr_of(&dry)
    );
    assert!(
        stdout_of(&dry).contains("dry-run: nothing to prune"),
        "dry-run reports the empty plan: {}",
        stdout_of(&dry)
    );

    let real = run_higgs(
        home.path(),
        &["node", "self-update", "--prune", "--prefix", prefix],
    );
    assert!(real.status.success(), "prune exits 0: {}", stderr_of(&real));
    assert!(
        stdout_of(&real).contains("nothing to prune"),
        "the real prune reports nothing removed: {}",
        stdout_of(&real)
    );
    // The live version dir was never touched.
    assert!(
        vdir.join("higgs").exists(),
        "prune kept the current version"
    );
}

// ── `node install-service` refusal arms (all return BEFORE any system mutation) ─────────

/// With NO `--higgs-home`, no `HIGGS_HOME`, and `HOME` scrubbed from the environment (a
/// cron-style invocation), the state dir the node paired under is unknowable — the install
/// REFUSES loudly and demands an explicit `--higgs-home` instead of pinning a guess that
/// would crash-loop the service against an empty store.
#[test]
fn install_service_refuses_when_home_is_unset_and_no_higgs_home() {
    let prefix = tempfile::tempdir().unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_higgs"))
        .args([
            "node",
            "install-service",
            "--dry-run",
            "--prefix",
            prefix.path().to_str().unwrap(),
        ])
        .env_remove("HOME")
        .env_remove("HIGGS_HOME")
        .output()
        .expect("spawn higgs");
    assert!(!out.status.success(), "HOME-less install exits non-zero");
    let err = stderr_of(&out);
    assert!(
        err.contains("HOME is unset"),
        "names the missing HOME: {err}"
    );
    assert!(
        err.contains("--higgs-home"),
        "points at the explicit fix: {err}"
    );
}

/// The exec preflight proves the service binary actually RUNS as the operator: a present
/// but non-executable file and a directory at the exec path are both refused with the
/// install-the-binary-first guidance — before any manager state is touched.
#[test]
fn install_service_refuses_non_executable_or_irregular_service_binary() {
    use std::os::unix::fs::PermissionsExt;
    let home = tempfile::tempdir().unwrap();

    // (a) A regular file with no exec bit: spawn fails (EACCES) → not runnable.
    let p1 = tempfile::tempdir().unwrap();
    let current = p1.path().join("bin").join("current");
    std::fs::create_dir_all(&current).unwrap();
    let exec = current.join("higgs");
    std::fs::write(&exec, "#!/bin/sh\nexit 0\n").unwrap();
    std::fs::set_permissions(&exec, std::fs::Permissions::from_mode(0o644)).unwrap();
    let out = run_higgs(
        home.path(),
        &[
            "node",
            "install-service",
            "--prefix",
            p1.path().to_str().unwrap(),
        ],
    );
    assert!(!out.status.success(), "non-exec binary refused");
    assert!(
        stderr_of(&out).contains("not an executable file for the operator"),
        "explains the non-runnable binary: {}",
        stderr_of(&out)
    );

    // (b) A DIRECTORY at the exec path: not a regular file → refused the same way
    // (no execve is ever attempted on it).
    let p2 = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(p2.path().join("bin").join("current").join("higgs")).unwrap();
    let out = run_higgs(
        home.path(),
        &[
            "node",
            "install-service",
            "--prefix",
            p2.path().to_str().unwrap(),
        ],
    );
    assert!(!out.status.success(), "directory-at-exec-path refused");
    assert!(
        stderr_of(&out).contains("not an executable file for the operator"),
        "same refusal for an irregular exec path: {}",
        stderr_of(&out)
    );
}

/// A service binary that HANGS in its `--version` probe must not stall install-service
/// forever: the preflight kills the probe's process group at its bounded deadline and
/// refuses the install as not-runnable.
#[test]
fn install_service_bounds_a_hanging_version_probe() {
    let home = tempfile::tempdir().unwrap();
    let prefix = tempfile::tempdir().unwrap();
    // `sleep 300` would outlive any test; the group-kill at the 10s probe deadline reaps it.
    service_prefix_with_binary(prefix.path(), "sleep 300");
    let start = Instant::now();
    let out = run_higgs(
        home.path(),
        &[
            "node",
            "install-service",
            "--prefix",
            prefix.path().to_str().unwrap(),
        ],
    );
    assert!(!out.status.success(), "hanging probe refused");
    assert!(
        stderr_of(&out).contains("not an executable file for the operator"),
        "a hang reads as not-runnable: {}",
        stderr_of(&out)
    );
    let took = start.elapsed();
    assert!(
        took >= Duration::from_secs(9) && took < Duration::from_secs(60),
        "the probe is deadline-bounded (~10s), not open-ended: {took:?}"
    );
}

/// A pre-existing `node.log` that is a FIFO would BLOCK the daemon's (and launchd's) log
/// open forever — refused up front, before the exec probe or any manager command.
#[test]
fn install_service_refuses_fifo_node_log() {
    let home = tempfile::tempdir().unwrap();
    let prefix = tempfile::tempdir().unwrap();
    service_prefix_with_binary(prefix.path(), "exit 0");
    let logs = prefix.path().join("logs");
    std::fs::create_dir_all(&logs).unwrap();
    let fifo = logs.join("node.log");
    let cpath = std::ffi::CString::new(fifo.to_str().unwrap()).unwrap();
    // SAFETY: mkfifo reads a valid NUL-terminated path; no memory retained.
    assert_eq!(
        unsafe { libc::mkfifo(cpath.as_ptr(), 0o644) },
        0,
        "mkfifo node.log"
    );

    let out = run_higgs(
        home.path(),
        &[
            "node",
            "install-service",
            "--prefix",
            prefix.path().to_str().unwrap(),
        ],
    );
    assert!(!out.status.success(), "FIFO node.log refused");
    let err = stderr_of(&out);
    assert!(
        err.contains("not a regular file") && err.contains("FIFO/socket/device"),
        "names the irregular log and the block risk: {err}"
    );
}

/// A `node.log` that is a SYMLINK is refused by the launchd R/W openability probe
/// (O_NOFOLLOW → ELOOP): following it would let the daemon's log writes land in whatever
/// file a peer pointed it at. macOS-only: the R/W probe is the launchd branch.
#[cfg(target_os = "macos")]
#[test]
fn install_service_refuses_symlink_node_log_for_agent() {
    let home = tempfile::tempdir().unwrap();
    let prefix = tempfile::tempdir().unwrap();
    service_prefix_with_binary(prefix.path(), "exit 0");
    let logs = prefix.path().join("logs");
    std::fs::create_dir_all(&logs).unwrap();
    // The symlink target is an innocent regular file — O_NOFOLLOW must refuse anyway.
    let target = home.path().join("victim.txt");
    std::fs::write(&target, "operator file\n").unwrap();
    std::os::unix::fs::symlink(&target, logs.join("node.log")).unwrap();

    let out = run_higgs(
        home.path(),
        &[
            "node",
            "install-service",
            "--prefix",
            prefix.path().to_str().unwrap(),
        ],
    );
    assert!(!out.status.success(), "symlink node.log refused");
    let err = stderr_of(&out);
    assert!(
        err.contains("is a SYMLINK"),
        "names the symlink redirect risk: {err}"
    );
    assert!(
        err.contains("not openable by the daemon"),
        "framed as the daemon's own open failing: {err}"
    );
    // The probe never wrote through the link.
    assert_eq!(
        std::fs::read_to_string(&target).unwrap(),
        "operator file\n",
        "the symlink target was never touched"
    );
}

/// A pre-existing read-only `logs/` dir (a stale root-owned leftover shape) fails the
/// dir-writability probe with the log-rotation rationale — the daemon could never recreate
/// `node.log` after a rotation, so the install refuses now rather than losing logs later.
#[cfg(target_os = "macos")]
#[test]
fn install_service_refuses_unwritable_logs_dir() {
    use std::os::unix::fs::PermissionsExt;
    let home = tempfile::tempdir().unwrap();
    let prefix = tempfile::tempdir().unwrap();
    service_prefix_with_binary(prefix.path(), "exit 0");
    let logs = prefix.path().join("logs");
    std::fs::create_dir_all(&logs).unwrap();
    std::fs::set_permissions(&logs, std::fs::Permissions::from_mode(0o555)).unwrap();

    let out = run_higgs(
        home.path(),
        &[
            "node",
            "install-service",
            "--prefix",
            prefix.path().to_str().unwrap(),
        ],
    );
    // Restore BEFORE asserting so tempdir cleanup works whatever the outcome.
    std::fs::set_permissions(&logs, std::fs::Permissions::from_mode(0o755)).unwrap();
    assert!(!out.status.success(), "read-only logs dir refused");
    let err = stderr_of(&out);
    assert!(
        err.contains("is not writable") && err.contains("recreate node.log"),
        "explains the rotation consequence: {err}"
    );
}

/// `--system`: a `logs` dir that RESOLVES OUTSIDE the prefix (a symlinked log dir) is
/// refused — an external log tree's ancestry can't be safely validated for the root
/// daemon's O_NOFOLLOW-less log re-open.
#[cfg(target_os = "macos")]
#[test]
fn install_service_system_refuses_logs_resolving_outside_prefix() {
    let home = tempfile::tempdir().unwrap();
    let prefix = tempfile::tempdir().unwrap();
    let elsewhere = tempfile::tempdir().unwrap();
    service_prefix_with_binary(prefix.path(), "exit 0");
    std::os::unix::fs::symlink(elsewhere.path(), prefix.path().join("logs")).unwrap();

    let out = run_higgs(
        home.path(),
        &[
            "node",
            "install-service",
            "--system",
            "--prefix",
            prefix.path().to_str().unwrap(),
        ],
    );
    assert!(
        !out.status.success(),
        "external logs dir refused for the daemon"
    );
    let err = stderr_of(&out);
    assert!(
        err.contains("resolves OUTSIDE the prefix"),
        "names the external-resolution refusal: {err}"
    );
}

/// `--system`: a macOS ACL granting WRITE to another principal anywhere on the prefix
/// ancestry is a binary-swap vector the 0755 mode bits cannot see — refused with the
/// `chmod -N` remedy.
#[cfg(target_os = "macos")]
#[test]
fn install_service_system_refuses_write_granting_acl_on_prefix() {
    let home = tempfile::tempdir().unwrap();
    let prefix = tempfile::tempdir().unwrap();
    service_prefix_with_binary(prefix.path(), "exit 0");
    let st = Command::new("/bin/chmod")
        .args(["+a", "everyone allow write"])
        .arg(prefix.path())
        .status()
        .expect("chmod +a");
    assert!(st.success(), "grant the peer-write ACL");

    let out = run_higgs(
        home.path(),
        &[
            "node",
            "install-service",
            "--system",
            "--prefix",
            prefix.path().to_str().unwrap(),
        ],
    );
    // Strip the ACL before asserting so cleanup/reruns are unaffected.
    let _ = Command::new("/bin/chmod")
        .arg("-N")
        .arg(prefix.path())
        .status();
    assert!(!out.status.success(), "write-granting ACL refused");
    let err = stderr_of(&out);
    assert!(
        err.contains("macOS ACL granting write") && err.contains("chmod -N"),
        "names the ACL vector and the remedy: {err}"
    );
}

/// `--system`: a `deny` ACE and an allow-to-SELF ACE grant a peer nothing — the ACL walk
/// must pass them and let the install proceed to the NEXT refusal (here a dangling `logs`
/// symlink, which refuses rather than skip its resolved-ancestry validation).
#[cfg(target_os = "macos")]
#[test]
fn install_service_system_ignores_deny_and_self_acls() {
    let home = tempfile::tempdir().unwrap();
    let prefix = tempfile::tempdir().unwrap();
    service_prefix_with_binary(prefix.path(), "exit 0");
    // Benign ACEs: macOS homes carry deny entries; an allow-to-self grants no peer access.
    let me = std::env::var("USER").expect("USER set");
    for ace in [
        format!("user:{me} allow write"),
        "everyone deny delete".to_string(),
    ] {
        let st = Command::new("/bin/chmod")
            .args(["+a", &ace])
            .arg(prefix.path())
            .status()
            .expect("chmod +a");
        assert!(st.success(), "apply benign ACE {ace:?}");
    }
    // The controlled LATER refusal: logs exists (lstat-visible) but cannot resolve.
    std::os::unix::fs::symlink(
        prefix.path().join("nonexistent-target"),
        prefix.path().join("logs"),
    )
    .unwrap();

    let out = run_higgs(
        home.path(),
        &[
            "node",
            "install-service",
            "--system",
            "--prefix",
            prefix.path().to_str().unwrap(),
        ],
    );
    let _ = Command::new("/bin/chmod")
        .arg("-N")
        .arg(prefix.path())
        .status();
    assert!(!out.status.success(), "still refused, but NOT for the ACLs");
    let err = stderr_of(&out);
    assert!(
        !err.contains("macOS ACL granting write"),
        "deny/self ACEs are not misread as peer write grants: {err}"
    );
    assert!(
        err.contains("could not be resolved"),
        "the dangling logs symlink is the refusal that fired: {err}"
    );
}

/// `--system`: a prefix that CANNOT BE INSPECTED (an `everyone deny readsecurity` ACE
/// makes even `stat` fail EACCES) FAILS CLOSED — refused as unverifiable rather than
/// assumed safe, since the denial could be hiding a peer write grant.
#[cfg(target_os = "macos")]
#[test]
fn install_service_system_fails_closed_when_prefix_cannot_be_inspected() {
    let home = tempfile::tempdir().unwrap();
    let prefix = tempfile::tempdir().unwrap();
    service_prefix_with_binary(prefix.path(), "exit 0");
    let st = Command::new("/bin/chmod")
        .args(["+a", "everyone deny readsecurity"])
        .arg(prefix.path())
        .status()
        .expect("chmod +a");
    assert!(st.success(), "deny ACL reads on the prefix");

    let out = run_higgs(
        home.path(),
        &[
            "node",
            "install-service",
            "--system",
            "--prefix",
            prefix.path().to_str().unwrap(),
        ],
    );
    let _ = Command::new("/bin/chmod")
        .arg("-N")
        .arg(prefix.path())
        .status();
    assert!(
        !out.status.success(),
        "an uninspectable prefix fails closed"
    );
    let err = stderr_of(&out);
    assert!(
        err.contains("could not be checked") && err.contains("refusing rather than assume"),
        "refused as unverifiable, never assumed safe: {err}"
    );
}

/// `--system`: the daemon exec ancestry is GROUP-strict — the printed re-run execs the
/// prefix binary via sudo, so a group-writable `bin` is a root-escalation vector even
/// though group-write is the safe credential-group feature for the operator-run agent.
#[cfg(target_os = "macos")]
#[test]
fn install_service_system_rejects_group_writable_bin() {
    use std::os::unix::fs::PermissionsExt;
    let home = tempfile::tempdir().unwrap();
    let prefix = tempfile::tempdir().unwrap();
    let bin = prefix.path().join("bin");
    std::fs::create_dir_all(bin.join("current")).unwrap();
    std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o775)).unwrap();

    let out = run_higgs(
        home.path(),
        &[
            "node",
            "install-service",
            "--system",
            "--prefix",
            prefix.path().to_str().unwrap(),
        ],
    );
    assert!(
        !out.status.success(),
        "group-writable bin refused for the daemon"
    );
    let err = stderr_of(&out);
    assert!(
        err.contains("GROUP/OTHER-writable"),
        "the strict (group-rejecting) walk fired: {err}"
    );
}

// ── self-update boot guard (install-shaped layouts in a temp root) ──────────────────────

/// A trial that already SPENT its boot-failure budget rolls `current` back BEFORE the
/// daemon runs: the process exits cleanly having repointed `current` to the previous
/// version and poisoned the crash-looping one — the auto-recovery an operator relies on
/// after a bad update.
#[test]
fn boot_guard_rolls_back_a_spent_trial_before_the_daemon_starts() {
    let root = tempfile::tempdir().unwrap();
    let home = tempfile::tempdir().unwrap();
    let bin = install_layout(root.path(), true);
    // The on-disk trial contract stage_and_flip writes: current is v9.9.9, prev v0.0.1,
    // and the boot-fail counter has already reached the budget (3).
    std::fs::write(
        bin.join(".update-trial"),
        serde_json::json!({ "to": "v9.9.9", "prev": "v0.0.1" }).to_string(),
    )
    .unwrap();
    std::fs::write(bin.join(".update-bootfails"), "3").unwrap();

    let out = Command::new(bin.join("v9.9.9").join("higgs"))
        .args(["--node", "--list"])
        .env("HIGGS_HOME", home.path())
        .env("HIGGS_IROH_LOCAL", "1")
        .output()
        .expect("spawn install-shaped higgs");
    assert!(
        out.status.success(),
        "the rollback exit is clean: {}",
        stderr_of(&out)
    );
    assert!(
        stderr_of(&out).contains("rolled back to v0.0.1"),
        "announces the rollback target: {}",
        stderr_of(&out)
    );
    // `current` was atomically repointed to the known-good previous version…
    let cur = std::fs::read_link(bin.join("current")).expect("current is a symlink");
    assert_eq!(
        cur.to_string_lossy(),
        "v0.0.1",
        "current flipped to the rollback target"
    );
    // …and the crash-looping version is poisoned so a re-push is refused.
    let poisoned = std::fs::read_to_string(bin.join(".update-failed")).unwrap_or_default();
    assert!(
        poisoned.contains("v9.9.9"),
        "the failed version is recorded as poisoned: {poisoned:?}"
    );
}

/// Without the install shape (`current` symlink missing) the boot guard must NOT run —
/// spent-looking markers beside an unmanaged layout never flip anything, and the daemon
/// proceeds normally.
#[test]
fn boot_guard_skips_a_layout_without_a_current_symlink() {
    let root = tempfile::tempdir().unwrap();
    let home = tempfile::tempdir().unwrap();
    let bin = install_layout(root.path(), false);
    std::fs::write(
        bin.join(".update-trial"),
        serde_json::json!({ "to": "v9.9.9", "prev": "v0.0.1" }).to_string(),
    )
    .unwrap();
    std::fs::write(bin.join(".update-bootfails"), "3").unwrap();

    let out = Command::new(bin.join("v9.9.9").join("higgs"))
        .args(["--node", "--list"])
        .env("HIGGS_HOME", home.path())
        .env("HIGGS_IROH_LOCAL", "1")
        .output()
        .expect("spawn non-install-shaped higgs");
    assert!(
        out.status.success(),
        "--list runs normally: {}",
        stderr_of(&out)
    );
    assert!(
        stdout_of(&out).contains("no saved hubs"),
        "the daemon one-shot ran instead of the guard: {}",
        stdout_of(&out)
    );
    assert!(
        !stderr_of(&out).contains("rolled back"),
        "no rollback outside the managed layout: {}",
        stderr_of(&out)
    );
    // The markers were left exactly as found (the guard never engaged).
    assert_eq!(
        std::fs::read_to_string(bin.join(".update-bootfails")).unwrap(),
        "3",
        "spent-looking counter untouched"
    );
}

// ── `higgs --node` daemon: config-driven failure arms ───────────────────────────────────

/// A SAVED hub whose ticket no longer parses (config corruption) fails the daemon fast
/// with a non-zero exit — it must not spin dialing garbage.
#[test]
fn node_daemon_saved_garbage_ticket_fails_fast() {
    let home = tempfile::tempdir().unwrap();
    let mut cfg = InstanceConfig {
        name: "cov-args2(test)".into(),
        ..Default::default()
    };
    cfg.remember_hub(SavedHub {
        hub_id: "some-hub-id".into(),
        ticket: "definitely-not-an-endpoint-ticket".into(),
        label: "corrupt".into(),
        last_used_ms: 1,
    });
    cfg.save(&home.path().join("config.json")).unwrap();

    let mut child = Command::new(env!("CARGO_BIN_EXE_higgs"))
        .arg("--node")
        .env("HIGGS_HOME", home.path())
        .env("HIGGS_IROH_LOCAL", "1")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn higgs --node");
    let st = wait_exit(&mut child, 30);
    let Some(st) = st else {
        let _ = Proc(child); // SIGTERM + reap before failing
        panic!("daemon did not exit on a garbage saved ticket");
    };
    assert!(!st.success(), "garbage saved ticket exits non-zero");
}

/// A config that becomes CORRUPT while the bare daemon waits to be paired ends the wait
/// with a non-zero exit (an environment failure, reported — not an endless poll).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn node_daemon_wait_loop_exits_on_corrupt_config() {
    let home = tempfile::tempdir().unwrap();
    let mut child = tokio::process::Command::new(env!("CARGO_BIN_EXE_higgs"))
        .arg("--node")
        .env("HIGGS_HOME", home.path())
        .env("HIGGS_IROH_LOCAL", "1")
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .spawn()
        .expect("spawn higgs --node");
    let mut lines = BufReader::new(child.stdout.take().unwrap()).lines();
    assert!(
        read_until(&mut lines, "waiting to be paired", 30)
            .await
            .is_some(),
        "the bare daemon entered the pairing wait"
    );
    // Corrupt the config UNDER the waiting daemon; its next 3s poll must fail it out.
    std::fs::write(home.path().join("config.json"), "{ not-json !!").unwrap();
    let st = tokio::time::timeout(Duration::from_secs(30), child.wait())
        .await
        .expect("daemon exits within 30s of the corruption")
        .expect("wait");
    assert!(
        !st.success(),
        "corrupt config is a reported failure, not a hang"
    );
}

/// The bare-wait daemon (run from an INSTALL-SHAPED layout, so the self-update confirm
/// hooks are live) picks up a pairing persisted into config.json, connects, and — after
/// the hub drops the connection — reports the close, then escalates to the periodic
/// "what to check" block after 3 straight redial failures. This is the seamless pairing
/// handoff plus the remote operator's reconnect diagnostics, end to end.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn node_daemon_wait_detects_pairing_then_reports_reconnect_diagnostics() {
    let root = tempfile::tempdir().unwrap();
    let home = tempfile::tempdir().unwrap();
    let bin = install_layout(root.path(), true); // no trial markers: hooks are benign no-ops

    // A raw hub: the FIRST dial is allowlisted on the fly and admitted; later redials are
    // gated against an EMPTY allowlist (no token) and rejected — fast, so the failure
    // counter accrues quickly.
    let hub = minimal_ep().await;
    let hub_id = hub.id().to_string();
    let ticket = EndpointTicket::new(hub.addr()).to_string();
    let allow_dir = tempfile::tempdir().unwrap();
    let (conn_tx, mut conn_rx) = tokio::sync::mpsc::unbounded_channel();
    let hub_task = tokio::spawn(async move {
        let mut allow = Allowlist::load(&allow_dir.path().join("pairings.json")).unwrap();
        let mut reject_allow =
            Allowlist::load(&allow_dir.path().join("reject-pairings.json")).unwrap();
        let mut tokens = PairingTokens::new();
        let mut admitted_once = false;
        while let Some(incoming) = hub.accept().await {
            let Ok(conn) = incoming.await else { continue };
            if !admitted_once {
                // Learn the node's TLS identity from the connection itself and allowlist
                // it, so the tokenless service-style dial is admitted.
                let peer = conn.remote_id().to_string();
                let _ = allow.add(peer, Some("cov-node".into()));
                let outcome = gate_connection(
                    &conn,
                    &mut allow,
                    &mut tokens,
                    now_ms(),
                    &HubIdentity::new(hub_id.clone()),
                    None,
                    HELLO_DEADLINE,
                )
                .await;
                assert!(
                    matches!(outcome, GateOutcome::Admitted { .. }),
                    "first dial admitted: {outcome:?}"
                );
                admitted_once = true;
                let _ = conn_tx.send(conn);
            } else {
                let outcome = gate_connection(
                    &conn,
                    &mut reject_allow,
                    &mut tokens,
                    now_ms(),
                    &HubIdentity::new(hub_id.clone()),
                    None,
                    HELLO_DEADLINE,
                )
                .await;
                assert!(
                    matches!(outcome, GateOutcome::Rejected { .. }),
                    "redials are rejected: {outcome:?}"
                );
            }
        }
    });

    let mut child = tokio::process::Command::new(bin.join("v9.9.9").join("higgs"))
        .arg("--node")
        .env("HIGGS_HOME", home.path())
        .env("HIGGS_IROH_LOCAL", "1")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn install-shaped higgs --node");
    let mut out_lines = BufReader::new(child.stdout.take().unwrap()).lines();
    let mut err_lines = BufReader::new(child.stderr.take().unwrap()).lines();

    assert!(
        read_until(&mut out_lines, "waiting to be paired", 30)
            .await
            .is_some(),
        "the bare daemon entered the pairing wait"
    );
    // Persist the pairing UNDER the waiting daemon — exactly what the one-shot
    // `higgs --node <ticket> <token>` does — and let the 3s poll pick it up.
    let cfg_path = home.path().join("config.json");
    let mut cfg = InstanceConfig::load(&cfg_path).expect("daemon wrote its config");
    cfg.remember_hub(SavedHub {
        hub_id: hub_id_placeholder(&ticket),
        ticket: ticket.clone(),
        label: "covhub".into(),
        last_used_ms: now_ms(),
    });
    cfg.save(&cfg_path).expect("persist the pairing");

    assert!(
        read_until(&mut out_lines, "hub pairing detected", 30)
            .await
            .is_some(),
        "the wait loop detected the persisted pairing"
    );
    assert!(
        read_until(&mut out_lines, "paired with hub", 60)
            .await
            .is_some(),
        "the daemon connected to the hub"
    );
    // The hub drops the admitted connection → the daemon reports the close and redials.
    let conn = tokio::time::timeout(Duration::from_secs(30), conn_rx.recv())
        .await
        .expect("hub-side connection arrives")
        .expect("channel open");
    drop(conn);
    assert!(
        read_until(&mut err_lines, "hub connection closed; reconnecting", 30)
            .await
            .is_some(),
        "the close is reported before redialing"
    );
    // Three straight rejected redials trigger the periodic what-to-check guidance.
    assert!(
        read_until(&mut err_lines, "still unreachable after 3 attempts", 90)
            .await
            .is_some(),
        "the escalating diagnostics block fired after 3 failures"
    );

    sigterm(&child);
    let _ = child.wait().await;
    hub_task.abort();
}

/// The label-first `--hub` lookup means the stored hub_id is opaque here; any stable
/// non-empty string keeps the row valid.
fn hub_id_placeholder(ticket: &str) -> String {
    ticket.chars().take(16).collect()
}

// ── the pairing one-shot: fast-reject + cancel arms ─────────────────────────────────────

/// An explicit-ticket pairing against a hub that REJECTS the node fails after its bounded
/// attempts with the `pairing failed` verdict and the what-to-check advice — never an
/// endless opaque retry loop.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pairing_rejected_by_hub_reports_pairing_failed_with_advice() {
    let home = tempfile::tempdir().unwrap();
    // A hub that admits nothing: empty allowlist, no minted tokens.
    let hub = minimal_ep().await;
    let hub_id = hub.id().to_string();
    let ticket = EndpointTicket::new(hub.addr()).to_string();
    let allow_dir = tempfile::tempdir().unwrap();
    let hub_task = tokio::spawn(async move {
        let mut allow = Allowlist::load(&allow_dir.path().join("pairings.json")).unwrap();
        let mut tokens = PairingTokens::new();
        while let Some(incoming) = hub.accept().await {
            let Ok(conn) = incoming.await else { continue };
            let _ = gate_connection(
                &conn,
                &mut allow,
                &mut tokens,
                now_ms(),
                &HubIdentity::new(hub_id.clone()),
                None,
                HELLO_DEADLINE,
            )
            .await;
        }
    });

    let mut child = tokio::process::Command::new(env!("CARGO_BIN_EXE_higgs"))
        .args(["--node", &ticket])
        .env("HIGGS_HOME", home.path())
        .env("HIGGS_IROH_LOCAL", "1")
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .expect("spawn higgs --node <ticket>");
    let mut err_lines = BufReader::new(child.stderr.take().unwrap()).lines();
    assert!(
        read_until(&mut err_lines, "pairing failed", 120)
            .await
            .is_some(),
        "the bounded attempts end in the pairing-failed verdict"
    );
    assert!(
        read_until(&mut err_lines, "pairing could not reach the hub", 30)
            .await
            .is_some(),
        "the likely-causes advice follows the verdict"
    );
    let st = tokio::time::timeout(Duration::from_secs(30), child.wait())
        .await
        .expect("pairing one-shot exits")
        .expect("wait");
    assert!(!st.success(), "a failed pairing exits non-zero");
    hub_task.abort();
}

/// A SIGTERM during the pairing connect attempt cancels IMMEDIATELY (the biased select
/// honors the operator over an in-flight attempt) with the `pairing cancelled` verdict —
/// never persisting, never handing off.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pairing_cancelled_by_sigterm_during_connect_attempt() {
    let home = tempfile::tempdir().unwrap();
    // A hub that ACCEPTS the transport connection but never answers the HELLO — the
    // attempt hangs inside its 30s budget, which the operator's signal must preempt.
    let hub = minimal_ep().await;
    let ticket = EndpointTicket::new(hub.addr()).to_string();
    let hub_task = tokio::spawn(async move {
        let mut held = Vec::new();
        while let Some(incoming) = hub.accept().await {
            if let Ok(conn) = incoming.await {
                held.push(conn); // keep it open, say nothing
            }
        }
    });

    let mut child = tokio::process::Command::new(env!("CARGO_BIN_EXE_higgs"))
        .args(["--node", &ticket])
        .env("HIGGS_HOME", home.path())
        .env("HIGGS_IROH_LOCAL", "1")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .expect("spawn higgs --node <ticket>");
    let mut out_lines = BufReader::new(child.stdout.take().unwrap()).lines();
    let mut err_lines = BufReader::new(child.stderr.take().unwrap()).lines();
    // Wait until the connect loop is the active phase (a signal during the preflight is
    // buffered and still lands on the first connect select — but anchoring on this line
    // makes the test's intent explicit).
    assert!(
        read_until(&mut out_lines, "connecting to hub", 60)
            .await
            .is_some(),
        "the pairing reached its connect phase"
    );
    sigterm(&child);
    assert!(
        read_until(&mut err_lines, "pairing cancelled", 30)
            .await
            .is_some(),
        "the operator's stop is honored mid-attempt"
    );
    let st = tokio::time::timeout(Duration::from_secs(30), child.wait())
        .await
        .expect("cancelled pairing exits")
        .expect("wait");
    assert!(!st.success(), "a cancelled pairing exits non-zero");
    hub_task.abort();
}

// ── `link pair`: durable-removal failure on a node's self-retire ────────────────────────

/// When the `link pair` loop CANNOT persist the allowlist removal for a node's
/// `M_NODE_LEAVE` (the hub home is unwritable), the node gets the explicit
/// `leave failed` error — never a false `left` — and stays paired.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn link_pair_leave_with_unwritable_store_reports_leave_failed() {
    use std::os::unix::fs::PermissionsExt;
    let home = tempfile::tempdir().unwrap();

    let mut child = tokio::process::Command::new(env!("CARGO_BIN_EXE_higgs"))
        .args(["link", "pair"])
        .env("HIGGS_HOME", home.path())
        .env("HIGGS_IROH_LOCAL", "1")
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn higgs link pair");
    let mut lines = BufReader::new(child.stdout.take().unwrap()).lines();

    // Read the printed ticket + token.
    let (mut ticket, mut token) = (None, None);
    let read = tokio::time::timeout(Duration::from_secs(30), async {
        while let Ok(Some(line)) = lines.next_line().await {
            if let Some(rest) = line.strip_prefix("pairing token: ") {
                token = rest.split_whitespace().next().map(str::to_string);
            } else if let Some(rest) = line.strip_prefix("ticket       : ") {
                ticket = Some(rest.trim().to_string());
            }
            if ticket.is_some() && token.is_some() {
                break;
            }
        }
    })
    .await;
    assert!(read.is_ok(), "link pair printed its ticket+token");
    let ticket: EndpointTicket = ticket.expect("ticket").parse().expect("valid ticket");
    let addr = ticket.endpoint_addr().clone();
    let token = token.expect("token");

    // Pair a real node (the admission persists it into pairings.json while the home is
    // still writable).
    let node = minimal_ep().await;
    let node_id = node.id().to_string();
    let (conn, _hello) = connect_node(
        &node,
        addr,
        node_id.clone(),
        "cov-leaver".into(),
        Some(token),
    )
    .await
    .expect("node pairs with the link-pair loop");
    assert!(
        read_until(&mut lines, "paired", 30).await.is_some(),
        "link pair admitted the node"
    );

    // Now make the durable removal impossible and ask to leave.
    std::fs::set_permissions(home.path(), std::fs::Permissions::from_mode(0o555))
        .expect("chmod hub home read-only");
    let err = send_leave(&conn)
        .await
        .expect_err("leave cannot persist → error");
    // Restore BEFORE asserting so teardown/cleanup always works.
    std::fs::set_permissions(home.path(), std::fs::Permissions::from_mode(0o700))
        .expect("restore hub home perms");
    assert!(
        err.to_string().contains("leave failed"),
        "the node sees the explicit persistence failure: {err}"
    );
    // The node was NOT retired: the persisted allowlist still contains it.
    assert!(
        Allowlist::load(&home.path().join("pairings.json"))
            .unwrap()
            .contains(&node_id),
        "a failed durable removal keeps the node paired"
    );

    drop(conn);
    sigterm(&child);
    let _ = child.wait().await;
}
