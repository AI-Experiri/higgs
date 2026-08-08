//! Service installation for the node daemon (`higgs node install-service`).
//!
//! Everything that DECIDES or RENDERS is a pure function here — the CLI glue
//! (`node/cli.rs`) only resolves the invoking user, writes the rendered file,
//! and runs the returned commands. That split is the test seam: unit tests
//! cover every rendering/plan branch without touching launchd or systemd.
//!
//! Platform choices (docs: the release/update plan):
//! - USER-SPACE BY DEFAULT, one cross-platform dial ([`ServiceScope`]):
//!   the default [`ServiceScope::LoginBound`] never touches a system domain
//!   and never prompts — macOS: LaunchAgent in `~/Library/LaunchAgents`
//!   (gui/<uid> session domain); Linux: systemd USER unit, NO linger.
//! - `--system` ([`ServiceScope::SurvivesLogout`]) = boot-start + logout
//!   survival — macOS: a LaunchDAEMON (`/Library/LaunchDaemons/`, root-owned,
//!   sudo to install) still pinned to the operator via `UserName`; Linux: the
//!   SAME user unit + `loginctl enable-linger` (best-effort; never a system
//!   service).
//! - The systemd path must run AS THE OPERATOR ([`RootRequirement::Refuse`]):
//!   a root-run `systemctl --user` targets root's manager and writes a
//!   root-owned unit the operator cannot later update. The macOS agent path
//!   refuses root for the same reason (root would bootstrap root's gui
//!   domain); only the `--system` daemon REQUIRES root.
//! - All variants point at `<prefix>/bin/current/higgs` — the `current`
//!   symlink is the single authority (install.sh flips it atomically), so an
//!   update or rollback never edits the service files.
//! - Restart policy: always restart, 5 s apart, with NO start-limit window —
//!   a crash-looping binary must keep retrying until an operator intervenes
//!   (systemd's default 5-starts/10 s limit would otherwise permanently brick
//!   the unit on one bad update: the exact outage self-update must not cause).

use std::path::{Path, PathBuf};

/// How long the installed service keeps running — ONE cross-platform dial,
/// mapped by each OS onto its native mechanism (never one enum per OS).
/// USER-SPACE BY DEFAULT is the policy: nothing touches a system domain
/// unless explicitly asked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServiceScope {
    /// DEFAULT — the service lives with the operator's LOGIN SESSION: starts
    /// at login, stops at logout. (A locked screen is NOT a logout — the node
    /// keeps serving behind it.)
    /// - macOS: LaunchAgent in `~/Library/LaunchAgents` (no sudo, no password).
    /// - Linux: systemd user unit only — write + `systemctl --user enable /
    ///   daemon-reload / restart`, NO `loginctl enable-linger` (zero prompts).
    LoginBound,
    /// `--system` OPT-IN — the service OUTLIVES the session: starts at boot,
    /// keeps running after logout.
    /// - macOS: system LaunchDaemon in `/Library/LaunchDaemons` (sudo once to
    ///   install; the daemon still RUNS as the operator via `UserName`).
    /// - Linux: the SAME user unit plus `loginctl enable-linger` — still no
    ///   system service, linger just boots/keeps the user manager.
    SurvivesLogout,
    // FUTURE (planned, not implemented): truly always-on — SurvivesLogout
    // PLUS keeping the machine AWAKE while serving (system sleep suspends the
    // node regardless of scope, so an unattended worker box needs it).
    // macOS: caffeinate / IOPMAssertionCreateWithName; Linux: systemd-inhibit
    // --what=sleep.
    // KeepAwake,
}

/// Which service manager a [`ServicePlan`] targets (the per-OS MECHANISM the
/// cross-platform [`ServiceScope`] resolved to).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServiceKind {
    /// macOS `launchd` USER agent (`~/Library/LaunchAgents`, gui domain) —
    /// the [`ServiceScope::LoginBound`] default.
    LaunchdAgent,
    /// macOS `launchd` system daemon (`/Library/LaunchDaemons`) — the
    /// [`ServiceScope::SurvivesLogout`] opt-in.
    Launchd,
    /// Linux systemd USER unit (with or without linger, per scope).
    SystemdUser,
}

/// How a plan relates to root privilege.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RootRequirement {
    /// Must run as root (macOS LaunchDaemon → `/Library/LaunchDaemons`).
    Require,
    /// Must run as the OPERATOR, never root (Linux systemd USER unit — a
    /// root-run `systemctl --user` targets the wrong manager).
    Refuse,
}

/// One command the CLI runs after writing the unit file. `argv` is exec'd
/// directly (no shell). A `best_effort` command that fails is reported and
/// skipped — its remedy is in the plan's notes; a non-best-effort failure is
/// fatal. `as_operator` drops to the OPERATOR's credentials when the install
/// runs as root — REQUIRED for any command whose path traverses the operator's
/// home (a root `rm`/write through an operator-controlled ancestor symlink
/// could be redirected onto another file; with dropped credentials the
/// operator can only ever touch what they already own).
#[derive(Debug, Clone)]
pub struct ServiceCommand {
    pub argv: Vec<String>,
    pub best_effort: bool,
    pub as_operator: bool,
}

impl ServiceCommand {
    fn required(argv: Vec<String>) -> Self {
        Self {
            argv,
            best_effort: false,
            as_operator: false,
        }
    }
    fn best_effort(argv: Vec<String>) -> Self {
        Self {
            argv,
            best_effort: true,
            as_operator: false,
        }
    }
    fn required_as_operator(argv: Vec<String>) -> Self {
        Self {
            argv,
            best_effort: false,
            as_operator: true,
        }
    }
}

/// Everything `install-service` will do, decided up front: one file to write,
/// the commands to run after, and the guidance to print. `--dry-run` renders
/// exactly this plan without acting on it.
#[derive(Debug, Clone)]
pub struct ServicePlan {
    pub kind: ServiceKind,
    /// Where the unit/plist file goes.
    pub unit_path: PathBuf,
    /// The full rendered unit/plist content.
    pub unit_content: String,
    /// Commands to run AFTER writing the file, in order.
    pub commands: Vec<ServiceCommand>,
    /// The plan's relationship to root.
    pub root: RootRequirement,
    /// Operator guidance printed at the end (log path, manual fallbacks).
    ///
    /// RENDERING CONTRACT (`cli.rs` `render_plan_notes`): a note whose first
    /// `:`-delimited token is one of `logs`/`state`/`models`/`config`/`status`/
    /// `stop` renders as an aligned quick-reference row; everything else renders
    /// as a `!`-marked prose paragraph. Don't start a PROSE note with one of
    /// those tokens (start it with a multi-word lead-in instead).
    pub notes: Vec<String>,
}

/// The launchd job label / systemd unit name stem. One node daemon per machine.
pub const SERVICE_NAME: &str = "com.higgs.node";
/// The systemd user-unit file name.
pub const SYSTEMD_UNIT: &str = "higgs-node.service";

/// The DEPLOYMENT-CONFIG environment variables the node daemon reads that must
/// survive into the installed service — otherwise a node configured in the
/// operator's shell reboots under the service with DIFFERENT behavior. This is
/// an ALLOWLIST, not "preserve everything": it is the security boundary for the
/// `--env` flag (an operator must never be able to inject `LD_PRELOAD`/`DYLD_*`
/// into the daemon) AND it is kept DELIBERATELY SMALL — two vars:
/// - `HIGGS_HF_ENDPOINT`: the HuggingFace base a node pulls models from
///   (`download.rs`, higgs's own override). An enterprise mirror node that loses
///   it silently hits PUBLIC HuggingFace (data egress / air-gapped failure).
/// - `HIGGS_ENGINE`: which inference engine backend the node builds
///   (`worker/mod.rs`).
///
/// NOT in the list, on purpose:
/// - debug/verbosity knobs (`HIGGS_VERBOSE`, `HIGGS_WORKER_VERBOSE`, `RUST_LOG`)
///   — ad-hoc, must not be baked into a permanent service;
/// - test-only vars (`HIGGS_IROH_LOCAL`, `HIGGS_TEST_*`);
/// - `HIGGS_HOME`/`HIGGS_MODEL_DIR` — their own dedicated flags (absolutized,
///   `..`-validated, sudo-proof);
/// - the `huggingface-hub` CRATE's own env (`HF_HOME`, `HF_TOKEN`,
///   `HF_TOKEN_PATH`, `HF_HUB_CACHE`, `HF_HUB_DISABLE_IMPLICIT_TOKEN`, …). These
///   are ADVANCED HF-client tuning and a deliberate RESIDUAL, not an oversight:
///   (a) `HF_TOKEN`/token-path vars touch SECRETS that must never be serialized
///   into a 0644 unit — the node's real auth is the token FILE under `HF_HOME`
///   (default `~/.cache/huggingface/token`), which survives the service via the
///   pinned `HOME`; (b) the non-secret ones are relative-capable PATHS that
///   would each need the same absolutize+validate handling as HIGGS_HOME, an
///   open-ended surface not warranted for a marginal case. A node needing a
///   custom HF-client layout under the service sets it in the unit by hand; the
///   common case (token file + default cache under HOME) works unchanged.
pub const PRESERVED_ENV: &[&str] = &["HIGGS_HF_ENDPOINT", "HIGGS_ENGINE"];

/// The app-config env vars this install is NOT pinning — cleared from the service
/// child (systemd `UnsetEnvironment=`, launchd `launchctl unsetenv`) so a stale
/// value in the inherited manager / launchd context can't leak in: an omitted var
/// is NOT an absent one. HIGGS_MODEL_DIR unless pinned; each [`PRESERVED_ENV`] var
/// unless carried via `--env`; and ALWAYS the test-only `HIGGS_IROH_LOCAL` (it
/// would flip the node to the in-process/local transport and break fleet).
fn unpinned_app_env(model_dir: Option<&Path>, extra_env: &[(String, String)]) -> Vec<&'static str> {
    let mut v: Vec<&'static str> = Vec::new();
    if model_dir.is_none() {
        v.push("HIGGS_MODEL_DIR");
    }
    for k in PRESERVED_ENV {
        if !extra_env.iter().any(|(ek, _)| ek == k) {
            v.push(k);
        }
    }
    v.push("HIGGS_IROH_LOCAL");
    v
}

/// Builds the install plan for the current platform. `user`/`home`/`uid` are
/// the OPERATOR's identity (under `sudo` that is `SUDO_USER`'s passwd entry,
/// not root's — the caller resolves this; `uid` targets the macOS agent's
/// `gui/<uid>` domain); `prefix` is the install root holding `bin/current`
/// (default `~/.higgs`); `higgs_home` is the daemon's STATE dir (`$HIGGS_HOME`
/// else `<home>/.higgs`, resolved by the caller) — pinned into the service env
/// so a node paired under a custom `HIGGS_HOME` boots against that same state
/// instead of the `~/.higgs` default; `model_dir` is the optional EXTRA MODEL
/// SCAN ROOT (`$HIGGS_MODEL_DIR`, the README's documented pairing knob) — a
/// node paired with `HIGGS_MODEL_DIR=/models` must not reboot into a service
/// that scans only the defaults and advertises nothing; `scope` is the
/// cross-platform persistence dial (see [`ServiceScope`]); `extra_env` is the
/// resolved set of preserved deployment-config vars ([`PRESERVED_ENV`], e.g.
/// `HIGGS_HF_ENDPOINT`) to pin into the daemon environment.
pub fn plan_install(
    user: &str,
    home: &Path,
    uid: u32,
    prefix: &Path,
    higgs_home: &Path,
    model_dir: Option<&Path>,
    extra_env: &[(String, String)],
    scope: ServiceScope,
) -> ServicePlan {
    let mut plan = if cfg!(target_os = "macos") {
        match scope {
            ServiceScope::LoginBound => {
                plan_launchd_agent(home, uid, prefix, higgs_home, model_dir, extra_env)
            }
            ServiceScope::SurvivesLogout => {
                plan_launchd(user, home, uid, prefix, higgs_home, model_dir, extra_env)
            }
        }
    } else {
        plan_systemd_user(user, home, prefix, higgs_home, model_dir, extra_env, scope)
    };
    // Surfaced right after the `state:` note so --dry-run shows the full env the
    // daemon will boot with.
    let mut at = plan.notes.iter().position(|n| n.starts_with("state:"));
    if let Some(d) = model_dir {
        let i = at.map_or(plan.notes.len(), |i| i + 1);
        plan.notes
            .insert(i, format!("models: HIGGS_MODEL_DIR={}", d.display()));
        at = Some(i);
    }
    for (k, v) in extra_env {
        let i = at.map_or(plan.notes.len(), |i| i + 1);
        plan.notes.insert(i, format!("config: {k}={v}"));
        at = Some(i);
    }
    plan
}

fn higgs_current(prefix: &Path) -> PathBuf {
    prefix.join("bin").join("current").join("higgs")
}

fn log_path(prefix: &Path) -> PathBuf {
    prefix.join("logs").join("node.log")
}

/// The path of the LoginBound agent plist for `home` — one definition, used by
/// the agent plan, the daemon plan's cross-scope cleanup, and the CLI's
/// conflict check, so a rename can never desynchronize them.
pub fn agent_plist_path(home: &Path) -> PathBuf {
    home.join("Library")
        .join("LaunchAgents")
        .join(format!("{SERVICE_NAME}.plist"))
}

/// Cross-scope conflict (macOS, agent install): launchd allows the SAME label
/// in the gui and system domains, so installing the LoginBound agent while a
/// `--system` LaunchDaemon is registered would run TWO nodes side by side
/// (port conflicts, duplicate fleet registration) — and the no-sudo agent
/// install cannot remove root's files itself. Refuse with the exact cleanup.
/// The daemon plist path is a PARAMETER (the unit-test seam; prod passes the
/// real `/Library/LaunchDaemons` path, which a non-root test cannot create).
pub fn refuse_daemon_conflict(daemon_plist: &Path) -> std::io::Result<()> {
    // `plist_present`, NOT `.exists()`: `.exists()` follows the symlink (a dangling
    // daemon-plist symlink would read as absent) AND falls OPEN on a metadata error
    // (EACCES/EIO), letting the agent install proceed beside a daemon plist that is
    // actually there — two nodes once the fault clears. Presence is conservative.
    if plist_present(daemon_plist) {
        // `;` between the two cleanup commands, NOT `&&`: with a STALE plist
        // (daemon already unloaded) the bootout fails, and an `&&` chain would
        // then never remove the plist — every retry would hit this same
        // refusal forever. The rm must run regardless of the bootout outcome.
        return Err(std::io::Error::other(format!(
            "a --system LaunchDaemon is installed at {} — the login-bound agent would run \
             ALONGSIDE it (launchd allows the same label in both domains: two nodes, one \
             machine). Remove the daemon first:  sudo launchctl bootout \
             system/{} ; sudo rm {}   — then re-run (without sudo), or keep --system.",
            daemon_plist.display(),
            SERVICE_NAME,
            daemon_plist.display()
        )));
    }
    Ok(())
}

/// POSIX single-quotes `s` so it survives a shell as exactly ONE argument (the
/// `'\''` idiom closes the quote, emits an escaped `'`, reopens). For paths in a
/// copy-pasteable cleanup command — an operator home with a space
/// (`/Volumes/Node Homes/alice`) must stay one `rm` argument. (Mirrors cli.rs's
/// private copy; kept module-local because service.rs is the LOWER module — cli.rs
/// depends on it, not the reverse.)
fn shell_single_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', r"'\''"))
}

/// True unless `path` is DEFINITELY absent (`NotFound`). An INDETERMINATE metadata
/// error (EIO / EACCES / ELOOP / …) counts as PRESENT — the SAFE direction for a
/// cross-scope presence check: if we cannot prove a plist is gone, assume it is
/// there rather than fall open into the very dual-run the caller guards against.
/// Shared with the CLI's rollback `agent_present` probes so they agree on the
/// conservative reading (a transient fault that ALSO fails the agent `rm` must not
/// read the agent as gone and then keep the daemon plist beside it).
pub(crate) fn plist_present(path: &Path) -> bool {
    !matches!(
        std::fs::symlink_metadata(path),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound
    )
}

/// Presence of a plist that lives UNDER `trusted_root` (the operator's HOME for the
/// login-agent plist), disambiguating an ENOENT. `symlink_metadata` returns ENOENT
/// both when the plist LEAF is genuinely absent (benign — no agent installed) AND
/// when an INTERMEDIATE component is gone (a network/symlinked home unavailable
/// mid-check), where the plist may exist once the home returns. We anchor on
/// `trusted_root` (HOME — always present on a live operator's system; `~/Library/
/// LaunchAgents` itself may legitimately not exist yet, so it is NOT a safe anchor):
/// ENOENT reads as ABSENT only while the root still resolves; if the root is gone,
/// read PRESENT (conservative) so the coexistence guard / rollback cannot fall open
/// into a two-node state. A non-NotFound error is PRESENT (same as [`plist_present`]).
pub(crate) fn plist_present_under(path: &Path, trusted_root: &Path) -> bool {
    match std::fs::symlink_metadata(path) {
        Ok(_) => true,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => !trusted_root.exists(),
        Err(_) => true,
    }
}

/// Symmetric cross-scope guard for the DAEMON (re)install (macOS). The daemon
/// plan legitimately TEARS DOWN a login agent as part of an agent→daemon SWITCH —
/// but only the CLEAN switch, where no daemon plist exists yet (a first `--system`
/// install). If a daemon plist ALREADY exists AND an agent plist is ALSO present,
/// the auto-switch is unsafe: a reinstall KEEPS the pre-existing daemon plist
/// (`should_rollback_unit` is suppressed for a pre-existing unit), so if the
/// required agent-plist `rm` then fails, the plan's `enable` has already cleared
/// any `launchctl disable` on the daemon label and BOTH plists load at reboot (two
/// nodes). Supported flows CANNOT reach this state — the agent install
/// ([`refuse_daemon_conflict`]) refuses while a daemon plist exists — so it only
/// arises from manual tampering or a corrupt half-switch; refuse with the cleanup
/// rather than silently risk the dual-run. Both paths are PARAMETERS (the unit-test
/// seam; prod passes the real `/Library/LaunchDaemons` + `~/Library/LaunchAgents`
/// paths a non-root test cannot create). Presence is [`plist_present`] (not
/// `exists`): a dangling/planted symlink still counts (the predicate
/// `unit_preexisted` uses to gate the rollback this guard protects), and an
/// INDETERMINATE metadata error counts as PRESENT so a transient fault cannot fall
/// open into the dual-run.
pub fn refuse_cross_scope_coexistence(
    daemon_plist: &Path,
    agent_plist: &Path,
    agent_root: &Path,
) -> std::io::Result<()> {
    // The daemon plist lives under /Library/LaunchDaemons (a system dir that never
    // vanishes → plain `plist_present`); the agent plist lives under the operator's
    // HOME, so its presence is anchored on `agent_root` to disambiguate an ENOENT
    // caused by a vanished home from a genuine leaf-absence.
    if plist_present(daemon_plist) && plist_present_under(agent_plist, agent_root) {
        return Err(std::io::Error::other(format!(
            "both a --system LaunchDaemon plist ({}) and a login-bound agent plist ({}) already \
             exist — an unsupported mixed state the daemon (re)install cannot safely resolve \
             (enabling the daemon while the agent survives would run TWO nodes at the next \
             reboot/login). Remove ONE first, then re-run: keep the daemon with  launchctl \
             bootout gui/$(id -u)/{} ; rm {}   (as the operator), or keep the agent with  sudo \
             launchctl bootout system/{} ; sudo rm {}.",
            daemon_plist.display(),
            agent_plist.display(),
            SERVICE_NAME,
            shell_single_quote(&agent_plist.to_string_lossy()),
            SERVICE_NAME,
            shell_single_quote(&daemon_plist.to_string_lossy()),
        )));
    }
    Ok(())
}

/// A remedy hint appended to a service-command FAILURE. A launchd AGENT
/// ([`ServiceKind::LaunchdAgent`], the LoginBound default) loads into
/// `gui/<uid>`, a domain that exists only for an active GUI login session — a
/// headless Mac reached ONLY over SSH (no console/auto-login) has none, so its
/// `launchctl enable`/`bootstrap gui/<uid>` fails. Worker nodes are often
/// headless, and the required-command failure returns BEFORE the plan notes, so
/// the failure message itself points at the fix. Empty for the other kinds
/// (the daemon/systemd unit need no session).
pub fn agent_headless_hint(kind: ServiceKind) -> &'static str {
    match kind {
        ServiceKind::LaunchdAgent => {
            "  (if this is a headless Mac reached only over SSH, the login-bound agent has no \
             GUI session to load into — re-run with --system for the always-on LaunchDaemon)"
        }
        ServiceKind::Launchd | ServiceKind::SystemdUser => "",
    }
}

/// Downgrade honesty (Linux, LoginBound install): if LINGER is already enabled
/// for the operator (flag file under systemd's linger dir), the "login-bound"
/// unit will in fact STILL survive logout. Linger is a per-user flag other
/// services may rely on, so it is never auto-disabled — the note surfaces the
/// state and the exact opt-out command instead. The linger dir is a PARAMETER
/// (unit-test seam; prod passes `/var/lib/systemd/linger`).
pub fn linger_note(linger_dir: &Path, user: &str) -> Option<String> {
    linger_dir.join(user).exists().then(|| {
        format!(
            "note: linger is ENABLED for {user}, so this unit will still run after logout \
             despite the login-bound default — run `loginctl disable-linger {user}` to make \
             logout stop it (careful: other user services may rely on linger)"
        )
    })
}

fn plan_launchd(
    user: &str,
    home: &Path,
    uid: u32,
    prefix: &Path,
    higgs_home: &Path,
    model_dir: Option<&Path>,
    extra_env: &[(String, String)],
) -> ServicePlan {
    let unit_path = PathBuf::from(format!("/Library/LaunchDaemons/{SERVICE_NAME}.plist"));
    // App-config vars this install is NOT pinning — cleared from the launchd context
    // so a stale `launchctl setenv HIGGS_IROH_LOCAL` (etc.) can't leak into the new
    // daemon (the plist can only SET vars, not unset). Space-joined literals.
    let clear_env = unpinned_app_env(model_dir, extra_env).join(" ");
    ServicePlan {
        kind: ServiceKind::Launchd,
        // Order: enable → agent teardown (REQUIRED rm → gui bootout → REQUIRED
        // assert-gone) → bootout(system) → bootstrap. Two properties drive it:
        //
        // (1) `enable` runs FIRST. It is the step whose FAILURE must not strand the
        //     node: were it placed AFTER the agent teardown and it then failed
        //     (however rare — a root-local override-DB write), the running agent
        //     would already be booted out AND its plist removed, leaving NO node and
        //     nothing to restore. First, an `enable` failure aborts before anything
        //     is torn down, so the running agent survives untouched. `enable` clears
        //     a persistent `launchctl disable`, which survives reboots and would
        //     otherwise keep the job from loading after bootstrap with no obvious
        //     cause; enabling an already-enabled job is a no-op.
        //
        //     RESIDUAL (accepted, INERT): if a LATER required step aborts (an
        //     un-removable agent plist at the `rm`, or the assert-agent-gone), the
        //     `enable` has already run and is left set. But it loads NOTHING: on the
        //     `rm`-abort path the CLI rolls back the just-written daemon plist (the
        //     agent plist survives → `should_rollback_unit` true), so no plist for
        //     the label exists to load; the enablement merely clears a hypothetical
        //     prior `disable` (a label the operator is actively installing), trivial
        //     to re-assert. Reordering `enable` after the asserts to avoid this
        //     traded an inert residue for the real stranding above — not worth it.
        //
        // (2) The REQUIRED agent-plist `rm` runs before ANY destructive `bootout`,
        //     so if an immutable/ACL plist can't be removed we abort while any
        //     healthy OLD daemon is still running (no outage), the CLI rolls back the
        //     just-written daemon plist, and the surviving agent plist is the ONLY
        //     definition left (no reboot double-run).
        //
        // CROSS-SCOPE SWITCH (agent → daemon): launchd allows the SAME label in
        // the gui and system domains, so a leftover LoginBound agent would run
        // ALONGSIDE the new daemon (two nodes). Root can clean the operator's
        // agent, so the switch boots it out of gui/<uid> and removes its plist.
        commands: vec![
            // ENABLE FIRST — the NON-destructive required step (see property (1)):
            // if it fails, nothing has been torn down, so the running agent is
            // preserved. Clears a persistent `launchctl disable` so the daemon can
            // load after bootstrap.
            ServiceCommand::required(vec![
                "launchctl".into(),
                "enable".into(),
                format!("system/{SERVICE_NAME}"),
            ]),
            // AGENT PLIST REMOVAL — before ANY destructive `bootout` and before the
            // daemon plist goes live. AS THE OPERATOR: this path traverses the
            // operator's home — a root rm through an ancestor symlink
            // (`~/Library/LaunchAgents -> /Library/LaunchDaemons`) could be
            // redirected onto any file named com.higgs.node.plist; with dropped
            // credentials the operator can only delete what they already own.
            //
            // REQUIRED (not best-effort): `rm -f` already succeeds on a MISSING
            // plist (the fresh-install / no-prior-agent case), so a NON-zero exit
            // means the plist EXISTS and could not be removed — e.g. it is
            // immutable (`chflags uchg`) or ACL-protected. If it can't be removed,
            // ABORT here — before `bootout gui` stops the running agent and before
            // `bootout system` stops any old daemon — so the running node is NOT
            // disrupted; the CLI then ROLLS BACK the just-written daemon plist, so
            // the surviving agent plist is the ONLY definition left (no reboot
            // double-run; there is no runtime singleton lock).
            ServiceCommand::required_as_operator(vec![
                "rm".into(),
                "-f".into(),
                agent_plist_path(home).to_string_lossy().into_owned(),
            ]),
            // Now stop the old agent, if loaded (best-effort — legitimately fails
            // on a headless Mac with no GUI session, see `agent_headless_hint`, and
            // is a no-op on a fresh machine). Reached only after the plist was
            // removed, so it never leaves a loaded agent with a live plist.
            ServiceCommand::best_effort(vec![
                "launchctl".into(),
                "bootout".into(),
                format!("gui/{uid}/{SERVICE_NAME}"),
            ]),
            // ASSERT THE AGENT IS GONE before bootstrapping the daemon — REQUIRED.
            // The gui bootout above is best-effort (it legitimately fails on a
            // headless Mac with no GUI session), so a bootout that FAILED while the
            // agent was still LOADED would otherwise let this daemon bootstrap
            // ALONGSIDE it (two nodes until logout). `launchctl print gui/<uid>/
            // <label>` exits 0 only when the job is loaded; `if … then exit 1`
            // ABORTS the install then. When absent / headless the print fails, the
            // `if` is false, and this exits 0. The label is a fixed literal; nothing
            // is interpolated into the shell from operator input.
            ServiceCommand::required(vec![
                "sh".into(),
                "-c".into(),
                format!(
                    "if launchctl print gui/{uid}/{SERVICE_NAME} >/dev/null 2>&1; then echo \
                     'the login-bound agent is still LOADED (gui bootout failed) — boot it out \
                     (launchctl bootout gui/{uid}/{SERVICE_NAME}) before installing the daemon' \
                     >&2; exit 1; fi"
                ),
            ]),
            // Boot out any OLD daemon (best-effort — no-op when never loaded; a
            // re-install must re-read the plist).
            ServiceCommand::best_effort(vec![
                "launchctl".into(),
                "bootout".into(),
                format!("system/{SERVICE_NAME}"),
            ]),
            // Clear the UNPINNED app-config vars from the launchd context so a stale
            // `launchctl setenv` can't leak into the daemon (best-effort; the var
            // list is fixed literals, never operator input).
            ServiceCommand::best_effort(vec![
                "sh".into(),
                "-c".into(),
                format!("for v in {clear_env}; do launchctl unsetenv \"$v\"; done"),
            ]),
            ServiceCommand::required(vec![
                "launchctl".into(),
                "bootstrap".into(),
                "system".into(),
                unit_path.to_string_lossy().into_owned(),
            ]),
        ],
        unit_content: launchd_plist(Some(user), home, prefix, higgs_home, model_dir, extra_env),
        unit_path,
        root: RootRequirement::Require,
        notes: vec![
            format!("logs:  {}", log_path(prefix).display()),
            format!("state: HIGGS_HOME={}", higgs_home.display()),
            format!("status: sudo launchctl print system/{SERVICE_NAME}"),
            format!("stop:   sudo launchctl bootout system/{SERVICE_NAME}"),
            // `;` not `&&`: on a stale plist the bootout fails and must not
            // block the rm.
            "switch back to login-bound: remove the daemon first (sudo launchctl bootout \
             system/com.higgs.node ; sudo rm /Library/LaunchDaemons/com.higgs.node.plist), \
             then re-run without --system"
                .to_string(),
            // Honest disclosure of the residual: the agent-plist rm is REQUIRED
            // (an un-removable plist ABORTS before bootstrap, so it never silently
            // reloads the agent at the next login), but the preceding gui `bootout`
            // stays best-effort — it legitimately fails on a headless Mac with no
            // gui session. If it is skipped while an agent is somehow still loaded,
            // that agent keeps running until the next logout alongside this daemon.
            "if a `skipped:` line above names the gui bootout, an old login-bound agent may \
             still be loaded until the next logout — finish the switch by hand: launchctl \
             bootout gui/<uid>/com.higgs.node (as the operator)"
                .to_string(),
        ],
    }
}

/// The macOS DEFAULT: a launchd USER AGENT — fully user-space (plist in the
/// operator's own `~/Library/LaunchAgents`, bootstrapped into their `gui/<uid>`
/// session domain), no sudo, no password. Trade-off (stated in the notes):
/// login-bound — starts at login, stops at LOGOUT (a locked screen is NOT a
/// logout; the node keeps serving behind it). `--system` is the always-on
/// LaunchDaemon opt-in.
fn plan_launchd_agent(
    home: &Path,
    uid: u32,
    prefix: &Path,
    higgs_home: &Path,
    model_dir: Option<&Path>,
    extra_env: &[(String, String)],
) -> ServicePlan {
    let unit_path = agent_plist_path(home);
    // App-config vars this install is NOT pinning — cleared from the gui launchd
    // context so a stale `launchctl setenv` can't leak into the new agent.
    let clear_env = unpinned_app_env(model_dir, extra_env).join(" ");
    ServicePlan {
        kind: ServiceKind::LaunchdAgent,
        // Same enable → bootout → bootstrap order as the daemon (enable is the
        // non-destructive required step; if it fails, a healthy running agent
        // has not been booted out), targeting the operator's gui domain.
        commands: vec![
            ServiceCommand::required(vec![
                "launchctl".into(),
                "enable".into(),
                format!("gui/{uid}/{SERVICE_NAME}"),
            ]),
            ServiceCommand::best_effort(vec![
                "launchctl".into(),
                "bootout".into(),
                format!("gui/{uid}/{SERVICE_NAME}"),
            ]),
            // Clear the UNPINNED app-config vars from the gui launchd context
            // (best-effort; fixed literals, never operator input).
            ServiceCommand::best_effort(vec![
                "sh".into(),
                "-c".into(),
                format!("for v in {clear_env}; do launchctl unsetenv \"$v\"; done"),
            ]),
            ServiceCommand::required(vec![
                "launchctl".into(),
                "bootstrap".into(),
                format!("gui/{uid}"),
                unit_path.to_string_lossy().into_owned(),
            ]),
        ],
        // No UserName pin: an agent already runs as the session's user (launchd
        // ignores UserName outside the system domain).
        unit_content: launchd_plist(None, home, prefix, higgs_home, model_dir, extra_env),
        unit_path,
        root: RootRequirement::Refuse,
        // Quick-reference `key:` notes stay CONTIGUOUS (before any prose note)
        // so the renderer's aligned block isn't split — same as the other plans.
        notes: vec![
            format!("logs:  {}", log_path(prefix).display()),
            format!("state: HIGGS_HOME={}", higgs_home.display()),
            format!("status: launchctl print gui/{uid}/{SERVICE_NAME}"),
            format!("stop:   launchctl bootout gui/{uid}/{SERVICE_NAME}"),
            "login-bound: the agent starts at login and STOPS AT LOGOUT (a locked screen is \
             fine) — for an always-on node re-run with --system (LaunchDaemon, needs sudo)"
                .to_string(),
            // BOUNDED RESIDUAL (parity with the systemd note): install-time exec/log
            // access is probed under THIS shell's supplementary groups, but the agent
            // is bootstrapped into `gui/<uid>` — the GUI login session, whose groups
            // were fixed at LOGIN. If access to the prefix/log/model dir depends on a
            // group gained AFTER login (a fresh `dseditgroup`/`usermod`, or a `newgrp`
            // subshell), the probe can PASS while the session manager still lacks the
            // group and the agent fails to start/log or advertises no models. Log out
            // and back in (or reboot) before relying on it. Owner-based access (the
            // default: a prefix under your own home) is unaffected.
            "group-based access note: if the prefix/log/model dir is reachable only via a \
             group you were added to AFTER this login, fully log out and back in (or reboot) \
             before relying on the agent — the GUI session keeps its login-time groups"
                .to_string(),
        ],
    }
}

fn plan_systemd_user(
    user: &str,
    home: &Path,
    prefix: &Path,
    higgs_home: &Path,
    model_dir: Option<&Path>,
    extra_env: &[(String, String)],
    scope: ServiceScope,
) -> ServicePlan {
    // The unit lives at a STABLE ABSOLUTE path in the operator's OWN tree (never
    // a version dir — it points through `bin/current`), and is installed by
    // ENABLING IT BY ABSOLUTE PATH (`systemctl --user enable <abspath>`). That
    // lets systemd decide where the unit goes, so we never guess the search dir —
    // `$XDG_CONFIG_HOME`, `$SYSTEMD_UNIT_PATH`, and the transient/`.control`
    // entries in `UnitPath` all vary and mis-guessing would write a unit the
    // manager can't see (or restart a stale one). enable-by-path also means NO
    // `systemctl` runs during discovery.
    let unit_path = prefix.join(SYSTEMD_UNIT);
    let mut commands = vec![
        // ENABLE BY ABSOLUTE PATH, with --force — systemctl(1): a unit given
        // by full path is "linked into the unit lookup path" if OUTSIDE the
        // search dirs, and simply enabled if already INSIDE one; `--force`
        // OVERWRITES any conflicting symlink (a stale enablement/link from an
        // old `--prefix` on a migration). Doing it in ONE forced command —
        // with NO preceding `disable` — is deliberate: a `disable` would
        // strip the OLD boot-enablement, and if this `enable` then failed
        // (e.g. an absurd `--prefix` UNDER a search hierarchy that systemd
        // refuses), the node would be left DISABLED at reboot. This way a
        // failure leaves the prior enablement intact. Then reload so the
        // manager sees the freshly-linked unit.
        ServiceCommand::required(vec![
            "systemctl".into(),
            "--user".into(),
            "enable".into(),
            "--force".into(),
            unit_path.to_string_lossy().into_owned(),
        ]),
        ServiceCommand::required(vec![
            "systemctl".into(),
            "--user".into(),
            "daemon-reload".into(),
        ]),
        ServiceCommand::required(vec![
            "systemctl".into(),
            "--user".into(),
            "restart".into(),
            SYSTEMD_UNIT.into(),
        ]),
    ];
    // Quick-reference `key:` notes stay CONTIGUOUS (before any prose note) so
    // the renderer's aligned block isn't split — same order as the launchd plans.
    let mut notes = vec![
        format!("logs:  {}", log_path(prefix).display()),
        format!("state: HIGGS_HOME={}", higgs_home.display()),
        format!("status: systemctl --user status {SYSTEMD_UNIT}"),
        // Linked units must exist when the manager loads its units: a
        // prefix on a volume mounted AFTER the user manager starts leaves
        // the link dangling and the node down.
        format!(
            "the unit file stays at {} — keep --prefix on a filesystem that is mounted at \
             boot (a late-mounted volume leaves the linked unit unloadable until a manual \
             `systemctl --user daemon-reload`)",
            unit_path.display()
        ),
        // BOUNDED RESIDUAL: install-time exec/log access is probed under THIS
        // shell's supplementary groups, but the unit runs under the `systemd
        // --user` manager, whose groups were fixed at the operator's LOGIN. If
        // access to the prefix or log depends on a group gained AFTER that login
        // (fresh `usermod -aG`, a `newgrp`/`sg` subshell), the probe can PASS while
        // the older manager still lacks the group and the service fails to start or
        // log. `daemon-reexec` does NOT refresh credentials — only a full re-login
        // (or reboot) does. Owner-based access (a prefix under the operator's own
        // home — the default) is unaffected.
        "group-based access note: if the prefix/log is reachable only via a group you were \
         added to AFTER this login, fully log out and back in (or reboot) before relying on \
         the service — the running user manager keeps its login-time groups"
            .to_string(),
    ];
    match scope {
        // DEFAULT: pure user unit, ZERO possible prompts — no linger call at
        // all. The trade-off is stated: login-bound until the operator opts in.
        ServiceScope::LoginBound => notes.push(format!(
            "login-bound: the unit runs only while {user} has a session (starts at login, \
             stops at logout; a locked screen is fine) — for an always-on node re-run with \
             --system (adds `loginctl enable-linger`, still a user service)"
        )),
        // --system: linger keeps the user manager (and this unit) running at
        // boot with no login session. It commonly needs polkit/root, so it is
        // BEST-EFFORT: a failure prints the notes' sudo fallback rather than
        // aborting an otherwise-successful install.
        ServiceScope::SurvivesLogout => {
            commands.push(ServiceCommand::best_effort(vec![
                "loginctl".into(),
                "enable-linger".into(),
                user.into(),
            ]));
            notes.push(format!(
                "if enable-linger failed above: sudo loginctl enable-linger {user} (else the \
                 node stops at logout)"
            ));
        }
    }
    ServicePlan {
        kind: ServiceKind::SystemdUser,
        unit_content: systemd_unit(home, prefix, higgs_home, model_dir, extra_env),
        commands,
        root: RootRequirement::Refuse,
        notes,
        unit_path,
    }
}

/// Renders the macOS LaunchDaemon plist. All interpolated values are
/// XML-escaped — a home like `/Users/a&b` must not produce invalid XML.
///
/// - `user` is `Some(operator)` for the SYSTEM daemon — `UserName` pins it to
///   the operator (a LaunchDaemon defaults to root; the node must own
///   `~/.higgs` state as the operator) — and `None` for the USER AGENT, which
///   already runs as the session's user (launchd ignores `UserName` outside
///   the system domain, so an agent plist must not carry one).
/// - `EnvironmentVariables.HOME` is set explicitly: launchd does not populate
///   HOME for daemons, and higgs resolves `~/.higgs` through it.
/// - `EnvironmentVariables.HIGGS_HOME` pins the STATE dir explicitly so the
///   daemon reads the same store the node was paired under — set even when it
///   equals the `HOME`-derived default, so the daemon never depends on that
///   derivation.
/// - `KeepAlive` + `ThrottleInterval 5` = always restart, 5 s apart, forever —
///   launchd has no systemd-style start limit to disable.
pub fn launchd_plist(
    user: Option<&str>,
    home: &Path,
    prefix: &Path,
    higgs_home: &Path,
    model_dir: Option<&Path>,
    extra_env: &[(String, String)],
) -> String {
    let bin = xml_escape(&higgs_current(prefix).to_string_lossy());
    // Only the SYSTEM daemon pins UserName; an agent runs as the session user.
    let user_block = user
        .map(|u| {
            format!(
                "\n    <key>UserName</key>\n    <string>{}</string>",
                xml_escape(u)
            )
        })
        .unwrap_or_default();
    let home_s = xml_escape(&home.to_string_lossy());
    let higgs_home_s = xml_escape(&higgs_home.to_string_lossy());
    // Optional extra model scan root (`$HIGGS_MODEL_DIR` at pairing time): a
    // node paired with it must not reboot into a service that scans only the
    // defaults and advertises no models. Omitted entirely when unset.
    let model_env = model_dir
        .map(|d| {
            format!(
                "\n        <key>HIGGS_MODEL_DIR</key>\n        <string>{}</string>",
                xml_escape(&d.to_string_lossy())
            )
        })
        .unwrap_or_default();
    // Preserved deployment-config env (allowlisted PRESERVED_ENV vars) — each an
    // XML-escaped dict entry so the daemon reboots with the same config the
    // operator paired under.
    let extra: String = extra_env
        .iter()
        .map(|(k, v)| {
            format!(
                "\n        <key>{}</key>\n        <string>{}</string>",
                xml_escape(k),
                xml_escape(v)
            )
        })
        .collect();
    let log = xml_escape(&log_path(prefix).to_string_lossy());
    format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{SERVICE_NAME}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{bin}</string>
        <string>--node</string>
    </array>{user_block}
    <key>WorkingDirectory</key>
    <string>{home_s}</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>HOME</key>
        <string>{home_s}</string>
        <key>HIGGS_HOME</key>
        <string>{higgs_home_s}</string>{model_env}{extra}
    </dict>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>ThrottleInterval</key>
    <integer>5</integer>
    <key>Umask</key>
    <integer>18</integer>
    <key>StandardOutPath</key>
    <string>{log}</string>
    <key>StandardErrorPath</key>
    <string>{log}</string>
</dict>
</plist>
"#
    )
}

/// Renders the Linux systemd user unit.
///
/// - `StartLimitIntervalSec=0` lives in `[Unit]` (its section since systemd
///   230) and disables the start-rate limiter entirely — without it, five
///   crashes in 10 s mark the unit `failed` PERMANENTLY, which would turn one
///   bad binary swap into an outage that survives the fix.
/// - `Type=exec` (not the default `simple`): systemd treats the unit as
///   started only after `execve` succeeds, so `systemctl restart` FAILS on a
///   missing/unusable binary — surfacing a mistyped prefix at install time
///   instead of a silently boot-enabled crash loop.
/// - `%` is escaped to `%%` everywhere: a literal `%` in the prefix is
///   otherwise read as a systemd specifier (`%h` → home, etc.) and would
///   rewrite the path.
/// - `ExecStart` is double-QUOTED (it word-splits, so a space in the path
///   needs quoting), but `StandardOutput=append:`/`StandardError=append:`
///   take the whole rest of the line verbatim and are NOT quoted — a literal
///   `"` would become part of the path and fail systemd's absolute-path check
///   (the logs go to the journal instead and node.log stays empty).
/// - `Environment=HOME=...` PINS the operator's PASSWD home (the caller resolves
///   it; the daemon runs AS that operator). A systemd USER unit would otherwise
///   inherit `HOME` from the user manager, which can be stale/custom — and the
///   daemon derives its HuggingFace token file (`~/.cache/huggingface/token`) and
///   the default model scan roots (HF / Ollama / LM Studio) from `HOME`. Pinning
///   it to the passwd home (matching the macOS plist, which sets HOME too) makes
///   those deterministic for the COMMON layout (default HF cache under the
///   operator's own home) and is what lets `HF_TOKEN` stay OUT of
///   [`PRESERVED_ENV`]. A node paired under a DIFFERENT `$HOME` or a custom
///   `HF_HOME` is the documented PRESERVED_ENV residual — that advanced layout
///   needs the HF-client env set in the unit by hand; the state dir is still
///   honored via the explicit `HIGGS_HOME` pin below.
/// - `Environment=HIGGS_HOME=...` pins the daemon's STATE dir so a node paired
///   under a custom `HIGGS_HOME` boots against that store instead of the
///   `$HOME/.higgs` default. The whole assignment is double-quoted (so a space
///   in the path is one value, not a second `VAR=` token) and the value is
///   C-escaped for `\`/`"`, with `%` doubled — but `$` is NOT doubled: unlike
///   `ExecStart`, systemd does not expand `$VAR` inside `Environment=` values.
pub fn systemd_unit(
    home: &Path,
    prefix: &Path,
    higgs_home: &Path,
    model_dir: Option<&Path>,
    extra_env: &[(String, String)],
) -> String {
    // ExecStart is a double-quoted command WORD: systemd C-unescapes it, so a
    // literal `\` or `"` must be escaped too (not just the `%` specifier).
    let bin = systemd_exec_escape(&higgs_current(prefix).to_string_lossy());
    // append: is a bare value (no C-unescaping, no word-splitting): only `%`
    // is special there.
    let log = systemd_escape(&log_path(prefix).to_string_lossy());
    // Environment= value: C-escaped inside quotes, % doubled, $ kept literal.
    let home_env = systemd_env_escape(&home.to_string_lossy());
    let higgs_home = systemd_env_escape(&higgs_home.to_string_lossy());
    // Optional extra model scan root — same Environment= escaping; whole line
    // omitted when unset (the runtime treats absent and empty alike).
    let model_env = model_dir
        .map(|d| {
            format!(
                "Environment=\"HIGGS_MODEL_DIR={}\"\n",
                systemd_env_escape(&d.to_string_lossy())
            )
        })
        .unwrap_or_default();
    // Preserved deployment-config env (allowlisted) — one quoted Environment=
    // line each, same escaping as HIGGS_HOME.
    let extra: String = extra_env
        .iter()
        .map(|(k, v)| {
            // The key is an allowlisted PRESERVED_ENV literal (no specials), but
            // escape it too for uniformity; the value gets full env escaping.
            format!(
                "Environment=\"{}={}\"\n",
                systemd_env_escape(k),
                systemd_env_escape(v)
            )
        })
        .collect();
    // A systemd USER unit INHERITS the user manager's environment, so OMITTING a
    // var here does NOT make it absent: a stale `HIGGS_MODEL_DIR=/mnt/old` or an
    // old `HIGGS_HF_ENDPOINT` retained by the manager (a prior `import-environment`
    // / login shell) would leak into the daemon — silently scanning the wrong
    // models or hitting the wrong endpoint after a reinstall. So EXPLICITLY
    // UnsetEnvironment= every app-config var this install is NOT pinning, making the
    // config deterministic from the unit alone. (Never unsets a pinned var, so it
    // never fights the Environment= lines; var names are literals — no escaping.)
    let unset = unpinned_app_env(model_dir, extra_env);
    let unset_env = if unset.is_empty() {
        String::new()
    } else {
        format!("UnsetEnvironment={}\n", unset.join(" "))
    };
    format!(
        r#"[Unit]
Description=higgs node daemon (fleet worker)
StartLimitIntervalSec=0

[Service]
Type=exec
Environment="HOME={home_env}"
Environment="HIGGS_HOME={higgs_home}"
{model_env}{extra}{unset_env}ExecStart="{bin}" --node
Restart=always
RestartSec=5
StandardOutput=append:{log}
StandardError=append:{log}

[Install]
WantedBy=default.target
"#
    )
}

/// Minimal XML escape for plist string content (the five XML special chars).
fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

/// Escapes a literal string for a BARE systemd value (e.g. `append:PATH`):
/// only `%` (the specifier sigil) is special — doubled to `%%`. No
/// word-splitting or C-unescaping happens for these settings.
fn systemd_escape(s: &str) -> String {
    s.replace('%', "%%")
}

/// Escapes a literal string for a DOUBLE-QUOTED word inside `ExecStart`, which
/// systemd C-unescapes: `\` → `\\` and `"` → `\"` (backslash first so the
/// backslashes added for quotes are not re-doubled), plus `%` → `%%` (specifier
/// sigil) and `$` → `$$` (environment-variable expansion — systemd expands
/// `${VAR}`/`$VAR` in ExecStart words even inside double quotes, so a literal
/// `$` in the prefix would otherwise be swallowed or rewritten).
fn systemd_exec_escape(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('%', "%%")
        .replace('$', "$$")
}

/// Escapes a literal string for a DOUBLE-QUOTED `Environment=` value: systemd
/// C-unescapes the quoted string (`\` → `\\`, `"` → `\"`, backslash first) and
/// treats `%` as a specifier (`%%`). It does NOT expand `$VAR` in
/// `Environment=` values (only `ExecStart` words expand), so — unlike
/// [`systemd_exec_escape`] — a literal `$` stays a single dollar and must NOT
/// be doubled (doubling would write a bogus `$$` into the path).
fn systemd_env_escape(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('%', "%%")
}

#[cfg(test)]
#[path = "service_tests.rs"]
mod tests;
