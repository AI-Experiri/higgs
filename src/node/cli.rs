//! Lean P1 CLI to hand-drive pairing. Full fleet CLI (`link ls`, QR, keys) is P6.
//!
//! Because pairing tokens live in memory (intentionally short-lived, §7), `link pair`
//! both mints a token AND runs the accept loop in one process — a separate `pair`
//! process couldn't share the token store with a separate listener.

use std::io::{Error, Result};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use iroh_tickets::endpoint::EndpointTicket;

use std::path::{Path, PathBuf};

use crate::auth::{Allowlist, PairingTokens};
use crate::config::{config_path, name_or_init, InstanceConfig, Role, SavedHub};
use crate::home::ensure_home;
use crate::node::identity::{bind_endpoint, load_or_create_secret};
use crate::node::runtime::{NodeConfig, NodeRuntime};
use crate::node::{dial_and_hello, gate_connection, GateOutcome, HubIdentity, HELLO_DEADLINE};
use crate::remote::{HelloResult, PAIRING_TOKEN_TTL_MS as TOKEN_TTL_MS};

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn runtime() -> Result<tokio::runtime::Runtime> {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
}

fn key_path() -> Result<std::path::PathBuf> {
    Ok(ensure_home()?.join("endpoint.key"))
}

fn pairings_path() -> Result<std::path::PathBuf> {
    Ok(ensure_home()?.join("pairings.json"))
}

/// `higgs link <pair|status>` — hub-side fleet (Surface A).
pub fn run_link(args: &[String]) -> Result<()> {
    match args.first().map(String::as_str) {
        Some("pair") => run_link_pair(),
        Some("status") => run_link_status(),
        other => {
            eprintln!("usage: higgs link <pair|status> (got {other:?})");
            Err(Error::other("unknown link subcommand"))
        }
    }
}

/// Mint a one-time token, print the pairing ticket, and accept dials until Ctrl-C.
fn run_link_pair() -> Result<()> {
    let rt = runtime()?;
    rt.block_on(async {
        let sk = load_or_create_secret(&key_path()?)?;
        let endpoint = bind_endpoint(sk).await.map_err(Error::other)?;
        let hub_id = endpoint.id().to_string();
        // The hub's persistent friendly name, sent to each node in its HELLO result.
        let identity = HubIdentity {
            id: hub_id.clone(),
            name: name_or_init(Role::Hub, &hub_id, &crate::system::hostname())?,
        };
        let mut allow = Allowlist::load(&pairings_path()?)?;
        let mut tokens = PairingTokens::new();
        let token = tokens.mint(now_ms(), TOKEN_TTL_MS);

        // Wait (bounded) for a home relay so the ticket carries a relay URL and is
        // dialable from outside the hub's LAN. On a relay-less / offline setup we fall
        // back to whatever addresses we have (LAN-only) with a warning.
        if tokio::time::timeout(Duration::from_secs(10), endpoint.online())
            .await
            .is_err()
        {
            eprintln!("warning: no relay connected yet — ticket may only be dialable on the local network");
        }
        let ticket = EndpointTicket::new(endpoint.addr());

        println!("higgs hub    : {} ({hub_id})", identity.name);
        println!("pairing token: {token}   (single-use)");
        println!("ticket       : {ticket}");
        // The persistent daemon (`--node`), NOT the one-shot `node connect`: the daemon saves
        // the hub so a later bare `higgs --node` reconnects on its own (no token, no ticket).
        println!("on the node:  higgs --node {ticket} {token}");
        println!("listening for dials (Ctrl-C to stop)…");

        // SIGINT/SIGTERM ends the accept loop and returns cleanly so the process runs its
        // at-exit handlers (and, under coverage, flushes its profile) rather than dying mid-accept.
        let shutdown = crate::shutdown_signal();
        tokio::pin!(shutdown);
        loop {
            let incoming = tokio::select! {
                _ = &mut shutdown => break,
                incoming = endpoint.accept() => incoming,
            };
            let Some(incoming) = incoming else { break };
            let conn = match incoming.await {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("incoming connection failed: {e}");
                    continue;
                }
            };
            let peer = conn.remote_id().to_string();
            let outcome = gate_connection(
                &conn, &mut allow, &mut tokens, now_ms(), &identity, Some("paired-node".into()),
                HELLO_DEADLINE,
            )
            .await;
            match outcome {
                GateOutcome::Admitted {
                    agreed_version,
                    software_version,
                    // The CLI pairing loop keeps no fleet cache to push into.
                    fleet_events: _,
                    update_failed: _,
                    reports_update_failures: _,
                    target: _,
                    variant: _,
                    update_capable: _,
                    version_capable: _,
                    log_capable: _,
                } => {
                    // An all-filtered HELLO version sanitizes to empty — print a
                    // placeholder, not "higgs ," (T14 r22).
                    let sv = if software_version.is_empty() {
                        "?"
                    } else {
                        software_version.as_str()
                    };
                    println!("paired {peer} (higgs {sv}, protocol v{agreed_version})");
                    // Hold the connection until the node has read the HELLO result (it closes
                    // after reading) — and, if the node immediately asks to LEAVE (self-retire),
                    // handle it here too, so `higgs node leave` works against this CLI loop as
                    // well as the production hub server.
                    link_pair_post_admit(&conn, &mut allow, &peer).await;
                }
                GateOutcome::Rejected { code } => {
                    println!("rejected {peer} [{code}]");
                }
            }
        }
        Ok(())
    })
}

/// After `link pair` admits a node, hold the connection (bounded) so the node reads its HELLO
/// reply — and if the node immediately opens a `M_NODE_LEAVE` stream (self-retire), do the
/// DURABLE allowlist removal + ack so `higgs node leave` works against this CLI loop too. The
/// loop owns no fleet, so the removal IS the retire — the hub server seeds nodes from this same
/// `pairings.json`, so a removed node stays gone. A normal daemon opens no such stream (it opens
/// a uni log stream), so this just waits out the bounded window exactly as before.
async fn link_pair_post_admit(
    conn: &iroh::endpoint::Connection,
    allow: &mut Allowlist,
    peer: &str,
) {
    use tokio::io::{AsyncBufReadExt, BufReader};

    use crate::node::write_frame;
    use crate::rpc::{self, RpcError, RpcFrame, RpcResponse};

    let accepted = tokio::select! {
        _ = tokio::time::timeout(Duration::from_secs(5), conn.closed()) => return,
        a = conn.accept_bi() => a,
    };
    let Ok((mut send, recv)) = accepted else {
        return;
    };
    let Ok(Some(line)) = BufReader::new(recv).lines().next_line().await else {
        return;
    };
    let req = match rpc::decode(&line) {
        Ok(RpcFrame::Request(r)) => r,
        _ => return,
    };
    if req.method != crate::remote::M_NODE_LEAVE {
        return;
    }
    let resp = match allow.remove(peer) {
        Ok(()) => {
            println!("node {peer} left (self-retired)");
            RpcResponse {
                jsonrpc: "2.0".into(),
                id: req.id,
                result: Some(serde_json::json!({ "status": "left" })),
                error: None,
            }
        }
        Err(e) => RpcResponse {
            jsonrpc: "2.0".into(),
            id: req.id,
            result: None,
            error: Some(RpcError {
                code: -32000,
                message: format!("leave failed: {e}"),
                data: None,
            }),
        },
    };
    let _ = write_frame(&mut send, &RpcFrame::Response(resp)).await;
    let _ = send.finish();
    // Wait for the node to read the ack + close before returning — the caller drops `conn` on the
    // next loop iteration, which would otherwise truncate the reply mid-flight.
    let _ = tokio::time::timeout(Duration::from_secs(2), conn.closed()).await;
}

/// Print this hub's identity and the count of paired nodes.
fn run_link_status() -> Result<()> {
    let id = load_or_create_secret(&key_path()?)?.public();
    let allow = Allowlist::load(&pairings_path()?)?;
    println!("higgs hub id : {id}");
    println!("paired nodes : {}", allow.len());
    Ok(())
}

/// `higgs node <connect|leave|install-service>` — one-shot node-side ops.
pub fn run_node(args: &[String]) -> Result<()> {
    match args.first().map(String::as_str) {
        Some("connect") => run_node_connect(&args[1..]),
        Some("leave") => run_node_leave(&args[1..]),
        Some("install-service") => run_node_install_service(&args[1..]),
        Some("self-update") => run_node_self_update(&args[1..]),
        _ => {
            // NEVER echo the unrecognized subcommand token: `higgs node --url=<secret>`, a bare
            // `higgs node https://…/<capability>`, a dash-prefixed `-https://…/<secret>`, or even
            // an opaque flag-shaped token could be a capability. Syntax cannot prove a token is
            // non-secret, so print only the usage (which lists the valid subcommands).
            eprintln!("usage: higgs node <connect <ticket> [token] | leave [--hub <label|id>] | install-service [--dry-run] [--prefix <dir>] [--higgs-home <dir>] [--model-dir <dir>] [--env KEY=VALUE] [--system] | self-update [--url <manifest-url> | --tarball <f> --manifest <f> --manifest-sig <f>] [--prefix <dir>] [--allow-downgrade] [--dry-run] [--rollback] [--prune]> (unrecognized subcommand)");
            Err(Error::other("unknown node subcommand"))
        }
    }
}

/// The OPERATOR a service must be pinned to — resolved from the passwd
/// database, never from `$USER`/`$HOME` (sudo rewrites those inconsistently
/// across macOS and Linux). Under `sudo`, the operator is `SUDO_USER`'s
/// entry: root is the INSTALLER, never the service user.
struct Operator {
    user: String,
    home: std::path::PathBuf,
    uid: libc::uid_t,
    gid: libc::gid_t,
}

fn operator_identity() -> Result<Operator> {
    // SAFETY: geteuid has no preconditions and no failure mode.
    let euid = unsafe { libc::geteuid() };
    if euid == 0 {
        let sudo_user = std::env::var("SUDO_USER")
            .ok()
            .filter(|u| !u.is_empty())
            .ok_or_else(|| {
                // Platform-specific: macOS NEEDS root (so guide toward sudo);
                // Linux REFUSES root (so guide toward the operator's own
                // shell) — a one-size message would contradict the next check.
                if cfg!(target_os = "macos") {
                    // The DEFAULT is now a user-space LaunchAgent that REFUSES
                    // root, so a bare `sudo …` would hit the opposite gate. Tell
                    // the operator the two real paths: the login-bound agent (no
                    // sudo, from their own account) or the always-on daemon
                    // (`sudo … --system`), which is the only variant that wants
                    // root — and even then root cannot guess WHICH user, hence
                    // running from the operator's account.
                    Error::other(
                        "running as root with no SUDO_USER — install FROM the operator's login \
                         account, not a bare root shell: for the default login-bound agent run \
                         `higgs node install-service` WITHOUT sudo; for the always-on \
                         LaunchDaemon run `sudo higgs node install-service --system` (root \
                         cannot guess which user to pin the daemon to)",
                    )
                } else {
                    Error::other(
                        "install the Linux systemd user service AS THE OPERATOR, not root — \
                         log in as the operator and run `higgs node install-service` WITHOUT \
                         sudo (only `loginctl enable-linger` may need sudo, which the tool prints)",
                    )
                }
            })?;
        let op = passwd_by_name(&sudo_user)?;
        // A nested sudo sets SUDO_USER=root. Root is the INSTALLER, never the
        // service user — pinning a daemon to root (with an operator-writable
        // --prefix binary) would run operator-writable code as root. Check the
        // resolved UID, not the NAME: an ALTERNATE uid-0 account (`toor`, a
        // custom root) is just as dangerous as literal `root`.
        if op.uid == 0 {
            return Err(Error::other(format!(
                "SUDO_USER {:?} resolves to UID 0 — run `sudo higgs node install-service` \
                 directly from the operator's login shell, not from a nested root session (the \
                 service must be pinned to a non-root user)",
                op.user
            )));
        }
        Ok(op)
    } else {
        passwd_by_uid(euid)
    }
}

/// Refuses a `..` component in a resolved service path (from ANY source — flag,
/// env, or `<home>/.higgs` default). `std::path::absolute` keeps `..` verbatim
/// (it can't collapse it without touching the FS — symlink semantics), and
/// systemd silently IGNORES a `StandardOutput=append:` path containing `..`, so
/// the daemon would run but lose its logs. The per-flag `dir_value` guard
/// catches flag values early; this catches the env/default sources on the final
/// resolved value.
fn reject_dotdot(source: &str, path: &Path) -> Result<()> {
    if path
        .components()
        .any(|c| c == std::path::Component::ParentDir)
    {
        return Err(Error::other(format!(
            "{source} resolves to {} which contains a `..` component — use a resolved absolute \
             path (a `..` is kept verbatim and would silently misdirect the daemon's logs)",
            path.display()
        )));
    }
    Ok(())
}

/// The daemon's STATE dir, resolved AT INSTALL TIME so the service can pin it.
///
/// `higgs::home::higgs_home` resolves `$HIGGS_HOME` (else `$HOME/.higgs`) at
/// RUNTIME. The installed service sets `HOME` to the operator's home, so the
/// daemon's boot-time default already lands on `<op.home>/.higgs` — but if the
/// node was PAIRED under a custom `HIGGS_HOME` (a non-default state dir), the
/// daemon must be pinned to that same dir or it boots against `~/.higgs`, finds
/// no saved hub, and crash-loops. Capture the override here (made ABSOLUTE, so
/// a relative `HIGGS_HOME=./state` does not later resolve against the daemon's
/// own cwd) and pin it into the service env; with no override, fall back to
/// `<op.home>/.higgs` — an explicit pin equal to the boot-time default, which
/// is harmless and self-documented in `--dry-run`.
///
/// (Under `sudo` on macOS the operator's `HIGGS_HOME` env is stripped unless
/// kept by sudoers — the SUDO-PROOF carrier is the `--higgs-home` ARGV flag,
/// which the printed re-run/Next-steps commands use; this env fallback serves
/// the direct, non-elevated run. `--dry-run` prints the pinned value so a
/// mismatch is visible before committing.)
///
/// The no-override default mirrors the RUNTIME's own derivation
/// (`dirs::home_dir()` = `$HOME`) — pairing ran in the operator's shell, so
/// that is where the state lives. When the invocation's `$HOME` DISAGREES with
/// the passwd home (a `su -m` / `sudo -u` / `systemd-run --uid` shell that
/// kept another user's HOME), there is no way to know which home the node
/// actually paired under — guessing either way installs a service that boots
/// against the wrong (or unreadable) store and restart-loops, reporting
/// success today and failing from then on. That ambiguity REFUSES loudly and
/// demands an explicit `--higgs-home`. Elevated (euid 0) runs skip the check
/// ($HOME under sudo is rewritten to root's — the passwd home is the honest
/// fallback, and the elevated run normally receives the exact dir via
/// `--higgs-home` from the printed hint anyway).
///
/// A SET-but-EMPTY `HIGGS_HOME` is refused too (never treated as unset): the
/// runtime would resolve it as a relative empty path, so silently ignoring it
/// here would pin a different dir than the one the daemon would derive.
fn operator_higgs_home(op: &Operator, euid: libc::uid_t) -> Result<std::path::PathBuf> {
    // Read the ambient env ONCE and delegate to the pure seam (testable without
    // process-global env mutation).
    operator_higgs_home_from(
        std::env::var_os("HIGGS_HOME"),
        std::env::var_os("HOME"),
        op,
        euid,
    )
}

fn operator_higgs_home_from(
    higgs_home_env: Option<std::ffi::OsString>,
    home_env: Option<std::ffi::OsString>,
    op: &Operator,
    euid: libc::uid_t,
) -> Result<std::path::PathBuf> {
    match higgs_home_env {
        Some(v) if v.is_empty() => {
            return Err(Error::other(
                "HIGGS_HOME is set but EMPTY — the higgs runtime would resolve that as a \
                 relative path, so it cannot be honored or ignored; unset it, or pass \
                 --higgs-home <dir> explicitly",
            ))
        }
        Some(v) => {
            return Ok(std::path::absolute(&v).unwrap_or_else(|_| std::path::PathBuf::from(v)))
        }
        None => {}
    }
    if euid == 0 {
        // ELEVATED (sudo): the operator's HOME is reset to root's and their
        // HIGGS_HOME env is stripped, so the paired state dir is UNKNOWABLE here.
        // Guessing `<passwd-home>/.higgs` would silently pin the wrong (empty) store
        // for a node paired under a custom HIGGS_HOME — and bake that wrong value
        // into the printed re-run hint. Require an explicit `--higgs-home`;
        // install.sh's Next-steps command always passes it, so only a hand-typed
        // `sudo … --system` hits this, with a clear fix.
        return Err(Error::other(
            "an elevated (sudo) install cannot tell which state dir this node paired under \
             (sudo strips your HOME/HIGGS_HOME) — re-run with an explicit --higgs-home <dir> \
             (normally <your-home>/.higgs; install.sh's printed Next-steps command already \
             includes it)",
        ));
    }
    match home_env.filter(|v| !v.is_empty()) {
        Some(h) => {
            let home = std::path::absolute(&h).unwrap_or_else(|_| std::path::PathBuf::from(h));
            if home != op.home {
                return Err(Error::other(format!(
                    "HOME ({}) differs from the passwd home for uid {} ({}) — cannot infer which \
                     one this node's state was paired under; re-run with an explicit --higgs-home \
                     <dir> (the dir holding the node's saved hubs, normally <home>/.higgs)",
                    home.display(),
                    op.uid,
                    op.home.display()
                )));
            }
        }
        None => {
            // HOME is UNSET/empty (a `cron`/`env -u HOME` invocation): the node may
            // have paired under a CUSTOM HOME (its state at <that-home>/.higgs), which
            // we cannot recover here. REFUSE rather than silently pin the passwd-home
            // default — that would install a service pointed at an empty store and
            // restart-loop. (Same reasoning as the elevated branch above; install.sh's
            // Next-steps always passes --higgs-home, so only a hand-run install with
            // HOME stripped hits this.)
            return Err(Error::other(format!(
                "HOME is unset — cannot infer which state dir uid {} paired this node under \
                 (it may have used a custom HOME); re-run with an explicit --higgs-home <dir> \
                 (normally <your-home>/.higgs)",
                op.uid
            )));
        }
    }
    Ok(op.home.join(".higgs"))
}

/// A minimal, ROOT-OWNED search path for the trusted system tools this flow
/// spawns (`systemctl`, `launchctl`, `loginctl`, `sh`, `mkdir`). install-service
/// runs privileged (root on macOS via sudo); resolving a bare program name
/// through the AMBIENT `PATH` would let a binary planted in an operator- or
/// group-writable `PATH` dir run as root. Rust's `Command` resolves a bare name
/// against the Command's OWN env `PATH` (verified), so pinning it here defeats
/// that injection. Every privileged spawn sets `.env("PATH", TRUSTED_PATH)`.
pub(crate) const TRUSTED_PATH: &str = "/usr/sbin:/usr/bin:/sbin:/bin";

/// POSIX single-quotes `s` so it survives a shell as exactly ONE argument
/// (the `'\''` idiom closes the quote, emits an escaped `'`, reopens). Used to
/// print a copy-pasteable re-run command with an operator-controlled prefix.
fn shell_single_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', r"'\''"))
}

/// An UNPREDICTABLE, same-directory temp file name (`<base>.<random-hex>`). On a
/// group-writable prefix (which the credential group handling supports) a
/// predictable PID-derived name would let a group peer pre-plant a symlink at
/// it. Seeded from the process's random HashMap keys (from the OS RNG) — no
/// external crate. Paired with `create_new(true)` (`O_EXCL`, which never follows
/// or clobbers an existing path) at every use, so a symlink race can neither
/// redirect the write nor be pre-planted.
pub(crate) fn temp_name(base: &str) -> String {
    use std::hash::{BuildHasher, Hasher};
    let mut h = std::collections::hash_map::RandomState::new().build_hasher();
    h.write(base.as_bytes());
    h.write_u32(std::process::id());
    format!("{base}.{:016x}", h.finish())
}

/// Proves `dir` is writable BY THE CALLER by creating and unlinking a temp file
/// in it. `mkdir -p` on a PRE-EXISTING dir is a no-op that proves nothing about
/// writability (a stale root-owned `logs/` from a prior `sudo` run passes it),
/// and probing only `node.log` misses that the daemon must CREATE node.log after
/// a log rotation removes it — which needs write on the DIR, not the file. The
/// temp is opened `O_EXCL` under a random name so a group peer can't pre-plant a
/// symlink and redirect the create/truncate onto an operator-owned file.
/// (Used in-process on the Linux/operator path; the macOS root path runs the
/// equivalent `mktemp` create-then-remove through `drop_to_operator`.)
fn probe_dir_writable(dir: &Path) -> Result<()> {
    let probe = dir.join(temp_name(".higgs-writeprobe"));
    std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&probe)
        .and_then(|_| std::fs::remove_file(&probe))
        .map_err(|e| {
            Error::other(format!(
                "{} is not writable: {e} — the daemon could not recreate node.log after a log \
                 rotation; fix its ownership (or remove it) and re-run",
                dir.display()
            ))
        })
}

/// Proves the daemon (same user) can OPEN `node_log` the way ITS service manager
/// will — `want_rw` (macOS launchd opens `StandardErrorPath` READ/WRITE) or plain
/// APPEND (Linux systemd `append:`). A stale root-owned / mode-restricted file
/// surfaces NOW, not as silent no-logging later. When the file does not exist yet
/// it is CREATED with an EXPLICIT 0644 via fchmod: O_CREAT's mode is masked by the
/// process umask, so under a restrictive umask (0777) the created file would lack
/// even OWNER bits — this probe's already-open fd works the one time, but the
/// daemon's LATER open fails and the enabled node stays down despite a passing
/// preflight. The fchmod neutralizes the umask for the file WE create
/// (`create_new`/O_EXCL, so a pre-planted symlink is never followed); a
/// PRE-EXISTING file's mode is the operator's business — a bad one fails the probe
/// loudly right here, and is never silently chmodded.
///
/// The existing-file open is `O_NOFOLLOW`: a `node.log` that is a SYMLINK is
/// REFUSED, not followed. A peer who planted `node.log -> ~/.higgs/config.json`
/// while `logs/` was momentarily writable would otherwise have the daemon write
/// its log lines straight into that operator file (the world-writable refusal on
/// `logs/` does not remove a symlink already sitting inside it). O_NOFOLLOW turns
/// that redirect into a loud ELOOP refusal here.
fn probe_log_openable(node_log: &Path, want_rw: bool) -> std::io::Result<()> {
    use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
    let mut open_existing = std::fs::OpenOptions::new();
    let mut open_create = std::fs::OpenOptions::new();
    if want_rw {
        open_existing.read(true).write(true);
        open_create.read(true).write(true);
    } else {
        open_existing.append(true);
        open_create.append(true);
    }
    match open_existing.custom_flags(libc::O_NOFOLLOW).open(node_log) {
        Ok(_) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            let f = open_create.create_new(true).open(node_log)?;
            f.set_permissions(std::fs::Permissions::from_mode(0o644))
        }
        Err(e) if e.raw_os_error() == Some(libc::ELOOP) => Err(std::io::Error::other(format!(
            "{} is a SYMLINK — refusing to open the daemon's log through it (a peer may have \
             planted it to redirect the daemon's writes into one of your files); remove it and \
             re-run",
            node_log.display()
        ))),
        Err(e) => Err(e),
    }
}

/// APPEND-openable probe (Linux systemd `append:`). See [`probe_log_openable`].
fn ensure_appendable_log(node_log: &Path) -> std::io::Result<()> {
    probe_log_openable(node_log, false)
}

/// READ/WRITE-openable probe (macOS launchd opens the log R/W). See
/// [`probe_log_openable`].
fn ensure_rw_log(node_log: &Path) -> std::io::Result<()> {
    probe_log_openable(node_log, true)
}

/// Whether a directory-fsync failure should ABORT the install (vs. a soft warning).
/// Warn ONLY when the filesystem simply cannot fsync a directory (ENOTSUP /
/// EOPNOTSUPP / EINVAL / EBADF — a benign no-op on some FS types); any OTHER error
/// (EIO, ENOSPC, …) is a REAL storage failure that left the unit's rename
/// non-durable, so the install must abort before the destructive manager commands.
fn dir_fsync_should_abort(raw_os_error: Option<i32>) -> bool {
    // ENOTSUP and EOPNOTSUPP are the SAME value on Linux (distinct on macOS), so a
    // `Some(ENOTSUP) | Some(EOPNOTSUPP)` pattern is an unreachable-arm error there.
    // Compare by value in a guard instead — correct whether or not they coincide.
    !matches!(
        raw_os_error,
        Some(e)
            if e == libc::ENOTSUP
                || e == libc::EOPNOTSUPP
                || e == libc::EINVAL
                || e == libc::EBADF
    )
}

/// Whether the service's EXEC ancestry must reject GROUP-write (not just OTHER).
/// Only the macOS `--system` LaunchDaemon: its prefix binary is run via `sudo`
/// (the printed `sudo <prefix>/bin/current/higgs … --system`), so a group peer's
/// swapped `current`/`higgs` would run as ROOT. The login-bound agent and the
/// systemd USER unit run as the OPERATOR (no sudo), where group-write is the safe
/// credential-group feature.
fn exec_ancestry_rejects_group(kind: crate::node::service::ServiceKind) -> bool {
    matches!(kind, crate::node::service::ServiceKind::Launchd)
}

/// macOS: whether `path` carries an ACL that GRANTS WRITE to someone OTHER than the
/// operator. An ACL can grant a named user/group write/add_file/delete_child while
/// the POSIX mode stays `0755`, so the mode-bit ancestry checks miss it — for a
/// `--system` daemon (installed as root, re-exec'd as the operator) such a peer
/// could swap the binary. `ls -lde` lists each ACL entry on its own numbered line
/// (`N: <principal> <allow|deny> <perms>`). We flag ONLY an `allow` entry whose
/// perms include a write-capable right AND whose principal is not the operator's
/// own user — a `deny` entry (macOS homes carry a benign `everyone deny delete`),
/// a read-only `allow`, or an allow-to-self grants a peer NOTHING. Best-effort: any
/// spawn error or a non-macOS build returns false (`ls -e` is macOS-only).
/// Whether ONE `ls -lde` line is an ACL entry granting a WRITE-capable right to a
/// principal OTHER than `self_principal` (which is `user:<op>`). An entry is
/// `N: <principal…> [inherited] <allow|deny> <perms>`. Parse from the RIGHT, because
/// the PRINCIPAL is a Directory Service record name printed UNQUOTED by `ls`, so a
/// group like `group:Higgs Operators` spans MULTIPLE whitespace fields. The perms is
/// ALWAYS the last field (comma-joined, no spaces) and allow/deny the second-to-last,
/// no matter how wide the principal is (an `inherited` flag, if present, sits BEFORE
/// allow/deny and so never shifts them) — anchor on those. Scan ONLY the perms for a
/// write right (a whole-line search would false-match a principal like `group:writers`
/// on "write"); a `deny` or a read-only `allow` grants nothing.
///
/// The SELF exclusion (an allow-to-self grants a peer nothing) is CONSERVATIVE: the
/// principal is the FULL span `fields[1..len-2]`, INCLUDING a trailing `inherited`
/// word if present. Because `ls` does not quote the record name, an inherited ACE for
/// the operator (`user:alice inherited allow …`) is TEXTUALLY IDENTICAL to a DIRECT
/// ACE for a DIFFERENT user literally named `alice inherited`. Comparing the full span
/// (never a flag-stripped name) self-excludes ONLY the operator's own unambiguous ACE
/// and NEVER mistakes a `<self> inherited` peer for the operator (which would silently
/// drop that peer's write grant). The only cost is over-warning on a genuine INHERITED
/// self-grant — the SAFE direction.
fn acl_line_grants_write(line: &str, self_principal: &str) -> bool {
    // Write-capable ACL rights (any one lets a principal replace/relink an entry).
    const WRITE: &[&str] = &[
        "write",
        "add_file",
        "add_subdirectory",
        "append",
        "delete",
        "delete_child",
        "writeattr",
        "writeextattr",
        "chown",
        "writesecurity",
    ];
    let f: Vec<&str> = line.split_whitespace().collect();
    // Need at least: index, one principal field, allow/deny, perms.
    if f.len() < 4 {
        return false;
    }
    // Field 0 must be an ACE index `N:`.
    let is_entry = f[0]
        .strip_suffix(':')
        .is_some_and(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()));
    if !is_entry {
        return false;
    }
    let perms = f[f.len() - 1];
    let typ = f[f.len() - 2];
    if typ != "allow" {
        return false; // deny (or malformed) grants a peer nothing
    }
    // Full principal span (index+1 .. allow/deny), including a trailing `inherited`.
    let principal = f[1..f.len() - 2].join(" ");
    if principal == self_principal {
        return false; // unambiguously the operator's OWN ACE — grants a peer nothing
    }
    WRITE.iter().any(|w| perms.contains(w))
}

fn has_writable_acl(path: &Path, op_user: &str) -> bool {
    if !cfg!(target_os = "macos") {
        return false;
    }
    // A DEFINITELY-absent path carries no ACL (benign — e.g. a not-yet-created
    // node.log). But a path that EXISTS whose ACL we CANNOT read is an INSPECTION
    // FAILURE, not proof of "no ACL": a root-owned ancestor can DENY the operator
    // `readsecurity` (so `ls -lde` fails with EACCES) while GRANTING a peer write —
    // reading that as safe would let the peer swap the binary. Treat it as PRESENT.
    match std::fs::symlink_metadata(path) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return false,
        _ => {}
    }
    let self_principal = format!("user:{op_user}");
    match std::process::Command::new("ls")
        .env("PATH", TRUSTED_PATH)
        .arg("-lde")
        .arg(path)
        .output()
    {
        Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout)
            .lines()
            .any(|l| acl_line_grants_write(l, &self_principal)),
        // `ls` failed on a path that EXISTS → we could not read its ACL → UNSAFE.
        _ => true,
    }
}

/// Refuses a WRITE-granting macOS ACL ANYWHERE on `leaf`'s physical ancestry up to
/// `/` (like install.sh's `_gw_walk`). A non-inheriting `allow add_file,
/// delete_child` on an ANCESTOR (e.g. `/Users/alice`) lets a peer replace the
/// subtree below it while every managed endpoint stays ACL-free — so endpoint-only
/// checks miss it. Safe to walk to `/` because [`has_writable_acl`] flags ONLY an
/// allow-to-non-owner write grant: system dirs carry `deny` ACLs, which are ignored.
/// No-op off macOS.
fn refuse_writable_acl_ancestry(leaf: &Path, op_user: &str, what: &str) -> Result<()> {
    let mut cur = Some(leaf);
    while let Some(d) = cur {
        if has_writable_acl(d, op_user) {
            return Err(Error::other(format!(
                "{what} {} carries a macOS ACL granting write to another user — an ACL grants \
                 access independent of the 0755 mode bits, so a peer could replace this subtree \
                 and run code as you (as root for a sudo'd --system). Remove it (`chmod -N {}`) \
                 and re-install.",
                d.display(),
                d.display()
            )));
        }
        cur = d.parent();
    }
    Ok(())
}

/// Canonicalize a REQUIRED path (the service prefix / binary), refusing on ANY
/// failure — NEVER falling open to a lexical-only check. A path that cannot be
/// resolved at install time is either missing (the binary was never installed) or a
/// TOCTOU race: a peer renames a resolved-path component away so the resolved-ancestry
/// walk is SKIPPED, restores a genuine binary for the `--version` probe, then swaps a
/// payload before the root `sudo …` exec. Both must refuse.
fn resolve_required(path: &Path, what: &str) -> Result<std::path::PathBuf> {
    std::fs::canonicalize(path).map_err(|e| {
        Error::other(format!(
            "could not resolve {what} {} ({e}) — install the binary (run install.sh) and keep \
             the prefix path STABLE; a path that vanishes mid-validation is refused rather than \
             validated only by its lexical spelling (a repoint race). Re-run.",
            path.display()
        ))
    })
}

/// Canonicalize a path validated ONLY IF present (the daemon `logs` dir, which a
/// fresh install creates LATER): a definite ABSENCE (`NotFound`) is benign → `None`,
/// the caller skips. But a path that EXISTS — a dir, or a SYMLINK (even to a
/// now-hidden target, which `symlink_metadata` still lstat-sees as present) — yet
/// fails to resolve is a hide/TOCTOU signal → refuse rather than fall open and skip
/// the resolved-ancestry walk.
fn resolve_if_present(path: &Path, what: &str) -> Result<Option<std::path::PathBuf>> {
    match std::fs::symlink_metadata(path) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        _ => std::fs::canonicalize(path).map(Some).map_err(|e| {
            Error::other(format!(
                "the {what} {} exists but could not be resolved ({e}) — refusing rather than \
                 skip its resolved-ancestry validation (a symlink whose target was hidden \
                 mid-check, or an IO error). Re-run.",
                path.display()
            ))
        }),
    }
}

/// Refuse a write-granting macOS ACL anywhere on the DAEMON exec path, walking BOTH
/// the LEXICAL `exec_target` (`<prefix>/bin/current/higgs` AS SPELLED — the path the
/// service unit and the printed `sudo …` re-run actually use) AND its RESOLVED form.
/// The mode-bit `exec_ancestry` walks both chains; the ACL walk MUST too, or a
/// write-granting ACL on a 0755 LEXICAL parent (which the mode bits pass) reached
/// through an operator-owned symlink lets a peer repoint that symlink to their tree
/// between validation and the root exec — code as root. The resolved walk uses
/// `resolve_required` so a canonicalize FAILURE refuses (never skips). No-op off macOS.
fn refuse_exec_acls(exec_target: &Path, op_user: &str) -> Result<()> {
    refuse_writable_acl_ancestry(exec_target, op_user, "service binary path")?;
    let resolved = resolve_required(exec_target, "resolved service binary")?;
    refuse_writable_acl_ancestry(&resolved, op_user, "resolved service binary")?;
    Ok(())
}

/// Validate the service unit dir before the plist is published into it. Two
/// hardenings over a naive resolved-only check:
///
/// (1) canonicalize MUST succeed (the caller `create_dir_all`'d `dir` immediately
///     before): a failure means a concurrent remove/repoint race — REFUSE rather
///     than SKIP validation and then write+bootstrap into an unvalidated (possibly
///     peer-controlled) directory. A prior `if let Ok(real_dir)` silently skipped.
///
/// (2) walk BOTH the LEXICAL ancestry (the path the plist is actually written to /
///     launchd loads from — a group/ACL-writable lexical parent reached THROUGH a
///     symlinked unit dir, e.g. `~/Library` when `~/Library/LaunchAgents` is a
///     symlink, lives on THIS chain, not the resolved one) AND the RESOLVED ancestry
///     (the symlink target's own chain). The resolved-only walk missed the lexical
///     parent; the write uses the lexical path, so it must be checked too — the same
///     both-chains rule `exec_ancestry` and install.sh's `_gw_walk` apply.
///
/// For a LAUNCHD dir even GROUP-write is a plist-replacement vector (`_strict`) plus
/// a write-granting ACL; for SYSTEMD the dir IS the already-validated prefix where
/// group-write is the credential-group feature (group-tolerant re-check).
fn validate_unit_dir(
    dir: &Path,
    kind: crate::node::service::ServiceKind,
    op_uid: u32,
    op_user: &str,
) -> Result<()> {
    let real_dir = std::fs::canonicalize(dir).map_err(|e| {
        Error::other(format!(
            "could not resolve the service unit dir {} ({e}) — refusing rather than install the \
             plist into an unvalidated directory (a concurrent remove/repoint race). Re-run.",
            dir.display()
        ))
    })?;
    for d in [dir, real_dir.as_path()] {
        if matches!(kind, crate::node::service::ServiceKind::SystemdUser) {
            refuse_writable_ancestry(d, d, "service unit dir", op_uid)?;
        } else {
            refuse_writable_ancestry_strict(d, d, "service unit dir", op_uid)?;
            refuse_writable_acl_ancestry(d, op_user, "service unit dir")?;
        }
    }
    Ok(())
}

/// Whether a system LaunchDaemon with our label is currently LOADED into launchd,
/// via `launchctl print system/<label>`'s EXIT CODE (0 = present/loaded) — readable
/// WITHOUT root and needing no output parsing. A bootstrapped job SURVIVES deletion
/// of its plist, so this catches a daemon a partial `bootout ; rm` cleanup left
/// loaded. Best-effort: any spawn error (no launchctl, non-macOS) → false, falling
/// back to the plist-existence check.
fn daemon_job_loaded() -> bool {
    std::process::Command::new("launchctl")
        .env("PATH", TRUSTED_PATH)
        .arg("print")
        .arg(format!("system/{}", crate::node::service::SERVICE_NAME))
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Refuses a pre-existing `node.log` that is NOT a regular file. A FIFO / socket /
/// device passes the symlink + permission checks, but the daemon's log OPEN then
/// BLOCKS on it (a FIFO open-for-write waits forever for a reader) — hanging the
/// installer's append/R-W probe, and launchd's own later open. `reject_symlink`
/// (the macOS `--system` DAEMON) ALSO refuses a SYMLINK: ROOT launchd re-opens the
/// daemon log with NO O_NOFOLLOW on every restart, so a `node.log -> <root file>`
/// would let root follow it and corrupt that file. For other kinds a symlink is
/// left to the operator-context O_NOFOLLOW / `[ -L ]` probes. A regular file or an
/// absent path is fine.
fn refuse_irregular_log(node_log: &Path, reject_symlink: bool) -> Result<()> {
    if let Ok(m) = std::fs::symlink_metadata(node_log) {
        let ft = m.file_type();
        if ft.is_symlink() {
            if reject_symlink {
                return Err(Error::other(format!(
                    "{} is a SYMLINK — the root daemon's launchd re-opens its log without \
                     O_NOFOLLOW, so a symlink here would redirect root's writes into another \
                     file; remove it and re-run",
                    node_log.display()
                )));
            }
        } else if !ft.is_file() {
            return Err(Error::other(format!(
                "{} exists but is not a regular file ({ft:?}) — a FIFO/socket/device would \
                 block the daemon's log open; remove it and re-run",
                node_log.display()
            )));
        }
    }
    Ok(())
}

/// Refuses a service-critical path a peer could subvert. In order:
/// - a SYMLINK node is SKIPPED (its own mode is meaningless — Linux always lstat's
///   it `0777`; the containing dir + the caller's resolved-chain walk carry the
///   real protection);
/// - UNTRUSTED OWNER → refuse: a dir owned by anyone but root or the operator
///   (`op_uid`) is renamable/replaceable by that owner regardless of its mode;
/// - OTHER-write (and, when `reject_group`, GROUP-write) → refuse, unless exempted
///   by a trusted-owner STICKY bit (`0o1000`) when `allow_sticky` — sticky stops a
///   non-owner from renaming entries, safe for an ANCESTOR *above* the prefix but
///   NOT for a MANAGED leaf where a predictable entry (`current`/`node.log`) can be
///   CREATED. GROUP-write is allowed by default (the credential-group prefix
///   feature); `reject_group` is for paths run/opened as ROOT (the `--system`
///   daemon exec path, its log path, the launchd unit dir), where group-write is a
///   root-escalation vector. A missing path is fine.
fn refuse_writable_mode(
    path: &Path,
    what: &str,
    allow_sticky: bool,
    op_uid: u32,
    reject_group: bool,
) -> Result<()> {
    use std::os::unix::fs::MetadataExt;
    let m = match std::fs::symlink_metadata(path) {
        Ok(m) => m,
        // A DEFINITELY-absent path is fine — a not-yet-created ancestor is created
        // safely later. But an INDETERMINATE error (EACCES/EIO) must NOT fall open:
        // we cannot prove the dir is trusted, so refuse rather than silently allow a
        // possibly-hostile ancestor (the same conservative reading as the plist
        // presence checks). NotFound stays the benign, common case.
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => {
            return Err(Error::other(format!(
                "{what} {} could not be checked ({e}) — refusing rather than assume it is safe; \
                 ensure every parent is readable, then re-run.",
                path.display()
            )))
        }
    };
    // OWNERSHIP first — and it applies to a SYMLINK NODE too, BEFORE the symlink is
    // skipped. A dir OWNED by an untrusted user is a rename/replace vector
    // REGARDLESS of its mode bits — the owner can chmod/rename/relink its entries at
    // will. A plain `0755 /srv/drop` owned by an attacker, with the operator's
    // prefix (or TMPDIR) beneath it, lets that attacker rename the subtree and
    // substitute the service binary / downloaded artifact → code as the operator.
    // And a SYMLINK owned by a peer is a REPOINT vector: in a sticky group-writable
    // ancestor (which the strict walk sticky-EXEMPTS), sticky lets the LINK's owner
    // replace it, so a peer-owned `~/Library/LaunchAgents` symlink would pass both
    // ancestry walks yet be repointed between validation and the write/bootstrap
    // (TOCTOU) — redirecting the plist into another operator-writable dir. So verify
    // the owner (the symlink's OWN lstat uid) BEFORE skipping a symlink's mode bits.
    // Every node on a managed ancestry must be owned by root or the operator.
    let owner_trusted = m.uid() == 0 || m.uid() == op_uid;
    if !owner_trusted {
        return Err(Error::other(format!(
            "{what} {} is owned by an UNTRUSTED uid {} (not root or you, uid {op_uid}) — that \
             user can rename, replace, or (for a symlink) repoint it and run code as you; use a \
             path whose every parent is owned by you or root.",
            path.display(),
            m.uid()
        )));
    }
    // A symlink with a TRUSTED owner is fine — its mode bits are meaningless (the OS
    // governs by the target, covered by the resolved walk), and only its trusted
    // owner can repoint it.
    if m.file_type().is_symlink() {
        return Ok(());
    }
    // Sticky exempts a world-writable dir (its owner is trusted per the check
    // above, so only sticky's own-entry-rename semantics matter here).
    let sticky_safe = m.mode() & 0o1000 != 0;
    let bad = m.mode() & 0o002 != 0 || (reject_group && m.mode() & 0o020 != 0);
    if bad && !(allow_sticky && sticky_safe) {
        let who = if reject_group {
            "GROUP/OTHER-writable"
        } else {
            "OTHER-writable"
        };
        return Err(Error::other(format!(
            "{what} {} is {who} (mode {:o}, owner uid {}) — a local user could replace the \
             daemon's binary/plist, plant its log, or rename this path and run code as you. \
             Run `chmod {} {}` and re-install.",
            path.display(),
            m.mode() & 0o7777,
            m.uid(),
            if reject_group { "go-w" } else { "o-w" },
            path.display()
        )));
    }
    Ok(())
}

/// Refuses a world-writable entry ANYWHERE on `leaf`'s ancestry up to `/`. A dir
/// is MANAGED (no sticky exemption) when it is within `managed_root` (the prefix
/// subtree) OR it directly holds the service's PREDICTABLE entry — `leaf` itself
/// and the directory that CONTAINS it (the version dir holding `higgs`, the logs
/// dir holding `node.log`). Sticky does NOT stop a peer from CREATING that
/// predictable name, so those must be strict EVEN when they resolve OUTSIDE the
/// prefix (e.g. `current` → an external `01777` version dir: a peer pre-creates
/// `higgs` there and the preflight execs it as the operator). Only PURE ANCESTOR
/// dirs above get the sticky exemption (their sole risk is a rename, which sticky
/// blocks). System dirs (`/`, `/home`) are normally `0755` and pass.
fn refuse_writable_ancestry(
    leaf: &Path,
    managed_root: &Path,
    what: &str,
    op_uid: u32,
) -> Result<()> {
    refuse_writable_ancestry_mode(leaf, managed_root, what, op_uid, false)
}

/// [`refuse_writable_ancestry`] that ALSO rejects GROUP-write along the chain —
/// for a leaf OUTSIDE the group-trusted prefix (the launchd unit dir).
fn refuse_writable_ancestry_strict(
    leaf: &Path,
    managed_root: &Path,
    what: &str,
    op_uid: u32,
) -> Result<()> {
    refuse_writable_ancestry_mode(leaf, managed_root, what, op_uid, true)
}

/// Refuse a peer-subvertible `<prefix>/bin` tree before `self-update` stages a new
/// binary into it (DESIGN-remote §9 P3, called from `self_update::stage_and_flip`).
/// A peer who can write into `bin` (or an ancestor), or who holds a write-granting
/// macOS ACL there, could pre-create/​swap the staged `v<ver>/higgs` or repoint
/// `current` between the flip and the operator's restart — running their code as the
/// operator. This applies the SAME hardened walk the install-service exec path uses,
/// in the OPERATOR (non-strict: group-write is the credential-group feature, as for
/// the login-bound agent) context keyed on the CURRENT euid: the LEXICAL `bin` chain,
/// the RESOLVED `bin` chain (a `--prefix` symlinked through an attacker-owned tree),
/// and macOS write-ACLs on both — each up to `/`, with `bin`/prefix MANAGED (no
/// sticky exemption) so a `01777 bin` is refused. Reuses the install path's own
/// helpers so the two surfaces cannot drift. No ACL work off macOS.
/// Refuse a write-granting macOS ACL on `dir` ITSELF (its ancestry is validated
/// separately by [`refuse_unsafe_operator_bin_tree`]). A PRE-EXISTING `v<ver>` dir that a
/// peer was granted `add_file,delete_child` on would let them swap the just-published
/// binary before the operator restarts — even at mode `0755`, because `chmod` does NOT
/// remove an extended ACL. Called after the version dir is (re)created, before publishing
/// into it. No-op off macOS. Reuses the install path's own `has_writable_acl`.
pub(crate) fn refuse_writable_acl_on(dir: &Path, what: &str) -> Result<()> {
    // SAFETY: geteuid has no preconditions and no failure mode.
    let euid = unsafe { libc::geteuid() };
    let op = passwd_by_uid(euid)?;
    if has_writable_acl(dir, &op.user) {
        return Err(Error::other(format!(
            "{what} {} carries a macOS ACL granting write to another user — a peer could swap the \
             published binary through it; remove it (`chmod -N {}`) and re-run",
            dir.display(),
            dir.display()
        )));
    }
    Ok(())
}

pub(crate) fn refuse_unsafe_operator_bin_tree(bin: &Path) -> Result<()> {
    // SAFETY: geteuid has no preconditions and no failure mode.
    let euid = unsafe { libc::geteuid() };
    // The operator is whoever is running this — resolve their passwd name for the
    // ACL self-principal (ACLs name a Directory Service record, not a uid).
    let op = passwd_by_uid(euid)?;
    let prefix = bin.parent().unwrap_or(bin);
    // Resolve BOTH so the symlinked-prefix chain is walked too; refuse (never fall
    // open to lexical-only) on a canonicalize failure — the same race the install
    // path refuses.
    let cprefix = resolve_required(prefix, "install prefix")?;
    let cbin = resolve_required(bin, "install bin dir")?;
    // Mode walks (non-strict — group-write allowed, operator context). `managed_root
    // = prefix` marks `bin`/prefix MANAGED (no sticky exemption); pure ancestors above
    // keep it (sticky blocks their only risk, a rename).
    refuse_writable_ancestry(bin, prefix, "install bin dir", op.uid)?;
    refuse_writable_ancestry(&cbin, &cprefix, "resolved install bin dir", op.uid)?;
    // macOS write-granting ACLs on both chains.
    refuse_writable_acl_ancestry(bin, &op.user, "install bin dir")?;
    refuse_writable_acl_ancestry(&cbin, &op.user, "resolved install bin dir")?;
    Ok(())
}

fn refuse_writable_ancestry_mode(
    leaf: &Path,
    managed_root: &Path,
    what: &str,
    op_uid: u32,
    reject_group: bool,
) -> Result<()> {
    // The directory that will hold the predictable entry: the leaf itself if it
    // is a directory (logs), else its parent (the version dir holding the binary).
    let entry_dir = if leaf.is_dir() {
        Some(leaf.to_path_buf())
    } else {
        leaf.parent().map(Path::to_path_buf)
    };
    let mut cur = Some(leaf);
    while let Some(d) = cur {
        let managed = d.starts_with(managed_root) || d == leaf || entry_dir.as_deref() == Some(d);
        let label = if d == leaf { what } else { "path ancestor" };
        refuse_writable_mode(d, label, !managed, op_uid, reject_group)?;
        cur = d.parent();
    }
    Ok(())
}

/// Whether a REQUIRED-command failure should DELETE the just-written unit file.
/// The rollback exists for ONE case: the agent→daemon switch where the leftover
/// agent plist would join the just-written daemon plist at reboot (two nodes). So
/// three conditions must ALL hold:
/// - a BRAND-NEW launchd plist (`is_launchd && !unit_preexisted`): a REINSTALL must
///   KEEP its new unit so the node still returns (the failure is usually transient
///   — the operator retries); and a systemd unit must stay or its `enable` symlink
///   is left DANGLING (launchd loads plists directly, no symlink, so removing a
///   new one is clean).
/// - the leftover AGENT plist STILL EXISTS (`agent_plist_present`): the rollback is
///   PHASE-AWARE. If the required agent-rm already SUCCEEDED (agent gone) and a
///   LATER command failed (e.g. `bootstrap`), the daemon plist is the ONLY
///   definition left — deleting it would leave NOTHING (no running node, none at
///   reboot). Keep it: it loads at reboot and the node returns. Only when the agent
///   plist survives (rm failed / not yet run) does deleting the daemon plist avoid
///   the two-at-reboot.
fn should_rollback_unit(
    kind: crate::node::service::ServiceKind,
    unit_preexisted: bool,
    agent_plist_present: bool,
) -> bool {
    use crate::node::service::ServiceKind;
    matches!(kind, ServiceKind::LaunchdAgent | ServiceKind::Launchd)
        && !unit_preexisted
        && agent_plist_present
}

/// Roll back a just-written unit file: `unlink` it AND flush the parent directory
/// so the REMOVAL is durable. Without the dir fsync a crash right after the unlink
/// could resurrect the plist (the directory entry not yet on stable storage) and
/// load it at the next reboot BESIDE the surviving service — the exact two-node
/// state the rollback exists to prevent. Returns the human-readable status suffix
/// (` — rolled back …` / ` — COULD NOT roll back …`). The post-unlink dir fsync is
/// best-effort: a failure to flush the unlink is a far narrower window than not
/// unlinking at all (and macOS dir fsync is itself best-effort), and we are already
/// on an abort path — so it is WARNED, not escalated. Only the unlink itself failing
/// downgrades the message to "COULD NOT roll back".
fn rollback_unit_file(unit_path: &Path, unit_dir: &Path) -> String {
    match std::fs::remove_file(unit_path) {
        Ok(()) => {
            if let Err(e) = std::fs::File::open(unit_dir).and_then(|d| d.sync_all()) {
                eprintln!(
                    "warning: rolled back {} but could not flush its removal to disk ({e}); the \
                     unlink likely reached the disk anyway — run `sync` so the rollback survives \
                     an immediate power loss",
                    unit_path.display()
                );
            }
            format!(" — rolled back {}", unit_path.display())
        }
        // The unlink itself FAILED (immutable/ACL/read-only remount/IO): the new
        // plist REMAINS and may load beside the surviving scope — say so rather than
        // falsely claim recovery.
        Err(e) => format!(
            " — COULD NOT roll back {} ({e}); remove it by hand or it may load beside the \
             surviving service at the next reboot/login",
            unit_path.display()
        ),
    }
}

/// True iff `path` is (or resolves through a symlink to) a REGULAR file.
/// `metadata` follows symlinks, so a dangling `current` link returns false.
fn is_regular_file(path: &Path) -> bool {
    std::fs::metadata(path)
        .map(|m| m.is_file())
        .unwrap_or(false)
}

/// True iff the OPERATOR can actually RUN `path` — proven by executing it with
/// `--version` AS the operator (`--version` is handled at the top of main: it
/// prints the version and exits 0, no config/bind/side-effects). This is
/// stronger than an `access(X_OK)` permission check: a `0755` binary that is
/// wrong-architecture, truncated, missing its interpreter, or on a `noexec`
/// mount all PASS `X_OK` but FAIL `execve`. Since install-service is the
/// explicit step where the operator trusts this binary to run (install.sh never
/// executes it), proving it actually starts NOW — before bootout/restart tears
/// down a healthy daemon — is the honest preflight. Root (macOS sudo) drops to
/// the operator so the run has the daemon's real credentials, not root's.
/// A non-regular / missing / dangling path short-circuits to false (no execve).
/// The run is BOUNDED by a timeout: a malformed/malicious binary that hangs in
/// its loader or `--version` handling must not stall install-service forever —
/// on expiry the WHOLE PROCESS GROUP is killed, reaped, and treated as
/// not-runnable. Killing the immediate child alone would orphan any descendants
/// a `--version` handler had forked (`sleep 300 & wait`), leaving them running
/// after install-service returns; the probe runs in its own session/group (via
/// `setsid` in `pre_exec`) so one `kill(-pgid)` reaps the entire subtree.
const EXEC_PREFLIGHT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

fn operator_can_exec(path: &Path, op: &Operator, euid: libc::uid_t) -> Result<bool> {
    use std::os::unix::process::CommandExt;
    if !is_regular_file(path) {
        return Ok(false);
    }
    let mut cmd = std::process::Command::new(path);
    // SCRUBBED env, not just a pinned PATH: this is the one spawn of the
    // UNTRUSTED binary, and the operator's shell env carries transient
    // credentials (SSH_AUTH_SOCK, AWS_*, tokens) that the eventual service —
    // which launchd/systemd start with a minimal env — would never see; a
    // malicious artifact must not get a one-shot read of them here. env_clear
    // also strips loader vars (LD_PRELOAD/DYLD_*), so the preflight execs the
    // binary the way the SERVICE will, not the way the shell would. Allowlist:
    // PATH (trusted) + HOME (what the service sets; `--version` exits at the
    // top of main and reads no state, so HIGGS_HOME is irrelevant here).
    cmd.arg("--version")
        .env_clear()
        .env("PATH", TRUSTED_PATH)
        .env("HOME", &op.home)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());
    // Run the probe in its OWN session/process group, so a `--version` handler
    // that forks descendants can be reaped as a group on timeout. `setsid` makes
    // the child a group leader (pgid == pid), so `kill(-pid)` targets the whole
    // subtree. Registered FIRST (pre_exec runs in registration order), before
    // any credential drop — setsid needs no privilege and cannot fail here (a
    // freshly-forked child is never already a group leader).
    // SAFETY: the closure runs in the forked child and calls only `setsid`, a
    // bare async-signal-safe syscall — no allocation, no inherited-lock risk.
    unsafe {
        cmd.pre_exec(|| {
            if libc::setsid() == -1 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
    // SPAWN (not `status()`): so a hang can be bounded. `?` only propagates a
    // group-resolution failure; execve happens in the spawn.
    let spawned = if euid == 0 {
        // Root: run AS the operator (our own run would use root's credentials).
        drop_to_operator(&mut cmd, op)?.spawn()
    } else {
        cmd.spawn()
    };
    // A spawn/execve failure (EACCES on a non-exec or noexec mount, ENOEXEC on a
    // wrong-arch/truncated image, ENOENT on a missing interpreter) means the
    // operator cannot run it → false, NOT a hard error.
    let mut child = match spawned {
        Ok(c) => c,
        Err(_) => return Ok(false),
    };
    // Kill the probe's whole process group (the child is its leader via setsid),
    // reaping any descendants a `--version` handler forked — orphaned to init
    // otherwise. Done on EVERY exit path, not just timeout: a `--version` that
    // does `sleep 300 &` then exits 0 would leave the sleep running even though
    // the child itself already returned. SAFETY: `kill` is a bare syscall; a
    // negative pid targets the process group; ESRCH (already gone) is ignored.
    let pid = child.id() as libc::pid_t;
    let reap_group = || unsafe {
        libc::kill(-pid, libc::SIGKILL);
    };
    let deadline = std::time::Instant::now() + EXEC_PREFLIGHT_TIMEOUT;
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                reap_group(); // reap any descendants the child left behind
                let _ = child.wait();
                return Ok(status.success());
            }
            Ok(None) => {
                if std::time::Instant::now() >= deadline {
                    reap_group();
                    let _ = child.kill(); // in case the group send raced setsid
                    let _ = child.wait(); // reap the immediate child
                    return Ok(false);
                }
                std::thread::sleep(std::time::Duration::from_millis(25));
            }
            Err(_) => {
                reap_group();
                return Ok(false);
            }
        }
    }
}

/// Runs a `getpw*_r` call with a GROWING buffer. A fixed buffer is wrong for
/// NSS/LDAP-backed users whose record exceeds it: `getpw*_r` returns `ERANGE`
/// (not "no entry"), which must be retried with more space — otherwise a valid
/// operator cannot install the service. Starts at the libc hint
/// (`_SC_GETPW_R_SIZE_MAX`) and doubles on `ERANGE` up to a sane ceiling.
/// `rc == 0 && result.is_null()` is the genuine not-found; any other nonzero
/// (or exhaustion) surfaces `err_msg`.
fn passwd_lookup(
    err_msg: String,
    call: impl Fn(
        *mut libc::passwd,
        *mut libc::c_char,
        libc::size_t,
        *mut *mut libc::passwd,
    ) -> libc::c_int,
) -> Result<Operator> {
    // SAFETY: sysconf(_SC_GETPW_R_SIZE_MAX) has no preconditions; a negative
    // return just means "no hint", handled below.
    let hint = unsafe { libc::sysconf(libc::_SC_GETPW_R_SIZE_MAX) };
    let mut cap: usize = if hint > 0 { hint as usize } else { 4096 };
    loop {
        // SAFETY: `call` (getpw{nam,uid}_r) writes only into the buffers passed;
        // `result` is NULL (not-found/error) or points at `pwd`, whose string
        // fields point into `buf` — both outlive the field copy in
        // operator_from_passwd before this scope drops.
        unsafe {
            let mut pwd: libc::passwd = std::mem::zeroed();
            let mut buf = vec![0u8; cap];
            let mut result: *mut libc::passwd = std::ptr::null_mut();
            let rc = call(&mut pwd, buf.as_mut_ptr().cast(), buf.len(), &mut result);
            if rc == 0 && !result.is_null() {
                return Ok(operator_from_passwd(&pwd));
            }
            if rc == libc::ERANGE && cap < (1 << 20) {
                cap = cap.saturating_mul(2);
                continue;
            }
            return Err(Error::other(err_msg));
        }
    }
}

/// Looks a user up in the passwd database by name via `getpwnam_r`.
fn passwd_by_name(name: &str) -> Result<Operator> {
    let cname = std::ffi::CString::new(name)
        .map_err(|_| Error::other(format!("user name {name:?} contains a NUL byte")))?;
    passwd_lookup(
        format!("no passwd entry for user {name:?}"),
        move |pwd, buf, len, result| unsafe {
            libc::getpwnam_r(cname.as_ptr(), pwd, buf, len, result)
        },
    )
}

/// Looks a user up in the passwd database by uid via `getpwuid_r`.
fn passwd_by_uid(uid: libc::uid_t) -> Result<Operator> {
    passwd_lookup(
        format!("no passwd entry for uid {uid}"),
        move |pwd, buf, len, result| unsafe { libc::getpwuid_r(uid, pwd, buf, len, result) },
    )
}

/// Copies the fields we need OUT of a passwd row (whose strings live in a
/// stack buffer about to drop).
///
/// # Safety
/// `pwd.pw_name` / `pwd.pw_dir` must be valid NUL-terminated strings — true
/// for any row `getpw*_r` reported success for.
unsafe fn operator_from_passwd(pwd: &libc::passwd) -> Operator {
    let user = std::ffi::CStr::from_ptr(pwd.pw_name)
        .to_string_lossy()
        .into_owned();
    let home = std::path::PathBuf::from(
        std::ffi::CStr::from_ptr(pwd.pw_dir)
            .to_string_lossy()
            .into_owned(),
    );
    Operator {
        user,
        home,
        uid: pwd.pw_uid,
        gid: pwd.pw_gid,
    }
}

/// Resolves the operator's FULL supplementary-group list in the PARENT process
/// (before any fork), via `getgrouplist`. This MUST NOT run inside `pre_exec`:
/// `getgrouplist`/`initgroups` consult NSS and may allocate/lock, and after a
/// `fork` in a multithreaded process (higgs is embedded in a tokio runtime) the
/// only async-signal-safe operations in the child are bare syscalls — an NSS or
/// malloc lock held by another thread at fork time would deadlock the child.
/// `drop_to_operator` therefore feeds this pre-built array to `setgroups`, which
/// IS a plain syscall.
fn operator_group_list(user: &std::ffi::CStr, gid: libc::gid_t) -> Result<Vec<libc::gid_t>> {
    // Back the buffer with gid_t (what `setgroups` needs); `getgrouplist` fills
    // `*mut gid_t` on Linux but `*mut c_int` on macOS — both 32-bit, so cast the
    // pointer for the query on macOS.
    let mut count: libc::c_int = 32;
    loop {
        let mut buf = vec![0 as libc::gid_t; count.max(1) as usize];
        let mut n = buf.len() as libc::c_int;
        // SAFETY: `buf` holds `n` gid_t slots; getgrouplist writes at most `n`
        // entries and updates `n` to the actual/required count.
        let rc = unsafe {
            #[cfg(target_os = "macos")]
            {
                libc::getgrouplist(
                    user.as_ptr(),
                    gid as _,
                    buf.as_mut_ptr() as *mut libc::c_int,
                    &mut n,
                )
            }
            #[cfg(not(target_os = "macos"))]
            {
                libc::getgrouplist(user.as_ptr(), gid, buf.as_mut_ptr(), &mut n)
            }
        };
        if rc >= 0 {
            buf.truncate(n.max(0) as usize);
            // getgrouplist always includes the primary gid; keep it defensive.
            if !buf.contains(&gid) {
                buf.push(gid);
            }
            return Ok(buf);
        }
        // rc == -1: buffer too small; `n` now holds the needed size. Grow and
        // retry, guarding against a non-advancing count and a runaway loop.
        count = if n > count {
            n
        } else {
            count.saturating_mul(2)
        };
        if count > 65536 {
            return Err(Error::other("operator has an implausible number of groups"));
        }
    }
}

/// Arranges for `cmd` to run with the OPERATOR's FULL credential set — the
/// same one launchd/systemd give the daemon (`UserName`): the supplementary
/// groups (resolved in the PARENT by [`operator_group_list`]), then the primary
/// gid, then the uid. `CommandExt::uid`/`gid` alone would `setgroups(0, NULL)`,
/// dropping the operator's secondary groups — so a prefix writable only through
/// a group (e.g. `root:higgs 0770`) would be falsely rejected by the probe.
fn drop_to_operator<'a>(
    cmd: &'a mut std::process::Command,
    op: &Operator,
) -> Result<&'a mut std::process::Command> {
    use std::os::unix::process::CommandExt;
    let user = std::ffi::CString::new(op.user.as_bytes())
        .map_err(|_| Error::other("operator name contains a NUL byte"))?;
    let (uid, gid) = (op.uid, op.gid);
    // Pin a trusted PATH so `sh`/`mkdir` resolve from system dirs, never a
    // planted binary on the inherited PATH (this can run under sudo/root).
    cmd.env("PATH", TRUSTED_PATH);
    // Resolve groups NOW, in the parent — see operator_group_list's note.
    let groups = operator_group_list(&user, gid)?;
    // SAFETY: the closure runs in the forked child and calls ONLY async-signal-
    // safe syscalls (setgroups/setgid/setuid) on the pre-resolved `groups` Vec
    // it owns — no allocation, no NSS, so no inherited-lock deadlock.
    unsafe {
        cmd.pre_exec(move || {
            if libc::setgroups(groups.len() as _, groups.as_ptr()) != 0 {
                return Err(std::io::Error::last_os_error());
            }
            if libc::setgid(gid) != 0 {
                return Err(std::io::Error::last_os_error());
            }
            if libc::setuid(uid) != 0 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
    Ok(cmd)
}

/// Parses the major version out of `systemctl --user show --property=Version`
/// output, whose relevant line is `Version=NNN` (the value may carry a build
/// suffix on newer systemd, e.g. `Version=254 (254.1-2)` or a leading `v`).
/// Returns `None` if no such line is present.
fn parse_systemd_show_version(show_output: &str) -> Option<u32> {
    let value = show_output
        .lines()
        .find_map(|l| l.trim().strip_prefix("Version="))?;
    let digits: String = value
        .trim_start_matches('v')
        .chars()
        .take_while(char::is_ascii_digit)
        .collect();
    digits.parse().ok()
}

/// Best-effort probe of the RUNNING user manager's major version, over D-Bus —
/// NOT `systemctl --version`, which reports the CLIENT binary. After a systemd
/// package upgrade that didn't reexec a lingering old `--user` manager, the
/// client would read new while the manager that actually parses the unit is
/// old. `None` when there is no reachable user manager or the output is
/// unparseable — callers must not hard fail on `None` (a truly absent manager
/// surfaces later when the plan's `systemctl --user` commands run).
fn systemd_manager_major() -> Option<u32> {
    let out = std::process::Command::new("systemctl")
        .env("PATH", TRUSTED_PATH) // resolve systemctl from a trusted dir
        .args(["--user", "show", "--property=Version"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    parse_systemd_show_version(&String::from_utf8_lossy(&out.stdout))
}

/// `higgs node self-update [--url <manifest-url> | --tarball <f> --manifest <f>
/// --manifest-sig <f>] [--prefix <dir>] [--allow-downgrade] [--dry-run] [--rollback]
/// [--prune]` — verified, atomic binary swap of this node's install (DESIGN-remote §9
/// P3 apply core + P4 network fetch, `self_update.rs`).
///
/// The UPDATE path needs a VERIFIED source: either `--url <manifest-url>` to FETCH the
/// signed manifest + its `.minisig` + the tarball over HTTPS (P4, `HttpSource`), or the
/// local `--tarball`/`--manifest`/`--manifest-sig` triple the operator fetched by other
/// means (scp, `curl`, a shared mount). Either way it verifies the manifest against a
/// pinned release key, checks eligibility (newer version, matching target+variant),
/// stages, smoke-tests, and atomically flips `current`. A dev/default build pins NO key,
/// so the update path refuses at verification (HG081) even after a successful fetch —
/// self-update is impossible until a release build pins a key. The hub-initiated
/// M_UPDATE push (and the jigglebot UI) build on this same fetch+apply core.
///
/// `--url` must be a STABLE, DIRECT manifest URL whose `.minisig` and artifact live as
/// path-siblings; redirects are NOT followed (that would open an SSRF surface and make the
/// sibling URLs ambiguous), and the manifest URL's own directory is where the `.minisig` +
/// artifact are fetched from — matching the M_UPDATE `artifact_url` (a direct URL, §9). A
/// REDIRECTING source (e.g. a GitHub Release asset URL, which 302s to storage) surfaces a
/// clear error: fetch those with the local `--tarball`/`--manifest`/`--manifest-sig` triple
/// (the operator retrieves the assets with `curl`/`gh`, which follow redirects, then passes
/// the files) — the same way `install.sh` consumes the GitHub release channel.
///
/// `--rollback` (repoint `current` to the recorded previous version) and `--prune`
/// (drop old version dirs, keeping current + the rollback target) need no signature
/// — they only rearrange already-installed version dirs. `--dry-run` verifies +
/// checks eligibility and reports, touching nothing.
fn run_node_self_update(args: &[String]) -> Result<()> {
    use crate::node::self_update as su;
    // Refuse to run as root: the node service runs as the UNPRIVILEGED operator (even
    // the macOS --system LaunchDaemon is pinned to the operator via UserName), so the
    // install tree, the update lock, and the staged binary must all be owned by that
    // user. A `sudo self-update` would create a root-owned 0600 lock the operator-run
    // daemon can't open (its boot hooks would then read every acquire as contention and
    // never roll back) and would judge a root-owned target executable via root's own
    // access(). Run it AS the operator.
    // SAFETY: geteuid has no preconditions and no failure mode.
    if unsafe { libc::geteuid() } == 0 {
        return Err(Error::other(
            "refusing to run `self-update` as root — run it as the node's operator user (the \
             unprivileged user the service runs as), so the update artifacts, lock, and staged \
             binary are owned by that user",
        ));
    }
    let mut prefix: Option<std::path::PathBuf> = None;
    let mut tarball: Option<std::path::PathBuf> = None;
    let mut manifest: Option<std::path::PathBuf> = None;
    let mut manifest_sig: Option<std::path::PathBuf> = None;
    let mut url: Option<String> = None;
    let mut allow_downgrade = false;
    let mut dry_run = false;
    let mut do_rollback = false;
    let mut do_prune = false;
    // A value-taking path flag: non-empty, and a leading `-` is a swallowed flag
    // (`--tarball --dry-run` errors rather than reading a file literally named
    // `--dry-run`) — same guard shape as install-service's dir_value.
    fn path_value(flag: &str, val: Option<&String>) -> Result<std::path::PathBuf> {
        let v = val.ok_or_else(|| Error::other(format!("{flag} needs a path")))?;
        if v.is_empty() {
            return Err(Error::other(format!("{flag} must not be empty")));
        }
        if v.starts_with('-') {
            return Err(Error::other(format!(
                "{flag} expected a path but got the flag {v:?} — a value is missing"
            )));
        }
        Ok(std::path::PathBuf::from(v))
    }
    // A value-taking string flag (--url), same non-empty + swallowed-flag guard. The
    // rejected value is NEVER echoed: a `--url` value can carry a secret (a capability path
    // or a credential), so printing it in an error would leak it to stderr/logs.
    fn str_value(flag: &str, val: Option<&String>) -> Result<String> {
        let v = val.ok_or_else(|| Error::other(format!("{flag} needs a value")))?;
        if v.is_empty() {
            return Err(Error::other(format!("{flag} must not be empty")));
        }
        if v.starts_with('-') {
            return Err(Error::other(format!(
                "{flag} expected a value but the next token looks like a flag (starts with \
                 '-') — a value is missing"
            )));
        }
        Ok(v.clone())
    }
    // --prefix specifically must be made ABSOLUTE before it derives `bin`: a RELATIVE
    // prefix truncates the writable-ancestry walk (the lexical walk stops at `.`/the
    // relative root and never visits the working directory's real ancestry, so a
    // world-writable cwd — or an operator-owned symlink inside it that a peer replaces
    // after validation — would be missed). `std::path::absolute` is lexical (works
    // before the dir exists); a `..` is then rejected (kept verbatim, would misdirect).
    fn dir_value(flag: &str, val: Option<&String>) -> Result<std::path::PathBuf> {
        let v = path_value(flag, val)?;
        let abs = std::path::absolute(&v)
            .map_err(|e| Error::other(format!("{flag} {}: {e}", v.display())))?;
        if abs
            .components()
            .any(|c| c == std::path::Component::ParentDir)
        {
            return Err(Error::other(format!(
                "{flag} {} contains a `..` component — pass a resolved absolute path",
                v.display()
            )));
        }
        Ok(abs)
    }
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--prefix" => prefix = Some(dir_value("--prefix", it.next())?),
            "--tarball" => tarball = Some(path_value("--tarball", it.next())?),
            "--manifest" => manifest = Some(path_value("--manifest", it.next())?),
            "--manifest-sig" => manifest_sig = Some(path_value("--manifest-sig", it.next())?),
            "--url" => url = Some(str_value("--url", it.next())?),
            "--allow-downgrade" => allow_downgrade = true,
            "--dry-run" => dry_run = true,
            "--rollback" => do_rollback = true,
            "--prune" => do_prune = true,
            _ => {
                // NEVER echo the unrecognized token: a bare positional (a manifest URL passed
                // WITHOUT `--url`), `--url=<secret>`, a dash-prefixed `-https://…`, or an opaque
                // flag-shaped token could all be a capability, and syntax cannot prove a token
                // is non-secret. Print only the usage (which names the valid flags).
                return Err(Error::other(
                    "unrecognized argument — usage: higgs node self-update [--url <manifest-url> \
                     | --tarball <f> --manifest <f> --manifest-sig <f>] [--prefix <dir>] \
                     [--allow-downgrade] [--dry-run] [--rollback] [--prune] (pass the manifest \
                     URL as `--url <manifest-url>`, space-separated)",
                ));
            }
        }
    }
    if do_rollback && do_prune {
        return Err(Error::other(
            "--rollback and --prune are separate maintenance ops — run one at a time",
        ));
    }
    // Default `bin` = the `<prefix>/bin` of the install THIS binary runs from,
    // derived from `current_exe` — the SAME dir the boot-guard uses, so a bare
    // `self-update` always targets the running installation. An explicit --prefix
    // (a non-standard layout, or a different install) overrides. NOT `higgs_home()`:
    // the STATE dir can differ from where the BINARY is installed (--prefix vs
    // --higgs-home at install time), and updating the state dir's `bin/` would touch
    // the wrong (or a nonexistent) tree.
    let bin = match prefix {
        Some(p) => su::bin_dir(&p),
        None => self_update_bin_dir().ok_or_else(|| {
            Error::other(
                "cannot locate this binary's install dir (bin/v<ver>/higgs layout) — pass \
                 --prefix <dir> to point self-update at the installation to update",
            )
        })?,
    };
    // Map a typed self-update error to the CLI's io::Result surface.
    let as_io = |e: crate::diagnostic::HiggsError| Error::other(e.to_string());

    if do_rollback {
        // --dry-run reports the target without flipping; the real op takes the update
        // lock so it can't interleave with a concurrent stage/flip.
        if dry_run {
            let prev = su::rollback_target(&bin).map_err(as_io)?;
            println!("dry-run: would roll current back to {prev}");
            return Ok(());
        }
        let _lock = su::UpdateLock::acquire(&bin).map_err(as_io)?;
        let prev = su::rollback(&bin).map_err(as_io)?;
        println!("rolled current back to {prev}");
        return Ok(());
    }
    if do_prune {
        // --dry-run lists what WOULD be removed without deleting; the real op takes the
        // update lock so it can't remove a version dir a concurrent update is flipping to.
        if dry_run {
            let plan = su::prune_plan(&bin).map_err(as_io)?;
            if plan.is_empty() {
                println!("dry-run: nothing to prune");
            } else {
                println!("dry-run: would prune {}", plan.join(", "));
            }
            return Ok(());
        }
        let _lock = su::UpdateLock::acquire(&bin).map_err(as_io)?;
        let pruned = su::prune(&bin).map_err(as_io)?;
        if pruned.is_empty() {
            println!("nothing to prune");
        } else {
            println!("pruned {}", pruned.join(", "));
        }
        return Ok(());
    }

    // UPDATE path — a verified source: either a network URL (--url, P4) or the local
    // --tarball/--manifest/--manifest-sig triple (P3). Both feed the SAME verify → check
    // → stage → flip pipeline; the network bytes are signature+sha256 verified exactly
    // like local ones (a dev build pins no key, so either still fails closed at HG081).
    let source: Box<dyn su::UpdateSource> = match url {
        Some(u) => {
            if tarball.is_some() || manifest.is_some() || manifest_sig.is_some() {
                return Err(Error::other(
                    "pass EITHER --url <manifest-url> to fetch, OR the local --tarball/--manifest/\
                     --manifest-sig triple — not both",
                ));
            }
            Box::new(su::HttpSource::new(&u).map_err(as_io)?)
        }
        None => {
            let (Some(tarball), Some(manifest), Some(manifest_sig)) =
                (tarball, manifest, manifest_sig)
            else {
                return Err(Error::other(
                    "self-update needs a source: --url <manifest-url> to fetch it, or \
                     --tarball <f> --manifest <f> --manifest-sig <f> for local files. \
                     --rollback / --prune need no source.",
                ));
            };
            Box::new(su::LocalSource {
                manifest,
                manifest_sig,
                tarball,
            })
        }
    };
    // Judge the update against the version INSTALLED as `current` (the binary being
    // replaced), NOT this invoking process — a stale old binary run directly must not
    // be able to flip `current` to something older than what is installed.
    let running = su::installed_identity(&bin);
    // The CLI update runs with the operator watching the terminal — the failure is shown directly,
    // not persisted for a later HELLO — so the authenticated-version out-param is discarded.
    let verified =
        su::verify_and_check(source.as_ref(), &running, allow_downgrade, &mut None, None)
            .map_err(as_io)?;
    println!(
        "verified update {} -> {} (key {}, target {}, variant {})",
        running.version,
        verified.manifest.version,
        verified.key_id,
        verified.manifest.target,
        verified.manifest.variant
    );
    if dry_run {
        println!(
            "dry-run: authentic + eligible; nothing staged. Re-run without --dry-run to apply."
        );
        return Ok(());
    }
    // Serialize the stage->flip critical section against a concurrent self-update.
    let _lock = su::UpdateLock::acquire(&bin).map_err(as_io)?;
    su::stage_and_flip(
        &bin,
        &verified.manifest,
        &verified.artifact,
        allow_downgrade,
        &su::smoke_run,
    )
    .map_err(as_io)?;
    println!(
        "staged and flipped current -> v{}. Restart the node service to run it; the boot-guard \
         rolls back automatically if it crash-loops, or `higgs node self-update --rollback` \
         reverts it manually.",
        verified.manifest.version
    );
    Ok(())
}

/// `higgs node install-service [--dry-run] [--prefix <dir>] [--higgs-home <dir>]
/// [--model-dir <dir>] [--system]` — install the node service for this platform.
/// USER-SPACE, LOGIN-BOUND BY DEFAULT (macOS LaunchAgent / Linux systemd user
/// unit without linger — no sudo, no prompts); `--system` opts into always-on
/// (macOS LaunchDaemon needing sudo / the same Linux unit + enable-linger).
/// `--dry-run` prints exactly what would be written and run — the inspection
/// surface for a `sudo`-shy operator, and the integration tests' seam (no
/// system state is touched).
fn run_node_install_service(args: &[String]) -> Result<()> {
    let mut dry_run = false;
    let mut prefix: Option<std::path::PathBuf> = None;
    let mut higgs_home_flag: Option<std::path::PathBuf> = None;
    let mut model_dir_flag: Option<std::path::PathBuf> = None;
    // Preserved deployment-config env from --env KEY=VALUE flags (sudo-proof
    // argv path; KEY must be an allowlisted PRESERVED_ENV var — the flag can
    // NEVER inject arbitrary env like LD_PRELOAD into the daemon).
    let mut env_flags: Vec<(String, String)> = Vec::new();
    // USER-SPACE BY DEFAULT: LoginBound (macOS LaunchAgent / Linux user unit
    // without linger — no sudo, no prompts). `--system` opts into
    // SurvivesLogout (macOS LaunchDaemon needing sudo; Linux the same user
    // unit + enable-linger).
    let mut scope = crate::node::service::ServiceScope::LoginBound;
    let mut it = args.iter();
    // Shared parse for the two value-taking dir flags: non-empty, a leading `-`
    // is a swallowed flag (`--prefix --dry-run` must error, not install into a
    // dir literally named after the flag — a real such dir would be written
    // `./-h`, matching install.sh's value guard), and made ABSOLUTE now — a
    // relative path would render a relative `ExecStart`/state dir, which
    // systemd rejects / the daemon would resolve against its own cwd.
    // `absolute` is purely lexical, so it works before the dir exists.
    fn dir_value(flag: &str, val: Option<&String>) -> Result<std::path::PathBuf> {
        let dir = val.ok_or_else(|| Error::other(format!("{flag} needs a directory")))?;
        if dir.is_empty() {
            return Err(Error::other(format!("{flag} must not be empty")));
        }
        if dir.starts_with('-') {
            return Err(Error::other(format!(
                "{flag} expected a directory but got the flag {dir:?} — a value is missing"
            )));
        }
        let abs =
            std::path::absolute(dir).map_err(|e| Error::other(format!("{flag} {dir:?}: {e}")))?;
        // Reject a `..` component: `std::path::absolute` is purely lexical and
        // KEEPS `..` (it does not resolve it), but systemd's
        // `StandardOutput=append:` requires a NORMALIZED path and silently
        // IGNORES a directive containing `..` — so a `--prefix …/x/../y` would
        // run the daemon (ExecStart's execve resolves `..`) yet send its logs
        // NOWHERE. Refuse LOUDLY here rather than lose logs silently; the
        // operator passes a resolved absolute path.
        if abs
            .components()
            .any(|c| c == std::path::Component::ParentDir)
        {
            return Err(Error::other(format!(
                "{flag} {dir:?} contains a `..` component — pass a resolved absolute path (a \
                 `..` in a service path is kept verbatim and would silently misdirect the \
                 daemon's logs)"
            )));
        }
        Ok(abs)
    }
    // Pin a bounded umask for the whole install: the `logs/` dir (and the
    // node.log the daemon later recreates) must never be group/world-writable,
    // even under a permissive caller `umask 000`. A world-writable log dir on a
    // multi-user host lets a local peer pre-plant node.log or tamper with logs;
    // the same bound also flows to the `mkdir` child on the macOS root path
    // (a child inherits the parent's umask). File modes we set explicitly
    // (0644 unit, 0644 log) are unaffected; this only floors the DIR modes.
    // SAFETY: umask has no preconditions and cannot fail.
    unsafe { libc::umask(0o022) };
    while let Some(a) = it.next() {
        match a.as_str() {
            "--dry-run" => dry_run = true,
            "--prefix" => prefix = Some(dir_value("--prefix", it.next())?),
            // The STATE dir override as ARGV: unlike the HIGGS_HOME env var,
            // an argument survives `sudo` under EVERY sudoers policy (env vars
            // on a sudo command line are rejected by command-specific/NOSETENV
            // rules), so the printed elevation re-runs carry the state dir
            // through this flag.
            "--higgs-home" => higgs_home_flag = Some(dir_value("--higgs-home", it.next())?),
            // The extra MODEL SCAN ROOT as ARGV (`HIGGS_MODEL_DIR`, the
            // README's documented pairing knob) — same sudo-proof rationale.
            "--model-dir" => model_dir_flag = Some(dir_value("--model-dir", it.next())?),
            // A preserved deployment-config var as sudo-proof ARGV: `--env
            // KEY=VALUE`, KEY restricted to the PRESERVED_ENV allowlist (an
            // operator must never inject LD_PRELOAD/DYLD_* into a daemon that
            // may run as root-installed). Used by the printed elevation re-runs
            // and install.sh's Next-steps to carry a value sudo would strip.
            "--env" => {
                let kv = it
                    .next()
                    .ok_or_else(|| Error::other("--env needs KEY=VALUE"))?;
                let (k, v) = kv
                    .split_once('=')
                    .ok_or_else(|| Error::other(format!("--env expects KEY=VALUE, got {kv:?}")))?;
                if !crate::node::service::PRESERVED_ENV.contains(&k) {
                    return Err(Error::other(format!(
                        "--env {k:?} is not a preserved config var — allowed: {}",
                        crate::node::service::PRESERVED_ENV.join(", ")
                    )));
                }
                if v.is_empty() {
                    return Err(Error::other(format!("--env {k}= has an empty value")));
                }
                // Last --env for a key wins; keep it a set, not a multimap.
                env_flags.retain(|(ek, _)| ek != k);
                env_flags.push((k.to_string(), v.to_string()));
            }
            // Opt-in: outlive the login session (boot-start + logout survival).
            "--system" => scope = crate::node::service::ServiceScope::SurvivesLogout,
            other => {
                return Err(Error::other(format!(
                    "unknown flag {other:?} — usage: higgs node install-service [--dry-run] \
                     [--prefix <dir>] [--higgs-home <dir>] [--model-dir <dir>] [--env KEY=VALUE] \
                     [--system]"
                )))
            }
        }
    }

    let op = operator_identity()?;
    let prefix = prefix.unwrap_or_else(|| op.home.join(".higgs"));
    // SAFETY: geteuid has no preconditions and no failure mode.
    let euid = unsafe { libc::geteuid() };
    // The STATE dir the daemon must read — pinned into the service env so a node
    // paired under a custom HIGGS_HOME does not boot against ~/.higgs. Separate
    // from --prefix (binary/logs). Precedence: the --higgs-home FLAG (sudo-proof
    // argv) wins over the HIGGS_HOME env var, which wins over `~/.higgs` — and
    // an AMBIGUOUS default (HOME ≠ passwd home, empty HIGGS_HOME) refuses
    // rather than guesses.
    let higgs_home = match higgs_home_flag {
        Some(h) => h,
        None => operator_higgs_home(&op, euid)?,
    };
    // The optional extra model scan root, mirroring the runtime's own read
    // (non-empty `HIGGS_MODEL_DIR`, empty treated as unset — cli.rs pairing
    // does exactly that); flag beats env; absolutized so the daemon's cwd
    // can't reinterpret it. Unset stays unset — there is no default scan-root
    // override, so no ambiguity to refuse.
    let model_dir = model_dir_flag.or_else(|| {
        std::env::var_os("HIGGS_MODEL_DIR")
            .filter(|v| !v.is_empty())
            .map(|v| std::path::absolute(&v).unwrap_or_else(|_| std::path::PathBuf::from(v)))
    });
    // The preserved deployment-config env: for each allowlisted PRESERVED_ENV
    // var, a `--env` FLAG wins (sudo-proof); else the var's value from THIS
    // env if set & non-empty (mirrors the runtime's own read — empty is unset).
    // Deterministic order (allowlist order) so the rendered service + re-run
    // hint are stable.
    let extra_env: Vec<(String, String)> = crate::node::service::PRESERVED_ENV
        .iter()
        .filter_map(|&k| {
            if let Some((_, v)) = env_flags.iter().find(|(ek, _)| ek == k) {
                Some((k.to_string(), v.clone()))
            } else {
                std::env::var_os(k)
                    .filter(|v| !v.is_empty())
                    .map(|v| (k.to_string(), v.to_string_lossy().into_owned()))
            }
        })
        .collect();
    // Reject a `..` component in the RESOLVED state/model/prefix dirs, whatever
    // the source. `dir_value` already guards the FLAGS, but higgs_home/model_dir
    // can also come from the HIGGS_HOME/HIGGS_MODEL_DIR ENV or the
    // `<home>/.higgs` default (a `..` in `$HOME`), which bypass it. systemd
    // keeps `..` verbatim and silently misdirects `append:` logs, so catch it
    // here on the final values — one authoritative check for every source.
    reject_dotdot("--prefix / prefix", &prefix)?;
    reject_dotdot("HIGGS_HOME", &higgs_home)?;
    if let Some(d) = model_dir.as_deref() {
        reject_dotdot("HIGGS_MODEL_DIR", d)?;
    }
    // No systemd discovery here: the systemd plan LINKS an absolute unit path,
    // so the manager decides where it lives (no XDG/UnitPath guessing) — and no
    // `systemctl` is spawned before the root gate (a PATH-injection surface in a
    // `sudo` run).
    let mut plan = crate::node::service::plan_install(
        &op.user,
        &op.home,
        op.uid,
        &prefix,
        &higgs_home,
        model_dir.as_deref(),
        &extra_env,
        scope,
    );
    // Linux LoginBound + linger already on: the unit would still survive
    // logout — say so (with the opt-out) rather than let the default lie.
    if matches!(plan.kind, crate::node::service::ServiceKind::SystemdUser)
        && matches!(scope, crate::node::service::ServiceScope::LoginBound)
    {
        if let Some(n) =
            crate::node::service::linger_note(Path::new("/var/lib/systemd/linger"), &op.user)
        {
            plan.notes.push(n);
        }
    }

    if dry_run {
        println!("would write {}:", plan.unit_path.display());
        println!("---");
        print!("{}", plan.unit_content);
        println!("---");
        for cmd in &plan.commands {
            if cmd.as_operator {
                println!("would run (as the operator): {}", cmd.argv.join(" "));
            } else {
                println!("would run: {}", cmd.argv.join(" "));
            }
        }
        for n in &plan.notes {
            println!("note: {n}");
        }
        return Ok(());
    }

    // Verify the binary the unit will exec is present AND executable BY THE
    // OPERATOR — BEFORE any destructive step or destructive GUIDANCE. launchd
    // has no Type=exec acknowledgement: a KeepAlive job over a missing/typoed/
    // non-exec binary spawn-fails forever, and the `launchctl bootout` in the
    // plan has already stopped the old one. systemd's Type=exec fails `restart`,
    // but only AFTER the old unit is stopped — so refuse up front on both. This
    // runs ahead of the root gate so a bad --prefix fails before the sudo
    // prompt, and stays observable without root. It also runs BEFORE the
    // cross-scope conflict refusal below: that refusal tells the operator to
    // TEAR DOWN a healthy daemon, which they must not be asked to do for a
    // typoed `--prefix` whose binary does not even exist.
    // Refuse a WORLD-writable dir ANYWHERE on the PHYSICAL paths the service will
    // exec or write, BEFORE trusting them. `umask` bounds only NEW dirs, so a
    // pre-existing 0777 anywhere on a chain slips through. We validate the FULL
    // physical ancestry, not just endpoints: a world-writable ANCESTOR lets a peer
    // rename/replace the subtree below it and run code as the operator (or
    // redirect the daemon's log). Managed dirs (the prefix and below) are STRICT;
    // ancestors ABOVE the prefix are sticky-exempt (sticky blocks the only attack
    // there, a rename). GROUP-write is allowed (the credential-group prefix).
    let exec_target = prefix.join("bin").join("current").join("higgs");
    // The PHYSICAL prefix — the managed-vs-ancestor sticky boundary for every walk
    // below. Canonicalize resolves a symlinked `--prefix` to the real dir the walks
    // must actually guard; REFUSE on failure (`resolve_required`) rather than fall
    // back to the lexical prefix — an unresolvable prefix is broken or a race.
    let cprefix = resolve_required(&prefix, "service prefix")?;
    // A macOS `--system` LaunchDaemon is INSTALLED via `sudo` — the printed
    // Next-steps runs `sudo <prefix>/bin/current/higgs … --system`, so the prefix
    // binary is exec'd as ROOT. GROUP-write on the exec path (the credential-group
    // feature, fine when the binary runs as the OPERATOR) is then a ROOT-escalation
    // vector: a group peer swaps `current`/`higgs` and the next `sudo …/higgs`
    // runs their payload as root. So the DAEMON exec ancestry rejects group-write
    // too; the login-bound agent / systemd unit run as the operator (no sudo),
    // where group-write is the safe credential-group feature.
    let daemon = exec_ancestry_rejects_group(plan.kind);
    let exec_ancestry = |leaf: &Path, root: &Path, what: &str| -> Result<()> {
        if daemon {
            refuse_writable_ancestry_strict(leaf, root, what, op.uid)
        } else {
            refuse_writable_ancestry(leaf, root, what, op.uid)
        }
    };
    // (1) The physical PREFIX itself + its ancestors. Covers a world-writable
    //     prefix AND — when `--prefix` is a SYMLINK — the real target dir + its
    //     ancestors, which (2)/(3) can miss: the lexical walk SKIPS the symlink
    //     node, and the resolved walk follows `current`, which may resolve OUTSIDE
    //     the prefix and never revisit it.
    exec_ancestry(&cprefix, &cprefix, "service prefix")?;
    // (2) LEXICAL exec chain (`<prefix>/bin/current/…` as spelled) — catches an
    //     other-writable `bin` even if `current` is repointed AFTER canonicalize
    //     (the TOCTOU the resolved walk alone would miss).
    exec_ancestry(&exec_target, &prefix, "service binary path")?;
    // (3) RESOLVED exec chain — through the `current` symlink to the real version
    //     dir + binary + every ancestor up to `/`. REFUSE on a canonicalize failure
    //     (`resolve_required`): a peer who renames the resolved version dir away
    //     during this walk would otherwise SKIP it, then restore a genuine binary for
    //     the `--version` probe and swap a payload before the root `sudo …` exec.
    let resolved_exec = resolve_required(&exec_target, "resolved service binary")?;
    exec_ancestry(&resolved_exec, &cprefix, "resolved service binary")?;
    // (3b) macOS ACLs for the DAEMON: the mode-bit walks miss an ACL that GRANTS a
    //      peer write on the exec/log paths — a peer could then swap the binary
    //      (root install / operator re-exec) or plant `node.log` as a symlink (root
    //      launchd re-opens it without O_NOFOLLOW → root writes into the target).
    //      Walk write-granting ACLs across the FULL ancestry of the prefix AND the
    //      exec path — BOTH the LEXICAL chain (as the service/`sudo …` re-run spell
    //      it) and the RESOLVED chain (`refuse_exec_acls`), mirroring the mode-bit
    //      walks (2)+(3): an ACL on a 0755 LEXICAL parent reached through an
    //      operator-owned symlink passes the mode bits AND a resolved-only ACL walk,
    //      yet lets a peer repoint that symlink before the root exec. A non-inheriting
    //      ACL on an ANCESTOR (e.g. /Users/alice), not just the managed endpoints, is
    //      also a swap vector. Safe to walk to `/`: only allow-to-non-owner write ACLs
    //      are flagged, and system dirs carry `deny` ACLs (ignored). No-op off macOS.
    if daemon {
        refuse_writable_acl_ancestry(&cprefix, &op.user, "service prefix")?;
        refuse_exec_acls(&exec_target, &op.user)?;
        // A logs dir that EXISTS but won't resolve (a symlink to a hidden target)
        // refuses; absent → skip (the daemon creates it fresh later). Built on the
        // already-canonical `cprefix` (NOT the lexical `prefix`) so an ENOENT means
        // the logs LEAF is absent — never a vanished intermediate prefix symlink,
        // which would be an ambiguous ENOENT read as a benign skip.
        if let Some(real_logs) = resolve_if_present(&cprefix.join("logs"), "daemon logs dir")? {
            // The daemon's logs dir must resolve UNDER the prefix (an external log
            // dir's node.log-symlink-plant vector is hard to bound); the prefix
            // ancestry ACL-walk above covers logs' ancestry once it is under it.
            if !real_logs.starts_with(&cprefix) {
                return Err(Error::other(format!(
                    "the daemon's logs dir resolves OUTSIDE the prefix ({}) — refuse: an \
                     external/symlinked log dir's ancestry can't be safely validated (a peer \
                     could redirect the root daemon's log). Keep logs under {}.",
                    real_logs.display(),
                    cprefix.display()
                )));
            }
            refuse_writable_acl_ancestry(&real_logs.join("node.log"), &op.user, "node.log")?;
        }
    }
    // (4) LOGS — the daemon creates the predictable `node.log` here, so a
    //     world-writable logs dir OR any world-writable ANCESTOR of it (which lets
    //     a peer rename the dir and plant node.log as a symlink) is refused.
    //     Resolve physically first (`refuse_*` skips a symlink node, and logs has
    //     no resolved-walk backstop) then walk the whole ancestry. For the macOS
    //     `--system` DAEMON, ROOT launchd re-opens `StandardOutPath`/`Error` on
    //     every restart WITHOUT O_NOFOLLOW — the SAME root-context open the exec
    //     path is hardened for — so the log path is GROUP-STRICT too (a group peer
    //     on a group-writable logs/ could plant `node.log -> a root file` and have
    //     root launchd corrupt it). Absent → the daemon creates it fresh under
    //     umask 022 (nothing yet).
    // Resolve-or-refuse (absent → skip, created fresh; exists-but-unresolvable →
    // refuse rather than skip the walk on a symlinked-away logs dir). Built on the
    // canonical `cprefix` so an ENOENT is a genuine leaf-absence, never a vanished
    // intermediate prefix symlink misread as a benign skip.
    if let Some(real_logs) = resolve_if_present(&cprefix.join("logs"), "log dir")? {
        exec_ancestry(&real_logs, &cprefix, "log dir")?;
    }
    // (5) node.log — a pre-existing OTHER-writable log (a stale attacker-owned
    //     mode-0666 file) passes the openability probes below (append/R-W both
    //     succeed on a group/other-writable file) yet then lets ANY local user
    //     forge or truncate the daemon's logs, or hold it open to fill the state
    //     filesystem. For the daemon (root launchd open), GROUP-write is refused
    //     too. Refuse it here, in the preflight (BEFORE the binary/launchd steps).
    //     Absent → fine (created fresh).
    let node_log_pre = prefix.join("logs").join("node.log");
    refuse_writable_mode(&node_log_pre, "node.log", false, op.uid, daemon)?;
    // A SYMLINK node.log: the agent (operator context) is left to the O_NOFOLLOW /
    // `[ -L ]` probes, but ROOT launchd re-opens the DAEMON's log with NO
    // O_NOFOLLOW on every restart, so a symlinked `node.log` would let root follow
    // it into an operator/root file — refuse it up front for the daemon.
    refuse_irregular_log(&node_log_pre, daemon)?;
    if !operator_can_exec(&exec_target, &op, euid)? {
        return Err(Error::other(format!(
            "{} is not an executable file for the operator — install the binary first (run \
             install.sh, or fix --prefix). Refusing to (re)install the service over a missing \
             or non-executable binary.",
            exec_target.display()
        )));
    }

    // NOTE: concurrent install-service runs are NOT serialized (an flock attempt was
    // removed — see the residual note). The cross-scope guards below handle the
    // realistic SEQUENTIAL same-operator case; a lock to close the exotic concurrent
    // window introduced a worse NFS-home regression (flock on a read-only dir fd is
    // emulated via fcntl over NFS and needs a WRITABLE fd → fails) and could not cover
    // the cross-operator (Alice-agent + Bob-daemon) two-node case anyway.
    //
    // Cross-scope guard: no silent dual-run when switching daemon → agent. AFTER
    // the binary preflight, so we never send the operator to remove a working
    // daemon for an install that would then fail on a missing binary.
    if matches!(plan.kind, crate::node::service::ServiceKind::LaunchdAgent) {
        let daemon_plist = format!(
            "/Library/LaunchDaemons/{}.plist",
            crate::node::service::SERVICE_NAME
        );
        // A bootstrapped launchd job SURVIVES deletion of its plist, so
        // plist-existence alone misses a daemon left LOADED by a partial cleanup
        // (the printed `bootout ; rm` where `bootout` failed but `rm` succeeded).
        // Also refuse when the job is still LOADED — detected by `launchctl print
        // system/<label>`'s EXIT CODE (readable without root, no output parsing).
        if daemon_job_loaded() {
            return Err(Error::other(format!(
                "a --system LaunchDaemon ({}) is still LOADED (even if its plist was removed) — \
                 the login-bound agent would run ALONGSIDE it. Boot it out first:  sudo \
                 launchctl bootout system/{}   (then `sudo rm {}` if the plist remains), and \
                 re-run.",
                crate::node::service::SERVICE_NAME,
                crate::node::service::SERVICE_NAME,
                daemon_plist
            )));
        }
        crate::node::service::refuse_daemon_conflict(Path::new(&daemon_plist))?;
    }

    // Symmetric cross-scope guard for the DAEMON install: the plan switches an
    // agent→daemon by tearing the agent down, but that is safe ONLY for a CLEAN
    // switch (no pre-existing daemon plist). If a daemon plist ALREADY exists AND
    // an agent plist is ALSO present, a reinstall keeps the pre-existing daemon
    // plist (rollback suppressed), so an agent-`rm` failure after the plan's
    // `enable` leaves BOTH loadable → two nodes at reboot. Refuse the mixed state.
    if matches!(plan.kind, crate::node::service::ServiceKind::Launchd) {
        crate::node::service::refuse_cross_scope_coexistence(
            &plan.unit_path,
            &crate::node::service::agent_plist_path(&op.home),
            &op.home,
        )?;
    }

    // Re-run command that PRESERVES the resolved --prefix + the RESOLVED state
    // dir: dropping --prefix would target the default prefix, and dropping the
    // state dir would silently pin the DEFAULT into the service (sudo strips the
    // operator's env, so a copy-pasted `sudo …` re-run of a custom-HIGGS_HOME
    // install would otherwise install a crash-looping daemon pointed at ~/.higgs).
    // The state dir travels as the `--higgs-home` ARGV FLAG, not a `HIGGS_HOME=`
    // prefix: sudo rejects command-line env vars under command-specific/NOSETENV
    // sudoers policies (only `ALL` rules grant implicit SETENV), while arguments
    // always pass. Shell-quoted so paths with spaces stay one word.
    //
    // The binary is `exec_target` — the PREFLIGHTED `<prefix>/bin/current/higgs`
    // whose full ancestry was just strict-validated (for the daemon, group-write
    // rejected) — NOT `current_exe()`: the running binary may sit in a
    // group-writable build/share dir a peer could swap between this hint and the
    // operator copying the `sudo …` command → root. The prefix binary is the one
    // the service will exec anyway, so pinning it here is both safe and correct.
    let rerun = format!(
        "{} node install-service --prefix {} --higgs-home {}{}{}",
        shell_single_quote(&exec_target.to_string_lossy()),
        shell_single_quote(&prefix.to_string_lossy()),
        shell_single_quote(&higgs_home.to_string_lossy()),
        // The model scan root AND every preserved config var ride along the
        // same way when present — an elevated re-run must not silently drop
        // them (empty daemon inventory / wrong HF endpoint). Each `--env` KEY
        // is an allowlisted literal; the VALUE is shell-quoted.
        model_dir
            .as_deref()
            .map(|d| format!(" --model-dir {}", shell_single_quote(&d.to_string_lossy())))
            .unwrap_or_default()
            + &extra_env
                .iter()
                .map(|(k, v)| format!(" --env {}", shell_single_quote(&format!("{k}={v}"))))
                .collect::<String>(),
        // --system rides along too: a copy-pasted re-run must not silently
        // change the service SCOPE from always-on back to login-bound.
        match scope {
            crate::node::service::ServiceScope::SurvivesLogout => " --system",
            crate::node::service::ServiceScope::LoginBound => "",
        }
    );
    match plan.root {
        crate::node::service::RootRequirement::Require if euid != 0 => {
            return Err(Error::other(format!(
                "--system installs a macOS LaunchDaemon in /Library/LaunchDaemons, which needs \
                 root — re-run: sudo {rerun}   (preview first by adding --dry-run; or drop \
                 --system for the no-sudo, login-bound LaunchAgent)"
            )));
        }
        crate::node::service::RootRequirement::Refuse if euid == 0 => {
            // Both user-space paths (macOS LaunchAgent, Linux systemd user
            // unit) install into the OPERATOR's own session/manager — a root
            // run would target root's gui domain / root's user manager and
            // leave files the operator cannot later update.
            let what = match plan.kind {
                crate::node::service::ServiceKind::LaunchdAgent => {
                    "the macOS LaunchAgent installs into your own ~/Library/LaunchAgents and \
                     your login session"
                }
                _ => "the Linux systemd user service targets your own user manager",
            };
            return Err(Error::other(format!(
                "{what} — re-run WITHOUT sudo: {rerun}   (use --system if you want the \
                 always-on variant)"
            )));
        }
        _ => {}
    }

    // The systemd unit relies on `Type=exec` AND `StandardOutput=append:`, both
    // introduced in systemd 240 (Jan 2019). On an older manager systemd would
    // ignore them (append: logging silently lost; a broken binary NOT surfaced
    // by `restart`) yet still report enable/restart success — so refuse up
    // front rather than install a unit that looks healthy but is not. Absent /
    // unparseable systemctl is left to fail later when the plan's commands run.
    if matches!(plan.kind, crate::node::service::ServiceKind::SystemdUser) {
        if let Some(v) = systemd_manager_major() {
            if v < 240 {
                return Err(Error::other(format!(
                    "systemd {v} is too old for this unit — it needs >= 240 (Jan 2019) for \
                     Type=exec and append: logging (the rendered unit uses BOTH, so installing \
                     it by hand would hit the same limits). Upgrade systemd to >= 240, or run \
                     `higgs --node` under a supervisor of your choice."
                )));
            }
        }
    }

    // Logs dir: the daemon (running as the operator) and the operator's own
    // CLI runs both write here, so it must be operator-owned. When root
    // (macOS), CREATE IT AS THE OPERATOR by dropping to their uid/gid in a
    // `mkdir -p` child — a chown-as-root would follow an operator-planted
    // ANCESTOR symlink (`~/.higgs -> /somewhere`) to a system path, but a
    // mkdir running with the operator's own credentials can only ever create
    // dirs the operator could already create. No chown, no symlink games.
    let logs = prefix.join("logs");
    let node_log = logs.join("node.log");
    if euid == 0 {
        // Create the dir AS THE OPERATOR (`mkdir -p` succeeding proves nothing
        // about writability if it already exists root-owned).
        let st = drop_to_operator(
            std::process::Command::new("mkdir").arg("-p").arg(&logs),
            &op,
        )?
        .status()?;
        if !st.success() {
            return Err(Error::other(format!(
                "could not create {} as {} (uid {}) — check the prefix is under the operator's \
                 own writable tree",
                logs.display(),
                op.user,
                op.uid
            )));
        }
        // Prove the operator can CREATE files in logs/ (a pre-existing root-owned
        // dir passes `mkdir -p` but blocks the daemon from recreating node.log
        // after rotation). `: > "$1/.f" && rm` creates+removes a temp under the
        // operator's creds; the path is a positional arg, never interpolated.
        let st = drop_to_operator(
            std::process::Command::new("sh")
                .arg("-c")
                // `mktemp` creates a UNIQUE file with O_EXCL semantics (random
                // name, never following a symlink) — proving dir-writability
                // without a predictable-name symlink race a group peer could win.
                .arg("t=\"$(mktemp \"$1/.higgs-writeprobe.XXXXXX\")\" && rm -f \"$t\"")
                .arg("higgs-install-service")
                .arg(&logs),
            &op,
        )?
        .status()?;
        if !st.success() {
            return Err(Error::other(format!(
                "{} is not writable by {} (uid {}) — likely a stale root-owned log dir from a \
                 prior `sudo install.sh`; the daemon could not recreate node.log after a log \
                 rotation. Fix its ownership (or remove {}) and re-run",
                logs.display(),
                op.user,
                op.uid,
                logs.display()
            )));
        }
        // OPEN node.log READ/WRITE as the operator, non-truncating. Sources
        // DISAGREE on launchd's exact open mode for a shared StandardOut/Error
        // path (`launchd.plist(5)` describes StandardErrorPath as opened
        // read/write; some launchd source shows O_WRONLY|O_APPEND). We pick the
        // STRICTER R/W probe on purpose — it is the SAFE-FAILURE choice: if
        // launchd needs read, this catches a mode-0200 log NOW with a clear
        // error instead of a silent "daemon won't start"; if launchd needs only
        // write, the only thing R/W over-rejects is a write-only (mode-0200)
        // log, which no one legitimately creates (install.sh's own append-open
        // makes node.log 0644). `touch` is weaker still (a mode-0400 file passes
        // on a timestamp update). `exec 3<> "$1"` does O_RDWR|O_CREAT (NO
        // truncate — existing log content is preserved) and exits; the path is a
        // positional arg, never interpolated into the script text. `umask 022`
        // pins a CREATED node.log at 0644: O_CREAT's mode is masked by the
        // inherited umask, and a restrictive one (0777) would otherwise create a
        // file even the OWNER cannot reopen — passing this probe once but
        // failing the daemon's own open later. An existing file is untouched
        // (umask only applies at create).
        //
        // GUARD a pre-planted symlink first: POSIX shell redirection has no
        // O_NOFOLLOW, so `exec 3<>` would FOLLOW a `node.log -> ~/.higgs/config`
        // and later corrupt that file (the Linux path uses O_NOFOLLOW; see
        // `ensure_appendable_log`). `[ -L "$1" ]` rejects it. The test→exec window
        // is a couple of shell builtins wide; `logs/` is already verified NOT
        // world-writable, so only the operator (or, on a credential-group prefix,
        // a group peer) could swap it in that window — a documented residual, not
        // a world-open race.
        let st = drop_to_operator(
            std::process::Command::new("sh")
                .arg("-c")
                .arg(
                    "[ -L \"$1\" ] && { echo \"$1 is a symlink\" >&2; exit 66; }; \
                     umask 022; exec 3<> \"$1\"",
                )
                .arg("higgs-install-service")
                .arg(&node_log),
            &op,
        )?
        .status()?;
        if !st.success() {
            return Err(Error::other(format!(
                "{} is not read/write-openable by {} (uid {}) — likely a stale root-owned or \
                 mode-restricted log from a prior `sudo install.sh`; remove {} and re-run",
                node_log.display(),
                op.user,
                op.uid,
                logs.display()
            )));
        }
    } else {
        std::fs::create_dir_all(&logs)?;
        // Prove logs/ is operator-writable (a pre-existing root-owned dir passes
        // create_dir_all but blocks node.log recreation after rotation).
        probe_dir_writable(&logs)?;
        // Prove the daemon (this same user) can OPEN its log the way ITS service
        // manager will, surfacing a stale root-owned / mode-restricted file now
        // rather than as silent no-logging later. Selected by PLATFORM, NOT
        // elevation: macOS launchd opens `StandardErrorPath` READ/WRITE, so the
        // non-root LaunchAgent (this branch on macOS) must get the R/W probe — an
        // append-only probe would pass a write-only `node.log` / ACL that launchd
        // then fails to open, and a REINSTALL boots out the working agent before
        // discovering it. Linux systemd `append:`-opens it. First create pins
        // 0644 regardless of umask (see probe_log_openable).
        let probe = if cfg!(target_os = "macos") {
            ensure_rw_log(&node_log)
        } else {
            ensure_appendable_log(&node_log)
        };
        probe.map_err(|e| {
            Error::other(format!(
                "{} is not openable by the daemon ({}): {e} — remove {} and re-run",
                node_log.display(),
                if cfg!(target_os = "macos") {
                    "launchd opens it read/write"
                } else {
                    "systemd append-opens it"
                },
                logs.display()
            ))
        })?;
    }

    if let Some(dir) = plan.unit_path.parent() {
        std::fs::create_dir_all(dir)?;
        // Validate the unit dir's ancestry before publishing the plist: if
        // `~/Library/LaunchAgents` (or a lexical/resolved ancestor) is writable by
        // another user — via mode bits OR a macOS ACL that the atomic 0644 write
        // cannot strip — a peer could DELETE or REPLACE the installed plist and have
        // launchd exec code as the operator at the next login. `validate_unit_dir`
        // walks BOTH chains and refuses a canonicalize race (see its doc).
        validate_unit_dir(dir, plan.kind, op.uid, &op.user)?;
    }
    // Did a unit already exist at this path? A REINSTALL replaces a working unit;
    // an abort must NOT delete it (that would leave the node absent). Captured
    // BEFORE the atomic rename overwrites it, so the failure rollback below only
    // removes a BRAND-NEW plist. `symlink_metadata` so a broken symlink still reads
    // as "present" (we never silently delete something at the target).
    let unit_preexisted = std::fs::symlink_metadata(&plan.unit_path).is_ok();
    // Write ATOMICALLY: a temp file in the same dir + rename over the target,
    // so a crash or full disk mid-write can never leave a truncated unit that
    // survives to the next manager restart. The temp is same-directory (so the
    // rename stays on one filesystem) and created O_EXCL under a RANDOM name: on
    // a group-writable prefix a predictable name would let a group peer pre-plant
    // a symlink and have this write+chmod land on an operator-owned file.
    let unit_dir = plan.unit_path.parent().unwrap_or_else(|| Path::new("."));
    let base = format!(
        ".{}.tmp",
        plan.unit_path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default()
    );
    let tmp_unit = unit_dir.join(temp_name(&base));
    {
        use std::io::Write;
        use std::os::unix::fs::PermissionsExt;
        // O_CREAT|O_EXCL — fails rather than follow/clobber a pre-planted path.
        let mut f = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp_unit)?;
        f.write_all(plan.unit_content.as_bytes()).inspect_err(|_| {
            let _ = std::fs::remove_file(&tmp_unit);
        })?;
        // Force mode 0644 regardless of the caller's umask: launchd REFUSES a
        // group/world-writable plist, and a systemd unit must not be writable
        // by a shared group either. Owner-write, everyone-read. Applied
        // THROUGH THE OPEN HANDLE (fchmod — File::set_permissions), NOT a
        // path-based chmod after close: re-resolving the path would open a
        // window where a dir-writable group peer swaps the (visible, if
        // random-named) temp for a symlink and the chmod lands on some other
        // operator-owned file (e.g. loosening ~/.ssh keys to 0644). The fd can
        // only ever chmod the file O_EXCL just created.
        f.set_permissions(std::fs::Permissions::from_mode(0o644))
            .inspect_err(|_| {
                let _ = std::fs::remove_file(&tmp_unit);
            })?;
        // Flush the CONTENT to disk before the rename makes it live: rename
        // orders metadata, not data — without this, a crash after the rename
        // can leave a zero-length unit/plist that only surfaces at the next
        // reboot (the manager commands below read the page cache and succeed).
        // sync_all also forces a delayed-allocation ENOSPC to fail HERE, on the
        // temp, instead of corrupting the installed unit. (Same discipline as
        // install.sh's staged-data `sync` before its publish rename.)
        f.sync_all().inspect_err(|_| {
            let _ = std::fs::remove_file(&tmp_unit);
        })?;
    }
    if let Err(e) = std::fs::rename(&tmp_unit, &plan.unit_path) {
        let _ = std::fs::remove_file(&tmp_unit);
        return Err(e);
    }
    // Flush the DIRECTORY entry so the completed rename itself survives a
    // crash (fsync on the dir fd; fully effective on Linux — the fleet-node
    // target — best-effort on macOS, matching install.sh's `sync` residual).
    // A failure is WARNED, not fatal — a DECIDED middle ground: the unit is
    // already correct in the running system (the manager commands below read
    // it fine), and aborting a functionally-complete install on an fs that
    // cannot fsync a directory (ENOTSUP-class) would be worse than the narrow
    // power-loss window; but swallowing an EIO/ENOSPC silently would report
    // durability the disk refused — so the operator is told, with the remedy.
    if let Err(e) = std::fs::File::open(unit_dir).and_then(|d| d.sync_all()) {
        if dir_fsync_should_abort(e.raw_os_error()) {
            // A REAL storage error (EIO / ENOSPC / …): the rename was NOT made
            // durable, so a crash could lose the unit or restore stale state.
            // ABORT before the destructive service-manager commands rather than
            // proceed on an unflushed unit. (The file content was already
            // sync_all'd; only the dir entry's durability failed.)
            //
            // ROLL BACK the just-renamed plist first, with the SAME phase-aware rule
            // the command loop uses: the plist is already on disk (if the rename
            // later reaches stable storage it would LOAD at reboot), and the
            // agent→daemon switch's agent-plist rm has NOT run yet, so a surviving
            // agent plist + this new daemon plist would be two nodes. Remove the new
            // launchd plist so only the survivor remains.
            // CONSERVATIVE presence: an INDETERMINATE metadata error (EIO/EACCES)
            // counts as PRESENT, not absent. The same transient fault that fails the
            // required agent `rm` can also fail this probe — reading the agent as
            // GONE would suppress the rollback and leave the new daemon plist beside
            // the surviving agent (two nodes once the fs recovers). Only a DEFINITE
            // NotFound reads as absent.
            let agent_present = crate::node::service::plist_present_under(
                &crate::node::service::agent_plist_path(&op.home),
                &op.home,
            );
            let rolled = if should_rollback_unit(plan.kind, unit_preexisted, agent_present) {
                // Remove the new plist AND fsync the dir so the rollback is durable
                // (this abort is itself a REAL storage error, so the removal's own
                // flush may fail too — reported best-effort inside the helper).
                rollback_unit_file(&plan.unit_path, unit_dir)
            } else {
                String::new()
            };
            return Err(Error::other(format!(
                "could not flush {} to disk ({e}) — the unit's rename was not made durable (a \
                 real storage error, not merely an fs that can't fsync a directory); refusing \
                 to proceed before the service-manager steps{rolled}. Check disk space/errors \
                 and re-run.",
                unit_dir.display()
            )));
        }
        eprintln!(
            "warning: could not flush {} to disk ({e}) — this filesystem does not support \
             directory fsync; the unit is written but may not survive an immediate power loss; \
             run `sync` to be safe",
            unit_dir.display()
        );
    }
    let style = crate::node::preflight::Style::auto();
    println!(
        "{}",
        style.ok(&format!("wrote {}", plan.unit_path.display()))
    );

    for cmd in &plan.commands {
        let (prog, rest) = cmd.argv.split_first().expect("plan commands are non-empty");
        // Pin a trusted PATH: these (launchctl/systemctl/loginctl) can run as
        // root (macOS), so a bare name must NEVER resolve to a planted binary.
        let mut command = std::process::Command::new(prog);
        command.env("PATH", TRUSTED_PATH).args(rest);
        // `as_operator` commands touch paths under the OPERATOR's home — when
        // elevated, run them with the operator's dropped credentials so an
        // ancestor symlink can never redirect a root fs operation (the
        // operator can only touch what they already own). A non-elevated run
        // already IS the operator.
        let status = if cmd.as_operator && euid == 0 {
            drop_to_operator(&mut command, &op).and_then(std::process::Command::status)
        } else {
            command.status()
        };
        let joined = cmd.argv.join(" ");
        // On a launchd-agent (headless-Mac) failure, point at the --system fix
        // (the required-command failure returns before the plan notes).
        let headless_hint = crate::node::service::agent_headless_hint(plan.kind);
        // ROLL BACK the just-written unit file if a REQUIRED command fails, but
        // ONLY when it is a BRAND-NEW launchd plist. A mid-sequence abort (e.g. an
        // immutable leftover agent plist the required rm can't remove during an
        // agent→daemon switch, where the daemon plist is new) would otherwise leave
        // this new definition on disk to LOAD at the next reboot ALONGSIDE the
        // surviving agent plist — two nodes; removing it makes the survivor the
        // only one. But NOT when we REPLACED a pre-existing unit (a reinstall — an
        // abort must keep the intended new config so the node still returns; the
        // operator retries), and NOT for systemd (deleting the unit would leave its
        // `enable` symlink DANGLING; launchd loads plists directly, no symlink, so
        // removing a new one is clean). Best-effort commands never trigger this.
        let rollback_unit = || {
            // PHASE-AWARE: only roll back while the leftover agent plist still
            // exists (checked live, at the moment of failure). If the agent→daemon
            // switch's required rm already removed it and a LATER command failed,
            // the daemon plist is the only definition left — keep it so the node
            // returns at reboot rather than deleting it and leaving nothing.
            // CONSERVATIVE presence: an INDETERMINATE metadata error (EIO/EACCES)
            // counts as PRESENT, not absent. The same transient fault that fails the
            // required agent `rm` can also fail this probe — reading the agent as
            // GONE would suppress the rollback and leave the new daemon plist beside
            // the surviving agent (two nodes once the fs recovers). Only a DEFINITE
            // NotFound reads as absent.
            let agent_present = crate::node::service::plist_present_under(
                &crate::node::service::agent_plist_path(&op.home),
                &op.home,
            );
            if should_rollback_unit(plan.kind, unit_preexisted, agent_present) {
                // Remove the new plist AND fsync the dir so the removal is durable —
                // otherwise a crash could resurrect it beside the survivor at reboot
                // (the two-node state this rollback prevents).
                rollback_unit_file(&plan.unit_path, unit_dir)
            } else {
                // Reinstall (kept so the node returns), systemd (kept to avoid a
                // dangling enable symlink), or the agent already torn down (keep the
                // daemon plist — it is the only definition left): nothing to remove.
                String::new()
            }
        };
        match status {
            Ok(st) if st.success() => println!("{}", style.ok(&format!("ran: {joined}"))),
            Ok(st) if cmd.best_effort => {
                println!("{}", style.warn(&format!("skipped ({st}): {joined}")))
            }
            Ok(st) => {
                let rolled = rollback_unit();
                return Err(Error::other(format!(
                    "command failed ({st}): {joined}{rolled}; fix the manager state and \
                     re-run{headless_hint}"
                )));
            }
            Err(e) if cmd.best_effort => {
                println!("{}", style.warn(&format!("skipped: {joined} ({e})")))
            }
            Err(e) => {
                let rolled = rollback_unit();
                return Err(Error::other(format!(
                    "could not run {joined}: {e}{rolled}{headless_hint}"
                )));
            }
        }
    }
    print!("{}", render_plan_notes(&style, &plan.notes));
    Ok(())
}

/// Renders the install plan's notes for a human: the short `key: value`
/// quick-reference lines (logs/state/status/…) become an aligned block with
/// bold keys, and each long advisory paragraph gets its own `!`-marked entry
/// separated by a blank line — instead of one undifferentiated wall of text.
/// Purely presentational: every note string is rendered verbatim (colors are
/// tty-gated via [`crate::node::preflight::Style`], so pipes/logs see plain
/// bytes).
fn render_plan_notes(style: &crate::node::preflight::Style, notes: &[String]) -> String {
    use std::fmt::Write as _;
    // The quick-reference keys the plans emit (`service.rs` + the models:/config:
    // lines `plan_install` inserts). Anything else is prose guidance.
    const KV_KEYS: &[&str] = &["logs", "state", "models", "config", "status", "stop"];
    fn kv(n: &str) -> Option<(&str, &str)> {
        n.split_once(':')
            .filter(|(k, _)| KV_KEYS.contains(&k.trim()))
    }
    let width = notes
        .iter()
        .filter_map(|n| kv(n).map(|(k, _)| k.trim().len()))
        .max()
        .unwrap_or(0);
    let mut out = String::new();
    let mut in_kv_block = false;
    for n in notes {
        if let Some((k, v)) = kv(n) {
            if !in_kv_block {
                out.push('\n');
                in_kv_block = true;
            }
            let _ = writeln!(
                out,
                "  {}{} {}",
                style.head(&format!("{}:", k.trim())),
                " ".repeat(width - k.trim().len()),
                v.trim()
            );
        } else {
            out.push('\n');
            in_kv_block = false;
            let _ = writeln!(out, "{}", style.warn(n));
        }
    }
    out
}

/// `higgs node leave [--hub <label|id>]` — self-retire: dial the saved hub (default, or the one
/// selected by `--hub`), ask it to retire this node, and on success forget that hub locally so a
/// bare `higgs --node` no longer dials it. Nodes persist by default; this is the explicit opt-out
/// (the node-side counterpart of the operator's hub-side Retire).
fn run_node_leave(args: &[String]) -> Result<()> {
    let cfg_path = config_path()?;
    let cfg = InstanceConfig::load(&cfg_path)?;
    let hub = match args.first().map(String::as_str) {
        Some("--hub") => {
            let sel = args
                .get(1)
                .ok_or_else(|| Error::other("usage: higgs node leave --hub <label|id>"))?;
            cfg.find_hub(sel).ok_or_else(|| {
                Error::other(format!(
                    "no saved hub matching {sel:?} — see `higgs --node --list`"
                ))
            })?
        }
        Some(other) => {
            return Err(Error::other(format!(
                "unknown flag {other:?} — usage: higgs node leave [--hub <label|id>]"
            )))
        }
        None => cfg
            .default_saved_hub()
            .ok_or_else(|| Error::other("no saved hub to leave — nothing to do"))?,
    };
    let hub_id = hub.hub_id.clone();
    let ticket_str = hub.ticket.clone();
    let ticket: EndpointTicket = ticket_str.parse().map_err(Error::other)?;
    let target = ticket.endpoint_addr().clone();

    let rt = runtime()?;
    rt.block_on(async {
        let sk = load_or_create_secret(&key_path()?)?;
        let endpoint = bind_endpoint(sk).await.map_err(Error::other)?;
        let self_id = endpoint.id().to_string();
        let name = name_or_init(Role::Node, &self_id, &crate::system::hostname())?;
        let (conn, hello) =
            crate::node::connect_node(&endpoint, target, self_id, name, None).await?;
        println!(
            "connected to hub {} ({})",
            crate::remote::sanitize_display(&hello.hub_name),
            hello.node_id
        );
        crate::node::send_leave(&conn).await?;
        println!(
            "left hub {} — retired from its fleet",
            crate::remote::sanitize_display(&hello.hub_name)
        );
        Ok::<(), Error>(())
    })?;

    // Retired hub-side; forget it locally (re-load so a concurrent name write isn't clobbered).
    let mut cfg = InstanceConfig::load(&cfg_path)?;
    cfg.remove_hub(&hub_id);
    cfg.save(&cfg_path)?;
    println!("removed {hub_id} from saved hubs");
    Ok(())
}

/// The full `source()` chain of an error, ` → `-joined. The reconnect log
/// must say WHY a dial failed — the top-level message alone ("timed out")
/// hides the actionable cause underneath.
fn error_chain(e: &dyn std::error::Error) -> String {
    let mut out = e.to_string();
    let mut cur = e.source();
    while let Some(cause) = cur {
        out.push_str(" → ");
        out.push_str(&cause.to_string());
        cur = cause.source();
    }
    out
}

fn run_node_connect(args: &[String]) -> Result<()> {
    let ticket_str = args
        .first()
        .ok_or_else(|| Error::other("usage: higgs node connect <ticket> [token]"))?;
    let token = args.get(1).cloned();
    let ticket: EndpointTicket = ticket_str.parse().map_err(Error::other)?;
    let target = ticket.endpoint_addr().clone();

    let rt = runtime()?;
    rt.block_on(async {
        let sk = load_or_create_secret(&key_path()?)?;
        let endpoint = bind_endpoint(sk).await.map_err(Error::other)?;
        let self_id = endpoint.id().to_string();
        let name = name_or_init(Role::Node, &self_id, &crate::system::hostname())?;
        println!("higgs node   : {name} ({self_id})");
        let res = dial_and_hello(&endpoint, target, self_id, name, token).await?;
        // Both hub-controlled strings are display-sanitized (T14 r24 — the
        // one-shot pairing path is the FIRST thing a user runs against an
        // untrusted ticket, exactly where terminal spoofing matters most).
        let label = res
            .assigned_label
            .as_deref()
            .map(crate::remote::sanitize_display)
            .unwrap_or_else(|| "-".into());
        println!(
            "paired with hub {} ({}) (protocol v{}, label {label})",
            crate::remote::sanitize_display(&res.hub_name),
            res.node_id,
            res.agreed_version,
        );
        Ok(())
    })
}

/// Re-load `config.json`, record this hub as the default, and persist — best-effort, called
/// once after the FIRST successful admission so a later bare `higgs --node` reconnects to it.
/// Re-loading (rather than mutating a stale in-memory copy) preserves whatever `name_or_init`
/// wrote concurrently. The hub's id/label come from its HELLO result (authoritative); the
/// `ticket` is the exact string we dialed. A persistence failure is logged, never fatal — the
/// node stays connected regardless.
/// How often the bare (service-run) daemon re-reads config.json while waiting to be
/// paired — this poll IS the seamless pairing handoff (see the wait loop's comment).
const HUB_WAIT_POLL: Duration = Duration::from_secs(3);
/// Reminder cadence while waiting, in polls (~5 minutes at [`HUB_WAIT_POLL`]).
const HUB_WAIT_REMIND_EVERY: u64 = 100;
/// Pairing one-shot: attempts and per-attempt budget. Two attempts ride out a transient
/// relay/holepunch flake; the preflight already vetoed hopeless environments.
const PAIR_ATTEMPTS: u32 = 2;
const PAIR_ATTEMPT_TIMEOUT: Duration = Duration::from_secs(30);

/// Stop flag for the bare-wait pairing loop's plain signal handler. The handler stays
/// installed until the serve path's tokio `shutdown_listener` replaces the disposition —
/// no `SIG_DFL` gap — and the async block re-checks the flag right after the boot
/// record, so a stop landing in the handover window still exits gracefully.
static WAIT_STOP: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
extern "C" fn wait_stop(_sig: libc::c_int) {
    WAIT_STOP.store(true, std::sync::atomic::Ordering::SeqCst);
}

/// True when the pairing process runs against a NON-DEFAULT `HIGGS_HOME`. The installed
/// node service always manages the DEFAULT home, so a custom-home node (tests, side-by-
/// side experiments) is its own process — pairing must never hand off to a service that
/// manages a different state directory.
fn custom_higgs_home() -> bool {
    if std::env::var_os("HIGGS_HOME").is_none() {
        return false;
    }
    // MIRROR the runtime exactly: `home::higgs_home()` treats ANY set value —
    // including the empty string — as the override, so compare what the runtime
    // would actually use against the default (same `dirs::home_dir()` fallback,
    // so a scrubbed $HOME can't misclassify an explicitly-default HIGGS_HOME).
    // An explicit spelling of ~/.higgs is still the default; anything else
    // (empty included) is a custom home.
    let default = dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".higgs");
    crate::home::higgs_home() != default
}

/// Absolute service-manager binary (never the ambient PATH — same trust posture as the
/// install-service exec path).
fn service_manager_bin() -> &'static str {
    if cfg!(target_os = "macos") {
        "/bin/launchctl"
    } else if Path::new("/usr/bin/systemctl").exists() {
        "/usr/bin/systemctl"
    } else {
        "/bin/systemctl"
    }
}

/// True when a higgs node service exists for this user — INSTALLED (macOS agent/daemon
/// plist) or LIVE in the service manager (a loaded launchd job / an active-or-enabled
/// systemd unit, which can outlive its unit file). Both sides matter: a plist with a
/// dead job must still hand off (the restart below revives it), and a loaded job whose
/// plist was removed must NOT be shadowed by a second foreground node.
fn service_present() -> bool {
    if custom_higgs_home() {
        return false; // the service manages the DEFAULT home, not this one
    }
    // NOTE: no config-parsing home checks here. A service pinned to a DIFFERENT
    // state dir has a different node key, never supersedes our connection, and the
    // behavioral takeover-verify then keeps us serving in the foreground — the
    // wrong-home case self-corrects without reading plists/unit files.
    if cfg!(target_os = "macos") {
        agent_installed() || system_daemon_installed() || service_alive()
    } else {
        let unit = crate::node::service::SYSTEMD_UNIT;
        let run = |arg: &str| {
            std::process::Command::new(service_manager_bin())
                .args(["--user", arg, "--quiet", unit])
                .status()
                .is_ok_and(|s| s.success())
        };
        run("is-enabled") || run("is-active")
    }
}

/// The user LaunchAgent plist exists for this user.
fn agent_installed() -> bool {
    dirs::home_dir()
        .map(|h| crate::node::service::agent_plist_path(&h))
        .is_some_and(|p| p.exists())
}

/// The system LaunchDaemon plist exists (root-managed, unreachable without sudo).
fn system_daemon_installed() -> bool {
    Path::new("/Library/LaunchDaemons")
        .join(format!("{}.plist", crate::node::service::SERVICE_NAME))
        .exists()
}

/// True when the service manager reports the node service LOADED/ACTIVE right now.
fn service_alive() -> bool {
    if cfg!(target_os = "macos") {
        let domain_loaded = |domain: String| {
            std::process::Command::new(service_manager_bin())
                .args(["print", &domain])
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status()
                .is_ok_and(|s| s.success())
        };
        // gui domain (the agent), plus a BEST-EFFORT system-domain probe: it catches a
        // loaded root daemon whose plist was removed where macOS allows unprivileged
        // reads; where it refuses, the behavioral takeover-verify decides anyway.
        domain_loaded(format!(
            "gui/{}/{}",
            unsafe { libc::getuid() },
            crate::node::service::SERVICE_NAME
        )) || domain_loaded(format!("system/{}", crate::node::service::SERVICE_NAME))
    } else {
        std::process::Command::new(service_manager_bin())
            .args([
                "--user",
                "is-active",
                "--quiet",
                crate::node::service::SYSTEMD_UNIT,
            ])
            .status()
            .is_ok_and(|s| s.success())
    }
}

/// How long the pairing waits for the restarted/waiting service to TAKE OVER —
/// i.e. for the hub to supersede our admitted connection with the service's own
/// dial. Covers the manager respawn throttle (5s), the waiting service's 3s
/// config poll, and a relay-path dial.
const HANDOFF_TAKEOVER_WINDOW: Duration = Duration::from_secs(20);

/// The hub replaces a duplicate-identity dial by closing the OLD connection with
/// code 0 / reason "retired" (`NodeTransport::close` on admit-replace) — observing
/// that close IS the behavioral proof a correct-home service took the pairing over.
fn close_is_supersede(err: &iroh::endpoint::ConnectionError) -> bool {
    matches!(
        err,
        iroh::endpoint::ConnectionError::ApplicationClosed(f)
            if f.error_code == 0u32.into() && f.reason.as_ref() == b"retired"
    )
}

/// Best-effort restart of the installed service (absolute manager paths). The
/// result is deliberately IGNORED: takeover is verified behaviorally (hub-side
/// supersede of our connection), so a failed or impossible restart — e.g. an
/// unreachable root LaunchDaemon, which converges via its own KeepAlive/poll —
/// simply means the takeover window expires and we stay in the foreground.
fn restart_service_best_effort() {
    if cfg!(target_os = "macos") {
        let _ = std::process::Command::new(service_manager_bin())
            .args([
                "kickstart",
                "-k",
                &format!(
                    "gui/{}/{}",
                    unsafe { libc::getuid() },
                    crate::node::service::SERVICE_NAME
                ),
            ])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status();
    } else {
        let _ = std::process::Command::new(service_manager_bin())
            .args(["--user", "restart", crate::node::service::SYSTEMD_UNIT])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status();
    }
}

fn persist_hub(cfg_path: &Path, hello: &HelloResult, ticket: &str) {
    let mut cfg = match InstanceConfig::load(cfg_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("higgs node: could not load config to save hub: {e}");
            return;
        }
    };
    cfg.remember_hub(SavedHub {
        hub_id: hello.node_id.clone(),
        ticket: ticket.to_string(),
        // Sanitized at PERSIST (T14 r24): the label is hub-controlled and is
        // replayed into the terminal on every later `--list`, so a dirty value
        // must never enter config.json in the first place.
        label: crate::remote::sanitize_display(&hello.hub_name),
        last_used_ms: now_ms(),
    });
    if let Err(e) = cfg.save(cfg_path) {
        eprintln!("higgs node: failed to save hub to config: {e}");
    }
}

/// Print the node's saved hubs (`higgs --node --list`); `★` marks the default a bare
/// `higgs --node` dials. No network.
fn list_saved_hubs(cfg: &InstanceConfig, node_id: &str) -> Result<()> {
    if cfg.hubs.is_empty() {
        println!("higgs node {node_id}: no saved hubs — pair with: higgs --node <ticket> <token>");
        return Ok(());
    }
    println!("higgs node {node_id} — saved hubs:");
    for h in &cfg.hubs {
        let default = if cfg.default_hub.as_deref() == Some(h.hub_id.as_str()) {
            " ★default"
        } else {
            ""
        };
        let short: String = h.hub_id.chars().take(8).collect();
        // Sanitize at print too (T14 r24): labels persisted before the
        // sanitize-at-persist fix may already be dirty in config.json.
        println!(
            "  {} ({})  last-used {}ms{}",
            crate::remote::sanitize_display(&h.label),
            short,
            h.last_used_ms,
            default
        );
    }
    Ok(())
}

/// The `<prefix>/bin` dir this binary is installed under, derived from the running
/// executable's OWN path — BUT only when that path actually has the install shape
/// `<prefix>/bin/v<semver>/higgs`: `current_exe`'s parent must be a `v<semver>` dir
/// AND its grandparent (the candidate `bin`) must hold a `current` symlink. Returns
/// `None` for anything else — a dev build (`target/debug/higgs`), a binary copied to
/// `~/bin/higgs`, or a standalone `/tmp/higgs` — so the boot-guard and the
/// self-update CLI never treat an arbitrary grandparent dir as a higgs install (a
/// blind grandparent would let `--prune` recurse into `$HOME`/`/var`). Callers that
/// get `None` skip the guard / require an explicit `--prefix`. `pub(crate)` so the node's
/// `M_NODE_UPDATE` control handler can target the SAME install dir the boot-guard uses.
pub(crate) fn self_update_bin_dir() -> Option<std::path::PathBuf> {
    // CANONICALIZE first: a service launched via `<prefix>/bin/current/higgs` (macOS
    // launchd — `current_exe` may keep the `current` symlink) and a symlinked `bin`
    // (`bin -> /srv/releases`) both resolve to the real `<realbin>/v<semver>/higgs`.
    // Without this the parent would be `current` (not a version dir) and the guard would
    // silently never run — so a bad update on a symlinked layout would never roll back.
    let exe = std::fs::canonicalize(std::env::current_exe().ok()?).ok()?;
    // The executable itself must be named `higgs` (not some other bundled binary).
    if exe.file_name()? != std::ffi::OsStr::new("higgs") {
        return None;
    }
    let vdir = exe.parent()?;
    // Its immediate parent must be a `v<semver>` version dir…
    if !crate::node::self_update::is_version_dir_name(vdir.file_name()?.to_str()?) {
        return None;
    }
    let bin = vdir.parent()?;
    // …and the grandparent (the managed bin dir) must carry the `current` symlink. We do
    // NOT require it be *named* `bin`: a supported `bin -> /srv/releases` layout resolves
    // the grandparent to the real dir, whose name is not `bin`. The `v<semver>` parent +
    // `current` symlink + is_managed/is_version_dir_name guards on `--prune` keep the
    // (contrived) case of a DIFFERENT app that mimics this exact layout AND names its
    // binary `higgs` from being pruned beyond its own `v<semver>` dirs.
    if !std::fs::symlink_metadata(bin.join("current"))
        .map(|m| m.file_type().is_symlink())
        .unwrap_or(false)
    {
        return None;
    }
    Some(bin.to_path_buf())
}

/// Self-update boot-guard ROLLBACK CHECK, run from `main()` BEFORE logging/tracing for
/// the `--node` path (the earliest stable point in the trialed binary). If a prior
/// self-update's trial has already spent its failure budget, roll `current` back to the
/// previous version here and return `true` (the caller EXITS so the service manager
/// re-execs the now-`current` OLD binary). A no-op (`false`) with no spent trial.
///
/// The RECORD/CONFIRM this check pairs with only run for a DAEMON-SERVING `higgs --node
/// <args>` invocation — the only kind whose outcome reflects the updated binary's health.
/// A ticket (first arg not a flag), `--hub <sel>`, or the bare form (default saved hub)
/// all attempt to serve; `--list` (inventory) and any other unknown `--flag` (an argument
/// error) do NOT. Both the boot-attempt RECORD and the clean-exit CONFIRM gate on this, so
/// an operator's `--node --list` / `--node --bogus` one-shots neither spend the rollback
/// budget nor falsely commit a trial the daemon never actually exercised.
///
/// It only ROLLS BACK; the boot-attempt is RECORDED later, at the TOP of
/// `run_node_daemon_body`'s async block — before the risky bind/runtime/serve init, but
/// AFTER the non-serving one-shots (`--list`, an argument error, a no-saved-hub print+exit)
/// have already returned — so those never spend the budget nor confirm an untested update.
/// The trial is COMMITTED only by a real hub admission or by `ALIVE_GRACE` of serve uptime,
/// never by a benign non-serving exit.
pub fn node_boot_preflight() -> bool {
    use crate::node::self_update as su;
    let Some(bin) = self_update_bin_dir() else {
        return false;
    };
    match su::boot_rollback_if_spent(&bin) {
        Ok(Some(prev)) => {
            eprintln!(
                "higgs node: the updated binary failed to start repeatedly — rolled back to \
                 {prev}. Restart to run it."
            );
            true
        }
        Ok(None) => false,
        Err(e) => {
            eprintln!("higgs node: self-update boot-guard skipped: {e}");
            false
        }
    }
}

/// `higgs --node [<ticket> [token]] | --list | --hub <label|id>` — the persistent node daemon.
///
/// Resolves WHICH hub to dial, then loops: dial, complete HELLO, serve the hub's
/// `higgs/node/*` control RPCs, and reconnect with backoff if the link drops (the EndpointId is
/// stable, so re-pairing is never needed). On the FIRST admission it saves the hub to
/// `config.json`, so afterwards a bare `higgs --node` reconnects on its own — no token, no
/// ticket. Modes:
/// - `<ticket> [token]` — pair/connect to a hub explicitly (token only on first enrollment).
/// - bare — connect to the default saved hub (none saved → print how to pair, exit 0).
/// - `--list` — print saved hubs and exit.
/// - `--hub <label|id>` — connect to a specific saved hub and make it the default.
pub fn run_node_daemon(args: &[String]) -> Result<()> {
    // The spent-trial ROLLBACK check already ran in `node_boot_preflight` (from `main()`,
    // before logging). The boot-attempt is RECORDED at the TOP of the body's async block
    // (before the risky init) and the trial COMMITTED on admission/​ALIVE_GRACE — while a
    // `--list`/​arg-error/​no-saved-hub exit returns before that async block, so it neither
    // spends the budget nor confirms an untested update. `confirm_bin` is the install's bin
    // dir for those in-serve hooks.
    let confirm_bin = self_update_bin_dir();
    run_node_daemon_body(args, &confirm_bin)
}

/// The daemon body proper (config → hub resolution → serve loop). Split from
/// [`run_node_daemon`] so the wrapper resolves `confirm_bin` once and hands it in; the boot
/// attempt is recorded at the TOP of this body's async block (before the risky init) and the
/// self-update trial is confirmed on a healthy boot / CLEAN return. `confirm_bin` is the
/// install's bin dir, threaded through for the in-serve confirm hooks.
fn run_node_daemon_body(args: &[String], confirm_bin: &Option<std::path::PathBuf>) -> Result<()> {
    let id = load_or_create_secret(&key_path()?)?.public().to_string();
    let cfg_path = config_path()?;
    let cfg = InstanceConfig::load(&cfg_path)?;

    // `--list` short-circuits (no bind, no network).
    if args.first().map(String::as_str) == Some("--list") {
        return list_saved_hubs(&cfg, &id);
    }

    // Resolve the hub to dial + whether to present a one-time token. `pairing` marks the
    // EXPLICIT-ticket invocation — the human-run enrollment command — which gets the
    // preflight + one-shot handoff treatment; service/`--hub` reconnects do not.
    let mut pairing = false;
    let (ticket_str, token): (String, Option<String>) = match args.first().map(String::as_str) {
        Some("--hub") => {
            let sel = args
                .get(1)
                .ok_or_else(|| Error::other("usage: higgs --node --hub <label|id>"))?;
            let hub = cfg.find_hub(sel).ok_or_else(|| {
                Error::other(format!(
                    "no saved hub matching {sel:?} — see `higgs --node --list`"
                ))
            })?;
            (hub.ticket.clone(), None)
        }
        Some(flag) if flag.starts_with("--") => {
            return Err(Error::other(format!(
                "unknown flag {flag:?} — usage: higgs --node [<ticket> [token]] | --list | --hub <label|id>"
            )));
        }
        // An explicit ticket (first-time pairing, or an explicit re-dial); token optional.
        Some(ticket) => {
            pairing = true;
            (ticket.to_string(), args.get(1).cloned())
        }
        // Bare: connect to the default saved hub. With none saved, WAIT for one instead of
        // exiting: the service manager (launchd KeepAlive / systemd Restart) respawned an
        // exiting daemon every few seconds forever, spamming the log with the pair hint —
        // and the wait is also the seamless pairing handoff: the moment `higgs --node
        // <ticket> <token>` persists the hub into config.json, THIS already-running service
        // picks it up on the next poll and connects, no kickstart, no restart.
        None => match cfg.default_saved_hub() {
            Some(hub) => (hub.ticket.clone(), None),
            None => {
                let name = name_or_init(Role::Node, &id, &crate::system::hostname())?;
                println!("higgs node   : {name} ({id})");
                println!(
                    "no saved hub yet — waiting to be paired (run: higgs --node <ticket> \
                     <token>)"
                );
                // An operator stop during the wait is a GRACEFUL shutdown, not a crash —
                // commit the trial before exiting so the recorded boot attempt above
                // never falsely spends rollback budget (same rule as the serve loop's
                // SIGTERM handling). The wait is sync, so a plain signal flag suffices.
                unsafe {
                    let handler = wait_stop as extern "C" fn(libc::c_int) as libc::sighandler_t;
                    libc::signal(libc::SIGTERM, handler);
                    libc::signal(libc::SIGINT, handler);
                }
                // The wait is part of THIS boot: a trialed binary that crashes while
                // waiting must spend rollback budget like any other early-init crash,
                // or a broken update crash-loops here forever with no auto-rollback.
                // (A quick pairing can record this boot twice — once here, once at the
                // serve-path record — which only spends budget FASTER, the safe bias;
                // a healthy wait clears everything at ALIVE_GRACE below.)
                if let Some(bin) = confirm_bin {
                    if let Err(e) = crate::node::self_update::record_boot_attempt(
                        bin,
                        env!("CARGO_PKG_VERSION"),
                    ) {
                        // Mirror the serve path: untrackable health must not run a trial.
                        eprintln!("higgs node: self-update boot-counter write failed ({e})");
                        if let Ok(Some(prev)) = crate::node::self_update::force_rollback_trial(
                            bin,
                            env!("CARGO_PKG_VERSION"),
                        ) {
                            eprintln!(
                                "higgs node: could not track the update's health (disk \
                                 full?) — rolled back to {prev}. Restart to run it."
                            );
                            return Ok(());
                        }
                    }
                }
                let mut polls: u64 = 0;
                let mut trial_committed = false;
                loop {
                    std::thread::sleep(HUB_WAIT_POLL);
                    polls += 1;
                    if WAIT_STOP.load(std::sync::atomic::Ordering::SeqCst) {
                        if let Some(bin) = confirm_bin {
                            let _ = crate::node::self_update::confirm_alive(
                                bin,
                                env!("CARGO_PKG_VERSION"),
                            );
                        }
                        println!("higgs node: stopped while waiting to be paired");
                        return Ok(());
                    }
                    // Same health rule as the serve loop: a trialed binary that simply
                    // STAYS UP for ALIVE_GRACE is healthy — an unpaired wait must not
                    // leave the self-update trial pending (it would block later updates).
                    if !trial_committed
                        && polls.saturating_mul(HUB_WAIT_POLL.as_secs())
                            >= crate::node::self_update::ALIVE_GRACE.as_secs()
                    {
                        // Markers are rooted at the INSTALL bin dir, same as every other
                        // confirm site — current_exe would miss them. Only mark committed
                        // on success so a transient failure retries next poll.
                        match confirm_bin {
                            Some(bin) => {
                                // confirm_alive silently no-ops on lock contention, so
                                // "committed" is judged by the marker actually clearing.
                                let _ = crate::node::self_update::confirm_alive(
                                    bin,
                                    env!("CARGO_PKG_VERSION"),
                                );
                                trial_committed = !crate::node::self_update::is_trial_pending_for(
                                    bin,
                                    env!("CARGO_PKG_VERSION"),
                                );
                            }
                            None => trial_committed = true,
                        }
                    }
                    // A corrupt/unreadable config is an ENVIRONMENT failure, not a bad
                    // binary — commit any pending trial before exiting so the recorded
                    // boot attempt never spends rollback budget on it.
                    let fresh = match InstanceConfig::load(&cfg_path) {
                        Ok(c) => c,
                        Err(e) => {
                            if let Some(bin) = confirm_bin {
                                let _ = crate::node::self_update::confirm_alive(
                                    bin,
                                    env!("CARGO_PKG_VERSION"),
                                );
                            }
                            return Err(e);
                        }
                    };
                    if let Some(hub) = fresh.default_saved_hub() {
                        println!("hub pairing detected — connecting");
                        // The flag handler STAYS installed — the serve path's tokio
                        // shutdown_listener replaces the disposition, and the async
                        // block re-checks the flag after the boot record, so a stop
                        // in the handover window is never lost.
                        break (hub.ticket.clone(), None);
                    }
                    // A quiet reminder every ~5 minutes, not two lines per second.
                    if polls.is_multiple_of(HUB_WAIT_REMIND_EVERY) {
                        println!("still waiting to be paired — run: higgs --node <ticket> <token>");
                    }
                }
            }
        },
    };
    // A malformed SAVED ticket is config corruption — an ENVIRONMENT failure; commit
    // any pending trial so the wait path's recorded boot attempt never spends budget.
    // A malformed EXPLICIT ticket (pairing) is an operator typo: no boot attempt was
    // recorded and a typo is no evidence of binary health — leave the trial alone.
    let ticket: EndpointTicket = match ticket_str.parse() {
        Ok(t) => t,
        Err(e) => {
            if let (false, Some(bin)) = (pairing, confirm_bin.as_ref()) {
                let _ = crate::node::self_update::confirm_alive(bin, env!("CARGO_PKG_VERSION"));
            }
            return Err(Error::other(e));
        }
    };
    let target = ticket.endpoint_addr().clone();

    let rt = runtime()?;
    rt.block_on(async {
        // BOOT-GUARD (P3): install the SIGTERM/SIGINT handler AND record this boot attempt toward the
        // self-update rollback budget FIRST — before the risky daemon init (bind, NodeRuntime, serve)
        // below. `shutdown_LISTENER` registers the OS handlers SYNCHRONOUSLY here, so a SIGTERM during
        // that init is BUFFERED (observed later at the serve `select!`), never default-terminated. The
        // counter is set before init and cleared on a healthy boot (admission / ALIVE_GRACE) or a
        // GRACEFUL shutdown (`confirm_alive`), so:
        //   * a CRASH in early init — an update that passes the `--version` smoke test but dies during
        //     bind / runtime / serve setup — leaves the counter set, so `boot_rollback_if_spent` rolls
        //     back after `BOOT_FAIL_BUDGET`. (Recording at serve-commit, as before, meant such an early
        //     crash NEVER accrued, so a bad update crash-looped forever with NO auto-rollback.)
        //   * an operator SIGTERM during init is caught -> the serve loop breaks -> `confirm_alive`
        //     commits the trial -> NO falsely-spent budget.
        // This ONE operator-signal future lives from HERE through the update drain below (the serve
        // loop + drain borrow it), so a SIGTERM is never caught by tokio yet observed by nobody.
        // (A crash strictly BEFORE this — in the small pre-block config/ticket parse — is a narrower
        // residual: no tokio context exists there to install the handler, and the `--list`/no-hub
        // short-circuits must run first; documented rather than covered.)
        let operator = crate::shutdown::shutdown_listener();
        tokio::pin!(operator);
        if let Some(bin) = confirm_bin {
            if let Err(e) =
                crate::node::self_update::record_boot_attempt(bin, env!("CARGO_PKG_VERSION"))
            {
                eprintln!("higgs node: self-update boot-counter write failed ({e})");
                if let Ok(Some(prev)) =
                    crate::node::self_update::force_rollback_trial(bin, env!("CARGO_PKG_VERSION"))
                {
                    eprintln!(
                        "higgs node: could not track the update's health (disk full?) — rolled \
                         back to {prev}. Restart to run it."
                    );
                    return Ok(false); // not an update restart — exit; the next start runs `prev`
                }
            }
        }
        // A stop that arrived while the wait loop's plain handler was still the
        // disposition (before shutdown_listener above replaced it) only set the flag —
        // honor it now as the same graceful exit the wait loop promises.
        if WAIT_STOP.load(std::sync::atomic::Ordering::SeqCst) {
            if let Some(bin) = confirm_bin {
                let _ = crate::node::self_update::confirm_alive(bin, env!("CARGO_PKG_VERSION"));
            }
            println!("higgs node: stopped while waiting to be paired");
            return Ok(false);
        }
        let sk = load_or_create_secret(&key_path()?)?;
        let endpoint = bind_endpoint(sk).await.map_err(Error::other)?;
        let self_id = endpoint.id().to_string();
        // This node's persistent friendly name (`node-<eid8>(<host>)`), sent in every HELLO so
        // the hub labels it in the fleet view. Generated + persisted on first run, reused after.
        let name = name_or_init(Role::Node, &self_id, &crate::system::hostname())?;
        // Model roots: the same defaults the standalone runtime uses (standard LM Studio /
        // HF / Ollama dirs) plus the HIGGS_MODEL_DIR override, so a node can actually
        // scan/load real models — not an empty set.
        let mut hc = crate::HiggsConfig::default();
        if let Ok(dir) = std::env::var("HIGGS_MODEL_DIR") {
            if !dir.is_empty() {
                hc.lmstudio_dirs.push(std::path::PathBuf::from(dir));
            }
        }
        // Pulled models (M_PULL) land in ~/.higgs/models/<org>/<model>/*.gguf — an LM-Studio
        // layout that `HiggsConfig::default()` already includes as a scan root, so a
        // just-pulled model is loadable with no extra push here.
        // A node has no UI; its worker stderr is relayed to the hub. HIGGS_VERBOSE=1 keeps
        // the full llama.cpp dump (default off drops the per-load metadata flood).
        // The PROCESS-GLOBAL bus (the one the daemon's own tracing lands in) when
        // running as the real binary — M_NODE_LOGS serves its Serve ring to the
        // hub. Tests/embedders without a global get a private bus as before.
        let bus = crate::log_bus::LogBus::global()
            .unwrap_or_else(|| Arc::new(crate::log_bus::LogBus::new()));
        if std::env::var("HIGGS_VERBOSE").is_ok_and(|v| v == "1" || v == "true") {
            bus.set_verbose(true);
        }
        let node = Arc::new(NodeRuntime::new(NodeConfig {
            bus,
            lmstudio_dirs: hc.lmstudio_dirs,
            hf_dirs: hc.hf_dirs,
            ollama_dirs: hc.ollama_dirs,
            idle_ttl: crate::node::runtime::DEFAULT_IDLE_TTL,
        }));
        if !pairing {
            println!("higgs node   : {name} ({self_id}); connecting to hub…");
        }

        // Self-update "N seconds alive" commit: if a trial for this version is pending,
        // a binary that simply STAYS UP for ALIVE_GRACE (even without reaching the hub —
        // a hub outage is not a bad binary) commits the update, so the boot-guard only
        // rolls back a binary that dies WITHIN the grace window. Complements the faster
        // admission-based confirm below. RETRIES on lock contention (a long apply/prune
        // holding the update lock at the grace moment would otherwise silently no-op the
        // one-shot commit and leave a healthy version on trial forever). Version-gated +
        // lock-guarded inside confirm_alive.
        if let Some(bin) = confirm_bin.clone() {
            tokio::spawn(async move {
                use crate::node::self_update as su;
                tokio::time::sleep(su::ALIVE_GRACE).await;
                for _ in 0..su::CONFIRM_MAX_ATTEMPTS {
                    let _ = su::confirm_alive(&bin, env!("CARGO_PKG_VERSION"));
                    if !su::is_trial_pending_for(&bin, env!("CARGO_PKG_VERSION")) {
                        break;
                    }
                    tokio::time::sleep(su::CONFIRM_RETRY_INTERVAL).await;
                }
            });
        }

        // The one-time token is sent until a connect SUCCEEDS (HELLO admitted); a failed
        // attempt (hub offline, relay flake, HELLO timeout) must keep it for retry, or an
        // unallowlisted node could never pair. Reconnects after success rely on allowlist
        // membership, so the token is cleared only then.
        let mut token = token;
        // Persist the hub into config.json once, after the FIRST admission, so a later bare
        // `higgs --node` reconnects to it without a ticket or token.
        let mut saved = false;

        // A connection the PAIRING block below already established and hands to the serve
        // loop's first iteration (foreground fall-through) — never redialed away.
        let mut preconnected: Option<iroh::endpoint::Connection> = None;
        // True only on the pairing-foreground path: a LATE service takeover (hub
        // supersede after the wait window expired) must end this process instead of
        // redialing into a duplicate-identity fight with the service.
        let mut watch_supersede = false;

        // ── PAIRING MODE: gated preflight → one-shot connect → seamless service handoff ──
        // The human-run enrollment (`higgs --node <ticket> [token]`) self-diagnoses instead
        // of looping into an opaque timeout (docs/pairing-preflight-checklist.md). On
        // success with a node service installed, it hands the connection to the service and
        // EXITS — never a second foreground node flapping against the service with the same
        // identity. Without a service it stays up as a clearly-labeled foreground node.
        if pairing {
            let style = crate::node::preflight::Style::auto();
            println!("higgs node   : {name} ({self_id})");
            let report = crate::node::preflight::run(&target, &style).await;
            if report.hopeless() {
                eprintln!(
                    "{}",
                    style.fail(
                        "no usable path to the hub — fix the failed checks above and re-run \
                         this command"
                    )
                );
                // An ENVIRONMENT failure is not a bad binary: commit any pending
                // self-update trial so repeated preflight failures cannot spend the
                // boot-rollback budget and roll back a healthy version.
                if let Some(bin) = confirm_bin {
                    let _ =
                        crate::node::self_update::confirm_alive(bin, env!("CARGO_PKG_VERSION"));
                }
                return Err(Error::other("pairing preflight failed"));
            }
            // A Ctrl-C/SIGTERM during pairing must cancel NOW — not after the 2×30s
            // connect budget — and must never persist or hand off afterwards. An
            // operator cancel is a graceful exit: commit any pending trial first.
            // (A signal during the preflight is buffered by the tokio signal stream
            // and caught at the first connect-loop select below.)
            let cancelled = |style: &crate::node::preflight::Style| {
                if let Some(bin) = confirm_bin {
                    let _ =
                        crate::node::self_update::confirm_alive(bin, env!("CARGO_PKG_VERSION"));
                }
                eprintln!("{}", style.warn("pairing cancelled"));
                Error::other("pairing cancelled")
            };
            println!("{}", style.head("higgs pair: connecting to hub…"));
            let mut last_err: Option<Error> = None;
            let mut admitted = None;
            for _ in 0..PAIR_ATTEMPTS {
                let attempt = tokio::time::timeout(
                    PAIR_ATTEMPT_TIMEOUT,
                    crate::node::connect_node(
                        &endpoint,
                        target.clone(),
                        self_id.clone(),
                        name.clone(),
                        token.clone(),
                    ),
                );
                tokio::pin!(attempt);
                let outcome = tokio::select! {
                    // biased: a cancel racing a successful connect must win — never
                    // persist or hand off after the operator asked to stop.
                    biased;
                    _ = &mut operator => return Err(cancelled(&style)),
                    r = &mut attempt => r,
                };
                match outcome {
                    Ok(Ok(ok)) => {
                        admitted = Some(ok);
                        break;
                    }
                    Ok(Err(e)) => last_err = Some(e),
                    Err(_) => {
                        last_err = Some(Error::other(format!(
                            "timed out after {}s",
                            PAIR_ATTEMPT_TIMEOUT.as_secs()
                        )))
                    }
                }
            }
            match admitted {
                Some((conn, hello)) => {
                    persist_hub(&cfg_path, &hello, &ticket_str);
                    println!(
                        "  {}",
                        style.ok(&format!(
                            "paired with hub {} ({}) (protocol v{})",
                            crate::remote::sanitize_display(&hello.hub_name),
                            hello.node_id,
                            hello.agreed_version
                        ))
                    );
                    if let Some(bin) = confirm_bin {
                        let _ = crate::node::self_update::confirm_alive(
                            bin,
                            env!("CARGO_PKG_VERSION"),
                        );
                    }
                    // The handoff depends on the hub actually LANDING in config.json (the
                    // service connects from that file) — persist_hub is best-effort, so
                    // verify by re-reading. A failed save falls through to foreground
                    // serving: the pairing still works, and the warning names the problem.
                    // The DEFAULT row must be this hub with this exact ticket: the bare
                    // service dials default_saved_hub(), and remember_hub() always sets
                    // default_hub on a successful save — so a matching non-default row is
                    // a stale leftover from an earlier pairing, not proof this save landed.
                    let hub_saved = InstanceConfig::load(&cfg_path).ok().is_some_and(|c| {
                        c.default_saved_hub()
                            .is_some_and(|h| h.hub_id == hello.node_id && h.ticket == ticket_str)
                    });
                    if !hub_saved {
                        println!(
                            "  {}",
                            style.warn(
                                "could not save the hub to config.json — staying in the \
                                 foreground (fix the config path/permissions and re-pair \
                                 for an always-on service handoff)"
                            )
                        );
                    }
                    if hub_saved && service_present() {
                        // BEHAVIORAL handoff (replaces the old config-parsing design):
                        // best-effort restart the installed service, then wait for the
                        // HUB to close OUR admitted connection with the "retired"
                        // supersede (the hub replaces a duplicate-identity dial). Only a
                        // service that manages THIS node's state dir has this node's key
                        // — so the supersede IS the proof of a correct handoff. A
                        // wrong-home / dead / stale service never supersedes us and we
                        // simply keep serving in the foreground. We exit only AFTER the
                        // successor is demonstrably connected to the hub — strictly
                        // stronger than any service-manager liveness proxy, and no
                        // parsing of plists/unit files at all.
                        restart_service_best_effort();
                        println!(
                            "  {}",
                            style.head("waiting for the installed service to take over…")
                        );
                        let took_over = matches!(
                            tokio::time::timeout(HANDOFF_TAKEOVER_WINDOW, conn.closed()).await,
                            Ok(reason) if close_is_supersede(&reason)
                        );
                        if took_over {
                            drop(conn);
                            println!(
                                "  {}",
                                style.ok(
                                    "handed off — the node service is connected to your \
                                     hub now"
                                )
                            );
                            if let Some(bin) = confirm_bin {
                                let _ = crate::node::self_update::confirm_alive(
                                    bin,
                                    env!("CARGO_PKG_VERSION"),
                                );
                            }
                            return Ok(false);
                        }
                        println!(
                            "  {}",
                            style.warn(
                                "the installed service did not take over — continuing in \
                                 the foreground so the node stays online (check: \
                                 launchctl print gui/$UID/com.higgs.node, systemctl \
                                 --user status higgs-node, or for a system daemon: sudo \
                                 launchctl kickstart -k system/com.higgs.node)"
                            )
                        );
                    }
                    if hub_saved && !service_present() {
                        println!(
                            "  {}",
                            style.warn(
                                "no node service installed — running in the foreground \
                                 (for always-on, run: higgs node install-service)"
                            )
                        );
                    }
                    // Fall into the daemon loop below as the (sole) node process, SERVING
                    // the already-admitted connection — dropping and redialing here would
                    // sever the link the hub just gated (and burn nothing: the token is
                    // already consumed hub-side, so the loop's later reconnects go via the
                    // allowlist).
                    preconnected = Some(conn);
                    token = None;
                    saved = true;
                    watch_supersede = true;
                }
                None => {
                    let e = last_err.unwrap_or_else(|| Error::other("no attempt ran"));
                    eprintln!("  {}", style.fail(&format!("pairing failed: {e}")));
                    // Same budget rule as the preflight gate: a connect failure is an
                    // environment problem, never spent against the update boot budget.
                    if let Some(bin) = confirm_bin {
                        let _ = crate::node::self_update::confirm_alive(
                            bin,
                            env!("CARGO_PKG_VERSION"),
                        );
                    }
                    eprintln!(
                        "{}",
                        crate::node::preflight::connect_failure_advice(
                            &report,
                            cfg!(target_os = "macos"),
                            crate::node::preflight::is_ssh_session(),
                        )
                    );
                    return Err(Error::other("pairing failed"));
                }
            }
        }
        // SIGINT/SIGTERM ends the loop so we can drain resident workers — a dropped Supervisor does
        // not reap its child, so an undrained exit would orphan models. The `operator` signal future
        // and the boot-attempt record were BOTH set up at the TOP of this block (before the risky
        // init) — see the boot-guard comment there. `operator` lives from there through the update
        // drain below: the serve loop borrows it (via `await_node_shutdown`, which reports the
        // shutdown CAUSE — an operator stop vs a self-update restart request, decided by WHICH trigger
        // fires — biased so an operator stop wins a tie), and the drain re-borrows it so an operator
        // SIGTERM arriving DURING the (bounded) drain still aborts the re-exec (a monotonic
        // Update→Operator override). ONE observation alive means no gap where a SIGTERM is caught by
        // tokio's handler yet observed by nobody.

        // The serve loop yields the shutdown CAUSE it broke on (`break 'serve cause`) — no mutable
        // flag, and the cause is the one latched by the resolving `await_node_shutdown` arm. The
        // selector BORROWS `operator` (`as_mut`) so the future itself outlives this loop and is
        // free to re-observe in the drain below.
        let shutdown_cause = {
            let shutdown = crate::node::self_update::await_node_shutdown(operator.as_mut());
            tokio::pin!(shutdown);
            // Consecutive redial failures — drives the periodic what-to-check
            // block below so an operator tailing node.log REMOTELY sees the
            // likely fix, not just a wall of identical timeouts.
            let mut connect_failures: u64 = 0;
            'serve: loop {
                // First iteration after a foreground pairing: serve the ALREADY-admitted
                // connection instead of redialing (see the pairing block above).
                if let Some(conn) = preconnected.take() {
                    // Keep a handle to inspect the close reason after serving: if the
                    // installed service took over LATE (hub superseded us with
                    // "retired"), exit gracefully instead of redialing into a
                    // duplicate-identity fight — the service is the rightful node now.
                    let probe = conn.clone();
                    tokio::select! {
                        cause = &mut shutdown => break 'serve cause,
                        _ = crate::node::serve_node(conn, node.clone()) => {
                            if watch_supersede
                                && probe.close_reason().is_some_and(|e| close_is_supersede(&e))
                            {
                                println!(
                                    "node service took over the pairing — exiting \
                                     (the service is connected to your hub)"
                                );
                                break 'serve crate::node::self_update::ShutdownCause::Operator;
                            }
                            eprintln!("hub connection closed; reconnecting…");
                        }
                    }
                }
                tokio::select! {
                    cause = &mut shutdown => break 'serve cause,
                res = crate::node::connect_node(&endpoint, target.clone(), self_id.clone(), name.clone(), token.clone()) => {
                    match res {
                        Ok((conn, hello)) => {
                            token = None; // admitted — token burned hub-side; don't resend it
                            connect_failures = 0;
                            if !saved {
                                saved = true;
                                persist_hub(&cfg_path, &hello, &ticket_str);
                                // Step 3: the freshly-flipped binary proved it can start AND reach
                                // the hub — commit the self-update trial so the boot-guard won't
                                // roll it back on the next restart. Version-gated (only clears a
                                // trial FOR this running version, so an old daemon admitted while a
                                // newer version sits on trial does not clear it) + a no-op when no
                                // trial is pending. Best-effort: a marker-clear IO error must never
                                // fail an otherwise-healthy connection.
                                if let Some(bin) = confirm_bin {
                                    let _ = crate::node::self_update::confirm_alive(
                                        bin,
                                        env!("CARGO_PKG_VERSION"),
                                    );
                                }
                            }
                            println!(
                                "paired with hub {} ({}) (protocol v{})",
                                crate::remote::sanitize_display(&hello.hub_name),
                                hello.node_id,
                                hello.agreed_version
                            );
                            tokio::select! {
                                cause = &mut shutdown => break 'serve cause,
                                _ = crate::node::serve_node(conn, node.clone()) => {
                                    eprintln!("hub connection closed; reconnecting…");
                                }
                            }
                        }
                        Err(e) => {
                            connect_failures += 1;
                            // The FULL cause chain — the top-level line alone
                            // ("timed out") says nothing an operator can act on.
                            eprintln!(
                                "higgs node: connect failed (attempt {connect_failures}): {}",
                                error_chain(&e)
                            );
                            // Roughly once a minute of failures, say what to
                            // CHECK — the operator tailing this log is usually
                            // remote and cannot see this machine's screen.
                            if connect_failures == 3 || connect_failures.is_multiple_of(20) {
                                eprintln!(
                                    "higgs node: still unreachable after {connect_failures} attempts — check:"
                                );
                                #[cfg(target_os = "macos")]
                                eprintln!(
                                    "  • {}",
                                    crate::node::fleet::macos_lna_advice(
                                        "On THIS machine's own screen (not SSH)"
                                    )
                                );
                                eprintln!(
                                    "  • the hub must be up and reachable (its Fleet tab shows \
                                     'Hub active')"
                                );
                                eprintln!(
                                    "  • full checklist: docs/pairing-preflight-checklist.md"
                                );
                            }
                        }
                    }
                }
            }
            tokio::select! {
                cause = &mut shutdown => break 'serve cause,
                _ = tokio::time::sleep(Duration::from_secs(3)) => {}
            }
            }
        };
        // The loop is only ever left via a GRACEFUL shutdown (SIGTERM / logout) — a
        // served-then-stopped session, NOT a failure. Commit the self-update trial here so
        // a SHORT-lived service session (e.g. a login-bound agent that starts and stops
        // within ALIVE_GRACE, before ever admitting to an offline hub) does not accrue a
        // spurious boot-failure toward the rollback budget. Version + trial gated inside.
        // (Safe on an UPDATE restart too: confirm_alive is version-gated to THIS running
        // version, so it cannot commit the just-staged NEWER version's trial.)
        if let Some(bin) = confirm_bin {
            let _ = crate::node::self_update::confirm_alive(bin, env!("CARGO_PKG_VERSION"));
        }
        // P4b (c): the shutdown CAUSE was LATCHED when the loop broke (`shutdown_cause`, set by the
        // resolving `await_node_shutdown` arm) — not sampled here. This single value decides BOTH
        // the drain below AND the re-exec after the runtime exits, so the two cannot disagree, and
        // a detached update finishing DURING this teardown cannot reclassify an operator stop: the
        // loop already broke as `Operator`, and the update simply activates on the next start.
        let mut update_restart =
            shutdown_cause == crate::node::self_update::ShutdownCause::Update;
        // An UPDATE restart (a `request_self_restart` trigger) DRAINS before the teardown — the
        // broken serve loop already quiesces NEW work (its accept loop is gone; reconnect stopped),
        // and `drain_until_idle` quiesces new chat leases then waits (bounded) for the in-flight
        // generations AND their stream handlers to finish, so the update doesn't truncate them; on
        // deadline, proceed — `shutdown_all` then stops the stragglers exactly as a plain stop
        // would. An operator/service-manager stop keeps today's prompt stop — service managers
        // expect it, and their escalation timeouts stay honored.
        //
        // OPERATOR OVERRIDE (monotonic Update→Operator): the drain can take up to the deadline, so
        // keep watching the SAME operator-signal future. If an operator SIGTERM/SIGINT lands DURING
        // the drain, abandon the re-exec and stop promptly — an operator who asks to stop is
        // honored even mid-update, and the staged, verified update activates on the next start
        // instead. `biased` toward the operator so its arrival wins the poll. Without this, a
        // SIGTERM during the drain would be caught by tokio's still-installed handler yet observed
        // by no one, and the node would re-exec against the operator's wishes.
        if update_restart {
            println!("higgs node: staged update — draining in-flight generations…");
            let drain = node.drain_until_idle(Duration::from_millis(500), UPDATE_DRAIN_DEADLINE);
            match crate::node::self_update::drain_with_operator_override(operator.as_mut(), drain)
                .await
            {
                // The drain ran to completion (or its deadline): still an update restart, re-exec.
                Some(drained) => {
                    if !drained {
                        eprintln!(
                            "higgs node: drain deadline ({}s) reached — stopping the remaining \
                             generations",
                            UPDATE_DRAIN_DEADLINE.as_secs()
                        );
                    }
                }
                // An operator stop landed mid-drain: abandon the re-exec, stop promptly.
                None => {
                    eprintln!(
                        "higgs node: operator stop during the update drain — stopping now; the \
                         staged update activates on the next start"
                    );
                    update_restart = false;
                }
            }
        }
        println!("higgs node: draining resident workers…");
        node.shutdown_all().await;
        // FINAL operator gate before committing to the re-exec (P4b (c)): `shutdown_all` reaps
        // workers for BOTH an operator stop and an update restart, so it always runs to completion
        // — but it can take a few seconds, and a SIGTERM delivered while it ran is buffered by
        // tokio's handler with nobody watching. Consume it HERE, at the decision point: if the
        // operator asked to stop by now, skip the re-exec (the staged update activates on the next
        // start). Only checked while `operator` is still pending — it is, unless the drain override
        // already fired and cleared `update_restart`.
        if update_restart
            && crate::node::self_update::operator_stop_pending(operator.as_mut()).await
        {
            eprintln!(
                "higgs node: operator stop during teardown — stopping now; the staged update \
                 activates on the next start"
            );
            update_restart = false;
        }
        Ok(update_restart)
    })
    // P4b (c): activate the staged update IMMEDIATELY by re-executing through `current` —
    // re-entering `main()` runs `node_boot_preflight` (the boot-guard) exactly like a service-
    // manager restart, but also works for a node run WITHOUT one (previously such a node just
    // exited). Gated on the SAME latched cause that decided the drain — cleared to `false` if an
    // operator stop overrode the drain, so an operator's mid-update stop never re-execs. The
    // ORIGINAL argv tail is re-passed so an explicit `--hub <sel>` selection
    // survives (a burned pairing token in the argv is harmless — the node is allowlisted by
    // now, and the gate checks the allowlist first). The process image is replaced on success;
    // on failure fall through to a normal exit — the service manager (or the next manual
    // start) runs `current` anyway.
    .map(|update_restart| {
        if update_restart {
            if let Some(bin) = self_update_bin_dir() {
                use std::os::unix::process::CommandExt;
                let target = bin.join("current/higgs");
                eprintln!(
                    "higgs node: re-executing {} to activate the staged update",
                    target.display()
                );
                let err = std::process::Command::new(&target)
                    .arg("--node")
                    .args(args)
                    .exec();
                eprintln!(
                    "higgs node: re-exec failed ({err}) — exiting; the next start runs the \
                     staged update"
                );
            }
        }
    })
}

/// How long an UPDATE restart waits for in-flight generations to finish before stopping the
/// stragglers (P4b (c)). Long enough for a typical chat generation to complete; bounded so a
/// runaway/stuck generation cannot stall the update indefinitely — on expiry the node truncates
/// exactly as the pre-drain behavior always did. Only the self-update restart waits; an
/// operator/service-manager stop is never delayed.
const UPDATE_DRAIN_DEADLINE: Duration = Duration::from_secs(60);

#[cfg(test)]
#[path = "cli_tests.rs"]
mod tests;
