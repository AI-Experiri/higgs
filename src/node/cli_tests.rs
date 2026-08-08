use super::*;

#[test]
fn link_rejects_unknown_subcommand() {
    assert!(run_link(&["bogus".into()]).is_err());
    assert!(run_link(&[]).is_err());
}

#[test]
fn shell_single_quote_survives_a_shell_as_one_argument() {
    assert_eq!(shell_single_quote("/opt/higgs"), "'/opt/higgs'");
    assert_eq!(shell_single_quote("/opt/Higgs Fleet"), "'/opt/Higgs Fleet'");
    // An embedded single quote uses the '\'' idiom.
    assert_eq!(shell_single_quote("a'b"), r"'a'\''b'");
}

#[test]
fn parse_systemd_show_version_reads_the_version_property() {
    // `systemctl --user show --property=Version` output shapes across releases.
    assert_eq!(parse_systemd_show_version("Version=249\n"), Some(249));
    assert_eq!(parse_systemd_show_version("Version=237\n"), Some(237)); // Ubuntu 18.04
                                                                        // 240 is the floor (Type=exec + append:); one below must read as 239.
    assert_eq!(parse_systemd_show_version("Version=239\n"), Some(239));
    // Newer systemd may append a build suffix, or prefix a `v`.
    assert_eq!(
        parse_systemd_show_version("Version=254 (254.1-2)\n"),
        Some(254)
    );
    assert_eq!(parse_systemd_show_version("Version=v257\n"), Some(257));
    // The property can arrive amid other lines (without --value).
    assert_eq!(
        parse_systemd_show_version("Names=foo\nVersion=250\nOther=1\n"),
        Some(250)
    );
    // Unparseable shapes → None (caller proceeds best-effort, never panics).
    assert_eq!(parse_systemd_show_version("garbage"), None);
    assert_eq!(parse_systemd_show_version(""), None);
    assert_eq!(parse_systemd_show_version("Version=\n"), None);
}

#[test]
fn operator_can_exec_actually_runs_the_binary_as_the_operator() {
    use std::os::unix::fs::PermissionsExt;
    // The test runs as a non-root user, so operator_can_exec takes the
    // "already the operator" branch and EXECUTES the binary (`--version`).
    let euid = unsafe { libc::geteuid() };
    assert_ne!(
        euid, 0,
        "coverage of the operator branch needs a non-root run"
    );
    let op = passwd_by_uid(euid).expect("current uid has a passwd entry");
    let dir = tempfile::tempdir().unwrap();

    // A regular file that RUNS (exits 0) → true. The stub ignores its args.
    let bin = dir.path().join("higgs");
    std::fs::write(&bin, b"#!/bin/sh\nexit 0\n").unwrap();
    std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o755)).unwrap();
    assert!(operator_can_exec(&bin, &op, euid).unwrap());

    // A binary that runs but FAILS (`--version` exits non-zero) → false. This is
    // the case a mere access(X_OK) check would have WRONGLY accepted.
    std::fs::write(&bin, b"#!/bin/sh\nexit 3\n").unwrap();
    std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o755)).unwrap();
    assert!(!operator_can_exec(&bin, &op, euid).unwrap());

    // Owner has NO exec bit (0644) → execve EACCES for the owner → false (a
    // spawn failure is `false`, not a hard error).
    std::fs::write(&bin, b"#!/bin/sh\nexit 0\n").unwrap();
    std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o644)).unwrap();
    assert!(!operator_can_exec(&bin, &op, euid).unwrap());

    // mode-0001: exec bit set only for `other` — the owner still can't execve.
    std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o001)).unwrap();
    assert!(!operator_can_exec(&bin, &op, euid).unwrap());

    // A directory, a missing path, and a DANGLING symlink → false.
    std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o755)).unwrap();
    assert!(!operator_can_exec(dir.path(), &op, euid).unwrap());
    assert!(!operator_can_exec(&dir.path().join("nope"), &op, euid).unwrap());
    let dangling = dir.path().join("current");
    std::os::unix::fs::symlink(dir.path().join("gone"), &dangling).unwrap();
    assert!(!operator_can_exec(&dangling, &op, euid).unwrap());
}

#[test]
fn temp_name_is_unpredictable_not_pid_derived() {
    // The temp name must NOT be the old predictable `<base>.<pid>` (a group peer
    // could pre-plant a symlink at that path); it carries a random hex suffix.
    let base = ".higgs-writeprobe";
    let n = temp_name(base);
    assert!(n.starts_with(&format!("{base}.")), "keeps the base: {n}");
    let pid_form = format!("{base}.{}", std::process::id());
    assert_ne!(n, pid_form, "must not be the predictable PID form");
    // The suffix is 16 hex chars (from a u64 hash), not the PID.
    let suffix = n.strip_prefix(&format!("{base}.")).unwrap();
    assert_eq!(suffix.len(), 16, "16-hex suffix: {n}");
    assert!(suffix.chars().all(|c| c.is_ascii_hexdigit()), "{n}");
}

#[test]
fn probe_dir_writable_catches_a_non_writable_dir() {
    use std::os::unix::fs::PermissionsExt;
    // Must run as a non-root user (root bypasses the write bit).
    assert_ne!(unsafe { libc::geteuid() }, 0, "needs a non-root run");
    let dir = tempfile::tempdir().unwrap();
    // A normal writable dir → Ok, and the probe file is cleaned up (create+unlink).
    assert!(probe_dir_writable(dir.path()).is_ok());
    assert!(
        std::fs::read_dir(dir.path()).unwrap().next().is_none(),
        "probe must leave no temp file behind"
    );
    // A read-only dir (0555) → Err: the daemon couldn't recreate node.log there.
    std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(0o555)).unwrap();
    assert!(probe_dir_writable(dir.path()).is_err());
    // Restore so tempdir cleanup can remove it.
    std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(0o755)).unwrap();
}

#[test]
fn operator_can_exec_scrubs_the_ambient_env_from_the_untrusted_binary() {
    use std::os::unix::fs::PermissionsExt;
    // The preflight is the ONE spawn of the untrusted binary, and the caller's
    // shell env carries transient credentials the eventual service never sees.
    // The probe must run env-SCRUBBED (allowlist: PATH + HOME). Canary: `CARGO`
    // is always set in a cargo-test process env — a stub that FAILS when it can
    // see CARGO only passes if the env was cleared.
    assert!(
        std::env::var_os("CARGO").is_some(),
        "test precondition: cargo sets CARGO in the test env"
    );
    let euid = unsafe { libc::geteuid() };
    let op = passwd_by_uid(euid).expect("current uid has a passwd entry");
    let dir = tempfile::tempdir().unwrap();
    let bin = dir.path().join("higgs");
    // Exit 7 (→ not-runnable) if ANY ambient var leaked; also prove the
    // allowlist survives: HOME must be present (the service sets it too).
    std::fs::write(
        &bin,
        b"#!/bin/sh\n[ -z \"$CARGO\" ] || exit 7\n[ -n \"$HOME\" ] || exit 8\nexit 0\n",
    )
    .unwrap();
    std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o755)).unwrap();
    assert!(
        operator_can_exec(&bin, &op, euid).unwrap(),
        "the probe must not leak the ambient env (and must keep HOME)"
    );
}

#[test]
fn ensure_appendable_log_neutralizes_a_hostile_umask_on_create() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().unwrap();
    let log = dir.path().join("node.log");
    // Constrain the process umask around the create. 0o077 keeps OWNER bits, so
    // a concurrently-running test that creates a file during this window is
    // unaffected — yet it distinguishes the fix: without the explicit fchmod
    // the created log would be 0600 here (0666 & !077), not 0644.
    let old = unsafe { libc::umask(0o077) };
    let created = ensure_appendable_log(&log);
    unsafe { libc::umask(old) };
    created.unwrap();
    let mode = std::fs::metadata(&log).unwrap().permissions().mode() & 0o777;
    assert_eq!(
        mode, 0o644,
        "a probe-created log must be 0644 REGARDLESS of umask (the daemon must \
         be able to reopen it later)"
    );

    // A PRE-EXISTING file's mode is surfaced, never mutated: 0600 stays 0600.
    std::fs::set_permissions(&log, std::fs::Permissions::from_mode(0o600)).unwrap();
    ensure_appendable_log(&log).unwrap();
    let mode = std::fs::metadata(&log).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode, 0o600, "an existing log's mode must not be changed");

    // A non-appendable existing file fails LOUDLY (the preflight's purpose).
    std::fs::set_permissions(&log, std::fs::Permissions::from_mode(0o400)).unwrap();
    assert!(ensure_appendable_log(&log).is_err());
    std::fs::set_permissions(&log, std::fs::Permissions::from_mode(0o644)).unwrap();
}

#[test]
fn ensure_appendable_log_refuses_a_preplanted_node_log_symlink() {
    // A peer who planted `node.log -> <operator file>` while logs/ was momentarily
    // writable would, without O_NOFOLLOW, have the probe FOLLOW the link (open
    // succeeds) and the daemon later append its logs into that file. The open must
    // be O_NOFOLLOW so a symlink at node.log is REFUSED (ELOOP), not followed.
    let dir = tempfile::tempdir().unwrap();
    let secret = dir.path().join("config.json");
    std::fs::write(&secret, b"important\n").unwrap();
    let log = dir.path().join("node.log");
    std::os::unix::fs::symlink(&secret, &log).unwrap();
    assert!(
        ensure_appendable_log(&log).is_err(),
        "a symlink at node.log must be refused (O_NOFOLLOW), not followed"
    );
    // The symlink itself is left in place (we refuse, we do not silently mutate),
    // and the operator's target file was never opened for append through it.
    assert!(std::fs::symlink_metadata(&log)
        .unwrap()
        .file_type()
        .is_symlink());
    assert_eq!(std::fs::read(&secret).unwrap(), b"important\n");
}

#[test]
fn rw_log_probe_rejects_a_write_only_log_that_append_would_accept() {
    use std::os::unix::fs::PermissionsExt;
    // macOS launchd opens StandardErrorPath READ/WRITE; Linux systemd `append:`-
    // opens it. The R/W probe (`ensure_rw_log`, used for the macOS LaunchAgent)
    // must REJECT a WRITE-ONLY (0200) node.log — which the daemon's own R/W open
    // would fail — while the APPEND probe (Linux) legitimately ACCEPTS it (append
    // needs only write). This is the difference a reinstall must catch BEFORE it
    // boots out the working agent.
    let dir = tempfile::tempdir().unwrap();
    let log = dir.path().join("node.log");
    std::fs::write(&log, b"").unwrap();
    std::fs::set_permissions(&log, std::fs::Permissions::from_mode(0o200)).unwrap();
    assert!(
        ensure_rw_log(&log).is_err(),
        "the R/W probe must reject a write-only log (launchd opens it read/write)"
    );
    assert!(
        ensure_appendable_log(&log).is_ok(),
        "the append probe accepts a write-only log (append needs only write)"
    );
    std::fs::set_permissions(&log, std::fs::Permissions::from_mode(0o644)).unwrap();
}

#[test]
fn operator_can_exec_reaps_descendants_even_on_fast_success() {
    use std::os::unix::fs::PermissionsExt;
    // A `--version` that forks a descendant and EXITS 0 quickly (does NOT time
    // out) must still not leak the child: the probe group-kills on every exit
    // path, not just timeout. Returns true (exit 0) AND the descendant is dead.
    let euid = unsafe { libc::geteuid() };
    let op = passwd_by_uid(euid).expect("current uid has a passwd entry");
    let dir = tempfile::tempdir().unwrap();
    let bin = dir.path().join("higgs");
    let pidfile = dir.path().join("descendant.pid");
    std::fs::write(
        &bin,
        format!(
            "#!/bin/sh\nsleep 300 &\necho $! > \"{}\"\nexit 0\n",
            pidfile.display()
        ),
    )
    .unwrap();
    std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o755)).unwrap();

    let started = std::time::Instant::now();
    let ok = operator_can_exec(&bin, &op, euid).unwrap();
    assert!(ok, "a binary whose --version exits 0 is runnable");
    assert!(
        started.elapsed() < std::time::Duration::from_secs(5),
        "must return promptly on fast success, not wait for the timeout"
    );
    let pid: libc::pid_t = std::fs::read_to_string(&pidfile)
        .expect("descendant wrote its pid")
        .trim()
        .parse()
        .expect("pid is numeric");
    let mut alive = true;
    for _ in 0..200 {
        let rc = unsafe { libc::kill(pid, 0) };
        if rc == -1 && std::io::Error::last_os_error().raw_os_error() == Some(libc::ESRCH) {
            alive = false;
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    if alive {
        unsafe { libc::kill(pid, libc::SIGKILL) };
    }
    assert!(
        !alive,
        "the forked descendant (pid {pid}) must be group-killed even on a fast exit-0"
    );
}

#[test]
fn operator_can_exec_bounds_a_hanging_binary_and_reaps_descendants() {
    use std::os::unix::fs::PermissionsExt;
    // A binary that HANGS on `--version` must be KILLED by the preflight timeout
    // and reported not-runnable — never stall install-service forever. AND any
    // descendant it forked (a `sleep 300 & wait` wrapper) must be reaped WITH
    // it: the probe runs in its own process group and the timeout group-kills,
    // so a descendant is not orphaned to init and left running. That the call
    // returns AT ALL (near the timeout) is the first property; that the
    // descendant is dead afterwards is the second.
    let euid = unsafe { libc::geteuid() };
    let op = passwd_by_uid(euid).expect("current uid has a passwd entry");
    let dir = tempfile::tempdir().unwrap();
    let bin = dir.path().join("higgs");
    let pidfile = dir.path().join("descendant.pid");
    // Fork a `sleep 300` DESCENDANT, record its pid, then hang in `wait`. A
    // non-interactive `sh` has no job control, so the background job stays in
    // the shell's process group — reachable by the group kill (only if the
    // probe put the shell in its own group via setsid).
    std::fs::write(
        &bin,
        format!(
            "#!/bin/sh\nsleep 300 &\necho $! > \"{}\"\nwait\n",
            pidfile.display()
        ),
    )
    .unwrap();
    std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o755)).unwrap();

    let started = std::time::Instant::now();
    let ok = operator_can_exec(&bin, &op, euid).unwrap();
    let elapsed = started.elapsed();
    assert!(!ok, "a hanging binary is not runnable");
    assert!(
        elapsed >= EXEC_PREFLIGHT_TIMEOUT
            && elapsed < EXEC_PREFLIGHT_TIMEOUT + std::time::Duration::from_secs(5),
        "must return NEAR the timeout ({EXEC_PREFLIGHT_TIMEOUT:?}), not early and not hang: {elapsed:?}"
    );

    // The descendant's pid was recorded before the hang; it must now be dead
    // (group-killed + reaped by init), NOT orphaned and still sleeping.
    let pid: libc::pid_t = std::fs::read_to_string(&pidfile)
        .expect("descendant wrote its pid")
        .trim()
        .parse()
        .expect("pid is numeric");
    // The group SIGKILL is async and init reaps asynchronously — poll briefly
    // for the pid to disappear (kill(pid, 0) → ESRCH once gone).
    let mut alive = true;
    for _ in 0..200 {
        let rc = unsafe { libc::kill(pid, 0) };
        if rc == -1 && std::io::Error::last_os_error().raw_os_error() == Some(libc::ESRCH) {
            alive = false;
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    // Belt-and-suspenders: if the fix regressed and it is still alive, kill it
    // so the test does not leak a 300 s sleep.
    if alive {
        unsafe { libc::kill(pid, libc::SIGKILL) };
    }
    assert!(
        !alive,
        "the forked descendant (pid {pid}) must be group-killed with the probe, not orphaned"
    );
}

#[test]
fn operator_group_list_resolves_groups_in_the_parent() {
    // The parent-side resolver (moved OUT of pre_exec for async-signal-safety)
    // must return the operator's real supplementary set, always including the
    // primary gid — otherwise the group-writable-prefix probe would regress.
    let uid = unsafe { libc::geteuid() };
    let op = passwd_by_uid(uid).expect("current uid has a passwd entry");
    let user = std::ffi::CString::new(op.user.as_bytes()).unwrap();
    let groups = operator_group_list(&user, op.gid).expect("group list resolves");
    assert!(!groups.is_empty(), "at least the primary gid");
    assert!(
        groups.contains(&op.gid),
        "primary gid {} missing from {groups:?}",
        op.gid
    );
}

#[test]
fn node_rejects_unknown_subcommand() {
    assert!(run_node(&["bogus".into()]).is_err());
    assert!(run_node(&[]).is_err());
}

#[test]
fn node_connect_requires_a_ticket() {
    // No ticket arg → usage error, before any runtime/bind.
    assert!(run_node(&["connect".into()]).is_err());
}

#[test]
fn node_connect_rejects_malformed_ticket() {
    // A malformed ticket fails at parse, before any runtime/bind/network.
    let err = run_node(&["connect".into(), "not-a-ticket".into()]).unwrap_err();
    assert!(!err.to_string().is_empty());
}

#[test]
fn install_service_env_flag_is_allowlisted() {
    // The --env allowlist is a security boundary: a preserved config var is
    // accepted, but arbitrary env (LD_PRELOAD, DYLD_*) must be rejected so an
    // operator can never inject code into a root-installed daemon.
    // A well-formed allowlisted pair parses (verified via a full dry-run below);
    // here we pin the REJECTIONS the parser must make, straight from the
    // service allowlist.
    assert!(crate::node::service::PRESERVED_ENV.contains(&"HIGGS_HF_ENDPOINT"));
    assert!(!crate::node::service::PRESERVED_ENV.contains(&"LD_PRELOAD"));
    assert!(!crate::node::service::PRESERVED_ENV.contains(&"DYLD_INSERT_LIBRARIES"));
    // HIGGS_HOME/HIGGS_MODEL_DIR are handled by dedicated flags, NOT --env.
    assert!(!crate::node::service::PRESERVED_ENV.contains(&"HIGGS_HOME"));
    assert!(!crate::node::service::PRESERVED_ENV.contains(&"HIGGS_MODEL_DIR"));
    // Debug knobs must never be baked into a permanent service.
    assert!(!crate::node::service::PRESERVED_ENV.contains(&"RUST_LOG"));
    assert!(!crate::node::service::PRESERVED_ENV.contains(&"HIGGS_VERBOSE"));
}

#[test]
fn refuse_other_writable_blocks_world_write_but_allows_group() {
    use std::os::unix::fs::PermissionsExt;
    let euid = unsafe { libc::geteuid() };
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path().join("prefix");
    std::fs::create_dir_all(&d).unwrap();

    // 0755 (normal) → ok. 0775 GROUP-writable → allowed (credential-group).
    std::fs::set_permissions(&d, std::fs::Permissions::from_mode(0o755)).unwrap();
    assert!(refuse_writable_mode(&d, "prefix", false, euid, false).is_ok());
    std::fs::set_permissions(&d, std::fs::Permissions::from_mode(0o775)).unwrap();
    assert!(refuse_writable_mode(&d, "prefix", false, euid, false).is_ok());

    // 0777 / 0757 world-writable (no sticky) → refused whether strict or not.
    std::fs::set_permissions(&d, std::fs::Permissions::from_mode(0o777)).unwrap();
    let err = refuse_writable_mode(&d, "prefix", false, euid, false)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("OTHER-writable") && err.contains("chmod o-w"),
        "{err}"
    );
    std::fs::set_permissions(&d, std::fs::Permissions::from_mode(0o757)).unwrap();
    assert!(refuse_writable_mode(&d, "prefix", false, euid, false).is_err());

    // A DEFINITELY-missing dir (NotFound) is fine (created bounded later).
    assert!(refuse_writable_mode(&dir.path().join("nope"), "prefix", false, euid, false).is_ok());

    // But an INDETERMINATE stat error must NOT fall open: a path under a
    // NON-searchable (0000) parent → EACCES → refuse (we cannot prove it is safe),
    // NOT the old `Err(_) => Ok()` allow. Distinct from the benign NotFound above.
    assert_ne!(euid, 0, "needs a non-root run (root bypasses the mode)");
    let locked = dir.path().join("locked");
    std::fs::create_dir(&locked).unwrap();
    let under = locked.join("child");
    std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o000)).unwrap();
    let verdict = refuse_writable_mode(&under, "ancestor", false, euid, false);
    std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o755)).unwrap();
    assert!(
        verdict.is_err(),
        "an un-stattable (EACCES) path must refuse, not fall open to allow"
    );

    std::fs::set_permissions(&d, std::fs::Permissions::from_mode(0o755)).unwrap();
}

#[test]
fn refuse_other_writable_sticky_is_ancestor_only_not_managed() {
    use std::os::unix::fs::PermissionsExt;
    let euid = unsafe { libc::geteuid() };
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path().join("p");
    std::fs::create_dir_all(&d).unwrap();
    // 01777 world-writable + STICKY (/tmp-style):
    std::fs::set_permissions(&d, std::fs::Permissions::from_mode(0o1777)).unwrap();
    //  - as an ANCESTOR (allow_sticky=true) → ALLOWED (sticky blocks the rename).
    assert!(
        refuse_writable_mode(&d, "ancestor", true, euid, false).is_ok(),
        "sticky ancestor must be allowed"
    );
    //  - as a MANAGED leaf (allow_sticky=false) → REFUSED: a peer can still CREATE
    //    a predictable entry (node.log symlink) even in a sticky dir.
    assert!(
        refuse_writable_mode(&d, "log dir", false, euid, false).is_err(),
        "a sticky MANAGED dir must still be refused (predictable-entry plant)"
    );
    std::fs::set_permissions(&d, std::fs::Permissions::from_mode(0o755)).unwrap();
}

#[test]
fn refuse_other_writable_skips_a_symlink_node_even_when_world_writable_moded() {
    use std::os::unix::ffi::OsStrExt;
    let euid = unsafe { libc::geteuid() };
    // A symlink's OWN mode is not security-meaningful and MUST be skipped: Linux
    // ignores symlink perms entirely (`lstat` always reports 0777), so checking
    // it would reject the normal `bin/current` symlink and lock EVERY Linux
    // operator out of install-service. On macOS a symlink defaults to 0755 (so a
    // naive check passes there by accident, hiding the Linux lockout) — force it
    // world-writable via fchmodat(AT_SYMLINK_NOFOLLOW) so the revert is observable
    // on THIS platform too: without the skip, symlink_metadata() reports 0777 and
    // refuse_other_writable would return Err.
    let dir = tempfile::tempdir().unwrap();
    let target = dir.path().join("real");
    std::fs::write(&target, b"x").unwrap();
    let link = dir.path().join("current");
    std::os::unix::fs::symlink(&target, &link).unwrap();
    let c = std::ffi::CString::new(link.as_os_str().as_bytes()).unwrap();
    // Best-effort: macOS honors it (making this a real fail-on-revert here);
    // Linux symlinks are already 0777 and return ENOTSUP, which is fine.
    unsafe { libc::fchmodat(libc::AT_FDCWD, c.as_ptr(), 0o777, libc::AT_SYMLINK_NOFOLLOW) };
    assert!(
        refuse_writable_mode(&link, "service binary path", false, euid, false).is_ok(),
        "a symlink node must be SKIPPED (its 0777 lstat mode is not a real world-write)"
    );
}

#[test]
fn refuse_writable_mode_rejects_a_symlink_owned_by_an_untrusted_uid() {
    use std::os::unix::fs::MetadataExt;
    // A symlink's mode bits are meaningless, but its OWNER (lstat uid) matters: the
    // link's owner can REPOINT it — in a sticky group-writable ancestor (which the
    // strict walk sticky-exempts) a peer-owned symlink is a TOCTOU redirect vector.
    // So the owner is validated BEFORE the mode-bit skip. We cannot chown to a real
    // peer without root, so make the operator's own uid look untrusted by passing a
    // DIFFERENT op_uid: the symlink (owned by euid) must then refuse.
    let euid = unsafe { libc::geteuid() };
    let dir = tempfile::tempdir().unwrap();
    let target = dir.path().join("real");
    std::fs::write(&target, b"x").unwrap();
    let link = dir.path().join("LaunchAgents");
    std::os::unix::fs::symlink(&target, &link).unwrap();
    assert_eq!(
        std::fs::symlink_metadata(&link).unwrap().uid(),
        euid,
        "precondition: the link is owned by us"
    );
    // TRUSTED owner (op_uid == the link's owner) → skipped (Ok): only they repoint it.
    assert!(
        refuse_writable_mode(&link, "unit dir", false, euid, true).is_ok(),
        "a symlink owned by the operator is fine"
    );
    // UNTRUSTED owner (a different op_uid, so the real owner looks like a peer) →
    // REFUSE as a repoint vector.
    let other = if euid == 1 { 2 } else { 1 };
    let err = refuse_writable_mode(&link, "unit dir", false, other, true)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("UNTRUSTED uid") && err.contains("repoint"),
        "a symlink with an untrusted owner must refuse as a repoint vector: {err}"
    );
}

#[test]
fn only_the_macos_daemon_exec_rejects_group_write() {
    use crate::node::service::ServiceKind::*;
    // The macOS `--system` LaunchDaemon is installed via `sudo`, so a group peer's
    // swapped binary would run as ROOT → its exec ancestry rejects group-write. The
    // login-bound agent and the systemd USER unit run as the operator (no sudo),
    // where group-write is the safe credential-group feature.
    assert!(exec_ancestry_rejects_group(Launchd));
    assert!(!exec_ancestry_rejects_group(LaunchdAgent));
    assert!(!exec_ancestry_rejects_group(SystemdUser));
}

#[test]
fn acl_line_grants_write_parses_from_the_right_for_spaced_principals() {
    // Pure-fn seam (no real ACL needed, so a spaced Directory Service group — which
    // needs admin to create on-box — is testable). The principal is printed UNQUOTED
    // by `ls`, so a group with a space spans multiple fields; the parser anchors on
    // the LAST field (perms) and second-to-last (allow/deny), tolerating both the
    // space AND the optional `inherited` flag.
    let me = "user:alice";
    // Spaced DS record name, DIRECT allow-write → flagged (a left-parse read
    // `Operators` as the type and MISSED this).
    assert!(acl_line_grants_write(
        " 1: group:Higgs Operators allow add_file,delete_child",
        me
    ));
    // Spaced principal, INHERITED allow-write → flagged (both shifts at once).
    assert!(acl_line_grants_write(
        " 0: group:Higgs Operators inherited allow add_file,file_inherit,directory_inherit",
        me
    ));
    // Spaced principal, DENY → not flagged (grants a peer nothing).
    assert!(!acl_line_grants_write(
        " 0: group:Higgs Operators deny add_file",
        me
    ));
    // Spaced principal, READ-ONLY allow → not flagged.
    assert!(!acl_line_grants_write(
        " 0: group:Higgs Operators allow read,list,search",
        me
    ));
    // Non-spaced everyone allow-write → flagged.
    assert!(acl_line_grants_write(" 0: group:everyone allow write", me));
    // Allow-to-SELF (direct, unambiguous) → not flagged.
    assert!(!acl_line_grants_write(" 0: user:alice allow write", me));
    // AMBIGUITY (r48): an inherited ACE for operator `alice` renders IDENTICALLY to a
    // direct ACE for a DIFFERENT user named `alice inherited`. Comparing the FULL
    // principal span (never a flag-stripped name) treats this as a NON-self grant and
    // FLAGS it — a peer `user:alice inherited` must NOT masquerade as operator `alice`
    // and slip its write grant past the check.
    assert!(acl_line_grants_write(
        " 0: user:alice inherited allow add_file,delete_child",
        me
    ));
    // A principal that CONTAINS "write" but grants READ only → NOT flagged (perms are
    // scanned, never the whole line, so `group:writers` is not a false match).
    assert!(!acl_line_grants_write(" 0: group:writers allow read", me));
    // The `ls -l` header line (mode/size/name, not an ACE `N:`) → not an entry.
    assert!(!acl_line_grants_write(
        "drwxr-xr-x@ 3 alice wheel 96 Jul 21 12:00 dir",
        me
    ));
    // An ACE with no perms field → too few fields → not flagged.
    assert!(!acl_line_grants_write(" 0: group:everyone allow", me));
}

#[cfg(target_os = "macos")]
#[test]
fn has_writable_acl_flags_write_grants_but_not_deny_or_self() {
    let me = passwd_by_uid(unsafe { libc::geteuid() }).unwrap().user;
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path().join("acl");
    std::fs::create_dir_all(&d).unwrap();
    let chmod = |spec: &str| {
        std::process::Command::new("chmod")
            .args(["+a", spec])
            .arg(&d)
            .status()
            .unwrap()
            .success()
    };
    // No ACL → false.
    assert!(!has_writable_acl(&d, &me), "no ACL initially");
    // A benign `everyone deny delete` (the kind macOS homes carry) → NOT flagged:
    // it grants a peer no write, so flagging it would falsely block the default
    // ~/.higgs install.
    assert!(chmod("everyone deny delete"));
    assert!(
        !has_writable_acl(&d, &me),
        "a deny-only ACL must NOT be flagged"
    );
    // An `allow write` to EVERYONE (a real write grant to a non-owner) → flagged.
    assert!(chmod("everyone allow write"));
    assert!(
        has_writable_acl(&d, &me),
        "an allow-write ACL to a non-owner must be flagged"
    );
    // Reset; an `allow write` to the OPERATOR THEMSELVES grants a peer nothing → NOT
    // flagged.
    let d2 = dir.path().join("acl2");
    std::fs::create_dir_all(&d2).unwrap();
    let ok = std::process::Command::new("chmod")
        .args(["+a", &format!("user:{me} allow write")])
        .arg(&d2)
        .status()
        .unwrap()
        .success();
    assert!(ok, "chmod +a self");
    assert!(
        !has_writable_acl(&d2, &me),
        "an allow-write to the operator's own user must NOT be flagged"
    );

    // INHERITED ACE: macOS renders a propagated ACE as `principal INHERITED allow
    // perms` — an extra field before `allow`. A parser expecting `principal allow`
    // would miss an inherited group-write grant on a 0755 dir (the exact HIGH). Set
    // an INHERITABLE allow-write ACE on a parent; a child created under it inherits
    // the ACE and MUST be flagged.
    let parent = dir.path().join("inh_parent");
    std::fs::create_dir_all(&parent).unwrap();
    let ok = std::process::Command::new("chmod")
        .args(["+a", "everyone allow write,file_inherit,directory_inherit"])
        .arg(&parent)
        .status()
        .unwrap()
        .success();
    assert!(ok, "chmod +a inheritable on parent");
    let child = parent.join("child");
    std::fs::create_dir(&child).unwrap();
    assert!(
        has_writable_acl(&child, &me),
        "an INHERITED allow-write ACE (`principal inherited allow …`) must be flagged"
    );

    // INSPECTION FAILURE: a dir that EXISTS but whose ACL we cannot READ (here:
    // `everyone deny readsecurity` → `ls -lde` fails EACCES) must be read as UNSAFE
    // (present), NOT "no ACL" — a root-owned ancestor could hide a peer-write grant
    // this way. (An ABSENT path stays benign → false.)
    let unreadable = dir.path().join("unreadable");
    std::fs::create_dir_all(&unreadable).unwrap();
    let ok = std::process::Command::new("chmod")
        .args(["+a", "everyone deny readsecurity"])
        .arg(&unreadable)
        .status()
        .unwrap()
        .success();
    assert!(ok, "chmod +a deny readsecurity");
    let verdict = has_writable_acl(&unreadable, &me);
    // Reset so tempdir cleanup is unhindered, regardless of the assertion outcome.
    let _ = std::process::Command::new("chmod")
        .arg("-N")
        .arg(&unreadable)
        .status();
    assert!(
        verdict,
        "an existing dir whose ACL cannot be read must be treated as UNSAFE (present)"
    );
    assert!(
        !has_writable_acl(&dir.path().join("does-not-exist"), &me),
        "a definitely-absent path carries no ACL"
    );
}

#[cfg(target_os = "macos")]
#[test]
fn refuse_exec_acls_catches_a_write_acl_on_a_lexical_parent_behind_a_symlink() {
    // The daemon exec path is used LEXICALLY by the service unit and the printed
    // `sudo …` re-run. A write-granting ACL on a 0755 LEXICAL parent (mode bits pass)
    // reached THROUGH an operator-owned symlink is invisible to a resolved-ONLY ACL
    // walk (which sees only the clean symlink target) — a peer with that ACL repoints
    // the symlink before the root exec. `refuse_exec_acls` must walk the LEXICAL chain
    // too and refuse.
    let me = passwd_by_uid(unsafe { libc::geteuid() }).unwrap().user;
    let root = tempfile::tempdir().unwrap();
    // Clean RESOLVED target: <root>/clean (holds the "binary").
    let clean = root.path().join("clean");
    std::fs::create_dir_all(&clean).unwrap();
    // Lexical parent carrying a write ACL, with a symlink INTO the clean target.
    let aclparent = root.path().join("aclparent");
    std::fs::create_dir_all(&aclparent).unwrap();
    assert!(std::process::Command::new("chmod")
        .args(["+a", "everyone allow write"])
        .arg(&aclparent)
        .status()
        .unwrap()
        .success());
    let link = aclparent.join("bin");
    std::os::unix::fs::symlink(&clean, &link).unwrap();
    // exec_target spelled through the symlink: <aclparent>/bin/higgs. Its LEXICAL
    // ancestry includes the ACL'd `aclparent`; its RESOLVED form is <root>/clean/higgs
    // (clean ancestry).
    let exec_target = link.join("higgs");
    let verdict = refuse_exec_acls(&exec_target, &me);
    assert!(
        verdict.is_err(),
        "a write ACL on a LEXICAL parent behind a symlinked exec path must refuse"
    );
    // Sanity: the RESOLVED target ancestry alone is clean (so a resolved-only walk
    // would have PASSED — that is the gap this closes).
    assert!(
        refuse_writable_acl_ancestry(&clean.join("higgs"), &me, "resolved").is_ok(),
        "the resolved target ancestry is clean (resolved-only would miss the ACL)"
    );
}

#[cfg(target_os = "macos")]
#[test]
fn refuse_writable_acl_ancestry_catches_a_write_acl_on_an_ancestor_not_just_the_leaf() {
    // The exec/unit-dir ACL check must walk the FULL ancestry, not only the managed
    // endpoints: a write-granting ACL on a PARENT (e.g. `~/Library`) lets a peer
    // replace the subtree below it while every endpoint stays ACL-free. An
    // endpoint-only check misses this; the ancestry walk must catch it.
    let me = passwd_by_uid(unsafe { libc::geteuid() }).unwrap().user;
    let root = tempfile::tempdir().unwrap();
    let parent = root.path().join("parent");
    let leaf = parent.join("child");
    std::fs::create_dir_all(&leaf).unwrap();
    // Clean ancestry → OK.
    refuse_writable_acl_ancestry(&leaf, &me, "test path").expect("a clean ancestry must pass");
    // Grant EVERYONE write on the PARENT only (the leaf stays ACL-free).
    let ok = std::process::Command::new("chmod")
        .args(["+a", "everyone allow write"])
        .arg(&parent)
        .status()
        .unwrap()
        .success();
    assert!(ok, "chmod +a on parent");
    // The leaf itself carries no ACL, so an endpoint-only check would pass — the
    // ancestry walk must REFUSE because of the parent's grant.
    assert!(
        !has_writable_acl(&leaf, &me),
        "sanity: the leaf itself has no write ACL (endpoint-only check would miss it)"
    );
    let err = refuse_writable_acl_ancestry(&leaf, &me, "test path")
        .expect_err("a write ACL on an ANCESTOR must be refused");
    assert!(
        err.to_string().contains("macOS ACL granting write"),
        "the refusal must name the ACL vector: {err}"
    );
}

#[test]
fn operator_higgs_home_from_infers_or_refuses_the_state_dir() {
    // Pure seam (env passed as params, no process-global mutation). HIGGS_HOME wins
    // when set; an EMPTY one is refused; elevated (euid 0) with none refuses; a
    // matching HOME resolves the default; a mismatched HOME refuses; and — the r54
    // fix — an UNSET HOME (cron/`env -u HOME`) REFUSES rather than silently pinning
    // the passwd default (which could restart-loop on a custom-HOME-paired node).
    use std::ffi::OsString;
    let op = passwd_by_uid(unsafe { libc::geteuid() }).expect("passwd entry");
    let some = |s: &str| Some(OsString::from(s));

    // HIGGS_HOME set → used verbatim (absolutized).
    assert_eq!(
        operator_higgs_home_from(some("/srv/state"), None, &op, op.uid).unwrap(),
        std::path::Path::new("/srv/state")
    );
    // HIGGS_HOME set but EMPTY → refuse.
    assert!(operator_higgs_home_from(some(""), None, &op, op.uid).is_err());
    // Elevated (euid 0), no HIGGS_HOME → refuse (sudo strips HOME/HIGGS_HOME).
    assert!(operator_higgs_home_from(None, some("/root"), &op, 0).is_err());
    // Non-elevated, HOME matches the passwd home → the default store resolves.
    assert_eq!(
        operator_higgs_home_from(None, Some(op.home.clone().into_os_string()), &op, op.uid)
            .unwrap(),
        op.home.join(".higgs")
    );
    // Non-elevated, HOME differs → refuse (ambiguous paired dir).
    assert!(
        operator_higgs_home_from(None, some("/some/other/home"), &op, op.uid).is_err(),
        "a mismatched HOME must refuse"
    );
    // Non-elevated, HOME UNSET → refuse (r54): cannot infer a custom-HOME-paired dir.
    let err = operator_higgs_home_from(None, None, &op, op.uid)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("HOME is unset"),
        "an unset HOME must refuse, not pin the passwd default: {err}"
    );
    // An EMPTY HOME is treated like unset → refuse.
    assert!(operator_higgs_home_from(None, some(""), &op, op.uid).is_err());
}

#[test]
fn dir_fsync_aborts_on_real_errors_but_warns_on_unsupported() {
    // A filesystem that simply can't fsync a directory (ENOTSUP/EOPNOTSUPP/EINVAL/
    // EBADF) → warn, don't abort. A REAL storage error (EIO/ENOSPC/…) left the
    // rename non-durable → abort before the destructive manager commands.
    assert!(!dir_fsync_should_abort(Some(libc::ENOTSUP)));
    assert!(!dir_fsync_should_abort(Some(libc::EOPNOTSUPP)));
    assert!(!dir_fsync_should_abort(Some(libc::EINVAL)));
    assert!(!dir_fsync_should_abort(Some(libc::EBADF)));
    assert!(dir_fsync_should_abort(Some(libc::EIO)));
    assert!(dir_fsync_should_abort(Some(libc::ENOSPC)));
    assert!(
        dir_fsync_should_abort(None),
        "an unknown error aborts (conservative)"
    );
}

#[test]
fn daemon_job_loaded_is_false_when_no_daemon_is_loaded() {
    // The probe shells `launchctl print system/com.higgs.node` and reads its EXIT
    // CODE (readable without root). No higgs daemon is loaded in the test env, so
    // it must report false (and never panic / hang). Best-effort: a non-macOS /
    // no-launchctl box also yields false. (The true path needs a loaded daemon,
    // which a test can't create; the refuse-on-loaded wiring is inspection-covered.)
    assert!(!daemon_job_loaded());
}

#[test]
fn refuse_writable_ancestry_strict_also_rejects_group_write() {
    use std::os::unix::fs::PermissionsExt;
    // The launchd unit dir (`~/Library/LaunchAgents`) is OUTSIDE the group-trusted
    // prefix: a GROUP peer who could write there can replace the plist and have
    // launchd exec code as the operator. The group-tolerant walk allows group-write
    // (credential-group prefix); the STRICT walk must REJECT it.
    let euid = unsafe { libc::geteuid() };
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path().join("LaunchAgents");
    std::fs::create_dir_all(&d).unwrap();
    std::fs::set_permissions(&d, std::fs::Permissions::from_mode(0o775)).unwrap();
    assert!(
        refuse_writable_ancestry(&d, &d, "unit dir", euid).is_ok(),
        "the group-tolerant walk allows group-write (credential group)"
    );
    assert!(
        refuse_writable_ancestry_strict(&d, &d, "unit dir", euid).is_err(),
        "the strict walk (launchd unit dir) must reject group-write"
    );
    std::fs::set_permissions(&d, std::fs::Permissions::from_mode(0o755)).unwrap();
}

#[test]
fn resolve_required_refuses_an_unresolvable_path_never_falls_back_to_lexical() {
    // The service prefix/binary MUST resolve; a canonicalize failure (missing binary,
    // or a peer renaming a resolved component away mid-check) must REFUSE, never fall
    // open to the lexical spelling — the exec/ACL resolved walks would otherwise skip.
    let dir = tempfile::tempdir().unwrap();
    assert!(
        resolve_required(dir.path(), "prefix").is_ok(),
        "a real dir resolves"
    );
    let err = resolve_required(&dir.path().join("nope"), "service binary")
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("could not resolve"),
        "a missing path must refuse: {err}"
    );
    // A symlink whose target is GONE (the hide/race vector) → Err, not a fallback.
    let dangling = dir.path().join("link");
    std::os::unix::fs::symlink(dir.path().join("gone"), &dangling).unwrap();
    assert!(
        resolve_required(&dangling, "service binary").is_err(),
        "a symlink to a vanished target must refuse"
    );
}

#[test]
fn resolve_if_present_skips_absent_but_refuses_a_hidden_symlink_target() {
    // The logs dir is validated ONLY IF present. A DEFINITE absence is benign (created
    // fresh later) → None. But a path that EXISTS as a symlink whose target is HIDDEN
    // must REFUSE, not silently skip — the exact fail-open a symlinked-away external
    // logs dir exploits to dodge the resolved-ancestry walk.
    let dir = tempfile::tempdir().unwrap();
    assert!(
        resolve_if_present(&dir.path().join("logs"), "log dir")
            .unwrap()
            .is_none(),
        "a definitely-absent path is a benign skip"
    );
    let real = dir.path().join("logs");
    std::fs::create_dir(&real).unwrap();
    assert!(
        resolve_if_present(&real, "log dir").unwrap().is_some(),
        "a real dir resolves"
    );
    std::fs::remove_dir(&real).unwrap();
    // `logs` is now a DANGLING symlink (symlink_metadata still lstat-sees the link, so
    // it is NOT NotFound) → canonicalize fails → Err, NOT a skip.
    std::os::unix::fs::symlink(dir.path().join("hidden-target"), &real).unwrap();
    let err = resolve_if_present(&real, "log dir")
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("could not be resolved"),
        "a symlink to a hidden target must refuse, not skip: {err}"
    );
}

#[test]
fn validate_unit_dir_refuses_a_canonicalize_failure_instead_of_skipping() {
    use crate::node::service::ServiceKind::*;
    // A prior `if let Ok(real_dir) = canonicalize(dir)` SILENTLY skipped all checks
    // when canonicalize failed, then wrote+bootstrapped into the unvalidated dir.
    // A dir that cannot be resolved (here: it does not exist — standing in for a
    // concurrent remove/repoint race) must REFUSE, not fall through to Ok.
    let euid = unsafe { libc::geteuid() };
    let dir = tempfile::tempdir().unwrap();
    let gone = dir.path().join("vanished");
    let err = validate_unit_dir(&gone, Launchd, euid, "op")
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("could not resolve the service unit dir"),
        "a canonicalize failure must refuse, not skip: {err}"
    );
}

#[test]
fn validate_unit_dir_catches_a_group_writable_lexical_parent_behind_a_symlink() {
    use crate::node::service::ServiceKind::*;
    use std::os::unix::fs::PermissionsExt;
    // The unit dir is written/loaded via its LEXICAL path. If that path reaches a
    // group-writable parent THROUGH a symlink (e.g. `~/Library/LaunchAgents` symlinked
    // out of a group-writable `~/Library`), a resolved-ONLY walk sees only the clean
    // symlink TARGET and misses the hijackable lexical parent. validate_unit_dir must
    // walk BOTH chains and refuse.
    let euid = unsafe { libc::geteuid() };
    let root = tempfile::tempdir().unwrap();
    // Clean resolved target: <root>/clean/unit (all 0755).
    let clean_unit = root.path().join("clean").join("unit");
    std::fs::create_dir_all(&clean_unit).unwrap();
    // Group-writable lexical parent: <root>/gw (0775), with a symlink `unit` → the
    // clean target. The unit dir passed in is <root>/gw/unit (lexical parent = gw).
    let gw = root.path().join("gw");
    std::fs::create_dir_all(&gw).unwrap();
    let lexical_unit = gw.join("unit");
    std::os::unix::fs::symlink(&clean_unit, &lexical_unit).unwrap();
    std::fs::set_permissions(&gw, std::fs::Permissions::from_mode(0o775)).unwrap();

    // Resolved-only would PASS (the target ancestry is clean); the lexical walk must
    // catch the group-writable `gw` and REFUSE (launchd = strict, group-write is a
    // plist-replacement vector).
    let verdict = validate_unit_dir(&lexical_unit, Launchd, euid, "op");
    std::fs::set_permissions(&gw, std::fs::Permissions::from_mode(0o755)).unwrap();
    assert!(
        verdict.is_err(),
        "a group-writable LEXICAL parent behind a symlinked unit dir must refuse"
    );
}

#[test]
fn refuse_irregular_log_rejects_a_fifo_but_allows_regular_and_absent() {
    let dir = tempfile::tempdir().unwrap();
    // Absent → fine (created fresh). (`reject_symlink=false` = the non-daemon kind.)
    let absent = dir.path().join("node.log");
    assert!(refuse_irregular_log(&absent, false).is_ok());
    // Regular file → fine.
    std::fs::write(&absent, b"").unwrap();
    assert!(refuse_irregular_log(&absent, false).is_ok());
    std::fs::remove_file(&absent).unwrap();
    // FIFO → refused (a log open on it would BLOCK the installer forever).
    let fifo = dir.path().join("node.log");
    let c = std::ffi::CString::new(fifo.to_string_lossy().as_bytes()).unwrap();
    assert_eq!(unsafe { libc::mkfifo(c.as_ptr(), 0o644) }, 0, "mkfifo");
    let err = refuse_irregular_log(&fifo, false).unwrap_err().to_string();
    assert!(
        err.contains("not a regular file"),
        "must name the problem: {err}"
    );
    std::fs::remove_file(&fifo).unwrap();
    // A SYMLINK node.log: allowed for the non-daemon (operator context, O_NOFOLLOW
    // probes), but REFUSED for the DAEMON (`reject_symlink=true`) — ROOT launchd
    // re-opens it without O_NOFOLLOW, so it would redirect root's writes.
    let link = dir.path().join("node.log");
    std::os::unix::fs::symlink(dir.path().join("target"), &link).unwrap();
    assert!(
        refuse_irregular_log(&link, false).is_ok(),
        "a symlink is left to O_NOFOLLOW for the non-daemon kind"
    );
    let err = refuse_irregular_log(&link, true).unwrap_err().to_string();
    assert!(
        err.contains("SYMLINK") && err.contains("O_NOFOLLOW"),
        "the daemon must refuse a symlinked node.log: {err}"
    );
    std::fs::remove_file(&link).unwrap();
}

#[test]
fn rollback_only_removes_a_brand_new_launchd_plist() {
    use crate::node::service::ServiceKind::*;
    // A BRAND-NEW daemon plist WHILE the leftover agent plist SURVIVES (the switch's
    // required rm failed / not yet run) → roll back, so the daemon plist does not
    // load at reboot beside the surviving agent (two nodes).
    assert!(should_rollback_unit(Launchd, false, true));
    assert!(should_rollback_unit(LaunchdAgent, false, true));
    // PHASE-AWARE: if the agent is already GONE (rm succeeded, then a later command
    // like `bootstrap` failed), the daemon plist is the ONLY definition left — KEEP
    // it (deleting would leave nothing; it loads at reboot and the node returns).
    assert!(!should_rollback_unit(Launchd, false, false));
    assert!(!should_rollback_unit(LaunchdAgent, false, false));
    // A REINSTALL (the unit PRE-EXISTED) → KEEP it; deleting would leave the node
    // absent at the next login/reboot (the failure is usually transient — retry).
    assert!(!should_rollback_unit(Launchd, true, true));
    assert!(!should_rollback_unit(LaunchdAgent, true, true));
    // systemd → NEVER delete (even when new): removing the unit would leave its
    // `enable` symlink dangling.
    assert!(!should_rollback_unit(SystemdUser, false, true));
    assert!(!should_rollback_unit(SystemdUser, true, true));
}

#[test]
fn rollback_unit_file_removes_the_plist_and_reports_honestly() {
    // The rollback must actually UNLINK the just-written plist (so it can't load
    // beside the survivor at reboot) and report "rolled back". If the unlink fails
    // (nothing there to remove), it must say "COULD NOT roll back" rather than
    // falsely claim recovery. The post-unlink dir fsync is best-effort durability
    // (not observable here); the removal + message IS the mutant-testable contract.
    let dir = tempfile::tempdir().unwrap();
    let unit = dir.path().join("com.higgs.node.plist");
    std::fs::write(&unit, b"<plist/>").unwrap();
    let msg = rollback_unit_file(&unit, dir.path());
    assert!(
        !unit.exists(),
        "the plist must be unlinked so it cannot load beside the survivor"
    );
    assert!(
        msg.contains("rolled back") && !msg.contains("COULD NOT"),
        "a successful unlink must report `rolled back`: {msg}"
    );
    // A second call (file already gone) → the unlink fails → honest failure text.
    let msg2 = rollback_unit_file(&unit, dir.path());
    assert!(
        msg2.contains("COULD NOT roll back"),
        "a failed unlink must report `COULD NOT roll back`: {msg2}"
    );
}

#[test]
fn refuse_writable_mode_rejects_an_untrusted_owner_regardless_of_mode() {
    use std::os::unix::fs::MetadataExt;
    // A dir OWNED by an untrusted user is a rename/replace vector even at a benign
    // 0755 (the owner can chmod/rename it). The temp dir is owned by the current
    // uid; vary `op_uid` to exercise the owner gate without needing root to chown.
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path().join("drop");
    std::fs::create_dir_all(&d).unwrap();
    // Pin 0755 explicitly (a concurrent test's umask must not make it other-writable).
    std::fs::set_permissions(&d, std::fs::Permissions::from_mode(0o755)).unwrap();
    let owner = std::fs::metadata(&d).unwrap().uid();
    // op_uid == owner → trusted → a plain 0755 passes.
    assert!(refuse_writable_mode(&d, "prefix", false, owner, false).is_ok());
    // op_uid != owner (and owner != root) → UNTRUSTED → refused even though 0755.
    let err = refuse_writable_mode(&d, "prefix", false, owner.wrapping_add(1), false)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("UNTRUSTED"),
        "a dir owned by another user must be refused: {err}"
    );
}

#[test]
fn refuse_other_writable_sticky_requires_a_trusted_owner() {
    use std::os::unix::fs::{MetadataExt, PermissionsExt};
    // Sticky exempts a world-writable dir ONLY when owned by root or the OPERATOR:
    // sticky lets the DIRECTORY OWNER rename ANY entry, so an attacker-OWNED sticky
    // dir is still a swap vector. The temp dir is owned by the current uid; vary
    // `op_uid` to exercise the ownership gate without needing root to chown.
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path().join("drop");
    std::fs::create_dir_all(&d).unwrap();
    std::fs::set_permissions(&d, std::fs::Permissions::from_mode(0o1777)).unwrap();
    let owner = std::fs::metadata(&d).unwrap().uid();
    // op_uid == the dir's owner → trusted → sticky exemption applies → allowed.
    assert!(
        refuse_writable_mode(&d, "ancestor", true, owner, false).is_ok(),
        "a sticky ancestor owned by the operator is exempt"
    );
    // op_uid != owner (and owner != root) → UNTRUSTED → refused despite sticky.
    let other = owner.wrapping_add(1);
    assert!(
        refuse_writable_mode(&d, "ancestor", true, other, false).is_err(),
        "a sticky ancestor owned by SOMEONE ELSE must be refused (its owner can rename entries)"
    );
    std::fs::set_permissions(&d, std::fs::Permissions::from_mode(0o755)).unwrap();
}

#[test]
fn refuse_writable_ancestry_marks_the_external_entry_dir_strict() {
    use std::os::unix::fs::PermissionsExt;
    let euid = unsafe { libc::geteuid() };
    // The directory that holds the service's PREDICTABLE entry (a version dir
    // holding `higgs`) must be STRICT even OUTSIDE the managed prefix: sticky does
    // NOT stop a peer from CREATING that predictable name, so a STICKY
    // world-writable external entry dir (e.g. `current` → an external 01777 dir a
    // peer plants `higgs` in) must still be refused — else the preflight execs an
    // attacker binary as the operator.
    let dir = tempfile::tempdir().unwrap();
    let managed = dir.path().join("prefix"); // an UNRELATED managed root
    std::fs::create_dir_all(&managed).unwrap();
    let ext = dir.path().join("ext");
    let ext_ver = ext.join("v1"); // an external version dir
    std::fs::create_dir_all(&ext_ver).unwrap();
    let bin = ext_ver.join("higgs");
    std::fs::write(&bin, b"x").unwrap();
    // ext/v1 STICKY world-writable (01777): a peer may pre-create `higgs` there.
    std::fs::set_permissions(&ext_ver, std::fs::Permissions::from_mode(0o1777)).unwrap();
    assert!(
        refuse_writable_ancestry(&bin, &managed, "resolved service binary", euid).is_err(),
        "an external STICKY world-writable ENTRY (version) dir must be refused"
    );
    // A PURE ANCESTOR above the entry dir keeps its sticky exemption (rename-only
    // risk, which sticky blocks) — guards against over-strictness.
    std::fs::set_permissions(&ext_ver, std::fs::Permissions::from_mode(0o755)).unwrap();
    std::fs::set_permissions(&ext, std::fs::Permissions::from_mode(0o1777)).unwrap();
    assert!(
        refuse_writable_ancestry(&bin, &managed, "resolved service binary", euid).is_ok(),
        "a sticky PURE-ANCESTOR above the entry dir stays exempt"
    );
    std::fs::set_permissions(&ext, std::fs::Permissions::from_mode(0o755)).unwrap();
}

#[test]
fn refuse_writable_ancestry_strict_inside_prefix_sticky_ok_above() {
    use std::os::unix::fs::PermissionsExt;
    let euid = unsafe { libc::geteuid() };
    let dir = tempfile::tempdir().unwrap();
    // Layout: <root>/drop/prefix/bin/higgs. prefix = <root>/drop/prefix.
    let prefix = dir.path().join("drop").join("prefix");
    let bin = prefix.join("bin");
    std::fs::create_dir_all(&bin).unwrap();
    let leaf = bin.join("higgs");
    std::fs::write(&leaf, b"x").unwrap();

    // A world-writable dir INSIDE the prefix (bin) → refused even if sticky.
    std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o1777)).unwrap();
    assert!(
        refuse_writable_ancestry(&leaf, &prefix, "binary", euid).is_err(),
        "a sticky world-writable MANAGED bin must be refused"
    );
    std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o755)).unwrap();

    // A world-writable ANCESTOR above the prefix (drop) → refused if NOT sticky…
    let drop = dir.path().join("drop");
    std::fs::set_permissions(&drop, std::fs::Permissions::from_mode(0o777)).unwrap();
    assert!(
        refuse_writable_ancestry(&leaf, &prefix, "binary", euid).is_err(),
        "a non-sticky world-writable ancestor must be refused"
    );
    // …but ALLOWED if sticky (the operator-owned prefix can't be renamed).
    std::fs::set_permissions(&drop, std::fs::Permissions::from_mode(0o1777)).unwrap();
    assert!(
        refuse_writable_ancestry(&leaf, &prefix, "binary", euid).is_ok(),
        "a sticky ancestor above the prefix must be allowed"
    );
    std::fs::set_permissions(&drop, std::fs::Permissions::from_mode(0o755)).unwrap();
}

// --- error_chain: full cause chain, no duplicated adjacent segments ---

#[test]
fn error_chain_walks_sources_and_dedupes_identical_adjacent_segments() {
    // io::Error::other delegates Display to the inner error AND returns it from
    // source() — a naive walk would print the same message twice at the top.
    let inner = std::io::Error::new(std::io::ErrorKind::TimedOut, "timed out");
    let wrapped = std::io::Error::other(inner);
    assert_eq!(error_chain(&wrapped), "timed out");

    #[derive(Debug)]
    struct Outer(std::io::Error);
    impl std::fmt::Display for Outer {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "dial failed")
        }
    }
    impl std::error::Error for Outer {
        fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
            Some(&self.0)
        }
    }
    let chained = Outer(std::io::Error::other(std::io::Error::new(
        std::io::ErrorKind::TimedOut,
        "timed out",
    )));
    assert_eq!(error_chain(&chained), "dial failed → timed out");
}

#[test]
fn render_plan_notes_separates_quick_reference_from_prose() {
    let style = crate::node::preflight::Style { enabled: false };
    let notes = vec![
        "logs:  /p/logs/node.log".to_string(),
        "state: HIGGS_HOME=/p".to_string(),
        "the unit file stays at /p/higgs-node.service — keep --prefix mounted at boot".to_string(),
        "status: systemctl --user status higgs-node.service".to_string(),
        "login-bound: the unit runs only while op has a session".to_string(),
    ];
    let out = render_plan_notes(&style, &notes);
    // kv lines: aligned two-space-indented `key:` entries, content verbatim.
    assert!(out.contains("  logs:   /p/logs/node.log"), "{out}");
    assert!(out.contains("  state:  HIGGS_HOME=/p"), "{out}");
    assert!(
        out.contains("  status: systemctl --user status higgs-node.service"),
        "{out}"
    );
    // prose lines: `! `-marked paragraphs, separated by a blank line.
    assert!(out.contains("\n\n! the unit file stays at"), "{out}");
    assert!(
        out.contains("\n\n! login-bound: the unit runs only"),
        "{out}"
    );
}
