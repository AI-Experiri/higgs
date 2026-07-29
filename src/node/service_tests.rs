use std::path::Path;

use super::*;

fn p(s: &str) -> &Path {
    Path::new(s)
}

// ── launchd plist rendering ────────────────────────────────────────────────────

#[test]
fn plist_pins_the_operator_and_routes_through_current() {
    let plist = launchd_plist(
        Some("alice"),
        p("/Users/alice"),
        p("/Users/alice/.higgs"),
        p("/Users/alice/.higgs"),
        None,
        &[],
    );
    assert!(plist.contains("<string>com.higgs.node</string>"));
    assert!(plist.contains("<string>/Users/alice/.higgs/bin/current/higgs</string>"));
    assert!(plist.contains("<string>--node</string>"));
    // UserName pins the daemon to the operator — a LaunchDaemon defaults to root.
    assert!(plist.contains("<key>UserName</key>\n    <string>alice</string>"));
    // launchd gives daemons no HOME; higgs resolves ~/.higgs through it.
    assert!(plist.contains("<key>HOME</key>\n        <string>/Users/alice</string>"));
    assert!(plist.contains("<key>KeepAlive</key>\n    <true/>"));
    assert!(plist.contains("<key>RunAtLoad</key>\n    <true/>"));
    assert!(plist.contains("<string>/Users/alice/.higgs/logs/node.log</string>"));
    // Umask 022 (18 decimal) so launchd re-creates node.log 0644 after a rotation
    // (the installer's umask does not govern launchd's later log recreation).
    assert!(
        plist.contains("<key>Umask</key>\n    <integer>18</integer>"),
        "plist must pin Umask=022 (18): {plist}"
    );
}

#[test]
fn plist_pins_a_custom_higgs_home_independent_of_prefix() {
    // A node paired under a custom HIGGS_HOME must boot against that state dir,
    // not the prefix/HOME-derived default. The plist pins it explicitly, and it
    // is a SEPARATE value from the install prefix.
    let plist = launchd_plist(
        Some("alice"),
        p("/Users/alice"),
        p("/opt/higgs"),
        p("/var/lib/higgs-state"),
        None,
        &[],
    );
    assert!(
        plist.contains("<key>HIGGS_HOME</key>\n        <string>/var/lib/higgs-state</string>"),
        "plist must pin the custom state dir: {plist}"
    );
    // The binary still routes through the prefix — HIGGS_HOME did not overwrite it.
    assert!(plist.contains("<string>/opt/higgs/bin/current/higgs</string>"));
}

#[test]
fn plist_xml_escapes_operator_controlled_paths() {
    let plist = launchd_plist(
        Some("a&b"),
        p("/Users/a&b"),
        p("/Users/a&b/.higgs"),
        p("/state/a&b"),
        None,
        &[],
    );
    assert!(plist.contains("<string>a&amp;b</string>"));
    assert!(plist.contains("<string>/Users/a&amp;b/.higgs/bin/current/higgs</string>"));
    // The HIGGS_HOME value is XML-escaped too.
    assert!(plist.contains("<key>HIGGS_HOME</key>\n        <string>/state/a&amp;b</string>"));
    assert!(
        !plist.contains("a&b"),
        "raw ampersand must never reach the XML"
    );
}

// ── systemd unit rendering ─────────────────────────────────────────────────────

#[test]
fn systemd_unit_survives_crash_loops_and_routes_through_current() {
    let unit = systemd_unit(
        p("/home/bob"),
        p("/home/bob/.higgs"),
        p("/home/bob/.higgs"),
        None,
        &[],
    );
    // Type=exec so `systemctl restart` fails on a missing/unusable binary
    // (default `simple` returns success before execve).
    assert!(unit.contains("Type=exec"));
    assert!(unit.contains("ExecStart=\"/home/bob/.higgs/bin/current/higgs\" --node"));
    // The state dir is pinned into the unit env (quoted assignment).
    assert!(unit.contains("Environment=\"HIGGS_HOME=/home/bob/.higgs\"\n"));
    assert!(unit.contains("Restart=always"));
    assert!(unit.contains("RestartSec=5"));
    // append: takes the rest of the line verbatim — NO quotes (a literal `"`
    // would become part of the path and fail systemd's absolute-path check).
    assert!(unit.contains("StandardOutput=append:/home/bob/.higgs/logs/node.log\n"));
    assert!(unit.contains("StandardError=append:/home/bob/.higgs/logs/node.log\n"));
    assert!(
        !unit.contains("append:\""),
        "append: path must not be quoted"
    );
    assert!(unit.contains("WantedBy=default.target"));
}

#[test]
fn systemd_unit_pins_the_operator_home() {
    // A systemd USER unit inherits HOME from the user manager, which can be
    // stale/custom — yet the daemon derives its HF token file
    // (~/.cache/huggingface/token) and the default model scan roots from HOME. The
    // unit must PIN it (like the macOS plist), env-escaped and quoted.
    let unit = systemd_unit(
        p("/home/bob"),
        p("/home/bob/.higgs"),
        p("/home/bob/.higgs"),
        None,
        &[],
    );
    assert!(
        unit.contains("Environment=\"HOME=/home/bob\"\n"),
        "the unit must pin HOME so the HF token + default model roots resolve: {unit}"
    );
    // A home with a `%` specifier is doubled (env escaping), like HIGGS_HOME.
    let unit = systemd_unit(
        p("/home/%h bob"),
        p("/home/bob/.higgs"),
        p("/home/bob/.higgs"),
        None,
        &[],
    );
    assert!(
        unit.contains("Environment=\"HOME=/home/%%h bob\"\n"),
        "HOME must get Environment= escaping: {unit}"
    );
}

#[test]
fn systemd_unit_unsets_inherited_app_config_when_not_pinned() {
    // A systemd USER unit inherits the manager env; omitting a var must NOT let a
    // stale manager value leak, so the unit UnsetEnvironment=s the app-config vars
    // this install is NOT pinning. Nothing pinned → all app vars unset (+ the
    // test-only transport override, always).
    let unit = systemd_unit(
        p("/home/bob"),
        p("/home/bob/.higgs"),
        p("/home/bob/.higgs"),
        None,
        &[],
    );
    let line = unit
        .lines()
        .find(|l| l.starts_with("UnsetEnvironment="))
        .expect("UnsetEnvironment line");
    for v in [
        "HIGGS_MODEL_DIR",
        "HIGGS_HF_ENDPOINT",
        "HIGGS_ENGINE",
        "HIGGS_IROH_LOCAL",
    ] {
        assert!(line.contains(v), "must unset {v}: {line}");
    }
    // Pinning HIGGS_MODEL_DIR + HIGGS_HF_ENDPOINT → those are NOT unset (the
    // Environment= pin must stand); the unpinned ones still unset.
    let env = vec![("HIGGS_HF_ENDPOINT".to_string(), "https://hf".to_string())];
    let unit = systemd_unit(
        p("/home/bob"),
        p("/home/bob/.higgs"),
        p("/home/bob/.higgs"),
        Some(p("/m")),
        &env,
    );
    let line = unit
        .lines()
        .find(|l| l.starts_with("UnsetEnvironment="))
        .expect("UnsetEnvironment line");
    assert!(
        !line.contains("HIGGS_MODEL_DIR"),
        "a pinned model dir must NOT be unset: {line}"
    );
    assert!(
        !line.contains("HIGGS_HF_ENDPOINT"),
        "a pinned endpoint must NOT be unset: {line}"
    );
    assert!(
        line.contains("HIGGS_ENGINE") && line.contains("HIGGS_IROH_LOCAL"),
        "unpinned + test-only vars stay unset: {line}"
    );
    assert!(unit.contains("Environment=\"HIGGS_MODEL_DIR=/m\""));
}

#[test]
fn start_limit_is_disabled_in_the_unit_section() {
    // StartLimitIntervalSec belongs to [Unit] (systemd ≥230); in [Service] it
    // is silently ignored and one bad update would brick the unit after five
    // crashes. Pin the SECTION, not just the presence.
    let unit = systemd_unit(
        p("/home/bob"),
        p("/home/bob/.higgs"),
        p("/home/bob/.higgs"),
        None,
        &[],
    );
    let unit_at = unit.find("[Unit]").expect("[Unit] section");
    let service_at = unit.find("[Service]").expect("[Service] section");
    let limit_at = unit
        .find("StartLimitIntervalSec=0")
        .expect("start-limit off");
    assert!(unit_at < limit_at && limit_at < service_at);
}

#[test]
fn systemd_unit_escapes_percent_specifiers_in_the_prefix() {
    // A literal `%` in the prefix is a systemd specifier sigil (`%h` = home,
    // …). Unescaped, `/opt/%h/.higgs` would be rewritten to the user's home
    // and the unit would exec a nonexistent path. It must be doubled to `%%`.
    let unit = systemd_unit(
        p("/home/bob"),
        p("/opt/%h/.higgs"),
        p("/opt/%h/.higgs"),
        None,
        &[],
    );
    assert!(
        unit.contains("ExecStart=\"/opt/%%h/.higgs/bin/current/higgs\" --node"),
        "unit: {unit}"
    );
    assert!(
        !unit.contains("/opt/%h/"),
        "a raw % specifier must not survive"
    );
}

#[test]
fn environment_pins_a_custom_state_dir_and_escapes_specials() {
    // A custom HIGGS_HOME is pinned independent of the prefix, quoted (so a
    // space is one value), with `%` doubled and `"`/`\` C-escaped — but a
    // literal `$` KEPT single, since Environment= (unlike ExecStart) does not
    // expand `$VAR`.
    let unit = systemd_unit(
        p("/home/bob"),
        p("/home/bob/.higgs"),
        p(r#"/srv/higgs state/%h/a"b\c/${X}"#),
        None,
        &[],
    );
    assert!(
        unit.contains(r#"Environment="HIGGS_HOME=/srv/higgs state/%%h/a\"b\\c/${X}""#),
        "unit: {unit}"
    );
    // The state dir is separate from the prefix-derived binary path.
    assert!(unit.contains("ExecStart=\"/home/bob/.higgs/bin/current/higgs\" --node"));
    // A raw % specifier must not survive in the env value.
    assert!(!unit.contains("=/srv/higgs state/%h/"), "unit: {unit}");
    // $ must NOT be doubled in Environment= (no env expansion there).
    assert!(
        !unit.contains("$${X}"),
        "env value must keep a single $: {unit}"
    );
}

#[test]
fn execstart_escapes_quote_and_backslash_in_the_prefix() {
    // ExecStart is a C-unescaped quoted word: a `"` or `\` in the prefix would
    // break the command syntax unless escaped (`\"`, `\\`). append: paths are
    // NOT C-unescaped, so their backslash stays literal.
    let unit = systemd_unit(
        p("/home/bob"),
        p(r#"/opt/a"b\c/.higgs"#),
        p("/home/bob/.higgs"),
        None,
        &[],
    );
    assert!(
        unit.contains(r#"ExecStart="/opt/a\"b\\c/.higgs/bin/current/higgs" --node"#),
        "unit: {unit}"
    );
}

#[test]
fn execstart_escapes_dollar_but_append_paths_keep_it_literal() {
    // systemd expands `${VAR}`/`$VAR` in ExecStart words even inside double
    // quotes; a literal `$` must be doubled to `$$` or the path is rewritten
    // (`/opt/${HOME}/.higgs` → the manager's $HOME) and `Type=exec` fails the
    // restart. `StandardOutput=append:` paths get specifier (%) expansion but
    // NOT env-var expansion, so their `$` stays a single literal dollar —
    // doubling it there would write a bogus `$$` into the filename.
    let unit = systemd_unit(
        p("/home/bob"),
        p("/opt/${HOME}/.higgs"),
        p("/home/bob/.higgs"),
        None,
        &[],
    );
    assert!(
        unit.contains(r#"ExecStart="/opt/$${HOME}/.higgs/bin/current/higgs" --node"#),
        "unit: {unit}"
    );
    assert!(
        !unit.contains(r#"ExecStart="/opt/${HOME}"#),
        "a raw $ in ExecStart must not survive"
    );
    assert!(
        unit.contains("append:/opt/${HOME}/.higgs/logs/node.log\n"),
        "append: path must keep a single literal $ (no env expansion there)"
    );
}

// ── plans ──────────────────────────────────────────────────────────────────────

#[test]
fn launchd_plan_requires_root_and_reloads_the_job() {
    let plan = plan_launchd(
        "alice",
        p("/Users/alice"),
        501,
        p("/Users/alice/.higgs"),
        p("/Users/alice/.higgs"),
        None,
        &[],
    );
    assert_eq!(plan.kind, ServiceKind::Launchd);
    assert_eq!(plan.root, RootRequirement::Require);
    // The state dir the daemon reads is surfaced for a --dry-run inspection.
    assert!(plan
        .notes
        .iter()
        .any(|n| n == "state: HIGGS_HOME=/Users/alice/.higgs"));
    assert_eq!(
        plan.unit_path,
        Path::new("/Library/LaunchDaemons/com.higgs.node.plist")
    );
    // enable FIRST (the required step whose FAILURE must not strand the node — if
    // it fails, nothing is torn down and the running agent survives) → REQUIRED
    // agent-plist rm (before any destructive bootout) → best-effort gui bootout →
    // assert-gone → bootout system → bootstrap.
    assert_eq!(
        plan.commands[0].argv,
        ["launchctl", "enable", "system/com.higgs.node"]
    );
    assert!(!plan.commands[0].best_effort);
    // commands[1] is the REQUIRED agent-plist rm.
    assert_eq!(plan.commands[1].argv[0], "rm");
    assert!(!plan.commands[1].best_effort);
    // commands[2] is the best-effort gui bootout.
    assert_eq!(plan.commands[2].argv[..2], ["launchctl", "bootout"]);
    assert!(plan.commands[2].best_effort);
    let bootstrap = plan
        .commands
        .iter()
        .find(|c| c.argv.get(1).map(String::as_str) == Some("bootstrap"))
        .expect("bootstrap command");
    assert!(!bootstrap.best_effort);
    assert!(bootstrap
        .argv
        .contains(&plan.unit_path.to_string_lossy().into_owned()));
}

#[test]
fn launchd_plan_teardown_of_the_agent_plist_is_required() {
    // Switching agent→daemon, an UNDELETABLE agent plist (immutable via `chflags
    // uchg`, or ACL-protected) must ABORT the install BEFORE the daemon
    // bootstraps — otherwise it reloads the agent at the next GUI login and runs
    // TWO nodes on one machine (there is no runtime singleton lock). So the
    // agent-plist `rm` must be a REQUIRED command, not best-effort (`rm -f` still
    // exits 0 on a MISSING plist, so requiring it only bites an un-removable one).
    let plan = plan_launchd(
        "alice",
        p("/Users/alice"),
        501,
        p("/Users/alice/.higgs"),
        p("/Users/alice/.higgs"),
        None,
        &[],
    );
    let agent = agent_plist_path(p("/Users/alice"))
        .to_string_lossy()
        .into_owned();
    let rm_i = plan
        .commands
        .iter()
        .position(|c| c.argv.first().map(String::as_str) == Some("rm") && c.argv.contains(&agent))
        .expect("the plan must rm the leftover agent plist during the switch");
    let rm = &plan.commands[rm_i];
    assert!(
        !rm.best_effort,
        "the agent-plist rm must be REQUIRED — an un-removable plist must abort, not proceed to bootstrap"
    );
    assert!(
        rm.as_operator,
        "the rm must run as the operator (home-traversal safety)"
    );
    let bs_i = plan
        .commands
        .iter()
        .position(|c| c.argv.get(1).map(String::as_str) == Some("bootstrap"))
        .expect("bootstrap command");
    assert!(
        rm_i < bs_i,
        "the agent teardown must run before the daemon bootstrap"
    );
    // ORDERING: the REQUIRED rm must precede EVERY destructive `bootout` — both
    // the gui bootout (stops the running AGENT) and the system bootout (stops any
    // old daemon). An un-removable agent plist must abort while the running node is
    // STILL UP, so the abort never leaves the machine with a node stopped and the
    // new one not bootstrapped (a service outage); the CLI then rolls back the
    // just-written daemon plist so no reboot double-run remains.
    let gui_bootout_i = plan
        .commands
        .iter()
        .position(|c| c.argv == ["launchctl", "bootout", "gui/501/com.higgs.node"])
        .expect("plan must boot the old agent out of the gui domain");
    let sys_bootout_i = plan
        .commands
        .iter()
        .position(|c| c.argv == ["launchctl", "bootout", "system/com.higgs.node"])
        .expect("plan must boot the old system daemon out");
    assert!(
        rm_i < gui_bootout_i && rm_i < sys_bootout_i,
        "the required agent-plist rm must run BEFORE both bootouts (no-outage-on-abort)"
    );
    // ONE-NODE INVARIANT: a REQUIRED "assert the agent is gone" step must run AFTER
    // the best-effort gui bootout and BEFORE the daemon bootstrap. If the gui
    // bootout FAILED while the agent stayed loaded, this aborts (via `launchctl
    // print gui/<uid>/<label>` exit code) rather than bootstrapping the daemon
    // alongside the still-loaded agent (two nodes).
    let assert_gone_i = plan
        .commands
        .iter()
        .position(|c| {
            c.argv.first().map(String::as_str) == Some("sh")
                && c.argv
                    .iter()
                    .any(|a| a.contains("print gui/501/com.higgs.node"))
        })
        .expect("plan must assert the agent is gone before bootstrap");
    assert!(
        !plan.commands[assert_gone_i].best_effort,
        "the assertion must be REQUIRED"
    );
    assert!(
        gui_bootout_i < assert_gone_i && assert_gone_i < bs_i,
        "assert-agent-gone must run after the gui bootout and before the bootstrap"
    );
    // NO-STRANDING ORDERING: the REQUIRED `launchctl enable system/<label>` must
    // run BEFORE the agent teardown (the `rm` and the gui bootout). `enable` is the
    // required step most worth protecting against FAILURE: were it placed after the
    // teardown and then failed, the running agent would already be booted out and
    // its plist removed, leaving NO node and nothing to restore. First, an `enable`
    // failure aborts before anything is torn down, so the running agent survives.
    // (The residue — a left-set enablement if a LATER step aborts — is inert: the
    // CLI rolls back the daemon plist on that path, so nothing loads.)
    let enable_i = plan
        .commands
        .iter()
        .position(|c| c.argv == ["launchctl", "enable", "system/com.higgs.node"])
        .expect("plan must enable the system label");
    assert!(
        !plan.commands[enable_i].best_effort,
        "the enable must be REQUIRED"
    );
    assert!(
        enable_i < rm_i && enable_i < gui_bootout_i,
        "the `enable` must run BEFORE the agent teardown (an enable failure must not strand a \
         torn-down node)"
    );
}

#[test]
fn systemd_plan_refuses_root_lingers_the_operator_and_enables_by_path() {
    let plan = plan_systemd_user(
        "bob",
        p("/home/bob"),
        p("/home/bob/.higgs"),
        p("/home/bob/.higgs"),
        None,
        &[],
        ServiceScope::SurvivesLogout,
    );
    assert_eq!(plan.kind, ServiceKind::SystemdUser);
    // Must run as the operator, never root.
    assert_eq!(plan.root, RootRequirement::Refuse);
    // The state dir the daemon reads is surfaced for a --dry-run inspection.
    assert!(plan
        .notes
        .iter()
        .any(|n| n == "state: HIGGS_HOME=/home/bob/.higgs"));
    // The linked-unit boot-persistence limitation is surfaced: a prefix on a
    // volume not mounted when the user manager starts leaves the unit unloaded.
    assert!(
        plan.notes
            .iter()
            .any(|n| n.contains("mounted at boot") && n.contains("higgs-node.service")),
        "notes must warn about a late-mounted prefix: {:?}",
        plan.notes
    );
    // The install-time access probe runs under THIS shell's groups, but the unit
    // runs under the user manager (login-time groups). The bounded residual — a
    // group gained after login needs a full re-login before the service sees it —
    // must be disclosed so a group-permitted prefix/log doesn't silently fail.
    assert!(
        plan.notes.iter().any(|n| {
            n.contains("group-based access") && n.contains("log out") && n.contains("login-time")
        }),
        "notes must disclose the manager-vs-shell supplementary-groups residual: {:?}",
        plan.notes
    );
    // The unit lives at a STABLE absolute path in the operator's own tree (never
    // a version dir, never a guessed config dir) and is installed by ENABLING it
    // BY ABSOLUTE PATH — systemd links-if-needed + enables, deciding placement.
    assert_eq!(
        plan.unit_path,
        Path::new("/home/bob/.higgs/higgs-node.service")
    );
    // Enabled BY ABSOLUTE PATH with --force (overwrites a stale link on
    // migration), in ONE command.
    let enable_force = [
        "systemctl",
        "--user",
        "enable",
        "--force",
        "/home/bob/.higgs/higgs-node.service",
    ];
    assert!(plan.commands.iter().any(|c| c.argv == enable_force));
    // No bare `link` (it EEXITs on a prefix under a search dir) — enable-by-path
    // handles both the outside-search-path and inside-search-path cases.
    assert!(
        !plan
            .commands
            .iter()
            .any(|c| c.argv.get(2).map(String::as_str) == Some("link")),
        "must not use bare `link`"
    );
    // NO `disable` — a `disable` would strip the old boot-enablement, and if the
    // forced `enable` then failed (absurd prefix under a search hierarchy) the
    // node would be left disabled at reboot. `--force` handles migration instead.
    assert!(
        !plan
            .commands
            .iter()
            .any(|c| c.argv.get(2).map(String::as_str) == Some("disable")),
        "must not `disable` (would strand a failed migration)"
    );
    // enable-linger is BEST-EFFORT (commonly needs polkit; notes carry the
    // sudo fallback), so a denial does not abort an otherwise-good install.
    let linger = plan.commands.last().expect("commands");
    assert_eq!(linger.argv[..2], ["loginctl", "enable-linger"]);
    assert!(linger.argv.contains(&"bob".to_string()));
    assert!(linger.best_effort);
    // restart (run the NEW binary now — enable-by-path persists it at boot, but
    // `enable` alone would leave an already-running old process untouched on a
    // reinstall; `restart` picks up the freshly-flipped `current`).
    assert!(plan
        .commands
        .iter()
        .any(|c| c.argv == ["systemctl", "--user", "restart", SYSTEMD_UNIT]));
}

#[test]
fn plan_install_maps_the_scope_onto_this_platform() {
    // USER-SPACE BY DEFAULT: LoginBound must NEVER touch a system domain —
    // macOS gets the LaunchAgent (Refuse root, no sudo), Linux the user unit.
    let plan = plan_install(
        "x",
        p("/tmp/h"),
        501,
        p("/tmp/h/.higgs"),
        p("/tmp/h/.higgs"),
        None,
        &[],
        ServiceScope::LoginBound,
    );
    if cfg!(target_os = "macos") {
        assert_eq!(plan.kind, ServiceKind::LaunchdAgent);
        assert_eq!(plan.root, RootRequirement::Refuse);
    } else {
        assert_eq!(plan.kind, ServiceKind::SystemdUser);
        assert_eq!(plan.root, RootRequirement::Refuse);
    }

    // --system (SurvivesLogout): macOS opts into the LaunchDaemon (Require
    // root); Linux stays the SAME user unit (never a system service).
    let plan = plan_install(
        "x",
        p("/tmp/h"),
        501,
        p("/tmp/h/.higgs"),
        p("/tmp/h/.higgs"),
        None,
        &[],
        ServiceScope::SurvivesLogout,
    );
    if cfg!(target_os = "macos") {
        assert_eq!(plan.kind, ServiceKind::Launchd);
        assert_eq!(plan.root, RootRequirement::Require);
    } else {
        assert_eq!(plan.kind, ServiceKind::SystemdUser);
        assert_eq!(plan.root, RootRequirement::Refuse);
    }
}

// ── the LoginBound macOS agent (the user-space default) ───────────────────────

#[test]
fn agent_plan_is_fully_user_space_and_targets_the_gui_domain() {
    let plan = plan_launchd_agent(
        p("/Users/alice"),
        501,
        p("/Users/alice/.higgs"),
        p("/Users/alice/.higgs"),
        None,
        &[],
    );
    assert_eq!(plan.kind, ServiceKind::LaunchdAgent);
    // No sudo, ever: the agent must refuse a root run (it would bootstrap
    // root's gui domain and write a root-owned plist in the wrong home).
    assert_eq!(plan.root, RootRequirement::Refuse);
    // The plist lives in the OPERATOR's own tree, not a system dir.
    assert_eq!(
        plan.unit_path,
        Path::new("/Users/alice/Library/LaunchAgents/com.higgs.node.plist")
    );
    // Same non-destructive-first order as the daemon, on gui/<uid>.
    assert_eq!(
        plan.commands[0].argv,
        ["launchctl", "enable", "gui/501/com.higgs.node"]
    );
    assert!(!plan.commands[0].best_effort);
    assert_eq!(
        plan.commands[1].argv,
        ["launchctl", "bootout", "gui/501/com.higgs.node"]
    );
    assert!(plan.commands[1].best_effort);
    // A best-effort clear-env step (launchctl unsetenv of the unpinned app vars)
    // runs before bootstrap so a stale `launchctl setenv` can't leak in.
    assert_eq!(plan.commands[2].argv[0], "sh");
    assert!(plan.commands[2].argv[2].contains("launchctl unsetenv"));
    assert!(plan.commands[2].best_effort);
    let bootstrap = plan
        .commands
        .iter()
        .find(|c| c.argv.get(1).map(String::as_str) == Some("bootstrap"))
        .expect("bootstrap command");
    assert_eq!(
        bootstrap.argv,
        [
            "launchctl",
            "bootstrap",
            "gui/501",
            "/Users/alice/Library/LaunchAgents/com.higgs.node.plist"
        ]
    );
    assert!(!bootstrap.best_effort);
    // An agent runs as the session user — launchd ignores UserName outside the
    // system domain, so the plist must NOT carry one.
    assert!(
        !plan.unit_content.contains("UserName"),
        "agent plist must not pin UserName: {}",
        plan.unit_content
    );
    // The login-bound trade-off is stated, with the --system escape hatch.
    assert!(
        plan.notes
            .iter()
            .any(|n| n.contains("STOPS AT LOGOUT") && n.contains("--system")),
        "notes must state the login-bound trade-off: {:?}",
        plan.notes
    );
    // PARITY with the systemd plan (r54): the agent runs under the GUI session's
    // LOGIN-TIME groups, but the install probes under this shell's groups — a group
    // gained after login (newgrp) can pass preflight while the session lacks it. The
    // macOS agent plan must disclose this residual, as the systemd plan does.
    assert!(
        plan.notes.iter().any(|n| {
            n.contains("group-based access") && n.contains("log out") && n.contains("login-time")
        }),
        "the agent plan must disclose the supplementary-groups residual: {:?}",
        plan.notes
    );
}

// ── the Linux scope split (linger is the --system opt-in) ─────────────────────

#[test]
fn a_linger_enabled_host_notes_loginctl_but_the_login_bound_default_never_runs_it() {
    // Reproduces GitHub's Linux runner, where linger is ALREADY enabled for the
    // user. The login-bound systemd default then surfaces linger_note() — a heads-up
    // that MENTIONS `loginctl disable-linger` — while the plan's COMMANDS still run NO
    // loginctl. This is exactly why the CLI dry-run integration test must match the
    // command (`would run: … loginctl`), not the bare word: on such a host the note
    // legitimately contains "loginctl".
    let plan = plan_systemd_user(
        "runner",
        p("/home/runner"),
        p("/home/runner/.higgs"),
        p("/home/runner/.higgs"),
        None,
        &[],
        ServiceScope::LoginBound,
    );
    assert!(
        !plan
            .commands
            .iter()
            .any(|c| c.argv.first().map(String::as_str) == Some("loginctl")),
        "the login-bound default must never RUN loginctl: {:?}",
        plan.commands
    );
    // On a linger-enabled host the heads-up note DOES mention loginctl — a note, not a command.
    let linger_dir = tempfile::tempdir().unwrap();
    std::fs::write(linger_dir.path().join("runner"), b"").unwrap();
    let note = linger_note(linger_dir.path(), "runner").expect("linger note when the flag exists");
    assert!(
        note.contains("loginctl"),
        "the linger heads-up mentions loginctl in a NOTE (not a run command): {note}"
    );
}

#[test]
fn systemd_login_bound_default_never_calls_linger() {
    // DEFAULT (LoginBound): write + enable + daemon-reload + restart — and
    // NOTHING else. No loginctl call means zero possible polkit/sudo prompts.
    let plan = plan_systemd_user(
        "bob",
        p("/home/bob"),
        p("/home/bob/.higgs"),
        p("/home/bob/.higgs"),
        None,
        &[],
        ServiceScope::LoginBound,
    );
    assert!(
        !plan
            .commands
            .iter()
            .any(|c| c.argv.first().map(String::as_str) == Some("loginctl")),
        "LoginBound must not touch loginctl: {:?}",
        plan.commands
    );
    // The trade-off is stated, with the --system escape hatch.
    assert!(
        plan.notes
            .iter()
            .any(|n| n.contains("login-bound") && n.contains("--system")),
        "notes must state the login-bound trade-off: {:?}",
        plan.notes
    );

    // --system (SurvivesLogout): the SAME commands plus best-effort linger.
    let plan = plan_systemd_user(
        "bob",
        p("/home/bob"),
        p("/home/bob/.higgs"),
        p("/home/bob/.higgs"),
        None,
        &[],
        ServiceScope::SurvivesLogout,
    );
    let linger = plan
        .commands
        .iter()
        .find(|c| c.argv.first().map(String::as_str) == Some("loginctl"))
        .expect("SurvivesLogout adds enable-linger");
    assert_eq!(linger.argv[1], "enable-linger");
    assert!(linger.best_effort);
}

// ── model scan root (HIGGS_MODEL_DIR) handoff ─────────────────────────────────

#[test]
fn services_pin_the_extra_model_scan_root_when_present() {
    // A node paired with HIGGS_MODEL_DIR=/models (the README's documented knob)
    // must not reboot into a service that scans only the defaults and
    // advertises no models — both renderers pin it, escaped, when present.
    let plist = launchd_plist(
        Some("alice"),
        p("/Users/alice"),
        p("/Users/alice/.higgs"),
        p("/Users/alice/.higgs"),
        Some(p("/srv/mo&dels")),
        &[],
    );
    assert!(
        plist.contains("<key>HIGGS_MODEL_DIR</key>\n        <string>/srv/mo&amp;dels</string>"),
        "plist must pin the model dir, XML-escaped: {plist}"
    );

    let unit = systemd_unit(
        p("/home/bob"),
        p("/home/bob/.higgs"),
        p("/home/bob/.higgs"),
        Some(p("/srv/mo dels/%h")),
        &[],
    );
    assert!(
        unit.contains("Environment=\"HIGGS_MODEL_DIR=/srv/mo dels/%%h\"\n"),
        "unit must pin the model dir with Environment= escaping: {unit}"
    );

    // The plan surfaces it as a note (after `state:`) for --dry-run inspection.
    let plan = plan_install(
        "alice",
        p("/Users/alice"),
        501,
        p("/Users/alice/.higgs"),
        p("/Users/alice/.higgs"),
        Some(p("/srv/models")),
        &[],
        ServiceScope::LoginBound,
    );
    let state_at = plan
        .notes
        .iter()
        .position(|n| n.starts_with("state:"))
        .expect("state note");
    assert_eq!(
        plan.notes.get(state_at + 1).map(String::as_str),
        Some("models: HIGGS_MODEL_DIR=/srv/models"),
        "notes: {:?}",
        plan.notes
    );
}

#[test]
fn services_omit_the_model_scan_root_when_unset() {
    // No override → no HIGGS_MODEL_DIR anywhere (the runtime treats absent and
    // empty alike, so pinning an empty value would be noise).
    let plist = launchd_plist(
        Some("alice"),
        p("/Users/alice"),
        p("/Users/alice/.higgs"),
        p("/Users/alice/.higgs"),
        None,
        &[],
    );
    assert!(!plist.contains("HIGGS_MODEL_DIR"), "plist: {plist}");
    let unit = systemd_unit(
        p("/home/bob"),
        p("/home/bob/.higgs"),
        p("/home/bob/.higgs"),
        None,
        &[],
    );
    // Not PINNED (no Environment= line) — it now appears in UnsetEnvironment= so a
    // stale manager value can't leak, which is correct.
    assert!(
        !unit.contains("Environment=\"HIGGS_MODEL_DIR="),
        "unit: {unit}"
    );
    let plan = plan_install(
        "x",
        p("/tmp/h"),
        501,
        p("/tmp/h/.higgs"),
        p("/tmp/h/.higgs"),
        None,
        &[],
        ServiceScope::LoginBound,
    );
    assert!(!plan.notes.iter().any(|n| n.starts_with("models:")));
}

// ── cross-scope switching (agent ⇄ daemon) ────────────────────────────────────

#[test]
fn daemon_plan_tears_down_a_leftover_agent() {
    // Switching LoginBound → --system must not leave the agent running: launchd
    // allows the same label in gui AND system domains, so without teardown TWO
    // nodes would run. Root can clean the operator's agent — the daemon plan
    // boots it out of gui/<uid> and removes its plist (best-effort no-ops on a
    // fresh machine), BEFORE the system bootstrap.
    let plan = plan_launchd(
        "alice",
        p("/Users/alice"),
        501,
        p("/Users/alice/.higgs"),
        p("/Users/alice/.higgs"),
        None,
        &[],
    );
    let argvs: Vec<Vec<&str>> = plan
        .commands
        .iter()
        .map(|c| c.argv.iter().map(String::as_str).collect())
        .collect();
    let gui_bootout = argvs
        .iter()
        .position(|a| a == &["launchctl", "bootout", "gui/501/com.higgs.node"])
        .expect("daemon plan must boot the agent out of the gui domain");
    let rm_agent = argvs
        .iter()
        .position(|a| {
            a == &[
                "rm",
                "-f",
                "/Users/alice/Library/LaunchAgents/com.higgs.node.plist",
            ]
        })
        .expect("daemon plan must remove the agent plist");
    let bootstrap = argvs
        .iter()
        .position(|a| a.first() == Some(&"launchctl") && a.get(1) == Some(&"bootstrap"))
        .expect("bootstrap present");
    assert!(
        gui_bootout < bootstrap && rm_agent < bootstrap,
        "agent teardown must precede the system bootstrap: {argvs:?}"
    );
    // The gui bootout stays best-effort (a headless Mac has no gui session to
    // boot out of); the agent-plist rm is REQUIRED, so an un-removable (immutable
    // / ACL) plist ABORTS before bootstrap rather than silently reloading the
    // agent at the next login (`rm -f` still exits 0 on a fresh machine's absent
    // plist, so requiring it only bites a genuinely un-removable one).
    assert!(plan.commands[gui_bootout].best_effort);
    assert!(!plan.commands[rm_agent].best_effort);
    // The rm traverses the OPERATOR's home — it must run with the operator's
    // dropped credentials, never as root (an ancestor symlink could otherwise
    // redirect a root rm onto any file named com.higgs.node.plist).
    assert!(
        plan.commands[rm_agent].as_operator,
        "agent-plist rm must drop to the operator"
    );
    assert!(!plan.commands[gui_bootout].as_operator);
    // A skipped teardown is disclosed with the manual remedy.
    assert!(
        plan.notes
            .iter()
            .any(|n| n.contains("skipped:") && n.contains("finish the switch by hand")),
        "notes must disclose a skipped teardown: {:?}",
        plan.notes
    );
    // The reverse switch is documented for the operator.
    assert!(
        plan.notes
            .iter()
            .any(|n| n.contains("switch back to login-bound")),
        "notes: {:?}",
        plan.notes
    );
}

#[test]
fn agent_plist_path_is_the_single_source_for_all_uses() {
    // The daemon's rm target, the agent's unit_path, and the CLI's conflict
    // check must all agree — one helper defines the path.
    assert_eq!(
        agent_plist_path(p("/Users/alice")),
        Path::new("/Users/alice/Library/LaunchAgents/com.higgs.node.plist")
    );
}

// ── preserved deployment-config env (PRESERVED_ENV / --env) ───────────────────

#[test]
fn services_pin_preserved_config_env_escaped() {
    // Deployment config the daemon reads (HIGGS_HF_ENDPOINT, HIGGS_ENGINE) must
    // survive into the service, or an enterprise node reboots hitting public HF.
    let env = vec![
        (
            "HIGGS_HF_ENDPOINT".to_string(),
            "https://hf.corp/a b".to_string(),
        ),
        ("HIGGS_ENGINE".to_string(), "llamacpp".to_string()),
    ];
    // launchd: XML-escaped dict entries after HIGGS_HOME.
    let plist = launchd_plist(
        Some("alice"),
        p("/Users/alice"),
        p("/Users/alice/.higgs"),
        p("/Users/alice/.higgs"),
        None,
        &env,
    );
    assert!(plist
        .contains("<key>HIGGS_HF_ENDPOINT</key>\n        <string>https://hf.corp/a b</string>"));
    assert!(plist.contains("<key>HIGGS_ENGINE</key>\n        <string>llamacpp</string>"));

    // systemd: one quoted Environment= line each, env-escaped (% doubled).
    let env2 = vec![("HIGGS_HF_ENDPOINT".to_string(), "https://hf/%h".to_string())];
    let unit = systemd_unit(
        p("/home/bob"),
        p("/home/bob/.higgs"),
        p("/home/bob/.higgs"),
        None,
        &env2,
    );
    assert!(
        unit.contains("Environment=\"HIGGS_HF_ENDPOINT=https://hf/%%h\"\n"),
        "unit: {unit}"
    );

    // The plan surfaces each as a `config:` note for --dry-run inspection.
    let plan = plan_install(
        "alice",
        p("/Users/alice"),
        501,
        p("/Users/alice/.higgs"),
        p("/Users/alice/.higgs"),
        None,
        &env,
        ServiceScope::LoginBound,
    );
    assert!(plan
        .notes
        .iter()
        .any(|n| n == "config: HIGGS_HF_ENDPOINT=https://hf.corp/a b"));
    assert!(plan
        .notes
        .iter()
        .any(|n| n == "config: HIGGS_ENGINE=llamacpp"));
}

#[test]
fn services_omit_preserved_env_when_none() {
    // No preserved config → no extra env entries anywhere.
    let plist = launchd_plist(
        Some("alice"),
        p("/Users/alice"),
        p("/Users/alice/.higgs"),
        p("/Users/alice/.higgs"),
        None,
        &[],
    );
    assert!(!plist.contains("HIGGS_HF_ENDPOINT") && !plist.contains("HIGGS_ENGINE"));
    let unit = systemd_unit(
        p("/home/bob"),
        p("/home/bob/.higgs"),
        p("/home/bob/.higgs"),
        None,
        &[],
    );
    // Not PINNED (no Environment= lines) — they now ride UnsetEnvironment= instead,
    // so a stale manager value can't leak into the daemon.
    assert!(
        !unit.contains("Environment=\"HIGGS_HF_ENDPOINT=")
            && !unit.contains("Environment=\"HIGGS_ENGINE=")
    );
}

#[test]
fn refuse_daemon_conflict_blocks_a_dual_run_and_names_the_cleanup() {
    // With a --system daemon plist present, the agent install must refuse
    // (launchd would run BOTH: two nodes on one machine) and hand the operator
    // the exact sudo cleanup. Injectable path = the unit-test seam (the real
    // /Library/LaunchDaemons is root-owned and untouchable from a test).
    let dir = tempfile::tempdir().unwrap();
    let plist = dir.path().join("com.higgs.node.plist");

    // No daemon → no conflict.
    assert!(refuse_daemon_conflict(&plist).is_ok());

    // Daemon present → loud refusal with the remedy.
    std::fs::write(&plist, b"<plist/>").unwrap();
    let err = refuse_daemon_conflict(&plist).unwrap_err().to_string();
    assert!(
        err.contains("ALONGSIDE"),
        "must explain the dual-run: {err}"
    );
    assert!(
        err.contains("sudo launchctl bootout system/com.higgs.node") && err.contains("sudo rm"),
        "must name the exact cleanup: {err}"
    );
    // `;` between the cleanup commands, never `&&`: a STALE plist (daemon
    // already unloaded) fails the bootout, and an && chain would then never
    // remove the plist — the operator would loop on this refusal forever.
    assert!(
        err.contains("; sudo rm") && !err.contains("&&"),
        "cleanup must not be &&-chained: {err}"
    );

    // CONSERVATIVE presence (`plist_present`, not `.exists()`): a DANGLING daemon
    // -plist symlink (a planted / half-removed daemon) still refuses — `.exists()`
    // would follow it to the missing target and read absent, letting the agent
    // install proceed beside it.
    std::fs::remove_file(&plist).unwrap();
    std::os::unix::fs::symlink(dir.path().join("gone"), &plist).unwrap();
    assert!(
        refuse_daemon_conflict(&plist).is_err(),
        "a dangling daemon-plist symlink must still refuse (conservative presence)"
    );
}

#[test]
fn refuse_cross_scope_coexistence_blocks_only_when_both_plists_exist() {
    // The DAEMON (re)install auto-switches an agent→daemon by tearing the agent
    // down — safe only for a CLEAN switch (no pre-existing daemon plist). When a
    // daemon plist ALREADY exists AND an agent plist also exists, a reinstall
    // keeps the pre-existing daemon plist (rollback suppressed), so an agent-`rm`
    // failure after the plan's `enable` leaves BOTH loadable (two nodes). Guard it.
    // Injectable paths = the unit-test seam. Uses symlink_metadata, so a DANGLING
    // symlink still counts as present (matching `unit_preexisted`).
    let dir = tempfile::tempdir().unwrap();
    let daemon = dir.path().join("com.higgs.node.plist");
    let agent = dir.path().join("agent.plist");

    // Neither, or only ONE, present → no conflict (a first switch or a plain
    // daemon reinstall must proceed).
    assert!(refuse_cross_scope_coexistence(&daemon, &agent, dir.path()).is_ok());
    std::fs::write(&daemon, b"<plist/>").unwrap();
    assert!(
        refuse_cross_scope_coexistence(&daemon, &agent, dir.path()).is_ok(),
        "a daemon reinstall with NO agent must proceed"
    );
    std::fs::remove_file(&daemon).unwrap();
    std::fs::write(&agent, b"<plist/>").unwrap();
    assert!(
        refuse_cross_scope_coexistence(&daemon, &agent, dir.path()).is_ok(),
        "a clean agent→daemon switch (no pre-existing daemon plist) must proceed"
    );

    // BOTH present → refuse with a remedy naming both cleanup directions.
    std::fs::write(&daemon, b"<plist/>").unwrap();
    let err = refuse_cross_scope_coexistence(&daemon, &agent, dir.path())
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("mixed state") && err.contains("TWO nodes"),
        "must explain the dual-run risk: {err}"
    );
    assert!(
        err.contains("bootout gui/") && err.contains("sudo launchctl bootout system/"),
        "must offer BOTH cleanup directions (keep daemon / keep agent): {err}"
    );

    // A DANGLING daemon-plist symlink still counts as present (planted-symlink /
    // half-removed state) — symlink_metadata, not exists.
    std::fs::remove_file(&daemon).unwrap();
    std::os::unix::fs::symlink(dir.path().join("gone"), &daemon).unwrap();
    assert!(
        refuse_cross_scope_coexistence(&daemon, &agent, dir.path()).is_err(),
        "a dangling daemon-plist symlink beside an agent must still refuse"
    );

    // HOME-UNAVAILABLE (r51): with a real daemon plist and the agent reading ENOENT
    // ONLY because the operator's home (agent_root) is gone, the agent must be read
    // as PRESENT (conservative) so the guard still refuses — not fall open to a
    // two-node install. A GONE agent_root + an absent agent path → still refuse.
    std::fs::remove_file(&daemon).unwrap();
    std::fs::write(&daemon, b"<plist/>").unwrap();
    let gone_root = dir.path().join("vanished-home");
    let agent_under_gone = gone_root.join("Library/LaunchAgents/com.higgs.node.plist");
    assert!(
        refuse_cross_scope_coexistence(&daemon, &agent_under_gone, &gone_root).is_err(),
        "an agent path under a VANISHED home must read present (conservative) and refuse"
    );
    // Sanity: with the SAME absent agent path but a PRESENT root, the leaf is a
    // genuine absence → the guard proceeds (no false refusal on a normal reinstall).
    assert!(
        refuse_cross_scope_coexistence(&daemon, &dir.path().join("no-agent.plist"), dir.path())
            .is_ok(),
        "an absent agent under a PRESENT root is a benign leaf-absence — must proceed"
    );
}

#[test]
fn refuse_cross_scope_coexistence_shell_quotes_the_cleanup_paths() {
    // The cleanup command is copy-pasted into a shell. An operator home with a
    // SPACE (`/Volumes/Node Homes/alice/...`) must stay ONE `rm` argument, else the
    // suggested cleanup silently removes the wrong (or no) file and every retry
    // refuses again. The interpolated paths must be POSIX single-quoted.
    let dir = tempfile::tempdir().unwrap();
    let daemon = dir.path().join("com.higgs.node.plist");
    let agent = dir.path().join("Node Homes").join("agent.plist");
    std::fs::create_dir_all(agent.parent().unwrap()).unwrap();
    std::fs::write(&daemon, b"<plist/>").unwrap();
    std::fs::write(&agent, b"<plist/>").unwrap();
    let err = refuse_cross_scope_coexistence(&daemon, &agent, dir.path())
        .unwrap_err()
        .to_string();
    // The spaced agent path appears single-quoted as one argument (not bare).
    let quoted = format!("'{}'", agent.display());
    assert!(
        err.contains(&quoted),
        "the spaced agent path must be single-quoted in the cleanup command: {err}"
    );
}

#[test]
fn plist_present_reads_indeterminate_metadata_as_present() {
    use std::os::unix::fs::PermissionsExt;
    // Only a DEFINITE NotFound reads as absent; an INDETERMINATE error (EACCES from
    // a non-searchable parent, standing in for EIO) reads as PRESENT — the safe
    // direction so a transient fault can't fall open into the dual-run.
    assert_ne!(
        unsafe { libc::geteuid() },
        0,
        "needs a non-root run (root bypasses the mode)"
    );
    let dir = tempfile::tempdir().unwrap();
    let missing = dir.path().join("nope.plist");
    assert!(
        !plist_present(&missing),
        "a definitely-absent path is not present"
    );
    let present = dir.path().join("here.plist");
    std::fs::write(&present, b"x").unwrap();
    assert!(plist_present(&present), "an existing file is present");
    // A dangling symlink is present (symlink_metadata stats the link itself).
    let dangling = dir.path().join("dangling.plist");
    std::os::unix::fs::symlink(dir.path().join("gone"), &dangling).unwrap();
    assert!(
        plist_present(&dangling),
        "a dangling symlink counts as present"
    );
    // A path under a NON-searchable (0000) dir → EACCES → indeterminate → present.
    let locked = dir.path().join("locked");
    std::fs::create_dir(&locked).unwrap();
    let hidden = locked.join("agent.plist");
    std::fs::write(&hidden, b"x").unwrap();
    std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o000)).unwrap();
    let indeterminate = plist_present(&hidden);
    // Restore so tempdir cleanup can remove it, regardless of the assertion.
    std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o755)).unwrap();
    assert!(
        indeterminate,
        "an indeterminate (EACCES) metadata error must read as PRESENT, not absent"
    );
}

#[test]
fn linger_note_surfaces_only_when_the_flag_file_exists() {
    // LoginBound on a machine where linger is ALREADY enabled would silently
    // still survive logout — the note must say so (with the opt-out) exactly
    // when systemd's per-user flag file exists, and never auto-disable it.
    let dir = tempfile::tempdir().unwrap();
    assert!(
        linger_note(dir.path(), "bob").is_none(),
        "no flag file → no note"
    );
    std::fs::write(dir.path().join("bob"), b"").unwrap();
    let note = linger_note(dir.path(), "bob").expect("flag file → note");
    assert!(
        note.contains("still run after logout") && note.contains("disable-linger bob"),
        "note must state the reality and the opt-out: {note}"
    );
    // Another user's flag is not bob's.
    assert!(linger_note(dir.path(), "alice").is_none());
}

#[test]
fn agent_headless_hint_only_for_the_login_bound_agent() {
    // The headless-Mac remedy applies ONLY to the LaunchAgent (gui/<uid> needs
    // a GUI session); the daemon and systemd unit need no session, so no hint.
    assert!(agent_headless_hint(ServiceKind::LaunchdAgent).contains("--system"));
    assert!(agent_headless_hint(ServiceKind::LaunchdAgent).contains("headless"));
    assert_eq!(agent_headless_hint(ServiceKind::Launchd), "");
    assert_eq!(agent_headless_hint(ServiceKind::SystemdUser), "");
}
