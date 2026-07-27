//! Integration coverage for the REL-P2 install surface: the REAL `higgs`
//! binary's `node install-service --dry-run` (renders the full service plan
//! without touching system state — the same seam a sudo-shy operator uses),
//! and `install.sh` in tarball mode against a throwaway prefix (install →
//! update flip → rollback → tamper refusal), exercising the exact script an
//! operator runs on a fresh machine.

use std::os::unix::fs::MetadataExt;
use std::path::Path;
use std::process::Command;

fn higgs(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_higgs"))
        .args(args)
        .output()
        .expect("spawn higgs")
}

#[test]
fn install_service_dry_run_renders_the_platform_plan() {
    let out = higgs(&["node", "install-service", "--dry-run"]);
    assert!(out.status.success(), "dry-run failed: {out:?}");
    let stdout = String::from_utf8_lossy(&out.stdout);

    // Platform-independent invariants: the daemon runs THROUGH the atomic
    // `current` symlink, and nothing is written in a dry run.
    assert!(stdout.contains("would write "));
    assert!(stdout.contains("bin/current/higgs"));
    assert!(stdout.contains("--node"));

    if cfg!(target_os = "macos") {
        // DEFAULT = the user-space LaunchAgent: plist in the OPERATOR's own
        // ~/Library/LaunchAgents, bootstrapped into their gui session domain —
        // never the system domain, no sudo.
        assert!(stdout.contains("Library/LaunchAgents/com.higgs.node.plist"));
        assert!(
            !stdout.contains("/Library/LaunchDaemons"),
            "default must not touch the system domain: {stdout}"
        );
        assert!(stdout.contains("would run: launchctl bootstrap gui/"));
        // An agent runs as the session user; no UserName pin.
        assert!(!stdout.contains("<key>UserName</key>"));
    } else {
        // DEFAULT = systemd user unit, crash-loop-proof restart policy, and NO
        // linger call (that is the --system opt-in).
        assert!(stdout.contains("higgs-node.service"));
        assert!(stdout.contains("Restart=always"));
        assert!(stdout.contains("StartLimitIntervalSec=0"));
        assert!(
            !stdout.contains("loginctl"),
            "default must not call loginctl: {stdout}"
        );
    }
}

#[test]
fn install_service_system_flag_opts_into_the_always_on_variant() {
    // --system flips the SAME install to the always-on mechanism: macOS = the
    // system LaunchDaemon (root-owned dir, UserName pin); Linux = the same
    // user unit plus best-effort enable-linger.
    let out = higgs(&["node", "install-service", "--dry-run", "--system"]);
    assert!(out.status.success(), "dry-run failed: {out:?}");
    let stdout = String::from_utf8_lossy(&out.stdout);
    if cfg!(target_os = "macos") {
        // The unit is WRITTEN to the system domain...
        assert!(stdout.contains("would write /Library/LaunchDaemons/com.higgs.node.plist"));
        assert!(stdout.contains("<key>UserName</key>"));
        assert!(stdout.contains("would run: launchctl bootstrap system"));
        // ...and the plan TEARS DOWN a leftover login-bound agent (best-effort
        // bootout + plist removal) so switching scopes never runs two nodes.
        assert!(
            stdout.contains("would run: launchctl bootout gui/"),
            "--system must boot a leftover agent out: {stdout}"
        );
        assert!(
            stdout.contains("would run (as the operator): rm -f")
                && stdout.contains("LaunchAgents"),
            "--system must remove the leftover agent plist AS THE OPERATOR: {stdout}"
        );
    } else {
        assert!(stdout.contains("higgs-node.service"));
        assert!(stdout.contains("would run: loginctl enable-linger"));
    }
}

#[test]
fn install_service_dry_run_honors_a_custom_prefix() {
    let out = higgs(&[
        "node",
        "install-service",
        "--dry-run",
        "--prefix",
        "/opt/higgs-elsewhere",
    ]);
    assert!(out.status.success(), "dry-run failed: {out:?}");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("/opt/higgs-elsewhere/bin/current/higgs"));
    assert!(stdout.contains("/opt/higgs-elsewhere/logs/node.log"));
}

#[test]
fn install_service_dry_run_pins_a_custom_higgs_home() {
    // A node paired under a custom HIGGS_HOME must boot against that state dir,
    // not the ~/.higgs default. The install captures the override from the
    // environment and pins it into the daemon env; --dry-run surfaces the exact
    // value so a mismatch is visible before committing.
    let out = Command::new(env!("CARGO_BIN_EXE_higgs"))
        .args(["node", "install-service", "--dry-run"])
        .env("HIGGS_HOME", "/var/lib/higgs-state")
        .output()
        .expect("spawn higgs");
    assert!(out.status.success(), "dry-run failed: {out:?}");
    let stdout = String::from_utf8_lossy(&out.stdout);
    // Surfaced as an operator-facing note...
    assert!(
        stdout.contains("state: HIGGS_HOME=/var/lib/higgs-state"),
        "custom state dir must be surfaced: {stdout}"
    );
    // ...and pinned into the rendered daemon env (plist dict on macOS, unit
    // Environment= on Linux).
    if cfg!(target_os = "macos") {
        assert!(
            stdout.contains("<key>HIGGS_HOME</key>")
                && stdout.contains("<string>/var/lib/higgs-state</string>"),
            "plist must pin the state dir: {stdout}"
        );
    } else {
        assert!(
            stdout.contains("Environment=\"HIGGS_HOME=/var/lib/higgs-state\""),
            "unit must pin the state dir: {stdout}"
        );
    }
}

#[test]
fn install_service_dry_run_pins_a_custom_model_dir() {
    // A node paired with HIGGS_MODEL_DIR (README's documented knob) must not
    // reboot into a service that scans only the defaults — the env override is
    // captured and pinned into the daemon env; unset → omitted entirely.
    let out = Command::new(env!("CARGO_BIN_EXE_higgs"))
        .args(["node", "install-service", "--dry-run"])
        .env("HIGGS_MODEL_DIR", "/srv/models")
        .output()
        .expect("spawn higgs");
    assert!(out.status.success(), "dry-run failed: {out:?}");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("models: HIGGS_MODEL_DIR=/srv/models"),
        "model dir must be surfaced: {stdout}"
    );
    assert!(
        stdout.contains("HIGGS_MODEL_DIR"),
        "model dir must reach the rendered env: {stdout}"
    );

    let out = Command::new(env!("CARGO_BIN_EXE_higgs"))
        .args(["node", "install-service", "--dry-run"])
        .env_remove("HIGGS_MODEL_DIR")
        .output()
        .expect("spawn higgs");
    assert!(out.status.success(), "dry-run failed: {out:?}");
    let stdout = String::from_utf8_lossy(&out.stdout);
    // No override → not PINNED (no `models:` note, no Environment=/plist pin). It
    // now rides the deterministic UNSET list (UnsetEnvironment= / launchctl
    // unsetenv) so a stale inherited value can't leak — which is correct, so we
    // assert only that it is not pinned, not that the name is wholly absent.
    assert!(
        !stdout.contains("models: HIGGS_MODEL_DIR")
            && !stdout.contains("Environment=\"HIGGS_MODEL_DIR=")
            && !stdout.contains("<key>HIGGS_MODEL_DIR</key>"),
        "no override → HIGGS_MODEL_DIR must not be pinned: {stdout}"
    );
}

#[test]
fn install_service_preserves_allowlisted_config_env() {
    // A node configured with HIGGS_HF_ENDPOINT (enterprise mirror) must reboot
    // under the service with it — else it silently hits public HuggingFace.
    // The env override is captured and pinned into the daemon env; a `config:`
    // note surfaces it.
    let out = Command::new(env!("CARGO_BIN_EXE_higgs"))
        .args(["node", "install-service", "--dry-run"])
        .env("HIGGS_HF_ENDPOINT", "https://hf.corp/api")
        .output()
        .expect("spawn higgs");
    assert!(out.status.success(), "dry-run failed: {out:?}");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("config: HIGGS_HF_ENDPOINT=https://hf.corp/api"),
        "config env must be surfaced: {stdout}"
    );
    assert!(
        stdout.contains("HIGGS_HF_ENDPOINT"),
        "config env must reach the rendered daemon env: {stdout}"
    );

    // The --env flag beats the environment and is the sudo-proof carrier.
    let out = Command::new(env!("CARGO_BIN_EXE_higgs"))
        .args([
            "node",
            "install-service",
            "--dry-run",
            "--env",
            "HIGGS_HF_ENDPOINT=https://hf.flag/api",
        ])
        .env("HIGGS_HF_ENDPOINT", "https://hf.env/api")
        .output()
        .expect("spawn higgs");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("config: HIGGS_HF_ENDPOINT=https://hf.flag/api")
            && !stdout.contains("hf.env"),
        "flag must beat env: {stdout}"
    );
}

#[test]
fn install_service_full_flag_surface_renders_together() {
    // Exercise the whole resolution path with EVERY flag at once (dry-run):
    // --prefix + --higgs-home + --model-dir + --env (x2) + --system. This
    // covers the combined extra_env capture, plan threading, and note ordering
    // (state → models → config…) in one pass.
    let out = Command::new(env!("CARGO_BIN_EXE_higgs"))
        .args([
            "node",
            "install-service",
            "--dry-run",
            "--system",
            "--prefix",
            "/opt/hg",
            "--higgs-home",
            "/srv/state",
            "--model-dir",
            "/srv/models",
            "--env",
            "HIGGS_HF_ENDPOINT=https://hf.corp",
            "--env",
            "HIGGS_ENGINE=llamacpp",
        ])
        .output()
        .expect("spawn higgs");
    assert!(out.status.success(), "dry-run failed: {out:?}");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("/opt/hg/bin/current/higgs"));
    assert!(stdout.contains("state: HIGGS_HOME=/srv/state"));
    assert!(stdout.contains("models: HIGGS_MODEL_DIR=/srv/models"));
    assert!(stdout.contains("config: HIGGS_HF_ENDPOINT=https://hf.corp"));
    assert!(stdout.contains("config: HIGGS_ENGINE=llamacpp"));
}

#[test]
fn install_service_captures_config_env_on_the_real_path() {
    // Non-dry-run agent install with a missing binary: exercises the full
    // resolution path (flag parse → operator identity → higgs_home/model_dir/
    // extra_env capture → plan → conflict check → binary preflight) and stops
    // SAFELY at the preflight refusal — no plist written, no agent bootstrapped
    // — while covering the real-path extra_env capture that dry-run skips less.
    let tmp = tempfile::tempdir().unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_higgs"))
        .args([
            "node",
            "install-service",
            "--prefix",
            tmp.path().to_str().unwrap(),
        ])
        .env("HIGGS_HF_ENDPOINT", "https://hf.corp/api")
        .output()
        .expect("spawn higgs");
    assert!(!out.status.success(), "missing binary must refuse");
    // The resolved exec-path walk refuses an unresolvable binary (missing, OR a
    // repoint race) BEFORE any bootstrap, pointing at install.sh.
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("install the binary"),
        "must refuse at the exec-path preflight, before any bootstrap: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn install_service_never_serializes_hf_secrets_or_client_env() {
    // POLICY: the PRESERVED_ENV allowlist is deliberately small (HIGGS_HF_ENDPOINT
    // + HIGGS_ENGINE). The huggingface-hub CRATE's own env — especially the
    // SECRET `HF_TOKEN`/token-path vars — must NEVER be serialized into the
    // world-readable unit. (The node's real HF auth is the token FILE under
    // HOME, which survives the service via the pinned HOME; custom HF-client
    // layouts are a documented residual.) A poisoned HF_* env in the install
    // shell must not leak into the rendered plan.
    let out = Command::new(env!("CARGO_BIN_EXE_higgs"))
        .args(["node", "install-service", "--dry-run"])
        .env("HF_TOKEN", "hf_secretshouldnotleak")
        .env("HF_HOME", "/srv/hf-cache")
        .env("HF_TOKEN_PATH", "/secure/hf.token")
        .output()
        .expect("spawn higgs");
    assert!(out.status.success(), "dry-run failed: {out:?}");
    let stdout = String::from_utf8_lossy(&out.stdout);
    for leak in [
        "hf_secretshouldnotleak",
        "HF_TOKEN",
        "HF_HOME",
        "HF_TOKEN_PATH",
    ] {
        assert!(
            !stdout.contains(leak),
            "HF client/secret env {leak:?} must NOT reach the rendered unit/notes: {stdout}"
        );
    }
    // The one HF var that IS preserved is higgs's own endpoint override.
    let out = Command::new(env!("CARGO_BIN_EXE_higgs"))
        .args(["node", "install-service", "--dry-run"])
        .env("HIGGS_HF_ENDPOINT", "https://hf.corp/api")
        .output()
        .expect("spawn higgs");
    assert!(String::from_utf8_lossy(&out.stdout)
        .contains("config: HIGGS_HF_ENDPOINT=https://hf.corp/api"));
}

#[test]
fn install_service_env_flag_rejects_non_allowlisted_keys() {
    // SECURITY: --env must NEVER let an operator inject arbitrary env (e.g.
    // LD_PRELOAD) into a service that can be root-installed.
    for bad in [
        "LD_PRELOAD=/tmp/evil.so",
        "DYLD_INSERT_LIBRARIES=/tmp/x",
        "HIGGS_HOME=/tmp/x",
        "PATH=/tmp",
    ] {
        let out = higgs(&["node", "install-service", "--dry-run", "--env", bad]);
        assert!(!out.status.success(), "must reject --env {bad}");
        assert!(
            String::from_utf8_lossy(&out.stderr).contains("not a preserved config var"),
            "must name the allowlist rejection for {bad}"
        );
    }
    // Malformed (no `=`) and empty-value forms are rejected too.
    let out = higgs(&[
        "node",
        "install-service",
        "--dry-run",
        "--env",
        "HIGGS_ENGINE",
    ]);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("KEY=VALUE"));
    let out = higgs(&[
        "node",
        "install-service",
        "--dry-run",
        "--env",
        "HIGGS_ENGINE=",
    ]);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("empty value"));
}

#[test]
fn install_service_refuses_an_ambiguous_home_passwd_mismatch() {
    // With no explicit state dir and an invocation HOME that DISAGREES with the
    // passwd home (`su -m` / `sudo -u` shells), there is no way to know which
    // home the node paired under — guessing either way installs a service that
    // restart-loops against the wrong store. The CLI must REFUSE loudly and
    // demand --higgs-home, not guess.
    let out = Command::new(env!("CARGO_BIN_EXE_higgs"))
        .args(["node", "install-service", "--dry-run"])
        .env_remove("HIGGS_HOME")
        .env("HOME", "/srv/pair-home")
        .output()
        .expect("spawn higgs");
    assert!(!out.status.success(), "the mismatch must refuse: {out:?}");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("differs from the passwd home") && stderr.contains("--higgs-home"),
        "refusal must name the mismatch and the remedy: {stderr}"
    );

    // The SAME mismatch with an explicit --higgs-home proceeds (dry-run) and
    // pins exactly the given dir — the explicit flag resolves the ambiguity.
    let out = Command::new(env!("CARGO_BIN_EXE_higgs"))
        .args([
            "node",
            "install-service",
            "--dry-run",
            "--higgs-home",
            "/srv/pair-home/.higgs",
        ])
        .env_remove("HIGGS_HOME")
        .env("HOME", "/srv/pair-home")
        .output()
        .expect("spawn higgs");
    assert!(out.status.success(), "explicit flag must proceed: {out:?}");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("state: HIGGS_HOME=/srv/pair-home/.higgs"));
}

#[test]
fn install_service_refuses_a_set_but_empty_higgs_home() {
    // The runtime treats a SET-but-EMPTY HIGGS_HOME as a (broken) relative
    // path, so the installer can neither honor nor ignore it — refuse loudly.
    let out = Command::new(env!("CARGO_BIN_EXE_higgs"))
        .args(["node", "install-service", "--dry-run"])
        .env("HIGGS_HOME", "")
        .output()
        .expect("spawn higgs");
    assert!(
        !out.status.success(),
        "empty HIGGS_HOME must refuse: {out:?}"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("set but EMPTY"),
        "refusal must explain the empty override: {stderr}"
    );
}

#[test]
fn install_service_higgs_home_flag_beats_the_env_var() {
    // The --higgs-home ARGV flag is the sudo-proof carrier (printed re-runs use
    // it); it must take precedence over an inherited HIGGS_HOME env var, and get
    // the same guards as --prefix (absolutized, flag-swallow rejected).
    let out = Command::new(env!("CARGO_BIN_EXE_higgs"))
        .args([
            "node",
            "install-service",
            "--dry-run",
            "--higgs-home",
            "/opt/state-from-flag",
        ])
        .env("HIGGS_HOME", "/opt/state-from-env")
        .output()
        .expect("spawn higgs");
    assert!(out.status.success(), "dry-run failed: {out:?}");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("state: HIGGS_HOME=/opt/state-from-flag"),
        "flag must beat the env var: {stdout}"
    );
    assert!(
        !stdout.contains("state-from-env"),
        "the env value must not leak into the plan: {stdout}"
    );

    // Value guards match --prefix: a swallowed flag and an empty value error.
    let out = higgs(&["node", "install-service", "--higgs-home", "--dry-run"]);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("a value is missing"));
    let out = higgs(&["node", "install-service", "--higgs-home", ""]);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("must not be empty"));
}

#[test]
fn install_service_rejects_unknown_flags() {
    let out = higgs(&["node", "install-service", "--bogus"]);
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("unknown flag"), "stderr: {stderr}");
}

#[test]
fn install_service_makes_a_relative_prefix_absolute() {
    // A relative prefix would render a relative ExecStart, which systemd
    // rejects (a slash-bearing exec path must be absolute). The CLI must
    // absolutize it before rendering — assert the rendered path is absolute.
    let out = Command::new(env!("CARGO_BIN_EXE_higgs"))
        .args([
            "node",
            "install-service",
            "--dry-run",
            "--prefix",
            "relnode",
        ])
        .current_dir(std::env::temp_dir())
        .output()
        .expect("spawn higgs");
    assert!(out.status.success(), "dry-run failed: {out:?}");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("/relnode/bin/current/higgs"),
        "relative prefix should be absolutized: {stdout}"
    );
    assert!(!stdout.contains("\"relnode/bin"), "must not stay relative");
}

#[test]
fn install_service_rejects_a_dotdot_in_a_dir_flag() {
    // `std::path::absolute` keeps `..`; systemd silently IGNORES a
    // `StandardOutput=append:` path containing `..`, so a `--prefix …/x/../y`
    // would run the daemon but lose its logs. The CLI must refuse it LOUDLY,
    // for every dir flag.
    for flag in ["--prefix", "--higgs-home", "--model-dir"] {
        let out = higgs(&[
            "node",
            "install-service",
            "--dry-run",
            flag,
            "/home/alice/tmp/../.higgs",
        ]);
        assert!(!out.status.success(), "{flag} with `..` must refuse");
        assert!(
            String::from_utf8_lossy(&out.stderr).contains("contains a `..` component"),
            "{flag}: refusal must name the `..` problem"
        );
    }
    // A clean absolute path with no `..` still works.
    let out = higgs(&[
        "node",
        "install-service",
        "--dry-run",
        "--prefix",
        "/opt/higgs",
    ]);
    assert!(
        out.status.success(),
        "a clean prefix must still work: {out:?}"
    );
}

#[test]
fn install_service_rejects_a_dotdot_from_the_environment() {
    // `..` rejection must cover EVERY source, not just flags: a `..` in the
    // HIGGS_HOME / HIGGS_MODEL_DIR ENV (which bypasses the flag parser) must be
    // caught on the resolved value too.
    for (var, val) in [
        ("HIGGS_HOME", "/srv/a/../state"),
        ("HIGGS_MODEL_DIR", "/srv/m/../models"),
    ] {
        let out = Command::new(env!("CARGO_BIN_EXE_higgs"))
            .args(["node", "install-service", "--dry-run"])
            .env(var, val)
            .output()
            .expect("spawn higgs");
        assert!(!out.status.success(), "env {var} with `..` must refuse");
        assert!(
            String::from_utf8_lossy(&out.stderr).contains("`..` component"),
            "{var}: refusal must name the `..` problem"
        );
    }
}

#[test]
fn install_service_rejects_an_empty_prefix() {
    let out = higgs(&["node", "install-service", "--prefix", ""]);
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("must not be empty"), "stderr: {stderr}");
}

#[test]
fn install_service_rejects_a_flag_where_prefix_expects_a_value() {
    // `--prefix --dry-run` must NOT install into a dir named `--dry-run`.
    let out = higgs(&["node", "install-service", "--prefix", "--dry-run"]);
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("a value is missing"), "stderr: {stderr}");
}

#[test]
fn install_service_refuses_when_the_current_binary_is_missing() {
    // A real (non-dry-run) install must verify <prefix>/bin/current/higgs is an
    // executable file BEFORE tearing down a running daemon — a typoed --prefix
    // or an un-installed binary must refuse, not bootout-then-crashloop. The
    // check runs ahead of the root gate, so it is observable without sudo.
    let tmp = tempfile::tempdir().unwrap();
    let out = higgs(&[
        "node",
        "install-service",
        "--prefix",
        tmp.path().to_str().unwrap(),
    ]);
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    // The exec-path resolved walk refuses an unresolvable binary (missing/typoed
    // prefix, OR a repoint race) BEFORE any teardown, pointing at install.sh.
    assert!(
        stderr.contains("install the binary"),
        "must refuse a missing binary before any teardown: {stderr}"
    );
}

/// macOS only: with a valid binary present, a non-root run hits the LaunchDaemon
/// root gate, which refuses BEFORE touching launchd. Its re-run hint must carry
/// THIS binary's path and the resolved `--prefix` (not a bare `higgs` that
/// sudo's secure_path could resolve elsewhere / to the default prefix).
/// (Not run on Linux: there a non-root run would proceed past the gate into a
/// real `systemctl --user` install — an unwanted side effect in a test.)
#[cfg(target_os = "macos")]
#[test]
fn install_service_rerun_hint_carries_exe_and_resolved_prefix() {
    use std::os::unix::fs::PermissionsExt;
    let tmp = tempfile::tempdir().unwrap();
    let curdir = tmp.path().join("bin/current");
    std::fs::create_dir_all(&curdir).unwrap();
    let bin = curdir.join("higgs");
    std::fs::write(&bin, b"#!/bin/sh\nexit 0\n").unwrap();
    std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o755)).unwrap();

    // --system: only the LaunchDaemon variant needs (and refuses without)
    // root — the default agent would install for real, which a test must not.
    // Pass --model-dir and --env too so the hint's carry-along of THOSE flags is
    // exercised (a revert dropping them would silently install a daemon with an
    // empty inventory / wrong HF endpoint).
    let models = tmp.path().join("models");
    std::fs::create_dir_all(&models).unwrap();
    let out = higgs(&[
        "node",
        "install-service",
        "--system",
        "--prefix",
        tmp.path().to_str().unwrap(),
        "--model-dir",
        models.to_str().unwrap(),
        "--env",
        "HIGGS_HF_ENDPOINT=https://hf.corp",
    ]);
    assert!(
        !out.status.success(),
        "non-root macOS --system must refuse (needs root)"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains(&format!("--prefix '{}'", tmp.path().display())),
        "re-run hint must carry the resolved --prefix: {stderr}"
    );
    // The re-run must pin the PREFLIGHTED PREFIX binary (<prefix>/bin/current/higgs),
    // whose full ancestry was just strict-validated — NOT `current_exe()` (this
    // running test binary), which could sit in a group-writable build/share dir a
    // peer swaps before the operator copies the `sudo …` command → ROOT exec.
    let prefix_bin = curdir.join("higgs");
    assert!(
        stderr.contains(&format!("'{}'", prefix_bin.display())),
        "re-run hint must carry the PREFIX binary (<prefix>/bin/current/higgs): {stderr}"
    );
    assert!(
        !stderr.contains(env!("CARGO_BIN_EXE_higgs")),
        "re-run hint must NOT carry the running binary's path (current_exe): {stderr}"
    );
    // The hint must also carry --model-dir and each --env (sudo strips the
    // operator's env; a copy-pasted re-run that dropped them installs a daemon
    // with an empty inventory / public-HuggingFace endpoint).
    assert!(
        stderr.contains(&format!("--model-dir '{}'", models.display())),
        "re-run hint must carry --model-dir: {stderr}"
    );
    assert!(
        stderr.contains("--env 'HIGGS_HF_ENDPOINT=https://hf.corp'"),
        "re-run hint must carry each preserved --env: {stderr}"
    );
    // The hint must preserve the SCOPE: a copy-pasted re-run must not silently
    // downgrade the requested always-on daemon to a login-bound agent. Match
    // the COMMAND form (`' --system` — the flag right after the last quoted
    // dir value), not bare "--system", which also appears in the message prose.
    assert!(
        stderr.contains("' --system"),
        "the re-run COMMAND must carry --system: {stderr}"
    );
    // The hint must also pin the RESOLVED state dir — as the `--higgs-home`
    // ARGV flag, NOT a `HIGGS_HOME=` env prefix: sudo strips the operator's
    // env AND rejects command-line env vars under command-specific/NOSETENV
    // sudoers policies, so only an argument reliably survives the copy-pasted
    // `sudo …` re-run.
    assert!(
        stderr.contains("--higgs-home '"),
        "re-run hint must carry the state dir as an argv flag: {stderr}"
    );
    assert!(
        !stderr.contains("HIGGS_HOME='"),
        "no env-var prefix — sudo may reject it: {stderr}"
    );

    // With a custom HIGGS_HOME set, the hint carries THAT exact value (quoted).
    let out = Command::new(env!("CARGO_BIN_EXE_higgs"))
        .args([
            "node",
            "install-service",
            "--system",
            "--prefix",
            tmp.path().to_str().unwrap(),
        ])
        .env("HIGGS_HOME", "/srv/higgs state")
        .output()
        .expect("spawn higgs");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("--higgs-home '/srv/higgs state'"),
        "re-run hint must carry the custom state dir, shell-quoted: {stderr}"
    );
}

#[test]
fn install_service_rejects_a_short_flag_where_prefix_expects_a_value() {
    // A single-dash flag (`-h`, a mistyped `-bogus`) must also be rejected, not
    // taken as a directory named `-h` — matching install.sh's value guard.
    for bad in ["-h", "-bogus"] {
        let out = higgs(&["node", "install-service", "--prefix", bad]);
        assert!(!out.status.success(), "should reject --prefix {bad}");
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(stderr.contains("a value is missing"), "stderr: {stderr}");
    }
}

/// SECURITY: `install-service` must never resolve a privileged helper through
/// the ambient PATH. It once ran `systemctl` during discovery — BEFORE the root
/// gate and even in `--dry-run` — so a planted `systemctl` on an operator's PATH
/// could run (as root, under sudo). Discovery is gone (the systemd plan links an
/// absolute unit), so a dry run must spawn NO `systemctl` at all. Plant a fake
/// one that drops a sentinel and assert the sentinel never appears.
#[test]
fn install_service_dry_run_spawns_no_ambient_path_systemctl() {
    use std::os::unix::fs::PermissionsExt;
    let tmp = tempfile::tempdir().unwrap();
    let fakebin = tmp.path().join("bin");
    std::fs::create_dir_all(&fakebin).unwrap();
    let sentinel = tmp.path().join("PLANTED_RAN");
    // A fake `systemctl` (and `launchctl`) that would touch the sentinel if run.
    for tool in ["systemctl", "launchctl"] {
        let p = fakebin.join(tool);
        std::fs::write(&p, format!("#!/bin/sh\ntouch '{}'\n", sentinel.display())).unwrap();
        std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    let orig_path = std::env::var("PATH").unwrap_or_default();
    let out = Command::new(env!("CARGO_BIN_EXE_higgs"))
        .args(["node", "install-service", "--dry-run"])
        .env("PATH", format!("{}:{}", fakebin.display(), orig_path))
        .output()
        .expect("spawn higgs");
    assert!(out.status.success(), "dry-run should succeed: {out:?}");
    assert!(
        !sentinel.exists(),
        "a dry run must not execute an ambient-PATH systemctl/launchctl"
    );
}

// ── install.sh (tarball mode, isolated prefix) ─────────────────────────────────

/// This target's artifact suffix, matching install.sh's uname detection.
fn suffix() -> &'static str {
    if cfg!(target_os = "macos") {
        "aarch64-apple-darwin"
    } else {
        "x86_64-unknown-linux-gnu"
    }
}

/// Builds a fake-but-wellformed release artifact set (executable `higgs`
/// stub, tarball, canonical .sha256) for `version` under `dir`.
fn stage_artifact(dir: &Path, version: &str) -> std::path::PathBuf {
    let name = format!("higgs-v{version}-{}", suffix());
    let pkg = dir.join(format!("pkg-{version}"));
    std::fs::create_dir_all(&pkg).unwrap();
    std::fs::write(
        pkg.join("higgs"),
        format!("#!/bin/sh\n[ \"$1\" = --version ] && echo \"higgs {version}\"\nexit 0\n"),
    )
    .unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(pkg.join("higgs"), std::fs::Permissions::from_mode(0o755))
            .unwrap();
    }
    let tar = dir.join(format!("{name}.tar.gz"));
    let st = Command::new("tar")
        .args([
            "-C",
            pkg.to_str().unwrap(),
            "-czf",
            tar.to_str().unwrap(),
            "higgs",
        ])
        .status()
        .unwrap();
    assert!(st.success());
    let sha_out = Command::new("shasum")
        .args(["-a", "256", &format!("{name}.tar.gz")])
        .current_dir(dir)
        .output()
        .unwrap();
    assert!(sha_out.status.success());
    std::fs::write(dir.join(format!("{name}.tar.gz.sha256")), &sha_out.stdout).unwrap();
    tar
}

fn run_install(tarball: &Path, prefix: &Path) -> std::process::Output {
    install_cmd()
        .arg("--tarball")
        .arg(tarball)
        .arg("--prefix")
        .arg(prefix)
        .output()
        .expect("spawn install.sh")
}

fn install_cmd() -> Command {
    let mut c = Command::new("bash");
    c.arg(concat!(env!("CARGO_MANIFEST_DIR"), "/install.sh"));
    c
}

#[test]
fn install_sh_clears_a_pending_self_update_trial() {
    // A manual (re)install supersedes any pending node self-update trial — even a
    // same-version repair — so the P3 boot-guard markers must be cleared, else the next
    // node boot would roll the freshly-installed binary back on a stale spent trial.
    let tmp = tempfile::tempdir().unwrap();
    let stage = tmp.path().join("stage");
    std::fs::create_dir_all(&stage).unwrap();
    let prefix = tmp.path().join("prefix");
    let t = stage_artifact(&stage, "0.0.1");
    assert!(run_install(&t, &prefix).status.success());
    let bindir = prefix.join("bin");
    std::fs::write(
        bindir.join(".update-trial"),
        br#"{"to":"v0.0.1","prev":"v0.0.0"}"#,
    )
    .unwrap();
    std::fs::write(bindir.join(".update-bootfails"), b"3").unwrap();
    // A rolled-back-version poison list too — a deliberate reinstall clears it so a fixed
    // rebuild of a formerly-crash-looping version can be applied again. Use a stray DIRECTORY
    // (not a file) so this also proves install.sh uses `rm -rf`, not `rm -f` (which fails on a
    // dir and would wedge the poison mechanism).
    std::fs::create_dir_all(bindir.join(".update-failed").join("junk")).unwrap();
    // Reinstall the same version (a repair).
    assert!(run_install(&t, &prefix).status.success());
    assert!(
        !bindir.join(".update-trial").exists(),
        "trial marker cleared by install"
    );
    assert!(
        !bindir.join(".update-bootfails").exists(),
        "boot-fail counter cleared by install"
    );
    assert!(
        !bindir.join(".update-failed").exists(),
        "rolled-back-version poison list cleared by install"
    );
}

#[test]
fn install_sh_drops_the_variant_marker() {
    // install.sh records the build variant into `v<ver>/.variant` so the node
    // self-updater can read the INSTALLED variant of `current` and refuse an update
    // that would silently switch it (DESIGN-remote §9 P3).
    let tmp = tempfile::tempdir().unwrap();
    let stage = tmp.path().join("stage");
    std::fs::create_dir_all(&stage).unwrap();
    let prefix = tmp.path().join("prefix");
    let t = stage_artifact(&stage, "0.0.1");
    let out = run_install(&t, &prefix);
    assert!(out.status.success(), "install failed: {out:?}");
    let marker = prefix.join("bin/v0.0.1/.variant");
    let v = std::fs::read_to_string(&marker).expect("variant marker written by install.sh");
    assert!(
        ["metal", "cpu", "cuda"].contains(&v.trim()),
        "variant marker must be a known acceleration variant, got {v:?}"
    );
}

#[test]
fn install_sh_next_steps_always_pin_the_state_dir_as_a_flag() {
    // The printed service command ALWAYS carries the state dir as the
    // sudo-proof `--higgs-home` ARGV flag (%q quoted): HIGGS_HOME when set,
    // else THIS shell's $HOME/.higgs (the runtime's own derivation) — sudo
    // strips env vars, so a copy-pasted macOS command would otherwise pin the
    // passwd-home default, which diverges from the pairing state whenever HOME
    // is overridden.
    let tmp = tempfile::tempdir().unwrap();
    let stage = tmp.path().join("stage");
    std::fs::create_dir_all(&stage).unwrap();
    let prefix = tmp.path().join("prefix");
    let t = stage_artifact(&stage, "0.0.1");

    let out = install_cmd()
        .arg("--tarball")
        .arg(&t)
        .arg("--prefix")
        .arg(&prefix)
        .env("HIGGS_HOME", "/srv/higgs state")
        .output()
        .expect("spawn install.sh");
    assert!(out.status.success(), "install failed: {out:?}");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("--higgs-home /srv/higgs\\ state"),
        "Next-steps must pin the custom state dir as a %q-quoted flag: {stdout}"
    );

    // WITHOUT the env: the flag pins $HOME/.higgs (matches where pairing in
    // this shell would have written state). WITH a HIGGS_MODEL_DIR: the model
    // scan root rides along as its own flag (%q-quoted); absent otherwise.
    let out = install_cmd()
        .arg("--tarball")
        .arg(&t)
        .arg("--prefix")
        .arg(&prefix)
        .env_remove("HIGGS_HOME")
        .env("HOME", "/srv/pair home")
        .env("HIGGS_MODEL_DIR", "/srv/mo dels")
        .output()
        .expect("spawn install.sh");
    assert!(out.status.success(), "reinstall failed: {out:?}");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("--higgs-home /srv/pair\\ home/.higgs"),
        "no HIGGS_HOME env → the flag pins $HOME/.higgs: {stdout}"
    );
    assert!(
        stdout.contains("--model-dir /srv/mo\\ dels"),
        "HIGGS_MODEL_DIR must ride along as a %q-quoted flag: {stdout}"
    );

    // Without the model-dir env, the flag is absent.
    let out = run_install(&t, &prefix);
    assert!(out.status.success(), "reinstall failed: {out:?}");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        !stdout.contains("--model-dir"),
        "no HIGGS_MODEL_DIR env → no flag: {stdout}"
    );

    // A RELATIVE HIGGS_HOME/HIGGS_MODEL_DIR is absolutized against the pairing
    // cwd before printing — else the copy-pasted flag, re-absolutized by the
    // CLI against a DIFFERENT cwd, would pin a different store. Run install.sh
    // with cwd = a known dir and a relative override; the printed flag must be
    // absolute (rooted at that cwd), not the bare relative value.
    let cwd = tmp.path().join("pairingcwd");
    std::fs::create_dir_all(&cwd).unwrap();
    let out = install_cmd()
        .current_dir(&cwd)
        .arg("--tarball")
        .arg(&t)
        .arg("--prefix")
        .arg(&prefix)
        .env("HIGGS_HOME", "relstate")
        .env("HIGGS_MODEL_DIR", "relmodels")
        .output()
        .expect("spawn install.sh");
    assert!(out.status.success(), "install failed: {out:?}");
    let stdout = String::from_utf8_lossy(&out.stdout);
    // Robust against /var↔/private/var: assert the flag is ABSOLUTE and ends in
    // the relative name, and never the bare relative value (which the CLI would
    // re-root at its own cwd).
    let hh = stdout
        .split("--higgs-home ")
        .nth(1)
        .and_then(|s| s.split_whitespace().next())
        .expect("--higgs-home present");
    assert!(
        hh.starts_with('/') && hh.ends_with("/relstate"),
        "relative HIGGS_HOME must be absolutized against the pairing cwd, got {hh:?}"
    );
    assert!(
        !stdout.contains("--higgs-home relstate"),
        "must not print the bare relative value: {stdout}"
    );
    let md = stdout
        .split("--model-dir ")
        .nth(1)
        .and_then(|s| s.split_whitespace().next())
        .expect("--model-dir present");
    assert!(
        md.starts_with('/') && md.ends_with("/relmodels"),
        "relative HIGGS_MODEL_DIR must be absolutized, got {md:?}"
    );

    // A preserved deployment-config var (HIGGS_HF_ENDPOINT) rides along as a
    // sudo-proof `--env` flag; absent when unset.
    let out = install_cmd()
        .arg("--tarball")
        .arg(&t)
        .arg("--prefix")
        .arg(&prefix)
        .env("HIGGS_HF_ENDPOINT", "https://hf.corp/x")
        .output()
        .expect("spawn install.sh");
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("--env HIGGS_HF_ENDPOINT=https://hf.corp/x"),
        "HIGGS_HF_ENDPOINT must ride along as a --env flag: {stdout}"
    );
    let out = run_install(&t, &prefix);
    assert!(out.status.success());
    assert!(
        !String::from_utf8_lossy(&out.stdout).contains("--env"),
        "no config env → no --env flag"
    );
}

#[test]
fn install_sh_refuses_a_version_that_conflicts_with_the_tarball() {
    // `--tarball` is authoritative, but SILENTLY overriding an explicit `--version`
    // would let automation pinning `--version 0.0.2` be handed a validly-signed
    // OLDER `higgs-v0.0.1-…` tarball and perform a downgrade (the 0.0.1 manifest
    // verifies fine against 0.0.1). A mismatch must refuse; a match is fine.
    let tmp = tempfile::tempdir().unwrap();
    let stage = tmp.path().join("stage");
    std::fs::create_dir_all(&stage).unwrap();
    let t = stage_artifact(&stage, "0.0.1"); // the tarball is v0.0.1

    let out = install_cmd()
        .arg("--tarball")
        .arg(&t)
        .arg("--version")
        .arg("0.0.2") // pinned a DIFFERENT (newer) version
        .arg("--prefix")
        .arg(tmp.path().join("prefix"))
        .output()
        .expect("spawn install.sh");
    assert!(
        !out.status.success(),
        "a --version conflicting with the tarball must refuse (no silent downgrade)"
    );
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("conflicts with tarball version"),
        "refusal must name the conflict: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    // A MATCHING --version installs fine.
    let out = install_cmd()
        .arg("--tarball")
        .arg(&t)
        .arg("--version")
        .arg("0.0.1")
        .arg("--prefix")
        .arg(tmp.path().join("prefix-ok"))
        .output()
        .expect("spawn install.sh");
    assert!(
        out.status.success(),
        "a --version MATCHING the tarball must succeed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn install_sh_refuses_empty_home_but_ignores_empty_model_dir() {
    // A set-but-empty HIGGS_HOME is broken at the runtime level (resolved as a
    // cwd-relative path); install.sh must REFUSE rather than silently substitute
    // $HOME/.higgs into the Next-steps command (→ empty-store restart loop). But
    // HIGGS_MODEL_DIR is an OPTIONAL extra scan root — the runtime AND
    // install-service treat empty as UNSET, so install.sh must IGNORE an empty
    // value (matching them), not block a provisioning env that exports it blank.
    let tmp = tempfile::tempdir().unwrap();
    let stage = tmp.path().join("stage");
    std::fs::create_dir_all(&stage).unwrap();
    let t = stage_artifact(&stage, "0.0.1");

    // HIGGS_HOME="" → refused, naming the var.
    let out = install_cmd()
        .arg("--tarball")
        .arg(&t)
        .arg("--prefix")
        .arg(tmp.path().join("prefix-home"))
        .env("HIGGS_HOME", "")
        .output()
        .expect("spawn install.sh");
    assert!(!out.status.success(), "empty HIGGS_HOME must refuse");
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("HIGGS_HOME is set but EMPTY"),
        "refusal must name HIGGS_HOME"
    );

    // HIGGS_MODEL_DIR="" → IGNORED; the install SUCCEEDS.
    let out = install_cmd()
        .arg("--tarball")
        .arg(&t)
        .arg("--prefix")
        .arg(tmp.path().join("prefix-model"))
        .env("HIGGS_MODEL_DIR", "")
        .output()
        .expect("spawn install.sh");
    assert!(
        out.status.success(),
        "empty HIGGS_MODEL_DIR must be IGNORED (treated as unset), not refused: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn install_sh_pins_system_tools_against_a_planted_helper() {
    // The PATH prepend must make a system helper (`mkdir`) resolve to the
    // trusted /usr/bin copy, NOT a planted one earlier on PATH — closing the
    // planted-helper vector (which also gated the /proc/environ PAT read).
    use std::os::unix::fs::PermissionsExt;
    let tmp = tempfile::tempdir().unwrap();
    let stage = tmp.path().join("stage");
    std::fs::create_dir_all(&stage).unwrap();
    let t = stage_artifact(&stage, "0.0.1");
    let fakebin = tmp.path().join("fakebin");
    std::fs::create_dir_all(&fakebin).unwrap();
    let ran = tmp.path().join("PLANTED_MKDIR_RAN");
    // A planted `mkdir` that leaves a sentinel, then delegates so the install
    // still proceeds (proving the test would otherwise be reached).
    std::fs::write(
        fakebin.join("mkdir"),
        format!(
            "#!/bin/sh\ntouch '{}'\nexec /bin/mkdir \"$@\"\n",
            ran.display()
        ),
    )
    .unwrap();
    std::fs::set_permissions(
        fakebin.join("mkdir"),
        std::fs::Permissions::from_mode(0o755),
    )
    .unwrap();
    let orig = std::env::var("PATH").unwrap_or_default();
    let out = install_cmd()
        .arg("--tarball")
        .arg(&t)
        .arg("--prefix")
        .arg(tmp.path().join("prefix"))
        .env("PATH", format!("{}:{}", fakebin.display(), orig))
        .output()
        .expect("spawn install.sh");
    assert!(out.status.success(), "install failed: {out:?}");
    assert!(
        !ran.exists(),
        "the planted mkdir must NOT run — the PATH prepend resolves the trusted /usr/bin/mkdir first"
    );
}

// NOTE: the `xattr` hardening (install.sh: macOS-guard + trusted absolute
// /usr/bin/xattr) is only OBSERVABLE on Linux — where `xattr` is absent from
// the trusted system dirs so a bare `command -v xattr` would fall through the
// PATH tail to a planted copy. On this macOS test host /usr/bin/xattr exists,
// so the r4 PATH-prepend already resolves it first and no planted copy ever
// runs; a fail-on-revert test is therefore not possible here. Covered by
// reasoning + the guard (see install.sh's quarantine-strip block), consistent
// with the task's other Linux-only-path residuals.

#[test]
fn install_sh_absolutizes_a_relative_dir_against_the_physical_cwd() {
    // A relative HIGGS_HOME must be resolved against the PHYSICAL cwd (pwd -P),
    // matching the runtime's getcwd — not bash's logical $PWD. From a symlinked
    // cwd the two differ; the printed flag must carry the physical spelling so
    // repointing/removing the symlink can't later break the service's state dir.
    let tmp = tempfile::tempdir().unwrap();
    let stage = tmp.path().join("stage");
    std::fs::create_dir_all(&stage).unwrap();
    let t = stage_artifact(&stage, "0.0.1");
    let physical = tmp.path().join("physical");
    std::fs::create_dir_all(&physical).unwrap();
    let link = tmp.path().join("link");
    std::os::unix::fs::symlink(&physical, &link).unwrap();
    let out = install_cmd()
        .current_dir(&link) // kernel cwd resolves to `physical`
        .env("PWD", &link) // but bash keeps the logical symlink spelling
        .arg("--tarball")
        .arg(&t)
        .arg("--prefix")
        .arg(tmp.path().join("prefix"))
        .env("HIGGS_HOME", "relstate")
        .output()
        .expect("spawn install.sh");
    assert!(out.status.success(), "install failed: {out:?}");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let hh = stdout
        .split("--higgs-home ")
        .nth(1)
        .and_then(|s| s.split_whitespace().next())
        .expect("--higgs-home present");
    assert!(
        hh.ends_with("/relstate") && !hh.contains("/link/"),
        "relative HIGGS_HOME must resolve against the PHYSICAL cwd (no symlink component): {hh:?}"
    );
}

// NOTE: the SHA-tool hardening (install.sh: trusted absolute
// /usr/bin/sha256sum | /usr/bin/shasum, never a bare `command -v sha256sum`) is
// the ARTIFACT INTEGRITY check — a planted `sha256sum` could otherwise emit a
// signed manifest's hash for a malicious tarball. It is only OBSERVABLE where
// no `sha256sum` sits in a trusted PATH dir (standard macOS ships `shasum`
// only); a fail-on-revert test is not possible on a host that HAS one in
// /usr/sbin:/usr/bin:/sbin:/bin (the r4 PATH-prepend then resolves the trusted
// copy first regardless). Covered by reasoning + the absolute-path pick, and
// the install→tamper lifecycle test below proves the real tool REJECTS a
// tampered artifact. Consistent with the xattr Linux-only residual above.

#[test]
fn install_sh_refuses_a_dotdot_in_a_state_env() {
    // install-service's dir flags reject `..`; install.sh must refuse it in
    // HIGGS_HOME/HIGGS_MODEL_DIR too, else the printed Next-steps command would
    // fail the CLI's own validation. Loud + early, matching the empty refusal.
    let tmp = tempfile::tempdir().unwrap();
    let stage = tmp.path().join("stage");
    std::fs::create_dir_all(&stage).unwrap();
    let t = stage_artifact(&stage, "0.0.1");
    for var in ["HIGGS_HOME", "HIGGS_MODEL_DIR"] {
        let out = install_cmd()
            .arg("--tarball")
            .arg(&t)
            .arg("--prefix")
            .arg(tmp.path().join("prefix"))
            .env(var, "/srv/node/../state")
            .output()
            .expect("spawn install.sh");
        assert!(!out.status.success(), "{var} with `..` must refuse");
        assert!(
            String::from_utf8_lossy(&out.stderr).contains(&format!("{var} contains a '..'")),
            "refusal must name {var}"
        );
    }
}

#[test]
fn install_sh_works_with_home_unset_given_an_explicit_prefix() {
    // A scrubbed provisioning env (HOME unset) must be able to `--help` and to
    // install with an explicit --prefix — HOME is only needed for the DEFAULT
    // prefix. `set -u` must not abort on the `$HOME/.higgs` default expansion.
    let out = Command::new("bash")
        .arg(concat!(env!("CARGO_MANIFEST_DIR"), "/install.sh"))
        .arg("--help")
        .env_remove("HOME")
        .output()
        .expect("spawn install.sh");
    assert!(
        out.status.success(),
        "--help must work with HOME unset: {out:?}"
    );

    let tmp = tempfile::tempdir().unwrap();
    let stage = tmp.path().join("stage");
    std::fs::create_dir_all(&stage).unwrap();
    let t = stage_artifact(&stage, "0.0.1");
    let prefix = tmp.path().join("prefix");
    let out = Command::new("bash")
        .arg(concat!(env!("CARGO_MANIFEST_DIR"), "/install.sh"))
        .args([
            "--tarball",
            t.to_str().unwrap(),
            "--prefix",
            prefix.to_str().unwrap(),
        ])
        .env_remove("HOME")
        .output()
        .expect("spawn install.sh");
    assert!(
        out.status.success(),
        "explicit --prefix must work with HOME unset: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(prefix.join("bin/v0.0.1/higgs").exists());

    // The DEFAULT (no --prefix) with HOME unset must fail with a CLEAR message,
    // not a raw `set -u` abort.
    let out = Command::new("bash")
        .arg(concat!(env!("CARGO_MANIFEST_DIR"), "/install.sh"))
        .args(["--tarball", t.to_str().unwrap()])
        .env_remove("HOME")
        .output()
        .expect("spawn install.sh");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("HOME is unset") && !stderr.contains("unbound variable"),
        "must give a clear error, not a set -u abort: {stderr}"
    );
}

#[test]
fn install_sh_advises_setting_home_instead_of_an_unusable_system_command() {
    // With both HOME and HIGGS_HOME unset (a scrubbed provisioning env, explicit
    // --prefix), NO state dir can be derived. Since r54 the CLI REFUSES an unset HOME
    // for BOTH the login-bound and the --system install (rather than silently pinning
    // the passwd default). So BOTH Next-steps lines must emit the FIX (set HIGGS_HOME)
    // instead of a command the CLI would reject. CROSS-PLATFORM (r58): the always-on
    // guard is no longer gated on `$sys_sudo`, so this holds on Linux (--system there
    // is a non-elevated user unit that STILL refuses an unset HOME) as well as macOS.
    let tmp = tempfile::tempdir().unwrap();
    let stage = tmp.path().join("stage");
    std::fs::create_dir_all(&stage).unwrap();
    let t = stage_artifact(&stage, "0.0.1");
    let prefix = tmp.path().join("prefix");
    let out = Command::new("bash")
        .arg(concat!(env!("CARGO_MANIFEST_DIR"), "/install.sh"))
        .args([
            "--tarball",
            t.to_str().unwrap(),
            "--prefix",
            prefix.to_str().unwrap(),
        ])
        .env_remove("HOME")
        .env_remove("HIGGS_HOME")
        .env_remove("HIGGS_MODEL_DIR")
        .output()
        .expect("spawn install.sh");
    assert!(
        out.status.success(),
        "install still succeeds (login-bound works): {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("always-on: set HIGGS_HOME"),
        "must advise setting HIGGS_HOME rather than print an unusable command: {stdout}"
    );
    // Must NOT print a `sudo … --system` command that the CLI would reject.
    assert!(
        !stdout.contains("always-on: sudo "),
        "must NOT print the unusable sudo --system command with no --higgs-home: {stdout}"
    );
    // r57: the LOGIN-BOUND `service:` line must ALSO advise setting HIGGS_HOME — with
    // HOME unset the CLI now rejects even that command, so printing it is unusable.
    assert!(
        stdout.contains("service: set HIGGS_HOME"),
        "the login-bound service line must also advise setting HIGGS_HOME: {stdout}"
    );
    // With no state dir derivable, NEITHER Next-steps line prints an `install-service`
    // command (both are notes), so the operator is never handed one the CLI rejects.
    assert!(
        !stdout.contains("node install-service"),
        "no Next-steps command may be printed when HOME is unset: {stdout}"
    );
}

#[test]
fn install_sh_scrubs_perl_code_execution_env() {
    // perl honors PERL5OPT/PERL5LIB — install.sh runs perl (the atomic publish
    // rename; `shasum` is also perl), so a poisoned `PERL5OPT=-M<module>` +
    // `PERL5LIB=<dir>` would load and EXECUTE attacker code at the verify/publish
    // step. install.sh must scrub them. (TAR_OPTIONS is scrubbed in the same
    // `unset`; not asserted here because this host's bsdtar ignores it, masking
    // a revert — perl behaves identically everywhere, so it is the host-valid
    // canary for the whole class.)
    use std::os::unix::fs::PermissionsExt;
    let tmp = tempfile::tempdir().unwrap();
    let stage = tmp.path().join("stage");
    std::fs::create_dir_all(&stage).unwrap();
    let t = stage_artifact(&stage, "0.0.1");
    // A perl module that touches a sentinel when LOADED.
    let libdir = tmp.path().join("perllib");
    std::fs::create_dir_all(&libdir).unwrap();
    let pwned = tmp.path().join("PWNED");
    std::fs::write(
        libdir.join("Pwn.pm"),
        format!(
            "package Pwn; BEGIN {{ open(my $f, '>', '{}') or 1; }} 1;\n",
            pwned.display()
        ),
    )
    .unwrap();
    std::fs::set_permissions(
        libdir.join("Pwn.pm"),
        std::fs::Permissions::from_mode(0o644),
    )
    .unwrap();
    let out = install_cmd()
        .arg("--tarball")
        .arg(&t)
        .arg("--prefix")
        .arg(tmp.path().join("prefix"))
        .env("PERL5LIB", &libdir)
        .env("PERL5OPT", "-MPwn")
        .output()
        .expect("spawn install.sh");
    assert!(out.status.success(), "install failed: {out:?}");
    assert!(
        !pwned.exists(),
        "PERL5OPT/PERL5LIB must be scrubbed — perl must not load the injected module"
    );
}

#[test]
fn install_sh_refuses_a_dotdot_in_the_home_fallback() {
    // With no HIGGS_HOME, the state dir falls back to `$HOME/.higgs`. If $HOME
    // itself carries a `..`, the resolved fallback does too — abspath must
    // refuse it (else the printed Next-steps command fails the CLI's `..`
    // check). This is the source the env-var guard alone misses.
    let tmp = tempfile::tempdir().unwrap();
    let stage = tmp.path().join("stage");
    std::fs::create_dir_all(&stage).unwrap();
    let t = stage_artifact(&stage, "0.0.1");
    let prefix = tmp.path().join("prefix");
    let out = install_cmd()
        .arg("--tarball")
        .arg(&t)
        .arg("--prefix")
        .arg(&prefix)
        .env_remove("HIGGS_HOME")
        .env("HOME", "/srv/node/../home")
        .output()
        .expect("spawn install.sh");
    assert!(
        !out.status.success(),
        "a `..` in $HOME's fallback must refuse"
    );
    // Ordering: the refusal must happen BEFORE any publish — the `..` check runs
    // up front, so `current` is never flipped to the "rejected" version.
    assert!(
        !prefix.join("bin/current").exists() && !prefix.join("bin/v0.0.1").exists(),
        "a `..` refusal must not publish or flip current"
    );
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("'..' component"),
        "refusal must name the `..` problem: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn install_sh_resolves_the_prefix_physically() {
    // The prefix is resolved with `cd -P`/`pwd -P` (physical), so a symlinked or
    // `..`-bearing --prefix is pinned to its CANONICAL path — install-time and
    // the service unit agree, and repointing the symlink later can't break it.
    let tmp = tempfile::tempdir().unwrap();
    let stage = tmp.path().join("stage");
    std::fs::create_dir_all(&stage).unwrap();
    let t = stage_artifact(&stage, "0.0.1");
    let physical = tmp.path().join("physical");
    std::fs::create_dir_all(&physical).unwrap();
    let link = tmp.path().join("link");
    std::os::unix::fs::symlink(&physical, &link).unwrap();
    // --prefix names the SYMLINK; the resolved/reported prefix must be physical.
    let out = install_cmd()
        .arg("--tarball")
        .arg(&t)
        .arg("--prefix")
        .arg(&link)
        .output()
        .expect("spawn install.sh");
    assert!(out.status.success(), "install failed: {out:?}");
    let stdout = String::from_utf8_lossy(&out.stdout);
    // The binary went into the physical dir, and Next-steps reports it — no
    // "/link/" symlink component survives.
    assert!(
        physical.join("bin/v0.0.1/higgs").exists(),
        "installed under physical"
    );
    let svc = stdout
        .lines()
        .find(|l| l.contains("node install-service --prefix"))
        .expect("service line");
    assert!(
        !svc.contains("/link/") && !svc.contains("/link "),
        "prefix must be reported physically (no symlink component): {svc}"
    );
}

#[test]
fn install_sh_refuses_a_preexisting_world_writable_prefix() {
    // `umask 022` bounds NEW dirs, but a PRE-EXISTING 0777 prefix is a privesc
    // vector (a local user replaces the binary). install.sh must refuse an
    // OTHER-writable prefix — while allowing a GROUP-writable one (the
    // credential-group prefix feature).
    use std::os::unix::fs::PermissionsExt;
    let tmp = tempfile::tempdir().unwrap();
    let stage = tmp.path().join("stage");
    std::fs::create_dir_all(&stage).unwrap();
    let t = stage_artifact(&stage, "0.0.1");

    // 0777 world-writable prefix → refuse.
    let ww = tmp.path().join("worldwritable");
    std::fs::create_dir_all(&ww).unwrap();
    std::fs::set_permissions(&ww, std::fs::Permissions::from_mode(0o777)).unwrap();
    let out = run_install(&t, &ww);
    assert!(!out.status.success(), "world-writable prefix must refuse");
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("world-writable"),
        "refusal must name the world-write problem: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    // 0775 group-writable prefix → allowed (installs fine).
    let gw = tmp.path().join("groupwritable");
    std::fs::create_dir_all(&gw).unwrap();
    std::fs::set_permissions(&gw, std::fs::Permissions::from_mode(0o775)).unwrap();
    let out = run_install(&t, &gw);
    assert!(
        out.status.success(),
        "group-writable prefix must be allowed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn install_service_checks_the_lexical_bin_chain_not_just_resolved() {
    // An other-writable `bin` lets a peer repoint `current` AFTER canonicalize,
    // so the LEXICAL chain (<prefix>/bin/current/…) must be checked, not only
    // the resolved target. Setup: bin is 0777, `current` dangles (canonicalize
    // fails → the resolved walk is skipped), so ONLY the lexical walk can catch
    // the world-writable bin. It must refuse with the PERMISSION error, not fall
    // through to the missing-binary error.
    use std::os::unix::fs::PermissionsExt;
    let tmp = tempfile::tempdir().unwrap();
    let bin = tmp.path().join("bin");
    std::fs::create_dir_all(&bin).unwrap();
    std::os::unix::fs::symlink("gone", bin.join("current")).unwrap(); // dangling
    std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o777)).unwrap();

    let out = higgs(&[
        "node",
        "install-service",
        "--prefix",
        tmp.path().to_str().unwrap(),
    ]);
    assert!(!out.status.success());
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("OTHER-writable"),
        "the lexical bin chain must be refused for a world-writable bin: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o755)).unwrap();
}

#[test]
fn install_sh_refuses_a_sticky_world_writable_logs_dir() {
    // The logs dir is MANAGED (node.log is created there), so a STICKY
    // world-writable logs (01777) must STILL refuse — sticky only exempts
    // ANCESTORS, not managed leaf dirs.
    use std::os::unix::fs::PermissionsExt;
    let tmp = tempfile::tempdir().unwrap();
    let stage = tmp.path().join("stage");
    std::fs::create_dir_all(&stage).unwrap();
    let t = stage_artifact(&stage, "0.0.1");
    let prefix = tmp.path().join("prefix");
    std::fs::create_dir_all(prefix.join("logs")).unwrap();
    std::fs::set_permissions(prefix.join("logs"), std::fs::Permissions::from_mode(0o1777)).unwrap();
    let out = run_install(&t, &prefix);
    assert!(
        !out.status.success(),
        "a sticky world-writable logs dir must still refuse"
    );
    assert!(String::from_utf8_lossy(&out.stderr).contains("world-writable"));
    std::fs::set_permissions(prefix.join("logs"), std::fs::Permissions::from_mode(0o755)).unwrap();
}

#[test]
fn install_service_refuses_a_world_writable_logs_dir() {
    // A world-writable `logs/` lets a peer pre-plant `node.log` as a symlink to
    // an operator file; the append-open follows it and corrupts that file.
    // install-service must refuse it (the exec path is clean here — only logs).
    use std::os::unix::fs::PermissionsExt;
    let tmp = tempfile::tempdir().unwrap();
    let bin = tmp.path().join("bin");
    let v1 = bin.join("v1");
    std::fs::create_dir_all(&v1).unwrap();
    let exe = v1.join("higgs");
    std::fs::write(&exe, b"#!/bin/sh\nexit 0\n").unwrap();
    std::fs::set_permissions(&exe, std::fs::Permissions::from_mode(0o755)).unwrap();
    std::os::unix::fs::symlink("v1", bin.join("current")).unwrap();
    let logs = tmp.path().join("logs");
    std::fs::create_dir_all(&logs).unwrap();
    std::fs::set_permissions(&logs, std::fs::Permissions::from_mode(0o777)).unwrap();

    let out = higgs(&[
        "node",
        "install-service",
        "--prefix",
        tmp.path().to_str().unwrap(),
    ]);
    assert!(!out.status.success(), "world-writable logs dir must refuse");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("OTHER-writable") && stderr.contains("log dir"),
        "must refuse the logs dir: {stderr}"
    );

    // STICKY world-writable (01777) logs must ALSO refuse: it is a MANAGED dir
    // (node.log is created there), so sticky does NOT help — a peer can still
    // create node.log as a symlink. (Sticky is exempted only for ANCESTORS.)
    std::fs::set_permissions(&logs, std::fs::Permissions::from_mode(0o1777)).unwrap();
    let out = higgs(&[
        "node",
        "install-service",
        "--prefix",
        tmp.path().to_str().unwrap(),
    ]);
    assert!(
        !out.status.success(),
        "a STICKY world-writable logs dir must still refuse (managed dir)"
    );
    std::fs::set_permissions(&logs, std::fs::Permissions::from_mode(0o755)).unwrap();
}

#[test]
fn install_service_refuses_an_other_writable_node_log() {
    // A stale attacker-owned mode-0666 node.log passes the openability probe but
    // lets any local user forge/truncate the daemon's logs or hold it open to fill
    // the state fs. Refuse it in the world-writable preflight. (Exit-3 stub: if
    // the check is reverted, the binary preflight refuses next — never launchd.)
    use std::os::unix::fs::PermissionsExt;
    let tmp = tempfile::tempdir().unwrap();
    let bin = tmp.path().join("bin");
    let v1 = bin.join("v1");
    std::fs::create_dir_all(&v1).unwrap();
    let exe = v1.join("higgs");
    std::fs::write(&exe, b"#!/bin/sh\nexit 3\n").unwrap();
    std::fs::set_permissions(&exe, std::fs::Permissions::from_mode(0o755)).unwrap();
    std::os::unix::fs::symlink("v1", bin.join("current")).unwrap();
    let logs = tmp.path().join("logs");
    std::fs::create_dir_all(&logs).unwrap();
    let node_log = logs.join("node.log");
    std::fs::write(&node_log, b"").unwrap();
    std::fs::set_permissions(&node_log, std::fs::Permissions::from_mode(0o666)).unwrap();

    let out = higgs(&[
        "node",
        "install-service",
        "--prefix",
        tmp.path().to_str().unwrap(),
    ]);
    assert!(
        !out.status.success(),
        "an other-writable node.log must refuse before activation"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("OTHER-writable") && stderr.contains("node.log"),
        "must refuse the world-writable node.log: {stderr}"
    );
    std::fs::set_permissions(&node_log, std::fs::Permissions::from_mode(0o644)).unwrap();
}

#[cfg(target_os = "macos")]
#[test]
fn install_sh_warns_that_system_is_unavailable_for_a_group_writable_prefix() {
    // A --system LaunchDaemon is installed via `sudo <prefix>/bin/current/higgs`
    // (the prefix binary as ROOT). A GROUP-writable prefix would let a group peer
    // swap that binary and run code as root — a check inside install-service is too
    // late. So install.sh must NOT advise the sudo --system command for a
    // group-writable prefix; it warns instead. (Login-bound, no-sudo, is fine.)
    use std::os::unix::fs::PermissionsExt;
    let tmp = tempfile::tempdir().unwrap();
    let stage = tmp.path().join("stage");
    std::fs::create_dir_all(&stage).unwrap();
    let t = stage_artifact(&stage, "0.0.1");
    let prefix = tmp.path().join("prefix");
    std::fs::create_dir_all(&prefix).unwrap();
    std::fs::set_permissions(&prefix, std::fs::Permissions::from_mode(0o775)).unwrap();

    let out = run_install(&t, &prefix);
    assert!(
        out.status.success(),
        "install into a group-writable prefix still succeeds (login-bound is fine): {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("always-on: UNAVAILABLE"),
        "the --system Next-steps must warn for a group-writable prefix: {stdout}"
    );
    assert!(
        !stdout.contains("--system   # survives logout"),
        "must NOT advise the sudo --system command for a group-writable prefix: {stdout}"
    );
    std::fs::set_permissions(&prefix, std::fs::Permissions::from_mode(0o755)).unwrap();
}

#[cfg(target_os = "macos")]
#[test]
fn install_sh_warns_that_system_is_unavailable_for_an_acl_on_the_exec_path() {
    // A macOS ACL granting a peer WRITE (while the mode stays 0755, which the
    // mode-bit checks miss) makes `sudo … --system` of the binary a root vector, so
    // it must suppress the --system advice. A benign `everyone deny delete` (which
    // real homes carry) must NOT — it grants nothing.
    let tmp = tempfile::tempdir().unwrap();
    let stage = tmp.path().join("stage");
    std::fs::create_dir_all(&stage).unwrap();
    let t = stage_artifact(&stage, "0.0.1");
    let prefix = tmp.path().join("prefix");
    std::fs::create_dir_all(&prefix).unwrap();

    // (a) A DENY-only ACL → still advises --system (no write granted).
    assert!(std::process::Command::new("chmod")
        .args(["+a", "everyone deny delete"])
        .arg(&prefix)
        .status()
        .unwrap()
        .success());
    let out = run_install(&t, &prefix);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        String::from_utf8_lossy(&out.stdout).contains("--system   # survives logout"),
        "a benign deny-only ACL must NOT block --system: {}",
        String::from_utf8_lossy(&out.stdout)
    );

    // (b) An `allow write` to the OPERATOR'S OWN user → still advises --system:
    //     the self-exclusion is a LITERAL field comparison (a regex would let a
    //     name like `ops.a` wrongly match another user `opsXa`).
    let me = String::from_utf8_lossy(
        &std::process::Command::new("id")
            .arg("-un")
            .output()
            .unwrap()
            .stdout,
    )
    .trim()
    .to_string();
    let prefix_self = tmp.path().join("prefix-self");
    std::fs::create_dir_all(&prefix_self).unwrap();
    assert!(std::process::Command::new("chmod")
        .args(["+a", &format!("user:{me} allow write")])
        .arg(&prefix_self)
        .status()
        .unwrap()
        .success());
    let out = run_install(&t, &prefix_self);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        String::from_utf8_lossy(&out.stdout).contains("--system   # survives logout"),
        "an allow-write to the OPERATOR'S own user must NOT block --system: {}",
        String::from_utf8_lossy(&out.stdout)
    );

    // (c) An `allow write` to everyone (a non-owner) → suppress --system.
    assert!(std::process::Command::new("chmod")
        .args(["+a", "everyone allow write"])
        .arg(&prefix)
        .status()
        .unwrap()
        .success());
    let out = run_install(&t, &prefix);
    assert!(
        out.status.success(),
        "install still succeeds (login-bound is fine): {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        String::from_utf8_lossy(&out.stdout).contains("always-on: UNAVAILABLE"),
        "a write-granting ACL on the exec path must suppress --system: {}",
        String::from_utf8_lossy(&out.stdout)
    );
}

#[cfg(target_os = "macos")]
#[test]
fn install_sh_warns_that_system_is_unavailable_for_an_inherited_write_acl() {
    // macOS renders a PROPAGATED ACE as `principal INHERITED allow perms` — an extra
    // field before `allow`. A parser expecting `principal allow` misses it entirely.
    // Inherited ACEs PERSIST after their source ACL is removed, so a 0755 prefix can
    // carry an inherited group-write grant with an OTHERWISE-CLEAN ancestry — nothing
    // an old `$3==allow` parser would flag. The exec-path ACL walk must still catch
    // the inherited grant and withhold the root `sudo … --system` advice, else a peer
    // swaps the binary and runs code as root.
    let tmp = tempfile::tempdir().unwrap();
    let stage = tmp.path().join("stage");
    std::fs::create_dir_all(&stage).unwrap();
    let t = stage_artifact(&stage, "0.0.1");
    // Inheritable allow-write on `outer`; `prefix` created under it inherits the ACE;
    // then REMOVE outer's ACL so the ONLY write grant left is the INHERITED one on
    // prefix (and the exec dirs install creates beneath it) — the ancestry is clean.
    let outer = tmp.path().join("outer");
    std::fs::create_dir_all(&outer).unwrap();
    assert!(std::process::Command::new("chmod")
        .args(["+a", "everyone allow write,file_inherit,directory_inherit"])
        .arg(&outer)
        .status()
        .unwrap()
        .success());
    let prefix = outer.join("prefix");
    std::fs::create_dir(&prefix).unwrap(); // inherits the ACE at creation
    assert!(std::process::Command::new("chmod")
        .args(["-N"]) // strip outer's ACL; prefix keeps its inherited copy
        .arg(&outer)
        .status()
        .unwrap()
        .success());

    let out = run_install(&t, &prefix);
    assert!(
        out.status.success(),
        "install still succeeds (login-bound is fine): {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("always-on: UNAVAILABLE"),
        "an INHERITED write ACL on the exec path must suppress --system: {stdout}"
    );
}

#[cfg(target_os = "macos")]
#[test]
fn install_sh_warns_that_system_is_unavailable_for_a_group_writable_ancestor() {
    // Not just the prefix ITSELF: a group-writable ANCESTOR above a 0755 prefix
    // (e.g. `/srv/fleet` group-writable, prefix `/srv/fleet/higgs` 0755) lets a
    // group peer RENAME the whole prefix subtree and swap the binary → root via
    // the printed `sudo … --system`. The exec-path group-write walk must cover the
    // FULL ancestry, not just prefix/bin/verdir.
    use std::os::unix::fs::PermissionsExt;
    let tmp = tempfile::tempdir().unwrap();
    let stage = tmp.path().join("stage");
    std::fs::create_dir_all(&stage).unwrap();
    let t = stage_artifact(&stage, "0.0.1");
    let outer = tmp.path().join("fleet");
    let prefix = outer.join("higgs");
    std::fs::create_dir_all(&prefix).unwrap();
    std::fs::set_permissions(&prefix, std::fs::Permissions::from_mode(0o755)).unwrap();
    std::fs::set_permissions(&outer, std::fs::Permissions::from_mode(0o770)).unwrap(); // group-writable ancestor

    let out = run_install(&t, &prefix);
    assert!(
        out.status.success(),
        "install still succeeds (login-bound is fine): {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("always-on: UNAVAILABLE"),
        "a group-writable ANCESTOR must suppress the --system advice: {stdout}"
    );
    std::fs::set_permissions(&outer, std::fs::Permissions::from_mode(0o755)).unwrap();
}

#[test]
fn install_sh_refuses_a_symlinked_bin_whose_physical_ancestor_is_world_writable() {
    // `bin -> shared/alice-bin` (shared 0777 non-sticky, alice-bin 0755): the
    // LEXICAL prefix walk misses `shared`; the PHYSICAL bindir walk must catch it,
    // else a peer replaces the published binary and the printed install-service
    // command execs attacker code as the operator (before the Rust preflight runs).
    use std::os::unix::fs::PermissionsExt;
    let tmp = tempfile::tempdir().unwrap();
    let stage = tmp.path().join("stage");
    std::fs::create_dir_all(&stage).unwrap();
    let t = stage_artifact(&stage, "0.0.1");
    let prefix = tmp.path().join("prefix");
    std::fs::create_dir_all(&prefix).unwrap();
    let shared = tmp.path().join("shared");
    let real_bin = shared.join("alice-bin");
    std::fs::create_dir_all(&real_bin).unwrap();
    std::fs::set_permissions(&shared, std::fs::Permissions::from_mode(0o777)).unwrap();
    std::os::unix::fs::symlink(&real_bin, prefix.join("bin")).unwrap();

    let out = run_install(&t, &prefix);
    assert!(
        !out.status.success(),
        "a symlinked bin under a world-writable physical dir must refuse"
    );
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("world-writable"),
        "must refuse the physical bin ancestor: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    std::fs::set_permissions(&shared, std::fs::Permissions::from_mode(0o755)).unwrap();
}

#[test]
fn install_sh_warns_when_a_symlinked_bin_hides_a_group_writable_lexical_prefix() {
    // The MIRROR of the physical case above: `bin -> clean-bin` (clean-bin 0755)
    // while the LEXICAL prefix that CONTAINS the `bin` symlink is GROUP-writable. A
    // group peer with write on the prefix can replace the `bin` symlink to point at
    // their own tree, so the printed `sudo … --system` execs their binary as ROOT.
    // A walk that resolves `cd -P` FIRST follows the symlink to the clean target and
    // never sees the group-writable prefix; the exec-path walk must check the
    // LEXICAL chain too (as install-service's Rust preflight does). Login-bound
    // install still succeeds (no sudo), but --system must be withheld.
    use std::os::unix::fs::PermissionsExt;
    let tmp = tempfile::tempdir().unwrap();
    let stage = tmp.path().join("stage");
    std::fs::create_dir_all(&stage).unwrap();
    let t = stage_artifact(&stage, "0.0.1");
    let prefix = tmp.path().join("prefix");
    std::fs::create_dir_all(&prefix).unwrap();
    // A CLEAN 0755 real bin dir the symlink points at.
    let clean_bin = tmp.path().join("clean-bin");
    std::fs::create_dir_all(&clean_bin).unwrap();
    std::fs::set_permissions(&clean_bin, std::fs::Permissions::from_mode(0o755)).unwrap();
    std::os::unix::fs::symlink(&clean_bin, prefix.join("bin")).unwrap();
    // The prefix itself is GROUP-writable — the lexical parent of the `bin` symlink.
    std::fs::set_permissions(&prefix, std::fs::Permissions::from_mode(0o775)).unwrap();

    let out = run_install(&t, &prefix);
    assert!(
        out.status.success(),
        "install still succeeds (login-bound is fine): {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("always-on: UNAVAILABLE"),
        "a group-writable LEXICAL prefix behind a symlinked bin must suppress --system: {stdout}"
    );
    assert!(
        !stdout.contains("always-on: sudo "),
        "must NOT advise the sudo --system command when the lexical prefix is group-writable: {stdout}"
    );
    std::fs::set_permissions(&prefix, std::fs::Permissions::from_mode(0o755)).unwrap();
}

#[test]
fn install_sh_refuses_a_group_writable_or_deep_tmpdir() {
    // The integrity workdir must not be renamable by any other user: reject a
    // GROUP-writable non-sticky $TMPDIR, and a private $TMPDIR under a
    // world-writable physical ANCESTOR. (r27 caught only other-write on the base.)
    use std::os::unix::fs::PermissionsExt;
    let tmp = tempfile::tempdir().unwrap();
    let stage = tmp.path().join("stage");
    std::fs::create_dir_all(&stage).unwrap();
    let t = stage_artifact(&stage, "0.0.1");

    // (a) GROUP-writable non-sticky TMPDIR (0770) → refuse.
    let grp = tmp.path().join("grp-tmp");
    std::fs::create_dir_all(&grp).unwrap();
    std::fs::set_permissions(&grp, std::fs::Permissions::from_mode(0o770)).unwrap();
    let out = install_cmd()
        .arg("--tarball")
        .arg(&t)
        .arg("--prefix")
        .arg(tmp.path().join("p1"))
        .env("TMPDIR", &grp)
        .output()
        .expect("spawn install.sh");
    assert!(
        !out.status.success(),
        "a group-writable non-sticky TMPDIR must refuse"
    );
    assert!(String::from_utf8_lossy(&out.stderr).contains("TMPDIR path"));
    std::fs::set_permissions(&grp, std::fs::Permissions::from_mode(0o755)).unwrap();

    // (b) A private 0700 TMPDIR under a WORLD-writable ANCESTOR → refuse.
    let outer = tmp.path().join("outer");
    let inner = outer.join("inner");
    std::fs::create_dir_all(&inner).unwrap();
    std::fs::set_permissions(&inner, std::fs::Permissions::from_mode(0o700)).unwrap();
    std::fs::set_permissions(&outer, std::fs::Permissions::from_mode(0o777)).unwrap();
    let out = install_cmd()
        .arg("--tarball")
        .arg(&t)
        .arg("--prefix")
        .arg(tmp.path().join("p2"))
        .env("TMPDIR", &inner)
        .output()
        .expect("spawn install.sh");
    assert!(
        !out.status.success(),
        "a private TMPDIR under a world-writable ancestor must refuse"
    );
    assert!(String::from_utf8_lossy(&out.stderr).contains("TMPDIR path"));
    std::fs::set_permissions(&outer, std::fs::Permissions::from_mode(0o755)).unwrap();
}

#[cfg(target_os = "macos")]
#[test]
fn install_sh_refuses_a_tmpdir_with_a_peer_write_acl() {
    // The integrity workdir holds UNVERIFIED downloads. A 0755 TMPDIR (mode bits
    // clean) carrying an ACL that grants a PEER add_file/delete_child lets that peer
    // rename the workdir and swap the tarball + its .sha256 — a no-`--pubkey` install
    // then publishes attacker code. The mode-bit walk misses the ACL; the TMPDIR
    // ancestry walk must ALSO reject a peer-write ACL.
    let tmp = tempfile::tempdir().unwrap();
    let stage = tmp.path().join("stage");
    std::fs::create_dir_all(&stage).unwrap();
    let t = stage_artifact(&stage, "0.0.1");
    let acl_tmp = tmp.path().join("acl-tmp");
    std::fs::create_dir_all(&acl_tmp).unwrap();
    // 0755 mode (clean bits) + an everyone allow-write ACL (the vector).
    assert!(std::process::Command::new("chmod")
        .args(["+a", "everyone allow write"])
        .arg(&acl_tmp)
        .status()
        .unwrap()
        .success());
    let out = install_cmd()
        .arg("--tarball")
        .arg(&t)
        .arg("--prefix")
        .arg(tmp.path().join("p"))
        .env("TMPDIR", &acl_tmp)
        .output()
        .expect("spawn install.sh");
    assert!(
        !out.status.success(),
        "a TMPDIR with a peer-write ACL must refuse: {}",
        String::from_utf8_lossy(&out.stdout)
    );
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("macOS ACL granting write"),
        "must name the ACL vector: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[cfg(target_os = "macos")]
#[test]
fn install_sh_refuses_a_tmpdir_whose_acl_cannot_be_read() {
    // A TMPDIR whose security info cannot be READ (`everyone deny readsecurity`)
    // could HIDE a peer-write grant. It must be REFUSED, not silently accepted. On
    // macOS a `deny readsecurity` also blocks `find -uid`, so the OWNER check ("owned
    // by an untrusted user") is the check that fires first in the install flow — the
    // `_has_write_acl_for_peer` ACL tri-state is defense-in-depth behind it (the
    // PRIMARY reachable inspection-failure fix is the Rust `has_writable_acl`, which
    // has no owner-check pre-empt — see its unit test). Either way, the SAFE OUTCOME
    // is: refuse, never proceed.
    let tmp = tempfile::tempdir().unwrap();
    let stage = tmp.path().join("stage");
    std::fs::create_dir_all(&stage).unwrap();
    let t = stage_artifact(&stage, "0.0.1");
    let unreadable_tmp = tmp.path().join("unreadable-tmp");
    std::fs::create_dir_all(&unreadable_tmp).unwrap();
    assert!(std::process::Command::new("chmod")
        .args(["+a", "everyone deny readsecurity"])
        .arg(&unreadable_tmp)
        .status()
        .unwrap()
        .success());
    let out = install_cmd()
        .arg("--tarball")
        .arg(&t)
        .arg("--prefix")
        .arg(tmp.path().join("p"))
        .env("TMPDIR", &unreadable_tmp)
        .output()
        .expect("spawn install.sh");
    // Reset the ACL so tempdir cleanup is unhindered.
    let _ = std::process::Command::new("chmod")
        .arg("-N")
        .arg(&unreadable_tmp)
        .status();
    assert!(
        !out.status.success(),
        "a TMPDIR whose ACL cannot be read must refuse: {}",
        String::from_utf8_lossy(&out.stdout)
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("untrusted user") || stderr.contains("macOS ACL"),
        "must refuse an unreadable-security TMPDIR (owner check or ACL check): {stderr}"
    );
}

#[test]
fn install_sh_refuses_an_unresolvable_tmpdir_instead_of_falling_back_to_lexical() {
    // A TMPDIR whose `cd -P` fails (a dangling symlink) must DIE, not fall back to
    // the lexical path — a lexical fallback would let a peer make the resolve fail,
    // pass the ancestry walk against a restored target, then repoint the link before
    // `mktemp` lands the integrity workdir in their dir.
    let tmp = tempfile::tempdir().unwrap();
    let stage = tmp.path().join("stage");
    std::fs::create_dir_all(&stage).unwrap();
    let t = stage_artifact(&stage, "0.0.1");
    let dangling = tmp.path().join("dangling-tmpdir");
    std::os::unix::fs::symlink(tmp.path().join("gone-target"), &dangling).unwrap();
    let out = install_cmd()
        .arg("--tarball")
        .arg(&t)
        .arg("--prefix")
        .arg(tmp.path().join("p"))
        .env("TMPDIR", &dangling)
        .output()
        .expect("spawn install.sh");
    assert!(
        !out.status.success(),
        "a dangling-symlink TMPDIR must refuse, not fall back to lexical"
    );
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("could not be resolved"),
        "must refuse an unresolvable TMPDIR: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn install_sh_refuses_a_world_writable_tmpdir() {
    // The integrity workdir lives under $TMPDIR; a world-writable NON-sticky
    // $TMPDIR lets a peer rename the workdir and swap the tarball + sha sidecar
    // under a no-pubkey install. Refuse it; a private base installs fine.
    use std::os::unix::fs::PermissionsExt;
    let tmp = tempfile::tempdir().unwrap();
    let stage = tmp.path().join("stage");
    std::fs::create_dir_all(&stage).unwrap();
    let t = stage_artifact(&stage, "0.0.1");

    // 0777 non-sticky TMPDIR (owned by us, but any user can rename entries) → refuse.
    let hostile = tmp.path().join("hostile-tmp");
    std::fs::create_dir_all(&hostile).unwrap();
    std::fs::set_permissions(&hostile, std::fs::Permissions::from_mode(0o777)).unwrap();
    let out = install_cmd()
        .arg("--tarball")
        .arg(&t)
        .arg("--prefix")
        .arg(tmp.path().join("prefix"))
        .env("TMPDIR", &hostile)
        .output()
        .expect("spawn install.sh");
    assert!(
        !out.status.success(),
        "a world-writable non-sticky TMPDIR must refuse"
    );
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("TMPDIR"),
        "refusal must name TMPDIR: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    std::fs::set_permissions(&hostile, std::fs::Permissions::from_mode(0o755)).unwrap();

    // A private (0700) TMPDIR installs fine.
    let priv_tmp = tmp.path().join("priv-tmp");
    std::fs::create_dir_all(&priv_tmp).unwrap();
    std::fs::set_permissions(&priv_tmp, std::fs::Permissions::from_mode(0o700)).unwrap();
    let out = install_cmd()
        .arg("--tarball")
        .arg(&t)
        .arg("--prefix")
        .arg(tmp.path().join("prefix-ok"))
        .env("TMPDIR", &priv_tmp)
        .output()
        .expect("spawn install.sh");
    assert!(
        out.status.success(),
        "a private TMPDIR must install fine: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[cfg(target_os = "macos")]
#[test]
fn install_service_agent_refuses_a_write_only_node_log() {
    // macOS LaunchAgent (default, NON-root): launchd opens the log READ/WRITE, so
    // the preflight must probe R/W — a write-only node.log must be refused HERE,
    // before a reinstall boots out the working agent. The probe is selected by
    // PLATFORM, not elevation: the old `euid == 0` gate sent the non-root agent to
    // the append probe, which would have WRONGLY accepted this write-only log.
    use std::os::unix::fs::PermissionsExt;
    let tmp = tempfile::tempdir().unwrap();
    let bin = tmp.path().join("bin");
    let v1 = bin.join("v1");
    std::fs::create_dir_all(&v1).unwrap();
    let exe = v1.join("higgs");
    std::fs::write(&exe, b"#!/bin/sh\nexit 0\n").unwrap();
    std::fs::set_permissions(&exe, std::fs::Permissions::from_mode(0o755)).unwrap();
    std::os::unix::fs::symlink("v1", bin.join("current")).unwrap();
    let logs = tmp.path().join("logs");
    std::fs::create_dir_all(&logs).unwrap();
    let node_log = logs.join("node.log");
    std::fs::write(&node_log, b"").unwrap();
    std::fs::set_permissions(&node_log, std::fs::Permissions::from_mode(0o200)).unwrap();

    let out = higgs(&[
        "node",
        "install-service",
        "--prefix",
        tmp.path().to_str().unwrap(),
    ]);
    assert!(
        !out.status.success(),
        "a write-only node.log must refuse (launchd opens it read/write)"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("not openable") && stderr.contains("read/write"),
        "must refuse at the R/W log probe (before any launchd exec): {stderr}"
    );
    std::fs::set_permissions(&node_log, std::fs::Permissions::from_mode(0o644)).unwrap();
}

#[test]
fn install_service_refuses_a_world_writable_ancestor_of_the_resolved_logs() {
    // The logs check must walk the RESOLVED logs' whole ANCESTRY, not just the
    // endpoint: `logs -> shared/logs` with `shared` world-writable (non-sticky)
    // lets a peer RENAME shared/logs and plant node.log, even though the logs dir
    // itself is 0755. (The exec stub exits 3 so that if the walk is reverted, the
    // binary preflight refuses next — the test never reaches a real launchd exec.)
    use std::os::unix::fs::PermissionsExt;
    let tmp = tempfile::tempdir().unwrap();
    let bin = tmp.path().join("bin");
    let v1 = bin.join("v1");
    std::fs::create_dir_all(&v1).unwrap();
    let exe = v1.join("higgs");
    std::fs::write(&exe, b"#!/bin/sh\nexit 3\n").unwrap();
    std::fs::set_permissions(&exe, std::fs::Permissions::from_mode(0o755)).unwrap();
    std::os::unix::fs::symlink("v1", bin.join("current")).unwrap();
    // logs -> shared/logs; shared is 0777 (non-sticky), the logs dir itself 0755.
    let shared = tmp.path().join("shared");
    let real_logs = shared.join("logs");
    std::fs::create_dir_all(&real_logs).unwrap();
    std::fs::set_permissions(&shared, std::fs::Permissions::from_mode(0o777)).unwrap();
    std::os::unix::fs::symlink(&real_logs, tmp.path().join("logs")).unwrap();

    let out = higgs(&[
        "node",
        "install-service",
        "--prefix",
        tmp.path().to_str().unwrap(),
    ]);
    assert!(
        !out.status.success(),
        "a world-writable ANCESTOR of the resolved logs must refuse"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("OTHER-writable"),
        "must refuse the resolved logs' ancestry, not just the endpoint: {stderr}"
    );
    std::fs::set_permissions(&shared, std::fs::Permissions::from_mode(0o755)).unwrap();
}

#[test]
fn install_service_refuses_a_world_writable_symlinked_prefix_target() {
    // A symlinked `--prefix` whose REAL target is world-writable, with `current`
    // resolving OUTSIDE the prefix, is caught ONLY by the physical-prefix walk:
    // the lexical walk skips the prefix symlink node, and the resolved exec walk
    // follows `current` outside and never revisits the real prefix. (Stub exits 3
    // so a reverted prefix-walk falls through to the binary preflight, not launchd.)
    use std::os::unix::fs::PermissionsExt;
    let tmp = tempfile::tempdir().unwrap();
    let real = tmp.path().join("real");
    std::fs::create_dir_all(real.join("bin")).unwrap();
    // current -> a clean version dir OUTSIDE the prefix subtree.
    let outside = tmp.path().join("outside").join("v1");
    std::fs::create_dir_all(&outside).unwrap();
    let exe = outside.join("higgs");
    std::fs::write(&exe, b"#!/bin/sh\nexit 3\n").unwrap();
    std::fs::set_permissions(&exe, std::fs::Permissions::from_mode(0o755)).unwrap();
    std::os::unix::fs::symlink(&outside, real.join("bin").join("current")).unwrap();
    // The REAL prefix is world-writable; --prefix is a SYMLINK to it.
    std::fs::set_permissions(&real, std::fs::Permissions::from_mode(0o777)).unwrap();
    let link = tmp.path().join("link");
    std::os::unix::fs::symlink(&real, &link).unwrap();

    let out = higgs(&[
        "node",
        "install-service",
        "--prefix",
        link.to_str().unwrap(),
    ]);
    assert!(
        !out.status.success(),
        "a world-writable symlinked-prefix target must refuse"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("OTHER-writable") && stderr.contains("service prefix"),
        "must refuse the physical prefix behind the symlink: {stderr}"
    );
    std::fs::set_permissions(&real, std::fs::Permissions::from_mode(0o755)).unwrap();
}

#[test]
fn install_service_refuses_a_symlinked_logs_pointing_at_world_writable() {
    // r22 made refuse_other_writable SKIP a symlink node (its own mode is
    // meaningless). logs/ has no resolved-walk backstop like the exec path, so the
    // logs check must RESOLVE logs physically — else a `logs -> /tmp/shared` whose
    // target is 0777 (a node.log-symlink plant surface) would pass unchecked.
    use std::os::unix::fs::PermissionsExt;
    let tmp = tempfile::tempdir().unwrap();
    let bin = tmp.path().join("bin");
    let v1 = bin.join("v1");
    std::fs::create_dir_all(&v1).unwrap();
    let exe = v1.join("higgs");
    std::fs::write(&exe, b"#!/bin/sh\nexit 0\n").unwrap();
    std::fs::set_permissions(&exe, std::fs::Permissions::from_mode(0o755)).unwrap();
    std::os::unix::fs::symlink("v1", bin.join("current")).unwrap();
    // logs is a SYMLINK to a world-writable shared dir (the symlink's own mode is
    // 0755, but the TARGET is 0777).
    let shared = tmp.path().join("shared-logs");
    std::fs::create_dir_all(&shared).unwrap();
    std::fs::set_permissions(&shared, std::fs::Permissions::from_mode(0o777)).unwrap();
    std::os::unix::fs::symlink(&shared, tmp.path().join("logs")).unwrap();

    let out = higgs(&[
        "node",
        "install-service",
        "--prefix",
        tmp.path().to_str().unwrap(),
    ]);
    assert!(
        !out.status.success(),
        "a symlinked logs -> world-writable target must refuse"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("OTHER-writable") && stderr.contains("log dir"),
        "must refuse the resolved logs dir: {stderr}"
    );
    std::fs::set_permissions(&shared, std::fs::Permissions::from_mode(0o755)).unwrap();
}

#[test]
fn install_sh_refuses_a_symlinked_bin_pointing_at_world_writable() {
    // install.sh's ancestry walk used `find <symlink>`, which inspects the
    // SYMLINK's own 0755 mode — so a `bin -> /tmp/shared` whose target is 0777
    // (into which the install would publish current/v<ver>, a swap target for any
    // local user) slipped through. The walk must resolve the physical dir.
    use std::os::unix::fs::PermissionsExt;
    let tmp = tempfile::tempdir().unwrap();
    let stage = tmp.path().join("stage");
    std::fs::create_dir_all(&stage).unwrap();
    let t = stage_artifact(&stage, "0.0.1");
    let prefix = tmp.path().join("prefix");
    std::fs::create_dir_all(&prefix).unwrap();
    let shared = tmp.path().join("shared-bin");
    std::fs::create_dir_all(&shared).unwrap();
    std::fs::set_permissions(&shared, std::fs::Permissions::from_mode(0o777)).unwrap();
    std::os::unix::fs::symlink(&shared, prefix.join("bin")).unwrap();

    let out = run_install(&t, &prefix);
    assert!(
        !out.status.success(),
        "a symlinked bin -> world-writable target must refuse"
    );
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("world-writable"),
        "refusal must name the world-write problem: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    std::fs::set_permissions(&shared, std::fs::Permissions::from_mode(0o755)).unwrap();
}

#[test]
fn install_service_refuses_a_world_writable_version_dir() {
    // The exec path is <prefix>/bin/current/higgs, current → v<ver>. A
    // world-writable VERSION dir (not just prefix/bin) is a swap target too, so
    // install-service must resolve `current` and refuse an other-writable
    // version dir BEFORE the binary preflight trusts what's inside it.
    use std::os::unix::fs::PermissionsExt;
    let tmp = tempfile::tempdir().unwrap();
    let bin = tmp.path().join("bin");
    let v1 = bin.join("v1");
    std::fs::create_dir_all(&v1).unwrap();
    let exe = v1.join("higgs");
    std::fs::write(&exe, b"#!/bin/sh\nexit 0\n").unwrap();
    std::fs::set_permissions(&exe, std::fs::Permissions::from_mode(0o755)).unwrap();
    std::os::unix::fs::symlink("v1", bin.join("current")).unwrap();
    // prefix and bin are fine; only the VERSION dir is world-writable.
    std::fs::set_permissions(&v1, std::fs::Permissions::from_mode(0o777)).unwrap();

    let out = higgs(&[
        "node",
        "install-service",
        "--prefix",
        tmp.path().to_str().unwrap(),
    ]);
    assert!(
        !out.status.success(),
        "world-writable version dir must refuse"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("OTHER-writable"),
        "must refuse the world-writable version dir on the exec path: {stderr}"
    );
    std::fs::set_permissions(&v1, std::fs::Permissions::from_mode(0o755)).unwrap();
}

#[test]
fn install_sh_refuses_a_preexisting_world_writable_version_dir_on_reinstall() {
    // A prior `umask 000` install could leave bin/v0.0.1 at 0777 even after the
    // operator hardens prefix+bin. A reinstall PRESERVES that dir, so install.sh
    // must check the version dir too.
    use std::os::unix::fs::PermissionsExt;
    let tmp = tempfile::tempdir().unwrap();
    let stage = tmp.path().join("stage");
    std::fs::create_dir_all(&stage).unwrap();
    let t = stage_artifact(&stage, "0.0.1");
    let prefix = tmp.path().join("prefix");
    // First install (clean).
    assert!(run_install(&t, &prefix).status.success());
    // Make the version dir world-writable, then reinstall the SAME version.
    std::fs::set_permissions(
        prefix.join("bin/v0.0.1"),
        std::fs::Permissions::from_mode(0o777),
    )
    .unwrap();
    let out = run_install(&t, &prefix);
    assert!(
        !out.status.success(),
        "reinstall over a world-writable version dir must refuse"
    );
    assert!(String::from_utf8_lossy(&out.stderr).contains("world-writable"));
    std::fs::set_permissions(
        prefix.join("bin/v0.0.1"),
        std::fs::Permissions::from_mode(0o755),
    )
    .unwrap();
}

#[test]
fn install_sh_bounds_dir_modes_against_a_permissive_umask() {
    // Under `umask 000` the install tree would be WORLD-WRITABLE without the
    // pinned `umask 022`, letting a local user replace the binary/current/unit
    // and run code as the operator. Every created dir must be at most 0755
    // (no group/other write).
    use std::os::unix::fs::PermissionsExt;
    let tmp = tempfile::tempdir().unwrap();
    let stage = tmp.path().join("stage");
    std::fs::create_dir_all(&stage).unwrap();
    let t = stage_artifact(&stage, "0.0.1");
    let prefix = tmp.path().join("prefix");
    // Run through `sh -c 'umask 000; …'` so the child install.sh inherits the
    // permissive umask (which its own `umask 022` must then override).
    let out = Command::new("sh")
        .arg("-c")
        .arg(format!(
            "umask 000; exec bash {} --tarball {} --prefix {}",
            concat!(env!("CARGO_MANIFEST_DIR"), "/install.sh"),
            t.display(),
            prefix.display()
        ))
        .output()
        .expect("spawn install.sh under umask 000");
    assert!(out.status.success(), "install failed: {out:?}");
    for d in [
        prefix.join("bin"),
        prefix.join("bin/v0.0.1"),
        prefix.join("logs"),
    ] {
        if let Ok(m) = std::fs::metadata(&d) {
            let mode = m.permissions().mode() & 0o777;
            assert_eq!(
                mode & 0o022,
                0,
                "{} must not be group/world-writable, got {mode:o}",
                d.display()
            );
        }
    }
}

#[test]
fn install_sh_installs_updates_rolls_back_and_refuses_tamper() {
    let tmp = tempfile::tempdir().unwrap();
    let stage = tmp.path().join("stage");
    std::fs::create_dir_all(&stage).unwrap();
    let prefix = tmp.path().join("root");

    // Install v0.0.1: versioned dir + atomic `current` symlink.
    let t1 = stage_artifact(&stage, "0.0.1");
    let out = run_install(&t1, &prefix);
    assert!(out.status.success(), "install failed: {out:?}");
    assert!(prefix.join("bin/v0.0.1/higgs").exists());
    assert_eq!(
        std::fs::read_link(prefix.join("bin/current")).unwrap(),
        Path::new("v0.0.1")
    );

    // Update to v0.0.2: current flips, old version dir stays (rollback fodder).
    let t2 = stage_artifact(&stage, "0.0.2");
    let out = run_install(&t2, &prefix);
    assert!(out.status.success(), "update failed: {out:?}");
    assert_eq!(
        std::fs::read_link(prefix.join("bin/current")).unwrap(),
        Path::new("v0.0.2")
    );
    assert!(prefix.join("bin/v0.0.1/higgs").exists());

    // The Next-steps service command ALWAYS carries the resolved --prefix
    // (install-service defaults to the passwd home, which can differ). The
    // DEFAULT line is the user-space scope — NO sudo on either OS; the
    // always-on line carries --system (with sudo only on macOS, where it is a
    // LaunchDaemon). install.sh resolves the prefix PHYSICALLY (`pwd -P`), so
    // compare against the canonical path (on macOS /var → /private/var).
    let steps = String::from_utf8_lossy(&out.stdout);
    let canon_prefix = prefix.canonicalize().unwrap_or_else(|_| prefix.clone());
    assert!(
        steps.contains(&format!(
            "node install-service --prefix {}",
            canon_prefix.display()
        )),
        "Next-steps must carry the resolved --prefix: {steps}"
    );
    assert!(
        !steps.contains("service: sudo"),
        "the default service line must be user-space, no sudo: {steps}"
    );
    assert!(
        steps.contains("--system"),
        "Next-steps must offer the always-on --system variant: {steps}"
    );
    if cfg!(target_os = "macos") {
        assert!(
            steps.contains("always-on: sudo "),
            "macOS always-on line needs sudo (LaunchDaemon): {steps}"
        );
    } else {
        assert!(
            !steps.contains("always-on: sudo "),
            "Linux always-on stays user-space (linger), no sudo: {steps}"
        );
    }

    // Rollback = re-run the old version's installer; only the symlink moves.
    let out = run_install(&t1, &prefix);
    assert!(out.status.success(), "rollback failed: {out:?}");
    assert_eq!(
        std::fs::read_link(prefix.join("bin/current")).unwrap(),
        Path::new("v0.0.1")
    );

    // Tampered tarball: sha256 mismatch refuses BEFORE touching `current`.
    let mut bytes = std::fs::read(&t2).unwrap();
    bytes.extend_from_slice(b"tampered");
    std::fs::write(&t2, bytes).unwrap();
    let out = run_install(&t2, &prefix);
    assert!(!out.status.success(), "tampered install must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("MISMATCH"), "stderr: {stderr}");
    assert_eq!(
        std::fs::read_link(prefix.join("bin/current")).unwrap(),
        Path::new("v0.0.1"),
        "a refused install must not move current"
    );

    // No flip-temp litter across the whole lifecycle (install, update, rollback,
    // refused install): a leftover `.current.tmp.*` is the raw material for the
    // stale-link flip hazard, so the trap must have cleaned every staged link.
    let litter: Vec<_> = std::fs::read_dir(prefix.join("bin"))
        .unwrap()
        .filter_map(std::result::Result::ok)
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.starts_with(".current.tmp."))
        .collect();
    assert!(
        litter.is_empty(),
        "flip-temp litter left behind: {litter:?}"
    );
}

/// Reinstalling the version that `current` already points at must not truncate
/// the (possibly running) binary: it stages + renames rather than extracting
/// in place. We can't hold the binary open portably, but we can assert the
/// reinstall succeeds and leaves a working binary + the same symlink.
#[test]
fn install_sh_reinstalls_the_current_version_safely() {
    let tmp = tempfile::tempdir().unwrap();
    let stage = tmp.path().join("stage");
    std::fs::create_dir_all(&stage).unwrap();
    let prefix = tmp.path().join("root");
    let t = stage_artifact(&stage, "0.0.1");

    assert!(run_install(&t, &prefix).status.success());
    let inode_before = std::fs::metadata(prefix.join("bin/v0.0.1/higgs"))
        .unwrap()
        .ino();
    // Reinstall the SAME version.
    assert!(
        run_install(&t, &prefix).status.success(),
        "reinstall failed"
    );
    let inode_after = std::fs::metadata(prefix.join("bin/v0.0.1/higgs"))
        .unwrap()
        .ino();
    assert_ne!(
        inode_before, inode_after,
        "a safe reinstall replaces the file via rename (new inode), never truncates in place"
    );
    assert_eq!(
        std::fs::read_link(prefix.join("bin/current")).unwrap(),
        Path::new("v0.0.1")
    );
}

/// If the published path `v<ver>/higgs` already exists as a DIRECTORY, BSD `mv`
/// would descend and nest the binary as `v<ver>/higgs/higgs`, leaving
/// `current/higgs` a directory while the install reports success. Publishing
/// via rename(2) must FAIL the install instead of silently nesting.
#[test]
fn install_sh_refuses_to_publish_over_a_directory() {
    let tmp = tempfile::tempdir().unwrap();
    let stage = tmp.path().join("stage");
    std::fs::create_dir_all(&stage).unwrap();
    let prefix = tmp.path().join("root");
    let t1 = stage_artifact(&stage, "0.0.1");

    // Pre-plant a directory exactly where the published `higgs` file must land.
    std::fs::create_dir_all(prefix.join("bin/v0.0.1/higgs")).unwrap();

    let out = run_install(&t1, &prefix);
    assert!(
        !out.status.success(),
        "install must refuse to nest into a directory: {out:?}"
    );
    // The bogus nested file must NOT exist, and `current` must never be flipped.
    assert!(
        !prefix.join("bin/v0.0.1/higgs/higgs").exists(),
        "rename(2) must not nest the binary inside the directory"
    );
    assert!(
        !prefix.join("bin/current").exists(),
        "a failed publish must not flip current"
    );
}

/// `bash -x install.sh` must NOT print the PAT in its trace: the `-K -` pipe
/// keeps it out of argv/ps, but `set -x` would expand `$HIGGS_GITHUB_TOKEN`
/// into the trace and into any CI/debug log. Force the GitHub path (no
/// `--tarball`) with a fake `curl` that exits 1 (no real network), and assert
/// the sentinel token never lands in the captured trace.
#[test]
fn install_sh_keeps_the_pat_out_of_an_xtrace() {
    let tmp = tempfile::tempdir().unwrap();
    let fakebin = tmp.path().join("bin");
    std::fs::create_dir_all(&fakebin).unwrap();
    let curl = fakebin.join("curl");
    std::fs::write(&curl, "#!/bin/sh\nexit 1\n").unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&curl, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    let sentinel = "SENTINEL_PAT_do_not_leak_1234567890";
    let orig_path = std::env::var("PATH").unwrap_or_default();
    let out = Command::new("bash")
        .arg("-x")
        .arg(concat!(env!("CARGO_MANIFEST_DIR"), "/install.sh"))
        .env("HIGGS_GITHUB_TOKEN", sentinel)
        // Fake curl FIRST so gh_api's fetch fails offline; real tools follow.
        .env("PATH", format!("{}:{}", fakebin.display(), orig_path))
        .output()
        .expect("spawn bash -x install.sh");
    // It fails (faked curl / missing jq), but the token must never be traced.
    assert!(
        !out.status.success(),
        "should fail on the GitHub path offline"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !stderr.contains(sentinel),
        "the PAT leaked into the `bash -x` trace:\n{stderr}"
    );
}

/// SECURITY: the root-safety guard must use bash's `$EUID` (no `id` subprocess),
/// and the PAT must be dropped from the environment BEFORE the first subprocess
/// so no child (uname, jq, …) inherits it. Plant a fake `id` (should NEVER run)
/// and a fake `uname` (runs early — must NOT see the token in its env) and
/// assert neither leaks.
#[test]
fn install_sh_guard_uses_euid_and_drops_the_pat_before_subprocesses() {
    use std::os::unix::fs::PermissionsExt;
    let tmp = tempfile::tempdir().unwrap();
    let fakebin = tmp.path().join("bin");
    std::fs::create_dir_all(&fakebin).unwrap();
    let id_ran = tmp.path().join("ID_RAN");
    let token_leaked = tmp.path().join("TOKEN_LEAKED");
    // Fake `id`: records that it was called (the guard must use $EUID instead).
    std::fs::write(
        fakebin.join("id"),
        format!(
            "#!/bin/sh\ntouch '{}'\nexec /usr/bin/id \"$@\"\n",
            id_ran.display()
        ),
    )
    .unwrap();
    // Fake `uname`: leaks if it inherits the PAT, then runs the real uname.
    std::fs::write(
        fakebin.join("uname"),
        format!(
            "#!/bin/sh\n[ -n \"${{HIGGS_GITHUB_TOKEN:-}}\" ] && printf leaked > '{}'\nexec /usr/bin/uname \"$@\"\n",
            token_leaked.display()
        ),
    )
    .unwrap();
    for f in ["id", "uname"] {
        std::fs::set_permissions(fakebin.join(f), std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    let orig_path = std::env::var("PATH").unwrap_or_default();
    // No --tarball → the GitHub path; it will fail at the (unreachable) network,
    // but the guard + platform detection run first, which is all we assert on.
    let _ = Command::new("bash")
        .arg(concat!(env!("CARGO_MANIFEST_DIR"), "/install.sh"))
        .env("HIGGS_GITHUB_TOKEN", "SENTINEL_PAT_leak_me_not")
        .env("PATH", format!("{}:{}", fakebin.display(), orig_path))
        .output()
        .expect("spawn bash install.sh");
    assert!(
        !id_ran.exists(),
        "root guard must use $EUID, not an `id` subprocess"
    );
    assert!(
        !token_leaked.exists(),
        "the PAT must be unset before the first subprocess (uname inherited it)"
    );
}

/// SECURITY: `SSLKEYLOGFILE` must be scrubbed before any subprocess — curl
/// honors it and writes TLS session secrets to disk, from which the PAT could be
/// recovered. `unset SSLKEYLOGFILE` at the top removes it from every child's env.
/// Plant a `uname` that leaks if it inherited it, and assert it never does.
#[test]
fn install_sh_scrubs_sslkeylogfile() {
    use std::os::unix::fs::PermissionsExt;
    let tmp = tempfile::tempdir().unwrap();
    let fakebin = tmp.path().join("bin");
    std::fs::create_dir_all(&fakebin).unwrap();
    let leaked = tmp.path().join("SSLKEYLOGFILE_LEAKED");
    std::fs::write(
        fakebin.join("uname"),
        format!(
            "#!/bin/sh\n[ -n \"${{SSLKEYLOGFILE:-}}\" ] && printf leaked > '{}'\nexec /usr/bin/uname \"$@\"\n",
            leaked.display()
        ),
    )
    .unwrap();
    std::fs::set_permissions(
        fakebin.join("uname"),
        std::fs::Permissions::from_mode(0o755),
    )
    .unwrap();
    let orig_path = std::env::var("PATH").unwrap_or_default();
    let _ = Command::new("bash")
        .arg(concat!(env!("CARGO_MANIFEST_DIR"), "/install.sh"))
        .args(["--tarball", "/nonexistent/higgs.tar.gz"])
        .env("SSLKEYLOGFILE", "/tmp/higgs-keylog-should-be-scrubbed")
        .env("PATH", format!("{}:{}", fakebin.display(), orig_path))
        .output()
        .expect("spawn bash install.sh");
    assert!(
        !leaked.exists(),
        "SSLKEYLOGFILE must be scrubbed before the first subprocess"
    );
}

/// SECURITY: the `#!/bin/bash -p` (privileged) shebang must make bash IGNORE
/// `$BASH_ENV` — otherwise arbitrary code in a planted BASH_ENV file runs before
/// the root guard (as root under sudo) and can read the PAT. Exec the script
/// DIRECTLY (only that honors the shebang) with a BASH_ENV that drops a sentinel
/// and assert it never runs.
#[test]
fn install_sh_privileged_mode_ignores_bash_env() {
    use std::os::unix::fs::PermissionsExt;
    let tmp = tempfile::tempdir().unwrap();
    // Copy the script so we can exec it directly (chmod) without touching the
    // repo file; the shebang (with `-p`) travels with the content.
    let script = tmp.path().join("install.sh");
    std::fs::copy(concat!(env!("CARGO_MANIFEST_DIR"), "/install.sh"), &script).unwrap();
    std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
    let ran = tmp.path().join("BASH_ENV_RAN");
    let envfile = tmp.path().join("evil.sh");
    std::fs::write(&envfile, format!("touch '{}'\n", ran.display())).unwrap();
    let _ = Command::new(&script)
        // A missing tarball → the script exits fast, but BASH_ENV is processed
        // at bash STARTUP (before the body) regardless, so the sentinel would
        // already exist if `-p` weren't suppressing it.
        .args(["--tarball", "/nonexistent/higgs.tar.gz"])
        .env("BASH_ENV", &envfile)
        .output()
        .expect("exec install.sh directly");
    assert!(
        !ran.exists(),
        "privileged mode (#!/bin/bash -p) must ignore BASH_ENV"
    );
}

/// SECURITY: capturing the PAT into `_gh_token` must not RE-EXPORT it. Bash
/// preserves the export attribute of a variable imported from the environment,
/// so if `_gh_token` arrives already-exported the assignment would re-export the
/// PAT to every subprocess. `export -n _gh_token` clears it. Pre-export
/// `_gh_token`, plant a `uname` that leaks it, and assert it never does.
#[test]
fn install_sh_does_not_reexport_the_captured_pat() {
    use std::os::unix::fs::PermissionsExt;
    let tmp = tempfile::tempdir().unwrap();
    let fakebin = tmp.path().join("bin");
    std::fs::create_dir_all(&fakebin).unwrap();
    let leaked = tmp.path().join("GH_TOKEN_VAR_LEAKED");
    std::fs::write(
        fakebin.join("uname"),
        format!(
            "#!/bin/sh\n[ -n \"${{_gh_token:-}}\" ] && printf leaked > '{}'\nexec /usr/bin/uname \"$@\"\n",
            leaked.display()
        ),
    )
    .unwrap();
    std::fs::set_permissions(
        fakebin.join("uname"),
        std::fs::Permissions::from_mode(0o755),
    )
    .unwrap();
    let orig_path = std::env::var("PATH").unwrap_or_default();
    let _ = Command::new("bash")
        .arg(concat!(env!("CARGO_MANIFEST_DIR"), "/install.sh"))
        .env("HIGGS_GITHUB_TOKEN", "SENTINEL_PAT_reexport")
        // A child env var is imported by bash as EXPORTED — the capture must
        // strip that attribute so the PAT isn't re-exported through it.
        .env("_gh_token", "placeholder")
        .env("PATH", format!("{}:{}", fakebin.display(), orig_path))
        .output()
        .expect("spawn bash install.sh");
    assert!(
        !leaked.exists(),
        "the captured PAT must not be re-exported via a pre-exported _gh_token"
    );
}

/// SECURITY: the PAT is piped into curl, so a `curl` planted on the operator's
/// PATH must NEVER receive it. gh_api pins the trusted absolute `/usr/bin/curl`,
/// so a planted curl is bypassed. Plant one that records if it ran and assert it
/// never does. (Briefly reaches api.github.com for a non-existent repo → a fast
/// 404; offline it just fails to connect — either way the planted curl is not
/// called.)
#[test]
fn install_sh_uses_trusted_absolute_curl_not_a_planted_one() {
    use std::os::unix::fs::PermissionsExt;
    if !Path::new("/usr/bin/curl").exists() {
        eprintln!("skipping: no /usr/bin/curl on this host");
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let fakebin = tmp.path().join("bin");
    std::fs::create_dir_all(&fakebin).unwrap();
    let planted_ran = tmp.path().join("PLANTED_CURL_RAN");
    std::fs::write(
        fakebin.join("curl"),
        format!(
            "#!/bin/sh\ncat >/dev/null 2>&1\ntouch '{}'\nexit 1\n",
            planted_ran.display()
        ),
    )
    .unwrap();
    std::fs::set_permissions(fakebin.join("curl"), std::fs::Permissions::from_mode(0o755)).unwrap();
    let orig_path = std::env::var("PATH").unwrap_or_default();
    let _ = Command::new("bash")
        .arg(concat!(env!("CARGO_MANIFEST_DIR"), "/install.sh"))
        // A non-existent repo → the real curl gets a fast 404 and dies; we only
        // care that the PLANTED curl was never reached.
        .args(["--repo", "AI-Experiri/does-not-exist-xyz-99999"])
        .env("HIGGS_GITHUB_TOKEN", "not_a_real_token")
        .env("PATH", format!("{}:{}", fakebin.display(), orig_path))
        .output()
        .expect("spawn bash install.sh");
    assert!(
        !planted_ran.exists(),
        "gh_api must invoke /usr/bin/curl, never a PATH-planted curl"
    );
}

/// A `--pubkey` install whose signed manifest is for a DIFFERENT version must
/// be refused — a signed old manifest cannot be replayed under a new name.
#[test]
fn install_sh_refuses_a_manifest_for_the_wrong_version() {
    // Requires the minisign CLI; skip cleanly when absent.
    if Command::new("minisign").arg("-v").output().is_err() {
        eprintln!("skipping: minisign CLI not installed");
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let stage = tmp.path().join("stage");
    std::fs::create_dir_all(&stage).unwrap();
    let prefix = tmp.path().join("root");

    // Mint a throwaway signing key.
    let keydir = tmp.path().join("keys");
    std::fs::create_dir_all(&keydir).unwrap();
    let sk = keydir.join("mini.key");
    let pk = keydir.join("mini.pub");
    let g = Command::new("minisign")
        .args(["-G", "-W", "-f", "-s"])
        .arg(&sk)
        .arg("-p")
        .arg(&pk)
        .output()
        .unwrap();
    assert!(g.status.success(), "keygen: {g:?}");
    let pub_b64 = std::fs::read_to_string(&pk)
        .unwrap()
        .lines()
        .nth(1)
        .unwrap()
        .to_string();

    // Stage a v0.0.2 tarball but sign a manifest that CLAIMS version 0.0.1
    // (the replay: old manifest + a tarball renamed to the new version).
    let t2 = stage_artifact(&stage, "0.0.2");
    let name2 = format!("higgs-v0.0.2-{}", suffix());
    let sha = String::from_utf8(
        Command::new("shasum")
            .args(["-a", "256"])
            .arg(&t2)
            .output()
            .unwrap()
            .stdout,
    )
    .unwrap()
    .split_whitespace()
    .next()
    .unwrap()
    .to_string();
    let target = if cfg!(target_os = "macos") {
        "aarch64-apple-darwin"
    } else {
        "x86_64-unknown-linux-gnu"
    };
    let variant = if cfg!(target_os = "macos") {
        "metal"
    } else {
        "cpu"
    };
    let manifest = stage.join(format!("{name2}.manifest"));
    std::fs::write(
        &manifest,
        format!(
            r#"{{"schema":1,"version":"0.0.1","commit":"deadbeef","file":"{name2}.tar.gz","target":"{target}","variant":"{variant}","sha256":"{sha}"}}"#
        ),
    )
    .unwrap();
    let s = Command::new("minisign")
        .args(["-S", "-s"])
        .arg(&sk)
        .arg("-m")
        .arg(&manifest)
        .arg("-x")
        .arg(stage.join(format!("{name2}.manifest.minisig")))
        .output()
        .unwrap();
    assert!(s.status.success(), "sign: {s:?}");

    let out = install_cmd()
        .arg("--tarball")
        .arg(&t2)
        .arg("--prefix")
        .arg(&prefix)
        .arg("--pubkey")
        .arg(&pub_b64)
        .output()
        .unwrap();
    assert!(
        !out.status.success(),
        "wrong-version manifest must be refused"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("version 0.0.1, expected 0.0.2"),
        "stderr: {stderr}"
    );
}

/// A relative `--prefix` must resolve to an absolute path BEFORE the temp
/// workdir is entered/cleaned — otherwise the install would land in (and be
/// deleted with) the workdir. Run from a known cwd and assert the install
/// materialized under the resolved absolute path.
#[test]
fn install_sh_resolves_a_relative_prefix_safely() {
    let tmp = tempfile::tempdir().unwrap();
    let stage = tmp.path().join("stage");
    std::fs::create_dir_all(&stage).unwrap();
    let t = stage_artifact(&stage, "0.0.1");

    let out = install_cmd()
        .current_dir(tmp.path())
        .arg("--tarball")
        .arg(&t)
        .arg("--prefix")
        .arg("relnode") // relative to tmp
        .output()
        .expect("spawn");
    assert!(
        out.status.success(),
        "relative-prefix install failed: {out:?}"
    );
    // It must exist under tmp/relnode, and still exist after the process exits
    // (i.e. it was NOT created inside the auto-deleted workdir).
    assert!(tmp.path().join("relnode/bin/v0.0.1/higgs").exists());
}
