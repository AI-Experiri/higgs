//! Integration coverage for the REAL `higgs node self-update` CLI surface
//! (DESIGN-remote §9 P3). The build pins the real release key, so a manifest
//! whose signature does not verify under it is fail-closed (HG082) — the
//! reachable outcomes here are: the verified update refusing an unverifiable
//! signature, argument parsing, and the no-signature maintenance ops
//! (`--rollback`, `--prune`) against a hand-built `bin/` layout.
//! Everything runs as a plain one-shot CLI (no server, no GGUF).

use std::path::Path;
use std::process::Command;

fn higgs(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_higgs"))
        .args(args)
        .output()
        .expect("spawn higgs")
}

/// Build `<root>/bin` with `v<ver>/higgs` dirs and `current -> v<current>`, as
/// install.sh would. Returns the bin path.
fn install_layout(root: &Path, versions: &[&str], current: &str) -> std::path::PathBuf {
    use std::os::unix::fs::PermissionsExt;
    let bin = root.join("bin");
    for v in versions {
        let vdir = bin.join(format!("v{v}"));
        std::fs::create_dir_all(&vdir).unwrap();
        let higgs = vdir.join("higgs");
        std::fs::write(&higgs, b"#!/bin/sh\n").unwrap();
        // A runnable (0755) binary — the rollback-target check requires it.
        std::fs::set_permissions(&higgs, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    std::os::unix::fs::symlink(format!("v{current}"), bin.join("current")).unwrap();
    bin
}

fn current_target(bin: &Path) -> Option<String> {
    std::fs::read_link(bin.join("current"))
        .ok()
        .map(|t| t.to_string_lossy().into_owned())
}

// ---- update path: fail-closed on an unverifiable signature ---------------

#[test]
fn self_update_refuses_an_unverifiable_signature() {
    let tmp = tempfile::tempdir().unwrap();
    // A local source of arbitrary bytes — verification refuses (HG082: the signature
    // does not verify under the pinned key) before it ever fetches the tarball.
    let m = tmp.path().join("m.json");
    let s = tmp.path().join("m.minisig");
    let t = tmp.path().join("higgs.tar.gz");
    std::fs::write(&m, br#"{"schema":1,"version":"9.9.9"}"#).unwrap();
    std::fs::write(&s, b"untrusted comment: x\nAAAA\n").unwrap();
    std::fs::write(&t, b"not really a tarball").unwrap();

    let out = higgs(&[
        "node",
        "self-update",
        "--prefix",
        tmp.path().to_str().unwrap(),
        "--tarball",
        t.to_str().unwrap(),
        "--manifest",
        m.to_str().unwrap(),
        "--manifest-sig",
        s.to_str().unwrap(),
    ]);
    assert!(
        !out.status.success(),
        "an unverifiable self-update must fail: {out:?}"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("HG082"),
        "expected fail-closed HG082, got: {stderr}"
    );
}

// ---- update path: network fetch (--url, P4) ------------------------------

/// A minimal loopback HTTP/1.1 server that answers the next `n` requests, replying with
/// `sig` for any path containing `.minisig` and `manifest` otherwise. Returns the manifest
/// URL. The thread serves exactly `n` requests then exits (the pipeline fetches the
/// manifest + its sig, then fails HG082 before the artifact, so `n = 2`).
fn spawn_http_fixture(manifest: &[u8], sig: &[u8], n: usize) -> String {
    use std::io::{Read, Write};
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let manifest = manifest.to_vec();
    let sig = sig.to_vec();
    std::thread::spawn(move || {
        for _ in 0..n {
            let Ok((mut stream, _)) = listener.accept() else {
                return;
            };
            let mut buf = [0u8; 2048];
            let read = stream.read(&mut buf).unwrap_or(0);
            let req = String::from_utf8_lossy(&buf[..read]);
            let body: &[u8] = if req.contains(".minisig") {
                &sig
            } else {
                &manifest
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
fn self_update_url_fetches_then_fails_closed_on_an_unverifiable_signature() {
    // The --url path FETCHES the manifest + sig over loopback http, then refuses at
    // verification (HG082: does not verify under the pinned key). Reaching HG082 (not a
    // connect/HG088 error) proves the fetch pipeline ran end-to-end.
    let tmp = tempfile::tempdir().unwrap();
    let url = spawn_http_fixture(
        br#"{"schema":1,"version":"9.9.9","commit":"x","file":"higgs.tar.gz","target":"t","variant":"v","sha256":"00"}"#,
        b"untrusted comment: x\nRWQAAA==\n",
        2,
    );
    let out = higgs(&[
        "node",
        "self-update",
        "--prefix",
        tmp.path().to_str().unwrap(),
        "--url",
        &url,
    ]);
    assert!(
        !out.status.success(),
        "unverifiable --url must fail: {out:?}"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("HG082"),
        "expected fetch-then-fail-closed HG082, got: {stderr}"
    );
}

#[test]
fn self_update_url_refuses_plaintext_http_to_a_remote_host() {
    // A non-loopback http:// URL is refused BEFORE any fetch (HG088), so no network I/O.
    let tmp = tempfile::tempdir().unwrap();
    let out = higgs(&[
        "node",
        "self-update",
        "--prefix",
        tmp.path().to_str().unwrap(),
        "--url",
        "http://example.com/higgs.manifest",
    ]);
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("HG088"),
        "expected HG088 fetch refusal, got: {stderr}"
    );
}

#[test]
fn self_update_dash_prefixed_url_is_not_echoed() {
    // A URL with a stray leading `-` (a common typo / shell mangle) starts with `-` but is
    // NOT a flag name — it must be redacted, not echoed as a "flag".
    let tmp = tempfile::tempdir().unwrap();
    let out = higgs(&[
        "node",
        "self-update",
        "--prefix",
        tmp.path().to_str().unwrap(),
        "-https://updates.example/download/PATHSECRET/higgs.manifest",
    ]);
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !stderr.contains("PATHSECRET"),
        "a dash-prefixed URL value must not be echoed: {stderr}"
    );
}

#[test]
fn node_dispatch_dash_prefixed_url_is_not_echoed() {
    let out = higgs(&[
        "node",
        "-https://updates.example/download/NODESECRET/higgs.manifest",
    ]);
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !stderr.contains("NODESECRET"),
        "a dash-prefixed URL at the node dispatch must not be echoed: {stderr}"
    );
}

#[test]
fn self_update_positional_url_is_not_echoed() {
    // A manifest URL passed WITHOUT `--url` (a bare positional) must not be echoed — it can
    // carry a capability path. `split('=')` alone would echo it (no `=`); only redacting
    // bare positionals hides it.
    let tmp = tempfile::tempdir().unwrap();
    let out = higgs(&[
        "node",
        "self-update",
        "--prefix",
        tmp.path().to_str().unwrap(),
        "https://updates.example/download/PATHSECRET/higgs.manifest",
    ]);
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !stderr.contains("PATHSECRET"),
        "a bare positional URL must not be echoed: {stderr}"
    );
}

#[test]
fn node_dispatch_positional_url_is_not_echoed() {
    // `higgs node <bare-url>` (no subcommand) must not echo the URL at the node dispatcher.
    let out = higgs(&[
        "node",
        "https://updates.example/download/PATHSECRET/higgs.manifest",
    ]);
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !stderr.contains("PATHSECRET"),
        "a bare positional subcommand must not be echoed: {stderr}"
    );
}

#[test]
fn node_dispatch_usage_does_not_echo_an_inline_secret() {
    // `higgs node --url=<secret>` (self-update omitted) lands the URL as the unknown
    // subcommand; the top-level node usage must echo only up to `=`, not the secret.
    let out = higgs(&[
        "node",
        "--url=https://updates.example/download/PATHSECRET/higgs.manifest",
    ]);
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !stderr.contains("PATHSECRET"),
        "the node usage must not echo the inline secret: {stderr}"
    );
}

#[test]
fn self_update_unknown_flag_error_does_not_echo_an_inline_secret() {
    // `--url=<value>` (the GNU `=` spelling) is not accepted by this two-token parser and
    // falls to the unknown-flag arm; that error must echo only the flag NAME, not the
    // secret-bearing value after `=`.
    let tmp = tempfile::tempdir().unwrap();
    let out = higgs(&[
        "node",
        "self-update",
        "--prefix",
        tmp.path().to_str().unwrap(),
        "--url=https://updates.example/download/PATHSECRET/higgs.manifest",
    ]);
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !stderr.contains("PATHSECRET"),
        "the unknown-flag error must not echo the inline secret value: {stderr}"
    );
}

#[test]
fn self_update_url_flag_guard_does_not_echo_a_secret_value() {
    // A `--url` value starting with `-` trips the swallowed-flag guard; the error must NOT
    // echo the value (it can carry a capability path / credential).
    let tmp = tempfile::tempdir().unwrap();
    let out = higgs(&[
        "node",
        "self-update",
        "--prefix",
        tmp.path().to_str().unwrap(),
        "--url",
        "-https://updates.example/download/PATHSECRET/higgs.manifest",
    ]);
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !stderr.contains("PATHSECRET"),
        "the --url guard must not echo the secret value: {stderr}"
    );
}

#[test]
fn self_update_url_and_local_source_are_mutually_exclusive() {
    let tmp = tempfile::tempdir().unwrap();
    let t = tmp.path().join("higgs.tar.gz");
    std::fs::write(&t, b"x").unwrap();
    let out = higgs(&[
        "node",
        "self-update",
        "--prefix",
        tmp.path().to_str().unwrap(),
        "--url",
        "https://example.com/higgs.manifest",
        "--tarball",
        t.to_str().unwrap(),
    ]);
    assert!(!out.status.success());
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("not both"),
        "expected a mutual-exclusion error"
    );
}

#[test]
fn self_update_dry_run_also_fails_closed() {
    let tmp = tempfile::tempdir().unwrap();
    let m = tmp.path().join("m.json");
    let s = tmp.path().join("m.minisig");
    let t = tmp.path().join("higgs.tar.gz");
    std::fs::write(&m, br#"{"schema":1,"version":"9.9.9"}"#).unwrap();
    std::fs::write(&s, b"untrusted comment: x\nAAAA\n").unwrap();
    std::fs::write(&t, b"x").unwrap();
    let out = higgs(&[
        "node",
        "self-update",
        "--dry-run",
        "--prefix",
        tmp.path().to_str().unwrap(),
        "--tarball",
        t.to_str().unwrap(),
        "--manifest",
        m.to_str().unwrap(),
        "--manifest-sig",
        s.to_str().unwrap(),
    ]);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("HG082"));
}

// ---- argument surface ----------------------------------------------------

#[test]
fn self_update_without_a_source_explains_it_needs_one() {
    let tmp = tempfile::tempdir().unwrap();
    let out = higgs(&[
        "node",
        "self-update",
        "--prefix",
        tmp.path().to_str().unwrap(),
    ]);
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("--tarball"),
        "should name the source flags: {stderr}"
    );
}

#[test]
fn self_update_with_a_partial_source_still_needs_all_three() {
    // Only --tarball (no --manifest/--manifest-sig) is not a complete source.
    let tmp = tempfile::tempdir().unwrap();
    let t = tmp.path().join("a.tar.gz");
    std::fs::write(&t, b"x").unwrap();
    let out = higgs(&[
        "node",
        "self-update",
        "--prefix",
        tmp.path().to_str().unwrap(),
        "--tarball",
        t.to_str().unwrap(),
    ]);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("--manifest"));
}

#[test]
fn self_update_allow_downgrade_still_fails_closed_on_an_unverifiable_signature() {
    // --allow-downgrade parses and threads through, but an unverifiable signature still
    // refuses at verification (HG082) before eligibility.
    let tmp = tempfile::tempdir().unwrap();
    let m = tmp.path().join("m.json");
    let s = tmp.path().join("m.minisig");
    let t = tmp.path().join("a.tar.gz");
    std::fs::write(&m, br#"{"schema":1,"version":"0.0.1"}"#).unwrap();
    std::fs::write(&s, b"untrusted comment: x\nAAAA\n").unwrap();
    std::fs::write(&t, b"x").unwrap();
    let out = higgs(&[
        "node",
        "self-update",
        "--allow-downgrade",
        "--prefix",
        tmp.path().to_str().unwrap(),
        "--tarball",
        t.to_str().unwrap(),
        "--manifest",
        m.to_str().unwrap(),
        "--manifest-sig",
        s.to_str().unwrap(),
    ]);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("HG082"));
}

#[test]
fn self_update_prune_removes_only_the_extra_versions() {
    // The real (non-dry-run) prune path: keeps current + rollback target, removes the rest.
    let tmp = tempfile::tempdir().unwrap();
    let bin = install_layout(tmp.path(), &["0.8.0", "0.9.0", "1.0.0", "2.0.0"], "2.0.0");
    std::fs::write(
        bin.join(".update-trial"),
        br#"{"to":"v2.0.0","prev":"v1.0.0"}"#,
    )
    .unwrap();
    let out = higgs(&[
        "node",
        "self-update",
        "--prune",
        "--prefix",
        tmp.path().to_str().unwrap(),
    ]);
    assert!(out.status.success(), "prune failed: {out:?}");
    assert!(
        !bin.join("v0.8.0").exists() && !bin.join("v0.9.0").exists(),
        "extras pruned"
    );
    assert!(
        bin.join("v1.0.0").exists() && bin.join("v2.0.0").exists(),
        "current+prev kept"
    );
}

#[test]
fn self_update_rejects_an_unknown_flag() {
    let out = higgs(&["node", "self-update", "--wat"]);
    assert!(!out.status.success());
    // The unrecognized token is NEVER echoed (it could be a capability); only usage is shown.
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("unrecognized argument"), "{stderr}");
    assert!(
        !stderr.contains("--wat"),
        "the token must not be echoed: {stderr}"
    );
}

#[test]
fn self_update_without_prefix_needs_a_real_install_layout() {
    // With no --prefix the bin dir is derived from current_exe; the test binary is not in
    // a `bin/v<semver>/higgs` layout, so it must ask for an explicit --prefix.
    let out = higgs(&["node", "self-update", "--rollback"]);
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("--prefix") || stderr.contains("install dir"),
        "should ask for --prefix: {stderr}"
    );
}

#[test]
fn self_update_rejects_an_empty_prefix() {
    let out = higgs(&["node", "self-update", "--rollback", "--prefix", ""]);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("must not be empty"));
}

#[test]
fn self_update_rejects_an_empty_manifest_path() {
    let out = higgs(&["node", "self-update", "--manifest", ""]);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("must not be empty"));
}

#[test]
fn self_update_rejects_a_flag_used_as_a_value() {
    // `--tarball --dry-run` must error (a value is missing), not read a file
    // literally named `--dry-run`.
    let out = higgs(&["node", "self-update", "--tarball", "--dry-run"]);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("a value is missing"));
}

#[test]
fn self_update_rejects_a_dotdot_prefix() {
    // A `..`-containing --prefix is refused (it is absolutized but kept verbatim, and a
    // `..` in a service/install path misdirects the ancestry validation).
    let out = higgs(&[
        "node",
        "self-update",
        "--rollback",
        "--prefix",
        "/opt/x/../y",
    ]);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("`..`"));
}

#[test]
fn self_update_rejects_rollback_and_prune_together() {
    let out = higgs(&["node", "self-update", "--rollback", "--prune"]);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("one at a time"));
}

// ---- rollback (no signature) ---------------------------------------------

#[test]
fn self_update_rollback_repoints_current() {
    let tmp = tempfile::tempdir().unwrap();
    let bin = install_layout(tmp.path(), &["1.0.0", "2.0.0"], "2.0.0");
    // A trial marker recording v1.0.0 as the rollback target.
    std::fs::write(
        bin.join(".update-trial"),
        br#"{"to":"v2.0.0","prev":"v1.0.0"}"#,
    )
    .unwrap();
    let out = higgs(&[
        "node",
        "self-update",
        "--rollback",
        "--prefix",
        tmp.path().to_str().unwrap(),
    ]);
    assert!(out.status.success(), "rollback failed: {out:?}");
    assert_eq!(current_target(&bin), Some("v1.0.0".to_string()));
    assert!(!bin.join(".update-trial").exists(), "trial cleared");
}

#[test]
fn self_update_dry_run_rollback_does_not_mutate() {
    let tmp = tempfile::tempdir().unwrap();
    let bin = install_layout(tmp.path(), &["1.0.0", "2.0.0"], "2.0.0");
    std::fs::write(
        bin.join(".update-trial"),
        br#"{"to":"v2.0.0","prev":"v1.0.0"}"#,
    )
    .unwrap();
    let out = higgs(&[
        "node",
        "self-update",
        "--dry-run",
        "--rollback",
        "--prefix",
        tmp.path().to_str().unwrap(),
    ]);
    assert!(out.status.success(), "dry-run rollback failed: {out:?}");
    assert!(String::from_utf8_lossy(&out.stdout).contains("would roll"));
    // current is UNCHANGED and the trial marker survives.
    assert_eq!(current_target(&bin), Some("v2.0.0".to_string()));
    assert!(bin.join(".update-trial").exists());
}

#[test]
fn self_update_dry_run_prune_does_not_delete() {
    let tmp = tempfile::tempdir().unwrap();
    let bin = install_layout(tmp.path(), &["0.9.0", "1.0.0", "2.0.0"], "2.0.0");
    std::fs::write(
        bin.join(".update-trial"),
        br#"{"to":"v2.0.0","prev":"v1.0.0"}"#,
    )
    .unwrap();
    let out = higgs(&[
        "node",
        "self-update",
        "--dry-run",
        "--prune",
        "--prefix",
        tmp.path().to_str().unwrap(),
    ]);
    assert!(out.status.success(), "dry-run prune failed: {out:?}");
    assert!(String::from_utf8_lossy(&out.stdout).contains("would prune"));
    // Nothing was actually removed.
    assert!(bin.join("v0.9.0").exists(), "dry-run must not delete");
}

#[test]
fn self_update_rollback_refuses_a_stale_trial() {
    // current -> v3 (a manual install) but a leftover trial names v2 → `--rollback` must
    // refuse to undo the manual repair.
    let tmp = tempfile::tempdir().unwrap();
    let bin = install_layout(tmp.path(), &["1.0.0", "3.0.0"], "3.0.0");
    std::fs::write(
        bin.join(".update-trial"),
        br#"{"to":"v2.0.0","prev":"v1.0.0"}"#,
    )
    .unwrap();
    let out = higgs(&[
        "node",
        "self-update",
        "--rollback",
        "--prefix",
        tmp.path().to_str().unwrap(),
    ]);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("stale"));
    // current is untouched.
    assert_eq!(current_target(&bin), Some("v3.0.0".to_string()));
}

#[test]
fn self_update_prune_refuses_a_non_install_dir() {
    // A --prefix whose bin/ has no `current` symlink is not a managed install — prune
    // must refuse (guards against pruning an unrelated dir's `v…` entries).
    let tmp = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(tmp.path().join("bin/v1.2.3")).unwrap(); // no current symlink
    let out = higgs(&[
        "node",
        "self-update",
        "--prune",
        "--prefix",
        tmp.path().to_str().unwrap(),
    ]);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("not a higgs install"));
    assert!(tmp.path().join("bin/v1.2.3").exists(), "nothing deleted");
}

#[test]
fn self_update_rollback_without_a_previous_fails() {
    let tmp = tempfile::tempdir().unwrap();
    install_layout(tmp.path(), &["1.0.0"], "1.0.0"); // no trial marker
    let out = higgs(&[
        "node",
        "self-update",
        "--rollback",
        "--prefix",
        tmp.path().to_str().unwrap(),
    ]);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("HG087"));
}

// ---- prune (no signature) ------------------------------------------------

#[test]
fn self_update_prune_keeps_current_and_rollback_target() {
    let tmp = tempfile::tempdir().unwrap();
    let bin = install_layout(tmp.path(), &["0.9.0", "1.0.0", "2.0.0", "3.0.0"], "2.0.0");
    std::fs::write(
        bin.join(".update-trial"),
        br#"{"to":"v2.0.0","prev":"v1.0.0"}"#,
    )
    .unwrap();
    let out = higgs(&[
        "node",
        "self-update",
        "--prune",
        "--prefix",
        tmp.path().to_str().unwrap(),
    ]);
    assert!(out.status.success(), "prune failed: {out:?}");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("v0.9.0"));
    assert!(stdout.contains("v3.0.0"));
    // Kept: current (v2.0.0) + rollback target (v1.0.0).
    assert!(bin.join("v1.0.0").exists());
    assert!(bin.join("v2.0.0").exists());
    // Pruned: the rest.
    assert!(!bin.join("v0.9.0").exists());
    assert!(!bin.join("v3.0.0").exists());
    assert_eq!(current_target(&bin), Some("v2.0.0".to_string()));
}
