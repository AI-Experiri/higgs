use super::*;

fn root() -> (tempfile::TempDir, std::path::PathBuf) {
    let home = tempfile::tempdir().expect("home");
    let models = home.path().join("models");
    std::fs::create_dir_all(&models).expect("models dir");
    (home, models)
}

#[test]
fn a_fresh_key_acquires_and_is_gone_after_drop() {
    let (_h, models) = root();
    let lock = DownloadLock::acquire(&models, "acme/m", "m.gguf").expect("fresh key acquires");
    let p = lock.path().to_path_buf();
    assert!(p.exists(), "lock file created on acquire");
    drop(lock);
    // The file itself persists (cheap sentinel; no per-transfer cleanup churn).
    // What matters is that the FLOCK is released so the next acquire succeeds.
    let re = DownloadLock::acquire(&models, "acme/m", "m.gguf").expect("re-acquire after drop");
    drop(re);
}

#[test]
fn a_second_acquire_of_a_held_key_refuses_with_hg090() {
    // The whole point of this module: while ANY process on the machine holds
    // the lock, another `acquire` for the same key returns HG090
    // (`DownloadInFlight`) — no bytes move, no ledger row created, no
    // heuristic. Two flock attempts on the same file from two open fds
    // conflict deterministically.
    let (_h, models) = root();
    let held = DownloadLock::acquire(&models, "acme/m", "m.gguf").expect("first acquire");
    let err = DownloadLock::acquire(&models, "acme/m", "m.gguf")
        .expect_err("second acquire on the held key must refuse");
    assert!(
        matches!(err, HiggsError::DownloadInFlight { .. }),
        "flock contention surfaces as HG090: {err}"
    );
    // Different keys are independent.
    let _other =
        DownloadLock::acquire(&models, "acme/m", "other.gguf").expect("distinct key untouched");
    drop(held);
    let _re = DownloadLock::acquire(&models, "acme/m", "m.gguf")
        .expect("key freed once the first guard drops");
}

#[test]
fn an_uncreatable_locks_dir_is_an_io_error_not_contention() {
    // A regular FILE squatting the `.download-locks` path makes
    // `create_dir_all` fail. That is a filesystem fault (HG034), NOT
    // "someone is downloading" — reporting HG090 here would send the
    // operator hunting a phantom duplicate on a perms/disk problem.
    let (_h, models) = root();
    std::fs::write(locks_dir(&models), b"squatter").expect("squat the dir path");
    let err = DownloadLock::acquire(&models, "acme/m", "m.gguf")
        .expect_err("dir creation failure must refuse");
    assert!(
        matches!(err, HiggsError::HubFileWrite { .. }),
        "I/O failure is HG034, never HG090 contention: {err}"
    );
}

#[test]
fn an_unopenable_lock_file_is_an_io_error_not_contention() {
    // A DIRECTORY squatting the lock-file path makes the read/write open
    // fail — same contract: HG034, not phantom contention.
    let (_h, models) = root();
    let path = lock_path(&models, "acme/m", "m.gguf");
    std::fs::create_dir_all(&path).expect("squat the lock path with a dir");
    let err = DownloadLock::acquire(&models, "acme/m", "m.gguf")
        .expect_err("unopenable lock file must refuse");
    assert!(
        matches!(err, HiggsError::HubFileWrite { .. }),
        "I/O failure is HG034, never HG090 contention: {err}"
    );
    // The same squatted path makes the read-side probe conservative:
    // it cannot open the file, so it cannot PROVE staleness.
    assert!(
        !is_key_stale(&models, "acme/m", "m.gguf"),
        "unprobeable key is never proven stale"
    );
}

#[test]
fn different_repos_and_files_do_not_collide() {
    // Deterministic per-key path — sanity: distinct keys produce distinct
    // lock files, and each is independently acquirable.
    let (_h, models) = root();
    let a = lock_path(&models, "acme/m", "m.gguf");
    let b = lock_path(&models, "acme/n", "m.gguf");
    let c = lock_path(&models, "acme/m", "n.gguf");
    assert_ne!(a, b);
    assert_ne!(a, c);
    assert_ne!(b, c);
    let ga = DownloadLock::acquire(&models, "acme/m", "m.gguf").unwrap();
    let gb = DownloadLock::acquire(&models, "acme/n", "m.gguf").unwrap();
    let gc = DownloadLock::acquire(&models, "acme/m", "n.gguf").unwrap();
    drop((ga, gb, gc));
}

#[test]
fn a_maximally_long_key_still_produces_a_name_max_safe_lock_path() {
    // The dest_path budget allows 255-byte org + 255-byte model + 218-byte
    // file → up to 728+ bytes of raw key. Any plain-text encoding blows
    // NAME_MAX (255) before `open`. The sha256-hex encoding is fixed at
    // 64 chars regardless of input — proves the encoder handles the worst
    // case by acquiring and asserting the filename length.
    let (_h, models) = root();
    let long_repo = format!("{}/{}", "a".repeat(255), "b".repeat(255));
    let long_file = format!("{}.gguf", "c".repeat(213));
    let lock = DownloadLock::acquire(&models, &long_repo, &long_file)
        .expect("worst-case key acquires (hash-encoded)");
    let name = lock
        .path()
        .file_name()
        .and_then(|s| s.to_str())
        .expect("utf8");
    assert!(
        name.len() <= 255,
        "lock filename must fit NAME_MAX: {} bytes",
        name.len()
    );
    assert!(name.ends_with(".lock"));
    // sha256 hex = 64 chars + ".lock" (5) = 69.
    assert_eq!(name.len(), 69, "sha256-hex is fixed length");
}

#[test]
fn distinct_valid_keys_that_would_collide_under_a_separator_scheme_hash_apart() {
    // The pre-hash scheme (`repo.replace('/', "--") + "__" + file`) had a
    // real collision: `(a/b--c, m.gguf)` and `(a--b/c, m.gguf)` both flatten
    // to `a--b--c__m.gguf`. With sha256(repo || 0x00 || file), the null
    // separator can't appear in either input (dest_path bans it), so the
    // hashed byte sequences differ → distinct lock files → both can hold
    // simultaneously without a false HG090.
    let (_h, models) = root();
    let a = lock_path(&models, "a/b--c", "m.gguf");
    let b = lock_path(&models, "a--b/c", "m.gguf");
    assert_ne!(
        a, b,
        "the two keys must hash to distinct lock files, or the pre-hash \
         collision returns"
    );
    // Both are simultaneously acquirable.
    let ga = DownloadLock::acquire(&models, "a/b--c", "m.gguf").unwrap();
    let gb = DownloadLock::acquire(&models, "a--b/c", "m.gguf").unwrap();
    drop((ga, gb));
}

#[test]
fn locks_dir_is_dot_hidden_inside_models_root() {
    // The scanner walks `<org>/<model>/` DIRECTORIES; a dot-prefixed sibling
    // is never a model (same discipline the ledger `.downloads.json` uses).
    let (_h, models) = root();
    let d = locks_dir(&models);
    assert_eq!(d, models.join(".download-locks"));
    assert!(
        d.file_name()
            .and_then(|s| s.to_str())
            .unwrap_or_default()
            .starts_with('.'),
        "hidden from the model scanner"
    );
}

#[test]
fn is_key_stale_only_returns_true_when_lock_file_exists_and_is_unheld() {
    // The read-side probe used by the ledger sweep. Semantics:
    //   (a) lock file does not exist   → NOT stale (missing-file is not
    //       proof; a fresh test row or bootstrap window would be false-swept).
    //   (b) lock file exists, unheld   → STALE (residue).
    //   (c) lock file exists, held     → NOT stale (live owner).
    let (_h, models) = root();
    // (a) no lock file at all
    assert!(
        !is_key_stale(&models, "acme/m", "m.gguf"),
        "missing lock file must NOT be treated as proof of staleness"
    );
    // (c) held by us
    let held = DownloadLock::acquire(&models, "acme/m", "m.gguf").expect("acquire");
    assert!(
        !is_key_stale(&models, "acme/m", "m.gguf"),
        "a lock actively held is NOT stale"
    );
    // (b) released → file persists → stale
    drop(held);
    assert!(
        is_key_stale(&models, "acme/m", "m.gguf"),
        "lock file exists but nobody holds it → stale (residue)"
    );
}
