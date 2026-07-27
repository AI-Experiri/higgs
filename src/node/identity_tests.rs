use super::*;
use tempfile::TempDir;

#[test]
fn secret_roundtrips_to_stable_id() {
    let dir = std::env::temp_dir().join("higgs-key-roundtrip-test");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("endpoint.key");
    let _ = std::fs::remove_file(&path);

    let sk1 = load_or_create_secret(&path).unwrap(); // generates + persists
    let sk2 = load_or_create_secret(&path).unwrap(); // reads the same file
    assert_eq!(sk1.public(), sk2.public(), "id stable across loads");
    let _ = std::fs::remove_file(&path);
}

#[cfg(unix)]
#[test]
fn generated_key_file_is_private() {
    use std::os::unix::fs::PermissionsExt;
    let dir = std::env::temp_dir().join("higgs-key-perms-test");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("endpoint.key");
    let _ = std::fs::remove_file(&path);
    load_or_create_secret(&path).unwrap();
    let mode = std::fs::metadata(&path).unwrap().permissions().mode();
    assert_eq!(mode & 0o777, 0o600, "endpoint.key must be owner-only");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn create_adopts_existing_key_on_race() {
    // Simulate losing the create race: the file already holds a key, so
    // create_secret must return THAT key, not the freshly generated one.
    let dir = std::env::temp_dir().join("higgs-key-race-test");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("endpoint.key");
    let _ = std::fs::remove_file(&path);

    let winner = SecretKey::generate();
    write_secret_exclusive(&path, &winner.to_bytes()).unwrap();
    let adopted = create_secret(&path).unwrap();
    assert_eq!(
        adopted.public(),
        winner.public(),
        "must adopt the winner's key"
    );
    let _ = std::fs::remove_file(&path);
}

#[test]
fn corrupt_key_file_is_rejected() {
    let dir = std::env::temp_dir().join("higgs-key-corrupt-test");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("endpoint.key");
    std::fs::write(&path, b"too short").unwrap();
    let err = load_or_create_secret(&path).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    let _ = std::fs::remove_file(&path);
}

/// An over-long key file (more than 32 bytes) also fails `parse_secret`'s `try_from`
/// (identity.rs:23-30) — `InvalidData`, never silently re-minted. Covers the size-mismatch
/// branch from the other side of the boundary (the existing test is the too-short case).
#[test]
fn oversized_key_file_is_rejected() {
    let dir = TempDir::new().expect("tmp");
    let path = dir.path().join("endpoint.key");
    std::fs::write(&path, vec![0u8; 33]).expect("write oversized key");
    let err = load_or_create_secret(&path).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
}

/// A genuine (non-`NotFound`) read error must propagate, NOT take the create path
/// (identity.rs:17-18) — regenerating over a transient read failure would change this
/// node's stable identity. A directory at `path` makes `std::fs::read` fail with a kind
/// other than `NotFound`, exercising the `Err(e) => Err(e)` arm.
#[test]
fn read_error_propagates_does_not_regenerate() {
    let dir = TempDir::new().expect("tmp");
    // The key "path" is itself a directory → read() errors with a non-NotFound kind.
    let path = dir.path().join("endpoint.key");
    std::fs::create_dir(&path).expect("make a dir where the key file would be");
    let err = load_or_create_secret(&path).unwrap_err();
    assert_ne!(
        err.kind(),
        std::io::ErrorKind::NotFound,
        "a directory read is not NotFound — it must surface, not regenerate"
    );
    // The directory is untouched (no key file was minted over the read error).
    assert!(path.is_dir(), "create path must not have run");
}

/// `bind_endpoint` in LAN-local mode (`HIGGS_IROH_LOCAL=1`) binds a relay-disabled endpoint
/// with our ALPN and the supplied secret's stable id (identity.rs:89-100, the local branch).
/// Hermetic — no relay/DNS — and asserts the endpoint id matches the secret's public key.
// The env lock is held across the bind/close awaits to serialize HIGGS_IROH_LOCAL with other
// env-mutating tests; this is a current-thread #[tokio::test] so there's no Send/deadlock hazard.
#[allow(clippy::await_holding_lock)]
#[tokio::test]
async fn bind_endpoint_local_mode_binds_with_stable_id() {
    let _env = crate::TEST_ENV_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let prev = std::env::var_os("HIGGS_IROH_LOCAL");
    // SAFETY: serialized by TEST_ENV_LOCK; restored before the lock releases.
    unsafe { std::env::set_var("HIGGS_IROH_LOCAL", "1") };
    struct Restore(Option<std::ffi::OsString>);
    impl Drop for Restore {
        fn drop(&mut self) {
            // SAFETY: still under TEST_ENV_LOCK.
            unsafe {
                match self.0.take() {
                    Some(v) => std::env::set_var("HIGGS_IROH_LOCAL", v),
                    None => std::env::remove_var("HIGGS_IROH_LOCAL"),
                }
            }
        }
    }
    let _restore = Restore(prev);

    let sk = SecretKey::generate();
    let want_id = sk.public();
    let ep = bind_endpoint(sk).await.expect("bind local endpoint");
    assert_eq!(
        ep.id(),
        want_id,
        "endpoint id is derived from the supplied secret (stable)"
    );
    ep.close().await;
}
