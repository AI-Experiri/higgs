
use super::*;

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
