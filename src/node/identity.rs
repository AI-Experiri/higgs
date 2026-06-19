//! iroh identity: a persisted ed25519 SecretKey → a stable EndpointId across restarts.

use std::path::Path;

use iroh::{Endpoint, SecretKey};

use crate::remote::ALPN;

/// Load the 32 secret bytes from `path`, or generate + persist them (chmod 0600).
/// The `EndpointId` derived from this key is stable across restarts.
pub fn load_or_create_secret(path: &Path) -> std::io::Result<SecretKey> {
    match std::fs::read(path) {
        Ok(bytes) => parse_secret(&bytes),
        // Only a genuinely-absent file takes the create path. A real read error
        // (permissions, ACL, I/O) must fail loudly — never regenerate over it,
        // which would change this node's stable identity.
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => create_secret(path),
        Err(e) => Err(e),
    }
}

fn parse_secret(bytes: &[u8]) -> std::io::Result<SecretKey> {
    let arr = <[u8; 32]>::try_from(bytes).map_err(|_| {
        // Corrupt key file: fail loudly rather than silently minting a new id
        // (which would silently drop the node out of every hub's allowlist).
        std::io::Error::new(std::io::ErrorKind::InvalidData, "endpoint.key is not 32 bytes")
    })?;
    Ok(SecretKey::from_bytes(&arr))
}

/// Generate and persist a new secret with a temp-file + atomic-publish protocol, so
/// concurrent first-starts never expose a partial key and never clobber each other:
///   1. write the full key to a unique private temp file (complete + flushed),
///   2. `hard_link` it to `endpoint.key` — atomic, fails `AlreadyExists` if a peer
///      already published, and only ever exposes the complete file,
///   3. remove the temp.
/// If a peer won the race, adopt THAT key — all processes agree on one stable id.
fn create_secret(path: &Path) -> std::io::Result<SecretKey> {
    let sk = SecretKey::generate();
    let dir = path.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."));
    let tmp = dir.join(format!(
        "endpoint.key.tmp.{}.{:x}",
        std::process::id(),
        rand::random::<u64>()
    ));
    write_secret_exclusive(&tmp, &sk.to_bytes())?;

    let published = std::fs::hard_link(&tmp, path);
    let _ = std::fs::remove_file(&tmp); // temp is redundant once linked (or on failure)
    match published {
        Ok(()) => Ok(sk),
        // A peer published first: adopt its complete, already-linked key.
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
            parse_secret(&std::fs::read(path)?)
        }
        Err(e) => Err(e),
    }
}

/// Exclusively create a private (`0600`) file and write the secret. `create_new`
/// fails with `AlreadyExists` if another process got there first (handled above),
/// and the `0600` mode applies at create time — never default-perms-then-chmod,
/// which would leave the key exposed in a crash/race window.
fn write_secret_exclusive(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    let mut opts = std::fs::OpenOptions::new();
    opts.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    let mut f = opts.open(path)?;
    f.write_all(bytes)?;
    f.flush()
}

/// Bind an iroh Endpoint with a stable id and our ALPN, ready to dial and accept.
pub async fn bind_endpoint(sk: SecretKey) -> Result<Endpoint, iroh::endpoint::BindError> {
    Endpoint::builder(iroh::endpoint::presets::N0)
        .secret_key(sk)
        .alpns(vec![ALPN.to_vec()])
        .bind()
        .await
}

#[cfg(test)]
mod tests {
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
        assert_eq!(adopted.public(), winner.public(), "must adopt the winner's key");
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
}
