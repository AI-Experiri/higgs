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
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "endpoint.key is not 32 bytes",
        )
    })?;
    Ok(SecretKey::from_bytes(&arr))
}

/// Generate and persist a new secret with a temp-file + atomic-publish protocol, so
/// concurrent first-starts never expose a partial key and never clobber each other.
/// We write the full key to a unique private temp file (complete + flushed), then
/// `hard_link` it to `endpoint.key` (atomic; fails `AlreadyExists` if a peer already
/// published, and only ever exposes the complete file), then remove the temp. If a peer
/// won the race, we adopt THAT key — all processes agree on one stable id.
fn create_secret(path: &Path) -> std::io::Result<SecretKey> {
    let sk = SecretKey::generate();
    let dir = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
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
///
/// Default is `presets::N0` (public relays + DNS discovery) for WAN/NAT reach. Setting
/// `HIGGS_IROH_LOCAL=1` switches to a relay-disabled, LAN-local bind (no public relay or
/// DNS dependency) — used by the cross-process integration test so it stays hermetic;
/// peers connect via the direct addresses carried in the pairing ticket.
pub async fn bind_endpoint(sk: SecretKey) -> Result<Endpoint, iroh::endpoint::BindError> {
    let builder = if std::env::var_os("HIGGS_IROH_LOCAL").is_some() {
        Endpoint::builder(iroh::endpoint::presets::Minimal).relay_mode(iroh::RelayMode::Disabled)
    } else {
        Endpoint::builder(iroh::endpoint::presets::N0)
    };
    builder
        .secret_key(sk)
        .alpns(vec![ALPN.to_vec()])
        .bind()
        .await
}

#[cfg(test)]
#[path = "identity_tests.rs"]
mod tests;
