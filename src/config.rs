//! Per-instance config persisted at `~/.higgs/config.json` (override the home dir with
//! `HIGGS_HOME`). Holds the friendly NAME every higgs instance (hub or node) carries.
//! Everything here is PUBLIC info — never a secret (the identity secret lives in `endpoint.key`,
//! `0600`), so a leaked `config.json` grants nothing on its own.
//!
//! Single home for the naming state, mirroring `auth.rs` (the allowlist) so the home directory
//! has one obvious file per concern: `endpoint.key`, `pairings.json`, `api_keys.json`,
//! `models/`, and this `config.json`. A node's saved hubs (for restart reconnect) extend this
//! same struct in a later unit — `#[serde(default)]` keeps a name-only file forward-compatible.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// The role an instance runs as. Fixes the friendly-name prefix (`hub-…` vs `node-…`) so a
/// machine is identifiable at a glance. A hub that was first a node keeps its original prefix
/// until renamed (the name is generated once, then reused) — an accepted, rare edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    Hub,
    Node,
}

impl Role {
    /// The lowercase prefix used in a friendly name.
    pub fn as_str(self) -> &'static str {
        match self {
            Role::Hub => "hub",
            Role::Node => "node",
        }
    }
}

/// On-disk shape of `config.json`. Today just the friendly `name`; a node's saved hubs extend
/// this struct in a later unit (each new field `#[serde(default)]`, so a name-only file written
/// now still loads then).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InstanceConfig {
    /// The friendly instance name, e.g. `node-a1b2c3d4(studio-mac)`. Empty only on a config
    /// that predates naming; [`name_or_init`] fills it in on first run.
    #[serde(default)]
    pub name: String,
}

impl InstanceConfig {
    /// Load from `path`; a missing file is the default (empty) config. Other read errors
    /// (permissions, corruption) fail loudly rather than silently resetting the instance name.
    pub fn load(path: &Path) -> std::io::Result<Self> {
        match std::fs::read(path) {
            Ok(bytes) => serde_json::from_slice(&bytes)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(e),
        }
    }

    /// Persist atomically: write a temp file, fsync it, then rename over the live
    /// `config.json`. A failure mid-write leaves the existing file untouched (rename is atomic
    /// on the same filesystem), so disk never holds partial/empty JSON. Mirrors
    /// [`crate::auth::Allowlist`]'s save.
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        use std::io::Write;
        let bytes = serde_json::to_vec_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let dir = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        let tmp = dir.join(format!(
            ".config.json.tmp.{}.{:x}",
            std::process::id(),
            rand::random::<u64>()
        ));
        let write_tmp = || -> std::io::Result<()> {
            let mut f = std::fs::File::create(&tmp)?;
            f.write_all(&bytes)?;
            f.sync_all()
        };
        if let Err(e) = write_tmp() {
            let _ = std::fs::remove_file(&tmp);
            return Err(e);
        }
        std::fs::rename(&tmp, path).inspect_err(|_| {
            let _ = std::fs::remove_file(&tmp);
        })
    }
}

/// `~/.higgs/config.json` (creating the home dir if absent). Honors `HIGGS_HOME`.
fn config_path() -> std::io::Result<PathBuf> {
    Ok(crate::home::ensure_home()?.join("config.json"))
}

/// Build the friendly name `<role>-<eid8>(<hostname>)`, e.g. `node-a1b2c3d4(studio-mac)` or
/// `hub-3f9a2b1c(studio-mac)`. `<eid8>` is the first 8 chars of the EndpointId (its canonical
/// z-base-32 string). A shorter id is used whole; an empty hostname drops the parens
/// (`<role>-<eid8>`), so the name is always non-empty as long as the id is.
pub fn friendly_name(role: Role, endpoint_id: &str, hostname: &str) -> String {
    let eid8: String = endpoint_id.chars().take(8).collect();
    if hostname.is_empty() {
        format!("{}-{}", role.as_str(), eid8)
    } else {
        format!("{}-{}({})", role.as_str(), eid8, hostname)
    }
}

/// The persisted instance name, generating + persisting one on first run. Reads/writes
/// `~/.higgs/config.json` via [`config_path`]. Idempotent: once a name exists it is returned
/// verbatim — so an operator-renamed instance keeps its chosen name across restarts, and a
/// node that was named before becoming a hub keeps its first prefix (the accepted cross-role
/// edge). Only the FIRST call (empty name) writes.
pub fn name_or_init(role: Role, endpoint_id: &str, hostname: &str) -> std::io::Result<String> {
    let path = config_path()?;
    let mut cfg = InstanceConfig::load(&path)?;
    if cfg.name.is_empty() {
        cfg.name = friendly_name(role, endpoint_id, hostname);
        cfg.save(&path)?;
    }
    Ok(cfg.name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn friendly_name_uses_role_eid8_and_host() {
        assert_eq!(
            friendly_name(Role::Node, "a1b2c3d4e5f6", "studio-mac"),
            "node-a1b2c3d4(studio-mac)"
        );
        assert_eq!(
            friendly_name(Role::Hub, "3f9a2b1c0000", "studio-mac"),
            "hub-3f9a2b1c(studio-mac)"
        );
    }

    #[test]
    fn friendly_name_drops_parens_when_host_empty() {
        assert_eq!(friendly_name(Role::Node, "a1b2c3d4e5", ""), "node-a1b2c3d4");
    }

    #[test]
    fn friendly_name_uses_short_id_whole() {
        // An id shorter than 8 chars is taken as-is (never panics on a short slice).
        assert_eq!(friendly_name(Role::Hub, "abc", "m"), "hub-abc(m)");
    }

    #[test]
    fn config_save_load_roundtrip() {
        let dir = std::env::temp_dir().join("higgs-config-roundtrip-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.json");
        let _ = std::fs::remove_file(&path);

        let cfg = InstanceConfig {
            name: "node-aa(box)".into(),
        };
        cfg.save(&path).unwrap();

        let back = InstanceConfig::load(&path).unwrap();
        assert_eq!(back.name, "node-aa(box)");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn load_missing_file_is_default() {
        let path = std::env::temp_dir().join("higgs-config-absent-xyz.json");
        let _ = std::fs::remove_file(&path);
        let cfg = InstanceConfig::load(&path).unwrap();
        assert!(cfg.name.is_empty());
    }

    #[test]
    fn load_tolerates_unknown_future_fields() {
        // A config.json written by a NEWER build (with saved-hub fields) must still load on
        // this build — unknown keys are ignored, so an upgrade/downgrade never wipes the name.
        let dir = std::env::temp_dir().join("higgs-config-future-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.json");
        std::fs::write(
            &path,
            br#"{"name":"hub-3f9a2b1c(srv)","hubs":[{"ticket":"t"}],"default_hub":"x"}"#,
        )
        .unwrap();
        let cfg = InstanceConfig::load(&path).unwrap();
        assert_eq!(cfg.name, "hub-3f9a2b1c(srv)");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn name_or_init_generates_then_reuses() {
        // Serialize with other HIGGS_HOME-mutating tests; restore the prior value after.
        let _env = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let prev = std::env::var_os("HIGGS_HOME");
        let tmp = std::env::temp_dir().join("higgs-name-or-init-test");
        let _ = std::fs::remove_dir_all(&tmp);
        // SAFETY: serialized by TEST_ENV_LOCK; restored below.
        unsafe { std::env::set_var("HIGGS_HOME", &tmp) };

        // First call generates + persists.
        let first = name_or_init(Role::Node, "a1b2c3d4e5f6", "box").unwrap();
        assert_eq!(first, "node-a1b2c3d4(box)");
        // The name landed in config.json.
        let on_disk = InstanceConfig::load(&tmp.join("config.json")).unwrap();
        assert_eq!(on_disk.name, "node-a1b2c3d4(box)");

        // Second call reuses it verbatim — even with a DIFFERENT role/id/host (idempotent).
        let second = name_or_init(Role::Hub, "zzzzzzzz", "other").unwrap();
        assert_eq!(
            second, "node-a1b2c3d4(box)",
            "existing name is never overwritten"
        );

        // SAFETY: still under the lock.
        unsafe {
            match prev {
                Some(v) => std::env::set_var("HIGGS_HOME", v),
                None => std::env::remove_var("HIGGS_HOME"),
            }
        }
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
