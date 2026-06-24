//! Surface A auth — the machine allowlist (pairings.json) + one-time pairing tokens.
//! Owned INSIDE the crate (the crate's own serde); no common/engine/jigglebot import,
//! preserving the one-way dependency rule (DESIGN-remote.md §7 fix #10).

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// On-disk shape of `pairings.json`: each paired node's EndpointId (canonical
/// string) → an optional human label.
#[derive(Debug, Default, Serialize, Deserialize)]
struct PairingsFile {
    nodes: BTreeMap<String, Option<String>>,
}

/// The Surface A allowlist: paired `EndpointId`s the hub admits. Backed by
/// `pairings.json`; mutations persist immediately (small file, infrequent writes).
pub struct Allowlist {
    path: PathBuf,
    file: PairingsFile,
}

impl Allowlist {
    /// Load from `path`; a missing file is an empty allowlist. Other read errors
    /// (permissions, corruption) fail loudly rather than silently emptying it.
    pub fn load(path: &Path) -> std::io::Result<Self> {
        let file = match std::fs::read(path) {
            Ok(bytes) => serde_json::from_slice(&bytes)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => PairingsFile::default(),
            Err(e) => return Err(e),
        };
        Ok(Self {
            path: path.to_path_buf(),
            file,
        })
    }

    /// Is this EndpointId (canonical string) paired?
    pub fn contains(&self, id: &str) -> bool {
        self.file.nodes.contains_key(id)
    }

    /// The persisted human label for a paired id, if any.
    pub fn label(&self, id: &str) -> Option<String> {
        self.file.nodes.get(id).cloned().flatten()
    }

    /// Every paired EndpointId (ascending) — used to seed the hub's fleet view with known
    /// nodes at startup so they appear (disconnected) before they reconnect.
    pub fn ids(&self) -> Vec<String> {
        self.file.nodes.keys().cloned().collect()
    }

    /// Add a paired id (idempotent); persists.
    pub fn add(&mut self, id: String, label: Option<String>) -> std::io::Result<()> {
        self.mutate(|nodes| {
            nodes.insert(id, label);
        })
    }

    /// Remove a paired id (revocation); persists.
    pub fn remove(&mut self, id: &str) -> std::io::Result<()> {
        self.mutate(|nodes| {
            nodes.remove(id);
        })
    }

    /// Apply `f`, persist, and only keep the change if the save succeeded. On a
    /// persistence error (disk full, permissions) the in-memory map is rolled back
    /// to the persisted state, so authorization decisions never diverge from disk.
    fn mutate(
        &mut self,
        f: impl FnOnce(&mut BTreeMap<String, Option<String>>),
    ) -> std::io::Result<()> {
        let snapshot = self.file.nodes.clone();
        f(&mut self.file.nodes);
        if let Err(e) = self.save() {
            self.file.nodes = snapshot;
            return Err(e);
        }
        Ok(())
    }

    /// Number of paired nodes (for `link status`).
    pub fn len(&self) -> usize {
        self.file.nodes.len()
    }

    /// True when no nodes are paired.
    pub fn is_empty(&self) -> bool {
        self.file.nodes.is_empty()
    }

    /// Persist atomically: write to a temp file, fsync it, then rename over the live
    /// `pairings.json`. A failure mid-write leaves the existing file untouched (rename
    /// is atomic on the same filesystem), so disk never holds partial/empty JSON.
    fn save(&self) -> std::io::Result<()> {
        use std::io::Write;
        let bytes = serde_json::to_vec_pretty(&self.file)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let dir = self
            .path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        let tmp = dir.join(format!(
            ".pairings.json.tmp.{}.{:x}",
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
        std::fs::rename(&tmp, &self.path).inspect_err(|_| {
            let _ = std::fs::remove_file(&tmp);
        })
    }
}

/// Why a pairing token was rejected (maps to HG022 at the gate).
#[derive(Debug, PartialEq, Eq)]
pub enum TokenError {
    /// Token never minted, or already burned (single-use).
    UnknownOrUsed,
    /// Token minted but its TTL has elapsed.
    Expired,
}

/// In-memory mint/burn store for one-time pairing tokens. Not persisted: a token is a
/// **single-use** bootstrap credential (effectively non-expiring — see
/// [`crate::remote::PAIRING_TOKEN_TTL_MS`]); a hub restart simply invalidates pending ones, so
/// the operator re-mints. Once a node is admitted the token is burned and the pairing persists
/// via the keypair + allowlist until retire — no token thereafter. Single home for token state.
#[derive(Default)]
pub struct PairingTokens {
    /// token string → expiry epoch-ms.
    live: HashMap<String, u64>,
}

impl PairingTokens {
    pub fn new() -> Self {
        Self::default()
    }

    /// Mint a single-use token valid for `ttl_ms` from `now_ms`. Returns `htk_<hex>`.
    pub fn mint(&mut self, now_ms: u64, ttl_ms: u64) -> String {
        use rand::RngCore;
        let mut raw = [0u8; 16];
        rand::rngs::OsRng.fill_bytes(&mut raw);
        let tok = format!("htk_{}", hex_encode(&raw));
        self.live.insert(tok.clone(), now_ms.saturating_add(ttl_ms));
        tok
    }

    /// Validate a presented token WITHOUT consuming it. `Ok(())` means it may admit
    /// the peer; the caller [`burn`](Self::burn)s it only after the pairing persists,
    /// so a failed save leaves the token usable for a retry.
    pub fn validate(&self, token: &str, now_ms: u64) -> Result<(), TokenError> {
        match self.live.get(token) {
            None => Err(TokenError::UnknownOrUsed),
            Some(&expiry) if now_ms > expiry => Err(TokenError::Expired),
            Some(_) => Ok(()),
        }
    }

    /// Consume a token (single-use). Call after the pairing is durably persisted.
    pub fn burn(&mut self, token: &str) {
        self.live.remove(token);
    }

    /// Validate and burn in one step (convenience for callers with no persistence
    /// step between the two, e.g. tests).
    pub fn validate_and_burn(&mut self, token: &str, now_ms: u64) -> Result<(), TokenError> {
        self.validate(token, now_ms)?;
        self.burn(token);
        Ok(())
    }
}

fn hex_encode(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(s, "{b:02x}");
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id_str() -> String {
        "aa".repeat(32)
    }

    #[test]
    fn add_contains_remove_roundtrip() {
        let dir = std::env::temp_dir().join("higgs-allow-roundtrip-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("pairings.json");
        let _ = std::fs::remove_file(&path);

        let mut allow = Allowlist::load(&path).unwrap();
        assert!(!allow.contains(&id_str()));
        allow.add(id_str(), Some("studio-mac".into())).unwrap();
        assert!(allow.contains(&id_str()));
        assert_eq!(allow.len(), 1);

        // persisted: a fresh load sees it
        let reloaded = Allowlist::load(&path).unwrap();
        assert!(reloaded.contains(&id_str()));

        // ids() enumerates paired ids (for fleet seeding).
        allow.add(id_str(), None).unwrap();
        assert_eq!(allow.ids(), vec![id_str()], "ids lists the paired node");

        allow.remove(&id_str()).unwrap();
        assert!(!allow.contains(&id_str()));
        assert!(allow.is_empty());
        assert!(allow.ids().is_empty(), "ids empty after removal");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn load_missing_file_is_empty() {
        let path = std::env::temp_dir().join("higgs-allow-absent-xyz.json");
        let _ = std::fs::remove_file(&path);
        let allow = Allowlist::load(&path).unwrap();
        assert!(allow.is_empty());
    }

    #[test]
    fn failed_save_rolls_back_in_memory() {
        // Parent dir does not exist → load sees NotFound (empty), but save's write
        // fails, so add() must return Err AND leave the allowlist unchanged.
        let path = std::env::temp_dir()
            .join("higgs-allow-rollback-missing-dir")
            .join("pairings.json");
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
        let mut allow = Allowlist::load(&path).unwrap();
        let err = allow.add(id_str(), None).unwrap_err();
        assert!(err.kind() == std::io::ErrorKind::NotFound || err.raw_os_error().is_some());
        assert!(
            !allow.contains(&id_str()),
            "rolled back: not paired in memory"
        );
        assert!(allow.is_empty());
    }

    #[test]
    fn token_mint_validate_burn() {
        let mut tokens = PairingTokens::new();
        let tok = tokens.mint(now_ms(), 600_000);
        assert!(tok.starts_with("htk_"));
        assert!(tokens.validate_and_burn(&tok, now_ms()).is_ok());
        // single-use: second validation fails
        assert_eq!(
            tokens.validate_and_burn(&tok, now_ms()),
            Err(TokenError::UnknownOrUsed)
        );
    }

    #[test]
    fn validate_does_not_consume_until_burned() {
        let mut tokens = PairingTokens::new();
        let tok = tokens.mint(now_ms(), 600_000);
        // validate is idempotent — the token survives repeated peeks (e.g. a failed save)
        assert!(tokens.validate(&tok, now_ms()).is_ok());
        assert!(tokens.validate(&tok, now_ms()).is_ok());
        tokens.burn(&tok);
        // now consumed
        assert_eq!(
            tokens.validate(&tok, now_ms()),
            Err(TokenError::UnknownOrUsed)
        );
    }

    #[test]
    fn token_expires() {
        let mut tokens = PairingTokens::new();
        let minted_at = 1_000_000u64;
        let tok = tokens.mint(minted_at, 600_000);
        assert_eq!(
            tokens.validate_and_burn(&tok, minted_at + 660_000),
            Err(TokenError::Expired)
        );
    }

    #[test]
    fn unknown_token_is_rejected() {
        let mut tokens = PairingTokens::new();
        assert_eq!(
            tokens.validate_and_burn("htk_nope", now_ms()),
            Err(TokenError::UnknownOrUsed)
        );
    }

    fn now_ms() -> u64 {
        1_750_000_000_000
    }
}
