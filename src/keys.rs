//! API-key authentication for the HTTP surface (DESIGN-remote.md §6, P5).
//!
//! A bearer token (`Authorization: Bearer hgk_…`) is matched against `api_keys.json` in the
//! higgs home dir. Tokens are stored ONLY as their SHA-256 hex digest — the plaintext is
//! shown once at mint time and never persisted — and compared in constant time so a stored
//! digest can't leak via timing. Each key carries a set of [`Scope`]s; `Admin` is a superset.
//!
//! **Auth is opt-in:** an empty keystore means the surface is OPEN (the embedded in-process
//! host wants no gate). Auth turns on the moment the first key is added.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;

higgs_const_enum! {
    /// What a key is allowed to do. `Admin` implies every other scope. Wire
    /// type for the key-management surface (`/api/higgs/keys`), so the
    /// frontend gets the const-object form.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(rename_all = "lowercase")]
    pub enum Scope {
        /// `POST /v1/chat/completions`.
        Chat,
        /// Model listing/details (`GET /v1/models`, `GET /api/higgs/models*`).
        Models,
        /// Everything, including management (load/unload/settings/worker/nodes/keys).
        Admin,
    }
}

impl Scope {
    /// Parse a scope name; `None` for an unknown value.
    pub fn parse(s: &str) -> Option<Scope> {
        match s.trim().to_ascii_lowercase().as_str() {
            "chat" => Some(Scope::Chat),
            "models" => Some(Scope::Models),
            "admin" => Some(Scope::Admin),
            _ => None,
        }
    }
}

/// One stored key: its digest, a human label, and its granted scopes.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct ApiKey {
    /// Lowercase hex SHA-256 of the plaintext token. The plaintext is never stored.
    pub sha256: String,
    /// Human label (e.g. `"ci"`, `"laptop"`).
    pub label: String,
    /// Scopes this key grants.
    pub scopes: Vec<Scope>,
}

/// Redacted: only a short digest prefix ever reaches `Debug` output (logs,
/// panics, `{:?}` in errors). The full digest is not itself a credential, but
/// it is a stable key identifier — keep it out of anything greppable by habit.
impl std::fmt::Debug for ApiKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ApiKey")
            .field("label", &self.label)
            .field("scopes", &self.scopes)
            .field("sha256", &format_args!("{}…", self.sha256_prefix()))
            .finish()
    }
}

impl ApiKey {
    /// Does this key satisfy `required`? `Admin` satisfies anything.
    fn grants(&self, required: Scope) -> bool {
        self.scopes
            .iter()
            .any(|s| *s == Scope::Admin || *s == required)
    }

    /// The first 12 hex chars of the digest — the display identifier the CLI
    /// and the key-management API show (never the plaintext, never the full digest).
    pub fn sha256_prefix(&self) -> &str {
        &self.sha256[..self.sha256.len().min(12)]
    }
}

/// The keystore: the set of [`ApiKey`]s loaded from `api_keys.json`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ApiKeys {
    #[serde(default)]
    keys: Vec<ApiKey>,
}

/// Lowercase hex of a byte slice.
fn bytes_to_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// SHA-256 hex of `token` — the at-rest digest form.
pub fn hash_token(token: &str) -> String {
    bytes_to_hex(&Sha256::digest(token.as_bytes()))
}

/// Generate a fresh opaque token (`hgk_` + 32 url-safe-ish hex chars from 16 random bytes).
pub fn mint_token(rand_bytes: [u8; 16]) -> String {
    format!("hgk_{}", bytes_to_hex(&rand_bytes))
}

impl ApiKeys {
    /// Load the keystore from `path`; a missing file is an empty (auth-disabled) store.
    pub fn load(path: &Path) -> std::io::Result<Self> {
        match std::fs::read(path) {
            Ok(bytes) => serde_json::from_slice(&bytes)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(e),
        }
    }

    /// Persist the keystore to `path` (pretty JSON; atomic temp-file + rename). `rename(2)`
    /// replaces the destination in place on Unix, so repeated `keys add`/`remove` overwrite
    /// the existing file. (higgs targets Unix/macOS — the llama.cpp FFI build isn't
    /// Windows-portable — so the non-atomic-rename-replace platform is out of scope.)
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let tmp = path.with_extension("json.tmp");
        std::fs::write(
            &tmp,
            serde_json::to_vec_pretty(self).expect("api keys serialize"),
        )?;
        std::fs::rename(&tmp, path)
    }

    /// Whether auth is OFF (no keys configured) — the surface is open.
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &ApiKey> {
        self.keys.iter()
    }

    /// Add a key by its plaintext token (stored as a digest). Returns the digest.
    pub fn add(&mut self, token: &str, label: String, scopes: Vec<Scope>) -> String {
        let sha256 = hash_token(token);
        self.keys.retain(|k| k.sha256 != sha256); // replace an identical token
        self.keys.push(ApiKey {
            sha256: sha256.clone(),
            label,
            scopes,
        });
        sha256
    }

    /// Remove a key by label; returns how many were removed.
    pub fn remove_label(&mut self, label: &str) -> usize {
        let before = self.keys.len();
        self.keys.retain(|k| k.label != label);
        before - self.keys.len()
    }

    /// Does a presented bearer `token` authorize `required`? Constant-time digest compare
    /// across ALL keys (no early return on the first byte mismatch). An empty store always
    /// authorizes (auth disabled).
    pub fn authorizes(&self, token: &str, required: Scope) -> bool {
        if self.keys.is_empty() {
            return true;
        }
        let presented = hash_token(token);
        let presented = presented.as_bytes();
        let mut ok = false;
        for k in &self.keys {
            // Constant-time digest equality; only a hex-length match can be equal.
            let matches =
                k.sha256.len() == presented.len() && k.sha256.as_bytes().ct_eq(presented).into();
            if matches && k.grants(required) {
                ok = true;
            }
        }
        ok
    }
}

/// The default keystore path under the higgs home dir.
pub fn keys_path() -> std::io::Result<PathBuf> {
    Ok(crate::home::ensure_home()?.join("api_keys.json"))
}

/// Parse a comma-separated scope list (e.g. `"chat,models"`); unknown names error.
fn parse_scopes(s: &str) -> std::io::Result<Vec<Scope>> {
    s.split(',')
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .map(|p| {
            Scope::parse(p).ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!("unknown scope {p:?}"),
                )
            })
        })
        .collect()
}

/// Shown after a mutating `keys` subcommand: changes only take effect on (re)start.
const RESTART_NOTICE: &str = "↻ restart higgs for this to take effect on a running server.";

/// `higgs keys <add|list|remove>` — manage the API-key store (P5).
pub fn run_keys(args: &[String]) -> std::io::Result<()> {
    run_keys_at(&keys_path()?, args)
}

/// Core of [`run_keys`] against an explicit keystore `path` (so it's testable without env).
fn run_keys_at(path: &Path, args: &[String]) -> std::io::Result<()> {
    match args.first().map(String::as_str) {
        Some("add") => {
            let label = args.get(1).cloned().ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "usage: higgs keys add <label> [chat,models]",
                )
            })?;
            let scopes = parse_scopes(args.get(2).map(String::as_str).unwrap_or("chat,models"))?;
            if scopes.is_empty() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "at least one scope required",
                ));
            }
            let token = mint_token(rand::random());
            let mut keys = ApiKeys::load(path)?;
            keys.add(&token, label.clone(), scopes.clone());
            keys.save(path)?;
            println!("added key {label:?} with scopes {scopes:?}");
            println!("token (shown ONCE — store it now): {token}");
            println!("{RESTART_NOTICE}");
            Ok(())
        }
        Some("list") => {
            let keys = ApiKeys::load(path)?;
            if keys.is_empty() {
                println!("no API keys configured — the HTTP surface is OPEN");
            } else {
                for k in keys.iter() {
                    println!(
                        "{:<16} {:?}  sha256:{}…",
                        k.label,
                        k.scopes,
                        k.sha256_prefix()
                    );
                }
            }
            Ok(())
        }
        Some("remove") => {
            let label = args.get(1).cloned().ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "usage: higgs keys remove <label>",
                )
            })?;
            let mut keys = ApiKeys::load(path)?;
            let n = keys.remove_label(&label);
            keys.save(path)?;
            println!("removed {n} key(s) labeled {label:?}");
            println!("{RESTART_NOTICE}");
            Ok(())
        }
        other => {
            eprintln!("usage: higgs keys <add|list|remove> (got {other:?})");
            Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "unknown keys subcommand",
            ))
        }
    }
}

#[cfg(test)]
#[path = "keys_tests.rs"]
mod tests;
