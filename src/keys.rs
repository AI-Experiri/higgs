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
    /// type for the key-management control surface (the `keys` list/mint/revoke
    /// ops; formerly `/api/higgs/keys`), so the frontend gets the const-object form.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(rename_all = "lowercase")]
    pub enum Scope {
        /// `POST /v1/chat/completions`.
        Chat,
        /// Model listing/details (`GET /v1/models`, and the `models` control-op).
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
    /// Unix-ms when the key was minted. `None` for keys from a pre-timestamp
    /// store (serde default) — the UI renders that as unknown. (No magic-0
    /// sentinel: absence is modeled as `None`, per the crate's no-magic-int
    /// rule.)
    #[serde(default)]
    pub created_at_ms: Option<u64>,
    /// Unix-ms of the last successful authorization with this key, `None` if
    /// never used. Updated IN MEMORY by the auth middleware (throttled — see
    /// `Higgs::touch_api_key`); the on-disk copy is refreshed only when an
    /// explicit mint/revoke persists the live store (carrying whatever stamps
    /// it holds then). So a restart shows the stamps as of the LAST mint/
    /// revoke — "never" if none happened since the key was used — key IDENTITY
    /// is the durable part of the file, usage is best-effort history.
    #[serde(default)]
    pub last_used_ms: Option<u64>,
    /// Internal, in-memory-only key: minted at runtime by an in-process embedder
    /// (jigglebot) to authenticate ITSELF to the embedded higgs over the normal
    /// bearer path. `true` ⇒ this key is NEVER persisted to `api_keys.json`
    /// ([`ApiKeys::save`] filters it) and NEVER shown in the key-management list
    /// ([`ApiKeys::visible`]) — it is not a user credential. It authorizes exactly
    /// like any other key. Absent/`false` for every user-created key.
    ///
    /// `skip_deserializing`: this flag is set SOLELY by [`ApiKeys::add_internal`]
    /// at runtime — it is an in-memory-only concept and must NEVER be honored from
    /// disk. Without this, a hand-edited/tampered `api_keys.json` carrying
    /// `"hidden": true` on an Admin key would load a credential that `authorizes()`
    /// (counts all keys) accepts but `visible()` hides from the management list and
    /// `remove_label()` refuses to delete — a stealth backdoor the owner cannot
    /// see or revoke. Ignoring `hidden` on load makes every persisted key visible
    /// and manageable. Safe because `save()` never writes a hidden key, so no
    /// legitimate on-disk store ever carries `hidden: true` to round-trip.
    #[serde(
        default,
        skip_serializing_if = "std::ops::Not::not",
        skip_deserializing
    )]
    pub hidden: bool,
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

/// Current wall-clock time in unix-ms (0 if the clock is before the epoch).
fn now_unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
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
        // Persist only user-created keys — an internal in-memory key
        // ([`Self::add_internal`]) must never touch disk (it's regenerated each
        // boot by the embedder). `skip_serializing_if` on `hidden` also elides the
        // flag from the persisted user keys.
        let persistable = ApiKeys {
            keys: self.keys.iter().filter(|k| !k.hidden).cloned().collect(),
        };
        let tmp = path.with_extension("json.tmp");
        std::fs::write(
            &tmp,
            serde_json::to_vec_pretty(&persistable).expect("api keys serialize"),
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
            created_at_ms: Some(now_unix_ms()),
            last_used_ms: None,
            hidden: false,
        });
        sha256
    }

    /// Register an INTERNAL, in-memory-only key for a caller-provided plaintext
    /// `token` (stored only as its digest).
    ///
    /// For an in-process embedder (jigglebot) to authenticate itself to the
    /// embedded higgs over the normal bearer path — no special auth branch. The
    /// embedder OWNS the token value (a fixed dev token shared with the browser
    /// proxy, or a fresh random one in production) and presents it as a bearer.
    /// The key is flagged [`ApiKey::hidden`], so it is never written to disk
    /// ([`Self::save`]) and never listed ([`Self::visible`]).
    pub fn add_internal(&mut self, token: &str, label: String, scopes: Vec<Scope>) {
        // [codex r1] An empty/whitespace token is NEVER a credential. Registering one
        // would arm auth with a trivially-guessable key whose digest is `hash_token("")`,
        // so any caller whose presented bearer trims to "" would authorize. A caller that
        // fumbles its token value (empty env/config) must not create one; refuse it so a
        // bad token leaves auth OFF rather than trivially-open. (Not reachable over the
        // `/v1` bearer path — header normalization keeps a client from presenting an empty
        // bearer — so this is defense-in-depth, proved at the keys layer in keys_tests.rs.)
        if token.trim().is_empty() {
            tracing::warn!(
                "higgs: ignoring an internal-token registration with an empty token (no credential armed)"
            );
            return;
        }
        let sha256 = hash_token(token);
        // Replace any prior registration: an identical digest (idempotent re-add)
        // OR a previous HIDDEN key under the same logical label (rotation). Without
        // the label drop, re-registering a NEW internal token would leave the OLD
        // hidden key authorized too — it's not persisted, not listed, and not
        // removable via `remove_label`, so the stale bearer would live forever.
        // Only hidden keys are dropped by label, never a user's visible key that
        // happens to share it.
        self.keys
            .retain(|k| k.sha256 != sha256 && !(k.hidden && k.label == label));
        self.keys.push(ApiKey {
            sha256,
            label,
            scopes,
            created_at_ms: Some(now_unix_ms()),
            last_used_ms: None,
            hidden: true,
        });
    }

    /// Iterate only the VISIBLE (user-created, persisted) keys — the set the
    /// key-management API lists. Hidden internal keys ([`Self::add_internal`]) are
    /// excluded. Auth and the LAN guards use [`Self::iter`] (ALL keys) instead.
    pub fn visible(&self) -> impl Iterator<Item = &ApiKey> {
        self.keys.iter().filter(|k| !k.hidden)
    }

    /// Remove VISIBLE keys by label; returns how many were removed. HIDDEN
    /// internal keys ([`Self::add_internal`]) are NEVER removed by label — the
    /// embedder's in-memory token is not part of the user-facing key-management
    /// surface, so a key revoke (`Higgs::revoke_key`, formerly
    /// `DELETE /api/higgs/keys/{label}`) that happens to name it
    /// (e.g. `jigglebot (internal)`) is a no-op rather than a way to strand the
    /// embedder's own auth. Disk-loaded stores hold no hidden keys, so the CLI
    /// `keys remove` path is unaffected.
    pub fn remove_label(&mut self, label: &str) -> usize {
        let before = self.keys.len();
        self.keys.retain(|k| k.hidden || k.label != label);
        before - self.keys.len()
    }

    /// Does a presented bearer `token` authorize `required`? Constant-time digest compare
    /// across ALL keys (no early return on the first byte mismatch). An empty store always
    /// authorizes (auth disabled).
    pub fn authorizes(&self, token: &str, required: Scope) -> bool {
        self.keys.is_empty() || self.authorizing_sha(token, required).is_some()
    }

    /// The digest of the key that authorizes `token` for `required`, or `None`.
    /// Same constant-time scan as [`Self::authorizes`] (every key is compared,
    /// no early return) — the matched digest lets the auth middleware record
    /// last-used without a second lookup. An empty store returns `None` (the
    /// caller treats auth as disabled via [`Self::is_empty`]).
    pub fn authorizing_sha(&self, token: &str, required: Scope) -> Option<String> {
        let presented = hash_token(token);
        let presented = presented.as_bytes();
        let mut found: Option<String> = None;
        for k in &self.keys {
            // Constant-time digest equality; only a hex-length match can be equal.
            let matches =
                k.sha256.len() == presented.len() && k.sha256.as_bytes().ct_eq(presented).into();
            if matches && k.grants(required) && found.is_none() {
                found = Some(k.sha256.clone());
            }
        }
        found
    }

    /// Record a successful authorization on the key with `sha256`. MONOTONIC:
    /// `last_used_ms` only advances (a stale/reordered `now_ms` ≤ the stored
    /// value is a no-op), so a request whose clock read was overtaken by a
    /// concurrent one can't move the stamp backward. Returns whether anything
    /// changed (false for an unknown digest or a non-advancing timestamp).
    pub fn touch(&mut self, sha256: &str, now_ms: u64) -> bool {
        match self.keys.iter_mut().find(|k| k.sha256 == sha256) {
            Some(k) if k.last_used_ms.is_none_or(|t| now_ms > t) => {
                k.last_used_ms = Some(now_ms);
                true
            }
            _ => false,
        }
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
