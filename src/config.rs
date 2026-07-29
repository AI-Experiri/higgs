//! Per-instance config persisted at `~/.higgs/config.json` (override the home dir with
//! `HIGGS_HOME`). Holds the friendly NAME every higgs instance (hub or node) carries.
//! Everything here is PUBLIC info — never a secret (the identity secret lives in `endpoint.key`,
//! `0600`), so a leaked `config.json` grants nothing on its own.
//!
//! Single home for the naming state, mirroring `auth.rs` (the allowlist) so the home directory
//! has one obvious file per concern: `endpoint.key`, `pairings.json`, `api_keys.json`,
//! `models/`, and this `config.json`. A node also records here the hubs it has paired with, so a
//! restarted node reconnects to its hub by itself (no re-pair) — see [`SavedHub`]. All of it is
//! still PUBLIC info; the reconnect is authenticated by the node's keypair against the hub
//! allowlist, not by anything stored here.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::diagnostic::HiggsError;
use crate::worker::engine::LoadParams;

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

/// A hub a node has paired with, saved so a restarted node reconnects without re-pairing.
/// `hub_id` is the hub's `EndpointId` (the dedup key + what `default_hub` points at, learned
/// authoritatively from the hub's HELLO result); `ticket` is the dialable
/// [`EndpointTicket`](iroh_tickets::endpoint) string; `label` is the hub's friendly name;
/// `last_used_ms` records the last successful connect (newest = the natural default).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SavedHub {
    pub hub_id: String,
    pub ticket: String,
    pub label: String,
    pub last_used_ms: u64,
}

/// A per-model record persisted in `config.json`, keyed by HuggingFace repo id. Records the
/// parameters a model was last successfully loaded with — so the UI can DISPLAY "loaded with
/// ctx=…, gpu_layers=…" and a future autoload can reload it with the same params — replacing the
/// old scan-time load-probe (loadability is now learned only at actual load; see the
/// probe-removal change). Still PUBLIC info (just load knobs + a timestamp).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ModelRecord {
    /// The load params of the last successful load, or `None` if this model has only ever carried
    /// flags (never been loaded on this instance). Read tolerantly via [`lenient_load`] so both the
    /// current engine-tagged `LoadParams` shape AND a pre-umbrella tag-less flat record load.
    #[serde(default, deserialize_with = "lenient_load")]
    pub load: Option<LoadParams>,
    /// Unix-ms timestamp of the last successful load (`0` = unknown / never loaded).
    #[serde(default)]
    pub last_loaded_ms: u64,
}

/// Deserialize `ModelRecord.load` tolerantly. Accepts the current engine-tagged
/// `LoadParams` shape (`{"engine":"LlamaCpp", …}`) AND an older **tag-less flat**
/// object (records written before the `LoadParams` became an engine umbrella),
/// and degrades any unrecognizable value to `None` rather than failing the whole
/// `config.json` load (which would lose the instance name + saved hubs).
fn lenient_load<'de, D>(d: D) -> Result<Option<LoadParams>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let v = Option::<serde_json::Value>::deserialize(d)?;
    Ok(v.and_then(|val| {
        serde_json::from_value::<LoadParams>(val.clone())
            .ok()
            .or_else(|| {
                serde_json::from_value::<crate::worker::engine::llamacpp::params::LlamaCppParams>(
                    val,
                )
                .ok()
                .map(LoadParams::llamacpp)
            })
    }))
}

/// On-disk shape of `config.json`. `name` is always present; `hubs`/`default_hub` are node-side
/// only (default-empty for a pure hub, whose file is just `{"name":"hub-…"}`); `models` records
/// per-model load info on any instance. Every field is `#[serde(default)]`, so a name-only file
/// written by an older build still loads.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InstanceConfig {
    /// The friendly instance name, e.g. `node-a1b2c3d4(studio-mac)`. Empty only on a config
    /// that predates naming; [`name_or_init`] fills it in on first run.
    #[serde(default)]
    pub name: String,
    /// Hubs this node has paired with (most-recent connects toward the end). Empty for a hub.
    #[serde(default)]
    pub hubs: Vec<SavedHub>,
    /// The `hub_id` a bare `higgs --node` dials. Empty/none for a hub or an unpaired node.
    #[serde(default)]
    pub default_hub: Option<String>,
    /// Per-model load records, keyed by HuggingFace repo id. `BTreeMap` for stable,
    /// diff-friendly JSON ordering. Empty by default; populated on the first successful load.
    #[serde(default)]
    pub models: BTreeMap<String, ModelRecord>,
    /// Extra CORS origins allowed on the HTTP surface beyond the built-in
    /// loopback/tauri set (exact match against the request `Origin`, e.g.
    /// `"https://tools.example"`). Applied at server start — restart to change.
    /// CORS only protects BROWSER clients; non-browser access is gated by API
    /// keys, not this list.
    #[serde(default)]
    pub cors_origins: Vec<String>,
}

impl InstanceConfig {
    /// Insert or replace a hub (keyed by `hub_id`) and make it the default. Called after a
    /// successful HELLO, so the latest ticket/label/`last_used_ms` always win and a bare
    /// `higgs --node` thereafter reconnects to it. Idempotent on the id (never duplicates).
    pub fn remember_hub(&mut self, hub: SavedHub) {
        let id = hub.hub_id.clone();
        self.hubs.retain(|h| h.hub_id != id);
        self.hubs.push(hub);
        self.default_hub = Some(id);
    }

    /// The default hub's saved entry, if the `default_hub` id still resolves to one.
    pub fn default_saved_hub(&self) -> Option<&SavedHub> {
        let id = self.default_hub.as_ref()?;
        self.hubs.iter().find(|h| &h.hub_id == id)
    }

    /// Find a saved hub by exact `label`, else by `hub_id` prefix (a short 8-char id works) —
    /// for `higgs --node --hub <label|id>`. Label is tried first so a human name is unambiguous.
    pub fn find_hub(&self, needle: &str) -> Option<&SavedHub> {
        self.hubs
            .iter()
            .find(|h| h.label == needle)
            .or_else(|| self.hubs.iter().find(|h| h.hub_id.starts_with(needle)))
    }

    /// Drop a saved hub by `hub_id` (after `higgs node leave` retires it hub-side). If it was
    /// the default, promote the most-recently-used remaining hub to default (else clear it), so
    /// a bare `higgs --node` still has a sensible target. Returns whether a hub was removed.
    pub fn remove_hub(&mut self, hub_id: &str) -> bool {
        let before = self.hubs.len();
        self.hubs.retain(|h| h.hub_id != hub_id);
        let removed = self.hubs.len() != before;
        if self.default_hub.as_deref() == Some(hub_id) {
            self.default_hub = self
                .hubs
                .iter()
                .max_by_key(|h| h.last_used_ms)
                .map(|h| h.hub_id.clone());
        }
        removed
    }

    /// Record a successful load of model `id` with `params` at `now_ms`, replacing any prior
    /// record's load info. Called after a load succeeds so the UI can show what the model was
    /// loaded with and a future autoload can reload it the same way. `now_ms` is passed in (this
    /// module stays clock-free) — the caller stamps it.
    pub fn record_load(&mut self, id: &str, params: LoadParams, now_ms: u64) {
        let rec = self.models.entry(id.to_string()).or_default();
        rec.load = Some(params);
        rec.last_loaded_ms = now_ms;
    }

    /// The persisted per-model record for `id`, if one exists.
    pub fn model_record(&self, id: &str) -> Option<&ModelRecord> {
        self.models.get(id)
    }
}

impl InstanceConfig {
    /// Load from `path`; a missing file is the default (empty) config. Other read errors
    /// (permissions, corruption) fail loudly rather than silently resetting the instance name.
    pub fn load(path: &Path) -> std::io::Result<Self> {
        // The return type stays io::Result (many `?`/`.ok()` callers); the HG code
        // rides INSIDE the io::Error so its Display surfaces it at the CLI/startup.
        match std::fs::read(path) {
            // Present but unparseable JSON = corruption (HG041) — distinct from an
            // I/O error so the fix ("delete to reset") is unambiguous.
            Ok(bytes) => serde_json::from_slice(&bytes).map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    HiggsError::StoreCorrupted {
                        store: "config".into(),
                        path: path.display().to_string(),
                        detail: e.to_string(),
                    },
                )
            }),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            // A real read I/O error (permissions, …) = HG040.
            Err(e) => {
                let kind = e.kind();
                Err(std::io::Error::new(
                    kind,
                    HiggsError::PersistenceFailed {
                        store: "config".into(),
                        path: path.display().to_string(),
                        source: e,
                    },
                ))
            }
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
        // Ensure the target directory exists (a no-op for the real `~/.higgs`, which
        // `ensure_home` creates; needed when the path points at a fresh dir, e.g. a
        // per-instance test home). Mirrors the `models.json` store's flush.
        std::fs::create_dir_all(dir)?;
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
pub fn config_path() -> std::io::Result<PathBuf> {
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
#[path = "config_tests.rs"]
mod tests;
