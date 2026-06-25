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
    fn load_missing_file_is_default() {
        let path = std::env::temp_dir().join("higgs-config-absent-xyz.json");
        let _ = std::fs::remove_file(&path);
        let cfg = InstanceConfig::load(&path).unwrap();
        assert!(cfg.name.is_empty());
    }

    fn hub(id: &str, ticket: &str, label: &str, ts: u64) -> SavedHub {
        SavedHub {
            hub_id: id.into(),
            ticket: ticket.into(),
            label: label.into(),
            last_used_ms: ts,
        }
    }

    #[test]
    fn remember_hub_dedups_by_id_and_sets_default() {
        let mut cfg = InstanceConfig::default();
        cfg.remember_hub(hub("aaa111", "tkt-old", "hub-a(srv)", 1));
        assert_eq!(cfg.default_hub.as_deref(), Some("aaa111"));
        assert_eq!(cfg.hubs.len(), 1);

        // Re-remembering the SAME hub_id replaces (newer ticket/label/ts win), no duplicate.
        cfg.remember_hub(hub("aaa111", "tkt-new", "hub-a2(srv)", 5));
        assert_eq!(cfg.hubs.len(), 1, "deduped by hub_id");
        assert_eq!(cfg.hubs[0].ticket, "tkt-new");
        assert_eq!(cfg.hubs[0].label, "hub-a2(srv)");

        // A different hub is added and becomes the new default.
        cfg.remember_hub(hub("bbb222", "tkt-b", "hub-b(box)", 9));
        assert_eq!(cfg.hubs.len(), 2);
        assert_eq!(cfg.default_hub.as_deref(), Some("bbb222"));
        assert_eq!(cfg.default_saved_hub().unwrap().ticket, "tkt-b");
    }

    #[test]
    fn find_hub_by_label_then_id_prefix() {
        let mut cfg = InstanceConfig::default();
        cfg.remember_hub(hub("a1b2c3d4ffff", "tkt", "hub-a1b2c3d4(srv)", 1));
        // Exact label.
        assert_eq!(
            cfg.find_hub("hub-a1b2c3d4(srv)").map(|h| h.hub_id.as_str()),
            Some("a1b2c3d4ffff")
        );
        // Short id prefix.
        assert_eq!(
            cfg.find_hub("a1b2c3d4").map(|h| h.hub_id.as_str()),
            Some("a1b2c3d4ffff")
        );
        // No match.
        assert!(cfg.find_hub("nope").is_none());
    }

    #[test]
    fn remove_hub_drops_and_repoints_default() {
        let mut cfg = InstanceConfig::default();
        cfg.remember_hub(hub("aaa", "tkt-a", "hub-a", 1));
        cfg.remember_hub(hub("bbb", "tkt-b", "hub-b", 9)); // default = bbb (last remembered)
        assert_eq!(cfg.default_hub.as_deref(), Some("bbb"));

        // Removing the default promotes the most-recently-used remaining hub (aaa).
        assert!(cfg.remove_hub("bbb"));
        assert_eq!(cfg.hubs.len(), 1);
        assert_eq!(cfg.default_hub.as_deref(), Some("aaa"), "default repointed");

        // Removing the last hub clears the default; removing an unknown id is a no-op.
        assert!(!cfg.remove_hub("ghost"));
        assert!(cfg.remove_hub("aaa"));
        assert!(cfg.hubs.is_empty());
        assert!(
            cfg.default_hub.is_none(),
            "default cleared with no hubs left"
        );
    }

    #[test]
    fn default_saved_hub_none_when_id_dangles() {
        // A default_hub id with no matching entry resolves to None (defensive against a hand-
        // edited config), rather than panicking.
        let cfg = InstanceConfig {
            name: "node-x(box)".into(),
            hubs: vec![],
            default_hub: Some("ghost".into()),
            ..Default::default()
        };
        assert!(cfg.default_saved_hub().is_none());
    }

    #[test]
    fn config_roundtrips_hubs_and_default() {
        let dir = std::env::temp_dir().join("higgs-config-hubs-roundtrip-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.json");
        let _ = std::fs::remove_file(&path);

        let mut cfg = InstanceConfig {
            name: "node-aa(box)".into(),
            ..Default::default()
        };
        cfg.remember_hub(hub("h1", "tkt1", "hub-1(srv)", 7));
        cfg.save(&path).unwrap();

        let back = InstanceConfig::load(&path).unwrap();
        assert_eq!(back.name, "node-aa(box)");
        assert_eq!(back.hubs, cfg.hubs);
        assert_eq!(back.default_hub.as_deref(), Some("h1"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn record_load_stores_and_replaces_per_model() {
        let mut cfg = InstanceConfig::default();
        assert!(cfg.model_record("org/m").is_none());

        let p1 = LoadParams::base(4096, 99, 0);
        cfg.record_load("org/m", p1.clone(), 1000);
        let rec = cfg.model_record("org/m").expect("record present");
        assert_eq!(rec.load.as_ref().unwrap().ctx_len(), 4096);
        assert_eq!(rec.last_loaded_ms, 1000);

        // A second load of the SAME id replaces the load info (latest wins), no duplicate key.
        let p2 = LoadParams::base(8192, 0, 0);
        cfg.record_load("org/m", p2, 2000);
        assert_eq!(cfg.models.len(), 1, "keyed by id, no duplicate");
        let rec = cfg.model_record("org/m").unwrap();
        assert_eq!(rec.load.as_ref().unwrap().ctx_len(), 8192);
        assert_eq!(rec.last_loaded_ms, 2000);
    }

    /// Back-compat: a `config.json` written before `LoadParams` became an engine
    /// umbrella stored a tag-less flat `load` object. It must still deserialize
    /// (mapped to the `LlamaCpp` variant), and a corrupt `load` value must degrade
    /// to `None` rather than failing the whole config load.
    #[test]
    fn model_record_load_accepts_legacy_flat_and_degrades_garbage() {
        let legacy = r#"{"name":"node-x","models":{"org/m":{"load":{"ctx_len":2048,"gpu_layers":12,"threads":4},"last_loaded_ms":7}}}"#;
        let cfg: InstanceConfig = serde_json::from_str(legacy).unwrap();
        let rec = cfg.model_record("org/m").expect("legacy record present");
        assert_eq!(rec.load.as_ref().unwrap().ctx_len(), 2048);
        assert_eq!(rec.load.as_ref().unwrap().gpu_layers(), 12);
        assert_eq!(rec.last_loaded_ms, 7);

        // A non-object `load` value degrades to None (whole config still loads).
        let garbage = r#"{"name":"n","models":{"org/m":{"load":42,"last_loaded_ms":1}}}"#;
        let cfg2: InstanceConfig = serde_json::from_str(garbage).unwrap();
        assert!(cfg2.model_record("org/m").unwrap().load.is_none());
    }

    #[test]
    fn config_roundtrips_model_records() {
        let dir = std::env::temp_dir().join("higgs-config-models-roundtrip-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.json");
        let _ = std::fs::remove_file(&path);

        let mut cfg = InstanceConfig {
            name: "node-aa(box)".into(),
            ..Default::default()
        };
        cfg.record_load("org/m", LoadParams::base(2048, 12, 0), 42);
        cfg.save(&path).unwrap();

        let back = InstanceConfig::load(&path).unwrap();
        assert_eq!(back.models, cfg.models);
        assert_eq!(back.model_record("org/m").unwrap().last_loaded_ms, 42);
        assert_eq!(
            back.model_record("org/m")
                .unwrap()
                .load
                .as_ref()
                .unwrap()
                .ctx_len(),
            2048
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn load_tolerates_unknown_future_fields() {
        // A config.json written by a NEWER build (with extra keys) must still load on this
        // build — unknown keys are ignored, so an upgrade/downgrade never wipes the name. A pure
        // hub's minimal `{"name":…}` (no hubs/default_hub) also defaults in cleanly.
        let dir = std::env::temp_dir().join("higgs-config-future-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.json");
        std::fs::write(
            &path,
            br#"{"name":"hub-3f9a2b1c(srv)","some_future_field":true}"#,
        )
        .unwrap();
        let cfg = InstanceConfig::load(&path).unwrap();
        assert_eq!(cfg.name, "hub-3f9a2b1c(srv)");
        assert!(cfg.hubs.is_empty());
        assert!(cfg.default_hub.is_none());
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
