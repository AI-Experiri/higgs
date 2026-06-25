//! Per-node unified model store: `~/.higgs/models.json` (`ProfileStore` default
//! impl + the GGUF-metadata cache + observed-perf record).
//!
//! Separate from `config.json` (which stays lean). One coherent record per model
//! id carries: a GGUF-metadata **cache** (keyed by `{path,size,mtime}` — the
//! on-disk GGUFs remain the source of truth), the durable saved **tuning**
//! profile (reused on the next load), and the durable observed **perf** (passive
//! tok/s). Hardware-specific ⇒ stored on the machine that produced it; the hub
//! pulls it over the scan RPC. Written atomically (temp → fsync → rename), `0600`.

use std::collections::BTreeMap;
use std::io;
use std::path::{Path, PathBuf};

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use crate::worker::engine::{LoadParams, SamplingParams};

use super::{BenchResult, ModelMeta, ProfileStore, ResourceBudget, TuneProvenance};

/// The on-disk shape of `models.json`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelsStore {
    /// One record per model id (HF `org/model`, `ollama/name:tag`, …).
    #[serde(default)]
    pub models: BTreeMap<String, ModelEntry>,
}

/// One model's record: cached metadata + saved tuning + observed perf. Each part
/// is independent and optional; vanished-file entries are retained so tuning/perf
/// survive a temporary move and re-attach when the model returns.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelEntry {
    /// Cached GGUF facts + the `{path,size,mtime}` cache key (a cache, not truth).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<ModelMetaCache>,
    /// The saved autotune profile, reused on the next load ("tune once").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tuning: Option<TuneRecord>,
    /// Observed passive perf (last/avg tok/s).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub perf: Option<ModelPerf>,
}

/// Cached GGUF metadata plus the key that validates it against the on-disk file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelMetaCache {
    /// The cached typed facts.
    pub meta: ModelMeta,
    /// Cache key: re-derive on a mismatch.
    pub key: MetaCacheKey,
}

/// `{path,size,mtime}` cache key for the GGUF-metadata cache.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MetaCacheKey {
    /// Absolute GGUF path.
    pub path: String,
    /// File size in bytes.
    pub size_bytes: u64,
    /// File mtime in unix-ms.
    pub mtime_ms: u64,
}

/// A saved tuning result — durable, reused on the next load.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TuneRecord {
    /// The saved load profile (engine umbrella).
    pub profile: LoadParams,
    /// The saved sampling profile (engine umbrella).
    pub sampling: SamplingParams,
    /// The caps it was derived within.
    pub budget: ResourceBudget,
    /// Where the values came from.
    pub provenance: TuneProvenance,
    /// Measured generation tok/s if a bench produced this (P2), else `None`.
    pub bench_tps: Option<f32>,
    /// Unix-ms when this profile was saved.
    pub tuned_at_ms: u64,
}

/// Observed passive performance (§7.1) — real decode timing, never synthetic.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct ModelPerf {
    /// Most-recent generation tok/s.
    pub last_gen_tps: f32,
    /// Most-recent prompt-processing tok/s.
    pub last_prompt_tps: f32,
    /// Most-recent time-to-first-token (ms).
    pub last_ttft_ms: f32,
    /// Running average generation tok/s.
    pub avg_gen_tps: f32,
    /// Running average prompt tok/s.
    pub avg_prompt_tps: f32,
    /// Number of samples folded into the averages.
    pub samples: u64,
    /// Unix-ms of the last sample.
    pub measured_at_ms: u64,
}

impl ModelPerf {
    /// Fold one decode sample into the record: refresh `last_*` and roll
    /// `avg' = (avg*n + x) / (n+1)`.
    fn record(&mut self, sample: BenchResult, now_ms: u64) {
        let n = self.samples as f32;
        self.avg_gen_tps = (self.avg_gen_tps * n + sample.gen_tps) / (n + 1.0);
        self.avg_prompt_tps = (self.avg_prompt_tps * n + sample.prompt_tps) / (n + 1.0);
        self.last_gen_tps = sample.gen_tps;
        self.last_prompt_tps = sample.prompt_tps;
        self.last_ttft_ms = sample.ttft_ms;
        self.samples += 1;
        self.measured_at_ms = now_ms;
    }
}

/// The default JSON-backed model store over `~/.higgs/models.json`. Interior
/// mutability (a `Mutex`) so `ProfileStore` / perf writes take `&self`.
pub struct JsonModelStore {
    path: PathBuf,
    inner: Mutex<ModelsStore>,
}

impl JsonModelStore {
    /// Open the store under `home` (`<home>/models.json`). A missing file is the
    /// empty store; a corrupt file is logged and treated as empty rather than
    /// failing (the GGUFs remain the source of truth; tuning/perf can be re-earned).
    pub fn open(home: &Path) -> io::Result<Self> {
        let path = home.join("models.json");
        let inner = match std::fs::read(&path) {
            Ok(bytes) => serde_json::from_slice::<ModelsStore>(&bytes).unwrap_or_else(|e| {
                tracing::warn!(error = %e, "higgs: models.json corrupt; starting empty");
                ModelsStore::default()
            }),
            Err(e) if e.kind() == io::ErrorKind::NotFound => ModelsStore::default(),
            Err(e) => return Err(e),
        };
        Ok(Self {
            path,
            inner: Mutex::new(inner),
        })
    }

    /// Open the store under the per-node higgs home (`~/.higgs` or `$HIGGS_HOME`).
    pub fn open_home() -> io::Result<Self> {
        Self::open(&crate::home::higgs_home())
    }

    /// The saved tuning record for `id`.
    pub fn tuning(&self, id: &str) -> Option<TuneRecord> {
        self.inner.lock().models.get(id)?.tuning.clone()
    }

    /// Persist (in memory) a tuning record for `id`.
    pub fn put_tuning(&self, id: &str, record: TuneRecord) {
        self.inner
            .lock()
            .models
            .entry(id.to_string())
            .or_default()
            .tuning = Some(record);
    }

    /// Update ONLY the saved load `profile` for `id` (e.g. a successful load with
    /// accepted/edited params), preserving an existing record's sampling / budget /
    /// provenance — or defaulting them when there is no record yet. This keeps the
    /// saved profile a plain load reuses in sync with the LAST accepted load, so an
    /// unload/reload doesn't silently revert to a stale tune suggestion.
    pub fn set_profile(&self, id: &str, profile: LoadParams, now_ms: u64) {
        let mut guard = self.inner.lock();
        let entry = guard.models.entry(id.to_string()).or_default();
        match entry.tuning.as_mut() {
            Some(rec) => {
                rec.profile = profile;
                rec.tuned_at_ms = now_ms;
            }
            None => {
                entry.tuning = Some(TuneRecord {
                    profile,
                    sampling: SamplingParams::default(),
                    budget: ResourceBudget::default(),
                    provenance: TuneProvenance::Heuristic,
                    bench_tps: None,
                    tuned_at_ms: now_ms,
                });
            }
        }
    }

    /// The observed perf for `id`.
    pub fn perf(&self, id: &str) -> Option<ModelPerf> {
        self.inner.lock().models.get(id)?.perf
    }

    /// Fold a decode sample into `id`'s observed perf (rolling average).
    pub fn record_perf(&self, id: &str, sample: BenchResult, now_ms: u64) {
        let mut guard = self.inner.lock();
        let entry = guard.models.entry(id.to_string()).or_default();
        let mut perf = entry.perf.unwrap_or_default();
        perf.record(sample, now_ms);
        entry.perf = Some(perf);
    }

    /// Cache GGUF metadata for `id` with its `{path,size,mtime}` key.
    pub fn put_meta(&self, id: &str, cache: ModelMetaCache) {
        self.inner
            .lock()
            .models
            .entry(id.to_string())
            .or_default()
            .meta = Some(cache);
    }

    /// Return the cached metadata for `id` IFF its cache key still matches the
    /// on-disk `{path,size,mtime}` — else `None` (a re-derivation is needed).
    pub fn meta_if_fresh(
        &self,
        id: &str,
        path: &str,
        size_bytes: u64,
        mtime_ms: u64,
    ) -> Option<ModelMeta> {
        let guard = self.inner.lock();
        let cache = guard.models.get(id)?.meta.as_ref()?;
        (cache.key.path == path
            && cache.key.size_bytes == size_bytes
            && cache.key.mtime_ms == mtime_ms)
            .then(|| cache.meta.clone())
    }

    /// A snapshot of the whole entry for `id` (used by the hub-scan summary).
    pub fn entry(&self, id: &str) -> Option<ModelEntry> {
        self.inner.lock().models.get(id).cloned()
    }

    /// Persist the store atomically: temp file → `sync_all` → rename, `0600`.
    pub fn flush(&self) -> io::Result<()> {
        use std::io::Write;
        use std::os::unix::fs::OpenOptionsExt;

        let bytes = {
            let guard = self.inner.lock();
            serde_json::to_vec_pretty(&*guard)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?
        };
        let dir = self
            .path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        std::fs::create_dir_all(dir)?;
        let tmp = dir.join(format!(
            ".models.json.tmp.{}.{:x}",
            std::process::id(),
            rand::random::<u64>()
        ));
        let res = (|| -> io::Result<()> {
            let mut f = std::fs::OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .mode(0o600)
                .open(&tmp)?;
            f.write_all(&bytes)?;
            f.sync_all()?;
            drop(f);
            std::fs::rename(&tmp, &self.path)
        })();
        if res.is_err() {
            let _ = std::fs::remove_file(&tmp);
        }
        res
    }
}

impl ProfileStore for JsonModelStore {
    fn tuning(&self, id: &str) -> Option<TuneRecord> {
        JsonModelStore::tuning(self, id)
    }
    fn put_tuning(&self, id: &str, record: TuneRecord) {
        JsonModelStore::put_tuning(self, id, record);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tune_record() -> TuneRecord {
        TuneRecord {
            profile: LoadParams::base(8192, u32::MAX, 8),
            sampling: SamplingParams::default(),
            budget: ResourceBudget::default(),
            provenance: TuneProvenance::Heuristic,
            bench_tps: None,
            tuned_at_ms: 100,
        }
    }

    fn perf_sample(gen: f32) -> BenchResult {
        BenchResult {
            gen_tps: gen,
            prompt_tps: gen * 4.0,
            ttft_ms: 50.0,
        }
    }

    fn meta_cache(path: &str, size: u64, mtime: u64) -> ModelMetaCache {
        ModelMetaCache {
            meta: ModelMeta {
                id: "org/m".into(),
                size_bytes: size,
                block_count: Some(32),
                ..Default::default()
            },
            key: MetaCacheKey {
                path: path.into(),
                size_bytes: size,
                mtime_ms: mtime,
            },
        }
    }

    #[test]
    fn store_roundtrips_tuning_and_perf_and_invalidates_meta() {
        let home = tempfile::tempdir().unwrap();
        let s = JsonModelStore::open(home.path()).unwrap();
        s.put_tuning("org/m", tune_record());
        s.record_perf("org/m", perf_sample(42.0), 1000);
        s.flush().unwrap();

        // Reopen: tuning + perf persisted durably.
        let s2 = JsonModelStore::open(home.path()).unwrap();
        assert!(s2.tuning("org/m").is_some());
        assert!((s2.perf("org/m").unwrap().avg_gen_tps - 42.0).abs() < 0.1);
        assert_eq!(s2.perf("org/m").unwrap().samples, 1);

        // Meta cache invalidates when {size,mtime} changes; tuning survives.
        s2.put_meta("org/m", meta_cache("/x.gguf", 100, 1));
        assert!(s2.meta_if_fresh("org/m", "/x.gguf", 100, 1).is_some());
        assert!(
            s2.meta_if_fresh("org/m", "/x.gguf", 100, 2).is_none(),
            "mtime changed → stale"
        );
        assert!(
            s2.meta_if_fresh("org/m", "/other.gguf", 100, 1).is_none(),
            "path changed → stale"
        );
        assert!(
            s2.tuning("org/m").is_some(),
            "tuning retained across meta change"
        );
    }

    #[test]
    fn set_profile_updates_profile_preserving_other_fields() {
        let home = tempfile::tempdir().unwrap();
        let s = JsonModelStore::open(home.path()).unwrap();
        // With an existing record, set_profile changes ONLY the profile + timestamp.
        let mut rec = tune_record();
        rec.provenance = TuneProvenance::Bench;
        rec.bench_tps = Some(9.0);
        s.put_tuning("org/m", rec);
        s.set_profile("org/m", LoadParams::base(1234, 0, 2), 99);
        let got = s.tuning("org/m").unwrap();
        assert_eq!(got.profile.ctx_len(), 1234, "profile updated");
        assert_eq!(got.profile.gpu_layers(), 0, "CPU-only accepted profile");
        assert_eq!(
            got.provenance,
            TuneProvenance::Bench,
            "provenance preserved"
        );
        assert_eq!(got.bench_tps, Some(9.0), "bench_tps preserved");
        assert_eq!(got.tuned_at_ms, 99);
        // With NO prior record, set_profile creates one with defaults.
        s.set_profile("org/new", LoadParams::base(2048, u32::MAX, 8), 50);
        let fresh = s.tuning("org/new").unwrap();
        assert_eq!(fresh.profile.ctx_len(), 2048);
        assert_eq!(fresh.provenance, TuneProvenance::Heuristic);
    }

    #[test]
    fn record_perf_rolls_running_average() {
        let home = tempfile::tempdir().unwrap();
        let s = JsonModelStore::open(home.path()).unwrap();
        s.record_perf("org/m", perf_sample(30.0), 1);
        s.record_perf("org/m", perf_sample(50.0), 2);
        let p = s.perf("org/m").unwrap();
        assert_eq!(p.samples, 2);
        assert!(
            (p.avg_gen_tps - 40.0).abs() < 0.01,
            "avg of 30 and 50 is 40"
        );
        assert_eq!(p.last_gen_tps, 50.0, "last is the most recent");
    }

    #[test]
    fn missing_file_is_empty_store() {
        let home = tempfile::tempdir().unwrap();
        let s = JsonModelStore::open(home.path()).unwrap();
        assert!(s.tuning("anything").is_none());
        assert!(s.perf("anything").is_none());
    }
}
