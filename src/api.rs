//! `Higgs` public facade and `HiggsConfig` — the host-facing API.
//!
//! One `Higgs` instance per host app. Typed facade over
//! [`Supervisor`](crate::supervisor::Supervisor): `Higgs` owns the facade-level
//! state (config, the load/unload lifecycle mutex, the inference admission gate,
//! the idle `last_activity` stamp) and delegates worker process management, RPC
//! correlation, and load-replay state to the supervisor. The host maps its own
//! config table onto [`HiggsConfig`].

use std::path::PathBuf;
use std::sync::Arc;

use serde_json::json;
use tokio::sync::{broadcast, mpsc};

use crate::diagnostic::HiggsError;
use crate::log_bus::{LogBus, LogLine, LogSource};
use crate::supervisor::{HiggsEvent, Supervisor};
use crate::worker::engine::LoadParams;
use crate::worker::models::HiggsModel;
use crate::worker::{M_LOAD, M_STATUS, M_UNLOAD};

// Submodules split out of this file (see api/README.md, api/DESIGN.md). `pub use`
// keeps every existing `crate::api::*` path resolving unchanged.
mod guards;
mod reaper;
mod types;

#[cfg(test)]
use guards::fits_in_memory;
use guards::validate_repo_id;
pub(crate) use guards::{guard_memory_headroom, path_within_roots};
use reaper::idle_reaper;
pub use types::*;

/// The in-process handle to the higgs runtime. One instance per host app.
///
/// The host-facing facade over the [`Supervisor`]. `Higgs` owns the facade-level
/// state — the live [`HiggsConfig`], the `lifecycle` mutex that serializes
/// load/unload/stop, the `inference_gate` admission semaphore, and the
/// `last_activity` stamp the idle reaper reads — while the [`Supervisor`] owns
/// the worker process, RPC correlation, and the load-replay state. Constructing
/// `Higgs` does not start the worker; call [`start`](Self::start) when the host
/// is ready to serve requests (a worker is spawned lazily on the first
/// [`load`](Self::load)).
pub struct Higgs {
    sup: Arc<Supervisor>,
    config: parking_lot::Mutex<HiggsConfig>,
    /// Serializes load/unload so spawn-on-load and kill-on-unload never
    /// interleave (protects last_load and the supervisor proc handle).
    lifecycle: tokio::sync::Mutex<()>,
    /// Inference admission gate — at most [`MAX_CONCURRENT_INFERENCE`] chat
    /// requests in flight. A `try_acquire_owned` failure on the chat path is a
    /// [`HiggsError::ServerBusy`] (HTTP 503); the owned permit rides the spawned
    /// generation task and releases on its completion (success/error/timeout).
    inference_gate: Arc<tokio::sync::Semaphore>,
    /// Admission gate for REMOTE (fleet-routed) chats — capped at
    /// [`MAX_CONCURRENT_INFERENCE`] like the local gate, but SEPARATE so a flood of remote
    /// requests can't grow hub/node tasks + streams unbounded, AND so it doesn't entangle the
    /// idle reaper's "acquire all LOCAL permits to unload" logic (remote traffic must not keep
    /// a local model resident — see `chat_stream`). `None` until a fleet is installed is not
    /// needed: the gate exists always; it's only used on the remote branch.
    remote_gate: Arc<tokio::sync::Semaphore>,
    /// Instant of the most recent chat request, stamped at the top of
    /// [`chat_stream`](Self::chat_stream). The idle reaper reads it to decide
    /// when the loaded model has been idle past [`IDLE_UNLOAD_TTL`]. A plain
    /// `parking_lot::Mutex` — read/written for a single `Instant` copy only,
    /// never held across `.await`.
    last_activity: parking_lot::Mutex<std::time::Instant>,
    /// Runtime "Log Incoming Tokens" toggle for the serve layer. When `true`, the
    /// chat path emits an extra INFO `higgs:`-target line per request carrying the
    /// (capped) flattened incoming prompt CONTENT so the Developer Logs show the
    /// actual prompt. This is the explicit OPT-IN that overrides the redact-by-
    /// default policy (no prompt content at info); default `false`, so the
    /// redaction policy is unchanged unless the user turns this on. A plain atomic
    /// — set/read in isolation, no critical section, never across `.await`.
    log_incoming_tokens: std::sync::atomic::AtomicBool,
    /// Runtime "Just-in-Time loading" toggle for the serve layer. When `true`
    /// (the default), a chat request for a scanned-but-unloaded model triggers an
    /// on-demand [`load`](Self::load) — swapping out any currently-resident model
    /// (higgs serves one model at a time) — instead of the `[HG003]` 404. When
    /// `false`, the chat path keeps the explicit-load behavior: an unloaded model
    /// is a 404. A plain atomic — set/read in isolation, no critical section,
    /// never across `.await`. Defaults to `true` (JIT on).
    jit_enabled: std::sync::atomic::AtomicBool,
    /// Runtime "Auto-unload idle models" toggle. When `true` (the default), the
    /// idle reaper unloads the loaded model after [`idle_ttl_minutes`](Self::idle_ttl_minutes)
    /// of no inference. When `false`, the reaper never unloads — a model stays
    /// resident until an explicit unload. Read by the reaper each tick, so a
    /// change takes effect without restart. A plain atomic — set/read in
    /// isolation, never across `.await`.
    auto_unload_idle: std::sync::atomic::AtomicBool,
    /// Runtime idle auto-unload TTL, in minutes. The idle reaper unloads the
    /// loaded model once the time since the last chat exceeds this. Seeded from
    /// [`IDLE_UNLOAD_TTL`] (5 minutes) and read by the reaper each tick, so a
    /// change takes effect without restart. A plain atomic — set/read in
    /// isolation, never across `.await`.
    idle_ttl_minutes: std::sync::atomic::AtomicU64,
    /// Per-load idle-TTL override, in minutes (HOST-SIDE only — never sent to the
    /// worker). `0` means "no override"; any non-zero value takes precedence over
    /// [`idle_ttl_minutes`](Self::idle_ttl_minutes) in the idle reaper for the
    /// CURRENTLY-loaded model. Set at load time from the load request and cleared
    /// on unload so a stale override never outlives its model. A plain atomic —
    /// set/read in isolation, never across `.await`.
    loaded_idle_ttl_override: std::sync::atomic::AtomicU64,
    /// Runtime "serving on/off" gate for the `/v1` inference surface. When `false`,
    /// the `/v1` inference endpoints return `[HG019]` → 503 while the
    /// `/api/higgs/*` control surface stays reachable so the user can re-enable.
    /// Defaults to `true` (serving on). A plain atomic — set/read in isolation,
    /// never across `.await`.
    serving_enabled: std::sync::atomic::AtomicBool,
    /// Model-support verdict cache, keyed by `(architecture, quant, engine_version)`
    /// — NOT per file. A probe runs once per distinct `(arch, quant)` for a given
    /// engine version; every model sharing that key inherits the cached verdict
    /// `(loadable, reason)`. In-memory only (higgs writes no files; persistence is
    /// deferred). A plain `parking_lot::Mutex` over a `HashMap` — held only for the
    /// map read/insert, never across `.await`.
    probe_cache: parking_lot::Mutex<std::collections::HashMap<SupportKey, SupportVerdict>>,
    /// Cached host device list gathered once via a transient sysinfo worker (see
    /// [`Supervisor::sysinfo`](crate::supervisor::Supervisor::sysinfo)). Hardware
    /// is static-ish, so it is gathered on first request and reused; `None` until
    /// the first successful gather. A failed gather leaves it `None` so a later
    /// request retries. A plain `parking_lot::Mutex` — held only for the
    /// read/insert, never across `.await`.
    device_cache: parking_lot::Mutex<Option<Vec<crate::system::GpuDevice>>>,
    /// The remote fleet (paired nodes + model routing), installed when the hub runs an
    /// iroh listener. `None` for a pure-local higgs. `/v1` chat for a remote-resident model
    /// routes through this instead of the local `Supervisor` (the two correlation domains
    /// stay separate, DESIGN-remote.md §2.3).
    fleet: parking_lot::Mutex<Option<Arc<crate::node::fleet::HubFleet>>>,
    /// API-key store gating the HTTP surface (P5). Default is empty = auth OFF (the embedded
    /// in-process host wants no gate); the standalone binary loads `api_keys.json` at startup.
    /// Swapped wholesale, so a reload is one atomic store.
    api_keys: parking_lot::Mutex<Arc<crate::keys::ApiKeys>>,
    /// The running hub (P3), when the server is in hub mode — used by the pairing API to mint
    /// node-join tokens. `None` for a pure-local / non-hub server.
    hub: parking_lot::Mutex<Option<Arc<crate::node::hub::Hub>>>,
}

impl Higgs {
    /// Construct the facade WITHOUT spawning the worker, owning a fresh
    /// [`LogBus`]. The serve-layer [`HiggsLogLayer`](crate::log_bus::HiggsLogLayer)
    /// is NOT wired to this internal bus — request events won't appear in the
    /// Developer Logs. Use [`with_log_bus`](Self::with_log_bus) when the caller
    /// installs the tracing layer.
    ///
    /// Call [`start`](Self::start) when the host is ready.
    pub fn new(config: HiggsConfig) -> Self {
        Self::with_log_bus(config, Arc::new(LogBus::new()))
    }

    /// Construct the facade WITHOUT spawning the worker, sharing `bus` with the
    /// caller's serve-layer [`HiggsLogLayer`](crate::log_bus::HiggsLogLayer).
    ///
    /// Worker stderr and captured serve-layer request events then flow into the
    /// same Developer-Log history+stream. The caller is responsible for
    /// installing `HiggsLogLayer::new(bus.clone())` on its tracing subscriber.
    pub fn with_log_bus(config: HiggsConfig, bus: Arc<LogBus>) -> Self {
        Self {
            sup: Arc::new(Supervisor::spawn(bus)),
            config: parking_lot::Mutex::new(config),
            lifecycle: tokio::sync::Mutex::new(()),
            inference_gate: Arc::new(tokio::sync::Semaphore::new(MAX_CONCURRENT_INFERENCE)),
            remote_gate: Arc::new(tokio::sync::Semaphore::new(MAX_CONCURRENT_INFERENCE)),
            last_activity: parking_lot::Mutex::new(std::time::Instant::now()),
            log_incoming_tokens: std::sync::atomic::AtomicBool::new(false),
            jit_enabled: std::sync::atomic::AtomicBool::new(true),
            auto_unload_idle: std::sync::atomic::AtomicBool::new(true),
            idle_ttl_minutes: std::sync::atomic::AtomicU64::new(IDLE_UNLOAD_TTL_MINUTES),
            loaded_idle_ttl_override: std::sync::atomic::AtomicU64::new(0),
            serving_enabled: std::sync::atomic::AtomicBool::new(true),
            probe_cache: parking_lot::Mutex::new(std::collections::HashMap::new()),
            device_cache: parking_lot::Mutex::new(None),
            fleet: parking_lot::Mutex::new(None),
            api_keys: parking_lot::Mutex::new(std::sync::Arc::new(crate::keys::ApiKeys::default())),
            hub: parking_lot::Mutex::new(None),
        }
    }

    /// Install the remote fleet (hub mode) — `/v1` chat for a remote-resident model then
    /// routes through it. Idempotent; replaces any prior fleet.
    pub fn set_fleet(&self, fleet: Arc<crate::node::fleet::HubFleet>) {
        *self.fleet.lock() = Some(fleet);
    }

    /// The installed remote fleet, if any (hub mode).
    pub fn fleet(&self) -> Option<Arc<crate::node::fleet::HubFleet>> {
        self.fleet.lock().clone()
    }

    /// Install/replace the API-key store gating the HTTP surface (P5). An empty store
    /// disables auth. The standalone binary calls this with the loaded `api_keys.json`.
    pub fn set_api_keys(&self, keys: Arc<crate::keys::ApiKeys>) {
        *self.api_keys.lock() = keys;
    }

    /// A snapshot handle to the current API-key store (cheap `Arc` clone) — read by the
    /// serve-layer auth middleware on each request.
    pub fn api_keys(&self) -> Arc<crate::keys::ApiKeys> {
        self.api_keys.lock().clone()
    }

    /// Install the running hub (P3 hub mode) so the pairing API can mint node-join tokens.
    pub fn set_hub(&self, hub: Arc<crate::node::hub::Hub>) {
        *self.hub.lock() = Some(hub);
    }

    /// The running hub, if the server is in hub mode.
    pub fn hub(&self) -> Option<Arc<crate::node::hub::Hub>> {
        self.hub.lock().clone()
    }

    /// Whether "Verbose Logging" is on. Single home is the [`LogBus`] (read by
    /// the serve completion line AND the worker drain); delegated via the supervisor.
    pub fn verbose(&self) -> bool {
        self.sup.log_verbose()
    }

    /// Turn "Verbose Logging" on or off at runtime. Sets the one-home flag (the
    /// serve layer + log layer pick it up instantly) AND fire-and-forget pushes
    /// the level to the running worker so its llama.cpp engine-log filter flips
    /// (INFO+ ↔ DEBUG+) live, without a reload or blocking the caller.
    pub fn set_verbose(&self, v: bool) {
        self.sup.set_log_verbose(v);
        self.sup.set_worker_verbose(v);
    }

    /// Whether serve-layer "Log Incoming Tokens" is on (the incoming-prompt line).
    pub fn log_incoming_tokens(&self) -> bool {
        self.log_incoming_tokens
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Turn serve-layer "Log Incoming Tokens" on or off at runtime. Enabling it
    /// opts into logging prompt CONTENT, overriding the redact-by-default policy.
    pub fn set_log_incoming_tokens(&self, v: bool) {
        self.log_incoming_tokens
            .store(v, std::sync::atomic::Ordering::Relaxed);
    }

    /// Whether the Developer Logs are in un-redacted DEBUG mode — the log layer
    /// then emits non-message structured fields (incl. prompt content). Off by
    /// default. Lives on the [`LogBus`] (the layer's only handle); delegated here.
    pub fn log_show_fields(&self) -> bool {
        self.sup.log_show_fields()
    }

    /// Toggle the un-redacted DEBUG log mode at runtime. Enabling it surfaces ALL
    /// structured log fields — including prompt CONTENT — for debugging.
    pub fn set_log_show_fields(&self, v: bool) {
        self.sup.set_log_show_fields(v);
    }

    /// Whether serve-layer "Just-in-Time loading" is on (default `true`). When
    /// on, a chat for a scanned-but-unloaded model loads it on demand instead of
    /// returning `[HG003]`.
    pub fn jit_enabled(&self) -> bool {
        self.jit_enabled.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Turn serve-layer "Just-in-Time loading" on or off at runtime.
    pub fn set_jit_enabled(&self, v: bool) {
        self.jit_enabled
            .store(v, std::sync::atomic::Ordering::Relaxed);
    }

    /// Whether the idle reaper auto-unloads the loaded model after the idle TTL
    /// (default `true`). When `false`, the reaper never unloads — a loaded model
    /// stays resident until an explicit unload regardless of idle time.
    pub fn auto_unload_idle(&self) -> bool {
        self.auto_unload_idle
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Turn idle auto-unload on or off at runtime. Read by the idle reaper each
    /// tick, so a change takes effect without a restart.
    pub fn set_auto_unload_idle(&self, v: bool) {
        self.auto_unload_idle
            .store(v, std::sync::atomic::Ordering::Relaxed);
    }

    /// Idle minutes after which the loaded model is auto-unloaded (default seeded
    /// from [`IDLE_UNLOAD_TTL`]). Read by the idle reaper each tick.
    pub fn idle_ttl_minutes(&self) -> u64 {
        self.idle_ttl_minutes
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Set the idle auto-unload TTL in minutes at runtime. Read by the idle
    /// reaper each tick, so a change takes effect without a restart.
    pub fn set_idle_ttl_minutes(&self, minutes: u64) {
        self.idle_ttl_minutes
            .store(minutes, std::sync::atomic::Ordering::Relaxed);
    }

    /// Active per-load idle-TTL override in minutes, or `None` when no override
    /// is set. Set at load time from the load request and cleared on unload, so
    /// it reflects only the currently-loaded model. `0` in the atomic means "no
    /// override" and reads back as `None`.
    pub fn loaded_idle_ttl_override(&self) -> Option<u64> {
        match self
            .loaded_idle_ttl_override
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            0 => None,
            n => Some(n),
        }
    }

    /// Set (or clear, with `None`) the per-load idle-TTL override in minutes. The
    /// idle reaper prefers this over [`idle_ttl_minutes`](Self::idle_ttl_minutes)
    /// for the currently-loaded model. `None` stores `0` (no override).
    pub fn set_loaded_idle_ttl_override(&self, minutes: Option<u64>) {
        self.loaded_idle_ttl_override
            .store(minutes.unwrap_or(0), std::sync::atomic::Ordering::Relaxed);
    }

    /// Whether the `/v1` inference surface is currently serving (default `true`).
    /// When `false`, the `/v1` inference endpoints return `[HG019]` → 503 while
    /// the `/api/higgs/*` control surface stays reachable.
    pub fn serving_enabled(&self) -> bool {
        self.serving_enabled
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Turn `/v1` inference serving on or off at runtime. Read by the chat
    /// boundary on each request, so a change takes effect immediately without a
    /// restart; the control surface is unaffected.
    pub fn set_serving_enabled(&self, v: bool) {
        self.serving_enabled
            .store(v, std::sync::atomic::Ordering::Relaxed);
    }

    /// Bring up control only — does NOT spawn a worker.
    ///
    /// A worker is spawned lazily by [`load`](Self::load) (spawn-on-load,
    /// LM-Studio model). `scan` runs host-side and needs no worker. The serve
    /// layer holds `Arc<Higgs>` for control regardless of worker liveness.
    ///
    /// Spawns the idle reaper background task, which auto-unloads the loaded
    /// model after [`IDLE_UNLOAD_TTL`] with no inference. The task holds a
    /// `Weak<Higgs>` so it self-terminates when the host drops its `Arc<Higgs>`.
    pub async fn start(self: &Arc<Self>) -> Result<(), HiggsError> {
        let weak = Arc::downgrade(self);
        tokio::spawn(idle_reaper(weak));
        Ok(())
    }

    /// Gracefully shut down the worker (2 s timeout).
    ///
    /// Holds the `lifecycle` mutex for the whole body so a deliberate stop never
    /// interleaves with a concurrent `load`/`unload` (which would let a load
    /// spawn + M_LOAD + emit `ModelLoaded` race this kill). Also clears the
    /// load-replay state: a deliberate worker stop must not leave `last_load`
    /// behind for `attempt_restart` to resurrect the model.
    pub async fn stop(&self) {
        let _lifecycle = self.lifecycle.lock().await;
        self.sup.clear_last_load();
        self.sup.stop().await;
        // A deliberate worker stop unloads the model — drop any per-load idle-TTL
        // override so it never applies to a model loaded after a restart.
        self.set_loaded_idle_ttl_override(None);
    }

    /// Scan all configured model directories and return the discovered models.
    ///
    /// Runs host-side: model scanning is pure Rust (`ggus` + `memmap2` + `std::fs`,
    /// no llama.cpp FFI) so it needs no worker. Returns `Err` [HG001] if a
    /// configured root exists but cannot be read.
    pub async fn scan(&self) -> Result<Vec<HiggsModel>, HiggsError> {
        let (lmstudio, hf, ollama) = {
            let cfg = self.config.lock();
            (
                cfg.lmstudio_dirs.clone(),
                cfg.hf_dirs.clone(),
                cfg.ollama_dirs.clone(),
            )
        };
        // Scanning does blocking I/O (`std::fs::read_dir`, file open, `memmap2`
        // mmap). `status` polls `scan` and `load` calls it, so running inline
        // would block a tokio runtime thread — offload to a blocking thread.
        // A `JoinError` here means the scan task itself panicked, which is an
        // unrecoverable bug rather than a host-facing condition.
        tokio::task::spawn_blocking(move || {
            let mut store = crate::worker::models::ModelStore::default();
            store.scan(&lmstudio, &hf, &ollama).map(<[_]>::to_vec)
        })
        .await
        .expect("higgs model scan task panicked")
    }

    /// Resolve the Gate-1 (engine-loadability) verdict for each distinct model
    /// `(architecture, quant)` combination, probing only the ones not already
    /// cached for the current engine version.
    ///
    /// `reps` is one representative `(arch, quant, path)` per distinct
    /// `(arch, quant)` — the caller (control layer) dedups by `(arch, quant)` and
    /// picks any one file's path. Probing ONE file per combo and caching the
    /// verdict means every model sharing that `(arch, quant)` inherits it, so a
    /// directory of N quants of the same repo costs at most one probe per quant.
    ///
    /// The cache key is `(arch, quant, engine_version)`: the lookup uses this
    /// binary's engine version, and a stored verdict carries the version the probe
    /// worker reported — the same binary, so the strings match and a re-probe
    /// after an engine upgrade is forced (the key changes). Returns a map
    /// `(arch, quant) -> (loadable, reason)`. A probe-infrastructure failure for a
    /// path yields `(false, Some("<context>"))` for that combo (never a panic).
    pub async fn probe_support(
        &self,
        reps: Vec<(String, String, String)>,
    ) -> std::collections::HashMap<(String, String), (bool, Option<String>)> {
        // This binary's engine version — used to form lookup keys. A stored
        // verdict carries the probe worker's reported version, which is the same
        // value (same binary), so hits match and an engine upgrade invalidates.
        let engine_version = crate::worker::engine::llamacpp::engine_version();
        let mut result: std::collections::HashMap<(String, String), (bool, Option<String>)> =
            std::collections::HashMap::new();
        // Partition into cache hits and the paths that still need a probe.
        let mut to_probe: Vec<String> = Vec::new();
        // Map a probe path back to its (arch, quant) so we can key the verdict.
        let mut path_combo: std::collections::HashMap<String, (String, String)> =
            std::collections::HashMap::new();
        {
            let cache = self.probe_cache.lock();
            for (arch, quant, path) in reps {
                let key = (arch.clone(), quant.clone(), engine_version.clone());
                if let Some(verdict) = cache.get(&key) {
                    result.insert((arch, quant), verdict.clone());
                } else {
                    path_combo.insert(path.clone(), (arch, quant));
                    to_probe.push(path);
                }
            }
        }
        if to_probe.is_empty() {
            return result;
        }
        // Probe the misses in a transient, crash-isolated worker.
        let verdicts = self.sup.probe_paths(to_probe).await;
        let mut cache = self.probe_cache.lock();
        for (path, (loadable, reason, probe_version)) in verdicts {
            let Some((arch, quant)) = path_combo.remove(&path) else {
                continue;
            };
            // Key the stored verdict on the version the worker reported.
            cache.insert(
                (arch.clone(), quant.clone(), probe_version),
                (loadable, reason.clone()),
            );
            result.insert((arch, quant), (loadable, reason));
        }
        result
    }

    /// Load a model by HuggingFace repo id.
    ///
    /// `params` overrides `default_load` when supplied. On success, records the
    /// load params for post-restart replay and emits [`HiggsEvent::ModelLoaded`].
    pub async fn load(&self, id: &str, params: Option<LoadParams>) -> Result<(), HiggsError> {
        // Serialize the whole load/unload lifecycle: spawn-on-load and
        // kill-on-unload must never interleave (protects last_load and the
        // supervisor proc handle). Held for the entire method body.
        let _lifecycle = self.lifecycle.lock().await;
        // load() reads as a sequence of named guard/resolve steps; each helper
        // keeps the exact same order and diagnostic codes as before.
        let explicit_params = params.is_some();
        let mut p = params.unwrap_or_else(|| self.config.lock().default_load.clone());
        // 1. Charset guard ([HG015]) — reject traversal/escape ids before any FS use.
        validate_repo_id(id)?;
        // 2. Scan host-side and resolve the GGUF path ([HG002] if not found).
        let model = self.resolve_model(id).await?;
        // 3. Path-traversal guard ([HG015]) — resolved path must stay inside the roots.
        self.guard_path_within_roots(id, &model.path)?;
        // 4. Pre-load RAM headroom guard ([HG017]) — refuse before spawning a worker.
        guard_memory_headroom(id, model.size_bytes)?;
        // When the caller didn't pin ctx_len, default it to the model's trained
        // context (capped at DEFAULT_CTX_CAP) rather than the hardcoded 4096 —
        // otherwise an agent asking for a large max_tokens overflows n_ctx
        // ([HG005]). The UI can still request the full trained window explicitly.
        if !explicit_params {
            if let Some(train) = model.ctx_train {
                p.ctx_len = (train as u32).min(DEFAULT_CTX_CAP);
            }
        }
        // Serialize the full LoadParams (base + every optional override) and
        // merge `id`/`path` in. Absent optionals (`skip_serializing_if`) simply
        // don't appear, so a quick-load carries exactly the three base fields —
        // the worker then sees no overrides and reproduces current behavior.
        let mut req_params =
            serde_json::to_value(&p).expect("LoadParams serializes to a JSON object");
        if let Some(obj) = req_params.as_object_mut() {
            obj.insert("id".into(), json!(id));
            obj.insert("path".into(), json!(model.path));
        }
        // Spawn-on-load: if no worker is live, bring one up named `higgs(<id>)`
        // before sending M_LOAD. A redundant call while a worker is running is a
        // no-op (single-reader invariant in the supervisor).
        self.sup.start_for(id)?;
        // If M_LOAD fails (bad GGUF, OOM, …) the worker is alive but holds no
        // model — that contradicts kill-on-unload. Tear it down before
        // returning. Call `self.sup.stop()` DIRECTLY (not `self.stop()`): we
        // already hold the `lifecycle` mutex, and `Higgs::stop()` would re-take
        // it → deadlock. `record_last_load`/`ModelLoaded` stay on success only.
        if let Err(e) = self.sup.request(M_LOAD, req_params.clone()).await {
            self.sup.clear_last_load();
            self.sup.stop().await;
            return Err(e);
        }
        self.sup.record_last_load(req_params);
        // Stamp activity on a successful load: the idle reaper measures the TTL
        // from the last chat OR load. Without this, a model loaded while the
        // process has been idle past IDLE_UNLOAD_TTL would be eligible for
        // auto-unload on the reaper's very next tick — before the user gets to
        // send a single chat. A fresh load IS recent activity.
        *self.last_activity.lock() = std::time::Instant::now();
        self.sup.emit(HiggsEvent::ModelLoaded { id: id.to_owned() });
        Ok(())
    }

    /// Unload the current model.
    ///
    /// Emits [`HiggsEvent::ModelUnloaded`] with an empty id when no model id
    /// is available at the facade layer (v1 limitation; worker tracks it).
    pub async fn unload(&self) -> Result<(), HiggsError> {
        // Serialize the whole load/unload lifecycle (see `load`): held for the
        // entire method body so a concurrent load cannot re-set last_load after
        // the clear or race start_for against this stop.
        let _lifecycle = self.lifecycle.lock().await;
        // TODO(v2): single RPC — status+unload is TOCTOU if worker state changes between calls (v1: worker serializes, benign)
        // Capture id from status before unloading so the event carries it.
        let id = self.loaded_id().await.unwrap_or_default();
        // Drop the load-replay state BEFORE the unload/stop awaits: if a respawn
        // races the stop, there must be nothing left for it to replay. Clearing
        // after the awaits leaves a window where attempt_restart could reload the
        // model the user just unloaded.
        self.sup.clear_last_load();
        // Best-effort graceful in-worker unload, then KILL the worker process
        // (spawn-on-load / kill-on-unload). `stop()` sets the deliberate-stop flag
        // so the death triggers no respawn, drains stdin, and reaps the process.
        let _ = self.sup.request(M_UNLOAD, serde_json::Value::Null).await;
        self.sup.stop().await;
        // Clear the per-load idle-TTL override so it never outlives its model: a
        // stale override must not apply to the next loaded model.
        self.set_loaded_idle_ttl_override(None);
        self.sup.emit(HiggsEvent::ModelUnloaded { id });
        Ok(())
    }

    /// Return a live status snapshot.
    ///
    /// `worker_alive` is `true` iff the RPC round-trip succeeded. `loaded` is
    /// independently best-effort: an RPC failure yields `worker_alive:false` with
    /// `loaded:None`; a malformed `loaded` shape in an otherwise-OK response yields
    /// `worker_alive:true` with `loaded:None`.
    pub async fn status(&self) -> Result<HiggsStatus, HiggsError> {
        let result = self.sup.request(M_STATUS, serde_json::Value::Null).await;
        let worker_alive = result.is_ok();
        let v = result.unwrap_or(serde_json::Value::Null);

        // Scan moved host-side: the worker no longer scans (its `ModelStore` is
        // empty), so model metadata and the on-disk count both come from ONE
        // host-side FS walk (pure Rust, no worker RPC), reused below.
        let scan = self.scan().await.unwrap_or_default();
        let models_on_disk = scan.len() as u32;

        // The worker's M_STATUS reports `id`/`ctx_len`/`gpu_layers`/`threads`
        // from the live model, but `arch`/`quant`/`size_bytes`/
        // `max_context_length`/`has_chat_template` come back null (its store is
        // empty). Enrich those from the matching host-scanned `HiggsModel` while
        // keeping the worker-reported id/ctx_len verbatim.
        let loaded = v.get("loaded").and_then(|l| {
            if l.is_null() {
                return None;
            }
            let id = l.get("id")?.as_str()?.to_owned();
            let scanned = scan.iter().find(|m| m.id == id);
            Some(LoadedInfo {
                ctx_len: l.get("ctx_len")?.as_u64()? as u32,
                gpu_layers: l.get("gpu_layers")?.as_u64()? as u32,
                threads: l.get("threads")?.as_u64()? as u32,
                arch: scanned.and_then(|m| m.arch.clone()),
                quant: scanned.and_then(|m| m.quant.clone()),
                max_context_length: scanned.and_then(|m| m.ctx_train),
                size_bytes: scanned.map(|m| m.size_bytes),
                has_chat_template: scanned.map(|m| m.has_chat_template),
                idle_ttl_minutes: self.loaded_idle_ttl_override(),
                id,
            })
        });

        Ok(HiggsStatus {
            worker_alive,
            loaded,
            models_on_disk,
        })
    }

    /// Stream a chat completion.
    ///
    /// `model` is the id the serve layer resolved this chat against (via
    /// [`ensure_loaded`](crate::serve)). It is carried to the worker, which
    /// refuses to generate (`[HG018]` → 503) if a concurrent JIT load swapped the
    /// resident model out between resolution and dispatch — binding each chat to
    /// its resolved model so a swap errors instead of serving the wrong model.
    ///
    /// Returns `(receiver, join_handle)`:
    /// - `receiver` carries streaming deltas — each item is one content chunk
    ///   from the worker; this is the canonical output for SSE / streaming consumers.
    /// - `join_handle` resolves with the final [`ChatOutcome`] when generation is
    ///   complete (or `Err` if the worker fails); `ChatOutcome::content` is the
    ///   full concatenated text and is the canonical output for non-streaming
    ///   consumers (`/v1` with `stream: false`).  Both are retained on purpose —
    ///   callers choose which representation they need.
    ///
    /// Concurrent callers are each accepted and routed their own deltas via a
    /// per-request keyed channel; the worker executes requests serially (single-
    /// threaded stdin loop) so throughput is sequential but callers never clobber
    /// each other's streams.
    pub async fn chat_stream(
        &self,
        model: String,
        messages_json: String,
        max_tokens: usize,
        temperature: f32,
        tools_json: Option<String>,
    ) -> Result<
        (
            mpsc::UnboundedReceiver<String>,
            tokio::task::JoinHandle<Result<ChatOutcome, HiggsError>>,
        ),
        HiggsError,
    > {
        // Remote routing FIRST: if a fleet is installed and this model lives on a remote node,
        // relay the chat over that node's transport. A remote model uses NEITHER the local
        // worker NOR the local idle timer, so this path does NOT stamp `last_activity` (else
        // remote traffic would keep an idle LOCAL model resident) and uses a SEPARATE
        // admission gate. Bind the clone to a `let` so the parking_lot guard drops HERE, not
        // held across the `.await` (an `if let` scrutinee temporary would make this !Send).
        let fleet = self.fleet.lock().clone();
        if let Some(fleet) = fleet {
            if fleet.is_remote(&model).await {
                // Bound concurrent REMOTE chats — the node-side relay has no semaphore, so
                // without this a client could open unbounded hub/node generations + streams.
                // Separate from `inference_gate` so it never blocks the reaper's
                // acquire-all-LOCAL-permits unload of an idle local model.
                let permit = Arc::clone(&self.remote_gate)
                    .try_acquire_owned()
                    .map_err(|_| HiggsError::ServerBusy {
                        in_flight: MAX_CONCURRENT_INFERENCE,
                        max: MAX_CONCURRENT_INFERENCE,
                    })?;
                let (rx, fut) = fleet
                    .chat(&model, messages_json, max_tokens, temperature, tools_json)
                    .await?;
                let handle = tokio::spawn(async move {
                    let _permit = permit; // held for the whole remote generation
                    Ok(chat_outcome_from_value(&fut.await?))
                });
                return Ok((rx, handle));
            }
        }

        // Local path. Stamp last-activity so the idle reaper never unloads a model that is
        // actively serving. Done before the admission gate so even a request that ends up
        // rejected (ServerBusy) still counts as recent activity — a busy server is by
        // definition not idle. Lock held for one `Instant` write only, never across `.await`.
        *self.last_activity.lock() = std::time::Instant::now();

        // Admission gate: bound concurrent in-flight inference so the no-auth
        // server can't be flooded. A full gate is a capacity signal (HTTP 503),
        // not a failure — the client may retry. The owned permit is moved into
        // the spawned generation task below so it is held for the WHOLE request
        // (queue wait + generation) and released on any outcome.
        let permit = Arc::clone(&self.inference_gate)
            .try_acquire_owned()
            .map_err(|_| HiggsError::ServerBusy {
                in_flight: MAX_CONCURRENT_INFERENCE,
                max: MAX_CONCURRENT_INFERENCE,
            })?;
        // Mint the request, register its keyed sink, and obtain a future that
        // drives the M_CHAT RPC to completion and removes the sink on any
        // outcome — all of it lives in `Supervisor::chat` so this facade does not
        // touch request-id minting, the sink map, or `request_with_id` directly.
        // `rx` is returned to the caller now; `call` is awaited inside the spawned
        // generation task (so the admission permit, kept here, rides it).
        let (rx, call) = self
            .sup
            .chat(model, messages_json, max_tokens, temperature, tools_json);
        let handle = tokio::spawn(async move {
            // Hold the admission permit for the whole generation; dropping it
            // here (on any return path) releases the gate slot. Bound to a name
            // so it is not dropped early.
            let _permit = permit;
            let result = call.await;

            Ok(chat_outcome_from_value(&result?))
        });

        Ok((rx, handle))
    }

    /// Subscribe to worker lifecycle events.
    pub fn events(&self) -> broadcast::Receiver<HiggsEvent> {
        self.sup.events()
    }

    /// Return up to `n` recent Developer-Log lines (oldest first), optionally
    /// restricted to one [`LogSource`] (`None` = worker stderr + serve events).
    pub fn logs(&self, n: usize, filter: Option<LogSource>) -> Vec<String> {
        self.sup.logs(n, filter)
    }

    /// Subscribe to live Developer-Log lines pushed after this call. The SSE
    /// log-stream handler pairs this with [`logs`](Self::logs) for
    /// replay-then-live delivery; filter each by [`LogLine::source`].
    pub fn subscribe_logs(&self) -> tokio::sync::broadcast::Receiver<LogLine> {
        self.sup.subscribe_logs()
    }

    /// Snapshot of the configured default load parameters.
    ///
    /// The serve router uses this to fill fields absent from a partial
    /// load request — config stays the single home for the defaults.
    pub(crate) fn default_load(&self) -> LoadParams {
        self.config.lock().default_load.clone()
    }

    /// Read-only snapshot of the effective server config for `GET
    /// /api/higgs/system`. Clones the scan dirs and load defaults from the live
    /// [`HiggsConfig`] and pairs them with the two fixed invariants ([`BIND_HOST`],
    /// [`DEFAULT_CTX_CAP`]). Pure read — no worker RPC, no mutation.
    pub fn server_config(&self) -> HiggsServerConfig {
        let cfg = self.config.lock();
        let to_strings = |dirs: &[PathBuf]| dirs.iter().map(|p| p.display().to_string()).collect();
        HiggsServerConfig {
            bind_host: BIND_HOST.to_owned(),
            lmstudio_dirs: to_strings(&cfg.lmstudio_dirs),
            hf_dirs: to_strings(&cfg.hf_dirs),
            ollama_dirs: to_strings(&cfg.ollama_dirs),
            default_load: cfg.default_load.clone(),
            default_ctx_cap: DEFAULT_CTX_CAP,
            limits: HiggsLimits {
                max_body_bytes: crate::serve::MAX_BODY_BYTES as u64,
                control_timeout_secs: crate::serve::CONTROL_TIMEOUT.as_secs(),
                chat_timeout_secs: crate::supervisor::CHAT_RPC_TIMEOUT.as_secs(),
                max_output_tokens: crate::serve::MAX_OUTPUT_TOKENS,
                max_concurrent_inference: MAX_CONCURRENT_INFERENCE as u32,
                memory_headroom_fraction: MEMORY_HEADROOM_FRACTION,
                idle_unload_ttl_secs: IDLE_UNLOAD_TTL.as_secs(),
            },
        }
    }

    /// Host compute devices (CPU/GPU/accel), gathered once via a transient
    /// sysinfo worker and cached. Hardware is static-ish, so a cache hit returns
    /// immediately; a miss spawns a crash-isolated transient worker, runs
    /// M_SYSINFO, caches the result, and returns it. A gather failure returns an
    /// empty `Vec` and leaves the cache empty so a later call retries — `GET
    /// /api/higgs/system` still returns hardware/runtime without devices.
    ///
    /// This is the canonical home for the gathered device list; the
    /// `SystemInfo::gather` path takes it as input rather than reading FFI itself
    /// (the FFI lives in the worker, not the host).
    pub async fn sysinfo(&self) -> Vec<crate::system::GpuDevice> {
        if let Some(cached) = self.device_cache.lock().clone() {
            return cached;
        }
        let gpus = self.sup.sysinfo().await;
        // Cache only a non-empty result: an empty list usually means the gather
        // failed (spawn/EOF/timeout), so leave the cache empty to retry later.
        if !gpus.is_empty() {
            *self.device_cache.lock() = Some(gpus.clone());
        }
        gpus
    }

    /// Test-only: build a `Higgs` over a pre-built (mock) supervisor.
    ///
    /// Lets sibling modules (`serve`) reuse the duplex mock seam without
    /// access to this module's private fields.
    #[cfg(test)]
    pub(crate) fn with_supervisor(sup: Arc<Supervisor>, config: HiggsConfig) -> Self {
        Self {
            sup,
            config: parking_lot::Mutex::new(config),
            lifecycle: tokio::sync::Mutex::new(()),
            inference_gate: Arc::new(tokio::sync::Semaphore::new(MAX_CONCURRENT_INFERENCE)),
            remote_gate: Arc::new(tokio::sync::Semaphore::new(MAX_CONCURRENT_INFERENCE)),
            last_activity: parking_lot::Mutex::new(std::time::Instant::now()),
            log_incoming_tokens: std::sync::atomic::AtomicBool::new(false),
            jit_enabled: std::sync::atomic::AtomicBool::new(true),
            auto_unload_idle: std::sync::atomic::AtomicBool::new(true),
            idle_ttl_minutes: std::sync::atomic::AtomicU64::new(IDLE_UNLOAD_TTL_MINUTES),
            loaded_idle_ttl_override: std::sync::atomic::AtomicU64::new(0),
            serving_enabled: std::sync::atomic::AtomicBool::new(true),
            probe_cache: parking_lot::Mutex::new(std::collections::HashMap::new()),
            device_cache: parking_lot::Mutex::new(None),
            fleet: parking_lot::Mutex::new(None),
            api_keys: parking_lot::Mutex::new(std::sync::Arc::new(crate::keys::ApiKeys::default())),
            hub: parking_lot::Mutex::new(None),
        }
    }

    // ── private ───────────────────────────────────────────────────────────────

    /// Resolve a model id to its scanned [`HiggsModel`] (with the GGUF path).
    ///
    /// Scan moved host-side, so the worker's `ModelStore` is empty on a fresh
    /// spawn-on-load worker: the model is resolved HERE and its GGUF path carried
    /// in the M_LOAD params. Without this the worker's `store.get(id)` returns
    /// HG002 for every normal load. Takes the first matching model; returns
    /// `Err` [HG002] `ModelNotFound` when no scanned model has this id.
    async fn resolve_model(&self, id: &str) -> Result<HiggsModel, HiggsError> {
        self.scan()
            .await?
            .into_iter()
            .find(|m| m.id == id)
            .ok_or_else(|| HiggsError::ModelNotFound { id: id.to_owned() })
    }

    /// Reject a resolved GGUF `path` that escapes every configured scan root.
    ///
    /// The path comes from a host-side scan of those roots, so this holds for
    /// every legitimate load; the check rejects any path that escapes the roots
    /// (symlink/`..` escape) before it is sent to the worker for FFI loading.
    /// `Err` [HG015] `InvalidModelId` on an escape.
    fn guard_path_within_roots(&self, id: &str, path: &str) -> Result<(), HiggsError> {
        let scan_roots: Vec<PathBuf> = {
            let cfg = self.config.lock();
            cfg.lmstudio_dirs
                .iter()
                .chain(cfg.hf_dirs.iter())
                .chain(cfg.ollama_dirs.iter())
                .cloned()
                .collect()
        };
        if path_within_roots(path, &scan_roots) {
            Ok(())
        } else {
            Err(HiggsError::InvalidModelId {
                id: id.to_owned(),
                reason: format!("resolved path {path} is outside every configured scan directory"),
            })
        }
    }

    /// Best-effort: ask the worker for the currently loaded model id.
    async fn loaded_id(&self) -> Option<String> {
        let v = self
            .sup
            .request(M_STATUS, serde_json::Value::Null)
            .await
            .ok()?;
        v.get("loaded")?.get("id")?.as_str().map(ToOwned::to_owned)
    }
}

// ── Tests (see api/tests.rs) ──
#[cfg(test)]
mod tests;
