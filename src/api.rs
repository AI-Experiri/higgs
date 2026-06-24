//! `Higgs` public facade and `HiggsConfig` — the host-facing API.
//!
//! One `Higgs` instance per host app. Typed facade over a co-located LOCAL
//! [`NodeRuntime`](crate::node::runtime::NodeRuntime) — the same multi-worker
//! engine remote nodes run (P4b). `Higgs` owns the facade-level state (config,
//! the load lifecycle mutex, the inference admission gate, the serve-layer
//! toggles) and delegates worker spawn/load/unload/status/chat, idle
//! auto-unload, and the Developer-Log bus to the node. A remote-resident model
//! routes through the `fleet` instead. The host maps its own config table onto
//! [`HiggsConfig`].

use std::path::PathBuf;
use std::sync::Arc;

use tokio::sync::{broadcast, mpsc};

use crate::diagnostic::HiggsError;
use crate::log_bus::{LogBus, LogLine, LogSource};
use crate::node::runtime::{NodeConfig, NodeRuntime, DEFAULT_IDLE_TTL};
use crate::node::worker_id::WorkerId;
use crate::remote::{InventoryWorker, NodeInventory, NodeLoadParams};
use crate::supervisor::HiggsEvent;
use crate::worker::engine::LoadParams;
use crate::worker::models::HiggsModel;

// Submodules split out of this file (see api/README.md, api/DESIGN.md). `pub use`
// keeps every existing `crate::api::*` path resolving unchanged.
mod guards;
mod types;

#[cfg(test)]
use guards::fits_in_memory;
use guards::validate_repo_id;
pub(crate) use guards::{guard_memory_headroom, path_within_roots};
pub use types::*;

/// The in-process handle to the higgs runtime. One instance per host app.
///
/// The host-facing facade over the co-located LOCAL [`NodeRuntime`]. `Higgs` owns
/// the facade-level state — the live [`HiggsConfig`], the `lifecycle` mutex that
/// serializes concurrent loads, the `inference_gate` admission semaphore, and the
/// serve-layer toggles — while the node owns the worker processes, RPC
/// correlation, idle auto-unload, and the load-replay state. Constructing `Higgs`
/// does not start a worker; workers are spawned lazily, one per [`load`](Self::load).
pub struct Higgs {
    /// The co-located LOCAL node (P4b): an in-process multi-worker `NodeRuntime`, the same
    /// engine remote nodes run. Local inference/load/unload/status route through it; a remote
    /// model routes through the `fleet`. (Replaces the old single direct `Supervisor`.)
    local: Arc<NodeRuntime>,
    config: parking_lot::Mutex<HiggsConfig>,
    /// Serializes concurrent loads so two racing JIT loads of the same id can't
    /// each spawn a worker (the node load is additive — it never dedups).
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
    /// on-demand additive [`load`](Self::load) (a fresh worker alongside any
    /// others — the local node is multi-model) instead of the `[HG003]` 404. When
    /// `false`, the chat path keeps the explicit-load behavior: an unloaded model
    /// is a 404. A plain atomic — set/read in isolation, no critical section,
    /// never across `.await`. Defaults to `true` (JIT on).
    jit_enabled: std::sync::atomic::AtomicBool,
    /// Runtime "Auto-unload idle models" toggle, mirrored to the node's live
    /// [`IdleConfig`](crate::node::runtime::IdleConfig) by
    /// [`set_auto_unload_idle`](Self::set_auto_unload_idle). When `true` (the
    /// default), the node's per-worker idle reaper unloads a worker after the idle
    /// TTL of no inference; when `false` it never auto-unloads. This atomic is the
    /// facade-side mirror the Server-Settings getter reads. A plain atomic —
    /// set/read in isolation, never across `.await`.
    auto_unload_idle: std::sync::atomic::AtomicBool,
    /// Runtime idle auto-unload TTL, in minutes, mirrored to the node's live
    /// [`IdleConfig`](crate::node::runtime::IdleConfig) by
    /// [`set_idle_ttl_minutes`](Self::set_idle_ttl_minutes). Seeded from the node's
    /// [`DEFAULT_IDLE_TTL`] (60 minutes). This atomic is the facade-side mirror the
    /// Server-Settings getter reads. A plain atomic — set/read in isolation, never
    /// across `.await`.
    idle_ttl_minutes: std::sync::atomic::AtomicU64,
    /// Per-load idle-TTL override, in minutes (HOST-SIDE display only). `0` means
    /// "no override". Set by the control settings endpoint at load time and cleared
    /// on unload, and surfaced in [`status`](Self::status)'s `LoadedInfo`. NOTE: the
    /// node's reaper enforces a single per-node TTL ([`IdleConfig`](crate::node::runtime::IdleConfig));
    /// per-load override ENFORCEMENT is a documented follow-up, so today this value
    /// is reported but not independently enforced. A plain atomic — set/read in
    /// isolation, never across `.await`.
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
    /// Cached host device list gathered once via the node's transient sysinfo
    /// worker ([`NodeRuntime::gpus`](crate::node::runtime::NodeRuntime::gpus)).
    /// Hardware is static-ish, so it is gathered on first request and reused; `None` until
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
    /// Serializes the hub kill-switch lifecycle ops (`enable`/`disable`). Held across the WHOLE
    /// check→start→publish (enable) / shutdown→disconnect→clear (disable) sequence so two
    /// concurrent enables can't both pass the `hub().is_none()` check, bind two endpoints + accept
    /// loops, and orphan the loser (whose accept loop would keep admitting nodes after a later
    /// disable, defeating the kill switch). SEPARATE from `lifecycle` (local model load/unload) so
    /// the two never cross-couple. An async `Mutex` because the guard is held across `.await`.
    hub_lifecycle: tokio::sync::Mutex<()>,
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
        // Build the co-located LOCAL node from the same config + bus. The node owns the
        // workers; it shares `bus` so local worker stderr lands in the Developer Logs.
        let node_config = NodeConfig {
            bus,
            lmstudio_dirs: config.lmstudio_dirs.clone(),
            hf_dirs: config.hf_dirs.clone(),
            ollama_dirs: config.ollama_dirs.clone(),
            idle_ttl: DEFAULT_IDLE_TTL,
        };
        Self::with_local(Arc::new(NodeRuntime::new(node_config)), config)
    }

    /// Construct the facade around an already-built local [`NodeRuntime`] — the single home for
    /// the facade's default field initialization. Production goes through [`with_log_bus`];
    /// tests inject a fake-worker-backed `NodeRuntime` here.
    pub(crate) fn with_local(local: Arc<NodeRuntime>, config: HiggsConfig) -> Self {
        Self {
            local,
            config: parking_lot::Mutex::new(config),
            lifecycle: tokio::sync::Mutex::new(()),
            inference_gate: Arc::new(tokio::sync::Semaphore::new(MAX_CONCURRENT_INFERENCE)),
            remote_gate: Arc::new(tokio::sync::Semaphore::new(MAX_CONCURRENT_INFERENCE)),
            log_incoming_tokens: std::sync::atomic::AtomicBool::new(false),
            jit_enabled: std::sync::atomic::AtomicBool::new(true),
            // Mirror the node's idle defaults (auto-unload on, 60-min TTL) so the
            // Server-Settings getters read the same values the node reaper enforces.
            auto_unload_idle: std::sync::atomic::AtomicBool::new(true),
            idle_ttl_minutes: std::sync::atomic::AtomicU64::new(DEFAULT_IDLE_TTL.as_secs() / 60),
            loaded_idle_ttl_override: std::sync::atomic::AtomicU64::new(0),
            serving_enabled: std::sync::atomic::AtomicBool::new(true),
            probe_cache: parking_lot::Mutex::new(std::collections::HashMap::new()),
            device_cache: parking_lot::Mutex::new(None),
            fleet: parking_lot::Mutex::new(None),
            api_keys: parking_lot::Mutex::new(std::sync::Arc::new(crate::keys::ApiKeys::default())),
            hub: parking_lot::Mutex::new(None),
            hub_lifecycle: tokio::sync::Mutex::new(()),
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

    /// Tear down the running-hub handle (the kill switch's network-off step). The caller first
    /// closes the endpoint (`Hub::shutdown`) and the node transports
    /// (`HubFleet::disconnect_all`); this then drops the hub so `hub()` is `None` (pairing 409s,
    /// status reports disabled). The `fleet` is deliberately KEPT installed so the route table
    /// survives and re-enabling is a pure reconnect.
    pub fn clear_hub(&self) {
        *self.hub.lock() = None;
    }

    /// The process [`LogBus`] this facade was built with (the one its serve layer reads). The
    /// hub kill switch hands it to `start_hub` when (re-)enabling so relayed remote-worker logs
    /// land in the same Developer-Log console.
    pub fn log_bus(&self) -> Arc<LogBus> {
        self.local.bus().clone()
    }

    /// The lock serializing hub enable/disable (the kill switch). Callers hold it across the
    /// whole lifecycle op (see the `hub_lifecycle` field doc) so concurrent toggles can't race.
    pub fn hub_lifecycle(&self) -> &tokio::sync::Mutex<()> {
        &self.hub_lifecycle
    }

    /// Whether "Verbose Logging" is on. Single home is the [`LogBus`] (read by
    /// the serve completion line AND the worker drain); delegated via the supervisor.
    pub fn verbose(&self) -> bool {
        self.local.bus().verbose()
    }

    /// Turn "Verbose Logging" on or off at runtime on the local node bus (the serve layer +
    /// log layer pick it up instantly; newly-spawned local workers inherit it at load time).
    /// NOTE: with multiple local workers the level is no longer pushed live into
    /// already-running workers' llama.cpp filters — it takes effect on the next load.
    pub fn set_verbose(&self, v: bool) {
        self.local.bus().set_verbose(v);
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
        self.local.bus().show_fields()
    }

    /// Toggle the un-redacted DEBUG log mode at runtime. Enabling it surfaces ALL
    /// structured log fields — including prompt CONTENT — for debugging.
    pub fn set_log_show_fields(&self, v: bool) {
        self.local.bus().set_show_fields(v);
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
        self.local.idle().set_enabled(v); // drive the local node reaper live
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
        self.local
            .idle()
            .set_ttl(std::time::Duration::from_secs(minutes * 60));
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
    /// Idle auto-unload now lives IN the local `NodeRuntime` (per-worker), so the engine no
    /// longer runs its own reaper. This spawns a relay draining the local node's per-worker
    /// stderr into the shared Developer-Log bus (the node tags lines on per-worker buses,
    /// otherwise separate from the engine bus).
    pub async fn start(self: &Arc<Self>) -> Result<(), HiggsError> {
        let mut logs = self.local.subscribe_logs();
        let bus = self.local.bus().clone();
        tokio::spawn(async move {
            loop {
                match logs.recv().await {
                    Ok((_worker, line)) => bus.push(LogSource::Worker, line),
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {}
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                }
            }
        });
        Ok(())
    }

    /// Gracefully shut down EVERY local worker (the node's `shutdown_all` drain).
    ///
    /// Holds the `lifecycle` mutex for the whole body so a deliberate stop never
    /// interleaves with a concurrent `load` (which would let a fresh worker slip
    /// past the drain). The node's drain reaps each worker's process and clears its
    /// load-replay state, so no worker survives to be auto-restarted.
    pub async fn stop(&self) {
        let _lifecycle = self.lifecycle.lock().await;
        // Drain every local worker (graceful shutdown of the local node's registry).
        self.local.shutdown_all().await;
        // No model is loaded after a full drain — drop any per-load idle-TTL override.
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
        let verdicts = self.local.probe_paths(to_probe).await;
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
        // Serialize concurrent loads at the facade so two racing JIT loads of the
        // same id can't each spawn a worker (the node load is additive — it never
        // dedups). Held for the whole method body.
        let _lifecycle = self.lifecycle.lock().await;
        // 1. Charset guard ([HG015]) — reject traversal/escape ids before any FS
        //    use. The node does resolve ([HG002]), the path-traversal guard
        //    ([HG015]), and the RAM headroom guard ([HG017]); this charset check
        //    is the one guard that stays host-side (it precedes any FS access).
        validate_repo_id(id)?;
        // Idempotent per model: if a worker for this raw id is already resident,
        // this is a no-op success. The node load is ADDITIVE (it never dedups), so
        // without this a second explicit load — or two racing JIT loads of the same
        // unloaded model — would spawn a duplicate worker. The `lifecycle` mutex
        // serializes concurrent loads, so the common check-then-load is race-free.
        // (To run N instances of one model, the node API is additive; the facade
        // deliberately presents one-worker-per-model.)
        //
        // RESIDUAL (cancellation): the mutex is released if a load future is
        // CANCELLED mid-flight (after its `NodeMsg::Load` is sent but before the
        // reply). A second same-id load can then acquire the mutex, observe no
        // COMMITTED worker (the first's commit is still pending), and spawn a
        // duplicate. This self-heals: the cancelled load's `LoadCommit` reaps its
        // worker (the caller is gone, so the keep-insert is skipped), and the RAM
        // headroom guard bounds the transient overlap — end state is one worker. A
        // fully cancellation-safe dedup would require the node actor to track and
        // dedup IN-FLIGHT loads, changing its additive contract (used by the remote
        // path); deferred as not worth that for a self-healing, memory-bounded race.
        if self.local.instances().await.iter().any(|(_, m)| m == id) {
            return Ok(());
        }
        // Map the host `LoadParams` onto the node's lean `NodeLoadParams`. Only
        // id/ctx_len/gpu_layers/threads cross to the node today; the rich engine
        // overrides (use_mmap/use_mlock/n_batch/…) are NOT carried — matching the
        // remote path (DESIGN-remote.md), a documented gap. When the caller pins
        // no params, ctx_len is left `None` so the node defaults it to the model's
        // trained context (capped at DEFAULT_CTX_CAP), exactly as the old facade did.
        let np = match params {
            Some(p) => NodeLoadParams {
                id: id.to_owned(),
                ctx_len: Some(p.ctx_len),
                gpu_layers: Some(p.gpu_layers),
                threads: Some(p.threads),
            },
            None => {
                let d = self.config.lock().default_load.clone();
                NodeLoadParams {
                    id: id.to_owned(),
                    ctx_len: None,
                    gpu_layers: Some(d.gpu_layers),
                    threads: Some(d.threads),
                }
            }
        };
        // Additive load on the local node: spawns a fresh worker for `id` (the
        // node emits `ModelLoaded` on commit). resolve / headroom / path-traversal
        // failures surface as their mapped HiggsError.
        self.local.load(np).await.map(|_| ())
    }

    /// Unload ALL locally-resident models.
    ///
    /// The control surface is single-button ("unload"), so this drains every local
    /// worker. Each [`NodeRuntime::unload`](crate::node::runtime::NodeRuntime::unload)
    /// removes the worker from the registry synchronously, then AWAITS its
    /// `Supervisor::stop()` (the node fires the reply only after the process is
    /// reaped) and emits one [`HiggsEvent::ModelUnloaded`].
    ///
    /// `shutdown_all` is NOT used here: it is terminal (it permanently rejects
    /// future loads), whereas unload must leave the node ready to load again.
    ///
    /// CONCURRENCY: the `lifecycle` mutex serializes this against `load` (no fresh
    /// worker can slip past the drain — `load` is the only thing that ADDS a worker).
    /// The one other concurrent mutator is the node's idle reaper, which only
    /// REMOVES workers. If it grabs a snapshotted worker between this loop's
    /// `instances()` snapshot and that worker's `unload`, our `unload` returns
    /// `no_worker` (ignored): that worker was already removed AND its stop is reaped
    /// by the node on a tracked teardown task (no leak/orphan — the actor holds
    /// itself alive until every in-flight stop completes). So the post-condition
    /// "no worker resident" always holds on return; the only residual is that a
    /// reaper-grabbed worker's process exit may finish microseconds after this
    /// returns, which is harmless (a following additive `load` is gated by the RAM
    /// headroom guard, and multi-model workers coexist by design anyway).
    pub async fn unload(&self) -> Result<(), HiggsError> {
        let _lifecycle = self.lifecycle.lock().await;
        for (worker, _model) in self.local.instances().await {
            // Ignore `no_worker`: a concurrent idle-reap already removed+reaped it
            // (see the CONCURRENCY note above). The reaper never ADDS, so after this
            // single pass the registry is empty.
            let _ = self.local.unload(worker).await;
        }
        // Clear the per-load idle-TTL override so it never outlives its model.
        self.set_loaded_idle_ttl_override(None);
        Ok(())
    }

    /// Return a live status snapshot of the PRIMARY local instance.
    ///
    /// The local node may host several models (additive load); the single-model
    /// control surface reports the PRIMARY — the lowest worker id. `worker_alive`
    /// is `true` iff a worker is resident and its `M_STATUS` round-trip succeeded.
    /// `loaded` is independently best-effort: no resident worker yields
    /// `worker_alive:false`/`loaded:None`; a malformed `loaded` shape yields
    /// `worker_alive:true`/`loaded:None`. `/v1/models` lists ALL served instances
    /// via [`local_served_ids`](Self::local_served_ids), not just this primary.
    pub async fn status(&self) -> Result<HiggsStatus, HiggsError> {
        // Scan host-side (pure Rust, no worker RPC): model metadata + on-disk
        // count both come from ONE FS walk, reused for enrichment below.
        let scan = self.scan().await.unwrap_or_default();
        let models_on_disk = scan.len() as u32;

        // Primary = lowest worker id. No resident worker → idle state.
        let mut instances = self.local.instances().await;
        instances.sort_by_key(|(w, _)| *w);
        let Some(primary) = instances.first().map(|(w, _)| *w) else {
            return Ok(HiggsStatus {
                worker_alive: false,
                loaded: None,
                loaded_all: Vec::new(),
                models_on_disk,
            });
        };

        // The local node is multi-model (one worker per resident model). Report EVERY worker —
        // each tagged with its `worker_id` and its `/v1` served id — so the UI can show a card
        // (and a per-worker log pane) per worker, not just the primary. `served` maps the served
        // id back per worker (local is deduped one-worker-per-model, so served == raw id).
        let served = self.local_served().await;
        let worker_to_served: std::collections::HashMap<WorkerId, String> =
            served.into_iter().map(|(s, (w, _raw))| (w, s)).collect();
        // Live load params come from ONE probe of the PRIMARY — EXACTLY as before this
        // multi-model change (a busy primary's `M_STATUS` already queued once per poll; we add NO
        // new per-worker RPCs). SECONDARY workers are listed from CACHED state (the instance set +
        // the host-side scan), so a busy/wedged secondary never adds an RPC, never stalls the
        // snapshot, and never queues an orphaned status request behind a long generation. (Their
        // live ctx/gpu/threads come as a stub for now; per-worker live stats are the T9 follow-up.)
        let primary_status = self.local.status(primary).await;
        let worker_alive = primary_status.is_ok();
        let primary_info = primary_status
            .ok()
            .and_then(|v| self.loaded_info_from(&v, &scan));

        // `loaded` (back-compat) = the primary's SUCCESSFUL probe ONLY — `None` when it failed or
        // was malformed, preserving the legacy `/api/higgs/status.loaded` contract (a stub must
        // never masquerade as a real loaded model for provider seeding). The `loaded_all` list MAY
        // carry a stub for the primary so it still appears, but `loaded` does not.
        let loaded = primary_info.clone().map(|mut i| {
            if let Some(served_id) = worker_to_served.get(&primary) {
                i.id = served_id.clone();
            }
            i.worker_id = primary.0;
            i
        });

        let mut loaded_all = Vec::with_capacity(instances.len());
        for (worker, raw) in &instances {
            let served_id = worker_to_served.get(worker).cloned().unwrap_or(raw.clone());
            let info = if *worker == primary {
                // Primary in the LIST: full live params from the probe, or a stub if it failed —
                // so a busy/wedged primary still shows a card (its live params just default).
                primary_info
                    .clone()
                    .map(|mut i| {
                        i.id = served_id.clone();
                        i.worker_id = worker.0;
                        i
                    })
                    .unwrap_or_else(|| self.loaded_info_stub(served_id, worker.0, raw, &scan))
            } else {
                self.loaded_info_stub(served_id, worker.0, raw, &scan)
            };
            loaded_all.push(info);
        }

        Ok(HiggsStatus {
            worker_alive,
            loaded,
            loaded_all,
            models_on_disk,
        })
    }

    /// A [`LoadedInfo`] built WITHOUT a worker RPC: `id`/`worker_id` + host-side scan metadata
    /// (`arch`/`quant`/`size_bytes`/`max_context_length`). The live load params
    /// (`ctx_len`/`gpu_layers`/`threads`) default to `0` — [`status`](Self::status) enriches them
    /// from a bounded probe when the worker answers in time. Lets EVERY resident worker appear in
    /// `loaded_all` even while its worker is busy generating (and so can't answer `M_STATUS`).
    fn loaded_info_stub(
        &self,
        served_id: String,
        worker_id: u32,
        raw_model: &str,
        scan: &[HiggsModel],
    ) -> LoadedInfo {
        let scanned = scan.iter().find(|m| m.id == raw_model);
        LoadedInfo {
            id: served_id,
            worker_id,
            ctx_len: 0,
            gpu_layers: 0,
            threads: 0,
            arch: scanned.and_then(|m| m.arch.clone()),
            quant: scanned.and_then(|m| m.quant.clone()),
            max_context_length: scanned.and_then(|m| m.ctx_train),
            size_bytes: scanned.map(|m| m.size_bytes),
            has_chat_template: scanned.map(|m| m.has_chat_template),
            idle_ttl_minutes: None,
        }
    }

    /// Enrich a worker's raw `M_STATUS` value into a [`LoadedInfo`] using the
    /// host-side `scan`. The worker reports `id`/`ctx_len`/`gpu_layers`/`threads`
    /// from the live model, but `arch`/`quant`/`size_bytes`/`max_context_length`/
    /// `has_chat_template` come back null (its store is empty) — those are filled
    /// from the matching scanned [`HiggsModel`]. `None` when nothing is loaded or
    /// the shape is malformed. Shared by [`status`](Self::status) (primary) and
    /// [`local_loaded_info`](Self::local_loaded_info) (a specific served id).
    fn loaded_info_from(&self, v: &serde_json::Value, scan: &[HiggsModel]) -> Option<LoadedInfo> {
        let l = v.get("loaded")?;
        if l.is_null() {
            return None;
        }
        let id = l.get("id")?.as_str()?.to_owned();
        let scanned = scan.iter().find(|m| m.id == id);
        Some(LoadedInfo {
            // None by default; the caller (status/local_loaded_info) sets the real worker id.
            worker_id: 0,
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
        // LOCAL routing FIRST: a locally-resident served id wins — it is faster and
        // keeps routing CONSISTENT with `/v1/models` (which lists local served ids
        // first) and `ensure_loaded` (local-first), so a listed local instance is
        // always reachable even if a remote node happens to expose the same served
        // string. `model` is a SERVED id (`org/model`, `org/model-1`); the worker
        // matches on the RAW model, so the raw id rides the wire.
        //
        // Served suffix ids are EPHEMERAL (a pure function of the live instance set —
        // see `node::served`), so in the narrow window between the serve-layer gate's
        // resolution and this one, an unload/idle-reap can renumber suffixes. This
        // resolves at DISPATCH time (the authoritative current meaning) and the worker
        // runs the exact tokenizer [HG005] check, so a stale gate estimate degrades to
        // the worker's own backstop rather than a wrong answer — never a panic.
        if let Some((worker, raw_model)) = self.local_served().await.remove(&model) {
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
            // Lease the worker's Supervisor for the whole generation. The lease stamps
            // the worker's last-activity on acquire and (on drop) re-stamps + drops the
            // in-flight reference, so the node's idle reaper never unloads a worker
            // mid-chat. A dead/unloaded worker here is a mapped error.
            let lease = self.local.chat_handle(worker).await?;
            // Mint the request, register its keyed sink, and obtain a future that
            // drives the M_CHAT RPC to completion and removes the sink on any outcome —
            // all of it lives in `Supervisor::chat` (reached via the lease's `Deref`).
            // `rx` is returned to the caller now; `call` (and the lease that keeps the
            // worker alive) ride the spawned generation task with the admission permit.
            let (rx, call) = lease.chat(
                raw_model,
                messages_json,
                max_tokens,
                temperature,
                tools_json,
            );
            let handle = tokio::spawn(async move {
                // Hold the admission permit AND the lease for the whole generation;
                // dropping them here (on any return path) releases the gate slot and
                // ends the worker's in-flight hold. Bound to names so neither drops early.
                let _permit = permit;
                let _lease = lease;
                let result = call.await;

                Ok(chat_outcome_from_value(&result?))
            });
            return Ok((rx, handle));
        }

        // REMOTE next: not local, so if a fleet is installed and this model lives on a
        // remote node, relay the chat over that node's transport. A remote model uses
        // NEITHER a local worker NOR the local node's idle timer, and uses a SEPARATE
        // admission gate (so a flood of remote traffic can't grow hub/node tasks
        // unbounded, and never blocks the local idle reaper). Bind the clone to a `let`
        // so the parking_lot guard drops HERE, not held across the `.await` (an `if let`
        // scrutinee temporary would make this !Send).
        let fleet = self.fleet.lock().clone();
        if let Some(fleet) = fleet {
            if fleet.is_remote(&model).await {
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

        // Neither local nor remote serves this id (the serve layer JIT-loads a
        // scanned-but-unloaded model BEFORE calling this, so reaching here means the
        // model was unloaded out from under the request).
        Err(HiggsError::ModelNotFound { id: model })
    }

    /// Resolve the live LOCAL served-id → `(worker, raw model)` map (P4b). N local
    /// workers serving the same raw model coexist as N served ids
    /// (`org/model`, `org/model-1`, …) via the shared
    /// [`served_ids`](crate::node::served::served_ids) algorithm; a chat for a
    /// served id leases its worker and sends the RAW model on the wire.
    async fn local_served(&self) -> std::collections::HashMap<String, (WorkerId, String)> {
        let instances = self.local.instances().await;
        // All local instances share one location marker `()` — collisions are
        // resolved across workers exactly as the remote fleet resolves across nodes.
        let located: Vec<((), WorkerId, String)> =
            instances.iter().map(|(w, m)| ((), *w, m.clone())).collect();
        let by_worker: std::collections::HashMap<WorkerId, String> =
            instances.into_iter().collect();
        crate::node::served::served_ids(&located)
            .into_iter()
            .filter_map(|(served, ((), worker))| {
                by_worker
                    .get(&worker)
                    .map(|raw| (served, (worker, raw.clone())))
            })
            .collect()
    }

    /// Every LOCAL served model id, sorted — the input to `/v1/models` (joined with
    /// the fleet's remote ids by the serve layer).
    pub async fn local_served_ids(&self) -> Vec<String> {
        let mut ids: Vec<String> = self.local_served().await.into_keys().collect();
        ids.sort();
        ids
    }

    /// The LOCAL machine as a first-class [`NodeView`] (`endpoint_id = "local"`, `is_local`,
    /// always connected), so the serve layer can list it alongside the remote fleet in one
    /// shape. Each resident worker is tagged with its `/v1` served id, exactly as the fleet tags
    /// remote workers. `label` is the instance's `config.json` name, passed in by the caller.
    ///
    /// Hardware is sampled fresh (CPU/RAM, ~one update interval) but GPUs are deliberately NOT
    /// enumerated here — that would spawn a transient worker on every poll of this endpoint. The
    /// local machine's full hardware (incl. GPUs) is already served by `GET /api/higgs/system`.
    pub async fn local_node_view(&self, label: String) -> crate::node::fleet::NodeView {
        // ONE instance snapshot drives BOTH the served-id map and the worker list, so a
        // concurrent local load/unload can't make `/api/higgs/nodes` disagree with `/v1/models`
        // (a worker shown with a stale/empty served id). Same `served_ids` algorithm as
        // `local_served`/`local_served_ids`, applied to this single snapshot — no worker spawn.
        let instances = self.local.instances().await;
        let located: Vec<((), WorkerId, String)> =
            instances.iter().map(|(w, m)| ((), *w, m.clone())).collect();
        let by_worker: std::collections::HashMap<u32, String> =
            crate::node::served::served_ids(&located)
                .into_iter()
                .map(|(served, ((), worker))| (worker.0, served))
                .collect();
        let workers: Vec<InventoryWorker> = instances
            .into_iter()
            .map(|(worker, model)| InventoryWorker {
                worker_id: worker.0,
                model,
                served_id: by_worker.get(&worker.0).cloned().unwrap_or_default(),
            })
            .collect();
        // CPU/RAM/engine snapshot with NO GPU enumeration (empty `gpus` → no transient worker).
        let (hardware, runtime) = tokio::task::spawn_blocking(|| {
            crate::system::SystemInfo::gather_hardware_runtime(vec![])
        })
        .await
        .expect("hardware sample task");
        let inventory = NodeInventory {
            hostname: crate::system::hostname(),
            os: std::env::consts::OS.to_string(),
            workers,
            hardware,
            runtime,
        };
        crate::node::fleet::NodeView {
            node_id: 0, // remote NodeIds start at 1; 0 is the local sentinel
            endpoint_id: "local".to_string(),
            connected: true,
            is_local: true,
            label,
            inventory: Some(inventory),
        }
    }

    /// [`LoadedInfo`] for a LOCAL served id — resolves served → worker, reads that
    /// worker's status, and enriches it from the host scan. The reported `id` is the
    /// SERVED id (not the raw model) so a follow-up [`chat_stream`](Self::chat_stream)
    /// resolves the SAME worker. `None` when the id is not locally served. Used by
    /// the serve layer's loaded-model gate ([`ensure_loaded`](crate::serve)).
    pub async fn local_loaded_info(&self, served: &str) -> Option<LoadedInfo> {
        let (worker, _raw) = self.local_served().await.remove(served)?;
        let v = self.local.status(worker).await.ok()?;
        let scan = self.scan().await.unwrap_or_default();
        let mut info = self.loaded_info_from(&v, &scan)?;
        info.id = served.to_owned();
        info.worker_id = worker.0;
        Some(info)
    }

    /// Subscribe to worker lifecycle events (ModelLoaded/ModelUnloaded) from the
    /// local node's event fan-out.
    pub fn events(&self) -> broadcast::Receiver<HiggsEvent> {
        self.local.events()
    }

    /// Return up to `n` recent Developer-Log lines (oldest first), optionally
    /// restricted to one [`LogSource`] (`None` = worker stderr + serve events).
    pub fn logs(&self, n: usize, filter: Option<LogSource>) -> Vec<String> {
        self.local.bus().snapshot(n, filter)
    }

    /// Subscribe to live Developer-Log lines pushed after this call. The SSE
    /// log-stream handler pairs this with [`logs`](Self::logs) for
    /// replay-then-live delivery; filter each by [`LogLine::source`].
    pub fn subscribe_logs(&self) -> tokio::sync::broadcast::Receiver<LogLine> {
        self.local.bus().subscribe()
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
        let gpus = self.local.gpus().await;
        // Cache only a non-empty result: an empty list usually means the gather
        // failed (spawn/EOF/timeout), so leave the cache empty to retry later.
        if !gpus.is_empty() {
            *self.device_cache.lock() = Some(gpus.clone());
        }
        gpus
    }
}

// ── Tests (see api/tests.rs) ──
#[cfg(test)]
mod tests;
