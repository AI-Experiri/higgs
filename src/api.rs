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
use tracing::warn;

use crate::diagnostic::HiggsError;
use crate::log_bus::{LogBus, LogLine, LogSource};
use crate::supervisor::{HiggsEvent, Supervisor};
use crate::worker::engine::LoadParams;
use crate::worker::models::HiggsModel;
use crate::worker::{M_LOAD, M_STATUS, M_UNLOAD};

// ── HiggsConfig ───────────────────────────────────────────────────────────────

higgs_ts! {
    /// Host-supplied configuration (the host maps its own config table onto this).
    #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
    pub struct HiggsConfig {
        /// LM Studio model directories to scan.
        pub lmstudio_dirs: Vec<PathBuf>,
        /// HuggingFace Hub cache directories to scan.
        ///
        /// Note: HuggingFace hardcodes `~/.cache/huggingface/hub` on ALL platforms —
        /// it does not follow XDG or macOS conventions. We use
        /// `dirs::home_dir().join(".cache/huggingface/hub")`, NOT `dirs::cache_dir()`.
        pub hf_dirs: Vec<PathBuf>,
        /// Ollama model store directories to scan.
        pub ollama_dirs: Vec<PathBuf>,
        /// Load parameters used when none are supplied by the caller.
        pub default_load: LoadParams,
    }
}

impl Default for HiggsConfig {
    fn default() -> Self {
        let home = dirs::home_dir();

        // Helper: build a path from home; return empty vec when home is unknown.
        let home_path = |segments: &[&str]| -> Vec<PathBuf> {
            match &home {
                Some(h) => {
                    let mut p = h.clone();
                    for s in segments {
                        p = p.join(s);
                    }
                    vec![p]
                }
                None => vec![],
            }
        };

        let lmstudio_dirs = {
            // LM Studio < 0.3 stores models in ~/.lmstudio/models.
            // LM Studio >= 0.3 uses ~/.cache/lm-studio/models.
            // Higgs scans both; the host can narrow via config.
            let mut dirs = Vec::new();
            if let Some(h) = &home {
                dirs.push(h.join(".lmstudio").join("models"));
                dirs.push(h.join(".cache").join("lm-studio").join("models"));
            }
            dirs
        };

        // HuggingFace hardcodes ~/.cache on ALL platforms — do NOT use dirs::cache_dir().
        let hf_dirs = home_path(&[".cache", "huggingface", "hub"]);

        let ollama_dirs = home_path(&[".ollama", "models"]);

        let threads = {
            let avail = std::thread::available_parallelism()
                .map(std::num::NonZero::get)
                .unwrap_or(1);
            avail.saturating_sub(2).max(1) as u32
        };

        Self {
            lmstudio_dirs,
            hf_dirs,
            ollama_dirs,
            default_load: LoadParams {
                ctx_len: 4096,
                gpu_layers: u32::MAX,
                threads,
                ..Default::default()
            },
        }
    }
}

higgs_ts! {
    /// Read-only snapshot of the serve-layer safety limits, surfaced inside
    /// [`HiggsServerConfig`] so the Server Settings UI can show the actual
    /// hardening posture of the no-auth loopback server. Every field mirrors a
    /// documented `const` in this module (or `serve::mod`); none is yet mutable
    /// at runtime — this is honest disclosure, not a control surface.
    #[derive(Debug, Clone, serde::Serialize)]
    pub struct HiggsLimits {
        /// Max request-body bytes before a `413` (`serve::MAX_BODY_BYTES`).
        #[ts(type = "number")]
        pub max_body_bytes: u64,
        /// Whole-request timeout for the `/api/higgs/*` control surface, in
        /// seconds (`serve::CONTROL_TIMEOUT`). The streaming `/v1` chat path is
        /// deliberately un-timed at the HTTP layer.
        #[ts(type = "number")]
        pub control_timeout_secs: u64,
        /// Worker chat-RPC timeout in seconds — the bound on a single generation
        /// (`supervisor::CHAT_RPC_TIMEOUT`); a timeout is `[HG016]` → 504.
        #[ts(type = "number")]
        pub chat_timeout_secs: u64,
        /// Absolute upper cap on a request's `max_tokens` ([`MAX_OUTPUT_TOKENS`]).
        #[ts(type = "number")]
        pub max_output_tokens: u32,
        /// Max concurrent in-flight chat requests before a `503` busy
        /// ([`MAX_CONCURRENT_INFERENCE`]).
        #[ts(type = "number")]
        pub max_concurrent_inference: u32,
        /// Fraction of available RAM a load may claim before `[HG017]` → 503
        /// ([`MEMORY_HEADROOM_FRACTION`]).
        pub memory_headroom_fraction: f64,
        /// Idle minutes after which the loaded model is auto-unloaded
        /// ([`IDLE_UNLOAD_TTL`]).
        #[ts(type = "number")]
        pub idle_unload_ttl_secs: u64,
    }
}

higgs_ts! {
    /// Read-only snapshot of the server's effective configuration, surfaced at
    /// `GET /api/higgs/system` so the UI can show the real scan dirs, load
    /// defaults, bind host, and safety limits without inventing anything. Derived
    /// entirely from [`HiggsConfig`] plus the fixed invariants ([`BIND_HOST`],
    /// [`DEFAULT_CTX_CAP`]) and the documented serve-layer limit consts; carries
    /// no mutable state and no mutating endpoint.
    #[derive(Debug, Clone, serde::Serialize)]
    pub struct HiggsServerConfig {
        /// Loopback host the listener binds to — always [`BIND_HOST`] (localhost).
        pub bind_host: String,
        /// Configured LM Studio scan directories, as absolute path strings.
        pub lmstudio_dirs: Vec<String>,
        /// Configured HuggingFace Hub cache scan directories, as path strings.
        pub hf_dirs: Vec<String>,
        /// Configured Ollama store scan directories, as path strings.
        pub ollama_dirs: Vec<String>,
        /// Load parameters applied when a load request omits them.
        pub default_load: LoadParams,
        /// Context-window cap applied to an auto (unpinned) load — a huge-context
        /// model's window is the trained length but never exceeds this.
        #[ts(type = "number")]
        pub default_ctx_cap: u32,
        /// Serve-layer safety limits (read-only disclosure of the hardening).
        pub limits: HiggsLimits,
    }
}

// ── Output types ──────────────────────────────────────────────────────────────

higgs_ts! {
    /// Info about the currently loaded model.
    #[derive(Debug, Clone, serde::Serialize)]
    pub struct LoadedInfo {
        /// HuggingFace repo id of the resident model.
        pub id: String,
        /// Context window size in tokens.
        #[ts(type = "number")]
        pub ctx_len: u32,
        /// GPU layers offloaded; u32::MAX means all.
        #[ts(type = "number")]
        pub gpu_layers: u32,
        /// Worker threads used during generation.
        #[ts(type = "number")]
        pub threads: u32,
        // Model metadata from the store — present when the worker has scanned the model.
        /// Model architecture read from GGUF header (e.g. `"llama"`, `"gemma3"`).
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        pub arch: Option<String>,
        /// Quantization tag (e.g. `Q4_K_M`), if present.
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        pub quant: Option<String>,
        /// Training context length from GGUF header (model's maximum). Distinct from
        /// `ctx_len` which is the actually loaded window size.
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(type = "number")]
        #[ts(optional)]
        pub max_context_length: Option<u64>,
        /// File size in bytes.
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(type = "number")]
        #[ts(optional)]
        pub size_bytes: Option<u64>,
        /// Whether `tokenizer.chat_template` is present in the GGUF header.
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        pub has_chat_template: Option<bool>,
        /// Active per-load idle-TTL override in minutes, if one was set at load
        /// time. Absent when the loaded model uses the global idle TTL.
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(type = "number")]
        #[ts(optional)]
        pub idle_ttl_minutes: Option<u64>,
    }
}

higgs_ts! {
    /// Live status snapshot returned by [`Higgs::status`].
    #[derive(Debug, Clone, serde::Serialize)]
    pub struct HiggsStatus {
        /// Whether the worker process is currently alive.
        pub worker_alive: bool,
        /// Info about the loaded model, if any.
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        pub loaded: Option<LoadedInfo>,
        /// Number of models discovered in the last scan.
        #[ts(type = "number")]
        pub models_on_disk: u32,
    }
}

/// Build a [`ChatOutcome`] from a worker `M_CHAT` result `Value` (same shape whether the
/// worker is local or relayed from a remote node).
pub(crate) fn chat_outcome_from_value(result: &serde_json::Value) -> ChatOutcome {
    ChatOutcome {
        content: result.get("content").and_then(|v| v.as_str()).unwrap_or("").to_owned(),
        finish_reason: result
            .get("finish_reason")
            .and_then(|v| v.as_str())
            .unwrap_or("stop")
            .to_owned(),
        tool_calls: result.get("tool_calls").filter(|v| !v.is_null()).cloned(),
        prompt_tokens: result.get("prompt_tokens").and_then(serde_json::Value::as_u64).unwrap_or(0) as u32,
        completion_tokens: result
            .get("completion_tokens")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0) as u32,
    }
}

/// Final outcome of a completed chat request.
#[derive(Debug, Clone)]
pub struct ChatOutcome {
    /// Assistant text after tool-call parsing (the OpenAI message `content`).
    pub content: String,
    /// OpenAI finish_reason ("stop" or "length"); the boundary upgrades this to
    /// "tool_calls" when [`tool_calls`](Self::tool_calls) is present.
    pub finish_reason: String,
    /// Parsed OpenAI `tool_calls` array, or `None` when the turn emitted none.
    pub tool_calls: Option<serde_json::Value>,
    /// Prompt token count from the engine (for OpenAI `usage.prompt_tokens`).
    pub prompt_tokens: u32,
    /// Completion token count from the engine (for OpenAI `usage.completion_tokens`).
    pub completion_tokens: u32,
}

// ── Higgs ─────────────────────────────────────────────────────────────────────

/// Default context-window cap used when a load does not pin `ctx_len`: the
/// model's trained context is used but never exceeds this, so a huge-context
/// model doesn't allocate an enormous KV cache by default. A caller (the UI)
/// can still request the full trained window explicitly.
pub const DEFAULT_CTX_CAP: u32 = 32_768;

/// Loopback address the embedded higgs listener binds to.
///
/// higgs is localhost-only by contract: the launcher binds `127.0.0.1:0`
/// (ephemeral port) and never exposes the server on a routable interface. The
/// port varies per boot, but the host is this fixed invariant — surfaced in the
/// read-only [`HiggsServerConfig`] so the UI can state "Network: localhost only"
/// honestly. Single home for the bind host.
pub const BIND_HOST: &str = "127.0.0.1";

/// Maximum number of chat/inference requests admitted concurrently. The worker
/// executes generations serially (single-threaded stdin loop — see
/// `concurrency.md`), so additional admitted requests queue at the worker; this
/// gate caps how many may queue at once so the no-auth loopback server can't be
/// flooded into unbounded memory/queue growth. A full gate returns
/// [`HiggsError::ServerBusy`] → HTTP 503 (vllm/ollama "all slots busy" capacity
/// signal). Scoped to the inference path only (control RPCs are unaffected).
///
/// 8 is the documented higgs value: generous headroom above the single-sequence
/// worker (ollama's `OLLAMA_NUM_PARALLEL` default is 1, `OLLAMA_MAX_QUEUE` 512)
/// while still bounding a flood. Grouped here with the other serve-layer limits
/// for a later lift into `HiggsConfig` + the Server Settings UI; the true
/// parallel-execution work (`max_concurrent_requests` in `concurrency.md`) is a
/// separate, deferred effort and does not change this admission ceiling.
pub const MAX_CONCURRENT_INFERENCE: usize = 8;

// ── Phase B hardening knobs ─────────────────────────────────────────────────
//
// Documented `const`s (not config yet — a later phase lifts the user-facing
// ones into `HiggsConfig` + the Server Settings UI). Grouped here so that lift
// is a mechanical move, alongside [`MAX_CONCURRENT_INFERENCE`] above.

/// Fraction of currently-available system RAM a model load is allowed to claim
/// before it is refused with [`HiggsError::InsufficientMemory`] (HTTP 503). The
/// estimated need (the GGUF file size on disk, a lower-bound proxy for resident
/// weights) must not exceed `available_ram * MEMORY_HEADROOM_FRACTION`, leaving
/// the remaining `1 − fraction` as headroom for the KV cache, compute buffers,
/// and the rest of the system. 0.8 is ollama's placement rule verbatim
/// (`server/sched.go`: `predictedForLoad > freeMemory*80/100` — "Use 80% of free
/// memory as threshold to leave headroom").
pub const MEMORY_HEADROOM_FRACTION: f64 = 0.8;

/// Idle period with no inference after which the loaded model is auto-unloaded
/// to free memory. The idle reaper (spawned by [`Higgs::start`]) checks at
/// [`IDLE_REAP_INTERVAL`] and unloads once the time since the last chat exceeds
/// this. 5 minutes is ollama's `keep_alive` default verbatim
/// (`envconfig/config.go`: `keepAlive = 5 * time.Minute`).
pub const IDLE_UNLOAD_TTL: std::time::Duration = std::time::Duration::from_secs(5 * 60);

/// Default idle auto-unload TTL in minutes — the seed for the runtime-mutable
/// [`Higgs::idle_ttl_minutes`] atomic. Mirrors [`IDLE_UNLOAD_TTL`] (5 minutes,
/// ollama `keep_alive`) expressed in the minutes unit the Server Settings UI
/// edits. Kept as the const default; the live value is read from the atomic so
/// a runtime change takes effect without a restart.
pub const IDLE_UNLOAD_TTL_MINUTES: u64 = 5;

/// How often the idle reaper wakes to compare idle time against
/// [`IDLE_UNLOAD_TTL`]. No reference fixes a poll cadence (ollama uses a timer
/// per loaded model rather than a poll); 30 s is the documented higgs value — a
/// negligible wakeup that bounds the post-TTL unload latency to at most this
/// interval. Grouped with the other Phase B knobs for a later config lift.
pub const IDLE_REAP_INTERVAL: std::time::Duration = std::time::Duration::from_secs(30);

/// A Gate-1 support verdict: `(loadable, reason)` where `reason` is the engine's
/// verbatim load error when `!loadable`, else `None`.
pub(crate) type SupportVerdict = (bool, Option<String>);

/// Support-cache key: `(architecture, quant, engine_version)`. One verdict per
/// distinct `(arch, quant)` for a given engine version (NOT per file).
pub(crate) type SupportKey = (String, String, String);

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
        // Stamp last-activity so the idle reaper never unloads a model that is
        // actively serving. Done before the admission gate so even a request
        // that ends up rejected (ServerBusy) still counts as recent activity —
        // a busy server is by definition not idle. Lock held for one `Instant`
        // write only, never across `.await`.
        *self.last_activity.lock() = std::time::Instant::now();

        // Remote routing: if a fleet is installed and this model lives on a remote node,
        // relay the chat over that node's transport instead of the local worker. Remote
        // capacity is the node's concern, so this bypasses the LOCAL admission gate. The
        // remote final `Value` has the same shape as a local worker result.
        // Bind the clone to a `let` so the parking_lot guard drops HERE, not held across
        // the `.await` below (an `if let` scrutinee temporary would, making this !Send).
        let fleet = self.fleet.lock().clone();
        if let Some(fleet) = fleet {
            if fleet.is_remote(&model) {
                let (rx, fut) = fleet
                    .chat(&model, messages_json, max_tokens, temperature, tools_json)
                    .await?;
                let handle =
                    tokio::spawn(async move { Ok(chat_outcome_from_value(&fut.await?)) });
                return Ok((rx, handle));
            }
        }

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

/// Validate a model id's charset before it is used to resolve a filesystem path.
///
/// Identity is the HuggingFace repo id (`org/model`); Ollama-sourced ids keep
/// their `ollama/{name}:{tag}` form (see [`HiggsModel::id`]). The accepted
/// charset mirrors ollama's byte-level name validation
/// (`types/model/name.go`): ASCII alphanumerics plus `_ - . / :`. The structural
/// separators `/` (org/model) and `:` (ollama tag) are permitted, but a path
/// component of exactly `..` is rejected outright — that is the traversal vector
/// — as are empty ids and absolute paths. `Err(HiggsError::InvalidModelId)` (→
/// 400) on any violation.
fn validate_repo_id(id: &str) -> Result<(), HiggsError> {
    let reject = |reason: &str| {
        Err(HiggsError::InvalidModelId {
            id: id.to_owned(),
            reason: reason.to_owned(),
        })
    };
    if id.is_empty() {
        return reject("id is empty");
    }
    if id.starts_with('/') || id.starts_with('\\') {
        return reject("id must not be an absolute path");
    }
    if id.contains('\0') {
        return reject("id contains a NUL byte");
    }
    for ch in id.chars() {
        let ok = ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.' | '/' | ':');
        if !ok {
            return reject(&format!("id contains an illegal character {ch:?}"));
        }
    }
    // A `..` path component is the traversal vector — reject it even though its
    // bytes are individually legal. Split on both separators a path may use.
    if id.split(['/', '\\']).any(|seg| seg == "..") {
        return reject("id contains a `..` path component");
    }
    Ok(())
}

/// Whether `path` canonicalizes to a location inside one of `roots`.
///
/// Both sides are canonicalized (resolving symlinks and `..`) before the prefix
/// comparison, so a symlink or `..` that escapes a root is caught. A root that
/// does not exist on disk is skipped (a missing scan dir is legitimate — see
/// `HiggsConfig`). Returns `false` when `path` itself cannot be canonicalized
/// (e.g. it does not exist) — a non-existent resolved model path is not a valid
/// load target.
pub(crate) fn path_within_roots(path: &str, roots: &[PathBuf]) -> bool {
    let Ok(canon_path) = std::fs::canonicalize(path) else {
        return false;
    };
    roots.iter().any(|root| {
        std::fs::canonicalize(root)
            .map(|canon_root| canon_path.starts_with(&canon_root))
            .unwrap_or(false)
    })
}

/// Currently-available system RAM in bytes, via the same `sysinfo` path that
/// backs `GET /api/higgs/system`. "Available" (not merely "free") is what an
/// allocation can realistically claim, so it is the right basis for the load
/// headroom guard. A fresh `System` is sampled per call (loads are infrequent,
/// so the sampling cost is irrelevant) — no shared state to keep coherent.
fn available_system_memory() -> u64 {
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    sys.available_memory()
}

/// Whether a load needing `needed_bytes` fits within the safe RAM headroom over
/// `available_bytes` — i.e. `needed <= available * MEMORY_HEADROOM_FRACTION`
/// (ollama's `predictedForLoad <= freeMemory*80/100` placement rule). Pure
/// arithmetic, factored out of [`Higgs::load`] so the threshold is unit-testable
/// without provisioning multi-gigabyte fixtures.
fn fits_in_memory(needed_bytes: u64, available_bytes: u64) -> bool {
    #[allow(clippy::cast_precision_loss)]
    let safe = (available_bytes as f64) * MEMORY_HEADROOM_FRACTION;
    (needed_bytes as f64) <= safe
}

/// Pre-load RAM headroom guard: refuse a load whose estimated memory need
/// (`needed_bytes`, the GGUF file size on disk — a lower-bound proxy for
/// resident weights) exceeds [`MEMORY_HEADROOM_FRACTION`] of currently-available
/// system RAM. Checked before spawning a worker so an oversized load fails fast
/// with `Err` [HG017] `InsufficientMemory` → 503 (retryable) instead of
/// OOM-killing the worker (an opaque [HG004]/[HG006]). The same `sysinfo` path
/// that backs `GET /api/higgs/system` reads available memory.
pub(crate) fn guard_memory_headroom(id: &str, needed_bytes: u64) -> Result<(), HiggsError> {
    let available = available_system_memory();
    if fits_in_memory(needed_bytes, available) {
        return Ok(());
    }
    warn!(
        id,
        needed_bytes,
        available_bytes = available,
        "higgs: refusing load — insufficient memory headroom"
    );
    Err(HiggsError::InsufficientMemory {
        id: id.to_owned(),
        needed_bytes,
        available_bytes: available,
        headroom_fraction: MEMORY_HEADROOM_FRACTION,
    })
}

/// Idle reaper: every [`IDLE_REAP_INTERVAL`], unload the loaded model once the
/// time since the last chat exceeds the runtime idle TTL (ollama `keep_alive`).
///
/// The TTL and the on/off switch are read from the live atoms
/// ([`Higgs::auto_unload_idle`], [`Higgs::idle_ttl_minutes`]) on EVERY tick, so a
/// Server-Settings change takes effect without a restart: when auto-unload is
/// off the reaper skips entirely, and the TTL is `idle_ttl_minutes` minutes
/// (seeded from [`IDLE_UNLOAD_TTL`]). A per-load override
/// ([`Higgs::loaded_idle_ttl_override`], set at load time and cleared on unload)
/// takes precedence over the global TTL for the currently-loaded model.
///
/// Holds a `Weak<Higgs>` so it terminates when the host drops its `Arc<Higgs>`.
/// It never unloads mid-generation and never races a just-admitted chat: the
/// reaper atomically acquires ALL [`MAX_CONCURRENT_INFERENCE`] inference permits
/// before unloading. Holding every permit proves zero in-flight requests AND
/// blocks any new `chat_stream` admission until the unload finishes and the
/// permits drop. An in-flight request also re-stamps `last_activity`, so a long
/// generation keeps the model resident regardless.
/// The idle [`Instant`] is copied out from under the `parking_lot` guard before
/// any `.await`, honoring the never-hold-a-lock-across-await rule; the unload
/// itself runs through the existing [`Higgs::unload`] path (which takes the
/// lifecycle mutex), so it serializes correctly against a concurrent load.
async fn idle_reaper(weak: std::sync::Weak<Higgs>) {
    loop {
        tokio::time::sleep(IDLE_REAP_INTERVAL).await;
        // Upgrade per tick: a failed upgrade means the host dropped Higgs — exit.
        let Some(higgs) = weak.upgrade() else {
            return;
        };
        // Auto-unload disabled at runtime → never reap. Read each tick so a live
        // toggle takes effect without a restart.
        if !higgs.auto_unload_idle() {
            continue;
        }
        // Read the effective TTL each tick (minutes → Duration). A per-load
        // override (set at load time, cleared on unload) wins over the global
        // runtime TTL for the currently-loaded model; otherwise the global TTL
        // (seeded from IDLE_UNLOAD_TTL) applies. Read each tick so a live change
        // to either takes effect immediately.
        let mins = higgs
            .loaded_idle_ttl_override()
            .unwrap_or_else(|| higgs.idle_ttl_minutes());
        let ttl = std::time::Duration::from_secs(mins * 60);
        // Copy the idle instant out under the lock, then drop the guard before
        // any await (never hold a parking_lot lock across .await).
        let idle_for = {
            let last = *higgs.last_activity.lock();
            last.elapsed()
        };
        if idle_for < ttl {
            continue;
        }
        // Don't unload while any inference is in flight, and don't let a new
        // request slip in mid-unload (TOCTOU). Acquiring ALL permits is atomic
        // proof of both: success means zero in-flight AND — because the reaper
        // now holds every permit — no `chat_stream` can `try_acquire_owned`
        // until we drop them after the unload completes. A failure means a
        // generation is running (it holds a permit); skip this tick. (A running
        // request also re-stamps `last_activity`, so a long generation keeps the
        // model resident regardless.)
        let Ok(_all_permits) = Arc::clone(&higgs.inference_gate)
            .try_acquire_many_owned(MAX_CONCURRENT_INFERENCE as u32)
        else {
            continue;
        };
        // Only unload if a model is actually loaded — otherwise the unload path
        // would needlessly kill an idle (modelless) worker or no-op. A dead/
        // empty worker reports `loaded: None`. The held permits gate out any new
        // chat for the duration of this status+unload.
        match higgs.status().await {
            Ok(st) if st.loaded.is_some() => {
                warn!(
                    idle_secs = idle_for.as_secs(),
                    "higgs: auto-unloading idle model (keep_alive TTL exceeded)"
                );
                if let Err(e) = higgs.unload().await {
                    warn!(error = %e, "higgs: idle auto-unload failed");
                }
            }
            _ => { /* nothing loaded, or status unavailable — nothing to reap */ }
        }
        // `_all_permits` drops here, reopening the gate for new chats.
        // Drop the strong ref before sleeping so we never pin Higgs alive.
        drop(higgs);
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::supervisor::WorkerHalves;
    use crate::worker::N_CHAT_CHUNK;
    use parking_lot::Mutex;
    use serde_json::json;
    use tokio::io::AsyncWriteExt;

    // ── Test seam (mirrored from supervisor::tests::make_supervisor) ──────────

    /// Build a `Supervisor` plus duplex test handles.
    fn make_supervisor() -> (
        Supervisor,
        tokio::io::DuplexStream, // test_write: write responses → supervisor reads
        tokio::io::DuplexStream, // test_read:  supervisor writes requests → test reads
    ) {
        let (sup_write, test_read) = tokio::io::duplex(64 * 1024);
        let (test_write, sup_read) = tokio::io::duplex(64 * 1024);

        let sup_write_cell = Arc::new(Mutex::new(Some(sup_write)));
        let sup_read_cell = Arc::new(Mutex::new(Some(sup_read)));

        let sup = Supervisor::with_factory(Box::new(move |_ring, _model| {
            let write =
                sup_write_cell
                    .lock()
                    .take()
                    .ok_or_else(|| HiggsError::WorkerSpawnFailed {
                        source: std::io::Error::other("mock: no more write halves"),
                    })?;
            let read =
                sup_read_cell
                    .lock()
                    .take()
                    .ok_or_else(|| HiggsError::WorkerSpawnFailed {
                        source: std::io::Error::other("mock: no more read halves"),
                    })?;
            Ok(WorkerHalves {
                write: Box::new(write),
                read: Box::new(read),
                proc: None,
            })
        }));

        sup.start_for("test-model").expect("mock start");
        (sup, test_write, test_read)
    }

    async fn write_response(
        stream: &mut tokio::io::DuplexStream,
        id: u64,
        result: serde_json::Value,
    ) {
        use crate::rpc::{encode, RpcFrame, RpcResponse};
        let line = encode(&RpcFrame::Response(RpcResponse {
            jsonrpc: "2.0".into(),
            id,
            result: Some(result),
            error: None,
        }));
        stream
            .write_all(format!("{line}\n").as_bytes())
            .await
            .unwrap();
        stream.flush().await.unwrap();
    }

    // ── Phase A2: repo-id charset + path-traversal guards ────────────────────

    #[test]
    fn validate_repo_id_accepts_legitimate_ids() {
        for id in [
            "org/model",
            "lmstudio-community/NVIDIA-Nemotron-3-Nano-4B-GGUF",
            "ollama/llama3:8b",
            "google/gemma-3.1-12b",
        ] {
            validate_repo_id(id).unwrap_or_else(|e| panic!("{id} should be valid: {e}"));
        }
    }

    #[test]
    fn validate_repo_id_rejects_traversal_and_illegal() {
        for (id, why) in [
            ("", "empty"),
            ("/etc/passwd", "absolute"),
            ("..", "dotdot"),
            ("org/../../etc/passwd", "embedded dotdot"),
            ("org/model;rm -rf", "illegal char"),
            ("org/model\0", "nul"),
        ] {
            let err = validate_repo_id(id).expect_err(why);
            assert!(
                matches!(err, HiggsError::InvalidModelId { .. }),
                "{why}: {err}"
            );
            assert!(err.to_string().starts_with("[HG015]"), "{why}: {err}");
        }
    }

    #[test]
    fn path_within_roots_contains_and_escapes() {
        let root = tempfile::TempDir::new().unwrap();
        let inside = root.path().join("org").join("m.gguf");
        std::fs::create_dir_all(inside.parent().unwrap()).unwrap();
        std::fs::write(&inside, b"x").unwrap();
        let roots = vec![root.path().to_path_buf()];
        assert!(path_within_roots(inside.to_str().unwrap(), &roots));

        // A path outside every root (a different temp dir) is rejected.
        let other = tempfile::TempDir::new().unwrap();
        let outside = other.path().join("escape.gguf");
        std::fs::write(&outside, b"x").unwrap();
        assert!(!path_within_roots(outside.to_str().unwrap(), &roots));

        // A non-existent path can't canonicalize → rejected.
        assert!(!path_within_roots("/nope/does/not/exist.gguf", &roots));
    }

    /// `load` rejects an id that fails charset validation with [HG015], before
    /// any scan or worker RPC.
    #[tokio::test]
    async fn load_rejects_traversal_id() {
        let higgs = Higgs::new(HiggsConfig {
            lmstudio_dirs: vec![],
            hf_dirs: vec![],
            ollama_dirs: vec![],
            default_load: HiggsConfig::default().default_load,
        });
        let err = higgs
            .load("org/../../etc/passwd", None)
            .await
            .expect_err("traversal id must be rejected");
        assert!(matches!(err, HiggsError::InvalidModelId { .. }));
    }

    /// `probe_support` returns a cached `(arch, quant)` verdict WITHOUT probing.
    ///
    /// The cache is pre-seeded for the current engine version; the rep's combo is
    /// a hit, so `probe_paths` (which would spawn-fail under the mock factory and
    /// yield a `false` verdict) is never consulted — the returned verdict is the
    /// seeded `(true, None)`. A second combo is a miss and goes to the probe path,
    /// proving the partition.
    #[tokio::test]
    async fn probe_support_cache_hit_skips_probe() {
        let (sup, _tw, _tr) = make_supervisor();
        let higgs = Higgs {
            sup: Arc::new(sup),
            config: parking_lot::Mutex::new(HiggsConfig::default()),
            lifecycle: tokio::sync::Mutex::new(()),
            inference_gate: Arc::new(tokio::sync::Semaphore::new(MAX_CONCURRENT_INFERENCE)),
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
        };
        let ev = crate::worker::engine::llamacpp::engine_version();
        // Seed a HIT for (llama, Q4_K_M, <this engine version>).
        higgs
            .probe_cache
            .lock()
            .insert(("llama".into(), "Q4_K_M".into(), ev.clone()), (true, None));
        let out = higgs
            .probe_support(vec![
                // Hit: returns the seeded verdict, no probe.
                ("llama".into(), "Q4_K_M".into(), "/seeded/path.gguf".into()),
                // Miss: probe path (mock factory spawn-fails) → false verdict.
                ("gemma4".into(), "Q8_0".into(), "/miss/path.gguf".into()),
            ])
            .await;
        assert_eq!(
            out.get(&("llama".into(), "Q4_K_M".into())),
            Some(&(true, None))
        );
        let (miss_loadable, miss_reason) = out
            .get(&("gemma4".into(), "Q8_0".into()))
            .cloned()
            .expect("miss combo present");
        assert!(
            !miss_loadable,
            "miss combo is not loadable under spawn-fail"
        );
        assert!(miss_reason.is_some(), "miss carries a reason");
        // The miss verdict was stored (the probe path inserts under the version
        // the worker reported — empty here because spawn failed before any reply).
        let _ = ev;
        assert!(higgs
            .probe_cache
            .lock()
            .keys()
            .any(|(a, q, _)| a == "gemma4" && q == "Q8_0"));
    }

    /// The inference admission gate returns `ServerBusy` once all permits are
    /// taken; releasing a permit re-opens a slot.
    #[tokio::test]
    async fn inference_gate_rejects_when_full() {
        let (sup, _tw, _tr) = make_supervisor();
        let higgs = Higgs {
            sup: Arc::new(sup),
            config: parking_lot::Mutex::new(HiggsConfig::default()),
            lifecycle: tokio::sync::Mutex::new(()),
            // One-slot gate so the test deterministically fills it.
            inference_gate: Arc::new(tokio::sync::Semaphore::new(1)),
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
        };
        // Take the only permit and hold it.
        let held = Arc::clone(&higgs.inference_gate)
            .try_acquire_owned()
            .expect("first permit");
        // chat_stream must now fail fast with ServerBusy (no worker RPC).
        let err = higgs
            .chat_stream(
                "org/model".to_owned(),
                r#"[{"role":"user","content":"hi"}]"#.to_owned(),
                8,
                0.0,
                None,
            )
            .await
            .expect_err("gate full → ServerBusy");
        assert!(matches!(err, HiggsError::ServerBusy { .. }), "got {err}");
        drop(held);
        // With the permit released, a request is admitted again (it then fails
        // later for lack of a real worker response, but admission succeeds).
        assert!(
            Arc::clone(&higgs.inference_gate)
                .try_acquire_owned()
                .is_ok(),
            "slot re-opens after release"
        );
    }

    // ── Phase B: RAM headroom guard arithmetic ───────────────────────────────

    #[test]
    fn fits_in_memory_respects_headroom_fraction() {
        // Exactly at 80% of available fits; one byte over does not.
        let available = 10_000_000_000u64; // 10 GB
        let safe = 8_000_000_000u64; // 80%
        assert!(fits_in_memory(safe, available), "at the threshold fits");
        assert!(
            fits_in_memory(safe - 1, available),
            "below the threshold fits"
        );
        assert!(
            !fits_in_memory(safe + 1, available),
            "above the threshold is refused"
        );
        // A model larger than ALL available RAM is always refused.
        assert!(!fits_in_memory(available + 1, available));
        // Zero-size edge: always fits.
        assert!(fits_in_memory(0, available));
    }

    /// `load` refuses a model whose file size exceeds the RAM headroom with
    /// [HG017], before spawning a worker. Uses a fixture whose declared size is
    /// forced over the limit by checking against a tiny synthetic available
    /// value is not possible through `load` (it reads real RAM), so this asserts
    /// the typed-error path via the pure guard plus the diagnostic wiring; the
    /// end-to-end refusal is covered by `fits_in_memory_respects_headroom_fraction`
    /// and the HG017 status-mapping test in `serve::mod`.
    #[test]
    fn insufficient_memory_diagnostic_is_503_capacity() {
        let err = HiggsError::InsufficientMemory {
            id: "org/model".into(),
            needed_bytes: 8_000_000_000,
            available_bytes: 4_000_000_000,
            headroom_fraction: MEMORY_HEADROOM_FRACTION,
        };
        assert!(err.to_string().starts_with("[HG017]"));
    }

    // ── Verbose toggle: default false, set/get round-trip ────────────────────

    #[test]
    fn verbose_defaults_false_and_round_trips() {
        let higgs = Higgs::new(HiggsConfig::default());
        assert!(!higgs.verbose(), "verbose defaults to false");
        higgs.set_verbose(true);
        assert!(higgs.verbose(), "set_verbose(true) is observed");
        higgs.set_verbose(false);
        assert!(!higgs.verbose(), "set_verbose(false) is observed");
    }

    // ── Log-incoming-tokens toggle: default false, set/get round-trip ─────────

    #[test]
    fn log_incoming_tokens_defaults_false_and_round_trips() {
        let higgs = Higgs::new(HiggsConfig::default());
        assert!(
            !higgs.log_incoming_tokens(),
            "log_incoming_tokens defaults to false"
        );
        higgs.set_log_incoming_tokens(true);
        assert!(
            higgs.log_incoming_tokens(),
            "set_log_incoming_tokens(true) is observed"
        );
        higgs.set_log_incoming_tokens(false);
        assert!(
            !higgs.log_incoming_tokens(),
            "set_log_incoming_tokens(false) is observed"
        );
    }

    // ── JIT toggle: default TRUE, set/get round-trip ────────────────────────

    #[test]
    fn jit_enabled_defaults_true_and_round_trips() {
        let higgs = Higgs::new(HiggsConfig::default());
        assert!(higgs.jit_enabled(), "JIT defaults to ON (true)");
        higgs.set_jit_enabled(false);
        assert!(!higgs.jit_enabled(), "set_jit_enabled(false) is observed");
        higgs.set_jit_enabled(true);
        assert!(higgs.jit_enabled(), "set_jit_enabled(true) is observed");
    }

    // ── Idle auto-unload toggles: defaults + round-trip ──────────────────────

    #[test]
    fn idle_unload_settings_default_and_round_trip() {
        let higgs = Higgs::new(HiggsConfig::default());
        // Defaults: auto-unload ON, TTL 5 minutes (seeded from IDLE_UNLOAD_TTL).
        assert!(
            higgs.auto_unload_idle(),
            "auto-unload defaults to ON (true)"
        );
        assert_eq!(higgs.idle_ttl_minutes(), 5, "TTL defaults to 5 minutes");
        assert_eq!(
            IDLE_UNLOAD_TTL_MINUTES * 60,
            IDLE_UNLOAD_TTL.as_secs(),
            "minutes seed must equal the Duration const"
        );

        higgs.set_auto_unload_idle(false);
        assert!(!higgs.auto_unload_idle(), "set_auto_unload_idle(false)");
        higgs.set_auto_unload_idle(true);
        assert!(higgs.auto_unload_idle(), "set_auto_unload_idle(true)");

        higgs.set_idle_ttl_minutes(30);
        assert_eq!(higgs.idle_ttl_minutes(), 30, "set_idle_ttl_minutes(30)");
    }

    // ── Per-load idle-TTL override: default None, set/clear round-trip ────────

    #[test]
    fn loaded_idle_ttl_override_defaults_none_and_round_trips() {
        let higgs = Higgs::new(HiggsConfig::default());
        // No override by default (0 in the atomic reads back as None).
        assert_eq!(
            higgs.loaded_idle_ttl_override(),
            None,
            "override defaults to None"
        );
        // Set an override → reads back as Some(n).
        higgs.set_loaded_idle_ttl_override(Some(30));
        assert_eq!(
            higgs.loaded_idle_ttl_override(),
            Some(30),
            "set Some(30) is observed"
        );
        // Clear with None → back to None.
        higgs.set_loaded_idle_ttl_override(None);
        assert_eq!(
            higgs.loaded_idle_ttl_override(),
            None,
            "set None clears the override"
        );
    }

    /// The reaper's effective-TTL expression prefers the per-load override over
    /// the global `idle_ttl_minutes` — the exact `unwrap_or_else` the reaper runs.
    #[test]
    fn reaper_prefers_loaded_override_over_global_ttl() {
        let higgs = Higgs::new(HiggsConfig::default());
        // Global TTL is 5 (the default); with no override the effective value is
        // the global TTL.
        let effective = |h: &Higgs| {
            h.loaded_idle_ttl_override()
                .unwrap_or_else(|| h.idle_ttl_minutes())
        };
        assert_eq!(effective(&higgs), 5, "no override → global TTL (5)");
        // With an override set, it wins regardless of the global TTL.
        higgs.set_loaded_idle_ttl_override(Some(42));
        assert_eq!(effective(&higgs), 42, "override (42) wins over global");
        // Clearing the override falls back to the global TTL again.
        higgs.set_loaded_idle_ttl_override(None);
        assert_eq!(effective(&higgs), 5, "cleared → global TTL (5) again");
    }

    // ── Serving on/off gate: default true, set/get round-trip ─────────────────

    #[test]
    fn serving_enabled_defaults_true_and_round_trips() {
        let higgs = Higgs::new(HiggsConfig::default());
        assert!(higgs.serving_enabled(), "serving defaults to ON (true)");
        higgs.set_serving_enabled(false);
        assert!(!higgs.serving_enabled(), "set_serving_enabled(false)");
        higgs.set_serving_enabled(true);
        assert!(higgs.serving_enabled(), "set_serving_enabled(true)");
    }

    // ── Reaper respects the runtime auto-unload toggle and TTL ────────────────
    //
    // The reaper reads `auto_unload_idle` and `idle_ttl_minutes` from the live
    // atoms each tick. These tests drive the same decision predicate the reaper
    // uses (read flag → skip if off; read TTL → skip if idle_for < ttl) against a
    // facade whose runtime values are set via the public accessors, proving a
    // change takes effect without a restart.

    #[test]
    fn reaper_skips_when_auto_unload_disabled() {
        let higgs = Higgs::new(HiggsConfig::default());
        higgs.set_auto_unload_idle(false);
        // With auto-unload off, the reaper's first guard short-circuits: no
        // unload regardless of how long the model has been idle.
        assert!(!higgs.auto_unload_idle(), "reaper would skip this tick");
    }

    #[test]
    fn reaper_uses_runtime_ttl_not_const() {
        let higgs = Higgs::new(HiggsConfig::default());
        // Raise the TTL to 60 minutes at runtime.
        higgs.set_idle_ttl_minutes(60);
        let ttl = std::time::Duration::from_secs(higgs.idle_ttl_minutes() * 60);
        // A model idle for 10 minutes is BELOW the new 60-minute TTL → not reaped,
        // even though it exceeds the old 5-minute const default.
        let idle_for = std::time::Duration::from_secs(10 * 60);
        assert!(idle_for < ttl, "runtime TTL (60m) keeps a 10m-idle model");
        assert!(
            idle_for > IDLE_UNLOAD_TTL,
            "the same 10m idle WOULD reap under the old 5m const — proving the \
             reaper must read the runtime value, not the const"
        );
    }

    // ── Test 1: default config paths ─────────────────────────────────────────

    #[test]
    fn default_config_paths() {
        let home = match dirs::home_dir() {
            Some(h) => h,
            None => return, // skip if no home dir
        };

        let cfg = HiggsConfig::default();

        let has_suffix = |dirs: &[PathBuf], suffix: &str| dirs.iter().any(|p| p.ends_with(suffix));

        assert!(
            has_suffix(&cfg.lmstudio_dirs, ".lmstudio/models")
                || cfg
                    .lmstudio_dirs
                    .iter()
                    .any(|p| p.ends_with("lm-studio/models")),
            "lmstudio_dirs should contain .lmstudio/models or lm-studio/models"
        );
        assert!(
            cfg.hf_dirs
                .iter()
                .any(|p| { p == &home.join(".cache").join("huggingface").join("hub") }),
            "hf_dirs must use ~/.cache/huggingface/hub (not XDG cache_dir)"
        );
        assert!(
            cfg.ollama_dirs
                .iter()
                .any(|p| p.ends_with(".ollama/models")),
            "ollama_dirs should contain .ollama/models"
        );
    }

    // ── Test 2: scan runs host-side with no worker ───────────────────────────

    /// `scan()` runs host-side (pure Rust, no worker RPC): with a fresh facade
    /// that never spawned a worker and empty config dirs, it returns `Ok(empty)`.
    #[tokio::test]
    async fn scan_runs_host_side_without_worker() {
        // Empty config dirs → nothing to scan → Ok(empty). The point is that no
        // worker is live (start() never called) yet scan succeeds.
        let higgs = Higgs::new(HiggsConfig {
            lmstudio_dirs: vec![],
            hf_dirs: vec![],
            ollama_dirs: vec![],
            default_load: HiggsConfig::default().default_load,
        });

        let models = higgs.scan().await.expect("host-side scan should succeed");
        assert!(models.is_empty(), "empty dirs yield no models");

        // No worker was ever spawned: status reports worker_alive=false.
        let st = higgs.status().await.expect("status");
        assert!(!st.worker_alive, "scan must not spawn a worker");
    }

    // ── Test 3: load then status maps ─────────────────────────────────────────

    #[tokio::test]
    async fn load_then_status_maps() {
        use tokio::io::{AsyncBufReadExt, BufReader};

        let (sup, mut test_write, test_read) = make_supervisor();
        // `load` resolves the GGUF path host-side, so point config at a fixture.
        let dir = tempfile::TempDir::new().unwrap();
        crate::serve::test_support::write_gguf_fixture(dir.path(), "org/model");
        let cfg = HiggsConfig {
            lmstudio_dirs: vec![dir.path().to_path_buf()],
            hf_dirs: vec![],
            ollama_dirs: vec![],
            default_load: HiggsConfig::default().default_load,
        };
        let higgs = Higgs {
            sup: Arc::new(sup),
            config: parking_lot::Mutex::new(cfg),
            lifecycle: tokio::sync::Mutex::new(()),
            inference_gate: std::sync::Arc::new(tokio::sync::Semaphore::new(
                crate::api::MAX_CONCURRENT_INFERENCE,
            )),
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
        };
        let mut events_rx = higgs.events();
        // `load`/`status` run a host-side scan (on a blocking thread) before each
        // RPC, so drive the operation future concurrently with the responder: the
        // responder reads the request line (proving the id is pending) and only
        // then writes the reply. A fixed pre-sleep + sequential write would race
        // the scan and drop the response.
        let mut lines = BufReader::new(test_read).lines();

        // Issue load — mock responds with ok.
        let load_fut = higgs.load("org/model", None);
        let (load_res, _) = tokio::join!(load_fut, async {
            lines.next_line().await.unwrap().expect("M_LOAD request");
            write_response(&mut test_write, 1, json!({"id": "org/model"})).await;
        });
        load_res.expect("load should succeed");

        // ModelLoaded event must arrive.
        let ev = tokio::time::timeout(std::time::Duration::from_millis(100), events_rx.recv())
            .await
            .expect("timeout")
            .expect("recv");
        assert!(matches!(ev, HiggsEvent::ModelLoaded { id } if id == "org/model"));

        // Issue status — mock responds with loaded info.
        let status_fut = higgs.status();
        let (st, _) = tokio::join!(status_fut, async {
            lines.next_line().await.unwrap().expect("M_STATUS request");
            write_response(
                &mut test_write,
                2,
                json!({
                    "loaded": { "id": "org/model", "ctx_len": 4096, "gpu_layers": 4294967295u64, "threads": 4 },
                    "models_scanned": 3,
                }),
            )
            .await;
        });
        let st = st.expect("status should succeed");
        assert!(st.worker_alive);
        // models_on_disk now comes from a host-side scan of the config dirs
        // (one GGUF fixture), not the worker's `models_scanned`.
        assert_eq!(st.models_on_disk, 1);
        let li = st.loaded.expect("loaded should be Some");
        assert_eq!(li.id, "org/model");
        assert_eq!(li.ctx_len, 4096);
        assert_eq!(li.gpu_layers, u32::MAX);
    }

    // ── Test 3b: status loaded info includes model metadata ──────────────────

    #[tokio::test]
    async fn status_loaded_info_includes_model_metadata() {
        use tokio::io::{AsyncBufReadExt, BufReader};

        let (sup, mut test_write, test_read) = make_supervisor();
        // Metadata now comes from the HOST scan, not the worker response: point
        // config at a GGUF fixture (arch=llama, ctx_train=4096, chat template)
        // so the host-scanned `HiggsModel` enriches the worker-reported `loaded`.
        let dir = tempfile::TempDir::new().unwrap();
        crate::serve::test_support::write_gguf_fixture(dir.path(), "org/model");
        let cfg = HiggsConfig {
            lmstudio_dirs: vec![dir.path().to_path_buf()],
            hf_dirs: vec![],
            ollama_dirs: vec![],
            default_load: HiggsConfig::default().default_load,
        };
        let higgs = Higgs {
            sup: Arc::new(sup),
            config: parking_lot::Mutex::new(cfg),
            lifecycle: tokio::sync::Mutex::new(()),
            inference_gate: std::sync::Arc::new(tokio::sync::Semaphore::new(
                crate::api::MAX_CONCURRENT_INFERENCE,
            )),
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
        };

        // `status` runs a host-side scan (on a blocking thread) before M_STATUS,
        // so drive the future concurrently with a responder that reads the
        // request line before replying — a fixed sleep would race the scan. The
        // worker reports only id/ctx_len/gpu_layers/threads; the metadata fields
        // are filled host-side from the fixture.
        let mut lines = BufReader::new(test_read).lines();
        let status_fut = higgs.status();
        let (st, _) = tokio::join!(status_fut, async {
            lines.next_line().await.unwrap().expect("M_STATUS request");
            write_response(
                &mut test_write,
                1,
                json!({
                    "loaded": {
                        "id": "org/model",
                        "ctx_len": 4096,
                        "gpu_layers": 99,
                        "threads": 4,
                    },
                    "models_scanned": 1,
                }),
            )
            .await;
        });
        let st = st.expect("status should succeed");
        let li = st.loaded.expect("loaded should be Some");
        assert_eq!(li.id, "org/model");
        assert_eq!(li.arch.as_deref(), Some("llama"));
        assert_eq!(li.quant.as_deref(), Some("Q4_K_M"));
        assert_eq!(li.max_context_length, Some(4096));
        assert!(li.size_bytes.is_some(), "size_bytes from fixture file");
        assert_eq!(li.has_chat_template, Some(true));
    }

    // ── Test 3c: host-resolved load carries the GGUF path (no worker scan) ────
    //
    // Regression: after scan moved host-side, the worker's ModelStore is empty,
    // so the worker can only resolve a path if the host puts it in M_LOAD params.
    // This asserts `load(id)` resolves the path host-side and includes it in the
    // M_LOAD request — proving the load works WITHOUT a prior worker scan. If the
    // path-passing were removed (worker fell back to its empty `store.get(id)`),
    // the params would carry no `path` and this test would fail.
    #[tokio::test]
    async fn load_carries_host_resolved_path() {
        use tokio::io::{AsyncBufReadExt, BufReader};

        let (sup, mut test_write, test_read) = make_supervisor();

        // Real GGUF fixture so the host-side scan discovers the id with a path.
        let dir = tempfile::TempDir::new().unwrap();
        crate::serve::test_support::write_gguf_fixture(dir.path(), "org/model");
        let cfg = HiggsConfig {
            lmstudio_dirs: vec![dir.path().to_path_buf()],
            hf_dirs: vec![],
            ollama_dirs: vec![],
            default_load: HiggsConfig::default().default_load,
        };
        let higgs = Higgs {
            sup: Arc::new(sup),
            config: parking_lot::Mutex::new(cfg),
            lifecycle: tokio::sync::Mutex::new(()),
            inference_gate: std::sync::Arc::new(tokio::sync::Semaphore::new(
                crate::api::MAX_CONCURRENT_INFERENCE,
            )),
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
        };

        // Drive the load. `load` first runs a host-side scan (on a blocking
        // thread) before sending M_LOAD, so drive the load future concurrently
        // with a responder that reads the request line (proving id=1 is pending)
        // before replying. A fixed pre-sleep would race the scan and drop the
        // response.
        let mut lines = BufReader::new(test_read).lines();
        let load_fut = higgs.load("org/model", None);
        let (load_res, line) = tokio::join!(load_fut, async {
            let line = lines.next_line().await.unwrap().expect("M_LOAD request");
            write_response(&mut test_write, 1, json!({"id": "org/model"})).await;
            line
        });
        load_res.expect("host-resolved load should succeed");

        // The M_LOAD request carries the fixture path resolved host-side.
        let v: serde_json::Value = serde_json::from_str(&line).unwrap();
        assert_eq!(v["method"], M_LOAD);
        let path = v["params"]["path"].as_str().expect("path in M_LOAD params");
        assert!(path.ends_with(".gguf"), "path was: {path}");
        assert!(path.contains("org/model"), "path was: {path}");
    }

    // ── Test 4: chat_stream delivers chunks and outcome ────────────────────────
    //
    // Verifies end-to-end: alloc_request_id allocates id=1; chat_stream registers
    // the sink under that id and sends M_CHAT with request_id=1; the test injects
    // N_CHAT_CHUNK notifications tagged request_id=1; route_notification delivers
    // them to rx; the final response for RPC id=1 resolves the outcome handle.

    #[tokio::test]
    async fn chat_stream_delivers() {
        let (sup, mut test_write, _test_read) = make_supervisor();
        let higgs = Higgs {
            sup: Arc::new(sup),
            config: parking_lot::Mutex::new(HiggsConfig::default()),
            lifecycle: tokio::sync::Mutex::new(()),
            inference_gate: std::sync::Arc::new(tokio::sync::Semaphore::new(
                crate::api::MAX_CONCURRENT_INFERENCE,
            )),
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
        };

        let (mut rx, handle) = higgs
            .chat_stream(
                "org/model".to_owned(),
                r#"[{"role":"user","content":"hi"}]"#.to_owned(),
                256,
                0.7,
                None,
            )
            .await
            .expect("chat_stream should succeed");

        // Inject chunk notifications tagged with request_id=1 (the first allocated id).
        use crate::rpc::{encode, RpcFrame, RpcNotification};
        for delta in &["hel", "lo"] {
            let notif = encode(&RpcFrame::Notification(RpcNotification {
                jsonrpc: "2.0".into(),
                method: N_CHAT_CHUNK.into(),
                params: json!({ "request_id": 1u64, "delta": delta }),
            }));
            test_write
                .write_all(format!("{notif}\n").as_bytes())
                .await
                .unwrap();
        }
        test_write.flush().await.unwrap();

        tokio::time::sleep(std::time::Duration::from_millis(30)).await;

        // Final response for M_CHAT (RPC id=1) — includes token counts.
        write_response(
            &mut test_write,
            1,
            json!({"content": "hello", "finish_reason": "stop", "prompt_tokens": 10, "completion_tokens": 3}),
        )
        .await;

        let outcome = tokio::time::timeout(std::time::Duration::from_millis(500), handle)
            .await
            .expect("join timeout")
            .expect("join error")
            .expect("chat outcome error");

        assert_eq!(outcome.content, "hello");
        assert_eq!(outcome.finish_reason, "stop");
        assert_eq!(outcome.prompt_tokens, 10);
        assert_eq!(outcome.completion_tokens, 3);

        // Chunks must have arrived.
        let chunk1 = rx.try_recv().expect("chunk 1");
        let chunk2 = rx.try_recv().expect("chunk 2");
        assert_eq!(chunk1, "hel");
        assert_eq!(chunk2, "lo");
    }

    // ── Test 5: chat_stream against dead worker removes sink ─────────────────

    /// When the chat request fails (write_tx is None — worker not running), the
    /// spawned task removes the sink on the error path so the map stays clean.
    #[tokio::test]
    async fn chat_stream_dead_worker_removes_sink() {
        // Build a Supervisor with no worker halves — factory always fails.
        let sup = crate::supervisor::Supervisor::with_factory(Box::new(|_ring, _model| {
            Err(HiggsError::WorkerSpawnFailed {
                source: std::io::Error::other("mock: no worker"),
            })
        }));
        // Do NOT call start() — write_tx stays None (dead worker).

        let higgs = Higgs {
            sup: Arc::new(sup),
            config: parking_lot::Mutex::new(HiggsConfig::default()),
            lifecycle: tokio::sync::Mutex::new(()),
            inference_gate: std::sync::Arc::new(tokio::sync::Semaphore::new(
                crate::api::MAX_CONCURRENT_INFERENCE,
            )),
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
        };

        // chat_stream registers the sink then the spawned task encounters dead worker.
        let (_rx, handle) = higgs
            .chat_stream(
                "org/model".to_owned(),
                r#"[{"role":"user","content":"hi"}]"#.to_owned(),
                8,
                0.0,
                None,
            )
            .await
            .expect("chat_stream itself should not fail");

        // The spawned task must return an Err (worker dead).
        let result = tokio::time::timeout(std::time::Duration::from_millis(200), handle)
            .await
            .expect("join timeout")
            .expect("join error");
        assert!(result.is_err(), "chat against dead worker must fail");

        // After the failed request, the sink map must be empty (remove_chat_sink was called).
        assert_eq!(
            higgs.sup.chat_sinks_count(),
            0,
            "chat_sinks must be empty after failed request"
        );
    }
}
