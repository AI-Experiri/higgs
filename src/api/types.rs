//! Wire/response types, constants, and the chat-outcome decoder for the `Higgs` facade.
//! Split out of `api.rs` (see `api/DESIGN.md`); `api.rs` re-exports these so existing
//! `crate::api::*` paths are unchanged.

use std::path::PathBuf;

use crate::worker::engine::{GpuLayers, LoadParams};

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
            default_load: LoadParams::base(4096, GpuLayers::All, threads),
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
        /// Effective idle seconds after which an idle worker is auto-unloaded — the
        /// live value the node reaper enforces (default
        /// [`DEFAULT_IDLE_TTL`](crate::node::runtime::DEFAULT_IDLE_TTL), 60 min;
        /// runtime-mutable via `/api/higgs/settings`). Always equals the settings
        /// endpoint's `idle_ttl_minutes × 60`.
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
        /// The node-local worker id serving this model. Distinguishes the entries of
        /// [`HiggsStatus::loaded_all`] (one worker per resident model) so the UI can key a
        /// per-worker card + its per-worker log pane. Always present.
        #[ts(type = "number")]
        pub worker_id: u32,
        /// Context window size in tokens.
        #[ts(type = "number")]
        pub ctx_len: u32,
        /// GPU layers offloaded (`GpuLayers::All` = every layer; `Count { n }` = explicit).
        pub gpu_layers: GpuLayers,
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
        /// Per-load idle-TTL override in minutes. RESERVED: per-load idle-TTL
        /// enforcement is a deferred follow-up (the node reaper applies one per-node
        /// TTL to every worker), so this is currently ALWAYS absent — every loaded
        /// model uses the global idle TTL (`/api/higgs/settings`). It becomes
        /// populated only once the reaper honors per-worker overrides.
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
        /// Info about the PRIMARY loaded model (lowest worker id), if any. Kept for the status
        /// bar + provider seeding; `loaded_all` carries every resident model.
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        pub loaded: Option<LoadedInfo>,
        /// EVERY resident local model — one entry per worker (the local node is multi-model:
        /// additive loads, one worker per model), sorted by worker id, `loaded` first. Always an
        /// array (empty when nothing is loaded). The UI's "Loaded Models" section renders one card
        /// per entry. `#[serde(default)]` keeps deserialization tolerant; clients still default it
        /// defensively (`status.loaded_all ?? []`) as cheap insurance.
        #[serde(default)]
        pub loaded_all: Vec<LoadedInfo>,
        /// Number of models discovered in the last scan.
        #[ts(type = "number")]
        pub models_on_disk: u32,
    }
}

/// Build a [`ChatOutcome`] from a worker `M_CHAT` result `Value` (same shape whether the
/// worker is local or relayed from a remote node).
pub(crate) fn chat_outcome_from_value(result: &serde_json::Value) -> ChatOutcome {
    ChatOutcome {
        content: result
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_owned(),
        finish_reason: result
            .get("finish_reason")
            .and_then(|v| v.as_str())
            .unwrap_or("stop")
            .to_owned(),
        tool_calls: result.get("tool_calls").filter(|v| !v.is_null()).cloned(),
        prompt_tokens: result
            .get("prompt_tokens")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0) as u32,
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

/// Upper bound on the runtime-settable idle auto-unload TTL, in minutes. `PUT
/// /api/higgs/settings` takes an unbounded `u64` straight from the request body, so
/// [`Higgs::set_idle_ttl_minutes`] clamps to this before the `× 60` seconds conversion —
/// otherwise a huge value would overflow `minutes * 60` (a debug-build panic, a release
/// wrap). 100 years is "effectively never unload" (the canonical never-unload control is
/// the `auto_unload_idle` toggle), so the clamp restricts no real use while keeping every
/// `× 60` conversion (the reaper TTL, `/api/higgs/system`) overflow-safe and consistent.
pub const MAX_IDLE_TTL_MINUTES: u64 = 100 * 365 * 24 * 60;

// NOTE: the idle auto-unload TTL + reaper cadence live with the node reaper now
// (`node::runtime`: `DEFAULT_IDLE_TTL` = 60 min, `reap_interval()` derives the poll
// cadence from the live TTL). The old engine-era `IDLE_UNLOAD_TTL` (5 min) /
// `IDLE_UNLOAD_TTL_MINUTES` / `IDLE_REAP_INTERVAL` constants were removed: they were
// dead after the reaper moved into the node and their stale 5-minute value contradicted
// the real 60-minute default. The live TTL is read from `Higgs::idle_ttl_minutes`.
