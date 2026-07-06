//! Request/response wire structs for higgs's `/api/higgs/*` control surface.
//!
//! Each type is ts-rs exported to `frontend/src/lib/generated/higgs/` and
//! re-exported from `frontend/src/lib/types.ts`. The `/v1` surface uses
//! `async-openai` wire types verbatim, so only the control shapes live here.

use crate::worker::engine::{FlashAttn, KvCacheKind, LoadParams};
use crate::worker::models::HiggsModel;

higgs_ts! {
    /// Confirmation body for mutating control routes; serializes as
    /// `{"status":"ok"}`. Standalone equivalent of the gateway's `StatusOk`
    /// (higgs imports nothing from jigglebot); responses with extra fields
    /// compose it via `#[serde(flatten)]`.
    #[derive(Debug, Clone, serde::Serialize)]
    pub struct HiggsOk {
        /// Literal `"ok"`.
        pub status: String,
    }
}

impl HiggsOk {
    /// Build the canonical `{"status":"ok"}` body.
    pub fn new() -> Self {
        Self {
            status: "ok".into(),
        }
    }
}

impl Default for HiggsOk {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[path = "wire_tests.rs"]
mod tests;

higgs_ts! {
    /// Response for `GET /api/higgs/hub`: whether this server is a fleet hub right now.
    ///
    /// `enabled` is false when hub mode is off (no hub installed) — the Fleet tab shows a
    /// "hub mode off" panel instead of inferring it from a `/pair` 409. When enabled, `hub_id`
    /// is the hub's stable iroh endpoint id and `node_count` is how many nodes it has admitted
    /// (connected or not).
    #[derive(Debug, Clone, serde::Serialize)]
    pub struct HiggsHubStatus {
        /// True when this server runs as a fleet hub (accepting node dials).
        pub enabled: bool,
        /// The hub's stable id (iroh endpoint id); `None` when hub mode is off.
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        pub hub_id: Option<String>,
        /// Nodes admitted to the fleet (connected or not); 0 when hub mode is off.
        pub node_count: u32,
    }
}

higgs_ts! {
    /// One load-relevant GGUF header key/value, curated for the UI so a support
    /// mismatch can be pinned to a specific field (e.g. `general.architecture =
    /// gemma4`). Only keys that bear on loadability are surfaced — giant arrays
    /// (token lists, merges) are deliberately skipped. `value` is the
    /// human-readable rendering of the header value.
    #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
    pub struct GgufComponent {
        /// GGUF metadata key, e.g. `general.architecture` or `llama.block_count`.
        pub key: String,
        /// The value as a display string.
        pub value: String,
    }
}

higgs_ts! {
    /// Response for `GET /api/higgs/models`: live scan results plus the loaded id.
    #[derive(Debug, serde::Serialize)]
    pub struct HiggsModelsResponse {
        /// Models discovered by a live scan of the configured directories.
        pub models: Vec<HiggsModelEntry>,
        /// Id of the currently loaded model, if any — matches `HiggsModel::id`.
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        pub loaded_id: Option<String>,
    }
}

higgs_ts! {
    /// The analytical resource fit for a model's saved profile against the node's
    /// CURRENT free resources — the numbers behind the `Servable`/`Unservable`
    /// badge, so the UI can say "≈22 GB needed, 18 GB free" instead of a bare
    /// verdict. Present only when a profile was actually evaluated for fit
    /// (i.e. the model is `servable` or `unservable`); `None` otherwise.
    ///
    /// ADVISORY: `free_*` come from the cached device snapshot (a live refresh
    /// would spawn a sysinfo worker per poll), so render the figures as "≈".
    #[derive(Debug, Clone, serde::Serialize)]
    pub struct ModelFit {
        /// Estimated VRAM the profile needs to load + serve.
        #[ts(type = "number")]
        pub needed_vram_bytes: u64,
        /// Estimated system RAM the profile needs.
        #[ts(type = "number")]
        pub needed_ram_bytes: u64,
        /// GPU free VRAM at the last device snapshot.
        #[ts(type = "number")]
        pub free_vram_bytes: u64,
        /// System free RAM at the last device snapshot.
        #[ts(type = "number")]
        pub free_ram_bytes: u64,
    }
}

higgs_ts! {
    /// Per-model entry in `GET /api/higgs/models`: enriches [`HiggsModel`] with
    /// request-derived fields computed by the control handler.
    #[derive(Debug, serde::Serialize)]
    pub struct HiggsModelEntry {
        /// All canonical model fields (id, path, size_bytes, quant, source, arch, ctx_train, has_chat_template).
        #[serde(flatten)]
        pub model: HiggsModel,
        /// Load state: `"loaded"` if this model is currently resident, `"not-loaded"` otherwise.
        pub state: String,
        /// File format — always `"gguf"` for higgs-discovered models.
        pub format: String,
        /// Gate 2: whether higgs has a tool-call parser that matches this model's
        /// chat template. `false` means the model can be served but tool calls
        /// can't be parsed; `support_reason` explains. (There is no scan-time
        /// load probe — engine loadability is learned only at actual load.)
        pub tool_calls: bool,
        /// The fixed Gate-2 message when `!tool_calls`
        /// (`"no tool-call parser matches this model's template"`), else `None`.
        /// There is no scan-time load probe, so this never carries an engine load
        /// error — a failed load is reported by the load endpoint itself.
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        ///
        /// The curated `gguf_components` list — used by the UI to pin a support
        /// mismatch to a specific header field — rides the flattened
        /// [`HiggsModel`] (its single home); it is not re-declared here.
        pub support_reason: Option<String>,
        /// The params this model was last successfully loaded with on THIS instance,
        /// persisted in `config.json` (`None` if it has never been loaded here).
        /// Lets the UI show "last loaded with ctx=…, gpu_layers=…" for any scanned
        /// model — replacing the removed scan-time load probe with real load history.
        /// A `ctx_len` of `0` means AUTO (the engine picks the model's trained
        /// context, capped) — the UI should render that as "auto", not "0".
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        pub last_load: Option<LoadParams>,
        /// Readiness state for this model on THIS node — the contract the UI badges
        /// and (future) autonomous agents read. Derived from profile presence,
        /// staleness, residency, live resource fit, and the serving toggle.
        pub readiness: crate::serve::readiness::ModelReadiness,
        /// The resource fit numbers behind a `servable`/`unservable` readiness —
        /// `Some` only when a profile was evaluated for fit, so the UI can show
        /// the needed-vs-free gap. `None` for the other states.
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        pub fit: Option<ModelFit>,
        /// The last ANALYTICAL tune's load params (the engine-tagged umbrella).
        /// Together with `benched_load` these are the two selectable saved
        /// param sets both load surfaces offer; `tune_provenance` says which
        /// one is the ACTIVE profile (the JIT/readiness default). `None` when
        /// no analytical tune has run.
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        pub tuned_load: Option<LoadParams>,
        /// The last TURBOTUNE (measured benchmark) config's load params — the
        /// "Benchmarked" selectable set; `bench_tps` is its measured decode
        /// throughput. `None` when the model was never benchmarked.
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        pub benched_load: Option<LoadParams>,
        /// How the ACTIVE ("latest") saved tune profile — the JIT/readiness
        /// default — was produced: `Heuristic` (analytical), `Card` (model-card
        /// sampling), or `Bench` (turbotune-measured). Informational about the
        /// active record's origin; NOT a guaranteed selector into `tuned_load` /
        /// `benched_load`, because a bare load with edited params demotes the
        /// active record to a `Heuristic` distinct from both saved sets (its
        /// params ride `last_load`). `None` when the model has no tune record yet.
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        pub tune_provenance: Option<crate::tune::TuneProvenance>,
        /// Measured decode throughput (tokens/sec) of the `benched_load` set —
        /// the winning turbotune candidate's speed, read from the same saved
        /// benchmark record as `benched_load` (NOT from the active profile, so a
        /// later bare load editing the active params does not disturb it).
        /// Present whenever that record carries a measured throughput; a fresh
        /// turbotune replaces it and `benched_load` together. `None` when the
        /// model was never benchmarked.
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        pub bench_tps: Option<f32>,
    }
}

higgs_ts! {
    /// Request body for `POST /api/higgs/models/load`.
    ///
    /// Absent load parameters fall back to the host-configured defaults.
    #[derive(Debug, serde::Deserialize)]
    pub struct HiggsLoadRequest {
        /// HuggingFace repo id of the model to load.
        pub id: String,
        /// Context window ([`CtxLen::Auto`] = trained context; `Fixed { n }` = pinned).
        #[ts(optional)]
        pub ctx_len: Option<crate::worker::engine::CtxLen>,
        /// GPU layers to offload (`GpuLayers::All` = every layer; `Count { n }` = explicit).
        #[ts(optional)]
        pub gpu_layers: Option<crate::worker::engine::GpuLayers>,
        /// Worker threads used during generation.
        #[ts(type = "number")]
        #[ts(optional)]
        pub threads: Option<u32>,
        /// Memory-map the GGUF instead of reading it into RAM.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        pub use_mmap: Option<bool>,
        /// Lock model pages in RAM (prevent swap).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        pub use_mlock: Option<bool>,
        /// Logical batch size for prompt decode.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        #[ts(type = "number")]
        #[ts(optional)]
        pub n_batch: Option<u32>,
        /// Physical (micro) batch size.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        #[ts(type = "number")]
        #[ts(optional)]
        pub n_ubatch: Option<u32>,
        /// Offload the KV cache & KQV ops to the GPU.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        pub offload_kqv: Option<bool>,
        /// RoPE base frequency override.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        #[ts(type = "number")]
        #[ts(optional)]
        pub rope_freq_base: Option<f32>,
        /// RoPE frequency scale (context extension).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        #[ts(type = "number")]
        #[ts(optional)]
        pub rope_freq_scale: Option<f32>,
        /// Flash-attention policy.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        pub flash_attn: Option<FlashAttn>,
        /// KV cache key data type.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        pub type_k: Option<KvCacheKind>,
        /// KV cache value data type.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        pub type_v: Option<KvCacheKind>,
        /// Sampler RNG seed for reproducible generation.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        #[ts(type = "number")]
        #[ts(optional)]
        pub seed: Option<u32>,
        /// Per-load idle-TTL override in minutes. RESERVED / forward-compat: per-load
        /// idle-TTL enforcement is a deferred follow-up, so this is currently ACCEPTED
        /// but NOT enforced — the node reaper applies one per-node TTL to every worker
        /// (the global TTL, `/api/higgs/settings`), and the host neither stores nor
        /// surfaces this value. It will take effect once the reaper honors per-worker
        /// overrides (host-side only either way — never forwarded to the worker/engine).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        #[ts(type = "number")]
        #[ts(optional)]
        pub idle_ttl_minutes: Option<u64>,
        /// The FULL engine load params, as the engine-tagged [`LoadParams`] umbrella
        /// (`{"engine":"LlamaCpp", …}`) — an accepted autotune suggestion or a manual
        /// edit of the complete §4/§5 surface. When present it SUPERSEDES the flat
        /// fields above (kept for back-compat); its base fields are used as-is. This
        /// is what the frontend sends after `[Tune params]` fills the pane — and the
        /// engine tag means a future MLX engine sends `{"engine":"Mlx", …}` here
        /// without a new wire field.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        pub params: Option<LoadParams>,
    }
}

higgs_ts! {
    /// Response for `POST /api/higgs/models/load`: `{"status":"ok","id":…}`.
    #[derive(Debug, serde::Serialize)]
    pub struct HiggsLoadResponse {
        /// Confirmation status; always `{"status":"ok"}` on success.
        #[serde(flatten)]
        pub status: HiggsOk,
        /// Id of the model that was loaded.
        pub id: String,
    }
}

higgs_ts! {
    /// Response for `GET /api/higgs/logs`: `{"lines":[…]}`.
    #[derive(Debug, serde::Serialize)]
    pub struct HiggsLogsResponse {
        /// Worker stderr tail, oldest first.
        pub lines: Vec<String>,
    }
}

higgs_ts! {
    /// Body for `GET`/`PUT /api/higgs/logs/settings`: the runtime Developer-Log
    /// toggles. `GET` returns the current state of both; `PUT` carries both and
    /// sets both. The log settings higgs actually backs.
    #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
    pub struct LogSettings {
        /// Whether the serve-layer verbose serving line is enabled — when `true`,
        /// the chat path emits an extra `higgs: served …` completion line per
        /// request into the Developer Logs.
        pub verbose: bool,
        /// Whether the serve-layer incoming-prompt line is enabled — when `true`,
        /// the chat path emits a `higgs: incoming …` line carrying the (capped)
        /// prompt CONTENT per request. This is the explicit opt-in that overrides
        /// the redact-by-default policy; default `false`.
        pub log_incoming_tokens: bool,
        /// DEBUG: when `true`, the Developer-Log layer also emits non-message
        /// structured fields — INCLUDING prompt CONTENT — un-redacting the logs
        /// for debugging. Off by default. `#[serde(default)]` so older PUT bodies
        /// that omit it deserialize as `false`.
        #[serde(default)]
        pub show_log_fields: bool,
    }
}

higgs_ts! {
    /// Body for `GET`/`PUT /api/higgs/settings`: the runtime server-behavior
    /// flags higgs actually backs. `GET` returns the current state; `PUT` carries
    /// it and sets it. Distinct from [`LogSettings`] (Developer-Log toggles) —
    /// this is the server-behavior namespace, designed to grow as more runtime
    /// flags (e.g. a server on/off) are added.
    #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
    pub struct HiggsRuntimeSettings {
        /// Whether just-in-time model loading is enabled. When `true` (the
        /// default), a chat request for a scanned-but-unloaded model loads it on
        /// demand — swapping out any resident model (higgs serves one at a time) —
        /// instead of returning a 404. When `false`, an unloaded model is a 404
        /// (explicit-load only).
        pub jit_enabled: bool,
        /// Whether the idle reaper auto-unloads the loaded model after the idle
        /// TTL (default `true`). When `false`, a loaded model stays resident
        /// until an explicit unload regardless of idle time. Read by the reaper
        /// each tick, so a change takes effect without a restart.
        pub auto_unload_idle: bool,
        /// Idle minutes after which the loaded model is auto-unloaded (default
        /// 5, ollama `keep_alive`). Read by the reaper each tick. Only in effect
        /// when [`auto_unload_idle`](Self::auto_unload_idle) is `true`.
        #[ts(type = "number")]
        pub idle_ttl_minutes: u64,
        /// Whether the `/v1` inference surface is serving (default `true`). When
        /// `false`, the `/v1` inference endpoints return `[HG019]` → 503 while the
        /// `/api/higgs/*` control surface stays reachable so the server can be
        /// re-enabled. Read by the chat boundary on each request, so a change
        /// takes effect without a restart.
        pub serving_enabled: bool,
    }
}

higgs_ts! {
    /// Error body for control routes: the rendered `HiggsError` display
    /// (diagnostic code included), as `{"error":"<display>"}`.
    #[derive(Debug, serde::Serialize)]
    pub struct HiggsErrorResponse {
        /// Human-readable failure, e.g. `[HG003] model not loaded: …`.
        pub error: String,
    }
}

higgs_ts! {
    /// Response for `GET /api/higgs/version`.
    #[derive(Debug, serde::Serialize)]
    pub struct HiggsVersionResponse {
        /// Higgs crate version from Cargo.toml (`CARGO_PKG_VERSION`).
        pub higgs: String,
        /// Human-readable engine name.
        pub engine: String,
        /// Engine version reported at runtime by `ggml_version()` (e.g. `"0.9.7"`) —
        /// the actual vendored ggml/llama.cpp engine version.
        pub engine_version: String,
        /// `llama-cpp-2` Rust binding crate version (e.g. `"0.1.139"`).
        pub binding: String,
        /// File formats this runtime supports.
        pub supported_formats: Vec<String>,
    }
}

higgs_ts! {
    /// One configured API key as the management surface lists it: label,
    /// scopes, and a short digest prefix as its display identifier. NEVER the
    /// plaintext token (shown once at mint) and never the full digest.
    #[derive(Debug, Clone, serde::Serialize)]
    pub struct HiggsKeyEntry {
        pub label: String,
        pub scopes: Vec<crate::keys::Scope>,
        /// First 12 hex chars of the stored SHA-256 digest.
        pub sha256_prefix: String,
        /// Unix-ms the key was minted; `None` for keys from a pre-timestamp
        /// store (render as unknown).
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(optional, type = "number")]
        pub created_at_ms: Option<u64>,
        /// Unix-ms of the last successful authorization, `None` if never used.
        /// Served from the LIVE store (lags at most the ~1-min touch throttle).
        /// Usage is best-effort history: it reaches disk only when a later
        /// mint/revoke persists the store, so a restart shows the stamps as of
        /// that last mutation ("never" if none since).
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(optional, type = "number")]
        pub last_used_ms: Option<u64>,
    }
}

higgs_ts! {
    /// `GET /api/higgs/keys` — the configured keys plus whether auth is
    /// currently gating the surface (false ⇔ zero keys ⇔ open).
    #[derive(Debug, Clone, serde::Serialize)]
    pub struct HiggsKeysList {
        pub auth_enabled: bool,
        pub keys: Vec<HiggsKeyEntry>,
    }
}

higgs_ts! {
    /// `POST /api/higgs/keys` request: mint a key. Omitted `scopes` defaults to
    /// `[chat, models]` (the CLI's default) — pass `["admin"]` explicitly for a
    /// management key.
    #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
    pub struct HiggsMintKeyRequest {
        pub label: String,
        #[ts(optional)]
        pub scopes: Option<Vec<crate::keys::Scope>>,
    }
}

higgs_ts! {
    /// `POST /api/higgs/keys` response. `token` is the plaintext — shown THIS
    /// ONCE, never persisted, never logged; the caller must store it now.
    #[derive(Debug, Clone, serde::Serialize)]
    pub struct HiggsMintKeyResponse {
        pub label: String,
        pub scopes: Vec<crate::keys::Scope>,
        pub token: String,
    }
}

higgs_ts! {
    /// `DELETE /api/higgs/keys/{label}` response: how many keys the label matched.
    /// Revoking the last key turns auth OFF on a LOOPBACK bind only; a
    /// LAN-exposed server REFUSES last-key revocation outright ([HG059], 409) —
    /// the runtime counterpart of the [HG058] startup guarantee.
    #[derive(Debug, Clone, serde::Serialize)]
    pub struct HiggsKeyRemoved {
        #[ts(type = "number")]
        pub removed: u64,
        pub auth_enabled: bool,
    }
}
