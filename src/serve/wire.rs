//! Request/response wire structs for higgs's `/api/higgs/*` control surface.
//!
//! Each type is ts-rs exported to `frontend/src/lib/generated/higgs/` and
//! re-exported from `frontend/src/lib/types.ts`. The `/v1` surface uses
//! `async-openai` wire types verbatim, so only the control shapes live here.

use crate::worker::engine::{FlashAttn, KvCacheKind};
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
mod tests {
    use super::HiggsOk;

    /// `HiggsOk::default()` (the `Default` impl) yields the same `{"status":"ok"}`
    /// body as `new()`, and both serialize to the canonical wire shape.
    #[test]
    fn higgs_ok_default_matches_new_and_serializes() {
        let from_default = HiggsOk::default();
        let from_new = HiggsOk::new();
        assert_eq!(from_default.status, "ok");
        assert_eq!(from_default.status, from_new.status);
        assert_eq!(
            serde_json::to_value(&from_default).unwrap(),
            serde_json::json!({ "status": "ok" }),
        );
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
        /// Gate 1: whether our llama.cpp engine can LOAD this model, as proved by
        /// a real probe of a representative GGUF for this `(arch, quant)`. `false`
        /// means a load attempt failed; `support_reason` carries the verbatim
        /// engine reason.
        pub loadable: bool,
        /// Gate 2: whether higgs has a tool-call parser that matches this model's
        /// chat template. `false` means the model loads but tool calls can't be
        /// parsed; `support_reason` explains.
        pub tool_calls: bool,
        /// The exact reason this model isn't fully supported, or `None` when it is.
        /// When `!loadable`, this is the engine's VERBATIM load error (e.g.
        /// `"unknown model architecture: 'gemma4'"`). When `loadable && !tool_calls`,
        /// it is `"no tool-call parser matches this model's template"`.
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        ///
        /// The curated `gguf_components` list — used by the UI to pin a support
        /// mismatch to a specific header field — rides the flattened
        /// [`HiggsModel`] (its single home); it is not re-declared here.
        pub support_reason: Option<String>,
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
        /// Context window size in tokens.
        #[ts(type = "number")]
        #[ts(optional)]
        pub ctx_len: Option<u32>,
        /// GPU layers to offload; u32::MAX means all.
        #[ts(type = "number")]
        #[ts(optional)]
        pub gpu_layers: Option<u32>,
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
        /// Per-load idle-TTL override in minutes. When set, the idle reaper uses
        /// this instead of the global TTL for THIS loaded model. Absent = use
        /// global. HOST-SIDE only — never forwarded to the worker/engine.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        #[ts(type = "number")]
        #[ts(optional)]
        pub idle_ttl_minutes: Option<u64>,
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
