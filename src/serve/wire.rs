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
