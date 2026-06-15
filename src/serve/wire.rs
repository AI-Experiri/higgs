//! Request/response wire structs for higgs's `/api/higgs/*` control surface.
//!
//! Each type is ts-rs exported to `frontend/src/lib/generated/` and re-exported
//! from `frontend/src/lib/types.ts`. The `/v1` surface uses `async-openai` wire
//! types verbatim, so only the control shapes live here.

use crate::worker::models::HiggsModel;

/// Confirmation body for mutating control routes; serializes as
/// `{"status":"ok"}`. Standalone equivalent of the gateway's `StatusOk`
/// (higgs imports nothing from jigglebot); responses with extra fields
/// compose it via `#[serde(flatten)]`.
#[derive(Debug, Clone, serde::Serialize, ts_rs::TS)]
#[ts(export, export_to = "../../../frontend/src/lib/generated/")]
pub struct HiggsOk {
    /// Literal `"ok"`.
    pub status: String,
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

/// Response for `GET /api/higgs/models`: live scan results plus the loaded id.
#[derive(Debug, serde::Serialize, ts_rs::TS)]
#[ts(export, export_to = "../../../frontend/src/lib/generated/")]
pub struct HiggsModelsResponse {
    /// Models discovered by a live scan of the configured directories.
    pub models: Vec<HiggsModelEntry>,
    /// Id of the currently loaded model, if any — matches `HiggsModel::id`.
    #[ts(optional)]
    pub loaded_id: Option<String>,
}

/// Per-model entry in `GET /api/higgs/models`: enriches [`HiggsModel`] with
/// request-derived fields computed by the control handler.
#[derive(Debug, serde::Serialize, ts_rs::TS)]
#[ts(export, export_to = "../../../frontend/src/lib/generated/")]
pub struct HiggsModelEntry {
    /// All canonical model fields (id, path, size_bytes, quant, source, arch, ctx_train, has_chat_template).
    #[serde(flatten)]
    pub model: HiggsModel,
    /// Load state: `"loaded"` if this model is currently resident, `"not-loaded"` otherwise.
    pub state: String,
    /// File format — always `"gguf"` for higgs-discovered models.
    pub format: String,
}

/// Request body for `POST /api/higgs/models/load`.
///
/// Absent load parameters fall back to the host-configured defaults.
#[derive(Debug, serde::Deserialize, ts_rs::TS)]
#[ts(export, export_to = "../../../frontend/src/lib/generated/")]
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
}

/// Response for `POST /api/higgs/models/load`: `{"status":"ok","id":…}`.
#[derive(Debug, serde::Serialize, ts_rs::TS)]
#[ts(export, export_to = "../../../frontend/src/lib/generated/")]
pub struct HiggsLoadResponse {
    /// Confirmation status; always `{"status":"ok"}` on success.
    #[serde(flatten)]
    pub status: HiggsOk,
    /// Id of the model that was loaded.
    pub id: String,
}

/// Response for `GET /api/higgs/logs`: `{"lines":[…]}`.
#[derive(Debug, serde::Serialize, ts_rs::TS)]
#[ts(export, export_to = "../../../frontend/src/lib/generated/")]
pub struct HiggsLogsResponse {
    /// Worker stderr tail, oldest first.
    pub lines: Vec<String>,
}

/// Error body for control routes: the rendered `HiggsError` display
/// (diagnostic code included), as `{"error":"<display>"}`.
#[derive(Debug, serde::Serialize, ts_rs::TS)]
#[ts(export, export_to = "../../../frontend/src/lib/generated/")]
pub struct HiggsErrorResponse {
    /// Human-readable failure, e.g. `[HG003] model not loaded: …`.
    pub error: String,
}

/// Response for `GET /api/higgs/version`.
#[derive(Debug, serde::Serialize, ts_rs::TS)]
#[ts(export, export_to = "../../../frontend/src/lib/generated/")]
pub struct HiggsVersionResponse {
    /// Higgs crate version from Cargo.toml (`CARGO_PKG_VERSION`).
    pub higgs: String,
    /// Human-readable engine name.
    pub engine: String,
    /// llama-cpp-2 dependency version.
    pub engine_version: String,
    /// File formats this runtime supports.
    pub supported_formats: Vec<String>,
}
