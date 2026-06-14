//! axum Router for higgs: OpenAI-compatible `/v1` plus `/api/higgs/*` control.
//!
//! All `/v1` bodies are `async-openai` wire types verbatim — nothing
//! hand-rolled. Control bodies are the `Higgs*` structs below (ts-rs exported
//! for the frontend). `/v1/models` lists LOADED models only (no JIT in v1:
//! chat against an unloaded model is a 404). Spec:
//! docs/superpowers/specs/2026-06-12-higgs-runtime-design.md
//!
//! ## Adding a control endpoint
//!
//! 1. Define a `Higgs*` response struct with `#[derive(serde::Serialize, ts_rs::TS)]`
//!    and `#[ts(export, export_to = "…/frontend/src/lib/generated/")]`; add its
//!    re-export line to `frontend/src/lib/types.ts`.
//! 2. Write an `async fn control_<name>(State(higgs): State<Arc<Higgs>>) -> Response`.
//! 3. Register it in [`router`] under `/api/higgs/<name>`.

mod stream;

use std::sync::Arc;

use async_openai::error::{ApiError, WrappedError};
use async_openai::types::chat::{
    ChatChoice, ChatCompletionMessageToolCalls,
    ChatCompletionRequestAssistantMessageContent as AssistantContent,
    ChatCompletionRequestAssistantMessageContentPart as AssistantPart,
    ChatCompletionRequestDeveloperMessageContent as DeveloperContent,
    ChatCompletionRequestDeveloperMessageContentPart as DeveloperPart,
    ChatCompletionRequestMessage, ChatCompletionRequestMessage as Msg,
    ChatCompletionRequestSystemMessageContent as SystemContent,
    ChatCompletionRequestSystemMessageContentPart as SystemPart,
    ChatCompletionRequestToolMessageContent as ToolContent,
    ChatCompletionRequestToolMessageContentPart as ToolPart,
    ChatCompletionRequestUserMessageContent as UserContent,
    ChatCompletionRequestUserMessageContentPart as UserPart, ChatCompletionResponseMessage,
    CompletionUsage, CreateChatCompletionRequest, CreateChatCompletionResponse, FinishReason, Role,
};
use async_openai::types::models::{ListModelResponse, Model};
use axum::extract::{Path, Query, State};
use axum::http::{HeaderValue, Method, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};

use tower_http::cors::{AllowOrigin, CorsLayer};

use crate::api::{ChatOutcome, Higgs};
use crate::diagnostic::HiggsError;
use crate::system::SystemInfo;
use crate::worker::engine::LoadParams;
use crate::worker::models::HiggsModel;

// ── Control wire structs ──────────────────────────────────────────────────────

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

/// Query parameters for `GET /api/higgs/logs` (`?n=200`).
#[derive(Debug, serde::Deserialize)]
struct LogsQuery {
    /// Maximum number of tail lines to return (default 200).
    n: Option<usize>,
}

// ── Error mapping ─────────────────────────────────────────────────────────────

/// Map a `HiggsError` to its HTTP status — the single status table for both
/// surfaces: HG002/HG003 → 404, HG005 → 400, HG006/HG007 → 503, else 500.
pub(crate) fn http_status(err: &HiggsError) -> StatusCode {
    match err {
        HiggsError::ModelNotFound { .. } | HiggsError::ModelNotLoaded { .. } => {
            StatusCode::NOT_FOUND
        }
        HiggsError::ContextOverflow { .. } => StatusCode::BAD_REQUEST,
        HiggsError::WorkerSpawnFailed { .. } | HiggsError::WorkerDead { .. } => {
            StatusCode::SERVICE_UNAVAILABLE
        }
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

/// Build the OpenAI error envelope `{"error":{message,type,code}}` for `/v1`.
///
/// `type` is `invalid_request_error` for 4xx and `server_error` otherwise;
/// `code` is `model_not_found` on 404 (the only coded case on this surface).
fn v1_envelope(status: StatusCode, message: String) -> WrappedError {
    let kind = if status.is_client_error() {
        "invalid_request_error"
    } else {
        "server_error"
    };
    let code = (status == StatusCode::NOT_FOUND).then(|| "model_not_found".to_owned());
    WrappedError {
        error: ApiError {
            message,
            r#type: Some(kind.to_owned()),
            param: None,
            code,
        },
    }
}

/// Serialize the `/v1` error envelope to a JSON string (SSE `data:` payloads).
pub(crate) fn v1_envelope_json(status: StatusCode, message: String) -> String {
    serde_json::to_string(&v1_envelope(status, message))
        .expect("error envelope serialization cannot fail")
}

/// `/v1` error response: mapped status + OpenAI error envelope body.
fn v1_error(err: &HiggsError) -> (StatusCode, Json<WrappedError>) {
    let status = http_status(err);
    (status, Json(v1_envelope(status, err.to_string())))
}

/// Control-route error response: mapped status + `{"error":"<display>"}` body.
fn control_error(err: &HiggsError) -> (StatusCode, Json<HiggsErrorResponse>) {
    (
        http_status(err),
        Json(HiggsErrorResponse {
            error: err.to_string(),
        }),
    )
}

// ── Router ────────────────────────────────────────────────────────────────────

use crate::LLAMA_CPP_2_VERSION;

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

/// Build the higgs router: OpenAI-compatible `/v1` + `/api/higgs/*` control.
///
/// The host nests or merges this into its own server; all state flows through
/// the shared [`Higgs`] facade.
pub fn router(higgs: Arc<Higgs>) -> Router {
    Router::new()
        .route("/v1/models", get(v1_models))
        .route("/v1/chat/completions", post(v1_chat_completions))
        .route("/api/higgs/models", get(control_models))
        .route("/api/higgs/models/load", post(control_load))
        .route("/api/higgs/models/unload", post(control_unload))
        .route("/api/higgs/models/{*id}", get(control_model_by_id))
        .route("/api/higgs/status", get(control_status))
        .route("/api/higgs/system", get(control_system))
        .route("/api/higgs/logs", get(control_logs))
        .route("/api/higgs/worker/start", post(control_worker_start))
        .route("/api/higgs/worker/stop", post(control_worker_stop))
        .route("/api/higgs/version", get(control_version))
        .layer(local_cors())
        .with_state(higgs)
}

/// CORS for higgs's standalone listener: the frontend calls higgs's own port
/// cross-origin (from the Tauri webview or the dev server), so the local UI
/// origins must be allowed. Localhost/webview only — not a public surface.
fn local_cors() -> CorsLayer {
    let origins: Vec<HeaderValue> = [
        "tauri://localhost",
        "http://tauri.localhost",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]
    .iter()
    .filter_map(|o| o.parse().ok())
    .collect();
    CorsLayer::new()
        .allow_origin(AllowOrigin::list(origins))
        .allow_methods([Method::GET, Method::POST])
        .allow_headers(tower_http::cors::Any)
}

// ── /v1 handlers ──────────────────────────────────────────────────────────────

/// Current Unix time in whole seconds (OpenAI `created` field).
fn now_secs() -> u32 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as u32
}

/// Fresh `chatcmpl-…` response id (mirrors shimmy/omlx id construction).
fn chatcmpl_id() -> String {
    format!("chatcmpl-{}", uuid::Uuid::new_v4().simple())
}

/// `GET /v1/models` — LOADED models only: answers "what can serve chat right
/// now", never the on-disk catalog (that is the control `models` route).
async fn v1_models(State(higgs): State<Arc<Higgs>>) -> Response {
    tracing::info!("higgs: GET /v1/models");
    match higgs.status().await {
        Ok(status) => {
            let data = status
                .loaded
                .into_iter()
                .map(|l| Model {
                    id: l.id,
                    object: "model".to_owned(),
                    created: now_secs(),
                    owned_by: "higgs".to_owned(),
                })
                .collect();
            Json(ListModelResponse {
                object: "list".to_owned(),
                data,
            })
            .into_response()
        }
        Err(err) => {
            tracing::warn!(error = %err, "higgs: /v1/models failed");
            v1_error(&err).into_response()
        }
    }
}

/// `POST /v1/chat/completions` — no JIT in v1: the named model must already
/// be loaded; unknown and on-disk-but-unloaded ids both get the HG003 404.
async fn v1_chat_completions(
    State(higgs): State<Arc<Higgs>>,
    Json(req): Json<CreateChatCompletionRequest>,
) -> Response {
    tracing::info!(model = %req.model, stream = req.stream.unwrap_or(false), "higgs: POST /v1/chat/completions");

    // Loaded-model gate (a dead worker also reports nothing loaded).
    let loaded = match higgs.status().await {
        Ok(s) => s.loaded,
        Err(err) => {
            tracing::warn!(error = %err, "higgs: chat status check failed");
            return v1_error(&err).into_response();
        }
    };
    if loaded.is_none_or(|l| l.id != req.model) {
        let err = HiggsError::ModelNotLoaded {
            id: req.model.clone(),
        };
        tracing::warn!(model = %req.model, "higgs: chat for unloaded model");
        return v1_error(&err).into_response();
    }

    let pairs = match messages_to_pairs(&req.messages) {
        Ok(p) => p,
        // v1 is text-only: image/audio/file/refusal content parts → 400.
        Err(reject) => {
            tracing::warn!(detail = %reject, "higgs: chat request rejected");
            let status = StatusCode::BAD_REQUEST;
            return (status, Json(v1_envelope(status, reject))).into_response();
        }
    };

    // Both the deprecated `max_tokens` and `max_completion_tokens` are
    // honored; the newer field wins when both are present. Default 1024.
    #[allow(deprecated)]
    let max_tokens = req.max_completion_tokens.or(req.max_tokens).unwrap_or(1024) as usize;
    let temperature = req.temperature.unwrap_or(0.7);

    // Serialize the OpenAI `tools` array to a JSON string for the chat template.
    // A serialization failure is a malformed request body → 400.
    let tools_json = match req.tools.as_ref().map(serde_json::to_string).transpose() {
        Ok(t) => t,
        Err(err) => {
            tracing::warn!(error = %err, "higgs: chat tools serialization failed");
            let status = StatusCode::BAD_REQUEST;
            return (
                status,
                Json(v1_envelope(status, format!("invalid tools: {err}"))),
            )
                .into_response();
        }
    };

    let (deltas, outcome) = match higgs
        .chat_stream(pairs, max_tokens, temperature, tools_json)
        .await
    {
        Ok(pair) => pair,
        Err(err) => {
            tracing::warn!(error = %err, "higgs: chat_stream failed to start");
            return v1_error(&err).into_response();
        }
    };

    if req.stream.unwrap_or(false) {
        stream::chat_sse(chatcmpl_id(), req.model, now_secs(), deltas, outcome).into_response()
    } else {
        // Non-streaming: ChatOutcome.content is the canonical full text; the
        // delta receiver is dropped (the worker-side sender no-ops once closed).
        drop(deltas);
        match outcome.await {
            Ok(Ok(out)) => Json(chat_response(req.model, &out)).into_response(),
            Ok(Err(err)) => {
                tracing::warn!(error = %err, "higgs: chat failed");
                v1_error(&err).into_response()
            }
            // JoinError: the chat task panicked or was aborted — not a HiggsError.
            Err(join_err) => {
                tracing::warn!(error = %join_err, "higgs: chat task failed");
                let status = StatusCode::INTERNAL_SERVER_ERROR;
                (
                    status,
                    Json(v1_envelope(status, format!("chat task failed: {join_err}"))),
                )
                    .into_response()
            }
        }
    }
}

/// Build the non-streaming chat completion response from a final outcome.
fn chat_response(model: String, out: &ChatOutcome) -> CreateChatCompletionResponse {
    // Deserialize the parser-produced `tool_calls` array into the async-openai
    // wire type. This data was produced by the crate's OAI-compat parser, so it
    // is already OpenAI-shaped; a deserialize failure is an internal bug — log
    // the actual serde error and degrade to no tool calls rather than panic.
    let tool_calls: Option<Vec<ChatCompletionMessageToolCalls>> =
        out.tool_calls.as_ref().and_then(|v| {
            serde_json::from_value(v.clone())
                .map_err(|err| {
                    tracing::error!(error = %err, "higgs: tool_calls deserialize failed");
                    err
                })
                .ok()
        });
    // Per the OpenAI spec, a turn that emits tool calls reports finish_reason
    // "tool_calls" regardless of the engine's stop/length signal.
    let finish_reason = if tool_calls.is_some() {
        FinishReason::ToolCalls
    } else {
        stream::finish_reason_from(&out.finish_reason)
    };
    // function_call / system_fingerprint are deprecated-but-required fields of
    // the async-openai wire structs; populating them with None is the only way
    // to construct the type.
    #[allow(deprecated)]
    CreateChatCompletionResponse {
        id: chatcmpl_id(),
        choices: vec![ChatChoice {
            index: 0,
            message: ChatCompletionResponseMessage {
                content: Some(out.content.clone()),
                refusal: None,
                tool_calls,
                annotations: None,
                role: Role::Assistant,
                function_call: None,
                audio: None,
            },
            finish_reason: Some(finish_reason),
            logprobs: None,
        }],
        created: now_secs(),
        model,
        service_tier: None,
        system_fingerprint: None,
        object: "chat.completion".to_owned(),
        // Real token counts from the engine, wired through ChatOutcome.
        usage: Some(CompletionUsage {
            prompt_tokens: out.prompt_tokens,
            completion_tokens: out.completion_tokens,
            total_tokens: out.prompt_tokens + out.completion_tokens,
            ..Default::default()
        }),
    }
}

/// Flatten OpenAI request messages into the worker's `(role, content)` pairs.
///
/// v1 is text-only: plain `Text` content and text-part arrays (joined with
/// `\n`, the shimmy convention) pass through; any other part kind
/// (image/audio/file/refusal) is rejected with a description the caller
/// turns into a 400.
fn messages_to_pairs(
    messages: &[ChatCompletionRequestMessage],
) -> Result<Vec<(String, String)>, String> {
    /// Join already-extracted text parts (shimmy convention: `\n`).
    fn join(parts: &[String]) -> String {
        parts.join("\n")
    }

    messages
        .iter()
        .map(|m| match m {
            Msg::Developer(d) => {
                let text = match &d.content {
                    DeveloperContent::Text(s) => s.clone(),
                    DeveloperContent::Array(parts) => join(
                        &parts
                            .iter()
                            .map(|DeveloperPart::Text(t)| t.text.clone())
                            .collect::<Vec<_>>(),
                    ),
                };
                Ok(("developer".to_owned(), text))
            }
            Msg::System(s) => {
                let text = match &s.content {
                    SystemContent::Text(t) => t.clone(),
                    SystemContent::Array(parts) => join(
                        &parts.iter().map(|SystemPart::Text(t)| t.text.clone()).collect::<Vec<_>>(),
                    ),
                };
                Ok(("system".to_owned(), text))
            }
            Msg::User(u) => {
                let text = match &u.content {
                    UserContent::Text(t) => t.clone(),
                    UserContent::Array(parts) => join(
                        &parts
                            .iter()
                            .map(|p| match p {
                                UserPart::Text(t) => Ok(t.text.clone()),
                                UserPart::ImageUrl(_) | UserPart::InputAudio(_) | UserPart::File(_) => {
                                    Err("user message contains a non-text content part (v1 is text-only)"
                                        .to_owned())
                                }
                            })
                            .collect::<Result<Vec<_>, _>>()?,
                    ),
                };
                Ok(("user".to_owned(), text))
            }
            Msg::Assistant(a) => {
                let text = match &a.content {
                    None => String::new(),
                    Some(AssistantContent::Text(t)) => t.clone(),
                    Some(AssistantContent::Array(parts)) => join(
                        &parts
                            .iter()
                            .map(|p| match p {
                                AssistantPart::Text(t) => Ok(t.text.clone()),
                                AssistantPart::Refusal(_) => Err(
                                    "assistant message contains a refusal content part (v1 is text-only)"
                                        .to_owned(),
                                ),
                            })
                            .collect::<Result<Vec<_>, _>>()?,
                    ),
                };
                Ok(("assistant".to_owned(), text))
            }
            Msg::Tool(t) => {
                let text = match &t.content {
                    ToolContent::Text(s) => s.clone(),
                    ToolContent::Array(parts) => join(
                        &parts.iter().map(|ToolPart::Text(p)| p.text.clone()).collect::<Vec<_>>(),
                    ),
                };
                Ok(("tool".to_owned(), text))
            }
            Msg::Function(f) => {
                Ok(("function".to_owned(), f.content.clone().unwrap_or_default()))
            }
        })
        .collect()
}

// ── /api/higgs/* handlers ─────────────────────────────────────────────────────

/// `GET /api/higgs/models` — live scan of all configured directories, plus
/// the currently loaded model id.
// v1: two RPC round-trips (scan + status) — catalog reads are UI-paced, latency acceptable; single-RPC catalog is a v2 item
async fn control_models(State(higgs): State<Arc<Higgs>>) -> Response {
    tracing::info!("higgs: GET /api/higgs/models");
    let models = match higgs.scan().await {
        Ok(m) => m,
        Err(err) => {
            tracing::warn!(error = %err, "higgs: scan failed");
            return control_error(&err).into_response();
        }
    };
    let loaded_id = match higgs.status().await {
        Ok(s) => s.loaded.map(|l| l.id),
        Err(err) => {
            tracing::warn!(error = %err, "higgs: status failed");
            return control_error(&err).into_response();
        }
    };
    let entries: Vec<HiggsModelEntry> = models
        .into_iter()
        .map(|m| {
            let is_loaded = loaded_id.as_deref() == Some(m.id.as_str());
            HiggsModelEntry {
                state: if is_loaded {
                    "loaded".to_owned()
                } else {
                    "not-loaded".to_owned()
                },
                format: "gguf".to_owned(),
                model: m,
            }
        })
        .collect();
    Json(HiggsModelsResponse {
        models: entries,
        loaded_id,
    })
    .into_response()
}

/// `GET /api/higgs/models/{*id}` — single enriched model by HuggingFace repo id.
///
/// The wildcard captures the full remaining path so slashed HF repo ids
/// (`org/model`, `lmstudio-community/Foo-GGUF`) round-trip correctly.
/// Returns [`HiggsModelEntry`] on success, or 404 [`HiggsErrorResponse`] when the
/// id is absent from the scanned catalog.
async fn control_model_by_id(State(higgs): State<Arc<Higgs>>, Path(id): Path<String>) -> Response {
    tracing::info!(id = %id, "higgs: GET /api/higgs/models/{{*id}}");
    let models = match higgs.scan().await {
        Ok(m) => m,
        Err(err) => {
            tracing::warn!(id = %id, error = %err, "higgs: scan failed");
            return control_error(&err).into_response();
        }
    };
    let loaded_id = match higgs.status().await {
        Ok(s) => s.loaded.map(|l| l.id),
        Err(err) => {
            tracing::warn!(id = %id, error = %err, "higgs: status failed");
            return control_error(&err).into_response();
        }
    };
    match models.into_iter().find(|m| m.id == id) {
        Some(model) => {
            let is_loaded = loaded_id.as_deref() == Some(model.id.as_str());
            let entry = HiggsModelEntry {
                state: if is_loaded {
                    "loaded".to_owned()
                } else {
                    "not-loaded".to_owned()
                },
                format: "gguf".to_owned(),
                model,
            };
            Json(entry).into_response()
        }
        None => {
            let err = HiggsError::ModelNotFound { id };
            tracing::warn!(error = %err, "higgs: model not found");
            control_error(&err).into_response()
        }
    }
}

/// `POST /api/higgs/models/load` — load a model by id; absent parameters
/// fall back to the host-configured defaults.
async fn control_load(
    State(higgs): State<Arc<Higgs>>,
    Json(req): Json<HiggsLoadRequest>,
) -> Response {
    tracing::warn!(id = %req.id, "higgs: loading model");
    let params = if req.ctx_len.is_none() && req.gpu_layers.is_none() && req.threads.is_none() {
        None // fully default load — Higgs::load applies default_load itself
    } else {
        let base = higgs.default_load();
        Some(LoadParams {
            ctx_len: req.ctx_len.unwrap_or(base.ctx_len),
            gpu_layers: req.gpu_layers.unwrap_or(base.gpu_layers),
            threads: req.threads.unwrap_or(base.threads),
        })
    };
    match higgs.load(&req.id, params).await {
        Ok(()) => Json(HiggsLoadResponse {
            status: HiggsOk::new(),
            id: req.id,
        })
        .into_response(),
        Err(err) => {
            tracing::warn!(id = %req.id, error = %err, "higgs: load failed");
            control_error(&err).into_response()
        }
    }
}

/// `POST /api/higgs/models/unload` — unload the current model.
async fn control_unload(State(higgs): State<Arc<Higgs>>) -> Response {
    tracing::warn!("higgs: unloading model");
    match higgs.unload().await {
        Ok(()) => Json(HiggsOk::new()).into_response(),
        Err(err) => {
            tracing::warn!(error = %err, "higgs: unload failed");
            control_error(&err).into_response()
        }
    }
}

/// `GET /api/higgs/status` — live engine + model status snapshot.
async fn control_status(State(higgs): State<Arc<Higgs>>) -> Response {
    tracing::info!("higgs: GET /api/higgs/status");
    match higgs.status().await {
        Ok(status) => Json(status).into_response(),
        Err(err) => {
            tracing::warn!(error = %err, "higgs: status failed");
            control_error(&err).into_response()
        }
    }
}

/// `GET /api/higgs/logs?n=200` — worker stderr tail, oldest first.
async fn control_logs(
    State(higgs): State<Arc<Higgs>>,
    Query(q): Query<LogsQuery>,
) -> Json<HiggsLogsResponse> {
    tracing::info!(n = q.n.unwrap_or(200), "higgs: GET /api/higgs/logs");
    Json(HiggsLogsResponse {
        lines: higgs.logs(q.n.unwrap_or(200)),
    })
}

/// `POST /api/higgs/worker/start` — spawn the worker process.
async fn control_worker_start(State(higgs): State<Arc<Higgs>>) -> Response {
    tracing::warn!("higgs: starting worker");
    match higgs.start().await {
        Ok(()) => Json(HiggsOk::new()).into_response(),
        Err(err) => {
            tracing::warn!(error = %err, "higgs: worker start failed");
            control_error(&err).into_response()
        }
    }
}

/// `POST /api/higgs/worker/stop` — gracefully shut down the worker.
async fn control_worker_stop(State(higgs): State<Arc<Higgs>>) -> Json<HiggsOk> {
    tracing::warn!("higgs: stopping worker");
    higgs.stop().await;
    Json(HiggsOk::new())
}

/// `GET /api/higgs/version` — higgs build version and engine info.
async fn control_version() -> Json<HiggsVersionResponse> {
    tracing::info!("higgs: GET /api/higgs/version");
    Json(HiggsVersionResponse {
        higgs: env!("CARGO_PKG_VERSION").to_owned(),
        engine: "llama.cpp via llama-cpp-2".to_owned(),
        engine_version: LLAMA_CPP_2_VERSION.to_owned(),
        supported_formats: vec!["gguf".to_owned()],
    })
}

/// `GET /api/higgs/system` — host hardware (CPU/RAM/load) + inference runtime.
///
/// Gathering samples CPU load over a short interval, so it runs on a blocking
/// thread to avoid stalling the async executor.
async fn control_system() -> Json<SystemInfo> {
    tracing::info!("higgs: GET /api/higgs/system");
    let info = tokio::task::spawn_blocking(SystemInfo::gather)
        .await
        .expect("system info gather task");
    Json(info)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::HiggsConfig;
    use crate::rpc::{encode, RpcFrame, RpcNotification, RpcResponse};
    use crate::supervisor::{Supervisor, WorkerHalves};
    use crate::worker::N_CHAT_CHUNK;
    use async_openai::types::chat::{CreateChatCompletionStreamResponse, FinishReason};
    use axum::body::Body;
    use axum::http::Request;
    use http_body_util::BodyExt;
    use parking_lot::Mutex;
    use serde_json::json;
    use std::collections::VecDeque;
    use std::time::Duration;
    use tokio::io::AsyncWriteExt;
    use tower::ServiceExt;

    // ── Test seam (mirrored from api::tests::make_supervisor) ────────────────

    /// Build a `Supervisor` plus duplex test handles and its captured log ring.
    fn make_supervisor() -> (
        Supervisor,
        tokio::io::DuplexStream, // test_write: write responses → supervisor reads
        tokio::io::DuplexStream, // test_read:  supervisor writes requests → test reads
        Arc<Mutex<VecDeque<String>>>, // stderr ring (push lines for logs tests)
    ) {
        let (sup_write, test_read) = tokio::io::duplex(64 * 1024);
        let (test_write, sup_read) = tokio::io::duplex(64 * 1024);

        let sup_write_cell = Arc::new(Mutex::new(Some(sup_write)));
        let sup_read_cell = Arc::new(Mutex::new(Some(sup_read)));
        let ring_cell: Arc<Mutex<Option<Arc<Mutex<VecDeque<String>>>>>> =
            Arc::new(Mutex::new(None));
        let ring_capture = Arc::clone(&ring_cell);

        let sup = Supervisor::with_factory(Box::new(move |ring| {
            *ring_capture.lock() = Some(ring);
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
            })
        }));

        sup.start().expect("mock start");
        let ring = ring_cell.lock().take().expect("factory ran on start");
        (sup, test_write, test_read, ring)
    }

    async fn write_response(
        stream: &mut tokio::io::DuplexStream,
        id: u64,
        result: serde_json::Value,
    ) {
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

    async fn write_chunk_notification(
        stream: &mut tokio::io::DuplexStream,
        request_id: u64,
        delta: &str,
    ) {
        let line = encode(&RpcFrame::Notification(RpcNotification {
            jsonrpc: "2.0".into(),
            method: N_CHAT_CHUNK.into(),
            // request_id matches the M_CHAT RPC id so route_notification
            // delivers this delta to the correct keyed sink.
            params: json!({ "request_id": request_id, "delta": delta }),
        }));
        stream
            .write_all(format!("{line}\n").as_bytes())
            .await
            .unwrap();
        stream.flush().await.unwrap();
    }

    /// Wrap a mock supervisor in a `Higgs` facade and build the router.
    fn make_app(sup: Supervisor) -> Router {
        router(Arc::new(Higgs::with_supervisor(
            Arc::new(sup),
            HiggsConfig::default(),
        )))
    }

    fn get(uri: &str) -> Request<Body> {
        Request::builder().uri(uri).body(Body::empty()).unwrap()
    }

    fn post_json(uri: &str, body: &serde_json::Value) -> Request<Body> {
        Request::builder()
            .method("POST")
            .uri(uri)
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap()
    }

    async fn body_bytes(resp: Response) -> Vec<u8> {
        resp.into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes()
            .to_vec()
    }

    fn loaded_status_json() -> serde_json::Value {
        json!({
            "loaded": { "id": "org/model", "ctx_len": 4096, "gpu_layers": 99, "threads": 4 },
            "models_scanned": 1,
        })
    }

    // ── Test 1: /v1/models empty when nothing is loaded ──────────────────────

    #[tokio::test]
    async fn v1_models_empty_when_unloaded() {
        let (sup, mut test_write, _test_read, _ring) = make_supervisor();
        let app = make_app(sup);

        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(30)).await;
            write_response(
                &mut test_write,
                1,
                json!({"loaded": null, "models_scanned": 0}),
            )
            .await;
        });

        let resp = app.oneshot(get("/v1/models")).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let list: ListModelResponse = serde_json::from_slice(&body_bytes(resp).await).unwrap();
        assert_eq!(list.object, "list");
        assert!(list.data.is_empty());
    }

    // ── Test 2: /v1/models lists the loaded model ─────────────────────────────

    #[tokio::test]
    async fn v1_models_lists_loaded() {
        let (sup, mut test_write, _test_read, _ring) = make_supervisor();
        let app = make_app(sup);

        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(30)).await;
            write_response(&mut test_write, 1, loaded_status_json()).await;
        });

        let resp = app.oneshot(get("/v1/models")).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let list: ListModelResponse = serde_json::from_slice(&body_bytes(resp).await).unwrap();
        assert_eq!(list.data.len(), 1);
        assert_eq!(list.data[0].id, "org/model");
        assert_eq!(list.data[0].object, "model");
        assert_eq!(list.data[0].owned_by, "higgs");
    }

    // ── Test 3: chat against an unloaded model → 404 HG003 envelope ──────────

    #[tokio::test]
    async fn chat_unloaded_404_hg003() {
        let (sup, mut test_write, _test_read, _ring) = make_supervisor();
        let app = make_app(sup);

        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(30)).await;
            write_response(
                &mut test_write,
                1,
                json!({"loaded": null, "models_scanned": 0}),
            )
            .await;
        });

        let req = post_json(
            "/v1/chat/completions",
            &json!({"model": "org/missing", "messages": [{"role": "user", "content": "hi"}]}),
        );
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);

        let body = String::from_utf8(body_bytes(resp).await).unwrap();
        assert!(
            body.contains("[HG003]"),
            "body carries the diagnostic code: {body}"
        );
        assert!(
            body.contains("model_not_found"),
            "envelope code present: {body}"
        );
        assert!(
            body.contains("invalid_request_error"),
            "envelope type present: {body}"
        );
    }

    // ── Test 4: non-streaming chat returns ChatOutcome.content ───────────────

    #[tokio::test]
    async fn chat_nonstream_returns_content() {
        let (sup, mut test_write, _test_read, _ring) = make_supervisor();
        let app = make_app(sup);

        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(30)).await;
            write_response(&mut test_write, 1, loaded_status_json()).await; // status gate
            tokio::time::sleep(Duration::from_millis(50)).await;
            write_response(
                &mut test_write,
                2,
                json!({"content": "hello", "finish_reason": "stop", "prompt_tokens": 12, "completion_tokens": 5}), // higgs/chat
            )
            .await;
        });

        let req = post_json(
            "/v1/chat/completions",
            &json!({"model": "org/model", "messages": [{"role": "user", "content": "hi"}]}),
        );
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let chat: CreateChatCompletionResponse =
            serde_json::from_slice(&body_bytes(resp).await).unwrap();
        assert_eq!(chat.object, "chat.completion");
        assert_eq!(chat.model, "org/model");
        assert!(chat.id.starts_with("chatcmpl-"));
        assert_eq!(chat.choices[0].message.content.as_deref(), Some("hello"));
        assert_eq!(chat.choices[0].finish_reason, Some(FinishReason::Stop));
        let usage = chat.usage.expect("usage must be present");
        assert_eq!(usage.prompt_tokens, 12);
        assert_eq!(usage.completion_tokens, 5);
        assert_eq!(usage.total_tokens, 17);
    }

    // ── Test 5: streaming chat SSE framing ───────────────────────────────────

    #[tokio::test]
    async fn chat_stream_sse_framing() {
        let (sup, mut test_write, _test_read, _ring) = make_supervisor();
        let app = make_app(sup);

        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(30)).await;
            write_response(&mut test_write, 1, loaded_status_json()).await; // status gate
            tokio::time::sleep(Duration::from_millis(50)).await;
            // Chunks tagged with request_id=2 (the M_CHAT RPC id, allocated after status id=1).
            write_chunk_notification(&mut test_write, 2, "hel").await;
            write_chunk_notification(&mut test_write, 2, "lo").await;
            tokio::time::sleep(Duration::from_millis(50)).await;
            write_response(
                &mut test_write,
                2,
                json!({"content": "hello", "finish_reason": "stop"}), // higgs/chat
            )
            .await;
        });

        let req = post_json(
            "/v1/chat/completions",
            &json!({
                "model": "org/model",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": true,
            }),
        );
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let content_type = resp
            .headers()
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap();
        assert!(
            content_type.starts_with("text/event-stream"),
            "got: {content_type}"
        );

        let body = String::from_utf8(body_bytes(resp).await).unwrap();
        let datas: Vec<&str> = body
            .lines()
            .filter_map(|l| l.strip_prefix("data: "))
            .collect();
        assert_eq!(datas.len(), 5, "role + 2 deltas + finish + [DONE]: {body}");

        let parse =
            |s: &str| -> CreateChatCompletionStreamResponse { serde_json::from_str(s).unwrap() };
        let role = parse(datas[0]);
        assert_eq!(role.choices[0].delta.role, Some(Role::Assistant));
        assert_eq!(role.object, "chat.completion.chunk");
        assert_eq!(
            parse(datas[1]).choices[0].delta.content.as_deref(),
            Some("hel")
        );
        assert_eq!(
            parse(datas[2]).choices[0].delta.content.as_deref(),
            Some("lo")
        );
        let finish = parse(datas[3]);
        assert_eq!(finish.choices[0].finish_reason, Some(FinishReason::Stop));
        assert_eq!(datas[4], "[DONE]");
    }

    // ── Test 6: control load + unload roundtrip ──────────────────────────────

    #[tokio::test]
    async fn control_load_unload_roundtrip() {
        let (sup, mut test_write, _test_read, _ring) = make_supervisor();
        let app = make_app(sup);

        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(30)).await;
            write_response(&mut test_write, 1, json!({"id": "org/model"})).await; // higgs/load
            tokio::time::sleep(Duration::from_millis(50)).await;
            write_response(&mut test_write, 2, loaded_status_json()).await; // status (unload id capture)
            tokio::time::sleep(Duration::from_millis(30)).await;
            write_response(&mut test_write, 3, serde_json::Value::Null).await; // higgs/unload
        });

        let resp = app
            .clone()
            .oneshot(post_json(
                "/api/higgs/models/load",
                &json!({"id": "org/model"}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let v: serde_json::Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
        assert_eq!(v["status"], "ok");
        assert_eq!(v["id"], "org/model");

        let resp = app
            .oneshot(post_json("/api/higgs/models/unload", &json!({})))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let v: serde_json::Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
        assert_eq!(v["status"], "ok");
    }

    // ── Test 7: control_model_by_id with a slashed HF repo id ───────────────
    //
    // This is the regression test for the wildcard-route bug: with the old
    // single-segment `{id}` route, a request to `/api/higgs/models/org/model`
    // (literal slash in the path, as real curl sends) never matched — axum
    // treated `org` and `model` as separate segments. The test previously used
    // `org%2Fmodel` (percent-encoded) which happened to work against the broken
    // route because `%2F` is a single segment. Using a literal slash here
    // ensures the wildcard `{*id}` route is exercised as real callers do.

    #[tokio::test]
    async fn control_model_by_id_found_slashed() {
        let (sup, mut test_write, _test_read, _ring) = make_supervisor();
        let app = make_app(sup);

        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(30)).await;
            // scan response — realistic HF repo id with slash
            write_response(
                &mut test_write,
                1,
                serde_json::json!([{
                    "id": "lmstudio-community/NVIDIA-Nemotron-3-Nano-4B-GGUF",
                    "path": "/models/model.gguf",
                    "size_bytes": 4000000000u64,
                    "quant": "Q4_K_M",
                    "source": "LmStudio",
                    "arch": "llama",
                    "ctx_train": 8192u64,
                    "has_chat_template": true,
                }]),
            )
            .await;
            tokio::time::sleep(Duration::from_millis(30)).await;
            // status response (nothing loaded)
            write_response(
                &mut test_write,
                2,
                serde_json::json!({"loaded": null, "models_scanned": 1}),
            )
            .await;
        });

        // Literal slash in the URL — this is what real curl sends and what
        // the old `{id}` route could never match.
        let resp = app
            .oneshot(get(
                "/api/higgs/models/lmstudio-community/NVIDIA-Nemotron-3-Nano-4B-GGUF",
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let v: serde_json::Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
        assert_eq!(v["id"], "lmstudio-community/NVIDIA-Nemotron-3-Nano-4B-GGUF");
        assert_eq!(v["state"], "not-loaded");
        assert_eq!(v["format"], "gguf");
        assert_eq!(v["arch"], "llama");
    }

    // ── Test 8: control_model_by_id not found (slashed id) ───────────────────

    #[tokio::test]
    async fn control_model_by_id_not_found() {
        let (sup, mut test_write, _test_read, _ring) = make_supervisor();
        let app = make_app(sup);

        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(30)).await;
            write_response(&mut test_write, 1, serde_json::json!([])).await;
            tokio::time::sleep(Duration::from_millis(30)).await;
            write_response(
                &mut test_write,
                2,
                serde_json::json!({"loaded": null, "models_scanned": 0}),
            )
            .await;
        });

        // Slashed id that does not exist in the catalog → 404 HG002.
        let resp = app
            .oneshot(get("/api/higgs/models/org/nope"))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        let v: serde_json::Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
        assert!(v["error"].as_str().unwrap().contains("[HG002]"));
    }

    // ── Test 9: version endpoint ──────────────────────────────────────────────

    #[tokio::test]
    async fn version_endpoint() {
        let (sup, _test_write, _test_read, _ring) = make_supervisor();
        let app = make_app(sup);

        let resp = app.oneshot(get("/api/higgs/version")).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let v: serde_json::Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
        assert!(v["higgs"].as_str().is_some(), "higgs version present");
        assert_eq!(v["engine"], "llama.cpp via llama-cpp-2");
        assert!(v["engine_version"].as_str().is_some());
        let fmts = v["supported_formats"].as_array().expect("array");
        assert!(fmts.contains(&serde_json::Value::String("gguf".to_owned())));
    }

    // ── (original Test 7, now test 10): logs endpoint shape and tail semantics ─

    #[tokio::test]
    async fn logs_endpoint_shapes() {
        let (sup, _test_write, _test_read, ring) = make_supervisor();
        {
            let mut r = ring.lock();
            r.push_back("line one".to_owned());
            r.push_back("line two".to_owned());
            r.push_back("line three".to_owned());
        }
        let app = make_app(sup);

        let resp = app.oneshot(get("/api/higgs/logs?n=2")).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let v: serde_json::Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
        assert_eq!(
            v["lines"],
            json!(["line two", "line three"]),
            "tail of n, oldest first"
        );
    }
}
