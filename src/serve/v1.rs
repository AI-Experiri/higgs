//! The OpenAI-compatible `/v1` surface: `GET /v1/models` and
//! `POST /v1/chat/completions`.
//!
//! All bodies are `async-openai` wire types verbatim — nothing hand-rolled.
//! `/v1/models` lists LOADED models only. Chat has JIT (just-in-time loading)
//! ON by default: a request for a scanned-but-unloaded model loads it on demand
//! (swapping out any resident model — higgs serves one at a time) before
//! serving; with JIT off, chat against an unloaded model is a 404. Errors render
//! as the OpenAI envelope `{"error":{message,type,code}}`.

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
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;

use crate::api::{ChatOutcome, Higgs, LoadedInfo};
use crate::diagnostic::HiggsError;
use crate::worker::engine::CtxLen;

use super::http_status;
use super::stream;
use super::{MAX_OUTPUT_TOKENS, PROMPT_BYTES_PER_TOKEN};

/// Build the OpenAI error envelope `{"error":{message,type,code}}` for `/v1`.
///
/// `type` is `invalid_request_error` for 4xx and `server_error` otherwise;
/// `code` is `model_not_found` on 404 (the only coded case on this surface).
///
/// The message is [`redact_paths`]-sanitized: the `/v1` surface is the untrusted
/// OpenAI-interop boundary, so host filesystem paths and bind addresses are
/// stripped from the client-facing string. The full unredacted Display (with
/// paths) is still logged server-side at the origin (four-pillar pillar 1) and
/// returned verbatim on the `/api/higgs/*` control surface, which is ours.
fn v1_envelope(status: StatusCode, message: &str) -> WrappedError {
    let kind = if status.is_client_error() {
        "invalid_request_error"
    } else {
        "server_error"
    };
    let code = (status == StatusCode::NOT_FOUND).then(|| "model_not_found".to_owned());
    WrappedError {
        error: ApiError {
            message: redact_paths(message),
            r#type: Some(kind.to_owned()),
            param: None,
            code,
        },
    }
}

/// Strip host filesystem paths and `host:port` bind addresses from a string
/// before it crosses the `/v1` client boundary. A no-auth interop surface must
/// not leak the host's directory layout or listen address in error text.
///
/// Replaces each whitespace-delimited token that looks like an absolute path
/// (starts with `/`, or a Windows drive like `C:\`) or a `host:port` literal
/// with `<redacted>`. Diagnostic codes (`[HG004]`), ids (`org/model` — no
/// leading slash), and human prose are preserved. Conservative by design: it
/// only redacts tokens that are unambiguously a path or address, so a normal
/// error message reads unchanged.
fn redact_paths(message: &str) -> String {
    /// Whether `tok` looks like an absolute host path or a `host:port` address.
    fn is_sensitive(tok: &str) -> bool {
        // Trim surrounding punctuation a path/address may be wrapped in.
        let t = tok.trim_matches(|c: char| matches!(c, '(' | ')' | '\'' | '"' | ',' | '.'));
        if t.is_empty() {
            return false;
        }
        // Unix absolute path: leading `/` (an `org/model` id has no leading slash).
        if t.starts_with('/') {
            return true;
        }
        // Windows drive path: `C:\…` or `C:/…`.
        let bytes = t.as_bytes();
        if bytes.len() >= 3
            && bytes[0].is_ascii_alphabetic()
            && bytes[1] == b':'
            && (bytes[2] == b'\\' || bytes[2] == b'/')
        {
            return true;
        }
        // `host:port` bind address: `127.0.0.1:8081`, `localhost:11434`.
        if let Some((host, port)) = t.rsplit_once(':') {
            if !host.is_empty() && !port.is_empty() && port.bytes().all(|b| b.is_ascii_digit()) {
                return true;
            }
        }
        false
    }

    message
        .split(' ')
        .map(|tok| if is_sensitive(tok) { "<redacted>" } else { tok })
        .collect::<Vec<_>>()
        .join(" ")
}

/// The OpenAI-standard `error.code` for higgs errors that map to one. A context
/// overflow — raised locally by the prompt pre-check or propagated from the worker's
/// tokenizer-exact `[HG005]` — surfaces as `context_length_exceeded` (the code the
/// OpenAI SDK and downstream clients special-case), rather than the `null` it was.
fn v1_error_code(err: &HiggsError) -> Option<&'static str> {
    match err {
        HiggsError::ContextOverflow { .. } => Some("context_length_exceeded"),
        HiggsError::WorkerRpc { worker_code, .. } if worker_code.as_deref() == Some("HG005") => {
            Some("context_length_exceeded")
        }
        // A model that isn't Prepared (or whose profile is stale) can't be served
        // until the user acts — surface one client-stable code for both.
        HiggsError::NotPrepared { .. } | HiggsError::ProfileStale { .. } => {
            Some("model_not_prepared")
        }
        _ => None,
    }
}

/// Build the OpenAI error envelope for a `HiggsError`: the shared status-based
/// envelope plus the specific OpenAI `code` when one applies ([`v1_error_code`]).
/// Shared by the non-streaming response ([`v1_error`]) and the SSE error payload
/// ([`v1_sse_error`]) so a context overflow surfaces `context_length_exceeded` on
/// BOTH the streaming and non-streaming paths.
fn v1_envelope_for(err: &HiggsError) -> WrappedError {
    let mut env = v1_envelope(http_status(err), &err.to_string());
    if let Some(code) = v1_error_code(err) {
        env.error.code = Some(code.to_owned());
    }
    env
}

/// `/v1` error response: mapped status + OpenAI error envelope body.
fn v1_error(err: &HiggsError) -> (StatusCode, Json<WrappedError>) {
    (http_status(err), Json(v1_envelope_for(err)))
}

/// The SSE error-payload JSON for a mid-stream `HiggsError` — same status + code
/// mapping as [`v1_error`], so a streaming context overflow (the worker's
/// tokenizer-exact `[HG005]` after the serve-layer lower-bound check passed) carries
/// `context_length_exceeded` just like the non-streaming path.
pub(super) fn v1_sse_error(err: &HiggsError) -> String {
    serde_json::to_string(&v1_envelope_for(err)).expect("error envelope serialization cannot fail")
}

/// `/v1` error response for a non-[`HiggsError`] failure (a malformed body, a
/// panicked chat task) — `status` + OpenAI error envelope body. Sibling of
/// [`v1_error`] so the envelope shape lives in exactly one place.
fn v1_envelope_response(status: StatusCode, message: &str) -> (StatusCode, Json<WrappedError>) {
    (status, Json(v1_envelope(status, message)))
}

/// `/v1` 400 response with a custom message (malformed request body). Routed
/// through [`HiggsError::InvalidRequest`] so the body carries the `[HG049]` code
/// + resolution, like every other non-success reply.
fn v1_bad_request(message: &str) -> (StatusCode, Json<WrappedError>) {
    let err = HiggsError::InvalidRequest {
        detail: message.to_owned(),
    };
    v1_envelope_response(StatusCode::BAD_REQUEST, &err.to_string())
}

/// `/v1` 500 response with a custom message (an internal, non-`HiggsError` failure).
fn v1_internal(message: &str) -> (StatusCode, Json<WrappedError>) {
    v1_envelope_response(StatusCode::INTERNAL_SERVER_ERROR, message)
}

/// Emit the "Verbose Logging" serving line for a completed chat. ONE INFO
/// event on the `higgs` target so [`HiggsLogLayer`](crate::log_bus::HiggsLogLayer)
/// mirrors it into the live Developer Logs.
///
/// The log layer captures only the event's `message` field (structured fields
/// are deliberately dropped — no prompt content reaches the logs), so the token
/// count, finish reason, and elapsed ms are baked INTO the message text. Shared
/// by the non-streaming path here and the streaming path in `stream`, so both
/// produce the identical `higgs: served …` line. Called only when
/// [`Higgs::verbose`](crate::api::Higgs::verbose) is true.
pub(super) fn log_served(
    model: &str,
    finish_reason: &str,
    completion_tokens: u32,
    started: std::time::Instant,
) {
    let ms = started.elapsed().as_millis();
    // The whole line is the `message` (the only field the log layer captures).
    tracing::info!(
        "{}",
        served_message(model, finish_reason, completion_tokens, ms)
    );
}

/// The verbose serving line text, e.g.
/// `higgs: served org/model — 12 tok, finish=length, 1234ms`. Built here (pure,
/// no tracing) so the exact Developer-Log wording is unit-testable and the
/// numbers are guaranteed to land in the captured `message`.
fn served_message(model: &str, finish_reason: &str, completion_tokens: u32, ms: u128) -> String {
    format!("higgs: served {model} — {completion_tokens} tok, finish={finish_reason}, {ms}ms")
}

/// Max characters of the incoming prompt preview baked into the "Log Incoming
/// Tokens" line. Caps a single request so one large prompt can't flood the log
/// ring; the full prompt still goes to the model unchanged. 800 chars is a
/// generous one-line preview (most chat turns fit) while bounding the worst case.
const INCOMING_PREVIEW_CHARS: usize = 800;

/// Emit the "Log Incoming Tokens" line for a chat request: ONE INFO event on the
/// `higgs` target carrying the flattened incoming prompt, capped to
/// [`INCOMING_PREVIEW_CHARS`]. The log layer captures only the event `message`,
/// so the prompt is baked INTO the message text (that is why it appears at all).
/// Called only when [`Higgs::log_incoming_tokens`](crate::api::Higgs::log_incoming_tokens)
/// is true — the explicit opt-in that logs prompt CONTENT, overriding the
/// redact-by-default policy. Reuses [`messages_to_pairs`] for flattening; a
/// non-text body (already rejected by the gate) degrades to an empty preview.
fn log_incoming(model: &str, messages: &[ChatCompletionRequestMessage]) {
    tracing::info!("{}", incoming_message(model, messages));
}

/// The incoming-prompt line text, e.g.
/// `higgs: incoming org/model — 42 chars: hello there`. Built here (pure, no
/// tracing) so the wording and the cap are unit-testable. `chars` is the
/// flattened prompt's full char length (pre-cap); `preview` is its first
/// [`INCOMING_PREVIEW_CHARS`] chars with a `…` suffix when truncated.
fn incoming_message(model: &str, messages: &[ChatCompletionRequestMessage]) -> String {
    // Flatten via the same (role, content) extraction the handler validates with;
    // join roles' text with spaces into one previewable line. A rejected non-text
    // body never reaches here (the gate rejects it first) — degrade to empty.
    let flat = messages_to_pairs(messages)
        .map(|pairs| {
            pairs
                .iter()
                .map(|(_, content)| content.as_str())
                .collect::<Vec<_>>()
                .join(" ")
        })
        .unwrap_or_default();
    let chars = flat.chars().count();
    let preview: String = if chars > INCOMING_PREVIEW_CHARS {
        let head: String = flat.chars().take(INCOMING_PREVIEW_CHARS).collect();
        format!("{head}…")
    } else {
        flat
    };
    format!("higgs: incoming {model} — {chars} chars: {preview}")
}

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
///
/// The local node may host several models at once (additive load); each resident
/// instance is one served id (`org/model`, `org/model-1`, …). Nothing loaded
/// means an empty list — the correct OpenAI answer for "no models can serve chat
/// right now" — so this never gates on liveness. `GET /api/higgs/status` still
/// exposes `worker_alive` truthfully for diagnostics.
pub(super) async fn v1_models(State(higgs): State<Arc<Higgs>>) -> Response {
    tracing::info!("higgs: GET /v1/models");
    let mut data: Vec<Model> = higgs
        .local_served_ids()
        .await
        .into_iter()
        .map(|id| Model {
            id,
            object: "model".to_owned(),
            created: now_secs(),
            owned_by: "higgs".to_owned(),
        })
        .collect();
    // Also advertise remote-resident models — they are valid chat targets routed
    // through the fleet (skip any already listed by a local worker).
    if let Some(fleet) = higgs.fleet() {
        for id in fleet.routed_models().await {
            if !data.iter().any(|m| m.id == id) {
                data.push(Model {
                    id,
                    object: "model".to_owned(),
                    created: now_secs(),
                    owned_by: "higgs".to_owned(),
                });
            }
        }
    }
    Json(ListModelResponse {
        object: "list".to_owned(),
        data,
    })
    .into_response()
}

/// `POST /v1/chat/completions`. With JIT on (the default), a request for a
/// scanned-but-unloaded model loads it on demand (swapping out any resident
/// model — higgs serves one at a time) before serving. With JIT off, the named
/// model must already be loaded: unknown and on-disk-but-unloaded ids both get
/// the HG003 404.
pub(super) async fn v1_chat_completions(
    State(higgs): State<Arc<Higgs>>,
    Json(req): Json<CreateChatCompletionRequest>,
) -> Response {
    tracing::info!(model = %req.model, stream = req.stream.unwrap_or(false), "higgs: POST /v1/chat/completions");
    // Start the serve clock now so the optional verbose completion line can
    // report wall-clock elapsed (gate + validation + generation) per request.
    let started = std::time::Instant::now();

    // Serving gate: when serving is toggled off, the /v1 inference surface
    // refuses with [HG019] → 503 (rendered the same way as every other chat-
    // boundary HiggsError). The /api/higgs/* control surface stays reachable so
    // the user can re-enable. Checked before the loaded-model gate so no JIT
    // load or worker RPC runs while serving is off.
    if !higgs.serving_enabled() {
        let err = HiggsError::ServingDisabled;
        tracing::warn!(model = %req.model, "higgs: chat refused — serving disabled");
        return v1_error(&err).into_response();
    }

    // Gate + validation: the named model must be loaded, and sampling / prompt /
    // content checks pass — each maps to its own status. Any failure short-circuits.
    // Returns the resolved (now-resident) model id, which binds the chat dispatch:
    // the worker rejects (HG018) if a concurrent JIT load swaps it out before
    // generation, so a swap errors instead of serving the wrong model.
    let (resolved_model, max_tokens) = match gate_and_validate(&higgs, &req).await {
        Ok(pair) => pair,
        Err(resp) => return resp,
    };

    // Opt-in prompt logging: when "Log Incoming Tokens" is on, emit ONE INFO
    // line carrying the (capped) incoming prompt CONTENT. Fires after the
    // loaded-model + content gate so only valid requests are logged, before
    // dispatch. When off, nothing extra is logged (redact-by-default intact).
    if higgs.log_incoming_tokens() {
        log_incoming(&req.model, &req.messages);
    }

    // Serialize the OpenAI `messages` and `tools` arrays for the chat template,
    // or 400 on a malformed body.
    let messages_json = match serialize_messages(&req) {
        Ok(s) => s,
        Err(resp) => return *resp,
    };
    let tools_json = match serialize_tools(&req) {
        Ok(t) => t,
        Err(resp) => return *resp,
    };

    // `max_tokens` is the context-clamped generation budget from `gate_and_validate`
    // (`min(requested or inferred, n_ctx − prompt, MAX_OUTPUT_TOKENS)`), so dispatch
    // always asks for a budget that fits the loaded window.
    // Per-request sampler set from the OpenAI body (only the fields the client
    // actually sent become `Some`). `chat_stream` overlays this onto the model's
    // tuned/card-recommended base, so a tuned model serves with its recommendation
    // by default while these per-request fields still override.
    let sampling = build_sampling(&req);

    let (deltas, outcome) = match higgs
        .chat_stream(
            resolved_model,
            messages_json,
            max_tokens,
            sampling,
            tools_json,
        )
        .await
    {
        Ok(pair) => pair,
        Err(err) => {
            tracing::warn!(error = %err, "higgs: chat_stream failed to start");
            return v1_error(&err).into_response();
        }
    };

    if req.stream.unwrap_or(false) {
        // Verbose serving line for the streaming path fires once the final
        // outcome is known inside the SSE assembly (see `stream::chat_sse`);
        // pass the gate so it logs only when verbose is on.
        let model = req.model.clone();
        // OpenAI `stream_options: { include_usage: true }` → emit a terminal usage chunk.
        let include_usage = req
            .stream_options
            .as_ref()
            .is_some_and(|o| o.include_usage == Some(true));
        stream::chat_sse(
            chatcmpl_id(),
            model,
            now_secs(),
            deltas,
            outcome,
            higgs.verbose(),
            started,
            include_usage,
        )
        .into_response()
    } else {
        // Non-streaming: ChatOutcome.content is the canonical full text; the
        // delta receiver is dropped (the worker-side sender no-ops once closed).
        drop(deltas);
        match outcome.await {
            Ok(Ok(out)) => {
                if higgs.verbose() {
                    log_served(
                        &req.model,
                        &out.finish_reason,
                        out.completion_tokens,
                        started,
                    );
                }
                Json(chat_response(req.model, &out)).into_response()
            }
            Ok(Err(err)) => {
                tracing::warn!(error = %err, "higgs: chat failed");
                v1_error(&err).into_response()
            }
            // JoinError: the chat task panicked or was aborted — not a HiggsError.
            Err(join_err) => {
                tracing::warn!(error = %join_err, "higgs: chat task failed");
                v1_internal(&format!("chat task failed: {join_err}")).into_response()
            }
        }
    }
}

/// Resolve the model that will serve this chat, loading it on demand when JIT
/// is on. Returns the [`LoadedInfo`] of the now-resident requested model, or an
/// `Err(response)` carrying the mapped failure.
///
/// - Requested model already loaded LOCALLY → returns its [`LoadedInfo`] (no load).
/// - Else remote-resident → permissive [`LoadedInfo`] (the fleet routes it).
/// - Not loaded and JIT OFF → 404 `[HG003]` `ModelNotLoaded` (explicit-load).
/// - Not loaded and JIT ON → JIT path: the id must be a scanned model
///   (`[HG002]` `ModelNotFound` → 404 otherwise — never attempt to load an
///   unknown id), then [`Higgs::load`] loads it with host defaults. The local node
///   is multi-model, so the load is ADDITIVE (a fresh worker, idempotent per model)
///   — it does not swap out other models. `load()` takes the lifecycle mutex,
///   serializing concurrent JIT loads. A load failure (insufficient memory
///   `[HG017]` → 503, bad GGUF, worker spawn failure, …) surfaces as its mapped
///   error — NOT a silent 404.
///
/// LOCAL-first: a locally-served id wins over a remote one of the same name, so a
/// model the user explicitly loaded locally is preferred (and stays consistent with
/// `chat_stream` + `/v1/models`, which are also local-first).
async fn ensure_loaded(higgs: &Arc<Higgs>, model: &str) -> Result<LoadedInfo, Response> {
    // Already locally served (by served id) — serve it, no load. The local node is
    // multi-model: this resolves the SPECIFIC resident instance (`org/model`,
    // `org/model-1`, …), so a chat for an already-loaded model never re-loads.
    if let Some(loaded) = higgs.local_loaded_info(model).await {
        return Ok(loaded);
    }

    // Remote-resident model: the fleet routes the chat to its node, so skip the LOCAL
    // scan/JIT gate (which would 404 it as HG003/HG002). The remote worker enforces the
    // exact prompt-vs-context check (HG005), so we report a permissive `ctx_len` here and
    // defer prompt-fit to it.
    let is_remote = match higgs.fleet() {
        Some(f) => f.is_remote(model).await,
        None => false,
    };
    if is_remote {
        return Ok(LoadedInfo {
            id: model.to_owned(),
            worker_id: 0, // remote-resident placeholder — the fleet routes it, no local worker
            // Remote-resident: the fleet routes it; we have no local probe, so the LIVE
            // params are unknown. `None` keeps the local prompt-fit gate permissive (the
            // remote worker's [HG005] is the backstop).
            ctx_len: None,
            gpu_layers: None,
            threads: None,
            arch: None,
            quant: None,
            max_context_length: None,
            size_bytes: None,
            has_chat_template: None,
            idle_ttl_minutes: None,
        });
    }

    // Not loaded. With JIT off, keep the explicit-load behavior: 404 [HG003].
    if !higgs.jit_enabled() {
        let err = HiggsError::ModelNotLoaded {
            id: model.to_owned(),
        };
        tracing::warn!(model = %model, "higgs: chat for unloaded model (JIT off)");
        return Err(v1_error(&err).into_response());
    }

    // JIT path. The requested id must be a scanned model — never try to load an
    // unknown id (that is a [HG002] 404, not a load attempt). A suffixed served id
    // (`org/model-1`) is never a scanned id, so it can't be JIT-loaded — only an
    // already-resident extra instance is addressable by its suffix.
    let scanned = match higgs.scan().await {
        Ok(models) => models,
        Err(err) => {
            tracing::warn!(error = %err, "higgs: JIT scan failed");
            return Err(v1_error(&err).into_response());
        }
    };
    if !scanned.iter().any(|m| m.id == model) {
        let err = HiggsError::ModelNotFound {
            id: model.to_owned(),
        };
        tracing::warn!(model = %model, "higgs: JIT chat for unknown model");
        return Err(v1_error(&err).into_response());
    }

    // Readiness gate: JIT only loads a model that has been Prepared (a fresh,
    // canonical tuning profile). An un-profiled or stale model is REFUSED with a
    // coded error — never a silent load with dumb defaults (the old behaviour that
    // produced a wrong context window). See `crate::serve::readiness`.
    // Capture the VALIDATED profile from the readiness check so the load below uses
    // exactly what was gated — no second `models.json` read that a concurrent
    // unlink/chmod/retune could turn into a silent default load (gate bypass).
    let profile = match higgs.profile_state(model).await {
        Ok(crate::api::ProfileState::Ready(p)) => p,
        Ok(crate::api::ProfileState::Missing) => {
            let err = HiggsError::NotPrepared {
                id: model.to_owned(),
            };
            tracing::warn!(model = %model, "higgs: JIT chat for un-prepared model");
            return Err(v1_error(&err).into_response());
        }
        Ok(crate::api::ProfileState::Stale) => {
            let err = HiggsError::ProfileStale {
                id: model.to_owned(),
            };
            tracing::warn!(model = %model, "higgs: JIT chat for stale profile");
            return Err(v1_error(&err).into_response());
        }
        // A store-read failure (models.json unreadable) is surfaced as HG040, not
        // masked as `model_not_prepared`.
        Err(e) => {
            tracing::warn!(model = %model, error = %e, "higgs: JIT readiness check failed");
            return Err(v1_error(&e).into_response());
        }
    };

    // Load on demand with the VALIDATED profile, as a no-resync REUSE
    // (`from_request = false`). One always-on INFO line so the load is visible in
    // the Developer Logs. The local load is ADDITIVE — it spawns a new worker
    // alongside any others (and is idempotent per model), no swap.
    tracing::info!("higgs: JIT loading {model}");
    if let Err(err) = higgs.load_inner(model, Some(profile), false).await {
        tracing::warn!(model = %model, error = %err, "higgs: JIT load failed");
        return Err(v1_error(&err).into_response());
    }

    // Re-resolve: the requested model must now be served.
    match higgs.local_loaded_info(model).await {
        Some(loaded) => Ok(loaded),
        None => {
            // Load reported success but the model isn't resident — surface the
            // real not-loaded condition rather than a spurious success.
            let err = HiggsError::ModelNotLoaded {
                id: model.to_owned(),
            };
            tracing::warn!(model = %model, "higgs: JIT load succeeded but model not resident");
            Err(v1_error(&err).into_response())
        }
    }
}

/// Run the pre-dispatch gate and request validation for a chat request:
/// loaded-model check, sampling-param ranges, prompt-vs-context fit, and the
/// v1 text-only content check. Returns `Err(response)` with the first failing
/// gate's mapped response; on success returns the resolved (now-resident) model
/// id, which the caller threads into the chat dispatch so the worker can reject
/// (HG018) a model swapped out by a concurrent JIT load before generation.
///
/// higgs is spawn-on-load. With JIT off, an unloaded or unknown model —
/// including the idle no-worker state and a crashed-worker state — falls through
/// to the HG003 404 `ModelNotLoaded` gate. With JIT on (the default), a
/// scanned-but-unloaded model is loaded on demand here (see [`ensure_loaded`])
/// before validation. There is no separate worker-down 503 on this surface;
/// `GET /api/higgs/status` exposes `worker_alive` for diagnostics instead.
async fn gate_and_validate(
    higgs: &Arc<Higgs>,
    req: &CreateChatCompletionRequest,
) -> Result<(String, usize), Response> {
    // Loaded-model gate (JIT-aware): resolve the model that will serve this
    // request, loading it on demand when JIT is on. Returns the LoadedInfo for
    // the now-resident requested model, or the mapped error response.
    let loaded = ensure_loaded(higgs, &req.model).await?;

    // Validate sampling params (temperature/top_p/n/penalties/max_tokens) BEFORE
    // dispatching to the worker — out-of-range → 400 [HG013]. Ranges mirror vllm.
    if let Err(err) = validate_sampling(req) {
        tracing::warn!(error = %err, "higgs: chat sampling param rejected");
        return Err(v1_error(&err).into_response());
    }

    // Resolve the generation budget against the loaded window: CLAMP `max_tokens` to
    // what fits after the prompt (so a large budget truncates rather than erroring),
    // rejecting (400 `context_length_exceeded`) ONLY when the prompt alone overflows.
    // The worker's tokenizer-exact [HG005] check remains the authoritative backstop.
    let max_gen = match fit_generation_budget(req, loaded.ctx_len) {
        Ok(budget) => budget,
        Err(err) => {
            tracing::warn!(error = %err, "higgs: chat prompt exceeds context window");
            return Err(v1_error(&err).into_response());
        }
    };

    // v1 is text-only: reject image/audio/file/refusal content parts → 400.
    // (Validation only — the flattened pairs are discarded; the engine receives
    // the verbatim OpenAI messages JSON below so tool_calls / tool_call_id are
    // preserved for multi-turn tool loops.)
    if let Err(reject) = messages_to_pairs(&req.messages) {
        tracing::warn!(detail = %reject, "higgs: chat request rejected");
        return Err(v1_bad_request(&reject).into_response());
    }
    // The resolved model id (the gate proved it resident) + the context-clamped
    // generation budget, so the caller dispatches with a budget that always fits.
    Ok((loaded.id, max_gen))
}

/// Serialize the OpenAI `messages` array verbatim for the chat template — the
/// engine feeds it to the GGUF template via the crate's OAI-compat apply, which
/// parses assistant tool_calls and tool tool_call_id natively. `Err(400)` on a
/// malformed body.
fn serialize_messages(req: &CreateChatCompletionRequest) -> Result<String, Box<Response>> {
    serde_json::to_string(&req.messages).map_err(|err| {
        tracing::warn!(error = %err, "higgs: messages serialization failed");
        Box::new(v1_bad_request(&format!("invalid messages: {err}")).into_response())
    })
}

/// Serialize the OpenAI `tools` array (if present) to a JSON string for the chat
/// template. `Err(400)` on a malformed body.
fn serialize_tools(req: &CreateChatCompletionRequest) -> Result<Option<String>, Box<Response>> {
    req.tools
        .as_ref()
        .map(serde_json::to_string)
        .transpose()
        .map_err(|err| {
            tracing::warn!(error = %err, "higgs: chat tools serialization failed");
            Box::new(v1_bad_request(&format!("invalid tools: {err}")).into_response())
        })
}

/// Interpret a worker-produced `tool_calls` value into the async-openai wire
/// type, the SINGLE canonical interpretation shared by the non-streaming and
/// streaming paths so identical worker output yields the same `finish_reason` on
/// both. This data was produced by the crate's OAI-compat parser, so it is
/// already OpenAI-shaped; a value that does not deserialize into the wire type
/// (e.g. a bare string or a malformed `[{}]` element) is an internal bug — the
/// actual serde error is logged and the call degrades to `None` rather than
/// panicking or forcing a spurious `tool_calls` finish reason.
///
/// `None` (no calls) when the value is absent, deserializes to an empty array,
/// or fails to deserialize.
pub(super) fn interpret_tool_calls(
    tool_calls: &Option<serde_json::Value>,
) -> Option<Vec<ChatCompletionMessageToolCalls>> {
    let calls: Vec<ChatCompletionMessageToolCalls> = tool_calls.as_ref().and_then(|v| {
        serde_json::from_value(v.clone())
            .map_err(|err| {
                tracing::error!(error = %err, "higgs: tool_calls deserialize failed");
                err
            })
            .ok()
    })?;
    // An empty array carries no calls — treat it as "no tool calls" so the
    // engine's stop/length signal drives finish_reason.
    if calls.is_empty() {
        None
    } else {
        Some(calls)
    }
}

/// Build the non-streaming chat completion response from a final outcome.
fn chat_response(model: String, out: &ChatOutcome) -> CreateChatCompletionResponse {
    let tool_calls = interpret_tool_calls(&out.tool_calls);
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
                // OpenAI: a tool-call turn carries content: null. Emit None when
                // there are tool calls and no surviving text content.
                content: if tool_calls.is_some() && out.content.is_empty() {
                    None
                } else {
                    Some(out.content.clone())
                },
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
            total_tokens: out.prompt_tokens.saturating_add(out.completion_tokens),
            ..Default::default()
        }),
    }
}

/// Effective generation budget for a request: `max_completion_tokens` wins over
/// the deprecated `max_tokens`; default 1024 when neither is set. Shared by
/// validation, the prompt fit-check, and the dispatch call so all three agree.
#[allow(deprecated)]
fn effective_max_tokens(req: &CreateChatCompletionRequest) -> u32 {
    req.max_completion_tokens.or(req.max_tokens).unwrap_or(1024)
}

/// Validate the chat request's sampling parameters against their accepted
/// ranges; `Err(HiggsError::InvalidSamplingParam)` (→ 400) on the first
/// violation. Ranges mirror vllm `SamplingParams._verify_args`, except `n`:
/// `temperature >= 0`, `top_p` in `(0, 1]`, `presence_penalty` and
/// `frequency_penalty` in `[-2, 2]`, `max_tokens >= 1` and `<= MAX_OUTPUT_TOKENS`.
/// `n` must be exactly `1` — higgs serves a single choice, so n>1 is rejected
/// rather than silently honored as one choice.
fn validate_sampling(req: &CreateChatCompletionRequest) -> Result<(), HiggsError> {
    let invalid = |param: &str, detail: String| HiggsError::InvalidSamplingParam {
        param: param.to_owned(),
        detail,
    };

    if let Some(t) = req.temperature {
        if !t.is_finite() || t < 0.0 {
            return Err(invalid("temperature", "must be a finite value >= 0".into()));
        }
    }
    if let Some(p) = req.top_p {
        if !(p > 0.0 && p <= 1.0) {
            return Err(invalid("top_p", "must be in (0, 1]".into()));
        }
    }
    if let Some(n) = req.n {
        // higgs serves exactly one choice per request — n>1 (multiple choices)
        // is unsupported, and n==0 is out of range. Only n==1 is accepted.
        if n != 1 {
            return Err(invalid(
                "n",
                format!("must be 1 (higgs serves a single choice; n={n} is unsupported)"),
            ));
        }
    }
    if let Some(pp) = req.presence_penalty {
        if !(-2.0..=2.0).contains(&pp) {
            return Err(invalid("presence_penalty", "must be in [-2, 2]".into()));
        }
    }
    if let Some(fp) = req.frequency_penalty {
        if !(-2.0..=2.0).contains(&fp) {
            return Err(invalid("frequency_penalty", "must be in [-2, 2]".into()));
        }
    }
    // max_tokens: vllm requires >= 1; higgs additionally caps the upper bound so
    // a single request can't pin the worker generating an unbounded stream.
    let max_tokens = effective_max_tokens(req);
    if max_tokens == 0 {
        return Err(invalid("max_tokens", "must be at least 1".into()));
    }
    if max_tokens > MAX_OUTPUT_TOKENS {
        return Err(invalid(
            "max_tokens",
            format!("must not exceed {MAX_OUTPUT_TOKENS}"),
        ));
    }
    Ok(())
}

/// Map the OpenAI request's sampling fields onto the engine sampling umbrella —
/// only the fields the client actually sent become `Some`, so `chat_stream` can
/// overlay this onto a model's tuned/card base without clobbering recommended
/// samplers the client didn't mention. OpenAI exposes a narrow subset:
/// `temperature` (→ `temperature`), `top_p` (→ `top_p`), `presence_penalty`
/// (→ `penalty_present`), `frequency_penalty` (→ `penalty_freq`). The ranges were
/// already validated by [`validate_sampling`]. A request that sets none yields an
/// all-`None` set, so the model's base (or the engine's 0.7 default) stands.
fn build_sampling(req: &CreateChatCompletionRequest) -> crate::worker::engine::SamplingParams {
    use crate::worker::engine::llamacpp::params::LlamaCppSamplingParams;
    crate::worker::engine::SamplingParams::llamacpp(LlamaCppSamplingParams {
        temperature: req.temperature,
        top_p: req.top_p,
        penalty_present: req.presence_penalty,
        penalty_freq: req.frequency_penalty,
        ..Default::default()
    })
}

/// Resolve the GENERATION budget for a request against the loaded context window —
/// the largest `max_tokens` that fits after the prompt, INFERRED when the client
/// didn't ask. Returns `Err(ContextOverflow)` (→ 400 `context_length_exceeded`) ONLY
/// when the prompt ALONE cannot fit the window (no room left to generate) — the
/// genuine overflow. When the prompt fits, the budget is CLAMPED to
/// `min(requested or available, available, MAX_OUTPUT_TOKENS)` rather than rejected,
/// so an oversized `max_tokens` truncates (`finish_reason: "length"`) instead of
/// failing the request.
///
/// The serve layer has no tokenizer, so the prompt count is a conservative LOWER
/// bound (`prompt_bytes / PROMPT_BYTES_PER_TOKEN`) — the worker runs the exact
/// tokenizer check. An AUTO/unknown window can't be bounded here, so the requested
/// budget (or the 1024 default) stands, capped at the absolute limit; the worker's
/// `[HG005]` is the backstop.
#[allow(deprecated)]
fn fit_generation_budget(
    req: &CreateChatCompletionRequest,
    ctx_len: Option<CtxLen>,
) -> Result<usize, HiggsError> {
    // Sum the byte length of every message's textual content (the same pairs the
    // worker sees). Serializing the messages would add role/JSON-structure/escape
    // bytes and push the estimate ABOVE the true token count; counting only the
    // content keeps `bytes / PROMPT_BYTES_PER_TOKEN` a conservative LOWER bound.
    let prompt_bytes: usize = messages_to_pairs(&req.messages)
        .map(|pairs| pairs.iter().map(|(_, content)| content.len()).sum())
        .unwrap_or(0);
    let prompt_tokens_est = prompt_bytes / PROMPT_BYTES_PER_TOKEN;
    let requested = req
        .max_completion_tokens
        .or(req.max_tokens)
        .map(|t| t as usize);
    match ctx_len {
        // Not probed (None) or AUTO → can't bound by context here; honor the request
        // (or the 1024 default), capped at the absolute limit. Worker [HG005] backstops.
        None | Some(CtxLen::Auto) => Ok(requested.unwrap_or(1024).min(MAX_OUTPUT_TOKENS as usize)),
        Some(CtxLen::Fixed { n }) => {
            let n_ctx = n as usize;
            // The prompt ALONE doesn't fit → no room to generate → genuine overflow.
            if prompt_tokens_est >= n_ctx {
                return Err(HiggsError::ContextOverflow {
                    prompt_tokens: prompt_tokens_est,
                    max_gen: requested.unwrap_or(0),
                    n_ctx,
                });
            }
            // Room left after the prompt. Infer the budget when omitted; otherwise
            // honor the request but CLAMP to what fits + the absolute cap (≥ 1).
            let available = n_ctx - prompt_tokens_est;
            let budget = requested.unwrap_or(available);
            Ok(budget.min(available).min(MAX_OUTPUT_TOKENS as usize).max(1))
        }
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

#[cfg(test)]
#[path = "v1_tests.rs"]
mod tests;
