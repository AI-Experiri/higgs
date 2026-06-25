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

/// Serialize the `/v1` error envelope to a JSON string (SSE `data:` payloads).
pub(super) fn v1_envelope_json(status: StatusCode, message: &str) -> String {
    serde_json::to_string(&v1_envelope(status, message))
        .expect("error envelope serialization cannot fail")
}

/// `/v1` error response: mapped status + OpenAI error envelope body.
fn v1_error(err: &HiggsError) -> (StatusCode, Json<WrappedError>) {
    let status = http_status(err);
    (status, Json(v1_envelope(status, &err.to_string())))
}

/// `/v1` error response for a non-[`HiggsError`] failure (a malformed body, a
/// panicked chat task) — `status` + OpenAI error envelope body. Sibling of
/// [`v1_error`] so the envelope shape lives in exactly one place.
fn v1_envelope_response(status: StatusCode, message: &str) -> (StatusCode, Json<WrappedError>) {
    (status, Json(v1_envelope(status, message)))
}

/// `/v1` 400 response with a custom message (malformed request body).
fn v1_bad_request(message: &str) -> (StatusCode, Json<WrappedError>) {
    v1_envelope_response(StatusCode::BAD_REQUEST, message)
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
    let resolved_model = match gate_and_validate(&higgs, &req).await {
        Ok(id) => id,
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

    // Effective generation budget (validated above to be in [1, MAX_OUTPUT_TOKENS]).
    let max_tokens = effective_max_tokens(&req) as usize;
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
            ctx_len: u32::MAX,
            gpu_layers: 0,
            threads: 0,
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

    // Load on demand (host defaults). One always-on INFO line so the load is
    // visible in the Developer Logs. The local load is ADDITIVE — it spawns a new
    // worker alongside any others (and is idempotent per model), no swap.
    tracing::info!("higgs: JIT loading {model}");
    if let Err(err) = higgs.load(model, None).await {
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
) -> Result<String, Response> {
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

    // Early prompt-vs-context reject: a prompt whose conservative token estimate
    // plus the generation budget cannot fit the loaded window is a client error
    // (400 [HG005]) — reject here instead of shipping it to the worker. The
    // worker's tokenizer-exact [HG005] check remains the authoritative backstop.
    if let Err(err) = check_prompt_fits(req, loaded.ctx_len) {
        tracing::warn!(error = %err, "higgs: chat prompt exceeds context window");
        return Err(v1_error(&err).into_response());
    }

    // v1 is text-only: reject image/audio/file/refusal content parts → 400.
    // (Validation only — the flattened pairs are discarded; the engine receives
    // the verbatim OpenAI messages JSON below so tool_calls / tool_call_id are
    // preserved for multi-turn tool loops.)
    if let Err(reject) = messages_to_pairs(&req.messages) {
        tracing::warn!(detail = %reject, "higgs: chat request rejected");
        return Err(v1_bad_request(&reject).into_response());
    }
    // The resolved model id (its `.id` is the model the gate proved resident),
    // returned so the caller binds the chat to it (worker HG018 check).
    Ok(loaded.id)
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

/// Cheap early reject when the prompt's conservative token estimate plus the
/// generation budget cannot fit the loaded context window — `Err(ContextOverflow)`
/// (→ 400). The serve layer has no tokenizer, so the estimate is a LOWER bound
/// (`prompt_bytes / PROMPT_BYTES_PER_TOKEN`); the worker runs the exact
/// tokenizer check (the authoritative `[HG005]`). This only fires when the
/// prompt is unambiguously over the window.
fn check_prompt_fits(req: &CreateChatCompletionRequest, ctx_len: u32) -> Result<(), HiggsError> {
    // Sum the byte length of every message's textual content. Serializing the
    // messages would add role/JSON-structure/escape bytes and push the estimate
    // ABOVE the true token count — turning this lower-bound precheck into a
    // false-reject. Counting only the extracted content text (the same pairs the
    // worker sees) keeps `bytes / PROMPT_BYTES_PER_TOKEN` a conservative lower
    // bound. A non-text part here just means 0 bytes; the worker still runs the
    // authoritative tokenizer check.
    let prompt_bytes: usize = messages_to_pairs(&req.messages)
        .map(|pairs| pairs.iter().map(|(_, content)| content.len()).sum())
        .unwrap_or(0);
    let prompt_tokens_est = prompt_bytes / PROMPT_BYTES_PER_TOKEN;
    let max_gen = effective_max_tokens(req) as usize;
    let n_ctx = ctx_len as usize;
    if prompt_tokens_est + max_gen > n_ctx {
        return Err(HiggsError::ContextOverflow {
            prompt_tokens: prompt_tokens_est,
            max_gen,
            n_ctx,
        });
    }
    Ok(())
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
mod tests {
    use super::*;
    use async_openai::types::chat::CreateChatCompletionStreamResponse;
    use serde_json::json;
    use tower::ServiceExt;

    use super::super::test_support::*;

    fn parse_messages(v: serde_json::Value) -> Vec<ChatCompletionRequestMessage> {
        serde_json::from_value(v).expect("messages deserialize")
    }

    // ── served_message: verbose serving line wording ────────────────────────

    #[test]
    fn served_message_format() {
        // The numbers must land in the message text (the log layer captures
        // only `message`), so they're greppable in the Developer Logs.
        assert_eq!(
            served_message("org/model", "length", 12, 1234),
            "higgs: served org/model — 12 tok, finish=length, 1234ms"
        );
        assert_eq!(
            served_message("ollama/llama3:8b", "stop", 0, 7),
            "higgs: served ollama/llama3:8b — 0 tok, finish=stop, 7ms"
        );
    }

    // ── incoming_message: log-incoming-tokens line wording + cap ─────────────

    #[test]
    fn incoming_message_format_and_cap() {
        // The flattened prompt content lands in the message text (the log layer
        // captures only `message`), so it's greppable in the Developer Logs.
        let msgs = parse_messages(json!([
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "hello there"}
        ]));
        assert_eq!(
            incoming_message("org/model", &msgs),
            "higgs: incoming org/model — 20 chars: be brief hello there"
        );

        // A prompt longer than the cap is truncated to INCOMING_PREVIEW_CHARS
        // with a `…` suffix, while `chars` reports the full pre-cap length so one
        // request can't flood the log ring.
        let big = "x".repeat(INCOMING_PREVIEW_CHARS + 50);
        let msgs = parse_messages(json!([{"role": "user", "content": big}]));
        let line = incoming_message("org/model", &msgs);
        assert!(
            line.starts_with(&format!(
                "higgs: incoming org/model — {} chars: ",
                INCOMING_PREVIEW_CHARS + 50
            )),
            "full char count reported: {line}"
        );
        assert!(line.ends_with('…'), "truncation marker present: {line}");
        let preview = line.rsplit("chars: ").next().unwrap();
        // Capped preview: INCOMING_PREVIEW_CHARS chars + the `…` suffix.
        assert_eq!(
            preview.chars().count(),
            INCOMING_PREVIEW_CHARS + 1,
            "preview capped to INCOMING_PREVIEW_CHARS + ellipsis"
        );
    }

    // ── redact_paths: client-facing /v1 error sanitization ───────────────────

    #[test]
    fn redact_paths_strips_paths_and_addresses() {
        // Unix absolute path is redacted; the diagnostic code and prose survive.
        let r = redact_paths("[HG001] model dir unreadable: /Users/me/.lmstudio/models: oops");
        assert!(!r.contains("/Users/me"), "host path leaked: {r}");
        assert!(r.contains("[HG001]"), "code preserved: {r}");
        assert!(r.contains("<redacted>"), "redaction marker present: {r}");

        // host:port bind address is redacted.
        let r = redact_paths("bind failed on 127.0.0.1:8081");
        assert!(!r.contains("127.0.0.1:8081"), "bind address leaked: {r}");

        // Windows drive path is redacted.
        let r = redact_paths(r"engine load failed C:\Users\m\model.gguf bad");
        assert!(!r.contains(r"C:\Users"), "windows path leaked: {r}");
    }

    #[test]
    fn redact_paths_preserves_ids_and_prose() {
        // A model id (no leading slash) and ordinary words are NOT redacted.
        let r = redact_paths("[HG003] model not loaded: org/model — load it first");
        assert_eq!(
            r, "[HG003] model not loaded: org/model — load it first",
            "ids and prose must pass through unchanged"
        );
    }

    #[test]
    fn v1_envelope_redacts_message() {
        // End-to-end: a path-carrying message is sanitized in the built envelope.
        let env = v1_envelope(
            StatusCode::INTERNAL_SERVER_ERROR,
            "[HG004] engine failed to load org/model: /opt/models/x.gguf mmap error",
        );
        assert!(
            !env.error.message.contains("/opt/models"),
            "envelope leaked host path: {}",
            env.error.message
        );
        assert!(env.error.message.contains("[HG004]"));
    }

    // ── Test 1: /v1/models empty when nothing is loaded ──────────────────────

    #[tokio::test]
    async fn v1_models_empty_when_unloaded() {
        // Nothing loaded (no resident worker) → empty served set → empty list.
        let app = make_app();
        let resp = app.oneshot(get("/v1/models")).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let list: ListModelResponse = serde_json::from_slice(&body_bytes(resp).await).unwrap();
        assert_eq!(list.object, "list");
        assert!(list.data.is_empty());
    }

    // ── Test 2: /v1/models lists the loaded model ─────────────────────────────

    #[tokio::test]
    async fn v1_models_lists_loaded() {
        // Load a fixture model so it becomes a served instance, then list it.
        let dir = tempfile::TempDir::new().unwrap();
        write_gguf_fixture(dir.path(), "org/model");
        let higgs = make_higgs_with_lmstudio(dir.path().to_path_buf());
        higgs.load("org/model", None).await.expect("load");
        let app = app_for(higgs);

        let resp = app.oneshot(get("/v1/models")).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let list: ListModelResponse = serde_json::from_slice(&body_bytes(resp).await).unwrap();
        assert_eq!(list.data.len(), 1);
        assert_eq!(list.data[0].id, "org/model");
        assert_eq!(list.data[0].object, "model");
        assert_eq!(list.data[0].owned_by, "higgs");
    }

    // ── Test 3: JIT OFF + chat against an unloaded model → 404 HG003 envelope ─

    #[tokio::test]
    async fn chat_unloaded_404_hg003() {
        // JIT off: an unloaded model is the explicit-load HG003 404 (not a JIT load).
        let app = make_app_jit_off();

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

    // ── Serving disabled: /v1 chat → 503 HG019 (control surface stays up) ────
    //
    // With serving toggled off, a chat request is refused at the boundary with
    // HG019 → 503 BEFORE any loaded-model gate or worker RPC. Mirrors the
    // unloaded-model boundary test but asserts the serving gate fires first (no
    // M_STATUS is sent — the idle supervisor would fail it, but the gate returns
    // before reaching it).

    #[tokio::test]
    async fn chat_serving_disabled_503_hg019() {
        let app = make_app_serving_off();
        let req = post_json(
            "/v1/chat/completions",
            &json!({"model": "org/model", "messages": [{"role": "user", "content": "hi"}]}),
        );
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::SERVICE_UNAVAILABLE,
            "serving off → 503, not 404/500"
        );
        let body = String::from_utf8(body_bytes(resp).await).unwrap();
        assert!(body.contains("[HG019]"), "carries HG019: {body}");
        assert!(
            body.contains("server_error"),
            "503 envelope type is server_error: {body}"
        );
    }

    // ── C1: idle (no worker) — /v1/models 200 empty, /v1/chat 404 ────────────
    //
    // higgs is spawn-on-load: the normal idle state has NO worker, so
    // status().worker_alive is false. /v1/models must still answer 200 with an
    // empty list (the correct OpenAI answer for "nothing loaded"), and chat for
    // any model must be a 404 HG003 — NOT a 503. The idle supervisor never
    // spawns, so its M_STATUS RPC fails and status() reports worker_alive:false
    // with loaded:None, no worker I/O involved.

    #[tokio::test]
    async fn v1_models_idle_no_worker_200_empty() {
        let app = make_app();
        let resp = app.oneshot(get("/v1/models")).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "idle /v1/models is 200, not 503"
        );
        let list: ListModelResponse = serde_json::from_slice(&body_bytes(resp).await).unwrap();
        assert_eq!(list.object, "list");
        assert!(list.data.is_empty(), "idle list is empty");
    }

    #[tokio::test]
    async fn v1_chat_idle_no_worker_404_hg003() {
        // JIT off: idle chat is the explicit-load HG003 404, not a JIT attempt.
        let app = make_app_jit_off();
        let req = post_json(
            "/v1/chat/completions",
            &json!({"model": "org/model", "messages": [{"role": "user", "content": "hi"}]}),
        );
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::NOT_FOUND,
            "idle chat is 404 model-not-loaded, not 503"
        );
        let body = String::from_utf8(body_bytes(resp).await).unwrap();
        assert!(body.contains("[HG003]"), "carries HG003: {body}");
        assert!(body.contains("model_not_found"), "envelope code: {body}");
    }

    // ── JIT ON + unknown model → 404 HG002 (no load attempt) ─────────────────
    //
    // JIT is on by default. A chat for an id absent from the scan must NOT
    // attempt a load — it is a ModelNotFound 404 [HG002]. The idle supervisor
    // never spawns and the default config dirs hold no fixture, so the scan is
    // empty and the id is unknown.

    #[tokio::test]
    async fn v1_chat_jit_on_unknown_model_404_hg002() {
        // Default app → JIT on. Empty temp config so the scan finds nothing.
        let dir = tempfile::TempDir::new().unwrap();
        let app = make_app_with_lmstudio(dir.path().to_path_buf());
        let req = post_json(
            "/v1/chat/completions",
            &json!({"model": "org/unknown", "messages": [{"role": "user", "content": "hi"}]}),
        );
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::NOT_FOUND,
            "JIT + unknown id is 404 model-not-found"
        );
        let body = String::from_utf8(body_bytes(resp).await).unwrap();
        assert!(
            body.contains("[HG002]"),
            "carries HG002 (not HG003): {body}"
        );
        assert!(body.contains("model_not_found"), "envelope code: {body}");
    }

    // ── JIT ON + scanned-but-unloaded model → loads on demand, then serves ───
    //
    // JIT is on by default. A chat for a scanned model that isn't currently
    // loaded must trigger a load (M_LOAD) and then serve. The mock RPC sequence
    // proves the JIT load path is taken: M_STATUS (nothing loaded) → M_LOAD (the
    // JIT swap) → M_STATUS (now resident) → M_CHAT. With only-keep-last, after
    // the M_LOAD the requested model is the resident one.

    #[tokio::test]
    async fn v1_chat_jit_on_scanned_loads_then_serves() {
        // JIT on (default). Host-side scan discovers `org/model` but nothing is
        // loaded, so the chat triggers a JIT load (the stateful fake worker spawns
        // and records it), then serves. The model is NOT pre-loaded — reaching 200
        // with the completion proves the JIT load path ran.
        let dir = tempfile::TempDir::new().unwrap();
        write_gguf_fixture(dir.path(), "org/model");
        let app = make_app_with_lmstudio(dir.path().to_path_buf());

        let req = post_json(
            "/v1/chat/completions",
            &json!({"model": "org/model", "messages": [{"role": "user", "content": "hi"}]}),
        );
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "JIT load then serve returns 200"
        );
        let chat: CreateChatCompletionResponse =
            serde_json::from_slice(&body_bytes(resp).await).unwrap();
        assert_eq!(chat.model, "org/model");
        assert_eq!(chat.choices[0].message.content.as_deref(), Some("hello"));
    }

    // ── Test 4: non-streaming chat returns ChatOutcome.content ───────────────

    #[tokio::test]
    async fn chat_nonstream_returns_content() {
        // Pre-load a fixture model, then chat it. The stateful fake worker returns
        // `content:"hello", finish:"stop", prompt_tokens:10, completion_tokens:3`.
        let dir = tempfile::TempDir::new().unwrap();
        write_gguf_fixture(dir.path(), "org/model");
        let higgs = make_higgs_with_lmstudio(dir.path().to_path_buf());
        higgs.load("org/model", None).await.expect("load");
        let app = app_for(higgs);

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
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 3);
        assert_eq!(usage.total_tokens, 13);
    }

    // ── Test 5: streaming chat SSE framing ───────────────────────────────────

    #[tokio::test]
    async fn chat_stream_sse_framing() {
        // Pre-load a fixture model, then stream a chat. The stateful fake worker
        // streams two deltas `he`/`llo` then a final `hello`/stop response, so the
        // SSE assembly emits: role + 2 deltas + finish + [DONE].
        let dir = tempfile::TempDir::new().unwrap();
        write_gguf_fixture(dir.path(), "org/model");
        let higgs = make_higgs_with_lmstudio(dir.path().to_path_buf());
        higgs.load("org/model", None).await.expect("load");
        let app = app_for(higgs);

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
            Some("he")
        );
        assert_eq!(
            parse(datas[2]).choices[0].delta.content.as_deref(),
            Some("llo")
        );
        let finish = parse(datas[3]);
        assert_eq!(finish.choices[0].finish_reason, Some(FinishReason::Stop));
        assert_eq!(datas[4], "[DONE]");
    }

    // ── messages_to_pairs: pure role-flattening ──────────────────────────────
    //
    // Parse OpenAI request messages from JSON (their canonical wire form) so the
    // async-openai untagged content enums deserialize exactly as a real request
    // would, then assert the flattened (role, text) pairs.

    #[test]
    fn messages_to_pairs_role_interleaving_and_multiturn() {
        let msgs = parse_messages(json!([
            {"role": "system", "content": "be terse"},
            {"role": "developer", "content": "dev note"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "answer one"},
            {"role": "user", "content": "second"},
            {"role": "tool", "content": "tool out", "tool_call_id": "call_1"}
        ]));
        let pairs = messages_to_pairs(&msgs).expect("all text → Ok");
        assert_eq!(
            pairs,
            vec![
                ("system".to_owned(), "be terse".to_owned()),
                ("developer".to_owned(), "dev note".to_owned()),
                ("user".to_owned(), "first".to_owned()),
                ("assistant".to_owned(), "answer one".to_owned()),
                ("user".to_owned(), "second".to_owned()),
                ("tool".to_owned(), "tool out".to_owned()),
            ],
            "order and roles preserved across the multi-turn conversation"
        );
    }

    #[test]
    fn messages_to_pairs_text_part_arrays_joined_with_newline() {
        // Every role whose content can be a text-part array: system, developer,
        // user, assistant, tool. Parts join with `\n` (shimmy convention).
        let msgs = parse_messages(json!([
            {"role": "system", "content": [
                {"type": "text", "text": "a"}, {"type": "text", "text": "b"}
            ]},
            {"role": "developer", "content": [
                {"type": "text", "text": "c"}, {"type": "text", "text": "d"}
            ]},
            {"role": "user", "content": [
                {"type": "text", "text": "e"}, {"type": "text", "text": "f"}
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": "g"}, {"type": "text", "text": "h"}
            ]},
            {"role": "tool", "tool_call_id": "t1", "content": [
                {"type": "text", "text": "i"}, {"type": "text", "text": "j"}
            ]}
        ]));
        let pairs = messages_to_pairs(&msgs).expect("text parts → Ok");
        assert_eq!(pairs[0], ("system".to_owned(), "a\nb".to_owned()));
        assert_eq!(pairs[1], ("developer".to_owned(), "c\nd".to_owned()));
        assert_eq!(pairs[2], ("user".to_owned(), "e\nf".to_owned()));
        assert_eq!(pairs[3], ("assistant".to_owned(), "g\nh".to_owned()));
        assert_eq!(pairs[4], ("tool".to_owned(), "i\nj".to_owned()));
    }

    #[test]
    fn messages_to_pairs_assistant_none_content_is_empty_string() {
        // An assistant turn with only tool_calls (no content) flattens to "".
        let msgs = parse_messages(json!([
            {"role": "assistant", "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "f", "arguments": "{}"}}
            ]}
        ]));
        let pairs = messages_to_pairs(&msgs).expect("None content → Ok");
        assert_eq!(pairs, vec![("assistant".to_owned(), String::new())]);
    }

    #[test]
    fn messages_to_pairs_rejects_user_image_part() {
        let msgs = parse_messages(json!([
            {"role": "user", "content": [
                {"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": "http://x/y.png"}}
            ]}
        ]));
        let err = messages_to_pairs(&msgs).expect_err("image part → Err");
        assert!(
            err.contains("non-text content part"),
            "rejection message: {err}"
        );
    }

    #[test]
    fn messages_to_pairs_rejects_assistant_refusal_part() {
        let msgs = parse_messages(json!([
            {"role": "assistant", "content": [
                {"type": "refusal", "refusal": "I cannot help with that"}
            ]}
        ]));
        let err = messages_to_pairs(&msgs).expect_err("refusal part → Err");
        assert!(err.contains("refusal content part"), "message: {err}");
    }

    // ── validate_sampling / check_prompt_fits: pure request validation ────────

    fn req(v: serde_json::Value) -> CreateChatCompletionRequest {
        serde_json::from_value(v).expect("request deserialize")
    }

    #[test]
    fn build_sampling_maps_openai_subset_and_omits_absent() {
        // Absent sampling → an all-None set (the model base / engine default stands).
        let bare = build_sampling(&req(json!({
            "model": "m", "messages": [{"role": "user", "content": "hi"}]
        })));
        let s = bare.as_llamacpp();
        assert!(
            s.temperature.is_none()
                && s.top_p.is_none()
                && s.penalty_present.is_none()
                && s.penalty_freq.is_none(),
            "absent OpenAI fields → None: {s:?}"
        );
        // Present fields map across (presence→present, frequency→freq).
        let full = build_sampling(&req(json!({
            "model": "m", "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.3, "top_p": 0.9,
            "presence_penalty": 1.0, "frequency_penalty": -0.5,
        })));
        let s = full.as_llamacpp();
        assert_eq!(s.temperature, Some(0.3));
        assert_eq!(s.top_p, Some(0.9));
        assert_eq!(s.penalty_present, Some(1.0), "presence_penalty → present");
        assert_eq!(s.penalty_freq, Some(-0.5), "frequency_penalty → freq");
        // Fields outside the OpenAI subset are never invented.
        assert!(s.top_k.is_none() && s.min_p.is_none());
    }

    #[test]
    fn validate_sampling_accepts_in_range_and_absent() {
        // All absent → defaults are valid.
        validate_sampling(&req(json!({
            "model": "m", "messages": [{"role": "user", "content": "hi"}]
        })))
        .expect("absent params valid");
        // In-range values valid (temperature 0 and 2, top_p 1, penalties at bounds).
        validate_sampling(&req(json!({
            "model": "m", "messages": [{"role": "user", "content": "hi"}],
            "temperature": 2.0, "top_p": 1.0, "n": 1,
            "presence_penalty": -2.0, "frequency_penalty": 2.0,
            "max_tokens": 256
        })))
        .expect("in-range params valid");
    }

    #[test]
    fn validate_sampling_rejects_each_out_of_range() {
        let base = |extra: serde_json::Value| {
            let mut m = json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]});
            let obj = m.as_object_mut().unwrap();
            for (k, v) in extra.as_object().unwrap() {
                obj.insert(k.clone(), v.clone());
            }
            m
        };
        for (extra, who) in [
            (json!({"temperature": -0.1}), "temperature"),
            (json!({"top_p": 0.0}), "top_p"),
            (json!({"top_p": 1.5}), "top_p"),
            (json!({"n": 0}), "n"),
            // C2: higgs serves a single choice, so n>1 is rejected (not honored
            // as one choice). Both n==0 and n==2 flag the `n` param.
            (json!({"n": 2}), "n"),
            (json!({"presence_penalty": 2.1}), "presence_penalty"),
            (json!({"frequency_penalty": -2.1}), "frequency_penalty"),
            (json!({"max_tokens": 0}), "max_tokens"),
            (json!({"max_tokens": MAX_OUTPUT_TOKENS + 1}), "max_tokens"),
        ] {
            let err = validate_sampling(&req(base(extra))).expect_err("must reject");
            let HiggsError::InvalidSamplingParam { param, .. } = &err else {
                panic!("expected InvalidSamplingParam, got {err}");
            };
            assert_eq!(param, who, "wrong param flagged for {who}");
            assert!(err.to_string().starts_with("[HG013]"));
        }
    }

    #[test]
    fn effective_max_tokens_precedence() {
        // max_completion_tokens wins over max_tokens; default 1024 when absent.
        assert_eq!(
            effective_max_tokens(&req(json!({
                "model": "m", "messages": [], "max_tokens": 10, "max_completion_tokens": 20
            }))),
            20
        );
        assert_eq!(
            effective_max_tokens(&req(json!({
                "model": "m", "messages": [], "max_tokens": 10
            }))),
            10
        );
        assert_eq!(
            effective_max_tokens(&req(json!({"model": "m", "messages": []}))),
            1024
        );
    }

    #[test]
    fn check_prompt_fits_rejects_oversized_prompt() {
        // A tiny window with a long prompt overflows the conservative estimate.
        let long = "x".repeat(8000); // ~2000 estimated tokens at 4 bytes/token.
        let err = check_prompt_fits(
            &req(json!({
                "model": "m",
                "messages": [{"role": "user", "content": long}],
                "max_tokens": 16
            })),
            128,
        )
        .expect_err("oversized prompt must overflow");
        assert!(matches!(err, HiggsError::ContextOverflow { .. }));
        assert!(err.to_string().starts_with("[HG005]"));
    }

    #[test]
    fn check_prompt_fits_accepts_small_prompt() {
        check_prompt_fits(
            &req(json!({
                "model": "m",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 16
            })),
            4096,
        )
        .expect("small prompt fits a large window");
    }

    // ── chat_response: pure ChatOutcome → response mapping ────────────────────

    #[test]
    fn chat_response_without_tool_calls() {
        let out = ChatOutcome {
            content: "hi there".into(),
            finish_reason: "stop".into(),
            tool_calls: None,
            prompt_tokens: 10,
            completion_tokens: 7,
        };
        let resp = chat_response("org/model".into(), &out);
        assert_eq!(resp.model, "org/model");
        assert_eq!(resp.object, "chat.completion");
        let choice = &resp.choices[0];
        assert_eq!(choice.message.content.as_deref(), Some("hi there"));
        assert!(choice.message.tool_calls.is_none());
        assert_eq!(choice.finish_reason, Some(FinishReason::Stop));
        assert_eq!(choice.message.role, Role::Assistant);
        let usage = resp.usage.expect("usage present");
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 7);
        assert_eq!(usage.total_tokens, 17);
    }

    #[test]
    fn chat_response_length_finish_reason_passthrough() {
        let out = ChatOutcome {
            content: "truncated".into(),
            finish_reason: "length".into(),
            tool_calls: None,
            prompt_tokens: 3,
            completion_tokens: 1024,
        };
        let resp = chat_response("org/model".into(), &out);
        assert_eq!(
            resp.choices[0].finish_reason,
            Some(FinishReason::Length),
            "engine length signal passes through when no tool calls"
        );
    }

    #[test]
    fn chat_response_with_tool_calls_forces_finish_reason() {
        let out = ChatOutcome {
            content: String::new(),
            // Engine says "stop" but tool_calls must force "tool_calls".
            finish_reason: "stop".into(),
            tool_calls: Some(json!([{
                "id": "call_99",
                "type": "function",
                "function": {"name": "search", "arguments": "{\"q\":\"rust\"}"}
            }])),
            prompt_tokens: 20,
            completion_tokens: 4,
        };
        let resp = chat_response("org/model".into(), &out);
        let choice = &resp.choices[0];
        assert_eq!(
            choice.finish_reason,
            Some(FinishReason::ToolCalls),
            "tool_calls overrides the engine stop signal"
        );
        let calls = choice
            .message
            .tool_calls
            .as_ref()
            .expect("tool_calls populated");
        assert_eq!(calls.len(), 1);
        let ChatCompletionMessageToolCalls::Function(call) = &calls[0] else {
            panic!("expected a function tool call, got {:?}", calls[0]);
        };
        assert_eq!(call.id, "call_99");
        assert_eq!(call.function.name, "search");
        assert_eq!(call.function.arguments, "{\"q\":\"rust\"}");
        // OpenAI: a pure tool-call turn (empty content) carries content: null.
        assert!(
            choice.message.content.is_none(),
            "tool-call turn with empty content must emit null content, not empty string"
        );
        let usage = resp.usage.expect("usage present");
        assert_eq!(usage.total_tokens, 24);
    }

    #[test]
    fn chat_response_malformed_tool_calls_degrade_to_none() {
        // tool_calls that don't deserialize into the async-openai wire type must
        // degrade to no tool calls (logged internally) rather than panic; the
        // finish reason then comes from the engine signal.
        let out = ChatOutcome {
            content: "text".into(),
            finish_reason: "stop".into(),
            // A bare string is not a tool_calls array.
            tool_calls: Some(json!("not a tool call array")),
            prompt_tokens: 1,
            completion_tokens: 1,
        };
        let resp = chat_response("org/model".into(), &out);
        assert!(
            resp.choices[0].message.tool_calls.is_none(),
            "malformed tool_calls degrade to None"
        );
        assert_eq!(resp.choices[0].finish_reason, Some(FinishReason::Stop));
    }
}
