//! The OpenAI-compatible `/v1` surface: `GET /v1/models` and
//! `POST /v1/chat/completions`.
//!
//! All bodies are `async-openai` wire types verbatim — nothing hand-rolled.
//! `/v1/models` lists what chat can REACH: loaded models plus servable
//! (JIT-loadable) catalog models. Chat has JIT (just-in-time loading)
//! ON by default: a request for a scanned-but-unloaded model loads it on demand
//! (swapping out any resident model — higgs serves one at a time) before
//! serving; with JIT off, chat against an unloaded model is a 404. Errors render
//! as the OpenAI envelope `{"error":{message,type,code}}`.

use std::sync::Arc;

use async_openai::error::{ApiError, WrappedError};
use async_openai::types::chat::{
    ChatCompletionMessageToolCalls,
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
    ChatCompletionRequestUserMessageContentPart as UserPart, CompletionUsage,
    CreateChatCompletionRequest, FinishReason, Role,
};
use async_openai::types::models::{ListModelResponse, Model};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;

use crate::api::{ChatOutcome, Higgs};
use crate::diagnostic::HiggsError;

use super::http_status;
use super::stream;
use super::MAX_OUTPUT_TOKENS;

/// Build the OpenAI error envelope `{"error":{message,type,code}}` for `/v1`.
///
/// `type` is `invalid_request_error` for 4xx and `server_error` otherwise;
/// `code` is `model_not_found` on 404 (the only coded case on this surface).
///
/// The message is [`redact_paths`]-sanitized: the `/v1` surface is the untrusted
/// OpenAI-interop boundary, so host filesystem paths and bind addresses are
/// stripped from the client-facing string. The full unredacted Display (with
/// paths) is still logged server-side at the origin (four-pillar pillar 1) and
/// returned verbatim on the in-process control surface, which is ours.
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

/// `GET /v1/models` — answers "what can serve chat right now": every LOADED
/// model plus every SERVABLE one (prepared + fits + serving on — JIT loads it
/// on demand). Never the raw on-disk catalog (that is the control `models`
/// route): unprepared/stale/oversized models are excluded because the JIT gate
/// would refuse them.
///
/// The local node may host several models at once (additive load); each resident
/// instance is one served id (`org/model`, `org/model-1`, …). Nothing loaded
/// means an empty list — the correct OpenAI answer for "no models can serve chat
/// right now" — so this never gates on liveness. The `status` control-op still
/// exposes `worker_alive` truthfully for diagnostics.
pub(super) async fn v1_models(State(higgs): State<Arc<Higgs>>) -> Response {
    tracing::info!("higgs: GET /v1/models");
    // The reachable-id UNION (local-served ∪ JIT-servable ∪ fleet-routed, minus a
    // transiently-benchmarking local candidate) lives on the facade
    // (`Higgs::chat_model_ids`); this handler is a thin caller mapping each id to
    // the OpenAI `Model` envelope so the in-process embedder's model list and this
    // endpoint cannot diverge.
    let data: Vec<Model> = higgs
        .chat_model_ids()
        .await
        .into_iter()
        .map(|id| Model {
            id,
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

/// `POST /v1/chat/completions`. With JIT on (the default), a request for a
/// scanned-but-unloaded model loads it on demand (swapping out any resident
/// model — higgs serves one at a time) before serving. With JIT off, the named
/// model must already be loaded: unknown and on-disk-but-unloaded ids both get
/// the HG003 404.
pub(super) async fn v1_chat_completions(
    State(higgs): State<Arc<Higgs>>,
    Json(raw): Json<serde_json::Value>,
) -> Response {
    // Deserialize through the typed struct for VALIDATION (same 4xx semantics,
    // now in the OpenAI error envelope), but KEEP the raw body: fields the
    // async-openai types do not model must survive to the chat template —
    // notably assistant `reasoning_content` echo-back (DeepSeek/Kimi multi-turn
    // convention; genai echoes it) and `chat_template_kwargs` (llama.cpp's
    // per-request template knobs, e.g. `{"enable_thinking": false}`).
    let req: CreateChatCompletionRequest = match serde_json::from_value(raw.clone()) {
        Ok(r) => r,
        Err(err) => {
            tracing::warn!(error = %err, "higgs: chat body failed typed validation");
            return v1_bad_request(&format!("invalid request body: {err}")).into_response();
        }
    };
    tracing::info!(model = %req.model, stream = req.stream.unwrap_or(false), "higgs: POST /v1/chat/completions");
    // Start the serve clock now so the optional verbose completion line can
    // report wall-clock elapsed (gate + validation + generation) per request.
    let started = std::time::Instant::now();

    // Serving gate: when serving is toggled off, the /v1 inference surface
    // refuses with [HG019] → 503 (rendered the same way as every other chat-
    // boundary HiggsError). The in-process control surface stays reachable so
    // the user can re-enable. Checked before the loaded-model gate so no JIT
    // load or worker RPC runs while serving is off.
    if !higgs.serving_enabled() {
        let err = HiggsError::ServingDisabled;
        tracing::warn!(model = %req.model, "higgs: chat refused — serving disabled");
        return v1_error(&err).into_response();
    }

    // The RAW `messages` array rides to the chat template verbatim (typed
    // validation already passed above), so extension fields the typed structs
    // would drop — assistant `reasoning_content` — reach the template render. It
    // is computed BEFORE the gate so the shared facade gate (`prepare_chat`) can
    // estimate the prompt from the same bytes.
    let messages_json = raw_messages_json(&raw);

    // Gate + validation: the named model must be loaded, and sampling / prompt /
    // content checks pass — each maps to its own status. Any failure short-circuits.
    // Returns the resolved (now-resident) model id, which binds the chat dispatch:
    // the worker rejects (HG018) if a concurrent JIT load swaps it out before
    // generation, so a swap errors instead of serving the wrong model.
    let (resolved_model, max_tokens) = match gate_and_validate(&higgs, &req, &messages_json).await {
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

    let tools_json = match serialize_tools(&req) {
        Ok(t) => t,
        Err(resp) => return *resp,
    };
    // llama.cpp-style per-request template knobs (object → JSON string for the
    // template apply). Non-object values are ignored rather than 400 — the
    // typed validation above never sees this key, matching llama.cpp's lenient
    // handling of its own extension field — but leave a coded trace so a
    // client whose kwargs silently do nothing can be debugged from the logs.
    let chat_template_kwargs = match raw.get("chat_template_kwargs") {
        Some(v) if v.is_object() => Some(v.to_string()),
        Some(v) => {
            tracing::warn!(value = %v, "[HG055] chat_template_kwargs is not a JSON object; ignored");
            None
        }
        None => None,
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
            chat_template_kwargs,
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
/// the `status` control-op exposes `worker_alive` for diagnostics instead.
async fn gate_and_validate(
    higgs: &Arc<Higgs>,
    req: &CreateChatCompletionRequest,
    messages_json: &str,
) -> Result<(String, usize), Response> {
    // Shared gate (facade): JIT-load the model, resolve the now-resident served id,
    // and clamp the generation budget to the loaded window. This is the SINGLE
    // source of the load+resolve+clamp logic — the in-process embedder calls the
    // same `Higgs::prepare_chat`, so the two callers cannot diverge. Its
    // `HiggsError` maps through the standard `/v1` error envelope.
    let requested = requested_max_tokens(req);
    let prepared = match higgs
        .prepare_chat(&req.model, requested, messages_json)
        .await
    {
        Ok(prepared) => prepared,
        Err(err) => {
            tracing::warn!(error = %err, "higgs: chat gate failed");
            return Err(v1_error(&err).into_response());
        }
    };

    // Validate sampling params (temperature/top_p/n/penalties/max_tokens) BEFORE
    // dispatching to the worker — out-of-range → 400 [HG013]. Ranges mirror vllm.
    // Kept at the `/v1` boundary: it is v1-wire-shaped (reads the OpenAI request).
    if let Err(err) = validate_sampling(req) {
        tracing::warn!(error = %err, "higgs: chat sampling param rejected");
        return Err(v1_error(&err).into_response());
    }

    // v1 is text-only: reject image/audio/file/refusal content parts → 400. Kept at
    // the boundary: it is a v1-wire content check. (Validation only — the flattened
    // pairs are discarded; the engine receives the verbatim OpenAI messages JSON so
    // tool_calls / tool_call_id are preserved for multi-turn tool loops.)
    if let Err(reject) = messages_to_pairs(&req.messages) {
        tracing::warn!(detail = %reject, "higgs: chat request rejected");
        return Err(v1_bad_request(&reject).into_response());
    }
    // The resolved model id (the gate proved it resident) + the context-clamped
    // generation budget, so the caller dispatches with a budget that always fits.
    Ok((prepared.resolved_model, prepared.max_gen))
}

/// The client's requested generation budget as an `Option<usize>` for the shared
/// facade gate ([`Higgs::prepare_chat`]): `max_completion_tokens` wins over the
/// deprecated `max_tokens`; `None` (neither set) lets the facade infer the budget
/// (fill the fixed window, else the 1024 default) rather than a magic sentinel.
#[allow(deprecated)]
fn requested_max_tokens(req: &CreateChatCompletionRequest) -> Option<usize> {
    req.max_completion_tokens
        .or(req.max_tokens)
        .map(|t| t as usize)
}

/// The RAW `messages` array from the request body, serialized verbatim for the
/// chat template. The typed deserialize already validated the array's shape;
/// passing the raw JSON (not a re-serialization of the typed structs) preserves
/// extension fields async-openai does not model — assistant `reasoning_content`
/// echo-back — which the crate's `common_chat` message parser accepts and uses.
fn raw_messages_json(raw: &serde_json::Value) -> String {
    // `messages` exists: the typed validation above required it.
    raw.get("messages")
        .cloned()
        .unwrap_or(serde_json::Value::Array(vec![]))
        .to_string()
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
///
/// The body is the local [`v1_wire`](super::v1_wire) mirror — byte-parity with
/// async-openai's `CreateChatCompletionResponse` plus the `reasoning_content`
/// message field the crate lacks (absent when the model emitted no thinking).
fn chat_response(model: String, out: &ChatOutcome) -> super::v1_wire::ReasoningChatResponse {
    let tool_calls = interpret_tool_calls(&out.tool_calls);
    // Per the OpenAI spec, a turn that emits tool calls reports finish_reason
    // "tool_calls" regardless of the engine's stop/length signal.
    let finish_reason = if tool_calls.is_some() {
        FinishReason::ToolCalls
    } else {
        stream::finish_reason_from(&out.finish_reason)
    };
    super::v1_wire::ReasoningChatResponse {
        id: chatcmpl_id(),
        choices: vec![super::v1_wire::ReasoningChatChoice {
            index: 0,
            message: super::v1_wire::ReasoningResponseMessage {
                // OpenAI: a tool-call turn carries content: null. Emit None when
                // there are tool calls and no surviving text content.
                content: if tool_calls.is_some() && out.content.is_empty() {
                    None
                } else {
                    Some(out.content.clone())
                },
                refusal: None,
                tool_calls,
                role: Role::Assistant,
                reasoning_content: out.reasoning_content.clone(),
            },
            finish_reason: Some(finish_reason),
        }],
        created: now_secs(),
        model,
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
