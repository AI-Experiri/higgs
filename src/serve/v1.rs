//! The OpenAI-compatible `/v1` surface: `GET /v1/models` and
//! `POST /v1/chat/completions`.
//!
//! All bodies are `async-openai` wire types verbatim — nothing hand-rolled.
//! `/v1/models` lists LOADED models only (no JIT in v1: chat against an
//! unloaded model is a 404). Errors render as the OpenAI envelope
//! `{"error":{message,type,code}}`.

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

use crate::api::{ChatOutcome, Higgs};
use crate::diagnostic::HiggsError;

use super::http_status;
use super::stream;

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
pub(super) fn v1_envelope_json(status: StatusCode, message: String) -> String {
    serde_json::to_string(&v1_envelope(status, message))
        .expect("error envelope serialization cannot fail")
}

/// `/v1` error response: mapped status + OpenAI error envelope body.
fn v1_error(err: &HiggsError) -> (StatusCode, Json<WrappedError>) {
    let status = http_status(err);
    (status, Json(v1_envelope(status, err.to_string())))
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
pub(super) async fn v1_models(State(higgs): State<Arc<Higgs>>) -> Response {
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
pub(super) async fn v1_chat_completions(
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rpc::{encode, RpcFrame, RpcNotification};
    use crate::worker::N_CHAT_CHUNK;
    use async_openai::types::chat::CreateChatCompletionStreamResponse;
    use serde_json::json;
    use std::time::Duration;
    use tokio::io::AsyncWriteExt;
    use tower::ServiceExt;

    use super::super::test_support::*;

    /// Deliver a streaming chat-chunk notification keyed to `request_id`.
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

    fn parse_messages(v: serde_json::Value) -> Vec<ChatCompletionRequestMessage> {
        serde_json::from_value(v).expect("messages deserialize")
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
