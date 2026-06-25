//! SSE assembly for `/v1/chat/completions` streaming responses.
//!
//! Bridges the [`Higgs::chat_stream`](crate::api::Higgs::chat_stream) pair
//! (delta receiver + outcome handle) onto the OpenAI chunk protocol:
//! assistant-role chunk → content delta chunks → finish chunk → `data: [DONE]`.
//! Assembly is channel-based (shimmy / local.ai pattern): a spawned task pushes
//! ready `data:` payload strings into an mpsc channel the SSE body drains.

use std::convert::Infallible;

use async_openai::types::chat::{
    ChatChoiceStream, ChatCompletionMessageToolCallChunk, ChatCompletionMessageToolCalls,
    ChatCompletionStreamResponseDelta, CompletionUsage, CreateChatCompletionStreamResponse,
    FinishReason, FunctionCallStream, FunctionType, Role,
};
use axum::response::sse::{Event, KeepAlive, Sse};
use futures::Stream;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use crate::api::ChatOutcome;
use crate::diagnostic::HiggsError;

/// Build the SSE response for one streaming chat completion.
///
/// `deltas` carries worker content chunks; `outcome` resolves when generation
/// completes. Each emitted payload becomes one `data:` line; the stream closes
/// after the trailing `data: [DONE]`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn chat_sse(
    id: String,
    model: String,
    created: u32,
    deltas: mpsc::UnboundedReceiver<String>,
    outcome: JoinHandle<Result<ChatOutcome, HiggsError>>,
    verbose: bool,
    started: std::time::Instant,
    include_usage: bool,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let (tx, rx) = mpsc::unbounded_channel::<String>();
    tokio::spawn(assemble(
        id,
        model,
        created,
        deltas,
        outcome,
        tx,
        verbose,
        started,
        include_usage,
    ));
    let stream = futures::stream::unfold(rx, |mut rx| async move {
        rx.recv()
            .await
            .map(|payload| (Ok::<_, Infallible>(Event::default().data(payload)), rx))
    });
    Sse::new(stream).keep_alive(KeepAlive::default())
}

/// Drive one chat stream to completion, pushing `data:` payloads into `tx`.
///
/// The supervisor's chat sink only closes on worker death — end of generation
/// is detected by `outcome` resolving, after which residual deltas still
/// queued in the channel are drained so no content is lost. On failure the
/// OpenAI error envelope is emitted as a data event (OpenAI's own mid-stream
/// error convention), followed by `[DONE]` either way.
#[allow(clippy::too_many_arguments)]
async fn assemble(
    id: String,
    model: String,
    created: u32,
    deltas: mpsc::UnboundedReceiver<String>,
    outcome: JoinHandle<Result<ChatOutcome, HiggsError>>,
    tx: mpsc::UnboundedSender<String>,
    verbose: bool,
    started: std::time::Instant,
    include_usage: bool,
) {
    let send = |payload: String| {
        // client disconnected — sends discard silently; assemble still runs to outcome so the spawned task joins clean
        let _ = tx.send(payload);
    };

    send(chunk_payload(
        &id,
        &model,
        created,
        Some(Role::Assistant),
        None,
        None,
        None,
    ));

    let joined = drain_deltas(&id, &model, created, deltas, outcome, &send).await;

    match joined {
        Ok(Ok(out)) => {
            // Verbose serving line fires here — the streaming path's final
            // outcome is known. Same `higgs: served …` line as the
            // non-streaming path (shared `v1::log_served`).
            if verbose {
                super::v1::log_served(&model, &out.finish_reason, out.completion_tokens, started);
            }
            emit_outcome(&id, &model, created, &out, &send);
            // OpenAI `stream_options.include_usage`: after the finish chunk, emit one final
            // chunk with empty `choices` and the populated `usage` block (real engine token
            // counts), so a streaming client gets the same accounting as the non-stream path.
            if include_usage {
                send(usage_payload(
                    &id,
                    &model,
                    created,
                    out.prompt_tokens,
                    out.completion_tokens,
                ));
            }
        }
        Ok(Err(err)) => {
            tracing::warn!(error = %err, "higgs: chat stream failed mid-generation");
            send(super::v1::v1_envelope_json(
                super::http_status(&err),
                &err.to_string(),
            ));
        }
        // JoinError: the chat task panicked or was aborted — not a HiggsError.
        // Surface it as a coded HG044 so the SSE error envelope says what to do.
        Err(join_err) => {
            let e = crate::diagnostic::HiggsError::ChatTaskFailed {
                detail: join_err.to_string(),
            };
            tracing::warn!(error = %e, "higgs: chat task failed");
            send(super::v1::v1_envelope_json(
                super::http_status(&e),
                &e.to_string(),
            ));
        }
    }
    send("[DONE]".to_owned());
}

/// Stream content delta chunks until generation completes, returning the joined
/// outcome. Deltas are drained ahead of the finish (biased select) so all
/// content precedes it; on early sink close (worker death) the outcome carries
/// the error.
async fn drain_deltas(
    id: &str,
    model: &str,
    created: u32,
    mut deltas: mpsc::UnboundedReceiver<String>,
    mut outcome: JoinHandle<Result<ChatOutcome, HiggsError>>,
    send: &impl Fn(String),
) -> Result<Result<ChatOutcome, HiggsError>, tokio::task::JoinError> {
    loop {
        tokio::select! {
            // Drain deltas first so all content precedes the finish chunk.
            biased;
            maybe = deltas.recv() => match maybe {
                Some(delta) => {
                    send(chunk_payload(id, model, created, None, Some(delta), None, None));
                }
                // Sink closed early: worker died — the outcome carries the error.
                None => break outcome.await,
            },
            joined = &mut outcome => {
                // Generation finished; flush deltas still queued in the channel.
                while let Ok(delta) = deltas.try_recv() {
                    send(chunk_payload(id, model, created, None, Some(delta), None, None));
                }
                break joined;
            }
        }
    }
}

/// Map a completed outcome onto its terminal chunks: an optional tool-call delta
/// chunk (full name + arguments) followed by the finish chunk. The tool-call
/// interpretation is shared with the non-streaming path ([`interpret_tool_calls`]
/// in `v1`), so identical worker output yields the same `finish_reason` on both:
/// a value that fails to deserialize (e.g. `[{}]`) degrades to no tool calls and
/// the engine's stop/length signal drives the finish reason.
///
/// [`interpret_tool_calls`]: super::v1::interpret_tool_calls
fn emit_outcome(id: &str, model: &str, created: u32, out: &ChatOutcome, send: &impl Fn(String)) {
    let tool_calls = super::v1::interpret_tool_calls(&out.tool_calls).map(stream_tool_calls);
    let finish = if tool_calls.is_some() {
        FinishReason::ToolCalls
    } else {
        finish_reason_from(&out.finish_reason)
    };
    if let Some(tc) = tool_calls {
        send(chunk_payload(
            id,
            model,
            created,
            None,
            None,
            Some(tc),
            None,
        ));
    }
    send(chunk_payload(
        id,
        model,
        created,
        None,
        None,
        None,
        Some(finish),
    ));
}

/// Convert validated OpenAI tool calls (from the shared
/// [`interpret_tool_calls`](super::v1::interpret_tool_calls)) into streaming
/// tool-call chunks — one chunk per call with its full name + arguments (we
/// buffer the call rather than streaming argument deltas). The input is already
/// non-empty and well-formed (degradation happened in `interpret_tool_calls`),
/// so this is a pure shape map.
fn stream_tool_calls(
    calls: Vec<ChatCompletionMessageToolCalls>,
) -> Vec<ChatCompletionMessageToolCallChunk> {
    calls
        .into_iter()
        .enumerate()
        .map(|(i, call)| match call {
            ChatCompletionMessageToolCalls::Function(c) => ChatCompletionMessageToolCallChunk {
                index: i as u32,
                id: Some(c.id),
                r#type: Some(FunctionType::Function),
                function: Some(FunctionCallStream {
                    name: Some(c.function.name),
                    arguments: Some(c.function.arguments),
                }),
            },
            // higgs's OAI-compat parser only ever emits function calls, so a
            // Custom call is not produced in practice; surface its id so the
            // chunk is still well-formed (no streaming-function shape exists).
            ChatCompletionMessageToolCalls::Custom(c) => ChatCompletionMessageToolCallChunk {
                index: i as u32,
                id: Some(c.id),
                r#type: Some(FunctionType::Function),
                function: None,
            },
        })
        .collect()
}

/// Serialize one `chat.completion.chunk` into its `data:` payload string.
fn chunk_payload(
    id: &str,
    model: &str,
    created: u32,
    role: Option<Role>,
    content: Option<String>,
    tool_calls: Option<Vec<ChatCompletionMessageToolCallChunk>>,
    finish_reason: Option<FinishReason>,
) -> String {
    // function_call / system_fingerprint are deprecated-but-required fields of
    // the async-openai wire structs; populating them with None is the only way
    // to construct the type.
    #[allow(deprecated)]
    let chunk = CreateChatCompletionStreamResponse {
        id: id.to_owned(),
        choices: vec![ChatChoiceStream {
            index: 0,
            delta: ChatCompletionStreamResponseDelta {
                content,
                function_call: None,
                tool_calls,
                role,
                refusal: None,
            },
            finish_reason,
            logprobs: None,
        }],
        created,
        model: model.to_owned(),
        service_tier: None,
        system_fingerprint: None,
        object: "chat.completion.chunk".to_owned(),
        usage: None,
    };
    serde_json::to_string(&chunk).expect("chunk serialization cannot fail")
}

/// Serialize the terminal usage-only chunk for `stream_options.include_usage`: empty
/// `choices`, populated `usage` (mirrors the non-streaming `usage` block).
fn usage_payload(
    id: &str,
    model: &str,
    created: u32,
    prompt_tokens: u32,
    completion_tokens: u32,
) -> String {
    #[allow(deprecated)]
    let chunk = CreateChatCompletionStreamResponse {
        id: id.to_owned(),
        choices: vec![], // OpenAI: the usage-only chunk carries no choices
        created,
        model: model.to_owned(),
        service_tier: None,
        system_fingerprint: None,
        object: "chat.completion.chunk".to_owned(),
        usage: Some(CompletionUsage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens.saturating_add(completion_tokens),
            ..Default::default()
        }),
    };
    serde_json::to_string(&chunk).expect("usage chunk serialization cannot fail")
}

/// Map the worker's finish reason string to the OpenAI enum.
///
/// The worker only emits `"stop"` or `"length"` in v1; anything else
/// collapses to `Stop`.
pub(crate) fn finish_reason_from(s: &str) -> FinishReason {
    if s == "length" {
        FinishReason::Length
    } else {
        FinishReason::Stop
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Collect all payloads `assemble` produces for the given inputs. `include_usage`
    /// toggles the OpenAI `stream_options.include_usage` terminal usage chunk.
    async fn run_assemble_opts(
        deltas: mpsc::UnboundedReceiver<String>,
        outcome: JoinHandle<Result<ChatOutcome, HiggsError>>,
        include_usage: bool,
    ) -> Vec<String> {
        let (tx, mut rx) = mpsc::unbounded_channel();
        assemble(
            "chatcmpl-t".into(),
            "org/model".into(),
            1,
            deltas,
            outcome,
            tx,
            false,
            std::time::Instant::now(),
            include_usage,
        )
        .await;
        let mut out = Vec::new();
        while let Ok(p) = rx.try_recv() {
            out.push(p);
        }
        out
    }

    /// The common case: assemble without the usage chunk.
    async fn run_assemble(
        deltas: mpsc::UnboundedReceiver<String>,
        outcome: JoinHandle<Result<ChatOutcome, HiggsError>>,
    ) -> Vec<String> {
        run_assemble_opts(deltas, outcome, false).await
    }

    #[test]
    fn finish_reason_mapping() {
        assert_eq!(finish_reason_from("stop"), FinishReason::Stop);
        assert_eq!(finish_reason_from("length"), FinishReason::Length);
        assert_eq!(finish_reason_from("???"), FinishReason::Stop);
    }

    #[tokio::test]
    async fn assemble_frames_in_order() {
        let (dtx, drx) = mpsc::unbounded_channel();
        dtx.send("hel".to_owned()).unwrap();
        dtx.send("lo".to_owned()).unwrap();
        let outcome = tokio::spawn(async {
            Ok(ChatOutcome {
                content: "hello".into(),
                finish_reason: "stop".into(),
                tool_calls: None,
                prompt_tokens: 0,
                completion_tokens: 0,
            })
        });

        let payloads = run_assemble(drx, outcome).await;
        assert_eq!(payloads.len(), 5, "role + 2 deltas + finish + [DONE]");

        let parse =
            |s: &str| -> CreateChatCompletionStreamResponse { serde_json::from_str(s).unwrap() };
        let role = parse(&payloads[0]);
        assert_eq!(role.choices[0].delta.role, Some(Role::Assistant));
        assert_eq!(role.object, "chat.completion.chunk");

        assert_eq!(
            parse(&payloads[1]).choices[0].delta.content.as_deref(),
            Some("hel")
        );
        assert_eq!(
            parse(&payloads[2]).choices[0].delta.content.as_deref(),
            Some("lo")
        );

        let finish = parse(&payloads[3]);
        assert_eq!(finish.choices[0].finish_reason, Some(FinishReason::Stop));
        assert_eq!(finish.choices[0].delta.content, None);

        assert_eq!(payloads[4], "[DONE]");
    }

    #[tokio::test]
    async fn include_usage_emits_a_terminal_usage_chunk() {
        let (dtx, drx) = mpsc::unbounded_channel();
        dtx.send("hi".to_owned()).unwrap();
        let outcome = tokio::spawn(async {
            Ok(ChatOutcome {
                content: "hi".into(),
                finish_reason: "stop".into(),
                tool_calls: None,
                prompt_tokens: 11,
                completion_tokens: 4,
            })
        });

        let payloads = run_assemble_opts(drx, outcome, true).await;
        // role + delta + finish + usage + [DONE]
        assert_eq!(
            payloads.len(),
            5,
            "usage chunk added before [DONE]: {payloads:?}"
        );
        assert_eq!(payloads[4], "[DONE]");

        let usage_chunk: CreateChatCompletionStreamResponse =
            serde_json::from_str(&payloads[3]).unwrap();
        assert!(usage_chunk.choices.is_empty(), "usage chunk has no choices");
        let usage = usage_chunk.usage.expect("usage block present");
        assert_eq!(usage.prompt_tokens, 11);
        assert_eq!(usage.completion_tokens, 4);
        assert_eq!(usage.total_tokens, 15);
    }

    #[tokio::test]
    async fn no_usage_chunk_without_include_usage() {
        let (dtx, drx) = mpsc::unbounded_channel();
        dtx.send("hi".to_owned()).unwrap();
        let outcome = tokio::spawn(async {
            Ok(ChatOutcome {
                content: "hi".into(),
                finish_reason: "stop".into(),
                tool_calls: None,
                prompt_tokens: 11,
                completion_tokens: 4,
            })
        });
        let payloads = run_assemble(drx, outcome).await;
        // role + delta + finish + [DONE] — NO usage chunk.
        assert_eq!(
            payloads.len(),
            4,
            "no usage chunk when not requested: {payloads:?}"
        );
        assert!(
            payloads.iter().all(
                |p| serde_json::from_str::<CreateChatCompletionStreamResponse>(p)
                    .map(|c| c.usage.is_none())
                    .unwrap_or(true)
            ),
            "no chunk carries a usage block"
        );
    }

    #[tokio::test]
    async fn assemble_error_emits_envelope_then_done() {
        let (_dtx, drx) = mpsc::unbounded_channel::<String>();
        let outcome = tokio::spawn(async {
            Err(HiggsError::WorkerDead {
                context: "gone".into(),
            })
        });

        let payloads = run_assemble(drx, outcome).await;
        assert_eq!(payloads.len(), 3, "role + error envelope + [DONE]");
        assert!(
            payloads[1].contains("[HG007]"),
            "envelope carries the diagnostic code"
        );
        assert!(payloads[1].contains("server_error"));
        assert_eq!(payloads[2], "[DONE]");
    }

    /// Worker dies mid-stream: some deltas arrive, then the sender drops
    /// (simulating worker death), and the outcome future resolves with
    /// `Err(HiggsError::WorkerDead)`. The SSE output must contain the
    /// delivered delta payloads, then the error envelope, then `[DONE]`.
    #[tokio::test]
    async fn assemble_worker_death_mid_stream() {
        let (dtx, drx) = mpsc::unbounded_channel();
        dtx.send("he".to_owned()).unwrap();
        // Drop the sender to simulate worker death closing the delta channel.
        drop(dtx);

        let outcome = tokio::spawn(async {
            Err(HiggsError::WorkerDead {
                context: "worker panicked mid-generation".into(),
            })
        });

        let payloads = run_assemble(drx, outcome).await;
        // Expected: role chunk + "he" delta + error envelope + [DONE]
        assert_eq!(payloads.len(), 4, "role + delta + error envelope + [DONE]");

        let parse =
            |s: &str| -> CreateChatCompletionStreamResponse { serde_json::from_str(s).unwrap() };
        assert_eq!(
            parse(&payloads[0]).choices[0].delta.role,
            Some(Role::Assistant)
        );
        assert_eq!(
            parse(&payloads[1]).choices[0].delta.content.as_deref(),
            Some("he")
        );

        // Error envelope carries the diagnostic code and error type.
        assert!(
            payloads[2].contains("[HG007]"),
            "envelope carries the diagnostic code"
        );
        assert!(payloads[2].contains("server_error"));

        assert_eq!(payloads[3], "[DONE]", "[DONE] is always last");
    }

    /// Parse a worker `tool_calls` value through the shared interpreter (as the
    /// assemble path does), then map to streaming chunks. Mirrors the real flow.
    fn stream_from_value(v: serde_json::Value) -> Option<Vec<ChatCompletionMessageToolCallChunk>> {
        super::super::v1::interpret_tool_calls(&Some(v)).map(stream_tool_calls)
    }

    #[test]
    fn stream_tool_calls_maps_fields() {
        let chunks = stream_from_value(serde_json::json!([
            {
                "id": "call_abc",
                "type": "function",
                "function": { "name": "get_weather", "arguments": "{\"city\":\"SF\"}" }
            },
            {
                "id": "call_def",
                "type": "function",
                "function": { "name": "lookup", "arguments": "{}" }
            }
        ]))
        .expect("non-empty well-formed array maps to Some");
        assert_eq!(chunks.len(), 2);

        assert_eq!(chunks[0].index, 0);
        assert_eq!(chunks[0].id.as_deref(), Some("call_abc"));
        assert_eq!(chunks[0].r#type, Some(FunctionType::Function));
        let f0 = chunks[0].function.as_ref().expect("function present");
        assert_eq!(f0.name.as_deref(), Some("get_weather"));
        assert_eq!(f0.arguments.as_deref(), Some("{\"city\":\"SF\"}"));

        assert_eq!(chunks[1].index, 1);
        assert_eq!(chunks[1].id.as_deref(), Some("call_def"));
        let f1 = chunks[1].function.as_ref().expect("function present");
        assert_eq!(f1.name.as_deref(), Some("lookup"));
        assert_eq!(f1.arguments.as_deref(), Some("{}"));
    }

    #[test]
    fn stream_tool_calls_none_for_empty_or_malformed() {
        // Empty array → None (no calls to emit).
        assert!(stream_from_value(serde_json::json!([])).is_none());
        // Non-array / wrong-shaped values fail to deserialize → None (same
        // degradation as the non-streaming path).
        assert!(stream_from_value(serde_json::json!({"function": {"name": "x"}})).is_none());
        assert!(stream_from_value(serde_json::json!("not an array")).is_none());
        assert!(stream_from_value(serde_json::Value::Null).is_none());
        // A malformed element `[{}]` — no id/type/function — does NOT deserialize
        // into the wire type, so the whole value degrades to None (matching the
        // non-streaming path: identical worker output → same finish_reason).
        assert!(stream_from_value(serde_json::json!([{}])).is_none());
    }

    /// An outcome carrying tool_calls must emit a tool-call delta chunk plus a
    /// finish chunk whose reason is `tool_calls` (overriding the engine's
    /// stop/length signal), before `[DONE]`.
    #[tokio::test]
    async fn assemble_tool_calls_path() {
        let (_dtx, drx) = mpsc::unbounded_channel::<String>();
        let outcome = tokio::spawn(async {
            Ok(ChatOutcome {
                content: String::new(),
                // Engine said "stop" but tool_calls must force "tool_calls".
                finish_reason: "stop".into(),
                tool_calls: Some(serde_json::json!([{
                    "id": "call_1",
                    "type": "function",
                    "function": { "name": "search", "arguments": "{\"q\":\"x\"}" }
                }])),
                prompt_tokens: 0,
                completion_tokens: 0,
            })
        });

        let payloads = run_assemble(drx, outcome).await;
        // role chunk + tool_calls delta + finish chunk + [DONE]
        assert_eq!(payloads.len(), 4, "role + tool_calls + finish + [DONE]");

        let parse =
            |s: &str| -> CreateChatCompletionStreamResponse { serde_json::from_str(s).unwrap() };

        // payloads[1] carries the tool-call delta.
        let tc = parse(&payloads[1]);
        let calls = tc.choices[0]
            .delta
            .tool_calls
            .as_ref()
            .expect("tool_calls delta present");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id.as_deref(), Some("call_1"));
        assert_eq!(
            calls[0].function.as_ref().unwrap().name.as_deref(),
            Some("search")
        );
        // No finish_reason on the tool-call delta chunk itself.
        assert_eq!(tc.choices[0].finish_reason, None);

        // payloads[2] is the finish chunk: reason forced to tool_calls.
        let finish = parse(&payloads[2]);
        assert_eq!(
            finish.choices[0].finish_reason,
            Some(FinishReason::ToolCalls)
        );

        assert_eq!(payloads[3], "[DONE]");
    }

    /// Outcome with an empty tool_calls array behaves like a normal completion:
    /// `stream_tool_calls` returns None, so the finish reason comes from the
    /// engine signal and no tool-call delta chunk is emitted.
    #[tokio::test]
    async fn assemble_empty_tool_calls_falls_back_to_finish_reason() {
        let (_dtx, drx) = mpsc::unbounded_channel::<String>();
        let outcome = tokio::spawn(async {
            Ok(ChatOutcome {
                content: "done".into(),
                finish_reason: "length".into(),
                tool_calls: Some(serde_json::json!([])),
                prompt_tokens: 0,
                completion_tokens: 0,
            })
        });

        let payloads = run_assemble(drx, outcome).await;
        // role + finish + [DONE] — no tool-call delta for an empty array.
        assert_eq!(payloads.len(), 3, "role + finish + [DONE]");
        let parse =
            |s: &str| -> CreateChatCompletionStreamResponse { serde_json::from_str(s).unwrap() };
        let finish = parse(&payloads[1]);
        assert_eq!(finish.choices[0].finish_reason, Some(FinishReason::Length));
        assert_eq!(payloads[2], "[DONE]");
    }

    /// C3 parity: a malformed `tool_calls` (`[{}]`, which does NOT deserialize
    /// into the async-openai wire type) must degrade to no tool calls on the
    /// streaming path just as it does on the non-streaming path — no tool-call
    /// delta, and finish_reason comes from the engine signal (NOT forced to
    /// `tool_calls`). The matching non-stream assertion lives in `v1`'s
    /// `chat_response_malformed_tool_calls_degrade_to_none`.
    #[tokio::test]
    async fn assemble_malformed_tool_calls_degrade_to_finish_reason() {
        let (_dtx, drx) = mpsc::unbounded_channel::<String>();
        let outcome = tokio::spawn(async {
            Ok(ChatOutcome {
                content: "text".into(),
                finish_reason: "stop".into(),
                // `[{}]` — a non-empty array whose element lacks id/type/function.
                tool_calls: Some(serde_json::json!([{}])),
                prompt_tokens: 1,
                completion_tokens: 1,
            })
        });

        let payloads = run_assemble(drx, outcome).await;
        // role + finish + [DONE] — no tool-call delta for the degraded value.
        assert_eq!(payloads.len(), 3, "role + finish + [DONE]: {payloads:?}");
        let parse =
            |s: &str| -> CreateChatCompletionStreamResponse { serde_json::from_str(s).unwrap() };
        let finish = parse(&payloads[1]);
        assert_eq!(
            finish.choices[0].finish_reason,
            Some(FinishReason::Stop),
            "malformed tool_calls must not force a tool_calls finish reason"
        );
        assert_eq!(payloads[2], "[DONE]");
    }
}
