//! SSE assembly for `/v1/chat/completions` streaming responses.
//!
//! Bridges the [`Higgs::chat_stream`](crate::api::Higgs::chat_stream) pair
//! (tagged-delta receiver + outcome handle) onto the OpenAI chunk protocol:
//! assistant-role chunk → reasoning/content/tool-call delta chunks (in engine
//! order — reasoning first by construction) → finish chunk → `data: [DONE]`.
//! Assembly is channel-based (shimmy / local.ai pattern): a spawned task pushes
//! ready `data:` payload strings into an mpsc channel the SSE body drains.
//!
//! Chunk bodies are the local [`v1_wire`](super::v1_wire) mirrors — byte-parity
//! with async-openai plus the `reasoning_content` delta field the crate lacks.

use std::convert::Infallible;

use async_openai::types::chat::{
    ChatCompletionMessageToolCallChunk, ChatCompletionMessageToolCalls, CompletionUsage,
    CreateChatCompletionStreamResponse, FinishReason, FunctionCallStream, FunctionType, Role,
};
use axum::response::sse::{Event, KeepAlive, Sse};
use futures::Stream;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use super::v1_wire::{ReasoningChoiceStream, ReasoningStreamChunk, ReasoningStreamDelta};
use crate::api::ChatOutcome;
use crate::diagnostic::HiggsError;
use crate::worker::engine::{ChatDelta, ChatDeltaKind};

/// Build the SSE response for one streaming chat completion.
///
/// `deltas` carries the worker's tagged chunks; `outcome` resolves when
/// generation completes. Each emitted payload becomes one `data:` line; the
/// stream closes after the trailing `data: [DONE]`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn chat_sse(
    id: String,
    model: String,
    created: u32,
    deltas: mpsc::UnboundedReceiver<ChatDelta>,
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
    deltas: mpsc::UnboundedReceiver<ChatDelta>,
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
        ReasoningStreamDelta {
            role: Some(Role::Assistant),
            ..Default::default()
        },
        None,
    ));

    let (joined, streamed_tool_calls) =
        drain_deltas(&id, &model, created, deltas, outcome, &send).await;

    match joined {
        Ok(Ok(out)) => {
            // Verbose serving line fires here — the streaming path's final
            // outcome is known. Same `higgs: served …` line as the
            // non-streaming path (shared `v1::log_served`).
            if verbose {
                super::v1::log_served(&model, &out.finish_reason, out.completion_tokens, started);
            }
            emit_outcome(&id, &model, created, &out, streamed_tool_calls, &send);
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
            // `v1_sse_error` maps the code (a worker-exact [HG005] →
            // context_length_exceeded), matching the non-streaming path.
            send(super::v1::v1_sse_error(&err));
        }
        // JoinError: the chat task panicked or was aborted — not a HiggsError.
        // Surface it as a coded HG044 so the SSE error envelope says what to do.
        Err(join_err) => {
            let e = crate::diagnostic::HiggsError::ChatTaskFailed {
                detail: join_err.to_string(),
            };
            tracing::warn!(error = %e, "higgs: chat task failed");
            send(super::v1::v1_sse_error(&e));
        }
    }
    send("[DONE]".to_owned());
}

/// Map one tagged worker delta onto its OpenAI chunk delta, or `None` for a
/// tool-call fragment that does not deserialize (logged and dropped — the
/// buffered terminal chunk from the final parse still covers the call).
fn delta_chunk(delta: &ChatDelta) -> Option<ReasoningStreamDelta> {
    match delta.kind {
        ChatDeltaKind::Content => Some(ReasoningStreamDelta {
            content: Some(delta.text.clone()),
            ..Default::default()
        }),
        ChatDeltaKind::Reasoning => Some(ReasoningStreamDelta {
            reasoning_content: Some(delta.text.clone()),
            ..Default::default()
        }),
        ChatDeltaKind::ToolCall => match serde_json::from_str(&delta.text) {
            Ok(frag) => Some(ReasoningStreamDelta {
                tool_calls: Some(vec![frag]),
                ..Default::default()
            }),
            Err(err) => {
                tracing::warn!(error = %err, fragment = %delta.text, "[HG056] malformed streamed tool-call fragment dropped; terminal buffered chunk covers the call");
                None
            }
        },
    }
}

/// Stream tagged delta chunks until generation completes, returning the joined
/// outcome plus whether any tool-call fragment was streamed (so the terminal
/// buffered tool-call chunk is emitted ONLY as the fallback). Deltas are
/// drained ahead of the finish (biased select) so all content precedes it; on
/// early sink close (worker death) the outcome carries the error.
async fn drain_deltas(
    id: &str,
    model: &str,
    created: u32,
    mut deltas: mpsc::UnboundedReceiver<ChatDelta>,
    mut outcome: JoinHandle<Result<ChatOutcome, HiggsError>>,
    send: &impl Fn(String),
) -> (
    Result<Result<ChatOutcome, HiggsError>, tokio::task::JoinError>,
    bool,
) {
    let mut streamed_tool_calls = false;
    let mut emit = |delta: ChatDelta| {
        if let Some(d) = delta_chunk(&delta) {
            // Count a tool fragment as streamed only when it actually EMITTED —
            // a malformed (dropped) fragment must not suppress the terminal
            // buffered tool-call chunk, or the client would see
            // finish_reason:"tool_calls" with no call data at all.
            streamed_tool_calls |= delta.kind == ChatDeltaKind::ToolCall;
            send(chunk_payload(id, model, created, d, None));
        }
    };
    let joined = loop {
        tokio::select! {
            // Drain deltas first so all content precedes the finish chunk.
            biased;
            maybe = deltas.recv() => match maybe {
                Some(delta) => emit(delta),
                // Sink closed early: worker died — the outcome carries the error.
                None => break outcome.await,
            },
            joined = &mut outcome => {
                // Generation finished; flush deltas still queued in the channel.
                while let Ok(delta) = deltas.try_recv() {
                    emit(delta);
                }
                break joined;
            }
        }
    };
    (joined, streamed_tool_calls)
}

/// Map a completed outcome onto its terminal chunks: an optional tool-call
/// chunk (full name + arguments) followed by the finish chunk. The tool-call
/// interpretation is shared with the non-streaming path ([`interpret_tool_calls`]
/// in `v1`), so identical worker output yields the same `finish_reason` on both:
/// a value that fails to deserialize (e.g. `[{}]`) degrades to no tool calls and
/// the engine's stop/length signal drives the finish reason.
///
/// `streamed_tool_calls` — the incremental parser already streamed the call
/// fragments, so the buffered terminal chunk is SKIPPED (it exists for the
/// fallback path where nothing streams the calls); `finish_reason` still
/// reports `tool_calls` either way.
///
/// [`interpret_tool_calls`]: super::v1::interpret_tool_calls
fn emit_outcome(
    id: &str,
    model: &str,
    created: u32,
    out: &ChatOutcome,
    streamed_tool_calls: bool,
    send: &impl Fn(String),
) {
    let tool_calls = super::v1::interpret_tool_calls(&out.tool_calls).map(stream_tool_calls);
    let finish = if streamed_tool_calls || tool_calls.is_some() {
        FinishReason::ToolCalls
    } else {
        finish_reason_from(&out.finish_reason)
    };
    if let Some(tc) = tool_calls.filter(|_| !streamed_tool_calls) {
        send(chunk_payload(
            id,
            model,
            created,
            ReasoningStreamDelta {
                tool_calls: Some(tc),
                ..Default::default()
            },
            None,
        ));
    }
    send(chunk_payload(
        id,
        model,
        created,
        ReasoningStreamDelta::default(),
        Some(finish),
    ));
}

/// Convert validated OpenAI tool calls (from the shared
/// [`interpret_tool_calls`](super::v1::interpret_tool_calls)) into streaming
/// tool-call chunks — one chunk per call with its full name + arguments (the
/// FALLBACK when the incremental parser streamed no fragments). The input is
/// already non-empty and well-formed (degradation happened in
/// `interpret_tool_calls`), so this is a pure shape map.
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
    delta: ReasoningStreamDelta,
    finish_reason: Option<FinishReason>,
) -> String {
    let chunk = ReasoningStreamChunk {
        id: id.to_owned(),
        choices: vec![ReasoningChoiceStream {
            index: 0,
            delta,
            finish_reason,
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
#[path = "stream_tests.rs"]
mod tests;
