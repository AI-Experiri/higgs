//! SSE assembly for `/v1/chat/completions` streaming responses.
//!
//! Bridges the [`Higgs::chat_stream`](crate::api::Higgs::chat_stream) pair
//! (delta receiver + outcome handle) onto the OpenAI chunk protocol:
//! assistant-role chunk → content delta chunks → finish chunk → `data: [DONE]`.
//! Assembly is channel-based (shimmy / local.ai pattern): a spawned task pushes
//! ready `data:` payload strings into an mpsc channel the SSE body drains.

use std::convert::Infallible;

use async_openai::types::chat::{
    ChatChoiceStream, ChatCompletionStreamResponseDelta, CreateChatCompletionStreamResponse,
    FinishReason, Role,
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
pub(crate) fn chat_sse(
    id: String,
    model: String,
    created: u32,
    deltas: mpsc::UnboundedReceiver<String>,
    outcome: JoinHandle<Result<ChatOutcome, HiggsError>>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let (tx, rx) = mpsc::unbounded_channel::<String>();
    tokio::spawn(assemble(id, model, created, deltas, outcome, tx));
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
async fn assemble(
    id: String,
    model: String,
    created: u32,
    mut deltas: mpsc::UnboundedReceiver<String>,
    mut outcome: JoinHandle<Result<ChatOutcome, HiggsError>>,
    tx: mpsc::UnboundedSender<String>,
) {
    let send = |payload: String| {
        // client disconnected — sends discard silently; assemble still runs to outcome so the spawned task joins clean
        let _ = tx.send(payload);
    };

    send(chunk_payload(&id, &model, created, Some(Role::Assistant), None, None));

    let joined = loop {
        tokio::select! {
            // Drain deltas first so all content precedes the finish chunk.
            biased;
            maybe = deltas.recv() => match maybe {
                Some(delta) => {
                    send(chunk_payload(&id, &model, created, None, Some(delta), None));
                }
                // Sink closed early: worker died — the outcome carries the error.
                None => break outcome.await,
            },
            joined = &mut outcome => {
                // Generation finished; flush deltas still queued in the channel.
                while let Ok(delta) = deltas.try_recv() {
                    send(chunk_payload(&id, &model, created, None, Some(delta), None));
                }
                break joined;
            }
        }
    };

    match joined {
        Ok(Ok(out)) => send(chunk_payload(
            &id,
            &model,
            created,
            None,
            None,
            Some(finish_reason_from(&out.finish_reason)),
        )),
        Ok(Err(err)) => {
            tracing::warn!(error = %err, "higgs: chat stream failed mid-generation");
            send(super::v1_envelope_json(super::http_status(&err), err.to_string()));
        }
        // JoinError: the chat task panicked or was aborted — not a HiggsError.
        Err(join_err) => {
            tracing::warn!(error = %join_err, "higgs: chat task failed");
            send(super::v1_envelope_json(
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                format!("chat task failed: {join_err}"),
            ));
        }
    }
    send("[DONE]".to_owned());
}

/// Serialize one `chat.completion.chunk` into its `data:` payload string.
fn chunk_payload(
    id: &str,
    model: &str,
    created: u32,
    role: Option<Role>,
    content: Option<String>,
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
                tool_calls: None,
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

    /// Collect all payloads `assemble` produces for the given inputs.
    async fn run_assemble(
        deltas: mpsc::UnboundedReceiver<String>,
        outcome: JoinHandle<Result<ChatOutcome, HiggsError>>,
    ) -> Vec<String> {
        let (tx, mut rx) = mpsc::unbounded_channel();
        assemble("chatcmpl-t".into(), "org/model".into(), 1, deltas, outcome, tx).await;
        let mut out = Vec::new();
        while let Ok(p) = rx.try_recv() {
            out.push(p);
        }
        out
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
            Ok(ChatOutcome { content: "hello".into(), finish_reason: "stop".into() })
        });

        let payloads = run_assemble(drx, outcome).await;
        assert_eq!(payloads.len(), 5, "role + 2 deltas + finish + [DONE]");

        let parse = |s: &str| -> CreateChatCompletionStreamResponse {
            serde_json::from_str(s).unwrap()
        };
        let role = parse(&payloads[0]);
        assert_eq!(role.choices[0].delta.role, Some(Role::Assistant));
        assert_eq!(role.object, "chat.completion.chunk");

        assert_eq!(parse(&payloads[1]).choices[0].delta.content.as_deref(), Some("hel"));
        assert_eq!(parse(&payloads[2]).choices[0].delta.content.as_deref(), Some("lo"));

        let finish = parse(&payloads[3]);
        assert_eq!(finish.choices[0].finish_reason, Some(FinishReason::Stop));
        assert_eq!(finish.choices[0].delta.content, None);

        assert_eq!(payloads[4], "[DONE]");
    }

    #[tokio::test]
    async fn assemble_error_emits_envelope_then_done() {
        let (_dtx, drx) = mpsc::unbounded_channel::<String>();
        let outcome = tokio::spawn(async {
            Err(HiggsError::WorkerDead { context: "gone".into() })
        });

        let payloads = run_assemble(drx, outcome).await;
        assert_eq!(payloads.len(), 3, "role + error envelope + [DONE]");
        assert!(payloads[1].contains("[HG007]"), "envelope carries the diagnostic code");
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
            Err(HiggsError::WorkerDead { context: "worker panicked mid-generation".into() })
        });

        let payloads = run_assemble(drx, outcome).await;
        // Expected: role chunk + "he" delta + error envelope + [DONE]
        assert_eq!(payloads.len(), 4, "role + delta + error envelope + [DONE]");

        let parse = |s: &str| -> CreateChatCompletionStreamResponse {
            serde_json::from_str(s).unwrap()
        };
        assert_eq!(parse(&payloads[0]).choices[0].delta.role, Some(Role::Assistant));
        assert_eq!(parse(&payloads[1]).choices[0].delta.content.as_deref(), Some("he"));

        // Error envelope carries the diagnostic code and error type.
        assert!(payloads[2].contains("[HG007]"), "envelope carries the diagnostic code");
        assert!(payloads[2].contains("server_error"));

        assert_eq!(payloads[3], "[DONE]", "[DONE] is always last");
    }
}
