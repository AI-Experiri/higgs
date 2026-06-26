
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
