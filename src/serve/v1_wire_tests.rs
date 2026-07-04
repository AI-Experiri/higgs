//! Serde-parity proofs: each mirror (with `reasoning_content: None`)
//! serializes to the EXACT `Value` the async-openai struct produces, so
//! swapping the wire types cannot change the `/v1` protocol. Plus the one
//! deliberate extension: `reasoning_content` present ⇔ `Some`.

use super::*;
use async_openai::types::chat::{
    ChatChoice, ChatChoiceStream, ChatCompletionResponseMessage, ChatCompletionStreamResponseDelta,
    CreateChatCompletionResponse, CreateChatCompletionStreamResponse,
};
use serde_json::{json, to_value};

/// The crate's usage block, shared by both parity fixtures.
fn usage() -> CompletionUsage {
    CompletionUsage {
        prompt_tokens: 3,
        completion_tokens: 5,
        total_tokens: 8,
        ..Default::default()
    }
}

#[test]
fn final_response_parity_with_async_openai() {
    #[allow(deprecated)]
    let crate_resp = CreateChatCompletionResponse {
        id: "chatcmpl-x".into(),
        choices: vec![ChatChoice {
            index: 0,
            message: ChatCompletionResponseMessage {
                content: Some("hi".into()),
                refusal: None,
                tool_calls: None,
                annotations: None,
                role: Role::Assistant,
                function_call: None,
                audio: None,
            },
            finish_reason: Some(FinishReason::Stop),
            logprobs: None,
        }],
        created: 7,
        model: "m".into(),
        service_tier: None,
        system_fingerprint: None,
        object: "chat.completion".into(),
        usage: Some(usage()),
    };
    let mirror = ReasoningChatResponse {
        id: "chatcmpl-x".into(),
        choices: vec![ReasoningChatChoice {
            index: 0,
            message: ReasoningResponseMessage {
                content: Some("hi".into()),
                refusal: None,
                tool_calls: None,
                role: Role::Assistant,
                reasoning_content: None,
            },
            finish_reason: Some(FinishReason::Stop),
        }],
        created: 7,
        model: "m".into(),
        object: "chat.completion".into(),
        usage: Some(usage()),
    };
    assert_eq!(
        to_value(&mirror).unwrap(),
        to_value(&crate_resp).unwrap(),
        "final-response mirror must be byte-parity with async-openai when reasoning is None"
    );
}

#[test]
fn stream_chunk_parity_with_async_openai() {
    #[allow(deprecated)]
    let crate_chunk = CreateChatCompletionStreamResponse {
        id: "chatcmpl-x".into(),
        choices: vec![ChatChoiceStream {
            index: 0,
            delta: ChatCompletionStreamResponseDelta {
                content: Some("hi".into()),
                function_call: None,
                tool_calls: None,
                role: None,
                refusal: None,
            },
            finish_reason: None,
            logprobs: None,
        }],
        created: 7,
        model: "m".into(),
        service_tier: None,
        system_fingerprint: None,
        object: "chat.completion.chunk".into(),
        usage: None,
    };
    let mirror = ReasoningStreamChunk {
        id: "chatcmpl-x".into(),
        choices: vec![ReasoningChoiceStream {
            index: 0,
            delta: ReasoningStreamDelta {
                content: Some("hi".into()),
                ..Default::default()
            },
            finish_reason: None,
        }],
        created: 7,
        model: "m".into(),
        service_tier: None,
        system_fingerprint: None,
        object: "chat.completion.chunk".into(),
        usage: None,
    };
    assert_eq!(
        to_value(&mirror).unwrap(),
        to_value(&crate_chunk).unwrap(),
        "stream-chunk mirror must be byte-parity with async-openai when reasoning is None"
    );
}

#[test]
fn reasoning_content_absent_when_none_present_when_some() {
    let none = to_value(ReasoningStreamDelta::default()).unwrap();
    assert!(
        none.get("reasoning_content").is_none(),
        "reasoning_content must be ABSENT (not null) when None: {none}"
    );
    // The base delta fields stay explicit nulls (crate behavior).
    assert!(none.get("content").is_some_and(serde_json::Value::is_null));

    let some = to_value(ReasoningStreamDelta {
        reasoning_content: Some("thinking".into()),
        ..Default::default()
    })
    .unwrap();
    assert_eq!(some["reasoning_content"], json!("thinking"));

    let msg = to_value(ReasoningResponseMessage {
        content: Some(String::new()),
        refusal: None,
        tool_calls: None,
        role: Role::Assistant,
        reasoning_content: Some("partial think".into()),
    })
    .unwrap();
    assert_eq!(msg["reasoning_content"], json!("partial think"));
}
