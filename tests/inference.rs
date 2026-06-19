//! Black-box integration test for higgs's OpenAI `/v1/*` inference surface.
//!
//! Spawns the real `higgs`, loads a Nemotron model, and exercises chat
//! (non-streaming + streaming) and tool calling (non-streaming + streaming)
//! plus `/v1/models`. `#[ignore]` because it loads a multi-GB GGUF and runs real
//! generation; run with `cargo test -p higgs --test inference -- --ignored`.

mod common;

use common::{nemotron_id, spawn};
use serde_json::{json, Value};

/// The get_weather tool used by the tool-calling assertions.
fn weather_tool() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": { "location": { "type": "string" } },
                "required": ["location"]
            }
        }
    })
}

#[tokio::test]
#[ignore = "integration: spawns higgs + loads a real GGUF, runs generation (run with --ignored)"]
async fn inference_and_tools() {
    let srv = spawn(11501).await;
    let c = reqwest::Client::new();

    // Load Nemotron (or skip if it isn't installed).
    let models: Value = c
        .get(format!("{}/api/higgs/models", srv.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let Some(id) = nemotron_id(&models) else {
        eprintln!("SKIP inference_and_tools: no Nemotron model on disk");
        return;
    };

    // ── /v1/models is EMPTY before any model is loaded ────────────────────────
    let empty: Value = c
        .get(format!("{}/v1/models", srv.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(empty["object"], "list", "/v1/models is a list");
    assert!(
        empty["data"].as_array().unwrap().is_empty(),
        "/v1/models is empty with nothing loaded"
    );

    // ── Chat for an unloaded model → 404 OpenAI error envelope ────────────────
    let unloaded = c
        .post(format!("{}/v1/chat/completions", srv.base))
        .json(&json!({
            "model": id, "stream": false,
            "messages": [{ "role": "user", "content": "hi" }]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(unloaded.status(), 404, "chat for unloaded model is 404");
    let env: Value = unloaded.json().await.unwrap();
    assert_eq!(
        env["error"]["code"], "model_not_found",
        "404 envelope codes model_not_found: {env:?}"
    );
    assert_eq!(
        env["error"]["type"], "invalid_request_error",
        "404 is an invalid_request_error"
    );

    let load = c
        .post(format!("{}/api/higgs/models/load", srv.base))
        .json(&json!({ "id": id }))
        .send()
        .await
        .unwrap();
    assert!(load.status().is_success(), "model load succeeded");

    // /v1/models lists the loaded model.
    let v1models: Value = c
        .get(format!("{}/v1/models", srv.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(
        v1models["data"]
            .as_array()
            .unwrap()
            .iter()
            .any(|m| m["id"] == json!(id)),
        "/v1/models lists the loaded model"
    );

    let chat = |body: Value| {
        c.post(format!("{}/v1/chat/completions", srv.base))
            .json(&body)
            .send()
    };

    // ── Non-streaming plain chat ──────────────────────────────────────────
    let resp: Value = chat(json!({
        "model": id, "stream": false,
        "messages": [{ "role": "user", "content": "Say hello in one word." }]
    }))
    .await
    .unwrap()
    .json()
    .await
    .unwrap();
    let choice = &resp["choices"][0];
    assert!(
        choice["message"]["content"]
            .as_str()
            .is_some_and(|s| !s.is_empty()),
        "plain chat returns non-empty content"
    );
    assert!(
        matches!(
            choice["finish_reason"].as_str(),
            Some("stop") | Some("length")
        ),
        "plain finish_reason is stop|length"
    );
    assert!(
        resp["usage"]["prompt_tokens"].as_u64().unwrap() > 0
            && resp["usage"]["completion_tokens"].as_u64().unwrap() > 0,
        "usage token counts are non-zero"
    );

    // ── Multi-turn messages + BOTH max_tokens and max_completion_tokens ───────
    // max_completion_tokens wins when both are set; a small cap → finish "length".
    let resp: Value = chat(json!({
        "model": id, "stream": false,
        "max_tokens": 200, "max_completion_tokens": 8,
        "messages": [
            { "role": "system", "content": "You are terse." },
            { "role": "user", "content": "Name a color." },
            { "role": "assistant", "content": "Blue." },
            { "role": "user", "content": "Name another one." }
        ]
    }))
    .await
    .unwrap()
    .json()
    .await
    .unwrap();
    let choice = &resp["choices"][0];
    assert!(
        matches!(
            choice["finish_reason"].as_str(),
            Some("stop") | Some("length")
        ),
        "multi-turn finish_reason is stop|length"
    );
    assert!(
        resp["usage"]["completion_tokens"].as_u64().unwrap() <= 8,
        "max_completion_tokens (8) caps the completion"
    );

    // ── Diverse message roles + array content parts (messages_to_pairs) ───────
    // developer + system + array-of-text user + assistant + tool roles all map
    // to (role, text) pairs without rejection.
    let resp: Value = chat(json!({
        "model": id, "stream": false,
        "max_completion_tokens": 8,
        "messages": [
            { "role": "developer", "content": "Be brief." },
            { "role": "system", "content": [{ "type": "text", "text": "One word answers." }] },
            { "role": "user", "content": [{ "type": "text", "text": "Pick a fruit." }] },
            { "role": "assistant", "content": "Apple." },
            { "role": "tool", "tool_call_id": "call_x", "content": "ignored" },
            { "role": "user", "content": "Now pick a vegetable." }
        ]
    }))
    .await
    .unwrap()
    .json()
    .await
    .unwrap();
    assert!(
        resp["choices"][0]["message"]["content"].is_string(),
        "diverse-role chat returns content: {resp:?}"
    );

    // ── A non-text content part (image_url) → 400 invalid_request_error ───────
    let img = c
        .post(format!("{}/v1/chat/completions", srv.base))
        .json(&json!({
            "model": id, "stream": false,
            "messages": [{
                "role": "user",
                "content": [
                    { "type": "text", "text": "describe" },
                    { "type": "image_url", "image_url": { "url": "https://example.com/x.png" } }
                ]
            }]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(img.status(), 400, "image content part is rejected 400");
    let img_env: Value = img.json().await.unwrap();
    assert_eq!(
        img_env["error"]["type"], "invalid_request_error",
        "non-text part is an invalid_request_error: {img_env:?}"
    );

    // ── A tool is offered but the prompt should NOT trigger one ───────────────
    let resp: Value = chat(json!({
        "model": id, "stream": false,
        "messages": [{ "role": "user", "content": "Just say hi back, one word." }],
        "tools": [weather_tool()]
    }))
    .await
    .unwrap()
    .json()
    .await
    .unwrap();
    let choice = &resp["choices"][0];
    assert!(
        choice["message"]["tool_calls"].is_null(),
        "plain greeting does not call a tool: {choice:?}"
    );
    assert!(
        choice["message"]["content"]
            .as_str()
            .is_some_and(|s| !s.is_empty()),
        "no-tool turn still returns content"
    );
    assert_ne!(
        choice["finish_reason"], "tool_calls",
        "no-tool turn does not finish with tool_calls"
    );

    // ── Malformed request: missing `model` field → 4xx (no 5xx, no panic) ─────
    let bad = c
        .post(format!("{}/v1/chat/completions", srv.base))
        .json(&json!({
            "messages": [{ "role": "user", "content": "hi" }]
        }))
        .send()
        .await
        .unwrap();
    assert!(
        bad.status().is_client_error(),
        "missing model is a 4xx, got {}",
        bad.status()
    );

    // ── Non-streaming tool call ───────────────────────────────────────────
    let resp: Value = chat(json!({
        "model": id, "stream": false,
        "messages": [{ "role": "user", "content": "What is the weather in Paris? Use the get_weather tool." }],
        "tools": [weather_tool()]
    }))
    .await
    .unwrap()
    .json()
    .await
    .unwrap();
    let choice = &resp["choices"][0];
    let calls = choice["message"]["tool_calls"]
        .as_array()
        .expect("tool_calls present");
    assert_eq!(calls.len(), 1, "exactly one tool call");
    assert_eq!(
        calls[0]["function"]["name"], "get_weather",
        "calls get_weather"
    );
    let args = calls[0]["function"]["arguments"].as_str().unwrap();
    assert!(
        args.to_lowercase().contains("paris"),
        "arguments mention Paris: {args}"
    );
    assert_eq!(
        choice["finish_reason"], "tool_calls",
        "finish_reason tool_calls"
    );

    // ── Streaming plain chat ──────────────────────────────────────────────
    let body = chat(json!({
        "model": id, "stream": true,
        "messages": [{ "role": "user", "content": "Count to three." }]
    }))
    .await
    .unwrap()
    .text()
    .await
    .unwrap();
    assert!(body.contains("data:"), "stream has data lines");
    assert!(body.contains("[DONE]"), "stream terminates with [DONE]");
    assert!(
        body.contains("\"content\""),
        "stream carries content deltas"
    );

    // ── Streaming tool call ───────────────────────────────────────────────
    let body = chat(json!({
        "model": id, "stream": true,
        "messages": [{ "role": "user", "content": "What is the weather in Paris? Use the get_weather tool." }],
        "tools": [weather_tool()]
    }))
    .await
    .unwrap()
    .text()
    .await
    .unwrap();
    // The tool-call envelope must NOT leak into content deltas.
    assert!(
        !body.contains("<tool_call>") && !body.contains("<function="),
        "no tool-call markup leaks into the stream"
    );
    assert!(
        body.contains("tool_calls"),
        "stream emits a tool_calls delta"
    );
    assert!(
        body.contains("\"finish_reason\":\"tool_calls\""),
        "stream finishes with tool_calls"
    );
    assert!(body.contains("[DONE]"), "stream terminates with [DONE]");
}
