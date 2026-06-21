//! Black-box integration test for higgs's OpenAI `/v1/*` inference surface.
//!
//! Spawns the real `higgs`, loads a tiny on-disk model, and exercises chat
//! (non-streaming + streaming) plus `/v1/models` and the request-validation
//! paths. The model is `ggml-org`'s ~1MB `stories260K.gguf` (see `common`): a
//! real llama-arch toy GGUF that loads and generates, so the full template →
//! tokenize → decode → detokenize engine path is covered in CI.
//!
//! The tiny model embeds NO chat template (the engine falls back to chatml) and
//! has no tool-call training, so tool calling is covered at the REQUEST-path
//! level — tools are offered and accepted, the response is well-formed, and no
//! tool-call markup leaks into the stream — without asserting the toy model
//! emits a real structured call (which only a tool-trained model would).

mod common;

use common::{spawn_with_tiny_model, tiny_gguf_path, TINY_MODEL_ID};
use serde_json::{json, Value};

/// The get_weather tool used by the tool-path assertions.
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
async fn inference_and_tools() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("SKIP inference_and_tools: tiny gguf not found (set HIGGS_TEST_GGUF)");
        return;
    };
    let srv = spawn_with_tiny_model(11501, &gguf).await;
    let c = reqwest::Client::new();
    let id = TINY_MODEL_ID;

    // The staged tiny model is discoverable via the live scan.
    let models: Value = c
        .get(format!("{}/api/higgs/models", srv.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(
        models["models"]
            .as_array()
            .unwrap()
            .iter()
            .any(|m| m["id"] == json!(id)),
        "scan lists the staged tiny model"
    );

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

    // ── Chat for an UNKNOWN model id → 404 OpenAI error envelope ──────────────
    // JIT is on by default, but JIT only loads a SCANNED id; an unknown id is a
    // 404 `model_not_found`, never a load attempt.
    let unknown = c
        .post(format!("{}/v1/chat/completions", srv.base))
        .json(&json!({
            "model": "no-such-org/no-such-model", "stream": false,
            "messages": [{ "role": "user", "content": "hi" }]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(unknown.status(), 404, "chat for unknown model is 404");
    let env: Value = unknown.json().await.unwrap();
    assert_eq!(
        env["error"]["code"], "model_not_found",
        "404 envelope codes model_not_found: {env:?}"
    );
    assert_eq!(
        env["error"]["type"], "invalid_request_error",
        "404 is an invalid_request_error"
    );

    // ── JIT auto-load: chat for a SCANNED-but-unloaded model loads it on demand.
    // This exercises the v1 JIT path (scan → load → serve) end-to-end against the
    // real engine, distinct from the explicit `/api/higgs/models/load` below.
    let jit: Value = c
        .post(format!("{}/v1/chat/completions", srv.base))
        .json(&json!({
            "model": id, "stream": false,
            "messages": [{ "role": "user", "content": "hi" }]
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(
        jit["choices"][0]["message"]["content"].is_string(),
        "JIT-loaded chat returns content: {jit:?}"
    );

    // Explicitly (re)load via the control surface — idempotent when already
    // resident; keeps the rest of the test on the explicit-load contract.
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

    // ── Tool REQUEST path (non-streaming) ─────────────────────────────────────
    // The tiny model has no tool training, so we don't require a structured call;
    // we require the tools-bearing request to succeed and return a well-formed
    // choice (either plain content or a tool_calls array — never both empty).
    let resp = chat(json!({
        "model": id, "stream": false,
        "messages": [{ "role": "user", "content": "What is the weather in Paris? Use the get_weather tool." }],
        "tools": [weather_tool()]
    }))
    .await
    .unwrap();
    assert!(
        resp.status().is_success(),
        "tools-bearing request succeeds, got {}",
        resp.status()
    );
    let resp: Value = resp.json().await.unwrap();
    let choice = &resp["choices"][0];
    let has_content = choice["message"]["content"]
        .as_str()
        .is_some_and(|s| !s.is_empty());
    let has_calls = choice["message"]["tool_calls"].is_array();
    assert!(
        has_content || has_calls,
        "tools turn returns content or tool_calls: {choice:?}"
    );
    assert!(
        matches!(
            choice["finish_reason"].as_str(),
            Some("stop") | Some("length") | Some("tool_calls")
        ),
        "tools-turn finish_reason is a known reason: {choice:?}"
    );

    // ── Streaming plain chat (SSE) ────────────────────────────────────────────
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
    assert!(
        body.contains("chat.completion.chunk"),
        "stream chunks carry the chunk object type"
    );

    // ── Streaming tool request (SSE) ──────────────────────────────────────────
    // The structured tool call depends on tool training the toy model lacks, so
    // we assert the stream is well-formed and never leaks raw tool-call markup
    // into the content deltas — the invariant that must hold for ANY model.
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
    assert!(body.contains("data:"), "tool stream has data lines");
    assert!(body.contains("[DONE]"), "tool stream terminates with [DONE]");
    assert!(
        !body.contains("<tool_call>") && !body.contains("<function="),
        "no tool-call markup leaks into the stream"
    );

    // ── Sampling-parameter passthrough (non-streaming) ────────────────────────
    // A request carrying the full OpenAI sampling surface must parse and succeed —
    // exercises the param-extraction branches in the /v1 handler.
    let resp: Value = chat(json!({
        "model": id, "stream": false,
        "messages": [{ "role": "user", "content": "Tell a very short story." }],
        "max_tokens": 16,
        "temperature": 0.8,
        "top_p": 0.95,
        "presence_penalty": 0.1,
        "frequency_penalty": 0.1,
        "seed": 42,
        "stop": ["\n\n"]
    }))
    .await
    .unwrap()
    .json()
    .await
    .unwrap();
    assert!(
        resp["choices"][0]["message"]["content"].is_string(),
        "sampling-param request returns content: {resp:?}"
    );

    // ── Streaming with usage accounting (stream_options.include_usage) ─────────
    let body = chat(json!({
        "model": id, "stream": true,
        "messages": [{ "role": "user", "content": "Hello there." }],
        "max_tokens": 8,
        "stream_options": { "include_usage": true }
    }))
    .await
    .unwrap()
    .text()
    .await
    .unwrap();
    assert!(body.contains("[DONE]"), "usage stream terminates with [DONE]");
    // Find the terminal usage chunk: a `data:` line whose `usage` is a NON-null object with
    // real token counts (not the per-chunk `"usage":null`). This is the OpenAI include_usage
    // contract the genai client relies on for streamed token accounting.
    let usage_chunk = body
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter(|p| *p != "[DONE]")
        .filter_map(|p| serde_json::from_str::<Value>(p).ok())
        .find(|v| v["usage"].is_object())
        .expect("a chunk carries a non-null usage block");
    assert!(
        usage_chunk["usage"]["completion_tokens"].as_u64().is_some()
            && usage_chunk["usage"]["prompt_tokens"].as_u64().is_some(),
        "usage chunk has real prompt+completion token counts: {usage_chunk}"
    );
    assert!(
        usage_chunk["choices"].as_array().is_some_and(Vec::is_empty),
        "the usage chunk carries no choices (OpenAI convention): {usage_chunk}"
    );

    // ── Streaming request for an UNKNOWN model → error before any stream opens ──
    let unknown_stream = c
        .post(format!("{}/v1/chat/completions", srv.base))
        .json(&json!({
            "model": "no-such-org/no-such-model", "stream": true,
            "messages": [{ "role": "user", "content": "hi" }]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(unknown_stream.status(), 404, "streaming unknown model is 404 (pre-stream error)");

    // ── Zero max_tokens exercises the boundary branch (clamped or rejected, never 5xx) ──
    let zero = c
        .post(format!("{}/v1/chat/completions", srv.base))
        .json(&json!({
            "model": id, "stream": false, "max_tokens": 0,
            "messages": [{ "role": "user", "content": "hi" }]
        }))
        .send()
        .await
        .unwrap();
    assert!(!zero.status().is_server_error(), "max_tokens=0 must not 5xx, got {}", zero.status());

    // ── Token logging ON + a tool-result conversation ─────────────────────────
    // Turning on "Log Incoming Tokens" exercises the prompt/response logging path; the
    // multi-turn history (assistant tool_call → tool result) exercises message + tool
    // (de)serialization in the /v1 handler.
    let cur: Value = c
        .get(format!("{}/api/higgs/logs/settings", srv.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let mut on = cur.clone();
    on["log_incoming_tokens"] = json!(true);
    let put = c
        .put(format!("{}/api/higgs/logs/settings", srv.base))
        .json(&on)
        .send()
        .await
        .unwrap();
    assert!(put.status().is_success(), "enable token logging");

    let resp = chat(json!({
        "model": id, "stream": false, "max_tokens": 8,
        "messages": [
            { "role": "system", "content": "You are a weather bot." },
            { "role": "user", "content": "Weather in Paris?" },
            { "role": "assistant", "content": null, "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": { "name": "get_weather", "arguments": "{\"city\":\"Paris\"}" }
            }]},
            { "role": "tool", "tool_call_id": "call_1", "content": "15C and sunny" },
            { "role": "user", "content": "Thanks, summarize." }
        ],
        "tools": [weather_tool()]
    }))
    .await
    .unwrap();
    assert!(resp.status().is_success(), "tool-result conversation succeeds: {}", resp.status());

    // A STREAMING chat with token logging still on — exercises the streamed response
    // logging path (log_served on the SSE branch).
    let body = chat(json!({
        "model": id, "stream": true, "max_tokens": 8,
        "messages": [{ "role": "user", "content": "One more line please." }]
    }))
    .await
    .unwrap()
    .text()
    .await
    .unwrap();
    assert!(body.contains("[DONE]"), "logged streaming chat still terminates");

    // A stop-sequence request exercises the stop-handling branch; the result must still be
    // a well-formed completion with a known finish_reason.
    let resp: Value = chat(json!({
        "model": id, "stream": false, "max_tokens": 16,
        "messages": [{ "role": "user", "content": "Say: one two three four five." }],
        "stop": [" "]
    }))
    .await
    .unwrap()
    .json()
    .await
    .unwrap();
    assert!(
        matches!(
            resp["choices"][0]["finish_reason"].as_str(),
            Some("stop") | Some("length")
        ),
        "stop-sequence chat has a known finish_reason: {resp:?}"
    );
}
