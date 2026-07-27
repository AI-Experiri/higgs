//! In-process integration test for higgs's OpenAI `/v1/*` inference surface.
//!
//! higgs is a library: `/v1` is the ONLY HTTP surface it serves. This test binds
//! the real `serve_v1` router on an ephemeral loopback port (`serve_v1_local`) over
//! an in-process `Higgs` whose worker runs REAL llama.cpp via the `worker_exe` seam,
//! then drives chat (non-streaming + streaming), `/v1/models`, and the
//! request-validation paths exactly as an external client would. The three CONTROL
//! steps (prepare, explicit reload, token-logging toggle) run through the in-process
//! facade — the `/api/higgs/*` HTTP surface is gone.

mod common;

use common::{higgs_local, serve_v1_local, TINY_MODEL_ID};
use higgs::{LogSettings, TuneRequest};
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
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP inference_and_tools: tiny gguf not found (set HIGGS_TEST_GGUF)");
        return;
    };
    let id = TINY_MODEL_ID;

    // Prepare (tune) so a subsequent JIT chat is allowed by the readiness gate.
    higgs
        .tune(TuneRequest {
            id: id.to_owned(),
            mode: None,
            budget: None,
            pins: None,
        })
        .await
        .expect("prepare (tune) the tiny model");

    // The staged tiny model is discoverable via the live scan (control facade).
    let entries = higgs.model_entries().await.expect("model_entries");
    assert!(
        entries.iter().any(|m| m.model.id == id),
        "scan lists the staged tiny model"
    );

    // Serve the real `/v1` router on an ephemeral loopback port.
    let (base, guard) = serve_v1_local(higgs.handle()).await;
    let c = reqwest::Client::new();

    // ── /v1/models lists the SERVABLE (prepared, unloaded) tiny model ─────────
    let listed: Value = c
        .get(format!("{base}/v1/models"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(listed["object"], "list", "/v1/models is a list");
    assert!(
        listed["data"]
            .as_array()
            .unwrap()
            .iter()
            .any(|m| m["id"] == json!(id)),
        "servable tiny model is advertised before any load: {listed}"
    );

    // ── Chat for an UNKNOWN model id → 404 OpenAI error envelope ──────────────
    let unknown = c
        .post(format!("{base}/v1/chat/completions"))
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
    let jit: Value = c
        .post(format!("{base}/v1/chat/completions"))
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

    // Explicitly (re)load via the CONTROL facade — idempotent when already
    // resident; keeps the rest of the test on the explicit-load contract.
    higgs.load(id, None).await.expect("model load succeeded");

    // /v1/models lists the loaded model.
    let v1models: Value = c
        .get(format!("{base}/v1/models"))
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
        c.post(format!("{base}/v1/chat/completions"))
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
        .post(format!("{base}/v1/chat/completions"))
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
        .post(format!("{base}/v1/chat/completions"))
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
    assert!(
        body.contains("[DONE]"),
        "tool stream terminates with [DONE]"
    );
    assert!(
        !body.contains("<tool_call>") && !body.contains("<function="),
        "no tool-call markup leaks into the stream"
    );

    // ── Sampling-parameter passthrough (non-streaming) ────────────────────────
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
    assert!(
        body.contains("[DONE]"),
        "usage stream terminates with [DONE]"
    );
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
        .post(format!("{base}/v1/chat/completions"))
        .json(&json!({
            "model": "no-such-org/no-such-model", "stream": true,
            "messages": [{ "role": "user", "content": "hi" }]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(
        unknown_stream.status(),
        404,
        "streaming unknown model is 404 (pre-stream error)"
    );

    // ── Zero max_tokens exercises the boundary branch (clamped or rejected, never 5xx) ──
    let zero = c
        .post(format!("{base}/v1/chat/completions"))
        .json(&json!({
            "model": id, "stream": false, "max_tokens": 0,
            "messages": [{ "role": "user", "content": "hi" }]
        }))
        .send()
        .await
        .unwrap();
    assert!(
        !zero.status().is_server_error(),
        "max_tokens=0 must not 5xx, got {}",
        zero.status()
    );

    // ── Token logging ON + a tool-result conversation ─────────────────────────
    // Enable "Log Incoming Tokens" via the CONTROL facade, then exercise the
    // prompt/response logging path with a multi-turn tool-result history.
    higgs.set_logs_settings(&LogSettings {
        log_incoming_tokens: true,
        ..higgs.logs_settings()
    });

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
    assert!(
        resp.status().is_success(),
        "tool-result conversation succeeds: {}",
        resp.status()
    );

    // A STREAMING chat with token logging still on — exercises the streamed
    // response logging path (log_served on the SSE branch).
    let body = chat(json!({
        "model": id, "stream": true, "max_tokens": 8,
        "messages": [{ "role": "user", "content": "One more line please." }]
    }))
    .await
    .unwrap()
    .text()
    .await
    .unwrap();
    assert!(
        body.contains("[DONE]"),
        "logged streaming chat still terminates"
    );

    // A stop-sequence request exercises the stop-handling branch.
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

    // Graceful teardown: drain the server (never leave an SSE stream open), then
    // stop the facade's worker.
    guard.shutdown().await;
    higgs.shutdown().await;
}
