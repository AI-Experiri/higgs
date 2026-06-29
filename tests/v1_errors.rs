//! Black-box integration tests for the OpenAI `/v1` ERROR + edge paths.
//!
//! Spawns the real `higgs` and drives the `/v1/chat/completions` + `/v1/models`
//! surface over HTTP, exercising the request-validation and error branches in
//! `src/serve/v1.rs` and the SSE assembly in `src/serve/stream.rs`:
//!   * out-of-range sampling params  → 400 `[HG013]` invalid_request_error
//!   * unknown model (JIT on)        → 404 `model_not_found` `[HG002]`
//!   * malformed / empty JSON body   → 4xx (no 5xx, no panic)
//!   * missing required `model`      → 4xx
//!   * `stream: true` happy path     → SSE deltas drained to `[DONE]`
//!   * `stream: false` happy path    → full completion envelope
//!   * `GET /v1/models`              → OpenAI shape, lists the loaded model
//!
//! The tiny model is `ggml-org`'s ~1MB `stories260K.gguf` (see `common`). Each
//! test picks a UNIQUE port from the 12100 base range and SIGTERM-reaps the
//! server on drop. SSE streams are read to completion then dropped — never left
//! open (an open stream blocks graceful shutdown).

mod common;

use common::{spawn_with_tiny_model, tiny_gguf_path, TINY_MODEL_ID};
use serde_json::{json, Value};

/// Explicitly load the tiny model via the control surface so the subsequent
/// chat request's failing gate is unambiguously the one under test (not a JIT
/// load). Idempotent when already resident.
async fn load_tiny(client: &reqwest::Client, base: &str) {
    let load = client
        .post(format!("{base}/api/higgs/models/load"))
        .json(&json!({ "id": TINY_MODEL_ID }))
        .send()
        .await
        .unwrap();
    assert!(
        load.status().is_success(),
        "tiny model load succeeded, got {}",
        load.status()
    );
}

// ── Out-of-range sampling param → 400 [HG013] invalid_request_error ──────────
//
// `top_p` must be in (0, 1]; 2.0 is out of range. The model is loaded first so
// the sampling-validation gate (which runs AFTER the loaded-model gate in
// `gate_and_validate`) is the unambiguous point of failure.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn invalid_sampling_param_top_p_400_hg013() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("SKIP invalid_sampling_param_top_p_400_hg013: tiny gguf not found");
        return;
    };
    let srv = spawn_with_tiny_model(12100, &gguf).await;
    let c = reqwest::Client::new();
    load_tiny(&c, &srv.base).await;

    let resp = c
        .post(format!("{}/v1/chat/completions", srv.base))
        .json(&json!({
            "model": TINY_MODEL_ID, "stream": false,
            "top_p": 2.0,
            "messages": [{ "role": "user", "content": "hi" }]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        400,
        "top_p out of (0,1] is a 400 invalid_request_error"
    );
    let env: Value = resp.json().await.unwrap();
    assert_eq!(
        env["error"]["type"], "invalid_request_error",
        "400 envelope type is invalid_request_error: {env:?}"
    );
    let msg = env["error"]["message"].as_str().unwrap_or_default();
    assert!(msg.contains("[HG013]"), "message carries HG013: {env:?}");
    assert!(
        msg.contains("top_p"),
        "message names the offending param: {env:?}"
    );
    // On a 4xx the OpenAI envelope `code` is reserved for model_not_found (404).
    assert!(
        env["error"]["code"].is_null(),
        "HG013 4xx carries no `code`: {env:?}"
    );
}

// ── A second out-of-range param: presence_penalty must be in [-2, 2] ─────────
//
// Distinct sampling-validation branch from top_p, asserting the same 400 [HG013]
// envelope shape so the param-range table in `validate_sampling` is exercised.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn invalid_sampling_param_presence_penalty_400_hg013() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("SKIP invalid_sampling_param_presence_penalty_400_hg013: tiny gguf not found");
        return;
    };
    let srv = spawn_with_tiny_model(12101, &gguf).await;
    let c = reqwest::Client::new();
    load_tiny(&c, &srv.base).await;

    let resp = c
        .post(format!("{}/v1/chat/completions", srv.base))
        .json(&json!({
            "model": TINY_MODEL_ID, "stream": false,
            "presence_penalty": 5.0,
            "messages": [{ "role": "user", "content": "hi" }]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        400,
        "presence_penalty out of [-2,2] is a 400"
    );
    let env: Value = resp.json().await.unwrap();
    assert_eq!(env["error"]["type"], "invalid_request_error");
    let msg = env["error"]["message"].as_str().unwrap_or_default();
    assert!(msg.contains("[HG013]"), "message carries HG013: {env:?}");
    assert!(
        msg.contains("presence_penalty"),
        "message names the offending param: {env:?}"
    );
}

// ── n != 1 is rejected (higgs serves a single choice) → 400 [HG013] ──────────
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn invalid_sampling_param_n_not_one_400_hg013() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("SKIP invalid_sampling_param_n_not_one_400_hg013: tiny gguf not found");
        return;
    };
    let srv = spawn_with_tiny_model(12102, &gguf).await;
    let c = reqwest::Client::new();
    load_tiny(&c, &srv.base).await;

    let resp = c
        .post(format!("{}/v1/chat/completions", srv.base))
        .json(&json!({
            "model": TINY_MODEL_ID, "stream": false,
            "n": 2,
            "messages": [{ "role": "user", "content": "hi" }]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400, "n>1 is unsupported → 400");
    let env: Value = resp.json().await.unwrap();
    assert_eq!(env["error"]["type"], "invalid_request_error");
    let msg = env["error"]["message"].as_str().unwrap_or_default();
    assert!(msg.contains("[HG013]"), "message carries HG013: {env:?}");
    assert!(msg.contains('n'), "message references n: {env:?}");
}

// ── max_tokens above the cap → 400 [HG013] ───────────────────────────────────
//
// `max_tokens` must be <= MAX_OUTPUT_TOKENS (32_768); above it is a 400. This
// validation runs without needing a loaded model only AFTER the loaded gate, so
// the model is loaded first.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn max_tokens_over_cap_400_hg013() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("SKIP max_tokens_over_cap_400_hg013: tiny gguf not found");
        return;
    };
    let srv = spawn_with_tiny_model(12103, &gguf).await;
    let c = reqwest::Client::new();
    load_tiny(&c, &srv.base).await;

    let resp = c
        .post(format!("{}/v1/chat/completions", srv.base))
        .json(&json!({
            "model": TINY_MODEL_ID, "stream": false,
            "max_tokens": 1_000_000,
            "messages": [{ "role": "user", "content": "hi" }]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400, "max_tokens over the cap is a 400");
    let env: Value = resp.json().await.unwrap();
    assert_eq!(env["error"]["type"], "invalid_request_error");
    assert!(
        env["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("[HG013]"),
        "message carries HG013: {env:?}"
    );
}

// ── Unknown model (JIT on default) → 404 model_not_found [HG002] ─────────────
//
// JIT auto-load is on by default, but JIT only loads a SCANNED id; an id absent
// from the scan is a `model_not_found` 404, NEVER a load attempt. The envelope
// codes `model_not_found` (the only coded case on this surface) and the message
// carries [HG002] (distinct from the explicit-load [HG003]).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unknown_model_404_model_not_found() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("SKIP unknown_model_404_model_not_found: tiny gguf not found");
        return;
    };
    let srv = spawn_with_tiny_model(12104, &gguf).await;
    let c = reqwest::Client::new();

    let resp = c
        .post(format!("{}/v1/chat/completions", srv.base))
        .json(&json!({
            "model": "no-such-org/no-such-model", "stream": false,
            "messages": [{ "role": "user", "content": "hi" }]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404, "unknown model is a 404");
    let env: Value = resp.json().await.unwrap();
    assert_eq!(
        env["error"]["code"], "model_not_found",
        "404 envelope codes model_not_found: {env:?}"
    );
    assert_eq!(
        env["error"]["type"], "invalid_request_error",
        "404 is an invalid_request_error: {env:?}"
    );
    assert!(
        env["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("[HG002]"),
        "JIT unknown-id message carries HG002 (not HG003): {env:?}"
    );
}

// ── Malformed body branches: empty body, invalid JSON, missing `model` ───────
//
// All three must be a 4xx caught at the boundary (axum JSON extractor or the
// handler's own validation) — never a 5xx or a panic. Exercises the
// request-rejection edges around `v1_chat_completions`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn malformed_and_missing_field_bodies_are_4xx() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("SKIP malformed_and_missing_field_bodies_are_4xx: tiny gguf not found");
        return;
    };
    let srv = spawn_with_tiny_model(12105, &gguf).await;
    let c = reqwest::Client::new();
    let url = format!("{}/v1/chat/completions", srv.base);

    // Empty body with a JSON content-type → the extractor rejects it 4xx.
    let empty = c
        .post(&url)
        .header("content-type", "application/json")
        .body("")
        .send()
        .await
        .unwrap();
    assert!(
        empty.status().is_client_error(),
        "empty body is a 4xx, got {}",
        empty.status()
    );

    // Syntactically invalid JSON → 4xx (no 5xx, no panic).
    let invalid = c
        .post(&url)
        .header("content-type", "application/json")
        .body("{not valid json")
        .send()
        .await
        .unwrap();
    assert!(
        invalid.status().is_client_error(),
        "invalid JSON is a 4xx, got {}",
        invalid.status()
    );

    // Well-formed JSON but missing the required `model` field → 4xx.
    let missing_model = c
        .post(&url)
        .json(&json!({ "messages": [{ "role": "user", "content": "hi" }] }))
        .send()
        .await
        .unwrap();
    assert!(
        missing_model.status().is_client_error(),
        "missing `model` is a 4xx, got {}",
        missing_model.status()
    );

    // Well-formed JSON but missing the required `messages` field → 4xx.
    let missing_messages = c
        .post(&url)
        .json(&json!({ "model": TINY_MODEL_ID }))
        .send()
        .await
        .unwrap();
    assert!(
        missing_messages.status().is_client_error(),
        "missing `messages` is a 4xx, got {}",
        missing_messages.status()
    );
}

// ── stream: true happy path — drain the SSE to [DONE], then drop ─────────────
//
// Reads the FULL SSE body to a String (which closes the stream) so it's never
// left open, then asserts the assistant-role chunk, content deltas, finish, and
// the trailing [DONE] — the framing produced by `stream::chat_sse` / `assemble`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stream_true_happy_path_drains_to_done() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("SKIP stream_true_happy_path_drains_to_done: tiny gguf not found");
        return;
    };
    let srv = spawn_with_tiny_model(12106, &gguf).await;
    let c = reqwest::Client::new();
    load_tiny(&c, &srv.base).await;

    // `.text()` reads the body to completion and drops the response — the stream
    // is never left open (which would block graceful shutdown).
    let body = c
        .post(format!("{}/v1/chat/completions", srv.base))
        .json(&json!({
            "model": TINY_MODEL_ID, "stream": true, "max_tokens": 8,
            "messages": [{ "role": "user", "content": "Count to three." }]
        }))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();

    assert!(body.contains("data:"), "stream has data lines: {body}");
    assert!(body.contains("[DONE]"), "stream terminates with [DONE]");
    assert!(
        body.contains("chat.completion.chunk"),
        "chunks carry the chunk object type"
    );

    // The first data chunk announces the assistant role; at least one carries a
    // content delta. Parse the framed `data:` payloads (skipping [DONE]).
    let chunks: Vec<Value> = body
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter(|p| *p != "[DONE]")
        .filter_map(|p| serde_json::from_str::<Value>(p).ok())
        .collect();
    assert!(!chunks.is_empty(), "at least one parseable chunk: {body}");
    assert_eq!(
        chunks[0]["choices"][0]["delta"]["role"], "assistant",
        "first chunk announces the assistant role: {body}"
    );
    assert!(
        chunks
            .iter()
            .any(|ch| ch["choices"][0]["delta"]["content"].is_string()),
        "at least one content delta: {body}"
    );
    assert!(
        chunks.iter().any(|ch| matches!(
            ch["choices"][0]["finish_reason"].as_str(),
            Some("stop") | Some("length")
        )),
        "a finish chunk carries a known reason: {body}"
    );
}

// ── stream: true streaming AND stream: false non-streaming, same server ──────
//
// Drives both the SSE path and the buffered non-streaming path against one
// loaded model so the two `req.stream` branches of `v1_chat_completions` are
// exercised side by side. The stream is read to completion (closed) before the
// non-streaming request.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stream_and_nonstream_both_succeed() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("SKIP stream_and_nonstream_both_succeed: tiny gguf not found");
        return;
    };
    let srv = spawn_with_tiny_model(12107, &gguf).await;
    let c = reqwest::Client::new();
    load_tiny(&c, &srv.base).await;

    // Streaming branch — drained to [DONE], stream closed by `.text()`.
    let stream_body = c
        .post(format!("{}/v1/chat/completions", srv.base))
        .json(&json!({
            "model": TINY_MODEL_ID, "stream": true, "max_tokens": 8,
            "messages": [{ "role": "user", "content": "Say hi." }]
        }))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(
        stream_body.contains("[DONE]"),
        "streaming branch terminates with [DONE]: {stream_body}"
    );

    // Non-streaming branch — a full chat.completion envelope.
    let resp: Value = c
        .post(format!("{}/v1/chat/completions", srv.base))
        .json(&json!({
            "model": TINY_MODEL_ID, "stream": false, "max_tokens": 8,
            "messages": [{ "role": "user", "content": "Say hi." }]
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        resp["object"], "chat.completion",
        "non-streaming envelope object type: {resp:?}"
    );
    assert!(
        resp["id"]
            .as_str()
            .is_some_and(|s| s.starts_with("chatcmpl-")),
        "non-streaming id is a chatcmpl id: {resp:?}"
    );
    assert!(
        resp["choices"][0]["message"]["content"].is_string(),
        "non-streaming returns content: {resp:?}"
    );
    assert!(
        matches!(
            resp["choices"][0]["finish_reason"].as_str(),
            Some("stop") | Some("length")
        ),
        "non-streaming finish_reason is stop|length: {resp:?}"
    );
    assert!(
        resp["usage"]["total_tokens"].as_u64().unwrap_or(0) > 0,
        "non-streaming reports real token usage: {resp:?}"
    );
}

// ── GET /v1/models returns the loaded model in OpenAI shape ──────────────────
//
// Empty before any load (the correct OpenAI answer for "nothing can serve chat
// now"), then after an explicit load the model appears with the OpenAI `Model`
// shape: id, object:"model", owned_by:"higgs", numeric created.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn v1_models_openai_shape_after_load() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("SKIP v1_models_openai_shape_after_load: tiny gguf not found");
        return;
    };
    let srv = spawn_with_tiny_model(12108, &gguf).await;
    let c = reqwest::Client::new();

    // Empty list before any load.
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
        "/v1/models is empty with nothing loaded: {empty:?}"
    );

    load_tiny(&c, &srv.base).await;

    let listed: Value = c
        .get(format!("{}/v1/models", srv.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(listed["object"], "list");
    let data = listed["data"].as_array().unwrap();
    let m = data
        .iter()
        .find(|m| m["id"] == json!(TINY_MODEL_ID))
        .unwrap_or_else(|| panic!("/v1/models lists the loaded model: {listed:?}"));
    assert_eq!(m["object"], "model", "OpenAI model object type: {m:?}");
    assert_eq!(m["owned_by"], "higgs", "owned_by is higgs: {m:?}");
    assert!(
        m["created"].as_u64().is_some(),
        "created is a numeric unix timestamp: {m:?}"
    );
}

/// Load the tiny model with an EXPLICIT small context window so the overflow / clamp
/// behavior is deterministic regardless of the model's trained window.
async fn load_tiny_with_ctx(client: &reqwest::Client, base: &str, n: u32) {
    let load = client
        .post(format!("{base}/api/higgs/models/load"))
        .json(&json!({
            "id": TINY_MODEL_ID,
            "params": { "engine": "LlamaCpp", "ctx_len": { "kind": "fixed", "n": n } }
        }))
        .send()
        .await
        .unwrap();
    assert!(
        load.status().is_success(),
        "load with ctx {n} succeeded, got {}",
        load.status()
    );
}

// ── Step 5: prompt that ALONE overflows → 400 `context_length_exceeded` ───────────
//
// A prompt larger than the window can't generate anything → the genuine overflow,
// surfaced with the OpenAI-standard `context_length_exceeded` code (was `null`).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn prompt_overflow_400_context_length_exceeded() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("SKIP prompt_overflow: tiny gguf not found");
        return;
    };
    let srv = spawn_with_tiny_model(12110, &gguf).await;
    let c = reqwest::Client::new();
    load_tiny_with_ctx(&c, &srv.base, 256).await;

    // ~5000 estimated tokens (20000 chars / 4 bytes) ≫ the 256-token window.
    let resp = c
        .post(format!("{}/v1/chat/completions", srv.base))
        .json(&json!({
            "model": TINY_MODEL_ID, "stream": false,
            "messages": [{ "role": "user", "content": "x".repeat(20000) }]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400, "an over-window prompt is a 400");
    let env: Value = resp.json().await.unwrap();
    assert_eq!(
        env["error"]["code"], "context_length_exceeded",
        "the standard OpenAI code, not null: {env:?}"
    );
}

// ── Step 5: oversized max_tokens is CLAMPED, not rejected ─────────────────────────
//
// A small prompt with a `max_tokens` far larger than the window used to 400 [HG005].
// Now the budget is clamped (serve layer AND the worker's tokenizer-exact backstop)
// so the request succeeds and generates, truncating at the window.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn oversized_max_tokens_is_clamped_not_rejected() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("SKIP clamp: tiny gguf not found");
        return;
    };
    let srv = spawn_with_tiny_model(12111, &gguf).await;
    let c = reqwest::Client::new();
    load_tiny_with_ctx(&c, &srv.base, 256).await;

    let resp = c
        .post(format!("{}/v1/chat/completions", srv.base))
        .json(&json!({
            "model": TINY_MODEL_ID, "stream": false,
            "max_tokens": 16384, // ≫ the 256 window: OLD → 400; NEW → clamp → 200
            "messages": [{ "role": "user", "content": "hi" }]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        200,
        "an oversized max_tokens is clamped to fit, not rejected"
    );
    let body: Value = resp.json().await.unwrap();
    assert!(
        body["choices"][0]["message"].is_object(),
        "a completion was generated: {body:?}"
    );
    // It truncated at the window (didn't run the full 16384) → finish_reason length.
    assert_eq!(
        body["choices"][0]["finish_reason"], "length",
        "generation truncated at the window: {body:?}"
    );
}
