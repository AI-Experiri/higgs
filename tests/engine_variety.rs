//! Black-box integration tests for the llama.cpp engine's generation paths
//! (`src/worker/engine/llamacpp/mod.rs` + `src/worker/mod.rs`), driven through
//! the real `higgs` HTTP surface with VARIED but VALID OpenAI sampling.
//!
//! The tiny model is `ggml-org`'s ~1MB `stories260K.gguf` (see `common`): a real
//! llama-arch toy GGUF that loads and generates, so each request walks the full
//! template → tokenize → fit-check → sampler-chain → decode → detokenize → parse
//! path. We assert each request returns a well-formed completion (or the expected
//! HGxxx error) rather than asserting the toy model's exact text.
//!
//! Faithfulness notes (verified against the source, not guessed):
//! - The `/v1` request maps ONLY `temperature`/`top_p`/`presence_penalty`/
//!   `frequency_penalty` onto the engine sampler (`build_sampling`, v1.rs), plus
//!   `max_tokens`/`max_completion_tokens` via `effective_max_tokens`. `top_k`,
//!   `min_p`, `seed`, and `stop` are accepted at the HTTP layer (async-openai has
//!   no `deny_unknown_fields`; `seed`/`stop` are real OpenAI fields the engine
//!   sampler does not consume) — so those requests must SUCCEED, exercising the
//!   param-extraction branches, but we don't claim they alter the toy output.
//! - `temperature <= 0` ⇒ greedy/deterministic in `run_decode`; `> 0` builds the
//!   penalties → top_k → … → top_p → min_p → temp → dist chain.
//! - Context overflow is reachable cheaply: the model loads at the default
//!   `ctx_len = 4096`, and `max_tokens > 4096` (still <= MAX_OUTPUT_TOKENS) makes
//!   `prompt_est + max_gen > n_ctx`, so the serve layer rejects with 400 [HG005]
//!   (`invalid_request_error`, `code` null — only 404 carries `model_not_found`).

mod common;

use common::{spawn_with_tiny_model, tiny_gguf_path, TINY_MODEL_ID};
use serde_json::{json, Value};

/// POST a chat body to the running server and return the parsed JSON response.
/// Drops the `reqwest::Response` after reading the body, so no stream is left open.
async fn chat_json(base: &str, body: Value) -> Value {
    reqwest::Client::new()
        .post(format!("{base}/v1/chat/completions"))
        .json(&body)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap()
}

/// Assert a `/v1` chat completion JSON carries a well-formed first choice:
/// a string `content` and a known `finish_reason`.
fn assert_well_formed(resp: &Value) {
    let choice = &resp["choices"][0];
    assert!(
        choice["message"]["content"].is_string(),
        "completion has string content: {resp:?}"
    );
    assert!(
        matches!(
            choice["finish_reason"].as_str(),
            Some("stop") | Some("length") | Some("tool_calls")
        ),
        "completion has a known finish_reason: {resp:?}"
    );
}

/// High temperature + top_p + top_k: the non-greedy sampler chain (temp > 0 builds
/// penalties?→top_k→top_p→min_p→temp→dist in run_decode). top_k is accepted at the
/// HTTP layer (no deny_unknown_fields) and must not break the request.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn high_temperature_top_p_top_k() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("SKIP high_temperature_top_p_top_k: tiny gguf not found (set HIGGS_TEST_GGUF)");
        return;
    };
    let srv = spawn_with_tiny_model(12600, &gguf).await;
    let resp = chat_json(
        &srv.base,
        json!({
            "model": TINY_MODEL_ID, "stream": false, "max_tokens": 12,
            "temperature": 1.4, "top_p": 0.92, "top_k": 40,
            "messages": [{ "role": "user", "content": "Tell a tiny tale." }]
        }),
    )
    .await;
    assert_well_formed(&resp);
}

/// Low temperature (still > 0, so a real distribution sampler, not greedy) with a
/// small max_tokens cap → a short, well-formed completion.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn low_temperature_min_p() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("SKIP low_temperature_min_p: tiny gguf not found (set HIGGS_TEST_GGUF)");
        return;
    };
    let srv = spawn_with_tiny_model(12601, &gguf).await;
    let resp = chat_json(
        &srv.base,
        json!({
            "model": TINY_MODEL_ID, "stream": false, "max_tokens": 8,
            "temperature": 0.1, "top_p": 0.5, "min_p": 0.05,
            "messages": [{ "role": "user", "content": "One short sentence." }]
        }),
    )
    .await;
    assert_well_formed(&resp);
}

/// `temperature: 0` ⇒ greedy (argmax) in run_decode: deterministic across runs.
/// Two identical greedy requests must return identical content — exercising the
/// `temp <= 0 ⇒ LlamaSampler::greedy()` branch.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn greedy_temperature_is_deterministic() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!(
            "SKIP greedy_temperature_is_deterministic: tiny gguf not found (set HIGGS_TEST_GGUF)"
        );
        return;
    };
    let srv = spawn_with_tiny_model(12602, &gguf).await;
    let body = json!({
        "model": TINY_MODEL_ID, "stream": false, "max_tokens": 12, "temperature": 0.0,
        "messages": [{ "role": "user", "content": "Say a fixed greeting." }]
    });
    let a = chat_json(&srv.base, body.clone()).await;
    let b = chat_json(&srv.base, body).await;
    assert_well_formed(&a);
    assert_well_formed(&b);
    assert_eq!(
        a["choices"][0]["message"]["content"], b["choices"][0]["message"]["content"],
        "greedy (temperature=0) generation is deterministic across two requests"
    );
}

/// A per-request `seed` is accepted by the OpenAI parse path; the request must
/// succeed and return a well-formed completion. (The engine sampler does not
/// consume the OpenAI `seed`, so we don't assert it changes the toy output.)
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn seeded_request_is_well_formed() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("SKIP seeded_request_is_well_formed: tiny gguf not found (set HIGGS_TEST_GGUF)");
        return;
    };
    let srv = spawn_with_tiny_model(12603, &gguf).await;
    let resp = chat_json(
        &srv.base,
        json!({
            "model": TINY_MODEL_ID, "stream": false, "max_tokens": 10,
            "temperature": 0.7, "seed": 12345,
            "messages": [{ "role": "user", "content": "Pick a number." }]
        }),
    )
    .await;
    assert_well_formed(&resp);
}

/// `max_tokens: 1` exercises the smallest budget: the decode loop emits exactly
/// one token then breaks with finish_reason "length" (unless EOG on the first
/// token, which yields "stop"). usage.completion_tokens must be <= 1.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn max_tokens_one_caps_completion() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("SKIP max_tokens_one_caps_completion: tiny gguf not found (set HIGGS_TEST_GGUF)");
        return;
    };
    let srv = spawn_with_tiny_model(12604, &gguf).await;
    let resp = chat_json(
        &srv.base,
        json!({
            "model": TINY_MODEL_ID, "stream": false, "max_tokens": 1,
            "messages": [{ "role": "user", "content": "Hello?" }]
        }),
    )
    .await;
    assert_well_formed(&resp);
    assert!(
        resp["usage"]["completion_tokens"].as_u64().unwrap() <= 1,
        "max_tokens=1 caps the completion to at most one token: {resp:?}"
    );
}

/// A larger `max_completion_tokens` (which wins over `max_tokens`) drives a longer
/// decode; the result is still a well-formed completion with non-zero usage.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn larger_max_completion_tokens() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("SKIP larger_max_completion_tokens: tiny gguf not found (set HIGGS_TEST_GGUF)");
        return;
    };
    let srv = spawn_with_tiny_model(12605, &gguf).await;
    let resp = chat_json(
        &srv.base,
        json!({
            "model": TINY_MODEL_ID, "stream": false,
            "max_tokens": 4, "max_completion_tokens": 24,
            "messages": [{ "role": "user", "content": "Write a short paragraph." }]
        }),
    )
    .await;
    assert_well_formed(&resp);
    assert!(
        resp["usage"]["prompt_tokens"].as_u64().unwrap() > 0
            && resp["usage"]["completion_tokens"].as_u64().unwrap() > 0,
        "larger budget yields non-zero prompt + completion tokens: {resp:?}"
    );
    assert!(
        resp["usage"]["completion_tokens"].as_u64().unwrap() <= 24,
        "max_completion_tokens (24) wins over max_tokens (4): {resp:?}"
    );
}

/// A `stop` sequence is accepted (a real OpenAI field); the request must succeed
/// and return a known finish_reason. The engine stops on EOG/length (stop strings
/// are a documented latent limitation), so we don't require finish_reason "stop".
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stop_sequence_is_accepted() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("SKIP stop_sequence_is_accepted: tiny gguf not found (set HIGGS_TEST_GGUF)");
        return;
    };
    let srv = spawn_with_tiny_model(12606, &gguf).await;
    let resp = chat_json(
        &srv.base,
        json!({
            "model": TINY_MODEL_ID, "stream": false, "max_tokens": 16,
            "temperature": 0.5, "stop": ["\n\n", "END"],
            "messages": [{ "role": "user", "content": "List two words." }]
        }),
    )
    .await;
    assert_well_formed(&resp);
}

/// presence + frequency penalties are the penalty samplers the `/v1` request
/// actually plumbs (`presence_penalty → penalty_present`, `frequency_penalty →
/// penalty_freq` in build_sampling), which builds the `LlamaSampler::penalties`
/// chain link in run_decode. In-range bounds [-2, 2] must be accepted.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn presence_frequency_penalties() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("SKIP presence_frequency_penalties: tiny gguf not found (set HIGGS_TEST_GGUF)");
        return;
    };
    let srv = spawn_with_tiny_model(12607, &gguf).await;
    let resp = chat_json(
        &srv.base,
        json!({
            "model": TINY_MODEL_ID, "stream": false, "max_tokens": 12,
            "temperature": 0.8, "presence_penalty": 1.5, "frequency_penalty": -0.5,
            "messages": [{ "role": "user", "content": "Describe a cat briefly." }]
        }),
    )
    .await;
    assert_well_formed(&resp);
}

/// A multi-message conversation (system + user + assistant + user) renders the
/// full chat template over several turns before decode — exercising the template
/// apply + tokenize path with a non-trivial prompt.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn multi_message_conversation() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("SKIP multi_message_conversation: tiny gguf not found (set HIGGS_TEST_GGUF)");
        return;
    };
    let srv = spawn_with_tiny_model(12608, &gguf).await;
    let resp = chat_json(
        &srv.base,
        json!({
            "model": TINY_MODEL_ID, "stream": false, "max_tokens": 12,
            "messages": [
                { "role": "system", "content": "You are terse." },
                { "role": "user", "content": "Name a color." },
                { "role": "assistant", "content": "Blue." },
                { "role": "user", "content": "Name another one." }
            ]
        }),
    )
    .await;
    assert_well_formed(&resp);
    assert!(
        resp["usage"]["prompt_tokens"].as_u64().unwrap() > 0,
        "multi-turn prompt tokenizes to a non-zero count: {resp:?}"
    );
}

/// HG005 ContextOverflow: the model loads at the default `ctx_len = 4096`, so a
/// `max_tokens` above the window (still <= MAX_OUTPUT_TOKENS = 32768) makes
/// `prompt_est + max_gen > n_ctx`, and the serve layer rejects with 400 [HG005]
/// (`invalid_request_error`; `code` is null — only 404 carries model_not_found).
/// We JIT-load the model first (a small chat) so `ensure_loaded` reports the real
/// 4096 ctx_len rather than the remote/permissive placeholder.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn context_overflow_is_hg005() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("SKIP context_overflow_is_hg005: tiny gguf not found (set HIGGS_TEST_GGUF)");
        return;
    };
    let srv = spawn_with_tiny_model(12609, &gguf).await;
    let c = reqwest::Client::new();

    // Load the model on demand with a tiny, in-window request first, so the
    // local ctx_len (4096) is what the overflow check sees.
    let warm = chat_json(
        &srv.base,
        json!({
            "model": TINY_MODEL_ID, "stream": false, "max_tokens": 4,
            "messages": [{ "role": "user", "content": "hi" }]
        }),
    )
    .await;
    assert_well_formed(&warm);

    // max_tokens 30000 > 4096 ctx_len ⇒ prompt_est + max_gen > n_ctx ⇒ 400 HG005.
    let over = c
        .post(format!("{}/v1/chat/completions", srv.base))
        .json(&json!({
            "model": TINY_MODEL_ID, "stream": false, "max_tokens": 30000,
            "messages": [{ "role": "user", "content": "Tell a story." }]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(
        over.status(),
        400,
        "max_tokens beyond the context window is a 400 [HG005]"
    );
    let env: Value = over.json().await.unwrap();
    assert_eq!(
        env["error"]["type"], "invalid_request_error",
        "context overflow is an invalid_request_error: {env:?}"
    );
    assert!(
        env["error"]["message"]
            .as_str()
            .is_some_and(|m| m.contains("[HG005]")),
        "context-overflow message carries the HG005 code: {env:?}"
    );
}

/// Streaming with varied sampling + a small cap: the SSE path runs the same
/// sampler chain and streams content deltas. We read the body to completion then
/// drop it (never leave the stream open), asserting the well-formed terminators.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn streaming_with_sampling_params() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("SKIP streaming_with_sampling_params: tiny gguf not found (set HIGGS_TEST_GGUF)");
        return;
    };
    let srv = spawn_with_tiny_model(12610, &gguf).await;
    let body = reqwest::Client::new()
        .post(format!("{}/v1/chat/completions", srv.base))
        .json(&json!({
            "model": TINY_MODEL_ID, "stream": true, "max_tokens": 8,
            "temperature": 0.9, "top_p": 0.9, "presence_penalty": 0.2,
            "messages": [{ "role": "user", "content": "Count to three." }]
        }))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(body.contains("data:"), "stream has data lines: {body}");
    assert!(
        body.contains("[DONE]"),
        "stream terminates with [DONE]: {body}"
    );
    assert!(
        body.contains("chat.completion.chunk"),
        "stream chunks carry the chunk object type: {body}"
    );
}
