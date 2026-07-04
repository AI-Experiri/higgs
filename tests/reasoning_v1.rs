//! `reasoning_content` on `/v1/chat/completions` — final AND streaming — with
//! a REAL thinking model (Qwen3-0.6B, whose template has `<think>` markers).
//!
//! Fail-on-revert: with `reasoning_format: None` / `enable_thinking: false` at
//! template-apply (the pre-feature state), no `reasoning_content` key exists
//! anywhere and think text leaks inline into `content` — every assertion here
//! fails. Skips when the test fleet is absent (like `chat_fleet.rs`).

mod common;

use common::{fleet_dir, spawn_with_model_root};
use serde_json::{json, Value};

/// The fleet's thinking model (template contains `<think>` markers).
const QWEN3: &str = "qwen/Qwen3-0.6B";

/// Non-stream: the final message separates thinking from the answer.
#[tokio::test]
async fn reasoning_content_on_final_message() {
    let Some(root) = fleet_dir() else {
        eprintln!("SKIP reasoning_content_on_final_message: fleet absent");
        return;
    };
    let srv = spawn_with_model_root(13500, &root).await;
    let c = reqwest::Client::new();
    load(&c, &srv.base, QWEN3).await;

    // Deterministic, and small enough that generation may truncate mid-think —
    // the lenient parse must STILL yield reasoning_content (content may be "").
    let resp: Value = c
        .post(format!("{}/v1/chat/completions", srv.base))
        .json(&json!({
            "model": QWEN3, "stream": false, "max_tokens": 96, "temperature": 0.0,
            "messages": [{ "role": "user", "content": "What is 2+2?" }]
        }))
        .send()
        .await
        .expect("chat request")
        .json()
        .await
        .expect("chat json");

    let msg = &resp["choices"][0]["message"];
    let reasoning = msg["reasoning_content"].as_str().unwrap_or_default();
    assert!(
        !reasoning.trim().is_empty(),
        "thinking model yields non-empty reasoning_content: {resp}"
    );
    let content = msg["content"].as_str().unwrap_or_default();
    assert!(
        !content.contains("<think>") && !reasoning.contains("<think>"),
        "think markup never leaks into content or reasoning: {resp}"
    );
    // The wire stays OpenAI-shaped around the new field.
    assert_eq!(resp["object"], "chat.completion", "envelope intact: {resp}");
    assert!(
        resp["usage"]["completion_tokens"].as_u64().unwrap_or(0) > 0,
        "usage intact: {resp}"
    );
}

/// Streaming: reasoning arrives as `delta.reasoning_content` chunks BEFORE the
/// first content chunk, and think markup never rides `delta.content`.
#[tokio::test]
async fn reasoning_content_streams_as_deltas() {
    let Some(root) = fleet_dir() else {
        eprintln!("SKIP reasoning_content_streams_as_deltas: fleet absent");
        return;
    };
    let srv = spawn_with_model_root(13501, &root).await;
    let c = reqwest::Client::new();
    load(&c, &srv.base, QWEN3).await;

    let resp = c
        .post(format!("{}/v1/chat/completions", srv.base))
        .json(&json!({
            "model": QWEN3, "stream": true, "max_tokens": 96, "temperature": 0.0,
            "messages": [{ "role": "user", "content": "What is 2+2?" }]
        }))
        .send()
        .await
        .expect("chat request");
    assert!(resp.status().is_success(), "stream chat accepted");
    let payloads = collect_sse_payloads(resp).await;

    let mut reasoning = String::new();
    let mut content = String::new();
    let mut first_content_seen = false;
    let mut reasoning_after_content = false;
    for p in &payloads {
        let Ok(v) = serde_json::from_str::<Value>(p) else {
            continue; // "[DONE]"
        };
        let delta = &v["choices"][0]["delta"];
        if let Some(r) = delta["reasoning_content"].as_str() {
            reasoning.push_str(r);
            reasoning_after_content |= first_content_seen;
        }
        if let Some(t) = delta["content"].as_str() {
            if !t.is_empty() {
                first_content_seen = true;
            }
            content.push_str(t);
        }
    }
    assert!(
        !reasoning.trim().is_empty(),
        "thinking streamed as delta.reasoning_content chunks: {payloads:?}"
    );
    assert!(
        !content.contains("<think>") && !reasoning.contains("<think>"),
        "think markup never rides the stream: content={content:?} reasoning={reasoning:?}"
    );
    assert!(
        !reasoning_after_content,
        "reasoning chunks precede content chunks (generation order): {payloads:?}"
    );

    // Stream/non-stream parity: same server, same resident model, greedy
    // decode — the final message's reasoning_content must equal the streamed
    // reasoning deltas concatenated (both come from the same PEG parse; a
    // divergence means one path re-shapes text the other doesn't).
    let non_stream: Value = c
        .post(format!("{}/v1/chat/completions", srv.base))
        .json(&json!({
            "model": QWEN3, "stream": false, "max_tokens": 96, "temperature": 0.0,
            "messages": [{ "role": "user", "content": "What is 2+2?" }]
        }))
        .send()
        .await
        .expect("parity chat request")
        .json()
        .await
        .expect("parity chat json");
    let final_reasoning = non_stream["choices"][0]["message"]["reasoning_content"]
        .as_str()
        .unwrap_or_default();
    assert_eq!(
        final_reasoning.trim(),
        reasoning.trim(),
        "final reasoning_content equals concatenated stream deltas"
    );
}

/// Load a fleet model (JIT-ready) before chatting.
async fn load(c: &reqwest::Client, base: &str, id: &str) {
    let r = c
        .post(format!("{base}/api/higgs/models/load"))
        .json(&json!({ "id": id, "ctx_len": 2048 }))
        .send()
        .await
        .expect("load request");
    assert!(r.status().is_success(), "model load succeeds");
}

/// Drain one SSE body into its `data:` payload strings (stops at `[DONE]`, so
/// the stream is never left open — graceful-shutdown rule).
async fn collect_sse_payloads(resp: reqwest::Response) -> Vec<String> {
    use futures::StreamExt;
    let mut stream = resp.bytes_stream();
    let mut buf = String::new();
    let mut out = Vec::new();
    while let Some(chunk) = stream.next().await {
        buf.push_str(&String::from_utf8_lossy(&chunk.expect("sse chunk")));
        while let Some(nl) = buf.find('\n') {
            let line = buf[..nl].trim().to_owned();
            buf.drain(..=nl);
            if let Some(payload) = line.strip_prefix("data: ") {
                if payload == "[DONE]" {
                    return out;
                }
                out.push(payload.to_owned());
            }
        }
    }
    out
}

/// Per-request thinking OFF via llama.cpp-style `chat_template_kwargs`
/// (`{"enable_thinking": false}` — Qwen3's template prefills an empty think
/// block). Fail-on-revert for the kwargs threading: if the field stops
/// reaching the template apply, thinking happens and this fails.
#[tokio::test]
async fn thinking_off_via_chat_template_kwargs() {
    let Some(root) = fleet_dir() else {
        eprintln!("SKIP thinking_off_via_chat_template_kwargs: fleet absent");
        return;
    };
    let srv = spawn_with_model_root(13502, &root).await;
    let c = reqwest::Client::new();
    load(&c, &srv.base, QWEN3).await;

    let resp: Value = c
        .post(format!("{}/v1/chat/completions", srv.base))
        .json(&json!({
            "model": QWEN3, "stream": false, "max_tokens": 48, "temperature": 0.0,
            "chat_template_kwargs": { "enable_thinking": false },
            "messages": [{ "role": "user", "content": "What is 2+2?" }]
        }))
        .send()
        .await
        .expect("chat request")
        .json()
        .await
        .expect("chat json");

    let msg = &resp["choices"][0]["message"];
    assert!(
        msg["reasoning_content"].as_str().is_none_or(str::is_empty),
        "thinking disabled per-request yields no reasoning_content: {resp}"
    );
    assert!(
        msg["content"]
            .as_str()
            .is_some_and(|s| !s.trim().is_empty()),
        "non-thinking turn still answers: {resp}"
    );
}

/// Truncation mid-think (tiny max_tokens): the lenient parse keeps the partial
/// thinking as `reasoning_content`, `content` may be empty, and the finish
/// reason stays `length` — never an error. (The omlx alternative — folding the
/// unclosed block back into content — is explicitly NOT higgs's policy.)
#[tokio::test]
async fn truncation_mid_think_keeps_reasoning() {
    let Some(root) = fleet_dir() else {
        eprintln!("SKIP truncation_mid_think_keeps_reasoning: fleet absent");
        return;
    };
    let srv = spawn_with_model_root(13503, &root).await;
    let c = reqwest::Client::new();
    load(&c, &srv.base, QWEN3).await;

    let resp: Value = c
        .post(format!("{}/v1/chat/completions", srv.base))
        .json(&json!({
            "model": QWEN3, "stream": false, "max_tokens": 8, "temperature": 0.0,
            "messages": [{ "role": "user", "content": "What is 2+2?" }]
        }))
        .send()
        .await
        .expect("chat request")
        .json()
        .await
        .expect("chat json");

    let choice = &resp["choices"][0];
    assert_eq!(choice["finish_reason"], "length", "truncated: {resp}");
    assert!(
        choice["message"]["reasoning_content"]
            .as_str()
            .is_some_and(|s| !s.trim().is_empty()),
        "partial thinking survives truncation as reasoning_content: {resp}"
    );
}

/// Multi-turn echo-back: an assistant message carrying `reasoning_content`
/// (the DeepSeek/Kimi convention genai follows) is ACCEPTED and the raw field
/// reaches the template parser — the request must not 400 and the follow-up
/// turn must complete.
#[tokio::test]
async fn multi_turn_reasoning_echo_back_accepted() {
    let Some(root) = fleet_dir() else {
        eprintln!("SKIP multi_turn_reasoning_echo_back_accepted: fleet absent");
        return;
    };
    let srv = spawn_with_model_root(13504, &root).await;
    let c = reqwest::Client::new();
    load(&c, &srv.base, QWEN3).await;

    let resp = c
        .post(format!("{}/v1/chat/completions", srv.base))
        .json(&json!({
            "model": QWEN3, "stream": false, "max_tokens": 48, "temperature": 0.0,
            "messages": [
                { "role": "user", "content": "What is 2+2?" },
                { "role": "assistant", "content": "4", "reasoning_content": "2 plus 2 equals 4." },
                { "role": "user", "content": "And doubled?" }
            ]
        }))
        .send()
        .await
        .expect("chat request");
    assert!(
        resp.status().is_success(),
        "echo-back turn accepted: {}",
        resp.status()
    );
    let body: Value = resp.json().await.expect("chat json");
    assert_eq!(body["object"], "chat.completion", "valid reply: {body}");
}
