//! Tool-call chats against the REAL test fleet (fetched by
//! `scripts/fetch_test_fleet.sh`; every test SKIPs when the fleet is absent —
//! see `common::fleet_dir`; the models are test-only downloads, never part of
//! the live app's scan roots).
//!
//! The chat pipeline (template apply + tool-call/reasoning parse) is
//! llama.cpp's own `common_chat` machinery via the AI-Experiri llama-cpp-rs
//! fork — higgs ships no parser of its own (the PEG auto-parser derived from
//! each GGUF template covers every target family; verified live in this test).
//!
//! - **Fleet E2E** (`fleet_e2e_tool_chat`): spawn higgs on the fleet root,
//!   load each model, run one tool-call chat, assert a well-formed response —
//!   and, when the model emits a structured call, assert it parses cleanly
//!   (small models don't emit calls deterministically, so that half is
//!   opportunistic).
//!
//! Port: 12960.

mod common;

use common::{fleet_dir, FLEET};
use serde_json::{json, Value};

/// The get_weather tool used by the E2E tool chats.
const TOOLS: &str = r#"[
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"}
                },
                "required": ["city"]
            }
        }
    }
]"#;

/// Live E2E over the whole fleet: load each model in turn (a reload replaces
/// the resident model), run ONE tool-call chat, and assert a well-formed
/// response. When structured `tool_calls` come back, assert the OpenAI shape
/// and that no call markup leaked into `content`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fleet_e2e_tool_chat() {
    let Some(root) = fleet_dir() else {
        eprintln!("SKIP fleet_e2e_tool_chat: fleet absent");
        return;
    };
    let srv = common::spawn_with_model_root(12960, &root).await;
    let c = reqwest::Client::new();
    let tools: Value = serde_json::from_str(TOOLS).expect("tools parse");

    for (slug, id, _) in FLEET {
        let r = c
            .post(format!("{}/api/higgs/models/load", srv.base))
            .json(&json!({ "id": id, "ctx_len": 2048 }))
            .send()
            .await
            .expect("load request");
        assert!(r.status().is_success(), "{slug}: load succeeds");

        let resp: Value = c
            .post(format!("{}/v1/chat/completions", srv.base))
            .json(&json!({
                "model": id,
                "stream": false,
                "max_tokens": 256,
                "messages": [
                    { "role": "system", "content": "You are a helpful assistant." },
                    { "role": "user", "content": "What's the weather in Paris? Use the get_weather tool." }
                ],
                "tools": tools
            }))
            .send()
            .await
            .expect("chat request")
            .json()
            .await
            .expect("chat json");

        let msg = &resp["choices"][0]["message"];
        assert!(
            msg["content"].is_string() || msg["tool_calls"].is_array(),
            "{slug}: response has content or tool_calls: {resp}"
        );
        if let Some(calls) = msg["tool_calls"].as_array() {
            assert!(
                !calls.is_empty(),
                "{slug}: tool_calls non-empty when present"
            );
            let f = &calls[0]["function"];
            assert_eq!(f["name"], "get_weather", "{slug}: correct tool: {resp}");
            assert!(
                f["arguments"].is_string(),
                "{slug}: OpenAI arguments are a JSON STRING: {resp}"
            );
            let content = msg["content"].as_str().unwrap_or("");
            assert!(
                !content.contains("<tool_call>") && !content.contains("<function"),
                "{slug}: no call markup in content: {resp}"
            );
            eprintln!("E2E {slug}: structured tool call parsed");
        } else {
            eprintln!("E2E {slug}: no structured call this run (content path)");
        }
    }

    // Drain-all unload so the server tears down cleanly.
    let r = c
        .post(format!("{}/api/higgs/models/unload", srv.base))
        .json(&json!({}))
        .send()
        .await
        .expect("unload request");
    assert!(r.status().is_success(), "unload all succeeds");
}
