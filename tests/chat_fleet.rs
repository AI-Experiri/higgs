//! Tool-parser validation against the REAL test fleet (fetched by
//! `scripts/fetch_test_fleet.sh`; every test SKIPs when the fleet is absent —
//! see `common::fleet_dir`; the models are test-only downloads, never part of
//! the live app's scan roots).
//!
//! The PRIMARY chat pipeline (template apply + tool-call parse) is llama.cpp's
//! own `common_chat` machinery via the AI-Experiri llama-cpp-rs fork; higgs's
//! `ToolParserRegistry` is the FALLBACK for formats the crate parser rejects.
//! These tests pin the fallback registry against REAL model emissions:
//!
//! - **Parser goldens**: feed `tests/fixtures/chat/<slug>/reply.raw` — a real
//!   tool-call emission captured once from the live model — through the
//!   registry parser the model's template selects, and compare the structured
//!   result against `tool_calls.golden` (bless with `UPDATE_GOLDENS=1`).
//! - **Fleet E2E** (`fleet_e2e_tool_chat`): spawn higgs on the fleet root,
//!   load each model, run one tool-call chat, assert a well-formed response —
//!   and, when the model emits a structured call, assert it parses cleanly
//!   (small models don't emit calls deterministically, so that half is
//!   opportunistic).
//!
//! Port: 12960.

mod common;

use std::path::Path;

use common::{fleet_dir, FLEET};
use higgs::worker::models::ModelStore;
use higgs::worker::tool_parser::ToolParserRegistry;
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

/// Scan the fleet root (LM-Studio layout) and return the model's embedded chat
/// template. Panics on a missing model/template — the fleet-presence gate ran
/// before this.
fn fleet_template(root: &Path, id: &str) -> String {
    let mut store = ModelStore::default();
    store
        .scan(&[root.to_path_buf()], &[], &[])
        .expect("scan fleet root");
    store
        .get(id)
        .unwrap_or_else(|| panic!("fleet model {id} not cataloged"))
        .chat_template
        .clone()
        .unwrap_or_else(|| panic!("fleet model {id} has no chat template"))
}

/// Read a fixture file for `slug` (committed; parser goldens are re-blessed
/// with `UPDATE_GOLDENS=1`).
fn fixture(slug: &str, name: &str) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/chat")
        .join(slug)
        .join(name);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("fixture {} missing ({e})", path.display()))
}

/// PARSER golden: the registry parser selected by the model's template must
/// turn the captured real emission into exactly the golden `tool_calls`.
/// The fixed `id_seed` makes call ids deterministic.
fn assert_parser_golden(root: &Path, slug: &str, id: &str) {
    let template = fleet_template(root, id);
    let registry = ToolParserRegistry::default();
    let parser = registry
        .select(&template)
        .unwrap_or_else(|| panic!("{slug}: no registry parser matches the template"));
    let reply = fixture(slug, "reply.raw");
    let calls = parser.parse(&reply, "goldenseed").unwrap_or_else(|| {
        panic!(
            "{slug}: parser {} found no tool call in reply.raw",
            parser.id()
        )
    });
    // Bless mode (test-side only): UPDATE_GOLDENS=1 rewrites the golden from
    // the current parse for manual review, instead of asserting against it.
    if std::env::var_os("UPDATE_GOLDENS").is_some() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/chat")
            .join(slug)
            .join("tool_calls.golden");
        let pretty = serde_json::to_string_pretty(&Value::Array(calls)).expect("serialize calls");
        std::fs::write(&path, pretty).expect("write golden");
        eprintln!("BLESSED {}", path.display());
        return;
    }
    let golden: Value = serde_json::from_str(&fixture(slug, "tool_calls.golden"))
        .expect("tool_calls.golden parses");
    assert_eq!(
        Value::Array(calls),
        golden,
        "{slug}: parsed tool_calls must match the golden"
    );
}

macro_rules! parser_golden {
    ($name:ident, $idx:expr) => {
        #[test]
        fn $name() {
            let Some(root) = fleet_dir() else {
                eprintln!("SKIP {}: fleet absent", stringify!($name));
                return;
            };
            let (slug, id, _) = FLEET[$idx];
            assert_parser_golden(&root, slug, id);
        }
    };
}

parser_golden!(parser_golden_qwen3, 0);
parser_golden!(parser_golden_llama32, 2);
parser_golden!(parser_golden_deepseek_r1, 3);
// Gemma-3 has no parser golden: its template carries no tool-call markers at
// all (gemma-3 has no tool-calling; higgs's gemma parsers target Gemma-4
// templates). The E2E below covers its graceful no-tools behavior.

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
