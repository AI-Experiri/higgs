//! Qwen3 ChatML JSON `<tool_call>` family.
//!
//! Handles the format Qwen3 / Qwen3-VL GGUF chat templates declare: a JSON
//! object (`{"name": …, "arguments": {…}}`) wrapped in `<tool_call>` tags,
//! optionally preceded by a `<think>…</think>` reasoning block and repeated for
//! multiple calls:
//!
//! ```text
//! <think>
//! The user wants the weather.
//! </think>
//! <tool_call>
//! {"name": "get_weather", "arguments": {"location": "Paris"}}
//! </tool_call>
//! ```
//!
//! This is the plain ChatML JSON tool-call dialect (the Qwen3 family), distinct
//! from the XML `<function=…><parameter=…>` family (Qwen3-Coder / Nemotron /
//! GLM). The body between the tags is parsed straight as JSON — `arguments` is
//! already a typed object, so no per-value coercion is needed.
//!
//! Ported from Ollama's `qwen3.go` parser: only its parsing logic (extract the
//! `<tool_call>` body, JSON-decode `name` + `arguments`) is reproduced. Ollama's
//! streaming state machine and partial-tag overlap heuristics are intentionally
//! omitted — this layer parses complete text.

use serde_json::{json, Value};

use super::{build_tool_call, content_outside_calls, strip_leading_reasoning, ToolCallParser};

/// Parser for the Qwen3 ChatML JSON `<tool_call>` tool-call family.
pub struct QwenJsonParser;

impl ToolCallParser for QwenJsonParser {
    fn id(&self) -> &'static str {
        "qwen-json"
    }

    fn handles(&self, chat_template: &str) -> bool {
        // Qwen3 renders calls as a JSON object inside `<tool_call>` tags. Require
        // `<tool_call>` AND the absence of the XML-family markers so this never
        // steals Qwen3-Coder's `<function=…>` XML nor GLM's `<arg_key>` dialect.
        chat_template.contains("<tool_call>")
            && !chat_template.contains("<function=")
            && !chat_template.contains("<arg_key>")
    }

    fn open_markers(&self) -> &'static [&'static str] {
        &["<tool_call>"]
    }

    fn parse(&self, text: &str, id_seed: &str) -> Option<Vec<Value>> {
        let mut calls = Vec::new();
        let mut rest = text;
        while let Some(start) = rest.find("<tool_call>") {
            let after = &rest[start + "<tool_call>".len()..];
            let Some(end) = after.find("</tool_call>") else {
                break;
            };
            if let Some(call) = parse_one(&after[..end], id_seed, calls.len()) {
                calls.push(call);
            }
            rest = &after[end + "</tool_call>".len()..];
        }
        if calls.is_empty() {
            None
        } else {
            Some(calls)
        }
    }

    fn content(&self, text: &str) -> String {
        let out = content_outside_calls(text, "<tool_call>", "</tool_call>");
        strip_leading_reasoning(&out, "</think>").trim().to_string()
    }
}

/// Parse one `<tool_call>` block body — a `{"name":…,"arguments":{…}}` JSON
/// object — into an OpenAI tool-call object. Returns `None` if the body is not
/// valid JSON or carries an empty/missing function name.
fn parse_one(block: &str, id_seed: &str, index: usize) -> Option<Value> {
    let parsed: Value = serde_json::from_str(block.trim()).ok()?;
    let name = parsed.get("name")?.as_str()?.trim();
    if name.is_empty() {
        return None;
    }
    // `arguments` is already a typed JSON object in the Qwen3 dialect; default to
    // an empty object when absent. Re-serialize to the OpenAI string form.
    let args = parsed
        .get("arguments")
        .cloned()
        .unwrap_or_else(|| json!({}));

    Some(build_tool_call(name, &args.to_string(), id_seed, index))
}

#[cfg(test)]
#[path = "qwen_json_tests.rs"]
mod tests;
