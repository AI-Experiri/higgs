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
mod tests {
    use super::*;

    /// The exact byte layout Qwen3 emits: a `<think>` block then a JSON object
    /// inside `<tool_call>` tags.
    const QWEN3: &str = "<think>\nThe user wants weather.\n</think>\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Paris\"}}\n</tool_call>\n";

    fn p() -> QwenJsonParser {
        QwenJsonParser
    }

    #[test]
    fn handles_qwen_template_only() {
        // Plain ChatML JSON template with <tool_call> and no XML markers.
        assert!(p().handles("…{{ '<tool_call>\\n' }}{{ tc | tojson }}{{ '\\n</tool_call>' }}…"));
        // Must NOT steal the XML-function family.
        assert!(!p().handles("… '<tool_call>' … '<function=' ~ name … '<parameter=' ~ k …"));
        // Must NOT steal the GLM <arg_key> dialect.
        assert!(!p().handles("… '<tool_call>' … '<arg_key>' … '<arg_value>' …"));
        // No <tool_call> at all → not this format.
        assert!(!p().handles("<|im_start|>{{ role }} plain chatml, no tool markers"));
    }

    #[test]
    fn parses_qwen3_single_call() {
        let calls = p().parse(QWEN3, "abc").expect("one call");
        assert_eq!(calls.len(), 1);
        let f = &calls[0]["function"];
        assert_eq!(f["name"], "get_weather");
        let args: Value = serde_json::from_str(f["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(args["location"], "Paris");
        assert_eq!(calls[0]["type"], "function");
        assert_eq!(calls[0]["id"], "call_abc_0");
    }

    #[test]
    fn parses_multiple_args_incl_json_typed_value() {
        let text = "<tool_call>\n{\"name\": \"search\", \"arguments\": {\"query\": \"rust\", \"limit\": 5, \"filters\": {\"lang\": \"en\"}}}\n</tool_call>";
        let calls = p().parse(text, "z").unwrap();
        let args: Value =
            serde_json::from_str(calls[0]["function"]["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(args["query"], "rust");
        assert_eq!(args["limit"], 5); // numeric stays typed
        assert_eq!(args["filters"]["lang"], "en"); // nested object stays typed
    }

    #[test]
    fn parses_two_tool_calls() {
        let text = "<tool_call>\n{\"name\": \"a\", \"arguments\": {}}\n</tool_call>\n<tool_call>\n{\"name\": \"b\", \"arguments\": {\"x\": 1}}\n</tool_call>";
        let calls = p().parse(text, "s").unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0]["function"]["name"], "a");
        assert_eq!(calls[1]["function"]["name"], "b");
        assert_eq!(calls[1]["id"], "call_s_1");
    }

    #[test]
    fn no_tool_call_returns_none() {
        assert!(p().parse("just a normal answer", "x").is_none());
        // Unterminated block — no closing tag.
        assert!(p()
            .parse("<tool_call>\n{\"name\": \"x\", \"arguments\": {}}", "x")
            .is_none());
        // Empty function name is rejected.
        assert!(p()
            .parse(
                "<tool_call>\n{\"name\": \"\", \"arguments\": {}}\n</tool_call>",
                "x"
            )
            .is_none());
    }

    #[test]
    fn content_strips_think_and_tool_call() {
        assert_eq!(p().content(QWEN3), "");
        let with_preamble =
            "Sure.\n<tool_call>\n{\"name\": \"x\", \"arguments\": {}}\n</tool_call>";
        assert_eq!(p().content(with_preamble), "Sure.");
    }

    #[test]
    fn content_strips_forced_open_think_without_opening_tag() {
        // Template forced the think block open → only a trailing </think>.
        let forced = "Reasoning about the request.\n</think>\n<tool_call>\n{\"name\": \"x\", \"arguments\": {}}\n</tool_call>";
        assert_eq!(p().content(forced), "");
    }

    #[test]
    fn content_preserves_text_after_a_tool_call() {
        // Contract: content is the text with tool-call markup REMOVED — text after
        // (and between) calls must survive, not be dropped at the first marker.
        let text = "Calling.<tool_call>{\"name\":\"f\",\"arguments\":{}}</tool_call>Done.";
        assert_eq!(p().content(text), "Calling.Done.");
    }
}
