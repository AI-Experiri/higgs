//! DeepSeek-V3 unicode-tag tool-call family.
//!
//! Handles the format the DeepSeek-V3 / R1 GGUF chat templates declare, using
//! special `▁` (U+2581) bracket tags:
//!
//! ```text
//! <｜tool▁calls▁begin｜>
//! <｜tool▁call▁begin｜>NAME<｜tool▁sep｜>{"key":"value"}<｜tool▁call▁end｜>
//! <｜tool▁call▁begin｜>NAME2<｜tool▁sep｜>{...}<｜tool▁call▁end｜>
//! <｜tool▁calls▁end｜>
//! ```
//!
//! The name comes immediately after `<｜tool▁call▁begin｜>`, the `<｜tool▁sep｜>`
//! tag separates it from the argument object, and the arguments are a raw JSON
//! object (no ```json fence). We parse it because llama.cpp's vendored
//! `common_chat` does not cover the DeepSeek-V3 unicode-tag dialect.
//!
//! Ported from the Ollama Go parser (`model/parsers/deepseek3.go`): we keep the
//! common-case extraction and drop the streaming state machine, the
//! `<｜tool▁output▁begin｜>` round-trip handling, and the partial-tag overlap
//! heuristics (those only matter for incremental streaming, not complete text).

use serde_json::Value;

use super::{build_tool_call, content_outside_calls, strip_leading_reasoning, ToolCallParser};

// DeepSeek-V3 unicode bracket tags (the `▁` is U+2581, not an ASCII underscore).
const CALLS_BEGIN: &str = "<｜tool▁calls▁begin｜>";
const CALLS_END: &str = "<｜tool▁calls▁end｜>";
const CALL_BEGIN: &str = "<｜tool▁call▁begin｜>";
const CALL_END: &str = "<｜tool▁call▁end｜>";
const TOOL_SEP: &str = "<｜tool▁sep｜>";

/// Parser for the DeepSeek-V3 unicode-tag (`<｜tool▁call▁begin｜>…`) family.
pub struct DeepSeek3Parser;

impl ToolCallParser for DeepSeek3Parser {
    fn id(&self) -> &'static str {
        "deepseek3"
    }

    fn handles(&self, chat_template: &str) -> bool {
        // The DeepSeek-V3 template renders calls via these `▁` (U+2581) unicode
        // tags. They are unique to this family — no other format uses them — so
        // matching the calls-begin and sep tags cannot collide with the generic
        // `<tool_call>` XML families.
        chat_template.contains(CALLS_BEGIN) && chat_template.contains(TOOL_SEP)
    }

    fn open_markers(&self) -> &'static [&'static str] {
        &[CALLS_BEGIN]
    }

    fn parse(&self, text: &str, id_seed: &str) -> Option<Vec<Value>> {
        // The call envelope lives between calls-begin and calls-end; bound the
        // scan to it so trailing content is never mistaken for a call.
        let start = text.find(CALLS_BEGIN)?;
        let region = &text[start + CALLS_BEGIN.len()..];
        let region = match region.find(CALLS_END) {
            Some(end) => &region[..end],
            None => region,
        };

        let mut calls = Vec::new();
        let mut rest = region;
        while let Some(bs) = rest.find(CALL_BEGIN) {
            let after = &rest[bs + CALL_BEGIN.len()..];
            let Some(be) = after.find(CALL_END) else {
                break;
            };
            if let Some(call) = parse_one(&after[..be], id_seed, calls.len()) {
                calls.push(call);
            }
            rest = &after[be + CALL_END.len()..];
        }

        if calls.is_empty() {
            None
        } else {
            Some(calls)
        }
    }

    fn content(&self, text: &str) -> String {
        let out = content_outside_calls(text, CALLS_BEGIN, CALLS_END);
        strip_leading_reasoning(&out, "</think>").trim().to_string()
    }
}

/// Parse one `NAME<｜tool▁sep｜>{args}` call body into an OpenAI tool-call object.
///
/// The name is everything before the separator; the arguments are the raw JSON
/// object after it. Only well-formed JSON object args are accepted (matching the
/// Go parser, which `json.Unmarshal`s into a map and drops the call on error).
fn parse_one(body: &str, id_seed: &str, index: usize) -> Option<Value> {
    let sep = body.find(TOOL_SEP)?;
    let name = body[..sep].trim();
    if name.is_empty() {
        return None;
    }
    let args_raw = body[sep + TOOL_SEP.len()..].trim();

    // Go unmarshals into ToolCallFunctionArguments (a JSON object) and drops the
    // call on any error, including an empty body; reject the call if the args are
    // not a valid JSON object.
    let args = match serde_json::from_str::<Value>(args_raw) {
        Ok(Value::Object(map)) => map,
        _ => return None,
    };

    Some(build_tool_call(
        name,
        &Value::Object(args).to_string(),
        id_seed,
        index,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Single call, exact byte layout from the Ollama Go `tool_call_simple` test.
    const SINGLE: &str = "I'll check the weather.<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{\"location\":\"Paris\"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>";

    fn p() -> DeepSeek3Parser {
        DeepSeek3Parser
    }

    #[test]
    fn handles_deepseek_template_only() {
        assert!(p().handles(&format!("…{CALLS_BEGIN}…{TOOL_SEP}…")));
        // Generic XML tool-call template has none of the ▁ unicode tags.
        assert!(!p().handles("<tool_call>\n<function= … <parameter= …"));
        assert!(!p().handles("<|im_start|>{{ role }} plain chatml"));
    }

    #[test]
    fn parses_single_call() {
        let calls = p().parse(SINGLE, "abc").expect("one call");
        assert_eq!(calls.len(), 1);
        let f = &calls[0]["function"];
        assert_eq!(f["name"], "get_weather");
        let args: Value = serde_json::from_str(f["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(args["location"], "Paris");
        assert_eq!(calls[0]["type"], "function");
        assert_eq!(calls[0]["id"], "call_abc_0");
    }

    #[test]
    fn parses_multiple_calls() {
        let text = "Getting weather for both cities.<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{\"location\":\"Paris\"}<｜tool▁call▁end｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{\"location\":\"London\"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>";
        let calls = p().parse(text, "s").unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0]["function"]["name"], "get_weather");
        assert_eq!(calls[1]["id"], "call_s_1");
        let a1: Value =
            serde_json::from_str(calls[1]["function"]["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(a1["location"], "London");
    }

    #[test]
    fn parses_complex_typed_arguments() {
        // From the Go `complex_tool_arguments` test: arrays, nested objects,
        // bools and numbers must keep their JSON types.
        let text = "Processing data.<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>process_data<｜tool▁sep｜>{\"items\":[\"item1\",\"item2\"],\"config\":{\"enabled\":true,\"threshold\":0.95}}<｜tool▁call▁end｜><｜tool▁calls▁end｜>";
        let calls = p().parse(text, "z").unwrap();
        let args: Value =
            serde_json::from_str(calls[0]["function"]["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(args["items"][0], "item1");
        assert_eq!(args["config"]["enabled"], true);
        assert_eq!(args["config"]["threshold"], 0.95);
    }

    #[test]
    fn no_tool_call_returns_none() {
        assert!(p().parse("Hello, how are you?", "x").is_none());
        // calls-begin present but no well-formed call body.
        assert!(p()
            .parse("text<｜tool▁calls▁begin｜><｜tool▁calls▁end｜>", "x")
            .is_none());
    }

    #[test]
    fn content_strips_think_and_calls() {
        assert_eq!(p().content(SINGLE), "I'll check the weather.");
        // Explicit reasoning block before the answer.
        let with_think = "Let me check...</think>I'll get that.<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{}<｜tool▁call▁end｜><｜tool▁calls▁end｜>";
        assert_eq!(p().content(with_think), "I'll get that.");
    }

    #[test]
    fn content_strips_forced_open_think_without_opening_tag() {
        let forced = "Reasoning about the weather request.\n</think>\nDone.";
        assert_eq!(p().content(forced), "Done.");
    }
}
