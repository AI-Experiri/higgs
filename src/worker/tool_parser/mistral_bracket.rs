//! Mistral/Ministral bracket tool-call family.
//!
//! Handles the format Mistral and Ministral GGUF chat templates declare —
//! the call envelope is a literal bracket token sequence rather than XML:
//!
//! ```text
//! [TOOL_CALLS]get_weather[ARGS]{"location": "Paris"}
//! ```
//!
//! Each call is `[TOOL_CALLS]` then the bare function NAME, then `[ARGS]`, then
//! a single JSON object holding the arguments. Multiple calls are concatenated:
//!
//! ```text
//! [TOOL_CALLS]get_weather[ARGS]{"location":"NYC"}[TOOL_CALLS]get_time[ARGS]{"timezone":"EST"}
//! ```
//!
//! Reasoning, when present, is wrapped in `[THINK]…[/THINK]` (Mistral's own
//! tags), but the template may force the block open so only a trailing
//! `[/THINK]` is emitted — `content()` handles both.
//!
//! We parse it because llama.cpp's vendored `common_chat` does not cover the
//! Ministral bracket dialect; the format is read straight from the model's own
//! embedded chat template.
//!
//! Ported from Ollama's `ministral.go`. We keep the streaming parser's core
//! parsing logic (the `[TOOL_CALLS]…[ARGS]{json}` shape and the brace-balanced
//! JSON-end finder) but drop its incremental state machine and partial-tag
//! overlap heuristics, which only matter for token-by-token streaming.

use serde_json::{Map, Value};

use super::{build_tool_call, strip_leading_reasoning, ToolCallParser};

const TOOL_CALLS_TAG: &str = "[TOOL_CALLS]";
const ARGS_TAG: &str = "[ARGS]";
const THINK_END_TAG: &str = "[/THINK]";

/// Parser for the Mistral/Ministral `[TOOL_CALLS]NAME[ARGS]{json}` family.
pub struct MistralBracketParser;

impl ToolCallParser for MistralBracketParser {
    fn id(&self) -> &'static str {
        "mistral-bracket"
    }

    fn handles(&self, chat_template: &str) -> bool {
        // The template renders calls via these literal bracket tokens. Both are
        // required so this does not collide with the XML `<function=…>` family
        // or any generic `<tool_call>` JSON family — `[ARGS]` in particular is
        // unique to the Ministral dialect.
        chat_template.contains(TOOL_CALLS_TAG) && chat_template.contains(ARGS_TAG)
    }

    fn open_markers(&self) -> &'static [&'static str] {
        &["[TOOL_CALLS]"]
    }

    fn parse(&self, text: &str, id_seed: &str) -> Option<Vec<Value>> {
        let mut calls = Vec::new();
        let mut rest = text;
        while let Some(start) = rest.find(TOOL_CALLS_TAG) {
            let after = &rest[start + TOOL_CALLS_TAG.len()..];
            // NAME runs up to the [ARGS] separator.
            let Some(asep) = after.find(ARGS_TAG) else {
                break;
            };
            let name = after[..asep].trim();
            let args_body = &after[asep + ARGS_TAG.len()..];
            // ARGS is a single JSON object; find its balanced closing brace.
            let Some(jend) = find_json_end(args_body) else {
                break;
            };
            let json_str = &args_body[..=jend];
            if !name.is_empty() {
                if let Some(call) = build_call(name, json_str, id_seed, calls.len()) {
                    calls.push(call);
                }
            }
            rest = &args_body[jend + 1..];
        }
        if calls.is_empty() {
            None
        } else {
            Some(calls)
        }
    }

    fn content(&self, text: &str) -> String {
        let head = text.find(TOOL_CALLS_TAG).map_or(text, |i| &text[..i]);
        strip_leading_reasoning(head, THINK_END_TAG)
            .trim()
            .to_string()
    }
}

/// Build one OpenAI tool-call object from a name and a raw JSON arguments
/// string. The args object is re-serialized so the wire `arguments` is a
/// canonical JSON STRING; an unparseable body yields an empty object.
fn build_call(name: &str, json_str: &str, id_seed: &str, index: usize) -> Option<Value> {
    let args = match serde_json::from_str::<Value>(json_str) {
        Ok(Value::Object(map)) => map,
        // Non-object or invalid JSON: emit empty arguments rather than dropping
        // the call. (Ollama errors here; we degrade instead.)
        _ => Map::new(),
    };
    Some(build_tool_call(
        name,
        &Value::Object(args).to_string(),
        id_seed,
        index,
    ))
}

/// Index of the closing brace that completes the root JSON value at the start
/// of `s`, or `None` if it never closes. Brace/bracket depth aware and skips
/// braces inside string literals (honoring `\` escapes). Ported verbatim from
/// Ollama's `findJSONEnd`; byte-indexed since braces, brackets, quotes and
/// backslash are all ASCII.
fn find_json_end(s: &str) -> Option<usize> {
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escaped = false;
    for (i, b) in s.bytes().enumerate() {
        if in_string {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_string = false;
            }
            continue;
        }
        match b {
            b'"' => in_string = true,
            b'{' | b'[' => depth += 1,
            b'}' | b']' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p() -> MistralBracketParser {
        MistralBracketParser
    }

    #[test]
    fn handles_mistral_template_only() {
        // A template that renders both bracket tokens.
        assert!(p().handles("… '[TOOL_CALLS]' ~ name ~ '[ARGS]' ~ args …"));
        // XML family — no bracket tokens.
        assert!(!p().handles("… '<function=' ~ name … '<parameter=' ~ k …"));
        // Has [TOOL_CALLS] but no [ARGS] — not this dialect.
        assert!(!p().handles("plain [TOOL_CALLS] only, no args tag"));
    }

    #[test]
    fn parses_single_call() {
        // Ground truth from ministral_test.go line 46.
        let text = r#"[TOOL_CALLS]get_weather[ARGS]{"location": "San Francisco"}"#;
        let calls = p().parse(text, "abc").expect("one call");
        assert_eq!(calls.len(), 1);
        let f = &calls[0]["function"];
        assert_eq!(f["name"], "get_weather");
        let args: Value = serde_json::from_str(f["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(args["location"], "San Francisco");
        assert_eq!(calls[0]["type"], "function");
        assert_eq!(calls[0]["id"], "call_abc_0");
    }

    #[test]
    fn parses_multiple_args_with_nested_json_value() {
        // From ministral_test.go line 64 — nested object argument.
        let text = r#"[TOOL_CALLS]update_config[ARGS]{"settings": {"user": {"name": "John", "age": 30}}, "theme": "dark"}"#;
        let calls = p().parse(text, "z").unwrap();
        let args: Value =
            serde_json::from_str(calls[0]["function"]["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(args["theme"], "dark");
        assert_eq!(args["settings"]["user"]["name"], "John");
        assert_eq!(args["settings"]["user"]["age"], 30); // numeric stays typed
    }

    #[test]
    fn parses_two_tool_calls() {
        // From ministral_test.go line 147 — concatenated calls.
        let text = r#"[TOOL_CALLS]get_weather[ARGS]{"location": "NYC"}[TOOL_CALLS]get_time[ARGS]{"timezone": "EST"}"#;
        let calls = p().parse(text, "s").unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0]["function"]["name"], "get_weather");
        assert_eq!(calls[1]["function"]["name"], "get_time");
        assert_eq!(calls[1]["id"], "call_s_1");
    }

    #[test]
    fn no_tool_call_returns_none() {
        assert!(p().parse("just a normal answer", "x").is_none());
        // Unterminated JSON — find_json_end never closes.
        assert!(p().parse("[TOOL_CALLS]test[ARGS]{", "x").is_none());
        // Missing [ARGS] separator.
        assert!(p().parse("[TOOL_CALLS]test{}", "x").is_none());
    }

    #[test]
    fn content_strips_think_and_tool_call() {
        // Preamble before the first [TOOL_CALLS] is the content.
        let with_preamble = r#"Sure, let me check.
[TOOL_CALLS]get_weather[ARGS]{"location": "NYC"}"#;
        assert_eq!(p().content(with_preamble), "Sure, let me check.");

        // Explicit [THINK]…[/THINK] block is stripped.
        let with_think = r#"[THINK]reasoning here[/THINK]done[TOOL_CALLS]x[ARGS]{}"#;
        assert_eq!(p().content(with_think), "done");
    }

    #[test]
    fn content_strips_forced_open_think_without_opening_tag() {
        // Template forced the think block open → only a trailing [/THINK].
        // (ministral_test.go line 337 exercises this lead-in shape.)
        let forced = "let me think[/THINK][TOOL_CALLS]test[ARGS]{}";
        assert_eq!(p().content(forced), "");
    }
}
