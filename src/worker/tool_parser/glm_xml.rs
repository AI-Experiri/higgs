//! GLM-4.6 XML key/value tool-call family.
//!
//! GLM-4.x chat templates render calls as an XML-ish block: the function name
//! as bare text, followed by repeated `<arg_key>`/`<arg_value>` pairs:
//!
//! ```text
//! <tool_call>get_weather
//! <arg_key>location</arg_key>
//! <arg_value>Paris</arg_value>
//! <arg_key>unit</arg_key>
//! <arg_value>celsius</arg_value>
//! </tool_call>
//! ```
//!
//! This differs from the Qwen JSON family (which ALSO opens with `<tool_call>`
//! but carries a JSON object inside) by its `<arg_key>`/`<arg_value>` markers —
//! that pair is what `handles()` keys on so the two never collide. We parse it
//! ourselves because llama.cpp's vendored `common_chat` does not cover the GLM
//! key/value dialect.

use serde_json::{json, Map, Value};

use super::ToolCallParser;

const TOOL_OPEN: &str = "<tool_call>";
const TOOL_CLOSE: &str = "</tool_call>";
const KEY_OPEN: &str = "<arg_key>";
const KEY_CLOSE: &str = "</arg_key>";
const VAL_OPEN: &str = "<arg_value>";
const VAL_CLOSE: &str = "</arg_value>";

/// Parser for the GLM-4.6 XML `<arg_key>`/`<arg_value>` tool-call family.
pub struct GlmXmlParser;

impl ToolCallParser for GlmXmlParser {
    fn id(&self) -> &'static str {
        "glm-xml"
    }

    fn handles(&self, chat_template: &str) -> bool {
        // The GLM template emits each argument as `<arg_key>` … `<arg_value>`.
        // Requiring BOTH distinguishes GLM from the Qwen JSON family, which
        // also opens with `<tool_call>` but has no arg_key/arg_value markers.
        chat_template.contains(KEY_OPEN) && chat_template.contains(VAL_OPEN)
    }

    fn open_markers(&self) -> &'static [&'static str] {
        &[TOOL_OPEN]
    }

    fn parse(&self, text: &str, id_seed: &str) -> Option<Vec<Value>> {
        let mut calls = Vec::new();
        let mut rest = text;
        while let Some(start) = rest.find(TOOL_OPEN) {
            let after = &rest[start + TOOL_OPEN.len()..];
            let Some(end) = after.find(TOOL_CLOSE) else {
                break;
            };
            if let Some(call) = parse_one(&after[..end], id_seed, calls.len()) {
                calls.push(call);
            }
            rest = &after[end + TOOL_CLOSE.len()..];
        }
        if calls.is_empty() {
            None
        } else {
            Some(calls)
        }
    }

    fn content(&self, text: &str) -> String {
        let head = text.find(TOOL_OPEN).map_or(text, |i| &text[..i]);
        strip_think(head).trim().to_string()
    }
}

/// Remove a leading reasoning block. Handles both an explicit `<think>…</think>`
/// and the template-forced-open case (no opening tag, just a trailing
/// `</think>`): everything up to and including the first `</think>` is dropped.
fn strip_think(s: &str) -> &str {
    match s.find("</think>") {
        Some(end) => &s[end + "</think>".len()..],
        None => s,
    }
}

/// Parse one `<tool_call>` block body into an OpenAI tool-call object.
///
/// The function name is the text before the first `<arg_key>` (or the whole
/// body when there are no args). Each subsequent `<arg_key>K</arg_key>` is
/// paired with the following `<arg_value>V</arg_value>`.
///
/// Argument-value coercion: GLM's reference parser uses the tool's declared
/// JSON-Schema type to coerce values, but higgs has no tool schema at parse
/// time, so — like the XML `<function=…>` parser — we re-parse a value as JSON
/// when it is valid (numbers/bools/objects/arrays stay typed) and otherwise
/// keep the literal as a string. Ollama's `repairGLM46XML` tag-repair heuristic
/// for malformed output is intentionally NOT ported; only the clean path.
fn parse_one(block: &str, id_seed: &str, index: usize) -> Option<Value> {
    // Function name = text up to the first <arg_key> (or whole block if none).
    let name_end = block.find(KEY_OPEN).unwrap_or(block.len());
    let name = block[..name_end].trim();
    if name.is_empty() {
        return None;
    }

    let mut args = Map::new();
    let mut rest = &block[name_end..];
    while let Some(ks) = rest.find(KEY_OPEN) {
        let after_k = &rest[ks + KEY_OPEN.len()..];
        let Some(ke) = after_k.find(KEY_CLOSE) else {
            break;
        };
        let key = after_k[..ke].trim();
        let after_kc = &after_k[ke + KEY_CLOSE.len()..];

        let Some(vs) = after_kc.find(VAL_OPEN) else {
            break;
        };
        let after_v = &after_kc[vs + VAL_OPEN.len()..];
        let Some(ve) = after_v.find(VAL_CLOSE) else {
            break;
        };
        let raw = after_v[..ve].trim();
        let value =
            serde_json::from_str::<Value>(raw).unwrap_or_else(|_| Value::String(raw.to_string()));
        if !key.is_empty() {
            args.insert(key.to_string(), value);
        }
        rest = &after_v[ve + VAL_CLOSE.len()..];
    }

    Some(json!({
        "id": format!("call_{id_seed}_{index}"),
        "type": "function",
        "function": {
            "name": name,
            // OpenAI `arguments` is a JSON STRING, not an object.
            "arguments": Value::Object(args).to_string(),
        }
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Byte layout taken verbatim from Ollama's `glm46_test.go` ground truth.
    const SINGLE: &str =
        "<think>Let me check.</think><tool_call>get_weather\n<arg_key>location</arg_key>\n<arg_value>Paris</arg_value>\n</tool_call>";

    fn p() -> GlmXmlParser {
        GlmXmlParser
    }

    #[test]
    fn handles_glm_template_only() {
        assert!(p().handles("… <arg_key>{{ k }}</arg_key>\n<arg_value>{{ v }}</arg_value> …"));
        // Qwen JSON family also uses <tool_call> but no arg_key/arg_value.
        assert!(!p().handles("<tool_call>\n{\"name\": \"x\", \"arguments\": {}}\n</tool_call>"));
        assert!(!p().handles("<|im_start|>{{ role }} plain chatml, no markers"));
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
    fn parses_multiple_args_and_json_value() {
        let text = "<tool_call>search\n<arg_key>query</arg_key>\n<arg_value>rust</arg_value>\n<arg_key>limit</arg_key>\n<arg_value>5</arg_value>\n<arg_key>filters</arg_key>\n<arg_value>{\"lang\":\"en\"}</arg_value>\n</tool_call>";
        let calls = p().parse(text, "z").unwrap();
        let args: Value =
            serde_json::from_str(calls[0]["function"]["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(args["query"], "rust");
        assert_eq!(args["limit"], 5); // numeric, re-parsed as JSON
        assert_eq!(args["filters"]["lang"], "en"); // nested object
    }

    #[test]
    fn parses_two_tool_calls() {
        let text = "<tool_call>a\n<arg_key>k</arg_key>\n<arg_value>1</arg_value>\n</tool_call>between<tool_call>b</tool_call>";
        let calls = p().parse(text, "s").unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0]["function"]["name"], "a");
        assert_eq!(calls[1]["function"]["name"], "b");
        assert_eq!(calls[1]["id"], "call_s_1");
    }

    #[test]
    fn no_tool_call_returns_none() {
        assert!(p().parse("just a normal answer", "x").is_none());
        assert!(p()
            .parse("<tool_call>x\n<arg_key>k</arg_key>", "x")
            .is_none()); // unterminated
    }

    #[test]
    fn content_strips_think_and_tool_call() {
        assert_eq!(p().content(SINGLE), "");
        let with_preamble = "Here's the answer:<tool_call>test</tool_call>done";
        assert_eq!(p().content(with_preamble), "Here's the answer:");
    }

    #[test]
    fn content_strips_forced_open_think_without_opening_tag() {
        // Template forced the think block open → only a trailing </think>.
        let forced = "Let me check the weather.\n</think>\n<tool_call>get_weather\n<arg_key>location</arg_key>\n<arg_value>Paris</arg_value>\n</tool_call>";
        assert_eq!(p().content(forced), "");
    }
}
