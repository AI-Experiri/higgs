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

use serde_json::{Map, Value};

use super::{build_tool_call, content_outside_calls, strip_leading_reasoning, ToolCallParser};

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
        let out = content_outside_calls(text, TOOL_OPEN, TOOL_CLOSE);
        strip_leading_reasoning(&out, "</think>").trim().to_string()
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

    Some(build_tool_call(
        name,
        &Value::Object(args).to_string(),
        id_seed,
        index,
    ))
}

#[cfg(test)]
#[path = "glm_xml_tests.rs"]
mod tests;
