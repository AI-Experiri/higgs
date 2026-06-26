//! XML `<function=…>` tool-call family.
//!
//! Handles the format several GGUF chat templates declare verbatim — Nemotron
//! (`nemotron_h`), Qwen3-Coder:
//!
//! ```text
//! <tool_call>
//! <function=NAME>
//! <parameter=KEY>
//! VALUE
//! </parameter>
//! </function>
//! </tool_call>
//! ```
//!
//! The format is read straight from the model's own embedded chat template; we
//! parse it because llama.cpp's vendored `common_chat` rejects valid output for
//! this family at completion (a non-null derived PEG that still throws FFI -3 on
//! `nemotron_h`).

use serde_json::{Map, Value};

use super::{build_tool_call, content_outside_calls, strip_leading_reasoning, ToolCallParser};

/// Parser for the XML `<function=…><parameter=…>` tool-call family.
pub struct XmlFunctionParser;

impl ToolCallParser for XmlFunctionParser {
    fn id(&self) -> &'static str {
        "xml-function"
    }

    fn handles(&self, chat_template: &str) -> bool {
        // The template renders calls via these literal markers (e.g. Nemotron:
        // `'<function=' ~ tool_call.name` and `'<parameter=' ~ args_name`).
        chat_template.contains("<function=") && chat_template.contains("<parameter=")
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

/// Parse one `<tool_call>` block body into an OpenAI tool-call object.
fn parse_one(block: &str, id_seed: &str, index: usize) -> Option<Value> {
    let fstart = block.find("<function=")?;
    let after_f = &block[fstart + "<function=".len()..];
    let nend = after_f.find('>')?;
    let name = after_f[..nend].trim();
    if name.is_empty() {
        return None;
    }
    let body = &after_f[nend + 1..];

    // Collect <parameter=KEY> VALUE </parameter> pairs into an arguments object.
    let mut args = Map::new();
    let mut rest = body;
    while let Some(ps) = rest.find("<parameter=") {
        let after_p = &rest[ps + "<parameter=".len()..];
        let Some(kend) = after_p.find('>') else { break };
        let key = after_p[..kend].trim();
        let vbody = &after_p[kend + 1..];
        let Some(vend) = vbody.find("</parameter>") else {
            break;
        };
        let raw = vbody[..vend].trim();
        // The template emits non-string args via `tojson` (objects/arrays/
        // numbers/bools); strings are emitted raw. Re-parse JSON when it is
        // valid, else keep the literal as a string.
        let value =
            serde_json::from_str::<Value>(raw).unwrap_or_else(|_| Value::String(raw.to_string()));
        if !key.is_empty() {
            args.insert(key.to_string(), value);
        }
        rest = &vbody[vend + "</parameter>".len()..];
    }

    Some(build_tool_call(
        name,
        &Value::Object(args).to_string(),
        id_seed,
        index,
    ))
}

#[cfg(test)]
#[path = "xml_function_tests.rs"]
mod tests;
