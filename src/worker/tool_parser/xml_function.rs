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
mod tests {
    use super::*;

    /// The exact format the Nemotron GGUF template documents.
    const NEMOTRON: &str = "<think>\nThe user wants weather.\n</think>\n<tool_call>\n<function=get_weather>\n<parameter=location>\nParis\n</parameter>\n</function>\n</tool_call>\n";

    fn p() -> XmlFunctionParser {
        XmlFunctionParser
    }

    #[test]
    fn handles_xml_template_only() {
        assert!(p().handles("… '<function=' ~ name … '<parameter=' ~ k …"));
        assert!(!p().handles("<|im_start|>{{ role }} plain chatml, no markers"));
    }

    #[test]
    fn parses_nemotron_single_call() {
        let calls = p().parse(NEMOTRON, "abc").expect("one call");
        assert_eq!(calls.len(), 1);
        let f = &calls[0]["function"];
        assert_eq!(f["name"], "get_weather");
        let args: Value = serde_json::from_str(f["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(args["location"], "Paris");
        assert_eq!(calls[0]["type"], "function");
        assert_eq!(calls[0]["id"], "call_abc_0");
    }

    #[test]
    fn content_strips_think_and_tool_call() {
        assert_eq!(p().content(NEMOTRON), "");
        let with_preamble = "Sure.\n<tool_call>\n<function=x>\n</function>\n</tool_call>";
        assert_eq!(p().content(with_preamble), "Sure.");
    }

    #[test]
    fn content_strips_forced_open_think_without_opening_tag() {
        // Template forced the think block open → only a trailing </think>.
        let forced = "The user wants weather. Let me do that.\n</think>\n<tool_call>\n<function=get_weather>\n</function>\n</tool_call>";
        assert_eq!(p().content(forced), "");
    }

    #[test]
    fn parses_multiple_params_and_json_values() {
        let text = "<tool_call>\n<function=search>\n<parameter=query>\nrust\n</parameter>\n<parameter=limit>\n5\n</parameter>\n<parameter=filters>\n{\"lang\":\"en\"}\n</parameter>\n</function>\n</tool_call>";
        let calls = p().parse(text, "z").unwrap();
        let args: Value =
            serde_json::from_str(calls[0]["function"]["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(args["query"], "rust");
        assert_eq!(args["limit"], 5); // numeric, via tojson
        assert_eq!(args["filters"]["lang"], "en"); // nested object
    }

    #[test]
    fn parses_two_tool_calls() {
        let text = "<tool_call>\n<function=a>\n</function>\n</tool_call>\n<tool_call>\n<function=b>\n</function>\n</tool_call>";
        let calls = p().parse(text, "s").unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0]["function"]["name"], "a");
        assert_eq!(calls[1]["function"]["name"], "b");
        assert_eq!(calls[1]["id"], "call_s_1");
    }

    #[test]
    fn no_tool_call_returns_none() {
        assert!(p().parse("just a normal answer", "x").is_none());
        assert!(p().parse("<tool_call>\n<function=x>", "x").is_none()); // unterminated
    }
}
