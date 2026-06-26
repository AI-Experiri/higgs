//! FunctionGemma `<start_function_call>…<end_function_call>` tool-call family.
//!
//! Handles the call format Gemma function-calling GGUF chat templates declare —
//! a `call:NAME{key:value,…}` body wrapped in start/end markers:
//!
//! ```text
//! <start_function_call>call:get_weather{city:<escape>Paris<escape>}<end_function_call>
//! ```
//!
//! Body grammar (ported from Ollama's `functiongemma.go`):
//! - `call:NAME{ARGS}` — NAME is everything up to the first `{`, ARGS the body
//!   inside the outermost `{}`.
//! - ARGS is `key:value` pairs joined by top-level commas. Commas inside nested
//!   `{}` / `[]` do not split (depth tracking), and commas inside an
//!   `<escape>…<escape>` span are literal too.
//! - Values are typed: `true`/`false` → bool, integer/float literals → number,
//!   `[…]` → array, `{…}` → object (recursively), and `<escape>…<escape>` is a
//!   literal string (the escape tags are stripped). Anything else stays a string.
//!
//! Divergences from the Go source, kept deliberate:
//! - Keys, values, and segments are NOT trimmed — we match Go's `splitArguments`
//!   /`parseArguments` verbatim, which preserves surrounding whitespace. The
//!   GGUF template never emits padded `key:value` pairs, so this is observable
//!   only on malformed input, where matching the reference is the safer choice.
//! - An empty function name (`call:{…}`) is SKIPPED here; Go appends an
//!   empty-named call. Skipping is safer — an unnamed call is never dispatchable
//!   — so we drop it rather than emit a malformed tool call.
//!
//! We parse it because llama.cpp's vendored `common_chat` has no parser for this
//! family — the markers are Gemma-specific and unique to this template.

use serde_json::{Map, Value};

use super::{build_tool_call, content_outside_calls, strip_leading_reasoning, ToolCallParser};

const OPEN: &str = "<start_function_call>";
const CLOSE: &str = "<end_function_call>";
const ESCAPE: &str = "<escape>";

/// Parser for the FunctionGemma `<start_function_call>call:NAME{…}` family.
pub struct FunctionGemmaParser;

impl ToolCallParser for FunctionGemmaParser {
    fn id(&self) -> &'static str {
        "function-gemma"
    }

    fn handles(&self, chat_template: &str) -> bool {
        // The Gemma function-calling template renders calls with these literal
        // markers; the `<start_function_call>` / `<end_function_call>` pair is
        // unique to this family and does not collide with XML `<tool_call>` or
        // JSON `[TOOL_CALLS]` formats.
        chat_template.contains(OPEN) && chat_template.contains(CLOSE)
    }

    fn open_markers(&self) -> &'static [&'static str] {
        &[OPEN]
    }

    fn parse(&self, text: &str, id_seed: &str) -> Option<Vec<Value>> {
        let mut calls = Vec::new();
        let mut rest = text;
        while let Some(start) = rest.find(OPEN) {
            let after = &rest[start + OPEN.len()..];
            let Some(end) = after.find(CLOSE) else {
                break;
            };
            if let Some(call) = parse_one(&after[..end], id_seed, calls.len()) {
                calls.push(call);
            }
            rest = &after[end + CLOSE.len()..];
        }
        if calls.is_empty() {
            None
        } else {
            Some(calls)
        }
    }

    fn content(&self, text: &str) -> String {
        let out = content_outside_calls(text, OPEN, CLOSE);
        strip_leading_reasoning(&out, "</think>").trim().to_string()
    }
}

/// Parse one `call:NAME{ARGS}` body into an OpenAI tool-call object.
fn parse_one(block: &str, id_seed: &str, index: usize) -> Option<Value> {
    let call = block.strip_prefix("call:").or_else(|| {
        // Tolerate leading whitespace/newlines before `call:`.
        let trimmed = block.trim_start();
        trimmed.strip_prefix("call:")
    })?;
    let brace = call.find('{')?;
    let name = call[..brace].trim();
    if name.is_empty() {
        // Divergence (documented in the module header): Go emits an empty-named
        // call here; we skip it because an unnamed call is never dispatchable.
        return None;
    }
    // Arguments live inside the outermost `{…}`. We take everything from the
    // first `{` to the last `}` (Ollama's regex is greedy to the final brace).
    let args_body = &call[brace + 1..];
    let args_body = args_body.strip_suffix('}').unwrap_or({
        match args_body.rfind('}') {
            Some(rb) => &args_body[..rb],
            None => args_body,
        }
    });

    let args = parse_object(args_body);

    Some(build_tool_call(
        name,
        &Value::Object(args).to_string(),
        id_seed,
        index,
    ))
}

/// Parse a `key:value,key:value` body into a JSON object.
///
/// Keys and values are taken verbatim (no trimming) to match Go's
/// `parseArguments`/`parseObject`.
fn parse_object(body: &str) -> Map<String, Value> {
    let mut map = Map::new();
    for part in split_top_level(body) {
        let Some(colon) = part.find(':') else {
            continue;
        };
        let key = &part[..colon];
        let value = parse_value(&part[colon + 1..]);
        if !key.is_empty() {
            map.insert(key.to_string(), value);
        }
    }
    map
}

/// Parse a single value, coercing by FunctionGemma type rules.
fn parse_value(value: &str) -> Value {
    // Escaped string: `<escape>…<escape>` — strip the tags, keep the literal.
    if let Some(inner) = value
        .strip_prefix(ESCAPE)
        .and_then(|v| v.strip_suffix(ESCAPE))
    {
        return Value::String(inner.to_string());
    }
    if value == "true" {
        return Value::Bool(true);
    }
    if value == "false" {
        return Value::Bool(false);
    }
    if let Some(num) = parse_number(value) {
        return num;
    }
    if let Some(inner) = value.strip_prefix('[').and_then(|v| v.strip_suffix(']')) {
        let arr = split_top_level(inner)
            .into_iter()
            .map(parse_value)
            .collect();
        return Value::Array(arr);
    }
    if let Some(inner) = value.strip_prefix('{').and_then(|v| v.strip_suffix('}')) {
        return Value::Object(parse_object(inner));
    }
    Value::String(value.to_string())
}

/// Parse an integer or float literal. Returns `None` for non-numeric input.
/// Only the whole string counts (an int must round-trip exactly, matching the
/// Go `Sscanf`-then-reformat check).
fn parse_number(s: &str) -> Option<Value> {
    if let Ok(i) = s.parse::<i64>() {
        if i.to_string() == s {
            return Some(Value::Number(i.into()));
        }
    }
    if let Ok(f) = s.parse::<f64>() {
        return serde_json::Number::from_f64(f).map(Value::Number);
    }
    None
}

/// Split a `key:value` body on top-level commas, respecting `{}` / `[]` nesting
/// and `<escape>…<escape>` spans (commas inside either are not separators).
///
/// Iterates on char boundaries so every returned `&str` slice is UTF-8-safe —
/// a multibyte value such as `日本語` never produces a mid-codepoint slice.
fn split_top_level(body: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0i32;
    let mut in_escape = false;
    let mut start = 0usize;
    let mut chars = body.char_indices().peekable();
    while let Some((i, ch)) = chars.next() {
        if body[i..].starts_with(ESCAPE) {
            in_escape = !in_escape;
            // Advance past the rest of the `<escape>` tag on char boundaries.
            let tag_end = i + ESCAPE.len();
            while chars.peek().is_some_and(|&(j, _)| j < tag_end) {
                chars.next();
            }
            continue;
        }
        if !in_escape {
            match ch {
                '{' | '[' => depth += 1,
                '}' | ']' => depth -= 1,
                ',' if depth == 0 => {
                    let seg = &body[start..i];
                    if !seg.is_empty() {
                        parts.push(seg);
                    }
                    // Resume after this single-byte comma.
                    start = i + 1;
                }
                _ => {}
            }
        }
    }
    let seg = &body[start..];
    if !seg.is_empty() {
        parts.push(seg);
    }
    parts
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    /// Byte layout matching Ollama's `functiongemma_test.go` ground truth.
    const SINGLE: &str =
        "<start_function_call>call:get_weather{city:<escape>Paris<escape>}<end_function_call>";

    fn p() -> FunctionGemmaParser {
        FunctionGemmaParser
    }

    fn args_of(call: &Value) -> Value {
        serde_json::from_str(call["function"]["arguments"].as_str().unwrap()).unwrap()
    }

    #[test]
    fn handles_gemma_template_only() {
        assert!(p().handles("… '<start_function_call>call:' ~ name … '<end_function_call>' …"));
        assert!(!p().handles("<tool_call>\n<function= … no gemma markers"));
    }

    #[test]
    fn parses_single_call() {
        let calls = p().parse(SINGLE, "abc").expect("one call");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["function"]["name"], "get_weather");
        assert_eq!(calls[0]["type"], "function");
        assert_eq!(calls[0]["id"], "call_abc_0");
        assert_eq!(args_of(&calls[0])["city"], "Paris");
    }

    #[test]
    fn parses_multiple_args_with_typed_values() {
        // numbers, bools, and a JSON-object value alongside an escaped string.
        let text = "<start_function_call>call:configure{a:1,ratio:0.5,enabled:true,name:<escape>x<escape>,opts:{deep:true}}<end_function_call>";
        let calls = p().parse(text, "z").unwrap();
        let a = args_of(&calls[0]);
        assert_eq!(a["a"], 1);
        assert_eq!(a["ratio"], 0.5);
        assert_eq!(a["enabled"], true);
        assert_eq!(a["name"], "x");
        assert_eq!(a["opts"]["deep"], true); // nested object, top-level comma respected
    }

    #[test]
    fn parses_array_argument() {
        let text = "<start_function_call>call:process{items:[<escape>a<escape>,<escape>b<escape>,<escape>c<escape>]}<end_function_call>";
        let a = args_of(&p().parse(text, "q").unwrap()[0]);
        assert_eq!(a["items"], json!(["a", "b", "c"]));
    }

    #[test]
    fn parses_two_tool_calls() {
        let text = "<start_function_call>call:get_weather{city:<escape>Paris<escape>}<end_function_call><start_function_call>call:get_weather{city:<escape>London<escape>}<end_function_call>";
        let calls = p().parse(text, "s").unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(args_of(&calls[0])["city"], "Paris");
        assert_eq!(args_of(&calls[1])["city"], "London");
        assert_eq!(calls[1]["id"], "call_s_1");
    }

    #[test]
    fn no_tool_call_returns_none() {
        assert!(p().parse("Hello, world!", "x").is_none());
        // unterminated: open marker but no close.
        assert!(p().parse("<start_function_call>call:x{", "x").is_none());
    }

    #[test]
    fn content_strips_preamble_and_think() {
        // Content before the first call is reported.
        let t = "Let me check.<start_function_call>call:get_weather{city:<escape>Paris<escape>}<end_function_call>";
        assert_eq!(p().content(t), "Let me check.");
        // Leading reasoning block stripped (forced-open: trailing </think> only).
        let forced = "thinking out loud\n</think>\nLet me check.<start_function_call>call:x{}<end_function_call>";
        assert_eq!(p().content(forced), "Let me check.");
    }

    // --- Regression tests for the ported bugs ---------------------------------

    /// Bug 1: a multibyte UTF-8 value (`日本語`) inside an escaped string must
    /// not panic. The old byte-stepping `split_top_level` sliced mid-codepoint.
    #[test]
    fn parses_multibyte_utf8_value() {
        let text =
            "<start_function_call>call:translate{text:<escape>日本語<escape>}<end_function_call>";
        let calls = p().parse(text, "u").expect("one call");
        assert_eq!(args_of(&calls[0])["text"], "日本語");
    }

    /// Bug 1 (split path): a top-level comma after a multibyte escaped value
    /// still splits correctly and on a char boundary.
    #[test]
    fn splits_after_multibyte_value() {
        let text = "<start_function_call>call:f{a:<escape>café<escape>,b:1}<end_function_call>";
        let a = args_of(&p().parse(text, "u").unwrap()[0]);
        assert_eq!(a["a"], "café");
        assert_eq!(a["b"], 1);
    }

    /// JSON-quoted value with an embedded comma stays inside the escaped span —
    /// the comma must not split the pair. The literal includes the quotes since
    /// only `<escape>` strips, not JSON quoting.
    #[test]
    fn escaped_value_with_embedded_comma_does_not_split() {
        let text =
            "<start_function_call>call:f{msg:<escape>hello, world<escape>,n:2}<end_function_call>";
        let a = args_of(&p().parse(text, "u").unwrap()[0]);
        assert_eq!(a["msg"], "hello, world");
        assert_eq!(a["n"], 2);
    }

    /// Bug 2: content AFTER a tool call is reported (Go `content_after_tool_call`).
    #[test]
    fn content_after_tool_call() {
        let t = "<start_function_call>call:test{}<end_function_call>Done!";
        let calls = p().parse(t, "a").unwrap();
        assert_eq!(calls[0]["function"]["name"], "test");
        assert_eq!(p().content(t), "Done!");
    }

    /// Bug 2: content BETWEEN two tool calls is reported
    /// (Go `content_between_tool_calls`).
    #[test]
    fn content_between_tool_calls() {
        let t = "<start_function_call>call:first{}<end_function_call>Some text here<start_function_call>call:second{}<end_function_call>";
        let calls = p().parse(t, "b").unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0]["function"]["name"], "first");
        assert_eq!(calls[1]["function"]["name"], "second");
        assert_eq!(p().content(t), "Some text here");
    }

    /// Bug 4: an empty function name (`call:{…}`) is skipped, not emitted.
    #[test]
    fn empty_name_call_is_skipped() {
        assert!(p()
            .parse("<start_function_call>call:{a:1}<end_function_call>", "x")
            .is_none());
    }
}
