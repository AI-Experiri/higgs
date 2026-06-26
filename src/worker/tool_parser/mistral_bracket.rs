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
        // Remove each `[TOOL_CALLS]NAME[ARGS]{json}` span and KEEP the text outside every
        // call — preamble, gaps between calls, and the trailing remainder (the shared
        // `content()` contract). This dialect has no closing marker, so — unlike the paired
        // parsers' `content_outside_calls` — the span end is found exactly as `parse()` finds
        // it: `[ARGS]` then the balanced JSON object. An incomplete trailing envelope (no
        // `[ARGS]`, or unterminated JSON) is dropped as incomplete markup, not content.
        let mut out = String::new();
        let mut rest = text;
        let trailing = loop {
            let Some(start) = rest.find(TOOL_CALLS_TAG) else {
                break rest; // no further call — the remainder is content
            };
            out.push_str(&rest[..start]);
            let after = &rest[start + TOOL_CALLS_TAG.len()..];
            let Some(asep) = after.find(ARGS_TAG) else {
                break ""; // incomplete call (no [ARGS]) — drop the rest
            };
            let args_body = &after[asep + ARGS_TAG.len()..];
            let Some(jend) = find_json_end(args_body) else {
                break ""; // unterminated JSON — drop the rest
            };
            rest = &args_body[jend + 1..];
        };
        out.push_str(trailing);
        strip_leading_reasoning(&out, THINK_END_TAG)
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
#[path = "mistral_bracket_tests.rs"]
mod tests;
