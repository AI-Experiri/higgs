//! Llama-3 JSON tool-call family.
//!
//! Handles the format the Llama-3.x instruct chat templates teach:
//!
//! ```text
//! <|python_tag|>{"name": "get_weather", "parameters": {"city": "Paris"}}
//! {"name": "get_weather", "parameters": {"city": "Paris"}}
//! ```
//!
//! Llama-3.1 prefixes tool calls with the `<|python_tag|>` special token
//! (`Environment: ipython`); Llama-3.2's template instructs a BARE JSON object
//! (`Respond in the format {"name": function name, "parameters": …}`). Both
//! carry the arguments under `"parameters"` (`"arguments"` accepted for
//! lenience). Parallel 3.1-style calls are `;`-separated after the tag.
//!
//! FALLBACK for the crate's primary `parse_response_oaicompat` (restored by
//! the AI-Experiri llama-cpp-rs fork), and the capability sniff behind
//! `supports_tools` for this family.

use serde_json::Value;

use super::{build_tool_call, strip_leading_reasoning, ToolCallParser};

const PYTHON_TAG: &str = "<|python_tag|>";

/// Parser for the Llama-3 JSON (`<|python_tag|>` / bare `{"name": …,
/// "parameters": …}`) family.
pub struct LlamaJsonParser;

impl ToolCallParser for LlamaJsonParser {
    fn id(&self) -> &'static str {
        "llama-json"
    }

    fn handles(&self, chat_template: &str) -> bool {
        // Llama-3.1 templates render the builtin-tools `<|python_tag|>`;
        // Llama-3.2's JSON-tool template embeds this literal instruction line.
        // Neither marker appears in any other family's template.
        chat_template.contains(PYTHON_TAG)
            || chat_template.contains(r#"{"name": function name, "parameters": dictionary"#)
    }

    fn open_markers(&self) -> &'static [&'static str] {
        // The bare-JSON call's literal prefix `{"name"` — both dialects emit
        // it. A false positive (a JSON ANSWER that merely starts like a call)
        // is safe: the decode loop flushes the withheld text at end-of-turn
        // when the full text does not parse as a call (see
        // `ToolCallStreamFilter::take_suppressed`). RESIDUAL: a call emitted
        // with nonstandard spacing (`{ "name"…`) streams as content deltas;
        // the final parse still extracts it for the non-streaming response.
        &[PYTHON_TAG, "{\"name\""]
    }

    fn parse(&self, text: &str, id_seed: &str) -> Option<Vec<Value>> {
        // A tool-call turn is the call itself (optionally after a reasoning
        // block) — not JSON embedded in prose. Strip reasoning, strip the
        // optional tag, then require the remainder to BE the call(s).
        let s = strip_leading_reasoning(text, "</think>").trim().to_string();
        let s = s.strip_prefix(PYTHON_TAG).unwrap_or(&s).trim().to_string();
        if !s.starts_with('{') {
            return None;
        }

        // 3.1 parallel style: `{…};{…}`. Segment with a JSON-aware scan —
        // `serde_json`'s stream deserializer consumes exactly one object per
        // step (a `;` INSIDE a JSON string never splits) — then require the
        // only inter-object separators to be `;`/whitespace.
        let mut calls = Vec::new();
        let mut rest = s.as_str();
        loop {
            // Between objects the ONLY legal filler is `;` + whitespace — two
            // call-shaped objects butted together without a separator are
            // malformed output, not a parallel call, and drop the turn.
            let trimmed = rest.trim_start_matches(|c: char| c == ';' || c.is_whitespace());
            if !calls.is_empty() && trimmed.len() == rest.len() && !rest.is_empty() {
                return None;
            }
            rest = trimmed;
            if rest.is_empty() {
                break;
            }
            let mut stream = serde_json::Deserializer::from_str(rest).into_iter::<Value>();
            let parsed = stream.next()?.ok()?;
            let consumed = stream.byte_offset();
            let name = parsed.get("name")?.as_str()?;
            // Llama uses "parameters"; accept "arguments" for lenience.
            let args = parsed
                .get("parameters")
                .or_else(|| parsed.get("arguments"))?
                .as_object()?;
            calls.push(build_tool_call(
                name,
                &Value::Object(args.clone()).to_string(),
                id_seed,
                calls.len(),
            ));
            rest = &rest[consumed..];
        }
        if calls.is_empty() {
            None
        } else {
            Some(calls)
        }
    }

    fn content(&self, text: &str) -> String {
        // A parsed turn IS the call (whole-turn format) — no surrounding
        // content survives; a non-call turn is returned untouched by the
        // caller (parse() returned None), so empty is correct here.
        let _ = text;
        String::new()
    }
}

#[cfg(test)]
#[path = "llama_json_tests.rs"]
mod tests;
