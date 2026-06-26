//! Gemma-4 `call:NAME{…}` tool-call family.
//!
//! Gemma 4 emits tool calls with a bespoke envelope and a bespoke argument
//! syntax — not JSON and not XML:
//!
//! ```text
//! <|tool_call>call:get_weather{location:<|"|>Paris<|"|>,units:<|"|>metric<|"|>}<tool_call|>
//! ```
//!
//! Inside the `{…}` block, argument values are encoded as follows:
//!   - strings are wrapped in the Gemma string delimiter `<|"|>VALUE<|"|>`
//!     (so the value may contain literal `"`, newlines, etc. unescaped),
//!   - everything else (numbers, `true`/`false`/`null`, nested `{…}` objects,
//!     `[…]` arrays) is written as bare JSON literals — including ordinary
//!     JSON double-quoted strings (`"a,b"`),
//!   - object keys are bare identifiers (`location:`), not quoted.
//!
//! To recover OpenAI arguments we translate this to JSON: extract each
//! `<|"|>…<|"|>` span as a JSON string, quote the bare keys, and feed the rest
//! to a JSON parser (numbers/bools/null/objects/arrays/JSON strings parse as-is).
//!
//! Thinking, when enabled, uses Gemma's own channel markers
//! `<|channel>thought\n…<channel|>` — distinct from the `<think>` block other
//! families use.
//!
//! We parse this ourselves because llama.cpp's vendored `common_chat` does not
//! cover the Gemma-4 dialect.
//!
//! Omitted vs. Ollama's Go parser: the union-type / unclosed-delimiter REPAIR
//! heuristics (`repairGemma4*`) are intentionally not ported — they are
//! schema-gated, best-effort recovery for malformed model output and are not
//! needed for the common, well-formed path. We parse the common path only.

use serde_json::{Map, Value};

use super::{build_tool_call, content_outside_calls, strip_leading_reasoning, ToolCallParser};

/// Opening envelope marker for a Gemma-4 tool call.
const OPEN_TAG: &str = "<|tool_call>";
/// Closing envelope marker for a Gemma-4 tool call.
const CLOSE_TAG: &str = "<tool_call|>";
/// Gemma's string-value delimiter (wraps string args, both sides).
const STR_DELIM: &str = "<|\"|>";
/// Closing marker for a Gemma-4 thinking (channel) block (`<|channel>thought…`
/// opens it; we strip up to and including this close tag).
const CHANNEL_CLOSE_TAG: &str = "<channel|>";

/// Parser for the Gemma-4 `call:NAME{…}` tool-call family.
pub struct Gemma4Parser;

impl ToolCallParser for Gemma4Parser {
    fn id(&self) -> &'static str {
        "gemma4"
    }

    fn handles(&self, chat_template: &str) -> bool {
        // The Gemma-4 template renders calls with the literal `<|tool_call>` /
        // `<tool_call|>` envelope and `call:` prefix, and wraps string args in
        // the `<|"|>` delimiter. These markers are unique to Gemma-4 and do not
        // collide with the generic `<tool_call>` (no pipe) XML family.
        chat_template.contains(OPEN_TAG) && chat_template.contains(STR_DELIM)
    }

    fn open_markers(&self) -> &'static [&'static str] {
        &[OPEN_TAG]
    }

    fn parse(&self, text: &str, id_seed: &str) -> Option<Vec<Value>> {
        let mut calls = Vec::new();
        let mut rest = text;
        while let Some(start) = rest.find(OPEN_TAG) {
            let after = &rest[start + OPEN_TAG.len()..];
            // A call ends at the close tag; if absent, the call is unterminated.
            let Some(end) = after.find(CLOSE_TAG) else {
                break;
            };
            if let Some(call) = parse_one(&after[..end], id_seed, calls.len()) {
                calls.push(call);
            }
            rest = &after[end + CLOSE_TAG.len()..];
        }
        if calls.is_empty() {
            None
        } else {
            Some(calls)
        }
    }

    fn content(&self, text: &str) -> String {
        let out = content_outside_calls(text, OPEN_TAG, CLOSE_TAG);
        strip_leading_reasoning(strip_channel(&out), "</think>")
            .trim()
            .to_string()
    }
}

/// Remove a leading Gemma-4 thinking (channel) block. Gemma 4 emits reasoning
/// as `<|channel>thought\n…<channel|>` (the channel name prefix `thought\n`
/// lives inside the block). Everything up to and including the first
/// `<channel|>` close tag is dropped, so the reasoning markup never leaks into
/// assistant content. When no close tag is present the input is returned
/// unchanged.
fn strip_channel(s: &str) -> &str {
    match s.find(CHANNEL_CLOSE_TAG) {
        Some(end) => &s[end + CHANNEL_CLOSE_TAG.len()..],
        None => s,
    }
}

/// Parse one `call:NAME{args}` body (the text between the envelope tags) into
/// an OpenAI tool-call object.
fn parse_one(block: &str, id_seed: &str, index: usize) -> Option<Value> {
    let block = block.trim();
    let after_call = block.strip_prefix("call:")?;
    // Name runs up to the opening brace of the args object.
    let brace = after_call.find('{')?;
    let name = after_call[..brace].trim();
    if name.is_empty() {
        return None;
    }
    let args_block = &after_call[brace..];

    let json_str = gemma4_args_to_json(args_block);
    // The common path produces a valid JSON object; if it does not, treat the
    // call as unparseable (repair heuristics are intentionally not ported).
    let args = match serde_json::from_str::<Value>(&json_str) {
        Ok(Value::Object(map)) => map,
        _ => Map::new(),
    };

    Some(build_tool_call(
        name,
        &Value::Object(args).to_string(),
        id_seed,
        index,
    ))
}

/// Translate Gemma-4's argument syntax into JSON text.
///
/// Three transforms, applied in a single byte-oriented pass:
///   1. each `<|"|>…<|"|>` span becomes a JSON-escaped string literal,
///   2. each ordinary JSON double-quoted value (`"a,b:c"`) is copied verbatim,
///      so structural bytes (`,` `:` `{`) inside the string are never
///      reinterpreted,
///   3. each bare object key (an identifier immediately after `{` or `,` and
///      followed by `:`) is wrapped in double quotes.
///
/// Everything else (numbers, `true`/`false`/`null`, structural `{}[]:,`) is
/// passed through untouched, so it parses as ordinary JSON.
///
/// The pass works on bytes and rebuilds a `String` at the end. Every `&str` slice
/// below lands on a char boundary: marker/delimiter scans hit ASCII bytes, and a
/// bare value/key advances by whole UTF-8 runes (`bare_key_end` and the rune-width
/// copy below), so Unicode survives intact.
fn gemma4_args_to_json(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len() + 16);
    let mut i = 0;
    while i < bytes.len() {
        // A Gemma string span: copy its inner text out as a JSON string literal.
        if s[i..].starts_with(STR_DELIM) {
            let inner_start = i + STR_DELIM.len();
            if let Some(rel) = s[inner_start..].find(STR_DELIM) {
                let inner = &s[inner_start..inner_start + rel];
                out.extend_from_slice(Value::String(inner.to_string()).to_string().as_bytes());
                i = inner_start + rel + STR_DELIM.len();
                continue;
            }
            // Unterminated delimiter — copy the rest verbatim and stop.
            out.extend_from_slice(&bytes[i..]);
            break;
        }

        // An ordinary JSON double-quoted value: copy the whole quoted span
        // (open quote → first UNESCAPED close quote) untouched, so commas,
        // colons, and braces inside the string are never treated as structure.
        if bytes[i] == b'"' {
            if let Some(end) = json_quoted_string_end(bytes, i) {
                out.extend_from_slice(&bytes[i..end]);
                i = end;
                continue;
            }
        }

        // Copy one rune: a non-ASCII lead byte begins a multibyte rune (a bare
        // value/key), so advance its full width — a single-byte step would land `i`
        // mid-codepoint and panic the next `s[i..]` slice.
        let c = bytes[i];
        let w = if c < 0x80 {
            1
        } else {
            s[i..].chars().next().map_or(1, char::len_utf8)
        };
        out.extend_from_slice(&bytes[i..i + w]);
        i += w;

        // After a `{` or `,`, a bare identifier key followed by `:` is quoted.
        if c == b'{' || c == b',' {
            let ks = skip_space(s, i);
            out.extend_from_slice(&bytes[i..ks]);
            let ke = bare_key_end(s, ks);
            if ke > ks && ke < bytes.len() && bytes[ke] == b':' {
                out.push(b'"');
                out.extend_from_slice(&bytes[ks..ke]);
                out.extend_from_slice(b"\":");
                i = ke + 1;
            } else {
                i = ks;
            }
        }
    }
    // `out` is assembled from valid-UTF-8 fragments of `s` plus ASCII structure
    // and serde's UTF-8 string encoding, so it is always valid UTF-8.
    String::from_utf8(out).expect("gemma4 args translation produced valid UTF-8")
}

/// End index (exclusive, just past the closing quote) of a JSON double-quoted
/// string starting at the opening quote `start`. Honors backslash escaping so
/// an escaped `\"` does not terminate the span. Returns `None` when the string
/// is unterminated.
fn json_quoted_string_end(bytes: &[u8], start: usize) -> Option<usize> {
    let mut escaped = false;
    let mut i = start + 1;
    while i < bytes.len() {
        let c = bytes[i];
        if c == b'"' && !escaped {
            return Some(i + 1);
        }
        escaped = c == b'\\' && !escaped;
        i += 1;
    }
    None
}

/// Index of the first non-space byte at or after `start`.
fn skip_space(s: &str, start: usize) -> usize {
    let bytes = s.as_bytes();
    let mut i = start;
    while i < bytes.len() && bytes[i].is_ascii_whitespace() {
        i += 1;
    }
    i
}

/// End index of a bare key (`_`, ASCII letters, ASCII digits, or any non-ASCII
/// UTF-8 letter/digit) starting at `start`. Returns `start` when the first
/// char is not a key char. Advances over whole UTF-8 runes so the returned
/// index is always on a char boundary.
fn bare_key_end(s: &str, start: usize) -> usize {
    let mut i = start;
    for c in s[start..].chars() {
        // ASCII key chars plus any non-ASCII alphanumeric (e.g. accented or
        // CJK identifier characters Gemma may emit as a bare key).
        if c == '_' || c.is_ascii_alphanumeric() || (!c.is_ascii() && c.is_alphanumeric()) {
            i += c.len_utf8();
        } else {
            break;
        }
    }
    i
}

#[cfg(test)]
#[path = "gemma4_tests.rs"]
mod tests;
