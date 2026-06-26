//! Engine-agnostic tool-call parser registry.
//!
//! A tool-call parser turns a model's generated TEXT into structured OpenAI
//! `tool_calls`. Text is the one output every engine produces identically, so
//! this layer sits ABOVE the [`HiggsEngine`](crate::worker::engine::HiggsEngine)
//! trait and is reused unchanged by every backend (llama.cpp today; MLX, CUDA,
//! etc. in future). Nothing here imports an engine type — the only input is a
//! `&str`.
//!
//! It exists because llama.cpp's vendored `common_chat` auto-parser only covers
//! ~6 families and rejects valid output for others (e.g. `nemotron_h`'s
//! `<function=…><parameter=…>` XML). Each registered [`ToolCallParser`] owns one
//! format family, declared by the model's own GGUF chat template — not a
//! per-model catalog. Selection sniffs the chat template (the approach
//! mlx-lm / omlx use), so the right parser is known before generation — the
//! [`ToolCallStreamFilter`] reuses that selection to suppress the call envelope
//! from streamed content.

mod deepseek3;
mod function_gemma;
mod gemma4;
mod glm_xml;
mod mistral_bracket;
mod qwen_json;
mod stream_filter;
mod xml_function;

use serde_json::{json, Value};

pub use deepseek3::DeepSeek3Parser;
pub use function_gemma::FunctionGemmaParser;
pub use gemma4::Gemma4Parser;
pub use glm_xml::GlmXmlParser;
pub use mistral_bracket::MistralBracketParser;
pub use qwen_json::QwenJsonParser;
pub use stream_filter::ToolCallStreamFilter;
pub use xml_function::XmlFunctionParser;

/// Remove a leading reasoning block from `s`, given the family's reasoning
/// close tag (`</think>`, `[/THINK]`, …).
///
/// Handles both an explicit `<open>…<close_tag>` block and the
/// template-forced-open case (no opening tag, just a trailing `close_tag`):
/// everything up to and including the first `close_tag` is dropped. When no
/// close tag is present the input is returned unchanged. `close_tag` must be
/// non-empty.
pub(crate) fn strip_leading_reasoning<'a>(s: &'a str, close_tag: &str) -> &'a str {
    match s.find(close_tag) {
        Some(end) => &s[end + close_tag.len()..],
        None => s,
    }
}

/// The assistant text OUTSIDE every complete `open`..`close` tool-call span —
/// preamble, gaps between calls, and the trailing remainder — concatenated in order,
/// so non-call text after/between calls is preserved (the `content` contract). An
/// unterminated `open` drops the rest (incomplete call markup, not content).
///
/// Span detection is the FIRST `close` after each `open` — IDENTICAL to every parser's
/// `parse()` (e.g. `qwen_json`/`glm_xml`), so `content` removes exactly the spans `parse`
/// extracts. The shared limitation: a call whose ARGUMENTS literally contain the `close`
/// sentinel (e.g. a JSON string `"…</tool_call>…"`) cuts the span early — `parse` then drops
/// that call (its body is truncated/invalid) and a tail can survive here. Accepted: real chat
/// templates never emit the close sentinel inside an argument, and these parsers are
/// deliberately complete-text only (no streaming/partial-tag heuristics — see module docs);
/// handling it correctly needs structured, JSON-aware span detection across every parser.
pub(crate) fn content_outside_calls(text: &str, open: &str, close: &str) -> String {
    let mut out = String::new();
    let mut rest = text;
    while let Some(start) = rest.find(open) {
        out.push_str(&rest[..start]);
        let after = &rest[start + open.len()..];
        match after.find(close) {
            Some(end) => rest = &after[end + close.len()..],
            None => return out,
        }
    }
    out.push_str(rest);
    out
}

/// Build one OpenAI tool-call object from a function name and its already-
/// serialized `arguments` JSON string.
///
/// `arguments` is emitted verbatim as the OpenAI `arguments` field (which is a
/// JSON STRING, not an object). The call `id` follows the shared
/// `call_{id_seed}_{index}` format so ids are stable across the families.
pub(crate) fn build_tool_call(name: &str, arguments: &str, id_seed: &str, index: usize) -> Value {
    json!({
        "id": format!("call_{id_seed}_{index}"),
        "type": "function",
        "function": {
            "name": name,
            // OpenAI `arguments` is a JSON STRING, not an object.
            "arguments": arguments,
        }
    })
}

/// One tool-call output format family (e.g. the XML `<function=…>` family).
///
/// Implementations are pure text transforms — given the model's generated text
/// they extract OpenAI `tool_calls`. They hold no per-request state, so one
/// instance serves every request and every engine.
pub trait ToolCallParser: Send + Sync {
    /// Stable identifier for logs/diagnostics (e.g. `"xml-function"`).
    fn id(&self) -> &'static str;

    /// Whether this parser handles the model whose GGUF chat template is
    /// `chat_template` — matched on the format markers the template declares.
    fn handles(&self, chat_template: &str) -> bool;

    /// Literal strings that open a tool call in this format (e.g.
    /// `["<tool_call>"]`). The streaming filter withholds content from the
    /// first occurrence so the call envelope never leaks as assistant text.
    fn open_markers(&self) -> &'static [&'static str];

    /// Parse complete generated `text` into the OpenAI `tool_calls` array, or
    /// `None` when the turn emitted no tool call. `id_seed` supplies the per-call
    /// id suffix so the caller controls id generation.
    fn parse(&self, text: &str, id_seed: &str) -> Option<Vec<Value>>;

    /// The assistant `content` to report alongside parsed calls: the generated
    /// text with tool-call markup and any leading reasoning block removed.
    fn content(&self, text: &str) -> String;
}

/// The set of known tool-call parsers, selected per model by chat-template sniff.
///
/// Stateless and cheap to construct; an engine holds one and consults it when
/// its own (engine-specific) primary parser cannot handle the output.
pub struct ToolParserRegistry {
    parsers: Vec<Box<dyn ToolCallParser>>,
}

impl ToolParserRegistry {
    /// Registry with every built-in parser. New format families are added here.
    ///
    /// Order is most-specific → most-generic: `select` returns the first
    /// `handles` match, so families that share an opening marker (all the
    /// `<tool_call>` dialects) must be ordered so the discriminating ones win.
    /// `QwenJsonParser` is the generic `<tool_call>`-JSON catch-all and stays
    /// last; its `handles` already excludes the `<function=`/`<arg_key>` markers
    /// the XML and GLM families use, so the ordering is belt-and-suspenders.
    pub fn with_defaults() -> Self {
        Self {
            parsers: vec![
                Box::new(XmlFunctionParser), // <function=…><parameter=…> (Nemotron, Qwen3-Coder)
                Box::new(GlmXmlParser),      // <tool_call> + <arg_key>/<arg_value> (GLM)
                Box::new(DeepSeek3Parser),   // <｜tool▁calls▁begin｜> unicode tags
                Box::new(MistralBracketParser), // [TOOL_CALLS] … [ARGS]
                Box::new(Gemma4Parser),      // <|tool_call> … <tool_call|>
                Box::new(FunctionGemmaParser), // <start_function_call> … <end_function_call>
                Box::new(QwenJsonParser),    // <tool_call>{json} — generic, last
            ],
        }
    }

    /// The parser for the model whose chat template is `chat_template`, or `None`
    /// when no registered parser recognizes the format. First match wins.
    pub fn select(&self, chat_template: &str) -> Option<&dyn ToolCallParser> {
        self.parsers
            .iter()
            .find(|p| p.handles(chat_template))
            .map(AsRef::as_ref)
    }
}

impl Default for ToolParserRegistry {
    fn default() -> Self {
        Self::with_defaults()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Marker fragment present in the Nemotron / Qwen3-Coder GGUF chat template.
    const XML_TEMPLATE: &str =
        "…{{- '<tool_call>\\n<function=' ~ tool_call.name }} … <parameter=' ~ args_name …";

    #[test]
    fn selects_xml_parser_by_template() {
        let reg = ToolParserRegistry::with_defaults();
        let p = reg.select(XML_TEMPLATE).expect("xml parser selected");
        assert_eq!(p.id(), "xml-function");
    }

    #[test]
    fn no_match_for_unrelated_template() {
        let reg = ToolParserRegistry::with_defaults();
        // A plain chatml/JSON template has no <function=…> markers.
        assert!(reg
            .select("{% for m in messages %}<|im_start|>{{ m.role }}")
            .is_none());
    }

    #[test]
    fn strip_leading_reasoning_explicit_block() {
        // Explicit <think>…</think> — everything up to and including the close.
        assert_eq!(
            strip_leading_reasoning("<think>reasoning</think>answer", "</think>"),
            "answer"
        );
    }

    #[test]
    fn strip_leading_reasoning_forced_open() {
        // Template forced the block open → only a trailing close tag.
        assert_eq!(
            strip_leading_reasoning("reasoning</think>answer", "</think>"),
            "answer"
        );
    }

    #[test]
    fn strip_leading_reasoning_no_close_tag_unchanged() {
        assert_eq!(
            strip_leading_reasoning("plain answer", "</think>"),
            "plain answer"
        );
    }

    #[test]
    fn strip_leading_reasoning_alternate_close_tag() {
        // Mistral's [/THINK] close tag.
        assert_eq!(
            strip_leading_reasoning("reasoning[/THINK]done", "[/THINK]"),
            "done"
        );
    }

    #[test]
    fn build_tool_call_shape() {
        let call = build_tool_call("get_weather", r#"{"location":"Paris"}"#, "abc", 0);
        assert_eq!(call["id"], "call_abc_0");
        assert_eq!(call["type"], "function");
        assert_eq!(call["function"]["name"], "get_weather");
        // arguments is a JSON STRING verbatim, not an object.
        assert_eq!(call["function"]["arguments"], r#"{"location":"Paris"}"#);
    }

    #[test]
    fn build_tool_call_index_in_id() {
        let call = build_tool_call("f", "{}", "s", 2);
        assert_eq!(call["id"], "call_s_2");
    }
}
