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
//! mlx-lm / omlx use), so the right parser is known before generation, which the
//! future streaming path also needs.

mod deepseek3;
mod function_gemma;
mod gemma4;
mod glm_xml;
mod mistral_bracket;
mod qwen_json;
mod stream_filter;
mod xml_function;

use serde_json::Value;

pub use deepseek3::DeepSeek3Parser;
pub use function_gemma::FunctionGemmaParser;
pub use gemma4::Gemma4Parser;
pub use glm_xml::GlmXmlParser;
pub use mistral_bracket::MistralBracketParser;
pub use qwen_json::QwenJsonParser;
pub use stream_filter::ToolCallStreamFilter;
pub use xml_function::XmlFunctionParser;

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
}
