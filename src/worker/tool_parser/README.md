# tool_parser

## Table of Contents
- [What It Is](#what-it-is)
- [Files](#files)

## What It Is

Engine-agnostic tool-call parsers. A parser turns a model's generated **text**
into structured OpenAI `tool_calls`. Text is the one output every inference
engine produces identically, so this layer sits **above** the `HiggsEngine`
trait — nothing here imports an engine type; the only input is a `&str`. The
same registry is reused unchanged by llama.cpp today and any future MLX/CUDA
engine.

It exists because llama.cpp's vendored `common_chat` auto-parser only covers a
handful of format families and rejects valid output for others (e.g.
`nemotron_h`'s `<function=…><parameter=…>` XML). The crate parser is the
primary; this registry is the fallback. Each parser owns one format family and
declares whether it `handles` a model by sniffing the model's own GGUF chat
template — not a per-model catalog.

Adding a format = one `ToolCallParser` impl + one line in
`ToolParserRegistry::with_defaults()`.

## Files

| Path | What it does |
|------|-------------|
| `mod.rs` | `ToolCallParser` trait, `ToolParserRegistry` (template-sniff selection, ordering rationale), re-exports |
| `stream_filter.rs` | `ToolCallStreamFilter` — suppresses the tool-call envelope from streamed content deltas (marker-aware, holds back partial markers) |
| `xml_function.rs` | `XmlFunctionParser` — `<function=…><parameter=…>` XML (Nemotron, Qwen3-Coder) |
| `qwen_json.rs` | `QwenJsonParser` — `<tool_call>{json}` ChatML JSON (Qwen3 / Qwen3-VL); generic catch-all, ordered last |
| `deepseek3.rs` | `DeepSeek3Parser` — `<｜tool▁calls▁begin｜>` unicode-tag envelope (DeepSeek-V3 / R1) |
| `glm_xml.rs` | `GlmXmlParser` — `<tool_call>` + `<arg_key>`/`<arg_value>` pairs (GLM-4.x) |
| `mistral_bracket.rs` | `MistralBracketParser` — `[TOOL_CALLS]name[ARGS]{json}` (Mistral / Ministral) |
| `gemma4.rs` | `Gemma4Parser` — `<|tool_call>call:NAME{…}<tool_call|>` bespoke envelope + arg syntax (Gemma 4) |
| `function_gemma.rs` | `FunctionGemmaParser` — `<start_function_call>call:NAME{…}<end_function_call>` (FunctionGemma) |
