//! llama.cpp engine via the `llama-cpp-2` crate (Metal by default on macOS).
//!
//! The ONLY file allowed to import `llama_cpp_2`. The decode loop copies the
//! crate's own `examples/simple`: prompt batch feed, then per token
//! sample → EOG check → detokenize → sink → re-batch → decode.

use std::num::NonZeroU32;
use std::sync::OnceLock;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaChatTemplate, LlamaModel};
use llama_cpp_2::openai::OpenAIChatTemplateParams;
use llama_cpp_2::sampling::LlamaSampler;

use super::{GenParams, HiggsEngine, LoadParams};
use crate::diagnostic::HiggsError;
use crate::worker::tool_parser::{ToolCallStreamFilter, ToolParserRegistry};

/// Process-wide llama.cpp backend handle — the FFI global init must run
/// exactly once per process.
static BACKEND: OnceLock<LlamaBackend> = OnceLock::new();

/// Initialize-once accessor for the llama.cpp backend.
fn backend() -> &'static LlamaBackend {
    // init()'s only error is BackendAlreadyInitialized, unreachable under OnceLock.
    BACKEND.get_or_init(|| LlamaBackend::init().expect("sole llama backend init"))
}

/// A resident model plus the load-time state `chat()` needs to serve it.
struct LoadedModel {
    model: LlamaModel,
    /// Load-time knobs; `ctx_len`/`threads` shape the per-request context.
    params: LoadParams,
}

/// llama.cpp-backed [`HiggsEngine`]. Hosts one loaded model at a time (v1);
/// each chat builds a fresh context (naive full re-prefill per request).
#[derive(Default)]
pub struct LlamaCppEngine {
    loaded: Option<LoadedModel>,
    /// Engine-agnostic fallback parsers, consulted when the crate's own
    /// template parser rejects the output (e.g. nemotron XML). Shared shape:
    /// a future MLX/CUDA engine constructs the same registry.
    tool_parsers: ToolParserRegistry,
}

impl HiggsEngine for LlamaCppEngine {
    fn load(&mut self, path: &str, params: &LoadParams) -> Result<(), HiggsError> {
        // Drop any resident model first — one loaded model at a time (v1).
        self.loaded = None;
        let model_params = LlamaModelParams::default().with_n_gpu_layers(params.gpu_layers);
        let model = LlamaModel::load_from_file(backend(), path, &model_params).map_err(|e| {
            HiggsError::EngineLoadFailed {
                id: path.to_string(),
                reason: e.to_string(),
            }
        })?;
        self.loaded = Some(LoadedModel {
            model,
            params: params.clone(),
        });
        Ok(())
    }

    fn unload(&mut self) {
        self.loaded = None;
    }

    fn is_loaded(&self) -> bool {
        self.loaded.is_some()
    }

    fn chat(
        &mut self,
        messages_json: &str,
        params: &GenParams,
        sink: &mut dyn FnMut(&str),
    ) -> Result<super::ChatResult, HiggsError> {
        let Some(loaded) = self.loaded.as_ref() else {
            // defensive guard; worker checks first — id unknown at engine level
            return Err(HiggsError::ModelNotLoaded {
                id: "unloaded".into(),
            });
        };
        let gen_fail = |stage: &'static str, reason: String| HiggsError::GenerationFailed {
            stage: stage.to_string(),
            reason,
        };

        // GGUF-embedded chat template; fall back to "chatml" when the model embeds none.
        let template = match loaded.model.chat_template(None) {
            Ok(t) => t,
            Err(e) => {
                tracing::warn!(error = %e, "GGUF chat template unavailable; falling back to chatml");
                LlamaChatTemplate::new("chatml")
                    .map_err(|e| gen_fail("chatml fallback template", e.to_string()))?
            }
        };
        // Single path for tools and no-tools alike: the OAI-compat apply parses
        // the verbatim OpenAI `messages` JSON (preserving assistant `tool_calls`
        // and tool `tool_call_id` for multi-turn tool loops), renders the GGUF
        // template (with the tools array when present), AND returns the
        // serialized PEG parser + chat_format the crate's vendored `common_chat`
        // selected for this model. We keep `tmpl_result` alive across the decode
        // so `parse_response_oaicompat` can turn the raw output back into an
        // OpenAI message (content + tool_calls) — no parser invented here.
        //
        // `add_bos: false` — the prompt is tokenized below with `AddBos::Always`,
        // so the template must not also prepend BOS (would double it).
        // (Grammar-constrained sampling via tmpl_result.grammar is deferred.)
        let oai_params = OpenAIChatTemplateParams {
            messages_json,
            tools_json: params.tools_json.as_deref(),
            tool_choice: None,
            json_schema: None,
            grammar: None,
            reasoning_format: None,
            chat_template_kwargs: None,
            add_generation_prompt: true,
            use_jinja: true,
            parallel_tool_calls: false,
            enable_thinking: false,
            add_bos: false,
            add_eos: false,
            parse_tool_calls: params.tools_json.is_some(),
        };
        let tmpl_result = loaded
            .model
            .apply_chat_template_oaicompat(&template, &oai_params)
            .map_err(|e| gen_fail("apply chat template", e.to_string()))?;
        let prompt = tmpl_result.prompt.as_str();

        // Fit check BEFORE any decode: prompt + full generation budget must fit.
        let tokens = loaded
            .model
            .str_to_token(prompt, AddBos::Always)
            .map_err(|e| gen_fail("tokenize prompt", e.to_string()))?;
        let n_ctx = loaded.params.ctx_len as usize;
        if tokens.len() + params.max_tokens > n_ctx {
            return Err(HiggsError::ContextOverflow {
                prompt_tokens: tokens.len(),
                max_gen: params.max_tokens,
                n_ctx,
            });
        }
        // prompt_tokens is the length of the tokenized, template-applied prompt.
        let prompt_tokens = tokens.len() as u32;

        if params.max_tokens == 0 {
            return Ok(super::ChatResult {
                content: String::new(),
                finish_reason: "length",
                tool_calls: None,
                prompt_tokens,
                completion_tokens: 0,
            });
        }

        // Fresh context per request (v1 simplicity: naive full re-prefill).
        // n_batch sized to the context so any fit-checked prompt decodes in
        // one llama_decode call (llama.cpp's simple.cpp sizes it the same way).
        let threads = i32::try_from(loaded.params.threads).unwrap_or(i32::MAX);
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(loaded.params.ctx_len))
            .with_n_batch(loaded.params.ctx_len.max(1))
            .with_n_threads(threads)
            .with_n_threads_batch(threads);
        let mut ctx = loaded
            .model
            .new_context(backend(), ctx_params)
            .map_err(|e| gen_fail("create context", e.to_string()))?;

        // Prompt feed: logits only for the last prompt token (example shape).
        let mut batch = LlamaBatch::new(tokens.len().max(1), 1);
        let last_index = tokens.len().saturating_sub(1);
        for (i, token) in tokens.into_iter().enumerate() {
            let pos = i32::try_from(i).map_err(|e| gen_fail("prompt position", e.to_string()))?;
            batch
                .add(token, pos, &[0], i == last_index)
                .map_err(|e| gen_fail("batch add", e.to_string()))?;
        }
        ctx.decode(&mut batch)
            .map_err(|e| gen_fail("prompt decode", e.to_string()))?;

        // Greedy when temperature is zero, else temp + dist with real per-request
        // entropy (F5: was hardcoded seed 1234 — fully deterministic across requests).
        let mut sampler = if params.temperature <= 0.0 {
            LlamaSampler::chain_simple([LlamaSampler::greedy()])
        } else {
            LlamaSampler::chain_simple([
                LlamaSampler::temp(params.temperature),
                LlamaSampler::dist(rand::random::<u32>()),
            ])
        };

        // Tool-call streaming: when tools were requested and a registry parser
        // recognizes this model's format, suppress the call envelope from the
        // streamed content (the structured tool_calls are emitted at end of
        // stream by `serve::stream`, parsed from `full` below). The same parser
        // does the final parse in the fallback branch.
        let tmpl_str = template.to_str().unwrap_or("");
        let selected_parser = params
            .tools_json
            .as_ref()
            .and_then(|_| self.tool_parsers.select(tmpl_str));
        let mut stream_filter =
            selected_parser.map(|p| ToolCallStreamFilter::new(p.open_markers()));

        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut full = String::new();
        let mut n_generated: usize = 0;
        let mut n_cur = batch.n_tokens();
        let finish_reason = loop {
            let token = sampler.sample(&ctx, batch.n_tokens() - 1);
            sampler.accept(token);
            if loaded.model.is_eog_token(token) {
                break "stop";
            }
            let piece = loaded
                .model
                .token_to_piece(token, &mut decoder, true, None)
                .map_err(|e| gen_fail("detokenize", e.to_string()))?;
            // The UTF-8 decoder buffers partial multi-byte sequences — only
            // forward pieces that decoded to visible text.
            if !piece.is_empty() {
                match stream_filter.as_mut() {
                    Some(f) => f.push(&piece, sink),
                    None => sink(&piece),
                }
                full.push_str(&piece);
            }
            n_generated += 1;
            if n_generated >= params.max_tokens {
                break "length";
            }
            batch.clear();
            batch
                .add(token, n_cur, &[0], true)
                .map_err(|e| gen_fail("batch add", e.to_string()))?;
            n_cur += 1;
            ctx.decode(&mut batch)
                .map_err(|e| gen_fail("loop decode", e.to_string()))?;
        };

        // Flush the UTF-8 decoder: a response ending mid-multi-byte sequence (e.g.
        // CJK) would otherwise silently truncate the final character. The final call
        // with last=true drains any buffered incomplete bytes.
        let mut tail = String::new();
        let _ = decoder.decode_to_string(&[], &mut tail, true);
        if !tail.is_empty() {
            match stream_filter.as_mut() {
                Some(f) => f.push(&tail, sink),
                None => sink(&tail),
            }
            full.push_str(&tail);
        }
        // Flush any safe content the filter held back (a tail that never became
        // a marker); suppressed content stays withheld.
        if let Some(f) = stream_filter.as_mut() {
            f.finish(sink);
        }

        // Parse the full generation into an OpenAI message.
        //   Primary: the parser the template apply selected — covers the
        //     families llama.cpp's vendored common_chat handles (Qwen, Mistral,
        //     Llama-3, Hermes, …), all derived from the GGUF template.
        //   Fallback: when that parser rejects the output — which it does for
        //     formats it cannot auto-derive, e.g. nemotron_h's
        //     `<function=…><parameter=…>` XML — parse the format the GGUF
        //     template itself declares, via `tool_parse`. Both paths read the
        //     GGUF; neither curates per-model.
        let (content, tool_calls) = match tmpl_result.parse_response_oaicompat(&full, false) {
            Ok(msg_json) => {
                let parsed: serde_json::Value = serde_json::from_str(&msg_json)
                    .map_err(|e| gen_fail("parse response json", e.to_string()))?;
                let tool_calls = parsed.get("tool_calls").filter(|v| !v.is_null()).cloned();
                // `content` is null when the turn is purely tool calls. Only fall
                // back to the raw generated text when there are NO tool calls —
                // otherwise the tool-call markup would be returned as assistant
                // content *alongside* the structured tool_calls (OpenAI requires
                // content to be empty/null on a tool-call turn).
                let content = parsed
                    .get("content")
                    .and_then(|v| v.as_str())
                    .map(ToOwned::to_owned)
                    .unwrap_or_else(|| {
                        if tool_calls.is_some() {
                            String::new()
                        } else {
                            full.clone()
                        }
                    });
                (content, tool_calls)
            }
            Err(e) => {
                // Crate parser declined. Use the registry parser already selected
                // for this model (by chat-template sniff) to parse the text.
                match selected_parser {
                    Some(parser) => {
                        let seed = uuid::Uuid::new_v4().simple().to_string();
                        match parser.parse(&full, &seed) {
                            Some(calls) => {
                                tracing::debug!(error = %e, parser = parser.id(), "crate parse rejected output; registry parser recovered tool calls");
                                (parser.content(&full), Some(serde_json::Value::Array(calls)))
                            }
                            // Parser matched the format but the turn had no call.
                            None => (full.clone(), None),
                        }
                    }
                    // No registered parser for this model's format: preserve text.
                    None => {
                        tracing::warn!(error = %e, "crate parse rejected output and no registry parser matched; returning raw text");
                        (full.clone(), None)
                    }
                }
            }
        };

        Ok(super::ChatResult {
            content,
            finish_reason,
            tool_calls,
            prompt_tokens,
            // n_generated counts tokens emitted in the decode loop (one per iteration).
            completion_tokens: n_generated as u32,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Phase-1 milestone: first real token from a local GGUF through the
    /// full template → tokenize → decode → detokenize path.
    #[test]
    #[ignore = "needs HIGGS_TEST_GGUF pointing at a small local gguf"]
    fn first_token_from_real_model() {
        let path = std::env::var("HIGGS_TEST_GGUF").expect("set HIGGS_TEST_GGUF");
        let mut e = LlamaCppEngine::default();
        e.load(
            &path,
            &LoadParams {
                ctx_len: 2048,
                gpu_layers: u32::MAX,
                threads: 4,
            },
        )
        .unwrap();
        assert!(e.is_loaded());
        let mut out = String::new();
        let result = e
            .chat(
                r#"[{"role":"user","content":"Say hi in one word."}]"#,
                &GenParams {
                    max_tokens: 8,
                    temperature: 0.0,
                    tools_json: None,
                },
                &mut |d| out.push_str(d),
            )
            .unwrap();
        println!(
            "model said: {:?} (finish_reason={}, prompt_tokens={}, completion_tokens={})",
            result.content, result.finish_reason, result.prompt_tokens, result.completion_tokens
        );
        assert!(!result.content.is_empty());
        assert_eq!(result.content, out);
        assert!(result.finish_reason == "stop" || result.finish_reason == "length");
        assert!(result.prompt_tokens > 0, "prompt_tokens must be non-zero");
        assert!(
            result.completion_tokens > 0,
            "completion_tokens must be non-zero"
        );
        e.unload();
        assert!(!e.is_loaded());
    }
}
