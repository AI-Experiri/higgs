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
use llama_cpp_2::model::{AddBos, LlamaChatMessage, LlamaChatTemplate, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;

use super::{EngineMessage, GenParams, HiggsEngine, LoadParams};
use crate::diagnostic::HiggsError;

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
        messages: &[EngineMessage],
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
        let chat: Vec<LlamaChatMessage> = messages
            .iter()
            .map(|m| LlamaChatMessage::new(m.role.clone(), m.content.clone()))
            .collect::<Result<_, _>>()
            .map_err(|e| gen_fail("chat message build", e.to_string()))?;
        let prompt = loaded
            .model
            .apply_chat_template(&template, &chat, true)
            .map_err(|e| gen_fail("apply chat template", e.to_string()))?;

        // Fit check BEFORE any decode: prompt + full generation budget must fit.
        let tokens = loaded
            .model
            .str_to_token(&prompt, AddBos::Always)
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
                sink(&piece);
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
            sink(&tail);
            full.push_str(&tail);
        }

        Ok(super::ChatResult {
            content: full,
            finish_reason,
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
                &[EngineMessage {
                    role: "user".into(),
                    content: "Say hi in one word.".into(),
                }],
                &GenParams {
                    max_tokens: 8,
                    temperature: 0.0,
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
