//! Engine abstraction. v1: llama.cpp. Future: MLX (mlxcel cxx pattern), other runtimes.

pub mod llamacpp;

use crate::diagnostic::HiggsError;

/// One chat message on the engine boundary (OpenAI role vocabulary).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EngineMessage {
    pub role: String,
    pub content: String,
}

/// Generation parameters for one request.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GenParams {
    pub max_tokens: usize,
    pub temperature: f32,
}

/// Parameters fixed at load time.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, ts_rs::TS)]
#[ts(export, export_to = "../../../frontend/src/lib/generated/")]
pub struct LoadParams {
    #[ts(type = "number")]
    pub ctx_len: u32,
    /// Layers offloaded to GPU; u32::MAX = all (LM Studio "max" semantics).
    #[ts(type = "number")]
    pub gpu_layers: u32,
    #[ts(type = "number")]
    pub threads: u32,
}

/// Result returned by [`HiggsEngine::chat`], carrying the generated text,
/// finish reason, and token counts for OpenAI-standard `usage` reporting.
pub struct ChatResult {
    /// Concatenated full text of the generation.
    pub content: String,
    /// OpenAI finish_reason: `"stop"` (EOG) or `"length"` (max_tokens hit).
    pub finish_reason: &'static str,
    /// Number of tokens in the rendered prompt (after chat-template application).
    pub prompt_tokens: u32,
    /// Number of tokens emitted during the decode loop.
    pub completion_tokens: u32,
}

/// A local inference engine able to host one loaded model (v1).
pub trait HiggsEngine: Send {
    /// Load the GGUF at `path`. Fails with [HG004] on engine errors.
    fn load(&mut self, path: &str, params: &LoadParams) -> Result<(), HiggsError>;
    /// Unload the current model (no-op when nothing is loaded).
    fn unload(&mut self);
    /// True when a model is resident.
    fn is_loaded(&self) -> bool;
    /// Render the GGUF-embedded chat template over `messages` and stream
    /// completion deltas into `sink`. Returns a [`ChatResult`] with the full
    /// text, finish reason, and prompt/completion token counts.
    /// Fails with [HG005] when the prompt cannot fit, [HG004] on load-state errors,
    /// [HG011] on generation failures (context create, prompt decode, sampler, detokenize,
    /// loop decode).
    fn chat(
        &mut self,
        messages: &[EngineMessage],
        params: &GenParams,
        sink: &mut dyn FnMut(&str),
    ) -> Result<ChatResult, HiggsError>;
}
