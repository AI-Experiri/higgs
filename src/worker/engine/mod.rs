//! Engine abstraction. v1: llama.cpp. Future: MLX (mlxcel cxx pattern), other runtimes.

pub mod llamacpp;

use crate::diagnostic::HiggsError;

/// Generation parameters for one request.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GenParams {
    pub max_tokens: usize,
    pub temperature: f32,
    /// OpenAI-compatible `tools` array serialized to a JSON string, or `None`
    /// when the request carries no tools. Fed verbatim to the GGUF chat
    /// template alongside the messages; the crate's vendored `common_chat`
    /// renders the tool grammar and selects the matching tool-call parser. We
    /// invent no parser of our own.
    pub tools_json: Option<String>,
}

higgs_ts! {
    /// Parameters fixed at load time.
    ///
    /// The three base fields (`ctx_len`/`gpu_layers`/`threads`) are always
    /// present — the quick-load / `default_load` path fills them. Every other
    /// field is `Option`: absent (`None`) means "use the engine default", which
    /// reproduces the pre-expansion behavior exactly. Each optional maps to a
    /// real `llama-cpp-2` 0.1.139 builder call, applied only inside `llamacpp.rs`
    /// (the sole file allowed to name `llama_cpp_2`).
    #[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
    #[serde(default)]
    pub struct LoadParams {
        #[ts(type = "number")]
        pub ctx_len: u32,
        /// Layers offloaded to GPU; u32::MAX = all (LM Studio "max" semantics).
        #[ts(type = "number")]
        pub gpu_layers: u32,
        #[ts(type = "number")]
        pub threads: u32,
        /// Memory-map the GGUF instead of reading it into RAM. `None` = engine
        /// default. Applied via `LlamaModelParams::with_use_mmap`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        pub use_mmap: Option<bool>,
        /// Lock model pages in RAM (prevent swap). `None` = engine default.
        /// Applied via `LlamaModelParams::with_use_mlock`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        pub use_mlock: Option<bool>,
        /// Logical batch size for prompt decode. `None` keeps the current
        /// default (`ctx_len.max(1)` — one-shot prefill). Applied via
        /// `LlamaContextParams::with_n_batch`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        #[ts(type = "number")]
        #[ts(optional)]
        pub n_batch: Option<u32>,
        /// Physical (micro) batch size. `None` = engine default. Applied via
        /// `LlamaContextParams::with_n_ubatch`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        #[ts(type = "number")]
        #[ts(optional)]
        pub n_ubatch: Option<u32>,
        /// Offload the KV cache & KQV ops to the GPU. `None` = engine default.
        /// Applied via `LlamaContextParams::with_offload_kqv`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        pub offload_kqv: Option<bool>,
        /// RoPE base frequency override. `None` = use the GGUF's trained value.
        /// Applied via `LlamaContextParams::with_rope_freq_base`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        #[ts(type = "number")]
        #[ts(optional)]
        pub rope_freq_base: Option<f32>,
        /// RoPE frequency scale (context extension). `None` = trained value.
        /// Applied via `LlamaContextParams::with_rope_freq_scale`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        #[ts(type = "number")]
        #[ts(optional)]
        pub rope_freq_scale: Option<f32>,
        /// Flash-attention policy. `None` = engine default. Applied via
        /// `LlamaContextParams::with_flash_attention_policy`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        pub flash_attn: Option<FlashAttn>,
        /// KV cache key data type. `None` = engine default (F16). Applied via
        /// `LlamaContextParams::with_type_k`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        pub type_k: Option<KvCacheKind>,
        /// KV cache value data type. `None` = engine default (F16). Applied via
        /// `LlamaContextParams::with_type_v`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        pub type_v: Option<KvCacheKind>,
        /// Sampler RNG seed. `None` keeps the current behavior: a fresh random
        /// seed per request (`LlamaSampler::dist(rand::random())`). When set,
        /// generation is reproducible. Greedy decoding (temperature 0) ignores it.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        #[ts(type = "number")]
        #[ts(optional)]
        pub seed: Option<u32>,
    }
}

higgs_ts! {
    /// Flash-attention policy, mirroring llama.cpp's `llama_flash_attn_type`
    /// (AUTO = -1, DISABLED = 0, ENABLED = 1). Engine-agnostic at this layer;
    /// mapped to the raw `llama_cpp_sys_2` value only inside `llamacpp.rs`.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
    #[serde(rename_all = "lowercase")]
    pub enum FlashAttn {
        /// Let llama.cpp decide per-model (raw value -1).
        Auto,
        /// Force flash attention off (raw value 0).
        Off,
        /// Force flash attention on (raw value 1).
        On,
    }
}

higgs_ts! {
    /// KV-cache element type (the LM Studio subset of GGML types usable for the
    /// K/V cache). Engine-agnostic here; mapped to `llama_cpp_2`'s `KvCacheType`
    /// only inside `llamacpp.rs`.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
    pub enum KvCacheKind {
        /// 32-bit float — highest precision, largest cache.
        F32,
        /// 16-bit float — the engine default.
        F16,
        /// 8-bit, block size 0.
        Q8_0,
        /// 5-bit, block size 1.
        Q5_1,
        /// 5-bit, block size 0.
        Q5_0,
        /// 4-bit, block size 1.
        Q4_1,
        /// 4-bit, block size 0 — smallest cache.
        Q4_0,
    }
}

/// Result returned by [`HiggsEngine::chat`], carrying the generated text,
/// finish reason, and token counts for OpenAI-standard `usage` reporting.
pub struct ChatResult {
    /// Assistant text after tool-call parsing — the `content` field of the
    /// OpenAI message produced by `parse_response_oaicompat`. When the model
    /// emitted tool calls, the call markup is stripped out and lives in
    /// [`tool_calls`](Self::tool_calls) instead; otherwise this is the full
    /// generation verbatim.
    pub content: String,
    /// OpenAI finish_reason: `"stop"` (EOG) or `"length"` (max_tokens hit).
    /// Note: when [`tool_calls`](Self::tool_calls) is present the boundary
    /// (`serve`) reports `"tool_calls"` per the OpenAI spec.
    pub finish_reason: &'static str,
    /// Parsed OpenAI `tool_calls` array (verbatim from `parse_response_oaicompat`),
    /// or `None` when the model emitted no tool calls.
    pub tool_calls: Option<serde_json::Value>,
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
    /// Render the GGUF-embedded chat template over `messages_json` and stream
    /// completion deltas into `sink`. Returns a [`ChatResult`] with the full
    /// text, finish reason, and prompt/completion token counts.
    ///
    /// `messages_json` is the request's OpenAI `messages` array serialized
    /// verbatim — including assistant `tool_calls` and tool `tool_call_id`, so
    /// multi-turn tool loops round-trip. This is the engine-neutral boundary:
    /// each engine renders this same OpenAI JSON by its own means (llama.cpp via
    /// `common_chat`; a future MLX engine via its own jinja renderer). The
    /// template-apply mechanism never crosses above this trait.
    ///
    /// Fails with [HG005] when the prompt cannot fit, [HG003] when no model is
    /// loaded, [HG011] on generation failures (context create, prompt decode,
    /// sampler, detokenize, loop decode).
    fn chat(
        &mut self,
        messages_json: &str,
        params: &GenParams,
        sink: &mut dyn FnMut(&str),
    ) -> Result<ChatResult, HiggsError>;

    /// Attempt to load the GGUF at `path` into a throwaway model handle to learn
    /// whether THIS engine can load it (Gate 1 — "can our llama.cpp load it").
    ///
    /// Probe-only: the loaded handle is dropped immediately and never stored as
    /// the resident model, so a probe never disturbs a model already being
    /// served (`&self`, no resident-slot mutation). Returns `(true, None)` when
    /// the load succeeds, or `(false, Some(reason))` carrying the engine's
    /// verbatim error string (e.g. `"unknown model architecture: 'gemma4'"`)
    /// when it fails — that exact reason is what the UI shows as the mismatch.
    fn probe(&self, path: &str) -> (bool, Option<String>);

    /// Enumerate the host's compute devices (CPU/GPU/accel) as this engine sees
    /// them. Cheap and read-only — no model load, no resident-state mutation —
    /// so it is safe to call at any time, including on a fresh worker. Returns an
    /// empty vec when the engine exposes no devices.
    fn devices(&self) -> Vec<crate::system::GpuDevice>;
}
