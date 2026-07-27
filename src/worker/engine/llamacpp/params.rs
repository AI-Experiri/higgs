//! Engine-specific load + sampling parameter types for the llama.cpp engine.
//!
//! These are the concrete payloads carried by the engine-tagged umbrellas
//! [`crate::worker::engine::LoadParams`] / [`crate::worker::engine::SamplingParams`]
//! (`LoadParams::LlamaCpp(LlamaCppParams)`, …). They are named and shaped after
//! llama.cpp directly — every field maps to a real `llama-cpp-2 = 0.1.139`
//! builder call applied only inside `llamacpp/mod.rs` (the sole file allowed to
//! name `llama_cpp_2`). A future `MlxParams` would sit beside these as the `Mlx`
//! variant; the suggester (`src/tune/`) derives the llama.cpp set here.
//!
//! Coverage notes (gated on the 0.1.139 binding surface, see `docs/DESIGN-autotune.md` §4):
//! - `split_mode`/`main_gpu`/`devices` are carried for completeness but their
//!   derivation + apply are DEFERRED (single-GPU/Metal target) — no-op defaults.
//! - `tensor_split` is FFI-only (no safe setter) and intentionally absent.
//! - speculative/NextN is C++-only in the vendored tree — out of scope.

use serde::{Deserialize, Serialize};

use crate::worker::engine::{CtxLen, FlashAttn, GpuLayers, KvCacheKind};

higgs_const_enum! {
    /// Multi-GPU split mode, mirroring `llama-cpp-2`'s `LlamaSplitMode`
    /// (`None`/`Layer`/`Row`). Single-GPU/Metal target ⇒ derivation deferred;
    /// the default is a no-op. Mapped to the wrapper enum inside `llamacpp/mod.rs`.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    pub enum SplitMode {
        /// Single device — no split.
        None,
        /// Split by layers across devices (the wrapper default).
        Layer,
        /// Split by rows across devices.
        Row,
    }
}

higgs_const_enum! {
    /// RoPE context-scaling type, mirroring the `llama-cpp-2` 0.1.139 wrapper's
    /// `RopeScalingType` variants (`Unspecified`/`None`/`Linear`/`Yarn`). NOTE the
    /// 0.1.139 wrapper omits the sys-level `LongRope` — reachable only via a newer
    /// wrapper or raw FFI, so it is intentionally absent here.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    pub enum RopeScalingType {
        /// Leave unspecified (use the GGUF's trained value).
        Unspecified,
        /// Disable RoPE scaling.
        None,
        /// Linear position scaling.
        Linear,
        /// YaRN scaling.
        Yarn,
    }
}

higgs_ts! {
    /// A GGUF metadata override applied at load via `append_kv_override` (advanced
    /// / manual). `value` is the textual form llama.cpp parses per the key's type.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct KvOverride {
        /// GGUF metadata key, e.g. `llama.context_length`.
        pub key: String,
        /// Override value as a string (llama.cpp coerces by the key's declared type).
        pub value: String,
    }
}

higgs_ts! {
    /// DRY (Don't Repeat Yourself) repetition-sampler parameters, mapped to
    /// `LlamaSampler::dry(model, multiplier, base, allowed_length, last_n, breakers)`.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct DryParams {
        /// Penalty multiplier (0 disables).
        #[ts(type = "number")]
        pub multiplier: f32,
        /// Exponential base for the penalty growth.
        #[ts(type = "number")]
        pub base: f32,
        /// Minimum sequence length that triggers the penalty.
        #[ts(type = "number")]
        pub allowed_length: i32,
        /// How many recent tokens DRY scans (`-1` = whole context).
        #[ts(type = "number")]
        pub penalty_last_n: i32,
        /// Sequence-breaker strings that reset the DRY match window. Always
        /// serialized (possibly `[]`) — honest required-array binding.
        #[serde(default)]
        pub sequence_breakers: Vec<String>,
    }
}

higgs_ts! {
    /// Mirostat adaptive-perplexity sampler parameters. `version` selects
    /// `mirostat` (1) vs `mirostat_v2` (2); the v1 `m` scalar is fixed to
    /// llama.cpp's default (100) and not exposed.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct MirostatParams {
        /// Algorithm version: `1` or `2`.
        #[ts(type = "number")]
        pub version: u32,
        /// Target entropy (tau).
        #[ts(type = "number")]
        pub tau: f32,
        /// Learning rate (eta).
        #[ts(type = "number")]
        pub eta: f32,
    }
}

higgs_ts! {
    /// A single logit bias: shift `token`'s logit by `bias` before sampling.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct LogitBias {
        /// Vocabulary token id.
        #[ts(type = "number")]
        pub token: i32,
        /// Additive logit bias (negative discourages, positive encourages).
        #[ts(type = "number")]
        pub bias: f32,
    }
}

higgs_ts! {
    /// GBNF grammar constraint, mapped to `LlamaSampler::grammar(model, gbnf, root)`.
    /// Lazy-trigger variants (words/tokens/patterns) are deferred (DESIGN §9).
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct GrammarParams {
        /// The GBNF grammar source.
        pub gbnf: String,
        /// The grammar root rule name (e.g. `"root"`).
        pub root: String,
    }
}

higgs_ts! {
    /// Full llama.cpp load parameters — the `LlamaCpp` payload of the
    /// [`crate::worker::engine::LoadParams`] umbrella. Covers every model+context
    /// knob the `llama-cpp-2` 0.1.139 bindings expose (DESIGN §4a/§4b). The three
    /// base fields (`ctx_len`/`gpu_layers`/`threads`) are always present — the
    /// quick-load / `default_load` / suggester path fills them; "all on GPU" is
    /// the typed `GpuLayers::All` (the old `u32::MAX` sentinel is gone). Every other field
    /// is optional: absent (`None`/empty) means "use the engine default", which
    /// reproduces the pre-expansion behavior exactly.
    #[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, higgs_macros::TsParamHelp)]
    #[serde(default)]
    pub struct LlamaCppParams {
        // ── model params (LlamaModelParams) ──────────────────────────────────
        /// Context window (`with_n_ctx`). [`CtxLen::Auto`] = the engine's trained context
        /// (the old `ctx_len == 0` sentinel); [`CtxLen::Fixed`] pins an explicit window.
        #[help = "How many tokens of conversation the model can hold at once — prompt, chat history, and the reply all share this window. Larger windows let long chats and big documents fit without truncation, but the KV cache grows with it, costing more RAM/VRAM and slower prompt processing; Auto uses the model's trained maximum."]
        pub ctx_len: CtxLen,
        /// Layers offloaded to GPU (`with_n_gpu_layers`). [`GpuLayers::All`] = every layer
        /// (the old `u32::MAX` sentinel); [`GpuLayers::Count`] carries an explicit count.
        #[help = "How many of the model's transformer layers run on the GPU instead of the CPU; All puts the entire model in VRAM for the fastest generation. Lower the count when the model doesn't fit in VRAM — each layer moved to the CPU frees GPU memory but slows inference, since CPU layers are much slower per token."]
        pub gpu_layers: GpuLayers,
        /// Worker threads used during generation (`with_n_threads`; also seeds
        /// `n_threads_batch` when that is unset).
        #[ts(type = "number")]
        #[help = "Number of CPU threads used while generating tokens (and for prompt processing unless a separate batch-thread count is set). More threads speed up the CPU-bound parts up to roughly your physical core count; going beyond that adds scheduling overhead and can compete with other applications for CPU time."]
        pub threads: u32,
        /// Memory-map the GGUF instead of reading it into RAM (`with_use_mmap`).
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        #[help = "Maps the GGUF file directly from disk instead of copying it into process memory, so loads are near-instant and the OS shares pages between processes. Disabling it forces a full read into RAM, which slows loading but avoids page-fault stalls on slow disks during the first pass through the weights."]
        pub use_mmap: Option<bool>,
        /// Lock model pages in RAM, preventing swap (`with_use_mlock`).
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        #[help = "Pins the model's memory pages so the OS can never swap them out to disk, keeping generation latency steady under memory pressure. Only enable it when there's comfortably enough free RAM — locked pages are unavailable to every other application and can starve the rest of the system."]
        pub use_mlock: Option<bool>,
        /// Keep MoE expert tensors on the CPU (`add_cpu_moe_override`, pinned
        /// apply). Boolean only — no numeric `n_cpu_moe` binding exists. The
        /// suggester flips this on under VRAM pressure when RAM still fits.
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        #[help = "For Mixture-of-Experts models, keeps the large expert tensors in system RAM while the attention layers stay on the GPU. This lets an MoE model far bigger than your VRAM still run GPU-accelerated, at the cost of slower expert lookups over the CPU-GPU bus; it does nothing for dense (non-MoE) models."]
        pub cpu_moe: Option<bool>,
        /// Per-tensor-buffer-type → CPU regex overrides (`add_cpu_buft_override`,
        /// pinned apply; advanced/manual). Drives finer MoE offload later. Always
        /// serialized (possibly `[]`) so the bindings' required-array shape is honest.
        #[help = "Advanced: regex patterns matching tensor names that should be kept in CPU memory rather than offloaded to the GPU. Useful for hand-tuning which weights occupy scarce VRAM (e.g. pinning select expert tensors), but a wrong pattern silently moves hot tensors to the CPU and degrades speed."]
        pub cpu_buft_overrides: Vec<String>,
        /// Multi-GPU split mode (`with_split_mode`). DEFERRED-derive; no-op default.
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        #[help = "How the model is divided across multiple GPUs: by layer (each GPU owns whole layers) or by row (tensors are sharded across GPUs). Layer splitting is simpler and usually faster for generation; row splitting can balance very large models better but adds inter-GPU traffic. Irrelevant with a single GPU."]
        pub split_mode: Option<SplitMode>,
        /// Primary GPU index when `split_mode = None` (`with_main_gpu`). DEFERRED.
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(type = "number")]
        #[ts(optional)]
        #[help = "Which GPU hosts the scratch buffers and small tensors when the model is not being split across devices. Change it to steer the load onto a specific card in a multi-GPU machine (e.g. keep the display GPU free); on single-GPU systems index 0 is the only valid choice."]
        pub main_gpu: Option<i32>,
        /// Explicit device selection (`with_devices`). DEFERRED (multi-GPU). Wire
        /// type is `u32` (ts-portable); the apply path converts to `usize`.
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        #[help = "Restricts the load to a specific set of GPUs instead of letting the engine use every visible device. Useful to reserve a card for other work or to avoid a slow GPU dragging down a multi-GPU split; listing a single device effectively pins the whole model to it."]
        pub devices: Option<Vec<u32>>,
        // ── context params (LlamaContextParams) ──────────────────────────────
        /// Logical batch size for prompt decode (`with_n_batch`). `None` keeps the
        /// current default (`ctx_len.max(1)` — one-shot prefill).
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(type = "number")]
        #[ts(optional)]
        #[help = "Maximum number of prompt tokens submitted to the engine in one logical batch during prefill. Larger batches process long prompts faster by amortizing per-call overhead, but need bigger compute buffers; the default covers the whole context in one pass."]
        pub n_batch: Option<u32>,
        /// Physical (micro) batch size (`with_n_ubatch`).
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(type = "number")]
        #[ts(optional)]
        #[help = "The number of tokens actually computed simultaneously in one kernel launch — the logical batch is chopped into chunks of this size. Raising it improves prompt-processing throughput on strong GPUs; lowering it shrinks the compute-buffer memory footprint when VRAM is tight."]
        pub n_ubatch: Option<u32>,
        /// Parallel sequence slots (`with_n_seq_max`); default 1.
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(type = "number")]
        #[ts(optional)]
        #[help = "How many independent sequences the context can serve at once (each gets its own slice of KV cache). Above 1 enables serving concurrent requests from one loaded model, but every extra slot multiplies KV-cache memory; leave at 1 for a single-user chat."]
        pub n_seq_max: Option<u32>,
        /// Threads for batch/prompt processing (`with_n_threads_batch`). `None`
        /// reuses `threads` (today's shared behavior).
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(type = "number")]
        #[ts(optional)]
        #[help = "CPU threads used only for the batch/prefill phase, separate from the generation-thread count. Prefill parallelizes better than generation, so it can profitably use more threads; unset, it simply reuses the generation thread count."]
        pub n_threads_batch: Option<u32>,
        /// Flash-attention policy (`with_flash_attention_policy`).
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        #[help = "Whether to use the fused flash-attention kernel, which computes attention with far less memory traffic — typically faster and lighter on VRAM for long contexts. Forcing it on fails on backends without the kernel, and quantized KV-cache types require it; Auto lets the engine decide per device."]
        pub flash_attn: Option<FlashAttn>,
        /// Offload the KV cache & KQV ops to the GPU (`with_offload_kqv`).
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        #[help = "Keeps the KV cache and the attention (KQV) computation on the GPU rather than in system RAM. On = faster attention at the cost of VRAM that grows with context length; off frees that VRAM for model layers but every generated token pays a slow CPU attention pass."]
        pub offload_kqv: Option<bool>,
        /// Full sliding-window attention (`with_swa_full`).
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        #[help = "For models trained with sliding-window attention (e.g. Gemma), allocates the full KV cache instead of only the window. This enables correct cache reuse for prompts longer than the window (faster re-prompting), but costs the same KV memory as a non-windowed model of equal context."]
        pub swa_full: Option<bool>,
        /// KV cache key data type (`with_type_k`).
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        #[help = "Data type used to store the attention keys, normally f16. Quantized types (q8_0, q4_0) shrink the KV cache substantially — letting much longer contexts fit in memory — at a small accuracy cost, and they require flash attention to be available."]
        pub type_k: Option<KvCacheKind>,
        /// KV cache value data type (`with_type_v`).
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        #[help = "Data type used to store the attention values, normally f16. Quantizing values compounds the memory savings of a quantized key cache for long contexts, but is the more accuracy-sensitive half — prefer quantizing keys first, and note quantized types require flash attention."]
        pub type_v: Option<KvCacheKind>,
        /// RoPE scaling type (`with_rope_scaling_type`).
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        #[help = "Strategy (linear or YaRN) for stretching the model's rotary position embeddings so it can attend beyond its trained context length. Needed only when forcing a context larger than the model was trained for; the wrong mode for a given model degrades long-range coherence."]
        pub rope_scaling_type: Option<RopeScalingType>,
        /// RoPE base frequency override (`with_rope_freq_base`).
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(type = "number")]
        #[ts(optional)]
        #[help = "Overrides the base frequency of the rotary position embedding, the constant that sets how position information rotates through the dimensions. Some fine-tunes ship a non-standard base to reach longer contexts; setting it incorrectly scrambles positional understanding, so only override when the model card says to."]
        pub rope_freq_base: Option<f32>,
        /// RoPE frequency scale / context extension (`with_rope_freq_scale`).
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(type = "number")]
        #[ts(optional)]
        #[help = "Linear scale factor applied to RoPE positions — 0.5 makes the model treat a 2x-trained-length context as if it were in range. A quick way to extend context without YaRN, but coherence degrades as the factor shrinks; leave unset to respect the trained length."]
        pub rope_freq_scale: Option<f32>,
        /// GGUF metadata overrides (`append_kv_override`, pinned apply; advanced).
        /// Always serialized (possibly `[]`) — honest required-array binding.
        #[help = "Advanced: key=value overrides applied on top of the GGUF file's own metadata at load time, e.g. to correct a wrong architecture field or force an experimental setting. These change how the engine interprets the model, so a bad override can make a working model fail to load or produce garbage."]
        pub kv_overrides: Vec<KvOverride>,
        /// Load-pinned sampler RNG seed. `None` = a fresh random seed per request.
        /// A per-request `LlamaCppSamplingParams::seed` overrides this when set.
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(type = "number")]
        #[ts(optional)]
        #[help = "Fixes the random seed used by the samplers for every request served by this load, making generations reproducible for testing and comparisons. Unset, each request draws a fresh seed for natural variety; a per-request seed in the sampling parameters still overrides this value."]
        pub seed: Option<u32>,
    }
}

higgs_ts! {
    /// Full llama.cpp sampler parameters — the `LlamaCpp` payload of the
    /// [`crate::worker::engine::SamplingParams`] umbrella. Every sampler the
    /// 0.1.139 bindings expose is representable (DESIGN §4c); `None` ⇒ omit that
    /// sampler from the chain. The chain order is fixed in `llamacpp/mod.rs`. The
    /// suggester only ever auto-fills the common ones (temp/top_k/top_p/min_p/
    /// penalties from the HF card); the rest are user-settable but never derived.
    #[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
    #[serde(default)]
    pub struct LlamaCppSamplingParams {
        /// Sampling temperature (`temp`). `<= 0` ⇒ greedy.
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(type = "number")]
        #[ts(optional)]
        pub temperature: Option<f32>,
        /// Dynamic-temperature range (`temp_ext`); pairs with `dynatemp_exponent`.
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(type = "number")]
        #[ts(optional)]
        pub dynatemp_range: Option<f32>,
        /// Dynamic-temperature exponent (`temp_ext`).
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(type = "number")]
        #[ts(optional)]
        pub dynatemp_exponent: Option<f32>,
        /// Top-k cutoff (`top_k`).
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(type = "number")]
        #[ts(optional)]
        pub top_k: Option<i32>,
        /// Top-p / nucleus cutoff (`top_p`, with `min_keep = 1`).
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(type = "number")]
        #[ts(optional)]
        pub top_p: Option<f32>,
        /// Min-p cutoff (`min_p`, with `min_keep = 1`).
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(type = "number")]
        #[ts(optional)]
        pub min_p: Option<f32>,
        /// Locally-typical-p cutoff (`typical`, with `min_keep = 1`).
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(type = "number")]
        #[ts(optional)]
        pub typical_p: Option<f32>,
        /// Top-n-sigma cutoff (`top_n_sigma`).
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(type = "number")]
        #[ts(optional)]
        pub top_n_sigma: Option<f32>,
        /// XTC probability (`xtc`); pairs with `xtc_threshold`.
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(type = "number")]
        #[ts(optional)]
        pub xtc_probability: Option<f32>,
        /// XTC threshold (`xtc`).
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(type = "number")]
        #[ts(optional)]
        pub xtc_threshold: Option<f32>,
        /// Repetition-penalty window (`penalties` `last_n`).
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(type = "number")]
        #[ts(optional)]
        pub penalty_last_n: Option<i32>,
        /// Repetition penalty (`penalties` `repeat`).
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(type = "number")]
        #[ts(optional)]
        pub penalty_repeat: Option<f32>,
        /// Frequency penalty (`penalties` `freq`).
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(type = "number")]
        #[ts(optional)]
        pub penalty_freq: Option<f32>,
        /// Presence penalty (`penalties` `present`).
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(type = "number")]
        #[ts(optional)]
        pub penalty_present: Option<f32>,
        /// DRY repetition sampler (`dry`).
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        pub dry: Option<DryParams>,
        /// Mirostat adaptive sampler (`mirostat`/`mirostat_v2`).
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        pub mirostat: Option<MirostatParams>,
        /// Per-token logit biases (`logit_bias`). Always serialized (possibly `[]`)
        /// — honest required-array binding.
        pub logit_bias: Vec<LogitBias>,
        /// GBNF grammar constraint (`grammar`).
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        pub grammar: Option<GrammarParams>,
        /// Per-request sampler RNG seed (`dist`); overrides the load-pinned seed.
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(type = "number")]
        #[ts(optional)]
        pub seed: Option<u32>,
    }
}

impl LlamaCppParams {
    /// Build the three base fields (the quick-load shape), leaving every optional
    /// at its `None`/empty default — the pre-expansion behavior.
    pub fn base(ctx_len: CtxLen, gpu_layers: GpuLayers, threads: u32) -> Self {
        Self {
            ctx_len,
            gpu_layers,
            threads,
            ..Default::default()
        }
    }

    /// True when any ENGINE OVERRIDE beyond the three base fields is set — i.e. the
    /// rich `params` payload carries something the worker must apply. The base
    /// fields (`ctx_len`/`gpu_layers`/`threads`) ride their own dedicated wire
    /// fields, so a base-only load needs no payload (smaller wire; forward-compat).
    pub fn has_overrides(&self) -> bool {
        // EXHAUSTIVE destructure: a new field added to the struct breaks this
        // at compile time, so it can never be silently forgotten here — a
        // forgotten field would make a load whose only override is that field
        // read as "no overrides" and be DROPPED by the hub's emptiness filter
        // (and skipped by the local attach gate): silent loss, not refusal.
        let Self {
            ctx_len: _,
            gpu_layers: _,
            threads: _,
            use_mmap,
            use_mlock,
            cpu_moe,
            cpu_buft_overrides,
            split_mode,
            main_gpu,
            devices,
            n_batch,
            n_ubatch,
            n_seq_max,
            n_threads_batch,
            flash_attn,
            offload_kqv,
            swa_full,
            type_k,
            type_v,
            rope_scaling_type,
            rope_freq_base,
            rope_freq_scale,
            kv_overrides,
            seed,
        } = self;
        use_mmap.is_some()
            || use_mlock.is_some()
            || cpu_moe.is_some()
            || split_mode.is_some()
            || main_gpu.is_some()
            || devices.is_some()
            || n_batch.is_some()
            || n_ubatch.is_some()
            || n_seq_max.is_some()
            || n_threads_batch.is_some()
            || flash_attn.is_some()
            || offload_kqv.is_some()
            || swa_full.is_some()
            || type_k.is_some()
            || type_v.is_some()
            || rope_scaling_type.is_some()
            || rope_freq_base.is_some()
            || rope_freq_scale.is_some()
            || seed.is_some()
            || !cpu_buft_overrides.is_empty()
            || !kv_overrides.is_empty()
    }
}

impl LlamaCppSamplingParams {
    /// Overlay a per-request sampler set (`over`) onto this base (a model's
    /// tuned/card-recommended defaults): every field `over` sets wins; the rest
    /// fall back to `self`. This is the chat-time merge — an OpenAI request that
    /// pins only `temperature`/`top_p`/penalties overrides those while the base's
    /// other samplers (`top_k`/`min_p`/`typical_p`/…) still apply, so a tuned
    /// model serves with its recommendation by default yet stays per-request
    /// overridable. The always-present vecs (`logit_bias`) follow last-writer-wins:
    /// a non-empty `over` vec replaces the base's, an empty one keeps the base's.
    pub fn overlaid_with(&self, over: &LlamaCppSamplingParams) -> LlamaCppSamplingParams {
        LlamaCppSamplingParams {
            temperature: over.temperature.or(self.temperature),
            dynatemp_range: over.dynatemp_range.or(self.dynatemp_range),
            dynatemp_exponent: over.dynatemp_exponent.or(self.dynatemp_exponent),
            top_k: over.top_k.or(self.top_k),
            top_p: over.top_p.or(self.top_p),
            min_p: over.min_p.or(self.min_p),
            typical_p: over.typical_p.or(self.typical_p),
            top_n_sigma: over.top_n_sigma.or(self.top_n_sigma),
            xtc_probability: over.xtc_probability.or(self.xtc_probability),
            xtc_threshold: over.xtc_threshold.or(self.xtc_threshold),
            penalty_last_n: over.penalty_last_n.or(self.penalty_last_n),
            penalty_repeat: over.penalty_repeat.or(self.penalty_repeat),
            penalty_freq: over.penalty_freq.or(self.penalty_freq),
            penalty_present: over.penalty_present.or(self.penalty_present),
            dry: over.dry.clone().or_else(|| self.dry.clone()),
            mirostat: over.mirostat.clone().or_else(|| self.mirostat.clone()),
            logit_bias: if over.logit_bias.is_empty() {
                self.logit_bias.clone()
            } else {
                over.logit_bias.clone()
            },
            grammar: over.grammar.clone().or_else(|| self.grammar.clone()),
            seed: over.seed.or(self.seed),
        }
    }

    /// The name of the first ADVANCED sampler this set requests that the llama.cpp
    /// engine does not yet apply (it needs the model/vocab handle to build): `grammar`,
    /// `logit_bias`, `dry`, or `mirostat`. `None` when only supported samplers are set.
    /// The worker uses this to FAIL LOUD ([`HG013`](crate::diagnostic::HiggsError::InvalidSamplingParam))
    /// rather than silently return unconstrained/unbiased output — see `run_decode`.
    pub fn unsupported_sampler(&self) -> Option<&'static str> {
        if self.grammar.is_some() {
            Some("grammar")
        } else if !self.logit_bias.is_empty() {
            Some("logit_bias")
        } else if self.dry.is_some() {
            Some("dry")
        } else if self.mirostat.is_some() {
            Some("mirostat")
        } else {
            None
        }
    }
}

#[cfg(test)]
#[path = "params_tests.rs"]
mod tests;
