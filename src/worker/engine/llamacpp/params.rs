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

use crate::worker::engine::{FlashAttn, KvCacheKind};

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
    /// quick-load / `default_load` / suggester path fills them; `gpu_layers ==
    /// u32::MAX` means "all on GPU" (LM Studio "max" semantics). Every other field
    /// is optional: absent (`None`/empty) means "use the engine default", which
    /// reproduces the pre-expansion behavior exactly.
    #[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
    #[serde(default)]
    pub struct LlamaCppParams {
        // ── model params (LlamaModelParams) ──────────────────────────────────
        /// Context window size in tokens (`with_n_ctx`).
        #[ts(type = "number")]
        pub ctx_len: u32,
        /// Layers offloaded to GPU; `u32::MAX` = all (`with_n_gpu_layers`).
        #[ts(type = "number")]
        pub gpu_layers: u32,
        /// Worker threads used during generation (`with_n_threads`; also seeds
        /// `n_threads_batch` when that is unset).
        #[ts(type = "number")]
        pub threads: u32,
        /// Memory-map the GGUF instead of reading it into RAM (`with_use_mmap`).
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        pub use_mmap: Option<bool>,
        /// Lock model pages in RAM, preventing swap (`with_use_mlock`).
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        pub use_mlock: Option<bool>,
        /// Keep MoE expert tensors on the CPU (`add_cpu_moe_override`, pinned
        /// apply). Boolean only — no numeric `n_cpu_moe` binding exists. The
        /// suggester flips this on under VRAM pressure when RAM still fits.
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        pub cpu_moe: Option<bool>,
        /// Per-tensor-buffer-type → CPU regex overrides (`add_cpu_buft_override`,
        /// pinned apply; advanced/manual). Drives finer MoE offload later. Always
        /// serialized (possibly `[]`) so the bindings' required-array shape is honest.
        pub cpu_buft_overrides: Vec<String>,
        /// Multi-GPU split mode (`with_split_mode`). DEFERRED-derive; no-op default.
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        pub split_mode: Option<SplitMode>,
        /// Primary GPU index when `split_mode = None` (`with_main_gpu`). DEFERRED.
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(type = "number")]
        #[ts(optional)]
        pub main_gpu: Option<i32>,
        /// Explicit device selection (`with_devices`). DEFERRED (multi-GPU). Wire
        /// type is `u32` (ts-portable); the apply path converts to `usize`.
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        pub devices: Option<Vec<u32>>,
        // ── context params (LlamaContextParams) ──────────────────────────────
        /// Logical batch size for prompt decode (`with_n_batch`). `None` keeps the
        /// current default (`ctx_len.max(1)` — one-shot prefill).
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(type = "number")]
        #[ts(optional)]
        pub n_batch: Option<u32>,
        /// Physical (micro) batch size (`with_n_ubatch`).
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(type = "number")]
        #[ts(optional)]
        pub n_ubatch: Option<u32>,
        /// Parallel sequence slots (`with_n_seq_max`); default 1.
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(type = "number")]
        #[ts(optional)]
        pub n_seq_max: Option<u32>,
        /// Threads for batch/prompt processing (`with_n_threads_batch`). `None`
        /// reuses `threads` (today's shared behavior).
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(type = "number")]
        #[ts(optional)]
        pub n_threads_batch: Option<u32>,
        /// Flash-attention policy (`with_flash_attention_policy`).
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        pub flash_attn: Option<FlashAttn>,
        /// Offload the KV cache & KQV ops to the GPU (`with_offload_kqv`).
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        pub offload_kqv: Option<bool>,
        /// Full sliding-window attention (`with_swa_full`).
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        pub swa_full: Option<bool>,
        /// KV cache key data type (`with_type_k`).
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        pub type_k: Option<KvCacheKind>,
        /// KV cache value data type (`with_type_v`).
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        pub type_v: Option<KvCacheKind>,
        /// RoPE scaling type (`with_rope_scaling_type`).
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        pub rope_scaling_type: Option<RopeScalingType>,
        /// RoPE base frequency override (`with_rope_freq_base`).
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(type = "number")]
        #[ts(optional)]
        pub rope_freq_base: Option<f32>,
        /// RoPE frequency scale / context extension (`with_rope_freq_scale`).
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(type = "number")]
        #[ts(optional)]
        pub rope_freq_scale: Option<f32>,
        /// GGUF metadata overrides (`append_kv_override`, pinned apply; advanced).
        /// Always serialized (possibly `[]`) — honest required-array binding.
        pub kv_overrides: Vec<KvOverride>,
        /// Load-pinned sampler RNG seed. `None` = a fresh random seed per request.
        /// A per-request `LlamaCppSamplingParams::seed` overrides this when set.
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(type = "number")]
        #[ts(optional)]
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
    pub fn base(ctx_len: u32, gpu_layers: u32, threads: u32) -> Self {
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
        self.use_mmap.is_some()
            || self.use_mlock.is_some()
            || self.cpu_moe.is_some()
            || self.split_mode.is_some()
            || self.main_gpu.is_some()
            || self.devices.is_some()
            || self.n_batch.is_some()
            || self.n_ubatch.is_some()
            || self.n_seq_max.is_some()
            || self.n_threads_batch.is_some()
            || self.flash_attn.is_some()
            || self.offload_kqv.is_some()
            || self.swa_full.is_some()
            || self.type_k.is_some()
            || self.type_v.is_some()
            || self.rope_scaling_type.is_some()
            || self.rope_freq_base.is_some()
            || self.rope_freq_scale.is_some()
            || self.seed.is_some()
            || !self.cpu_buft_overrides.is_empty()
            || !self.kv_overrides.is_empty()
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
mod tests {
    use super::*;

    /// A lean JSON object carrying only a couple of base fields deserializes into
    /// the full struct with every other field at its `None`/empty default — the
    /// back-compat / quick-load shape.
    #[test]
    fn lean_json_deserializes_into_full_params() {
        let p: LlamaCppParams =
            serde_json::from_value(serde_json::json!({"ctx_len": 4096, "gpu_layers": 32})).unwrap();
        assert_eq!(p.ctx_len, 4096);
        assert_eq!(p.gpu_layers, 32);
        assert!(p.type_k.is_none() && p.cpu_moe.is_none() && p.n_seq_max.is_none());
        assert!(p.cpu_buft_overrides.is_empty() && p.kv_overrides.is_empty());
    }

    /// `Option` overrides are omitted when `None`; the always-serialized vec
    /// fields appear as `[]` (so their required-array bindings stay honest). A
    /// quick-load thus carries only the base fields + the empty advanced vecs.
    #[test]
    fn absent_optionals_do_not_serialize() {
        let bare = LlamaCppParams::base(4096, u32::MAX, 4);
        let v = serde_json::to_value(&bare).unwrap();
        let obj = v.as_object().unwrap();
        assert!(
            obj.contains_key("ctx_len")
                && obj.contains_key("gpu_layers")
                && obj.contains_key("threads"),
            "base fields present: {obj:?}"
        );
        // Vec fields always serialize (possibly empty) — matches the required-array binding.
        assert_eq!(obj["cpu_buft_overrides"], serde_json::json!([]));
        assert_eq!(obj["kv_overrides"], serde_json::json!([]));
        // Every Option override is omitted when None.
        assert!(
            !obj.contains_key("type_k")
                && !obj.contains_key("flash_attn")
                && !obj.contains_key("cpu_moe"),
            "None options omitted: {obj:?}"
        );
    }

    /// A fully-populated round-trip preserves the new fields (cpu_moe, n_seq_max,
    /// rope_scaling_type, kv_overrides) — the expanded coverage survives the wire.
    #[test]
    fn full_params_round_trip() {
        let full = LlamaCppParams {
            ctx_len: 8192,
            gpu_layers: u32::MAX,
            threads: 8,
            cpu_moe: Some(true),
            n_seq_max: Some(4),
            n_threads_batch: Some(6),
            swa_full: Some(false),
            type_k: Some(KvCacheKind::Q8_0),
            rope_scaling_type: Some(RopeScalingType::Yarn),
            kv_overrides: vec![KvOverride {
                key: "llama.context_length".into(),
                value: "8192".into(),
            }],
            ..Default::default()
        };
        let json = serde_json::to_string(&full).unwrap();
        let back: LlamaCppParams = serde_json::from_str(&json).unwrap();
        assert_eq!(back, full);
    }

    /// Overlaying a request set onto a card base: the request's set fields win,
    /// the base's other samplers survive. This is the chat-time merge that lets a
    /// tuned/card recommendation (top_k/min_p/…) apply while a per-request
    /// temperature/top_p/penalty still overrides.
    #[test]
    fn overlaid_with_request_overrides_base_keeps_rest() {
        let base = LlamaCppSamplingParams {
            temperature: Some(0.6),
            top_k: Some(40),
            min_p: Some(0.05),
            penalty_repeat: Some(1.1),
            logit_bias: vec![LogitBias {
                token: 7,
                bias: 1.0,
            }],
            ..Default::default()
        };
        // A request that sets only temperature + top_p (the common OpenAI subset).
        let req = LlamaCppSamplingParams {
            temperature: Some(1.2),
            top_p: Some(0.9),
            ..Default::default()
        };
        let merged = base.overlaid_with(&req);
        assert_eq!(merged.temperature, Some(1.2), "request temp wins");
        assert_eq!(merged.top_p, Some(0.9), "request top_p applied");
        assert_eq!(merged.top_k, Some(40), "base top_k survives");
        assert_eq!(merged.min_p, Some(0.05), "base min_p survives");
        assert_eq!(merged.penalty_repeat, Some(1.1), "base penalty survives");
        assert_eq!(merged.logit_bias.len(), 1, "base logit_bias survives");
        // An all-empty request leaves the base untouched.
        let untouched = base.overlaid_with(&LlamaCppSamplingParams::default());
        assert_eq!(untouched, base);
        // A request logit_bias replaces the base's (last-writer-wins on the vec).
        let with_bias = base.overlaid_with(&LlamaCppSamplingParams {
            logit_bias: vec![LogitBias {
                token: 9,
                bias: -2.0,
            }],
            ..Default::default()
        });
        assert_eq!(with_bias.logit_bias.len(), 1);
        assert_eq!(with_bias.logit_bias[0].token, 9, "request bias wins");
    }

    /// `unsupported_sampler` reports `None` for a set of only-supported samplers and
    /// the precise field name for each not-yet-applied advanced sampler — the worker's
    /// fail-loud guard (HG013) against silently dropping a grammar/logit_bias/dry/mirostat.
    #[test]
    fn unsupported_sampler_flags_only_unapplied_advanced_samplers() {
        // Supported-only set (temperature/top_p/penalties) ⇒ nothing flagged.
        let ok = LlamaCppSamplingParams {
            temperature: Some(0.7),
            top_p: Some(0.9),
            penalty_repeat: Some(1.1),
            ..Default::default()
        };
        assert_eq!(ok.unsupported_sampler(), None, "supported samplers pass");
        // Each advanced sampler is flagged by name.
        assert_eq!(
            LlamaCppSamplingParams {
                grammar: Some(GrammarParams {
                    gbnf: "root ::= \"x\"".to_string(),
                    root: "root".to_string(),
                }),
                ..Default::default()
            }
            .unsupported_sampler(),
            Some("grammar"),
        );
        assert_eq!(
            LlamaCppSamplingParams {
                logit_bias: vec![LogitBias {
                    token: 1,
                    bias: 2.0,
                }],
                ..Default::default()
            }
            .unsupported_sampler(),
            Some("logit_bias"),
        );
        assert_eq!(
            LlamaCppSamplingParams {
                mirostat: Some(MirostatParams {
                    version: 2,
                    tau: 5.0,
                    eta: 0.1,
                }),
                ..Default::default()
            }
            .unsupported_sampler(),
            Some("mirostat"),
        );
    }

    /// `has_overrides` is false for a base-only load and true once any engine
    /// override (an optional field or a non-empty advanced vec) is set.
    #[test]
    fn has_overrides_detects_engine_overrides() {
        assert!(
            !LlamaCppParams::base(4096, u32::MAX, 8).has_overrides(),
            "base-only load carries no overrides"
        );
        let mut p = LlamaCppParams::base(4096, u32::MAX, 8);
        p.flash_attn = Some(FlashAttn::On);
        assert!(p.has_overrides(), "an optional override counts");
        let mut p2 = LlamaCppParams::base(4096, u32::MAX, 8);
        p2.kv_overrides.push(KvOverride {
            key: "k".into(),
            value: "v".into(),
        });
        assert!(p2.has_overrides(), "a non-empty advanced vec counts");
    }

    /// The sampling type covers the full sampler surface; a lean card-derived
    /// subset (temp/top_k/top_p/min_p/penalties) round-trips with the rest `None`.
    #[test]
    fn sampling_params_lean_round_trip() {
        let s = LlamaCppSamplingParams {
            temperature: Some(0.6),
            top_k: Some(20),
            top_p: Some(0.95),
            min_p: Some(0.0),
            penalty_repeat: Some(1.1),
            ..Default::default()
        };
        let json = serde_json::to_string(&s).unwrap();
        let back: LlamaCppSamplingParams = serde_json::from_str(&json).unwrap();
        assert_eq!(back, s);
        assert!(back.mirostat.is_none() && back.grammar.is_none() && back.dry.is_none());
    }
}
