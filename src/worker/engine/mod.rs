//! Engine abstraction. v1: llama.cpp. Future: MLX (mlxcel cxx pattern), other runtimes.
//!
//! # Adding a new engine (three steps, no other file changes)
//!
//! The worker talks to inference backends ONLY through the [`HiggsEngine`] trait, and picks
//! one at startup from the [`REGISTRY`]. To add an engine — say `mlx`:
//!
//! 1. **Create a submodule** `engine/mlx/mod.rs` with a type implementing [`HiggsEngine`]
//!    (`load`/`unload`/`is_loaded`/`chat`/`probe`/`devices`). Keep every backend-specific
//!    dependency (FFI, crates) confined to that submodule — the trait is the only thing the
//!    rest of higgs sees. Declare it here: `pub mod mlx;`.
//! 2. **Register it** by adding ONE line to [`REGISTRY`]:
//!    `EngineEntry { name: "mlx", build: || Box::new(mlx::MlxEngine::default()) }`.
//! 3. **Select it** at runtime with `HIGGS_ENGINE=mlx` (the worker reads this; the first
//!    registry entry is the default). Per-model selection can layer on top later.
//!
//! That's the whole contract: implement the trait, add a registry line. No edits to the
//! worker dispatch, the supervisor, the node runtime, or the serve layer — they are all
//! engine-agnostic above [`HiggsEngine`].

pub mod llamacpp;

use crate::diagnostic::HiggsError;

/// Generation parameters for one request.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GenParams {
    pub max_tokens: usize,
    /// The full sampler set for this request (engine umbrella; the `LlamaCpp`
    /// variant carries temperature/top_k/top_p/min_p/penalties/…). The decode loop
    /// builds the ordered `LlamaSampler` chain from it. `temperature <= 0` ⇒ greedy.
    pub sampling: SamplingParams,
    /// OpenAI-compatible `tools` array serialized to a JSON string, or `None`
    /// when the request carries no tools. Fed verbatim to the GGUF chat
    /// template alongside the messages; the crate's vendored `common_chat`
    /// renders the tool grammar and selects the matching tool-call parser. We
    /// invent no parser of our own.
    pub tools_json: Option<String>,
}

higgs_ts! {
    /// Engine-tagged **load-parameter umbrella**. `LoadParams::LlamaCpp(LlamaCppParams)`
    /// today; a future `Mlx(MlxParams)` slots beside it. "All are `LoadParams`."
    ///
    /// Serialized **internally tagged** on the `engine` field, so the JSON is the
    /// flattened llama.cpp payload plus `"engine":"LlamaCpp"`. The worker dispatches
    /// on the variant; the suggester (`src/tune/`) derives the concrete payload.
    /// The concrete fields live in [`llamacpp::params::LlamaCppParams`] (full
    /// `llama-cpp-2` 0.1.139 coverage) — applied only inside `llamacpp/mod.rs`.
    #[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
    #[serde(tag = "engine")]
    pub enum LoadParams {
        /// The llama.cpp engine's load parameters.
        LlamaCpp(llamacpp::params::LlamaCppParams),
    }
}

impl Default for LoadParams {
    fn default() -> Self {
        LoadParams::LlamaCpp(llamacpp::params::LlamaCppParams::default())
    }
}

impl LoadParams {
    /// Wrap llama.cpp params in the umbrella.
    pub fn llamacpp(p: llamacpp::params::LlamaCppParams) -> Self {
        LoadParams::LlamaCpp(p)
    }

    /// Build the umbrella from just the three base llama.cpp fields (the
    /// quick-load shape; every optional left at its engine default).
    pub fn base(ctx_len: u32, gpu_layers: GpuLayers, threads: u32) -> Self {
        LoadParams::LlamaCpp(llamacpp::params::LlamaCppParams::base(
            ctx_len, gpu_layers, threads,
        ))
    }

    /// The llama.cpp payload (the only engine variant today).
    pub fn as_llamacpp(&self) -> &llamacpp::params::LlamaCppParams {
        match self {
            LoadParams::LlamaCpp(p) => p,
        }
    }

    /// Base-field accessor: context window in tokens.
    pub fn ctx_len(&self) -> u32 {
        self.as_llamacpp().ctx_len
    }

    /// Base-field accessor: GPU layers ([`GpuLayers::All`] = every layer).
    pub fn gpu_layers(&self) -> GpuLayers {
        self.as_llamacpp().gpu_layers
    }

    /// Base-field accessor: generation threads.
    pub fn threads(&self) -> u32 {
        self.as_llamacpp().threads
    }
}

higgs_ts! {
    /// Engine-tagged **sampling-parameter umbrella**, mirroring [`LoadParams`].
    /// `SamplingParams::LlamaCpp(LlamaCppSamplingParams)` today. Carried by
    /// [`GenParams`] and persisted alongside a tuning profile.
    #[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
    #[serde(tag = "engine")]
    pub enum SamplingParams {
        /// The llama.cpp engine's sampler parameters.
        LlamaCpp(llamacpp::params::LlamaCppSamplingParams),
    }
}

impl Default for SamplingParams {
    fn default() -> Self {
        SamplingParams::LlamaCpp(llamacpp::params::LlamaCppSamplingParams::default())
    }
}

impl SamplingParams {
    /// Wrap llama.cpp sampling params in the umbrella.
    pub fn llamacpp(p: llamacpp::params::LlamaCppSamplingParams) -> Self {
        SamplingParams::LlamaCpp(p)
    }

    /// The llama.cpp payload (the only engine variant today).
    pub fn as_llamacpp(&self) -> &llamacpp::params::LlamaCppSamplingParams {
        match self {
            SamplingParams::LlamaCpp(p) => p,
        }
    }
}

higgs_const_enum! {
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

higgs_const_enum! {
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

higgs_ts! {
    /// How many model layers to offload to the GPU. Replaces the old `gpu_layers: u32`
    /// where `u32::MAX` was a sentinel for "all" — the intent now lives in the type, not a
    /// magic number. A data enum (carries `n`), so it is a ts-rs tagged union (the documented
    /// exception to the const-enum rule), internally tagged on `kind`:
    /// `{"kind":"all"}` / `{"kind":"count","n":32}`. Mapped to the raw `with_n_gpu_layers`
    /// int ONLY inside `llamacpp/mod.rs`.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
    #[serde(tag = "kind", rename_all = "snake_case")]
    pub enum GpuLayers {
        /// Offload every layer to the GPU (LM Studio "max").
        All,
        /// Offload exactly `n` layers; `n == 0` is CPU-only.
        Count {
            /// Number of layers offloaded.
            #[ts(type = "number")]
            n: u32,
        },
    }
}

impl Default for GpuLayers {
    /// Preserves the pre-enum derived default (`gpu_layers: 0`, i.e. CPU-only).
    fn default() -> Self {
        GpuLayers::Count { n: 0 }
    }
}

impl GpuLayers {
    /// Every layer offloaded (the old `gpu_layers == u32::MAX`).
    pub fn all() -> Self {
        GpuLayers::All
    }

    /// Offload exactly `n` layers (`n == 0` → CPU-only).
    pub fn count(n: u32) -> Self {
        GpuLayers::Count { n }
    }

    /// `true` when every layer is offloaded.
    pub fn is_all(&self) -> bool {
        matches!(self, GpuLayers::All)
    }

    /// `true` when no layers are offloaded (CPU-only) — the old `gpu_layers == 0`.
    pub fn is_cpu_only(&self) -> bool {
        matches!(self, GpuLayers::Count { n: 0 })
    }

    /// The raw `with_n_gpu_layers` value for the llama.cpp FFI: `u32::MAX` for
    /// [`GpuLayers::All`] (llama.cpp treats an over-large count as "all" — exactly the
    /// old sentinel), else the explicit count. Called ONLY at the FFI boundary in
    /// `llamacpp/mod.rs`.
    pub fn to_n_gpu_layers(&self) -> u32 {
        match self {
            GpuLayers::All => u32::MAX,
            GpuLayers::Count { n } => *n,
        }
    }
}

impl<'de> serde::Deserialize<'de> for GpuLayers {
    /// Lenient: accept EITHER a bare integer (the legacy/persisted form, `u32::MAX` = all)
    /// or the canonical tagged object (`{"kind":"all"}` / `{"kind":"count","n":N}`).
    /// [`Serialize`] always emits the tagged object, so a config.json written before this
    /// change still reads, and migrates forward to the tagged form on its next save.
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        #[serde(untagged)]
        enum Wire {
            Num(u64),
            Tagged(Tagged),
        }
        #[derive(serde::Deserialize)]
        #[serde(tag = "kind", rename_all = "snake_case")]
        enum Tagged {
            All,
            Count { n: u32 },
        }
        Ok(match Wire::deserialize(deserializer)? {
            // `u32::MAX` is the legacy "all" sentinel; any other int is an explicit count.
            Wire::Num(n) if n >= u32::MAX as u64 => GpuLayers::All,
            Wire::Num(n) => GpuLayers::Count { n: n as u32 },
            Wire::Tagged(Tagged::All) => GpuLayers::All,
            Wire::Tagged(Tagged::Count { n }) => GpuLayers::Count { n },
        })
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

    /// Enumerate the host's compute devices (CPU/GPU/accel) as this engine sees
    /// them. Cheap and read-only — no model load, no resident-state mutation —
    /// so it is safe to call at any time, including on a fresh worker. Returns an
    /// empty vec when the engine exposes no devices.
    fn devices(&self) -> Vec<crate::system::GpuDevice>;
}

/// One compiled-in engine: a stable selector name + a zero-arg constructor. The build
/// closure is a plain `fn` pointer so the registry is a `const` table (no allocation, usable
/// in tests and tooling).
pub struct EngineEntry {
    /// The `HIGGS_ENGINE` value that selects this engine (lowercase, stable).
    pub name: &'static str,
    /// Construct a fresh, model-less instance of this engine.
    pub build: fn() -> Box<dyn HiggsEngine>,
}

/// Every engine compiled into this build. The FIRST entry is the default. Adding an engine
/// is one line here (see the module docs) — nothing else in higgs needs to change.
pub const REGISTRY: &[EngineEntry] = &[EngineEntry {
    name: "llamacpp",
    build: || Box::new(llamacpp::LlamaCppEngine::default()),
}];

/// The default engine's name (the first [`REGISTRY`] entry). Const-true that the registry is
/// never empty would be nice, but `REGISTRY[0]` already enforces it at build time.
pub fn default_engine_name() -> &'static str {
    REGISTRY[0].name
}

/// The set of selectable engine names, for diagnostics / `--help`.
pub fn engine_names() -> Vec<&'static str> {
    REGISTRY.iter().map(|e| e.name).collect()
}

/// Build the engine selected by `name` (case-insensitive). `None` or an unknown name falls
/// back to the default (first registry entry); the chosen name is returned so the caller can
/// log it and warn on an unknown request.
pub fn build_engine(name: Option<&str>) -> (Box<dyn HiggsEngine>, &'static str) {
    let requested = name.map(str::trim).filter(|s| !s.is_empty());
    if let Some(req) = requested {
        if let Some(entry) = REGISTRY.iter().find(|e| e.name.eq_ignore_ascii_case(req)) {
            return ((entry.build)(), entry.name);
        }
        tracing::warn!(
            requested = req,
            default = default_engine_name(),
            available = ?engine_names(),
            "higgs: unknown HIGGS_ENGINE; falling back to default"
        );
    }
    let entry = &REGISTRY[0];
    ((entry.build)(), entry.name)
}

#[cfg(test)]
mod registry_tests {
    use super::*;

    #[test]
    fn default_is_llamacpp_and_registry_nonempty() {
        assert!(
            !engine_names().is_empty(),
            "at least one engine is registered"
        );
        assert_eq!(default_engine_name(), "llamacpp");
        assert!(engine_names().contains(&"llamacpp"));
    }

    #[test]
    fn build_selects_by_name_case_insensitively() {
        let (_e, name) = build_engine(Some("LlamaCpp"));
        assert_eq!(name, "llamacpp", "case-insensitive match");
    }

    #[test]
    fn build_falls_back_to_default_for_unknown_or_absent() {
        assert_eq!(build_engine(None).1, "llamacpp");
        assert_eq!(build_engine(Some("")).1, "llamacpp");
        assert_eq!(
            build_engine(Some("nope")).1,
            "llamacpp",
            "unknown → default"
        );
    }
}
