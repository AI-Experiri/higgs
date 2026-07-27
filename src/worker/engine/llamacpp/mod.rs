//! llama.cpp engine via the `llama-cpp-2` crate (Metal by default on macOS).
//!
//! The ONLY file allowed to import `llama_cpp_2`. The decode loop copies the
//! crate's own `examples/simple`: prompt batch feed, then per token
//! sample → EOG check → detokenize → sink → re-batch → decode.

use std::num::NonZeroU32;
use std::sync::OnceLock;

use llama_cpp_2::context::params::{
    KvCacheType, LlamaContextParams, RopeScalingType as LlamaRopeScalingType,
};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::kv_overrides::ParamOverrideValue;
use llama_cpp_2::model::params::{LlamaModelParams, LlamaSplitMode};
use llama_cpp_2::model::{AddBos, ChatTemplateResult, LlamaChatTemplate, LlamaModel};
use llama_cpp_2::openai::{ChatParseStateOaicompat, OpenAIChatTemplateParams};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;

use self::params::{LlamaCppParams, RopeScalingType, SplitMode};
use super::{CtxLen, EngineDelta, FlashAttn, GenParams, HiggsEngine, KvCacheKind, LoadParams};
use crate::diagnostic::HiggsError;
use crate::system::{DeviceKind, GpuDevice};

/// This engine's log control: the worker-side `tracing` subscriber install, the
/// llama.cpp/ggml level + module filters, and the live verbose toggle. All
/// llama.cpp log filtering lives here — a different engine ships its own.
pub mod logging;

/// Engine-specific load + sampling parameter types (`LlamaCppParams`,
/// `LlamaCppSamplingParams`, and the llama.cpp enums) — the payloads of the
/// `engine::LoadParams` / `engine::SamplingParams` umbrellas.
pub mod params;

/// Process-wide llama.cpp backend handle — the FFI global init must run
/// exactly once per process.
static BACKEND: OnceLock<LlamaBackend> = OnceLock::new();

/// Initialize-once accessor for the llama.cpp backend.
fn backend() -> &'static LlamaBackend {
    // init()'s only error is BackendAlreadyInitialized, unreachable under OnceLock.
    BACKEND.get_or_init(|| LlamaBackend::init().expect("sole llama backend init"))
}

/// The vendored engine's version string — `ggml_version()` (e.g. `"0.9.7"`).
///
/// This is the real, runtime-reported version of the ggml/llama.cpp engine
/// linked into this build, distinct from the `llama-cpp-2` Rust binding version.
/// The upstream llama.cpp `bNNNN` build number is not obtainable from the crate
/// (the vendored tree ships without git, so its build-info constants are unset
/// and unbound). `ggml_version()` returns a static C string and needs no model
/// or context, so it is safe to call at any time.
pub fn engine_version() -> String {
    // SAFETY: `ggml_version()` returns a pointer to a static, NUL-terminated C
    // string baked into the engine; it takes no arguments and never fails.
    unsafe {
        std::ffi::CStr::from_ptr(llama_cpp_sys_2::ggml_version())
            .to_string_lossy()
            .into_owned()
    }
}

/// Map ggml's raw backend-device type to the engine-agnostic [`DeviceKind`].
///
/// The raw values are `GGML_BACKEND_DEVICE_TYPE_CPU` (0), `_GPU` (1), `_ACCEL`
/// (2). Any unknown future value is treated as an accelerator. Confined to this
/// file per the engine boundary.
fn device_type_to_kind(raw: llama_cpp_sys_2::ggml_backend_dev_type) -> DeviceKind {
    match raw {
        llama_cpp_sys_2::GGML_BACKEND_DEVICE_TYPE_CPU => DeviceKind::Cpu,
        llama_cpp_sys_2::GGML_BACKEND_DEVICE_TYPE_GPU => DeviceKind::Gpu,
        _ => DeviceKind::Accel,
    }
}

/// Read one ggml backend device's name, description, kind, and memory.
///
/// SAFETY: `dev` is a valid `ggml_backend_dev_t` obtained from
/// `ggml_backend_dev_get(i)` for `i < ggml_backend_dev_count()`. `name`/
/// `description` return static, NUL-terminated C strings owned by the engine
/// (never freed by us); `ggml_backend_dev_memory` writes two `usize` out-params
/// and never fails. All calls are read-only and allocate nothing engine-side.
unsafe fn read_device(dev: llama_cpp_sys_2::ggml_backend_dev_t) -> GpuDevice {
    let name = std::ffi::CStr::from_ptr(llama_cpp_sys_2::ggml_backend_dev_name(dev))
        .to_string_lossy()
        .into_owned();
    let description = std::ffi::CStr::from_ptr(llama_cpp_sys_2::ggml_backend_dev_description(dev))
        .to_string_lossy()
        .into_owned();
    let kind = device_type_to_kind(llama_cpp_sys_2::ggml_backend_dev_type(dev));
    let mut free: usize = 0;
    let mut total: usize = 0;
    llama_cpp_sys_2::ggml_backend_dev_memory(dev, &mut free, &mut total);
    GpuDevice {
        name,
        description,
        kind,
        vram_total_bytes: total as u64,
        vram_free_bytes: free as u64,
    }
}

/// Enumerate every compute device ggml's loaded backends expose.
///
/// Engine-native: queries ggml's own backend-device registry, so the result
/// reflects exactly what llama.cpp will run on (Metal on macOS, CPU otherwise).
/// The llama.cpp backend is initialized first via [`backend`] — device
/// registration happens at backend init, so a cold call (no model loaded) still
/// sees every device. A device that fails to report is skipped, never panics.
pub fn device_info() -> Vec<GpuDevice> {
    // Ensure ggml's backends are registered before enumerating.
    let _ = backend();
    // SAFETY: `ggml_backend_dev_count` takes no arguments and returns the number
    // of registered devices; each index `< count` is valid for
    // `ggml_backend_dev_get`, which returns a non-null device handle.
    unsafe {
        let count = llama_cpp_sys_2::ggml_backend_dev_count();
        let mut out = Vec::with_capacity(count);
        for i in 0..count {
            let dev = llama_cpp_sys_2::ggml_backend_dev_get(i);
            if dev.is_null() {
                continue;
            }
            out.push(read_device(dev));
        }
        out
    }
}

/// A resident model plus the load-time state `chat()` needs to serve it.
struct LoadedModel {
    model: LlamaModel,
    /// Load-time knobs; `ctx_len`/`threads` shape the per-request context. The
    /// concrete llama.cpp payload (the worker selected this engine, so it only
    /// ever holds the `LlamaCpp` variant's params).
    params: LlamaCppParams,
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
        // Reset the engine-diagnostic capture so a previous load's llama.cpp ERROR
        // lines can't leak into THIS load's failure reason.
        logging::clear_engine_diagnostics();
        // The worker selected the llamacpp engine, so the umbrella always carries
        // the LlamaCpp variant here; destructure it for the concrete payload. (When
        // a second engine variant lands this `let` stops being irrefutable, forcing
        // the dispatch to be made explicit — the desired compile-time reminder.)
        let LoadParams::LlamaCpp(p) = params;
        // Build an engine-load failure. llama.cpp emits the REAL cause (e.g.
        // `unknown model architecture: 'gemma4'`) as a separate ERROR log event,
        // while the FFI `Result` carries only an opaque "null result from llama
        // cpp". Prefer the captured engine ERROR lines (drained here) over that
        // opaque string; fall back to the binding's message when the engine logged
        // nothing (e.g. an OOM kill). Only one failure site fires per load, so this
        // single drain is unambiguous.
        let load_err = |e: &dyn std::fmt::Display| {
            let diagnostics = logging::take_engine_diagnostics();
            let reason = if diagnostics.is_empty() {
                e.to_string()
            } else {
                diagnostics.join("; ")
            };
            HiggsError::EngineLoadFailed {
                id: path.to_string(),
                reason,
            }
        };

        // Safe move-based builder for the params that have plain `with_*` setters.
        let mut model_params =
            LlamaModelParams::default().with_n_gpu_layers(p.gpu_layers.to_n_gpu_layers());
        if let Some(b) = p.use_mmap {
            model_params = model_params.with_use_mmap(b);
        }
        if let Some(b) = p.use_mlock {
            model_params = model_params.with_use_mlock(b);
        }
        // Multi-GPU knobs are carried for completeness (single-GPU/Metal target);
        // applied when explicitly set, never derived.
        if let Some(sm) = p.split_mode {
            model_params = model_params.with_split_mode(split_mode_to_llama(sm));
        }
        if let Some(g) = p.main_gpu {
            model_params = model_params.with_main_gpu(g);
        }
        if let Some(devs) = &p.devices {
            let usize_devs: Vec<usize> = devs.iter().map(|d| *d as usize).collect();
            model_params = model_params
                .with_devices(&usize_devs)
                .map_err(|e| load_err(&e))?;
        }
        // `add_cpu_moe_override` / `add_cpu_buft_override` / `append_kv_override` take
        // `Pin<&mut Self>` and build a SELF-REFERENTIAL struct (the params hold raw
        // pointers into the override Vecs / pattern CStrings). The move-based chain
        // above can't host them, so when any is requested we pin the params, mutate
        // through the pin, and keep that allocation — and the buft pattern CStrings —
        // alive across `load_from_file`. The common case keeps the simple move chain.
        let needs_pin = p.cpu_moe == Some(true)
            || !p.cpu_buft_overrides.is_empty()
            || !p.kv_overrides.is_empty();
        let model = if needs_pin {
            // The buft pattern is stored as a raw POINTER into these CStrings, so they
            // must outlive the load below. (`append_kv_override` instead COPIES the key
            // + value into the struct, so kv key strings need no keep-alive.)
            let patterns: Vec<std::ffi::CString> = p
                .cpu_buft_overrides
                .iter()
                .filter_map(|s| std::ffi::CString::new(s.as_str()).ok())
                .collect();
            let mut pinned = Box::pin(model_params);
            if p.cpu_moe == Some(true) {
                pinned.as_mut().add_cpu_moe_override();
            }
            for cstr in &patterns {
                pinned.as_mut().add_cpu_buft_override(cstr);
            }
            for ov in &p.kv_overrides {
                match std::ffi::CString::new(ov.key.as_str()) {
                    // llama.cpp COPIES the key into a fixed 128-byte buffer with no
                    // bounds check and would panic on overflow, so skip a key whose
                    // NUL-terminated form won't fit (a degenerate input — real GGUF
                    // metadata keys are short) rather than crash the worker.
                    Ok(key) if key.as_bytes_with_nul().len() <= 128 => pinned
                        .as_mut()
                        .append_kv_override(&key, parse_kv_override_value(&ov.value)),
                    Ok(_) => tracing::warn!(
                        key = %ov.key,
                        "higgs: skipping kv_override key longer than 128 bytes"
                    ),
                    Err(_) => {
                        tracing::warn!(key = %ov.key, "higgs: skipping kv_override with NUL in key")
                    }
                }
            }
            let model =
                LlamaModel::load_from_file(backend(), path, &pinned).map_err(|e| load_err(&e));
            // Keep `patterns` alive until the load has consumed the params.
            drop(patterns);
            model?
        } else {
            LlamaModel::load_from_file(backend(), path, &model_params).map_err(|e| load_err(&e))?
        };
        self.loaded = Some(LoadedModel {
            model,
            params: p.clone(),
        });
        Ok(())
    }

    fn unload(&mut self) {
        self.loaded = None;
    }

    fn is_loaded(&self) -> bool {
        self.loaded.is_some()
    }

    fn devices(&self) -> Vec<GpuDevice> {
        device_info()
    }

    fn chat(
        &mut self,
        messages_json: &str,
        params: &GenParams,
        sink: &mut dyn FnMut(EngineDelta<'_>),
    ) -> Result<super::ChatResult, HiggsError> {
        let Some(loaded) = self.loaded.as_ref() else {
            // defensive guard; worker checks first — id unknown at engine level
            return Err(HiggsError::ModelNotLoaded {
                id: "unloaded".into(),
            });
        };

        // 1. Render the GGUF chat template over the OpenAI messages + tools.
        let template = load_template(&loaded.model);
        let tmpl_result = apply_template(&loaded.model, &template, messages_json, params)?;

        // 2. Tokenize and resolve the context-fitted generation budget (clamps an
        //    oversized max_tokens; rejects [HG005] only if the prompt alone overflows).
        let (tokens, fitted_max_tokens) = fit_check(
            &loaded.model,
            tmpl_result.prompt.as_str(),
            &loaded.params,
            params,
        )?;
        let prompt_tokens = tokens.len() as u32;

        // Defensive: the serve layer (serve/v1.rs) already rejects max_tokens==0
        // with HG013 before the request reaches the worker, so this never fires
        // in production. Kept so the engine is correct in isolation.
        if params.max_tokens == 0 {
            return Ok(super::ChatResult {
                content: String::new(),
                finish_reason: "length",
                tool_calls: None,
                prompt_tokens,
                completion_tokens: 0,
                reasoning_content: None,
            });
        }

        // The crate's INCREMENTAL parser for this template (same serialized PEG
        // parser the final parse uses): splits the live stream into content /
        // reasoning / tool-call deltas, llama.cpp-server style. `Err` (no
        // serialized parser — only the legacy non-jinja route, which renders no
        // tool markup) degrades to raw content pieces.
        let stream_state = tmpl_result.streaming_state_oaicompat().ok();

        // 3. Decode loop — streams tagged deltas into `sink`. Use the
        //    context-FITTED budget so generation truncates at the window, not [HG005].
        let gen_params = GenParams {
            max_tokens: fitted_max_tokens,
            ..params.clone()
        };
        let decoded = run_decode(loaded, tokens, &gen_params, stream_state, sink)?;

        // 4. Parse the full generation into an OpenAI message
        //    (content + tool_calls + reasoning_content).
        let parsed = parse_output(&tmpl_result, &decoded.full);

        Ok(super::ChatResult {
            content: parsed.content,
            finish_reason: decoded.finish_reason,
            tool_calls: parsed.tool_calls,
            prompt_tokens,
            // n_generated counts tokens emitted in the decode loop (one per iteration).
            completion_tokens: decoded.n_generated as u32,
            reasoning_content: parsed.reasoning_content,
        })
    }
}

/// Map the engine-agnostic [`FlashAttn`] policy to the raw `llama_cpp_sys_2`
/// value llama.cpp expects (AUTO = -1, DISABLED = 0, ENABLED = 1). The sys type
/// is a plain integer alias; confined to this file per the engine boundary.
fn flash_attn_to_sys(fa: FlashAttn) -> llama_cpp_sys_2::llama_flash_attn_type {
    let raw: i32 = match fa {
        FlashAttn::Auto => -1,
        FlashAttn::Off => 0,
        FlashAttn::On => 1,
    };
    raw as llama_cpp_sys_2::llama_flash_attn_type
}

/// Map the engine-agnostic [`KvCacheKind`] to `llama_cpp_2`'s [`KvCacheType`].
/// Confined to this file per the engine boundary.
fn kv_cache_to_llama(k: KvCacheKind) -> KvCacheType {
    match k {
        KvCacheKind::F32 => KvCacheType::F32,
        KvCacheKind::F16 => KvCacheType::F16,
        KvCacheKind::Q8_0 => KvCacheType::Q8_0,
        KvCacheKind::Q5_1 => KvCacheType::Q5_1,
        KvCacheKind::Q5_0 => KvCacheType::Q5_0,
        KvCacheKind::Q4_1 => KvCacheType::Q4_1,
        KvCacheKind::Q4_0 => KvCacheType::Q4_0,
    }
}

/// Parse a string-valued GGUF metadata override into the typed
/// [`ParamOverrideValue`] llama.cpp expects, guessing the kind bool → int → float
/// → string (a typical numeric/bool override parses to its kind; anything else is
/// kept as a string, truncated to the 127-byte C field). Confined to this file.
fn parse_kv_override_value(s: &str) -> ParamOverrideValue {
    if let Ok(b) = s.parse::<bool>() {
        return ParamOverrideValue::Bool(b);
    }
    if let Ok(i) = s.parse::<i64>() {
        return ParamOverrideValue::Int(i);
    }
    if let Ok(f) = s.parse::<f64>() {
        return ParamOverrideValue::Float(f);
    }
    let mut arr = [0 as std::os::raw::c_char; 128];
    for (i, &byte) in s.as_bytes().iter().take(127).enumerate() {
        arr[i] = byte as std::os::raw::c_char;
    }
    ParamOverrideValue::Str(arr)
}

/// Map higgs's [`SplitMode`] to `llama-cpp-2`'s [`LlamaSplitMode`]. Confined to
/// this file per the engine boundary.
fn split_mode_to_llama(sm: SplitMode) -> LlamaSplitMode {
    match sm {
        SplitMode::None => LlamaSplitMode::None,
        SplitMode::Layer => LlamaSplitMode::Layer,
        SplitMode::Row => LlamaSplitMode::Row,
    }
}

/// Map higgs's [`RopeScalingType`] to `llama-cpp-2`'s [`LlamaRopeScalingType`].
/// Confined to this file per the engine boundary.
fn rope_scaling_to_llama(r: RopeScalingType) -> LlamaRopeScalingType {
    match r {
        RopeScalingType::Unspecified => LlamaRopeScalingType::Unspecified,
        RopeScalingType::None => LlamaRopeScalingType::None,
        RopeScalingType::Linear => LlamaRopeScalingType::Linear,
        RopeScalingType::Yarn => LlamaRopeScalingType::Yarn,
    }
}

/// Construct a `GenerationFailed` diagnostic for a named decode stage.
fn gen_fail(stage: &'static str, reason: &impl ToString) -> HiggsError {
    HiggsError::GenerationFailed {
        stage: stage.to_string(),
        reason: reason.to_string(),
    }
}

/// The model's GGUF-embedded chat template, falling back to `"chatml"` when the
/// model embeds none. The fallback constructor cannot fail for the literal
/// `"chatml"`, so its error path is unreachable in practice.
fn load_template(model: &LlamaModel) -> LlamaChatTemplate {
    match model.chat_template(None) {
        Ok(t) => t,
        Err(e) => {
            tracing::warn!(error = %e, "GGUF chat template unavailable; falling back to chatml");
            LlamaChatTemplate::new("chatml").expect("chatml is a valid built-in template name")
        }
    }
}

/// Apply the OAI-compat chat template over the verbatim OpenAI `messages` JSON
/// (and `tools`, when present), returning the crate's template result.
///
/// The result carries the rendered prompt AND the serialized PEG parser +
/// chat_format the crate's vendored `common_chat` selected for this model; the
/// caller keeps it alive across the decode so `parse_response_oaicompat` can turn
/// the raw output back into an OpenAI message — no parser invented here.
///
/// `add_bos: false` — the prompt is tokenized with `AddBos::Always`, so the
/// template must not also prepend BOS (would double it). Grammar-constrained
/// sampling via `tmpl_result.grammar` is deferred.
fn apply_template(
    model: &LlamaModel,
    template: &LlamaChatTemplate,
    messages_json: &str,
    params: &GenParams,
) -> Result<ChatTemplateResult, HiggsError> {
    let oai_params = OpenAIChatTemplateParams {
        messages_json,
        tools_json: params.tools_json.as_deref(),
        tool_choice: None,
        json_schema: None,
        grammar: None,
        // "auto" == deepseek: the auto-parser bakes a reasoning-extraction rule
        // into the serialized PEG parser whenever the template has reasoning
        // markers, so BOTH the final parse and the streaming diffs separate
        // `reasoning_content` from `content`. Matches llama.cpp's own server
        // default; a no-op for templates without reasoning markers.
        reasoning_format: Some("auto"),
        chat_template_kwargs: params.chat_template_kwargs.as_deref(),
        add_generation_prompt: true,
        use_jinja: true,
        parallel_tool_calls: false,
        // Template-default thinking (upstream inputs default). `false` would
        // ACTIVELY SUPPRESS thinking on Qwen3/DeepSeek-style templates (they
        // prefill an empty think block); templates that never read the jinja
        // var are unaffected. llama.cpp's server enables it whenever the
        // template supports thinking.
        enable_thinking: true,
        add_bos: false,
        add_eos: false,
        parse_tool_calls: params.tools_json.is_some(),
    };
    model
        .apply_chat_template_oaicompat(template, &oai_params)
        .map_err(|e| HiggsError::TemplateRenderFailed {
            reason: e.to_string(),
        })
    // KNOWN LIMITATION: `tmpl_result.additional_stops` (stop STRINGS the template
    // declares) is not honored — the decode loop stops only on EOG tokens /
    // max_tokens. The supported families (Qwen, Llama-3, Gemma, Nemotron)
    // terminate on EOG tokens and declare no stop strings, so this is latent.
}

/// The context window the engine will ACTUALLY use: the pinned [`CtxLen::Fixed`] window,
/// or — for [`CtxLen::Auto`] — the model's trained context (exactly what llama.cpp picks
/// for `with_n_ctx(None)`). Keeps the `Auto` FFI sentinel (`to_n_ctx() == 0`) from being
/// read as a real size by the fit-check / batch sizing (which would reject every chat or
/// force a 1-token batch). In the worker path `ctx_len` is always `Fixed` (handle_load
/// coerces), so the `Auto` arm hardens DIRECT engine loads.
fn effective_n_ctx(load: &LlamaCppParams, model: &LlamaModel) -> u32 {
    match load.ctx_len {
        CtxLen::Fixed { n } => n,
        CtxLen::Auto => model.n_ctx_train(),
    }
}

/// Tokenize `prompt` and resolve the GENERATION budget against `n_ctx` — the largest
/// generation that fits after the prompt. Returns `(prompt_tokens, fitted_max_gen)`.
///
/// Rejects with [HG005] `ContextOverflow` ONLY when the prompt ALONE exceeds the
/// window (no room left to generate). Otherwise CLAMPS the budget to `n_ctx − prompt`
/// (≥ 1) so an oversized `max_tokens` truncates (`finish_reason: "length"`) rather
/// than failing. This is the authoritative, tokenizer-exact backstop for the serve
/// layer's lower-bound `fit_generation_budget` estimate, which can leave the budget a
/// few tokens too high (chat-template tokens it can't see).
fn fit_check(
    model: &LlamaModel,
    prompt: &str,
    load: &LlamaCppParams,
    gen: &GenParams,
) -> Result<(Vec<LlamaToken>, usize), HiggsError> {
    let tokens = model
        .str_to_token(prompt, AddBos::Always)
        .map_err(|e| gen_fail("tokenize prompt", &e))?;
    let n_ctx = effective_n_ctx(load, model) as usize;
    // The prompt alone can't fit → no room to generate → genuine overflow.
    if tokens.len() >= n_ctx {
        return Err(HiggsError::ContextOverflow {
            prompt_tokens: tokens.len(),
            max_gen: gen.max_tokens,
            n_ctx,
        });
    }
    // Clamp the generation budget to the room left after the prompt (≥ 1 token).
    let fitted_max_gen = gen.max_tokens.min(n_ctx - tokens.len()).max(1);
    Ok((tokens, fitted_max_gen))
}

/// The result of [`run_decode`]: the full generated text, the OpenAI finish
/// reason, and the number of tokens emitted in the loop.
struct DecodeOutput {
    full: String,
    finish_reason: &'static str,
    n_generated: usize,
}

/// Feed the prompt `tokens`, then sample → detokenize → stream → re-batch →
/// decode until an EOG token or `max_tokens`.
///
/// Streaming: with `stream_state` (the crate's incremental parser, present
/// whenever the template apply produced a serialized parser) each raw piece is
/// fed to `update(piece, is_partial=true)` and the returned OpenAI delta JSONs
/// are routed as tagged [`EngineDelta`]s — content, reasoning, tool-call —
/// exactly llama.cpp-server's per-token parse-and-diff. One final
/// `update("", false)` after the loop releases text the lenient parser held
/// back. Without it (only the legacy non-jinja route, which renders no tool
/// or reasoning markup), raw pieces stream as `Content`.
///
/// A fresh context is built per request (v1 simplicity: naive full re-prefill);
/// `n_batch` is sized to the context so any fit-checked prompt decodes in one
/// `llama_decode` call (matching llama.cpp's `simple.cpp`).
fn run_decode(
    loaded: &LoadedModel,
    tokens: Vec<LlamaToken>,
    params: &GenParams,
    stream_state: Option<ChatParseStateOaicompat>,
    sink: &mut dyn FnMut(EngineDelta<'_>),
) -> Result<DecodeOutput, HiggsError> {
    let model = &loaded.model;
    let lp = &loaded.params;
    let threads = i32::try_from(lp.threads).unwrap_or(i32::MAX);
    // n_batch: use the pinned value when present, else the current default
    // (ctx_len.max(1) — one-shot prefill of any fit-checked prompt).
    let n_batch = lp
        .n_batch
        .unwrap_or_else(|| effective_n_ctx(lp, model).max(1));
    // `n_threads_batch` splits from `n_threads` when set, else reuses `threads`.
    let threads_batch = lp
        .n_threads_batch
        .map(|n| i32::try_from(n).unwrap_or(i32::MAX))
        .unwrap_or(threads);
    let mut ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(lp.ctx_len.to_n_ctx()))
        .with_n_batch(n_batch)
        .with_n_threads(threads)
        .with_n_threads_batch(threads_batch);
    // Optional context-params overrides — absent (None) leaves the engine default.
    if let Some(n) = lp.n_ubatch {
        ctx_params = ctx_params.with_n_ubatch(n);
    }
    if let Some(n) = lp.n_seq_max {
        ctx_params = ctx_params.with_n_seq_max(n);
    }
    if let Some(b) = lp.offload_kqv {
        ctx_params = ctx_params.with_offload_kqv(b);
    }
    if let Some(b) = lp.swa_full {
        ctx_params = ctx_params.with_swa_full(b);
    }
    if let Some(f) = lp.rope_freq_base {
        ctx_params = ctx_params.with_rope_freq_base(f);
    }
    if let Some(f) = lp.rope_freq_scale {
        ctx_params = ctx_params.with_rope_freq_scale(f);
    }
    if let Some(r) = lp.rope_scaling_type {
        ctx_params = ctx_params.with_rope_scaling_type(rope_scaling_to_llama(r));
    }
    if let Some(fa) = lp.flash_attn {
        ctx_params = ctx_params.with_flash_attention_policy(flash_attn_to_sys(fa));
    }
    if let Some(k) = lp.type_k {
        ctx_params = ctx_params.with_type_k(kv_cache_to_llama(k));
    }
    if let Some(v) = lp.type_v {
        ctx_params = ctx_params.with_type_v(kv_cache_to_llama(v));
    }
    let mut ctx = model
        .new_context(backend(), ctx_params)
        .map_err(|e| gen_fail("create context", &e))?;

    // Prompt feed: logits only for the last prompt token (example shape).
    let mut batch = LlamaBatch::new(tokens.len().max(1), 1);
    let last_index = tokens.len().saturating_sub(1);
    for (i, token) in tokens.into_iter().enumerate() {
        let pos = i32::try_from(i).map_err(|e| gen_fail("prompt position", &e))?;
        batch
            .add(token, pos, &[0], i == last_index)
            .map_err(|e| gen_fail("batch add", &e))?;
    }
    ctx.decode(&mut batch)
        .map_err(|e| gen_fail("prompt decode", &e))?;

    // Build the sampler chain from the request's sampling params (the engine
    // umbrella's LlamaCpp variant). `temperature <= 0` ⇒ greedy (deterministic,
    // ignoring the other samplers, matching llama.cpp). Otherwise the standard order:
    // penalties → top_k → typical → top_p → min_p → top_n_sigma → xtc → temp(/dynatemp)
    // → dist. The seed is the per-request `sampling.seed`, else the load-pinned seed,
    // else fresh entropy.
    let s = params.sampling.as_llamacpp();
    // Advanced samplers needing the model/vocab handle — dry, mirostat, grammar,
    // logit_bias — are carried in the umbrella type but not yet applied here. FAIL
    // LOUD rather than build a chain that silently omits them: returning unconstrained
    // / unbiased output when a caller explicitly requested a grammar or logit bias is a
    // correctness lie. (Unreachable from today's request mappers — `build_sampling`,
    // tune, and card parsing never populate these — so this guards a future caller, not
    // current traffic.) HG013 is the sampling-parameter diagnostic.
    if let Some(param) = s.unsupported_sampler() {
        return Err(HiggsError::InvalidSamplingParam {
            param: param.to_string(),
            detail: "accepted by the API but not yet applied by the llama.cpp engine — \
                     omit it so the response is never silently unconstrained/unbiased \
                     (this sampler lands in a later release)"
                .to_string(),
        });
    }
    let temp = s.temperature.unwrap_or(0.7);
    let seed = s.seed.or(lp.seed).unwrap_or_else(rand::random::<u32>);
    let mut chain: Vec<LlamaSampler> = Vec::new();
    if temp <= 0.0 {
        chain.push(LlamaSampler::greedy());
    } else {
        if s.penalty_repeat.is_some() || s.penalty_freq.is_some() || s.penalty_present.is_some() {
            chain.push(LlamaSampler::penalties(
                s.penalty_last_n.unwrap_or(64),
                s.penalty_repeat.unwrap_or(1.0),
                s.penalty_freq.unwrap_or(0.0),
                s.penalty_present.unwrap_or(0.0),
            ));
        }
        if let Some(k) = s.top_k {
            chain.push(LlamaSampler::top_k(k));
        }
        if let Some(p) = s.typical_p {
            chain.push(LlamaSampler::typical(p, 1));
        }
        if let Some(p) = s.top_p {
            chain.push(LlamaSampler::top_p(p, 1));
        }
        if let Some(p) = s.min_p {
            chain.push(LlamaSampler::min_p(p, 1));
        }
        if let Some(n) = s.top_n_sigma {
            chain.push(LlamaSampler::top_n_sigma(n));
        }
        if let (Some(prob), Some(thr)) = (s.xtc_probability, s.xtc_threshold) {
            chain.push(LlamaSampler::xtc(prob, thr, 1, seed));
        }
        match s.dynatemp_range {
            Some(range) => chain.push(LlamaSampler::temp_ext(
                temp,
                range,
                s.dynatemp_exponent.unwrap_or(1.0),
            )),
            None => chain.push(LlamaSampler::temp(temp)),
        }
        chain.push(LlamaSampler::dist(seed));
    }
    let mut sampler = LlamaSampler::chain_simple(chain);

    // Owned so a mid-stream parse failure can drop the state and degrade the
    // REST of the stream to raw pieces (llama.cpp's diff guard throws when a
    // re-parse is not an extension of the previous one; rare, but it must not
    // kill the generation — the final full-text parse still shapes the result).
    let mut stream_state = stream_state;
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut full = String::new();
    let mut n_generated: usize = 0;
    let mut n_cur = batch.n_tokens();
    let finish_reason = loop {
        let token = sampler.sample(&ctx, batch.n_tokens() - 1);
        sampler.accept(token);
        if model.is_eog_token(token) {
            break "stop";
        }
        let piece = model
            .token_to_piece(token, &mut decoder, true, None)
            .map_err(|e| gen_fail("detokenize", &e))?;
        // The UTF-8 decoder buffers partial multi-byte sequences — only
        // forward pieces that decoded to visible text.
        if !piece.is_empty() {
            emit_piece(&mut stream_state, &piece, sink);
            full.push_str(&piece);
        }
        n_generated += 1;
        if n_generated >= params.max_tokens {
            break "length";
        }
        batch.clear();
        batch
            .add(token, n_cur, &[0], true)
            .map_err(|e| gen_fail("batch add", &e))?;
        n_cur += 1;
        ctx.decode(&mut batch)
            .map_err(|e| gen_fail("loop decode", &e))?;
    };

    // Flush the UTF-8 decoder: a response ending mid-multi-byte sequence (e.g.
    // CJK) would otherwise silently truncate the final character. The final call
    // with last=true drains any buffered incomplete bytes.
    let mut tail = String::new();
    let _ = decoder.decode_to_string(&[], &mut tail, true);
    if !tail.is_empty() {
        emit_piece(&mut stream_state, &tail, sink);
        full.push_str(&tail);
    }
    // End of stream. Parser path: one non-partial update on empty added text
    // releases everything the lenient parse held back (llama.cpp-server's
    // terminal `is_partial=false` re-parse of the SAME state — this is also
    // what streams the tail of a truncated think block).
    if let Some(st) = stream_state.as_mut() {
        route_parsed_deltas(st, "", false, sink);
    }
    Ok(DecodeOutput {
        full,
        finish_reason,
        n_generated,
    })
}

/// Route one raw decoded piece into the sink: through the crate's incremental
/// parser when active (tagged deltas), else as raw content (legacy non-jinja
/// templates only — no tool/reasoning markup exists on that path).
///
/// On a parser error the state is DROPPED and the current piece degrades to
/// raw content — text the parser was still holding back is lost from the
/// stream (bounded: at most the current in-flight parse span), but the final
/// full-text parse still shapes the non-streaming result. Warned once here.
fn emit_piece(
    stream_state: &mut Option<ChatParseStateOaicompat>,
    piece: &str,
    sink: &mut dyn FnMut(EngineDelta<'_>),
) {
    if let Some(st) = stream_state.as_mut() {
        if route_parsed_deltas(st, piece, true, sink) {
            return;
        }
        tracing::warn!(
            "[HG052] incremental chat parse failed mid-generation; raw content deltas for the remainder"
        );
        *stream_state = None;
        // fall through: emit this piece raw so the stream keeps flowing
    }
    sink(EngineDelta::Content(piece));
}

/// Feed `text_added` to the incremental parser and route every returned OpenAI
/// delta JSON into the sink as tagged [`EngineDelta`]s. ONE diff can carry
/// several keys at once (reasoning_content + content + a tool_call fragment —
/// llama.cpp's diff shape), so each key is extracted independently; nothing is
/// dropped because its neighbor key was present. Returns `false` on a parser
/// error (caller degrades to raw streaming).
fn route_parsed_deltas(
    st: &mut ChatParseStateOaicompat,
    text_added: &str,
    is_partial: bool,
    sink: &mut dyn FnMut(EngineDelta<'_>),
) -> bool {
    let deltas = match st.update(text_added, is_partial) {
        Ok(d) => d,
        Err(e) => {
            tracing::debug!(error = %e, "chat parse state update failed");
            return false;
        }
    };
    for delta_json in deltas {
        let Ok(delta) = serde_json::from_str::<serde_json::Value>(&delta_json) else {
            tracing::debug!(delta = %delta_json, "unparseable stream delta json; skipped");
            continue;
        };
        if let Some(r) = delta.get("reasoning_content").and_then(|v| v.as_str()) {
            if !r.is_empty() {
                sink(EngineDelta::Reasoning(r));
            }
        }
        if let Some(c) = delta.get("content").and_then(|v| v.as_str()) {
            if !c.is_empty() {
                sink(EngineDelta::Content(c));
            }
        }
        // The fork serializes tool-call fragments under `tool_calls` (an array
        // of OpenAI chunk-shaped fragments). Forward each fragment verbatim —
        // the serve layer re-emits them as `delta.tool_calls` chunks.
        if let Some(calls) = delta.get("tool_calls").and_then(|v| v.as_array()) {
            for frag in calls {
                let frag_json = frag.to_string();
                sink(EngineDelta::ToolCall(&frag_json));
            }
        }
    }
    true
}

/// The final parse of one generation: the OpenAI message fields higgs reports.
struct ParsedOutput {
    content: String,
    tool_calls: Option<serde_json::Value>,
    /// Model thinking (`reasoning_content`), trimmed; `None` when absent/empty.
    reasoning_content: Option<String>,
}

/// Parse the full generation into an OpenAI message
/// (content + tool_calls + reasoning_content).
///
/// The parser is the one the template apply selected — llama.cpp's PEG
/// auto-parser derived from the GGUF template covers every target family
/// (verified live against Qwen/Llama-3/DeepSeek/Gemma-3/Nemotron/Gemma-4, and
/// by upstream's own test suite for GLM/Mistral). A parse `Err` is therefore a
/// genuine anomaly (malformed output, invalid UTF-8): the raw text is
/// preserved as content with a warning — no second-guessing parser of our own.
fn parse_output(tmpl_result: &ChatTemplateResult, full: &str) -> ParsedOutput {
    match tmpl_result.parse_response_oaicompat(full, false) {
        Ok(msg_json) => match serde_json::from_str::<serde_json::Value>(&msg_json) {
            Ok(parsed) => {
                let tool_calls = parsed.get("tool_calls").filter(|v| !v.is_null()).cloned();
                let reasoning_content = parsed
                    .get("reasoning_content")
                    .and_then(|v| v.as_str())
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(ToOwned::to_owned);
                // `content` is null when the turn is purely tool calls. Only fall
                // back to the raw generated text when there are NO tool calls —
                // otherwise the call markup would be returned as assistant content
                // *alongside* the structured tool_calls (OpenAI requires content
                // to be empty/null on a tool-call turn).
                let content = parsed
                    .get("content")
                    .and_then(|v| v.as_str())
                    .map(ToOwned::to_owned)
                    .unwrap_or_else(|| {
                        if tool_calls.is_some() {
                            String::new()
                        } else {
                            full.to_string()
                        }
                    });
                ParsedOutput {
                    content,
                    tool_calls,
                    reasoning_content,
                }
            }
            Err(e) => {
                // The crate returned non-JSON — an internal bug, not a model
                // quirk. Preserve the raw text so the turn still answers.
                tracing::error!(error = %e, "[HG054] crate parse returned non-JSON message (internal bug); returning raw text");
                ParsedOutput {
                    content: full.to_string(),
                    tool_calls: None,
                    reasoning_content: None,
                }
            }
        },
        Err(e) => {
            tracing::warn!(error = %e, "[HG053] crate parse rejected the generation; returning raw text as content");
            ParsedOutput {
                content: full.to_string(),
                tool_calls: None,
                reasoning_content: None,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic (greedy) sampling umbrella for tests — `temperature: 0`
    /// makes the decode loop pick `argmax`, so generation is reproducible.
    fn greedy_sampling() -> crate::worker::engine::SamplingParams {
        crate::worker::engine::SamplingParams::llamacpp(super::params::LlamaCppSamplingParams {
            temperature: Some(0.0),
            ..Default::default()
        })
    }

    /// Phase-1 milestone: first real token from a local GGUF through the
    /// full template → tokenize → decode → detokenize path.
    #[test]
    #[ignore = "needs HIGGS_TEST_GGUF pointing at a small local gguf"]
    fn first_token_from_real_model() {
        let path = std::env::var("HIGGS_TEST_GGUF").expect("set HIGGS_TEST_GGUF");
        let mut e = LlamaCppEngine::default();
        e.load(
            &path,
            &LoadParams::llamacpp(LlamaCppParams {
                ctx_len: crate::worker::engine::CtxLen::Fixed { n: 2048 },
                gpu_layers: crate::worker::engine::GpuLayers::All,
                threads: 4,
                ..Default::default()
            }),
        )
        .unwrap();
        assert!(e.is_loaded());
        let mut out = String::new();
        let result = e
            .chat(
                r#"[{"role":"user","content":"Say hi in one word."}]"#,
                &GenParams {
                    max_tokens: 8,
                    sampling: greedy_sampling(),
                    tools_json: None,
                    chat_template_kwargs: None,
                },
                &mut |d: EngineDelta<'_>| out.push_str(d.text()),
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

    /// A string-valued GGUF override parses to the matching `ParamOverrideValue`
    /// kind: bool → int → float → string (in that precedence).
    #[test]
    fn kv_override_value_parses_by_type() {
        assert!(matches!(
            parse_kv_override_value("true"),
            ParamOverrideValue::Bool(true)
        ));
        assert!(matches!(
            parse_kv_override_value("8192"),
            ParamOverrideValue::Int(8192)
        ));
        assert!(matches!(
            parse_kv_override_value("1.5"),
            ParamOverrideValue::Float(_)
        ));
        assert!(matches!(
            parse_kv_override_value("llama"),
            ParamOverrideValue::Str(_)
        ));
    }

    /// `FlashAttn` maps to the exact llama.cpp raw values (AUTO=-1, OFF=0, ON=1).
    #[test]
    fn flash_attn_maps_to_sys_values() {
        assert_eq!(flash_attn_to_sys(FlashAttn::Auto) as i32, -1);
        assert_eq!(flash_attn_to_sys(FlashAttn::Off) as i32, 0);
        assert_eq!(flash_attn_to_sys(FlashAttn::On) as i32, 1);
    }

    /// Every `KvCacheKind` maps to its matching `llama_cpp_2` `KvCacheType`.
    #[test]
    fn kv_cache_maps_to_llama_types() {
        // KvCacheType is not PartialEq; match on the result to assert the arm.
        for (kind, ok) in [
            (
                KvCacheKind::F32,
                matches!(kv_cache_to_llama(KvCacheKind::F32), KvCacheType::F32),
            ),
            (
                KvCacheKind::F16,
                matches!(kv_cache_to_llama(KvCacheKind::F16), KvCacheType::F16),
            ),
            (
                KvCacheKind::Q8_0,
                matches!(kv_cache_to_llama(KvCacheKind::Q8_0), KvCacheType::Q8_0),
            ),
            (
                KvCacheKind::Q5_1,
                matches!(kv_cache_to_llama(KvCacheKind::Q5_1), KvCacheType::Q5_1),
            ),
            (
                KvCacheKind::Q5_0,
                matches!(kv_cache_to_llama(KvCacheKind::Q5_0), KvCacheType::Q5_0),
            ),
            (
                KvCacheKind::Q4_1,
                matches!(kv_cache_to_llama(KvCacheKind::Q4_1), KvCacheType::Q4_1),
            ),
            (
                KvCacheKind::Q4_0,
                matches!(kv_cache_to_llama(KvCacheKind::Q4_0), KvCacheType::Q4_0),
            ),
        ] {
            assert!(ok, "{kind:?} mapped to the wrong KvCacheType");
        }
    }

    /// The umbrella serializes internally-tagged on `engine`, flattening the
    /// llama.cpp payload (so the wire stays close to the old flat shape, plus an
    /// `engine` discriminator). Field-level serde coverage lives in `params.rs`.
    #[test]
    fn umbrella_tags_engine_and_flattens_payload() {
        let lp = LoadParams::llamacpp(LlamaCppParams {
            ctx_len: crate::worker::engine::CtxLen::Fixed { n: 8192 },
            gpu_layers: crate::worker::engine::GpuLayers::Count { n: 32 },
            threads: 6,
            type_k: Some(KvCacheKind::Q8_0),
            flash_attn: Some(FlashAttn::On),
            ..Default::default()
        });
        let v = serde_json::to_value(&lp).unwrap();
        assert_eq!(v["engine"], "LlamaCpp");
        assert_eq!(
            v["ctx_len"],
            serde_json::json!({"kind": "fixed", "n": 8192})
        );
        assert_eq!(v["type_k"], "Q8_0");
        let back: LoadParams = serde_json::from_value(v).unwrap();
        assert_eq!(back, lp);
    }
}
