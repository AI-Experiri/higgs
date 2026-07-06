//! Autotune suggester — a **pure advisor** that, given typed GGUF metadata + host
//! hardware (+ optional measured bench), computes a nominal llama.cpp load +
//! sampling parameter set with a fit verdict and rationale. It **never
//! auto-applies**: the caller (the `tune` route / UI / the `autotune_on_load`
//! seam) reviews, optionally edits, then loads with the result.
//!
//! Structure (each unit independently testable behind a narrow trait — DI, no
//! globals; see `src/tune/DESIGN.md` and `docs/DESIGN-autotune.md`):
//! - [`DeriveStrategy`] ([`derive::HeuristicStrategy`]) — GGUF/hardware → base params.
//! - [`VramEstimator`]/[`RamEstimator`] ([`vram`]) — fit verdicts over `system::fits_vram`.
//! - [`SamplingSource`] ([`card_sampling::HfCardSource`]) — HF-card recommended sampling.
//! - [`ProfileStore`] ([`store::JsonModelStore`]) — per-node `~/.higgs/models.json`.
//! - [`Suggester`] — composes the above into a [`TuneSuggestion`].
//!
//! Data types stay llama.cpp-concrete ([`LlamaCppParams`]); only the *behaviors*
//! are abstracted, so extension happens at the strategy/source layer.

pub mod bench;
pub mod card_sampling;
pub mod context;
pub mod derive;
pub mod store;
pub mod vram;

use serde::{Deserialize, Serialize};

use crate::system::HardwareInfo;
use crate::worker::engine::llamacpp::params::{LlamaCppParams, LlamaCppSamplingParams};
use crate::worker::engine::{CtxLen, GpuLayers, KvCacheKind, LoadParams, SamplingParams};
use crate::worker::models::HiggsModel;

use store::TuneRecord;

higgs_ts! {
    /// Caps the user can place on how much of the machine a load may consume. The
    /// suggester derives params *within* these caps instead of against the full
    /// machine. Every field optional; `None` ⇒ uncapped (derive against the whole
    /// machine using the default heuristics). A **suggester input only** — it
    /// shapes the derived params, it is not itself a llama.cpp/worker param.
    #[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
    #[serde(default)]
    pub struct ResourceBudget {
        /// Host RAM ceiling in bytes (CPU-resident weights + CPU KV + cpu_moe experts).
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(type = "number")]
        #[ts(optional)]
        pub max_ram_bytes: Option<u64>,
        /// GPU VRAM ceiling in bytes (else: Σ GPU VRAM × headroom fraction).
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(type = "number")]
        #[ts(optional)]
        pub max_vram_bytes: Option<u64>,
        /// Inference-thread ceiling (caps `n_threads` / `n_threads_batch`).
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(type = "number")]
        #[ts(optional)]
        pub max_cpu_threads: Option<u32>,
    }
}

higgs_const_enum! {
    /// What the `tune` route should do.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
    #[serde(rename_all = "lowercase")]
    pub enum TuneMode {
        /// Cheap, pure: heuristics + best-effort HF-card lookup. No model load. (Default.)
        #[default]
        Suggest,
        /// Explicit, heavy, offline: load candidate(s) and measure tok/s (P2).
        Benchmark,
    }
}

higgs_const_enum! {
    /// Where a suggestion's values came from — surfaced so the UI can label them.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
    pub enum TuneProvenance {
        /// Static heuristics (GGUF/hardware-driven defaults). (Default.)
        #[default]
        Heuristic,
        /// Sampling refined from the model's HuggingFace card.
        Card,
        /// Refined by a measured offline benchmark (P2).
        Bench,
    }
}

higgs_const_enum! {
    /// Tri-state memory fit verdict. Each variant carries a `#[help]`
    /// explanation surfaced to the frontend as `FitVerdictHelp.ts` (tooltip on
    /// the verdict chips), so the wording lives HERE, next to the thresholds
    /// that define it — not hand-copied into the UI.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    #[derive(higgs_macros::TsParamHelp)]
    pub enum FitVerdict {
        /// Comfortably within the budget (≤ 80%).
        #[help = "The estimated memory footprint uses at most 80% of your resource budget for this device. There is comfortable headroom for the OS, other apps, and cache growth during long chats — safe to load. The resource budget is set in Settings → Hardware."]
        Fits,
        /// Close to the budget (80–95%) — loads, but little headroom.
        #[help = "The estimated footprint lands between 80% and 95% of your resource budget. It should load and serve, but headroom is thin — memory pressure from other apps or very long prompts can tip it over. Consider a smaller context length, a quantized KV cache, or a higher resource budget in Settings → Hardware."]
        Tight,
        /// Exceeds the budget (> 95%) — will likely OOM or fail to load.
        #[help = "The estimated footprint exceeds 95% of your resource budget, so the load will most likely fail or force the system into swapping. Lower the context length, quantize the KV cache, offload fewer GPU layers, or raise the resource budget in Settings → Hardware."]
        Overflow,
    }
}

higgs_ts! {
    /// A memory fit verdict plus the byte figures it was computed from.
    #[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
    pub struct FitReport {
        /// Fits / Tight / Overflow.
        pub verdict: FitVerdict,
        /// Estimated bytes the load needs against this budget.
        #[ts(type = "number")]
        pub needed_bytes: u64,
        /// The budget basis in bytes (the VRAM/RAM cap or detected total).
        #[ts(type = "number")]
        pub budget_bytes: u64,
    }
}

higgs_ts! {
    /// The suggester's output: a full parameter set, fit verdicts, rationale, and
    /// provenance. The button's suggested numbers populate the editable load-params
    /// fields; the user can hand-adjust before loading. `load`/`sampling` are the
    /// engine umbrellas (the `LlamaCpp` variant today).
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct TuneSuggestion {
        /// The model id this suggestion is for.
        pub id: String,
        /// Suggested load parameters (the umbrella; `LlamaCpp` variant).
        pub load: LoadParams,
        /// Suggested sampling parameters (the umbrella; `LlamaCpp` variant).
        pub sampling: SamplingParams,
        /// GPU VRAM fit verdict for the suggested load.
        pub vram_fit: FitReport,
        /// Host RAM fit verdict for the suggested load.
        pub ram_fit: FitReport,
        /// Human-readable notes explaining the non-default choices + the verdict.
        pub rationale: Vec<String>,
        /// Where the values came from.
        pub provenance: TuneProvenance,
        /// The caps the suggestion was derived within (echoed for a re-tune).
        pub budget: ResourceBudget,
    }
}

higgs_ts! {
    /// Request body for `POST /api/higgs/models/tune` (the model id is in the body,
    /// not the path — higgs ids are slashed/colon'd).
    #[derive(Debug, Clone, Deserialize)]
    pub struct TuneRequest {
        /// HuggingFace repo id (or `ollama/name:tag`) of the model to tune.
        pub id: String,
        /// Suggest (default, cheap) vs Benchmark (explicit, offline — P2). Omittable
        /// — `None` ⇒ `Suggest` server-side.
        #[serde(default)]
        #[ts(optional)]
        pub mode: Option<TuneMode>,
        /// Optional resource caps to derive within.
        #[serde(default)]
        #[ts(optional)]
        pub budget: Option<ResourceBudget>,
        /// BENCHMARK-mode pins: hold these load params fixed while turbotune
        /// searches the rest. A pinned KV cache type suppresses the KV-quant
        /// candidate rungs; pinned GPU layers suppress the half-layers rung.
        /// Ignored in `Suggest` mode (the analytical suggestion stays free —
        /// the user edits it in the UI and loads with explicit params).
        #[serde(default)]
        #[ts(optional)]
        pub pins: Option<TunePins>,
    }
}

higgs_ts! {
    /// User-pinned load params for a turbotune run — each `Some` value is
    /// applied to the benchmark seed verbatim and EXCLUDED from the candidate
    /// search (turbotune measures only the unpinned dimensions).
    #[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
    pub struct TunePins {
        /// Pin the context window (e.g. `Fixed { n: 16384 }`).
        #[serde(default)]
        #[ts(optional)]
        pub ctx_len: Option<CtxLen>,
        /// Pin the GPU offload — suppresses the half-layers candidate rung.
        #[serde(default)]
        #[ts(optional)]
        pub gpu_layers: Option<GpuLayers>,
        /// Pin the K-cache type — suppresses the KV-quant candidate rungs.
        #[serde(default)]
        #[ts(optional)]
        pub type_k: Option<KvCacheKind>,
        /// Pin the V-cache type — suppresses the KV-quant candidate rungs.
        #[serde(default)]
        #[ts(optional)]
        pub type_v: Option<KvCacheKind>,
    }
}

higgs_ts! {
    /// Request for `POST /api/higgs/models/estimate`: the memory footprint of a
    /// CANDIDATE load (the user's current context window / KV types / GPU offload),
    /// so the UI shows "≈ X GiB VRAM · Fits/Tight/Overflow" live as they edit. Pure +
    /// cheap (no model load) — higgs OWNS the formula (reuses the suggester's VRAM/RAM
    /// estimators); the frontend never reimplements it. The verdict is measured against
    /// the supplied `budget` (the app's % cap resolved to bytes — so the live verdict
    /// matches the budget-aware tune), or the detected machine when no budget is sent.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct EstimateRequest {
        /// The model whose footprint to estimate (a scanned id).
        pub id: String,
        /// Candidate context window.
        pub ctx_len: CtxLen,
        /// Candidate GPU offload (`None` ⇒ all layers).
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        pub gpu_layers: Option<GpuLayers>,
        /// Candidate KV key cache type (`None` ⇒ F16).
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        pub type_k: Option<KvCacheKind>,
        /// Candidate KV value cache type (`None` ⇒ F16).
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        pub type_v: Option<KvCacheKind>,
        /// Candidate KV-cache offload (`Some(false)` keeps KV in host RAM, not VRAM;
        /// `None` ⇒ on GPU, the default). Memory-affecting, so the estimate honors it.
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        pub offload_kqv: Option<bool>,
        /// Candidate MoE expert offload to CPU (a tuned, non-visible param that moves
        /// expert weights off the GPU). `None` ⇒ off. Memory-affecting on MoE models.
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        pub cpu_moe: Option<bool>,
        /// Resource cap the verdict is measured against (the app's % budget resolved
        /// to bytes). `None` ⇒ the detected machine — keeps the live estimate
        /// consistent with the budget-aware tune when a budget is set.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        pub budget: Option<ResourceBudget>,
    }
}

higgs_ts! {
    /// Response for `POST /api/higgs/models/estimate`: the VRAM + RAM footprint of the
    /// candidate load against the detected machine (verdict + needed/basis bytes).
    #[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
    pub struct EstimateReport {
        /// GPU-VRAM footprint + fit verdict.
        pub vram: FitReport,
        /// Host-RAM footprint + fit verdict.
        pub ram: FitReport,
    }
}

/// Typed GGUF facts the suggester needs — the host-side projection of a scanned
/// [`HiggsModel`]. Internal (cached in `models.json`); not a wire type.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ModelMeta {
    /// Model id (HF `org/model` / `ollama/name:tag`).
    pub id: String,
    /// GGUF file size in bytes — the lower-bound proxy for resident weights.
    pub size_bytes: u64,
    /// Architecture (e.g. `"llama"`).
    pub arch: Option<String>,
    /// Transformer block (layer) count.
    pub block_count: Option<u32>,
    /// Attention query-head count.
    pub head_count: Option<u32>,
    /// Attention KV/GQA head count (the KV-cache size driver).
    pub head_count_kv: Option<u32>,
    /// Embedding / hidden size.
    pub embedding_length: Option<u32>,
    /// Trained context length.
    pub ctx_train: Option<u64>,
    /// MoE expert count (0/None ⇒ dense).
    pub expert_count: Option<u32>,
    /// Whether the GGUF embeds a chat template.
    pub has_chat_template: bool,
}

impl ModelMeta {
    /// Project a scanned [`HiggsModel`] into the suggester's typed input.
    pub fn from_model(m: &HiggsModel) -> Self {
        Self {
            id: m.id.clone(),
            size_bytes: m.size_bytes,
            arch: m.arch.clone(),
            block_count: m.block_count,
            head_count: m.head_count,
            head_count_kv: m.head_count_kv,
            embedding_length: m.embedding_length,
            ctx_train: m.ctx_train,
            expert_count: m.expert_count,
            has_chat_template: m.has_chat_template,
        }
    }

    /// Attention head dimension: `embedding_length / head_count` when both are
    /// known, else the common `128` fallback (accepting some estimate error on
    /// outlier architectures).
    pub fn head_dim(&self) -> u32 {
        match (self.embedding_length, self.head_count) {
            (Some(embd), Some(heads)) if heads > 0 => (embd / heads).max(1),
            _ => 128,
        }
    }

    /// KV/GQA head count for the KV-cache estimate, falling back to `head_count`
    /// (no GQA) then a conservative `8`.
    pub fn kv_heads(&self) -> u32 {
        self.head_count_kv.or(self.head_count).unwrap_or(8).max(1)
    }

    /// True when this is a Mixture-of-Experts model (drives `cpu_moe` back-off).
    pub fn is_moe(&self) -> bool {
        self.expert_count.unwrap_or(0) > 0
    }
}

/// A measured offline benchmark result (P2).
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct BenchResult {
    /// Generation throughput (tokens/second).
    pub gen_tps: f32,
    /// Prompt-processing throughput (tokens/second).
    pub prompt_tps: f32,
    /// Time-to-first-token in milliseconds.
    pub ttft_ms: f32,
}

// ── Extension seams (narrow traits — DI) ────────────────────────────────────

/// Provides typed GGUF metadata for a model id (from the scan + `models.json` cache).
pub trait ModelMetaProvider {
    /// Look up typed metadata for `id`. `Err` when the model is unknown/unreadable.
    fn meta(&self, id: &str) -> Result<ModelMeta, crate::diagnostic::HiggsError>;
}

/// Provides the host hardware snapshot (cached; a worker round-trip on a miss).
pub trait HardwareProvider {
    /// The host CPU/RAM/GPU snapshot.
    fn hardware(&self) -> Result<HardwareInfo, crate::diagnostic::HiggsError>;
}

/// Estimates the GPU-VRAM fit of a candidate load.
pub trait VramEstimator {
    /// VRAM fit verdict for `load` against the budget.
    fn estimate(
        &self,
        load: &LlamaCppParams,
        meta: &ModelMeta,
        hw: &HardwareInfo,
        budget: &ResourceBudget,
    ) -> FitReport;
}

/// Estimates the host-RAM fit of a candidate load (CPU-resident weights/KV/experts).
pub trait RamEstimator {
    /// RAM fit verdict for `load` against the budget.
    fn estimate(
        &self,
        load: &LlamaCppParams,
        meta: &ModelMeta,
        hw: &HardwareInfo,
        budget: &ResourceBudget,
    ) -> FitReport;
}

/// Recommends sampling parameters for a model (e.g. from its HF card).
pub trait SamplingSource {
    /// Recommended sampling; empty (all-`None`) when nothing is known.
    fn recommend(&self, meta: &ModelMeta) -> LlamaCppSamplingParams;
}

/// Derives base load parameters from metadata + hardware within a budget.
pub trait DeriveStrategy {
    /// The heuristic base load parameters.
    fn derive(
        &self,
        meta: &ModelMeta,
        hw: &HardwareInfo,
        budget: &ResourceBudget,
    ) -> LlamaCppParams;
}

/// Measures tok/s for a candidate config by actually loading it (P2, offline).
pub trait Benchmarker {
    /// Measure `cand` for model `id`.
    fn measure(
        &self,
        id: &str,
        cand: &LlamaCppParams,
    ) -> Result<BenchResult, crate::diagnostic::HiggsError>;
}

/// Reads/writes the saved per-model tuning profile.
pub trait ProfileStore {
    /// The saved tuning record for `id`, if any.
    fn tuning(&self, id: &str) -> Option<TuneRecord>;
    /// Persist a tuning record for `id`.
    fn put_tuning(&self, id: &str, record: TuneRecord);
}

/// True when the host has a usable GPU (non-zero summed VRAM).
pub fn has_gpu(hw: &HardwareInfo) -> bool {
    hw.vram_total_bytes > 0
}

/// Composes the derive/estimate/sampling units into a [`TuneSuggestion`]. Generic
/// over the trait impls so tests inject fakes (no worker, no disk, no network).
pub struct Suggester<D, V, R, S> {
    /// Base-parameter derivation.
    pub derive: D,
    /// VRAM fit estimator.
    pub vram: V,
    /// RAM fit estimator.
    pub ram: R,
    /// Sampling recommendation source.
    pub sampling: S,
}

impl
    Suggester<
        derive::HeuristicStrategy,
        vram::StaticVramEstimator,
        vram::StaticRamEstimator,
        card_sampling::EmptySamplingSource,
    >
{
    /// The default, fully-static suggester (no network, no model load) — the
    /// `autotune_on_load` / offline shape. Wire a [`card_sampling::HfCardSource`]
    /// in for the explicit button path that may fetch the HF card.
    pub fn static_default() -> Self {
        Suggester {
            derive: derive::HeuristicStrategy,
            vram: vram::StaticVramEstimator,
            ram: vram::StaticRamEstimator,
            sampling: card_sampling::EmptySamplingSource,
        }
    }
}

impl<D, V, R, S> Suggester<D, V, R, S>
where
    D: DeriveStrategy,
    V: VramEstimator,
    R: RamEstimator,
    S: SamplingSource,
{
    /// Run the full suggestion: derive (fresh, within budget) → sampling → VRAM
    /// fit → MoE back-off (only when RAM still fits) → RAM fit → rationale. Pure
    /// over its inputs. Saved-profile *reuse* is a load-seam concern, not here.
    pub fn suggest(
        &self,
        meta: &ModelMeta,
        hw: &HardwareInfo,
        budget: &ResourceBudget,
    ) -> TuneSuggestion {
        let mut rationale: Vec<String> = Vec::new();

        // 1. Heuristic base — always derived FRESH within the supplied budget. A
        //    prior saved profile is NOT reused here: that is a load-seam concern
        //    (`Higgs::load` reuses the saved profile on a plain load). Re-running
        //    `tune` with a stricter budget must re-derive within it, not echo stale
        //    params — so the budget is genuinely honored on every (re-)tune.
        let mut load = self.derive.derive(meta, hw, budget);
        let mut provenance = TuneProvenance::Heuristic;

        // 2. Sampling from the source (HF card / empty).
        let sampling = self.sampling.recommend(meta);
        if sampling != LlamaCppSamplingParams::default() {
            provenance = TuneProvenance::Card;
            rationale.push(SAMPLING_REFINED_NOTE.to_owned());
        }

        // 3. VRAM fit.
        let mut vram_fit = self.vram.estimate(&load, meta, hw, budget);

        // 4. MoE back-off: if VRAM overflows on a GPU host and this is an MoE
        //    model, try pushing experts to CPU — but only accept it if RAM holds.
        if vram_fit.verdict == FitVerdict::Overflow
            && meta.is_moe()
            && has_gpu(hw)
            && load.cpu_moe != Some(true)
        {
            let mut candidate = load.clone();
            candidate.cpu_moe = Some(true);
            let ram_if_moe = self.ram.estimate(&candidate, meta, hw, budget);
            if ram_if_moe.verdict != FitVerdict::Overflow {
                let vram_if_moe = self.vram.estimate(&candidate, meta, hw, budget);
                load = candidate;
                vram_fit = vram_if_moe;
                rationale.push("moved MoE experts to CPU (cpu_moe) to fit GPU VRAM".into());
            } else {
                rationale.push(
                    "VRAM overflows and offloading experts to CPU would exceed the RAM budget"
                        .into(),
                );
            }
        }

        // 5. VRAM-budget back-off: when the user supplied an explicit `max_vram_bytes`
        //    cap and the load still overflows it, reduce GPU offload (partial layers,
        //    or CPU-only) so the suggestion HONORS the cap — instead of an all-GPU
        //    profile that would later be reused for a plain load and blow the budget.
        //    (No cap ⇒ unchanged: all-GPU + an honest overflow verdict.)
        if budget.max_vram_bytes.is_some()
            && vram_fit.verdict == FitVerdict::Overflow
            && !load.gpu_layers.is_cpu_only()
        {
            let fitted = vram::gpu_layers_within_budget(&load, meta, hw, budget);
            // Reduce GPU offload to `fitted` layers when that's a genuine cut: `All` always
            // caps to the fitted count; a partial offload caps only when it's smaller.
            let cut = match load.gpu_layers {
                GpuLayers::All => Some(fitted),
                GpuLayers::Count { n } if fitted < n => Some(fitted),
                GpuLayers::Count { .. } => None,
            };
            if let Some(n) = cut {
                load.gpu_layers = GpuLayers::Count { n };
                // (vram_fit is recomputed in step 6 against the final load + context.)
                rationale.push(if n == 0 {
                    "no GPU offload fits the VRAM budget — using CPU-only".into()
                } else {
                    format!("reduced GPU offload to {n} layers to fit the VRAM budget")
                });
            }
        }

        // 5b. Re-derive the context for the FINAL offload. The base context was derived
        //     for the initial all-GPU placement; the MoE / VRAM back-offs above may have
        //     moved experts to CPU or reduced GPU layers, which changes where the KV
        //     lands and thus how large a window the budget affords. `derive_ctx` inverts
        //     the forward estimators for the FINAL `load`, so a CPU-only fallback derives
        //     against RAM and a partial offload against whichever pool binds — instead of
        //     being pinned at MIN_CTX. Idempotent for an unchanged all-GPU load. Done
        //     BEFORE the RAM fit so the verdict reflects the final context.
        let (ctx, deriv) = derive::derive_ctx(&load, meta, hw, budget);
        load.ctx_len = CtxLen::Fixed { n: ctx };
        let mut ctx_note = match meta.ctx_train {
            Some(t) if u64::from(ctx) >= t => {
                format!("context {ctx} — the model's full trained window")
            }
            _ => format!("context {ctx} — the largest that fits the budget"),
        };
        if deriv.methods > 1 {
            ctx_note.push_str(&format!(
                " (avg of {} methods, range {}–{})",
                deriv.methods, deriv.min, deriv.max
            ));
        } else {
            ctx_note.push_str(" (analytical)");
        }
        rationale.push(ctx_note);

        // 6. Final fit verdicts — recomputed against the FINAL load (incl. the
        //    re-derived context in 5b, which for a partial offload changes the
        //    GPU-resident KV), so the verdicts reflect exactly what will load.
        vram_fit = self.vram.estimate(&load, meta, hw, budget);
        let ram_fit = self.ram.estimate(&load, meta, hw, budget);
        rationale.push(verdict_note("GPU VRAM", vram_fit));
        rationale.push(verdict_note("system RAM", ram_fit));

        TuneSuggestion {
            id: meta.id.clone(),
            load: LoadParams::llamacpp(load),
            sampling: SamplingParams::llamacpp(sampling),
            vram_fit,
            ram_fit,
            rationale,
            provenance,
            budget: budget.clone(),
        }
    }
}

/// One-line human note for a fit verdict.
/// The rationale line noting the model card refined sampling — shared by the
/// analytical suggest and the benchmark's [`benchmarked_rationale`] so the note reads
/// identically whichever path produced it.
pub(crate) const SAMPLING_REFINED_NOTE: &str = "sampling refined from the model card";

/// The rationale for a benchmarked config. It REPLACES the analytical seed's
/// rationale: the seed's context / GPU-layer / fit lines describe the SEED, and
/// a benchmark can diverge from it on several dimensions (a pinned context or KV
/// type, the winning KV-quant rung, a half-offload rung), so keeping those lines
/// would contradict the benchmarked config actually saved. Narrates the measured
/// throughput, the benchmarked config's context, and its recomputed fit;
/// `sampling_refined` re-adds the card-sampling note (a load-only benchmark
/// leaves sampling unchanged).
pub(crate) fn benchmarked_rationale(
    benchmarked: &LlamaCppParams,
    ctx_train: Option<u64>,
    vram_fit: FitReport,
    ram_fit: FitReport,
    gen_tps: f32,
    sampling_refined: bool,
) -> Vec<String> {
    let mut lines = Vec::new();
    if sampling_refined {
        lines.push(SAMPLING_REFINED_NOTE.to_owned());
    }
    lines.push(format!(
        "Turbotune measured {gen_tps:.1} tok/s — fastest of the benched configs on this hardware"
    ));
    lines.push(match benchmarked.ctx_len {
        CtxLen::Auto => "context: engine default (capped at the node context limit)".to_owned(),
        // A pin can set a context ABOVE `ctx_train` (the analytical path clamps to
        // it, but pins are applied verbatim), so distinguish exactly-trained from
        // beyond-trained — only the former is the "full trained window".
        CtxLen::Fixed { n } => match ctx_train {
            Some(t) if u64::from(n) > t => {
                format!("context {n} — pinned beyond the model's trained window ({t})")
            }
            Some(t) if u64::from(n) == t => {
                format!("context {n} — the model's full trained window")
            }
            _ => format!("context {n} — the benchmarked configuration"),
        },
    });
    lines.push(verdict_note("GPU VRAM", vram_fit));
    lines.push(verdict_note("system RAM", ram_fit));
    lines
}

fn verdict_note(label: &str, fit: FitReport) -> String {
    let pct = if fit.budget_bytes > 0 {
        (fit.needed_bytes as f64 / fit.budget_bytes as f64 * 100.0).round() as u64
    } else {
        0
    };
    let gib = |b: u64| format!("{:.1} GiB", b as f64 / (1u64 << 30) as f64);
    match fit.verdict {
        FitVerdict::Fits => format!(
            "{label}: fits ({} of {}, {pct}%)",
            gib(fit.needed_bytes),
            gib(fit.budget_bytes)
        ),
        FitVerdict::Tight => format!(
            "{label}: tight ({} of {}, {pct}%) — little headroom",
            gib(fit.needed_bytes),
            gib(fit.budget_bytes)
        ),
        FitVerdict::Overflow => format!(
            "{label}: overflow ({} of {}, {pct}%) — may fail to load",
            gib(fit.needed_bytes),
            gib(fit.budget_bytes)
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A benchmark pin can set a context ABOVE the model's trained window (pins
    /// are applied verbatim; only the analytical path clamps to `ctx_train`), so
    /// the benchmarked rationale must NOT call an over-trained pin "the full
    /// trained window" — only an exactly-trained context earns that label.
    /// Fail-on-revert: collapsing the `> t` and `== t` arms back to `>= t`
    /// mislabels the 8192-over-4096 pin.
    #[test]
    fn benchmarked_rationale_labels_an_over_trained_pin_honestly() {
        let fit = FitReport {
            verdict: FitVerdict::Fits,
            needed_bytes: 1,
            budget_bytes: 10,
        };
        let ctx_line = |n: u32, trained: u64| {
            let load = LlamaCppParams {
                ctx_len: CtxLen::Fixed { n },
                ..Default::default()
            };
            benchmarked_rationale(&load, Some(trained), fit, fit, 12.0, false)
                .into_iter()
                .find(|l| l.starts_with("context "))
                .expect("a context rationale line")
        };
        // Pinned ABOVE the trained window → NOT "full trained window".
        let over = ctx_line(8192, 4096);
        assert!(over.contains("8192"), "{over}");
        assert!(
            !over.contains("full trained window"),
            "an over-trained pin must not be called the full trained window: {over}"
        );
        assert!(
            over.contains("beyond"),
            "it is labelled as beyond-trained: {over}"
        );
        // Pinned EXACTLY at the trained window → that IS the full window.
        assert!(
            ctx_line(4096, 4096).contains("full trained window"),
            "an exactly-trained context is the full window"
        );
        // Pinned BELOW → the benchmarked configuration.
        assert!(ctx_line(2048, 4096).contains("benchmarked configuration"));
    }

    #[test]
    fn budget_default_is_uncapped() {
        let b = ResourceBudget::default();
        assert!(
            b.max_ram_bytes.is_none() && b.max_vram_bytes.is_none() && b.max_cpu_threads.is_none()
        );
        // Round-trips and serializes empty when uncapped.
        assert_eq!(serde_json::to_value(&b).unwrap(), serde_json::json!({}));
    }

    #[test]
    fn tune_mode_defaults_to_suggest_and_renames_lowercase() {
        assert_eq!(TuneMode::default(), TuneMode::Suggest);
        assert_eq!(
            serde_json::to_string(&TuneMode::Benchmark).unwrap(),
            "\"benchmark\""
        );
        let m: TuneMode = serde_json::from_str("\"suggest\"").unwrap();
        assert_eq!(m, TuneMode::Suggest);
    }

    #[test]
    fn model_meta_head_dim_and_kv_heads() {
        let m = ModelMeta {
            embedding_length: Some(4096),
            head_count: Some(32),
            head_count_kv: Some(8),
            expert_count: Some(0),
            ..Default::default()
        };
        assert_eq!(m.head_dim(), 128); // 4096/32
        assert_eq!(m.kv_heads(), 8); // GQA value
        assert!(!m.is_moe());
        // Fallbacks when fields are absent.
        let bare = ModelMeta::default();
        assert_eq!(bare.head_dim(), 128);
        assert_eq!(bare.kv_heads(), 8);
    }

    // ── Suggester orchestration ──────────────────────────────────────────────
    use crate::system::{DeviceKind, GpuDevice};

    fn hw(vram: u64, ram: u64, cores: u32) -> HardwareInfo {
        HardwareInfo {
            cpu_name: "t".into(),
            arch: "aarch64".into(),
            cpu_cores: cores,
            ram_total_bytes: ram,
            ram_used_bytes: ram / 8,
            cpu_usage_percent: 1.0,
            gpus: if vram > 0 {
                vec![GpuDevice {
                    name: "Metal".into(),
                    description: "g".into(),
                    kind: DeviceKind::Gpu,
                    vram_total_bytes: vram,
                    vram_free_bytes: vram,
                }]
            } else {
                vec![]
            },
            vram_total_bytes: vram,
        }
    }

    fn dense_meta(size: u64) -> ModelMeta {
        ModelMeta {
            id: "org/m".into(),
            size_bytes: size,
            block_count: Some(32),
            head_count: Some(32),
            head_count_kv: Some(8),
            embedding_length: Some(4096),
            ctx_train: Some(8192),
            expert_count: Some(0),
            ..Default::default()
        }
    }

    fn moe_meta(size: u64) -> ModelMeta {
        ModelMeta {
            expert_count: Some(8),
            ..dense_meta(size)
        }
    }

    fn default_suggester() -> Suggester<
        derive::HeuristicStrategy,
        vram::StaticVramEstimator,
        vram::StaticRamEstimator,
        card_sampling::EmptySamplingSource,
    > {
        Suggester::static_default()
    }

    struct FakeCard(LlamaCppSamplingParams);
    impl SamplingSource for FakeCard {
        fn recommend(&self, _m: &ModelMeta) -> LlamaCppSamplingParams {
            self.0.clone()
        }
    }

    #[test]
    fn suggest_default_provenance_and_fits_note() {
        let s = default_suggester().suggest(
            &dense_meta(4u64 << 30),
            &hw(24u64 << 30, 64u64 << 30, 8),
            &ResourceBudget::default(),
        );
        assert_eq!(s.provenance, TuneProvenance::Heuristic);
        assert_eq!(s.vram_fit.verdict, FitVerdict::Fits);
        assert!(
            s.rationale.iter().any(|r| r.contains("fits")),
            "{:?}",
            s.rationale
        );
        assert!(matches!(s.load, LoadParams::LlamaCpp(_)));
    }

    #[test]
    fn cpu_only_backoff_rederives_context_against_ram() {
        // GPU host, but an explicit VRAM cap too small to hold the 8 GiB model → the
        // suggester backs GPU offload off to CPU-only. The context must THEN be
        // re-derived against the (ample) RAM budget — not left pinned at MIN_CTX from
        // the tiny VRAM basis. This is the derive↔back-off interaction codex flagged.
        let budget = ResourceBudget {
            // Below the ~800 MiB compute overhead → not even one layer fits the GPU,
            // so the back-off goes all the way to CPU-only (not a partial offload).
            max_vram_bytes: Some(500u64 << 20),
            ..Default::default()
        };
        let s = default_suggester().suggest(
            &dense_meta(8u64 << 30),
            &hw(24u64 << 30, 128u64 << 30, 8),
            &budget,
        );
        // LoadParams has a single (LlamaCpp) variant today — an irrefutable bind.
        let LoadParams::LlamaCpp(load) = &s.load;
        assert!(
            load.gpu_layers.is_cpu_only(),
            "a tiny VRAM budget backs off to CPU-only: {:?}",
            load.gpu_layers
        );
        // 128 GiB RAM easily holds the 8 GiB model + KV for the full 8192 window, so
        // the CPU-only context recovers ctx_train — NOT the MIN_CTX it would be pinned
        // at if the context were never re-derived off the 2 GiB VRAM basis.
        let CtxLen::Fixed { n } = load.ctx_len else {
            panic!("expected a fixed ctx")
        };
        assert_eq!(
            n, 8192,
            "CPU-only context re-derived against RAM to the trained window, got {n}"
        );
    }

    #[test]
    fn suggest_card_sampling_sets_card_provenance() {
        let card = FakeCard(LlamaCppSamplingParams {
            temperature: Some(0.6),
            top_p: Some(0.95),
            ..Default::default()
        });
        let s = Suggester {
            derive: derive::HeuristicStrategy,
            vram: vram::StaticVramEstimator,
            ram: vram::StaticRamEstimator,
            sampling: card,
        }
        .suggest(
            &dense_meta(4u64 << 30),
            &hw(24u64 << 30, 64u64 << 30, 8),
            &ResourceBudget::default(),
        );
        assert_eq!(s.provenance, TuneProvenance::Card);
        assert_eq!(s.sampling.as_llamacpp().temperature, Some(0.6));
        assert!(s.rationale.iter().any(|r| r.contains("model card")));
    }

    #[test]
    fn suggest_tight_verdict_emits_note() {
        // 8 GiB weights + ~1 GiB KV + overhead ≈ 9.8 GiB against an 11 GiB GPU →
        // above the 80% headroom but under 95% → Tight.
        let s = default_suggester().suggest(
            &dense_meta(8u64 << 30),
            &hw(11u64 << 30, 64u64 << 30, 8),
            &ResourceBudget::default(),
        );
        assert_eq!(s.vram_fit.verdict, FitVerdict::Tight);
        assert!(
            s.rationale.iter().any(|r| r.contains("tight")),
            "{:?}",
            s.rationale
        );
    }

    #[test]
    fn suggest_moe_overflow_with_ram_also_full_keeps_overflow() {
        // A 40 GiB MoE that overflows an 8 GiB GPU AND a 16 GiB RAM box → cpu_moe
        // is NOT accepted (RAM can't hold the experts) and VRAM stays Overflow.
        let s = default_suggester().suggest(
            &moe_meta(40u64 << 30),
            &hw(8u64 << 30, 16u64 << 30, 8),
            &ResourceBudget::default(),
        );
        assert_eq!(
            s.load.as_llamacpp().cpu_moe,
            None,
            "cpu_moe rejected: {:?}",
            s.rationale
        );
        assert_eq!(s.vram_fit.verdict, FitVerdict::Overflow);
        assert!(
            s.rationale.iter().any(|r| r.contains("RAM budget")),
            "explains the RAM-budget rejection: {:?}",
            s.rationale
        );
    }

    #[test]
    fn suggest_honors_vram_cap_by_backing_off_gpu_layers() {
        let meta = dense_meta(8u64 << 30);
        let hw24 = hw(24u64 << 30, 64u64 << 30, 8);

        // A `max_vram_bytes: 0` cap (avoid GPU) → CPU-only, fitting the cap.
        let zero = ResourceBudget {
            max_vram_bytes: Some(0),
            ..Default::default()
        };
        let s0 = default_suggester().suggest(&meta, &hw24, &zero);
        assert_eq!(
            s0.load.as_llamacpp().gpu_layers,
            GpuLayers::Count { n: 0 },
            "0 VRAM cap → CPU-only: {:?}",
            s0.rationale
        );
        assert_eq!(s0.vram_fit.verdict, FitVerdict::Fits);

        // A small (4 GiB) cap can't hold the 8 GiB weights all-GPU → partial offload,
        // not the all-GPU u32::MAX that would blow the cap.
        let small = ResourceBudget {
            max_vram_bytes: Some(4u64 << 30),
            ..Default::default()
        };
        let ss = default_suggester().suggest(&meta, &hw24, &small);
        assert!(
            ss.load.as_llamacpp().gpu_layers != GpuLayers::All,
            "small VRAM cap backs off all-GPU: {:?}",
            ss.rationale
        );
        assert_ne!(
            ss.vram_fit.verdict,
            FitVerdict::Overflow,
            "now fits the cap"
        );
    }

    #[test]
    fn suggest_judges_corrupt_metadata_as_wont_fit() {
        // A corrupt/hostile GGUF header (absurd block_count/ctx/heads) overflows the
        // raw KV-byte product; `kv_cache_bytes` saturates rather than panicking. This
        // exercises the END-TO-END contract that matters: the saturated KV must still
        // make the model read "won't fit", through the partial-offload backoff path a
        // VRAM cap takes (where `vram_fit` alone could cosmetically read Fits after
        // `*frac` scaling). Because the un-offloaded KV remainder `kv*(1-frac)` is
        // astronomically large for ANY frac, `ram_fit` (or `vram_fit`) is Overflow —
        // the combined verdict is never a false "fits". Must not panic.
        let evil = ModelMeta {
            id: "org/evil".into(),
            size_bytes: 8u64 << 30,
            block_count: Some(u32::MAX),
            head_count: Some(u32::MAX),
            head_count_kv: Some(u32::MAX),
            embedding_length: Some(u32::MAX),
            ctx_train: Some(u64::MAX),
            expert_count: Some(0),
            ..Default::default()
        };
        let hw24 = hw(24u64 << 30, 64u64 << 30, 8);
        // A VRAM cap forces the gpu_layers_within_budget partial-offload backoff.
        let capped = ResourceBudget {
            max_vram_bytes: Some(24u64 << 30),
            ..Default::default()
        };
        let s = default_suggester().suggest(&evil, &hw24, &capped);
        assert!(
            s.vram_fit.verdict == FitVerdict::Overflow || s.ram_fit.verdict == FitVerdict::Overflow,
            "corrupt metadata must read won't-fit, not a false 'fits' \
             (vram={:?}, ram={:?}, rationale={:?})",
            s.vram_fit.verdict,
            s.ram_fit.verdict,
            s.rationale
        );
    }
    /// The FitVerdict tooltip help follows the param-help convention: every
    /// variant annotated, ≥2 sentences, real thresholds — the frontend renders
    /// these verbatim (FitVerdictHelp.ts), so this is the wording gate.
    #[test]
    fn fit_verdict_help_covers_every_variant() {
        let keys: Vec<&str> = FitVerdict::PARAM_HELP.iter().map(|(k, _)| *k).collect();
        assert_eq!(keys, ["Fits", "Tight", "Overflow"]);
        for (variant, text) in FitVerdict::PARAM_HELP {
            let sentences =
                text.matches(". ").count() + usize::from(text.trim_end().ends_with('.'));
            assert!(
                sentences >= 2,
                "help for `{variant}` must be ≥2 sentences: {text:?}"
            );
            assert!(
                text.contains('%'),
                "help for `{variant}` cites its real threshold: {text:?}"
            );
        }
    }
}
