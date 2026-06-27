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

pub mod card_sampling;
pub mod derive;
pub mod store;
pub mod vram;

use serde::{Deserialize, Serialize};

use crate::system::HardwareInfo;
use crate::worker::engine::llamacpp::params::{LlamaCppParams, LlamaCppSamplingParams};
use crate::worker::engine::{GpuLayers, LoadParams, SamplingParams};
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
    /// Tri-state memory fit verdict.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    pub enum FitVerdict {
        /// Comfortably within the budget (≤ 80%).
        Fits,
        /// Close to the budget (80–95%) — loads, but little headroom.
        Tight,
        /// Exceeds the budget (> 95%) — will likely OOM or fail to load.
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
        // NOTE: per-field `overrides` (pre-applying user pins during the suggest) are
        // DEFERRED — the user edits the suggestion in the UI and loads with the full
        // engine-tagged `params` on the load request (explicit, end-to-end). Re-adding
        // them needs a proper partial type (all fields `Option`), a later phase.
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
            rationale.push("sampling refined from the model card".into());
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
                vram_fit = self.vram.estimate(&load, meta, hw, budget);
                rationale.push(if n == 0 {
                    "no GPU offload fits the VRAM budget — using CPU-only".into()
                } else {
                    format!("reduced GPU offload to {n} layers to fit the VRAM budget")
                });
            }
        }

        // 6. Final RAM fit + verdict notes.
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
}
