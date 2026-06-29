//! Heuristic derivation strategy (`DeriveStrategy` default impl).
//!
//! Static, GGUF/hardware-driven defaults — the planning brain ported from
//! TurboLLM's `profile.ts`, corrected per the adoption study. Pure over its
//! inputs; the MoE back-off + precedence merge live in [`super::Suggester::suggest`].

use crate::system::HardwareInfo;
use crate::worker::engine::llamacpp::params::LlamaCppParams;
use crate::worker::engine::{CtxLen, FlashAttn, GpuLayers, KvCacheKind};

use super::context::{AverageStrategy, CtxDerivation};
use super::{has_gpu, DeriveStrategy, ModelMeta, ResourceBudget};

/// Floor for an auto-derived context window. Even a tight budget keeps at least this
/// many tokens (a model trained on fewer is the only exception — never exceed its
/// trained window). Replaces the old flat 8192 *cap*: context now scales UP to the
/// budget instead of being pinned at a constant.
const MIN_CTX: u32 = 4096;

/// Resolve a budget basis to bytes: an EXPLICIT cap (the app's `max_*_bytes`) is the
/// ceiling — no extra headroom, the user already carved their margin; a DETECTED
/// total keeps the `MEMORY_HEADROOM_FRACTION` (0.8) safety the fit verdict uses.
fn budget_basis(explicit: Option<u64>, detected: u64) -> u64 {
    match explicit {
        Some(cap) => cap,
        None => (detected as f64 * crate::api::MEMORY_HEADROOM_FRACTION) as u64,
    }
}

/// Derive the context window for `load` from the resource budget: the largest window
/// whose footprint fits BOTH the VRAM and RAM budgets, clamped to `MIN_CTX..ctx_train`.
///
/// The derivation is offload-config aware — it inverts the FORWARD estimators for
/// `load` (see [`super::context`]), so an all-GPU load is VRAM-bound, a CPU-only
/// fallback is RAM-bound, and a partial / `cpu_moe` offload is bound by whichever
/// pool fills first. `load.ctx_len` is ignored (the inversion drives it). Returns the
/// clamped context plus the raw [`CtxDerivation`] (the per-method spread, for the
/// rationale).
pub fn derive_ctx(
    load: &LlamaCppParams,
    meta: &ModelMeta,
    hw: &HardwareInfo,
    budget: &ResourceBudget,
) -> (u32, CtxDerivation) {
    // The trained window is the hard upper bound — never derive beyond what the model
    // supports. Missing metadata falls back to MIN_CTX.
    let ctx_train = meta
        .ctx_train
        .map(|c| c.min(u32::MAX as u64) as u32)
        .unwrap_or(MIN_CTX);
    let vram_budget = budget_basis(budget.max_vram_bytes, hw.vram_total_bytes);
    let ram_budget = budget_basis(budget.max_ram_bytes, hw.ram_total_bytes);
    let derivation =
        AverageStrategy::analytical_only().derive_ctx(load, meta, hw, vram_budget, ram_budget);
    // Clamp to a sane window: ≥ MIN_CTX (but never above a small model's trained
    // window) and ≤ ctx_train.
    let lo = MIN_CTX.min(ctx_train);
    let ctx = derivation.ctx.clamp(lo, ctx_train);
    (ctx, derivation)
}

/// The default heuristic strategy.
pub struct HeuristicStrategy;

/// Inference threads when uncapped: `floor(cores / 2)`, at least 1. NOTE this is
/// the *tuner's* heuristic, deliberately distinct from higgs's non-tuned default
/// (`available_parallelism() - 2`); it supersedes that when tuning.
fn auto_threads(hw: &HardwareInfo, budget: &ResourceBudget) -> u32 {
    let half = (hw.cpu_cores / 2).max(1);
    match budget.max_cpu_threads {
        Some(cap) => half.min(cap.max(1)),
        None => half,
    }
}

/// `derive_default` — the base parameter set (DESIGN §6).
pub fn derive_default(
    meta: &ModelMeta,
    hw: &HardwareInfo,
    budget: &ResourceBudget,
) -> LlamaCppParams {
    let threads = auto_threads(hw, budget);
    // Build the load with its offload config first, then derive the context AGAINST it
    // (the context depends on where the KV lands). `ctx_len` starts as a placeholder —
    // `derive_ctx` ignores it and drives the value.
    let mut load = LlamaCppParams {
        ctx_len: CtxLen::Fixed { n: MIN_CTX },
        // All layers on GPU when one is present, else CPU-only.
        gpu_layers: if has_gpu(hw) {
            GpuLayers::All
        } else {
            GpuLayers::Count { n: 0 }
        },
        threads,
        // Split batch threads from generation threads (same value by default).
        n_threads_batch: Some(threads),
        flash_attn: Some(FlashAttn::On),
        type_k: Some(KvCacheKind::F16),
        type_v: Some(KvCacheKind::F16),
        n_seq_max: Some(1),
        ..Default::default()
    };
    let (ctx, _derivation) = derive_ctx(&load, meta, hw, budget);
    load.ctx_len = CtxLen::Fixed { n: ctx };
    load
}

impl DeriveStrategy for HeuristicStrategy {
    fn derive(
        &self,
        meta: &ModelMeta,
        hw: &HardwareInfo,
        budget: &ResourceBudget,
    ) -> LlamaCppParams {
        derive_default(meta, hw, budget)
    }
}

#[cfg(test)]
#[path = "derive_tests.rs"]
mod tests;
