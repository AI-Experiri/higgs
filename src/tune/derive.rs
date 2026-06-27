//! Heuristic derivation strategy (`DeriveStrategy` default impl).
//!
//! Static, GGUF/hardware-driven defaults — the planning brain ported from
//! TurboLLM's `profile.ts`, corrected per the adoption study. Pure over its
//! inputs; the MoE back-off + precedence merge live in [`super::Suggester::suggest`].

use crate::system::HardwareInfo;
use crate::worker::engine::llamacpp::params::LlamaCppParams;
use crate::worker::engine::{FlashAttn, GpuLayers, KvCacheKind};

use super::{has_gpu, DeriveStrategy, ModelMeta, ResourceBudget};

/// Default context-window cap for an auto-derived load — a huge-context model is
/// capped so it doesn't allocate an enormous KV cache by default.
const CTX_CAP: u32 = 8192;

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
    let ctx_train = meta.ctx_train.unwrap_or(CTX_CAP as u64);
    let ctx = (ctx_train.min(CTX_CAP as u64)).max(1) as u32;
    let threads = auto_threads(hw, budget);
    LlamaCppParams {
        ctx_len: ctx,
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
    }
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
