//! Analytical context estimator: linearly INVERTS the forward VRAM/RAM estimators
//! (the single source of truth) for the actual load. See `context/DESIGN.md`.

use crate::system::HardwareInfo;
use crate::tune::vram::{StaticRamEstimator, StaticVramEstimator};
use crate::tune::{ModelMeta, RamEstimator, ResourceBudget, VramEstimator};
use crate::worker::engine::llamacpp::params::LlamaCppParams;
use crate::worker::engine::CtxLen;

/// Derives the largest context that fits a budget by inverting the forward memory
/// model. A trait so additional methods — empirical regression, engine self-report,
/// an offline probe — can be added and AVERAGED ([`super::AverageStrategy`]) without
/// touching the caller; [`Analytical`] is the closed-form default.
pub trait ContextEstimator {
    /// The largest `n_ctx` whose forward footprint fits BOTH budgets for THIS load's
    /// offload configuration. Multi-constraint: the context must fit VRAM (the
    /// GPU-resident KV) AND RAM (the CPU-resident KV + weights). Unclamped (the caller
    /// clamps to a `MIN..ctx_train` window); `u32::MAX` when neither pool grows with
    /// context (degenerate metadata).
    fn max_ctx_for_budget(
        &self,
        load: &LlamaCppParams,
        meta: &ModelMeta,
        hw: &HardwareInfo,
        vram_budget: u64,
        ram_budget: u64,
    ) -> u32;
}

/// Closed-form estimator. The forward footprint of each pool is LINEAR in `n_ctx`
/// (`need(n) = base + slope·n`), so evaluating the forward estimator at `n = 0` and
/// `n = 1` recovers `base` and `slope`, and the inverse is a single division — no
/// search. Reusing the FORWARD estimators (not a re-derived formula) means partial
/// GPU offload, `cpu_moe`, and `offload_kqv` are all priced exactly as the fit
/// verdict prices them, so a derived context never disagrees with its own verdict.
pub struct Analytical;

impl ContextEstimator for Analytical {
    fn max_ctx_for_budget(
        &self,
        load: &LlamaCppParams,
        meta: &ModelMeta,
        hw: &HardwareInfo,
        vram_budget: u64,
        ram_budget: u64,
    ) -> u32 {
        // Footprint of each pool at a given context (uncapped budget → raw needed bytes).
        let uncapped = ResourceBudget::default();
        let footprint = |n: u32| {
            let mut probe = load.clone();
            probe.ctx_len = CtxLen::Fixed { n };
            (
                StaticVramEstimator
                    .estimate(&probe, meta, hw, &uncapped)
                    .needed_bytes,
                StaticRamEstimator
                    .estimate(&probe, meta, hw, &uncapped)
                    .needed_bytes,
            )
        };
        let (vram0, ram0) = footprint(0);
        let (vram1, ram1) = footprint(1);

        // Largest n such that `base + slope·n <= budget`. A zero slope means this pool
        // doesn't grow with context (e.g. an all-GPU load charges no per-token RAM): it
        // imposes no context limit IF its fixed cost fits, but blocks ALL context (0)
        // if the fixed cost ALONE already exceeds the budget — otherwise a tight RAM cap
        // below the fixed overhead would be silently ignored.
        let invert = |budget: u64, base: u64, slope: u64| -> u32 {
            if slope == 0 {
                if base <= budget {
                    u32::MAX
                } else {
                    0
                }
            } else {
                (budget.saturating_sub(base) / slope).min(u32::MAX as u64) as u32
            }
        };
        let max_vram = invert(vram_budget, vram0, vram1.saturating_sub(vram0));
        let max_ram = invert(ram_budget, ram0, ram1.saturating_sub(ram0));
        // The context must fit BOTH pools → the tighter constraint wins.
        max_vram.min(max_ram)
    }
}

#[cfg(test)]
#[path = "analytical_tests.rs"]
mod tests;
