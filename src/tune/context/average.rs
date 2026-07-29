//! The averaging ensemble over [`ContextEstimator`]s — the user-chosen default so
//! no single method dominates. Today it holds only [`Analytical`]; the mechanism is
//! in place to add more methods later without touching the caller.

use crate::system::HardwareInfo;
use crate::tune::ModelMeta;
use crate::worker::engine::llamacpp::params::LlamaCppParams;

use super::{Analytical, ContextEstimator};

/// The averaged budget-derived context plus the per-method spread, for the tune
/// rationale. With a single estimator `min == max == ctx`; the spread becomes
/// informative once more methods are wired.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CtxDerivation {
    /// The averaged largest context that fits the budget.
    pub ctx: u32,
    /// The smallest per-method estimate (the most conservative).
    pub min: u32,
    /// The largest per-method estimate (the most optimistic).
    pub max: u32,
    /// How many estimators contributed (1 today).
    pub methods: u32,
}

/// Combines several [`ContextEstimator`]s by AVERAGING their
/// `max_ctx_for_budget` — the default ensemble. Extend by adding estimators in
/// [`new`](Self::new); the caller is unchanged.
pub struct AverageStrategy {
    estimators: Vec<Box<dyn ContextEstimator + Send + Sync>>,
}

impl AverageStrategy {
    /// The default ensemble: analytical only, for now.
    pub fn analytical_only() -> Self {
        Self {
            estimators: vec![Box::new(Analytical)],
        }
    }

    /// A custom ensemble (for tests / future methods).
    pub fn new(estimators: Vec<Box<dyn ContextEstimator + Send + Sync>>) -> Self {
        Self { estimators }
    }

    /// The averaged largest context that fits both budgets for `load`'s offload
    /// config, with the per-method spread. Empty ensemble → all-zero (caller clamps up).
    pub fn derive_ctx(
        &self,
        load: &LlamaCppParams,
        meta: &ModelMeta,
        hw: &HardwareInfo,
        vram_budget: u64,
        ram_budget: u64,
    ) -> CtxDerivation {
        let vals: Vec<u32> = self
            .estimators
            .iter()
            .map(|e| e.max_ctx_for_budget(load, meta, hw, vram_budget, ram_budget))
            .collect();
        if vals.is_empty() {
            return CtxDerivation {
                ctx: 0,
                min: 0,
                max: 0,
                methods: 0,
            };
        }
        // Average in u128 so a `u32::MAX` method (degenerate KV) can't overflow.
        let sum: u128 = vals.iter().map(|&v| v as u128).sum();
        let avg = (sum / vals.len() as u128).min(u32::MAX as u128) as u32;
        CtxDerivation {
            ctx: avg,
            min: *vals.iter().min().unwrap(),
            max: *vals.iter().max().unwrap(),
            methods: vals.len() as u32,
        }
    }
}

#[cfg(test)]
#[path = "average_tests.rs"]
mod tests;
