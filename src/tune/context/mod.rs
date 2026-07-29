//! Budget-aware context-length derivation — "the largest context window that fits
//! the resource budget" — the inverse of the forward VRAM/RAM estimator.
//!
//! ## Why a module (not a constant)
//!
//! The old autotune capped context at a flat `8192`, ignoring both the model's
//! trained window and how much memory is actually available. This module derives
//! the context from the budget instead: invert the linear `need(n_ctx)` memory
//! model to get the largest `n_ctx` that fits, then clamp to a sane
//! `MIN_CTX..ctx_train` window.
//!
//! ## Structure (see `DESIGN.md`)
//!
//! - [`ContextEstimator`] / [`Analytical`] ([`analytical`]) — `max_ctx_for_budget`,
//!   which INVERTS the forward VRAM/RAM estimators for the actual load.
//! - [`AverageStrategy`] / [`CtxDerivation`] ([`average`]) — the ensemble that
//!   averages several estimators (analytical only today) and reports the spread.
//!
//! `Analytical` reuses the FORWARD estimators (`vram::StaticVramEstimator` /
//! `StaticRamEstimator`) rather than a re-derived formula, so the inverse prices
//! partial offload / `cpu_moe` / `offload_kqv` exactly as the fit verdict does — a
//! derived context never disagrees with the footprint the UI shows for it.

pub mod analytical;
pub mod average;

pub use analytical::{Analytical, ContextEstimator};
pub use average::{AverageStrategy, CtxDerivation};
