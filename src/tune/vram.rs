//! VRAM / RAM fit estimators (`VramEstimator` / `RamEstimator` default impls).
//!
//! Pure static math (no model load). The KV-cache term uses the **GQA KV head
//! count** (`head_count_kv`), NOT the query `head_count` — using the latter
//! over-estimates KV by the GQA factor (often 4–8×) and would wreck the verdict.
//! The fits/tight/overflow tri-state is computed here; `system::fits_vram` is the
//! reused `≤ budget` boolean primitive (the 80% headroom tier), not a re-port.

use crate::system::{fits_vram, HardwareInfo};
use crate::worker::engine::llamacpp::params::LlamaCppParams;
use crate::worker::engine::{CtxLen, GpuLayers, KvCacheKind};

use super::{
    has_gpu, FitReport, FitVerdict, ModelMeta, RamEstimator, ResourceBudget, VramEstimator,
};

/// Flat compute/overhead buffer added to every resident estimate (~the KV
/// scratch + compute graph). A coarse constant; the estimate is a planning aid.
const COMPUTE_OVERHEAD_BYTES: u64 = 800 * 1024 * 1024;

/// Fraction of an MoE model's weights that live in the expert tensors — what
/// `cpu_moe` moves to host RAM. A rough planning constant (experts dominate MoE
/// weight; e.g. ~95% on large MoEs, less on small ones).
const EXPERT_FRACTION: f64 = 0.75;

/// The `system::fits_vram` headroom fraction reused for the "Fits" tier.
const FIT_HEADROOM: f64 = crate::api::MEMORY_HEADROOM_FRACTION;

/// The "Tight" upper bound (fraction of the budget basis).
const TIGHT_CEILING: f64 = 0.95;

/// "Fits" headroom fraction. An EXPLICIT budget (`max_*_bytes` set) is the user's
/// hard ceiling — they already carved their margin (e.g. 75% of FREE) — so we do
/// NOT re-apply the 80% safety on top (no `0.75 × 0.8` double-discount): the budget
/// is filled directly (`1.0`). The DETECTED total keeps the 80% safety tier.
fn fit_frac(explicit_budget: bool) -> f64 {
    if explicit_budget {
        1.0
    } else {
        FIT_HEADROOM
    }
}

/// "Tight" upper bound; collapses to the ceiling (`1.0`) for an explicit budget,
/// so an explicit budget reads Fits up to its ceiling and Overflow beyond it.
fn tight_frac(explicit_budget: bool) -> f64 {
    if explicit_budget {
        1.0
    } else {
        TIGHT_CEILING
    }
}

/// Resolve a candidate context window for ESTIMATION the same way the load path
/// resolves it: `Auto` becomes the model's trained window **capped at**
/// [`crate::api::DEFAULT_CTX_CAP`] (the node's auto-context cap), so a live estimate
/// matches what would actually load. Estimating a long-context model's full
/// `ctx_train` (e.g. 1M) would otherwise show a huge KV / false Overflow for a load
/// that the node will cap. A `Fixed` request passes through untouched.
pub fn resolve_estimate_ctx(ctx_len: CtxLen, ctx_train: Option<u64>) -> CtxLen {
    match ctx_len {
        CtxLen::Auto => {
            let cap = crate::api::DEFAULT_CTX_CAP as u64;
            // No trained-context metadata → the load leaves `ctx_len` absent and the
            // worker falls back to 4096 (llama.cpp's default n_ctx). Estimate against
            // THAT, not the cap, or a metadata-less model overstates KV by up to 8×.
            let trained = ctx_train.unwrap_or(WORKER_FALLBACK_CTX);
            CtxLen::Fixed {
                n: trained.min(cap) as u32,
            }
        }
        other => other,
    }
}

/// llama.cpp's default `n_ctx` when a load leaves the context unset and the GGUF
/// carries no trained-context metadata — the worker's fallback window.
const WORKER_FALLBACK_CTX: u64 = 4096;

/// The turbotune benchmark's per-candidate fit verdict, used to filter the
/// candidate set BEFORE any load (§ `pinned_bench_candidates`). RAM-overflow
/// takes precedence — a half-GPU-layers candidate can pass VRAM yet overflow an
/// explicit RAM cap — else the VRAM verdict drives the fit + headroom gate.
///
/// The candidate's `Auto` context is normalized to the node's load-time cap
/// FIRST ([`resolve_estimate_ctx`]) so a pinned (or seed) `Auto` on a
/// long-context model is estimated against the window that will ACTUALLY load
/// (`DEFAULT_CTX_CAP`), not the full `ctx_train` that [`effective_ctx`] would use
/// for `Auto` — which would falsely read Overflow and DROP a loadable candidate.
pub fn bench_fit(
    lc: &LlamaCppParams,
    meta: &ModelMeta,
    hw: &HardwareInfo,
    budget: &ResourceBudget,
) -> FitReport {
    let mut lc = lc.clone();
    lc.ctx_len = resolve_estimate_ctx(lc.ctx_len, meta.ctx_train);
    let ram = StaticRamEstimator.estimate(&lc, meta, hw, budget);
    if ram.verdict == FitVerdict::Overflow {
        return ram;
    }
    StaticVramEstimator.estimate(&lc, meta, hw, budget)
}

/// The benchmarked config's separate VRAM and RAM fit reports for the tune
/// RESPONSE (returned as `(vram, ram)`). Like [`bench_fit`], an `Auto` context is
/// normalized to the node's load-time cap FIRST, so the reported fit describes
/// the window that will actually load — not the full `ctx_train` that
/// [`effective_ctx`] uses for `Auto`, which would report a false Overflow for a
/// benchmarked that passed the filter and loaded fine. The persisted benchmarked keeps its
/// verbatim `Auto`; only these reported numbers use the normalized window.
pub fn benchmarked_fit_reports(
    benchmarked: &LlamaCppParams,
    meta: &ModelMeta,
    hw: &HardwareInfo,
    budget: &ResourceBudget,
) -> (FitReport, FitReport) {
    let mut lc = benchmarked.clone();
    lc.ctx_len = resolve_estimate_ctx(lc.ctx_len, meta.ctx_train);
    (
        StaticVramEstimator.estimate(&lc, meta, hw, budget),
        StaticRamEstimator.estimate(&lc, meta, hw, budget),
    )
}

/// Bytes-per-element for a KV-cache element type (block-quant overhead folded in
/// approximately). Defaults to F16 (2 bytes) for the practical KV set.
fn kv_type_bytes(kind: KvCacheKind) -> f64 {
    match kind {
        KvCacheKind::F32 => 4.0,
        KvCacheKind::F16 => 2.0,
        KvCacheKind::Q8_0 => 1.0,
        KvCacheKind::Q5_1 | KvCacheKind::Q5_0 => 0.6875,
        KvCacheKind::Q4_1 | KvCacheKind::Q4_0 => 0.5625,
    }
}

/// Effective context length for the KV estimate.
fn effective_ctx(load: &LlamaCppParams, meta: &ModelMeta) -> u64 {
    match load.ctx_len {
        CtxLen::Fixed { n } => n as u64,
        CtxLen::Auto => meta.ctx_train.unwrap_or(4096),
    }
}

/// Total KV-cache bytes for the full context (K + V across all layers), GQA-correct.
fn kv_cache_bytes(load: &LlamaCppParams, meta: &ModelMeta) -> u64 {
    let blocks = meta.block_count.unwrap_or(32) as u64;
    let ctx = effective_ctx(load, meta);
    let kv_heads = meta.kv_heads() as u64;
    let head_dim = meta.head_dim() as u64;
    let k_bytes = kv_type_bytes(load.type_k.unwrap_or(KvCacheKind::F16));
    let v_bytes = kv_type_bytes(load.type_v.unwrap_or(KvCacheKind::F16));
    // Saturating: a corrupt/hostile GGUF header (absurd block_count/ctx/heads) must not
    // overflow the u64 product (debug panic / release garbage) — saturate so the model
    // is judged "won't fit" rather than crashing the tune.
    let elems = blocks
        .saturating_mul(ctx)
        .saturating_mul(kv_heads)
        .saturating_mul(head_dim) as f64;
    (elems * (k_bytes + v_bytes)) as u64
}

/// Fraction of layers offloaded to the GPU (0.0 when CPU-only / no GPU).
fn gpu_fraction(load: &LlamaCppParams, meta: &ModelMeta, hw: &HardwareInfo) -> f64 {
    if !has_gpu(hw) {
        return 0.0;
    }
    let blocks = meta.block_count.unwrap_or(32) as u64;
    match load.gpu_layers {
        GpuLayers::All => 1.0,
        GpuLayers::Count { n: 0 } => 0.0,
        GpuLayers::Count { n } if n as u64 >= blocks => 1.0,
        GpuLayers::Count { n } => (n as f64 / blocks as f64).clamp(0.0, 1.0),
    }
}

/// Build a tri-state [`FitReport`] from a need against a budget basis. `fit_f` is
/// the "Fits" headroom fraction and `tight_f` the "Tight" upper bound (see
/// [`fit_frac`]/[`tight_frac`]): a detected total uses `0.8`/`0.95`; an explicit
/// budget uses `1.0`/`1.0` (the budget IS the ceiling — no double headroom).
fn report(needed: u64, basis: u64, fit_f: f64, tight_f: f64) -> FitReport {
    // A zero need always fits — notably a CPU-only load uses no VRAM (`needed == 0`)
    // even on a host with no GPU (`basis == 0`), which must read as Fits, not Overflow.
    if needed == 0 {
        return FitReport {
            verdict: FitVerdict::Fits,
            needed_bytes: 0,
            budget_bytes: basis,
        };
    }
    if basis == 0 {
        return FitReport {
            verdict: FitVerdict::Overflow,
            needed_bytes: needed,
            budget_bytes: 0,
        };
    }
    let verdict = if fits_vram(needed, basis, fit_f).fits {
        FitVerdict::Fits
    } else if (needed as f64) <= basis as f64 * tight_f {
        FitVerdict::Tight
    } else {
        FitVerdict::Overflow
    };
    FitReport {
        verdict,
        needed_bytes: needed,
        budget_bytes: basis,
    }
}

/// The largest `gpu_layers` value whose VRAM need fits the (`max_vram` or detected)
/// budget at the same 80% headroom the fit verdict uses — for backing GPU offload
/// DOWN to honor a VRAM cap. `u32::MAX` ("all") when every layer fits; `0`
/// (CPU-only) when none do. Mirrors `estimate`'s need model so the result fits.
pub fn gpu_layers_within_budget(
    load: &LlamaCppParams,
    meta: &ModelMeta,
    hw: &HardwareInfo,
    budget: &ResourceBudget,
) -> u32 {
    if !has_gpu(hw) {
        return 0;
    }
    let basis = budget.max_vram_bytes.unwrap_or(hw.vram_total_bytes);
    // An explicit VRAM budget is the ceiling (no extra 0.8); the detected total keeps it.
    let safe = basis as f64 * fit_frac(budget.max_vram_bytes.is_some());
    let overhead = COMPUTE_OVERHEAD_BYTES as f64;
    // Weights resident on the GPU at full offload (cpu_moe pulls experts off).
    let weights = meta.size_bytes as f64
        * if load.cpu_moe == Some(true) && meta.is_moe() {
            1.0 - EXPERT_FRACTION
        } else {
            1.0
        };
    let kv = if load.offload_kqv == Some(false) {
        0.0
    } else {
        kv_cache_bytes(load, meta) as f64
    };
    // needed(frac) = (weights + kv) * frac + overhead (frac = gpu_layers / blocks).
    let per_frac = weights + kv;
    if per_frac <= 0.0 {
        return u32::MAX;
    }
    let max_frac = (safe - overhead) / per_frac;
    if max_frac <= 0.0 {
        return 0;
    }
    if max_frac >= 1.0 {
        return u32::MAX;
    }
    let blocks = meta.block_count.unwrap_or(32) as f64;
    (max_frac * blocks).floor().max(0.0) as u32
}

/// Default GPU-VRAM estimator.
pub struct StaticVramEstimator;

impl VramEstimator for StaticVramEstimator {
    fn estimate(
        &self,
        load: &LlamaCppParams,
        meta: &ModelMeta,
        hw: &HardwareInfo,
        budget: &ResourceBudget,
    ) -> FitReport {
        let frac = gpu_fraction(load, meta, hw);
        let weights = meta.size_bytes as f64;
        // Weights resident on GPU; cpu_moe pulls the expert fraction off the GPU.
        let mut gpu_weights = weights * frac;
        if load.cpu_moe == Some(true) && meta.is_moe() {
            gpu_weights *= 1.0 - EXPERT_FRACTION;
        }
        // KV is on the GPU for offloaded layers unless offload_kqv was disabled.
        let kv_on_gpu = if load.offload_kqv == Some(false) {
            0.0
        } else {
            kv_cache_bytes(load, meta) as f64 * frac
        };
        let overhead = if frac > 0.0 {
            COMPUTE_OVERHEAD_BYTES as f64
        } else {
            0.0
        };
        let needed = (gpu_weights + kv_on_gpu + overhead) as u64;
        let basis = budget.max_vram_bytes.unwrap_or(hw.vram_total_bytes);
        let explicit = budget.max_vram_bytes.is_some();
        report(needed, basis, fit_frac(explicit), tight_frac(explicit))
    }
}

/// Default host-RAM estimator (the CPU-resident weights + CPU KV + cpu_moe experts).
pub struct StaticRamEstimator;

impl RamEstimator for StaticRamEstimator {
    fn estimate(
        &self,
        load: &LlamaCppParams,
        meta: &ModelMeta,
        hw: &HardwareInfo,
        budget: &ResourceBudget,
    ) -> FitReport {
        let frac = gpu_fraction(load, meta, hw);
        let weights = meta.size_bytes as f64;
        // Weights not offloaded to GPU stay in RAM …
        let mut cpu_weights = weights * (1.0 - frac);
        // … plus the experts cpu_moe pulls back off the GPU.
        if load.cpu_moe == Some(true) && meta.is_moe() {
            cpu_weights += weights * frac * EXPERT_FRACTION;
        }
        let kv_total = kv_cache_bytes(load, meta) as f64;
        let kv_on_gpu = if load.offload_kqv == Some(false) {
            0.0
        } else {
            kv_total * frac
        };
        let cpu_kv = (kv_total - kv_on_gpu).max(0.0);
        let needed = (cpu_weights + cpu_kv + COMPUTE_OVERHEAD_BYTES as f64) as u64;
        let basis = budget.max_ram_bytes.unwrap_or(hw.ram_total_bytes);
        let explicit = budget.max_ram_bytes.is_some();
        report(needed, basis, fit_frac(explicit), tight_frac(explicit))
    }
}

#[cfg(test)]
#[path = "vram_tests.rs"]
mod tests;
