//! Turbotune (G6): the MEASURED autotune. Beside the untouched analytical
//! `Suggest` path, `TuneMode::Benchmark` actually LOADS candidate configs and
//! measures token throughput, then saves the fastest as a `TuneProvenance::Bench`
//! profile — the "TurboLLM" approach.
//!
//! This module is the PURE decision core (candidate generation, the absolute
//! headroom gate, winner selection, aggregate-failure diagnosis) so it is
//! exhaustively unit-tested; the slow async orchestration (settle → load →
//! measure → unload, with cancellation and the log-fault watchdog) lives in
//! [`api`](crate::api) and drives these decisions.
//!
//! ## Two-phase search (mechanisms #1 + #3)
//!
//! Phase 1 is CHEAP: from the analytical suggestion (the seed), derive an
//! ordered set of candidate configs — fastest-likely first — and keep only
//! those that pass the analytical fit AND an ABSOLUTE ≥1 GiB VRAM headroom gate
//! (mechanism #2). When the seed's F16 KV cache overflows, quantized-KV variants
//! (q8_0, q4_0) are the calibration rungs (mechanism #3). Phase 2 benches the
//! survivors (bounded) and picks the measured winner (mechanism #8).

use crate::tune::{FitReport, FitVerdict};
use crate::worker::engine::llamacpp::params::LlamaCppParams;
use crate::worker::engine::{GpuLayers, KvCacheKind};

/// Absolute VRAM headroom a benched candidate must leave free (mechanism #2):
/// the fractional 80% fit is not enough on its own — a candidate that "fits" a
/// large budget can still leave too little slack for the compute graph / OS, so
/// the winner must also clear a hard 1 GiB floor. Skipped when there is no VRAM
/// budget basis (CPU-only machines).
pub const ABS_VRAM_HEADROOM_BYTES: u64 = 1 << 30;

/// The most candidates Phase 2 will actually load-and-measure. Each bench costs
/// a full load + a real decode, so the search stays SMALL — Phase 1's ordering
/// puts the fastest-likely config first, so the bounded set still finds the win.
pub const MAX_BENCHED_CANDIDATES: usize = 3;

/// One candidate config to bench, with the phrase that names it in events/logs.
#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    /// The load params to test-load and measure.
    pub load: LlamaCppParams,
    /// Human label for the bench event / rationale (e.g. "q8_0 KV cache").
    pub label: &'static str,
}

/// Whether a fit report clears BOTH the tri-state verdict (not `Overflow`) AND
/// the absolute headroom floor. `budget_bytes == 0` (no VRAM basis) skips the
/// absolute floor — a CPU-only candidate is gated by the RAM estimate elsewhere.
pub fn passes_headroom(fit: &FitReport, abs_floor: u64) -> bool {
    if fit.verdict == FitVerdict::Overflow {
        return false;
    }
    if fit.budget_bytes == 0 {
        return true; // no VRAM basis — nothing to floor
    }
    fit.budget_bytes.saturating_sub(fit.needed_bytes) >= abs_floor
}

/// Phase 1: build the ordered, fit-and-headroom-filtered candidate set for a
/// benchmark run, fastest-likely first. `seed` is the analytical suggestion's
/// load params; `layer_count` is the model's transformer block count (turns an
/// `All` offload into a concrete half); `estimate_vram` returns the VRAM
/// [`FitReport`] for a candidate (injected so this stays pure — the caller
/// passes the real `estimate_footprint`).
///
/// Ordering (fastest → slowest): full-GPU F16 KV → q8_0 KV → q4_0 KV → half
/// GPU layers. Only survivors of [`passes_headroom`] are kept, and at most
/// [`MAX_BENCHED_CANDIDATES`] are returned.
pub fn bench_candidates(
    seed: &LlamaCppParams,
    layer_count: Option<u32>,
    estimate_vram: impl Fn(&LlamaCppParams) -> FitReport,
) -> Vec<Candidate> {
    let mut out = Vec::new();

    // Fastest first: the seed as-is (F16 KV, seed offload).
    let mut variants: Vec<Candidate> = vec![Candidate {
        load: seed.clone(),
        label: "seed (F16 KV cache)",
    }];

    // KV-quant calibration rungs — cheaper VRAM, minor quality cost.
    for (kind, label) in [
        (KvCacheKind::Q8_0, "q8_0 KV cache"),
        (KvCacheKind::Q4_0, "q4_0 KV cache"),
    ] {
        let mut lc = seed.clone();
        lc.type_k = Some(kind);
        lc.type_v = Some(kind);
        // A quantized V cache REQUIRES flash attention in llama.cpp; leave
        // flash_attn as the engine default (auto) so the worker enables it.
        variants.push(Candidate { load: lc, label });
    }

    // Half the GPU layers — the biggest VRAM lever, slowest of the set.
    if let Some(half) = half_gpu_layers(seed, layer_count) {
        variants.push(Candidate {
            load: half,
            label: "half GPU layers on CPU",
        });
    }

    // Keep only the candidates that pass fit + absolute headroom, in order.
    for c in variants {
        if passes_headroom(&estimate_vram(&c.load), ABS_VRAM_HEADROOM_BYTES) {
            out.push(c);
            if out.len() >= MAX_BENCHED_CANDIDATES {
                break;
            }
        }
    }
    out
}

/// A `LlamaCppParams` with GPU layers halved, or `None` when there is nothing
/// meaningful to halve. `All` uses `layer_count` when known; unknown `All`
/// yields `None` (the KV-quant rungs already cover it — a blind all-CPU bench
/// would just be slow).
fn half_gpu_layers(seed: &LlamaCppParams, layer_count: Option<u32>) -> Option<LlamaCppParams> {
    let n = match seed.gpu_layers {
        GpuLayers::Count { n } if n >= 2 => n / 2,
        GpuLayers::All => match layer_count {
            Some(total) if total >= 2 => total / 2,
            _ => return None,
        },
        _ => return None,
    };
    let mut lc = seed.clone();
    lc.gpu_layers = GpuLayers::Count { n };
    Some(lc)
}

/// Pick the winning benched candidate: the highest generation throughput
/// (`gen_tps`). Returns the index into `results`, or `None` when empty. Ties
/// keep the EARLIER (faster-ordered / higher-quality-KV) candidate.
pub fn pick_winner(results: &[(Candidate, crate::tune::BenchResult)]) -> Option<usize> {
    // Fold keeping STRICTLY-greater, so a tie keeps the EARLIER candidate
    // (candidates are ordered fastest-likely / highest-KV-quality first).
    // `Iterator::max_by` would keep the LAST tie instead.
    let mut best: Option<usize> = None;
    for (i, (_, r)) in results.iter().enumerate() {
        let better = match best {
            None => true,
            Some(b) => r.gen_tps > results[b].1.gen_tps,
        };
        if better {
            best = Some(i);
        }
    }
    best
}

/// Aggregate-failure diagnosis (mechanism #9): a single human line naming EACH
/// candidate that was tried and why it failed, for the `[HG063]` error when the
/// whole benchmark exhausts without a single successful measurement.
pub fn aggregate_failure(attempts: &[(&'static str, String)]) -> String {
    if attempts.is_empty() {
        return "no candidate configs fit the resource budget for a benchmark".to_owned();
    }
    let per = attempts
        .iter()
        .map(|(label, why)| format!("{label}: {why}"))
        .collect::<Vec<_>>()
        .join("; ");
    format!("every benchmark candidate failed ({per})")
}

/// Generation (decode) throughput in tokens/sec, EXCLUDING prompt prefill. The
/// first token arrives at `ttft` (the end of prefill), so the decode window is
/// `total - ttft`, and the tokens produced WITHIN it are `completion_tokens - 1`
/// (llama.cpp's `eval time … N runs` convention: the prompt eval yields the first
/// token, decode yields the rest).
///
/// Excluding prefill is what keeps [`pick_winner`] honest: it ranks candidates by
/// `gen_tps`, and a config with a slow prompt-processing stage must NOT be scored
/// as a slow GENERATOR — only the decode rate should decide the winner. Measuring
/// `completion_tokens / total` instead would fold prefill/TTFT into the number and
/// pick a profile by end-to-end latency, not throughput. Returns 0 when nothing was
/// decoded (≤1 token) or the decode window is empty.
pub fn bench_gen_tps(
    completion_tokens: u32,
    ttft: std::time::Duration,
    total: std::time::Duration,
) -> f32 {
    let decode_tokens = completion_tokens.saturating_sub(1);
    if decode_tokens == 0 {
        return 0.0;
    }
    // An EMPTY decode window (total <= ttft — no streamed deltas, or first and
    // final timestamps in the same clock tick) is a FAILED measurement, not an
    // infinitely fast one: dividing by an epsilon here would hand the candidate an
    // absurd tok/s that wins the benchmark (codex r17). 0 keeps it from winning.
    let gen_secs = total.saturating_sub(ttft).as_secs_f32();
    if gen_secs <= 0.0 {
        return 0.0;
    }
    decode_tokens as f32 / gen_secs
}

#[cfg(test)]
#[path = "bench_tests.rs"]
mod tests;
