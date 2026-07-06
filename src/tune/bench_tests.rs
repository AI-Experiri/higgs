use super::*;

/// Existing-behavior call shape: no pins.
fn bench_candidates_nopins(
    seed: &LlamaCppParams,
    layer_count: Option<u32>,
    estimate_vram: impl Fn(&LlamaCppParams) -> FitReport,
) -> Vec<Candidate> {
    bench_candidates(seed, layer_count, &TunePins::default(), estimate_vram)
}

use crate::tune::BenchResult;
use crate::worker::engine::llamacpp::params::LlamaCppParams;
use crate::worker::engine::{CtxLen, GpuLayers, KvCacheKind};

fn seed(gpu_layers: GpuLayers) -> LlamaCppParams {
    LlamaCppParams {
        ctx_len: CtxLen::Fixed { n: 4096 },
        gpu_layers,
        ..Default::default()
    }
}

fn fit(verdict: FitVerdict, needed: u64, budget: u64) -> FitReport {
    FitReport {
        verdict,
        needed_bytes: needed,
        budget_bytes: budget,
    }
}

const GIB: u64 = 1 << 30;

/// The headroom gate rejects Overflow and anything under the absolute floor,
/// but accepts a comfortable fit and a no-VRAM-basis (CPU-only) candidate.
#[test]
fn headroom_gate() {
    assert!(
        passes_headroom(&fit(FitVerdict::Fits, 4 * GIB, 8 * GIB), GIB),
        "4 GiB slack"
    );
    assert!(
        !passes_headroom(&fit(FitVerdict::Overflow, 4 * GIB, 8 * GIB), GIB),
        "overflow rejected"
    );
    assert!(
        !passes_headroom(&fit(FitVerdict::Tight, 7 * GIB + 900_000_000, 8 * GIB), GIB),
        "under the 1 GiB floor rejected"
    );
    assert!(
        passes_headroom(&fit(FitVerdict::Tight, 7 * GIB, 8 * GIB), GIB),
        "exactly 1 GiB ok"
    );
    assert!(
        passes_headroom(&fit(FitVerdict::Fits, 0, 0), GIB),
        "no VRAM basis skips the floor"
    );
}

/// When the seed F16 KV fits with headroom, it's the fastest candidate and
/// leads the ordered set; quant/layer rungs follow (bounded to MAX).
#[test]
fn candidates_fastest_first_when_seed_fits() {
    // Everything fits comfortably.
    let cands = bench_candidates_nopins(&seed(GpuLayers::Count { n: 32 }), Some(32), |_| {
        fit(FitVerdict::Fits, 2 * GIB, 8 * GIB)
    });
    assert_eq!(cands.len(), MAX_BENCHED_CANDIDATES, "bounded to the cap");
    assert_eq!(
        cands[0].label, "seed (F16 KV cache)",
        "seed is fastest-first"
    );
    // The seed keeps F16 (no type_k override).
    assert_eq!(cands[0].load.type_k, None);
}

/// When F16 KV overflows but quantized KV fits, the seed is DROPPED and the
/// q8_0/q4_0 calibration rungs carry the run (mechanism #3).
#[test]
fn kv_quant_rungs_when_f16_overflows() {
    let cands = bench_candidates_nopins(&seed(GpuLayers::All), Some(40), |lc| {
        // F16 (no type override) overflows; any quantized KV fits.
        if lc.type_k.is_none() {
            fit(FitVerdict::Overflow, 10 * GIB, 8 * GIB)
        } else {
            fit(FitVerdict::Fits, 3 * GIB, 8 * GIB)
        }
    });
    assert!(
        !cands.iter().any(|c| c.label.contains("seed")),
        "F16 seed dropped"
    );
    assert_eq!(cands[0].label, "q8_0 KV cache");
    assert_eq!(cands[0].load.type_k, Some(KvCacheKind::Q8_0));
    assert_eq!(cands[0].load.type_v, Some(KvCacheKind::Q8_0));
}

/// A fixed GPU-layer count contributes a half-layers rung; `All` uses the
/// model's block count to halve it.
#[test]
fn half_layers_rung_present_for_count_and_known_all() {
    // Only the half-layers config fits, to isolate it.
    let only_half = |half_n: u32| {
        move |lc: &LlamaCppParams| {
            if lc.gpu_layers == (GpuLayers::Count { n: half_n }) {
                fit(FitVerdict::Fits, 2 * GIB, 8 * GIB)
            } else {
                fit(FitVerdict::Overflow, 10 * GIB, 8 * GIB)
            }
        }
    };
    let c = bench_candidates_nopins(&seed(GpuLayers::Count { n: 40 }), None, only_half(20));
    assert_eq!(c.len(), 1);
    assert_eq!(c[0].load.gpu_layers, GpuLayers::Count { n: 20 });
    assert_eq!(c[0].label, "half GPU layers on CPU");

    let c = bench_candidates_nopins(&seed(GpuLayers::All), Some(48), only_half(24));
    assert_eq!(c.len(), 1);
    assert_eq!(c[0].load.gpu_layers, GpuLayers::Count { n: 24 });
}

/// Nothing fits → an empty candidate set (the caller surfaces the aggregate
/// "no candidate fits" diagnosis).
#[test]
fn no_candidates_when_all_overflow() {
    let cands = bench_candidates_nopins(&seed(GpuLayers::All), None, |_| {
        fit(FitVerdict::Overflow, 20 * GIB, 8 * GIB)
    });
    assert!(cands.is_empty());
}

/// The benchmarked is the highest gen_tps; a tie keeps the earlier candidate.
#[test]
fn benchmarked_is_max_gen_tps() {
    let c = |label: &'static str| Candidate {
        load: LlamaCppParams::default(),
        label,
    };
    let r = |tps: f32| BenchResult {
        gen_tps: tps,
        ..Default::default()
    };
    let results = vec![(c("a"), r(12.0)), (c("b"), r(30.0)), (c("c"), r(30.0))];
    assert_eq!(
        pick_benchmarked(&results),
        Some(1),
        "b wins on tps; tie keeps earlier"
    );
    assert_eq!(pick_benchmarked(&[]), None, "empty → no benchmarked config");
}

/// The aggregate-failure line names every tried candidate + reason.
#[test]
fn aggregate_failure_names_each_candidate() {
    let s = aggregate_failure(&[
        ("seed (F16 KV cache)", "out of memory".into()),
        ("q8_0 KV cache", "worker died".into()),
    ]);
    assert!(s.contains("seed (F16 KV cache): out of memory"));
    assert!(s.contains("q8_0 KV cache: worker died"));
    // No candidates fit at all → the distinct "no candidate fits" phrasing.
    assert!(aggregate_failure(&[]).contains("no candidate configs fit"));
}

/// `bench_gen_tps` is DECODE-ONLY: prefill/TTFT is excluded, so a slow prompt
/// stage never skews the throughput `pick_benchmarked` ranks on. Fail-on-revert:
/// computing `completion_tokens / total` instead (prefill folded in) collapses
/// the 126 tok/s decode rate to ~42.7 and fails these bounds.
#[test]
fn bench_gen_tps_excludes_prompt_prefill() {
    use std::time::Duration;
    // 64 tokens; first token after 1000ms prefill; 1500ms total → 500ms decode.
    // Decode-only: (64-1) / 0.5s = 126 tok/s.  Prefill-included would be 64/1.5 ≈ 42.7.
    let tps = bench_gen_tps(64, Duration::from_millis(1000), Duration::from_millis(1500));
    assert!(
        (tps - 126.0).abs() < 0.5,
        "decode-only tok/s (prefill excluded): {tps}"
    );
    assert!(
        tps > 100.0,
        "must divide by the 500ms decode window, not the 1.5s total: {tps}"
    );
    // Degenerate: ≤1 token has no decode window → 0 (no divide-by-tiny blowup).
    assert_eq!(
        bench_gen_tps(1, Duration::from_millis(10), Duration::from_millis(20)),
        0.0
    );
    assert_eq!(
        bench_gen_tps(0, Duration::from_millis(10), Duration::from_millis(20)),
        0.0
    );
    // A longer-than-total prefill (clock skew / no streamed deltas) is an EMPTY
    // decode window — a FAILED measurement scored 0, NEVER an epsilon-divided
    // absurd tok/s that would win the benchmark (codex r17). Fail-on-revert:
    // restore `.max(f32::EPSILON)` and this becomes ~5.3e39.
    assert_eq!(
        bench_gen_tps(64, Duration::from_millis(2000), Duration::from_millis(1500)),
        0.0,
        "empty decode window scores 0"
    );
    assert_eq!(
        bench_gen_tps(64, Duration::from_millis(1500), Duration::from_millis(1500)),
        0.0,
        "same-tick timestamps score 0"
    );
}

// ── Turbotune pins (JD1) ─────────────────────────────────────────────────

/// A pinned KV cache type suppresses BOTH KV-quant rungs; a pinned GPU
/// offload suppresses the half-layers rung — turbotune must never measure a
/// candidate that overrides a user pin.
#[test]
fn pins_suppress_the_rungs_that_would_override_them() {
    let fits = |_: &LlamaCppParams| FitReport {
        verdict: FitVerdict::Fits,
        needed_bytes: 1,
        budget_bytes: 100 << 30,
    };
    // KV pin (type_k alone is enough): only the seed + half-layers survive.
    let pins = TunePins {
        type_k: Some(KvCacheKind::Q8_0),
        ..TunePins::default()
    };
    let c = bench_candidates(&seed(GpuLayers::Count { n: 32 }), Some(32), &pins, fits);
    assert!(
        c.iter()
            .all(|c| !c.label.contains("KV cache") || c.label.contains("seed")),
        "KV rungs must be suppressed: {:?}",
        c.iter().map(|c| c.label).collect::<Vec<_>>()
    );
    // GPU-layers pin: no half-layers rung.
    let pins = TunePins {
        gpu_layers: Some(GpuLayers::Count { n: 32 }),
        ..TunePins::default()
    };
    let c = bench_candidates(&seed(GpuLayers::Count { n: 32 }), Some(32), &pins, fits);
    assert!(
        c.iter().all(|c| !c.label.contains("half GPU layers")),
        "half-layers rung must be suppressed: {:?}",
        c.iter().map(|c| c.label).collect::<Vec<_>>()
    );
}

/// `apply_pins` overwrites exactly the pinned fields and leaves the rest.
#[test]
fn apply_pins_overwrites_only_the_pinned_fields() {
    let mut lc = seed(GpuLayers::All);
    let before_threads = lc.threads;
    apply_pins(
        &mut lc,
        &TunePins {
            ctx_len: Some(CtxLen::Fixed { n: 4096 }),
            gpu_layers: Some(GpuLayers::Count { n: 8 }),
            type_k: Some(KvCacheKind::Q4_0),
            type_v: Some(KvCacheKind::Q4_0),
        },
    );
    assert_eq!(lc.ctx_len, CtxLen::Fixed { n: 4096 });
    assert_eq!(lc.gpu_layers, GpuLayers::Count { n: 8 });
    assert_eq!(lc.type_k, Some(KvCacheKind::Q4_0));
    assert_eq!(lc.type_v, Some(KvCacheKind::Q4_0));
    assert_eq!(lc.threads, before_threads, "unpinned fields untouched");

    // No pins → no changes.
    let mut lc2 = seed(GpuLayers::All);
    apply_pins(&mut lc2, &TunePins::default());
    assert_eq!(lc2, seed(GpuLayers::All));
}

// ── pinned_bench_candidates seam + label honesty (round-1 findings) ─────

/// The seam api.rs uses: pins must reach BOTH the returned seed and every
/// candidate. Fail-on-revert guard for the api.rs pins wiring — dropping the
/// apply_pins step (or the pins arg) makes the pinned ctx/offload not stick.
#[test]
fn pinned_bench_candidates_applies_pins_to_every_candidate() {
    let fits = |_: &LlamaCppParams| fit(FitVerdict::Fits, 1, 100 << 30);
    let pins = TunePins {
        ctx_len: Some(CtxLen::Fixed { n: 8192 }),
        gpu_layers: Some(GpuLayers::Count { n: 20 }),
        type_k: Some(KvCacheKind::Q8_0),
        type_v: Some(KvCacheKind::Q8_0),
    };
    let cands = pinned_bench_candidates(&seed(GpuLayers::All), Some(40), &pins, fits);
    // Pinning KV + layers suppresses their rungs, so ONLY the seed candidate
    // survives — and it carries every pin.
    assert_eq!(cands.len(), 1);
    assert_eq!(cands[0].load.ctx_len, CtxLen::Fixed { n: 8192 });
    assert_eq!(cands[0].load.gpu_layers, GpuLayers::Count { n: 20 });
    assert_eq!(cands[0].load.type_k, Some(KvCacheKind::Q8_0));
}

/// The seed candidate's label is honest: "F16 KV cache" only when the seed
/// really is F16; a pinned KV type relabels it (the seed is not F16 then).
#[test]
fn seed_label_reflects_whether_kv_is_pinned() {
    let fits = |_: &LlamaCppParams| fit(FitVerdict::Fits, 1, 100 << 30);
    let unpinned = bench_candidates_nopins(&seed(GpuLayers::Count { n: 8 }), Some(8), fits);
    assert_eq!(unpinned[0].label, "seed (F16 KV cache)");

    let pins = TunePins {
        type_k: Some(KvCacheKind::Q8_0),
        ..TunePins::default()
    };
    let pinned = bench_candidates(&seed(GpuLayers::Count { n: 8 }), Some(8), &pins, fits);
    assert_eq!(
        pinned[0].label, "seed (pinned KV cache)",
        "a pinned KV type must not be labeled F16"
    );
}
