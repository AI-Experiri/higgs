# `tune/context` — budget-aware context-length derivation

Derives **the largest context window that fits the resource budget** — the inverse
of the forward VRAM/RAM estimator in `tune/vram.rs`. Replaces the old flat
`CTX_CAP = 8192`: context now scales with the memory actually available instead of
being pinned at a constant.

## Files

| File | Responsibility |
|------|----------------|
| `mod.rs` | Export barrel + module overview (no logic). |
| `analytical.rs` | `ContextEstimator` trait + `Analytical` — `max_ctx_for_budget`, which inverts the forward VRAM/RAM estimators for the actual load. |
| `average.rs` | `AverageStrategy` + `CtxDerivation` — the ensemble that averages estimators (analytical only today) and reports the per-method spread. |
| `*_tests.rs` | Unit tests (siblings, per the crate test-layout rule). |

## How it's used

`tune/derive.rs::derive_ctx` resolves the VRAM and RAM budget bases (an explicit
cap is the ceiling, a detected total keeps the 0.8 headroom), asks
`AverageStrategy::analytical_only().derive_ctx(load, …)` for the largest context
that fits BOTH pools for that load's offload config, then clamps it to
`MIN_CTX..ctx_train`. `Suggester::suggest` re-runs it after the VRAM/MoE back-offs so
the final context reflects the final offload, and adds the rationale.

## Consistency with the live estimate

`Analytical` derives a context by INVERTING the same forward estimators
(`StaticVramEstimator` / `StaticRamEstimator`) that the live footprint endpoint
(`POST /api/higgs/models/estimate`) uses, so a derived context never disagrees with
the verdict the UI displays for it — partial offload, `cpu_moe`, and `offload_kqv`
are priced identically on both sides.

## Extending (the "mechanism for later")

`AverageStrategy::new(vec![...])` takes any `Vec<Box<dyn ContextEstimator>>`. Adding
an empirical-regression, engine-self-report, or offline-probe method is a new
`impl ContextEstimator` plus one entry in the ensemble — the caller is unchanged,
and the reported spread (`min`/`max`) becomes informative. See `DESIGN.md`.
