# `tune/context` — budget-aware context-length derivation

Derives **the largest context window that fits the resource budget** — the inverse
of the forward VRAM/RAM estimators in `tune/vram.rs`. This is the Resource Budget
feature's context path: it replaces the old flat `CTX_CAP = 8192` so the context
window scales with the memory actually available (and with the model's trained
window) instead of being pinned at a constant.

## Files

| File | Responsibility |
|------|----------------|
| `mod.rs` | Export barrel + module overview (no logic). Re-exports `Analytical`, `ContextEstimator`, `AverageStrategy`, `CtxDerivation`. |
| `analytical.rs` | `ContextEstimator` trait + `Analytical` — `max_ctx_for_budget`, the closed-form estimator that INVERTS the forward VRAM/RAM estimators for the actual load. |
| `average.rs` | `AverageStrategy` + `CtxDerivation` — the ensemble that averages `ContextEstimator`s (analytical only today) and reports the per-method spread. |
| `analytical_tests.rs`, `average_tests.rs` | Unit tests (siblings, per the crate test-layout rule). |

## Public surface

- `trait ContextEstimator` (`analytical.rs`) — one method, `max_ctx_for_budget(load,
  meta, hw, vram_budget, ram_budget) -> u32`: the largest unclamped `n_ctx` whose
  forward footprint fits BOTH the VRAM and RAM budgets for `load`'s offload config.
  Returns `u32::MAX` when neither pool grows with context (degenerate metadata).
- `struct Analytical` (`analytical.rs`) — the closed-form default `impl
  ContextEstimator`. Evaluates the forward estimators at `n = 0` and `n = 1` to
  recover each pool's `base`/`slope`, then a single division inverts each; the
  tighter of the two constraints (`min`) wins.
- `struct AverageStrategy` (`average.rs`) — holds `Vec<Box<dyn ContextEstimator +
  Send + Sync>>`. Constructors: `analytical_only()` (the default ensemble) and
  `new(estimators)` (custom, for tests / future methods). `derive_ctx(...)` returns
  a `CtxDerivation`.
- `struct CtxDerivation` (`average.rs`) — `{ ctx, min, max, methods }`: the averaged
  budget-derived context plus the per-method spread and count. With a single
  estimator `min == max == ctx`.

## How the crate uses it

`tune/derive.rs::derive_ctx` is the sole caller:

1. Resolves each pool's budget basis via `budget_basis(explicit, detected)` — an
   explicit `ResourceBudget::max_vram_bytes` / `max_ram_bytes` cap is the ceiling
   (×1.0, the user already carved their margin); a detected total keeps
   `crate::api::MEMORY_HEADROOM_FRACTION` (×0.8), the same tier the fit verdict uses.
2. Calls `AverageStrategy::analytical_only().derive_ctx(load, meta, hw, vram_budget,
   ram_budget)` for the largest context that fits BOTH pools for that load's offload
   config.
3. Clamps the result to `MIN_CTX (4096) .. ctx_train` (never above the model's
   trained window; never below 4096 unless the model was trained on fewer tokens).

`Suggester::suggest` (`tune/mod.rs`) re-runs `derive_ctx` after the MoE / VRAM
back-offs so the final context reflects the final offload config, and folds the
`CtxDerivation` spread into the tune rationale.

## Consistency with the live estimate

`Analytical` derives a context by INVERTING the same forward estimators
(`vram::StaticVramEstimator` / `StaticRamEstimator`) that the live footprint
estimator `Higgs::estimate` (`src/api.rs`, via the pure `estimate_footprint`) uses,
so a derived context never disagrees with the verdict the UI displays for it —
partial GPU offload, `cpu_moe`, and `offload_kqv` are priced identically on both
sides. `Higgs::estimate` is a crate-API facade method (embedders call it directly);
there is no HTTP control route for it.

## Extending (the "mechanism for later")

`AverageStrategy::new(vec![...])` accepts any `Vec<Box<dyn ContextEstimator>>`.
Adding an empirical-regression, engine-self-report, or offline-probe method is a new
`impl ContextEstimator` plus one entry in the ensemble — the caller (`derive.rs`) is
unchanged, and the reported `min`/`max` spread becomes an informative confidence
range. See `DESIGN.md`.
