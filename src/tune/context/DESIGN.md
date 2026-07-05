# `tune/context` — design

## Problem

Autotune capped context at a flat `8192`, ignoring both the model's trained window
and the memory available. A 128 GB-VRAM machine got the same 8192 as a laptop; a
1M-context model was throttled for no reason. The Resource Budget feature wants
**the largest context that fits the budget**, so context scales with hardware and
with any user-set memory cap.

## The memory model (analytical) — invert the forward estimator

The forward footprint of EACH pool (VRAM, RAM) is linear in the context length:

```
need_pool(n_ctx) = base_pool + slope_pool · n_ctx
```

where `base`/`slope` depend on the load's offload config (how many layers' weights +
KV sit in that pool, whether `cpu_moe` moved experts, whether `offload_kqv` keeps KV
off the GPU). Rather than re-derive those coefficients, `Analytical::max_ctx_for_budget`
**evaluates the forward estimators** (`StaticVramEstimator` / `StaticRamEstimator`)
at `n_ctx = 0` and `n_ctx = 1` (the `footprint` closure), each against an uncapped
`ResourceBudget::default()` so it reads the raw `needed_bytes`:

```
base_pool  = need_pool(0)
slope_pool = need_pool(1) − need_pool(0)      // saturating_sub
```

This makes the inverse use the EXACT same math as the fit verdict — partial offload,
`cpu_moe`, and `offload_kqv` are all priced identically on both sides, for free.
It is the core invariant of the module: **a derived context never disagrees with the
footprint the UI shows for it, because it is derived from that same footprint.**

### Inverse (the point)

Because each pool's `need` is **linear**, the inverse is exact — a division, no
search. The `invert(budget, base, slope)` closure computes the largest `n` such that
`base + slope·n <= budget`, and the context must fit BOTH pools, so take the tighter:

```
max_pool(B) = (B − base_pool) / slope_pool                 // floor, saturating
max_ctx     = min(max_vram(vram_budget), max_ram(ram_budget))
```

`slope_pool == 0` ⇒ that pool's footprint doesn't grow with context (e.g. an all-GPU
load charges no per-token RAM, a CPU-only load no per-token VRAM). It then imposes
**no context limit if its fixed cost fits** (`u32::MAX`), but blocks **all** context
(`0`) if `base_pool` alone already exceeds the budget — otherwise a tight cap below
the fixed overhead would be silently ignored. A non-zero-slope budget below `base`
also yields `0` (`saturating_sub`); the caller clamps up to `MIN_CTX`.

### Why min-of-both (not a single basis)

An all-GPU load is VRAM-bound (RAM slope 0); a CPU-only fallback is RAM-bound (VRAM
slope 0); a **partial** offload is bound by whichever pool fills first. A single-basis
(`VRAM if has_gpu else RAM`) heuristic got the partial / `cpu_moe` cases wrong —
subtracting the full weights from VRAM and pinning context at `MIN_CTX` even when the
non-offloaded layers' KV had ample RAM. The two-constraint inverse is correct for
every offload configuration.

## Budget basis & clamps (in `derive.rs`)

`context/` computes the raw inversion; `tune/derive.rs::derive_ctx` owns the policy:

- **Basis:** each pool resolves its own budget via `budget_basis` —
  `budget.max_vram_bytes` / `budget.max_ram_bytes` vs. `hw.vram_total_bytes` /
  `hw.ram_total_bytes`.
- **Headroom:** an EXPLICIT `max_*_bytes` cap is the ceiling (×1.0 — the user already
  carved their margin); a DETECTED total keeps `crate::api::MEMORY_HEADROOM_FRACTION`
  (×0.8), the same tier the fit verdict uses (no double-discount).
- **Clamp:** `MIN_CTX (4096) .. ctx_train`. Never exceed the model's trained window
  (`meta.ctx_train`, falling back to `MIN_CTX` when metadata is missing); never drop
  below 4096 unless the model itself was trained on fewer tokens (`lo =
  MIN_CTX.min(ctx_train)`).
- **`load.ctx_len` is ignored** on input — the inversion drives it.
- **Re-derive after back-off:** `Suggester::suggest` derives once for the initial
  all-GPU load, then again after the MoE / VRAM back-offs, so the final context
  matches the final offload.

## Ensemble (the averaging mechanism)

`AverageStrategy` holds `Vec<Box<dyn ContextEstimator + Send + Sync>>` and
**averages** their `max_ctx_for_budget`, reporting `CtxDerivation { ctx, min, max,
methods }`. The sum is accumulated in `u128` so a `u32::MAX` method (degenerate KV)
can't overflow, and the average is re-clamped to `u32::MAX`. An empty ensemble
returns all-zero (the caller clamps up). Today the ensemble is `[Analytical]`, so
`min == max == ctx` and the rationale reads "(analytical)". The mechanism exists so
later methods slot in without touching `derive.rs` or the route:

- **Empirical** — a regression fit to observed `(model, ctx) → bytes` (oobabooga-style).
- **EngineReport** — llama.cpp's own KV/compute-buffer sizing, queried at probe time.
- **Probe / Calibrated** — load-and-measure once per host to learn the real overhead.

When ≥ 2 methods are wired the average smooths single-method error and the
`min..max` spread surfaces in the tune rationale as a confidence range.

## Why averaging (vs. min/max)

Averaging was the user's choice: no single method is authoritative yet, so blend
them and show the spread rather than trusting the most conservative (min, which can
under-use big GPUs) or the most optimistic (max, which risks OOM). The clamp to
`ctx_train` and the budget headroom keep the average safe.

## Concurrency & error codes

- **No concurrency:** these are pure, synchronous functions over borrowed inputs. No
  locks, no async, no shared state. `Box<dyn ContextEstimator + Send + Sync>` is
  bounded `Send + Sync` only so the ensemble can live inside other `Send` futures,
  not because anything here runs concurrently.
- **No `HGxxx` codes:** this module never fails — inversion is total (saturating
  arithmetic, `u32::MAX` / `0` sentinels for the degenerate cases). Error codes and
  fit verdicts belong to the forward path (`tune/vram.rs`, the estimate route).

## Residual / deferred

- Only `Analytical` is wired; the ensemble spread is currently trivial (`min == max
  == ctx`). The empirical / engine-report / probe methods above are the deferred
  work the averaging mechanism was built to accommodate.
