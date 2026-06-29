# `tune/context` — design

## Problem

Autotune capped context at a flat `8192`, ignoring both the model's trained window
and the memory available. A 128 GB-VRAM machine got the same 8192 as a laptop; a
1M-context model was throttled for no reason. We want **the largest context that
fits the budget**, so context scales with hardware.

## The memory model (analytical) — invert the forward estimator

The forward footprint of EACH pool (VRAM, RAM) is linear in the context length:

```
need_pool(n_ctx) = base_pool + slope_pool · n_ctx
```

where `base`/`slope` depend on the load's offload config (how many layers' weights +
KV sit in that pool, whether `cpu_moe` moved experts, whether `offload_kqv` keeps KV
off the GPU). Rather than re-derive those coefficients, `Analytical` **evaluates the
forward estimators** (`StaticVramEstimator` / `StaticRamEstimator`) at `n_ctx = 0`
and `n_ctx = 1`:

```
base_pool  = need_pool(0)
slope_pool = need_pool(1) − need_pool(0)
```

This makes the inverse use the EXACT same math as the fit verdict — partial offload,
`cpu_moe`, and `offload_kqv` are all priced identically on both sides, for free.

### Inverse (the point)

Because each pool's `need` is **linear**, the inverse is exact — a division, no
search. The context must fit BOTH pools, so take the tighter:

```
max_pool(B) = (B − base_pool) / slope_pool                 // floor, saturating
max_ctx     = min(max_vram(vram_budget), max_ram(ram_budget))
```

`slope_pool == 0` ⇒ that pool's footprint doesn't grow with context (e.g. an all-GPU
load charges no per-token RAM, a CPU-only load no per-token VRAM) ⇒ that pool imposes
no limit (`u32::MAX`). A budget below `base_pool` ⇒ `0` for that pool (the caller
clamps up to `MIN_CTX`).

### Why min-of-both (not a single basis)

An all-GPU load is VRAM-bound (RAM slope 0); a CPU-only fallback is RAM-bound (VRAM
slope 0); a **partial** offload is bound by whichever pool fills first. A single-basis
(`VRAM if has_gpu else RAM`) heuristic got the partial / `cpu_moe` cases wrong —
subtracting the full weights from VRAM and pinning context at `MIN_CTX` even when the
non-offloaded layers' KV had ample RAM. The two-constraint inverse is correct for
every offload configuration.

## Budget basis & clamps (in `derive.rs`)

- **Basis:** each pool resolves its own budget — `max_vram_bytes` / `max_ram_bytes`.
- **Headroom:** an EXPLICIT `max_*_bytes` cap is the ceiling (×1.0 — the user already
  carved their margin); a DETECTED total keeps `MEMORY_HEADROOM_FRACTION` (×0.8), the
  same tier the fit verdict uses (no double-discount).
- **Clamp:** `MIN_CTX (4096) .. ctx_train`. Never exceed the model's trained window;
  never drop below 4096 unless the model itself was trained on fewer tokens.
- **Re-derive after back-off:** `suggest` derives once for the initial all-GPU load,
  then again after the MoE / VRAM back-offs, so the final context matches the final
  offload.

## Ensemble (the averaging mechanism)

`AverageStrategy` holds `Vec<Box<dyn ContextEstimator>>` and **averages** their
`max_ctx_for_budget`, reporting `{ctx, min, max, methods}`. Today the ensemble is
`[Analytical]`, so `min == max == ctx` and the rationale reads "(analytical)". The
mechanism exists so later methods slot in without touching `derive.rs` or the route:

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
