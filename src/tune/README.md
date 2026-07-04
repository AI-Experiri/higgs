# `src/tune/` — autotune suggester

A **pure advisor**: given typed GGUF metadata + host hardware (+ optional measured
bench), it computes a nominal llama.cpp load + sampling parameter set with a fit
verdict and rationale. It **never auto-applies** — the caller (the `tune` route,
the UI, or the `autotune_on_load` seam) reviews, optionally edits, then loads.

Adopted from TurboLLM's auto-tune brain; the *logic* ports to Rust, the
spawn-with-flags *mechanism* does not (higgs links llama.cpp in-process). Full
design + the binding-gated parameter inventory: `docs/DESIGN-autotune.md`.

## Files

| File | Responsibility |
|------|----------------|
| `mod.rs` | Types (`ResourceBudget`, `TuneRequest`/`TuneSuggestion`, `FitReport`/`FitVerdict`, `ModelMeta`), the narrow traits (`DeriveStrategy`/`VramEstimator`/`RamEstimator`/`SamplingSource`/`ProfileStore`/…), and the `Suggester` orchestration (derive → sampling → precedence merge → VRAM fit → MoE back-off → RAM fit → rationale). |
| `derive.rs` | `HeuristicStrategy: DeriveStrategy` — GGUF/hardware → base params (`derive_default`). |
| `vram.rs` | `StaticVramEstimator`/`StaticRamEstimator` — GQA-correct KV math + tri-state fit over `system::fits_vram`. |
| `card_sampling.rs` | HF-card recommended sampling: the deterministic parser + the async (fail-open) fetch + the `SamplingSource` impls. |
| `store.rs` | `JsonModelStore` over per-node `~/.higgs/models.json` (`ModelEntry` = meta cache + tuning + perf); backs `ProfileStore`. |

## Design tenets

- **Engine-specific data, abstracted behavior.** Param types stay
  llama.cpp-concrete (`LlamaCppParams`); only the *behaviors* sit behind traits,
  so a new recommendation source / derivation heuristic / bench backend / store
  is a new trait impl, not a param-type change.
- **Pure & unit-tested** — estimators/derive/parse are pure
  over their inputs, tested with fakes (no worker, no disk, no network), which is
  what keeps the unit gate green.
- **Reuse, don't re-port.** The VRAM budget primitive is `system::fits_vram`; the
  tri-state thresholds + the KV term are tune's addition on top.
- **GQA-correct.** The KV-cache estimate uses `head_count_kv`, never the query
  `head_count` (which over-estimates KV by the GQA factor).

## Coverage

`tune/*` is the **unit gate's** job (pure logic, ≥90% lines) — it is on the unit
coverage set. The end-to-end `tune` route + load path lands
on the integration gate.
