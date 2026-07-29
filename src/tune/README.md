# `src/tune/` — autotune (analytical suggester + Turbotune bench core)

Two cooperating pieces, both **pure**:

1. **Analytical `Suggest`** — a *pure advisor*: given typed GGUF metadata + host
   hardware (+ optional HF-card sampling), it computes a nominal llama.cpp load +
   sampling parameter set with VRAM/RAM fit verdicts and a rationale. It **never
   auto-applies** — the caller (the `Higgs::tune` crate-API method, the UI, or the
   `autotune_on_load` seam) reviews, optionally edits, then loads.
2. **Turbotune `Benchmark` decision core** (`bench.rs`, G6) — the *pure* candidate
   generation / headroom gate / benchmarked selection / aggregate-failure diagnosis for
   the MEASURED autotune. The slow async orchestration that actually loads and
   measures candidates lives in `api.rs` (`run_benchmark`), **not here** — this
   module keeps only the deterministic, exhaustively unit-tested decisions.

Adopted from TurboLLM's auto-tune brain: the *logic* ports to Rust, the
spawn-with-CLI-flags *mechanism* does not (higgs links llama.cpp in-process). Full
design + the binding-gated parameter inventory: `docs/DESIGN-autotune.md`.

## Files

| File | Responsibility |
|------|----------------|
| `mod.rs` | **Not a barrel** — owns the wire/data types (`ResourceBudget`, `TuneMode`, `TuneProvenance`, `FitVerdict`/`FitReport`, `TuneRequest`/`TunePins`/`TuneSuggestion`, `EstimateRequest`/`EstimateReport`, `ModelMeta`, `BenchResult`), the narrow DI traits (`DeriveStrategy`/`VramEstimator`/`RamEstimator`/`SamplingSource`/`ProfileStore`/`ModelMetaProvider`/`HardwareProvider`/`Benchmarker`), and the `Suggester` orchestration (`suggest`: derive → sampling → VRAM fit → MoE back-off → VRAM-budget back-off → context re-derive → final fits). |
| `derive.rs` | `HeuristicStrategy: DeriveStrategy` — GGUF/hardware → base params (`derive_default`); `derive_ctx` inverts the estimators for the budget-largest context. |
| `vram.rs` | `StaticVramEstimator`/`StaticRamEstimator` — GQA-correct KV math + the tri-state fit report over `system::fits_vram`; `gpu_layers_within_budget` (offload back-off); `resolve_estimate_ctx`. |
| `bench.rs` | Turbotune PURE core: `bench_candidates` (ordered, pin-aware survivors), `apply_pins`/`pinned_bench_candidates` (TuneRequest pins seam), `passes_headroom` (1 GiB floor), `pick_benchmarked` (earliest-tie), `bench_gen_tps` (decode-only tok/s: `(tokens−1)/(total−ttft)`, empty window ⇒ 0.0), `aggregate_failure` (HG063 text), `Candidate`, `ABS_VRAM_HEADROOM_BYTES`, `MAX_BENCHED_CANDIDATES`. |
| `card_sampling.rs` | HF-card recommended sampling: deterministic drop-not-clamp parsers (`parse_generation_config`, `parse_card_sampling`) + the async fail-open fetch (`fetch_card_sampling`) + the `SamplingSource` impls (`EmptySamplingSource`, `StaticSamplingSource`). |
| `store.rs` | `JsonModelStore` over per-node `~/.higgs/models.json` (`ModelEntry` = `{path,size,mtime}`-keyed meta cache + saved `TuneRecord` + observed `ModelPerf`); backs `ProfileStore`. |
| `context/` | Budget-aware context-length derivation (the inverse estimator). Own `README.md`/`DESIGN.md`; used by `derive::derive_ctx`. |

## Public surface (how the rest of the crate uses it)

- **`Suggester<D,V,R,S>` + `Suggester::static_default()` + `.suggest(meta, hw, budget)`**
  — the `Higgs::tune` facade method (`src/api.rs`) runs the analytical `Suggest`
  path, wiring `StaticSamplingSource` when a card fetch succeeded (`fetch_card_bounded`)
  else the fully-static default. Generic over the traits so tests inject fakes (no
  worker, no disk, no network).
- **`bench::{bench_candidates, pinned_bench_candidates, apply_pins, pick_benchmarked, bench_gen_tps, aggregate_failure}`** —
  the `Higgs::tune` Benchmark mode (`src/api.rs`: `turbotune_bench` → `run_benchmark`)
  uses these to pick and label candidates, score each measurement (`bench_gen_tps`,
  prefill excluded), and build the `HG063` "no working config" detail.
- **`StaticVramEstimator`/`StaticRamEstimator` + `EstimateReport`** — the
  `Higgs::estimate` facade method (`src/api.rs`, via `estimate_footprint`) reuses the
  estimators for the live per-edit footprint; the frontend never reimplements the
  formula.
- **`JsonModelStore`** — the per-node model store: `tuning`/`put_tuning`/`all_tuning`
  (active saved-profile reuse), `tuning_profiles`/`all_tuning_profiles` (the
  `(active, analytical, bench)` dual-profile triple the models list serves),
  `set_profile` (keep the reused profile in sync with the last accepted load, refresh
  staleness anchors), `record_perf`/`perf` (passive tok/s), `put_meta`/`meta_if_fresh`
  (GGUF-metadata cache), `flush` (atomic persist).
- **`TuneSuggestion`/`FitReport`/`FitVerdict`/`ResourceBudget`/…** — ts-rs wire types
  (`higgs_ts!` / `higgs_const_enum!`) consumed by the jigglebot frontend; `FitVerdict`
  carries `#[help]` that generates `FitVerdictHelp.ts` for the verdict-chip tooltips.

There is **no** HTTP tune/estimate route — `serve::v1_router` serves only the strict
OpenAI `/v1` surface. The analytical `Suggest` + estimators are reached through the
crate-API facade methods `Higgs::tune` / `Higgs::estimate` (`src/api.rs`). The
per-model tune *views* the models list exposes (`tuned_load` / `benched_load` /
`tune_provenance` / `bench_tps` on `HiggsModelEntry`) are assembled by
`serve/control.rs` (`model_entry` + `TuneProfileViews::from_triple` / `active_records`)
from the store's dual-profile triples — one `models.json` snapshot per list pass.

## Design tenets

- **Engine-specific data, abstracted behavior.** Param types stay
  llama.cpp-concrete (`LlamaCppParams`); only the *behaviors* sit behind narrow
  traits, so a new sampling source / derivation heuristic / bench backend / store is
  a new trait impl, not a param-type change.
- **Pure & unit-tested.** The estimators, derive, parsers, and the whole Turbotune
  decision core are pure over their inputs, tested with fakes — which is what keeps
  the **unit gate (≥90%)** green. The slow async bench/tune orchestration is `api.rs`'
  job and lands on the integration gate.
- **Reuse, don't re-port.** The VRAM budget primitive is `system::fits_vram`; the
  tri-state thresholds + the KV term are tune's addition on top.
- **GQA-correct.** The KV-cache estimate uses `head_count_kv`, never the query
  `head_count` (which over-estimates KV by the 4–8× GQA factor).
