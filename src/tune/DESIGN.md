# `src/tune/` — design

> The authoritative, codex-converged spec is `docs/DESIGN-autotune.md` (with the
> binding-gated parameter inventory and the numbered mechanisms #1–#9). This file is
> the module-local summary that travels with the code (folder-docs rule).

## Two modes, one module

`TuneMode` (serialized **lowercase**: `"suggest"` / `"benchmark"`) selects the path:

- **`Suggest`** (default, cheap, pure) — heuristics + best-effort HF-card sampling,
  **no model load**. This is the `Suggester::suggest` core below.
- **`Benchmark`** (G6 "Turbotune", explicit, heavy, offline) — actually loads
  candidate configs and measures tok/s, then saves the fastest as a
  `TuneProvenance::Bench` profile. The *decisions* are `bench.rs` (pure); the *async
  load→measure→unload orchestration with cancellation + a log-fault watchdog* is
  `api.rs::run_benchmark`.

`TuneProvenance` (serialized **PascalCase**: `Heuristic` / `Card` / `Bench`) labels
where a suggestion's numbers came from; `Bench` was added with G6.

## Boundary

```
POST /api/higgs/models/tune {id, mode?, budget?}     serve/mod.rs route
        │  serve/control.rs::control_tune  (thin) → api.rs (async)
        ▼
   look up ModelMeta (scan + models.json meta cache)   ── ModelMetaProvider
   read HardwareInfo (cached; a worker round-trip)      ── HardwareProvider
   [Suggest]  best-effort async HF-card fetch (bounded) ── card_sampling::fetch_card_sampling
        ▼
   Suggester{..}.suggest(meta, hw, budget) -> TuneSuggestion       (pure)
   [Benchmark] bench_candidates → run_benchmark (load+measure) → pick_winner
        ▼
   persist TuneRecord to ~/.higgs/models.json (ProfileStore), return the result
```

`suggest` is **pure**: it takes the providers' outputs as plain values, so it is
unit-tested with fakes. The host (`api.rs`) wires the concrete impls; the suggester
core depends only on the traits. `POST /api/higgs/models/estimate` reuses the same
estimators for a live per-edit footprint (`EstimateRequest`/`EstimateReport`).

## `Suggester::suggest` (analytical, pure)

Ordered pipeline — **derived FRESH within the supplied budget on every (re-)tune**; a
prior saved profile is deliberately NOT reused here (that is a load-seam concern in
`Higgs::load`, so a stricter re-tune re-derives instead of echoing stale params):

1. `derive.derive(meta, hw, budget)` — the heuristic base (`derive_default`).
2. `sampling.recommend(meta)` — HF-card sampling (or empty); a non-default result
   flips provenance to `Card`.
3. **VRAM fit** of the base.
4. **MoE back-off**: VRAM `Overflow` on a GPU host for an MoE model → try
   `cpu_moe = true`, accept **only if RAM still fits** (offloading experts to RAM is
   not free under a RAM budget), else keep `Overflow` with a note.
5. **VRAM-budget back-off**: when an explicit `max_vram_bytes` is set and it still
   overflows, cut GPU offload via `vram::gpu_layers_within_budget` (partial layers, or
   CPU-only) so the saved profile HONORS the cap instead of a full-GPU profile that a
   later plain load would reuse and blow. No cap ⇒ unchanged (honest `Overflow`).
5b. **Context re-derive** for the FINAL offload (`derive::derive_ctx`): the base
   context was sized for the initial all-GPU placement; the MoE / layer back-offs
   change where the KV lands, so the window is re-derived against whichever pool now
   binds (CPU-only ⇒ RAM-bound; partial ⇒ whichever fills first). Idempotent for an
   unchanged all-GPU load; done BEFORE the RAM fit so verdicts match.
6. **Final VRAM + RAM fit** against the final load, plus human `rationale` notes.

> The old DESIGN's "saved-profile-replaces + user-`overrides`-field-merge" precedence
> is **gone**: `suggest` no longer reuses a saved profile, and `TuneRequest.overrides`
> is DEFERRED (see the `TuneRequest` doc-comment) — the user edits the suggestion in
> the UI and loads with the full engine-tagged params. Precedence now lives entirely
> in the load seam (`docs/DESIGN-autotune.md` §3.1).

## Heuristics (`derive_default`)

- `gpu_layers = if has_gpu { All } else { Count{0} }` (CPU-only).
- `threads = min(floor(cores/2).max(1), budget.max_cpu_threads)` — the *tuner's*
  heuristic, deliberately distinct from higgs's non-tuned `available_parallelism()-2`;
  `n_threads_batch` = same.
- `flash_attn = On`, `type_k = type_v = F16`, `n_seq_max = 1`.
- `ctx_len` is **not** a flat cap: `derive_ctx` picks the budget-largest window,
  clamped to `MIN_CTX (4096) .. ctx_train`.

## Fit math (`vram.rs`)

```
needed_vram = gpu_weights + kv_on_gpu + overhead(800 MiB when frac>0)
gpu_weights = size_bytes × gpu_fraction   (× (1-EXPERT_FRACTION 0.75) when cpu_moe on MoE)
kv_bytes    = block_count × ctx × head_count_kv × head_dim × (k_bytes + v_bytes)  // GQA, saturating
head_dim    = embedding_length / head_count  (else 128)
basis       = max_vram_bytes  OR  Σ GPU VRAM
verdict     = Fits | Tight | Overflow   (see thresholds below)
```

RAM mirrors it for the CPU-resident side (un-offloaded weights + `cpu_moe` experts +
CPU-side KV). `offload_kqv == Some(false)` keeps KV in RAM (VRAM KV term → 0).

**Explicit-budget vs detected-total thresholds** (`fit_frac`/`tight_frac`): a
DETECTED total keeps the `system::fits_vram` 80% safety tier (Fits ≤ 80%, Tight ≤ 95%,
else Overflow). An EXPLICIT `max_*_bytes` is the user's hard ceiling — they already
carved their margin — so we do **not** re-apply 80% on top (no `0.75×0.8`
double-discount): Fits fills to `1.0`, Overflow only beyond it.

`kv_cache_bytes` uses **saturating** multiplies so a corrupt/hostile GGUF header
(absurd block/ctx/head counts) reads "won't fit" instead of overflowing `u64`
(debug panic / release garbage).

## Turbotune bench core (`bench.rs`, pure)

- **`bench_candidates(seed, layer_count, estimate_vram)`** — Phase-1 candidate set,
  fastest-likely first: `seed (F16 KV)` → `q8_0 KV` → `q4_0 KV` → `half GPU layers`.
  KV-quant rungs are the cheap VRAM calibration (a quantized V cache needs flash
  attention, left at the engine `auto` default). Half-layers via `half_gpu_layers`
  (unknown `All` with no `layer_count` → skipped). Kept only if `passes_headroom`,
  capped at `MAX_BENCHED_CANDIDATES = 3` (each bench = a full load + real decode).
- **`passes_headroom(fit, abs_floor)`** — rejects `Overflow`; on a VRAM basis also
  requires an ABSOLUTE `ABS_VRAM_HEADROOM_BYTES = 1 GiB` free (`saturating_sub`), on
  top of the fractional fit — a config that "fits" a large budget can still starve the
  compute graph / OS. `budget_bytes == 0` (CPU-only, no VRAM basis) skips the floor
  (the RAM estimate gates it elsewhere).
- **`bench_gen_tps(completion_tokens, ttft, total)`** — the decode-only throughput a
  measured candidate is scored by: `(tokens − 1) / (total − ttft)` (the first token is
  the prefill's output; only the decode window counts, so a slow prompt-processing
  stage can't be mis-scored as a slow GENERATOR). An empty decode window or ≤ 1 token
  returns **0.0** — a failed measurement, not an infinitely fast one.
- **`pick_winner(results)`** — highest `gen_tps`; a manual **strictly-greater** fold
  so a tie keeps the EARLIER (higher-KV-quality / faster-ordered) candidate
  (`max_by` would keep the last).
- **`aggregate_failure(attempts)`** — the single human line naming each tried
  candidate and why it failed; becomes the `[HG063]` `detail`.

## Persistence (`store.rs`)

Per-node `~/.higgs/models.json` (separate from the lean `config.json`). One
`ModelEntry` per id, each part independent + optional:

- a `{path,size,mtime}`-keyed **meta cache** (`meta_if_fresh` — the GGUFs are the
  source of truth),
- a durable **`TuneRecord`** (`profile` + `sampling` + `budget` + `provenance` +
  optional `bench_tps`), reused on the next load ("tune once"),
- durable observed **`ModelPerf`** (passive tok/s, rolling average via `record`).

**Staleness anchors** on `TuneRecord`: `hw_fingerprint` + `model_file_sig`
(`#[serde(default)]`). `is_stale` returns true only when a *present* anchor no longer
matches — **empty anchors are grandfathered** (pre-staleness or bare-load profiles
stay loadable across upgrade instead of all flipping to `NeedsRetune`). `set_profile`
keeps the reused profile in sync with the last accepted load and **refreshes** (not
clears) the anchors, so a *later* hardware/GGUF change is still flagged. When the
written params **differ** from the saved profile (an OOM-degraded fallback or an edited
explicit reload), it also resets `provenance` to `Heuristic` and clears `bench_tps`
(`None`) — those metrics described the OLD config, and the store must not claim a
benchmark throughput for a config that was never benchmarked.

Interior mutability (`parking_lot::Mutex`) so the `ProfileStore` / perf / meta writes
take `&self`. `flush` persists atomically: temp → `sync_all` → rename, `0600`. A
missing file is the empty store; a corrupt file is logged and treated as empty.
Hardware-specific ⇒ stored on the producing machine; the hub pulls it over the scan
RPC. The suggester depends only on `ProfileStore`, so the file could become a DB
without touching tune logic.

## HF-card sampling (`card_sampling.rs`)

Fail-open, best-effort: `fetch_card_sampling` skips non-HF ids, prefers the
STRUCTURED `generation_config.json`, then falls back to scraping `README.md` prose
(all most GGUF quant repos ship). Both parsers are **drop-not-clamp** (a wrong number
is worse than no number) and only accept in-range values; `top_k` accepts an
integer-valued JSON float but drops a fractional one. The prose parser is
token-aware (whole-word key match, code fences stripped, a bounded search window
truncated at the next recognized key) so an unrelated number never becomes a
recommendation. The async fetch is bounded by a caller timeout; the sync suggester
then gets a `StaticSamplingSource` injected (default: `EmptySamplingSource`, no net).

## Error codes

This module owns no `HGxxx` *emission* — the Turbotune codes live in `diagnostic.rs`
and are emitted by `api.rs`. The one contribution: `bench::aggregate_failure` builds
the **HG063** ("Turbotune benchmark found no working config") `detail` text. For
context, the sibling orchestration codes are **HG060** (OOM ladder exhausted),
**HG061** (OOM retry rung), **HG062** (VRAM-recovery wait timeout), **HG064**
(benchmark aborted by a terminal shutdown), **HG065** (a candidate rejected/failed,
moving on), **HG067** (benchmark refused: model loaded — unload first) and **HG068**
(load/chat/unload refused: a benchmark owns the model — retry ~5 min).

## Deferred / out of scope

Multi-GPU derivation (`split_mode`/`main_gpu`/`devices`), `tensor_split` (FFI-only),
integer MoE search (no binding), speculative/NextN (C++-only in the vendored tree),
and per-field `TuneRequest.overrides` (needs a proper all-`Option` partial type).
Carried in the type surface where the binding allows, but not derived.
