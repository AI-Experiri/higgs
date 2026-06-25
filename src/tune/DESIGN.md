# `src/tune/` — design

> The authoritative, codex-converged spec is `docs/DESIGN-autotune.md` (with the
> binding-gated parameter inventory). This file is the module-local summary that
> travels with the code (folder-docs rule).

## Boundary

```
POST /api/higgs/models/tune {id, mode, budget?, overrides?}
        │  (serve/control.rs handler — async)
        ▼
   look up ModelMeta (scan + models.json meta cache)   ── ModelMetaProvider
   read HardwareInfo (cached worker round-trip)         ── HardwareProvider
   [Suggest mode] best-effort async HF-card fetch       ── card_sampling::fetch_card_sampling
        ▼
   Suggester::suggest(meta, hw, budget, saved, overrides) -> TuneSuggestion
        │   derive → sampling → precedence → vram fit → MoE back-off → ram fit
        ▼
   persist TuneRecord to ~/.higgs/models.json (ProfileStore), return TuneSuggestion
```

`suggest` is **pure**: it takes the providers' outputs as plain values, so it is
unit-tested with fakes. The host wires the concrete impls; the suggester core
depends only on the traits.

## Suggester precedence (in `suggest`)

1. `derive_default(meta, hw, budget)` — the heuristic base.
2. recommended `sampling` from the `SamplingSource` (HF card / empty).
3. a **saved profile** (a prior tune the user kept) **replaces** the derived base.
4. explicit **user overrides** field-merge on top (win).
5. **VRAM fit**; then **MoE back-off**: if VRAM overflows on a GPU host for an MoE
   model, try `cpu_moe = true` — accept **only if RAM still fits**, else keep
   `Overflow` (offloading to RAM is not a free escape under a RAM budget).
6. **RAM fit** + rationale notes.

This mirrors the load-seam precedence (`docs/DESIGN-autotune.md` §3.1): explicit >
saved profile > `default_load`/autotune > `derive_default`.

## Heuristics (`derive_default`, DESIGN §6)

- `ctx_len = min(ctx_train.unwrap_or(8192), 8192)`.
- `gpu_layers = if has_gpu { u32::MAX (all) } else { 0 }`.
- `threads = min(floor(cores/2).max(1), budget.max_cpu_threads)` — the tuner's
  heuristic (distinct from higgs's non-tuned `available_parallelism()-2`).
- `flash_attn = On`, `type_k = type_v = F16`, `n_seq_max = 1`.

## Fit math (`vram.rs`, DESIGN §6)

```
needed_vram = gpu_weights + kv_on_gpu + overhead
kv_bytes    = block_count × ctx × head_count_kv × head_dim × (k_bytes + v_bytes)   // GQA!
head_dim    = embedding_length / head_count  (else 128)
budget      = max_vram_bytes  OR  Σ GPU VRAM
verdict     = Fits (≤ 80% via system::fits_vram) | Tight (≤ 95%) | Overflow
```

`cpu_moe` removes the expert fraction from `gpu_weights` (and adds it to the RAM
estimate); `gpu_layers` scales the offloaded fraction. RAM mirrors it for the
CPU-resident side.

## Persistence (`store.rs`, DESIGN §8)

Per-node `~/.higgs/models.json` (separate from `config.json`). One `ModelEntry`
per id: a `{path,size,mtime}`-keyed **meta cache** (the GGUFs are the source of
truth), a durable **tuning** profile (reused next load), and durable observed
**perf** (passive tok/s, rolling average). Atomic temp→fsync→rename, `0600`.
Hardware-specific ⇒ stored on the producing machine; the hub pulls it over the
scan RPC. The suggester depends only on `ProfileStore`, so the file can become a
DB without touching tune logic.

## Deferred / out of scope

Multi-GPU derivation (`split_mode`/`main_gpu`/`devices`), `tensor_split`
(FFI-only), integer MoE search (no binding), speculative/NextN (C++-only in the
vendored tree), the offline `Benchmark` mode (P2). All carried in the type surface
where the binding allows, but not derived.
