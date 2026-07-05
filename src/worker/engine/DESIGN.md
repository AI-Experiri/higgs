# `worker/engine` — design notes

## Why a trait, and why here

higgs must support more than one inference backend over time (llama.cpp today; MLX or a remote
shim later) **without** rippling change through the rest of the system. The answer is a single
narrow trait, `HiggsEngine`, placed at the lowest layer that still hides every backend detail:

- **Below the trait**: the backend crate, its FFI, its model handle, its sampler, its template
  renderer, and every raw FFI integer. All of it lives in one submodule (`llamacpp/`) and is
  named nowhere else.
- **Above the trait**: `Box<dyn HiggsEngine>` plus the engine-agnostic types in `mod.rs`. The
  worker request loop, supervisor, node runtime, and serve/API layer never learn which engine
  they drive.

That is what makes "add an engine" a local change: a new submodule + one `REGISTRY` line
(`build_engine` in `mod.rs`, consumed by `worker/mod.rs`).

## The boundary is OpenAI JSON, not rendered prompts

`chat()` takes `messages_json` — the OpenAI `messages` array (with `tools`, `tool_calls`, `tool`
results) serialized verbatim — **not** a pre-rendered prompt string. Chat templating is
engine-specific (llama.cpp ships `common_chat`); rendering above the trait would bake in one
engine's template mechanism and re-introduce the coupling the trait exists to remove. So the
trait hands each engine the raw conversation and lets it render by its own means. higgs invents
**no** chat template and **no** tool-call grammar of its own.

The return, `ChatResult`, is likewise OpenAI-shaped: `content` (tool-call markup already
stripped into `tool_calls`), `reasoning_content` (model thinking the parse extracted;
`None`/partial on mid-think truncation — *not* an error), `finish_reason` (`"stop"`/`"length"`;
the serve boundary rewrites to `"tool_calls"` when calls are present), and prompt/completion
token counts for OpenAI `usage`.

## Streaming deltas: tagged, never a bare string

Generation is a three-way stream — assistant answer, model thinking, tool-call fragments — and
that distinction must survive every hop (worker RPC → supervisor demux → serve SSE / fleet
relay) so `/v1` can emit `delta.content`, `delta.reasoning_content`, and `delta.tool_calls`
separately. Two representations, deliberately split by ownership:

- `EngineDelta<'a>` — **borrowed**, what the engine writes into the `sink` callback. Zero-copy;
  the worker serializes it straight onto the RPC wire.
- `ChatDelta { kind, text }` — **owned**, the item type of every downstream channel.

`ChatDelta::encode_chunk_params` / `decode_chunk_params` are the RPC codec, designed **additive
over the pre-reasoning wire** (`{request_id, delta}`): `Content` is unchanged; `Reasoning` adds
`kind:"reasoning"` (an old peer harmlessly shows thinking as content); `ToolCall` rides the
fragment on a new `tool` field with `delta:""` so an old peer forwards an empty chunk instead of
leaking call JSON into the answer. `ChatDeltaKind::from_wire` defaults unknown/absent `kind` to
`Content` (old-peer compatibility).

## Modeling intent with types, not magic ints

`GpuLayers` and `CtxLen` exist to kill two sentinels the crate's "no magic-int" rule forbids:
the old `gpu_layers: u32` used `u32::MAX` for "all layers", and the old `ctx_len: u32` used `0`
for "auto/trained context". The intent now lives in the type (`GpuLayers::All` /
`CtxLen::Auto`), and the raw FFI integer is produced only at the boundary by `to_n_gpu_layers()`
/ `to_n_ctx()`, called only inside `llamacpp/mod.rs`. `FlashAttn` and `KvCacheKind` do the same
for their raw llama.cpp values.

**Back-compat is explicit, not accidental.** `GpuLayers` and `CtxLen` hand-roll `Deserialize`
to accept **either** the canonical tagged object **or** a bare legacy integer, while `Serialize`
always emits the tagged object — so a `config.json`/`models.json` written before the enum still
reads and migrates forward on its next save. Both lenient paths use `u32::try_from` and
**reject** an out-of-`u32`-range legacy int rather than silently wrapping with `as u32` (the old
typed field rejected it too); a hand-crafted `{"kind":"fixed","n":0}` normalizes to `Auto` via
`CtxLen::fixed`, matching the numeric-`0` sentinel. These edges are pinned by
`registry_tests::{gpu_layers,ctx_len}_deserialize_is_lenient_and_range_checked`.

## const-enum vs. ts-rs union (frontend contract)

Crate rule: every **value-less** engine-agnostic enum the frontend uses is a
`higgs_const_enum!` so it emits a TypeScript **const-object** (usable as a value, `KvCacheKind.F16`)
rather than a bare `"a" | "b"` union. Hence `FlashAttn` and `KvCacheKind` use
`higgs_const_enum!`. `GpuLayers` and `CtxLen` **carry a field** (`n`), so they need ts-rs's
discriminated-union output and stay on `higgs_ts!` — the single documented exception. (`higgs_ts!`
adds `#[ts(export)]`; `higgs_const_enum!` keeps `ts_rs::TS` for path resolution but lets the
`TsConstEnum` derive own the writer — see `src/ts_export.rs`.) `ChatDelta*` and `EngineDelta` are
purely internal (RPC/decode) and cross no ts-rs boundary, so they carry no export macro.

## Selection: a `const` registry, not a plugin system

Engines are compiled in and listed in `REGISTRY` (`const [EngineEntry]`, `build` a plain `fn`
pointer). We deliberately do NOT dynamically load / `dlopen` plugins:

- Compiled-in engines are type-checked, vendored, reproducible — no ABI drift, no runtime
  discovery failure, no supply-chain surface.
- The `const` table is usable from tests and tooling at zero cost.
- Selection is one string (`HIGGS_ENGINE`, case-insensitive); unknown/empty falls back to the
  default with a `tracing::warn!` rather than failing to start. Per-model selection can layer on
  top later without touching the trait.

## Loadability: learned at `load()`, no separate probe

There is **no** `probe()` gate. An earlier design had a `probe(path) -> (bool, Option<reason>)`
that loaded into a throwaway handle to pre-check model→engine compatibility; it was removed
(task "P1: Remove gate-1 probe; learn loadability at actual load"). Loadability is now learned
at the real `load()` call, whose `HG004` failure carries the engine's verbatim reason. This
removes a redundant full model load and a second code path that could disagree with the real
one. (Note: the module-level doc comment in `mod.rs` still lists `probe` in its prose — that
comment is stale; the trait itself has no such method.)

## Object safety & threading

`HiggsEngine: Send` and every method is dyn-compatible, so the worker holds a
`Box<dyn HiggsEngine>` and drives it from its single-threaded request loop. The worker *process*
is the isolation boundary; the engine need not be `Sync`. `devices()` takes `&self` and mutates
no resident state, so it is safe to call on a fresh, model-less worker.

## Error codes

This module *declares* the codes on the trait but the actual `HiggsError`s are constructed one
layer down in `llamacpp/mod.rs` (they are that submodule's codes; full text lives in
`src/diagnostic.rs` and `llamacpp/`'s docs):

| Code | Raised when |
|------|-------------|
| `HG003` | `chat()` called with no model resident. |
| `HG004` | `load()` failed (engine could not make the GGUF resident). |
| `HG005` | `chat()` prompt cannot fit the context window. |
| `HG011` | `chat()` generation failure (context create, prompt decode, sampler, detokenize, decode loop). |

## Invariants

- The trait is the **only** higgs↔backend contract; a backend crate name (`llama_cpp_2`, …) and
  every raw FFI integer appear in exactly one submodule (`llamacpp/`).
- `chat()` receives OpenAI `messages_json`; all templating happens below the trait.
- `REGISTRY[0]` is the default and the registry is never empty (enforced at build time by
  `REGISTRY[0]`; asserted by `registry_tests::default_is_llamacpp_and_registry_nonempty`).
- `GpuLayers`/`CtxLen` serialize **only** as the tagged object; they deserialize leniently from
  the legacy bare int and reject out-of-`u32` values.
- Raw FFI ints (`to_n_gpu_layers`, `to_n_ctx`, `FlashAttn`/`KvCacheKind` raw values) are produced
  only at the `llamacpp` boundary — never above the trait.
- Chat deltas keep their `ChatDeltaKind` tag across every hop; the RPC codec is additive over the
  pre-reasoning wire so old peers degrade safely.

## Deferred / residual

- Only one engine (`llamacpp`) is registered today; the multi-engine machinery is proven by the
  registry unit tests but has no second implementation yet (MLX is planned separately).
- Per-model engine selection is not implemented — selection is process-global via `HIGGS_ENGINE`.
- The stale `probe` mention in the `mod.rs` module doc comment should be dropped on the next code
  touch (docs-only cleanup; behavior is already correct).
