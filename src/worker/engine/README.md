# `worker/engine` — the inference engine layer

This module is the **only** seam between higgs and a concrete inference backend. Everything
above it (worker dispatch, supervisor, node runtime, serve/API layer) is engine-agnostic: it
speaks the [`HiggsEngine`](mod.rs) trait and the engine-neutral parameter/delta types defined
here, and never names a backend crate.

```
worker request loop  ──>  Box<dyn HiggsEngine>  ──>  llamacpp::LlamaCppEngine  ──>  llama.cpp FFI
                            ▲   (selected from REGISTRY at startup)
                            └── HIGGS_ENGINE picks which entry
```

## The trait

`HiggsEngine` (`mod.rs`) is object-safe and `Send`; the worker holds a `Box<dyn HiggsEngine>`.
It is **five** methods (there is no separate `probe()` — loadability is learned at the real
`load()`):

| Method | Responsibility | Errors |
|---|---|---|
| `load(path, params)` | Make the GGUF at `path` resident with the given `&LoadParams`. | `HG004` |
| `unload()` | Drop the resident model (no-op if none). | — |
| `is_loaded()` | Whether a model is resident. | — |
| `chat(messages_json, params, sink)` | Render the OpenAI `messages` (incl. `tools`) via the model's own template, stream `EngineDelta`s into `sink`, return a `ChatResult`. | `HG003`/`HG005`/`HG011` |
| `devices()` | Enumerate host compute devices (`Vec<crate::system::GpuDevice>`); cheap, read-only, `&self`. | — |

`messages_json` is the engine-neutral boundary: the request's OpenAI `messages` array
serialized verbatim (assistant `tool_calls`, `tool_call_id` results and all). Each engine
renders it by its own means (llama.cpp via the vendored `common_chat`; a future engine via its
own renderer). No chat template or tool-call grammar is ever applied above this trait. These
error codes are *declared* on the trait but *raised* inside `llamacpp/mod.rs` (the codes are
that submodule's; see its docs).

## Engine-agnostic parameter & delta types (all in `mod.rs`)

These are the types the whole crate passes across the trait. Concrete llama.cpp fields live one
layer down in `llamacpp::params`; nothing here names a backend crate or a raw FFI int.

| Type | Kind | Shape / notes |
|---|---|---|
| `LoadParams` | `higgs_ts!` enum, `#[serde(tag="engine")]` | Load-parameter umbrella; only variant today is `LlamaCpp(LlamaCppParams)`. Accessors: `base`, `as_llamacpp`, `ctx_len`, `gpu_layers`, `threads`. |
| `SamplingParams` | `higgs_ts!` enum, `#[serde(tag="engine")]` | Sampling umbrella mirroring `LoadParams`; `LlamaCpp(LlamaCppSamplingParams)`. Carried by `GenParams`. |
| `GenParams` | struct | One request's generation params: `max_tokens`, `sampling`, `tools_json`, `chat_template_kwargs`. |
| `GpuLayers` | `higgs_ts!` **data** enum, `#[serde(tag="kind")]` | `All` / `Count{n}`; replaces the old `u32::MAX` = "all" sentinel. `to_n_gpu_layers()` yields the raw FFI int (called only in `llamacpp/mod.rs`). Custom lenient `Deserialize` still reads a bare legacy int. |
| `CtxLen` | `higgs_ts!` **data** enum, `#[serde(tag="kind")]` | `Auto` / `Fixed{n}`; replaces the old `0` = "auto" sentinel. `to_n_ctx()` yields the raw FFI int. `fixed(0)` normalizes to `Auto`. Custom lenient `Deserialize` reads a bare legacy int. |
| `FlashAttn` | `higgs_const_enum!` (value-less) | `Auto`/`Off`/`On`; mirrors `llama_flash_attn_type` (-1/0/1). Mapped to the raw value only in `llamacpp`. |
| `KvCacheKind` | `higgs_const_enum!` (value-less) | KV-cache element type (`F32`,`F16`,`Q8_0`,`Q5_1`,`Q5_0`,`Q4_1`,`Q4_0`). Mapped to `KvCacheType` only in `llamacpp`. |
| `ChatDeltaKind` | enum | Which stream a delta belongs to: `Content`/`Reasoning`/`ToolCall`. `from_wire()` for RPC decode. |
| `EngineDelta<'a>` | enum (borrowed) | What the engine emits into `sink` — zero-copy `Content`/`Reasoning`/`ToolCall`. |
| `ChatDelta` | struct (owned) | `{kind, text}`; the item of every chat-delta channel. `encode_chunk_params` / `decode_chunk_params` are the additive-over-old RPC wire codec. |
| `ChatResult` | struct | `chat()`'s return: `content`, `finish_reason`, `tool_calls`, `prompt_tokens`, `completion_tokens`, `reasoning_content`. |

**Why two enum macros (recent):** value-less engine-agnostic enums (`FlashAttn`, `KvCacheKind`)
use `higgs_const_enum!` so the frontend gets a usable TS **const-object** (`KvCacheKind.F16`),
per the crate rule "every value-less enum the frontend uses is a const enum." `GpuLayers` and
`CtxLen` **carry data** (`n`), so they stay on `higgs_ts!` and emit a ts-rs discriminated union —
the documented exception to the const-enum rule.

## Selection & registry

Engines are listed in `REGISTRY` (a `const [EngineEntry { name, build }]`, `build` a plain `fn`
pointer). The worker picks one at startup with `build_engine(HIGGS_ENGINE)` (`worker/mod.rs`);
the first entry is the default (`llamacpp`). Unknown/empty selectors fall back to the default
with a `tracing::warn!`. Helpers: `default_engine_name()`, `engine_names()` (for `--help` /
diagnostics).

## Adding an engine (3 steps, no other file changes)

1. **Implement** `HiggsEngine` in a new submodule (e.g. `engine/mlx/mod.rs`); confine all
   backend deps to it. Declare `pub mod mlx;` in `mod.rs`.
2. **Register** one line in `REGISTRY`:
   `EngineEntry { name: "mlx", build: || Box::new(mlx::MlxEngine::default()) }`.
3. **Select** at runtime: `HIGGS_ENGINE=mlx`.

See [DESIGN.md](DESIGN.md) for why the boundary sits exactly here.

## Who uses this module

- `worker/mod.rs` — `build_engine(HIGGS_ENGINE)`, then `.load` / `.chat` / `.devices`.
- `supervisor.rs`, `actor.rs`, `delta_queue.rs` — `ChatDelta` / `ChatDeltaKind` (demux, bounded
  merge, RPC decode).
- `api.rs` — `LoadParams`, `SamplingParams`, `CtxLen`, `GpuLayers`, `KvCacheKind` (request/serve
  mapping; `CtxLen::Auto` is capped at `DEFAULT_CTX_CAP` there, not here).
- `config.rs`, `remote.rs`, `tune/*`, `load_robustness.rs` — persist/derive `LoadParams` /
  `SamplingParams` and the base enums.

## Files

| Path | What it does |
|------|-------------|
| `mod.rs` | The `HiggsEngine` trait + `EngineEntry`/`REGISTRY`/`build_engine`; all engine-agnostic types (`LoadParams`, `SamplingParams`, `GenParams`, `GpuLayers`, `CtxLen`, `FlashAttn`, `KvCacheKind`, `ChatDelta*`, `EngineDelta`, `ChatResult`). Names no backend crate. |
| `llamacpp/` | The llama.cpp engine implementation (params, template render, decode loop, native-log routing). **Owns its own README/DESIGN** and is the only place `llama_cpp_2` / FFI is named. |
