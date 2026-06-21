# `worker/engine` — the inference engine layer

This module is the **only** seam between higgs and a concrete inference backend. Everything
above it (worker dispatch, supervisor, node runtime, serve layer) is engine-agnostic: it
speaks the [`HiggsEngine`](mod.rs) trait and never names a backend crate.

```
worker dispatch loop  ──>  Box<dyn HiggsEngine>  ──>  llamacpp::LlamaCppEngine  ──>  llama.cpp FFI
                              ▲   (selected from REGISTRY at startup)
                              └── HIGGS_ENGINE picks which entry
```

## The contract

`HiggsEngine` (object-safe, `Send`) is six methods:

| Method | Responsibility |
|---|---|
| `load(path, params)` | Make the GGUF at `path` resident with the given `LoadParams`. `HG004` on failure. |
| `unload()` | Drop the resident model (no-op if none). |
| `is_loaded()` | Whether a model is resident. |
| `chat(messages_json, params, sink)` | Render the OpenAI `messages` (incl. `tools`) via the model's own template, stream deltas into `sink`, return a `ChatResult` (text, finish reason, tool calls, token counts). |
| `probe(path)` | Cheap "can THIS engine load it?" check — load into a throwaway handle, drop it. `(true, None)` or `(false, Some(reason))`. |
| `devices()` | Enumerate host compute devices (CPU/GPU/accel) as this engine sees them. |

`messages_json` is the engine-neutral boundary: the OpenAI `messages` array verbatim. Each
engine renders it by its own means (llama.cpp via vendored `common_chat`; a future engine via
its own jinja renderer). The template-apply mechanism never crosses above this trait.

## Selection & registry

Engines are listed in `REGISTRY` (a `const` table of `EngineEntry { name, build }`). The
worker picks one at startup via `build_engine(HIGGS_ENGINE)`; the first entry is the default
(`llamacpp`). Unknown/empty selectors fall back to the default with a warning.

## Adding an engine (3 steps)

1. **Implement** `HiggsEngine` in a new submodule (e.g. `engine/mlx/mod.rs`); confine all
   backend deps to it. Declare `pub mod mlx;` in `mod.rs`.
2. **Register** one line in `REGISTRY`:
   `EngineEntry { name: "mlx", build: || Box::new(mlx::MlxEngine::default()) }`.
3. **Select** at runtime: `HIGGS_ENGINE=mlx`.

No other file changes. See [DESIGN.md](DESIGN.md) for why the boundary sits exactly here.

## Files

| Path | What it does |
|------|-------------|
| `mod.rs` | `HiggsEngine` trait, `GenParams`/`LoadParams`/`ChatResult`, and the engine `REGISTRY` + `build_engine`. |
| `llamacpp/mod.rs` | The llama.cpp engine: GGUF chat templates, context fit-check, sampler, token streaming, device enumeration. The only file allowed to name `llama_cpp_2`. |
| `llamacpp/logging.rs` | Routes llama.cpp/ggml native log callbacks into higgs's `LogBus`. |
