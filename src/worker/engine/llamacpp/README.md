# `worker/engine/llamacpp/` — the llama.cpp engine (FFI boundary)

The concrete [`HiggsEngine`](../) implementation backed by the `llama-cpp-2` crate (which binds
the vendored C/C++ llama.cpp). This is the **sole FFI boundary** in higgs: this directory is the
only module that touches `llama_cpp_2` / `llama_cpp_sys_2` (`mod.rs` for all engine/model FFI,
`logging.rs` only to route the C log callback). Everything else in the crate is written against
the `Engine` trait, so the FFI stays contained here.

It runs inside the [`worker/`](../../) subprocess (crash isolation), so an FFI fault takes down
only the worker.

## File map

| File | Responsibility |
|------|----------------|
| `mod.rs` | The engine: trait methods (`load`/`unload`/`is_loaded`/`chat`/`probe`/`devices`/`version`), the chat pipeline (template apply → tokenize + fit-check → streaming decode → final parse), the streaming decode loop + sampler, tool-call parsing (crate parser + registry fallback), GPU/device enumeration, and the FFI enum mappings (FlashAttn, KV-cache kind, device kind). Holds `LlamaCppEngine` (`Option<LoadedModel>` + a `ToolParserRegistry`). |
| `logging.rs` | Worker log control: installs the `tracing` subscriber (stderr, no ANSI — the host drains raw text), routes llama.cpp/ggml's C log callback into `tracing` (target `llama-cpp-2`), and filters verbosity live (INFO+ normal / DEBUG+ verbose, with load-time KV/hyperparameter noise suppressed unless verbose; WARN/ERROR always surfaced). Toggled by `M_LOG_LEVEL` via an atomic. |

## `HiggsEngine` methods (what the FFI does)

| Method | FFI / behavior |
|--------|----------------|
| `load` | `LlamaBackend::init()` + `LlamaModel::load_from_file()` with `use_mmap`/`use_mlock`/`gpu_layers`. One model at a time; replaces any resident. |
| `unload` / `is_loaded` | Drop the handle (`Drop` frees FFI resources) / presence check. |
| `chat` | The pipeline: load+apply the GGUF chat template → tokenize + context-fit (`[HG005]` on overflow) → streaming decode (sample → detokenize → `sink(delta)` → re-batch) → final `parse_output` (content, finish_reason, tool_calls, token counts). |
| `probe` | `load_from_file(with_vocab_only(true))` — cheap Gate-1 loadability check on a throwaway handle; returns `(loadable, reason)` with the engine's verbatim error. |
| `devices` | Enumerate ggml backend devices (Metal on macOS, CPU otherwise) → `Vec<GpuDevice>` with VRAM stats. |
| `version` | `ggml_version()` — the real runtime ggml version (NOT the crate version); part of the support-cache key so an engine upgrade invalidates verdicts. |

## Key types

- `LlamaCppEngine` — the engine struct: `Option<LoadedModel>` + a `ToolParserRegistry`.
- `LoadedModel` — the resident `LlamaModel` handle + its `LoadParams`; FFI freed on drop.
- Per-request transients (in the decode loop): `LlamaContext` (built from the load-time
  context params), `LlamaSampler` (greedy or temperature+seed), `LlamaBatch`, a
  `ToolCallStreamFilter`, and a UTF-8 decoder buffering partial multi-byte sequences.

See `DESIGN.md` for the decode pipeline, sampling/seed, tool-call streaming, and FFI-safety
invariants.
