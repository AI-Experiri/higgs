# `worker/engine/llamacpp/` — the llama.cpp engine (FFI boundary)

The concrete [`HiggsEngine`](../mod.rs) implementation backed by the
[`llama-cpp-2`](https://crates.io/crates/llama-cpp-2) crate (which binds the vendored C/C++
llama.cpp). higgs runs on the **AI-Experiri `llama-cpp-rs` fork**, which restores the crate's
`oaicompat` chat API — so llama.cpp's own `common_chat` renders the GGUF chat template AND parses
the model's output back into OpenAI content / tool-calls / reasoning. **There is no minijinja
template layer in higgs** (it was deleted); nothing above this directory renders or parses chat
markup.

This is the **sole FFI boundary** in higgs: `mod.rs` is the only file in the crate allowed to name
`llama_cpp_2` / `llama_cpp_sys_2` (`logging.rs` touches `llama_cpp_2` only to install the log
hook). Everything else is written against the `HiggsEngine` trait, so the FFI stays contained here
and a second engine (MLX, …) drops in without changing any caller.

**Unix/macOS only** — the vendored llama.cpp is built with Metal on macOS, CPU elsewhere; Windows
is unsupported (a build feature of the binding, not a runtime gate). The engine runs inside the
[`worker/`](../../) subprocess, so an FFI fault takes down only the worker.

## File map

| File | Responsibility |
|------|----------------|
| `mod.rs` | The engine itself (**not** a re-export barrel). Holds `LlamaCppEngine` (`Option<LoadedModel>`), the `HiggsEngine` impl (`load`/`unload`/`is_loaded`/`chat`/`devices`), the chat pipeline (template apply → tokenize + fit-check → streaming decode → final parse), the sampler-chain builder, the incremental-parse delta routing, GPU/device enumeration, and every FFI enum mapping (FlashAttn, KV-cache kind, split mode, RoPE scaling, device kind, GGUF kv-override value). Also the process-wide `BACKEND` OnceLock and the free fns `engine_version()` / `device_info()`. |
| `params.rs` | Engine-specific parameter payloads: `LlamaCppParams` (load knobs) and `LlamaCppSamplingParams` (sampler knobs) — the `LlamaCpp` variants of the `engine::LoadParams` / `engine::SamplingParams` umbrellas — plus their helper structs (`KvOverride`, `DryParams`, `MirostatParams`, `LogitBias`, `GrammarParams`) and enums (`SplitMode`, `RopeScalingType`). `LlamaCppParams` derives `TsParamHelp` so every field's `#[help = …]` string is emitted to the TS bindings. |
| `logging.rs` | This engine's worker-side log control: installs the `tracing` subscriber (stderr, no ANSI — the supervisor drains raw text), routes llama.cpp/ggml's FFI logs into `tracing` (target `llama-cpp-2`), filters verbosity live, and captures engine ERROR lines so a load failure can report the engine's own words. |
| `*_tests.rs` | Sibling unit tests (`params_tests.rs`, `logging_tests.rs`); not covered here. |

## `HiggsEngine` methods (what the FFI does)

| Method | FFI / behavior |
|--------|----------------|
| `load` | Drops any resident model, clears the engine-diagnostic buffer, then `LlamaModel::load_from_file` with the requested `LlamaModelParams` (`gpu_layers`, `use_mmap`/`use_mlock`, and — via a pinned, self-referential builder — `cpu_moe`, `cpu_buft_overrides`, `kv_overrides`). One model at a time (v1). On failure returns **`[HG004]`** `EngineLoadFailed` with the drained engine ERROR text (not the opaque binding string). |
| `unload` / `is_loaded` | Drop the `LoadedModel` (FFI freed on `Drop`) / presence check. |
| `chat` | The pipeline (see DESIGN): `apply_chat_template_oaicompat` → `fit_check` (tokenize + clamp budget) → `run_decode` (streaming sample loop) → `parse_response_oaicompat` final parse. Streams tagged `EngineDelta`s (content / reasoning / tool-call) into `sink`; returns a `ChatResult` with content, `finish_reason`, `tool_calls`, `reasoning_content`, and prompt/completion token counts. |
| `devices` | `device_info()` — enumerate ggml backend devices (Metal on macOS, CPU otherwise) into `Vec<GpuDevice>` with VRAM stats. Cheap, read-only, safe on a model-less engine. |

## Public surface

- `LlamaCppEngine` (`mod.rs`) — the engine struct (`#[derive(Default)]`, holds
  `Option<LoadedModel>`). Constructed via the `engine::REGISTRY` entry named `"llamacpp"`; the
  worker only ever talks to it through the `HiggsEngine` trait.
- `engine_version() -> String` (`mod.rs`) — the vendored ggml/llama.cpp runtime version
  (`ggml_version()`), surfaced by the `Higgs` facade `version()` (`HiggsVersionResponse` in
  `api/embed.rs`) and the system-info runtime block (`system.rs`). Distinct from the `llama-cpp-2`
  binding version (`LLAMA_CPP_2_VERSION`).
- `device_info() -> Vec<GpuDevice>` (`mod.rs`) — device enumeration used by `devices()` and by
  worker/tune paths that need VRAM before a model is loaded.
- `params::LlamaCppParams` / `params::LlamaCppSamplingParams` and their helper structs/enums — the
  wire payloads carried by the `engine::{LoadParams, SamplingParams}` umbrellas; also consumed by
  `src/tune/` (the autotune suggester derives the llama.cpp set) and re-emitted to the TS bindings.
- `logging::{install_worker_logging, set_engine_verbose, clear_engine_diagnostics,
  take_engine_diagnostics}` — called by the worker at startup, by the log-level RPC, and by
  `LlamaCppEngine::load` around a load attempt.

## Key internal types

- `LoadedModel` (`mod.rs`) — the resident `LlamaModel` + the `LlamaCppParams` it was loaded with
  (shapes the per-request context); FFI freed on drop.
- Per-request transients built in `run_decode`: a fresh `LlamaContext` (v1 does a naive full
  re-prefill per request), a `LlamaSampler` chain, a `LlamaBatch`, an optional
  `ChatParseStateOaicompat` incremental parser, and an `encoding_rs` UTF-8 decoder that buffers
  partial multi-byte sequences.

See `DESIGN.md` for the decode pipeline, the sampler chain + seed policy, the incremental-parse
delta routing, the FFI-safety invariants, and the HGxxx codes this module owns.
