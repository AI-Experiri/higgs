# `worker/engine/llamacpp/` — design notes

## Why this is the only FFI file

All `unsafe`/FFI lives behind the `HiggsEngine` trait so the rest of higgs never touches C
bindings. This directory is the one place that touches `llama_cpp_2` / `llama_cpp_sys_2`
(`mod.rs` for the engine/model FFI, `logging.rs` only to route the C log callback into
`tracing`); the trait is the seam a second engine (MLX, …) would implement without changing any
caller. It runs in the `worker/` subprocess, so an FFI fault is isolated to that process.

Build is **Unix/macOS-only**: the vendored llama.cpp is built with Metal on macOS, CPU
elsewhere; Windows is unsupported (a compile-time feature of the binding, not a runtime gate).

## The chat pipeline (`chat` → helpers)

1. **Template.** Load the GGUF's embedded `tokenizer.chat_template`; apply it OAI-compat
   (`add_bos = false` — the tokenizer adds BOS) to the OpenAI `messages` (+ tools when a parser
   wants them). A model with no embedded template falls back to `chatml`.
2. **Fit-check (`[HG005]`).** Tokenize the rendered prompt (`AddBos::Always`) once, then reject
   if `prompt_tokens + max_tokens > n_ctx` BEFORE any decode. This is the authoritative
   tokenizer check (the serve layer's byte-estimate is only a cheap pre-filter).
3. **Streaming decode (`run_decode`).** Build a `LlamaContext` from the load-time context params
   (n_ctx, n_batch/n_ubatch, offload_kqv, rope_freq_*, flash_attn, type_k/type_v), feed the
   prompt batch (logits on the last token only), then loop: sample → `accept` → detokenize →
   stream the piece via `sink` → re-batch the one token → decode. Stops on an EOG token
   (`finish_reason = "stop"`) or `max_tokens` (`"length"`).
4. **Final parse.** Parse the full generated turn into `{content, finish_reason, tool_calls,
   prompt_tokens, completion_tokens}`.

UTF-8 across detokenization: an `encoding_rs` decoder buffers partial multi-byte sequences so a
token boundary mid-CJK/emoji never corrupts a character.

## Sampling & reproducibility

A fresh `LlamaSampler` per request: greedy when `temperature <= 0`, else temperature +
distribution. When `LoadParams.seed` is set, sampling is seeded (deterministic across runs);
unset draws a fresh random seed per request (per-request entropy), greedy ignores the seed.

## Tool-call streaming filter (and its known limitation)

When a registry parser matches the model's chat template, a `ToolCallStreamFilter` suppresses
the tool-call *envelope* (open marker onward) from the streamed deltas — clients see clean
content, and the full turn is parsed for structured `tool_calls` AFTER streaming completes, so
the structured data is never lost. KNOWN LIMITATION: the filter is installed only when a
*registry* parser matches; a model whose tool format is handled by the crate's primary
OAI-compat parser but recognized by no registry parser (e.g. Llama-3 `<|python_tag|>`) streams
raw markup in the deltas (the final structured parse still succeeds). The target families
(Gemma, Qwen, DeepSeek, Nemotron) are covered. Template-declared `additional_stops` are also not
honored mid-decode (only EOG + `max_tokens`); latent because supported families terminate on
EOG.

## FFI safety

The handful of `unsafe` blocks are narrow and documented at their call sites:

- `ggml_version()` / device strings — read static, NUL-terminated C strings (no allocation,
  no lifetime hazard).
- `device_info()` enumerates `ggml_backend_dev_*` by index after `backend()` registers the
  backends; it bounds the index, skips null/failed handles, and never panics on a report
  failure.
- Position/index conversions to `i32` use `try_from` → `[HG011]` on overflow rather than a
  panic (unreachable in practice — positions never exceed `n_ctx`).

## Logging (`logging.rs`)

Only THIS engine's logging lives here (a future engine ships its own). It routes the binding's
C log callback into `tracing` and gates verbosity with a two-stage filter: engine events pass at
INFO+ (normal) or DEBUG+ (verbose); load-time noise (the `llama_model_loader` KV dump, the
`print_info` hyperparameter block) is dropped in normal mode unless it is WARN/ERROR. The verbose
flag is an atomic flipped live by `M_LOG_LEVEL`, read per event, so it takes effect without a
restart. Output is plain stderr (no ANSI) because the host drains it as raw text into the
Developer Logs.

## Boundaries / what does NOT belong here

- The `Engine` trait + registry → `../` (`worker/engine`).
- The RPC loop + worker state → `../../` (`worker/mod.rs`).
