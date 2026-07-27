# `worker/engine/llamacpp/` — design notes

## Why this is the only FFI file

All `unsafe`/FFI lives behind the `HiggsEngine` trait so the rest of higgs never touches C
bindings. `mod.rs` is the one place that names `llama_cpp_2` / `llama_cpp_sys_2` (the engine +
model FFI); `logging.rs` names `llama_cpp_2` only to install the log hook. The trait is the seam a
second engine (MLX, …) implements without changing any caller (`engine::REGISTRY`). The engine
runs in the `worker/` subprocess, so an FFI fault is isolated there.

**Unix/macOS-only.** The vendored llama.cpp builds with Metal on macOS, CPU elsewhere; Windows is
unsupported (a compile-time feature of the binding, not a runtime gate). higgs is pinned to the
**AI-Experiri `llama-cpp-rs` fork** specifically because it restores the crate's `oaicompat` chat
API — `apply_chat_template_oaicompat` / `parse_response_oaicompat` / the streaming
`ChatParseStateOaicompat`. That restoration is why higgs has **no minijinja layer**: llama.cpp's
own `common_chat` both renders the GGUF chat template and parses the model's raw output back into
OpenAI content / tool-calls / reasoning. higgs never hand-parses chat markup.

```text
  rest of higgs (NO FFI)              this dir  =  the FFI boundary            vendored C/C++
  ─────────────────────────      ─────────────────────────────────────    ───────────────────
                            HiggsEngine trait
  worker/mod.rs RPC loop  ───────────────────▶   LlamaCppEngine (mod.rs)
  engine::REGISTRY["llamacpp"]   load/unload/     { Option<LoadedModel> }
                                 is_loaded/chat/          │
                                 devices                  │  llama_cpp_2 /
                                                          ├──────────────────▶  ggml + llama.cpp
                                                          │  llama_cpp_sys_2     (Metal on macOS,
  logging.rs  ──────────────────────────────────────────┘  (Rust binding)       CPU elsewhere)
    installs the FFI log hook (route_engine_logs_to_tracing) — the only other
    file allowed to name llama_cpp_2; everything else is trait-only.
```

## Process-wide backend

`LlamaBackend::init()` is a once-per-process global (the FFI global init). It is held in a
`static BACKEND: OnceLock<LlamaBackend>` and reached via `backend()`; the only init error is
`BackendAlreadyInitialized`, unreachable under `OnceLock`. `device_info()` forces `backend()`
first so device registration has happened even on a cold, model-less worker.

## The chat pipeline (`chat` → helpers)

```text
  chat(messages_json, params, &mut sink)  runs its helpers in sequence, left to right:

  load_template ─▶ apply_template ─▶ fit_check ─▶ run_decode ────────▶ parse_output ─▶ ChatResult
  (GGUF tmpl /     (oaicompat:       (tokenize;   (fresh LlamaContext;   (parse_response_   { content,
   "chatml"         prompt + PEG      HG005 if     sample→accept→EOG       oaicompat →        tool_calls,
   fallback)        parser +          prompt≥ctx   loop; re-batch→        content /          reasoning,
                    chat_format;      else CLAMP    decode)               tool_calls /       token counts }
                    HG050 on fail)    max_tokens)       │                 reasoning)
                                                        │  each raw piece
                                                        ▼
                                              route_parsed_deltas
                                              (ChatParseStateOaicompat; HG052 → raw remainder)
                                                        │
                                          EngineDelta::{Content, Reasoning, ToolCall}
                                                        │
                                                        ▼
                                              &mut sink  (serve layer fans out to /v1 SSE chunks)
```

1. **Template apply** (`load_template` + `apply_template`). Load the GGUF's embedded chat template
   (falling back to the built-in `"chatml"` when the model embeds none), then
   `apply_chat_template_oaicompat` over the verbatim OpenAI `messages` JSON (+ `tools` when
   present). Key `OpenAIChatTemplateParams` choices: `use_jinja: true`; `add_bos: false` (the
   prompt is tokenized with `AddBos::Always`, so the template must not also prepend BOS);
   `add_generation_prompt: true`; `reasoning_format: Some("auto")` and `enable_thinking: true` so
   reasoning-capable templates (Qwen3/DeepSeek-style) separate `reasoning_content` from `content`
   in both the final parse and the streaming diffs (matches llama.cpp's server defaults);
   `parse_tool_calls` on iff the request carried `tools`. Failure → **`[HG050]`**
   `TemplateRenderFailed` (distinct from a generation failure: the prompt could not even be built).
   The result carries the rendered prompt **and** the serialized PEG parser + `chat_format` that
   `common_chat` selected; the caller keeps it alive across the decode so the final parse reuses
   the exact same parser — none is invented here.
2. **Fit-check** (`fit_check`). Tokenize the rendered prompt once (`str_to_token`, `AddBos::Always`)
   and resolve the generation budget against the effective context window (`effective_n_ctx`:
   `CtxLen::Fixed { n }`, or the model's trained context for `CtxLen::Auto`). Semantics: **reject
   with `[HG005]` `ContextOverflow` ONLY when the prompt alone `>= n_ctx`** (no room to generate);
   otherwise **CLAMP** `max_tokens` to `n_ctx − prompt_tokens` (≥ 1) so an oversized request
   truncates (`finish_reason: "length"`) instead of failing. This is the authoritative,
   tokenizer-exact backstop for the serve layer's cheaper lower-bound estimate, which can leave the
   budget a few chat-template tokens too high.
3. **Streaming decode** (`run_decode`). Build a fresh `LlamaContext` from the load-time context
   params (`n_ctx`, `n_batch`/`n_ubatch`, `n_seq_max`, `n_threads`/`n_threads_batch`, `offload_kqv`,
   `swa_full`, `rope_freq_*`, `rope_scaling_type`, `flash_attn`, `type_k`/`type_v`), feed the prompt
   batch (logits on the last prompt token only), then loop: `sample` → `accept` → EOG check →
   `token_to_piece` → route the piece → re-batch the one token → `decode`. Stops on an EOG token
   (`finish_reason = "stop"`) or `max_tokens` (`"length"`). `n_batch` defaults to
   `effective_n_ctx.max(1)` so any fit-checked prompt prefills in one `decode` call (mirrors
   llama.cpp's `simple.cpp`).
4. **Final parse** (`parse_output`). `parse_response_oaicompat` on the full generated text yields
   an OpenAI message; higgs extracts `content`, `tool_calls`, and `reasoning_content` (trimmed).
   `content` is left empty on a pure tool-call turn (OpenAI requires it); on a non-tool turn a
   `null` content falls back to the raw text.

### UTF-8 across detokenization

An `encoding_rs` UTF-8 decoder buffers partial multi-byte sequences, so a token boundary
mid-CJK/emoji never corrupts a character. Only non-empty decoded pieces are forwarded; after the
loop a final `decode_to_string(&[], …, last=true)` drains any buffered trailing bytes (a response
ending mid-multi-byte sequence would otherwise silently truncate its last character).

## Streaming: incremental parse → tagged deltas

When the template apply produced a serialized parser, `run_decode` obtains a
`ChatParseStateOaicompat` (`streaming_state_oaicompat`). Each raw piece is fed to
`update(piece, is_partial=true)` (`route_parsed_deltas`), and every returned OpenAI delta JSON is
split independently into `EngineDelta::Reasoning` / `Content` / `ToolCall` — one diff can carry
several keys at once (llama.cpp's diff shape), so nothing is dropped because a neighbor key was
present. Tool-call fragments are forwarded verbatim; the serve layer re-emits them as
`delta.tool_calls` chunks. A single terminal `update("", false)` after the loop releases text the
lenient parser held back (including the tail of a truncated think block).

**Degradation, not death.** A mid-stream parser error drops the state and streams the remainder as
raw `Content` (warned once, **`[HG052]`**); the non-streaming result is still shaped by the final
`parse_output`. The final parse itself is lenient: a rejected parse preserves the raw text as
content (**`[HG053]`** warn), and a non-JSON message from the crate (an internal bug) does the same
(**`[HG054]`** error). When there is no serialized parser (only the legacy non-jinja route, which
emits no tool/reasoning markup), pieces stream as plain `Content`.

## Sampling & reproducibility

`run_decode` builds a fresh `LlamaSampler` chain per request from the merged
`LlamaCppSamplingParams` (`params.sampling.as_llamacpp()`):

- `temperature <= 0` ⇒ `greedy()` (deterministic argmax; the other samplers and the seed are
  ignored, matching llama.cpp).
- Otherwise the fixed order: `penalties` → `top_k` → `typical` → `top_p` → `min_p` → `top_n_sigma`
  → `xtc` → `temp`/`temp_ext` (dynatemp when `dynatemp_range` is set) → `dist`. Each sampler is
  appended only when its param is set (`None` ⇒ omitted).
- Seed precedence: per-request `sampling.seed`, else the load-pinned `LlamaCppParams.seed`, else
  fresh `rand::random()` entropy.

**Fail-loud on unbuilt samplers.** `grammar`, `logit_bias`, `dry`, and `mirostat` are representable
in `LlamaCppSamplingParams` but not yet applied by this engine (they need the model/vocab handle).
Rather than silently return unconstrained/unbiased output, `run_decode` rejects such a request with
**`[HG013]`** `InvalidSamplingParam` (via `LlamaCppSamplingParams::unsupported_sampler`). This is
unreachable from today's request mappers (none populate those fields) — it guards a future caller.

## Load-time params & the pinned builder (`load`)

The common path uses the move-based `LlamaModelParams` builder (`with_n_gpu_layers`, `with_use_mmap`,
`with_use_mlock`, and — carried but derivation-DEFERRED for the single-GPU/Metal target —
`with_split_mode`/`with_main_gpu`/`with_devices`). Three knobs build a **self-referential** struct
(the params hold raw pointers into override vecs / pattern CStrings): `cpu_moe`
(`add_cpu_moe_override`), `cpu_buft_overrides` (`add_cpu_buft_override`), and `kv_overrides`
(`append_kv_override`). When any is requested the params are `Box::pin`ned, mutated through the pin,
and the pattern `CString`s are kept alive across `load_from_file`. Guards: a `kv_override` key whose
NUL-terminated form exceeds llama.cpp's fixed 128-byte buffer is **skipped with a warn** (llama.cpp
would `memcpy` past the buffer with no bounds check); `parse_kv_override_value` guesses the override
kind bool → int → float → string (127-byte C field).

`gpu_layers`/`ctx_len` use engine-agnostic enums (`GpuLayers::All`, `CtxLen::Auto`) rather than the
raw `u32::MAX`/`0` FFI sentinels; `to_n_gpu_layers()` / `to_n_ctx()` / `effective_n_ctx()` convert
only at the FFI edge, and `effective_n_ctx` keeps the `Auto` sentinel (`to_n_ctx() == 0`) from being
read as a real size by the fit-check / batch sizing.

## Error codes this module owns

Returned as `HiggsError`:

| Code | Variant | Where |
|------|---------|-------|
| `HG003` | `ModelNotLoaded` | defensive guard at the top of `chat` (the worker checks first). |
| `HG004` | `EngineLoadFailed` | `load` failure; reason is the drained engine ERROR text (see logging). |
| `HG005` | `ContextOverflow` | `fit_check`, only when the prompt alone exceeds `n_ctx`. |
| `HG011` | `GenerationFailed` | `gen_fail(stage, …)` at each decode stage (tokenize, create context, prompt/loop decode, batch add, position `i32` conversion, detokenize). |
| `HG013` | `InvalidSamplingParam` | `run_decode`, an advanced sampler was requested but is not yet applied. |
| `HG050` | `TemplateRenderFailed` | `apply_template`, the GGUF Jinja template failed over this request. |

Log-only diagnostics (never returned, only traced): `HG052` (incremental parse failed mid-stream →
raw remainder), `HG053` (final parse rejected → raw text as content), `HG054` (crate parse returned
non-JSON — internal bug → raw text).

## Logging (`logging.rs`)

Only THIS engine's logging lives here (a future engine ships its own). `install_worker_logging`
builds a `tracing_subscriber::registry` with two layers:

1. **`EngineDiagnosticCapture`** — a standalone `Layer`, filtered to the `"llama-cpp-2"` target at
   `ERROR` only, that appends each engine ERROR line to a bounded (`MAX_ENGINE_DIAGNOSTICS = 64`)
   `Mutex<Vec<String>>`. It is independent of the UI verbosity gate; its per-layer filter scopes
   callsite INTEREST to ERROR so it does not re-enable the DEBUG/INFO engine traffic the fmt layer
   suppresses at the source. This exists because a failed `load_from_file` returns only an opaque
   binding error ("null result from llama cpp") while the REAL cause (e.g.
   `unknown model architecture: 'gemma4'`) is emitted as a separate log event. `load` calls
   `clear_engine_diagnostics()` before the attempt and `take_engine_diagnostics()` on failure,
   joining the captured lines into the `[HG004]` reason (falling back to the binding string when the
   engine logged nothing, e.g. an OOM kill).
2. **fmt layer** — stderr, `with_ansi(false)` (the supervisor renders the drain as plain text;
   color escapes would show as literal garbage), gated by `EngineLogFilter`.

`EngineLogFilter` has two gates, both keyed off a live-atomic `verbose` flag seeded from
`HIGGS_WORKER_VERBOSE=1` at spawn and flipped at runtime by `set_engine_verbose` (called from the
worker's log-level RPC — read per event, so it takes effect without a restart): a **level** gate
(`"llama-cpp-2"` events pass at INFO+ normal, DEBUG+ verbose; other targets always pass) and a
**module** gate that drops `NOISY_ENGINE_MODULES` — the `llama_model_loader` per-KV dump and the
`print_info` hyperparameter block, which llama.cpp emits unconditionally at INFO — in normal mode,
while always surfacing WARN/ERROR. `route_engine_logs_to_tracing` (`send_logs_to_tracing`) is the
only place allowed to touch the binding's log hook.

## Concurrency / locking

The engine holds one model and is driven by the single-threaded worker RPC loop — no per-engine
lock. The shared process-wide state is `BACKEND` (`mod.rs`, `OnceLock<LlamaBackend>`, init-once via
`backend()`) plus, in `logging.rs`, `ENGINE_VERBOSE` (`OnceLock<Arc<AtomicBool>>`, `Relaxed`
load/store) and `ENGINE_DIAGNOSTICS` (`Mutex<Vec<String>>`). The diagnostic buffer is cleared → written (by the log layer during load) →
drained on failure within a single `load` call; because the worker serves loads one at a time,
there is no interleaving of two load windows over the buffer.

## FFI safety

The `unsafe` blocks are narrow and documented at their call sites: `engine_version()` and the device
name/description strings read static, NUL-terminated C strings (no allocation, no lifetime hazard);
`device_info()` enumerates `ggml_backend_dev_*` by index after `backend()` registers the backends,
bounds the index, skips null handles, and never panics on a report failure. Position/index
conversions to `i32` use `try_from` → `[HG011]` on overflow rather than panicking (unreachable in
practice — positions never exceed `n_ctx`).

## Known limitations / deferred

- **Template `additional_stops`** (stop STRINGS a template declares) are NOT honored — the decode
  loop stops only on EOG tokens / `max_tokens`. Latent: the supported families (Qwen, Llama-3,
  Gemma, DeepSeek, Nemotron) terminate on EOG and declare no stop strings.
- **Grammar-constrained sampling** (`tmpl_result.grammar`) and the advanced samplers
  (`grammar`/`logit_bias`/`dry`/`mirostat`) are deferred — requested use fails loud with `[HG013]`
  rather than silently degrading.
- **Multi-GPU** (`split_mode`/`main_gpu`/`devices`, `tensor_split`) is carried in `LlamaCppParams`
  for completeness but its derivation is deferred (single-GPU/Metal target); `tensor_split` has no
  safe setter and is intentionally absent. Speculative/NextN decoding is C++-only in the vendored
  tree and out of scope.
- **One model / one context (v1).** The engine hosts a single model, and every chat builds a fresh
  `LlamaContext` (naive full re-prefill per request — no KV reuse across turns yet).

## Boundaries / what does NOT belong here

- The `HiggsEngine` trait, the engine registry, and the engine-agnostic umbrellas
  (`LoadParams`/`SamplingParams`/`GenParams`/`EngineDelta`/`ChatResult`, `CtxLen`/`GpuLayers`/
  `FlashAttn`/`KvCacheKind`) → `../` (`worker/engine`).
- The RPC loop + worker state → `../../` (`worker/mod.rs`).
- The autotune suggester that derives `LlamaCppParams` → `src/tune/`.
