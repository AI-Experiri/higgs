---
title: Endpoints
description: Every route exposed by higgs — /v1 OpenAI-compatible and /api/higgs/* control.
---

## Overview

higgs exposes two route groups:

| Group | Purpose |
|-------|---------|
| `/v1/*` | OpenAI-compatible inference — `chat/completions`, `models` — chat clients, OpenAI SDK drop-ins |
| `/api/higgs/*` | Control plane — `status`, `system`, `version`, model catalog + `models/load`·`unload`, `worker/stop`, `logs` (snapshot + SSE stream) and `logs/settings`, `settings`, plus the fleet surface `nodes`, `nodes/load`·`unload`, and `pair` |

All routes are mounted by `higgs::serve::router(Arc<Higgs>)`.

When [API keys](/system-design/#securing-the-api) are configured, every route
except the health probes (`/health` and `/api/higgs/health`) requires
`Authorization: Bearer hgk_…` with a sufficient scope (`chat` / `models` /
`admin`); a missing or insufficient token gets `401`.

Both surfaces also expose a cheap readiness probe (`GET /health`,
`GET /api/higgs/health`) and sit behind a shared **serve-layer hardening** stack
(Host guard, body limit, control timeout, panic recovery) — see the sections
below.

---

## Error Mapping

The same status table applies to both surfaces:

| HiggsError | HTTP status |
|------------|-------------|
| HG002 ModelNotFound | 404 |
| HG003 ModelNotLoaded | 404 |
| HG005 ContextOverflow | 400 |
| HG006 WorkerSpawnFailed | 503 |
| HG007 WorkerDead | 503 |
| HG012 ForbiddenHost | 403 |
| HG013 InvalidSamplingParam | 400 |
| HG014 ServerBusy | 503 |
| HG015 InvalidModelId | 400 |
| HG016 ChatTimeout | 504 |
| HG017 InsufficientMemory | 503 |
| HG018 ResidentModelMismatch | 503 |
| HG019 ServingDisabled | 503 |
| HG027 NodeUnreachable | 503 |
| HG021/HG025/HG026 (sysinfo/download/update) | 500 |
| anything else | 500 |
| missing / insufficient API key | 401 |

The pairing & handshake codes **HG022–HG024** and **HG028** are iroh
control-plane errors (bad/expired token, no agreed protocol version, not
allow-listed, no HELLO in time) — they surface on the node-dial path, never as an
HTTP response.

**`/v1` error envelope:**

```json
{
  "error": {
    "message": "[HG003] model not loaded: org/model — load it explicitly first",
    "type": "invalid_request_error",
    "code": "model_not_found"
  }
}
```

`type` is `invalid_request_error` for 4xx, `server_error` otherwise.
`code` is `model_not_found` on 404; absent on other statuses.

The `/v1` envelope `message` is **path-redacted**: absolute filesystem paths and
`host:port` bind addresses are replaced with `<redacted>` before crossing the
client boundary (diagnostic code, model id, and prose are preserved). **By
default** no prompt CONTENT is logged at `info` on the `/v1` path — the
`log_incoming_tokens` and `show_log_fields` toggles explicitly opt into logging
prompt/field content. The full unredacted message (with
paths) is still logged server-side at the origin and returned on the control
surface below — which is ours, not an interop boundary.

**Control error envelope** (NOT redacted — full Display, paths included):

```json
{ "error": "[HG003] model not loaded: org/model — load it explicitly first" }
```

---

## Health Routes

### GET /health  ·  GET /api/higgs/health

Cheap readiness probe. Returns 200 as soon as the server is up — **no worker
RPC**. This answers "is the higgs server reachable?", not "is a model loaded?".
Use `GET /api/higgs/status` for load state.

Both paths are identical; `/health` is the conventional top-level probe and
`/api/higgs/health` keeps the control plane self-contained.

**Response:** a bare `200 OK` with an **empty body** (no JSON) — it answers only
"is the control surface reachable?".

**curl:**

```sh
curl -i http://localhost:11434/health
curl -i http://localhost:11434/api/higgs/health
```

---

## /v1 Routes

### GET /v1/models

Returns the **loaded** local model plus any **remotely-routed** models whose node
is currently connected (servable right now). An empty list means nothing is loaded
or routable. Use `GET /api/higgs/models` for the full on-disk catalog.

higgs is **spawn-on-load**, so the idle state has no worker; this endpoint still
returns `200` with `{"data":[]}` then (it never gates on `worker_alive`). A
crashed worker likewise presents here as an empty list — `GET /api/higgs/status`
exposes `worker_alive` for diagnostics. This surface never returns a worker-down
`503`.

**Response (200):**

```json
{
  "object": "list",
  "data": [
    {
      "id": "org/model-name",
      "object": "model",
      "created": 1718000000,
      "owned_by": "higgs"
    }
  ]
}
```

**curl:**

```sh
curl http://localhost:11434/v1/models
```

---

### POST /v1/chat/completions

Chat with a model. **JIT (just-in-time) loading is ON by default** (a
runtime-toggleable flag — see `GET`/`PUT /api/higgs/settings`):

- **JIT on (default), model scanned-on-disk but not resident** — higgs loads it
  on demand and then serves the chat. higgs is **only-keep-last** (one model at a
  time), so a request for model `B` while `A` is resident swaps `A` out and ends
  with `B` resident. The JIT load goes through the same `Higgs::load()` path, so
  the RAM-headroom guard (`[HG017]`), charset guard (`[HG015]`), and
  path-within-roots guard (`[HG015]`) all still apply — a JIT load that fails
  surfaces the **real** mapped load error (`503 [HG017]`, `400`, worker spawn
  failure, etc.), **not** a silent `404`. A single INFO line is emitted per JIT
  load: `higgs: JIT loading {model} (was {prev})` (`prev` is `none` if nothing
  was loaded).
- **Concurrent JIT swap (resident-model binding)** — each chat is **bound to the
  model it resolved against**. The serve layer proves the model resident, then
  releases the lifecycle lock before dispatch; under concurrent load another
  request's JIT load can swap the resident model in that window (only-keep-last).
  The worker — the only place that knows the truly-resident id at generation
  time — compares the requested model to its loaded one and **refuses rather than
  serve the wrong model**: `503 [HG018] ResidentModelMismatch`, a **retryable**
  signal (the client's retry re-JITs the requested model). higgs never serves a
  chat with a model other than the one the request resolved to, and never
  silently 404s the swap.
- **JIT on, unknown model id** (not in the on-disk catalog) — `404 [HG002]
  ModelNotFound`. higgs never tries to load an id it hasn't scanned.
- **JIT off** — an unloaded model is `404 [HG003] ModelNotLoaded` (the explicit
  load-only behavior; load first via `POST /api/higgs/models/load`). Because
  higgs is spawn-on-load ("no worker" == "nothing loaded"), the idle state and a
  crashed-worker state both surface as this same `404` — the `/v1` surface never
  returns a worker-down `503`.

v1 is **text-only** — image, audio, and file content parts are rejected with 400.
Both `max_tokens` (deprecated) and `max_completion_tokens` are accepted; the newer field wins.

**Tools.** The OpenAI `tools` array is accepted and passed verbatim to the
model's GGUF chat template. When the model emits a tool call it is returned as
spec-shaped `tool_calls` with `finish_reason: "tool_calls"` — in both the
non-streaming and streaming responses. A malformed `tools` body is a 400.

**Validation & limits.** The request is checked before dispatch:

- **400** `[HG013]` — invalid sampling param (vllm `SamplingParams` ranges:
  `temperature >= 0`, `top_p` in `(0, 1]`, `presence_penalty` &
  `frequency_penalty` in `[-2, 2]`; `n` must be exactly `1` — higgs serves a
  single choice, so `n>1` is rejected), or `max_tokens` /
  `max_completion_tokens` above the cap of `32768`. **Note:** `top_p` and the
  penalties are range-validated for OpenAI compatibility but **not yet forwarded**
  to the sampler — only `temperature` and the max-token budget affect generation
  today (`stop` strings are likewise not applied; `seed` is a load-time param).
- **400** `[HG005]` — prompt exceeds the loaded model's context window. The
  serve layer early-rejects on a conservative estimate (`prompt_bytes / 4` +
  `max_tokens` vs `ctx_len`); the worker's exact tokenizer check is the
  authoritative backstop.
- **503** `[HG014]` — server busy: the inference admission gate (at most 8
  concurrent in-flight chat requests) is full. Retryable.
- **504** `[HG016]` — chat RPC timeout: a wedged-but-alive worker did not reply
  within the worker chat-RPC timeout (600 s). The HTTP layer never times the SSE
  stream itself; this bound lives one layer down in the supervisor.

**Request body (OpenAI wire):**

```json
{
  "model": "org/model-name",
  "messages": [
    { "role": "system", "content": "You are a helpful assistant." },
    { "role": "user", "content": "What is the weather in Paris?" }
  ],
  "stream": false,
  "max_completion_tokens": 256,
  "temperature": 0.7,
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
          "type": "object",
          "properties": { "city": { "type": "string" } },
          "required": ["city"]
        }
      }
    }
  ]
}
```

**Non-streaming response (200):**

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1718000001,
  "model": "org/model-name",
  "choices": [
    {
      "index": 0,
      "message": { "role": "assistant", "content": "4." },
      "finish_reason": "stop"
    }
  ],
  "usage": { "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0 }
}
```

**Non-streaming response with a tool call (200):**

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1718000001,
  "model": "org/model-name",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": null,
        "tool_calls": [
          {
            "id": "call_abc123",
            "type": "function",
            "function": {
              "name": "get_weather",
              "arguments": "{\"city\":\"Paris\"}"
            }
          }
        ]
      },
      "finish_reason": "tool_calls"
    }
  ],
  "usage": { "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0 }
}
```

**Streaming response (200, `stream: true`):**

```
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1718000001,"model":"org/model-name","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1718000001,"model":"org/model-name","choices":[{"index":0,"delta":{"content":"4"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1718000001,"model":"org/model-name","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

The stream always ends with `data: [DONE]`. On mid-stream errors the OpenAI error envelope is emitted as a `data:` event before `[DONE]`.

**Streaming with a tool call.** For **registry-matched** tool-call formats
(higgs's built-in parser families — Gemma, Qwen, GLM, DeepSeek, Mistral/Ministral,
Nemotron) the tool-call envelope
is suppressed from the content deltas; the structured call is sent as a single
delta (full name + arguments, not argument-by-argument) just before a finish chunk
whose reason is `tool_calls`. **Known limitation:** a format the underlying crate's
primary parser handles but the registry does **not** match (e.g. Llama-3's
`<|python_tag|>`) installs no stream filter, so its raw markup may appear in
content deltas — though the final structured `tool_calls` is still returned (and
non-streaming `content` is unaffected).

```
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk",...,"choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk",...,"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_abc123","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Paris\"}"}}]},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk",...,"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}

data: [DONE]
```

**curl (streaming):**

```sh
curl -N -H "Content-Type: application/json" \
  -d '{"model":"org/model","messages":[{"role":"user","content":"hi"}],"stream":true}' \
  http://localhost:11434/v1/chat/completions
```

---

## /api/higgs/* Control Routes

### GET /api/higgs/models

Live scan of all configured model directories plus the currently loaded model id.

**Response (200):**

```json
{
  "models": [
    {
      "id": "org/model-name",
      "path": "/home/user/.cache/lm-studio/models/org/model-name/model-Q4_K_M.gguf",
      "size_bytes": 4200000000,
      "quant": "Q4_K_M",
      "source": "LmStudio",
      "arch": "llama",
      "ctx_train": 131072,
      "has_chat_template": true,
      "supports_tools": true,
      "supports_reasoning": false,
      "gguf_components": [
        { "key": "general.architecture", "value": "llama" },
        { "key": "general.file_type", "value": "Q4_K_M" },
        { "key": "llama.context_length", "value": "131072" }
      ],
      "state": "loaded",
      "format": "gguf",
      "tool_calls": true
    }
  ],
  "loaded_id": "org/model-name"
}
```

Each entry is a `HiggsModelEntry`: all `HiggsModel` fields (`#[serde(flatten)]`d
to the top level) **plus** the control-computed `state` (`"loaded"` /
`"not-loaded"`), `format` (always `"gguf"`), and the support verdict below.
`source` is one of `"LmStudio"`, `"HfCache"`, `"Ollama"`.
`quant`, `arch`, `ctx_train` are omitted when unreadable from the GGUF header;
`support_reason` is present only when higgs can't parse the model's tool calls.

#### Model support detection fields

Each `HiggsModelEntry` carries a host-side tool-call support verdict plus the
curated GGUF header fields the verdict is computed from. The scan is **pure
host-side and fast** — it never loads a GGUF to test it. Engine loadability is
learned at **actual load time** (`POST /api/higgs/models/load` returns the
engine's verbatim error if the model fails to load); there is no scan-time
pre-flight verdict.

| Field | Type | Meaning |
|-------|------|---------|
| `tool_calls` | boolean | Whether higgs has a tool-call parser matching the model's chat template (host-side chat-template sniff, zero FFI, no worker). |
| `support_reason` | string (optional) | When `!tool_calls`: the fixed string `"no tool-call parser matches this model's template"`. **Omitted** when `tool_calls` is true. It never carries an engine load error. |
| `gguf_components` | `GgufComponent[]` | Curated load-relevant GGUF header fields (this field lives on the flattened `HiggsModel` — its single home). Each `GgufComponent = { key: string, value: string }`. |

The curated `gguf_components` keys are: `gguf.version`, `general.architecture`,
`general.file_type` (quant), `general.quantization_version`,
`tokenizer.ggml.model`, `tokenizer.ggml.pre`, `{arch}.context_length`,
`{arch}.block_count`, `{arch}.attention.head_count`. Giant arrays (token
lists/merges) are deliberately skipped so the UI can pin a support mismatch to a
specific component.

**Example — a model whose tool calls higgs can't parse** (no matching
tool-call parser for its chat template):

```json
{
  "id": "org/some-model",
  "path": "/home/user/.cache/lm-studio/models/org/some-model/model-Q4_K_M.gguf",
  "size_bytes": 4200000000,
  "quant": "Q4_K_M",
  "source": "LmStudio",
  "arch": "llama",
  "has_chat_template": true,
  "tool_calls": false,
  "support_reason": "no tool-call parser matches this model's template",
  "gguf_components": [
    { "key": "general.architecture", "value": "llama" },
    { "key": "general.file_type", "value": "Q4_K_M" }
  ]
}
```

**curl:**

```sh
curl http://localhost:11434/api/higgs/models
```

---

### POST /api/higgs/models/load

Load a model by HuggingFace repo id. This **spawns the worker process** (named
`higgs(<id>)`) — with nothing loaded there is no worker. The GGUF path is
resolved host-side and carried into the worker. Load parameters fall back to
`HiggsConfig.default_load` when absent.

**Request body:**

```json
{
  "id": "org/model-name",
  "ctx_len": 4096,
  "gpu_layers": 4294967295,
  "threads": 4,
  "use_mmap": true,
  "use_mlock": false,
  "n_batch": 2048,
  "n_ubatch": 512,
  "offload_kqv": true,
  "rope_freq_base": 10000.0,
  "rope_freq_scale": 1.0,
  "flash_attn": "auto",
  "type_k": "F16",
  "type_v": "F16",
  "seed": 42,
  "idle_ttl_minutes": 30
}
```

All fields except `id` are optional. Sending NO **engine load** field is a
fully-default load (`HiggsConfig.default_load` plus the auto context cap). The
moment any engine load field is present, the three base fields
(`ctx_len`/`gpu_layers`/`threads`) fall back to `default_load` and every other
field that is **absent** uses the engine default — i.e. omitting an optional
reproduces the pre-expansion behavior exactly. `idle_ttl_minutes` is **excluded**
from this check (it's a host-side reaper setting, not an engine param), so a
request carrying only `idle_ttl_minutes` still takes the fully-default load path.

`gpu_layers: 4294967295` (`u32::MAX`) means "offload all layers" (LM Studio "max" semantics).

`idle_ttl_minutes` is a **host-side per-load idle-TTL override** (NOT a load
param sent to the worker): when set, the idle reaper uses it instead of the
global `idle_ttl_minutes` for this loaded model. Cleared on unload.

**Expanded load parameters** (each maps to one `llama-cpp-2` 0.1.139 call):

| Field             | Type                                            | Maps to (`llama-cpp-2`)                  | Default when absent                          |
| ----------------- | ----------------------------------------------- | ---------------------------------------- | -------------------------------------------- |
| `use_mmap`        | bool                                            | `LlamaModelParams::with_use_mmap`        | engine default                               |
| `use_mlock`       | bool                                            | `LlamaModelParams::with_use_mlock`       | engine default                               |
| `n_batch`         | number                                          | `LlamaContextParams::with_n_batch`       | `ctx_len.max(1)` (one-shot prefill)          |
| `n_ubatch`        | number                                          | `LlamaContextParams::with_n_ubatch`      | engine default                               |
| `offload_kqv`     | bool                                            | `LlamaContextParams::with_offload_kqv`   | engine default                               |
| `rope_freq_base`  | number                                          | `LlamaContextParams::with_rope_freq_base`| GGUF trained value                           |
| `rope_freq_scale` | number                                          | `LlamaContextParams::with_rope_freq_scale`| GGUF trained value                          |
| `flash_attn`      | `"auto"` \| `"off"` \| `"on"`                   | `with_flash_attention_policy` (-1/0/1)   | engine default                               |
| `type_k`          | `F32`\|`F16`\|`Q8_0`\|`Q5_1`\|`Q5_0`\|`Q4_1`\|`Q4_0` | `LlamaContextParams::with_type_k`   | F16                                          |
| `type_v`          | same as `type_k`                                | `LlamaContextParams::with_type_v`        | F16                                          |
| `seed`            | number                                          | `LlamaSampler::dist(seed)`               | fresh random seed per request                |

**Omitted / deferred:**

- **Unified KV cache** (`kv_unified`) — OMITTED: `llama-cpp-2` 0.1.139 exposes no
  safe setter for it. Not faked.
- **Max concurrent sequences** (`n_seq_max > 1`) — DEFERRED: requires reworking
  the single-sequence decode loop. Out of scope for this pass; sequence handling
  is unchanged.

The `id` is validated before anything is resolved or spawned:

- **400** `[HG015]` — invalid model id. `validate_repo_id` allows ASCII
  alphanumerics plus `_ - . / :` (mirroring ollama's byte-level name
  validation) and rejects empty, absolute paths, NUL, illegal chars, and any
  `..` path component. The resolved GGUF path must also canonicalize **inside**
  a configured scan dir (`path_within_roots`) — a symlink or `..` escape never
  reaches the worker FFI loader.
- **503** `[HG017]` — insufficient memory. Before spawning a worker, the load is
  refused if the model's GGUF file size exceeds `available_ram * 0.8`
  (`MEMORY_HEADROOM_FRACTION`, ollama's placement rule). Retryable: free memory
  or unload another model and try again.

**Response (200):**

```json
{ "status": "ok", "id": "org/model-name" }
```

**curl:**

```sh
curl -X POST -H "Content-Type: application/json" \
  -d '{"id":"org/model-name"}' \
  http://localhost:11434/api/higgs/models/load
```

---

### POST /api/higgs/models/unload

Unload the current model. This **kills the worker process** (kill-on-unload), so
after this there is no higgs worker until the next load.

The model is **also** auto-unloaded after the runtime `idle_ttl_minutes` (default
5 — `IDLE_UNLOAD_TTL`, ollama's `keep_alive` default) with no chat request — a
background idle reaper frees memory on an idle host through this same path. The
reaper reads `idle_ttl_minutes` and `auto_unload_idle` each tick (settable via
`PUT /api/higgs/settings`), so auto-unload can be turned off entirely or its TTL
changed without a restart. A **per-load override** — `idle_ttl_minutes` on the
`POST /api/higgs/models/load` body — takes precedence over the global TTL for the
currently-loaded model (host-side only, never sent to the worker); it is cleared
on unload. The reaper never unloads while a request is in flight, and any chat
resets the idle timer.

**Response (200):**

```json
{ "status": "ok" }
```

**curl:**

```sh
curl -X POST http://localhost:11434/api/higgs/models/unload
```

---

### GET /api/higgs/status

Live status snapshot. `worker_alive` is true iff an RPC round-trip to the worker succeeded.

**Response (200):**

```json
{
  "worker_alive": true,
  "loaded": {
    "id": "org/model-name",
    "ctx_len": 4096,
    "gpu_layers": 4294967295,
    "threads": 4,
    "arch": "llama",
    "quant": "Q4_K_M",
    "max_context_length": 131072,
    "size_bytes": 4200000000,
    "has_chat_template": true
  },
  "models_on_disk": 3
}
```

`loaded` is absent when no model is loaded (then `worker_alive` is also false —
no worker exists). `models_on_disk` and the `arch`/`quant`/`max_context_length`/
`size_bytes`/`has_chat_template` fields are computed **host-side** (the worker
holds no model catalog); the worker reports only
`id`/`ctx_len`/`gpu_layers`/`threads`.

**curl:**

```sh
curl http://localhost:11434/api/higgs/status
```

---

### GET /api/higgs/system

Host hardware, inference runtime, and the **read-only effective config**. This is
purely informational — there is **no mutating counterpart**.

**Response (200):**

```json
{
  "hardware": {
    "cpu_name": "Apple M3 Max",
    "arch": "aarch64",
    "cpu_cores": 16,
    "ram_total_bytes": 137438953472,
    "ram_used_bytes": 40000000000,
    "cpu_usage_percent": 12.5,
    "gpus": [
      {
        "name": "Metal",
        "description": "Apple M3 Max",
        "kind": "Gpu",
        "vram_total_bytes": 103079215104,
        "vram_free_bytes": 80000000000
      }
    ],
    "vram_total_bytes": 103079215104
  },
  "runtime": { "backend": "Metal", "engine": "llama.cpp", "version": "...", "binding": "..." },
  "config": {
    "bind_host": "127.0.0.1",
    "lmstudio_dirs": ["/home/user/.cache/lm-studio/models"],
    "hf_dirs": ["/home/user/.cache/huggingface/hub"],
    "ollama_dirs": ["/home/user/.ollama/models"],
    "default_load": { "ctx_len": 4096, "gpu_layers": 4294967295, "threads": 4 },
    "default_ctx_cap": 32768,
    "limits": {
      "max_body_bytes": 33554432,
      "control_timeout_secs": 120,
      "chat_timeout_secs": 600,
      "max_output_tokens": 32768,
      "max_concurrent_inference": 8,
      "memory_headroom_fraction": 0.8,
      "idle_unload_ttl_secs": 300
    }
  }
}
```

`config` (`HiggsServerConfig`) is built by `Higgs::server_config()` — a pure read,
no worker RPC. `bind_host` is always `"127.0.0.1"`; `default_ctx_cap` is `32768`
(the auto-context cap); `gpu_layers: 4294967295` (`u32::MAX`) means "offload all".

**`hardware.gpus` / `hardware.vram_total_bytes`** — the compute devices the
**worker** enumerated via ggml's own backend-device FFI
(`ggml_backend_dev_*`), gathered for the host the worker runs on. Each `GpuDevice`
carries `name`, `description`, `kind` (`Cpu` / `Gpu` / `Accel`), and device memory
(`vram_total_bytes` / `vram_free_bytes`). `vram_total_bytes` at the hardware level
sums only the **GPU** devices' totals (`0` when none). The host gathers this once
through a transient, crash-isolated sysinfo worker (M_SYSINFO; HG021 on failure →
empty list) and caches it; a failed gather still returns full hardware/runtime
without devices. `fits_vram(model_size, vram_total, memory_headroom_fraction)` is
a host-side VRAM-fit **helper** built from this (reuses the existing
`MEMORY_HEADROOM_FRACTION`) — it is not auto-applied on load; the enforced pre-load
check is the RAM headroom guard.

**curl:**

```sh
curl http://localhost:11434/api/higgs/system
```

---

### GET /api/higgs/logs

Developer-log snapshot tail. Useful for diagnosing load failures and llama.cpp
output. Lines come from the `LogBus` history ring, which holds both worker child
stderr **and** captured `higgs`-targeted serve-layer tracing events (e.g. request
lines like `higgs: GET /v1/models`). This is a one-shot tail — use
`/api/higgs/logs/stream` for live updates.

**Query:**

- `?n=200` — number of history lines (default 200).
- `?source=serve|worker|node:<node-id>:<worker-id>` — restrict to one origin:
  `serve` (the higgs control plane + its worker interactions), `worker` (the local
  model worker's own stderr), or a specific **remote worker** by its hub-local
  numeric `NodeId` and `WorkerId`. Omit for all merged. Each buffered line is
  tagged with its origin in the `LogBus`; the filter is applied server-side and the
  wire shape is unchanged (still `{ "lines": string[] }`).

**Response (200):**

```json
{
  "lines": [
    "llama_model_load: loading model from /path/to/model.gguf",
    "llama_model_load: model size = 4.20 GB"
  ]
}
```

**curl:**

```sh
curl "http://localhost:11434/api/higgs/logs?n=50"
```

---

### GET /api/higgs/logs/stream

Live developer-log stream over **Server-Sent Events** (`text/event-stream`). The
handler subscribes to the `LogBus` live broadcast **first**, then replays the last
`n` lines from the history ring, then streams new lines as they arrive — so no
line is missed between the snapshot and the live tap. Each log line is one SSE
`data:` frame. The stream feeds the Developer Logs terminal in the higgs pane.

It is registered on the **no-timeout** streaming router (like
`/v1/chat/completions`) so a long-lived stream is never aborted at the HTTP layer.
If the broadcast lags (a slow consumer drops lines), the handler emits a
`[log stream lagged — dropped N lines]` marker frame and keeps streaming.

**Query:** `?n=200` (lines to replay before going live; default 200) and
`?source=serve|worker|node:<node-id>:<worker-id>` (same origin filter as the
snapshot `GET /api/higgs/logs`).

**Response (200, `text/event-stream`):**

```
data: llama_model_load: loading model from /path/to/model.gguf

data: llama_model_load: model size = 4.20 GB

data: 2026-06-15 12:00:01 [INFO] higgs: GET /v1/models
```

**curl:**

```sh
curl -N "http://localhost:11434/api/higgs/logs/stream?n=50"
```

---

### GET /api/higgs/settings

Current **server-behavior** runtime settings. This namespace is distinct from
`/api/higgs/logs/settings` (developer-log toggles) and is designed to grow as
more server-behavior flags are added.

`jit_enabled` is an in-process runtime `AtomicBool` on the `Higgs` facade
defaulting to `true` — **not** persisted to disk or config. When `true`, a
`POST /v1/chat/completions` for a scanned-but-unloaded model triggers an
on-demand load (only-keep-last swap) before serving; when `false`, an unloaded
model is `404 [HG003] ModelNotLoaded`.

`auto_unload_idle` (`AtomicBool`, default `true`) and `idle_ttl_minutes`
(`AtomicU64`, default `5`, seeded from `IDLE_UNLOAD_TTL_MINUTES` / `IDLE_UNLOAD_TTL`)
are runtime-mutable atoms read by the idle reaper **each tick**. When
`auto_unload_idle` is `false` the reaper skips unloading entirely; otherwise it
uses `idle_ttl_minutes` minutes as the TTL. A change via `PUT` takes effect
without a restart.

`serving_enabled` (`AtomicBool`, default `true`) gates the `/v1` **inference**
surface. When `false`, `POST /v1/chat/completions` is refused at the boundary
with `503 [HG019] ServingDisabled` (before any loaded-model gate or worker RPC),
while every `/api/higgs/*` control route — including this one — stays reachable
so the server can be re-enabled. It does **not** unbind the listener.

**Response (200) — `HiggsRuntimeSettings`:**

```json
{ "jit_enabled": true, "auto_unload_idle": true, "idle_ttl_minutes": 5, "serving_enabled": true }
```

**curl:**

```sh
curl http://localhost:11434/api/higgs/settings
```

---

### PUT /api/higgs/settings

Set the server-behavior runtime settings. The body carries
`HiggsRuntimeSettings { jit_enabled, auto_unload_idle, idle_ttl_minutes, serving_enabled }`.

**Request body — `HiggsRuntimeSettings`:**

```json
{ "jit_enabled": false, "auto_unload_idle": true, "idle_ttl_minutes": 5, "serving_enabled": true }
```

**Response (200) — `HiggsOk`:**

```json
{ "status": "ok" }
```

**curl:**

```sh
curl -X PUT -H "Content-Type: application/json" \
  -d '{"jit_enabled":false,"auto_unload_idle":true,"idle_ttl_minutes":5,"serving_enabled":true}' \
  http://localhost:11434/api/higgs/settings
```

---

### GET /api/higgs/logs/settings

Current developer-log settings. All three are in-process **runtime** toggles —
**not** persisted to disk or config, so they reset to `false` on restart.
`verbose` and `show_log_fields` live on the `LogBus`; `log_incoming_tokens` lives
on the `Higgs` facade — all reached through `Higgs`.

- `verbose` gates the extra serve-layer **completion** log line on the chat path
  (off by default).
- `log_incoming_tokens` gates an extra serve-layer **incoming-prompt** line that
  logs prompt CONTENT (off by default). It is the explicit opt-in that overrides
  the redact-by-default policy (no prompt content at info) — see PUT below.
- `show_log_fields` includes a tracing event's **structured fields** in captured
  log lines (off by default = redact). Turning it on can surface structured field
  values — including prompt content — so it is an explicit opt-in like the above.

**Response (200):**

```json
{ "verbose": false, "log_incoming_tokens": false, "show_log_fields": false }
```

**curl:**

```sh
curl http://localhost:11434/api/higgs/logs/settings
```

---

### PUT /api/higgs/logs/settings

Set the developer-log settings. The body carries **all** flags and each is set,
so toggling one preserves the others.

With `verbose: true`, every completed `POST /v1/chat/completions` (streaming and
non-streaming) emits ONE extra INFO line on the `higgs` tracing target — `higgs:
served <model> — <N> tok, finish=<reason>, <ms>ms` — which the `HiggsLogLayer`
mirrors into the Developer Logs.

With `log_incoming_tokens: true`, every chat request emits ONE extra INFO line —
`higgs: incoming <model> — <N> chars: <preview>` — carrying the flattened
incoming prompt CONTENT, capped to the first 800 chars (`…` suffix when longer)
so one request can't flood the log ring. **This is an explicit opt-in that logs
prompt content, overriding the otherwise redact-by-default policy** (no prompt
content at info). Default `false`, so redaction is unchanged unless turned on.

With both `false` (the default) only the existing request-entry line (`higgs:
POST /v1/chat/completions`) appears.

`show_log_fields: true` likewise includes tracing events' structured field values
in captured lines (default `false` = redact), which can include prompt content.

**Request body:**

```json
{ "verbose": true, "log_incoming_tokens": true, "show_log_fields": true }
```

**Response (200):**

```json
{ "status": "ok" }
```

**curl:**

```sh
curl -X PUT -H "Content-Type: application/json" \
  -d '{"verbose":true,"log_incoming_tokens":true,"show_log_fields":true}' \
  http://localhost:11434/api/higgs/logs/settings
```

---

### POST /api/higgs/worker/stop

Gracefully shut down the worker: a best-effort `higgs/shutdown` RPC (2 s timeout),
then wait up to `WORKER_EXIT_TIMEOUT` (5 s) for the process to exit before a
force-kill (SIGKILL). There is no `/worker/start`
counterpart — loading a model **is** the start (it spawns the worker), and
unloading **is** the stop. Use `POST /api/higgs/models/load` to bring a worker up.

**Response (200):**

```json
{ "status": "ok" }
```

**curl:**

```sh
curl -X POST http://localhost:11434/api/higgs/worker/stop
```

---

### GET /api/higgs/version

Build + engine versions — no worker needed.

```json
{ "higgs": "0.1.0", "engine": "llama.cpp", "engine_version": "…", "binding": "0.1.139", "supported_formats": ["gguf"] }
```

---

### GET /api/higgs/models/{*id}

One model's details (the `{*id}` wildcard captures the full `org/model` slash
path), including support detection. The handler runs a **fresh scan** per request,
so `404 [HG002]` means the id is absent from the current on-disk catalog.

---

### GET /api/higgs/nodes

The remote-fleet view (hub mode). Returns one entry per node the hub has ever
admitted — connected or not — with its stable id, connection state, hardware
snapshot, and resident workers. Empty when not running as a hub.

```json
[
  {
    "node_id": 1,
    "endpoint_id": "…",
    "connected": true,
    "inventory": { "hostname": "…", "os": "linux", "workers": [ { "worker_id": 1, "model": "org/model" } ], "hardware": { … }, "runtime": { … } }
  }
]
```

---

### POST /api/higgs/nodes/load · /api/higgs/nodes/unload

Load or unload a model on a paired node (hub mode). `load` records a durable route
so subsequent `/v1/chat/completions` for that model go to the node automatically.

```jsonc
// POST /api/higgs/nodes/load   { "node": "<endpoint-id>", "model": "org/model" }
//   → { "status": "ok", "worker_id": 1 }
// POST /api/higgs/nodes/unload { "model": "org/model" }
//   → { "status": "ok" }
```

---

### POST /api/higgs/pair

Mint a single-use node-pairing token (hub mode). Returns a connection ticket, the
token, and a ready-to-run node command. `409 Conflict` when the server isn't a hub
— as do the mutating `nodes/load` and `nodes/unload` routes. (`GET /api/higgs/nodes`
instead returns an empty `[]` when not a hub.)

```json
{ "hub_id": "…", "ticket": "…", "token": "…", "node_command": "higgs --node <ticket> <token>" }
```

See [Remote Fleet](/remote-fleet/) for the full pairing flow.

---

## Serve-layer hardening

Every request — both surfaces — passes through a shared middleware stack
(`higgs::serve::router`) following established ollama/vllm practice. When
[API keys](/system-design/#securing-the-api) are configured, a Bearer-token check
runs **in addition** to these layers (missing/insufficient token → `401`); with no
keys the surface is unauthenticated and these layers are its defense for a loopback
deployment.

| Guard | Mechanism | On violation |
|-------|-----------|--------------|
| **Host header** | DNS-rebinding guard: the `Host` header's host portion (sans `:port`) must be loopback — any IPv4 in `127.0.0.0/8` (`127.*`), `localhost`, or IPv6 `::1` / `[::1]`. A missing Host is also rejected (ollama behavior). | **403** `[HG012] forbidden host: <host>` |
| **Body size** | `MAX_BODY_BYTES` = 32 MB via axum `DefaultBodyLimit`. | **413** |
| **Control timeout** | `CONTROL_TIMEOUT` = 120 s `tower_http::timeout::TimeoutLayer`, applied **only** to `/api/higgs/*` and `/v1/models` — **except** the SSE routes (`/v1/chat/completions`, `/api/higgs/logs/stream`), which are on the no-timeout router. | **408** |
| **Panic recovery** | `tower_http::catch_panic::CatchPanicLayer` — a handler panic is caught and rendered. | **500** (connection survives) |

**Why the timeout skips chat streaming.** `POST /v1/chat/completions` is **not**
under `TimeoutLayer` — an SSE generation must never be aborted at the HTTP layer.
Its duration is bounded separately by the worker chat-RPC timeout.

**Host-guard 403 body (both surfaces):**

The Host guard is middleware that runs **before** route handling, so it returns
the same response on `/v1` and `/api/higgs/*`: a `403` with a **plain-text** body
(not a JSON envelope) —

```text
[HG012] forbidden host: evil.example.com
```
