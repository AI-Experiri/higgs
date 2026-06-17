---
title: Endpoints
description: Every route exposed by higgs — /v1 OpenAI-compatible and /api/higgs/* control.
---

## Overview

higgs exposes two route groups:

| Group | Purpose |
|-------|---------|
| `/v1/*` | OpenAI-compatible inference — chat clients, OpenAI SDK drop-ins |
| `/api/higgs/*` | Control plane — scan, load, unload, status, system, logs (snapshot + SSE stream + verbose toggle), worker stop, version |

All routes are mounted by `higgs::serve::router(Arc<Higgs>)`.

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
| anything else | 500 |

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
client boundary (diagnostic code, model id, and prose are preserved). No prompt
CONTENT is logged at `info` on the `/v1` path. The full unredacted message (with
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

**Response (200):**

```json
{ "status": "ok" }
```

**curl:**

```sh
curl http://localhost:8081/health
curl http://localhost:8081/api/higgs/health
```

---

## /v1 Routes

### GET /v1/models

Returns the **loaded** model only. An empty list means no model is currently
loaded. Use `GET /api/higgs/models` for the full on-disk catalog.

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
curl http://localhost:8081/v1/models
```

---

### POST /v1/chat/completions

Chat with the loaded model. Requires a model to be loaded first (`POST
/api/higgs/models/load`). An unloaded or unknown model is a `404 [HG003]
ModelNotLoaded`. Because higgs is spawn-on-load ("no worker" == "nothing
loaded"), the idle state and a crashed-worker state both surface as this same
`404` — the `/v1` surface never returns a worker-down `503`.

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
  `max_completion_tokens` above the cap of `32768`.
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
        "content": "",
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

**Streaming with a tool call.** The tool-call envelope is suppressed from the
content deltas; the structured call is sent as a single delta (full name +
arguments, not argument-by-argument) just before a finish chunk whose reason is
`tool_calls`:

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
  http://localhost:8081/v1/chat/completions
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
      "has_chat_template": true
    }
  ],
  "loaded_id": "org/model-name"
}
```

`source` is one of `"LmStudio"`, `"HfCache"`, `"Ollama"`.
`quant`, `arch`, `ctx_train` are omitted when unreadable from the GGUF header.

**curl:**

```sh
curl http://localhost:8081/api/higgs/models
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
  "seed": 42
}
```

All fields except `id` are optional. Sending NO load field is a fully-default
load (`HiggsConfig.default_load` plus the auto context cap). The moment any
field is present, the three base fields (`ctx_len`/`gpu_layers`/`threads`) fall
back to `default_load` and every other field that is **absent** uses the engine
default — i.e. omitting an optional reproduces the pre-expansion behavior
exactly.

`gpu_layers: 4294967295` (`u32::MAX`) means "offload all layers" (LM Studio "max" semantics).

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
  http://localhost:8081/api/higgs/models/load
```

---

### POST /api/higgs/models/unload

Unload the current model. This **kills the worker process** (kill-on-unload), so
after this there is no higgs worker until the next load.

The model is **also** auto-unloaded after 5 minutes (`IDLE_UNLOAD_TTL`, ollama's
`keep_alive` default) with no chat request — a background idle reaper frees
memory on an idle host through this same path. The reaper never unloads while a
request is in flight, and any chat resets the idle timer.

**Response (200):**

```json
{ "status": "ok" }
```

**curl:**

```sh
curl -X POST http://localhost:8081/api/higgs/models/unload
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
curl http://localhost:8081/api/higgs/status
```

---

### GET /api/higgs/system

Host hardware, inference runtime, and the **read-only effective config**. This is
purely informational — there is **no mutating counterpart**.

**Response (200):**

```json
{
  "hardware": { "...": "host CPU / RAM / GPU" },
  "runtime": { "backend": "llama.cpp", "engine": "...", "version": "...", "binding": "..." },
  "config": {
    "bind_host": "127.0.0.1",
    "lmstudio_dirs": ["/home/user/.cache/lm-studio/models"],
    "hf_dirs": ["/home/user/.cache/huggingface/hub"],
    "ollama_dirs": ["/home/user/.ollama/models"],
    "default_load": { "ctx_len": 4096, "gpu_layers": 4294967295, "threads": 4 },
    "default_ctx_cap": 32768
  }
}
```

`config` (`HiggsServerConfig`) is built by `Higgs::server_config()` — a pure read,
no worker RPC. `bind_host` is always `"127.0.0.1"`; `default_ctx_cap` is `32768`
(the auto-context cap); `gpu_layers: 4294967295` (`u32::MAX`) means "offload all".

**curl:**

```sh
curl http://localhost:8081/api/higgs/system
```

---

### GET /api/higgs/logs

Developer-log snapshot tail. Useful for diagnosing load failures and llama.cpp
output. Lines come from the `LogBus` history ring, which holds both worker child
stderr **and** captured `higgs`-targeted serve-layer tracing events (e.g. request
lines like `higgs: GET /v1/models`). This is a one-shot tail — use
`/api/higgs/logs/stream` for live updates.

**Query:** `?n=200` (default 200 lines)

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
curl "http://localhost:8081/api/higgs/logs?n=50"
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

**Query:** `?n=200` (lines to replay before going live; default 200)

**Response (200, `text/event-stream`):**

```
data: llama_model_load: loading model from /path/to/model.gguf

data: llama_model_load: model size = 4.20 GB

data: 2026-06-15 12:00:01 [INFO] higgs: GET /v1/models
```

**curl:**

```sh
curl -N "http://localhost:8081/api/higgs/logs/stream?n=50"
```

---

### GET /api/higgs/logs/settings

Current developer-log settings. Both flags are in-process **runtime** toggles
(`AtomicBool`s on the `Higgs` facade) — **not** persisted to disk or config, so
they reset to `false` on restart.

- `verbose` gates the extra serve-layer **completion** log line on the chat path
  (off by default).
- `log_incoming_tokens` gates an extra serve-layer **incoming-prompt** line that
  logs prompt CONTENT (off by default). It is the explicit opt-in that overrides
  the redact-by-default policy (no prompt content at info) — see PUT below.

**Response (200):**

```json
{ "verbose": false, "log_incoming_tokens": false }
```

**curl:**

```sh
curl http://localhost:8081/api/higgs/logs/settings
```

---

### PUT /api/higgs/logs/settings

Set the developer-log settings. The body carries **both** flags and both are
set, so toggling one preserves the other.

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

**Request body:**

```json
{ "verbose": true, "log_incoming_tokens": true }
```

**Response (200):**

```json
{ "status": "ok" }
```

**curl:**

```sh
curl -X PUT -H "Content-Type: application/json" \
  -d '{"verbose":true,"log_incoming_tokens":true}' \
  http://localhost:8081/api/higgs/logs/settings
```

---

### POST /api/higgs/worker/stop

Gracefully shut down the worker (2 second timeout). There is no `/worker/start`
counterpart — loading a model **is** the start (it spawns the worker), and
unloading **is** the stop. Use `POST /api/higgs/models/load` to bring a worker up.

**Response (200):**

```json
{ "status": "ok" }
```

**curl:**

```sh
curl -X POST http://localhost:8081/api/higgs/worker/stop
```

---

## Serve-layer hardening

Every request — both surfaces — passes through a shared middleware stack
(`higgs::serve::router`) following established ollama/vllm practice. higgs has
**no auth**; these layers are the defense for a loopback no-auth server.

| Guard | Mechanism | On violation |
|-------|-----------|--------------|
| **Host header** | DNS-rebinding guard: the `Host` header's host portion (sans `:port`) must be loopback (`127.0.0.1` / `localhost` / `::1` / `[::1]`). A missing Host is also rejected (ollama behavior). | **403** `[HG012] forbidden host: <host>` |
| **Body size** | `MAX_BODY_BYTES` = 32 MB via axum `DefaultBodyLimit`. | **413** |
| **Control timeout** | `CONTROL_TIMEOUT` = 120 s `tower_http::timeout::TimeoutLayer`, applied **only** to `/api/higgs/*` and `/v1/models` — **except** the SSE routes (`/v1/chat/completions`, `/api/higgs/logs/stream`), which are on the no-timeout router. | **408** |
| **Panic recovery** | `tower_http::catch_panic::CatchPanicLayer` — a handler panic is caught and rendered. | **500** (connection survives) |

**Why the timeout skips chat streaming.** `POST /v1/chat/completions` is **not**
under `TimeoutLayer` — an SSE generation must never be aborted at the HTTP layer.
Its duration is bounded separately by the worker chat-RPC timeout.

**`/v1` 403 envelope (Host guard):**

```json
{
  "error": {
    "message": "[HG012] forbidden host: evil.example.com",
    "type": "invalid_request_error",
    "code": null
  }
}
```

The control surface returns the plain `{ "error": "[HG012] forbidden host: …" }`
envelope.
