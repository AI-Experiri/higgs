---
title: System Design
description: higgs architecture — crate API, worker model, engine trait, model scan, and host integration.
---

## Crate Boundary

higgs is a **standalone Rust crate** with zero dependency edges to any other jigglebot crate.
A host application constructs a `Higgs` facade, mounts the router, and talks to it through the Rust API or over HTTP.

In production the host is the jigglebot server itself (`backend/server/src/higgs/`
launcher), which constructs `Higgs`, serves the router on an ephemeral
`127.0.0.1` port, and stores that origin in `config.higgs_base_url`.

```
┌──────────────────────────────────────────────┐
│                  host app                    │
│  Arc::new(Higgs::new(HiggsConfig))            │
│  serve(serve::router(h), 127.0.0.1:0)         │
│  h.start()  ← near-no-op: NO worker yet       │
└───────────────────┬──────────────────────────┘
                    │ Rust API (launcher only)
                    │ HTTP /api/higgs/* (pane), /v1 (agents)
                    ▼
┌──────────────────────────────────────────────┐
│              higgs crate                     │
│  api.rs      Higgs facade (scan = host-side) │
│  supervisor  Worker manager + RPC correlator │
│  serve/      Axum router (/v1 + /api/higgs/) │  PURE RUST
│  rpc.rs      NDJSON JSON-RPC 2.0 codec       │  (cannot crash
│  worker/     Re-exec'd subprocess            │   the host)
│  diagnostic  HiggsError HG001–HG017       │
└───────────────────┬──────────────────────────┘
                    │ stdio (NDJSON JSON-RPC 2.0)
                    │ ONLY while a model is loaded
                    ▼
         ┌─────────────────────┐
         │   worker process    │  spawn-on-load /
         │   higgs(<model>)    │  kill-on-unload
         │ <binary> --higgs-   │  (zero process when
         │         worker      │   nothing loaded —
         │  llama.cpp engine   │   ISOLATED crash domain)
         └─────────────────────┘
```

---

## Higgs Facade API

`Higgs` is the host-facing handle. State is **split** between the facade and the
`Supervisor` it wraps — neither owns all of it:

| Owner | State |
|-------|-------|
| `Higgs` (facade, `api.rs`) | the live `HiggsConfig`, the load/unload/stop **lifecycle mutex**, the **inference admission gate** (semaphore), the **`last_activity`** idle stamp |
| `Supervisor` (`supervisor.rs`) | the worker **process** handle, **RPC correlation** (`pending`), per-request **chat sinks**, the stderr **ring**, the **`last_load`** replay params, the `stopped`/`running` lifetime flags |

**Event ownership is split too** (deliberate, so each event is emitted exactly
once at its true origin):

- the **facade** emits `ModelLoaded` / `ModelUnloaded` after `load()` / `unload()`
  succeed (it drives those transitions),
- the **supervisor** emits `WorkerDied` / `WorkerRestarted` from its reader/restart
  path, and the post-restart `ModelLoaded` from the load **replay** (it owns the
  death→respawn lifecycle the facade never sees).

| Method | What it does |
|--------|-------------|
| `Higgs::new(config)` | Construct without spawning the worker |
| `start()` | Near-no-op — control surface only; spawns **no** worker |
| `stop()` | Kill the worker (2 s graceful timeout) + clear load-replay state |
| `scan()` | Host-side scan of model dirs (no worker), return `Vec<HiggsModel>` |
| `load(id, params?)` | Resolve GGUF path host-side, **spawn** the worker, send M_LOAD |
| `unload()` | M_UNLOAD then **kill** the worker |
| `status()` | `HiggsStatus { worker_alive, loaded, models_on_disk }` |
| `chat_stream(messages, max_tokens, temp)` | Returns `(delta_rx, outcome_handle)` |
| `events()` | Subscribe to `HiggsEvent` broadcast |
| `logs(n)` | Last n worker stderr lines |

---

## Worker Model

The llama.cpp FFI runs inside a **re-exec'd copy of the host binary** launched with `--higgs-worker`.
The supervisor communicates with it over stdio using NDJSON JSON-RPC 2.0.

The worker is **spawned on load and killed on unload** — there is no idle worker.
`start()` spawns nothing. `load(M)` spawns a worker (argv0 `higgs(<M>)`, ≤64
chars) then sends M_LOAD; `unload()` sends M_UNLOAD then kills it. With nothing
loaded, zero higgs worker processes exist. A failed load tears the worker back
down. A `tokio` lifecycle mutex serialises load/unload/stop so spawn-on-load and
kill-on-unload never interleave.

```
UNEXPECTED DEATH (stdout EOF / read error, NOT a deliberate stop):
  deliberate stop?  → do nothing, no event
  unexpected        → single respawn (1 s backoff)
                          argv0 re-stamped higgs(<loaded id>); old child reaped
                          replay higgs/load (bounded RPC timeout) → emit ModelLoaded
                          emit WorkerRestarted
                          (scan is host-side — NO scan replay)

FACTORY FAILURE on respawn → emit WorkerDied, no further retry
```

The mpsc channel between `Supervisor::request()` callers and the writer task serialises
concurrent requests onto the single stdin pipe — no mutex on the I/O path.

---

## Engine Trait

`HiggsEngine` is the seam between the worker loop and the inference backend.

```rust
trait HiggsEngine: Send {
    fn load(&mut self, path: &str, params: &LoadParams) -> Result<(), HiggsError>;
    fn unload(&mut self);
    fn is_loaded(&self) -> bool;
    fn chat(
        &mut self,
        messages: &[EngineMessage],
        params: &GenParams,
        sink: &mut dyn FnMut(&str),   // token callback
    ) -> Result<(String, &'static str), HiggsError>;
}
```

v1 ships `LlamaCppEngine`. The trait exists so future backends (MLX, etc.) can be
swapped without touching the RPC or supervisor layers.

`chat` applies the **GGUF-embedded chat template** before sampling, so every model
uses its own prompt format without host-side template management.

Context fit-check runs before sampling: if `prompt_tokens + max_gen > n_ctx` the call
returns `HG005 ContextOverflow` instead of silently truncating or hanging.

---

## Model Scan Flow

Scan runs **host-side** (pure Rust: `ggus` + `memmap2` + `std::fs`, no llama.cpp
FFI, no worker RPC), wrapped in `spawn_blocking`. The model list is therefore
available with no worker live. `load()` resolves the chosen model's GGUF path
from this same scan and carries it in the M_LOAD params; the worker holds no
model catalog of its own.

```
ModelStore::scan(lmstudio_roots, hf_roots, ollama_roots)   (host process)
  │
  ├── LM Studio (< 0.3)  ~/.lmstudio/models/org/model/*.gguf
  ├── LM Studio (>= 0.3) ~/.cache/lm-studio/models/org/model/*.gguf
  │     id = path-derived (org/model)
  │
  ├── HuggingFace        ~/.cache/huggingface/hub/
  │     models--org--model/snapshots/<hash>/*.gguf
  │     id = org/model  (from dir name; NOT dirs::cache_dir())
  │
  └── Ollama             ~/.ollama/models/
        manifests/…/<name>/<tag>  →  resolve sha256 blob path
        id = ollama/<name>:<tag>

  Per file: mmap GGUF header → arch, ctx_train, has_chat_template, quant
  Missing root: silently skipped
  Unreadable root: HG001 ModelDirUnreadable
```

---

## Tool calling

`/v1/chat/completions` accepts an OpenAI `tools` array and returns OpenAI
`tool_calls` (streaming and non-streaming). Both `/v1` surfaces stay strict
OpenAI — `tool_calls` are spec-shaped. higgs invents no tool-call format of its
own: the model side is the GGUF-embedded chat template, applied with tools via
`apply_chat_template_with_tools_oaicompat`.

### Tools flow (request → prompt)

```
POST /v1/chat/completions  { tools: [...] }   (async-openai wire)
        │  serve: serde_json::to_string(tools)  → JSON string (400 on bad body)
        ▼
chat_stream(..., tools_json)  →  M_CHAT RPC params
        │
        ▼
GenParams.tools_json: Option<String>          (worker boundary)
        │
        ▼
engine: apply_chat_template_with_tools_oaicompat(messages, tools_json)
        │  the GGUF template renders the tool grammar; the vendored
        ▼  common_chat selects its matching tool-call parser
   rendered prompt → decode loop
```

### Parse pipeline (generation → tool_calls)

```
full generated text
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│ PRIMARY   crate parse_response_oaicompat                     │
│           covers the families llama.cpp's common_chat        │
│           handles (Qwen, Mistral, Llama-3, Hermes, …)        │
└─────────────────────────────────────────────────────────────┘
      │ Ok → content + tool_calls          │ Err (crate declined)
      ▼                                     ▼
  done                          ┌───────────────────────────────────┐
                                │ FALLBACK  our ToolParserRegistry  │
                                │   parser selected by chat-template│
                                │   sniff — handles formats the     │
                                │   crate rejects (e.g. nemotron_h  │
                                │   <function=…><parameter=…> XML)  │
                                └───────────────────────────────────┘
                                   │ Some calls        │ no parser / no call
                                   ▼                   ▼
                              content + calls       RAW: return text verbatim,
                                                    no tool_calls (warn logged)
```

The registry is **engine-agnostic** — pure `&str → tool_calls`, no llama
dependencies — so a future MLX/CUDA engine reuses it unchanged. It ships 7
parsers, each owning one format family declared by the model's own GGUF chat
template: `xml-function` (Nemotron/Qwen3-Coder), `qwen-json`, `deepseek3`,
`glm-xml`, `mistral-bracket`, `gemma4`, `function-gemma`. Adding a format is one
`ToolCallParser` impl plus one line in `with_defaults()`. (Implementation:
`worker/tool_parser/` — see its README/DESIGN.)

### Streaming suppression

```
decode loop piece ──▶ ToolCallStreamFilter ──▶ content delta (SSE)
                         │  marker-aware:
                         │  • holds back a tail that could still grow into an
                         │    open marker (marker never split across pieces)
                         │  • once a full open marker is seen, suppresses the
                         │    envelope and the rest of the turn
                         ▼
                   tool-call envelope never leaks as assistant text

end of stream:  tool_calls parsed from the FULL generation (pipeline above)
                → emitted as a final SSE delta
                → finish chunk with finish_reason "tool_calls"
```

---

## How higgs works with jigglebot

higgs runs **embedded in-process** inside the jigglebot server. The only module
that imports the higgs crate is the launcher (`backend/server/src/higgs/`); it
owns the `Arc<Higgs>` and serves higgs's self-contained router on its own
localhost listener. jigglebot's gateway and event bus are never in higgs's path.

jigglebot reaches higgs by exactly three routes:

- the **library** (the launcher holds the `Arc<Higgs>`),
- the **frontend control surface** — the higgs pane hits `/api/higgs/*` directly,
- **agents** — through genai `AdapterKind::Higgs` over `/v1`.

The only jigglebot-side knowledge of the address is `config.higgs_base_url`.

### Read-only config surfacing

`GET /api/higgs/system` returns `SystemInfo { hardware, runtime, config }`, where
`config` is a `HiggsServerConfig` derived **read-only** from `HiggsConfig` plus the
`BIND_HOST` (`"127.0.0.1"`) and `DEFAULT_CTX_CAP` (`32768`) consts in `api.rs`. It
is produced by `Higgs::server_config()` — a pure read, no worker RPC, no mutation.
higgs exposes **no endpoint to change config**; the frontend higgs pane reads this
to display effective scan dirs, load defaults, and bind host, never to edit them.

### Boot sequence

```
init_app() (lib.rs)
  │
  config.higgs_base_url = higgs::launch().await   ← BEFORE provider load
  │    │
  │    ├─ Arc::new(Higgs::new(HiggsConfig::default()))   ← facade, no worker
  │    ├─ bind 127.0.0.1:0  → readback ephemeral addr
  │    ├─ spawn serve_with_shutdown(router)  (detached, owns Arc for the
  │    │     process lifetime — control surface up FIRST, survives a dead worker)
  │    ├─ higgs.start().await   ← near-no-op (best-effort; no worker spawned)
  │    └─ return "http://127.0.0.1:<port>"   (None on failure → jigglebot
  │         runs without embedded local models; non-fatal)
  │
  ProviderRegistry::load(config)  → seeds runtime-only higgs provider at base_url
  /api/meta                       → returns config.higgs_base_url = resolved addr

main.rs / tauri-app main()
  │
  └─ maybe_run_worker()   ← MUST be first, before tracing init
       if argv has --higgs-worker → higgs::worker::worker_main(); exit(0)
       (the re-exec target: load() spawns current_exe() with --higgs-worker;
        the worker owns stdin/stdout for NDJSON JSON-RPC — anything that writes
        stdout first corrupts the wire)
```

### Crash isolation

The serve + control layers are **pure Rust** and cannot crash jigglebot. Only the
worker (llama.cpp FFI) is a separate process, so a segfault there can never reach
jigglebot — the supervisor respawns it once, and the provider lifecycle reconciles
on the next switch/attach.

### Provider model sync — lazy, on switch/attach

There is **no event→registry sync task**. A single runtime-only `higgs` provider
(`ProviderBackend::Higgs`, fixed-UUID, never persisted) is seeded once at startup
pointing at `higgs_base_url`. Its `model` field starts empty; higgs's chat gate
rejects a request whose model isn't the loaded one. The gateway closes the gap by
**lazily syncing** that field from higgs's `GET /v1/models` (loaded-only) at the
points where a pool is (re)built:

```
provider switch / attach targeting higgs (WS ProviderSwitch, attach_provider,
  set_providers, handlers_agents) → sync_higgs_model_if_targeted(slugs)
boot restore of a saved higgs attachment → attach_provider_pool

  sync_higgs_model():  GET {higgs_base_url}/v1/models
                       → if loaded model changed, update()+rebuild the provider
```

Deliberately **NOT** on `GET /api/providers`: a read must not mutate runtime
state (`sync_higgs_model` can retire the entry, breaking an attached pool).
Source: `ProviderRegistry::sync_higgs_model` / `sync_higgs_model_if_targeted`
(`backend/server/src/provider/higgs.rs`).

### Local = just a provider URL

Because higgs serves OpenAI wire at `/v1`, agents consume loaded models through the
standard `GenaiProvider` path — the same code path used for any OpenAI-compatible
cloud provider. The only difference is `ProviderBackend::Higgs`, which signals
`adapter_kind = Higgs`, no api_key, and `base_url = {higgs_base_url}/v1`.

```
Agent actor (provider phase)
  │
  ▼
ProviderPool → GenaiProvider { backend: Higgs, base_url: "http://127.0.0.1:{port}/v1" }
  │
  ▼
POST /v1/chat/completions          ← higgs::serve handles this
  │
  ▼
llama.cpp worker (via stdio RPC; spawned on load)
```

### Mid-session swap limitation

If higgs is already attached to an agent and the user loads a *different* model
from the Higgs pane (which only POSTs to `/api/higgs/models/load`), that agent's
pre-built pool keeps the old model id and chat fails (`[HG003]`) until the
provider is re-selected (which re-syncs + rebuilds). Auto-rebuilding attached
pools on a pane-driven load is a future enhancement.

---

## Security / HTTP hardening

higgs has **no auth** — there are no API keys, no tokens, no sessions. The threat
model is therefore "a no-auth HTTP server on a shared host," and the entire serve
layer is built to keep that surface loopback-only. The hardening follows
established ollama/vllm practice; the layers wrap **every** request on both
surfaces.

```
request
  │
  ▼
┌─────────────────────────────────────────────────────────────┐
│ CatchPanicLayer        handler panic → structured 500        │
│                        (connection survives, never dropped)  │
├─────────────────────────────────────────────────────────────┤
│ Host guard (#1 defense) Host header host-part must be        │
│                        loopback (127.0.0.1/localhost/::1/    │
│                        [::1]); missing Host → reject too     │
│                        violation → 403 [HG012] ForbiddenHost │
├─────────────────────────────────────────────────────────────┤
│ DefaultBodyLimit       MAX_BODY_BYTES = 32 MB → 413          │
├─────────────────────────────────────────────────────────────┤
│ TimeoutLayer 120 s     ONLY /api/higgs/* + /v1/models        │
│                        NOT /v1/chat/completions (SSE)        │
└─────────────────────────────────────────────────────────────┘
  │
  ▼
router (/v1, /api/higgs/*, /health, /api/higgs/health)

PER-HANDLER GUARDS (beyond the shared stack above):

  POST /v1/chat/completions
    │  sampling validation (vllm ranges)  → 400 [HG013]
    │  max_tokens > 32768                 → 400 [HG013]
    │  prompt_bytes/4 + max_tokens > ctx  → 400 [HG005] (early)
    │  admission gate (Semaphore, ≤ 8)    → 503 [HG014] when full
    ▼  dispatch → M_CHAT (bounded by CHAT_RPC_TIMEOUT 600 s)
                                          → 504 [HG016] on timeout

  POST /api/higgs/models/load
    │  validate_repo_id + path_within_roots → 400 [HG015]
    │  RAM headroom guard: file_size > avail_ram*0.8 → 503 [HG017]
    ▼  spawn worker → M_LOAD
```

### Pre-load RAM headroom guard

Before spawning a worker, `load()` reads available system RAM (the same
`sysinfo` path that backs `GET /api/higgs/system`) and refuses any model whose
GGUF file size exceeds `available_ram * MEMORY_HEADROOM_FRACTION` (0.8). The
GGUF file size is a lower-bound proxy for resident weights; the remaining 20% is
headroom for the KV cache, compute buffers, and the rest of the system. A
refusal is `503 [HG017] InsufficientMemory` — a **retryable capacity** signal:
the user can unload another model or free memory and retry. Failing here (rather
than letting the worker OOM mid-load) turns an opaque `[HG004]`/`[HG006]` crash
into a clean, actionable error. The 0.8 fraction is ollama's placement rule
verbatim (`server/sched.go`: "Use 80% of free memory as threshold to leave
headroom").

### Idle auto-unload (keep_alive TTL)

`Higgs::start()` spawns an **idle reaper** background task that auto-unloads the
loaded model after `IDLE_UNLOAD_TTL` (5 minutes — ollama's `keep_alive` default)
with no inference, freeing memory on an idle host. Every `IDLE_REAP_INTERVAL`
(30 s) it compares `now − last_activity` against the TTL. `last_activity` is
stamped at the top of every `chat_stream`. The reaper never unloads
mid-generation: it unloads only when the inference admission semaphore is fully
open (zero in-flight requests) and a model is actually loaded, and it reuses the
ordinary `unload()` path (which takes the lifecycle mutex, so it serializes
against a concurrent load). It holds a `Weak<Higgs>`, so it self-terminates when
the host drops its `Arc<Higgs>`. The `Instant` is copied out from under the
`parking_lot` guard before any `.await`, honoring the never-hold-a-lock-across-
await rule.

### Log redaction on the /v1 boundary

The `/v1` surface is the untrusted OpenAI-interop boundary. No prompt CONTENT is
logged at `info` on that path — only the model id, the stream flag, and
lengths/ids. Every client-facing `/v1` error message is `redact_paths`-sanitized:
absolute filesystem paths and `host:port` bind addresses are replaced with
`<redacted>`, so the host's directory layout and listen address never leak in
error text. Diagnostic codes (`[HG004]`), model ids (`org/model`), and human
prose are preserved. The full unredacted `Display` (with paths) is still logged
**server-side at the origin** (four-pillar pillar 1) and returned verbatim on the
`/api/higgs/*` control surface — which is ours, not an interop boundary.

### DNS-rebinding Host guard — the #1 defense

For a loopback no-auth server the dominant attack is **DNS rebinding**: a page in
the user's browser resolves an attacker-controlled hostname to `127.0.0.1` and
fires requests at higgs with the browser's implicit trust. The guard defeats this
by requiring the request's `Host` header host portion (sans `:port`) to be a
loopback name — `127.0.0.1`, `localhost`, `::1`, or `[::1]`. A non-loopback or
**missing** Host is a `403 [HG012] ForbiddenHost` (matching ollama). The browser
cannot forge the `Host` header, so a rebound origin is blocked before it reaches
any handler.

### CORS is browser-only

CORS, like the Host guard, only constrains **browser** clients — the browser
enforces it. Neither protects a non-browser client (curl, a script, a process on
the network) that simply sets its own headers. The only real boundary for those
clients is the bind address: an ephemeral loopback listener (the embedded
jigglebot path) is unreachable off-box.

### Body limit & panic recovery

- **Body limit** — `MAX_BODY_BYTES` = 32 MB caps request bodies so a malformed
  or hostile client can't exhaust memory; oversized bodies are `413`.
- **Panic recovery** — `CatchPanicLayer` converts a handler panic into a
  structured `500` rather than a dropped connection, so one bad request can't
  take down the listener.

### Control timeout, but never the SSE stream

```
/api/higgs/* , /v1/models  ──▶ TimeoutLayer(120s)  ──▶ 408 if exceeded
/v1/chat/completions        ──▶ (no HTTP timeout)  ──▶ bounded by the
                                                        worker chat-RPC timeout
                                                        CHAT_RPC_TIMEOUT = 600 s
```

Control endpoints are bounded request/response, so a 120 s cap is safe. A chat
**completion is an SSE stream** — aborting it at the HTTP layer would truncate a
legitimate long generation mid-token. Its duration is bounded one layer down by
the worker chat-RPC timeout instead, so the HTTP layer never cuts the stream.

`CHAT_RPC_TIMEOUT` = 600 s (in `supervisor.rs`) bounds a single M_CHAT
(chat/inference) RPC round-trip — it is **the** layer that bounds streaming chat
duration, since the HTTP layer deliberately does not time `/v1/chat/completions`.
A wedged-but-alive worker that never replies trips this timeout, surfacing as
`HG016 ChatTimeout` → HTTP `504`.

### Chat request validation & limits

Before a chat request is dispatched, the serve layer applies a chain of
per-handler guards (the worker FFI loader is never reached on a rejection):

- **Sampling validation** — vllm `SamplingParams` ranges, checked at
  `/v1/chat/completions` **before** dispatch: `temperature >= 0` (finite),
  `top_p` in `(0, 1]`, `presence_penalty` & `frequency_penalty` in
  `[-2, 2]`, `max_tokens` in `[1, 32768]`. `n` must be exactly `1` — higgs
  serves a single choice, so `n>1` is **rejected** (not silently honored as one
  choice). Out of range → `HG013 InvalidSamplingParam` → `400`.
- **max_tokens cap** — `MAX_OUTPUT_TOKENS` = 32768. A request with `max_tokens`
  / `max_completion_tokens` above this → `HG013` → `400`.
- **Prompt-vs-context early reject** — a conservative estimate
  `prompt_bytes / 4` (const `PROMPT_BYTES_PER_TOKEN` = 4) plus `max_tokens` is
  compared to the loaded model's `ctx_len`. If it cannot fit → `HG005
  ContextOverflow` → `400`, rejected early at the serve layer. The serve layer
  has **no tokenizer**; the worker's exact tokenizer check remains the
  authoritative `HG005` backstop.

### /v1 idle & crashed-worker behavior (deliberate decision)

higgs is **spawn-on-load**, so the normal idle state has **no worker process**
(`status().worker_alive == false`). On the `/v1` (OpenAI-interop) surface this is
not an error condition:

- **`GET /v1/models`** always returns `200` with the loaded models as the list —
  an empty `{"data":[]}` when nothing is loaded. An empty list is the correct
  OpenAI answer for "nothing can serve chat right now"; `/v1/models` never gates
  on `worker_alive`.
- **`POST /v1/chat/completions`** for an unloaded or unknown model is `404
  [HG003] ModelNotLoaded`. Because "no worker" == "nothing loaded", the idle
  state and a crashed-worker state both fall through this same `404` gate — the
  `/v1` surface **never** emits a worker-down `503`.

This is a deliberate decision: the `/v1` surface answers strictly in OpenAI
terms (a model is loaded or it is not). The control surface still tells the truth
about the worker — `GET /api/higgs/status` exposes `worker_alive` for
diagnostics. A crashed worker therefore presents on `/v1` as an empty
`/v1/models` and a `404` chat, while `/api/higgs/status` shows
`worker_alive:false`.

### Inference admission gate

`MAX_CONCURRENT_INFERENCE` = 8 — a `tokio` Semaphore in the `Higgs` facade —
caps in-flight chat requests at 8. A full gate → `HG014 ServerBusy` → `503`
(a retryable capacity signal). It is scoped to the chat path only.

This is **distinct** from the deferred worker-slot `max_concurrent_requests`
design in `concurrency.md`: that is about true parallel execution *inside* the
worker; this is an HTTP-layer flood gate sitting *over* the still-single-sequence
worker. The two must not be conflated.

### Model-id validation on load

`POST /api/higgs/models/load` validates the repo id before anything else:

- **`validate_repo_id`** allows ASCII alphanumerics plus `_ - . / :` (mirrors
  ollama `types/model/name.go` byte-level validation). It rejects empty,
  absolute paths, NUL, illegal chars, and any `..` path component → `HG015
  InvalidModelId` → `400`.
- **`path_within_roots`** additionally requires the resolved GGUF path to
  canonicalize **inside** a configured scan directory (lmstudio / hf / ollama
  dirs), else `HG015` → `400`. A symlink or `..` escape therefore never reaches
  the worker FFI loader.

### /health readiness

`GET /health` and `GET /api/higgs/health` answer "is the server reachable?" with a
cheap `200` and **no worker RPC** — they report liveness of the control surface,
not whether a model is loaded (`/api/higgs/status` covers load state). They sit
behind the same hardening stack.

### Non-loopback bind: SECURITY WARNING

The standalone `higgs-server` binary honors `HIGGS_BIND`. A **non-loopback**
value (e.g. `0.0.0.0`) is allowed but logs a prominent startup **SECURITY
WARNING**: higgs has no auth, and the Host guard + CORS only protect *browser*
clients — any non-browser client on the network can reach the API directly. The
**embedded jigglebot path always binds an ephemeral `127.0.0.1` port** and never
takes this risk.

---

## Embedding higgs in any host app

The same integration points work for any Axum host:

1. **Construct** — `let h = Arc::new(Higgs::new(config));`
2. **Serve the router** — `serve(higgs::serve::router(Arc::clone(&h)), listener)`
3. **Start** — `h.start().await` (control only; the worker spawns on first load)
4. **Drive the lifecycle** — call `h.load(id, …)` / `h.unload()`; optionally
   subscribe to `h.events()` for `ModelLoaded` / `ModelUnloaded` / `WorkerDied` /
   `WorkerRestarted`

The host binary must also dispatch `--higgs-worker` to `higgs::worker::worker_main()`
so the re-exec pattern works:

```rust
if std::env::args().any(|a| a == "--higgs-worker") {
    higgs::worker::worker_main();
    return;
}
```
