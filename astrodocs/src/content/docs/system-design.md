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
│  diagnostic  HiggsError HG001–HG020       │
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
| `Higgs::new(config)` | Construct without spawning the worker (builds its own internal `LogBus` — worker stderr only, no serve-event capture) |
| `Higgs::with_log_bus(config, bus)` | Construct sharing a caller-supplied `Arc<LogBus>` — used when the caller also installs `HiggsLogLayer` on its tracing subscriber so serve-layer events reach the dev logs |
| `start()` | Near-no-op — control surface only; spawns **no** worker |
| `stop()` | Kill the worker (2 s graceful timeout) + clear load-replay state |
| `scan()` | Host-side scan of model dirs (no worker), return `Vec<HiggsModel>` |
| `load(id, params?)` | Resolve GGUF path host-side, **spawn** the worker, send M_LOAD |
| `unload()` | M_UNLOAD then **kill** the worker |
| `status()` | `HiggsStatus { worker_alive, loaded, models_on_disk }` |
| `chat_stream(messages, max_tokens, temp)` | Returns `(delta_rx, outcome_handle)` |
| `events()` | Subscribe to `HiggsEvent` broadcast |
| `logs(n)` | Last n developer-log lines (`LogBus` snapshot) |
| `subscribe_logs()` | Live `broadcast::Receiver<String>` of new developer-log lines |

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
    fn probe(&self, path: &str) -> (bool, Option<String>);
    fn devices(&self) -> Vec<GpuDevice>;  // ggml backend-device enumeration
}
```

### Hardware enumeration — M_SYSINFO

`devices()` enumerates the host's compute devices through ggml's **own**
backend-device FFI (`ggml_backend_dev_count` / `_get` / `_name` / `_description`
/ `_type` / `_memory`), bound in `llama_cpp_sys_2` and called only from
`worker/engine/llamacpp.rs`. The worker is where the engine (and Metal) is linked,
so it is the correct host for this — engine-native and ready for future remote
workers. The **M_SYSINFO** RPC (`higgs/sysinfo`) replies `{ gpus: [GpuDevice, …] }`;
it is cheap, read-only, and never mutates the resident-model slot.

The host runs M_SYSINFO in a **separate, transient, crash-isolated worker**
(`Supervisor::sysinfo`, same pattern as `probe_paths`), never the serving worker,
and `Higgs::sysinfo()` caches the result (hardware is static-ish). On
spawn/EOF/timeout the device list is empty (HG021) — `GET /api/higgs/system` still
returns hardware/runtime. The host-side `fits_vram(model_size, vram_total,
headroom)` decision is built from the summed GPU VRAM using the existing
`MEMORY_HEADROOM_FRACTION`.

v1 ships `LlamaCppEngine`. The trait exists so future backends (MLX, etc.) can be
swapped without touching the RPC or supervisor layers.

`chat` applies the **GGUF-embedded chat template** before sampling, so every model
uses its own prompt format without host-side template management.

Context fit-check runs before sampling: if `prompt_tokens + max_gen > n_ctx` the call
returns `HG005 ContextOverflow` instead of silently truncating or hanging.

### Load parameters (`LoadParams`)

`LoadParams` carries three always-present base fields (`ctx_len`, `gpu_layers`,
`threads`) plus a set of **optional** engine knobs. Each optional is `None` by
default, and `None` means "use the llama.cpp engine default" — so a quick-load
(only the base fields pinned) reproduces the original behavior exactly. Each
optional maps to one concrete `llama-cpp-2` 0.1.139 builder call, applied **only**
inside `worker/engine/llamacpp.rs` (the single file allowed to name
`llama_cpp_2` / `llama_cpp_sys_2`):

- **Model params** (in `load()`): `use_mmap` → `with_use_mmap`,
  `use_mlock` → `with_use_mlock`.
- **Context params** (in `run_decode()`): `n_batch` → `with_n_batch`
  (falls back to `ctx_len.max(1)`), `n_ubatch` → `with_n_ubatch`,
  `offload_kqv` → `with_offload_kqv`, `rope_freq_base` → `with_rope_freq_base`,
  `rope_freq_scale` → `with_rope_freq_scale`,
  `flash_attn` (`FlashAttn::{Auto,Off,On}` → -1/0/1) →
  `with_flash_attention_policy`, `type_k`/`type_v`
  (`KvCacheKind` → `llama_cpp_2::context::params::KvCacheType`) →
  `with_type_k`/`with_type_v`.
- **Sampler** (in `run_decode()`): `seed` threads into `LlamaSampler::dist(seed)`
  for reproducible generation; `None` keeps the per-request random seed.

The engine-agnostic enums `FlashAttn` and `KvCacheKind` live in
`worker/engine/mod.rs`; the `llama_cpp_2` types they map to never escape
`llamacpp.rs`.

**Omitted / deferred** (invent nothing): unified KV cache (`kv_unified`) is
OMITTED — 0.1.139 has no safe setter; multi-sequence (`n_seq_max > 1`) is
DEFERRED — it needs a decode-loop rework and is out of scope for this pass.

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

## Model Support Detection

Each scanned model carries a **two-gate support verdict** so the UI can show
exactly whether higgs can serve it — and, if not, the precise reason. The two
gates are independent: Gate 1 proves the engine can LOAD the model, Gate 2 proves
higgs can PARSE its tool calls.

```
HiggsModelEntry
  │
  ├─ loadable  ── Gate 1 ── engine CAN load (probe of (arch,quant))
  ├─ tool_calls ─ Gate 2 ── a tool-call parser matches the template
  └─ support_reason (optional)
        │  !loadable            → engine's VERBATIM load error
        │  loadable && !tool_calls → "no tool-call parser matches this model's template"
        └─ fully supported      → omitted
```

### Gate 1 — engine loadability (transient probe worker)

Loadability is proved by actually asking the engine to load the GGUF — but never
in the serving worker. A **separate transient probe worker** is spawned (re-exec
of the same binary, `production_factory(bus, "probe")`), crash-isolated exactly
like the serving worker. The probe loads the GGUF into a throwaway handle and
drops it immediately.

```
Higgs                              probe worker (re-exec, "probe")
  │  M_PROBE { path }                       │
  ├────────────────────────────────────────▶ LlamaModel::load_from_file(
  │                                          │     path, with_vocab_only(true))
  │                                          │  handle DROPPED immediately —
  │                                          │  NEVER stored as resident
  │  { loadable, reason, engine_version }    │
  ◀────────────────────────────────────────┤
  │
  └─ crash / timeout / spawn-fail ⇒ (false, Some("<context>"))   [HG020]
```

- The handle is **dropped immediately** and **never stored as resident**, so a
  probe never disturbs a model being served.
- The verbatim engine error string becomes `support_reason` when `loadable` is
  false.
- **Vocab-only caveat:** the pinned llama-cpp-2 0.1.139 lacks `no_alloc`. A
  `with_vocab_only(true)` probe validates the architecture/header but **cannot**
  catch a quant/tensor mismatch — that needs a later llama-cpp-2. So Gate 1 today
  proves "the engine recognizes this arch and header," not "every tensor loads."

### Verdict caching — per `(arch, quant, engine_version)`, not per file

A probe is expensive (a process spawn + a model load), so verdicts are cached by
the **combo**, not the file. One probe per distinct `(architecture, quant)`; every
model sharing that combo inherits the verdict.

```
key = (architecture, quant, engine_version)
        │
        ├─ hit  → reuse verdict, no probe
        └─ miss → spawn probe worker → cache (loadable, reason)

store: Mutex<HashMap<key, verdict>> on Higgs   (in-memory; no persistence yet)
engine_version: comes from the probe worker's OWN M_PROBE reply
                (the probing binary is the version source)
```

The `engine_version` component of the key is supplied by the probe worker itself
(its M_PROBE reply), so the key tracks the binary that actually performed the
probe.

### The M_PROBE RPC

```
request:  { path }
reply:    { loadable, reason, engine_version }

per-path timeout
crash / timeout / spawn-fail  ⇒  (false, Some("<context>"))
                                  never a panic, never a hang
                                  diagnostic HG020 ProbeWorkerFailed
```

### Gate 2 — tool-call parsing (host-side, zero FFI)

Gate 2 is pure Rust on the host — no worker, no FFI:

```
ToolParserRegistry::with_defaults()
     .select(chat_template)
     .is_some()          → tool_calls = true / false
```

It selects a parser from the model's chat template against the same
`ToolParserRegistry` the worker's fallback path uses (see **Tool calling**). A
match means higgs can turn that model's emitted calls into OpenAI `tool_calls`.

---

## Developer Logs (LogBus)

`LogBus` (`log_bus.rs`) is the **single home** for every developer-log line. Two
sources feed it; two endpoints read it.

```
SOURCES                            LogBus                       ENDPOINTS
  worker child stderr ──┐      ┌──────────────────────┐
   (supervisor reader   ├─push─▶ history ring (2000)   │──snapshot(n)──▶ GET /api/higgs/logs
    task → bus.push)    │      │   (snapshot source)  │
                        │      ├──────────────────────┤
  HiggsLogLayer ────────┘      │ broadcast tap (256)  │──subscribe──▶ GET /api/higgs/logs/stream
   (tracing layer, target      └──────────────────────┘                (SSE: subscribe first,
    starts "higgs")                                                      replay last n, then live)

LogBus::push(line) appends to the ring AND sends on the broadcast — one place,
both sinks fed.
```

Before this, only worker stderr reached the dev logs. `HiggsLogLayer` is a
`tracing_subscriber::Layer` that captures events whose target starts with
`higgs`, formats each as `YYYY-MM-DD HH:MM:SS [LEVEL] message`, and pushes it
into the bus — so serve-layer request events (e.g. `higgs: GET /v1/models`) now
appear in the Developer Logs. Only the tracing `message` field is captured (no
structured fields, no prompt content — redaction-safe).

### Verbose serve logging (runtime toggle)

Serving logs are gated behind a runtime **verbose** toggle — an `AtomicBool` on
the `Higgs` facade, default `false`, **not** persisted to disk/config, flipped via
`PUT /api/higgs/logs/settings` (read back with `GET /api/higgs/logs/settings`).

When verbose is ON, every completed chat completion on `POST
/v1/chat/completions` — **both** the streaming and non-streaming paths — emits one
extra INFO line on the `higgs` tracing target, so `HiggsLogLayer` mirrors it into
the Developer Logs:

```
higgs: served org/model — 12 tok, finish=length, 1234ms
```

When verbose is OFF (the default), nothing extra is logged — only the existing
request-entry line (`higgs: POST /v1/chat/completions`) appears.

### Log incoming tokens (runtime toggle — opt-in prompt content)

A second `AtomicBool` on the `Higgs` facade, `log_incoming_tokens` — default
`false`, **not** persisted, set via the same `PUT /api/higgs/logs/settings`
(which carries both flags). When ON, every chat request emits one extra INFO line
on the `higgs` target carrying the flattened incoming prompt CONTENT, capped to
the first 800 chars (`…` suffix when longer) so one request can't flood the log
ring:

```
higgs: incoming org/model — 42 chars: hello there
```

This deliberately logs prompt **content** and is the **explicit opt-in that
overrides the redact-by-default policy** (the boundary otherwise keeps prompt
content out of the logs). Default `false`, so the redaction-by-default posture is
unchanged unless the user turns this on. The line fires after the loaded-model +
content gate (only valid requests are logged), before dispatch. When OFF, nothing
extra is logged.

Wiring: the caller creates the `LogBus` **before** the tracing subscriber,
installs `HiggsLogLayer::new(bus.clone())` on it, and passes the same
`Arc<LogBus>` to `Higgs::with_log_bus(config, bus)`. `Higgs::new` instead builds
its own internal bus (worker stderr only). The supervisor holds the bus
(`Supervisor::spawn(bus)`); its stderr reader calls `bus.push`, `logs(n)`
delegates to `bus.snapshot(n)`, and `subscribe_logs()` returns `bus.subscribe()`.
The SSE handler handles broadcast `Lagged` gracefully (warns, emits a marker
line, keeps streaming).

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
  │    ├─ Arc::new(Higgs::with_log_bus(HiggsConfig::default(), bus))  ← facade,
  │    │     no worker; shares the process-global LogBus whose HiggsLogLayer
  │    │     main.rs already installed (via jigglebot_server::higgs::log_layer())
  │    │     so serve-layer events reach the Developer Logs
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
pre-built pool keeps the old model id. With JIT on (default), a chat carrying
that stale id re-loads it on demand (only-keep-last swap) — undoing the
pane-driven load; with JIT off, the chat fails (`[HG003]`) until the provider is
re-selected (which re-syncs + rebuilds). Auto-rebuilding attached pools on a
pane-driven load is a future enhancement.

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
loaded model after the idle TTL with no inference, freeing memory on an idle
host. Every `IDLE_REAP_INTERVAL` (30 s) it reads two **live runtime atoms** off
the `Higgs` facade each tick: `auto_unload_idle` (default `true`) and
`idle_ttl_minutes` (default `5`, seeded from `IDLE_UNLOAD_TTL_MINUTES` /
`IDLE_UNLOAD_TTL` — ollama's `keep_alive` default). When `auto_unload_idle` is
`false` it never reaps; otherwise it compares `now − last_activity` against the
effective TTL. The effective TTL is the **per-load override**
(`loaded_idle_ttl_override`, set from the `idle_ttl_minutes` field on the load
request and cleared on unload — host-side only, never sent to the worker) when
present, else the global `idle_ttl_minutes`. Both globals are settable via
`PUT /api/higgs/settings`, so a Server-Settings change takes effect without a
restart (the const remains only as the default seed). `last_activity` is stamped
at the top of every `chat_stream`.
The reaper never unloads
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
- **`POST /v1/chat/completions`** routing depends on the **JIT flag**
  (`jit_enabled`, ON by default — see below):
  - **JIT on, model scanned-but-unloaded** → higgs loads it on demand
    (only-keep-last swap) and serves. A failed JIT load surfaces the real mapped
    load error (`503 [HG017]`, `400`, spawn failure), **not** a silent `404`.
  - **JIT on, unknown id** (not in the on-disk catalog) → `404 [HG002]
    ModelNotFound` — higgs never tries to load an id it hasn't scanned.
  - **JIT off** → an unloaded model is `404 [HG003] ModelNotLoaded`. Because "no
    worker" == "nothing loaded", the idle state and a crashed-worker state both
    fall through this same `404` gate — the `/v1` surface **never** emits a
    worker-down `503`.

This is a deliberate decision: the `/v1` surface answers strictly in OpenAI
terms. The control surface still tells the truth about the worker — `GET
/api/higgs/status` exposes `worker_alive` for diagnostics.

### JIT (just-in-time) loading

JIT is a runtime `AtomicBool` on the `Higgs` facade, **ON by default** and
toggled via `PUT /api/higgs/settings` (`HiggsRuntimeSettings { jit_enabled }`),
read via `GET /api/higgs/settings`. It is **not** persisted to disk/config — it
resets to `true` on restart.

```
POST /v1/chat/completions, model not resident
  │
  ├─ jit_enabled == false ──────────────────► 404 [HG003] ModelNotLoaded
  │
  └─ jit_enabled == true
       ├─ model in on-disk catalog ──► Higgs::load(model)   ← only-keep-last:
       │                                  swaps out any resident model,
       │                                  same load() path → RAM-headroom (HG017),
       │                                  charset/path guards (HG015) all apply;
       │                                  a failed load surfaces the real error
       │                                  (503/400/…), NOT a silent 404.
       │                                  INFO: "higgs: JIT loading {model} (was {prev})"
       │                                  → then serve the chat normally
       │
       └─ unknown id (not scanned) ──► 404 [HG002] ModelNotFound
```

**Resident-model binding (concurrent-swap safety).** The serve layer proves the
requested model resident (loading it if needed), then releases the `lifecycle`
lock before dispatching the chat. Under concurrent load, another request's JIT
load can swap the resident model out in that window (only-keep-last). To prevent
serving the **wrong** model, each chat is bound to the model it resolved
against: the resolved id rides the M_CHAT params, and the worker — the only
layer that knows the truly-resident id at generation time — refuses with
`503 [HG018] ResidentModelMismatch` if they differ rather than generate. HG018 is
**retryable**: the client's retry re-JITs the requested model. This does not
serialize whole generations under the lifecycle lock (which would block all
chats during a generation) — the swap-thrash under alternating-model load is the
accepted only-keep-last tradeoff; the binding check only guarantees correctness.

`/v1/models` is unchanged — it always lists **loaded** models only, never the
JIT-reachable on-disk catalog. This server-behavior namespace
(`/api/higgs/settings`) is separate from the developer-log toggles
(`/api/higgs/logs/settings`) and is designed to grow more flags over time.

### Serving on/off

`serving_enabled` is a runtime `AtomicBool` on the `Higgs` facade, **ON by
default**, toggled via `PUT /api/higgs/settings` and read via `GET`. higgs's
listener is nested in the gateway, so "off" cannot unbind it; the honest meaning
is that the `/v1` **inference** endpoints refuse while the `/api/higgs/*` control
surface (status, models, settings, logs, load/unload) stays reachable so the
server can be re-enabled. `POST /v1/chat/completions` checks the gate **first**
— before the loaded-model gate, any JIT load, or any worker RPC — and returns
`503 [HG019] ServingDisabled` when off. `GET /v1/models` is **not** gated. Like
the other runtime flags it is not persisted and resets to `true` on restart.

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

The standalone `higgs` binary honors `HIGGS_BIND`. A **non-loopback**
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
