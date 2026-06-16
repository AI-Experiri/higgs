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
│  diagnostic  HiggsError HG001–HG011          │
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

`Higgs` is the host-facing handle. All state lives in the `Supervisor` behind it.

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
from this same scan and carries it in the M_LOAD params; the worker's own store
stays empty.

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
