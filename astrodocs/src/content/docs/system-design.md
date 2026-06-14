---
title: System Design
description: higgs architecture — crate API, worker model, engine trait, model scan, and host integration.
---

## Crate Boundary

higgs is a **standalone Rust crate** with zero dependency edges to any other jigglebot crate.
A host application constructs a `Higgs` facade, mounts the router, and talks to it through the Rust API or over HTTP.

```
┌──────────────────────────────────────────────┐
│                  host app                    │
│  Higgs::new(HiggsConfig)  →  h.start()       │
│  Router::new().merge(serve::router(h))        │
└───────────────────┬──────────────────────────┘
                    │ Rust API (zero-copy)
                    ▼
┌──────────────────────────────────────────────┐
│              higgs crate                     │
│  api.rs      Higgs facade                    │
│  supervisor  Worker manager + RPC correlator │
│  serve/      Axum router (/v1 + /api/higgs/) │
│  rpc.rs      NDJSON JSON-RPC 2.0 codec       │
│  worker/     Re-exec'd subprocess            │
│  diagnostic  HiggsError HG001–HG011          │
└───────────────────┬──────────────────────────┘
                    │ stdio (NDJSON JSON-RPC 2.0)
                    ▼
         ┌─────────────────────┐
         │   worker process    │
         │ <binary> --higgs-   │
         │         worker      │
         │  llama.cpp engine   │
         └─────────────────────┘
```

---

## Higgs Facade API

`Higgs` is the host-facing handle. All state lives in the `Supervisor` behind it.

| Method | What it does |
|--------|-------------|
| `Higgs::new(config)` | Construct without spawning the worker |
| `start()` | Spawn worker + run initial scan |
| `stop()` | Graceful shutdown (2 s timeout) |
| `scan()` | Scan model dirs, return `Vec<HiggsModel>` |
| `load(id, params?)` | Load a model by HuggingFace repo id |
| `unload()` | Unload the current model |
| `status()` | `HiggsStatus { worker_alive, loaded, models_on_disk }` |
| `chat_stream(messages, max_tokens, temp)` | Returns `(delta_rx, outcome_handle)` |
| `events()` | Subscribe to `HiggsEvent` broadcast |
| `logs(n)` | Last n worker stderr lines |

---

## Worker Model

The llama.cpp FFI runs inside a **re-exec'd copy of the host binary** launched with `--higgs-worker`.
The supervisor communicates with it over stdio using NDJSON JSON-RPC 2.0.

```
NORMAL DEATH (stdout EOF):
  stopped flag set?  → do nothing (intentional stop)
  not stopped        → single respawn (1 s backoff)
                           replay higgs/scan params
                           replay higgs/load params   → emit ModelLoaded
                           emit WorkerRestarted

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

```
ModelStore::scan(lmstudio_roots, hf_roots, ollama_roots)
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

The integration is live. jigglebot holds `Arc<Higgs>` in `AppState` and wires it
at three points in the server startup sequence.

### Boot sequence

```
lib.rs  init_app()
  │
  ├─ Higgs::new(config.higgs)     ← constructs facade, no worker yet
  │
  ├─ higgs.start().await          ← spawns worker subprocess + initial scan
  │    non-fatal: if worker can't start (llama.cpp missing)
  │    jigglebot continues without local model support
  │
  └─ AppState { …, higgs }        ← Arc<Higgs> stored in shared state

server.rs  run()  (after TCP bind — actual port now known)
  │
  ├─ spawn_higgs_provider_sync(&higgs, provider_registry, base_url)
  │    base_url = "http://127.0.0.1:{actual_port}"
  │    higgs_sync appends /v1 before storing as provider's base_url
  │
  └─ app.merge(higgs::serve::router(Arc<Higgs>))
       mounts /v1/*         (OpenAI-compatible chat completions, models list)
       mounts /api/higgs/*  (scan, load, unload, status, logs control routes)

main.rs
  │
  └─ Commands::HiggsWorker → higgs::worker::worker_main()
       dispatched BEFORE building the tokio runtime (worker_main is sync)
       this is the re-exec target: supervisor spawns the same binary with
       --higgs-worker and communicates via stdio NDJSON JSON-RPC 2.0
```

### Provider sync — HiggsEvent → ProviderRegistry

`spawn_higgs_provider_sync` subscribes to `higgs.events()` (a `broadcast::Receiver`)
and translates worker lifecycle events into `ProviderRegistry` mutations:

```
HiggsEvent::ModelLoaded { id }
  → ProviderConfig {
        backend:  ProviderBackend::Higgs,
        api_key:  "" (empty — no key for local inference),
        model:    id  (HuggingFace repo id verbatim),
        base_url: Some("http://127.0.0.1:{port}/v1"),
        max_concurrent: DEFAULT_MAX_CONCURRENT_LOCAL,
        …
    }
  → registry.add(config)      — runtime-only, NEVER persisted to disk
  → slug_map.insert(id, slug) — track for later removal

HiggsEvent::ModelUnloaded { id }
  → slug_map.remove(id) → registry.remove_provider(slug)
  (LM Studio eject semantics — provider disappears from the list entirely)

HiggsEvent::WorkerDied
  → for each tracked slug: registry.remove_provider(slug)
  → slug_map.drain()

HiggsEvent::WorkerRestarted
  → no-op (ModelLoaded events follow for each re-loaded model)
```

### Local = just a provider URL

Because higgs serves OpenAI wire at `/v1`, agents consume loaded models through the
standard `GenaiProvider` path — the same code path used for any OpenAI-compatible
cloud provider. The only difference is `ProviderBackend::Higgs`, which signals
`adapter_kind = OpenAI`, `no api_key`, and `base_url = gateway /v1`.

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
llama.cpp worker (via stdio RPC)
```

---

### Design decisions

Two deliberate choices that could have gone differently:

**1. ModelUnloaded = remove_provider, not retire**

When a model is unloaded (or the worker dies), `remove_provider` is called — the
provider entry disappears from the registry entirely. An alternative was `retire`,
which marks the provider stale but leaves a ghost entry. `retire` was rejected
because repeated load/unload cycles accumulate ghost entries: the provider list
grows unboundedly and stale-provider cleanup machinery (`check_all`, health polling)
runs on entries that can never recover until restart. `remove_provider` matches
LM Studio's eject semantics (the model is gone — its provider should be too) and
keeps the registry clean across arbitrary load/unload cycles.

**2. Higgs providers excluded from check_all() health polling**

`ProviderRegistry::check_all()` (the periodic health-check loop) does not poll
higgs-backed providers. Liveness is event-driven via `higgs_sync`: the supervisor
emits `WorkerDied` / `ModelUnloaded` when the model stops being available, and
`higgs_sync` removes the provider at that point. If `check_all` were to poll higgs
providers, genai's `all_model_names` call would hit `api.openai.com` instead of
the local endpoint (genai resolves OpenAI model names from a static list for the
`OpenAI` adapter kind). The higgs provider would be marked stale spuriously.
Event-driven removal is both correct and cheaper.

---

### Known limitation

`broadcast::Receiver` has a finite channel capacity (64). Under sustained rapid
model churn, a `RecvError::Lagged` could cause `higgs_sync` to miss a
`ModelUnloaded` event. If that happens, the provider lingers in the registry until
the next `WorkerDied` event clears the entire map. This is logged at `warn` level
(`higgs_sync lagged — some model events were missed`) but not yet reconciled: there
is no periodic re-sync between registry state and actual worker state.
Normal usage (one model loaded at a time, infrequent swaps) cannot trigger the lag.

---

## Embedding higgs in any host app

The same three integration points work for any Axum host:

1. **Construct and start** — `Higgs::new(config).start().await`
2. **Mount the router** — `Router::new().merge(higgs::serve::router(Arc::clone(&h)))`
3. **Subscribe to events** — `h.events()` returns a broadcast receiver; map
   `HiggsEvent::ModelLoaded` / `ModelUnloaded` to whatever your host's provider
   lifecycle expects

The host binary must also dispatch `--higgs-worker` to `higgs::worker::worker_main()`
so the re-exec pattern works:

```rust
if std::env::args().any(|a| a == "--higgs-worker") {
    higgs::worker::worker_main();
    return;
}
```
