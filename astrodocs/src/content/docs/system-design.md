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

## How higgs works with jigglebot

> **Phase 2 integration — not yet wired.** The description below reflects the planned
> integration layer. Phase 1 (this crate) is fully standalone.

```
jigglebot host app (phase 2)
  │
  ├── boot.rs mounts higgs::serve::router() alongside the main gateway
  │
  ├── HiggsProviderAdapter (planned) converts HiggsEvent → ProviderConfig upsert
  │     ModelLoaded  → register higgs as a provider in ProviderRegistry
  │     ModelUnloaded → mark provider stale or remove
  │
  └── frontend WS bridge (planned) relays HiggsEvent to the UI via SessionEvent
        so the model selector reflects loaded/unloaded state in real time
```

The crate is designed so the host integration is entirely additive — higgs itself
changes nothing.

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
