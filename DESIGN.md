# higgs — Design

## Table of Contents
- [Crate Boundary](#crate-boundary)
- [Worker Lifecycle](#worker-lifecycle)
- [Chat Request Sequence](#chat-request-sequence)
- [Model Scan Flow](#model-scan-flow)
- [Endpoint Surface Map](#endpoint-surface-map)

---

## Crate Boundary

```
┌─────────────────────────────────────────────────────────────┐
│                         host app                            │
│  production: jigglebot server (backend/server/src/higgs/    │
│  launcher); also any Axum host in general                   │
│                                                             │
│  HiggsConfig { lmstudio_dirs, hf_dirs, ollama_dirs,        │
│                default_load }                               │
│                                                             │
│  let h = Arc::new(Higgs::new(config)); // construct, no worker│
│  serve(higgs::serve::router(h), 127.0.0.1:0) // ephemeral   │
│  h.start().await;               // near-no-op (no worker yet)│
└──────────────────────────┬──────────────────────────────────┘
                           │ Rust API only (launcher)
                           │ HTTP /api/higgs/* (frontend pane)
                           │ HTTP /v1 (agents, genai Higgs adapter)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                         higgs crate                         │
│  (standalone — zero dep edges to engine / common / server)  │
│  serve + control = PURE RUST (cannot crash the host)        │
│                                                             │
│  api.rs        Higgs facade: scan(host-side) / load /       │
│                unload / chat_stream / status / events / logs │
│  supervisor.rs Worker process manager + RPC correlator      │
│  serve/        Axum router (/v1 + /api/higgs/*)             │
│  rpc.rs        NDJSON JSON-RPC 2.0 codec                    │
│  worker/       Re-exec'd subprocess: engine (model store    │
│                empty — scan is host-side; path comes in M_LOAD)│
│  diagnostic.rs HiggsError HG001–HG011                      │
└──────────────────────────┬──────────────────────────────────┘
                           │ stdio (NDJSON JSON-RPC 2.0)
                           │ ONLY while a model is loaded
                           ▼
               ┌───────────────────────┐
               │   worker process      │  spawn-on-load /
               │   higgs(<model>)      │  kill-on-unload
               │   (re-exec'd binary   │  (zero process when
               │    --higgs-worker)    │   nothing loaded —
               │   llama.cpp engine    │   ISOLATED crash domain)│
               └───────────────────────┘
```

---

## Worker Lifecycle

```
  Higgs::start()  →  near-no-op: NO worker spawned (control surface only).
                     scan() runs host-side and needs no worker.

  ┌──────────────┬──────────────────────────────────────────────┐
  │ nothing      │  ZERO higgs worker processes (zero idle RAM). │
  │ loaded       │  chat for an unloaded model → 404 (no worker).│
  └──────────────┴──────────────────────────────────────────────┘

  load(M)   (lifecycle mutex held for the whole body)
        │
        ├─ scan() host-side → resolve M's GGUF path
        ├─ Supervisor::start_for(M)  → SPAWN worker process
        │     <binary> --higgs-worker, argv0 = higgs(<M>) (≤64 chars)
        │     (stdin/stdout piped; stderr → ring buffer, cap 2000)
        │     writer task owns stdin; reader task owns stdout
        ├─ send M_LOAD { id, path, ctx_len, gpu_layers, threads }
        │     fail → clear_last_load + stop() (tear worker back down)
        └─ ok   → record_last_load(params) + emit ModelLoaded

  unload()  (lifecycle mutex held for the whole body)
        │
        ├─ clear_last_load()  (so a racing respawn cannot replay it)
        ├─ send M_UNLOAD (best-effort graceful)
        ├─ Supervisor::stop()  → KILL worker process
        └─ emit ModelUnloaded

  reader loop dispatch (while a worker is live):
        ├─ Response  (has id) → correlate by id → oneshot reply
        └─ Notification N_CHAT_CHUNK → keyed chat sink (by request_id)

  UNEXPECTED DEATH (stdout EOF / read error, NOT a deliberate stop):
        └─ single respawn attempt (1 s backoff)
                  ├─ argv0 re-stamped higgs(<loaded id>); old child reaped
                  ├─ replay last load (M_LOAD, bounded RPC timeout)
                  │    → emit ModelLoaded on confirmed success
                  └─ emit WorkerRestarted
                       (scan is host-side — NO scan replay)

  TERMINAL: deliberate stop → no respawn, no event.
            factory failure on respawn → WorkerDied, no loop.
```

---

## Chat Request Sequence

```
  host / HTTP client
       │
       │  POST /v1/chat/completions  {model, messages, stream, max_tokens}
       ▼
  serve::mod  v1_chat_completions()
       │
       ├─── status check: is model loaded?
       │         HG003 → 404 if not
       │
       ├─── messages_to_pairs()   (v1 text-only; rejects image/audio → 400)
       │
       └─── Higgs::chat_stream(messages_json, max_tokens, temperature, tools_json)
                  │
                  └─── Supervisor::register_chat_sink(request_id)  (installs mpsc sink)
                       Supervisor::request_with_id(request_id, M_CHAT, params)
                            │ NDJSON line  →  worker stdin
                            │
                            ▼
                       WorkerState::dispatch  M_CHAT
                            │
                            └─── HiggsEngine::chat(messages, params, sink)
                                      │
                                      ├── fit-check: prompt_tokens + max_gen ≤ n_ctx
                                      │        HG005 → 400 if overflow
                                      │
                                      ├── GGUF-embedded chat template applied
                                      │
                                      └── llama.cpp sampling loop
                                               each decoded token:
                                               → N_CHAT_CHUNK notification  →  stdout
                                               → supervisor reader task  →  chat sink (mpsc)
                                               → SSE assemble task  →  "data:" line
                                               → HTTP client

                       final RPC response {content, finish_reason}
                            │
                            └── outcome JoinHandle resolves
                                 → stream::assemble sends finish chunk
                                 → "data: [DONE]"

  Non-streaming: drop(deltas), await outcome → single JSON body
```

---

## Model Scan Flow

```
  Higgs::scan()   — HOST-SIDE (pure Rust: ggus + memmap2 + std::fs,
        │            NO llama.cpp FFI, NO worker RPC). Wrapped in
        │            spawn_blocking. Available with no worker live.
        └─── ModelStore::scan(lmstudio_dirs, hf_dirs, ollama_dirs)
                  (the worker no longer scans; load() resolves the
                   GGUF path here and carries it in M_LOAD params)

  Three store layouts (read-only — higgs NEVER writes):

  LM Studio (< 0.3)          LM Studio (>= 0.3)
  ~/.lmstudio/models/        ~/.cache/lm-studio/models/
  └── org/                   └── org/
      └── model/                 └── model/
          └── *.gguf                 └── *.gguf
  id = parent-dir-based      id = parent-dir-based

  HuggingFace Hub
  ~/.cache/huggingface/hub/
  └── models--org--model/
      └── snapshots/
          └── <hash>/
              └── *.gguf
  id = org/model  (from dir name)

  Ollama
  ~/.ollama/models/
  ├── manifests/registry.ollama.ai/<name>/<tag>
  │     manifest JSON → resolves sha256 blob path
  └── blobs/sha256-<hash>   ← the GGUF file
  id = ollama/<name>:<tag>

  Per file (all three stores):
    ggus metadata mmap → arch, ctx_train, has_chat_template
    quant tag parsed from filename
    missing/unreadable GGUF → fields omitted (resilient scan)
    HG001 on unreadable root directory
```

---

## Endpoint Surface Map

```
  ┌─────────────────────────────────────────────────────────────────┐
  │  /v1  (OpenAI-compatible)                                       │
  ├──────────────────────────┬──────────────────────────────────────┤
  │  GET  /v1/models         │  Loaded models only (ListModelResponse)│
  │  POST /v1/chat/completions│  stream or non-stream               │
  │                          │  OpenAI error envelope on failure    │
  └──────────────────────────┴──────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────┐
  │  /api/higgs/*  (control)                                        │
  ├──────────────────────────┬──────────────────────────────────────┤
  │  GET  /api/higgs/models  │  Host-side scan + loaded_id         │
  │  GET  /api/higgs/models/{*id}  │  Single enriched model by id  │
  │  POST /api/higgs/models/load   │  Load by id (spawns worker)   │
  │  POST /api/higgs/models/unload │  Unload current (kills worker)│
  │  GET  /api/higgs/status  │  HiggsStatus {worker_alive, loaded, │
  │                          │              models_on_disk}         │
  │  GET  /api/higgs/system  │  Host CPU/RAM + inference runtime   │
  │  GET  /api/higgs/logs    │  Worker stderr tail (?n=200)        │
  │  POST /api/higgs/worker/stop  │  Graceful shutdown (2 s)       │
  │  GET  /api/higgs/version │  Build version + engine info        │
  └──────────────────────────┴──────────────────────────────────────┘
  (no /worker/start — load IS the start, unload IS the stop)

  Error mapping (same table for both surfaces):
  ┌────────────────────────────┬──────────┐
  │  HiggsError variant        │  HTTP    │
  ├────────────────────────────┼──────────┤
  │  HG002 ModelNotFound       │  404     │
  │  HG003 ModelNotLoaded      │  404     │
  │  HG005 ContextOverflow     │  400     │
  │  HG006 WorkerSpawnFailed   │  503     │
  │  HG007 WorkerDead          │  503     │
  │  anything else             │  500     │
  └────────────────────────────┴──────────┘

  /v1 errors:  {"error":{"message":"…","type":"…","code":"…"}}
  control errors: {"error":"[HGxxx] …"}
```
