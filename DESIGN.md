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
│  (jigglebot server — in phase 2; any Axum host in general)  │
│                                                             │
│  HiggsConfig { lmstudio_dirs, hf_dirs, ollama_dirs,        │
│                default_load }                               │
│                                                             │
│  let h = Higgs::new(config);    // facade construction      │
│  h.start().await;               // spawns worker + scan     │
│  Router::new().merge(higgs::serve::router(Arc::clone(&h)))  │
└──────────────────────────┬──────────────────────────────────┘
                           │ Rust API only
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                         higgs crate                         │
│  (standalone — zero dep edges to engine / common / server)  │
│                                                             │
│  api.rs        Higgs facade: scan / load / unload /         │
│                chat_stream / status / events / logs         │
│  supervisor.rs Worker process manager + RPC correlator      │
│  serve/        Axum router (/v1 + /api/higgs/*)             │
│  rpc.rs        NDJSON JSON-RPC 2.0 codec                    │
│  worker/       Re-exec'd subprocess: engine + model store   │
│  diagnostic.rs HiggsError HG001–HG011                      │
└──────────────────────────┬──────────────────────────────────┘
                           │ stdio (NDJSON JSON-RPC 2.0)
                           ▼
               ┌───────────────────────┐
               │   worker process      │
               │   (re-exec'd binary   │
               │    --higgs-worker)    │
               │   llama.cpp engine    │
               └───────────────────────┘

  Note: phase 2 integration (jigglebot provider upsert,
  WS events, config merge) is not yet wired — see astrodocs
  system-design page for the planned integration layer.
```

---

## Worker Lifecycle

```
  host calls Higgs::start()
        │
        ▼
  Supervisor::start()
        │
        ├─── spawn child process:  <binary> --higgs-worker
        │    (stdin piped ← supervisor writes requests)
        │    (stdout piped → supervisor reads responses/notifs)
        │    (stderr → ring buffer, cap 2000 lines)
        │
        ├─── writer task   owns stdin half
        │    drains mpsc channel → write_all + flush
        │
        └─── reader task   owns stdout half
             BufReader::lines() loop
                  │
                  ├─ Response (has id, no method)
                  │      → correlate by id → oneshot reply
                  │
                  └─ Notification (no id)  N_CHAT_CHUNK
                         → active chat sink (mpsc)

  NORMAL DEATH (stdout EOF):
        │
        ├─── stopped flag set? → do nothing (intentional stop)
        │
        └─── not stopped → single respawn attempt (1 s backoff)
                  │
                  ├─── replay last scan params  (higgs/scan)
                  ├─── replay last load params  (higgs/load)
                  │    → emit HiggsEvent::ModelLoaded on success
                  │
                  └─── emit HiggsEvent::WorkerRestarted
                           │
                           ├── factory fails → emit WorkerDied
                           │                    no further retry
                           └── replay fails  → HG009 logged;
                                               worker stays up
                                               (scan/load may fail)

  TERMINAL: factory failure on respawn → WorkerDied, no loop
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
       └─── Higgs::chat_stream(pairs, max_tokens, temperature)
                  │
                  └─── Supervisor::take_chat_sink()  (installs mpsc sink)
                       Supervisor::request(M_CHAT, params)
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
  Higgs::scan(lmstudio_dirs, hf_dirs, ollama_dirs)
        │
        └─── Supervisor::request(M_SCAN, {lmstudio:[…], hf:[…], ollama:[…]})
                  │ NDJSON  →  worker stdin
                  ▼
             WorkerState: ModelStore::scan(lmstudio, hf, ollama)

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
  │  GET  /api/higgs/models  │  Live scan + loaded_id              │
  │  POST /api/higgs/models/load   │  Load by id; body: HiggsLoadRequest│
  │  POST /api/higgs/models/unload │  Unload current model         │
  │  GET  /api/higgs/status  │  HiggsStatus {worker_alive, loaded, │
  │                          │              models_on_disk}         │
  │  GET  /api/higgs/logs    │  Worker stderr tail (?n=200)        │
  │  POST /api/higgs/worker/start │  Spawn worker                  │
  │  POST /api/higgs/worker/stop  │  Graceful shutdown (2 s)       │
  └──────────────────────────┴──────────────────────────────────────┘

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
