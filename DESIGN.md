# higgs — Design

## Table of Contents
- [Crate Boundary](#crate-boundary)
- [Worker Lifecycle](#worker-lifecycle)
- [Chat Request Sequence](#chat-request-sequence)
- [Model Scan Flow](#model-scan-flow)
- [Developer Log Bus](#developer-log-bus)
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
│  worker/       Re-exec'd subprocess: engine only (no model   │
│                catalog — scan is host-side; path comes in M_LOAD)│
│  diagnostic.rs HiggsError HG001–HG016                      │
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
  │ loaded       │  chat for an unloaded model: JIT on (default) │
  │              │  → load+serve; JIT off → 404 (no worker).     │
  └──────────────┴──────────────────────────────────────────────┘

  Higgs::start() also SPAWNS the idle reaper task (holds Weak<Higgs>):
        every IDLE_REAP_INTERVAL (30 s) it reads two LIVE runtime atoms:
          auto_unload_idle (default true) — false ⇒ skip, never reap
          idle_ttl_minutes (default 5, seeded IDLE_UNLOAD_TTL_MINUTES/_TTL)
        if auto_unload_idle AND now − last_activity > idle_ttl_minutes min
        AND inference gate fully open (no in-flight) AND a model is loaded
        → unload() (reuses the unload path below). Atoms are PUT-settable
        (/api/higgs/settings) so changes apply without restart.
        Self-terminates when the host drops its Arc<Higgs>.

  load(M)   (lifecycle mutex held for the whole body)
        │
        ├─ scan() host-side → resolve M's GGUF path
        ├─ RAM headroom guard: refuse if M's file size > available_ram * 0.8
        │     → InsufficientMemory [HG017] → 503 (BEFORE spawning a worker)
        ├─ Supervisor::start_for(M)  → SPAWN worker process
        │     <binary> --higgs-worker, argv0 = higgs(<M>) (≤64 chars)
        │     (stdin/stdout piped; stderr → ring buffer, cap 2000)
        │     writer task owns stdin; reader task owns stdout
        ├─ send M_LOAD { id, path, <full LoadParams> }
        │     base: ctx_len, gpu_layers, threads
        │     optional overrides (absent ⇒ engine default ⇒ prior behavior):
        │       use_mmap, use_mlock, n_batch, n_ubatch, offload_kqv,
        │       rope_freq_base, rope_freq_scale, flash_attn (auto/off/on),
        │       type_k, type_v (KV cache GGML type), seed
        │     applied per-field via llama-cpp-2 0.1.139 builder calls in
        │     worker/engine/llamacpp.rs (only file naming llama_cpp_2)
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
       ├─── status check: is model resident?  (gate_and_validate)
       │      not resident → branch on jit_enabled (AtomicBool, default true):
       │        JIT on,  scanned id  → Higgs::load() (only-keep-last swap),
       │                                then serve. failed load → real mapped
       │                                error (503 HG017 / 400 / …), NOT 404.
       │                                INFO "higgs: JIT loading {m} (was {prev})"
       │        JIT on,  unknown id  → 404 HG002 ModelNotFound (never loads it)
       │        JIT off, unloaded    → 404 HG003 ModelNotLoaded — covers idle
       │                                (no worker) AND crashed-worker;
       │                                /v1 never emits a worker-down 503
       │
       ├─── validate_sampling()   (temp/top_p/n=1/penalties/max_tokens; vllm ranges)
       │         HG013 → 400 if out of range
       │
       ├─── check_prompt_fits()   (prompt_bytes/4 + max_tokens vs loaded ctx_len)
       │         HG005 → 400 early reject (worker tokenizer is the exact backstop)
       │
       ├─── messages_to_pairs()   (v1 text-only; rejects image/audio → 400)
       │
       └─── Higgs::chat_stream(messages_json, max_tokens, temperature, tools_json)
                  │
                  ├─── stamp last_activity = Instant::now()  (keeps idle reaper at bay)
                  │
                  ├─── inference_gate.try_acquire_owned()  (admission, max 8)
                  │         HG014 → 503 if full (permit rides the gen task)
                  │
                  └─── Supervisor::register_chat_sink(request_id)  (installs mpsc sink)
                       Supervisor::request_with_id(request_id, M_CHAT, params)
                            │ bounded by CHAT_RPC_TIMEOUT (600 s)
                            │   HG016 → 504 if a wedged worker never replies
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

  /v1 error envelope: every client-facing message is redact_paths()-sanitized —
  absolute filesystem paths + host:port bind addresses → "<redacted>". No prompt
  CONTENT is logged at info (only model id, stream flag, lengths/ids). The full
  Display (with paths) still logs server-side at the origin and returns verbatim
  on /api/higgs/* (the control surface is ours).
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

## Developer Log Bus

```
  TWO SOURCES                          LogBus (log_bus.rs)        TWO SINKS
  ───────────                          ─── SINGLE HOME ───        ─────────

  worker child stderr ──┐                                   ┌─ GET /api/higgs/logs
   (supervisor reader    │                                  │     snapshot(n)
    task: bus.push)      │      ┌──────────────────────┐    │     → ring tail (n lines)
                         ├─push─▶│ history ring         │────┤
  HiggsLogLayer ─────────┘      │   cap RING_CAP 2000   │    │
   tracing events whose         │   (snapshot source)  │    └─ GET /api/higgs/logs/stream
   target starts "higgs"        ├──────────────────────┤        SSE (no http timeout)
   formatted:                   │ broadcast::Sender    │        control_logs_stream:
   YYYY-MM-DD HH:MM:SS          │   <String>           │──live──▶ subscribe FIRST
   [LEVEL] message              │   cap BROADCAST_CAP  │          then replay last n ring
   (message field ONLY —        │       256 (live tap) │          then stream live lines
    no fields, no prompt =      └──────────────────────┘          one line = one data: frame
    redaction-safe)                                               Lagged → warn + marker,
                                                                  keep streaming
  Every line enters via LogBus::push(line):
    append to ring  +  send on broadcast   (one place, both sinks fed)

  subscribe_logs() → broadcast::Receiver        snapshot(n) → last n ring lines

  Construction: caller builds LogBus BEFORE the tracing subscriber, installs
  HiggsLogLayer::new(bus.clone()) on it, passes the same Arc to
  Higgs::with_log_bus(config, bus). Higgs::new builds its own internal bus
  (worker stderr only — no serve-event capture).
  Supervisor::spawn(bus) holds the bus; the stderr reader calls bus.push.

  DEV-LOG runtime toggles — LogSettings { verbose, log_incoming_tokens }:
    Higgs facade holds two AtomicBools (default false; NOT persisted).
    PUT /api/higgs/logs/settings carries BOTH and sets both; GET reads both.

    VERBOSE serve-completion line:
      verbose ON  → each completed POST /v1/chat/completions (stream + non-stream)
                    emits one extra INFO line on the `higgs` target →
                    HiggsLogLayer → Developer Logs:
                      higgs: served <model> — <N> tok, finish=<reason>, <ms>ms
      verbose OFF → only the existing entry line `higgs: POST /v1/chat/completions`.

    LOG INCOMING TOKENS (opt-in prompt content):
      log_incoming_tokens ON  → each chat request emits one extra INFO line on the
                    `higgs` target carrying the flattened incoming prompt CONTENT,
                    capped to 800 chars (`…` when longer); fires after the
                    loaded-model + content gate, before dispatch:
                      higgs: incoming <model> — <N> chars: <preview>
                    This is the EXPLICIT OPT-IN that overrides the redact-by-default
                    policy (no prompt content at info).
      log_incoming_tokens OFF → nothing extra; redaction-by-default intact.
```

---

## Endpoint Surface Map

```
  ┌─────────────────────────────────────────────────────────────────┐
  │  /v1  (OpenAI-compatible)                                       │
  ├──────────────────────────┬──────────────────────────────────────┤
  │  GET  /v1/models         │  Loaded models only (ListModelResponse)│
  │                          │  idle (no worker) → 200 {"data":[]}  │
  │  POST /v1/chat/completions│  stream or non-stream               │
  │                          │  not resident → JIT branch (default on):│
  │                          │   JIT on, scanned  → load+serve         │
  │                          │   JIT on, unknown  → 404 HG002          │
  │                          │   JIT off          → 404 HG003          │
  │                          │  OpenAI error envelope on failure    │
  └──────────────────────────┴──────────────────────────────────────┘
  /v1 NEVER returns 503 worker-down: spawn-on-load means "no worker" ==
  "nothing loaded", so (JIT off) a crashed worker presents as empty
  /v1/models + 404 chat. A failed JIT load surfaces the real mapped error
  (503 HG017 / 400 / …), not 404. GET /api/higgs/status exposes worker_alive.

  ┌─────────────────────────────────────────────────────────────────┐
  │  /api/higgs/*  (control)                                        │
  ├──────────────────────────┬──────────────────────────────────────┤
  │  GET  /api/higgs/models  │  Host-side scan + loaded_id         │
  │  GET  /api/higgs/models/{*id}  │  Single enriched model by id  │
  │  POST /api/higgs/models/load   │  Load by id (spawns worker)   │
  │  POST /api/higgs/models/unload │  Unload current (kills worker)│
  │  GET  /api/higgs/status  │  HiggsStatus {worker_alive, loaded, │
  │                          │              models_on_disk}         │
  │  GET  /api/higgs/system  │  SystemInfo {hardware, runtime,     │
  │                          │  config: HiggsServerConfig}(read-only)│
  │  GET  /api/higgs/settings│  HiggsRuntimeSettings {jit_enabled,  │
  │                          │  auto_unload_idle, idle_ttl_minutes}  │
  │                          │  (server-behavior ns; grows) (read) │
  │  PUT  /api/higgs/settings│  set all three (runtime atoms,       │
  │                          │  not persisted) → HiggsOk            │
  │  GET  /api/higgs/logs    │  Dev-log snapshot tail (?n=200)     │
  │  GET  /api/higgs/logs/stream │  SSE replay-then-live dev logs  │
  │                          │  (?n=200; no-timeout router)         │
  │  GET  /api/higgs/logs/settings │ LogSettings {verbose,         │
  │                          │  log_incoming_tokens} (read)         │
  │  PUT  /api/higgs/logs/settings │ set both (runtime, not        │
  │                          │  persisted) → gate serve-done +      │
  │                          │  incoming-prompt lines               │
  │  POST /api/higgs/worker/stop  │  Graceful shutdown (2 s)       │
  │  GET  /api/higgs/version │  Build version + engine info        │
  └──────────────────────────┴──────────────────────────────────────┘
  (no /worker/start — load IS the start, unload IS the stop)

  ┌─────────────────────────────────────────────────────────────────┐
  │  /health, /api/higgs/health  │ 200 immediately, NO worker RPC    │
  │                              │ ("server up", not "model loaded") │
  └─────────────────────────────────────────────────────────────────┘

  Serve-layer hardening (request flows top→bottom; first reject wins):
  ┌──────────────────────────────────────────────────────────────────┐
  │  local_cors           browser cross-origin: loopback/tauri only   │
  │  host_guard           DNS-rebind: Host must be loopback → 403 HG012│
  │  CatchPanicLayer      handler panic → structured 500              │
  │  DefaultBodyLimit     body > MAX_BODY_BYTES (32 MB) → 413          │
  │  ── split by route ──                                             │
  │  /api/higgs/* + /v1/models : TimeoutLayer(CONTROL_TIMEOUT 120 s)  │
  │  /v1/chat/completions      : NO http timeout — SSE must not abort │
  │  /api/higgs/logs/stream    : NO http timeout — SSE log stream     │
  └──────────────────────────────────────────────────────────────────┘
  Non-loopback HIGGS_BIND (standalone bin) → startup SECURITY WARNING.

  Inference-path guards (in handler/facade, not tower layers):
  ┌──────────────────────────────────────────────────────────────────┐
  │  validate_sampling   temp/top_p/n/penalties/max_tokens → 400 HG013│
  │  check_prompt_fits   prompt est + max_tokens > ctx_len → 400 HG005│
  │  inference_gate      > MAX_CONCURRENT_INFERENCE (8)   → 503 HG014│
  │  CHAT_RPC_TIMEOUT    wedged worker, no reply in 600 s → 504 HG016│
  │  validate_repo_id    charset / `..` traversal on load → 400 HG015│
  │  path_within_roots   resolved path escapes scan dirs  → 400 HG015│
  └──────────────────────────────────────────────────────────────────┘

  Error mapping (same table for both surfaces):
  ┌────────────────────────────┬──────────┐
  │  HiggsError variant        │  HTTP    │
  ├────────────────────────────┼──────────┤
  │  HG002 ModelNotFound       │  404     │
  │  HG003 ModelNotLoaded      │  404     │
  │  HG005 ContextOverflow     │  400     │
  │  HG006 WorkerSpawnFailed   │  503     │
  │  HG007 WorkerDead          │  503     │
  │  HG013 InvalidSamplingParam│  400     │
  │  HG014 ServerBusy          │  503     │
  │  HG015 InvalidModelId      │  400     │
  │  HG016 ChatTimeout         │  504     │
  │  anything else             │  500     │
  └────────────────────────────┴──────────┘

  /v1 errors:  {"error":{"message":"…","type":"…","code":"…"}}
  control errors: {"error":"[HGxxx] …"}
```
