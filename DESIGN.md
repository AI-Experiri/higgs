# higgs — Design (crate + loose top-level modules)

The WHY behind the crate architecture and the loose `src/*.rs` modules. The five sub-folders
(`api/`, `node/`, `serve/`, `tune/`, `worker/`) carry their own `DESIGN.md`; this document owns
the crate boundary, the worker lifecycle, the chat/scan/log flows, and the design of the loose
modules — including the recent additions **delta_queue** (G1), **keys** (G4), and
**load_robustness** (G5).

## Table of Contents
- [Crate Boundary](#crate-boundary)
- [Worker Lifecycle](#worker-lifecycle)
- [Load-Event Push (ModelLoadPhase)](#load-event-push-modelloadphase)
- [Chat Request Sequence](#chat-request-sequence)
- [Delta Backpressure (G1)](#delta-backpressure-g1)
- [Model Scan Flow](#model-scan-flow)
- [Developer Log Bus](#developer-log-bus)
- [API-Key Auth (G4)](#api-key-auth-g4)
- [Load Robustness (G5)](#load-robustness-g5)
- [On-disk State](#on-disk-state)
- [Actor Runtime](#actor-runtime)
- [Endpoint Surface & Error Mapping](#endpoint-surface--error-mapping)
- [Diagnostic Code Registry](#diagnostic-code-registry)

---

## Crate Boundary

```
┌─────────────────────────────────────────────────────────────┐
│                         host app                            │
│  production: jigglebot server launcher; or any Axum host    │
│  HiggsConfig { lmstudio_dirs, hf_dirs, ollama_dirs,         │
│                default_load, worker_exe }                   │
│  let h = Arc::new(Higgs::new(cfg)); h.start().await;        │
│  CONTROL: h.load(), h.chat_stream(), h.status(), … (Rust)   │
│  HTTP /v1: serve_v1(h, listener, shutdown_signal())         │
└──────────────────────────┬──────────────────────────────────┘
              in-process    │    /v1 chat+models
              Rust facade   │    over a socket
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                       higgs crate                           │
│  (standalone — zero dep edges to jigglebot)                 │
│  facade + control = PURE RUST (cannot crash the host)       │
│                                                             │
│  api.rs        Higgs facade → co-located LOCAL NodeRuntime   │
│  api/embed.rs  in-process control methods (model_entries,   │
│                mint_key, hub_enable, pair, …)               │
│  serve/control.rs  PURE control helpers the facade shares   │
│  supervisor.rs Worker process manager + RPC correlator      │
│  actor.rs      generic mailbox runtime + ReplyDemux         │
│  serve/        Axum /v1-ONLY router (chat + models + health) │
│  node/         iroh remote fleet (hub / node / transport)   │
│  rpc.rs        NDJSON JSON-RPC 2.0 codec                    │
│  diagnostic.rs HiggsError HG001–HG069                      │
└──────────────────────────┬──────────────────────────────────┘
                           │ stdio (NDJSON JSON-RPC 2.0)
                           │ ONLY while a model is loaded
                           ▼
               ┌───────────────────────┐
               │   worker process      │  spawn-on-load / kill-on-unload
               │   higgs(<model>)      │  (zero process when nothing loaded —
               │   --higgs-worker      │   ISOLATED crash domain)
               │   llama.cpp engine    │
               └───────────────────────┘
```

**Control-plane flow (no HTTP for control):**

```
  embedder (jigglebot)                        external OpenAI client
        │ Rust call                                  │ HTTP
        ▼                                            ▼
  Higgs facade  ──────────────────────────►  serve::v1  (POST /v1/chat/completions,
  (api.rs + api/embed.rs)                              GET /v1/models, GET /health)
        │ delegates to                               │  (uses the SAME facade)
        ▼                                            │
  serve::control PURE helpers  ◄──────────────────────┘
  (model_entry row · hub_status · decide_mint/revoke · TuneProfileViews)
        │
        ▼
  NodeRuntime → Supervisor → worker  (local)   |   HubFleet → node → worker  (remote)
```

**Why a separate worker process:** the llama.cpp FFI is native and can segfault or leak; a
worker crash must never take down the host. The control plane (`Higgs` facade + the `/v1`
router) is pure Rust, so `/v1` and the crate control API stay up even with no worker.
`Higgs::new` spawns nothing — the
first `load()` spawns the worker, `unload()` kills it (zero idle RAM). Since v-fork the
`llama-cpp-2` binding resolves to the AI-Experiri fork (restored OpenAI-compat chat API);
`LLAMA_CPP_2_VERSION` = `"0.1.151"` is baked from the lock file as the reported binding version.

**Facade over a local node (`api.rs`):** `Higgs` is a typed wrapper over a co-located LOCAL
`NodeRuntime` — the SAME multi-worker engine remote nodes run — so local and remote share one
code path. `Higgs` owns only facade-level state: the live `HiggsConfig` (`parking_lot::Mutex`),
the load-lifecycle mutex, two `Semaphore` inference gates (local + remote, each
`MAX_CONCURRENT_INFERENCE`), the runtime toggles (JIT / idle-reaper / serving as atoms), the
readiness gate, and the tune/estimate/OOM-retry wiring. Everything worker-touching delegates to
the node.

---

## Worker Lifecycle

```
  Higgs::start()  →  near-no-op: NO worker; scan() is host-side.
                     ALSO spawns the idle reaper task (holds Weak<Higgs>):
                       every 30 s, if auto_unload_idle AND idle > effective TTL
                       AND the inference gate is fully open → unload().
                       Self-terminates when the host drops its Arc<Higgs>.

  load(M)  (lifecycle mutex held for the whole body)
        ├─ scan() host-side → resolve M's GGUF path
        ├─ readiness/JIT gate: model must be Prepared (HG046) & profile fresh (HG047)
        ├─ RAM headroom guard: file > available_ram * MEMORY_HEADROOM_FRACTION
        │     → InsufficientMemory [HG017] → 503 (BEFORE spawning a worker)
        ├─ Supervisor::start_for(M) → SPAWN worker (<bin> --higgs-worker, argv0 higgs(<M>))
        ├─ send M_LOAD { path, <full LoadParams> }
        │     on OOM [HG004]: run the load_robustness ladder (settle → KV-off →
        │        fewer layers), each rung a coded HG061 event; exhausted → HG060
        └─ ok → record_last_load(params) + emit ModelLoaded

  unload()  ├─ clear_last_load() (a racing respawn can't replay it)
            ├─ M_UNLOAD (graceful) → Supervisor::stop() (kill) → emit ModelUnloaded

  UNEXPECTED DEATH (stdout EOF, not a deliberate stop):
        └─ single respawn (1 s backoff) → replay last M_LOAD → emit WorkerRestarted
           (scan is host-side — no scan replay)
  TERMINAL: deliberate stop → no respawn; factory failure on respawn → WorkerDied.
```

**Supervisor transport (`supervisor.rs`):** `tokio::process::Command` with owned stdio halves
(mirroring `mcp`'s `add_local`). A writer task owns stdin and drains an mpsc channel (serialising
concurrent callers); a reader task owns stdout and dispatches each NDJSON line via `ReplyDemux`
— responses correlate by id, `N_CHAT_CHUNK` notifications route to the keyed chat sink. No mutex
on the I/O path. RPC round-trips are bounded so an alive-but-wedged worker can never hang a
caller: `CONTROL_RPC_TIMEOUT` 120 s (scan/load/status/unload), `CHAT_RPC_TIMEOUT` 600 s (`HG016`
→ 504), `SYSINFO_RPC_TIMEOUT` 15 s (empty device list on expiry). The HTTP layer deliberately
does NOT time `/v1/chat/completions` — a long SSE stream must outlive any per-request timeout.

---

## Load-Event Push (ModelLoadPhase)

Model-load progress is **pushed**, not polled. `Higgs::load` emits a `ModelLoadEvent` at every real
seam in `load_inner` over a `tokio::sync::broadcast` channel; an embedder taps it with
`Higgs::subscribe_load_events()` and relays it on its own plane (higgs itself exposes no HTTP event
route). Each event carries the model `id`, the `phase`, an `at_ms` stamp, and — on `Failed` — the
`HGxxx` `code`.

```
  subscribe_load_events()  (broadcast<ModelLoadEvent>)
        ▲                        one event per transition
        │
  Higgs::load(id, params) ── load_inner ──►
        │
        ▼   (state machine — every phase maps to an observed transition; none faked)

     Queued ──► Preparing ──► LoadingWeights ──► Finalizing ──► Ready   (terminal)
        │           │              │                 │
        └───────────┴──────────────┴─────────────────┴──────────► Failed (terminal, carries code)

     Queued          waiting behind another in-flight load (per-facade load lock held)
     Preparing       validate id/profile, resolve params, capture tune anchors
     LoadingWeights  the multi-second worker load: mmap weights → GPU upload → KV alloc
     Finalizing      load ok; persist the load record + sync the tuning profile
     Ready/Failed    terminal (Failed's ModelLoadEvent.code carries the HGxxx)
```

---

## Chat Request Sequence

```
  POST /v1/chat/completions → serve::v1
    ├─ serving gate (HG019), auth gate (HG048)
    ├─ resident? no → JIT branch (jit_enabled, default on):
    │     scanned → Higgs::load() then serve (a failed load surfaces the REAL
    │               mapped error — 503 HG017 / 400 / … — NOT 404)
    │     unknown → 404 HG002; JIT off & unloaded → 404 HG003
    ├─ validate_sampling()  temp/top_p/penalties/max_tokens (vllm ranges) → 400 HG013
    ├─ check_prompt_fits()  prompt est + max_tokens vs ctx_len → 400 HG005
    └─ Higgs::chat_stream(...)
          ├─ stamp last_activity (keeps the idle reaper at bay)
          ├─ inference_gate.try_acquire (max 8) → 503 HG014 if full
          └─ Supervisor::register_chat_sink(id) → request_with_id(M_CHAT)
                bounded by CHAT_RPC_TIMEOUT (HG016 → 504)
                worker: chat template render (HG050) → llama.cpp sampling loop
                each token → N_CHAT_CHUNK → reader → DeltaSender → SSE "data:" line

  Non-streaming: drop(DeltaReceiver), await outcome → single JSON body.
  /v1 error envelope: every client-facing message is path/host redacted.
```

`/v1` **never** returns a worker-down 503: spawn-on-load means "no worker" == "nothing loaded",
so (JIT off) a crashed worker presents as an empty `/v1/models` + 404 chat. `Higgs::status()`
(the in-process facade method) is where `worker_alive` is exposed.

---

## Delta Backpressure (G1)

`delta_queue.rs` replaced the old `mpsc::unbounded_channel<ChatDelta>` (one queue entry per
generated token, unbounded) with a **bounded, merging** SPSC queue — one per chat request,
between a producer (supervisor demux / hub transport reader) and its single streaming consumer
(SSE assembly / fleet relay).

**Why merging.** A slow or stalled SSE client used to accumulate one allocation + channel entry
per token, unbounded in count. The queue MERGES consecutive same-kind deltas in place (the
vLLM/omlx "merging collector" pattern):
- **entries** are bounded by kind-alternations (content ↔ reasoning ↔ tool-call), not token
  count — a stalled client holds a handful of growing strings, not thousands of fragments;
- **bytes** are bounded by the generation itself (already bounded by `max_tokens`/`ctx_len`);
  merging is lossless (text concatenated in arrival order);
- tool-call fragments never merge — each is a complete JSON fragment the `/v1` chunk protocol
  preserves.

**Invariants.** SPSC by construction (one queue per request). `DeltaSender` drop closes the
queue (receiver drains, then sees end-of-stream), matching the old `UnboundedSender` semantics.
`DeltaReceiver` drop frees the backlog and makes sends no-ops — so the non-streaming `/v1` path
(which drops the receiver and awaits only the final outcome) never pointlessly buffers. A
`Notify` permit is stored when nobody waits, so a push racing the check-then-await is never lost.
`recv()` does the pop AND the closed/overflowed check in ONE critical section — with separate
locks, a `send(final delta)` + sender-drop landing between them could return `None` with that
delta still buffered, silently losing the tail of the stream.

**Safety cap.** `CAP_BYTES` (8 MB) of UNDELIVERED backlog is the pathological-stall limit: a
client that holds the connection but never reads while a huge generation streams trips the cap,
the buffer is dropped, the queue closes, and the consumer surfaces `HG057`
(`ChatStreamOverflow`) — a LOUD coded failure instead of unbounded memory growth. A healthy
request never approaches it.

---

## Model Scan Flow

```
  Higgs::scan()  — HOST-SIDE (pure Rust: ggus + memmap2 + std::fs, NO FFI, NO worker),
        │            wrapped in spawn_blocking. Available with no worker live.
        └─ ModelStore::scan(lmstudio_dirs, hf_dirs, ollama_dirs)
             load() resolves the GGUF path here and carries it in M_LOAD.

  Read-only stores (higgs NEVER writes into them):
    LM Studio   ~/.lmstudio/models/ | ~/.cache/lm-studio/models/  (id = parent dirs)
    HuggingFace ~/.cache/huggingface/hub/models--org--model/…     (id = org/model)
    Ollama      ~/.ollama/models/ manifests+blobs                 (id = ollama/<name>:<tag>)

  Downloads land ONLY in ~/.higgs/models/ (download.rs) — never a scanned store.
  Per file: ggus metadata mmap → arch/ctx_train/chat_template; quant tag from filename;
            unreadable file → fields omitted (resilient); HG001 on unreadable root.
```

---

## Developer Log Bus

`log_bus.rs` is the single home for Developer-Log lines. `LogBus` holds a bounded history ring
**per `LogSource`** plus a live `broadcast` tap:

- **Sources:** `Serve` (higgs `tracing` events via `HiggsLogLayer`), `Worker` (legacy unkeyed
  local stderr + transient probes), `LocalWorker{worker}` (per-loaded-model console), and
  `RemoteWorker{node,worker}` (a remote node's worker stderr relayed over iroh).
- **Why per-source rings:** a chatty worker (a model load dumping thousands of llama.cpp
  metadata lines) must not evict the serve history. Each `?source=` console keeps its own
  `RING_CAP` (2000) of history. Rings for dead workers are reclaimed (`evict_local` /
  `evict_remote` / `evict_node`).
- **Sinks (in-process crate API):** `Higgs::logs(n, source)` snapshots the ring tail;
  `Higgs::subscribe_logs()` returns a `broadcast::Receiver<LogLine>` — subscribe first, replay
  the ring, then stream live (a `Lagged` subscriber skips the gap and keeps streaming). The
  embedder relays these lines on its own plane (there is no higgs HTTP log route). `snapshot(None)`
  re-interleaves rings by a monotonic `seq`.
- **Concurrency invariant:** the `seq` stamp is assigned WHILE holding the destination ring's
  lock, so the stamp and the `push_back` are atomic per ring — otherwise two concurrent pushes
  could stamp N/N+1 but insert in the opposite order, and `snapshot` (sorts by seq) would
  disagree with the per-source insertion order.
- **Redaction policy:** `MessageVisitor` captures only `message` + the typed `error`
  (`[HGxxx] …`) field. Other structured fields (which may carry prompt content) are dropped
  unless the un-redacted DEBUG toggle (`show_fields`) is on. A separate `verbose` atom gates the
  extra per-request completion line and the worker drain's keep-everything behavior.

---

## API-Key Auth (G4)

`keys.rs` gates the HTTP surface. Design points:
- **Opt-in / fail-open for embedded, fail-closed for LAN.** An empty keystore means the surface
  is OPEN — the embedded in-process host wants no gate. Auth turns on the moment the first key
  exists. But `serve_v1` refuses to serve a non-loopback listener with zero keys (`HG058`) — and,
  as a backstop, with keys of which NONE is Admin-capable (`HG069`, which would lock out the
  key-management surface on that bind) — and refuses to revoke the last key while LAN-exposed
  (`HG059`), because the Host guard + CORS only protect *browser* clients. Both refusals run on
  the REAL bound address, so a library/embedded caller handing higgs its own listener can't bypass
  them. Revoking the last Admin-capable key while OTHER keys remain is likewise refused (`HG066` →
  409): the key-management surface itself would lock out.
- **At-rest = digest only.** Tokens are `hgk_<hex>`; only their SHA-256 hex digest is persisted
  in `api_keys.json`. Plaintext is shown once at mint time. `authorizes()` hashes the presented
  token and compares against ALL stored digests in **constant time** (`subtle::ConstantTimeEq`,
  no early return), so a stored digest can't leak via timing.
- **Scopes.** `Chat` (`POST /v1/chat/completions`), `Models` (listing), `Admin` (everything,
  incl. management). `Admin` is a superset. `Scope` is a `higgs_const_enum!` so the frontend
  gets the const-object form.
- **Two mutation paths.** The CLI (`higgs keys add/list/remove`, `run_keys`) edits `api_keys.json`
  offline and prints a restart notice (a served instance snapshots keys at serve time). The
  in-process crate API (`Higgs::mint_key` / `revoke_key`, over `mutate_api_keys`, Admin scope)
  mutates the LIVE keystore — a minted or revoked key gates the very next request, no restart. An
  embedder can also register an in-memory-only internal token (`register_internal_token`) to
  authenticate itself. Saves are atomic (temp + rename).

---

## Load Robustness (G5)

`load_robustness.rs` is the PURE decision logic wired into `api.rs`'s load path; the API supplies
the real error/VRAM reading and applies the returned params (so the logic is exhaustively
unit-tested without a GPU).

- **Classify first.** `is_oom_reason(reason)` matches backend-agnostic allocator signatures
  (`out of memory`, `cudaMalloc failed`, `MTLBuffer`, `ggml_backend_alloc`, …) against the joined
  llama.cpp engine error lines. Only an OOM-classified `HG004` is eligible for the ladder — a
  corrupt-GGUF failure would fail identically on retry and is returned immediately.
- **Bounded degrade ladder.** `oom_ladder(base, layer_count)` returns cheapest-relief-first
  rungs: (1) plain retry after a settle (a transient alloc — the just-unloaded model, a peer
  process — may have cleared); (2) KV cache to system memory (`offload_kqv=false`, the biggest
  single VRAM lever that keeps all layers resident); (3) KV-off AND fewer GPU layers (CUMULATIVE
  — built from rung 2, halving `Count{n}` or the known `layer_count`, else dropping to all-CPU).
  Rung 2 is SKIPPED when it wouldn't change anything: the base already carries
  `offload_kqv=false` (a reloaded previously-degraded profile) or the load is CPU-only
  (`Count{0}` keeps KV in RAM regardless) — a guaranteed-OOM duplicate under a lying `HG061`
  line. Rung 3's label is honest about what THIS walk degraded: when the KV cache was already
  in system memory it says so instead of claiming to have moved it. Deterministic and bounded —
  no unbounded gpu-layer halving. Each rung taken emits a coded `HG061`; exhausting the ladder
  is `HG060` (`LoadOomExhausted`) with an aggregate diagnosis.
- **VRAM settle, not poll.** `SETTLE_BEFORE_RETRY` (750 ms) is a cheap, worker-free time pause
  before each rung — a GPU driver/allocator often needs a beat to reclaim the just-failed
  allocation. A poll-until-free variant was DEFERRED (a fresh VRAM read spawns a transient
  sysinfo worker per poll, and unified memory frees lazily). The `api` wiring passes the settle
  as a parameter so tests inject `Duration::ZERO`. `HG062` logs a VRAM-recovery wait timeout.
- **Deferred with evidence:** a true *stall-based* load timeout needs a llama.cpp load-progress
  callback surfaced over a new worker notification (an FFI + worker-protocol change), so it's not
  shipped; a dead worker already fast-fails via the reply demux, and an OOM is rescued by the
  ladder.

---

## On-disk State

Everything lives under `~/.higgs` (`home.rs`; `HIGGS_HOME` override). One file per concern, each
saved atomically (temp + fsync + rename — Unix/macOS only, matching the FFI build target):

| File | Module | Contents |
|---|---|---|
| `endpoint.key` (`0600`) | `node/identity.rs` | iroh `SecretKey` — the only secret |
| `pairings.json` | `auth.rs` | `Allowlist` — paired node ids → labels |
| `api_keys.json` | `keys.rs` | `ApiKeys` — key digests + scopes |
| `config.json` | `config.rs` | `InstanceConfig` — friendly name, saved hubs, per-model load records, extra CORS origins |
| `models/` | `download.rs` | GGUFs pulled via `M_PULL` |
| `models.json` (per node) | `tune/store.rs` | tuning profiles / readiness |

Corruption vs I/O are distinguished: a present-but-unparseable store is `HG041` (fix: delete to
reset); a read/write/rename OS error is `HG040`. Config load tolerates old shapes (`lenient_load`
reads both the engine-tagged and pre-umbrella flat `LoadParams`) so a schema bump never loses the
instance name or saved hubs.

---

## Actor Runtime

`actor.rs` is the generic mailbox runtime written ONCE and reused by `Supervisor`,
`NodeRuntime`, and the per-node transport. An `Actor` contributes only its `Msg` set + `handle`;
`spawn_actor{,_with}` provides the mailbox, recv loop, and graceful shutdown (last-`Handle`-drop
closes the mailbox → `on_stop` drains owned OS resources in async context). `WeakHandle` lets an
actor post follow-up "commit" messages to itself without pinning its own mailbox open — the
**slow-I/O rule**: a `handle` never `.await`s a slow downstream RPC (that would serialize every
op); it spawns the slow work and applies the result via a commit message.

`ReplyDemux` is the reader-side correlation shared by every RPC *client*: `pending` (id →
`oneshot` waiter) and `chat_sinks` (`request_id` → `DeltaSender`, the merging `delta_queue`).
EOF/death cancels every pending waiter and drops every sink, ending in-flight streams cleanly.

---

## Endpoint Surface & Error Mapping

```
  /v1 serve-layer hardening (outer layers run first; src/serve/mod.rs):
    local_cors      → browser cross-origin: loopback/tauri + configured origins only
    host_guard      → DNS-rebind: Host must be loopback → 403 HG012
                      (relaxed on a keyed non-loopback bind — LAN clients' own Host)
    auth_guard      → missing/insufficient key → 401 HG048  (when keys configured)
    CatchPanicLayer → handler panic → structured 500
    DefaultBodyLimit→ body > MAX_BODY_BYTES → 413
    split by route  → GET /v1/models: TimeoutLayer (CONTROL_TIMEOUT);
                      POST /v1/chat/completions (SSE): NO whole-request timeout

  Inference-path guards (handler/facade, not tower layers):
    serving toggle  → HG019 503     validate_sampling → HG013 400
    check_prompt_fits→ HG005 400     inference_gate    → HG014 503
    CHAT_RPC_TIMEOUT → HG016 504     validate_repo_id/path_within_roots → HG015 400
```

Only `/v1` (chat + models) and `/health` are served; there is no `/api/higgs/*` HTTP surface. The
in-process crate control methods return `HiggsError` directly to the embedder, which maps it as it
sees fit. Error mapping for the `/v1` path (the true origin status survives the hub→node hops via
`error.data.code`):

| HiggsError | HTTP | | HiggsError | HTTP |
|---|---|---|---|---|
| HG002 ModelNotFound | 404 | | HG017 InsufficientMemory | 503 |
| HG003 ModelNotLoaded | 404 | | HG018 ResidentModelMismatch | 503 |
| HG005 ContextOverflow | 400 | | HG019 ServingDisabled | 503 |
| HG006 WorkerSpawnFailed | 503 | | HG046 NotPrepared / HG047 ProfileStale | 4xx (JIT gate) |
| HG007 WorkerDead | 503 | | HG048 Unauthorized | 401 |
| HG013 InvalidSamplingParam | 400 | | HG049 InvalidRequest | 400 |
| HG014 ServerBusy | 503 | | HG057 ChatStreamOverflow | 5xx (stream abort) |
| HG015 InvalidModelId | 400 | | HG060 LoadOomExhausted / HG063 BenchExhausted | 503 |
| HG016 ChatTimeout | 504 | | HG064 BenchCancelled / HG066 / HG067 | 409 |
| HG059 LastKeyOnLan | 409 | | HG068 BenchInProgress | 503 |
| | | | anything else | 500 |

`/v1` errors serialize as `{"error":{"message":"…","type":"…","code":"…"}}` and are
path/host-redacted (`serve::v1::redact_paths`). The in-process crate control methods instead return
the typed `HiggsError` with its full `Display` (`[HGxxx] …`) — the embedder decides how to surface
it.

---

## Diagnostic Code Registry

`HiggsError` (`diagnostic.rs`) is **append-only — never renumber a code.** Retired variants
(e.g. `HG020`, the removed scan-time loadability probe) keep their number reserved rather than
being reused.

| Range | Owner | Theme |
|---|---|---|
| HG001–HG021 | scan / worker / serve | model + engine + worker + host-guard + sampling + memory faults |
| HG022–HG028 | `node/`, `auth.rs`, `remote.rs` | remote pairing / version / allowlist / handshake |
| HG029–HG036 | `hub.rs`, `download.rs` | HuggingFace fetch taxonomy (auth/404/rate-limit/http/transport/write/exhausted) |
| HG037–HG039 | `rpc.rs`, `node/` | control-plane RPC + peer-protocol + hub-rejection |
| HG040–HG045 | `config.rs`/`auth.rs`/`keys.rs`, serve | on-disk store I/O + corruption + internal + control-surface-down |
| HG046–HG050 | `api.rs`, `serve/`, worker | readiness gate + auth + invalid request + template render |
| HG057–HG069 | `delta_queue.rs`, `serve/mod.rs`, `keys.rs`, `api.rs`, `tune/` | stream overflow + LAN-bind guards (HG058/HG069 refused in `serve_v1`, HG059) + OOM ladder + Turbotune (HG063/HG064/HG067/HG068) + last-Admin-key lockout (HG066) |

**Log-only codes** (no variant — they ride a `tracing` line so a debugging agent can grep the
Developer Logs, and the request still succeeds): HG051 (undecodable `N_CHAT_CHUNK` dropped),
HG052/HG053/HG054 (incremental/full parse fallbacks), HG055 (bad `chat_template_kwargs`
ignored), HG056 (malformed streamed tool-call fragment dropped), HG061 (OOM rung taken), HG062
(VRAM-recovery wait timed out), HG065 (Turbotune candidate rejected).

---

## Residual / deferred items

- **Stall-based load timeout (G5):** needs a llama.cpp load-progress callback (FFI + worker
  protocol) — deferred with evidence; the OOM ladder + dead-worker fast-fail cover most cases.
- **Per-worker (per-load) idle TTL on the remote wire:** `NodeLoadParams` carries the full
  `LlamaCppParams` only on the LOCAL path; forwarding rich params (and a per-load TTL) to a
  remote `M_NODE_LOAD` is gated behind a protocol-version bump (an older `deny_unknown_fields`
  node would reject an unknown field).
- **`log_bus` double-clone (C1):** each line is cloned into the ring String and the broadcast
  `LogLine.text`; an `Arc<str>` would just move the alloc to the String-typed SSE boundary — no
  net win without reworking the log-stream channel type. Left as a documented TODO.
</content>
