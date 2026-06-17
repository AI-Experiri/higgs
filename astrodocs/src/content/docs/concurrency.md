---
title: Concurrency model
description: Current request-keyed routing design and deferred parallel-execution options for higgs.
---

## Table of contents

1. [Current: single-sequence engine + request-keyed routing](#current)
2. [Deferred: true parallel execution (design space)](#deferred)

---

## Current: single-sequence engine + request-keyed routing {#current}

### What is implemented

| Layer | Behaviour |
|---|---|
| Worker process | Spawn-on-load / kill-on-unload — at most one worker, named `higgs(<model>)`; zero process when nothing loaded |
| Lifecycle serialisation | A `tokio::sync::Mutex` serialises `load` / `unload` / `stop` so spawn-on-load and kill-on-unload never interleave |
| Worker execution | Single-threaded stdin loop — one `higgs/chat` at a time |
| Supervisor routing | Per-request `mpsc::unbounded_channel` keyed by `request_id` |
| Concurrent callers | Accepted; serialised at the worker; each caller's deltas are isolated |
| Control RPC bound | `scan`/`load`/`status`/`unload` capped at 120 s; chat is unbounded (capped by `max_tokens`) |
| Rejection policy | None — no 409, no busy signal. Chat for an unloaded model: JIT on (default) → on-demand load (only-keep-last swap) then serve; JIT off → `404 [HG003]`; unknown id → `404 [HG002]` |

### Request-keyed routing flow

```
caller-A                supervisor                  worker
   │                        │                          │
   │  alloc_request_id()→1  │                          │
   │  register_sink(1)       │                          │
   │  request_with_id(1,     │  M_CHAT {request_id:1}  │
   │    M_CHAT, params) ────►│─────────────────────────►│
   │                         │                          │ (generates tokens)
caller-B                     │  N_CHAT_CHUNK            │
   │  alloc_request_id()→2  │  {request_id:1, delta:…} │
   │  register_sink(2)       │◄─────────────────────────│
   │  request_with_id(2,     │ route_notification:       │
   │    M_CHAT, params) ────►│  chat_sinks[1].send(…)   │
   │                         │                          │
   │  rx(1) ◄── delta ───────│                          │
   │                         │  (worker finishes A,     │
   │                         │   then starts B)         │
   │                         │  N_CHAT_CHUNK            │
   │                         │  {request_id:2, delta:…} │
   │                         │◄─────────────────────────│
   │                         │ route_notification:       │
   │                         │  chat_sinks[2].send(…)   │
   │  rx(2) ◄── delta ───────│                          │
```

### Key invariants

- `alloc_request_id()` generates the id; the SAME id is used as the JSON-RPC
  frame `id` and as `params.request_id` in the M_CHAT body.
- The worker echoes `params.request_id` verbatim in every `N_CHAT_CHUNK`
  notification (see `worker/mod.rs` M_CHAT handler, `request_id` capture).
- `route_notification` looks up `chat_sinks[request_id]` — no clobber, even
  under concurrent callers.
- On worker death, `on_worker_death` calls `chat_sinks.lock().clear()`, which
  drops all senders and closes all receivers cleanly (EOF to every consumer).

### References (copied verbatim)

- omlx `scheduler.py`: uid→request_id bimap; each request gets a unique id
  that is echoed in every streaming notification.
- shimmy `api.rs`: per-request `mpsc::unbounded_channel`; the channel is
  registered before the RPC is sent and removed on completion/error.
- llama.cpp `n_parallel=1` (default): single decode sequence; no KV-cache
  sharing between requests.

---

## Deferred: true parallel execution (design space) {#deferred}

> These are **documented options for a future effort**, not yet implemented.
> Nothing below is active code.

### What "true parallel" means

The worker runs multiple generation sequences concurrently on the same model
using llama.cpp continuous batching.  The supervisor queues requests into the
batch instead of serialising them on stdin.

### Worker loop (omlx `scheduler.py` pattern)

```
┌─ batch loop ───────────────────────────────────────────────────┐
│  while active_sequences:                                        │
│    for each pending admission (non-blocking):                   │
│      if free_slot: assign_sequence(req)                         │
│    batch = build_batch(active_sequences)                        │
│    llama_decode(batch)          # one interleaved step          │
│    for each sequence in batch:                                  │
│      token = sample(seq.i_batch)                               │
│      if EOS or max_tokens: finalise(seq); emit finish           │
│      else: emit N_CHAT_CHUNK(request_id=seq.id, delta=token)   │
└────────────────────────────────────────────────────────────────┘
```

### llama.cpp API (`llama-cpp-2` crate)

| Concept | llama-cpp-2 call |
|---|---|
| Reserve N parallel slots | `LlamaContext::with_n_seq_max(N)` at context creation |
| Assign a new sequence | `context.add_sequence(seq_id, tokens)` |
| One interleaved decode step | `context.decode(batch)` |
| Sample token for sequence i | `context.sample(sampler, i_batch)` |
| Free a finished sequence | `context.clear_kv_cache_seq(seq_id)` |

Reference: `llama.cpp/examples/parallel/parallel.cpp` decode loop.

### Option choices for the deferred work

#### max_concurrent_requests

- Set at startup only (context is allocated once with `n_seq_max = N`).
- Default 1 (current behaviour, zero config change needed).
- omlx and llama.cpp parallel example both use startup-only configuration.

#### Per-sequence context budget

Two approaches:

| Approach | KV-cache allocation | Behaviour |
|---|---|---|
| Shared (omlx / llama.cpp default) | `ctx_len / N` tokens per slot | Fits more sequences; each gets a shorter window |
| Full per-slot (LM Studio style) | `ctx_len` tokens per slot | Longer window per request; fewer parallel slots possible |

#### Slot assignment

- Assign the lowest-idle slot (llama.cpp `parallel.cpp` convention).
- Track slot state: `Free`, `Active`, `Draining`.

#### Overflow policy (all slots busy)

- omlx: return HTTP 503 (`{"error": "all slots busy"}`) when every slot is active.
- This is the only place a rejection is reference-grounded; it is a capacity
  signal, not a concurrency prohibition.
- In higgs terms: emit `HG012` (or a new code) only when `active >= N`
  and `N > 1`; with `N=1` the current serialise-and-queue behaviour is retained.

### Migration path (when this work is picked up)

1. Add `max_concurrent_requests: usize` to `HiggsConfig` (default 1).
2. Replace the single `LlamaContext` with a pool of `n_seq_max` sequences.
3. Replace the synchronous `engine::chat()` with an async task per request
   that feeds tokens into the shared batch loop.
4. The supervisor's keyed sink map (`chat_sinks`) is already correct for N>1;
   no routing changes needed.
5. Add overflow rejection (HG012 or new code) only when all slots are active.
