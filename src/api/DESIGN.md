# `api/` — design notes

## Why this module exists

`Higgs` is the single host-facing seam over the runtime. Everything a host or the serve
layer can do — load/unload a model, run a chat, read status/logs/events, toggle runtime
settings, tune/estimate, query hardware — goes through one `Higgs` value. That one entry
point is deliberate: hosts depend on a small, stable surface, and the wire types
(`higgs_ts!`/`higgs_const_enum!`) are generated into `bindings/` from `api/types.rs`.

## Why it is split (and how)

The facade had grown past ~2,300 lines. The split groups code by **responsibility**, not
by layer, and is **behavior-preserving** (`api.rs` re-exports the submodules, so
`crate::api::X` paths are unchanged):

- **`types.rs`** — the data vocabulary (wire types + constants + the `ChatOutcome`
  decoder). These change for protocol reasons and carry no behavior; isolating them keeps
  the impl file about logic.
- **`guards.rs`** — pure validation/containment/headroom functions with no `Higgs` state.
  `pub(crate)` because `node::runtime` reuses `guard_memory_headroom` and
  `path_within_roots` (one guard implementation, two callers). Being state-free makes them
  unit-testable without provisioning multi-gigabyte fixtures (`fits_in_memory` is the pure
  arithmetic seam).
- **`api.rs`** keeps the `Higgs` struct + its core lifecycle `impl`, because those are
  tightly coupled to the struct's private fields and to each other.

(An earlier `reaper.rs` held the engine-level idle auto-unload loop; P4b moved idle
auto-unload INTO the node — per-worker, uniform local+remote — so that file was removed.)

## How the facade routes (post-P4b)

`Higgs` holds `local: Arc<NodeRuntime>` (the co-located multi-worker node) and an optional
`fleet` (remote nodes). It is **local-first**: listing, loaded-model gating, and chat
routing all prefer a locally-resident model over a remote one of the same served id.

- **`load`** is additive on the node (one worker per model) and made **idempotent per
  raw id** at the facade (`load_inner_impl` early-returns if that raw id is resident — the
  node itself never dedups). **`unload`** drains every local worker; **`unload_one`**
  targets one served id; **`status`** reports the PRIMARY (lowest worker id) instance.
- **`chat_stream`** resolves a SERVED id → `(worker, raw model)` via `local_served`,
  leases an `inference_gate` permit, and sends the RAW model on the wire; a non-local id
  falls through to the fleet (bounded by the SEPARATE `remote_gate`).
- **`local_served_ids`** feeds `/v1/models` (∪ the fleet's routed models).

Worker-process lifecycle, RPC correlation, idle auto-unload, and the Developer-Log bus all
live in the node, not the facade.

### Load lifecycle & SSE phases

`load` → `load_inner` → `load_inner_impl`. `load_inner` brackets the load with the
terminal `ModelLoadPhase` events pushed over `GET /api/higgs/events`: it emits `Queued`
FIRST (before the possibly-contended `lifecycle` lock, so the UI bar appears instantly),
and a `TerminalGuard`+drop guarantees a terminal `Ready`/`Failed` fires even on client
cancellation. `load_inner_impl` emits the mid-load phases at their real seams
(`Preparing` after the resident no-op check → `LoadingWeights` around the multi-second
worker load → `Finalizing` for the bookkeeping). Failure carries the diagnostic code on
the event. A `LoadingGuard` publishes/clears the `loading` snapshot so a concurrent
`status` can show a progress indicator; the load itself walks the G5 OOM degrade ladder
(`run_oom_ladder`: plain retry → KV to system memory → fewer GPU layers, HG061 per rung,
HG060 on exhaustion) and syncs the saved tuning profile — for an ACCEPTED explicit load
(anchored to a hardware fingerprint + file signature captured BEFORE the slow load), AND
for a REUSE load the ladder had to DEGRADE (`loaded_np != requested_np`), so the next
reload uses the fitting config instead of re-walking the ladder. `set_profile` drops the
record's stale `provenance`/`bench_tps` whenever the written params differ from the saved
profile — a degraded fallback or an edited explicit reload was never the benchmarked
config, so the store must not claim its throughput.

## G6 Turbotune — measured autotune, EXCLUSIVE benchmark (RECENT)

`tune(req)` normally returns an analytical `Suggest`. When `req.mode == Benchmark` it
runs **Turbotune**: `turbotune_bench` seeds candidate configs from the analytical
suggestion (`bench::bench_candidates`, fit/headroom-filtered, no loads), then LOADS +
measures each fastest-first via the free `run_benchmark` orchestrator, overriding the
saved profile with the fastest MEASURED config (`provenance = Bench`, `bench_tps` set).
Each measurement `bench_unload_id`s the model's prior candidate, loads THIS candidate
VERBATIM through the private node `local.load` (`node_params_for` — no OOM-ladder
degrade, no config persist; the candidate set is itself the degrade ladder), runs
`measure_gen_tps` (a short real 64-token, temp-0 decode; `gen_tps` is DECODE-ONLY via
`bench::bench_gen_tps` = `(completion_tokens − 1) / (total − ttft)` — prefill excluded
so `pick_benchmarked` ranks generation rate, not end-to-end latency skewed by prompt costs),
and tears down. Every candidate failing yields `[HG063]` (`BenchExhausted`); a
per-candidate failure logs `[HG065]` and moves on.

**The benchmark is EXCLUSIVE — it owns its model; public ops are refused, not raced.**
One facade field makes that correct: `benchmarking: parking_lot::Mutex<HashSet<String>>`,
set by `begin_benchmark(id)` UNDER the `lifecycle` lock and cleared by the RAII
`BenchmarkingGuard` on every exit (including drop/panic). The contract, both directions:

- A benchmark REFUSES to start if the model is LOADED (`[HG067]` `BenchModelLoaded` —
  unload first) or ALREADY benchmarking (`[HG068]`). Both checked under `lifecycle`, so
  a racing public load can't slip between the check and the flag-set.
- While a benchmark runs, every public path for that model refuses with `[HG068]`
  (`BenchInProgress`, retry ~5 min): `load_inner_impl` (explicit + JIT load, checked
  under `lifecycle` BEFORE the resident no-op), the serve layer's `ensure_loaded`
  (checked AFTER the awaited resident lookup — a pre-check would go stale across the
  await), `unload`/`unload_one`/`worker-stop` (a drain would evict the candidate
  mid-measure), and the public `chat_stream` local resolution, which SKIPS a
  benchmark-owned worker (`serve_public` flag) — with a fleet installed the chat can
  still route to a REMOTE copy of the same id. The bench's OWN loads / teardown /
  measurement ride private paths (`local.load`, `bench_unload_id`,
  `chat_stream_inner(serve_public = false)`), so it never refuses itself.
- The ONLY op not refused is a terminal `stop` (shutdown): `stop()` sets
  `shutting_down: AtomicBool` BEFORE draining, and `run_benchmark`'s per-candidate
  cancel hook polls it — the bench aborts cleanly with `[HG064]` (`BenchCancelled`)
  instead of iterating candidates against a draining node. (Normally the in-flight
  benchmark request drains before `stop` runs; this is defense-in-depth.)

CANCELLATION SAFETY: if the benchmark future is dropped mid-candidate (long-op route
timeout / client disconnect), the OWNING `BenchmarkingGuard` — which tracks the LIVE
candidate's `WorkerId` in its `live` slot — spawns a task that unloads exactly that
worker and only THEN clears the [HG068] flag (clearing first would open a window where
a public chat/load could adopt the doomed candidate before the unload kills it). The
leftover candidate is thus auto-reclaimed instead of being served publicly with an
unchosen config. Worker-id-scoped (never id-scoped), so it cannot stomp a worker a user
loads after the flag clears; on every normal exit the slot is `None` and the flag is
cleared synchronously. `/api/higgs/models/tune`
and `/models/load` sit in the serve layer's `LONG_OP_TIMEOUT` (20 min) router group so a
multi-candidate benchmark / multi-rung OOM ladder is not 408-cancelled by the 120 s
control timeout.

## Concurrency / locking model

`Higgs` mixes lock flavors by hold pattern; the doc-comment on each field states its
discipline. Summary:

- **`lifecycle: tokio::sync::Mutex<()>`** — serializes `load`/`unload`/`unload_one`/`stop`
  so two racing JIT loads of the same id can't each spawn a worker and a deliberate stop
  never interleaves with a load. Held across `.await`.
- **`hub_lifecycle: tokio::sync::Mutex<()>`** — separate serialization of the hub kill
  switch enable/disable across its whole check→start→publish / shutdown→clear sequence;
  SEPARATE from `lifecycle` so model ops and hub ops never cross-couple. Held across
  `.await`.
- **`inference_gate` / `remote_gate: Arc<Semaphore>`** — bound in-flight local vs remote
  chats at `MAX_CONCURRENT_INFERENCE` each. `try_acquire_owned` failure on the chat path
  → `ServerBusy` (503); the owned permit rides the spawned generation task and releases on
  any outcome. They are SEPARATE so remote traffic can't grow node tasks unbounded AND so
  it doesn't entangle the idle reaper's "acquire all LOCAL permits to unload" logic.
- **Atomics** (`AtomicBool`/`AtomicU64`, always set/read in isolation, never across an
  `.await`): the serve-layer toggles `log_incoming_tokens`, `jit_enabled`,
  `auto_unload_idle`, `idle_ttl_minutes`, `serving_enabled`, `lan_exposed`, plus
  `shutting_down` (set once by `stop`; polled by the Turbotune cancel hook).
- **`parking_lot::Mutex<…>`** — held only across synchronous work, NEVER across an
  `.await`: `config`; the `*_io` RMW serializers (`config_io`, `models_io`, `keys_io`)
  that make each read-modify-write of `config.json`/`models.json`/`api_keys.json` atomic
  w.r.t. the others (the whole file is atomically temp+renamed, so last-writer-wins would
  otherwise clobber); the Turbotune `benchmarking` id set (set/cleared under `lifecycle`,
  read lock-free by the HG068 gates); the caches `device_cache`, `estimate_meta_cache`;
  the swappable installers `fleet`, `api_keys`, `hub`; the `loading` snapshot; and
  `config_path`.
- **`load_events: broadcast::Sender<ModelLoadEvent>`** — live SSE fan-out; a `send` with
  no subscribers is a harmless no-op, and there is no replay ring (the bar is transient).

## Invariants

1. **No public-path churn.** `api.rs` re-exports submodule items; the split is invisible
   outside `api/` and was verified by codex convergence + both coverage gates.
2. **Privacy is intentional.** Wire types/constants are `pub` (some cross the bindings
   boundary); the guard helpers reused by the node are `pub(crate)`; `fits_in_memory` is
   re-exported only under `#[cfg(test)]`.
3. **Success-before-persist.** The saved tuning profile is synced only AFTER a real load
   succeeds (never for the idempotent resident no-op), so a plain reload can't reuse a
   profile that was never validated by a load. The synced params are the ones that
   ACTUALLY loaded (the OOM ladder's successful rung), and a params CHANGE clears the
   record's stale bench provenance/throughput.
4. **A benchmark owns its model exclusively.** It refuses to start against a loaded
   model ([HG067]); while it runs, every public load/chat/unload/worker-stop for that
   model refuses ([HG068]) rather than racing it; only a terminal shutdown cancels it
   ([HG064]). The bench's own loads/teardown/measurement are on private, ungated paths
   so it never refuses itself.
5. **Loopback-only.** `BIND_HOST` is `127.0.0.1`; the `lan_exposed` flag gates the
   last-key-revoke refusal (HG059) so a LAN-exposed server can't flip auth off at runtime.

## Error codes this module owns / raises

Raised directly in this folder:

| Code | Variant | Where / meaning |
|------|---------|-----------------|
| HG014 | `ServerBusy` | `chat_stream` admission gate full → 503 (retryable). |
| HG015 | `InvalidModelId` | `guards::validate_repo_id` — bad charset / `..` traversal → 400. |
| HG017 | `InsufficientMemory` | `guards::guard_memory_headroom` — load exceeds `MEMORY_HEADROOM_FRACTION` of available RAM → 503. |
| HG040 | `PersistenceFailed` | `tune` / `sync_saved_profile` — `models.json` open/flush failed. |
| HG044 | `ChatTaskFailed` | `measure_gen_tps` — the spawned generation task join failed. |
| HG047 | `ProfileStale` | `load_inner_impl` reuse of a stale saved profile (hardware/file changed since Prepare) — re-tune or load with explicit params. |
| HG060 | `LoadOomExhausted` | `run_oom_ladder` exhausted the OOM degrade ladder. |
| HG063 | `BenchExhausted` | `run_benchmark` found no working config. |
| HG064 | `BenchCancelled` | Turbotune bench aborted by a terminal shutdown (`stop` sets `shutting_down`; the per-candidate cancel hook trips). |
| HG067 | `BenchModelLoaded` | `turbotune_bench` — the model is loaded; unload it first to benchmark → 409. |
| HG068 | `BenchInProgress` | `load_inner_impl` / `unload` / `unload_one` / a second `turbotune_bench` — a benchmark owns the model; retry ~5 min → 503. (Also raised by the serve layer's `ensure_loaded`.) |
| HG002 | `ModelNotFound` | `tune` scan miss for the requested id. |
| HG019 | `ServingDisabled` | The `serving_enabled` toggle (surfaced by the serve layer when off). |

`[HG061]` (degrade-rung retry) and `[HG065]` (a failed bench candidate) are structured
LOG lines, not error variants. Codes like HG001/HG002/HG005/HG016/HG004/HG006 originate in
the node/worker/scan paths this facade delegates to.

## Boundaries / what does NOT belong here

- Worker-process lifecycle, RPC correlation, restart FSM → `supervisor.rs`.
- Multi-worker orchestration + the node idle reaper → `node/runtime.rs`.
- Remote fleet routing + served-instance ids → `node/fleet.rs`, `node/served.rs`.
- The candidate generation / benchmarked-pick / fit math for Turbotune → `tune/bench.rs`,
  `tune/vram.rs` (this module only orchestrates load→measure→cancel around them).

## Deferred / residual items

- **In-flight-load dedup on cancellation** (`load_inner_impl`): a load cancelled between
  `NodeMsg::Load` and its reply can let a racing same-id load spawn a duplicate worker.
  Self-healing (the cancelled load's `LoadCommit` reaps its worker) and RAM-bounded; a
  fully cancellation-safe dedup would change the node's additive contract — deferred.
  The SAME window exists when a Turbotune benchmark is cancelled while its candidate
  `local.load` is still pending (codex r19): `live` is unset, so the guard clears the
  [HG068] flag and a racing public load can start before the doomed candidate load
  commits-and-reaps. Identical mechanism, identical self-healing bound — covered by
  this deferral, not a new class.
- **Served-id stability** (`unload_one`/`chat_stream`): suffixed ids (`org/model-1`)
  renumber when a sibling is reaped, so a request from a stale snapshot can hit a different
  instance. A property of the served-id SCHEME; the stable worker-id/generation selector is
  deferred.
- **Per-load idle-TTL** (`LoadedInfo::idle_ttl_minutes`): reserved; the reaper applies one
  per-node TTL, so it is currently always absent.
- Prompt-throughput in `measure_gen_tps` is approximated from prompt tokens over TTFT;
  exact prefill timing is a worker-side refinement.
