# `api/` — design notes

## Why this module exists

`Higgs` is the single host-facing seam over the runtime, and — since the `/api/higgs/*`
HTTP handlers were removed — it is also higgs's **only control plane**. Everything a
host can do (load/unload a model, run a chat, read status/logs/events, toggle runtime
settings, tune/estimate, mint/revoke keys, drive hub/fleet, query hardware) goes through
one `Higgs` value as a typed, in-process method returning `Result<_, HiggsError>` — no
HTTP required. The only HTTP surface higgs still serves is the strict OpenAI `/v1`
(`serve::v1_router` → `serve::serve_v1`); the embedder maps a `HiggsError` however it
wishes. That one entry point is deliberate: hosts depend on a small, stable surface, and
the wire types (`higgs_ts!`/`higgs_const_enum!`) are generated into `bindings/` from
`api/types.rs`.

```
   embedder                     ┌──────────────────  Higgs  (Arc)  ──────────────────┐
   (jigglebot) ────────────────▶│  api.rs    struct + core lifecycle impl            │
                       crate    │  embed.rs  in-process control-plane impl           │
   serve::v1_router ───────────▶│  types.rs  wire types + consts                     │
   (POST /v1/chat, GET /v1/     │  guards.rs pure load guards (also used by node)    │
    models, GET /health)        └────────┬───────────────────────────┬──────────────┘
   Arc<Higgs>                            │                            │
                            delegates to │                            │ delegates to
                                         ▼                            ▼
                           serve::control (PURE helpers)      node::runtime  (LOCAL)
                           model_entry / hub_status           workers, RPC correlation,
                           decide_mint / decide_revoke        idle reaper, load-replay
                           validate_key_label                        │
                           TuneProfileViews / active_records         │  remote-resident id?
                                                                     ▼
                                                            node::fleet (REMOTE nodes)
```

There is NO `/api/higgs/*` HTTP surface, NO standalone control server, and NO
jigglebot reverse-proxy: an embedder calls the facade methods directly.

## Why it is split (and how)

The facade had grown past ~2,300 lines. The split groups code by **responsibility**, not
by layer, and is **behavior-preserving** (`api.rs` re-exports the submodules, so
`crate::api::X` paths are unchanged):

- **`types.rs`** — the data vocabulary (wire types + constants + the `ChatOutcome`
  decoder + `PreparedChat`/`PairInfo`). These change for protocol reasons and carry no
  behavior; isolating them keeps the impl files about logic.
- **`guards.rs`** — pure validation/containment/headroom functions with no `Higgs` state.
  `pub(crate)` because `node::runtime` reuses `guard_memory_headroom` and
  `path_within_roots` (one guard implementation, two callers). Being state-free makes them
  unit-testable without provisioning multi-gigabyte fixtures (`fits_in_memory` is the pure
  arithmetic seam).
- **`embed.rs`** — the in-process control plane: a second `impl Higgs` holding the
  behavior the deleted `/api/higgs/*` handlers used to carry (models list, load/unload
  shaping, hub/fleet ops, trusted key mint/revoke, the chat gate, the `/v1/models`
  union). It stays THIN by delegating the row-formatting / mint-revoke-decision /
  hub-status / tune-view logic to the pure `serve::control` sub-primitives — the same
  helpers the (now removed) HTTP handlers shared — so the in-process and any future
  surface cannot diverge.
- **`api.rs`** keeps the `Higgs` struct + its core lifecycle `impl` (construction,
  load/unload/status/chat, the tune/estimate/Turbotune engine, the runtime toggles),
  because those are tightly coupled to the struct's private fields and to each other.

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
- **`resolve_loaded`** (in `embed.rs`, behind `prepare_chat`) is the chat gate: already
  served locally → its `LoadedInfo`; remote-resident → a permissive placeholder (the
  fleet routes it); benchmark-owned and not remote → `[HG068]`; not loaded + JIT off →
  `[HG003]`; not loaded + JIT on → a scanned (`[HG002]` else) + Prepared
  (`[HG046]`/`[HG047]` else) model is loaded with its VALIDATED profile.
- **`chat_model_ids`** feeds `/v1/models`: `local_served_ids()` ∪ (JIT-on)
  `servable_model_ids()` ∪ `fleet().routed_models()`, minus a model whose only local
  candidate is transiently benchmarking (unless a remote node also serves it).

Worker-process lifecycle, RPC correlation, idle auto-unload, and the Developer-Log bus all
live in the node, not the facade.

### Load lifecycle & load-phase events

`load` → `load_inner` → `load_inner_impl`. `load_inner` brackets the load with terminal
`ModelLoadPhase` events pushed over the `load_events` broadcast (subscribed with
`subscribe_load_events()` — a PUSH channel of `ModelLoadEvent`, NOT an HTTP endpoint;
the embedder fans it out to its own UI transport). It emits `Queued` FIRST (before the
possibly-contended `lifecycle` lock, so the UI bar appears instantly), and a
`TerminalGuard`+drop guarantees a terminal `Ready`/`Failed` fires even on client
cancellation. `load_inner_impl` emits the mid-load phases at their real seams
(`Preparing` after the resident no-op check → `LoadingWeights` around the multi-second
worker load → `Finalizing` for the bookkeeping). Failure carries the diagnostic code on
the event.

```
   subscribe_load_events()  ──▶  broadcast::Receiver<ModelLoadEvent>

   Queued ─▶ Preparing ─▶ LoadingWeights ─▶ Finalizing ─▶ Ready   (terminal)
     │           │              │                             ▲
     └───────────┴──────────────┴───────────────────────────▶ Failed  (terminal, carries HGxxx)
```

A `LoadingGuard` publishes/clears the `loading` snapshot so a concurrent `status` can show
a progress indicator; the load itself walks the G5 OOM degrade ladder (`run_oom_ladder`:
plain retry → KV to system memory → fewer GPU layers, HG061 per rung, HG060 on
exhaustion) and syncs the saved tuning profile — for an ACCEPTED explicit load (anchored
to a hardware fingerprint + file signature captured BEFORE the slow load), AND for a REUSE
load the ladder had to DEGRADE (`loaded_np != requested_np`), so the next reload uses the
fitting config instead of re-walking the ladder. `set_profile` drops the record's stale
`provenance`/`bench_tps` whenever the written params differ from the saved profile — a
degraded fallback or an edited explicit reload was never the benchmarked config, so the
store must not claim its throughput.

## G6 Turbotune — measured autotune, EXCLUSIVE benchmark

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
  under `lifecycle` BEFORE the resident no-op), the chat gate's `resolve_loaded`
  (checked AFTER the awaited resident lookup — a pre-check would go stale across the
  await), `unload`/`unload_one`/`worker_stop` (a drain would evict the candidate
  mid-measure), and the public `chat_stream` local resolution, which SKIPS a
  benchmark-owned worker (`serve_public` flag) — with a fleet installed the chat can
  still route to a REMOTE copy of the same id. The bench's OWN loads / teardown /
  measurement ride private paths (`local.load`, `bench_unload_id`,
  `chat_stream_inner(serve_public = false)`), so it never refuses itself.
- The ONLY op not refused is a terminal `stop` (shutdown): `stop()` sets
  `shutting_down: AtomicBool` BEFORE draining, and `run_benchmark`'s per-candidate
  cancel hook polls it — the bench aborts cleanly with `[HG064]` (`BenchCancelled`)
  instead of iterating candidates against a draining node. (Normally the in-flight
  benchmark drains before `stop` runs; this is defense-in-depth.)

CANCELLATION SAFETY: if the benchmark future is dropped mid-candidate (client disconnect
/ a caller-imposed deadline), the OWNING `BenchmarkingGuard` — which tracks the LIVE
candidate's `WorkerId` in its `live` slot — spawns a task that unloads exactly that
worker and only THEN clears the [HG068] flag (clearing first would open a window where
a public chat/load could adopt the doomed candidate before the unload kills it). The
leftover candidate is thus auto-reclaimed instead of being served publicly with an
unchosen config. Worker-id-scoped (never id-scoped), so it cannot stomp a worker a user
loads after the flag clears; on every normal exit the slot is `None` and the flag is
cleared synchronously.

Because control is IN-PROCESS, a multi-candidate benchmark / multi-rung OOM ladder is
bounded only by the caller's own await — there is no HTTP request timeout to 408-cancel
it (the removed `/api/higgs/*` control routes once needed a 20-minute long-op group; the
`/v1` chat path is separately un-timed at the HTTP layer).

## Concurrency / locking model

`Higgs` mixes lock flavors by hold pattern; the doc-comment on each field states its
discipline. Summary:

- **`lifecycle: tokio::sync::Mutex<()>`** — serializes `load`/`unload`/`unload_one`/`stop`
  so two racing JIT loads of the same id can't each spawn a worker and a deliberate stop
  never interleaves with a load. Held across `.await`.
- **`hub_lifecycle: tokio::sync::Mutex<()>`** — separate serialization of the hub kill
  switch enable/disable across its whole check→start→publish / shutdown→clear sequence;
  SEPARATE from `lifecycle` so model ops and hub ops never cross-couple. Held across
  `.await`. (`hub_enable`/`hub_disable`/`pair`/`node_label` in `embed.rs` all take it.)
- **`inference_gate` / `remote_gate: Arc<Semaphore>`** — bound in-flight local vs remote
  chats at `MAX_CONCURRENT_INFERENCE` each. `try_acquire_owned` failure on the chat path
  → `ServerBusy` (503); the owned permit rides the spawned generation task and releases on
  any outcome. They are SEPARATE so remote traffic can't grow node tasks unbounded AND so
  it doesn't entangle the idle reaper's "acquire all LOCAL permits to unload" logic.
- **Atomics** (`AtomicBool`/`AtomicU64`, always set/read in isolation, never across an
  `.await`): the serve-layer toggles `log_incoming_tokens`, `jit_enabled`,
  `auto_unload_idle`, `idle_ttl_minutes`, `serving_enabled`, plus `shutting_down`
  (set once by `stop`; polled by the Turbotune cancel hook), `lan_override` (the
  manual LAN-exposure override for an embedder serving through its own stack) and
  `next_serve_id` (the monotonic id source for the serve registry).
  `lan_exposed` is NOT an atomic — it is computed: `lan_override` OR any live
  non-loopback listener in the `serves` registry below.
- **`parking_lot::Mutex<…>`** — held only across synchronous work, NEVER across an
  `.await`: `config`; the `*_io` RMW serializers (`config_io`, `models_io`, `keys_io`)
  that make each read-modify-write of `config.json`/`models.json`/`api_keys.json` atomic
  w.r.t. the others (the whole file is atomically temp+renamed, so last-writer-wins would
  otherwise clobber); the Turbotune `benchmarking` id set (set/cleared under `lifecycle`,
  read lock-free by the HG068 gates); the caches `device_cache`, `estimate_meta_cache`;
  the swappable installers `fleet`, `api_keys`, `hub`; the `loading` snapshot;
  `key_touch_throttle`; `config_path`; and the **serve registry** `serves`
  (`Vec<ServeSlot>` — one entry per LIVE `/v1` listener, holding its bound address,
  LAN flag, and the extra CORS origins its layer enforces). Lock order is
  `keys_io` → `serves`, never the reverse: `arm_lan_serve` and `revoke_key` both
  take `keys_io` first, which is what makes the LAN-arm and the last-key-revoke
  decision atomic w.r.t. each other.
- **`tokio::sync::Mutex<()>`** (async, MAY be held across an `.await`):
  `serve_lifecycle`, held by `serve_v1` across a listener's registration and across
  `ServeGuard::release()` + the `Higgs::stop()` that may follow. `stop()` is TERMINAL
  (`shutting_down` never resets), so a listener registering between a departing last
  listener's release and its stop would inherit a permanently dead facade.
- **`load_events: broadcast::Sender<ModelLoadEvent>`** — live PUSH fan-out; a `send` with
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
5. **Trusted control keeps the auth invariants.** `mint_key`/`revoke_key` skip ONLY the
   bearer check; they still enforce label validation, `Duplicate`, `BootstrapNeedsAdmin`
   ([HG066] last-admin), and the last-key-on-LAN refusal ([HG059], gated by the
   `lan_exposed()` state) via the shared `serve::control` decisions — so an in-process
   caller cannot flip auth off in a way the HTTP path forbade.
6. **Loopback-only by default.** `BIND_HOST` is `127.0.0.1`; `lan_exposed()` is COMPUTED
   from the live-listener registry (any deliberate non-loopback bind, or the manual
   `lan_override`), so the key-revoke gate can refuse dropping the last key while
   LAN-exposed — and the refusal lifts the moment the last such listener goes away.

## Error codes this module owns / raises

Raised directly in this folder (`api.rs`, `embed.rs`, `guards.rs`):

| Code | Variant | Where / meaning |
|------|---------|-----------------|
| HG014 | `ServerBusy` | `chat_stream` admission gate full → 503 (retryable). |
| HG015 | `InvalidModelId` | `guards::validate_repo_id` — bad charset / `..` traversal → 400. |
| HG017 | `InsufficientMemory` | `guards::guard_memory_headroom` — load exceeds `MEMORY_HEADROOM_FRACTION` of available RAM → 503. |
| HG040 | `PersistenceFailed` | `tune` / `sync_saved_profile` / `node_label` — `models.json`/`config.json` open/flush failed. |
| HG044 | `ChatTaskFailed` | `measure_gen_tps` — the spawned generation task join failed. |
| HG046 | `NotPrepared` | `resolve_loaded` JIT gate — the model was never Prepared (no saved profile). |
| HG047 | `ProfileStale` | `resolve_loaded` / `load_inner_impl` reuse of a stale saved profile (hardware/file changed since Prepare) — re-tune or load with explicit params. |
| HG060 | `LoadOomExhausted` | `run_oom_ladder` exhausted the OOM degrade ladder. |
| HG063 | `BenchExhausted` | `run_benchmark` found no working config. |
| HG064 | `BenchCancelled` | Turbotune bench aborted by a terminal shutdown (`stop` sets `shutting_down`; the per-candidate cancel hook trips). |
| HG067 | `BenchModelLoaded` | `turbotune_bench` — the model is loaded; unload it first to benchmark → 409. |
| HG068 | `BenchInProgress` | `load_inner_impl` / `resolve_loaded` / `unload` / `unload_one` / a second `turbotune_bench` — a benchmark owns the model; retry ~5 min → 503. |
| HG002 | `ModelNotFound` | `tune`/`estimate`/`model_by_id`/`resolve_loaded` scan miss for the requested id. |
| HG003 | `ModelNotLoaded` | `resolve_loaded` — the model is not resident and JIT is off. |
| — | `HubControlFailed` | `hub_enable`/`pair`/`node_*` — the server is not a hub, or an iroh op failed → 409/500. |

`[HG061]` (degrade-rung retry) and `[HG065]` (a failed bench candidate) are structured
LOG lines, not error variants. `[HG019]` `ServingDisabled` is raised by the SERVE layer
(`serve::v1`) when the facade's `serving_enabled` toggle is off. Codes like
HG001/HG005/HG016/HG018 originate in the node/worker/scan paths this facade delegates to.

## Boundaries / what does NOT belong here

- Worker-process lifecycle, RPC correlation, restart FSM → `supervisor.rs`.
- Multi-worker orchestration + the node idle reaper → `node/runtime.rs`.
- Remote fleet routing + served-instance ids → `node/fleet.rs`, `node/served.rs`.
- The row-formatting / mint-revoke decision / hub-status / tune-view PURE helpers that
  `embed.rs` delegates to → `serve/control.rs`.
- The candidate generation / benchmarked-pick / fit math for Turbotune → `tune/bench.rs`,
  `tune/vram.rs` (this module only orchestrates load→measure→cancel around them).
- Router assembly, the auth/host/CORS layers, and HTTP status mapping → `serve/mod.rs`,
  `serve/v1.rs`.

## Deferred / residual items

- **In-flight-load dedup on cancellation** (`load_inner_impl`): a load cancelled between
  `NodeMsg::Load` and its reply can let a racing same-id load spawn a duplicate worker.
  Self-healing (the cancelled load's `LoadCommit` reaps its worker) and RAM-bounded; a
  fully cancellation-safe dedup would change the node's additive contract — deferred.
  The SAME window exists when a Turbotune benchmark is cancelled while its candidate
  `local.load` is still pending: `live` is unset, so the guard clears the [HG068] flag
  and a racing public load can start before the doomed candidate load commits-and-reaps.
  Identical mechanism, identical self-healing bound — covered by this deferral.
- **Served-id stability** (`unload_one`/`chat_stream`): suffixed ids (`org/model-1`)
  renumber when a sibling is reaped, so a request from a stale snapshot can hit a different
  instance. A property of the served-id SCHEME; the stable worker-id/generation selector is
  deferred.
- **Per-load idle-TTL** (`LoadedInfo::idle_ttl_minutes`): reserved; the reaper applies one
  per-node TTL, so it is currently always absent.
- Prompt-throughput in `measure_gen_tps` is approximated from prompt tokens over TTFT;
  exact prefill timing is a worker-side refinement.
</content>
