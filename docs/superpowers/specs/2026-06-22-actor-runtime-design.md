# higgs Actor Runtime — Design & Plan

Actionable plan to convert higgs's fleet/runtime to a pure actor model. Detailed
message shapes, error handling, and edge cases are decided **during implementation**
(codex convergence on the code per `CLAUDE.md`), not pre-specified here.

## Goal

Make every stateful concurrency owner a message-passing **actor with private
state**, so the lock-based race classes (`HubFleet`/`NodeRuntime` TOCTOU) can't
exist. Unify local and remote behind one path. **Invisible to clients.**

## Locked model

```
Engine Runtime (actor)        ← the ONLY client surface; the library handle
  └─ owns Node Runtime(s)     ← never a Supervisor directly
        └─ owns Supervisor(s)
              └─ owns Worker   ← separate OS process (process-actor)
```

Invariants:
1. Two long-running actor roles: **Engine Runtime** (master/control + sole client
   surface) and **Node Runtime** (per machine, headless).
2. Strict ownership: Engine → Node Runtimes only; Node → Supervisors; Supervisor →
   one Worker. Engine owns **zero** direct Supervisors.
3. All actors, message-passing, private state, **no shared mutexes**.
4. **Worker = process-actor** (stdio mailbox, owns the model, crash-isolated).
5. **Local is not special** — a co-located local Node Runtime, same interface as
   remote.
6. **HubFleet = the Engine's owned read-model** of the fleet + routing table
   (synced from each node's inventory); the query surface for the jigglebot tabs.
7. **Clients talk only to the Engine.** `client → Engine → Node → Worker`.

## Key decisions (no code — just the rules)

- **OpenAI compat is invariant.** One URL (the Engine's); `/v1/chat/completions`
  and `/v1/models` unchanged on the wire; routing is internal. `/v1/models` keeps
  returning the aggregated servable set.
- **iroh = the transport for off-machine nodes only.** Local node = in-process
  channel, never iroh. One `NodeHandle` abstraction (Local | Remote) is the single
  seam; routing never branches on local-vs-remote.
- **Non-blocking handlers.** An actor `handle()` does only fast state work; slow
  downstream RPCs (load/scan/inventory/pull/chat) are **spawned and committed via a
  follow-up message** — so a slow load can't head-of-line-block retire.
- **Generation tokens, not locks.** Each node carries a generation bumped on
  retire/disconnect; a spawned op's commit applies only if the gen is unchanged. The
  fleet already has this as the `epochs` map (bumped after every load/unload/kill/
  route-drop, checked on inventory commit); the actor migration **promotes `epochs`
  to private actor state** so the commit gen-check needs no lock, and deletes the
  mutex-protected route/epoch maps rather than adding new guards.
- **Streaming bypasses mailboxes.** A chat request carries a sink (channel sender)
  threaded down at setup; the Supervisor's reader pumps tokens straight into it
  (Worker → Supervisor → sink → client SSE). Mailboxes carry only setup + final.
- **Duplicate model instances → deterministic served ids.** Same model on N
  workers = N instances; the served id is a deterministic function of the live set
  (`org/model`, `org/model-1`, …, sorted by node/worker), rebuildable from
  inventory, no persistence. `/v1/chat` targets a served id → exact (node, worker).
- **Local node placement: in-process actor** (workers are already process-isolated;
  a separate local process buys little). Remote nodes stay separate processes.

## Plan (each phase: tests + both coverage gates green + codex convergence)

- **P1 — Node Runtime → actor.** Convert `NodeRuntime` to a `spawn_actor` actor;
  keep its method API as thin mailbox wrappers. Removes node-side mutexes.
- **P2 — Supervisor → actor.** Wrap `Supervisor` as an actor. Its `ReplyDemux` is
  currently shared `Arc` + mutex-protected maps (`actor.rs`), so P2 must **privatize
  the pending-reply state into the actor** (mailbox-owned), not keep the shared demux
  — the reader/writer split stays but the correlation map moves behind the mailbox.
- **P3 — Engine Runtime actor; fold in HubFleet.** Fleet read-model + routing become
  Engine actor state (mutexes deleted); `/v1` + `/api/higgs/*` handlers send engine
  messages. **Dissolves the hub-side races.** Re-key routing by **served instance id**,
  not raw model id: today routes are `model → (node, worker)` (`fleet.rs` `routes`
  map) and `/v1/models` dedupes by id, so a second load of the same model overwrites
  the first — P3 must change the route/read-model key so N instances coexist (the
  deterministic-served-id rule above).
- **P4 — Local = a local Node Runtime.** Replace the `Higgs`-facade direct-local
  Supervisor path; local + remote share one path via `NodeHandle`.
- **P5 — Streaming via the sink handoff** end-to-end.
- **P6 — jigglebot four tabs** (My Servers / My Models / My Fleet) on the engine
  control surface, reusing existing bindings.

P3/P4 are the structural payoff; P1/P2 are mechanical scaffolding that keep the tree
green. Each phase lands behind the existing API so nothing breaks mid-migration.

## Testing & coverage (≥90% unit, ≥75% real integration)

- **Unit (≥90%):** actors are `handle(state, msg)` over private state with deps
  injected → drive the mailbox, assert state/replies, deterministically. The old race
  tests become deterministic sequence tests. The `NodeHandle` (Local | Remote) seam
  does not exist yet (today `api.rs` owns a direct local `Supervisor` and branches
  remote-fleet-vs-local-supervisor); it is **created in P4**, so P1–P3 unit tests use
  the existing `SupervisorSpawner` and the fake `NodeHandle::Local` arrives with P4.
- **Integration (≥75%, REAL):** live-iroh + spawned-process paths driven through the
  Engine's `/v1` + `/api/higgs/*` with a tiny GGUF. New live-iroh files go into
  `coverage-unit.sh`'s `--ignore-filename-regex` and are carried by
  `coverage-integration.sh`.
- **Required: two node runtimes on one machine.** Spawn **two** `higgs --node`
  processes on localhost (`HIGGS_IROH_LOCAL=1`), both paired into one Engine. Assert:
  both connected in `/api/higgs/nodes`; same model on both → two served ids in
  `/v1/models`; a streamed `/v1/chat` to each hits the right node; retire one → it
  leaves the fleet while the other keeps streaming.

## Out of scope / non-negotiable

- No change to the `/v1` wire shape, no second URL, no breaking `/v1/models`.
- Keep the host/worker OS-process split (crash isolation) and the worker's sync
  stdio loop.
