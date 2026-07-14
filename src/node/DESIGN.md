# `src/node/` — Design

Design notes for the iroh remote-worker layer. Full spec: `../../docs/DESIGN-remote.md`.
Roadmap + decisions: `../../docs/superpowers/plans/2026-06-19-iroh-remote-roadmap.md`.

## Topology — two hops, never collapsed

```
   external client        HUB                          NODE               WORKER
   POST /v1/chat  ─▶  resolve served→(node,worker)  serve_node loop    llama.cpp child
                     gate + per-node transport  ── iroh ──▶ control/data ── stdio ──▶ engine
                  ◀── stream chunks back ◀─────────────────────────────◀── N_CHAT_CHUNK
```

The hub borrows remote GPUs; it never talks to a remote worker directly. It talks to the
**node**, which drives its own llama.cpp children over local stdio via the unchanged
`Supervisor`. Reusing `Supervisor` as the per-worker unit is deliberate — it already bridges
to the child's synchronous `serve_state`, so the iroh boundary lives only at the node's relay
(`data::relay_chat`) and the hub's transport (`transport::NodeTransport`), never inside a
worker.

## Control plane — the embedder drives the crate API, not HTTP

There is **no `/api/higgs/*` HTTP control surface**. Fleet control is the in-process `Higgs`
facade (`../api/embed.rs`); the only HTTP the hub serves is the strict OpenAI `/v1`. So every
hub/fleet primitive in this folder is reached from a facade method, and node control rides the
iroh transport — `dispatch_node_control` — never an HTTP route:

```
  embedder                 Higgs facade (api/embed.rs)     this folder's primitives
  ─────────                ───────────────────────────     ────────────────────────
  .hub_enable() ──────────▶ start_hub(bus, fleet)  ───────▶ hub::start_hub → Hub + accept loop
  .hub_disable() ─────────▶ hub.shutdown()+fleet.disconnect_all
  .pair() ────────────────▶ hub.mint_pairing()  (ticket + one-time token)
  .node_retire()/.node_label()/.nodes() ─▶ hub.retire()/set_label()/labels(), fleet.nodes_view()
  .node_load(params?) ─▶ fleet.load(node, model, params?) (params ⇒ node's negotiated
                       protocol ≥ 2, else [HG078]; bare loads work against any node)
  .node_unload()/.node_scan() ─▶ fleet.unload()/scan_node()
  .node_chat_test() ─▶ "local"-sentinel gate ([HG076], pre-not-a-hub) ─▶ fleet.node_id() gate
                       ([HG075] unknown node) ─▶ fleet.served_on()+resolve()
                       ([HG074]/[HG076]) then fleet.chat_pinned() (always-remote, node-pinned at
                       dispatch — a re-homed id refuses [HG077], never mis-attests)
  POST /v1/chat (serve::v1) ─▶ Higgs::chat_stream ─▶ fleet.is_remote()/resolve()/chat()
                                                        │
                                       transport::NodeTransport (one iroh bidi stream per RPC)
                                                        │  higgs/node/*  |  M_CHAT / M_NODE_PULL
                                                        ▼
  on the NODE: mod::handle_node_stream fans each inbound method to the right plane:
        control ─▶ control::dispatch_node_control ─▶ NodeRuntime op
        data    ─▶ data::relay_chat / relay_pull   ─▶ Supervisor (via ChatLease) / GGUF download
```

## Core design decisions

- **Reuse, don't reinvent.** The wire is the existing `RpcFrame` NDJSON; remote just adds the
  additive `higgs/node/*` control namespace + a HELLO handshake (`crate::remote`). `Supervisor`
  is the per-worker unit, unchanged. The net-new piece is `NodeRuntime` (the multi-worker
  registry).
- **Two dispatchers, never merged.** Control (`higgs/node/*`) → `control::dispatch_node_control`
  → `NodeRuntime`. Data (`M_CHAT` / `M_NODE_PULL`) → `data::relay_chat` / `relay_pull`. A control
  RPC never reaches `WorkerState`; chat is never multiplexed onto a control frame.
  `mod::handle_node_stream` is the node-side fan-out that routes each inbound method to the right
  plane.
- **Identity is cryptographic.** `EndpointId` = the ed25519 public key; QUIC/TLS proves the
  dialer holds the private key. Both directions validate self-declaration against the TLS
  `remote_id()`: hub-side `gate_read_hello` requires `hello.node_id == peer` and `role == "node"`;
  node-side `validate_hub_hello` requires `role == "hub"`, matching `node_id`, and a version WE
  support (so a malformed reply can't poison the node's saved-hub state).
- **Stable ids, never reused.** `WorkerId(u32)`/`NodeId(u32)` are `Copy` so `LogSource::RemoteWorker`
  stays `Copy`; `WorkerRegistry` and `NodeIdAllocator` are monotonic and never rewind `next`, so a
  stale `(node, worker)` reference can never alias a different worker/node.

## Concurrency model — actors, not mutexes

Both stateful pieces are single-owner **actors** (mailbox + one owning task; see
`crate::actor` and `docs/superpowers/plans/2026-06-19-P0-actor-runtime.md`). A handler does only fast
synchronous state work; every slow downstream RPC runs OFF the actor thread, so a slow op can
never head-of-line-block a fast one.

### `NodeRuntime` (`runtime.rs`)

Private state = a `WorkerRegistry<Arc<Supervisor>>` + activity/in-flight maps behind one
mailbox — no mutex, so concurrent ops can't interleave across an `.await`.

- **Spawn-and-commit load.** `load` is the one op that mutates state AFTER a slow RPC. `Load`
  reserves an id synchronously, runs the slow `do_load` (resolve → RAM headroom guard → spawn
  Supervisor → `M_LOAD`) detached, then applies the registry insert via a `LoadCommit` message.
  So a slow load never blocks an unload/retire. `LoadCommit` delivers the caller reply FIRST
  then inserts (no `.await` between), so a cancelled caller reaps the worker instead of
  wedging a committed-but-unacked one.
- **Cancellation safety.** A dropped `Supervisor` does NOT reap its child. `do_load` holds a
  `StopOnDrop` panic-net; every teardown funnels through `reap` (a tracked task bumping
  `inflight_stops`, posting `ReapDone`); `shutdown_all` blocks until `inflight_loads +
  inflight_stops == 0`. A strong `keepalive` self-handle is held exactly while work is in
  flight, so the load→commit→reap→done chain drains in-actor even if every external handle
  drops mid-op. `ChatLease` (deref to `Supervisor`, held for the whole generation) posts
  `ChatEnd` on drop so the idle reaper measures idle from the END of a generation and never
  reaps a worker mid-chat.
- **Live idle policy.** `IdleConfig` (atomics + a `Notify`) is shared with the reaper task,
  which sleeps `reap_interval(ttl)` OR wakes immediately on change — so a Server-Settings TTL /
  on-off toggle takes effect without a restart. The reaper holds a `WeakHandle` so it never
  keeps an idle actor alive.
- **Domain gate at the dispatch choke point.** Every generation — the local `/v1` path, the
  in-process embed API, AND the hub relay (`data.rs relay_chat`, which never runs the facade's
  `resolve_loaded`) — takes its lease from the `ChatHandle` handler. That handler refuses the
  lease with `[HG079]` when the worker's `LoadFacts.domain` (captured when `do_load` RESOLVED
  the file, alongside its arch) is not `Llm`: llama.cpp will happily sample fluent nonsense
  from an embedding/reranker head, so the refusal must live where every dispatch converges,
  on load-time facts a post-load file deletion or scan failure can't reopen. The facade's own
  scan-derived `[HG079]` arms in `resolve_loaded` are fast-path courtesies, not the
  enforcement. Refusal is not a chat: no activity stamp, no in-flight hold, no `ChatStart`.

### `HubFleet` (`fleet.rs`)

Private state = `nodes` (connected → transport), `routes` (`(node, worker) → raw model`),
`node_ids`, `inventories` (+ dual `PulledAt` freshness stamps), `versions`/`software_versions`
(T14), `event_nodes`/`pending_pushes`/`fallback_inflight` and the `FleetEvent` sender (T10),
`epochs`, `admit_gen` — all behind one mailbox. This dissolved the
old 7-mutex TOCTOU class: compound transitions (`AdmitNode`, `RemoveInstanceIf`,
`CommitInventory`, `Retire`, `DisconnectAll`) are each ONE message, applied all-or-nothing, and
the cross-map `nodes_view` snapshot is atomic. Slow iroh RPCs run in the async wrapper methods
(fast state read → slow RPC off-actor → fast atomic commit message).

Two generation counters, not locks, guard the two races:

- **Per-node `epoch`** — bumped on every load/unload/kill/instance-drop/(re)admission. A slow
  `refresh_inventory` records `epoch_before`, and `CommitInventory` stores its (possibly stale)
  result only if the epoch is unchanged — so a slow connect-time fetch can't clobber newer state.
- **Monotonic `admit_gen`** — each accept loop is bound to the generation current when it
  started (`bump_admit_gen`). `AdmitNode` admits only while its gen still matches; the kill
  switch's `DisconnectAll` bumps the gen atomically with draining transports, so an admission
  task from the now-closing loop that races past disable is REFUSED (and a later re-enable's
  newer gen means it can never match again).

**Durable routes, transient transports.** The `--node` daemon reuses ONE `NodeRuntime` across
reconnects, so its workers/ids persist. Instance routes survive a dropped connection; only the
per-connection transport comes and goes (`drop_transport_if`, Arc-identity guarded so a stale
close-watcher can't drop a freshly reconnected transport). A genuine node-process restart leaves
stale routes that self-heal on the first worker-gone error (`route_invalidating`: HG006/HG007/
HG018 → instance dropped). Since T10, an APPLIED push or seq-ordered pull whose snapshot no
longer contains a routed worker also drops that route proactively (see below).

## Live fleet events (T10)

The pull model (connect / lifecycle / debounced chat-end re-pulls) is replaced, for capable
nodes, by node PUSH: every worker-state change on the node actor (chat start/end via
`ChatHandle`/`ChatEnd`, `LoadCommit`, `Unload`/`Kill`, the idle-reap sweep) emits a
`NodeFleetEvent { kind, snapshot_seq, workers }` — the FULL worker snapshot, sequenced by the
SAME actor counter as `Inventory` pulls, so pushes and pulls are totally ordered by node data
order. `serve_node` relays them on a dedicated uni stream (`N_FLEET_EVENT`; in-order by
construction); the `fleet_events` HELLO capability advertises it (additive, no protocol bump).

**Hub-side guard stack**, in order, all inside the `CommitWorkers` actor handler:

1. **Capability** — pushes are accepted only from an admission that DECLARED `fleet_events`
   (an undeclared admission stays a pure pull-model node; mixing freshness models on one
   cache made the T14 age tests flake).
2. **Transport identity (CAS)** — only the node's CURRENT transport may push; a buffered
   event from a replaced connection is dropped (its high seq would otherwise freeze out a
   restarted process whose counter restarted).
3. **Kind allowlist + lenient decode** — node-origin kinds only (`ChatStart`/`ChatEnd`/
   `WorkerLoaded`/`WorkerUnloaded`/`Resync`); a KNOWN hub-local kind off the wire is a
   protocol violation (no phantom `NodeDropped`), while an UNKNOWN future kind still applies
   its snapshot and re-broadcasts as the generic `InventorySynced`.
4. **Seq ordering** — newer-seq wins; the receive-stamp fallback exists only for seq-less
   caches (legacy pairs, post-re-admission strips) and NEVER reconciles routes (cross-process
   diffs could delete a just-installed route for a reused worker id).
5. **Route reconciliation** — under seq order, a worker present in the old snapshot but
   absent from the new one is really gone: its route drops with the merge (CAS'd on the old
   model, no epoch bump — commit-derived removals must not invalidate concurrent higher-seq
   pulls).

**Cache-less pushes** (the event outran the connect pull) are RETAINED (`pending_pushes`,
newest seq wins) and replayed on top of the next committed pull when newer; ONE
generation-tokened fallback pull runs per node (150 s hard bound, pre-pull ownership check,
5 retries with give-up that extends only for fresh pending data, Weak fleet ref) — further
cache-less pushes defer to it.

**Every `FleetEvent` emit lives inside the actor handler that performs the state change**
(admit → `NodeConnected`; drop/retire/kill-switch drain → `NodeDropped`; committed pull /
route install / seq-ordered reconciliation → `InventorySynced`; applied push → its kind;
hub enable/disable → `HubStateChanged`, emitted AFTER the state publishes), so subscribers
can never observe an event ordering that contradicts state order. Events are invalidation
signals, not data carriers — consumers re-read `nodes_view`.

**Relay robustness** (`relay_fleet_events`): subscribes in `serve_node` BEFORE the accept
loop (no first-event loss window); survives stream-level failures by watching
`send.stopped()` while idle and reopening; recovery emits a FRESH `Resync` snapshot via
`FleetResnapshot` rather than resending the stale event (frozen `idle_ms` must never be
re-stamped fresh at the hub — receipt-time stamping errs fresh-ward only by healthy-stream
transit, documented); both relays are `AbortOnDrop`-guarded so a cancelled `serve_node`
cannot leak them.

### Pairing-lock two-phase gate (`mod.rs`, `hub.rs`)

The one place with real locks is admission (`Arc<Mutex<Allowlist>>` + `Arc<Mutex<PairingTokens>>`),
split so a slow/malicious peer can't starve other joins or a concurrent pairing mint
(`Higgs::pair()` → `Hub::mint_pairing`):

1. **`gate_read_hello`** — lock-FREE. Bounds the whole pre-HELLO window (including `accept_bi`,
   which iroh defers until the opener writes) by `HELLO_DEADLINE`, caps buffered bytes
   (`MAX_HELLO_BYTES`), and runs the anti-spoof identity + version checks. A stalled peer is
   dropped here (HG028) without ever touching the pairing locks.
2. **`gate_admit`** — locks HELD by the caller. Synchronous in-memory allowlist/token decision
   (persist-then-burn) + a small post-auth reply. `hub.rs` registers the admitted node into the
   fleet INSIDE the same allowlist critical section as the admit, closing the admit→register
   window where a concurrent `Hub::retire` could re-introduce a just-retired node. Rejections
   write the typed frame under the lock but **grace-close (`close_after_reject`) only AFTER
   releasing the locks** — a rejected peer can stall the 2s close the full timeout, so holding
   the locks across it would serialize every other admission.

`gate_connection` is the lock-held convenience wrapper used by `cli.rs`; production `hub.rs`
uses the split path.

## Served-instance-id scheme (`served.rs`, `fleet.rs`)

Routes are keyed by INSTANCE — `(node, worker) → raw model` — so loads are ADDITIVE: N workers
serving the same model coexist as N instances (not only-keep-last). Each instance's SERVED id is
a deterministic, collision-free function of the live set: assigned in sorted `(model, location,
worker)` order against a global taken-set, the nth instance gets `org/model`, `org/model-1`, …,
and a candidate that clashes with a literal model name bumps to the next free suffix — so every
instance is reachable and the result is identical every time. It is derived on demand by
`served_ids`, **never persisted**, so a disconnect never renumbers survivors.

`served_ids` is generic over the instance LOCATION `L`, so the hub fleet (`L = NodeKey`) and the
local engine (P4b) share ONE algorithm. On the wire, `/v1` addresses a SERVED id but chat sends
the RAW model the worker loaded (`fleet::chat` → `transport::chat`), so a served suffix never
looks like a model mismatch (HG018). `nodes_view` tags each worker with `served_id_for_worker`,
which returns the served id only while the route's model still matches the worker's reported
model (a stale post-restart route yields `""` rather than an unreachable label).

## Error codes this module owns

Produced at origin here (relayed worker-origin codes like HG002/003/005/006/007/016/017/018 pass
through via `worker_origin_code_data`, they are not owned here):

| Code | Where | Meaning |
|---|---|---|
| HG022 | `gate_admit` | pairing token expired/used/unknown |
| HG023 | `gate_read_hello` / `validate_hub_hello` | protocol version mismatch (typed frame, then close) |
| HG024 | `gate_read_hello` / `gate_admit` | identity/role spoof, or not-allowlisted with no token |
| HG025 | `data::relay_pull` | GGUF download failure (per-fetcher classified) |
| HG026 | `control` (`M_NODE_UPDATE`) | node update unsupported (this build ships no updater) |
| HG027 | `fleet` (`NodeUnreachable`) | node not connected / dead transport |
| HG028 | `gate_read_hello` | handshake stalled past `HELLO_DEADLINE` |
| HG036 | `data::relay_pull` | both download fetchers exhausted |
| HG037 | `control` / `hub`/`node` stream dispatch | unknown method = protocol skew (→501, keeps -32601) |
| HG038 | `connect_node` / `validate_hub_hello` | protocol violation (unexpected/mismatched hub reply) |
| HG039 | `hub_rejection` | generic hub-request-rejected fallback (only for an uncoded rejection) |
| HG040 | `gate_admit` / `hub::serve_node_requests` | pairings-store persistence failure (disk/permissions) |
| HG051 | `transport::chat` reader | undecodable remote chat chunk dropped |
| HG057 | `data::relay_chat` | relayed chat stream overflowed (backlog dropped; RPC fails loudly) |

## Invariants

- The HELLO `node_id` equals the TLS `remote_id()` (both directions). A node can only ever
  retire ITSELF: `serve_node_requests` / `link_pair_post_admit` ignore any `M_NODE_LEAVE`
  payload and authenticate by the connection's peer id.
- `Hub::retire` holds the allowlist lock across BOTH the allowlist removal AND `fleet.retire`, so
  retire is mutually exclusive with admit+register — no concurrent admit can re-add the node in
  the gap. Node self-leave (`serve_node_requests`) does the DURABLE allowlist removal FIRST and
  only replies `left` if it persisted (crash-safe: a hub crash before `fleet.retire` still leaves
  the node gone, since it's no longer seeded from the allowlist on restart).
- Single writer per data stream (`relay_chat`/`relay_pull`): chunks/progress first, then final —
  never raced. Both are driven in a detached task holding the `ChatLease` so `Supervisor` cleanup
  runs even on a mid-chat hub disconnect, and are cancelled on `conn.closed()` / `send.stopped()`.
- `NodeLoadParams` base fields (ctx_len/gpu_layers/threads) stay authoritative on the node;
  `worker_load_params` OMITS absent base fields (never serializes `null`, which would fail the
  worker's required-`u32` deser and drop every rich override).

## Deferred / residual items

- **Cross-worker VRAM fit-check** (`do_load` NOTE): only the local RAM headroom guard runs; the
  §4.2b VRAM fit-check across workers is pending.
- **Remote sampling forwarding** (`relay_chat`): the hub→node wire carries only `temperature`; the
  rest of the sampler set / card-recommended base is not applied on the relay path (DESIGN-autotune
  §9).
- **Per-worker generation token** (`RemoveInstanceIf` residual): a node PROCESS restart that reuses
  the exact same worker id for the SAME model could in principle let a stale invalidation drop the
  new instance. Not readily reachable (the restart drops the transport first → `WorkerDead`, and
  unload/kill re-resolve per call); a full fix needs a per-worker gen token on the wire. The same
  missing token is the root of the T10 route/identity residuals below.
- **T10 accepted residuals** (each documented at its site in `fleet.rs`/`mod.rs`): ms-scale
  push freshness understatement on a healthy stream (receipt-time stamps); stale hardware
  readings under a fresh worker stamp on a chatty never-re-pulled node; a final event buffered
  across a disconnect is dropped (disconnected cache = last-known state; accepting it would
  reopen the stale-connection hazard); a route whose worker never entered the cache can still
  go stale (pre-T10 HG007 self-heal contract); a CAS-refused load's same-daemon orphan worker
  waits for the idle reaper (reap-by-id is unsafe across process restarts); a `serve_node`
  cancelled mid-stream leaves detached handlers until the hub's next timed-out op drops the
  transport.
- **`is_remote` breadth**: intentionally counts a served id whose node is currently DISCONNECTED
  (routes outlive a transport drop) so a chat yields the accurate HG027 (host offline → reconnect)
  rather than HG002. Accepted residual: a stale route to an offline node shadows a same-named
  local JIT load — narrow, degrades to a clear diagnostic.
- **Drop-without-`shutdown_all`**: `Drop` can't be async, so the sole orphan window is a
  `NodeRuntime` dropped then immediate `process::exit` with no `shutdown_all`. The daemon always
  calls `shutdown_all` first (fully awaited); `on_stop` is the best-effort awaited backstop; the
  reap-log-eviction race is bounded to a few in-flight lines.
</content>
