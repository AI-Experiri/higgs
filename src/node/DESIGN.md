# `src/node/` — Design

Design notes for the iroh remote-worker layer. Full spec: `../../DESIGN-remote.md`.
Phase roadmap + decisions: `../../docs/superpowers/plans/2026-06-19-iroh-remote-roadmap.md`.

## Topology — two hops, never collapsed

```
   external client          HUB                         NODE                 WORKER
   POST /v1/chat  ─▶  resolve model→(node,worker)   serve_node loop     llama.cpp child
   (P5 Bearer)        gate + per-node transport   ── iroh ──▶ control/data ── stdio ──▶ engine
                  ◀── stream chunks back ◀────────────────────────────────◀── N_CHAT_CHUNK
```

The hub borrows remote GPUs. It never talks to a remote worker directly: it talks to the
**node**, which drives its own llama.cpp children over local stdio via the unchanged
`Supervisor`. Reusing `Supervisor` as the per-worker unit is deliberate — it already
bridges to the child's synchronous `serve_state`.

## Key design decisions

- **Reuse, don't reinvent.** The wire is the existing `RpcFrame` NDJSON; remote adds the
  additive `higgs/node/*` namespace + a HELLO handshake. `Supervisor` is the per-worker
  unit, unchanged. The net-new piece is `NodeRuntime` (the multi-worker registry).
- **The node relays via `Supervisor::chat()`, not `SyncIoBridge` into `serve_state`**
  (codex-validated correction to the spec). The worker therefore stays pure-sync stdio
  forever — the iroh boundary lives at the node's relay (and, in P3, the hub's transport),
  never inside the worker.
- **Two dispatchers, never merged.** Control (`higgs/node/*`) → `control::dispatch_node_control`
  → `NodeRuntime`. Data (`M_CHAT`) → `data::relay_chat` → `Supervisor::chat`. A control RPC
  never reaches `WorkerState`; chat is never multiplexed onto a control frame.
- **`WorkerId` is `u32` + `Copy`** so `LogSource::RemoteWorker` (P4) stays `Copy`. Ids are
  monotonic and never reused, so a stale `(node,worker)` reference can't alias.
- **Identity is cryptographic.** `EndpointId` = the ed25519 public key; QUIC/TLS proves the
  dialer holds the private key. The HELLO `node_id` MUST equal the TLS `remote_id()`
  (anti-spoof). Auth surface A = an `EndpointId` allowlist admitted by one-time pairing
  tokens (`../auth.rs`).

## Lifecycle & safety invariants

- **Post-HELLO gate** (`gate_connection`): the whole pre-HELLO window (including
  `accept_bi`, which iroh defers until the opener writes) is bounded by a deadline → HG028;
  version mismatch → typed HG023 *then* close; not-allowlisted/bad-token → HG024/HG022.
- **Cancellation-safe lifecycle.** A dropped `Supervisor` does NOT reap its child, so
  `NodeRuntime` guards every uncommitted/dropped worker with `StopOnDrop` (detached stop),
  drains on `shutdown_all`, and has a `Drop` backstop. Control dispatch and the chat relay
  are tied to connection + stream liveness so a hub disconnect can't orphan a worker.
- **Load mirrors `Higgs::load`**: resolve `id`→path off the executor (scan + canonical
  containment guard), RAM headroom guard, `ctx_train` default, and `record_last_load` so a
  respawn replays the model.
- **Honest capabilities & params.** HELLO advertises only implemented capabilities
  (`chat=true`; `download`/`log_stream` arrive in P4b/P4). `NodeLoadParams` uses
  `deny_unknown_fields` so a node rejects a param it can't honor rather than silently drop
  it. Worker-origin diagnostic codes (HG003/HG005/HG018) survive the relay.

## Phase status

P1 (pairing/HELLO) and P2 (NodeRuntime + control plane) are complete; the chat relay
(P3 core) is implemented and covered by the real-process integration test. Remaining P3
(hub-side per-node transport, `WorkerProc` trait, wedged-worker reap, HG027), P4
(inventory + `LogSource`), P4b (`M_PULL`), P5 (Bearer auth), P6 (UI) — see the roadmap.
