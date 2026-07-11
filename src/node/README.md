# `src/node/` — remote-worker fleet over iroh

Pair two `higgs` instances over iroh QUIC so a **hub** can borrow the GPUs of remote
**nodes**. A hub is the `higgs` server clients hit on `/v1`; a node is a `higgs` that dials
out and runs real llama.cpp child workers on the hub's behalf. The hub never talks to a
remote worker directly — **two hops, never collapsed**:

```
hub ──iroh──▶ node ──stdio──▶ worker (llama.cpp child)
```

See `DESIGN.md` (this folder) for the why/invariants and `../../docs/DESIGN-remote.md` for the
full spec.

## Files

| File | Responsibility |
|---|---|
| `mod.rs` | **Not a barrel** — it declares the child modules AND owns the live-iroh handshake surface: `HELLO_DEADLINE`, `HubIdentity`, `GateOutcome`, the two-phase hub gate (`gate_read_hello` lock-free + `gate_admit` locked, wrapped by `gate_connection`), the node dialers (`connect_node`, `dial_and_hello`, `send_leave`), the persistent node serve loop (`serve_node` → `handle_node_stream` + `relay_worker_logs`), and the shared frame helpers (`write_frame`, `worker_origin_code_data`, `close_after_reject`). |
| `runtime.rs` | `NodeRuntime` — the net-new multi-worker orchestrator, an **actor** (private `WorkerRegistry<Arc<Supervisor>>`, no mutex). Lifecycle (`load`/`unload`/`kill`/`status`/`scan`/`sysinfo`/`inventory`), idle auto-unload reaper (`IdleConfig`), lease-based chat (`ChatLease`), cancellation-safe teardown (`StopOnDrop` + tracked `reap` + `shutdown_all`), log + lifecycle-event fan-out. |
| `hub.rs` | Production hub listener: `start_hub` binds the endpoint, seeds/reuses the `HubFleet`, spawns `spawn_accept_loop` (gates each dial, registers admitted nodes), and returns the live `Hub` (mint pairings, `retire`/`set_label`/`labels`, `shutdown`). Also `serve_node_requests` (hub side of node self-`leave`). |
| `fleet.rs` | `HubFleet` — the hub's fleet read-model + `model→(node,worker)` routing table, an **actor** (was 7 mutexes). Node admission/retire, durable instance routes, per-node epochs, served-id derivation, the atomic `nodes_view`, and the remote ops `scan_node`/`load`/`unload`/`kill`/`chat`/`refresh_inventory`. `NodeView` is the ts-rs UI wire type. |
| `transport.rs` | `NodeTransport` — hub-side per-node client over one live iroh `Connection`: `request()` (one `higgs/node/*` control RPC per bidi stream) and `chat()` (relay `M_CHAT`, stream `N_CHAT_CHUNK` + final). One stream per call is the demux. |
| `data.rs` | Node-side DATA relay: `relay_chat` bridges a hub chat stream to `Supervisor::chat()` (via a `ChatLease`), `relay_pull` downloads a GGUF (`M_NODE_PULL`) streaming `N_PROGRESS`. Single writer per stream; cancelled on connection/stream drop. |
| `control.rs` | Node-side CONTROL dispatch: `dispatch_node_control` maps a `higgs/node/*` request to a `NodeRuntime` op and builds the JSON-RPC reply (carrying the origin HG code in `data`). |
| `served.rs` | `served_ids` — pure, deterministic served-instance-id derivation (`org/model`, `org/model-1`, …), generic over instance location so the remote fleet and the local engine share one algorithm. |
| `identity.rs` | Persisted ed25519 `SecretKey` → stable `EndpointId` (`load_or_create_secret`, atomic temp+hard_link publish, `0600`); `bind_endpoint` (N0 relays by default, `HIGGS_IROH_LOCAL` → relay-disabled LAN mode for tests). |
| `node_id.rs` | `NodeId(u32)` (Copy) + `NodeIdAllocator` — the hub's stable per-paired-node handle (`n-1`), distinct from the long `EndpointId`; used for `LogSource::RemoteWorker` and the UI. |
| `worker_id.rs` | `WorkerId(u32)` (Copy) + `WorkerRegistry<T>` — the node's monotonic, never-reused worker ids (`reserve`/`insert_reserved` for the load spawn-and-commit). |
| `cli.rs` | Hand-drive CLI: `higgs link pair/status` (hub), `higgs node connect/leave` (one-shot), `higgs --node [<ticket> [token]] | --list | --hub <sel>` (persistent daemon: dial → HELLO → `serve_node` → reconnect with backoff, saved hubs in `config.json`). |

`*_tests.rs`, `e2e_tests.rs`, and `test_support.rs` are the test sidecars (see the layout
rule in `../../CLAUDE.md`).

## Public surface (what the rest of the crate uses)

- **`hub::{start_hub, Hub}`** — there is **no `/api/higgs/*` HTTP control surface**; the embedder's
  `Higgs` facade (`../api/embed.rs`) drives the hub via the crate API. `Higgs::hub_enable()` calls
  `start_hub(bus, existing_fleet)` and holds the `Hub` alive; the rest of `Hub`'s methods back
  facade calls, not routes: `mint_pairing()`/`hub_id()` ← `Higgs::pair()`, `retire()` ←
  `node_retire()`, `set_label()` ← `node_label()`, `labels()` ← `nodes()`, `shutdown()` ←
  `hub_disable()` (the kill switch), and `serve_node_requests` accepts a node's self-`leave` on its
  own connection.
- **`fleet::{HubFleet, NodeView, NodeKey}`** — the `Higgs` facade routes `/v1` chat through the
  fleet (`api.rs`): `is_remote(served)` (remote-vs-local decision) + `resolve` / `chat(...)`
  (relay); `routed_models()` is folded into `Higgs::chat_model_ids()` for `GET /v1/models`;
  `nodes_view()` ← `Higgs::nodes()` (Fleet view, merged with allowlist labels + the local node);
  `load`/`unload` ← `node_load`/`node_unload` (`kill` is the force-unload variant);
  `served_on(node)` + `resolve` + `chat_pinned` ← `node_chat_test` (the Fleet view's
  per-node link proof — always relayed, never local: the `"local"` sentinel is refused
  outright [HG076], and the pin rides the same
  resolution that picks the transport so a concurrently re-homed id is refused
  [HG077], never mis-attested); `disconnect_all()` ← `hub_disable()` (kill
  switch). `NodeView` derives ts-rs bindings.
- **`runtime::{NodeRuntime, NodeConfig, IdleConfig, DEFAULT_IDLE_TTL}`** — the node daemon owns a
  `NodeRuntime`; the local single-machine engine also uses it as its own multi-worker orchestrator
  (`instances()` feeds served-id derivation, `events()`/`subscribe_logs()`/`bus()` feed the SSE +
  Developer-Log surfaces, `idle()` wires Server-Settings auto-unload).
- **`served::served_ids`** — reused by both the fleet and the local engine (P4b).
- **`{connect_node, dial_and_hello, send_leave, serve_node, gate_connection, HubIdentity,
  GateOutcome, HELLO_DEADLINE}`** — the transport handshake, used by `cli.rs` and `hub.rs`.
- **`identity::{load_or_create_secret, bind_endpoint}`**, **`node_id::{NodeId, NodeIdAllocator}`**,
  **`worker_id::{WorkerId, WorkerRegistry}`** — the identity/id primitives.

## Auth (two surfaces)

- **Surface A (machines):** `EndpointId` allowlist (`~/.higgs/pairings.json`) + one-time pairing
  tokens (`../auth.rs`). The dialer's TLS `remote_id()` proves it holds the key; the self-declared
  HELLO `node_id` must equal it (anti-spoof).
- **Surface B (callers):** Bearer API keys on `/v1` — outside this folder.

## Trying it (two terminals)

```sh
higgs link pair                 # hub: mint a pairing ticket + token, listen
higgs --node <ticket> <token>   # node: dial the hub, serve it, persist it for next time
```

## Tests

- Unit tests: the `_tests.rs` sidecars (registry, dispatch, gate, served-ids, fleet actor).
- End-to-end over a **real spawned `higgs` process**: `../../tests/remote_node_e2e.rs`,
  `../../tests/remote_pairing.rs`, `../../tests/remote_hub_e2e.rs` (route survives reconnect,
  kill switch), plus the in-process `e2e_tests.rs`.
</content>
</invoke>
