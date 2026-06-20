# `src/node/` — remote-worker transport (iroh)

The node/hub remote-worker layer: pair two higgs instances over iroh QUIC and let a
**hub** borrow the GPUs of remote **nodes**. See `DESIGN.md` (this folder) for how it
fits together and `../../DESIGN-remote.md` for the full spec.

## Modules

| File | Responsibility |
|---|---|
| `identity.rs` | Persisted ed25519 `SecretKey` → stable `EndpointId`; `bind_endpoint` (N0 relay by default; `HIGGS_IROH_LOCAL` → relay-disabled local mode for tests). |
| `worker_id.rs` | `WorkerId(u32)` (Copy) + `WorkerRegistry` — monotonic, never-reused ids. |
| `runtime.rs` | `NodeRuntime` — the multi-worker orchestrator: `HashMap<WorkerId, Arc<Supervisor>>`, lifecycle (`load`/`unload`/`kill`/`status`/`sysinfo`/`scan`), cancellation-safe via `StopOnDrop`, graceful `shutdown_all`. |
| `control.rs` | `dispatch_node_control` — routes `higgs/node/*` control RPCs to `NodeRuntime`. |
| `data.rs` | `relay_chat` — bridges a hub chat stream to `Supervisor::chat()`, streaming `N_CHAT_CHUNK` + final back. |
| `mod.rs` | Endpoint bind/dial (`connect_node`), the post-HELLO gate (`gate_connection`), the persistent node serve loop (`serve_node` + per-stream dispatch). |
| `cli.rs` | `higgs link pair/status`, `higgs node connect`, `higgs --node` (persistent daemon). |
| `test_support.rs` | `#[cfg(test)]` fakes shared by the unit tests (fake worker, fake runtime). |

## Two roles, one binary

- **hub** — the higgs the UI/agents/clients hit; accepts node dials, gates them, issues
  `higgs/node/*` control RPCs and chat over iroh.
- **node** — a higgs that dials out to a hub and serves it: runs real llama.cpp child
  workers (one `Supervisor` each) and relays the hub's control + chat to them.

The hub never speaks to a remote worker directly — **two hops, never collapsed**:
`hub ──iroh──▶ node ──stdio──▶ worker`.

## Auth (two surfaces)

- **Surface A (machines):** `EndpointId` allowlist (`pairings.json`) + one-time pairing
  tokens — `../auth.rs`.
- **Surface B (callers):** Bearer API keys on `/v1` — lands in P5.

## Trying it (two terminals)

```sh
# hub: mint a pairing ticket + listen
higgs link pair
# node: dial the hub and serve it
higgs --node <ticket> <token>
```

## Tests

- Unit tests: inline `#[cfg(test)]` in each module (registry, dispatch, runtime, wire).
- Full workflow over a **real spawned `higgs` process** (pair → load real model →
  sysinfo cpu/mem → chat → status): `../../tests/remote_node_e2e.rs`.
- Pairing/handshake over real iroh: `../../tests/remote_pairing.rs`.
