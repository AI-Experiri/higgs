---
title: Remote Fleet
description: higgs's distributed design — pair other machines over an encrypted iroh QUIC link and route OpenAI inference to their GPUs. Two-hop topology, pairing & identity, multi-worker nodes, durable routes, model pull, and log relay.
---

The remote fleet lets one higgs **hub** borrow the GPUs of other machines
(**nodes**) over an encrypted peer-to-peer [iroh](https://iroh.computer) QUIC
link. A model loaded on a node is served through the hub's ordinary
`/v1/chat/completions` — callers never know it ran on another box.

No central server, no open inbound ports: iroh handles NAT traversal and the
connection is authenticated end-to-end by each peer's cryptographic identity.

For the complete protocol spec, see `DESIGN-remote.md` and `src/node/DESIGN.md` in
the repository.

## Two hops, never one

```
 external client           HUB                          NODE                 WORKER
 POST /v1/chat   ─▶  resolve model → (node, worker)   serve_node loop     llama.cpp child
 (Bearer auth)       gate + per-node transport  ── iroh QUIC ─▶ control / data ── stdio ─▶ engine
                 ◀── stream chunks back ◀──────────────────────────────────◀── N_CHAT_CHUNK
```

The hub **never speaks to a remote worker directly.** It talks to the *node*,
which drives its own local llama.cpp children over stdio through the unchanged
[`Supervisor`](/system-design/#worker-model). This is deliberate: reusing
`Supervisor` as the per-worker unit means the worker stays pure-synchronous stdio
in *every* topology — the iroh boundary lives at the node's relay and the hub's
transport, never inside the worker.

Two invariants follow:

- **Reuse over reinvention.** The wire is the same `RpcFrame` NDJSON used locally;
  remote only *adds* the `higgs/node/*` control namespace and a HELLO handshake.
  The one net-new component is `NodeRuntime`, the node's multi-worker registry.
- **Control and data planes stay separate.** Control RPCs (`higgs/node/load`,
  `…/unload`, `…/inventory`) dispatch to `NodeRuntime`; chat (`M_CHAT`) relays
  through `Supervisor::chat`. A control RPC never reaches a worker's chat state,
  and chat is never multiplexed onto a control frame.

## Roles

| Role | What it is | How it runs |
|------|------------|-------------|
| **Hub** | The higgs server that owns the OpenAI clients and routing. | `HIGGS_HUB=1 higgs` — the serving hub: it serves HTTP **and** accepts node dials. (`higgs link pair`/`status` is a pairing-only CLI, not a serving hub.) |
| **Node** | A persistent daemon on a GPU machine that hosts workers. | `higgs --node <ticket> <token>` — dials the hub and stays connected. |
| **Worker** | One llama.cpp child, one model. | Spawned by the node's `NodeRuntime`, one `Supervisor` each. |

## Pairing & identity

Identity is **cryptographic**. A peer's `EndpointId` is its ed25519 public key,
and QUIC/TLS proves the dialer holds the matching private key. The HELLO message's
`node_id` must equal the TLS-verified remote identity, so a node cannot spoof
another's id.

Pairing admits a node's `EndpointId` to a persistent **allowlist**:

1. The hub issues a **single-use token** (short TTL) and a connection **ticket**
   — over the API (`POST /api/higgs/pair`) or the CLI (`higgs link pair`).
2. The node dials with the ticket + token. A **version-negotiated HELLO** runs
   inside a bounded post-connect window; a mismatch is a typed error, then close.
3. On success the node's `EndpointId` is added to the allowlist; subsequent
   reconnects need no token. `higgs link status` shows this hub's id and its
   paired-node count.

HELLO also advertises a node's **capabilities** (chat, model download, log
streaming) honestly — the hub only uses what a node actually implements.

## Transport

Each connected node has one `NodeTransport` on the hub — a client over the iroh
connection (ALPN `higgs/remote/1`) that issues control RPCs and opens chat
streams. Connections are compared by identity, so a stale failure can never tear
down a freshly reconnected transport.

## Multi-worker nodes

A node runs **many workers at once**. `NodeRuntime` is a registry of
`WorkerId → Supervisor`; each `WorkerId` is a monotonic `u32` that is **never
reused**, so a stale `(node, worker)` reference can't alias a different worker.
Loading a model on a node reuses the same load *procedure* as the local path:
resolve the id → path off the executor (with the same canonical-containment
guard), run the RAM-headroom check, default the context from the model, and
record the load so a respawn replays it. A node load accepts `ctx_len`,
`gpu_layers`, and `threads`, and rejects any parameter it can't honor.

## Routing & self-healing

The hub keeps **durable routes**: `model → (node, worker)`. They survive
reconnects, because a node's workers persist across a dropped connection.

- A `/v1/chat/completions` for a routed model is sent to its node transparently.
- If a remote worker is gone (a node restarted), the node reports worker-gone and
  the hub **drops the stale route** and refreshes its inventory — a later explicit
  load re-establishes a route. (The hub does not silently re-route on its own.)
- If a node is merely disconnected, calls fail fast with a clear error and the
  route is **kept** until it reconnects.
- An unload the hub owes a node that was offline is **reconciled on reconnect**,
  so a displaced worker is never leaked.

## Remote model pull

> **Status: node-side plumbing, not yet end-to-end.** The protocol op and the
> node handler exist, but no hub caller / HTTP route invokes it yet (`HubFleet`
> exposes load/unload/chat, not pull). The pieces below are implemented and tested
> on the node side; wiring a hub trigger is the remaining step.

higgs defines model pull **onto a node** over the wire (the `M_NODE_PULL` op): the
node fetches a GGUF from HuggingFace into its own `~/.higgs/models/` and
**progress streams back** (`N_PROGRESS`) as it downloads. The model is then
loadable on that node like any local file. The destination path is validated
against the node's scan roots — a download can't escape them.

## Log relay

A remote worker's stderr is **relayed back** into the hub's Developer-Log console,
filed under `LogSource::RemoteWorker { node, worker }`. In
[`/api/higgs/logs`](/endpoints/) you can filter it with
`?source=node:<node-id>:<worker-id>`, where `<node-id>` is the hub-local numeric
`NodeId` (not the iroh `EndpointId`) — so a model running on another machine is as
observable as a local one.

## Fleet view

With the server in hub mode, `GET /api/higgs/nodes` returns the whole fleet —
every paired node (connected or not), its stable id, connection state, hardware
snapshot, and resident workers (id → model). The node's own inventory reply is
authoritative and is refreshed after every hub-driven lifecycle change.

## Securing a hub

A hub that accepts node dials usually also exposes its HTTP surface beyond
loopback, so gate it with API keys (see [Overview → Security](/overview/#security)):
the `/v1` and `/api/higgs/*` surface becomes Bearer-authenticated the moment the
first key exists. The iroh control plane is *separately* protected by the pairing
allowlist — the two auth surfaces are independent (callers vs. machines).

## Safety under failure

The whole remote lifecycle is **cancellation-safe**. A dropped `Supervisor` does
not reap its child, so `NodeRuntime` guards every uncommitted worker with a
detached stop-on-drop, drains all workers on shutdown, and keeps a `Drop`
backstop. Control dispatch and the chat relay are tied to connection and stream
liveness, so a hub disconnect mid-chat can never orphan a worker on the node.
