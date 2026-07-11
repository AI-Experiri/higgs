# higgs — Remote-Worker Design (iroh)

Status: **design spec, pre-code** (task #11). Branch `feat/iroh-remote`.
Rev 2 (2026-06-19): deps vetted on crates.io (§3.0, all ≥100k dl); `WorkerId` LOCKED to `u32`
(so `LogSource` stays `Copy`); node identity — hostname / OS / IP — added (§4.2.1).
Read alongside `DESIGN.md` (local worker lifecycle), `CONFIG.md` (config surface),
`src/supervisor.rs` / `src/worker/mod.rs` / `src/api.rs` / `src/rpc.rs` /
`src/log_bus.rs` / `src/system.rs` / `src/diagnostic.rs` (the seam). **Invent nothing**
where a standard exists; **declare net-new honestly** where one doesn't. The wire is a
pure transport swap under the existing `RpcFrame` NDJSON; new methods are additive `M_*`
extensions; auth = iroh `EndpointId` allowlist + OpenAI Bearer. But the **node-level
multi-worker orchestrator, `WorkerId`, the HF downloader, and the per-node hub transport
are NET-NEW code** — today's higgs is single-worker (one `Supervisor` = one child, only
keep-last). This doc marks reuse vs net-new at every seam.

## Table of Contents
- [1. Goal + Topology](#1-goal--topology)
- [2. Reuse vs Net-New](#2-reuse-vs-net-new)
- [3. iroh Transport](#3-iroh-transport)
  - [3.0 Vetted dependencies (Crate-First)](#30-vetted-dependencies-crate-first)
- [4. The Wire — HELLO + M_* + Streams](#4-the-wire--hello--m_--streams)
  - [4.2.1 Node identity — name / OS / IP](#421-node-identity--name--os--ip-for-display)
- [5. The higgs Seam — file:line](#5-the-higgs-seam--fileline)
- [6. LogSource 2→N](#6-logsource-2n)
- [7. Auth — Two Surfaces](#7-auth--two-surfaces)
  - [7.1 New diagnostics — HG022–HG027](#71-new-diagnostics--hg022hg027)
- [8. Settings/Home + the lean CLI](#8-settingshome--the-lean-cli)
- [9. Updates — binary vs home](#9-updates--binary-vs-home)
- [10. Phased Plan (tasks #12–#18)](#10-phased-plan-tasks-1218)
- [11. Open Decisions / Risks](#11-open-decisions--risks)

---

## 1. Goal + Topology

**Goal.** Let one higgs (the **hub**, the one the UI/agents/external clients already
talk to) borrow the GPUs of other machines running higgs (**nodes**), so a chat for a
model resident on a remote node streams back as if it were local. No new wire format,
no change to the llama.cpp worker, no inbound port on the nodes. A node may host **many
concurrent workers** (one per loaded model) — that multi-worker capability is **net-new
at the node layer** (§2).

```
   phone / laptop NODE (behind NAT/CGNAT)          desktop HUB (also behind NAT)
   ┌──────────────────────────────┐                ┌────────────────────────────────┐
   │ higgs --node                 │                │ higgs (embedded in jigglebot)   │
   │   EndpointId = N_a           │ ── DIAL ─────▶ │   EndpointId = H                │
   │   ~/.higgs/endpoint.key      │   iroh QUIC    │   allowlist { N_a, N_b, … }     │
   │   NodeRuntime (NEW):         │  (relay +      │   /v1 + /api/higgs/* + UI       │
   │     HashMap<WorkerId,        │   hole-punch)  │   per-node iroh transport (NEW) │
   │       Arc<Supervisor>>       │                │   HashMap<NodeId,NodeView>      │
   │     each Supervisor → 1 child │                └────────────────┬───────────────┘
   └──────────────┬───────────────┘                                 ▲
                  │ N× stdio NDJSON  (one per worker)     external client (OpenAI SDK)
                  ▼ (existing supervisor.rs, reused)      POST /v1/chat  Bearer sk-…
          llama.cpp child WORKERS  (1..N)                         (surface B)

   hub ── controls ──▶ node ── controls ──▶ worker      TWO HOPS, NEVER COLLAPSED
   (a higgs)           (a higgs)            (llama.cpp child, local stdio)
```

**Definitions (no conflation):**

| Term | Is | Owns | Talks to |
|---|---|---|---|
| **hub** | the higgs the user/UI/agents/external clients hit | allowlist, API keys, aggregate `/v1` + `/api/higgs/*`, fleet inventory, **per-node iroh transport (NEW)** | nodes (iroh) + its own local worker (stdio) |
| **node** | a remote higgs with an uplink — *a hub-with-an-uplink* | **`NodeRuntime` (NEW): a registry of `WorkerId → Arc<Supervisor>`**, its own disk/models | hub (iroh, dials out) + its workers (stdio) |
| **worker** | the llama.cpp FFI child — **unchanged** | one resident model | only its node, over local stdio; **one `Supervisor` per worker (reused as-is)** |

The hub **never** speaks to a remote worker directly. It speaks to the node; the node
speaks to its workers over the *existing* local stdio path. Two hops, never one.

---

## 2. Reuse vs Net-New

The **transport seam** already exists and is genuinely reused. The **node-level
multi-worker orchestrator does not exist** and is net-new. State both honestly.

### 2.1 What is REUSED (transport-generic, ~zero change)

The supervisor was written transport-agnostic on purpose (`supervisor.rs:28-30` — "No
transport trait. … the mpsc channel serialises concurrent callers"). One `Supervisor`
drives one llama.cpp child over one NDJSON wire, and nothing in it cares whether the
bytes cross stdio or a QUIC stream.

```
                 REUSED — one Supervisor = one worker, unchanged
  ┌──────────────────────────────────────────────────────────────────────┐
  │  RpcFrame NDJSON wire ........................ rpc.rs (whole file)     │
  │  id-correlation pending map .................. supervisor.rs:1120      │
  │  per-request chat_sinks (request_id echo) .... supervisor.rs:164-167   │
  │  write_tx mpsc serializer / writer_task ...... supervisor.rs           │
  │  dispatch / reader_task frame routing ........ supervisor.rs           │
  │  M_* worker method vocabulary ................ worker/mod.rs:19-37     │
  │  serve_state sync loop + engine FFI .......... worker/mod.rs (UNCHANGED)│
  │  one Supervisor as the PER-WORKER unit ....... supervisor.rs (UNCHANGED)│
  └──────────────────────────────────────────────────────────────────────┘
```

### 2.2 What is NET-NEW (does not exist today)

```
                 NET-NEW — node-level multi-worker + transport + downloader
  ┌──────────────────────────────────────────────────────────────────────┐
  │  NodeRuntime  (src/node.rs) ........ owns HashMap<WorkerId,Arc<Supervisor>> │
  │       N concurrent Supervisors per node — NEW orchestration layer      │
  │  WorkerId  (newtype) ............... per-node worker key (NEW, §5.4a)   │
  │  hub per-node iroh transport ....... own pending/correlation per NodeId │
  │       — SEPARATE from the hub's local-worker Supervisor (§2.3, fix B)  │
  │  WorkerProc trait .................. abstracts proc:Option<Child> leak  │
  │  remote_factory / spawn_remote ..... iroh stream halves into a Supervisor│
  │  LogSource::RemoteWorker{node,worker} keyed N-way log rings (§6)        │
  │  HF downloader + N_PROGRESS ........ M_PULL — NO download exists today  │
  │       (worker is read-only scan, worker/mod.rs:108,228)  (§4.4, fix D) │
  │  iroh Endpoint bind + accept + HELLO gate + pairing/allowlist          │
  └──────────────────────────────────────────────────────────────────────┘
```

**Why `NodeRuntime` and NOT `Higgs`/`Supervisor` for multi-worker:**

| Type | Today | Why it can't carry N workers |
|---|---|---|
| `Supervisor` | owns ONE `proc: Mutex<Option<Child>>` (`supervisor.rs:201`); `running: AtomicBool` (`:190`) **refuses a 2nd spawn** (`:762-765`); `stop(&self)` takes **no id** (`:625`) | one Supervisor is structurally one child — keep it that way (the per-worker unit) |
| `Higgs` | owns ONE `sup: Arc<Supervisor>` (`api.rs:337`); `load` is **only-keep-last**, "higgs serves one model at a time" (`api.rs:363-364`); replaces the resident model on each load | only-keep-last by construction — routing N workers through it would mean rewriting its core invariant |

So: **reuse `Supervisor` as the per-worker unit (unchanged); add a NEW `NodeRuntime`
that owns `HashMap<WorkerId, Arc<Supervisor>>` and manages N of them.** Do not route
multi-worker through `Higgs`.

### 2.3 Two correlation domains, NEVER collapsed (fix B)

The hub runs **two physically distinct transports**, each with its own pending /
correlation map:

```
  HUB
  ├─ local-worker Supervisor  (supervisor.rs, wired to the hub's OWN local child stdin :194)
  │     pending keyed by id; chat_sinks keyed by request_id  — LOCAL stdio only
  │
  └─ per-node iroh transport  (NEW, src/node.rs hub side)
        one connection per NodeId; its OWN pending/correlation
        routes per (NodeId, WorkerId);  bytes cross QUIC, never stdio
```

The hub MUST NOT reuse its local-worker `Supervisor`'s stdin/pending for hub↔node iroh
traffic — that stdin belongs to the hub's *own* local llama.cpp child. The per-node iroh
transport is a separate construct keyed by `NodeId`, routing per `(NodeId, WorkerId)`.
**Never merge the two correlation domains.** (The hub-side chat *relay shape* — a pending
map keyed by id, chat_sinks keyed by request_id — is the same *pattern* as the
Supervisor's, but it is a separate instance on the iroh side, not the local one.)

### 2.4 The single transport seam (HalvesFactory)

Where iroh enters the *per-worker* path is exactly one place: the factory closure.

```
  Box<dyn AsyncWrite> + Box<dyn AsyncRead>  ← iroh SendStream/RecvStream
       instead of child.stdin / child.stdout
  proc: Option<Child>  →  proc: Option<Box<dyn WorkerProc>>
       (local = SIGKILL ; remote = QUIC conn close)
```

**The seam is `HalvesFactory`** (`supervisor.rs:148`):
```rust
type HalvesFactory =
    Box<dyn Fn(Arc<LogBus>, &str) -> Result<WorkerHalves, HiggsError> + Send + Sync>;
```
It returns `WorkerHalves { write, read, proc }`. iroh's bidi-stream halves **already**
implement tokio `AsyncWrite`/`AsyncRead` + `Send + Unpin`, so they box straight into the
existing `WriteHalf`/`ReadHalf` slots (`supervisor.rs:120,122`). The supervisor's request
correlation, chat routing, writer/reader tasks, codec, and restart FSM consume *only* the
boxed dyn halves — none know the bytes now cross a QUIC stream. iroh enters here exactly as
`production_factory` captures `current_exe()` and nothing else.

### 2.5 The actor model — one hand-rolled runtime, not a crate (Crate-First, assessed)

higgs's worker management **is** an actor model, built **hand-rolled on tokio channels** — the
same methodology as jigglebot's agent layer (`backend/engine`: tokio mpsc/oneshot/broadcast; **no
actor-framework crate anywhere in the workspace**). To avoid re-hand-rolling the loop per actor,
the mailbox + receive-loop + spawn + shutdown machinery is factored into **one shared `actor`
module, written once**, and every actor builds on it:

```rust
// src/actor.rs (NEW, foundational) — the runtime is written ONCE:
trait Actor { type Msg; async fn handle(&mut self, msg: Self::Msg); }
fn spawn_actor<A: Actor>(state: A) -> Handle<A::Msg>;   // mailbox + recv loop + shutdown
```

Each actor contributes only its **own message set + `handle`** — "serve different messages based
on what it does." Nobody re-implements the loop:

| actor | role | `Msg` it handles | contributes | reuses |
|---|---|---|---|---|
| **Worker** | server (child process) | `WorkerReq`: Load·Chat·Unload·Sysinfo·Probe | `handle` = dispatch the engine | `spawn_actor` |
| **Supervisor** | RPC client (parent) | `SupCmd`: Request·Chat·Stop·SetVerbose | `handle` = forward + register `pending` | `spawn_actor` + reply-demux\* |
| **NodeRuntime** | node supervisor | `NodeOp`: Load·Unload·Kill·Scan·Pull·Sysinfo | `handle` = registry + lifecycle ops | `spawn_actor` |
| **per-node transport** | RPC client (hub) | hub→node ops | `handle` = forward over iroh | `spawn_actor` + reply-demux\* |

\*reply-demux = the client half (a reader task correlating inbound replies → `pending` /
`chat_sinks`); also written **once** and shared by the two RPC clients (Supervisor + transport).
The Worker is a server — replies inline, no demux. State stays per-actor: `Supervisor` keeps
pending/chat_sinks/proc (:152-201); `Worker` keeps `WorkerState{engine,loaded}`
(worker/mod.rs:110-114). Each is "an isolated unit waiting on its mailbox, reacting, holding
state" — the actor definition; the worker is the purest (a share-nothing OS process,
crash-isolated by re-exec).

> **Worker rides the same runtime.** To use `spawn_actor`, the worker process runs a minimal tokio
> runtime and its blocking FFI (`engine.chat`) goes through `spawn_blocking` — already the pattern
> the remote data path uses (§5.3). It stays a separate, crash-isolated process; it just shares
> the actor runtime instead of a hand-rolled sync stdin loop. **Foundational refactor lands first
> (P0):** factor `actor.rs` out of today's `Supervisor`, then port `Worker`, `NodeRuntime`, and the
> transport onto it.

**Why one hand-rolled runtime, NOT an actor-framework or job-queue crate** (apalis, asynq,
taskflow-rs, distributed-scheduler, pueue, ractor, actix, processmanager — assessed 2026-06-19):
a small `trait Actor` + `spawn_actor` over tokio *is* the whole runtime; a framework adds a
dependency + paradigm absent from the workspace and *still* doesn't supervise the hard part:

| crate class | manages | why it's the wrong tool here |
|---|---|---|
| job queue (apalis, asynq, taskflow-rs) | stateless tasks pulled from a broker, run to completion | a worker is a persistent GPU process with a *streaming* bidi RPC channel, not a discrete job; forces a Redis/Postgres/AMQP broker the p2p design has no place for |
| cluster scheduler (distributed-scheduler) | cron tasks across a cluster via Redis/etcd/Consul | no cluster, no coordination store — iroh is the transport |
| CLI daemon (pueue) | shell commands in a local queue | a CLI tool, not an embeddable library |
| actor framework (ractor, actix, processmanager) | **in-process** tokio actors / tasks | supervises in-process objects, not an external OS child holding VRAM; one-msg→one-reply fights streaming `N_CHAT_CHUNK`; the real failure mode is child `exit()`, not an actor panic. Our `spawn_actor` is smaller, fits the external-process + streaming reality, and matches the workspace methodology |

`RpcFrame` (the wire) + the transport-generic `HalvesFactory` seam (§2.1, §2.4) stay the shared
substrate; `NodeRuntime` adds only the registry + routing as another `Actor`. **The only external
crates this feature needs are transport (`iroh`) and model download (`hf-hub`)** (§3.0).

> **One real (optional) slot.** An `M_PULL` HuggingFace download IS a genuine discrete,
> run-to-completion job, so `apalis` with a **SQLite backend (no new broker)** could add
> retry/backoff + resume-across-restart to the *download path only* — it never touches worker
> supervision, chat RPC, or routing. Deferred: `hf-hub` alone covers P4b; adopt `apalis` only if
> resumable downloads become a requirement.

---

## 3. iroh Transport

Crate versions (verified against docs.rs, iroh 1.0.0): `iroh = "1.0"`, `iroh-tickets = "1.0"`,
`iroh-base = "1.0"`. MSRV 1.91. **No `quinn`, no `irpc`, no `quic-rpc`** — our NDJSON `RpcFrame`
rides raw `open_bi`/`accept_bi` streams directly (confirmed VIABLE, §2.4, §3.3). `Cargo.toml`
deps must be added — none present today (verified: `Cargo.toml` `[dependencies]` has no
iroh/quinn/iroh-tickets/iroh-base). **`iroh-tickets` is a SEPARATE crate** — `EndpointTicket`
is NOT re-exported by iroh core (§3.2).

### 3.0 Vetted dependencies (Crate-First)

Every net-new crate was checked on crates.io (2026-06-19) per the Crate-First rule; **all are
≥100k all-time downloads** — none fell under the bar. Lowest of the set is `iroh-tickets`
(125k), unavoidable (`EndpointTicket` is not re-exported by iroh core) and from the same iroh
1.0 release as the rest. `WorkerId` and node-IP need **no crate at all** (see notes).

| crate | role | all-time dl | phase |
|---|---|---|---|
| `iroh` | QUIC p2p transport | 918k | P1 |
| `iroh-base` | `SecretKey` / `EndpointId` / `EndpointAddr` | 924k | P1 |
| `iroh-tickets` | `EndpointTicket` (pairing string) | 125k | P1 |
| `toml` | node `config.toml` parse | 694M | P2 |
| `hf-hub` | `M_PULL` model download | 10.2M | P4b |
| `sha2` | API-key SHA-256 hash | 695M | P5 |
| `subtle` | constant-time key compare | 541M | P5 |
| `qrcode` *(optional)* | pairing QR (else plain ticket string) | 15.5M | P6 |
| `minisign-verify` *(deferred)* | update-artifact signature verify | 8.2M | #18 |

**Deliberately NO crate (use a primitive / existing dep) — each also closes a risk:**
- **`WorkerId` = `u32` newtype** (`Copy`) — *not* `smol_str`/`compact_str`. Wire carries it as a
  number (`"worker_id": 1`). A `u32` is `Copy`, so `LogSource` stays `Copy` (closes risk #5) —
  see §5.4a / §6.
- **Node IP = iroh `Connection::paths()` → `Path::remote_addr()`** (`TransportAddr::Ip(SocketAddr)`;
  `Path::is_ip()`/`is_relay()` = direct-vs-relay) — *not* `local-ip-address` / `if-addrs`. iroh
  already discovers direct LAN/WAN socket addrs during NAT traversal (§4.2.1).
- **hostname / OS = `sysinfo`** (already a dep) — `System::host_name()` / `name()` /
  `os_version()`, folded into `HardwareInfo` (§4.2.1).

`reqwest` (already a dev-dep) is pulled in by `hf-hub` as its async HTTP client; promote it to a
normal dependency only if a direct HTTP call outside `hf-hub` is ever needed.

> **iroh 1.0 API reference (verified against live docs.rs — no PROVISIONAL items remain):**
> - Identity = **`EndpointId`** (`= PublicKey`); **`NodeId` does NOT exist in iroh.** In this
>   doc "NodeId" is the higgs-domain newtype that **WRAPS** iroh's `EndpointId` —
>   `NodeId(EndpointId)` — never an iroh type name (§3.1, §6).
> - Address = **`EndpointAddr`** (NOT `NodeAddr`). Ctors: `EndpointAddr::new(id)`,
>   `EndpointAddr::from_parts(...)`, `.with_relay_url(RelayUrl)`, `.with_addrs(impl
>   IntoIterator<Item = TransportAddr>)`, `.with_ip_addr(SocketAddr)`. **`with_direct_addresses`
>   does NOT exist** (§3.2, §7).
> - Allowlist accessor = **`Connection::remote_id() -> EndpointId`** (NOT `remote_node_id()`,
>   NOT `node_id`) — connection types live under `iroh::endpoint::*` (§3.2, §3.3).
> - Ticket = **`iroh_tickets::endpoint::EndpointTicket`** from the separate `iroh-tickets`
>   crate (NOT `NodeTicket`, not re-exported by iroh). Accessor **`.endpoint_addr()`** (NOT
>   `.node_addr()`). String round-trip via `Display`/`FromStr`; bytes via the `Ticket` trait's
>   `encode_bytes`/`decode_bytes` (NOT `to_bytes`/`from_bytes`) (§3.2, §7).
> - Bind with persisted key = **`Endpoint::builder(preset).secret_key(sk).bind().await`**
>   (async); `iroh::SecretKey::from_bytes(&[u8;32])` (infallible, by ref) ↔ `to_bytes() ->
>   [u8;32]`. The `Endpoint::bind(preset)` shortcut takes NO key (§3.1).
> - Streams = `Connection::open_bi()`/`accept_bi()` are **named futures** (not `async fn`);
>   `.await -> Result<(SendStream, RecvStream), ConnectionError>`. `SendStream`: tokio
>   `AsyncWrite`; `RecvStream`: tokio `AsyncRead`; both boxable (§3.3).
> - Peer address = **`Connection::paths() -> PathList`**, iterate `Path`; `Path::remote_addr() ->
>   &TransportAddr` (`Ip(SocketAddr)` | `Relay(RelayUrl)`); `Path::is_ip()` / `is_relay()` for
>   direct-vs-relay. **No `Connection`-level address accessor; `ConnectionType` was REMOVED in 1.0**
>   (used by §4.2.1 `NodeView`).
> - Accept loop = `Endpoint::accept().await -> Option<Incoming>` → `incoming.await? ->
>   Connection`; gate on `Connection::remote_id()` AFTER the await (§3.2). ALPN via builder
>   `alpns(Vec<Vec<u8>>)`.

### 3.1 Endpoint bind with a persisted SecretKey

The `EndpointId` *is* the ed25519 public key — persist the 32 secret bytes and the id is
stable across restarts. Single canonical home: `~/.higgs/endpoint.key` (raw 32 bytes,
chmod 0600). `ALPN = b"higgs/remote/1"`.

```rust
use iroh::{Endpoint, SecretKey, endpoint::presets};
const ALPN: &[u8] = b"higgs/remote/1";

fn load_or_create_secret(path: &std::path::Path) -> anyhow::Result<SecretKey> {
    if let Ok(bytes) = std::fs::read(path) {
        let arr: [u8; 32] = bytes.as_slice().try_into()?;
        Ok(SecretKey::from_bytes(&arr))                 // deterministic id
    } else {
        let sk = SecretKey::generate(&mut rand::rngs::OsRng);
        std::fs::write(path, sk.to_bytes())?;           // [u8;32], chmod 0600
        Ok(sk)
    }
}

let sk = load_or_create_secret(&home.join("endpoint.key"))?;
let endpoint = Endpoint::builder(presets::N0)           // n0 relay + discovery defaults
    .secret_key(sk)                                     // stable EndpointId
    .alpns(vec![ALPN.to_vec()])                         // required to ACCEPT inbound
    .bind().await?;                                     // Result<Endpoint, BindError>
let my_id = endpoint.id();                              // EndpointId, stable across restarts
```
Presets: `N0` (relay + n0 DNS/pkarr discovery), `N0DisableRelay`, `Minimal`, `Empty`.

### 3.2 Dial / accept / allowlist gate

| Who | Verb | API |
|---|---|---|
| node → hub (reconnect) | dial by id | `endpoint.connect(hub_id, ALPN).await?` — discovery re-resolves the path |
| node → hub (first contact) | dial by ticket | `endpoint.connect(ticket.endpoint_addr().clone(), ALPN).await?` — `EndpointTicket::endpoint_addr() -> &EndpointAddr` (verified, iroh-tickets 1.0) |
| hub | accept conn | `while let Some(incoming) = endpoint.accept().await { let conn = incoming.await?; … }` — `accept() -> Option<Incoming>`, `incoming.await -> Connection` |
| hub | accept control bi | `let (send, recv) = conn.accept_bi().await?` (opener writes HELLO first, §3.3) |
| hub | authenticate | `conn.remote_id() -> EndpointId` — cryptographically proven from the peer TLS cert |
| hub | gate (POST-HELLO) | read HELLO; `if !allow.contains(&peer) && !valid_token { conn.close(…) }` (HG024 / HG022) |
| hub | gate (HELLO-stalled) | bounded timer from `accept()`; no HELLO within window → drop (HG028, §3.2.1) |

`remote_id()` is the auth primitive: QUIC/TLS already proves the dialer holds that
`EndpointId`'s private key. **The allowlist gate is POST-HELLO, not at `accept()`** —
because `accept_bi()` only fires after the opener writes (§3.3), and the HELLO frame carries
the `pairing_token` that admits a not-yet-allowlisted id. The explicit sequence:

```
accept conn → arm HELLO-deadline timer → conn.accept_bi() → read first frame = HELLO
  → if no HELLO before deadline                       → drop conn  HG028 HandshakeStalled (§3.2.1)
  → if remote_id ∈ allowlist                          → admit (paired node)
  → elif HELLO.pairing_token valid (unexpired/unused) → admit + add to allowlist (§7)
  → else                                              → typed close HG024 NotAllowlisted
                                                         (or HG022 PairingTokenInvalid)
```
The allowlist is just the set of paired `EndpointId`s — no app-level password. The one
exception that lets an id *enter* the allowlist is a valid pairing token in the HELLO
frame (§7).

#### 3.2.1 Post-HELLO gate timeout (fix E)

A peer that completes QUIC/TLS then **never sends HELLO** would otherwise hold the
connection (and the `accept_bi()` await) indefinitely — a pre-auth DoS, since `accept()`
admitted it before any allowlist check. **Mitigation:** arm a bounded timer
(`HELLO_DEADLINE`, default 5 s) the moment `accept()` returns; if no HELLO frame arrives
before it fires, `conn.close(..)` and emit **HG028 HandshakeStalled** (non-fatal,
per-conn). This bounds the pre-auth window and caps the number of half-open admitted
connections. Enforced by an integration test in P1.

### 3.3 Streams → AsyncRead/Write (the box that fits the seam)

`Connection::open_bi()`/`accept_bi()` return **named futures** (not `async fn`) that
`.await -> Result<(SendStream, RecvStream), ConnectionError>`. `SendStream` impls tokio
`AsyncWrite`, `RecvStream` impls tokio `AsyncRead`; both are boxable → they box into
`WorkerHalves.write`/`.read` unchanged. Connection types (`Connection`, `SendStream`,
`RecvStream`, `ConnectionError`) live under **`iroh::endpoint::*`**.

```rust
use iroh::endpoint::{Connection, SendStream, RecvStream};
let (send, recv): (SendStream, RecvStream) = conn.open_bi().await?;  // named future
let w: Box<dyn tokio::io::AsyncWrite + Unpin + Send> = Box::new(send);
let r: Box<dyn tokio::io::AsyncRead  + Unpin + Send> = Box::new(recv);
// → WorkerHalves { write: w, read: r, proc: <remote WorkerProc> }
```

> **Our NDJSON over raw `open_bi`/`accept_bi` is VIABLE (confirmed).** Reusing the existing
> `RpcFrame` NDJSON codec directly over these raw bidi streams needs **no `irpc` / `quic-rpc`
> dependency** — the streams are plain tokio `AsyncRead`/`AsyncWrite`, which is exactly what the
> existing codec and `serve_state` already consume.

> **`accept_bi()` gotcha (confirmed in docs):** QUIC defers stream creation to the first
> flush — the acceptor's `accept_bi()` does not fire until the *opener writes the first
> byte*. The opener MUST send immediately. Our NDJSON wire already opens with a HELLO line,
> so the opener sends HELLO first and this is satisfied for free.

**Stream classes per node↔hub connection:**

```
  one CONTROL stream  (bidi, long-lived, per node)  → higgs/node/* RPCs (NODE dispatch, §5.4a)
  N   DATA   streams  (bidi, per worker, lazy)       → M_CHAT / N_CHAT_CHUNK / N_LOG_LINE / N_PROGRESS
                                                        (serve_state relay, §5.4b)
```
First bidi = control; it terminates at the node's **`NodeRuntime` dispatch** (§5.4a) — NOT
in any worker. Each subsequent data stream is opened lazily on first chat for a given
`WorkerId` and is bridged into that worker's unchanged `serve_state`. Streams are cheap; one
connection multiplexes all of them with no extra dial. A chatty download never
head-of-line-blocks the control stream or another worker's chat. Liveness is iroh's job —
no app-level heartbeat frame (§3.4).

### 3.4 Liveness + reconnect

Liveness is **iroh-native** (invent-nothing): QUIC keepalive PINGs keep the conn warm and
`conn.closed()` fires on death. There is **no app-level heartbeat frame** — an `M_HEARTBEAT`
RPC would duplicate the transport's own keepalive. Periodic fleet deltas (loaded set, cpu/ram)
are not a liveness signal; they ride `M_INVENTORY{reason:"refresh"}` (§4.2), the single home
for inventory.

| Event | Signal | Handling |
|---|---|---|
| keepalive | QUIC PING (iroh, automatic) | keeps the conn alive through NAT; no app frame |
| stream EOF | `recv.read()` → `Ok(0)` | NDJSON read loop sees EOF, exits — no special handling |
| connection death | `conn.closed().await -> ConnectionError` | sentinel via `tokio::select!` alongside the supervisor loop; hub retires the node (HG027) |
| reconnect | re-`connect(hub_id, ALPN)` | id is stable; discovery re-resolves; iroh re-does hole-punch transparently |
| **wedged worker** (FFI hang) | **no transport signal** — conn warm, `conn.closed()` never fires | hub escalation policy, §3.4.1 (fix F) |

One `Endpoint` per process lifetime; re-dial per drop with backoff.

#### 3.4.1 Wedged-worker reap (fix F)

A remote worker's FFI (`engine.chat`) can hang with the node up and the connection warm —
so `conn.closed()` never fires and no transport-level signal surfaces it. Without an app-level
policy the worker's GPU stays pinned silently. Hub escalation:

```
chat stalls past CHAT_DEADLINE (no N_CHAT_CHUNK / no final)
  → hub sends M_KILL(worker_id) over the control stream
  → no ack within KILL_ACK_DEADLINE
      → hub redials the node (id stable, discovery re-resolves)
      → re-issue M_KILL
          → still no ack  → retire the node (HG027), drop its rings (§6), surface to UI
```
No silent GPU pin: every step is observable, and the terminal state (HG027) tells the UI the
node is gone. The node's own `Supervisor` restart FSM may also reap the wedged child locally on
its side; the hub policy covers the case where the node never responds.

### 3.5 Relay / discovery (the lean config)

| Need | Setting |
|---|---|
| WAN / NAT (default) | `presets::N0` — n0 public relays + DNS/pkarr discovery |
| same-LAN, zero infra | `presets::Minimal` + `.discovery_local_network()` (mDNS) + `RelayMode::Disabled` |
| self-hosted relay | `presets::Minimal` + `.relay_mode(RelayMode::Custom(map))` |

Default ships `presets::N0` so a phone on cellular reaches a desktop hub through relay,
upgrading to direct via hole-punch when possible.

---

## 4. The Wire — HELLO + M_* + Streams

The wire is `RpcFrame` (Request / Response / Notification), NDJSON, JSON-RPC 2.0,
slash-namespaced methods, `id` correlation, `request_id` echo for chunk routing —
**byte-identical** to the local worker wire (`rpc.rs`, `worker/mod.rs:19-37`). Remote adds
a new `higgs/node/*` namespace and the HELLO handshake; nothing in the existing vocabulary
changes.

### 4.1 HELLO — first frame, the update anchor

The **first RpcFrame on the control stream** is a `higgs/node/hello` request (node→hub)
answered by a `higgs/node/hello` result (hub→node). **Direction: node→hub — the NODE is the
caller** (it dialed; it sends HELLO first per §3.3). The hub *receives* HELLO and dispatches
it (§5.4c). HELLO carries version vectors so future changes negotiate instead of breaking —
the single anchor that makes M_UPDATE (§9) purely additive.

```jsonc
// node → hub : RpcRequest { method:"higgs/node/hello", id:1, params: HelloParams }
{ "role": "node",                         // "node"|"hub" — self-declared, cross-checked vs allowlist
  "node_id": "z32-endpoint-id-of-self",   // iroh EndpointId, z-base-32; MUST equal the QUIC peer id
  "pairing_token": "htk_8f3c…",           // OPTIONAL — only on first join; omitted once paired
  "protocol_versions": [1, 2],               // every wire-protocol major this peer can SPEAK
  "min_supported": 1,                      // lowest major it will still ACCEPT
  "software_version": "0.4.2",            // higgs build (semver) — informational + M_UPDATE gating
  "capabilities": { "chat": true, "download": true, "log_stream": true, "update": false } }

// hub → node : RpcResponse { id:1, result: HelloResult }
{ "role": "hub",
  "node_id": "z32-endpoint-id-of-hub",
  "agreed_version": 2,                     // the single major both sides pin for this session
  "software_version": "0.4.2",
  "capabilities": { "update_push": true, "log_aggregate": true },
  "assigned_label": "studio-mac" }         // hub's human label for this node (UI + LogSource)
```

**Negotiation:**
```
agreed = max( intersect(node.protocol_versions, hub.protocol_versions) )
require agreed >= node.min_supported  AND  agreed >= hub.min_supported
no such agreed  →  TYPED CLOSE  HG023 VersionMismatch (fatal)
```
`capabilities` is an **open map**: a peer NEVER hard-fails on an unknown key — it ignores it.
Only `protocol_versions`/`min_supported` gate the handshake. That is what lets a newer hub
talk to an older node and vice-versa. Version mismatch returns a typed `RpcError` carrying
`HG023`, *then* closes the QUIC stream — a typed close (vs a silent RST) is what tells the
node's UI "you must update," not "network broke."

### 4.2 Control-plane methods (`higgs/node/*` — additive)

New constants live in a new `remote` module, namespaced `higgs/node/*` to stay
unmistakably distinct from the local worker `higgs/*` methods.

> **Two layers, two dispatchers.** `higgs/node/*` control RPCs terminate at the **node's
> `NodeRuntime` dispatch** (a new dispatch in `src/node.rs`, §5.4a) — they NEVER reach
> `WorkerState::dispatch`. `WorkerState` (`worker/mod.rs:110-114`) owns only `engine` +
> `loaded`: a *single* llama.cpp child, no `ModelStore`, no HF-download, no multi-worker
> registry. It physically cannot serve scan/pull/load-across-workers/kill-one-worker. The
> ONLY thing reused on `serve_state`/`WorkerState` is the per-worker **data** stream
> (`M_CHAT`/`N_CHAT_CHUNK` relay, §5.4b). Tables below split node→hub from hub→node by
> **caller direction** (fix A).

**(a) node → hub — registration (the NODE is the caller; the HUB receives + dispatches):**

The node owns an **outbound requester** to the hub (its end of the per-node iroh transport,
§2.3). It *sends* these; the hub *dispatches inbound* (§5.4c). These are NOT hub methods
calling `Supervisor::request`.

| Const | Method | Sent by | Params | Hub does (on receive) |
|---|---|---|---|---|
| `M_HELLO` | `higgs/node/hello` | node → hub | `HelloParams` | gate (pairing/allowlist §7) → reply `HelloResult` |
| `M_INVENTORY` | `higgs/node/inventory` | hub → node (request) | `{}` → `NodeInventory` | AS SHIPPED the HUB pulls inventory on connect and after each lifecycle op (`refresh_inventory`); the design's node-push never landed |

```jsonc
// NodeInventory — push payload. reason:"boot" = first full report;
// reason:"refresh" = periodic delta (loaded set + cpu/ram), the SOLE periodic frame
// (no separate heartbeat — iroh keepalive carries liveness, §3.4). Composes existing
// types verbatim — no parallel struct:
{ "node_id":"z32…", "label":"studio-mac", "software_version":"0.4.2",
  "hardware": HardwareInfo,             // src/system.rs:57  — +hostname/os_name/os_version (§4.2.1), now Deserialize
  "runtime":  RuntimeInfo,              // src/system.rs:86  — now Deserialize (hub parses remote inventory)
  "models_on_disk": [ HiggsModel, … ],  // src/worker/models.rs — UNCHANGED (read-only scan)
  "cpu_usage_percent": 41.0, "ram_used_bytes": 9000000000,   // refresh deltas
  "workers": [ { "worker_id":1, "loaded": LoadedInfo } ] }   // worker_id:u32 (§5.4a); LoadedInfo src/api.rs:170 UNCHANGED
```
AS SHIPPED `M_INVENTORY` is a hub→node REQUEST (the hub pulls on connect and after
each lifecycle op via `refresh_inventory`); the push-style/event-driven design above
never landed, and the shipped `NodeInventory` carries no `reason` field. The hub keeps
`HashMap<NodeId, NodeView>` (`NodeView` = `NodeInventory` + hub-observed addr/path, §4.2.1) —
the single home for the fleet view that `/api/higgs/nodes` (UI panel) renders. `worker_id` is a
`WorkerId` (§5.4a, `u32`) owned by the node's `NodeRuntime` registry; **`HardwareInfo`/
`RuntimeInfo` come from `SystemInfo::gather(config, gpus)` (`system.rs:125`), NOT from
`Higgs::sysinfo` (which returns only `Vec<GpuDevice>`)** — see fix C in §4.2(b).

### 4.2.1 Node identity — name / OS / IP (for display)

The fleet UI shows, per node: **machine name, OS, hardware, and the address that reaches it.**
Split by **single home** — node-gathered facts vs hub-observed facts:

```
  NODE-GATHERED  (rides in NodeInventory.hardware; ALSO enriches local /api/higgs/system)
    HardwareInfo gains 3 fields, gathered in SystemInfo::gather (system.rs:125) via sysinfo:
      hostname    = System::host_name()    // "studio-mac"
      os_name     = System::name()         // "macOS" / "Ubuntu"
      os_version  = System::os_version()   // "15.5" / "22.04"
    (HardwareInfo already carries cpu_name, arch, cpu_cores, RAM, gpus[], vram — unchanged.)

  HUB-OBSERVED  (transport knowledge — NEVER in the node-gathered inventory)
    NodeView wraps the inventory with what ONLY the hub knows, from the live iroh connection.
    iroh 1.0 conns are MULTIPATH — read the address per PATH (no Connection-level accessor):
      conn.paths() → Path::remote_addr() -> &TransportAddr     // Ip(SocketAddr) | Relay(RelayUrl)
      observed_addr = the Ip(SocketAddr) of the active path    // "73.12.4.8:51820"
      path          = Path::is_ip() ? Direct : Relay           // ConnectionType REMOVED in 1.0
```

```rust
// hub fleet map entry: NodeView, not bare NodeInventory. observed_addr/path are HUB facts
// (one home, hub side); `inventory` is the NODE's self-report. Never merge the two.
pub struct NodeView {
    pub inventory:     NodeInventory,  // node-gathered (hardware incl. hostname/os, workers, models)
    pub observed_addr: String,         // Path::remote_addr() (TransportAddr::Ip) — NEVER node-self-reported
    pub path:          NodePath,       // Direct | Relay  (iroh connection type)
    pub online:        bool,
    pub last_seen_ms:  u64,
}
pub enum NodePath { Direct, Relay }
```
Why IP is hub-side and not a `NodeInventory` field: a node's own private `192.168.x` is often
meaningless to the hub user; the address that actually *reaches* it — and whether the path is
direct or relayed — is known only at the hub's `Connection`. **No `local-ip-address` crate** —
iroh already did the discovery during NAT traversal (§3.0).

> **Deserialize note.** `HardwareInfo`/`RuntimeInfo` derive `Serialize` only today
> (`system.rs:56,85`); the hub must DESERIALIZE them out of remote inventory, so both gain
> `#[derive(Deserialize)]` (`GpuDevice`/`DeviceKind` already have it). One-line derive change,
> P4. The 3 new `HardwareInfo` fields are `Option<String>` (`#[ts(optional)]`) — `host_name()`
> et al. return `Option` and a headless host may have none.

**(b) hub → node — lifecycle ops (the HUB is the caller; the node dispatches):** the hub
*sends* these over the control stream via its per-node iroh requester; the node's
`NodeRuntime` dispatch (`src/node.rs`, §5.4a) receives the frame and runs a node-local
operation, then replies. **None reach `WorkerState`** — they live one layer up in
`NodeRuntime`, which owns the `ModelStore` + the `HashMap<WorkerId, Arc<Supervisor>>`.

| Const | Method | Params | Result | NodeRuntime does |
|---|---|---|---|---|
| `M_LOAD` | `higgs/node/load` | `NodeLoadParams` (`id`, `ctx_len?`, `gpu_layers?`, `threads?`, rich `params?` — no per-load idle TTL; `deny_unknown_fields`) | `{ "worker_id", "loaded": LoadedInfo }` | **fit-check VRAM** (fix below) → spawn a NEW `Supervisor` → assign a NEW `WorkerId` → insert into registry → load model |
| `M_UNLOAD` | `higgs/node/unload` | `{ "worker_id" }` | `StatusOk` | look up worker → `Supervisor::stop()` → remove from registry → free `WorkerId` → drop its log ring (§6) |
| `M_KILL` | `higgs/node/kill` | `{ "worker_id" }` | `StatusOk` | look up worker → `Supervisor::stop()` (force-reap that ONE child) → remove → free `WorkerId` → drop ring |
| `M_SCAN` | `higgs/node/scan` | `{}` | `{ "models": [HiggsModel…] }` | the node's on-disk catalog (read-only scan); inventory is the SEPARATE `M_INVENTORY` op |
| `M_SYSINFO` | `higgs/node/sysinfo` | `{}` | `{ "hardware":HardwareInfo, "runtime":RuntimeInfo }` | `SystemInfo::gather(config, Higgs::sysinfo)` — see fix C below |
| `M_PULL` | `higgs/node/pull` | `NodePullParams { request_id, repo, file, revision? }` | `N_PROGRESS` stream, then `{ "path" }` | AS SHIPPED: single-file pull with streamed progress (`remote.rs`) |

> **`M_LOAD` = orchestrator spawns a NEW worker (multi-worker, net-new).** The node may
> already host other workers; `M_LOAD` does NOT replace them. `NodeRuntime` spawns a fresh
> `Supervisor` (one new child), assigns the next `WorkerId`, inserts it into the registry, and
> returns `{ worker_id, loaded }`. This is **net-new orchestration** — it does NOT go through
> `Higgs::load` (which is only-keep-last, `api.rs:363-364`). A node spawning its very first
> worker may reuse the existing single-`Supervisor` path; the 2nd+ worker is the net-new part.

> **VRAM fit-check before spawning an additional worker (fix, net-new wiring).** Before
> spawning a new `Supervisor` on `M_LOAD`, `NodeRuntime` runs the **existing FitAssessment
> path** against the node's current free VRAM (`GpuDevice::vram_free_bytes`, `system.rs:50`)
> minus what already-resident workers consume. If it won't fit → reply `HG017
> InsufficientMemory` (existing code), do not spawn. Reuse the assessment logic; the net-new
> part is summing across N concurrent workers instead of assuming one.

> **`M_KILL` targets the NODE registry, not the worker wire (net-new).** It maps to
> `NodeRuntime` looking up `worker_id` → `Supervisor::stop()` on that one (`stop(&self)` takes
> no id today, `supervisor.rs:625` — selection is done by the registry lookup, one Supervisor
> per worker). It is NOT the worker's own `M_SHUTDOWN` (intercepted by `serve_state` at
> `worker/mod.rs:62`, which ends only the *current* stream's worker and cannot select by id).

> **`M_SYSINFO` / inventory hardware (fix C).** `Higgs::sysinfo` (`api.rs:996`) returns
> **`Vec<GpuDevice>` — NOT hardware+runtime.** The hardware/runtime struct is
> `SystemInfo::gather(config, gpus)` (`system.rs:125`), which **takes** the `GpuDevice` list
> and produces `HardwareInfo` (`system.rs:57`) + `RuntimeInfo` (`system.rs:86`). The device
> list itself originates from `engine.devices()` inside the worker (`worker/mod.rs:282`),
> surfaced by `Higgs::sysinfo`. So the node builds `M_SYSINFO`/HELLO/inventory hardware via
> `SystemInfo::gather(config, Higgs::sysinfo)` — never `Higgs::sysinfo` alone.

> **`M_PULL` is NET-NEW (fix D — no download exists).** The worker is **read-only scan**:
> "No model catalog — scanning is host-side and the GGUF path arrives in the M_LOAD params"
> (`worker/mod.rs:108`); "Scan moved host-side… the worker holds no catalog of its own"
> (`worker/mod.rs:228`). There is **no HF download anywhere in higgs today.** `M_PULL` requires
> a **NEW downloader + progress subsystem** (via `hf-hub`, §3.0/§4.4). It downloads into
> higgs's **OWN** `~/.higgs/models/` dir (or HF cache) — **NEVER** into the read-only scanned
> LM-Studio / HF-cache / Ollama dirs. `M_PULL` is excluded from any "reuse verbatim" claim.

> **Conflation flag #1.** `M_LOAD`/`M_UNLOAD`/`M_SYSINFO` are *names reused* from the worker
> vocabulary, but on `higgs/node/*` they target a **node**, not a worker — `NodeRuntime` then
> re-issues the *worker* `higgs/load` over local stdio to the chosen `Supervisor`. Same verb,
> two layers. The `higgs/node/` prefix is **mandatory** so a reader never mistakes a hub→node
> `M_LOAD` for a node→worker `M_LOAD`. Never drop the prefix.

> **Conflation flag #2.** Control plane vs data plane are physically separate streams
> (1 control/node, N data/worker) AND physically separate dispatchers (`NodeRuntime` vs
> `serve_state`). Control = `higgs/node/*` RPCs → `src/node.rs` dispatch. Data = `M_CHAT` /
> `N_CHAT_CHUNK` / `N_LOG_LINE` / `N_PROGRESS` → `serve_state`. Never multiplex chat onto the
> control stream; never route a control RPC into `WorkerState`.

### 4.3 /v1 chat — hub→node→worker (the two-hop relay)

```
external client          HUB                          NODE                    WORKER
  POST /v1/chat   ─▶  resolve model→(node,worker)  open data stream(worker)   local stdio
  Bearer sk-…         gate: API key scope             │
                      M_CHAT over iroh ─────────────▶ │ M_CHAT over stdio ──▶ generate
                      (per-node transport, §2.3)      │ (EXISTING supervisor.chat)
                  ◀── N_CHAT_CHUNK (iroh) ──────────  │ ◀── N_CHAT_CHUNK (stdio, UNCHANGED)
   SSE deltas    ◀─   relay chunks to client          │
                  ◀── RpcResponse{result} ──────────  │ ◀── final RpcResponse (UNCHANGED)
```
Method = **existing `M_CHAT = "higgs/chat"`**, notification = **existing
`N_CHAT_CHUNK = "higgs/chat/chunk"`** — byte-identical to the worker wire. The node is a
*transparent relay*: forwards the hub's `M_CHAT` to the worker's `Supervisor`, forwards
`N_CHAT_CHUNK`/final-response back. The hub's relay is the *same shape* as `supervisor.rs`
(`pending` keyed by `id`, `chat_sinks` keyed by `request_id`) but is a **separate instance on
the per-node iroh transport** (§2.3) — not the hub's local-worker Supervisor. The hub→node
`M_CHAT` params add one routing field:
```jsonc
{ "request_id":7, "worker_id":1,         // worker_id = NEW (u32): selects the node-local worker
  "model":"org/model-a", "messages_json":"[…]", "max_tokens":512,
  "temperature":0.7, "tools":null }
```
Final result = existing M_CHAT result (`content/finish_reason/tool_calls/prompt_tokens/
completion_tokens`, `worker/mod.rs:373-379`). No new shape.

> **request_id collision (fix #4).** Two independent `alloc_request_id`
> (`supervisor.rs:346`, an `AtomicU64` per Supervisor): the **hub's** per-node iroh transport
> allocates the id on the hub→node `M_CHAT`, and the **node's** stdio `Supervisor` for that
> worker allocates `7` again for an unrelated local chat. They share no counter, so ids collide.
> **Resolution: the node re-allocates a fresh node-local stdio `request_id` for the downstream
> worker call and keeps a per-data-stream translation table `hub_id ↔ node_id`.** Inbound
> `N_CHAT_CHUNK` from the worker (node-id) is rewritten to the hub-id before relay. Per-stream
> table ⇒ two streams' ids never alias.

> **Per-worker data-plane streaming (fix #3).** Each per-worker data stream gets its OWN
> `spawn_blocking` task with its own `SyncIoBridge` (§5.3) — one blocking thread per active
> worker stream — because `serve_state` is sync and `handle_chat` streams `N_CHAT_CHUNK`
> through the sync writer (`worker/mod.rs:356-365`). Tradeoff (§11): the sync writer applies
> **backpressure** — a slow hub `SendStream` (slow relay, congested link) blocks the write and
> **stalls token generation** on that worker until it drains. One slow consumer throttles its
> own generation, not others'.

### 4.4 Log + progress data notifications

| Const | Method | Direction | Routed by | Payload |
|---|---|---|---|---|
| `N_LOG_LINE` | `higgs/log/line` | node → hub | `(node, worker)` → LogSource ring | `{ "node_id","worker_id","text" }` |
| `N_PROGRESS` | `higgs/pull/progress` | node → hub | `repo` (download-sink map, mirrors `chat_sinks`) | `{ "repo","file","downloaded_bytes","total_bytes","done" }` |

`N_LOG_LINE` → `bus.push(LogSource::RemoteWorker{node,worker}, text)` (§6). `N_PROGRESS`
terminal frame `{…,"done":true}`; mid-transfer failure → `RpcError{ data.code:"HG025" }`.
Hub re-emits both onto the existing Developer-Logs and download-progress SSE channels —
no new bus mechanism. `N_PROGRESS` is part of the **net-new** `M_PULL` downloader (fix D).

---

## 5. The higgs Seam — file:line

All anchors are in the crate at `feat/iroh-remote`. Cargo deps `[dependencies]` have
**no iroh/quinn** yet — add them (§3).

### 5.1 The `HalvesFactory` seam — add a second factory

```rust
// supervisor.rs:148  (UNCHANGED signature)
type HalvesFactory =
    Box<dyn Fn(Arc<LogBus>, &str) -> Result<WorkerHalves, HiggsError> + Send + Sync>;
```
- arg `Arc<LogBus>`: factory wires its own log drain (`production_factory` spawns the
  stderr drain at `supervisor.rs:1347-1354`, push at `:1352`).
- arg `&str`: model id — cosmetic argv0 label only.
- `Fn` (called once per (re)spawn): `do_spawn` (`:769`), `spawn_replacement` (`:1059`),
  transient `probe_paths` (`:505`), transient `sysinfo` (`:579`).
- injected as one field `Inner.factory`, set in `Supervisor::spawn` (`:230-246`),
  test override `with_factory` (`:253-269`).

```rust
// WorkerHalves  (supervisor.rs:128-139)
pub(crate) struct WorkerHalves {
    pub(crate) write: WriteHalf,   // Box<dyn AsyncWrite + Unpin + Send>  (:120)
    pub(crate) read:  ReadHalf,    // Box<dyn AsyncRead  + Unpin + Send>  (:122)
    pub(crate) proc:  Option<tokio::process::Child>,   // liveness — see §5.2 (becomes WorkerProc)
}
```
**`remote_factory(bus, model)`** (NEW): against an already-paired node, open one iroh
bidi *data* stream, return `WorkerHalves { write: Box::new(send), read: Box::new(recv),
proc: Some(<remote WorkerProc>) }`. Drain the node's `N_LOG_LINE` stream into `bus` as
`LogSource::RemoteWorker{node,worker}` (the wire equivalent of the local stderr drain). No
change to `Inner`, `do_spawn`, `writer_task`, `reader_task`, `probe_one`, `sysinfo_one`, or
the codec. A remote-backed supervisor is `Supervisor::spawn` with the factory swapped — add a
`spawn_remote` ctor mirroring `spawn` (swap point `supervisor.rs:230-246`). **One
`remote_factory`-backed `Supervisor` per remote worker** — `NodeRuntime` (hub side) holds them
keyed by `WorkerId`, just as the node side does.

### 5.2 Liveness trait — replace the one `proc: Option<Child>` leak

`proc` is the **only** local-process-specific leak (`WorkerHalves.proc` `:139`;
`Inner.proc: tokio::sync::Mutex<Option<Child>>` `:201`). Every site that touches it:

| Op | Site | What it does to the `Child` |
|---|---|---|
| stash @ spawn | `do_spawn` `:782` | `*guard = halves.proc` |
| stop / reap | `stop()` `:640` | `wait()` w/ `WORKER_EXIT_TIMEOUT`, else `start_kill()`+`wait()` |
| transient reap | `probe_paths` `:555` | wait-then-kill |
| transient reap | `sysinfo` `:603` | wait-then-kill |
| reap old (give-up) | `reap_old_child` `:1070` | `old.wait()` |
| reap abandoned | `reap_child` `:1081` | `start_kill()`+`wait()` |
| install/reap (respawn) | `install_child` `:1092` | `old.wait()`, stash new |

`stop()` also drops `write_tx` to EOF the writer and clears `running` — both
transport-generic. The leak is purely the `Child` `wait/start_kill` calls. Abstract it:

```rust
trait WorkerProc: Send {
    async fn wait(&mut self);        // resolves when the worker is gone
    async fn force_kill(&mut self);  // local: start_kill()+wait();  remote: close QUIC conn
}
```
- **Local:** wraps `tokio::process::Child`. `wait()`→`child.wait()`; `force_kill()`→`start_kill()+wait()`.
- **Remote:** wraps the iroh `Connection`/`SendStream`. `wait()` resolves on connection-closed;
  `force_kill()`→`conn.close(..)` / `finish()`. No OS process to reap — close the streams.

Change `proc: Option<tokio::process::Child>` → `proc: Option<Box<dyn WorkerProc>>` on
`WorkerHalves.proc` (`:139`) and `Inner.proc` (`:201`). The 7 sites collapse from
`match timeout(wait){Ok=>{} Err=>{start_kill;wait}}` to `proc.force_kill()` / `proc.wait()`.
Test factory keeps `None`. This is the single liveness-abstraction PR.

### 5.3 Data-stream bridge — iroh into the unchanged `serve_state`

This is the **per-worker DATA path ONLY** (`M_CHAT`/`N_CHAT_CHUNK` relay) — control RPCs do
NOT come here (§5.4a). `serve_state` is **synchronous** std::io (`worker/mod.rs:54`):
```rust
fn serve_state(mut state: WorkerState, reader: impl BufRead, mut writer: impl Write)
```
Drives `reader.lines()` → `decode` → `state.dispatch` → `respond`, streaming `N_CHAT_CHUNK`
mid-request through the same sync `writer` (`handle_chat` sink loop `:356-365`).

Because `serve_state` is sync `BufRead`/`Write` and iroh streams are async, **bridge** — one
bridge + one `spawn_blocking` **per data stream** (per worker, fix #3); the FFI `engine.chat`
is blocking anyway so it must run on a blocking thread regardless:
```
   per worker stream:  iroh RecvStream/SendStream
        ──[tokio_util::io::SyncIoBridge inside its OWN spawn_blocking]──▶  serve_state
```
> **Backpressure note (→ §11).** The sync writer blocks when the QUIC `SendStream` is slow to
> drain; that stalls `N_CHAT_CHUNK` and therefore token generation on that worker. Per-stream
> isolation means a slow consumer throttles only its own stream.

**Reject** an async `serve_state_async` sibling: it would duplicate the loop. On the data path
`WorkerState`/`dispatch`/`engine` are untouched — but note the node, not `serve_state`, holds
the relay's request_id translation table (§4.3 fix #4), since `serve_state` runs against the
node's REAL stdio worker with node-local ids.

### 5.4 Node module — `NodeRuntime` + two dispatchers, never one

New module **`src/node.rs`** (sibling to `supervisor.rs`) owns both sides:
- **hub side (NEW):** `remote_factory` (§5.1); the paired-node registry (`NodeId →
  EndpointAddr/ticket`); the **per-node iroh transport** with its OWN pending/correlation,
  keyed by `NodeId`, routing per `(NodeId, WorkerId)` (§2.3, fix B — **distinct from the
  hub's local-worker Supervisor**); the `HashMap<NodeId, NodeView>` fleet view (§4.2.1); the
  hub→node outbound calls; inbound HELLO/inventory dispatch (§5.4c).
- **node side (NEW):** `Endpoint::bind`, accept loop → post-HELLO gate + HELLO-stalled timer
  (§3.2.1), the node's outbound requester to the hub (sends hello/inventory, §5.4c), and the
  **two dispatchers** below. **The node owns `NodeRuntime`: `HashMap<WorkerId,
  Arc<Supervisor>>`** — the net-new multi-worker registry.

The iroh `Endpoint` does **not** go on `Supervisor`'s `Inner` — `supervisor.rs` stays
transport-pure (`:28-30`). The Endpoint lives in `node.rs`, captured by `remote_factory`
exactly as `production_factory` captures `current_exe()`. This honours the crate-boundary
rule ("Supervision lives INSIDE the crate; callers never see process plumbing").

The local entrypoint selector is the inline argv check, *before any stdout write*:
```rust
// src/bin/higgs.rs:33-34   (binary is `higgs`, see §8 noun)
if std::env::args().skip(1).any(|a| a == "--higgs-worker") { higgs::worker::worker_main(); return; }
```
**New node arm**, parallel to it: a `--node` arg → run the iroh listener loop.

#### 5.4a Node-side CONTROL dispatch (NodeRuntime, NOT `WorkerState`)

`higgs/node/*` control RPCs arrive on the node's **control** stream and terminate in a **new
`NodeRuntime` dispatch in `src/node.rs`** that operates on the `HashMap<WorkerId,
Arc<Supervisor>>` registry + the node's `ModelStore`. They NEVER enter `WorkerState::dispatch`
— `WorkerState` (`worker/mod.rs:110-114`) has only `engine` + `loaded` (one child). Constants
live in a new `remote` module:

| `higgs/node/*` const | NodeRuntime dispatch arm | Backed by |
|---|---|---|
| `M_HELLO` | (node EMITS — outbound, §5.4c) | version vector + `SystemInfo::gather(config, Higgs::sysinfo)` (fix C) |
| `M_INVENTORY` | (node EMITS — outbound, §5.4c) | `Higgs::scan` (`api.rs:604`) + `SystemInfo::gather` + per-worker `LoadedInfo` |
| `M_SCAN` | re-scan node disk → `NodeInventory` | `Higgs::scan` (`api.rs:604`, read-only `ModelStore::scan`) |
| `M_LOAD` | **fit-check → spawn NEW Supervisor → assign WorkerId → insert** | NET-NEW orchestration (NOT `Higgs::load` only-keep-last) |
| `M_UNLOAD` | look up `worker_id` → `Supervisor::stop()` → remove → free WorkerId → drop ring | reuse `Supervisor::stop` (`:625`) + registry removal |
| `M_KILL` | look up `worker_id` → `Supervisor::stop()` (force-reap ONE) → remove → drop ring | reuse `Supervisor::stop` (NOT `M_SHUTDOWN`) |
| `M_SYSINFO` | `SystemInfo::gather(config, Higgs::sysinfo)` (fix C) | `Higgs::sysinfo`=`Vec<GpuDevice>` (`api.rs:996`) folded by `gather` (`system.rs:125`) |
| `M_PULL` | **NEW HF downloader** → `~/.higgs/models/` (fix D) | NET-NEW; progress on data plane (§4.4) |

> **`WorkerId` — NEW newtype, the per-node worker key. LOCKED: `u32` (fix, multi-worker).**
> Single home = the node's `NodeRuntime` registry. Decided via Crate-First (§3.0): a `u32`
> newtype needs **no string crate** (no `smol_str`/`compact_str`) AND is `Copy`, so it keeps
> `LogSource` `Copy` (§6) with zero ripple. The wire carries it **as a number**:
> ```rust
> #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
> pub struct WorkerId(u32);        // wire: "worker_id": 1   (UI may RENDER it as "w-1")
> ```
> Lifecycle: **assigned on `M_LOAD`** (monotonic per node, new worker spawned), **freed on
> `M_UNLOAD`/`M_KILL`**. It keys both the data-stream routing (§4.3) and
> `LogSource::RemoteWorker` (§6).
>
> **`Copy` — RESOLVED, no tension.** `WorkerId(u32)` is `Copy` and `NodeId(EndpointId)` is
> `Copy` (32-byte key), so `LogSource::RemoteWorker{node,worker}` is `Copy` and `LogSource`
> keeps its `#[derive(Copy)]` unchanged — no demotion to `Clone`, no ripple to `LogLine.source`
> or `broadcast::Sender<LogLine>` (§6). The string `"w-1"` is a **display rendering only**; the
> type and the wire are `u32`. The forbidden state — `u32` in memory but `"w-1"` on the wire —
> does **not** occur here: the wire is numeric.

#### 5.4b Per-worker DATA dispatch (reuse `serve_state`)

The ONLY reuse of `serve_state`/`WorkerState`: each per-worker data stream is bridged (§5.3)
into the `Arc<Supervisor>` for that `worker_id` (looked up in the `NodeRuntime` registry),
which drives the unchanged `serve_state` of the real llama.cpp child. `M_CHAT` in,
`N_CHAT_CHUNK`/final out, request_id translated per stream (§4.3 fix #4). No new dispatch arm
— the existing chat path with the transport swapped.

#### 5.4c Direction split — who sends, who dispatches (fix A)

```
  NODE side                                   HUB side
  ─────────                                   ────────
  (a) dispatch hub→node ops                   (c) SEND hub→node ops
      M_LOAD/M_UNLOAD/M_KILL/M_SCAN/              via per-node iroh requester (§2.3):
      M_SYSINFO/M_PULL  →  NodeRuntime           load()/unload()/kill()/scan()/sysinfo()/pull()
      (§5.4a)                                     correlate replies on the per-node pending map
  (b) SEND node→hub frames                    (c) DISPATCH inbound node→hub frames
      hello()  → M_HELLO  (on dial)               receive M_HELLO  → gate (§7) → reply HelloResult
      inventory() → M_INVENTORY (push)            receive M_INVENTORY → store in fleet map → StatusOk
      via the node's outbound requester           (these are NOT hub methods calling
                                                   Supervisor::request — the node is the caller)
```

The **hub does NOT call `Supervisor::request` for `M_HELLO`/`M_INVENTORY`** — those are
inbound; the hub dispatches them. The hub's outbound (load/unload/kill/scan/sysinfo/pull) goes
through the **per-node iroh transport's** own request/correlate (a separate instance, §2.3),
NOT the hub's local-worker `Supervisor`. The *shape* mirrors `Supervisor::request`
(`supervisor.rs:354-369`) + `correlate` (`:1120`), but the instance is per-node and iroh-bound.

---

## 6. LogSource 2→N

Current enum (`log_bus.rs:57-63`):
```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]      // :57 — Copy today
pub enum LogSource { Serve, Worker }              // Worker carries no id today
```
Two fixed rings back it (`LogBus`, `log_bus.rs`): `serve` + `worker` as
`Mutex<VecDeque<(u64,String)>>`; `seq` is the global interleave counter; `LogLine.source`
(`:81`) and `broadcast::Sender<LogLine>` carry it **by value**.

**Add a `RemoteWorker{node,worker}` variant (fix #5/#6 + fix G):**
```rust
// higgs-domain newtype — WRAPS iroh's EndpointId (= PublicKey, 32-byte key, Copy).
// iroh has NO `NodeId` type; this is ours, never an iroh type name.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NodeId(iroh::EndpointId);

// WorkerId is the SAME type as §5.4a — LOCKED u32 (Copy). Both fields Copy ⇒ LogSource Copy.
pub enum LogSource {                                  // STAYS Copy (NodeId Copy + WorkerId u32 Copy)
    Serve,
    Worker,                                           // local worker — UNCHANGED (single-node back-compat)
    RemoteWorker { node: NodeId, worker: WorkerId },  // NEW
}
```

> **`Copy` decision (fix G, ties to §5.4a) — RESOLVED: `LogSource` stays `Copy`.** Because
> `WorkerId` is locked to `u32` (§3.0/§5.4a — needs no string crate and is `Copy`) and `NodeId`
> wraps iroh's `EndpointId` (32-byte key, `Copy`), `RemoteWorker{node,worker}` is `Copy`, so
> `LogSource` keeps its `#[derive(Copy)]` unchanged. **No demotion to `Clone`, no ripple** to
> `LogLine.source` (`:81`) or `broadcast::Sender<LogLine>` — they keep copying a tiny enum. The
> earlier compact-string option (which would have forced `Copy`→`Clone`) is dropped.

Every match-on-source site and its change:

| Site | Today | After |
|---|---|---|
| `LogSource::parse` `:67-73` | hardcoded `"serve"\|"worker"` arms → `Option<LogSource>` | add a structured arm: `s.split(':')` `["node",id,worker]` → `RemoteWorker{node,worker}`; unparseable → `None`. Still `Option<LogSource>`. |
| `push` ring select `:178-181` | `match { Serve=>serve, Worker=>worker }` | + `RemoteWorker` → look up/create `workers[(node,worker)]` ring |
| `snapshot` `:197-212` | merge two rings by `seq` (`:206` sort) | merge `serve` + `worker` + all remote rings by `seq` (sort unchanged) |
| `LogBus` fields | two fixed `Mutex<VecDeque>` | `serve` + `worker` + `remote: Mutex<HashMap<(NodeId,WorkerId), VecDeque<(u64,String)>>>` |
| factory push | `production_factory` `:1352` `Worker` | `remote_factory` log drain pushes `RemoteWorker{node,worker}` |
| HTTP filter | `serve/control.rs:44` `filter()` → `LogSource::parse` | `?source=node:<id>:<worker>` parses via the new arm; signature `Option<LogSource>` unchanged; SSE drop-by-source unchanged |

### 6.1 Ring eviction — bound the map (fix G)

The hub is long-lived; remote rings would grow unboundedly without a reclaim path. **Reclaim
rules:**

```
  M_UNLOAD(worker) / M_KILL(worker)  →  drop remote[(node, worker)]            (one ring)
  node retire (HG027) / disconnect   →  drop ALL remote[(node, *)]            (all node rings)
```

`NodeRuntime` (hub side) drives both: the worker-level drop on unload/kill, the node-level
sweep on retire/disconnect. So the `remote` HashMap is bounded by *currently live* workers
across *currently connected* nodes — it cannot grow without bound on the long-lived hub. State
the reclaim explicitly so a dropped worker's ring is freed, not leaked.

The existing per-source-ring design already prevents one source flooding another; the global
`seq` already supports N-way interleave. ts-rs: `LogSource` is not currently exported; if a
nodes UI needs it on the wire, it gets a `higgs_ts!` export then (P6).

---

## 7. Auth — Two Surfaces

Two **different trust questions**, never conflated — no shared token, store, or code path.

| | **Surface A — hub ↔ node** | **Surface B — external client → hub /v1** |
|---|---|---|
| Question | "is this *machine* allowed to join my fleet?" | "is this *caller* allowed to use my models?" |
| Primitive | iroh `EndpointId` allowlist | OpenAI Bearer API keys |
| Join | one-time expiring **pairing token** (`htk_…`) | key created in UI/CLI, shown once |
| Standing | allowlist membership (QUIC/TLS proves peer key) | `Authorization: Bearer hgk_…` (drafted as `sk-higgs-…`) |
| At rest | `pairings.json` (`Vec<EndpointId>` + label) | `api_keys.json` (SHA-256 hash, prefix, scopes) — never plaintext |
| Checked | transport-level, **post-HELLO** via `remote_id()` (§3.2) | app-level middleware on `/v1` (the drafted `/api/higgs/*` surface was later deleted) |
| CLI noun | `link …` | `keys …` |
| Reject diag | `HG024 NotAllowlisted` / `HG022 PairingTokenInvalid` / `HG028 HandshakeStalled` | `401 { "error": { "code":"invalid_api_key" } }` |

**Surface A flow:** hub mints `htk_<random>` (single-use, effectively non-expiring — the token
only gates first enrollment; persistence comes from the keypair + allowlist, not a clock, so a
node killed before it could save its hub can still pair on the next run), encodes hub
`EndpointId` + token + relay URL into a QR/pairing string. The dial coordinates are an
`EndpointTicket` built from `EndpointAddr::new(hub_id).with_relay_url(relay)` (and/or
`.with_addrs(..)` / `.with_ip_addr(..)` for known direct addrs — **`with_direct_addresses` does
NOT exist**); the ticket round-trips to a string via `Display`/`FromStr` (or bytes via the
`Ticket` trait's `encode_bytes`/`decode_bytes`), with `htk_…` carried alongside. Node dials
(`endpoint.connect(ticket.endpoint_addr().clone(), ALPN)`), HELLO with `pairing_token`. Hub
validates (unexpired, unused) → adds the node's `EndpointId` to the allowlist → token burned.
Reconnects present **no token** — pure allowlist membership.

**Surface B record:**
```jsonc
{ "id":"key_01H…", "name":"laptop-cli",
  "hash":"sha256:9f86d0…", "prefix":"sk-higgs-9f86",   // prefix for UI listing
  "scopes":["chat","models"],                          // chat | models | admin
  "created_ms":1750000000000, "last_used_ms":1750000500000, "disabled":false }
```
Middleware hashes the presented bearer (`sha2`), constant-time compares (`subtle`, §3.0), checks scope + `disabled`,
reuses the existing `/v1` error envelope. `admin` scope gates `/api/higgs/*` mutations + node
management.

> **Store ownership — inside the crate (fix #10).** Both `pairings.json` and `api_keys.json`
> (in `~/.higgs`, §8) are owned **inside the higgs crate** — a new `src/auth.rs` (sibling to
> `node.rs`), serialized with the crate's OWN `serde` derives. **No `common` / `engine` /
> jigglebot import** touches them, preserving the one-way dependency (DESIGN.md Crate Boundary).
> `node.rs` reads the allowlist from `auth.rs` at the post-HELLO gate (§3.2).

> **Conflation flag #3.** `link` (surface A, machines/nodes) and `keys` (surface B, callers)
> are deliberately different CLI nouns. Never fold API-key management under `link`.

### 7.1 New diagnostics — HG022–HG028

Seven new codes, appended in increasing order after the current max `HG021`
(`diagnostic.rs:185`, `SysinfoWorkerFailed`) per the four-pillar append-only rule. They follow
the existing `HiggsError` style: `[HGxxx]` baked into the snafu `display`,
`#[diagnostic(code(HGxxx))]`, `severity(Error)` only on fatal variants. Origin = where the
variant is constructed.

**Gate-outcome reconciliation (fix H).** The post-HELLO gate has **four** distinct outcomes,
each its own code — no "two auth rejections" miscount:

```
  not in allowlist & no valid token  →  HG024 NotAllowlisted    (non-fatal)
  pairing token bad/expired/used     →  HG022 PairingTokenInvalid(non-fatal)
  no agreed protocol version         →  HG023 VersionMismatch    (FATAL, typed close)
  QUIC/TLS done but no HELLO in time  →  HG028 HandshakeStalled   (non-fatal, §3.2.1)
```

| Code | Variant | snafu display message | Severity | Origin (module) |
|---|---|---|---|---|
| HG022 | `PairingTokenInvalid` | `[HG022] pairing token invalid (expired, used, or unknown): {detail}` | non-fatal | `node.rs` (post-HELLO gate, §3.2) |
| HG023 | `VersionMismatch` | `[HG023] no agreed protocol version: peer speaks {peer:?}, we accept {ours:?}` | **fatal** `severity(Error)` (typed close) | `node.rs` (HELLO negotiation, §4.1) |
| HG024 | `NotAllowlisted` | `[HG024] peer {endpoint_id} is not in the allowlist and presented no valid pairing token` | non-fatal | `node.rs` (post-HELLO gate, §3.2) |
| HG025 | `DownloadFailed` | `[HG025] model download failed: {repo}: {detail}` | non-fatal | `node.rs` (`M_PULL` HF download, §4.4) |
| HG026 | `NotImplemented` | `[HG026] not implemented: {method}` | non-fatal | `node.rs` (`M_UPDATE` stub today, §9) |
| HG027 | `NodeUnreachable` | `[HG027] node {endpoint_id} unreachable; retired from fleet: {detail}` | non-fatal (retire, best-effort) | `node.rs` (conn-closed / dial failure / wedged-worker escalation, §3.4, §3.4.1) |
| HG028 | `HandshakeStalled` | `[HG028] peer {endpoint_id} completed QUIC but sent no HELLO within {window}s; dropped` | non-fatal | `node.rs` (post-HELLO timeout, §3.2.1) |

> **HG026 use-site note.** HG026 = `NotImplemented`, used by the `M_UPDATE` stub (§9). The two
> auth-gate *rejections* are HG022 (bad token) and HG024 (not allowlisted); HG023 (version) is a
> fatal close, HG028 (handshake-stalled) is the new pre-auth-DoS guard. No spare auth code is
> needed — four codes for four outcomes, no miscount.

---

## 8. Settings/Home + the lean CLI

**Home:** `~/.higgs/` (XDG-overridable). One canonical home per piece of identity/state.

```
~/.higgs/
  endpoint.key       32 raw ed25519 secret bytes, chmod 0600  → stable EndpointId (§3.1)
  pairings.json      hub: paired node EndpointIds + labels (surface A allowlist)
  api_keys.json      hub: SHA-256-hashed Bearer keys + scopes (surface B)
  config.toml        node: hub EndpointId, relay prefs, model dirs (mirrors HiggsConfig)
  models/            node-local downloaded GGUFs — the ONLY M_PULL write target (fix D)
```
`models/` is higgs's OWN dir — `M_PULL` writes here, **never** into the read-only scanned
LM-Studio / HF-cache / Ollama dirs. The hub embeds higgs inside jigglebot (existing
`Higgs::new(config)` launcher). A **node** runs the lean standalone CLI — same one binary, node
role.

> **Binary noun (fix I — rename DONE).** The CLI binary **is `higgs`**: package name is
> `higgs` (`Cargo.toml [package] name = "higgs"`) and the bin source is **`src/bin/higgs.rs`**,
> so the produced executable is **`higgs`** (`CARGO_BIN_EXE_higgs`). The CLI examples below use
> `higgs …` verbatim — no substitution, no pending rename.

**CLI surface (mirrors `lms link` / `lms`):**
```
# hub side (surface A — fleet of machines). DRAFT: only `link pair` and
# `link status` shipped; enable/disable became the hub_enable/hub_disable
# control ops, and `link ls` became the `nodes` op / Fleet view.
higgs link status            # listener on/off, hub EndpointId, paired-node count, live sessions
higgs link pair              # mint a pairing token + print QR/string

# node side
higgs --node                 # boot as a node (binds Endpoint, idles for hub dial config)
higgs node connect <pairstr> # dial hub, HELLO with pairing_token (first join, surface A)

# hub side (surface B — callers, NOT machines — separate noun)
higgs keys add X chat        # mint Bearer, printed ONCE (shipped as add/list/remove,
higgs keys list              #  not the create/ls/revoke drafted here)
higgs keys remove X
```

---

## 9. Updates — binary vs home

**Two independent update axes — never co-mingled:**

```
  BINARY  (the higgs executable)               HOME  (~/.higgs, persisted state)
  ──────────────────────────────              ──────────────────────────────────
  swapped by an updater (M_UPDATE, later)     migrated in place by the running binary
  signed, pinned-key verified                 endpoint.key NEVER rotated on update
  re-exec to apply                            pairings/api_keys survive every swap
  bumps software_version (HELLO)              schema versioned independently
```
The persisted `SecretKey` (`endpoint.key`) means a node keeps its `EndpointId` across binary
swaps — it stays in the hub's allowlist with no re-pairing. The home is the durable identity;
the binary is replaceable.

**`M_UPDATE` reserved now (additive, gated off):**
```rust
pub const M_UPDATE: &str = "higgs/node/update";   // hub → node
// HELLO capabilities already carry { "update": <node self-updates>, "update_push": <hub pushes> }
```
```jsonc
// M_UPDATE params (reserved — handler returns HG026 NotImplemented today)
{ "target_version":"0.5.0",
  "artifact_url":"https://…/higgs-0.5.0-aarch64-apple-darwin.tar.gz",
  "signature":"minisign:RWS…",          // detached sig over the artifact
  "pinned_key_id":"higgs-release-2026" } // which pinned pubkey must verify it
```
Ship a `const HIGGS_UPDATE_PUBKEYS: &[(&str,&str)]` (key_id → pubkey) compiled into the
binary today (empty-capable). A node that later self-updates verifies `signature` against
`pinned_key_id` before swapping its binary — no TOFU on the update itself. **Why it's free
now:** HELLO already advertises `software_version` + `update`/`update_push`, so a hub knows
which nodes *could* accept a push without a new handshake. The later updater only fills the
`M_UPDATE` body and populates the pubkey table — zero wire change, capabilities are an open map.

**Update sequence (later):** `drain → swap → rejoin`:
```
hub M_UPDATE ─▶ node: verify sig → finish in-flight chats (drain) → download+swap binary
            ─▶ node re-execs → re-binds same endpoint.key (same EndpointId) → re-dials hub
            ─▶ HELLO with new software_version → allowlist passes (id unchanged) → rejoin
```

---

## 10. Phased Plan (tasks #12–#18)

Each phase is independently shippable and gated by `quality.sh`. **Net-new node code
(NodeRuntime + registry + WorkerId) lands in P2/P3; the M_PULL downloader is its own
sub-phase.**

```
P0  Actor runtime (foundational) src/actor.rs: `trait Actor { type Msg; handle }` + `spawn_actor`
    ─────────────────────────    (mailbox + recv loop + shutdown, written ONCE) + the client
                                 reply-demux helper (reader task → pending/chat_sinks, §2.5).
                                 Factor it out of today's Supervisor; port Worker onto it (minimal
                                 tokio runtime + spawn_blocking FFI) so Supervisor + Worker share
                                 ONE runtime. NodeRuntime + per-node transport (P2/P3) are then just
                                 more `Actor` impls. NO new dep (tokio only).
                                 Verify: existing supervisor + worker tests stay green on the shared
                                 runtime; no behaviour change, no duplicated loop.

P1  Pairing + handshake  (#12)   TASK 1 (FIRST): scaffold the iroh Endpoint + SecretKey
    ─────────────────────────    persistence (~/.higgs/endpoint.key) + pairing/HELLO against the
                                 CONFIRMED iroh 1.0 API (names verified, §3 reference; no
                                 PROVISIONAL items remain).
                                 THEN: ALPN; src/auth.rs allowlist + pairings.json; pairing token
                                 mint/burn; HELLO frame + version negotiation; post-HELLO gate +
                                 HELLO-stalled timer. HG022/HG023/HG024/HG028. No chat yet.
                                 Verify: two binaries pair, HELLO agrees, stranger rejected post-HELLO,
                                 silent post-QUIC peer dropped after the deadline (HG028).
                                 iroh 1.0 API reference (confirmed): EndpointId (= PublicKey; iroh has
                                 NO NodeId), Connection::remote_id(), EndpointAddr (+ with_relay_url /
                                 with_addrs / with_ip_addr; NO with_direct_addresses),
                                 iroh_tickets::endpoint::EndpointTicket::endpoint_addr(),
                                 Endpoint::builder(preset).secret_key(sk).bind().await,
                                 SecretKey::from_bytes/to_bytes, open_bi/accept_bi → (SendStream,
                                 RecvStream), Endpoint::accept() → Incoming → Connection.

P2  Node mode + NodeRuntime (#13) src/node.rs: bind + accept loop + post-HELLO gate + NEW
    ─────────────────────────    NodeRuntime { HashMap<WorkerId, Arc<Supervisor>> } + WorkerId
                                 newtype (u32, LOCKED §3.0/§5.4a) + node-side CONTROL dispatch
                                 (§5.4a — registry ops, NOT WorkerState) + outbound hello/inventory
                                 requester (§5.4c). DATA bridge per worker → serve_state
                                 (SyncIoBridge per stream in spawn_blocking). --node / node connect.
                                 M_LOAD spawns a 2nd+ Supervisor (multi-worker, NET-NEW) with the
                                 VRAM fit-check; M_KILL/M_UNLOAD free the WorkerId.
                                 Verify: node hosts 2 concurrent workers, M_SYSINFO + M_STATUS over iroh.

P3  Hub seam + relay      (#14)  remote_factory + spawn_remote ctor; WorkerProc trait replacing
    ─────────────────────────    proc:Option<Child> across the 7 reap sites; per-node iroh transport
                                 on the hub with its OWN pending/correlation, keyed by NodeId, routing
                                 per (NodeId, WorkerId) — DISTINCT from the local-worker Supervisor
                                 (§2.3); hub-side chat relay (request_id translation table);
                                 wedged-worker reap policy. HG027 NodeUnreachable.
                                 Verify: /v1 chat to a remote-resident worker streams back; a wedged
                                 worker escalates M_KILL → redial → retire (HG027).

P4  Inventory + LogSource (#15)  M_INVENTORY{boot|refresh} (NO M_HEARTBEAT — deltas fold into
    ─────────────────────────    refresh, iroh keepalive = liveness) / M_SCAN;
                                 HashMap<NodeId,NodeView> (+hostname/os/IP §4.2.1; HW/RT gain
                                 Deserialize); LogSource::RemoteWorker{node,worker}
                                 (stays Copy — WorkerId u32, §6) + keyed
                                 remote ring map + eviction on unload/kill/retire (§6.1);
                                 LogSource::parse node arm; N_LOG_LINE relay.
                                 Per-node ?source=node:<id>:<worker> selector.
                                 Verify: 2 nodes' worker logs interleave + filter by (node,worker);
                                 a killed worker's ring is reclaimed.

P4b M_PULL downloader     (#15b) NEW HF downloader sub-phase (fix D — no download exists today):
    ─────────────────────────    `hf-hub` (§3.0) → ~/.higgs/models/ ONLY; N_PROGRESS on the data
                                 plane; HG025 DownloadFailed. Never writes scanned dirs.
                                 Verify: M_PULL downloads a GGUF into ~/.higgs/models, progress streams,
                                 a subsequent M_SCAN/M_LOAD sees it.

P5  Bearer auth           (#16)  api_keys.json + SHA-256 middleware on /v1 + /api/higgs/*;
    ─────────────────────────    scopes (chat|models|admin); keys CLI. 401 envelope.
                                 Verify: scoped key allows chat, admin gates node mgmt.

P6  UI                    (#17)  /api/higgs/nodes panel (fleet view from NodeView map — incl.
    ─────────────────────────    observed_addr + path, §4.2.1); pairing QR flow; per-node +
                                 per-worker load/unload/kill; keys management pane. ts-rs exports
                                 for NodeView + NodeInventory + NodePath / HelloResult / LogSource.
                                 Verify (Playwright): pair a node, load 2 workers, chat, see logs.

(#18  M_UPDATE)                  Reserved this design (§9): const + capability + pubkey table +
                                 HG026 stub. Real updater is a later, separate task.
```

---

## 11. Open Decisions / Risks

| # | Item | Decision / mitigation |
|---|---|---|
| 1 | **No remote SIGKILL** | A node behind NAT cannot be force-killed by the hub. `WorkerProc::force_kill` for remote = close the QUIC conn; the node's own supervisor SIGKILLs the local worker. If the node is unreachable, the hub can only retire it (HG027) — actual reap is best-effort. **Accepted.** |
| 2 | **Relay latency** | First chat token over a relay path (pre-hole-punch) adds RTT. iroh upgrades to direct transparently; measure P50/P99 first-token in P3. If relay-only paths dominate, document a self-hosted relay (`presets::Minimal` + `RelayMode::Custom`). |
| 3 | **Type-publish seam** | `NodeView` (+ `NodeInventory`/`NodePath`)/`HelloResult`/`HelloParams`/`LogSource` cross to the frontend only at P6. Until then they are crate-internal serde shapes. When published, add `higgs_ts!` exports (barrel rules unchanged). Do not export pre-P6. |
| 4 | **`WorkerId` repr + lifecycle** | **LOCKED: `u32` newtype (§3.0/§5.4a)** — no string crate, `Copy`, numeric wire (`"worker_id":1`). Single home = `NodeRuntime` registry; assigned on `M_LOAD`, freed on `M_UNLOAD`/`M_KILL`. Must survive a node-side respawn (restart FSM) so the hub's data stream + request_id table (§4.3) stay valid, OR the hub re-opens the stream on `WorkerId` change — resolve the respawn-stability detail in P3 (prefer a stable id across respawn). |
| 5 | **`LogSource` Copy** | **RESOLVED: stays `Copy`.** `WorkerId` locked to `u32` (`Copy`) + `NodeId` wraps `EndpointId` (`Copy`) ⇒ `RemoteWorker` is `Copy` ⇒ no demotion, no change to `LogLine.source` (`:81`) / `broadcast::Sender<LogLine>` (§6). |
| 6 | **`accept_bi()` first-write + HELLO-stalled** | Opener must write HELLO immediately or the acceptor hangs (§3.3); a peer that completes QUIC then never sends HELLO is dropped after `HELLO_DEADLINE` (HG028, §3.2.1) to bound the pre-auth window. Both covered by P1 integration tests. |
| 7 | **Pairing-token transport** | Token TTL + single-use guards a leaked QR. Token never grants more than "add my EndpointId to the allowlist." A stolen token = one rogue node id, revocable by removing it from `pairings.json`. **Accepted** with the standard revocation path. |
| 8 | **iroh 1.0 API surface** | **VERIFIED against docs.rs (iroh 1.0.0) — no PROVISIONAL items remain.** `EndpointTicket::endpoint_addr()` (from the separate `iroh-tickets` crate) and `EndpointAddr::with_relay_url`/`with_addrs`/`with_ip_addr` (NO `with_direct_addresses`) are confirmed. Identity is `EndpointId` (iroh has no `NodeId`); gate accessor is `Connection::remote_id()`. Full confirmed list in §3 reference + §10-P1. |
| 9 | **Data-plane backpressure** (fix #3) | Each per-worker data stream runs `serve_state` over a sync `SyncIoBridge` in its own `spawn_blocking` (§5.3); `handle_chat` streams `N_CHAT_CHUNK` through the sync writer (`worker/mod.rs:356-365`). A slow QUIC `SendStream` blocks the write and **stalls generation on that worker** until it drains. Per-stream isolation confines the stall. Measure with §11-2 in P3; escape hatch = an async writer with a bounded buffer (NOT a `serve_state_async` rewrite). |
| 10 | **request_id collision** (fix #4) | Hub per-node iroh transport and node stdio `Supervisor` both `alloc_request_id` (`supervisor.rs:346`) independently → ids alias. Decided: node re-allocates a fresh node-local id + per-data-stream `hub_id ↔ node_id` table; inbound `N_CHAT_CHUNK` rewritten to hub-id before relay. Per-stream table ⇒ no cross-stream aliasing. Implement in P3. |
| 11 | **Wedged remote worker** (fix F) | FFI hang with conn warm: `conn.closed()` never fires. Hub policy = chat-stall past deadline → M_KILL(worker_id) → no ack → redial → re-M_KILL → still stuck → retire node (HG027) + surface. No silent GPU pin (§3.4.1). |
| 12 | **VRAM fit before extra worker** | A node hosting N workers must FIT-CHECK free VRAM (reuse the FitAssessment path against `GpuDevice::vram_free_bytes`, summed across resident workers) before spawning the N+1th. Won't-fit → HG017 InsufficientMemory, no spawn (§4.2b). |

---

**Relevant files:** `src/supervisor.rs` (seam + liveness — REUSED per-worker unit),
`src/worker/mod.rs` (M_* vocab + `serve_state` + read-only-scan note), `src/rpc.rs`
(RpcFrame wire), `src/log_bus.rs` (LogSource), `src/api.rs` (`Higgs` facade — `sysinfo`
returns `Vec<GpuDevice>` at :996, `scan` at :604, `load` only-keep-last :363), `src/system.rs`
(`SystemInfo::gather(config, gpus)` :125 → HardwareInfo :57 +hostname/os_name/os_version,
RuntimeInfo :86; both gain `Deserialize` for remote inventory, §4.2.1) +
`src/worker/models.rs` (HiggsModel scan, LoadedInfo at api.rs:170), `src/bin/higgs.rs`
(role selector :33; bin = `higgs`), `src/serve/control.rs` (logs filter :44),
`src/diagnostic.rs` (HG codes — max HG021 at :185, remote adds HG022–HG028), **new
`src/node.rs`** (Endpoint + `NodeRuntime` registry + per-node iroh transport + remote_factory),
**new `src/auth.rs`** (pairings.json / api_keys.json), `Cargo.toml` (add — all vetted ≥100k dl,
§3.0: iroh + iroh-base + iroh-tickets 1.0 [P1], toml [P2], hf-hub [P4b], sha2 + subtle [P5],
qrcode optional [P6], minisign-verify deferred [#18]; `WorkerId`=u32 + node-IP need NO crate).
