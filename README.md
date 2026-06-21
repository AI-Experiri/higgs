# higgs

**A self-contained local LLM runtime in Rust.** Point it at a folder of GGUF files and it
serves OpenAI-compatible inference over HTTP — embedded in your Axum app or as a standalone
binary — with crash-isolated worker processes, a pluggable engine layer, and an optional
peer-to-peer **remote fleet** that runs models on other machines over an encrypted QUIC link.

```
┌──────────────────────────────────────────────────────────────────────┐
│  OpenAI client  ─HTTP→  /v1/chat/completions   /v1/models              │
│  Admin / UI     ─HTTP→  /api/higgs/{status,models,system,nodes,logs}   │
└───────────────────────────────┬──────────────────────────────────────┘
                                 │  (pure-Rust router + facade, can't crash)
                          ┌──────▼───────┐
                          │    Higgs     │  facade: load / chat / status / logs
                          └──────┬───────┘
            local model          │              remote model
        ┌────────────────────────┴───────────────────────────┐
        ▼                                                      ▼
   Supervisor ──stdio JSON-RPC──> worker process        HubFleet ──iroh QUIC──> node ──stdio──> worker
        │                          (llama.cpp engine)                            (another machine)
        ▼
   GGUF on disk
```

## Table of Contents
- [Features](#features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [HTTP API](#http-api)
- [Remote Fleet](#remote-fleet)
- [Adding a New Engine](#adding-a-new-engine)
- [Diagnostics](#diagnostics)
- [Testing & Coverage](#testing--coverage)
- [File Map](#file-map)

---

## Features

**Inference**
- OpenAI-compatible `POST /v1/chat/completions` — streaming (SSE) and non-streaming.
- `GET /v1/models` model listing (local + remotely-routable).
- Sampling controls: `temperature`, `top_p`, `max_tokens`, `seed`, `stop`, presence/frequency penalties.
- Tool / function calling: the GGUF-embedded chat template renders the OpenAI `tools` array and the matching tool-call parser is selected automatically (no custom parser invented). Multi-turn tool loops round-trip (`assistant.tool_calls` → `tool` results).
- Usage accounting (`prompt_tokens` / `completion_tokens`), with `stream_options.include_usage`.

**Model management**
- Host-side model discovery across **LM Studio**, **HuggingFace cache**, and **Ollama** stores — works even with no worker running.
- **Spawn-on-load / kill-on-unload**: zero model loaded ⇒ zero worker process ⇒ zero idle RAM.
- **Just-in-time (JIT) load**: a chat for a scanned-but-unloaded model loads it on demand.
- **Idle reaper**: auto-unloads a model after a configurable idle TTL.
- Pre-load **RAM/VRAM headroom guard** so a model that can't fit is rejected before it thrashes the host.
- Rich load tuning: `ctx_len`, `gpu_layers`, `threads`, mmap/mlock, batch sizes, KV-cache type, flash-attention.

**Reliability & isolation**
- The inference engine runs in a **separate worker process** (re-exec with `--higgs-worker`); a worker crash can never take down the host.
- The **supervisor** restarts a crashed worker once and **replays the last load**, so the model self-heals.
- Pure-Rust control plane (router + facade) — only the worker is native FFI.

**Observability**
- Unified **Developer Logs**: worker stderr + serve-layer tracing in one `LogBus`.
- `GET /api/higgs/logs` snapshot tail and `GET /api/higgs/logs/stream` live SSE.
- `?source=` filter: `serve`, `worker`, or a remote worker `node:<id>:<worker>`.
- `GET /api/higgs/system` — real CPU/RAM/GPU/VRAM + engine/runtime versions.

**Remote fleet (peer-to-peer)**
- Run models on **other machines** over an encrypted [iroh](https://iroh.computer) QUIC link — no central server, no open inbound ports required.
- **Pairing** with one-time tokens + a persistent allowlist; version-negotiated HELLO handshake.
- The hub routes `/v1` chat to a remote-resident worker transparently (two hops: hub → node → worker).
- **Durable routes** survive reconnects; a dead worker self-heals or re-routes.
- Remote worker stderr is **relayed back** into the hub's log console, tagged per node/worker.
- `GET /api/higgs/nodes` — fleet view: each node's identity, connection state, hardware, and resident workers.

**Extensibility**
- A single [`HiggsEngine`](src/worker/engine/mod.rs) trait is the only seam between higgs and an inference backend. Adding a new engine (e.g. MLX) is **implement the trait + one registry line** — see [Adding a New Engine](#adding-a-new-engine).

---

## Architecture

### Process model (local)

higgs keeps the control plane and the native engine in **separate processes** so the engine
can never crash the host:

```
host process (your app, or the `higgs` binary)
│
│  Arc<Higgs>            ← pure Rust: router, facade, model scan, log bus
│     │
│     │ load(model)
│     ▼
│  Supervisor  ──spawn──>  worker process  ("higgs(<model>)")
│     │   newline-delimited JSON-RPC 2.0 over stdio       │
│     │   M_LOAD / M_CHAT / M_STATUS / M_SYSINFO          │  HiggsEngine (llama.cpp FFI)
│     │<──────── N_CHAT_CHUNK stream + final ────────────┤
│     ▼                                                    ▼
│  restart-on-crash + replay last load               GGUF on disk
```

- **`Higgs`** (`src/api.rs`) is the host-facing facade: `load`, `chat_stream`, `status`, `logs`.
- **`Supervisor`** (`src/supervisor.rs`) owns one worker child, correlates JSON-RPC requests, routes streamed chat chunks, and restarts + replays on crash.
- The **worker** (`src/worker/`) is a JSON-RPC dispatch loop over stdio driving a `HiggsEngine`.
- **Model scanning** is host-side, so `/v1/models` and `/api/higgs/models` work with no worker.

### Two surfaces, one router

```
            ┌───────────────── Axum Router ─────────────────┐
  /v1/*  ───┤  OpenAI compat:  chat/completions, models      │
            │                                                 │
/api/higgs/*┤  control: status, system, nodes, models/*,     │
            │           logs, logs/stream, settings, worker   │
            └─────────────────────────────────────────────────┘
```

Mount it in any Axum host with one call, or run the standalone `higgs` binary.

### Remote fleet (two hops, never one)

```
   HUB (has the OpenAI clients)                 NODE (has the GPUs)
   ┌────────────────────────┐                  ┌──────────────────────────┐
   │ Higgs ── HubFleet       │                  │ NodeRuntime               │
   │   │       │             │   iroh QUIC      │   │  (N workers)          │
   │   │   NodeTransport ────┼===(ALPN higgs/   │   ├─ Supervisor ─ worker  │
   │   │       │             │    remote/1)=====┼──>│   (stdio JSON-RPC)    │
   │   │   routes:           │  M_NODE_LOAD     │   ├─ Supervisor ─ worker  │
   │   │   model→(node,wkr)  │  M_CHAT          │   └─ ...                  │
   │   │   N_LOG_LINE  <─────┼──────────────────┼── relayed worker stderr   │
   └────────────────────────┘                  └──────────────────────────┘
```

Invariants: the hub **never** speaks to a remote worker directly; control (`higgs/node/*`) and
data (`M_CHAT`) are separate dispatchers; the worker stays pure-sync stdio in every topology.
See `DESIGN-remote.md` for the full spec.

---

## Quick Start

```bash
# Build
cargo build --release

# Run the standalone server against a folder of GGUF models
HIGGS_MODEL_DIR=/path/to/models HIGGS_BIND=127.0.0.1 HIGGS_PORT=11434 \
  ./target/release/higgs

# Chat (OpenAI-compatible)
curl localhost:11434/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "your-org/your-model",
  "messages": [{"role":"user","content":"Hello!"}],
  "stream": false
}'
```

Environment knobs: `HIGGS_MODEL_DIR` (extra LM-Studio scan root), `HIGGS_BIND`, `HIGGS_PORT`,
`HIGGS_HOME` (state dir, default `~/.higgs`), `HIGGS_ENGINE` (engine selector, default
`llamacpp`), `HIGGS_VERBOSE=1` (keep full engine stderr).

## Usage

### Embed in an Axum app

```rust
use std::sync::Arc;
use higgs::{Higgs, HiggsConfig};

let higgs = Arc::new(Higgs::new(HiggsConfig::default()));
let app = axum::Router::new().merge(higgs::serve::router(higgs.clone()));
// serve `app` on any listener — /v1/* and /api/higgs/* are now live.
```

### Drive the facade directly

```rust
let (mut deltas, done) = higgs
    .chat_stream(model, messages_json, /*max_tokens*/ 256, /*temp*/ 0.7, /*tools*/ None)
    .await?;
while let Some(delta) = deltas.recv().await { print!("{delta}"); }
let outcome = done.await??;   // ChatOutcome { content, finish_reason, usage, ... }
```

## HTTP API

| Method & path | Purpose |
|---|---|
| `POST /v1/chat/completions` | OpenAI chat (stream + non-stream, tools, sampling params) |
| `GET /v1/models` | List loaded + remotely-routable models |
| `GET /api/higgs/status` | Live engine + loaded-model status |
| `GET /api/higgs/system` | Host hardware + engine/runtime versions |
| `GET /api/higgs/nodes` | Remote fleet view (per-node identity, hardware, workers) |
| `GET /api/higgs/models` | Scanned model catalog |
| `GET /api/higgs/models/{org}/{model}` | One model's details |
| `POST /api/higgs/models/load` · `…/unload` | Load / unload a model |
| `POST /api/higgs/worker/stop` | Stop the worker process |
| `GET /api/higgs/logs` · `…/logs/stream` | Developer-Log tail (snapshot / live SSE), `?source=` filter |
| `GET·PUT /api/higgs/settings` · `…/logs/settings` | Runtime + logging toggles |
| `GET /api/higgs/version` · `/health` | Build version · health check |

## Securing the API

The HTTP surface is **open by default** (intended for loopback / embedded use). To require
authentication, add API keys — the gate turns on as soon as the first key exists:

```bash
higgs keys add ci chat,models   # mint a key (scopes: chat | models | admin)
#   token (shown ONCE …): hgk_…
higgs keys list
higgs keys remove ci
```

Clients then send `Authorization: Bearer hgk_…`. Scopes: `chat` (POST `/v1/chat/completions`),
`models` (model listing), `admin` (everything, incl. management). Tokens are stored only as a
SHA-256 digest in `~/.higgs/api_keys.json`. The standalone server loads keys at startup and
**fails closed** if the file is present but unreadable; **restart higgs** for key changes to
take effect on a running server. Health checks (`/health`) are always open.

## Remote Fleet

```bash
# Option A — run the SERVER itself as a hub (recommended): one process serves HTTP
# AND accepts node dials. Mint a join token over the API:
HIGGS_HUB=1 higgs                                  # server in hub mode
curl -X POST localhost:11434/api/higgs/pair        # → { ticket, token, node_command }

# Option B — hand-pair from the CLI (separate accept loop):
higgs link pair        # prints hub id + a single-use token (10m) + ticket
higgs link status      # list pairings

# On the NODE: run the persistent daemon, dialing the hub. Pulled/scanned models load here.
HIGGS_MODEL_DIR=/path/to/models higgs --node <ticket> <token>

# Pull a model onto a node from HuggingFace (lands in the node's ~/.higgs/models/):
#   issued by the hub over M_NODE_PULL; progress streams as N_PROGRESS.
```

With the server in hub mode (`HIGGS_HUB=1`), `GET /api/higgs/nodes` lists every paired
node (connected or not) with its hardware + resident workers, and `/v1/chat/completions`
for a remote-resident model is routed there automatically.

Once paired, the hub can load a model on the node (via `HubFleet`/the API) and any
`/v1/chat/completions` for that model is routed there automatically; `GET /api/higgs/nodes`
shows the node, its hardware, and its resident workers. See `src/node/DESIGN.md`.

## Adding a New Engine

higgs talks to inference backends only through the [`HiggsEngine`](src/worker/engine/mod.rs)
trait and selects one at startup from a registry. Adding an engine is **three steps, no other
file changes**:

1. **Implement the trait** in a new submodule, e.g. `src/worker/engine/mlx/mod.rs`:
   ```rust
   #[derive(Default)]
   pub struct MlxEngine { /* backend handle */ }
   impl higgs::worker::engine::HiggsEngine for MlxEngine {
       fn load(&mut self, path: &str, p: &LoadParams) -> Result<(), HiggsError> { … }
       fn unload(&mut self) { … }
       fn is_loaded(&self) -> bool { … }
       fn chat(&mut self, messages_json: &str, p: &GenParams,
               sink: &mut dyn FnMut(&str)) -> Result<ChatResult, HiggsError> { … }
       fn probe(&self, path: &str) -> (bool, Option<String>) { … }
       fn devices(&self) -> Vec<GpuDevice> { … }
   }
   ```
   Keep all backend-specific dependencies (FFI, crates) inside that submodule.
2. **Register it** — one line in `REGISTRY` (`src/worker/engine/mod.rs`):
   ```rust
   EngineEntry { name: "mlx", build: || Box::new(mlx::MlxEngine::default()) },
   ```
3. **Select it** at runtime: `HIGGS_ENGINE=mlx`. The first registry entry is the default.

Nothing in the worker dispatch, supervisor, node runtime, or serve layer needs to change —
they are all engine-agnostic above the trait. See `src/worker/engine/DESIGN.md`.

## Diagnostics

Every failure is a typed `HiggsError` carrying a stable `HGxxx` code (e.g. `HG002` model not
found, `HG004` engine load failure, `HG007` worker unavailable, `HG017` insufficient RAM,
`HG022`–`HG028` remote pairing/transport). Codes map to HTTP statuses at the serve boundary
and ride the JSON-RPC `error.data.code` so the true origin status survives the hub→node hops.

## Testing & Coverage

```bash
scripts/quality.sh               # fast gate: fmt + clippy + test + bindings sync
cargo test                       # unit + integration only
scripts/coverage.sh              # both coverage gates with a combined summary
```

### Quality gate (`scripts/quality.sh`)

The fast pre-commit gate. Runs, in order:
1. `cargo fmt --all` (apply, then `--check`) — the style is pinned in `rustfmt.toml`
   (`edition 2021`, `max_width 100`); keep changes formatted or the gate fails.
2. `cargo clippy --all-targets -D warnings`.
3. `cargo test` — the full suite, which also **regenerates** the ts-rs bindings under
   `bindings/higgs/` (emitted by the `higgs_ts!` macro's `#[ts(export)]` derive tests).
4. **Bindings sync** — fails if `bindings/` drifted after the test pass, so a Rust wire-type
   change can't land without the regenerated TypeScript committed alongside it.

### Coverage gates (`scripts/coverage.sh`)

Two independent gates, run separately:
- **Unit** (`coverage-unit.sh`): `cargo test --lib`, **≥90% lines** (excludes the daemon `main` + FFI, which only run in a spawned process).
- **Integration** (`coverage-integration.sh`): the `tests/` targets only, **≥75% lines** (excludes the pure-logic `tool_parser` subtree the unit gate owns).

```bash
scripts/coverage.sh              # both gates; runs both even if one fails, then a summary
scripts/coverage.sh -u           # unit gate only          (--unit)
scripts/coverage.sh -i           # integration gate only   (--integration)
scripts/coverage.sh -u --open    # unit gate + open its HTML report
scripts/coverage.sh --html       # write HTML report(s) under target/
```

Any non-selector flag is forwarded verbatim to `cargo llvm-cov` (`--html`, `--open`, `--json`,
`--summary-only`, `--output-dir DIR`, …). When both gates run, both execute even if the first
fails and a combined pass/fail + line-% summary prints at the end; the script exits non-zero if
any gate failed. Requires `cargo-llvm-cov` (`cargo install cargo-llvm-cov`).

Integration tests spawn the real `higgs` binary and drive it over HTTP + iroh against a real
~1MB GGUF (`HIGGS_TEST_GGUF` overrides the path; tests **skip** when it's absent), exercising
spawn → pair → load → chat → unload end to end.

## File Map

| Path | What it does |
|------|-------------|
| `src/lib.rs` | Crate root — re-exports `Higgs`, `HiggsConfig`, `HiggsError`, `HiggsEvent` |
| `src/api.rs` | `Higgs` facade + `HiggsConfig`; status/config/outcome types; idle reaper; fleet hook |
| `src/log_bus.rs` | `LogBus` (history rings + live broadcast; local + remote-worker sources) and `HiggsLogLayer` |
| `src/diagnostic.rs` | `HiggsError` enum with diagnostic codes (HG001–HG028) |
| `src/rpc.rs` | NDJSON JSON-RPC 2.0 encode/decode — the supervisor↔worker wire |
| `src/supervisor.rs` | Worker process supervisor: spawn, restart+replay, request correlation, chat-chunk routing |
| `src/system.rs` | Host hardware/runtime snapshot (CPU/RAM/GPU/VRAM, engine versions) |
| `src/serve/mod.rs` | Axum router: `/v1/*` + `/api/higgs/*` (incl. the no-timeout SSE log stream) |
| `src/serve/v1.rs` · `stream.rs` | OpenAI `/v1` handlers · SSE chunk assembly |
| `src/serve/control.rs` | `/api/higgs/*` control handlers (status/system/nodes/logs/settings) |
| `src/worker/mod.rs` | Worker entry point — JSON-RPC dispatch loop, engine selection |
| `src/worker/models.rs` | Model discovery (LM Studio, HuggingFace, Ollama); `HiggsModel`, `ModelStore` |
| `src/worker/engine/mod.rs` | `HiggsEngine` trait + engine **registry** (add an engine here) |
| `src/worker/engine/llamacpp/` | llama.cpp engine impl: chat templates, fit-check, token streaming, device enum |
| `src/worker/tool_parser/` | Per-dialect tool-call parsers (selected by the chat template) |
| `src/remote.rs` | Remote wire vocabulary: ALPN, HELLO, `higgs/node/*` methods, inventory, version negotiation |
| `src/auth.rs` | Pairing allowlist + one-time tokens |
| `src/node/mod.rs` | iroh bind / accept-gate / dial; node serve loop; log relay |
| `src/node/cli.rs` | `link pair/status`, `node connect`, `--node` daemon |
| `src/node/runtime.rs` | `NodeRuntime`: multi-worker registry, load/unload/sysinfo/inventory |
| `src/node/transport.rs` | Hub-side per-node iroh client (`NodeTransport`) |
| `src/node/fleet.rs` | `HubFleet`: routing, durable routes, log relay, inventory/`NodeView` |
| `src/node/identity.rs` | Persisted iroh `SecretKey` / stable `EndpointId` |

For the remote design rationale and invariants, see `DESIGN-remote.md` and `src/node/DESIGN.md`.
