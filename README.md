# higgs

**A self-contained local LLM runtime in Rust.** Point it at a folder of GGUF files and it
serves OpenAI-compatible inference over HTTP — embedded in your Axum app or as a standalone
binary — with crash-isolated worker processes, a pluggable engine layer, an optional
peer-to-peer **remote fleet** over encrypted QUIC, and opt-in API-key auth.

higgs is a **standalone crate**: it imports nothing from any jigglebot crate (`src/lib.rs`).
The control plane (router + `Higgs` facade) is pure Rust and cannot crash; only the worker
process links the llama.cpp FFI.

```
┌──────────────────────────────────────────────────────────────────────┐
│  OpenAI client  ─HTTP→  /v1/chat/completions   /v1/models              │
│  Admin / UI     ─HTTP→  /api/higgs/{status,models,system,nodes,logs}   │
└───────────────────────────────┬──────────────────────────────────────┘
                                 │  (pure-Rust router + facade, can't crash)
                          ┌──────▼───────┐
                          │    Higgs     │  facade: load / chat / status / logs / tune
                          └──────┬───────┘
            local model          │              remote model
        ┌────────────────────────┴───────────────────────────┐
        ▼                                                      ▼
  NodeRuntime ──Supervisor──stdio JSON-RPC──> worker    HubFleet ──iroh QUIC──> node ──> worker
  (multi-worker registry)     (llama.cpp engine)                               (another machine)
        ▼
   GGUF on disk
```

## Scope of this document

This is the **crate-level** README. It covers the crate as a whole plus the **loose top-level
modules** in `src/*.rs`. The five sub-folders own their own `README.md` + `DESIGN.md` and are
only named + linked here:

| Sub-folder | What it owns | Docs |
|---|---|---|
| `src/api/` | `Higgs` facade internals split out of `api.rs`: `guards.rs` (memory/repo-id/path guards), `types.rs` (`HiggsConfig`, `HiggsServerConfig`, wire status/outcome types) | `src/api/README.md` |
| `src/node/` | iroh remote fleet: hub, node runtime, transport, fleet routing, identity, CLI | `src/node/README.md`, `DESIGN-remote.md` |
| `src/serve/` | Axum router: `/v1/*` + `/api/higgs/*` handlers, SSE assembly, hardening layers | `src/serve/README.md` |
| `src/tune/` | analytical + measured (Turbotune) autotune, per-model `models.json` store | `src/tune/README.md` |
| `src/worker/` | re-exec'd worker: JSON-RPC dispatch loop, model discovery, `HiggsEngine` trait + llama.cpp engine | `src/worker/README.md` |

The **`higgs-macros`** workspace crate (`higgs-macros/`) provides two proc-macro derives used
by the ts-rs bindings: `TsConstEnum` (emit a TypeScript const-object for a unit-variant enum,
usable as a value) and `TsParamHelp` (emit `PARAM_HELP` tooltip text from `#[help]` attrs). It
is a standalone workspace member because higgs cannot depend on jigglebot's `macros` crate.

---

## Loose top-level modules (`src/*.rs`)

| File | Responsibility |
|------|----------------|
| `lib.rs` | Crate root — module map; re-exports `Higgs`, `HiggsConfig`, `HiggsError`, `HiggsEvent`, `LogBus`/`HiggsLogLayer`/`log_filter`, `run_standalone`/`shutdown_signal`/`StandaloneConfig`; `DEFAULT_PORT` (31415); the crate-internal `LLAMA_CPP_2_VERSION` (`"0.1.151"`, the bundled binding). `#[macro_use] mod ts_export` first so `higgs_ts!` is in scope everywhere. |
| `api.rs` | The `Higgs` public **facade**. Typed wrapper over a co-located LOCAL `NodeRuntime` (the same multi-worker engine remote nodes run). Owns facade state — live `HiggsConfig`, the load-lifecycle mutex, the two inference admission gates, serve-layer toggles, the readiness/JIT gate, tune/estimate, the OOM retry wiring — and delegates spawn/load/unload/status/chat to the node. |
| `supervisor.rs` | Manages ONE worker child: spawn by re-exec (`--higgs-worker`), NDJSON JSON-RPC 2.0 over stdio, correlate responses by id, route chat-chunk notifications, restart-once + replay-last-load on unexpected death. Owns `HiggsEvent`, the RPC timeouts (`CONTROL` 120 s, `CHAT_RPC_TIMEOUT` 600 s, `SYSINFO` 15 s). |
| `actor.rs` | The generic actor runtime (`Actor` trait, `Handle`/`WeakHandle`, `spawn_actor{,_with}`) written once, plus `ReplyDemux` (request-id → response waiter + `request_id` → chat sink). Reused by `Supervisor`, `NodeRuntime`, and the per-node transport. |
| `delta_queue.rs` | **(G1)** Bounded, MERGING chat-delta channel — the backpressure buffer between a delta producer and its single streaming consumer. Coalesces consecutive same-kind deltas in place (entries bounded by kind-alternations, not token count); `CAP_BYTES` (8 MB) hard cap trips `HG057` on a stalled client instead of unbounded growth. `delta_channel()` → `(DeltaSender, DeltaReceiver)`. |
| `rpc.rs` | NDJSON JSON-RPC 2.0 codec — the supervisor↔worker AND hub↔node wire. `RpcRequest`/`RpcResponse`/`RpcError`/`RpcNotification`/`RpcFrame`, `encode`/`decode` (`HG008` on bad frame), `method_not_found` (`HG037`). Soft-validates the `jsonrpc` version (warn, not reject). |
| `diagnostic.rs` | `HiggsError` — every higgs failure, each carrying a stable `HGxxx` code baked into `Display`. Append-only; never renumber. Also documents the **log-only** codes (HG051–HG056, HG061/HG062/HG065) that ride a `tracing` line, not a variant. |
| `system.rs` | Host hardware + inference-runtime snapshot for `/api/higgs/system`: `SystemInfo`, `HardwareInfo`, `RuntimeInfo`, `GpuDevice`, `DeviceKind`, plus `fits_vram`/`FitAssessment` and `HardwareInfo::{fingerprint, free_vram_bytes, is_unified_memory}`. `hostname()`. |
| `log_bus.rs` | The single home for Developer-Log lines: `LogBus` (a bounded history ring PER `LogSource` + a live broadcast tap), `HiggsLogLayer` (tracing layer mirroring `higgs`-target events), `log_filter`. `LogSource` = `Serve`/`Worker`/`LocalWorker`/`RemoteWorker`. |
| `keys.rs` | **(G4)** API-key auth: `ApiKeys`/`ApiKey`/`Scope` (`Chat`/`Models`/`Admin`). Tokens stored only as a SHA-256 hex digest; constant-time compare; empty store ⇒ auth OFF (open). `hash_token`/`mint_token`; `higgs keys <add\|list\|remove>` CLI (`run_keys`); live mutation over HTTP via `/api/higgs/keys` (Admin scope). |
| `auth.rs` | Remote-fleet **allowlist** (`pairings.json` → `Allowlist`) and one-time **pairing tokens** (`PairingTokens`, mint/validate/burn). Atomic saves; corruption → `HG041`, I/O → `HG040`. |
| `config.rs` | `~/.higgs/config.json` (`InstanceConfig`): friendly `name`, node-side saved hubs (`SavedHub` + `remember_hub`/`find_hub`/`remove_hub`), per-model load records (`ModelRecord`), extra `cors_origins`. `Role`, `friendly_name`, `name_or_init`. |
| `home.rs` | `~/.higgs` home dir (`higgs_home`, `ensure_home`); `HIGGS_HOME` override. The single home for all identity/state files. |
| `hub.rs` | **HuggingFace Hub client — PRIMARY fetch path.** `HubFetcher` (streaming) + `fetch_bytes` (in-memory card/config), honoring `HIGGS_HF_ENDPOINT`. `classify_hf`/`http_status_to_error` map failures to distinct `HG029`–`HG035`. |
| `download.rs` | Model downloader (`M_PULL`): `Fetcher` trait, `HttpFetcher` (reqwest FALLBACK), `download`/`download_dual` (primary+fallback → `HG036` if both fail), `dest_path` path-traversal guard, atomic `.part`+rename into `~/.higgs/models/`. |
| `load_robustness.rs` | **(G5)** Pure OOM decision logic wired into the load path: `is_oom_reason`, `oom_ladder` (settle → KV-to-RAM → fewer GPU layers → `HG060`), `LoadRung`, `SETTLE_BEFORE_RETRY` (750 ms VRAM settle). |
| `remote.rs` | Remote wire vocabulary: `ALPN`, `M_HELLO`, the `higgs/node/*` methods, `HelloParams`/`HelloResult`, `NodeLoadParams`/`NodeChatParams`/`NodeInventory`/`InventoryWorker`, capabilities, `negotiate_version`, `PAIRING_TOKEN_TTL_MS`. |
| `standalone.rs` | The testable core of the `higgs` binary: `run_standalone` (build `Higgs` → load keys → bind-exposure guards `HG058`/`HG069` → optional hub → serve), `StandaloneConfig`, `shutdown_signal`, `is_loopback_bind`. |
| `ts_export.rs` | The single ts-rs export path: `higgs_ts!` (wrap a type → export TS to `bindings/higgs/`) and `higgs_const_enum!` (const-object enum via `higgs_macros::TsConstEnum`). |
| `bin/higgs.rs` | Thin `main()`: detect the `--higgs-worker` re-exec role, route CLI subcommands (`keys`, `link`, `node`), install the tracing subscriber, hand off to `run_standalone`. |

---

## Public surface

Everything a host app touches is re-exported from the crate root (`src/lib.rs`):

```rust
pub use api::{Higgs, HiggsConfig};
pub use diagnostic::HiggsError;
pub use log_bus::{log_filter, HiggsLogLayer, LogBus};
pub use standalone::{run_standalone, shutdown_signal, StandaloneConfig};
pub use supervisor::HiggsEvent;
pub const DEFAULT_PORT: u16 = 31415;   // (pi) — override with HIGGS_PORT
```

`Higgs` is the facade the rest of the crate (and the embedder) drives:

| Method | Purpose |
|---|---|
| `new(cfg)` / `with_log_bus(cfg, bus)` | Construct (no worker spawned) |
| `start()` / `stop()` | Spawn the idle reaper (near-no-op otherwise) / tear down |
| `scan()` → `Vec<HiggsModel>` | Host-side model discovery (no worker needed) |
| `load(id, params)` / `unload()` / `unload_one(served)` | Load / unload (spawn-on-load, kill-on-unload) |
| `chat_stream(model, messages, max_tokens, temp, tools)` | Streaming chat → `(DeltaReceiver, JoinHandle<ChatOutcome>)` |
| `status()` → `HiggsStatus` | Live worker/loaded/on-disk status |
| `sysinfo()` / `hardware()` | Worker-gathered GPU devices / host hardware snapshot |
| `tune(req)` / `estimate(req)` | Analytical + measured autotune / footprint estimate |
| `model_readiness(id)` | Readiness state for the JIT prepare gate |
| `events()` / `subscribe_logs()` / `subscribe_load_events()` | Broadcast subscriptions |
| `set_fleet`/`fleet`, `set_hub`/`hub`, `set_api_keys`/`api_keys`/`mutate_api_keys` | Wire in the remote fleet, hub, and keystore |

The router is mounted with `higgs::serve::router(higgs.clone())` (see `src/serve/`). The
`Higgs` methods are engine-agnostic above the `HiggsEngine` trait (`src/worker/engine/`).

---

## Quick Start

```bash
cargo build --release

# Serve a folder of GGUF models (loopback, port 31415)
HIGGS_MODEL_DIR=/path/to/models ./target/release/higgs

curl localhost:31415/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "your-org/your-model",
  "messages": [{"role":"user","content":"Hello!"}],
  "stream": false
}'
```

Env knobs: `HIGGS_MODEL_DIR` (extra LM-Studio scan root), `HIGGS_BIND`, `HIGGS_PORT`,
`HIGGS_HOME` (state dir, default `~/.higgs`), `HIGGS_ENGINE` (default `llamacpp`),
`HIGGS_HUB=1` (run as a fleet hub), `HIGGS_HF_ENDPOINT` (HuggingFace mirror/proxy),
`HIGGS_VERBOSE=1`.

### Embed in an Axum app

```rust
use std::sync::Arc;
use higgs::{Higgs, HiggsConfig};

let higgs = Arc::new(Higgs::new(HiggsConfig::default()));
let app = axum::Router::new().merge(higgs::serve::router(higgs.clone()));
higgs.start().await?;   // spawns the idle reaper; no worker yet
```

---

## HTTP API

| Method & path | Purpose |
|---|---|
| `POST /v1/chat/completions` | OpenAI chat — stream + non-stream, tools, sampling params, `reasoning_content` |
| `GET /v1/models` | Loaded + remotely-routable models |
| `GET /api/higgs/status` · `/system` · `/nodes` | Live status · host hardware+runtime · remote fleet view |
| `GET /api/higgs/models` · `…/models/{*id}` | Scanned catalog · one model's details |
| `POST /api/higgs/models/load` · `…/unload` | Load / unload a model |
| `GET·PUT /api/higgs/settings` · `…/logs/settings` | Runtime toggles (JIT, idle reaper, serving) · logging toggles |
| `GET /api/higgs/logs` · `…/logs/stream` | Developer-Log snapshot / live SSE, `?source=` filter |
| `POST /api/higgs/worker/stop` | Stop the worker process |
| `GET /api/higgs/version` · `/health` | Build version · always-open health check |

See `src/serve/` for the full surface (fleet/hub admin, tune/estimate, keys).

---

## Securing the API

The HTTP surface is **open by default** (loopback / embedded use). Add API keys to turn on
auth — the gate turns on the moment the first key exists (`src/keys.rs`):

```bash
higgs keys add ci chat,models   # scopes: chat | models | admin (admin = superset)
higgs keys list                 # shows label + sha256 prefix (never the token)
higgs keys remove ci
```

Clients then send `Authorization: Bearer hgk_…`. Tokens are stored **only** as a SHA-256
digest in `~/.higgs/api_keys.json` and compared in constant time. The standalone server loads
keys at startup and **fails closed** if the file is present but unreadable; **restart** higgs
for CLI key changes to apply. Keys can also be minted/listed/revoked LIVE over the Admin-scoped
HTTP API (`GET·POST /api/higgs/keys`, `DELETE /api/higgs/keys/{label}`) — those apply on the
very next request, no restart. Health checks stay open.

**Fail-closed LAN guard:** binding beyond loopback (`HIGGS_BIND` ≠ `127.0.0.1`) with **zero**
keys is refused at startup (`HG058`), and so is binding with keys of which **none is
Admin-capable** (`HG069` — the key-management API would be locked out). Revoking the last key
while LAN-bound is likewise refused (`HG059`), as is revoking the last Admin-capable key while
other keys remain (`HG066`).

---

## Remote Fleet (peer-to-peer)

Run models on **other machines** over an encrypted [iroh](https://iroh.computer) QUIC link —
no central server, no open inbound ports. See `src/node/README.md` and `DESIGN-remote.md`.

```bash
HIGGS_HUB=1 higgs                                  # server in hub mode
curl -X POST localhost:31415/api/higgs/pair        # → { ticket, token, node_command }
HIGGS_MODEL_DIR=/models higgs --node <ticket> <token>   # pair a node
higgs --node                                       # reconnect to the saved default hub
higgs node leave                                   # node self-retires from its hub
```

Invariants: the hub **never** speaks to a remote worker directly (two hops: hub → node →
worker); control (`higgs/node/*`) and data (`M_CHAT`) are separate dispatchers; the worker
stays pure-sync stdio in every topology; pairing tokens are single-use and version-negotiated.

---

## Adding a New Engine

higgs talks to backends only through the `HiggsEngine` trait (`src/worker/engine/mod.rs`) and
selects one at startup from a registry. Adding an engine is: **implement the trait in a new
submodule, add one `REGISTRY` line, select with `HIGGS_ENGINE=<name>`.** Nothing in the worker
dispatch, supervisor, node runtime, or serve layer changes. See `src/worker/engine/DESIGN.md`.

---

## Diagnostics

Every failure is a typed `HiggsError` (`src/diagnostic.rs`) carrying a stable `HGxxx` code
(e.g. `HG002` model not found, `HG017` insufficient RAM, `HG022`–`HG048` remote/auth,
`HG057` stream overflow, `HG060` OOM ladder exhausted). Codes map to HTTP statuses at the serve
boundary and ride the JSON-RPC `error.data.code` so the true origin status survives the
hub→node hops. See `DESIGN.md` for the full registry.

---

## Testing & Coverage

```bash
scripts/quality.sh               # fast gate: fmt + clippy + test + ts-rs bindings sync
scripts/coverage.sh              # both coverage gates (unit ≥90%, integration ≥80%)
```

- **Unit tests** live in a sibling `<name>_tests.rs` wired as a child `#[path] mod tests`
  (never inline). `mod.rs` files are export barrels only.
- **Integration tests** live in `tests/` — they spawn the real `higgs` binary and drive it over
  HTTP + the in-process iroh gate against a tiny GGUF (`HIGGS_TEST_GGUF`; tests **skip** when
  absent, so set it before measuring coverage).
- ts-rs bindings regenerate on `cargo test`; the const-object enums need the second pass
  (`cargo test macro_run -- --ignored`) — both are wired into `scripts/quality.sh`.
</content>
</invoke>
