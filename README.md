# higgs

**A self-contained local LLM runtime in Rust.** Point it at a folder of GGUF files and it
serves OpenAI-compatible inference over a strict `/v1` HTTP surface — driven by an **in-process
Rust control API** — with crash-isolated worker processes, a pluggable engine layer, an optional
peer-to-peer **remote fleet** over encrypted QUIC, and opt-in API-key auth.

higgs is **library-first**. An embedder (jigglebot) constructs a `Higgs` (an `Arc`) and drives
everything — load, chat, status, tune, keys, fleet — through the crate API; the only thing served
over a socket is OpenAI `/v1` (chat + models), via `serve::serve_v1`. The `higgs` **binary** is a
**node-only** daemon (it serves no HTTP of its own).

higgs is a **standalone crate**: it imports nothing from any jigglebot crate (`src/lib.rs`). The
control plane (the `Higgs` facade + the `/v1` router) is pure Rust and cannot crash; only the
worker process links the llama.cpp FFI.

```
┌──────────────────────────────────────────────────────────────────────┐
│  external OpenAI client ─HTTP→  /v1/chat/completions   /v1/models      │
│  embedder (jigglebot)   ─Rust→  Higgs facade methods (in-process)      │
└───────────────────────────────┬──────────────────────────────────────┘
                                 │  (pure-Rust facade + /v1 router, can't crash)
                          ┌──────▼───────┐
                          │    Higgs     │  facade: load / chat / status / logs / tune / keys
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
| `src/api/` | `Higgs` facade internals split out of `api.rs`: `embed.rs` (the **in-process control-plane methods** the embedder calls instead of HTTP), `guards.rs` (memory/repo-id/path guards), `types.rs` (`HiggsConfig`, `HiggsServerConfig`, `ModelLoadPhase`/`ModelLoadEvent`, wire status/outcome types) | `src/api/README.md` |
| `src/node/` | iroh remote fleet: hub, node runtime, transport, fleet routing, identity, CLI | `src/node/README.md`, `DESIGN-remote.md` |
| `src/serve/` | The Axum **`/v1`-only** router (`v1_router`/`serve_v1`: chat + models + `/health`), SSE assembly, hardening layers, and `control.rs` — the PURE control helpers the facade delegates to | `src/serve/README.md` |
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
| `lib.rs` | Crate root — module map; re-exports `Higgs`/`HiggsConfig`, the embed I/O + event types (`ChatOutcome`, `HiggsStatus`, `LoadedInfo`, `ModelLoadEvent`, `ModelLoadPhase`, `PairInfo`, `PreparedChat`), `HiggsError`, `Scope`, `LogBus`/`HiggsLogLayer`/`log_filter`, the `serve::wire` control shapes, `HiggsEvent`, the tune + engine types, `shutdown_signal`; `DEFAULT_PORT` (31415 — a documented default for external clients; the node-only binary binds no port); the crate-internal `LLAMA_CPP_2_VERSION` (`"0.1.151"`, the bundled binding). `#[macro_use] mod ts_export` first so `higgs_ts!` is in scope everywhere. |
| `api.rs` | The `Higgs` public **facade**. Typed wrapper over a co-located LOCAL `NodeRuntime` (the same multi-worker engine remote nodes run). Owns facade state — live `HiggsConfig`, the load-lifecycle mutex, the two inference admission gates, serve-layer toggles, the readiness/JIT gate, tune/estimate, the OOM retry wiring — and delegates spawn/load/unload/status/chat to the node. Its control-plane methods (`model_entries`, `mint_key`, `hub_enable`, `pair`, …) live in the sibling `api/embed.rs`. |
| `supervisor.rs` | Manages ONE worker child: spawn by re-exec (`--higgs-worker`), NDJSON JSON-RPC 2.0 over stdio, correlate responses by id, route chat-chunk notifications, restart-once + replay-last-load on unexpected death. Owns `HiggsEvent`, the RPC timeouts (`CONTROL` 120 s, `CHAT_RPC_TIMEOUT` 600 s, `SYSINFO` 15 s). |
| `actor.rs` | The generic actor runtime (`Actor` trait, `Handle`/`WeakHandle`, `spawn_actor{,_with}`) written once, plus `ReplyDemux` (request-id → response waiter + `request_id` → chat sink). Reused by `Supervisor`, `NodeRuntime`, and the per-node transport. |
| `delta_queue.rs` | **(G1)** Bounded, MERGING chat-delta channel — the backpressure buffer between a delta producer and its single streaming consumer. Coalesces consecutive same-kind deltas in place (entries bounded by kind-alternations, not token count); `CAP_BYTES` (8 MB) hard cap trips `HG057` on a stalled client instead of unbounded growth. `delta_channel()` → `(DeltaSender, DeltaReceiver)`. |
| `rpc.rs` | NDJSON JSON-RPC 2.0 codec — the supervisor↔worker AND hub↔node wire. `RpcRequest`/`RpcResponse`/`RpcError`/`RpcNotification`/`RpcFrame`, `encode`/`decode` (`HG008` on bad frame), `method_not_found` (`HG037`). Soft-validates the `jsonrpc` version (warn, not reject). |
| `diagnostic.rs` | `HiggsError` — every higgs failure, each carrying a stable `HGxxx` code baked into `Display`. Append-only; never renumber. Also documents the **log-only** codes (HG051–HG056, HG061/HG062/HG065) that ride a `tracing` line, not a variant. |
| `system.rs` | Host hardware + inference-runtime snapshot for `Higgs::server_config()` / the system view: `SystemInfo`, `HardwareInfo`, `RuntimeInfo`, `GpuDevice`, `DeviceKind`, plus `fits_vram`/`FitAssessment` and `HardwareInfo::{fingerprint, free_vram_bytes, is_unified_memory}`. `hostname()`. |
| `log_bus.rs` | The single home for Developer-Log lines: `LogBus` (a bounded history ring PER `LogSource` + a live broadcast tap), `HiggsLogLayer` (tracing layer mirroring `higgs`-target events), `log_filter`. `LogSource` = `Serve`/`Worker`/`LocalWorker`/`RemoteWorker`. Read via `Higgs::logs()`; streamed via `Higgs::subscribe_logs()`. |
| `keys.rs` | **(G4)** API-key auth: `ApiKeys`/`ApiKey`/`Scope` (`Chat`/`Models`/`Admin`). Tokens stored only as a SHA-256 hex digest; constant-time compare; empty store ⇒ auth OFF (open). `hash_token`/`mint_token`; `higgs keys <add\|list\|remove>` CLI (`run_keys`); live mutation via the crate API (`Higgs::mint_key`/`revoke_key`/`mutate_api_keys`, Admin scope). |
| `auth.rs` | Remote-fleet **allowlist** (`pairings.json` → `Allowlist`) and one-time **pairing tokens** (`PairingTokens`, mint/validate/burn). Atomic saves; corruption → `HG041`, I/O → `HG040`. |
| `config.rs` | `~/.higgs/config.json` (`InstanceConfig`): friendly `name`, node-side saved hubs (`SavedHub` + `remember_hub`/`find_hub`/`remove_hub`), per-model load records (`ModelRecord`), extra `cors_origins`. `Role`, `friendly_name`, `name_or_init`. |
| `home.rs` | `~/.higgs` home dir (`higgs_home`, `ensure_home`); `HIGGS_HOME` override. The single home for all identity/state files. |
| `hub.rs` | **HuggingFace Hub client — PRIMARY fetch path.** `HubFetcher` (streaming) + `fetch_bytes` (in-memory card/config), honoring `HIGGS_HF_ENDPOINT`. `classify_hf`/`http_status_to_error` map failures to distinct `HG029`–`HG035`. |
| `download.rs` | Model downloader (`M_PULL`): `Fetcher` trait, `HttpFetcher` (reqwest FALLBACK), `download`/`download_dual` (primary+fallback → `HG036` if both fail), `dest_path` path-traversal guard, atomic `.part`+rename into `~/.higgs/models/`. |
| `load_robustness.rs` | **(G5)** Pure OOM decision logic wired into the load path: `is_oom_reason`, `oom_ladder` (settle → KV-to-RAM → fewer GPU layers → `HG060`), `LoadRung`, `SETTLE_BEFORE_RETRY` (750 ms VRAM settle). |
| `remote.rs` | Remote wire vocabulary: `ALPN`, `M_HELLO`, the `higgs/node/*` methods, `HelloParams`/`HelloResult`, `NodeLoadParams`/`NodeChatParams`/`NodeInventory`/`InventoryWorker`, capabilities, `negotiate_version`, `PAIRING_TOKEN_TTL_MS`. |
| `shutdown.rs` | `shutdown_signal()` — the Ctrl-C / SIGTERM future an embedder passes to `serve_v1` for graceful drain. |
| `ts_export.rs` | The single ts-rs export path: `higgs_ts!` (wrap a type → export TS to `bindings/higgs/`) and `higgs_const_enum!` (const-object enum via `higgs_macros::TsConstEnum`). |
| `bin/higgs.rs` | Thin `main()`: detect the `--higgs-worker` re-exec role, install the tracing subscriber, then dispatch the **node-only** subcommands (`--node`, `node`, `link`, `keys`) + `--version`. There is **no** standalone HTTP server — a bare invocation prints a usage note and exits non-zero. |

---

## Public surface

Everything a host app touches is re-exported from the crate root (`src/lib.rs`):

```rust
pub use api::{Higgs, HiggsConfig};
pub use api::{ChatOutcome, HiggsStatus, LoadedInfo, ModelLoadEvent, ModelLoadPhase, PairInfo, PreparedChat};
pub use diagnostic::HiggsError;
pub use keys::Scope;
pub use log_bus::{log_filter, HiggsLogLayer, LogBus, LogLine, LogSource};
pub use serve::wire::{HiggsHubStatus, HiggsModelEntry, HiggsRuntimeSettings, HiggsVersionResponse, LogSettings, /* … */};
pub use shutdown::shutdown_signal;
pub use supervisor::HiggsEvent;
pub use tune::{EstimateReport, EstimateRequest, TuneRequest, TuneSuggestion};
pub use worker::engine::{ChatDelta, ChatDeltaKind, LoadParams, SamplingParams};
pub const DEFAULT_PORT: u16 = 31415;   // (pi) — a documented default for external clients
```

`Higgs` is the facade the rest of the crate (and the embedder) drives. Control is these methods —
**not** an HTTP route:

| Method | Purpose |
|---|---|
| `new(cfg)` / `with_log_bus(cfg, bus)` | Construct (no worker spawned) |
| `start()` / `stop()` | Spawn the idle reaper (near-no-op otherwise) / tear down |
| `scan()` → `Vec<HiggsModel>` | Host-side model discovery (no worker needed) |
| `load(id, params)` / `unload()` / `unload_one(served)` | Load / unload (spawn-on-load, kill-on-unload) |
| `chat_stream(model, messages_json, max_tokens, sampling, tools_json, chat_template_kwargs)` | Streaming chat → `(DeltaReceiver, JoinHandle<Result<ChatOutcome, HiggsError>>)` |
| `status()` → `HiggsStatus` | Live worker/loaded/on-disk status |
| `model_entries()` / `model_by_id(id)` | Per-model control rows (load state / format / tool-call verdict / last-load params) |
| `sysinfo()` / `hardware()` / `server_config()` | Worker-gathered GPU devices / host hardware snapshot / effective read-only config |
| `tune(req)` / `estimate(req)` | Analytical + measured autotune / footprint estimate |
| `subscribe_load_events()` / `subscribe_logs()` / `events()` | Broadcast subscriptions (load phases / dev-log lines / worker events) |
| `mint_key` / `revoke_key` / `api_keys` / `mutate_api_keys` | Live API-key management (Admin scope) |
| `set_fleet`/`fleet`, `set_hub`/`hub`, `hub_enable`/`hub_disable`, `pair`, `nodes()` | Wire in / drive the remote fleet + hub |
| `version()` | Build version |

Serve the OpenAI `/v1` surface with `higgs::serve::serve_v1(higgs, listener, shutdown)` (or build
the router with `higgs::serve::v1_router(higgs.clone())` and merge it into your own app). The
`Higgs` methods are engine-agnostic above the `HiggsEngine` trait (`src/worker/engine/`).

---

## Quick Start

### Install a signed release (no clone)

Releases ship as minisign-signed tarballs (macOS arm64 `metal`, Linux x86_64 `cpu`/`cuda` —
see [RELEASING.md](RELEASING.md)). The repo is private, so downloads authenticate with a
**fine-grained PAT** (GitHub → Settings → Developer settings → Fine-grained tokens; scope it
to this repo, permission **Contents: Read**). One token, two commands, no clone:

```bash
export TOKEN=github_pat_…   # your fine-grained PAT (Contents: Read)

# 1) fetch install.sh — one file via the contents API, not a checkout:
curl -fsSL -H "Authorization: Bearer $TOKEN" -H "Accept: application/vnd.github.raw" \
  https://api.github.com/repos/AI-Experiri/higgs/contents/install.sh -o install.sh
chmod +x install.sh

# 2) install a release — the same token downloads the signed tarball; --pubkey
#    verifies its minisign signature against the repo's pinned release key:
HIGGS_GITHUB_TOKEN=$TOKEN ./install.sh --version 0.1.0-beta.1 \
  --pubkey RWQF20Rd+9jIND7YdCJsl3btwZKBAvF1EzRyHWiR8MH7C0JHhRSRdzSC
```

Run it **as the operator, never sudo**, and as `./install.sh` (not `bash install.sh` — the
`#!/bin/bash -p` hardening only applies when it runs as a file). Pre-releases (any version
with a `-`) are skipped by the default "latest", so pin `--version` for a beta. On Linux add
`--cuda` for the CUDA build. It lands in `~/.higgs/bin/v<ver>/`, flips `~/.higgs/bin/current`,
and prints the `install-service` command to run it as a login-bound service.

After the first install the token is no longer needed: nodes update themselves
(`higgs node self-update`, one-step `--rollback`) or are pushed to from a hub — see
[RELEASING.md](RELEASING.md) Part D. `uninstall.sh` (fetched the same way) tears the service
down cleanly, keeping `~/.higgs` unless `--purge`.

### Build from source

```bash
cargo build --release
```

The `higgs` **binary is node-only** — it serves no HTTP itself. To serve `/v1`, **embed the
crate** (below). To run a worker node in a fleet:

```bash
# Join a hub as a persistent worker node
HIGGS_MODEL_DIR=/path/to/models ./target/release/higgs --node <ticket> <token>
```

A crate host that binds `/v1` on port 31415 then answers:

```bash
curl localhost:31415/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "your-org/your-model",
  "messages": [{"role":"user","content":"Hello!"}],
  "stream": false
}'
```

Env knobs read by the binary/worker: `HIGGS_MODEL_DIR` (extra LM-Studio scan root, honored by
`--node`), `HIGGS_VERBOSE=1` (keep the full llama.cpp per-load dump on a node worker),
`HIGGS_HOME` (state dir, default `~/.higgs`), `HIGGS_ENGINE` (worker engine, default `llamacpp`),
`HIGGS_HF_ENDPOINT` (HuggingFace mirror/proxy), `HIGGS_IROH_LOCAL` (dev/local iroh transport),
`RUST_LOG` (tracing filter). There is **no** `HIGGS_BIND`/`HIGGS_PORT` — the binary binds no
socket; an embedder chooses the listener it hands to `serve_v1`.

### Embed in an Axum app

```rust
use std::sync::Arc;
use higgs::{Higgs, HiggsConfig};

let higgs = Arc::new(Higgs::new(HiggsConfig::default()));
higgs.start().await?;                       // spawns the idle reaper; no worker yet

// Serve /v1 on a listener you own (serve_v1 runs the HG058/HG069 LAN-bind key guards
// on the REAL bound address, then serves gracefully until `shutdown` resolves):
let listener = tokio::net::TcpListener::bind(("127.0.0.1", higgs::DEFAULT_PORT)).await?;
higgs::serve::serve_v1(higgs.clone(), listener, higgs::shutdown_signal()).await?;

// …or merge the router into your own app and drive control via the crate API:
let app = axum::Router::new().merge(higgs::serve::v1_router(higgs.clone()));
```

---

## HTTP API

The **entire** socket surface is `/v1` plus a health probe:

| Method & path | Purpose |
|---|---|
| `POST /v1/chat/completions` | OpenAI chat — stream + non-stream, tools, sampling params, `reasoning_content` |
| `GET /v1/models` | Loaded + remotely-routable models |
| `GET /health` | Always-open readiness probe (no worker RPC — "server reachable", not "model loaded") |

Everything else — status, model catalog, load/unload, tune/estimate, logs, settings, keys,
fleet/hub — is the **in-process `Higgs` facade** (see the method table above), not an HTTP route.

---

## Securing the API

The `/v1` surface is **open by default** (loopback / embedded use). Add API keys to turn on
auth — the gate turns on the moment the first key exists (`src/keys.rs`):

```bash
higgs keys add ci chat,models   # scopes: chat | models | admin (admin = superset)
higgs keys list                 # shows label + sha256 prefix (never the token)
higgs keys remove ci
```

Clients then send `Authorization: Bearer hgk_…`. Tokens are stored **only** as a SHA-256 digest
in `~/.higgs/api_keys.json` and compared in constant time. The `higgs keys` CLI edits the file
offline (**restart** the served instance for CLI changes to apply, since keys are snapshotted at
serve time). An embedder can also mint/revoke keys **live** via the crate API (`Higgs::mint_key` /
`revoke_key` / `mutate_api_keys`) — those apply on the very next request, no restart. Health checks
stay open.

**Fail-closed LAN guard (enforced in `serve_v1` on the real bound address):** serving a
non-loopback listener with **zero** keys is refused (`HG058`), and so is serving one whose keys
are all **non-Admin** (`HG069` — the key-management API would be locked out). Revoking the last
key while LAN-exposed is likewise refused (`HG059`), as is revoking the last Admin-capable key
while other keys remain (`HG066`).

---

## Remote Fleet (peer-to-peer)

Run models on **other machines** over an encrypted [iroh](https://iroh.computer) QUIC link —
no central server, no open inbound ports. See `src/node/README.md` and `DESIGN-remote.md`.

```bash
# Hub side: mint a one-time token, print the pairing ticket, accept dials until Ctrl-C
higgs link pair
higgs link status                                       # hub id + allowlist

# Node side
HIGGS_MODEL_DIR=/models higgs --node <ticket> <token>   # pair + join as a worker
higgs --node                                            # reconnect to the saved default hub
higgs node leave                                        # node self-retires from its hub
```

An embedder can instead run the hub **in-process** via the facade (`Higgs::hub_enable` mints the
endpoint; `Higgs::pair` returns a `{ ticket, token }` `PairInfo`). Invariants: the hub **never**
speaks to a remote worker directly (two hops: hub → node → worker); control (`higgs/node/*`) and
data (`M_CHAT`) are separate dispatchers; the worker stays pure-sync stdio in every topology;
pairing tokens are single-use and version-negotiated.

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
`HG057` stream overflow, `HG060` OOM ladder exhausted). Codes map to HTTP statuses at the `/v1`
boundary and ride the JSON-RPC `error.data.code` so the true origin status survives the hub→node
hops. See `DESIGN.md` for the full registry.

---

## Testing & Coverage

```bash
scripts/quality.sh               # fast gate: fmt + clippy + test + ts-rs bindings sync
scripts/coverage.sh              # both coverage gates (unit ≥90%, integration ≥80%)
```

- **Unit tests** live in a sibling `<name>_tests.rs` wired as a child `#[path] mod tests`
  (never inline). `mod.rs` files are export barrels only.
- **Integration tests** live in `tests/` — they spawn the real `higgs` binary and drive it over
  the `/v1` HTTP surface + the in-process crate/iroh gate against a tiny GGUF (`HIGGS_TEST_GGUF`;
  tests **skip** when absent, so set it before measuring coverage).
- ts-rs bindings regenerate on `cargo test`; the const-object enums need the second pass
  (`cargo test macro_run -- --ignored`) — both are wired into `scripts/quality.sh`.
