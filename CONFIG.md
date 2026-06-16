# higgs Configuration

## Table of Contents
- [Process & Environment](#process--environment)
- [HiggsConfig Fields](#higgsconfig-fields)
- [Effective Config (read-only surface)](#effective-config-read-only-surface)
- [Model Directory Layouts](#model-directory-layouts)
- [Build Note: LIBCLANG_PATH](#build-note-libclang_path)
- [Serve-Layer Limits & Hardening](#serve-layer-limits--hardening)
- [Non-Configurable Defaults](#non-configurable-defaults)
- [Links](#links)

---

## Process & Environment

**Production: higgs runs embedded in-process inside the jigglebot server.**
The `backend/server/src/higgs/` launcher constructs `Higgs::new(HiggsConfig::
default())`, binds an **ephemeral** `127.0.0.1:0` listener (the OS picks the
port), and serves `higgs::serve::router` on it. The resolved origin is stored
in `config.higgs_base_url` BEFORE the provider registry / `/api/meta` read it,
and seeded as a runtime-only provider — see
[server config](../server/src/config/CONFIG.md) and
[`backend/server/src/higgs/CONFIG.md`](../server/src/higgs/CONFIG.md). The
embedded path reads no `HIGGS_*` env vars and binds no fixed port (so it never
collides with a separately-running higgs/Ollama on 11434).

**Standalone / dev: the `higgs-server` binary** (`src/bin/higgs-server.rs`) runs
higgs as its own process with a fixed bind/port. It is NOT the production path.
It reads three environment variables; everything else comes from
`HiggsConfig::default()` (there is no config file at this layer).

| Env var | Default | Effect | Where |
|---------|---------|--------|-------|
| `HIGGS_BIND` | `127.0.0.1` | Bind address. A non-loopback value (`0.0.0.0`, a LAN IP) exposes the **no-auth** surface LAN-wide and logs a prominent `tracing::warn!` SECURITY WARNING at startup. | `higgs-server` only |
| `HIGGS_PORT` | `11434` | Listen port. | `higgs-server` only |
| `RUST_LOG` | `info` | tracing filter. | both |

```sh
higgs-server                                      # 127.0.0.1:11434 (standalone)
HIGGS_BIND=0.0.0.0 HIGGS_PORT=1234 higgs-server   # LAN-reachable on :1234
```

A host that embeds the crate directly (as jigglebot does) constructs its own
`HiggsConfig` and passes it to `Higgs::new`; the fields below are that struct's
defaults.

---

## HiggsConfig Fields

Defaults from `HiggsConfig::default()` — what the `higgs-server` binary uses.
An embedding host may override any field when it constructs `HiggsConfig`.

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `lmstudio_dirs` | `Vec<PathBuf>` | `[~/.lmstudio/models, ~/.cache/lm-studio/models]` | Both LM Studio < 0.3 and >= 0.3 paths are included by default; the host can narrow the list |
| `hf_dirs` | `Vec<PathBuf>` | `[~/.cache/huggingface/hub]` | HuggingFace hardcodes `~/.cache` on ALL platforms — it does not follow XDG or macOS conventions. higgs resolves this as `dirs::home_dir().join(".cache/huggingface/hub")`, NOT `dirs::cache_dir()` |
| `ollama_dirs` | `Vec<PathBuf>` | `[~/.ollama/models]` | |
| `default_load.ctx_len` | `u32` | `4096` | Context window tokens for new loads when the caller does not supply params |
| `default_load.gpu_layers` | `u32` | `u32::MAX` | `u32::MAX` means all layers offloaded (LM Studio "max" semantics) |
| `default_load.threads` | `u32` | `available_cpus - 2` (min 1) | Worker threads during generation; computed from `std::thread::available_parallelism()` |

---

## Effective Config (read-only surface)

The resolved config is surfaced **read-only** at `GET /api/higgs/system` as the
`config: HiggsServerConfig` field (alongside `hardware` and `runtime`). It is
built by `Higgs::server_config()` — a pure read, no worker RPC, no mutation;
there is **no endpoint to change config**.

| `HiggsServerConfig` field | Source |
|---------------------------|--------|
| `bind_host` | `BIND_HOST` const (`api.rs`) — always `"127.0.0.1"` |
| `lmstudio_dirs` / `hf_dirs` / `ollama_dirs` | the configured scan dirs as path strings |
| `default_load` | `HiggsConfig.default_load` (`ctx_len`, `gpu_layers`, `threads`) |
| `default_ctx_cap` | `DEFAULT_CTX_CAP` const (`api.rs`) = `32768` |

---

## Model Directory Layouts

higgs reads the following directory trees during a scan. **It never writes into any of these stores.**

```
~/.lmstudio/models/          (LM Studio < 0.3)
  org/
    model-name/
      *.gguf

~/.cache/lm-studio/models/   (LM Studio >= 0.3)
  org/
    model-name/
      *.gguf

~/.cache/huggingface/hub/    (HuggingFace Hub)
  models--org--model-name/
    snapshots/
      <commit-hash>/
        *.gguf

~/.ollama/models/            (Ollama)
  manifests/registry.ollama.ai/<name>/<tag>   (JSON manifest)
  blobs/sha256-<hash>                          (GGUF blob)
```

All roots are optional — a missing directory is silently skipped. An existing but unreadable root produces `[HG001] ModelDirUnreadable`.

---

## Build Note: LIBCLANG_PATH

The `llama-cpp-2` crate requires `libclang` at compile time (bindgen dependency). On machines where libclang is not on the default path, set:

```sh
export LIBCLANG_PATH=/path/to/libclang/lib
```

The project prefix `env -u LIBCLANG_PATH cargo …` unsets any stale value when the correct path is already on `PATH`. When building for the first time on a new machine, set `LIBCLANG_PATH` explicitly before running `cargo build`.

---

## Serve-Layer Limits & Hardening

The serve layer (`src/serve/mod.rs`) applies established ollama/vllm HTTP
hardening. These are documented `const`s today (not yet config); a later phase
lifts the user-facing ones into `HiggsConfig` + the Server Settings UI.

| Item | Value | Source const | Effect |
|------|-------|--------------|--------|
| Max request body | 32 MB | `serve::MAX_BODY_BYTES` | Oversized body ⇒ `413`. Caps `/v1/chat/completions` + control bodies (vllm uses ~4 MB; ours is larger for long transcripts). |
| Control timeout | 120 s | `serve::CONTROL_TIMEOUT` | Whole-request timeout on `/api/higgs/*` + `/v1/models` only. **Not** applied to `/v1/chat/completions` — a long SSE stream must never be aborted at the HTTP layer (its duration is bounded by the worker chat-RPC timeout). |
| Chat-RPC timeout | 600 s | `supervisor::CHAT_RPC_TIMEOUT` | Bounds a single chat/inference RPC round-trip — the layer that bounds streaming chat duration. A wedged-but-alive worker (no chunks, no final response) ⇒ `504 [HG016]`. Generous (long large-model generations); no reference fixes a chat-RPC ceiling (vllm/ollama bound per-request output via `max_tokens`). |
| Max output tokens | 32768 | `serve::MAX_OUTPUT_TOKENS` | `max_tokens`/`max_completion_tokens` above this ⇒ `400 [HG013]`. Matches `DEFAULT_CTX_CAP`; bounds per-request generation regardless of the loaded model. |
| Sampling validation | vllm ranges | `serve::v1::validate_sampling` | `temperature >= 0`, `top_p ∈ (0,1]`, `n >= 1`, `presence_penalty`/`frequency_penalty ∈ [-2,2]`, `max_tokens ∈ [1, MAX_OUTPUT_TOKENS]`. Out-of-range ⇒ `400 [HG013]` before dispatch. Ranges mirror vllm `SamplingParams._verify_args`. |
| Prompt fit pre-check | 4 bytes/token | `serve::PROMPT_BYTES_PER_TOKEN` | Conservative early reject: `prompt_bytes/4 + max_tokens > ctx_len` ⇒ `400 [HG005]`. Lower-bound estimate (serve layer has no tokenizer); the worker's exact `[HG005]` check is the authoritative backstop. |
| Inference admission gate | 8 | `api::MAX_CONCURRENT_INFERENCE` | At most 8 concurrent in-flight chat requests; a full gate ⇒ `503 [HG014]` (capacity signal, retryable). Scoped to the chat path only. Ollama's `OLLAMA_NUM_PARALLEL` default is 1, `OLLAMA_MAX_QUEUE` 512; 8 gives flood-proof headroom over the single-sequence worker. Distinct from the deferred worker-slot `max_concurrent_requests` in `concurrency.md`. |
| Repo-id charset guard | alnum + `_-./:` | `api::validate_repo_id` | Load id charset mirrors ollama `types/model/name.go`; a `..` component, absolute path, NUL, or illegal char ⇒ `400 [HG015]`. |
| Path-traversal guard | within scan roots | `api::path_within_roots` | The resolved GGUF path must canonicalize inside a configured scan dir, else `400 [HG015]` — a symlink/`..` escape never reaches the worker FFI loader. |
| Host guard | loopback only | `serve::is_loopback_host` | DNS-rebinding defense. `Host` header (sans `:port`) must be `127.0.0.1` / `localhost` / `::1`, else `403 [HG012]`. Missing `Host` ⇒ `403` (ollama behavior). |
| RAM headroom guard | 0.8 of available | `api::MEMORY_HEADROOM_FRACTION` | Pre-load: refuse a load whose GGUF file size exceeds `available_ram * 0.8` ⇒ `503 [HG017]` (capacity, retryable) — checked before spawning a worker so an oversized load fails fast instead of OOM-killing the worker. 0.8 is ollama's `freeMemory*80/100` placement rule (`server/sched.go`). |
| Idle auto-unload TTL | 5 min | `api::IDLE_UNLOAD_TTL` | Background reaper (spawned by `Higgs::start`) unloads the loaded model after 5 min with no inference, freeing memory. 5 min = ollama's `keep_alive` default (`envconfig/config.go`). Never unloads mid-generation (gated on a fully-open inference semaphore) or with nothing loaded. |
| Idle reaper interval | 30 s | `api::IDLE_REAP_INTERVAL` | How often the reaper checks idle time vs. the TTL; bounds post-TTL unload latency. No reference fixes a poll cadence (ollama uses a per-model timer); 30 s is the documented higgs value. |
| Log redaction | host paths / `host:port` | `serve::v1::redact_paths` | The client-facing `/v1` error envelope strips absolute filesystem paths and bind addresses (replaced with `<redacted>`). No prompt CONTENT is logged at `info` on the `/v1` path (only model id, stream flag, lengths/ids). Full unredacted Display (with paths) is still logged server-side at the origin and returned on the `/api/higgs/*` control surface (ours). |
| Panic recovery | — | `CatchPanicLayer` | A handler panic returns a structured `500` instead of dropping the connection (ollama gin Recovery). |
| CORS | loopback + tauri origins | `serve::local_cors` | Cross-origin only from loopback / Tauri webview origins. |

`GET /health` (and `/api/higgs/health`) is a cheap readiness probe: `200` as
soon as the server is up, with **no** worker RPC ("server reachable", not "model
loaded").

**No-auth threat model:** higgs has no authentication. The Host guard + CORS
protect *browser* clients from DNS rebinding / cross-origin reads. They do **not**
protect non-browser clients — which is why a non-loopback `HIGGS_BIND` logs a
SECURITY WARNING. The embedded (jigglebot) path always binds ephemeral loopback.

---

## Non-Configurable Defaults

| Item | Value | Source |
|------|-------|--------|
| stderr ring buffer cap | 2000 lines | `supervisor.rs` — hardcoded |
| event broadcast channel cap | 64 | `supervisor.rs` — hardcoded |
| respawn backoff | 1 second | `supervisor.rs` — hardcoded |
| respawn attempts per death | 1 | supervisor restarts once; factory failure is terminal |
| graceful stop timeout | 2 seconds | `Higgs::stop()` — hardcoded |
| `/api/higgs/logs` default tail | 200 lines | `LogsQuery.n` default |

---

## Links

- Root configuration reference: [CONFIG.md](../../CONFIG.md)
- Full field reference (gateway / metrics / agent): [backend/server/src/config/CONFIG.md](../server/src/config/CONFIG.md)
