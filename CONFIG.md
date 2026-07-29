# higgs Configuration

## Table of Contents
- [Process & Environment](#process--environment)
- [HiggsConfig Fields](#higgsconfig-fields)
- [Per-load overrides (LoadParams)](#per-load-overrides-loadparams)
- [Effective Config (read-only surface)](#effective-config-read-only-surface)
- [Model Directory Layouts](#model-directory-layouts)
- [Build Note: LIBCLANG_PATH](#build-note-libclang_path)
- [Serve-Layer Limits & Hardening](#serve-layer-limits--hardening)
- [Runtime toggles & Non-Configurable Defaults](#runtime-toggles--non-configurable-defaults)
- [Links](#links)

---

## Process & Environment

higgs is **library-first**. There are two ways it runs:

**Embedded (production).** A host app (jigglebot) constructs `Higgs::new(HiggsConfig::default())`
(an `Arc`), calls `higgs.start().await`, binds a **loopback** `TcpListener`, and serves the strict
OpenAI `/v1` surface on it with `higgs::serve::serve_v1(higgs, listener, shutdown)`. The embedder
drives ALL control — load, chat, status, tune, keys, fleet — through the in-process `Higgs` facade
(no HTTP control surface). The embedder chooses the bind address and port itself; higgs reads no
`HIGGS_BIND`/`HIGGS_PORT`.

**Node daemon.** The `higgs` **binary** (`src/bin/higgs.rs`) is **node-only** — it serves no HTTP.
It runs the `--higgs-worker` re-exec role and the fleet subcommands (`--node`, `node`, `link`,
`keys`). A node worker scans models and relays its output to a hub over iroh.

Environment variables actually read by the binary/worker (grep-verified):

| Env var | Default | Effect | Read in |
|---------|---------|--------|---------|
| `HIGGS_HOME` | `~/.higgs` | Home dir for all identity/state files (`endpoint.key`, `pairings.json`, `api_keys.json`, `config.json`, `models/`, per-node `models.json`). | `home.rs` |
| `HIGGS_MODEL_DIR` | — | Extra model scan root in LM-Studio layout (`<dir>/{org}/{model}/*.gguf`), honored by `--node`. | `node/cli.rs` |
| `HIGGS_ENGINE` | first registry entry (`llamacpp`) | Selects the worker engine implementation. | `worker/mod.rs` |
| `HIGGS_HF_ENDPOINT` | HuggingFace | HuggingFace mirror / enterprise proxy / test server base URL (primary + fallback fetch). | `hub.rs`, `download.rs` |
| `HIGGS_VERBOSE` | `0` | `1`/`true` keeps the full llama.cpp per-load dump on a node worker. | `node/cli.rs` |
| `HIGGS_WORKER_VERBOSE` | `0` | `1` raises the worker's own llama.cpp log verbosity. | `worker/engine/llamacpp/logging.rs` |
| `HIGGS_IROH_LOCAL` | unset | Set ⇒ iroh uses a local/dev discovery+relay path (test/LAN); unset ⇒ the normal n0 discovery. | `node/hub.rs`, `node/identity.rs` |
| `RUST_LOG` | `info` | tracing filter for the terminal `fmt` layer. | `bin/higgs.rs` |

```sh
# Node daemon (serves no HTTP itself):
HIGGS_MODEL_DIR=/models higgs --node <ticket> <token>
```

There is **no config file at the process layer** — an embedding host constructs its own
`HiggsConfig` (fields below) and passes it to `Higgs::new`. `~/.higgs/config.json` holds only
per-instance identity/fleet state (friendly name, saved hubs, per-model load records, extra CORS
origins — see `config.rs`), not the `HiggsConfig` defaults.

---

## HiggsConfig Fields

Defaults from `HiggsConfig::default()` (`src/api/types.rs`). An embedding host may override any
field when it constructs `HiggsConfig`.

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `lmstudio_dirs` | `Vec<PathBuf>` | `[~/.lmstudio/models, ~/.cache/lm-studio/models]` | Both LM Studio < 0.3 and ≥ 0.3 paths; the host can narrow the list |
| `hf_dirs` | `Vec<PathBuf>` | `[~/.cache/huggingface/hub]` | HuggingFace hardcodes `~/.cache` on ALL platforms — resolved as `dirs::home_dir().join(".cache/huggingface/hub")`, NOT `dirs::cache_dir()` |
| `ollama_dirs` | `Vec<PathBuf>` | `[~/.ollama/models]` | |
| `default_load` | `LoadParams` | `LoadParams::base(CtxLen::Fixed { n: 4096 }, GpuLayers::All, threads)` | The engine-tagged load-params umbrella applied when a load request omits params (see below) |
| `worker_exe` | `Option<PathBuf>` | `None` | The executable that hosts the `--higgs-worker` role. `None` ⇒ `std::env::current_exe()` (correct for the `higgs` binary and for an embedder whose own binary answers `--higgs-worker`). `#[serde(skip)]` + `#[ts(skip)]` — a runtime/embedder concern, off the wire |

`default_load` base fields (always present; the quick-load / `default_load` / suggester path fills
them):

| Base field | Type | Default | Notes |
|-------|------|---------|-------|
| `ctx_len` | `CtxLen` | `Fixed { n: 4096 }` | Context window; `CtxLen` models the intent (`Fixed`/auto) rather than a magic-int sentinel |
| `gpu_layers` | `GpuLayers` | `All` | `GpuLayers::All` offloads every layer (the old `u32::MAX` sentinel, now typed); `GpuLayers::Count { n }` pins an explicit count |
| `threads` | `u32` | `available_cpus - 2` (min 1) | Worker generation threads, from `std::thread::available_parallelism()` |

---

## Per-load overrides (LoadParams)

`LoadParams` is an **engine-tagged umbrella**; for the llama.cpp engine the payload is
`LlamaCppParams` (`src/worker/engine/llamacpp/params.rs`, mapping to `llama-cpp-2`
`0.1.151` — the bundled binding, `LLAMA_CPP_2_VERSION`). Beyond the three base fields, a load
request may pin a broad set of **`Option`** knobs: **absent = engine default = pre-expansion
behavior**. Each is applied only in `worker/engine/llamacpp.rs`.

The knobs group as follows (the authoritative per-field reference — types, ranges, and the ⓘ
tooltip text — is the `#[help]`-annotated `LlamaCppParams` struct and the generated `PARAM_HELP`):

- **Model / placement:** `use_mmap`, `use_mlock`, `cpu_moe`, `split_mode`, `main_gpu`, `devices`.
- **Context / batch:** `n_batch` (default `ctx_len.max(1)`), `n_ubatch`, `n_seq_max`,
  `n_threads_batch`, `offload_kqv`, `swa_full`, `flash_attn` (`auto`/`off`/`on`).
- **KV cache:** `type_k`, `type_v` (`KvCacheKind`: F32/F16/Q8_0/Q5_1/Q5_0/Q4_1/Q4_0; default F16).
- **RoPE:** `rope_scaling_type`, `rope_freq_base`, `rope_freq_scale` (default = GGUF trained value).
- **Sampler chain:** `temperature`, `dynatemp_range`/`dynatemp_exponent`, `top_k`, `top_p`,
  `min_p`, `typical_p`, `top_n_sigma`, `xtc_probability`/`xtc_threshold`, the penalty knobs
  (`penalty_last_n`/`penalty_repeat`/`penalty_freq`/`penalty_present`), `dry`, `mirostat`,
  `grammar`, and `seed` (default = a fresh random seed per request).

See `src/worker/engine/llamacpp/` (README + DESIGN) for the exhaustive field list and each knob's
`llama-cpp-2` setter.

---

## Effective Config (read-only surface)

The resolved config is surfaced **read-only** by the `Higgs::server_config()` facade method
(`src/api.rs`) as a `HiggsServerConfig` (`src/api/types.rs`) — a pure read, no worker RPC, no
mutation. There is no endpoint (and no facade method) to change it.

| `HiggsServerConfig` field | Source |
|---------------------------|--------|
| `bind_host` | `BIND_HOST` const (`api/types.rs`) — always `"127.0.0.1"` |
| `lmstudio_dirs` / `hf_dirs` / `ollama_dirs` | the configured scan dirs, as path strings |
| `default_load` | `HiggsConfig.default_load` |
| `default_ctx_cap` | `DEFAULT_CTX_CAP` const (`api/types.rs`) = `32768` |
| `limits` | `HiggsLimits` — a read-only disclosure of the serve-layer hardening consts (body limit, control timeout, chat timeout, max output tokens, concurrency, RAM headroom, effective idle-TTL) |

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

All roots are optional — a missing directory is silently skipped. An existing but unreadable root
produces `[HG001] ModelDirUnreadable`. Models pulled via `M_PULL` land ONLY in `~/.higgs/models/`
(`download.rs`) — never a scanned store.

---

## Build Note: LIBCLANG_PATH

The `llama-cpp-2` crate requires `libclang` at compile time (bindgen dependency). On machines where
libclang is not on the default path, set:

```sh
export LIBCLANG_PATH=/path/to/libclang/lib
```

The project prefix `env -u LIBCLANG_PATH cargo …` unsets any stale value when the correct path is
already on `PATH`. When building for the first time on a new machine, set `LIBCLANG_PATH` explicitly
before running `cargo build`.

---

## Serve-Layer Limits & Hardening

The `/v1` serve layer (`src/serve/mod.rs`) applies established ollama/vllm HTTP hardening. These are
documented `const`s today (not yet config); a later phase lifts the user-facing ones into
`HiggsConfig`. The layer stack (outer runs first): `local_cors` → `host_guard` → `auth_guard` →
`CatchPanicLayer` → `DefaultBodyLimit`, then per-route split.

| Item | Value | Source const/fn | Effect |
|------|-------|--------------|--------|
| Max request body | 32 MB | `serve::MAX_BODY_BYTES` | Oversized body ⇒ `413`. Caps `/v1/chat/completions` (long transcripts) — vllm uses ~4 MB; ours is larger. |
| Control timeout | 120 s | `serve::CONTROL_TIMEOUT` | Whole-request timeout on **`GET /v1/models` only**. **Not** applied to `POST /v1/chat/completions` — a long SSE stream must never be aborted at the HTTP layer (bounded instead by the worker chat-RPC timeout). |
| Chat-RPC timeout | 600 s | `supervisor::CHAT_RPC_TIMEOUT` | Bounds a single chat/inference RPC round-trip. A wedged-but-alive worker (no chunks, no final response) ⇒ `504 [HG016]`. |
| Max output tokens | 32768 | `serve::MAX_OUTPUT_TOKENS` | `max_tokens`/`max_completion_tokens` above this ⇒ `400 [HG013]`. Matches `DEFAULT_CTX_CAP`. |
| Sampling validation | vllm ranges | `serve::v1::validate_sampling` | `temperature >= 0`, `top_p ∈ (0,1]`, `n >= 1`, `presence_penalty`/`frequency_penalty ∈ [-2,2]`, `max_tokens ∈ [1, MAX_OUTPUT_TOKENS]`. Out-of-range ⇒ `400 [HG013]` before dispatch. |
| Prompt fit pre-check | 4 bytes/token | `serve::PROMPT_BYTES_PER_TOKEN` | Conservative early reject: `prompt_bytes/4 + max_tokens > ctx_len` ⇒ `400 [HG005]`. Lower-bound estimate (serve layer has no tokenizer); the worker's exact `[HG005]` check is the authoritative backstop. |
| Inference admission gate | 8 | `api::MAX_CONCURRENT_INFERENCE` | At most 8 concurrent in-flight chat requests; a full gate ⇒ `503 [HG014]` (retryable). Scoped to the chat path. (A separate `remote_gate` of the same size covers fleet-routed chat.) |
| Repo-id charset guard | alnum + `_-./:` | `api::guards::validate_repo_id` | Load id charset mirrors ollama `types/model/name.go`; a `..` component, absolute path, NUL, or illegal char ⇒ `400 [HG015]`. |
| Path-traversal guard | within scan roots | `api::guards::path_within_roots` | The resolved GGUF path must canonicalize inside a configured scan dir, else `400 [HG015]` — a symlink/`..` escape never reaches the worker FFI loader. |
| Host guard | loopback only | `serve::is_loopback_host` | DNS-rebinding defense. `Host` header (sans `:port`) must be `127.0.0.1` / `localhost` / `::1`, else `403 [HG012]`; missing `Host` ⇒ `403`. **Relaxed** on a keyed non-loopback bind (LAN clients send their server's own LAN host). |
| RAM headroom guard | 0.8 of available | `api::MEMORY_HEADROOM_FRACTION` | Pre-load: refuse a load whose GGUF size exceeds `available_ram * 0.8` ⇒ `503 [HG017]` — checked before spawning a worker. 0.8 is ollama's `freeMemory*80/100` placement rule. |
| Log redaction | host paths / `host:port` | `serve::v1::redact_paths` | The client-facing `/v1` error envelope strips absolute filesystem paths and bind addresses. No prompt CONTENT is logged at `info` on `/v1` (only model id, stream flag, lengths/ids). The in-process crate control methods return the full unredacted `HiggsError` Display to the embedder. |
| Panic recovery | — | `CatchPanicLayer` | A handler panic returns a structured `500` instead of dropping the connection. |
| CORS | loopback + tauri + configured | `serve::local_cors` | Cross-origin only from loopback / Tauri webview origins plus any `config.json` `cors_origins`. |
| API-key auth | opt-in | `serve::auth_guard` + `keys.rs` | When keys are configured, the route's required scope must be met by a `Bearer hgk_…`, else `401 [HG048]`. Empty keystore ⇒ auth OFF (loopback-only, guaranteed by the `[HG058]` LAN refusal). |
| Keyless-LAN refusal | — | `serve::serve_v1` | A non-loopback listener with zero keys ⇒ refuse to serve (`[HG058]`); with no Admin-capable key ⇒ refuse (`[HG069]`). Runs on the REAL bound address. |

`GET /health` is a cheap readiness probe: `200` as soon as the server is up, with **no** worker RPC
("server reachable", not "model loaded").

**Threat model:** the `/v1` surface is open by default and safe only because a keyless bind is
loopback-only — the Host guard + CORS protect *browser* clients from DNS-rebinding / cross-origin
reads, and `serve_v1` refuses a keyless non-loopback listener (`[HG058]`). To expose higgs on a LAN,
mint API keys first (at least one Admin-capable); the bind is then bearer-gated on every data route.

---

## Runtime toggles & Non-Configurable Defaults

The runtime toggles are **facade state**, driven via the in-process crate API — **not** an HTTP
route. They are `Atomic*` on `Higgs`, **not** persisted (each resets on restart). The
`HiggsRuntimeSettings` / `LogSettings` wire structs (`src/serve/wire.rs`) are the SHAPES an embedder
uses to read/write them in bulk.

| Toggle | Default | Facade getter / setter | Effect |
|--------|---------|------------------------|--------|
| JIT (just-in-time) loading | `true` | `jit_enabled()` / `set_jit_enabled()` | When on, a `POST /v1/chat/completions` for a scanned-but-unloaded model triggers an on-demand load (only-keep-last swap) via the same `Higgs::load()` path — RAM-headroom (`[HG017]`), charset & path-within-roots (`[HG015]`) guards still apply; a failed JIT load surfaces the REAL mapped error, not `404`. Unknown id ⇒ `404 [HG002]`. When off, an unloaded model ⇒ `404 [HG003]`. |
| Idle auto-unload | `true` | `auto_unload_idle()` / `set_auto_unload_idle()` | Master switch the node reaper reads each tick; `false` ⇒ never reaps. |
| Idle TTL minutes | `60` | `idle_ttl_minutes()` / `set_idle_ttl_minutes()` | Seeded from `node::runtime::DEFAULT_IDLE_TTL` (60 min). The reaper reads it live (change takes effect without restart); never unloads mid-generation (gated on a fully-open inference semaphore) or with nothing loaded. |
| Serving enabled | `true` | `serving_enabled()` / `set_serving_enabled()` | When `false`, `/v1` inference returns `[HG019]` → 503; read on each chat request. |
| Verbose serve logging | `false` | `logs_settings()` / `set_logs_settings()` (`LogSettings.verbose`) | When on, each completed `POST /v1/chat/completions` emits one extra `higgs: served …` INFO line into the Developer Logs. |
| Log incoming tokens | `false` | `logs_settings()` / `set_logs_settings()` (`LogSettings.log_incoming_tokens`) | Explicit opt-in that logs the flattened incoming prompt CONTENT (capped ~800 chars), overriding the redact-by-default policy. |

Other non-configurable defaults:

| Item | Value | Source |
|------|-------|--------|
| dev-log history ring cap | 2000 lines | `log_bus.rs` — `RING_CAP` (per-`LogSource` ring; the snapshot source for `Higgs::logs()`) |
| dev-log live broadcast cap | 256 | `log_bus.rs` — `BROADCAST_CAP` (live tap for `Higgs::subscribe_logs()`) |
| event broadcast channel cap | 64 | `supervisor.rs` — hardcoded |
| respawn backoff | 1 second | `supervisor.rs` — hardcoded |
| respawn attempts per death | 1 | supervisor restarts once; factory failure is terminal |
| graceful stop timeout | 2 seconds | `supervisor.rs` `from_secs(2)` (reached via `Higgs::stop()`) — hardcoded |
| idle reaper interval | derived from TTL | `node::runtime::reap_interval(ttl)` = `(ttl/4).clamp(50 ms, 60 s)` — a 60-min TTL polls once a minute |
| default listen port | 31415 (pi) | `DEFAULT_PORT` const — a documented default for external clients; the node-only binary binds no port, and an embedder picks its own |

---

## Links

- Root orientation: [README.md](README.md)
- Crate design & invariants: [DESIGN.md](DESIGN.md)
- Engine params (full `LlamaCppParams` reference): `src/worker/engine/llamacpp/`
- Remote fleet: `src/node/README.md`, [DESIGN-remote.md](DESIGN-remote.md)
