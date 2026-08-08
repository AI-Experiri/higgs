# `worker/` — the inference subprocess (host↔worker split)

higgs runs inference in a **separate OS process** for crash isolation: a model that segfaults
or OOMs takes down only the worker, never the host. `worker/` is that subprocess side. The
host (`supervisor.rs`) re-execs this binary, and the two speak **NDJSON JSON-RPC over
stdin/stdout** — one request line in, its response line out, plus chat-token notifications
streamed mid-request. Logs go to stderr; the supervisor is the ONLY client.

The worker owns the engine (llama.cpp FFI) and the loaded model; it holds **no model
catalog** — the host scans and passes a resolved GGUF path in `M_LOAD`.

## File / submodule map

| File / submodule | Responsibility |
|------------------|----------------|
| `mod.rs` | The RPC server: `worker_main()` entry, the stdin run loop (`serve`/`serve_state`), method dispatch, the `M_*`/`N_*` protocol constants, `WorkerState` (the engine + the loaded `(id, LoadParams)`), the chat streaming sink, and JSON-RPC error routing (`to_rpc_error`). |
| `models.rs` | The model store: read-only discovery across LM Studio / HuggingFace-cache / Ollama dirs (`ModelStore::scan`), producing `HiggsModel` records enriched from the GGUF header. **Pure Rust — no FFI** (`gguf-rs-lib` + `memmap2` + `std::fs`), so it runs **host-side** (the host scans; `M_LOAD` carries the resolved path). It lives here because it owns the `HiggsModel`/`HiggsModelSource` wire types. |
| `models_tests.rs` | Unit tests for `models.rs` (sibling test file; not production code). |
| `engine/` | The `HiggsEngine` trait + the pluggable engine registry (`HIGGS_ENGINE`). The seam the rest of higgs is written against. Has its own README/DESIGN. |
| `engine/llamacpp/` | The concrete llama.cpp engine — the only FFI boundary. Has its own README/DESIGN. |

## The RPC protocol (defined in `mod.rs`)

**Requests** — `M_*`, each gets exactly one response (matching `id`):

| Const | Method string | Meaning |
|-------|---------------|---------|
| `M_LOAD` | `higgs/load` | Load the GGUF at a host-resolved `path` (+ `id`, `ctx_len`, `gpu_layers`, `threads`, and any engine-specific `LlamaCppParams` knobs deserialized flat from the params). Replies `{id}`. |
| `M_UNLOAD` | `higgs/unload` | Drop the resident model; clear the tracked id. Replies `{}`. |
| `M_STATUS` | `higgs/status` | Report `{loaded: {id, ctx_len, gpu_layers, threads}}` or `{loaded: null}`. |
| `M_CHAT` | `higgs/chat` | Run inference (`model` bind, `messages_json`, `max_tokens`, `sampling` — the serialized `SamplingParams` umbrella; absent/malformed falls back to a legacy top-level `temperature` — optional `tools`, `chat_template_kwargs`, `request_id`). Streams `N_CHAT_CHUNK`, then replies `{content, finish_reason, tool_calls, prompt_tokens, completion_tokens, reasoning_content}`. |
| `M_SYSINFO` | `higgs/sysinfo` | Enumerate compute devices `{gpus: [GpuDevice, …]}` (cheap, read-only, no model load). |
| `M_LOG_LEVEL` | `higgs/log_level` | Toggle engine log verbosity live (`{verbose}`). Replies `{}`. |
| `M_SHUTDOWN` | `higgs/shutdown` | Reply `{}`, then exit the loop. |

**Notification** — `N_CHAT_CHUNK` (`higgs/chat/chunk`): one streamed delta
(`{request_id, kind, delta, …}`, shaped by `ChatDelta::encode_chunk_params`), fire-and-forget
(the host routes it to the request's keyed sink).

JSON-RPC error `code`s: `-32700` parse, `-32602` invalid params, `-32601` unknown method,
`-32000` application error. Each carries `data.code` = the originating `HG*` code so the host's
`http_status` table maps a worker failure to the right HTTP status instead of collapsing to 500.

## `models.rs` public surface

- `ModelStore` — `scan(lmstudio, hf, ollama) -> Result<&[HiggsModel]>` walks every root, dedups by
  `(id, path)`, sorts, replaces the catalog; `models()` returns it; `get(id)` looks up by repo id
  (lexically-first path when a repo has multiple variants).
- `HiggsModel` — one discoverable GGUF: identity (`id` = HF `org/model`, or `ollama/{name}:{tag}`),
  `path`, `size_bytes`, `quant`, `source`, plus GGUF-header enrichment: `arch`, `ctx_train`,
  `block_count`, `head_count`, `head_count_kv`, `embedding_length`, `expert_count` (the autotune
  KV-VRAM inputs), `has_chat_template`, `supports_tools`, `supports_reasoning`, `gguf_components`
  (curated load-relevant header fields for the UI), and a transient host-only `chat_template`
  (`serde(skip)`).
- `HiggsModelSource` — `LmStudio` / `HfCache` / `Ollama` (a `higgs_const_enum!`).

## How the rest of the crate uses it

- `Higgs::scan` (`src/api.rs`) builds a `ModelStore::default()`, scans the configured roots off
  the executor (`spawn_blocking`), and returns the `Vec<HiggsModel>` catalog. `node/runtime.rs`
  (`resolve_model`, `do_scan`) does the same to resolve a repo id to its on-disk GGUF path
  (passed into `M_LOAD`, path-guarded to stay within a scan root) and to serve the node's model
  catalog.
- `Higgs::model_entries` (`src/api/embed.rs`) folds each scanned `HiggsModel` — plus its load
  state, Gate-2 tool-call verdict, last-load params, readiness/fit, and the dual tune profiles —
  into a `HiggsModelEntry` row via `serve::control::model_entry` (`serve/wire.rs` defines the
  wire type). This is the enriched models-list view the crate API exposes (control runs
  in-process; there is **no** `/api/higgs/*` HTTP surface — that route was retired).
- `tune/mod.rs` (`ModelMeta::from_model`) projects a `HiggsModel` into the autotune suggester's
  typed input.
- `worker_main()` is called from the host binary's `main()` for the `--higgs-worker` role.

See `DESIGN.md` for the run-loop invariants and the host↔worker contract.
