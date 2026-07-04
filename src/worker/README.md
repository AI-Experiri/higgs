# `worker/` — the inference subprocess (host↔worker split)

higgs runs inference in a **separate OS process** for crash isolation: a model that segfaults
or OOMs takes down only the worker, never the host. `worker/` is that subprocess side. The
host (`supervisor.rs`) re-execs this binary, and the two speak **NDJSON JSON-RPC over
stdin/stdout** — one request line in, its response line out, plus chat-token notifications
streamed mid-request.

The worker owns the engine (llama.cpp FFI) and the loaded model; it holds **no model
catalog** — the host scans and passes a resolved GGUF path in `M_LOAD`.

## File / submodule map

| File / submodule | Responsibility |
|------------------|----------------|
| `mod.rs` | The RPC server: the stdin run loop, method dispatch, the M_*/N_* protocol constants, `WorkerState` (the engine + the loaded `(id, LoadParams)`), the chat streaming sink, and JSON-RPC error routing. |
| `models.rs` | The model store: read-only discovery across LM Studio / HuggingFace-cache / Ollama dirs, enriching GGUF headers (arch, quant, trained context, chat template, tool/reasoning support). Pure Rust (no FFI) — so it runs **host-side** (the host scans; `M_LOAD` carries the resolved path). It lives here next to the model types it produces. |
| `engine/` | The `HiggsEngine` trait + the pluggable engine registry (`HIGGS_ENGINE`). The seam the rest of higgs is written against. Has its own README/DESIGN. |
| `engine/llamacpp/` | The concrete llama.cpp engine — the only FFI boundary. Has its own README/DESIGN. |

## The RPC protocol (defined in `mod.rs`)

**Requests** — `M_*`, each gets exactly one response (matching `id`):

| Const | Method string | Meaning |
|-------|---------------|---------|
| `M_LOAD` | `higgs/load` | Load the GGUF at a host-resolved `path` (+ `id`, `ctx_len`, `gpu_layers`, `threads`, optional engine knobs). |
| `M_UNLOAD` | `higgs/unload` | Drop the resident model. |
| `M_STATUS` | `higgs/status` | Report `{loaded: {id, ctx_len, gpu_layers, threads}}` or `{loaded: null}`. |
| `M_CHAT` | `higgs/chat` | Run inference (`model` bind, `messages_json`, `max_tokens`, `sampling` — the serialized `SamplingParams` umbrella; absent/malformed falls back to a legacy top-level `temperature` — optional `tools`, `request_id`). Streams `N_CHAT_CHUNK`, then replies `{content, finish_reason, tool_calls, prompt_tokens, completion_tokens}`. |
| `M_SYSINFO` | `higgs/sysinfo` | Enumerate compute devices `{gpus: [...]}` (cheap, no model load). |
| `M_LOG_LEVEL` | `higgs/log_level` | Toggle engine log verbosity live (`{verbose}`). |
| `M_SHUTDOWN` | `higgs/shutdown` | Reply, then exit the loop. |

**Notification** — `N_CHAT_CHUNK` (`higgs/chat/chunk`): one streamed content delta
`{request_id, delta}`, fire-and-forget (the host routes it to the request's keyed sink).

JSON-RPC error `code`s: `-32700` parse, `-32602` invalid params, `-32601` unknown method,
`-32000` application error (carries `data.code` = the HG* code so the host maps it to the right
HTTP status).

See `DESIGN.md` for the run-loop invariants and the host↔worker contract.
