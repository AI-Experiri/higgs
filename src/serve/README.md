# `serve/` — the HTTP surface

`serve/` is the axum HTTP layer over the [`Higgs`](../api) facade. It exposes two distinct
surfaces and the shared middleware/router that front them:

- **`/v1/*`** — the strict OpenAI-compatible surface (`GET /v1/models`,
  `POST /v1/chat/completions`), bodies verbatim `async-openai` wire types. This is the
  untrusted interop boundary: error text is path-redacted, and it is multi-model + local-first
  (lists all served ids, JIT-loads scanned models on demand).
- **`/api/higgs/*`** — higgs's OWN control surface (scan, load/unload, status, system, logs +
  log-stream SSE, settings, version, worker stop, and the hub/fleet pairing + node routes). Its
  shapes are higgs's own (`wire.rs`), and error text is unredacted (it is our surface).

`router(Arc<Higgs>)` assembles both behind the middleware stack; the standalone binary and the
embedded host both serve it.

## File map

| File | Responsibility |
|------|----------------|
| `mod.rs` | `router()` assembly + the middleware/policy layer: DNS-rebinding host guard (loopback-only `Host`), body-size limit (413), Bearer-token auth + per-route scope (`required_scope`), local-origin CORS, the `HiggsError → StatusCode` mapping (`http_status`, including per-worker-code mapping), `/health`, and the serve-layer limit constants (`MAX_BODY_BYTES`, `CONTROL_TIMEOUT`, `MAX_OUTPUT_TOKENS`, `PROMPT_BYTES_PER_TOKEN`). |
| `v1.rs` | The OpenAI `/v1` handlers: `v1_models` (served-id listing), `v1_chat_completions` (serving gate → `ensure_loaded`/JIT → validation → dispatch), message flattening, sampling/prompt-fit validation, the OpenAI error envelope + path redaction, and the response builders. |
| `control.rs` | The `/api/higgs/*` handlers: models list (+ Gate-1/Gate-2 support verdict per model), single model by id, load/unload, status, system, logs (snapshot + SSE stream), log/runtime settings GET/PUT, version, worker stop, and the hub/fleet routes (pair, nodes, node load/unload/models/retire). |
| `stream.rs` | SSE assembly for streaming chat: turns the chat delta receiver + final outcome into OpenAI `chat.completion.chunk` frames (role → deltas → finish → `[DONE]`), with the verbose serving line and optional terminal usage chunk. |
| `wire.rs` | higgs's own control-surface wire types (`HiggsModelsResponse`, `HiggsModelEntry`, `HiggsLoadRequest`, settings, version, logs, `HiggsOk`, error response) via `higgs_ts!`. |
| `test_support.rs` (`#[cfg(test)]`) | The handler-test harness: builds a `Higgs` over a stateful fake `NodeRuntime` (auto-responding worker, no llama.cpp), the `make_app*`/`make_higgs*`/`app_for` builders, GGUF-fixture writer, and request/response helpers. |

See `DESIGN.md` for the two-surface split, the middleware order, and the local-first routing.
