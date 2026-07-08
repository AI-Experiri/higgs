# `serve/` — the `/v1` HTTP surface

higgs is library-first, so the ONLY thing served over a socket is the strict
OpenAI-compatible **`/v1`** (chat + models). `serve/` is the axum layer over the
[`Higgs`](../api) facade that fronts it. There is **no `/api/higgs/*` HTTP
control surface** — control runs in-process through the crate API
([`Higgs::…`](../api), backed by `src/api/embed.rs`); `serve/control.rs` is now a
set of PURE helpers those facade methods delegate to, not HTTP handlers.

- **`/v1/*`** — the strict OpenAI surface (`GET /v1/models`,
  `POST /v1/chat/completions`). Request/response bodies are `async-openai` wire
  types verbatim, with two local mirrors (`v1_wire.rs`) that add the
  `reasoning_content` field the crate lacks. This is the untrusted interop
  boundary: error text is path-redacted (`redact_paths`), and it is multi-model +
  local-first (lists every reachable served id, JIT-loads a scanned-but-unloaded
  model on demand). The handlers are thin — the JIT/local-first routing and the
  reachable-id union live on the facade (`Higgs::prepare_chat` /
  `Higgs::chat_model_ids`), so the in-process embedder and this HTTP endpoint
  cannot diverge.
- **`GET /health`** — a cheap always-open readiness probe (server reachable, not
  "a model is loaded"), mirroring vllm's `/health`.

## File map

| File | Responsibility |
|------|----------------|
| `mod.rs` | Router assembly + the middleware/policy stack: `v1_router()` / `v1_router_with_host_policy()` / `serve_v1()`; the DNS-rebinding `host_guard`, Bearer `auth_guard` + per-path `required_scope`, local-origin CORS, body-size limit, the `/v1/models` timeout, catch-panic; the shared `http_status` (`HiggsError → StatusCode`) table; `/health`; and the serve-layer limit `const`s (`MAX_BODY_BYTES`, `CONTROL_TIMEOUT`, `LONG_OP_TIMEOUT`, `MAX_OUTPUT_TOKENS`, `PROMPT_BYTES_PER_TOKEN`). Re-exports `wire::*`. |
| `v1.rs` | The OpenAI `/v1` handlers: `v1_models` (maps the facade's `chat_model_ids()` reachable-id union onto OpenAI `Model` envelopes) and `v1_chat_completions` (serving gate → `Higgs::prepare_chat` → `gate_and_validate` → `Higgs::chat_stream`). Also v1-wire validation (`validate_sampling`, text-only `messages_to_pairs`), the OpenAI error envelope + `redact_paths`/`v1_error_code`, `build_sampling`, verbose/incoming log lines, tool-call interpretation, and the non-streaming response builder. |
| `control.rs` | PURE control sub-primitives the `Higgs` facade calls directly (NOT HTTP handlers): per-model row formatting (`model_entry` + the dual-profile `TuneProfileViews`/`active_records` plumbing and the Gate-2 `tool_calls` verdict), the hub-status snapshot (`hub_status`), and the API-key decision cores (`decide_mint`/`decide_revoke` + `validate_key_label`/`keystore_io_error`). |
| `stream.rs` | SSE assembly for streaming chat: bridges the `chat_stream` `(DeltaReceiver, outcome)` pair onto OpenAI `chat.completion.chunk` frames (role → reasoning/content/tool-call deltas → finish → `[DONE]`), with the verbose serving line, optional terminal `include_usage` chunk, and the `[HG057]` overflow guard that reads the bounded `delta_queue`. |
| `readiness.rs` | `ModelReadiness` (const-enum: `Discovered`/`Profiled`/`Servable`/`Unservable`/`NeedsRetune`/`Loaded`) + its pure derivation (`derive_readiness`, `ReadinessInputs`, `footprint_fits_free`). The per-model contract the UI badges and the facade's JIT readiness gate read. |
| `wire.rs` | higgs's own control-surface wire structs via `higgs_ts!` (ts-rs exported to `bindings/higgs/*.ts`): the crate-API return/request shapes the facade builds and accepts — `HiggsModelsResponse`, `HiggsModelEntry`, `ModelFit`, `HiggsLoadRequest`/`HiggsLoadResponse`, `LogSettings`, `HiggsRuntimeSettings`, `HiggsHubStatus`, `HiggsVersionResponse`, `HiggsErrorResponse`, `HiggsOk`, and the key shapes `HiggsKeyEntry`/`HiggsKeysList`/`HiggsMintKeyRequest`/`HiggsMintKeyResponse`/`HiggsKeyRemoved`. |
| `v1_wire.rs` | Local mirrors of the two async-openai `/v1` response shapes higgs must extend with `reasoning_content` (`ReasoningChatResponse`/`ReasoningResponseMessage`/`ReasoningChatChoice` and the streaming `ReasoningStreamChunk`/`ReasoningChoiceStream`/`ReasoningStreamDelta`). Serde byte-parity with the crate types is load-bearing (proven in `v1_wire_tests.rs`). |
| `*_tests.rs`, `test_support.rs` | Unit tests (sibling `_tests.rs` per prod file) and the `#[cfg(test)]` handler harness (`Higgs` over a fake `NodeRuntime`, `make_app*`/`make_higgs*` builders, GGUF-fixture writer). Not part of the runtime surface. |

## Public / crate surface

- `pub fn v1_router(higgs: Arc<Higgs>) -> Router` — the loopback-guarded `/v1`
  router for embedders. The relaxed-host variant `v1_router_with_host_policy`
  (takes an `enforce_loopback_host: bool`) is **`pub(crate)` by design** (never
  `pub`): the only way to reach the relaxed policy is `serve_v1`, which runs the
  `[HG058]` keyless-LAN and `[HG069]` no-Admin-key refusals and records
  `lan_exposed` first.
- `pub async fn serve_v1(higgs, listener, shutdown)` — the single
  graceful-shutdown entry point and the ONLY HTTP surface higgs exposes; drains
  in-flight requests then calls `Higgs::stop`. An embedder that wants an external
  OpenAI endpoint (e.g. jigglebot) calls it with its own `shutdown` future; the
  node-only `higgs` binary itself runs the iroh daemon, not this HTTP server.
- `pub mod readiness` (`ModelReadiness` + derivation) — public because
  `ModelReadiness` is a field on the re-exported `wire::HiggsModelEntry`.
- `pub use wire::*` — the control wire structs are part of the crate's public
  surface and the ts-rs export path (`bindings/higgs/*.ts`).
- `pub(crate) fn http_status(&HiggsError) -> StatusCode` — the one status table,
  shared by the `/v1` handlers and the SSE error path.
- `pub const` limits (`MAX_BODY_BYTES`, `CONTROL_TIMEOUT`, `LONG_OP_TIMEOUT`,
  `MAX_OUTPUT_TOKENS`, `PROMPT_BYTES_PER_TOKEN`) — grouped for a later
  `HiggsConfig` lift; the fit/clamp consts (`MAX_OUTPUT_TOKENS`,
  `PROMPT_BYTES_PER_TOKEN`) are consumed by the facade's `fit_generation_budget`.

Everything else (individual handlers, the envelope/redaction helpers, the SSE
assembly, the mint/revoke deciders) is `pub(super)`/`pub(crate)`/private and
reached only through the router or the facade.

See `DESIGN.md` for the single-surface rationale, the middleware order, the
auth/scope model, how the JIT/local-first routing now lives on the facade, the
streaming backpressure design, and the full list of `HGxxx` codes this module
owns and maps.
