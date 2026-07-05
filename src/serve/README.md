# `serve/` — the HTTP surface

`serve/` is the axum HTTP layer over the [`Higgs`](../api) facade. It exposes two
distinct surfaces plus the shared router/middleware that front them. One
`router(Arc<Higgs>)` assembles both; the standalone `higgs` binary and the
embedded in-process host both serve it (via `serve_with_shutdown`).

- **`/v1/*`** — the strict OpenAI-compatible surface (`GET /v1/models`,
  `POST /v1/chat/completions`). Request/response bodies are `async-openai` wire
  types verbatim, with two local mirrors (`v1_wire.rs`) that add the
  `reasoning_content` field the crate lacks. This is the untrusted interop
  boundary: error text is path-redacted, and it is multi-model + local-first
  (lists every served id, JIT-loads a scanned-but-unloaded model on demand).
- **`/api/higgs/*`** — higgs's OWN control surface (scan, load/unload, tune,
  estimate, status, system, logs + log/event SSE, settings, keys, version,
  worker stop, and the hub/fleet pairing + node routes). Its shapes are higgs's
  own (`wire.rs`), and error text is the full unredacted `HiggsError` display
  (it is our surface, not third-party).

## File map

| File | Responsibility |
|------|----------------|
| `mod.rs` | Router assembly + the middleware/policy stack: `router()` / `router_with_host_policy()` / `serve_with_shutdown()`; the DNS-rebinding `host_guard`, Bearer `auth_guard` + per-route `required_scope`, local-origin CORS, body-size limit, control/long-op timeouts, catch-panic; the shared `http_status` (`HiggsError → StatusCode`) table; `/health`; and the serve-layer limit `const`s (`MAX_BODY_BYTES`, `CONTROL_TIMEOUT`, `LONG_OP_TIMEOUT`, `MAX_OUTPUT_TOKENS`, `PROMPT_BYTES_PER_TOKEN`). Re-exports `wire::*`. |
| `v1.rs` | The OpenAI `/v1` handlers: `v1_models` (served-id listing) and `v1_chat_completions` (serving gate → `ensure_loaded`/JIT → `gate_and_validate` → dispatch). Also message flattening (`messages_to_pairs`), sampling/prompt-fit validation, the OpenAI error envelope + `redact_paths`, verbose/incoming log lines, and the non-streaming response builder. |
| `control.rs` | The `/api/higgs/*` handlers: models list (+ per-model `readiness`/`fit`/Gate-2 `tool_calls` verdict), single model by id, load/tune/estimate/unload, status, system, logs (snapshot + SSE) and log/runtime settings GET/PUT, version, worker stop, the **G4 key-management** routes (`control_keys_list`/`_mint`/`_revoke` + the pure `decide_mint`/`decide_revoke`), and the hub/fleet routes (pair, hub enable/disable, nodes list/load/unload/retire/label, node models). |
| `stream.rs` | SSE assembly for streaming chat: bridges the `chat_stream` `(DeltaReceiver, outcome)` pair onto OpenAI `chat.completion.chunk` frames (role → reasoning/content/tool-call deltas → finish → `[DONE]`), with the verbose serving line, optional terminal `include_usage` chunk, and the `[HG057]` overflow guard that reads the bounded `delta_queue`. |
| `readiness.rs` | `ModelReadiness` (const-enum: `Discovered`/`Profiled`/`Servable`/`Unservable`/`NeedsRetune`/`Loaded`) + its pure derivation (`derive_readiness`, `ReadinessInputs`, `footprint_fits_free`). The per-model contract the UI badges and JIT gate read. |
| `wire.rs` | higgs's own control-surface wire structs via `higgs_ts!` (ts-rs exported): `HiggsModelsResponse`, `HiggsModelEntry`, `ModelFit`, `HiggsLoadRequest`/`HiggsLoadResponse`, `LogSettings`, `HiggsRuntimeSettings`, `HiggsHubStatus`, `HiggsVersionResponse`, `HiggsErrorResponse`, `HiggsOk`, and the key shapes `HiggsKeyEntry`/`HiggsKeysList`/`HiggsMintKeyRequest`/`HiggsMintKeyResponse`/`HiggsKeyRemoved`. |
| `v1_wire.rs` | Local mirrors of the two async-openai `/v1` response shapes higgs must extend with `reasoning_content` (`ReasoningChatResponse`/`ReasoningResponseMessage`/`ReasoningChatChoice` and the streaming `ReasoningStreamChunk`/`ReasoningChoiceStream`/`ReasoningStreamDelta`). Serde byte-parity with the crate types is load-bearing (proven in `v1_wire_tests.rs`). |
| `*_tests.rs`, `test_support.rs` | Unit tests (sibling `_tests.rs` per prod file) and the `#[cfg(test)]` handler harness (`Higgs` over a fake `NodeRuntime`, `make_app*`/`make_higgs*` builders, GGUF-fixture writer). Not part of the runtime surface. |

## Public / crate surface

- `pub fn router(higgs: Arc<Higgs>) -> Router` — the loopback-guarded router for
  embedders. The relaxed-host variant `router_with_host_policy` is
  **`pub(crate)` by design** (never `pub`; the crate visibility exists only for
  the in-crate timeout DI test): the only way an embedder reaches it is
  `serve_with_shutdown`, which runs the `[HG058]` keyless-LAN and `[HG069]`
  no-Admin-key refusals and records `lan_exposed` first.
- `pub async fn serve_with_shutdown(higgs, listener, shutdown)` — the single
  graceful-shutdown entry point; drains in-flight requests then calls
  `Higgs::stop`. The binary wires it to SIGTERM/Ctrl-C.
- `pub mod readiness` (`ModelReadiness` + derivation) — public because
  `ModelReadiness` is a field on the re-exported `wire::HiggsModelEntry`.
- `pub use wire::*` — the control wire structs are part of the crate's public
  surface and the ts-rs export path (`bindings/higgs/*.ts`).
- `pub(crate) fn http_status(&HiggsError) -> StatusCode` — the one status table,
  shared by both surfaces and the SSE error path.
- `pub const` limits (`MAX_BODY_BYTES`, `CONTROL_TIMEOUT`, `LONG_OP_TIMEOUT`,
  `MAX_OUTPUT_TOKENS`, `PROMPT_BYTES_PER_TOKEN`) — grouped for a later `HiggsConfig` lift.

Everything else (individual handlers, envelope/redaction helpers, the SSE
assembly, the mint/revoke deciders) is `pub(super)`/private and reached only
through the router.

See `DESIGN.md` for the two-surface split, the middleware order, the auth/scope
model, the JIT/local-first routing, the streaming backpressure design, and the
full list of `HGxxx` codes this module owns.
