# `serve/` — design notes

## Why two surfaces

higgs serves the same engine through two HTTP surfaces with deliberately
different contracts, fronted by one `router` (no second URL/port):

- **`/v1` is the untrusted interop boundary.** It must match the OpenAI shape
  byte-for-byte (clients are third-party SDKs), so bodies are `async-openai`
  types verbatim, errors use the OpenAI `{"error":{message,type,code}}` envelope
  (`v1::v1_envelope`), and that message is **path-redacted** (`v1::redact_paths`)
  — host filesystem paths and `host:port` addresses are stripped so the no-auth
  loopback server never leaks its layout. `/v1/models` answers "what can serve
  chat right now" (served ids ∪ JIT-servable ∪ fleet-routed), never the raw
  on-disk catalog.
- **`/api/higgs/*` is our own control surface.** Shapes are higgs's own
  (`wire.rs`), error text is the full unredacted `HiggsError` display
  (`control::control_error`), and it carries everything the UI needs: the
  on-disk catalog with readiness/fit/tool verdicts, load/tune/estimate/unload,
  status, system, the Developer-Log snapshot + live SSE, the load-lifecycle
  event SSE, runtime settings, key management, and hub/fleet control.

Keeping them separate lets `/v1` stay strict and minimal while the control
surface evolves freely.

### `v1_wire.rs` — the one place `/v1` isn't verbatim

async-openai 0.41 has no `reasoning_content` field and no extension hook, so the
two chat-response shapes higgs must extend are hand-mirrored in `v1_wire.rs`
(non-streaming `ReasoningChatResponse`, streaming `ReasoningStreamChunk`). Serde
**byte-parity** with the crate structs is load-bearing (same field set,
skip/null semantics); `v1_wire_tests.rs` proves it by comparing serialized
`Value`s. Everything else on `/v1` still serializes from crate types.

## Middleware order (mod.rs)

Every request passes the same policy stack. In Tower the LAST `.layer(...)`
wraps the earlier ones, so request entry runs **outermost-first** — the reverse
of the source order. Request-entry order:

1. **CORS** (outermost, `local_cors`): only local origins — the Tauri webview
   (`tauri://localhost`, `http://tauri.localhost`), any loopback HTTP origin,
   plus `higgs.extra_cors_origins()` — are allowed. higgs is localhost/webview
   only, not a public surface.
2. **Host guard** (`host_guard`, DNS-rebinding defense): when enforced, the
   `Host` header must be a loopback host (`localhost`, `127.0.0.0/8`, `::1`) or
   it is `403 [HG012]`. Runs **before** auth so a foreign host is rejected before
   any key work. Relaxed (skipped) only on a deliberate non-loopback bind, which
   `[HG058]` guarantees is key-gated.
3. **Auth + scope** (`auth_guard`): an empty keystore = auth OFF (the embedded
   in-process host wants no gate). When keys exist, `required_scope(method, path)`
   maps the route to a `Scope` and the `Authorization: Bearer hgk_…` token must
   carry it; failure is `401 [HG048]` with `WWW-Authenticate: Bearer`.
4. **Catch-panic**, then **body limit** (innermost): a handler panic becomes a
   `500`; a body over `MAX_BODY_BYTES` (32 MiB) is `413` before the handler runs.

The **control** sub-router additionally carries `CONTROL_TIMEOUT` (120 s). The
**long-op** sub-router (`/api/higgs/models/load` + `/api/higgs/models/tune`) carries
`LONG_OP_TIMEOUT` (20 min) instead: a load can walk the G5 OOM degrade-retry ladder
(several worker loads + settle sleeps) and a Turbotune benchmark loads + measures
several candidate configs — both routinely exceed 120 s, and under the control cap
they would be 408-cancelled before the ladder returns HG060 / a profile is saved.
`router_with_host_policy` takes both timeouts as parameters (a DI seam — tests build
the real router with a tiny control timeout and prove the long-op routes escape it);
production passes the two consts. The **streaming** sub-router
(`/v1/chat/completions`, `/api/higgs/logs/stream`, `/api/higgs/events`) is
deliberately **un-timed** at the HTTP layer — an SSE stream must outlive any
per-request bound (chat duration is bounded separately by the worker chat-RPC
timeout).

### Auth scope model (`required_scope`, fail-closed)

`/health` + `/api/higgs/health` → open (`None`). `/v1/chat/completions` → `Chat`.
`/v1/models` and `GET /api/higgs/models*` → `Models`. **Everything else** under
`/v1/` or `/api/higgs/` → `Admin`. That final catch-all is the **fail-closed
default**: any new control route (including the G4 key routes and every
`nodes/*`, `hub/*`, `settings` mutation) is Admin-scoped automatically unless
explicitly relaxed — a forgotten route can never accidentally serve open.

## G4 — API-key management over the control surface

`control_keys_list` / `control_keys_mint` / `control_keys_revoke`
(`GET`/`POST`/`DELETE /api/higgs/keys[/{label}]`) manage keys at runtime, all
Admin-scoped via the fail-closed default above. Mutations go through
`Higgs::mutate_api_keys` (file read-modify-write under a lock + in-memory
hot-swap), so a minted/revoked key gates the **very next** request — no restart.
It persists + re-installs ONLY when the closure actually changed the store: a
REJECTED request (unauthorized mint, duplicate label, unknown revoke, last-key/
last-Admin conflict) does no disk write, so an unwritable keystore can't turn the
intended 401/400/409 into an HG040 500.

The decision logic is factored into the pure, unit-tested `decide_mint` /
`decide_revoke`, run **inside** the keystore lock so the empty-store bootstrap
window can't be raced:

- **Bootstrap mint** (empty store) is allowed unauthenticated and MUST grant
  `Admin` — it defaults to `[admin]` when scopes are omitted and rejects explicit
  non-admin scopes (`Mint::BootstrapNeedsAdmin`), because a non-admin first key
  would flip auth ON yet be unable to reach the Admin key API, locking the HTTP
  surface out of itself with no recovery path.
- A second unauthenticated mint that reaches the lock after the first key landed
  sees a non-empty store, re-checks its bearer, and is refused
  (`Mint::Unauthorized`). Revoke re-derives authorization the same way
  (`decide_revoke`).
- Later omitted-scopes mints default to `[chat, models]`.
- Labels are validated at mint (`1–64` chars from `[A-Za-z0-9._-]`, and the
  literal `.` / `..` are rejected — they pass the charset but URL parsers
  normalize dot-segments away) so they
  round-trip through the single-segment `DELETE /api/higgs/keys/{label}` route —
  an unrevokable key can't be created.
- The plaintext token is returned exactly once (mint response) and never logged
  or stored; the list route shows only label/scopes/`sha256_prefix`.

### Last-key / LAN interlock (`[HG058]`/`[HG059]`)

Two guards keep a keyed LAN bind from silently going open:

- **Startup (`[HG058]`)**: `serve_with_shutdown` refuses to serve a non-loopback
  listener with zero keys (auth off AND host guard relaxed = whole surface open).
  It tears the facade down (`Higgs::stop`) and returns `LanBindWithoutKeys` as an
  `io::Error` — a startup refusal, so it does **not** flow through `http_status`.
- **Runtime (`[HG059]`)**: `serve_with_shutdown` records
  `higgs.set_lan_exposed(!loopback)`; `decide_revoke` then refuses to revoke the
  LAST key while `lan_exposed` (`Revoke::LastKeyOnLan` → `409`), the runtime
  counterpart of the startup guarantee.

`router_with_host_policy` is `pub(crate)` — never `pub` — precisely so the
relaxed-host policy is reachable ONLY after these guards have run (the crate
visibility exists solely for the in-crate timeout DI test).

## JIT / local-first / multi-model routing

The local node is multi-model, so `/v1` is genuinely multi-model and
**local-first** (consistent with `../api/DESIGN.md`):

- **`v1_models`** = `local_served_ids()` (every resident local instance) ∪ the
  JIT-servable catalog ids (only while `jit_enabled`) ∪ the fleet's routed
  models, skipping name collisions. Never the raw catalog — unprepared / stale /
  oversized models are hidden because the JIT gate would refuse them, and a
  model owned by a running benchmark is filtered out unless a remote node also
  serves the id (`ensure_loaded` would refuse it `[HG068]`).
- **`ensure_loaded`** (the chat gate) resolves LOCAL-first: an already-served id
  wins (`local_loaded_info`, no reload); else a fleet-remote model returns a
  permissive `LoadedInfo` (the remote worker's `[HG005]` is the prompt-fit
  backstop); else with JIT off it is `404 [HG003]`. With JIT on, the id must be a
  scanned model (else `404 [HG002]` — never load an unknown id), must pass the
  **readiness gate** (`profile_state`: `Missing → [HG046]`, `Stale → [HG047]`,
  both `400`), then loads additively with the **validated** profile captured from
  the gate (no second `models.json` read a concurrent retune could poison).
- **`gate_and_validate`** returns the RESOLVED model id, which binds dispatch:
  the worker rejects (`[HG018]`) if a concurrent JIT load swaps the model out
  before generation, so a swap errors instead of serving the wrong model.
- **`control_models`** flags EVERY resident model `loaded` (the full served set),
  not just the primary; the singular `loaded_id` field stays the primary for
  back-compat.

### Request validation on `/v1` (before dispatch)

- **Serving toggle**: `!serving_enabled` short-circuits to `[HG019]` → 503 before
  any JIT load or worker RPC, so the control surface stays reachable to re-enable.
- **Sampling** (`validate_sampling`, ranges mirror vllm): `temperature ≥ 0`,
  `top_p ∈ (0,1]`, `n == 1` (higgs serves one choice), penalties `∈ [-2,2]`,
  `max_tokens ∈ [1, MAX_OUTPUT_TOKENS]` → `400 [HG013]`.
- **Prompt fit** (`fit_generation_budget`): the serve layer has no tokenizer, so
  it uses a conservative LOWER bound (`prompt_bytes / PROMPT_BYTES_PER_TOKEN`).
  It **clamps** an oversized `max_tokens` to what fits after the prompt (truncate,
  `finish_reason:"length"`) and only errors `[HG005] context_length_exceeded`
  when the prompt ALONE overflows. The worker's tokenizer-exact `[HG005]` is the
  authoritative backstop; an `Auto`/unknown window defers entirely to it.
- **Content**: `/v1` is text-only — image/audio/file/refusal parts → `400`. The
  flattened pairs are used only for validation; the RAW `messages` JSON rides to
  the template verbatim so extension fields (assistant `reasoning_content`,
  `chat_template_kwargs`) survive. A non-object `chat_template_kwargs` is ignored
  with a `[HG055]` warn trace (llama.cpp's own lenient handling).

## Streaming (stream.rs) + G1 bounded delta queue

For `stream: true`, `chat_sse` spawns `assemble`, which drains the
`chat_stream` `(DeltaReceiver, outcome)` pair onto OpenAI chunks: assistant-role
chunk → reasoning/content/tool-call deltas (engine order, reasoning first) →
optional terminal tool-call chunk → finish chunk → `data: [DONE]`, plus a final
usage-only chunk when `stream_options.include_usage` is set. A mid-stream
`HiggsError` is emitted as an OpenAI error data frame (OpenAI's own convention)
before `[DONE]`; a `JoinError` becomes `[HG044] ChatTaskFailed`.

**Backpressure (G1)**: the payload channel between `assemble` and the SSE body is
intentionally small (`SSE_BUFFER_CHUNKS = 32`). When a stalled client fills it,
`push` blocks, assembly stops pulling, and the upstream bounded
[`delta_queue`](../delta_queue) MERGES the backlog into a few growing runs
instead of fragmenting it. If a client stops reading entirely and the queue's
overflow cap trips, `drain_deltas` returns the buffered byte count and `assemble`
fails the stream LOUDLY with `[HG057] ChatStreamOverflow` (the generation
finished server-side, but this client's content is incomplete — don't finish as
if it were delivered). A malformed streamed tool-call fragment is dropped with a
`[HG056]` warn; the terminal buffered chunk still covers the call.

The two control SSE streams differ by design: `control_logs_stream` subscribes
then replays a history prefix (bounded ring, `Lagged` → gap marker + continue);
`control_events_stream` (`/api/higgs/events`, model-load lifecycle) has NO replay
ring — it unfolds directly over the broadcast receiver so no task lingers during
quiet periods, and a `Lagged` skip only coarsens the progress bar (documented
residual). **Never leave an SSE stream open across shutdown** — it blocks the
graceful drain.

## `http_status` — the one status table

`http_status` is the single `HiggsError → StatusCode` map, shared by `/v1`, the
control surface, and the SSE error path (so a streaming `[HG005]` surfaces
`context_length_exceeded` on both paths, via `v1_error_code`). Notable arms:

- `HG002`/`HG003` → 404; `HG005`/`HG013`/`HG015`/`HG046`/`HG047`/`HG049`/`HG050`
  → 400; `HG006`/`HG007`/`HG014`/`HG017`/`HG018`/`HG019` → 503; `HG016` → 504;
  `HG037` → 501; `HG038`/`HG039` → 502; `HG040`–`HG044` → 500 (default).
- **`HG059` (`LastKeyOnLan`), `HG066` (`LastAdminKey`), `HG064` (`BenchCancelled`),
  and `HG067` (`BenchModelLoaded`) → 409** — refused state transitions the client
  resolves (mint a replacement key / unload the model first).
- **`HG060` (`LoadOomExhausted`), `HG063` (`BenchExhausted`), and `HG068`
  (`BenchInProgress`) → 503** — a load that exhausted the OOM degrade ladder / a
  benchmark that found no fitting config / a model owned by a running benchmark are
  retryable capacity signals, not client errors.
- `WorkerRpc` maps by its carried `worker_code` so a worker-origin (or remote-
  node-propagated) code keeps its true status across the process boundary
  (`HG002/003→404`, `HG005/HG050→400`, `HG006/007/017/018/060→503`, `HG016→504`,
  `HG037→501`, `HG038→502`, else 500).

## HGxxx codes this module owns (raised in serve code)

| Code | Where | Status |
|------|-------|--------|
| `HG012` `ForbiddenHost` | `host_guard` | 403 |
| `HG048` `Unauthorized` | `auth_guard` / `unauthorized()` | 401 |
| `HG049` `InvalidRequest` | `v1_bad_request`, key label/scope checks | 400 |
| `HG005` `ContextOverflow` | `fit_generation_budget` | 400 |
| `HG013` `InvalidSamplingParam` | `validate_sampling` | 400 |
| `HG046`/`HG047` `NotPrepared`/`ProfileStale` | `ensure_loaded` readiness gate | 400 |
| `HG019` `ServingDisabled` | chat serving gate | 503 |
| `HG002`/`HG003` `ModelNotFound`/`ModelNotLoaded` | `ensure_loaded`, `control_model_by_id` | 404 |
| `HG055` | `chat_template_kwargs` not a JSON object | warn-only (ignored) |
| `HG056` | malformed streamed tool-call fragment | warn-only (dropped) |
| `HG057` `ChatStreamOverflow` | `stream::assemble` | SSE error frame (envelope 500) |
| `HG058` `LanBindWithoutKeys` | `serve_with_shutdown` startup | refusal (`io::Error`, not HTTP) |
| `HG069` `LanBindWithoutAdminKey` | `serve_with_shutdown` startup (keys present, none Admin — the key-management API would be locked out) | refusal (`io::Error`, not HTTP) |
| `HG059` `LastKeyOnLan` | `decide_revoke` | 409 |
| `HG066` `LastAdminKey` | `decide_revoke` (revoking the last Admin key while others remain) | 409 |
| `HG068` `BenchInProgress` | `ensure_loaded` (a benchmark owns the model; refused only if no remote node serves the id) | 503 |
| `HG040` `PersistenceFailed` | `keystore_io_error`, local relabel | 500 |
| `HG042` `InternalFault` | `control_system` blocking-join panic | 500 |
| `HG043` `HubControlFailed` | node retire/label | 500 (404 for unknown node) |
| `HG044` `ChatTaskFailed` | `stream::assemble` JoinError | 500 |

Hub-only routes (`pair`, `hub/enable|disable`, `nodes/*`) return a bare `409`
(`not_a_hub`) when no fleet is installed. `HG060`/`HG063`/`HG064`/`HG067` (and the
facade-side `HG068` raise sites) originate in lower layers (load ladder / turbotune)
and reach clients through `http_status`.

## Concurrency / locking

The serve layer is mostly stateless request handlers; the interesting locks are
held briefly for correctness:

- **Keystore mutations** (`mutate_api_keys`) run `decide_mint`/`decide_revoke`
  and the mutation atomically inside the store lock — closes the bootstrap TOCTOU
  (two unauthenticated mints while empty; only the one that still finds it empty
  bootstraps, the loser re-checks its bearer and is refused).
- **Hub lifecycle** (`higgs.hub_lifecycle().lock()`) serializes
  `/pair`, `hub/enable`, `hub/disable`, and remote `nodes/label` so a mint/relabel
  can't clone a closing hub or write a stale `pairings.json` (each handler
  documents its window). `hub/disable` publishes `hub() == None` synchronously
  before tearing down so a lock-waiting `/pair` sees `None` → 409.

## Boundaries / what does NOT belong here

- Facade state + load/chat/status/tune logic → `../api`.
- Worker process + RPC correlation → `../worker` / `../supervisor.rs`.
- Multi-worker orchestration, served ids, remote fleet/hub → `../node`.
- The bounded/merging delta channel itself → `../delta_queue`.

## Testing

Handler tests build a `Higgs` over a stateful fake `NodeRuntime`
(`test_support.rs`) whose fake worker auto-responds to load/status/chat with no
llama.cpp — so tests drive the REAL load/chat/JIT path (write a GGUF fixture,
load it, assert) instead of hand-driving worker stdio. Unit tests live in the
sibling `*_tests.rs` files (per the crate test-layout rule); `readiness.rs`,
`decide_mint`/`decide_revoke`, `redact_paths`, `fit_generation_budget`, and the
`v1_wire` parity checks are the pure seams. End-to-end coverage (real process,
real model over HTTP) lives in `tests/` (the integration gate).
