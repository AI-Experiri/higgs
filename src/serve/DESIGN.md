# `serve/` — design notes

## One surface: strict OpenAI `/v1`

higgs is library-first. Control (scan, load/unload, tune, estimate, status,
settings, keys, hub/fleet) runs **in-process** through the [`Higgs`](../api)
facade — there is no HTTP control plane. The only thing served over a socket is
the strict OpenAI-compatible **`/v1`** (chat + models) plus a `/health` probe.

`/v1` is the **untrusted interop boundary.** It must match the OpenAI shape
byte-for-byte (clients are third-party SDKs), so bodies are `async-openai` types
verbatim, errors use the OpenAI `{"error":{message,type,code}}` envelope
(`v1::v1_envelope`), and that message is **path-redacted** (`v1::redact_paths`) —
host filesystem paths and `host:port` addresses are stripped so the no-auth
loopback server never leaks its layout. `/v1/models` answers "what can serve chat
right now" (the reachable-id union), never the raw on-disk catalog.

```
  external OpenAI client                embedder (in-process)
          │ HTTP /v1                            │ crate API
          ▼                                     ▼
  ┌────────────────────────────┐      ┌───────────────────────────┐
  │  serve_v1  (mod.rs)        │      │  Higgs facade (../api)    │
  │  ┌──────────────────────┐  │      │  prepare_chat             │
  │  │ CORS → host_guard →  │  │      │  chat_model_ids           │
  │  │ auth_guard → catch → │  │      │  chat_stream, load, tune, │
  │  │ body-limit           │  │      │  status, keys, hub, …     │
  │  └─────────┬────────────┘  │      └──────────┬────────────────┘
  │   POST /v1/chat/completions │                 │ delegates to
  │   GET  /v1/models           │                 ▼
  │   GET  /health              │      ┌───────────────────────────┐
  └─────────┬───────────────────┘      │  serve/control.rs (PURE)  │
            │ v1.rs handlers            │  model_entry, hub_status, │
            │ (thin: validate + map)    │  decide_mint/revoke,      │
            └──────────► Higgs ─────────┤  validate_key_label       │
              prepare_chat /            └───────────────────────────┘
              chat_model_ids /                     │
              chat_stream                          ▼
                                        ../worker · ../node (fleet/hub)
```

The `/v1` handlers are deliberately **thin**. The JIT/local-first routing, the
reachable-id union, and the prompt-fit clamp all live on the facade
(`Higgs::prepare_chat` → `resolve_loaded` + `fit_generation_budget`,
`Higgs::chat_model_ids`), so the in-process embedder and the HTTP endpoint share
ONE implementation and cannot diverge. serve keeps only what is genuinely
`/v1`-wire-shaped: the OpenAI envelope, sampling-range validation, the text-only
content check, tool-call interpretation, and SSE assembly.

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
   in-process host wants no gate). When keys exist, `required_scope(path)` maps
   the route to a `Scope` and the `Authorization: Bearer hgk_…` token must carry
   it; failure is `401 [HG048]` with `WWW-Authenticate: Bearer`.
4. **Catch-panic**, then **body limit** (innermost): a handler panic becomes a
   `500`; a body over `MAX_BODY_BYTES` (32 MiB) is `413` before the handler runs.

### Timeouts

`serve_v1` builds two sub-routers:

- **`/v1/chat/completions`** (streaming) is **un-timed** at the HTTP layer — an
  SSE stream must outlive any per-request bound. Its duration is bounded
  separately by the worker chat-RPC timeout, a different layer.
- **`/v1/models`** (a cheap non-streaming call) carries `CONTROL_TIMEOUT`
  (120 s) so a wedged request can't pin a connection forever.

`CONTROL_TIMEOUT` and `LONG_OP_TIMEOUT` (20 min) are documented `const`s grouped
for a later `HiggsConfig` lift. `LONG_OP_TIMEOUT` once bounded the long
`/api/higgs/models/{load,tune}` HTTP routes; those ops are now facade methods
(each underlying load/decode independently bounded by the worker load/RPC
timeouts), so the const is retained for reference but no longer wraps a serve
route. `MAX_OUTPUT_TOKENS` (absolute `max_tokens` cap) and `PROMPT_BYTES_PER_TOKEN`
(the tokenizer-free lower-bound ratio) are consumed by the facade's
`fit_generation_budget`; serve reads `MAX_OUTPUT_TOKENS` too, as the upper bound
in `validate_sampling`.

### Auth scope model (`required_scope`, fail-closed)

`required_scope` reads only the request **path** (the method no longer
disambiguates a scope — the `/api/higgs/*` routes that needed it are gone):

- `/health` → open (`None`).
- `/v1/chat/completions` → `Chat`.
- `/v1/models` → `Models`.
- **Everything else** under `/v1/` → `Admin`.

That final catch-all is the **fail-closed default**: any future `/v1` route is
Admin-scoped automatically unless explicitly relaxed — a forgotten route can
never accidentally serve open.

## API-key management — the mint / revoke DECISION cores (control.rs)

Key mutations run on the facade (`Higgs::mint_key` / `Higgs::revoke_key`, backed
by `mutate_api_keys`: file read-modify-write under a lock + in-memory hot-swap,
so a minted/revoked key gates the **very next** request — no restart). serve owns
only the PURE, unit-tested decision cores those methods call **inside** the
keystore lock, so the empty-store bootstrap window can't be raced:

- **Bootstrap mint** (empty store) is allowed unauthenticated and MUST grant
  `Admin` — it defaults to `[admin]` when scopes are omitted and rejects explicit
  non-admin scopes (`Mint::BootstrapNeedsAdmin`), because a non-admin first key
  would flip auth ON yet be unable to reach the Admin key API, locking the
  surface out of itself with no recovery path. (`bootstrap` is derived from the
  WHOLE store being empty, hidden internal keys included — an embedder's hidden
  Admin means the store is already manageable, so a first VISIBLE key is not
  force-promoted to Admin.)
- A second unauthenticated mint that reaches the lock after the first key landed
  sees a non-empty store, re-checks its bearer, and is refused
  (`Mint::Unauthorized`). Revoke re-derives authorization the same way
  (`decide_revoke`).
- `trusted` (an in-process facade caller) short-circuits ONLY the bearer branch;
  every structural invariant (duplicate, bootstrap-needs-admin, scope defaults,
  last-admin, last-key-on-LAN) still runs.
- Later omitted-scopes mints default to `[chat, models]`.
- Labels are validated (`validate_key_label`: `1–64` chars from `[A-Za-z0-9._-]`,
  and the literal `.` / `..` are rejected — they pass the charset but URL parsers
  normalize dot-segments away) so a key stays revocable by a single-segment
  label.
- The plaintext token is returned exactly once (mint response) and never logged
  or stored.

### Last-key / last-Admin / LAN interlocks (`[HG058]`/`[HG059]`/`[HG066]`/`[HG069]`)

Guards that keep a keyed LAN bind from silently going open, all enforced against
the REAL bound address (a library caller handing us its own listener can't
bypass them):

- **Startup `[HG058]`**: `serve_v1` refuses to serve a non-loopback listener with
  zero keys (auth off AND host guard relaxed = whole surface open). It tears the
  facade down (`Higgs::stop`, so a pre-`start()`ed worker doesn't leak) and
  returns `LanBindWithoutKeys` as an `io::Error` — a startup refusal, so it does
  **not** flow through `http_status`.
- **Startup `[HG069]`**: `serve_v1` likewise refuses a non-loopback listener
  whose keys are ALL non-Admin — the Admin-only key API would be locked out.
- **Atomicity of the LAN gate**: a non-loopback `serve_v1` runs its `[HG058]`/
  `[HG069]` key checks **and** arms its LAN exposure in one critical section
  (`Higgs::arm_lan_serve`, holding `keys_io`) — the same lock `revoke_key` commits
  under (it reads `lan_exposed()` *inside* `mutate_api_keys`). Without that, a revoke
  could read `lan_exposed() == false`, the serve could pass its key check against the
  not-yet-published store, and the revoke would then empty it: a **keyless listener on
  a LAN**. Serialized, one always loses. Lock order is `keys_io` → `serves`.
- **Runtime `[HG059]`/`[HG066]`**: `serve_v1` registers each live listener with
  `higgs.register_serve(…)`. On a normal exit it calls
  `ServeGuard::release()` — which deregisters **before** the terminal worker drain
  (so a stopped listener never keeps disclosing itself or forcing `[HG059]` while
  workers shut down) and reports whether it was the **last** listener; only the last
  one calls `higgs.stop()`, since draining the shared node while a sibling still
  serves would strand it. On task **cancellation** `release` never runs and the
  guard's `Drop` deregisters instead (an aborted future runs destructors but no code
  past its await, so an explicit-only call would leak the registration). Serve state is
  per-listener, not flat slots: `lan_exposed` is "any live non-loopback listener"
  (so a loopback serve starting, or a sibling exiting, can't strip a live LAN
  listener's protection); `bind_host` discloses the **primary**
  (first-registered) listener. The extra-CORS allowlist is NOT per-listener (G7):
  every layer reads the facade's LIVE list per request, `serve_v1` publishes the
  persisted list at listener start, `set_cors_origins` publishes as it persists —
  so an API write applies immediately and `restart_required` is only "the
  persisted file diverged from the live list" (a hand-edit, or the loser of
  two concurrent API writes whose persist/publish orders inverted — either
  way disclosed, and the next write or serve start reconciles). A rebind's
  zero-listener moment is covered by `Higgs::reserve_rebind` (an RAII count the
  last-listener stop() decision respects; [HG073] refuses a stopped facade). `decide_revoke`
  refuses to revoke the
  LAST key while `lan_exposed` (`Revoke::LastKeyOnLan` → `409`), and refuses to
  revoke the last VISIBLE Admin key while other visible keys remain
  (`Revoke::LastAdminKey` → `409`) — the runtime counterparts of the startup
  guarantees. Both operate over `ks.visible()` so a hidden internal token never
  skews the count.

`v1_router_with_host_policy` is `pub(crate)` — never `pub` — precisely so the
relaxed-host policy (`enforce_loopback_host = false`) is reachable ONLY after
these guards have run.

## JIT / local-first routing (thin serve, facade-owned logic)

The local node is multi-model, so `/v1` is genuinely multi-model and
**local-first** (consistent with `../api/DESIGN.md`). serve now only maps the
facade's decisions:

- **`v1_models`** maps `Higgs::chat_model_ids()` — the reachable-id union
  (`local_served_ids` ∪ the JIT-servable catalog ids while `jit_enabled` ∪ the
  fleet's routed models, minus a transiently-benchmarking local candidate) — onto
  the OpenAI `Model` envelope. Never the raw catalog: unprepared/stale/oversized
  models are hidden because the JIT gate would refuse them. Nothing reachable ⇒
  empty list (the correct OpenAI answer), so it never gates on worker liveness.
- **`gate_and_validate`** calls `Higgs::prepare_chat`, which owns the gate:
  `resolve_loaded` resolves LOCAL-first (already-served id wins with no reload;
  else a fleet-remote model returns a permissive placeholder whose remote
  `[HG005]` is the prompt-fit backstop; else with JIT off it is `404 [HG003]`;
  with JIT on the id must be scanned (`404 [HG002]` else), must pass the readiness
  gate (`Missing → [HG046]`, `Stale → [HG047]`, both `400`), then loads the
  VALIDATED profile). It returns the **resolved model id** — which binds dispatch,
  so the worker rejects (`[HG018]`) if a concurrent JIT load swaps the model out
  before generation — plus the context-clamped generation budget.

### Request validation on `/v1` (what serve still checks itself)

- **Serving toggle**: `!serving_enabled` short-circuits to `[HG019]` → 503 in
  `v1_chat_completions` BEFORE any facade gate or worker RPC, so re-enabling
  serving is always reachable through the crate API.
- **Sampling** (`validate_sampling`, ranges mirror vllm): `temperature ≥ 0`,
  `top_p ∈ (0,1]`, `n == 1` (higgs serves one choice), penalties `∈ [-2,2]`,
  `max_tokens ∈ [1, MAX_OUTPUT_TOKENS]` → `400 [HG013]`.
- **Content**: `/v1` is text-only — image/audio/file/refusal parts →
  `400 [HG049]` via `messages_to_pairs`. The flattened pairs are used only for
  validation; the RAW `messages` JSON rides to the template verbatim so extension
  fields (assistant `reasoning_content`, `chat_template_kwargs`) survive. A
  non-object `chat_template_kwargs` is ignored with a `[HG055]` warn trace
  (matching llama.cpp's own lenient handling).
- **Prompt fit** (facade `fit_generation_budget`, NOT serve): the tokenizer-free
  lower bound (`prompt_bytes / PROMPT_BYTES_PER_TOKEN`) clamps an oversized
  `max_tokens` to what fits and errors `[HG005] context_length_exceeded` only when
  the prompt ALONE overflows. The worker's tokenizer-exact `[HG005]` is the
  authoritative backstop; serve maps both to `context_length_exceeded` via
  `v1_error_code`.

## Streaming (stream.rs) + G1 bounded delta queue

For `stream: true`, `chat_sse` spawns `assemble`, which drains the `chat_stream`
`(DeltaReceiver, outcome)` pair onto OpenAI chunks: assistant-role chunk →
reasoning/content/tool-call deltas (engine order, reasoning first) → optional
terminal tool-call chunk → finish chunk → `data: [DONE]`, plus a final usage-only
chunk when `stream_options.include_usage` is set. A mid-stream `HiggsError` is
emitted as an OpenAI error data frame (OpenAI's own convention) before `[DONE]`;
a `JoinError` becomes `[HG044] ChatTaskFailed`.

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

**Never leave an SSE stream open across shutdown** — it blocks the graceful
drain. (The `/api/higgs/logs/stream` and `/api/higgs/events` control SSE streams
no longer exist on any socket; model-load progress is a PUSH broadcast an
embedder consumes in-process via `Higgs::subscribe_load_events`.)

## `http_status` — the one status table

`http_status` is the single `HiggsError → StatusCode` map, shared by the `/v1`
handlers and the SSE error path (so a streaming `[HG005]` surfaces
`context_length_exceeded` on both paths, via `v1_error_code`). Notable arms:

- `HG002`/`HG003` → 404; `HG005`/`HG013`/`HG015`/`HG046`/`HG047`/`HG050` → 400;
  `HG006`/`HG007`/`HG014`/`HG017`/`HG018`/`HG019` → 503; `HG016` → 504; `HG037`
  → 501; `HG038`/`HG039` → 502; `HG040`–`HG044` → 500 (default).
- `HG048` (`Unauthorized`) → 401.
- **`HG059` (`LastKeyOnLan`), `HG066` (`LastAdminKey`), `HG064` (`BenchCancelled`),
  and `HG067` (`BenchModelLoaded`) → 409** — refused state transitions the client
  resolves (mint a replacement key / unload the model first).
- **`HG060` (`LoadOomExhausted`), `HG063` (`BenchExhausted`), `HG068`
  (`BenchInProgress`), and `ServingDisabled`/`NodeUnreachable` → 503** — a load
  that exhausted the OOM degrade ladder / a benchmark that found no fitting config
  / a model owned by a running benchmark / a down or unreachable node are all
  retryable capacity signals, not client errors.
- `RpcMethodNotFound` → 501; `ProtocolViolation`/`HubRequestRejected` → 502.
- `WorkerRpc` maps by its carried `worker_code` so a worker-origin (or remote-
  node-propagated) code keeps its true status across the process boundary
  (`HG002/003→404`, `HG005/HG050→400`, `HG006/007/017/018/060→503`, `HG016→504`,
  `HG037→501`, `HG038→502`, else 500).

## HGxxx codes this module owns

Codes RAISED in serve code:

| Code | Where | Status |
|------|-------|--------|
| `HG012` `ForbiddenHost` | `host_guard` | 403 |
| `HG048` `Unauthorized` | `auth_guard` / `unauthorized()` | 401 |
| `HG049` `InvalidRequest` | `v1_bad_request` (malformed body) | 400 |
| `HG072` `InvalidKeyRequest` | keystore validation: `validate_key_label`, the mint scope/duplicate/bootstrap rules, revoke of an unknown label | 400 |
| `HG013` `InvalidSamplingParam` | `validate_sampling` | 400 |
| `HG019` `ServingDisabled` | `v1_chat_completions` serving gate | 503 |
| `HG055` | `chat_template_kwargs` not a JSON object | warn-only (ignored) |
| `HG056` | malformed streamed tool-call fragment | warn-only (dropped) |
| `HG057` `ChatStreamOverflow` | `stream::assemble` | SSE error frame (envelope 500) |
| `HG044` `ChatTaskFailed` | `stream::assemble` JoinError | 500 |
| `HG058` `LanBindWithoutKeys` | `serve_v1` startup | refusal (`io::Error`, not HTTP) |
| `HG069` `LanBindWithoutAdminKey` | `serve_v1` startup (keys present, none Admin) | refusal (`io::Error`, not HTTP) |
| `HG059` `LastKeyOnLan` | `decide_revoke` (control.rs, via facade) | 409 |
| `HG066` `LastAdminKey` | `decide_revoke` (control.rs, via facade) | 409 |
| `HG040` `PersistenceFailed` | `keystore_io_error` (control.rs, via facade) | 500 |

Codes serve only MAPS (raised in `../api` / `../worker` / `../node`, reaching
clients through `http_status`): `HG002`/`HG003` (facade `resolve_loaded`),
`HG005` (facade `fit_generation_budget` + worker), `HG046`/`HG047` (facade
readiness gate), `HG050`, `HG060`/`HG063`/`HG064`/`HG067`/`HG068` (load ladder /
turbotune), and the worker/node-propagated `WorkerRpc` codes.

## Concurrency / locking

The serve layer is mostly stateless request handlers; the interesting locks live
on the facade and are exercised through the control cores here:

- **Keystore mutations** (facade `mutate_api_keys`) run `decide_mint` /
  `decide_revoke` and the mutation atomically inside the store lock — closing the
  bootstrap TOCTOU (two unauthenticated mints while empty; only the one that still
  finds it empty bootstraps, the loser re-checks its bearer and is refused). It
  persists + re-installs ONLY when the closure actually changed the store, so a
  rejected mint/revoke does no disk write and can't turn an intended 401/400/409
  into an `HG040` 500.

## Boundaries / what does NOT belong here

- Facade state + control logic (scan, load/chat/status/tune, JIT gate, key
  mutation glue, load-event broadcast) → `../api`.
- Worker process + RPC correlation → `../worker` / `../supervisor.rs`.
- Multi-worker orchestration, served ids, remote fleet/hub → `../node`.
- The bounded/merging delta channel itself → `../delta_queue`.

## Testing

Handler tests build a `Higgs` over a stateful fake `NodeRuntime`
(`test_support.rs`) whose fake worker auto-responds to load/status/chat with no
llama.cpp — so tests drive the REAL load/chat/JIT path (write a GGUF fixture,
load it, assert) instead of hand-driving worker stdio. Unit tests live in the
sibling `*_tests.rs` files (per the crate test-layout rule); `readiness.rs`,
`decide_mint`/`decide_revoke`, `redact_paths`, and the `v1_wire` parity checks
are the pure seams. End-to-end coverage (real process, real model over HTTP)
lives in `tests/` (the integration gate).
