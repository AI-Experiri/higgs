# `serve/` — design notes

## Why two surfaces

higgs serves the same engine through two HTTP surfaces with deliberately different contracts:

- **`/v1` is the untrusted interop boundary.** It must match the OpenAI shape byte-for-byte
  (clients are third-party SDKs), so bodies are `async-openai` types verbatim, errors use the
  OpenAI `{"error":{message,type,code}}` envelope, and that message is **path-redacted** —
  host filesystem paths and `host:port` addresses are stripped so the no-auth loopback server
  never leaks its layout. `/v1/models` answers "what can serve chat right now" (loaded only),
  never the on-disk catalog.
- **`/api/higgs/*` is our own control surface.** Shapes are higgs's own (`wire.rs`), error
  text is the full unredacted `HiggsError` Display (it's ours), and it carries everything the
  UI needs: the on-disk catalog with support verdicts, load/unload, status, system info, the
  Developer-Log snapshot + live SSE stream, runtime settings, and hub/fleet management.

Keeping them separate means the OpenAI surface stays strict and minimal while the control
surface can evolve freely. There is no second URL/port — one `router` fronts both.

## Middleware order (mod.rs)

Every request passes the same policy stack before a handler runs. In Tower the LAST
`.layer(...)` wraps the earlier ones, so request entry runs **outermost-first** — the reverse
of the source order in `mod.rs` (which lists `DefaultBodyLimit`, `CatchPanicLayer`,
`auth_guard`, `host_guard`, `local_cors`). Request-entry order is therefore:

1. **CORS** (outermost): only local origins (tauri/localhost/loopback) are allowed.
2. **Host guard** (DNS-rebinding defense): the `Host` header must be loopback; a foreign host
   is `403`. higgs is localhost-only by contract — this is the #1 protection and runs
   **before** auth, so a non-loopback request is rejected before any key work.
3. **Auth + scope**: `required_scope(method, path)` maps each route to a `Scope`
   (Chat/Models/Admin/none); when an API-key store is installed, the Bearer token must carry
   that scope. The default empty store = auth OFF (the embedded in-process host wants no gate).
4. **Catch-panic**, then **body limit** (innermost): a handler panic becomes a `500`; an
   oversized body is `413` before the handler runs.

`http_status` is the single `HiggsError → StatusCode` table, including the per-worker-code
mapping (a `WorkerRpc` error maps to its origin code's status, e.g. HG002→404, HG005→400,
HG018→503, unknown→500) so a worker-reported error keeps its semantics across the process
boundary.

## Local-first, multi-model routing (post-P4b)

The local node is multi-model, so the `/v1` surface is genuinely multi-model and **local-first**
— consistent with the facade (see `../api/DESIGN.md`):

- **`v1_models`** lists `higgs.local_served_ids()` (every resident local instance, by served
  id) ∪ the fleet's routed models (skipping name collisions). A locally-loaded model is always
  listed.
- **`ensure_loaded`** (the chat gate) checks the LOCAL served set first (`local_loaded_info`),
  then the fleet, then — with JIT on — additively loads a scanned-but-unloaded model. JIT off
  is the explicit-load HG003 404. A suffixed served id (`org/model-1`) is never JIT-loaded (it
  only addresses an already-resident extra instance).
- **`control_models`** flags EVERY resident model loaded (the full served set), not just the
  primary; the singular `loaded_id` field stays the primary for back-compat.

## Streaming (stream.rs)

The chat path returns `(delta receiver, outcome handle)` from the facade. For `stream: true`,
`chat_sse` assembles SSE frames (role chunk → one content delta per received token → finish
chunk → `[DONE]`), with the verbose serving line emitted once the final outcome is known and an
optional terminal usage chunk for `stream_options.include_usage`. The token deltas flow through
the facade's channel sink, never through a mailbox — the P5 sink handoff. Never leave an SSE
stream open across a shutdown: it blocks the server's graceful drain.

## Testing

Handler tests build a `Higgs` over a stateful fake `NodeRuntime` (`test_support.rs`) whose fake
worker auto-responds to load/status/chat with no llama.cpp — so tests drive the REAL load/chat
path (load a fixture model, then assert) instead of hand-driving worker stdio. "Nothing loaded"
is the natural idle state (no resident worker), so there is no separate idle-supervisor seam.
End-to-end coverage (real process, real model) lives in `tests/` (the integration gate).

## Boundaries / what does NOT belong here

- The facade state + load/chat/status logic → `../api`.
- Worker process + RPC correlation → `../supervisor.rs`.
- Multi-worker orchestration, served ids, remote fleet → `../node`.
