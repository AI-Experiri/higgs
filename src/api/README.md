# `api/` — the `Higgs` facade (higgs's control surface)

`Higgs` (in `api.rs`) is the single in-process handle a host app holds — one instance
per app. It is BOTH the typed facade over a co-located LOCAL
[`NodeRuntime`](../node/runtime.rs) (the same multi-worker engine remote nodes run,
P4b) AND **higgs's only control plane**: since the `/api/higgs/*` HTTP handlers were
removed, control (load/unload, tune/estimate, status, keys, hub/fleet, settings, logs)
is these `Higgs` methods, called in-process — no HTTP required. The ONLY HTTP surface
higgs still serves is the strict OpenAI `/v1` (chat + models), built by
`serve::v1_router` and served by `serve::serve_v1`.

`Higgs` owns the facade-level state — the live `HiggsConfig`, the `lifecycle` mutex
that serializes loads, the `inference_gate`/`remote_gate` admission semaphores, the
runtime toggles, the swappable installers (`fleet`/`api_keys`/`hub`), and the Turbotune
`benchmarking` exclusivity set — and delegates worker spawn/load/unload/status/chat,
idle auto-unload, RPC correlation, and the Developer-Log bus to the node. A model that
is resident on a REMOTE node routes through the optional `fleet` instead; the facade is
**local-first** everywhere (a locally-resident served id always wins over a remote one
of the same name).

`api.rs` is the module root: it defines the `Higgs` struct + its core lifecycle `impl`,
declares the `embed`/`guards`/`types` submodules, and **re-exports** their public items
(`pub use types::*;`, `pub(crate) use guards::{…}`) so every `crate::api::*` path
resolves unchanged. The in-process control methods themselves live in `embed.rs` as a
second `impl Higgs`.

## File map

| File | Responsibility |
|------|----------------|
| `api.rs` | The `Higgs` struct (all facade state) + its core lifecycle `impl`: construction (`new`/`with_log_bus`/`with_local`), lifecycle (`start`/`stop`/`load`/`load_inner`/`unload`/`unload_one`/`status`/`scan`/`chat_stream`), local routing (`local_served`/`local_served_ids`/`local_node_view`/`local_loaded_info`/`servable_model_ids`), events/logs (`events`/`logs`/`subscribe_logs`/`subscribe_load_events`), the runtime toggles/getters, config/tune surface (`server_config`/`sysinfo`/`hardware`/`tune`/`estimate`/`model_readiness`/`profile_state`/`with_config_mut`), the installers/keys (`set_fleet`/`fleet`, `set_hub`/`hub`/`clear_hub`, `set_api_keys`/`api_keys`/`mutate_api_keys`/`register_internal_token`, `set_lan_exposed`/`lan_exposed`/`arm_lan_serve`/`extra_cors_origins`), the **per-listener serve registry** (`serves`/`next_serve_id`/`serve_lifecycle`/`lan_override` state + `register_serve`/`deregister_serve`/`bound_addr`/`any_serve_live` + the G7 live-CORS slot (`live_cors_snapshot`/`publish_live_cors`/`live_cors_or_init`) and rebind seam (`reserve_rebind`/`RebindReservation`, `rebind_reservations`)), and the **G6 Turbotune** engine (`turbotune_bench`/`measure_gen_tps`/`bench_unload_id` + the exclusivity helpers `is_benchmarking`/`begin_benchmark`). Also the module's free helpers: `run_oom_ladder`, `run_benchmark`, `node_params_for`, `overlay_sampling`, `profile_stale`, `file_sig`, `now_unix_ms`, the `ProfileState` enum, and the `ServeSlot`/`ServeGuard` + `BenchmarkingGuard` RAII guards. Declares the submodules and re-exports them. Bottom-wired `#[cfg(test)] mod tests;` → `tests.rs`. |
| `embed.rs` | The **in-process control plane** — a second `impl Higgs` that carries the behavior of the deleted `/api/higgs/*` HTTP handlers as typed methods returning `Result<_, HiggsError>` (never an HTTP `Response`): the chat gate (`prepare_chat`/`resolve_loaded` + the `fit_generation_budget`/`estimate_prompt_bytes` helpers), the models-list assembly (`model_entries`/`model_by_id`), hub/fleet ops (`hub_enable`/`hub_disable`/`pair`/`node_load`/`node_unload`/`node_retire`/`node_label`/`node_scan`/`node_chat_test`/`nodes`), load/unload shaping (`load_flat`/`unload_spec`), trusted key mint/revoke (`mint_key`/`revoke_key` — the latter reads `lan_exposed()` INSIDE the keystore critical section so the [HG059] gate can't race a LAN serve start; every keystore VALIDATION failure — bad/duplicate label, empty scopes, first-key-needs-admin, revoke of an unknown label — raises [HG072], split from the `/v1` body error [HG049] so a token mint is never told to check the OpenAI chat schema), the extra-CORS-origins surface (`cors_settings`/`set_cors_origins` + the free `validate_cors_origin`/`validate_and_dedup_cors_origins` helpers, which canonicalize to the exact string a browser sends in `Origin` and reject anything else with [HG071]), the log/worker/version wrappers (`logs_settings`/`set_logs_settings`/`worker_stop`/`version`), and the `/v1/models` union (`chat_model_ids`). These delegate to the PURE sub-primitives in [`serve::control`](../serve/control.rs) (`model_entry`, `active_records`, `TuneProfileViews`, `hub_status`, `decide_mint`/`decide_revoke`, `validate_key_label`). Bottom-wired `#[cfg(test)] #[path = "embed_tests.rs"] mod tests;`. |
| `types.rs` | Wire/response types + the runtime constants. `higgs_ts!` structs (`HiggsConfig`, `HiggsLimits`, `HiggsServerConfig`, `LoadedInfo`, `ModelLoading`, `ModelLoadEvent`, `HiggsStatus`), the `higgs_const_enum!` `ModelLoadPhase`, the plain `PreparedChat`/`PairInfo`/`ChatOutcome` structs + `chat_outcome_from_value`, and the consts (`DEFAULT_CTX_CAP`, `BIND_HOST`, `MAX_CONCURRENT_INFERENCE`, `MEMORY_HEADROOM_FRACTION`, `MAX_IDLE_TTL_MINUTES`). Pure data + constants, no behavior. |
| `guards.rs` | Pure host-side load guards, no `Higgs` state: `validate_repo_id` (charset/`..`-traversal → `InvalidModelId` HG015), `path_within_roots` (canonicalized containment), `guard_memory_headroom` (pre-load RAM headroom → `InsufficientMemory` HG017) + its helpers `available_system_memory`/`fits_in_memory`. |
| `tests.rs` / `embed_tests.rs` | The facade's unit tests (`use super::*`), wired as child `mod tests` from `api.rs` and `embed.rs` respectively so both keep private-item access. Not covered here. |
| `README.md` / `DESIGN.md` | This file / the rationale + invariants + diagrams. |

## Public surface

**Types & constants** (re-exported from `types.rs`; some cross the `bindings/` TS
boundary via `higgs_ts!`/`higgs_const_enum!`):
`HiggsConfig`, `HiggsServerConfig`, `HiggsLimits`, `HiggsStatus`, `LoadedInfo`,
`ModelLoading`, `ModelLoadEvent`, `ModelLoadPhase`, `ChatOutcome`, `PreparedChat`,
`PairInfo`; consts `DEFAULT_CTX_CAP` (32768), `BIND_HOST` (`127.0.0.1`),
`MAX_CONCURRENT_INFERENCE` (8), `MEMORY_HEADROOM_FRACTION` (0.8),
`MAX_IDLE_TTL_MINUTES` (100 years, in minutes).

**Guards** are `pub(crate)`: `guard_memory_headroom`, `path_within_roots` (both reused
by `node::runtime`'s load path — one implementation, two callers:
`node/runtime.rs` calls `crate::api::path_within_roots` and
`crate::api::guard_memory_headroom`). `fits_in_memory` is re-exported only under
`#[cfg(test)]`.

**`Higgs` methods** the crate leans on:
- Construction: `new` / `with_log_bus` (share the caller's `LogBus`) / `with_local`.
- Lifecycle: `load(id, params)`, `load_flat(req)` (flat-request shaping), `unload()`
  (drain all), `unload_one(served)`, `unload_spec(id?)`, `status()`, `scan()`,
  `stop()`, `start()`, and `chat_stream(model, messages_json, max_tokens, sampling,
  tools_json, chat_template_kwargs)` → a `DeltaReceiver` + a
  `JoinHandle<Result<ChatOutcome, _>>`. The pre-dispatch gate is `prepare_chat(model,
  max_tokens, messages_json)` → `PreparedChat { resolved_model, max_gen }`.
- Models view: `model_entries()` / `model_by_id(id)` (enriched rows), `chat_model_ids()`
  (the `/v1/models` union), `local_served_ids()`, `servable_model_ids()`.
- Events/logs: `events()` (`HiggsEvent`), `subscribe_load_events()` (broadcast of
  `ModelLoadEvent`), `subscribe_logs()` / `logs(n, filter)`,
  `logs_settings()` / `set_logs_settings()`, `worker_stop()`.
- Config/tune/hardware: `server_config()`, `sysinfo()`, `hardware()`, `tune(req)`,
  `estimate(req)`, `model_readiness(...)`, `profile_state(id)`, `version()`.
- Hub/fleet: `nodes()`, `pair()`, `hub_enable()` / `hub_disable()`, `node_load` /
  `node_unload` / `node_retire` / `node_label` / `node_scan` / `node_chat_test` (one
  short prompt relayed to a served instance ON a specific node via
  `fleet.chat_pinned` — always-remote AND node-pinned at dispatch, so a reply
  proves the live iroh link to that exact node; refusal ladder [HG075] unknown
  node / [HG074] nothing routed / [HG076] bad served operand / [HG077] target
  moved concurrently), plus the installers
  `set_fleet`/`fleet`, `set_hub`/`hub`/`clear_hub`.
- Keys/security: `mint_key(label, scopes?)` / `revoke_key(label)` (trusted, bearer-free
  but every structural invariant intact — `revoke_key` reads `lan_exposed()` INSIDE the
  keystore critical section so the `[HG059]` last-key gate can't race a LAN serve start),
  `set_api_keys`/`api_keys`/`mutate_api_keys`, `register_internal_token` (the embedder's
  hidden in-memory token), `set_lan_exposed` (a manual override, ORed in) /
  `lan_exposed()` (the override OR any live non-loopback listener) / `arm_lan_serve`
  (the `[HG058]`/`[HG069]` LAN startup checks + the exposure arming, in ONE keystore
  critical section).
- CORS: `cors_settings()` / `set_cors_origins(origins)` — the extra browser-origin
  allowlist. Origins are validated AND canonicalized to the exact string a browser
  sends in `Origin` (`validate_cors_origin`, rejecting anything else with `[HG071]`),
  deduped (`validate_and_dedup_cors_origins`), and persisted to `config.json`;
  `extra_cors_origins()` re-canonicalizes on READ (the file is hand-editable).
- Serve-layer toggles (each with a `set_*`): `jit_enabled`, `auto_unload_idle`,
  `idle_ttl_minutes`, `serving_enabled`, `log_incoming_tokens`, `verbose`,
  `log_show_fields`.

## How the rest of the crate uses it

- `serve/` (the `/v1`-only HTTP router) holds an `Arc<Higgs>`: `v1_router` maps
  `POST /v1/chat/completions`, `GET /v1/models`, and `GET /health` onto the facade;
  `serve_v1` binds the listener and runs the `[HG058]`/`[HG069]` startup refusals.
  There are NO `/api/higgs/*` routes — control is the facade methods above.
- `serve/control.rs` holds the PURE sub-primitives (`model_entry`, `hub_status`,
  `decide_mint`/`decide_revoke`, the dual-profile tune views) that `embed.rs` delegates
  to; the split keeps the row/decision logic unit-testable without a `Higgs`.
- `node/runtime.rs` reuses the two `pub(crate)` guards for its own load path.
- The `higgs` binary (`src/bin/higgs.rs`) is **node-only** (`--node` daemon, `node`,
  `link`, `keys` subcommands); it does not serve control over HTTP. An embedder
  (jigglebot) constructs `Higgs`, drives control through these methods in-process, and
  — if it wants an external OpenAI surface — serves `/v1` via `serve_v1`.

> Idle auto-unload lives INSIDE the node (`node/runtime.rs`) since P4b — the old
> `api/reaper.rs` engine-level loop was removed; the facade's idle toggles just mirror
> into the node's live `IdleConfig`.

See `DESIGN.md` for the rationale, the control/data-flow and load-phase diagrams, the
concurrency model, and the error codes.
</content>
</invoke>
