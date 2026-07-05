# `api/` — the `Higgs` host-facing facade

`Higgs` (in `api.rs`) is the single in-process handle a host app holds — one instance
per app. It is the typed facade over a co-located LOCAL
[`NodeRuntime`](../node/runtime.rs) (the same multi-worker engine remote nodes run,
P4b). `Higgs` owns the facade-level state — the live `HiggsConfig`, the `lifecycle`
mutex that serializes loads, the `inference_gate`/`remote_gate` admission semaphores,
the runtime toggles, and the Turbotune `benchmarking` exclusivity set — and delegates worker
spawn/load/unload/status/chat, idle auto-unload, RPC correlation, and the Developer-Log
bus to the node. A model that is resident on a REMOTE node routes through the optional
`fleet` instead; the facade is **local-first** everywhere (a locally-resident served id
always wins over a remote one of the same name).

`api.rs` is the module root: it defines the `Higgs` struct + its core `impl`, declares
the `guards`/`types` submodules, and **re-exports** their public items (`pub use
types::*;`, `pub(crate) use guards::{…}`) so every existing `crate::api::*` path keeps
resolving unchanged after the split.

## File map

| File | Responsibility |
|------|----------------|
| `api.rs` | The `Higgs` struct (all facade state) + its core `impl`: construction (`new`/`with_log_bus`/`with_local`), lifecycle (`start`/`stop`/`load`/`unload`/`unload_one`/`status`/`scan`/`chat_stream`), local routing (`local_served`/`local_served_ids`/`local_node_view`/`local_loaded_info`/`servable_model_ids`), events/logs (`events`/`logs`/`subscribe_logs`/`subscribe_load_events`), the runtime toggles/getters, config/tune surface (`server_config`/`sysinfo`/`hardware`/`tune`/`estimate`/`model_readiness`/`profile_state`/`with_config_mut`), and the **G6 Turbotune** engine (`turbotune_bench`/`measure_gen_tps`/`bench_unload_id` + the exclusivity helpers `is_benchmarking`/`begin_benchmark`). Also the module's free helpers: `run_oom_ladder`, `run_benchmark`, `node_params_for`, `overlay_sampling`, `profile_stale`, `file_sig`, `now_unix_ms`, the `ProfileState` enum, and the `BenchmarkingGuard` RAII guard. Declares the submodules and re-exports them. |
| `api/types.rs` | Wire/response types + the runtime constants. `higgs_ts!` structs (`HiggsConfig`, `HiggsLimits`, `HiggsServerConfig`, `LoadedInfo`, `ModelLoading`, `ModelLoadEvent`, `HiggsStatus`), the `higgs_const_enum!` `ModelLoadPhase`, the `ChatOutcome` decode target + `chat_outcome_from_value`, and the consts (`DEFAULT_CTX_CAP`, `BIND_HOST`, `MAX_CONCURRENT_INFERENCE`, `MEMORY_HEADROOM_FRACTION`, `MAX_IDLE_TTL_MINUTES`). Pure data + constants, no behavior. |
| `api/guards.rs` | Pure host-side load guards, no `Higgs` state: `validate_repo_id` (charset/`..`-traversal → `InvalidModelId` HG015), `path_within_roots` (canonicalized containment), `guard_memory_headroom` (pre-load RAM headroom → `InsufficientMemory` HG017) + its helpers `available_system_memory`/`fits_in_memory`. |
| `api/tests.rs` | The facade's `#[cfg(test)] mod tests` (child module, `use super::*`), wired from `api.rs`. Not covered here. |
| `README.md` / `DESIGN.md` | This file / the rationale + invariants. |

## Public surface

**Types & constants** (re-exported from `types.rs`; some cross the `bindings/` TS
boundary via `higgs_ts!`/`higgs_const_enum!`):
`HiggsConfig`, `HiggsServerConfig`, `HiggsLimits`, `HiggsStatus`, `LoadedInfo`,
`ModelLoading`, `ModelLoadEvent`, `ModelLoadPhase`, `ChatOutcome`; consts
`DEFAULT_CTX_CAP` (32768), `BIND_HOST` (`127.0.0.1`), `MAX_CONCURRENT_INFERENCE` (8),
`MEMORY_HEADROOM_FRACTION` (0.8), `MAX_IDLE_TTL_MINUTES`.

**Guards** are `pub(crate)`: `guard_memory_headroom`, `path_within_roots` (both reused
by `node::runtime`'s load path — one implementation, two callers). `fits_in_memory` is
re-exported only under `#[cfg(test)]`.

**`Higgs` methods** the crate leans on:
- Construction: `new` / `with_log_bus` (share the caller's `LogBus`) / `with_local`.
- Lifecycle: `load(id, params)`, `unload()` (drain all), `unload_one(served)`,
  `status()`, `scan()`, `stop()`, `start()`, and `chat_stream(model, messages_json,
  max_tokens, sampling, tools_json, chat_template_kwargs)` → a `DeltaReceiver` + a
  `JoinHandle<Result<ChatOutcome, _>>`.
- Routing/listing: `local_served_ids()` (feeds `/v1/models`), `local_node_view(label)`,
  `local_loaded_info(served)`, `servable_model_ids()`.
- Events/logs: `events()` (`HiggsEvent`), `subscribe_load_events()` (SSE
  `ModelLoadEvent`), `subscribe_logs()` / `logs(n, filter)`.
- Config/tune/hardware: `server_config()` (`GET /api/higgs/system`), `sysinfo()`,
  `hardware()`, `tune(req)`, `estimate(req)`, `model_readiness(...)`,
  `profile_state(id)`.
- Installers/toggles: `set_fleet`/`fleet`, `set_hub`/`hub`/`clear_hub`,
  `set_api_keys`/`api_keys`/`mutate_api_keys`, `set_lan_exposed`, and the atomic
  serve-layer toggles `jit_enabled`, `auto_unload_idle`, `idle_ttl_minutes`,
  `serving_enabled`, `log_incoming_tokens`, `verbose`, `log_show_fields`
  (each with a `set_*`).

## How the rest of the crate uses it

`serve/` (the HTTP router) holds an `Arc<Higgs>` and maps every `/v1/*` and
`/api/higgs/*` endpoint onto these methods. `node/` reuses the two `pub(crate)` guards
for its own load path. The standalone `bin/` constructs a `Higgs`, installs keys/hub/
fleet, and calls `serve_with_shutdown`. Because `api.rs` re-exports the submodule items,
all of these import `crate::api::{Higgs, DEFAULT_CTX_CAP, guard_memory_headroom,
path_within_roots, …}` exactly as before the split.

> Idle auto-unload lives INSIDE the node (`node/runtime.rs`) since P4b — the old
> `api/reaper.rs` engine-level loop was removed; the facade's idle toggles just mirror
> into the node's live `IdleConfig`.

See `DESIGN.md` for the rationale, the concurrency model, and the error codes.
