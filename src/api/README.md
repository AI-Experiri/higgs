# `api/` — the `Higgs` host-facing facade

`Higgs` is the in-process handle a host app holds: one instance per app. It owns the
facade-level state (config, the load lifecycle mutex, the inference admission gate, the
serve-layer toggles) and delegates worker spawn/load/unload/status/chat, idle auto-unload,
and the Developer-Log bus to a co-located **multi-worker `NodeRuntime`** — the same engine
remote nodes run (P4b). A remote-resident model routes through the `HubFleet` instead.

This was one ~2,300-line `api.rs`; it is split into focused files. `api.rs` remains the
module root — it defines the `Higgs` struct and its core lifecycle methods, declares the
submodules, and **re-exports** their public items so every existing `crate::api::*` path
keeps resolving unchanged.

## File map

| File | Responsibility |
|------|----------------|
| `api.rs` | The `Higgs` struct + its core `impl` (construct, `load`/`unload`/`status`/`chat_stream`/`scan`/`events`/`logs`/`sysinfo`/`stop`/`start`, the served-id helpers `local_served`/`local_served_ids`/`local_loaded_info`, and `with_config_mut` — the lock-serialized read-modify-write of `config.json` for per-model load records + instance rename), plus submodule declarations and re-exports. Delegates to the `local: Arc<NodeRuntime>` field. On a successful `load`, persists the effective load params to `config.json` (per-model `ModelRecord`) so the UI can show what a model was loaded with — best-effort, never failing a good load. |
| `api/types.rs` | Wire/response types (`HiggsConfig`, `HiggsStatus`, `LoadedInfo`, `ChatOutcome`, server config, …, all via `higgs_ts!`), the runtime constants (`DEFAULT_CTX_CAP`, `MAX_CONCURRENT_INFERENCE`, idle-TTL constants, …), and `chat_outcome_from_value`. |
| `api/guards.rs` | Host-side load guards: `validate_repo_id` (charset/traversal), `path_within_roots` (containment), and `guard_memory_headroom` (pre-load RAM check) + its helpers. |
| `api/tests.rs` | The crate's `#[cfg(test)] mod tests` for the facade. Reaches facade internals via `use super::*` (child modules can see parent-private items). |

> Idle auto-unload moved INTO the node (`node/runtime.rs`) in P4b, so the old
> `api/reaper.rs` engine-level reaper loop was removed; the facade's idle settings just
> mirror into the node's live `IdleConfig`.

## Re-exports

`api.rs` carries `pub use types::*;` plus targeted `pub(crate) use` for the guards. Public
constants/types stay public; internal helpers stay `pub(crate)`. So callers in `serve/`,
`node/`, and the standalone binary continue to import e.g.
`crate::api::{Higgs, DEFAULT_CTX_CAP, guard_memory_headroom, path_within_roots}` exactly as
before — the split is invisible outside `api/`. `node::runtime` reuses `guard_memory_headroom`
and `path_within_roots` for the node's load path (one guard implementation, two callers).

See `DESIGN.md` for the rationale and boundaries.
