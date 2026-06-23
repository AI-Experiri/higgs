# `api/` — the `Higgs` host-facing facade

`Higgs` is the in-process handle a host app holds: one instance per app. It owns the
facade-level state (config, the load/unload lifecycle mutex, the inference admission
gate) and delegates worker-process management + RPC correlation to the `Supervisor`
(and, for remote models, to the `HubFleet`).

This was one ~2,300-line `api.rs`; it is split into focused files. `api.rs` remains the
module root — it defines the `Higgs` struct and its core lifecycle methods, declares the
submodules, and **re-exports** their public items so every existing `crate::api::*` path
keeps resolving unchanged.

## File map

| File | Responsibility |
|------|----------------|
| `api.rs` | The `Higgs` struct + its core `impl` (construct, `load`/`unload`/`status`/`chat_stream`/`scan`/`probe_support`/`events`/`logs`/`sysinfo`/`stop`/`start`), plus submodule declarations and re-exports. |
| `api/types.rs` | Wire/response types (`HiggsConfig`, `HiggsStatus`, `LoadedInfo`, `ChatOutcome`, server config, …, all via `higgs_ts!`), the runtime constants (`DEFAULT_CTX_CAP`, `MAX_CONCURRENT_INFERENCE`, idle-TTL constants, …), the `Support*` type aliases, and `chat_outcome_from_value`. |
| `api/guards.rs` | Host-side load guards: `validate_repo_id` (charset/traversal), `path_within_roots` (containment), and `guard_memory_headroom` (pre-load RAM check) + its helpers. |
| `api/reaper.rs` | The engine-level idle auto-unload reaper loop (`idle_reaper`). |
| `api/tests.rs` | The crate's `#[cfg(test)] mod tests` for the facade. Reaches facade internals via `use super::*` (child modules can see parent-private items). |

## Re-exports

`api.rs` carries `pub use types::*;` plus targeted `pub(crate) use` for the guards and the
reaper. Public constants/types stay public; internal helpers stay `pub(crate)`. So callers
in `serve/`, `node/`, and the standalone binary continue to import e.g.
`crate::api::{Higgs, DEFAULT_CTX_CAP, guard_memory_headroom, path_within_roots}` exactly as
before — the split is invisible outside `api/`.

See `DESIGN.md` for the rationale and boundaries.
