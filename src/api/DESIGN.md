# `api/` — design notes

## Why this module exists

`Higgs` is the single host-facing seam over the runtime. Everything a host or the serve
layer can do — load/unload a model, run a chat, read status/logs/events, toggle runtime
settings, probe model support, query hardware — goes through one `Higgs` value. Keeping that
one entry point is deliberate: hosts depend on a small, stable surface, and the wire types
(`higgs_ts!`) are generated into `bindings/` from here.

## Why it was split

The facade had grown to ~2,300 lines in a single file: ~40% was the inlined test module,
~20% was doc comments, and the rest mixed wire types, constants, the 16-field struct, ~16
trivial config-toggle methods, the load/chat lifecycle, path/memory guards, and the idle
reaper. That breadth made the file hard to hold in context and hard to change safely.

The split groups code by **responsibility**, not by layer:

- **`types.rs`** — the data vocabulary (wire types + constants + the chat decoder). These
  change for protocol reasons and have no behavior; isolating them keeps the struct/impl
  files about logic.
- **`guards.rs`** — pure, testable validation/containment/headroom functions with no `Higgs`
  state. They are `pub(crate)` because `node::runtime` reuses `guard_memory_headroom` and
  `path_within_roots` for the node's load path (one guard implementation, two callers).
- **`tests.rs`** — moved verbatim; it is the largest single block and belongs beside, not
  inside, the code.

(An earlier `reaper.rs` held the engine-level idle auto-unload loop; P4b moved idle
auto-unload INTO the node — per-worker, uniform local+remote — so that file was removed.)

`api.rs` keeps the `Higgs` struct and its core lifecycle `impl`, because those are tightly
coupled to the struct's private fields and to each other.

## Invariants preserved by the split

1. **No public-path churn.** `api.rs` re-exports submodule items, so `crate::api::X` paths
   used by `serve/`, `node/`, and `bin/` are unchanged. The split is behavior-preserving and
   was verified by codex convergence + both coverage gates.
2. **Privacy is intentional.** Wire types/constants are `pub` (some cross the bindings
   boundary); the guard helpers (`guard_memory_headroom`, `path_within_roots`) are
   `pub(crate)` (the node reuses them); `fits_in_memory` is re-exported only under
   `#[cfg(test)]` (it has no non-test caller).
3. **Child-module test access.** `tests.rs` uses `use super::*`; a child module can see its
   parent's private items, so the tests still exercise facade internals without widening
   visibility.

## How the facade routes (post-P4b)

`Higgs` holds `local: Arc<NodeRuntime>` (the co-located multi-worker node) and an optional
`fleet: HubFleet` (remote nodes). The facade is **local-first** everywhere — listing,
loaded-model gating, and chat routing all prefer a locally-resident model over a remote one
of the same served id, so a model the user loaded locally is always reachable:

- **`load`** is additive on the node (one worker per model) and deduped by raw id at the
  facade; **`unload`** drains every local worker; **`status`** reports the PRIMARY (lowest
  worker id) instance.
- **`chat_stream`** resolves a SERVED id → `(worker, raw model)` via
  [`local_served`], leases that worker, and sends the RAW model on the wire; if not local it
  falls through to the fleet.
- **`local_served_ids`** feeds `/v1/models` (∪ the fleet's routed models).

Idle auto-unload, worker spawn/RPC correlation, and the Developer-Log bus live in the node,
not the facade.

## Boundaries / what does NOT belong here

- Worker-process lifecycle, RPC correlation, restart FSM → `supervisor.rs`.
- Multi-worker orchestration + the node idle reaper → `node/runtime.rs`.
- Remote fleet routing + served-instance ids → `node/fleet.rs`, `node/served.rs`.
