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
  `path_within_roots` for the remote load path (one guard implementation, two callers).
- **`reaper.rs`** — the one long-running background loop. It only needs `Weak<Higgs>` +
  three constants, so it carries no facade internals and reads as a self-contained policy.
- **`tests.rs`** — moved verbatim; it is the largest single block and belongs beside, not
  inside, the code.

`api.rs` keeps the `Higgs` struct and its core lifecycle `impl`, because those are tightly
coupled to the struct's private fields and to each other.

## Invariants preserved by the split

1. **No public-path churn.** `api.rs` re-exports submodule items, so `crate::api::X` paths
   used by `serve/`, `node/`, and `bin/` are unchanged. The split is behavior-preserving and
   was verified by codex convergence + both coverage gates.
2. **Privacy is intentional.** Wire types/constants are `pub` (some cross the bindings
   boundary); guards/reaper helpers are `pub(crate)`; `fits_in_memory` is re-exported only
   under `#[cfg(test)]` (it has no non-test caller).
3. **Child-module test access.** `tests.rs` uses `use super::*`; a child module can see its
   parent's private items, so the tests still exercise facade internals without widening
   visibility.

## Boundaries / what does NOT belong here

- Worker-process lifecycle, RPC correlation, restart FSM → `supervisor.rs`.
- Multi-worker orchestration + the node idle reaper → `node/runtime.rs`.
- Remote fleet routing + served-instance ids → `node/fleet.rs`, `node/served.rs`.

## Forward note (P4b)

The actor-runtime migration's P4b replaces `Higgs`'s direct `Supervisor` with an in-process
`NodeRuntime` behind a `NodeHandle (Local | Remote)` seam (unified local+remote routing,
multi-model local). That change lands in `api.rs` (the struct + lifecycle); this split is a
prerequisite that makes it tractable. See
`docs/superpowers/plans/2026-06-22-p4b-local-as-noderuntime.md`.
