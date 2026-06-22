# P4b — Local runs on a NodeRuntime (full multi-model local) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Every task ends green (builds, `scripts/quality.sh` passes) and is codex-converged (3 stable rounds per CLAUDE.md) before the next. Steps use `- [ ]`.

**Goal:** Replace the `Higgs` facade's single direct `Supervisor` with an in-process multi-worker `NodeRuntime`, so local inference behaves exactly like a remote node — concurrent multi-model serving, deterministic served instance ids, and one uniform routing path (`NodeHandle { Local | Remote }`). Remove the engine-level `idle_reaper` (the NodeRuntime's per-worker reaper from P4a now covers local too).

**Architecture:** The Engine (`Higgs`) owns `local: Arc<NodeRuntime>` (in-process, multi-worker) and `fleet: Option<Arc<HubFleet>>` (remote). A `NodeHandle (Local | Remote)` seam resolves a *served instance id* to either a local worker or a remote fleet route; `/v1` chat/models and `/api/higgs/*` never branch inline on local-vs-remote. Served ids are computed **globally** over the union of local + remote instances with the same collision-free greedy algorithm already in `fleet.rs::served_ids` (factor it out and reuse).

**Tech stack:** Rust, the existing `actor.rs` runtime, `NodeRuntime` (P1+P4a), `HubFleet` (P3).

**Status decision (locked):** "Full multi-model local" — `/v1/models` lists every local + remote served id; `/v1/chat` targets a served id resolving to an exact worker; local `load` is additive; local JIT (`ensure_loaded`) is idempotent (load only if no local instance of that raw model exists).

---

## File structure

- `src/node/runtime.rs` — add the facade-needed surface to `NodeRuntime`: `instances()` (Vec<(WorkerId, raw_model)>), `events()` broadcast, `bus()`/log accessors, `probe_paths` (via a transient Supervisor from the spawner), and a `status(worker)` already exists. Add `subscribe_logs` already exists (per-worker relay) — but the engine wants the *bus* log stream; expose `bus`.
- `src/node/served.rs` (new, small) — pull the collision-free served-id algorithm out of `fleet.rs` into a shared `served_ids(instances: impl Iterator<Item=(Loc, WorkerId, &str)>) -> HashMap<String, (Loc, WorkerId)>` generic over a `Loc` key, so both the fleet and the engine reuse it. Re-point `fleet.rs::served_ids` at it.
- `src/api.rs` — the bulk: replace `sup: Arc<Supervisor>` with `local: Arc<NodeRuntime>` + a kept `bus: Arc<LogBus>`; add `NodeHandle`; rewrite `load`/`unload`/`status`/`loaded_id`/`chat_stream`/`scan`/`probe_support`/`events`/`logs`/`subscribe_logs`/`sysinfo` + the log-verbose toggles to go through `local`/`fleet`/`NodeHandle`; delete `idle_reaper` + `last_activity`/`inference_gate`-as-reaper-coordination (keep `inference_gate` as the admission gate); wire `idle_ttl_minutes` into the local `NodeConfig.idle_ttl`.
- `src/serve/v1.rs` — `/v1/models` lists unified served ids; `ensure_loaded` JIT becomes idempotent-local; `/v1/chat` resolves a served id via `NodeHandle`.
- `src/serve/control.rs` — `load`/`unload`/`status` handlers move to served-id / multi-model semantics; local appears in `nodes_view` as a node (optional; or a separate local view).
- Tests: `api.rs` (`load_*`, `status_*`, `chat_stream_*`, `reaper_*` → delete the 4 engine-reaper tests, they move to runtime.rs which already has them), `serve/*` integration, bindings.

## NodeHandle seam

```rust
enum NodeHandle {
    Local { rt: Arc<NodeRuntime>, worker: WorkerId },
    Remote { fleet: Arc<HubFleet>, served: String },
}
impl NodeHandle {
    async fn chat(self, msgs, max, temp, tools) -> Result<(Receiver, impl Future), HiggsError> { … }
}
```
Engine resolves `served_id -> NodeHandle` via the global served map (local instances + `fleet` instances), then dispatches. Chat to a local handle = `rt.chat_handle(worker)` + `lease.chat(...)` (P4a lease keeps it reaper-safe); remote = `fleet.chat(served, …)`.

---

## Tasks

### Task 1: Extract the shared served-id algorithm
**Files:** Create `src/node/served.rs`; Modify `src/node/fleet.rs` (`served_ids`), `src/node/mod.rs` (module decl).
- [ ] Write a unit test in `served.rs`: three instances of `org/m` across two locations + a literal `org/m-1` → four unique, deterministic served ids (port the `fleet.rs` collision test, generalized over a `Loc`).
- [ ] Implement `pub(crate) fn served_ids<L: Ord + Clone>(instances: &[(L, WorkerId, String)]) -> HashMap<String,(L,WorkerId)>` (the greedy sorted assignment from `fleet.rs`).
- [ ] Re-point `fleet.rs::served_ids` to call it with `L = NodeKey`. Run `cargo test --lib node::`.
- [ ] Quality gate + codex convergence. Commit.

### Task 2: NodeRuntime facade surface
**Files:** `src/node/runtime.rs` (+tests).
- [ ] Add `instances(&self) -> Vec<(WorkerId, String)>` (message over the registry; reuse `snapshot_workers`).
- [ ] Add an `events()` broadcast to `NodeActor` and `emit` `ModelLoaded`/`ModelUnloaded` on commit/unload (carry the raw model id).
- [ ] Add `bus(&self) -> Arc<LogBus>` accessor (the node's `config.bus`), and node-level `log_verbose`/`set_log_verbose`/`set_worker_verbose`/`log_show_fields`/`set_log_show_fields` proxied to the bus.
- [ ] Add `probe_paths(&self, paths) -> Vec<verdict>` that spawns a transient Supervisor via the spawner (mirrors `Supervisor::probe_paths`).
- [ ] Unit tests for `instances`, `events`, `probe`. Quality + convergence. Commit.

### Task 3: Engine owns a local NodeRuntime + NodeHandle (data plane)
**Files:** `src/api.rs`.
- [ ] Replace `sup: Arc<Supervisor>` with `local: Arc<NodeRuntime>` (built with `idle_ttl` from the engine's idle settings) and keep `bus: Arc<LogBus>`.
- [ ] Add `NodeHandle` and an engine `resolve_served(&self, served) -> Option<NodeHandle>` combining `local.instances()` + `fleet` instances through `served::served_ids`.
- [ ] Rewrite `chat_stream` to resolve a `NodeHandle` and dispatch (local lease path / remote fleet path), keeping the admission `inference_gate`/`remote_gate` policy.
- [ ] Delete `idle_reaper` + its spawn in `start`; delete the 4 reaper unit tests (covered by runtime.rs P4a tests). Keep `last_activity` only if still used elsewhere (it isn't once the reaper is gone — remove).
- [ ] Update `chat_stream_*` tests. Quality + convergence. Commit.

### Task 4: Multi-model load/unload/status/scan/probe through the local runtime
**Files:** `src/api.rs`, `src/serve/control.rs`, `src/serve/v1.rs`, bindings.
- [ ] `load(id, params)` → additive `local.load(NodeLoadParams)`; keep the host-side resolve/guards (`validate_repo_id`, path-within-roots, headroom) before delegating (NodeRuntime also guards — dedupe).
- [ ] `unload(served)` → resolve served → `local.unload(worker)` (or `fleet.unload`).
- [ ] `status()` → return a multi-instance snapshot (new shape) OR keep a back-compat single "primary" plus a new `instances()` listing; pick per serve/control needs. Regenerate ts-rs bindings.
- [ ] `ensure_loaded` (v1 JIT) → idempotent: if no local instance of the raw model, `load` one; else reuse.
- [ ] `/v1/models` → unified served ids (`local.instances` + `fleet`), collision-free.
- [ ] `probe_support` → `local.probe_paths`. `scan` unchanged (host FS walk). `sysinfo` → `local.sysinfo`.
- [ ] Rewrite affected `api.rs` + serve tests; bindings sync. Quality + convergence. Commit.

### Task 5: Control surface + local-as-node visibility
**Files:** `src/serve/control.rs`, `src/api.rs`, integration tests.
- [ ] Decide local visibility in `/api/higgs/nodes` (either add the local node to the view or a sibling local view). Implement + integration test (spawn process, assert local served ids + a remote node coexist).
- [ ] Full `scripts/coverage.sh` (both gates) + final codex convergence over the whole P4b diff. Commit.

## Notes / decisions to make during implementation
- **Status shape**: the single biggest API decision. Prefer additive (keep `status()` returning the primary/first local instance for back-compat; add `local_instances()` for the multi view) to bound test churn, unless serve/control clearly needs full multi.
- **JIT semantics**: idempotent local load (not additive) for `/v1/chat` auto-load; explicit control-plane load may be additive.
- **probe/events/logs**: these are the non-obvious NodeRuntime additions; they gate Tasks 3–4.
- Keep `/v1` wire shape unchanged; keep the host/worker process split.
