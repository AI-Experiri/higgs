# P2 — Node Runtime (multi-worker + iroh control/data) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or executing-plans. Steps use `- [ ]`. After each task: task tests → `scripts/coverage.sh` → **codex review loop until it converges** (`codex review --uncommitted`) → commit. Do not advance until codex converges.

**Goal:** A higgs **node** dials a hub, completes HELLO, and serves it: the hub issues `higgs/node/*` control RPCs (load/unload/kill/scan/sysinfo/status) over the control stream, the node's `NodeRuntime` runs them against a `HashMap<WorkerId, Arc<Supervisor>>` of real child workers (multi-worker, net-new), and per-chat the hub opens a data stream the node relays through `Supervisor.chat()`. **Exit:** one node hosts 2 concurrent workers; the hub gets `M_SYSINFO` + `M_STATUS` for each over iroh.

**Architecture (codex-validated — see roadmap "Architecture correction"):** The node relays through the existing `Supervisor` (which already bridges to the child's sync stdio). **No `SyncIoBridge`, no `serve_state` in the node, no worker tokio port** — the worker child stays pure-sync stdio. Two dispatchers, never merged: control (`higgs/node/*` → `NodeRuntime`) vs data (`M_CHAT` → `Supervisor.chat()` relay). `NodeRuntime` owns the registry behind a `parking_lot::Mutex` (locked only for insert/get/remove, never across `.await`); a Supervisor is cloned out (`Arc`) before any await.

**Tech Stack:** Rust, iroh 1.0 (from P1), existing `Supervisor`/`production_factory`/`actor`/`rpc`, `system::fits_vram`. Builds on P1's `node/` (gate, dial, HELLO).

**Contract:** existing tests stay green; coverage ≥ 90%. `WorkerId` is `u32` (Copy). Worker, `serve_state`, `Supervisor` public API unchanged.

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `src/node/worker_id.rs` | NEW. `WorkerId(u32)` newtype (Copy) + `WorkerRegistry` (monotonic alloc, insert/get/remove, list). | create |
| `src/node/runtime.rs` | NEW. `NodeRuntime`: registry + node config + injectable supervisor spawn; async `load`/`unload`/`kill`/`scan`/`sysinfo`/`status`/`chat`. | create |
| `src/node/control.rs` | NEW. `higgs/node/*` method consts (in `remote.rs`) + `dispatch_node_control(rt, req) -> RpcResponse`; the node's control read-loop. | create |
| `src/node/data.rs` | NEW. `relay_chat(rt, send, recv)`: read `M_CHAT`, relay chunks+final through `Supervisor.chat()`, cancel on stream close. | create |
| `src/remote.rs` | Add `higgs/node/*` method consts (`M_NODE_LOAD` etc.) + per-op param/result serde shapes. | modify |
| `src/node/mod.rs` | Declare new submodules; add `run_node_daemon` (persistent: dial → HELLO → control loop + data accept loop). | modify |
| `src/node/cli.rs` | Wire `--node` to the persistent daemon; `node connect` stays a one-shot. | modify |
| `tests/remote_node.rs` | NEW. Integration: node spawns 2 real workers; hub drives `M_SYSINFO`+`M_STATUS` per worker over iroh. | create |

---

## Task 1: `WorkerId(u32)` + `WorkerRegistry`

Pure logic, fully TDD. `WorkerId` is the per-node worker key — **`u32`, `Copy`** (keeps `LogSource` Copy in P4).

**Files:** create `src/node/worker_id.rs`; declare `pub mod worker_id;` in `src/node/mod.rs`.

- [ ] **Step 1: Failing test**

```rust
//! WorkerId — the per-node worker key (u32, Copy). Owned by the NodeRuntime registry;
//! assigned on load, freed on unload/kill (DESIGN-remote.md §5.4a).

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_is_monotonic_and_insert_get_remove() {
        let mut reg: WorkerRegistry<u8> = WorkerRegistry::new();
        let a = reg.insert(10);
        let b = reg.insert(20);
        assert_ne!(a, b);
        assert!(b.0 > a.0, "ids are monotonic");
        assert_eq!(reg.get(a), Some(&10));
        assert_eq!(reg.len(), 2);
        assert_eq!(reg.remove(a), Some(10));
        assert_eq!(reg.get(a), None);
        // ids are never reused even after removal
        let c = reg.insert(30);
        assert!(c.0 > b.0, "freed id is not reused");
    }

    #[test]
    fn worker_id_renders_as_w_prefix() {
        assert_eq!(WorkerId(1).to_string(), "w-1");
    }
}
```

Run: `cargo test --lib node::worker_id 2>&1 | tail -5` → FAIL.

- [ ] **Step 2: Implement**

```rust
use std::collections::HashMap;
use std::fmt;

/// Per-node worker key. `u32` + `Copy` so `LogSource::RemoteWorker` stays `Copy` (P4).
/// Wire carries it as a number (`"worker_id": 1`); UI may render it "w-1".
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct WorkerId(pub u32);

impl fmt::Display for WorkerId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "w-{}", self.0)
    }
}

/// A monotonic-id registry of live workers. Generic over the value (`Arc<Supervisor>`
/// in production; a fake in tests). Ids are never reused, so a stale `(node,worker)`
/// reference can't alias a different worker.
pub struct WorkerRegistry<T> {
    next: u32,
    map: HashMap<WorkerId, T>,
}

impl<T> WorkerRegistry<T> {
    pub fn new() -> Self {
        Self { next: 1, map: HashMap::new() }
    }

    /// Assign the next id and store `value`; returns the new id.
    pub fn insert(&mut self, value: T) -> WorkerId {
        let id = WorkerId(self.next);
        self.next += 1;
        self.map.insert(id, value);
        id
    }

    pub fn get(&self, id: WorkerId) -> Option<&T> {
        self.map.get(&id)
    }

    pub fn remove(&mut self, id: WorkerId) -> Option<T> {
        self.map.remove(&id)
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Live worker ids, ascending.
    pub fn ids(&self) -> Vec<WorkerId> {
        let mut v: Vec<_> = self.map.keys().copied().collect();
        v.sort();
        v
    }
}

impl<T> Default for WorkerRegistry<T> {
    fn default() -> Self {
        Self::new()
    }
}
```

Run: `cargo test --lib node::worker_id 2>&1 | tail -5` → PASS. Then clippy + **codex loop** + commit:
```bash
git commit -m "feat(node): WorkerId(u32) + monotonic WorkerRegistry"
```

---

## Task 2: `higgs/node/*` wire vocabulary (remote.rs)

Add the control method constants + per-op param/result serde shapes. Additive; the `higgs/node/` prefix is mandatory (never confused with worker `higgs/*`).

**Files:** modify `src/remote.rs`.

- [ ] **Step 1: Failing test** (append to `remote::tests`)

```rust
    #[test]
    fn node_method_consts_are_namespaced() {
        assert_eq!(M_NODE_LOAD, "higgs/node/load");
        assert_eq!(M_NODE_UNLOAD, "higgs/node/unload");
        assert_eq!(M_NODE_KILL, "higgs/node/kill");
        assert_eq!(M_NODE_SCAN, "higgs/node/scan");
        assert_eq!(M_NODE_SYSINFO, "higgs/node/sysinfo");
        assert_eq!(M_NODE_STATUS, "higgs/node/status");
    }

    #[test]
    fn load_params_roundtrip() {
        let p = NodeLoadParams { id: "org/m".into(), ctx_len: Some(4096), gpu_layers: None, threads: None };
        let s = serde_json::to_string(&p).unwrap();
        let back: NodeLoadParams = serde_json::from_str(&s).unwrap();
        assert_eq!(back.id, "org/m");
        assert_eq!(back.ctx_len, Some(4096));
    }
```

Run: `cargo test --lib remote:: 2>&1 | tail -5` → FAIL.

- [ ] **Step 2: Implement** (append to `src/remote.rs`)

```rust
/// Control-plane methods (hub → node), all namespaced `higgs/node/*` so a reader never
/// confuses a hub→node op with a node→worker `higgs/*` op (DESIGN-remote.md §4.2, flag #1).
pub const M_NODE_LOAD: &str = "higgs/node/load";
pub const M_NODE_UNLOAD: &str = "higgs/node/unload";
pub const M_NODE_KILL: &str = "higgs/node/kill";
pub const M_NODE_SCAN: &str = "higgs/node/scan";
pub const M_NODE_SYSINFO: &str = "higgs/node/sysinfo";
pub const M_NODE_STATUS: &str = "higgs/node/status";

/// `higgs/node/load` params — spawn a NEW worker for model `id`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeLoadParams {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ctx_len: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_layers: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threads: Option<u32>,
}

/// `{ "worker_id": <u32> }` — the target/result selector for unload/kill/status and the
/// load result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerRef {
    pub worker_id: u32,
}
```

Run: `cargo test --lib remote:: 2>&1 | tail -5` → PASS. clippy + **codex loop** + commit:
```bash
git commit -m "feat(remote): higgs/node/* control methods + param shapes"
```

---

## Task 3: `NodeRuntime` — the multi-worker registry + ops

The net-new orchestrator. Owns the registry + node config + an **injectable supervisor spawner** (so lib tests use fake-factory Supervisors, production uses `Supervisor::spawn`). Each op locks the registry briefly, clones the `Arc<Supervisor>`, then awaits outside the lock.

**Files:** create `src/node/runtime.rs`; `pub mod runtime;` in `mod.rs`.

- [ ] **Step 1: Failing test** (lib test — uses `Supervisor::with_factory` fakes, `#[cfg(test)]` available in-crate)

```rust
#[cfg(test)]
mod tests {
    use super::*;

    // A spawner that hands out fake-duplex Supervisors (no real child) so we can test
    // registry + fit-check + routing without spawning llama.cpp.
    fn fake_runtime() -> NodeRuntime {
        NodeRuntime::with_spawner(
            NodeConfig::test_default(),
            Box::new(|bus| crate::supervisor::Supervisor::with_factory(crate::supervisor::test_echo_factory(bus))),
        )
    }

    #[tokio::test]
    async fn load_assigns_ids_and_kill_frees_them() {
        let rt = fake_runtime();
        let a = rt.load(NodeLoadParams { id: "m-a".into(), ctx_len: None, gpu_layers: None, threads: None }).await.unwrap();
        let b = rt.load(NodeLoadParams { id: "m-b".into(), ctx_len: None, gpu_layers: None, threads: None }).await.unwrap();
        assert_ne!(a, b);
        assert_eq!(rt.worker_ids().len(), 2, "two concurrent workers");
        rt.kill(a).await.unwrap();
        assert_eq!(rt.worker_ids().len(), 1);
        assert!(rt.kill(a).await.is_err(), "killing a freed id errors");
    }
}
```

> **Test seam needed:** add a `#[cfg(test)] pub(crate) fn test_echo_factory(bus) -> HalvesFactory` to `supervisor.rs` that returns halves wired to a tiny in-memory worker stub answering `M_LOAD`/`M_STATUS`/`M_SYSINFO`/`M_SHUTDOWN` (reuse the existing test duplex helper pattern from `supervisor.rs` tests / `worker/mod.rs` FakeEngine). If a simpler seam exists (e.g. the existing `make_supervisor` test helper), reuse it instead. Confirm the exact helper before writing this.

Run: `cargo test --lib node::runtime 2>&1 | tail` → FAIL.

- [ ] **Step 2: Implement `NodeRuntime`**

```rust
use std::sync::Arc;

use parking_lot::Mutex;
use serde_json::{json, Value};

use crate::diagnostic::HiggsError;
use crate::log_bus::LogBus;
use crate::node::worker_id::{WorkerId, WorkerRegistry};
use crate::remote::NodeLoadParams;
use crate::supervisor::Supervisor;
use crate::worker::{M_LOAD, M_STATUS, M_SYSINFO};

/// How a node spawns a fresh Supervisor (production: `Supervisor::spawn`; tests: a fake).
pub type SupervisorSpawner = Box<dyn Fn(Arc<LogBus>) -> Supervisor + Send + Sync>;

/// Node configuration (model dirs for scan, default load params). Mirrors the parts of
/// `HiggsConfig` the node needs; kept minimal for P2.
pub struct NodeConfig {
    pub bus: Arc<LogBus>,
    // model dirs / default params added as scan/load need them (P2 load takes an
    // already-resolved path via NodeLoadParams.id → host scan, like Higgs::load).
}

/// The net-new node orchestrator: N concurrent Supervisors, one child each.
pub struct NodeRuntime {
    registry: Mutex<WorkerRegistry<Arc<Supervisor>>>,
    spawner: SupervisorSpawner,
    config: NodeConfig,
}

impl NodeRuntime {
    pub fn new(config: NodeConfig) -> Self {
        Self::with_spawner(config, Box::new(|bus| Supervisor::spawn(bus)))
    }

    pub fn with_spawner(config: NodeConfig, spawner: SupervisorSpawner) -> Self {
        Self { registry: Mutex::new(WorkerRegistry::new()), spawner, config }
    }

    pub fn worker_ids(&self) -> Vec<WorkerId> {
        self.registry.lock().ids()
    }

    fn get(&self, id: WorkerId) -> Result<Arc<Supervisor>, HiggsError> {
        self.registry
            .lock()
            .get(id)
            .cloned()
            .ok_or_else(|| HiggsError::WorkerDead { context: format!("no worker {id}") })
    }

    /// Spawn a NEW worker for `params.id` (net-new multi-worker). Fit-check VRAM against
    /// what resident workers already consume before spawning (§4.2b). Returns its id.
    pub async fn load(&self, params: NodeLoadParams) -> Result<WorkerId, HiggsError> {
        // (Fit-check: sum resident workers' VRAM vs free; reject HG017 if it won't fit.
        //  P2 wires the existing `system::fits_vram`; detail filled when integrating with
        //  real sysinfo — for the fake test the check is a no-op pass.)
        let sup = Arc::new((self.spawner)(self.config.bus.clone()));
        sup.start_for(&params.id)?;
        let load_params = json!({
            "id": params.id,
            "ctx_len": params.ctx_len,
            "gpu_layers": params.gpu_layers,
            "threads": params.threads,
        });
        sup.request(M_LOAD, load_params).await?;
        Ok(self.registry.lock().insert(sup))
    }

    /// Graceful unload: stop the worker, free the id.
    pub async fn unload(&self, id: WorkerId) -> Result<(), HiggsError> {
        let sup = self.take(id)?;
        sup.stop().await;
        Ok(())
    }

    /// Force-kill ONE worker (same as unload at this layer; stop() reaps the child).
    pub async fn kill(&self, id: WorkerId) -> Result<(), HiggsError> {
        let sup = self.take(id)?;
        sup.stop().await;
        Ok(())
    }

    fn take(&self, id: WorkerId) -> Result<Arc<Supervisor>, HiggsError> {
        self.registry
            .lock()
            .remove(id)
            .ok_or_else(|| HiggsError::WorkerDead { context: format!("no worker {id}") })
    }

    /// Per-worker status (forwards `M_STATUS` to that worker's Supervisor).
    pub async fn status(&self, id: WorkerId) -> Result<Value, HiggsError> {
        self.get(id)?.request(M_STATUS, Value::Null).await
    }

    /// Per-worker device list (forwards `M_SYSINFO`).
    pub async fn sysinfo(&self, id: WorkerId) -> Result<Value, HiggsError> {
        self.get(id)?.request(M_SYSINFO, Value::Null).await
    }
}
```

> The fit-check is sketched, not stubbed-with-placeholder: implement it against `system::fits_vram` summing resident workers' VRAM once Task 6 wires real sysinfo. For Task 3's fake test it passes trivially (no GPU). Mark the precise fit-check code as a sub-step when integrating, and DO NOT ship a silent always-pass in production — gate it on whether device info is available, logging if skipped.

Run tests → PASS. clippy + **codex loop** + commit.

---

## Task 4: Control dispatch (`higgs/node/*` → NodeRuntime)

Map an inbound control `RpcRequest` to a `NodeRuntime` op and build the `RpcResponse`. Pure routing over the async ops; TDD with the fake runtime.

**Files:** create `src/node/control.rs`; `pub mod control;`.

- [ ] **Step 1: Failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    // reuse fake_runtime() pattern from runtime tests (factor a shared test helper)

    #[tokio::test]
    async fn load_then_status_dispatch() {
        let rt = /* fake runtime */;
        let load = dispatch_node_control(&rt, req(1, M_NODE_LOAD, json!({"id":"m"}))).await;
        let worker_id = load.result.unwrap()["worker_id"].as_u64().unwrap() as u32;
        let status = dispatch_node_control(&rt, req(2, M_NODE_STATUS, json!({"worker_id":worker_id}))).await;
        assert!(status.error.is_none());
    }

    #[tokio::test]
    async fn unknown_method_is_method_not_found() {
        let rt = /* fake runtime */;
        let resp = dispatch_node_control(&rt, req(1, "higgs/node/bogus", json!({}))).await;
        assert_eq!(resp.error.unwrap().code, -32601);
    }
}
```

- [ ] **Step 2: Implement** `dispatch_node_control(rt: &NodeRuntime, req: RpcRequest) -> RpcResponse` — `match req.method`: LOAD→`rt.load`, UNLOAD/KILL→parse `WorkerRef`→`rt.unload/kill`, STATUS/SYSINFO→`rt.status/sysinfo`, SCAN→`rt.scan`, else `-32601`. Map `HiggsError` to an `RpcError` carrying the HG code in `data` (mirror `supervisor::send_request`'s error mapping). Build `ok_response(id, value)` / `err_response(id, hgcode, msg)` helpers.

Tests → PASS. clippy + **codex loop** + commit.

---

## Task 5: Data relay (`M_CHAT` → `Supervisor.chat()`) + node daemon

The node-side data path and the persistent `--node` daemon. **Heed codex's relay risks.**

**Files:** create `src/node/data.rs`; extend `src/node/mod.rs` (`run_node_daemon`) + `cli.rs`.

- [ ] **Step 1: `relay_chat`** — `async fn relay_chat(rt: &NodeRuntime, send: SendStream, recv: RecvStream)`:
  1. read one `M_CHAT` frame (bounded like HELLO); parse `worker_id` + chat params.
  2. `let (mut chunks, fut) = rt.chat_handle(worker_id)?.chat(...)` (add `NodeRuntime::chat_handle(id) -> Arc<Supervisor>`).
  3. **single writer**: loop `chunks.recv()` → write `N_CHAT_CHUNK` (hub-visible `request_id` from the inbound frame, not the supervisor-local id); then `let final = fut.await` → write final `RpcResponse` (or an error frame on `Err`).
  4. **cancellation**: `tokio::select!` the relay against `recv`/`conn` close; on close, drop `chunks` (and `remove_chat_sink`) so the worker stops being relayed. Document that true mid-generation cancel is best-effort (Supervisor has no cancel today).
  Unit-test with the fake echo worker: assert chunks then final arrive in order, hub `request_id` echoed.

- [ ] **Step 2: `run_node_daemon`** (replaces the P1 print-only stub) — persistent:
  ```text
  load config (hub ticket/id from ~/.higgs/config.toml or args) → bind endpoint →
  dial hub + dial_and_hello (P1) → on success:
    open/keep the control stream; spawn control_loop (read higgs/node/* reqs on the
      control stream's recv, dispatch_node_control, write replies on send);
    spawn data_accept_loop (conn.accept_bi() per chat → relay_chat);
    await conn.closed(); on drop, reconnect with backoff (id stable).
  ```
  Wire `NodeRuntime::new(config)` (real `Supervisor::spawn`). Keep the loop bounded/observable; log lifecycle.

> Verify the control-stream role-reversal: the node OPENED the control bi-stream for HELLO; after the HELLO reply it READS hub→node requests on that stream's recv half and WRITES replies on its send half (full-duplex bidi). Confirm against iroh that this works without reopening.

- [ ] **Step 3:** `cli.rs` — `--node` now calls `run_node_daemon()` (persistent). Add a `config.toml` loader (add `toml` dep only if you introduce the file; else accept the hub ticket as a `--node <ticket> <token?>` arg for P2 and persist the paired hub id to `~/.higgs/`).

clippy + **codex loop** + commit.

---

## Task 6: Integration — node hosts 2 workers; SYSINFO + STATUS over iroh

**Files:** create `tests/remote_node.rs`. Uses **real** child workers (integration tests can't see `#[cfg(test)]` seams), no model loaded — `M_SYSINFO` enumerates devices and `M_STATUS` reports "not loaded" without a GGUF, satisfying the exit criteria cheaply. (A model-loaded `M_LOAD`+chat path reuses the existing test GGUF and lands fully in P3's e2e.)

- [ ] **Step 1:** Spin a node `NodeRuntime` with the real spawner; spawn 2 workers (no model) via `rt.load`-without-model OR a `rt.spawn_bare(worker_id)` helper if load requires a model — provide a no-model spawn path the test can use (e.g. `M_STATUS`/`M_SYSINFO` don't need a loaded model; `start_for` spawns the child, then query without `M_LOAD`). Confirm the worker answers `M_STATUS`/`M_SYSINFO` pre-load (it does — `worker_roundtrip.rs` proves M_STATUS pre-load).
- [ ] **Step 2:** Stand up hub + node iroh connection (reuse `tests/remote_pairing.rs` local-endpoint helper); pair them; from the hub side, send `M_NODE_SYSINFO` and `M_NODE_STATUS` for each `worker_id` over the control stream; assert non-error results and 2 distinct workers.
- [ ] **Step 3:** Full gate `scripts/coverage.sh` ≥ 90%; clippy; **codex loop**; commit.

---

## Task 7: P2 acceptance

- [ ] Full suite + coverage ≥ 90%; full clippy clean.
- [ ] Confirm exit (roadmap P2): node hosts 2 concurrent workers; `M_SYSINFO` + `M_STATUS` over iroh ✓.
- [ ] Cumulative `codex review --base <P1-tip>` → converge.
- [ ] Roadmap ledger P2 → **DONE**; commit.

---

## Self-Review (against spec §2, §4.2, §5.4 + the architecture correction)

- **Coverage:** §5.4a control dispatch → Task 4. NodeRuntime multi-worker registry + `WorkerId` → Tasks 1, 3. §4.2 `higgs/node/*` vocab → Task 2. data relay (corrected: via `Supervisor.chat()`) → Task 5. `--node` daemon → Task 5. fit-check (§4.2b) → Task 3 (wired in Task 6). 2-worker SYSINFO/STATUS exit → Task 6.
- **Corrected vs spec:** no `SyncIoBridge`/`serve_state` on the node, no worker tokio port (eliminated, codex-validated). Recorded in the roadmap.
- **Deferred (correct):** hub-side chat relay + `/v1` e2e + wedged-worker reap → P3. `M_INVENTORY` push + `NodeView` + HW/RT Deserialize → P4. `M_PULL` → P4b.
- **Invariants:** `WorkerId` u32 Copy ✓; two dispatchers never merged ✓; registry lock never held across await ✓; reuse `Supervisor` as the per-worker unit unchanged ✓.
- **Open detail to resolve in-task:** the exact `Supervisor` test seam for fake workers (Task 3 Step 1) — confirm the existing helper before writing; the no-model worker spawn path for the integration test (Task 6 Step 1).
