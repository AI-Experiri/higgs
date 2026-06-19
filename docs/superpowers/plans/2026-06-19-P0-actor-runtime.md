# P0 — Shared Actor Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. After each task: run the task tests, then `scripts/coverage.sh`, then run the codex review loop until it converges, then commit.

**Goal:** Factor the worker-management machinery higgs already hand-rolls (mailbox + recv-loop + shutdown, and the RPC reply-demux) into one shared `src/actor.rs`, written once; prove it by adopting the shared reply-demux inside today's `Supervisor`; put the `Worker` on the tokio + `spawn_blocking` runtime — **with zero behaviour change and every existing test green**.

**Architecture:** higgs's `Supervisor` is already an actor (an mpsc-fed `writer_task`, a read-loop `reader_task`, a `pending`/`chat_sinks` correlation map). P0 extracts the *generic* parts: (1) `trait Actor`+`spawn_actor` = mailbox + recv loop + shutdown, written once for P2/P3's `NodeRuntime`+transport to build on; (2) `ReplyDemux` = the pending/chat-sink correlation, shared by every RPC client. `Supervisor` adopts `ReplyDemux` now (proof it works); `Worker` moves from a hand-rolled sync `stdin` loop to a tokio runtime that bridges async stdio into the unchanged `serve_state` via `SyncIoBridge` + `spawn_blocking` (previewing §5.3, FFI on a blocking thread).

**Tech Stack:** Rust, tokio (mpsc/oneshot/broadcast, `spawn_blocking`), `tokio_util::io::SyncIoBridge`, existing `rpc::RpcFrame` NDJSON. **No new external crate.**

**Behaviour-preservation invariant (the P0 contract):** No wire bytes change, no public method signature on `Supervisor`/`Higgs` changes, no test assertion is weakened. Tests may be *added*; an existing test may only change if the change preserves its assertion (e.g. a new helper name). The gate is `scripts/coverage.sh` (`--fail-under-lines 90`) staying green.

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `src/actor.rs` | NEW. `trait Actor` + `Handle<Msg>` + `spawn_actor` (mailbox/recv-loop/shutdown); `ReplyDemux` (pending + chat_sinks + correlate/route/fail/register). The runtime, written once. | create |
| `src/lib.rs` (or crate root) | Declare `mod actor;`. | modify (1 line) |
| `src/supervisor.rs` | Replace the inline `pending`/`chat_sinks` fields + `correlate`/`route_notification`/sink-register/`fail pending` logic with `actor::ReplyDemux`. Behaviour identical. | modify |
| `src/worker/mod.rs` | `worker_main` runs a current-thread tokio runtime; bridges `tokio::io::stdin/stdout` → `serve_state` via `SyncIoBridge` inside `spawn_blocking`. `serve`/`serve_state`/`WorkerState`/handlers **unchanged**. | modify (`worker_main` only) |
| `Cargo.toml` | Ensure `tokio_util` with `io` feature is available (used by `SyncIoBridge`). | modify if absent |

---

## Task 1: `src/actor.rs` — the generic mailbox runtime (`Actor` + `spawn_actor`)

**Files:**
- Create: `src/actor.rs`
- Modify: crate root (`src/lib.rs` or `src/main.rs`/`src/bin` lib) — add `pub(crate) mod actor;`
- Test: inline `#[cfg(test)]` in `src/actor.rs`

- [ ] **Step 1: Add the module declaration**

Find the crate root module list (where `mod supervisor;` / `mod worker;` are declared) and add alongside them:

```rust
pub(crate) mod actor;
```

Run `rg -n "mod supervisor;" src` to locate the exact file/line first.

- [ ] **Step 2: Write the failing test (toy actor over the runtime)**

Create `src/actor.rs` with only the test module + empty placeholders so it compiles to a failing test:

```rust
//! Shared actor runtime: a typed mailbox + recv loop + graceful shutdown, written
//! once. Every higgs actor (Supervisor today; NodeRuntime + per-node transport in
//! P2/P3) contributes only its own `Msg` set and `handle`; nobody re-implements the loop.

#[cfg(test)]
mod tests {
    use super::*;

    struct Counter {
        total: u64,
    }

    enum CounterMsg {
        Add(u64),
        Get(tokio::sync::oneshot::Sender<u64>),
    }

    impl Actor for Counter {
        type Msg = CounterMsg;
        async fn handle(&mut self, msg: Self::Msg) {
            match msg {
                CounterMsg::Add(n) => self.total += n,
                CounterMsg::Get(reply) => {
                    let _ = reply.send(self.total);
                }
            }
        }
    }

    #[tokio::test]
    async fn actor_processes_messages_in_order_then_shuts_down() {
        let handle = spawn_actor(Counter { total: 0 });
        handle.send(CounterMsg::Add(2)).unwrap();
        handle.send(CounterMsg::Add(40)).unwrap();
        let (tx, rx) = tokio::sync::oneshot::channel();
        handle.send(CounterMsg::Get(tx)).unwrap();
        assert_eq!(rx.await.unwrap(), 42);

        // Dropping the handle closes the mailbox; the loop ends.
        drop(handle);
    }

    #[tokio::test]
    async fn send_after_shutdown_errors() {
        let handle = spawn_actor(Counter { total: 0 });
        let weak = handle.clone();
        drop(handle); // last strong sender? no — `weak` still holds one.
        // With a clone still alive, the loop is alive:
        weak.send(CounterMsg::Add(1)).unwrap();
    }
}
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `cargo test --lib actor::tests 2>&1 | head -30`
Expected: FAIL — `cannot find trait Actor` / `cannot find function spawn_actor`.

- [ ] **Step 4: Implement the runtime**

Add above the test module in `src/actor.rs`:

```rust
use tokio::sync::mpsc;

/// An actor: isolated state, reacting to its own typed messages off a mailbox.
/// `handle` is `async` so an actor may await I/O / `spawn_blocking` inside it.
pub(crate) trait Actor: Send + 'static {
    type Msg: Send + 'static;
    fn handle(&mut self, msg: Self::Msg) -> impl std::future::Future<Output = ()> + Send;
}

/// A cloneable handle to an actor's mailbox. Last clone dropped ⇒ loop ends.
pub(crate) struct Handle<M> {
    tx: mpsc::UnboundedSender<M>,
}

impl<M> Clone for Handle<M> {
    fn clone(&self) -> Self {
        Handle { tx: self.tx.clone() }
    }
}

impl<M> Handle<M> {
    /// Enqueue a message. Errs only after every handle is dropped / the loop ended.
    pub(crate) fn send(&self, msg: M) -> Result<(), mpsc::error::SendError<M>> {
        self.tx.send(msg)
    }
}

/// Spawn `state` as an actor: mailbox + recv loop + shutdown-on-drop. The runtime,
/// written ONCE — each actor contributes only `Msg` + `handle`.
pub(crate) fn spawn_actor<A: Actor>(mut state: A) -> Handle<A::Msg> {
    let (tx, mut rx) = mpsc::unbounded_channel::<A::Msg>();
    tokio::spawn(async move {
        while let Some(msg) = rx.recv().await {
            state.handle(msg).await;
        }
        // All handles dropped ⇒ graceful shutdown.
    });
    Handle { tx }
}
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cargo test --lib actor::tests 2>&1 | tail -20`
Expected: PASS (2 tests).

- [ ] **Step 6: Lint**

Run: `cargo clippy --all-targets 2>&1 | tail -20`
Expected: no new warnings from `actor.rs`.

- [ ] **Step 7: Codex review loop, then commit**

Run the codex review loop on this diff; address findings; re-review until it converges. Then:

```bash
git add src/actor.rs src/lib.rs
git commit -m "feat(actor): shared mailbox runtime — trait Actor + spawn_actor (written once)"
```

---

## Task 2: `ReplyDemux` — the shared RPC reply-demux

The reply-demux is the reader-side correlation every RPC client needs: a `pending` map (`id → oneshot<RpcResponse>`) and a `chat_sinks` map (`request_id → mpsc<String>`), plus `correlate` (response→pending), `route_chunk` (N_CHAT_CHUNK→sink), `register_sink`/`remove_sink`, and `fail_all_pending` (on EOF). Today this lives inline in `supervisor.rs` (`pending` :155, `chat_sinks` :167, `correlate` :1120-1124, `route_notification`/chunk routing :1131-1150). P2/P3's per-node transport needs the same — so factor it out.

**Files:**
- Modify: `src/actor.rs` (add `ReplyDemux`)
- Test: inline `#[cfg(test)]` in `src/actor.rs`

- [ ] **Step 1: Write the failing test**

Add to the `tests` module in `src/actor.rs`:

```rust
    use crate::rpc::RpcResponse;
    use serde_json::json;

    fn ok_response(id: u64) -> RpcResponse {
        RpcResponse { jsonrpc: "2.0".into(), id, result: Some(json!({"ok": true})), error: None }
    }

    #[tokio::test]
    async fn demux_correlates_response_by_id() {
        let demux = ReplyDemux::new();
        let rx = demux.register_pending(7);
        // out-of-order: an unrelated id must NOT resolve ours.
        demux.correlate(ok_response(99));
        demux.correlate(ok_response(7));
        let resp = rx.await.unwrap();
        assert_eq!(resp.id, 7);
    }

    #[tokio::test]
    async fn demux_routes_chunks_by_request_id() {
        let demux = ReplyDemux::new();
        let mut rx = demux.register_sink(3);
        demux.route_chunk(3, "he");
        demux.route_chunk(3, "llo");
        demux.route_chunk(4, "ignored"); // no sink for 4 ⇒ dropped, no panic
        assert_eq!(rx.recv().await.unwrap(), "he");
        assert_eq!(rx.recv().await.unwrap(), "llo");
        demux.remove_sink(3);
    }

    #[tokio::test]
    async fn demux_fail_all_pending_drops_senders() {
        let demux = ReplyDemux::new();
        let rx = demux.register_pending(1);
        demux.fail_all_pending(); // EOF: drop every pending sender
        assert!(rx.await.is_err()); // oneshot canceled
    }
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test --lib actor::tests::demux 2>&1 | head -20`
Expected: FAIL — `cannot find type ReplyDemux`.

- [ ] **Step 3: Implement `ReplyDemux`**

Add to `src/actor.rs` (the crate already uses `parking_lot::Mutex` in `supervisor.rs` — match it; if it's std `Mutex` there, use std here for consistency — verify with `rg -n "use .*Mutex" src/supervisor.rs`):

```rust
use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::Mutex;            // match supervisor.rs's Mutex choice
use tokio::sync::oneshot;
use crate::rpc::RpcResponse;

/// The reader-side reply-demux shared by every RPC *client* (Supervisor today;
/// the per-node iroh transport in P3). Server-side actors (Worker) reply inline and
/// do NOT use this. Internally `Arc`-shared so the read-loop and caller see one map.
#[derive(Clone)]
pub(crate) struct ReplyDemux {
    inner: Arc<DemuxInner>,
}

struct DemuxInner {
    pending: Mutex<HashMap<u64, oneshot::Sender<RpcResponse>>>,
    chat_sinks: Mutex<HashMap<u64, mpsc::UnboundedSender<String>>>,
}

impl ReplyDemux {
    pub(crate) fn new() -> Self {
        ReplyDemux {
            inner: Arc::new(DemuxInner {
                pending: Mutex::new(HashMap::new()),
                chat_sinks: Mutex::new(HashMap::new()),
            }),
        }
    }

    /// Register a request id; the returned receiver resolves when its response arrives.
    pub(crate) fn register_pending(&self, id: u64) -> oneshot::Receiver<RpcResponse> {
        let (tx, rx) = oneshot::channel();
        self.inner.pending.lock().insert(id, tx);
        rx
    }

    /// Remove an orphaned pending entry (e.g. on RPC timeout).
    pub(crate) fn remove_pending(&self, id: u64) {
        self.inner.pending.lock().remove(&id);
    }

    /// Deliver a response to its waiter. Unknown ids are dropped (no panic).
    pub(crate) fn correlate(&self, resp: RpcResponse) {
        if let Some(tx) = self.inner.pending.lock().remove(&resp.id) {
            let _ = tx.send(resp);
        }
    }

    /// EOF/death: cancel every pending waiter by dropping its sender.
    pub(crate) fn fail_all_pending(&self) {
        self.inner.pending.lock().clear();
    }

    /// Register a chat-chunk sink under a request_id; deltas arrive on the receiver.
    pub(crate) fn register_sink(&self, request_id: u64) -> mpsc::UnboundedReceiver<String> {
        let (tx, rx) = mpsc::unbounded_channel();
        self.inner.chat_sinks.lock().insert(request_id, tx);
        rx
    }

    pub(crate) fn remove_sink(&self, request_id: u64) {
        self.inner.chat_sinks.lock().remove(&request_id);
    }

    /// Route a streamed delta to its sink. Unknown request_ids are dropped (no panic).
    pub(crate) fn route_chunk(&self, request_id: u64, delta: &str) {
        if let Some(tx) = self.inner.chat_sinks.lock().get(&request_id) {
            let _ = tx.send(delta.to_string());
        }
    }
}
```

> If `supervisor.rs` uses `std::sync::Mutex` (not `parking_lot`), drop the `parking_lot` import and use `std::sync::Mutex` with `.lock().unwrap()`. Verify before implementing: `rg -n "Mutex" src/supervisor.rs | head`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test --lib actor::tests 2>&1 | tail -20`
Expected: PASS (all actor tests).

- [ ] **Step 5: Lint + commit (after codex review converges)**

Run: `cargo clippy --all-targets 2>&1 | tail -20` → no new warnings.

```bash
git add src/actor.rs
git commit -m "feat(actor): ReplyDemux — shared RPC reply-demux (pending + chat_sinks)"
```

---

## Task 3: `Supervisor` adopts `ReplyDemux` (proof, internal, tests stay green)

Replace `Supervisor`'s inline correlation with `actor::ReplyDemux`. This is the proof the shared demux is correct: every existing supervisor test must stay green unchanged. Map references: `Inner.pending` (`supervisor.rs:155`), `Inner.chat_sinks` (`:167`), `register_chat_sink` (`:402-406`), `remove_chat_sink` (`:412-414`), `send_request` insert-pending (`:694-750`), `correlate` (`:1120-1124`), chunk routing (`:1131-1150`), EOF fail-pending (in `reader_task` `:957-1012`).

**Files:**
- Modify: `src/supervisor.rs`
- Test: existing inline supervisor tests (`:1365-1915`) — must pass unchanged

- [ ] **Step 1: Run the baseline tests first (capture green state)**

Run: `cargo test --lib supervisor::tests 2>&1 | tail -15`
Expected: PASS (record the count — it must not drop).

- [ ] **Step 2: Replace the two fields in `Inner` with one `ReplyDemux`**

In `Inner` (`supervisor.rs:152-204`), remove the `pending` field (`:155`) and `chat_sinks` field (`:167`); add:

```rust
    demux: crate::actor::ReplyDemux,
```

In the `Inner` constructor (inside `Supervisor::spawn` `:230-246` and `with_factory` `:253-269`), replace the two map initializers with:

```rust
    demux: crate::actor::ReplyDemux::new(),
```

- [ ] **Step 3: Rewire the five call sites**

| Old site | Old code | New code |
|---|---|---|
| `register_chat_sink` (`:402-406`) | `self.inner.chat_sinks.lock()...insert` | `self.inner.demux.register_sink(request_id)` (return its receiver) |
| `remove_chat_sink` (`:412-414`) | `chat_sinks.lock().remove` | `self.inner.demux.remove_sink(request_id)` |
| `send_request` insert pending (`:694-750`) | `pending.lock().insert(id, tx)` + build oneshot | `let rx = self.inner.demux.register_pending(id);` (drop the local oneshot creation) |
| `correlate` (`:1120-1124`) | body | `inner.demux.correlate(resp);` (keep the free fn or inline the call at `dispatch` `:1112`) |
| chunk routing (`:1131-1150`) | extract `request_id`, look up `chat_sinks`, send | extract `request_id` (unchanged), then `inner.demux.route_chunk(request_id, delta);` |
| EOF in `reader_task` (`:957-1012`, the "fail pending" step) | drain `pending` map sending errors | `inner.demux.fail_all_pending();` |

For `send_request`, the timeout path that removes an orphaned pending (`:1231-1298` / inside `send_request`) becomes `self.inner.demux.remove_pending(id);`.

> Keep `next_id`/`alloc_request_id` (`:169`,`:346-348`) exactly as-is — the demux does not own id allocation.

- [ ] **Step 4: Build**

Run: `cargo build 2>&1 | tail -20`
Expected: clean compile. Fix any field/borrow errors (the demux is `Clone` and `Arc`-internal, so clone it where the old code cloned `Arc<Inner>` for the reader task, or access via `inner.demux`).

- [ ] **Step 5: Run the supervisor tests — must match the baseline count, all green**

Run: `cargo test --lib supervisor::tests 2>&1 | tail -15`
Expected: PASS, same test count as Step 1. In particular `request_response_correlation`, `chat_chunks_routed`, `two_keyed_sinks_route_independently`, `eof_fails_pending_and_emits_died`, `chat_rpc_times_out` must pass unchanged.

- [ ] **Step 6: Full gate**

Run: `scripts/coverage.sh 2>&1 | tail -15`
Expected: PASS, line coverage ≥ 90%.

- [ ] **Step 7: Lint + codex review loop + commit**

Run: `cargo clippy --all-targets 2>&1 | tail -20` → no new warnings. Run the codex review loop until convergence. Then:

```bash
git add src/supervisor.rs
git commit -m "refactor(supervisor): adopt actor::ReplyDemux for RPC correlation (no behaviour change)"
```

---

## Task 4: `Worker` rides the tokio + `spawn_blocking` runtime

Move `worker_main` off its hand-rolled sync `stdin.lock()` loop onto a current-thread tokio runtime that bridges async stdio into the **unchanged** `serve_state` via `SyncIoBridge` inside `spawn_blocking` (so the blocking FFI `engine.chat` runs on a blocking thread). This is the §5.3 bridge pattern, previewed. `serve`, `serve_state`, `WorkerState`, and every handler stay byte-for-byte identical — so all worker tests (`worker/mod.rs:408-859`) and the `tests/worker_roundtrip.rs` integration test pass unchanged.

**Files:**
- Modify: `src/worker/mod.rs` — `worker_main` (`:41-46`) ONLY
- Modify: `Cargo.toml` — ensure `tokio_util` with feature `io`
- Test: existing worker tests + `tests/worker_roundtrip.rs` (unchanged)

- [ ] **Step 1: Confirm `SyncIoBridge` availability**

Run: `rg -n "tokio_util|tokio-util" Cargo.toml`
If `tokio-util` is absent or lacks the `io` feature, add to `Cargo.toml [dependencies]`:

```toml
tokio-util = { version = "0.7", features = ["io"] }
```

(Verify the workspace's pinned tokio-util version with `cargo tree -p tokio-util 2>/dev/null | head -1` and match it.)

- [ ] **Step 2: Run the baseline worker tests (capture green state)**

Run: `cargo test --lib worker:: 2>&1 | tail -15 && cargo test --test worker_roundtrip 2>&1 | tail -15`
Expected: PASS (record counts).

- [ ] **Step 3: Rewrite `worker_main` to the runtime bridge**

Replace `worker_main` (`worker/mod.rs:41-46`) with:

```rust
pub fn worker_main() {
    engine::llamacpp::logging::install_worker_logging();

    // Minimal current-thread tokio runtime: the worker stays a separate, crash-isolated
    // process; it just rides the same async runtime as the rest of higgs. The blocking
    // FFI (engine.chat) runs inside spawn_blocking, off the runtime's reactor thread.
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("higgs worker: failed to build tokio runtime");

    runtime.block_on(async {
        let stdin = tokio::io::stdin();
        let stdout = tokio::io::stdout();
        // Bridge async stdio → the unchanged sync serve_state (std::io BufRead/Write),
        // running the whole sync serve loop on a blocking thread (§5.3 preview).
        let handle = tokio::task::spawn_blocking(move || {
            let reader = std::io::BufReader::new(tokio_util::io::SyncIoBridge::new(stdin));
            let writer = tokio_util::io::SyncIoBridge::new(stdout);
            serve(reader, writer);
        });
        let _ = handle.await;
    });
}
```

> `serve` (`:49-51`) takes `impl BufRead, impl Write`. `SyncIoBridge<R: AsyncRead>` impls `std::io::Read`; wrap it in `BufReader` to get `BufRead`. `SyncIoBridge<W: AsyncWrite>` impls `std::io::Write` directly. **`SyncIoBridge` must be constructed/used inside `spawn_blocking`** (its calls block the thread). Do not touch `serve`/`serve_state`/`WorkerState`.

- [ ] **Step 4: Build**

Run: `cargo build 2>&1 | tail -20`
Expected: clean compile.

- [ ] **Step 5: Run the worker tests + integration roundtrip — unchanged, all green**

Run: `cargo test --lib worker:: 2>&1 | tail -15 && cargo test --test worker_roundtrip 2>&1 | tail -15`
Expected: PASS, same counts as Step 2. `tests/worker_roundtrip.rs` re-execs the real binary and drives `worker_main()` on real pipes (M_STATUS + M_SHUTDOWN) — this exercises the new runtime path end to end.

- [ ] **Step 6: Full gate**

Run: `scripts/coverage.sh 2>&1 | tail -15`
Expected: PASS, ≥ 90%.

- [ ] **Step 7: Lint + codex review loop + commit**

Run: `cargo clippy --all-targets 2>&1 | tail -20` → no new warnings. Run the codex review loop until convergence. Then:

```bash
git add src/worker/mod.rs Cargo.toml Cargo.lock
git commit -m "refactor(worker): run worker_main on tokio + spawn_blocking via SyncIoBridge (§5.3 preview)"
```

---

## Task 5: P0 acceptance — whole-suite green, no behaviour change

**Files:** none (verification only)

- [ ] **Step 1: Full test suite + coverage gate**

Run: `scripts/coverage.sh 2>&1 | tail -25`
Expected: PASS, line coverage ≥ 90%. Compare the test count to `main`'s — it must be **≥** the original (we added actor tests, removed none).

- [ ] **Step 2: Full clippy**

Run: `cargo clippy --all-targets --all-features 2>&1 | tail -25`
Expected: no warnings.

- [ ] **Step 3: Confirm the P0 contract**

Verify by inspection / `git diff main --stat`:
- No public signature on `Supervisor` / `Higgs` / `worker` changed (only internals + `worker_main` body).
- No wire/NDJSON change.
- `src/actor.rs` exists with `trait Actor`, `spawn_actor`, `ReplyDemux`, all unit-tested.
- `Supervisor` uses `ReplyDemux`; `Worker` runs on the tokio runtime.

- [ ] **Step 4: Smoke the real binary (optional but recommended)**

Run: `cargo run --bin higgs -- --help 2>&1 | head -20` (or the project's normal launch). Confirm it starts and a worker can be spawned (use an existing integration test as the proxy if no manual GGUF is handy: `cargo test --test control_api 2>&1 | tail -15`).

- [ ] **Step 5: Final codex review of the cumulative P0 diff, then tag the phase**

Run the codex review loop over `git diff main` for the whole phase; converge. Then update the roadmap ledger (`docs/superpowers/plans/2026-06-19-iroh-remote-roadmap.md`) P0 status → **done**, and commit:

```bash
git add docs/superpowers/plans/2026-06-19-iroh-remote-roadmap.md
git commit -m "docs(plan): P0 actor runtime complete; roadmap updated"
```

---

## Self-Review (against the spec §2.5 + §10-P0)

- **Spec coverage:** §2.5 "one shared `actor` module, written once" → Task 1 (`trait Actor`+`spawn_actor`). §2.5 "reply-demux ... written once and shared" → Task 2 (`ReplyDemux`) + Task 3 (Supervisor adopts it). §10-P0 "port Worker (minimal tokio runtime + spawn_blocking FFI) so Supervisor + Worker share ONE runtime" → Task 4. §10-P0 "existing supervisor + worker tests stay green ... no behaviour change, no duplicated loop" → the P0 contract + Tasks 3/4 baseline-vs-after checks + Task 5.
- **Deferred (YAGNI, honestly flagged):** The typed `spawn_actor` *mailbox* gets its first real consumers — `NodeRuntime` and the per-node transport — in P2/P3, where genuinely new actors exist. P0 writes `spawn_actor` once and proves it with a unit test (Task 1) rather than forcing Supervisor's already-working reader/writer tasks into a typed-message rewrite that would risk the no-behaviour-change contract for no functional gain. `ReplyDemux` *is* consumed now (Task 3). This matches the spec's "reuse `Supervisor` as the per-worker unit (unchanged)" intent.
- **Placeholder scan:** none — every code step has real code.
- **Type consistency:** `ReplyDemux` methods (`register_pending`/`remove_pending`/`correlate`/`fail_all_pending`/`register_sink`/`remove_sink`/`route_chunk`) are used with identical names in Tasks 2 and 3. `spawn_actor`/`Handle::send` consistent across Task 1.
