//! Worker supervision: spawns the worker role by re-executing the current
//! binary with `--higgs-worker`, speaks NDJSON JSON-RPC over its stdio,
//! correlates responses by id, routes chat-chunk notifications, restarts the
//! worker on death (one attempt per death), and re-loads the last model
//! after restart.
//!
//! ## Transport shape — mirroring `mcp/registry.rs add_local`
//!
//! The reference implementation uses `tokio::process::Command` with owned
//! stdio halves.  Higgs mirrors this directly:
//!
//! ```text
//!  production factory                  test factory
//!  ─────────────────                   ────────────
//!  tokio::process::Command             tokio::io::duplex halves
//!    .stdin(Stdio::piped())
//!    .stdout(Stdio::piped())
//!         │
//!  child.stdin.take()  ──────── write_tx (UnboundedSender<String>)
//!                                         │
//!                                 writer task (owns ChildStdin)
//!                                 drains channel → write_all → flush
//!         │
//!  child.stdout.take() ─────── reader task (BufReader::new(stdout).lines())
//!                               dispatches each line → correlate / notify
//! ```
//!
//! No transport trait.  No mutex between writer and reader.  The mpsc channel
//! serialises concurrent callers onto the single writer task — same pattern as
//! rmcp's `TokioChildProcess` / LSP client writers.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::{broadcast, mpsc, oneshot};
use tracing::{info, warn};

use crate::diagnostic::HiggsError;
use crate::rpc::{self, RpcFrame, RpcNotification, RpcRequest};
use crate::worker::{M_LOAD, M_SCAN, M_SHUTDOWN, N_CHAT_CHUNK};

/// Events the host application subscribes to.
#[derive(Debug, Clone, serde::Serialize, ts_rs::TS)]
#[ts(export, export_to = "../../../frontend/src/lib/generated/")]
pub enum HiggsEvent {
    /// A model finished loading and is ready to serve requests.
    ModelLoaded { id: String },
    /// A model was unloaded and is no longer available.
    ModelUnloaded { id: String },
    /// The worker process exited unexpectedly.
    WorkerDied,
    /// The worker process was restarted after a death.
    WorkerRestarted,
}

// ── Worker halves ─────────────────────────────────────────────────────────────
//
// The factory returns owned async I/O halves.  `do_spawn` takes them,
// spawns a writer task (owns the write half, drains the mpsc channel) and
// a reader task (owns the read half, feeds the dispatch loop), then stores
// only the mpsc sender in `Inner`.  No mutex touches the I/O path.

/// Boxed async write half — stdin for production, one DuplexStream end for tests.
type WriteHalf = Box<dyn tokio::io::AsyncWrite + Unpin + Send + 'static>;
/// Boxed async read half — stdout for production, one DuplexStream end for tests.
type ReadHalf = Box<dyn tokio::io::AsyncRead + Unpin + Send + 'static>;

/// Owned I/O halves returned by the factory on each (re)spawn.
///
/// The factory also takes responsibility for wiring the stderr drain task so
/// callers never see process plumbing directly.
pub(crate) struct WorkerHalves {
    /// Write half (supervisor writes requests here).
    pub(crate) write: WriteHalf,
    /// Read half (supervisor reads responses/notifications here).
    pub(crate) read: ReadHalf,
}

/// Factory: produces fresh I/O halves on each (re)spawn.
///
/// Receives the stderr ring so the production impl can wire its drain task.
/// Test factories receive the ring too but may ignore it.
type HalvesFactory =
    Box<dyn Fn(Arc<Mutex<VecDeque<String>>>) -> Result<WorkerHalves, HiggsError> + Send + Sync>;

/// Shared supervisor state.
struct Inner {
    /// Pending request id → oneshot reply channel.
    /// Lock held for insert/remove only — never across `.await`.
    pending: Mutex<HashMap<u64, oneshot::Sender<rpc::RpcResponse>>>,
    /// Active chat-chunk sink (one in-flight chat at a time; worker serializes).
    /// Lock held for swap/take only.
    ///
    /// v1 invariant: only one chat request is in flight at a time because the
    /// worker serialises chat; a `debug_assert!` in `take_chat_sink` catches
    /// accidental clobbering of an active sink during development.
    chat_sink: Mutex<Option<mpsc::UnboundedSender<String>>>,
    /// Monotonically increasing request id counter.
    next_id: AtomicU64,
    /// Broadcast channel for lifecycle events (cap 64).
    events_tx: broadcast::Sender<HiggsEvent>,
    /// Ring buffer of recent stderr lines (cap 2000).
    /// `Arc`-wrapped so the factory closure can clone the handle.
    stderr_ring: Arc<Mutex<VecDeque<String>>>,
    /// Params of the last successful `higgs/scan`; replayed before load after restart.
    last_scan: Mutex<Option<Value>>,
    /// Params of the last successful `higgs/load`; replayed after restart.
    last_load: Mutex<Option<Value>>,
    /// Set on `stop()` — suppresses respawn after death.
    stopped: AtomicBool,
    /// mpsc sender into the writer task for the active worker lifetime.
    /// `None` when no worker is live.  Replacing this drops the previous
    /// sender, which signals the old writer task to exit.
    write_tx: Mutex<Option<mpsc::UnboundedSender<String>>>,
    /// Factory that builds fresh I/O halves on each (re)spawn.
    factory: HalvesFactory,
}

/// Worker process supervisor.
///
/// Spawns the higgs worker by re-executing the current binary with
/// `--higgs-worker`, speaks NDJSON JSON-RPC 2.0 over its stdio, correlates
/// request/response pairs by id, routes `higgs/chat/chunk` notifications to
/// the active chat receiver, and restarts the worker (1s backoff) on death
/// (at most one attempt per death; factory failure → terminal `WorkerDied`,
/// no retry loop).
///
/// One in-flight chat at a time: the worker serialises chat requests anyway;
/// this is a documented v1 invariant.
pub(crate) struct Supervisor {
    inner: Arc<Inner>,
}

impl Supervisor {
    /// Create a supervisor using the production child-process factory.
    ///
    /// Does not spawn the worker yet; call `start()`.
    pub(crate) fn spawn() -> Self {
        let (events_tx, _) = broadcast::channel(64);
        let inner = Arc::new(Inner {
            pending: Mutex::new(HashMap::new()),
            chat_sink: Mutex::new(None),
            next_id: AtomicU64::new(1),
            events_tx,
            stderr_ring: Arc::new(Mutex::new(VecDeque::with_capacity(2000))),
            last_scan: Mutex::new(None),
            last_load: Mutex::new(None),
            stopped: AtomicBool::new(false),
            write_tx: Mutex::new(None),
            factory: Box::new(production_factory),
        });
        Self { inner }
    }

    /// Create a supervisor with an injected factory — for tests only.
    ///
    /// The factory is called once per (re)spawn.  A test that wants to simulate
    /// EOF-then-factory-failure returns `Err(WorkerSpawnFailed)` on the second call.
    #[cfg(test)]
    pub(crate) fn with_factory(factory: HalvesFactory) -> Self {
        let (events_tx, _) = broadcast::channel(64);
        let inner = Arc::new(Inner {
            pending: Mutex::new(HashMap::new()),
            chat_sink: Mutex::new(None),
            next_id: AtomicU64::new(1),
            events_tx,
            stderr_ring: Arc::new(Mutex::new(VecDeque::with_capacity(2000))),
            last_scan: Mutex::new(None),
            last_load: Mutex::new(None),
            stopped: AtomicBool::new(false),
            write_tx: Mutex::new(None),
            factory,
        });
        Self { inner }
    }

    /// Subscribe to worker lifecycle events.
    pub fn events(&self) -> broadcast::Receiver<HiggsEvent> {
        self.inner.events_tx.subscribe()
    }

    /// Return up to `n` recent stderr lines from the worker (oldest first).
    pub fn logs(&self, n: usize) -> Vec<String> {
        let ring = self.inner.stderr_ring.lock();
        let skip = ring.len().saturating_sub(n);
        ring.iter().skip(skip).cloned().collect()
    }

    /// Spawn the worker and start the reader task.
    ///
    /// Returns `Err` only if the initial spawn fails.
    pub(crate) fn start(&self) -> Result<(), HiggsError> {
        self.do_spawn()?;
        Ok(())
    }

    /// Send a request to the worker and await its response.
    ///
    /// Returns `Err(WorkerDead)` [HG007] if the worker dies before replying.
    /// Returns `Err(WorkerRpc)` [HG009] if the worker replies with a JSON-RPC error.
    pub(crate) async fn request(&self, method: &str, params: Value) -> Result<Value, HiggsError> {
        let id = self.inner.next_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = oneshot::channel();
        {
            self.inner.pending.lock().insert(id, tx);
        }
        let line = rpc::encode(&RpcFrame::Request(RpcRequest {
            jsonrpc: "2.0".into(),
            id,
            method: method.to_string(),
            params,
        }));
        // Send through the mpsc channel — the writer task owns stdin and
        // serialises all concurrent writes without a mutex on the I/O path.
        let send_result = {
            let guard = self.inner.write_tx.lock();
            match guard.as_ref() {
                Some(tx) => tx.send(line).map_err(|_| ()),
                None => Err(()),
            }
        };
        if send_result.is_err() {
            self.inner.pending.lock().remove(&id);
            return Err(HiggsError::WorkerDead {
                context: "no worker running".into(),
            });
        }
        match rx.await {
            Ok(resp) => {
                if let Some(err) = resp.error {
                    Err(HiggsError::WorkerRpc {
                        method: method.to_string(),
                        message: err.message,
                    })
                } else {
                    Ok(resp.result.unwrap_or(Value::Null))
                }
            }
            Err(_) => Err(HiggsError::WorkerDead {
                context: "worker died before response".into(),
            }),
        }
    }

    /// Install a new chat-chunk receiver for the next in-flight chat request.
    ///
    /// Only one chat may be in flight at a time (v1 invariant: the worker
    /// serialises).  The caller owns the returned receiver and reads deltas
    /// until the sender is dropped (end of stream).
    ///
    /// # Panics (debug only)
    /// `debug_assert!`s that no existing sink is live when called, catching
    /// accidental in-flight clobbering during development.
    pub(crate) fn take_chat_sink(&self) -> mpsc::UnboundedReceiver<String> {
        let (tx, rx) = mpsc::unbounded_channel();
        let mut guard = self.inner.chat_sink.lock();
        debug_assert!(
            guard.as_ref().is_none_or(mpsc::UnboundedSender::is_closed),
            "take_chat_sink called while a chat sink is already live (v1 single-in-flight invariant)"
        );
        *guard = Some(tx);
        rx
    }

    /// Clear the active chat-chunk sink.
    ///
    /// Called on the error path of a chat request to prevent a stale sender
    /// from tripping the `debug_assert!` in the next `take_chat_sink` call.
    pub(crate) fn clear_chat_sink(&self) {
        *self.inner.chat_sink.lock() = None;
    }

    /// Gracefully shut down the worker (2s timeout) then drop the write channel.
    ///
    /// Sets the deliberate-stop flag so death does not trigger respawn.
    pub(crate) async fn stop(&self) {
        self.inner.stopped.store(true, Ordering::Relaxed);
        let _ = tokio::time::timeout(
            std::time::Duration::from_secs(2),
            self.request(M_SHUTDOWN, Value::Null),
        )
        .await;
        // Drop the sender — the writer task exits when its channel closes.
        *self.inner.write_tx.lock() = None;
    }

    /// Record the params of a successful `higgs/scan` for post-restart replay.
    pub(crate) fn record_last_scan(&self, params: Value) {
        *self.inner.last_scan.lock() = Some(params);
    }

    /// Record the params of a successful `higgs/load` for post-restart replay.
    pub(crate) fn record_last_load(&self, params: Value) {
        *self.inner.last_load.lock() = Some(params);
    }

    /// Emit a lifecycle event on the broadcast channel.
    ///
    /// Used by the [`Higgs`](crate::api::Higgs) facade to publish
    /// `ModelLoaded` / `ModelUnloaded` after the corresponding RPC succeeds.
    pub(crate) fn emit(&self, event: HiggsEvent) {
        let _ = self.inner.events_tx.send(event);
    }

    /// Return a clone of the last-recorded scan params (for test introspection).
    #[cfg(test)]
    pub(crate) fn last_scan_params(&self) -> Option<Value> {
        self.inner.last_scan.lock().clone()
    }

    // ── private ──────────────────────────────────────────────────────────────

    /// Build fresh I/O halves, spawn the writer and reader tasks, and install
    /// the mpsc sender in `Inner`.
    fn do_spawn(&self) -> Result<(), HiggsError> {
        let halves = (self.inner.factory)(self.inner.stderr_ring.clone())?;
        let (write_tx, write_rx) = mpsc::unbounded_channel::<String>();
        *self.inner.write_tx.lock() = Some(write_tx);
        // Writer task: owns the write half, drains the mpsc channel.
        // Exits when the sender half is dropped (stop() or do_spawn replacing it).
        tokio::spawn(writer_task(halves.write, write_rx));
        // Reader task: owns the read half, dispatches lines, triggers restart on EOF/error.
        let inner = Arc::clone(&self.inner);
        tokio::spawn(reader_task(inner, halves.read));
        Ok(())
    }
}

/// Drains the write channel into the worker's stdin.
///
/// Exits when the sender is dropped (channel closed) or a write fails.
/// Write failure is silent here — the reader will observe the corresponding
/// EOF or error and drive the death / restart path.
async fn writer_task(mut write: WriteHalf, mut rx: mpsc::UnboundedReceiver<String>) {
    while let Some(line) = rx.recv().await {
        let bytes = line.as_bytes();
        if write.write_all(bytes).await.is_err() {
            break;
        }
        if write.write_all(b"\n").await.is_err() {
            break;
        }
        if write.flush().await.is_err() {
            break;
        }
    }
}

/// Reads lines from the worker, dispatches each, and handles death / restart.
///
/// Runs as a tokio task.  On EOF or read error:
/// 1. Fails all pending requests.
/// 2. Broadcasts `WorkerDied`.
/// 3. If not stopped: waits 1 s, attempts one respawn; on factory failure →
///    broadcasts terminal `WorkerDied` and exits.
async fn reader_task(inner: Arc<Inner>, read: ReadHalf) {
    let mut reader = BufReader::new(read).lines();

    loop {
        let line_result = reader.next_line().await;
        match line_result {
            Ok(Some(line)) => dispatch(&inner, &line),
            Ok(None) => {
                // EOF — worker exited cleanly.
                on_worker_death(&inner, None);
                if inner.stopped.load(Ordering::Relaxed) {
                    return;
                }
                // 1 s backoff then one respawn attempt.
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                match attempt_restart(&inner).await {
                    Some(new_read) => {
                        reader = BufReader::new(new_read).lines();
                        // Continue outer loop on the new transport.
                    }
                    None => return,
                }
            }
            Err(e) => {
                // I/O error on the read half.
                on_worker_death(&inner, Some(e.to_string()));
                if inner.stopped.load(Ordering::Relaxed) {
                    return;
                }
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                match attempt_restart(&inner).await {
                    Some(new_read) => {
                        reader = BufReader::new(new_read).lines();
                    }
                    None => return,
                }
            }
        }
    }
}

/// Attempt a single respawn: call factory, install new write channel, return
/// new read half on success.  Returns `None` and exits if factory fails.
async fn attempt_restart(inner: &Arc<Inner>) -> Option<ReadHalf> {
    match (inner.factory)(inner.stderr_ring.clone()) {
        Ok(halves) => {
            let (write_tx, write_rx) = mpsc::unbounded_channel::<String>();
            *inner.write_tx.lock() = Some(write_tx);
            tokio::spawn(writer_task(halves.write, write_rx));
            let _ = inner.events_tx.send(HiggsEvent::WorkerRestarted);
            info!("higgs worker restarted");
            replay_scan_then_load(inner);
            Some(halves.read)
        }
        Err(e) => {
            warn!(error = %e, "higgs worker respawn failed — giving up");
            // Doc promise: terminal factory failure → broadcasts WorkerDied before exit.
            let _ = inner.events_tx.send(HiggsEvent::WorkerDied);
            None
        }
    }
}

/// Decode one line and dispatch to the appropriate handler.
fn dispatch(inner: &Arc<Inner>, line: &str) {
    match rpc::decode(line) {
        Ok(RpcFrame::Response(resp)) => correlate(inner, resp),
        Ok(RpcFrame::Notification(notif)) => route_notification(inner, &notif),
        Ok(RpcFrame::Request(_)) => { /* workers never send requests */ }
        Err(e) => warn!(error = %e, "higgs: dropped malformed worker line"),
    }
}

/// Route a response to the waiting oneshot by id.
fn correlate(inner: &Arc<Inner>, resp: rpc::RpcResponse) {
    if let Some(tx) = inner.pending.lock().remove(&resp.id) {
        let _ = tx.send(resp);
    }
}

/// Route a `higgs/chat/chunk` notification to the active chat sink.
fn route_notification(inner: &Arc<Inner>, notif: &RpcNotification) {
    if notif.method != N_CHAT_CHUNK {
        return;
    }
    let Some(delta) = notif.params.get("delta").and_then(|v| v.as_str()) else {
        return;
    };
    let delta = delta.to_string();
    let sink = inner.chat_sink.lock();
    if let Some(tx) = sink.as_ref() {
        let _ = tx.send(delta);
    }
}

/// Handle worker death: fail all pending requests, clear write channel, broadcast event.
fn on_worker_death(inner: &Arc<Inner>, reason: Option<String>) {
    // Drain and drop all pending senders — the rx.await in `request` sees Err.
    let drained: Vec<_> = inner.pending.lock().drain().collect();
    for (_id, tx) in drained {
        drop(tx);
    }
    // Drop the write sender — the writer task exits when channel closes.
    *inner.write_tx.lock() = None;
    // Clear chat sink.
    *inner.chat_sink.lock() = None;

    let _ = inner.events_tx.send(HiggsEvent::WorkerDied);
    if let Some(r) = reason {
        warn!(detail = %r, "higgs worker died");
    } else {
        info!("higgs worker exited (EOF)");
    }
}

/// Best-effort fire-and-forget replay of one RPC request (no response tracking).
fn replay_fire_and_forget(inner: &Arc<Inner>, method: &str, params: Value) {
    let id = inner.next_id.fetch_add(1, Ordering::Relaxed);
    let line = rpc::encode(&RpcFrame::Request(RpcRequest {
        jsonrpc: "2.0".into(),
        id,
        method: method.to_string(),
        params,
    }));
    let guard = inner.write_tx.lock();
    if let Some(tx) = guard.as_ref() {
        if let Err(e) = tx.send(line) {
            warn!(error = %e, method, "higgs: replay send failed");
        }
    }
}

/// Best-effort replay of the last successful scan (if any) then load (if any) after restart.
///
/// Scan is replayed first (fire-and-forget) so the worker's model index is
/// populated before the load request arrives.  The load replay awaits its
/// response so we can emit [`HiggsEvent::ModelLoaded`] only on confirmed
/// success — replay failures are intentionally logged-and-dropped (worker just
/// restarted; user re-drives if needed).
fn replay_scan_then_load(inner: &Arc<Inner>) {
    if let Some(p) = inner.last_scan.lock().clone() {
        replay_fire_and_forget(inner, M_SCAN, p);
    }
    if let Some(load_params) = inner.last_load.lock().clone() {
        // Extract the model id from recorded params before moving them into the task.
        let id = load_params
            .get("id")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_owned();
        let inner2 = Arc::clone(inner);
        tokio::spawn(async move {
            // Replay failures are intentionally logged-and-dropped (worker just
            // restarted; user re-drives).
            match replay_load_await(&inner2, load_params).await {
                Ok(()) => {
                    let _ = inner2.events_tx.send(HiggsEvent::ModelLoaded { id });
                }
                Err(e) => {
                    warn!(error = %e, "higgs: replayed load failed after restart");
                }
            }
        });
    }
}

/// Send a `higgs/load` RPC and await its response, correlating via the pending map.
///
/// Shares the same correlation plumbing as [`Supervisor::request`] but is
/// called from within the supervisor internals after a worker restart.
async fn replay_load_await(inner: &Arc<Inner>, params: Value) -> Result<(), HiggsError> {
    let id = inner.next_id.fetch_add(1, Ordering::Relaxed);
    let (tx, rx) = oneshot::channel();
    inner.pending.lock().insert(id, tx);

    let line = rpc::encode(&RpcFrame::Request(RpcRequest {
        jsonrpc: "2.0".into(),
        id,
        method: M_LOAD.to_string(),
        params,
    }));

    let send_result = {
        let guard = inner.write_tx.lock();
        match guard.as_ref() {
            Some(tx) => tx.send(line).map_err(|_| ()),
            None => Err(()),
        }
    };
    if send_result.is_err() {
        inner.pending.lock().remove(&id);
        return Err(HiggsError::WorkerDead {
            context: "replay load: no worker running".into(),
        });
    }

    match rx.await {
        Ok(resp) => {
            if let Some(err) = resp.error {
                Err(HiggsError::WorkerRpc {
                    method: M_LOAD.into(),
                    message: err.message,
                })
            } else {
                Ok(())
            }
        }
        Err(_) => Err(HiggsError::WorkerDead {
            context: "replay load: worker died before response".into(),
        }),
    }
}

/// Production factory: re-exec current binary with `--higgs-worker`.
///
/// Spawns via `tokio::process::Command` with `stdin` + `stdout` piped.
/// Takes owned `ChildStdin` and `ChildStdout` halves — independently owned
/// by construction, no mutex between reader and writer.
/// Wires a blocking stderr drain task to fill the ring.
fn production_factory(
    stderr_ring: Arc<Mutex<VecDeque<String>>>,
) -> Result<WorkerHalves, HiggsError> {
    use std::process::Stdio;
    use tokio::process::Command;

    let exe = std::env::current_exe().map_err(|e| HiggsError::WorkerSpawnFailed { source: e })?;
    let mut child = Command::new(exe)
        .arg("--higgs-worker")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| HiggsError::WorkerSpawnFailed { source: e })?;

    // Take ownership of each half independently — no shared mutex required.
    let stdin = child.stdin.take().expect("stdin piped");
    let stdout = child.stdout.take().expect("stdout piped");
    let stderr = child.stderr.take().expect("stderr piped");

    // Drain stderr asynchronously: tokio::process::ChildStderr is AsyncRead,
    // not std::io::Read, so we read it with a tokio task using AsyncBufReadExt.
    // `child` is moved in so it stays alive until stderr drains.
    tokio::spawn(async move {
        use tokio::io::AsyncBufReadExt;
        let mut lines = tokio::io::BufReader::new(stderr).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            let mut ring = stderr_ring.lock();
            if ring.len() == 2000 {
                ring.pop_front();
            }
            ring.push_back(line);
        }
        drop(child);
    });

    Ok(WorkerHalves {
        write: Box::new(stdin),
        read: Box::new(stdout),
    })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ── Test seam ─────────────────────────────────────────────────────────────
    //
    // `tokio::io::duplex(N)` yields a bidirectional pair.  We need two
    // independent pairs:
    //   sup_write → test_read   (supervisor writes requests; test reads them)
    //   test_write → sup_read   (test writes responses; supervisor reads them)
    //
    // The factory returns `WorkerHalves { write: sup_write, read: sup_read }`.
    // The test controls `test_write` (inject responses) and `test_read` (observe requests).

    /// Build a supervisor plus test control handles.
    ///
    /// Returns `(supervisor, test_write, test_read)`.
    /// - `test_write`: write responses/notifications here (supervisor reads them).
    /// - `test_read`:  read requests the supervisor sent to the "worker".
    fn make_supervisor() -> (
        Supervisor,
        tokio::io::DuplexStream, // test writes → supervisor reads
        tokio::io::DuplexStream, // supervisor writes → test reads
    ) {
        // Pair A: supervisor write half ←→ test_read
        let (sup_write, test_read) = tokio::io::duplex(64 * 1024);
        // Pair B: test_write ←→ supervisor read half
        let (test_write, sup_read) = tokio::io::duplex(64 * 1024);

        // Wrap both halves in Arc<Mutex<Option<…>>> so the factory closure
        // (Fn, not FnOnce) can hand them out exactly once.
        let sup_write_cell = Arc::new(Mutex::new(Some(sup_write)));
        let sup_read_cell = Arc::new(Mutex::new(Some(sup_read)));

        let sup = Supervisor::with_factory(Box::new(move |_ring| {
            let write =
                sup_write_cell
                    .lock()
                    .take()
                    .ok_or_else(|| HiggsError::WorkerSpawnFailed {
                        source: std::io::Error::other("mock: no more write halves"),
                    })?;
            let read =
                sup_read_cell
                    .lock()
                    .take()
                    .ok_or_else(|| HiggsError::WorkerSpawnFailed {
                        source: std::io::Error::other("mock: no more read halves"),
                    })?;
            Ok(WorkerHalves {
                write: Box::new(write),
                read: Box::new(read),
            })
        }));

        sup.start().expect("mock start failed");

        (sup, test_write, test_read)
    }

    fn ok_response(id: u64, result: Value) -> String {
        rpc::encode(&RpcFrame::Response(rpc::RpcResponse {
            jsonrpc: "2.0".into(),
            id,
            result: Some(result),
            error: None,
        }))
    }

    fn err_response(id: u64, code: i64, message: &str) -> String {
        rpc::encode(&RpcFrame::Response(rpc::RpcResponse {
            jsonrpc: "2.0".into(),
            id,
            result: None,
            error: Some(rpc::RpcError {
                code,
                message: message.into(),
            }),
        }))
    }

    fn chunk_notif(request_id: u64, delta: &str) -> String {
        rpc::encode(&RpcFrame::Notification(rpc::RpcNotification {
            jsonrpc: "2.0".into(),
            method: N_CHAT_CHUNK.into(),
            params: json!({"request_id": request_id, "delta": delta}),
        }))
    }

    /// Write a line into `stream`, appending `\n`, and flush.
    async fn write_line(stream: &mut tokio::io::DuplexStream, line: &str) {
        use tokio::io::AsyncWriteExt;
        stream
            .write_all(format!("{line}\n").as_bytes())
            .await
            .expect("test write_line");
        stream.flush().await.expect("test flush");
    }

    // ─── Test 1: out-of-order response correlation ───────────────────────────

    #[tokio::test]
    async fn request_response_correlation() {
        let (sup, mut test_write, _test_read) = make_supervisor();

        // Issue two requests concurrently. The supervisor assigns ids 1 and 2.
        let fut1 = sup.request("higgs/ping", json!({"n": 1}));
        let fut2 = sup.request("higgs/ping", json!({"n": 2}));

        // Give both requests time to register their pending entries.
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;

        // Respond to id 2 first, then id 1.
        write_line(&mut test_write, &ok_response(2, json!({"n": 2}))).await;
        write_line(&mut test_write, &ok_response(1, json!({"n": 1}))).await;

        let (r1, r2) = tokio::join!(fut1, fut2);
        assert_eq!(r1.unwrap(), json!({"n": 1}));
        assert_eq!(r2.unwrap(), json!({"n": 2}));
    }

    // ─── Test 2: chat-chunk routing ──────────────────────────────────────────

    #[tokio::test]
    async fn chat_chunks_routed() {
        let (sup, mut test_write, _test_read) = make_supervisor();

        let mut rx = sup.take_chat_sink();

        let deltas = ["hello", " world", "!"];
        for d in &deltas {
            write_line(&mut test_write, &chunk_notif(42, d)).await;
        }
        tokio::time::sleep(std::time::Duration::from_millis(30)).await;

        for expected in &deltas {
            let got = rx.try_recv().expect("delta expected");
            assert_eq!(got, *expected);
        }
    }

    // ─── Test 3: EOF fails pending + emits WorkerDied ────────────────────────

    #[tokio::test]
    async fn eof_fails_pending_and_emits_died() {
        // Factory: first call succeeds (using duplex), second call fails (no more halves).
        let (sup_write_1, test_read_1) = tokio::io::duplex(64 * 1024);
        let (test_write_1, sup_read_1) = tokio::io::duplex(64 * 1024);

        let sup_write_cell = Arc::new(Mutex::new(Some(sup_write_1)));
        let sup_read_cell = Arc::new(Mutex::new(Some(sup_read_1)));

        let sup = Supervisor::with_factory(Box::new(move |_ring| {
            let write =
                sup_write_cell
                    .lock()
                    .take()
                    .ok_or_else(|| HiggsError::WorkerSpawnFailed {
                        source: std::io::Error::other("mock: no more halves"),
                    })?;
            let read =
                sup_read_cell
                    .lock()
                    .take()
                    .ok_or_else(|| HiggsError::WorkerSpawnFailed {
                        source: std::io::Error::other("mock: no more halves"),
                    })?;
            Ok(WorkerHalves {
                write: Box::new(write),
                read: Box::new(read),
            })
        }));
        sup.start().expect("start");

        let mut events = sup.events();

        // Register a pending request directly (bypass the network send so
        // the pending entry exists before EOF arrives).
        let (reply_tx, reply_rx) = oneshot::channel::<rpc::RpcResponse>();
        let pending_id = sup.inner.next_id.fetch_add(1, Ordering::Relaxed);
        sup.inner.pending.lock().insert(pending_id, reply_tx);

        // Give the reader task time to start.
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;

        // Trigger EOF by dropping the test write end.
        drop(test_write_1);
        drop(test_read_1);

        // The pending oneshot should be dropped (channel closed → Err).
        let result = tokio::time::timeout(std::time::Duration::from_millis(500), reply_rx).await;
        assert!(
            matches!(result, Ok(Err(_))),
            "pending request should fail on EOF"
        );

        // First WorkerDied must arrive (worker EOF).
        let first_died = tokio::time::timeout(std::time::Duration::from_millis(2000), async {
            loop {
                match events.recv().await {
                    Ok(HiggsEvent::WorkerDied) => return true,
                    Ok(_) => continue,
                    Err(_) => return false,
                }
            }
        })
        .await;
        assert!(
            matches!(first_died, Ok(true)),
            "first WorkerDied event expected (EOF)"
        );

        // After the 1s respawn backoff, the factory fails (no more halves) →
        // a second terminal WorkerDied must be broadcast before the reader task exits.
        let second_died = tokio::time::timeout(std::time::Duration::from_millis(2000), async {
            loop {
                match events.recv().await {
                    Ok(HiggsEvent::WorkerDied) => return true,
                    Ok(_) => continue,
                    Err(_) => return false,
                }
            }
        })
        .await;
        assert!(
            matches!(second_died, Ok(true)),
            "second WorkerDied event expected (factory failure)"
        );
    }

    // ─── Test 4: worker RPC error maps to HG009 ──────────────────────────────

    #[tokio::test]
    async fn worker_error_maps_to_hg009() {
        let (sup, mut test_write, _test_read) = make_supervisor();

        let fut = sup.request(M_LOAD, json!({"id": "org/bad"}));
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        write_line(
            &mut test_write,
            &err_response(1, -32000, "model file corrupt"),
        )
        .await;

        let err = fut.await.expect_err("should be an error");
        let display = err.to_string();
        assert!(display.contains("[HG009]"), "display: {display}");
    }

    // ─── Test 5: stderr ring caps at 2000 ────────────────────────────────────

    #[tokio::test]
    async fn logs_ring_caps() {
        // Verify ring-cap logic directly on a standalone ring (the production
        // path fills this via a stderr drain thread; the mechanics are identical).
        let ring: Arc<Mutex<VecDeque<String>>> =
            Arc::new(Mutex::new(VecDeque::with_capacity(2000)));

        for i in 0..2100usize {
            let mut r = ring.lock();
            if r.len() == 2000 {
                r.pop_front();
            }
            r.push_back(format!("line-{i}"));
        }

        let r = ring.lock();
        assert_eq!(r.len(), 2000);
        // 2100 pushed, 100 dropped → oldest remaining is line-100.
        assert_eq!(r.front().unwrap(), "line-100");
    }

    // ─── Test 6: restart replays scan before load and emits ModelLoaded ─────────
    //
    // The mock factory hands out two transport pairs:
    //   pair-1: first worker lifetime (killed by dropping test_write_1)
    //   pair-2: second worker lifetime (receives replayed RPCs; the test drives
    //           it by reading the replayed higgs/load and writing back an OK)
    //
    // Duplex wiring — each pair (A, B) means writing to A comes out of B:
    //   sup_write_1  ↔ _obs_rx_1  : supervisor writes requests; test ignores them
    //   test_write_1 ↔ sup_read_1 : drop triggers EOF on supervisor's read half
    //   sup_write_2  ↔ obs_rx_2   : supervisor writes replayed msgs; test reads here
    //   test_tx_2    ↔ sup_read_2 : test writes mock OK responses back to supervisor

    #[tokio::test]
    async fn restart_replays_scan_then_load() {
        let (sup_write_1, _obs_rx_1) = tokio::io::duplex(64 * 1024);
        let (test_write_1, sup_read_1) = tokio::io::duplex(64 * 1024);

        let (sup_write_2, mut obs_rx_2) = tokio::io::duplex(64 * 1024);
        let (mut test_tx_2, sup_read_2) = tokio::io::duplex(64 * 1024);

        let cell_sup_write_1 = Arc::new(Mutex::new(Some(sup_write_1)));
        let cell_sup_read_1 = Arc::new(Mutex::new(Some(sup_read_1)));
        let cell_sup_write_2 = Arc::new(Mutex::new(Some(sup_write_2)));
        let cell_sup_read_2 = Arc::new(Mutex::new(Some(sup_read_2)));

        let call_count = Arc::new(std::sync::atomic::AtomicU32::new(0));
        let call_count2 = Arc::clone(&call_count);

        let sup =
            Supervisor::with_factory(Box::new(move |_ring| {
                let n = call_count2.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if n == 0 {
                    let write = cell_sup_write_1.lock().take().unwrap();
                    let read = cell_sup_read_1.lock().take().unwrap();
                    Ok(WorkerHalves {
                        write: Box::new(write),
                        read: Box::new(read),
                    })
                } else {
                    let write = cell_sup_write_2.lock().take().ok_or_else(|| {
                        HiggsError::WorkerSpawnFailed {
                            source: std::io::Error::other(
                                "mock: second factory call after cells exhausted",
                            ),
                        }
                    })?;
                    let read = cell_sup_read_2.lock().take().ok_or_else(|| {
                        HiggsError::WorkerSpawnFailed {
                            source: std::io::Error::other(
                                "mock: second factory call after cells exhausted",
                            ),
                        }
                    })?;
                    Ok(WorkerHalves {
                        write: Box::new(write),
                        read: Box::new(read),
                    })
                }
            }));
        sup.start().expect("start");

        let mut events = sup.events();

        // Record scan and load so replay has something to send.
        sup.record_last_scan(json!({"dirs": ["/models"]}));
        sup.record_last_load(json!({"id": "org/model"}));

        // Wait for the reader task to settle, then trigger EOF on first transport.
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        drop(test_write_1);

        // After 1s backoff + restart, the second transport receives
        // higgs/scan first, then higgs/load.  Read them from obs_rx_2, which
        // is the duplex peer of sup_write_2 — data written to sup_write_2 by
        // the supervisor's writer task comes out here.
        let deadline = std::time::Duration::from_millis(2500);
        let received = tokio::time::timeout(deadline, async {
            use tokio::io::AsyncBufReadExt;
            let mut lines = BufReader::new(&mut obs_rx_2).lines();
            let mut msgs = Vec::new();
            // Collect exactly 2 lines.
            while msgs.len() < 2 {
                match lines.next_line().await {
                    Ok(Some(l)) => msgs.push(l),
                    _ => break,
                }
            }
            msgs
        })
        .await
        .expect("timeout waiting for replayed messages");

        assert_eq!(
            received.len(),
            2,
            "expected exactly 2 replayed messages, got: {received:?}"
        );

        let first: serde_json::Value = serde_json::from_str(&received[0]).expect("valid json");
        let second: serde_json::Value = serde_json::from_str(&received[1]).expect("valid json");
        assert_eq!(
            first["method"], M_SCAN,
            "first replayed method must be higgs/scan"
        );
        assert_eq!(
            second["method"], M_LOAD,
            "second replayed method must be higgs/load"
        );
        assert_eq!(first["params"], json!({"dirs": ["/models"]}));
        assert_eq!(second["params"], json!({"id": "org/model"}));

        // Reply to the replayed higgs/load with an OK response so the await
        // path resolves and ModelLoaded is emitted.
        let load_id = second["id"]
            .as_u64()
            .expect("load request must carry an id");
        let ok_line = ok_response(load_id, json!({"id": "org/model"}));
        write_line(&mut test_tx_2, &ok_line).await;

        // ModelLoaded must arrive on the event channel.
        let got_loaded = tokio::time::timeout(std::time::Duration::from_millis(500), async {
            loop {
                match events.recv().await {
                    Ok(HiggsEvent::ModelLoaded { id }) => return Some(id),
                    Ok(_) => continue,
                    Err(_) => return None,
                }
            }
        })
        .await
        .expect("timeout waiting for ModelLoaded");

        assert_eq!(
            got_loaded.as_deref(),
            Some("org/model"),
            "ModelLoaded must carry the replayed model id"
        );
    }
}
