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
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::Command;
use tokio::sync::{broadcast, mpsc, oneshot};
use tracing::{info, warn};

use crate::diagnostic::HiggsError;
use crate::rpc::{self, RpcFrame, RpcNotification, RpcRequest};
use crate::worker::{M_LOAD, M_SHUTDOWN, N_CHAT_CHUNK};

/// How long `stop()` waits for the worker to exit on its own (after stdin
/// closes) before SIGKILL-ing it. Generous enough for the child to free a
/// loaded model and run its at-exit handlers; bounded so a wedged worker can
/// never outlive its supervisor.
const WORKER_EXIT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

/// Bound on a single control RPC (scan/load/status/unload) round-trip. These
/// are not streaming and complete quickly — even loading a large GGUF is well
/// under this. The bound exists so a worker that is ALIVE but wedged (e.g. an
/// FFI call hung inside llama.cpp, or stdout polluted so its real response is
/// dropped on decode) can never hang the caller forever — without it,
/// `rx.await` blocks indefinitely since no EOF arrives to drain `pending`.
/// Chat (M_CHAT via `request_with_id`) is intentionally NOT bounded here: it is
/// already capped by `max_tokens` and streams progress chunks.
const CONTROL_RPC_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(120);

higgs_ts! {
    /// Events the host application subscribes to.
    #[derive(Debug, Clone, serde::Serialize)]
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
    /// Live child process handle, for the production factory only.
    ///
    /// `stop()` uses it to wait for the worker to exit on its own (after stdin
    /// closes) and to SIGKILL it as a last resort. `None` for the in-memory
    /// test factory, which has no OS process to reap.
    pub(crate) proc: Option<tokio::process::Child>,
}

/// Factory: produces fresh I/O halves on each (re)spawn.
///
/// Receives the stderr ring so the production impl can wire its drain task.
/// Test factories receive the ring too but may ignore it.
/// The `&str` model argument is the model id to be loaded; the production impl
/// stamps it into the worker's argv0 (`higgs(<model>)`) so the process is
/// identifiable in `ps`. It is cosmetic only — the model still loads via M_LOAD.
type HalvesFactory = Box<
    dyn Fn(Arc<Mutex<VecDeque<String>>>, &str) -> Result<WorkerHalves, HiggsError> + Send + Sync,
>;

/// Shared supervisor state.
struct Inner {
    /// Pending request id → oneshot reply channel.
    /// Lock held for insert/remove only — never across `.await`.
    pending: Mutex<HashMap<u64, oneshot::Sender<rpc::RpcResponse>>>,
    /// Per-request chat-chunk sinks keyed by the M_CHAT request id.
    ///
    /// When the worker emits an `N_CHAT_CHUNK` notification it echoes the
    /// `request_id` from the original M_CHAT params.  Each concurrent caller
    /// registers its own `mpsc::unbounded_channel` here before sending the
    /// M_CHAT request; `route_notification` delivers each delta to the
    /// matching sink.  Sinks are removed on completion or error.
    ///
    /// Worker execution is still serialised (single-threaded stdin loop); the
    /// map just ensures each caller gets ONLY its own deltas with no clobber.
    /// Lock held for insert/remove/lookup only — never across `.await`.
    chat_sinks: Mutex<HashMap<u64, mpsc::UnboundedSender<String>>>,
    /// Monotonically increasing request id counter.
    next_id: AtomicU64,
    /// Broadcast channel for lifecycle events (cap 64).
    events_tx: broadcast::Sender<HiggsEvent>,
    /// Ring buffer of recent stderr lines (cap 2000).
    /// `Arc`-wrapped so the factory closure can clone the handle.
    stderr_ring: Arc<Mutex<VecDeque<String>>>,
    /// Params of the last successful `higgs/load`; replayed after restart.
    last_load: Mutex<Option<Value>>,
    /// Set on `stop()` — suppresses respawn after death.
    stopped: AtomicBool,
    /// True for the entire lifetime of a live worker — from the spawn that
    /// started it through any in-loop restart, until the reader gives up or
    /// `stop()` reaps it. `do_spawn` refuses to start a second worker while
    /// this is set, which guarantees a single reader task: without it, a
    /// `start()` (e.g. `POST /api/higgs/worker/start`) on a running worker
    /// would spawn a second reader whose later EOF clears the live worker's
    /// `write_tx`. Set across restart so the death→respawn window is also
    /// covered.
    running: AtomicBool,
    /// mpsc sender into the writer task for the active worker lifetime.
    /// `None` when no worker is live.  Replacing this drops the previous
    /// sender, which signals the old writer task to exit.
    write_tx: Mutex<Option<mpsc::UnboundedSender<String>>>,
    /// Live worker child process for the current lifetime (production only).
    ///
    /// Held so `stop()` can wait for the worker to exit cleanly after stdin
    /// closes — which lets the child run its at-exit handlers (under coverage
    /// instrumentation, this is what flushes the worker's profile) — and
    /// SIGKILL it only as a timeout fallback. `None` under the test factory.
    proc: tokio::sync::Mutex<Option<tokio::process::Child>>,
    /// Factory that builds fresh I/O halves on each (re)spawn.
    factory: HalvesFactory,
}

/// Worker process supervisor.
///
/// Spawns the higgs worker by re-executing the current binary with
/// `--higgs-worker`, speaks NDJSON JSON-RPC 2.0 over its stdio, correlates
/// request/response pairs by id, routes `higgs/chat/chunk` notifications to
/// per-request receivers keyed by `request_id`, and restarts the worker (1s
/// backoff) on death (at most one attempt per death; factory failure →
/// terminal `WorkerDied`, no retry loop).
///
/// Concurrent callers are each routed their own deltas via the keyed sink map;
/// the worker serialises execution (single-threaded stdin loop) so throughput
/// is single-sequence but correctness is guaranteed for any number of callers.
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
            chat_sinks: Mutex::new(HashMap::new()),
            next_id: AtomicU64::new(1),
            events_tx,
            stderr_ring: Arc::new(Mutex::new(VecDeque::with_capacity(2000))),
            last_load: Mutex::new(None),
            stopped: AtomicBool::new(false),
            running: AtomicBool::new(false),
            write_tx: Mutex::new(None),
            proc: tokio::sync::Mutex::new(None),
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
            chat_sinks: Mutex::new(HashMap::new()),
            next_id: AtomicU64::new(1),
            events_tx,
            stderr_ring: Arc::new(Mutex::new(VecDeque::with_capacity(2000))),
            last_load: Mutex::new(None),
            stopped: AtomicBool::new(false),
            running: AtomicBool::new(false),
            write_tx: Mutex::new(None),
            proc: tokio::sync::Mutex::new(None),
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

    /// Spawn a worker (named `higgs(<model>)` in `ps`) and start the reader task.
    ///
    /// Called by `load()` when no worker is live. `model` is stamped into the
    /// worker's argv0 only — the model still loads via the M_LOAD RPC. Idempotent:
    /// a call while a worker is already running is a no-op (single-reader invariant).
    /// Returns `Err` only if the spawn fails.
    pub(crate) fn start_for(&self, model: &str) -> Result<(), HiggsError> {
        self.do_spawn(model)?;
        Ok(())
    }

    /// Allocate a monotonically increasing request id.
    ///
    /// Used by callers that must register a chat sink BEFORE sending the
    /// request — the same id is used for the RPC frame and for `request_id`
    /// in the M_CHAT params so chunk routing matches.
    pub(crate) fn alloc_request_id(&self) -> u64 {
        self.inner.next_id.fetch_add(1, Ordering::Relaxed)
    }

    /// Send a request to the worker and await its response.
    ///
    /// Returns `Err(WorkerDead)` [HG007] if the worker dies before replying.
    /// Returns `Err(WorkerRpc)` [HG009] if the worker replies with a JSON-RPC error.
    pub(crate) async fn request(&self, method: &str, params: Value) -> Result<Value, HiggsError> {
        let id = self.alloc_request_id();
        match tokio::time::timeout(CONTROL_RPC_TIMEOUT, self.send_request(id, method, params)).await
        {
            Ok(result) => result,
            Err(_) => {
                // Timed out: drop the orphaned pending entry so it doesn't leak,
                // and surface a worker-unavailable error rather than hanging.
                self.inner.pending.lock().remove(&id);
                warn!(method, "higgs: control RPC timed out");
                Err(HiggsError::WorkerDead {
                    context: format!("{method} timed out after {CONTROL_RPC_TIMEOUT:?}"),
                })
            }
        }
    }

    /// Send a request using a pre-allocated id.
    ///
    /// Used for M_CHAT so the caller can register the chat sink under the
    /// SAME id before sending the request — ensuring chunk routing matches.
    pub(crate) async fn request_with_id(
        &self,
        id: u64,
        method: &str,
        params: Value,
    ) -> Result<Value, HiggsError> {
        self.send_request(id, method, params).await
    }

    /// Register a per-request chat-chunk receiver keyed by `request_id`.
    ///
    /// Creates a `(tx, rx)` pair, inserts `tx` into `chat_sinks[request_id]`,
    /// and returns `rx`.  Infallible: concurrent callers each get their own
    /// channel and their own deltas routed independently.  The caller must
    /// call [`remove_chat_sink`](Self::remove_chat_sink) when done.
    pub(crate) fn register_chat_sink(&self, request_id: u64) -> mpsc::UnboundedReceiver<String> {
        let (tx, rx) = mpsc::unbounded_channel();
        self.inner.chat_sinks.lock().insert(request_id, tx);
        rx
    }

    /// Remove the chat-chunk sink for `request_id`.
    ///
    /// Called on completion or error to release the map entry.  The dropped
    /// sender closes the receiver, signalling end-of-stream to the consumer.
    pub(crate) fn remove_chat_sink(&self, request_id: u64) {
        self.inner.chat_sinks.lock().remove(&request_id);
    }

    /// Gracefully shut down the worker, then wait for the process to exit.
    ///
    /// Sets the deliberate-stop flag so death does not trigger respawn, sends a
    /// best-effort `higgs/shutdown` (2s timeout), then drops the write channel —
    /// closing the worker's stdin so its read loop hits EOF and `worker_main()`
    /// returns normally. A clean exit lets the child run its at-exit handlers
    /// (under coverage instrumentation, that is what flushes the worker's
    /// profile). It is then reaped, waiting up to `WORKER_EXIT_TIMEOUT` for a
    /// self-exit and SIGKILL-ing as a last resort so a wedged worker can never
    /// outlive its supervisor.
    pub(crate) async fn stop(&self) {
        // Release so a racing attempt_restart's Acquire load observes the stop
        // before it installs a respawned worker (F1 deliberate-unload race).
        self.inner.stopped.store(true, Ordering::Release);
        let _ = tokio::time::timeout(
            std::time::Duration::from_secs(2),
            self.request(M_SHUTDOWN, Value::Null),
        )
        .await;
        // Drop the sender — the writer task exits when its channel closes, which
        // drops the worker's stdin and signals EOF to its read loop.
        *self.inner.write_tx.lock() = None;
        // Reap the process: prefer a clean self-exit (flushes coverage),
        // fall back to SIGKILL on timeout.
        let mut guard = self.inner.proc.lock().await;
        if let Some(mut child) = guard.take() {
            match tokio::time::timeout(WORKER_EXIT_TIMEOUT, child.wait()).await {
                Ok(_) => {} // worker exited on its own — at-exit handlers ran.
                Err(_) => {
                    // Wedged: force-kill and reap so we never leak the process.
                    let _ = child.start_kill();
                    let _ = child.wait().await;
                }
            }
        }
        // Release the lifetime flag synchronously so a subsequent start() can
        // spawn without racing the reader task's own clear on EOF.
        self.inner.running.store(false, Ordering::Release);
    }

    /// Record the params of a successful `higgs/load` for post-restart replay.
    pub(crate) fn record_last_load(&self, params: Value) {
        *self.inner.last_load.lock() = Some(params);
    }

    /// Forget the recorded `higgs/load` replay params — called after an explicit
    /// unload so a later unexpected worker restart does NOT reload the model the
    /// user just unloaded.
    pub(crate) fn clear_last_load(&self) {
        *self.inner.last_load.lock() = None;
    }

    /// Emit a lifecycle event on the broadcast channel.
    ///
    /// Used by the [`Higgs`](crate::api::Higgs) facade to publish
    /// `ModelLoaded` / `ModelUnloaded` after the corresponding RPC succeeds.
    pub(crate) fn emit(&self, event: HiggsEvent) {
        let _ = self.inner.events_tx.send(event);
    }

    /// Return a clone of the last-recorded load params (for test introspection).
    #[cfg(test)]
    pub(crate) fn last_load_params(&self) -> Option<Value> {
        self.inner.last_load.lock().clone()
    }

    /// Return the number of active chat sinks (for test introspection).
    #[cfg(test)]
    pub(crate) fn chat_sinks_count(&self) -> usize {
        self.inner.chat_sinks.lock().len()
    }

    // ── private ──────────────────────────────────────────────────────────────

    /// Send a request frame with the given `id` and await its response.
    ///
    /// Shared implementation used by both [`request`](Self::request) (which
    /// allocates a fresh id) and [`request_with_id`](Self::request_with_id)
    /// (which uses a pre-allocated id from [`alloc_request_id`](Self::alloc_request_id)).
    async fn send_request(
        &self,
        id: u64,
        method: &str,
        params: Value,
    ) -> Result<Value, HiggsError> {
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
                    // Recover the worker's origin diagnostic code (HG002/HG005/…)
                    // from the JSON-RPC `data` so the boundary maps the true
                    // status rather than collapsing to 500.
                    let worker_code = err
                        .data
                        .as_ref()
                        .and_then(|d| d.get("code"))
                        .and_then(|c| c.as_str())
                        .map(ToOwned::to_owned);
                    Err(HiggsError::WorkerRpc {
                        method: method.to_string(),
                        message: err.message,
                        worker_code,
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

    /// Build fresh I/O halves, spawn the writer and reader tasks, and install
    /// the mpsc sender in `Inner`.
    ///
    /// Resets `stopped` to `false` before launching so a stop→start cycle
    /// restores normal auto-restart behavior (F2: stop() sets stopped=true but
    /// start() must allow auto-restart on the new worker lifetime).
    fn do_spawn(&self, model: &str) -> Result<(), HiggsError> {
        // Refuse to spawn a second worker while one is already live (single-reader
        // invariant — see `Inner::running`). Idempotent: a redundant start() on a
        // running worker is a no-op, not an error or a leaked second process.
        if self.inner.running.swap(true, Ordering::AcqRel) {
            warn!("higgs: start() called while worker already running — ignoring");
            return Ok(());
        }
        // Reset the deliberate-stop flag so the new worker's reader_task
        // will auto-restart on unexpected death (stop→start cycle fix).
        self.inner.stopped.store(false, Ordering::Relaxed);
        let halves = match (self.inner.factory)(self.inner.stderr_ring.clone(), model) {
            Ok(h) => h,
            Err(e) => {
                // Spawn failed: release the running flag so a later retry can spawn.
                self.inner.running.store(false, Ordering::Release);
                return Err(e);
            }
        };
        // Stash the child handle so stop() can wait on / kill it. No concurrent
        // holder exists at (re)spawn time — stop() is terminal and never races a
        // spawn — so a non-blocking lock is sufficient and avoids blocking_lock
        // inside the async runtime.
        if let Ok(mut guard) = self.inner.proc.try_lock() {
            *guard = halves.proc;
        }
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
/// 2. Broadcasts `WorkerDied` only for genuine unexpected death (not clean stop).
/// 3. If not stopped: waits 1 s, re-checks `stopped` after the sleep, then
///    attempts one respawn; on factory failure → broadcasts terminal `WorkerDied`
///    and exits.
async fn reader_task(inner: Arc<Inner>, read: ReadHalf) {
    let mut reader = BufReader::new(read).lines();

    // Every terminal exit `break`s so the single post-loop clear releases the
    // `running` lifetime flag exactly once. A successful restart `continue`s the
    // loop on the new transport without clearing it (the lifetime persists).
    loop {
        let line_result = reader.next_line().await;
        match line_result {
            Ok(Some(line)) => dispatch(&inner, &line),
            Ok(None) => {
                // EOF — worker exited.
                let deliberate = inner.stopped.load(Ordering::Relaxed);
                // Pass `deliberate` so on_worker_death suppresses the event on clean stop.
                on_worker_death(&inner, None, deliberate);
                if deliberate {
                    break;
                }
                // 1 s backoff, then re-check stopped before respawning.
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                if inner.stopped.load(Ordering::Relaxed) {
                    break;
                }
                match attempt_restart(&inner).await {
                    Some(new_read) => {
                        reader = BufReader::new(new_read).lines();
                        // Continue outer loop on the new transport.
                    }
                    None => break,
                }
            }
            Err(e) => {
                // I/O error on the read half.
                let deliberate = inner.stopped.load(Ordering::Relaxed);
                on_worker_death(&inner, Some(e.to_string()), deliberate);
                if deliberate {
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                // Re-check stopped after backoff sleep (F1: stop() may have raced).
                if inner.stopped.load(Ordering::Relaxed) {
                    break;
                }
                match attempt_restart(&inner).await {
                    Some(new_read) => {
                        reader = BufReader::new(new_read).lines();
                    }
                    None => break,
                }
            }
        }
    }
    // Lifetime over (clean stop or respawn give-up): release the flag so a
    // future start() can spawn a fresh worker. Idempotent with stop()'s clear.
    inner.running.store(false, Ordering::Release);
}

/// Attempt a single respawn: call factory, install new write channel, return
/// new read half on success.  Returns `None` and exits if factory fails.
async fn attempt_restart(inner: &Arc<Inner>) -> Option<ReadHalf> {
    // Stamp the respawned worker's argv0 with the loaded model id (if any) so it
    // is identifiable in `ps`; the model is still re-loaded via the M_LOAD replay.
    let model = inner
        .last_load
        .lock()
        .as_ref()
        .and_then(|p| p.get("id").and_then(|v| v.as_str()).map(ToOwned::to_owned))
        .unwrap_or_default();
    match (inner.factory)(inner.stderr_ring.clone(), &model) {
        Ok(halves) => {
            // Race guard: stop()/unload() may have set `stopped` after the
            // reader's pre-call check but while factory() was spawning. If so,
            // reap the just-spawned child and abandon the restart WITHOUT
            // installing it or replaying the load — never resurrect a worker the
            // user explicitly unloaded.
            if inner.stopped.load(Ordering::Acquire) {
                if let Some(mut child) = halves.proc {
                    // start_kill() only sends the signal; wait() reaps so the
                    // abandoned child never lingers as a zombie (mirrors stop()).
                    let _ = child.start_kill();
                    let _ = child.wait().await;
                }
                return None;
            }
            // Stash the restarted child so stop() can later wait on / SIGKILL it.
            // do_spawn does this for the initial spawn; the respawn path must too,
            // or a worker restarted after a death would be unreapable by stop().
            //
            // First reap the OLD child being overwritten: it already exited (its
            // EOF is what triggered this restart), but `tokio::process::Child`
            // does NOT reap on drop, so dropping it without `wait()` leaves a
            // zombie. Take it out and wait — it returns immediately since the
            // process is already gone.
            let mut proc_guard = inner.proc.lock().await;
            if let Some(mut old) = proc_guard.take() {
                let _ = old.wait().await;
            }
            *proc_guard = halves.proc;
            drop(proc_guard);
            let (write_tx, write_rx) = mpsc::unbounded_channel::<String>();
            *inner.write_tx.lock() = Some(write_tx);
            tokio::spawn(writer_task(halves.write, write_rx));
            let _ = inner.events_tx.send(HiggsEvent::WorkerRestarted);
            info!("higgs worker restarted");
            replay_load(inner);
            Some(halves.read)
        }
        Err(e) => {
            warn!(error = %e, "higgs worker respawn failed — giving up");
            // Reap the OLD child too: its EOF triggered this restart so it has
            // already exited, but `tokio::process::Child` does not reap on drop —
            // without `wait()` it lingers as a zombie. (Symmetric to the Ok branch.)
            let mut proc_guard = inner.proc.lock().await;
            if let Some(mut old) = proc_guard.take() {
                let _ = old.wait().await;
            }
            drop(proc_guard);
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

/// Route a `higgs/chat/chunk` notification to the matching per-request sink.
///
/// The notification params carry `request_id` (echoed from the M_CHAT request)
/// and `delta`.  The sink map is looked up by `request_id`; if no entry exists
/// (sink already removed or unknown id) the chunk is silently dropped.
fn route_notification(inner: &Arc<Inner>, notif: &RpcNotification) {
    if notif.method != N_CHAT_CHUNK {
        return;
    }
    let Some(request_id) = notif
        .params
        .get("request_id")
        .and_then(serde_json::Value::as_u64)
    else {
        return;
    };
    let Some(delta) = notif.params.get("delta").and_then(|v| v.as_str()) else {
        return;
    };
    let delta = delta.to_string();
    let sinks = inner.chat_sinks.lock();
    if let Some(tx) = sinks.get(&request_id) {
        let _ = tx.send(delta);
    }
}

/// Handle worker death: fail all pending requests, clear write channel, and
/// (for unexpected deaths only) broadcast `WorkerDied` and log at warn.
///
/// `deliberate` is true when `stop()` was called before the EOF was observed —
/// in that case neither the broadcast event nor the warn log is emitted (clean
/// shutdown is not a death event). This enforces four-pillar pillar 3: log once
/// at origin, not at every boundary that observes the same event.
fn on_worker_death(inner: &Arc<Inner>, reason: Option<String>, deliberate: bool) {
    // Drain and drop all pending senders — the rx.await in `request` sees Err.
    let drained: Vec<_> = inner.pending.lock().drain().collect();
    for (_id, tx) in drained {
        drop(tx);
    }
    // Drop the write sender — the writer task exits when channel closes.
    *inner.write_tx.lock() = None;
    // Drop all chat sinks — closes each receiver, ending all in-flight streams.
    inner.chat_sinks.lock().clear();

    if deliberate {
        // Clean stop: no event, no warn — the shutdown was requested.
        info!("higgs worker stopped (deliberate)");
        return;
    }

    // Unexpected death: broadcast event (origin log once here).
    let _ = inner.events_tx.send(HiggsEvent::WorkerDied);
    if let Some(r) = reason {
        warn!(detail = %r, "higgs worker died");
    } else {
        warn!("higgs worker exited unexpectedly (EOF)");
    }
}

/// Best-effort replay of the last successful load (if any) after restart.
///
/// Scan is host-side now and needs no replay; only the load is re-driven so a
/// model resident before the death is reloaded into the respawned worker. The
/// load replay awaits its response so we emit [`HiggsEvent::ModelLoaded`] only on
/// confirmed success. Replay failure is intentionally logged-and-dropped (worker
/// just restarted; the user re-drives if needed).
fn replay_load(inner: &Arc<Inner>) {
    let Some(load_params) = inner.last_load.lock().clone() else {
        return;
    };

    let inner2 = Arc::clone(inner);
    tokio::spawn(async move {
        let id = load_params
            .get("id")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_owned();
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

/// Send an RPC request and await its response, correlating via the pending map.
///
/// Used for replay of scan (and any future awaited replay RPC). Returns the
/// raw result value on success or a typed error on failure.
async fn replay_rpc_await(
    inner: &Arc<Inner>,
    method: &str,
    params: Value,
) -> Result<Value, HiggsError> {
    let id = inner.next_id.fetch_add(1, Ordering::Relaxed);
    let (tx, rx) = oneshot::channel();
    inner.pending.lock().insert(id, tx);

    let line = rpc::encode(&RpcFrame::Request(RpcRequest {
        jsonrpc: "2.0".into(),
        id,
        method: method.to_string(),
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
            context: format!("replay {method}: no worker running"),
        });
    }

    // Bound the await like `request()` does: a respawned worker that accepts
    // stdin but wedges would otherwise leak this task and the `pending[id]`
    // entry forever (no EOF arrives to drain it). On timeout, remove the
    // orphaned pending entry (same cleanup `request()` does) and give up — no
    // retry loop.
    match tokio::time::timeout(CONTROL_RPC_TIMEOUT, rx).await {
        Ok(Ok(resp)) => {
            if let Some(err) = resp.error {
                let worker_code = err
                    .data
                    .as_ref()
                    .and_then(|d| d.get("code"))
                    .and_then(|c| c.as_str())
                    .map(ToOwned::to_owned);
                Err(HiggsError::WorkerRpc {
                    method: method.to_string(),
                    message: err.message,
                    worker_code,
                })
            } else {
                Ok(resp.result.unwrap_or(Value::Null))
            }
        }
        Ok(Err(_)) => Err(HiggsError::WorkerDead {
            context: format!("replay {method}: worker died before response"),
        }),
        Err(_) => {
            inner.pending.lock().remove(&id);
            warn!(method, "higgs: replay RPC timed out");
            Err(HiggsError::WorkerDead {
                context: format!("replay {method} timed out after {CONTROL_RPC_TIMEOUT:?}"),
            })
        }
    }
}

/// Send a `higgs/load` RPC and await its response via the shared replay helper.
///
/// Delegates to [`replay_rpc_await`] with `M_LOAD`; exists as a named function
/// to keep `replay_scan_then_load` readable.
async fn replay_load_await(inner: &Arc<Inner>, params: Value) -> Result<(), HiggsError> {
    replay_rpc_await(inner, M_LOAD, params).await.map(|_| ())
}

/// Production factory: re-exec current binary with `--higgs-worker`.
///
/// Spawns via `tokio::process::Command` with `stdin` + `stdout` piped.
/// Takes owned `ChildStdin` and `ChildStdout` halves — independently owned
/// by construction, no mutex between reader and writer.
/// Wires a blocking stderr drain task to fill the ring.
fn production_factory(
    stderr_ring: Arc<Mutex<VecDeque<String>>>,
    model: &str,
) -> Result<WorkerHalves, HiggsError> {
    let exe = std::env::current_exe().map_err(|e| HiggsError::WorkerSpawnFailed { source: e })?;
    let mut cmd = Command::new(exe);
    // Stamp argv0 as `higgs(<model>)` so the worker is identifiable in `ps`.
    // Cosmetic only — the model still loads via the M_LOAD RPC. `tokio::process::
    // Command::arg0` is cross-platform (no-op effect on platforms without argv0).
    // Truncate the label to a char-boundary-safe 64 chars: a pathological model
    // id would otherwise grow argv unbounded and risk E2BIG at spawn (HG006),
    // turning a normal not-found into a spawn failure. RPC `id` is unaffected.
    let label: String = model.chars().take(64).collect();
    cmd.arg0(format!("higgs({label})"));
    let mut child = cmd
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
    // The drain owns only the stderr half; the `child` handle now lives in
    // `Inner.proc` so `stop()` can wait on / kill the process. When the worker
    // exits, the stderr pipe closes and this task ends.
    tokio::spawn(async move {
        let mut lines = tokio::io::BufReader::new(stderr).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            let mut ring = stderr_ring.lock();
            if ring.len() == 2000 {
                ring.pop_front();
            }
            ring.push_back(line);
        }
    });

    Ok(WorkerHalves {
        write: Box::new(stdin),
        read: Box::new(stdout),
        proc: Some(child),
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

        let sup = Supervisor::with_factory(Box::new(move |_ring, _model| {
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
                proc: None,
            })
        }));

        sup.start_for("test-model").expect("mock start failed");

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
                data: None,
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

    // ─── Test 1-a: spawn-on-start_for then kill-on-stop lifecycle ────────────
    //
    // No worker is live until start_for; a request then correlates. After stop()
    // the write_tx is cleared so a subsequent request fails WorkerDead — proving
    // stop() tears the worker down (kill-on-unload at the supervisor layer).

    #[tokio::test]
    async fn start_for_then_stop_lifecycle() {
        let (sup_write, _test_read) = tokio::io::duplex(64 * 1024);
        let (mut test_write, sup_read) = tokio::io::duplex(64 * 1024);
        let sup_write_cell = Arc::new(Mutex::new(Some(sup_write)));
        let sup_read_cell = Arc::new(Mutex::new(Some(sup_read)));

        let sup = Supervisor::with_factory(Box::new(move |_ring, _model| {
            Ok(WorkerHalves {
                write: Box::new(sup_write_cell.lock().take().expect("one spawn")),
                read: Box::new(sup_read_cell.lock().take().expect("one spawn")),
                proc: None,
            })
        }));

        // Before start_for: no write_tx → request fails immediately.
        assert!(
            sup.request("higgs/ping", json!({})).await.is_err(),
            "no worker before start_for"
        );

        // start_for spawns a worker; a request now correlates. The pre-spawn
        // request already consumed id 1, so this one is id 2.
        sup.start_for("org/model").expect("spawn");
        let fut = sup.request("higgs/ping", json!({}));
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        write_line(&mut test_write, &ok_response(2, json!({"ok": true}))).await;
        assert_eq!(fut.await.unwrap(), json!({"ok": true}));

        // stop() kills the worker; a later request fails WorkerDead.
        sup.stop().await;
        let err = sup.request("higgs/ping", json!({})).await.unwrap_err();
        assert!(err.to_string().contains("[HG007]"), "display: {err}");
    }

    // ─── Test 1-b: redundant start() is an idempotent no-op ──────────────────
    //
    // The mock factory hands out exactly one set of duplex halves, so a second
    // real spawn would fail (`mock: no more write halves`). The `running` guard
    // means a `start()` on a live worker is never a second spawn — it returns
    // Ok without touching the factory, preserving the single reader and the
    // original transport. Guards the supervisor.rs:225 "start while running"
    // race that would otherwise let an old reader clear the new write_tx.

    #[tokio::test]
    async fn redundant_start_is_noop() {
        let (sup, mut test_write, _test_read) = make_supervisor();

        // Already started by make_supervisor; a second start must not spawn.
        sup.start_for("test-model")
            .expect("redundant start is a no-op, not a second spawn");

        // The original worker transport is intact: a request still correlates.
        let fut = sup.request("higgs/ping", json!({}));
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        write_line(&mut test_write, &ok_response(1, json!({"ok": true}))).await;
        assert_eq!(fut.await.unwrap(), json!({"ok": true}));
    }

    // ─── Test 2: chat-chunk routing (keyed) ──────────────────────────────────

    #[tokio::test]
    async fn chat_chunks_routed() {
        let (sup, mut test_write, _test_read) = make_supervisor();

        // Register a keyed sink; request_id=42 matches the notification.
        let mut rx = sup.register_chat_sink(42);

        let deltas = ["hello", " world", "!"];
        for d in &deltas {
            write_line(&mut test_write, &chunk_notif(42, d)).await;
        }
        tokio::time::sleep(std::time::Duration::from_millis(30)).await;

        for expected in &deltas {
            let got = rx.try_recv().expect("delta expected");
            assert_eq!(got, *expected);
        }

        sup.remove_chat_sink(42);
    }

    // ─── Test 2-b: two keyed sinks route independently ───────────────────────
    //
    // Registers sinks for request_id 1 and 2; feeds N_CHAT_CHUNK notifications
    // for each; asserts each receiver gets ONLY its own deltas in order.

    #[tokio::test]
    async fn two_keyed_sinks_route_independently() {
        let (sup, mut test_write, _test_read) = make_supervisor();

        let mut rx1 = sup.register_chat_sink(1);
        let mut rx2 = sup.register_chat_sink(2);

        // Feed deltas for request_id 2 first, then request_id 1.
        write_line(&mut test_write, &chunk_notif(2, "alpha")).await;
        write_line(&mut test_write, &chunk_notif(1, "beta")).await;
        write_line(&mut test_write, &chunk_notif(2, "gamma")).await;
        write_line(&mut test_write, &chunk_notif(1, "delta")).await;

        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        // rx1 must only see deltas tagged request_id=1.
        let r1_a = rx1.try_recv().expect("rx1 first chunk");
        let r1_b = rx1.try_recv().expect("rx1 second chunk");
        assert_eq!(r1_a, "beta");
        assert_eq!(r1_b, "delta");
        assert!(rx1.try_recv().is_err(), "rx1 must have no more chunks");

        // rx2 must only see deltas tagged request_id=2.
        let r2_a = rx2.try_recv().expect("rx2 first chunk");
        let r2_b = rx2.try_recv().expect("rx2 second chunk");
        assert_eq!(r2_a, "alpha");
        assert_eq!(r2_b, "gamma");
        assert!(rx2.try_recv().is_err(), "rx2 must have no more chunks");

        sup.remove_chat_sink(1);
        sup.remove_chat_sink(2);
    }

    // ─── Test 3: EOF fails pending + emits WorkerDied ────────────────────────

    #[tokio::test]
    async fn eof_fails_pending_and_emits_died() {
        // Factory: first call succeeds (using duplex), second call fails (no more halves).
        let (sup_write_1, test_read_1) = tokio::io::duplex(64 * 1024);
        let (test_write_1, sup_read_1) = tokio::io::duplex(64 * 1024);

        let sup_write_cell = Arc::new(Mutex::new(Some(sup_write_1)));
        let sup_read_cell = Arc::new(Mutex::new(Some(sup_read_1)));

        let sup = Supervisor::with_factory(Box::new(move |_ring, _model| {
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
                proc: None,
            })
        }));
        sup.start_for("test-model").expect("start");

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
    async fn restart_replays_load() {
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
            Supervisor::with_factory(Box::new(move |_ring, _model| {
                let n = call_count2.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if n == 0 {
                    let write = cell_sup_write_1.lock().take().unwrap();
                    let read = cell_sup_read_1.lock().take().unwrap();
                    Ok(WorkerHalves {
                        write: Box::new(write),
                        read: Box::new(read),
                        proc: None,
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
                        proc: None,
                    })
                }
            }));
        sup.start_for("test-model").expect("start");

        let mut events = sup.events();

        // Record load so replay has something to send. Scan is host-side now —
        // no scan replay; only the load is re-driven after restart.
        sup.record_last_load(json!({"id": "org/model"}));

        // Wait for the reader task to settle, then trigger EOF on first transport.
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        drop(test_write_1);

        // After 1s backoff + restart, the second transport receives the replayed
        // higgs/load (awaited). obs_rx_2 is the duplex peer of sup_write_2 — data
        // written by the supervisor's writer task comes out here. The test reads
        // the load RPC then replies OK so ModelLoaded is emitted.
        let deadline = std::time::Duration::from_millis(3000);

        use tokio::io::AsyncBufReadExt;
        let load_line = tokio::time::timeout(deadline, async {
            let mut lines = BufReader::new(&mut obs_rx_2).lines();
            lines
                .next_line()
                .await
                .unwrap()
                .expect("load line expected")
        })
        .await
        .expect("timeout waiting for replayed load message");

        let load: serde_json::Value = serde_json::from_str(&load_line).expect("valid json");
        assert_eq!(load["method"], M_LOAD, "replayed method must be higgs/load");
        assert_eq!(load["params"], json!({"id": "org/model"}));

        // Reply to the replayed higgs/load so ModelLoaded is emitted.
        let load_id = load["id"].as_u64().expect("load request must carry an id");
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

    /// `logs(n)` returns the tail of the stderr ring, oldest first, and clamps
    /// to the ring length when fewer than `n` lines are present.
    #[tokio::test]
    async fn logs_tail_and_clamp() {
        let (sup, _tw, _tr) = make_supervisor();
        {
            let mut ring = sup.inner.stderr_ring.lock();
            ring.push_back("a".to_owned());
            ring.push_back("b".to_owned());
            ring.push_back("c".to_owned());
        }
        // Tail of 2 → last two, oldest first.
        assert_eq!(sup.logs(2), vec!["b".to_owned(), "c".to_owned()]);
        // n larger than the ring → all lines, no panic.
        assert_eq!(
            sup.logs(100),
            vec!["a".to_owned(), "b".to_owned(), "c".to_owned()]
        );
        // n == 0 → empty.
        assert!(sup.logs(0).is_empty());
    }

    /// `record_last_load` persists replay params, and `alloc_request_id` hands
    /// out strictly increasing ids.
    #[tokio::test]
    async fn record_params_and_alloc_ids() {
        let (sup, _tw, _tr) = make_supervisor();
        assert!(sup.last_load_params().is_none(), "no load recorded yet");

        sup.record_last_load(json!({"id": "org/model"}));
        assert_eq!(sup.last_load_params(), Some(json!({"id": "org/model"})));

        let a = sup.alloc_request_id();
        let b = sup.alloc_request_id();
        assert!(b > a, "ids strictly increase: {a} then {b}");
    }

    /// `register_chat_sink` then `remove_chat_sink` adds and releases the keyed
    /// map entry; the dropped sender closes the receiver (end-of-stream).
    #[tokio::test]
    async fn chat_sink_register_and_remove() {
        let (sup, _tw, _tr) = make_supervisor();
        assert_eq!(sup.chat_sinks_count(), 0);

        let mut rx = sup.register_chat_sink(42);
        assert_eq!(sup.chat_sinks_count(), 1);

        sup.remove_chat_sink(42);
        assert_eq!(sup.chat_sinks_count(), 0);
        // The dropped sender closes the receiver → recv yields None.
        assert!(rx.recv().await.is_none(), "removed sink closes the stream");
    }

    /// A `route_notification` with an unrelated method, a missing `request_id`,
    /// or a missing `delta` is a silent no-op (no sink delivery, no panic).
    #[tokio::test]
    async fn route_notification_ignores_malformed() {
        let (sup, _tw, _tr) = make_supervisor();
        let mut rx = sup.register_chat_sink(7);

        // Wrong method.
        route_notification(
            &sup.inner,
            &rpc::RpcNotification {
                jsonrpc: "2.0".into(),
                method: "higgs/other".into(),
                params: json!({"request_id": 7, "delta": "x"}),
            },
        );
        // Right method but no request_id.
        route_notification(
            &sup.inner,
            &rpc::RpcNotification {
                jsonrpc: "2.0".into(),
                method: N_CHAT_CHUNK.into(),
                params: json!({"delta": "x"}),
            },
        );
        // Right method, request_id present, but no delta.
        route_notification(
            &sup.inner,
            &rpc::RpcNotification {
                jsonrpc: "2.0".into(),
                method: N_CHAT_CHUNK.into(),
                params: json!({"request_id": 7}),
            },
        );

        // None of the above delivered anything to the sink.
        assert!(
            rx.try_recv().is_err(),
            "malformed notifications deliver nothing"
        );

        // A well-formed one delivers.
        route_notification(
            &sup.inner,
            &rpc::RpcNotification {
                jsonrpc: "2.0".into(),
                method: N_CHAT_CHUNK.into(),
                params: json!({"request_id": 7, "delta": "hi"}),
            },
        );
        assert_eq!(rx.try_recv().unwrap(), "hi");
    }
}
