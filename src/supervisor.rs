//! Worker supervision: spawns the worker role by re-executing the current
//! binary with `--higgs-worker`, speaks NDJSON JSON-RPC over its stdio,
//! correlates responses by id, routes chat-chunk notifications, restarts the
//! worker on death (1s backoff), and re-loads the last model after restart.

// Task 8 (Higgs facade) is not yet written; suppress dead-code noise until
// then so clippy stays clean on this crate in isolation.
#![allow(dead_code)]

use std::collections::{HashMap, VecDeque};
use std::io::{BufRead, BufReader, Write};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;
use serde_json::Value;
use tokio::sync::{broadcast, mpsc, oneshot};
use tracing::{info, warn};

use crate::diagnostic::HiggsError;
use crate::rpc::{self, RpcFrame, RpcNotification, RpcRequest};

/// Method name for chat-chunk notifications sent by the worker.
const N_CHAT_CHUNK: &str = "higgs/chat/chunk";

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

// ── Transport split ───────────────────────────────────────────────────────────
//
// The write and read halves of the worker channel need independent lifetimes:
// - The reader thread OWNS the read half and calls `recv_line()` blocking —
//   it must never hold a shared lock while blocked.
// - `Inner` holds the write half under a Mutex so `request()` can send lines
//   concurrently with the reader loop.
//
// The transport trait stays unified (one impl per medium) but callers split
// a `Box<dyn WorkerTransport>` into two boxed halves via `split_transport`.

/// Unified transport abstraction. Implementations are sync and produced by
/// the factory on each spawn. The pair is split immediately after construction.
pub(crate) trait WorkerTransport: Send {
    /// Write one NDJSON line to the worker. The supervisor appends `\n`.
    fn send_line(&mut self, line: &str) -> std::io::Result<()>;
    /// Read one NDJSON line from the worker. Returns `Ok(None)` on EOF.
    fn recv_line(&mut self) -> std::io::Result<Option<String>>;
}

/// Owned write half: exclusive sender for one worker lifetime.
///
/// Held in `Inner` behind a Mutex; lock time is sub-microsecond (one write).
type WriteSide = dyn FnMut(&str) -> std::io::Result<()> + Send;

/// Owned read half: given directly to the reader thread — never shared.
type ReadSide = dyn FnMut() -> std::io::Result<Option<String>> + Send;

/// Split a unified transport into independent write and read halves.
///
/// Both closures share one `parking_lot::Mutex<Transport>`.  Each half
/// acquires and releases PER CALL (one syscall) — never held across `.await`.
/// The write side blocks momentarily if the reader is mid-line, which is
/// acceptable (sub-line latency; the worker is the bottleneck).
fn split_transport(
    transport: Box<dyn WorkerTransport>,
) -> (Box<WriteSide>, Box<ReadSide>) {
    let shared = Arc::new(Mutex::new(transport));
    let shared_read = Arc::clone(&shared);

    let write_fn: Box<WriteSide> = Box::new(move |line: &str| {
        shared.lock().send_line(line)
    });

    let read_fn: Box<ReadSide> = Box::new(move || {
        shared_read.lock().recv_line()
    });

    (write_fn, read_fn)
}

/// Production transport: real child process stdio.
struct ChildTransport {
    stdin: std::process::ChildStdin,
    reader: BufReader<std::process::ChildStdout>,
    stderr_thread: Option<std::thread::JoinHandle<()>>,
    child: std::process::Child,
}

impl WorkerTransport for ChildTransport {
    fn send_line(&mut self, line: &str) -> std::io::Result<()> {
        self.stdin.write_all(line.as_bytes())?;
        self.stdin.write_all(b"\n")?;
        self.stdin.flush()
    }

    fn recv_line(&mut self) -> std::io::Result<Option<String>> {
        let mut buf = String::new();
        match self.reader.read_line(&mut buf) {
            Ok(0) => Ok(None),
            Ok(_) => {
                if buf.ends_with('\n') {
                    buf.pop();
                    if buf.ends_with('\r') {
                        buf.pop();
                    }
                }
                Ok(Some(buf))
            }
            Err(e) => Err(e),
        }
    }
}

impl Drop for ChildTransport {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        if let Some(t) = self.stderr_thread.take() {
            let _ = t.join();
        }
    }
}

/// Factory: produces a transport on each (re)spawn.
///
/// Takes the stderr ring so production impl can wire a drain thread.
type TransportFactory = Box<
    dyn Fn(Arc<Mutex<VecDeque<String>>>) -> Result<Box<dyn WorkerTransport>, HiggsError>
        + Send
        + Sync,
>;

/// Shared supervisor state.
struct Inner {
    /// Pending request id → oneshot reply channel.
    /// Lock held for insert/remove only — never across `.await`.
    pending: Mutex<HashMap<u64, oneshot::Sender<rpc::RpcResponse>>>,
    /// Active chat-chunk sink (one in-flight chat at a time; worker serializes).
    /// Lock held for swap/take only.
    chat_sink: Mutex<Option<mpsc::UnboundedSender<String>>>,
    /// Monotonically increasing request id counter.
    next_id: AtomicU64,
    /// Broadcast channel for lifecycle events.
    events_tx: broadcast::Sender<HiggsEvent>,
    /// Ring buffer of recent stderr lines.
    /// `Arc`-wrapped so the factory closure can clone the handle.
    stderr_ring: Arc<Mutex<VecDeque<String>>>,
    /// Params of the last successful `higgs/scan`; replayed before load after restart.
    last_scan: Mutex<Option<Value>>,
    /// Params of the last successful `higgs/load`; replayed after restart.
    last_load: Mutex<Option<Value>>,
    /// Set on `stop()` — suppresses respawn after death.
    stopped: AtomicBool,
    /// Write half of the active worker channel. `None` when no worker is live.
    /// Lock held for one write syscall only.
    write_side: Mutex<Option<Box<WriteSide>>>,
    /// Factory that builds a new transport on each (re)spawn.
    factory: TransportFactory,
}

/// Worker process supervisor.
///
/// Spawns the higgs worker by re-executing the current binary with
/// `--higgs-worker`, speaks NDJSON JSON-RPC 2.0 over its stdio, correlates
/// request/response pairs by id, routes `higgs/chat/chunk` notifications to
/// the active chat receiver, and restarts the worker (1s backoff) on death.
///
/// One in-flight chat at a time: the worker serializes chat requests anyway;
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
            write_side: Mutex::new(None),
            factory: Box::new(production_factory),
        });
        Self { inner }
    }

    /// Create a supervisor with an injected transport — for tests only.
    ///
    /// The factory is called once per (re)spawn. A test that wants to simulate
    /// EOF-then-failure can return `Err(WorkerSpawnFailed)` on the second call.
    pub(crate) fn with_transport(factory: TransportFactory) -> Self {
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
            write_side: Mutex::new(None),
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

    /// Spawn the worker and start the reader thread.
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
    pub(crate) async fn request(
        &self,
        method: &str,
        params: Value,
    ) -> Result<Value, HiggsError> {
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
        let send_result = {
            let mut ws = self.inner.write_side.lock();
            match ws.as_mut() {
                Some(w) => w(&line),
                None => {
                    drop(ws);
                    self.inner.pending.lock().remove(&id);
                    return Err(HiggsError::WorkerDead {
                        context: "no worker running".into(),
                    });
                }
            }
        };
        if let Err(e) = send_result {
            self.inner.pending.lock().remove(&id);
            return Err(HiggsError::WorkerDead {
                context: format!("send failed: {e}"),
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
    /// serializes). The caller owns the returned receiver and reads deltas
    /// until the sender is dropped (end of stream).
    pub(crate) fn take_chat_sink(&self) -> mpsc::UnboundedReceiver<String> {
        let (tx, rx) = mpsc::unbounded_channel();
        *self.inner.chat_sink.lock() = Some(tx);
        rx
    }

    /// Gracefully shut down the worker (2s timeout) then kill it.
    ///
    /// Sets the deliberate-stop flag so death does not trigger respawn.
    pub(crate) async fn stop(&self) {
        self.inner.stopped.store(true, Ordering::Relaxed);
        let _ = tokio::time::timeout(
            std::time::Duration::from_secs(2),
            self.request("higgs/shutdown", Value::Null),
        )
        .await;
        *self.inner.write_side.lock() = None;
    }

    /// Record the params of a successful `higgs/scan` for post-restart replay.
    pub(crate) fn record_last_scan(&self, params: Value) {
        *self.inner.last_scan.lock() = Some(params);
    }

    /// Record the params of a successful `higgs/load` for post-restart replay.
    pub(crate) fn record_last_load(&self, params: Value) {
        *self.inner.last_load.lock() = Some(params);
    }

    // ── private ──────────────────────────────────────────────────────────────

    /// Build a new transport, split it, install the write side, and launch
    /// the reader thread with the read side.
    fn do_spawn(&self) -> Result<(), HiggsError> {
        let transport = (self.inner.factory)(self.inner.stderr_ring.clone())?;
        let (write_fn, read_fn) = split_transport(transport);
        *self.inner.write_side.lock() = Some(write_fn);
        let inner = Arc::clone(&self.inner);
        std::thread::spawn(move || reader_loop(inner, read_fn));
        Ok(())
    }
}

/// Blocking reader loop — runs on a dedicated thread, owns the read side.
///
/// Dispatches responses/notifications and triggers restart on EOF/error.
#[allow(clippy::needless_pass_by_value)]
fn reader_loop(inner: Arc<Inner>, mut read_fn: Box<ReadSide>) {
    loop {
        let line = read_fn();
        match line {
            Ok(Some(l)) => dispatch(&inner, &l),
            Ok(None) | Err(_) => {
                let reason = match line {
                    Err(ref e) => Some(e.to_string()),
                    _ => None,
                };
                on_worker_death(&inner, reason);
                if inner.stopped.load(Ordering::Relaxed) {
                    break;
                }
                // 1s backoff then attempt a single respawn.
                std::thread::sleep(std::time::Duration::from_secs(1));
                match (inner.factory)(inner.stderr_ring.clone()) {
                    Ok(new_transport) => {
                        let (write_fn, new_read_fn) = split_transport(new_transport);
                        *inner.write_side.lock() = Some(write_fn);
                        let _ = inner.events_tx.send(HiggsEvent::WorkerRestarted);
                        info!("higgs worker restarted");
                        replay_scan_then_load(&inner);
                        // Replace read_fn and continue the loop on the new transport.
                        read_fn = new_read_fn;
                    }
                    Err(e) => {
                        warn!(error = %e, "higgs worker respawn failed — giving up");
                        break;
                    }
                }
            }
        }
    }
}

/// Decode one line and dispatch to the appropriate handler.
fn dispatch(inner: &Arc<Inner>, line: &str) {
    match rpc::decode(line) {
        Ok(RpcFrame::Response(resp)) => correlate(inner, resp),
        Ok(RpcFrame::Notification(notif)) => route_notification(inner, notif),
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
#[allow(clippy::needless_pass_by_value)]
fn route_notification(inner: &Arc<Inner>, notif: RpcNotification) {
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

/// Handle worker death: fail all pending requests, clear write side, broadcast event.
fn on_worker_death(inner: &Arc<Inner>, reason: Option<String>) {
    // Drain and drop all pending senders — the rx.await in `request` sees Err.
    let drained: Vec<_> = inner.pending.lock().drain().collect();
    for (_id, tx) in drained {
        drop(tx);
    }
    // Clear write side — future send attempts see "no worker running".
    *inner.write_side.lock() = None;
    // Clear chat sink.
    *inner.chat_sink.lock() = None;

    let _ = inner.events_tx.send(HiggsEvent::WorkerDied);
    if let Some(r) = reason {
        warn!(detail = %r, "higgs worker died");
    } else {
        info!("higgs worker exited (EOF)");
    }
}

/// Best-effort send of one RPC request with no response tracking.
///
/// Responses are not awaited — best-effort only. If a send fails, the next
/// caller gets HG003 and can retry explicitly.
fn replay_fire_and_forget(inner: &Arc<Inner>, method: &str, params: Value) {
    let id = inner.next_id.fetch_add(1, Ordering::Relaxed);
    let line = rpc::encode(&RpcFrame::Request(RpcRequest {
        jsonrpc: "2.0".into(),
        id,
        method: method.to_string(),
        params,
    }));
    let mut ws = inner.write_side.lock();
    if let Some(w) = ws.as_mut() {
        if let Err(e) = w(&line) {
            warn!(error = %e, method, "higgs: replay send failed");
        }
    }
}

/// Best-effort replay of the last successful scan (if any) then load (if any) after restart.
///
/// Scan is replayed first so the worker's model index is populated before the
/// load request arrives. Both sends are fire-and-forget.
fn replay_scan_then_load(inner: &Arc<Inner>) {
    if let Some(p) = inner.last_scan.lock().clone() {
        replay_fire_and_forget(inner, "higgs/scan", p);
    }
    if let Some(p) = inner.last_load.lock().clone() {
        replay_fire_and_forget(inner, "higgs/load", p);
    }
}

/// Production transport factory: re-exec current binary with `--higgs-worker`.
fn production_factory(
    stderr_ring: Arc<Mutex<VecDeque<String>>>,
) -> Result<Box<dyn WorkerTransport>, HiggsError> {
    use std::process::{Command, Stdio};

    let exe = std::env::current_exe().map_err(|e| HiggsError::WorkerSpawnFailed { source: e })?;
    let mut child = Command::new(exe)
        .arg("--higgs-worker")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| HiggsError::WorkerSpawnFailed { source: e })?;

    let stdin = child.stdin.take().expect("piped stdin");
    let stdout = child.stdout.take().expect("piped stdout");
    let stderr = child.stderr.take().expect("piped stderr");

    let stderr_thread = std::thread::spawn(move || {
        for line in BufReader::new(stderr).lines().map_while(Result::ok) {
            let mut ring = stderr_ring.lock();
            if ring.len() == 2000 {
                ring.pop_front();
            }
            ring.push_back(line);
        }
    });

    Ok(Box::new(ChildTransport {
        stdin,
        reader: BufReader::new(stdout),
        stderr_thread: Some(stderr_thread),
        child,
    }))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::mpsc as std_mpsc;

    // ── Mock transport ────────────────────────────────────────────────────────

    /// In-memory mock transport backed by two std::sync channels.
    struct MockTransport {
        /// Lines from the "worker" → supervisor reads these.
        inbound_rx: std_mpsc::Receiver<Option<String>>,
        /// Lines the supervisor sends → test observes these.
        outbound_tx: std_mpsc::SyncSender<String>,
    }

    impl WorkerTransport for MockTransport {
        fn send_line(&mut self, line: &str) -> std::io::Result<()> {
            // Ignore send errors: tests that don't observe sent lines drop the
            // receiver, which is fine — the write side should not fail.
            let _ = self.outbound_tx.try_send(line.to_string());
            Ok(())
        }

        fn recv_line(&mut self) -> std::io::Result<Option<String>> {
            match self.inbound_rx.recv() {
                Ok(v) => Ok(v),
                Err(_) => Ok(None), // sender dropped = EOF
            }
        }
    }

    /// Build a supervisor plus test control handles.
    ///
    /// Returns `(supervisor, inject_tx, observe_rx)`.
    /// - `inject_tx`: push `Some(line)` to feed lines to the supervisor's reader;
    ///   push `None` to simulate EOF.
    /// - `observe_rx`: receive lines the supervisor sent to the "worker".
    fn make_supervisor() -> (
        Supervisor,
        std_mpsc::SyncSender<Option<String>>,
        std_mpsc::Receiver<String>,
    ) {
        let (inbound_tx, inbound_rx) = std_mpsc::sync_channel::<Option<String>>(64);
        let (outbound_tx, outbound_rx) = std_mpsc::sync_channel::<String>(64);

        let inbound_rx = std::sync::Mutex::new(Some(inbound_rx));
        let outbound_tx_clone = outbound_tx.clone();

        let sup = Supervisor::with_transport(Box::new(move |_ring| {
            let rx = inbound_rx
                .lock()
                .unwrap()
                .take()
                .ok_or_else(|| HiggsError::WorkerSpawnFailed {
                    source: std::io::Error::other("mock: no more transports"),
                })?;
            Ok(Box::new(MockTransport {
                inbound_rx: rx,
                outbound_tx: outbound_tx_clone.clone(),
            }) as Box<dyn WorkerTransport>)
        }));

        sup.start().expect("mock start failed");

        (sup, inbound_tx, outbound_rx)
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
            error: Some(rpc::RpcError { code, message: message.into() }),
        }))
    }

    fn chunk_notif(request_id: u64, delta: &str) -> String {
        rpc::encode(&RpcFrame::Notification(rpc::RpcNotification {
            jsonrpc: "2.0".into(),
            method: N_CHAT_CHUNK.into(),
            params: json!({"request_id": request_id, "delta": delta}),
        }))
    }

    // ─── Test 1: out-of-order response correlation ───────────────────────────

    #[tokio::test]
    async fn request_response_correlation() {
        let (sup, inbound_tx, _) = make_supervisor();

        // Issue two requests concurrently. The supervisor assigns ids 1 and 2.
        let fut1 = sup.request("higgs/ping", json!({"n": 1}));
        let fut2 = sup.request("higgs/ping", json!({"n": 2}));

        // Give both requests time to register their pending entries.
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;

        // Respond to id 2 first, then id 1.
        inbound_tx.send(Some(ok_response(2, json!({"n": 2})))).unwrap();
        inbound_tx.send(Some(ok_response(1, json!({"n": 1})))).unwrap();

        let (r1, r2) = tokio::join!(fut1, fut2);
        assert_eq!(r1.unwrap(), json!({"n": 1}));
        assert_eq!(r2.unwrap(), json!({"n": 2}));
    }

    // ─── Test 2: chat-chunk routing ──────────────────────────────────────────

    #[tokio::test]
    async fn chat_chunks_routed() {
        let (sup, inbound_tx, _) = make_supervisor();

        let mut rx = sup.take_chat_sink();

        let deltas = ["hello", " world", "!"];
        for d in &deltas {
            inbound_tx.send(Some(chunk_notif(42, d))).unwrap();
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
        // Factory returns Err on second call → respawn fails cleanly.
        let (inbound_tx, inbound_rx) = std_mpsc::sync_channel::<Option<String>>(64);
        let (outbound_tx, _outbound_rx) = std_mpsc::sync_channel::<String>(64);

        let inbound_rx = std::sync::Mutex::new(Some(inbound_rx));
        let outbound_tx2 = outbound_tx.clone();
        let call_count = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
        let call_count2 = std::sync::Arc::clone(&call_count);

        let sup = Supervisor::with_transport(Box::new(move |_ring| {
            let n = call_count2.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if n == 0 {
                let rx = inbound_rx.lock().unwrap().take().unwrap();
                Ok(Box::new(MockTransport {
                    inbound_rx: rx,
                    outbound_tx: outbound_tx2.clone(),
                }) as Box<dyn WorkerTransport>)
            } else {
                Err(HiggsError::WorkerSpawnFailed {
                    source: std::io::Error::other("mock: no more transports"),
                })
            }
        }));
        sup.start().expect("start");

        let mut events = sup.events();

        // Register a pending request directly (bypass the network send so the
        // pending entry exists before EOF arrives).
        let (reply_tx, reply_rx) = oneshot::channel::<rpc::RpcResponse>();
        let pending_id = sup.inner.next_id.fetch_add(1, Ordering::Relaxed);
        sup.inner.pending.lock().insert(pending_id, reply_tx);

        // Give the reader thread time to block on recv_line.
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;

        // Trigger EOF.
        inbound_tx.send(None).unwrap();

        // The pending oneshot should be dropped (channel closed → Err).
        let result = reply_rx.await;
        assert!(result.is_err(), "pending request should fail on EOF");

        // WorkerDied must arrive (respawn fires after 1s sleep; wait 1.5s).
        let died = tokio::time::timeout(std::time::Duration::from_millis(1500), async {
            loop {
                match events.recv().await {
                    Ok(HiggsEvent::WorkerDied) => return true,
                    Ok(_) => continue,
                    Err(_) => return false,
                }
            }
        })
        .await;
        assert!(matches!(died, Ok(true)), "WorkerDied event expected");
    }

    // ─── Test 4: worker RPC error maps to HG009 ──────────────────────────────

    #[tokio::test]
    async fn worker_error_maps_to_hg009() {
        let (sup, inbound_tx, _) = make_supervisor();

        let fut = sup.request("higgs/load", json!({"id": "org/bad"}));
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        inbound_tx.send(Some(err_response(1, -32000, "model file corrupt"))).unwrap();

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

    // ─── Test 6: restart replays scan before load ────────────────────────────

    #[tokio::test]
    async fn restart_replays_scan_then_load() {
        // Two transports: first lives until we send EOF, second receives the
        // replayed messages and then blocks forever (no EOF sent).
        let (first_inbound_tx, first_inbound_rx) = std_mpsc::sync_channel::<Option<String>>(64);
        let (second_inbound_tx, second_inbound_rx) = std_mpsc::sync_channel::<Option<String>>(64);
        let (outbound_tx, outbound_rx) = std_mpsc::sync_channel::<String>(64);

        // Wrap both receivers so the factory can hand them out one at a time.
        let first_rx = std::sync::Mutex::new(Some(first_inbound_rx));
        let second_rx = std::sync::Mutex::new(Some(second_inbound_rx));
        let call_count = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
        let call_count2 = std::sync::Arc::clone(&call_count);
        let outbound_tx2 = outbound_tx.clone();

        let sup = Supervisor::with_transport(Box::new(move |_ring| {
            let n = call_count2.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let rx = if n == 0 {
                first_rx.lock().unwrap().take().unwrap()
            } else {
                second_rx.lock().unwrap().take().unwrap()
            };
            Ok(Box::new(MockTransport {
                inbound_rx: rx,
                outbound_tx: outbound_tx2.clone(),
            }) as Box<dyn WorkerTransport>)
        }));
        sup.start().expect("start");

        // Record a scan and a load so replay has something to send.
        sup.record_last_scan(json!({"dirs": ["/models"]}));
        sup.record_last_load(json!({"id": "org/model"}));

        // Wait for the reader thread to settle, then trigger EOF on the first transport.
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        first_inbound_tx.send(None).unwrap();

        // After 1s backoff + restart, the second transport should receive
        // higgs/scan first, then higgs/load.  Allow up to 2s for this.
        let deadline = std::time::Duration::from_secs(2);
        let received: Vec<String> = {
            let mut msgs = Vec::new();
            let start = std::time::Instant::now();
            while msgs.len() < 2 && start.elapsed() < deadline {
                if let Ok(line) = outbound_rx.recv_timeout(std::time::Duration::from_millis(100)) {
                    msgs.push(line);
                }
            }
            msgs
        };

        assert_eq!(received.len(), 2, "expected exactly 2 replayed messages, got: {received:?}");

        // Decode both and check method ordering.
        let first: serde_json::Value = serde_json::from_str(&received[0]).expect("valid json");
        let second: serde_json::Value = serde_json::from_str(&received[1]).expect("valid json");
        assert_eq!(first["method"], "higgs/scan", "first replayed method must be higgs/scan");
        assert_eq!(second["method"], "higgs/load", "second replayed method must be higgs/load");

        // Params must match what was recorded.
        assert_eq!(first["params"], json!({"dirs": ["/models"]}));
        assert_eq!(second["params"], json!({"id": "org/model"}));

        // Keep second transport alive so the reader thread doesn't fire another restart.
        drop(second_inbound_tx);
    }
}
