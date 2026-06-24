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

use std::process::Stdio;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::Command;
use tokio::sync::{broadcast, mpsc};
use tracing::{info, warn};

use crate::diagnostic::HiggsError;
use crate::log_bus::{LogBus, LogLine, LogSource};
use crate::rpc::{self, RpcFrame, RpcNotification, RpcRequest};
use crate::system::GpuDevice;
use crate::worker::{M_LOAD, M_LOG_LEVEL, M_SHUTDOWN, M_SYSINFO, N_CHAT_CHUNK};

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

/// Bound on a single chat/inference RPC (M_CHAT) round-trip. Distinct from
/// [`CONTROL_RPC_TIMEOUT`] and far more generous: generation is long-running and
/// streams progress chunks, so a wedged chat must be tolerated far longer than a
/// scan/load/status control round-trip. The bound still exists so a worker that
/// is ALIVE but wedged mid-generation (an FFI call hung inside llama.cpp, no
/// further chunks, no final response) can never hang the caller — and thus the
/// HTTP connection — forever: without it, `rx.await` blocks indefinitely since
/// no EOF arrives to drain `pending`. This is the layer that bounds streaming
/// chat duration; the HTTP layer deliberately does NOT time
/// `/v1/chat/completions` (a long SSE stream must outlive any per-request HTTP
/// timeout). On expiry the caller gets [`HiggsError::ChatTimeout`] → HTTP 504.
/// 10 minutes covers a very long large-model generation while still bounding a
/// hang; no reference fixes a chat-RPC ceiling (vllm/ollama bound per-request
/// output via `max_tokens`, which higgs also caps), so this is the documented
/// higgs value, grouped with the other serve-layer limits for a later config
/// lift.
pub(crate) const CHAT_RPC_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(600);

/// Bound on the single M_SYSINFO round-trip in [`Supervisor::sysinfo`]. Device
/// enumeration is a cheap FFI registry read with no model load, so it completes
/// near-instantly; the bound exists only so a wedged transient worker can never
/// hang the `GET /api/higgs/system` handler. On expiry the device list is empty.
const SYSINFO_RPC_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(15);

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
pub(crate) type HalvesFactory =
    Box<dyn Fn(Arc<LogBus>, &str) -> Result<WorkerHalves, HiggsError> + Send + Sync>;

/// Shared supervisor state.
struct Inner {
    /// RPC reply-demux: request-id → response correlation, plus per-request
    /// chat-chunk sinks keyed by the M_CHAT `request_id`. The shared
    /// [`crate::actor::ReplyDemux`] (the same demux the per-node iroh transport
    /// reuses in P3). Each concurrent caller registers its own sink before sending
    /// M_CHAT, so the worker's echoed-`request_id` `N_CHAT_CHUNK` notifications
    /// route to ONLY that caller — worker execution stays serialised, the demux
    /// just prevents cross-caller clobber.
    demux: crate::actor::ReplyDemux,
    /// Monotonically increasing request id counter.
    next_id: AtomicU64,
    /// Worker-lifetime generation, bumped by every `do_spawn`. A `reader_task` captures
    /// its generation at spawn; if a later `do_spawn` (a `start_for` after this worker's
    /// `stop()`) bumps it, the now-stale reader exits on its next death WITHOUT touching
    /// the live worker's state — closes the stop()→start_for() stale-reader clobber race.
    generation: AtomicU64,
    /// Broadcast channel for lifecycle events (cap 64).
    events_tx: broadcast::Sender<HiggsEvent>,
    /// Single home for Developer-Log lines: bounded history ring (the `logs(n)`
    /// snapshot source) plus a live broadcast tap (the SSE-stream source). Both
    /// the worker stderr reader and the serve-layer tracing [`HiggsLogLayer`]
    /// push lines through this one bus. `Arc`-wrapped so the factory closure
    /// can clone the handle and so the caller's tracing layer shares it.
    bus: Arc<LogBus>,
    /// Params of the last successful `higgs/load`; replayed after restart.
    last_load: Mutex<Option<Value>>,
    /// Bumped on every `record_last_load`/`clear_last_load`. A crash-restart's async
    /// `replay_load` captures this; if it changed before the replay fires, an explicit
    /// load/unload superseded the captured model, so the replay is skipped rather than
    /// resurrecting a stale model over a newer one.
    load_epoch: AtomicU64,
    /// Set on `stop()` — suppresses respawn after death.
    stopped: AtomicBool,
    /// True for the entire lifetime of a live worker — from the spawn that
    /// started it through any in-loop restart, until the reader gives up or
    /// `stop()` reaps it. `do_spawn` refuses to start a second worker while
    /// this is set, which guarantees a single reader task: without it, a
    /// redundant `start_for()` (a `load()` while a worker is already live) on a
    /// running worker would spawn a second reader whose later EOF clears the live
    /// worker's `write_tx`. Set across restart so the death→respawn window is
    /// also covered.
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
///
/// ## Actor-model assessment (P2, per CLAUDE.md)
///
/// The Supervisor was assessed against the actor requirement and **already realizes
/// the actor model**; a literal `spawn_actor` mailbox rewrite is deliberately NOT
/// warranted:
/// - The **`reader_task`** is a single owning task that loops on the read half and
///   serially dispatches inbound frames, handles worker death, and drives the entire
///   restart FSM — i.e. it *is* the supervisor's inbound actor loop. The
///   single-reader invariant (`Inner::running`) guarantees exactly one such task.
/// - The **`writer_task`** drains an mpsc of outbound lines and owns stdin — a
///   textbook outbound mailbox; no mutex sits on the I/O path.
/// - The **worker process** is the executor and serialises all generation.
/// - **`ReplyDemux`** is the blessed RPC-client correlation map (see `actor.rs`): its
///   locks guard *single-op* map insert/remove only — there is no await-spanning
///   multi-step mutation of those maps, so they cannot exhibit the check-then-act /
///   TOCTOU race class that makes `Mutex` state unsafe and mandates an actor. Routing
///   responses/`N_CHAT_CHUNK` chunks directly through the demux (rather than a control
///   mailbox) is the intended streaming bypass — token deltas must NOT serialize
///   behind control messages (the P5 sink-handoff intent). The one ordering hazard —
///   a `send_request` racing `on_worker_death` — is closed by clearing `write_tx`
///   BEFORE `fail_all_pending` there, so a request issued during worker death fast-fails
///   instead of waiting out its RPC timeout (see `on_worker_death`).
/// - The only cross-task multi-step mutation — `stop()` versus the reader's
///   `attempt_restart` — is serialized by the documented `stopped` (Release/Acquire) +
///   `running` ordering (the F1/F2 hardening) plus the `generation` tag, not a lock-based
///   check-then-act, so a stop can never race a respawn into resurrecting an unloaded
///   worker, and a reader left over from a stopped lifetime can never clobber the next
///   worker's state (it sees a bumped `generation` and exits).
/// - `start_for`/`do_spawn` versus `stop()` is NOT internally guarded (do_spawn resets
///   `stopped=false`); instead it is a **caller-enforced** precondition — the local
///   `Higgs` serializes both under its `lifecycle` mutex, and the node runtime awaits
///   `start_for` before committing the Supervisor, so the two never run concurrently on
///   one Supervisor (see the `do_spawn` body comment). A new caller MUST preserve this.
///
/// **Accepted residual (safe-degrading, not a corruption race):** after a *crash*-driven
/// restart, `replay_load` re-sends `M_LOAD` from a spawned task, so there is a sub-ms
/// window where the respawned worker is visible (`write_tx` installed) but the model is
/// not yet resident. A chat that races into that window is rejected `HG003 ModelNotLoaded`
/// (retryable) rather than served — a transient model-availability window inherent to
/// crash recovery, with no state corruption/hang/orphan. Closing it would mean
/// pre-enqueuing the replay into the writer channel before publishing it, restructuring the
/// hardened replay FSM for marginal gain; left as a documented residual.
///
/// A stronger hazard — a stale crash-replay overwriting a NEWER explicit load (resident
/// model ≠ recorded model) — is closed by the `load_epoch`: `replay_load` captures it and
/// skips if an explicit load/unload bumped it first. The remaining residual is the tight
/// sub-window where the newer load's `record_last_load` has not landed when the replay
/// checks; there the two in-flight `M_LOAD`s race on writer FIFO order. Fully closing that
/// needs a single serialized load path (a follow-on if it ever matters in practice).
///
/// **Teardown-timeout residual class (safe-degrading):** because `stop()` touches
/// `write_tx` (parking_lot lock) and `proc` (tokio lock) separately rather than under one
/// guard, a request that races a `stop()` interleaved with a crash-restart can briefly see
/// a `write_tx` published for an already-reaped child, and resolve via its BOUNDED RPC
/// timeout instead of fast-failing. No corruption or orphan results — the child is always
/// reaped, the `generation` guard stops any stale reader from clobbering the next worker,
/// and the request returns a proper error (just later). Eliminating these sub-windows
/// entirely requires a single serialized lifecycle owner (the full mailbox-actor
/// conversion), which the assessment above judges not worth its cost/risk here.
///
/// Converting to a literal mailbox would fold the read-half-owning `reader_task` and
/// the streaming chunk path behind a control mailbox — fighting the bypass design,
/// rippling sync→async across ~20 `Higgs` (local-path) call sites, and risking the
/// most-hardened lifecycle FSM — for no race-elimination gain. The genuine lock-based
/// race class lives in `HubFleet` and is dissolved by folding it into the Engine actor
/// (P3), which is where that work belongs.
pub(crate) struct Supervisor {
    inner: Arc<Inner>,
}

impl Supervisor {
    /// Create a supervisor using the production child-process factory.
    ///
    /// Does not spawn the worker yet; the first `load()` calls `start_for()`.
    ///
    /// `bus` is the shared [`LogBus`] the caller also feeds with the serve-layer
    /// [`HiggsLogLayer`], so worker stderr and request-event lines land in one
    /// place.
    pub(crate) fn spawn(bus: Arc<LogBus>) -> Self {
        let (events_tx, _) = broadcast::channel(64);
        let inner = Arc::new(Inner {
            demux: crate::actor::ReplyDemux::new(),
            next_id: AtomicU64::new(1),
            generation: AtomicU64::new(0),
            load_epoch: AtomicU64::new(0),
            events_tx,
            bus,
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
            demux: crate::actor::ReplyDemux::new(),
            next_id: AtomicU64::new(1),
            generation: AtomicU64::new(0),
            load_epoch: AtomicU64::new(0),
            events_tx,
            bus: Arc::new(LogBus::new()),
            last_load: Mutex::new(None),
            stopped: AtomicBool::new(false),
            running: AtomicBool::new(false),
            write_tx: Mutex::new(None),
            proc: tokio::sync::Mutex::new(None),
            factory,
        });
        Self { inner }
    }

    // ── Per-supervisor log/event/verbose accessors ─────────────────────────────
    //
    // After the P4b engine swap the `Higgs` facade reads the Developer-Log bus and
    // lifecycle events from the LOCAL `NodeRuntime` (which owns per-worker buses +
    // its own event fan-out), not from a single Supervisor. These delegators are
    // kept — each has its own unit test below and they remain the natural per-worker
    // API the node will surface for per-worker log routing — so `#[allow(dead_code)]`
    // silences the unused-in-non-test-build warning without dropping the coverage.

    /// Subscribe to worker lifecycle events.
    #[allow(dead_code)]
    pub fn events(&self) -> broadcast::Receiver<HiggsEvent> {
        self.inner.events_tx.subscribe()
    }

    /// Return up to `n` recent Developer-Log lines (oldest first), optionally
    /// restricted to one [`LogSource`] (`None` = worker stderr + serve events).
    #[allow(dead_code)]
    pub fn logs(&self, n: usize, filter: Option<LogSource>) -> Vec<String> {
        self.inner.bus.snapshot(n, filter)
    }

    /// Subscribe to live Developer-Log lines pushed after this call. Pair with
    /// [`logs`](Self::logs) for replay-then-live SSE delivery; filter by
    /// [`LogLine::source`].
    #[allow(dead_code)]
    pub fn subscribe_logs(&self) -> broadcast::Receiver<LogLine> {
        self.inner.bus.subscribe()
    }

    /// The "Verbose Logging" toggle (single home on the [`LogBus`]). Read by the
    /// serve layer and the worker stderr drain.
    #[allow(dead_code)]
    pub fn log_verbose(&self) -> bool {
        self.inner.bus.verbose()
    }

    /// Set the "Verbose Logging" toggle.
    #[allow(dead_code)]
    pub fn set_log_verbose(&self, v: bool) {
        self.inner.bus.set_verbose(v);
    }

    /// Push the verbose level to the running worker so its engine-log filter
    /// flips live (INFO+ ↔ DEBUG+). Fire-and-forget: writes the M_LOG_LEVEL frame
    /// and does NOT register a pending entry or await a reply — a non-critical
    /// control must never block the settings handler on the worker. The worker's
    /// reply (if any) arrives id-less-of-pending and is harmlessly dropped. No
    /// worker → `write_tx` is `None` → nothing sent (the next spawn seeds the
    /// level from `HIGGS_WORKER_VERBOSE`).
    #[allow(dead_code)]
    pub fn set_worker_verbose(&self, v: bool) {
        let line = rpc::encode(&RpcFrame::Request(RpcRequest {
            jsonrpc: "2.0".into(),
            id: self.alloc_request_id(),
            method: M_LOG_LEVEL.to_string(),
            params: serde_json::json!({ "verbose": v }),
        }));
        if let Some(tx) = self.inner.write_tx.lock().as_ref() {
            let _ = tx.send(line);
        }
    }

    /// Whether Developer Logs are in un-redacted DEBUG mode (show structured
    /// fields incl. prompt content). Off by default.
    #[allow(dead_code)]
    pub fn log_show_fields(&self) -> bool {
        self.inner.bus.show_fields()
    }

    /// Toggle the un-redacted DEBUG log mode.
    #[allow(dead_code)]
    pub fn set_log_show_fields(&self, v: bool) {
        self.inner.bus.set_show_fields(v);
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
                self.inner.demux.remove_pending(id);
                warn!(method, "higgs: control RPC timed out");
                Err(HiggsError::WorkerDead {
                    context: format!("{method} timed out after {CONTROL_RPC_TIMEOUT:?}"),
                })
            }
        }
    }

    /// Send a request using a pre-allocated id, bounded by [`CHAT_RPC_TIMEOUT`].
    ///
    /// Used for M_CHAT so the caller can register the chat sink under the
    /// SAME id before sending the request — ensuring chunk routing matches.
    /// On timeout the orphaned `pending[id]` entry is removed (so it doesn't
    /// leak — no EOF arrives to drain it for an alive-but-wedged worker) and
    /// [`HiggsError::ChatTimeout`] is returned, mapping to HTTP 504.
    pub(crate) async fn request_with_id(
        &self,
        id: u64,
        method: &str,
        params: Value,
    ) -> Result<Value, HiggsError> {
        match tokio::time::timeout(CHAT_RPC_TIMEOUT, self.send_request(id, method, params)).await {
            Ok(result) => result,
            Err(_) => {
                self.inner.demux.remove_pending(id);
                warn!(method, "higgs: chat RPC timed out");
                Err(HiggsError::ChatTimeout {
                    elapsed: CHAT_RPC_TIMEOUT,
                })
            }
        }
    }

    /// Register a per-request chat-chunk receiver keyed by `request_id`.
    ///
    /// Creates a `(tx, rx)` pair, inserts `tx` into `chat_sinks[request_id]`,
    /// and returns `rx`.  Infallible: concurrent callers each get their own
    /// channel and their own deltas routed independently.  The caller must
    /// call [`remove_chat_sink`](Self::remove_chat_sink) when done.
    pub(crate) fn register_chat_sink(&self, request_id: u64) -> mpsc::UnboundedReceiver<String> {
        self.inner.demux.register_sink(request_id)
    }

    /// Remove the chat-chunk sink for `request_id`.
    ///
    /// Called on completion or error to release the map entry.  The dropped
    /// sender closes the receiver, signalling end-of-stream to the consumer.
    pub(crate) fn remove_chat_sink(&self, request_id: u64) {
        self.inner.demux.remove_sink(request_id);
    }

    /// Drive one chat (M_CHAT) request end to end, owning the full RPC plumbing.
    ///
    /// Returns `(rx, fut)`:
    /// - `rx` is the per-request delta receiver — its keyed chat sink is
    ///   registered SYNCHRONOUSLY here (before the request is sent), so a chunk
    ///   that arrives the instant the worker starts is never dropped. The caller
    ///   takes `rx` immediately for SSE / streaming.
    /// - `fut` resolves with the worker's raw final response `Value` (or a typed
    ///   error). The caller awaits it inside its own spawned task so its
    ///   admission permit rides the whole generation. The sink is removed on ANY
    ///   outcome (success, error, or timeout), keeping register→send→cleanup
    ///   atomic in one place — the facade never touches the sink map, the request
    ///   id, or `request_with_id`.
    ///
    /// `model` is the id the serve layer resolved this chat against — carried in
    /// the M_CHAT params so the worker can refuse (`[HG018]`) if a concurrent JIT
    /// load swapped the resident model out before generation. `messages_json` is
    /// the verbatim OpenAI messages array (carried as a JSON string so
    /// `tool_calls` / `tool_call_id` survive to the engine's chat template);
    /// `tools_json` is the optional serialized `tools` array.
    pub(crate) fn chat(
        &self,
        model: String,
        messages_json: String,
        max_tokens: usize,
        temperature: f32,
        tools_json: Option<String>,
    ) -> (
        mpsc::UnboundedReceiver<String>,
        impl std::future::Future<Output = Result<Value, HiggsError>>,
    ) {
        // One id for both the RPC frame `id` (response correlation) and the
        // `request_id` in the params (N_CHAT_CHUNK routing): the worker echoes
        // params.request_id in every chunk, and `route_notification` looks it up
        // in `chat_sinks`. Register the sink under that id BEFORE the request is
        // sent so no early chunk is lost.
        let request_id = self.alloc_request_id();
        let rx = self.register_chat_sink(request_id);
        // The future owns its own `Arc<Inner>` clone so it is `'static` for the
        // caller's `tokio::spawn`, independent of the `&self` borrow above.
        let inner = Arc::clone(&self.inner);
        let fut = async move {
            let sup = Supervisor { inner };
            let result = sup
                .request_with_id(
                    request_id,
                    crate::worker::M_CHAT,
                    serde_json::json!({
                        "request_id": request_id,
                        // Bind the chat to the model the serve layer resolved
                        // against so the worker rejects (HG018) rather than serve
                        // the wrong model after a concurrent JIT swap.
                        "model": model,
                        "messages_json": messages_json,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "tools": tools_json,
                    }),
                )
                .await;
            // Remove the sink on any outcome: success drops the sender (closing
            // the receiver); failure/timeout closes it too. Atomic with the
            // register above — both live in this one method.
            sup.remove_chat_sink(request_id);
            result
        };
        (rx, fut)
    }

    /// Enumerate the host's compute devices in a SEPARATE, transient worker —
    /// fully isolated from the serving worker (`self.inner`).
    ///
    /// Spawns one fresh worker via the same factory (`production_factory(bus,
    /// "sysinfo")`), runs a single M_SYSINFO round-trip on its raw stdio (no
    /// persistent reader/writer tasks, no `pending`/`chat_sinks` involvement),
    /// then reaps the child. The serving worker is never touched, so this can
    /// never evict a resident model or interfere with in-flight generation. On
    /// spawn failure / EOF / timeout the result is an empty `Vec` ([HG021]) — the
    /// caller still returns hardware/runtime without devices, never hangs.
    pub(crate) async fn sysinfo(&self) -> Vec<GpuDevice> {
        let halves = match (self.inner.factory)(self.inner.bus.clone(), "sysinfo") {
            Ok(h) => h,
            Err(e) => {
                warn!(error = %e, "higgs: sysinfo worker spawn failed");
                return Vec::new();
            }
        };
        let mut write = halves.write;
        let mut lines = BufReader::new(halves.read).lines();

        let gpus =
            match tokio::time::timeout(SYSINFO_RPC_TIMEOUT, sysinfo_one(&mut write, &mut lines))
                .await
            {
                Ok(v) => v,
                Err(_) => {
                    warn!("higgs: sysinfo RPC timed out");
                    Vec::new()
                }
            };

        // Reap the transient worker: close stdin so the worker exits its loop,
        // then wait/kill so it never lingers.
        drop(write);
        if let Some(mut child) = halves.proc {
            match tokio::time::timeout(WORKER_EXIT_TIMEOUT, child.wait()).await {
                Ok(_) => {}
                Err(_) => {
                    let _ = child.start_kill();
                    let _ = child.wait().await;
                }
            }
        }
        gpus
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
        // Bump so a stale crash-replay capturing an OLDER load is skipped.
        self.inner.load_epoch.fetch_add(1, Ordering::AcqRel);
    }

    /// The model id of the last recorded load (`None` if nothing is loaded). Used by the
    /// node runtime to report each resident worker's model in `M_NODE_INVENTORY`.
    pub(crate) fn loaded_model_id(&self) -> Option<String> {
        self.inner
            .last_load
            .lock()
            .as_ref()
            .and_then(|p| p.get("id").and_then(Value::as_str).map(str::to_string))
    }

    /// Forget the recorded `higgs/load` replay params — called after an explicit
    /// unload so a later unexpected worker restart does NOT reload the model the
    /// user just unloaded.
    ///
    /// Post-P4b the node's `unload`/`shutdown_all` reaps the whole Supervisor (its
    /// `last_load` dies with it), so nothing calls this anymore; kept (with its test)
    /// as the per-supervisor primitive a future explicit-unload-without-kill path
    /// would use.
    #[allow(dead_code)]
    pub(crate) fn clear_last_load(&self) {
        *self.inner.last_load.lock() = None;
        // Bump so a stale crash-replay capturing the now-cleared load is skipped.
        self.inner.load_epoch.fetch_add(1, Ordering::AcqRel);
    }

    /// Emit a lifecycle event on the broadcast channel.
    ///
    /// Post-P4b the LOCAL node owns the facade's event fan-out, so the `Higgs`
    /// facade no longer emits through a Supervisor; kept (with its tests) as the
    /// per-supervisor emit primitive the restart FSM's siblings use.
    #[allow(dead_code)]
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
        self.inner.demux.active_sink_count()
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
        let rx = self.inner.demux.register_pending(id);
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
            self.inner.demux.remove_pending(id);
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
        // Bump the worker-lifetime generation NOW — before installing the new `write_tx`
        // below — so that the instant this new lifetime begins, any reader left over from a
        // prior (stopped) lifetime is already marked stale. If we bumped only after
        // installing `write_tx`, an old reader hitting EOF in that window would see its own
        // generation still current and clobber the freshly installed channel. A bump
        // followed by a factory failure is harmless (generation only ever moves forward).
        let gen = self
            .inner
            .generation
            .fetch_add(1, Ordering::AcqRel)
            .wrapping_add(1);
        let halves = match (self.inner.factory)(self.inner.bus.clone(), model) {
            Ok(h) => h,
            Err(e) => {
                // Spawn failed: release the running flag so a later retry can spawn.
                self.inner.running.store(false, Ordering::Release);
                return Err(e);
            }
        };
        // Stash the child handle so stop() can wait on / kill it. No concurrent holder
        // exists at (re)spawn time because `start_for`/`do_spawn` is never concurrent
        // with `stop()` on the same Supervisor — this is a CALLER-ENFORCED precondition,
        // required because do_spawn's reset of `stopped=false` (above) would otherwise
        // defeat a stop()'s `stopped=true`:
        //   * local `Higgs` (api.rs) holds the `lifecycle` mutex across BOTH load()
        //     (which calls start_for) and unload()/stop() (which call sup.stop());
        //   * the node runtime (P1) awaits `start_for` fully inside `do_load` BEFORE the
        //     Supervisor is committed to the registry, and only a committed Supervisor is
        //     ever reaped/stopped.
        // Given that, a non-blocking `try_lock` is sufficient and avoids blocking_lock
        // inside the async runtime. (The reader's `attempt_restart` DOES race stop() — a
        // worker can die at any time — which is why it carries its own post-factory
        // `stopped` Acquire guard; do_spawn needs none under the precondition above.)
        if let Ok(mut guard) = self.inner.proc.try_lock() {
            *guard = halves.proc;
        }
        let (write_tx, write_rx) = mpsc::unbounded_channel::<String>();
        *self.inner.write_tx.lock() = Some(write_tx);
        // Writer task: owns the write half, drains the mpsc channel.
        // Exits when the sender half is dropped (stop() or do_spawn replacing it).
        tokio::spawn(writer_task(halves.write, write_rx));
        // Reader task: owns the read half, dispatches lines, triggers restart on EOF/error.
        // Tagged with the generation bumped ABOVE (before write_tx install), so a reader from
        // a prior lifetime is already stale by the time this worker is published.
        let inner = Arc::clone(&self.inner);
        tokio::spawn(reader_task(inner, halves.read, gen));
        Ok(())
    }
}

/// One synchronous M_SYSINFO round-trip on the transient sysinfo worker's raw
/// stdio. Encodes the request with the `rpc` codec the persistent path uses,
/// writes it, then reads lines until the matching response (id 1), skipping
/// strays. Returns the worker's `gpus` list, or an empty `Vec` on any
/// write/EOF/decode failure ([HG021] semantics — never hangs the sweep).
async fn sysinfo_one(
    write: &mut WriteHalf,
    lines: &mut tokio::io::Lines<BufReader<ReadHalf>>,
) -> Vec<GpuDevice> {
    let id: u64 = 1;
    let line = rpc::encode(&RpcFrame::Request(RpcRequest {
        jsonrpc: "2.0".into(),
        id,
        method: M_SYSINFO.to_string(),
        params: Value::Null,
    }));
    if write.write_all(line.as_bytes()).await.is_err()
        || write.write_all(b"\n").await.is_err()
        || write.flush().await.is_err()
    {
        warn!("higgs: sysinfo worker pipe broken");
        return Vec::new();
    }
    loop {
        match lines.next_line().await {
            Ok(Some(l)) if l.trim().is_empty() => continue,
            Ok(Some(l)) => match rpc::decode(&l) {
                Ok(RpcFrame::Response(resp)) if resp.id == id => {
                    if let Some(err) = resp.error {
                        warn!(detail = %err.message, "higgs: sysinfo worker returned an error");
                        return Vec::new();
                    }
                    let result = resp.result.unwrap_or(Value::Null);
                    // Deserialize the typed device list; a shape mismatch yields
                    // an empty list rather than a panic.
                    return result
                        .get("gpus")
                        .cloned()
                        .and_then(|g| serde_json::from_value::<Vec<GpuDevice>>(g).ok())
                        .unwrap_or_default();
                }
                Ok(_) => continue,
                Err(_) => continue,
            },
            Ok(None) => {
                warn!("higgs: sysinfo worker exited before replying");
                return Vec::new();
            }
            Err(e) => {
                warn!(error = %e, "higgs: sysinfo worker read error");
                return Vec::new();
            }
        }
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
async fn reader_task(inner: Arc<Inner>, read: ReadHalf, gen: u64) {
    let mut reader = BufReader::new(read).lines();

    // Every terminal exit `break`s so the single post-loop clear releases the
    // `running` lifetime flag exactly once. A successful restart `continue`s the
    // loop on the new transport without clearing it (the lifetime persists). A STALE
    // reader (a newer lifetime began) `return`s early WITHOUT clearing `running` — that
    // flag now belongs to the live worker.
    loop {
        let reason = match reader.next_line().await {
            Ok(Some(line)) => {
                dispatch(&inner, &line);
                continue;
            }
            Ok(None) => None,              // EOF — worker exited.
            Err(e) => Some(e.to_string()), // I/O error on the read half.
        };
        // Stale-reader guard: if a later `do_spawn` (a `start_for` after this worker's
        // `stop()`) bumped the generation, this reader belongs to a replaced lifetime. Exit
        // WITHOUT touching write_tx/pending/sinks/running — they are the live worker's now.
        if inner.generation.load(Ordering::Acquire) != gen {
            return;
        }
        let deliberate = inner.stopped.load(Ordering::Relaxed);
        // Pass `deliberate` so on_worker_death suppresses the event on clean stop.
        on_worker_death(&inner, reason, deliberate);
        if deliberate {
            break;
        }
        // 1 s backoff, then re-check stopped (F1: stop() may have raced) AND the generation
        // (a start_for may have begun a new lifetime during the sleep) before respawning.
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
        if inner.stopped.load(Ordering::Relaxed) {
            break;
        }
        if inner.generation.load(Ordering::Acquire) != gen {
            return;
        }
        match attempt_restart(&inner).await {
            Some(new_read) => {
                reader = BufReader::new(new_read).lines();
                // Continue outer loop on the new transport.
            }
            None => break,
        }
    }
    // Current-lifetime reader exiting (clean stop or respawn give-up): release the flag so a
    // future start() can spawn a fresh worker. Idempotent with stop()'s clear. (A stale
    // reader returned above and never reaches here, so it can't clear the live worker's flag.)
    inner.running.store(false, Ordering::Release);
}

/// Attempt a single respawn: call factory, install new write channel, return
/// new read half on success.  Returns `None` and exits if factory fails.
async fn attempt_restart(inner: &Arc<Inner>) -> Option<ReadHalf> {
    // Step 1 — build the replacement worker (argv0 stamped with the loaded id).
    let halves = match spawn_replacement(inner) {
        Ok(h) => h,
        Err(e) => {
            // Terminal factory failure: reap the dead old child and emit the
            // documented terminal WorkerDied before the reader gives up.
            warn!(error = %e, "higgs worker respawn failed — giving up");
            reap_old_child(inner).await;
            let _ = inner.events_tx.send(HiggsEvent::WorkerDied);
            return None;
        }
    };
    // Step 2 — stopped-race guard. stop()/unload() may have flipped `stopped`
    // after the reader's pre-call check but while factory() was spawning. If so,
    // reap the just-spawned child and abandon the restart WITHOUT installing it
    // or replaying — never resurrect a worker the user explicitly unloaded.
    if inner.stopped.load(Ordering::Acquire) {
        reap_child(halves.proc).await;
        return None;
    }
    // Step 3 — reap the OLD (dead) child and stash the new one so stop() can later wait on /
    // SIGKILL it. This re-checks `stopped` UNDER the proc lock (atomic with stop()'s reap):
    // if a stop() raced in after Step 2's check, install_child abandons + reaps the new child
    // and returns false — give up WITHOUT installing the writer or replaying, so a worker is
    // never resurrected after stop().
    if !install_child(inner, halves.proc).await {
        return None;
    }
    // Step 4 — install the new write channel + writer task (drops the old sender).
    install_writer(inner, halves.write);
    // Step 5 — announce the restart, then replay the last load (async, bounded).
    let _ = inner.events_tx.send(HiggsEvent::WorkerRestarted);
    info!("higgs worker restarted");
    replay_load(inner);
    Some(halves.read)
}

/// Call the factory for a respawn, stamping the loaded model id into argv0 so
/// the new worker is identifiable in `ps` (cosmetic — the model still reloads
/// via the M_LOAD replay). Returns the fresh I/O halves or the spawn error.
fn spawn_replacement(inner: &Arc<Inner>) -> Result<WorkerHalves, HiggsError> {
    let model = inner
        .last_load
        .lock()
        .as_ref()
        .and_then(|p| p.get("id").and_then(|v| v.as_str()).map(ToOwned::to_owned))
        .unwrap_or_default();
    (inner.factory)(inner.bus.clone(), &model)
}

/// Wait for `child` to exit, bounded by [`WORKER_EXIT_TIMEOUT`]; SIGKILL + reap on timeout.
///
/// Used wherever a child is reaped while a lock is held (restart path): a worker that EOF'd
/// its stdout but has not exited must NOT be able to wedge that lock indefinitely (which
/// would block `stop()`). Normally the EOF means the process already exited, so the wait
/// returns at once; the timeout+kill is the safety bound. Mirrors `stop()`'s reap.
async fn wait_or_kill(child: &mut tokio::process::Child) {
    if tokio::time::timeout(WORKER_EXIT_TIMEOUT, child.wait())
        .await
        .is_err()
    {
        let _ = child.start_kill();
        let _ = child.wait().await;
    }
}

/// Reap the OLD child currently stored in `inner.proc` (if any).
///
/// Its EOF is what triggered this restart, so it has already exited — but
/// `tokio::process::Child` does NOT reap on drop, so dropping it without `wait()` leaves a
/// zombie. The wait is bounded ([`wait_or_kill`]) so a not-yet-exited old worker can't wedge
/// the proc lock.
async fn reap_old_child(inner: &Arc<Inner>) {
    let mut proc_guard = inner.proc.lock().await;
    if let Some(mut old) = proc_guard.take() {
        wait_or_kill(&mut old).await;
    }
}

/// Force-kill and reap a just-spawned `child` that is being abandoned.
///
/// `start_kill()` only sends the signal; `wait()` reaps so the abandoned child
/// never lingers as a zombie (mirrors `stop()`). `None` (test factory) is a
/// no-op.
async fn reap_child(proc: Option<tokio::process::Child>) {
    if let Some(mut child) = proc {
        let _ = child.start_kill();
        let _ = child.wait().await;
    }
}

/// Reap the OLD child, then stash the NEW one in `inner.proc`. Returns `false` if a
/// concurrent `stop()` was observed and the new worker was abandoned instead of installed.
///
/// `do_spawn` does this for the initial spawn; the respawn path must too, or a worker
/// restarted after a death would be unreapable by `stop()`.
///
/// **stop()-race close:** `attempt_restart`'s pre-factory `stopped` check is a check-then-
/// act that races `stop()` across the `install_child` await — `stop()` sets `stopped=true`
/// FIRST and then takes THIS proc lock to reap, so re-checking `stopped` here UNDER the
/// proc lock makes install atomic with that reap. If `stop()` won the lock first we observe
/// `stopped` and abandon (reap the new child) rather than resurrect a worker after stop; if
/// we install first, `stop()`'s later proc-lock acquisition reaps the new child we stored.
async fn install_child(inner: &Arc<Inner>, new_proc: Option<tokio::process::Child>) -> bool {
    let mut proc_guard = inner.proc.lock().await;
    if inner.stopped.load(Ordering::Acquire) {
        // A stop() raced this restart and already ran (or is mid-reap holding nothing yet).
        // Do NOT resurrect: drop the lock, then reap the just-spawned child.
        drop(proc_guard);
        reap_child(new_proc).await;
        return false;
    }
    if let Some(mut old) = proc_guard.take() {
        // Bounded reap: a worker that EOF'd but hasn't exited can't wedge this lock (and
        // thereby block stop(), which contends for the same lock).
        wait_or_kill(&mut old).await;
    }
    *proc_guard = new_proc;
    true
}

/// Install a fresh write channel + writer task for the respawned worker.
///
/// Replacing `write_tx` drops the previous sender, signalling the old writer
/// task to exit; the new writer task owns `write`.
fn install_writer(inner: &Arc<Inner>, write: WriteHalf) {
    let (write_tx, write_rx) = mpsc::unbounded_channel::<String>();
    *inner.write_tx.lock() = Some(write_tx);
    tokio::spawn(writer_task(write, write_rx));
}

/// Decode one line and dispatch to the appropriate handler.
fn dispatch(inner: &Arc<Inner>, line: &str) {
    match rpc::decode(line) {
        Ok(RpcFrame::Response(resp)) => inner.demux.correlate(resp),
        Ok(RpcFrame::Notification(notif)) => route_notification(inner, &notif),
        Ok(RpcFrame::Request(_)) => { /* workers never send requests */ }
        Err(e) => warn!(error = %e, "higgs: dropped malformed worker line"),
    }
}

/// Route a `higgs/chat/chunk` notification to the matching per-request sink.
///
/// The notification params carry `request_id` (echoed from the M_CHAT request)
/// and `delta`.  The demux is looked up by `request_id`; if no entry exists
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
    inner.demux.route_chunk(request_id, delta);
}

/// Handle worker death: fail all pending requests, clear write channel, and
/// (for unexpected deaths only) broadcast `WorkerDied` and log at warn.
///
/// `deliberate` is true when `stop()` was called before the EOF was observed —
/// in that case neither the broadcast event nor the warn log is emitted (clean
/// shutdown is not a death event). This enforces four-pillar pillar 3: log once
/// at origin, not at every boundary that observes the same event.
fn on_worker_death(inner: &Arc<Inner>, reason: Option<String>, deliberate: bool) {
    // ORDER MATTERS — close the send→register race with a concurrent `send_request`:
    // clear `write_tx` FIRST, THEN fail pending. A racing send either (a) locks `write_tx`
    // after this clear → sees `None` → fast-fails and removes its own pending entry, or
    // (b) locked `write_tx` before the clear (so it registered its pending entry even
    // earlier, in code order) → that entry is caught by the `fail_all_pending` below. Either
    // way a request issued during worker death fails fast instead of waiting out its RPC
    // timeout. (If these ran in the opposite order, a send could register after
    // fail_all_pending yet still send through a not-yet-cleared `write_tx` and then block.)
    *inner.write_tx.lock() = None;
    // Fail all pending requests — the rx.await in `request` sees Err.
    inner.demux.fail_all_pending();
    // Drop all chat sinks — closes each receiver, ending all in-flight streams.
    inner.demux.clear_sinks();

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
    // Capture the load epoch alongside the params: if an explicit load/unload bumps it
    // before this async replay fires, the captured model is stale and the replay is skipped.
    let captured_epoch = inner.load_epoch.load(Ordering::Acquire);

    let inner2 = Arc::clone(inner);
    tokio::spawn(async move {
        // Residual-TOCTOU close: the reader re-checks `stopped` before installing
        // the respawned worker, but that check fires before this async replay is
        // even spawned. A deliberate stop/unload that flips `stopped` in the
        // narrow window between the install guard and this M_LOAD must NOT
        // resurrect the model. Re-check here, immediately before the load fires;
        // an Acquire load pairs with the Release store in `stop()`. (When unload
        // also cleared `last_load` first, the `let Some` above already returned —
        // this covers the path where `stopped` flips without that clear.)
        if inner2.stopped.load(Ordering::Acquire) {
            return;
        }
        // Stale-replay close: if an explicit load/unload superseded the captured model
        // (epoch changed), do NOT resurrect the old model over the newer intent. (A tight
        // residual remains: an explicit load whose `record_last_load` has not landed yet
        // won't be observed — the FIFO ordering of that in-flight M_LOAD then decides; this
        // narrows the window from "any delayed replay" to that sub-window. Documented in the
        // Supervisor assessment.)
        if inner2.load_epoch.load(Ordering::Acquire) != captured_epoch {
            return;
        }
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
/// Used for the post-restart load replay (and any future awaited replay RPC).
/// Scan is host-side, so restart replays only the load — never a scan. Returns
/// the raw result value on success or a typed error on failure.
async fn replay_rpc_await(
    inner: &Arc<Inner>,
    method: &str,
    params: Value,
) -> Result<Value, HiggsError> {
    let id = inner.next_id.fetch_add(1, Ordering::Relaxed);
    let rx = inner.demux.register_pending(id);

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
        inner.demux.remove_pending(id);
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
            inner.demux.remove_pending(id);
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
/// to keep [`replay_load`] readable.
async fn replay_load_await(inner: &Arc<Inner>, params: Value) -> Result<(), HiggsError> {
    replay_rpc_await(inner, M_LOAD, params).await.map(|_| ())
}

/// Production factory: re-exec current binary with `--higgs-worker`.
///
/// Spawns via `tokio::process::Command` with `stdin` + `stdout` piped.
/// Takes owned `ChildStdin` and `ChildStdout` halves — independently owned
/// by construction, no mutex between reader and writer.
/// Wires a blocking stderr drain task to fill the ring.
fn production_factory(bus: Arc<LogBus>, model: &str) -> Result<WorkerHalves, HiggsError> {
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
    // Seed the new worker's engine-log verbosity from the current toggle, so a
    // worker spawned while Verbose Logging is on starts at DEBUG (live toggles
    // thereafter go via the M_LOG_LEVEL RPC).
    cmd.env(
        "HIGGS_WORKER_VERBOSE",
        if bus.verbose() { "1" } else { "0" },
    );
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
            // Tagged `Worker` so it streams in the worker-only debug console;
            // appends to the history ring AND fans out to live SSE subscribers.
            bus.push(LogSource::Worker, line);
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
    use crate::worker::M_CHAT;
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

    // A spawn bumps the worker-lifetime generation (the tag the stale-reader guard compares
    // against), and a redundant start() while already running does NOT bump it.
    #[tokio::test]
    async fn spawn_bumps_generation_redundant_start_does_not() {
        // make_supervisor already called start_for once, so the spawn bumped gen 0 → 1.
        let (sup, _tw, _tr) = make_supervisor();
        assert_eq!(
            sup.inner.generation.load(Ordering::Acquire),
            1,
            "spawn bumped gen to 1"
        );
        // Redundant start() while running is an idempotent no-op (running.swap guard) — it
        // must NOT bump the generation or the live reader would be wrongly marked stale.
        sup.start_for("org/model").expect("redundant start ok");
        assert_eq!(
            sup.inner.generation.load(Ordering::Acquire),
            1,
            "redundant start does not bump generation"
        );
        sup.stop().await;
    }

    // A crash-replay must SKIP if an explicit load/unload superseded the captured model
    // (load_epoch bumped) — otherwise a stale replay would resurrect the old model over the
    // newer one. Drive replay_load directly, bump the epoch before the spawned task runs
    // (current-thread runtime: it can't run until we await), and assert NO M_LOAD is sent.
    #[tokio::test]
    async fn replay_load_skips_when_epoch_superseded() {
        let (sup, _tw, test_read) = make_supervisor(); // worker running (gen 1)
        sup.record_last_load(json!({ "id": "org/a", "path": "/x" }));
        // replay_load captures the current epoch synchronously, then spawns the replay task.
        replay_load(&sup.inner);
        // Supersede before the task runs: a newer explicit load/unload bumped the epoch.
        sup.inner.load_epoch.fetch_add(1, Ordering::AcqRel);
        // The replay task now observes a changed epoch and returns without sending M_LOAD.
        let mut lines = BufReader::new(test_read).lines();
        let got =
            tokio::time::timeout(std::time::Duration::from_millis(200), lines.next_line()).await;
        match got {
            // Timed out / EOF / error → nothing was sent → replay correctly skipped.
            Err(_) | Ok(Ok(None)) | Ok(Err(_)) => {}
            // A line was sent: it must NOT be an M_LOAD (that would be the stale replay).
            Ok(Ok(Some(line))) => {
                assert!(
                    !line.contains("higgs/load"),
                    "superseded replay sent M_LOAD: {line}"
                );
            }
        }
        sup.stop().await;
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
        let pending_id = sup.inner.next_id.fetch_add(1, Ordering::Relaxed);
        let reply_rx = sup.inner.demux.register_pending(pending_id);

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

    // ─── Test 3-b: chat RPC times out → HG016 ChatTimeout ────────────────────
    //
    // Uses a paused clock so the 600 s CHAT_RPC_TIMEOUT elapses instantly. The
    // worker never responds; request_with_id must remove its pending entry and
    // return ChatTimeout (→ 504), not hang.

    #[tokio::test(start_paused = true)]
    async fn chat_rpc_times_out() {
        let (sup, _test_write, _test_read) = make_supervisor();
        // Pre-allocate the id the way the chat path does.
        let id = sup.alloc_request_id();
        let fut = sup.request_with_id(id, M_CHAT, json!({"request_id": id}));
        // Advance past the chat-RPC timeout with no response written.
        tokio::time::advance(CHAT_RPC_TIMEOUT + std::time::Duration::from_secs(1)).await;
        let err = fut.await.expect_err("must time out");
        assert!(matches!(err, HiggsError::ChatTimeout { .. }), "got {err}");
        assert!(err.to_string().starts_with("[HG016]"));
        // The orphaned pending entry was removed (no leak).
        assert_eq!(sup.inner.demux.pending_count(), 0);
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
        // The supervisor's log history caps at the LogBus ring capacity (2000);
        // the production path fills it via the stderr drain task. Drive the same
        // bus the supervisor exposes and confirm the cap + oldest-first tail.
        let (sup, _tw, _tr) = make_supervisor();
        for i in 0..2100usize {
            sup.inner.bus.push(LogSource::Worker, format!("line-{i}"));
        }
        let snap = sup.logs(usize::MAX, None);
        assert_eq!(snap.len(), 2000);
        // 2100 pushed, 100 dropped → oldest remaining is line-100.
        assert_eq!(snap.first().unwrap(), "line-100");
    }

    // ─── Test 6: restart replays the load (scan is host-side) + emits ModelLoaded ─
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

    /// Residual-TOCTOU close (#1): when `stopped` flips between the restart's
    /// install guard and the async replay, `replay_load` must abort — no M_LOAD
    /// frame may reach the (respawned) worker, since the user deliberately
    /// stopped/unloaded. Drives `replay_load` directly with `stopped` already
    /// set and asserts nothing is written to the worker transport.
    #[tokio::test]
    async fn replay_aborts_when_stopped_flips() {
        let (sup, _test_write, test_read) = make_supervisor();

        // A load is recorded (as after a normal load) so replay HAS something to
        // send — proving the abort is driven by `stopped`, not an empty replay.
        sup.record_last_load(json!({"id": "org/model"}));
        // Deliberate stop/unload flipped the flag in the TOCTOU window.
        sup.inner.stopped.store(true, Ordering::Release);

        // Fire the replay: the spawned task re-checks `stopped` before the M_LOAD.
        replay_load(&sup.inner);

        // Give the spawned task time to run (and to NOT send anything).
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        // No frame must have been written to the worker. Read with a short
        // timeout: a clean abort means the read times out (nothing arrived).
        use tokio::io::AsyncBufReadExt;
        let mut lines = BufReader::new(test_read).lines();
        let got =
            tokio::time::timeout(std::time::Duration::from_millis(100), lines.next_line()).await;
        assert!(
            got.is_err(),
            "replay must send NO M_LOAD when stopped flipped, but a frame arrived"
        );
    }

    /// `logs(n)` returns the tail of the stderr ring, oldest first, and clamps
    /// to the ring length when fewer than `n` lines are present.
    #[tokio::test]
    async fn logs_tail_and_clamp() {
        let (sup, _tw, _tr) = make_supervisor();
        sup.inner.bus.push(LogSource::Worker, "a".to_owned());
        sup.inner.bus.push(LogSource::Worker, "b".to_owned());
        sup.inner.bus.push(LogSource::Worker, "c".to_owned());
        // Tail of 2 → last two, oldest first.
        assert_eq!(sup.logs(2, None), vec!["b".to_owned(), "c".to_owned()]);
        // n larger than the ring → all lines, no panic.
        assert_eq!(
            sup.logs(100, None),
            vec!["a".to_owned(), "b".to_owned(), "c".to_owned()]
        );
        // n == 0 → empty.
        assert!(sup.logs(0, None).is_empty());
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

    // ─── Transient-worker factory helper (sysinfo) ───────────────────────────
    //
    // `sysinfo` calls the factory directly for a SEPARATE, transient worker (no
    // `start_for`, no persistent reader/writer). This builds a supervisor whose
    // factory hands out exactly one duplex pair and returns the test control
    // handles for that transient worker.
    //
    //   sup_write ↔ test_read  : supervisor writes the M_SYSINFO request
    //   test_write ↔ sup_read  : test writes the worker's response back
    fn transient_supervisor() -> (
        Supervisor,
        tokio::io::DuplexStream, // test writes → supervisor reads
        tokio::io::DuplexStream, // supervisor writes → test reads
    ) {
        let (sup_write, test_read) = tokio::io::duplex(64 * 1024);
        let (test_write, sup_read) = tokio::io::duplex(64 * 1024);
        let sup_write_cell = Arc::new(Mutex::new(Some(sup_write)));
        let sup_read_cell = Arc::new(Mutex::new(Some(sup_read)));
        let sup = Supervisor::with_factory(Box::new(move |_ring, _model| {
            Ok(WorkerHalves {
                write: Box::new(sup_write_cell.lock().take().ok_or_else(|| {
                    HiggsError::WorkerSpawnFailed {
                        source: std::io::Error::other("mock: no more write halves"),
                    }
                })?),
                read: Box::new(sup_read_cell.lock().take().ok_or_else(|| {
                    HiggsError::WorkerSpawnFailed {
                        source: std::io::Error::other("mock: no more read halves"),
                    }
                })?),
                proc: None,
            })
        }));
        (sup, test_write, test_read)
    }

    /// Build a supervisor whose factory always fails — used to drive the
    /// spawn-failure verdict paths in `sysinfo` and `start_for`/`do_spawn`.
    fn failing_supervisor() -> Supervisor {
        Supervisor::with_factory(Box::new(|_ring, _model| {
            Err(HiggsError::WorkerSpawnFailed {
                source: std::io::Error::other("mock: spawn always fails"),
            })
        }))
    }

    /// Read one request line the supervisor wrote to the transient worker and
    /// return its decoded JSON-RPC request id.
    async fn read_request_id(obs: &mut tokio::io::DuplexStream) -> u64 {
        use tokio::io::AsyncBufReadExt;
        let mut lines = BufReader::new(obs).lines();
        let line = lines
            .next_line()
            .await
            .expect("read request line")
            .expect("request line present");
        let v: Value = serde_json::from_str(&line).expect("valid json frame");
        v["id"].as_u64().expect("request carries an id")
    }

    // ─── sysinfo: success → typed GpuDevice list deserialized from worker ─────

    #[tokio::test]
    async fn sysinfo_returns_devices() {
        let (sup, mut test_write, mut test_read) = transient_supervisor();
        let task = tokio::spawn(async move { sup.sysinfo().await });

        let id = read_request_id(&mut test_read).await;
        write_line(
            &mut test_write,
            &ok_response(
                id,
                json!({"gpus": [{
                    "name": "Metal",
                    "description": "Apple M3 Max",
                    "kind": "Gpu",
                    "vram_total_bytes": 42,
                    "vram_free_bytes": 21
                }]}),
            ),
        )
        .await;

        let gpus = task.await.expect("sysinfo task");
        assert_eq!(gpus.len(), 1);
        assert_eq!(gpus[0].name, "Metal");
        assert_eq!(gpus[0].vram_total_bytes, 42);
    }

    // ─── sysinfo_one: worker error reply → empty device list ─────────────────

    #[tokio::test]
    async fn sysinfo_error_reply_is_empty() {
        let (sup, mut test_write, mut test_read) = transient_supervisor();
        let task = tokio::spawn(async move { sup.sysinfo().await });

        let id = read_request_id(&mut test_read).await;
        write_line(&mut test_write, &err_response(id, -32000, "no devices")).await;

        assert!(task.await.expect("sysinfo task").is_empty());
    }

    // ─── sysinfo_one: stray frame skipped, then EOF → empty (covers Ok(_) arm) ─

    #[tokio::test]
    async fn sysinfo_skips_stray_then_eof() {
        let (sup, mut test_write, mut test_read) = transient_supervisor();
        let task = tokio::spawn(async move { sup.sysinfo().await });

        let _ = read_request_id(&mut test_read).await;
        // Blank + stray notification (skipped), then EOF before a real response.
        write_line(&mut test_write, "").await;
        write_line(&mut test_write, &chunk_notif(1, "noise")).await;
        drop(test_write);

        assert!(task.await.expect("sysinfo task").is_empty());
    }

    // ─── sysinfo: factory spawn failure → empty device list ──────────────────

    #[tokio::test]
    async fn sysinfo_spawn_failure_is_empty() {
        let sup = failing_supervisor();
        assert!(sup.sysinfo().await.is_empty());
    }

    // ─── sysinfo_one: pipe broken on write → empty ───────────────────────────

    #[tokio::test]
    async fn sysinfo_pipe_broken_on_write() {
        let (sup, _test_write, test_read) = transient_supervisor();
        drop(test_read);
        assert!(sup.sysinfo().await.is_empty());
    }

    // ─── request(): control RPC times out → WorkerDead, pending drained ──────
    //
    // Paused clock so the 120 s CONTROL_RPC_TIMEOUT elapses instantly. No
    // response is ever written; `request` must remove its orphaned pending entry
    // and return WorkerDead("… timed out …") rather than hang (lines 359-366).

    #[tokio::test(start_paused = true)]
    async fn control_rpc_times_out() {
        let (sup, _test_write, _test_read) = make_supervisor();
        let fut = sup.request("higgs/status", json!({}));
        tokio::time::advance(CONTROL_RPC_TIMEOUT + std::time::Duration::from_secs(1)).await;
        let err = fut.await.expect_err("must time out");
        assert!(matches!(err, HiggsError::WorkerDead { .. }), "got {err}");
        assert!(err.to_string().contains("timed out"), "display: {err}");
        // Orphaned pending entry removed — no leak.
        assert_eq!(sup.inner.demux.pending_count(), 0);
    }

    // ─── send_request(): worker dies before responding → WorkerDead ──────────
    //
    // Registers a request, then drops the supervisor's pending oneshot sender
    // (simulating on_worker_death draining it) so `rx.await` yields Err and the
    // "worker died before response" branch (lines 746-748) is taken.

    #[tokio::test]
    async fn send_request_worker_dies_before_response() {
        let (sup, _test_write, _test_read) = make_supervisor();
        // Spawn the request on its own task so it polls (registers its pending
        // entry + sends its frame) independently of this test's timeline.
        let inner = Arc::clone(&sup.inner);
        let task = tokio::spawn(async move {
            Supervisor { inner }
                .request("higgs/status", json!({}))
                .await
        });
        // Let the request register its pending entry + send its frame.
        tokio::time::sleep(std::time::Duration::from_millis(30)).await;
        // Fail the pending request (what on_worker_death does on death) →
        // dropping its sender closes the oneshot, so rx.await returns Err.
        assert_eq!(
            sup.inner.demux.pending_count(),
            1,
            "exactly one pending request"
        );
        sup.inner.demux.fail_all_pending();
        let err = task
            .await
            .expect("task")
            .expect_err("worker died before response");
        assert!(
            err.to_string().contains("worker died before response"),
            "display: {err}"
        );
    }

    // ─── send_request(): no worker running → WorkerDead immediately ──────────

    #[tokio::test]
    async fn send_request_no_worker_is_dead() {
        // No start_for → write_tx is None → send fails before any await.
        let sup = failing_supervisor();
        let err = sup.request("higgs/status", json!({})).await.unwrap_err();
        assert!(err.to_string().contains("[HG007]"), "display: {err}");
        assert_eq!(
            sup.inner.demux.pending_count(),
            0,
            "pending must be cleaned"
        );
    }

    // ─── do_spawn(): factory failure surfaces the error, running flag released ─

    #[tokio::test]
    async fn start_for_spawn_failure_releases_running() {
        let sup = failing_supervisor();
        let err = sup.start_for("org/model").expect_err("spawn must fail");
        assert!(
            matches!(err, HiggsError::WorkerSpawnFailed { .. }),
            "got {err}"
        );
        // running was released, so a later start can retry (and also fail).
        assert!(!sup.inner.running.load(Ordering::Relaxed));
        assert!(sup.start_for("org/model").is_err(), "retry also fails");
    }

    // ─── set_worker_verbose(): writes a frame when a worker is live ──────────
    //
    // Fire-and-forget M_LOG_LEVEL: no pending entry, no await. With a worker
    // live the frame is written to the worker transport; with none it is a
    // silent no-op (write_tx is None).

    #[tokio::test]
    async fn set_worker_verbose_writes_when_live() {
        let (sup, _test_write, mut test_read) = make_supervisor();
        sup.set_worker_verbose(true);

        use tokio::io::AsyncBufReadExt;
        let line = tokio::time::timeout(std::time::Duration::from_millis(500), async {
            let mut lines = BufReader::new(&mut test_read).lines();
            lines.next_line().await.unwrap().expect("frame written")
        })
        .await
        .expect("frame must arrive");
        let v: Value = serde_json::from_str(&line).expect("valid json");
        assert_eq!(v["method"], M_LOG_LEVEL);
        assert_eq!(v["params"]["verbose"], true);
    }

    #[tokio::test]
    async fn set_worker_verbose_noop_without_worker() {
        // No worker live → write_tx None → no-op, no panic.
        let sup = failing_supervisor();
        sup.set_worker_verbose(false); // must not panic
        sup.set_worker_verbose(true);
    }

    // ─── dispatch(): worker Request and malformed line are no-ops ─────────────
    //
    // Workers never send requests; a Request frame and an undecodable line both
    // hit the silent / warn arms of `dispatch` (lines 1114-1115) without panic.

    #[tokio::test]
    async fn dispatch_ignores_request_and_malformed() {
        let (sup, _tw, _tr) = make_supervisor();
        // Worker-originated Request frame — silently ignored.
        let req = rpc::encode(&RpcFrame::Request(RpcRequest {
            jsonrpc: "2.0".into(),
            id: 1,
            method: "higgs/whatever".into(),
            params: json!({}),
        }));
        dispatch(&sup.inner, &req);
        // Undecodable line — warn-and-drop.
        dispatch(&sup.inner, "{not valid json");
        dispatch(&sup.inner, "");
        // Pending map untouched, no panic.
        assert_eq!(sup.inner.demux.pending_count(), 0);
    }

    // ─── on_worker_death(): deliberate stop is silent; drains pending+sinks ───

    #[tokio::test]
    async fn on_worker_death_deliberate_is_silent() {
        let (sup, _tw, _tr) = make_supervisor();
        // Seed a pending entry and a chat sink so the clear branches execute.
        let rx = sup.inner.demux.register_pending(77);
        let mut sink_rx = sup.register_chat_sink(88);
        assert_eq!(sup.chat_sinks_count(), 1);

        on_worker_death(&sup.inner, None, true);

        // pending failed (oneshot closed), sinks cleared, write_tx None.
        assert_eq!(sup.inner.demux.pending_count(), 0);
        assert_eq!(sup.chat_sinks_count(), 0);
        assert!(sup.inner.write_tx.lock().is_none());
        assert!(rx.await.is_err(), "pending oneshot closed");
        assert!(sink_rx.recv().await.is_none(), "sink closed");
    }

    // ─── on_worker_death(): unexpected death with no reason → EOF warn arm ────

    #[tokio::test]
    async fn on_worker_death_unexpected_no_reason_emits() {
        let (sup, _tw, _tr) = make_supervisor();
        let mut events = sup.events();
        // reason = None exercises the "exited unexpectedly (EOF)" warn arm.
        on_worker_death(&sup.inner, None, false);
        let got = tokio::time::timeout(std::time::Duration::from_millis(200), events.recv())
            .await
            .expect("event within timeout")
            .expect("WorkerDied broadcast");
        assert!(matches!(got, HiggsEvent::WorkerDied));
    }

    // ─── on_worker_death(): unexpected death WITH a reason → warn(detail) arm ─

    #[tokio::test]
    async fn on_worker_death_unexpected_with_reason_emits() {
        let (sup, _tw, _tr) = make_supervisor();
        let mut events = sup.events();
        // reason = Some(..) exercises the `warn!(detail = %r, ...)` arm.
        on_worker_death(&sup.inner, Some("read error: broken pipe".into()), false);
        let got = tokio::time::timeout(std::time::Duration::from_millis(200), events.recv())
            .await
            .expect("event within timeout")
            .expect("WorkerDied broadcast");
        assert!(matches!(got, HiggsEvent::WorkerDied));
    }

    // ─── replay_load(): no recorded load → early return, nothing sent ────────

    #[tokio::test]
    async fn replay_load_empty_is_noop() {
        let (sup, _test_write, test_read) = make_supervisor();
        // No record_last_load → the `let Some(...) else { return }` returns.
        replay_load(&sup.inner);
        tokio::time::sleep(std::time::Duration::from_millis(30)).await;
        use tokio::io::AsyncBufReadExt;
        let mut lines = BufReader::new(test_read).lines();
        let got =
            tokio::time::timeout(std::time::Duration::from_millis(80), lines.next_line()).await;
        assert!(got.is_err(), "no replay frame when nothing recorded");
    }

    // ─── replay_load(): recorded load + worker error reply → logged-and-dropped ─
    //
    // Drives the Err arm of replay_load's spawned task (lines 1219-1220): the
    // replayed M_LOAD gets a JSON-RPC error back, so NO ModelLoaded is emitted.

    #[tokio::test]
    async fn replay_load_error_emits_no_model_loaded() {
        let (sup, mut test_write, mut test_read) = make_supervisor();
        let mut events = sup.events();
        sup.record_last_load(json!({"id": "org/model"}));

        replay_load(&sup.inner);

        // The replayed M_LOAD is written to the (live) worker transport; reply
        // with an error so replay_load_await returns Err.
        let load_id = read_request_id(&mut test_read).await;
        write_line(
            &mut test_write,
            &err_response(load_id, -32000, "load failed"),
        )
        .await;

        // No ModelLoaded must arrive within a short window.
        let got = tokio::time::timeout(std::time::Duration::from_millis(150), async {
            loop {
                match events.recv().await {
                    Ok(HiggsEvent::ModelLoaded { .. }) => return true,
                    Ok(_) => continue,
                    Err(_) => return false,
                }
            }
        })
        .await;
        assert!(got.is_err(), "error reply must NOT emit ModelLoaded");
    }

    // ─── replay_rpc_await(): no worker running → WorkerDead, pending cleaned ──

    #[tokio::test]
    async fn replay_rpc_await_no_worker() {
        // Build inner with no live worker (write_tx None).
        let sup = failing_supervisor();
        let err = replay_rpc_await(&sup.inner, M_LOAD, json!({"id": "x"}))
            .await
            .expect_err("no worker → WorkerDead");
        assert!(err.to_string().contains("no worker running"), "got {err}");
        assert_eq!(sup.inner.demux.pending_count(), 0);
    }

    // ─── replay_rpc_await(): worker dies before response → WorkerDead ─────────

    #[tokio::test]
    async fn replay_rpc_await_worker_dies() {
        let (sup, _test_write, _test_read) = make_supervisor();
        let inner = Arc::clone(&sup.inner);
        let fut = tokio::spawn(async move { replay_rpc_await(&inner, M_LOAD, json!({})).await });
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        // Fail the pending request so rx resolves Err (worker death simulation).
        assert_eq!(sup.inner.demux.pending_count(), 1);
        sup.inner.demux.fail_all_pending();
        let err = fut.await.expect("task").expect_err("worker died");
        assert!(
            err.to_string().contains("worker died before response"),
            "got {err}"
        );
    }

    // ─── replay_rpc_await(): worker accepts stdin but never replies → timeout ─
    //
    // Paused clock so CONTROL_RPC_TIMEOUT elapses instantly; the orphaned
    // pending entry must be removed (lines 1287-1292).

    #[tokio::test(start_paused = true)]
    async fn replay_rpc_await_times_out() {
        let (sup, _test_write, _test_read) = make_supervisor();
        let inner = Arc::clone(&sup.inner);
        let fut = tokio::spawn(async move { replay_rpc_await(&inner, M_LOAD, json!({})).await });
        // Let the request register + send before advancing the clock.
        tokio::task::yield_now().await;
        tokio::time::sleep(std::time::Duration::from_millis(1)).await;
        tokio::time::advance(CONTROL_RPC_TIMEOUT + std::time::Duration::from_secs(1)).await;
        let err = fut.await.expect("task").expect_err("must time out");
        assert!(err.to_string().contains("timed out"), "got {err}");
        assert_eq!(sup.inner.demux.pending_count(), 0, "orphan removed");
    }

    // ─── replay_rpc_await(): worker RPC error → WorkerRpc with worker_code ────
    //
    // Reply carries a `data.code` so the worker's origin diagnostic is recovered
    // (lines 1269-1278). Covers the success-path encode/insert too.

    #[tokio::test]
    async fn replay_rpc_await_worker_rpc_error() {
        let (sup, mut test_write, mut test_read) = make_supervisor();
        let inner = Arc::clone(&sup.inner);
        let fut = tokio::spawn(async move { replay_rpc_await(&inner, M_LOAD, json!({})).await });

        let id = read_request_id(&mut test_read).await;
        // Error reply carrying a worker diagnostic code in `data`.
        let frame = rpc::encode(&RpcFrame::Response(rpc::RpcResponse {
            jsonrpc: "2.0".into(),
            id,
            result: None,
            error: Some(rpc::RpcError {
                code: -32000,
                message: "load bad".into(),
                data: Some(json!({"code": "HG005"})),
            }),
        }));
        write_line(&mut test_write, &frame).await;

        let err = fut.await.expect("task").expect_err("worker rpc error");
        match err {
            HiggsError::WorkerRpc {
                worker_code,
                message,
                ..
            } => {
                assert_eq!(worker_code.as_deref(), Some("HG005"));
                assert_eq!(message, "load bad");
            }
            other => panic!("expected WorkerRpc, got {other}"),
        }
    }

    // ─── writer_task(): write failure on closed read end ends the task ───────
    //
    // The writer drains the channel into the worker's stdin. Dropping the read
    // end of the duplex makes `write_all` fail, breaking the loop (lines 938-944)
    // — silent, the reader drives the death path.

    #[tokio::test]
    async fn writer_task_exits_on_broken_pipe() {
        let (sup_write, test_read) = tokio::io::duplex(64);
        drop(test_read); // close the peer so writes fail
        let (tx, rx) = mpsc::unbounded_channel::<String>();
        let handle = tokio::spawn(writer_task(Box::new(sup_write), rx));
        // Send a line; the write fails and the task returns.
        tx.send("hello".to_string()).expect("send queued");
        // Task must complete (not hang) once the broken write breaks the loop.
        tokio::time::timeout(std::time::Duration::from_millis(500), handle)
            .await
            .expect("writer task should exit on broken pipe")
            .expect("task join");
    }

    // ─── reader_task: deliberate stop racing a death → break, no respawn ─────
    //
    // `stopped` is set before the EOF is observed, so on_worker_death is silent
    // and the reader breaks immediately (line 973) without the 1s backoff or a
    // respawn — exercising the `if deliberate { break; }` EOF arm.

    #[tokio::test]
    async fn reader_task_breaks_on_deliberate_stop_eof() {
        let (sup_write, _obs) = tokio::io::duplex(64 * 1024);
        let (test_write, sup_read) = tokio::io::duplex(64 * 1024);
        let cell_w = Arc::new(Mutex::new(Some(sup_write)));
        let cell_r = Arc::new(Mutex::new(Some(sup_read)));
        let sup = Supervisor::with_factory(Box::new(move |_ring, _model| {
            Ok(WorkerHalves {
                write: Box::new(cell_w.lock().take().expect("one spawn")),
                read: Box::new(cell_r.lock().take().expect("one spawn")),
                proc: None,
            })
        }));
        sup.start_for("org/model").expect("start");
        let mut events = sup.events();

        // Mark deliberate stop, THEN trigger EOF. The reader must break without
        // emitting WorkerDied and without attempting a respawn.
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        sup.inner.stopped.store(true, Ordering::Release);
        drop(test_write);

        // No WorkerDied event must arrive (deliberate stop is silent).
        let got = tokio::time::timeout(std::time::Duration::from_millis(300), events.recv()).await;
        assert!(got.is_err(), "deliberate EOF must not broadcast WorkerDied");
        // running released by the post-loop clear.
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        assert!(!sup.inner.running.load(Ordering::Relaxed));
    }

    // ─── attempt_restart(): stopped flips during respawn → reap + abandon ────
    //
    // The reader's pre-call check passes, but `stopped` flips while the factory
    // spawns the replacement. attempt_restart must abandon WITHOUT installing
    // the new worker or replaying (lines 1033-1036). Driven directly so the race
    // window is deterministic: stopped is already set, but last_load is present
    // and the factory succeeds, so the only reason to abandon is the guard.

    #[tokio::test]
    async fn attempt_restart_abandons_when_stopped() {
        let (sup_write, _obs) = tokio::io::duplex(64 * 1024);
        let (_test_write, sup_read) = tokio::io::duplex(64 * 1024);
        let cell_w = Arc::new(Mutex::new(Some(sup_write)));
        let cell_r = Arc::new(Mutex::new(Some(sup_read)));
        let sup = Supervisor::with_factory(Box::new(move |_ring, _model| {
            Ok(WorkerHalves {
                write: Box::new(cell_w.lock().take().ok_or_else(|| {
                    HiggsError::WorkerSpawnFailed {
                        source: std::io::Error::other("one spawn"),
                    }
                })?),
                read: Box::new(cell_r.lock().take().ok_or_else(|| {
                    HiggsError::WorkerSpawnFailed {
                        source: std::io::Error::other("one spawn"),
                    }
                })?),
                proc: None,
            })
        }));
        // Stopped already set → the post-factory guard fires.
        sup.inner.stopped.store(true, Ordering::Release);
        let got = attempt_restart(&sup.inner).await;
        assert!(got.is_none(), "must abandon restart when stopped flipped");
        // No write_tx installed (the new worker was abandoned, not wired in).
        assert!(sup.inner.write_tx.lock().is_none());
    }

    // ─── attempt_restart(): factory failure → reap old + terminal WorkerDied ─
    //
    // spawn_replacement fails, so attempt_restart logs, reaps the (None) old
    // child, broadcasts a terminal WorkerDied, and returns None (lines 1020-1027).

    #[tokio::test]
    async fn attempt_restart_factory_failure_is_terminal() {
        let sup = failing_supervisor();
        sup.record_last_load(json!({"id": "org/model"}));
        let mut events = sup.events();
        let got = attempt_restart(&sup.inner).await;
        assert!(got.is_none(), "factory failure → give up");
        let died = tokio::time::timeout(std::time::Duration::from_millis(200), events.recv())
            .await
            .expect("event")
            .expect("WorkerDied");
        assert!(matches!(died, HiggsEvent::WorkerDied));
    }

    // ─── spawn_replacement(): stamps the recorded model id into argv0 ────────
    //
    // The replacement factory call carries the last-load `id`. Drive it through
    // a factory that records the model arg it was handed.

    #[tokio::test]
    async fn spawn_replacement_passes_recorded_model() {
        let seen = Arc::new(Mutex::new(Vec::<String>::new()));
        let seen2 = Arc::clone(&seen);
        let sup = Supervisor::with_factory(Box::new(move |_ring, model| {
            seen2.lock().push(model.to_string());
            Err(HiggsError::WorkerSpawnFailed {
                source: std::io::Error::other("capture only"),
            })
        }));
        sup.record_last_load(json!({"id": "org/cool-model"}));
        let _ = spawn_replacement(&sup.inner);
        assert_eq!(seen.lock().as_slice(), &["org/cool-model".to_string()]);
    }

    // ─── reap_child(None) / install_writer: test-factory no-op reap paths ─────

    #[tokio::test]
    async fn reap_child_none_is_noop() {
        // None proc (test factory) → reap_child is a no-op, no panic.
        reap_child(None).await;
    }

    // ─── proc-reap helpers with a real (trivial) Child ───────────────────────
    //
    // The proc-reap branches in `stop()` / `sysinfo` and the
    // `reap_child` / `reap_old_child` / `install_child` helpers all need an
    // actual `tokio::process::Child` to `wait()`/`start_kill()`. We use a
    // SHORT-LIVED system command (`true` — exits immediately) purely as a Child
    // handle. This is NOT the higgs worker (no `--higgs-worker`, no FFI, no
    // model) — it is a deterministic, instantly-exiting dummy that exercises the
    // OS-process reap path without any real-worker plumbing.

    /// Spawn a trivial, instantly-exiting child purely to drive reap logic.
    fn dummy_child() -> tokio::process::Child {
        Command::new("true")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn `true`")
    }

    /// `reap_child(Some(child))` force-kills and reaps a real child without hang.
    #[tokio::test]
    async fn reap_child_some_kills_and_reaps() {
        reap_child(Some(dummy_child())).await;
    }

    /// `reap_old_child` + `install_child` reap the OLD child then stash the NEW.
    #[tokio::test]
    async fn install_and_reap_old_child() {
        let (sup, _tw, _tr) = make_supervisor();
        // Stash an old child, then reap it.
        *sup.inner.proc.lock().await = Some(dummy_child());
        reap_old_child(&sup.inner).await;
        assert!(sup.inner.proc.lock().await.is_none(), "old child reaped");

        // install_child reaps any old (None here) then stores the new one.
        install_child(&sup.inner, Some(dummy_child())).await;
        assert!(sup.inner.proc.lock().await.is_some(), "new child stashed");
        // Reap the one we just installed so the test leaves no live child.
        reap_old_child(&sup.inner).await;
    }

    /// `stop()` reaps the live child via the proc branch (waits for self-exit).
    #[tokio::test]
    async fn stop_reaps_live_child() {
        let (sup, _tw, _tr) = make_supervisor();
        // Populate the live-worker proc handle with a trivial child so stop()'s
        // proc-reap branch (child.wait()) executes against a real process.
        *sup.inner.proc.lock().await = Some(dummy_child());
        // Give `true` a moment to exit so wait() returns via the clean arm.
        tokio::time::sleep(std::time::Duration::from_millis(30)).await;
        sup.stop().await;
        assert!(
            sup.inner.proc.lock().await.is_none(),
            "child reaped by stop"
        );
        assert!(!sup.inner.running.load(Ordering::Relaxed));
    }

    // install_child's stop()-race close: when `stopped` is set (a stop() won the proc lock
    // first), install_child must NOT install the new worker — it reaps it and returns false
    // so attempt_restart gives up instead of resurrecting a worker after stop. When not
    // stopped, it installs and returns true.
    #[tokio::test]
    async fn install_child_aborts_and_reaps_when_stopped() {
        let (sup, _tw, _tr) = make_supervisor();

        // Not stopped → installs the new child and returns true.
        let installed = install_child(&sup.inner, Some(dummy_child())).await;
        assert!(installed, "installs when not stopped");
        assert!(
            sup.inner.proc.lock().await.is_some(),
            "new child stored on install"
        );

        // Now simulate a concurrent stop() having flipped `stopped` and reaped: clear proc,
        // set stopped. A subsequent restart's install_child must abandon + reap the child it
        // just spawned (return false), leaving proc empty — no resurrection.
        *sup.inner.proc.lock().await = None;
        sup.inner.stopped.store(true, Ordering::Release);
        let installed = install_child(&sup.inner, Some(dummy_child())).await;
        assert!(!installed, "aborts when stopped");
        assert!(
            sup.inner.proc.lock().await.is_none(),
            "aborted restart installs no worker after stop"
        );
    }
}
