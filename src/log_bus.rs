//! The single home for higgs Developer-Log lines.
//!
//! A [`LogBus`] holds a bounded history ring PER source (the snapshot the
//! `logs` control op returns — formerly `GET /api/higgs/logs`) and a live
//! broadcast tap (the `watch_logs` stream — formerly the `/logs/stream` SSE
//! endpoint). Every line — worker child stderr
//! ([`LogSource::Worker`]) AND higgs serve-layer `tracing` events
//! ([`LogSource::Serve`]) — enters through [`LogBus::push`], which appends to
//! that source's ring and sends the tagged line on the broadcast. Separate rings
//! mean a chatty worker (e.g. a model load dumping thousands of llama.cpp
//! metadata lines) never evicts the serve history, so the Developer-Logs console
//! stays populated while the Worker console scrolls.
//!
//! ## Wiring
//!
//! The bus is created by the caller (the embedded server's `main`, or the
//! standalone `higgs` binary), BEFORE the tracing subscriber is built,
//! then shared two ways:
//!
//! ```text
//!   let bus = LogBus::new();
//!   subscriber.with(HiggsLogLayer::new(bus.clone()))   // serve-layer events
//!   Higgs::with_log_bus(config, bus)                   // worker stderr + queries
//! ```
//!
//! This keeps the bus owned by the higgs crate (single home) while letting the
//! caller install the [`HiggsLogLayer`] on its own subscriber — the same way it
//! already owns and wires the subscriber. No process global, no `common`
//! dependency.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use parking_lot::Mutex;
use tokio::sync::broadcast;
use tracing::field::{Field, Visit};
use tracing::{Event, Subscriber};
use tracing_subscriber::layer::Context;
use tracing_subscriber::Layer;

use crate::node::node_id::NodeId;
use crate::node::worker_id::WorkerId;

/// History-ring capacity (lines). Matches the prior `stderr_ring` cap.
const RING_CAP: usize = 2000;

/// One source's bounded history ring: `(seq, text)` entries, oldest first.
type Ring = VecDeque<(u64, String)>;

/// Live-broadcast channel capacity (lines). A slow SSE subscriber that falls
/// this far behind is dropped to `Lagged`; the SSE handler skips the gap and
/// keeps streaming rather than crashing.
const BROADCAST_CAP: usize = 256;

/// Tracing target prefix captured by [`HiggsLogLayer`]. Only events whose
/// target starts with `higgs` (the crate's own module path) are mirrored into
/// the bus — never the host application's unrelated spans.
const HIGGS_TARGET_PREFIX: &str = "higgs";

/// Origin of a Developer-Log line, so the worker's output can be streamed
/// separately from higgs's own control-plane lines (e.g. a Worker-only debug
/// console). The hub's LOCAL child stderr is `Worker`; a REMOTE node's worker
/// stderr (relayed over iroh, P4) is `RemoteWorker { node, worker }` so two
/// nodes' workers stay separable. Stays `Copy` (both ids are `u32` newtypes) so
/// it can ride the broadcast/ring cheaply.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LogSource {
    /// higgs serve-layer / control-plane tracing (the `higgs: …` lines).
    Serve,
    /// The LOCAL model worker process's stderr (llama.cpp / ggml output),
    /// legacy UNKEYED form. Kept as the union FILTER selector (`?source=worker`
    /// matches every local worker) and for push sites that have no worker id
    /// (the transient sysinfo/GPU probes).
    Worker,
    /// One LOCAL worker's stderr, keyed by its [`WorkerId`] so each loaded
    /// model's console can be streamed on its own tab (`?source=worker:<id>`).
    LocalWorker { worker: WorkerId },
    /// A remote node's worker stderr, relayed over iroh and keyed by which node +
    /// which worker on it (`?source=node:<node>:<worker>`).
    RemoteWorker { node: NodeId, worker: WorkerId },
    /// A remote node's OWN DAEMON log (its `LogSource::Serve` lines, streamed over
    /// iroh on demand — `M_NODE_LOGS`), keyed by node (`?source=node:<node>`).
    /// Deliberately distinct from `RemoteWorker`: the daemon log is the higgs
    /// control plane, never model/worker output.
    RemoteNode { node: NodeId },
}

impl LogSource {
    /// Parse a `?source=` query value; `None` (absent/unknown) means all sources.
    /// Local per-worker form: `worker:<worker-id>` (e.g. `worker:3`). Remote
    /// selector form: `node:<node-id>:<worker-id>` (e.g. `node:1:2`).
    pub fn parse(s: &str) -> Option<LogSource> {
        match s {
            "serve" => Some(LogSource::Serve),
            "worker" => Some(LogSource::Worker),
            _ => {
                if let Some(w) = s.strip_prefix("worker:") {
                    return Some(LogSource::LocalWorker {
                        worker: WorkerId(w.parse().ok()?),
                    });
                }
                let rest = s.strip_prefix("node:")?;
                // `node:<id>` = the node's own daemon log; `node:<id>:<worker>` =
                // one of its workers.
                let Some((n, w)) = rest.split_once(':') else {
                    return Some(LogSource::RemoteNode {
                        node: NodeId(rest.parse().ok()?),
                    });
                };
                Some(LogSource::RemoteWorker {
                    node: NodeId(n.parse().ok()?),
                    worker: WorkerId(w.parse().ok()?),
                })
            }
        }
    }

    /// Whether a line tagged `self` should pass a `?source=` filter of `filter`.
    /// Exact match, plus one union rule: filter [`Worker`](LogSource::Worker)
    /// matches every [`LocalWorker`](LogSource::LocalWorker) line, so the legacy
    /// "worker" console shows all local workers' output interleaved.
    pub fn matches_filter(self, filter: LogSource) -> bool {
        self == filter
            || (filter == LogSource::Worker && matches!(self, LogSource::LocalWorker { .. }))
    }
}

/// A buffered Developer-Log line tagged with its [`LogSource`]. The history ring
/// and the live broadcast both carry these so a subscriber can filter by source.
#[derive(Clone, Debug)]
pub struct LogLine {
    /// Where the line came from.
    pub source: LogSource,
    /// The formatted line text (what the UI renders).
    pub text: String,
}

/// Append `(seq, text)` to a history ring, evicting the oldest at `RING_CAP`.
fn push_ring(ring: &mut Ring, seq: u64, text: String) {
    while ring.len() >= RING_CAP {
        ring.pop_front();
    }
    ring.push_back((seq, text));
}

/// Up to `n` most-recent line texts from one history ring (oldest first).
fn last_n(ring: &Ring, n: usize) -> Vec<String> {
    let skip = ring.len().saturating_sub(n);
    ring.iter().skip(skip).map(|(_, t)| t.clone()).collect()
}

/// The single home for higgs Developer-Log lines: a bounded history ring PER
/// source plus a live broadcast tap. Every log line enters via [`push`](Self::push).
///
/// Per-source rings (not one shared ring) so a chatty worker — e.g. a model
/// load dumping thousands of llama.cpp metadata lines — can't evict the
/// serve-layer history (or vice-versa); each `?source=` console keeps its own
/// independent `RING_CAP` of history.
///
/// Cloneable handle semantics are provided by wrapping in `Arc` at the call
/// site (`Arc<LogBus>`); the broadcast `Sender` and the `Mutex<VecDeque>`s are
/// the shared state.
#[derive(Debug)]
pub struct LogBus {
    /// Serve-layer history ring (`(seq, text)`, oldest first). `parking_lot`
    /// mutex held only for push/read — never across `.await`.
    serve: Mutex<Ring>,
    /// Worker-stderr history ring (`(seq, text)`, oldest first) — the legacy
    /// UNKEYED ring, still fed by push sites without a worker id (transient probes).
    worker: Mutex<Ring>,
    /// Per-LOCAL-worker stderr history rings, created on a worker's first line and
    /// reclaimed by [`evict_local`](Self::evict_local) on unload/idle-reap so a dead
    /// worker's ring doesn't leak. Each ring is independently `RING_CAP`-bounded.
    local: Mutex<HashMap<WorkerId, Ring>>,
    /// Per-(node,worker) remote-stderr history rings, created on the first relayed
    /// line and reclaimed by [`evict_remote`](Self::evict_remote) on unload/kill/retire
    /// so a dead worker's ring doesn't leak. Each ring is independently `RING_CAP`-bounded.
    remote: Mutex<HashMap<(NodeId, WorkerId), Ring>>,
    /// Per-node DAEMON-log history rings (`M_NODE_LOGS` lines), created on the
    /// first streamed line and reclaimed by [`evict_node`](Self::evict_node) on
    /// retire. Each ring is independently `RING_CAP`-bounded.
    remote_node: Mutex<HashMap<NodeId, Ring>>,
    /// Monotonic line counter — stamps every push so an unfiltered (`None`)
    /// snapshot can re-interleave the two rings in arrival order.
    seq: std::sync::atomic::AtomicU64,
    /// Live fan-out of every pushed line to current SSE subscribers.
    tx: broadcast::Sender<LogLine>,
    /// SERVE-ONLY live fan-out — the daemon-log stream (`M_NODE_LOGS` relay)
    /// subscribes HERE, not to the shared `tx`: on the shared channel a worker
    /// token flood advances every subscriber's cursor, so a Serve-filtered
    /// reader would lag (and DROP real daemon lines) because of traffic it
    /// never wanted. A dedicated channel makes daemon-log lag mean daemon-log
    /// volume, nothing else.
    serve_tx: broadcast::Sender<String>,
    /// DEBUG toggle, OFF by default. When `true`, [`HiggsLogLayer`] also emits
    /// non-message/non-`error` structured fields — INCLUDING prompt content —
    /// so logs can be debugged un-redacted. The flag lives here (not on `Higgs`)
    /// because the layer holds only the bus. Default off = the redaction policy.
    show_fields: std::sync::atomic::AtomicBool,
    /// "Verbose Logging" toggle, OFF by default. The single home for the user
    /// setting that means *show verbose logs*: read by the serve layer (extra
    /// per-request completion line, via `Higgs::verbose`) AND by the worker
    /// stderr drain — when `false` it drops llama.cpp's per-load metadata/tensor
    /// dump (keeping warnings/errors); when `true` it streams every worker line.
    /// Lives here so the drain (which holds only the bus) can read it.
    verbose: std::sync::atomic::AtomicBool,
    /// "Log Incoming Tokens" toggle, OFF by default. Read by the serve layer
    /// (`serve/v1.rs`) to decide whether to emit the redact-by-default
    /// incoming-prompt line. Lives on the bus (not on `Higgs`) so a node-side
    /// dispatcher — which only holds a `LogBus` — can flip it via
    /// [`LogBus::global`] on behalf of a remote worker, mirroring `show_fields`.
    log_incoming_tokens: std::sync::atomic::AtomicBool,
}

/// The PROCESS-GLOBAL bus — the one bound to the global tracing subscriber in
/// `bin/higgs.rs`. Set once at startup; read by the node daemon so its
/// `NodeRuntime` shares the SAME bus the daemon's own tracing lands in (the
/// `M_NODE_LOGS` stream serves those Serve-ring lines). A bus is naturally
/// process-global here because the tracing subscriber it feeds is too.
static GLOBAL_BUS: std::sync::OnceLock<Arc<LogBus>> = std::sync::OnceLock::new();

impl LogBus {
    /// Register the process-global bus (first call wins; later calls are no-ops —
    /// tests that build private buses never touch this).
    pub fn install_global(bus: Arc<LogBus>) {
        let _ = GLOBAL_BUS.set(bus);
    }

    /// The process-global bus, if `install_global` ran (the real binary); `None`
    /// under tests/embedders that manage their own buses.
    pub fn global() -> Option<Arc<LogBus>> {
        GLOBAL_BUS.get().cloned()
    }

    /// Create an empty bus with the default ring and broadcast capacities.
    pub fn new() -> Self {
        let (tx, _) = broadcast::channel(BROADCAST_CAP);
        let (serve_tx, _) = broadcast::channel(BROADCAST_CAP);
        Self {
            serve: Mutex::new(VecDeque::with_capacity(RING_CAP)),
            worker: Mutex::new(VecDeque::with_capacity(RING_CAP)),
            local: Mutex::new(HashMap::new()),
            remote: Mutex::new(HashMap::new()),
            remote_node: Mutex::new(HashMap::new()),
            seq: std::sync::atomic::AtomicU64::new(0),
            tx,
            serve_tx,
            show_fields: std::sync::atomic::AtomicBool::new(false),
            // Default ON: NL-V's operator-visible design is "always verbose,
            // toggle off during quiet periods, on again to debug". The tracing
            // subscriber's per-layer filter (`log_filter`) still admits
            // `higgs::*` down to DEBUG only, so TRACE lines still require the
            // caller to bump their subscriber filter — this flag governs the
            // Serve ring's promotion of DEBUG events into the follow stream.
            verbose: std::sync::atomic::AtomicBool::new(true),
            log_incoming_tokens: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Whether the layer emits non-message structured fields (incl. prompt
    /// content) — the un-redacted DEBUG mode. Off by default.
    pub fn show_fields(&self) -> bool {
        self.show_fields.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Toggle the un-redacted DEBUG mode at runtime.
    pub fn set_show_fields(&self, v: bool) {
        self.show_fields
            .store(v, std::sync::atomic::Ordering::Relaxed);
    }

    /// Whether "Verbose Logging" is on — drives the serve-layer extra line and
    /// the worker drain's keep-everything vs drop-the-dump behavior. Off by default.
    pub fn verbose(&self) -> bool {
        self.verbose.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Toggle "Verbose Logging" at runtime.
    pub fn set_verbose(&self, v: bool) {
        self.verbose.store(v, std::sync::atomic::Ordering::Relaxed);
    }

    /// Whether "Log Incoming Tokens" is on — the serve layer's prompt-content
    /// log line. Off by default (redact policy).
    pub fn log_incoming_tokens(&self) -> bool {
        self.log_incoming_tokens
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Toggle "Log Incoming Tokens" at runtime.
    pub fn set_log_incoming_tokens(&self, v: bool) {
        self.log_incoming_tokens
            .store(v, std::sync::atomic::Ordering::Relaxed);
    }

    /// Append `text` to its `source` history ring (evicting that ring's oldest
    /// when full) and fan it out to live subscribers. A send with no subscribers
    /// is a no-op — the line still lands in the ring for replay.
    // TODO(C1): `text` is cloned twice (ring String + broadcast LogLine.text).
    // Switching the ring tuple + LogLine.text to `Arc<str>` does NOT remove the
    // alloc: the SSE wire boundary (serve/control.rs) and snapshot()/last_n()
    // both hand out owned `String`s, so an Arc would just reintroduce a
    // `.to_string()` on the live path. Skipped — no net win without reworking
    // the String-typed log stream channel.
    pub fn push(&self, source: LogSource, text: String) {
        // Assign `seq` UNDER the destination ring's lock, not before it: the seq stamp and
        // the `push_back` must be atomic per ring. Otherwise two concurrent pushes to the
        // same source could stamp seq=N / seq=N+1 but insert in the opposite order (the
        // seq=N thread stalls between the fetch_add and the lock), leaving that ring's seqs
        // out of order vs. insertion — which `snapshot(None)` (sorts by seq) and
        // `snapshot(source)` (insertion order) would then disagree on.
        match source {
            LogSource::Serve => {
                let mut ring = self.serve.lock();
                push_ring(&mut ring, self.next_seq(), text.clone());
            }
            LogSource::Worker => {
                let mut ring = self.worker.lock();
                push_ring(&mut ring, self.next_seq(), text.clone());
            }
            LogSource::LocalWorker { worker } => {
                let mut local = self.local.lock();
                let seq = self.next_seq();
                let ring = local.entry(worker).or_default();
                push_ring(ring, seq, text.clone());
            }
            LogSource::RemoteWorker { node, worker } => {
                let mut remote = self.remote.lock();
                let seq = self.next_seq();
                let ring = remote.entry((node, worker)).or_default();
                push_ring(ring, seq, text.clone());
            }
            LogSource::RemoteNode { node } => {
                let mut rings = self.remote_node.lock();
                let seq = self.next_seq();
                let ring = rings.entry(node).or_default();
                push_ring(ring, seq, text.clone());
            }
        }
        // Err means zero subscribers — fine; the ring already has the line.
        if source == LogSource::Serve {
            let _ = self.serve_tx.send(text.clone());
        }
        let _ = self.tx.send(LogLine { source, text });
    }

    /// Next monotonic sequence stamp. Call WHILE holding the destination ring's lock so the
    /// stamp and the ring insertion are atomic per ring (see [`push`](Self::push)).
    fn next_seq(&self) -> u64 {
        self.seq.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    }

    /// Subscribe to SERVE lines only (the daemon log) — isolated from worker/
    /// model traffic so lag here always means daemon-log volume (see `serve_tx`).
    pub fn subscribe_serve(&self) -> broadcast::Receiver<String> {
        self.serve_tx.subscribe()
    }

    /// Reclaim a LOCAL worker's history ring (called when the worker leaves the node
    /// registry: unload, kill, or idle-reap), so a dead worker's lines don't linger
    /// forever. A no-op if it never logged. Called AFTER the worker's stop completes
    /// (its stderr pipe is closed), so only a line already in flight in the async
    /// relay chain can recreate a tiny ring — bounded at a few lines, accepted
    /// rather than serializing eviction against the relay task.
    pub fn evict_local(&self, worker: WorkerId) {
        self.local.lock().remove(&worker);
    }

    /// Reclaim a remote worker's history ring (called on remote unload/kill), so a finished
    /// worker's lines don't linger forever. A no-op if it never logged.
    pub fn evict_remote(&self, node: NodeId, worker: WorkerId) {
        self.remote.lock().remove(&(node, worker));
    }

    /// Reclaim ALL of a node's remote rings (called on node retire) — including those of
    /// workers no longer on a current route (e.g. a worker displaced by a reload), which a
    /// per-worker eviction would miss.
    pub fn evict_node(&self, node: NodeId) {
        self.remote.lock().retain(|(n, _), _| *n != node);
        self.remote_node.lock().remove(&node);
    }

    /// Up to `n` most-recent line texts (oldest first), restricted to one
    /// [`LogSource`] or, with `None`, the two rings re-interleaved by arrival.
    pub fn snapshot(&self, n: usize, filter: Option<LogSource>) -> Vec<String> {
        /// Merge several rings' `(seq, text)` entries by seq and keep the last `n`.
        fn merged<'a, I: Iterator<Item = &'a (u64, String)>>(iter: I, n: usize) -> Vec<String> {
            let mut all: Vec<(u64, &str)> = iter.map(|(q, t)| (*q, t.as_str())).collect();
            all.sort_by_key(|(q, _)| *q);
            let skip = all.len().saturating_sub(n);
            all.into_iter()
                .skip(skip)
                .map(|(_, t)| t.to_owned())
                .collect()
        }
        match filter {
            Some(LogSource::Serve) => last_n(&self.serve.lock(), n),
            // `worker` is the UNION selector: the legacy unkeyed ring plus every
            // local per-worker ring, re-interleaved by arrival (seq).
            Some(LogSource::Worker) => {
                let worker = self.worker.lock();
                let local = self.local.lock();
                merged(
                    worker.iter().chain(local.values().flat_map(|r| r.iter())),
                    n,
                )
            }
            Some(LogSource::LocalWorker { worker }) => self
                .local
                .lock()
                .get(&worker)
                .map(|ring| last_n(ring, n))
                .unwrap_or_default(),
            Some(LogSource::RemoteWorker { node, worker }) => self
                .remote
                .lock()
                .get(&(node, worker))
                .map(|ring| last_n(ring, n))
                .unwrap_or_default(),
            Some(LogSource::RemoteNode { node }) => self
                .remote_node
                .lock()
                .get(&node)
                .map(|ring| last_n(ring, n))
                .unwrap_or_default(),
            None => {
                // Always lock in a consistent order (serve, worker, local, remote,
                // remote_node) — no deadlock.
                let serve = self.serve.lock();
                let worker = self.worker.lock();
                let local = self.local.lock();
                let remote = self.remote.lock();
                let remote_node = self.remote_node.lock();
                merged(
                    serve
                        .iter()
                        .chain(worker.iter())
                        .chain(local.values().flat_map(|r| r.iter()))
                        .chain(remote.values().flat_map(|r| r.iter()))
                        .chain(remote_node.values().flat_map(|r| r.iter())),
                    n,
                )
            }
        }
    }

    /// Subscribe to live [`LogLine`]s pushed AFTER this call. Pair with
    /// [`snapshot`](Self::snapshot) for replay-then-live delivery; filter the
    /// received lines by [`LogLine::source`].
    pub fn subscribe(&self) -> broadcast::Receiver<LogLine> {
        self.tx.subscribe()
    }
}

impl Default for LogBus {
    fn default() -> Self {
        Self::new()
    }
}

/// A [`tracing`] layer that mirrors higgs serve-layer events into a [`LogBus`].
///
/// Captures only events whose target starts with `higgs` and formats each as
/// `TIMESTAMP [LEVEL] message` (e.g.
/// `2026-06-16 21:12:27 [INFO] higgs: GET /v1/models`), matching the worker
/// stderr line style and LM Studio's Developer-Log format. The formatted line
/// is pushed into the same ring+broadcast that carries worker stderr, so the
/// Developer Logs show request activity live — not just worker output.
///
/// Install it on the caller's subscriber registry alongside the existing
/// fmt/sqlite layers; it never replaces them (events still reach stdout/db).
pub struct HiggsLogLayer<B> {
    bus: B,
}

impl<B> HiggsLogLayer<B> {
    /// Build a layer feeding `bus` (typically `Arc<LogBus>`). `B` only needs to
    /// deref to a [`LogBus`].
    pub fn new(bus: B) -> Self {
        Self { bus }
    }
}

/// Collects the `message` and `error` fields of a tracing event. The `message`
/// is the human line; the `error` field — used by `warn!(error = %err, …)` —
/// carries typed diagnostic detail (e.g. a llama.cpp model-load failure reason)
/// and is appended so failures are debuggable in the Developer Logs. NO other
/// structured field is captured: critically, prompt CONTENT is never serialized
/// (redaction policy). `error` is exempt because it is always a typed
/// `[HGxxx] …` diagnostic, never user/prompt content.
#[derive(Default)]
struct MessageVisitor {
    message: String,
    error: Option<String>,
    /// Every OTHER field (`key`, value). Collected always, but emitted ONLY when
    /// the bus is in un-redacted DEBUG mode — these may contain prompt content.
    extra: Vec<(String, String)>,
}

impl Visit for MessageVisitor {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        match field.name() {
            "message" => self.message = format!("{value:?}"),
            "error" => self.error = Some(format!("{value:?}")),
            name => self.extra.push((name.to_owned(), format!("{value:?}"))),
        }
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        match field.name() {
            "message" => self.message = value.to_owned(),
            "error" => self.error = Some(value.to_owned()),
            name => self.extra.push((name.to_owned(), value.to_owned())),
        }
    }
}

/// `HH:MM:SS`-precision wall-clock timestamp for a log line, `YYYY-MM-DD
/// HH:MM:SS` (UTC). Hand-formatted from the Unix epoch so the crate needs no
/// `chrono`/`time` dependency — the worker stderr lines and LM Studio both use
/// a coarse second-precision local-style stamp, and the bus only needs ordering
/// and human readability, not sub-second precision or a timezone DB.
fn timestamp() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs();
    // Civil-date conversion (days since 1970 → Y-M-D), Howard Hinnant's algo.
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    let (hh, mm, ss) = (rem / 3600, (rem % 3600) / 60, rem % 60);
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02} {hh:02}:{mm:02}:{ss:02}")
}

impl<S, B> Layer<S> for HiggsLogLayer<B>
where
    S: Subscriber,
    B: std::ops::Deref<Target = LogBus> + 'static,
{
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        let meta = event.metadata();
        let target = meta.target();
        if !target.starts_with(HIGGS_TARGET_PREFIX) {
            return;
        }
        // "Verbose Logging" level gate for the Developer-Logs (serve) console:
        // off → INFO+ only; on → also DEBUG/TRACE. (DEBUG/TRACE only reach here
        // when the subscriber's per-layer filter admits them — see `log_filter`.)
        if !self.bus.verbose() && *meta.level() > tracing::Level::INFO {
            return;
        }
        let mut visitor = MessageVisitor::default();
        event.record(&mut visitor);
        if visitor.message.is_empty() {
            return;
        }
        // NL-V section badge: the second segment of the event target (after
        // the `higgs::` prefix) — e.g. `higgs::download::pull_stream` →
        // `[download]`, bare `higgs` → `[higgs]`. Rendered inline at the
        // START of every line so a reader can visually group / filter
        // client-side WITHOUT the daemon carrying a filter/routing concept.
        let section = target
            .strip_prefix(HIGGS_TARGET_PREFIX)
            .and_then(|rest| rest.strip_prefix("::"))
            .map(|rest| rest.split("::").next().unwrap_or(rest))
            .filter(|s| !s.is_empty())
            .unwrap_or("higgs");
        let mut line = format!(
            "[{section}] {} [{}] {}",
            timestamp(),
            meta.level(),
            visitor.message
        );
        // The typed `error` diagnostic is ALWAYS shown — it never carries prompt
        // content and is the whole point of debuggable failures.
        if let Some(err) = &visitor.error {
            line.push_str(" — ");
            line.push_str(err);
        }
        // DEBUG mode only: append every other field, which MAY include prompt
        // content. Redacted (dropped) by default.
        if self.bus.show_fields() {
            for (k, v) in &visitor.extra {
                line.push_str(&format!(" {k}={v}"));
            }
        }
        // Serve-layer tracing — the higgs main-server view (incl. its worker
        // interactions); the worker's own stderr is tagged Worker at its drain.
        self.bus.push(LogSource::Serve, line);
    }
}

/// Per-layer filter to install on the [`HiggsLogLayer`] so it admits
/// `higgs`-target events down to DEBUG — independent of the global subscriber
/// filter — letting "Verbose Logging" surface higgs DEBUG in the Developer-Logs
/// console (the layer's own gate then drops DEBUG/TRACE unless verbose). Scoped
/// to the `higgs` target only, so no other target's DEBUG is generated. Install
/// as `HiggsLogLayer::new(bus).with_filter(log_filter())`.
pub fn log_filter() -> tracing_subscriber::filter::Targets {
    tracing_subscriber::filter::Targets::new().with_target(
        HIGGS_TARGET_PREFIX,
        tracing::level_filters::LevelFilter::DEBUG,
    )
}

#[cfg(test)]
#[path = "log_bus_tests.rs"]
mod tests;
