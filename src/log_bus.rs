//! The single home for higgs Developer-Log lines.
//!
//! A [`LogBus`] holds a bounded history ring PER source (the snapshot source for
//! `GET /api/higgs/logs`) and a live broadcast tap (the source for the
//! `GET /api/higgs/logs/stream` SSE endpoint). Every line — worker child stderr
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
    /// The LOCAL model worker process's stderr (llama.cpp / ggml output).
    Worker,
    /// A remote node's worker stderr, relayed over iroh and keyed by which node +
    /// which worker on it (`?source=node:<node>:<worker>`).
    RemoteWorker { node: NodeId, worker: WorkerId },
}

impl LogSource {
    /// Parse a `?source=` query value; `None` (absent/unknown) means all sources.
    /// Remote selector form: `node:<node-id>:<worker-id>` (e.g. `node:1:2`).
    pub fn parse(s: &str) -> Option<LogSource> {
        match s {
            "serve" => Some(LogSource::Serve),
            "worker" => Some(LogSource::Worker),
            _ => {
                let rest = s.strip_prefix("node:")?;
                let (n, w) = rest.split_once(':')?;
                Some(LogSource::RemoteWorker {
                    node: NodeId(n.parse().ok()?),
                    worker: WorkerId(w.parse().ok()?),
                })
            }
        }
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
    /// Worker-stderr history ring (`(seq, text)`, oldest first).
    worker: Mutex<Ring>,
    /// Per-(node,worker) remote-stderr history rings, created on the first relayed
    /// line and reclaimed by [`evict_remote`](Self::evict_remote) on unload/kill/retire
    /// so a dead worker's ring doesn't leak. Each ring is independently `RING_CAP`-bounded.
    remote: Mutex<HashMap<(NodeId, WorkerId), Ring>>,
    /// Monotonic line counter — stamps every push so an unfiltered (`None`)
    /// snapshot can re-interleave the two rings in arrival order.
    seq: std::sync::atomic::AtomicU64,
    /// Live fan-out of every pushed line to current SSE subscribers.
    tx: broadcast::Sender<LogLine>,
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
}

impl LogBus {
    /// Create an empty bus with the default ring and broadcast capacities.
    pub fn new() -> Self {
        let (tx, _) = broadcast::channel(BROADCAST_CAP);
        Self {
            serve: Mutex::new(VecDeque::with_capacity(RING_CAP)),
            worker: Mutex::new(VecDeque::with_capacity(RING_CAP)),
            remote: Mutex::new(HashMap::new()),
            seq: std::sync::atomic::AtomicU64::new(0),
            tx,
            show_fields: std::sync::atomic::AtomicBool::new(false),
            verbose: std::sync::atomic::AtomicBool::new(false),
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
            LogSource::RemoteWorker { node, worker } => {
                let mut remote = self.remote.lock();
                let seq = self.next_seq();
                let ring = remote.entry((node, worker)).or_default();
                push_ring(ring, seq, text.clone());
            }
        }
        // Err means zero subscribers — fine; the ring already has the line.
        let _ = self.tx.send(LogLine { source, text });
    }

    /// Next monotonic sequence stamp. Call WHILE holding the destination ring's lock so the
    /// stamp and the ring insertion are atomic per ring (see [`push`](Self::push)).
    fn next_seq(&self) -> u64 {
        self.seq.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
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
    }

    /// Up to `n` most-recent line texts (oldest first), restricted to one
    /// [`LogSource`] or, with `None`, the two rings re-interleaved by arrival.
    pub fn snapshot(&self, n: usize, filter: Option<LogSource>) -> Vec<String> {
        match filter {
            Some(LogSource::Serve) => last_n(&self.serve.lock(), n),
            Some(LogSource::Worker) => last_n(&self.worker.lock(), n),
            Some(LogSource::RemoteWorker { node, worker }) => self
                .remote
                .lock()
                .get(&(node, worker))
                .map(|ring| last_n(ring, n))
                .unwrap_or_default(),
            None => {
                // Always lock in a consistent order (serve, worker, remote) — no deadlock.
                let serve = self.serve.lock();
                let worker = self.worker.lock();
                let remote = self.remote.lock();
                let mut all: Vec<(u64, &str)> = serve
                    .iter()
                    .chain(worker.iter())
                    .chain(remote.values().flat_map(|r| r.iter()))
                    .map(|(q, t)| (*q, t.as_str()))
                    .collect();
                all.sort_by_key(|(q, _)| *q);
                let skip = all.len().saturating_sub(n);
                all.into_iter()
                    .skip(skip)
                    .map(|(_, t)| t.to_owned())
                    .collect()
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
        if !meta.target().starts_with(HIGGS_TARGET_PREFIX) {
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
        let mut line = format!("{} [{}] {}", timestamp(), meta.level(), visitor.message);
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
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn push_writes_ring_and_broadcast() {
        let bus = Arc::new(LogBus::new());
        let mut rx = bus.subscribe();
        bus.push(LogSource::Serve, "line-1".to_owned());
        // Ring captured it (snapshot history).
        assert_eq!(bus.snapshot(10, None), vec!["line-1".to_owned()]);
        // Broadcast delivered the same line to the live subscriber.
        assert_eq!(rx.try_recv().unwrap().text, "line-1");
    }

    #[test]
    fn snapshot_returns_last_n_oldest_first() {
        let bus = LogBus::new();
        for i in 0..5 {
            bus.push(LogSource::Serve, format!("l{i}"));
        }
        assert_eq!(
            bus.snapshot(2, None),
            vec!["l3".to_owned(), "l4".to_owned()]
        );
        assert_eq!(bus.snapshot(0, None), Vec::<String>::new());
    }

    #[test]
    fn snapshot_filters_by_source() {
        let bus = LogBus::new();
        bus.push(LogSource::Serve, "serve-1".to_owned());
        bus.push(LogSource::Worker, "worker-1".to_owned());
        bus.push(LogSource::Serve, "serve-2".to_owned());
        assert_eq!(bus.snapshot(10, None).len(), 3, "no filter = all sources");
        assert_eq!(
            bus.snapshot(10, Some(LogSource::Worker)),
            vec!["worker-1".to_owned()]
        );
        assert_eq!(
            bus.snapshot(10, Some(LogSource::Serve)),
            vec!["serve-1".to_owned(), "serve-2".to_owned()]
        );
    }

    #[test]
    fn worker_flood_does_not_evict_serve_history() {
        // The bug: a single shared ring let worker model-load spam evict every
        // serve line, leaving the Developer Logs console empty. Per-source rings
        // keep each console's history independent.
        let bus = LogBus::new();
        bus.push(LogSource::Serve, "higgs: GET /v1/models".to_owned());
        bus.push(LogSource::Serve, "higgs: loading model".to_owned());
        for i in 0..(RING_CAP + 500) {
            bus.push(LogSource::Worker, format!("ggml line {i}"));
        }
        // Serve history survives the worker flood intact.
        assert_eq!(
            bus.snapshot(10, Some(LogSource::Serve)),
            vec![
                "higgs: GET /v1/models".to_owned(),
                "higgs: loading model".to_owned()
            ],
            "serve lines must not be evicted by worker output"
        );
        // The worker ring is still bounded (capped, not unbounded).
        assert_eq!(
            bus.snapshot(usize::MAX, Some(LogSource::Worker)).len(),
            RING_CAP
        );
    }

    #[test]
    fn concurrent_pushes_keep_ring_seq_monotonic() {
        // Per-ring invariant: a ring's seq stamps must increase with insertion order, even
        // under heavy concurrent pushes to ONE source. `snapshot(None)` sorts by seq while
        // `snapshot(source)` returns insertion order, so an out-of-order seq makes the two
        // views disagree on near-simultaneous lines. Guards the "stamp seq UNDER the ring
        // lock" fix; the prior fetch_add-before-lock could stamp seq=N then insert it after
        // seq=N+1 when the N thread stalled between the fetch_add and the lock.
        const THREADS: u64 = 8;
        // THREADS * PER == RING_CAP → the whole window is retained (no eviction to muddy it).
        const PER: u64 = (RING_CAP as u64) / THREADS;
        let bus = std::sync::Arc::new(LogBus::new());
        let handles: Vec<_> = (0..THREADS)
            .map(|t| {
                let bus = std::sync::Arc::clone(&bus);
                std::thread::spawn(move || {
                    for i in 0..PER {
                        bus.push(LogSource::Serve, format!("t{t}-{i}"));
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        let seqs: Vec<u64> = bus.serve.lock().iter().map(|(q, _)| *q).collect();
        assert_eq!(seqs.len() as u64, THREADS * PER, "no eviction at RING_CAP");
        assert!(
            seqs.windows(2).all(|w| w[0] < w[1]),
            "ring seqs must strictly increase with insertion order — an inversion makes \
             snapshot(None) reorder concurrently-pushed lines vs the per-source view",
        );
    }

    #[test]
    fn ring_evicts_oldest_at_capacity() {
        let bus = LogBus::new();
        for i in 0..(RING_CAP + 5) {
            bus.push(LogSource::Worker, format!("l{i}"));
        }
        let snap = bus.snapshot(RING_CAP + 100, None);
        assert_eq!(snap.len(), RING_CAP);
        // Oldest 5 were evicted; first surviving line is l5.
        assert_eq!(snap[0], "l5");
    }

    #[test]
    fn subscriber_only_sees_lines_after_subscribe() {
        let bus = LogBus::new();
        bus.push(LogSource::Serve, "before".to_owned());
        let mut rx = bus.subscribe();
        bus.push(LogSource::Serve, "after".to_owned());
        // "before" is NOT delivered live (it predates the subscription); it is
        // only available via the snapshot replay.
        assert_eq!(rx.try_recv().unwrap().text, "after");
        assert!(rx.try_recv().is_err());
        assert_eq!(
            bus.snapshot(10, None),
            vec!["before".to_owned(), "after".to_owned()]
        );
    }

    #[tokio::test]
    async fn lagged_subscriber_reports_lagged_then_recovers() {
        let bus = LogBus::new();
        let mut rx = bus.subscribe();
        // Overflow the broadcast channel without draining — the slow subscriber
        // falls behind and the next recv reports Lagged rather than crashing.
        for i in 0..(BROADCAST_CAP + 10) {
            bus.push(LogSource::Worker, format!("l{i}"));
        }
        match rx.recv().await {
            Err(broadcast::error::RecvError::Lagged(skipped)) => {
                assert!(skipped > 0, "expected a positive lagged count");
            }
            other => panic!("expected Lagged, got {other:?}"),
        }
        // After Lagged the receiver recovers and yields the still-buffered tail.
        let next = rx.recv().await.expect("recovers after lag");
        assert!(next.text.starts_with('l'));
    }

    #[test]
    fn layer_captures_error_field_and_redacts_other_fields() {
        use tracing_subscriber::layer::SubscriberExt;
        let bus = Arc::new(LogBus::new());
        let subscriber = tracing_subscriber::registry().with(HiggsLogLayer::new(bus.clone()));
        tracing::subscriber::with_default(subscriber, || {
            // A higgs-target load failure: the typed reason rides the `error` field.
            tracing::warn!(
                target: "higgs::test",
                error = "[HG004] engine failed to load m.gguf: out of memory",
                prompt = "secret user prompt content",
                "higgs: load failed"
            );
            // Wrong target — must be dropped entirely.
            tracing::info!(target: "not_higgs", "should be ignored");
        });
        let snap = bus.snapshot(10, None);
        assert_eq!(snap.len(), 1, "only the higgs-target event is captured");
        let line = &snap[0];
        assert!(
            line.contains("higgs: load failed"),
            "message present: {line}"
        );
        assert!(
            line.contains("[HG004] engine failed to load m.gguf: out of memory"),
            "error field appended so failures are debuggable: {line}"
        );
        assert!(
            !line.contains("secret user prompt content"),
            "non-error fields stay redacted by default (no prompt content): {line}"
        );
    }

    #[test]
    fn layer_show_fields_unredacts_for_debug() {
        use tracing_subscriber::layer::SubscriberExt;
        let bus = Arc::new(LogBus::new());
        bus.set_show_fields(true); // DEBUG mode on
        let subscriber = tracing_subscriber::registry().with(HiggsLogLayer::new(bus.clone()));
        tracing::subscriber::with_default(subscriber, || {
            tracing::info!(
                target: "higgs::test",
                prompt = "the actual user prompt",
                "higgs: chat"
            );
        });
        let snap = bus.snapshot(10, None);
        assert_eq!(snap.len(), 1);
        assert!(
            snap[0].contains("prompt=the actual user prompt"),
            "show mode appends other fields incl. prompt content: {}",
            snap[0]
        );
    }

    #[test]
    fn parses_remote_node_source_selector() {
        assert_eq!(
            LogSource::parse("node:1:2"),
            Some(LogSource::RemoteWorker {
                node: NodeId(1),
                worker: WorkerId(2)
            })
        );
        // Malformed selectors fall back to "all sources" (None).
        assert_eq!(LogSource::parse("node:1"), None);
        assert_eq!(LogSource::parse("node:x:2"), None);
        assert_eq!(LogSource::parse("bogus"), None);
    }

    #[test]
    fn remote_worker_lines_are_keyed_and_separable() {
        let bus = LogBus::new();
        let a = LogSource::RemoteWorker {
            node: NodeId(1),
            worker: WorkerId(1),
        };
        let b = LogSource::RemoteWorker {
            node: NodeId(2),
            worker: WorkerId(1),
        };
        bus.push(a, "a-line".to_owned());
        bus.push(b, "b-line".to_owned());
        bus.push(a, "a-line-2".to_owned());
        assert_eq!(
            bus.snapshot(10, Some(a)),
            vec!["a-line".to_owned(), "a-line-2".to_owned()]
        );
        assert_eq!(bus.snapshot(10, Some(b)), vec!["b-line".to_owned()]);
        // Unfiltered interleaves both remote workers in arrival order.
        assert_eq!(
            bus.snapshot(10, None),
            vec![
                "a-line".to_owned(),
                "b-line".to_owned(),
                "a-line-2".to_owned()
            ]
        );
    }

    #[test]
    fn evict_remote_reclaims_a_dead_workers_ring() {
        let bus = LogBus::new();
        let a = LogSource::RemoteWorker {
            node: NodeId(1),
            worker: WorkerId(7),
        };
        bus.push(a, "x".to_owned());
        assert_eq!(bus.snapshot(10, Some(a)).len(), 1);
        bus.evict_remote(NodeId(1), WorkerId(7));
        assert!(bus.snapshot(10, Some(a)).is_empty(), "ring reclaimed");
        // Evicting an unknown worker is a harmless no-op.
        bus.evict_remote(NodeId(9), WorkerId(9));
    }

    #[test]
    fn evict_node_reclaims_all_of_a_nodes_rings() {
        let bus = LogBus::new();
        let w1 = LogSource::RemoteWorker {
            node: NodeId(1),
            worker: WorkerId(1),
        };
        let w2 = LogSource::RemoteWorker {
            node: NodeId(1),
            worker: WorkerId(2),
        };
        let other = LogSource::RemoteWorker {
            node: NodeId(2),
            worker: WorkerId(1),
        };
        bus.push(w1, "a".to_owned());
        bus.push(w2, "b".to_owned()); // a displaced worker's ring, off any current route
        bus.push(other, "c".to_owned());
        bus.evict_node(NodeId(1));
        assert!(
            bus.snapshot(10, Some(w1)).is_empty(),
            "node 1 worker 1 reclaimed"
        );
        assert!(
            bus.snapshot(10, Some(w2)).is_empty(),
            "node 1 worker 2 reclaimed too"
        );
        assert_eq!(
            bus.snapshot(10, Some(other)),
            vec!["c".to_owned()],
            "other node untouched"
        );
    }

    #[test]
    fn remote_ring_is_capacity_bounded() {
        let bus = LogBus::new();
        let a = LogSource::RemoteWorker {
            node: NodeId(1),
            worker: WorkerId(1),
        };
        for i in 0..(RING_CAP + 10) {
            bus.push(a, format!("l{i}"));
        }
        assert_eq!(bus.snapshot(usize::MAX, Some(a)).len(), RING_CAP);
    }

    #[test]
    fn timestamp_has_expected_shape() {
        let ts = timestamp();
        // "YYYY-MM-DD HH:MM:SS" — 19 chars.
        assert_eq!(ts.len(), 19, "got {ts:?}");
        assert_eq!(&ts[4..5], "-");
        assert_eq!(&ts[10..11], " ");
        assert_eq!(&ts[13..14], ":");
    }
}
