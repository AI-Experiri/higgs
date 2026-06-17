//! The single home for higgs Developer-Log lines.
//!
//! A [`LogBus`] holds BOTH the bounded history ring (the snapshot source for
//! `GET /api/higgs/logs`) and a live broadcast tap (the source for the
//! `GET /api/higgs/logs/stream` SSE endpoint). Every line — worker child stderr
//! AND higgs serve-layer `tracing` events — enters through [`LogBus::push`],
//! which appends to the ring and sends on the broadcast in one place. There is
//! no second store: the ring is history, the broadcast is the live fan-out of
//! the same lines.
//!
//! ## Wiring
//!
//! The bus is created by the caller (the embedded server's `main`, or the
//! standalone `higgs-server` binary), BEFORE the tracing subscriber is built,
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

use std::collections::VecDeque;

use parking_lot::Mutex;
use tokio::sync::broadcast;
use tracing::field::{Field, Visit};
use tracing::{Event, Subscriber};
use tracing_subscriber::layer::Context;
use tracing_subscriber::Layer;

/// History-ring capacity (lines). Matches the prior `stderr_ring` cap.
const RING_CAP: usize = 2000;

/// Live-broadcast channel capacity (lines). A slow SSE subscriber that falls
/// this far behind is dropped to `Lagged`; the SSE handler skips the gap and
/// keeps streaming rather than crashing.
const BROADCAST_CAP: usize = 256;

/// Tracing target prefix captured by [`HiggsLogLayer`]. Only events whose
/// target starts with `higgs` (the crate's own module path) are mirrored into
/// the bus — never the host application's unrelated spans.
const HIGGS_TARGET_PREFIX: &str = "higgs";

/// The single home for higgs Developer-Log lines: a bounded history ring plus a
/// live broadcast tap. Every log line enters via [`push`](Self::push).
///
/// Cloneable handle semantics are provided by wrapping in `Arc` at the call
/// site (`Arc<LogBus>`); the broadcast `Sender` and the `Mutex<VecDeque>` are
/// the shared state.
#[derive(Debug)]
pub struct LogBus {
    /// Bounded history of recent lines (oldest first). Snapshot source for
    /// `logs(n)`. A `parking_lot::Mutex` held only for push/read of the deque —
    /// never across `.await`.
    ring: Mutex<VecDeque<String>>,
    /// Live fan-out of every pushed line to current SSE subscribers.
    tx: broadcast::Sender<String>,
}

impl LogBus {
    /// Create an empty bus with the default ring and broadcast capacities.
    pub fn new() -> Self {
        let (tx, _) = broadcast::channel(BROADCAST_CAP);
        Self {
            ring: Mutex::new(VecDeque::with_capacity(RING_CAP)),
            tx,
        }
    }

    /// Append `line` to the history ring (evicting the oldest when full) and
    /// fan it out to live subscribers. A send with no subscribers is a no-op —
    /// the line still lands in the ring for later snapshot/replay.
    pub fn push(&self, line: String) {
        {
            let mut ring = self.ring.lock();
            if ring.len() == RING_CAP {
                ring.pop_front();
            }
            ring.push_back(line.clone());
        }
        // Err means zero subscribers — fine; the ring already has the line.
        let _ = self.tx.send(line);
    }

    /// Up to `n` most-recent lines from the history ring (oldest first).
    pub fn snapshot(&self, n: usize) -> Vec<String> {
        let ring = self.ring.lock();
        let skip = ring.len().saturating_sub(n);
        ring.iter().skip(skip).cloned().collect()
    }

    /// Subscribe to live lines pushed AFTER this call. Pair with
    /// [`snapshot`](Self::snapshot) for replay-then-live delivery.
    pub fn subscribe(&self) -> broadcast::Receiver<String> {
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
}

impl Visit for MessageVisitor {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        match field.name() {
            "message" => self.message = format!("{value:?}"),
            "error" => self.error = Some(format!("{value:?}")),
            _ => {}
        }
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        match field.name() {
            "message" => self.message = value.to_owned(),
            "error" => self.error = Some(value.to_owned()),
            _ => {}
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
        let mut visitor = MessageVisitor::default();
        event.record(&mut visitor);
        if visitor.message.is_empty() {
            return;
        }
        let line = match visitor.error {
            Some(err) => format!(
                "{} [{}] {} — {}",
                timestamp(),
                meta.level(),
                visitor.message,
                err
            ),
            None => format!("{} [{}] {}", timestamp(), meta.level(), visitor.message),
        };
        self.bus.push(line);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn push_writes_ring_and_broadcast() {
        let bus = Arc::new(LogBus::new());
        let mut rx = bus.subscribe();
        bus.push("line-1".to_owned());
        // Ring captured it (snapshot history).
        assert_eq!(bus.snapshot(10), vec!["line-1".to_owned()]);
        // Broadcast delivered the same line to the live subscriber.
        assert_eq!(rx.try_recv().unwrap(), "line-1");
    }

    #[test]
    fn snapshot_returns_last_n_oldest_first() {
        let bus = LogBus::new();
        for i in 0..5 {
            bus.push(format!("l{i}"));
        }
        assert_eq!(bus.snapshot(2), vec!["l3".to_owned(), "l4".to_owned()]);
        assert_eq!(bus.snapshot(0), Vec::<String>::new());
    }

    #[test]
    fn ring_evicts_oldest_at_capacity() {
        let bus = LogBus::new();
        for i in 0..(RING_CAP + 5) {
            bus.push(format!("l{i}"));
        }
        let snap = bus.snapshot(RING_CAP + 100);
        assert_eq!(snap.len(), RING_CAP);
        // Oldest 5 were evicted; first surviving line is l5.
        assert_eq!(snap[0], "l5");
    }

    #[test]
    fn subscriber_only_sees_lines_after_subscribe() {
        let bus = LogBus::new();
        bus.push("before".to_owned());
        let mut rx = bus.subscribe();
        bus.push("after".to_owned());
        // "before" is NOT delivered live (it predates the subscription); it is
        // only available via the snapshot replay.
        assert_eq!(rx.try_recv().unwrap(), "after");
        assert!(rx.try_recv().is_err());
        assert_eq!(
            bus.snapshot(10),
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
            bus.push(format!("l{i}"));
        }
        match rx.recv().await {
            Err(broadcast::error::RecvError::Lagged(skipped)) => {
                assert!(skipped > 0, "expected a positive lagged count");
            }
            other => panic!("expected Lagged, got {other:?}"),
        }
        // After Lagged the receiver recovers and yields the still-buffered tail.
        let next = rx.recv().await.expect("recovers after lag");
        assert!(next.starts_with('l'));
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
        let snap = bus.snapshot(10);
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
            "non-error fields stay redacted (no prompt content): {line}"
        );
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
