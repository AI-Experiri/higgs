//! Bounded, merging chat-delta channel — the backpressure buffer between a
//! delta producer (supervisor demux, hub transport reader) and its single
//! streaming consumer (SSE assembly, fleet relay).
//!
//! The previous pipeline used `mpsc::unbounded_channel<ChatDelta>` at every
//! hop: a slow or stalled SSE client accumulated one queue ENTRY per generated
//! token (each with its own allocation and channel overhead), unbounded in
//! count. This queue instead MERGES consecutive same-kind deltas in place
//! (vLLM / omlx "merging collector" pattern), so:
//!
//! - **entries** are bounded by the number of kind alternations in the stream
//!   (content ↔ reasoning ↔ tool-call), not by token count — a stalled client
//!   holds a handful of growing strings, not thousands of fragments;
//! - **bytes** are bounded by the generation itself (which `max_tokens` /
//!   `ctx_len` already bound) — merging is LOSSLESS, text is concatenated in
//!   arrival order;
//! - a hard [`CAP_BYTES`] safety cap turns a pathological stall (client holds
//!   the connection but never reads while a huge generation streams) into a
//!   LOUD coded failure (`[HG057]`, [`HiggsError::ChatStreamOverflow`]) instead
//!   of unbounded memory growth: the buffer is dropped, the queue closes, and
//!   the consumer surfaces the error.
//!
//! Tool-call fragments are never merged — each is a complete JSON fragment
//! whose boundaries the `/v1` chunk protocol preserves; they are few per turn.
//!
//! Single-producer / single-consumer by construction (one queue per request);
//! the sender is `Clone`-free and closes the queue on drop, matching the
//! `UnboundedSender` semantics the demux relied on (drop ⇒ receiver sees end
//! of stream after draining).
//!
//! [`HiggsError::ChatStreamOverflow`]: crate::diagnostic::HiggsError::ChatStreamOverflow

use std::collections::VecDeque;
use std::sync::Arc;

use parking_lot::Mutex;
use tokio::sync::Notify;

use crate::worker::engine::{ChatDelta, ChatDeltaKind};

/// Hard cap on UNDELIVERED buffered delta bytes per request. Generations are
/// already bounded by `max_tokens`/`ctx_len` (hundreds of KB of text at the
/// extreme), so a healthy request never approaches this; the cap exists only
/// so a stalled client cannot pin unbounded memory. Tripping it aborts the
/// stream with `[HG057]`.
pub const CAP_BYTES: usize = 8 * 1024 * 1024;

/// Create one delta queue: `(sender, receiver)`. One per chat request.
pub fn delta_channel() -> (DeltaSender, DeltaReceiver) {
    let shared = Arc::new(Shared {
        state: Mutex::new(State {
            entries: VecDeque::new(),
            buffered_bytes: 0,
            overflowed: false,
            closed: false,
        }),
        notify: Notify::new(),
    });
    (
        DeltaSender {
            shared: Arc::clone(&shared),
        },
        DeltaReceiver { shared },
    )
}

struct State {
    /// Merged runs, in arrival order. Consecutive same-kind text deltas are
    /// one entry; tool-call fragments are always their own entry.
    entries: VecDeque<ChatDelta>,
    /// Total bytes across `entries` (the undelivered backlog).
    buffered_bytes: usize,
    /// The [`CAP_BYTES`] cap tripped: buffer dropped, stream aborted.
    overflowed: bool,
    /// Sender dropped (or map entry removed): end of stream after draining.
    closed: bool,
}

struct Shared {
    state: Mutex<State>,
    notify: Notify,
}

/// Producer half. Dropping it closes the queue (the receiver drains what is
/// buffered, then sees end-of-stream).
pub struct DeltaSender {
    shared: Arc<Shared>,
}

impl DeltaSender {
    /// Push one delta, merging into the tail entry when it has the same kind
    /// (tool-call fragments never merge). Silently ignored after close or
    /// overflow — matching the old `let _ = tx.send(..)` semantics.
    pub fn send(&self, delta: ChatDelta) {
        let mut s = self.shared.state.lock();
        if s.closed || s.overflowed {
            return;
        }
        if s.buffered_bytes.saturating_add(delta.text.len()) > CAP_BYTES {
            // Pathological stall: drop the backlog and abort the stream loudly
            // rather than growing without bound. `buffered_bytes` keeps its
            // final value so the consumer can report how much was pending.
            s.entries.clear();
            s.overflowed = true;
            drop(s);
            self.shared.notify.notify_one();
            return;
        }
        s.buffered_bytes += delta.text.len();
        match s.entries.back_mut() {
            Some(back) if back.kind == delta.kind && delta.kind != ChatDeltaKind::ToolCall => {
                back.text.push_str(&delta.text);
            }
            _ => s.entries.push_back(delta),
        }
        drop(s);
        self.shared.notify.notify_one();
    }
}

impl Drop for DeltaSender {
    fn drop(&mut self) {
        self.shared.state.lock().closed = true;
        self.shared.notify.notify_one();
    }
}

/// Consumer half (single consumer). `recv().await` yields merged runs in
/// order; `None` means the stream ended — either drained-after-close (normal)
/// or aborted by overflow (check [`overflowed`](Self::overflowed)).
pub struct DeltaReceiver {
    shared: Arc<Shared>,
}

impl DeltaReceiver {
    /// Await the next merged delta. `None` ⇔ closed-and-drained or overflowed.
    pub async fn recv(&mut self) -> Option<ChatDelta> {
        loop {
            // ONE critical section for both the pop and the end-of-stream check:
            // with separate locks, a producer `send(final delta)` + sender-drop
            // landing BETWEEN them made this return `None` with that delta still
            // buffered — silently losing the tail of the stream (the SSE consumer
            // treats `None` as complete and does no residual drain). Under one
            // lock, `None` can only mean closed-and-ACTUALLY-empty — the
            // documented drain-after-close contract.
            {
                let mut s = self.shared.state.lock();
                if let Some(d) = s.entries.pop_front() {
                    s.buffered_bytes = s.buffered_bytes.saturating_sub(d.text.len());
                    return Some(d);
                }
                if s.closed || s.overflowed {
                    return None;
                }
            }
            // `Notify` stores a permit when nobody is waiting, so a push that
            // lands between the check above and this await is never lost.
            self.shared.notify.notified().await;
        }
    }

    /// Non-blocking pop of the next merged delta (residual-drain loops).
    pub fn try_recv(&mut self) -> Option<ChatDelta> {
        self.pop()
    }

    /// Whether the stream was aborted by the [`CAP_BYTES`] overflow guard.
    /// Consumers surface this as [`HiggsError::ChatStreamOverflow`] (`[HG057]`)
    /// instead of ending the stream as if it completed.
    ///
    /// [`HiggsError::ChatStreamOverflow`]: crate::diagnostic::HiggsError::ChatStreamOverflow
    pub fn overflowed(&self) -> bool {
        self.shared.state.lock().overflowed
    }

    /// Bytes that were pending when the stream overflowed (for the error).
    pub fn buffered_bytes(&self) -> usize {
        self.shared.state.lock().buffered_bytes
    }

    fn pop(&mut self) -> Option<ChatDelta> {
        let mut s = self.shared.state.lock();
        let d = s.entries.pop_front()?;
        s.buffered_bytes = s.buffered_bytes.saturating_sub(d.text.len());
        Some(d)
    }
}

impl Drop for DeltaReceiver {
    /// Receiver gone ⇒ nobody will ever read: close the queue and FREE the
    /// backlog, so producer-side sends become no-ops — preserving the old
    /// `mpsc` behavior the non-streaming `/v1` path relies on (it drops the
    /// receiver and awaits only the final outcome; without this, every
    /// non-streaming generation would pointlessly buffer its whole delta
    /// stream until completion).
    fn drop(&mut self) {
        let mut s = self.shared.state.lock();
        s.closed = true;
        s.entries.clear();
        s.buffered_bytes = 0;
    }
}

impl std::fmt::Debug for DeltaReceiver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = self.shared.state.lock();
        f.debug_struct("DeltaReceiver")
            .field("entries", &s.entries.len())
            .field("buffered_bytes", &s.buffered_bytes)
            .field("overflowed", &s.overflowed)
            .field("closed", &s.closed)
            .finish()
    }
}

#[cfg(test)]
#[path = "delta_queue_tests.rs"]
mod tests;
