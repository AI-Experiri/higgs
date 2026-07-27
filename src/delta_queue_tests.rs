use super::*;

fn content(text: &str) -> ChatDelta {
    ChatDelta {
        kind: ChatDeltaKind::Content,
        text: text.to_owned(),
    }
}

fn reasoning(text: &str) -> ChatDelta {
    ChatDelta {
        kind: ChatDeltaKind::Reasoning,
        text: text.to_owned(),
    }
}

fn tool(text: &str) -> ChatDelta {
    ChatDelta {
        kind: ChatDeltaKind::ToolCall,
        text: text.to_owned(),
    }
}

/// Consecutive same-kind text deltas merge into ONE entry with the texts
/// concatenated in arrival order — the core bounded-entries property.
#[tokio::test]
async fn consecutive_same_kind_deltas_merge_losslessly() {
    let (tx, mut rx) = delta_channel();
    tx.send(content("Hel"));
    tx.send(content("lo "));
    tx.send(content("world"));
    drop(tx);
    assert_eq!(rx.recv().await, Some(content("Hello world")));
    assert_eq!(rx.recv().await, None);
    assert!(!rx.overflowed());
}

/// A kind change starts a new entry; order across kinds is preserved exactly.
#[tokio::test]
async fn kind_alternation_bounds_entries_and_preserves_order() {
    let (tx, mut rx) = delta_channel();
    tx.send(reasoning("think"));
    tx.send(reasoning("ing"));
    tx.send(content("answ"));
    tx.send(content("er"));
    tx.send(reasoning("more"));
    drop(tx);
    assert_eq!(rx.recv().await, Some(reasoning("thinking")));
    assert_eq!(rx.recv().await, Some(content("answer")));
    assert_eq!(rx.recv().await, Some(reasoning("more")));
    assert_eq!(rx.recv().await, None);
}

/// Tool-call fragments are complete JSON units — NEVER merged, even when
/// consecutive; each arrives as its own entry.
#[tokio::test]
async fn tool_call_fragments_never_merge() {
    let (tx, mut rx) = delta_channel();
    tx.send(tool(r#"{"index":0,"id":"a"}"#));
    tx.send(tool(r#"{"index":0,"arguments":"{}"}"#));
    drop(tx);
    assert_eq!(rx.recv().await, Some(tool(r#"{"index":0,"id":"a"}"#)));
    assert_eq!(
        rx.recv().await,
        Some(tool(r#"{"index":0,"arguments":"{}"}"#))
    );
    assert_eq!(rx.recv().await, None);
}

/// A consumer that pulls while the producer is live sees deltas promptly
/// (unmerged when it keeps up) — recv wakes on each push.
#[tokio::test]
async fn live_consumer_receives_each_delta() {
    let (tx, mut rx) = delta_channel();
    tx.send(content("a"));
    assert_eq!(rx.recv().await, Some(content("a")));
    tx.send(content("b"));
    assert_eq!(rx.recv().await, Some(content("b")));
    drop(tx);
    assert_eq!(rx.recv().await, None);
}

/// Exceeding CAP_BYTES drops the backlog, ends the stream, and flags
/// `overflowed` — the anti-OOM contract behind [HG057].
#[tokio::test]
async fn overflow_drops_backlog_and_flags() {
    let (tx, mut rx) = delta_channel();
    let chunk = "x".repeat(1024 * 1024); // 1 MiB per push
    for _ in 0..9 {
        tx.send(content(&chunk)); // 9 MiB total > CAP_BYTES (8 MiB)
    }
    // The 9th push trips the cap: backlog dropped, stream ended.
    assert_eq!(rx.recv().await, None);
    assert!(rx.overflowed());
    // The reported backlog is what was pending when the cap tripped (8 MiB).
    assert_eq!(rx.buffered_bytes(), 8 * 1024 * 1024);
    // Post-overflow sends are ignored, and the queue stays ended.
    tx.send(content("late"));
    assert_eq!(rx.try_recv(), None);
}

/// Sender drop closes the queue but the buffered backlog is still drained
/// first — close is end-of-stream, not data loss.
#[tokio::test]
async fn close_drains_backlog_before_none() {
    let (tx, mut rx) = delta_channel();
    tx.send(content("kept"));
    drop(tx);
    assert_eq!(rx.recv().await, Some(content("kept")));
    assert_eq!(rx.recv().await, None);
}

/// try_recv is non-blocking: None on empty-but-open, data when buffered.
#[tokio::test]
async fn try_recv_is_nonblocking() {
    let (tx, mut rx) = delta_channel();
    assert_eq!(rx.try_recv(), None);
    tx.send(content("x"));
    assert_eq!(rx.try_recv(), Some(content("x")));
    assert_eq!(rx.try_recv(), None);
}

/// Dropping the RECEIVER closes the queue and frees the backlog: subsequent
/// sends buffer NOTHING (the old `mpsc` contract the non-streaming `/v1` path
/// relies on when it drops the receiver and awaits only the final outcome).
#[tokio::test]
async fn receiver_drop_closes_and_frees_the_backlog() {
    let (tx, rx) = delta_channel();
    tx.send(content("buffered-before-drop"));
    drop(rx);
    tx.send(content("after-drop"));
    let s = tx.shared.state.lock();
    assert!(s.closed, "receiver drop marks the queue closed");
    assert_eq!(s.entries.len(), 0, "backlog freed, post-drop sends no-op");
    assert_eq!(s.buffered_bytes, 0, "no bytes retained for a dead consumer");
}

/// buffered_bytes tracks the undelivered backlog down as entries are drained.
#[tokio::test]
async fn buffered_bytes_tracks_backlog() {
    let (tx, mut rx) = delta_channel();
    tx.send(content("abcd"));
    tx.send(reasoning("ef"));
    assert_eq!(rx.buffered_bytes(), 6);
    assert_eq!(rx.recv().await, Some(content("abcd")));
    assert_eq!(rx.buffered_bytes(), 2);
    assert_eq!(rx.recv().await, Some(reasoning("ef")));
    assert_eq!(rx.buffered_bytes(), 0);
}

/// The end-of-stream decision is made in ONE critical section with the pop: a
/// producer completing `send(final) + drop(sender)` while the consumer is between
/// "pop saw empty" and "closed-check" must NOT lose the final delta (Fable r2 —
/// the /v1 SSE consumer treats `None` as complete with no residual drain). The
/// interleaved window itself is not deterministically schedulable without loom;
/// this test pins the STATE that race produced — closed with a buffered backlog —
/// and asserts recv() drains it before reporting end-of-stream. The structural fix
/// (single lock) makes `None` ⇔ closed-and-actually-empty by construction.
#[tokio::test]
async fn recv_never_ends_the_stream_while_deltas_are_buffered() {
    let (tx, mut rx) = delta_channel();
    // Producer completes entirely before the consumer ever polls: the queue is
    // ALREADY closed with the final delta still buffered when recv() runs.
    tx.send(ChatDelta {
        kind: ChatDeltaKind::Content,
        text: "tail".into(),
    });
    drop(tx);
    assert_eq!(
        rx.recv().await.map(|d| d.text),
        Some("tail".into()),
        "the buffered tail delta is delivered despite the queue being closed"
    );
    assert!(rx.recv().await.is_none(), "then end-of-stream");
}
