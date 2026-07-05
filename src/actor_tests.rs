use super::*;
use crate::worker::engine::{ChatDelta, ChatDeltaKind};
use serde_json::json;
use std::sync::Arc;

/// A content-kind [`ChatDelta`] — what plain string chunks became.
fn content_delta(text: &str) -> ChatDelta {
    ChatDelta {
        kind: ChatDeltaKind::Content,
        text: text.into(),
    }
}

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
async fn send_succeeds_while_a_clone_keeps_the_loop_alive() {
    let handle = spawn_actor(Counter { total: 0 });
    let clone = handle.clone();
    drop(handle); // a clone still holds the mailbox open
    clone.send(CounterMsg::Add(1)).unwrap();
}

fn ok_response(id: u64) -> RpcResponse {
    RpcResponse {
        jsonrpc: "2.0".into(),
        id,
        result: Some(json!({ "ok": true })),
        error: None,
    }
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
    demux.route_chunk(3, content_delta("he"));
    demux.route_chunk(3, content_delta("llo"));
    demux.route_chunk(4, content_delta("ignored")); // no sink for 4 ⇒ dropped, no panic
                                                    // The sink is a merging delta queue: both buffered same-kind chunks arrive
                                                    // as ONE merged run (text preserved in order).
    assert_eq!(rx.recv().await.unwrap(), content_delta("hello"));
    demux.remove_sink(3);
}

#[tokio::test]
async fn demux_remove_pending_orphans_the_waiter() {
    let demux = ReplyDemux::new();
    let rx = demux.register_pending(5);
    demux.remove_pending(5);
    // a late response for a removed id is silently dropped...
    demux.correlate(ok_response(5));
    assert!(rx.await.is_err()); // ...and the waiter never resolves
}

/// An actor that does NOT override `on_stop`, so dropping its handle exercises the
/// trait's DEFAULT `on_stop` (`async {}`) before the recv loop ends. A drop-guard
/// flips a flag so the test can confirm the loop actually ran to completion.
struct DefaultStopActor {
    stopped: Arc<std::sync::atomic::AtomicBool>,
}

impl Actor for DefaultStopActor {
    type Msg = ();
    async fn handle(&mut self, _msg: ()) {}
    // on_stop intentionally NOT overridden — the default `async {}` runs.
}

impl Drop for DefaultStopActor {
    fn drop(&mut self) {
        self.stopped
            .store(true, std::sync::atomic::Ordering::SeqCst);
    }
}

#[tokio::test]
async fn default_on_stop_runs_on_graceful_shutdown() {
    let stopped = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let handle = spawn_actor(DefaultStopActor {
        stopped: stopped.clone(),
    });
    handle.send(()).unwrap();
    // Dropping the only handle closes the mailbox; the loop ends and the default
    // `on_stop` (async {}) runs before the actor state is dropped.
    drop(handle);
    // Yield until the spawned task has run through `on_stop` and dropped the state.
    for _ in 0..10_000 {
        if stopped.load(std::sync::atomic::Ordering::SeqCst) {
            break;
        }
        tokio::task::yield_now().await;
    }
    assert!(
        stopped.load(std::sync::atomic::Ordering::SeqCst),
        "the recv loop ended and the default on_stop ran"
    );
}

#[tokio::test]
async fn demux_fail_all_pending_drops_senders() {
    let demux = ReplyDemux::new();
    let rx = demux.register_pending(1);
    demux.fail_all_pending(); // EOF: drop every pending sender
    assert!(rx.await.is_err()); // oneshot canceled
}
