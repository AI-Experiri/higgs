//! Shared actor runtime: a typed mailbox + recv loop + graceful shutdown, written
//! once. Every higgs actor (Supervisor today; NodeRuntime + per-node transport in
//! P2/P3) contributes only its own `Msg` set and `handle`; nobody re-implements the
//! loop. See `DESIGN-remote.md` §2.5.

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::Mutex;
use tokio::sync::{mpsc, oneshot};

use crate::rpc::RpcResponse;

// `Actor` / `spawn_actor` / `Handle` are the generic mailbox runtime, written once
// here. Their first real consumers are `NodeRuntime` and the per-node iroh transport
// (P2/P3); `ReplyDemux` (below) is consumed now by `Supervisor`. The `allow(dead_code)`
// on the not-yet-wired items is removed when P2/P3 land.

/// An actor: isolated state, reacting to its own typed messages off a mailbox.
/// `handle` is `async` so an actor may await I/O / `spawn_blocking` inside it.
#[allow(dead_code)] // consumed by NodeRuntime + per-node transport in P2/P3
pub(crate) trait Actor: Send + 'static {
    type Msg: Send + 'static;
    fn handle(&mut self, msg: Self::Msg) -> impl std::future::Future<Output = ()> + Send;
}

/// A cloneable handle to an actor's mailbox. When the last clone is dropped the
/// mailbox closes and the recv loop ends (graceful shutdown).
#[allow(dead_code)] // consumed by NodeRuntime + per-node transport in P2/P3
pub(crate) struct Handle<M> {
    tx: mpsc::UnboundedSender<M>,
}

impl<M> Clone for Handle<M> {
    fn clone(&self) -> Self {
        Handle {
            tx: self.tx.clone(),
        }
    }
}

impl<M> Handle<M> {
    /// Enqueue a message. Errs only after every handle is dropped / the loop ended.
    #[allow(dead_code)] // consumed by NodeRuntime + per-node transport in P2/P3
    pub(crate) fn send(&self, msg: M) -> Result<(), mpsc::error::SendError<M>> {
        self.tx.send(msg)
    }
}

/// Spawn `state` as an actor: mailbox + recv loop + shutdown-on-last-handle-drop.
/// The runtime, written ONCE — each actor contributes only `Msg` + `handle`.
#[allow(dead_code)] // consumed by NodeRuntime + per-node transport in P2/P3
pub(crate) fn spawn_actor<A: Actor>(mut state: A) -> Handle<A::Msg> {
    let (tx, mut rx) = mpsc::unbounded_channel::<A::Msg>();
    tokio::spawn(async move {
        while let Some(msg) = rx.recv().await {
            state.handle(msg).await;
        }
        // All handles dropped ⇒ graceful shutdown.
    });
    Handle { tx }
}

/// The reader-side reply-demux shared by every RPC *client* (Supervisor today; the
/// per-node iroh transport in P3). Server-side actors (Worker) reply inline and do
/// NOT use this. Internally `Arc`-shared so a read-loop and the calling tasks see one
/// map. See `DESIGN-remote.md` §2.5.
#[derive(Clone)]
pub(crate) struct ReplyDemux {
    inner: Arc<DemuxInner>,
}

struct DemuxInner {
    /// Request id → response waiter (RPC correlation).
    pending: Mutex<HashMap<u64, oneshot::Sender<RpcResponse>>>,
    /// request_id → chat-chunk sink (streaming delta routing).
    chat_sinks: Mutex<HashMap<u64, mpsc::UnboundedSender<String>>>,
}

impl ReplyDemux {
    pub(crate) fn new() -> Self {
        ReplyDemux {
            inner: Arc::new(DemuxInner {
                pending: Mutex::new(HashMap::new()),
                chat_sinks: Mutex::new(HashMap::new()),
            }),
        }
    }

    /// Register a request id; the returned receiver resolves when its response
    /// arrives (or errors if the sender is dropped — e.g. on EOF).
    pub(crate) fn register_pending(&self, id: u64) -> oneshot::Receiver<RpcResponse> {
        let (tx, rx) = oneshot::channel();
        self.inner.pending.lock().insert(id, tx);
        rx
    }

    /// Remove an orphaned pending entry (e.g. on RPC timeout).
    pub(crate) fn remove_pending(&self, id: u64) {
        self.inner.pending.lock().remove(&id);
    }

    /// Deliver a response to its waiter. Unknown ids are dropped (no panic).
    pub(crate) fn correlate(&self, resp: RpcResponse) {
        if let Some(tx) = self.inner.pending.lock().remove(&resp.id) {
            let _ = tx.send(resp);
        }
    }

    /// EOF/death: cancel every pending waiter by dropping its sender.
    pub(crate) fn fail_all_pending(&self) {
        self.inner.pending.lock().clear();
    }

    /// Register a chat-chunk sink under a `request_id`; deltas arrive on the receiver.
    pub(crate) fn register_sink(&self, request_id: u64) -> mpsc::UnboundedReceiver<String> {
        let (tx, rx) = mpsc::unbounded_channel();
        self.inner.chat_sinks.lock().insert(request_id, tx);
        rx
    }

    /// Drop a chat-chunk sink (closes its receiver).
    pub(crate) fn remove_sink(&self, request_id: u64) {
        self.inner.chat_sinks.lock().remove(&request_id);
    }

    /// Route a streamed delta to its sink. Unknown request_ids are dropped (no panic).
    pub(crate) fn route_chunk(&self, request_id: u64, delta: &str) {
        if let Some(tx) = self.inner.chat_sinks.lock().get(&request_id) {
            let _ = tx.send(delta.to_string());
        }
    }

    /// Death: drop every chat sink, closing each receiver (ends in-flight streams).
    pub(crate) fn clear_sinks(&self) {
        self.inner.chat_sinks.lock().clear();
    }

    /// Number of live chat sinks (in-flight streaming requests; test introspection).
    #[cfg(test)]
    pub(crate) fn active_sink_count(&self) -> usize {
        self.inner.chat_sinks.lock().len()
    }

    /// Number of outstanding pending requests (test introspection only).
    #[cfg(test)]
    pub(crate) fn pending_count(&self) -> usize {
        self.inner.pending.lock().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

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
        demux.route_chunk(3, "he");
        demux.route_chunk(3, "llo");
        demux.route_chunk(4, "ignored"); // no sink for 4 ⇒ dropped, no panic
        assert_eq!(rx.recv().await.unwrap(), "he");
        assert_eq!(rx.recv().await.unwrap(), "llo");
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

    #[tokio::test]
    async fn demux_fail_all_pending_drops_senders() {
        let demux = ReplyDemux::new();
        let rx = demux.register_pending(1);
        demux.fail_all_pending(); // EOF: drop every pending sender
        assert!(rx.await.is_err()); // oneshot canceled
    }
}
