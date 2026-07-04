//! Shared actor runtime: a typed mailbox + recv loop + graceful shutdown, written
//! once. Every higgs actor (Supervisor today; NodeRuntime + per-node transport in
//! P2/P3) contributes only its own `Msg` set and `handle`; nobody re-implements the
//! loop. See `DESIGN-remote.md` §2.5.

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::Mutex;
use tokio::sync::{mpsc, oneshot};

use crate::rpc::RpcResponse;
use crate::worker::engine::ChatDelta;

// `Actor` / `spawn_actor` / `Handle` are the generic mailbox runtime, written once
// here. Consumed now by `NodeRuntime` (P1) and `ReplyDemux` by `Supervisor`; the
// per-node iroh transport joins in P2/P3.

/// An actor: isolated state, reacting to its own typed messages off a mailbox.
/// `handle` is `async` so an actor may await I/O / `spawn_blocking` inside it.
///
/// **Slow I/O rule (see CLAUDE.md):** a `handle` must NOT `await` a slow downstream
/// RPC — that would serialize every op behind it. Do only fast synchronous state work
/// per message; run slow work in a `tokio::spawn` and apply its result via a follow-up
/// "commit" message sent back through a [`WeakHandle`].
pub(crate) trait Actor: Send + 'static {
    type Msg: Send + 'static;
    fn handle(&mut self, msg: Self::Msg) -> impl std::future::Future<Output = ()> + Send;

    /// Called once after the mailbox closes (all handles dropped). Default no-op;
    /// actors owning OS resources (e.g. child workers) override it to drain them —
    /// unlike `Drop`, this runs in async context so it can `.await` the teardown.
    fn on_stop(&mut self) -> impl std::future::Future<Output = ()> + Send {
        async {}
    }
}

/// A cloneable handle to an actor's mailbox. When the last clone is dropped the
/// mailbox closes and the recv loop ends (graceful shutdown).
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
    pub(crate) fn send(&self, msg: M) -> Result<(), mpsc::error::SendError<M>> {
        self.tx.send(msg)
    }

    /// A non-owning handle: it does NOT keep the mailbox open, so an actor can hold one
    /// to itself (for commit-messages) without preventing its own graceful shutdown.
    pub(crate) fn downgrade(&self) -> WeakHandle<M> {
        WeakHandle {
            tx: self.tx.downgrade(),
        }
    }
}

/// A self-reference an actor keeps to post follow-up commit messages. Upgrades to a
/// live [`Handle`] only while at least one strong handle survives; once the actor is
/// shutting down, `upgrade()` returns `None` and the commit is dropped.
pub(crate) struct WeakHandle<M> {
    tx: mpsc::WeakUnboundedSender<M>,
}

impl<M> Clone for WeakHandle<M> {
    fn clone(&self) -> Self {
        WeakHandle {
            tx: self.tx.clone(),
        }
    }
}

impl<M> WeakHandle<M> {
    /// Upgrade to a live mailbox handle, or `None` if the actor has shut down.
    pub(crate) fn upgrade(&self) -> Option<Handle<M>> {
        self.tx.upgrade().map(|tx| Handle { tx })
    }
}

/// Spawn `state` as an actor: mailbox + recv loop + shutdown-on-last-handle-drop.
/// The runtime, written ONCE — each actor contributes only `Msg` + `handle`.
#[allow(dead_code)] // direct form kept for actors that need no self-handle (tests; future)
pub(crate) fn spawn_actor<A: Actor>(state: A) -> Handle<A::Msg> {
    spawn_actor_with(|_| state)
}

/// Like [`spawn_actor`], but `build` receives the actor's own [`Handle`] so the state
/// can stash a [`WeakHandle`] to itself (downgrade it — never store the strong handle,
/// which would pin the mailbox open forever). This is how an actor posts the commit
/// messages of the slow-I/O pattern back to itself.
pub(crate) fn spawn_actor_with<A: Actor>(
    build: impl FnOnce(Handle<A::Msg>) -> A,
) -> Handle<A::Msg> {
    let (tx, mut rx) = mpsc::unbounded_channel::<A::Msg>();
    let handle = Handle { tx };
    let mut state = build(handle.clone());
    tokio::spawn(async move {
        while let Some(msg) = rx.recv().await {
            state.handle(msg).await;
        }
        // All handles dropped ⇒ graceful shutdown: drain owned resources.
        state.on_stop().await;
    });
    handle
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
    chat_sinks: Mutex<HashMap<u64, mpsc::UnboundedSender<ChatDelta>>>,
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
    pub(crate) fn register_sink(&self, request_id: u64) -> mpsc::UnboundedReceiver<ChatDelta> {
        let (tx, rx) = mpsc::unbounded_channel();
        self.inner.chat_sinks.lock().insert(request_id, tx);
        rx
    }

    /// Drop a chat-chunk sink (closes its receiver).
    pub(crate) fn remove_sink(&self, request_id: u64) {
        self.inner.chat_sinks.lock().remove(&request_id);
    }

    /// Route a streamed delta to its sink. Unknown request_ids are dropped (no panic).
    pub(crate) fn route_chunk(&self, request_id: u64, delta: ChatDelta) {
        if let Some(tx) = self.inner.chat_sinks.lock().get(&request_id) {
            let _ = tx.send(delta);
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
#[path = "actor_tests.rs"]
mod tests;
