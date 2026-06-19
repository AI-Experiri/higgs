//! Shared actor runtime: a typed mailbox + recv loop + graceful shutdown, written
//! once. Every higgs actor (Supervisor today; NodeRuntime + per-node transport in
//! P2/P3) contributes only its own `Msg` set and `handle`; nobody re-implements the
//! loop. See `DESIGN-remote.md` §2.5.

use tokio::sync::mpsc;

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
        Handle { tx: self.tx.clone() }
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
