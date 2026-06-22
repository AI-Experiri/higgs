//! Hub fleet: the set of paired/connected nodes and the `model → (node, worker)` routing
//! the hub uses to send `/v1` chat to a remote-resident worker (DESIGN-remote.md §4.2/§4.3,
//! P3 hub seam).
//!
//! **Actor** (P3 of `docs/superpowers/specs/2026-06-22-actor-runtime-design.md`). The fleet
//! read-model (nodes / routes / node-ids / inventories / per-node generations /
//! owed-unloads) is **private actor state behind one mailbox — no mutexes**. This dissolves
//! the old 7-mutex TOCTOU class: every state transition is now a single message handled in
//! isolation, so two ops can never observe or commit a half-updated table, and the
//! cross-map snapshot (`nodes_view`) is atomic.
//!
//! Per `CLAUDE.md`, a handler does only fast synchronous state work; the slow iroh RPCs
//! (`M_NODE_SCAN`/`_INVENTORY`/`_LOAD`/`_UNLOAD`/`_KILL`, chat setup) run in the async
//! wrapper methods — NOT inside a handler — so a slow load can never head-of-line-block a
//! retire. The wrapper threads each op as: fast state read → slow RPC (off the actor) →
//! fast atomic commit message. Compound transitions (route insert↦displaced, inventory
//! generation-CAS, transport replace+bump, route-drop+log-evict+bump, full retire) are each
//! one message, so they apply all-or-nothing.
//!
//! Generation tokens, not locks: each node carries an `epoch` bumped on every
//! load/unload/kill/route-drop/(re)admission; a `refresh_inventory` commits its (possibly
//! stale) result only if the epoch is unchanged since it started. The map is private actor
//! state, so the check+store is a single message — no lock.
//!
//! Durable routes, transient transports: the `--node` daemon reuses ONE `NodeRuntime`
//! across reconnects, so its workers (and ids) persist through a dropped connection. Routes
//! are therefore keyed by `(node, worker)` and SURVIVE reconnect; only the per-connection
//! transport comes and goes. A genuine node-process restart leaves stale routes that
//! self-heal on the first worker-gone error (the node replies HG007 → route dropped). The
//! transport handle is compared by `Arc` identity so a stale failure can't drop a freshly
//! reconnected transport.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use iroh::endpoint::Connection;
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::sync::{mpsc, oneshot};

use crate::actor::{spawn_actor, Actor, Handle};
use crate::diagnostic::HiggsError;
use crate::log_bus::{LogBus, LogSource};
use crate::node::node_id::{NodeId, NodeIdAllocator};
use crate::node::transport::NodeTransport;
use crate::node::worker_id::WorkerId;
use crate::remote::{
    NodeInventory, M_NODE_INVENTORY, M_NODE_KILL, M_NODE_LOAD, M_NODE_SCAN, M_NODE_UNLOAD,
    N_LOG_LINE,
};
use crate::rpc::{self, RpcFrame};

/// A node key — the peer's canonical `EndpointId` string (same form as the allowlist).
pub type NodeKey = String;

/// Does this failure mean the durable route is stale (worker gone or now serving a
/// different model) and should be dropped? True for a node-reported worker-gone/down
/// (`HG006`/`HG007`) and a model mismatch (`HG018`: after a node restart the reused
/// worker id serves a different model — remotely there's no JIT to recover, so the route
/// must be re-resolved). NOT a dead transport (`WorkerDead`: the node may reconnect with
/// the worker intact — keep the route) or a client error (`HG005`).
fn route_invalidating(e: &HiggsError) -> bool {
    matches!(
        e,
        HiggsError::WorkerRpc { worker_code: Some(c), .. }
            if matches!(c.as_str(), "HG006" | "HG007" | "HG018")
    )
}

higgs_ts! {
/// The hub's UI/API view of one paired node: its stable id, endpoint, whether it's currently
/// connected, and its last-fetched inventory (host + resident workers + hardware/runtime).
#[derive(Debug, Clone, serde::Serialize)]
pub struct NodeView {
    pub node_id: u32,
    pub endpoint_id: String,
    pub connected: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[ts(optional)]
    pub inventory: Option<NodeInventory>,
}
}

/// The actor's typed mailbox. Reads carry a `reply` the wrapper awaits; writes are atomic
/// state transitions (some also reply so the wrapper can sequence the next slow RPC).
enum FleetMsg {
    // --- fast reads ---
    NodeId {
        node: NodeKey,
        reply: oneshot::Sender<Option<NodeId>>,
    },
    NodesView {
        reply: oneshot::Sender<Vec<NodeView>>,
    },
    NodeIds {
        reply: oneshot::Sender<Vec<NodeKey>>,
    },
    Resolve {
        model: String,
        reply: oneshot::Sender<Option<(NodeKey, WorkerId)>>,
    },
    IsRemote {
        model: String,
        reply: oneshot::Sender<bool>,
    },
    RoutedModels {
        reply: oneshot::Sender<Vec<String>>,
    },
    Transport {
        node: NodeKey,
        reply: oneshot::Sender<Result<Arc<NodeTransport>, HiggsError>>,
    },
    Epoch {
        node: NodeKey,
        reply: oneshot::Sender<u64>,
    },
    PendingOwed {
        node: NodeKey,
        reply: oneshot::Sender<Vec<WorkerId>>,
    },
    // --- fast atomic writes ---
    SeedNode {
        node: NodeKey,
        reply: oneshot::Sender<()>,
    },
    AssignNodeId {
        node: NodeKey,
        reply: oneshot::Sender<NodeId>,
    },
    /// Insert/replace a node's transport AND bump its epoch (atomic readmission), returning
    /// any prior transport for the wrapper to close.
    InsertTransport {
        node: NodeKey,
        transport: Arc<NodeTransport>,
        reply: oneshot::Sender<Option<Arc<NodeTransport>>>,
    },
    /// Remove a node's transport only if it's still the current one (Arc identity); closes it.
    DropTransportIf {
        node: NodeKey,
        transport: Arc<NodeTransport>,
        reply: oneshot::Sender<()>,
    },
    Retire {
        node: NodeKey,
        reply: oneshot::Sender<()>,
    },
    /// Insert a route, returning any displaced `(node, worker)`. Does NOT bump the epoch (the
    /// wrapper bumps after the displaced worker is reaped, matching the original ordering).
    InsertRoute {
        model: String,
        new: (NodeKey, WorkerId),
        reply: oneshot::Sender<Option<(NodeKey, WorkerId)>>,
    },
    /// CAS route-drop: remove only if it still equals `expected`; on removal also reclaim the
    /// worker's relayed-log ring and bump the node's epoch — all atomic.
    RemoveRouteIf {
        model: String,
        expected: (NodeKey, WorkerId),
        reply: oneshot::Sender<()>,
    },
    /// Commit a fetched inventory only if the node's epoch is unchanged since the fetch began.
    /// Boxed — `NodeInventory` is large and would bloat every `FleetMsg` otherwise.
    CommitInventory {
        node: NodeKey,
        epoch_before: u64,
        inventory: Box<NodeInventory>,
        reply: oneshot::Sender<()>,
    },
    BumpEpoch {
        node: NodeKey,
        reply: oneshot::Sender<()>,
    },
    EvictRemoteLogs {
        node: NodeKey,
        worker: WorkerId,
        reply: oneshot::Sender<()>,
    },
    PendingInsert {
        route: (NodeKey, WorkerId),
        reply: oneshot::Sender<()>,
    },
    PendingRemove {
        route: (NodeKey, WorkerId),
        reply: oneshot::Sender<()>,
    },
    #[cfg(test)]
    PendingHas {
        route: (NodeKey, WorkerId),
        reply: oneshot::Sender<bool>,
    },
}

/// Private actor state: the hub's whole fleet read-model + routing table, owned by one task.
struct FleetActor {
    /// Currently-connected nodes → their live transport (absent while disconnected).
    nodes: HashMap<NodeKey, Arc<NodeTransport>>,
    /// Durable routes: `model → (node, worker)`, survive reconnect.
    routes: HashMap<String, (NodeKey, WorkerId)>,
    /// Stable hub-local [`NodeId`] per `EndpointId`, for `LogSource::RemoteWorker` tagging.
    node_ids: NodeIdAllocator,
    /// Last-fetched inventory per node (host + workers + hw/rt). The node's reply is
    /// authoritative; refreshed on connect and after every hub-driven lifecycle change.
    inventories: HashMap<NodeKey, NodeInventory>,
    /// Per-node lifecycle generation, bumped on every load/unload/kill/route-drop. A
    /// `refresh_inventory` only commits its (possibly stale) result if this is unchanged
    /// since it started — so a slow connect-time fetch can't clobber a newer state.
    epochs: HashMap<NodeKey, u64>,
    /// Unloads the hub OWES a node but couldn't deliver because the node was disconnected.
    /// Drained on the node's next reconnect (`reconcile_pending_unloads`) so a displaced
    /// worker is reaped instead of leaking RAM/VRAM forever.
    pending_unloads: HashSet<(NodeKey, WorkerId)>,
    /// The hub's Developer-Log bus: relayed remote worker stderr lands here under
    /// `LogSource::RemoteWorker { node, worker }` so it shares the operator's log console.
    bus: Arc<LogBus>,
}

impl FleetActor {
    /// This node's current lifecycle generation (0 if never touched).
    fn epoch(&self, node: &str) -> u64 {
        self.epochs.get(node).copied().unwrap_or(0)
    }

    /// Bump a node's lifecycle generation so any in-flight `refresh_inventory` for it is
    /// invalidated and won't commit a pre-change snapshot.
    fn bump_epoch(&mut self, node: &str) {
        *self.epochs.entry(node.to_string()).or_insert(0) += 1;
    }

    /// Reclaim a remote worker's relayed-log ring, if the node has a known [`NodeId`].
    fn evict_remote_logs(&self, node: &str, worker: WorkerId) {
        if let Some(nid) = self.node_ids.get(node) {
            self.bus.evict_remote(nid, worker);
        }
    }

    /// On connection close, remove ONLY the transport — and only if it's still the current
    /// one (Arc identity), so a reconnect's fresh transport isn't dropped by a stale watcher.
    fn drop_transport_if(&mut self, node: &str, transport: &Arc<NodeTransport>) {
        let removed = if self
            .nodes
            .get(node)
            .is_some_and(|cur| Arc::ptr_eq(cur, transport))
        {
            self.nodes.remove(node)
        } else {
            None
        };
        if let Some(t) = removed {
            tracing::warn!(
                node,
                "higgs: node connection dropped; transport removed (routes kept)"
            );
            // Close so a wedged-but-open connection's close-watcher wakes and releases its
            // Arc (otherwise it would wait on `closed()` forever).
            t.close();
        }
    }

    /// CAS route-drop (see [`FleetMsg::RemoveRouteIf`]).
    fn remove_route_if(&mut self, model: &str, expected: &(NodeKey, WorkerId)) {
        let removed = if self.routes.get(model) == Some(expected) {
            self.routes.remove(model).is_some()
        } else {
            false
        };
        if removed {
            let (node, worker) = expected;
            self.evict_remote_logs(node, *worker);
            // Invalidate any in-flight inventory fetch; the caller refreshes after.
            self.bump_epoch(node);
        }
    }

    /// Explicitly retire a node: a FULL removal (operator action). Drops its transport,
    /// routes, cached inventory, relayed-log rings, owed unloads, AND its durable `NodeId`
    /// slot — all atomically, so the node disappears from the fleet view entirely.
    fn retire(&mut self, node: &str) {
        if let Some(t) = self.nodes.remove(node) {
            t.close();
        }
        // Reclaim ALL of this node's relayed-log rings — including any worker displaced by a
        // reload that's no longer on a current route (a per-route walk would miss those).
        if let Some(nid) = self.node_ids.get(node) {
            self.bus.evict_node(nid);
        }
        self.routes.retain(|_, (n, _)| n != node);
        // Bump so a refresh already in flight can't reinsert stale inventory after this.
        self.bump_epoch(node);
        self.inventories.remove(node);
        // The node is gone for good — drop any unloads we were owing it.
        self.pending_unloads.retain(|(n, _)| n != node);
        // Forget the durable id slot so the node leaves the fleet view (not left disconnected).
        self.node_ids.remove(node);
    }

    /// The fleet view, taken as ONE atomic snapshot across node-ids / nodes / inventories.
    fn nodes_view(&self) -> Vec<NodeView> {
        self.node_ids
            .all()
            .into_iter()
            .map(|(endpoint_id, node_id)| NodeView {
                node_id: node_id.0,
                connected: self.nodes.contains_key(&endpoint_id),
                inventory: self.inventories.get(&endpoint_id).cloned(),
                endpoint_id,
            })
            .collect()
    }

    /// Remote-resident model ids whose node is CURRENTLY connected (servable now), sorted.
    fn routed_models(&self) -> Vec<String> {
        let mut v: Vec<_> = self
            .routes
            .iter()
            .filter(|(_, (node, _))| self.nodes.contains_key(node))
            .map(|(model, _)| model.clone())
            .collect();
        v.sort();
        v
    }
}

impl Actor for FleetActor {
    type Msg = FleetMsg;

    async fn handle(&mut self, msg: FleetMsg) {
        match msg {
            FleetMsg::NodeId { node, reply } => {
                let _ = reply.send(self.node_ids.get(&node));
            }
            FleetMsg::NodesView { reply } => {
                let _ = reply.send(self.nodes_view());
            }
            FleetMsg::NodeIds { reply } => {
                let mut v: Vec<_> = self.nodes.keys().cloned().collect();
                v.sort();
                let _ = reply.send(v);
            }
            FleetMsg::Resolve { model, reply } => {
                let _ = reply.send(self.routes.get(&model).cloned());
            }
            FleetMsg::IsRemote { model, reply } => {
                let _ = reply.send(self.routes.contains_key(&model));
            }
            FleetMsg::RoutedModels { reply } => {
                let _ = reply.send(self.routed_models());
            }
            FleetMsg::Transport { node, reply } => {
                let _ = reply.send(self.nodes.get(&node).cloned().ok_or_else(|| {
                    HiggsError::NodeUnreachable {
                        endpoint_id: node.clone(),
                        detail: "node not connected".into(),
                    }
                }));
            }
            FleetMsg::Epoch { node, reply } => {
                let _ = reply.send(self.epoch(&node));
            }
            FleetMsg::PendingOwed { node, reply } => {
                let owed = self
                    .pending_unloads
                    .iter()
                    .filter(|(n, _)| *n == node)
                    .map(|(_, w)| *w)
                    .collect();
                let _ = reply.send(owed);
            }
            FleetMsg::SeedNode { node, reply } => {
                self.node_ids.assign(&node);
                let _ = reply.send(());
            }
            FleetMsg::AssignNodeId { node, reply } => {
                let _ = reply.send(self.node_ids.assign(&node));
            }
            FleetMsg::InsertTransport {
                node,
                transport,
                reply,
            } => {
                let replaced = self.nodes.insert(node.clone(), transport);
                // Bump on (re)admission so an inventory fetch from a PRIOR connection still in
                // flight can't commit its now-stale result over this fresh one.
                self.bump_epoch(&node);
                let _ = reply.send(replaced);
            }
            FleetMsg::DropTransportIf {
                node,
                transport,
                reply,
            } => {
                self.drop_transport_if(&node, &transport);
                let _ = reply.send(());
            }
            FleetMsg::Retire { node, reply } => {
                self.retire(&node);
                let _ = reply.send(());
            }
            FleetMsg::InsertRoute { model, new, reply } => {
                let _ = reply.send(self.routes.insert(model, new));
            }
            FleetMsg::RemoveRouteIf {
                model,
                expected,
                reply,
            } => {
                self.remove_route_if(&model, &expected);
                let _ = reply.send(());
            }
            FleetMsg::CommitInventory {
                node,
                epoch_before,
                inventory,
                reply,
            } => {
                if self.epoch(&node) == epoch_before {
                    self.inventories.insert(node, *inventory);
                }
                let _ = reply.send(());
            }
            FleetMsg::BumpEpoch { node, reply } => {
                self.bump_epoch(&node);
                let _ = reply.send(());
            }
            FleetMsg::EvictRemoteLogs {
                node,
                worker,
                reply,
            } => {
                self.evict_remote_logs(&node, worker);
                let _ = reply.send(());
            }
            FleetMsg::PendingInsert { route, reply } => {
                self.pending_unloads.insert(route);
                let _ = reply.send(());
            }
            FleetMsg::PendingRemove { route, reply } => {
                self.pending_unloads.remove(&route);
                let _ = reply.send(());
            }
            #[cfg(test)]
            FleetMsg::PendingHas { route, reply } => {
                let _ = reply.send(self.pending_unloads.contains(&route));
            }
        }
    }
}

/// The hub's view of its remote fleet: a thin handle over the actor's mailbox. Cloning the
/// underlying `Handle` keeps the actor alive; dropping the last one ends the loop.
pub struct HubFleet {
    handle: Handle<FleetMsg>,
    /// Immutable, set at construction — kept on the wrapper so `bus()` needs no round-trip.
    bus: Arc<LogBus>,
}

impl HubFleet {
    /// Build a fleet that files relayed remote logs into `bus` (the hub's own `LogBus`, the
    /// one its serve layer reads — so remote worker output appears in the same console).
    /// Spawns the actor task — must be called from within a Tokio runtime.
    pub fn new(bus: Arc<LogBus>) -> Self {
        let bus_for_actor = bus.clone();
        let handle = spawn_actor(FleetActor {
            nodes: HashMap::new(),
            routes: HashMap::new(),
            node_ids: NodeIdAllocator::new(),
            inventories: HashMap::new(),
            epochs: HashMap::new(),
            pending_unloads: HashSet::new(),
            bus: bus_for_actor,
        });
        Self { handle, bus }
    }

    /// Send a message carrying a `reply` and await it; `None` if the actor mailbox is gone
    /// (only possible after every handle drops — never while a caller holds `&self`).
    async fn ask<T: Send + 'static>(
        &self,
        make: impl FnOnce(oneshot::Sender<T>) -> FleetMsg,
    ) -> Option<T> {
        let (tx, rx) = oneshot::channel();
        self.handle.send(make(tx)).ok()?;
        rx.await.ok()
    }

    /// The stable hub-local [`NodeId`] for a node key, if it has ever been admitted.
    pub async fn node_id(&self, node: &str) -> Option<NodeId> {
        self.ask(|reply| FleetMsg::NodeId {
            node: node.to_string(),
            reply,
        })
        .await
        .flatten()
    }

    /// Pre-register a known (e.g. persisted-allowlisted) node so it appears in the fleet view
    /// as DISCONNECTED before it reconnects — assigns its stable `NodeId` without a transport.
    pub async fn seed_node(&self, node: &str) {
        self.ask(|reply| FleetMsg::SeedNode {
            node: node.to_string(),
            reply,
        })
        .await;
    }

    /// The hub Developer-Log bus this fleet relays remote worker stderr into.
    pub fn bus(&self) -> &Arc<LogBus> {
        &self.bus
    }

    /// Register/replace a paired node's transport (after the hub admits its HELLO). Routes
    /// are KEPT across reconnect (the node's workers persist). Closes any prior transport
    /// and spawns a watcher that drops the transport (only) when its connection closes.
    pub async fn add_node(self: &Arc<Self>, node: NodeKey, transport: Arc<NodeTransport>) {
        // Mint (or reuse) this node's stable NodeId so relayed logs tag consistently.
        let node_id = self
            .ask(|reply| FleetMsg::AssignNodeId {
                node: node.clone(),
                reply,
            })
            .await
            .unwrap_or(NodeId(0));
        // Read the node's relayed worker stderr (its uni stream) into the hub bus for THIS
        // connection; ends when the connection closes (accept_uni errors).
        tokio::spawn(read_remote_logs(
            transport.connection(),
            node_id,
            self.bus.clone(),
        ));

        // Insert the transport + bump the epoch atomically; close any prior transport.
        let replaced = self
            .ask(|reply| FleetMsg::InsertTransport {
                node: node.clone(),
                transport: transport.clone(),
                reply,
            })
            .await
            .flatten();
        if let Some(old) = replaced {
            old.close(); // free the old connection + wake its close-watcher
        }
        // On (re)connect: deliver any unloads owed to this node (displaced-while-offline
        // workers), THEN refresh the inventory for the fleet view. Best-effort, off the hot path.
        let inv_weak = Arc::downgrade(self);
        let inv_node = node.clone();
        tokio::spawn(async move {
            if let Some(fleet) = inv_weak.upgrade() {
                fleet.reconcile_pending_unloads(&inv_node).await;
                let _ = fleet.refresh_inventory(&inv_node).await;
            }
        });

        let weak = Arc::downgrade(self);
        let watched = transport;
        tokio::spawn(async move {
            watched.closed().await;
            if let Some(fleet) = weak.upgrade() {
                fleet.drop_transport_if(&node, &watched).await;
            }
        });
    }

    /// Fetch `node`'s on-disk model catalog over its live transport (`M_NODE_SCAN`). The
    /// node's reply is the authoritative scan of its own disk and is returned verbatim —
    /// read-only, no caching, no routes touched. HG027 when the node isn't connected.
    pub async fn scan_node(&self, node: &str) -> Result<Value, HiggsError> {
        let transport = self.transport(node).await?;
        match transport.request(M_NODE_SCAN, json!({})).await {
            Ok(v) => Ok(v),
            Err(e) => Err(self.handle_op_error(node, &transport, e).await),
        }
    }

    /// Fetch `node`'s inventory over its live transport and cache it for the fleet view. The
    /// node's reply is AUTHORITATIVE and stored verbatim — except a result is dropped if a
    /// lifecycle op changed this node's generation while the request was in flight, so a slow
    /// connect-time fetch can never resurrect a stale worker list.
    pub async fn refresh_inventory(&self, node: &str) -> Result<NodeInventory, HiggsError> {
        let epoch_before = self.epoch(node).await;
        let transport = self.transport(node).await?;
        let value = match transport.request(M_NODE_INVENTORY, json!({})).await {
            Ok(v) => v,
            Err(e) => return Err(self.handle_op_error(node, &transport, e).await),
        };
        let inventory: NodeInventory =
            serde_json::from_value(value).map_err(|e| HiggsError::WorkerDead {
                context: format!("node inventory decode failed: {e}"),
            })?;
        // Commit only if no lifecycle op superseded us (the check+store is one message).
        self.ask(|reply| FleetMsg::CommitInventory {
            node: node.to_string(),
            epoch_before,
            inventory: Box::new(inventory.clone()),
            reply,
        })
        .await;
        Ok(inventory)
    }

    /// This node's current lifecycle generation (0 if never touched).
    async fn epoch(&self, node: &str) -> u64 {
        self.ask(|reply| FleetMsg::Epoch {
            node: node.to_string(),
            reply,
        })
        .await
        .unwrap_or(0)
    }

    /// Bump a node's lifecycle generation — call AFTER mutating its routes so any in-flight
    /// `refresh_inventory` for it is invalidated and won't commit a pre-change snapshot.
    async fn bump_epoch(&self, node: &str) {
        self.ask(|reply| FleetMsg::BumpEpoch {
            node: node.to_string(),
            reply,
        })
        .await;
    }

    /// The fleet view: one [`NodeView`] per node the hub has ever admitted, taken as one
    /// atomic snapshot. Sorted by `NodeId` (stable order).
    pub async fn nodes_view(&self) -> Vec<NodeView> {
        self.ask(|reply| FleetMsg::NodesView { reply })
            .await
            .unwrap_or_default()
    }

    /// On connection close, remove ONLY the transport (Arc-identity guarded). Routes are kept
    /// (durable across reconnect); ops return HG027 until the node reconnects.
    async fn drop_transport_if(&self, node: &str, transport: &Arc<NodeTransport>) {
        self.ask(|reply| FleetMsg::DropTransportIf {
            node: node.to_string(),
            transport: transport.clone(),
            reply,
        })
        .await;
    }

    /// Explicitly retire a node: a FULL removal (operator action — the machine is being taken
    /// out of the fleet). Drops its transport, routes, cached inventory, relayed-log rings,
    /// owed unloads, AND its durable `NodeId` slot. Pairs with the hub removing it from the
    /// allowlist.
    pub async fn retire(&self, node: &str) {
        self.ask(|reply| FleetMsg::Retire {
            node: node.to_string(),
            reply,
        })
        .await;
    }

    /// Currently-connected node keys, ascending.
    pub async fn node_ids(&self) -> Vec<NodeKey> {
        self.ask(|reply| FleetMsg::NodeIds { reply })
            .await
            .unwrap_or_default()
    }

    /// The live transport for a node, or HG027 if it isn't currently connected.
    async fn transport(&self, node: &str) -> Result<Arc<NodeTransport>, HiggsError> {
        self.ask(|reply| FleetMsg::Transport {
            node: node.to_string(),
            reply,
        })
        .await
        .unwrap_or_else(|| Err(Self::actor_gone()))
    }

    /// Reclaim a remote worker's relayed-log ring, if the node has a known [`NodeId`].
    async fn evict_remote_logs(&self, node: &str, worker: WorkerId) {
        self.ask(|reply| FleetMsg::EvictRemoteLogs {
            node: node.to_string(),
            worker,
            reply,
        })
        .await;
    }

    /// Resolve `model` to its `(node, worker)` route or fail with HG002.
    async fn require_route(&self, model: &str) -> Result<(NodeKey, WorkerId), HiggsError> {
        self.resolve(model)
            .await
            .ok_or_else(|| HiggsError::ModelNotFound {
                id: model.to_string(),
            })
    }

    /// CAS route-drop: remove a route only if it STILL equals `expected`; on removal also
    /// reclaim the worker's relayed-log ring and bump the node's generation (all atomic).
    async fn remove_route_if(&self, model: &str, expected: &(NodeKey, WorkerId)) {
        self.ask(|reply| FleetMsg::RemoveRouteIf {
            model: model.to_string(),
            expected: expected.clone(),
            reply,
        })
        .await;
    }

    /// On a transport-level failure (`WorkerDead`), drop the dead transport (Arc-identity
    /// guarded) and remap to HG027. Routes are kept. Other errors pass through.
    async fn handle_op_error(
        &self,
        node: &str,
        used: &Arc<NodeTransport>,
        e: HiggsError,
    ) -> HiggsError {
        if matches!(e, HiggsError::WorkerDead { .. }) {
            self.drop_transport_if(node, used).await;
            return HiggsError::NodeUnreachable {
                endpoint_id: node.to_string(),
                detail: e.to_string(),
            };
        }
        e
    }

    /// Load `model` on `node` and record the route. A displaced worker (reload, or a
    /// concurrent load that lost the race) is unloaded best-effort so it's never orphaned.
    pub async fn load(&self, node: &str, model: &str) -> Result<WorkerId, HiggsError> {
        let transport = self.transport(node).await?;
        let result = match transport.request(M_NODE_LOAD, json!({ "id": model })).await {
            Ok(v) => v,
            Err(e) => return Err(self.handle_op_error(node, &transport, e).await),
        };
        let new = (node.to_string(), WorkerId(parse_worker_id(&result)?));
        let displaced = self
            .ask(|reply| FleetMsg::InsertRoute {
                model: model.to_string(),
                new: new.clone(),
                reply,
            })
            .await
            .flatten();
        if let Some(old) = displaced {
            if old != new {
                // Unload the worker this load displaced. If the OLD node is currently
                // disconnected the unload is recorded as pending and delivered on reconnect.
                self.best_effort_unload(&old).await;
            }
        }
        // Refresh the fleet view from the node's authoritative state after the load lands.
        self.bump_epoch(node).await;
        let _ = self.refresh_inventory(node).await;
        Ok(new.1)
    }

    /// Best-effort unload of a displaced worker via its node's CURRENT transport, also
    /// reclaiming its relayed-log ring. If the node is DISCONNECTED the unload can't be
    /// delivered, so it's recorded as pending and reaped on the node's next reconnect.
    async fn best_effort_unload(&self, route: &(NodeKey, WorkerId)) {
        self.evict_remote_logs(&route.0, route.1).await;
        match self.transport(&route.0).await {
            Ok(t) => {
                let res = t
                    .request(M_NODE_UNLOAD, json!({ "worker_id": route.1 .0 }))
                    .await;
                // Clear the obligation ONLY if the unload landed (Ok) or the node reports the
                // worker already gone (HG006/HG007). A real failure means the worker may still
                // be running, so KEEP it pending so the next reconnect retries.
                if res.is_ok() || res.as_ref().err().is_some_and(route_invalidating) {
                    self.pending_remove(route).await;
                } else {
                    self.pending_insert(route).await;
                }
            }
            // Node offline — owe it the unload until it reconnects.
            Err(_) => self.pending_insert(route).await,
        }
        // Re-sync the DISPLACED node's fleet view (it may differ from the load's target node
        // on a cross-node reload, where the caller only refreshes the new node).
        self.bump_epoch(&route.0).await;
        let _ = self.refresh_inventory(&route.0).await;
    }

    async fn pending_insert(&self, route: &(NodeKey, WorkerId)) {
        self.ask(|reply| FleetMsg::PendingInsert {
            route: route.clone(),
            reply,
        })
        .await;
    }

    async fn pending_remove(&self, route: &(NodeKey, WorkerId)) {
        self.ask(|reply| FleetMsg::PendingRemove {
            route: route.clone(),
            reply,
        })
        .await;
    }

    /// Deliver any unloads owed to `node` (recorded while it was offline), now that it's
    /// reconnected — so a worker displaced during a disconnect is reaped rather than leaked.
    /// Only targets workers the hub explicitly tried to unload (never the node's legitimate
    /// resident workers a fresh hub simply hasn't routed yet).
    async fn reconcile_pending_unloads(&self, node: &str) {
        let owed = self
            .ask(|reply| FleetMsg::PendingOwed {
                node: node.to_string(),
                reply,
            })
            .await
            .unwrap_or_default();
        if owed.is_empty() {
            return;
        }
        let Ok(t) = self.transport(node).await else {
            return;
        };
        for w in owed {
            let res = t.request(M_NODE_UNLOAD, json!({ "worker_id": w.0 })).await;
            // Clear on success OR when the node reports the worker already gone (HG006/HG007)
            // — the owed unload is satisfied either way. A transport error keeps it for the
            // next reconnect.
            if res.is_ok() || res.as_ref().err().is_some_and(route_invalidating) {
                self.pending_remove(&(node.to_string(), w)).await;
                self.evict_remote_logs(node, w).await;
            }
        }
    }

    /// Unload a model's remote worker and drop its route.
    pub async fn unload(&self, model: &str) -> Result<(), HiggsError> {
        self.unload_or_kill(model, M_NODE_UNLOAD).await
    }

    /// Force-kill a model's remote worker and drop its route.
    pub async fn kill(&self, model: &str) -> Result<(), HiggsError> {
        self.unload_or_kill(model, M_NODE_KILL).await
    }

    /// Shared unload/kill: clear the route on success OR when the node reports the worker
    /// already gone; node-down → HG027 (transport dropped).
    async fn unload_or_kill(&self, model: &str, method: &str) -> Result<(), HiggsError> {
        let (node, worker) = self.require_route(model).await?;
        let transport = self.transport(&node).await?;
        let res = transport
            .request(method, json!({ "worker_id": worker.0 }))
            .await;
        if res.is_ok() || res.as_ref().err().is_some_and(route_invalidating) {
            // remove_route_if reclaims the log ring + bumps the node's generation.
            self.remove_route_if(model, &(node.clone(), worker)).await;
            // Re-sync the fleet view from the node's authoritative state.
            let _ = self.refresh_inventory(&node).await;
        }
        match res {
            Ok(_) => Ok(()),
            Err(e) => Err(self.handle_op_error(&node, &transport, e).await),
        }
    }

    /// Resolve a model to its `(node, worker)`, if routed.
    pub async fn resolve(&self, model: &str) -> Option<(NodeKey, WorkerId)> {
        self.ask(|reply| FleetMsg::Resolve {
            model: model.to_string(),
            reply,
        })
        .await
        .flatten()
    }

    /// Is this model resident on some remote node?
    pub async fn is_remote(&self, model: &str) -> bool {
        self.ask(|reply| FleetMsg::IsRemote {
            model: model.to_string(),
            reply,
        })
        .await
        .unwrap_or(false)
    }

    /// Remote-resident model ids whose node is CURRENTLY connected (for `/v1/models`
    /// discovery = servable now). A route whose node is disconnected is hidden.
    pub async fn routed_models(&self) -> Vec<String> {
        self.ask(|reply| FleetMsg::RoutedModels { reply })
            .await
            .unwrap_or_default()
    }

    /// Error returned when the actor mailbox is gone (loop ended — only after all handles
    /// drop, which can't happen while a caller holds `&self`).
    fn actor_gone() -> HiggsError {
        HiggsError::NodeUnreachable {
            endpoint_id: String::new(),
            detail: "hub fleet stopped".into(),
        }
    }

    /// Relay a chat to the remote worker hosting `model`. Returns the streamed-delta
    /// receiver + a future resolving to the final result. A worker-gone failure drops the
    /// route (so retries re-resolve); a dead transport drops the transport (HG027).
    pub async fn chat(
        self: &Arc<Self>,
        model: &str,
        messages_json: String,
        max_tokens: usize,
        temperature: f32,
        tools_json: Option<String>,
    ) -> Result<
        (
            mpsc::UnboundedReceiver<String>,
            impl std::future::Future<Output = Result<serde_json::Value, HiggsError>> + Send,
        ),
        HiggsError,
    > {
        let (node, worker) = self.require_route(model).await?;
        let transport = self.transport(&node).await?;
        let (rx, fut) = match transport
            .chat(
                worker.0,
                model.to_string(),
                messages_json,
                max_tokens,
                temperature,
                tools_json,
            )
            .await
        {
            Ok(x) => x,
            Err(e) => return Err(self.handle_op_error(&node, &transport, e).await),
        };

        let fleet = self.clone();
        let model = model.to_string();
        let used = transport;
        let wrapped = async move {
            match fut.await {
                Ok(v) => Ok(v),
                Err(e) if route_invalidating(&e) => {
                    // Worker gone (node alive) — drop the route so a retry re-resolves, and
                    // re-sync the fleet view (remove_route_if bumped the node's generation).
                    fleet.remove_route_if(&model, &(node.clone(), worker)).await;
                    let _ = fleet.refresh_inventory(&node).await;
                    Err(e)
                }
                // Transport-level / other failure surfacing mid-stream: drop the dead
                // transport (Arc-identity guarded) and remap to HG027. The route is kept.
                Err(e) => Err(fleet.handle_op_error(&node, &used, e).await),
            }
        };
        Ok((rx, wrapped))
    }

    /// Test-only: is this `(node, worker)` currently an owed unload?
    #[cfg(test)]
    async fn pending_has(&self, route: &(NodeKey, WorkerId)) -> bool {
        self.ask(|reply| FleetMsg::PendingHas {
            route: route.clone(),
            reply,
        })
        .await
        .unwrap_or(false)
    }
}

/// Extract the `worker_id` from a node's `M_NODE_LOAD` reply, validating it is present and
/// fits a `u32` (the wire type) — a missing or out-of-range value is a protocol fault.
fn parse_worker_id(reply: &serde_json::Value) -> Result<u32, HiggsError> {
    let raw = reply
        .get("worker_id")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| HiggsError::WorkerDead {
            context: "node load reply missing worker_id".into(),
        })?;
    u32::try_from(raw).map_err(|_| HiggsError::WorkerDead {
        context: format!("node load reply worker_id {raw} out of u32 range"),
    })
}

/// Accept the node's uni stream(s) of `N_LOG_LINE` notifications and file each line into the
/// hub bus under `LogSource::RemoteWorker { node, worker }`. Returns when the connection
/// closes. Best-effort: a malformed frame is skipped, not fatal.
async fn read_remote_logs(conn: Connection, node: NodeId, bus: Arc<LogBus>) {
    while let Ok(recv) = conn.accept_uni().await {
        let bus = bus.clone();
        tokio::spawn(async move {
            let mut lines = BufReader::new(recv).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                let Ok(RpcFrame::Notification(n)) = rpc::decode(&line) else {
                    continue;
                };
                if n.method != N_LOG_LINE {
                    continue;
                }
                // The wire documents a u32 worker id; reject (skip) a malformed out-of-range
                // value rather than wrapping it and mis-filing the line under another worker.
                let worker = n
                    .params
                    .get("worker_id")
                    .and_then(serde_json::Value::as_u64)
                    .and_then(|w| u32::try_from(w).ok());
                let text = n.params.get("line").and_then(|v| v.as_str());
                if let (Some(w), Some(t)) = (worker, text) {
                    bus.push(
                        LogSource::RemoteWorker {
                            node,
                            worker: WorkerId(w),
                        },
                        t.to_string(),
                    );
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::serve_node;
    use crate::node::test_support::{fake_runtime, local_endpoint, stage_dummy_model};
    use crate::remote::ALPN;

    async fn fleet_with_one_node() -> (Arc<HubFleet>, NodeKey, String, tempfile::TempDir) {
        let (root, model_id) = stage_dummy_model("higgs-test/m");
        let hub = local_endpoint().await;
        let node = local_endpoint().await;
        let hub_addr = hub.addr();
        let node_key = node.id().to_string();

        let rt = Arc::new(fake_runtime(vec![root.path().to_path_buf()]));
        tokio::spawn(async move {
            let node_conn = node.connect(hub_addr, ALPN).await.expect("connect");
            serve_node(node_conn, rt).await;
        });
        let conn = hub.accept().await.expect("incoming").await.expect("conn");
        std::mem::forget(hub);

        let fleet = Arc::new(HubFleet::new(Arc::new(crate::log_bus::LogBus::new())));
        fleet
            .add_node(node_key.clone(), Arc::new(NodeTransport::new(conn)))
            .await;
        (fleet, node_key, model_id, root)
    }

    #[tokio::test]
    async fn load_routes_and_resolves() {
        let (fleet, node_key, model_id, _root) = fleet_with_one_node().await;
        assert!(!fleet.is_remote(&model_id).await);
        let worker = fleet.load(&node_key, &model_id).await.unwrap();
        assert!(fleet.is_remote(&model_id).await);
        assert_eq!(fleet.resolve(&model_id).await, Some((node_key, worker)));
    }

    #[tokio::test]
    async fn scan_node_returns_the_node_catalog() {
        let (fleet, node_key, model_id, _root) = fleet_with_one_node().await;
        let catalog = fleet.scan_node(&node_key).await.unwrap();
        let models = catalog["models"].as_array().expect("models array");
        assert!(
            models.iter().any(|m| m["id"] == model_id),
            "node catalog lists the staged model: {catalog}"
        );
        // A disconnected/unknown node is unreachable (HG027).
        assert!(fleet.scan_node("ghost").await.is_err());
    }

    #[tokio::test]
    async fn chat_relays_to_routed_worker() {
        let (fleet, node_key, model_id, _root) = fleet_with_one_node().await;
        fleet.load(&node_key, &model_id).await.unwrap();
        let (mut rx, fut) = fleet
            .chat(&model_id, "[]".into(), 8, 0.0, None)
            .await
            .unwrap();
        let collector = tokio::spawn(async move {
            let mut got = Vec::new();
            while let Some(d) = rx.recv().await {
                got.push(d);
            }
            got
        });
        let final_res = fut.await.unwrap();
        assert_eq!(collector.await.unwrap(), vec!["he", "llo"]);
        assert_eq!(final_res["content"], "hello");
    }

    #[tokio::test]
    async fn reload_retires_prior_worker_and_updates_route() {
        let (fleet, node_key, model_id, _root) = fleet_with_one_node().await;
        let w1 = fleet.load(&node_key, &model_id).await.unwrap();
        let w2 = fleet.load(&node_key, &model_id).await.unwrap();
        assert_ne!(w1, w2, "reload spawns a fresh worker (old one unloaded)");
        assert_eq!(fleet.resolve(&model_id).await, Some((node_key, w2)));
    }

    #[tokio::test]
    async fn unload_then_kill_drop_routes() {
        let (fleet, node_key, model_id, _root) = fleet_with_one_node().await;
        fleet.load(&node_key, &model_id).await.unwrap();
        fleet.unload(&model_id).await.unwrap();
        assert!(!fleet.is_remote(&model_id).await, "unload drops the route");

        fleet.load(&node_key, &model_id).await.unwrap();
        fleet.kill(&model_id).await.unwrap();
        assert!(!fleet.is_remote(&model_id).await, "kill drops the route");
    }

    #[tokio::test]
    async fn unload_and_chat_unrouted_model_error() {
        let (fleet, _node_key, _model_id, _root) = fleet_with_one_node().await;
        assert!(fleet.unload("nope/none").await.is_err());
        assert!(fleet.kill("nope/none").await.is_err());
        assert!(fleet.resolve("nope/none").await.is_none());
        assert!(fleet
            .chat("nope/none", "[]".into(), 8, 0.0, None)
            .await
            .is_err());
    }

    #[tokio::test]
    async fn routed_models_lists_connected_only() {
        let (fleet, node_key, model_id, _root) = fleet_with_one_node().await;
        assert!(fleet.routed_models().await.is_empty());
        fleet.load(&node_key, &model_id).await.unwrap();
        assert_eq!(fleet.routed_models().await, vec![model_id.clone()]);
        // After retiring the node, the route is gone → not advertised.
        fleet.retire(&node_key).await;
        assert!(fleet.routed_models().await.is_empty());
    }

    #[tokio::test]
    async fn displaced_unload_to_offline_node_is_recorded_then_reconciled() {
        let (fleet, node_key, model_id, _root) = fleet_with_one_node().await;
        let w = fleet.load(&node_key, &model_id).await.unwrap();

        // An unload owed to a node that ISN'T connected is recorded as pending (not lost).
        fleet
            .best_effort_unload(&("offline-node".to_string(), w))
            .await;
        assert!(
            fleet.pending_has(&("offline-node".to_string(), w)).await,
            "unload to an offline node is recorded as pending"
        );

        // When a node with a pending unload reconnects, reconciliation delivers it (the
        // connected fake node answers M_NODE_UNLOAD), clearing the pending entry.
        fleet.pending_insert(&(node_key.clone(), w)).await;
        fleet.reconcile_pending_unloads(&node_key).await;
        assert!(
            !fleet.pending_has(&(node_key.clone(), w)).await,
            "reconnect drains the owed unload"
        );

        // A pending unload for a worker the node NO LONGER has (e.g. it restarted) must also
        // be cleared — the node replies worker-gone (HG007), which counts as reconciled.
        fleet
            .pending_insert(&(node_key.clone(), WorkerId(9999)))
            .await;
        fleet.reconcile_pending_unloads(&node_key).await;
        assert!(
            !fleet.pending_has(&(node_key.clone(), WorkerId(9999))).await,
            "worker-gone reply clears the owed unload (not left pending forever)"
        );

        // Retiring a node clears anything still owed to it.
        fleet.best_effort_unload(&("gone".to_string(), w)).await;
        assert!(fleet.pending_has(&("gone".to_string(), w)).await);
        fleet.retire("gone").await;
        assert!(
            !fleet.pending_has(&("gone".to_string(), w)).await,
            "retire clears owed unloads for the node"
        );
    }

    #[test]
    fn chat_timeout_does_not_invalidate_a_route() {
        // A remote chat that times out (HG016) is NOT a worker-gone / dead-transport signal:
        // the node may be healthy on a long generation. So it must neither drop the route nor
        // (via handle_op_error, which only acts on WorkerDead) tear down the transport.
        let timeout = HiggsError::ChatTimeout {
            elapsed: std::time::Duration::from_secs(600),
        };
        assert!(
            !route_invalidating(&timeout),
            "chat timeout keeps the route"
        );
        assert!(
            !matches!(timeout, HiggsError::WorkerDead { .. }),
            "chat timeout is not WorkerDead, so handle_op_error passes it through unchanged"
        );
    }

    #[tokio::test]
    async fn seed_node_lists_a_known_node_as_disconnected() {
        let fleet = Arc::new(HubFleet::new(Arc::new(crate::log_bus::LogBus::new())));
        fleet.seed_node("endpointA").await;
        let views = fleet.nodes_view().await;
        assert_eq!(views.len(), 1, "seeded node appears in the view");
        assert_eq!(views[0].endpoint_id, "endpointA");
        assert!(
            !views[0].connected,
            "seeded node is disconnected (no transport yet)"
        );
        assert!(
            views[0].inventory.is_none(),
            "no inventory until it connects"
        );
        assert!(
            fleet.node_id("endpointA").await.is_some(),
            "got a stable NodeId"
        );
    }

    #[tokio::test]
    async fn ops_on_unknown_node_are_unreachable() {
        let fleet = Arc::new(HubFleet::new(Arc::new(crate::log_bus::LogBus::new())));
        let err = fleet.load("ghost", "m").await.unwrap_err();
        assert!(err.to_string().starts_with("[HG027]"), "got {err}");
    }

    #[tokio::test]
    async fn nodes_view_reflects_inventory_and_connection() {
        let (fleet, node_key, model_id, _root) = fleet_with_one_node().await;
        // Establish the inventory cache BEFORE any load (mirrors the connect-time fetch).
        let inv = fleet.refresh_inventory(&node_key).await.unwrap();
        assert!(
            inv.hardware.cpu_cores > 0,
            "inventory carries real hardware"
        );

        // A hub-driven load refreshes the cached view from the node's authoritative state.
        let w = fleet.load(&node_key, &model_id).await.unwrap();
        let workers = fleet.nodes_view().await[0]
            .inventory
            .as_ref()
            .unwrap()
            .workers
            .clone();
        assert!(
            workers
                .iter()
                .any(|x| x.worker_id == w.0 && x.model == model_id),
            "load refreshes the cached inventory: {workers:?}"
        );
        // Unload refreshes it back out.
        fleet.unload(&model_id).await.unwrap();
        let workers = fleet.nodes_view().await[0]
            .inventory
            .as_ref()
            .unwrap()
            .workers
            .clone();
        assert!(
            workers.iter().all(|x| x.worker_id != w.0),
            "unload removes it from the view"
        );

        let views = fleet.nodes_view().await;
        assert_eq!(views.len(), 1, "one node in the fleet view");
        let v = &views[0];
        assert_eq!(v.endpoint_id, node_key);
        assert!(v.connected, "node is connected");
        assert!(v.inventory.is_some(), "view carries the fetched inventory");

        // Retire is a FULL removal — the node leaves the fleet view entirely.
        fleet.retire(&node_key).await;
        assert!(
            fleet.nodes_view().await.is_empty(),
            "retired node is removed from the view (not left disconnected)"
        );
    }

    #[tokio::test]
    async fn retire_drops_node_and_routes() {
        let (fleet, node_key, model_id, _root) = fleet_with_one_node().await;
        fleet.load(&node_key, &model_id).await.unwrap();
        fleet.retire(&node_key).await;
        assert!(fleet.node_ids().await.is_empty());
        assert!(!fleet.is_remote(&model_id).await);
        assert!(fleet
            .chat(&model_id, "[]".into(), 8, 0.0, None)
            .await
            .is_err());
    }
}
