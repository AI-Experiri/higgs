//! Hub fleet: the set of paired/connected nodes and the `model → (node, worker)` routing
//! the hub uses to send `/v1` chat to a remote-resident worker (DESIGN-remote.md §4.2/§4.3,
//! P3 hub seam).
//!
//! **Actor** (P3 of `docs/superpowers/specs/2026-06-22-actor-runtime-design.md`). The fleet
//! read-model (nodes / instance routes / node-ids / inventories / per-node generations) is
//! **private actor state behind one mailbox — no mutexes**. This dissolves the old 7-mutex
//! TOCTOU class: every state transition is now a single message handled in isolation, so two
//! ops can never observe or commit a half-updated table, and the cross-map snapshot
//! (`nodes_view`) is atomic.
//!
//! Per `CLAUDE.md`, a handler does only fast synchronous state work; the slow iroh RPCs
//! (`M_NODE_SCAN`/`_INVENTORY`/`_LOAD`/`_UNLOAD`/`_KILL`, chat setup) run in the async
//! wrapper methods — NOT inside a handler — so a slow load can never head-of-line-block a
//! retire. The wrapper threads each op as: fast state read → slow RPC (off the actor) →
//! fast atomic commit message. Compound transitions (instance add, inventory generation-CAS,
//! transport replace+bump, instance-drop+log-evict+bump, full retire) are each one message,
//! so they apply all-or-nothing.
//!
//! **Served instance ids (P3b).** Routes are keyed by INSTANCE — `(node, worker) → raw model`
//! — so N workers serving the same model coexist (loads are additive, not only-keep-last).
//! Each instance's SERVED id is a deterministic function of the live set (`org/model`,
//! `org/model-1`, … sorted by `(node, worker)`), derived on demand by `served_ids`, never
//! persisted. `/v1` addresses a served id; chat sends the RAW model on the wire (the worker
//! matches on that, so a served suffix never looks like a model mismatch).
//!
//! Generation tokens, not locks: each node carries an `epoch` bumped on every
//! load/unload/kill/instance-drop/(re)admission; a `refresh_inventory` commits its (possibly
//! stale) result only if the epoch is unchanged since it started. The map is private actor
//! state, so the check+store is a single message — no lock.
//!
//! Durable routes, transient transports: the `--node` daemon reuses ONE `NodeRuntime`
//! across reconnects, so its workers (and ids) persist through a dropped connection. Instance
//! routes are keyed by `(node, worker)` and SURVIVE reconnect; only the per-connection
//! transport comes and goes. A genuine node-process restart leaves stale routes that
//! self-heal on the first worker-gone error (the node replies HG007 → instance dropped). The
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
    /// Resolve a SERVED instance id (`org/model`, `org/model-1`, …) to its instance + the
    /// RAW model the worker actually loaded (needed for the worker's model-match check).
    ResolveServed {
        served: String,
        reply: oneshot::Sender<Option<(NodeKey, WorkerId, String)>>,
    },
    IsRemote {
        served: String,
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
    /// Add an instance `(node, worker) → model`. Loads are ADDITIVE — N workers serving the
    /// same model coexist as N instances (distinct served ids). Worker ids are unique per
    /// node, so this never collides except when a restarted node reuses an id (then the stale
    /// instance is correctly replaced). Does NOT bump the epoch (the wrapper bumps after).
    AddInstance {
        node: NodeKey,
        worker: WorkerId,
        model: String,
        reply: oneshot::Sender<()>,
    },
    /// CAS instance-drop: remove `(node, worker)` only if it STILL serves `expected_model`
    /// (guards against a node restart reusing the id for a DIFFERENT model). On removal also
    /// reclaim the worker's relayed-log ring and bump the node's epoch — all atomic.
    ///
    /// Residual (pre-existing, unchanged from the model-keyed design): if a node PROCESS
    /// restarts and a fresh load reuses the exact same worker id for the SAME model, a stale
    /// invalidation could in principle drop the new instance. In practice the restart drops
    /// the transport first, so an in-flight chat resolves to `WorkerDead` (the transport
    /// branch, not the `route_invalidating` HG006/7 branch), and `unload`/`kill` re-resolve
    /// the instance fresh per call — so the window is not readily reachable. A full fix needs
    /// a per-worker generation token on the wire (deferred).
    RemoveInstanceIf {
        node: NodeKey,
        worker: WorkerId,
        expected_model: String,
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
}

/// Private actor state: the hub's whole fleet read-model + routing table, owned by one task.
struct FleetActor {
    /// Currently-connected nodes → their live transport (absent while disconnected).
    nodes: HashMap<NodeKey, Arc<NodeTransport>>,
    /// The live instance set: `(node, worker) → raw model id`, durable across reconnect. N
    /// workers serving the same model are N distinct instances; their SERVED ids
    /// (`org/model`, `org/model-1`, … sorted by `(node, worker)`) are derived on demand by
    /// [`FleetActor::served_ids`] — a deterministic function of this set, never persisted.
    routes: HashMap<(NodeKey, WorkerId), String>,
    /// Stable hub-local [`NodeId`] per `EndpointId`, for `LogSource::RemoteWorker` tagging.
    node_ids: NodeIdAllocator,
    /// Last-fetched inventory per node (host + workers + hw/rt). The node's reply is
    /// authoritative; refreshed on connect and after every hub-driven lifecycle change.
    inventories: HashMap<NodeKey, NodeInventory>,
    /// Per-node lifecycle generation, bumped on every load/unload/kill/route-drop. A
    /// `refresh_inventory` only commits its (possibly stale) result if this is unchanged
    /// since it started — so a slow connect-time fetch can't clobber a newer state.
    epochs: HashMap<NodeKey, u64>,
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

    /// CAS instance-drop (see [`FleetMsg::RemoveInstanceIf`]): remove `(node, worker)` only if
    /// it still serves `expected_model`.
    fn remove_instance_if(&mut self, node: &str, worker: WorkerId, expected_model: &str) {
        let key = (node.to_string(), worker);
        let removed = if self.routes.get(&key).map(String::as_str) == Some(expected_model) {
            self.routes.remove(&key).is_some()
        } else {
            false
        };
        if removed {
            self.evict_remote_logs(node, worker);
            // Invalidate any in-flight inventory fetch; the caller refreshes after.
            self.bump_epoch(node);
        }
    }

    /// Derive the SERVED-instance-id → `(node, worker)` map from the live instance set. The
    /// nth instance of a model gets `org/model`, `org/model-1`, … . Computed over ALL
    /// instances (not just connected ones) so a disconnect never renumbers the survivors;
    /// pure, no persistence.
    ///
    /// Collision-free and deterministic: instances are assigned in sorted `(model, node,
    /// worker)` order against a global taken-set, and a candidate served id that is already
    /// taken (e.g. a model literally named `org/model-1` clashing with the suffix of a second
    /// `org/model` instance) bumps to the next free suffix. So EVERY live instance always gets
    /// a unique, reachable served id — no instance is ever left unaddressable (un-unloadable).
    fn served_ids(&self) -> HashMap<String, (NodeKey, WorkerId)> {
        // Sort by (model, node, worker) so the mapping is a stable function of the live set.
        let mut entries: Vec<(&str, &NodeKey, WorkerId)> = self
            .routes
            .iter()
            .map(|((node, worker), model)| (model.as_str(), node, *worker))
            .collect();
        entries.sort_unstable();

        let mut taken: HashSet<String> = HashSet::new();
        let mut next_suffix: HashMap<&str, usize> = HashMap::new();
        let mut out = HashMap::new();
        for (model, node, worker) in entries {
            let i = next_suffix.entry(model).or_insert(0);
            // Find the first free served id at or past this model's running suffix.
            let served = loop {
                let candidate = if *i == 0 {
                    model.to_string()
                } else {
                    format!("{model}-{i}")
                };
                *i += 1;
                if taken.insert(candidate.clone()) {
                    break candidate;
                }
            };
            out.insert(served, (node.clone(), worker));
        }
        out
    }

    /// Explicitly retire a node: a FULL removal (operator action). Drops its transport,
    /// instances, cached inventory, relayed-log rings, AND its durable `NodeId` slot — all
    /// atomically, so the node disappears from the fleet view entirely.
    fn retire(&mut self, node: &str) {
        if let Some(t) = self.nodes.remove(node) {
            t.close();
        }
        // Reclaim ALL of this node's relayed-log rings — including any worker displaced by a
        // reload that's no longer on a current route (a per-route walk would miss those).
        if let Some(nid) = self.node_ids.get(node) {
            self.bus.evict_node(nid);
        }
        self.routes.retain(|(n, _), _| n != node);
        // Bump so a refresh already in flight can't reinsert stale inventory after this.
        self.bump_epoch(node);
        self.inventories.remove(node);
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

    /// Served instance ids whose node is CURRENTLY connected (servable now), sorted. A served
    /// id whose node is disconnected is hidden — chat to it would fail HG027 until reconnect.
    fn routed_models(&self) -> Vec<String> {
        let mut v: Vec<_> = self
            .served_ids()
            .into_iter()
            .filter(|(_, (node, _))| self.nodes.contains_key(node))
            .map(|(served, _)| served)
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
            FleetMsg::ResolveServed { served, reply } => {
                let resolved = self.served_ids().get(&served).and_then(|(node, worker)| {
                    self.routes
                        .get(&(node.clone(), *worker))
                        .map(|model| (node.clone(), *worker, model.clone()))
                });
                let _ = reply.send(resolved);
            }
            FleetMsg::IsRemote { served, reply } => {
                let _ = reply.send(self.served_ids().contains_key(&served));
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
            FleetMsg::AddInstance {
                node,
                worker,
                model,
                reply,
            } => {
                self.routes.insert((node, worker), model);
                let _ = reply.send(());
            }
            FleetMsg::RemoveInstanceIf {
                node,
                worker,
                expected_model,
                reply,
            } => {
                self.remove_instance_if(&node, worker, &expected_model);
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
        // On (re)connect: refresh the inventory for the fleet view. Best-effort, off the hot
        // path. (Loads are additive — no displaced workers are ever owed to an offline node —
        // so there are no pending unloads to reconcile.)
        let inv_weak = Arc::downgrade(self);
        let inv_node = node.clone();
        tokio::spawn(async move {
            if let Some(fleet) = inv_weak.upgrade() {
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
    /// out of the fleet). Drops its transport, instances, cached inventory, relayed-log rings,
    /// AND its durable `NodeId` slot. Pairs with the hub removing it from the allowlist.
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

    /// Resolve a SERVED instance id to `(node, worker, raw_model)` or fail with HG002. The
    /// raw model (what the worker actually loaded) is needed for the worker's model-match
    /// check — the served id (`org/model-1`) would otherwise look like a mismatch (HG018).
    async fn require_served(
        &self,
        served: &str,
    ) -> Result<(NodeKey, WorkerId, String), HiggsError> {
        self.ask(|reply| FleetMsg::ResolveServed {
            served: served.to_string(),
            reply,
        })
        .await
        .flatten()
        .ok_or_else(|| HiggsError::ModelNotFound {
            id: served.to_string(),
        })
    }

    /// CAS instance-drop: remove `(node, worker)` only if it STILL serves `expected_model`; on
    /// removal also reclaim the worker's relayed-log ring and bump the node's generation.
    async fn remove_instance_if(&self, node: &str, worker: WorkerId, expected_model: &str) {
        self.ask(|reply| FleetMsg::RemoveInstanceIf {
            node: node.to_string(),
            worker,
            expected_model: expected_model.to_string(),
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

    /// Load `model` on `node` and record a NEW instance. Loads are ADDITIVE: each call spawns
    /// a fresh worker on the node and adds an instance, so N loads of the same model coexist
    /// as N served ids (`org/model`, `org/model-1`, …). Returns the new worker's id.
    pub async fn load(&self, node: &str, model: &str) -> Result<WorkerId, HiggsError> {
        let transport = self.transport(node).await?;
        let result = match transport.request(M_NODE_LOAD, json!({ "id": model })).await {
            Ok(v) => v,
            Err(e) => return Err(self.handle_op_error(node, &transport, e).await),
        };
        let worker = WorkerId(parse_worker_id(&result)?);
        self.ask(|reply| FleetMsg::AddInstance {
            node: node.to_string(),
            worker,
            model: model.to_string(),
            reply,
        })
        .await;
        // Refresh the fleet view from the node's authoritative state after the load lands.
        self.bump_epoch(node).await;
        let _ = self.refresh_inventory(node).await;
        Ok(worker)
    }

    /// Unload a served instance's remote worker and drop its route.
    pub async fn unload(&self, served: &str) -> Result<(), HiggsError> {
        self.unload_or_kill(served, M_NODE_UNLOAD).await
    }

    /// Force-kill a served instance's remote worker and drop its route.
    pub async fn kill(&self, served: &str) -> Result<(), HiggsError> {
        self.unload_or_kill(served, M_NODE_KILL).await
    }

    /// Shared unload/kill: resolve the served id to its exact instance, clear it on success OR
    /// when the node reports the worker already gone; node-down → HG027 (transport dropped).
    async fn unload_or_kill(&self, served: &str, method: &str) -> Result<(), HiggsError> {
        let (node, worker, model) = self.require_served(served).await?;
        let transport = self.transport(&node).await?;
        let res = transport
            .request(method, json!({ "worker_id": worker.0 }))
            .await;
        if res.is_ok() || res.as_ref().err().is_some_and(route_invalidating) {
            // remove_instance_if reclaims the log ring + bumps the node's generation.
            self.remove_instance_if(&node, worker, &model).await;
            // Re-sync the fleet view from the node's authoritative state.
            let _ = self.refresh_inventory(&node).await;
        }
        match res {
            Ok(_) => Ok(()),
            Err(e) => Err(self.handle_op_error(&node, &transport, e).await),
        }
    }

    /// Resolve a served instance id to its `(node, worker)`, if routed.
    pub async fn resolve(&self, served: &str) -> Option<(NodeKey, WorkerId)> {
        self.require_served(served)
            .await
            .ok()
            .map(|(node, worker, _model)| (node, worker))
    }

    /// Is this served instance id resident on some remote node?
    pub async fn is_remote(&self, served: &str) -> bool {
        self.ask(|reply| FleetMsg::IsRemote {
            served: served.to_string(),
            reply,
        })
        .await
        .unwrap_or(false)
    }

    /// Served instance ids whose node is CURRENTLY connected (for `/v1/models` discovery =
    /// servable now). A served id whose node is disconnected is hidden.
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

    /// Relay a chat to the remote worker for `served` (a served instance id). Returns the
    /// streamed-delta receiver + a future resolving to the final result. The RAW model the
    /// worker loaded is sent on the wire (the worker matches on that, not the served id). A
    /// worker-gone failure drops the instance (so retries re-resolve); a dead transport drops
    /// the transport (HG027).
    pub async fn chat(
        self: &Arc<Self>,
        served: &str,
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
        let (node, worker, model) = self.require_served(served).await?;
        let transport = self.transport(&node).await?;
        let (rx, fut) = match transport
            .chat(
                worker.0,
                model.clone(),
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
        let used = transport;
        let wrapped = async move {
            match fut.await {
                Ok(v) => Ok(v),
                Err(e) if route_invalidating(&e) => {
                    // Worker gone (node alive) — drop the instance so a retry re-resolves, and
                    // re-sync the fleet view (remove_instance_if bumped the node's generation).
                    fleet.remove_instance_if(&node, worker, &model).await;
                    let _ = fleet.refresh_inventory(&node).await;
                    Err(e)
                }
                // Transport-level / other failure surfacing mid-stream: drop the dead
                // transport (Arc-identity guarded) and remap to HG027. The instance is kept.
                Err(e) => Err(fleet.handle_op_error(&node, &used, e).await),
            }
        };
        Ok((rx, wrapped))
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
    async fn loads_are_additive_two_instances_get_distinct_served_ids() {
        let (fleet, node_key, model_id, _root) = fleet_with_one_node().await;
        let w1 = fleet.load(&node_key, &model_id).await.unwrap();
        let w2 = fleet.load(&node_key, &model_id).await.unwrap();
        assert_ne!(w1, w2, "each load spawns a distinct worker");
        // Two instances of the same model → two served ids: `org/model` and `org/model-1`,
        // assigned deterministically by (node, worker) order.
        let served = format!("{model_id}-1");
        let mut routed = fleet.routed_models().await;
        routed.sort();
        assert_eq!(routed, vec![model_id.clone(), served.clone()]);
        // Both served ids resolve to the SAME node but DIFFERENT workers.
        let r0 = fleet.resolve(&model_id).await.unwrap();
        let r1 = fleet.resolve(&served).await.unwrap();
        assert_eq!(r0.0, node_key);
        assert_eq!(r1.0, node_key);
        assert_ne!(r0.1, r1.1, "distinct workers behind the two served ids");
        assert_eq!(
            [r0.1, r1.1]
                .iter()
                .copied()
                .collect::<std::collections::HashSet<_>>(),
            [w1, w2]
                .iter()
                .copied()
                .collect::<std::collections::HashSet<_>>(),
        );
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

    #[test]
    fn served_ids_are_collision_free_even_when_a_model_name_clashes_with_a_suffix() {
        // Two `org/model` instances want served ids `org/model` + `org/model-1`; a literal
        // model `org/model-1` also wants `org/model-1`. Every instance must still get a UNIQUE
        // reachable served id (no orphaned, un-unloadable worker).
        let n = |w| ("nodeA".to_string(), WorkerId(w));
        let mut routes = HashMap::new();
        routes.insert(n(1), "org/model".to_string());
        routes.insert(n(2), "org/model".to_string());
        routes.insert(n(3), "org/model-1".to_string());
        let actor = FleetActor {
            nodes: HashMap::new(),
            routes,
            node_ids: NodeIdAllocator::new(),
            inventories: HashMap::new(),
            epochs: HashMap::new(),
            bus: Arc::new(crate::log_bus::LogBus::new()),
        };
        let served = actor.served_ids();
        // Three instances → three distinct served ids, each mapping to a distinct worker.
        assert_eq!(served.len(), 3, "every instance is reachable: {served:?}");
        let workers: std::collections::HashSet<_> = served.values().map(|(_, w)| *w).collect();
        assert_eq!(workers.len(), 3, "no two served ids share a worker");
        // Deterministic: re-deriving yields the same mapping.
        assert_eq!(actor.served_ids(), served);
    }

    #[tokio::test]
    async fn unloading_one_instance_renumbers_the_survivor_to_the_base_served_id() {
        let (fleet, node_key, model_id, _root) = fleet_with_one_node().await;
        let _w1 = fleet.load(&node_key, &model_id).await.unwrap();
        let _w2 = fleet.load(&node_key, &model_id).await.unwrap();
        let suffixed = format!("{model_id}-1");
        assert!(fleet.is_remote(&suffixed).await, "two instances present");

        // Unload the BASE served id (the lower (node, worker)). The surviving instance
        // re-derives to the base served id (no stale `-1` left dangling).
        fleet.unload(&model_id).await.unwrap();
        let routed = fleet.routed_models().await;
        assert_eq!(
            routed,
            vec![model_id.clone()],
            "one instance left, served under the base id: {routed:?}"
        );
        assert!(
            !fleet.is_remote(&suffixed).await,
            "the `-1` served id is gone after one instance is unloaded"
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
