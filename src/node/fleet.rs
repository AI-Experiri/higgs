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

use std::collections::HashMap;
use std::sync::Arc;

use iroh::endpoint::Connection;
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::sync::oneshot;

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
/// The hub's UI/API view of one node: its stable id, endpoint, whether it's currently
/// connected, its operator label, whether it is the LOCAL machine, and its last-fetched
/// inventory (host + resident workers + hardware/runtime). The Fleet UI renders local and
/// remote nodes with this one shape.
#[derive(Debug, Clone, serde::Serialize)]
pub struct NodeView {
    pub node_id: u32,
    pub endpoint_id: String,
    pub connected: bool,
    /// The local machine the server runs on (`true`) vs a paired remote node (`false`). The UI
    /// hides Retire/Leave for the local node — there is nothing to un-pair.
    pub is_local: bool,
    /// Human label for the node. For a remote node this is the hub's allowlist label (the
    /// node's friendly name, operator-renamable) — the fleet leaves it empty and the serve layer
    /// fills it from the allowlist (the editable source of truth). For the local node it is the
    /// instance's `config.json` name.
    pub label: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[ts(optional)]
    pub inventory: Option<NodeInventory>,
    /// Age of the cached `inventory` snapshot in ms at the moment this view was
    /// taken, computed hub-side from a monotonic stamp (T14, the T9 freshness
    /// residual's fix). The hub re-pulls inventory on connect and after lifecycle
    /// ops but NEVER on chats — a remote row's idle/in-flight stats are only as
    /// fresh as this says. Absent for the LIVE local card (its stats are read per
    /// request) and when there is no cached inventory.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[ts(optional, type = "number")]
    pub inventory_age_ms: Option<u64>,
    /// The node's self-reported higgs semver from its HELLO at (re)admission
    /// (T14). Refreshed only by a reconnect — a node upgraded while admitted
    /// shows its old version until it reconnects. Absent when no HELLO carried
    /// one (pre-T14 admit paths) — the UI omits it, never fabricates.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[ts(optional)]
    pub software_version: Option<String>,
    /// The HELLO-negotiated wire-protocol major for the node's CURRENT admission
    /// (T14) — the same slot the per-load params gate reads ([HG078]). Absent =
    /// the admission predates version reporting (effectively the floor, 1) or
    /// the local card (no wire, no negotiation).
    #[serde(skip_serializing_if = "Option::is_none")]
    #[ts(optional)]
    pub protocol: Option<u32>,
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
    /// A single node's served instance ids (routes are durable, so a disconnected
    /// node still reports its instances), sorted for determinism.
    ServedOn {
        node: NodeKey,
        reply: oneshot::Sender<Vec<String>>,
    },
    Transport {
        node: NodeKey,
        reply: oneshot::Sender<Result<Arc<NodeTransport>, HiggsError>>,
    },
    /// The node's HELLO-negotiated protocol major, `None` if never admitted with one.
    NodeProtocol {
        node: NodeKey,
        reply: oneshot::Sender<Option<u32>>,
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
    /// Atomically admit a node: if `gen` is current (or `None` = unconditional, for direct/test
    /// callers), assign its stable `NodeId`, insert the transport, and bump the epoch — replying
    /// with `Some((node_id, replaced))` (the prior transport, if any, for the caller to close).
    /// If `gen` is stale (the kill switch bumped the generation: a disable raced this admission),
    /// reply `None` — nothing is assigned or inserted, so the caller closes the refused transport
    /// and skips all per-connection bookkeeping. One message → no TOCTOU between the generation
    /// check, the node-id assign, and the transport insert.
    AdmitNode {
        node: NodeKey,
        transport: Arc<NodeTransport>,
        /// The accept loop's admission generation; `None` = admit unconditionally.
        gen: Option<u64>,
        /// The HELLO-negotiated protocol major for THIS connection, `None` when the
        /// caller has none (tests/direct) — read back as the conservative floor 1.
        agreed_version: Option<u32>,
        /// The node's self-reported semver from its HELLO (T14 Fleet-view display),
        /// `None` when the caller has none (tests/direct). Same lifecycle as
        /// `agreed_version`: refreshed on every (re)admission, cleared at retire.
        software_version: Option<String>,
        #[allow(clippy::type_complexity)]
        reply: oneshot::Sender<Option<(NodeId, Option<Arc<NodeTransport>>)>>,
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
    /// Close + forget EVERY node's transport (mark all disconnected) AND disarm new-node
    /// admission, WITHOUT touching routes, inventories, or node-ids — so a hub re-enable /
    /// reconnect restores the fleet with its previously-loaded routes intact. Used by the hub
    /// kill switch.
    DisconnectAll {
        reply: oneshot::Sender<()>,
    },
    /// Bump the admission generation and return the NEW value — a fresh generation for an accept
    /// loop about to be spawned (`start_hub`). Invalidates any prior loop's in-flight admissions.
    BumpAdmitGen {
        reply: oneshot::Sender<u64>,
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
        /// Dual-clock stamp of the pull's START (see [`PulledAt`]).
        pulled_at: PulledAt,
        reply: oneshot::Sender<()>,
    },
    BumpEpoch {
        node: NodeKey,
        reply: oneshot::Sender<()>,
    },
    /// Chat-end refresh debounce, phase 1: may THIS completion own the (single)
    /// refresh for `node`? `true` = caller schedules the pull; `false` = one is
    /// already scheduled/running — this completion was coalesced into its
    /// trailing re-run flag instead of spawning another pull.
    ChatRefreshBegin {
        node: NodeKey,
        reply: oneshot::Sender<bool>,
    },
    /// Chat-end refresh debounce, phase 2: the owned pull finished. `true` =
    /// completions were coalesced meanwhile — run ONCE more (the flag resets);
    /// `false` = done, the slot is released.
    ChatRefreshEnd {
        node: NodeKey,
        reply: oneshot::Sender<bool>,
    },
}

/// Private actor state: the hub's whole fleet read-model + routing table, owned by one task.
/// When an inventory pull STARTED, on BOTH clocks. Neither clock alone is
/// trustworthy for an age: macOS `Instant` (CLOCK_UPTIME_RAW) FREEZES across
/// system sleep (a slept-through snapshot would read fresh), and `SystemTime`
/// zeroes out after a backward NTP/manual step (a stale snapshot would read
/// freshly synced until the clock catches up). Each stamp is a LOWER BOUND on
/// the true age, so the view takes the MAX of the two, which covers every
/// SINGLE-fault case. Accepted residual (T14 r3): a COMPOUND fault inside one
/// cache window — a backward wall step AND a sleep — under-reads on both
/// clocks at once (no std clock counts time across sleep monotonically), and
/// a forward wall step overstates; any lifecycle op or hub-routed chat
/// re-stamps and self-heals both.
#[derive(Debug, Clone, Copy)]
struct PulledAt {
    wall: std::time::SystemTime,
    mono: std::time::Instant,
}

impl PulledAt {
    fn now() -> Self {
        Self {
            wall: std::time::SystemTime::now(),
            mono: std::time::Instant::now(),
        }
    }
    /// Age in ms: the max of both clocks' elapsed (each saturating at 0).
    fn age_ms(&self) -> u64 {
        let wall = std::time::SystemTime::now()
            .duration_since(self.wall)
            .unwrap_or_default();
        wall.max(self.mono.elapsed()).as_millis() as u64
    }
}

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
    /// Last-fetched inventory per node (host + workers + hw/rt) with the DUAL
    /// stamp of when the pull STARTED. The node's reply is authoritative;
    /// refreshed on connect, after every hub-driven lifecycle change, and
    /// (debounced) after hub-routed chats complete — the stamp is what lets
    /// `nodes_view` report how stale the snapshot is (`inventory_age_ms`, the
    /// T9-residual fix). Stamped at fetch START (not commit) so a slow pull
    /// doesn't under-age data that was already old when it arrived.
    inventories: HashMap<NodeKey, (NodeInventory, PulledAt)>,
    /// Per-node chat-refresh debounce state (see [`FleetMsg::ChatRefreshBegin`]):
    /// present = a chat-end inventory refresh is scheduled or running for the
    /// node; the bool = a LATER chat completed meanwhile, so run once more when
    /// the current pull finishes (trailing coalescing). Bounds the chat-driven
    /// pulls to at most ONE in flight per node however bursty the chats.
    chat_refreshes: HashMap<NodeKey, bool>,
    /// The HELLO-negotiated protocol major per node, refreshed on every (re)admission —
    /// what feature gating (per-load params) reads today; a Fleet-view version display
    /// (T14) would read the same slot but does not exist yet. Absent until the
    /// node's first admission carries one; readers treat absent as the floor (1).
    versions: HashMap<NodeKey, u32>,
    /// The node's self-reported higgs semver (from its HELLO), per node — the Fleet
    /// view's version display (T14). Same lifecycle as `versions`: refreshed on every
    /// (re)admission (a node UPGRADED while admitted keeps its old value until it
    /// reconnects — the HELLO is the only source), cleared at retire. Absent for a
    /// node admitted by a pre-T14 caller path.
    software_versions: HashMap<NodeKey, String>,
    /// Per-node lifecycle generation, bumped on every load/unload/kill/route-drop. A
    /// `refresh_inventory` only commits its (possibly stale) result if this is unchanged
    /// since it started — so a slow connect-time fetch can't clobber a newer state.
    epochs: HashMap<NodeKey, u64>,
    /// The hub's Developer-Log bus: relayed remote worker stderr lands here under
    /// `LogSource::RemoteWorker { node, worker }` so it shares the operator's log console.
    bus: Arc<LogBus>,
    /// Monotonic ADMISSION GENERATION. Each hub accept loop is spawned bound to the generation
    /// current when it started (`bump_admit_gen`); every `AdmitNode` it issues carries that gen
    /// and is admitted only while it still equals this. The kill switch's `disconnect_all` bumps
    /// the generation (atomically with draining transports), so an admission task spawned by the
    /// now-closing accept loop that races past disable is REFUSED — and stays refused even after a
    /// quick re-enable spawns a fresh loop at a newer gen (the stale task's gen can never match
    /// again). Direct callers (tests) pass `None` to admit unconditionally.
    admit_gen: u64,
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
    /// Collision-free and deterministic — see [`crate::node::served::served_ids`], the shared
    /// algorithm reused by the local engine (P4b).
    fn served_ids(&self) -> HashMap<String, (NodeKey, WorkerId)> {
        let instances: Vec<(NodeKey, WorkerId, String)> = self
            .routes
            .iter()
            .map(|((node, worker), model)| (node.clone(), *worker, model.clone()))
            .collect();
        crate::node::served::served_ids(&instances)
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
        self.versions.remove(node);
        self.software_versions.remove(node);
        // Forget the durable id slot so the node leaves the fleet view (not left disconnected).
        self.node_ids.remove(node);
    }

    /// The fleet view, taken as ONE atomic snapshot across node-ids / nodes / inventories.
    /// Each resident worker is tagged with its hub-assigned served id (`org/model`,
    /// `org/model-1`, …) so the UI can show exactly what clients call to reach it.
    fn nodes_view(&self) -> Vec<NodeView> {
        // Invert served → (node, worker) into (node, worker) → served, so we can tag each
        // worker in each node's inventory with the id clients address it by. Computed once
        // per snapshot over ALL instances (connected or not) — same set `served_ids` uses.
        let rev: HashMap<(NodeKey, WorkerId), String> = self
            .served_ids()
            .into_iter()
            .map(|(served, key)| (key, served))
            .collect();
        self.node_ids
            .all()
            .into_iter()
            .map(|(endpoint_id, node_id)| {
                let mut inventory_age_ms = None;
                let inventory =
                    self.inventories
                        .get(&endpoint_id)
                        .cloned()
                        .map(|(mut inv, pulled_at)| {
                            // How stale this snapshot is, computed hub-side at VIEW
                            // time from BOTH clocks (see `PulledAt` — max of two
                            // lower bounds, errs stale-ward only).
                            inventory_age_ms = Some(pulled_at.age_ms());
                            for w in &mut inv.workers {
                                let key = (endpoint_id.clone(), WorkerId(w.worker_id));
                                w.served_id =
                                    served_id_for_worker(&key, &w.model, &self.routes, &rev);
                            }
                            inv
                        });
                NodeView {
                    node_id: node_id.0,
                    connected: self.nodes.contains_key(&endpoint_id),
                    // The fleet only tracks remote nodes; the local node is prepended by the serve
                    // layer. `label` is filled there from the allowlist (live, rename-aware).
                    is_local: false,
                    label: String::new(),
                    inventory,
                    inventory_age_ms,
                    software_version: self.software_versions.get(&endpoint_id).cloned(),
                    protocol: self.versions.get(&endpoint_id).copied(),
                    endpoint_id,
                }
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
                // Counts a served id with ANY durable route — including one whose node is
                // currently DISCONNECTED (routes survive a drop; see `drop_transport_if`).
                // This is intentional, NOT the connected-only `routed_models` set:
                //   * a chat to a disconnected-route id routes through `chat`, whose
                //     `transport(&node)` then returns HG027 (node unreachable → reconnect) —
                //     an ACCURATE error (the model IS fleet-known, its host is merely offline),
                //     strictly better than the HG002 "not found" a connected-only check would
                //     yield for a model the fleet still routes.
                // Residual (accepted): if the SAME model is also on local disk, a stale route
                // to a disconnected node yields HG027 instead of a local JIT load. Narrow, and
                // it degrades to a clear diagnostic — not a fault. See `is_remote`'s doc.
                let _ = reply.send(self.served_ids().contains_key(&served));
            }
            FleetMsg::RoutedModels { reply } => {
                let _ = reply.send(self.routed_models());
            }
            FleetMsg::ServedOn { node, reply } => {
                let mut v: Vec<_> = self
                    .served_ids()
                    .into_iter()
                    .filter(|(_, (n, _))| *n == node)
                    .map(|(served, _)| served)
                    .collect();
                v.sort();
                let _ = reply.send(v);
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
            FleetMsg::NodeProtocol { node, reply } => {
                let _ = reply.send(self.versions.get(&node).copied());
            }
            FleetMsg::SeedNode { node, reply } => {
                self.node_ids.assign(&node);
                let _ = reply.send(());
            }
            FleetMsg::AdmitNode {
                node,
                transport,
                gen,
                agreed_version,
                software_version,
                reply,
            } => {
                if matches!(gen, Some(g) if g != self.admit_gen) {
                    // Stale generation: this admission task belongs to an accept loop the kill
                    // switch already invalidated (a disable raced it; even a later re-enable
                    // spawns a NEWER gen). REFUSE atomically — assign nothing, insert nothing;
                    // the caller closes the refused transport. Keeps the kill switch airtight:
                    // a disabled hub neither connects nor seeds the node.
                    let _ = reply.send(None);
                } else {
                    let node_id = self.node_ids.assign(&node);
                    match agreed_version {
                        Some(v) => {
                            self.versions.insert(node.clone(), v);
                        }
                        // A version-less re-admission must not inherit the
                        // PREVIOUS connection's major — clear to the floor, as
                        // the field doc promises.
                        None => {
                            self.versions.remove(&node);
                        }
                    }
                    // Same rule for the semver display: never inherit a prior
                    // connection's value.
                    match software_version {
                        Some(v) => {
                            self.software_versions.insert(node.clone(), v);
                        }
                        None => {
                            self.software_versions.remove(&node);
                        }
                    }
                    let replaced = self.nodes.insert(node.clone(), transport);
                    // Bump on (re)admission so an inventory fetch from a PRIOR connection still in
                    // flight can't commit its now-stale result over this fresh one.
                    self.bump_epoch(&node);
                    let _ = reply.send(Some((node_id, replaced)));
                }
            }
            FleetMsg::BumpAdmitGen { reply } => {
                self.admit_gen = self.admit_gen.wrapping_add(1);
                let _ = reply.send(self.admit_gen);
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
            FleetMsg::DisconnectAll { reply } => {
                // Bump the admission generation FIRST (same message, so it's atomic with the
                // drain): any in-flight admission task from the now-closing accept loop that
                // calls `AdmitNode` after this carries the OLD gen and is refused, not
                // resurrected — and a later re-enable spawns an even newer gen, so it can never
                // match again.
                self.admit_gen = self.admit_gen.wrapping_add(1);
                // Drain transports (closing each so a wedged-open connection's close-watcher
                // wakes) but KEEP routes/inventories/node-ids — same "routes survive a dropped
                // connection" contract as `drop_transport_if`, applied to every node at once.
                for (node, t) in self.nodes.drain() {
                    tracing::info!(
                        node,
                        "higgs hub: disabling — node transport closed (route kept)"
                    );
                    t.close();
                }
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
                pulled_at,
                reply,
            } => {
                if self.epoch(&node) == epoch_before {
                    // The stamp is the pull's START (carried in, not taken here) —
                    // `nodes_view` reports the snapshot's age from it.
                    self.inventories.insert(node, (*inventory, pulled_at));
                }
                let _ = reply.send(());
            }
            FleetMsg::BumpEpoch { node, reply } => {
                self.bump_epoch(&node);
                let _ = reply.send(());
            }
            FleetMsg::ChatRefreshBegin { node, reply } => {
                let owned = match self.chat_refreshes.entry(node) {
                    std::collections::hash_map::Entry::Vacant(e) => {
                        e.insert(false);
                        true
                    }
                    std::collections::hash_map::Entry::Occupied(mut e) => {
                        // Coalesce: the running pull will go once more.
                        *e.get_mut() = true;
                        false
                    }
                };
                let _ = reply.send(owned);
            }
            FleetMsg::ChatRefreshEnd { node, reply } => {
                let again = match self.chat_refreshes.entry(node) {
                    std::collections::hash_map::Entry::Occupied(mut e) => {
                        if *e.get() {
                            *e.get_mut() = false;
                            true
                        } else {
                            e.remove();
                            false
                        }
                    }
                    // Slot vanished (e.g. a retire cleared state) — nothing owed.
                    std::collections::hash_map::Entry::Vacant(_) => false,
                };
                let _ = reply.send(again);
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
            chat_refreshes: HashMap::new(),
            software_versions: HashMap::new(),
            versions: HashMap::new(),
            epochs: HashMap::new(),
            bus: bus_for_actor,
            admit_gen: 0,
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
    pub async fn add_node(
        self: &Arc<Self>,
        node: NodeKey,
        transport: Arc<NodeTransport>,
        admit_gen: Option<u64>,
        agreed_version: Option<u32>,
        software_version: Option<String>,
    ) {
        // Atomically admit: assign the stable NodeId, insert the transport, and bump the epoch in
        // ONE actor message, gated by the admission generation. If the kill switch bumped the
        // generation (a disable raced this admission), the admit is REFUSED — close the transport
        // and skip ALL per-connection bookkeeping, so a disabled hub neither connects nor seeds
        // the node into the fleet view. `admit_gen = None` admits unconditionally (direct/test
        // callers that aren't the kill-switch-gated accept loop).
        let admitted = self
            .ask(|reply| FleetMsg::AdmitNode {
                node: node.clone(),
                transport: transport.clone(),
                gen: admit_gen,
                agreed_version,
                software_version,
                reply,
            })
            .await
            .flatten();
        let Some((node_id, replaced)) = admitted else {
            transport.close(); // refused (hub disabled mid-admission)
            return;
        };
        if let Some(old) = replaced {
            old.close(); // free the old connection + wake its close-watcher
        }
        // Read the node's relayed worker stderr (its uni stream) into the hub bus for THIS
        // connection; ends when the connection closes (accept_uni errors).
        tokio::spawn(read_remote_logs(
            transport.connection(),
            node_id,
            self.bus.clone(),
        ));

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

    /// Schedule the DEBOUNCED chat-end inventory refresh for `node` (T14 r2):
    /// if no chat-refresh is scheduled/running for the node, own the slot and
    /// spawn ONE detached pull after a short settle delay (the node records its
    /// `ChatEnd` when the chat's lease drops, immediately after the reply — the
    /// delay lets that land so the snapshot doesn't still say `in_flight`);
    /// otherwise coalesce into the running pull's trailing re-run. Best-effort:
    /// pull errors are dropped (the next lifecycle op or chat re-pulls).
    fn schedule_chat_refresh(self: &Arc<Self>, node: NodeKey) {
        const SETTLE: std::time::Duration = std::time::Duration::from_millis(250);
        let fleet = self.clone();
        tokio::spawn(async move {
            let owned = fleet
                .ask(|reply| FleetMsg::ChatRefreshBegin {
                    node: node.clone(),
                    reply,
                })
                .await
                .unwrap_or(false);
            if !owned {
                return; // coalesced into the owner's trailing re-run
            }
            loop {
                tokio::time::sleep(SETTLE).await;
                let _ = fleet.refresh_inventory(&node).await;
                let again = fleet
                    .ask(|reply| FleetMsg::ChatRefreshEnd {
                        node: node.clone(),
                        reply,
                    })
                    .await
                    .unwrap_or(false);
                if !again {
                    break;
                }
            }
        });
    }

    /// Fetch `node`'s inventory over its live transport and cache it for the fleet view. The
    /// node's reply is AUTHORITATIVE and stored verbatim — except a result is dropped if a
    /// lifecycle op changed this node's generation while the request was in flight, so a slow
    /// connect-time fetch can never resurrect a stale worker list.
    pub async fn refresh_inventory(&self, node: &str) -> Result<NodeInventory, HiggsError> {
        let epoch_before = self.epoch(node).await;
        // Stamp BEFORE the RPC: the node captures its stats at request time, so a
        // slow pull must not make already-old data read fresh at commit.
        let pulled_at = PulledAt::now();
        let transport = self.transport(node).await?;
        let value = match transport.request(M_NODE_INVENTORY, json!({})).await {
            Ok(v) => v,
            Err(e) => return Err(self.handle_op_error(node, &transport, e).await),
        };
        let inventory: NodeInventory =
            serde_json::from_value(value).map_err(|e| HiggsError::ProtocolViolation {
                peer_role: "node".into(),
                detail: format!("M_NODE_INVENTORY reply did not decode: {e}"),
            })?;
        // Commit only if no lifecycle op superseded us (the check+store is one message).
        self.ask(|reply| FleetMsg::CommitInventory {
            node: node.to_string(),
            epoch_before,
            inventory: Box::new(inventory.clone()),
            pulled_at,
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

    /// Close every node's transport and mark all nodes disconnected, WITHOUT dropping routes,
    /// inventories, or node-ids. Used by the hub kill switch: disabling stops all node network
    /// activity but keeps the route table, so re-enabling is a pure reconnect (previously-loaded
    /// remote routes survive — see `tests/remote_hub_e2e.rs::node_reconnects_and_route_survives`).
    pub async fn disconnect_all(&self) {
        self.ask(|reply| FleetMsg::DisconnectAll { reply }).await;
    }

    /// Bump the admission generation and return the fresh value, for an accept loop about to be
    /// spawned (`start_hub`). The loop binds to this gen and passes it to every `add_node`; a
    /// prior loop's in-flight admissions (older gen) are thereby invalidated. The kill switch's
    /// `disconnect_all` also bumps the gen, so a disabled fleet refuses stale admissions until
    /// the next enable spawns a loop at a newer gen.
    pub async fn bump_admit_gen(&self) -> u64 {
        self.ask(|reply| FleetMsg::BumpAdmitGen { reply })
            .await
            .unwrap_or(0)
    }

    /// A node's HELLO-negotiated protocol major, `None` if it was never admitted
    /// with one (a pre-plumbing admission or a direct/test `add_node`). Callers
    /// gating features treat `None` as the conservative floor (version 1).
    pub async fn node_protocol(&self, node: &str) -> Option<u32> {
        self.ask(|reply| FleetMsg::NodeProtocol {
            node: node.to_string(),
            reply,
        })
        .await
        .flatten()
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
    ///
    /// `params = None` — or a `Some(p)` that NORMALIZES to nothing: all fields
    /// `None`, count-zeros (ctx/threads and the rich thread/batch counts, which
    /// read as "auto"), and a no-override rich object all collapse to the
    /// classic bare `{ "id" }` (works against ANY node, never version-gated).
    /// A `Some(p)` with anything real left after normalization sends the full
    /// [`NodeLoadParams`](crate::remote::NodeLoadParams) — with `p.id`
    /// OVERWRITTEN by `model` (the route is recorded under `model`, so a
    /// divergent caller id would make the node load one model while the served
    /// id names another). Connectivity is checked FIRST: an offline/unknown
    /// node is HG027 even for a params-load, never a version refusal —
    /// gated on the node having negotiated protocol major ≥ 2 ([HG078] otherwise:
    /// some major-1 builds would parse the fields and older ones hard-reject —
    /// indistinguishable from here — and silently loading with the node's
    /// defaults when the caller asked for specific params would misreport what
    /// is running). An admission
    /// that predates version plumbing reads as the conservative floor (1).
    pub async fn load(
        &self,
        node: &str,
        model: &str,
        params: Option<crate::remote::NodeLoadParams>,
    ) -> Result<WorkerId, HiggsError> {
        // Connectivity FIRST: an offline/unknown node must surface as HG027,
        // not as a version refusal — after a hub cold start, seeded nodes are
        // version-less until they reconnect, and "update the node" would be
        // false advice for a current node that is merely disconnected.
        //
        // Accepted residual: `transport()` and `node_protocol()` are two actor
        // round trips, so a RETIRE landing between them clears the version and
        // a live params-load reads the floor → a spurious [HG078] refusal
        // (wrong refusal CLASS; never a wrong send — the retire also closed
        // the transport, so a retry gets HG027). The reverse interleaving
        // (re-admit with a newer version) sends correctly or fails HG027 on
        // the dead transport.
        let transport = self.transport(node).await?;
        // Normalize count-zeros to ABSENT before anything else, so the wire is
        // coherent regardless of which node build receives it (an older
        // major-2 node would coerce ctx 0 to a hardcoded 4096; a current one
        // reads it as absent — after this, neither ever sees a zero). A
        // normalized-empty params then falls into the bare arm below.
        // GpuLayers::Count{n:0} is deliberately untouched: an explicit 0 is
        // the meaningful CPU-only request, not an auto spelling.

        let params = params.map(|mut p| {
            p.ctx_len = p.ctx_len.filter(|c| *c > 0);
            p.threads = p.threads.filter(|t| *t > 0);
            // Rich count-zeros are the same "asking nothing" as base zeros —
            // the NODE strips them as absent anyway — so normalize them here
            // too, BEFORE the emptiness filter below: a rich object whose only
            // content is a zero count must not draw a version refusal.
            if let Some(lc) = p.params.as_mut() {
                lc.n_batch = lc.n_batch.filter(|n| *n > 0);
                lc.n_ubatch = lc.n_ubatch.filter(|n| *n > 0);
                lc.n_seq_max = lc.n_seq_max.filter(|n| *n > 0);
                lc.n_threads_batch = lc.n_threads_batch.filter(|n| *n > 0);
            }
            // An all-default rich object asks nothing either (`{"params": {}}`
            // parses to a default LlamaCppParams) — drop it so the bare arm's
            // "never version-refused for asking nothing" principle holds for
            // the rich field too, not just for absent/zero base fields.
            p.params = p
                .params
                .filter(crate::worker::engine::llamacpp::params::LlamaCppParams::has_overrides);
            p
        });
        let payload = match params {
            None => json!({ "id": model }),
            // ALL-None params serialize byte-identically to a bare load
            // (`skip_serializing_if` on every field) — treat them AS one, so a
            // caller's `params: {}` is never version-refused for asking nothing.
            // Exhaustive destructuring: adding a field to NodeLoadParams breaks
            // THIS match at compile time, so the emptiness check can never
            // silently ignore (and bare-drop) a new parameter.
            Some(crate::remote::NodeLoadParams {
                id: _,
                ctx_len: None,
                gpu_layers: None,
                threads: None,
                params: None,
            }) => {
                json!({ "id": model })
            }
            Some(mut p) => {
                let agreed = self.node_protocol(node).await.unwrap_or(1);
                if agreed < 2 {
                    return Err(HiggsError::NodeTooOldForParams {
                        endpoint_id: node.to_owned(),
                        agreed,
                    });
                }
                // The route below is recorded under `model` — force the wire id
                // to match so a direct caller with a divergent `p.id` cannot
                // make the node load one model while the served id names
                // another (the facade forces this too; here it is load-bearing
                // for `pub` callers).
                p.id = model.to_owned();
                serde_json::to_value(&p).map_err(|e| HiggsError::InternalFault {
                    context: "encode NodeLoadParams".into(),
                    detail: e.to_string(),
                })?
            }
        };
        let result = match transport.request(M_NODE_LOAD, payload).await {
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
    ///
    /// Known residual (pre-existing, documented per the chat-test round that
    /// closed the same class for chat): the caller's served id comes from ITS
    /// OWN earlier snapshot, and served ids renumber over a model's whole
    /// instance set — so an id captured before a concurrent unload/load can
    /// resolve to a DIFFERENT instance here and unload the wrong worker with a
    /// 200. Chat closed this with [`chat_pinned`](Self::chat_pinned); unload
    /// has no pinned variant yet (needs the same expected-node lever; candidate
    /// for the next fleet-surface task).
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

    /// Is this served instance id resident on some remote node? TRUE for ANY durable
    /// route, even to a currently-DISCONNECTED node (routes outlive a transport drop) —
    /// deliberately broader than [`routed_models`](Self::routed_models) (connected-only,
    /// for `/v1/models` discovery). Used by routing (`ensure_loaded` / `chat_stream`):
    /// a disconnected-route chat routes through [`chat`](Self::chat) → HG027 (node
    /// unreachable → reconnect), the accurate error for a fleet-known but offline host,
    /// vs. the misleading HG002 a connected-only check would give. See the `IsRemote`
    /// handler for the narrow accepted residual (stale route shadowing a local JIT load).
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

    /// One node's served instance ids, sorted. Derived from the DURABLE route set
    /// (like [`is_remote`](Self::is_remote), not the connected-only
    /// [`routed_models`](Self::routed_models)): a disconnected node still reports
    /// its instances, so a caller picking a chat-test target gets HG027 from the
    /// chat itself (accurate: node offline) rather than a misleading "nothing
    /// served" for a node that merely dropped its transport. Empty for an unknown
    /// node.
    pub async fn served_on(&self, node: &str) -> Vec<String> {
        self.ask(|reply| FleetMsg::ServedOn {
            node: node.to_string(),
            reply,
        })
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
        chat_template_kwargs: Option<String>,
    ) -> Result<
        (
            crate::delta_queue::DeltaReceiver,
            impl std::future::Future<Output = Result<serde_json::Value, HiggsError>> + Send,
        ),
        HiggsError,
    > {
        self.chat_inner(
            served,
            None,
            messages_json,
            max_tokens,
            temperature,
            tools_json,
            chat_template_kwargs,
        )
        .await
    }

    /// [`chat`](Self::chat), but REFUSES to dispatch unless the served id still
    /// resolves to `pin_node` — for callers whose result ATTESTS which node
    /// answered (the node chat test). Served ids are derived by renumbering the
    /// model's whole instance set ([`crate::node::served::served_ids`]), so a
    /// single unload or additive load elsewhere can re-home an id between a
    /// caller's own resolution and this dispatch; the pin is checked against the
    /// SAME resolution that picks the transport, so a re-homed (or freshly
    /// unrouted) id is refused ([HG077], transient — the caller re-resolves)
    /// instead of silently exercising, and then reporting, the wrong node.
    pub async fn chat_pinned(
        self: &Arc<Self>,
        served: &str,
        pin_node: &str,
        messages_json: String,
        max_tokens: usize,
        temperature: f32,
        tools_json: Option<String>,
        chat_template_kwargs: Option<String>,
    ) -> Result<
        (
            crate::delta_queue::DeltaReceiver,
            impl std::future::Future<Output = Result<serde_json::Value, HiggsError>> + Send,
        ),
        HiggsError,
    > {
        self.chat_inner(
            served,
            Some(pin_node),
            messages_json,
            max_tokens,
            temperature,
            tools_json,
            chat_template_kwargs,
        )
        .await
    }

    /// Shared body of [`chat`](Self::chat) / [`chat_pinned`](Self::chat_pinned):
    /// one `require_served` resolution drives the pin check (when any) AND the
    /// transport pick, so the two can never disagree.
    async fn chat_inner(
        self: &Arc<Self>,
        served: &str,
        pin_node: Option<&str>,
        messages_json: String,
        max_tokens: usize,
        temperature: f32,
        tools_json: Option<String>,
        chat_template_kwargs: Option<String>,
    ) -> Result<
        (
            crate::delta_queue::DeltaReceiver,
            impl std::future::Future<Output = Result<serde_json::Value, HiggsError>> + Send,
        ),
        HiggsError,
    > {
        // Under a pin, BOTH stale-pick shapes are the same transient conflict
        // ([HG077]): the id re-homed to another node, or it unrouted entirely.
        // Without the remap the unroute shape would surface as HG002 ("not
        // found on disk" — no disk was consulted) and jump status class.
        let resolved = self.require_served(served).await;
        let (node, worker, model) = match (resolved, pin_node) {
            (Ok(r), _) => r,
            (Err(HiggsError::ModelNotFound { .. }), Some(pin)) => {
                // No fabricated history: a DIRECT chat_pinned caller may hit this
                // on a first call with a bad id, not only via a concurrent unroute.
                return Err(HiggsError::ChatTestTargetMoved {
                    detail: format!(
                        "served instance {served} is not routed on any node at dispatch \
                         (pinned to node {pin})"
                    ),
                });
            }
            (Err(e), _) => return Err(e),
        };
        if let Some(pin) = pin_node {
            if node != pin {
                return Err(HiggsError::ChatTestTargetMoved {
                    detail: format!(
                        "served instance {served} resolved to node {node} at dispatch, not the \
                         pinned node {pin}"
                    ),
                });
            }
        }
        let transport = self.transport(&node).await?;
        let (rx, fut) = match transport
            .chat(
                worker.0,
                model.clone(),
                messages_json,
                max_tokens,
                temperature,
                tools_json,
                chat_template_kwargs,
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
                Ok(v) => {
                    // A completed hub-routed chat is activity the hub KNOWS about —
                    // re-pull the node's inventory so the fleet view's idle/
                    // in-flight stats and their age reflect it instead of aging a
                    // pre-chat snapshot (T14 r1). DEBOUNCED per node (r2): at most
                    // ONE pull in flight however bursty the chats — later
                    // completions coalesce into a single trailing re-run — and each
                    // pull waits a short settle so the node's own ChatEnd
                    // bookkeeping (recorded when its lease drops, just after the
                    // reply) lands before the snapshot is taken. Epoch-gating in
                    // refresh_inventory keeps a racing lifecycle op authoritative.
                    fleet.schedule_chat_refresh(node.clone());
                    Ok(v)
                }
                Err(e) if route_invalidating(&e) => {
                    // Worker gone (node alive) — drop the instance so a retry re-resolves
                    // (remove_instance_if bumps the node's generation), then re-sync the
                    // fleet view through the SAME per-node debounce as every other chat
                    // outcome (T14 r5): N concurrent stale-route failures must coalesce
                    // into one pull, not fan out one direct RPC each. The epoch bump
                    // already invalidates any pull that started before it.
                    fleet.remove_instance_if(&node, worker, &model).await;
                    fleet.schedule_chat_refresh(node.clone());
                    Err(e)
                }
                // Transport-level / other failure surfacing mid-stream: drop the dead
                // transport (Arc-identity guarded) and remap to HG027. The instance is kept.
                Err(e) => {
                    // A FAILED chat also ended node-side activity (the lease
                    // dropped, resetting the idle clock) — refresh here too
                    // (T14 r4), or the card shows a pre-chat "last active"
                    // indefinitely. ACCEPTED RESIDUAL (r5): a HUB-side timeout
                    // lands here while the node may legitimately still be
                    // generating — the pulled snapshot truthfully shows the
                    // chat in flight, and when it really ends no hub refresh
                    // fires (the node cannot push events yet; the fleet-event
                    // SSE work owns that). The UI renders such rows with their
                    // sync provenance ("N in flight · synced X ago"), so the
                    // aged claim is self-describing, never fresh-looking.
                    fleet.schedule_chat_refresh(node.clone());
                    Err(fleet.handle_op_error(&node, &used, e).await)
                }
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
        .ok_or_else(|| HiggsError::ProtocolViolation {
            peer_role: "node".into(),
            detail: "M_NODE_LOAD reply missing worker_id".into(),
        })?;
    u32::try_from(raw).map_err(|_| HiggsError::ProtocolViolation {
        peer_role: "node".into(),
        detail: format!("M_NODE_LOAD reply worker_id {raw} out of u32 range"),
    })
}

/// The served id to display for a resident worker in the fleet view: the route's served id,
/// but ONLY when the route's model still matches the worker's currently-reported model.
///
/// A node-process restart can leave a STALE route whose reused worker id now serves a DIFFERENT
/// model (the documented HG018 case, healed on the next hub-driven op). Tagging by `(node,
/// worker)` alone would then mislabel that worker — claiming a served id clients can't actually
/// reach on it. So when the route's model and the worker's model disagree (or there is no route),
/// the worker has no current served id and this returns `""`. `served_rev` is the inverse of
/// [`FleetActor::served_ids`] (`(node, worker) → served`), so its keys match `routes`.
fn served_id_for_worker(
    key: &(NodeKey, WorkerId),
    worker_model: &str,
    routes: &HashMap<(NodeKey, WorkerId), String>,
    served_rev: &HashMap<(NodeKey, WorkerId), String>,
) -> String {
    routes
        .get(key)
        .filter(|route_model| route_model.as_str() == worker_model)
        .and_then(|_| served_rev.get(key))
        .cloned()
        .unwrap_or_default()
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
#[path = "fleet_tests.rs"]
mod tests;
