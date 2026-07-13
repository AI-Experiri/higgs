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
    FleetEventKind, InventoryWorker, NodeFleetEvent, NodeInventory, M_NODE_INVENTORY, M_NODE_KILL,
    M_NODE_LOAD, M_NODE_SCAN, M_NODE_UNLOAD, N_FLEET_EVENT, N_LOG_LINE,
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
    /// taken, computed hub-side from a DUAL wall+monotonic stamp taken at the
    /// pull's start (see `PulledAt` — max of two lower bounds; errs stale-ward
    /// only). The hub re-pulls inventory on connect and after lifecycle ops,
    /// and a `fleet_events` node (T10) re-stamps it on every pushed state
    /// change — so its rows read fresh in event time. Only an event-LESS
    /// legacy node still relies on the debounced chat-end pull, and only there
    /// does node-local activity age invisibly until the next pull. Absent for
    /// the LIVE local card (its stats are read per request) and when there is
    /// no cached inventory.
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

higgs_ts! {
/// One live fleet event (T10), broadcast by the hub for real-time UIs: WHICH node
/// changed and WHAT kind of change. The event is a cache-invalidation signal, not a
/// data carrier — the hub has already folded the node's pushed snapshot into its
/// inventory cache when this fires, so a subscriber re-reads `nodes_view` (or its
/// serving layer's equivalent) for the actual state.
#[derive(Debug, Clone, serde::Serialize)]
pub struct FleetEvent {
    /// The node's canonical endpoint id (the [`NodeKey`], as `NodeView.endpoint_id`).
    pub endpoint_id: String,
    pub kind: FleetEventKind,
}
}

/// Capacity of the hub's [`FleetEvent`] broadcast. Events are tiny invalidation
/// signals; a subscriber that lags past 256 of them just refreshes on the next one.
const FLEET_EVENT_CAP: usize = 256;

/// Outcome of a `CommitWorkers` push merge. Only `Applied` (and a `NeedsFull`
/// whose fallback pull succeeds) re-broadcasts the public [`FleetEvent`] — a
/// `Stale` push changed nothing, so announcing its kind would hand subscribers
/// misleading (possibly reversed) signals about state that did not move (T10 r2 #4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PushOutcome {
    /// Merged into the cached inventory.
    Applied,
    /// Dropped: stale connection, or not newer than the cached snapshot.
    Stale,
    /// No cached inventory to merge into — the caller runs the (single,
    /// coalesced) full-inventory fallback pull, identified by this owner
    /// generation (r6 #2). The push itself was RETAINED for replay on the
    /// next committed pull (T10 r5 #1).
    NeedsFull(u64),
    /// Retained like `NeedsFull`, but a fallback pull is already in flight —
    /// the caller spawns nothing (T10 r5 #2).
    Deferred,
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
        /// Whether the node's HELLO advertised the `fleet_events` capability (T10):
        /// it pushes `N_FLEET_EVENT` state changes, so the hub's chat-end debounced
        /// re-pull is demoted to a legacy fallback for it. Same lifecycle as the
        /// version facts: refreshed per (re)admission, cleared at retire.
        fleet_events: bool,
        #[allow(clippy::type_complexity)]
        reply: oneshot::Sender<Option<(NodeId, Option<Arc<NodeTransport>>)>>,
    },
    /// Remove a node's transport only if it's still the current one (Arc identity); closes it.
    DropTransportIf {
        node: NodeKey,
        transport: Arc<NodeTransport>,
        /// `true` = the transport was current and actually removed (the node went
        /// disconnected now) — `false` for a stale watcher's no-op.
        reply: oneshot::Sender<bool>,
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
        /// `true` = the snapshot actually replaced the cache (epoch current AND
        /// newer by seq/stamp) — what `refresh_inventory` keys its
        /// `InventorySynced` announcement on (T10 r3 #2).
        reply: oneshot::Sender<bool>,
    },
    BumpEpoch {
        node: NodeKey,
        reply: oneshot::Sender<()>,
    },
    /// Merge a node-PUSHED worker snapshot (`N_FLEET_EVENT`, T10) into the cached
    /// inventory: replace `workers` + `snapshot_seq` + the freshness stamp, KEEP the
    /// pulled host/hardware/runtime fields. Guards: the transport-identity check below
    /// plus the same seq ordering as `CommitInventory` (pushes and pulls share the node
    /// actor's one seq counter, so they order against each other exactly). There is
    /// deliberately NO epoch gate (T10 r2 #1): the epoch protects PULLS, whose data was
    /// captured before hub-side route changes could land — a push is the node's own
    /// truth, ordered by its stream and its seq, and pre-sampling an epoch outside this
    /// message only opened a race where a lifecycle op between sample and commit
    /// discarded a valid final event (e.g. the idle-reap `WorkerUnloaded`) that nothing
    /// would ever resend, pinning a dead worker in the cache of an event-pushing node.
    CommitWorkers {
        node: NodeKey,
        /// The connection the event arrived on. The commit applies ONLY while this is
        /// still the node's CURRENT transport (Arc identity, T10 r1): without this
        /// check a push buffered from a replaced connection could land after the
        /// re-admission stripped the cached seq and install the OLD process's high seq,
        /// freezing out the new process until its counter catches up. (Retire empties
        /// the transport slot, so post-retire pushes are dropped here too.)
        transport: Arc<NodeTransport>,
        /// The pushed event's kind, re-broadcast (atomically with the merge, T10
        /// r4 #1) as the public [`FleetEvent`] when — and only when — the merge
        /// APPLIES (r2 #4).
        kind: FleetEventKind,
        workers: Vec<InventoryWorker>,
        snapshot_seq: u64,
        /// Dual-clock stamp of the event's RECEIPT (see [`PulledAt`]).
        pulled_at: PulledAt,
        reply: oneshot::Sender<PushOutcome>,
    },
    /// A NeedsFull fallback pull finished (either way). Clears the node's
    /// coalescing flag; replies `true` when the caller should RETRY (the pull
    /// left no cached inventory, a retained push is still waiting, and the node
    /// is still connected) — without the retry a single failed fallback would
    /// strand the pending push until the next lifecycle op (T10 r5 #2).
    /// Pre-pull ownership check for the fallback loop (r8, mirroring
    /// `ChatRefreshOwner`): is `gen` still the slot's owner? A re-admission
    /// clears the slot while a detached owner may be pre-pull or mid-sleep —
    /// without this check that stale owner still fires one full inventory RPC
    /// (150 s bound, hardware probe) CONCURRENTLY with the successor's.
    /// Residual (same shape as the chat-refresh one): a re-admission landing
    /// DURING the pull still overlaps by at most that one bounded pull.
    FallbackOwner {
        node: NodeKey,
        gen: u64,
        reply: oneshot::Sender<bool>,
    },
    FallbackDone {
        node: NodeKey,
        /// The owner generation from `PushOutcome::NeedsFull` — a stale owner
        /// (slot cleared/re-claimed since) gets `false` and exits (r6 #2).
        gen: u64,
        reply: oneshot::Sender<bool>,
    },
    /// The fallback owner exhausted its retry budget (r9 #1): clear the slot
    /// AND the retained push, so a node persistently returning malformed
    /// inventory cannot pin an immortal 1 Hz probe loop. The NEXT event from
    /// the node re-seeds the pending push and claims a fresh owner — recovery
    /// stays event-paced instead of timer-paced.
    FallbackAbandon {
        node: NodeKey,
        gen: u64,
        reply: oneshot::Sender<()>,
    },
    /// Does this node push `N_FLEET_EVENT`s (its HELLO advertised `fleet_events`)?
    NodePushesEvents {
        node: NodeKey,
        reply: oneshot::Sender<bool>,
    },
    /// Chat-end refresh debounce, phase 1: may THIS completion own the (single)
    /// refresh for `node`? `Some(gen)` = caller owns the slot under that
    /// generation and schedules the pull; `None` = one is already
    /// scheduled/running — this completion was coalesced into its trailing
    /// re-run flag instead of spawning another pull.
    ChatRefreshBegin {
        node: NodeKey,
        reply: oneshot::Sender<Option<u64>>,
    },
    /// Chat-end refresh debounce, pre-pull check: is `gen` still the slot's
    /// owner? A retire removes the slot while its detached owner may be mid-
    /// settle; without this check that stale owner would still fire its pull
    /// CONCURRENTLY with a re-paired successor's. `false` = stand down.
    ChatRefreshOwner {
        node: NodeKey,
        gen: u64,
        reply: oneshot::Sender<bool>,
    },
    /// Chat-end refresh debounce, phase 2: the owned pull finished. `gen` must
    /// match the slot's owner generation — a STALE owner (its slot dropped by a
    /// retire, possibly re-created by a re-paired node's chat) gets `false` and
    /// exits without touching the successor. `true` = completions were
    /// coalesced meanwhile — run ONCE more (the flag resets); `false` = done,
    /// the slot is released (or was never this owner's).
    ChatRefreshEnd {
        node: NodeKey,
        gen: u64,
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
/// re-stamps and self-heals both. The stamp is taken BEFORE the RPC, so it
/// predates the node's actual capture by the request's transit time — the
/// reported age (and any idle+age sum a client renders) can only OVERSTATE
/// staleness by that transit, never claim freshness the data lacks (T14
/// r11): the design's one allowed error direction.
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
    /// (A suspend landing in the microseconds BETWEEN the two reads below can
    /// under-read by the sleep once; the next view recomputes correctly.)
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
    /// node, keyed by an OWNER GENERATION (T14 r12: a retire + rapid re-pair can
    /// leave a STALE owner running — its End must not mutate a successor's slot,
    /// or two owners pull concurrently); the bool = a LATER chat completed
    /// meanwhile, so run once more when the current pull finishes (trailing
    /// coalescing). Bounds the chat-driven pulls to at most ONE live owner per
    /// node however bursty the chats.
    chat_refreshes: HashMap<NodeKey, (u64, bool)>,
    /// Monotonic owner-generation source for `chat_refreshes`.
    chat_refresh_gen: u64,
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
    /// Nodes whose CURRENT admission advertised the `fleet_events` capability (T10).
    /// Membership decides whether the chat-end debounced re-pull is scheduled at all
    /// (event-pushing nodes keep the hub cache fresh themselves). Refreshed per
    /// (re)admission, cleared at retire — same lifecycle as the version facts.
    event_nodes: HashSet<NodeKey>,
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
    /// The hub-level [`FleetEvent`] fan-out, cloned from the wrapper (T10 r4 #1):
    /// EVERY event is emitted HERE, inside the actor handler that performs the
    /// state change it announces — so the event order equals the state order.
    /// Wrapper-side emits allowed interleavings like a retire's `NodeDropped`
    /// landing BEFORE a just-committed pull's `InventorySynced` (a false sync
    /// after a terminal drop).
    events_tx: tokio::sync::broadcast::Sender<FleetEvent>,
    /// A push that arrived BEFORE any cached inventory existed (T10 r5 #1): its
    /// worker snapshot + seq are RETAINED here (newest seq wins) instead of
    /// discarded, and replayed on top of the next committed pull if the push is
    /// newer — otherwise an older, delayed connect-time pull committing after a
    /// failed fallback would resurrect state the push had already superseded
    /// (e.g. an idle-reaped worker), with no watermark left to reject it.
    /// Cleared per (re)admission and at retire (a new process's data order).
    pending_pushes: HashMap<NodeKey, (FleetEventKind, Vec<InventoryWorker>, u64, PulledAt)>,
    /// Nodes with a NeedsFull fallback pull IN FLIGHT (T10 r5 #2), keyed by an
    /// OWNER GENERATION (r6 #2, same pattern as `chat_refreshes`): further
    /// cache-less pushes update `pending_pushes` but spawn NO additional pull —
    /// each pull runs a hardware probe on the node, so per-event spawning under
    /// an event burst would pile up control streams and probe processes. The
    /// generation stops a STALE owner (its slot cleared by a re-admission, and
    /// possibly re-claimed by the new connection's own fallback) from removing
    /// or retrying against the successor's slot.
    fallback_inflight: HashMap<NodeKey, u64>,
    /// Monotonic owner-generation source for `fallback_inflight`.
    fallback_gen: u64,
}

impl FleetActor {
    /// Best-effort fleet-event emit (no subscribers ⇒ dropped).
    fn emit(&self, node: &str, kind: FleetEventKind) {
        let _ = self.events_tx.send(FleetEvent {
            endpoint_id: node.to_string(),
            kind,
        });
    }

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
    fn drop_transport_if(&mut self, node: &str, transport: &Arc<NodeTransport>) -> bool {
        let removed = if self
            .nodes
            .get(node)
            .is_some_and(|cur| Arc::ptr_eq(cur, transport))
        {
            self.nodes.remove(node)
        } else {
            None
        };
        match removed {
            Some(t) => {
                tracing::warn!(
                    node,
                    "higgs: node connection dropped; transport removed (routes kept)"
                );
                // Close so a wedged-but-open connection's close-watcher wakes and releases its
                // Arc (otherwise it would wait on `closed()` forever).
                t.close();
                true
            }
            None => false,
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
        self.event_nodes.remove(node);
        self.pending_pushes.remove(node);
        self.fallback_inflight.remove(node);
        // Drop any chat-refresh debounce slot too: a retired node needs no
        // trailing re-run, and an owner task mid-loop then sees the Vacant arm
        // at its next ChatRefreshEnd and exits cleanly.
        self.chat_refreshes.remove(node);
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
                fleet_events,
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
                    // A (re)admission invalidates the PREVIOUS process's data
                    // order (T14 r19): snapshot_seq is per-node-process, so a
                    // restarted node counts from 1 again — comparing its fresh
                    // pulls against a retained seq-100 snapshot would reject
                    // every refresh and freeze the old inventory forever. The
                    // retained snapshot stays displayable (continuity) but its
                    // seq is stripped, so the first new pull commits via the
                    // stamp arm and re-establishes seq ordering.
                    if let Some((inv, _)) = self.inventories.get_mut(&node) {
                        inv.snapshot_seq = None;
                    }
                    // Same rule for the semver display: never inherit a prior
                    // connection's value. An EMPTY string (a HELLO version that
                    // sanitized away entirely) is stored as ABSENT — never a
                    // present-but-blank display value (T14 r22).
                    match software_version.filter(|v| !v.is_empty()) {
                        Some(v) => {
                            self.software_versions.insert(node.clone(), v);
                        }
                        None => {
                            self.software_versions.remove(&node);
                        }
                    }
                    // A retained pre-cache push and its fallback slot belong to the
                    // PREVIOUS connection's data order — never replay them into the
                    // new process's cache (T10 r5 #1).
                    self.pending_pushes.remove(&node);
                    self.fallback_inflight.remove(&node);
                    // Same per-admission rule for the event-push capability (T10): a
                    // node DOWNGRADED to an event-less build must fall back to the
                    // debounced re-pull, so never inherit a prior connection's flag.
                    if fleet_events {
                        self.event_nodes.insert(node.clone());
                    } else {
                        self.event_nodes.remove(&node);
                    }
                    let replaced = self.nodes.insert(node.clone(), transport);
                    // Announced HERE, atomic with the insert (T10 r4 #1) — a
                    // wrapper-side emit could interleave with a racing retire's
                    // NodeDropped in the wrong order.
                    self.emit(&node, FleetEventKind::NodeConnected);
                    // The cached INVENTORY is deliberately KEPT across (re)admission
                    // (same last-known-state continuity as across a disconnect), so
                    // for the sub-second window until the post-connect refresh
                    // commits — or indefinitely if that refresh fails — the card can
                    // pair the NEW version facts with the previous process's workers/
                    // hardware (T14 r10, accepted residual). The T14 age stamp keeps
                    // counting from the OLD pull, so the staleness is self-describing;
                    // clearing here would flash the card empty on every reconnect and
                    // lose data a failed refresh can never restore.
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
                let removed = self.drop_transport_if(&node, &transport);
                if removed {
                    // A REAL disconnect (not a stale watcher's no-op) is an event.
                    self.emit(&node, FleetEventKind::NodeDropped);
                }
                let _ = reply.send(removed);
            }
            FleetMsg::Retire { node, reply } => {
                // Only a node the fleet actually knew is a drop event — a retire
                // of an unknown key must not fabricate one.
                let known = self.node_ids.get(&node).is_some();
                self.retire(&node);
                if known {
                    self.emit(&node, FleetEventKind::NodeDropped);
                }
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
                let drained: Vec<(NodeKey, Arc<NodeTransport>)> = self.nodes.drain().collect();
                for (node, t) in drained {
                    tracing::info!(
                        node,
                        "higgs hub: disabling — node transport closed (route kept)"
                    );
                    t.close();
                    // The kill switch's disconnects are fleet events too (T10 r2
                    // #5), emitted atomically with the drain (r4 #1).
                    self.emit(&node, FleetEventKind::NodeDropped);
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
                    // `nodes_view` reports the snapshot's age from it. SAME-epoch
                    // ordering guard (T14 r7): two pulls under one lifecycle epoch
                    // (a stalled connect-time pull vs a later chat-end pull) must
                    // not commit out of order — an older-started pull never
                    // overwrites a newer-started one (monotonic stamps, same
                    // process, directly comparable).
                    let newer = self.inventories.get(&node).is_none_or(|(cur_inv, cur)| {
                        // Prefer the NODE's snapshot sequence when both sides
                        // carry one (T14 r17): it is true data order — hub-side
                        // stamps are not under QUIC stream reordering, where an
                        // earlier-stamped pull can carry NEWER data. Fall back
                        // to the mono stamp against pre-r17 nodes.
                        match (inventory.snapshot_seq, cur_inv.snapshot_seq) {
                            (Some(new_seq), Some(old_seq)) => new_seq > old_seq,
                            // Fallback ONLY against pre-seq (legacy) nodes/pairs:
                            // the stamp is the hub's best available order there,
                            // and QUIC reordering can still invert it — an
                            // inherent residual with no hub-side fix (the node
                            // reports no data order), self-healing on the next
                            // refresh. Current nodes always take the seq arm.
                            _ => pulled_at.mono > cur.mono,
                        }
                    });
                    if newer {
                        self.inventories
                            .insert(node.clone(), (*inventory, pulled_at));
                        // A push retained from before any cache existed (r5 #1) is
                        // replayed ON TOP of this commit when it is the newer data
                        // (its kind announces it, below, like any applied push).
                        let replayed = match self.pending_pushes.remove(&node) {
                            Some((kind, workers, seq, at)) => {
                                let (inv, cur_at) =
                                    self.inventories.get_mut(&node).expect("just inserted");
                                if inv.snapshot_seq.is_none_or(|pull_seq| seq > pull_seq) {
                                    inv.workers = workers;
                                    inv.snapshot_seq = Some(seq);
                                    *cur_at = at;
                                    Some(kind)
                                } else {
                                    None
                                }
                            }
                            None => None,
                        };
                        // Announced atomically with the commit (T10 r3 #1 / r4 #1):
                        // the pull is where hub-initiated lifecycle changes land,
                        // and the node's own (seq-stale) push for the change is
                        // rightly silenced — this event is the invalidation.
                        self.emit(&node, FleetEventKind::InventorySynced);
                        if let Some(kind) = replayed {
                            self.emit(&node, kind);
                        }
                        let _ = reply.send(true);
                    } else {
                        let _ = reply.send(false);
                    }
                } else {
                    let _ = reply.send(false);
                }
            }
            FleetMsg::BumpEpoch { node, reply } => {
                self.bump_epoch(&node);
                let _ = reply.send(());
            }
            FleetMsg::CommitWorkers {
                node,
                transport,
                kind,
                workers,
                snapshot_seq,
                pulled_at,
                reply,
            } => {
                // Stale-connection guard first (see the variant doc): only the node's
                // CURRENT transport may push. A replaced/disconnected connection's
                // buffered event is dropped outright — no needs_full either (its data
                // is the OLD process's; the new admission runs its own connect pull).
                // The admission must also have DECLARED `fleet_events` (T10 r5): the
                // capability is the contract that pushes participate in this node's
                // cache ordering — an undeclared admission stays a pure pull-model
                // node (its debounced re-pull remains active), so accepting its
                // pushes would mix both freshness models on one cache.
                let current = self.event_nodes.contains(&node)
                    && self
                        .nodes
                        .get(&node)
                        .is_some_and(|cur| Arc::ptr_eq(cur, &transport));
                let outcome = if !current {
                    PushOutcome::Stale
                } else {
                    match self.inventories.get_mut(&node) {
                        Some((cur_inv, cur_at)) => {
                            // Same ordering rule as `CommitInventory`: prefer the node's
                            // seq (pushes always carry one); the stamp arm covers only a
                            // cached snapshot WITHOUT one (a legacy pull, or the seq
                            // stripped at re-admission — where accepting the fresh push
                            // re-establishes seq ordering, like the first new pull does).
                            let newer = match cur_inv.snapshot_seq {
                                Some(old_seq) => snapshot_seq > old_seq,
                                None => pulled_at.mono > cur_at.mono,
                            };
                            if newer {
                                cur_inv.workers = workers;
                                cur_inv.snapshot_seq = Some(snapshot_seq);
                                // The single freshness stamp now dates the WORKER
                                // snapshot — the thing pushes update and the thing
                                // the UI's per-row chips describe. ACCEPTED RESIDUAL
                                // (T10 r5 #3): the retained host/hardware readings
                                // (cpu%, ram) can be older than this stamp suggests
                                // on a chatty node that is never re-pulled; every
                                // lifecycle op and (re)connect still runs a full
                                // pull, which bounds it in practice. A split
                                // hardware stamp / periodic hardware re-pull is
                                // future work, not warranted for display-only data.
                                *cur_at = pulled_at;
                                self.emit(&node, kind);
                                PushOutcome::Applied
                            } else {
                                PushOutcome::Stale
                            }
                        }
                        // No snapshot to merge into (the event outran the connect-time
                        // pull): don't fabricate hostname/hardware — RETAIN the push
                        // for replay on the next committed pull (r5 #1) and have the
                        // caller run ONE coalesced full pull (r5 #2).
                        None => {
                            let newer = self
                                .pending_pushes
                                .get(&node)
                                .is_none_or(|(_, _, seq, _)| snapshot_seq > *seq);
                            if newer {
                                self.pending_pushes
                                    .insert(node.clone(), (kind, workers, snapshot_seq, pulled_at));
                            }
                            match self.fallback_inflight.entry(node.clone()) {
                                std::collections::hash_map::Entry::Vacant(e) => {
                                    self.fallback_gen += 1;
                                    e.insert(self.fallback_gen);
                                    PushOutcome::NeedsFull(self.fallback_gen)
                                }
                                std::collections::hash_map::Entry::Occupied(_) => {
                                    PushOutcome::Deferred
                                }
                            }
                        }
                    }
                };
                let _ = reply.send(outcome);
            }
            FleetMsg::FallbackOwner { node, gen, reply } => {
                let owner = self.fallback_inflight.get(&node) == Some(&gen);
                let _ = reply.send(owner);
            }
            FleetMsg::FallbackAbandon { node, gen, reply } => {
                if self.fallback_inflight.get(&node) == Some(&gen) {
                    self.fallback_inflight.remove(&node);
                    self.pending_pushes.remove(&node);
                }
                let _ = reply.send(());
            }
            FleetMsg::FallbackDone { node, gen, reply } => {
                if self.fallback_inflight.get(&node) != Some(&gen) {
                    // Stale owner: the slot was cleared by a re-admission/retire
                    // (and possibly re-claimed by a successor) — exit without
                    // touching it (r6 #2).
                    let _ = reply.send(false);
                    return;
                }
                self.fallback_inflight.remove(&node);
                let retry = !self.inventories.contains_key(&node)
                    && self.pending_pushes.contains_key(&node)
                    && self.nodes.contains_key(&node);
                if retry {
                    // Re-claim the slot (same gen — this owner continues) so
                    // racing pushes keep deferring to this caller's retry loop.
                    self.fallback_inflight.insert(node, gen);
                }
                let _ = reply.send(retry);
            }
            FleetMsg::NodePushesEvents { node, reply } => {
                let _ = reply.send(self.event_nodes.contains(&node));
            }
            FleetMsg::ChatRefreshBegin { node, reply } => {
                let owned = match self.chat_refreshes.entry(node) {
                    std::collections::hash_map::Entry::Vacant(e) => {
                        self.chat_refresh_gen += 1;
                        e.insert((self.chat_refresh_gen, false));
                        Some(self.chat_refresh_gen)
                    }
                    std::collections::hash_map::Entry::Occupied(mut e) => {
                        // Coalesce: the running pull will go once more.
                        e.get_mut().1 = true;
                        None
                    }
                };
                let _ = reply.send(owned);
            }
            FleetMsg::ChatRefreshOwner { node, gen, reply } => {
                let owner = self
                    .chat_refreshes
                    .get(&node)
                    .is_some_and(|(g, _)| *g == gen);
                let _ = reply.send(owner);
            }
            FleetMsg::ChatRefreshEnd { node, gen, reply } => {
                let again = match self.chat_refreshes.entry(node) {
                    std::collections::hash_map::Entry::Occupied(mut e) if e.get().0 == gen => {
                        if e.get().1 {
                            e.get_mut().1 = false;
                            true
                        } else {
                            e.remove();
                            false
                        }
                    }
                    // A different owner generation holds the slot (retire dropped
                    // ours; a re-paired node's chat re-created it) — a STALE owner
                    // must not mutate the successor's state. Exit without a re-run.
                    std::collections::hash_map::Entry::Occupied(_) => false,
                    // Slot vanished (a retire cleared state) — nothing owed.
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
    /// Live [`FleetEvent`] fan-out (T10): node-pushed state changes (already folded into
    /// the inventory cache when emitted) plus hub-local connect/drop/retire markers.
    /// No subscribers ⇒ sends are dropped no-ops.
    events_tx: tokio::sync::broadcast::Sender<FleetEvent>,
}

impl HubFleet {
    /// Build a fleet that files relayed remote logs into `bus` (the hub's own `LogBus`, the
    /// one its serve layer reads — so remote worker output appears in the same console).
    /// Spawns the actor task — must be called from within a Tokio runtime.
    pub fn new(bus: Arc<LogBus>) -> Self {
        let bus_for_actor = bus.clone();
        let (events_tx, _) = tokio::sync::broadcast::channel(FLEET_EVENT_CAP);
        let events_for_actor = events_tx.clone();
        let handle = spawn_actor(FleetActor {
            nodes: HashMap::new(),
            routes: HashMap::new(),
            node_ids: NodeIdAllocator::new(),
            inventories: HashMap::new(),
            chat_refreshes: HashMap::new(),
            chat_refresh_gen: 0,
            software_versions: HashMap::new(),
            event_nodes: HashSet::new(),
            versions: HashMap::new(),
            epochs: HashMap::new(),
            bus: bus_for_actor,
            admit_gen: 0,
            events_tx: events_for_actor,
            pending_pushes: HashMap::new(),
            fallback_inflight: HashMap::new(),
            fallback_gen: 0,
        });
        Self {
            handle,
            bus,
            events_tx,
        }
    }

    /// Subscribe to live [`FleetEvent`]s (T10). Fired AFTER the corresponding cache
    /// mutation, so a subscriber that re-reads the fleet view on receipt observes the
    /// change the event announced.
    pub fn subscribe_fleet_events(&self) -> tokio::sync::broadcast::Receiver<FleetEvent> {
        self.events_tx.subscribe()
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
        fleet_events: bool,
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
                fleet_events,
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
        // Read the node's uni-stream notifications for THIS connection — relayed worker
        // stderr into the hub bus, and fleet events (T10) into the inventory cache —
        // until the connection closes (accept_uni errors). Weak: the reader must not
        // keep a retired fleet alive.
        tokio::spawn(read_node_notifications(
            node_id,
            node.clone(),
            self.bus.clone(),
            Arc::downgrade(self),
            transport.clone(),
        ));

        // On (re)connect: refresh the inventory for the fleet view. Best-effort, off the hot
        // path. (Loads are additive — no displaced workers are ever owed to an offline node —
        // so there are no pending unloads to reconcile.) When it COMMITS, the refresh itself
        // announces `InventorySynced` (T10 r1 #3 / r3 #2): `NodeConnected` above fires before
        // this pull even starts, so an event-driven UI refreshing on it reads the pre-connect
        // cache — the commit-keyed event is what tells it the real snapshot landed.
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

    /// Schedule the DEBOUNCED chat-end inventory refresh for `node` (T14 r2) —
    /// since T10 a LEGACY FALLBACK: it stands down entirely for a node whose
    /// admission advertised `fleet_events` (its own pushes keep the cache fresh):
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
            // LEGACY FALLBACK ONLY (T10): a node that pushes `N_FLEET_EVENT`s keeps the
            // cache fresh itself (its ChatEnd event lands without any settle delay), so
            // the debounced re-pull would only burn an inventory RPC to learn what the
            // push already committed. Checked here (not at the Drop guard, which must
            // stay sync) — per chat end, so a mid-chat re-admission that changes the
            // capability is honored by the very next completion.
            if fleet.node_pushes_events(&node).await {
                return;
            }
            let Some(gen) = fleet
                .ask(|reply| FleetMsg::ChatRefreshBegin {
                    node: node.clone(),
                    reply,
                })
                .await
                .flatten()
            else {
                return; // coalesced into the owner's trailing re-run
            };
            loop {
                tokio::time::sleep(SETTLE).await;
                // Pre-pull ownership check (T14 r13): a retire during the settle
                // removed our slot (and a re-paired successor may already own a
                // new one) — stand down WITHOUT pulling, or two owners fire
                // concurrent inventory RPCs. Residual: a retire landing DURING
                // the pull below can still overlap the successor's first pull
                // by at most one iteration — the commit-side epoch + mono-stamp
                // guards keep the DATA correct either way; the cost is one
                // duplicate probe, accepted (a full cancellation plumb isn't
                // warranted for that window).
                let still_owner = fleet
                    .ask(|reply| FleetMsg::ChatRefreshOwner {
                        node: node.clone(),
                        gen,
                        reply,
                    })
                    .await
                    .unwrap_or(false);
                if !still_owner {
                    return;
                }
                // Hard bound (T14 r11): the owner MUST reach ChatRefreshEnd or
                // the node's debounce slot strands forever (later completions
                // only set the trailing flag). Chosen ABOVE the control-plane
                // request timeout so it fires only when the transport is stuck
                // in a state that timeout never covers (e.g. open_bi starved of
                // stream credit on a connection that never closes).
                let _ = tokio::time::timeout(
                    std::time::Duration::from_secs(150),
                    fleet.refresh_inventory(&node),
                )
                .await;
                let again = fleet
                    .ask(|reply| FleetMsg::ChatRefreshEnd {
                        node: node.clone(),
                        gen,
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
        // A COMMITTED pull announces InventorySynced from INSIDE the commit handler
        // (T10 r3 #1/#2, r4 #1): the pull is where hub-initiated lifecycle changes
        // land (the node's own seq-stale push is rightly silenced), and emitting
        // atomically with the commit keeps a guard-rejected pull silent AND keeps
        // event order equal to state order (no InventorySynced after a racing
        // retire's NodeDropped).
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

    /// Fold one node-pushed `N_FLEET_EVENT` into the fleet state (T10): merge its worker
    /// snapshot into the cached inventory under the same epoch + seq guards as a pull
    /// (`FleetMsg::CommitWorkers`), then re-broadcast the kind as a hub [`FleetEvent`].
    /// If the event outran the connect-time pull (no cached inventory yet), fall back to
    /// one full `refresh_inventory` — the event's data still lands, via the pull path.
    async fn apply_node_event(
        self: &Arc<Self>,
        node: &NodeKey,
        ev: NodeFleetEvent,
        transport: &Arc<NodeTransport>,
    ) {
        let outcome = self
            .ask(|reply| FleetMsg::CommitWorkers {
                node: node.clone(),
                transport: transport.clone(),
                kind: ev.kind,
                workers: ev.workers,
                snapshot_seq: ev.snapshot_seq,
                pulled_at: PulledAt::now(),
                reply,
            })
            .await
            .unwrap_or(PushOutcome::Stale);
        // The actor announced an APPLIED merge atomically (r4 #1); a Stale drop is
        // silent (r2 #4 — and if the newer state arrived via a pull instead, THAT
        // commit already announced InventorySynced, r3 #1). A cache-less push was
        // RETAINED for replay (r5 #1); NeedsFull means this caller owns the ONE
        // coalesced fallback pull (r5 #2 — Deferred callers spawn nothing). It is
        // spawned DETACHED (r4 #2): this fn runs on the event stream's reader
        // task, and awaiting the pull inline would head-of-line block every later
        // push behind a possibly-stuck RPC. Ordering is safe — the pull's commit
        // is seq-guarded, and the retained push replays on top when newer. On
        // failure the actor grants ONE paced retry at a time while the node stays
        // connected and the push is still waiting.
        if let PushOutcome::NeedsFull(gen) = outcome {
            // Weak (r9 #1): a detached retry loop must not keep a dropped fleet —
            // and with it the actor and the transport — alive.
            let fleet = Arc::downgrade(self);
            let node = node.clone();
            tokio::spawn(async move {
                // Bounded retries (r9 #1): a connected node that persistently
                // returns malformed inventory would otherwise drive this loop —
                // and its hardware probes — forever at 1 Hz.
                let mut attempts = 0u32;
                loop {
                    let Some(fleet) = fleet.upgrade() else {
                        return;
                    };
                    // Pre-pull ownership check (r8): a re-admission may have
                    // cleared (and a successor re-claimed) this owner's slot
                    // while it was being spawned or sleeping between retries —
                    // stand down WITHOUT pulling, or two owners fire concurrent
                    // inventory RPCs against the new connection.
                    let still_owner = fleet
                        .ask(|reply| FleetMsg::FallbackOwner {
                            node: node.clone(),
                            gen,
                            reply,
                        })
                        .await
                        .unwrap_or(false);
                    if !still_owner {
                        return;
                    }
                    // Hard bound (r6 #1, same 150 s rationale as the chat-refresh
                    // owner's): the control-plane request timeout starts only
                    // AFTER open_bi succeeds, so a node out of bidi credit (with
                    // its uni event stream healthy) would otherwise pin this
                    // owner — and with it the coalescing slot — forever.
                    let _ = tokio::time::timeout(
                        std::time::Duration::from_secs(150),
                        fleet.refresh_inventory(&node),
                    )
                    .await;
                    let retry = fleet
                        .ask(|reply| FleetMsg::FallbackDone {
                            node: node.clone(),
                            gen,
                            reply,
                        })
                        .await
                        .unwrap_or(false);
                    if !retry {
                        return;
                    }
                    attempts += 1;
                    if attempts >= 5 {
                        fleet
                            .ask(|reply| FleetMsg::FallbackAbandon {
                                node: node.clone(),
                                gen,
                                reply,
                            })
                            .await;
                        return;
                    }
                    drop(fleet); // don't hold the strong ref across the sleep
                    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                }
            });
        }
    }

    /// Does this node's CURRENT admission push `N_FLEET_EVENT`s? (T10 — decides whether
    /// the chat-end debounced re-pull is needed at all.)
    async fn node_pushes_events(&self, node: &str) -> bool {
        self.ask(|reply| FleetMsg::NodePushesEvents {
            node: node.to_string(),
            reply,
        })
        .await
        .unwrap_or(false)
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
        // The refresh must fire on EVERY exit of the wrapped future — success,
        // failure, AND hub-side abort (the WS bridge drops in-flight ops on
        // socket close; an outcome-arm-only schedule would skip the abort path
        // and leave the cache reporting pre-chat activity indefinitely, T14 r8).
        // A Drop guard owned by the future covers all three. It is DEBOUNCED
        // per node (T14 r2: one pull in flight, later completions coalesce into
        // one trailing re-run; 250 ms settle lets the node's ChatEnd land),
        // epoch-gated, and since T10 a LEGACY FALLBACK: a node whose admission
        // advertised `fleet_events` pushes its own ChatStart/ChatEnd snapshots
        // (`N_FLEET_EVENT`) — covering node-local chats, hub-side aborts (incl.
        // abort-before-start), and late timed-out generations the pull model
        // could never see — so `schedule_chat_refresh` stands down for it. The
        // old pull-model residuals (T14 r5/r20) now apply ONLY to event-less
        // legacy nodes, where the UI's sync provenance keeps the aged claim
        // honest.
        struct RefreshOnDrop {
            fleet: Arc<HubFleet>,
            node: NodeKey,
        }
        impl Drop for RefreshOnDrop {
            fn drop(&mut self) {
                // A future can be dropped outside a runtime (process teardown) —
                // scheduling spawns, so skip silently there; there is no view
                // left to refresh anyway.
                if tokio::runtime::Handle::try_current().is_ok() {
                    self.fleet.schedule_chat_refresh(self.node.clone());
                }
            }
        }
        let refresh_guard = RefreshOnDrop {
            fleet: fleet.clone(),
            node: node.clone(),
        };
        let wrapped = async move {
            let _refresh_guard = refresh_guard; // fires on every exit, incl. abort
            match fut.await {
                Ok(v) => Ok(v),
                Err(e) if route_invalidating(&e) => {
                    // Worker gone (node alive) — drop the instance so a retry
                    // re-resolves (remove_instance_if bumps the node's generation);
                    // the re-sync rides the guard's debounced refresh like every
                    // other outcome (T14 r5/r8) — N concurrent stale-route failures
                    // coalesce into one pull, and the epoch bump invalidates any
                    // pull that started before it.
                    fleet.remove_instance_if(&node, worker, &model).await;
                    Err(e)
                }
                // Transport-level / other failure surfacing mid-stream: drop the dead
                // transport (Arc-identity guarded) and remap to HG027. The instance is kept.
                // A FAILED chat also ended node-side activity (the guard's
                // refresh covers it, T14 r4/r8). ACCEPTED RESIDUAL (r5): a
                // HUB-side timeout lands here while the node may legitimately
                // still be generating — the pulled snapshot truthfully shows
                // the chat in flight, and when it really ends no hub refresh
                // fires (the node cannot push events yet; the fleet-event SSE
                // work owns that). The UI renders such rows with their sync
                // provenance ("N in flight · synced X ago").
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

/// Accept the node's uni stream(s) of notifications and dispatch by method: `N_LOG_LINE`
/// worker stderr into the hub bus under `LogSource::RemoteWorker { node, worker }`, and
/// `N_FLEET_EVENT` worker-state pushes (T10) into the inventory cache + the hub's
/// [`FleetEvent`] broadcast. Returns when the connection closes. Best-effort: a malformed
/// or unknown-method frame is skipped, not fatal (that skip is ALSO the forward/backward
/// compatibility rule — an older peer's reader ignores methods it doesn't know). `fleet`
/// is weak so a live connection's reader never keeps a dropped fleet alive.
async fn read_node_notifications(
    node: NodeId,
    node_key: NodeKey,
    bus: Arc<LogBus>,
    fleet: std::sync::Weak<HubFleet>,
    transport: Arc<NodeTransport>,
) {
    let conn = transport.connection();
    while let Ok(recv) = conn.accept_uni().await {
        let bus = bus.clone();
        let node_key = node_key.clone();
        let fleet = fleet.clone();
        let transport = transport.clone();
        tokio::spawn(async move {
            let mut lines = BufReader::new(recv).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                let Ok(RpcFrame::Notification(n)) = rpc::decode(&line) else {
                    continue;
                };
                if n.method == N_FLEET_EVENT {
                    // The node's push: fold it into the cache (epoch+seq guarded) and
                    // re-broadcast. Awaited IN-ORDER on this stream task — one uni
                    // stream carries all of a connection's events, so a burst can't
                    // commit out of order hub-side.
                    let Ok(ev) = serde_json::from_value::<NodeFleetEvent>(n.params) else {
                        continue;
                    };
                    // Node-origin kinds only (T10 r5 #4): the hub-local markers
                    // (NodeConnected/NodeDropped/InventorySynced) are the HUB's
                    // statements about connectivity and its own cache — a peer
                    // must not be able to broadcast a phantom disconnect (or a
                    // fake sync) by naming one in a push.
                    if !matches!(
                        ev.kind,
                        FleetEventKind::ChatStart
                            | FleetEventKind::ChatEnd
                            | FleetEventKind::WorkerLoaded
                            | FleetEventKind::WorkerUnloaded
                    ) {
                        continue;
                    }
                    if let Some(fleet) = fleet.upgrade() {
                        fleet.apply_node_event(&node_key, ev, &transport).await;
                    }
                    continue;
                }
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
