//! Hub fleet: the set of paired/connected nodes and the `model → (node, worker)` routing
//! the hub uses to send `/v1` chat to a remote-resident worker (DESIGN-remote.md §4.2/§4.3,
//! P3 hub seam).
//!
//! Routing is built from the hub's OWN remote loads. Two correlation domains stay separate:
//! this fleet's `NodeTransport`s never touch the hub's local `Supervisor`.
//!
//! Durable routes, transient transports: the `--node` daemon reuses ONE `NodeRuntime`
//! across reconnects, so its workers (and ids) persist through a dropped connection. Routes
//! are therefore keyed by `(node, worker)` and SURVIVE reconnect; only the per-connection
//! transport comes and goes. A genuine node-process restart leaves stale routes that
//! self-heal on the first worker-gone error (the node replies HG007 → route dropped). The
//! transport handle is compared by `Arc` identity so a stale failure can't drop a freshly
//! reconnected transport.

use std::collections::HashMap;
use std::sync::Arc;

use iroh::endpoint::Connection;
use parking_lot::Mutex;
use serde_json::json;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::sync::mpsc;

use crate::diagnostic::HiggsError;
use crate::log_bus::{LogBus, LogSource};
use crate::node::node_id::{NodeId, NodeIdAllocator};
use crate::node::transport::NodeTransport;
use crate::node::worker_id::WorkerId;
use crate::remote::{
    NodeInventory, M_NODE_INVENTORY, M_NODE_KILL, M_NODE_LOAD, M_NODE_UNLOAD, N_LOG_LINE,
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

/// The hub's UI/API view of one paired node: its stable id, endpoint, whether it's currently
/// connected, and its last-fetched inventory (host + resident workers + hardware/runtime).
#[derive(Debug, Clone, serde::Serialize)]
pub struct NodeView {
    pub node_id: u32,
    pub endpoint_id: String,
    pub connected: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inventory: Option<NodeInventory>,
}

/// The hub's view of its remote fleet.
pub struct HubFleet {
    /// Currently-connected nodes → their live transport (absent while disconnected).
    nodes: Mutex<HashMap<NodeKey, Arc<NodeTransport>>>,
    /// Durable routes: `model → (node, worker)`, survive reconnect.
    routes: Mutex<HashMap<String, (NodeKey, WorkerId)>>,
    /// Stable hub-local [`NodeId`] per `EndpointId`, for `LogSource::RemoteWorker` tagging.
    node_ids: Mutex<NodeIdAllocator>,
    /// Last-fetched inventory per node (host + workers + hw/rt), refreshed on connect and
    /// after every hub-driven lifecycle change. The node's reply is authoritative.
    inventories: Mutex<HashMap<NodeKey, NodeInventory>>,
    /// Per-node lifecycle generation, bumped on every load/unload/kill/route-drop. A
    /// `refresh_inventory` only commits its (possibly in-flight, now stale) result if this is
    /// unchanged since it started — so a slow connect-time fetch can't clobber a newer state.
    epochs: Mutex<HashMap<NodeKey, u64>>,
    /// The hub's Developer-Log bus: relayed remote worker stderr lands here under
    /// `LogSource::RemoteWorker { node, worker }` so it shares the operator's log console.
    bus: Arc<LogBus>,
    /// Unloads the hub OWES a node but couldn't deliver because the node was disconnected
    /// (e.g. a cross-node reload displaced a worker on a node that was offline). Drained on
    /// the node's next reconnect (`reconcile_pending_unloads`) so the displaced worker is
    /// reaped instead of leaking RAM/VRAM forever. Distinct from the hub-restart case (no
    /// pending entries → node-reported workers are preserved, not killed).
    pending_unloads: Mutex<std::collections::HashSet<(NodeKey, WorkerId)>>,
}

impl HubFleet {
    /// Build a fleet that files relayed remote logs into `bus` (the hub's own `LogBus`, the
    /// one its serve layer reads — so remote worker output appears in the same console).
    pub fn new(bus: Arc<LogBus>) -> Self {
        Self {
            nodes: Mutex::new(HashMap::new()),
            routes: Mutex::new(HashMap::new()),
            node_ids: Mutex::new(NodeIdAllocator::new()),
            inventories: Mutex::new(HashMap::new()),
            epochs: Mutex::new(HashMap::new()),
            bus,
            pending_unloads: Mutex::new(std::collections::HashSet::new()),
        }
    }

    /// The stable hub-local [`NodeId`] for a node key, if it has ever been admitted.
    pub fn node_id(&self, node: &str) -> Option<NodeId> {
        self.node_ids.lock().get(node)
    }

    /// Pre-register a known (e.g. persisted-allowlisted) node so it appears in the fleet view
    /// as DISCONNECTED before it reconnects — assigns its stable `NodeId` without a transport.
    pub fn seed_node(&self, node: &str) {
        self.node_ids.lock().assign(node);
    }

    /// The hub Developer-Log bus this fleet relays remote worker stderr into.
    pub fn bus(&self) -> &Arc<LogBus> {
        &self.bus
    }

    /// Register/replace a paired node's transport (after the hub admits its HELLO). Routes
    /// are KEPT across reconnect (the node's workers persist). Closes any prior transport
    /// and spawns a watcher that drops the transport (only) when its connection closes.
    pub fn add_node(self: &Arc<Self>, node: NodeKey, transport: Arc<NodeTransport>) {
        // Mint (or reuse) this node's stable NodeId so relayed logs tag consistently.
        let node_id = self.node_ids.lock().assign(&node);
        // Read the node's relayed worker stderr (its uni stream) into the hub bus for THIS
        // connection; ends when the connection closes (accept_uni errors).
        tokio::spawn(read_remote_logs(transport.connection(), node_id, self.bus.clone()));

        let replaced = self.nodes.lock().insert(node.clone(), transport.clone());
        if let Some(old) = replaced {
            old.close(); // free the old connection + wake its close-watcher
        }
        // Bump the generation on (re)admission so any inventory fetch from a PRIOR connection
        // still in flight can't commit its now-stale result over this fresh one.
        self.bump_epoch(&node);
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
                fleet.drop_transport_if(&node, &watched);
            }
        });
    }

    /// Fetch `node`'s inventory over its live transport and cache it for the fleet view. The
    /// node's `M_NODE_INVENTORY` reply is AUTHORITATIVE (its real resident workers) and stored
    /// verbatim — except a result is dropped if a lifecycle op changed this node's generation
    /// while the request was in flight (a newer op's own refresh wins), so a slow connect-time
    /// fetch can never resurrect a stale worker list.
    pub async fn refresh_inventory(&self, node: &str) -> Result<NodeInventory, HiggsError> {
        let epoch_before = self.epoch(node);
        let transport = self.transport(node)?;
        let value = transport
            .request(M_NODE_INVENTORY, json!({}))
            .await
            .map_err(|e| self.handle_op_error(node, &transport, e))?;
        let inventory: NodeInventory = serde_json::from_value(value).map_err(|e| {
            HiggsError::WorkerDead { context: format!("node inventory decode failed: {e}") }
        })?;
        // Commit only if no lifecycle op superseded us (epochs locked across the check+store).
        let mut epochs = self.epochs.lock();
        if *epochs.entry(node.to_string()).or_insert(0) == epoch_before {
            self.inventories.lock().insert(node.to_string(), inventory.clone());
        }
        Ok(inventory)
    }

    /// This node's current lifecycle generation (0 if never touched).
    fn epoch(&self, node: &str) -> u64 {
        self.epochs.lock().get(node).copied().unwrap_or(0)
    }

    /// Bump a node's lifecycle generation — call AFTER mutating its routes so any in-flight
    /// `refresh_inventory` for it is invalidated and won't commit a pre-change snapshot.
    fn bump_epoch(&self, node: &str) {
        *self.epochs.lock().entry(node.to_string()).or_insert(0) += 1;
    }

    /// The fleet view: one [`NodeView`] per node the hub has ever admitted, with live
    /// connected state + last-fetched inventory. Sorted by `NodeId` (stable order).
    pub fn nodes_view(&self) -> Vec<NodeView> {
        let assigned = self.node_ids.lock().all();
        let nodes = self.nodes.lock();
        let inventories = self.inventories.lock();
        assigned
            .into_iter()
            .map(|(endpoint_id, node_id)| NodeView {
                node_id: node_id.0,
                connected: nodes.contains_key(&endpoint_id),
                inventory: inventories.get(&endpoint_id).cloned(),
                endpoint_id,
            })
            .collect()
    }

    /// On connection close, remove ONLY the transport — and only if it's still the current
    /// one (Arc identity), so a reconnect's fresh transport isn't dropped by a stale
    /// watcher. Routes are kept (durable across reconnect); ops return HG027 until the node
    /// reconnects.
    fn drop_transport_if(&self, node: &str, transport: &Arc<NodeTransport>) {
        let removed = {
            let mut nodes = self.nodes.lock();
            if nodes.get(node).is_some_and(|cur| Arc::ptr_eq(cur, transport)) {
                nodes.remove(node)
            } else {
                None
            }
        };
        if let Some(t) = removed {
            tracing::warn!(node, "higgs: node connection dropped; transport removed (routes kept)");
            // Close so a wedged-but-open connection's close-watcher wakes and releases its
            // Arc (otherwise it would wait on `closed()` forever).
            t.close();
        }
    }

    /// Explicitly retire a node: drop its transport AND its routes (operator action / the
    /// node is truly gone). Closes the transport.
    pub fn retire(&self, node: &str) {
        if let Some(t) = self.nodes.lock().remove(node) {
            t.close();
        }
        // Reclaim ALL of this node's relayed-log rings — including any worker displaced by a
        // reload that's no longer on a current route (a per-route walk would miss those).
        if let Some(nid) = self.node_id(node) {
            self.bus.evict_node(nid);
        }
        self.routes.lock().retain(|_, (n, _)| n != node);
        // Bump the generation so a refresh already in flight can't reinsert stale inventory
        // after this removal.
        self.bump_epoch(node);
        self.inventories.lock().remove(node);
        // The node is gone for good — drop any unloads we were owing it.
        self.pending_unloads.lock().retain(|(n, _)| n != node);
    }

    pub fn node_ids(&self) -> Vec<NodeKey> {
        let mut v: Vec<_> = self.nodes.lock().keys().cloned().collect();
        v.sort();
        v
    }

    /// The live transport for a node, or HG027 if it isn't currently connected.
    fn transport(&self, node: &str) -> Result<Arc<NodeTransport>, HiggsError> {
        self.nodes
            .lock()
            .get(node)
            .cloned()
            .ok_or_else(|| HiggsError::NodeUnreachable {
                endpoint_id: node.to_string(),
                detail: "node not connected".into(),
            })
    }

    /// Reclaim a remote worker's relayed-log ring, if the node has a known [`NodeId`].
    /// Shared by every route-drop / unload path so the hub bus never keeps a gone worker's
    /// lines.
    fn evict_remote_logs(&self, node: &str, worker: WorkerId) {
        if let Some(nid) = self.node_id(node) {
            self.bus.evict_remote(nid, worker);
        }
    }

    /// Resolve `model` to its `(node, worker)` route or fail with HG002 — the precondition
    /// shared by chat/unload/kill.
    fn require_route(&self, model: &str) -> Result<(NodeKey, WorkerId), HiggsError> {
        self.resolve(model).ok_or_else(|| HiggsError::ModelNotFound { id: model.to_string() })
    }

    /// Remove a route only if it STILL equals `expected` — so a route from a concurrent op
    /// (which awaits a remote RPC) is never clobbered. When the route is actually removed,
    /// also reclaim the worker's relayed-log ring and drop it from the cached fleet view, so
    /// EVERY route-drop path (explicit unload/kill AND a chat-time worker-gone) keeps the
    /// node's logs + `/api/higgs/nodes` consistent.
    fn remove_route_if(&self, model: &str, expected: &(NodeKey, WorkerId)) {
        let removed = {
            let mut routes = self.routes.lock();
            if routes.get(model) == Some(expected) {
                routes.remove(model).is_some()
            } else {
                false
            }
        };
        if removed {
            let (node, worker) = expected;
            self.evict_remote_logs(node, *worker);
            // Invalidate any in-flight inventory fetch; the caller refreshes after.
            self.bump_epoch(node);
        }
    }

    /// On a transport-level failure (`WorkerDead` = the connection is gone), drop the dead
    /// transport — but only if `used` is still the current one (Arc identity), so a stale
    /// failure can't drop a freshly reconnected transport — and remap to HG027. Routes are
    /// kept (the node may reconnect with its workers intact). Other errors pass through.
    fn handle_op_error(&self, node: &str, used: &Arc<NodeTransport>, e: HiggsError) -> HiggsError {
        if matches!(e, HiggsError::WorkerDead { .. }) {
            self.drop_transport_if(node, used);
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
        let transport = self.transport(node)?;
        let result = transport
            .request(M_NODE_LOAD, json!({ "id": model }))
            .await
            .map_err(|e| self.handle_op_error(node, &transport, e))?;
        let new = (node.to_string(), WorkerId(parse_worker_id(&result)?));
        let displaced = self.routes.lock().insert(model.to_string(), new.clone());
        if let Some(old) = displaced {
            if old != new {
                // Unload the worker this load displaced. If the OLD node is currently
                // disconnected (a cross-node reload during a blip) the unload is recorded as
                // a pending unload and delivered when that node reconnects
                // (`reconcile_pending_unloads`), so the displaced worker is never leaked.
                self.best_effort_unload(&old).await;
            }
        }
        // Refresh the fleet view from the node's authoritative state after the load lands.
        self.bump_epoch(node);
        let _ = self.refresh_inventory(node).await;
        Ok(new.1)
    }

    /// Best-effort unload of a displaced worker via its node's CURRENT transport, also
    /// reclaiming its relayed-log ring so a reload doesn't leak the old worker's lines. If the
    /// node is DISCONNECTED the unload can't be delivered, so it's recorded as pending and
    /// reaped on the node's next reconnect — never silently leaked.
    async fn best_effort_unload(&self, route: &(NodeKey, WorkerId)) {
        self.evict_remote_logs(&route.0, route.1);
        match self.transport(&route.0) {
            Ok(t) => {
                let res = t.request(M_NODE_UNLOAD, json!({ "worker_id": route.1 .0 })).await;
                // Clear the obligation ONLY if the unload landed (Ok) or the node reports the
                // worker already gone (HG006/HG007). A real failure (transport hiccup mid-
                // request) means the worker may still be running, so KEEP it pending so the
                // next reconnect retries — otherwise the displaced worker leaks. Mirrors
                // `reconcile_pending_unloads`.
                if res.is_ok() || res.as_ref().err().is_some_and(route_invalidating) {
                    self.pending_unloads.lock().remove(route);
                } else {
                    self.pending_unloads.lock().insert(route.clone());
                }
            }
            // Node offline — owe it the unload until it reconnects.
            Err(_) => {
                self.pending_unloads.lock().insert(route.clone());
            }
        }
        // Re-sync the DISPLACED node's fleet view (it may differ from the load's target node
        // on a cross-node reload, where the caller only refreshes the new node).
        self.bump_epoch(&route.0);
        let _ = self.refresh_inventory(&route.0).await;
    }

    /// Deliver any unloads owed to `node` (recorded while it was offline), now that it's
    /// reconnected — so a worker displaced during a disconnect is reaped rather than leaked.
    /// Only targets workers the hub explicitly tried to unload (never the node's legitimate
    /// resident workers a fresh hub simply hasn't routed yet).
    async fn reconcile_pending_unloads(&self, node: &str) {
        let owed: Vec<WorkerId> = self
            .pending_unloads
            .lock()
            .iter()
            .filter(|(n, _)| n == node)
            .map(|(_, w)| *w)
            .collect();
        if owed.is_empty() {
            return;
        }
        let Ok(t) = self.transport(node) else { return };
        for w in owed {
            let res = t.request(M_NODE_UNLOAD, json!({ "worker_id": w.0 })).await;
            // Clear on success OR when the node reports the worker already gone
            // (HG006/HG007) — the owed unload is satisfied either way. Leaving a worker-gone
            // entry pending would leak forever AND, after a node restart reuses the id for a
            // DIFFERENT worker, a later reconcile could unload that legitimate worker. (Mirrors
            // `unload_or_kill`'s clear-on-worker-gone.) A transport error keeps it for the
            // next reconnect.
            if res.is_ok() || res.as_ref().err().is_some_and(route_invalidating) {
                self.pending_unloads.lock().remove(&(node.to_string(), w));
                self.evict_remote_logs(node, w);
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
        let (node, worker) = self.require_route(model)?;
        let transport = self.transport(&node)?;
        let res = transport.request(method, json!({ "worker_id": worker.0 })).await;
        if res.is_ok() || res.as_ref().err().is_some_and(route_invalidating) {
            // remove_route_if reclaims the log ring + bumps the node's generation.
            self.remove_route_if(model, &(node.clone(), worker));
            // Re-sync the fleet view from the node's authoritative state.
            let _ = self.refresh_inventory(&node).await;
        }
        res.map(|_| ()).map_err(|e| self.handle_op_error(&node, &transport, e))
    }

    /// Resolve a model to its `(node, worker)`, if routed.
    pub fn resolve(&self, model: &str) -> Option<(NodeKey, WorkerId)> {
        self.routes.lock().get(model).cloned()
    }

    /// Is this model resident on some remote node?
    pub fn is_remote(&self, model: &str) -> bool {
        self.routes.lock().contains_key(model)
    }

    /// Remote-resident model ids whose node is CURRENTLY connected (for `/v1/models`
    /// discovery = servable now). A route whose node is disconnected is hidden — chat to it
    /// would fail with HG027 until the node reconnects.
    pub fn routed_models(&self) -> Vec<String> {
        let nodes = self.nodes.lock();
        let mut v: Vec<_> = self
            .routes
            .lock()
            .iter()
            .filter(|(_, (node, _))| nodes.contains_key(node))
            .map(|(model, _)| model.clone())
            .collect();
        v.sort();
        v
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
        (mpsc::UnboundedReceiver<String>, impl std::future::Future<Output = Result<serde_json::Value, HiggsError>> + Send),
        HiggsError,
    > {
        let (node, worker) = self.require_route(model)?;
        let transport = self.transport(&node)?;
        let (rx, fut) = transport
            .chat(worker.0, model.to_string(), messages_json, max_tokens, temperature, tools_json)
            .await
            .map_err(|e| self.handle_op_error(&node, &transport, e))?;

        let fleet = self.clone();
        let model = model.to_string();
        let used = transport;
        let wrapped = async move {
            match fut.await {
                Ok(v) => Ok(v),
                Err(e) if route_invalidating(&e) => {
                    // Worker gone (node alive) — drop the route so a retry re-resolves, and
                    // re-sync the fleet view (remove_route_if bumped the node's generation).
                    fleet.remove_route_if(&model, &(node.clone(), worker));
                    let _ = fleet.refresh_inventory(&node).await;
                    Err(e)
                }
                // Transport-level / other failure surfacing mid-stream: drop the dead
                // transport (Arc-identity guarded) and remap to HG027, same as setup-time
                // failures. The durable route is kept (the node may reconnect).
                Err(e) => Err(fleet.handle_op_error(&node, &used, e)),
            }
        };
        Ok((rx, wrapped))
    }
}

/// Extract the `worker_id` from a node's `M_NODE_LOAD` reply, validating it is present and
/// fits a `u32` (the wire type) — a missing or out-of-range value is a protocol fault.
fn parse_worker_id(reply: &serde_json::Value) -> Result<u32, HiggsError> {
    let raw = reply.get("worker_id").and_then(serde_json::Value::as_u64).ok_or_else(|| {
        HiggsError::WorkerDead { context: "node load reply missing worker_id".into() }
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
                let Ok(RpcFrame::Notification(n)) = rpc::decode(&line) else { continue };
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
                        LogSource::RemoteWorker { node, worker: WorkerId(w) },
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
        fleet.add_node(node_key.clone(), Arc::new(NodeTransport::new(conn)));
        (fleet, node_key, model_id, root)
    }

    #[tokio::test]
    async fn load_routes_and_resolves() {
        let (fleet, node_key, model_id, _root) = fleet_with_one_node().await;
        assert!(!fleet.is_remote(&model_id));
        let worker = fleet.load(&node_key, &model_id).await.unwrap();
        assert!(fleet.is_remote(&model_id));
        assert_eq!(fleet.resolve(&model_id), Some((node_key, worker)));
    }

    #[tokio::test]
    async fn chat_relays_to_routed_worker() {
        let (fleet, node_key, model_id, _root) = fleet_with_one_node().await;
        fleet.load(&node_key, &model_id).await.unwrap();
        let (mut rx, fut) = fleet.chat(&model_id, "[]".into(), 8, 0.0, None).await.unwrap();
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
        assert_eq!(fleet.resolve(&model_id), Some((node_key, w2)));
    }

    #[tokio::test]
    async fn unload_then_kill_drop_routes() {
        let (fleet, node_key, model_id, _root) = fleet_with_one_node().await;
        fleet.load(&node_key, &model_id).await.unwrap();
        fleet.unload(&model_id).await.unwrap();
        assert!(!fleet.is_remote(&model_id), "unload drops the route");

        fleet.load(&node_key, &model_id).await.unwrap();
        fleet.kill(&model_id).await.unwrap();
        assert!(!fleet.is_remote(&model_id), "kill drops the route");
    }

    #[tokio::test]
    async fn unload_and_chat_unrouted_model_error() {
        let (fleet, _node_key, _model_id, _root) = fleet_with_one_node().await;
        assert!(fleet.unload("nope/none").await.is_err());
        assert!(fleet.kill("nope/none").await.is_err());
        assert!(fleet.resolve("nope/none").is_none());
        assert!(fleet.chat("nope/none", "[]".into(), 8, 0.0, None).await.is_err());
    }

    #[tokio::test]
    async fn routed_models_lists_connected_only() {
        let (fleet, node_key, model_id, _root) = fleet_with_one_node().await;
        assert!(fleet.routed_models().is_empty());
        fleet.load(&node_key, &model_id).await.unwrap();
        assert_eq!(fleet.routed_models(), vec![model_id.clone()]);
        // After retiring the node, the route is gone → not advertised.
        fleet.retire(&node_key);
        assert!(fleet.routed_models().is_empty());
    }

    #[tokio::test]
    async fn displaced_unload_to_offline_node_is_recorded_then_reconciled() {
        let (fleet, node_key, model_id, _root) = fleet_with_one_node().await;
        let w = fleet.load(&node_key, &model_id).await.unwrap();

        // An unload owed to a node that ISN'T connected is recorded as pending (not lost).
        fleet.best_effort_unload(&("offline-node".to_string(), w)).await;
        assert!(
            fleet.pending_unloads.lock().contains(&("offline-node".to_string(), w)),
            "unload to an offline node is recorded as pending"
        );

        // When a node with a pending unload reconnects, reconciliation delivers it (the
        // connected fake node answers M_NODE_UNLOAD), clearing the pending entry.
        fleet.pending_unloads.lock().insert((node_key.clone(), w));
        fleet.reconcile_pending_unloads(&node_key).await;
        assert!(
            !fleet.pending_unloads.lock().contains(&(node_key.clone(), w)),
            "reconnect drains the owed unload"
        );

        // A pending unload for a worker the node NO LONGER has (e.g. it restarted) must also
        // be cleared — the node replies worker-gone (HG007), which counts as reconciled.
        // Otherwise it would leak forever and could later kill a same-id worker after restart.
        fleet.pending_unloads.lock().insert((node_key.clone(), WorkerId(9999)));
        fleet.reconcile_pending_unloads(&node_key).await;
        assert!(
            !fleet.pending_unloads.lock().contains(&(node_key.clone(), WorkerId(9999))),
            "worker-gone reply clears the owed unload (not left pending forever)"
        );

        // Retiring a node clears anything still owed to it.
        fleet.best_effort_unload(&("gone".to_string(), w)).await;
        assert!(fleet.pending_unloads.lock().contains(&("gone".to_string(), w)));
        fleet.retire("gone");
        assert!(fleet.pending_unloads.lock().iter().all(|(n, _)| n != "gone"));
    }

    #[test]
    fn chat_timeout_does_not_invalidate_a_route() {
        // A remote chat that times out (HG016) is NOT a worker-gone / dead-transport signal:
        // the node may be healthy on a long generation. So it must neither drop the route nor
        // (via handle_op_error, which only acts on WorkerDead) tear down the transport.
        let timeout = HiggsError::ChatTimeout { elapsed: std::time::Duration::from_secs(600) };
        assert!(!route_invalidating(&timeout), "chat timeout keeps the route");
        assert!(
            !matches!(timeout, HiggsError::WorkerDead { .. }),
            "chat timeout is not WorkerDead, so handle_op_error passes it through unchanged"
        );
    }

    #[test]
    fn seed_node_lists_a_known_node_as_disconnected() {
        let fleet = Arc::new(HubFleet::new(Arc::new(crate::log_bus::LogBus::new())));
        fleet.seed_node("endpointA");
        let views = fleet.nodes_view();
        assert_eq!(views.len(), 1, "seeded node appears in the view");
        assert_eq!(views[0].endpoint_id, "endpointA");
        assert!(!views[0].connected, "seeded node is disconnected (no transport yet)");
        assert!(views[0].inventory.is_none(), "no inventory until it connects");
        assert!(fleet.node_id("endpointA").is_some(), "got a stable NodeId");
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
        assert!(inv.hardware.cpu_cores > 0, "inventory carries real hardware");

        // A hub-driven load refreshes the cached view from the node's authoritative state.
        let w = fleet.load(&node_key, &model_id).await.unwrap();
        let workers = fleet.nodes_view()[0].inventory.as_ref().unwrap().workers.clone();
        assert!(
            workers.iter().any(|x| x.worker_id == w.0 && x.model == model_id),
            "load refreshes the cached inventory: {workers:?}"
        );
        // Unload refreshes it back out.
        fleet.unload(&model_id).await.unwrap();
        let workers = fleet.nodes_view()[0].inventory.as_ref().unwrap().workers.clone();
        assert!(workers.iter().all(|x| x.worker_id != w.0), "unload removes it from the view");

        let views = fleet.nodes_view();
        assert_eq!(views.len(), 1, "one node in the fleet view");
        let v = &views[0];
        assert_eq!(v.endpoint_id, node_key);
        assert!(v.connected, "node is connected");
        assert!(v.inventory.is_some(), "view carries the fetched inventory");

        // After retire the node is still listed (id is durable) but disconnected + no inventory.
        fleet.retire(&node_key);
        let views = fleet.nodes_view();
        assert_eq!(views.len(), 1, "retired node keeps its durable id slot");
        assert!(!views[0].connected, "retired node is disconnected");
        assert!(views[0].inventory.is_none(), "retire clears the cached inventory");
    }

    #[tokio::test]
    async fn retire_drops_node_and_routes() {
        let (fleet, node_key, model_id, _root) = fleet_with_one_node().await;
        fleet.load(&node_key, &model_id).await.unwrap();
        fleet.retire(&node_key);
        assert!(fleet.node_ids().is_empty());
        assert!(!fleet.is_remote(&model_id));
        assert!(fleet.chat(&model_id, "[]".into(), 8, 0.0, None).await.is_err());
    }
}
