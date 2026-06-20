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

use parking_lot::Mutex;
use serde_json::json;
use tokio::sync::mpsc;

use crate::diagnostic::HiggsError;
use crate::node::transport::NodeTransport;
use crate::node::worker_id::WorkerId;
use crate::remote::{M_NODE_KILL, M_NODE_LOAD, M_NODE_UNLOAD};

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

/// The hub's view of its remote fleet.
#[derive(Default)]
pub struct HubFleet {
    /// Currently-connected nodes → their live transport (absent while disconnected).
    nodes: Mutex<HashMap<NodeKey, Arc<NodeTransport>>>,
    /// Durable routes: `model → (node, worker)`, survive reconnect.
    routes: Mutex<HashMap<String, (NodeKey, WorkerId)>>,
}

impl HubFleet {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register/replace a paired node's transport (after the hub admits its HELLO). Routes
    /// are KEPT across reconnect (the node's workers persist). Closes any prior transport
    /// and spawns a watcher that drops the transport (only) when its connection closes.
    pub fn add_node(self: &Arc<Self>, node: NodeKey, transport: Arc<NodeTransport>) {
        let replaced = self.nodes.lock().insert(node.clone(), transport.clone());
        if let Some(old) = replaced {
            old.close(); // free the old connection + wake its close-watcher
        }
        let weak = Arc::downgrade(self);
        let watched = transport;
        tokio::spawn(async move {
            watched.closed().await;
            if let Some(fleet) = weak.upgrade() {
                fleet.drop_transport_if(&node, &watched);
            }
        });
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
        self.routes.lock().retain(|_, (n, _)| n != node);
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

    /// Remove a route only if it STILL equals `expected` — so a route from a concurrent op
    /// (which awaits a remote RPC) is never clobbered.
    fn remove_route_if(&self, model: &str, expected: &(NodeKey, WorkerId)) {
        let mut routes = self.routes.lock();
        if routes.get(model) == Some(expected) {
            routes.remove(model);
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
        let worker_u64 = result
            .get("worker_id")
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| HiggsError::WorkerDead {
                context: "node load reply missing worker_id".into(),
            })?;
        let worker_id = u32::try_from(worker_u64).map_err(|_| HiggsError::WorkerDead {
            context: format!("node load reply worker_id {worker_u64} out of u32 range"),
        })?;
        let new = (node.to_string(), WorkerId(worker_id));
        let displaced = self.routes.lock().insert(model.to_string(), new.clone());
        if let Some(old) = displaced {
            if old != new {
                // Unload the worker this load displaced. Best-effort: if the OLD node is
                // currently disconnected (a cross-node reload during a blip) the unload is
                // skipped and that worker is reconciled when the node reconnects and reports
                // its resident workers via M_INVENTORY (P4) — the hub then reaps workers it
                // no longer has a route for. Not handled with an ad-hoc pending queue here.
                self.best_effort_unload(&old).await;
            }
        }
        Ok(new.1)
    }

    /// Best-effort unload of a displaced worker via its node's CURRENT transport.
    async fn best_effort_unload(&self, route: &(NodeKey, WorkerId)) {
        if let Ok(t) = self.transport(&route.0) {
            let _ = t.request(M_NODE_UNLOAD, json!({ "worker_id": route.1 .0 })).await;
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
        let (node, worker) = self.resolve(model).ok_or_else(|| HiggsError::ModelNotFound {
            id: model.to_string(),
        })?;
        let transport = self.transport(&node)?;
        let res = transport.request(method, json!({ "worker_id": worker.0 })).await;
        if res.is_ok() || res.as_ref().err().is_some_and(route_invalidating) {
            self.remove_route_if(model, &(node.clone(), worker));
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
        let (node, worker) = self.resolve(model).ok_or_else(|| HiggsError::ModelNotFound {
            id: model.to_string(),
        })?;
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
                    // Worker gone (node alive) — drop the route so a retry re-resolves.
                    fleet.remove_route_if(&model, &(node.clone(), worker));
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

        let fleet = Arc::new(HubFleet::new());
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
    async fn ops_on_unknown_node_are_unreachable() {
        let fleet = Arc::new(HubFleet::new());
        let err = fleet.load("ghost", "m").await.unwrap_err();
        assert!(err.to_string().starts_with("[HG027]"), "got {err}");
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
