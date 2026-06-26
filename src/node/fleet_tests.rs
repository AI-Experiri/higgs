
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
        .add_node(node_key.clone(), Arc::new(NodeTransport::new(conn)), None)
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
async fn disconnect_all_closes_transports_but_keeps_routes() {
    // The hub kill switch: disconnect_all severs every transport (nodes go offline) but the
    // route table SURVIVES, so re-enabling the hub is a pure reconnect, not a re-pair.
    let (fleet, node_key, model_id, _root) = fleet_with_one_node().await;
    fleet.load(&node_key, &model_id).await.unwrap();
    assert!(fleet.is_remote(&model_id).await, "route present after load");
    assert!(
        fleet.node_ids().await.contains(&node_key),
        "node connected before disconnect_all"
    );

    fleet.disconnect_all().await;

    assert!(
        !fleet.node_ids().await.contains(&node_key),
        "no connected nodes after disconnect_all"
    );
    // Route kept (durable across reconnect) and the node still appears — as DISCONNECTED,
    // not retired (retire removes the node-id slot; disconnect keeps it).
    assert!(
        fleet.is_remote(&model_id).await,
        "route survives disconnect_all"
    );
    let view = fleet.nodes_view().await;
    let n = view
        .iter()
        .find(|n| n.endpoint_id == node_key)
        .expect("node still listed after disconnect_all");
    assert!(!n.connected, "node shown disconnected, not removed");
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
async fn nodes_view_fills_served_id_per_worker() {
    // Two instances of the same model on one node → the fleet view must tag each
    // resident worker with its hub-assigned served id (`model`, `model-1`), NOT the
    // empty string the node reports.
    let (fleet, node_key, model_id, _root) = fleet_with_one_node().await;
    fleet.load(&node_key, &model_id).await.unwrap();
    fleet.load(&node_key, &model_id).await.unwrap();
    let view = fleet.nodes_view().await;
    let node = view
        .iter()
        .find(|n| n.endpoint_id == node_key)
        .expect("node present in fleet view");
    let inv = node.inventory.as_ref().expect("node inventory present");
    assert_eq!(
        inv.workers.len(),
        2,
        "two resident workers: {:?}",
        inv.workers
    );
    let mut served: Vec<String> = inv.workers.iter().map(|w| w.served_id.clone()).collect();
    served.sort();
    assert_eq!(
        served,
        vec![model_id.clone(), format!("{model_id}-1")],
        "each worker carries its hub-assigned served id: {:?}",
        inv.workers
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
        admit_gen: 0,
    };
    let served = actor.served_ids();
    // Three instances → three distinct served ids, each mapping to a distinct worker.
    assert_eq!(served.len(), 3, "every instance is reachable: {served:?}");
    let workers: std::collections::HashSet<_> = served.values().map(|(_, w)| *w).collect();
    assert_eq!(workers.len(), 3, "no two served ids share a worker");
    // Deterministic: re-deriving yields the same mapping.
    assert_eq!(actor.served_ids(), served);
}

#[test]
fn served_id_for_worker_guards_stale_route_model_mismatch() {
    let key = |w| ("nodeA".to_string(), WorkerId(w));
    let mut routes = HashMap::new();
    routes.insert(key(1), "org/a".to_string());
    routes.insert(key(2), "org/b".to_string());
    // The reverse map as nodes_view builds it from served_ids() (same keys as routes).
    let mut rev = HashMap::new();
    rev.insert(key(1), "org/a".to_string());
    rev.insert(key(2), "org/b".to_string());

    // Matching model → the worker is tagged with its served id.
    assert_eq!(
        served_id_for_worker(&key(1), "org/a", &routes, &rev),
        "org/a"
    );
    // Stale route: a restart reused worker 2 for a DIFFERENT model → no current served id.
    assert_eq!(served_id_for_worker(&key(2), "org/c", &routes, &rev), "");
    // No route at all for this worker → empty.
    assert_eq!(served_id_for_worker(&key(9), "org/x", &routes, &rev), "");
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
