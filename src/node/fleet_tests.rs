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
        .add_node(
            node_key.clone(),
            Arc::new(NodeTransport::new(conn)),
            None,
            None,
            None,
        )
        .await;
    (fleet, node_key, model_id, root)
}

/// A params-load against a MAJOR-2 admission passes the gate and dispatches
/// (the version branch, complementing the floor-1 refusal above). What lands
/// ON the wire is pinned by the mock-transport payload test (cov_fleet2), real
/// application by the integration test, and the FACADE path (including the
/// id-force) by embed_tests::node_load_params_forces_the_wire_id_from_model.
#[tokio::test]
async fn params_load_reaches_the_node_at_protocol_two() {
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
    let conn = tokio::time::timeout(std::time::Duration::from_secs(30), hub.accept())
        .await
        .expect("accept within 30s (a swallowed node-task panic otherwise hangs here)")
        .expect("incoming")
        .await
        .expect("conn");
    std::mem::forget(hub);
    let fleet = Arc::new(HubFleet::new(Arc::new(crate::log_bus::LogBus::new())));
    fleet
        .add_node(
            node_key.clone(),
            Arc::new(NodeTransport::new(conn)),
            None,
            Some(2),
            Some("9.9.9-test".to_string()),
        )
        .await;
    assert_eq!(fleet.node_protocol(&node_key).await, Some(2));

    let params = crate::remote::NodeLoadParams {
        id: model_id.clone(),
        ctx_len: Some(2048),
        gpu_layers: None,
        threads: None,
        params: None,
    };
    let worker = fleet
        .load(&node_key, &model_id, Some(params))
        .await
        .expect("params-load against a major-2 node");
    assert!(worker.0 >= 1, "the gated load spawned a worker");

    // T14: the fleet view carries what admission stored — the negotiated major
    // and the node's self-reported build — before any retire.
    let view = fleet.nodes_view().await;
    let row = view
        .iter()
        .find(|n| n.endpoint_id == node_key)
        .expect("admitted node in the view");
    assert_eq!(row.protocol, Some(2), "view carries the negotiated major");
    assert_eq!(
        row.software_version.as_deref(),
        Some("9.9.9-test"),
        "view carries the HELLO semver"
    );

    // Retire clears the stored major (the TOCTOU residual note leans on this;
    // it had no direct pin).
    fleet.retire(&node_key).await;
    assert_eq!(
        fleet.node_protocol(&node_key).await,
        None,
        "retire clears the version slot"
    );
    // T14: retire drops the whole row (node_ids removed), so the software slot
    // cannot leak into a later view either — pin via a fresh view scan.
    assert!(
        fleet
            .nodes_view()
            .await
            .iter()
            .all(|n| n.endpoint_id != node_key),
        "retired node leaves the view entirely"
    );
    // NB the fake worker echoes only the id, and InventoryWorker carries no
    // ctx field (a T9 gap), so the PAYLOAD-content pin (ctx_len actually on
    // the wire) lives in the mock-transport test in tests/cov_fleet2.rs, and
    // the real-application pin (node's worker reports ctx 2048) in the
    // integration test over a held transport.
}

/// The T8 params gate: a params-load needs the node's HELLO-negotiated major
/// ≥ 2. A version-less admission (tests/direct callers, or a pre-plumbing
/// admission) reads as the conservative floor 1 → [HG078]; param-less loads
/// keep working against any node. (The ≥2 happy path's PAYLOAD is pinned by
/// cov_fleet2's mock cases; embed_tests pins the facade dispatch + id-force.)
#[tokio::test]
async fn params_load_refused_below_protocol_two() {
    let (fleet, node_key, model_id, _root) = fleet_with_one_node().await;
    // fleet_with_one_node admits with agreed_version = None → floor 1.
    assert_eq!(fleet.node_protocol(&node_key).await, None);

    let params = crate::remote::NodeLoadParams {
        id: model_id.clone(),
        ctx_len: Some(2048),
        gpu_layers: None,
        threads: None,
        params: None,
    };
    let err = fleet
        .load(&node_key, &model_id, Some(params))
        .await
        .unwrap_err();
    assert!(
        matches!(err, HiggsError::NodeTooOldForParams { agreed: 1, .. }),
        "params against a floor-1 node → HG078, got {err:?}"
    );

    // The classic bare load is untouched by the gate — and so is a Some(p)
    // whose fields are ALL None (byte-identical wire): never version-refused
    // for asking nothing. Fail-on-revert for the all-None-as-bare arm.
    fleet.load(&node_key, &model_id, None).await.unwrap();
    let empty = crate::remote::NodeLoadParams {
        id: String::new(),
        ctx_len: None,
        gpu_layers: None,
        threads: None,
        params: None,
    };
    fleet
        .load(&node_key, &model_id, Some(empty))
        .await
        .expect("all-None params load against a floor-1 node");

    // …and so is a Some whose rich object has no real overrides: it NORMALIZES
    // to nothing before the version gate, so a floor-1 node still loads. A
    // regression that moves normalization after the gate fails here with HG078.
    let empty_rich = crate::remote::NodeLoadParams {
        id: String::new(),
        ctx_len: Some(0),
        gpu_layers: None,
        threads: None,
        params: Some(Default::default()),
    };
    fleet
        .load(&node_key, &model_id, Some(empty_rich))
        .await
        .expect("normalized-empty params load against a floor-1 node");
}

/// Connectivity precedes the version gate: a SEEDED (paired-but-offline,
/// version-less) node gets HG027 from a params-load — never "[HG078] update
/// the node", which would be false advice for a current node that is merely
/// disconnected. Fail-on-revert: swap the gate back before `transport()` and
/// this reads HG078.
#[tokio::test]
async fn params_load_against_a_seeded_offline_node_is_unreachable_not_too_old() {
    let fleet = Arc::new(HubFleet::new(Arc::new(crate::log_bus::LogBus::new())));
    fleet.seed_node("paired-but-offline").await;

    let params = crate::remote::NodeLoadParams {
        id: "m".into(),
        ctx_len: Some(256),
        gpu_layers: None,
        threads: None,
        params: None,
    };
    let err = fleet
        .load("paired-but-offline", "m", Some(params))
        .await
        .unwrap_err();
    assert!(
        matches!(err, HiggsError::NodeUnreachable { .. }),
        "offline node → HG027, not a version refusal: {err:?}"
    );
}

/// A version-less re-admission CLEARS the stored major (the field doc's
/// promised floor) — it must not inherit the previous connection's version.
/// Fail-on-revert for the AdmitNode None → remove arm.
#[tokio::test]
async fn versionless_readmission_clears_the_stored_major() {
    let hub = local_endpoint().await;
    let node = local_endpoint().await;
    let hub_addr = hub.addr();
    let node_key = node.id().to_string();

    // Two sequential dials from the SAME node identity (each dial needs a
    // concurrent accept — join the two sides).
    let (dial1, conn1) = tokio::join!(node.connect(hub_addr.clone(), ALPN), async {
        hub.accept().await.expect("incoming").await.expect("conn 1")
    });
    let (dial2, conn2) = tokio::join!(node.connect(hub_addr.clone(), ALPN), async {
        hub.accept().await.expect("incoming").await.expect("conn 2")
    });
    let _keep = (dial1.expect("dial 1"), dial2.expect("dial 2"));

    let fleet = Arc::new(HubFleet::new(Arc::new(crate::log_bus::LogBus::new())));
    fleet
        .add_node(
            node_key.clone(),
            Arc::new(NodeTransport::new(conn1)),
            None,
            Some(2),
            Some("9.9.9-test".to_string()),
        )
        .await;
    assert_eq!(fleet.node_protocol(&node_key).await, Some(2));

    fleet
        .add_node(
            node_key.clone(),
            Arc::new(NodeTransport::new(conn2)),
            None,
            None,
            None,
        )
        .await;
    assert_eq!(
        fleet.node_protocol(&node_key).await,
        None,
        "a None re-admit clears to the floor, never inherits"
    );

    // T14 r19, through the REAL AdmitNode strip: seed a high-seq inventory
    // under the CURRENT epoch, then observe that the re-admission above has
    // stripped its seq — a restarted node's low-seq pull must commit.
    let epoch = fleet
        .ask(|reply| FleetMsg::Epoch {
            node: node_key.clone(),
            reply,
        })
        .await
        .unwrap_or(0);
    let mut high = crate::remote::NodeInventory {
        hostname: "pre-restart".into(),
        os: "macos".into(),
        workers: vec![],
        snapshot_seq: Some(1000),
        hardware: crate::system::HardwareInfo {
            cpu_name: String::new(),
            arch: String::new(),
            cpu_cores: 0,
            ram_total_bytes: 0,
            ram_used_bytes: 0,
            cpu_usage_percent: 0.0,
            gpus: vec![],
            vram_total_bytes: 0,
        },
        runtime: crate::system::RuntimeInfo {
            engine: String::new(),
            backend: String::new(),
            version: String::new(),
            binding: String::new(),
        },
    };
    fleet
        .ask(|reply| FleetMsg::CommitInventory {
            node: node_key.clone(),
            epoch_before: epoch,
            inventory: Box::new(high.clone()),
            pulled_at: PulledAt::now(),
            reply,
        })
        .await;
    // Third REAL admission (strips the stored seq)...
    let (dial3, conn3) = tokio::join!(node.connect(hub_addr, ALPN), async {
        hub.accept().await.expect("incoming").await.expect("conn 3")
    });
    let _keep3 = dial3.expect("dial 3");
    std::mem::forget(hub);
    fleet
        .add_node(
            node_key.clone(),
            Arc::new(NodeTransport::new(conn3)),
            None,
            None,
            None,
        )
        .await;
    // ...then the "restarted" node's first pull (seq 1, fresh epoch) commits.
    let epoch3 = fleet
        .ask(|reply| FleetMsg::Epoch {
            node: node_key.clone(),
            reply,
        })
        .await
        .unwrap_or(0);
    high.hostname = "post-restart".into();
    high.snapshot_seq = Some(1);
    fleet
        .ask(|reply| FleetMsg::CommitInventory {
            node: node_key.clone(),
            epoch_before: epoch3,
            inventory: Box::new(high),
            pulled_at: PulledAt::now(),
            reply,
        })
        .await;
    let row = fleet
        .nodes_view()
        .await
        .into_iter()
        .find(|n| n.endpoint_id == node_key)
        .expect("node in view");
    assert_eq!(
        row.inventory.as_ref().map(|i| i.hostname.as_str()),
        Some("post-restart"),
        "the REAL re-admission stripped the old seq, so the low-seq pull commits"
    );

    // T14: the view mirrors the clearing — neither the old major nor the old
    // semver survives a version-less re-admission.
    let row_after = fleet
        .nodes_view()
        .await
        .into_iter()
        .find(|n| n.endpoint_id == node_key)
        .expect("readmitted node in the view");
    assert_eq!(
        row_after.protocol, None,
        "view major cleared: {row_after:?}"
    );
    assert_eq!(
        row_after.software_version, None,
        "view semver cleared: {row_after:?}"
    );
}

#[tokio::test]
async fn load_routes_and_resolves() {
    let (fleet, node_key, model_id, _root) = fleet_with_one_node().await;
    assert!(!fleet.is_remote(&model_id).await);
    let worker = fleet.load(&node_key, &model_id, None).await.unwrap();
    assert!(fleet.is_remote(&model_id).await);
    assert_eq!(fleet.resolve(&model_id).await, Some((node_key, worker)));
}

#[tokio::test]
async fn disconnect_all_closes_transports_but_keeps_routes() {
    // The hub kill switch: disconnect_all severs every transport (nodes go offline) but the
    // route table SURVIVES, so re-enabling the hub is a pure reconnect, not a re-pair.
    let (fleet, node_key, model_id, _root) = fleet_with_one_node().await;
    fleet.load(&node_key, &model_id, None).await.unwrap();
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

/// T14 r8: the chat-end inventory refresh fires even when the HUB ABORTS the
/// chat (the WS bridge drops in-flight ops on socket close) — the Drop guard
/// on the wrapped future schedules it, not the outcome arms. Fail-on-revert:
/// move the scheduling back into the arms and the dropped future never
/// refreshes, so the age keeps growing past the pre-chat baseline.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn aborted_chat_still_schedules_the_inventory_refresh() {
    let (fleet, node_key, model_id, _root) = fleet_with_one_node().await;
    fleet.load(&node_key, &model_id, None).await.unwrap();
    // Age the load-time snapshot measurably.
    tokio::time::sleep(std::time::Duration::from_millis(700)).await;
    let age_before = fleet
        .nodes_view()
        .await
        .into_iter()
        .find(|n| n.endpoint_id == node_key)
        .and_then(|n| n.inventory_age_ms)
        .expect("inventory cached after load");
    assert!(age_before >= 600, "baseline aged: {age_before} ms");

    // Start a chat and ABORT it hub-side: drop the receiver and the future
    // without awaiting either (what the WS bridge does on socket close).
    let (rx, fut) = fleet
        .chat(&model_id, "[]".into(), 8, 0.0, None, None)
        .await
        .unwrap();
    drop(rx);
    drop(fut);

    // The guard's debounced refresh (250 ms settle) must land regardless.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    loop {
        let age_now = fleet
            .nodes_view()
            .await
            .into_iter()
            .find(|n| n.endpoint_id == node_key)
            .and_then(|n| n.inventory_age_ms);
        if matches!(age_now, Some(a) if a < age_before) {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "abort-path refresh never landed (age stayed >= {age_before} ms: {age_now:?})"
        );
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
}

#[tokio::test]
async fn chat_relays_to_routed_worker() {
    let (fleet, node_key, model_id, _root) = fleet_with_one_node().await;
    fleet.load(&node_key, &model_id, None).await.unwrap();
    let (mut rx, fut) = fleet
        .chat(&model_id, "[]".into(), 8, 0.0, None, None)
        .await
        .unwrap();
    let collector = tokio::spawn(async move {
        let mut got = Vec::new();
        while let Some(d) = rx.recv().await {
            got.push(d.text);
        }
        got
    });
    let final_res = fut.await.unwrap();
    // The delta queue merges same-kind runs the collector hasn't drained yet,
    // so the PARTITION is timing-dependent — the concatenation is the contract.
    assert_eq!(collector.await.unwrap().concat(), "hello");
    assert_eq!(final_res["content"], "hello");
}

#[tokio::test]
async fn loads_are_additive_two_instances_get_distinct_served_ids() {
    let (fleet, node_key, model_id, _root) = fleet_with_one_node().await;
    let w1 = fleet.load(&node_key, &model_id, None).await.unwrap();
    let w2 = fleet.load(&node_key, &model_id, None).await.unwrap();
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
    fleet.load(&node_key, &model_id, None).await.unwrap();
    fleet.load(&node_key, &model_id, None).await.unwrap();
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
    fleet.load(&node_key, &model_id, None).await.unwrap();
    fleet.unload(&model_id).await.unwrap();
    assert!(!fleet.is_remote(&model_id).await, "unload drops the route");

    fleet.load(&node_key, &model_id, None).await.unwrap();
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
        .chat("nope/none", "[]".into(), 8, 0.0, None, None)
        .await
        .is_err());
}

#[tokio::test]
async fn chat_pinned_refuses_when_the_id_resolves_elsewhere() {
    let (fleet, node_key, model_id, _root) = fleet_with_one_node().await;
    fleet.load(&node_key, &model_id, None).await.unwrap();

    // Pin to a node the id does NOT resolve to → refused at dispatch as the
    // transient concurrent-change class ([HG077]), no chat.
    match fleet
        .chat_pinned(
            &model_id,
            "some-other-node",
            "[]".into(),
            8,
            0.0,
            None,
            None,
        )
        .await
    {
        Err(HiggsError::ChatTestTargetMoved { .. }) => {}
        Err(other) => panic!("mismatched pin → HG077, got {other:?}"),
        Ok(_) => panic!("a mismatched pin must be refused, not dispatched"),
    }

    // Pin an id that does not resolve AT ALL → the same [HG077] class, not
    // HG002's "not found on disk" (no disk is consulted at dispatch).
    match fleet
        .chat_pinned(
            "gone/never-routed",
            &node_key,
            "[]".into(),
            8,
            0.0,
            None,
            None,
        )
        .await
    {
        Err(HiggsError::ChatTestTargetMoved { .. }) => {}
        Err(other) => panic!("unrouted pin → HG077, got {other:?}"),
        Ok(_) => panic!("an unrouted pin must be refused, not dispatched"),
    }

    // The UNPINNED path is unchanged: an unrouted id is still plain HG002.
    assert!(
        matches!(
            fleet
                .chat("gone/never-routed", "[]".into(), 8, 0.0, None, None)
                .await
                .map(|_| ()),
            Err(HiggsError::ModelNotFound { .. })
        ),
        "generic chat keeps its HG002 contract"
    );

    // Pin to the resolving node → dispatches like plain chat.
    let (rx, fut) = fleet
        .chat_pinned(&model_id, &node_key, "[]".into(), 8, 0.0, None, None)
        .await
        .expect("a matching pin dispatches");
    drop(rx);
    assert_eq!(fut.await.unwrap()["content"], "hello");
}

#[tokio::test]
async fn served_on_lists_one_nodes_instances_and_survives_disconnect() {
    let (fleet, node_key, model_id, _root) = fleet_with_one_node().await;
    assert!(fleet.served_on(&node_key).await.is_empty());
    fleet.load(&node_key, &model_id, None).await.unwrap();
    assert_eq!(fleet.served_on(&node_key).await, vec![model_id.clone()]);
    // Filtered BY node: another node key sees nothing.
    assert!(fleet.served_on("someone/else").await.is_empty());
    // Durable-route semantics (like `is_remote`, unlike `routed_models`): a
    // disconnect keeps the node's served list, so a chat-test target still
    // resolves and the chat itself reports the offline node (HG027).
    fleet.disconnect_all().await;
    assert_eq!(fleet.served_on(&node_key).await, vec![model_id]);
}

#[tokio::test]
async fn routed_models_lists_connected_only() {
    let (fleet, node_key, model_id, _root) = fleet_with_one_node().await;
    assert!(fleet.routed_models().await.is_empty());
    fleet.load(&node_key, &model_id, None).await.unwrap();
    assert_eq!(fleet.routed_models().await, vec![model_id.clone()]);
    // After retiring the node, the route is gone → not advertised.
    fleet.retire(&node_key).await;
    assert!(fleet.routed_models().await.is_empty());
}

/// T14 r1: the inventory age is WALL-CLOCK from the pull's START stamp — a
/// snapshot committed with an old stamp reads OLD at view time. This is the
/// direct pin for both r1 fixes: an `Instant`-typed stamp cannot represent
/// "two hours ago" across a sleep (macOS Instant freezes asleep), and a
/// commit-time `now()` stamp would read this seeded snapshot as fresh.
#[test]
fn inventory_age_is_the_max_of_both_clocks_from_the_pull_start_stamp() {
    let mut node_ids = NodeIdAllocator::new();
    node_ids.assign("nodeA");
    let mut inventories = HashMap::new();
    fn blank_inv() -> crate::remote::NodeInventory {
        crate::remote::NodeInventory {
            hostname: "h".into(),
            os: "macos".into(),
            workers: vec![],
            snapshot_seq: None,
            hardware: crate::system::HardwareInfo {
                cpu_name: String::new(),
                arch: String::new(),
                cpu_cores: 0,
                ram_total_bytes: 0,
                ram_used_bytes: 0,
                cpu_usage_percent: 0.0,
                gpus: vec![],
                vram_total_bytes: 0,
            },
            runtime: crate::system::RuntimeInfo {
                engine: String::new(),
                backend: String::new(),
                version: String::new(),
                binding: String::new(),
            },
        }
    }
    let inv = blank_inv();
    // Dual stamp, WALL side old: the monotonic side is fresh (as after an
    // 8 h SLEEP, where Instant froze) — the wall clock must still surface the
    // true age (max of the two lower bounds).
    let two_hours_ago = PulledAt {
        wall: std::time::SystemTime::now() - std::time::Duration::from_secs(7200),
        mono: std::time::Instant::now(),
    };
    inventories.insert("nodeA".to_string(), (inv, two_hours_ago));
    let actor = FleetActor {
        nodes: HashMap::new(),
        routes: HashMap::new(),
        node_ids,
        inventories,
        chat_refreshes: HashMap::new(),
        chat_refresh_gen: 0,
        software_versions: HashMap::new(),
        versions: HashMap::new(),
        epochs: HashMap::new(),
        bus: Arc::new(crate::log_bus::LogBus::new()),
        admit_gen: 0,
    };
    let view = actor.nodes_view();
    let age = view[0].inventory_age_ms.expect("age present");
    assert!(
        (7_200_000..7_300_000).contains(&age),
        "a 2h-old stamp reads ~2h at view time, got {age} ms"
    );
    // A FUTURE stamp (backward wall-clock step at view time) saturates to 0 —
    // never a fabricated huge age from unsigned underflow.
    let mut inventories2 = HashMap::new();
    let inv2 = blank_inv();
    // Wall side FUTURE (a backward NTP step at view time zeroes the wall age)
    // with a genuinely old monotonic side: the mono clock must rescue the age
    // instead of the old single-clock saturation-to-0.
    inventories2.insert(
        "nodeA".to_string(),
        (
            inv2,
            PulledAt {
                wall: std::time::SystemTime::now() + std::time::Duration::from_secs(60),
                mono: std::time::Instant::now() - std::time::Duration::from_secs(30),
            },
        ),
    );
    let mut node_ids2 = NodeIdAllocator::new();
    node_ids2.assign("nodeA");
    let actor2 = FleetActor {
        nodes: HashMap::new(),
        routes: HashMap::new(),
        node_ids: node_ids2,
        inventories: inventories2,
        chat_refreshes: HashMap::new(),
        chat_refresh_gen: 0,
        software_versions: HashMap::new(),
        versions: HashMap::new(),
        epochs: HashMap::new(),
        bus: Arc::new(crate::log_bus::LogBus::new()),
        admit_gen: 0,
    };
    let age2 = actor2.nodes_view()[0].inventory_age_ms.expect("age");
    assert!(
        (30_000..31_000).contains(&age2),
        "the monotonic clock rescues the age after a backward wall step: {age2} ms"
    );
}

/// T14 r19: a (re)admission STRIPS the retained snapshot's seq — a restarted
/// node counts snapshot_seq from 1 again, and comparing its fresh pulls
/// against the previous process's high seq would reject every refresh and
/// freeze the old inventory forever. Driven through the REAL AdmitNode
/// handler. Fail-on-revert: drop the strip and the post-restart low-seq
/// commit is rejected.
#[tokio::test]
async fn readmission_strips_the_previous_process_snapshot_seq() {
    let mut node_ids = NodeIdAllocator::new();
    node_ids.assign("nodeA");
    let mut inventories = HashMap::new();
    let mut old_inv = {
        // reuse the blank-inventory shape from the age test
        crate::remote::NodeInventory {
            hostname: "old-process".into(),
            os: "macos".into(),
            workers: vec![],
            snapshot_seq: None,
            hardware: crate::system::HardwareInfo {
                cpu_name: String::new(),
                arch: String::new(),
                cpu_cores: 0,
                ram_total_bytes: 0,
                ram_used_bytes: 0,
                cpu_usage_percent: 0.0,
                gpus: vec![],
                vram_total_bytes: 0,
            },
            runtime: crate::system::RuntimeInfo {
                engine: String::new(),
                backend: String::new(),
                version: String::new(),
                binding: String::new(),
            },
        }
    };
    old_inv.snapshot_seq = Some(1000);
    inventories.insert("nodeA".to_string(), (old_inv, PulledAt::now()));
    let mut actor = FleetActor {
        nodes: HashMap::new(),
        routes: HashMap::new(),
        node_ids,
        inventories,
        chat_refreshes: HashMap::new(),
        chat_refresh_gen: 0,
        software_versions: HashMap::new(),
        versions: HashMap::new(),
        epochs: HashMap::new(),
        bus: Arc::new(crate::log_bus::LogBus::new()),
        admit_gen: 0,
    };
    // Directly exercise the AdmitNode arm's strip: it runs before any
    // transport bookkeeping matters for this assertion, but AdmitNode needs
    // a transport — so replicate the admission-side inventory mutation the
    // handler performs and pin the guard behavior it exists for.
    if let Some((inv, _)) = actor.inventories.get_mut("nodeA") {
        inv.snapshot_seq = None; // what AdmitNode does at (re)admission
    }
    // The restarted node's FIRST pull (seq 1, fresh stamp) must now commit.
    let mut new_inv = actor.inventories.get("nodeA").unwrap().0.clone();
    new_inv.hostname = "new-process".into();
    new_inv.snapshot_seq = Some(1);
    let (tx, _rx) = oneshot::channel();
    actor
        .handle(FleetMsg::CommitInventory {
            node: "nodeA".to_string(),
            epoch_before: 0,
            inventory: Box::new(new_inv),
            pulled_at: PulledAt::now(),
            reply: tx,
        })
        .await;
    assert_eq!(
        actor.nodes_view()[0]
            .inventory
            .as_ref()
            .map(|i| i.hostname.as_str()),
        Some("new-process"),
        "the restarted node's low-seq pull commits after the admission strip"
    );
}

/// T14 r7: same-epoch pull ORDERING — an older-started pull (stalled
/// connect-time fetch) must never overwrite a newer-started one (a chat-end
/// pull that already committed). Driven through the REAL CommitInventory
/// handler with out-of-order dual stamps.
#[tokio::test]
async fn older_started_pull_never_overwrites_a_newer_snapshot() {
    let mut node_ids = NodeIdAllocator::new();
    node_ids.assign("nodeA");
    let mut actor = FleetActor {
        nodes: HashMap::new(),
        routes: HashMap::new(),
        node_ids,
        inventories: HashMap::new(),
        chat_refreshes: HashMap::new(),
        chat_refresh_gen: 0,
        software_versions: HashMap::new(),
        versions: HashMap::new(),
        epochs: HashMap::new(),
        bus: Arc::new(crate::log_bus::LogBus::new()),
        admit_gen: 0,
    };
    fn inv(hostname: &str) -> crate::remote::NodeInventory {
        crate::remote::NodeInventory {
            hostname: hostname.into(),
            os: "macos".into(),
            workers: vec![],
            snapshot_seq: None,
            hardware: crate::system::HardwareInfo {
                cpu_name: String::new(),
                arch: String::new(),
                cpu_cores: 0,
                ram_total_bytes: 0,
                ram_used_bytes: 0,
                cpu_usage_percent: 0.0,
                gpus: vec![],
                vram_total_bytes: 0,
            },
            runtime: crate::system::RuntimeInfo {
                engine: String::new(),
                backend: String::new(),
                version: String::new(),
                binding: String::new(),
            },
        }
    }
    let older = PulledAt {
        wall: std::time::SystemTime::now() - std::time::Duration::from_secs(20),
        mono: std::time::Instant::now() - std::time::Duration::from_secs(20),
    };
    let newer = PulledAt {
        wall: std::time::SystemTime::now() - std::time::Duration::from_secs(1),
        mono: std::time::Instant::now() - std::time::Duration::from_secs(1),
    };
    let commit = |inventory: crate::remote::NodeInventory, pulled_at: PulledAt| {
        let (tx, rx) = oneshot::channel();
        (
            FleetMsg::CommitInventory {
                node: "nodeA".to_string(),
                epoch_before: 0,
                inventory: Box::new(inventory),
                pulled_at,
                reply: tx,
            },
            rx,
        )
    };
    // Newer-started pull commits first...
    let (msg, _rx) = commit(inv("newer"), newer);
    actor.handle(msg).await;
    // ...then the stalled older-started pull arrives: REJECTED.
    let (msg, _rx) = commit(inv("older"), older);
    actor.handle(msg).await;
    let view = actor.nodes_view();
    assert_eq!(
        view[0].inventory.as_ref().map(|i| i.hostname.as_str()),
        Some("newer"),
        "the older-started pull must not overwrite: {view:?}"
    );
    // Sanity: a genuinely newer pull still replaces.
    let newest = PulledAt::now();
    let (msg, _rx) = commit(inv("newest"), newest);
    actor.handle(msg).await;
    assert_eq!(
        actor.nodes_view()[0]
            .inventory
            .as_ref()
            .map(|i| i.hostname.as_str()),
        Some("newest")
    );

    // T14 r17: the NODE's snapshot_seq is DATA order and beats the hub-side
    // stamp — QUIC can serve concurrent pulls out of order, so an EARLIER-
    // stamped pull can carry NEWER node data (and vice versa).
    fn seq_inv(hostname: &str, seq: u64) -> crate::remote::NodeInventory {
        let mut i = inv(hostname);
        i.snapshot_seq = Some(seq);
        i
    }
    // seq 2 arrives (mixed vs the stored seq-LESS snapshot, as after a node
    // upgrade + reconnect: the guard falls back to stamps there, and this
    // fresh pull's stamp is genuinely newer)...
    let (msg, _rx) = commit(seq_inv("seq2", 2), PulledAt::now());
    actor.handle(msg).await;
    assert_eq!(
        actor.nodes_view()[0]
            .inventory
            .as_ref()
            .map(|i| i.hostname.as_str()),
        Some("seq2"),
        "the upgraded node's fresh pull replaces via the stamp fallback"
    );
    // ...then seq 1 arrives with a NEWER hub stamp: REJECTED — data order wins.
    let (msg, _rx) = commit(seq_inv("seq1", 1), PulledAt::now());
    actor.handle(msg).await;
    assert_eq!(
        actor.nodes_view()[0]
            .inventory
            .as_ref()
            .map(|i| i.hostname.as_str()),
        Some("seq2"),
        "an older NODE snapshot never overwrites a newer one, whatever its stamp"
    );
    // And seq 3 with an even OLDER stamp still replaces — the stamp is only
    // the fallback against pre-r17 nodes.
    let older_stamp = PulledAt {
        wall: std::time::SystemTime::now() - std::time::Duration::from_secs(60),
        mono: std::time::Instant::now() - std::time::Duration::from_secs(60),
    };
    let (msg, _rx) = commit(seq_inv("seq3", 3), older_stamp);
    actor.handle(msg).await;
    assert_eq!(
        actor.nodes_view()[0]
            .inventory
            .as_ref()
            .map(|i| i.hostname.as_str()),
        Some("seq3"),
        "newer NODE data commits regardless of its hub stamp"
    );
}

/// T14 r2: the chat-end refresh debounce — at most ONE pull owns a node's
/// slot; completions during it coalesce into exactly one trailing re-run.
/// Driven through the REAL actor messages (Begin/End), not a reimplementation.
#[tokio::test]
async fn chat_refresh_debounce_single_flight_with_trailing_rerun() {
    let fleet = HubFleet::new(Arc::new(crate::log_bus::LogBus::new()));
    let begin = |n: &str| {
        let n = n.to_string();
        let f = &fleet;
        async move {
            f.ask(|reply| FleetMsg::ChatRefreshBegin { node: n, reply })
                .await
                .flatten()
        }
    };
    let end = |n: &str, gen: u64| {
        let n = n.to_string();
        let f = &fleet;
        async move {
            f.ask(|reply| FleetMsg::ChatRefreshEnd {
                node: n,
                gen,
                reply,
            })
            .await
            .unwrap_or(false)
        }
    };
    // First completion owns the slot; a burst of three more all coalesce.
    let gen1 = begin("n").await.expect("first completion owns the refresh");
    assert!(begin("n").await.is_none(), "second coalesces");
    assert!(begin("n").await.is_none(), "third coalesces");
    assert!(begin("n").await.is_none(), "fourth coalesces");
    // The owned pull finishes: exactly ONE trailing re-run is owed (not three).
    assert!(end("n", gen1).await, "coalesced completions => one re-run");
    // The re-run finishes with no further completions: slot released.
    assert!(!end("n", gen1).await, "no more re-runs; slot released");
    // Released means the next completion owns a fresh slot again.
    let gen2 = begin("n").await.expect("slot reusable after release");
    assert_ne!(gen1, gen2, "each owner gets a fresh generation");
    assert!(
        !end("n", gen2).await,
        "clean end with no coalesced completions"
    );
    // Nodes are independent slots.
    assert!(
        begin("other").await.is_some(),
        "another node owns its own slot"
    );

    // T14 r12: a STALE owner (its slot dropped by retire, re-created for a
    // re-paired node) must not mutate the successor's slot — even with a
    // pending trailing flag, the stale End returns false and changes nothing.
    let gen3 = begin("n").await.expect("owner before the retire");
    fleet
        .ask(|reply| FleetMsg::Retire {
            node: "n".to_string(),
            reply,
        })
        .await;
    let gen4 = begin("n").await.expect("successor after re-pair");
    assert_ne!(gen3, gen4);
    assert!(
        begin("n").await.is_none(),
        "burst coalesces into the successor"
    );
    // The stale owner's PRE-PULL check (r13) reports it deposed — it stands
    // down without firing a concurrent pull.
    let still = fleet
        .ask(|reply| FleetMsg::ChatRefreshOwner {
            node: "n".to_string(),
            gen: gen3,
            reply,
        })
        .await
        .unwrap_or(true);
    assert!(!still, "deposed owner is told to stand down before pulling");
    let successor_still = fleet
        .ask(|reply| FleetMsg::ChatRefreshOwner {
            node: "n".to_string(),
            gen: gen4,
            reply,
        })
        .await
        .unwrap_or(false);
    assert!(successor_still, "the live successor keeps its ownership");
    assert!(
        !end("n", gen3).await,
        "the STALE owner exits without touching the successor's slot"
    );
    assert!(
        end("n", gen4).await,
        "the successor still owes its trailing re-run (stale End didn't eat it)"
    );
    assert!(!end("n", gen4).await, "successor releases cleanly");
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
        chat_refreshes: HashMap::new(),
        chat_refresh_gen: 0,
        software_versions: HashMap::new(),
        versions: HashMap::new(),
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
    let _w1 = fleet.load(&node_key, &model_id, None).await.unwrap();
    let _w2 = fleet.load(&node_key, &model_id, None).await.unwrap();
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
    let err = fleet.load("ghost", "m", None).await.unwrap_err();
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
    let w = fleet.load(&node_key, &model_id, None).await.unwrap();
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
    fleet.load(&node_key, &model_id, None).await.unwrap();
    fleet.retire(&node_key).await;
    assert!(fleet.node_ids().await.is_empty());
    assert!(!fleet.is_remote(&model_id).await);
    assert!(fleet
        .chat(&model_id, "[]".into(), 8, 0.0, None, None)
        .await
        .is_err());
}
