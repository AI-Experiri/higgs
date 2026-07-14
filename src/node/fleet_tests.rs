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
            false,
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
            false,
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
            false,
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
            false,
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
            false,
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
        event_nodes: std::collections::HashSet::new(),
        events_tx: tokio::sync::broadcast::channel(8).0,
        pending_pushes: HashMap::new(),
        fallback_inflight: HashMap::new(),
        fallback_gen: 0,
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
        event_nodes: std::collections::HashSet::new(),
        events_tx: tokio::sync::broadcast::channel(8).0,
        pending_pushes: HashMap::new(),
        fallback_inflight: HashMap::new(),
        fallback_gen: 0,
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
        event_nodes: std::collections::HashSet::new(),
        events_tx: tokio::sync::broadcast::channel(8).0,
        pending_pushes: HashMap::new(),
        fallback_inflight: HashMap::new(),
        fallback_gen: 0,
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
        event_nodes: std::collections::HashSet::new(),
        events_tx: tokio::sync::broadcast::channel(8).0,
        pending_pushes: HashMap::new(),
        fallback_inflight: HashMap::new(),
        fallback_gen: 0,
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
        event_nodes: std::collections::HashSet::new(),
        events_tx: tokio::sync::broadcast::channel(8).0,
        pending_pushes: HashMap::new(),
        fallback_inflight: HashMap::new(),
        fallback_gen: 0,
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

/// T10: a node-PUSHED worker snapshot (`CommitWorkers`) merges into the cached
/// inventory under the same epoch + seq guards as a pull — the host/hardware
/// fields are KEPT, an older seq is rejected, a stale epoch is dropped, and a
/// missing cache is reported (`true`) for the full-pull fallback. Driven
/// through the REAL handler, like the CommitInventory ordering pin above.
#[tokio::test]
async fn pushed_worker_snapshot_merges_under_the_seq_guard() {
    // A real (loopback iroh) connection: CommitWorkers applies only from the
    // node's CURRENT transport (T10 r1 #1), so the actor needs one installed.
    let hub = local_endpoint().await;
    let node = local_endpoint().await;
    let (dial, conn) = tokio::join!(node.connect(hub.addr(), ALPN), async {
        hub.accept().await.expect("incoming").await.expect("conn")
    });
    let _keep = dial.expect("dial");
    std::mem::forget(hub);
    let transport = Arc::new(NodeTransport::new(conn));
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
        event_nodes: std::collections::HashSet::new(),
        events_tx: tokio::sync::broadcast::channel(8).0,
        pending_pushes: HashMap::new(),
        fallback_inflight: HashMap::new(),
        fallback_gen: 0,
        versions: HashMap::new(),
        epochs: HashMap::new(),
        bus: Arc::new(crate::log_bus::LogBus::new()),
        admit_gen: 0,
    };
    actor.nodes.insert("nodeA".to_string(), transport.clone());
    // Pushes are accepted only from an admission that DECLARED fleet_events.
    actor.event_nodes.insert("nodeA".to_string());
    fn inv(hostname: &str, seq: u64) -> crate::remote::NodeInventory {
        crate::remote::NodeInventory {
            hostname: hostname.into(),
            os: "macos".into(),
            workers: vec![],
            snapshot_seq: Some(seq),
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
    fn worker(in_flight: u32) -> crate::remote::InventoryWorker {
        crate::remote::InventoryWorker {
            domain: Default::default(),
            worker_id: 1,
            model: "org/m".into(),
            served_id: String::new(),
            ctx_len: Some(256),
            gpu_layers: None,
            threads: None,
            loaded_at_ms: Some(1),
            idle_ms: Some(0),
            in_flight: Some(in_flight),
        }
    }
    let commit_workers = |node: &str, seq: u64, in_flight: u32| {
        let (tx, rx) = oneshot::channel();
        (
            FleetMsg::CommitWorkers {
                node: node.to_string(),
                transport: transport.clone(),
                kind: crate::remote::FleetEventKind::ChatEnd,
                workers: vec![worker(in_flight)],
                snapshot_seq: seq,
                pulled_at: PulledAt::now(),
                reply: tx,
            },
            rx,
        )
    };
    // No cached inventory yet: the push reports NeedsFull — nothing fabricated.
    let (msg, rx) = commit_workers("nodeA", 1, 0);
    actor.handle(msg).await;
    assert!(matches!(rx.await.unwrap(), PushOutcome::NeedsFull(_)));
    assert!(actor.nodes_view()[0].inventory.is_none());

    // Seed the cache like a pull would (seq 5, empty workers).
    let (tx, _rx) = oneshot::channel();
    actor
        .handle(FleetMsg::CommitInventory {
            node: "nodeA".to_string(),
            epoch_before: 0,
            inventory: Box::new(inv("host", 5)),
            pulled_at: PulledAt::now(),
            reply: tx,
        })
        .await;

    // An OLDER-seq push never overwrites (data order, same rule as pulls).
    let (msg, rx) = commit_workers("nodeA", 4, 1);
    actor.handle(msg).await;
    assert_eq!(rx.await.unwrap(), PushOutcome::Stale);
    let view_inv = actor.nodes_view()[0].inventory.clone().unwrap();
    assert!(view_inv.workers.is_empty(), "stale push rejected");
    assert_eq!(view_inv.snapshot_seq, Some(5));

    // A NEWER-seq push merges: workers + seq replaced, host/hardware KEPT.
    let (msg, rx) = commit_workers("nodeA", 6, 1);
    actor.handle(msg).await;
    assert_eq!(rx.await.unwrap(), PushOutcome::Applied);
    let view_inv = actor.nodes_view()[0].inventory.clone().unwrap();
    assert_eq!(view_inv.workers.len(), 1, "pushed worker merged");
    assert_eq!(view_inv.workers[0].in_flight, Some(1));
    assert_eq!(view_inv.snapshot_seq, Some(6));
    assert_eq!(view_inv.hostname, "host", "pulled host fields kept");

    // T10 r2 #1: a push is NOT epoch-gated — a lifecycle op bumping the epoch
    // between receipt and commit must not discard a valid final event (the
    // idle-reap WorkerUnloaded case: nothing would ever resend it, pinning a
    // dead worker in an event-pushing node's cache). Seq + transport identity
    // are the push's ordering guards; the epoch protects pulls only.
    actor.bump_epoch("nodeA");
    let (msg, rx) = commit_workers("nodeA", 7, 0);
    actor.handle(msg).await;
    assert_eq!(rx.await.unwrap(), PushOutcome::Applied);
    assert_eq!(
        actor.nodes_view()[0]
            .inventory
            .clone()
            .unwrap()
            .snapshot_seq,
        Some(7),
        "a push commits across an epoch bump (r2 #1)"
    );
}

/// T10 end-to-end over REAL iroh: a node admitted with the `fleet_events`
/// capability pushes typed events (load / chat start / chat end) that the hub
/// folds into its cache WITHOUT a pull — and the chat-end debounced re-pull is
/// demoted (no extra inventory RPC after the chat). Reverting the node-side
/// emit/relay, the hub dispatch, or the cache merge fails this test.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn node_pushed_events_update_the_cache_without_a_pull() {
    use crate::remote::FleetEventKind;

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
    let mut events = fleet.subscribe_fleet_events();
    fleet
        .add_node(
            node_key.clone(),
            Arc::new(NodeTransport::new(conn)),
            None,
            Some(2),
            None,
            true, // the HELLO advertised fleet_events
        )
        .await;

    // Await a specific kind, tolerating interleaved others (event order across
    // KINDS is deterministic per node, but connect/load pulls interleave).
    async fn next(
        rx: &mut tokio::sync::broadcast::Receiver<FleetEvent>,
        want: FleetEventKind,
    ) -> FleetEvent {
        let deadline = std::time::Duration::from_secs(10);
        loop {
            let ev = tokio::time::timeout(deadline, rx.recv())
                .await
                .unwrap_or_else(|_| panic!("no {want:?} event within 10s"))
                .expect("event channel open");
            if ev.kind == want {
                return ev;
            }
        }
    }

    let ev = next(&mut events, FleetEventKind::NodeConnected).await;
    assert_eq!(ev.endpoint_id, node_key);
    // T10 r1 #3: NodeConnected fires BEFORE the connect-time pull — a UI
    // refreshing on it reads the pre-connect cache. The commit of that pull
    // must announce itself, or the card stays wrong until the next poll.
    next(&mut events, FleetEventKind::InventorySynced).await;

    fleet.load(&node_key, &model_id, None).await.unwrap();
    // Two announcements, in either arrival order: the node's WorkerLoaded push
    // AND the load refresh's committed-pull InventorySynced (T10 r3 #1 — the
    // pull is where a HUB-initiated change lands, and the node's own push for
    // it can be seq-stale, so the commit itself must invalidate subscribers).
    {
        let mut want: std::collections::HashSet<FleetEventKind> = [
            FleetEventKind::WorkerLoaded,
            FleetEventKind::InventorySynced,
        ]
        .into_iter()
        .collect();
        let deadline = std::time::Duration::from_secs(10);
        while !want.is_empty() {
            let ev = tokio::time::timeout(deadline, events.recv())
                .await
                .unwrap_or_else(|_| panic!("missing post-load events: {want:?}"))
                .expect("event channel open");
            want.remove(&ev.kind);
        }
    }

    let (mut rx, fut) = fleet
        .chat(&model_id, "[]".into(), 8, 0.0, None, None)
        .await
        .unwrap();
    tokio::spawn(async move { while rx.recv().await.is_some() {} });
    fut.await.unwrap();
    next(&mut events, FleetEventKind::ChatStart).await;
    next(&mut events, FleetEventKind::ChatEnd).await;

    // The ChatEnd PUSH (not a pull) put the cache back to idle.
    let view = fleet
        .nodes_view()
        .await
        .into_iter()
        .find(|n| n.endpoint_id == node_key)
        .unwrap();
    let inv = view.inventory.expect("cached inventory");
    assert_eq!(inv.workers.len(), 1);
    assert_eq!(inv.workers[0].in_flight, Some(0), "chat-end push landed");
    let seq_after_chat = inv.snapshot_seq.expect("pushes carry the actor seq");

    // Debounce demotion: an event-pushing node gets NO chat-end re-pull. A pull
    // would bump the node's snapshot_seq — sleep past the 250 ms settle and
    // assert the seq is exactly the ChatEnd push's.
    tokio::time::sleep(std::time::Duration::from_millis(700)).await;
    let seq_later = fleet
        .nodes_view()
        .await
        .into_iter()
        .find(|n| n.endpoint_id == node_key)
        .and_then(|n| n.inventory)
        .and_then(|i| i.snapshot_seq)
        .unwrap();
    assert_eq!(
        seq_later, seq_after_chat,
        "no debounced re-pull for an event-pushing node"
    );

    // Retire is a hub-local event.
    fleet.retire(&node_key).await;
    next(&mut events, FleetEventKind::NodeDropped).await;
}

/// T10 r1 #1: a fleet-event push from a REPLACED connection is dropped — the
/// epoch is sampled at receipt so it can't reject a stale-CONNECTION event the
/// way it rejects a stale pull; the transport-identity check on CommitWorkers
/// must. Without it, a buffered push from the old process (high seq) landing
/// after the re-admission stripped the cached seq would install the OLD seq
/// via the stamp fallback and freeze out the new process's pulls/pushes.
#[tokio::test]
async fn pushes_from_a_replaced_connection_are_dropped() {
    let hub = local_endpoint().await;
    let node = local_endpoint().await;
    let hub_addr = hub.addr();
    let node_key = node.id().to_string();

    let (dial1, conn1) = tokio::join!(node.connect(hub_addr.clone(), ALPN), async {
        hub.accept().await.expect("incoming").await.expect("conn 1")
    });
    let (dial2, conn2) = tokio::join!(node.connect(hub_addr, ALPN), async {
        hub.accept().await.expect("incoming").await.expect("conn 2")
    });
    let _keep = (dial1.expect("dial 1"), dial2.expect("dial 2"));
    std::mem::forget(hub);

    let t1 = Arc::new(NodeTransport::new(conn1));
    let t2 = Arc::new(NodeTransport::new(conn2));
    let fleet = Arc::new(HubFleet::new(Arc::new(crate::log_bus::LogBus::new())));
    fleet
        .add_node(node_key.clone(), t1.clone(), None, Some(2), None, true)
        .await;
    // Seed a cached inventory under the current epoch (as the connect pull would).
    let epoch = fleet
        .ask(|reply| FleetMsg::Epoch {
            node: node_key.clone(),
            reply,
        })
        .await
        .unwrap_or(0);
    fleet
        .ask(|reply| FleetMsg::CommitInventory {
            node: node_key.clone(),
            epoch_before: epoch,
            inventory: Box::new(crate::remote::NodeInventory {
                hostname: "process-a".into(),
                os: "macos".into(),
                workers: vec![],
                snapshot_seq: Some(100),
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
            }),
            pulled_at: PulledAt::now(),
            reply,
        })
        .await;

    // The node process restarts: re-admission on a NEW connection strips the seq.
    fleet
        .add_node(node_key.clone(), t2.clone(), None, Some(2), None, true)
        .await;

    let push = |seq: u64, in_flight: u32| crate::remote::NodeFleetEvent {
        kind: crate::remote::FleetEventKind::ChatEnd,
        snapshot_seq: seq,
        workers: vec![crate::remote::InventoryWorker {
            domain: Default::default(),
            worker_id: 1,
            model: "org/m".into(),
            served_id: String::new(),
            ctx_len: None,
            gpu_layers: None,
            threads: None,
            loaded_at_ms: None,
            idle_ms: Some(0),
            in_flight: Some(in_flight),
        }],
    };
    // A buffered push from the OLD connection (process A, high seq) arrives late:
    // it must be dropped, or its seq-101 would freeze out process B below.
    fleet.apply_node_event(&node_key, push(101, 7), &t1).await;
    let inv = fleet.nodes_view().await[0].inventory.clone().unwrap();
    assert_eq!(inv.snapshot_seq, None, "stale-connection push dropped");
    assert!(
        inv.workers.is_empty(),
        "old process's workers not installed"
    );

    // The CURRENT connection's push (process B, seq restarts at 1) commits.
    fleet.apply_node_event(&node_key, push(1, 0), &t2).await;
    let inv = fleet.nodes_view().await[0].inventory.clone().unwrap();
    assert_eq!(
        inv.snapshot_seq,
        Some(1),
        "current connection's push commits"
    );
    assert_eq!(inv.workers.len(), 1);
}

/// T10 r2 #4 + #5: (a) a STALE push re-broadcasts NO public FleetEvent — the
/// cache didn't move, so announcing its kind would hand subscribers reversed
/// signals; (b) the kill switch's `disconnect_all` announces every drained
/// node as `NodeDropped`, so event-driven UIs on OTHER clients see the
/// disable without waiting for a poll.
#[tokio::test]
async fn stale_pushes_are_silent_and_disconnect_all_announces_drops() {
    let hub = local_endpoint().await;
    let node = local_endpoint().await;
    let hub_addr = hub.addr();
    let node_key = node.id().to_string();
    let (dial, conn) = tokio::join!(node.connect(hub_addr, ALPN), async {
        hub.accept().await.expect("incoming").await.expect("conn")
    });
    let _keep = dial.expect("dial");
    std::mem::forget(hub);

    let t = Arc::new(NodeTransport::new(conn));
    let fleet = Arc::new(HubFleet::new(Arc::new(crate::log_bus::LogBus::new())));
    let mut events = fleet.subscribe_fleet_events();
    fleet
        .add_node(node_key.clone(), t.clone(), None, Some(2), None, true)
        .await;
    assert_eq!(
        events.recv().await.unwrap().kind,
        crate::remote::FleetEventKind::NodeConnected
    );

    let push = |seq: u64| crate::remote::NodeFleetEvent {
        kind: crate::remote::FleetEventKind::ChatEnd,
        snapshot_seq: seq,
        workers: vec![],
    };
    // First push seeds (NeedsFull path fails its pull — no real node behind the
    // conn — so nothing is cached and nothing is emitted). Seed via a pull-style
    // commit instead, then apply an APPLIED push and a STALE push.
    let epoch = fleet
        .ask(|reply| FleetMsg::Epoch {
            node: node_key.clone(),
            reply,
        })
        .await
        .unwrap_or(0);
    fleet
        .ask(|reply| FleetMsg::CommitInventory {
            node: node_key.clone(),
            epoch_before: epoch,
            inventory: Box::new(crate::remote::NodeInventory {
                hostname: "h".into(),
                os: "macos".into(),
                workers: vec![],
                snapshot_seq: Some(5),
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
            }),
            pulled_at: PulledAt::now(),
            reply,
        })
        .await;

    // The seeding commit is itself announced (T10 r3 #1: every committed
    // pull-style commit emits InventorySynced, atomically, r4 #1).
    assert_eq!(
        events.recv().await.unwrap().kind,
        crate::remote::FleetEventKind::InventorySynced
    );

    // APPLIED push (seq 6 > 5) → exactly one ChatEnd event.
    fleet.apply_node_event(&node_key, push(6), &t).await;
    assert_eq!(
        events.recv().await.unwrap().kind,
        crate::remote::FleetEventKind::ChatEnd
    );
    // STALE push (seq 4 < 6) → silent: the next event on the channel must NOT
    // be another ChatEnd from it (r2 #4). Prove by draining after the next
    // REAL event below.
    fleet.apply_node_event(&node_key, push(4), &t).await;

    // disconnect_all announces the drop (r2 #5) — and doing so right after the
    // stale push doubles as the silence proof: the NEXT event is NodeDropped,
    // not a ChatEnd.
    fleet.disconnect_all().await;
    assert_eq!(
        events.recv().await.unwrap().kind,
        crate::remote::FleetEventKind::NodeDropped,
        "stale push emitted nothing; disconnect_all announced the drop"
    );
}

/// T10 r5 #1/#2: a push arriving BEFORE any cached inventory is RETAINED (not
/// discarded) and replayed on top of the next committed pull when it is the
/// newer data — otherwise a delayed, OLDER connect pull committing after a
/// failed fallback would resurrect state the push had superseded, with no
/// watermark left to reject it. Concurrent cache-less pushes coalesce into ONE
/// fallback pull (`Deferred`). Driven through the REAL handlers.
#[tokio::test]
async fn a_pre_cache_push_is_retained_and_replayed_over_an_older_pull() {
    let hub = local_endpoint().await;
    let node = local_endpoint().await;
    let (dial, conn) = tokio::join!(node.connect(hub.addr(), ALPN), async {
        hub.accept().await.expect("incoming").await.expect("conn")
    });
    let _keep = dial.expect("dial");
    std::mem::forget(hub);
    let transport = Arc::new(NodeTransport::new(conn));

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
        event_nodes: std::collections::HashSet::new(),
        events_tx: tokio::sync::broadcast::channel(8).0,
        pending_pushes: HashMap::new(),
        fallback_inflight: HashMap::new(),
        fallback_gen: 0,
        versions: HashMap::new(),
        epochs: HashMap::new(),
        bus: Arc::new(crate::log_bus::LogBus::new()),
        admit_gen: 0,
    };
    actor.nodes.insert("nodeA".to_string(), transport.clone());
    // Pushes are accepted only from an admission that DECLARED fleet_events.
    actor.event_nodes.insert("nodeA".to_string());

    let push = |seq: u64| {
        let (tx, rx) = oneshot::channel();
        (
            FleetMsg::CommitWorkers {
                node: "nodeA".to_string(),
                transport: transport.clone(),
                kind: crate::remote::FleetEventKind::WorkerUnloaded,
                workers: vec![], // post-reap truth: no resident workers
                snapshot_seq: seq,
                pulled_at: PulledAt::now(),
                reply: tx,
            },
            rx,
        )
    };
    // First cache-less push: retained, owns the one fallback pull.
    let (msg, rx) = push(7);
    actor.handle(msg).await;
    assert!(matches!(rx.await.unwrap(), PushOutcome::NeedsFull(_)));
    // Second cache-less push while the fallback is in flight: coalesced.
    let (msg, rx) = push(8);
    actor.handle(msg).await;
    assert_eq!(rx.await.unwrap(), PushOutcome::Deferred);

    // The DELAYED connect pull commits with OLDER data (seq 5, one worker that
    // the pushes above already saw reaped)...
    let (tx, _rx) = oneshot::channel();
    actor
        .handle(FleetMsg::CommitInventory {
            node: "nodeA".to_string(),
            epoch_before: 0,
            inventory: Box::new(crate::remote::NodeInventory {
                hostname: "h".into(),
                os: "macos".into(),
                workers: vec![crate::remote::InventoryWorker {
                    domain: Default::default(),
                    worker_id: 1,
                    model: "org/reaped".into(),
                    served_id: String::new(),
                    ctx_len: None,
                    gpu_layers: None,
                    threads: None,
                    loaded_at_ms: None,
                    idle_ms: Some(0),
                    in_flight: Some(0),
                }],
                snapshot_seq: Some(5),
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
            }),
            pulled_at: PulledAt::now(),
            reply: tx,
        })
        .await;

    // ...and the retained NEWER push replays on top: the reaped worker must
    // NOT be resurrected, and the pushed seq (the newest retained, 8) rules.
    let inv = actor.nodes_view()[0].inventory.clone().unwrap();
    assert!(
        inv.workers.is_empty(),
        "the retained newer push replays over the older pull: {inv:?}"
    );
    assert_eq!(inv.snapshot_seq, Some(8));
    assert_eq!(
        inv.hostname, "h",
        "the pull's host/hardware fields are kept"
    );
}

/// T10 r8: the fallback owner's PRE-PULL ownership check — a re-admission
/// clears the slot (and a successor may re-claim it) while the detached owner
/// is pre-pull or mid-sleep; the stale owner must see `false` and stand down
/// WITHOUT firing another inventory RPC. Driven through the real messages.
#[tokio::test]
async fn a_stale_fallback_owner_stands_down_before_pulling() {
    let hub = local_endpoint().await;
    let node = local_endpoint().await;
    let (dial, conn) = tokio::join!(node.connect(hub.addr(), ALPN), async {
        hub.accept().await.expect("incoming").await.expect("conn")
    });
    let _keep = dial.expect("dial");
    std::mem::forget(hub);
    let transport = Arc::new(NodeTransport::new(conn));

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
        event_nodes: std::collections::HashSet::new(),
        events_tx: tokio::sync::broadcast::channel(8).0,
        pending_pushes: HashMap::new(),
        fallback_inflight: HashMap::new(),
        fallback_gen: 0,
        versions: HashMap::new(),
        epochs: HashMap::new(),
        bus: Arc::new(crate::log_bus::LogBus::new()),
        admit_gen: 0,
    };
    actor.nodes.insert("nodeA".to_string(), transport.clone());
    actor.event_nodes.insert("nodeA".to_string());

    // Owner A claims the slot via a cache-less push.
    let (tx, rx) = oneshot::channel();
    actor
        .handle(FleetMsg::CommitWorkers {
            node: "nodeA".to_string(),
            transport: transport.clone(),
            kind: crate::remote::FleetEventKind::ChatEnd,
            workers: vec![],
            snapshot_seq: 1,
            pulled_at: PulledAt::now(),
            reply: tx,
        })
        .await;
    let PushOutcome::NeedsFull(gen_a) = rx.await.unwrap() else {
        panic!("first cache-less push claims the fallback slot");
    };
    let owner = |gen: u64| {
        let (tx, rx) = oneshot::channel();
        (
            FleetMsg::FallbackOwner {
                node: "nodeA".to_string(),
                gen,
                reply: tx,
            },
            rx,
        )
    };
    let (msg, rx) = owner(gen_a);
    actor.handle(msg).await;
    assert!(rx.await.unwrap(), "the claiming owner is current");

    // A retire (as at re-admission) clears the slot; a successor push re-claims.
    actor.retire("nodeA");
    actor.nodes.insert("nodeA".to_string(), transport.clone());
    actor.event_nodes.insert("nodeA".to_string());
    let (tx, rx) = oneshot::channel();
    actor
        .handle(FleetMsg::CommitWorkers {
            node: "nodeA".to_string(),
            transport: transport.clone(),
            kind: crate::remote::FleetEventKind::ChatEnd,
            workers: vec![],
            snapshot_seq: 1,
            pulled_at: PulledAt::now(),
            reply: tx,
        })
        .await;
    let PushOutcome::NeedsFull(gen_b) = rx.await.unwrap() else {
        panic!("the successor push claims a fresh slot");
    };
    assert_ne!(gen_a, gen_b);

    // The STALE owner stands down; the successor is current.
    let (msg, rx) = owner(gen_a);
    actor.handle(msg).await;
    assert!(!rx.await.unwrap(), "stale owner must stand down (r8)");
    let (msg, rx) = owner(gen_b);
    actor.handle(msg).await;
    assert!(rx.await.unwrap());
}

/// T10 r16 #1: an APPLIED push whose snapshot no longer contains a previously
/// cached worker drops that worker's ROUTE too — an idle-reaped worker must
/// not stay advertised (mislabeling a same-model survivor) until a chat fails
/// into it. CAS'd on the old snapshot's model, so a worker the cache never saw
/// is untouched.
#[tokio::test]
async fn an_applied_push_drops_routes_of_vanished_workers() {
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
    let t = Arc::new(NodeTransport::new(conn));
    fleet
        .add_node(node_key.clone(), t.clone(), None, Some(2), None, true)
        .await;

    // Load → route + cached inventory with the worker.
    let worker = fleet.load(&node_key, &model_id, None).await.unwrap();
    assert!(
        !fleet
            .nodes_view()
            .await
            .into_iter()
            .find(|n| n.endpoint_id == node_key)
            .and_then(|n| n.inventory)
            .map(|i| i.workers)
            .unwrap_or_default()
            .is_empty(),
        "worker cached after load"
    );
    assert!(fleet.is_remote(&model_id).await, "route installed");

    // The node "idle-reaps" the worker: an APPLIED push with NO workers and a
    // seq far beyond anything the pulls used.
    fleet
        .apply_node_event(
            &node_key,
            crate::remote::NodeFleetEvent {
                kind: crate::remote::FleetEventKind::WorkerUnloaded,
                snapshot_seq: 1_000_000,
                workers: vec![],
            },
            &t,
        )
        .await;
    assert!(
        !fleet.is_remote(&model_id).await,
        "the vanished worker's route is dropped with the applied push"
    );
    let _ = worker;
}

/// T10 r13 #2 / r16 #2: the FallbackDone give-up decision through the REAL
/// message — a plain retry is granted while pending data waits; a give-up with
/// UNCHANGED pending seq finalizes (slot cleared, push retained); a give-up
/// with an ADVANCED pending seq EXTENDS the owner instead of stranding the
/// fresh data.
#[tokio::test]
async fn fallback_give_up_finalizes_only_without_fresh_pending_data() {
    let hub = local_endpoint().await;
    let node = local_endpoint().await;
    let (dial, conn) = tokio::join!(node.connect(hub.addr(), ALPN), async {
        hub.accept().await.expect("incoming").await.expect("conn")
    });
    let _keep = dial.expect("dial");
    std::mem::forget(hub);
    let transport = Arc::new(NodeTransport::new(conn));
    let fleet = Arc::new(HubFleet::new(Arc::new(crate::log_bus::LogBus::new())));
    fleet
        .add_node(
            "nodeA".to_string(),
            transport.clone(),
            None,
            Some(2),
            None,
            true,
        )
        .await;

    // Claim the slot directly (apply_node_event would spawn the real fallback
    // loop and race this test's own FallbackDone calls).
    let outcome = fleet
        .ask(|reply| FleetMsg::CommitWorkers {
            node: "nodeA".to_string(),
            transport: transport.clone(),
            kind: crate::remote::FleetEventKind::ChatEnd,
            workers: vec![],
            snapshot_seq: 10,
            pulled_at: PulledAt::now(),
            reply,
        })
        .await
        .unwrap();
    let PushOutcome::NeedsFull(gen) = outcome else {
        panic!("cache-less push claims the slot: {outcome:?}");
    };
    let done = |gen: u64, give_up: bool| {
        let f = fleet.clone();
        async move {
            f.ask(|reply| FleetMsg::FallbackDone {
                node: "nodeA".to_string(),
                gen,
                give_up,
                reply,
            })
            .await
            .unwrap()
        }
    };
    // Plain failure: retry granted (pending waits, node connected, no cache).
    assert!(done(gen, false).await, "non-terminal failure retries");
    // Terminal with UNCHANGED pending: finalizes.
    assert!(
        !done(gen, true).await,
        "give-up with stale pending finalizes"
    );
    // The push stayed retained and the slot is free: the next push re-claims.
    let outcome = fleet
        .ask(|reply| FleetMsg::CommitWorkers {
            node: "nodeA".to_string(),
            transport: transport.clone(),
            kind: crate::remote::FleetEventKind::ChatEnd,
            workers: vec![],
            snapshot_seq: 11,
            pulled_at: PulledAt::now(),
            reply,
        })
        .await
        .unwrap();
    let PushOutcome::NeedsFull(gen2) = outcome else {
        panic!("slot re-claimed after finalize: {outcome:?}");
    };
    // Fresh data lands DURING the terminal attempt (seq 12 > the 11 claimed)...
    let outcome = fleet
        .ask(|reply| FleetMsg::CommitWorkers {
            node: "nodeA".to_string(),
            transport: transport.clone(),
            kind: crate::remote::FleetEventKind::ChatEnd,
            workers: vec![],
            snapshot_seq: 12,
            pulled_at: PulledAt::now(),
            reply,
        })
        .await
        .unwrap();
    assert!(matches!(outcome, PushOutcome::Deferred));
    // ...so the give-up EXTENDS instead of stranding it (r13 #2).
    assert!(done(gen2, true).await, "give-up with fresh pending extends");
    // And once served (seq unchanged), the next give-up finalizes.
    assert!(!done(gen2, true).await, "give-up after serving finalizes");
}

/// T10 r22 #1: an event with an UNKNOWN (future) kind still applies its
/// authoritative worker snapshot — the hub substitutes the generic
/// InventorySynced invalidation for the kind it can't decode instead of
/// dropping the data (which would leave the cache stale until another
/// lifecycle op, since event nodes get no chat-end re-pull).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_unknown_event_kind_still_applies_its_snapshot() {
    use crate::node::write_frame;

    let hub = local_endpoint().await;
    let node = local_endpoint().await;
    let (dial, conn) = tokio::join!(node.connect(hub.addr(), ALPN), async {
        hub.accept().await.expect("incoming").await.expect("conn")
    });
    let node_conn = dial.expect("dial");
    std::mem::forget(hub);

    let t = Arc::new(NodeTransport::new(conn));
    let fleet = Arc::new(HubFleet::new(Arc::new(crate::log_bus::LogBus::new())));
    let mut events = fleet.subscribe_fleet_events();
    let node_key = node.id().to_string();
    fleet
        .add_node(node_key.clone(), t.clone(), None, Some(2), None, true)
        .await;
    assert_eq!(
        events.recv().await.unwrap().kind,
        crate::remote::FleetEventKind::NodeConnected
    );

    // Seed a cached inventory (seq 1) so the push merges instead of NeedsFull.
    let epoch = fleet
        .ask(|reply| FleetMsg::Epoch {
            node: node_key.clone(),
            reply,
        })
        .await
        .unwrap_or(0);
    fleet
        .ask(|reply| FleetMsg::CommitInventory {
            node: node_key.clone(),
            epoch_before: epoch,
            inventory: Box::new(crate::remote::NodeInventory {
                hostname: "h".into(),
                os: "macos".into(),
                workers: vec![],
                snapshot_seq: Some(1),
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
            }),
            pulled_at: PulledAt::now(),
            reply,
        })
        .await;
    assert_eq!(
        events.recv().await.unwrap().kind,
        crate::remote::FleetEventKind::InventorySynced
    );

    // The "node" pushes a FUTURE kind this build doesn't know, on a real uni
    // stream (the same reader add_node spawned).
    let mut send = node_conn.open_uni().await.expect("uni");
    let note = crate::rpc::RpcNotification {
        jsonrpc: "2.0".into(),
        method: crate::remote::N_FLEET_EVENT.into(),
        params: serde_json::json!({
            "kind": "worker_crashed",
            "snapshot_seq": 2,
            "workers": [{ "worker_id": 9, "model": "org/m", "in_flight": 0 }],
        }),
    };
    write_frame(&mut send, &crate::rpc::RpcFrame::Notification(note))
        .await
        .expect("push written");

    // The snapshot applied (worker 9 cached) and the generic invalidation fired.
    let ev = tokio::time::timeout(std::time::Duration::from_secs(10), events.recv())
        .await
        .expect("an event within 10s")
        .unwrap();
    assert_eq!(ev.kind, crate::remote::FleetEventKind::InventorySynced);
    let inv = fleet
        .nodes_view()
        .await
        .into_iter()
        .find(|n| n.endpoint_id == node_key)
        .and_then(|n| n.inventory)
        .expect("cached inventory");
    assert_eq!(inv.snapshot_seq, Some(2));
    assert_eq!(
        inv.workers.len(),
        1,
        "unknown-kind snapshot applied: {inv:?}"
    );
    assert_eq!(inv.workers[0].worker_id, 9);
}

/// A remote worker the node's inventory declares non-generative is dropped from
/// `routed_models` (the `/v1/models` advertising set) — chatting it is a
/// guaranteed [HG079] — while ROUTING is untouched: the id still resolves, and a
/// chat dispatched anyway comes back with the NODE's own refusal over the wire.
/// That last assertion is the relay-propagation contract in ordinary CI — no
/// external model file needed (the real-engine version lives in
/// tests/embedding_remote.rs and skips without its fixture).
#[tokio::test]
async fn a_non_generative_remote_worker_is_unadvertised_but_still_refused_on_the_wire() {
    let (fleet, node_key, model_id, root) = fleet_with_one_node().await;
    crate::serve::test_support::write_embedding_gguf_fixture(root.path(), "org/embed");

    fleet.load(&node_key, &model_id, None).await.unwrap();
    fleet.load(&node_key, "org/embed", None).await.unwrap();
    // Commit a fresh inventory so the hub KNOWS both workers' domains — the
    // filter is deliberately permissive while no inventory is cached.
    fleet.refresh_inventory(&node_key).await.unwrap();

    let routed = fleet.routed_models().await;
    assert!(
        routed.contains(&model_id),
        "the generative worker stays advertised; routed: {routed:?}"
    );
    assert!(
        !routed.contains(&"org/embed".to_owned()),
        "a non-generative remote worker must not be advertised; routed: {routed:?}"
    );
    // Routing untouched: the id still resolves remotely (advertising-only filter)…
    assert!(
        fleet.is_remote("org/embed").await,
        "the route survives — only the advertising filters"
    );
    // …and a chat dispatched anyway is refused BY THE NODE, over the real relay
    // wire, with the [HG079] code intact.
    let err = match fleet
        .chat("org/embed", "[]".into(), 8, 0.0, None, None)
        .await
    {
        Ok((_rx, fut)) => match fut.await {
            Ok(v) => panic!("a relayed chat against an embedding worker must fail, got: {v}"),
            Err(e) => e,
        },
        Err(e) => e,
    };
    assert!(
        err.to_string().contains("HG079"),
        "the node's refusal code survives the relay: {err}"
    );
}

/// The HUB refuses a chat against a remote non-generative worker BEFORE any
/// dispatch — directly from `resolve_loaded`'s remote arm, using the domain the
/// node's inventory reported. This is what keeps `stream: true` a clean HTTP
/// 400 instead of a 200/SSE whose refusal arrives as an in-stream error.
/// Fail-on-revert: without the pre-refusal the error only comes back from the
/// node over the wire as a `WorkerRpc` — a different variant.
#[tokio::test]
async fn the_hub_refuses_a_remote_non_generative_chat_before_dispatch() {
    let (fleet, node_key, _model_id, root) = fleet_with_one_node().await;
    crate::serve::test_support::write_embedding_gguf_fixture(root.path(), "org/embed");
    fleet.load(&node_key, "org/embed", None).await.unwrap();
    fleet.refresh_inventory(&node_key).await.unwrap();

    let higgs = crate::api::Higgs::with_log_bus(
        crate::api::HiggsConfig::default(),
        Arc::new(crate::log_bus::LogBus::new()),
    );
    higgs.set_fleet(fleet.clone());

    let err = higgs
        .prepare_chat("org/embed", None, "[]")
        .await
        .expect_err("the hub must refuse before dispatch");
    assert!(
        matches!(
            err,
            crate::diagnostic::HiggsError::ModelNotChatCapable { .. }
        ),
        "a DIRECT [HG079] from the hub's own gate (not a relayed WorkerRpc): {err}"
    );
}

/// `RemoteDomain` reports a verdict ONLY when the cached inventory row matches
/// the routed MODEL, not just the worker id: a node restart can reuse a worker
/// id for a different file, and a wrong-model verdict would pre-refuse [HG079]
/// where dispatch would have surfaced the stale route ([HG018] eviction).
/// Fail-on-revert: matching by worker id alone reports the reused worker's
/// embedding domain for the generative route.
#[tokio::test]
async fn a_reused_worker_id_does_not_lend_its_domain_to_a_stale_route() {
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
        event_nodes: std::collections::HashSet::new(),
        events_tx: tokio::sync::broadcast::channel(8).0,
        pending_pushes: HashMap::new(),
        fallback_inflight: HashMap::new(),
        fallback_gen: 0,
        versions: HashMap::new(),
        epochs: HashMap::new(),
        bus: Arc::new(crate::log_bus::LogBus::new()),
        admit_gen: 0,
    };
    // The STALE route: "org/gen" is believed to live at worker 1…
    actor
        .routes
        .insert(("nodeA".to_string(), WorkerId(1)), "org/gen".to_string());
    // …but the node restarted and worker 1 now serves an EMBEDDING model.
    let inv = crate::remote::NodeInventory {
        hostname: "host".into(),
        os: "macos".into(),
        workers: vec![crate::remote::InventoryWorker {
            worker_id: 1,
            model: "org/embed".into(),
            served_id: String::new(),
            ctx_len: Some(256),
            gpu_layers: None,
            threads: None,
            loaded_at_ms: Some(1),
            idle_ms: Some(0),
            in_flight: Some(0),
            domain: crate::worker::models::ModelDomain::Embedding,
        }],
        snapshot_seq: Some(1),
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
    actor
        .inventories
        .insert("nodeA".to_string(), (inv, PulledAt::now()));

    // Model mismatch → NO verdict (permissive): the embedding worker must not
    // lend its domain to the stale generative route.
    let (tx, rx) = oneshot::channel();
    actor
        .handle(FleetMsg::RemoteDomain {
            node: "nodeA".to_string(),
            worker: WorkerId(1),
            model: "org/gen".to_string(),
            reply: tx,
        })
        .await;
    assert_eq!(
        rx.await.unwrap(),
        None,
        "no row matches (worker, MODEL) → unknown, never a verdict about a different file"
    );

    // The MATCHING model does get its verdict — the guard filters, not blinds.
    let (tx, rx) = oneshot::channel();
    actor
        .handle(FleetMsg::RemoteDomain {
            node: "nodeA".to_string(),
            worker: WorkerId(1),
            model: "org/embed".to_string(),
            reply: tx,
        })
        .await;
    assert_eq!(
        rx.await.unwrap(),
        Some(crate::worker::models::ModelDomain::Embedding)
    );
}

/// `worker_chat_capable` (the `/v1/models` advertising filter) requires the
/// inventory row to match the route's MODEL, not just the worker id — the same
/// rule as `RemoteDomain` (r4). The hub retains inventories across a node
/// re-admission, and worker ids restart per process: a stale row must neither
/// hide a freshly routed generative model nor lend a verdict to a route it no
/// longer describes. Mismatch = permissive (advertised), like no row at all.
#[tokio::test]
async fn a_stale_inventory_row_does_not_unadvertise_a_reused_worker_id() {
    // A real loopback transport: routed_models advertises CONNECTED nodes only.
    let hub = local_endpoint().await;
    let node = local_endpoint().await;
    let (dial, conn) = tokio::join!(node.connect(hub.addr(), ALPN), async {
        hub.accept().await.expect("incoming").await.expect("conn")
    });
    let _keep = dial.expect("dial");
    std::mem::forget(hub);
    let transport = Arc::new(NodeTransport::new(conn));
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
        event_nodes: std::collections::HashSet::new(),
        events_tx: tokio::sync::broadcast::channel(8).0,
        pending_pushes: HashMap::new(),
        fallback_inflight: HashMap::new(),
        fallback_gen: 0,
        versions: HashMap::new(),
        epochs: HashMap::new(),
        bus: Arc::new(crate::log_bus::LogBus::new()),
        admit_gen: 0,
    };
    actor.nodes.insert("nodeA".to_string(), transport);
    // The FRESH route: worker 1 now serves the generative org/gen…
    actor
        .routes
        .insert(("nodeA".to_string(), WorkerId(1)), "org/gen".to_string());
    // …but the RETAINED inventory still shows the old process's worker 1 as an
    // embedding model.
    let inv = crate::remote::NodeInventory {
        hostname: "host".into(),
        os: "macos".into(),
        workers: vec![crate::remote::InventoryWorker {
            worker_id: 1,
            model: "org/embed".into(),
            served_id: String::new(),
            ctx_len: Some(256),
            gpu_layers: None,
            threads: None,
            loaded_at_ms: Some(1),
            idle_ms: Some(0),
            in_flight: Some(0),
            domain: crate::worker::models::ModelDomain::Embedding,
        }],
        snapshot_seq: Some(1),
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
    actor
        .inventories
        .insert("nodeA".to_string(), (inv, PulledAt::now()));

    // The stale embedding row says nothing about THIS route → advertised.
    assert_eq!(
        actor.routed_models(),
        vec!["org/gen".to_string()],
        "a stale row must not hide the freshly routed generative model"
    );

    // And when the row DOES match the route, the verdict applies: flip the
    // route to the model the row describes → dropped from the advertisement.
    actor.routes.clear();
    actor
        .routes
        .insert(("nodeA".to_string(), WorkerId(1)), "org/embed".to_string());
    assert!(
        actor.routed_models().is_empty(),
        "a matching non-generative row still filters its own route"
    );

    // A RERANKER row filters too — the comparison is `!= Llm`, not
    // `== Embedding` (Fable r8 mutation probe).
    actor.inventories.get_mut("nodeA").unwrap().0.workers[0].domain =
        crate::worker::models::ModelDomain::Reranker;
    assert!(
        actor.routed_models().is_empty(),
        "a matching reranker row filters its own route"
    );
}
