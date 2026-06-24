//! P2 exit test: a node hosts 2 concurrent workers and answers the hub's control RPCs
//! (`M_NODE_LOAD` ×2, `M_NODE_SYSINFO`, `M_NODE_STATUS`) over a real iroh link.
//!
//! Workers are fakes (no llama.cpp) so the test is fast and self-contained, but the
//! transport, HELLO gate, control dispatch, model resolution, and registry are all real.

use std::sync::Arc;

use serde_json::json;

use crate::auth::{Allowlist, PairingTokens};
use crate::node::test_support::{fake_runtime, local_endpoint, node_rpc, stage_dummy_model};
use crate::node::{
    connect_node, gate_connection, serve_node, GateOutcome, HubIdentity, HELLO_DEADLINE,
};
use crate::remote::{M_NODE_LOAD, M_NODE_STATUS, M_NODE_SYSINFO};

#[tokio::test]
async fn node_hosts_two_workers_sysinfo_status_over_iroh() {
    let (model_root, model_id) = stage_dummy_model("higgs-test/m");
    let hub = local_endpoint().await;
    let node = local_endpoint().await;
    let hub_addr = hub.addr();
    let hub_id = hub.id().to_string();
    let node_id = node.id().to_string();

    // Pre-allowlist the node (reconnect path — no token needed).
    let allow_path =
        std::env::temp_dir().join(format!("higgs-e2e-allow-{}.json", std::process::id()));
    let _ = std::fs::remove_file(&allow_path);
    let mut allow = Allowlist::load(&allow_path).unwrap();
    allow
        .add(node_id.clone(), Some("test-node".into()))
        .unwrap();

    // Node side runs in a task: connect + serve the hub's control RPCs.
    let rt = Arc::new(fake_runtime(vec![model_root.path().to_path_buf()]));
    let rt_node = rt.clone();
    let node_task = tokio::spawn(async move {
        let (node_conn, _hello) = connect_node(&node, hub_addr, node_id, String::new(), None)
            .await
            .expect("connect");
        serve_node(node_conn, rt_node).await; // runs until the connection closes
    });

    // Hub side runs inline so its `Endpoint` stays alive for the whole test: accept the
    // node's dial, gate it (admit), then drive control RPCs on the same connection.
    let incoming = hub.accept().await.expect("incoming");
    let conn = incoming.await.expect("conn");
    let mut tokens = PairingTokens::new();
    let outcome = gate_connection(
        &conn,
        &mut allow,
        &mut tokens,
        1,
        &HubIdentity::new(hub_id),
        None,
        HELLO_DEADLINE,
    )
    .await;
    assert!(
        matches!(outcome, GateOutcome::Admitted { .. }),
        "node admitted"
    );

    // Load two workers over iroh.
    let r1 = node_rpc(&conn, 1, M_NODE_LOAD, json!({ "id": model_id })).await;
    let w1 = r1.result.expect("load1 ok")["worker_id"].as_u64().unwrap();
    let r2 = node_rpc(&conn, 2, M_NODE_LOAD, json!({ "id": model_id })).await;
    let w2 = r2.result.expect("load2 ok")["worker_id"].as_u64().unwrap();
    assert_ne!(w1, w2, "two distinct workers");
    assert_eq!(
        rt.worker_ids().await.len(),
        2,
        "node hosts 2 concurrent workers"
    );

    // M_SYSINFO (node-level) + M_STATUS (per worker) over iroh.
    let sys = node_rpc(&conn, 3, M_NODE_SYSINFO, json!({})).await;
    assert!(sys.error.is_none(), "sysinfo ok: {sys:?}");
    let hw = sys.result.unwrap();
    assert!(
        hw["hardware"]["cpu_cores"].as_u64().unwrap() > 0,
        "real cpu cores"
    );
    assert!(
        hw["hardware"]["ram_total_bytes"].as_u64().unwrap() > 0,
        "real ram"
    );
    assert!(hw["hardware"].get("gpus").is_some(), "gpu list present");

    let st = node_rpc(&conn, 4, M_NODE_STATUS, json!({ "worker_id": w1 })).await;
    assert!(st.error.is_none(), "status ok: {st:?}");
    assert!(st.result.unwrap().get("loaded").is_some());

    // Teardown.
    rt.shutdown_all().await;
    node_task.abort();
    let _ = std::fs::remove_file(&allow_path);
}
