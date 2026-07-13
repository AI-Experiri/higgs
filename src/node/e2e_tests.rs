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

/// T10 r1 #2: the fleet-event relay must SURVIVE a stream-level failure while
/// the connection stays healthy — the hub skips its chat-end re-pull for an
/// event-advertising node, so a relay that died with the connection alive
/// would freeze the hub's cache (no push AND no pull). The hub side resets
/// (STOP_SENDING) the first uni stream; the relay must reopen a fresh one and
/// events must keep arriving on it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fleet_event_relay_reopens_after_a_stream_reset() {
    use tokio::io::AsyncBufReadExt;

    let (model_root, model_id) = stage_dummy_model("higgs-test/m");
    let hub = local_endpoint().await;
    let node = local_endpoint().await;
    let (dial, hub_conn) = tokio::join!(node.connect(hub.addr(), crate::remote::ALPN), async {
        hub.accept().await.expect("incoming").await.expect("conn")
    });
    let node_conn = dial.expect("dial");
    std::mem::forget(hub);

    let rt = Arc::new(fake_runtime(vec![model_root.path().to_path_buf()]));
    let fleet_events = rt.subscribe_fleet_events();
    tokio::spawn(super::relay_fleet_events(
        node_conn,
        rt.clone(),
        fleet_events,
    ));

    // Emit fleet events continuously (each fake load emits WorkerLoaded) so
    // both the pre- and post-reset streams have traffic without timing games.
    let rt_load = rt.clone();
    let model = model_id.clone();
    let feeder = tokio::spawn(async move {
        loop {
            let _ = rt_load
                .load(crate::remote::NodeLoadParams {
                    id: model.clone(),
                    ctx_len: None,
                    gpu_layers: None,
                    threads: None,
                    params: None,
                })
                .await;
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }
    });

    // Read one event off the FIRST relay stream, then reset it hub-side.
    let mut recv1 = tokio::time::timeout(std::time::Duration::from_secs(10), async {
        hub_conn.accept_uni().await.expect("first uni stream")
    })
    .await
    .expect("relay opened its first stream");
    let mut lines1 = tokio::io::BufReader::new(&mut recv1).lines();
    let line1 = tokio::time::timeout(std::time::Duration::from_secs(10), lines1.next_line())
        .await
        .expect("an event within 10s")
        .expect("stream readable")
        .expect("a line");
    assert!(
        line1.contains(crate::remote::N_FLEET_EVENT),
        "first stream carries fleet events: {line1}"
    );
    drop(lines1);
    recv1.stop(1u32.into()).expect("reset the first stream");

    // The relay must reopen: a SECOND uni stream arrives, still carrying events.
    let recv2 = tokio::time::timeout(std::time::Duration::from_secs(10), async {
        hub_conn.accept_uni().await.expect("second uni stream")
    })
    .await
    .expect("relay reopened after the stream reset");
    let mut lines2 = tokio::io::BufReader::new(recv2).lines();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    loop {
        let line = tokio::time::timeout(std::time::Duration::from_secs(10), lines2.next_line())
            .await
            .expect("an event on the reopened stream within 10s")
            .expect("stream readable")
            .expect("a line");
        if line.contains(crate::remote::N_FLEET_EVENT) {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "no event after reopen"
        );
    }
    feeder.abort();
}

/// T10 r2 #2: a reset landing while the relay is IDLE — after its FINAL event
/// was buffered locally but possibly before the hub decoded it — must still be
/// detected (`send.stopped()`), and the LAST event must be resent on the fresh
/// stream with NO new node activity. Otherwise a lost trailing ChatEnd pins the
/// hub at `in_flight: 1` forever.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fleet_event_relay_resends_the_last_event_after_an_idle_reset() {
    use tokio::io::AsyncBufReadExt;

    let (model_root, model_id) = stage_dummy_model("higgs-test/m");
    let hub = local_endpoint().await;
    let node = local_endpoint().await;
    let (dial, hub_conn) = tokio::join!(node.connect(hub.addr(), crate::remote::ALPN), async {
        hub.accept().await.expect("incoming").await.expect("conn")
    });
    let node_conn = dial.expect("dial");
    std::mem::forget(hub);

    let rt = Arc::new(fake_runtime(vec![model_root.path().to_path_buf()]));
    let fleet_events = rt.subscribe_fleet_events();
    tokio::spawn(super::relay_fleet_events(
        node_conn,
        rt.clone(),
        fleet_events,
    ));

    // Exactly ONE event (a single load), then the node goes idle.
    rt.load(crate::remote::NodeLoadParams {
        id: model_id,
        ctx_len: None,
        gpu_layers: None,
        threads: None,
        params: None,
    })
    .await
    .expect("fake load");

    let mut recv1 = tokio::time::timeout(std::time::Duration::from_secs(10), async {
        hub_conn.accept_uni().await.expect("first uni stream")
    })
    .await
    .expect("relay opened its first stream");
    let mut lines1 = tokio::io::BufReader::new(&mut recv1).lines();
    let line1 = tokio::time::timeout(std::time::Duration::from_secs(10), lines1.next_line())
        .await
        .expect("the load event within 10s")
        .expect("stream readable")
        .expect("a line");
    assert!(line1.contains(crate::remote::N_FLEET_EVENT));
    drop(lines1);
    // Idle reset: no further node activity will ever produce another write.
    recv1.stop(1u32.into()).expect("reset the stream");

    // The relay must notice via stopped() and resend the LAST event unprompted.
    let recv2 = tokio::time::timeout(std::time::Duration::from_secs(10), async {
        hub_conn.accept_uni().await.expect("second uni stream")
    })
    .await
    .expect("relay reopened after the idle reset");
    let mut lines2 = tokio::io::BufReader::new(recv2).lines();
    let line2 = tokio::time::timeout(std::time::Duration::from_secs(10), lines2.next_line())
        .await
        .expect("the resent event within 10s — with NO new node activity")
        .expect("stream readable")
        .expect("a line");
    assert!(
        line2.contains(crate::remote::N_FLEET_EVENT),
        "a recovery event arrives on the fresh stream: {line2}"
    );
    // r26: the recovery is a FRESH Resync snapshot, not a verbatim resend — its
    // seq must ADVANCE past the lost event's (a resend would repeat it, letting
    // frozen idle_ms be stamped fresh at the hub), and its kind must be Resync
    // (reusing the lost kind would misdescribe the newer snapshot).
    let decode = |line: &str| -> (u64, String) {
        let frame: serde_json::Value = serde_json::from_str(line).expect("frame json");
        (
            frame["params"]["snapshot_seq"].as_u64().expect("seq"),
            frame["params"]["kind"].as_str().expect("kind").to_string(),
        )
    };
    let (seq1, _) = decode(&line1);
    let (seq2, kind2) = decode(&line2);
    assert!(
        seq2 > seq1,
        "recovery snapshot is FRESH (seq {seq2} > {seq1}), not a resend"
    );
    assert_eq!(kind2, "resync", "recovery carries the Resync kind");
}
