//! NQ integration test: passive network stats over a live in-process iroh link.
//!
//! Two endpoints in one process → hub accepts + gates the node → `fleet.add_node`
//! → `Higgs::network_stats(peer)` returns Some(NetworkStats) with a real path
//! sample and a monotonic uptime. Fail-on-revert: if `network_sample`'s iroh
//! reads compile but return no selected path, or if `connected_at` isn't
//! stamped on admit, the shape assertions fire.

use std::sync::Arc;
use std::time::Duration;

use higgs::auth::{Allowlist, PairingTokens};
use higgs::log_bus::LogBus;
use higgs::node::fleet::HubFleet;
use higgs::node::transport::NodeTransport;
use higgs::node::{connect_node, gate_connection, GateOutcome, HubIdentity, HELLO_DEADLINE};
use higgs::remote::{LinkPath, LinkState, ALPN};
use higgs::{Higgs, HiggsConfig};

async fn local_endpoint() -> iroh::Endpoint {
    iroh::Endpoint::builder(iroh::endpoint::presets::Minimal)
        .alpns(vec![ALPN.to_vec()])
        .relay_mode(iroh::RelayMode::Disabled)
        .bind()
        .await
        .expect("bind local endpoint")
}

/// Higgs::network_stats returns None when the hub is not running.
#[tokio::test]
async fn network_stats_none_when_hub_disabled() {
    let higgs = Higgs::new(HiggsConfig::default());
    assert!(
        higgs.network_stats("anything").await.is_none(),
        "no fleet installed → None"
    );
}

/// Fleet with no nodes admitted → any query reports Disconnected + no uptime.
#[tokio::test]
async fn network_stats_unpaired_reports_disconnected() {
    let bus = Arc::new(LogBus::new());
    let higgs = Arc::new(Higgs::with_log_bus(HiggsConfig::default(), bus.clone()));
    let fleet = Arc::new(HubFleet::new(bus));
    higgs.set_fleet(fleet);
    let stats = higgs
        .network_stats("never-paired")
        .await
        .expect("hub is up");
    assert_eq!(stats.state, LinkState::Disconnected);
    assert_eq!(stats.uptime_ms, None);
    assert_eq!(stats.path, None);
}

/// Full end-to-end over a live in-process iroh connection: after admit the
/// crate reports a live sample and a monotonic uptime, then Disconnected + no
/// uptime after retire.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn network_stats_live_admit_then_retire() {
    // Endpoints must OUTLIVE the connection — dropping either Endpoint
    // tears down its Connections.
    let hub_ep = local_endpoint().await;
    let node_ep = local_endpoint().await;
    let hub_addr = hub_ep.addr();
    let hub_id = hub_ep.id().to_string();
    let node_id = node_ep.id().to_string();

    let allow_path =
        std::env::temp_dir().join(format!("higgs-nq-allow-{}.json", std::process::id()));
    let _ = std::fs::remove_file(&allow_path);
    let mut allow = Allowlist::load(&allow_path).unwrap();
    let mut tokens = PairingTokens::new();
    let tok = tokens.mint(1_000, 600_000);

    // connect_node returns (Connection, HelloResult); we KEEP the node-side
    // Connection alive for the whole test so the hub-side conn doesn't get
    // torn down under us.
    let dial_task = tokio::spawn({
        let node_ep = node_ep.clone();
        let node_id = node_id.clone();
        async move { connect_node(&node_ep, hub_addr, node_id, String::new(), Some(tok)).await }
    });

    let incoming = tokio::time::timeout(Duration::from_secs(30), hub_ep.accept())
        .await
        .expect("node dialed")
        .expect("incoming");
    let conn = incoming.await.expect("conn established");
    let outcome = gate_connection(
        &conn,
        &mut allow,
        &mut tokens,
        2_000,
        &HubIdentity::new(hub_id),
        Some("test".into()),
        HELLO_DEADLINE,
    )
    .await;
    assert!(
        matches!(outcome, GateOutcome::Admitted { .. }),
        "gate admitted: {outcome:?}"
    );
    let peer = conn.remote_id().to_string();
    // Keep the node-side Connection alive for the test — dropping it would
    // close the QUIC connection and immediately mark the peer Disconnected.
    let (_node_conn, _hello) = dial_task
        .await
        .expect("dial join")
        .expect("connect_node ok");

    // Install Higgs + HubFleet and admit the paired connection.
    let bus = Arc::new(LogBus::new());
    let higgs = Arc::new(Higgs::with_log_bus(HiggsConfig::default(), bus.clone()));
    let fleet = Arc::new(HubFleet::new(bus));
    higgs.set_fleet(fleet.clone());
    fleet
        .add_node(
            peer.clone(),
            Arc::new(NodeTransport::new(conn)),
            None,
            None,
            None,
            false,
            None,
            true,
        )
        .await;

    // Give iroh a tick to select a path (Minimal preset with relay disabled
    // typically resolves Direct within the handshake, but this preserves the
    // test against tiny reorderings).
    tokio::time::sleep(Duration::from_millis(50)).await;

    let stats = higgs.network_stats(&peer).await.expect("fleet actor alive");
    assert_ne!(
        stats.state,
        LinkState::Disconnected,
        "admitted node must NOT read Disconnected"
    );
    assert!(
        matches!(stats.path, Some(LinkPath::Direct) | Some(LinkPath::Relay)),
        "admitted node must have a selected path, got {:?}",
        stats.path
    );
    let uptime = stats.uptime_ms.expect("uptime stamped on admit");
    assert!(uptime < 60_000, "uptime should be small right after admit");

    // Retire → Disconnected snapshot; no uptime, no path.
    fleet.retire(&peer).await;
    let stats = higgs.network_stats(&peer).await.expect("fleet actor alive");
    assert_eq!(stats.state, LinkState::Disconnected);
    assert_eq!(stats.uptime_ms, None);
    assert_eq!(stats.path, None);
}
