//! T10 fleet-event push, end-to-end against a REAL `higgs --node` child:
//! the node's HELLO advertises `fleet_events`, the hub admits it with the
//! capability, and worker-state changes (load, chat start/end) arrive as
//! `N_FLEET_EVENT` pushes the hub folds into its cache and re-broadcasts as
//! [`higgs::node::fleet::FleetEvent`]s via the `Higgs` facade subscription —
//! no chat-end debounced re-pull fires for the node (its snapshot_seq stays
//! exactly the ChatEnd push's).
//!
//! Skips when no tiny GGUF is available.

mod common;

use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use iroh_tickets::endpoint::EndpointTicket;

use higgs::auth::{Allowlist, PairingTokens};
use higgs::log_bus::LogBus;
use higgs::node::fleet::{FleetEvent, HubFleet};
use higgs::node::transport::NodeTransport;
use higgs::node::{gate_connection, GateOutcome, HubIdentity, HELLO_DEADLINE};
use higgs::remote::{FleetEventKind, ALPN};
use higgs::{Higgs, HiggsConfig};

use common::{stage_tiny_model, tiny_gguf_path, TINY_MODEL_ID};

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// A spawned `higgs --node` child, SIGTERM'd + reaped on drop (a graceful stop
/// flushes its llvm-cov profile; a hard kill drops it).
struct NodeProc(Child);
impl Drop for NodeProc {
    fn drop(&mut self) {
        // SAFETY: a plain kill(2) on our own child's pid.
        unsafe {
            libc::kill(self.0.id() as libc::pid_t, libc::SIGTERM);
        }
        let _ = self.0.wait();
    }
}

/// Await a specific event kind for `node`, tolerating interleaved other kinds.
async fn next_event(
    rx: &mut tokio::sync::broadcast::Receiver<FleetEvent>,
    node: &str,
    want: FleetEventKind,
) -> FleetEvent {
    loop {
        let ev = tokio::time::timeout(Duration::from_secs(30), rx.recv())
            .await
            .unwrap_or_else(|_| panic!("no {want:?} event within 30s"))
            .expect("fleet-event channel open");
        if ev.kind == want && ev.endpoint_id == node {
            return ev;
        }
    }
}

/// Real-node fleet-event push E2E; see the module docs.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn real_node_pushes_fleet_events_the_hub_rebroadcasts() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("SKIP real_node_pushes_fleet_events: no tiny GGUF (set HIGGS_TEST_GGUF)");
        return;
    };
    let scan_root = stage_tiny_model(&gguf);
    let node_home = tempfile::tempdir().expect("node home");

    // Hub: a local Higgs with a HubFleet installed, subscribed BEFORE the admit
    // so the NodeConnected marker is observed too.
    let bus = Arc::new(LogBus::new());
    let higgs = Arc::new(Higgs::with_log_bus(HiggsConfig::default(), bus.clone()));
    let fleet = Arc::new(HubFleet::new(bus));
    higgs.set_fleet(fleet.clone());
    let mut events = higgs
        .subscribe_fleet_events()
        .expect("fleet installed -> subscription available");

    // Hermetic hub iroh endpoint + a pairing token.
    let hub = iroh::Endpoint::builder(iroh::endpoint::presets::Minimal)
        .alpns(vec![ALPN.to_vec()])
        .relay_mode(iroh::RelayMode::Disabled)
        .bind()
        .await
        .expect("bind hub endpoint");
    let hub_id = hub.id().to_string();
    let ticket = EndpointTicket::new(hub.addr()).to_string();
    let mut allow = Allowlist::load(&node_home.path().join("hub-pairings.json")).unwrap();
    let mut tokens = PairingTokens::new();
    let token = tokens.mint(now_ms(), 600_000);

    // Spawn the real node process (hermetic iroh, real model dir).
    let child = Command::new(env!("CARGO_BIN_EXE_higgs"))
        .arg("--node")
        .arg(&ticket)
        .arg(&token)
        .env("HIGGS_HOME", node_home.path())
        .env("HIGGS_MODEL_DIR", scan_root.path())
        .env("HIGGS_IROH_LOCAL", "1")
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("spawn higgs --node");
    let _node = NodeProc(child);

    // Admit the node's dial through the REAL gate: its HELLO must advertise the
    // fleet_events capability, which flows into the fleet registration.
    let incoming = tokio::time::timeout(Duration::from_secs(30), hub.accept())
        .await
        .expect("node dialed within 30s")
        .expect("incoming");
    let conn = incoming.await.expect("connection");
    let peer = conn.remote_id().to_string();
    let outcome = gate_connection(
        &conn,
        &mut allow,
        &mut tokens,
        now_ms(),
        &HubIdentity::new(&hub_id),
        Some("test".into()),
        HELLO_DEADLINE,
    )
    .await;
    let GateOutcome::Admitted { fleet_events, .. } = outcome else {
        panic!("admitted: {outcome:?}");
    };
    assert!(
        fleet_events,
        "a current node's HELLO advertises fleet_events"
    );
    fleet
        .add_node(
            peer.clone(),
            Arc::new(NodeTransport::new(conn)),
            None,
            Some(2),
            None,
            fleet_events,
        )
        .await;
    next_event(&mut events, &peer, FleetEventKind::NodeConnected).await;

    // Load: the node pushes WorkerLoaded (in addition to the hub's own refresh).
    fleet
        .load(&peer, TINY_MODEL_ID, None)
        .await
        .expect("remote load");
    next_event(&mut events, &peer, FleetEventKind::WorkerLoaded).await;

    // Chat through the fleet: the node pushes ChatStart then ChatEnd.
    let messages = serde_json::json!([{ "role": "user", "content": "hi" }]).to_string();
    let (mut rx, fut) = fleet
        .chat(TINY_MODEL_ID, messages, 8, 0.7, None, None)
        .await
        .expect("chat routed");
    tokio::spawn(async move { while rx.recv().await.is_some() {} });
    fut.await.expect("chat final");
    next_event(&mut events, &peer, FleetEventKind::ChatStart).await;
    next_event(&mut events, &peer, FleetEventKind::ChatEnd).await;

    // The ChatEnd PUSH put the cache back to idle — and no debounced re-pull
    // follows for an event-pushing node (a pull would bump the node's
    // snapshot_seq; after the 250 ms settle window it must be unchanged).
    let seq_of = |views: Vec<higgs::node::fleet::NodeView>| {
        views
            .into_iter()
            .find(|n| n.endpoint_id == peer)
            .and_then(|n| n.inventory)
            .and_then(|i| i.snapshot_seq)
            .expect("cached inventory carries the pushed seq")
    };
    let view = fleet
        .nodes_view()
        .await
        .into_iter()
        .find(|n| n.endpoint_id == peer)
        .expect("node in view");
    let inv = view.inventory.clone().expect("cached inventory");
    assert_eq!(inv.workers.len(), 1, "one resident worker");
    assert_eq!(
        inv.workers[0].in_flight,
        Some(0),
        "chat-end push landed in the cache"
    );
    let seq_after_chat = seq_of(fleet.nodes_view().await);
    tokio::time::sleep(Duration::from_millis(700)).await;
    assert_eq!(
        seq_of(fleet.nodes_view().await),
        seq_after_chat,
        "no debounced re-pull for an event-pushing node"
    );

    // Retire: hub-local NodeDropped marker, clean teardown.
    fleet.retire(&peer).await;
    next_event(&mut events, &peer, FleetEventKind::NodeDropped).await;
}
