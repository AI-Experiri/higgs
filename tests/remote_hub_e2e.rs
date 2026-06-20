//! Hub-seam end-to-end: prove the hub's `/v1` chat path (`Higgs::chat_stream`) routes a
//! remote-resident model through the `HubFleet` to a REAL spawned `higgs --node` process.
//!
//! Flow: build a hub `Higgs` with a `HubFleet` installed → spawn a real node → pair over
//! hermetic iroh → `fleet.load` a real GGUF on the node → `higgs.chat_stream(model)` routes
//! remotely and streams real tokens back. This exercises the full hub seam (chat_stream →
//! fleet routing → NodeTransport → node relay → worker → model), nothing faked.
//!
//! Skips when no tiny GGUF is available.

mod common;

use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use iroh_tickets::endpoint::EndpointTicket;

use higgs::auth::{Allowlist, PairingTokens};
use higgs::node::fleet::HubFleet;
use higgs::node::transport::NodeTransport;
use higgs::node::{gate_connection, GateOutcome, HELLO_DEADLINE};
use higgs::remote::ALPN;
use higgs::{Higgs, HiggsConfig};

use common::{stage_tiny_model, tiny_gguf_path, TINY_MODEL_ID};

struct NodeProc(Child);
impl Drop for NodeProc {
    fn drop(&mut self) {
        // SIGTERM: graceful node shutdown flushes its llvm-cov profile (a hard kill drops it).
        unsafe {
            libc::kill(self.0.id() as libc::pid_t, libc::SIGTERM);
        }
        let _ = self.0.wait();
    }
}

fn now_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis() as u64
}

async fn hub_endpoint() -> iroh::Endpoint {
    iroh::Endpoint::builder(iroh::endpoint::presets::Minimal)
        .alpns(vec![ALPN.to_vec()])
        .relay_mode(iroh::RelayMode::Disabled)
        .bind()
        .await
        .expect("bind hub endpoint")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn hub_v1_chat_routes_to_remote_node() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("skipping remote_hub_e2e: no tiny GGUF (set HIGGS_TEST_GGUF)");
        return;
    };
    let scan_root = stage_tiny_model(&gguf);
    let node_home = tempfile::tempdir().expect("node home");

    // Hub: a local Higgs (no local models) with a HubFleet installed.
    let higgs = Arc::new(Higgs::new(HiggsConfig::default()));
    let fleet = Arc::new(HubFleet::new());
    higgs.set_fleet(fleet.clone());

    // Hub iroh endpoint + a pairing ticket/token.
    let hub = hub_endpoint().await;
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

    // Accept the node's dial, gate it, and register it in the fleet.
    let incoming = tokio::time::timeout(Duration::from_secs(30), hub.accept())
        .await
        .expect("node dialed within 30s")
        .expect("incoming");
    let conn = incoming.await.expect("connection");
    let peer = conn.remote_id().to_string();
    let outcome =
        gate_connection(&conn, &mut allow, &mut tokens, now_ms(), hub_id, Some("test".into()), HELLO_DEADLINE)
            .await;
    assert!(matches!(outcome, GateOutcome::Admitted { .. }), "node admitted: {outcome:?}");
    fleet.add_node(peer.clone(), Arc::new(NodeTransport::new(conn)));

    // Load a real model on the node via the fleet → records the route.
    fleet.load(&peer, TINY_MODEL_ID).await.expect("remote load");
    assert!(fleet.is_remote(TINY_MODEL_ID), "model is now remote-routable");

    // The hub's /v1 chat path: chat_stream routes the remote-resident model through the
    // fleet to the node, streaming real tokens back.
    let (mut deltas, handle) = higgs
        .chat_stream(
            TINY_MODEL_ID.to_string(),
            "[{\"role\":\"user\",\"content\":\"Once upon a time\"}]".to_string(),
            8,
            0.0,
            None,
        )
        .await
        .expect("chat_stream routed remotely");

    let collector = tokio::spawn(async move {
        let mut n = 0usize;
        while deltas.recv().await.is_some() {
            n += 1;
        }
        n
    });
    let outcome = handle.await.expect("join").expect("chat outcome");
    let chunk_count = collector.await.unwrap();

    assert!(
        chunk_count > 0 || !outcome.content.is_empty(),
        "remote chat generated tokens (chunks={chunk_count}, content={:?})",
        outcome.content
    );
    assert!(!outcome.finish_reason.is_empty(), "finish_reason set");
}
