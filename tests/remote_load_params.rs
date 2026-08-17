//! T8 end-to-end: per-load params actually APPLY on a real `higgs --node`
//! child — the hub's params-load (protocol major 2) reaches the node's worker,
//! which reports the requested ctx_len through `M_NODE_STATUS`, not the
//! trained-context default it would use for a bare load.
//!
//! The wire pin (payload carries ctx_len; bare load sends only `{id}`) lives in
//! `cov_fleet2.rs` over a mock node; the version-gate arms in `fleet_tests`.
//! This test is the real claim: hub → iroh → node → spawned worker → llama.cpp
//! context sized as requested. ctx 256 differs from the bare-load default (the
//! trained-cap 2048 for the tiny model), so a reverted sender (bare `{id}`)
//! yields 2048 and the assert fails — verified by running exactly that revert.
//!
//! SKIPs without the tiny GGUF, like every fleet e2e.

mod common;

use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use higgs::auth::{Allowlist, PairingTokens};
use higgs::node::fleet::HubFleet;
use higgs::node::transport::NodeTransport;
use higgs::node::{gate_connection, GateOutcome, HubIdentity, HELLO_DEADLINE};
use higgs::remote::{ALPN, M_NODE_STATUS};
use iroh_tickets::endpoint::EndpointTicket;

use common::{stage_tiny_model, tiny_gguf_path, TINY_MODEL_ID};

struct NodeProc(Child);
impl Drop for NodeProc {
    fn drop(&mut self) {
        // SIGTERM: graceful shutdown flushes the child's llvm-cov profile.
        unsafe {
            libc::kill(self.0.id() as libc::pid_t, libc::SIGTERM);
        }
        let _ = self.0.wait();
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn params_load_applies_ctx_on_a_real_node() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("skipping remote_load_params: no tiny GGUF (set HIGGS_TEST_GGUF)");
        return;
    };

    // Manual hub: own endpoint + allowlist + one-time token, so the test HOLDS
    // the NodeTransport and can query M_NODE_STATUS directly after the load.
    let hub = iroh::Endpoint::builder(iroh::endpoint::presets::Minimal)
        .alpns(vec![ALPN.to_vec()])
        .relay_mode(iroh::RelayMode::Disabled)
        .bind()
        .await
        .expect("bind hub endpoint");
    let hub_id = hub.id().to_string();
    let allow_dir = tempfile::tempdir().unwrap();
    let mut allow = Allowlist::load(&allow_dir.path().join("pairings.json")).unwrap();
    let mut tokens = PairingTokens::new();
    let ticket = EndpointTicket::new(hub.addr()).to_string();
    let token = tokens.mint(now_ms(), 600_000);

    let node_scan = stage_tiny_model(&gguf);
    let node_home = tempfile::tempdir().unwrap();
    let _node = NodeProc(
        Command::new(env!("CARGO_BIN_EXE_higgs"))
            .arg("--node")
            .arg(&ticket)
            .arg(&token)
            .env("HIGGS_HOME", node_home.path())
            .env("HIGGS_MODEL_DIR", node_scan.path())
            .env("HIGGS_IROH_LOCAL", "1")
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("spawn higgs --node"),
    );

    // Admit, CAPTURING the genuinely-negotiated version (a current child speaks 2).
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
    let GateOutcome::Admitted {
        agreed_version,
        software_version,
        fleet_events: _,
        update_failed: _,
        reports_update_failures: _,
        target: _,
        variant: _,
        update_capable: _,
        version_capable: _,
        log_capable: _,
        pull_capable: _,
        downloads: _,
    } = outcome
    else {
        panic!("admitted: {outcome:?}");
    };
    assert_eq!(agreed_version, 2, "a current node negotiates major 2");
    // T14: the gate carries the node's self-reported build out of the HELLO —
    // this spawned node IS this crate, so the semver must match exactly.
    assert_eq!(
        software_version,
        env!("CARGO_PKG_VERSION"),
        "the HELLO semver reaches the gate outcome"
    );

    let transport = Arc::new(NodeTransport::new(conn));
    let fleet = Arc::new(HubFleet::new(Arc::new(higgs::log_bus::LogBus::new())));
    fleet
        .add_node(
            peer.clone(),
            transport.clone(),
            None,
            Some(agreed_version),
            Some(software_version.clone()),
            false,
            None,
            true,
        )
        .await;

    // Params-load: ctx 256 — the bare-load default (trained-cap) is 2048 for
    // the tiny model, so a bare-payload revert reads 2048, not 256.
    let params = higgs::remote::NodeLoadParams {
        id: TINY_MODEL_ID.into(),
        ctx_len: Some(256),
        gpu_layers: None,
        threads: None,
        params: None,
    };
    let worker = fleet
        .load(&peer, TINY_MODEL_ID, Some(params))
        .await
        .expect("params-load against the real node");

    // The node's own status for that worker reports the APPLIED context.
    let status = transport
        .request(M_NODE_STATUS, serde_json::json!({ "worker_id": worker.0 }))
        .await
        .expect("node status");
    // NODE-side zero backstop, pinned DIRECTLY (the hub normalizes zeros away
    // before the wire — that arm is pinned in cov_fleet2 — so only a raw
    // request can reach the node's own filter): ctx 0 reads as ABSENT →
    // trained-cap default (2048 for the tiny model), never the worker's old
    // hardcoded 0→4096 coercion. Fail-on-revert: drop do_load's
    // `.filter(|c| *c > 0)` and this reads 4096.
    let raw = transport
        .request(
            higgs::remote::M_NODE_LOAD,
            serde_json::json!({ "id": TINY_MODEL_ID, "ctx_len": 0 }),
        )
        .await
        .expect("raw ctx-0 load");
    let w0 = raw
        .get("worker_id")
        .and_then(serde_json::Value::as_u64)
        .expect("worker id");
    let status0 = transport
        .request(M_NODE_STATUS, serde_json::json!({ "worker_id": w0 }))
        .await
        .expect("node status for the ctx-0 worker");
    let ctx0 = status0
        .pointer("/loaded/ctx_len/n")
        .and_then(serde_json::Value::as_u64);
    // Discriminate against the two WRONG outcomes without hardcoding the tiny
    // model's trained context: the old worker coercion read 4096; a leaked 0
    // would read 0. Scope caveat: a HIGGS_TEST_GGUF trained at exactly 4096
    // (correct outcome == coercion outcome) or with an unreadable trained ctx
    // (worker's own 4096 default is then CORRECT) would fail here spuriously —
    // the assert assumes a tiny model with a readable trained ctx ≠ 4096, as
    // the default stories260K (2048) is.
    assert!(
        ctx0.is_some() && ctx0 != Some(4096) && ctx0 != Some(0),
        "ctx 0 → the model's trained-cap default, never the 4096 coercion: {status0}"
    );

    // T14: the view row carries the node's HELLO-reported build (this spawned
    // node IS this crate), the negotiated major, and the AGE of the inventory
    // snapshot the hub is serving (stamped at commit — the T9 freshness
    // residual's anchor). Fail-on-revert: drop the nodes_view fill for any of
    // the three and its assert reads None.
    {
        let view = fleet.nodes_view().await;
        let row = view
            .iter()
            .find(|n| n.endpoint_id == peer)
            .expect("admitted node in the view");
        assert_eq!(
            row.software_version.as_deref(),
            Some(env!("CARGO_PKG_VERSION")),
            "view carries the node's HELLO semver: {row:?}"
        );
        assert_eq!(row.protocol, Some(2), "view carries the negotiated major");
        let age = row.inventory_age_ms.expect("inventory age stamped");
        assert!(
            age < 60_000,
            "the just-refreshed snapshot reads fresh, got {age} ms"
        );
    }

    // T9: the hub's own INVENTORY (refreshed by fleet.load) now carries the
    // per-worker stats — the ctx that applied, when it loaded, its idle clock
    // and in-flight count — with no per-worker RPC. Fail-on-revert: drop the
    // node's load_facts cache (or the snapshot fill) and ctx_len reads None.
    let view = fleet.nodes_view().await;
    let inv = view
        .iter()
        .find(|n| n.endpoint_id == peer)
        .and_then(|n| n.inventory.as_ref())
        .expect("inventory cached after the load");
    let row = inv
        .workers
        .iter()
        .find(|w| w.worker_id == worker.0)
        .expect("the params-loaded worker is in the inventory");
    assert_eq!(
        row.ctx_len,
        Some(256),
        "inventory carries the APPLIED ctx: {row:?}"
    );
    assert!(row.loaded_at_ms.is_some(), "loaded-at stamped: {row:?}");
    assert!(row.idle_ms.is_some(), "idle clock present: {row:?}");
    assert_eq!(row.in_flight, Some(0), "no chats in flight: {row:?}");

    // ctx_len is the typed CtxLen on the wire: {"kind":"fixed","n":256}.
    let ctx = status
        .pointer("/loaded/ctx_len/n")
        .and_then(serde_json::Value::as_u64);
    assert_eq!(
        ctx,
        Some(256),
        "the wire ctx_len reached llama.cpp (the bare-load default would be the \
         trained-cap 2048): {status}"
    );
}
