//! In-process production hub: build a hub `Higgs` (via the `higgs_local` harness) with the hub
//! network enabled, mint a pairing credential over the `Higgs::pair` facade, dial it with a real
//! `higgs --node` child process, and confirm the node shows up connected in `Higgs::nodes()`.
//! When a tiny GGUF is available (the default) it ALSO drives the remote node catalog/load/unload
//! facade ops and confirms the model becomes remotely routable in the real `/v1/models` surface.
//!
//! higgs is now library-first: the `/api/higgs/*` HTTP control surface is gone, so the hub is an
//! in-process `Higgs` (hub network via `hub_enable`) and control flows through typed facade
//! methods. Nodes are still real spawned `higgs --node` processes over hermetic iroh
//! (`HIGGS_IROH_LOCAL=1`), exactly as an operator's node would connect.

mod common;

use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::time::Duration;

use common::{higgs_local, serve_v1_local, stage_tiny_model, tiny_gguf_path, TINY_MODEL_ID};
use higgs::Higgs;

/// A spawned `higgs --node` child. SIGTERM (graceful, flushes llvm-cov) + reap on drop.
struct NodeProc(Child);
impl Drop for NodeProc {
    fn drop(&mut self) {
        // SAFETY: a plain kill(2) on our own child's pid.
        unsafe { libc::kill(self.0.id() as libc::pid_t, libc::SIGTERM) };
        let _ = self.0.wait();
    }
}

/// Spawn a real `higgs --node <ticket> <token>` daemon dialing the in-process hub over hermetic
/// iroh, with its own isolated `HIGGS_HOME`. When `model_dir` is set the node scans/serves that
/// staged tiny model.
#[allow(clippy::zombie_processes)] // reaped by NodeProc::drop (SIGTERM + wait)
fn spawn_node(ticket: &str, token: &str, home: &Path, model_dir: Option<&Path>) -> NodeProc {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_higgs"));
    cmd.arg("--node")
        .arg(ticket)
        .arg(token)
        .env("HIGGS_HOME", home)
        .env("HIGGS_IROH_LOCAL", "1")
        .stdout(Stdio::null())
        .stderr(Stdio::inherit());
    if let Some(dir) = model_dir {
        cmd.env("HIGGS_MODEL_DIR", dir);
    }
    NodeProc(cmd.spawn().expect("spawn higgs --node"))
}

/// Poll the fleet view until a REMOTE node (not the local machine) shows connected, returning its
/// stable `EndpointId`. `None` if none connects within ~30s.
async fn wait_remote_connected(higgs: &Higgs) -> Option<String> {
    for _ in 0..150 {
        if let Some(n) = higgs
            .nodes()
            .await
            .iter()
            .find(|n| n.connected && !n.is_local)
        {
            return Some(n.endpoint_id.clone());
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    None
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn hub_server_pairs_a_node_and_lists_it() {
    // The hub itself carries NO local models (empty staged set) so a model showing in /v1/models
    // proves REMOTE routability, not a local JIT-servable id.
    let Some(higgs) = higgs_local(&[]).await else {
        return;
    };
    higgs.hub_enable().await.expect("enable hub network");

    // Mint a pairing credential over the facade.
    let pair = higgs.pair().await.expect("mint pairing");
    assert!(
        !pair.ticket.is_empty() && pair.token.starts_with("htk_"),
        "got a ticket + token: {pair:?}"
    );

    // Stage the tiny model into a scan root for the node so it can serve a real model.
    let gguf = tiny_gguf_path().expect("tiny gguf present (higgs_local returned Some)");
    let staged = stage_tiny_model(&gguf);
    let node_home = tempfile::tempdir().unwrap();
    let _node = spawn_node(
        &pair.ticket,
        &pair.token,
        node_home.path(),
        Some(staged.path()),
    );

    // The node pairs and shows connected.
    let node_id = wait_remote_connected(&higgs)
        .await
        .expect("node paired and shows connected in nodes()");

    // ── Hub-mode status reflects the live hub + the admitted node (hub_enable is idempotent and
    // returns the current status). ──
    let hub_status = higgs.hub_enable().await.expect("hub status");
    assert!(hub_status.enabled, "hub mode on: {hub_status:?}");
    assert!(
        hub_status.hub_id.as_deref().is_some_and(|s| !s.is_empty()),
        "hub status carries the stable hub id: {hub_status:?}"
    );
    assert!(
        hub_status.node_count >= 1,
        "hub status counts the admitted node: {hub_status:?}"
    );

    // ── The node's on-disk catalog (M_NODE_SCAN over iroh): a `models` array listing the staged
    // model. ──
    let catalog = higgs.node_scan(&node_id).await.expect("node catalog ok");
    let models = catalog["models"].as_array().expect("models array");
    assert!(
        models.iter().any(|m| m["id"] == TINY_MODEL_ID),
        "remote node catalog lists the staged model: {catalog}"
    );

    // ── Remote-load: load the model on the paired node, then confirm it's remotely routable in
    // the real /v1/models surface. ──
    higgs
        .node_load(&node_id, TINY_MODEL_ID)
        .await
        .expect("remote load ok");

    let (base, guard) = serve_v1_local(higgs.handle()).await;
    let client = reqwest::Client::new();
    let listed: serde_json::Value = client
        .get(format!("{base}/v1/models"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(
        listed["data"]
            .as_array()
            .is_some_and(|d| d.iter().any(|m| m["id"] == TINY_MODEL_ID)),
        "remote model routable in /v1/models: {listed}"
    );
    guard.shutdown().await;

    // Unload it again.
    higgs
        .node_unload(TINY_MODEL_ID)
        .await
        .expect("remote unload ok");

    // ── Retire the node — it must disappear from the fleet entirely (not linger disconnected). ──
    higgs.node_retire(&node_id).await.expect("retire ok");
    let mut gone = false;
    for _ in 0..50 {
        if !higgs.nodes().await.iter().any(|n| n.endpoint_id == node_id) {
            gone = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert!(gone, "retired node is removed from the fleet");

    // The hub is still enabled after the retire, but no longer counts the node.
    let hub_after = higgs.hub_enable().await.expect("hub status after retire");
    assert!(hub_after.enabled, "hub still enabled: {hub_after:?}");
    assert_eq!(
        hub_after.node_count, 0,
        "retired node no longer counted: {hub_after:?}"
    );

    higgs.shutdown().await;
}

/// Editable labels: `Higgs::node_label` renames the local instance (config.json name) and a paired
/// remote node (allowlist label); both surface in `Higgs::nodes()`. No GGUF needed.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn hub_server_relabel_local_and_remote() {
    let Some(higgs) = higgs_local(&[]).await else {
        return;
    };
    higgs.hub_enable().await.expect("enable hub network");

    // Pair a node so there's a remote node to rename.
    let pair = higgs.pair().await.expect("mint pairing");
    let node_home = tempfile::tempdir().unwrap();
    let _node = spawn_node(&pair.ticket, &pair.token, node_home.path(), None);
    let node_id = wait_remote_connected(&higgs)
        .await
        .expect("node paired before relabel");

    // Rename the LOCAL instance and the REMOTE node.
    for (node, label) in [
        ("local", "studio-hub"),
        (node_id.as_str(), "renamed-remote"),
    ] {
        let renamed = higgs.node_label(node, label).await.expect("relabel ok");
        assert!(renamed, "relabel {node} found the target");
    }

    // Both new labels surface in the unified node view.
    let nodes = higgs.nodes().await;
    let local = nodes.iter().find(|n| n.is_local).expect("local node");
    assert_eq!(local.label, "studio-hub", "local renamed: {nodes:?}");
    let remote = nodes
        .iter()
        .find(|n| n.endpoint_id == node_id)
        .expect("remote node");
    assert_eq!(remote.label, "renamed-remote", "remote renamed: {nodes:?}");

    // An unknown remote id is a no-op miss (the old 404), never a silent insert.
    let miss = higgs
        .node_label("not-a-node", "x")
        .await
        .expect("label call ok");
    assert!(!miss, "unknown node → not found (Ok(false))");

    higgs.shutdown().await;
}

/// Node self-retire: a paired node runs `higgs node leave`, which dials the saved hub and asks it
/// to retire ITSELF. The node then disappears from `Higgs::nodes()` and forgets the hub locally.
/// No GGUF needed (pairing + leave path only).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn hub_server_node_self_leave() {
    let Some(higgs) = higgs_local(&[]).await else {
        return;
    };
    higgs.hub_enable().await.expect("enable hub network");

    // Pair a node daemon with a freshly-minted token.
    let pair = higgs.pair().await.expect("mint pairing");
    let node_home = tempfile::tempdir().unwrap();
    let node = spawn_node(&pair.ticket, &pair.token, node_home.path(), None);
    let node_id = wait_remote_connected(&higgs)
        .await
        .expect("node paired before leave");

    // Stop the daemon so it can't reconnect after it leaves, then run `higgs node leave` (which
    // makes its OWN connection, same HIGGS_HOME → same EndpointId, and asks the hub to retire it).
    drop(node);
    tokio::time::sleep(Duration::from_millis(500)).await;
    let leave = tokio::process::Command::new(env!("CARGO_BIN_EXE_higgs"))
        .args(["node", "leave"])
        .env("HIGGS_HOME", node_home.path())
        .env("HIGGS_IROH_LOCAL", "1")
        .output()
        .await
        .expect("run node leave");
    assert!(
        leave.status.success(),
        "node leave exits 0: {}",
        String::from_utf8_lossy(&leave.stderr)
    );

    // The node retired ITSELF — gone from nodes() entirely (not lingering disconnected).
    let mut gone = false;
    for _ in 0..50 {
        if !higgs.nodes().await.iter().any(|n| n.endpoint_id == node_id) {
            gone = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert!(gone, "node removed itself from the fleet via `node leave`");

    // And it forgot the hub locally, so a bare `higgs --node` no longer dials it.
    let cfg = higgs::config::InstanceConfig::load(&node_home.path().join("config.json")).unwrap();
    assert!(
        cfg.default_saved_hub().is_none(),
        "node forgot its saved hub after leaving"
    );

    higgs.shutdown().await;
}

/// The hub kill switch: `Higgs::hub_disable` stops ALL node network activity (closes the endpoint →
/// no inbound dials, no relay; closes node transports → nodes disconnected) while KEEPING the
/// fleet's node list/routes; `Higgs::hub_enable` turns it back on. Asserts the deterministic,
/// hermetic behavior: enabled→disabled→enabled status, pairing gated by the switch, and a paired
/// node going disconnected-but-still-listed on disable. (Live reconnect-with-route-survival is
/// covered by `remote_hub_e2e::node_reconnects_and_route_survives`, since a LAN-only rebind can
/// land on a new UDP port the node's cached ticket can't redial.)
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn hub_kill_switch_disables_then_reenables_the_network() {
    let Some(higgs) = higgs_local(&[]).await else {
        return;
    };
    // Boots enabled.
    let st = higgs.hub_enable().await.expect("enable hub network");
    assert!(st.enabled, "boots enabled: {st:?}");

    // Pair + dial a node; wait until it shows connected.
    let pair = higgs.pair().await.expect("mint pairing");
    let node_home = tempfile::tempdir().unwrap();
    let _node = spawn_node(&pair.ticket, &pair.token, node_home.path(), None);
    let node_id = wait_remote_connected(&higgs)
        .await
        .expect("node connected before disable");

    // ── DISABLE — network off. ──
    let dis = higgs.hub_disable().await;
    assert!(!dis.enabled, "disabled status: {dis:?}");

    // Pairing is gated off while disabled (the old 409).
    assert!(
        higgs.pair().await.is_err(),
        "pair refused while disabled (server is not a hub)"
    );

    // The node's transport is closed: it goes disconnected, but STAYS listed (route/seed kept, not
    // retired) — the hub isn't accepting, so it can't reconnect during the disabled window.
    let mut disconnected_listed = false;
    for _ in 0..50 {
        if higgs
            .nodes()
            .await
            .iter()
            .any(|n| n.endpoint_id == node_id && !n.connected)
        {
            disconnected_listed = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert!(
        disconnected_listed,
        "node disconnected but still listed (route kept) while disabled"
    );

    // ── ENABLE — network back on. ──
    let en = higgs.hub_enable().await.expect("re-enable hub network");
    assert!(en.enabled, "re-enabled status: {en:?}");
    assert!(
        en.hub_id.as_deref().is_some_and(|s| !s.is_empty()),
        "hub id present after re-enable: {en:?}"
    );

    // Pairing works again, and the re-armed fleet ADMITS a freshly-paired node — proving enable
    // re-armed admission (disable disarms it via disconnect_all) and the new endpoint accepts.
    let pair2 = higgs.pair().await.expect("mint pairing after re-enable");
    let node2_home = tempfile::tempdir().unwrap();
    let _node2 = spawn_node(&pair2.ticket, &pair2.token, node2_home.path(), None);
    let mut node2_connected = false;
    for _ in 0..150 {
        if higgs
            .nodes()
            .await
            .iter()
            .any(|n| n.connected && !n.is_local && n.endpoint_id != node_id)
        {
            node2_connected = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    assert!(
        node2_connected,
        "re-enabled hub admits a freshly-paired node (fleet re-armed)"
    );

    higgs.shutdown().await;
}
