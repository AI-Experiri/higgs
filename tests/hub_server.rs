//! Black-box P3 production hub: spawn a real `higgs` server in HUB mode, mint a pairing token
//! over `POST /api/higgs/pair`, dial it with a real `higgs --node`, and confirm the node shows
//! up connected in `GET /api/higgs/nodes`. The pairing path needs no GGUF and always runs;
//! when a tiny GGUF is available it ALSO drives `POST /api/higgs/nodes/load` and confirms the
//! model becomes remotely routable in `GET /v1/models`.

mod common;

use std::process::{Child, Command, Stdio};
use std::time::Duration;

use common::{stage_tiny_model, tiny_gguf_path, TINY_MODEL_ID};

struct Proc(Child);
impl Drop for Proc {
    fn drop(&mut self) {
        unsafe { libc::kill(self.0.id() as libc::pid_t, libc::SIGTERM) };
        let _ = self.0.wait();
    }
}

fn free_port() -> u16 {
    let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    l.local_addr().unwrap().port()
}

#[tokio::test]
async fn hub_server_pairs_a_node_and_lists_it() {
    let hub_home = tempfile::tempdir().unwrap();
    let node_home = tempfile::tempdir().unwrap();
    let port = free_port();

    // Spawn the hub server (HUB mode, hermetic iroh).
    let hub = Command::new(env!("CARGO_BIN_EXE_higgs"))
        .env("HIGGS_BIND", "127.0.0.1")
        .env("HIGGS_PORT", port.to_string())
        .env("HIGGS_HOME", hub_home.path())
        .env("HIGGS_HUB", "1")
        .env("HIGGS_IROH_LOCAL", "1")
        .env("RUST_LOG", "warn")
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("spawn hub");
    let _hub = Proc(hub);
    let base = format!("http://127.0.0.1:{port}");
    let c = reqwest::Client::new();

    // Wait for the server.
    let mut ready = false;
    for _ in 0..150 {
        if let Ok(r) = c.get(format!("{base}/health")).send().await {
            if r.status().is_success() {
                ready = true;
                break;
            }
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    assert!(ready, "hub server ready");

    // Mint a pairing token + ticket over the API.
    let pair: serde_json::Value = c
        .post(format!("{base}/api/higgs/pair"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let ticket = pair["ticket"].as_str().expect("ticket").to_string();
    let token = pair["token"].as_str().expect("token").to_string();
    assert!(
        !ticket.is_empty() && token.starts_with("htk_"),
        "got a ticket + token"
    );

    // If a tiny GGUF is available, stage it so the node can actually load a model via the
    // HTTP hub API below; otherwise the pairing-only path still runs.
    let staged = tiny_gguf_path().map(|g| stage_tiny_model(&g));

    // Spawn the node, dialing the hub with the minted token.
    let mut node_cmd = Command::new(env!("CARGO_BIN_EXE_higgs"));
    node_cmd
        .arg("--node")
        .arg(&ticket)
        .arg(&token)
        .env("HIGGS_HOME", node_home.path())
        .env("HIGGS_IROH_LOCAL", "1")
        .stdout(Stdio::null())
        .stderr(Stdio::inherit());
    if let Some(dir) = &staged {
        node_cmd.env("HIGGS_MODEL_DIR", dir.path());
    }
    let _node = Proc(node_cmd.spawn().expect("spawn node"));

    // Poll /api/higgs/nodes until the node appears connected.
    let mut connected = false;
    for _ in 0..150 {
        let nodes: serde_json::Value = c
            .get(format!("{base}/api/higgs/nodes"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        // The local node is always present + connected; wait for the REMOTE node specifically.
        if nodes.as_array().is_some_and(|a| {
            a.iter()
                .any(|n| n["connected"] == true && n["is_local"] != true)
        }) {
            connected = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    assert!(
        connected,
        "node paired and shows connected in /api/higgs/nodes"
    );

    // The paired node's stable EndpointId — used for the catalog + retire routes below.
    let nodes: serde_json::Value = c
        .get(format!("{base}/api/higgs/nodes"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let node_id = nodes
        .as_array()
        .and_then(|a| a.iter().find(|n| n["is_local"] != true))
        .and_then(|n| n["endpoint_id"].as_str())
        .expect("a remote node in /api/higgs/nodes")
        .to_string();

    // ── GET /api/higgs/hub — hub-mode status reflects the live hub + the admitted node. ──
    let hub_status: serde_json::Value = c
        .get(format!("{base}/api/higgs/hub"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(hub_status["enabled"], true, "hub mode on: {hub_status}");
    assert!(
        hub_status["hub_id"].as_str().is_some_and(|s| !s.is_empty()),
        "hub status carries the stable hub id: {hub_status}"
    );
    assert!(
        hub_status["node_count"].as_u64().is_some_and(|n| n >= 1),
        "hub status counts the admitted node: {hub_status}"
    );

    // ── GET /api/higgs/nodes/{node}/models — the node's on-disk catalog (M_NODE_SCAN over
    // iroh). Always answers 200 with a `models` array; with a staged GGUF it lists the model.
    let cat = c
        .get(format!("{base}/api/higgs/nodes/{node_id}/models"))
        .send()
        .await
        .unwrap();
    assert!(
        cat.status().is_success(),
        "node catalog ok: {}",
        cat.status()
    );
    let catalog: serde_json::Value = cat.json().await.unwrap();
    let models = catalog["models"].as_array().expect("models array");
    if staged.is_some() {
        assert!(
            models.iter().any(|m| m["id"] == TINY_MODEL_ID),
            "remote node catalog lists the staged model: {catalog}"
        );
    }

    // ── HTTP remote-load (only with a GGUF): load a model on the paired node via the new
    // POST /api/higgs/nodes/load, then confirm it's remotely routable in /v1/models. ──
    if staged.is_some() {
        let load = c
            .post(format!("{base}/api/higgs/nodes/load"))
            .json(&serde_json::json!({ "node": node_id, "model": TINY_MODEL_ID }))
            .send()
            .await
            .unwrap();
        assert!(
            load.status().is_success(),
            "remote load over HTTP ok: {}",
            load.status()
        );

        // The remote-resident model is now advertised in /v1/models.
        let models: serde_json::Value = c
            .get(format!("{base}/v1/models"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert!(
            models["data"]
                .as_array()
                .is_some_and(|d| d.iter().any(|m| m["id"] == TINY_MODEL_ID)),
            "remote model routable in /v1/models: {models}"
        );

        // Unload it again via HTTP.
        let unload = c
            .post(format!("{base}/api/higgs/nodes/unload"))
            .json(&serde_json::json!({ "model": TINY_MODEL_ID }))
            .send()
            .await
            .unwrap();
        assert!(
            unload.status().is_success(),
            "remote unload over HTTP ok: {}",
            unload.status()
        );
    }

    // ── POST /api/higgs/nodes/retire — remove the node from allowlist + fleet. It must
    // disappear from /api/higgs/nodes entirely (not linger as disconnected). ──
    let retire = c
        .post(format!("{base}/api/higgs/nodes/retire"))
        .json(&serde_json::json!({ "node": node_id }))
        .send()
        .await
        .unwrap();
    assert!(
        retire.status().is_success(),
        "retire over HTTP ok: {}",
        retire.status()
    );

    let mut gone = false;
    for _ in 0..50 {
        let nodes: serde_json::Value = c
            .get(format!("{base}/api/higgs/nodes"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        if nodes
            .as_array()
            .is_some_and(|a| !a.iter().any(|n| n["endpoint_id"] == node_id))
        {
            gone = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert!(gone, "retired node is removed from /api/higgs/nodes");

    // The hub is still enabled after the retire, but no longer counts the node.
    let hub_after: serde_json::Value = c
        .get(format!("{base}/api/higgs/hub"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(hub_after["enabled"], true, "hub still enabled: {hub_after}");
    assert_eq!(
        hub_after["node_count"], 0,
        "retired node no longer counted: {hub_after}"
    );
}

/// Editable labels: `POST /api/higgs/nodes/label` renames the local instance (config.json name)
/// and a paired remote node (allowlist label); both surface in `GET /api/higgs/nodes`. No GGUF.
#[tokio::test]
async fn hub_server_relabel_local_and_remote() {
    let hub_home = tempfile::tempdir().unwrap();
    let node_home = tempfile::tempdir().unwrap();
    let port = free_port();

    let _hub = Proc(
        Command::new(env!("CARGO_BIN_EXE_higgs"))
            .env("HIGGS_BIND", "127.0.0.1")
            .env("HIGGS_PORT", port.to_string())
            .env("HIGGS_HOME", hub_home.path())
            .env("HIGGS_HUB", "1")
            .env("HIGGS_IROH_LOCAL", "1")
            .env("RUST_LOG", "warn")
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("spawn hub"),
    );
    let base = format!("http://127.0.0.1:{port}");
    let c = reqwest::Client::new();

    let mut ready = false;
    for _ in 0..150 {
        if let Ok(r) = c.get(format!("{base}/health")).send().await {
            if r.status().is_success() {
                ready = true;
                break;
            }
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    assert!(ready, "hub server ready");

    // Pair a node so there's a remote node to rename.
    let pair: serde_json::Value = c
        .post(format!("{base}/api/higgs/pair"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let ticket = pair["ticket"].as_str().unwrap().to_string();
    let token = pair["token"].as_str().unwrap().to_string();
    let _node = Proc(
        Command::new(env!("CARGO_BIN_EXE_higgs"))
            .arg("--node")
            .arg(&ticket)
            .arg(&token)
            .env("HIGGS_HOME", node_home.path())
            .env("HIGGS_IROH_LOCAL", "1")
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("spawn node"),
    );
    let mut node_id = String::new();
    for _ in 0..150 {
        let nodes: serde_json::Value = c
            .get(format!("{base}/api/higgs/nodes"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        if let Some(n) = nodes.as_array().and_then(|a| {
            a.iter()
                .find(|n| n["connected"] == true && n["is_local"] != true)
        }) {
            node_id = n["endpoint_id"].as_str().unwrap().to_string();
            break;
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    assert!(!node_id.is_empty(), "node paired before relabel");

    // Rename the LOCAL instance and the REMOTE node.
    for (node, label) in [
        ("local", "studio-hub"),
        (node_id.as_str(), "renamed-remote"),
    ] {
        let resp = c
            .post(format!("{base}/api/higgs/nodes/label"))
            .json(&serde_json::json!({ "node": node, "label": label }))
            .send()
            .await
            .unwrap();
        assert!(
            resp.status().is_success(),
            "relabel {node} ok: {}",
            resp.status()
        );
    }

    // Both new labels surface in the unified node view.
    let nodes: serde_json::Value = c
        .get(format!("{base}/api/higgs/nodes"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let arr = nodes.as_array().expect("nodes array");
    let local = arr
        .iter()
        .find(|n| n["is_local"] == true)
        .expect("local node");
    assert_eq!(local["label"], "studio-hub", "local renamed: {nodes}");
    let remote = arr
        .iter()
        .find(|n| n["endpoint_id"] == node_id.as_str())
        .expect("remote node");
    assert_eq!(remote["label"], "renamed-remote", "remote renamed: {nodes}");

    // An unknown remote id is a 404 (never a silent insert).
    let miss = c
        .post(format!("{base}/api/higgs/nodes/label"))
        .json(&serde_json::json!({ "node": "not-a-node", "label": "x" }))
        .send()
        .await
        .unwrap();
    assert_eq!(miss.status().as_u16(), 404, "unknown node → 404");
}

/// Node self-retire: a paired node runs `higgs node leave`, which dials the saved hub and asks it
/// to retire ITSELF. The node then disappears from `/api/higgs/nodes` and forgets the hub locally.
/// No GGUF needed (pairing + leave path only).
#[tokio::test]
async fn hub_server_node_self_leave() {
    let hub_home = tempfile::tempdir().unwrap();
    let node_home = tempfile::tempdir().unwrap();
    let port = free_port();

    let _hub = Proc(
        Command::new(env!("CARGO_BIN_EXE_higgs"))
            .env("HIGGS_BIND", "127.0.0.1")
            .env("HIGGS_PORT", port.to_string())
            .env("HIGGS_HOME", hub_home.path())
            .env("HIGGS_HUB", "1")
            .env("HIGGS_IROH_LOCAL", "1")
            .env("RUST_LOG", "warn")
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("spawn hub"),
    );
    let base = format!("http://127.0.0.1:{port}");
    let c = reqwest::Client::new();

    let mut ready = false;
    for _ in 0..150 {
        if let Ok(r) = c.get(format!("{base}/health")).send().await {
            if r.status().is_success() {
                ready = true;
                break;
            }
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    assert!(ready, "hub server ready");

    // Pair a node daemon with a freshly-minted token.
    let pair: serde_json::Value = c
        .post(format!("{base}/api/higgs/pair"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let ticket = pair["ticket"].as_str().expect("ticket").to_string();
    let token = pair["token"].as_str().expect("token").to_string();
    let node = Proc(
        Command::new(env!("CARGO_BIN_EXE_higgs"))
            .arg("--node")
            .arg(&ticket)
            .arg(&token)
            .env("HIGGS_HOME", node_home.path())
            .env("HIGGS_IROH_LOCAL", "1")
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("spawn node"),
    );

    // Wait for the REMOTE node to connect, and capture its EndpointId.
    let mut node_id = String::new();
    for _ in 0..150 {
        let nodes: serde_json::Value = c
            .get(format!("{base}/api/higgs/nodes"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        if let Some(n) = nodes.as_array().and_then(|a| {
            a.iter()
                .find(|n| n["connected"] == true && n["is_local"] != true)
        }) {
            node_id = n["endpoint_id"].as_str().unwrap().to_string();
            break;
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    assert!(!node_id.is_empty(), "node paired before leave");

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

    // The node retired ITSELF — gone from /api/higgs/nodes entirely (not lingering disconnected).
    let mut gone = false;
    for _ in 0..50 {
        let nodes: serde_json::Value = c
            .get(format!("{base}/api/higgs/nodes"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        if nodes
            .as_array()
            .is_some_and(|a| !a.iter().any(|n| n["endpoint_id"] == node_id))
        {
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
}

/// The hub kill switch: `POST /api/higgs/hub/disable` stops ALL node network activity (closes
/// the endpoint → no inbound dials, no relay; closes node transports → nodes disconnected) while
/// KEEPING the fleet's node list/routes; `POST /api/higgs/hub/enable` turns it back on. Asserts
/// the deterministic, hermetic behavior: enabled→disabled→enabled status, pairing gated (200 vs
/// 409) by the switch, and a paired node going disconnected-but-still-listed on disable. (Live
/// reconnect-with-route-survival is covered by the fleet-level
/// `remote_hub_e2e::node_reconnects_and_route_survives` + the `disconnect_all` unit test, since a
/// LAN-only rebind can land on a new UDP port the node's cached ticket can't redial.)
#[tokio::test]
async fn hub_kill_switch_disables_then_reenables_the_network() {
    let hub_home = tempfile::tempdir().unwrap();
    let node_home = tempfile::tempdir().unwrap();
    let port = free_port();

    let hub = Command::new(env!("CARGO_BIN_EXE_higgs"))
        .env("HIGGS_BIND", "127.0.0.1")
        .env("HIGGS_PORT", port.to_string())
        .env("HIGGS_HOME", hub_home.path())
        .env("HIGGS_HUB", "1")
        .env("HIGGS_IROH_LOCAL", "1")
        .env("RUST_LOG", "warn")
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("spawn hub");
    let _hub = Proc(hub);
    let base = format!("http://127.0.0.1:{port}");
    let c = reqwest::Client::new();

    // Wait for the server.
    let mut ready = false;
    for _ in 0..150 {
        if let Ok(r) = c.get(format!("{base}/health")).send().await {
            if r.status().is_success() {
                ready = true;
                break;
            }
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    assert!(ready, "hub server ready");

    // Boots enabled (HIGGS_HUB=1).
    let st: serde_json::Value = c
        .get(format!("{base}/api/higgs/hub"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(st["enabled"], true, "boots enabled: {st}");

    // Pair + dial a node; wait until it shows connected.
    let pair: serde_json::Value = c
        .post(format!("{base}/api/higgs/pair"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let ticket = pair["ticket"].as_str().expect("ticket").to_string();
    let token = pair["token"].as_str().expect("token").to_string();
    let _node = Proc(
        Command::new(env!("CARGO_BIN_EXE_higgs"))
            .arg("--node")
            .arg(&ticket)
            .arg(&token)
            .env("HIGGS_HOME", node_home.path())
            .env("HIGGS_IROH_LOCAL", "1")
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("spawn node"),
    );
    let mut node_id = String::new();
    for _ in 0..150 {
        let nodes: serde_json::Value = c
            .get(format!("{base}/api/higgs/nodes"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        if let Some(n) = nodes.as_array().and_then(|a| {
            a.iter()
                .find(|n| n["connected"] == true && n["is_local"] != true)
        }) {
            node_id = n["endpoint_id"].as_str().unwrap().to_string();
            break;
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    assert!(!node_id.is_empty(), "node connected before disable");

    // ── DISABLE — network off. ──
    let dis: serde_json::Value = c
        .post(format!("{base}/api/higgs/hub/disable"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(dis["enabled"], false, "disabled status: {dis}");

    // Pairing is gated off while disabled.
    let pair_status = c
        .post(format!("{base}/api/higgs/pair"))
        .send()
        .await
        .unwrap()
        .status();
    assert_eq!(pair_status.as_u16(), 409, "pair 409 while disabled");

    // The node's transport is closed: it goes disconnected, but STAYS listed (route/seed kept,
    // not retired) — the hub isn't accepting, so it can't reconnect during the disabled window.
    let mut disconnected_listed = false;
    for _ in 0..50 {
        let nodes: serde_json::Value = c
            .get(format!("{base}/api/higgs/nodes"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        if nodes.as_array().is_some_and(|a| {
            a.iter()
                .any(|n| n["endpoint_id"] == node_id.as_str() && n["connected"] == false)
        }) {
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
    let en: serde_json::Value = c
        .post(format!("{base}/api/higgs/hub/enable"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(en["enabled"], true, "re-enabled status: {en}");
    assert!(
        en["hub_id"].as_str().is_some_and(|s| !s.is_empty()),
        "hub id present after re-enable: {en}"
    );

    // Pairing works again, and the re-armed fleet ADMITS a freshly-paired node — proving enable
    // re-armed admission (disable disarms it via disconnect_all) and the new endpoint accepts.
    let pair2: serde_json::Value = c
        .post(format!("{base}/api/higgs/pair"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let ticket2 = pair2["ticket"].as_str().expect("ticket2").to_string();
    let token2 = pair2["token"].as_str().expect("token2").to_string();
    let node2_home = tempfile::tempdir().unwrap();
    let _node2 = Proc(
        Command::new(env!("CARGO_BIN_EXE_higgs"))
            .arg("--node")
            .arg(&ticket2)
            .arg(&token2)
            .env("HIGGS_HOME", node2_home.path())
            .env("HIGGS_IROH_LOCAL", "1")
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("spawn node2"),
    );
    let mut node2_connected = false;
    for _ in 0..150 {
        let nodes: serde_json::Value = c
            .get(format!("{base}/api/higgs/nodes"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        if nodes.as_array().is_some_and(|a| {
            a.iter().any(|n| {
                n["connected"] == true
                    && n["is_local"] != true
                    && n["endpoint_id"] != node_id.as_str()
            })
        }) {
            node2_connected = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    assert!(
        node2_connected,
        "re-enabled hub admits a freshly-paired node (fleet re-armed)"
    );
}
