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
    assert!(!ticket.is_empty() && token.starts_with("htk_"), "got a ticket + token");

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
        if nodes.as_array().is_some_and(|a| {
            a.iter().any(|n| n["connected"] == true)
        }) {
            connected = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    assert!(connected, "node paired and shows connected in /api/higgs/nodes");

    // ── HTTP remote-load (only with a GGUF): load a model on the paired node via the new
    // POST /api/higgs/nodes/load, then confirm it's remotely routable in /v1/models. ──
    if staged.is_some() {
        let nodes: serde_json::Value = c
            .get(format!("{base}/api/higgs/nodes"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        let node_id = nodes[0]["endpoint_id"].as_str().expect("endpoint_id").to_string();

        let load = c
            .post(format!("{base}/api/higgs/nodes/load"))
            .json(&serde_json::json!({ "node": node_id, "model": TINY_MODEL_ID }))
            .send()
            .await
            .unwrap();
        assert!(load.status().is_success(), "remote load over HTTP ok: {}", load.status());

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
            models["data"].as_array().is_some_and(|d| d.iter().any(|m| m["id"] == TINY_MODEL_ID)),
            "remote model routable in /v1/models: {models}"
        );

        // Unload it again via HTTP.
        let unload = c
            .post(format!("{base}/api/higgs/nodes/unload"))
            .json(&serde_json::json!({ "model": TINY_MODEL_ID }))
            .send()
            .await
            .unwrap();
        assert!(unload.status().is_success(), "remote unload over HTTP ok: {}", unload.status());
    }
}
