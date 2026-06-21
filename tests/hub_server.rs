//! Black-box P3 production hub: spawn a real `higgs` server in HUB mode, mint a pairing token
//! over `POST /api/higgs/pair`, dial it with a real `higgs --node`, and confirm the node shows
//! up connected in `GET /api/higgs/nodes`. No GGUF needed — this exercises the hub listener +
//! pairing wiring, not inference, so it always runs.

use std::process::{Child, Command, Stdio};
use std::time::Duration;

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

    // Spawn the node, dialing the hub with the minted token.
    let node = Command::new(env!("CARGO_BIN_EXE_higgs"))
        .arg("--node")
        .arg(&ticket)
        .arg(&token)
        .env("HIGGS_HOME", node_home.path())
        .env("HIGGS_IROH_LOCAL", "1")
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("spawn node");
    let _node = Proc(node);

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
}
