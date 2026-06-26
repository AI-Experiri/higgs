//! Black-box HUB control routes that need a LIVE fleet — the ERROR/edge HTTP branches of the
//! node-mutation handlers in `src/serve/control.rs` that only fire when a hub IS enabled and a
//! node IS paired (so the `not_a_hub` 409 guard is passed and we reach `fleet`/`hub` ops).
//!
//! `hub_server.rs` already drives the HAPPY paths of these routes (pair → nodes → load → models →
//! retire/label/leave) end-to-end. This file deliberately targets the lines those don't reach:
//! the `Err(...)` arms of `control_nodes_load`, `control_node_models`, and `control_nodes_unload`
//! when the hub+fleet is present but the operation itself fails (unknown node → HG027 503,
//! unknown model → 4xx, unknown served id → HG002 404), plus the unknown-node retire/relabel
//! flows against a real (single-node) fleet.
//!
//! Pattern is copied verbatim from `hub_server.rs`: spawn a real `higgs` in HUB mode with hermetic
//! iroh, mint a pairing token over `POST /api/higgs/pair`, dial it with a real `higgs --node`, and
//! wait until the remote node shows connected in `GET /api/higgs/nodes`. Then exercise the error
//! arms over HTTP. Pairing needs no GGUF, so the node/fleet always stands up; the load-a-real-model
//! happy assertion runs only when a tiny GGUF is available.

mod common;

use std::process::{Child, Command, Stdio};
use std::time::Duration;

use common::{stage_tiny_model, tiny_gguf_path, TINY_MODEL_ID};

/// A spawned `higgs` (hub or node). SIGTERM on drop so coverage profiles flush.
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

/// Spawn a hub `higgs` (HUB mode, hermetic iroh) on `port`, returning the guard + base URL once
/// `/health` answers. Copied from `hub_server.rs`.
async fn spawn_hub(port: u16, home: &std::path::Path) -> (Proc, String, reqwest::Client) {
    let hub = Command::new(env!("CARGO_BIN_EXE_higgs"))
        .env("HIGGS_BIND", "127.0.0.1")
        .env("HIGGS_PORT", port.to_string())
        .env("HIGGS_HOME", home)
        .env("HIGGS_HUB", "1")
        .env("HIGGS_IROH_LOCAL", "1")
        .env("RUST_LOG", "warn")
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("spawn hub");
    let guard = Proc(hub);
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
    assert!(ready, "hub server ready on {base}");
    (guard, base, c)
}

/// Mint a pairing ticket+token over the hub API, spawn a real `higgs --node` dialing it (with the
/// staged model dir if `model_dir` is Some), and return the node guard once it shows connected,
/// along with its remote EndpointId.
async fn pair_node(
    base: &str,
    c: &reqwest::Client,
    node_home: &std::path::Path,
    model_dir: Option<&std::path::Path>,
) -> (Proc, String) {
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

    let mut cmd = Command::new(env!("CARGO_BIN_EXE_higgs"));
    cmd.arg("--node")
        .arg(&ticket)
        .arg(&token)
        .env("HIGGS_HOME", node_home)
        .env("HIGGS_IROH_LOCAL", "1")
        .stdout(Stdio::null())
        .stderr(Stdio::inherit());
    if let Some(dir) = model_dir {
        cmd.env("HIGGS_MODEL_DIR", dir);
    }
    let node = Proc(cmd.spawn().expect("spawn node"));

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
    assert!(!node_id.is_empty(), "remote node paired + connected");
    (node, node_id)
}

/// With a LIVE hub + a connected node, the node-mutation routes pass the `not_a_hub` 409 guard and
/// reach the fleet/hub ops — so their `Err(...)` arms fire on bad arguments:
///
/// - `POST /api/higgs/nodes/load` with an UNKNOWN node id → the fleet's `transport()` can't find a
///   live link → HG027 `NodeUnreachable` → 503 (`control_error` → `http_status`).
/// - `GET  /api/higgs/nodes/{unknown}/models` → same HG027 → 503.
/// - `POST /api/higgs/nodes/unload` with an UNKNOWN served id → `require_served` → HG002
///   `ModelNotFound` → 404.
/// - `POST /api/higgs/nodes/load` with a real node but an UNKNOWN model → the node relays a load
///   failure (HG002) back through the fleet → a non-2xx client error.
/// - `POST /api/higgs/nodes/retire` / `nodes/label` with an UNKNOWN node id against the real hub →
///   retire is a hub-side no-op success; relabel is the 404 unknown-node arm.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn hub_node_mutation_error_arms_fire_with_a_live_fleet() {
    let hub_home = tempfile::tempdir().unwrap();
    let node_home = tempfile::tempdir().unwrap();
    let port = free_port();

    let (_hub, base, c) = spawn_hub(port, hub_home.path()).await;

    // Stage the tiny model into the node's read-only model dir if available, so the
    // happy "load a real model on the node" assertion can run; otherwise pairing-only.
    let staged = tiny_gguf_path().map(|g| stage_tiny_model(&g));
    let (_node, node_id) = pair_node(
        &base,
        &c,
        node_home.path(),
        staged.as_ref().map(tempfile::TempDir::path),
    )
    .await;

    // ── load on an UNKNOWN node → HG027 NodeUnreachable → 503 (fleet present, so past the 409). ──
    let bad_node = "0000000000000000000000000000000000000000000000000000000000000000";
    let load_bad_node = c
        .post(format!("{base}/api/higgs/nodes/load"))
        .json(&serde_json::json!({ "node": bad_node, "model": TINY_MODEL_ID }))
        .send()
        .await
        .unwrap();
    assert_eq!(
        load_bad_node.status().as_u16(),
        503,
        "load on an unknown node → HG027 503 (not the 409 not-a-hub guard)"
    );
    let body: serde_json::Value = load_bad_node.json().await.unwrap();
    assert!(
        body["error"].as_str().is_some_and(|e| e.contains("HG027")),
        "503 carries the HG027 node-unreachable code: {body}"
    );

    // ── GET /api/higgs/nodes/{unknown}/models → HG027 503 (same unreachable path). ──
    let scan_bad = c
        .get(format!("{base}/api/higgs/nodes/{bad_node}/models"))
        .send()
        .await
        .unwrap();
    assert_eq!(
        scan_bad.status().as_u16(),
        503,
        "scanning an unknown node → HG027 503"
    );
    let scan_body: serde_json::Value = scan_bad.json().await.unwrap();
    assert!(
        scan_body["error"]
            .as_str()
            .is_some_and(|e| e.contains("HG027")),
        "scan 503 carries HG027: {scan_body}"
    );

    // ── unload an UNKNOWN served id → HG002 ModelNotFound → 404 (require_served fails). ──
    let unload_unknown = c
        .post(format!("{base}/api/higgs/nodes/unload"))
        .json(&serde_json::json!({ "model": "no-such/served-id" }))
        .send()
        .await
        .unwrap();
    assert_eq!(
        unload_unknown.status().as_u16(),
        404,
        "unloading an unknown served id → HG002 404"
    );
    let unload_body: serde_json::Value = unload_unknown.json().await.unwrap();
    assert!(
        unload_body["error"]
            .as_str()
            .is_some_and(|e| e.contains("HG002")),
        "unload 404 carries HG002: {unload_body}"
    );

    // ── load an UNKNOWN model on the REAL connected node → the node relays a load failure back
    // through the fleet; the hub surfaces it as a non-2xx client error (not a hang, not a 200). ──
    let load_bad_model = c
        .post(format!("{base}/api/higgs/nodes/load"))
        .json(&serde_json::json!({ "node": node_id, "model": "definitely/not-a-real-model" }))
        .send()
        .await
        .unwrap();
    let status = load_bad_model.status();
    assert!(
        status.is_client_error() || status.is_server_error(),
        "loading an unknown model on a real node fails (not 2xx): got {status}"
    );
    assert!(
        !status.is_success(),
        "unknown-model load is never a success"
    );

    // ── relabel an UNKNOWN remote node against the LIVE hub → the `Ok(false)` arm → 404. ──
    let relabel_unknown = c
        .post(format!("{base}/api/higgs/nodes/label"))
        .json(&serde_json::json!({ "node": bad_node, "label": "ghost" }))
        .send()
        .await
        .unwrap();
    assert_eq!(
        relabel_unknown.status().as_u16(),
        404,
        "relabel of an unknown node (hub enabled) → 404 unknown-node arm"
    );

    // ── The REAL node still loads a model (happy fleet-load over HTTP), proving the error arms
    // above didn't wedge the live link — only runs with a staged GGUF. ──
    if staged.is_some() {
        let load_ok = c
            .post(format!("{base}/api/higgs/nodes/load"))
            .json(&serde_json::json!({ "node": node_id, "model": TINY_MODEL_ID }))
            .send()
            .await
            .unwrap();
        assert!(
            load_ok.status().is_success(),
            "real model loads on the live node after the error arms: {}",
            load_ok.status()
        );
        let ok_body: serde_json::Value = load_ok.json().await.unwrap();
        assert_eq!(ok_body["status"], "ok", "load ok body: {ok_body}");
        assert!(
            ok_body["worker_id"].as_u64().is_some(),
            "load reply carries a worker_id: {ok_body}"
        );

        // The freshly-loaded model is now remotely routable in the OpenAI catalog.
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
            "the remote-loaded model is routable in /v1/models: {models}"
        );
    }

    // ── retire the UNKNOWN node against the live hub: the hub's allowlist-removal is a no-op for
    // an id it never admitted, but the route still answers 2xx (idempotent retire). The real node
    // is untouched and still listed. ──
    let retire_unknown = c
        .post(format!("{base}/api/higgs/nodes/retire"))
        .json(&serde_json::json!({ "node": bad_node }))
        .send()
        .await
        .unwrap();
    assert!(
        retire_unknown.status().is_success(),
        "retiring an unknown node is an idempotent no-op 2xx: {}",
        retire_unknown.status()
    );
    let nodes: serde_json::Value = c
        .get(format!("{base}/api/higgs/nodes"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(
        nodes
            .as_array()
            .is_some_and(|a| a.iter().any(|n| n["endpoint_id"] == node_id.as_str())),
        "the real node is still listed after the unknown-node retire: {nodes}"
    );

    // ── retire the REAL node → it leaves the fleet entirely (drops from /api/higgs/nodes). ──
    let retire_real = c
        .post(format!("{base}/api/higgs/nodes/retire"))
        .json(&serde_json::json!({ "node": node_id }))
        .send()
        .await
        .unwrap();
    assert!(
        retire_real.status().is_success(),
        "retire of the real node ok: {}",
        retire_real.status()
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
            .is_some_and(|a| !a.iter().any(|n| n["endpoint_id"] == node_id.as_str()))
        {
            gone = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert!(gone, "the retired real node is removed from the fleet");
}
