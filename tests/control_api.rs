//! Black-box integration test for higgs's `/api/higgs/*` control surface.
//!
//! Spawns the real `higgs-server` binary and drives the full control lifecycle
//! over HTTP against an on-disk Nemotron model: scan → load → status/models →
//! version/logs/by-id → unload. `#[ignore]` because it loads a multi-GB GGUF;
//! run with `cargo test -p higgs --test control_api -- --ignored`.

mod common;

use common::{nemotron_id, spawn};

#[tokio::test]
#[ignore = "integration: spawns higgs-server + loads a real GGUF (run with --ignored)"]
async fn control_api_lifecycle() {
    let srv = spawn(11500).await;
    let c = reqwest::Client::new();
    let get = |path: String| c.get(format!("{}{path}", srv.base)).send();

    // status: worker alive, nothing loaded yet.
    let status: serde_json::Value = get("/api/higgs/status".into())
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(status["worker_alive"], true, "worker should be alive");
    assert!(status["loaded"].is_null(), "nothing loaded at start");

    // models: a live scan of the configured dirs.
    let models: serde_json::Value = get("/api/higgs/models".into())
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let Some(id) = nemotron_id(&models) else {
        eprintln!("SKIP control_api_lifecycle: no Nemotron model on disk");
        return;
    };
    assert!(
        !models["models"].as_array().unwrap().is_empty(),
        "scan found models"
    );

    // load the model.
    let load: serde_json::Value = c
        .post(format!("{}/api/higgs/models/load", srv.base))
        .json(&serde_json::json!({ "id": id }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(load["status"], "ok", "load returns ok");

    // status now reports the loaded model.
    let status: serde_json::Value = get("/api/higgs/status".into())
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(status["loaded"]["id"], id, "status shows loaded id");
    assert!(
        status["loaded"]["ctx_len"].as_u64().unwrap() > 0,
        "loaded ctx_len > 0"
    );

    // models list marks that entry "loaded".
    let models: serde_json::Value = get("/api/higgs/models".into())
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let entry = models["models"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["id"] == serde_json::json!(id))
        .expect("loaded model in list");
    assert_eq!(entry["state"], "loaded", "entry state is loaded");
    assert_eq!(entry["format"], "gguf");
    // Capabilities are derived from the GGUF template: Nemotron supports tools
    // and emits a reasoning block.
    assert_eq!(entry["supports_tools"], true, "Nemotron supports tools");
    assert_eq!(entry["supports_reasoning"], true, "Nemotron reasons");

    // model-by-id (the wildcard route handles the slashed HF repo id).
    let by_id: serde_json::Value = get(format!("/api/higgs/models/{id}"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(by_id["id"], id, "models/{{id}} returns the entry");

    // model-by-id for a NON-existent id → 404 with a HG-coded error envelope.
    let missing = get("/api/higgs/models/no-such-org/no-such-model".into())
        .await
        .unwrap();
    assert_eq!(missing.status(), 404, "unknown model id is 404");
    let missing_body: serde_json::Value = missing.json().await.unwrap();
    assert!(
        missing_body["error"]
            .as_str()
            .is_some_and(|e| e.contains("HG")),
        "404 carries a HG-coded error: {missing_body:?}"
    );

    // version + logs respond with their shapes.
    let version: serde_json::Value = get("/api/higgs/version".into())
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(version.is_object(), "version is an object");

    let logs: serde_json::Value = get("/api/higgs/logs?n=50".into())
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(logs["lines"].is_array(), "logs has a lines array");

    // logs with n=0 and the default (no n) both answer with a lines array.
    for q in ["/api/higgs/logs?n=0", "/api/higgs/logs"] {
        let l: serde_json::Value = get(q.into()).await.unwrap().json().await.unwrap();
        assert!(l["lines"].is_array(), "{q} has a lines array");
    }
    let zero: serde_json::Value = get("/api/higgs/logs?n=0".into())
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        zero["lines"].as_array().unwrap().len(),
        0,
        "n=0 returns no lines"
    );

    // system: hardware + runtime panels (the LM-Studio-style info).
    let system: serde_json::Value = get("/api/higgs/system".into())
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(
        !system["hardware"]["cpu_name"].as_str().unwrap().is_empty(),
        "system reports a CPU name"
    );
    assert!(
        system["hardware"]["ram_total_bytes"].as_u64().unwrap() > 0,
        "system reports total RAM"
    );
    assert_eq!(system["runtime"]["engine"], "llama.cpp", "runtime engine");

    // unload → status clears.
    let unload: serde_json::Value = c
        .post(format!("{}/api/higgs/models/unload", srv.base))
        .json(&serde_json::json!({}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(unload["status"], "ok", "unload returns ok");

    let status: serde_json::Value = get("/api/higgs/status".into())
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(status["loaded"].is_null(), "nothing loaded after unload");

    // unload again with nothing loaded — idempotent, still {"status":"ok"}.
    let unload2: serde_json::Value = c
        .post(format!("{}/api/higgs/models/unload", srv.base))
        .json(&serde_json::json!({}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(unload2["status"], "ok", "unload with nothing loaded is ok");

    // Re-load with EXPLICIT load params (ctx_len/gpu_layers/threads) — exercises
    // the non-default LoadParams branch in control_load. ctx_len is echoed back.
    let load2: serde_json::Value = c
        .post(format!("{}/api/higgs/models/load", srv.base))
        .json(&serde_json::json!({
            "id": id, "ctx_len": 2048, "gpu_layers": 0, "threads": 2
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(load2["status"], "ok", "explicit-params load returns ok");
    let status: serde_json::Value = get("/api/higgs/status".into())
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        status["loaded"]["ctx_len"].as_u64().unwrap(),
        2048,
        "explicit ctx_len honored"
    );

    // ── Worker stop → start cycle (done LAST: stop kills the worker) ──────────
    let stop: serde_json::Value = c
        .post(format!("{}/api/higgs/worker/stop", srv.base))
        .json(&serde_json::json!({}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(stop["status"], "ok", "worker/stop returns ok");
    let status: serde_json::Value = get("/api/higgs/status".into())
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        status["worker_alive"], false,
        "worker_alive is false after stop"
    );

    let start: serde_json::Value = c
        .post(format!("{}/api/higgs/worker/start", srv.base))
        .json(&serde_json::json!({}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(start["status"], "ok", "worker/start returns ok");
    let status: serde_json::Value = get("/api/higgs/status".into())
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        status["worker_alive"], true,
        "worker_alive flips back to true after start"
    );
}
