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
}
