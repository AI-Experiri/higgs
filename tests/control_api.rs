//! Black-box integration test for higgs's `/api/higgs/*` control surface.
//!
//! Spawns the real `higgs` binary and drives the full control lifecycle
//! over HTTP against a tiny on-disk model: scan → load → status/models →
//! version/logs/by-id/system → unload → worker/stop. The model is `ggml-org`'s
//! ~1MB `stories260K.gguf` (see `common`), staged into a temp scan root, so the
//! test exercises the real engine load/unload path in CI without a multi-GB GGUF.

mod common;

use common::{spawn_with_tiny_model, tiny_gguf_path, TINY_MODEL_ID};

#[tokio::test]
async fn control_api_lifecycle() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("SKIP control_api_lifecycle: tiny gguf not found (set HIGGS_TEST_GGUF)");
        return;
    };
    let srv = spawn_with_tiny_model(11500, &gguf).await;
    let c = reqwest::Client::new();
    let get = |path: String| c.get(format!("{}{path}", srv.base)).send();

    // status at boot: spawn-on-load means NO worker until a model is loaded.
    let status: serde_json::Value = get("/api/higgs/status".into())
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        status["worker_alive"], false,
        "no worker before first load (spawn-on-load)"
    );
    assert!(status["loaded"].is_null(), "nothing loaded at start");

    // models: a live scan of the configured dirs — the staged tiny model is here.
    let models: serde_json::Value = get("/api/higgs/models".into())
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let id = TINY_MODEL_ID;
    let arr = models["models"].as_array().unwrap();
    assert!(!arr.is_empty(), "scan found models");
    let scanned = arr
        .iter()
        .find(|m| m["id"] == serde_json::json!(id))
        .expect("scan lists the staged tiny model");
    // Gate-1 probe must judge this real llama-arch GGUF loadable.
    assert_eq!(scanned["loadable"], true, "tiny model is loadable: {scanned}");

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
    assert_eq!(entry["arch"], "llama", "stories260K is a llama-arch GGUF");
    // The tiny model embeds NO chat template (the engine falls back to chatml),
    // so the template-derived capability flags are false.
    assert_eq!(
        entry["supports_tools"], false,
        "no embedded template → no tools"
    );
    assert_eq!(
        entry["supports_reasoning"], false,
        "no embedded template → no reasoning"
    );

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
    assert_eq!(version["engine"], "llama.cpp", "version reports the engine");

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

    // ── Worker stop (done LAST: stop kills the worker) ───────────────────────
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
}
