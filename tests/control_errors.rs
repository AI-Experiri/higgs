//! Black-box integration tests for the `/api/higgs/*` control ERROR + edge paths
//! in `src/serve/control.rs`.
//!
//! These spawn the real `higgs` binary and drive the control surface over HTTP,
//! focusing on the failure/edge branches the happy-path lifecycle test
//! (`control_api.rs`) does not exercise: invalid/unknown model ids on
//! load/unload/tune/by-id, the `/system` snapshot shape, the no-op
//! `worker/stop` with nothing loaded, `version`, `nodes` (local node present),
//! relabeling the LOCAL instance, and the `hub` status shape with no hub.
//!
//! Each test stages the tiny `stories260K.gguf` model (see `common`) and picks a
//! UNIQUE port from base 12200. Every test SKIPs cleanly when the GGUF is absent.
//! No SSE stream is opened (those block graceful shutdown).

mod common;

use common::{spawn_with_tiny_model, tiny_gguf_path, TINY_MODEL_ID};

/// `POST /api/higgs/models/load` with a path-traversal / structurally-invalid id
/// is rejected by the host-side `validate_repo_id` guard → 400 with the HG015
/// `InvalidModelId` code in the error envelope (it never reaches the worker).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn load_invalid_id_is_hg015_bad_request() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("SKIP load_invalid_id_is_hg015_bad_request: tiny gguf not found");
        return;
    };
    let srv = spawn_with_tiny_model(12200, &gguf).await;
    let c = reqwest::Client::new();

    // Every one of these trips a DISTINCT branch of `validate_repo_id`:
    // a `..` traversal component, an absolute path, and an illegal character.
    for bad in ["../etc/passwd", "/abs/path", "bad id with spaces"] {
        let resp = c
            .post(format!("{}/api/higgs/models/load", srv.base))
            .json(&serde_json::json!({ "id": bad }))
            .send()
            .await
            .unwrap();
        assert_eq!(
            resp.status(),
            400,
            "invalid id {bad:?} → 400 (id-validation guard)"
        );
        let body: serde_json::Value = resp.json().await.unwrap();
        assert!(
            body["error"]
                .as_str()
                .is_some_and(|e| e.contains("[HG015]")),
            "invalid id {bad:?} carries the HG015 code: {body}"
        );
    }
}

/// `POST /api/higgs/models/load` with a well-formed but UNKNOWN id passes the
/// validation guard, then fails the host-side scan resolve → 404 with the HG002
/// `ModelNotFound` code.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn load_unknown_id_is_hg002_not_found() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("SKIP load_unknown_id_is_hg002_not_found: tiny gguf not found");
        return;
    };
    let srv = spawn_with_tiny_model(12201, &gguf).await;
    let c = reqwest::Client::new();

    let resp = c
        .post(format!("{}/api/higgs/models/load", srv.base))
        .json(&serde_json::json!({ "id": "no-such-org/no-such-model" }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404, "unknown id → 404 (not found on disk)");
    let body: serde_json::Value = resp.json().await.unwrap();
    assert!(
        body["error"]
            .as_str()
            .is_some_and(|e| e.contains("[HG002]")),
        "unknown id carries the HG002 ModelNotFound code: {body}"
    );
}

/// `POST /api/higgs/models/unload {"id": ...}` for a model that is NOT loaded is
/// an idempotent no-op: `Higgs::unload_one` resolves the served id, finds no
/// resident worker, and returns `{"status":"ok"}` (200) — it must NOT error.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unload_not_loaded_model_is_ok_noop() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("SKIP unload_not_loaded_model_is_ok_noop: tiny gguf not found");
        return;
    };
    let srv = spawn_with_tiny_model(12202, &gguf).await;
    let c = reqwest::Client::new();

    // The tiny model is staged + scannable but NOT loaded — unloading it by id is
    // a clean no-op (idempotent), distinct from the destructive `{}` drain-all.
    let resp = c
        .post(format!("{}/api/higgs/models/unload", srv.base))
        .json(&serde_json::json!({ "id": TINY_MODEL_ID }))
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        200,
        "unload of a not-loaded model is a 200 no-op"
    );
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["status"], "ok", "no-op unload returns ok: {body}");

    // A never-existed id is also a clean no-op (resolve finds nothing → Ok).
    let resp = c
        .post(format!("{}/api/higgs/models/unload", srv.base))
        .json(&serde_json::json!({ "id": "ghost-org/ghost-model" }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200, "unload of an unknown id is a 200 no-op");
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["status"], "ok");
}

/// `GET /api/higgs/models/{id}` — the by-id wildcard route on both branches:
/// an UNKNOWN slashed id → 404 HG002, and the STAGED tiny id → 200 with the
/// enriched (`not-loaded`, `gguf`, llama-arch) entry.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn model_by_id_unknown_404_and_staged_ok() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("SKIP model_by_id_unknown_404_and_staged_ok: tiny gguf not found");
        return;
    };
    let srv = spawn_with_tiny_model(12203, &gguf).await;
    let c = reqwest::Client::new();

    // Unknown slashed id → 404 with the HG002 code (the not-found branch).
    let missing = c
        .get(format!(
            "{}/api/higgs/models/ghost-org/ghost-model",
            srv.base
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(missing.status(), 404, "unknown id → 404");
    let body: serde_json::Value = missing.json().await.unwrap();
    assert!(
        body["error"]
            .as_str()
            .is_some_and(|e| e.contains("[HG002]")),
        "by-id 404 carries the HG002 code: {body}"
    );

    // Staged id → 200 with the enriched entry (the found branch; nothing loaded,
    // so the load state reads `not-loaded`).
    let found = c
        .get(format!("{}/api/higgs/models/{}", srv.base, TINY_MODEL_ID))
        .send()
        .await
        .unwrap();
    assert_eq!(found.status(), 200, "staged id is found");
    let entry: serde_json::Value = found.json().await.unwrap();
    assert_eq!(
        entry["id"], TINY_MODEL_ID,
        "by-id returns the entry: {entry}"
    );
    assert_eq!(entry["state"], "not-loaded", "scanned-not-loaded entry");
    assert_eq!(entry["format"], "gguf", "gguf format");
    assert_eq!(entry["arch"], "llama", "stories260K is a llama-arch GGUF");
}

/// `POST /api/higgs/models/tune` with a non-existent id fails the scan resolve in
/// `Higgs::tune` → 404 with the HG002 `ModelNotFound` code (the tune error
/// branch in `control_tune`). The id is in the BODY, matching `/models/load`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn tune_bad_id_is_hg002_not_found() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("SKIP tune_bad_id_is_hg002_not_found: tiny gguf not found");
        return;
    };
    let srv = spawn_with_tiny_model(12204, &gguf).await;
    let c = reqwest::Client::new();

    let resp = c
        .post(format!("{}/api/higgs/models/tune", srv.base))
        .json(&serde_json::json!({ "id": "no-such-org/no-such-model" }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404, "tune of an unknown id → 404");
    let body: serde_json::Value = resp.json().await.unwrap();
    assert!(
        body["error"]
            .as_str()
            .is_some_and(|e| e.contains("[HG002]")),
        "tune not-found carries the HG002 code: {body}"
    );
}

/// `GET /api/higgs/system` — assert the hardware/runtime/config three-panel shape
/// of the `SystemInfo` snapshot (CPU/RAM under `hardware`, the engine identity
/// under `runtime`, and the effective server config + limits under `config`).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn system_reports_hardware_runtime_config_shape() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("SKIP system_reports_hardware_runtime_config_shape: tiny gguf not found");
        return;
    };
    let srv = spawn_with_tiny_model(12205, &gguf).await;
    let c = reqwest::Client::new();

    let sys: serde_json::Value = c
        .get(format!("{}/api/higgs/system", srv.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();

    // hardware panel: a non-empty CPU name and a positive total RAM.
    assert!(
        sys["hardware"]["cpu_name"]
            .as_str()
            .is_some_and(|s| !s.is_empty()),
        "hardware.cpu_name is a non-empty string: {sys}"
    );
    assert!(
        sys["hardware"]["ram_total_bytes"]
            .as_u64()
            .is_some_and(|n| n > 0),
        "hardware.ram_total_bytes > 0: {sys}"
    );
    assert!(
        sys["hardware"]["cpu_cores"].as_u64().is_some_and(|n| n > 0),
        "hardware.cpu_cores > 0: {sys}"
    );

    // runtime panel: the llama.cpp engine identity.
    assert_eq!(
        sys["runtime"]["engine"], "llama.cpp",
        "runtime.engine is llama.cpp: {sys}"
    );
    assert!(
        sys["runtime"]["version"]
            .as_str()
            .is_some_and(|s| !s.is_empty()),
        "runtime.version is the live ggml version: {sys}"
    );

    // config panel: the effective server config carries the limits sub-object,
    // including the live idle-unload TTL (default 60min × 60 = 3600s).
    assert!(sys["config"].is_object(), "config panel present: {sys}");
    assert_eq!(
        sys["config"]["limits"]["idle_unload_ttl_secs"]
            .as_u64()
            .expect("config.limits.idle_unload_ttl_secs present"),
        3600,
        "config.limits.idle_unload_ttl_secs is the live 60min×60: {sys}"
    );
}

/// `POST /api/higgs/worker/stop` with NOTHING loaded is a graceful no-op: it
/// unconditionally returns `{"status":"ok"}` (the bulk-unload is best-effort and
/// always Ok), and the server stays up + reports no live worker afterwards.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn worker_stop_with_nothing_loaded_is_ok() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("SKIP worker_stop_with_nothing_loaded_is_ok: tiny gguf not found");
        return;
    };
    let srv = spawn_with_tiny_model(12206, &gguf).await;
    let c = reqwest::Client::new();

    let stop: serde_json::Value = c
        .post(format!("{}/api/higgs/worker/stop", srv.base))
        .json(&serde_json::json!({}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        stop["status"], "ok",
        "worker/stop with nothing loaded is ok"
    );

    // The server is still up and reports no live worker (spawn-on-load).
    let status: serde_json::Value = c
        .get(format!("{}/api/higgs/status", srv.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        status["worker_alive"], false,
        "no worker after a no-op worker/stop: {status}"
    );
}

/// `GET /api/higgs/version` — the build/engine identity envelope: a higgs
/// version, the llama.cpp engine, a non-empty engine + binding version, and
/// `gguf` among the supported formats.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn version_reports_build_and_engine() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("SKIP version_reports_build_and_engine: tiny gguf not found");
        return;
    };
    let srv = spawn_with_tiny_model(12207, &gguf).await;
    let c = reqwest::Client::new();

    let v: serde_json::Value = c
        .get(format!("{}/api/higgs/version", srv.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(
        v["higgs"].as_str().is_some_and(|s| !s.is_empty()),
        "higgs version present: {v}"
    );
    assert_eq!(v["engine"], "llama.cpp", "engine reported: {v}");
    assert!(
        v["engine_version"].as_str().is_some_and(|s| !s.is_empty()),
        "engine_version present: {v}"
    );
    assert!(
        v["binding"].as_str().is_some_and(|s| !s.is_empty()),
        "binding version present: {v}"
    );
    let fmts = v["supported_formats"].as_array().expect("formats array");
    assert!(
        fmts.contains(&serde_json::Value::String("gguf".to_owned())),
        "gguf is a supported format: {v}"
    );
}

/// `GET /api/higgs/nodes` — the local node is a first-class node and appears
/// FIRST even with no fleet/hub installed (`endpoint_id == "local"`, `is_local`,
/// always connected, with a label + inventory).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn nodes_lists_local_node_first() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("SKIP nodes_lists_local_node_first: tiny gguf not found");
        return;
    };
    let srv = spawn_with_tiny_model(12208, &gguf).await;
    let c = reqwest::Client::new();

    let nodes: Vec<serde_json::Value> = c
        .get(format!("{}/api/higgs/nodes", srv.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        nodes.len(),
        1,
        "only the local node with no fleet: {nodes:?}"
    );
    let local = &nodes[0];
    assert_eq!(local["endpoint_id"], "local", "local sentinel id: {local}");
    assert_eq!(local["is_local"], true, "flagged local: {local}");
    assert_eq!(
        local["connected"], true,
        "local is always connected: {local}"
    );
    assert!(
        local["label"].as_str().is_some_and(|s| !s.is_empty()),
        "local node has a label: {local}"
    );
    assert!(
        local["inventory"].is_object(),
        "local inventory present: {local}"
    );
}

/// `POST /api/higgs/nodes/label {"node":"local", ...}` renames THIS instance:
/// it writes the new name into the isolated `config.json` (under the temp
/// `HIGGS_HOME`) and the next `GET /api/higgs/nodes` shows the local node's new
/// label. (`node:"local"` is the one relabel branch that needs no hub.)
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn nodes_label_renames_local_instance() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("SKIP nodes_label_renames_local_instance: tiny gguf not found");
        return;
    };
    let srv = spawn_with_tiny_model(12209, &gguf).await;
    let c = reqwest::Client::new();

    let new_name = "renamed-local-box";
    let resp = c
        .post(format!("{}/api/higgs/nodes/label", srv.base))
        .json(&serde_json::json!({ "node": "local", "label": new_name }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200, "local relabel succeeds (no hub needed)");
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["status"], "ok", "relabel returns ok: {body}");

    // The local node now reports the new label.
    let nodes: Vec<serde_json::Value> = c
        .get(format!("{}/api/higgs/nodes", srv.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let local = nodes
        .iter()
        .find(|n| n["endpoint_id"] == "local")
        .expect("local node present");
    assert_eq!(
        local["label"], new_name,
        "the local node shows the new name: {local}"
    );
}

/// `GET /api/higgs/hub` — with no hub installed the status reports disabled,
/// no hub id, and zero nodes (lets the Fleet tab show an explicit "hub off"
/// state rather than inferring it from a `/pair` 409).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn hub_status_without_hub_is_disabled() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("SKIP hub_status_without_hub_is_disabled: tiny gguf not found");
        return;
    };
    let srv = spawn_with_tiny_model(12210, &gguf).await;
    let c = reqwest::Client::new();

    let resp = c
        .get(format!("{}/api/higgs/hub", srv.base))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200, "hub status answers 200");
    let v: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(v["enabled"], false, "no hub installed → disabled: {v}");
    assert_eq!(v["node_count"], 0, "no nodes when disabled: {v}");
    assert!(
        v["hub_id"].is_null(),
        "hub_id omitted/null when disabled: {v}"
    );
}
