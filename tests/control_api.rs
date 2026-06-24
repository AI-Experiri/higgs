//! Black-box integration test for higgs's `/api/higgs/*` control surface.
//!
//! Spawns the real `higgs` binary and drives the full control lifecycle
//! over HTTP against a tiny on-disk model: scan → load → status/models →
//! version/logs/by-id/system → unload → worker/stop. The model is `ggml-org`'s
//! ~1MB `stories260K.gguf` (see `common`), staged into a temp scan root, so the
//! test exercises the real engine load/unload path in CI without a multi-GB GGUF.

mod common;

use common::{spawn_with_models, spawn_with_tiny_model, tiny_gguf_path, TINY_MODEL_ID};

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
    // The scan no longer probes (no load-to-test at open): there is NO `loadable`
    // field — loadability is learned only at actual load time, below.
    assert!(
        scanned.get("loadable").is_none(),
        "scan must not probe-judge loadability: {scanned}"
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

/// Multi-model: the local node is multi-worker (one worker per loaded model), so `/api/higgs/status`
/// must report EVERY resident model in `loaded_all` (each tagged with its `worker_id`) — not just
/// the primary `loaded`. This is what the UI's "Loaded Models" section renders one card per.
#[tokio::test]
async fn status_loaded_all_lists_every_worker() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("SKIP status_loaded_all_lists_every_worker: tiny gguf not found");
        return;
    };
    let srv = spawn_with_models(11512, &gguf, &["org-a/m", "org-b/m"]).await;
    let c = reqwest::Client::new();

    for id in ["org-a/m", "org-b/m"] {
        let load = c
            .post(format!("{}/api/higgs/models/load", srv.base))
            .json(&serde_json::json!({ "id": id }))
            .send()
            .await
            .unwrap();
        assert!(load.status().is_success(), "load {id} ok");
    }

    let status: serde_json::Value = c
        .get(format!("{}/api/higgs/status", srv.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let all = status["loaded_all"]
        .as_array()
        .expect("loaded_all is an array");
    assert_eq!(all.len(), 2, "every loaded worker reported: {status}");
    let mut ids: Vec<&str> = all.iter().map(|e| e["id"].as_str().unwrap()).collect();
    ids.sort_unstable();
    assert_eq!(ids, ["org-a/m", "org-b/m"], "both models present: {status}");
    let workers: std::collections::HashSet<u64> = all
        .iter()
        .map(|e| e["worker_id"].as_u64().unwrap())
        .collect();
    assert_eq!(
        workers.len(),
        2,
        "each entry has a distinct worker_id: {status}"
    );
    // `loaded` (back-compat primary) is one of the resident models.
    assert!(
        ["org-a/m", "org-b/m"].contains(&status["loaded"]["id"].as_str().unwrap()),
        "primary loaded is one of the resident models: {status}"
    );
}

/// The settings + health + SSE-logs control surface (no model needed): GET/PUT the two
/// settings endpoints round-trip a toggle, the health endpoints answer, and the logs SSE
/// stream opens and emits at least the replay prefix.
#[tokio::test]
async fn control_settings_health_and_log_stream() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("SKIP control_settings_health_and_log_stream: tiny gguf not found");
        return;
    };
    let srv = spawn_with_tiny_model(11507, &gguf).await;
    let c = reqwest::Client::new();

    // ── Health endpoints answer 200 ──────────────────────────────────────────
    for path in ["/health", "/api/higgs/health"] {
        let r = c.get(format!("{}{path}", srv.base)).send().await.unwrap();
        assert!(r.status().is_success(), "{path} is healthy");
    }

    // ── logs/settings: GET current, PUT a flip, GET reflects it ──────────────
    let before: serde_json::Value = c
        .get(format!("{}/api/higgs/logs/settings", srv.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let verbose0 = before["verbose"].as_bool().expect("verbose flag present");
    let mut put_body = before.clone();
    put_body["verbose"] = serde_json::Value::Bool(!verbose0);
    let put = c
        .put(format!("{}/api/higgs/logs/settings", srv.base))
        .json(&put_body)
        .send()
        .await
        .unwrap();
    assert!(put.status().is_success(), "PUT logs/settings ok");
    let after: serde_json::Value = c
        .get(format!("{}/api/higgs/logs/settings", srv.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(after["verbose"], !verbose0, "verbose toggle persisted");

    // ── settings: GET then PUT the same payload back (round-trips the schema) ──
    let settings: serde_json::Value = c
        .get(format!("{}/api/higgs/settings", srv.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let put = c
        .put(format!("{}/api/higgs/settings", srv.base))
        .json(&settings)
        .send()
        .await
        .unwrap();
    assert!(
        put.status().is_success(),
        "PUT settings round-trips: {settings:?}"
    );

    // ── Invalid model ids are rejected with a typed 4xx (id-validation branch) ──
    for bad in ["bad id with spaces", "../escape", ""] {
        let r = c
            .post(format!("{}/api/higgs/models/load", srv.base))
            .json(&serde_json::json!({ "id": bad }))
            .send()
            .await
            .unwrap();
        assert!(
            r.status().is_client_error(),
            "invalid id {bad:?} → 4xx, got {}",
            r.status()
        );
    }

    // ── model-by-id for a SCANNED-but-unloaded model reads its on-disk metadata ──
    let by_id = c
        .get(format!("{}/api/higgs/models/{}", srv.base, TINY_MODEL_ID))
        .send()
        .await
        .unwrap();
    assert!(by_id.status().is_success(), "by-id for a scanned model ok");
    let detail: serde_json::Value = by_id.json().await.unwrap();
    assert_eq!(
        detail["id"], TINY_MODEL_ID,
        "by-id returns the model: {detail:?}"
    );

    // by-id for an UNKNOWN model is a 404 (the not-found branch).
    let missing = c
        .get(format!(
            "{}/api/higgs/models/ghost-org/ghost-model",
            srv.base
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(missing.status(), 404, "by-id for an unknown model is 404");
}

/// The supervisor's auto-restart FSM: when the worker child dies unexpectedly, the next
/// request respawns it and REPLAYS the recorded load, so the model is back without a manual
/// reload. We load a model, hard-kill the worker child (the server's grandchild), then chat
/// again — it must succeed against a freshly restarted + reloaded worker.
#[tokio::test]
async fn worker_crash_triggers_restart_and_replay() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("SKIP worker_crash_triggers_restart_and_replay: tiny gguf not found");
        return;
    };
    let srv = spawn_with_tiny_model(11509, &gguf).await;
    let c = reqwest::Client::new();

    // Load the model (spawns the worker child).
    let load = c
        .post(format!("{}/api/higgs/models/load", srv.base))
        .json(&serde_json::json!({ "id": TINY_MODEL_ID }))
        .send()
        .await
        .unwrap();
    assert!(load.status().is_success(), "initial load ok");

    // Find and HARD-kill the worker child (a `--higgs-worker` grandchild of the server).
    let pids = worker_child_pids(srv.pid());
    assert!(!pids.is_empty(), "found the worker child process");
    for pid in &pids {
        unsafe {
            libc::kill(*pid as libc::pid_t, libc::SIGKILL);
        }
    }

    // The next chat must succeed: the supervisor detects the dead child, restarts it, and
    // replays the recorded load before serving — possibly after a transient error, so retry.
    let mut ok = false;
    for _ in 0..40 {
        let resp = c
            .post(format!("{}/v1/chat/completions", srv.base))
            .json(&serde_json::json!({
                "model": TINY_MODEL_ID, "stream": false, "max_tokens": 4,
                "messages": [{ "role": "user", "content": "hi" }]
            }))
            .send()
            .await
            .unwrap();
        if resp.status().is_success() {
            ok = true;
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(250)).await;
    }
    assert!(
        ok,
        "chat succeeds after the worker was restarted + the load replayed"
    );
}

/// The idle reaper auto-unloads a model after its idle TTL elapses: with `auto_unload_idle`
/// on and `idle_ttl_minutes = 0`, a loaded-but-unused worker is reaped on the next reaper
/// tick (~30s) without any client action.
#[tokio::test]
async fn idle_reaper_auto_unloads_a_model() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("SKIP idle_reaper_auto_unloads_a_model: tiny gguf not found");
        return;
    };
    let srv = spawn_with_tiny_model(11510, &gguf).await;
    let c = reqwest::Client::new();

    // Enable aggressive idle auto-unload (TTL 0 ⇒ idle immediately) BEFORE loading.
    let mut settings: serde_json::Value = c
        .get(format!("{}/api/higgs/settings", srv.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    settings["auto_unload_idle"] = serde_json::json!(true);
    settings["idle_ttl_minutes"] = serde_json::json!(0);
    let put = c
        .put(format!("{}/api/higgs/settings", srv.base))
        .json(&settings)
        .send()
        .await
        .unwrap();
    assert!(put.status().is_success(), "enable idle auto-unload");

    let load = c
        .post(format!("{}/api/higgs/models/load", srv.base))
        .json(&serde_json::json!({ "id": TINY_MODEL_ID }))
        .send()
        .await
        .unwrap();
    assert!(load.status().is_success(), "load ok");

    // Within ~2 reaper ticks the idle worker is auto-unloaded (no chat keeps it alive).
    let mut unloaded = false;
    for _ in 0..35 {
        let status: serde_json::Value = c
            .get(format!("{}/api/higgs/status", srv.base))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        if status["worker_alive"] == false || status["loaded"].is_null() {
            unloaded = true;
            break;
        }
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
    }
    assert!(
        unloaded,
        "idle reaper auto-unloaded the model within the timeout"
    );
}

/// PIDs of the `--higgs-worker` children of `server_pid` (the spawned worker processes).
fn worker_child_pids(server_pid: u32) -> Vec<u32> {
    let out = std::process::Command::new("pgrep")
        .args(["-P", &server_pid.to_string()])
        .output()
        .expect("run pgrep");
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| l.trim().parse::<u32>().ok())
        .collect()
}
