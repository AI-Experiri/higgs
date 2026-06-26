//! Black-box integration test for `src/supervisor.rs` multi-worker lifecycle.
//!
//! The local node is multi-worker: one `Supervisor` (one `--higgs-worker`
//! child) per distinct loaded model. This drives the REACHABLE supervisor
//! lifecycle paths over HTTP against the tiny on-disk model — the ones that do
//! NOT need a wedged worker to time out (`CONTROL_RPC_TIMEOUT`/`CHAT_RPC_TIMEOUT`
//! are 120s/600s) or fault injection:
//!
//! - concurrent loads of DIFFERENT models → multiple coexisting workers, each
//!   with its OWN child process (`do_spawn` / `production_factory` per id);
//! - each worker answers `M_STATUS` independently (`Supervisor::status`/`request`
//!   round-trips reported in `loaded_all`, each with a distinct `worker_id`);
//! - per-id unload (`unload_one` → `Supervisor::stop()` of ONE worker) leaves the
//!   OTHER worker(s) alive and serving — graceful per-worker stop, not a drain;
//! - reload the unloaded id → a fresh worker spawns (additive on the de-duped
//!   local instance set), back in `loaded_all`;
//! - chat against EACH distinct model drives that worker's reader/writer + the
//!   M_CHAT RPC end to end (the `chat()` register→send→cleanup path).
//!
//! NOT reachable here without timeouts / fault injection (left to unit tests in
//! `supervisor.rs::tests` via the injected duplex factory): a wedged-worker
//! `CONTROL_RPC_TIMEOUT`/`CHAT_RPC_TIMEOUT`/`SYSINFO_RPC_TIMEOUT` expiry, the
//! `attempt_restart`-vs-`stop()` race guards, the `WORKER_EXIT_TIMEOUT` SIGKILL
//! fallback, and the stale-reader `generation` guard.

mod common;

use common::{spawn_with_models, tiny_gguf_path};
use serde_json::{json, Value};

/// PIDs of the `--higgs-worker` children of `server_pid` — one per loaded model.
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

/// Multi-worker lifecycle: load 3 distinct models → each gets its own coexisting
/// worker (distinct child PID + distinct `worker_id`) → unload ONE → the others
/// survive and still serve → reload the unloaded id → it is back. Chat against
/// each resident model so every worker's RPC round-trip is exercised.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn multi_worker_coexist_unload_one_reload_and_chat() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("SKIP multi_worker_coexist_unload_one_reload_and_chat: tiny gguf not found (set HIGGS_TEST_GGUF)");
        return;
    };
    let ids = ["sup/a", "sup/b", "sup/c"];
    let srv = spawn_with_models(13200, &gguf, &ids).await;
    let c = reqwest::Client::new();

    let status = || async {
        c.get(format!("{}/api/higgs/status", srv.base))
            .send()
            .await
            .unwrap()
            .json::<Value>()
            .await
            .unwrap()
    };
    let loaded_ids = |s: &Value| -> Vec<String> {
        let mut v: Vec<String> = s["loaded_all"]
            .as_array()
            .expect("loaded_all is an array")
            .iter()
            .map(|e| e["id"].as_str().unwrap().to_owned())
            .collect();
        v.sort();
        v
    };

    // ── Load all three DISTINCT models: each spawns its OWN worker ──────────────
    for id in ids {
        let load = c
            .post(format!("{}/api/higgs/models/load", srv.base))
            .json(&json!({ "id": id }))
            .send()
            .await
            .unwrap();
        assert!(load.status().is_success(), "load {id} ok");
    }

    // ── Each model is a coexisting worker: distinct worker_id, own child PID ────
    let s = status().await;
    assert_eq!(
        loaded_ids(&s),
        ["sup/a", "sup/b", "sup/c"],
        "all three models resident as separate workers: {s}"
    );
    assert_eq!(
        s["worker_alive"], true,
        "primary worker answers M_STATUS: {s}"
    );
    let worker_ids: std::collections::HashSet<u64> = s["loaded_all"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| e["worker_id"].as_u64().expect("worker_id present"))
        .collect();
    assert_eq!(
        worker_ids.len(),
        3,
        "each resident model has a DISTINCT worker_id (own Supervisor): {s}"
    );
    // One OS child process per worker — `production_factory` spawns a fresh
    // `--higgs-worker` per `do_spawn`, so the server has 3 distinct worker kids.
    let pids = worker_child_pids(srv.pid());
    assert_eq!(
        pids.len(),
        3,
        "three distinct worker child processes coexist, got {pids:?}"
    );
    assert_eq!(
        pids.iter().collect::<std::collections::HashSet<_>>().len(),
        3,
        "worker child PIDs are all distinct: {pids:?}"
    );

    // ── Chat against EACH model: drives every worker's reader/writer + M_CHAT ───
    // Non-streaming so no SSE stream is left open. Short generation.
    for id in ids {
        let resp: Value = c
            .post(format!("{}/v1/chat/completions", srv.base))
            .json(&json!({
                "model": id, "stream": false, "max_tokens": 4,
                "messages": [{ "role": "user", "content": "hi" }]
            }))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert!(
            resp["choices"][0]["message"]["content"].is_string(),
            "chat against worker {id} returns content: {resp}"
        );
    }

    // ── Per-id unload: stop ONLY sup/b's worker; sup/a and sup/c survive ───────
    // `control_unload {"id":...}` → `unload_one` → `Supervisor::stop()` of that
    // ONE worker — a graceful per-worker stop, NOT the drain-all path.
    let unload: Value = c
        .post(format!("{}/api/higgs/models/unload", srv.base))
        .json(&json!({ "id": "sup/b" }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(unload["status"], "ok", "per-id unload of sup/b returns ok");

    let s = status().await;
    assert_eq!(
        loaded_ids(&s),
        ["sup/a", "sup/c"],
        "only sup/b unloaded — sup/a and sup/c workers survive: {s}"
    );
    assert_eq!(
        s["worker_alive"], true,
        "a surviving worker still answers M_STATUS after one sibling stops: {s}"
    );
    // The OS confirms exactly one worker child was reaped.
    let pids = worker_child_pids(srv.pid());
    assert_eq!(
        pids.len(),
        2,
        "exactly one worker child reaped by the per-id unload, got {pids:?}"
    );

    // A survivor still SERVES — its reader/writer + M_CHAT path is untouched by
    // the sibling's stop().
    let resp: Value = c
        .post(format!("{}/v1/chat/completions", srv.base))
        .json(&json!({
            "model": "sup/a", "stream": false, "max_tokens": 4,
            "messages": [{ "role": "user", "content": "hi" }]
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(
        resp["choices"][0]["message"]["content"].is_string(),
        "survivor sup/a still serves after sup/b was unloaded: {resp}"
    );

    // Chat for the UNLOADED id is a JIT-load (sup/b is a scanned id), so it comes
    // back as a fresh worker — exercise the explicit RELOAD path instead so we
    // assert the control-load → do_spawn path directly.
    let reload = c
        .post(format!("{}/api/higgs/models/load", srv.base))
        .json(&json!({ "id": "sup/b" }))
        .send()
        .await
        .unwrap();
    assert!(reload.status().is_success(), "reload of sup/b ok");

    let s = status().await;
    assert_eq!(
        loaded_ids(&s),
        ["sup/a", "sup/b", "sup/c"],
        "reloaded sup/b is a fresh worker, back in loaded_all: {s}"
    );
    let pids = worker_child_pids(srv.pid());
    assert_eq!(
        pids.len(),
        3,
        "reload spawned a fresh worker child (back to three), got {pids:?}"
    );

    // The reloaded worker serves too — fresh do_spawn → reader/writer wired.
    let resp: Value = c
        .post(format!("{}/v1/chat/completions", srv.base))
        .json(&json!({
            "model": "sup/b", "stream": false, "max_tokens": 4,
            "messages": [{ "role": "user", "content": "hi" }]
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(
        resp["choices"][0]["message"]["content"].is_string(),
        "reloaded sup/b serves a chat: {resp}"
    );

    // ── Reload the SAME id again (idempotent on the de-duped local set) ─────────
    // A second load of an already-resident id must NOT spawn a duplicate worker —
    // local is one-worker-per-model, so the child count stays put.
    let reload_same = c
        .post(format!("{}/api/higgs/models/load", srv.base))
        .json(&json!({ "id": "sup/a" }))
        .send()
        .await
        .unwrap();
    assert!(reload_same.status().is_success(), "idempotent reload ok");
    let s = status().await;
    assert_eq!(
        loaded_ids(&s),
        ["sup/a", "sup/b", "sup/c"],
        "idempotent reload of a resident id did not change the set: {s}"
    );
    assert_eq!(
        worker_child_pids(srv.pid()).len(),
        3,
        "idempotent reload spawned NO duplicate worker (still three)"
    );

    // ── Graceful bulk stop (worker/stop = non-terminal drain) reaps ALL ────────
    // Drives `Supervisor::stop()` for every worker; the node stays loadable.
    let stop: Value = c
        .post(format!("{}/api/higgs/worker/stop", srv.base))
        .json(&json!({}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(stop["status"], "ok", "worker/stop drains all workers");
    let s = status().await;
    assert_eq!(
        s["worker_alive"], false,
        "no worker alive after the bulk stop: {s}"
    );
    assert!(
        s["loaded_all"].as_array().unwrap().is_empty(),
        "loaded_all is empty after the bulk stop: {s}"
    );
}
