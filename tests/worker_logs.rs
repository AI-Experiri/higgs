//! Per-worker Developer-Log consoles + persisted-params stub, end-to-end over HTTP.
//!
//! Spawns a real `higgs`, loads TWO tiny models (two workers), and asserts:
//! - `GET /api/higgs/logs?source=worker:<id>` returns that worker's own stderr
//!   (the keyed `LogSource::LocalWorker` path — empty if the relay reverts to
//!   the unkeyed `LogSource::Worker` push);
//! - `?source=worker` is the UNION of every local worker's lines;
//! - unloading a model evicts its per-worker ring;
//! - `status.loaded_all` reports `ctx_len`/`gpu_layers`/`threads` for the
//!   SECONDARY (non-probed) worker from the persisted load record — `null`
//!   before that fix.
//!
//! Skips (like every spawn test) when no tiny GGUF is available.

mod common;

use common::{spawn_with_models, tiny_gguf_path};

const IDS: [&str; 2] = ["wl/alpha", "wl/beta"];

#[tokio::test]
async fn per_worker_log_consoles_and_stub_params() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("SKIP worker_logs: no tiny GGUF (set HIGGS_TEST_GGUF)");
        return;
    };
    let srv = spawn_with_models(13300, &gguf, &IDS).await;
    let c = reqwest::Client::new();

    // Load both models with an explicit ctx_len so the persisted record is distinctive.
    for id in IDS {
        let r = c
            .post(format!("{}/api/higgs/models/load", srv.base))
            .json(&serde_json::json!({ "id": id, "ctx_len": 512 }))
            .send()
            .await
            .expect("send load");
        assert_eq!(r.status(), 200, "load {id}");
    }

    // Two workers resident; map served id → worker_id from status.loaded_all.
    let status: serde_json::Value = c
        .get(format!("{}/api/higgs/status", srv.base))
        .send()
        .await
        .expect("get status")
        .json()
        .await
        .expect("status json");
    let loaded_all = status["loaded_all"].as_array().expect("loaded_all array");
    assert_eq!(loaded_all.len(), 2, "both workers listed: {status}");

    let mut worker_ids = Vec::new();
    for info in loaded_all {
        let w = info["worker_id"].as_u64().expect("worker_id") as u32;
        worker_ids.push(w);
        // EVERY entry (probed primary AND stub secondary) reports the load params:
        // the stub falls back to the persisted load record. `null` = fix reverted.
        assert_eq!(
            info["ctx_len"]["n"], 512,
            "loaded_all entry for worker {w} carries the recorded ctx_len: {info}"
        );
        assert!(
            !info["gpu_layers"].is_null() && !info["threads"].is_null(),
            "gpu_layers/threads filled from the load record for worker {w}: {info}"
        );
    }
    worker_ids.sort_unstable();
    worker_ids.dedup();
    assert_eq!(worker_ids.len(), 2, "two distinct worker ids");

    // Each worker has its OWN keyed console with its load-time stderr.
    let logs_for = |source: String| {
        let c = c.clone();
        let base = srv.base.clone();
        async move {
            let v: serde_json::Value = c
                .get(format!("{base}/api/higgs/logs?n=2000&source={source}"))
                .send()
                .await
                .expect("get logs")
                .json()
                .await
                .expect("logs json");
            v["lines"]
                .as_array()
                .expect("lines array")
                .iter()
                .map(|l| l.as_str().unwrap_or_default().to_owned())
                .collect::<Vec<_>>()
        }
    };
    // Log delivery crosses two async relays (worker bus → node broadcast → LogBus),
    // so lines may land shortly AFTER `load` returns — poll briefly before asserting.
    let mut per_worker: Vec<Vec<String>> = vec![Vec::new(), Vec::new()];
    for (i, w) in worker_ids.iter().enumerate() {
        for _ in 0..50 {
            per_worker[i] = logs_for(format!("worker:{w}")).await;
            if !per_worker[i].is_empty() {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }
        assert!(
            !per_worker[i].is_empty(),
            "worker {w} has its own keyed console (empty = lines still pushed unkeyed)"
        );
    }
    // A worker id that never existed has no console.
    assert!(
        logs_for("worker:9999".into()).await.is_empty(),
        "unknown worker id snapshots empty"
    );
    // The legacy `worker` selector is the union of both workers' consoles.
    let union = logs_for("worker".into()).await;
    assert!(
        union.len() >= per_worker[0].len().max(per_worker[1].len()),
        "`worker` unions all local workers ({} union vs {}/{} per-worker)",
        union.len(),
        per_worker[0].len(),
        per_worker[1].len()
    );

    // Unloading one model evicts ITS ring; the other worker's console survives.
    // Snapshot the doomed ring's size FIRST so the eviction check below can be
    // tolerant of the documented residual (a straggler line already in flight in
    // the async relay may recreate a tiny ring after eviction) while still
    // failing hard when eviction is reverted (the ring keeps its full history).
    let unloaded_worker = loaded_all
        .iter()
        .find(|i| i["id"] == IDS[0])
        .and_then(|i| i["worker_id"].as_u64())
        .expect("worker id for unloaded model") as u32;
    let before = logs_for(format!("worker:{unloaded_worker}")).await.len();
    assert!(before > 0, "unloaded worker logged before unload");
    let r = c
        .post(format!("{}/api/higgs/models/unload", srv.base))
        .json(&serde_json::json!({ "id": IDS[0] }))
        .send()
        .await
        .expect("send unload");
    assert_eq!(r.status(), 200, "unload {}", IDS[0]);
    let kept_worker = *worker_ids
        .iter()
        .find(|w| **w != unloaded_worker)
        .expect("other worker");
    // Evicted = the pre-unload history is GONE. Empty is the normal outcome; up to
    // a couple of straggler stop-time lines are the accepted relay residual. A
    // reverted eviction keeps the whole `before` history and fails the bound.
    let after = logs_for(format!("worker:{unloaded_worker}")).await.len();
    assert!(
        after <= 2.min(before.saturating_sub(1)),
        "unloaded worker's ring is evicted (before={before}, after={after})"
    );
    assert!(
        !logs_for(format!("worker:{kept_worker}")).await.is_empty(),
        "surviving worker's console is untouched"
    );
}
