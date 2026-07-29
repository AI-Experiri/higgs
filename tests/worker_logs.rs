//! Per-worker Developer-Log consoles + persisted-params, in-process via the crate API.
//!
//! higgs is a library: the old `/api/higgs/*` HTTP surface is gone, so this drives
//! the in-process `Higgs` facade (`load`, `status`, `logs`, `unload_spec`) against a
//! REAL local llama.cpp worker (the `worker_exe` DI seam in `common::higgs_local`).
//! Loads TWO tiny models (two workers) and asserts:
//! - `logs(_, Some(LogSource::LocalWorker { worker }))` returns that worker's own
//!   stderr (the keyed path — empty if the relay reverts to the unkeyed
//!   `LogSource::Worker` push);
//! - `LogSource::Worker` is the UNION of every local worker's lines;
//! - unloading a model evicts its per-worker ring;
//! - `status.loaded_all` reports `ctx_len`/`gpu_layers`/`threads` for BOTH workers
//!   (filled from the live probe / the persisted load record — `null` before that fix).
//!
//! Skips (like every harness test) when no tiny GGUF is available.

mod common;

use common::higgs_local;
use higgs::log_bus::LogSource;
use higgs::node::worker_id::WorkerId;
use higgs::worker::engine::llamacpp::params::LlamaCppParams;
use higgs::worker::engine::CtxLen;
use higgs::LoadParams;

const IDS: [&str; 2] = ["wl/alpha", "wl/beta"];

#[tokio::test]
async fn per_worker_log_consoles_and_stub_params() {
    let Some(higgs) = higgs_local(&IDS).await else {
        eprintln!("SKIP worker_logs: no tiny GGUF (set HIGGS_TEST_GGUF)");
        return;
    };

    // Load both models with an explicit ctx_len so the persisted record is distinctive.
    // Only ctx_len is pinned (as the original `{ "id", "ctx_len": 512 }` request did);
    // gpu_layers/threads keep their engine defaults.
    for id in IDS {
        let lp = LlamaCppParams {
            ctx_len: CtxLen::fixed(512),
            ..LlamaCppParams::default()
        };
        higgs
            .load(id, Some(LoadParams::llamacpp(lp)))
            .await
            .unwrap_or_else(|e| panic!("load {id}: {e:?}"));
    }

    // Two workers resident; map served id → worker_id from status.loaded_all.
    let status = higgs.status().await.expect("status");
    let loaded_all = &status.loaded_all;
    assert_eq!(loaded_all.len(), 2, "both workers listed: {status:?}");

    let mut worker_ids = Vec::new();
    for info in loaded_all {
        worker_ids.push(info.worker_id);
        // EVERY entry (probed primary AND stub secondary) reports the load params:
        // a busy stub falls back to the persisted load record. `None` = fix reverted.
        assert_eq!(
            info.ctx_len,
            Some(CtxLen::Fixed { n: 512 }),
            "loaded_all entry for worker {} carries the recorded ctx_len: {info:?}",
            info.worker_id
        );
        assert!(
            info.gpu_layers.is_some() && info.threads.is_some(),
            "gpu_layers/threads filled from the load record for worker {}: {info:?}",
            info.worker_id
        );
    }
    worker_ids.sort_unstable();
    worker_ids.dedup();
    assert_eq!(worker_ids.len(), 2, "two distinct worker ids");

    // Each worker has its OWN keyed console with its load-time stderr.
    let logs_for = |src: Option<LogSource>| higgs.logs(2000, src);
    // Log delivery crosses two async relays (worker bus → node broadcast → LogBus),
    // so lines may land shortly AFTER `load` returns — poll briefly before asserting.
    let mut per_worker: Vec<Vec<String>> = vec![Vec::new(), Vec::new()];
    for (i, w) in worker_ids.iter().enumerate() {
        for _ in 0..50 {
            per_worker[i] = logs_for(Some(LogSource::LocalWorker {
                worker: WorkerId(*w),
            }));
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
        logs_for(Some(LogSource::LocalWorker {
            worker: WorkerId(9999)
        }))
        .is_empty(),
        "unknown worker id snapshots empty"
    );
    // The legacy `worker` selector is the union of both workers' consoles.
    let union = logs_for(Some(LogSource::Worker));
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
        .find(|i| i.id == IDS[0])
        .map(|i| i.worker_id)
        .expect("worker id for unloaded model");
    let before = logs_for(Some(LogSource::LocalWorker {
        worker: WorkerId(unloaded_worker),
    }))
    .len();
    assert!(before > 0, "unloaded worker logged before unload");
    higgs
        .unload_spec(Some(IDS[0]))
        .await
        .unwrap_or_else(|e| panic!("unload {}: {e:?}", IDS[0]));
    let kept_worker = *worker_ids
        .iter()
        .find(|w| **w != unloaded_worker)
        .expect("other worker");
    // Evicted = the pre-unload history is GONE. Empty is the normal outcome; up to
    // a couple of straggler stop-time lines are the accepted relay residual. A
    // reverted eviction keeps the whole `before` history and fails the bound.
    let after = logs_for(Some(LogSource::LocalWorker {
        worker: WorkerId(unloaded_worker),
    }))
    .len();
    assert!(
        after <= 2.min(before.saturating_sub(1)),
        "unloaded worker's ring is evicted (before={before}, after={after})"
    );
    assert!(
        !logs_for(Some(LogSource::LocalWorker {
            worker: WorkerId(kept_worker),
        }))
        .is_empty(),
        "surviving worker's console is untouched"
    );

    higgs.shutdown().await;
}
