use super::*;
use crate::node::test_support::{fake_runtime as fake_runtime_with_dirs, fake_runtime_load_fails};
use crate::worker::engine::{CtxLen, GpuLayers};
use tempfile::TempDir;

/// Like `fake_runtime_with_models` but the worker fails every M_LOAD (post-spawn
/// failure) so the reap-the-spawned-worker path is exercised.
fn load_failing_runtime_with_models(ids: &[&str]) -> (NodeRuntime, TempDir) {
    let dir = TempDir::new().expect("staging dir");
    for id in ids {
        let model_dir = dir.path().join(id);
        std::fs::create_dir_all(&model_dir).expect("model dir");
        std::fs::write(model_dir.join("m.gguf"), b"GGUF\x00 dummy").expect("write dummy gguf");
    }
    let rt = fake_runtime_load_fails(vec![dir.path().to_path_buf()]);
    (rt, dir)
}

/// A fake-backed runtime whose scan root contains a dummy GGUF for each `id`, so the
/// real `load` path (resolve → spawn → M_LOAD) runs without llama.cpp. Keep the
/// returned `TempDir` alive for the test's duration.
fn fake_runtime_with_models(ids: &[&str]) -> (NodeRuntime, TempDir) {
    let dir = TempDir::new().expect("staging dir");
    for id in ids {
        let model_dir = dir.path().join(id);
        std::fs::create_dir_all(&model_dir).expect("model dir");
        std::fs::write(model_dir.join("m.gguf"), b"GGUF\x00 dummy").expect("write dummy gguf");
    }
    let rt = fake_runtime_with_dirs(vec![dir.path().to_path_buf()]);
    (rt, dir)
}

fn load_params(id: &str) -> NodeLoadParams {
    NodeLoadParams {
        id: id.into(),
        ctx_len: None,
        gpu_layers: None,
        threads: None,
        params: None,
    }
}

/// ONE zero policy: every count-zero reads as ABSENT — the base `threads: 0`
/// and the RICH zero thread/batch counts (`n_threads_batch`/`n_batch`/
/// `n_ubatch`/`n_seq_max`) are all SKIPPED so the engine/worker defaults
/// apply; `with_n_threads(0)` / `with_n_threads_batch(0)` GGML-assert
/// territory never leaves the node, at load OR at first chat. Fail-on-revert
/// for the skips.
#[test]
fn worker_load_params_floors_and_skips_degenerate_zeros() {
    use crate::worker::engine::llamacpp::params::LlamaCppParams;
    let rich: LlamaCppParams = serde_json::from_value(serde_json::json!({
        "n_threads_batch": 0,
        "n_batch": 0,
        "n_ubatch": 0,
        "n_seq_max": 0
    }))
    .expect("rich params parse");
    let v = worker_load_params("org/m", "/p/m.gguf", None, None, Some(0), &Some(rich));
    assert!(
        v.get("threads").is_none(),
        "base threads 0 reads as ABSENT (worker default), one zero policy: {v}"
    );
    for k in ["n_threads_batch", "n_batch", "n_ubatch", "n_seq_max"] {
        assert!(
            v.get(k).is_none(),
            "zero {k} is skipped (absence = engine default): {v}"
        );
    }

    // Base fields smuggled INSIDE the rich object are IGNORED — the base trio
    // is authoritative at the top level only (documented on the wire op) — and
    // this is contractual, not accidental.
    let rich: LlamaCppParams = serde_json::from_value(serde_json::json!({
        "ctx_len": { "kind": "fixed", "n": 8192 },
        "threads": 9
    }))
    .expect("rich params parse");
    let v = worker_load_params("org/m", "/p/m.gguf", Some(2048), None, None, &Some(rich));
    assert_eq!(
        v.get("ctx_len").and_then(serde_json::Value::as_u64),
        Some(2048),
        "the top-level base wins; the rich-object ctx is ignored: {v}"
    );
    assert!(
        v.get("threads").is_none(),
        "a rich-object base field never rides when the base is absent: {v}"
    );

    // NON-zero twins of the same fields still ride verbatim.
    let rich: LlamaCppParams = serde_json::from_value(serde_json::json!({
        "n_threads_batch": 3,
        "n_batch": 512
    }))
    .expect("rich params parse");
    let v = worker_load_params("org/m", "/p/m.gguf", None, None, None, &Some(rich));
    assert_eq!(
        v.get("n_threads_batch").and_then(serde_json::Value::as_u64),
        Some(3),
        "non-zero rich overrides still ride: {v}"
    );
    assert_eq!(
        v.get("n_batch").and_then(serde_json::Value::as_u64),
        Some(512),
        "non-zero rich overrides still ride: {v}"
    );
}

/// An absent base field (e.g. auto `ctx_len`) is OMITTED — not `null` — so the
/// worker's `LlamaCppParams` deserialize doesn't fail and drop the merged rich
/// overrides. The whole object must deserialize cleanly into `LlamaCppParams`.
#[test]
fn worker_load_params_omits_null_base_and_keeps_overrides() {
    use crate::worker::engine::llamacpp::params::LlamaCppParams;
    use crate::worker::engine::{FlashAttn, KvCacheKind};
    let mut rich = LlamaCppParams::base(CtxLen::Auto, GpuLayers::All, 4);
    rich.flash_attn = Some(FlashAttn::On);
    rich.type_k = Some(KvCacheKind::Q8_0);
    rich.cpu_moe = Some(true);
    // ctx_len None (auto) with rich overrides present.
    let v = worker_load_params(
        "org/m",
        "/x.gguf",
        None,
        Some(GpuLayers::All),
        Some(4),
        &Some(rich),
    );
    let obj = v.as_object().unwrap();
    assert!(
        !obj.contains_key("ctx_len"),
        "absent base field omitted, not null"
    );
    assert_eq!(obj["gpu_layers"], serde_json::json!({ "kind": "all" }));
    assert_eq!(obj["flash_attn"], "on", "rich override survives");
    assert_eq!(obj["type_k"], "Q8_0");
    assert_eq!(obj["cpu_moe"], true);
    // The whole object deserializes into LlamaCppParams (no null → no failure →
    // overrides not dropped). id/path are unknown fields, ignored by serde.
    let back: LlamaCppParams = serde_json::from_value(v).unwrap();
    assert_eq!(back.flash_attn, Some(FlashAttn::On));
    assert_eq!(back.type_k, Some(KvCacheKind::Q8_0));
    assert_eq!(back.cpu_moe, Some(true));
}

async fn load(rt: &NodeRuntime, id: &str) -> (WorkerId, Value) {
    rt.load(load_params(id)).await.unwrap()
}

#[tokio::test]
async fn load_assigns_ids_and_kill_frees_them() {
    let (rt, _dir) = fake_runtime_with_models(&["org/m-a", "org/m-b"]);
    let (a, _) = load(&rt, "org/m-a").await;
    let (b, _) = load(&rt, "org/m-b").await;
    assert_ne!(a, b);
    assert_eq!(rt.worker_ids().await.len(), 2, "two concurrent workers");
    rt.kill(a).await.unwrap();
    assert_eq!(rt.worker_ids().await.len(), 1);
    assert!(rt.kill(a).await.is_err(), "killing a freed id errors");
}

#[tokio::test]
async fn load_returns_worker_load_result() {
    let (rt, _dir) = fake_runtime_with_models(&["org/model"]);
    let (_, loaded) = load(&rt, "org/model").await;
    assert_eq!(loaded["id"], "org/model");
}

#[tokio::test]
async fn load_resolves_path_and_errors_when_model_absent() {
    // No model staged → resolution yields HG002 ModelNotFound, no worker spawned.
    let (rt, _dir) = fake_runtime_with_models(&[]);
    let err = rt.load(load_params("missing/model")).await.unwrap_err();
    assert!(err.to_string().starts_with("[HG002]"), "got {err}");
    assert!(
        rt.worker_ids().await.is_empty(),
        "no worker spawned on resolve failure"
    );
}

#[tokio::test]
async fn status_forwards_to_the_worker() {
    let (rt, _dir) = fake_runtime_with_models(&["org/m"]);
    let (id, _) = load(&rt, "org/m").await;
    let status = rt.status(id).await.unwrap();
    assert!(status.get("loaded").is_some());
    assert!(
        rt.status(WorkerId(999)).await.is_err(),
        "unknown worker errors"
    );
}

#[tokio::test]
async fn unload_stops_and_frees() {
    let (rt, _dir) = fake_runtime_with_models(&["org/m"]);
    let (id, _) = load(&rt, "org/m").await;
    rt.unload(id).await.unwrap();
    assert!(rt.worker_ids().await.is_empty());
}

#[tokio::test]
async fn inventory_lists_resident_workers_with_their_models() {
    let (rt, _dir) = fake_runtime_with_models(&["org/a", "org/b"]);
    let (wa, _) = load(&rt, "org/a").await;
    load(&rt, "org/b").await;
    let inv = rt.inventory().await.unwrap();
    assert!(
        inv["hardware"]["cpu_cores"].as_u64().unwrap() > 0,
        "real hw"
    );
    assert!(!inv["os"].as_str().unwrap().is_empty(), "os present");
    let workers = inv["workers"].as_array().unwrap();
    assert_eq!(workers.len(), 2, "both workers listed");
    let a = workers
        .iter()
        .find(|w| w["worker_id"].as_u64().unwrap() as u32 == wa.0)
        .unwrap();
    assert_eq!(a["model"], "org/a", "worker reports its model");
}

#[tokio::test]
async fn shutdown_all_drains_every_worker() {
    let (rt, _dir) = fake_runtime_with_models(&["org/a", "org/b"]);
    load(&rt, "org/a").await;
    load(&rt, "org/b").await;
    assert_eq!(rt.worker_ids().await.len(), 2);
    rt.shutdown_all().await;
    assert!(rt.worker_ids().await.is_empty(), "all workers drained");
}

#[tokio::test]
async fn scan_lists_staged_models() {
    let (rt, _dir) = fake_runtime_with_models(&["org/seen"]);
    let scanned = rt.scan().await.unwrap();
    let models = scanned["models"].as_array().expect("models array");
    assert!(
        models.iter().any(|m| m["id"] == "org/seen"),
        "scan lists the staged model: {scanned}"
    );
}

// `gpus()` enumerates host GPUs via a transient worker (the engine's `Higgs::sysinfo`
// input post-P4b). The fake worker reports an empty GPU list, returned without hanging.
#[tokio::test]
async fn gpus_enumerates_via_transient_worker() {
    let (rt, _dir) = fake_runtime_with_models(&[]);
    let gpus = rt.gpus().await;
    assert!(gpus.is_empty(), "fake worker reports no GPUs");
}

#[tokio::test]
async fn sysinfo_reports_real_hardware() {
    let (rt, _dir) = fake_runtime_with_models(&[]);
    let sys = rt.sysinfo().await.unwrap();
    assert!(sys["hardware"]["cpu_cores"].as_u64().unwrap() > 0);
    assert!(sys["runtime"].is_object(), "runtime block present");
}

// The spawn-and-commit property: two loads issued back-to-back run CONCURRENTLY (a slow
// load does not head-of-line-block the next op), and the same model on two workers gets
// two distinct ids — the per-node basis for duplicate served-id suffixes.
#[tokio::test]
async fn concurrent_loads_of_one_model_get_distinct_ids() {
    let (rt, _dir) = fake_runtime_with_models(&["org/dup"]);
    let (a, b) = tokio::join!(load(&rt, "org/dup"), load(&rt, "org/dup"));
    assert_ne!(a.0, b.0, "same model on two workers → two ids");
    assert_eq!(rt.worker_ids().await.len(), 2);
}

// A post-spawn load failure (worker spawned, M_LOAD errors) surfaces the error and
// leaves no committed worker — the spawned child is reaped.
#[tokio::test]
async fn post_spawn_load_failure_reaps_and_errors() {
    let (rt, _dir) = load_failing_runtime_with_models(&["org/m"]);
    let err = rt.load(load_params("org/m")).await.unwrap_err();
    assert!(err.to_string().contains("HG017"), "got {err}");
    assert!(
        rt.worker_ids().await.is_empty(),
        "failed load committed no worker"
    );
}

// A post-spawn load failure that resolves DURING shutdown is reaped through the teardown
// coordinator — shutdown_all completes cleanly with nothing left committed.
#[tokio::test]
async fn post_spawn_failure_during_shutdown_drains() {
    let (rt, _dir) = load_failing_runtime_with_models(&["org/a", "org/b"]);
    // Issue loads (which will fail) concurrently with shutdown so some resolve mid-drain.
    let (_l1, _l2, _s) = tokio::join!(
        rt.load(load_params("org/a")),
        rt.load(load_params("org/b")),
        rt.shutdown_all(),
    );
    assert!(rt.worker_ids().await.is_empty(), "drained after shutdown");
}

// shutdown_all is idempotent: a second call (after the coordinator already finished)
// returns immediately instead of hanging on a never-fired completion.
#[tokio::test]
async fn shutdown_all_is_idempotent() {
    let (rt, _dir) = fake_runtime_with_models(&["org/m"]);
    load(&rt, "org/m").await;
    rt.shutdown_all().await;
    // Second call must not block (would hang if it parked for a gone coordinator).
    tokio::time::timeout(std::time::Duration::from_secs(5), rt.shutdown_all())
        .await
        .expect("second shutdown_all returns promptly");
}

// Two concurrent shutdown_all callers both complete: the second parks behind the same
// coordinator and is released by the single ShutdownDone.
#[tokio::test]
async fn concurrent_shutdown_all_callers_all_complete() {
    let (rt, _dir) = fake_runtime_with_models(&["org/a", "org/b"]);
    load(&rt, "org/a").await;
    load(&rt, "org/b").await;
    let (a, b) = tokio::join!(rt.shutdown_all(), rt.shutdown_all());
    let _ = (a, b);
    assert!(rt.worker_ids().await.is_empty(), "all workers drained");
}

// A load issued AFTER shutdown_all is rejected (shutdown is terminal) — it must not
// resurrect a worker that would survive the drain.
#[tokio::test]
async fn load_after_shutdown_is_rejected() {
    let (rt, _dir) = fake_runtime_with_models(&["org/m"]);
    rt.shutdown_all().await;
    let err = rt.load(load_params("org/m")).await.unwrap_err();
    assert!(
        err.to_string().contains("shutting down"),
        "got {err} — load must be refused after shutdown"
    );
    assert!(
        rt.worker_ids().await.is_empty(),
        "no worker after a post-shutdown load"
    );
}

/// Poll until the runtime has no resident workers (idle reaper did its job), up to ~5s.
/// Robust against scheduling jitter under full-suite parallel load (vs a fixed sleep).
async fn wait_reaped(rt: &NodeRuntime) -> bool {
    for _ in 0..100 {
        if rt.worker_ids().await.is_empty() {
            return true;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    false
}

// `instances()` lists resident (worker, raw model) pairs, and lifecycle events fire on
// load/unload — the surface the local engine consumes in P4b.
#[tokio::test]
async fn instances_and_events_track_load_unload() {
    let (rt, _dir) = fake_runtime_with_models(&["org/a", "org/b"]);
    let mut events = rt.events();
    let (wa, _) = load(&rt, "org/a").await;
    load(&rt, "org/b").await;
    let mut insts = rt.instances().await;
    insts.sort();
    assert_eq!(insts.len(), 2);
    assert!(insts.iter().any(|(w, m)| *w == wa && m == "org/a"));
    // A load emitted a ModelLoaded event.
    let ev = events.recv().await.unwrap();
    assert!(matches!(ev, HiggsEvent::ModelLoaded { .. }));
    // Unload removes the instance and emits ModelUnloaded.
    rt.unload(wa).await.unwrap();
    assert_eq!(rt.instances().await.len(), 1);
}

// The idle reaper auto-unloads a worker that has had no chat activity for longer than the
// configured TTL.
#[tokio::test]
async fn idle_worker_is_auto_unloaded_after_ttl() {
    use crate::node::test_support::fake_runtime_with_idle_ttl;
    let dir = TempDir::new().expect("staging dir");
    let model_dir = dir.path().join("org/m");
    std::fs::create_dir_all(&model_dir).unwrap();
    std::fs::write(model_dir.join("m.gguf"), b"GGUF\x00 dummy").unwrap();
    let rt = fake_runtime_with_idle_ttl(
        vec![dir.path().to_path_buf()],
        std::time::Duration::from_millis(120),
    );
    load(&rt, "org/m").await;
    assert_eq!(rt.worker_ids().await.len(), 1);
    // Poll past the TTL + reap ticks (robust to scheduling jitter under load).
    assert!(
        wait_reaped(&rt).await,
        "idle worker auto-unloaded after the TTL"
    );
}

// Turning auto-unload OFF at runtime stops the reaper; turning it back on (with a tiny TTL)
// lets it reap again — the live Server-Settings behavior.
#[tokio::test]
async fn runtime_idle_config_toggles_take_effect_live() {
    use crate::node::test_support::fake_runtime_with_idle_ttl;
    let dir = TempDir::new().expect("staging dir");
    let model_dir = dir.path().join("org/m");
    std::fs::create_dir_all(&model_dir).unwrap();
    std::fs::write(model_dir.join("m.gguf"), b"GGUF\x00 dummy").unwrap();
    // Start with a tiny TTL but auto-unload DISABLED — the worker must survive.
    let rt = fake_runtime_with_idle_ttl(
        vec![dir.path().to_path_buf()],
        std::time::Duration::from_millis(120),
    );
    rt.idle().set_enabled(false);
    load(&rt, "org/m").await;
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    assert_eq!(
        rt.worker_ids().await.len(),
        1,
        "auto-unload disabled keeps the worker resident"
    );
    // Re-enable: now the idle worker is reaped.
    rt.idle().set_enabled(true);
    assert!(
        wait_reaped(&rt).await,
        "re-enabling auto-unload reaps the idle worker"
    );
}

// A chat in flight (the relay holds a ChatLease) keeps a worker resident past the TTL; once
// the lease drops, the idle clock restarts from the end of the chat and it is reaped.
#[tokio::test]
async fn in_flight_chat_prevents_idle_reap() {
    use crate::node::test_support::fake_runtime_with_idle_ttl;
    let dir = TempDir::new().expect("staging dir");
    let model_dir = dir.path().join("org/m");
    std::fs::create_dir_all(&model_dir).unwrap();
    std::fs::write(model_dir.join("m.gguf"), b"GGUF\x00 dummy").unwrap();
    let rt = fake_runtime_with_idle_ttl(
        vec![dir.path().to_path_buf()],
        std::time::Duration::from_millis(120),
    );
    let (id, _) = load(&rt, "org/m").await;
    // Hold a chat lease across the whole idle window — the worker must NOT be reaped.
    let lease = rt.chat_handle(id).await.expect("lease");
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    assert_eq!(
        rt.worker_ids().await.len(),
        1,
        "in-flight chat keeps the worker resident past the TTL"
    );
    // End the chat; now the idle clock runs and the worker is reaped.
    drop(lease);
    assert!(
        wait_reaped(&rt).await,
        "worker reaped once the chat ended and it went idle"
    );
}

// After shutdown_all, the registry is empty so per-worker ops report "no worker"
// (the actor itself is still alive — rt holds the handle).
#[tokio::test]
async fn ops_after_shutdown_report_no_worker() {
    let (rt, _dir) = fake_runtime_with_models(&["org/m"]);
    let (id, _) = load(&rt, "org/m").await;
    rt.shutdown_all().await;
    assert!(rt.status(id).await.is_err(), "stopped worker is gone");
    assert!(rt.unload(id).await.is_err(), "nothing left to unload");
}

// ── A `NodeRuntime` backed by the STATEFUL fake worker. ────────────────────────────

/// A stateful-fake-backed runtime staging a dummy GGUF per `id`, so the real load path
/// runs and `M_STATUS`/`M_UNLOAD`/`M_CHAT` behave like a (model-less) real worker.
fn stateful_runtime_with_models(ids: &[&str]) -> (NodeRuntime, TempDir) {
    use crate::node::test_support::fake_runtime_stateful;
    let dir = TempDir::new().expect("staging dir");
    for id in ids {
        let model_dir = dir.path().join(id);
        std::fs::create_dir_all(&model_dir).expect("model dir");
        std::fs::write(model_dir.join("m.gguf"), b"GGUF\x00 dummy").expect("write dummy gguf");
    }
    let rt = fake_runtime_stateful(vec![dir.path().to_path_buf()]);
    (rt, dir)
}

// The stateful fake records the model on M_LOAD and reports it back under `loaded` on
// M_STATUS — so `status()` round-trips the loaded model + its ctx_len/gpu_layers/threads.
#[tokio::test]
async fn stateful_status_reports_the_loaded_model() {
    let (rt, _dir) = stateful_runtime_with_models(&["org/stateful"]);
    let id = rt
        .load(NodeLoadParams {
            id: "org/stateful".into(),
            ctx_len: Some(2048),
            gpu_layers: Some(GpuLayers::Count { n: 0 }),
            threads: Some(4),
            params: None,
        })
        .await
        .expect("load")
        .0;
    let status = rt.status(id).await.expect("status");
    let loaded = &status["loaded"];
    assert_eq!(
        loaded["id"], "org/stateful",
        "status echoes the loaded model"
    );
    assert_eq!(loaded["ctx_len"], 2048, "status echoes ctx_len");
    assert_eq!(loaded["threads"], 4, "status echoes threads");
}

// Driving M_CHAT through a `ChatLease` exercises the stateful fake's chat branch (it
// streams `he`/`llo` then a final response with token counts). The lease's Deref hands the
// underlying Supervisor straight through, and dropping it posts ChatEnd.
#[tokio::test]
async fn chat_lease_drives_the_stateful_chat_path() {
    use crate::worker::M_CHAT;
    let (rt, _dir) = stateful_runtime_with_models(&["org/chat"]);
    let (id, _) = rt.load(load_params("org/chat")).await.expect("load");
    let lease = rt.chat_handle(id).await.expect("lease");
    // Deref the lease to the Supervisor and run one chat — the fake streams chunks then
    // a final response carrying the content + token counts.
    let resp = lease
        .request(
            M_CHAT,
            serde_json::json!({ "request_id": 1, "messages": [] }),
        )
        .await
        .expect("chat response");
    assert_eq!(
        resp["content"], "hello",
        "stateful fake returns the chat content"
    );
    assert_eq!(resp["completion_tokens"], 3, "token counts propagate");
    // Dropping the lease ends the chat (ChatEnd) without panicking; the worker stays
    // resident (default 60-min TTL).
    drop(lease);
    assert_eq!(
        rt.worker_ids().await.len(),
        1,
        "worker still resident after chat"
    );
}

// A worker whose M_STATUS always errors (busy mid-generation) still LOADS — so it is
// resident — but `status()` surfaces the worker's RPC error rather than a stub. This is the
// node-level view of the "resident-but-unreachable" condition (`fake_status_failing_factory`).
#[tokio::test]
async fn status_failing_worker_loads_but_status_errors() {
    use crate::node::test_support::fake_runtime_status_fails;
    let dir = TempDir::new().expect("staging dir");
    let model_dir = dir.path().join("org/busy");
    std::fs::create_dir_all(&model_dir).unwrap();
    std::fs::write(model_dir.join("m.gguf"), b"GGUF\x00 dummy").unwrap();
    let rt = fake_runtime_status_fails(vec![dir.path().to_path_buf()]);
    // M_LOAD echoes the id → the worker is recorded as resident.
    let (id, loaded) = rt.load(load_params("org/busy")).await.expect("load ok");
    assert_eq!(loaded["id"], "org/busy", "load echoes the served id");
    assert_eq!(rt.worker_ids().await.len(), 1, "worker is resident");
    // But the status probe errors (worker busy): the node forwards the worker's failure.
    assert!(
        rt.status(id).await.is_err(),
        "status probe surfaces the worker error"
    );
    // It is still listed in instances/inventory despite the failing probe.
    assert_eq!(
        rt.instances().await.len(),
        1,
        "resident despite status failure"
    );
}

// Multiple DISTINCT models can be resident at once, each on its own worker with its own id;
// instances() lists every (worker, model) pair and unloading one leaves the rest.
#[tokio::test]
async fn multiple_distinct_models_are_concurrently_resident() {
    let (rt, _dir) = fake_runtime_with_models(&["org/a", "org/b", "org/c"]);
    let (wa, _) = load(&rt, "org/a").await;
    let (_wb, _) = load(&rt, "org/b").await;
    let (_wc, _) = load(&rt, "org/c").await;
    let mut insts = rt.instances().await;
    insts.sort_by(|x, y| x.1.cmp(&y.1));
    assert_eq!(insts.len(), 3, "three resident workers");
    let models: Vec<&str> = insts.iter().map(|(_, m)| m.as_str()).collect();
    assert_eq!(
        models,
        vec!["org/a", "org/b", "org/c"],
        "each worker its own model"
    );
    // Unload the middle model: the other two stay resident.
    rt.unload(wa).await.expect("unload a");
    assert_eq!(rt.worker_ids().await.len(), 2, "the rest remain resident");
}

// A successful load that COMMITS during a shutdown drain is reaped (not kept) and the load
// is refused — the terminal-shutdown guard in `LoadCommit`'s Ok branch. Driven by racing many
// success loads against shutdown_all so some resolve mid-drain.
#[tokio::test]
async fn successful_load_during_shutdown_is_reaped_and_refused() {
    let (rt, _dir) = fake_runtime_with_models(&["org/a", "org/b", "org/c", "org/d"]);
    // Fire several loads concurrently with the drain so at least one commits after
    // `shutting_down` is set (reaped + refused) rather than before it.
    let (_a, _b, _c, _d, _s) = tokio::join!(
        rt.load(load_params("org/a")),
        rt.load(load_params("org/b")),
        rt.load(load_params("org/c")),
        rt.load(load_params("org/d")),
        rt.shutdown_all(),
    );
    // Whatever the interleaving, the drain leaves NOTHING committed: a load that resolved
    // before the drain was reaped by `drain()`, one that resolved during it by the Ok+
    // shutting_down branch.
    assert!(
        rt.worker_ids().await.is_empty(),
        "no worker survives the drain regardless of commit timing"
    );
}

// A chat lease held across an UNLOAD: when the lease drops it posts ChatEnd for a worker the
// registry no longer has, so the actor drops its stale bookkeeping (the `else` arm of
// ChatEnd) instead of leaking it. Must not panic and must leave an empty registry.
#[tokio::test]
async fn chat_end_for_unloaded_worker_drops_bookkeeping() {
    let (rt, _dir) = fake_runtime_with_models(&["org/m"]);
    let (id, _) = load(&rt, "org/m").await;
    let lease = rt.chat_handle(id).await.expect("lease");
    // Unload while the chat lease is still held — the worker leaves the registry.
    rt.unload(id).await.expect("unload");
    assert!(rt.worker_ids().await.is_empty(), "worker gone after unload");
    // Dropping the lease now posts ChatEnd for a worker that is no longer resident; the
    // actor takes the `forget_activity` branch. Give the message a moment to drain.
    drop(lease);
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    assert!(
        rt.worker_ids().await.is_empty(),
        "registry stays empty; stale ChatEnd dropped its bookkeeping"
    );
    // The actor is still healthy after the stale ChatEnd — a fresh op succeeds.
    assert!(
        rt.scan().await.is_ok(),
        "actor still serving after stale ChatEnd"
    );
}

// resolve_model's symlink-escape guard ([HG015]): a cataloged GGUF whose path canonicalizes
// OUTSIDE every configured scan root is rejected as InvalidModelId, with no worker spawned.
#[tokio::test]
async fn load_rejects_a_model_symlinked_outside_the_roots() {
    // The scan root holds a model dir, but its `m.gguf` is a symlink to a file OUTSIDE the
    // root — so it catalogs (the symlink is under the root) yet canonicalizes to escape.
    let root = TempDir::new().expect("scan root");
    let outside = TempDir::new().expect("outside dir");
    let target = outside.path().join("real.gguf");
    std::fs::write(&target, b"GGUF\x00 dummy").expect("write outside gguf");
    let model_dir = root.path().join("org/escape");
    std::fs::create_dir_all(&model_dir).expect("model dir");
    std::os::unix::fs::symlink(&target, model_dir.join("m.gguf")).expect("symlink gguf");

    let rt = fake_runtime_with_dirs(vec![root.path().to_path_buf()]);
    let err = rt.load(load_params("org/escape")).await.unwrap_err();
    assert!(
        err.to_string().contains("[HG015]") || err.to_string().contains("outside"),
        "symlink-escape rejected as InvalidModelId: {err}"
    );
    assert!(
        rt.worker_ids().await.is_empty(),
        "no worker spawned for an escaping path"
    );
}

// worker_load_params merges EVERY non-base, non-null override key into the M_LOAD object
// (the `obj.insert(k, v)` merge loop), while base fields stay authoritative and a null is
// skipped. Several overrides at once exercises the loop body, not just one branch.
#[test]
fn worker_load_params_merges_all_rich_overrides() {
    use crate::worker::engine::llamacpp::params::LlamaCppParams;
    use crate::worker::engine::{FlashAttn, KvCacheKind};
    let mut rich = LlamaCppParams::base(CtxLen::Auto, GpuLayers::Count { n: 10 }, 8);
    rich.flash_attn = Some(FlashAttn::Off);
    rich.type_k = Some(KvCacheKind::Q8_0);
    rich.type_v = Some(KvCacheKind::Q8_0);
    rich.cpu_moe = Some(false);
    rich.use_mmap = Some(true);
    let v = worker_load_params(
        "org/m",
        "/x.gguf",
        Some(4096),
        Some(GpuLayers::Count { n: 10 }),
        Some(8),
        &Some(rich),
    );
    let obj = v.as_object().expect("object");
    // Base fields (authoritative) present from the explicit args.
    assert_eq!(obj["ctx_len"], 4096);
    assert_eq!(
        obj["gpu_layers"],
        serde_json::json!({ "kind": "count", "n": 10 })
    );
    assert_eq!(obj["threads"], 8);
    // Every rich override merged in.
    assert_eq!(obj["flash_attn"], "off");
    assert_eq!(obj["type_k"], "Q8_0");
    assert_eq!(obj["type_v"], "Q8_0");
    assert_eq!(obj["cpu_moe"], false);
    assert_eq!(obj["use_mmap"], true);
}

// The idle reaper task exits cleanly once the runtime is dropped: with a tiny TTL the reaper
// ticks fast, and after the last handle drops its WeakHandle upgrade fails so the loop breaks.
// We can only observe this indirectly (no panic / no hang); the value is exercising the
// reaper's drop-exit branch with a fast cadence.
#[tokio::test]
async fn dropping_runtime_stops_the_idle_reaper() {
    use crate::node::test_support::fake_runtime_with_idle_ttl;
    let dir = TempDir::new().expect("staging dir");
    let model_dir = dir.path().join("org/m");
    std::fs::create_dir_all(&model_dir).unwrap();
    std::fs::write(model_dir.join("m.gguf"), b"GGUF\x00 dummy").unwrap();
    let rt = fake_runtime_with_idle_ttl(
        vec![dir.path().to_path_buf()],
        std::time::Duration::from_millis(50),
    );
    load(&rt, "org/m").await;
    // Drop the runtime: shutdown_all first so the worker is drained, then drop the handle so
    // the reaper's next tick sees a dead WeakHandle and exits.
    rt.shutdown_all().await;
    drop(rt);
    // Let several reap intervals elapse so the reaper definitely ticks and breaks.
    tokio::time::sleep(std::time::Duration::from_millis(150)).await;
    // Nothing to assert beyond "no panic / no hang" — the test passing means the reaper
    // loop broke instead of spinning against a dead handle.
}

// `worker_ids`/`instances`/`gpus` return EMPTY after the actor mailbox is gone. The mailbox
// only closes once the last `Handle` drops — which the wrapper holds — so via the public
// surface these graceful-empty fallbacks are not reachable without dropping the wrapper's
// handle. Covered here only to the extent the happy path exercises them (above); the
// actor-gone fallbacks are reported unreachable.
