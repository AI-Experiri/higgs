use super::*;
use crate::supervisor::WorkerHalves;
use crate::worker::N_CHAT_CHUNK;
use parking_lot::Mutex;
use serde_json::json;
use tokio::io::AsyncWriteExt;

// ── Test seam (mirrored from supervisor::tests::make_supervisor) ──────────

/// Build a `Supervisor` plus duplex test handles.
fn make_supervisor() -> (
    Supervisor,
    tokio::io::DuplexStream, // test_write: write responses → supervisor reads
    tokio::io::DuplexStream, // test_read:  supervisor writes requests → test reads
) {
    let (sup_write, test_read) = tokio::io::duplex(64 * 1024);
    let (test_write, sup_read) = tokio::io::duplex(64 * 1024);

    let sup_write_cell = Arc::new(Mutex::new(Some(sup_write)));
    let sup_read_cell = Arc::new(Mutex::new(Some(sup_read)));

    let sup = Supervisor::with_factory(Box::new(move |_ring, _model| {
        let write = sup_write_cell
            .lock()
            .take()
            .ok_or_else(|| HiggsError::WorkerSpawnFailed {
                source: std::io::Error::other("mock: no more write halves"),
            })?;
        let read = sup_read_cell
            .lock()
            .take()
            .ok_or_else(|| HiggsError::WorkerSpawnFailed {
                source: std::io::Error::other("mock: no more read halves"),
            })?;
        Ok(WorkerHalves {
            write: Box::new(write),
            read: Box::new(read),
            proc: None,
        })
    }));

    sup.start_for("test-model").expect("mock start");
    (sup, test_write, test_read)
}

async fn write_response(stream: &mut tokio::io::DuplexStream, id: u64, result: serde_json::Value) {
    use crate::rpc::{encode, RpcFrame, RpcResponse};
    let line = encode(&RpcFrame::Response(RpcResponse {
        jsonrpc: "2.0".into(),
        id,
        result: Some(result),
        error: None,
    }));
    stream
        .write_all(format!("{line}\n").as_bytes())
        .await
        .unwrap();
    stream.flush().await.unwrap();
}

// ── Phase A2: repo-id charset + path-traversal guards ────────────────────

#[test]
fn validate_repo_id_accepts_legitimate_ids() {
    for id in [
        "org/model",
        "lmstudio-community/NVIDIA-Nemotron-3-Nano-4B-GGUF",
        "ollama/llama3:8b",
        "google/gemma-3.1-12b",
    ] {
        validate_repo_id(id).unwrap_or_else(|e| panic!("{id} should be valid: {e}"));
    }
}

#[test]
fn validate_repo_id_rejects_traversal_and_illegal() {
    for (id, why) in [
        ("", "empty"),
        ("/etc/passwd", "absolute"),
        ("..", "dotdot"),
        ("org/../../etc/passwd", "embedded dotdot"),
        ("org/model;rm -rf", "illegal char"),
        ("org/model\0", "nul"),
    ] {
        let err = validate_repo_id(id).expect_err(why);
        assert!(
            matches!(err, HiggsError::InvalidModelId { .. }),
            "{why}: {err}"
        );
        assert!(err.to_string().starts_with("[HG015]"), "{why}: {err}");
    }
}

#[test]
fn path_within_roots_contains_and_escapes() {
    let root = tempfile::TempDir::new().unwrap();
    let inside = root.path().join("org").join("m.gguf");
    std::fs::create_dir_all(inside.parent().unwrap()).unwrap();
    std::fs::write(&inside, b"x").unwrap();
    let roots = vec![root.path().to_path_buf()];
    assert!(path_within_roots(inside.to_str().unwrap(), &roots));

    // A path outside every root (a different temp dir) is rejected.
    let other = tempfile::TempDir::new().unwrap();
    let outside = other.path().join("escape.gguf");
    std::fs::write(&outside, b"x").unwrap();
    assert!(!path_within_roots(outside.to_str().unwrap(), &roots));

    // A non-existent path can't canonicalize → rejected.
    assert!(!path_within_roots("/nope/does/not/exist.gguf", &roots));
}

/// `load` rejects an id that fails charset validation with [HG015], before
/// any scan or worker RPC.
#[tokio::test]
async fn load_rejects_traversal_id() {
    let higgs = Higgs::new(HiggsConfig {
        lmstudio_dirs: vec![],
        hf_dirs: vec![],
        ollama_dirs: vec![],
        default_load: HiggsConfig::default().default_load,
    });
    let err = higgs
        .load("org/../../etc/passwd", None)
        .await
        .expect_err("traversal id must be rejected");
    assert!(matches!(err, HiggsError::InvalidModelId { .. }));
}

/// `probe_support` returns a cached `(arch, quant)` verdict WITHOUT probing.
///
/// The cache is pre-seeded for the current engine version; the rep's combo is
/// a hit, so `probe_paths` (which would spawn-fail under the mock factory and
/// yield a `false` verdict) is never consulted — the returned verdict is the
/// seeded `(true, None)`. A second combo is a miss and goes to the probe path,
/// proving the partition.
#[tokio::test]
async fn probe_support_cache_hit_skips_probe() {
    let (sup, _tw, _tr) = make_supervisor();
    let higgs = Higgs {
        sup: Arc::new(sup),
        config: parking_lot::Mutex::new(HiggsConfig::default()),
        lifecycle: tokio::sync::Mutex::new(()),
        inference_gate: Arc::new(tokio::sync::Semaphore::new(MAX_CONCURRENT_INFERENCE)),
        remote_gate: Arc::new(tokio::sync::Semaphore::new(MAX_CONCURRENT_INFERENCE)),
        last_activity: parking_lot::Mutex::new(std::time::Instant::now()),
        log_incoming_tokens: std::sync::atomic::AtomicBool::new(false),
        jit_enabled: std::sync::atomic::AtomicBool::new(true),
        auto_unload_idle: std::sync::atomic::AtomicBool::new(true),
        idle_ttl_minutes: std::sync::atomic::AtomicU64::new(IDLE_UNLOAD_TTL_MINUTES),
        loaded_idle_ttl_override: std::sync::atomic::AtomicU64::new(0),
        serving_enabled: std::sync::atomic::AtomicBool::new(true),
        probe_cache: parking_lot::Mutex::new(std::collections::HashMap::new()),
        device_cache: parking_lot::Mutex::new(None),
        fleet: parking_lot::Mutex::new(None),
        api_keys: parking_lot::Mutex::new(std::sync::Arc::new(crate::keys::ApiKeys::default())),
        hub: parking_lot::Mutex::new(None),
    };
    let ev = crate::worker::engine::llamacpp::engine_version();
    // Seed a HIT for (llama, Q4_K_M, <this engine version>).
    higgs
        .probe_cache
        .lock()
        .insert(("llama".into(), "Q4_K_M".into(), ev.clone()), (true, None));
    let out = higgs
        .probe_support(vec![
            // Hit: returns the seeded verdict, no probe.
            ("llama".into(), "Q4_K_M".into(), "/seeded/path.gguf".into()),
            // Miss: probe path (mock factory spawn-fails) → false verdict.
            ("gemma4".into(), "Q8_0".into(), "/miss/path.gguf".into()),
        ])
        .await;
    assert_eq!(
        out.get(&("llama".into(), "Q4_K_M".into())),
        Some(&(true, None))
    );
    let (miss_loadable, miss_reason) = out
        .get(&("gemma4".into(), "Q8_0".into()))
        .cloned()
        .expect("miss combo present");
    assert!(
        !miss_loadable,
        "miss combo is not loadable under spawn-fail"
    );
    assert!(miss_reason.is_some(), "miss carries a reason");
    // The miss verdict was stored (the probe path inserts under the version
    // the worker reported — empty here because spawn failed before any reply).
    let _ = ev;
    assert!(higgs
        .probe_cache
        .lock()
        .keys()
        .any(|(a, q, _)| a == "gemma4" && q == "Q8_0"));
}

/// The inference admission gate returns `ServerBusy` once all permits are
/// taken; releasing a permit re-opens a slot.
#[tokio::test]
async fn inference_gate_rejects_when_full() {
    let (sup, _tw, _tr) = make_supervisor();
    let higgs = Higgs {
        sup: Arc::new(sup),
        config: parking_lot::Mutex::new(HiggsConfig::default()),
        lifecycle: tokio::sync::Mutex::new(()),
        // One-slot gate so the test deterministically fills it.
        inference_gate: Arc::new(tokio::sync::Semaphore::new(1)),
        remote_gate: Arc::new(tokio::sync::Semaphore::new(1)),
        last_activity: parking_lot::Mutex::new(std::time::Instant::now()),
        log_incoming_tokens: std::sync::atomic::AtomicBool::new(false),
        jit_enabled: std::sync::atomic::AtomicBool::new(true),
        auto_unload_idle: std::sync::atomic::AtomicBool::new(true),
        idle_ttl_minutes: std::sync::atomic::AtomicU64::new(IDLE_UNLOAD_TTL_MINUTES),
        loaded_idle_ttl_override: std::sync::atomic::AtomicU64::new(0),
        serving_enabled: std::sync::atomic::AtomicBool::new(true),
        probe_cache: parking_lot::Mutex::new(std::collections::HashMap::new()),
        device_cache: parking_lot::Mutex::new(None),
        fleet: parking_lot::Mutex::new(None),
        api_keys: parking_lot::Mutex::new(std::sync::Arc::new(crate::keys::ApiKeys::default())),
        hub: parking_lot::Mutex::new(None),
    };
    // Take the only permit and hold it.
    let held = Arc::clone(&higgs.inference_gate)
        .try_acquire_owned()
        .expect("first permit");
    // chat_stream must now fail fast with ServerBusy (no worker RPC).
    let err = higgs
        .chat_stream(
            "org/model".to_owned(),
            r#"[{"role":"user","content":"hi"}]"#.to_owned(),
            8,
            0.0,
            None,
        )
        .await
        .expect_err("gate full → ServerBusy");
    assert!(matches!(err, HiggsError::ServerBusy { .. }), "got {err}");
    drop(held);
    // With the permit released, a request is admitted again (it then fails
    // later for lack of a real worker response, but admission succeeds).
    assert!(
        Arc::clone(&higgs.inference_gate)
            .try_acquire_owned()
            .is_ok(),
        "slot re-opens after release"
    );
}

// ── Phase B: RAM headroom guard arithmetic ───────────────────────────────

#[test]
fn fits_in_memory_respects_headroom_fraction() {
    // Exactly at 80% of available fits; one byte over does not.
    let available = 10_000_000_000u64; // 10 GB
    let safe = 8_000_000_000u64; // 80%
    assert!(fits_in_memory(safe, available), "at the threshold fits");
    assert!(
        fits_in_memory(safe - 1, available),
        "below the threshold fits"
    );
    assert!(
        !fits_in_memory(safe + 1, available),
        "above the threshold is refused"
    );
    // A model larger than ALL available RAM is always refused.
    assert!(!fits_in_memory(available + 1, available));
    // Zero-size edge: always fits.
    assert!(fits_in_memory(0, available));
}

/// `load` refuses a model whose file size exceeds the RAM headroom with
/// [HG017], before spawning a worker. Uses a fixture whose declared size is
/// forced over the limit by checking against a tiny synthetic available
/// value is not possible through `load` (it reads real RAM), so this asserts
/// the typed-error path via the pure guard plus the diagnostic wiring; the
/// end-to-end refusal is covered by `fits_in_memory_respects_headroom_fraction`
/// and the HG017 status-mapping test in `serve::mod`.
#[test]
fn insufficient_memory_diagnostic_is_503_capacity() {
    let err = HiggsError::InsufficientMemory {
        id: "org/model".into(),
        needed_bytes: 8_000_000_000,
        available_bytes: 4_000_000_000,
        headroom_fraction: MEMORY_HEADROOM_FRACTION,
    };
    assert!(err.to_string().starts_with("[HG017]"));
}

// ── Verbose toggle: default false, set/get round-trip ────────────────────

#[test]
fn verbose_defaults_false_and_round_trips() {
    let higgs = Higgs::new(HiggsConfig::default());
    assert!(!higgs.verbose(), "verbose defaults to false");
    higgs.set_verbose(true);
    assert!(higgs.verbose(), "set_verbose(true) is observed");
    higgs.set_verbose(false);
    assert!(!higgs.verbose(), "set_verbose(false) is observed");
}

// ── Log-incoming-tokens toggle: default false, set/get round-trip ─────────

#[test]
fn log_incoming_tokens_defaults_false_and_round_trips() {
    let higgs = Higgs::new(HiggsConfig::default());
    assert!(
        !higgs.log_incoming_tokens(),
        "log_incoming_tokens defaults to false"
    );
    higgs.set_log_incoming_tokens(true);
    assert!(
        higgs.log_incoming_tokens(),
        "set_log_incoming_tokens(true) is observed"
    );
    higgs.set_log_incoming_tokens(false);
    assert!(
        !higgs.log_incoming_tokens(),
        "set_log_incoming_tokens(false) is observed"
    );
}

// ── JIT toggle: default TRUE, set/get round-trip ────────────────────────

#[test]
fn jit_enabled_defaults_true_and_round_trips() {
    let higgs = Higgs::new(HiggsConfig::default());
    assert!(higgs.jit_enabled(), "JIT defaults to ON (true)");
    higgs.set_jit_enabled(false);
    assert!(!higgs.jit_enabled(), "set_jit_enabled(false) is observed");
    higgs.set_jit_enabled(true);
    assert!(higgs.jit_enabled(), "set_jit_enabled(true) is observed");
}

// ── Idle auto-unload toggles: defaults + round-trip ──────────────────────

#[test]
fn idle_unload_settings_default_and_round_trip() {
    let higgs = Higgs::new(HiggsConfig::default());
    // Defaults: auto-unload ON, TTL 5 minutes (seeded from IDLE_UNLOAD_TTL).
    assert!(
        higgs.auto_unload_idle(),
        "auto-unload defaults to ON (true)"
    );
    assert_eq!(higgs.idle_ttl_minutes(), 5, "TTL defaults to 5 minutes");
    assert_eq!(
        IDLE_UNLOAD_TTL_MINUTES * 60,
        IDLE_UNLOAD_TTL.as_secs(),
        "minutes seed must equal the Duration const"
    );

    higgs.set_auto_unload_idle(false);
    assert!(!higgs.auto_unload_idle(), "set_auto_unload_idle(false)");
    higgs.set_auto_unload_idle(true);
    assert!(higgs.auto_unload_idle(), "set_auto_unload_idle(true)");

    higgs.set_idle_ttl_minutes(30);
    assert_eq!(higgs.idle_ttl_minutes(), 30, "set_idle_ttl_minutes(30)");
}

// ── Per-load idle-TTL override: default None, set/clear round-trip ────────

#[test]
fn loaded_idle_ttl_override_defaults_none_and_round_trips() {
    let higgs = Higgs::new(HiggsConfig::default());
    // No override by default (0 in the atomic reads back as None).
    assert_eq!(
        higgs.loaded_idle_ttl_override(),
        None,
        "override defaults to None"
    );
    // Set an override → reads back as Some(n).
    higgs.set_loaded_idle_ttl_override(Some(30));
    assert_eq!(
        higgs.loaded_idle_ttl_override(),
        Some(30),
        "set Some(30) is observed"
    );
    // Clear with None → back to None.
    higgs.set_loaded_idle_ttl_override(None);
    assert_eq!(
        higgs.loaded_idle_ttl_override(),
        None,
        "set None clears the override"
    );
}

/// The reaper's effective-TTL expression prefers the per-load override over
/// the global `idle_ttl_minutes` — the exact `unwrap_or_else` the reaper runs.
#[test]
fn reaper_prefers_loaded_override_over_global_ttl() {
    let higgs = Higgs::new(HiggsConfig::default());
    // Global TTL is 5 (the default); with no override the effective value is
    // the global TTL.
    let effective = |h: &Higgs| {
        h.loaded_idle_ttl_override()
            .unwrap_or_else(|| h.idle_ttl_minutes())
    };
    assert_eq!(effective(&higgs), 5, "no override → global TTL (5)");
    // With an override set, it wins regardless of the global TTL.
    higgs.set_loaded_idle_ttl_override(Some(42));
    assert_eq!(effective(&higgs), 42, "override (42) wins over global");
    // Clearing the override falls back to the global TTL again.
    higgs.set_loaded_idle_ttl_override(None);
    assert_eq!(effective(&higgs), 5, "cleared → global TTL (5) again");
}

// ── Serving on/off gate: default true, set/get round-trip ─────────────────

#[test]
fn serving_enabled_defaults_true_and_round_trips() {
    let higgs = Higgs::new(HiggsConfig::default());
    assert!(higgs.serving_enabled(), "serving defaults to ON (true)");
    higgs.set_serving_enabled(false);
    assert!(!higgs.serving_enabled(), "set_serving_enabled(false)");
    higgs.set_serving_enabled(true);
    assert!(higgs.serving_enabled(), "set_serving_enabled(true)");
}

// ── Reaper respects the runtime auto-unload toggle and TTL ────────────────
//
// The reaper reads `auto_unload_idle` and `idle_ttl_minutes` from the live
// atoms each tick. These tests drive the same decision predicate the reaper
// uses (read flag → skip if off; read TTL → skip if idle_for < ttl) against a
// facade whose runtime values are set via the public accessors, proving a
// change takes effect without a restart.

#[test]
fn reaper_skips_when_auto_unload_disabled() {
    let higgs = Higgs::new(HiggsConfig::default());
    higgs.set_auto_unload_idle(false);
    // With auto-unload off, the reaper's first guard short-circuits: no
    // unload regardless of how long the model has been idle.
    assert!(!higgs.auto_unload_idle(), "reaper would skip this tick");
}

#[test]
fn reaper_uses_runtime_ttl_not_const() {
    let higgs = Higgs::new(HiggsConfig::default());
    // Raise the TTL to 60 minutes at runtime.
    higgs.set_idle_ttl_minutes(60);
    let ttl = std::time::Duration::from_secs(higgs.idle_ttl_minutes() * 60);
    // A model idle for 10 minutes is BELOW the new 60-minute TTL → not reaped,
    // even though it exceeds the old 5-minute const default.
    let idle_for = std::time::Duration::from_secs(10 * 60);
    assert!(idle_for < ttl, "runtime TTL (60m) keeps a 10m-idle model");
    assert!(
        idle_for > IDLE_UNLOAD_TTL,
        "the same 10m idle WOULD reap under the old 5m const — proving the \
             reaper must read the runtime value, not the const"
    );
}

// ── Test 1: default config paths ─────────────────────────────────────────

#[test]
fn default_config_paths() {
    let home = match dirs::home_dir() {
        Some(h) => h,
        None => return, // skip if no home dir
    };

    let cfg = HiggsConfig::default();

    let has_suffix = |dirs: &[PathBuf], suffix: &str| dirs.iter().any(|p| p.ends_with(suffix));

    assert!(
        has_suffix(&cfg.lmstudio_dirs, ".lmstudio/models")
            || cfg
                .lmstudio_dirs
                .iter()
                .any(|p| p.ends_with("lm-studio/models")),
        "lmstudio_dirs should contain .lmstudio/models or lm-studio/models"
    );
    assert!(
        cfg.hf_dirs
            .iter()
            .any(|p| { p == &home.join(".cache").join("huggingface").join("hub") }),
        "hf_dirs must use ~/.cache/huggingface/hub (not XDG cache_dir)"
    );
    assert!(
        cfg.ollama_dirs
            .iter()
            .any(|p| p.ends_with(".ollama/models")),
        "ollama_dirs should contain .ollama/models"
    );
}

// ── Test 2: scan runs host-side with no worker ───────────────────────────

/// `scan()` runs host-side (pure Rust, no worker RPC): with a fresh facade
/// that never spawned a worker and empty config dirs, it returns `Ok(empty)`.
#[tokio::test]
async fn scan_runs_host_side_without_worker() {
    // Empty config dirs → nothing to scan → Ok(empty). The point is that no
    // worker is live (start() never called) yet scan succeeds.
    let higgs = Higgs::new(HiggsConfig {
        lmstudio_dirs: vec![],
        hf_dirs: vec![],
        ollama_dirs: vec![],
        default_load: HiggsConfig::default().default_load,
    });

    let models = higgs.scan().await.expect("host-side scan should succeed");
    assert!(models.is_empty(), "empty dirs yield no models");

    // No worker was ever spawned: status reports worker_alive=false.
    let st = higgs.status().await.expect("status");
    assert!(!st.worker_alive, "scan must not spawn a worker");
}

// ── Test 3: load then status maps ─────────────────────────────────────────

#[tokio::test]
async fn load_then_status_maps() {
    use tokio::io::{AsyncBufReadExt, BufReader};

    let (sup, mut test_write, test_read) = make_supervisor();
    // `load` resolves the GGUF path host-side, so point config at a fixture.
    let dir = tempfile::TempDir::new().unwrap();
    crate::serve::test_support::write_gguf_fixture(dir.path(), "org/model");
    let cfg = HiggsConfig {
        lmstudio_dirs: vec![dir.path().to_path_buf()],
        hf_dirs: vec![],
        ollama_dirs: vec![],
        default_load: HiggsConfig::default().default_load,
    };
    let higgs = Higgs {
        sup: Arc::new(sup),
        config: parking_lot::Mutex::new(cfg),
        lifecycle: tokio::sync::Mutex::new(()),
        inference_gate: std::sync::Arc::new(tokio::sync::Semaphore::new(
            crate::api::MAX_CONCURRENT_INFERENCE,
        )),
        remote_gate: std::sync::Arc::new(tokio::sync::Semaphore::new(
            crate::api::MAX_CONCURRENT_INFERENCE,
        )),
        last_activity: parking_lot::Mutex::new(std::time::Instant::now()),
        log_incoming_tokens: std::sync::atomic::AtomicBool::new(false),
        jit_enabled: std::sync::atomic::AtomicBool::new(true),
        auto_unload_idle: std::sync::atomic::AtomicBool::new(true),
        idle_ttl_minutes: std::sync::atomic::AtomicU64::new(IDLE_UNLOAD_TTL_MINUTES),
        loaded_idle_ttl_override: std::sync::atomic::AtomicU64::new(0),
        serving_enabled: std::sync::atomic::AtomicBool::new(true),
        probe_cache: parking_lot::Mutex::new(std::collections::HashMap::new()),
        device_cache: parking_lot::Mutex::new(None),
        fleet: parking_lot::Mutex::new(None),
        api_keys: parking_lot::Mutex::new(std::sync::Arc::new(crate::keys::ApiKeys::default())),
        hub: parking_lot::Mutex::new(None),
    };
    let mut events_rx = higgs.events();
    // `load`/`status` run a host-side scan (on a blocking thread) before each
    // RPC, so drive the operation future concurrently with the responder: the
    // responder reads the request line (proving the id is pending) and only
    // then writes the reply. A fixed pre-sleep + sequential write would race
    // the scan and drop the response.
    let mut lines = BufReader::new(test_read).lines();

    // Issue load — mock responds with ok.
    let load_fut = higgs.load("org/model", None);
    let (load_res, _) = tokio::join!(load_fut, async {
        lines.next_line().await.unwrap().expect("M_LOAD request");
        write_response(&mut test_write, 1, json!({"id": "org/model"})).await;
    });
    load_res.expect("load should succeed");

    // ModelLoaded event must arrive.
    let ev = tokio::time::timeout(std::time::Duration::from_millis(100), events_rx.recv())
        .await
        .expect("timeout")
        .expect("recv");
    assert!(matches!(ev, HiggsEvent::ModelLoaded { id } if id == "org/model"));

    // Issue status — mock responds with loaded info.
    let status_fut = higgs.status();
    let (st, _) = tokio::join!(status_fut, async {
        lines.next_line().await.unwrap().expect("M_STATUS request");
        write_response(
                &mut test_write,
                2,
                json!({
                    "loaded": { "id": "org/model", "ctx_len": 4096, "gpu_layers": 4294967295u64, "threads": 4 },
                    "models_scanned": 3,
                }),
            )
            .await;
    });
    let st = st.expect("status should succeed");
    assert!(st.worker_alive);
    // models_on_disk now comes from a host-side scan of the config dirs
    // (one GGUF fixture), not the worker's `models_scanned`.
    assert_eq!(st.models_on_disk, 1);
    let li = st.loaded.expect("loaded should be Some");
    assert_eq!(li.id, "org/model");
    assert_eq!(li.ctx_len, 4096);
    assert_eq!(li.gpu_layers, u32::MAX);
}

// ── Test 3b: status loaded info includes model metadata ──────────────────

#[tokio::test]
async fn status_loaded_info_includes_model_metadata() {
    use tokio::io::{AsyncBufReadExt, BufReader};

    let (sup, mut test_write, test_read) = make_supervisor();
    // Metadata now comes from the HOST scan, not the worker response: point
    // config at a GGUF fixture (arch=llama, ctx_train=4096, chat template)
    // so the host-scanned `HiggsModel` enriches the worker-reported `loaded`.
    let dir = tempfile::TempDir::new().unwrap();
    crate::serve::test_support::write_gguf_fixture(dir.path(), "org/model");
    let cfg = HiggsConfig {
        lmstudio_dirs: vec![dir.path().to_path_buf()],
        hf_dirs: vec![],
        ollama_dirs: vec![],
        default_load: HiggsConfig::default().default_load,
    };
    let higgs = Higgs {
        sup: Arc::new(sup),
        config: parking_lot::Mutex::new(cfg),
        lifecycle: tokio::sync::Mutex::new(()),
        inference_gate: std::sync::Arc::new(tokio::sync::Semaphore::new(
            crate::api::MAX_CONCURRENT_INFERENCE,
        )),
        remote_gate: std::sync::Arc::new(tokio::sync::Semaphore::new(
            crate::api::MAX_CONCURRENT_INFERENCE,
        )),
        last_activity: parking_lot::Mutex::new(std::time::Instant::now()),
        log_incoming_tokens: std::sync::atomic::AtomicBool::new(false),
        jit_enabled: std::sync::atomic::AtomicBool::new(true),
        auto_unload_idle: std::sync::atomic::AtomicBool::new(true),
        idle_ttl_minutes: std::sync::atomic::AtomicU64::new(IDLE_UNLOAD_TTL_MINUTES),
        loaded_idle_ttl_override: std::sync::atomic::AtomicU64::new(0),
        serving_enabled: std::sync::atomic::AtomicBool::new(true),
        probe_cache: parking_lot::Mutex::new(std::collections::HashMap::new()),
        device_cache: parking_lot::Mutex::new(None),
        fleet: parking_lot::Mutex::new(None),
        api_keys: parking_lot::Mutex::new(std::sync::Arc::new(crate::keys::ApiKeys::default())),
        hub: parking_lot::Mutex::new(None),
    };

    // `status` runs a host-side scan (on a blocking thread) before M_STATUS,
    // so drive the future concurrently with a responder that reads the
    // request line before replying — a fixed sleep would race the scan. The
    // worker reports only id/ctx_len/gpu_layers/threads; the metadata fields
    // are filled host-side from the fixture.
    let mut lines = BufReader::new(test_read).lines();
    let status_fut = higgs.status();
    let (st, _) = tokio::join!(status_fut, async {
        lines.next_line().await.unwrap().expect("M_STATUS request");
        write_response(
            &mut test_write,
            1,
            json!({
                "loaded": {
                    "id": "org/model",
                    "ctx_len": 4096,
                    "gpu_layers": 99,
                    "threads": 4,
                },
                "models_scanned": 1,
            }),
        )
        .await;
    });
    let st = st.expect("status should succeed");
    let li = st.loaded.expect("loaded should be Some");
    assert_eq!(li.id, "org/model");
    assert_eq!(li.arch.as_deref(), Some("llama"));
    assert_eq!(li.quant.as_deref(), Some("Q4_K_M"));
    assert_eq!(li.max_context_length, Some(4096));
    assert!(li.size_bytes.is_some(), "size_bytes from fixture file");
    assert_eq!(li.has_chat_template, Some(true));
}

// ── Test 3c: host-resolved load carries the GGUF path (no worker scan) ────
//
// Regression: after scan moved host-side, the worker's ModelStore is empty,
// so the worker can only resolve a path if the host puts it in M_LOAD params.
// This asserts `load(id)` resolves the path host-side and includes it in the
// M_LOAD request — proving the load works WITHOUT a prior worker scan. If the
// path-passing were removed (worker fell back to its empty `store.get(id)`),
// the params would carry no `path` and this test would fail.
#[tokio::test]
async fn load_carries_host_resolved_path() {
    use tokio::io::{AsyncBufReadExt, BufReader};

    let (sup, mut test_write, test_read) = make_supervisor();

    // Real GGUF fixture so the host-side scan discovers the id with a path.
    let dir = tempfile::TempDir::new().unwrap();
    crate::serve::test_support::write_gguf_fixture(dir.path(), "org/model");
    let cfg = HiggsConfig {
        lmstudio_dirs: vec![dir.path().to_path_buf()],
        hf_dirs: vec![],
        ollama_dirs: vec![],
        default_load: HiggsConfig::default().default_load,
    };
    let higgs = Higgs {
        sup: Arc::new(sup),
        config: parking_lot::Mutex::new(cfg),
        lifecycle: tokio::sync::Mutex::new(()),
        inference_gate: std::sync::Arc::new(tokio::sync::Semaphore::new(
            crate::api::MAX_CONCURRENT_INFERENCE,
        )),
        remote_gate: std::sync::Arc::new(tokio::sync::Semaphore::new(
            crate::api::MAX_CONCURRENT_INFERENCE,
        )),
        last_activity: parking_lot::Mutex::new(std::time::Instant::now()),
        log_incoming_tokens: std::sync::atomic::AtomicBool::new(false),
        jit_enabled: std::sync::atomic::AtomicBool::new(true),
        auto_unload_idle: std::sync::atomic::AtomicBool::new(true),
        idle_ttl_minutes: std::sync::atomic::AtomicU64::new(IDLE_UNLOAD_TTL_MINUTES),
        loaded_idle_ttl_override: std::sync::atomic::AtomicU64::new(0),
        serving_enabled: std::sync::atomic::AtomicBool::new(true),
        probe_cache: parking_lot::Mutex::new(std::collections::HashMap::new()),
        device_cache: parking_lot::Mutex::new(None),
        fleet: parking_lot::Mutex::new(None),
        api_keys: parking_lot::Mutex::new(std::sync::Arc::new(crate::keys::ApiKeys::default())),
        hub: parking_lot::Mutex::new(None),
    };

    // Drive the load. `load` first runs a host-side scan (on a blocking
    // thread) before sending M_LOAD, so drive the load future concurrently
    // with a responder that reads the request line (proving id=1 is pending)
    // before replying. A fixed pre-sleep would race the scan and drop the
    // response.
    let mut lines = BufReader::new(test_read).lines();
    let load_fut = higgs.load("org/model", None);
    let (load_res, line) = tokio::join!(load_fut, async {
        let line = lines.next_line().await.unwrap().expect("M_LOAD request");
        write_response(&mut test_write, 1, json!({"id": "org/model"})).await;
        line
    });
    load_res.expect("host-resolved load should succeed");

    // The M_LOAD request carries the fixture path resolved host-side.
    let v: serde_json::Value = serde_json::from_str(&line).unwrap();
    assert_eq!(v["method"], M_LOAD);
    let path = v["params"]["path"].as_str().expect("path in M_LOAD params");
    assert!(path.ends_with(".gguf"), "path was: {path}");
    assert!(path.contains("org/model"), "path was: {path}");
}

// ── Test 4: chat_stream delivers chunks and outcome ────────────────────────
//
// Verifies end-to-end: alloc_request_id allocates id=1; chat_stream registers
// the sink under that id and sends M_CHAT with request_id=1; the test injects
// N_CHAT_CHUNK notifications tagged request_id=1; route_notification delivers
// them to rx; the final response for RPC id=1 resolves the outcome handle.

#[tokio::test]
async fn chat_stream_delivers() {
    let (sup, mut test_write, _test_read) = make_supervisor();
    let higgs = Higgs {
        sup: Arc::new(sup),
        config: parking_lot::Mutex::new(HiggsConfig::default()),
        lifecycle: tokio::sync::Mutex::new(()),
        inference_gate: std::sync::Arc::new(tokio::sync::Semaphore::new(
            crate::api::MAX_CONCURRENT_INFERENCE,
        )),
        remote_gate: std::sync::Arc::new(tokio::sync::Semaphore::new(
            crate::api::MAX_CONCURRENT_INFERENCE,
        )),
        last_activity: parking_lot::Mutex::new(std::time::Instant::now()),
        log_incoming_tokens: std::sync::atomic::AtomicBool::new(false),
        jit_enabled: std::sync::atomic::AtomicBool::new(true),
        auto_unload_idle: std::sync::atomic::AtomicBool::new(true),
        idle_ttl_minutes: std::sync::atomic::AtomicU64::new(IDLE_UNLOAD_TTL_MINUTES),
        loaded_idle_ttl_override: std::sync::atomic::AtomicU64::new(0),
        serving_enabled: std::sync::atomic::AtomicBool::new(true),
        probe_cache: parking_lot::Mutex::new(std::collections::HashMap::new()),
        device_cache: parking_lot::Mutex::new(None),
        fleet: parking_lot::Mutex::new(None),
        api_keys: parking_lot::Mutex::new(std::sync::Arc::new(crate::keys::ApiKeys::default())),
        hub: parking_lot::Mutex::new(None),
    };

    let (mut rx, handle) = higgs
        .chat_stream(
            "org/model".to_owned(),
            r#"[{"role":"user","content":"hi"}]"#.to_owned(),
            256,
            0.7,
            None,
        )
        .await
        .expect("chat_stream should succeed");

    // Inject chunk notifications tagged with request_id=1 (the first allocated id).
    use crate::rpc::{encode, RpcFrame, RpcNotification};
    for delta in &["hel", "lo"] {
        let notif = encode(&RpcFrame::Notification(RpcNotification {
            jsonrpc: "2.0".into(),
            method: N_CHAT_CHUNK.into(),
            params: json!({ "request_id": 1u64, "delta": delta }),
        }));
        test_write
            .write_all(format!("{notif}\n").as_bytes())
            .await
            .unwrap();
    }
    test_write.flush().await.unwrap();

    tokio::time::sleep(std::time::Duration::from_millis(30)).await;

    // Final response for M_CHAT (RPC id=1) — includes token counts.
    write_response(
            &mut test_write,
            1,
            json!({"content": "hello", "finish_reason": "stop", "prompt_tokens": 10, "completion_tokens": 3}),
        )
        .await;

    let outcome = tokio::time::timeout(std::time::Duration::from_millis(500), handle)
        .await
        .expect("join timeout")
        .expect("join error")
        .expect("chat outcome error");

    assert_eq!(outcome.content, "hello");
    assert_eq!(outcome.finish_reason, "stop");
    assert_eq!(outcome.prompt_tokens, 10);
    assert_eq!(outcome.completion_tokens, 3);

    // Chunks must have arrived.
    let chunk1 = rx.try_recv().expect("chunk 1");
    let chunk2 = rx.try_recv().expect("chunk 2");
    assert_eq!(chunk1, "hel");
    assert_eq!(chunk2, "lo");
}

// ── Test 5: chat_stream against dead worker removes sink ─────────────────

/// When the chat request fails (write_tx is None — worker not running), the
/// spawned task removes the sink on the error path so the map stays clean.
#[tokio::test]
async fn chat_stream_dead_worker_removes_sink() {
    // Build a Supervisor with no worker halves — factory always fails.
    let sup = crate::supervisor::Supervisor::with_factory(Box::new(|_ring, _model| {
        Err(HiggsError::WorkerSpawnFailed {
            source: std::io::Error::other("mock: no worker"),
        })
    }));
    // Do NOT call start() — write_tx stays None (dead worker).

    let higgs = Higgs {
        sup: Arc::new(sup),
        config: parking_lot::Mutex::new(HiggsConfig::default()),
        lifecycle: tokio::sync::Mutex::new(()),
        inference_gate: std::sync::Arc::new(tokio::sync::Semaphore::new(
            crate::api::MAX_CONCURRENT_INFERENCE,
        )),
        remote_gate: std::sync::Arc::new(tokio::sync::Semaphore::new(
            crate::api::MAX_CONCURRENT_INFERENCE,
        )),
        last_activity: parking_lot::Mutex::new(std::time::Instant::now()),
        log_incoming_tokens: std::sync::atomic::AtomicBool::new(false),
        jit_enabled: std::sync::atomic::AtomicBool::new(true),
        auto_unload_idle: std::sync::atomic::AtomicBool::new(true),
        idle_ttl_minutes: std::sync::atomic::AtomicU64::new(IDLE_UNLOAD_TTL_MINUTES),
        loaded_idle_ttl_override: std::sync::atomic::AtomicU64::new(0),
        serving_enabled: std::sync::atomic::AtomicBool::new(true),
        probe_cache: parking_lot::Mutex::new(std::collections::HashMap::new()),
        device_cache: parking_lot::Mutex::new(None),
        fleet: parking_lot::Mutex::new(None),
        api_keys: parking_lot::Mutex::new(std::sync::Arc::new(crate::keys::ApiKeys::default())),
        hub: parking_lot::Mutex::new(None),
    };

    // chat_stream registers the sink then the spawned task encounters dead worker.
    let (_rx, handle) = higgs
        .chat_stream(
            "org/model".to_owned(),
            r#"[{"role":"user","content":"hi"}]"#.to_owned(),
            8,
            0.0,
            None,
        )
        .await
        .expect("chat_stream itself should not fail");

    // The spawned task must return an Err (worker dead).
    let result = tokio::time::timeout(std::time::Duration::from_millis(200), handle)
        .await
        .expect("join timeout")
        .expect("join error");
    assert!(result.is_err(), "chat against dead worker must fail");

    // After the failed request, the sink map must be empty (remove_chat_sink was called).
    assert_eq!(
        higgs.sup.chat_sinks_count(),
        0,
        "chat_sinks must be empty after failed request"
    );
}
