use super::*;
use crate::node::test_support::fake_runtime_stateful;

// ── Test seam ─────────────────────────────────────────────────────────────

/// A `Higgs` facade over a STATEFUL-fake-worker LOCAL node scanning `dirs` (no
/// llama.cpp): M_LOAD records the model, M_STATUS reports it, M_CHAT streams
/// `he`/`llo` + token counts. The node and the facade config both see `dirs`.
fn fake_higgs(dirs: Vec<PathBuf>) -> Higgs {
    let node = fake_runtime_stateful(dirs.clone());
    let cfg = HiggsConfig {
        lmstudio_dirs: dirs,
        hf_dirs: vec![],
        ollama_dirs: vec![],
        default_load: HiggsConfig::default().default_load,
    };
    Higgs::with_local(Arc::new(node), cfg)
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
    let higgs = fake_higgs(vec![]);
    let err = higgs
        .load("org/../../etc/passwd", None)
        .await
        .expect_err("traversal id must be rejected");
    assert!(matches!(err, HiggsError::InvalidModelId { .. }));
}

/// A successful load PERSISTS a per-model record (P2): after `load`, the instance's
/// `config.json` (a hermetic per-test temp path) carries a `ModelRecord` for that id with
/// the effective load params. A no-params load records `ctx_len: 0` (= AUTO).
#[tokio::test]
async fn load_persists_model_record() {
    let dir = tempfile::TempDir::new().unwrap();
    crate::serve::test_support::write_gguf_fixture(dir.path(), "org/model");
    let higgs = fake_higgs(vec![dir.path().to_path_buf()]);
    assert!(
        higgs.model_records().is_empty(),
        "no records before any load"
    );

    higgs.load("org/model", None).await.expect("load");
    let records = higgs.model_records();
    let rec = records.get("org/model").expect("record persisted on load");
    let load = rec.load.as_ref().expect("load params persisted");
    assert_eq!(load.ctx_len, 0, "no-params load records ctx_len 0 = auto");
    assert!(rec.last_loaded_ms > 0, "load stamps a timestamp");
}

/// The inference admission gate returns `ServerBusy` once all permits are
/// taken; releasing a permit re-opens a slot.
#[tokio::test]
async fn inference_gate_rejects_when_full() {
    // A loaded model so chat_stream resolves past the served-id lookup and reaches
    // the admission gate.
    let dir = tempfile::TempDir::new().unwrap();
    crate::serve::test_support::write_gguf_fixture(dir.path(), "org/model");
    let mut higgs = fake_higgs(vec![dir.path().to_path_buf()]);
    // One-slot gate so the test deterministically fills it (override the default capacity).
    higgs.inference_gate = Arc::new(tokio::sync::Semaphore::new(1));
    higgs.load("org/model", None).await.expect("load");
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
    // With the permit released, a request is admitted again.
    assert!(
        Arc::clone(&higgs.inference_gate)
            .try_acquire_owned()
            .is_ok(),
        "slot re-opens after release"
    );
}

/// `chat_stream` for a model that is not locally served is [HG002] ModelNotFound,
/// before any admission permit is taken (no worker, nothing to clean up).
#[tokio::test]
async fn chat_stream_unloaded_model_not_found() {
    let higgs = fake_higgs(vec![]);
    let err = higgs
        .chat_stream(
            "org/missing".to_owned(),
            r#"[{"role":"user","content":"hi"}]"#.to_owned(),
            8,
            0.0,
            None,
        )
        .await
        .expect_err("unloaded model → ModelNotFound");
    assert!(matches!(err, HiggsError::ModelNotFound { .. }), "got {err}");
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

/// The insufficient-memory diagnostic renders the HG017 code. The end-to-end
/// refusal is covered by `fits_in_memory_respects_headroom_fraction` and the
/// HG017 status-mapping test in `serve::mod`.
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

#[tokio::test]
async fn verbose_defaults_false_and_round_trips() {
    let higgs = fake_higgs(vec![]);
    assert!(!higgs.verbose(), "verbose defaults to false");
    higgs.set_verbose(true);
    assert!(higgs.verbose(), "set_verbose(true) is observed");
    higgs.set_verbose(false);
    assert!(!higgs.verbose(), "set_verbose(false) is observed");
}

// ── Log-incoming-tokens toggle: default false, set/get round-trip ─────────

#[tokio::test]
async fn log_incoming_tokens_defaults_false_and_round_trips() {
    let higgs = fake_higgs(vec![]);
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

#[tokio::test]
async fn jit_enabled_defaults_true_and_round_trips() {
    let higgs = fake_higgs(vec![]);
    assert!(higgs.jit_enabled(), "JIT defaults to ON (true)");
    higgs.set_jit_enabled(false);
    assert!(!higgs.jit_enabled(), "set_jit_enabled(false) is observed");
    higgs.set_jit_enabled(true);
    assert!(higgs.jit_enabled(), "set_jit_enabled(true) is observed");
}

// ── Idle auto-unload toggles: defaults + round-trip ──────────────────────

#[tokio::test]
async fn idle_unload_settings_default_and_round_trip() {
    let higgs = fake_higgs(vec![]);
    // Defaults: auto-unload ON, TTL 60 minutes (mirrors the node's DEFAULT_IDLE_TTL).
    assert!(
        higgs.auto_unload_idle(),
        "auto-unload defaults to ON (true)"
    );
    assert_eq!(
        higgs.idle_ttl_minutes(),
        DEFAULT_IDLE_TTL.as_secs() / 60,
        "TTL defaults to the node's idle TTL (60 minutes)"
    );

    higgs.set_auto_unload_idle(false);
    assert!(!higgs.auto_unload_idle(), "set_auto_unload_idle(false)");
    higgs.set_auto_unload_idle(true);
    assert!(higgs.auto_unload_idle(), "set_auto_unload_idle(true)");

    higgs.set_idle_ttl_minutes(30);
    assert_eq!(higgs.idle_ttl_minutes(), 30, "set_idle_ttl_minutes(30)");
    // The setter also drives the node's live idle policy.
    assert_eq!(
        higgs.local.idle().ttl(),
        std::time::Duration::from_secs(30 * 60),
        "set_idle_ttl_minutes propagates to the node reaper"
    );
}

// ── Per-load idle-TTL override: default None, set/clear round-trip ────────

#[tokio::test]
async fn loaded_idle_ttl_override_defaults_none_and_round_trips() {
    let higgs = fake_higgs(vec![]);
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

// ── Serving on/off gate: default true, set/get round-trip ─────────────────

#[tokio::test]
async fn serving_enabled_defaults_true_and_round_trips() {
    let higgs = fake_higgs(vec![]);
    assert!(higgs.serving_enabled(), "serving defaults to ON (true)");
    higgs.set_serving_enabled(false);
    assert!(!higgs.serving_enabled(), "set_serving_enabled(false)");
    higgs.set_serving_enabled(true);
    assert!(higgs.serving_enabled(), "set_serving_enabled(true)");
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
/// that never loaded a model and empty config dirs, it returns `Ok(empty)` and
/// status reports no resident worker.
#[tokio::test]
async fn scan_runs_host_side_without_worker() {
    let higgs = fake_higgs(vec![]);

    let models = higgs.scan().await.expect("host-side scan should succeed");
    assert!(models.is_empty(), "empty dirs yield no models");

    // No model loaded → no resident worker: status reports worker_alive=false.
    let st = higgs.status().await.expect("status");
    assert!(!st.worker_alive, "scan must not spawn a worker");
}

// ── Test 3: load then status maps ─────────────────────────────────────────

#[tokio::test]
async fn load_then_status_maps() {
    // `load` resolves the GGUF path host-side, so point config at a fixture (ctx_train=4096).
    let dir = tempfile::TempDir::new().unwrap();
    crate::serve::test_support::write_gguf_fixture(dir.path(), "org/model");
    let expected_gpu_layers = HiggsConfig::default().default_load.gpu_layers;
    let higgs = fake_higgs(vec![dir.path().to_path_buf()]);
    let mut events_rx = higgs.events();

    higgs
        .load("org/model", None)
        .await
        .expect("load should succeed");

    // ModelLoaded event must arrive (emitted by the node on commit).
    let ev = tokio::time::timeout(std::time::Duration::from_millis(500), events_rx.recv())
        .await
        .expect("timeout")
        .expect("recv");
    assert!(matches!(ev, HiggsEvent::ModelLoaded { id } if id == "org/model"));

    let st = higgs.status().await.expect("status should succeed");
    assert!(st.worker_alive);
    // models_on_disk comes from a host-side scan of the config dirs (one GGUF fixture).
    assert_eq!(st.models_on_disk, 1);
    let li = st.loaded.expect("loaded should be Some");
    assert_eq!(li.id, "org/model");
    // ctx_len defaults to the model's trained context (4096) when the caller pins none.
    assert_eq!(li.ctx_len, 4096);
    // gpu_layers from the host default round-trips through the worker into status.
    assert_eq!(li.gpu_layers, expected_gpu_layers);
}

// ── Test 3b: status loaded info includes model metadata ──────────────────

#[tokio::test]
async fn status_loaded_info_includes_model_metadata() {
    // Metadata comes from the HOST scan, not the worker response: point config at a GGUF
    // fixture (arch=llama, ctx_train=4096, chat template) so the host-scanned `HiggsModel`
    // enriches the worker-reported `loaded`.
    let dir = tempfile::TempDir::new().unwrap();
    crate::serve::test_support::write_gguf_fixture(dir.path(), "org/model");
    let higgs = fake_higgs(vec![dir.path().to_path_buf()]);

    higgs
        .load("org/model", None)
        .await
        .expect("load should succeed");

    let st = higgs.status().await.expect("status should succeed");
    let li = st.loaded.expect("loaded should be Some");
    assert_eq!(li.id, "org/model");
    assert_eq!(li.arch.as_deref(), Some("llama"));
    assert_eq!(li.quant.as_deref(), Some("Q4_K_M"));
    assert_eq!(li.max_context_length, Some(4096));
    assert!(li.size_bytes.is_some(), "size_bytes from fixture file");
    assert_eq!(li.has_chat_template, Some(true));
}

// ── Test 3c: load is idempotent per model ─────────────────────────────────

/// A second `load` of the same id is a no-op success — the facade dedups by raw
/// model so the additive node never spawns a duplicate worker. After two loads
/// there is exactly one resident instance.
#[tokio::test]
async fn load_is_idempotent_per_model() {
    let dir = tempfile::TempDir::new().unwrap();
    crate::serve::test_support::write_gguf_fixture(dir.path(), "org/model");
    let higgs = fake_higgs(vec![dir.path().to_path_buf()]);

    higgs.load("org/model", None).await.expect("first load");
    higgs
        .load("org/model", None)
        .await
        .expect("second load (no-op)");

    assert_eq!(
        higgs.local_served_ids().await,
        vec!["org/model".to_owned()],
        "exactly one resident instance after a duplicate load"
    );
}

// ── Test 3d: multi-model — two distinct models resident + individually served ─

/// The local node is multi-model: loading two distinct models keeps BOTH resident
/// (additive), each addressable by its own served id, and a chat for each routes to
/// its own worker. status() reports the PRIMARY (lowest worker id).
#[tokio::test]
async fn multi_model_both_served_and_reachable() {
    let dir = tempfile::TempDir::new().unwrap();
    crate::serve::test_support::write_gguf_fixture(dir.path(), "org/a");
    crate::serve::test_support::write_gguf_fixture(dir.path(), "org/b");
    let higgs = fake_higgs(vec![dir.path().to_path_buf()]);

    higgs.load("org/a", None).await.expect("load a");
    higgs.load("org/b", None).await.expect("load b");

    // Both are served (sorted), each by its raw id (no suffix — distinct models).
    assert_eq!(
        higgs.local_served_ids().await,
        vec!["org/a".to_owned(), "org/b".to_owned()],
        "both models are served at once"
    );

    // Each served id resolves to its own loaded instance.
    assert_eq!(
        higgs.local_loaded_info("org/a").await.map(|l| l.id),
        Some("org/a".to_owned())
    );
    assert_eq!(
        higgs.local_loaded_info("org/b").await.map(|l| l.id),
        Some("org/b".to_owned())
    );

    // A chat for each routes to its own worker and completes.
    for id in ["org/a", "org/b"] {
        let (_rx, handle) = higgs
            .chat_stream(
                id.to_owned(),
                r#"[{"role":"user","content":"hi"}]"#.to_owned(),
                32,
                0.0,
                None,
            )
            .await
            .unwrap_or_else(|e| panic!("chat_stream {id}: {e}"));
        let out = tokio::time::timeout(std::time::Duration::from_millis(500), handle)
            .await
            .expect("join timeout")
            .expect("join error")
            .expect("chat outcome");
        assert_eq!(out.content, "hello", "{id} served");
    }

    // status() reports the primary (lowest worker id = first loaded = org/a).
    let st = higgs.status().await.expect("status");
    assert!(st.worker_alive);
    assert_eq!(st.loaded.expect("loaded").id, "org/a", "primary is org/a");
}

// ── Test 4: chat_stream delivers chunks and outcome ────────────────────────

#[tokio::test]
async fn chat_stream_delivers() {
    // Load a fixture model, then stream a chat. The stateful fake worker streams
    // `he`/`llo` then a final `hello` with token counts.
    let dir = tempfile::TempDir::new().unwrap();
    crate::serve::test_support::write_gguf_fixture(dir.path(), "org/model");
    let higgs = fake_higgs(vec![dir.path().to_path_buf()]);
    higgs.load("org/model", None).await.expect("load");

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

    let outcome = tokio::time::timeout(std::time::Duration::from_millis(500), handle)
        .await
        .expect("join timeout")
        .expect("join error")
        .expect("chat outcome error");

    assert_eq!(outcome.content, "hello");
    assert_eq!(outcome.finish_reason, "stop");
    // Token usage from the worker's final response must propagate to the outcome.
    assert_eq!(outcome.prompt_tokens, 10);
    assert_eq!(outcome.completion_tokens, 3);

    // The streamed deltas must have arrived in order.
    let chunk1 = rx.recv().await.expect("chunk 1");
    let chunk2 = rx.recv().await.expect("chunk 2");
    assert_eq!(chunk1, "he");
    assert_eq!(chunk2, "llo");
}
