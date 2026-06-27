use super::*;
use crate::node::test_support::fake_runtime_stateful;
use crate::worker::engine::{CtxLen, GpuLayers};

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

/// A deterministic (greedy) sampling umbrella for `chat_stream` test calls.
fn greedy_sampling() -> crate::worker::engine::SamplingParams {
    crate::worker::engine::SamplingParams::llamacpp(
        crate::worker::engine::llamacpp::params::LlamaCppSamplingParams {
            temperature: Some(0.0),
            ..Default::default()
        },
    )
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
    assert_eq!(
        load.ctx_len(),
        CtxLen::Auto,
        "no-params load records ctx_len 0 = auto"
    );
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
            greedy_sampling(),
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
            greedy_sampling(),
            None,
        )
        .await
        .expect_err("unloaded model → ModelNotFound");
    assert!(matches!(err, HiggsError::ModelNotFound { .. }), "got {err}");
}

/// `overlay_sampling` is the chat-time merge: with NO stored profile the request
/// stands alone; with a stored card/tuned base, the request's set fields win while
/// the base's other samplers survive (so a plain-temperature chat still inherits
/// the recommended top_k/min_p/…).
#[test]
fn overlay_sampling_merges_stored_base_with_request() {
    use crate::worker::engine::llamacpp::params::LlamaCppSamplingParams;
    use crate::worker::engine::SamplingParams;

    // No stored profile → the request is returned verbatim.
    let req = SamplingParams::llamacpp(LlamaCppSamplingParams {
        temperature: Some(1.1),
        ..Default::default()
    });
    let out = overlay_sampling(None, req.clone());
    assert_eq!(out, req, "no base → request unchanged");

    // A stored card base (top_k/min_p) overlaid by a temperature-only request.
    let base = SamplingParams::llamacpp(LlamaCppSamplingParams {
        temperature: Some(0.6),
        top_k: Some(40),
        min_p: Some(0.05),
        ..Default::default()
    });
    let merged = overlay_sampling(Some(base), req);
    let s = merged.as_llamacpp();
    assert_eq!(s.temperature, Some(1.1), "request temp wins");
    assert_eq!(s.top_k, Some(40), "base top_k survives");
    assert_eq!(s.min_p, Some(0.05), "base min_p survives");
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
    // `/api/higgs/system` limits report the EFFECTIVE (live) idle TTL — the same
    // value the settings endpoint returns, never a stale fixed constant. (Regression
    // guard: it once reported a hardcoded 5-minute constant while the default is 60.)
    assert_eq!(
        higgs.server_config().limits.idle_unload_ttl_secs,
        higgs.idle_ttl_minutes() * 60,
        "/system idle_unload_ttl_secs tracks the live TTL (settings ↔ system agree)"
    );

    // A huge request value is CLAMPED, not overflowed — `minutes * 60` would otherwise
    // panic (debug) or wrap (release) since the TTL comes straight off the HTTP body.
    higgs.set_idle_ttl_minutes(u64::MAX);
    assert_eq!(
        higgs.idle_ttl_minutes(),
        MAX_IDLE_TTL_MINUTES,
        "an out-of-range idle TTL is clamped to MAX_IDLE_TTL_MINUTES"
    );
    // …and the clamped value still converts to seconds without overflow, keeping the
    // settings ↔ system views consistent at the bound.
    assert_eq!(
        higgs.server_config().limits.idle_unload_ttl_secs,
        MAX_IDLE_TTL_MINUTES * 60,
        "clamped TTL converts to seconds without overflow"
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
    let expected_gpu_layers = HiggsConfig::default().default_load.gpu_layers();
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
    assert_eq!(li.ctx_len, Some(CtxLen::Fixed { n: 4096 }));
    // gpu_layers from the host default round-trips through the worker into status.
    assert_eq!(li.gpu_layers, Some(expected_gpu_layers));
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
                greedy_sampling(),
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

// ── Test 3e: resident-but-busy worker — local_loaded_info must not say "not served" ─

/// A served id whose worker is resident but cannot answer `M_STATUS` (busy mid-generation
/// / briefly wedged) must still resolve to a LOADED stub — NOT `None`, which would mislead
/// the serve gate into a not-loaded `[HG003]`. The stub carries a permissive `ctx_len`
/// (`u32::MAX`) so the host prompt-fit gate can't reject a queued chat either.
#[tokio::test]
async fn local_loaded_info_busy_worker_returns_permissive_stub() {
    let dir = tempfile::TempDir::new().unwrap();
    crate::serve::test_support::write_gguf_fixture(dir.path(), "org/busy");
    // A node whose worker LOADS fine (so it is resident) but ERRORS every status probe.
    let node = crate::node::test_support::fake_runtime_status_fails(vec![dir.path().to_path_buf()]);
    let cfg = HiggsConfig {
        lmstudio_dirs: vec![dir.path().to_path_buf()],
        hf_dirs: vec![],
        ollama_dirs: vec![],
        default_load: HiggsConfig::default().default_load,
    };
    let higgs = Higgs::with_local(Arc::new(node), cfg);

    higgs
        .load("org/busy", None)
        .await
        .expect("load succeeds (M_LOAD ok)");

    // The served id resolves to a resident worker, so even though the status probe errors,
    // local_loaded_info reports a LOADED stub rather than None.
    let info = higgs
        .local_loaded_info("org/busy")
        .await
        .expect("resident worker resolves despite a failing status probe");
    assert_eq!(info.id, "org/busy", "stub carries the served id");
    assert_eq!(
        info.ctx_len, None,
        "busy-worker stub ctx_len is None (not probed) so the prompt-fit gate defers to HG005"
    );

    // And an id that is NOT served still resolves to None (no false positive).
    assert!(
        higgs.local_loaded_info("org/absent").await.is_none(),
        "an unserved id is still None"
    );
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
            greedy_sampling(),
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

// ── log_bus(): delegates to the local node's shared bus ───────────────────

/// `log_bus()` returns the SAME bus the local node carries (verbosity toggled on
/// one is observed on the other), and a fresh facade's verbose default is off.
#[tokio::test]
async fn log_bus_returns_node_shared_bus() {
    let higgs = fake_higgs(vec![]);
    let bus = higgs.log_bus();
    assert!(!bus.verbose(), "fresh bus defaults to non-verbose");
    // The bus is the node's: flipping verbose through the facade is visible on the
    // returned handle (same underlying LogBus, not a clone of state).
    higgs.set_verbose(true);
    assert!(bus.verbose(), "log_bus() hands back the node's live bus");
}

// ── set_fleet / fleet(): install + idempotent replace ─────────────────────

/// `fleet()` is `None` on a pure-local facade; `set_fleet` installs one and
/// `fleet()` then returns it, and a second `set_fleet` REPLACES it (idempotent).
#[tokio::test]
async fn set_fleet_installs_and_replaces() {
    let higgs = fake_higgs(vec![]);
    assert!(higgs.fleet().is_none(), "no fleet on a pure-local facade");

    let bus = higgs.log_bus();
    let fleet = Arc::new(crate::node::fleet::HubFleet::new(bus.clone()));
    higgs.set_fleet(Arc::clone(&fleet));
    assert!(higgs.fleet().is_some(), "fleet installed");

    // Replacing with a fresh fleet swaps the installed handle wholesale.
    let fleet2 = Arc::new(crate::node::fleet::HubFleet::new(bus));
    higgs.set_fleet(Arc::clone(&fleet2));
    assert!(
        Arc::ptr_eq(&higgs.fleet().expect("fleet present"), &fleet2),
        "set_fleet replaces the prior fleet"
    );
}

// ── chat_stream: fleet installed but the model is not remote → ModelNotFound ─

/// With a fleet installed but NO remote node serving the id, `chat_stream` falls
/// through the local-served lookup AND the `fleet.is_remote` check (which is
/// `false` for an unknown id) to the terminal [HG002] ModelNotFound. Exercises
/// the `fleet.lock().clone()` + `is_remote` branch without iroh networking.
#[tokio::test]
async fn chat_stream_with_fleet_unknown_model_not_found() {
    let higgs = fake_higgs(vec![]);
    let bus = higgs.log_bus();
    higgs.set_fleet(Arc::new(crate::node::fleet::HubFleet::new(bus)));

    let err = higgs
        .chat_stream(
            "org/not-here".to_owned(),
            r#"[{"role":"user","content":"hi"}]"#.to_owned(),
            8,
            greedy_sampling(),
            None,
        )
        .await
        .expect_err("no local + no remote route → ModelNotFound");
    assert!(matches!(err, HiggsError::ModelNotFound { .. }), "got {err}");
}

// ── start(): brings up the worker-stderr log relay without error ───────────

/// `start()` spawns the per-worker-stderr → Developer-Log relay and returns
/// `Ok(())`. (The fake worker has no real process stderr, so the relay loop's
/// per-line push is exercised by integration tests with a live worker; here we
/// cover the bring-up path is callable and clean.)
#[tokio::test]
async fn start_brings_up_log_relay() {
    let higgs = Arc::new(fake_higgs(vec![]));
    higgs.start().await.expect("start brings up the relay");
}

// ── load with rich engine overrides: persists the effective LlamaCpp set ───

/// A load carrying an ENGINE OVERRIDE beyond the base three (here `use_mmap`)
/// crosses to the node as a `params` payload and is PERSISTED as the effective
/// `LlamaCpp` `LoadParams` record (the `Some(lc)` effective branch), with the
/// node-resolved base fields stamped in. Because request params were supplied,
/// the accepted profile is also synced to `models.json`.
#[tokio::test]
async fn load_with_engine_overrides_persists_effective_llamacpp() {
    use crate::worker::engine::llamacpp::params::LlamaCppParams;

    let dir = tempfile::TempDir::new().unwrap();
    crate::serve::test_support::write_gguf_fixture(dir.path(), "org/model");
    let higgs = fake_higgs(vec![dir.path().to_path_buf()]);

    // ctx_len 0 = AUTO; gpu_layers all; an override (use_mmap) forces the rich path.
    let params = LoadParams::llamacpp(LlamaCppParams {
        ctx_len: CtxLen::Auto,
        gpu_layers: GpuLayers::All,
        threads: 4,
        use_mmap: Some(true),
        ..Default::default()
    });
    higgs
        .load("org/model", Some(params))
        .await
        .expect("load with overrides");

    // The persisted config record reflects the effective LlamaCpp set (override kept).
    let records = higgs.model_records();
    let rec = records.get("org/model").expect("record persisted");
    let load = rec.load.as_ref().expect("load params persisted");
    let lc = load.as_llamacpp();
    assert_eq!(lc.use_mmap, Some(true), "the engine override is persisted");
    assert_eq!(
        lc.gpu_layers,
        GpuLayers::All,
        "node-resolved gpu_layers stamped in"
    );
    assert_eq!(lc.threads, 4, "node-resolved threads stamped in");
    assert_eq!(lc.ctx_len, CtxLen::Auto, "auto ctx_len is CtxLen::Auto");

    // Request params were supplied → the accepted profile is synced to models.json.
    let store = higgs.models_store().expect("models store opens");
    let tuned = store.tuning("org/model").expect("accepted profile synced");
    assert_eq!(
        tuned.profile.as_llamacpp().use_mmap,
        Some(true),
        "the synced profile carries the override"
    );
}

// ── loaded_info_from: a null `loaded` value maps to None ──────────────────

/// `loaded_info_from` returns `None` when the worker's status `loaded` field is
/// explicitly null (nothing resident), distinct from a missing field. Covers the
/// `l.is_null()` early-out shared by `status` and `local_loaded_info`.
#[tokio::test]
async fn loaded_info_from_null_loaded_is_none() {
    let higgs = fake_higgs(vec![]);
    let v = serde_json::json!({ "loaded": serde_json::Value::Null });
    assert!(
        higgs.loaded_info_from(&v, &[]).is_none(),
        "null loaded → None"
    );
    // A wholly absent `loaded` key also yields None (the `?` on `v.get(\"loaded\")`).
    let empty = serde_json::json!({});
    assert!(
        higgs.loaded_info_from(&empty, &[]).is_none(),
        "missing loaded → None"
    );
}

// ── local_node_view: local machine as a first-class NodeView ───────────────

/// `local_node_view` reports the LOCAL machine as the `node_id = 0`, `is_local`,
/// always-connected sentinel, carrying the caller's label and one
/// [`InventoryWorker`] per resident model tagged with its served id. Covers the
/// worker-mapping loop and the sentinel construction.
#[tokio::test]
async fn local_node_view_lists_resident_workers() {
    let dir = tempfile::TempDir::new().unwrap();
    crate::serve::test_support::write_gguf_fixture(dir.path(), "org/model");
    let higgs = fake_higgs(vec![dir.path().to_path_buf()]);

    // No workers yet: a first-class but empty local node view.
    let empty = higgs.local_node_view("my-box".to_owned()).await;
    assert_eq!(empty.node_id, 0, "local sentinel node id");
    assert_eq!(empty.endpoint_id, "local");
    assert!(empty.is_local && empty.connected, "local is first-class");
    assert_eq!(empty.label, "my-box");
    let inv0 = empty.inventory.expect("inventory present");
    assert!(inv0.workers.is_empty(), "no workers before a load");

    // After a load, the view lists the worker tagged with its served id.
    higgs.load("org/model", None).await.expect("load");
    let view = higgs.local_node_view("my-box".to_owned()).await;
    let inv = view.inventory.expect("inventory present");
    assert_eq!(inv.workers.len(), 1, "one resident worker listed");
    let w = &inv.workers[0];
    assert_eq!(w.model, "org/model", "raw model on the worker");
    assert_eq!(w.served_id, "org/model", "served id (deduped == raw)");
}

// ── sysinfo + hardware: cache hit + the host hardware snapshot ─────────────

/// `sysinfo()` returns the cached device list on a hit (no worker round-trip),
/// and `hardware()` folds the (empty) GPU list into a full host CPU/RAM snapshot.
/// The fake worker reports no GPUs, so a cache MISS leaves the cache empty and a
/// non-empty result/store is only reachable with real FFI — primed here directly.
#[tokio::test]
async fn sysinfo_cache_hit_and_hardware_snapshot() {
    let higgs = fake_higgs(vec![]);

    // Cache MISS path: the fake sysinfo worker yields an empty list, which is NOT
    // cached (an empty result usually means a failed gather → retry later).
    let miss = higgs.sysinfo().await;
    assert!(miss.is_empty(), "fake worker reports no GPUs");
    assert!(
        higgs.device_cache.lock().is_none(),
        "an empty gather is not cached (so a later call retries)"
    );

    // Prime the cache directly (the only seam to a non-empty list without FFI), then
    // a hit returns it verbatim with no worker round-trip.
    let primed = vec![crate::system::GpuDevice {
        name: "TestGPU".to_owned(),
        description: "unit-test device".to_owned(),
        kind: crate::system::DeviceKind::Gpu,
        vram_total_bytes: 8_000_000_000,
        vram_free_bytes: 4_000_000_000,
    }];
    *higgs.device_cache.lock() = Some(primed.clone());
    let hit = higgs.sysinfo().await;
    assert_eq!(hit.len(), 1, "a cache hit returns the cached list");
    assert_eq!(hit[0].name, "TestGPU");

    // hardware() folds the (cached) GPU list into the host CPU/RAM snapshot.
    let hw = higgs.hardware().await;
    assert_eq!(hw.gpus.len(), 1, "hardware carries the cached GPU list");
    assert_eq!(hw.gpus[0].name, "TestGPU");
    assert!(hw.cpu_cores >= 1, "host CPU cores are reported");
}

// ── tune: derive within budget + persist as the saved profile ──────────────

/// `tune` (Suggest mode) for a scanned model: looks up its GGUF meta, derives a
/// load+sampling set within the budget, and PERSISTS it as the saved profile so
/// the next plain load reuses it. An `ollama/` id short-circuits the HF-card
/// fetch (no network), so the static-default suggester path runs deterministically.
#[tokio::test]
async fn tune_derives_and_persists_profile() {
    let dir = tempfile::TempDir::new().unwrap();
    let id = stage_ollama_model(dir.path(), "llama3", "8b");
    let higgs = fake_higgs_ollama(vec![dir.path().to_path_buf()]);

    let suggestion = higgs
        .tune(TuneRequest {
            id: id.clone(),
            mode: None, // Suggest (default)
            budget: None,
        })
        .await
        .expect("tune suggests");
    assert_eq!(suggestion.id, id, "suggestion is for the requested model");
    assert_eq!(
        suggestion.provenance,
        crate::tune::TuneProvenance::Heuristic,
        "no HF card for an ollama id → heuristic provenance"
    );

    // The suggestion is persisted as the saved tuning profile for this id.
    let store = higgs.models_store().expect("models store opens");
    let rec = store.tuning(&id).expect("tuning persisted by tune");
    assert_eq!(
        rec.profile, suggestion.load,
        "the persisted profile matches the suggested load"
    );
}

/// `tune` in Benchmark mode falls back to Suggest (P2 not yet available) and still
/// persists a profile — covering the benchmark-mode logging branch.
#[tokio::test]
async fn tune_benchmark_mode_falls_back_to_suggest() {
    let dir = tempfile::TempDir::new().unwrap();
    let id = stage_ollama_model(dir.path(), "qwen", "0.5b");
    let higgs = fake_higgs_ollama(vec![dir.path().to_path_buf()]);

    let suggestion = higgs
        .tune(TuneRequest {
            id: id.clone(),
            mode: Some(TuneMode::Benchmark),
            budget: None,
        })
        .await
        .expect("benchmark tune falls back to suggest");
    assert_eq!(suggestion.id, id);
    assert!(
        higgs.models_store().expect("store").tuning(&id).is_some(),
        "benchmark-fallback still persists a profile"
    );
}

/// `tune` for an id that is not scanned is [HG002] ModelNotFound, before any
/// hardware/card work.
#[tokio::test]
async fn tune_unknown_model_not_found() {
    let higgs = fake_higgs(vec![]);
    let err = higgs
        .tune(TuneRequest {
            id: "org/absent".to_owned(),
            mode: None,
            budget: None,
        })
        .await
        .expect_err("unknown model → ModelNotFound");
    assert!(matches!(err, HiggsError::ModelNotFound { .. }), "got {err}");
}

// ── load persistence failure: best-effort warn, load still succeeds ────────

/// When the per-model config record can't be persisted (the config path's parent
/// is a regular FILE, so both load+save fail), `load` logs the failure but STILL
/// returns success — persistence is best-effort. Covers the `with_config_mut`
/// error/warn arm of `load`.
#[tokio::test]
async fn load_succeeds_despite_config_persist_failure() {
    let dir = tempfile::TempDir::new().unwrap();
    crate::serve::test_support::write_gguf_fixture(dir.path(), "org/model");
    let higgs = fake_higgs(vec![dir.path().to_path_buf()]);

    // Point config.json UNDER a regular file: `<file>/config.json`. `InstanceConfig`
    // load+save both fail (the parent is not a directory), so `with_config_mut` errs.
    let blocker = dir.path().join("not-a-dir");
    std::fs::write(&blocker, b"x").unwrap();
    *higgs.config_path.lock() = Some(blocker.join("config.json"));

    higgs
        .load("org/model", None)
        .await
        .expect("load succeeds even when the config record can't be persisted");

    // The model IS resident despite the persistence failure, and no record was written.
    assert_eq!(
        higgs.local_served_ids().await,
        vec!["org/model".to_owned()],
        "the model loaded"
    );
    assert!(
        higgs.model_records().is_empty(),
        "the unwritable config holds no record"
    );
}

/// When the accepted-profile flush fails (a read-only models home), `sync_saved_profile`
/// logs the coded failure but the load still succeeds. Drives the
/// `store.flush()` error arm. Skips when running as root (perms are ignored).
#[tokio::test]
async fn sync_saved_profile_survives_flush_failure() {
    use std::os::unix::fs::PermissionsExt;

    // Root ignores file perms, so a read-only dir wouldn't fail the write — skip.
    if unsafe { libc::geteuid() } == 0 {
        return;
    }

    let dir = tempfile::TempDir::new().unwrap();
    crate::serve::test_support::write_gguf_fixture(dir.path(), "org/model");
    let higgs = fake_higgs(vec![dir.path().to_path_buf()]);

    // A READ-ONLY home dir: `models_store().open` succeeds (models.json absent →
    // empty store) but `flush()` can't create its temp file → Err. Point config.json
    // INSIDE this dir so `models_store` derives its home as the read-only dir.
    let home = dir.path().join("ro-home");
    std::fs::create_dir_all(&home).unwrap();
    *higgs.config_path.lock() = Some(home.join("config.json"));
    std::fs::set_permissions(&home, std::fs::Permissions::from_mode(0o500)).unwrap();

    // A load WITH request params triggers `sync_saved_profile` (from_request = true).
    let res = higgs
        .load(
            "org/model",
            Some(LoadParams::base(CtxLen::Auto, GpuLayers::All, 4)),
        )
        .await;

    // Restore perms so TempDir cleanup can remove the dir regardless of the outcome.
    std::fs::set_permissions(&home, std::fs::Permissions::from_mode(0o700)).unwrap();

    res.expect("load succeeds even when the accepted profile can't be flushed");
    assert_eq!(
        higgs.local_served_ids().await,
        vec!["org/model".to_owned()],
        "the model loaded despite the flush failure"
    );
}

// ── ollama staging helpers (for the no-network tune tests) ─────────────────

/// Stage a minimal Ollama model under `root` so a host-side scan catalogs it as
/// `ollama/<name>:<tag>` (an id `fetch_card_sampling` short-circuits, so `tune`
/// runs offline). Writes the manifest + a `GGUF`-magic blob the digest points at.
/// Returns the scanned model id.
fn stage_ollama_model(root: &std::path::Path, name: &str, tag: &str) -> String {
    // A tiny but valid-enough GGUF blob (the scanner only checks the 4-byte magic).
    let blob_bytes = b"GGUF\x00 dummy ollama blob";
    let hex = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    let blob_dir = root.join("blobs");
    std::fs::create_dir_all(&blob_dir).unwrap();
    std::fs::write(blob_dir.join(format!("sha256-{hex}")), blob_bytes).unwrap();

    let manifest = serde_json::json!({
        "layers": [{
            "mediaType": "application/vnd.ollama.image.model",
            "digest": format!("sha256:{hex}"),
        }],
    });
    let manifest_dir = root.join("manifests").join(name);
    std::fs::create_dir_all(&manifest_dir).unwrap();
    std::fs::write(manifest_dir.join(tag), manifest.to_string()).unwrap();

    format!("ollama/{name}:{tag}")
}

/// A `Higgs` facade whose config + node scan the given OLLAMA dirs (so `tune`
/// finds an `ollama/…` model staged by [`stage_ollama_model`]).
fn fake_higgs_ollama(ollama_dirs: Vec<PathBuf>) -> Higgs {
    let node = NodeRuntime::with_spawner(
        crate::node::runtime::NodeConfig {
            bus: Arc::new(crate::log_bus::LogBus::new()),
            lmstudio_dirs: vec![],
            hf_dirs: vec![],
            ollama_dirs: ollama_dirs.clone(),
            idle_ttl: DEFAULT_IDLE_TTL,
        },
        Arc::new(|_bus| {
            crate::supervisor::Supervisor::with_factory(
                crate::node::test_support::fake_worker_factory_stateful(),
            )
        }),
    );
    let cfg = HiggsConfig {
        lmstudio_dirs: vec![],
        hf_dirs: vec![],
        ollama_dirs,
        default_load: HiggsConfig::default().default_load,
    };
    Higgs::with_local(Arc::new(node), cfg)
}
