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
        worker_exe: None,
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

/// `load_inner` with `LoadPersist::ReuseSaved` is a REUSE — it loads the given
/// profile but does NOT re-write the saved tuning record. This is the JIT path's
/// no-resync seam: the JIT load passes the profile the readiness gate VALIDATED
/// (`ReuseSaved`), so it can't silently default-load if `models.json`
/// changes after the check (the check-then-load race), and doesn't churn the
/// saved profile on every JIT load. Fails-on-revert: have JIT load via the public
/// `load(.., None)` (re-read + reuse) and this seam is unused / the saved profile
/// re-syncs.
#[tokio::test]
async fn load_inner_reuse_does_not_resync_saved_profile() {
    use crate::worker::engine::{CtxLen, GpuLayers, LoadParams};
    let dir = tempfile::TempDir::new().unwrap();
    crate::serve::test_support::write_gguf_fixture(dir.path(), "org/model");
    let higgs = fake_higgs(vec![dir.path().to_path_buf()]);

    // Seed a known saved profile (ctx 4096, tuned_at 111).
    let store = higgs.models_store().expect("open store");
    store.put_tuning(
        "org/model",
        crate::tune::store::TuneRecord {
            profile: LoadParams::base(CtxLen::Fixed { n: 4096 }, GpuLayers::All, 8),
            sampling: Default::default(),
            budget: Default::default(),
            provenance: crate::tune::TuneProvenance::Heuristic,
            bench_tps: None,
            tuned_at_ms: 111,
            hw_fingerprint: String::new(),
            model_file_sig: String::new(),
        },
    );
    store.flush().expect("flush seed");

    // REUSE load with DIFFERENT params must NOT rewrite the saved profile.
    higgs
        .load_inner(
            "org/model",
            Some(LoadParams::base(
                CtxLen::Fixed { n: 1024 },
                GpuLayers::All,
                2,
            )),
            LoadPersist::ReuseSaved,
        )
        .await
        .expect("reuse load");

    let saved = higgs
        .models_store()
        .expect("reopen store")
        .tuning("org/model")
        .expect("profile still present");
    assert_eq!(
        saved.profile.ctx_len(),
        CtxLen::Fixed { n: 4096 },
        "a reuse load did NOT re-write the saved profile"
    );
    assert_eq!(saved.tuned_at_ms, 111, "a reuse load did NOT re-stamp it");
}

/// `load_ephemeral` persists NOTHING: the saved tuning profile is untouched
/// (different params notwithstanding) AND no `config.json` last-load record is
/// written. This is the benchmark/probe seam — a one-off load must leave the
/// user's stores byte-identical. Fails-on-revert: route `load_ephemeral`
/// through `LoadPersist::Explicit` (or restore the old `from_request` bool) and
/// the profile re-syncs / a model record appears.
#[tokio::test]
async fn load_ephemeral_persists_nothing() {
    use crate::worker::engine::{CtxLen, GpuLayers, LoadParams};
    let dir = tempfile::TempDir::new().unwrap();
    crate::serve::test_support::write_gguf_fixture(dir.path(), "org/model");
    let higgs = fake_higgs(vec![dir.path().to_path_buf()]);

    // Seed a known saved profile (ctx 4096, tuned_at 111).
    let store = higgs.models_store().expect("open store");
    store.put_tuning(
        "org/model",
        crate::tune::store::TuneRecord {
            profile: LoadParams::base(CtxLen::Fixed { n: 4096 }, GpuLayers::All, 8),
            sampling: Default::default(),
            budget: Default::default(),
            provenance: crate::tune::TuneProvenance::Heuristic,
            bench_tps: None,
            tuned_at_ms: 111,
            hw_fingerprint: String::new(),
            model_file_sig: String::new(),
        },
    );
    store.flush().expect("flush seed");

    // Ephemeral load with DIFFERENT params succeeds…
    higgs
        .load_ephemeral(
            "org/model",
            LoadParams::base(CtxLen::Fixed { n: 1024 }, GpuLayers::All, 2),
        )
        .await
        .expect("ephemeral load");

    // …and the saved profile is byte-identical to the seed…
    let saved = higgs
        .models_store()
        .expect("reopen store")
        .tuning("org/model")
        .expect("profile still present");
    assert_eq!(
        saved.profile.ctx_len(),
        CtxLen::Fixed { n: 4096 },
        "an ephemeral load did NOT re-write the saved profile"
    );
    assert_eq!(
        saved.tuned_at_ms, 111,
        "an ephemeral load did NOT re-stamp it"
    );
    // …and NO last-load record was written (`load` would have written one).
    assert!(
        higgs.model_records().is_empty(),
        "an ephemeral load wrote no config.json model record"
    );
}

/// An EPHEMERAL load is "EXACTLY these params or fail": the OOM degrade ladder
/// is SKIPPED — a degraded config would be one the caller never pinned, and (by
/// the ephemeral contract) nothing about it would be persisted or reported. The
/// first OOM surfaces as the load error, and the stores stay untouched.
/// Fails-on-revert: route Ephemeral through `run_oom_ladder` and the OOM-twice
/// fake DEGRADES to a successful load, so the `expect_err` fails.
#[tokio::test]
async fn ephemeral_load_fails_on_oom_instead_of_degrading() {
    use crate::worker::engine::{CtxLen, GpuLayers, LoadParams};
    let dir = tempfile::TempDir::new().unwrap();
    crate::serve::test_support::write_gguf_fixture(dir.path(), "org/model");
    let higgs = fake_higgs_oom_twice(vec![dir.path().to_path_buf()]);

    higgs
        .load_ephemeral(
            "org/model",
            LoadParams::base(CtxLen::Auto, GpuLayers::All, 8),
        )
        .await
        .expect_err("an OOMing ephemeral load fails instead of degrading");

    assert!(
        higgs.models_store().unwrap().tuning("org/model").is_none(),
        "a failed ephemeral load wrote no tuning profile"
    );
    assert!(
        higgs.model_records().is_empty(),
        "a failed ephemeral load wrote no config.json model record"
    );
}

/// An EPHEMERAL load of an already-RESIDENT model is refused ([HG080]): the
/// resident worker keeps its CURRENT params, so a silent no-op "success" would
/// hand the caller a config it never pinned. Fails-on-revert: drop the
/// Ephemeral check before the resident no-op and this returns Ok, failing the
/// `expect_err`.
#[tokio::test]
async fn ephemeral_load_refuses_a_resident_model() {
    use crate::worker::engine::{CtxLen, GpuLayers, LoadParams};
    let dir = tempfile::TempDir::new().unwrap();
    crate::serve::test_support::write_gguf_fixture(dir.path(), "org/model");
    let higgs = fake_higgs(vec![dir.path().to_path_buf()]);
    higgs.load("org/model", None).await.expect("resident load");

    let err = higgs
        .load_ephemeral(
            "org/model",
            LoadParams::base(CtxLen::Fixed { n: 1024 }, GpuLayers::All, 2),
        )
        .await
        .expect_err("ephemeral load of a resident model is refused");
    assert!(
        matches!(err, HiggsError::EphemeralResident { .. }),
        "expected [HG080] EphemeralResident, got {err}"
    );
}

/// `status.loading` surfaces an in-flight load (for the UI progress bar) and is
/// cleared once the load finishes. Fails-on-revert: stop reading `self.loading`
/// in `status()` and the injected in-flight load no longer shows; stop clearing
/// it around `local.load` and it stays set after the load.
#[tokio::test]
async fn status_surfaces_and_clears_in_flight_load() {
    let dir = tempfile::TempDir::new().unwrap();
    crate::serve::test_support::write_gguf_fixture(dir.path(), "org/model");
    let higgs = fake_higgs(vec![dir.path().to_path_buf()]);

    // Idle → no in-flight load.
    assert!(higgs.status().await.unwrap().loading.is_none());

    // An injected in-flight load is surfaced by `status`.
    *higgs.loading.lock() = Some(crate::api::types::ModelLoading {
        id: "org/model".into(),
        started_ms: 123,
    });
    let st = higgs.status().await.unwrap();
    assert_eq!(
        st.loading.as_ref().map(|l| l.id.as_str()),
        Some("org/model"),
        "an in-flight load is surfaced on status"
    );

    // A real load clears it (set + cleared around `local.load`).
    higgs.load("org/model", None).await.expect("load");
    assert!(
        higgs.status().await.unwrap().loading.is_none(),
        "loading cleared once the load finished"
    );
}

/// A successful load pushes the full ordered phase sequence
/// (`queued`→`preparing`→`loading_weights`→`finalizing`→`ready`) to load-event
/// subscribers; a failed load pushes a terminal `failed` carrying the HG code.
#[tokio::test]
async fn load_events_emit_ordered_phases_and_terminal() {
    use crate::api::types::ModelLoadPhase;
    let dir = tempfile::TempDir::new().unwrap();
    crate::serve::test_support::write_gguf_fixture(dir.path(), "org/model");
    let higgs = fake_higgs(vec![dir.path().to_path_buf()]);

    // Subscribe BEFORE the load so no phase is missed.
    let mut rx = higgs.subscribe_load_events();
    higgs.load("org/model", None).await.expect("load");

    let mut phases = Vec::new();
    while let Ok(ev) = rx.try_recv() {
        phases.push(ev.phase);
    }
    assert_eq!(
        phases,
        vec![
            ModelLoadPhase::Queued,
            ModelLoadPhase::Preparing,
            ModelLoadPhase::LoadingWeights,
            ModelLoadPhase::Finalizing,
            ModelLoadPhase::Ready,
        ],
        "successful load emits the ordered phase sequence"
    );

    // A failing load (bad id) emits a terminal `failed` carrying its HG code.
    let mut rx = higgs.subscribe_load_events();
    let _ = higgs.load("bad id with spaces", None).await;
    let mut events = Vec::new();
    while let Ok(ev) = rx.try_recv() {
        events.push(ev);
    }
    let last = events.last().expect("at least one event");
    assert_eq!(last.phase, ModelLoadPhase::Failed, "terminal is failed");
    assert_eq!(
        last.code.as_deref(),
        Some("HG015"),
        "failed terminal carries the diagnostic code: {events:?}"
    );
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
        cfg.lmstudio_dirs
            .iter()
            .any(|p| *p == crate::home::higgs_home().join("models")),
        "lmstudio_dirs must include the higgs models dir — catalog downloads \
         land there and every embedder's scan must see them"
    );
    // A pathological occupant (a FILE at the models path) must NOT be pushed:
    // `read_dir` on it would fail the WHOLE scan, unlike a missing dir.
    {
        let _env = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let tmp = tempfile::tempdir().unwrap();
        let prev = std::env::var_os("HIGGS_HOME");
        // SAFETY: serialized by TEST_ENV_LOCK; restored below.
        unsafe { std::env::set_var("HIGGS_HOME", tmp.path()) };
        let models = tmp.path().join("models");
        let absent = HiggsConfig::default();
        assert!(
            absent.lmstudio_dirs.contains(&models),
            "absent dir is included (the scanner treats it as empty)"
        );
        std::fs::write(&models, b"not a dir").unwrap();
        let occupied = HiggsConfig::default();
        assert!(
            !occupied.lmstudio_dirs.contains(&models),
            "a non-directory occupant is skipped, not scanned"
        );
        // SAFETY: serialized by TEST_ENV_LOCK.
        unsafe {
            match prev {
                Some(v) => std::env::set_var("HIGGS_HOME", v),
                None => std::env::remove_var("HIGGS_HOME"),
            }
        }
    }
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
/// the serve gate into a not-loaded `[HG003]`. The stub's params come from the PERSISTED
/// load record (what the worker was actually loaded with), so the prompt-fit gate sees the
/// real context window even while the worker is unprobeable.
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
        worker_exe: None,
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
    assert!(
        info.ctx_len.is_some(),
        "busy-worker stub ctx_len comes from the persisted load record (not a probe)"
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

    // The streamed deltas arrived — buffered same-kind chunks merge in the
    // delta queue into one run with the text intact and in order.
    let chunk = rx.recv().await.expect("merged content run");
    assert_eq!(chunk.text, "hello");
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
            pins: None,
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
            pins: None,
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
            pins: None,
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

    // A load WITH request params triggers `sync_saved_profile` (`Explicit`).
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
        worker_exe: None,
    };
    Higgs::with_local(Arc::new(node), cfg)
}

// ── G5: OOM degrade-retry ladder (run_oom_ladder) via an injected loader ─────

fn oom_err(msg: &str) -> HiggsError {
    HiggsError::EngineLoadFailed {
        id: "org/model".into(),
        reason: format!("ggml_backend_alloc_ctx_tensors: {msg}: out of memory"),
    }
}

fn corrupt_err() -> HiggsError {
    HiggsError::EngineLoadFailed {
        id: "org/model".into(),
        reason: "done_getting_tensors: wrong number of tensors".into(),
    }
}

fn np_count(n: u32) -> crate::remote::NodeLoadParams {
    crate::remote::NodeLoadParams {
        id: "org/model".into(),
        ctx_len: None,
        gpu_layers: Some(GpuLayers::Count { n }),
        threads: None,
        params: None,
    }
}

/// A load that fits first try makes exactly ONE call — no ladder built (the
/// common jigglebot path is unchanged).
#[tokio::test]
async fn ladder_load_that_fits_calls_once() {
    let calls = std::cell::Cell::new(0);
    let r = run_oom_ladder(
        "org/model",
        np_count(40),
        None,
        std::time::Duration::ZERO,
        |_p| {
            calls.set(calls.get() + 1);
            async { Ok(()) }
        },
    )
    .await;
    assert!(r.is_ok());
    assert_eq!(calls.get(), 1, "a fitting load makes one attempt");
}

/// OOM on attempt 0, then success on the first ladder rung (plain retry): the
/// ladder RESCUES the load. Fail-on-revert: a single-shot loader (no retry
/// loop) returns the OOM error instead of Ok.
#[tokio::test]
async fn ladder_retries_after_oom_then_succeeds() {
    let n = std::cell::Cell::new(0);
    let r = run_oom_ladder(
        "org/model",
        np_count(40),
        None,
        std::time::Duration::ZERO,
        |_p| {
            let attempt = n.get();
            n.set(attempt + 1);
            async move {
                if attempt == 0 {
                    Err(oom_err("first"))
                } else {
                    Ok(())
                }
            }
        },
    )
    .await;
    assert!(r.is_ok(), "OOM then rescued on the plain-retry rung");
    assert_eq!(n.get(), 2, "attempt 0 + one rung");
}

/// When the ladder is rescued on a DEGRADED rung (KV-off, not the plain retry),
/// `run_oom_ladder` returns THAT rung's params so the caller persists the config
/// that actually loaded — not the seed that OOMed. Fail-on-revert: return the seed
/// `np` instead of the successful rung's params and the returned config equals the
/// failing seed, so this `assert_ne!` fails (and `load_inner_impl` would save the
/// wrong profile that re-OOMs on every future load).
#[tokio::test]
async fn ladder_returns_the_degraded_rung_that_loaded() {
    let n = std::cell::Cell::new(0);
    let seed = np_count(40);
    let loaded = run_oom_ladder(
        "org/model",
        seed.clone(),
        Some(40),
        std::time::Duration::ZERO,
        |_p| {
            let a = n.get();
            n.set(a + 1);
            // seed (0) + plain-retry rung (1) OOM; the KV-off rung (2) loads.
            async move {
                if a < 2 {
                    Err(oom_err("oom"))
                } else {
                    Ok(())
                }
            }
        },
    )
    .await
    .expect("rescued on the KV-off rung");
    assert_ne!(
        loaded, seed,
        "returned the DEGRADED rung params (KV off), not the seed that OOMed"
    );
}

/// Every attempt OOMs → the aggregate [HG060] with the attempt count and the
/// LAST reason. A 40-layer base has 4 attempts (0 + retry + kv-off + halve).
#[tokio::test]
async fn ladder_exhausts_to_aggregate_hg060() {
    let n = std::cell::Cell::new(0);
    let r = run_oom_ladder(
        "org/model",
        np_count(40),
        None,
        std::time::Duration::ZERO,
        |_p| {
            let i = n.get();
            n.set(i + 1);
            async move { Err(oom_err(&format!("rung{i}"))) }
        },
    )
    .await;
    match r {
        Err(HiggsError::LoadOomExhausted { attempts, last }) => {
            assert_eq!(attempts, 4, "attempt 0 + 3 rungs for a Count base");
            assert!(last.contains("rung3"), "carries the final reason: {last}");
        }
        other => panic!("expected HG060 aggregate, got {other:?}"),
    }
    assert_eq!(n.get(), 4);
}

/// A NON-OOM failure is returned immediately — no retry (a corrupt GGUF fails
/// identically however it's loaded). Fail-on-revert: retrying regardless would
/// make more than one call.
#[tokio::test]
async fn ladder_does_not_retry_non_oom() {
    let n = std::cell::Cell::new(0);
    let r = run_oom_ladder(
        "org/model",
        np_count(40),
        None,
        std::time::Duration::ZERO,
        |_p| {
            n.set(n.get() + 1);
            async { Err(corrupt_err()) }
        },
    )
    .await;
    assert!(matches!(r, Err(HiggsError::EngineLoadFailed { .. })));
    assert_eq!(n.get(), 1, "non-OOM is not retried");
}

/// A different (non-OOM) fault surfacing on a degraded rung stops the ladder
/// and is surfaced as-is (the config change exposed a new problem).
#[tokio::test]
async fn ladder_stops_on_non_oom_mid_ladder() {
    let n = std::cell::Cell::new(0);
    let r = run_oom_ladder(
        "org/model",
        np_count(40),
        None,
        std::time::Duration::ZERO,
        |_p| {
            let i = n.get();
            n.set(i + 1);
            async move {
                if i == 0 {
                    Err(oom_err("first"))
                } else {
                    Err(corrupt_err()) // a rung reveals a non-memory fault
                }
            }
        },
    )
    .await;
    assert!(
        matches!(r, Err(HiggsError::EngineLoadFailed { reason, .. }) if reason.contains("tensors"))
    );
    assert_eq!(
        n.get(),
        2,
        "attempt 0 OOM + one rung that revealed the fault"
    );
}

// ── G6 Turbotune: run_benchmark orchestration (injected measure + cancel) ────

fn cand(label: &'static str) -> crate::tune::bench::Candidate {
    crate::tune::bench::Candidate {
        load: crate::worker::engine::llamacpp::params::LlamaCppParams::default(),
        label,
    }
}

fn bench(tps: f32) -> crate::tune::BenchResult {
    crate::tune::BenchResult {
        gen_tps: tps,
        ..Default::default()
    }
}

/// The orchestrator measures each candidate and returns the fastest.
#[tokio::test]
async fn benchmark_picks_the_fastest_candidate() {
    let cands = vec![cand("a"), cand("b"), cand("c")];
    let tps = std::cell::RefCell::new(vec![10.0f32, 25.0, 18.0].into_iter());
    let (benchmarked, result) = run_benchmark(
        cands,
        |_c| {
            let v = tps.borrow_mut().next().unwrap();
            async move { Ok(bench(v)) }
        },
        || false,
    )
    .await
    .expect("a benchmarked config");
    assert_eq!(benchmarked.label, "b", "b has the highest tps");
    assert_eq!(result.gen_tps, 25.0);
}

/// A candidate that FAILS to measure is skipped; a later one can still win.
#[tokio::test]
async fn benchmark_skips_failed_candidates() {
    let cands = vec![cand("bad"), cand("good")];
    let n = std::cell::Cell::new(0);
    let (benchmarked, _) = run_benchmark(
        cands,
        |_c| {
            let i = n.get();
            n.set(i + 1);
            async move {
                if i == 0 {
                    Err(HiggsError::EngineLoadFailed {
                        id: "x".into(),
                        reason: "oom".into(),
                    })
                } else {
                    Ok(bench(30.0))
                }
            }
        },
        || false,
    )
    .await
    .expect("the good one wins");
    assert_eq!(benchmarked.label, "good");
}

/// Every candidate failing yields the aggregate [HG063] with each named.
#[tokio::test]
async fn benchmark_all_fail_gives_aggregate_hg063() {
    let cands = vec![cand("one"), cand("two")];
    let err = run_benchmark(
        cands,
        |c| {
            let label = c.label;
            async move {
                Err(HiggsError::EngineLoadFailed {
                    id: label.into(),
                    reason: "nope".into(),
                })
            }
        },
        || false,
    )
    .await
    .expect_err("all failed");
    match err {
        HiggsError::BenchExhausted { detail } => {
            assert!(
                detail.contains("one") && detail.contains("two"),
                "names each: {detail}"
            );
        }
        other => panic!("expected HG063, got {other:?}"),
    }
}

/// A cancel signal that flips true aborts the run with [HG064] before the next
/// candidate. Fail-on-revert: never checking `cancelled` benches all candidates.
#[tokio::test]
async fn benchmark_cancels_between_candidates() {
    let cands = vec![cand("first"), cand("second")];
    let measured = std::cell::Cell::new(0);
    let err = run_benchmark(
        cands,
        |_c| {
            measured.set(measured.get() + 1);
            async move { Ok(bench(10.0)) }
        },
        // Cancel AFTER the first candidate is measured.
        || measured.get() >= 1,
    )
    .await
    .expect_err("cancelled");
    assert!(matches!(err, HiggsError::BenchCancelled));
    assert_eq!(measured.get(), 1, "the second candidate was never benched");
}

/// A benchmark's teardown (`bench_unload_id`) unloads ONLY the benched model's
/// worker — never other resident/serving models on a multi-model node. Fail-on-
/// revert: drain every worker (the old unscoped `bench_unload_all`) and the
/// unrelated model is evicted too.
#[tokio::test]
async fn bench_unload_id_leaves_other_models_resident() {
    let dir = tempfile::TempDir::new().unwrap();
    crate::serve::test_support::write_gguf_fixture(dir.path(), "org/a");
    crate::serve::test_support::write_gguf_fixture(dir.path(), "org/b");
    let higgs = fake_higgs(vec![dir.path().to_path_buf()]);
    higgs.load("org/a", None).await.expect("load a");
    higgs.load("org/b", None).await.expect("load b");

    // Scoped teardown for "org/a" must leave "org/b" resident.
    higgs.bench_unload_id("org/a").await;

    let resident: Vec<String> = higgs
        .local
        .instances()
        .await
        .into_iter()
        .map(|(_, m)| m)
        .collect();
    assert!(
        !resident.iter().any(|m| m == "org/a"),
        "the benched model's worker was unloaded"
    );
    assert!(
        resident.iter().any(|m| m == "org/b"),
        "the unrelated model must survive a scoped bench teardown: {resident:?}"
    );
}

/// A benchmark loads candidates through a DIRECT node load (not `load_inner`), so it
/// does NOT write a per-model `last_load` record to config for each candidate — the
/// bench persists only its winning tuning profile. Fail-on-revert: load candidates
/// via `load_inner` and the model gets a spurious `last_load` (the last benched
/// config) even though nothing was explicitly loaded.
#[tokio::test]
async fn benchmark_does_not_write_config_load_records() {
    let dir = tempfile::TempDir::new().unwrap();
    let id = stage_ollama_model(dir.path(), "tiny", "1b");
    let higgs = fake_higgs_ollama(vec![dir.path().to_path_buf()]);

    higgs
        .tune(TuneRequest {
            id: id.clone(),
            mode: Some(TuneMode::Benchmark),
            budget: None,
            pins: None,
        })
        .await
        .expect("benchmark");

    assert!(
        higgs
            .model_records()
            .get(&id)
            .and_then(|r| r.load.as_ref())
            .is_none(),
        "the bench's candidate loads must NOT write a last_load config record"
    );
}

/// A benchmark REFUSES to run on a LOADED model ([HG067]) — it would disrupt the
/// live worker and contaminate the measurement; the model must be unloaded first.
/// Fail-on-revert: drop the instances-loaded check in `turbotune_bench` and the
/// benchmark proceeds against the resident model instead of refusing.
#[tokio::test]
async fn benchmark_refuses_a_loaded_model() {
    let dir = tempfile::TempDir::new().unwrap();
    let id = stage_ollama_model(dir.path(), "tiny", "1b");
    let higgs = fake_higgs_ollama(vec![dir.path().to_path_buf()]);
    higgs.load(&id, None).await.expect("load");

    let err = higgs
        .tune(TuneRequest {
            id: id.clone(),
            mode: Some(TuneMode::Benchmark),
            budget: None,
            pins: None,
        })
        .await
        .expect_err("benchmark must refuse a loaded model");
    assert!(
        matches!(err, HiggsError::BenchModelLoaded { .. }),
        "expected HG067 BenchModelLoaded, got {err:?}"
    );
}

/// A public load REFUSES while the model is being benchmarked ([HG068]), and
/// succeeds again once the benchmark finishes. Fail-on-revert: drop the
/// `is_benchmarking` gate in `load_inner_impl` and the load proceeds against the
/// benchmarking model, racing the bench for it.
#[tokio::test]
async fn load_refuses_while_benchmarking() {
    let dir = tempfile::TempDir::new().unwrap();
    crate::serve::test_support::write_gguf_fixture(dir.path(), "org/model");
    let higgs = fake_higgs(vec![dir.path().to_path_buf()]);

    // Mark the model benchmarking (hold the guard) — a load must now refuse.
    let bench = higgs.begin_benchmark("org/model");
    let err = higgs
        .load("org/model", None)
        .await
        .expect_err("load must refuse while benchmarking");
    assert!(
        matches!(err, HiggsError::BenchInProgress { .. }),
        "expected HG068 BenchInProgress, got {err:?}"
    );

    // Once the benchmark ends, the load works again.
    drop(bench);
    higgs
        .load("org/model", None)
        .await
        .expect("load works after the benchmark releases the model");
}

/// `unload` (drain-all) REFUSES while any model is being benchmarked ([HG068]) —
/// a benchmark owns its candidate worker, and a drain would evict it mid-measure.
/// Fail-on-revert: drop the benchmarking guard in `unload` and it drains happily.
#[tokio::test]
async fn unload_all_refuses_while_benchmarking() {
    let higgs = fake_higgs(vec![]);
    let bench = higgs.begin_benchmark("org/model");
    let err = higgs
        .unload()
        .await
        .expect_err("unload-all must refuse while a benchmark runs");
    assert!(
        matches!(err, HiggsError::BenchInProgress { .. }),
        "expected HG068 BenchInProgress, got {err:?}"
    );
    // Releasing the benchmark lets the drain proceed again.
    drop(bench);
    higgs
        .unload()
        .await
        .expect("unload works after the release");
}

/// `unload_one` REFUSES ejecting a resident model that a benchmark owns ([HG068]),
/// so a per-card Eject can't evict the bench's candidate mid-measure. Fail-on-revert:
/// drop the `is_benchmarking` check in `unload_one` and the eject succeeds.
#[tokio::test]
async fn unload_one_refuses_while_benchmarking() {
    let dir = tempfile::TempDir::new().unwrap();
    crate::serve::test_support::write_gguf_fixture(dir.path(), "org/model");
    let higgs = fake_higgs(vec![dir.path().to_path_buf()]);
    higgs.load("org/model", None).await.expect("load");
    let bench = higgs.begin_benchmark("org/model");
    let err = higgs
        .unload_one("org/model")
        .await
        .expect_err("unload_one must refuse ejecting a benchmarking model");
    assert!(
        matches!(err, HiggsError::BenchInProgress { .. }),
        "expected HG068 BenchInProgress, got {err:?}"
    );
    drop(bench);
    higgs
        .unload_one("org/model")
        .await
        .expect("unload_one works after the release");
}

/// A SECOND benchmark for the same model REFUSES ([HG068]) instead of interleaving
/// with the first (their `bench_unload_id` calls would evict each other's candidate
/// workers). Fail-on-revert: drop the `is_benchmarking` check in `turbotune_bench`
/// and the second benchmark runs to completion (returns Ok) instead of erroring.
#[tokio::test]
async fn benchmark_refuses_while_already_benchmarking() {
    let dir = tempfile::TempDir::new().unwrap();
    let id = stage_ollama_model(dir.path(), "tiny", "1b");
    let higgs = fake_higgs_ollama(vec![dir.path().to_path_buf()]);
    // A benchmark already owns the model (guard held).
    let bench = higgs.begin_benchmark(&id);
    let err = higgs
        .tune(TuneRequest {
            id: id.clone(),
            mode: Some(TuneMode::Benchmark),
            budget: None,
            pins: None,
        })
        .await
        .expect_err("a second concurrent benchmark must refuse");
    assert!(
        matches!(err, HiggsError::BenchInProgress { .. }),
        "expected HG068 BenchInProgress, got {err:?}"
    );
    drop(bench);
}

/// A terminal `stop` (shutdown) aborts an in-flight Turbotune benchmark with
/// [HG064] rather than iterating candidates against the draining node. Here `stop`
/// runs FIRST (setting the shutdown flag + draining the node); the subsequent
/// benchmark trips the cancel hook before loading a single candidate. Fail-on-revert:
/// with the hook hard-coded `|| false`, the benchmark instead loads candidates
/// against the terminal node and every one fails → HG063 (BenchExhausted), not HG064.
#[tokio::test]
async fn benchmark_cancels_on_shutdown() {
    let dir = tempfile::TempDir::new().unwrap();
    let id = stage_ollama_model(dir.path(), "tiny", "1b");
    let higgs = fake_higgs_ollama(vec![dir.path().to_path_buf()]);
    // Terminal shutdown: sets `shutting_down` and drains the local node.
    higgs.stop().await;
    let err = higgs
        .tune(TuneRequest {
            id,
            mode: Some(TuneMode::Benchmark),
            budget: None,
            pins: None,
        })
        .await
        .expect_err("a benchmark racing shutdown must cancel");
    assert!(
        matches!(err, HiggsError::BenchCancelled),
        "shutdown aborts the benchmark with HG064 BenchCancelled, got {err:?}"
    );
}

// ── G6 Turbotune end-to-end via the fake worker (measure → persist Bench) ────

/// `tune` in Benchmark mode (Turbotune) does NOT fall back to Suggest: it LOADS +
/// MEASURES candidate configs through the stateful fake worker (`turbotune_bench`
/// → `run_benchmark` → `measure_gen_tps` → `bench_unload_all`) and saves the
/// FASTEST as a measured profile. Provenance flips to `Bench`, the persisted record
/// carries a positive `bench_tps` from a real (fake) decode, and the rationale gains
/// the "Turbotune measured …" line. Fail-on-revert: make Benchmark mode skip
/// `turbotune_bench` (Suggest fallback) and provenance stays `Heuristic`, `bench_tps`
/// stays `None`, and the line is absent. An `ollama/` id short-circuits the HF-card
/// fetch, so no network is touched (and no GPU is present, so every KV-variant
/// candidate fits → at least one gets benched).
#[tokio::test]
async fn tune_benchmark_measures_and_persists_bench_profile() {
    let dir = tempfile::TempDir::new().unwrap();
    let id = stage_ollama_model(dir.path(), "tiny", "1b");
    let higgs = fake_higgs_ollama(vec![dir.path().to_path_buf()]);

    let suggestion = higgs
        .tune(TuneRequest {
            id: id.clone(),
            mode: Some(TuneMode::Benchmark),
            budget: None,
            pins: None,
        })
        .await
        .expect("turbotune benchmark measures a config");

    assert_eq!(
        suggestion.provenance,
        crate::tune::TuneProvenance::Bench,
        "a measured benchmark stamps Bench provenance (not the Suggest fallback)"
    );
    assert!(
        suggestion
            .rationale
            .iter()
            .any(|r| r.contains("Turbotune measured")),
        "the measured throughput is explained in the rationale: {:?}",
        suggestion.rationale
    );

    // The persisted saved profile carries the MEASURED tok/s (a real fake decode
    // streams 3 completion tokens in >0 seconds, so gen_tps is strictly positive).
    let rec = higgs
        .models_store()
        .expect("models store opens")
        .tuning(&id)
        .expect("bench profile persisted");
    assert_eq!(
        rec.provenance,
        crate::tune::TuneProvenance::Bench,
        "the saved profile is the measured benchmarked config"
    );
    let tps = rec
        .bench_tps
        .expect("bench_tps recorded on the saved profile");
    assert!(
        tps > 0.0,
        "measured a positive generation throughput: {tps}"
    );
}

/// Request pins actually reach the benchmark: a `ctx_len` pinned on the
/// `TuneRequest` must be carried by the measured benchmarked config AND the persisted
/// profile. Fail-on-revert for the request-to-bench plumbing in `tune()`
/// (`let pins = req.pins...` feeding `turbotune_bench`): reverting it to
/// `TunePins::default()` (or dropping the arg) makes the persisted ctx fall
/// back to the analytical suggestion — 4242 is not a value the suggester
/// derives.
#[tokio::test]
async fn tune_benchmark_honors_request_pins_end_to_end() {
    use crate::worker::engine::CtxLen;
    let dir = tempfile::TempDir::new().unwrap();
    let id = stage_ollama_model(dir.path(), "tiny", "1b");
    let higgs = fake_higgs_ollama(vec![dir.path().to_path_buf()]);

    let suggestion = higgs
        .tune(TuneRequest {
            id: id.clone(),
            mode: Some(TuneMode::Benchmark),
            budget: None,
            pins: Some(crate::tune::TunePins {
                ctx_len: Some(CtxLen::Fixed { n: 4242 }),
                ..crate::tune::TunePins::default()
            }),
        })
        .await
        .expect("pinned turbotune benchmark succeeds");

    assert_eq!(
        suggestion.load.as_llamacpp().ctx_len,
        CtxLen::Fixed { n: 4242 },
        "the measured benchmarked config carries the pinned ctx_len"
    );
    // The rationale must describe the MEASURED WINNER, not the analytical seed:
    // no "context N" line may contradict the pinned/saved 4242 (P5). The seed's
    // analytical context (budget-derived, never 4242) would surface here if the
    // benchmark still appended to — rather than replaced — the seed's rationale.
    let ctx_lines: Vec<&String> = suggestion
        .rationale
        .iter()
        .filter(|l| l.starts_with("context "))
        .collect();
    assert!(
        !ctx_lines.is_empty(),
        "the benchmark rationale narrates the benchmarked's context: {:?}",
        suggestion.rationale
    );
    for l in &ctx_lines {
        assert!(
            l.contains("4242"),
            "no rationale context line may contradict the pinned benchmarked: {l:?}"
        );
    }
    let rec = higgs
        .models_store()
        .expect("models store opens")
        .tuning(&id)
        .expect("pinned bench profile persisted");
    assert_eq!(
        rec.profile.as_llamacpp().ctx_len,
        CtxLen::Fixed { n: 4242 },
        "the persisted profile carries the pinned ctx_len"
    );
}

// ── Readiness / servable / fit: the JIT-serving classification ────────────────

/// Seed a FRESH tuning profile straight into the per-instance store, anchored to
/// the CURRENT hardware fingerprint + on-disk model file signature so
/// `profile_stale` reads it as NOT stale (exactly like a real Prepare). `ctx` keeps
/// the KV footprint small so the RAM fit is dominated by the fixed compute overhead
/// (fits any dev host with free RAM). No llama.cpp — only the staleness anchors and
/// the load params matter to the readiness math.
async fn seed_fresh_profile(higgs: &Higgs, id: &str, ctx: CtxLen) {
    let path = higgs
        .scan()
        .await
        .ok()
        .and_then(|ms| ms.into_iter().find(|m| m.id == id).map(|m| m.path))
        .expect("fixture model is scannable");
    seed_fresh_profile_at(higgs, id, ctx, &path).await;
}

/// [`seed_fresh_profile`] anchored to an EXPLICIT file — `profile_stale` keys
/// freshness on the file signature, so a multi-variant id needs the record to
/// name which variant it was tuned against.
async fn seed_fresh_profile_at(higgs: &Higgs, id: &str, ctx: CtxLen, path: &str) {
    use crate::worker::engine::LoadParams;
    let hw = higgs.hardware().await;
    let store = higgs.models_store().expect("open models store");
    store.put_tuning(
        id,
        crate::tune::store::TuneRecord {
            profile: LoadParams::base(ctx, GpuLayers::All, 8),
            sampling: Default::default(),
            budget: Default::default(),
            provenance: crate::tune::TuneProvenance::Heuristic,
            bench_tps: None,
            tuned_at_ms: 1,
            hw_fingerprint: hw.fingerprint(),
            model_file_sig: file_sig(path),
        },
    );
    store.flush().expect("persist seeded profile");
}

/// The readiness classifier + `servable_model_ids` reflect the LIVE serving toggle
/// and residency. A fresh, fitting, non-resident profile with serving on is
/// `Servable` (a valid JIT target, surfaced by `servable_model_ids`) and carries a
/// `ModelFit`; turning serving OFF makes it `Profiled` (no fit, dropped from the
/// servable list); loading it makes it `Loaded` (dropped again — it's already
/// resident). Fail-on-revert: break the serving/loaded precedence in
/// `derive_readiness` (or drop the servable filter) and one of these transitions
/// misclassifies.
#[tokio::test]
async fn servable_readiness_reflects_serving_and_residency() {
    let dir = tempfile::TempDir::new().unwrap();
    crate::serve::test_support::write_gguf_fixture(dir.path(), "org/model");
    let higgs = fake_higgs(vec![dir.path().to_path_buf()]);
    seed_fresh_profile(&higgs, "org/model", CtxLen::Fixed { n: 512 }).await;

    let model = higgs
        .scan()
        .await
        .unwrap()
        .into_iter()
        .find(|m| m.id == "org/model")
        .expect("scanned");
    let hw = higgs.hardware().await;
    let tuning =
        crate::serve::control::active_records(&higgs.tuning_profiles().expect("tuning profiles"));

    // Servable: profiled, fresh, fits free resources, serving on, not resident.
    let (readiness, fit) = higgs.model_readiness(&model, &[], &hw, &tuning);
    assert_eq!(
        readiness,
        crate::serve::readiness::ModelReadiness::Servable,
        "fresh + fitting + serving on + not resident → Servable"
    );
    let fit = fit.expect("a servable model surfaces its needed-vs-free fit numbers");
    assert!(
        fit.needed_ram_bytes > 0,
        "the RAM footprint was actually computed"
    );
    assert_eq!(
        higgs.servable_model_ids().await,
        vec!["org/model".to_owned()],
        "a Servable model is advertised as a JIT target"
    );

    // Serving OFF → Profiled (cannot serve now), no fit computed, dropped from the list.
    higgs.set_serving_enabled(false);
    let (readiness, fit) = higgs.model_readiness(&model, &[], &hw, &tuning);
    assert_eq!(
        readiness,
        crate::serve::readiness::ModelReadiness::Profiled,
        "serving off → Profiled"
    );
    assert!(fit.is_none(), "no fit is computed while serving is off");
    assert!(
        higgs.servable_model_ids().await.is_empty(),
        "serving off → nothing is servable"
    );
    higgs.set_serving_enabled(true);

    // Resident → Loaded (precedence over fit), and no longer a JIT target.
    higgs.load("org/model", None).await.expect("load");
    let loaded_set = vec!["org/model".to_owned()];
    let (readiness, _) = higgs.model_readiness(&model, &loaded_set, &hw, &tuning);
    assert_eq!(
        readiness,
        crate::serve::readiness::ModelReadiness::Loaded,
        "a resident model is Loaded regardless of fit"
    );
    assert!(
        higgs.servable_model_ids().await.is_empty(),
        "a resident model isn't re-advertised as servable"
    );
}

/// The readiness classifier's Discovered (no profile) and NeedsRetune (stale
/// profile) branches. A scanned model with NO saved profile is `Discovered` (must
/// Prepare); a saved profile whose hardware fingerprint no longer matches is
/// `NeedsRetune` (hard-blocks serving). Neither is servable. Fail-on-revert: drop
/// the staleness check in `model_readiness` and the stale profile reads `Servable`
/// instead of `NeedsRetune`.
#[tokio::test]
async fn model_readiness_discovered_and_needs_retune() {
    let dir = tempfile::TempDir::new().unwrap();
    crate::serve::test_support::write_gguf_fixture(dir.path(), "org/model");
    let higgs = fake_higgs(vec![dir.path().to_path_buf()]);

    let model = higgs
        .scan()
        .await
        .unwrap()
        .into_iter()
        .find(|m| m.id == "org/model")
        .expect("scanned");
    let hw = higgs.hardware().await;

    // No profile → Discovered, not servable.
    let tuning =
        crate::serve::control::active_records(&higgs.tuning_profiles().expect("tuning profiles"));
    let (readiness, fit) = higgs.model_readiness(&model, &[], &hw, &tuning);
    assert_eq!(
        readiness,
        crate::serve::readiness::ModelReadiness::Discovered,
        "on disk, no profile → Discovered"
    );
    assert!(fit.is_none());
    assert!(higgs.servable_model_ids().await.is_empty());

    // A STALE profile (mismatched hardware fingerprint, right file sig) → NeedsRetune.
    let store = higgs.models_store().expect("store");
    store.put_tuning(
        "org/model",
        crate::tune::store::TuneRecord {
            profile: crate::worker::engine::LoadParams::base(
                CtxLen::Fixed { n: 512 },
                GpuLayers::All,
                8,
            ),
            sampling: Default::default(),
            budget: Default::default(),
            provenance: crate::tune::TuneProvenance::Heuristic,
            bench_tps: None,
            tuned_at_ms: 1,
            hw_fingerprint: "some-other-machine".into(),
            model_file_sig: file_sig(&model.path),
        },
    );
    store.flush().expect("flush");

    let tuning =
        crate::serve::control::active_records(&higgs.tuning_profiles().expect("tuning profiles"));
    let (readiness, fit) = higgs.model_readiness(&model, &[], &hw, &tuning);
    assert_eq!(
        readiness,
        crate::serve::readiness::ModelReadiness::NeedsRetune,
        "a hardware-mismatched profile → NeedsRetune"
    );
    assert!(fit.is_none(), "no fit for a stale profile");
    assert!(
        higgs.servable_model_ids().await.is_empty(),
        "a stale profile is not servable"
    );
}

/// `profile_state` (the JIT gate's freshness verdict): `Missing` with no record,
/// `Ready(profile)` for a fresh matching record (carrying the validated params),
/// `Stale` when the hardware fingerprint no longer matches, and a store-OPEN
/// failure surfaces as [HG040] `PersistenceFailed` — NOT a misleading `Missing`.
/// Fail-on-revert: map the store-open error to `Missing` and the last assertion
/// (an Err) fails; drop the staleness check and the stale record reads `Ready`.
#[tokio::test]
async fn profile_state_missing_ready_stale_and_store_fault() {
    let dir = tempfile::TempDir::new().unwrap();
    crate::serve::test_support::write_gguf_fixture(dir.path(), "org/model");
    let higgs = fake_higgs(vec![dir.path().to_path_buf()]);

    // No record → Missing.
    assert!(
        matches!(
            higgs.profile_state("org/model").await.unwrap(),
            ProfileState::Missing
        ),
        "no saved profile → Missing"
    );

    // Fresh matching record → Ready, carrying the validated load params.
    seed_fresh_profile(&higgs, "org/model", CtxLen::Fixed { n: 512 }).await;
    match higgs.profile_state("org/model").await.unwrap() {
        ProfileState::Ready(p) => assert_eq!(
            p.ctx_len(),
            CtxLen::Fixed { n: 512 },
            "Ready carries the validated profile params"
        ),
        other => panic!("expected Ready, got {other:?}"),
    }

    // Overwrite with a stale record (mismatched hardware fingerprint) → Stale.
    let path = higgs
        .scan()
        .await
        .unwrap()
        .into_iter()
        .find(|m| m.id == "org/model")
        .map(|m| m.path)
        .unwrap();
    let store = higgs.models_store().expect("store");
    store.put_tuning(
        "org/model",
        crate::tune::store::TuneRecord {
            profile: crate::worker::engine::LoadParams::base(
                CtxLen::Fixed { n: 512 },
                GpuLayers::All,
                8,
            ),
            sampling: Default::default(),
            budget: Default::default(),
            provenance: crate::tune::TuneProvenance::Heuristic,
            bench_tps: None,
            tuned_at_ms: 1,
            hw_fingerprint: "some-other-machine".into(),
            model_file_sig: file_sig(&path),
        },
    );
    store.flush().expect("flush");
    assert!(
        matches!(
            higgs.profile_state("org/model").await.unwrap(),
            ProfileState::Stale
        ),
        "a hardware-mismatched profile → Stale"
    );

    // A store that can't be OPENED (home is a regular FILE, so `models.json` under
    // it is ENOTDIR) surfaces as HG040 — a persistence fault, not `Missing`.
    let blocker = dir.path().join("blocker-file");
    std::fs::write(&blocker, b"x").unwrap();
    *higgs.config_path.lock() = Some(blocker.join("config.json"));
    let err = higgs
        .profile_state("org/model")
        .await
        .expect_err("an unopenable store is a fault, not Missing");
    assert!(
        matches!(err, HiggsError::PersistenceFailed { .. }),
        "store-open failure → HG040 PersistenceFailed, got {err}"
    );
}

/// `estimate` returns the VRAM+RAM footprint for a candidate load and MEMOIZES the
/// resolved GGUF metadata per id (the second call is a cache hit — no re-scan), and
/// an unknown id is [HG002] ModelNotFound. Fail-on-revert: drop the meta memo and
/// the second call still works (so the cache-hit branch is what this exercises for
/// coverage); a broken scan-find would 404 a present model.
#[tokio::test]
async fn estimate_returns_footprint_and_memoizes_meta() {
    let dir = tempfile::TempDir::new().unwrap();
    crate::serve::test_support::write_gguf_fixture(dir.path(), "org/model");
    let higgs = fake_higgs(vec![dir.path().to_path_buf()]);

    let req = EstimateRequest {
        id: "org/model".to_owned(),
        ctx_len: CtxLen::Fixed { n: 2048 },
        gpu_layers: None,
        type_k: None,
        type_v: None,
        offload_kqv: None,
        cpu_moe: None,
        budget: None,
    };
    let first = higgs.estimate(req.clone()).await.expect("estimate");
    assert!(
        first.ram.needed_bytes > 0,
        "the RAM footprint is actually computed"
    );
    // The per-id metadata memo is now primed; a second call hits it (same result).
    let second = higgs.estimate(req).await.expect("estimate (cache hit)");
    assert_eq!(second, first, "a cache hit returns the same footprint");

    // Unknown id → ModelNotFound (the scan-find miss).
    let err = higgs
        .estimate(EstimateRequest {
            id: "org/absent".to_owned(),
            ctx_len: CtxLen::Auto,
            gpu_layers: None,
            type_k: None,
            type_v: None,
            offload_kqv: None,
            cpu_moe: None,
            budget: None,
        })
        .await
        .expect_err("unknown model → ModelNotFound");
    assert!(matches!(err, HiggsError::ModelNotFound { .. }), "got {err}");
}

/// A REJECTED (non-mutating) key mutation must NOT write the keystore: an unwritable
/// `api_keys.json` would otherwise turn the intended 4xx (unauthorized mint, duplicate
/// label, unknown revoke, last-key/admin conflict) into an HG040 500, and rejected
/// requests would do pointless disk writes. Fail-on-revert: the old unconditional
/// `keys.save(...)` writes the store even when the closure changed nothing, so the
/// file appears.
#[tokio::test]
#[allow(clippy::await_holding_lock)] // TEST_ENV_LOCK spans the test (serializes HIGGS_HOME)
async fn rejected_key_mutation_does_not_write_the_keystore() {
    let _env = crate::TEST_ENV_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let home = tempfile::tempdir().unwrap();
    let prev = std::env::var_os("HIGGS_HOME");
    // SAFETY: serialized by TEST_ENV_LOCK; restored below.
    unsafe { std::env::set_var("HIGGS_HOME", home.path()) };

    let higgs = fake_higgs(vec![]);
    // A closure that REJECTS without touching the store — the keystore is unchanged.
    let decision = higgs
        .mutate_api_keys(|_ks| "rejected")
        .expect("a rejected mutation returns Ok(decision), never an io error");
    assert_eq!(decision, "rejected");
    assert!(
        !home.path().join("api_keys.json").exists(),
        "a rejected (non-mutating) key op must NOT write api_keys.json"
    );

    // Sanity (the guard still persists a REAL change): an accepting mutation writes.
    higgs
        .mutate_api_keys(|ks| {
            ks.add(
                &crate::keys::mint_token([3u8; 16]),
                "ci".into(),
                vec![crate::keys::Scope::Chat],
            );
        })
        .expect("accept persists");
    assert!(
        home.path().join("api_keys.json").exists(),
        "an accepting mutation DOES write api_keys.json"
    );

    // SAFETY: serialized by TEST_ENV_LOCK.
    unsafe {
        match prev {
            Some(v) => std::env::set_var("HIGGS_HOME", v),
            None => std::env::remove_var("HIGGS_HOME"),
        }
    }
}

/// `touch_api_key`'s throttle authority is the IN-MEMORY map, not the stored
/// `last_used_ms`. Fail-on-revert for round-1 finding: pre-seeding the in-memory
/// throttle with a fresh stamp must suppress the live-store update even though
/// the stored `last_used_ms` is still `None`. The old code read the stored
/// value, so a null stamp was never "fresh" and every request re-ran the
/// update — defeating the throttle exactly in the always-null case.
#[tokio::test]
#[allow(clippy::await_holding_lock)] // TEST_ENV_LOCK spans the test (serializes HIGGS_HOME)
async fn touch_throttle_is_the_in_memory_map_not_the_persisted_stamp() {
    let _env = crate::TEST_ENV_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let home = tempfile::tempdir().unwrap();
    let prev = std::env::var_os("HIGGS_HOME");
    // SAFETY: serialized by TEST_ENV_LOCK; restored below.
    unsafe { std::env::set_var("HIGGS_HOME", home.path()) };

    let higgs = fake_higgs(vec![]);
    let mut ks = crate::keys::ApiKeys::default();
    let sha = ks.add(
        &crate::keys::mint_token([7u8; 16]),
        "laptop".into(),
        vec![crate::keys::Scope::Chat],
    );
    higgs.set_api_keys(std::sync::Arc::new(ks));
    // Never used yet: the persisted stamp is None.
    assert_eq!(higgs.api_keys().iter().next().unwrap().last_used_ms, None);

    // Pre-seed the in-memory throttle with a FRESH stamp; the persisted
    // last_used_ms stays None. A touch must be throttled by the map.
    higgs
        .key_touch_throttle
        .lock()
        .insert(sha.clone(), now_unix_ms());
    higgs.touch_api_key(&sha);

    assert_eq!(
        higgs.api_keys().iter().next().unwrap().last_used_ms,
        None,
        "a fresh in-memory throttle stamp must suppress the live-store update even with a null last_used_ms"
    );

    // SAFETY: serialized by TEST_ENV_LOCK.
    unsafe {
        match prev {
            Some(v) => std::env::set_var("HIGGS_HOME", v),
            None => std::env::remove_var("HIGGS_HOME"),
        }
    }
}

/// `touch_api_key` updates last-used IN MEMORY ONLY and must never rewrite
/// `api_keys.json`. Fail-on-revert for round-2 finding: the `higgs keys add`
/// CLI writes the keystore directly (blessing a deferred restart), so a
/// touch-driven full-file rewrite from the live store — which never learned of
/// the CLI key — would silently ERASE it from disk before the operator's
/// restart. Here a key added out-of-band to the file survives a touch; the old
/// `mutate_api_keys(|ks| ks.touch(...))` persist deletes it.
///
/// (Not cleanly reachable over HTTP: the 60s per-key touch throttle means the
/// only live-store key is already touched during spawn readiness, so no
/// touch-save fires within a test window without a forbidden test-only knob.
/// A unit call on a fresh throttle exercises the exact persist decision.)
#[tokio::test]
#[allow(clippy::await_holding_lock)] // TEST_ENV_LOCK spans the test (serializes HIGGS_HOME)
async fn touch_never_rewrites_the_keystore_file() {
    let _env = crate::TEST_ENV_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let home = tempfile::tempdir().unwrap();
    let prev = std::env::var_os("HIGGS_HOME");
    // SAFETY: serialized by TEST_ENV_LOCK; restored below.
    unsafe { std::env::set_var("HIGGS_HOME", home.path()) };
    let keystore = home.path().join("api_keys.json");

    // Live store (as loaded at boot) holds only the admin key; the file matches.
    let mut live = crate::keys::ApiKeys::default();
    let sha = live.add(
        &crate::keys::mint_token([1u8; 16]),
        "admin".into(),
        vec![crate::keys::Scope::Admin],
    );
    live.save(&keystore).expect("seed keystore");
    let higgs = fake_higgs(vec![]);
    higgs.set_api_keys(std::sync::Arc::new(live));

    // Out-of-band: `higgs keys add laptop2` writes the FILE directly — the live
    // store does NOT know this key.
    let mut disk = crate::keys::ApiKeys::load(&keystore).unwrap();
    disk.add(
        &crate::keys::mint_token([2u8; 16]),
        "laptop2".into(),
        vec![crate::keys::Scope::Chat],
    );
    disk.save(&keystore).unwrap();

    // A touch on a FRESH throttle fires the persist decision.
    higgs.touch_api_key(&sha);

    // In-memory: admin's last-used advanced (visible to GET /keys).
    assert!(
        higgs
            .api_keys()
            .iter()
            .find(|k| k.sha256 == sha)
            .and_then(|k| k.last_used_ms)
            .is_some(),
        "touch stamps last-used on the live store"
    );
    // On disk: laptop2 MUST survive — a touch never rewrites the file.
    let on_disk = crate::keys::ApiKeys::load(&keystore).unwrap();
    assert!(
        on_disk.iter().any(|k| k.label == "laptop2"),
        "touch must not erase an out-of-band key; on-disk labels: {:?}",
        on_disk.iter().map(|k| &k.label).collect::<Vec<_>>()
    );

    // SAFETY: serialized by TEST_ENV_LOCK.
    unsafe {
        match prev {
            Some(v) => std::env::set_var("HIGGS_HOME", v),
            None => std::env::remove_var("HIGGS_HOME"),
        }
    }
}

/// PUBLIC `chat_stream` must SKIP a locally-resident worker whose model is being
/// benchmarked — it is a transient Turbotune candidate (the serve gate routed the
/// request elsewhere), and the bench unloads it between candidates. With no remote
/// fleet the skip falls through to ModelNotFound rather than streaming from it. The
/// bench's OWN measurement (`serve_public=false`, via `chat_stream_inner`) is exempt
/// and still reaches its candidate — covered by the end-to-end benchmark test.
/// Fail-on-revert: drop the benchmark filter in `chat_stream_inner` and the public
/// chat serves the bench-owned worker (Ok), failing this `expect_err`.
#[tokio::test]
async fn public_chat_skips_a_benchmarking_local_worker() {
    let dir = tempfile::TempDir::new().unwrap();
    crate::serve::test_support::write_gguf_fixture(dir.path(), "org/model");
    let higgs = fake_higgs(vec![dir.path().to_path_buf()]);
    higgs.load("org/model", None).await.expect("load candidate");
    // The race state: the id is resident as a transient benchmark candidate.
    let _bench = higgs.begin_benchmark("org/model");
    let err = higgs
        .chat_stream(
            "org/model".to_owned(),
            r#"[{"role":"user","content":"hi"}]"#.to_owned(),
            8,
            greedy_sampling(),
            None,
            None,
        )
        .await
        .expect_err("public chat must skip a benchmarking local worker");
    assert!(
        matches!(err, HiggsError::ModelNotFound { .. }),
        "skips the local bench candidate → ModelNotFound (no remote), got {err:?}"
    );
}

/// A `Higgs` whose loads OOM twice then degrade (fake `oom_twice`), for the G5
/// degraded-reuse-persist test.
fn fake_higgs_oom_twice(dirs: Vec<PathBuf>) -> Higgs {
    let node = crate::node::test_support::fake_runtime_oom_twice(dirs.clone());
    let cfg = HiggsConfig {
        lmstudio_dirs: dirs,
        hf_dirs: vec![],
        ollama_dirs: vec![],
        default_load: HiggsConfig::default().default_load,
        worker_exe: None,
    };
    Higgs::with_local(Arc::new(node), cfg)
}

/// A REUSE load (a saved profile, `ReuseSaved`) that the OOM ladder has to
/// DEGRADE must write the FITTING config back to the saved tuning profile — else every
/// reload re-reads the OOMing profile and re-walks the ladder (codex r10). Fail-on-revert:
/// restore `if let Some(anchors) { sync }` (sync only for explicit loads) and the degraded
/// reuse leaves the profile as the OOMing seed, so the `assert_ne` fails.
#[tokio::test]
async fn degraded_reuse_load_persists_the_fitting_profile() {
    let dir = tempfile::TempDir::new().unwrap();
    crate::serve::test_support::write_gguf_fixture(dir.path(), "org/model");
    let higgs = fake_higgs_oom_twice(vec![dir.path().to_path_buf()]);
    // Seed a saved tuning profile with the ORIGINAL (OOMing) params + Bench provenance.
    let seed = LoadParams::base(CtxLen::Auto, GpuLayers::All, 8);
    let store = higgs.models_store().unwrap();
    store.put_tuning(
        "org/model",
        crate::tune::store::TuneRecord {
            profile: seed.clone(),
            sampling: Default::default(),
            budget: Default::default(),
            provenance: crate::tune::TuneProvenance::Bench,
            bench_tps: Some(42.0),
            tuned_at_ms: 1,
            hw_fingerprint: "old".into(),
            model_file_sig: "old".into(),
        },
    );
    store.flush().unwrap();
    // REUSE the saved profile: OOMs twice, then the ladder loads a DEGRADED rung.
    higgs
        .load_inner("org/model", Some(seed.clone()), LoadPersist::ReuseSaved)
        .await
        .expect("degraded reuse load succeeds");
    let rec = higgs
        .models_store()
        .unwrap()
        .tuning("org/model")
        .expect("profile still present");
    assert_ne!(
        rec.profile, seed,
        "degraded reuse load persisted the FITTING config, not the OOMing seed"
    );
    // The degraded fallback was NOT benchmarked — the store must NOT keep the old Bench
    // provenance / bench_tps for it (codex r11). Fail-on-revert: preserve them in
    // `set_profile` (clear_bench=false) and these stay Bench/Some(42.0), failing here.
    assert_eq!(
        rec.provenance,
        crate::tune::TuneProvenance::Heuristic,
        "degraded config: stale Bench provenance cleared"
    );
    assert_eq!(
        rec.bench_tps, None,
        "degraded config: stale bench_tps cleared"
    );
}

/// The bench's OWN measurement (`serve_public = false`) must NOT compete with public
/// chats for the admission gate: with the gate saturated by chats on OTHER models, a
/// public chat refuses (ServerBusy) but the measurement still reaches its candidate —
/// otherwise gate contention masquerades as an HG065 candidate failure and skews
/// `pick_benchmarked` (Fable r1). Fail-on-revert: acquire the permit unconditionally in
/// `chat_stream_inner` and the measurement returns ServerBusy, failing the expect.
#[tokio::test]
async fn bench_measurement_bypasses_the_public_admission_gate() {
    let dir = tempfile::TempDir::new().unwrap();
    crate::serve::test_support::write_gguf_fixture(dir.path(), "org/model");
    let higgs = fake_higgs(vec![dir.path().to_path_buf()]);
    higgs.load("org/model", None).await.expect("load");

    // Saturate the PUBLIC admission gate (as 8 in-flight chats on other models would).
    let mut held = Vec::new();
    for _ in 0..MAX_CONCURRENT_INFERENCE {
        held.push(
            Arc::clone(&higgs.inference_gate)
                .try_acquire_owned()
                .expect("saturate gate"),
        );
    }

    // Public chat is refused — the gate is doing its job.
    let err = higgs
        .chat_stream(
            "org/model".to_owned(),
            r#"[{"role":"user","content":"hi"}]"#.to_owned(),
            8,
            greedy_sampling(),
            None,
            None,
        )
        .await
        .expect_err("public chat refuses on a full gate");
    assert!(matches!(err, HiggsError::ServerBusy { .. }), "got {err:?}");

    // The bench measurement is UNGATED and completes end-to-end.
    let (mut rx, handle) = higgs
        .chat_stream_inner(
            "org/model".to_owned(),
            r#"[{"role":"user","content":"hi"}]"#.to_owned(),
            8,
            greedy_sampling(),
            None,
            None,
            false,
        )
        .await
        .expect("bench measurement bypasses the full public gate");
    while rx.recv().await.is_some() {}
    let outcome = handle.await.expect("join").expect("chat ok");
    assert_eq!(
        outcome.content, "hello",
        "fake worker served the measurement"
    );
}

/// A benchmark future dropped mid-candidate must NOT leak the candidate worker as a
/// publicly-servable model (codex r15) — AND the [HG068] flag must OUTLIVE the
/// reclaim (codex r16): clearing it before the unload commits would open a window
/// where the doomed candidate is registered but no longer gated, so a public chat
/// could adopt it just before the unload kills it. This polls the invariant
/// "not-benchmarking ⇒ no worker resident" at every step. Fail-on-revert: clear the
/// flag synchronously in the guard's drop (the old two-guard order) and the first
/// poll sees the flag down while the worker is still registered.
#[tokio::test]
async fn dropped_benchmark_reclaims_the_live_candidate_worker() {
    let dir = tempfile::TempDir::new().unwrap();
    crate::serve::test_support::write_gguf_fixture(dir.path(), "org/model");
    let higgs = fake_higgs(vec![dir.path().to_path_buf()]);
    higgs.load("org/model", None).await.expect("load candidate");
    let (worker, _) = higgs.local.instances().await[0];

    // The exact state a dropped benchmark future leaves behind: the guard's
    // live-candidate slot holds the worker and the guard drops without a normal exit.
    let guard = higgs.begin_benchmark("org/model");
    *guard.live.lock() = Some(worker);
    drop(guard);

    // The spawned reclaim unloads the candidate and ONLY THEN clears the flag.
    let mut reclaimed = false;
    for _ in 0..100 {
        // Read the flag FIRST: the guard clears it strictly AFTER the unload commits,
        // so observing the flag down guarantees the worker is already gone at any
        // LATER read. (Reading residency first would race the reclaim completing
        // between the two reads and trip a spurious failure.)
        let gated = higgs.is_benchmarking("org/model");
        let resident = !higgs.local.instances().await.is_empty();
        assert!(
            gated || !resident,
            "the HG068 gate must outlive the reclaim — never registered-but-ungated"
        );
        if !resident && !gated {
            reclaimed = true;
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    assert!(reclaimed, "worker reclaimed and flag cleared");

    // Normal-exit path: an EMPTY slot clears the flag synchronously and touches no
    // worker — load a fresh (user) worker, drop an empty guard, worker survives.
    higgs.load("org/model", None).await.expect("user reload");
    drop(higgs.begin_benchmark("org/model"));
    assert!(
        !higgs.is_benchmarking("org/model"),
        "sync clear on normal exit"
    );
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    assert_eq!(
        higgs.local.instances().await.len(),
        1,
        "an empty guard never stomps a user worker"
    );
}

/// Candidates that "succeed" at 0 tok/s measured NOTHING (immediate EOS / empty
/// decode window): an all-zero run must surface as [HG063] BenchExhausted naming
/// every candidate — never crown the first candidate by ordering and persist a
/// Bench profile claiming a measurement (Fable r8). Fail-on-revert: accept Ok
/// results unconditionally in `run_benchmark` and this returns Ok(0.0 benchmarked).
#[tokio::test]
async fn all_zero_measurements_exhaust_the_benchmark() {
    let err = run_benchmark(
        vec![cand("a"), cand("b")],
        |_c| async move { Ok(bench(0.0)) },
        || false,
    )
    .await
    .expect_err("all-zero measurements must exhaust, not crown a benchmarked config");
    match err {
        HiggsError::BenchExhausted { detail } => {
            assert!(
                detail.contains("a: no measurable decode") && detail.contains("b:"),
                "detail names each zero-scored candidate: {detail}"
            );
        }
        other => panic!("expected HG063 BenchExhausted, got {other:?}"),
    }
    // A mixed run still works: the positive measurement wins, zeros are failures.
    let vals = std::cell::RefCell::new(vec![0.0f32, 7.5].into_iter());
    let (benchmarked, result) = run_benchmark(
        vec![cand("a"), cand("b")],
        |_c| {
            let v = vals.borrow_mut().next().unwrap();
            async move { Ok(bench(v)) }
        },
        || false,
    )
    .await
    .expect("the positive candidate wins");
    assert_eq!(benchmarked.label, "b");
    assert!(result.gen_tps > 7.0);
}

/// Benchmark candidates must respect the RAM budget too (codex r20): with an
/// explicit 1-byte RAM cap, EVERY candidate RAM-overflows, so the benchmark must
/// refuse with [HG063] "no candidate configs fit" — never load, measure, and
/// persist a config the RAM estimator calls Overflow. Fail-on-revert: filter on
/// the VRAM estimator alone (CPU-only budget_bytes=0 → passes) and the benchmark
/// runs to a persisted Bench "success".
#[tokio::test]
async fn benchmark_candidates_respect_the_ram_budget() {
    let dir = tempfile::TempDir::new().unwrap();
    let id = stage_ollama_model(dir.path(), "tiny", "1b");
    let higgs = fake_higgs_ollama(vec![dir.path().to_path_buf()]);
    let err = higgs
        .tune(TuneRequest {
            id,
            mode: Some(TuneMode::Benchmark),
            budget: Some(crate::tune::ResourceBudget {
                max_ram_bytes: Some(1),
                ..Default::default()
            }),
            pins: None,
        })
        .await
        .expect_err("a 1-byte RAM cap must exhaust the candidate set");
    assert!(
        matches!(err, HiggsError::BenchExhausted { .. }),
        "expected HG063 BenchExhausted, got {err:?}"
    );
}

/// A per-model unload for a model that is benchmarking BETWEEN candidates
/// (nothing resident) must refuse with [HG068] — not return a false success while
/// the benchmark keeps running (codex r20). Fail-on-revert: gate only on the
/// resolved resident worker and this returns Ok.
#[tokio::test]
async fn unload_one_refuses_between_candidates() {
    let higgs = fake_higgs(vec![]);
    let _bench = higgs.begin_benchmark("org/model");
    let err = higgs
        .unload_one("org/model")
        .await
        .expect_err("between-candidates per-model unload must refuse");
    assert!(
        matches!(err, HiggsError::BenchInProgress { .. }),
        "expected HG068 BenchInProgress, got {err:?}"
    );
}

/// A RESIDENT embedding model is refused for chat and not advertised. Explicit
/// loads of an embedding model are ALLOWED (the domain gate is a chat gate, not a
/// load gate — a future /v1/embeddings serves it), so residency is a reachable
/// state — and it must not re-open the chat door the scan-time classification
/// closed. Pins the two spots the integration test cannot reach with its
/// header-only fixture (which no real engine can load): the resident-path
/// `[HG079]` arm of `resolve_loaded`, and the `chat_model_ids` retain that drops
/// a served-but-embedding id from `/v1/models`. Fail-on-revert for both.
#[tokio::test]
async fn a_resident_embedding_model_is_refused_and_unlisted() {
    let dir = tempfile::TempDir::new().unwrap();
    crate::serve::test_support::write_embedding_gguf_fixture(dir.path(), "org/embed");
    let higgs = fake_higgs(vec![dir.path().to_path_buf()]);
    seed_fresh_profile(&higgs, "org/embed", CtxLen::Fixed { n: 512 }).await;

    // Explicit load succeeds — an embedding model may be resident.
    higgs.load("org/embed", None).await.expect("explicit load");
    assert_eq!(
        higgs.local_served_ids().await,
        vec!["org/embed".to_owned()],
        "the worker really serves it (residency is what makes this test non-vacuous)"
    );

    // Chat against the RESIDENT model → [HG079], from the resident arm (no JIT).
    let err = higgs
        .prepare_chat("org/embed", None, "[]")
        .await
        .expect_err("chat against a resident embedding model must be refused");
    assert!(
        matches!(
            err,
            crate::diagnostic::HiggsError::ModelNotChatCapable { .. }
        ),
        "expected [HG079] ModelNotChatCapable, got: {err}"
    );

    // …and `/v1/models` (the chat_model_ids union) does not advertise it, even
    // though local_served_ids does — served ≠ chat-reachable for this domain.
    assert!(
        !higgs
            .chat_model_ids()
            .await
            .contains(&"org/embed".to_owned()),
        "a resident embedding model must not be advertised as a chat target"
    );

    // THE choke point: the node actor's `ChatHandle` refuses the lease itself, on
    // the domain captured at LOAD time. This is the arm that also guards the hub
    // relay (`relay_chat` takes the same lease with no `resolve_loaded` in front)
    // and survives a post-load file deletion — neither is reachable through
    // `prepare_chat` above, so pin it directly.
    let (worker, _) = higgs
        .local
        .instances()
        .await
        .pop()
        .expect("the load left a resident worker");
    match higgs.local.chat_handle(worker, "org/embed").await {
        Err(err) => assert!(
            matches!(
                err,
                crate::diagnostic::HiggsError::ModelNotChatCapable { .. }
            ),
            "expected [HG079] from the ChatHandle gate, got: {err}"
        ),
        Ok(_) => panic!("the actor must refuse a chat lease on an embedding worker"),
    }

    // A MISMATCHED requested model gets the LEASE, not [HG079]: that request is
    // a stale route, and its verdict belongs to the worker's route-invalidating
    // [HG018] bind check — a domain refusal naming a model the client never
    // asked for would wedge the route forever (Fable r5). Fail-on-revert:
    // gating on the worker's identity alone refuses this too.
    assert!(
        higgs
            .local
            .chat_handle(worker, "some/other-model")
            .await
            .is_ok(),
        "a mismatched model must fall through to the worker's own HG018 check"
    );
}

/// Turbotune (Benchmark mode) refuses a non-generative model UP-FRONT with the
/// true reason ([HG079]) — its measurement is a CHAT generation, which the
/// dispatch gate refuses, so without this check every candidate would load,
/// fail, and burn a rung before surfacing a misleading "all candidates failed"
/// [HG063]. Fail-on-revert: remove the gate and this errors HG063, not HG079.
/// (The ANALYTICAL tune — plain Prepare — still works for these models; the
/// resident-embedding test above depends on exactly that.)
#[tokio::test]
async fn turbotune_refuses_a_non_generative_model_up_front() {
    let dir = tempfile::TempDir::new().unwrap();
    crate::serve::test_support::write_embedding_gguf_fixture(dir.path(), "org/embed");
    let higgs = fake_higgs(vec![dir.path().to_path_buf()]);
    let err = higgs
        .tune(TuneRequest {
            id: "org/embed".to_owned(),
            mode: Some(TuneMode::Benchmark),
            budget: None,
            pins: None,
        })
        .await
        .expect_err("benchmark mode must refuse an embedding model");
    assert!(
        matches!(
            err,
            crate::diagnostic::HiggsError::ModelNotChatCapable { .. }
        ),
        "expected the up-front [HG079], got: {err}"
    );
}

/// An id collision across scan variants must not hide a resident GENERATIVE
/// model: two GGUF files can share one catalog id (e.g. two quant files in one
/// model dir), and the LOAD path resolves the FIRST scanned variant
/// (`ModelStore::get`), so the advertising filter must judge the id by that
/// SAME variant — not by `any` same-id file being non-generative.
#[tokio::test]
async fn an_id_collision_does_not_hide_the_resident_generative_variant() {
    let dir = tempfile::TempDir::new().unwrap();
    // Two files, ONE id ("org/dual"). Scan sorts by (id, path):
    // "model-Q4_K_M.gguf" (generative) < "model-f16.gguf" (embedding), so the
    // generative variant is the one `ModelStore::get` — and any load — resolves.
    crate::serve::test_support::write_gguf_fixture(dir.path(), "org/dual");
    crate::serve::test_support::write_embedding_gguf_fixture(dir.path(), "org/dual");
    let higgs = fake_higgs(vec![dir.path().to_path_buf()]);

    // Precondition: BOTH variants scanned under the one id, generative first.
    let scanned = higgs.scan().await.expect("scan");
    let variants: Vec<_> = scanned.iter().filter(|m| m.id == "org/dual").collect();
    assert_eq!(variants.len(), 2, "both files share the id");
    assert_eq!(
        variants[0].domain,
        crate::worker::models::ModelDomain::Llm,
        "the FIRST variant (what a load resolves) is the generative one"
    );

    seed_fresh_profile(&higgs, "org/dual", CtxLen::Fixed { n: 512 }).await;
    higgs.load("org/dual", None).await.expect("load");
    assert!(
        higgs
            .chat_model_ids()
            .await
            .contains(&"org/dual".to_owned()),
        "the resident generative variant stays advertised despite the same-id \
         embedding file"
    );
}

/// A shadow file appearing under a resident model's id must not flip the chat
/// verdict: the resident checks read the worker's LOAD-TIME domain (the same
/// facts the `ChatHandle` gate enforces), never a re-scan. Fail-on-revert for
/// the scan-derived logic this replaces (codex r3): with the shadow sorting
/// path-first, a scan-based resident arm falsely refuses [HG079] and a
/// scan-based listing filter hides the id.
#[tokio::test]
async fn a_shadow_file_cannot_flip_a_resident_models_chat_verdict() {
    let dir = tempfile::TempDir::new().unwrap();
    crate::serve::test_support::write_gguf_fixture(dir.path(), "org/shadow");
    let higgs = fake_higgs(vec![dir.path().to_path_buf()]);
    seed_fresh_profile(&higgs, "org/shadow", CtxLen::Fixed { n: 512 }).await;
    higgs.load("org/shadow", None).await.expect("load");

    // AFTER the load, an embedding conversion lands under the SAME id with a
    // path-lexically FIRST filename — the variant a fresh scan resolves first.
    crate::serve::test_support::write_embedding_gguf_fixture_named(
        dir.path(),
        "org/shadow",
        "model-0-shadow.gguf",
    );
    let first = higgs
        .scan()
        .await
        .expect("scan")
        .into_iter()
        .find(|m| m.id == "org/shadow")
        .expect("scanned");
    assert_eq!(
        first.domain,
        crate::worker::models::ModelDomain::Embedding,
        "precondition: the scan's first variant is now the embedding shadow"
    );

    // The RESIDENT worker is still the generative file it loaded: chat resolves…
    higgs
        .prepare_chat("org/shadow", None, "[]")
        .await
        .expect("the resident generative model still chats — no false [HG079]");
    // …and the id stays advertised.
    assert!(
        higgs
            .chat_model_ids()
            .await
            .contains(&"org/shadow".to_owned()),
        "the resident generative model stays advertised despite the shadow file"
    );

    // The WIRE domain agrees: with BOTH sources present (facts say Llm, the
    // scan's first variant says Embedding), the load-time fact must WIN — this
    // is the conflicting-Some case the vanished-file test cannot pin (there the
    // scan side is None, so either `.or()` order passes; Fable r8). jigglebot's
    // picker gates sends on exactly this field.
    let status = higgs.status().await.expect("status");
    let info = status
        .loaded_all
        .iter()
        .find(|l| l.id == "org/shadow")
        .expect("resident");
    assert_eq!(
        info.domain,
        Some(crate::worker::models::ModelDomain::Llm),
        "status: the load-time fact outranks the shadow's scan verdict"
    );
    assert_eq!(
        higgs
            .local_loaded_info("org/shadow")
            .await
            .expect("resident")
            .domain,
        Some(crate::worker::models::ModelDomain::Llm),
        "local_loaded_info agrees"
    );
}

/// A scan row's `readiness` is NOT a reachability verdict — `chat_model_ids` is.
///
/// The mirror of the shadow test: the RESIDENT worker is the embedding file, and
/// the generative variant is the one a fresh scan resolves FIRST. That first row
/// therefore reads `Loaded` (readiness matches residency by ID, and this row's own
/// scanned domain is `Llm`, so no terminal state short-circuits it) — while chat
/// on the id is refused ([HG079]) against the resident worker's LOAD-TIME facts.
///
/// A consumer that derives its picker from `readiness` re-offers this id and every
/// send 400s; that is the divergence `HiggsModelsResponse::chat_model_ids` exists
/// to end. This test pins BOTH halves — the misleading row AND the honest list —
/// so neither can be "reconciled" by teaching `chat_model_ids` to trust readiness.
///
/// It also pins why `chat_model_ids`' JIT leg needs no `local_non_chat` guard
/// (only the remote leg has one): a resident id can never read `Servable`, so
/// `servable_model_ids` cannot re-admit an id the domain filter dropped.
#[tokio::test]
async fn a_loaded_readiness_row_does_not_make_a_non_chat_id_reachable() {
    let dir = tempfile::TempDir::new().unwrap();
    // Embedding only, so the load resolves IT and the worker's facts say Embedding.
    crate::serve::test_support::write_embedding_gguf_fixture(dir.path(), "org/flip");
    let higgs = fake_higgs(vec![dir.path().to_path_buf()]);
    seed_fresh_profile(&higgs, "org/flip", CtxLen::Fixed { n: 512 }).await;
    higgs.load("org/flip", None).await.expect("load");

    // AFTER the load, a generative variant lands under the SAME id with a
    // path-lexically FIRST filename ("model-Q4_K_M.gguf" < "model-f16.gguf").
    crate::serve::test_support::write_gguf_fixture(dir.path(), "org/flip");
    let entries = higgs.model_entries().await.expect("scan");
    let first = entries
        .iter()
        .find(|m| m.model.id == "org/flip")
        .expect("scanned");
    assert_eq!(
        first.model.domain,
        crate::worker::models::ModelDomain::Llm,
        "precondition: the scan's FIRST variant for the id is now the generative file"
    );

    // The misleading half: that row reads `Loaded`.
    assert_eq!(
        first.readiness,
        crate::serve::readiness::ModelReadiness::Loaded,
        "precondition: the first row reads Loaded — residency is matched by id, and \
         this row's own domain is Llm, so no terminal state masks it"
    );

    // The honest half: the id is NOT chat-reachable, and a chat proves it.
    assert!(
        !higgs
            .chat_model_ids()
            .await
            .contains(&"org/flip".to_owned()),
        "a `Loaded` readiness row must not make a non-generative resident reachable"
    );
    let err = higgs
        .prepare_chat("org/flip", None, "[]")
        .await
        .expect_err("chat against the resident embedding worker must be refused");
    assert!(
        matches!(
            err,
            crate::diagnostic::HiggsError::ModelNotChatCapable { .. }
        ),
        "expected [HG079] ModelNotChatCapable, got: {err}"
    );
}

/// `servable_model_ids` judges each id by its FIRST scan variant — the file a
/// JIT load actually resolves. A tuned generative SECOND variant behind an
/// embedding first file must not advertise an id every chat then refuses
/// ([HG079]). Fail-on-revert: evaluating every variant re-advertises it.
#[tokio::test]
async fn a_second_variant_cannot_advertise_an_id_the_load_wont_serve() {
    let dir = tempfile::TempDir::new().unwrap();
    // Embedding first ("model-0-…" sorts before "model-Q4_K_M.gguf"), generative second.
    crate::serve::test_support::write_embedding_gguf_fixture_named(
        dir.path(),
        "org/mixed",
        "model-0-embed.gguf",
    );
    crate::serve::test_support::write_gguf_fixture(dir.path(), "org/mixed");
    let higgs = fake_higgs(vec![dir.path().to_path_buf()]);
    // One tuning record per ID, anchored to the GENERATIVE second variant's file
    // — so THAT variant is genuinely prepared, fresh, and fitting (Servable on
    // its own terms). Only the first-variant rule withholds the id.
    let gen_path = dir
        .path()
        .join("org/mixed/model-Q4_K_M.gguf")
        .to_string_lossy()
        .into_owned();
    seed_fresh_profile_at(&higgs, "org/mixed", CtxLen::Fixed { n: 512 }, &gen_path).await;

    // Precondition for the fail-on-revert claim: evaluated ON ITS OWN, the
    // generative second variant reads Servable — the first-variant rule is the
    // ONLY thing keeping the id out of the advertisement.
    {
        let scan = higgs.scan().await.expect("scan");
        let second = scan
            .iter()
            .find(|m| m.id == "org/mixed" && m.path == gen_path)
            .expect("generative variant scanned");
        let hw = higgs.hardware().await;
        let tuning = crate::serve::control::active_records(
            &higgs.tuning_profiles().expect("tuning profiles"),
        );
        let (readiness, _) = higgs.model_readiness(second, &[], &hw, &tuning);
        assert_eq!(
            readiness,
            crate::serve::readiness::ModelReadiness::Servable,
            "the second variant is Servable on its own terms"
        );
    }

    assert!(
        !higgs
            .servable_model_ids()
            .await
            .contains(&"org/mixed".to_owned()),
        "the id's JIT target is the embedding first variant — advertising it \
         promises a chat that always refuses"
    );
    // …and the JIT gate agrees with the non-advertisement.
    let err = higgs
        .prepare_chat("org/mixed", None, "[]")
        .await
        .expect_err("JIT resolves the embedding first variant");
    assert!(matches!(
        err,
        crate::diagnostic::HiggsError::ModelNotChatCapable { .. }
    ));
}

/// The wire `LoadedInfo.domain` — what jigglebot's picker gates sends on — must
/// survive the FILE disappearing after the load: it reads the worker's
/// LOAD-TIME facts, with the scan only as a no-facts fallback. Fail-on-revert:
/// a scan-derived fill goes `None` (permissive) when the file is deleted, and
/// the picker would offer a resident embedding model whose every send refuses.
#[tokio::test]
async fn loaded_info_domain_survives_the_file_vanishing() {
    let dir = tempfile::TempDir::new().unwrap();
    crate::serve::test_support::write_embedding_gguf_fixture(dir.path(), "org/embed");
    crate::serve::test_support::write_embedding_gguf_fixture(dir.path(), "org/embed2");
    let higgs = fake_higgs(vec![dir.path().to_path_buf()]);
    seed_fresh_profile(&higgs, "org/embed", CtxLen::Fixed { n: 512 }).await;
    seed_fresh_profile(&higgs, "org/embed2", CtxLen::Fixed { n: 512 }).await;
    // TWO residents: org/embed becomes the PRIMARY (live-probed), org/embed2 a
    // SECONDARY (stub path) — the two loaded_all arms fill domains differently,
    // and both must survive the files vanishing.
    higgs.load("org/embed", None).await.expect("load");
    higgs.load("org/embed2", None).await.expect("load 2");

    // The files vanish AFTER the loads (the workers still serve their mmaps).
    std::fs::remove_file(dir.path().join("org/embed/model-f16.gguf")).expect("delete");
    std::fs::remove_file(dir.path().join("org/embed2/model-f16.gguf")).expect("delete 2");

    let status = higgs.status().await.expect("status");
    for id in ["org/embed", "org/embed2"] {
        let info = status
            .loaded_all
            .iter()
            .find(|l| l.id == id)
            .expect("still resident");
        assert_eq!(
            info.domain,
            Some(crate::worker::models::ModelDomain::Embedding),
            "{id}: the wire domain is the LOAD-TIME fact, not a re-scan of a vanished file"
        );
    }
    let info = higgs
        .local_loaded_info("org/embed")
        .await
        .expect("resident");
    assert_eq!(
        info.domain,
        Some(crate::worker::models::ModelDomain::Embedding),
        "local_loaded_info agrees"
    );
}

/// A RERANKER (RANK pooling) is refused at every gate an embedder is — the
/// comparisons are `!= Llm`, never `== Embedding`. The embedding fixtures alone
/// cannot catch that mutation (Fable r8: weakening any gate to `== Embedding`
/// survived the whole suite); this test kills the mutant class at the tune
/// gate, the JIT/resident chat gates, the actor choke point, and the listing.
#[tokio::test]
async fn a_reranker_is_refused_at_every_gate() {
    let dir = tempfile::TempDir::new().unwrap();
    crate::serve::test_support::write_reranker_gguf_fixture(dir.path(), "org/rr");
    let higgs = fake_higgs(vec![dir.path().to_path_buf()]);
    seed_fresh_profile(&higgs, "org/rr", CtxLen::Fixed { n: 512 }).await;

    // Turbotune refuses up-front (before any load).
    let err = higgs
        .tune(TuneRequest {
            id: "org/rr".to_owned(),
            mode: Some(TuneMode::Benchmark),
            budget: None,
            pins: None,
        })
        .await
        .expect_err("benchmark mode refuses a reranker");
    assert!(matches!(
        err,
        crate::diagnostic::HiggsError::ModelNotChatCapable { .. }
    ));

    // JIT chat refuses (unloaded, scanned) …
    let err = higgs
        .prepare_chat("org/rr", None, "[]")
        .await
        .expect_err("JIT refuses a reranker");
    assert!(matches!(
        err,
        crate::diagnostic::HiggsError::ModelNotChatCapable { .. }
    ));
    // …and it is never advertised as servable.
    assert!(!higgs
        .servable_model_ids()
        .await
        .contains(&"org/rr".to_owned()));

    // Explicit load is allowed; the RESIDENT gates refuse too.
    higgs.load("org/rr", None).await.expect("explicit load");
    let err = higgs
        .prepare_chat("org/rr", None, "[]")
        .await
        .expect_err("the resident reranker is refused");
    assert!(matches!(
        err,
        crate::diagnostic::HiggsError::ModelNotChatCapable { .. }
    ));
    let (worker, _) = higgs
        .local
        .instances()
        .await
        .pop()
        .expect("resident worker");
    assert!(
        higgs.local.chat_handle(worker, "org/rr").await.is_err(),
        "the actor choke point refuses a reranker lease"
    );
    assert!(
        !higgs.chat_model_ids().await.contains(&"org/rr".to_owned()),
        "a resident reranker is not advertised"
    );
}

/// The JIT advertising leg must not list an id whose durable REMOTE route is
/// known non-generative: dispatch consults the route BEFORE the JIT arm (routes
/// win over JIT by design), so its [HG079] pre-refusal preempts the local load
/// the ad would promise — the r9 rule's mirror image (Fable r10). Fail-on-revert:
/// the unconditional JIT leg re-advertises the id.
#[tokio::test]
async fn a_remote_non_chat_route_suppresses_the_jit_ad() {
    use crate::node::test_support::{fake_runtime, local_endpoint, stage_dummy_model};
    use crate::node::transport::NodeTransport;
    use crate::node::{fleet::HubFleet, serve_node};
    use crate::remote::ALPN;

    // LOCAL: a generative model under org/dual4, tuned + fresh — Servable.
    let local_dir = tempfile::TempDir::new().unwrap();
    crate::serve::test_support::write_gguf_fixture(local_dir.path(), "org/dual4");
    let higgs = fake_higgs(vec![local_dir.path().to_path_buf()]);
    seed_fresh_profile(&higgs, "org/dual4", CtxLen::Fixed { n: 512 }).await;
    assert!(
        higgs
            .servable_model_ids()
            .await
            .contains(&"org/dual4".to_owned()),
        "precondition: locally JIT-able on its own terms"
    );

    // REMOTE: a real loopback node serving an EMBEDDING file under the SAME id.
    let (node_root, _) = stage_dummy_model("org/other"); // scan root; the real model:
    crate::serve::test_support::write_embedding_gguf_fixture(node_root.path(), "org/dual4");
    let hub_ep = local_endpoint().await;
    let node_ep = local_endpoint().await;
    let hub_addr = hub_ep.addr();
    let node_key = node_ep.id().to_string();
    let rt = std::sync::Arc::new(fake_runtime(vec![node_root.path().to_path_buf()]));
    tokio::spawn(async move {
        let conn = node_ep.connect(hub_addr, ALPN).await.expect("connect");
        serve_node(conn, rt).await;
    });
    let conn = hub_ep
        .accept()
        .await
        .expect("incoming")
        .await
        .expect("conn");
    std::mem::forget(hub_ep);
    let fleet = std::sync::Arc::new(HubFleet::new(std::sync::Arc::new(
        crate::log_bus::LogBus::new(),
    )));
    fleet
        .add_node(
            node_key.clone(),
            std::sync::Arc::new(NodeTransport::new(conn)),
            None,
            None,
            None,
            false,
            None,
            true,
        )
        .await;
    fleet.load(&node_key, "org/dual4", None).await.unwrap();
    fleet.refresh_inventory(&node_key).await.unwrap();
    higgs.set_fleet(fleet.clone());

    // Dispatch would pre-refuse via the remote route (routes win over JIT)…
    let err = higgs
        .prepare_chat("org/dual4", None, "[]")
        .await
        .expect_err("the remote non-chat route preempts the JIT load");
    assert!(matches!(
        err,
        crate::diagnostic::HiggsError::ModelNotChatCapable { .. }
    ));
    // …so the ad must not stand.
    assert!(
        !higgs
            .chat_model_ids()
            .await
            .contains(&"org/dual4".to_owned()),
        "an id whose remote route is known non-generative is not advertised via JIT"
    );
}

#[tokio::test]
async fn a_cancelled_hg090_adopt_keeps_the_live_originals_log_step_slot() {
    // The HG090 discriminator lives in the CANCELLED arm (every production
    // HG090 emit is a `Cancelled{HG090}` terminal since the r51-r55 event
    // unification). A duplicate-refusal adopt shares the LIVE original's
    // (node, repo, file) key — clearing the slot would reset the original's
    // decile logging, and logging "download cancelled" would misreport a
    // transfer that is still running. A REAL cancel (HG089) still clears.
    let higgs = fake_higgs(vec![]);
    // The live original is mid-stream: its slot is tracked at decile 3.
    let progress = crate::catalog::wire::ModelDownloadEvent {
        node: Some("n1".into()),
        repo: "acme/m".into(),
        file: "m.gguf".into(),
        phase: crate::catalog::wire::ModelDownloadPhase::Downloading,
        downloaded_bytes: 3,
        total_bytes: Some(10),
        at_ms: 1,
        code: None,
        path: None,
    };
    higgs.log_download_event(&progress);
    assert_eq!(higgs.download_log_steps.lock().len(), 1);
    // A duplicate attempt's adopt terminal (same key, code HG090) must NOT
    // clear the original's slot.
    let adopt = crate::catalog::wire::ModelDownloadEvent {
        phase: crate::catalog::wire::ModelDownloadPhase::Cancelled,
        code: Some("HG090".into()),
        ..progress.clone()
    };
    higgs.log_download_event(&adopt);
    assert_eq!(
        higgs.download_log_steps.lock().len(),
        1,
        "an HG090 adopt is a one-off info line about the DUPLICATE; the \
         live original's throttle slot survives"
    );
    // A REAL cancel (HG089) is the original's own terminal — slot clears.
    let real_cancel = crate::catalog::wire::ModelDownloadEvent {
        phase: crate::catalog::wire::ModelDownloadPhase::Cancelled,
        code: Some("HG089".into()),
        ..progress.clone()
    };
    higgs.log_download_event(&real_cancel);
    assert!(
        higgs.download_log_steps.lock().is_empty(),
        "a real cancel clears the slot"
    );
}
