//! In-process integration test for higgs's CONTROL surface — now the crate API.
//!
//! higgs is a library: the old `/api/higgs/*` HTTP control surface is GONE, so
//! control runs through the in-process `Higgs` facade (`status`, `model_entries`,
//! `load`, `version`, `hardware`, `estimate`, `tune`, key/idle/log settings). A
//! REAL local llama.cpp worker still runs via the `worker_exe` DI seam (see
//! `common::higgs_local`), so the full load → status → unload engine path is
//! exercised in-process against the tiny on-disk `stories260K.gguf`.

mod common;

use std::collections::HashSet;

use common::{higgs_local, TINY_MODEL_ID};
use higgs::serve::readiness::ModelReadiness;
use higgs::system::DeviceKind;
use higgs::tune::{FitVerdict, ResourceBudget};
use higgs::worker::engine::{CtxLen, GpuLayers};
use higgs::{EstimateRequest, HiggsError, LoadParams, TuneRequest};

#[tokio::test]
async fn control_api_lifecycle() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP control_api_lifecycle: tiny gguf not found (set HIGGS_TEST_GGUF)");
        return;
    };
    let id = TINY_MODEL_ID;

    // status at boot: spawn-on-load means NO worker until a model is loaded.
    let status = higgs.status().await.expect("status");
    assert!(
        !status.worker_alive,
        "no worker before first load (spawn-on-load)"
    );
    assert!(status.loaded.is_none(), "nothing loaded at start");

    // models: a live scan of the configured dirs — the staged tiny model is here.
    let models = higgs.model_entries().await.expect("model_entries");
    assert!(!models.is_empty(), "scan found models");
    let scanned = models
        .iter()
        .find(|m| m.model.id == id)
        .expect("scan lists the staged tiny model");
    // The scan no longer probes loadability (no `loadable` field on the typed
    // entry) — it is learned only at actual load time, below.
    assert_eq!(scanned.state, "not-loaded", "scanned-but-unloaded entry");

    // load the model.
    higgs.load(id, None).await.expect("load returns ok");

    // status now reports the loaded model.
    let status = higgs.status().await.expect("status");
    let loaded = status.loaded.as_ref().expect("status shows loaded id");
    assert_eq!(loaded.id, id, "status shows loaded id");
    match loaded.ctx_len {
        Some(CtxLen::Fixed { n }) => assert!(n > 0, "loaded ctx_len > 0"),
        other => panic!("expected a fixed loaded ctx_len, got {other:?}"),
    }

    // models list marks that entry "loaded".
    let models = higgs.model_entries().await.expect("model_entries");
    let entry = models
        .iter()
        .find(|m| m.model.id == id)
        .expect("loaded model in list");
    assert_eq!(entry.state, "loaded", "entry state is loaded");
    assert_eq!(entry.format, "gguf");
    assert_eq!(
        entry.model.arch.as_deref(),
        Some("llama"),
        "stories260K is a llama-arch GGUF"
    );
    // The tiny model embeds NO chat template (engine falls back to chatml), so the
    // template-derived capability flags are false.
    assert!(
        !entry.model.supports_tools,
        "no embedded template → no tools"
    );
    assert!(
        !entry.model.supports_reasoning,
        "no embedded template → no reasoning"
    );
    // P2: the load persisted a per-model record — the entry carries `last_load`.
    assert!(
        entry.last_load.is_some(),
        "loaded model carries a persisted last_load record: {entry:?}"
    );

    // model-by-id returns the entry.
    let by_id = higgs.model_by_id(id).await.expect("model_by_id");
    assert_eq!(by_id.model.id, id, "model_by_id returns the entry");

    // model-by-id for a NON-existent id → ModelNotFound (was a 404 HG envelope).
    let missing = higgs.model_by_id("no-such-org/no-such-model").await;
    assert!(
        matches!(missing, Err(HiggsError::ModelNotFound { .. })),
        "unknown model id is ModelNotFound: {missing:?}"
    );

    // version reports the engine.
    let version = higgs.version();
    assert_eq!(version.engine, "llama.cpp", "version reports the engine");

    // logs: a lines vec; n=0 returns none.
    let _lines = higgs.logs(50, None);
    assert_eq!(higgs.logs(0, None).len(), 0, "n=0 returns no lines");

    // system: hardware panels + runtime engine.
    let hw = higgs.hardware().await;
    assert!(!hw.cpu_name.is_empty(), "system reports a CPU name");
    assert!(hw.ram_total_bytes > 0, "system reports total RAM");
    assert_eq!(higgs.version().engine, "llama.cpp", "runtime engine");

    // unload → status clears.
    higgs.unload().await.expect("unload");
    let status = higgs.status().await.expect("status");
    assert!(status.loaded.is_none(), "nothing loaded after unload");

    // unload again with nothing loaded — idempotent.
    higgs
        .unload()
        .await
        .expect("unload with nothing loaded is ok");

    // Re-load with EXPLICIT load params (ctx 2048, gpu_layers 0, threads 2) —
    // exercises the non-default LoadParams branch. ctx_len is echoed back.
    higgs
        .load(
            id,
            Some(LoadParams::base(
                CtxLen::Fixed { n: 2048 },
                GpuLayers::Count { n: 0 },
                2,
            )),
        )
        .await
        .expect("explicit-params load returns ok");
    let status = higgs.status().await.expect("status");
    let loaded = status.loaded.as_ref().expect("loaded");
    assert_eq!(
        loaded.ctx_len,
        Some(CtxLen::Fixed { n: 2048 }),
        "explicit ctx_len honored"
    );
    // Per-load idle_ttl_minutes is a no-op: the facade `load` API has no per-load
    // idle-TTL field, so status can never advertise an override it ignores. (The
    // reaper applies one per-node TTL; this stays structurally guaranteed.)
    assert!(
        loaded.idle_ttl_minutes.is_none(),
        "per-load idle_ttl_minutes must NOT be surfaced (reaper ignores it): {loaded:?}"
    );

    // ── worker/stop is NON-TERMINAL: the node stays loadable after it ─────────
    // `worker_stop` is a bulk UNLOAD, not the terminal `stop` drain. If it marked
    // the runtime shutting-down, the RELOAD below would be rejected and brick the
    // node until process restart.
    higgs.worker_stop().await.expect("worker_stop returns ok");
    let status = higgs.status().await.expect("status");
    assert!(
        !status.worker_alive,
        "worker_alive is false after worker_stop"
    );

    // REGRESSION: a load AFTER worker_stop must still succeed — non-terminal.
    higgs
        .load(id, None)
        .await
        .expect("load after worker_stop must still work — worker_stop is non-terminal");
    let status = higgs.status().await.expect("status");
    assert_eq!(
        status.loaded.as_ref().unwrap().id,
        id,
        "reloaded model is resident again after the non-terminal worker_stop"
    );

    // Graceful drain so the worker child is reaped deterministically.
    higgs.shutdown().await;
}

/// Multi-model: the local node is multi-worker (one worker per loaded model), so
/// `status().loaded_all` must report EVERY resident model (each tagged with its
/// `worker_id`) — not just the primary `loaded`.
#[tokio::test]
async fn status_loaded_all_lists_every_worker() {
    let Some(higgs) = higgs_local(&["org-a/m", "org-b/m"]).await else {
        eprintln!("SKIP status_loaded_all_lists_every_worker: tiny gguf not found");
        return;
    };

    for id in ["org-a/m", "org-b/m"] {
        higgs
            .load(id, None)
            .await
            .unwrap_or_else(|e| panic!("load {id}: {e:?}"));
    }

    let status = higgs.status().await.expect("status");
    assert_eq!(
        status.loaded_all.len(),
        2,
        "every loaded worker reported: {status:?}"
    );
    let mut ids: Vec<&str> = status.loaded_all.iter().map(|e| e.id.as_str()).collect();
    ids.sort_unstable();
    assert_eq!(
        ids,
        ["org-a/m", "org-b/m"],
        "both models present: {status:?}"
    );
    let workers: HashSet<u32> = status.loaded_all.iter().map(|e| e.worker_id).collect();
    assert_eq!(
        workers.len(),
        2,
        "each entry has a distinct worker_id: {status:?}"
    );
    // `loaded` (back-compat primary) is one of the resident models.
    assert!(
        ["org-a/m", "org-b/m"].contains(&status.loaded.as_ref().unwrap().id.as_str()),
        "primary loaded is one of the resident models: {status:?}"
    );

    higgs.shutdown().await;
}

/// Settings + log toggles round-trip through the facade, and the id-validation +
/// model-by-id branches answer as before. (The old HTTP `/health` +
/// `/api/higgs/health` probes are HTTP-only and covered by `serve_v1_local`'s
/// readiness poll in `tests/inference.rs`; the settings-schema JSON round-trip is
/// likewise HTTP-only — the typed getters/setters are exercised directly here.)
#[tokio::test]
async fn control_settings_and_validation() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP control_settings_and_validation: tiny gguf not found");
        return;
    };

    // ── log settings: read current, flip verbose, read reflects it ───────────
    let before = higgs.logs_settings();
    higgs.set_logs_settings(&higgs::LogSettings {
        verbose: !before.verbose,
        ..before.clone()
    });
    assert_eq!(
        higgs.logs_settings().verbose,
        !before.verbose,
        "verbose toggle persisted"
    );

    // ── runtime settings: flip JIT and read it back (schema round-trip) ──────
    let jit0 = higgs.jit_enabled();
    higgs.set_jit_enabled(!jit0);
    assert_eq!(higgs.jit_enabled(), !jit0, "jit toggle persisted");
    higgs.set_jit_enabled(jit0);

    // ── Invalid model ids are rejected with a typed error (id-validation) ────
    for bad in ["bad id with spaces", "../escape", ""] {
        let r = higgs.load(bad, None).await;
        assert!(r.is_err(), "invalid id {bad:?} → error, got {r:?}");
    }

    // ── model-by-id for a SCANNED-but-unloaded model reads on-disk metadata ──
    let detail = higgs
        .model_by_id(TINY_MODEL_ID)
        .await
        .expect("by-id for a scanned model ok");
    assert_eq!(detail.model.id, TINY_MODEL_ID, "by-id returns the model");

    // by-id for an UNKNOWN model → ModelNotFound (the not-found branch).
    let missing = higgs.model_by_id("ghost-org/ghost-model").await;
    assert!(
        matches!(missing, Err(HiggsError::ModelNotFound { .. })),
        "by-id for an unknown model is ModelNotFound: {missing:?}"
    );

    higgs.shutdown().await;
}

/// The supervisor's auto-restart FSM: when a worker child dies unexpectedly, the
/// next request respawns it and REPLAYS the recorded load. We load a model,
/// hard-kill the worker child (now a child of THIS test process — the seam spawns
/// it from the real higgs binary), then chat again — it must succeed against a
/// freshly restarted + reloaded worker.
#[tokio::test]
async fn worker_crash_triggers_restart_and_replay() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP worker_crash_triggers_restart_and_replay: tiny gguf not found");
        return;
    };

    higgs
        .load(TINY_MODEL_ID, None)
        .await
        .expect("initial load ok");

    // Find and HARD-kill the worker child (a `--higgs-worker` child of THIS process).
    let pids = worker_child_pids(std::process::id());
    assert!(!pids.is_empty(), "found the worker child process");
    for pid in &pids {
        // SAFETY: SIGKILL to a pid we just enumerated as our own child.
        unsafe {
            libc::kill(*pid as libc::pid_t, libc::SIGKILL);
        }
    }

    // The next chat must succeed: the supervisor detects the dead child, restarts
    // it, replays the recorded load, then serves — possibly after a transient error.
    let mut ok = false;
    for _ in 0..40 {
        let attempt = higgs
            .chat_stream(
                TINY_MODEL_ID.to_owned(),
                r#"[{"role":"user","content":"hi"}]"#.to_owned(),
                4,
                higgs::SamplingParams::default(),
                None,
                None,
            )
            .await;
        if let Ok((mut deltas, handle)) = attempt {
            while deltas.recv().await.is_some() {}
            if matches!(handle.await, Ok(Ok(_))) {
                ok = true;
                break;
            }
        }
        tokio::time::sleep(std::time::Duration::from_millis(250)).await;
    }
    assert!(
        ok,
        "chat succeeds after the worker was restarted + the load replayed"
    );

    higgs.shutdown().await;
}

/// The idle reaper auto-unloads a model after its idle TTL elapses: with
/// `auto_unload_idle` on and `idle_ttl_minutes = 0`, a loaded-but-unused worker is
/// reaped on the next reaper tick (fast cadence at TTL 0) without any client action.
#[tokio::test]
async fn idle_reaper_auto_unloads_a_model() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP idle_reaper_auto_unloads_a_model: tiny gguf not found");
        return;
    };

    // Aggressive idle auto-unload (TTL 0 ⇒ idle immediately) BEFORE loading.
    higgs.set_auto_unload_idle(true);
    higgs.set_idle_ttl_minutes(0);

    higgs.load(TINY_MODEL_ID, None).await.expect("load ok");

    // Within a few reaper ticks the idle worker is auto-unloaded.
    let mut unloaded = false;
    for _ in 0..100 {
        let status = higgs.status().await.expect("status");
        if !status.worker_alive || status.loaded.is_none() {
            unloaded = true;
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
    assert!(
        unloaded,
        "idle reaper auto-unloaded the model within the timeout"
    );

    higgs.shutdown().await;
}

/// `server_config()` reports the EFFECTIVE (live, runtime-mutable) idle TTL, and a
/// huge `idle_ttl_minutes` is clamped before the ×60 (no overflow). Reverting
/// either breaks this: a stale fixed value would disagree with the 60-min default,
/// and `u64::MAX * 60` would overflow (debug panic / release wrap) instead of clamping.
#[tokio::test]
async fn idle_ttl_effective_report_and_overflow_clamp() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP idle_ttl_effective_report_and_overflow_clamp: tiny gguf not found");
        return;
    };

    // ── Default: the live 60-min default (3600s), not a stale value ──
    assert_eq!(
        higgs.idle_ttl_minutes(),
        60,
        "settings default idle TTL is 60 minutes"
    );
    let default_secs = higgs.server_config().limits.idle_unload_ttl_secs;
    assert_eq!(
        default_secs, 3600,
        "idle_unload_ttl_secs is the live 60min×60"
    );
    assert_eq!(
        default_secs,
        higgs.idle_ttl_minutes() * 60,
        "system and settings agree"
    );

    // ── Change it: set a new TTL, /system tracks it without a restart ──
    higgs.set_idle_ttl_minutes(45);
    assert_eq!(
        higgs.idle_ttl_minutes(),
        45,
        "settings reflects the new TTL"
    );
    assert_eq!(
        higgs.server_config().limits.idle_unload_ttl_secs,
        45 * 60,
        "idle_unload_ttl_secs tracks the runtime change (45×60=2700)"
    );

    // ── Overflow clamp: u64::MAX minutes must NOT overflow ×60 ──
    higgs.set_idle_ttl_minutes(u64::MAX);
    let clamped_min = higgs.idle_ttl_minutes();
    assert!(
        clamped_min < u64::MAX,
        "huge idle_ttl_minutes is clamped, got {clamped_min}"
    );
    let clamped_secs = higgs.server_config().limits.idle_unload_ttl_secs;
    assert_eq!(
        clamped_secs,
        clamped_min.saturating_mul(60),
        "system seconds == clamped minutes ×60 (overflow-free, settings ↔ system agree)"
    );
    assert!(
        clamped_secs >= clamped_min,
        "seconds did not wrap below minutes (no overflow), {clamped_secs} vs {clamped_min}"
    );

    higgs.shutdown().await;
}

/// `estimate` returns the VRAM/RAM footprint of a CANDIDATE load, and the footprint
/// grows with context (KV cache is linear in n_ctx). A missing model is ModelNotFound.
#[tokio::test]
async fn estimate_returns_footprint_growing_with_context() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP estimate: no tiny GGUF (set HIGGS_TEST_GGUF)");
        return;
    };

    let small = higgs
        .estimate(estimate_req(TINY_MODEL_ID, 2048, None))
        .await
        .expect("estimate request");
    // RAM is always charged (weights + overhead); a CPU-only host reports vram == 0.
    assert!(small.ram.needed_bytes > 0, "footprint present: {small:?}");

    let large = higgs
        .estimate(estimate_req(TINY_MODEL_ID, 16384, None))
        .await
        .expect("estimate request");
    // KV cache is linear in n_ctx → a bigger context needs strictly more memory.
    let total = |r: &higgs::EstimateReport| r.vram.needed_bytes + r.ram.needed_bytes;
    assert!(
        total(&large) > total(&small),
        "bigger context → bigger total footprint (small {} vs large {})",
        total(&small),
        total(&large)
    );

    // A missing model → ModelNotFound (not a 500 or a silent 0).
    let missing = higgs
        .estimate(estimate_req("nope/missing", 2048, None))
        .await;
    assert!(
        matches!(missing, Err(HiggsError::ModelNotFound { .. })),
        "missing model → ModelNotFound: {missing:?}"
    );

    higgs.shutdown().await;
}

/// `estimate` measures the verdict against the supplied resource budget: a 1 MiB
/// RAM cap (which no real model fits) flips RAM Fits → Overflow.
#[tokio::test]
async fn estimate_honors_resource_budget() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP estimate_budget: no tiny GGUF (set HIGGS_TEST_GGUF)");
        return;
    };

    // Baseline: no budget → RAM comfortably fits the machine.
    let base = higgs
        .estimate(estimate_req(TINY_MODEL_ID, 2048, None))
        .await
        .expect("estimate request");
    assert_eq!(
        base.ram.verdict,
        FitVerdict::Fits,
        "no budget → RAM fits: {base:?}"
    );

    // A 1 MiB RAM budget can't hold the model+overhead → Overflow.
    let capped = higgs
        .estimate(estimate_req(
            TINY_MODEL_ID,
            2048,
            Some(ResourceBudget {
                max_ram_bytes: Some(1u64 << 20),
                ..Default::default()
            }),
        ))
        .await
        .expect("estimate request");
    assert_eq!(
        capped.ram.verdict,
        FitVerdict::Overflow,
        "a 1 MiB RAM budget overflows → the estimate honors the budget: {capped:?}"
    );

    higgs.shutdown().await;
}

/// `estimate` honors `offload_kqv`: disabling KV offload moves the KV cache off the
/// GPU, so the VRAM footprint drops (the param is threaded, not dropped). GPU-gated.
#[tokio::test]
async fn estimate_honors_offload_kqv() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP estimate_offload_kqv: no tiny GGUF (set HIGGS_TEST_GGUF)");
        return;
    };

    let req = |kqv: bool| EstimateRequest {
        id: TINY_MODEL_ID.to_owned(),
        ctx_len: CtxLen::Fixed { n: 16384 },
        gpu_layers: None,
        type_k: None,
        type_v: None,
        offload_kqv: Some(kqv),
        cpu_moe: None,
        budget: None,
    };
    let on = higgs.estimate(req(true)).await.expect("estimate request");
    let off = higgs.estimate(req(false)).await.expect("estimate request");

    let has_gpu = higgs
        .hardware()
        .await
        .gpus
        .iter()
        .any(|g| matches!(g.kind, DeviceKind::Gpu));
    if has_gpu {
        // KV leaves the GPU when offload_kqv=false → strictly less VRAM than on.
        assert!(
            off.vram.needed_bytes < on.vram.needed_bytes,
            "offload_kqv=false drops KV off the GPU (off {} vs on {})",
            off.vram.needed_bytes,
            on.vram.needed_bytes
        );
    }
    // RAM is always charged regardless of GPU presence — confirms the request was
    // accepted + estimated (and offload_kqv threaded) on ANY host.
    assert!(
        on.ram.needed_bytes > 0,
        "estimate accepted + threaded offload_kqv: {on:?}"
    );

    higgs.shutdown().await;
}

/// A Prepared, fitting model surfaces `Servable` readiness AND the `fit` detail (the
/// needed-vs-free numbers behind the badge) on `model_entries`. Fails-on-revert:
/// drop the `fit` field on `HiggsModelEntry` and the entry has no fit.
#[tokio::test]
async fn servable_model_entry_carries_fit_numbers() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP servable_model_entry_carries_fit_numbers: tiny gguf not found");
        return;
    };

    // Prepare → the tiny model gets a profile; serving is on by default and it
    // easily fits, so it reads back `Servable` with the fit numbers attached.
    higgs
        .tune(tune_req(TINY_MODEL_ID))
        .await
        .expect("prepare (tune)");
    let models = higgs.model_entries().await.expect("model_entries");
    let entry = models
        .iter()
        .find(|m| m.model.id == TINY_MODEL_ID)
        .expect("tiny model listed");
    assert_eq!(
        entry.readiness,
        ModelReadiness::Servable,
        "prepared tiny model is servable: {entry:?}"
    );
    let fit = entry
        .fit
        .as_ref()
        .expect("servable entry carries the fit detail");
    assert!(
        fit.needed_ram_bytes > 0,
        "fit reports the RAM the profile needs: {fit:?}"
    );
    // free_ram_bytes is a `u64` on the struct — present by construction.
    let _ = fit.free_ram_bytes;

    higgs.shutdown().await;
}

/// `model_entries` surfaces a store-read failure as HG040 instead of badging a
/// prepared model `Discovered` (the misleading state in exactly the persistence-
/// failure scenario). Fails-on-revert: collapse `tuning_profiles()` to empty-on-error
/// and the list returns Ok with `readiness: Discovered`.
#[tokio::test]
async fn models_list_with_unreadable_store_is_hg040() {
    use std::os::unix::fs::PermissionsExt;
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP models_list_with_unreadable_store_is_hg040: tiny gguf not found");
        return;
    };

    higgs
        .tune(tune_req(TINY_MODEL_ID))
        .await
        .expect("prepare (tune)");
    let mj = higgs.home().join("models.json");
    std::fs::set_permissions(&mj, std::fs::Permissions::from_mode(0o000))
        .expect("chmod models.json unreadable");

    let result = higgs.model_entries().await;
    // Restore perms so the temp dir cleans up regardless of the assertion outcome.
    let _ = std::fs::set_permissions(&mj, std::fs::Permissions::from_mode(0o644));

    let err = result.expect_err("model_entries fails when the store is unreadable");
    assert!(
        matches!(err, HiggsError::PersistenceFailed { .. }),
        "surfaces a persistence error, got {err:?}"
    );
    assert!(
        err.to_string().contains("[HG040]"),
        "the persistence error is HG040-coded: {err}"
    );

    higgs.shutdown().await;
}

// ── helpers ─────────────────────────────────────────────────────────────────

/// An `EstimateRequest` for `id` at a fixed context, with an optional budget.
fn estimate_req(id: &str, ctx_n: u32, budget: Option<ResourceBudget>) -> EstimateRequest {
    EstimateRequest {
        id: id.to_owned(),
        ctx_len: CtxLen::Fixed { n: ctx_n },
        gpu_layers: None,
        type_k: None,
        type_v: None,
        offload_kqv: None,
        cpu_moe: None,
        budget,
    }
}

/// A Suggest-mode `TuneRequest` (prepare) for `id`.
fn tune_req(id: &str) -> TuneRequest {
    TuneRequest {
        id: id.to_owned(),
        mode: None,
        budget: None,
        pins: None,
    }
}

/// PIDs of the `--higgs-worker` children of `parent_pid` (the spawned worker
/// processes). In-process, the parent is THIS test process.
fn worker_child_pids(parent_pid: u32) -> Vec<u32> {
    let out = std::process::Command::new("pgrep")
        .args(["-P", &parent_pid.to_string()])
        .output()
        .expect("run pgrep");
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| l.trim().parse::<u32>().ok())
        .collect()
}
