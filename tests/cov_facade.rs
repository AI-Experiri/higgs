//! In-process integration tests for the `Higgs` CONTROL FACADE error/edge paths
//! (`src/api.rs`, `src/api/embed.rs`, `src/api/guards.rs`) that the happy-path
//! lifecycle suites (`control_api.rs`, `control_errors.rs`, `control_fleet_routes.rs`)
//! do not reach.
//!
//! These drive the crate API directly (higgs is library-first — the old
//! `/api/higgs/*` HTTP surface is gone) against a REAL local llama.cpp worker (the
//! tiny `stories260K.gguf` via `common::higgs_local`). Every test SKIPs cleanly when
//! the GGUF is absent. Targets:
//!   * `prepare_chat` gate — JIT load, NotPrepared/ProfileStale/ModelNotLoaded, the
//!     prompt-byte/budget estimators.
//!   * node-mutation ops with NO hub → `HubControlFailed` (not-a-hub).
//!   * config/models-store persistence failures → best-effort warn vs hard error.
//!   * `servable_model_ids`, `register_internal_token`, `mint_key` empty-scope reject.
//!   * `chat_stream` unknown-model routing + the admission-gate `ServerBusy` flood.
//!   * the local node-view worker inventory, `events`/`subscribe_logs`, and the
//!     `validate_repo_id` NUL-byte guard.

mod common;

use std::time::Duration;

use common::{higgs_local, TINY_MODEL_ID};
use higgs::keys::Scope;
use higgs::tune::{TunePins, TuneRequest};
use higgs::worker::engine::CtxLen;
use higgs::{HiggsError, HiggsEvent, SamplingParams};

/// A Suggest-mode `TuneRequest` (prepare) for `id`.
fn tune_req(id: &str) -> TuneRequest {
    TuneRequest {
        id: id.to_owned(),
        mode: None,
        budget: None,
        pins: None,
    }
}

/// `prepare_chat` — the shared `/v1` chat gate lifted onto the facade. Walks the
/// gate states: un-Prepared → `NotPrepared`; after `tune` it JIT-loads and returns a
/// context-clamped budget; the prompt-byte estimator tolerates non-array / non-text
/// message content; and with JIT OFF an unloaded model is `ModelNotLoaded`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn prepare_chat_gate_states_and_budget() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP prepare_chat_gate_states_and_budget: tiny gguf not found");
        return;
    };
    let id = TINY_MODEL_ID;
    let msgs = r#"[{"role":"user","content":"hello there"}]"#;

    // ── JIT on (default) but never Prepared → NotPrepared (HG046). ──
    let err = higgs
        .prepare_chat(id, Some(8), msgs)
        .await
        .expect_err("un-prepared model is refused by the readiness gate");
    assert!(
        matches!(err, HiggsError::NotPrepared { .. }),
        "un-prepared JIT chat → NotPrepared: {err:?}"
    );
    assert!(err.to_string().contains("[HG046]"), "carries HG046: {err}");

    // ── Prepare, then prepare_chat JIT-loads the fresh profile and clamps the budget. ──
    higgs.tune(tune_req(id)).await.expect("prepare (tune)");
    let prep = higgs
        .prepare_chat(id, Some(8), msgs)
        .await
        .expect("prepared model resolves + JIT loads");
    assert_eq!(
        prep.resolved_model, id,
        "resolved served id is the model: {prep:?}"
    );
    assert!(
        prep.max_gen >= 1 && prep.max_gen <= 8,
        "budget clamped to the requested 8 (fits the window): {prep:?}"
    );

    // The model is now resident, so the remaining calls resolve without a reload and
    // exercise the prompt-byte estimator's tolerant branches.
    // Non-array messages_json → the estimator returns 0 (no prompt bytes) → still Ok.
    let prep2 = higgs
        .prepare_chat(id, Some(4), "not-a-json-array")
        .await
        .expect("non-array messages_json is tolerated (prompt estimate 0)");
    assert_eq!(
        prep2.max_gen, 4,
        "requested budget honored on a 0-byte prompt: {prep2:?}"
    );

    // A message whose `content` is a non-text scalar (number) → the estimator counts
    // it as 0 bytes rather than erroring.
    let prep3 = higgs
        .prepare_chat(id, Some(6), r#"[{"role":"user","content":123}]"#)
        .await
        .expect("non-text scalar content is tolerated by the estimator");
    assert!(
        prep3.max_gen >= 1,
        "still yields a positive budget: {prep3:?}"
    );

    // ── JIT OFF + not loaded → ModelNotLoaded (HG003). ──
    higgs.unload().await.expect("unload");
    higgs.set_jit_enabled(false);
    let err = higgs
        .prepare_chat(id, None, msgs)
        .await
        .expect_err("JIT off + unloaded model is refused");
    assert!(
        matches!(err, HiggsError::ModelNotLoaded { .. }),
        "JIT-off unloaded chat → ModelNotLoaded: {err:?}"
    );

    higgs.shutdown().await;
}

/// `prepare_chat` on a model whose saved profile is STALE (the GGUF changed since
/// Prepare, so its `file_sig` no longer matches) hard-blocks with `ProfileStale`
/// (HG047) — the JIT gate refuses a profile that may no longer fit.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn prepare_chat_stale_profile_is_hg047() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP prepare_chat_stale_profile_is_hg047: tiny gguf not found");
        return;
    };
    let id = TINY_MODEL_ID;

    // Prepare anchors the profile to the CURRENT model-file signature.
    higgs.tune(tune_req(id)).await.expect("prepare (tune)");

    // Mutate the staged GGUF so its length+mtime (the `file_sig`) changes — the saved
    // profile's anchor no longer matches → it reads as stale. Appending trailing bytes
    // leaves the GGUF header intact, so the model still scans (only the sig differs).
    {
        use std::io::Write;
        let path = higgs.staged_gguf(id);
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .expect("open staged gguf to mutate its signature");
        f.write_all(&[0u8; 128]).expect("append bytes");
        f.flush().expect("flush");
    }

    let err = higgs
        .prepare_chat(id, Some(8), r#"[{"role":"user","content":"hi"}]"#)
        .await
        .expect_err("a stale profile hard-blocks the JIT chat");
    assert!(
        matches!(err, HiggsError::ProfileStale { .. }),
        "changed model file → ProfileStale: {err:?}"
    );
    assert!(err.to_string().contains("[HG047]"), "carries HG047: {err}");

    higgs.shutdown().await;
}

/// The node-mutation facade methods with NO hub/fleet installed all fail with
/// `HubControlFailed` (the "not running as a hub" arm) — the `Err` branch the
/// live-fleet suite never hits because it enables the hub first.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn node_ops_without_a_hub_are_not_a_hub_errors() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP node_ops_without_a_hub_are_not_a_hub_errors: tiny gguf not found");
        return;
    };

    let load = higgs.node_load("some-node", TINY_MODEL_ID).await;
    let unload = higgs.node_unload(TINY_MODEL_ID).await;
    let retire = higgs.node_retire("some-node").await;
    let scan = higgs.node_scan("some-node").await;

    for (name, r) in [
        ("node_load", load.map(|_| ())),
        ("node_unload", unload),
        ("node_retire", retire),
        ("node_scan", scan.map(|_| ())),
    ] {
        let err = r.expect_err(&format!("{name} with no hub must fail"));
        assert!(
            matches!(err, HiggsError::HubControlFailed { .. }),
            "{name} with no hub → HubControlFailed: {err:?}"
        );
        assert!(
            err.to_string().contains("not running as a hub"),
            "{name} error names the not-a-hub condition: {err}"
        );
        assert!(
            err.to_string().contains("hub_enable"),
            "{name} error states the remedy (enable the hub): {err}"
        );
    }

    higgs.shutdown().await;
}

/// Config-persistence failures. When `config.json` can't be written:
///   * `node_label("local", …)` (the one relabel branch that writes config.json)
///     surfaces a hard `PersistenceFailed`;
///   * a `load` still SUCCEEDS — the per-model record write is best-effort — and
///   * with no persisted record surviving, a secondary worker's status card falls
///     back to a param-less stub (`ctx_len == None`). `unload_spec(None)` then drains.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn config_persistence_failures_are_hard_vs_best_effort() {
    let Some(higgs) = higgs_local(&["cov/a", "cov/b"]).await else {
        eprintln!("SKIP config_persistence_failures_are_hard_vs_best_effort: tiny gguf not found");
        return;
    };

    // Occupy the config.json path with a DIRECTORY so every read/write of it fails.
    let cfg = higgs.home().join("config.json");
    std::fs::create_dir(&cfg).expect("occupy config.json with a directory");

    // Relabelling the LOCAL node writes config.json → a hard PersistenceFailed.
    let err = higgs
        .node_label("local", "new-name")
        .await
        .expect_err("local relabel must fail when config.json is unwritable");
    assert!(
        matches!(err, HiggsError::PersistenceFailed { .. }),
        "unwritable config.json → PersistenceFailed: {err:?}"
    );

    // A load still succeeds — the per-model config record write is best-effort (logged,
    // never fatal). Load two so the status snapshot has a primary + a secondary.
    higgs
        .load("cov/a", None)
        .await
        .expect("load cov/a despite unwritable config");
    higgs
        .load("cov/b", None)
        .await
        .expect("load cov/b despite unwritable config");

    let status = higgs.status().await.expect("status");
    assert_eq!(
        status.loaded_all.len(),
        2,
        "both workers reported: {status:?}"
    );
    // No config record survived (config.json is a dir), so a stub-built card has no
    // load params — at least one entry reports `ctx_len == None`.
    assert!(
        status.loaded_all.iter().any(|e| e.ctx_len.is_none()),
        "a record-less worker card falls back to a param-less stub: {status:?}"
    );

    // `unload_spec(None)` drains ALL resident workers (the id-absent branch).
    higgs
        .unload_spec(None)
        .await
        .expect("unload_spec(None) drains all");
    let status = higgs.status().await.expect("status");
    assert!(
        status.loaded.is_none(),
        "everything unloaded after drain: {status:?}"
    );

    higgs.shutdown().await;
}

/// `servable_model_ids` — advertises exactly the Prepared+fitting catalog ids. A
/// never-tuned model is not servable (Discovered), a tuned one is, and a store-read
/// failure degrades to an empty list rather than propagating.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn servable_model_ids_reflects_readiness_and_store_health() {
    use std::os::unix::fs::PermissionsExt;
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!(
            "SKIP servable_model_ids_reflects_readiness_and_store_health: tiny gguf not found"
        );
        return;
    };

    // Never Prepared → not servable.
    assert!(
        higgs.servable_model_ids().await.is_empty(),
        "an un-prepared catalog advertises nothing servable"
    );

    // Prepared → servable (fresh profile, fits, serving on, not loaded).
    higgs
        .tune(tune_req(TINY_MODEL_ID))
        .await
        .expect("prepare (tune)");
    let servable = higgs.servable_model_ids().await;
    assert!(
        servable.contains(&TINY_MODEL_ID.to_owned()),
        "a prepared, fitting model is servable: {servable:?}"
    );

    // An unreadable store degrades servable-ids to empty (best-effort), never an error.
    let mj = higgs.home().join("models.json");
    std::fs::set_permissions(&mj, std::fs::Permissions::from_mode(0o000))
        .expect("chmod models.json unreadable");
    let degraded = higgs.servable_model_ids().await;
    let _ = std::fs::set_permissions(&mj, std::fs::Permissions::from_mode(0o644));
    assert!(
        degraded.is_empty(),
        "an unreadable store yields no servable ids (fail-soft): {degraded:?}"
    );

    higgs.shutdown().await;
}

/// `register_internal_token` adds a HIDDEN in-memory key (turning auth ON) that is
/// never listed by `visible()`, and `mint_key` with an EXPLICIT empty scope list is a
/// client error (`InvalidRequest`) regardless of store state.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn internal_token_arms_auth_and_empty_scopes_reject() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP internal_token_arms_auth_and_empty_scopes_reject: tiny gguf not found");
        return;
    };

    // Empty keystore = auth OFF.
    assert!(higgs.api_keys().is_empty(), "auth starts off (no keys)");

    // Registering an internal token arms auth but stays hidden from the visible list.
    higgs.register_internal_token("internal-secret-token", vec![Scope::Admin]);
    assert!(
        !higgs.api_keys().is_empty(),
        "the internal token arms auth (live keystore non-empty)"
    );
    assert_eq!(
        higgs.api_keys().visible().count(),
        0,
        "the internal token is hidden — never shown in the key list"
    );

    // An explicit empty scope list is rejected up front.
    let err = higgs
        .mint_key("empty", Some(vec![]))
        .expect_err("explicit empty scopes are rejected");
    assert!(
        matches!(err, HiggsError::InvalidKeyRequest { .. }),
        "empty scopes → InvalidKeyRequest: {err:?}"
    );

    higgs.shutdown().await;
}

/// `chat_stream` for a model that is neither locally resident nor remote-routed
/// resolves to `ModelNotFound` — `chat_stream` itself does no JIT load (that is the
/// gate's job), so an unloaded id has nowhere to route.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chat_stream_unrouted_model_is_not_found() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP chat_stream_unrouted_model_is_not_found: tiny gguf not found");
        return;
    };

    let r = higgs
        .chat_stream(
            "ghost-org/ghost-model".to_owned(),
            r#"[{"role":"user","content":"hi"}]"#.to_owned(),
            4,
            SamplingParams::default(),
            None,
            None,
        )
        .await;
    match r {
        Err(HiggsError::ModelNotFound { .. }) => {}
        other => panic!("unrouted chat_stream → ModelNotFound, got {other:?}"),
    }

    higgs.shutdown().await;
}

/// The local `NodeView` inventory lists every resident worker with its `/v1` served
/// id — the worker-mapping the empty-fleet `nodes()` test never populates.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn local_node_view_lists_resident_workers() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP local_node_view_lists_resident_workers: tiny gguf not found");
        return;
    };

    higgs.load(TINY_MODEL_ID, None).await.expect("load");
    let nodes = higgs.nodes().await;
    let local = nodes
        .iter()
        .find(|n| n.endpoint_id == "local")
        .expect("local node present");
    let inv = local.inventory.as_ref().expect("local inventory present");
    let worker = inv
        .workers
        .iter()
        .find(|w| w.served_id == TINY_MODEL_ID)
        .unwrap_or_else(|| panic!("resident worker listed with its served id: {inv:?}"));
    assert_eq!(
        worker.model, TINY_MODEL_ID,
        "worker carries the raw model id: {worker:?}"
    );

    higgs.shutdown().await;
}

/// `events()` fans out a `ModelLoaded` on load, and `subscribe_logs()` + `logs()`
/// surface the worker's stderr in the Developer-Log bus.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn load_emits_model_loaded_event_and_worker_logs() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP load_emits_model_loaded_event_and_worker_logs: tiny gguf not found");
        return;
    };

    let mut ev_rx = higgs.events();
    let mut log_rx = higgs.subscribe_logs();

    higgs.load(TINY_MODEL_ID, None).await.expect("load");

    // The ModelLoaded event was buffered on the broadcast (subscribed before the load).
    let mut got_loaded = false;
    for _ in 0..50 {
        match tokio::time::timeout(Duration::from_millis(200), ev_rx.recv()).await {
            Ok(Ok(HiggsEvent::ModelLoaded { id })) if id == TINY_MODEL_ID => {
                got_loaded = true;
                break;
            }
            // Other event / lagged / timeout tick — keep polling within the budget.
            _ => {}
        }
    }
    assert!(
        got_loaded,
        "events() delivered ModelLoaded for the loaded model"
    );

    // The worker's load stderr reaches the Developer-Log snapshot (relayed async).
    let mut has_logs = false;
    for _ in 0..40 {
        if !higgs.logs(500, None).is_empty() {
            has_logs = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert!(
        has_logs,
        "worker stderr surfaced in the Developer-Log snapshot"
    );
    // Drain one live line if present (exercises the live subscription path).
    let _ = tokio::time::timeout(Duration::from_millis(200), log_rx.recv()).await;

    higgs.shutdown().await;
}

/// A model id containing a NUL byte is rejected by the host-side charset guard
/// (`validate_repo_id`) before any filesystem use → `InvalidModelId` (HG015).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn load_id_with_nul_byte_is_invalid_model_id() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP load_id_with_nul_byte_is_invalid_model_id: tiny gguf not found");
        return;
    };

    let err = higgs
        .load("bad\0id", None)
        .await
        .expect_err("a NUL byte in the id is rejected");
    assert!(
        matches!(err, HiggsError::InvalidModelId { .. }),
        "NUL-byte id → InvalidModelId: {err:?}"
    );
    assert!(err.to_string().contains("[HG015]"), "carries HG015: {err}");

    higgs.shutdown().await;
}

/// A `tune` (Prepare) whose profile can't be persisted fails LOUDLY with
/// `PersistenceFailed` (HG040) — a saved profile is a serving precondition, so a
/// Prepare that can't write must not report success.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn tune_persist_failure_is_hg040() {
    use std::os::unix::fs::PermissionsExt;
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP tune_persist_failure_is_hg040: tiny gguf not found");
        return;
    };

    // Make HIGGS_HOME read-only so the store OPENS (models.json absent → empty) but the
    // atomic FLUSH (write temp + rename in the dir) fails.
    let home = higgs.home().to_path_buf();
    std::fs::set_permissions(&home, std::fs::Permissions::from_mode(0o500))
        .expect("chmod home read-only");
    let result = higgs.tune(tune_req(TINY_MODEL_ID)).await;
    // Restore before asserting so the temp dir can clean up regardless of outcome.
    let _ = std::fs::set_permissions(&home, std::fs::Permissions::from_mode(0o755));

    let err = result.expect_err("tune must fail when the profile can't be persisted");
    assert!(
        matches!(err, HiggsError::PersistenceFailed { .. }),
        "unwritable store → PersistenceFailed: {err:?}"
    );
    assert!(err.to_string().contains("[HG040]"), "carries HG040: {err}");

    higgs.shutdown().await;
}

/// A Suggest-mode `tune` that carries `pins` still SUCCEEDS — pins steer the
/// Benchmark search only and are ignored (with a log note) in Suggest mode.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn tune_suggest_mode_ignores_pins() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP tune_suggest_mode_ignores_pins: tiny gguf not found");
        return;
    };

    let req = TuneRequest {
        id: TINY_MODEL_ID.to_owned(),
        mode: None, // Suggest
        budget: None,
        pins: Some(TunePins {
            ctx_len: Some(CtxLen::Fixed { n: 8192 }),
            ..Default::default()
        }),
    };
    // The pins are contradictory for Suggest but must not fail it.
    let suggestion = higgs
        .tune(req)
        .await
        .expect("Suggest-mode tune ignores pins and still succeeds");
    // A real analytical suggestion came back (RAM is always charged).
    assert!(
        suggestion.ram_fit.needed_bytes > 0,
        "the suggestion carries a footprint: {suggestion:?}"
    );

    higgs.shutdown().await;
}

/// Flooding the chat admission gate past `MAX_CONCURRENT_INFERENCE` returns
/// `ServerBusy` (HTTP 503) for the over-cap requests — the no-auth server's flood
/// backstop. The worker serialises generation, so admitted requests hold their permit
/// (queue-wait + generation) long enough for the surplus to be refused.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chat_flood_past_admission_gate_is_server_busy() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP chat_flood_past_admission_gate_is_server_busy: tiny gguf not found");
        return;
    };

    higgs.load(TINY_MODEL_ID, None).await.expect("load");

    let msgs = r#"[{"role":"user","content":"Write a long story about the sea and its tides."}]"#;
    let mut streams = Vec::new();
    let mut busy = 0usize;
    for _ in 0..24 {
        match higgs
            .chat_stream(
                TINY_MODEL_ID.to_owned(),
                msgs.to_owned(),
                64,
                SamplingParams::default(),
                None,
                None,
            )
            .await
        {
            Ok(pair) => streams.push(pair),
            Err(HiggsError::ServerBusy { max, .. }) => {
                assert_eq!(max, 8, "the reported cap is MAX_CONCURRENT_INFERENCE");
                busy += 1;
            }
            Err(e) => panic!("unexpected chat_stream error under flood: {e:?}"),
        }
    }
    assert!(
        busy >= 1,
        "flooding past the admission gate returns ServerBusy 503 (admitted {}, busy {busy})",
        streams.len()
    );
    assert!(!streams.is_empty(), "at least one request was admitted");

    // Drain every admitted stream fully and join its task (never leave a stream open —
    // it would hang graceful shutdown) so the permits release.
    for (mut rx, handle) in streams {
        while rx.recv().await.is_some() {}
        let _ = handle.await;
    }

    higgs.shutdown().await;
}
