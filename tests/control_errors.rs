//! In-process integration tests for the CONTROL error + edge paths — now the
//! `Higgs` crate API (the old `/api/higgs/*` HTTP control surface is GONE).
//!
//! These focus on the failure/edge branches the happy-path lifecycle test
//! (`control_api.rs`) does not exercise: invalid/unknown model ids on
//! load/unload/tune/by-id, the `/system` snapshot shape (hardware + runtime +
//! config), the no-op `worker_stop` with nothing loaded, `version`, `nodes`
//! (local node present), relabeling the LOCAL instance, and the hub status shape
//! with no hub. Error asserts compare the TYPED `HiggsError` variant/code rather
//! than an HTTP envelope.
//!
//! Each test stages the tiny `stories260K.gguf` model (see `common`) and drives
//! the facade in-process. Every test SKIPs cleanly when the GGUF is absent.

mod common;

use common::{higgs_local, TINY_MODEL_ID};
use higgs::tune::TuneRequest;
use higgs::HiggsError;

/// A Suggest-mode `TuneRequest` (prepare) for `id`.
fn tune_req(id: &str) -> TuneRequest {
    TuneRequest {
        id: id.to_owned(),
        mode: None,
        budget: None,
        pins: None,
    }
}

/// `Higgs::load` with a path-traversal / structurally-invalid id is rejected by
/// the host-side `validate_repo_id` guard → `InvalidModelId` (HG015); it never
/// reaches the worker.
#[tokio::test]
async fn load_invalid_id_is_hg015_bad_request() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP load_invalid_id_is_hg015_bad_request: tiny gguf not found");
        return;
    };

    // Every one of these trips a DISTINCT branch of `validate_repo_id`:
    // a `..` traversal component, an absolute path, and an illegal character.
    for bad in ["../etc/passwd", "/abs/path", "bad id with spaces"] {
        let err = higgs
            .load(bad, None)
            .await
            .expect_err("invalid id is rejected by the id-validation guard");
        assert!(
            matches!(err, HiggsError::InvalidModelId { .. }),
            "invalid id {bad:?} is InvalidModelId: {err:?}"
        );
        assert!(
            err.to_string().contains("[HG015]"),
            "invalid id {bad:?} carries the HG015 code: {err}"
        );
    }

    higgs.shutdown().await;
}

/// `Higgs::load` with a well-formed but UNKNOWN id passes the validation guard,
/// then fails the host-side scan resolve → `ModelNotFound` (HG002).
#[tokio::test]
async fn load_unknown_id_is_hg002_not_found() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP load_unknown_id_is_hg002_not_found: tiny gguf not found");
        return;
    };

    let err = higgs
        .load("no-such-org/no-such-model", None)
        .await
        .expect_err("unknown id is not found on disk");
    assert!(
        matches!(err, HiggsError::ModelNotFound { .. }),
        "unknown id is ModelNotFound: {err:?}"
    );
    assert!(
        err.to_string().contains("[HG002]"),
        "unknown id carries the HG002 code: {err}"
    );

    higgs.shutdown().await;
}

/// `Higgs::unload_one` for a model that is NOT loaded is an idempotent no-op: it
/// resolves the served id, finds no resident worker, and returns Ok — it must NOT
/// error.
#[tokio::test]
async fn unload_not_loaded_model_is_ok_noop() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP unload_not_loaded_model_is_ok_noop: tiny gguf not found");
        return;
    };

    // The tiny model is staged + scannable but NOT loaded — unloading it by id is
    // a clean no-op (idempotent), distinct from the destructive drain-all.
    higgs
        .unload_one(TINY_MODEL_ID)
        .await
        .expect("unload of a not-loaded model is a no-op");

    // A never-existed id is also a clean no-op (resolve finds nothing → Ok).
    higgs
        .unload_one("ghost-org/ghost-model")
        .await
        .expect("unload of an unknown id is a no-op");

    higgs.shutdown().await;
}

/// `Higgs::model_by_id` on both branches: an UNKNOWN slashed id → `ModelNotFound`
/// (HG002), and the STAGED tiny id → the enriched (`not-loaded`, `gguf`,
/// llama-arch) entry.
#[tokio::test]
async fn model_by_id_unknown_404_and_staged_ok() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP model_by_id_unknown_404_and_staged_ok: tiny gguf not found");
        return;
    };

    // Unknown slashed id → ModelNotFound with the HG002 code (the not-found branch).
    let missing = higgs.model_by_id("ghost-org/ghost-model").await;
    let err = missing.expect_err("unknown id is not found");
    assert!(
        matches!(err, HiggsError::ModelNotFound { .. }),
        "by-id unknown is ModelNotFound: {err:?}"
    );
    assert!(
        err.to_string().contains("[HG002]"),
        "by-id not-found carries the HG002 code: {err}"
    );

    // Staged id → the enriched entry (the found branch; nothing loaded, so the
    // load state reads `not-loaded`).
    let entry = higgs
        .model_by_id(TINY_MODEL_ID)
        .await
        .expect("staged id is found");
    assert_eq!(entry.model.id, TINY_MODEL_ID, "by-id returns the entry");
    assert_eq!(entry.state, "not-loaded", "scanned-not-loaded entry");
    assert_eq!(entry.format, "gguf", "gguf format");
    assert_eq!(
        entry.model.arch.as_deref(),
        Some("llama"),
        "stories260K is a llama-arch GGUF"
    );

    higgs.shutdown().await;
}

/// `Higgs::tune` with a non-existent id fails the scan resolve → `ModelNotFound`
/// (HG002).
#[tokio::test]
async fn tune_bad_id_is_hg002_not_found() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP tune_bad_id_is_hg002_not_found: tiny gguf not found");
        return;
    };

    let err = higgs
        .tune(tune_req("no-such-org/no-such-model"))
        .await
        .expect_err("tune of an unknown id fails the scan resolve");
    assert!(
        matches!(err, HiggsError::ModelNotFound { .. }),
        "tune not-found is ModelNotFound: {err:?}"
    );
    assert!(
        err.to_string().contains("[HG002]"),
        "tune not-found carries the HG002 code: {err}"
    );

    higgs.shutdown().await;
}

/// `hardware()` + `version()` + `server_config()` — the hardware/runtime/config
/// three-panel shape of the old `/system` snapshot (CPU/RAM under hardware, the
/// engine identity under runtime, and the effective server config + limits under
/// config).
#[tokio::test]
async fn system_reports_hardware_runtime_config_shape() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP system_reports_hardware_runtime_config_shape: tiny gguf not found");
        return;
    };

    // hardware panel: a non-empty CPU name, a positive total RAM, and cores.
    let hw = higgs.hardware().await;
    assert!(
        !hw.cpu_name.is_empty(),
        "hardware.cpu_name is a non-empty string: {hw:?}"
    );
    assert!(
        hw.ram_total_bytes > 0,
        "hardware.ram_total_bytes > 0: {hw:?}"
    );
    assert!(hw.cpu_cores > 0, "hardware.cpu_cores > 0: {hw:?}");

    // runtime panel: the llama.cpp engine identity + a live engine version.
    let version = higgs.version();
    assert_eq!(version.engine, "llama.cpp", "runtime.engine is llama.cpp");
    assert!(
        !version.engine_version.is_empty(),
        "runtime.version is the live ggml version: {version:?}"
    );

    // config panel: the effective server config carries the limits, including the
    // live idle-unload TTL (default 60min × 60 = 3600s).
    assert_eq!(
        higgs.server_config().limits.idle_unload_ttl_secs,
        3600,
        "config.limits.idle_unload_ttl_secs is the live 60min×60"
    );

    higgs.shutdown().await;
}

/// `worker_stop` with NOTHING loaded is a graceful no-op: the bulk-unload is
/// best-effort and always Ok, and afterwards `status` reports no live worker.
#[tokio::test]
async fn worker_stop_with_nothing_loaded_is_ok() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP worker_stop_with_nothing_loaded_is_ok: tiny gguf not found");
        return;
    };

    higgs
        .worker_stop()
        .await
        .expect("worker_stop with nothing loaded is ok");

    // The facade is still usable and reports no live worker (spawn-on-load).
    let status = higgs.status().await.expect("status");
    assert!(
        !status.worker_alive,
        "no worker after a no-op worker_stop: {status:?}"
    );

    higgs.shutdown().await;
}

/// `version()` — the build/engine identity: a higgs version, the llama.cpp
/// engine, a non-empty engine + binding version, and `gguf` among the supported
/// formats.
#[tokio::test]
async fn version_reports_build_and_engine() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP version_reports_build_and_engine: tiny gguf not found");
        return;
    };

    let v = higgs.version();
    assert!(!v.higgs.is_empty(), "higgs version present: {v:?}");
    assert_eq!(v.engine, "llama.cpp", "engine reported: {v:?}");
    assert!(
        !v.engine_version.is_empty(),
        "engine_version present: {v:?}"
    );
    assert!(!v.binding.is_empty(), "binding version present: {v:?}");
    assert!(
        v.supported_formats.contains(&"gguf".to_owned()),
        "gguf is a supported format: {v:?}"
    );

    higgs.shutdown().await;
}

/// `nodes()` — the local node is a first-class node and appears FIRST even with
/// no fleet/hub installed (`endpoint_id == "local"`, `is_local`, always
/// connected, with a label + inventory).
#[tokio::test]
async fn nodes_lists_local_node_first() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP nodes_lists_local_node_first: tiny gguf not found");
        return;
    };

    let nodes = higgs.nodes().await;
    assert_eq!(
        nodes.len(),
        1,
        "only the local node with no fleet: {nodes:?}"
    );
    let local = &nodes[0];
    assert_eq!(local.endpoint_id, "local", "local sentinel id: {local:?}");
    assert!(local.is_local, "flagged local: {local:?}");
    assert!(local.connected, "local is always connected: {local:?}");
    assert!(!local.label.is_empty(), "local node has a label: {local:?}");
    assert!(
        local.inventory.is_some(),
        "local inventory present: {local:?}"
    );

    higgs.shutdown().await;
}

/// `node_label("local", ...)` renames THIS instance: it writes the new name into
/// the isolated `config.json` (under the temp `HIGGS_HOME`) and the next `nodes()`
/// shows the local node's new label. (`node == "local"` is the one relabel branch
/// that needs no hub.)
#[tokio::test]
async fn nodes_label_renames_local_instance() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP nodes_label_renames_local_instance: tiny gguf not found");
        return;
    };

    let new_name = "renamed-local-box";
    let renamed = higgs
        .node_label("local", new_name)
        .await
        .expect("local relabel succeeds (no hub needed)");
    assert!(renamed, "relabel of the local node reports renamed");

    // The local node now reports the new label.
    let nodes = higgs.nodes().await;
    let local = nodes
        .iter()
        .find(|n| n.endpoint_id == "local")
        .expect("local node present");
    assert_eq!(
        local.label, new_name,
        "the local node shows the new name: {local:?}"
    );

    higgs.shutdown().await;
}

/// The hub status with no hub installed reports disabled, no hub id, and zero
/// nodes (lets the Fleet tab show an explicit "hub off" state). `hub_disable` is
/// idempotent — a no-op returning the current (disabled) status when no hub is
/// installed.
#[tokio::test]
async fn hub_status_without_hub_is_disabled() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP hub_status_without_hub_is_disabled: tiny gguf not found");
        return;
    };

    let status = higgs.hub_disable().await;
    assert!(!status.enabled, "no hub installed → disabled: {status:?}");
    assert_eq!(status.node_count, 0, "no nodes when disabled: {status:?}");
    assert!(
        status.hub_id.is_none(),
        "hub_id omitted/null when disabled: {status:?}"
    );

    higgs.shutdown().await;
}
