//! Round-2 in-process integration coverage for facade + scan paths the round-1
//! suites (`cov_facade.rs`, `cov_serve.rs`, `cov_worker.rs`) and the pre-existing
//! control/serve suites leave uncovered. Every test drives the crate API against a
//! REAL local llama.cpp worker (the tiny `stories260K.gguf` via
//! `common::higgs_local`) and SKIPs cleanly when the GGUF is absent. Targets:
//!
//!   * `Higgs::load_flat` (`src/api/embed.rs`) — the flat `HiggsLoadRequest` load
//!     shape's THREE branches: no-pin → default load, partial flat pins → merge
//!     over `default_load`, and a full `params` umbrella that supersedes the flat
//!     fields. No round-1/existing test calls `load_flat`.
//!   * The `ModelLoadPhase::Failed` terminal load-event emission (`src/api.rs`
//!     `load_inner`) — a corrupt-GGUF load pushes a `Failed` phase carrying the
//!     diagnostic code. `higgs_events.rs` covers only the happy `…→Ready` path.
//!   * The DEEP `HG001 ModelDirUnreadable` arm in the LM-Studio walker
//!     (`src/worker/models.rs`) — an unreadable NESTED model dir (not the root)
//!     makes the innermost `read_dir` fail EACCES and aborts the scan with the
//!     typed error. Round-1's `cov_worker.rs` only exercises a root-is-a-file.

mod common;

use common::higgs_local;
use higgs::api::ModelLoadPhase;
use higgs::serve::HiggsLoadRequest;
use higgs::worker::engine::{CtxLen, GpuLayers};
use higgs::{HiggsError, LoadParams};
use serde_json::json;

/// `load_flat` builds the effective [`LoadParams`] from a flat `HiggsLoadRequest`:
///   * no pinned field + no `params`   → `None`   → a fully-default load;
///   * a partial flat pin (`ctx_len`)  → merged over `default_load` (base fields
///     fall back to the config default, pinned field applied);
///   * a full `params` umbrella        → supersedes every flat field verbatim.
/// Three DISTINCT model ids so all three loads are resident at once (the node is
/// additive), letting one status snapshot prove each branch's params landed.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn load_flat_none_partial_and_full_param_shapes() {
    let Some(higgs) = higgs_local(&["cov/a", "cov/b", "cov/c"]).await else {
        eprintln!("SKIP load_flat_none_partial_and_full_param_shapes: tiny gguf not found");
        return;
    };

    // (a) No pinned fields, no `params` → the None branch → a fully-default load.
    let req_a: HiggsLoadRequest =
        serde_json::from_value(json!({ "id": "cov/a" })).expect("build flat request a");
    higgs
        .load_flat(&req_a)
        .await
        .expect("no-pin flat load uses the default params");

    // (b) A partial flat pin (ctx_len only) → merged over `default_load`.
    let req_b: HiggsLoadRequest = serde_json::from_value(json!({
        "id": "cov/b",
        "ctx_len": { "kind": "fixed", "n": 256 }
    }))
    .expect("build flat request b");
    higgs
        .load_flat(&req_b)
        .await
        .expect("partial-pin flat load merges over the default base");

    // (c) A full `params` umbrella → supersedes the flat fields verbatim.
    let full = serde_json::to_value(LoadParams::base(CtxLen::fixed(384), GpuLayers::count(0), 0))
        .expect("serialize full LoadParams");
    let req_c: HiggsLoadRequest = serde_json::from_value(json!({
        "id": "cov/c",
        "params": full
    }))
    .expect("build flat request c");
    higgs
        .load_flat(&req_c)
        .await
        .expect("full-params flat load supersedes the flat fields");

    // One snapshot proves all three loads landed with the expected params.
    let st = higgs.status().await.expect("status");
    assert_eq!(
        st.loaded_all.len(),
        3,
        "all three flat loads are resident: {st:?}"
    );
    let by = |id: &str| {
        st.loaded_all
            .iter()
            .find(|e| e.id == id)
            .unwrap_or_else(|| panic!("{id} resident: {st:?}"))
    };
    // (a) default load: the live worker probe reports a concrete window.
    assert!(
        by("cov/a").ctx_len.is_some(),
        "the no-pin default load reports a context window: {st:?}"
    );
    // (b) the partial flat `ctx_len` pin is the loaded window.
    assert_eq!(
        by("cov/b").ctx_len,
        Some(CtxLen::Fixed { n: 256 }),
        "the partial-pinned ctx_len is applied: {st:?}"
    );
    // (c) the full `params` umbrella's ctx_len + gpu_layers both won.
    assert_eq!(
        by("cov/c").ctx_len,
        Some(CtxLen::Fixed { n: 384 }),
        "the full-params ctx_len supersedes the flat fields: {st:?}"
    );
    assert_eq!(
        by("cov/c").gpu_layers,
        Some(GpuLayers::Count { n: 0 }),
        "the full-params gpu_layers supersedes the flat fields: {st:?}"
    );

    higgs.shutdown().await;
}

/// A load that FAILS in the real worker (a corrupt, non-GGUF model file) pushes a
/// terminal `ModelLoadPhase::Failed` load event carrying the failure's diagnostic
/// code — the `Err` arm of `load_inner`'s terminal bracket, which the happy-path
/// events test never reaches. Fail-on-revert: dropping the `Failed` emit leaves the
/// stream with no terminal Failed phase, failing the assertion.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn failed_load_pushes_failed_phase_with_diagnostic_code() {
    let Some(higgs) = higgs_local(&["cov/bad"]).await else {
        eprintln!("SKIP failed_load_pushes_failed_phase_with_diagnostic_code: tiny gguf not found");
        return;
    };

    // Overwrite the staged GGUF with garbage (no GGUF magic) so the worker's
    // llama.cpp load rejects it — a real, coded load failure.
    std::fs::write(
        higgs.staged_gguf("cov/bad"),
        b"this file is not a valid gguf model",
    )
    .expect("overwrite staged gguf with garbage");

    // Subscribe BEFORE the load so no phase is missed (there is no replay ring).
    let mut rx = higgs.subscribe_load_events();

    let err = higgs
        .load("cov/bad", None)
        .await
        .expect_err("a corrupt gguf load must fail");
    assert!(
        !err.to_string().is_empty(),
        "the load failure carries a diagnostic: {err}"
    );

    // Drain the pushed phases: a terminal `Failed` (with a diagnostic code) must
    // appear, and the load must never report `Ready`.
    let mut failed_code: Option<Option<String>> = None;
    let mut phases = Vec::new();
    while let Ok(ev) = rx.try_recv() {
        assert_eq!(ev.id, "cov/bad", "the event carries the failing model id");
        if ev.phase == ModelLoadPhase::Failed {
            failed_code = Some(ev.code.clone());
        }
        phases.push(ev.phase);
    }
    let code =
        failed_code.unwrap_or_else(|| panic!("a terminal Failed phase was pushed: {phases:?}"));
    assert!(
        code.is_some(),
        "the Failed phase carries a diagnostic code (not None): {phases:?}"
    );
    assert!(
        !phases.contains(&ModelLoadPhase::Ready),
        "a failed load never reports Ready: {phases:?}"
    );

    higgs.shutdown().await;
}

/// The LM-Studio walker's DEEP `HG001 ModelDirUnreadable` arm: an unreadable
/// NESTED model directory (the leaf dir holding the GGUF, made mode-000) fails the
/// innermost `read_dir` with `EACCES` — a non-`NotFound` error the walker surfaces
/// as the typed `HG001`, aborting the whole scan rather than silently dropping the
/// model. Distinct from round-1's root-is-a-file case, which hits the OUTER
/// `read_dir`. SKIPs under root (mode bits are ignored, so the scan would succeed).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn scan_aborts_on_unreadable_nested_model_dir_is_hg001() {
    use std::os::unix::fs::PermissionsExt;

    let Some(higgs) = higgs_local(&["cov/locked"]).await else {
        eprintln!("SKIP scan_aborts_on_unreadable_nested_model_dir_is_hg001: tiny gguf not found");
        return;
    };
    // Root ignores the mode bits, so a mode-000 dir stays readable and the scan
    // would succeed — the assertion would be vacuous. Skip.
    if unsafe { libc::geteuid() } == 0 {
        eprintln!("SKIP scan_aborts_on_unreadable_nested_model_dir_is_hg001: running as root");
        higgs.shutdown().await;
        return;
    }

    // The leaf model dir holding the staged GGUF (`<scan>/cov/locked`).
    let model_dir = higgs
        .staged_gguf("cov/locked")
        .parent()
        .expect("staged gguf has a parent dir")
        .to_path_buf();
    std::fs::set_permissions(&model_dir, std::fs::Permissions::from_mode(0o000))
        .expect("chmod the nested model dir unreadable");

    let res = higgs.model_entries().await;

    // Restore BEFORE asserting so the TempDir cleanup can remove the dir regardless.
    let _ = std::fs::set_permissions(&model_dir, std::fs::Permissions::from_mode(0o755));

    let err = res.expect_err("an unreadable nested model dir aborts the scan");
    assert!(
        matches!(err, HiggsError::ModelDirUnreadable { .. }),
        "unreadable nested model dir → HG001 ModelDirUnreadable: {err:?}"
    );
    assert!(
        err.to_string().contains("[HG001]"),
        "the abort carries the HG001 code: {err}"
    );

    higgs.shutdown().await;
}
