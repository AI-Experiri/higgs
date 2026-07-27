//! `/v1/models` is JIT-truthful: it lists SERVABLE (prepared, fitting,
//! unloaded) models alongside resident ones, because a chat against either
//! succeeds (JIT loads the servable one on demand). Fail-on-revert: with the
//! old resident-only listing, a prepared-but-unloaded model is absent.
//!
//! Migrated to the library-first API: control (prepare/status/JIT toggle) is the
//! in-process `Higgs` facade; the `/v1/models` union is exercised over the REAL
//! `/v1` HTTP surface via `serve_v1_local`.

mod common;

use common::{higgs_local, serve_v1_local, TINY_MODEL_ID};
use higgs::TuneRequest;

#[tokio::test]
async fn v1_models_lists_servable_unloaded_model() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP v1_models_servable: no tiny GGUF (set HIGGS_TEST_GGUF)");
        return;
    };
    let (base, serve) = serve_v1_local(higgs.handle()).await;
    let c = reqwest::Client::new();

    // Fresh instance: nothing loaded, nothing prepared → /v1/models is empty
    // (a discovered model is NOT servable; the JIT gate would refuse it).
    let ids = v1_ids(&c, &base).await;
    assert!(
        ids.is_empty(),
        "unprepared model must not be advertised: {ids:?}"
    );

    // Prepare (autotune) WITHOUT loading: readiness becomes servable, so the
    // model is now a valid JIT chat target and must be listed.
    higgs
        .tune(tune_req(TINY_MODEL_ID))
        .await
        .expect("prepare (tune)");
    let ids = v1_ids(&c, &base).await;
    assert!(
        ids.iter().any(|id| id == TINY_MODEL_ID),
        "servable (prepared, unloaded) model is advertised on /v1/models: {ids:?}"
    );

    // And it is genuinely unloaded — status shows no resident worker.
    let status = higgs.status().await.expect("status");
    assert_eq!(
        status.loaded_all.len(),
        0,
        "listing came from servability, not residency: {status:?}"
    );

    serve.shutdown().await;
    higgs.shutdown().await;
}

/// The `/v1/models` ids.
async fn v1_ids(c: &reqwest::Client, base: &str) -> Vec<String> {
    let v: serde_json::Value = c
        .get(format!("{base}/v1/models"))
        .send()
        .await
        .expect("get /v1/models")
        .json()
        .await
        .expect("models json");
    v["data"]
        .as_array()
        .expect("data array")
        .iter()
        .map(|m| m["id"].as_str().unwrap_or_default().to_owned())
        .collect()
}

/// JIT toggle contract: with the JIT gate OFF an unloaded servable model is
/// NOT chat-reachable (`ensure_loaded` refuses instead of loading), so it must
/// vanish from `/v1/models`; re-enabling brings it back. Fail-on-revert: drop
/// the `jit_enabled()` guard in `v1_models` → the disabled-phase assertion fails.
#[tokio::test]
async fn servable_listing_respects_jit_toggle() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP servable_listing_respects_jit_toggle: no tiny GGUF (set HIGGS_TEST_GGUF)");
        return;
    };
    let (base, serve) = serve_v1_local(higgs.handle()).await;
    let c = reqwest::Client::new();
    higgs
        .tune(tune_req(TINY_MODEL_ID))
        .await
        .expect("prepare (tune)");
    assert!(
        v1_ids(&c, &base).await.iter().any(|id| id == TINY_MODEL_ID),
        "servable model listed while JIT is on (default)"
    );

    higgs.set_jit_enabled(false);
    assert!(
        !v1_ids(&c, &base).await.iter().any(|id| id == TINY_MODEL_ID),
        "JIT off: unloaded servable model is unreachable, so it is NOT advertised"
    );

    higgs.set_jit_enabled(true);
    assert!(
        v1_ids(&c, &base).await.iter().any(|id| id == TINY_MODEL_ID),
        "JIT back on: servable model advertised again"
    );

    serve.shutdown().await;
    higgs.shutdown().await;
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
