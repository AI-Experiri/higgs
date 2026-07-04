//! `/v1/models` is JIT-truthful: it lists SERVABLE (prepared, fitting,
//! unloaded) models alongside resident ones, because a chat against either
//! succeeds (JIT loads the servable one on demand). Fail-on-revert: with the
//! old resident-only listing, a prepared-but-unloaded model is absent.

mod common;

use common::{prepare_tiny, spawn_with_tiny_model, tiny_gguf_path, TINY_MODEL_ID};

#[tokio::test]
async fn v1_models_lists_servable_unloaded_model() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("SKIP v1_models_servable: no tiny GGUF (set HIGGS_TEST_GGUF)");
        return;
    };
    let srv = spawn_with_tiny_model(13400, &gguf).await;
    let c = reqwest::Client::new();

    // Fresh server: nothing loaded, nothing prepared → /v1/models is empty
    // (a discovered model is NOT servable; the JIT gate would refuse it).
    let ids = v1_ids(&c, &srv.base).await;
    assert!(
        ids.is_empty(),
        "unprepared model must not be advertised: {ids:?}"
    );

    // Prepare (autotune) WITHOUT loading: readiness becomes servable, so the
    // model is now a valid JIT chat target and must be listed.
    prepare_tiny(&srv.base).await;
    let ids = v1_ids(&c, &srv.base).await;
    assert!(
        ids.iter().any(|id| id == TINY_MODEL_ID),
        "servable (prepared, unloaded) model is advertised on /v1/models: {ids:?}"
    );

    // And it is genuinely unloaded — status shows no resident worker.
    let status: serde_json::Value = c
        .get(format!("{}/api/higgs/status", srv.base))
        .send()
        .await
        .expect("get status")
        .json()
        .await
        .expect("status json");
    assert_eq!(
        status["loaded_all"].as_array().map(Vec::len),
        Some(0),
        "listing came from servability, not residency: {status}"
    );
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
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("SKIP servable_listing_respects_jit_toggle: no tiny GGUF (set HIGGS_TEST_GGUF)");
        return;
    };
    let srv = spawn_with_tiny_model(13401, &gguf).await;
    let c = reqwest::Client::new();
    prepare_tiny(&srv.base).await;
    assert!(
        v1_ids(&c, &srv.base)
            .await
            .iter()
            .any(|id| id == TINY_MODEL_ID),
        "servable model listed while JIT is on (default)"
    );

    let set = |jit: bool| {
        c.put(format!("{}/api/higgs/settings", srv.base)).json(&serde_json::json!({
            "jit_enabled": jit, "auto_unload_idle": false, "idle_ttl_minutes": 30, "serving_enabled": true
        })).send()
    };
    assert!(set(false)
        .await
        .expect("set settings")
        .status()
        .is_success());
    assert!(
        !v1_ids(&c, &srv.base)
            .await
            .iter()
            .any(|id| id == TINY_MODEL_ID),
        "JIT off: unloaded servable model is unreachable, so it is NOT advertised"
    );
    assert!(set(true).await.expect("set settings").status().is_success());
    assert!(
        v1_ids(&c, &srv.base)
            .await
            .iter()
            .any(|id| id == TINY_MODEL_ID),
        "JIT back on: servable model advertised again"
    );
}
