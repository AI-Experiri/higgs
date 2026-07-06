//! Black-box integration test for Turbotune (G6): `POST /api/higgs/models/tune`
//! with `mode: "benchmark"` actually LOADS + measures the tiny model and saves
//! a `provenance: "bench"` profile with a measured `bench_tps`.
//!
//! Fail-on-revert: restoring the benchmark stub (falls through to Suggest) makes
//! the response provenance "heuristic" with no bench_tps, failing the asserts.

mod common;

use common::{spawn_with_tiny_model, tiny_gguf_path, TINY_MODEL_ID};

#[tokio::test]
async fn benchmark_tune_measures_and_saves_bench_profile() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("SKIP turbotune: tiny gguf not found (set HIGGS_TEST_GGUF)");
        return;
    };
    let srv = spawn_with_tiny_model(13600, &gguf).await;
    let c = reqwest::Client::new();

    // Turbotune: mode=benchmark actually loads + measures the tiny model.
    let r = c
        .post(format!("{}/api/higgs/models/tune", srv.base))
        .json(&serde_json::json!({ "id": TINY_MODEL_ID, "mode": "benchmark" }))
        .send()
        .await
        .expect("benchmark tune request");
    assert!(r.status().is_success(), "benchmark tune: {}", r.status());
    let suggestion: serde_json::Value = r.json().await.expect("suggestion json");

    // The measured run reports Bench provenance, not the analytical Heuristic.
    assert_eq!(
        suggestion["provenance"], "Bench",
        "benchmark provenance: {suggestion}"
    );
    // A rationale line carries the measured throughput.
    let rationale = suggestion["rationale"]
        .as_array()
        .expect("rationale array")
        .iter()
        .filter_map(|v| v.as_str())
        .collect::<Vec<_>>()
        .join(" | ");
    assert!(
        rationale.contains("tok/s"),
        "rationale reports measured tok/s: {rationale}"
    );

    // The saved profile (readable via /api/higgs/models) carries the bench
    // provenance + a positive bench_tps — the measurement persisted.
    let persisted =
        std::fs::read_to_string(srv.home().join("models.json")).expect("models.json written");
    let store: serde_json::Value = serde_json::from_str(&persisted).expect("models.json parses");
    let rec = &store["models"][TINY_MODEL_ID]["tuning"];
    assert_eq!(rec["provenance"], "Bench", "saved provenance: {rec}");
    let bench_tps = rec["bench_tps"].as_f64().expect("bench_tps recorded");
    assert!(
        bench_tps > 0.0,
        "measured a positive gen tok/s: {bench_tps}"
    );

    // The model was torn down after the benchmark (no leaked resident worker).
    let status: serde_json::Value = c
        .get(format!("{}/api/higgs/status", srv.base))
        .send()
        .await
        .expect("status")
        .json()
        .await
        .expect("status json");
    assert!(
        status["loaded"].is_null(),
        "benchmark unloaded its worker: {status}"
    );
}

/// [HG067] over HTTP: a turbotune benchmark (`POST /api/higgs/models/tune`,
/// mode=benchmark) against a LOADED model is refused with 409 — the benchmark
/// owns its model exclusively (it loads/unloads candidate configs), so it must
/// not race a resident worker. The unit test `benchmark_refuses_a_loaded_model`
/// proves the guard; this proves it is reachable on the control surface and
/// mapped to a 409 the UI can act on.
/// Fail-on-revert: dropping the instances-loaded check in `turbotune_bench` lets
/// the benchmark start on the loaded model, so the 409 assertion fails.
#[tokio::test]
async fn benchmark_refuses_a_loaded_model_over_http() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("SKIP turbotune HG067: tiny gguf not found (set HIGGS_TEST_GGUF)");
        return;
    };
    let srv = spawn_with_tiny_model(13601, &gguf).await;
    let c = reqwest::Client::new();

    // Load the model so the benchmark's exclusive-ownership gate must refuse.
    let load = c
        .post(format!("{}/api/higgs/models/load", srv.base))
        .json(&serde_json::json!({ "id": TINY_MODEL_ID, "ctx_len": 512 }))
        .send()
        .await
        .expect("load request");
    assert_eq!(load.status(), 200, "the model loads: {}", load.status());

    // Turbotune now must refuse with 409 [HG067] rather than benchmark a resident model.
    let bench = c
        .post(format!("{}/api/higgs/models/tune", srv.base))
        .json(&serde_json::json!({ "id": TINY_MODEL_ID, "mode": "benchmark" }))
        .send()
        .await
        .expect("benchmark tune request");
    assert_eq!(
        bench.status(),
        409,
        "benchmark of a loaded model is a conflict"
    );
    let body: serde_json::Value = bench.json().await.expect("error json");
    let msg = body["error"].as_str().unwrap_or_default();
    assert!(
        msg.contains("HG067") && msg.contains("loaded"),
        "the 409 carries the HG067 model-loaded diagnostic: {body}"
    );

    // The model is still resident (the refusal didn't disturb it); SIGTERM on
    // drop tears it down — no SSE stream left open.
    let status: serde_json::Value = c
        .get(format!("{}/api/higgs/status", srv.base))
        .send()
        .await
        .expect("status")
        .json()
        .await
        .expect("status json");
    assert!(
        !status["loaded"].is_null(),
        "the loaded model survived the refused benchmark: {status}"
    );
}
