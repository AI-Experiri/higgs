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
