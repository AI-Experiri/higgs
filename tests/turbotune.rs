//! In-process integration test for Turbotune (G6): `higgs.tune(TuneRequest {
//! mode: Benchmark, .. })` actually LOADS + measures the tiny model and saves a
//! `provenance: Bench` profile with a measured `bench_tps`.
//!
//! Fail-on-revert: restoring the benchmark stub (falls through to Suggest) makes
//! the suggestion provenance `Heuristic` with no bench_tps, failing the asserts.

mod common;

use common::{higgs_local, TINY_MODEL_ID};
use higgs::tune::{TuneMode, TuneProvenance};
use higgs::worker::engine::{CtxLen, GpuLayers};
use higgs::{HiggsError, LoadParams, TuneRequest};

#[tokio::test]
async fn benchmark_tune_measures_and_saves_bench_profile() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP turbotune: tiny gguf not found (set HIGGS_TEST_GGUF)");
        return;
    };

    // Turbotune: mode=Benchmark actually loads + measures the tiny model.
    let suggestion = higgs
        .tune(TuneRequest {
            id: TINY_MODEL_ID.to_owned(),
            mode: Some(TuneMode::Benchmark),
            budget: None,
            pins: None,
        })
        .await
        .expect("benchmark tune measures a config");

    // The measured run reports Bench provenance, not the analytical Heuristic.
    assert_eq!(
        suggestion.provenance,
        TuneProvenance::Bench,
        "a measured benchmark stamps Bench provenance (not the Suggest fallback)"
    );
    // A rationale line carries the measured throughput.
    let rationale = suggestion.rationale.join(" | ");
    assert!(
        rationale.contains("tok/s"),
        "rationale reports measured tok/s: {rationale}"
    );

    // The saved profile (persisted to $HIGGS_HOME/models.json) carries the bench
    // provenance + a positive bench_tps — the measurement persisted.
    let persisted =
        std::fs::read_to_string(higgs.home().join("models.json")).expect("models.json written");
    let store: serde_json::Value = serde_json::from_str(&persisted).expect("models.json parses");
    let rec = &store["models"][TINY_MODEL_ID]["tuning"];
    assert_eq!(rec["provenance"], "Bench", "saved provenance: {rec}");
    let bench_tps = rec["bench_tps"].as_f64().expect("bench_tps recorded");
    assert!(
        bench_tps > 0.0,
        "measured a positive gen tok/s: {bench_tps}"
    );

    // The model was torn down after the benchmark (no leaked resident worker).
    let status = higgs.status().await.expect("status");
    assert!(
        status.loaded.is_none(),
        "benchmark unloaded its worker: {status:?}"
    );

    higgs.shutdown().await;
}

/// [HG067] via the facade: a turbotune benchmark (`tune` with mode=Benchmark)
/// against a LOADED model is refused — the benchmark owns its model exclusively
/// (it loads/unloads candidate configs), so it must not race a resident worker.
/// The unit test `benchmark_refuses_a_loaded_model` proves the guard; this proves
/// it is reachable through the crate control API and mapped to the typed
/// `BenchModelLoaded` error the UI (via serve's 409) can act on.
/// Fail-on-revert: dropping the instances-loaded check in `turbotune_bench` lets
/// the benchmark start on the loaded model, so the refusal assertion fails.
#[tokio::test]
async fn benchmark_refuses_a_loaded_model() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP turbotune HG067: tiny gguf not found (set HIGGS_TEST_GGUF)");
        return;
    };

    // Load the model so the benchmark's exclusive-ownership gate must refuse.
    higgs
        .load(
            TINY_MODEL_ID,
            Some(LoadParams::base(
                CtxLen::Fixed { n: 512 },
                GpuLayers::All,
                0,
            )),
        )
        .await
        .expect("the model loads");

    // Turbotune now must refuse [HG067] rather than benchmark a resident model.
    let err = higgs
        .tune(TuneRequest {
            id: TINY_MODEL_ID.to_owned(),
            mode: Some(TuneMode::Benchmark),
            budget: None,
            pins: None,
        })
        .await
        .expect_err("benchmark of a loaded model is refused");
    assert!(
        matches!(err, HiggsError::BenchModelLoaded { .. }),
        "the refusal is the HG067 model-loaded diagnostic: {err:?}"
    );
    // The rendered message carries the HG067 code + the "loaded" cue the 409
    // surfaces to the client.
    let msg = err.to_string();
    assert!(
        msg.contains("HG067") && msg.contains("loaded"),
        "the error carries the HG067 model-loaded diagnostic: {msg}"
    );

    // The model is still resident (the refusal didn't disturb it).
    let status = higgs.status().await.expect("status");
    assert!(
        status.loaded.is_some(),
        "the loaded model survived the refused benchmark: {status:?}"
    );

    higgs.shutdown().await;
}
