//! STEP-1 fail-on-revert test for the `HiggsConfig::worker_exe` DI seam.
//!
//! higgs runs the model in a WORKER PROCESS it spawns by re-execing an
//! executable with `--higgs-worker`. In a cargo TEST binary `std::env::
//! current_exe()` is the libtest harness, which ignores `--higgs-worker` — so a
//! worker spawned from it dies and every real local load fails. The seam lets the
//! caller name WHICH executable hosts the worker role; the in-process harness
//! (`common::higgs_local`) sets `config.worker_exe = Some(CARGO_BIN_EXE_higgs)`,
//! the real worker-capable `higgs` binary.
//!
//! FAIL-ON-REVERT: revert the `config.worker_exe` threading in
//! `Higgs::with_log_bus` (so it always installs `Arc::new(Supervisor::spawn)` =
//! `current_exe()`) and the worker falls back to THIS libtest binary → it never
//! answers `M_LOAD` → `load` returns `Err` → this test fails at the load `expect`.
//! Restoring the threading makes it pass again. (Verified by hand: reverted the
//! api.rs threading, ran, saw the load fail; restored, saw it pass.)

mod common;

use common::{higgs_local, TINY_MODEL_ID};

#[tokio::test]
async fn worker_exe_seam_runs_real_local_llamacpp() {
    let Some(local) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP worker_exe_seam: tiny gguf not found (set HIGGS_TEST_GGUF)");
        return;
    };

    // Explicit load bypasses the readiness gate and spawns a worker THROUGH the
    // seam. If the seam is reverted, the worker is this libtest binary and this
    // load fails — the crisp fail-on-revert signal.
    local
        .load(TINY_MODEL_ID, None)
        .await
        .expect("real local llama.cpp worker loads the tiny model via the worker_exe seam");

    // A real chat must decode REAL tokens — only a genuine llama.cpp worker can.
    let sampling = higgs::SamplingParams::llamacpp(
        higgs::worker::engine::llamacpp::params::LlamaCppSamplingParams {
            temperature: Some(0.0),
            ..Default::default()
        },
    );
    let messages = r#"[{"role":"user","content":"Say hello."}]"#.to_owned();
    let (mut deltas, handle) = local
        .chat_stream(TINY_MODEL_ID.to_owned(), messages, 16, sampling, None, None)
        .await
        .expect("chat_stream dispatches to the real worker");

    // Drain the delta stream so generation runs to completion.
    let mut streamed = String::new();
    while let Some(delta) = deltas.recv().await {
        streamed.push_str(&delta.text);
    }
    let outcome = handle
        .await
        .expect("chat task joins")
        .expect("chat completes against the real worker");

    assert!(
        outcome.completion_tokens > 0,
        "real worker decoded tokens (completion_tokens > 0): {outcome:?}"
    );
    assert!(
        !outcome.content.is_empty() || !streamed.is_empty(),
        "real worker produced generated text: outcome={outcome:?} streamed={streamed:?}"
    );

    local.shutdown().await;
}
