//! In-process integration test for the model-load lifecycle event stream.
//!
//! higgs is library-first: load-progress events are pushed over the in-process
//! `Higgs::subscribe_load_events()` broadcast (the same stream `GET /v1`'s SSE
//! surface fans out). This test subscribes BEFORE a real load of the tiny on-disk
//! model, runs the load through the REAL local llama.cpp worker, and asserts the
//! pushed phases arrive in order: `Queued` → `Preparing` → `LoadingWeights` →
//! `Finalizing` → `Ready`. This proves the loading indicator is driven by PUSH
//! events (no polling) end-to-end.
//!
//! Fail-on-revert: if the phase emits are removed from `load_inner` the stream
//! yields no phases and the ordered-sequence assert fails.

mod common;

use common::{higgs_local, TINY_MODEL_ID};
use higgs::api::ModelLoadPhase;

#[tokio::test]
async fn events_stream_pushes_load_phases_in_order() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP events_stream: tiny gguf not found (set HIGGS_TEST_GGUF)");
        return;
    };

    // Subscribe BEFORE the load so no phase is missed (there is no replay). The
    // broadcast buffer holds every phase of a single load, so a drain after the
    // load returns the full ordered sequence.
    let mut rx = higgs.subscribe_load_events();

    // Trigger a REAL load through the local llama.cpp worker.
    higgs
        .load(TINY_MODEL_ID, None)
        .await
        .expect("load succeeds");

    // Drain the pushed phases.
    let mut phases = Vec::new();
    while let Ok(ev) = rx.try_recv() {
        assert_eq!(ev.id, TINY_MODEL_ID, "event carries the loaded model id");
        phases.push(ev.phase);
    }

    // Every real progress phase must appear, terminated by `Ready`.
    for want in [
        ModelLoadPhase::Queued,
        ModelLoadPhase::Preparing,
        ModelLoadPhase::LoadingWeights,
        ModelLoadPhase::Finalizing,
        ModelLoadPhase::Ready,
    ] {
        assert!(
            phases.contains(&want),
            "missing phase {want:?} in {phases:?}"
        );
    }

    // …and in order (subsequence check: each appears after the previous).
    let order = [
        ModelLoadPhase::Queued,
        ModelLoadPhase::Preparing,
        ModelLoadPhase::LoadingWeights,
        ModelLoadPhase::Finalizing,
        ModelLoadPhase::Ready,
    ];
    let mut idx = 0usize;
    for p in &phases {
        if idx < order.len() && *p == order[idx] {
            idx += 1;
        }
    }
    assert_eq!(idx, order.len(), "phases not in expected order: {phases:?}");

    higgs.shutdown().await;
}
