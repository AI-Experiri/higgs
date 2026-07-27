//! Integration tests for the G7 listener-rebind facade reservation
//! (`Higgs::reserve_rebind` / `RebindReservation`, `[HG073]`).
//!
//! A rebind is drain-old-then-bind-new — same-port rebinds cannot overlap
//! (binding a wildcard while the specific address is held is EADDRINUSE) —
//! so there is necessarily a moment with ZERO live listeners. Without a
//! reservation, that moment is indistinguishable from "the last listener
//! exited for good" and `serve_v1`'s exit path runs the TERMINAL facade
//! `stop()`, killing the facade the successor listener is about to serve.

mod common;

use common::{higgs_local, serve_v1_local, TINY_MODEL_ID};

/// The reservation keeps the facade alive through the zero-listener moment:
/// after the ONLY listener drains under a live reservation, the facade still
/// grants a further reservation AND serves a successor listener.
///
/// Fail-on-revert: without the reservation gate in `deregister_serve`, the
/// old listener's exit runs the terminal `stop()` — the post-drain
/// `reserve_rebind` then returns `[HG073]` and this test fails.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reservation_keeps_the_facade_alive_across_a_drain() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!(
            "SKIP reservation_keeps_the_facade_alive_across_a_drain: tiny gguf not found \
             (set HIGGS_TEST_GGUF)"
        );
        return;
    };
    let (base1, guard1) = serve_v1_local(higgs.handle()).await;

    let reservation = higgs
        .handle()
        .reserve_rebind()
        .await
        .expect("reserve on a live facade");
    // Drain the ONLY listener — the reservation stands in for the successor.
    guard1.shutdown().await;

    // The facade survived: a stopped facade would refuse this with [HG073].
    let probe = higgs
        .handle()
        .reserve_rebind()
        .await
        .expect("facade must still be alive after a reserved drain");
    drop(probe);

    // …and a successor listener genuinely serves on the still-live facade.
    let (base2, guard2) = serve_v1_local(higgs.handle()).await;
    assert_ne!(base1, base2, "ephemeral successor binds a fresh port");
    let resp = reqwest::get(format!("{base2}/v1/models")).await.unwrap();
    assert!(
        resp.status().is_success(),
        "successor listener must serve the surviving facade, got {}",
        resp.status()
    );

    guard2.shutdown().await;
    drop(reservation);
}

/// A facade whose terminal `stop()` already ran refuses new rebind
/// reservations with `[HG073]` — the caller learns before touching any
/// listener.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reserve_after_stop_returns_hg073() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!(
            "SKIP reserve_after_stop_returns_hg073: tiny gguf not found (set HIGGS_TEST_GGUF)"
        );
        return;
    };

    higgs.handle().stop().await;

    let err = higgs
        .handle()
        .reserve_rebind()
        .await
        .expect_err("a stopped facade must refuse a rebind reservation");
    assert!(
        err.to_string().starts_with("[HG073]"),
        "expected [HG073], got: {err}"
    );
}
