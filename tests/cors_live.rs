//! Integration test for G7 live CORS: a `set_cors_origins` write applies to a
//! RUNNING `/v1` listener without any rebind or restart — the layer reads the
//! facade's live allowlist per request.

mod common;

use common::{higgs_local, serve_v1_local, TINY_MODEL_ID};

/// An origin absent from the allowlist is refused CORS; after a live
/// `set_cors_origins` write — same listener, no re-serve — the SAME preflight
/// is allowed, and the settings disclose applied == persisted with no restart
/// pending.
///
/// Fail-on-revert: without `set_cors_origins` publishing to the live list
/// (or with the layer back on a build-time snapshot), the second preflight
/// still lacks the allow-origin header and this test fails.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cors_change_applies_to_a_running_listener() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!(
            "SKIP cors_change_applies_to_a_running_listener: tiny gguf not found \
             (set HIGGS_TEST_GGUF)"
        );
        return;
    };
    let (base, guard) = serve_v1_local(higgs.handle()).await;
    let client = reqwest::Client::new();
    let origin = "https://tools.example";

    let preflight = |client: reqwest::Client, base: String| async move {
        client
            .request(reqwest::Method::OPTIONS, format!("{base}/v1/models"))
            .header("origin", origin)
            .header("access-control-request-method", "GET")
            .send()
            .await
            .expect("preflight sends")
    };

    // Not allowlisted: the preflight response carries NO allow-origin header.
    let before = preflight(client.clone(), base.clone()).await;
    assert!(
        before
            .headers()
            .get("access-control-allow-origin")
            .is_none(),
        "an un-allowlisted origin must not be allowed before the write"
    );

    // Live write — no rebind, no re-serve.
    let settings = higgs
        .handle()
        .set_cors_origins(vec![origin.to_string()])
        .expect("valid origin persists");
    assert_eq!(settings.applied_origins, vec![origin.to_string()]);
    assert!(
        !settings.restart_required,
        "the write applied live — nothing pending"
    );

    // The SAME listener now allows the origin.
    let after = preflight(client.clone(), base.clone()).await;
    assert_eq!(
        after
            .headers()
            .get("access-control-allow-origin")
            .and_then(|v| v.to_str().ok()),
        Some(origin),
        "the live write must be enforced by the running listener"
    );

    guard.shutdown().await;
}
