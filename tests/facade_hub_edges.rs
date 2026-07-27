//! Integration coverage for the hub-gated facade paths in `src/api/embed.rs` that need a LIVE hub
//! (so the not-a-hub gate is PASSED and control reaches the courier / fleet). Toy-model-safe: the
//! courier fetches from a closed loopback port (fast connection-refused), never a real release.
//!
//! Covers: `node_update`/`fleet_update` past the gate into `release_courier`, `node_chat_test`'s
//! local-sentinel + unknown-node refusals, and `hub_disable`'s drain path.

mod common;

use common::{higgs_local, TINY_MODEL_ID};
use higgs::HiggsError;

/// Enable the local (offline) hub network for an in-process facade test.
fn enable_local_hub_env() {
    // SAFETY: `higgs_local` holds the process-global home lock for the test's lifetime, serializing
    // this with every other harness test; no other in-process test enables a hub.
    unsafe {
        std::env::set_var("HIGGS_IROH_LOCAL", "1");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn hub_gated_facade_paths_reach_the_courier_and_drain() {
    let Some(h) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP hub_gated_facade_paths: tiny gguf not found (set HIGGS_TEST_GGUF)");
        return;
    };
    enable_local_hub_env();
    h.hub_enable().await.expect("enable the local hub network");

    // node_update PAST the gate → the courier fetches from a CLOSED loopback port → a fetch error
    // (NOT a not-a-hub / hub-control error). This is the past-the-gate courier-reach path.
    let e = h
        .node_update("ghost-node", "http://127.0.0.1:9/d/x.manifest")
        .await
        .expect_err("an unreachable manifest URL fails at the courier fetch");
    assert!(
        matches!(
            e,
            HiggsError::UpdateFetchFailed { .. } | HiggsError::UpdateManifestInvalid { .. }
        ),
        "node_update reaches the courier, not the gate: {e:?}"
    );

    // fleet_update PAST the gate → no nodes are connected, so the courier returns an EMPTY report
    // (a systemic-error-free run), never a not-a-hub error.
    let report = h
        .fleet_update("http://127.0.0.1:9/releases/download/v1.2.3")
        .await
        .expect("fleet_update returns a report even with zero connected nodes");
    assert_eq!(
        report["results"].as_array().map(Vec::len),
        Some(0),
        "no connected nodes → empty results: {report}"
    );

    // node_chat_test: the LOCAL sentinel is refused (rides the fleet view, has no iroh hop to prove).
    let e = h
        .node_chat_test("local", None, None)
        .await
        .expect_err("the local node has no remote hop to test");
    assert!(
        matches!(e, HiggsError::InvalidChatTestTarget { .. }),
        "{e:?}"
    );

    // node_chat_test: an unknown endpoint id (hub enabled, node not in the fleet) → UnknownNode.
    let e = h
        .node_chat_test("some-unknown-endpoint-id", None, None)
        .await
        .expect_err("an unknown node is refused");
    assert!(matches!(e, HiggsError::UnknownNode { .. }), "{e:?}");

    // hub_disable drains + closes the fleet (zero nodes) and reports the disabled state.
    let status = h.hub_disable().await;
    let sj = serde_json::to_value(&status).expect("serialize hub status");
    assert_eq!(sj["enabled"], false, "hub_disable reports disabled: {sj}");
}
