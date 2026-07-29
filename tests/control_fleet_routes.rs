//! Hub control ops that need a LIVE fleet — the ERROR/edge arms of the node-mutation facade
//! methods (`Higgs::node_load` / `node_scan` / `node_unload` / `node_retire` / `node_label`)
//! that only fire when a hub IS enabled and a node IS paired (so the fleet/hub is present and
//! the op itself reaches the remote link).
//!
//! `remote_hub_e2e.rs` drives the HAPPY paths of the fleet seam (pair → load → chat → retire)
//! end-to-end. This file deliberately targets the `Err(...)` arms those don't reach: an unknown
//! node → HG027 `NodeUnreachable`; an unknown served id → HG002 `ModelNotFound`; an unknown model
//! on a REAL node → a relayed load failure; plus the unknown-node retire/relabel flows against a
//! real (single-node) fleet.
//!
//! higgs is now library-first: control is the in-process `Higgs` crate API, not the deleted
//! `/api/higgs/*` HTTP surface. So instead of spawning a hub `higgs` server + minting a pairing
//! token over `POST /api/higgs/pair`, we run the hub IN-PROCESS (`Higgs::hub_enable`), mint the
//! token over `Higgs::pair`, dial it with a real `higgs --node` child, and wait until the remote
//! node shows connected in `Higgs::nodes()`. Then exercise the error arms on the facade. Pairing
//! needs no GGUF for the node to CONNECT, but this harness stages the tiny model so the node can
//! also load it (the happy fleet-load assertion). The test SKIPs when no tiny GGUF is present.

mod common;

use std::process::{Child, Command, Stdio};
use std::time::Duration;

use higgs::diagnostic::HiggsError;

use common::{higgs_local, serve_v1_local, stage_tiny_model, tiny_gguf_path, TINY_MODEL_ID};

/// A spawned `higgs --node`. SIGTERM on drop so its coverage profile flushes.
struct Proc(Child);
impl Drop for Proc {
    fn drop(&mut self) {
        // SAFETY: a plain kill(2) on our own child's pid.
        unsafe { libc::kill(self.0.id() as libc::pid_t, libc::SIGTERM) };
        let _ = self.0.wait();
    }
}

/// With a LIVE in-process hub + a connected node, the node-mutation facade methods reach the
/// fleet/hub ops — so their `Err(...)` arms fire on bad arguments:
///
/// - `node_load` with an UNKNOWN node id → the fleet's `transport()` can't find a live link →
///   HG027 `NodeUnreachable`.
/// - `node_scan` of an UNKNOWN node → same HG027.
/// - `node_unload` of an UNKNOWN served id → `require_served` → HG002 `ModelNotFound`.
/// - `node_load` on the REAL node but an UNKNOWN model → the node relays a load failure back
///   through the fleet → `Err`.
/// - `node_label` / `node_retire` of an UNKNOWN node against the LIVE hub → relabel is the
///   `Ok(false)` unknown-node arm; retire is an idempotent no-op `Ok(())`.
///
/// Then the happy fleet-load lands (proving the error arms didn't wedge the link), the model is
/// routable over the real `/v1/models` surface, and retiring the real node drops it from the fleet.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn hub_node_mutation_error_arms_fire_with_a_live_fleet() {
    // In-process hub Higgs with the tiny model staged; skips cleanly when no tiny GGUF.
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("skipping control_fleet_routes: no tiny GGUF (set HIGGS_TEST_GGUF)");
        return;
    };

    // Turn the hub network ON in-process (installs a fresh HubFleet + accept loop), then mint a
    // one-time pairing credential — exactly what `POST /api/higgs/hub/enable` + `/pair` used to do.
    higgs.hub_enable().await.expect("hub enable");
    let pair = higgs.pair().await.expect("mint pairing credential");

    // Stage the tiny model into the node's OWN read-only model dir so the happy "load a real model
    // on the node" assertion can run. `tiny_gguf_path()` is Some (higgs_local returned Some).
    let gguf = tiny_gguf_path().expect("tiny gguf present (higgs_local returned Some)");
    let node_scan = stage_tiny_model(&gguf);
    let node_home = tempfile::tempdir().unwrap();

    // Spawn a real `higgs --node <ticket> <token>` dialing the in-process hub over hermetic iroh.
    let _node = Proc(
        Command::new(env!("CARGO_BIN_EXE_higgs"))
            .arg("--node")
            .arg(&pair.ticket)
            .arg(&pair.token)
            .env("HIGGS_HOME", node_home.path())
            .env("HIGGS_MODEL_DIR", node_scan.path())
            .env("HIGGS_IROH_LOCAL", "1")
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("spawn higgs --node"),
    );

    // Wait until the remote node shows connected in the unified fleet view.
    let mut node_id = String::new();
    for _ in 0..150 {
        if let Some(n) = higgs
            .nodes()
            .await
            .into_iter()
            .find(|n| n.connected && !n.is_local)
        {
            node_id = n.endpoint_id.clone();
            break;
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    assert!(!node_id.is_empty(), "remote node paired + connected");

    let bad_node = "0000000000000000000000000000000000000000000000000000000000000000";

    // ── load on an UNKNOWN node → HG027 NodeUnreachable (fleet present, so past the not-a-hub guard). ──
    let err = higgs
        .node_load(bad_node, TINY_MODEL_ID, None)
        .await
        .expect_err("load on an unknown node fails");
    assert!(
        matches!(err, HiggsError::NodeUnreachable { .. }),
        "load on an unknown node → HG027 NodeUnreachable, got {err:?}"
    );

    // ── scan an UNKNOWN node → HG027 NodeUnreachable (same unreachable path). ──
    let err = higgs
        .node_scan(bad_node)
        .await
        .expect_err("scanning an unknown node fails");
    assert!(
        matches!(err, HiggsError::NodeUnreachable { .. }),
        "scanning an unknown node → HG027 NodeUnreachable, got {err:?}"
    );

    // ── unload an UNKNOWN served id → HG002 ModelNotFound (require_served fails). ──
    let err = higgs
        .node_unload("no-such/served-id")
        .await
        .expect_err("unloading an unknown served id fails");
    assert!(
        matches!(err, HiggsError::ModelNotFound { .. }),
        "unloading an unknown served id → HG002 ModelNotFound, got {err:?}"
    );

    // ── load an UNKNOWN model on the REAL connected node → the node relays a load failure back
    // through the fleet; the hub surfaces it as an `Err` (not a hang, not an Ok). ──
    assert!(
        higgs
            .node_load(&node_id, "definitely/not-a-real-model", None)
            .await
            .is_err(),
        "loading an unknown model on a real node fails (never Ok)"
    );

    // ── relabel an UNKNOWN remote node against the LIVE hub → the `Ok(false)` unknown-node arm
    // (the serve layer used to map this to a 404). ──
    let renamed = higgs
        .node_label(bad_node, "ghost")
        .await
        .expect("relabel call itself succeeds (hub enabled)");
    assert!(
        !renamed,
        "relabel of an unknown node (hub enabled) → Ok(false) unknown-node arm"
    );

    // ── The REAL node loads a model (happy fleet-load), proving the error arms above didn't wedge
    // the live link. Returns the new remote worker's id. ──
    let worker = higgs
        .node_load(&node_id, TINY_MODEL_ID, None)
        .await
        .expect("real model loads on the live node after the error arms");
    assert!(
        worker.0 >= 1,
        "load reply carries a worker id: {}",
        worker.0
    );

    // ── The freshly-loaded remote model is now routable in the OpenAI catalog over the REAL /v1
    // HTTP surface (the old `GET /v1/models` assertion). ──
    let (base, serve) = serve_v1_local(higgs.handle()).await;
    let models: serde_json::Value = reqwest::Client::new()
        .get(format!("{base}/v1/models"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(
        models["data"]
            .as_array()
            .is_some_and(|d| d.iter().any(|m| m["id"] == TINY_MODEL_ID)),
        "the remote-loaded model is routable in /v1/models: {models}"
    );
    serve.shutdown().await;

    // ── retire the UNKNOWN node against the live hub: the allowlist-removal is a no-op for an id it
    // never admitted, but retire is idempotent → `Ok(())`. The real node is untouched + still listed. ──
    higgs
        .node_retire(bad_node)
        .await
        .expect("retiring an unknown node is an idempotent no-op Ok(())");
    assert!(
        higgs.nodes().await.iter().any(|n| n.endpoint_id == node_id),
        "the real node is still listed after the unknown-node retire"
    );

    // ── retire the REAL node → it leaves the fleet entirely (drops from the node view). ──
    higgs
        .node_retire(&node_id)
        .await
        .expect("retire of the real node ok");
    let mut gone = false;
    for _ in 0..50 {
        if !higgs.nodes().await.iter().any(|n| n.endpoint_id == node_id) {
            gone = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert!(gone, "the retired real node is removed from the fleet");

    higgs.shutdown().await;
}
