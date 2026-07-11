//! T7 end-to-end: `Higgs::node_chat_test` — the Fleet view's per-node "prove the
//! iroh link" action — against a REAL `higgs --node` child over hermetic
//! (relay-disabled) iroh, generating on a REAL tiny GGUF.
//!
//! The unit seam (`api/embed_tests.rs`) drives the method's arms over a fake
//! worker; this test is the actual claim: a test prompt fired at a SPECIFIC node
//! traverses hub → iroh transport → node → spawned worker → llama.cpp and comes
//! back with generated text and token counts. Also pins the [HG074]
//! nothing-served refusal on the LIVE (pre-load) node, where a wrong
//! implementation would surface a confusing chat error instead.
//!
//! SKIPs when no tiny GGUF is present (`HIGGS_TEST_GGUF`, else the on-disk
//! default) — same policy as every fleet e2e test.

mod common;

use std::process::{Child, Command, Stdio};
use std::time::Duration;

use higgs::diagnostic::HiggsError;

use common::{higgs_local, stage_tiny_model, tiny_gguf_path, TINY_MODEL_ID};

/// A spawned `higgs --node`. SIGTERM on drop so its coverage profile flushes.
struct Proc(Child);
impl Drop for Proc {
    fn drop(&mut self) {
        // SAFETY: a plain kill(2) on our own child's pid.
        unsafe { libc::kill(self.0.id() as libc::pid_t, libc::SIGTERM) };
        let _ = self.0.wait();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn node_chat_test_proves_the_iroh_link_end_to_end() {
    // In-process hub Higgs; skips cleanly when no tiny GGUF.
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("skipping node_chat_test: no tiny GGUF (set HIGGS_TEST_GGUF)");
        return;
    };

    higgs.hub_enable().await.expect("hub enable");
    let pair = higgs.pair().await.expect("mint pairing credential");

    // The node gets its OWN model dir with the tiny model staged.
    let gguf = tiny_gguf_path().expect("tiny gguf present (higgs_local returned Some)");
    let node_scan = stage_tiny_model(&gguf);
    let node_home = tempfile::tempdir().unwrap();

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

    // ── Connected node, NOTHING loaded yet → [HG074], the load-first refusal. ──
    let err = higgs
        .node_chat_test(&node_id, None, None)
        .await
        .expect_err("chat test with nothing served on the node fails");
    assert!(
        matches!(err, HiggsError::NodeNothingServed { ref endpoint_id } if *endpoint_id == node_id),
        "nothing served on the node → HG074, got {err:?}"
    );

    // ── Load the tiny model on the node, then the chat test traverses the REAL link. ──
    higgs
        .node_load(&node_id, TINY_MODEL_ID)
        .await
        .expect("tiny model loads on the live node");

    let report = higgs
        .node_chat_test(&node_id, None, None)
        .await
        .expect("node chat test over the live iroh link");
    assert_eq!(report.endpoint_id, node_id, "report names the tested node");
    assert_eq!(
        report.served_id, TINY_MODEL_ID,
        "the node's sole instance is the target"
    );
    assert!(
        !report.content.is_empty() || report.completion_tokens > 0,
        "the remote engine generated something: {report:?}"
    );
    assert!(
        report.completion_tokens > 0,
        "usage came back from the remote engine: {report:?}"
    );
    assert!(report.elapsed_ms > 0, "a real round trip takes time");

    // ── The explicit-instance form pins the SAME worker (served ↔ node match). ──
    let explicit = higgs
        .node_chat_test(&node_id, Some(TINY_MODEL_ID), Some("Reply with: pong"))
        .await
        .expect("explicit-served chat test");
    assert_eq!(explicit.served_id, TINY_MODEL_ID);

    // ── A served id routed on the node but a WRONG requested node → refused, no chat fired. ──
    let bad_node = "0000000000000000000000000000000000000000000000000000000000000000";
    let err = higgs
        .node_chat_test(bad_node, Some(TINY_MODEL_ID), None)
        .await
        .expect_err("mismatched node/served is refused");
    assert!(
        matches!(err, HiggsError::HubControlFailed { .. }),
        "node/served mismatch → refusal, got {err:?}"
    );
}
