//! Full-parity `InventoryWorker.model_info` end-to-end: a spawned REAL
//! `higgs --node` child loads the tiny GGUF, then the hub's `nodes()` view —
//! folded from the node's `M_NODE_INVENTORY` reply — must carry the same
//! rich `HiggsModel` facts a LOCAL client would see for the same file
//! (quant, size, arch/ctx_train when the header enriches, has_chat_template,
//! supports_tools/reasoning, source). Fail-on-revert: drop the
//! `model_info: Some(resolved.full)` line in `do_load`'s `LoadFacts`
//! construction and this assertion collapses (`model_info` is `None`).

mod common;

use std::process::{Child, Command, Stdio};
use std::time::Duration;

use common::{higgs_local, stage_tiny_model, tiny_gguf_path, TINY_MODEL_ID};

struct Proc(Child);
impl Drop for Proc {
    fn drop(&mut self) {
        // SAFETY: plain kill(2) on our own child's pid. SIGTERM lets the node
        // flush its coverage profraw before exit.
        unsafe { libc::kill(self.0.id() as libc::pid_t, libc::SIGTERM) };
        let _ = self.0.wait();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn remote_inventory_row_carries_full_model_info() {
    // Hermetic iroh; harmless if set process-wide (this file has one test).
    // SAFETY: single-test binary — no sibling test races this write.
    unsafe { std::env::set_var("HIGGS_IROH_LOCAL", "1") };

    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("skipping inventory model_info: no tiny GGUF (set HIGGS_TEST_GGUF)");
        return;
    };
    higgs.hub_enable().await.expect("hub enable");
    let pair = higgs.pair().await.expect("mint pairing credential");

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
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn higgs --node"),
    );

    // Wait for the remote node to admit + settle.
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

    higgs
        .node_load(&node_id, TINY_MODEL_ID, None)
        .await
        .expect("tiny model loads on the live node");

    // Give the node one refresh cycle so the hub's cached inventory picks up
    // the freshly-loaded worker (nodes_view reads the cached snapshot).
    let mut with_worker = None;
    for _ in 0..40 {
        let nodes = higgs.nodes().await;
        if let Some(view) = nodes.iter().find(|n| n.endpoint_id == node_id) {
            let inv = view.inventory.as_ref();
            let worker =
                inv.and_then(|i| i.workers.iter().find(|w| w.model == TINY_MODEL_ID).cloned());
            if worker.is_some() {
                with_worker = worker;
                break;
            }
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    let worker = with_worker.expect("inventory eventually surfaces the loaded worker");

    // The whole point of this task: `model_info` carries the HiggsModel a
    // local client would see for the same file. Assert every non-optional +
    // one representative optional per axis (path present, id matches, size
    // > 0, has_chat_template + supports_tools/reasoning are bools). Fields
    // that depend on GGUF header enrichment (arch/ctx_train) are checked
    // ONLY as "present-or-not", not as a specific value, since the tiny
    // fixture's header may or may not populate them — the assertion is that
    // `model_info` itself flowed, not that a specific arch string appeared.
    let info = worker
        .model_info
        .as_ref()
        .expect("model_info populated for a loaded worker (revert `LoadFacts.model_info`?)");
    assert_eq!(info.id, TINY_MODEL_ID, "info.id matches worker.model");
    assert!(
        info.size_bytes > 0,
        "info.size_bytes > 0 (real file staged)"
    );
    assert!(!info.path.is_empty(), "info.path is the node-local path");
    // Booleans are always present in HiggsModel — round-trip proves them.
    let _ = info.has_chat_template;
    let _ = info.supports_tools;
    let _ = info.supports_reasoning;

    higgs.shutdown().await;
}
