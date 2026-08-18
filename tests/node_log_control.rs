//! NL-V black-box e2e: `Higgs::set_node_log_level` against a REAL spawned
//! `higgs --node` child over hermetic iroh. Verifies (a) the wire round-trip
//! echoes the daemon's effective verbose state, (b) `M_NODE_LOGS` is no
//! longer honest-empty on a connected idle node — a lifecycle line lands and
//! is prefixed with its `[section]` badge (derived from the tracing target).

mod common;

use std::process::{Child, Command, Stdio};
use std::time::Duration;

use common::higgs_local;
use higgs::remote::NodeLogControlParams;

struct Proc(Child);
impl Drop for Proc {
    fn drop(&mut self) {
        // SAFETY: plain kill(2) on our own child's pid. SIGTERM lets the
        // node flush its coverage profraw before exit.
        unsafe { libc::kill(self.0.id() as libc::pid_t, libc::SIGTERM) };
        let _ = self.0.wait();
    }
}

/// Spawn a node paired with `higgs`, wait until it's connected, return its
/// endpoint id + the Proc guard so callers can hold the child alive.
async fn paired_node(higgs: &higgs::Higgs, home: &std::path::Path) -> (String, Proc) {
    higgs.hub_enable().await.expect("hub enable");
    let pair = higgs.pair().await.expect("mint pairing credential");
    let proc = Proc(
        Command::new(env!("CARGO_BIN_EXE_higgs"))
            .arg("--node")
            .arg(&pair.ticket)
            .arg(&pair.token)
            .env("HIGGS_HOME", home)
            .env("HIGGS_IROH_LOCAL", "1")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn higgs --node"),
    );
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
    (node_id, proc)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn set_node_log_level_round_trips_verbose_state() {
    // Wire contract: `verbose: None` reads back the current daemon state,
    // `Some(v)` overrides. The daemon defaults verbose=true (NL-V design:
    // always verbose so an incident already has DEBUG in the ring).
    let Some(higgs) = higgs_local(&[]).await else {
        eprintln!("skipping set_node_log_level: no tiny GGUF (set HIGGS_TEST_GGUF)");
        return;
    };
    let home = tempfile::tempdir().unwrap();
    let (node_id, _guard) = paired_node(&higgs, home.path()).await;

    // Read the default (no-op apply).
    let reply = higgs
        .set_node_log_level(&node_id, NodeLogControlParams { verbose: None })
        .await
        .expect("no-op read");
    assert!(reply.verbose, "daemon defaults verbose=true");

    // Flip off, confirm.
    let reply = higgs
        .set_node_log_level(
            &node_id,
            NodeLogControlParams {
                verbose: Some(false),
            },
        )
        .await
        .expect("verbose off");
    assert!(!reply.verbose);

    // Flip back on, confirm.
    let reply = higgs
        .set_node_log_level(
            &node_id,
            NodeLogControlParams {
                verbose: Some(true),
            },
        )
        .await
        .expect("verbose on");
    assert!(reply.verbose);

    higgs.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_connected_idle_node_emits_lifecycle_lines_with_section_badges() {
    // Two guarantees at once:
    //   (a) `M_NODE_LOGS` is no longer honest-empty — the daemon now
    //       emits tracing::info! on happy paths, so a subscriber sees
    //       something on a connected idle node.
    //   (b) each line carries a `[<section>]` badge derived from the
    //       tracing target — `higgs::node` → `[node]`,
    //       `higgs::worker` → `[worker]`, etc.
    let Some(higgs) = higgs_local(&[]).await else {
        eprintln!("skipping lifecycle_badge: no tiny GGUF (set HIGGS_TEST_GGUF)");
        return;
    };
    let home = tempfile::tempdir().unwrap();
    let (node_id, _guard) = paired_node(&higgs, home.path()).await;

    // Give the daemon a beat to write the connect line into the ring.
    let mut snap = Vec::new();
    for _ in 0..40 {
        snap = higgs.node_logs(&node_id, 100).await.expect("snapshot");
        if snap
            .iter()
            .any(|l| l.contains("connected to hub") || l.contains("node daemon starting"))
        {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert!(
        !snap.is_empty(),
        "the idle node now has SOMETHING to say (was empty pre-NL-V)"
    );
    // Every line must start with a `[<section>]` badge — the layer
    // prepends it inline from the tracing target's second segment.
    assert!(
        snap.iter().all(|l| l.starts_with('[') && l.contains("] ")),
        "every line carries a [<section>] badge: {snap:?}"
    );
    // The lifecycle lines target `higgs::node` → badge `[node]`.
    assert!(
        snap.iter().any(|l| l.starts_with("[node] ")),
        "at least one `[node]` line from the daemon lifecycle: {snap:?}"
    );

    higgs.shutdown().await;
}
