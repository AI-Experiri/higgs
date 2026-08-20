//! NL-V black-box e2e: `Higgs::set_node_log_level` against a REAL spawned
//! `higgs --node` child over hermetic iroh. Verifies (a) the wire round-trip
//! echoes the daemon's effective verbose state, (b) `M_NODE_LOGS` is no
//! longer honest-empty on a connected idle node — a lifecycle line lands and
//! is prefixed with its `[section]` badge (derived from the tracing target).

mod common;

use std::process::{Child, Command, Stdio};
use std::time::Duration;

use common::{higgs_local, stage_tiny_model, tiny_gguf_path, TINY_MODEL_ID};
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
/// endpoint id + the Proc guard so callers can hold the child alive. When
/// `staged_model_dir` is `Some`, the node child gets `HIGGS_MODEL_DIR` set
/// to that directory (needed to `node_load` a real model on the node); the
/// caller must keep the `TempDir` alive alongside the returned `Proc`.
async fn paired_node(
    higgs: &higgs::Higgs,
    home: &std::path::Path,
    staged_model_dir: Option<&std::path::Path>,
) -> (String, Proc) {
    higgs.hub_enable().await.expect("hub enable");
    let pair = higgs.pair().await.expect("mint pairing credential");
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_higgs"));
    cmd.arg("--node")
        .arg(&pair.ticket)
        .arg(&pair.token)
        .env("HIGGS_HOME", home)
        .env("HIGGS_IROH_LOCAL", "1")
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    if let Some(dir) = staged_model_dir {
        cmd.env("HIGGS_MODEL_DIR", dir);
    }
    let proc = Proc(cmd.spawn().expect("spawn higgs --node"));
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
    let (node_id, _guard) = paired_node(&higgs, home.path(), None).await;

    // Read the default (no-op apply).
    let reply = higgs
        .set_node_log_level(&node_id, NodeLogControlParams::default())
        .await
        .expect("no-op read");
    assert!(reply.verbose, "daemon defaults verbose=true");
    assert!(
        !reply.log_incoming_tokens,
        "daemon defaults log_incoming_tokens=false"
    );
    assert!(
        !reply.log_show_fields,
        "daemon defaults log_show_fields=false"
    );

    // Flip off, confirm.
    let reply = higgs
        .set_node_log_level(
            &node_id,
            NodeLogControlParams {
                verbose: Some(false),
                ..Default::default()
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
                ..Default::default()
            },
        )
        .await
        .expect("verbose on");
    assert!(reply.verbose);

    higgs.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn set_node_log_level_round_trips_incoming_tokens_and_show_fields() {
    // NL-VX: the two remaining serve-layer log toggles must round-trip
    // through the same M_NODE_LOG_LEVEL op — a remote worker gets the
    // same controls as a local one. Fields are independently optional:
    // only the ones present mutate, the rest report their current state.
    let Some(higgs) = higgs_local(&[]).await else {
        eprintln!("skipping set_node_log_level_extras: no tiny GGUF (set HIGGS_TEST_GGUF)");
        return;
    };
    let home = tempfile::tempdir().unwrap();
    let (node_id, _guard) = paired_node(&higgs, home.path(), None).await;

    // Flip both from their default (false → true) in one call; verbose omitted.
    let reply = higgs
        .set_node_log_level(
            &node_id,
            NodeLogControlParams {
                log_incoming_tokens: Some(true),
                log_show_fields: Some(true),
                ..Default::default()
            },
        )
        .await
        .expect("extras on");
    assert!(reply.log_incoming_tokens, "log_incoming_tokens applied");
    assert!(reply.log_show_fields, "log_show_fields applied");
    assert!(
        reply.verbose,
        "unspecified verbose keeps the daemon default (true)"
    );

    // Independence: flip ONLY log_incoming_tokens off; log_show_fields stays on.
    let reply = higgs
        .set_node_log_level(
            &node_id,
            NodeLogControlParams {
                log_incoming_tokens: Some(false),
                ..Default::default()
            },
        )
        .await
        .expect("only incoming_tokens off");
    assert!(!reply.log_incoming_tokens);
    assert!(reply.log_show_fields, "show_fields unchanged when omitted");

    // Restore both to defaults so we don't leak the un-redacted DEBUG mode
    // across tests sharing the process-global LogBus.
    let _ = higgs
        .set_node_log_level(
            &node_id,
            NodeLogControlParams {
                log_incoming_tokens: Some(false),
                log_show_fields: Some(false),
                ..Default::default()
            },
        )
        .await;

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
    let (node_id, _guard) = paired_node(&higgs, home.path(), None).await;

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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn log_incoming_tokens_fires_on_the_iroh_relay_path() {
    // NL-VX end-to-end: the whole point of exposing `log_incoming_tokens` on
    // the per-node op is that flipping it produces observable behavior for a
    // REMOTELY-loaded worker's chats. Before the relay-side fire in
    // `src/node/data.rs::relay_chat`, `serve/v1.rs:370` was the only reader,
    // and it never sees hub→iroh chats — so the toggle was silently a no-op
    // over iroh even though the wire reply reported it applied.
    //
    // Fail-on-revert: comment out the `if bus.log_incoming_tokens() { … }`
    // block in `relay_chat` and this assertion collapses (no matching line
    // ever lands in the node's log ring for the relayed chat).
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("skipping log_incoming_tokens_fires: no tiny GGUF (set HIGGS_TEST_GGUF)");
        return;
    };
    // Hermetic iroh for the parent hub too — the child process gets it via
    // `.env()` in `paired_node`, but the in-process hub needs the same knob
    // so `hub_enable` doesn't dial public relay infra offline. SAFETY: this
    // env write is process-wide, but `higgs_local()` above holds the harness
    // `home_lock` for the whole test lifetime, serializing every in-process
    // harness test in this binary — sibling tests can neither read nor spawn
    // children while this test runs. Restored implicitly on process exit
    // (test binary tears down; no cross-binary leak because each `tests/*.rs`
    // is its own binary).
    unsafe {
        std::env::set_var("HIGGS_IROH_LOCAL", "1");
    }
    let home = tempfile::tempdir().unwrap();
    let gguf = tiny_gguf_path().expect("tiny gguf present (higgs_local returned Some)");
    let node_scan = stage_tiny_model(&gguf);
    let (node_id, _guard) = paired_node(&higgs, home.path(), Some(node_scan.path())).await;

    higgs
        .node_load(&node_id, TINY_MODEL_ID, None)
        .await
        .expect("tiny model loads on the live node");

    // NEGATIVE PATH FIRST — flag OFF (the default, redact-by-default). A chat
    // must NOT put a `higgs: incoming` line in the ring, and its distinctive
    // prompt must not leak. Fail-on-revert for the GUARD: if a future edit
    // dropped the `if bus.log_incoming_tokens() { … }` around the emit
    // (making the line unconditional), this half of the test catches it.
    let reply = higgs
        .set_node_log_level(&node_id, NodeLogControlParams::default())
        .await
        .expect("no-op read");
    assert!(
        !reply.log_incoming_tokens,
        "default state: log_incoming_tokens=false"
    );
    let _report = higgs
        .node_chat_test(&node_id, Some(TINY_MODEL_ID), Some("nope-vx"))
        .await
        .expect("chat over iroh (flag OFF)");
    // Wait long enough for a would-be ring push to land, so a stale "no
    // line yet" snapshot can't false-pass this negative assertion.
    tokio::time::sleep(Duration::from_millis(400)).await;
    let snap_off = higgs.node_logs(&node_id, 400).await.expect("snapshot off");
    assert!(
        !snap_off.iter().any(|l| l.contains("higgs: incoming ")),
        "no incoming-prompt line while flag is OFF: {snap_off:?}"
    );
    assert!(
        !snap_off.iter().any(|l| l.contains("nope-vx")),
        "prompt content never leaks when flag is OFF: {snap_off:?}"
    );

    // POSITIVE PATH — flag ON. The same chat pattern MUST land a line, so
    // the toggle is observable end-to-end for iroh-relayed workers (the
    // whole point of NL-VX). Fail-on-revert for the FIRE: comment out the
    // `if bus.log_incoming_tokens() { … }` block in `relay_chat` and this
    // half fails.
    let reply = higgs
        .set_node_log_level(
            &node_id,
            NodeLogControlParams {
                log_incoming_tokens: Some(true),
                ..Default::default()
            },
        )
        .await
        .expect("flag on");
    assert!(reply.log_incoming_tokens, "flag applied");

    let _report = higgs
        .node_chat_test(&node_id, Some(TINY_MODEL_ID), Some("ping-vx"))
        .await
        .expect("chat over the iroh relay");

    // Poll the node's log ring for the "higgs: incoming" line — the layer
    // buffers, the ring push is on a different task than the emit, so allow
    // several rounds. Ring lines all carry a `[<section>]` badge; this line
    // is emitted from `higgs::node::data`, so the badge is `[node]`.
    let mut snap = Vec::new();
    for _ in 0..40 {
        snap = higgs.node_logs(&node_id, 400).await.expect("snapshot");
        if snap.iter().any(|l| l.contains("higgs: incoming ")) {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    let hit = snap
        .iter()
        .find(|l| l.contains("higgs: incoming "))
        .unwrap_or_else(|| {
            panic!(
                "no `higgs: incoming` line landed for the relayed chat \
                 (log_incoming_tokens fire missing from relay_chat); \
                 last snapshot: {snap:?}"
            )
        });
    // Same wording the /v1 path uses — a Log Terminal reader can't tell
    // routes apart.
    assert!(hit.contains(TINY_MODEL_ID), "line names the model: {hit}");
    assert!(
        hit.contains("ping-vx"),
        "line carries the prompt preview: {hit}"
    );

    higgs.shutdown().await;
}
