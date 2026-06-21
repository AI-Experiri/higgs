//! Black-box coverage of the lean pairing CLI (`higgs link …`, `higgs node …`,
//! `higgs --node`) by invoking the REAL binary as a subprocess with a temp `HIGGS_HOME`.
//!
//! These are the fast, model-less control paths in `node/cli.rs`: identity printing, status,
//! a one-shot node→hub dial (`node connect`) against an in-process test hub, and the
//! argument/parse error branches. No GGUF needed, so they always run.

use std::process::Command;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use iroh_tickets::endpoint::EndpointTicket;

use higgs::auth::{Allowlist, PairingTokens};
use higgs::node::{dial_and_hello, gate_connection, GateOutcome, HELLO_DEADLINE};
use higgs::remote::ALPN;

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// Run `higgs <args>` with an isolated HIGGS_HOME (+ hermetic iroh) and capture output.
fn run_higgs(home: &std::path::Path, args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_higgs"))
        .args(args)
        .env("HIGGS_HOME", home)
        .env("HIGGS_IROH_LOCAL", "1")
        .output()
        .expect("spawn higgs")
}

#[test]
fn link_status_prints_identity_and_zero_paired() {
    let home = tempfile::tempdir().unwrap();
    let out = run_higgs(home.path(), &["link", "status"]);
    assert!(out.status.success(), "link status exits 0");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("higgs hub id"), "prints hub id: {stdout}");
    assert!(
        stdout.contains("paired nodes : 0"),
        "no nodes paired yet: {stdout}"
    );
}

#[test]
fn node_daemon_without_ticket_prints_identity_and_usage() {
    let home = tempfile::tempdir().unwrap();
    let out = run_higgs(home.path(), &["--node"]);
    assert!(out.status.success(), "--node with no ticket exits 0");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("higgs node id"), "prints node id: {stdout}");
    assert!(stdout.contains("usage:"), "prints usage hint: {stdout}");
}

#[test]
fn keys_add_list_remove_via_cli() {
    let home = tempfile::tempdir().unwrap();
    // add a key → succeeds and prints a token once.
    let add = run_higgs(home.path(), &["keys", "add", "ci", "chat,models"]);
    assert!(add.status.success(), "keys add exits 0");
    let out = String::from_utf8_lossy(&add.stdout);
    assert!(
        out.contains("token (shown ONCE"),
        "prints the minted token: {out}"
    );
    assert!(out.contains("hgk_"), "token has the hgk_ prefix");

    // list shows it.
    let list = run_higgs(home.path(), &["keys", "list"]);
    assert!(list.status.success());
    assert!(
        String::from_utf8_lossy(&list.stdout).contains("ci"),
        "list shows the key"
    );

    // remove drops it; a bad scope / missing label / unknown subcommand all fail.
    assert!(run_higgs(home.path(), &["keys", "remove", "ci"])
        .status
        .success());
    assert!(!run_higgs(home.path(), &["keys", "add", "x", "bogus"])
        .status
        .success());
    assert!(!run_higgs(home.path(), &["keys", "add"]).status.success());
    assert!(!run_higgs(home.path(), &["keys", "bogus"]).status.success());
}

#[test]
fn cli_arg_errors_are_nonzero() {
    let home = tempfile::tempdir().unwrap();
    // Unknown subcommands + a malformed ticket all exit non-zero (no panic).
    for args in [
        vec!["link", "bogus"],
        vec!["node", "bogus"],
        vec!["node", "connect"],
        vec!["node", "connect", "not-a-ticket"],
    ] {
        let out = run_higgs(home.path(), &args);
        assert!(!out.status.success(), "`{args:?}` should fail");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn node_connect_dials_and_pairs_with_a_hub() {
    let home = tempfile::tempdir().unwrap();

    // In-process test hub: bind a hermetic endpoint, mint a token, build a ticket.
    let hub = iroh::Endpoint::builder(iroh::endpoint::presets::Minimal)
        .alpns(vec![ALPN.to_vec()])
        .relay_mode(iroh::RelayMode::Disabled)
        .bind()
        .await
        .expect("bind hub");
    let hub_id = hub.id().to_string();
    let ticket = EndpointTicket::new(hub.addr()).to_string();
    let mut allow = Allowlist::load(&home.path().join("hub-pairings.json")).unwrap();
    let mut tokens = PairingTokens::new();
    let token = tokens.mint(now_ms(), 600_000);

    // Accept + gate the node's dial concurrently with spawning the CLI.
    let gate = tokio::spawn(async move {
        let incoming = tokio::time::timeout(Duration::from_secs(30), hub.accept())
            .await
            .expect("dial within 30s")
            .expect("incoming");
        let conn = incoming.await.expect("connection");
        let outcome = gate_connection(
            &conn,
            &mut allow,
            &mut tokens,
            now_ms(),
            hub_id,
            Some("cli".into()),
            HELLO_DEADLINE,
        )
        .await;
        // Keep the connection alive briefly so the node reads the HELLO result.
        let _ = tokio::time::timeout(Duration::from_secs(5), conn.closed()).await;
        outcome
    });

    // `higgs node connect <ticket> <token>` in a blocking thread (it runs its own runtime).
    let node_home = home.path().to_path_buf();
    let out = tokio::task::spawn_blocking(move || {
        run_higgs(&node_home, &["node", "connect", &ticket, &token])
    })
    .await
    .unwrap();

    let outcome = gate.await.unwrap();
    assert!(
        matches!(outcome, GateOutcome::Admitted { .. }),
        "hub admitted the node: {outcome:?}"
    );
    assert!(
        out.status.success(),
        "node connect exits 0: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("paired with hub"),
        "node reports pairing: {stdout}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn node_connect_with_bad_token_is_rejected() {
    let home = tempfile::tempdir().unwrap();

    let hub = iroh::Endpoint::builder(iroh::endpoint::presets::Minimal)
        .alpns(vec![ALPN.to_vec()])
        .relay_mode(iroh::RelayMode::Disabled)
        .bind()
        .await
        .expect("bind hub");
    let hub_id = hub.id().to_string();
    let ticket = EndpointTicket::new(hub.addr()).to_string();
    // A hub with NO minted tokens and an empty allowlist → any dial with a token is HG022.
    let mut allow = Allowlist::load(&home.path().join("hub-pairings.json")).unwrap();
    let mut tokens = PairingTokens::new();

    let gate = tokio::spawn(async move {
        let incoming = tokio::time::timeout(Duration::from_secs(30), hub.accept())
            .await
            .expect("dial within 30s")
            .expect("incoming");
        let conn = incoming.await.expect("connection");
        gate_connection(
            &conn,
            &mut allow,
            &mut tokens,
            now_ms(),
            hub_id,
            Some("cli".into()),
            HELLO_DEADLINE,
        )
        .await
    });

    let node_home = home.path().to_path_buf();
    let out = tokio::task::spawn_blocking(move || {
        run_higgs(&node_home, &["node", "connect", &ticket, "htk_bogustoken"])
    })
    .await
    .unwrap();

    let outcome = gate.await.unwrap();
    assert!(
        matches!(outcome, GateOutcome::Rejected { .. }),
        "hub rejects a bad token: {outcome:?}"
    );
    assert!(
        !out.status.success(),
        "node connect with a bad token exits non-zero"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn link_pair_accepts_an_in_process_node_dial() {
    use tokio::io::{AsyncBufReadExt, BufReader};

    let home = tempfile::tempdir().unwrap();
    // Spawn the REAL hub-side `higgs link pair` accept loop; capture its stdout so we can
    // read the printed ticket + token and then dial it from an in-process node.
    let mut child = tokio::process::Command::new(env!("CARGO_BIN_EXE_higgs"))
        .args(["link", "pair"])
        .env("HIGGS_HOME", home.path())
        .env("HIGGS_IROH_LOCAL", "1")
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("spawn higgs link pair");

    let mut lines = BufReader::new(child.stdout.take().unwrap()).lines();
    // Read until both the ticket and token lines have been seen (link pair waits up to ~10s
    // for a relay before printing the ticket in hermetic mode).
    let (mut ticket, mut token) = (None, None);
    let read = tokio::time::timeout(Duration::from_secs(30), async {
        while let Ok(Some(line)) = lines.next_line().await {
            if let Some(rest) = line.strip_prefix("pairing token: ") {
                token = rest.split_whitespace().next().map(str::to_string);
            } else if let Some(rest) = line.strip_prefix("ticket       : ") {
                ticket = Some(rest.trim().to_string());
            }
            if ticket.is_some() && token.is_some() {
                break;
            }
        }
    })
    .await;
    assert!(
        read.is_ok(),
        "link pair printed its ticket+token within 30s"
    );
    let ticket: EndpointTicket = ticket.expect("ticket line").parse().expect("valid ticket");
    let token = token.expect("token line");

    // In-process node: dial the hub-pair process and complete HELLO with the token.
    let node = iroh::Endpoint::builder(iroh::endpoint::presets::Minimal)
        .alpns(vec![ALPN.to_vec()])
        .relay_mode(iroh::RelayMode::Disabled)
        .bind()
        .await
        .expect("bind node");
    let self_id = node.id().to_string();
    let result = dial_and_hello(&node, ticket.endpoint_addr().clone(), self_id, Some(token))
        .await
        .expect("node pairs with the link-pair hub");
    assert_eq!(result.role, "hub", "hub answered the HELLO");

    // SIGTERM (not SIGKILL): the pair listener exits its accept loop cleanly, which also
    // flushes its coverage profile under llvm-cov instrumentation.
    unsafe {
        libc::kill(child.id().expect("child pid") as libc::pid_t, libc::SIGTERM);
    }
    let _ = child.wait().await;
}
