//! Black-box coverage of the lean pairing CLI (`higgs link …`, `higgs node …`,
//! `higgs --node`) by invoking the REAL binary as a subprocess with a temp `HIGGS_HOME`.
//!
//! These are the fast, model-less control paths in `node/cli.rs`: identity printing, status,
//! a one-shot node→hub dial (`node connect`) against an in-process test hub, and the
//! argument/parse error branches. No GGUF needed, so they always run.

use std::path::Path;
use std::process::Command;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use iroh_tickets::endpoint::EndpointTicket;
use tokio::sync::Mutex;

use higgs::auth::{Allowlist, PairingTokens};
use higgs::config::InstanceConfig;
use higgs::node::{
    connect_node, dial_and_hello, gate_connection, send_leave, GateOutcome, HubIdentity,
    HELLO_DEADLINE,
};
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
fn node_daemon_bare_without_saved_hubs_prints_pair_hint() {
    let home = tempfile::tempdir().unwrap();
    // Bare `--node` with no saved hub explains how to pair and exits 0 (no network).
    let out = run_higgs(home.path(), &["--node"]);
    assert!(out.status.success(), "bare --node (no hubs) exits 0");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("higgs node"),
        "prints node identity: {stdout}"
    );
    assert!(
        stdout.contains("higgs --node <ticket>"),
        "prints the pairing hint: {stdout}"
    );
}

#[test]
fn node_list_empty_and_hub_miss() {
    let home = tempfile::tempdir().unwrap();
    // `--node --list` with no saved hubs exits 0 with a "no saved hubs" line.
    let list = run_higgs(home.path(), &["--node", "--list"]);
    assert!(list.status.success(), "--node --list exits 0");
    assert!(
        String::from_utf8_lossy(&list.stdout).contains("no saved hubs"),
        "empty list message: {}",
        String::from_utf8_lossy(&list.stdout)
    );
    // `--node --hub <unknown>` with nothing saved is an error (non-zero, no panic).
    let miss = run_higgs(home.path(), &["--node", "--hub", "ghost"]);
    assert!(!miss.status.success(), "--hub with no match exits non-zero");
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

#[test]
fn version_flag_prints_crate_version() {
    let home = tempfile::tempdir().unwrap();
    // `higgs --version` (and `-V`) print `higgs <crate version>` and exit 0. The
    // version MUST equal CARGO_PKG_VERSION so the CLI, Cargo.toml, and the release
    // tag/artifacts can never drift (the release workflow derives the tag from the
    // same Cargo.toml version).
    for flag in ["--version", "-V"] {
        let out = run_higgs(home.path(), &[flag]);
        assert!(out.status.success(), "`higgs {flag}` exits 0");
        assert_eq!(
            String::from_utf8_lossy(&out.stdout).trim(),
            format!("higgs {}", env!("CARGO_PKG_VERSION")),
            "`higgs {flag}` prints the crate version"
        );
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
            &HubIdentity::new(hub_id),
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
            &HubIdentity::new(hub_id),
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

/// Spawn `higgs --node <args>` against `home`, wait (≤30s) until it prints the pairing line,
/// and return the live child. Stdout is consumed by the reader; the caller SIGTERMs the child.
async fn run_node_until_paired(home: &Path, args: &[&str]) -> tokio::process::Child {
    use tokio::io::{AsyncBufReadExt, BufReader};
    let mut full = vec!["--node"];
    full.extend_from_slice(args);
    let mut child = tokio::process::Command::new(env!("CARGO_BIN_EXE_higgs"))
        .args(&full)
        .env("HIGGS_HOME", home)
        .env("HIGGS_IROH_LOCAL", "1")
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("spawn higgs --node");
    let mut lines = BufReader::new(child.stdout.take().unwrap()).lines();
    let paired = tokio::time::timeout(Duration::from_secs(30), async {
        while let Ok(Some(line)) = lines.next_line().await {
            if line.contains("paired with hub") {
                return true;
            }
        }
        false
    })
    .await;
    assert!(
        matches!(paired, Ok(true)),
        "node should pair within 30s (args {args:?})"
    );
    child
}

fn sigterm(child: &tokio::process::Child) {
    if let Some(pid) = child.id() {
        // SIGTERM (not SIGKILL): the node drains its workers + flushes coverage cleanly.
        unsafe {
            libc::kill(pid as libc::pid_t, libc::SIGTERM);
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn node_pairs_then_bare_reconnects_via_saved_hub_without_token() {
    // The Unit B core: pair once with a ticket+token, the node SAVES the hub to config.json,
    // and a later bare `higgs --node` (same HIGGS_HOME) reconnects via the allowlist — no token.
    let node_home = tempfile::tempdir().unwrap();
    let allow_dir = tempfile::tempdir().unwrap();

    // In-process hub that accepts repeatedly (pairing dial, then the bare-reconnect dial),
    // sharing ONE allowlist so the token-paired node is admitted token-free on reconnect.
    let hub = iroh::Endpoint::builder(iroh::endpoint::presets::Minimal)
        .alpns(vec![ALPN.to_vec()])
        .relay_mode(iroh::RelayMode::Disabled)
        .bind()
        .await
        .expect("bind hub");
    let hub_id = hub.id().to_string();
    let ticket = EndpointTicket::new(hub.addr()).to_string();
    let allow = Arc::new(Mutex::new(
        Allowlist::load(&allow_dir.path().join("pairings.json")).unwrap(),
    ));
    let tokens = Arc::new(Mutex::new(PairingTokens::new()));
    let token = tokens.lock().await.mint(now_ms(), 600_000);

    let hub_task = tokio::spawn(async move {
        loop {
            let Some(incoming) = hub.accept().await else {
                break;
            };
            let conn = match incoming.await {
                Ok(c) => c,
                Err(_) => continue,
            };
            let mut a = allow.lock().await;
            let mut t = tokens.lock().await;
            let _ = gate_connection(
                &conn,
                &mut a,
                &mut t,
                now_ms(),
                &HubIdentity {
                    id: hub_id.clone(),
                    name: "hub-e2e(srv)".into(),
                },
                Some("paired-node".into()),
                HELLO_DEADLINE,
            )
            .await;
            drop(a);
            drop(t);
            // Hold the connection open briefly so the node reads its HELLO reply; keep accepting.
            tokio::spawn(async move {
                let _ = tokio::time::timeout(Duration::from_secs(5), conn.closed()).await;
            });
        }
    });

    // Step 1 — pair with ticket + token. The node saves the hub + sets it default.
    let mut node1 = run_node_until_paired(node_home.path(), &[&ticket, &token]).await;
    sigterm(&node1);
    let _ = node1.wait().await;

    // The hub was persisted to config.json with the hub's friendly name as its label + default.
    let cfg = InstanceConfig::load(&node_home.path().join("config.json")).unwrap();
    let saved = cfg.default_saved_hub().expect("a default hub was saved");
    assert_eq!(saved.label, "hub-e2e(srv)", "saved the hub's friendly name");
    assert_eq!(saved.ticket, ticket, "saved the dialed ticket");

    // `--node --list` now shows the saved hub.
    let list = run_higgs(node_home.path(), &["--node", "--list"]);
    assert!(list.status.success());
    assert!(
        String::from_utf8_lossy(&list.stdout).contains("hub-e2e(srv)"),
        "list shows the saved hub: {}",
        String::from_utf8_lossy(&list.stdout)
    );

    // Step 2 — BARE `higgs --node`, same HIGGS_HOME, NO token: reconnects via the allowlist.
    let mut node2 = run_node_until_paired(node_home.path(), &[]).await;
    sigterm(&node2);
    let _ = node2.wait().await;

    hub_task.abort();
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
    let result = dial_and_hello(
        &node,
        ticket.endpoint_addr().clone(),
        self_id,
        String::new(),
        Some(token),
    )
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn link_pair_handles_node_leave() {
    use tokio::io::{AsyncBufReadExt, BufReader};

    let home = tempfile::tempdir().unwrap();
    // Spawn the real `higgs link pair` accept loop; read its ticket + token from stdout.
    let mut child = tokio::process::Command::new(env!("CARGO_BIN_EXE_higgs"))
        .args(["link", "pair"])
        .env("HIGGS_HOME", home.path())
        .env("HIGGS_IROH_LOCAL", "1")
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("spawn higgs link pair");

    let mut lines = BufReader::new(child.stdout.take().unwrap()).lines();
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

    // In-process node: pair (HELLO), then ask to LEAVE on the same connection.
    let node = iroh::Endpoint::builder(iroh::endpoint::presets::Minimal)
        .alpns(vec![ALPN.to_vec()])
        .relay_mode(iroh::RelayMode::Disabled)
        .bind()
        .await
        .expect("bind node");
    let self_id = node.id().to_string();
    let (conn, _hello) = connect_node(
        &node,
        ticket.endpoint_addr().clone(),
        self_id.clone(),
        String::new(),
        Some(token),
    )
    .await
    .expect("node pairs with the link-pair hub");
    // The node was added to the allowlist on admit.
    let pairings = home.path().join("pairings.json");
    assert!(
        Allowlist::load(&pairings).unwrap().contains(&self_id),
        "node is allowlisted after pairing"
    );

    // Now LEAVE — the link-pair loop handles it and removes the node from the allowlist.
    send_leave(&conn).await.expect("hub acks the leave");
    let mut gone = false;
    for _ in 0..50 {
        if !Allowlist::load(&pairings).unwrap().contains(&self_id) {
            gone = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert!(gone, "node removed from the allowlist after `node leave`");

    unsafe {
        libc::kill(child.id().expect("child pid") as libc::pid_t, libc::SIGTERM);
    }
    let _ = child.wait().await;
}
