//! Round-2 integration coverage for the fleet CLI + production hub error/reject arms.
//!
//! Targets branches the existing remote suite (`remote_cli`, `remote_pairing`,
//! `remote_hub_e2e`, `hub_server`, `cov_remote`) leaves uncovered — the *unhappy* paths
//! of the three node source files the integration gate solely owns:
//!
//! - `node/cli.rs`: the `higgs node leave` argument matrix (unknown flag / missing hub /
//!   nothing-to-leave), `higgs --node` unknown flag, `--node --list` printing a
//!   non-default saved hub, `link pair` REJECTING a bad-token dial and IGNORING a
//!   node-opened stream that is not a valid `M_NODE_LEAVE`, and `--node --hub <label>`
//!   resolving a saved hub then reporting a failed connect.
//! - `node/hub.rs`: the PRODUCTION accept loop's pre-auth (`gate_read_hello`) and
//!   post-auth (`gate_admit`) rejections, cold-start seeding of the persisted allowlist,
//!   and `serve_node_requests` handling an EOF / garbage / unknown-method node stream plus
//!   an allowlist-removal persistence failure (HG040).
//! - `node/data.rs`: the node chat relay's invalid-params (`-32602`) arm.
//!
//! Every test serializes on one process-global lock so a spawned child's environment
//! snapshot never races the in-process harness's `set_var` (the harness home-lock only
//! serializes among its own callers). In-process hub tests reach `node/hub.rs` through the
//! real `Higgs` facade (`hub_enable`/`pair`) and dial it from raw hermetic iroh endpoints,
//! exactly as an operator's node would.

mod common;

use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use iroh_tickets::endpoint::EndpointTicket;
use serde_json::json;
use tokio::io::{AsyncBufReadExt, BufReader};

use higgs::auth::{Allowlist, PairingTokens};
use higgs::config::{InstanceConfig, SavedHub};
use higgs::node::{
    connect_node, dial_and_hello, gate_connection, send_leave, GateOutcome, HubIdentity,
    HELLO_DEADLINE,
};
use higgs::remote::ALPN;
use higgs::rpc::{self, RpcFrame, RpcRequest};
use higgs::worker::M_CHAT;

use common::higgs_local;

// ── serialization + shared helpers ─────────────────────────────────────────────────────

/// One process-global async lock every test in this binary holds for its whole body. The
/// in-process harness (`higgs_local`) mutates `HIGGS_HOME`/`HIGGS_HF_ENDPOINT`/
/// `HIGGS_IROH_LOCAL` via `set_var`; a concurrent child `Command::spawn` snapshots the
/// environment. libtest runs test fns on parallel threads, so without this lock a set_var
/// could race a spawn's environ read. Holding it makes every test run strictly serially.
fn test_lock() -> &'static tokio::sync::Mutex<()> {
    static L: std::sync::OnceLock<tokio::sync::Mutex<()>> = std::sync::OnceLock::new();
    L.get_or_init(|| tokio::sync::Mutex::new(()))
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// Force the production iroh binds (`start_hub` → `bind_endpoint`) into relay-disabled
/// LAN-local mode so an in-process hub is hermetic + immediately dialable by the raw
/// `Minimal`/`RelayMode::Disabled` node endpoints these tests build, and `start_hub` skips
/// its 10s relay wait. SAFETY: every test serializes on [`test_lock`], so no other thread
/// reads/writes the process environment concurrently with this set.
fn force_iroh_local() {
    unsafe { std::env::set_var("HIGGS_IROH_LOCAL", "1") };
}

/// A hermetic (relay-disabled) iroh endpoint on the higgs remote ALPN — a raw stand-in for
/// a node dialing a hub.
async fn minimal_ep() -> iroh::Endpoint {
    iroh::Endpoint::builder(iroh::endpoint::presets::Minimal)
        .alpns(vec![ALPN.to_vec()])
        .relay_mode(iroh::RelayMode::Disabled)
        .bind()
        .await
        .expect("bind endpoint")
}

/// Run `higgs <args>` with an isolated `HIGGS_HOME` (+ hermetic iroh) and capture output.
fn run_higgs(home: &Path, args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_higgs"))
        .args(args)
        .env("HIGGS_HOME", home)
        .env("HIGGS_IROH_LOCAL", "1")
        .output()
        .expect("spawn higgs")
}

/// Read child lines until one contains `needle`, bounded by `secs`. Returns the matching
/// line, or `None` on timeout / stream end.
async fn read_until<R>(lines: &mut tokio::io::Lines<R>, needle: &str, secs: u64) -> Option<String>
where
    R: tokio::io::AsyncBufRead + Unpin,
{
    tokio::time::timeout(Duration::from_secs(secs), async {
        while let Ok(Some(l)) = lines.next_line().await {
            if l.contains(needle) {
                return Some(l);
            }
        }
        None
    })
    .await
    .ok()
    .flatten()
}

/// SIGTERM a tokio child (graceful: flushes its llvm-cov profile; a hard kill drops it).
fn sigterm(child: &tokio::process::Child) {
    if let Some(pid) = child.id() {
        unsafe { libc::kill(pid as libc::pid_t, libc::SIGTERM) };
    }
}

/// A spawned `higgs --node` child (std). SIGTERM + reap on drop.
struct NodeProc(Child);
impl Drop for NodeProc {
    fn drop(&mut self) {
        unsafe { libc::kill(self.0.id() as libc::pid_t, libc::SIGTERM) };
        let _ = self.0.wait();
    }
}

/// Dial the in-process production hub on `h` with a freshly-minted valid token from a raw
/// node endpoint, and wait until the hub's accept loop registers it as connected. Returns
/// the node endpoint (keep it alive), the admitted node-side connection, and the peer id.
async fn admit_into_hub(
    h: &common::LocalHiggs,
) -> (iroh::Endpoint, iroh::endpoint::Connection, String) {
    let pair = h.pair().await.expect("mint pairing");
    let ticket: EndpointTicket = pair.ticket.parse().expect("parse ticket");
    let addr = ticket.endpoint_addr().clone();
    let node = minimal_ep().await;
    let id = node.id().to_string();
    let (conn, _hello) = connect_node(&node, addr, id.clone(), "cov-node".into(), Some(pair.token))
        .await
        .expect("node admitted by the production hub");
    for _ in 0..100 {
        if h.nodes()
            .await
            .iter()
            .any(|n| n.connected && !n.is_local && n.endpoint_id == id)
        {
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    (node, conn, id)
}

// ── node/cli.rs: `higgs node leave` / `higgs --node` argument branches ──────────────────

/// The `higgs node leave` argument matrix and `higgs --node <unknown-flag>` all exit
/// non-zero with the specific usage/diagnostic on stderr — no network, no panic.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cli_node_leave_and_daemon_arg_errors() {
    let _serial = test_lock().lock().await;
    let home = tempfile::tempdir().unwrap();

    // `node leave --hub <unknown>`: the selector resolves to no saved hub → error.
    let miss = run_higgs(home.path(), &["node", "leave", "--hub", "ghost"]);
    assert!(!miss.status.success(), "leave --hub <miss> exits non-zero");
    assert!(
        String::from_utf8_lossy(&miss.stderr).contains("no saved hub matching"),
        "names the missing hub: {}",
        String::from_utf8_lossy(&miss.stderr)
    );

    // `node leave --hub` with no selector → usage error.
    let noselector = run_higgs(home.path(), &["node", "leave", "--hub"]);
    assert!(!noselector.status.success(), "leave --hub (bare) fails");
    assert!(
        String::from_utf8_lossy(&noselector.stderr).contains("usage: higgs node leave --hub"),
        "prints the --hub usage: {}",
        String::from_utf8_lossy(&noselector.stderr)
    );

    // `node leave --<garbage>` → unknown flag.
    let badflag = run_higgs(home.path(), &["node", "leave", "--frobnicate"]);
    assert!(!badflag.status.success(), "leave --frobnicate fails");
    assert!(
        String::from_utf8_lossy(&badflag.stderr).contains("unknown flag"),
        "names the unknown flag: {}",
        String::from_utf8_lossy(&badflag.stderr)
    );

    // `node leave` with nothing saved → nothing to leave.
    let nohub = run_higgs(home.path(), &["node", "leave"]);
    assert!(!nohub.status.success(), "leave with no saved hub fails");
    assert!(
        String::from_utf8_lossy(&nohub.stderr).contains("no saved hub to leave"),
        "explains there is nothing to leave: {}",
        String::from_utf8_lossy(&nohub.stderr)
    );

    // `--node --<garbage>` → unknown flag on the daemon parser.
    let daemon_badflag = run_higgs(home.path(), &["--node", "--frobnicate"]);
    assert!(
        !daemon_badflag.status.success(),
        "--node --frobnicate fails"
    );
    assert!(
        String::from_utf8_lossy(&daemon_badflag.stderr).contains("unknown flag"),
        "daemon names the unknown flag: {}",
        String::from_utf8_lossy(&daemon_badflag.stderr)
    );
}

/// `higgs --node --list` over a config carrying TWO saved hubs prints them both, marking
/// exactly the default one with `★default` (exercising both the default and the
/// non-default branch of the per-hub print).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn node_list_shows_default_and_non_default_hubs() {
    let _serial = test_lock().lock().await;
    let home = tempfile::tempdir().unwrap();

    // Build a config with two hubs; `remember_hub` makes the LAST one the default, so
    // alpha is a non-default saved hub and beta is the default.
    let mut cfg = InstanceConfig {
        name: "node-cov(test)".into(),
        ..Default::default()
    };
    cfg.remember_hub(SavedHub {
        hub_id: "alpha-endpoint-id-0001".into(),
        ticket: "tkt-alpha".into(),
        label: "alpha-hub".into(),
        last_used_ms: 111,
    });
    cfg.remember_hub(SavedHub {
        hub_id: "beta-endpoint-id-0002".into(),
        ticket: "tkt-beta".into(),
        label: "beta-hub".into(),
        last_used_ms: 222,
    });
    cfg.save(&home.path().join("config.json"))
        .expect("write config with two hubs");

    let out = run_higgs(home.path(), &["--node", "--list"]);
    assert!(out.status.success(), "--node --list exits 0");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("alpha-hub"), "lists alpha: {stdout}");
    assert!(stdout.contains("beta-hub"), "lists beta: {stdout}");
    assert!(
        stdout.contains("★default"),
        "marks exactly the default hub: {stdout}"
    );
    // The non-default hub line carries no star (the else branch ran).
    let alpha_line = stdout
        .lines()
        .find(|l| l.contains("alpha-hub"))
        .expect("alpha line present");
    assert!(
        !alpha_line.contains("★default"),
        "the non-default hub is NOT starred: {alpha_line:?}"
    );
}

// ── node/hub.rs: cold-start seed of the persisted allowlist ─────────────────────────────

/// A hub cold-started (first `hub_enable`, no pre-existing fleet) over a NON-EMPTY
/// `pairings.json` seeds each persisted node into the fleet view as disconnected — so a
/// previously-paired node shows up before it reconnects.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn hub_cold_start_seeds_allowlisted_node() {
    let _serial = test_lock().lock().await;
    let Some(h) = higgs_local(&[]).await else {
        eprintln!("SKIP hub_cold_start_seeds_allowlisted_node: no tiny GGUF");
        return;
    };
    force_iroh_local();

    // Pre-seed the allowlist BEFORE the hub is enabled: a node that paired in a past run.
    let seeded = format!("cov-seeded-node-{}", "0".repeat(40));
    {
        let mut allow = Allowlist::load(&h.home().join("pairings.json")).expect("load allowlist");
        allow
            .add(seeded.clone(), Some("ghost-node".into()))
            .expect("persist a pre-existing pairing");
    }

    // Cold start: first enable builds a fresh fleet and seeds it from pairings.json.
    h.hub_enable().await.expect("enable hub network");

    let seeded_view = h
        .nodes()
        .await
        .into_iter()
        .find(|n| n.endpoint_id == seeded)
        .expect("the persisted node was seeded into the fleet view");
    assert!(
        !seeded_view.connected,
        "a seeded-but-not-reconnected node shows disconnected: {seeded_view:?}"
    );
    assert!(!seeded_view.is_local, "the seeded node is a remote node");

    h.shutdown().await;
}

// ── node/hub.rs: production accept-loop rejections + serve_node_requests ─────────────────

/// Drive the PRODUCTION hub accept loop through every rejection + the node-request server:
/// a spoofed-identity dial is refused pre-auth (`gate_read_hello`), a valid HELLO with a
/// bad token is refused post-auth (`gate_admit`), and a genuinely-admitted node's opened
/// streams exercise `serve_node_requests` — an EOF stream, a garbage frame, and an
/// unknown-method request answered with a JSON-RPC method-not-found error. Throughout, only
/// the admitted node is ever counted connected.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn hub_production_rejects_bad_dials_and_serves_requests() {
    let _serial = test_lock().lock().await;
    let Some(h) = higgs_local(&[]).await else {
        eprintln!("SKIP hub_production_rejects_bad_dials_and_serves_requests: no tiny GGUF");
        return;
    };
    force_iroh_local();
    h.hub_enable().await.expect("enable hub network");
    let pair = h.pair().await.expect("mint pairing");
    let ticket: EndpointTicket = pair.ticket.parse().expect("parse ticket");
    let addr = ticket.endpoint_addr().clone();

    // (a) Pre-auth reject: a peer that claims a node_id != its TLS id is dropped by
    // gate_read_hello — the dial errors and nothing is admitted.
    let spoof_ep = minimal_ep().await;
    let spoof = connect_node(
        &spoof_ep,
        addr.clone(),
        "spoofed-not-my-real-endpoint-id".into(),
        "spoofer".into(),
        None,
    )
    .await;
    assert!(
        spoof.is_err(),
        "a spoofed-identity dial is rejected: {spoof:?}"
    );

    // (b) Post-auth reject: a well-formed HELLO with an invalid token → HG022.
    let badtok_ep = minimal_ep().await;
    let badtok_id = badtok_ep.id().to_string();
    let badtok = connect_node(
        &badtok_ep,
        addr.clone(),
        badtok_id,
        "badtoken".into(),
        Some("htk_deadbeefdeadbeef".into()),
    )
    .await
    .expect_err("a bad pairing token is rejected");
    assert!(
        badtok.to_string().contains("HG022"),
        "the node sees the typed HG022 token-invalid code: {badtok}"
    );

    // (c) Admit a real node, then drive serve_node_requests on its connection.
    let (_node, conn, admitted_id) = admit_into_hub(&h).await;

    // EOF stream: open + finish with no data → the hub reads None and loops on (303).
    let eof_reply = open_send_read(&conn, None).await;
    assert!(eof_reply.is_none(), "an EOF node stream draws no reply");

    // Garbage frame: a non-decodable line → the hub skips it and loops on (307).
    let garbage_reply = open_send_read(&conn, Some("this-is-not-a-json-rpc-frame")).await;
    assert!(
        garbage_reply.is_none(),
        "a garbage node stream draws no reply"
    );

    // Unknown method: a valid request whose method is not M_NODE_LEAVE → method-not-found.
    let req = RpcRequest {
        jsonrpc: "2.0".into(),
        id: 7,
        method: "higgs/node/definitely-not-leave".into(),
        params: json!({}),
    };
    let line = format!("{}\n", rpc::encode(&RpcFrame::Request(req)));
    let reply = open_send_read(&conn, Some(line.trim_end()))
        .await
        .expect("unknown-method request draws a reply");
    match rpc::decode(&reply).expect("decode hub reply") {
        RpcFrame::Response(resp) => {
            let err = resp
                .error
                .expect("an unknown node method is a method-not-found error");
            assert_eq!(err.code, -32601, "JSON-RPC method-not-found code: {err:?}");
        }
        other => panic!("expected a Response, got {other:?}"),
    }

    // Only the admitted node is ever connected; the two rejects never joined.
    let connected: Vec<String> = h
        .nodes()
        .await
        .into_iter()
        .filter(|n| n.connected && !n.is_local)
        .map(|n| n.endpoint_id)
        .collect();
    assert_eq!(
        connected,
        vec![admitted_id],
        "exactly the admitted node is connected (rejected dials never joined)"
    );

    drop(conn);
    h.shutdown().await;
}

/// Open a bidirectional stream on `conn`, optionally write one `line` (a newline is
/// appended), finish the send half, and return the FIRST reply line the peer writes back
/// (or `None` if the peer closes without replying).
async fn open_send_read(conn: &iroh::endpoint::Connection, line: Option<&str>) -> Option<String> {
    let (mut send, recv) = conn.open_bi().await.expect("open_bi");
    if let Some(l) = line {
        // `write_all`/`finish` are inherent iroh SendStream methods (no AsyncWriteExt).
        send.write_all(format!("{l}\n").as_bytes())
            .await
            .expect("write frame");
    }
    let _ = send.finish();
    let mut lines = BufReader::new(recv).lines();
    tokio::time::timeout(Duration::from_secs(5), lines.next_line())
        .await
        .ok()
        .and_then(Result::ok)
        .flatten()
}

/// A node's self-`leave` whose durable allowlist removal CANNOT persist (the hub's home is
/// made unwritable) is answered with the typed HG040 persistence error, not a false
/// `left` — so `higgs node leave` keeps the node's saved hub.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn hub_node_leave_persist_failure_is_hg040() {
    use std::os::unix::fs::PermissionsExt;

    let _serial = test_lock().lock().await;
    let Some(h) = higgs_local(&[]).await else {
        eprintln!("SKIP hub_node_leave_persist_failure_is_hg040: no tiny GGUF");
        return;
    };
    force_iroh_local();
    h.hub_enable().await.expect("enable hub network");

    let (_node, conn, _id) = admit_into_hub(&h).await;

    // Make the hub's home read-only so the allowlist's atomic save (temp create in this
    // dir) fails when the leave tries to remove the node.
    let home = h.home().to_path_buf();
    std::fs::set_permissions(&home, std::fs::Permissions::from_mode(0o500))
        .expect("chmod home read-only");

    let err = send_leave(&conn)
        .await
        .expect_err("leave with an unwritable store fails");
    // Restore BEFORE any assertion so shutdown can flush regardless of the result.
    std::fs::set_permissions(&home, std::fs::Permissions::from_mode(0o700))
        .expect("restore home perms");

    assert!(
        err.to_string().contains("HG040"),
        "a failed durable removal surfaces the typed HG040 persistence code: {err}"
    );

    drop(conn);
    h.shutdown().await;
}

// ── node/cli.rs: `link pair` reject + non-leave post-admit stream ────────────────────────

/// Spawn the real `higgs link pair` hub loop; a bad-token dial is printed as `rejected`,
/// and a genuinely-paired node that opens a stream which is NOT a valid `M_NODE_LEAVE`
/// (a non-leave request, a garbage frame, an immediate EOF) is IGNORED — the node stays
/// allowlisted (only a real leave retires it).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn link_pair_rejects_bad_token_and_ignores_non_leave() {
    let _serial = test_lock().lock().await;
    let home = tempfile::tempdir().unwrap();

    let mut child = tokio::process::Command::new(env!("CARGO_BIN_EXE_higgs"))
        .args(["link", "pair"])
        .env("HIGGS_HOME", home.path())
        .env("HIGGS_IROH_LOCAL", "1")
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn higgs link pair");
    let mut lines = BufReader::new(child.stdout.take().unwrap()).lines();

    // Read the printed ticket + token.
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
    let addr = ticket.endpoint_addr().clone();
    let token = token.expect("token line");

    // (a) A bad-token dial → the loop prints `rejected <peer> [<code>]`.
    let stranger = minimal_ep().await;
    let stranger_id = stranger.id().to_string();
    let _ = dial_and_hello(
        &stranger,
        addr.clone(),
        stranger_id,
        String::new(),
        Some("htk_notarealtoken".into()),
    )
    .await;
    assert!(
        read_until(&mut lines, "rejected", 20).await.is_some(),
        "link pair printed a rejection for the bad-token dial"
    );

    // A genuinely-paired node that opens a non-leave stream — three shapes, each on its own
    // admitted connection (a burned token → subsequent redials are allowlist-only).
    let node = minimal_ep().await;
    let node_id = node.id().to_string();
    let pairings = home.path().join("pairings.json");

    // (b) valid non-leave request.
    let req = RpcRequest {
        jsonrpc: "2.0".into(),
        id: 1,
        method: "higgs/node/not-leave".into(),
        params: json!({}),
    };
    let non_leave = format!("{}\n", rpc::encode(&RpcFrame::Request(req)));
    admit_and_send(
        &node,
        &addr,
        &node_id,
        Some(&token),
        &mut lines,
        Some(&non_leave),
    )
    .await;

    // (c) garbage (non-request) frame.
    admit_and_send(
        &node,
        &addr,
        &node_id,
        None,
        &mut lines,
        Some("garbage-non-frame\n"),
    )
    .await;

    // (d) immediate EOF (open + finish, no bytes).
    admit_and_send(&node, &addr, &node_id, None, &mut lines, None).await;

    // None of the non-leave streams retired the node: it is STILL allowlisted (a real
    // leave WOULD have removed it).
    assert!(
        Allowlist::load(&pairings).unwrap().contains(&node_id),
        "a non-leave post-admit stream never removes the node from the allowlist"
    );

    sigterm(&child);
    let _ = child.wait().await;
}

/// Pair `node` with the running `link pair` loop (waiting for its `paired` stdout line),
/// open one bidirectional stream, and send `frame` (or finish with no bytes when it is
/// `None`) — exercising one `link_pair_post_admit` non-leave path.
async fn admit_and_send<R>(
    node: &iroh::Endpoint,
    addr: &iroh::EndpointAddr,
    node_id: &str,
    token: Option<&str>,
    lines: &mut tokio::io::Lines<R>,
    frame: Option<&str>,
) where
    R: tokio::io::AsyncBufRead + Unpin,
{
    let (conn, _hello) = connect_node(
        node,
        addr.clone(),
        node_id.to_string(),
        String::new(),
        token.map(str::to_string),
    )
    .await
    .expect("node pairs with the link-pair loop");
    assert!(
        read_until(lines, "paired", 20).await.is_some(),
        "link pair admitted the node"
    );
    let (mut send, recv) = conn.open_bi().await.expect("open_bi");
    if let Some(f) = frame {
        send.write_all(f.as_bytes()).await.expect("write frame");
    }
    let _ = send.finish();
    // post_admit returns after reading (or on EOF); the loop then drops the hub-side conn,
    // which our recv observes as end-of-stream — a clean sync before the next redial.
    let mut rlines = BufReader::new(recv).lines();
    let _ = tokio::time::timeout(Duration::from_secs(5), rlines.next_line()).await;
    drop(conn);
}

// ── node/cli.rs: `--node --hub <label>` resolves a saved hub, then reports connect failure ─

/// `higgs --node --hub <label>` resolves the named saved hub from config.json (the success
/// branch of the `--hub` selector), dials it, and — the target being a hub that rejects the
/// unpaired node — reports the failed connect on stderr and keeps retrying.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn node_daemon_hub_flag_resolves_then_connect_fails() {
    let _serial = test_lock().lock().await;
    let node_home = tempfile::tempdir().unwrap();

    // A raw hub that ADMITS nothing (empty allowlist, no tokens) → every dial is rejected.
    let hub = minimal_ep().await;
    let hub_id = hub.id().to_string();
    let hub_ticket = EndpointTicket::new(hub.addr()).to_string();
    let allow_dir = tempfile::tempdir().unwrap();
    let hub_task = tokio::spawn(async move {
        let mut allow = Allowlist::load(&allow_dir.path().join("pairings.json")).unwrap();
        let mut tokens = PairingTokens::new();
        while let Some(incoming) = hub.accept().await {
            let Ok(conn) = incoming.await else { continue };
            let outcome = gate_connection(
                &conn,
                &mut allow,
                &mut tokens,
                now_ms(),
                &HubIdentity::new(hub_id.clone()),
                None,
                HELLO_DEADLINE,
            )
            .await;
            assert!(
                matches!(outcome, GateOutcome::Rejected { .. }),
                "the unpaired node is rejected: {outcome:?}"
            );
        }
    });

    // Save that hub under a friendly label so `--hub myhub` resolves it.
    let mut cfg = InstanceConfig {
        name: "node-covf(test)".into(),
        ..Default::default()
    };
    cfg.remember_hub(SavedHub {
        hub_id: hub_id_of(&hub_ticket),
        ticket: hub_ticket.clone(),
        label: "myhub".into(),
        last_used_ms: now_ms(),
    });
    cfg.save(&node_home.path().join("config.json"))
        .expect("write saved-hub config");

    let mut child = tokio::process::Command::new(env!("CARGO_BIN_EXE_higgs"))
        .args(["--node", "--hub", "myhub"])
        .env("HIGGS_HOME", node_home.path())
        .env("HIGGS_IROH_LOCAL", "1")
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn higgs --node --hub");
    let mut errlines = BufReader::new(child.stderr.take().unwrap()).lines();

    // Resolving the saved hub (the `--hub` success branch) then dialing the reject-only hub
    // surfaces the daemon's connect-failed diagnostic.
    assert!(
        read_until(&mut errlines, "connect failed", 30)
            .await
            .is_some(),
        "the daemon resolved the saved hub and reported the failed connect"
    );

    sigterm(&child);
    let _ = child.wait().await;
    hub_task.abort();
}

/// The hub_id field is opaque to `--hub` label lookup (it matches on `label` first), so any
/// stable non-empty string works; derive one from the ticket so it is deterministic.
fn hub_id_of(ticket: &str) -> String {
    ticket.chars().take(16).collect()
}

// ── node/data.rs: node chat relay rejects malformed params ──────────────────────────────

/// A hub-opened `M_CHAT` data stream whose params cannot deserialize into `NodeChatParams`
/// draws a JSON-RPC `-32602` (invalid params) reply from the node's chat relay — before any
/// worker is touched (so no GGUF is required).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn node_relay_rejects_malformed_chat_params() {
    let _serial = test_lock().lock().await;
    let node_home = tempfile::tempdir().unwrap();

    // A raw hub endpoint that admits the node, so we can open a data stream to it.
    let hub = minimal_ep().await;
    let hub_id = hub.id().to_string();
    let ticket = EndpointTicket::new(hub.addr()).to_string();
    let mut allow = Allowlist::load(&node_home.path().join("hub-pairings.json")).unwrap();
    let mut tokens = PairingTokens::new();
    let token = tokens.mint(now_ms(), 600_000);

    // Spawn the real node daemon (no model dir needed — the malformed-params path returns
    // before any worker is used).
    #[allow(clippy::zombie_processes)] // reaped by NodeProc::drop
    let _node = NodeProc(
        Command::new(env!("CARGO_BIN_EXE_higgs"))
            .arg("--node")
            .arg(&ticket)
            .arg(&token)
            .env("HIGGS_HOME", node_home.path())
            .env("HIGGS_IROH_LOCAL", "1")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn higgs --node"),
    );

    // Accept + gate the node's dial → an admitted hub-side connection.
    let incoming = tokio::time::timeout(Duration::from_secs(30), hub.accept())
        .await
        .expect("node dialed within 30s")
        .expect("incoming");
    let conn = incoming.await.expect("connection");
    let outcome = gate_connection(
        &conn,
        &mut allow,
        &mut tokens,
        now_ms(),
        &HubIdentity::new(hub_id),
        Some("cov".into()),
        HELLO_DEADLINE,
    )
    .await;
    assert!(
        matches!(outcome, GateOutcome::Admitted { .. }),
        "node admitted: {outcome:?}"
    );

    // Open a DATA stream carrying an M_CHAT whose params are not a NodeChatParams object.
    let req = RpcRequest {
        jsonrpc: "2.0".into(),
        id: 42,
        method: M_CHAT.into(),
        params: json!("not-a-chat-params-object"),
    };
    let (mut send, recv) = conn.open_bi().await.expect("open data stream");
    send.write_all(format!("{}\n", rpc::encode(&RpcFrame::Request(req))).as_bytes())
        .await
        .expect("write malformed chat");
    let _ = send.finish();

    let mut rlines = BufReader::new(recv).lines();
    let line = tokio::time::timeout(Duration::from_secs(15), rlines.next_line())
        .await
        .expect("relay replied within 15s")
        .expect("io")
        .expect("a reply line");
    match rpc::decode(&line).expect("decode relay reply") {
        RpcFrame::Response(resp) => {
            let err = resp.error.expect("malformed chat params → an error reply");
            assert_eq!(err.code, -32602, "JSON-RPC invalid-params code: {err:?}");
            assert!(
                err.message.contains("invalid chat params"),
                "the relay names the invalid chat params: {}",
                err.message
            );
        }
        other => panic!("expected a Response, got {other:?}"),
    }
}
