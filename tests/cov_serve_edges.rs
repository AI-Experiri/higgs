//! Integration coverage for the crash/degradation edges of the serving stack,
//! plus the pub render/classify surfaces reachable without launchd or a real
//! network:
//!
//! - `serve/stream.rs`: the mid-generation failure arm — a worker SIGKILLed
//!   while an SSE chat stream is open must surface the OpenAI error envelope
//!   as a data event and still close with `data: [DONE]` (never hang).
//! - `supervisor.rs`: the crash-restart FSM's failure arms driven end to end
//!   through the in-process facade — spawn failure (HG006 + sysinfo degrading
//!   to no devices), respawn give-up when the worker exe vanished, and the
//!   post-restart load REPLAY failing when the model file vanished (the model
//!   must not be silently resurrected as loaded).
//! - `node/preflight.rs`: the pure report/advice/classify functions plus the
//!   ticket-derived `run()` on an IP-literal relay (no DNS, no real network;
//!   the only sockets are loopback UDP probes at a port nothing serves).
//! - `node/service.rs`: the pub plan-guard + unit-render functions (`systemd_unit`
//!   escaping, cross-scope refusals, headless hint, linger note) — everything
//!   that DECIDES or RENDERS, no launchctl/systemctl ever runs.
//! - `load_robustness.rs`: the OOM classify + degrade-ladder pub API (`oom_ladder`
//!   is pure; the real-OOM trigger needs an actual VRAM exhaustion, so the
//!   ladder's CONTRACT is pinned here instead).

mod common;

use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use common::{higgs_local, serve_v1_local, TINY_MODEL_ID};
use higgs::load_robustness::{is_oom_reason, oom_ladder};
use higgs::node::preflight as pf;
use higgs::node::service as svc;
use higgs::remote::NodeLoadParams;
use higgs::worker::engine::llamacpp::params::LlamaCppParams;
use higgs::worker::engine::GpuLayers;
use higgs::{Higgs, HiggsConfig};
use iroh::{EndpointAddr, SecretKey, TransportAddr};
use serde_json::json;

// ── shared helpers ────────────────────────────────────────────────────────────

/// PIDs of the `--higgs-worker` children of THIS test process (workers spawn from
/// the real `higgs` binary via the `worker_exe` seam, so they are direct children
/// of the in-process test runtime). Safe across parallel test BINARIES (separate
/// processes have separate children); within this binary the `higgs_local` home
/// lock serializes every facade test, so the set is always ours alone.
fn worker_child_pids() -> Vec<u32> {
    let out = std::process::Command::new("pgrep")
        .args(["-P", &std::process::id().to_string()])
        .output()
        .expect("run pgrep");
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| l.trim().parse::<u32>().ok())
        .collect()
}

/// SIGKILL — an abrupt worker death (no shutdown handshake), the crash the
/// supervisor's restart FSM exists for.
fn kill9(pid: u32) {
    unsafe { libc::kill(pid as libc::pid_t, libc::SIGKILL) };
}

/// A facade with a caller-chosen `worker_exe`, sharing the isolated `HIGGS_HOME`
/// and scan root of an already-held `LocalHiggs` (whose process-global home lock
/// keeps the env stable for our whole test).
async fn facade_with_exe(scan_root: &Path, exe: std::path::PathBuf) -> Arc<Higgs> {
    let config = HiggsConfig {
        lmstudio_dirs: vec![scan_root.to_path_buf()],
        hf_dirs: vec![],
        ollama_dirs: vec![],
        default_load: HiggsConfig::default().default_load,
        worker_exe: Some(exe),
    };
    let higgs = Arc::new(Higgs::new(config));
    higgs.start().await.expect("facade start");
    higgs
}

// ── serve/stream.rs: worker death mid-SSE-stream ─────────────────────────────

/// A worker SIGKILLed while a `/v1/chat/completions` SSE stream is live must end
/// the stream LOUDLY: an OpenAI error envelope as a data event, then `[DONE]`,
/// then stream close — a client must never hang on a silent half-open stream.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sse_stream_emits_error_envelope_then_done_when_worker_dies_mid_generation() {
    let Some(local) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP sse worker-death test: tiny gguf not found (set HIGGS_TEST_GGUF)");
        return;
    };
    local
        .load(TINY_MODEL_ID, None)
        .await
        .expect("load tiny model");
    let (base, guard) = serve_v1_local(local.handle()).await;

    let pids = worker_child_pids();
    assert_eq!(
        pids.len(),
        1,
        "exactly one worker child after load: {pids:?}"
    );
    let worker = pids[0];

    // max_tokens near the tiny model's ctx budget so generation runs for many
    // tens of ms — the kill below lands right after the response HEADERS arrive
    // (assembly spawned, generation just starting), far inside that window.
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{base}/v1/chat/completions"))
        .json(&json!({
            "model": TINY_MODEL_ID, "stream": true, "max_tokens": 400,
            "messages": [{ "role": "user", "content": "Tell me a very long story about a dragon." }]
        }))
        .send()
        .await
        .expect("stream request accepted");
    assert!(
        resp.status().is_success(),
        "SSE stream opened: {}",
        resp.status()
    );

    // Kill the worker the instant the stream is open (direct syscall — no
    // subprocess latency), then drain the body to completion, bounded.
    kill9(worker);
    let body = tokio::time::timeout(Duration::from_secs(30), resp.text())
        .await
        .expect("SSE stream must CLOSE after worker death, not hang")
        .expect("read SSE body");

    let frames: Vec<&str> = body
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .collect();
    assert!(
        frames.first().is_some_and(|f| f.contains("assistant")),
        "stream opened with the assistant-role chunk: {frames:?}"
    );
    assert!(
        frames.iter().any(|f| f.contains("\"error\"")),
        "worker death surfaces the OpenAI error envelope mid-stream: {frames:?}"
    );
    assert_eq!(
        frames.last().copied(),
        Some("[DONE]"),
        "stream still terminates with [DONE] after the error: {frames:?}"
    );

    // Stream fully drained above — never leave an SSE stream open across teardown.
    guard.shutdown().await;
    local.shutdown().await;
}

// ── supervisor.rs: spawn failure fast-fails load and degrades sysinfo ────────

/// A worker exe that cannot be spawned surfaces `[HG006]` on load (no hang, no
/// partial worker) and makes the device probe DEGRADE to an empty list — the
/// facade still answers instead of failing the whole system query.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unspawnable_worker_exe_fails_load_with_hg006_and_sysinfo_returns_no_devices() {
    let Some(local) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP spawn-failure test: tiny gguf not found (set HIGGS_TEST_GGUF)");
        return;
    };
    // Points at a path that does not exist — every factory call fails at spawn.
    let bad = facade_with_exe(
        local.scan_root(),
        local.home().join("no-such-worker-binary"),
    )
    .await;

    let err = bad
        .load(TINY_MODEL_ID, None)
        .await
        .expect_err("load with an unspawnable worker exe must fail");
    assert!(
        err.to_string().contains("HG006"),
        "spawn failure carries the [HG006] worker-spawn code: {err}"
    );

    // The transient sysinfo worker uses the same factory: spawn failure must
    // yield an EMPTY device list (the documented degradation), not an error.
    assert!(
        bad.sysinfo().await.is_empty(),
        "sysinfo degrades to no devices when the worker cannot spawn"
    );

    bad.stop().await;
    local.shutdown().await;
}

// ── supervisor.rs: respawn give-up when the worker exe vanished ──────────────

/// A crash whose RESPAWN fails (worker exe deleted between spawns) must be
/// terminal-but-contained: the dead child is reaped, no replacement ever
/// appears, and the node survives to report a typed spawn error on the next
/// explicit load — never a crash loop, never a zombie.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn respawn_gives_up_cleanly_when_worker_exe_vanishes_and_node_survives() {
    let Some(local) = higgs_local(&[TINY_MODEL_ID, "cov/respawn"]).await else {
        eprintln!("SKIP respawn-give-up test: tiny gguf not found (set HIGGS_TEST_GGUF)");
        return;
    };
    // A COPY of the real higgs binary hosts the worker role, so deleting it
    // breaks only the RESPAWN (the running worker keeps its loaded image).
    let exe_dir = tempfile::tempdir().expect("temp exe dir");
    let exe_copy = exe_dir.path().join("higgs-copy");
    std::fs::copy(env!("CARGO_BIN_EXE_higgs"), &exe_copy).expect("copy higgs binary");
    let b = facade_with_exe(local.scan_root(), exe_copy.clone()).await;

    b.load("cov/respawn", None)
        .await
        .expect("load via copied worker exe");
    let pids = worker_child_pids();
    assert_eq!(pids.len(), 1, "one worker child after load: {pids:?}");

    // Break the respawn, then crash the worker.
    std::fs::remove_file(&exe_copy).expect("delete copied worker exe");
    kill9(pids[0]);

    // The reader observes EOF, backs off 1 s, fails the respawn, reaps the dead
    // child, and gives up: the child set empties and the worker reads dead.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(20);
    loop {
        let s = b.status().await.expect("status stays answerable");
        if !s.worker_alive && worker_child_pids().is_empty() {
            break;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "respawn give-up must reap the dead worker: alive={} kids={:?}",
            s.worker_alive,
            worker_child_pids()
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    // Stability window (bounded poll, spans several 1 s backoff periods): a
    // regressed give-up would respawn a child here — none may ever appear.
    for _ in 0..10 {
        tokio::time::sleep(Duration::from_millis(250)).await;
        assert!(
            worker_child_pids().is_empty(),
            "no replacement worker may appear after the terminal give-up"
        );
    }

    // The node SURVIVES the give-up: a fresh load of another model reports the
    // spawn failure as a typed [HG006] error instead of wedging.
    let err = b
        .load(TINY_MODEL_ID, None)
        .await
        .expect_err("load after the exe vanished must fail at spawn");
    assert!(
        err.to_string().contains("HG006"),
        "post-give-up load surfaces [HG006]: {err}"
    );

    b.stop().await;
    local.shutdown().await;
}

// ── supervisor.rs: crash-restart replay fails when the GGUF vanished ─────────

/// After a crash, the supervisor respawns the worker and REPLAYS the last load.
/// If the model file vanished in between, the replay must fail — respawn
/// attempts keep happening (a worker child reappears) but the model must NEVER
/// be reported as loaded again (no silent resurrection of a model that no
/// longer exists on disk), and the facade must stay answerable throughout.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn crash_restart_does_not_resurrect_model_whose_gguf_vanished() {
    let Some(local) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP replay-failure test: tiny gguf not found (set HIGGS_TEST_GGUF)");
        return;
    };
    local
        .load(TINY_MODEL_ID, None)
        .await
        .expect("load tiny model");
    let s = local.status().await.expect("status after load");
    assert!(
        s.worker_alive && s.loaded.is_some(),
        "model resident before the crash: {s:?}"
    );
    let pids = worker_child_pids();
    assert_eq!(pids.len(), 1, "one worker child after load: {pids:?}");
    let old = pids[0];

    // The model file vanishes while resident, then the worker crashes: the
    // restart succeeds (the worker exe still exists) but the M_LOAD replay must
    // fail against the deleted file.
    std::fs::remove_file(local.staged_gguf(TINY_MODEL_ID)).expect("delete staged gguf");
    kill9(old);

    // Restart proof: a NEW worker child appears (1 s backoff + respawn). The
    // replayed M_LOAD then fails against the deleted file and the respawned
    // worker dies again, so each child is short-lived — poll fast enough to
    // catch one of the (repeating, ~1 s apart) respawn windows.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(25);
    loop {
        if worker_child_pids().iter().any(|p| *p != old) {
            break;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "supervisor must attempt a respawn after the crash"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    // No-resurrection proof: across a multi-restart-cycle window the model must
    // NEVER read as loaded again (every replay fails against the deleted file),
    // and status must stay answerable the whole time (no wedge, no hang).
    for _ in 0..30 {
        let s = local
            .status()
            .await
            .expect("status stays answerable during the crash cycle");
        assert!(
            s.loaded.is_none(),
            "deleted model must never be resurrected as loaded: {s:?}"
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    local.shutdown().await;
}

// ── node/preflight.rs: styling + host-context advice ─────────────────────────

/// `Style` paints only when enabled; `Style::auto()`'s decision is consistent
/// with what it renders (colored escapes iff enabled — byte-plain otherwise).
#[test]
fn preflight_style_paints_only_when_enabled() {
    let on = pf::Style { enabled: true };
    assert_eq!(on.head("t"), "\x1b[1mt\x1b[0m");
    assert_eq!(on.ok("t"), "\x1b[32m✓ t\x1b[0m");
    assert_eq!(on.warn("t"), "\x1b[33m! t\x1b[0m");
    assert_eq!(on.fail("t"), "\x1b[31m✗ t\x1b[0m");

    let off = pf::Style { enabled: false };
    assert_eq!(off.fail("t"), "✗ t", "disabled style stays byte-plain");

    // auto() decides from the real tty/NO_COLOR state; whatever it decided,
    // its rendering must MATCH the decision.
    let auto = pf::Style::auto();
    if auto.enabled {
        assert!(auto.ok("x").contains("\x1b["));
    } else {
        assert_eq!(auto.ok("x"), "✓ x");
    }
}

/// The macOS Local Network advisory gates on platform and names the SSH
/// popup-cannot-appear caveat only for SSH sessions.
#[test]
fn preflight_local_network_advice_gates_on_macos_and_names_ssh() {
    assert_eq!(
        pf::local_network_advice(false, true),
        None,
        "not a macOS concern off-macOS"
    );
    let plain = pf::local_network_advice(true, false).expect("macOS advisory");
    assert!(plain.contains("Local Network"));
    assert!(!plain.contains("SSH"), "no SSH caveat for a local session");
    let ssh = pf::local_network_advice(true, true).expect("macOS+SSH advisory");
    assert!(
        ssh.contains("SSH") && ssh.contains("Terminal"),
        "SSH advisory names the popup limitation and the fix: {ssh}"
    );
}

// ── node/preflight.rs: DNS packet guards + loopback probes ───────────────────

/// The response validator rejects short and id-mismatched datagrams, and the
/// resolv.conf parser requires whitespace after the `nameserver` keyword.
#[test]
fn preflight_dns_guards_reject_short_mismatched_and_unspaced_lines() {
    assert!(
        !pf::dns_response_is_alive(&[0u8; 4], 7),
        "short datagram is not a response"
    );
    let mut resp = vec![0u8; 12];
    resp[0..2].copy_from_slice(&9u16.to_be_bytes());
    resp[2] = 0x80;
    assert!(
        !pf::dns_response_is_alive(&resp, 7),
        "txn-id mismatch is not OUR response"
    );
    assert!(
        pf::dns_response_is_alive(&resp, 9),
        "matching id + QR bit is"
    );
    assert_eq!(
        pf::parse_resolv_conf("nameserver1.2.3.4\nnameserver 8.8.8.8"),
        vec!["8.8.8.8".parse::<std::net::IpAddr>().unwrap()],
        "`nameserver1.2.3.4` (no whitespace) is NOT a nameserver line"
    );
}

/// Probing a nameserver nobody runs (loopback:53) reports DEAD (`None`) for
/// both address families, and a zero timeout gives up before any wait — the
/// probe can never block a preflight past its budget. Loopback-only traffic.
#[tokio::test]
async fn preflight_probe_reports_dead_for_silent_loopback_and_zero_timeout() {
    // IPv4 loopback: nothing serves DNS on 127.0.0.1:53 in this environment —
    // the send lands nowhere and the bounded wait expires → dead.
    let v4 = pf::probe_nameserver(
        "127.0.0.1".parse().unwrap(),
        "relay.example",
        Duration::from_millis(400),
    )
    .await;
    assert_eq!(v4, None, "silent 127.0.0.1:53 reads as a dead nameserver");

    // IPv6 loopback exercises the `[::]:0` bind arm.
    let v6 = pf::probe_nameserver(
        "::1".parse().unwrap(),
        "relay.example",
        Duration::from_millis(400),
    )
    .await;
    assert_eq!(v6, None, "silent [::1]:53 reads as a dead nameserver");

    // Zero budget: the deadline is already past on the first loop turn.
    let zero = pf::probe_nameserver("127.0.0.1".parse().unwrap(), "x", Duration::ZERO).await;
    assert_eq!(zero, None, "zero timeout returns dead without waiting");
}

// ── node/preflight.rs: ticket split, hopeless gate, failure advice ───────────

/// Ticket addresses split into relay hosts + direct socket addrs (unknown
/// custom transports are ignored), the hopeless gate fires only for a
/// relay-ONLY ticket needing DNS with EVERY nameserver dead, and the failure
/// advice names each applicable cause.
#[test]
fn preflight_ticket_split_hopeless_gate_and_advice_branches() {
    let pk = SecretKey::generate().public();
    let direct: SocketAddr = "192.168.7.9:7842".parse().unwrap();
    let addr = EndpointAddr::from_parts(
        pk,
        [
            TransportAddr::Relay("https://relay.example.com/".parse().unwrap()),
            TransportAddr::Ip(direct),
            // An unknown transport kind must be skipped, not break the split.
            TransportAddr::Custom("1f_00ff".parse::<iroh_base::CustomAddr>().unwrap()),
        ],
    );
    let (relays, directs) = pf::split_ticket_addrs(&addr);
    assert_eq!(relays, vec!["relay.example.com".to_string()]);
    assert_eq!(directs, vec![direct]);

    // Hopeless: relay-only + hostname relay + all configured nameservers dead.
    let dead_ns: std::net::IpAddr = "9.9.9.9".parse().unwrap();
    let hopeless = pf::Report {
        relay_hosts: vec!["relay.example.com".into()],
        direct_addrs: vec![],
        nameservers: vec![(dead_ns, false)],
        relay_resolves: Some(false),
    };
    assert!(!hopeless.any_live_dns());
    assert!(!hopeless.relay_viable());
    assert!(hopeless.hopeless(), "no path at all → hard gate");

    // One live nameserver defuses the gate (and makes the relay viable).
    let live = pf::Report {
        nameservers: vec![(dead_ns, false), ("9.9.9.10".parse().unwrap(), true)],
        relay_resolves: Some(true),
        ..hopeless.clone()
    };
    assert!(live.any_live_dns());
    assert!(live.relay_viable());
    assert!(!live.hopeless());

    // An IP-literal relay host needs no DNS: never hopeless.
    let literal = pf::Report {
        relay_hosts: vec!["127.0.0.1".into()],
        ..hopeless.clone()
    };
    assert!(!literal.hopeless(), "IP-literal relay is a DNS-free path");

    // A direct address is always a possible path: never hopeless.
    let with_direct = pf::Report {
        direct_addrs: vec![direct],
        ..hopeless.clone()
    };
    assert!(!with_direct.hopeless());

    // Advice: every applicable cause is named, in order.
    let advice = pf::connect_failure_advice(&with_direct, true, true);
    assert!(
        advice.contains("STALE TICKET"),
        "direct addr → stale-ticket cause: {advice}"
    );
    assert!(advice.contains("192.168.7.9:7842"));
    assert!(
        advice.contains("SSH"),
        "macOS+SSH advisory folded in: {advice}"
    );
    assert!(
        advice.contains("9.9.9.9"),
        "dead nameserver named: {advice}"
    );
    assert!(
        advice.contains("blocking UDP"),
        "the always-on last resort: {advice}"
    );

    // A direct-only ticket instead points at the shared-network requirement.
    let advice2 = pf::connect_failure_advice(&pf::Report::default(), false, false);
    assert!(
        advice2.contains("no relay address"),
        "direct-only cause: {advice2}"
    );
    assert!(
        !advice2.contains("STALE TICKET"),
        "no direct addr → no stale cause: {advice2}"
    );
    assert!(
        !advice2.contains("Local Network"),
        "no macOS advisory off-macOS: {advice2}"
    );
}

/// `run()` on a ticket whose relay host is an IP LITERAL: DNS is declared
/// unnecessary (relay_resolves = true) and NO nameserver is ever probed — a
/// LAN-only ticket must never touch the resolver path.
#[tokio::test]
async fn preflight_run_skips_dns_entirely_for_ip_literal_relay() {
    let pk = SecretKey::generate().public();
    let direct: SocketAddr = "127.0.0.1:4444".parse().unwrap();
    let addr = EndpointAddr::from_parts(
        pk,
        [
            TransportAddr::Relay("http://127.0.0.1:3340".parse().unwrap()),
            TransportAddr::Ip(direct),
        ],
    );
    let report = pf::run(&addr, &pf::Style { enabled: false }).await;
    assert_eq!(report.relay_hosts, vec!["127.0.0.1".to_string()]);
    assert_eq!(report.direct_addrs, vec![direct]);
    assert_eq!(
        report.relay_resolves,
        Some(true),
        "IP-literal relay counts as resolvable without DNS"
    );
    assert!(
        report.nameservers.is_empty(),
        "no nameserver may be probed for an IP-literal relay: {:?}",
        report.nameservers
    );
}

// ── node/service.rs: cross-scope guards ──────────────────────────────────────

/// The agent install refuses while a `--system` daemon plist exists (two nodes,
/// one label) and the refusal carries the exact cleanup; a missing plist passes.
#[test]
fn service_agent_install_refuses_while_daemon_plist_exists() {
    let dir = tempfile::tempdir().expect("tempdir");
    let daemon = dir.path().join("com.higgs.node.plist");
    assert!(
        svc::refuse_daemon_conflict(&daemon).is_ok(),
        "no daemon plist → agent install may proceed"
    );

    std::fs::write(&daemon, "plist").expect("plant daemon plist");
    let err = svc::refuse_daemon_conflict(&daemon).expect_err("daemon present → refuse");
    let msg = err.to_string();
    assert!(
        msg.contains("ALONGSIDE"),
        "names the dual-run hazard: {msg}"
    );
    assert!(
        msg.contains("launchctl bootout \nsystem/com.higgs.node") || msg.contains("bootout"),
        "carries the bootout cleanup: {msg}"
    );
    assert!(
        msg.contains(daemon.to_str().unwrap()),
        "names the plist path: {msg}"
    );
}

/// The daemon (re)install refuses the unsupported BOTH-plists mixed state with
/// a copy-pasteable cleanup (paths single-quoted so a space survives as one
/// argument); a vanished agent HOME reads conservatively as PRESENT.
#[test]
fn service_daemon_install_refuses_mixed_state_and_treats_vanished_home_as_present() {
    let dir = tempfile::tempdir().expect("tempdir");
    let daemon = dir.path().join("com.higgs.node.plist");
    std::fs::write(&daemon, "plist").expect("plant daemon plist");

    // Agent plist genuinely absent under a LIVE root → clean daemon reinstall.
    let root = dir.path().join("op home");
    std::fs::create_dir_all(&root).expect("agent root");
    let agent = root.join("Library/LaunchAgents/com.higgs.node.plist");
    assert!(
        svc::refuse_cross_scope_coexistence(&daemon, &agent, &root).is_ok(),
        "leaf-absent agent under a live home is a clean state"
    );

    // Both present → refuse, with both paths named and the agent path QUOTED
    // (its home has a space).
    std::fs::create_dir_all(agent.parent().unwrap()).expect("agent dir");
    std::fs::write(&agent, "plist").expect("plant agent plist");
    let err = svc::refuse_cross_scope_coexistence(&daemon, &agent, &root)
        .expect_err("both plists → unsupported mixed state");
    let msg = err.to_string();
    assert!(msg.contains("TWO nodes"), "names the dual-run risk: {msg}");
    assert!(
        msg.contains(&format!("'{}'", agent.display())),
        "agent path is single-quoted for copy-paste (space-safe): {msg}"
    );

    // Agent leaf ENOENT while its trusted root is GONE → conservative PRESENT
    // (the coexistence guard must not fall open on a vanished home).
    let gone_root = dir.path().join("vanished home");
    let gone_agent = gone_root.join("Library/LaunchAgents/com.higgs.node.plist");
    assert!(
        svc::refuse_cross_scope_coexistence(&daemon, &gone_agent, &gone_root).is_err(),
        "vanished agent home reads as PRESENT → still refuse"
    );
}

// ── node/service.rs: headless hint + linger note ─────────────────────────────

/// Only the gui-domain agent gets the headless-Mac remedy hint; the daemon and
/// systemd unit need no session, so their hint is empty.
#[test]
fn service_headless_hint_only_for_the_gui_agent() {
    let hint = svc::agent_headless_hint(svc::ServiceKind::LaunchdAgent);
    assert!(
        hint.contains("--system"),
        "points at the daemon alternative: {hint}"
    );
    assert_eq!(svc::agent_headless_hint(svc::ServiceKind::Launchd), "");
    assert_eq!(svc::agent_headless_hint(svc::ServiceKind::SystemdUser), "");
}

/// The linger note surfaces only when the flag file exists for the operator,
/// and names the exact opt-out command (linger is never auto-disabled).
#[test]
fn service_linger_note_reflects_flag_file_presence() {
    let dir = tempfile::tempdir().expect("tempdir");
    assert_eq!(
        svc::linger_note(dir.path(), "alice"),
        None,
        "no flag file → no note"
    );
    std::fs::write(dir.path().join("alice"), "").expect("plant linger flag");
    let note = svc::linger_note(dir.path(), "alice").expect("linger enabled → note");
    assert!(
        note.contains("disable-linger alice"),
        "note carries the opt-out command: {note}"
    );
}

// ── node/service.rs: systemd unit rendering + escaping ───────────────────────

/// The rendered systemd user unit pins HOME/HIGGS_HOME/model-dir/preserved env,
/// unsets every UNPINNED app-config var, and escapes each context correctly:
/// `$` doubled ONLY in ExecStart (Environment= takes it literally), `%` doubled
/// everywhere, quotes/backslashes C-escaped in quoted values.
#[test]
fn service_systemd_unit_pins_env_and_escapes_percent_dollar_quote_per_context() {
    let unit = svc::systemd_unit(
        Path::new("/home/op%h$d\"q\\b"),
        Path::new("/opt/hig gs%p$x"),
        Path::new("/home/op/.higgs"),
        Some(Path::new("/mnt/models dir")),
        &[(
            "HIGGS_HF_ENDPOINT".to_string(),
            "http://mirror:8080/x%y".to_string(),
        )],
    );

    // ExecStart word: C-escaped + % doubled + $ doubled (systemd expands $VAR
    // in ExecStart even inside double quotes).
    assert!(
        unit.contains("ExecStart=\"/opt/hig gs%%p$$x/bin/current/higgs\" --node"),
        "ExecStart escaping ($$, %%): {unit}"
    );
    // Environment= value: C-escaped + % doubled, but `$` stays SINGLE (systemd
    // does not expand $VAR in Environment=; doubling would corrupt the path).
    assert!(
        unit.contains("Environment=\"HOME=/home/op%%h$d\\\"q\\\\b\""),
        "Environment escaping (%% but single $, C-escaped quote/backslash): {unit}"
    );
    assert!(unit.contains("Environment=\"HIGGS_HOME=/home/op/.higgs\""));
    assert!(unit.contains("Environment=\"HIGGS_MODEL_DIR=/mnt/models dir\""));
    assert!(unit.contains("Environment=\"HIGGS_HF_ENDPOINT=http://mirror:8080/x%%y\""));
    // Unpinned app-config vars are explicitly unset (model dir + HF endpoint
    // are pinned here, so only the rest appear) — a stale manager var must not
    // leak into the daemon.
    assert!(
        unit.contains("UnsetEnvironment=HIGGS_ENGINE HIGGS_IROH_LOCAL"),
        "unpinned vars are unset explicitly: {unit}"
    );
    // append: is a bare value — % doubled, nothing else touched, NOT quoted.
    assert!(
        unit.contains("StandardOutput=append:/opt/hig gs%%p$x/logs/node.log"),
        "append: escaping (%% only, unquoted): {unit}"
    );
    // Crash-loop policy: always restart, no start-rate limit.
    assert!(unit.contains("StartLimitIntervalSec=0"));
    assert!(unit.contains("Restart=always"));
    assert!(unit.contains("Type=exec"));

    // Nothing pinned: EVERY app-config var lands in UnsetEnvironment.
    let bare = svc::systemd_unit(
        Path::new("/home/op"),
        Path::new("/opt/higgs"),
        Path::new("/home/op/.higgs"),
        None,
        &[],
    );
    assert!(
        bare.contains(
            "UnsetEnvironment=HIGGS_MODEL_DIR HIGGS_HF_ENDPOINT HIGGS_ENGINE HIGGS_IROH_LOCAL"
        ),
        "bare install unsets every app-config var: {bare}"
    );
    assert!(
        !bare.contains("HIGGS_MODEL_DIR="),
        "no model-dir pin when unset"
    );
}

// ── load_robustness.rs: OOM classification + degrade ladder ──────────────────

/// Only allocator-signature failures are OOM (retryable); anything else must
/// fail immediately (a degraded retry of a corrupt GGUF just wastes seconds).
#[test]
fn load_robustness_oom_classification_matches_allocator_signatures_only() {
    assert!(is_oom_reason(
        "ggml_backend_metal: FAILED TO ALLOCATE buffer"
    ));
    assert!(is_oom_reason("CUDA error: out of memory"));
    assert!(is_oom_reason("mtlbuffer allocation exceeded working set"));
    assert!(!is_oom_reason("unknown architecture 'quux' in GGUF header"));
    assert!(!is_oom_reason("tensor 'blk.0.attn' has invalid shape"));
}

/// The full ladder for an explicit all-GPU load: plain settle-retry, then KV
/// cache off the GPU, then KV-off PLUS a concrete layer reduction — halved via
/// the model's known layer total, or ALL-CPU when the total is unknown.
#[test]
fn load_robustness_ladder_settle_then_kv_off_then_cumulative_layer_reduction() {
    let base = NodeLoadParams {
        id: "m".into(),
        ctx_len: None,
        gpu_layers: Some(GpuLayers::All),
        threads: None,
        params: None,
    };

    // Known layer total → the layer rung halves it (cumulative on KV-off).
    let rungs = oom_ladder(&base, Some(24));
    assert_eq!(rungs.len(), 3, "settle + kv-off + layer rung: {rungs:?}");
    assert_eq!(rungs[0].params, base, "rung 1 is the plain settle-retry");
    assert_eq!(rungs[0].what, "");
    assert!(
        rungs[1].what.contains("KV cache"),
        "rung 2 names the KV move"
    );
    assert_eq!(
        rungs[1].params.params.as_ref().unwrap().offload_kqv,
        Some(false),
        "rung 2 forces offload_kqv=false"
    );
    assert_eq!(
        rungs[2].params.gpu_layers,
        Some(GpuLayers::Count { n: 12 }),
        "rung 3 halves the known 24-layer total"
    );
    assert!(
        rungs[2].what.contains("AND halved"),
        "rung 3 names BOTH cumulative degradations: {}",
        rungs[2].what
    );
    assert_eq!(
        rungs[2].params.params.as_ref().unwrap().offload_kqv,
        Some(false),
        "rung 3 KEEPS rung 2's KV-off (cumulative)"
    );

    // Unknown layer total → the deterministic last resort is ALL-CPU.
    let rungs = oom_ladder(&base, None);
    assert_eq!(rungs[2].params.gpu_layers, Some(GpuLayers::Count { n: 0 }));
    assert!(
        rungs[2].what.contains("all layers to the CPU"),
        "{}",
        rungs[2].what
    );

    // A fixed count halves directly (no layer_count needed).
    let counted = NodeLoadParams {
        gpu_layers: Some(GpuLayers::Count { n: 8 }),
        ..base.clone()
    };
    let rungs = oom_ladder(&counted, None);
    assert_eq!(rungs[2].params.gpu_layers, Some(GpuLayers::Count { n: 4 }));
}

/// Rungs that would change nothing are SKIPPED: a CPU-only load gets no KV/layer
/// rung (its KV cache is already in system memory), a 1-layer count has nothing
/// to halve, and a base that already carries `offload_kqv=false` skips the KV
/// rung — and the layer rung's phrase must not claim the KV cache moved.
#[test]
fn load_robustness_ladder_skips_noop_rungs_and_keeps_event_phrases_honest() {
    let base = NodeLoadParams {
        id: "m".into(),
        ctx_len: None,
        gpu_layers: None,
        threads: None,
        params: None,
    };

    // CPU-only: only the plain settle-retry remains.
    let cpu_only = NodeLoadParams {
        gpu_layers: Some(GpuLayers::Count { n: 0 }),
        ..base.clone()
    };
    let rungs = oom_ladder(&cpu_only, Some(24));
    assert_eq!(
        rungs.len(),
        1,
        "CPU-only load has nothing to degrade: {rungs:?}"
    );

    // A 1-layer fixed count: KV rung applies, layer rung does not (0-1 minimal).
    let one = NodeLoadParams {
        gpu_layers: Some(GpuLayers::Count { n: 1 }),
        ..base.clone()
    };
    let rungs = oom_ladder(&one, Some(24));
    assert_eq!(rungs.len(), 2, "settle + kv-off only: {rungs:?}");

    // Base ALREADY kv-off (a reloaded previously-degraded profile): the KV rung
    // is a byte-identical duplicate → skipped; the layer rung's phrase must say
    // the KV cache was ALREADY in system memory (the [HG061] event must not lie).
    let kv_already_off = NodeLoadParams {
        gpu_layers: Some(GpuLayers::Count { n: 8 }),
        params: Some(LlamaCppParams {
            offload_kqv: Some(false),
            ..LlamaCppParams::default()
        }),
        ..base.clone()
    };
    let rungs = oom_ladder(&kv_already_off, Some(24));
    assert_eq!(
        rungs.len(),
        2,
        "no separate KV rung when already off: {rungs:?}"
    );
    assert_eq!(
        rungs[1].params.gpu_layers,
        Some(GpuLayers::Count { n: 4 }),
        "the layer rung still halves the explicit count"
    );
    assert!(
        rungs[1].what.contains("KV cache already in system memory"),
        "phrase stays honest about what THIS walk degraded: {}",
        rungs[1].what
    );
}
