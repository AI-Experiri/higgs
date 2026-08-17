//! REL-P4e integration: the HUB release courier (`Higgs::fleet_update` / `Higgs::node_update`)
//! pushes a signature-verified self-update to a REAL spawned `higgs --node`.
//!
//! Flow: build a hub `Higgs` with a `HubFleet` → spawn a real node → pair over hermetic iroh →
//! stand up a LOCAL HTTP server serving the CI-shaped `higgs-v<ver>-<suffix>.{manifest,minisig,
//! tar.gz}` → call `fleet_update` (whole-fleet) and `node_update` (one node by exact URL) →
//! assert the node replies `accepted`.
//!
//! The node ACKs `{status:"accepted", target_version}` BEFORE it verifies + applies (a detached,
//! minutes-long job). Production verify uses the pubkeys COMPILED INTO the node, which a test
//! cannot satisfy, AND this spawned node is NOT a managed `bin/v<ver>/current` install — so the
//! detached apply fails fast (logged; the node stays up). This test therefore asserts the SENDER
//! contract: the courier resolves the per-node asset, fetches the manifest+sig, derives the
//! artifact URL, and pushes it — and the node ACCEPTS the well-formed push. The eventual verify
//! FAILURE surfacing via the node's next HELLO `update_failed` needs a managed-install layout (a
//! DEFERRED, separately-tracked item) and is not exercised here.
//!
//! Fail-on-revert anchor: reverting `HubFleet::push_update` (so nothing reaches the node) makes
//! both the `fleet_update` result and the `node_update` reply stop being `accepted`.
//!
//! No GGUF is needed (the node need not load a model to receive `M_NODE_UPDATE`), so this test
//! does NOT skip — it always runs.

use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use iroh_tickets::endpoint::EndpointTicket;

use higgs::auth::{Allowlist, PairingTokens};
use higgs::log_bus::LogBus;
use higgs::node::fleet::HubFleet;
use higgs::node::fleet::PinnedPush;
use higgs::node::release_courier::{self, asset_suffix};
use higgs::node::self_update::BuildIdentity;
use higgs::node::transport::NodeTransport;
use higgs::node::{gate_connection, GateOutcome, HubIdentity, HELLO_DEADLINE};
use higgs::remote::ALPN;
use higgs::{Higgs, HiggsConfig};

struct NodeProc(Child);
impl Drop for NodeProc {
    fn drop(&mut self) {
        // SIGTERM: graceful node shutdown flushes its llvm-cov profile (a hard kill drops it).
        unsafe {
            libc::kill(self.0.id() as libc::pid_t, libc::SIGTERM);
        }
        let _ = self.0.wait();
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

async fn hub_endpoint() -> iroh::Endpoint {
    iroh::Endpoint::builder(iroh::endpoint::presets::Minimal)
        .alpns(vec![ALPN.to_vec()])
        .relay_mode(iroh::RelayMode::Disabled)
        .bind()
        .await
        .expect("bind hub endpoint")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn hub_courier_pushes_release_and_node_accepts() {
    let node_home = tempfile::tempdir().expect("node home");

    // The build the node WILL report (it is this crate) — drives the per-node asset name.
    let build = BuildIdentity::current();
    let ver = "9.9.9"; // a plausible NEWER release; the base URL carries the tag `v9.9.9`.
    let suffix = asset_suffix(&build.target, &build.variant);
    let manifest_name = format!("higgs-v{ver}-{suffix}.manifest");
    let tarball_name = format!("higgs-v{ver}-{suffix}.tar.gz");
    let base_path = format!("/releases/download/v{ver}");

    // A LOCAL "release server" serving the CI-shaped trio. The manifest is a VALID
    // `UpdateManifest` (schema 1) so the courier's untrusted parse succeeds; the .minisig is any
    // text (the courier never verifies — the node does, against its compiled-in pins); the tarball
    // is never fetched here (this non-managed node's apply aborts before the download).
    let manifest_json = serde_json::json!({
        "schema": 1,
        "version": ver,
        "commit": "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
        "file": tarball_name,
        "target": build.target,
        "variant": build.variant,
        "sha256": "0000000000000000000000000000000000000000000000000000000000000000",
    })
    .to_string();
    let sig_text =
        "untrusted comment: signature from minisign secret key\nRWTestNotARealSignature==\n"
            .to_string();
    let tarball_bytes = vec![0u8; 64];

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind release server");
    let port = listener.local_addr().unwrap().port();
    let app = axum::Router::new()
        .route(
            &format!("{base_path}/{manifest_name}"),
            axum::routing::get(move || {
                let m = manifest_json.clone();
                async move { m }
            }),
        )
        .route(
            &format!("{base_path}/{manifest_name}.minisig"),
            axum::routing::get(move || {
                let s = sig_text.clone();
                async move { s }
            }),
        )
        .route(
            &format!("{base_path}/{tarball_name}"),
            axum::routing::get(move || {
                let t = tarball_bytes.clone();
                async move { t }
            }),
        );
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    // Hub: a `Higgs` with a `HubFleet` installed (the facade's `self.fleet()`).
    let bus = Arc::new(LogBus::new());
    let higgs = Arc::new(Higgs::with_log_bus(HiggsConfig::default(), bus.clone()));
    let fleet = Arc::new(HubFleet::new(bus.clone()));
    higgs.set_fleet(fleet.clone());

    let hub = hub_endpoint().await;
    let hub_id = hub.id().to_string();
    let ticket = EndpointTicket::new(hub.addr()).to_string();
    let mut allow = Allowlist::load(&node_home.path().join("hub-pairings.json")).unwrap();
    let mut tokens = PairingTokens::new();
    let token = tokens.mint(now_ms(), 600_000);

    let child = Command::new(env!("CARGO_BIN_EXE_higgs"))
        .arg("--node")
        .arg(&ticket)
        .arg(&token)
        .env("HIGGS_HOME", node_home.path())
        .env("HIGGS_IROH_LOCAL", "1")
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("spawn higgs --node");
    let _node = NodeProc(child);

    // Admit the node, capturing the build identity + `update` capability it reported in HELLO.
    let incoming = tokio::time::timeout(Duration::from_secs(30), hub.accept())
        .await
        .expect("node dialed within 30s")
        .expect("incoming");
    let conn = incoming.await.expect("connection");
    let peer = conn.remote_id().to_string();
    let outcome = gate_connection(
        &conn,
        &mut allow,
        &mut tokens,
        now_ms(),
        &HubIdentity::new(hub_id),
        Some("test".into()),
        HELLO_DEADLINE,
    )
    .await;
    let GateOutcome::Admitted {
        agreed_version,
        software_version,
        fleet_events,
        update_failed,
        reports_update_failures,
        target,
        variant,
        update_capable,
        version_capable,
        log_capable: _,
        pull_capable,
        downloads,
    } = outcome
    else {
        panic!("node must be admitted: {outcome:?}");
    };
    // The node reports its compiled-in build identity + the `update` capability (REL-P4e wire).
    assert_eq!(
        target.as_deref(),
        Some(build.target.as_str()),
        "node reports its build target in HELLO"
    );
    assert_eq!(
        variant.as_deref(),
        Some(build.variant.as_str()),
        "node reports its build variant in HELLO"
    );
    assert!(
        update_capable,
        "a current node advertises the `update` capability"
    );
    assert!(
        version_capable,
        "a current node advertises the `update_by_version` capability"
    );

    // Register the node + its build identity into the fleet exactly as the production accept loop
    // does — one atomic `add_node_with_identity`. Keep a second handle on the connection so we can
    // forge a STALE transport for the reconnect-CAS check below (Fix 4).
    let conn_for_stale = conn.clone();
    let transport = Arc::new(NodeTransport::new(conn));
    fleet
        .add_node_with_identity(
            peer.clone(),
            transport.clone(),
            None,
            Some(agreed_version),
            Some(software_version),
            fleet_events,
            update_failed,
            reports_update_failures,
            target,
            variant,
            update_capable,
            version_capable,
            true,
            pull_capable,
            downloads,
        )
        .await;

    let base_url = format!("http://127.0.0.1:{port}{base_path}");
    let manifest_url = format!("http://127.0.0.1:{port}{base_path}/{manifest_name}");

    // ── Fix 3: the FACADE gates on the HUB being ENABLED, not on a fleet existing. `higgs` has a
    // fleet but no hub, so both facades must fail `not_a_hub` (before any outbound fetch). ──
    let e = higgs
        .fleet_update(&base_url)
        .await
        .expect_err("fleet_update must refuse when the hub is disabled");
    assert!(
        matches!(e, higgs::diagnostic::HiggsError::HubControlFailed { .. }),
        "hub-disabled fleet_update is not_a_hub: {e:?}"
    );
    let e = higgs
        .node_update(&peer, &manifest_url)
        .await
        .expect_err("node_update must refuse when the hub is disabled");
    assert!(
        matches!(e, higgs::diagnostic::HiggsError::HubControlFailed { .. }),
        "hub-disabled node_update is not_a_hub: {e:?}"
    );

    // ── Whole-fleet push (courier directly, since the test fleet has no full Hub installed): the
    // courier derives the per-node asset from the base + node identity, binds the version, pushes. ──
    let report = release_courier::fleet_update(&fleet, &base_url)
        .await
        .expect("fleet_update returns a report");
    let results = report["results"]
        .as_array()
        .unwrap_or_else(|| panic!("results array in {report}"));
    assert_eq!(
        results.len(),
        1,
        "one connected, update-capable node: {report}"
    );
    assert_eq!(results[0]["node"], peer, "the report names the node");
    assert_eq!(
        results[0]["status"], "accepted",
        "the node accepted the pushed update: {}",
        results[0]
    );
    assert_eq!(results[0]["reply"]["status"], "accepted");
    assert_eq!(
        results[0]["reply"]["target_version"], ver,
        "the node echoes the manifest version it was pushed"
    );

    // ── Direct push to ONE node by its EXACT manifest URL ──
    let reply = release_courier::node_update(&fleet, &peer, &manifest_url)
        .await
        .expect("node_update returns the node's reply");
    assert_eq!(reply["status"], "accepted", "direct push accepted: {reply}");
    assert_eq!(reply["target_version"], ver);

    // ── Fix 4: a transport-PINNED push refuses to send when the pinning transport is no longer the
    // node's current one (a reconnect). A SECOND `NodeTransport` over the same connection is a
    // DIFFERENT Arc than the one the fleet holds, so the CAS reports `Reconnected` — never a bogus
    // "accepted" for a possibly-wrong asset. ──
    let stale = Arc::new(NodeTransport::new(conn_for_stale));
    let params = release_courier::resolve_push_params(&manifest_url, None)
        .await
        .expect("resolve params for the pinned-push check");
    let pinned = fleet
        .push_update_pinned(&peer, &stale, params)
        .await
        .expect("push_update_pinned resolves");
    assert!(
        matches!(pinned, PinnedPush::Reconnected),
        "a stale (non-current) transport must be reported Reconnected, not Accepted"
    );

    // ── UPx: the VERSION-ONLY trigger. The hub sends a bare semver; the node ACKs and
    // self-fetches from ITS OWN configured release_url (written into its config.json before
    // spawn). Fail-on-revert: without the node's M_NODE_UPDATE_VERSION handler this RPC is
    // method-not-found and the courier errors. ──
    let reply = release_courier::node_update_version(&fleet, &peer, ver)
        .await
        .expect("version-only trigger accepted");
    assert_eq!(
        reply["status"], "accepted",
        "version trigger ACKed: {reply}"
    );
    assert_eq!(reply["target_version"], ver);

    // A malformed version is refused HUB-SIDE before any push.
    let e = release_courier::node_update_version(&fleet, &peer, "not-a-version")
        .await
        .expect_err("non-semver version refused");
    assert!(e.to_string().contains("semver"), "{e}");

    // ── UPx: the release LISTING is a github.com-shape feature — the loopback test base is
    // refused with the actionable message, BEFORE any fetch. ──
    let e = release_courier::node_releases(&fleet, &peer, &base_url)
        .await
        .expect_err("listing from a non-github release_url is refused");
    assert!(e.to_string().contains("github.com"), "{e}");
}

/// `fleet_update` REPORTS a node it cannot target rather than failing the whole fleet: a connected
/// node that never advertised the `update` capability is SKIPPED (with a reason). Driven by
/// admitting the node into the fleet WITHOUT `set_build_identity` marking it update-capable.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn fleet_update_skips_a_non_update_capable_node() {
    let node_home = tempfile::tempdir().expect("node home");

    let bus = Arc::new(LogBus::new());
    let higgs = Arc::new(Higgs::with_log_bus(HiggsConfig::default(), bus.clone()));
    let fleet = Arc::new(HubFleet::new(bus.clone()));
    higgs.set_fleet(fleet.clone());

    let hub = hub_endpoint().await;
    let hub_id = hub.id().to_string();
    let ticket = EndpointTicket::new(hub.addr()).to_string();
    let mut allow = Allowlist::load(&node_home.path().join("hub-pairings.json")).unwrap();
    let mut tokens = PairingTokens::new();
    let token = tokens.mint(now_ms(), 600_000);

    let child = Command::new(env!("CARGO_BIN_EXE_higgs"))
        .arg("--node")
        .arg(&ticket)
        .arg(&token)
        .env("HIGGS_HOME", node_home.path())
        .env("HIGGS_IROH_LOCAL", "1")
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("spawn higgs --node");
    let _node = NodeProc(child);

    let incoming = tokio::time::timeout(Duration::from_secs(30), hub.accept())
        .await
        .expect("node dialed within 30s")
        .expect("incoming");
    let conn = incoming.await.expect("connection");
    let peer = conn.remote_id().to_string();
    let outcome = gate_connection(
        &conn,
        &mut allow,
        &mut tokens,
        now_ms(),
        &HubIdentity::new(hub_id),
        Some("test".into()),
        HELLO_DEADLINE,
    )
    .await;
    assert!(
        matches!(outcome, GateOutcome::Admitted { .. }),
        "{outcome:?}"
    );

    // Admit into the fleet but mark it NOT update-capable (a legacy/dev node's state): plain
    // `add_node` admits with no build identity, so target/variant/update-capable stay absent.
    let transport = Arc::new(NodeTransport::new(conn));
    fleet
        .add_node(
            peer.clone(),
            transport.clone(),
            None,
            Some(2),
            Some(env!("CARGO_PKG_VERSION").to_string()),
            false,
            None,
            false,
        )
        .await;

    // Courier directly (the test fleet has no full Hub installed). Use a LITERAL loopback base:
    // `fleet_update` builds the manifest client (resolve+pin the origin) up front, so a public host
    // like `example.com` would need DNS and break on an offline/DNS-blocked runner (codex r7 #2). A
    // literal loopback IP needs no DNS, and the sole node is SKIPPED (non-capable) before any fetch,
    // so port 9 is never connected.
    let report =
        release_courier::fleet_update(&fleet, "http://127.0.0.1:9/releases/download/v9.9.9")
            .await
            .expect("fleet_update returns a report even with an unreachable release host");
    let results = report["results"].as_array().expect("results array");
    assert_eq!(results.len(), 1, "the connected node is reported: {report}");
    assert_eq!(
        results[0]["status"], "skipped",
        "non-capable node skipped: {}",
        results[0]
    );
    // It never reached the release host (loopback port 9, nothing listening) — it was pre-filtered.
    // UPx: the version-only trigger is refused for a node WITHOUT the
    // `update_by_version` capability — with the actionable bootstrap hint, and
    // BEFORE any push. Fail-on-revert for the hub-side capability gate.
    let peer = results[0]["node"].as_str().expect("node key").to_string();
    let e = release_courier::node_update_version(&fleet, &peer, "9.9.9")
        .await
        .expect_err("version trigger refused for a non-capable node");
    assert!(e.to_string().contains("installer"), "{e}");
    // And the fleet-wide version trigger SKIPS it (never accepted-then-failed).
    let vreport =
        release_courier::fleet_update_version(&fleet, "9.9.9", "https://github.com/o/r/releases")
            .await
            .expect("fleet_update_version returns a report");
    assert_eq!(vreport["results"][0]["status"], "skipped", "{vreport}");

    assert!(
        results[0]["reason"]
            .as_str()
            .unwrap_or_default()
            .contains("update"),
        "reason names the missing capability: {}",
        results[0]
    );
}
