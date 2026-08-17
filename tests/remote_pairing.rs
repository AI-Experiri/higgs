//! P1 integration: pairing + HELLO handshake over a local (relay-disabled) iroh link.
//!
//! Two endpoints run in one process; the node dials the hub by its explicit
//! `EndpointAddr` (direct addrs from `addr()`), so no relay/discovery infra is needed.

use std::time::Duration;

use higgs::auth::{Allowlist, PairingTokens};
use higgs::node::{dial_and_hello, gate_connection, GateOutcome, HubIdentity, HELLO_DEADLINE};
use higgs::remote::{HelloResult, ALPN};
use higgs::rpc::{self, RpcFrame, RpcResponse};

/// Bind a local-only endpoint (relay disabled) for in-process testing.
async fn local_endpoint() -> iroh::Endpoint {
    iroh::Endpoint::builder(iroh::endpoint::presets::Minimal)
        .alpns(vec![ALPN.to_vec()])
        .relay_mode(iroh::RelayMode::Disabled)
        .bind()
        .await
        .expect("bind local endpoint")
}

/// A fresh, empty allowlist backed by a unique temp file.
fn temp_allowlist(tag: &str) -> Allowlist {
    let path = std::env::temp_dir().join(format!("higgs-p1-{tag}-{}.json", std::process::id()));
    let _ = std::fs::remove_file(&path);
    Allowlist::load(&path).unwrap()
}

#[tokio::test]
async fn valid_token_pairs() {
    let hub = local_endpoint().await;
    let node = local_endpoint().await;
    let hub_addr = hub.addr();
    let hub_id = hub.id().to_string();

    let mut allow = temp_allowlist("pair-ok");
    let mut tokens = PairingTokens::new();
    let tok = tokens.mint(1_000, 600_000);

    let hub_task = tokio::spawn(async move {
        let incoming = hub.accept().await.expect("incoming");
        let conn = incoming.await.expect("conn");
        let out = gate_connection(
            &conn,
            &mut allow,
            &mut tokens,
            2_000,
            &HubIdentity::new(hub_id),
            Some("studio".into()),
            HELLO_DEADLINE,
        )
        .await;
        // keep conn alive until the node has read its reply
        tokio::time::sleep(Duration::from_millis(100)).await;
        (out, allow.contains(&conn.remote_id().to_string()))
    });

    let node_id = node.id().to_string();
    let res = dial_and_hello(&node, hub_addr, node_id, String::new(), Some(tok)).await;
    assert!(res.is_ok(), "valid token should pair: {res:?}");
    assert_eq!(res.unwrap().agreed_version, 2);

    let (outcome, now_paired) = hub_task.await.unwrap();
    assert_eq!(
        outcome,
        GateOutcome::Admitted {
            agreed_version: 2,
            software_version: env!("CARGO_PKG_VERSION").to_string(),
            // A current node HELLO advertises the fleet_events push (T10).
            fleet_events: true,
            update_failed: None,
            // The in-process test node is NOT a managed install (self_update_bin_dir is None), so
            // it does not advertise `update_reporting` — the P4b (d) capability gate at work.
            reports_update_failures: false,
            // A current node reports its compiled-in build identity (REL-P4e) and advertises the
            // `update` capability — this test node IS this crate, so they match exactly.
            target: Some(higgs::node::self_update::BuildIdentity::current().target),
            variant: Some(higgs::node::self_update::BuildIdentity::current().variant),
            update_capable: true,
            version_capable: true,
            log_capable: true,
            // A current node reports its in-flight downloads on demand (DL slice).
            pull_capable: true,
            // The in-process test node has no download in flight to announce.
            downloads: vec![],
        }
    );
    assert!(now_paired, "node added to allowlist after pairing");
}

#[tokio::test]
async fn stranger_without_token_is_rejected_hg024() {
    let hub = local_endpoint().await;
    let node = local_endpoint().await;
    let hub_addr = hub.addr();
    let hub_id = hub.id().to_string();

    let mut allow = temp_allowlist("stranger");
    let mut tokens = PairingTokens::new();

    let hub_task = tokio::spawn(async move {
        let incoming = hub.accept().await.expect("incoming");
        let conn = incoming.await.expect("conn");
        gate_connection(
            &conn,
            &mut allow,
            &mut tokens,
            2_000,
            &HubIdentity::new(hub_id),
            None,
            HELLO_DEADLINE,
        )
        .await
    });

    let node_id = node.id().to_string();
    // No token, empty allowlist → hub rejects; the node's read sees the closed stream.
    let res = dial_and_hello(&node, hub_addr, node_id, String::new(), None).await;
    assert!(res.is_err(), "stranger must be rejected");

    let outcome = hub_task.await.unwrap();
    assert_eq!(outcome, GateOutcome::Rejected { code: "HG024" });
}

#[tokio::test]
async fn silent_peer_is_dropped_hg028() {
    let hub = local_endpoint().await;
    let node = local_endpoint().await;
    let hub_addr = hub.addr();
    let hub_id = hub.id().to_string();

    let mut allow = temp_allowlist("stalled");
    let mut tokens = PairingTokens::new();
    let short = Duration::from_millis(200);

    let hub_task = tokio::spawn(async move {
        let incoming = hub.accept().await.expect("incoming");
        let conn = incoming.await.expect("conn");
        gate_connection(
            &conn,
            &mut allow,
            &mut tokens,
            2_000,
            &HubIdentity::new(hub_id),
            None,
            short,
        )
        .await
    });

    // Node connects and opens the control stream but NEVER writes HELLO.
    let conn = node.connect(hub_addr, ALPN).await.expect("connect");
    let (_send, _recv) = conn.open_bi().await.expect("open_bi");
    // Hold the connection open, silent, past the deadline.
    tokio::time::sleep(Duration::from_millis(500)).await;

    let outcome = hub_task.await.unwrap();
    assert_eq!(outcome, GateOutcome::Rejected { code: "HG028" });
}

#[tokio::test]
async fn spoofed_node_id_is_rejected() {
    // A peer that claims a node_id different from its TLS-authenticated remote_id is
    // rejected even with a valid token (§4.1 identity check).
    let hub = local_endpoint().await;
    let node = local_endpoint().await;
    let hub_addr = hub.addr();
    let hub_id = hub.id().to_string();

    let mut allow = temp_allowlist("spoof");
    let mut tokens = PairingTokens::new();
    let tok = tokens.mint(1_000, 600_000);

    let hub_task = tokio::spawn(async move {
        let incoming = hub.accept().await.expect("incoming");
        let conn = incoming.await.expect("conn");
        gate_connection(
            &conn,
            &mut allow,
            &mut tokens,
            2_000,
            &HubIdentity::new(hub_id),
            None,
            HELLO_DEADLINE,
        )
        .await
    });

    // Lie about our identity: pass a bogus self_id that won't match remote_id().
    let res = dial_and_hello(
        &node,
        hub_addr,
        "deadbeef-not-my-id".into(),
        String::new(),
        Some(tok),
    )
    .await;
    assert!(res.is_err(), "spoofed node_id must be rejected");
    assert_eq!(
        hub_task.await.unwrap(),
        GateOutcome::Rejected { code: "HG024" }
    );
}

#[tokio::test]
async fn hello_exchanges_friendly_names() {
    // A node sends its friendly name in HELLO; the hub (a) stores it as the new pairing's
    // allowlist label and returns it as `assigned_label`, and (b) sends its OWN name back as
    // `hub_name` so the node can save the hub under a human label (Unit B).
    let hub = local_endpoint().await;
    let node = local_endpoint().await;
    let hub_addr = hub.addr();
    let hub_id = hub.id().to_string();

    let mut allow = temp_allowlist("names");
    let mut tokens = PairingTokens::new();
    let tok = tokens.mint(1_000, 600_000);

    let hub_task = tokio::spawn(async move {
        let incoming = hub.accept().await.expect("incoming");
        let conn = incoming.await.expect("conn");
        let identity = HubIdentity {
            id: hub_id,
            name: "hub-friendly(srv)".into(),
        };
        let out = gate_connection(
            &conn,
            &mut allow,
            &mut tokens,
            2_000,
            &identity,
            Some("fallback".into()),
            HELLO_DEADLINE,
        )
        .await;
        tokio::time::sleep(Duration::from_millis(100)).await;
        // The node's HELLO name became its allowlist label (NOT the "fallback").
        let label = allow.label(&conn.remote_id().to_string());
        (out, label)
    });

    let node_id = node.id().to_string();
    let res = dial_and_hello(
        &node,
        hub_addr,
        node_id,
        "node-friendly(box)".into(),
        Some(tok),
    )
    .await
    .expect("admitted");
    assert_eq!(
        res.hub_name, "hub-friendly(srv)",
        "hub name returned to node"
    );
    assert_eq!(
        res.assigned_label.as_deref(),
        Some("node-friendly(box)"),
        "node's own name is its assigned label"
    );

    let (outcome, label) = hub_task.await.unwrap();
    assert_eq!(
        outcome,
        GateOutcome::Admitted {
            agreed_version: 2,
            software_version: env!("CARGO_PKG_VERSION").to_string(),
            // A current node HELLO advertises the fleet_events push (T10).
            fleet_events: true,
            update_failed: None,
            // The in-process test node is NOT a managed install (self_update_bin_dir is None), so
            // it does not advertise `update_reporting` — the P4b (d) capability gate at work.
            reports_update_failures: false,
            // A current node reports its compiled-in build identity (REL-P4e) and advertises the
            // `update` capability — this test node IS this crate, so they match exactly.
            target: Some(higgs::node::self_update::BuildIdentity::current().target),
            variant: Some(higgs::node::self_update::BuildIdentity::current().variant),
            update_capable: true,
            version_capable: true,
            log_capable: true,
            // A current node reports its in-flight downloads on demand (DL slice).
            pull_capable: true,
            // The in-process test node has no download in flight to announce.
            downloads: vec![],
        }
    );
    assert_eq!(
        label.as_deref(),
        Some("node-friendly(box)"),
        "hub persisted the node's friendly name as its allowlist label"
    );
}

#[tokio::test]
async fn allowlisted_node_reconnects_without_token() {
    // A paired node reconnects on pure allowlist membership — no token needed.
    // (The HG023 version-mismatch reject path can't be produced in-process since both
    // endpoints are the same binary speaking [1]; it is covered by the `negotiate_*`
    // unit tests in `remote.rs`.)
    let hub = local_endpoint().await;
    let node = local_endpoint().await;
    let hub_addr = hub.addr();
    let hub_id = hub.id().to_string();
    let node_id = node.id().to_string();

    let path = std::env::temp_dir().join(format!("higgs-p1-preallow-{}.json", std::process::id()));
    let _ = std::fs::remove_file(&path);
    let mut allow = Allowlist::load(&path).unwrap();
    allow.add(node_id.clone(), Some("known".into())).unwrap();
    let mut tokens = PairingTokens::new();

    let hub_task = tokio::spawn(async move {
        let incoming = hub.accept().await.expect("incoming");
        let conn = incoming.await.expect("conn");
        let out = gate_connection(
            &conn,
            &mut allow,
            &mut tokens,
            2_000,
            &HubIdentity::new(hub_id),
            None,
            HELLO_DEADLINE,
        )
        .await;
        tokio::time::sleep(Duration::from_millis(100)).await;
        out
    });

    // Pre-allowlisted node reconnects with NO token — pure allowlist membership.
    let res = dial_and_hello(&node, hub_addr, node_id, String::new(), None)
        .await
        .expect("admitted");
    assert_eq!(res.agreed_version, 2);
    assert_eq!(
        res.assigned_label.as_deref(),
        Some("known"),
        "persisted label returned"
    );
    assert_eq!(
        hub_task.await.unwrap(),
        GateOutcome::Admitted {
            agreed_version: 2,
            software_version: env!("CARGO_PKG_VERSION").to_string(),
            // A current node HELLO advertises the fleet_events push (T10).
            fleet_events: true,
            update_failed: None,
            // The in-process test node is NOT a managed install (self_update_bin_dir is None), so
            // it does not advertise `update_reporting` — the P4b (d) capability gate at work.
            reports_update_failures: false,
            // A current node reports its compiled-in build identity (REL-P4e) and advertises the
            // `update` capability — this test node IS this crate, so they match exactly.
            target: Some(higgs::node::self_update::BuildIdentity::current().target),
            variant: Some(higgs::node::self_update::BuildIdentity::current().variant),
            update_capable: true,
            version_capable: true,
            log_capable: true,
            // A current node reports its in-flight downloads on demand (DL slice).
            pull_capable: true,
            // The in-process test node has no download in flight to announce.
            downloads: vec![],
        }
    );
    let _ = std::fs::remove_file(&path);
}

/// A node MUST reject a hub whose HELLO reply speaks an unsupported protocol — otherwise
/// `cli.rs` would persist that reply's `node_id` as the saved hub id and re-dial it under a
/// version we don't speak. Here the test plays a MISBEHAVING hub: it accepts the node's
/// dial, drains the node's HELLO request, then hand-writes a HelloResult with a VALID
/// identity (role=hub, node_id == our TLS id) but an UNSUPPORTED `agreed_version`. The real
/// node-side `validate_hub_hello` (via `dial_and_hello` → `connect_node`) must turn that into
/// an `Err`. On revert of the validation call this becomes `Ok` and the test fails.
#[tokio::test]
async fn node_rejects_incompatible_hub_hello() {
    use tokio::io::{AsyncBufReadExt, BufReader};

    let hub = local_endpoint().await;
    let node = local_endpoint().await;
    let hub_addr = hub.addr();
    let hub_id = hub.id().to_string();

    let hub_task = tokio::spawn(async move {
        let incoming = hub.accept().await.expect("incoming");
        let conn = incoming.await.expect("conn");
        // The node (opener) writes its HELLO first; accept that control stream.
        let (mut send, recv) = conn.accept_bi().await.expect("accept control stream");
        let mut lines = BufReader::new(recv).lines();
        let _hello_req = lines.next_line().await.expect("read node HELLO req");

        // Reply with an otherwise-valid HELLO carrying an UNSUPPORTED protocol version,
        // so ONLY the version branch of validate_hub_hello can reject it.
        let bad = HelloResult {
            role: "hub".into(),
            node_id: hub_id, // matches the TLS peer → the identity check passes
            hub_name: "evil-hub".into(),
            agreed_version: 999, // NOT in PROTOCOL_VERSIONS → must be rejected
            software_version: "9.9.9".into(),
            assigned_label: None,
            capabilities: Default::default(),
        };
        let resp = RpcResponse {
            jsonrpc: "2.0".into(),
            id: 1,
            result: Some(serde_json::to_value(bad).unwrap()),
            error: None,
        };
        // `write_all`/`finish` are inherent iroh SendStream methods (no AsyncWriteExt).
        send.write_all(format!("{}\n", rpc::encode(&RpcFrame::Response(resp))).as_bytes())
            .await
            .expect("write bad hello");
        send.finish().expect("finish");
        // Hold the conn open briefly so the node reads its reply before we drop.
        tokio::time::sleep(Duration::from_millis(200)).await;
    });

    let node_id = node.id().to_string();
    let res = dial_and_hello(&node, hub_addr, node_id, String::new(), None).await;
    assert!(
        res.is_err(),
        "node must reject a hub HELLO with an unsupported agreed_version: {res:?}"
    );
    // It's the version-mismatch diagnostic (HG023), not a transport error.
    let msg = res.unwrap_err().to_string();
    assert!(
        msg.contains("[HG023]"),
        "expected version-mismatch reject, got: {msg}"
    );

    let _ = hub_task.await;
}

/// A VALID pairing token whose pairing CANNOT be persisted (the backing dir is missing) is
/// rejected with HG040 (storage), NOT the misleading HG024 (not-allowlisted). The allowlist is
/// backed by a path inside a directory that does not exist: `load()` sees NotFound (empty, ok),
/// but the post-pairing `save()` fails → `allow.add()` errs → the gate takes the HG040 branch.
/// Reverting the fix (HG040 → HG024) makes this `assert_eq` fail.
#[tokio::test]
async fn valid_token_but_unwritable_store_is_hg040() {
    let hub = local_endpoint().await;
    let node = local_endpoint().await;
    let hub_addr = hub.addr();
    let hub_id = hub.id().to_string();

    // A path inside a directory that does NOT exist (so File::create in save() fails).
    let missing_dir = std::env::temp_dir().join(format!("higgs-p1-hg040-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&missing_dir);
    let path = missing_dir.join("pairings.json");
    let mut allow = Allowlist::load(&path).expect("empty allowlist loads when file absent");

    let mut tokens = PairingTokens::new();
    let tok = tokens.mint(1_000, 600_000);

    let hub_task = tokio::spawn(async move {
        let incoming = hub.accept().await.expect("incoming");
        let conn = incoming.await.expect("conn");
        let out = gate_connection(
            &conn,
            &mut allow,
            &mut tokens,
            2_000,
            &HubIdentity::new(hub_id),
            Some("studio".into()),
            HELLO_DEADLINE,
        )
        .await;
        tokio::time::sleep(Duration::from_millis(100)).await;
        // The persist failed, so the node must NOT be in the (rolled-back) allowlist.
        let paired = allow.contains(&conn.remote_id().to_string());
        (out, paired)
    });

    let node_id = node.id().to_string();
    let res = dial_and_hello(&node, hub_addr, node_id, String::new(), Some(tok)).await;
    let err = res.expect_err("unwritable store must reject pairing");
    // The node must receive the TYPED HG040 persistence code on the HELLO stream — NOT a
    // bare transport EOF. The hub writes an RpcError frame (carrying [HG040], which the save
    // path now stamps) before closing; the node relays that code via `hub_rejection`.
    assert!(
        err.to_string().contains("HG040"),
        "node must see the typed HG040 code, not a code-less EOF: {err}"
    );

    let (outcome, paired) = hub_task.await.unwrap();
    assert_eq!(
        outcome,
        GateOutcome::Rejected { code: "HG040" },
        "persist failure surfaces HG040 (storage), not HG024 (not-allowlisted)"
    );
    assert!(!paired, "failed persist rolls back the in-memory allowlist");

    let _ = std::fs::remove_dir_all(&missing_dir);
}
