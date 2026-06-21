//! P1 integration: pairing + HELLO handshake over a local (relay-disabled) iroh link.
//!
//! Two endpoints run in one process; the node dials the hub by its explicit
//! `EndpointAddr` (direct addrs from `addr()`), so no relay/discovery infra is needed.

use std::time::Duration;

use higgs::auth::{Allowlist, PairingTokens};
use higgs::node::{dial_and_hello, gate_connection, GateOutcome, HELLO_DEADLINE};
use higgs::remote::ALPN;

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
            hub_id,
            Some("studio".into()),
            HELLO_DEADLINE,
        )
        .await;
        // keep conn alive until the node has read its reply
        tokio::time::sleep(Duration::from_millis(100)).await;
        (out, allow.contains(&conn.remote_id().to_string()))
    });

    let node_id = node.id().to_string();
    let res = dial_and_hello(&node, hub_addr, node_id, Some(tok)).await;
    assert!(res.is_ok(), "valid token should pair: {res:?}");
    assert_eq!(res.unwrap().agreed_version, 1);

    let (outcome, now_paired) = hub_task.await.unwrap();
    assert_eq!(outcome, GateOutcome::Admitted { agreed_version: 1 });
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
            hub_id,
            None,
            HELLO_DEADLINE,
        )
        .await
    });

    let node_id = node.id().to_string();
    // No token, empty allowlist → hub rejects; the node's read sees the closed stream.
    let res = dial_and_hello(&node, hub_addr, node_id, None).await;
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
        gate_connection(&conn, &mut allow, &mut tokens, 2_000, hub_id, None, short).await
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
            hub_id,
            None,
            HELLO_DEADLINE,
        )
        .await
    });

    // Lie about our identity: pass a bogus self_id that won't match remote_id().
    let res = dial_and_hello(&node, hub_addr, "deadbeef-not-my-id".into(), Some(tok)).await;
    assert!(res.is_err(), "spoofed node_id must be rejected");
    assert_eq!(
        hub_task.await.unwrap(),
        GateOutcome::Rejected { code: "HG024" }
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
            hub_id,
            None,
            HELLO_DEADLINE,
        )
        .await;
        tokio::time::sleep(Duration::from_millis(100)).await;
        out
    });

    // Pre-allowlisted node reconnects with NO token — pure allowlist membership.
    let res = dial_and_hello(&node, hub_addr, node_id, None)
        .await
        .expect("admitted");
    assert_eq!(res.agreed_version, 1);
    assert_eq!(
        res.assigned_label.as_deref(),
        Some("known"),
        "persisted label returned"
    );
    assert_eq!(
        hub_task.await.unwrap(),
        GateOutcome::Admitted { agreed_version: 1 }
    );
    let _ = std::fs::remove_file(&path);
}
