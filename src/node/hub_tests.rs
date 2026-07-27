use super::*;
use crate::node::connect_node;
use crate::node::test_support::local_endpoint;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn accept_loop_admits_a_token_paired_node_into_the_fleet() {
    let hub_ep = local_endpoint().await;
    let node_ep = local_endpoint().await;
    let hub_addr = hub_ep.addr();
    let hub_id = hub_ep.id().to_string();

    let home = tempfile::tempdir().unwrap(); // kept alive for the test
    let fleet = Arc::new(HubFleet::new(Arc::new(LogBus::new())));
    let allow = Arc::new(Mutex::new(
        Allowlist::load(&home.path().join("p.json")).unwrap(),
    ));
    let tokens = Arc::new(Mutex::new(PairingTokens::new()));
    let token = tokens.lock().await.mint(now_ms(), TOKEN_TTL_MS);

    // Bind the loop to a fresh admission generation (mirrors `start_hub`).
    let admit_gen = fleet.bump_admit_gen().await;
    let identity = HubIdentity {
        id: hub_id,
        name: "hub-testname(srv)".into(),
    };
    spawn_accept_loop(hub_ep, fleet.clone(), allow, tokens, identity, admit_gen);

    // Node dials with the one-time token + completes HELLO (admitted + allowlisted).
    let self_id = node_ep.id().to_string();
    let (_conn, result) = connect_node(
        &node_ep,
        hub_addr,
        self_id.clone(),
        "node-x(box)".into(),
        Some(token),
    )
    .await
    .unwrap();
    assert_eq!(result.role, "hub");
    // The hub's friendly name rides the HELLO result.
    assert_eq!(result.hub_name, "hub-testname(srv)");

    // The accept loop registers the node in the fleet (poll briefly for the async add).
    let mut admitted = false;
    for _ in 0..50 {
        if fleet.node_ids().await.contains(&self_id) {
            admitted = true;
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    assert!(admitted, "node was admitted into the fleet");
    std::mem::forget(node_ep); // keep the conn alive past the test
}

#[tokio::test]
async fn mint_pairing_returns_a_token_and_ticket() {
    let home = tempfile::tempdir().unwrap();
    let ep = local_endpoint().await;
    let hub = Hub {
        hub_id: ep.id().to_string(),
        allow: Arc::new(Mutex::new(
            Allowlist::load(&home.path().join("p.json")).unwrap(),
        )),
        tokens: Arc::new(Mutex::new(PairingTokens::new())),
        endpoint: ep,
        fleet: Arc::new(HubFleet::new(Arc::new(LogBus::new()))),
    };
    let (ticket, token) = hub.mint_pairing().await;
    assert!(token.starts_with("htk_"), "minted token: {token}");
    assert!(!ticket.is_empty(), "non-empty ticket");
    assert!(!hub.hub_id().is_empty(), "hub id present");
    // The minted token validates against the hub's own store.
    assert!(hub.tokens.lock().await.validate(&token, now_ms()).is_ok());
}
