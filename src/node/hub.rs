//! Production hub listener (DESIGN-remote.md §3, P3 wiring): the piece that turns a running
//! `higgs` server into a HUB. It binds the hub's iroh endpoint, gates inbound node dials
//! against the persistent allowlist + one-time pairing tokens, and registers each admitted
//! node in the [`HubFleet`] the serve layer routes `/v1` chat through.
//!
//! One process owns BOTH the HTTP surface and the node-accept loop, so a token minted via the
//! pairing API is honored by the same accept loop (no cross-process token sharing). The
//! endpoint + token store live on the returned [`Hub`], which the server keeps alive and uses
//! to mint pairings.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use iroh::Endpoint;
use tokio::sync::Mutex;

use crate::auth::{Allowlist, PairingTokens};
use crate::log_bus::LogBus;
use crate::node::fleet::HubFleet;
use crate::node::identity::{bind_endpoint, load_or_create_secret};
use crate::node::transport::NodeTransport;
use crate::node::{gate_admit, gate_read_hello, GateOutcome, HELLO_DEADLINE};

/// Pairing-token lifetime: 10 minutes (matches the `link pair` CLI / §7).
const TOKEN_TTL_MS: u64 = 10 * 60 * 1000;

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// A running hub: its endpoint, fleet, and the shared allowlist + token store the accept loop
/// gates against. Held by the server for its lifetime; used to mint pairings.
pub struct Hub {
    pub endpoint: Endpoint,
    pub fleet: Arc<HubFleet>,
    allow: Arc<Mutex<Allowlist>>,
    tokens: Arc<Mutex<PairingTokens>>,
    hub_id: String,
}

impl Hub {
    /// The hub's canonical `EndpointId` (the node dials this).
    pub fn hub_id(&self) -> &str {
        &self.hub_id
    }

    /// Mint a one-time pairing token (valid [`TOKEN_TTL_MS`]) and return it with a dialable
    /// ticket for this hub — what an operator hands to a new node
    /// (`higgs --node <ticket> <token>`).
    pub async fn mint_pairing(&self) -> (String, String) {
        let token = self.tokens.lock().await.mint(now_ms(), TOKEN_TTL_MS);
        let ticket = iroh_tickets::endpoint::EndpointTicket::new(self.endpoint.addr()).to_string();
        (ticket, token)
    }

    /// Retire a node for good: remove its `EndpointId` from the persistent allowlist FIRST
    /// (so a reconnect can't silently re-admit without a fresh pairing token), then drop it
    /// from the fleet (transport, routes, cached inventory, relayed logs). Idempotent —
    /// retiring an unknown id no-ops on both. The allowlist write is persisted, so the
    /// retirement survives a hub restart.
    pub async fn retire(&self, node: &str) -> std::io::Result<()> {
        // Hold the allowlist lock across BOTH removals so retire is mutually exclusive with
        // the accept loop's admit+register critical section (which takes this same lock).
        // Otherwise a concurrent admit could re-add the node between the allowlist removal and
        // `fleet.retire`. `fleet.retire` awaits the fleet actor mailbox (NOT this allow lock),
        // so there is no await-under-lock deadlock.
        let mut allow = self.allow.lock().await;
        allow.remove(node)?;
        self.fleet.retire(node).await;
        Ok(())
    }
}

/// Build the hub from the persistent identity + allowlist under the higgs home dir, bind the
/// endpoint, spawn the accept loop, and return the live [`Hub`]. `bus` is the hub's log bus
/// (relayed remote worker stderr lands there). Call once at server startup in hub mode.
pub async fn start_hub(bus: Arc<LogBus>) -> std::io::Result<Hub> {
    let home = crate::home::ensure_home()?;
    let sk = load_or_create_secret(&home.join("endpoint.key"))?;
    let endpoint = bind_endpoint(sk).await.map_err(std::io::Error::other)?;
    // Wait (bounded) for a home relay so tickets minted by the pairing API carry a relay URL
    // and are dialable from outside the hub's LAN (mirrors `link pair`). On a relay-less /
    // local setup this times out and we proceed with whatever direct addresses we have.
    if tokio::time::timeout(std::time::Duration::from_secs(10), endpoint.online())
        .await
        .is_err()
    {
        tracing::warn!(
            "higgs hub: no relay connected yet — pairing tickets may only be dialable on the LAN"
        );
    }
    let allow = Allowlist::load(&home.join("pairings.json"))?;
    let fleet = Arc::new(HubFleet::new(bus));
    // Seed the fleet with persisted pairings so a just-restarted hub lists every paired node
    // (as disconnected) in /api/higgs/nodes before any of them reconnect.
    for id in allow.ids() {
        fleet.seed_node(&id).await;
    }
    let hub = Hub {
        hub_id: endpoint.id().to_string(),
        allow: Arc::new(Mutex::new(allow)),
        tokens: Arc::new(Mutex::new(PairingTokens::new())),
        endpoint: endpoint.clone(),
        fleet: fleet.clone(),
    };
    spawn_accept_loop(
        endpoint,
        fleet,
        hub.allow.clone(),
        hub.tokens.clone(),
        hub.hub_id.clone(),
    );
    Ok(hub)
}

/// Spawn the accept loop: gate each inbound dial and register admitted nodes in `fleet`.
/// Factored out (endpoint + state injected) so a test drives it with an in-process endpoint.
pub fn spawn_accept_loop(
    endpoint: Endpoint,
    fleet: Arc<HubFleet>,
    allow: Arc<Mutex<Allowlist>>,
    tokens: Arc<Mutex<PairingTokens>>,
    hub_id: String,
) {
    tokio::spawn(async move {
        while let Some(incoming) = endpoint.accept().await {
            let (fleet, allow, tokens, hub_id) =
                (fleet.clone(), allow.clone(), tokens.clone(), hub_id.clone());
            tokio::spawn(async move {
                let conn = match incoming.await {
                    Ok(c) => c,
                    Err(e) => {
                        tracing::warn!(error = %e, "higgs hub: inbound connection failed");
                        return;
                    }
                };
                let peer = conn.remote_id().to_string();
                // Phase 1 — read + validate HELLO WITHOUT the pairing locks, so a peer that
                // stalls before/at HELLO can't hold them for the deadline and starve other
                // joins or `POST /api/higgs/pair`.
                let handshake = match gate_read_hello(&conn, HELLO_DEADLINE).await {
                    Ok(h) => h,
                    Err(GateOutcome::Rejected { code }) => {
                        tracing::warn!(node = %peer, code, "higgs hub: node rejected (pre-auth)");
                        return;
                    }
                    Err(_) => return,
                };
                // Phase 2 — the lock-needing admit decision + reply (fast, in-memory + a
                // small post-auth write). Registration into the fleet happens INSIDE the same
                // allowlist critical section as the admit: `Hub::retire` takes this same lock
                // to remove a node from the allowlist + fleet, so registering under the lock
                // closes the admit→register window where a concurrent retire could otherwise
                // re-introduce a just-retired node into the fleet view. `add_node` awaits the
                // fleet actor mailbox (which never takes this allow lock), so holding the lock
                // across it preserves that mutual exclusion and can't deadlock.
                let mut allow = allow.lock().await;
                let mut tokens = tokens.lock().await;
                let outcome = gate_admit(
                    &conn,
                    handshake,
                    &mut allow,
                    &mut tokens,
                    now_ms(),
                    hub_id,
                    Some("paired-node".into()),
                )
                .await;
                match outcome {
                    GateOutcome::Admitted { .. } => {
                        tracing::info!(node = %peer, "higgs hub: node admitted");
                        fleet
                            .add_node(peer, Arc::new(NodeTransport::new(conn)))
                            .await;
                    }
                    GateOutcome::Rejected { code } => {
                        tracing::warn!(node = %peer, code, "higgs hub: node rejected");
                    }
                }
                drop(allow);
                drop(tokens);
            });
        }
    });
}

#[cfg(test)]
mod tests {
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

        spawn_accept_loop(hub_ep, fleet.clone(), allow, tokens, hub_id);

        // Node dials with the one-time token + completes HELLO (admitted + allowlisted).
        let self_id = node_ep.id().to_string();
        let (_conn, result) = connect_node(&node_ep, hub_addr, self_id.clone(), Some(token))
            .await
            .unwrap();
        assert_eq!(result.role, "hub");

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
}
