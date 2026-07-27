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
use crate::config::{name_or_init, Role};
use crate::log_bus::LogBus;
use crate::node::fleet::HubFleet;
use crate::node::identity::{bind_endpoint, load_or_create_secret};
use crate::node::transport::NodeTransport;
use crate::node::{gate_admit, gate_read_hello, GateOutcome, HubIdentity, HELLO_DEADLINE};
use crate::remote::PAIRING_TOKEN_TTL_MS as TOKEN_TTL_MS;

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

    /// Disable the hub network: close the iroh endpoint, which ends the accept loop and drops
    /// all relay connections (no more inbound dials, no relay phoning). Node transports are
    /// closed separately by the caller via `HubFleet::disconnect_all`; the fleet's route table
    /// is deliberately NOT touched, so a later `start_hub(bus, Some(fleet))` re-enables the hub
    /// with previously-loaded routes intact (the kill switch is network-only).
    pub async fn shutdown(&self) {
        self.endpoint.close().await;
    }

    /// Every paired node's `EndpointId` → its allowlist label (the node's friendly name, which
    /// an operator can rename). The serve layer merges these into the fleet view so each node
    /// shows its human name. Read-only snapshot under the allowlist lock.
    pub async fn labels(&self) -> std::collections::HashMap<String, Option<String>> {
        self.allow.lock().await.labels()
    }

    /// Rename a paired node (operator action): update its allowlist label; persists. Returns
    /// whether the node was found (`false` = unknown id, no change). Errors only on a persistence
    /// failure. The next fleet read (`Higgs::nodes()`, the `nodes` control op) reflects the new label.
    pub async fn set_label(&self, node: &str, label: Option<String>) -> std::io::Result<bool> {
        self.allow.lock().await.relabel(node, label)
    }

    /// Retire a node for good (operator action): remove its `EndpointId` from the persistent
    /// allowlist FIRST (so a reconnect can't silently re-admit without a fresh pairing token),
    /// then drop it from the fleet (transport, routes, cached inventory, relayed logs).
    /// Idempotent; the allowlist write is persisted, so it survives a hub restart.
    pub async fn retire(&self, node: &str) -> std::io::Result<()> {
        // Hold the allowlist lock across BOTH removals so retire is mutually exclusive with the
        // accept loop's admit+register critical section (which takes this same lock); otherwise a
        // concurrent admit could re-add the node between the allowlist removal and `fleet.retire`.
        // `fleet.retire` awaits the fleet actor mailbox (NOT this allow lock), so no deadlock.
        let mut allow = self.allow.lock().await;
        allow.remove(node)?;
        self.fleet.retire(node).await;
        Ok(())
    }
}

/// Build the hub from the persistent identity + allowlist under the higgs home dir, bind the
/// endpoint, spawn the accept loop, and return the live [`Hub`]. `bus` is the hub's log bus
/// (relayed remote worker stderr lands there). Call once at server startup in hub mode.
pub async fn start_hub(
    bus: Arc<LogBus>,
    existing_fleet: Option<Arc<HubFleet>>,
) -> std::io::Result<Hub> {
    let home = crate::home::ensure_home()?;
    let sk = load_or_create_secret(&home.join("endpoint.key"))?;
    let endpoint = bind_endpoint(sk).await.map_err(std::io::Error::other)?;
    // The hub's stable friendly name (`hub-<eid8>(<host>)`), generated + persisted on first run
    // and reused thereafter; sent to every node in its HELLO result so the node can label this
    // hub in its saved-hubs list (Unit B).
    let identity = HubIdentity {
        id: endpoint.id().to_string(),
        name: name_or_init(
            Role::Hub,
            &endpoint.id().to_string(),
            &crate::system::hostname(),
        )?,
    };
    // Wait (bounded) for a home relay so tickets minted by the pairing API carry a relay URL
    // and are dialable from outside the hub's LAN (mirrors `link pair`). On a relay-less /
    // local setup this times out and we proceed with whatever direct addresses we have.
    // SKIP the wait entirely in LAN-only mode (HIGGS_IROH_LOCAL): relay is disabled, so
    // `online()` can never resolve and we'd burn the full 10s on every (re-)enable for nothing.
    if std::env::var_os("HIGGS_IROH_LOCAL").is_none()
        && tokio::time::timeout(std::time::Duration::from_secs(10), endpoint.online())
            .await
            .is_err()
    {
        tracing::warn!(
            "higgs hub: no relay connected yet — pairing tickets may only be dialable on the LAN"
        );
    }
    let allow = Allowlist::load(&home.join("pairings.json"))?;
    // Re-enabling the hub (kill switch) reuses the EXISTING fleet so previously-loaded routes
    // survive the disable→enable cycle — the disable is network-only. A cold start builds a
    // fresh fleet and seeds it with the persisted allowlist so every paired node shows (as
    // disconnected) in /api/higgs/nodes before any of them reconnect.
    let fleet = match existing_fleet {
        Some(fleet) => fleet,
        None => {
            let fleet = Arc::new(HubFleet::new(bus));
            for id in allow.ids() {
                fleet.seed_node(&id).await;
            }
            fleet
        }
    };
    // A fresh admission generation for THIS accept loop. Bumping invalidates any prior loop's
    // in-flight admissions (so a disable→enable can't let an old loop's task resurrect a node).
    let admit_gen = fleet.bump_admit_gen().await;
    let hub = Hub {
        hub_id: identity.id.clone(),
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
        identity,
        admit_gen,
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
    identity: HubIdentity,
    admit_gen: u64,
) {
    tokio::spawn(async move {
        while let Some(incoming) = endpoint.accept().await {
            let (fleet, allow, tokens, identity) = (
                fleet.clone(),
                allow.clone(),
                tokens.clone(),
                identity.clone(),
            );
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
                // Build the hub identity LIVE from config.json so a `POST /api/higgs/nodes/label`
                // rename of the local instance is reflected in the `hub_name` sent to nodes that
                // pair/reconnect afterwards — without a hub restart. Falls back to the name
                // captured at `start_hub` if the config can't be read. (A dial is infrequent, so
                // the small read is negligible; the EndpointId never changes.)
                let live_identity = HubIdentity {
                    id: identity.id.clone(),
                    name: crate::config::config_path()
                        .ok()
                        .and_then(|p| crate::config::InstanceConfig::load(&p).ok())
                        .map(|c| c.name)
                        .filter(|n| !n.is_empty())
                        .unwrap_or_else(|| identity.name.clone()),
                };
                let mut allow_g = allow.lock().await;
                let mut tokens_g = tokens.lock().await;
                let outcome = gate_admit(
                    &conn,
                    handshake,
                    &mut allow_g,
                    &mut tokens_g,
                    now_ms(),
                    &live_identity,
                    Some("paired-node".into()),
                )
                .await;
                match outcome {
                    GateOutcome::Admitted {
                        agreed_version,
                        software_version,
                        fleet_events,
                        update_failed,
                        reports_update_failures,
                        target,
                        variant,
                        update_capable,
                    } => {
                        tracing::info!(node = %peer, "higgs hub: node admitted");
                        // add_node runs UNDER the allowlist lock (held here) so it's mutually
                        // exclusive with a concurrent retire — the register can't race a removal.
                        // Gated on THIS loop's admission generation: if the kill switch disabled
                        // (bumped the gen) since this loop started, the admit is refused.
                        let conn_for_requests = conn.clone();
                        let transport = Arc::new(NodeTransport::new(conn));
                        // Admit the node AND record its build identity + self-update capability from
                        // the SAME HELLO in ONE atomic message, so `fleet_update` can pick the
                        // matching release asset and skip non-capable nodes — and a `fleet_update`
                        // reacting to `NodeConnected` never sees this update-capable node before its
                        // identity is set (REL-P4e, codex r5 #2). Runs under the allowlist lock held
                        // here (retire, the only remover, takes the same lock).
                        fleet
                            .add_node_with_identity(
                                peer.clone(),
                                transport.clone(),
                                Some(admit_gen),
                                Some(agreed_version),
                                Some(software_version),
                                fleet_events,
                                update_failed,
                                reports_update_failures,
                                target,
                                variant,
                                update_capable,
                            )
                            .await;
                        // Accept node→hub requests (self-`leave`) on this connection. Holds the
                        // Arc, not the guard, and only locks it later (on leave) — so spawning it
                        // here, under the guard, can't deadlock.
                        tokio::spawn(serve_node_requests(
                            conn_for_requests,
                            peer,
                            allow.clone(),
                            fleet.clone(),
                        ));
                        drop(allow_g);
                        drop(tokens_g);
                    }
                    GateOutcome::Rejected { code } => {
                        tracing::warn!(node = %peer, code, "higgs hub: node rejected");
                        // gate_admit already wrote the typed rejection frame under the lock.
                        // Release the pairing locks BEFORE the grace-close so a slow/malicious
                        // rejected peer can't stall other admissions/token ops for the
                        // close-handshake timeout.
                        drop(allow_g);
                        drop(tokens_g);
                        crate::node::close_after_reject(&conn, code).await;
                    }
                }
            });
        }
    });
}

/// Hub side: accept NODE-opened bi streams on an admitted connection and handle node→hub
/// requests — currently only `M_NODE_LEAVE` (the node retiring itself). The node is identified by
/// `peer` (the connection's TLS `remote_id`, captured at admit), and any request payload is
/// IGNORED, so a node can only ever remove ITSELF. Runs until the connection closes (a daemon
/// node never opens such a stream, so this just idles for it). Separate stream set from the
/// hub→node control RPCs (which the hub OPENS) — QUIC multiplexes both directions.
async fn serve_node_requests(
    conn: iroh::endpoint::Connection,
    peer: String,
    allow: Arc<Mutex<Allowlist>>,
    fleet: Arc<HubFleet>,
) {
    use tokio::io::{AsyncBufReadExt, BufReader};

    use crate::node::write_frame;
    use crate::rpc::{self, RpcError, RpcFrame, RpcResponse};

    while let Ok((mut send, recv)) = conn.accept_bi().await {
        let mut lines = BufReader::new(recv).lines();
        let Ok(Some(line)) = lines.next_line().await else {
            continue;
        };
        let req = match rpc::decode(&line) {
            Ok(RpcFrame::Request(r)) => r,
            _ => continue,
        };
        if req.method != crate::remote::M_NODE_LEAVE {
            // Unknown request = a protocol skew (HG037, → 501): the shared helper
            // keeps -32601 and rides HG037 in data.code.
            let resp = RpcResponse {
                jsonrpc: "2.0".into(),
                id: req.id,
                result: None,
                error: Some(crate::rpc::method_not_found("hub", &req.method)),
            };
            let _ = write_frame(&mut send, &RpcFrame::Response(resp)).await;
            let _ = send.finish();
            continue;
        }
        // LEAVE. Tie success to the DURABLE allowlist removal: do it FIRST and reply `left` only
        // if it persisted — a failed persist replies an error so `higgs node leave` keeps the
        // node's saved hub (no false "left"). The in-memory `fleet.retire` (full removal:
        // node-id, routes, inventory) follows, and CLOSES this very connection, so it must run
        // AFTER the ack is delivered — hence the reply-then-wait-then-fleet-drop ordering.
        //
        // Splitting the allowlist removal from the fleet drop (vs `Hub::retire`'s atomic lock) is
        // safe here: the allowlist removal gates re-admission, so a concurrent dial can't re-pair
        // the leaving node token-free in the gap. And it is crash-safe — the durable removal is
        // persisted before the ack, so a hub crash before `fleet.retire` still leaves the node
        // gone (it is no longer seeded from the allowlist on restart).
        // Capture the allowlist's REAL on-disk path alongside the removal, so an
        // HG040 persistence failure below names the actual `pairings.json` (not a guess).
        let (durable, pairings_path) = {
            let mut g = allow.lock().await;
            let path = g.path().display().to_string();
            (g.remove(&peer), path)
        };
        let resp = match &durable {
            Ok(()) => RpcResponse {
                jsonrpc: "2.0".into(),
                id: req.id,
                result: Some(serde_json::json!({ "status": "left" })),
                error: None,
            },
            // The removal PERSISTS to the pairings store; a failure there (HG040) is
            // why the node's leave can't be confirmed — the node keeps its saved hub.
            // The remediation (check disk/permissions) is the hub operator's.
            Err(e) => {
                let pe = crate::diagnostic::HiggsError::PersistenceFailed {
                    store: "pairings".into(),
                    path: pairings_path,
                    source: std::io::Error::new(e.kind(), e.to_string()),
                };
                RpcResponse {
                    jsonrpc: "2.0".into(),
                    id: req.id,
                    result: None,
                    error: Some(RpcError {
                        code: -32000,
                        message: pe.to_string(),
                        data: crate::node::worker_origin_code_data(&pe),
                    }),
                }
            }
        };
        let _ = write_frame(&mut send, &RpcFrame::Response(resp)).await;
        let _ = send.finish();
        if durable.is_ok() {
            let _ = tokio::time::timeout(std::time::Duration::from_secs(2), conn.closed()).await;
            fleet.retire(&peer).await;
            tracing::info!(node = %peer, "higgs hub: node left (self-retired)");
        } else {
            tracing::error!(node = %peer, "higgs hub: node-leave allowlist removal failed");
        }
        return; // node is gone (or the error was reported); nothing more to serve here
    }
}

#[cfg(test)]
#[path = "hub_tests.rs"]
mod tests;
