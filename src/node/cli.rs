//! Lean P1 CLI to hand-drive pairing. Full fleet CLI (`link ls`, QR, keys) is P6.
//!
//! Because pairing tokens live in memory (intentionally short-lived, §7), `link pair`
//! both mints a token AND runs the accept loop in one process — a separate `pair`
//! process couldn't share the token store with a separate listener.

use std::io::{Error, Result};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use iroh_tickets::endpoint::EndpointTicket;

use crate::auth::{Allowlist, PairingTokens};
use crate::home::ensure_home;
use crate::node::identity::{bind_endpoint, load_or_create_secret};
use crate::node::runtime::{NodeConfig, NodeRuntime};
use crate::node::{dial_and_hello, gate_connection, GateOutcome, HELLO_DEADLINE};

/// Pairing-token lifetime: 10 minutes (DESIGN-remote.md §7).
const TOKEN_TTL_MS: u64 = 10 * 60 * 1000;

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn runtime() -> Result<tokio::runtime::Runtime> {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
}

fn key_path() -> Result<std::path::PathBuf> {
    Ok(ensure_home()?.join("endpoint.key"))
}

fn pairings_path() -> Result<std::path::PathBuf> {
    Ok(ensure_home()?.join("pairings.json"))
}

/// `higgs link <pair|status>` — hub-side fleet (Surface A).
pub fn run_link(args: &[String]) -> Result<()> {
    match args.first().map(String::as_str) {
        Some("pair") => run_link_pair(),
        Some("status") => run_link_status(),
        other => {
            eprintln!("usage: higgs link <pair|status> (got {other:?})");
            Err(Error::other("unknown link subcommand"))
        }
    }
}

/// Mint a one-time token, print the pairing ticket, and accept dials until Ctrl-C.
fn run_link_pair() -> Result<()> {
    let rt = runtime()?;
    rt.block_on(async {
        let sk = load_or_create_secret(&key_path()?)?;
        let endpoint = bind_endpoint(sk).await.map_err(Error::other)?;
        let hub_id = endpoint.id().to_string();
        let mut allow = Allowlist::load(&pairings_path()?)?;
        let mut tokens = PairingTokens::new();
        let token = tokens.mint(now_ms(), TOKEN_TTL_MS);

        // Wait (bounded) for a home relay so the ticket carries a relay URL and is
        // dialable from outside the hub's LAN. On a relay-less / offline setup we fall
        // back to whatever addresses we have (LAN-only) with a warning.
        if tokio::time::timeout(Duration::from_secs(10), endpoint.online())
            .await
            .is_err()
        {
            eprintln!("warning: no relay connected yet — ticket may only be dialable on the local network");
        }
        let ticket = EndpointTicket::new(endpoint.addr());

        println!("higgs hub id : {hub_id}");
        println!("pairing token: {token}   (valid 10m, single-use)");
        println!("ticket       : {ticket}");
        println!("on the node:  higgs node connect {ticket} {token}");
        println!("listening for dials (Ctrl-C to stop)…");

        loop {
            let Some(incoming) = endpoint.accept().await else { break };
            let conn = match incoming.await {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("incoming connection failed: {e}");
                    continue;
                }
            };
            let peer = conn.remote_id().to_string();
            let outcome = gate_connection(
                &conn, &mut allow, &mut tokens, now_ms(), hub_id.clone(), Some("paired-node".into()),
                HELLO_DEADLINE,
            )
            .await;
            match outcome {
                GateOutcome::Admitted { agreed_version } => {
                    println!("paired {peer} (protocol v{agreed_version})");
                    // Hold the connection until the node has read the HELLO result
                    // (it closes after reading); dropping it now can truncate the reply.
                    let _ = tokio::time::timeout(Duration::from_secs(5), conn.closed()).await;
                }
                GateOutcome::Rejected { code } => {
                    println!("rejected {peer} [{code}]");
                }
            }
        }
        Ok(())
    })
}

/// Print this hub's identity and the count of paired nodes.
fn run_link_status() -> Result<()> {
    let id = load_or_create_secret(&key_path()?)?.public();
    let allow = Allowlist::load(&pairings_path()?)?;
    println!("higgs hub id : {id}");
    println!("paired nodes : {}", allow.len());
    Ok(())
}

/// `higgs node connect <ticket> [token]` — dial a hub and complete HELLO.
pub fn run_node(args: &[String]) -> Result<()> {
    match args.first().map(String::as_str) {
        Some("connect") => run_node_connect(&args[1..]),
        other => {
            eprintln!("usage: higgs node connect <ticket> [token] (got {other:?})");
            Err(Error::other("unknown node subcommand"))
        }
    }
}

fn run_node_connect(args: &[String]) -> Result<()> {
    let ticket_str = args
        .first()
        .ok_or_else(|| Error::other("usage: higgs node connect <ticket> [token]"))?;
    let token = args.get(1).cloned();
    let ticket: EndpointTicket = ticket_str.parse().map_err(Error::other)?;
    let target = ticket.endpoint_addr().clone();

    let rt = runtime()?;
    rt.block_on(async {
        let sk = load_or_create_secret(&key_path()?)?;
        let endpoint = bind_endpoint(sk).await.map_err(Error::other)?;
        let self_id = endpoint.id().to_string();
        println!("higgs node id: {self_id}");
        let res = dial_and_hello(&endpoint, target, self_id, token).await?;
        println!(
            "paired with hub {} (protocol v{}, label {:?})",
            res.node_id, res.agreed_version, res.assigned_label
        );
        Ok(())
    })
}

/// `higgs --node [<ticket> [token]]` — run the persistent node daemon: dial the hub,
/// complete HELLO, then serve its `higgs/node/*` control RPCs, reconnecting with backoff
/// if the link drops (the EndpointId is stable, so re-pairing isn't needed). With no
/// ticket, just print this node's identity.
pub fn run_node_daemon(args: &[String]) -> Result<()> {
    let id = load_or_create_secret(&key_path()?)?.public();
    let Some(ticket_str) = args.first() else {
        println!("higgs node id: {id}");
        println!("usage: higgs --node <ticket> [token]   (run as a persistent node)");
        return Ok(());
    };
    let token = args.get(1).cloned();
    let ticket: EndpointTicket = ticket_str.parse().map_err(Error::other)?;
    let target = ticket.endpoint_addr().clone();

    let rt = runtime()?;
    rt.block_on(async {
        let sk = load_or_create_secret(&key_path()?)?;
        let endpoint = bind_endpoint(sk).await.map_err(Error::other)?;
        let self_id = endpoint.id().to_string();
        // Model roots: the same defaults the standalone runtime uses (standard LM Studio /
        // HF / Ollama dirs) plus the HIGGS_MODEL_DIR override, so a node can actually
        // scan/load real models — not an empty set.
        let mut hc = crate::HiggsConfig::default();
        if let Ok(dir) = std::env::var("HIGGS_MODEL_DIR") {
            if !dir.is_empty() {
                hc.lmstudio_dirs.push(std::path::PathBuf::from(dir));
            }
        }
        let node = Arc::new(NodeRuntime::new(NodeConfig {
            bus: Arc::new(crate::log_bus::LogBus::new()),
            lmstudio_dirs: hc.lmstudio_dirs,
            hf_dirs: hc.hf_dirs,
            ollama_dirs: hc.ollama_dirs,
        }));
        println!("higgs node id: {self_id}; connecting to hub…");

        // The one-time token is sent until a connect SUCCEEDS (HELLO admitted); a failed
        // attempt (hub offline, relay flake, HELLO timeout) must keep it for retry, or an
        // unallowlisted node could never pair. Reconnects after success rely on allowlist
        // membership, so the token is cleared only then.
        let mut token = token;
        // SIGINT/SIGTERM ends the loop so we can drain resident workers — a dropped
        // Supervisor does not reap its child, so an undrained exit would orphan models.
        let shutdown = crate::shutdown_signal();
        tokio::pin!(shutdown);
        loop {
            tokio::select! {
                _ = &mut shutdown => break,
                res = crate::node::connect_node(&endpoint, target.clone(), self_id.clone(), token.clone()) => {
                    match res {
                        Ok((conn, hello)) => {
                            token = None; // admitted — token burned hub-side; don't resend it
                            println!("paired with hub {} (protocol v{})", hello.node_id, hello.agreed_version);
                            tokio::select! {
                                _ = &mut shutdown => break,
                                _ = crate::node::serve_node(conn, node.clone()) => {
                                    eprintln!("hub connection closed; reconnecting…");
                                }
                            }
                        }
                        Err(e) => eprintln!("higgs node: connect failed: {e}"),
                    }
                }
            }
            tokio::select! {
                _ = &mut shutdown => break,
                _ = tokio::time::sleep(Duration::from_secs(3)) => {}
            }
        }
        println!("higgs node: draining resident workers…");
        node.shutdown_all().await;
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn link_rejects_unknown_subcommand() {
        assert!(run_link(&["bogus".into()]).is_err());
        assert!(run_link(&[]).is_err());
    }

    #[test]
    fn node_rejects_unknown_subcommand() {
        assert!(run_node(&["bogus".into()]).is_err());
        assert!(run_node(&[]).is_err());
    }

    #[test]
    fn node_connect_requires_a_ticket() {
        // No ticket arg → usage error, before any runtime/bind.
        assert!(run_node(&["connect".into()]).is_err());
    }

    #[test]
    fn node_connect_rejects_malformed_ticket() {
        // A malformed ticket fails at parse, before any runtime/bind/network.
        let err = run_node(&["connect".into(), "not-a-ticket".into()]).unwrap_err();
        assert!(!err.to_string().is_empty());
    }
}
