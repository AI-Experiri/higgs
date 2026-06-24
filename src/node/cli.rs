//! Lean P1 CLI to hand-drive pairing. Full fleet CLI (`link ls`, QR, keys) is P6.
//!
//! Because pairing tokens live in memory (intentionally short-lived, §7), `link pair`
//! both mints a token AND runs the accept loop in one process — a separate `pair`
//! process couldn't share the token store with a separate listener.

use std::io::{Error, Result};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use iroh_tickets::endpoint::EndpointTicket;

use std::path::Path;

use crate::auth::{Allowlist, PairingTokens};
use crate::config::{config_path, name_or_init, InstanceConfig, Role, SavedHub};
use crate::home::ensure_home;
use crate::node::identity::{bind_endpoint, load_or_create_secret};
use crate::node::runtime::{NodeConfig, NodeRuntime};
use crate::node::{dial_and_hello, gate_connection, GateOutcome, HubIdentity, HELLO_DEADLINE};
use crate::remote::{HelloResult, PAIRING_TOKEN_TTL_MS as TOKEN_TTL_MS};

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
        // The hub's persistent friendly name, sent to each node in its HELLO result.
        let identity = HubIdentity {
            id: hub_id.clone(),
            name: name_or_init(Role::Hub, &hub_id, &crate::system::hostname())?,
        };
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

        println!("higgs hub    : {} ({hub_id})", identity.name);
        println!("pairing token: {token}   (single-use)");
        println!("ticket       : {ticket}");
        // The persistent daemon (`--node`), NOT the one-shot `node connect`: the daemon saves
        // the hub so a later bare `higgs --node` reconnects on its own (no token, no ticket).
        println!("on the node:  higgs --node {ticket} {token}");
        println!("listening for dials (Ctrl-C to stop)…");

        // SIGINT/SIGTERM ends the accept loop and returns cleanly so the process runs its
        // at-exit handlers (and, under coverage, flushes its profile) rather than dying mid-accept.
        let shutdown = crate::shutdown_signal();
        tokio::pin!(shutdown);
        loop {
            let incoming = tokio::select! {
                _ = &mut shutdown => break,
                incoming = endpoint.accept() => incoming,
            };
            let Some(incoming) = incoming else { break };
            let conn = match incoming.await {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("incoming connection failed: {e}");
                    continue;
                }
            };
            let peer = conn.remote_id().to_string();
            let outcome = gate_connection(
                &conn, &mut allow, &mut tokens, now_ms(), &identity, Some("paired-node".into()),
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
        let name = name_or_init(Role::Node, &self_id, &crate::system::hostname())?;
        println!("higgs node   : {name} ({self_id})");
        let res = dial_and_hello(&endpoint, target, self_id, name, token).await?;
        println!(
            "paired with hub {} ({}) (protocol v{}, label {:?})",
            res.hub_name, res.node_id, res.agreed_version, res.assigned_label
        );
        Ok(())
    })
}

/// Re-load `config.json`, record this hub as the default, and persist — best-effort, called
/// once after the FIRST successful admission so a later bare `higgs --node` reconnects to it.
/// Re-loading (rather than mutating a stale in-memory copy) preserves whatever `name_or_init`
/// wrote concurrently. The hub's id/label come from its HELLO result (authoritative); the
/// `ticket` is the exact string we dialed. A persistence failure is logged, never fatal — the
/// node stays connected regardless.
fn persist_hub(cfg_path: &Path, hello: &HelloResult, ticket: &str) {
    let mut cfg = match InstanceConfig::load(cfg_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("higgs node: could not load config to save hub: {e}");
            return;
        }
    };
    cfg.remember_hub(SavedHub {
        hub_id: hello.node_id.clone(),
        ticket: ticket.to_string(),
        label: hello.hub_name.clone(),
        last_used_ms: now_ms(),
    });
    if let Err(e) = cfg.save(cfg_path) {
        eprintln!("higgs node: failed to save hub to config: {e}");
    }
}

/// Print the node's saved hubs (`higgs --node --list`); `★` marks the default a bare
/// `higgs --node` dials. No network.
fn list_saved_hubs(cfg: &InstanceConfig, node_id: &str) -> Result<()> {
    if cfg.hubs.is_empty() {
        println!("higgs node {node_id}: no saved hubs — pair with: higgs --node <ticket> <token>");
        return Ok(());
    }
    println!("higgs node {node_id} — saved hubs:");
    for h in &cfg.hubs {
        let default = if cfg.default_hub.as_deref() == Some(h.hub_id.as_str()) {
            " ★default"
        } else {
            ""
        };
        let short: String = h.hub_id.chars().take(8).collect();
        println!(
            "  {} ({})  last-used {}ms{}",
            h.label, short, h.last_used_ms, default
        );
    }
    Ok(())
}

/// `higgs --node [<ticket> [token]] | --list | --hub <label|id>` — the persistent node daemon.
///
/// Resolves WHICH hub to dial, then loops: dial, complete HELLO, serve the hub's
/// `higgs/node/*` control RPCs, and reconnect with backoff if the link drops (the EndpointId is
/// stable, so re-pairing is never needed). On the FIRST admission it saves the hub to
/// `config.json`, so afterwards a bare `higgs --node` reconnects on its own — no token, no
/// ticket. Modes:
/// - `<ticket> [token]` — pair/connect to a hub explicitly (token only on first enrollment).
/// - bare — connect to the default saved hub (none saved → print how to pair, exit 0).
/// - `--list` — print saved hubs and exit.
/// - `--hub <label|id>` — connect to a specific saved hub and make it the default.
pub fn run_node_daemon(args: &[String]) -> Result<()> {
    let id = load_or_create_secret(&key_path()?)?.public().to_string();
    let cfg_path = config_path()?;
    let cfg = InstanceConfig::load(&cfg_path)?;

    // `--list` short-circuits (no bind, no network).
    if args.first().map(String::as_str) == Some("--list") {
        return list_saved_hubs(&cfg, &id);
    }

    // Resolve the hub to dial + whether to present a one-time token.
    let (ticket_str, token): (String, Option<String>) = match args.first().map(String::as_str) {
        Some("--hub") => {
            let sel = args
                .get(1)
                .ok_or_else(|| Error::other("usage: higgs --node --hub <label|id>"))?;
            let hub = cfg.find_hub(sel).ok_or_else(|| {
                Error::other(format!(
                    "no saved hub matching {sel:?} — see `higgs --node --list`"
                ))
            })?;
            (hub.ticket.clone(), None)
        }
        Some(flag) if flag.starts_with("--") => {
            return Err(Error::other(format!(
                "unknown flag {flag:?} — usage: higgs --node [<ticket> [token]] | --list | --hub <label|id>"
            )));
        }
        // An explicit ticket (first-time pairing, or an explicit re-dial); token optional.
        Some(ticket) => (ticket.to_string(), args.get(1).cloned()),
        // Bare: connect to the default saved hub, or explain how to pair if there is none.
        None => match cfg.default_saved_hub() {
            Some(hub) => (hub.ticket.clone(), None),
            None => {
                let name = name_or_init(Role::Node, &id, &crate::system::hostname())?;
                println!("higgs node   : {name} ({id})");
                println!("no saved hub yet — pair with: higgs --node <ticket> <token>");
                return Ok(());
            }
        },
    };
    let ticket: EndpointTicket = ticket_str.parse().map_err(Error::other)?;
    let target = ticket.endpoint_addr().clone();

    let rt = runtime()?;
    rt.block_on(async {
        let sk = load_or_create_secret(&key_path()?)?;
        let endpoint = bind_endpoint(sk).await.map_err(Error::other)?;
        let self_id = endpoint.id().to_string();
        // This node's persistent friendly name (`node-<eid8>(<host>)`), sent in every HELLO so
        // the hub labels it in the fleet view. Generated + persisted on first run, reused after.
        let name = name_or_init(Role::Node, &self_id, &crate::system::hostname())?;
        // Model roots: the same defaults the standalone runtime uses (standard LM Studio /
        // HF / Ollama dirs) plus the HIGGS_MODEL_DIR override, so a node can actually
        // scan/load real models — not an empty set.
        let mut hc = crate::HiggsConfig::default();
        if let Ok(dir) = std::env::var("HIGGS_MODEL_DIR") {
            if !dir.is_empty() {
                hc.lmstudio_dirs.push(std::path::PathBuf::from(dir));
            }
        }
        // Pulled models (M_PULL) land in ~/.higgs/models/<org>/<model>/*.gguf — an LM-Studio
        // layout — so add it as a scan root, making a just-pulled model loadable.
        if let Ok(models) = crate::download::models_dir() {
            hc.lmstudio_dirs.push(models);
        }
        // A node has no UI; its worker stderr is relayed to the hub. HIGGS_VERBOSE=1 keeps
        // the full llama.cpp dump (default off drops the per-load metadata flood).
        let bus = Arc::new(crate::log_bus::LogBus::new());
        if std::env::var("HIGGS_VERBOSE").is_ok_and(|v| v == "1" || v == "true") {
            bus.set_verbose(true);
        }
        let node = Arc::new(NodeRuntime::new(NodeConfig {
            bus,
            lmstudio_dirs: hc.lmstudio_dirs,
            hf_dirs: hc.hf_dirs,
            ollama_dirs: hc.ollama_dirs,
            idle_ttl: crate::node::runtime::DEFAULT_IDLE_TTL,
        }));
        println!("higgs node   : {name} ({self_id}); connecting to hub…");

        // The one-time token is sent until a connect SUCCEEDS (HELLO admitted); a failed
        // attempt (hub offline, relay flake, HELLO timeout) must keep it for retry, or an
        // unallowlisted node could never pair. Reconnects after success rely on allowlist
        // membership, so the token is cleared only then.
        let mut token = token;
        // Persist the hub into config.json once, after the FIRST admission, so a later bare
        // `higgs --node` reconnects to it without a ticket or token.
        let mut saved = false;
        // SIGINT/SIGTERM ends the loop so we can drain resident workers — a dropped
        // Supervisor does not reap its child, so an undrained exit would orphan models.
        let shutdown = crate::shutdown_signal();
        tokio::pin!(shutdown);
        loop {
            tokio::select! {
                _ = &mut shutdown => break,
                res = crate::node::connect_node(&endpoint, target.clone(), self_id.clone(), name.clone(), token.clone()) => {
                    match res {
                        Ok((conn, hello)) => {
                            token = None; // admitted — token burned hub-side; don't resend it
                            if !saved {
                                saved = true;
                                persist_hub(&cfg_path, &hello, &ticket_str);
                            }
                            println!("paired with hub {} ({}) (protocol v{})", hello.hub_name, hello.node_id, hello.agreed_version);
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
