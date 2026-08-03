//! Pairing preflight — gated, self-diagnosing checks run by `higgs --node <ticket> <token>`
//! BEFORE and AROUND the pairing attempt, so a failed pairing names its cause and the exact
//! user action instead of a bare "timed out" (docs/pairing-preflight-checklist.md).
//!
//! Design rules:
//! - Every network-facing check is DERIVED FROM THE TICKET: relay/DNS checks run only when the
//!   ticket carries a relay address; direct-path advice only when it carries IP addresses. A
//!   direct-only ticket (the in-process test transport, LAN-only setups) never touches DNS.
//! - Checks are GATED: a hard failure stops before the connect attempt with the fix printed;
//!   re-running re-verifies. Soft findings (a dead resolver among live ones, no IPv6) print as
//!   warnings and continue.
//! - Output is for HUMANS: colored ✓/✗/! lines (tty-gated, NO_COLOR honored), same style as
//!   install.sh. The iroh WARN firehose is demoted separately (bin/higgs.rs default filter).
//!
//! The DNS probes deliberately speak raw UDP DNS (a hand-rolled A query) rather than the
//! system resolver: they must observe each configured nameserver INDIVIDUALLY — exactly what
//! the in-process hickory resolver experiences — not the OS resolver's silent failover that
//! masked a dead primary during the first real-hardware install.

use std::net::{IpAddr, SocketAddr};
use std::time::Duration;

use iroh::{EndpointAddr, TransportAddr};

/// Per-nameserver probe timeout. Deliberately short: a live server answers in tens of
/// milliseconds; anything past this is effectively dead for pairing purposes (hickory's
/// in-connect budget is only a few seconds total).
pub const DNS_PROBE_TIMEOUT: Duration = Duration::from_millis(1500);

// ---------------------------------------------------------------------------
// Colored output (tty-gated, NO_COLOR honored) — install.sh's palette.
// ---------------------------------------------------------------------------

/// Terminal styling for preflight lines. `enabled` is decided once at startup.
#[derive(Debug, Clone, Copy)]
pub struct Style {
    pub enabled: bool,
}

impl Style {
    /// Colors only when BOTH stdout and stderr are terminals and NO_COLOR is unset —
    /// captured output (tests, service logs, pipes) stays byte-plain.
    pub fn auto() -> Self {
        use std::io::IsTerminal;
        Self {
            enabled: std::io::stdout().is_terminal()
                && std::io::stderr().is_terminal()
                && std::env::var_os("NO_COLOR").is_none(),
        }
    }

    fn paint(&self, code: &str, s: &str) -> String {
        if self.enabled {
            format!("\x1b[{code}m{s}\x1b[0m")
        } else {
            s.to_string()
        }
    }
    pub fn head(&self, s: &str) -> String {
        self.paint("1", s)
    }
    pub fn ok(&self, s: &str) -> String {
        self.paint("32", &format!("✓ {s}"))
    }
    pub fn warn(&self, s: &str) -> String {
        self.paint("33", &format!("! {s}"))
    }
    pub fn fail(&self, s: &str) -> String {
        self.paint("31", &format!("✗ {s}"))
    }
}

// ---------------------------------------------------------------------------
// Environment facts (pure / trivially-injectable — unit tested).
// ---------------------------------------------------------------------------

/// True when this process runs inside an SSH session — the context where macOS can never
/// show its Local Network permission popup, so the advice must say so up front.
pub fn is_ssh_session() -> bool {
    ["SSH_CONNECTION", "SSH_TTY", "SSH_CLIENT"]
        .iter()
        .any(|k| std::env::var_os(k).is_some_and(|v| !v.is_empty()))
}

/// The macOS Local Network advisory. `None` off macOS. Pure on inputs for testability.
pub fn local_network_advice(macos: bool, ssh: bool) -> Option<String> {
    if !macos {
        return None;
    }
    let mut s = String::from(
        "macOS may block this program's local-network/UDP traffic until the Local Network \
         permission is granted (System Settings → Privacy & Security → Local Network).",
    );
    if ssh {
        s.push_str(
            " You are connected over SSH: the approval popup CANNOT appear here (Apple allows \
             no command-line grant). If the connection fails, run this once in the machine's \
             own Terminal and click Allow.",
        );
    }
    Some(s)
}

/// True when a relay host string is an IP literal — including the URL form of an
/// IPv6 literal, which keeps its brackets (`[2001:db8::1]`).
pub fn is_ip_literal(host: &str) -> bool {
    host.trim_start_matches('[')
        .trim_end_matches(']')
        .parse::<IpAddr>()
        .is_ok()
}

/// Split a ticket's addresses into (relay hosts, direct socket addresses).
pub fn split_ticket_addrs(addr: &EndpointAddr) -> (Vec<String>, Vec<SocketAddr>) {
    let mut relays = Vec::new();
    let mut direct = Vec::new();
    for a in &addr.addrs {
        match a {
            TransportAddr::Relay(url) => {
                if let Some(h) = url.host_str() {
                    relays.push(h.to_string());
                }
            }
            TransportAddr::Ip(sa) => direct.push(*sa),
            _ => {}
        }
    }
    (relays, direct)
}

// ---------------------------------------------------------------------------
// Raw-UDP DNS (per-nameserver probes) — packet build/parse are pure and unit tested.
// ---------------------------------------------------------------------------

/// Build a minimal RFC 1035 A-record query for `host` with transaction id `id`.
pub fn build_dns_query(id: u16, host: &str) -> Vec<u8> {
    let mut q = Vec::with_capacity(64);
    q.extend_from_slice(&id.to_be_bytes());
    q.extend_from_slice(&[0x01, 0x00]); // RD=1
    q.extend_from_slice(&[0, 1, 0, 0, 0, 0, 0, 0]); // QD=1
    for label in host.trim_end_matches('.').split('.') {
        q.push(label.len() as u8);
        q.extend_from_slice(label.as_bytes());
    }
    q.push(0); // root
    q.extend_from_slice(&[0, 1, 0, 1]); // QTYPE=A, QCLASS=IN
    q
}

/// True when `buf` is a well-formed response to transaction `id` — ANY RCODE counts:
/// the probe asks "is this SERVER alive", and a SERVFAIL/REFUSED reply is still a live
/// server answering (a broken upstream must not trip the all-dead hard gate).
pub fn dns_response_is_alive(buf: &[u8], id: u16) -> bool {
    if buf.len() < 12 {
        return false;
    }
    if buf[0..2] != id.to_be_bytes() {
        return false;
    }
    buf[2] & 0x80 != 0
}

/// True when the response carries at least one answer record (a real resolution, not
/// just a live server).
pub fn dns_response_has_answer(buf: &[u8], id: u16) -> bool {
    dns_response_is_alive(buf, id) && buf.len() >= 8 && u16::from_be_bytes([buf[6], buf[7]]) > 0
}

/// Parse `nameserver` lines from resolv.conf-format text (what hickory itself reads on
/// both macOS and Linux). Order preserved; comments and garbage ignored.
pub fn parse_resolv_conf(text: &str) -> Vec<IpAddr> {
    text.lines()
        .filter_map(|l| {
            let l = l.trim();
            let rest = l.strip_prefix("nameserver")?;
            // resolv.conf grammar requires whitespace after the keyword —
            // `nameserver1.2.3.4` is NOT a nameserver line.
            if !rest.starts_with([' ', '\t']) {
                return None;
            }
            rest.split_whitespace().next()?.parse().ok()
        })
        .collect()
}

/// The system's configured nameservers, in order (empty on read failure).
pub fn system_nameservers() -> Vec<IpAddr> {
    std::fs::read_to_string("/etc/resolv.conf")
        .map(|t| parse_resolv_conf(&t))
        .unwrap_or_default()
}

/// Probe one nameserver with an A query for `host`: `Some(true)` = answered with a
/// resolution, `Some(false)` = alive but no answer, `None` = dead (no reply in time).
pub async fn probe_nameserver(ns: IpAddr, host: &str, timeout: Duration) -> Option<bool> {
    let bind: SocketAddr = match ns {
        IpAddr::V4(_) => "0.0.0.0:0".parse().unwrap(),
        IpAddr::V6(_) => "[::]:0".parse().unwrap(),
    };
    let sock = tokio::net::UdpSocket::bind(bind).await.ok()?;
    // Randomize the txn id from the socket's ephemeral port (no rand dependency needed;
    // this is a health probe, not a security boundary).
    let id = sock.local_addr().ok().map(|a| a.port()).unwrap_or(0x5147);
    let target = SocketAddr::new(ns, 53);
    let q = build_dns_query(id, host);
    sock.send_to(&q, target).await.ok()?;
    // Keep receiving until the deadline: a stray/mismatched datagram (wrong source,
    // stale txn id) must not end the probe early and misreport a live server as dead.
    let deadline = tokio::time::Instant::now() + timeout;
    let mut buf = [0u8; 512];
    loop {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        if remaining.is_zero() {
            return None;
        }
        match tokio::time::timeout(remaining, sock.recv_from(&mut buf)).await {
            Ok(Ok((n, from))) if from == target && dns_response_is_alive(&buf[..n], id) => {
                return Some(dns_response_has_answer(&buf[..n], id));
            }
            Ok(Ok(_)) => continue, // stray datagram — keep waiting
            _ => return None,      // socket error or deadline
        }
    }
}

// ---------------------------------------------------------------------------
// Report + advice (pure — unit tested).
// ---------------------------------------------------------------------------

/// What the preflight learned; drives both the gate and the post-failure advice.
#[derive(Debug, Default, Clone)]
pub struct Report {
    /// Relay hostnames in the ticket (empty = direct-only ticket, DNS not needed).
    pub relay_hosts: Vec<String>,
    /// Direct IP addresses in the ticket.
    pub direct_addrs: Vec<SocketAddr>,
    /// Per-nameserver probe outcome: (server, alive).
    pub nameservers: Vec<(IpAddr, bool)>,
    /// Whether the relay hostname resolved via at least one live nameserver.
    pub relay_resolves: Option<bool>,
}

impl Report {
    /// At least one configured nameserver answered.
    pub fn any_live_dns(&self) -> bool {
        self.nameservers.iter().any(|(_, alive)| *alive)
    }
    /// The relay path is plausibly usable (no relay in ticket counts as unusable).
    pub fn relay_viable(&self) -> bool {
        !self.relay_hosts.is_empty() && self.relay_resolves == Some(true)
    }
    /// HARD GATE: no viable path at all — a relay-ONLY ticket whose relay needs DNS
    /// while EVERY configured nameserver is dead. Connecting would be a guaranteed
    /// opaque timeout. Deliberately narrow: an alive-but-unanswering resolver does NOT
    /// gate (the relay may resolve via AAAA or a path this A-probe can't see — let the
    /// dial try), and an IP-literal relay host never needs DNS at all.
    pub fn hopeless(&self) -> bool {
        // DNS is only load-bearing when NO relay host is an IP literal — one literal
        // is a DNS-free relay path, so the gate must not fire.
        let needs_dns =
            !self.relay_hosts.is_empty() && !self.relay_hosts.iter().any(|h| is_ip_literal(h));
        self.direct_addrs.is_empty()
            && needs_dns
            && !self.nameservers.is_empty()
            && !self.any_live_dns()
    }
}

/// The advice printed when the CONNECT attempt fails after a clean-enough preflight.
/// Pure so every branch is unit tested.
pub fn connect_failure_advice(r: &Report, macos: bool, ssh: bool) -> String {
    let mut lines = Vec::new();
    lines.push("pairing could not reach the hub. Most likely causes, in order:".to_string());
    if !r.direct_addrs.is_empty() {
        lines.push(format!(
            "- STALE TICKET: the hub may have restarted since this ticket was minted (its \
             direct address {} would then be outdated). Mint a fresh 'Pair a node' in the hub \
             UI and run the new command.",
            r.direct_addrs[0]
        ));
    }
    if let Some(advice) = local_network_advice(macos, ssh) {
        lines.push(format!("- {advice}"));
    }
    let dead: Vec<String> = r
        .nameservers
        .iter()
        .filter(|(_, alive)| !alive)
        .map(|(ip, _)| ip.to_string())
        .collect();
    if !dead.is_empty() {
        lines.push(format!(
            "- DNS: configured nameserver(s) {} did not answer — fix or remove them in your \
             network settings (a dead first resolver can starve lookups even when a later one \
             works).",
            dead.join(", ")
        ));
    }
    if r.relay_hosts.is_empty() {
        lines.push(
            "- the ticket carries no relay address: only the direct path exists, so hub and \
             node must share a network."
                .to_string(),
        );
    }
    lines.push(
        "- if every check above looks clean, the network may be blocking UDP for this \
         process."
            .to_string(),
    );
    lines.join("\n")
}

/// Run the ticket-derived preflight, printing as it goes. Returns the report; the caller
/// gates on [`Report::hopeless`].
pub async fn run(addr: &EndpointAddr, style: &Style) -> Report {
    let (relay_hosts, direct_addrs) = split_ticket_addrs(addr);
    let mut report = Report {
        relay_hosts: relay_hosts.clone(),
        direct_addrs: direct_addrs.clone(),
        ..Report::default()
    };

    println!("{}", style.head("higgs pair: preflight"));
    // Ticket summary — the user sees exactly where we will try to go.
    let relay_disp = if relay_hosts.is_empty() {
        "none".to_string()
    } else {
        relay_hosts.join(", ")
    };
    println!(
        "  {}",
        style.ok(&format!(
            "ticket: hub {} · relay {} · {} direct address(es)",
            addr.id.fmt_short(),
            relay_disp,
            direct_addrs.len()
        ))
    );

    // Host context: macOS + SSH advisory (informational — macOS gives no way to query the
    // Local Network permission state, so this can only warn, not verify).
    if let Some(advice) = local_network_advice(cfg!(target_os = "macos"), is_ssh_session()) {
        println!("  {}", style.warn(&advice));
    }

    // DNS checks only when the ticket actually needs a name resolved. An IP-literal
    // relay host needs none — say so and skip the probes entirely.
    let dns_host = relay_hosts.iter().find(|h| !is_ip_literal(h));
    if !relay_hosts.is_empty() && dns_host.is_none() {
        println!(
            "  {}",
            style.ok(&format!(
                "relay {} is an IP literal — DNS not required",
                relay_hosts[0]
            ))
        );
        report.relay_resolves = Some(true);
    }
    if let Some(dns_host) = dns_host {
        let servers = system_nameservers();
        if servers.is_empty() {
            println!(
                "  {}",
                style.warn("DNS: no nameservers found in system config")
            );
        }
        let mut any_resolved = false;
        for ns in servers {
            let outcome = probe_nameserver(ns, dns_host, DNS_PROBE_TIMEOUT).await;
            match outcome {
                Some(answered) => {
                    println!("  {}", style.ok(&format!("DNS {ns}: answering")));
                    report.nameservers.push((ns, true));
                    if answered {
                        any_resolved = true;
                    }
                }
                None => {
                    println!(
                        "  {}",
                        style.fail(&format!(
                            "DNS {ns}: NOT RESPONDING — fix or remove it in your network \
                             settings"
                        ))
                    );
                    report.nameservers.push((ns, false));
                }
            }
        }
        report.relay_resolves = Some(any_resolved);
        if any_resolved {
            println!("  {}", style.ok(&format!("relay {dns_host} resolves")));
        } else {
            println!(
                "  {}",
                style.fail(&format!(
                    "relay {dns_host} did not resolve via any configured nameserver"
                ))
            );
        }
    }
    report
}

#[cfg(test)]
#[path = "preflight_tests.rs"]
mod tests;
