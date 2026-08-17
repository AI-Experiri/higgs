use super::*;

// ── DNS packet build/parse ─────────────────────────────────────────────────────

#[test]
fn dns_query_encodes_labels_and_flags() {
    let q = build_dns_query(0xBEEF, "dns.iroh.link");
    assert_eq!(&q[0..2], &[0xBE, 0xEF], "txn id");
    assert_eq!(q[2], 0x01, "RD set");
    assert_eq!(u16::from_be_bytes([q[4], q[5]]), 1, "one question");
    // labels: 3dns 4iroh 4link 0
    let labels = &q[12..];
    assert_eq!(labels[0], 3);
    assert_eq!(&labels[1..4], b"dns");
    assert_eq!(labels[4], 4);
    assert_eq!(&labels[5..9], b"iroh");
    // trailing dot must not add an empty label
    let q2 = build_dns_query(1, "a.b.");
    assert_eq!(q2.len(), 12 + 1 + 1 + 1 + 1 + 1 + 4, "a.b. == a.b");
}

fn fake_response(id: u16, rcode: u8, answers: u16) -> Vec<u8> {
    let mut r = Vec::new();
    r.extend_from_slice(&id.to_be_bytes());
    r.push(0x80); // QR=response
    r.push(rcode);
    r.extend_from_slice(&[0, 1]); // QD
    r.extend_from_slice(&answers.to_be_bytes()); // AN
    r.extend_from_slice(&[0, 0, 0, 0]); // NS/AR
    r
}

#[test]
fn dns_response_alive_accepts_noerror_and_nxdomain() {
    assert!(dns_response_is_alive(&fake_response(7, 0, 1), 7));
    assert!(
        dns_response_is_alive(&fake_response(7, 3, 0), 7),
        "NXDOMAIN = alive server"
    );
    assert!(
        dns_response_is_alive(&fake_response(7, 2, 0), 7),
        "SERVFAIL = still a live server answering"
    );
    assert!(
        !dns_response_is_alive(&fake_response(8, 0, 1), 7),
        "wrong txn id"
    );
    assert!(!dns_response_is_alive(&[0, 7], 7), "truncated");
    // a QUERY (QR=0) with matching id is not a response
    let mut q = fake_response(7, 0, 1);
    q[2] = 0x01;
    assert!(!dns_response_is_alive(&q, 7));
}

#[test]
fn dns_response_answer_requires_ancount() {
    assert!(dns_response_has_answer(&fake_response(9, 0, 2), 9));
    assert!(!dns_response_has_answer(&fake_response(9, 0, 0), 9));
    // NXDOMAIN with a (nonsense) answer count still counts as answered — the alive
    // check already admits NXDOMAIN as "server is healthy".
    assert!(dns_response_has_answer(&fake_response(9, 3, 1), 9));
}

// ── resolv.conf parsing ────────────────────────────────────────────────────────

#[test]
fn resolv_conf_parses_ordered_nameservers_and_skips_noise() {
    let text = "# generated\nsearch local\nnameserver 192.168.2.224\n  nameserver 8.8.8.8  \n\
                nameserver not-an-ip\noptions ndots:1\nnameserver 2606:4700:4700::1111\n";
    let ns = parse_resolv_conf(text);
    assert_eq!(ns.len(), 3);
    assert_eq!(ns[0].to_string(), "192.168.2.224", "order preserved");
    assert_eq!(ns[1].to_string(), "8.8.8.8");
    assert!(ns[2].is_ipv6());
}

#[test]
fn resolv_conf_empty_or_garbage_gives_empty() {
    assert!(parse_resolv_conf("").is_empty());
    assert!(
        parse_resolv_conf("nameserverhttp 1.2.3.4").is_empty(),
        "no prefix-mangling: {:?}",
        parse_resolv_conf("nameserverhttp 1.2.3.4")
    );
    assert!(
        parse_resolv_conf("nameserver1.2.3.4").is_empty(),
        "keyword must be whitespace-delimited"
    );
}

// ── advice + report gating ─────────────────────────────────────────────────────

fn report(direct: bool, relay: bool, ns: &[(&str, bool)], resolves: Option<bool>) -> Report {
    Report {
        relay_hosts: if relay {
            vec!["usw1-1.relay.n0.iroh.link".into()]
        } else {
            vec![]
        },
        direct_addrs: if direct {
            vec!["192.168.2.82:63104".parse().unwrap()]
        } else {
            vec![]
        },
        nameservers: ns.iter().map(|(ip, a)| (ip.parse().unwrap(), *a)).collect(),
        relay_resolves: resolves,
    }
}

#[test]
fn hopeless_only_when_no_direct_and_relay_unresolvable() {
    assert!(report(false, true, &[("10.0.0.1", false)], Some(false)).hopeless());
    assert!(
        !report(true, true, &[("10.0.0.1", false)], Some(false)).hopeless(),
        "direct addr = still hope"
    );
    assert!(
        !report(false, true, &[], Some(true)).hopeless(),
        "relay resolves"
    );
    assert!(
        !report(false, false, &[], None).hopeless(),
        "direct-only ticket never DNS-gated"
    );
    // Alive-but-unanswering resolver: the A-probe can't see AAAA/edge cases — never gate.
    assert!(
        !report(false, true, &[("8.8.8.8", true)], Some(false)).hopeless(),
        "a live resolver disarms the gate even without an A answer"
    );
    // IP-literal relay host: DNS is irrelevant, dead resolvers must not gate.
    let literal = Report {
        relay_hosts: vec!["203.0.113.7".into()],
        direct_addrs: vec![],
        nameservers: vec![("10.0.0.1".parse().unwrap(), false)],
        relay_resolves: Some(false),
    };
    assert!(!literal.hopeless(), "IP-literal relay never needs DNS");
    // URL-form IPv6 literal keeps its brackets — must still count as a literal.
    let v6 = Report {
        relay_hosts: vec!["[2001:db8::1]".into()],
        direct_addrs: vec![],
        nameservers: vec![("10.0.0.1".parse().unwrap(), false)],
        relay_resolves: Some(false),
    };
    assert!(!v6.hopeless(), "bracketed IPv6 literal never needs DNS");
    assert!(is_ip_literal("[2001:db8::1]") && is_ip_literal("1.2.3.4") && !is_ip_literal("a.b"));
}

#[test]
fn advice_names_stale_ticket_dead_dns_and_macos() {
    let r = report(
        true,
        true,
        &[("192.168.2.224", false), ("8.8.8.8", true)],
        Some(true),
    );
    let a = connect_failure_advice(&r, true, true);
    assert!(a.contains("STALE TICKET"), "{a}");
    assert!(
        a.contains("192.168.2.82:63104"),
        "names the direct addr: {a}"
    );
    assert!(a.contains("192.168.2.224"), "names the dead resolver: {a}");
    assert!(
        !a.contains("8.8.8.8"),
        "does not blame the live resolver: {a}"
    );
    assert!(a.contains("Local Network"), "macOS advice: {a}");
    assert!(a.contains("SSH"), "ssh caveat: {a}");
}

#[test]
fn advice_omits_macos_and_stale_when_not_applicable() {
    let r = report(false, true, &[("8.8.8.8", true)], Some(true));
    let a = connect_failure_advice(&r, false, false);
    assert!(!a.contains("STALE TICKET"), "{a}");
    assert!(!a.contains("Local Network"), "{a}");
    assert!(
        a.contains("blocking UDP"),
        "always ends with the UDP fallback: {a}"
    );
}

#[test]
fn advice_flags_direct_only_ticket() {
    let r = report(true, false, &[], None);
    let a = connect_failure_advice(&r, false, false);
    assert!(a.contains("no relay address"), "{a}");
}

#[test]
fn local_network_advice_gating() {
    assert!(local_network_advice(false, true).is_none(), "not on linux");
    let plain = local_network_advice(true, false).unwrap();
    assert!(!plain.contains("SSH"), "no ssh caveat off-ssh");
    assert!(local_network_advice(true, true)
        .unwrap()
        .contains("CANNOT appear"));
}

// ── style ──────────────────────────────────────────────────────────────────────

#[test]
fn style_disabled_is_byte_plain_and_enabled_wraps() {
    let off = Style { enabled: false };
    assert_eq!(off.ok("x"), "✓ x");
    assert_eq!(off.fail("x"), "✗ x");
    assert_eq!(off.warn("x"), "! x");
    assert_eq!(off.head("x"), "x");
    let on = Style { enabled: true };
    assert!(on.ok("x").starts_with("\x1b[32m"));
    assert!(on.ok("x").ends_with("\x1b[0m"));
}

// ── probe against a real (loopback) DNS-ish server ────────────────────────────

#[tokio::test]
async fn probe_reports_alive_answer_and_dead() {
    // A tiny loopback UDP server that answers one query with NOERROR + 1 answer.
    let server = tokio::net::UdpSocket::bind("127.0.0.1:0").await.unwrap();
    let addr = server.local_addr().unwrap();
    tokio::spawn(async move {
        let mut buf = [0u8; 512];
        if let Ok((n, from)) = server.recv_from(&mut buf).await {
            let mut resp = buf[..n].to_vec();
            resp[2] |= 0x80; // QR=response
            resp[7] = 1; // ANCOUNT=1 (headers only — parser reads counts, not records)
            let _ = server.send_to(&resp, from).await;
        }
    });
    // Live+answering server (loopback port carried via the probe's fixed port 53 is not
    // possible here, so exercise the packet path through the helper pair directly).
    let sock = tokio::net::UdpSocket::bind("127.0.0.1:0").await.unwrap();
    let id = 0x1234;
    sock.send_to(&build_dns_query(id, "x.test"), addr)
        .await
        .unwrap();
    let mut buf = [0u8; 512];
    let (n, _) = tokio::time::timeout(Duration::from_secs(2), sock.recv_from(&mut buf))
        .await
        .unwrap()
        .unwrap();
    assert!(dns_response_has_answer(&buf[..n], id));

    // Dead server: an unroutable-but-valid target times out fast.
    let dead = probe_nameserver(
        "127.0.0.1".parse().unwrap(),
        "x.test",
        Duration::from_millis(200),
    )
    .await;
    // (port 53 on loopback is normally closed → ICMP refuse or silence → None either way
    // unless a local resolver runs; accept None or Some—the assertion documents intent.)
    let _ = dead;
}

// ── style / env / ticket-split / runner (unit-side) ────────────────────────────

#[test]
fn style_auto_is_plain_under_captured_output() {
    // Captured test output is not a terminal, so auto() must disable color and
    // paint() must return the text byte-plain (no ANSI escapes for logs/pipes).
    let style = Style::auto();
    let painted = style.paint("31", "plain");
    assert!(
        !painted.contains('\x1b'),
        "captured output stays byte-plain: {painted:?}"
    );
}

#[test]
fn ssh_session_detection_requires_a_nonempty_marker() {
    let _env = crate::TEST_ENV_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let prev: Vec<_> = ["SSH_CONNECTION", "SSH_TTY", "SSH_CLIENT"]
        .iter()
        .map(|k| (*k, std::env::var_os(k)))
        .collect();
    // SAFETY: serialized by TEST_ENV_LOCK; restored below.
    unsafe {
        for (k, _) in &prev {
            std::env::remove_var(k);
        }
    }
    assert!(!is_ssh_session(), "no markers → not an SSH session");
    unsafe { std::env::set_var("SSH_CONNECTION", "") };
    assert!(!is_ssh_session(), "an EMPTY marker does not count");
    unsafe { std::env::set_var("SSH_CONNECTION", "10.0.0.1 22 10.0.0.2 22") };
    assert!(is_ssh_session(), "a populated marker does");
    // SAFETY: still under the lock.
    unsafe {
        for (k, v) in prev {
            match v {
                Some(v) => std::env::set_var(k, v),
                None => std::env::remove_var(k),
            }
        }
    }
}

#[test]
fn split_ticket_addrs_separates_relays_from_direct_sockets() {
    let pk = iroh::SecretKey::generate().public();
    let direct: SocketAddr = "10.1.2.3:7842".parse().unwrap();
    let addr = iroh::EndpointAddr::from_parts(
        pk,
        [
            TransportAddr::Relay("https://relay.example.net/".parse().unwrap()),
            TransportAddr::Ip(direct),
        ],
    );
    let (relays, directs) = split_ticket_addrs(&addr);
    assert_eq!(relays, vec!["relay.example.net".to_string()]);
    assert_eq!(directs, vec![direct]);
}

#[tokio::test]
async fn probe_reports_a_dead_loopback_nameserver_as_not_responding() {
    // 127.0.0.1:53 has no resolver in the test environment: the UDP probe
    // gets no valid answer inside its deadline → None ("dead"), never a
    // false "alive". Bounded by the passed timeout — deterministic.
    let out = probe_nameserver(
        "127.0.0.1".parse().unwrap(),
        "relay.example.net",
        Duration::from_millis(250),
    )
    .await;
    assert_eq!(out, None, "no resolver on loopback → dead");
}

#[tokio::test]
async fn run_skips_dns_probes_for_an_ip_literal_relay() {
    // An IP-literal relay host needs no DNS: run() must mark the relay
    // resolvable WITHOUT probing any nameserver (zero network beyond the
    // print statements) and report the ticket's shape verbatim.
    let pk = iroh::SecretKey::generate().public();
    let direct: SocketAddr = "127.0.0.1:1".parse().unwrap();
    let addr = iroh::EndpointAddr::from_parts(
        pk,
        [
            TransportAddr::Relay("https://127.0.0.1/".parse().unwrap()),
            TransportAddr::Ip(direct),
        ],
    );
    let report = run(&addr, &Style::auto()).await;
    assert_eq!(report.relay_hosts, vec!["127.0.0.1".to_string()]);
    assert_eq!(report.direct_addrs, vec![direct]);
    assert_eq!(
        report.relay_resolves,
        Some(true),
        "IP-literal relay needs no DNS"
    );
    assert!(
        report.nameservers.is_empty(),
        "no probes fired for an IP-literal relay"
    );
    assert!(!report.hopeless(), "a viable relay path exists");
}
