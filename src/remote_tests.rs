use super::*;

/// T14 r10: the peer-controlled version string is display-sanitized at the
/// gate — ANSI escapes and newlines cannot reach a pairing terminal, while a
/// normal semver passes through unchanged (the e2e equality pins rely on it).
#[test]
fn version_sanitizer_strips_terminal_control_and_keeps_semver() {
    assert_eq!(sanitize_version("0.1.0-beta.1"), "0.1.0-beta.1");
    assert_eq!(sanitize_version("1.2.3+build_7"), "1.2.3+build_7");
    // ANSI clear-screen + CRLF injection reduce to their safe residue (the
    // ESC, brackets, CR/LF, and space are all dropped — nothing executes).
    assert_eq!(sanitize_version("1.0\x1b[2J\r\nfake ok"), "1.02Jfakeok");
    assert_eq!(sanitize_version("\u{1b}[31mred\u{1b}[0m"), "31mred0m");
    // Length cap.
    assert_eq!(sanitize_version(&"9".repeat(200)).len(), 64);
    // The NAME sanitizer keeps normal friendly names intact (spaces, parens)
    // and strips only terminal-control characters.
    assert_eq!(
        sanitize_display("hub-abc12345(my host)"),
        "hub-abc12345(my host)"
    );
    assert_eq!(sanitize_display("evil\u{1b}[2Jname\r\n"), "evil[2Jname");
    // Bidi overrides/isolates are FORMAT (Cf) chars is_control misses — they
    // visually reorder a printed line, the same spoof class (r23).
    assert_eq!(sanitize_display("ok\u{202E}deriapnu"), "okderiapnu");
    assert_eq!(sanitize_display("a\u{2066}b\u{2069}c\u{200F}"), "abc");
    // Other INVISIBLE format chars (ZWSP/WJ/BOM/tag chars) also drop (r26).
    assert_eq!(
        sanitize_display("a\u{200B}b\u{2060}c\u{FEFF}d\u{E0041}"),
        "abcd"
    );
    // LINE/PARAGRAPH separators (Zl/Zp) are NOT Cc — `is_control` misses them, but a JSON/JS/terminal
    // renderer treats them as a newline, so an `update_failed.reason` like "HG084\u{2028}update
    // succeeded" could forge a second line. They (and the soft hyphen + other Cf gaps) must drop.
    assert_eq!(
        sanitize_display("HG084\u{2028}spoof\u{2029}line\u{00AD}hyphen"),
        "HG084spooflinehyphen"
    );
    assert_eq!(
        sanitize_display("a\u{0600}b\u{06DD}c\u{070F}d\u{08E2}e\u{180E}f\u{FFF9}g"),
        "abcdefg"
    );
    // The COMPLETE Cf set — including the deprecated/format chars and supplementary-plane format
    // controls a piecemeal list keeps missing (U+206F, U+0890, U+110BD, U+13430, U+1BCA0, U+1D173).
    assert_eq!(
        sanitize_display("a\u{206F}b\u{0890}c\u{110BD}d\u{13430}e\u{1BCA0}f\u{1D173}g\u{E0041}h"),
        "abcdefgh"
    );
    assert_eq!(sanitize_display(&"x".repeat(300)).len(), 128);
}

/// P4b (d): a node advertises the `update_reporting` capability ONLY when it can actually inspect
/// its `.update-lastfail` marker (a MANAGED install) — so the hub can trust a `None` from it as
/// authoritative. A non-managed/dev launch must NOT advertise it (its always-`None` report would
/// otherwise erase a stored failure it can't speak to).
#[test]
fn node_capabilities_gates_update_reporting_on_managed() {
    let managed = node_capabilities(true);
    assert_eq!(
        managed.get("update_reporting"),
        Some(&serde_json::Value::Bool(true)),
        "a managed install advertises update_reporting"
    );
    let non_managed = node_capabilities(false);
    assert!(
        !non_managed.contains_key("update_reporting"),
        "a non-managed launch must NOT advertise update_reporting"
    );
    // The other capabilities are unconditional either way.
    assert!(non_managed.contains_key("chat") && managed.contains_key("update"));
}

/// T9 version-skew: a PRE-stats node's inventory (no per-worker stat keys)
/// still decodes on a current hub — the fields are additive Options with
/// serde defaults, exactly why no protocol bump was needed.
#[test]
fn legacy_inventory_worker_decodes_without_stats() {
    let w: InventoryWorker =
        serde_json::from_value(serde_json::json!({ "worker_id": 3, "model": "org/m" }))
            .expect("legacy row decodes");
    assert_eq!(w.worker_id, 3);
    assert!(w.ctx_len.is_none() && w.loaded_at_ms.is_none() && w.in_flight.is_none());
    // The old-node PERMISSIVE contract: a domain-less legacy row reads as `Llm`
    // (the pre-domain behaviour) — any other default would drop every old
    // node's workers from /v1/models and pre-refuse their chats (Fable r8).
    assert_eq!(w.domain, crate::worker::models::ModelDomain::Llm);
    // And a CURRENT row round-trips its stats.
    let full: InventoryWorker = serde_json::from_value(serde_json::json!({
        "worker_id": 4, "model": "org/m", "ctx_len": 256,
        "loaded_at_ms": 1, "idle_ms": 2, "in_flight": 1
    }))
    .expect("stats row decodes");
    assert_eq!(full.ctx_len, Some(256));
    assert_eq!(full.in_flight, Some(1));
}

#[test]
fn negotiate_picks_max_common_version() {
    assert_eq!(negotiate_version(&[1], 1, &[1], 1), Ok(1));
    assert_eq!(negotiate_version(&[1, 2], 1, &[1], 1), Ok(1));
    assert_eq!(negotiate_version(&[1, 2], 1, &[1, 2], 1), Ok(2));
}

#[test]
fn negotiate_fails_with_no_overlap() {
    assert_eq!(
        negotiate_version(&[2], 2, &[1], 1),
        Err(VersionMismatch {
            peer: vec![2],
            ours: vec![1]
        })
    );
}

#[test]
fn negotiate_fails_when_overlap_below_a_min() {
    // agreed would be 1, but the peer refuses anything below 2.
    assert_eq!(
        negotiate_version(&[1, 2], 2, &[1], 1),
        Err(VersionMismatch {
            peer: vec![1, 2],
            ours: vec![1]
        })
    );
}

fn sample_params() -> HelloParams {
    HelloParams {
        role: "node".into(),
        node_id: "z32id".into(),
        name: "node-z32id000(box)".into(),
        pairing_token: Some("htk_abc".into()),
        protocol_versions: vec![1],
        min_supported: 1,
        software_version: "0.4.2".into(),
        update_failed: None,
        target: Some("aarch64-apple-darwin".into()),
        variant: Some("metal".into()),
        capabilities: node_capabilities(true),
    }
}

#[test]
fn hello_params_roundtrip_json() {
    let p = sample_params();
    let s = serde_json::to_string(&p).unwrap();
    let back: HelloParams = serde_json::from_str(&s).unwrap();
    assert_eq!(back.node_id, "z32id");
    assert_eq!(back.pairing_token.as_deref(), Some("htk_abc"));
    assert_eq!(
        back.capabilities.get("chat"),
        Some(&serde_json::Value::Bool(true))
    );
}

#[test]
fn hello_carries_friendly_names() {
    // The node's name rides HelloParams; an older node omitting it still parses (empty).
    let p = sample_params();
    let back: HelloParams = serde_json::from_str(&serde_json::to_string(&p).unwrap()).unwrap();
    assert_eq!(back.name, "node-z32id000(box)");
    let older = r#"{"role":"node","node_id":"z","protocol_versions":[1],
            "min_supported":1,"software_version":"0.1.0"}"#;
    assert_eq!(serde_json::from_str::<HelloParams>(older).unwrap().name, "");

    // The hub's name rides HelloResult; an older hub omitting it still parses (empty).
    let r = HelloResult {
        role: "hub".into(),
        node_id: "hubid".into(),
        hub_name: "hub-3f9a2b1c(srv)".into(),
        agreed_version: 1,
        software_version: "0.4.2".into(),
        assigned_label: Some("node-z32id000(box)".into()),
        capabilities: hub_capabilities(),
    };
    let back: HelloResult = serde_json::from_str(&serde_json::to_string(&r).unwrap()).unwrap();
    assert_eq!(back.hub_name, "hub-3f9a2b1c(srv)");
    let older_hub = r#"{"role":"hub","node_id":"h","agreed_version":1,"software_version":"0.1.0"}"#;
    assert_eq!(
        serde_json::from_str::<HelloResult>(older_hub)
            .unwrap()
            .hub_name,
        ""
    );
}

#[test]
fn hello_params_omits_token_when_absent() {
    let mut p = sample_params();
    p.pairing_token = None;
    let s = serde_json::to_string(&p).unwrap();
    assert!(
        !s.contains("pairing_token"),
        "absent token must not serialize"
    );
}

#[test]
fn node_method_consts_are_namespaced() {
    assert_eq!(M_NODE_LOAD, "higgs/node/load");
    assert_eq!(M_NODE_UNLOAD, "higgs/node/unload");
    assert_eq!(M_NODE_KILL, "higgs/node/kill");
    assert_eq!(M_NODE_SCAN, "higgs/node/scan");
    assert_eq!(M_NODE_SYSINFO, "higgs/node/sysinfo");
    assert_eq!(M_NODE_STATUS, "higgs/node/status");
}

#[test]
fn load_params_roundtrip() {
    let p = NodeLoadParams {
        id: "org/m".into(),
        ctx_len: Some(4096),
        gpu_layers: None,
        threads: None,
        params: None,
    };
    let s = serde_json::to_string(&p).unwrap();
    let back: NodeLoadParams = serde_json::from_str(&s).unwrap();
    assert_eq!(back.id, "org/m");
    assert_eq!(back.ctx_len, Some(4096));
    // absent optionals don't serialize
    assert!(!s.contains("gpu_layers"));
}

#[test]
fn load_params_reject_unhonorable_fields() {
    // A node with no idle reaper must reject (not silently drop) a TTL param.
    let with_ttl = r#"{"id":"m","idle_ttl_minutes":30}"#;
    assert!(serde_json::from_str::<NodeLoadParams>(with_ttl).is_err());
}

#[test]
fn worker_ref_roundtrip() {
    let r = WorkerRef { worker_id: 7 };
    let back: WorkerRef = serde_json::from_str(&serde_json::to_string(&r).unwrap()).unwrap();
    assert_eq!(back.worker_id, 7);
}

#[test]
fn node_load_result_carries_worker_id_and_loaded() {
    let r = NodeLoadResult {
        worker_id: 3,
        loaded: serde_json::json!({"id":"m"}),
    };
    let back: NodeLoadResult = serde_json::from_str(&serde_json::to_string(&r).unwrap()).unwrap();
    assert_eq!(back.worker_id, 3);
    assert_eq!(back.loaded["id"], "m");
}

#[test]
fn hello_tolerates_missing_and_unknown_capabilities() {
    // Forward/back compat: an older peer omits `capabilities` entirely...
    let older = r#"{"role":"node","node_id":"z","protocol_versions":[1],
            "min_supported":1,"software_version":"0.1.0"}"#;
    let p: HelloParams = serde_json::from_str(older).unwrap();
    assert!(p.capabilities.is_empty());

    // ...and a newer peer advertises an unknown capability we simply keep, not reject.
    let newer = r#"{"role":"node","node_id":"z","protocol_versions":[1],
            "min_supported":1,"software_version":"9.9.9",
            "capabilities":{"telepathy":true,"chat":true}}"#;
    let p2: HelloParams = serde_json::from_str(newer).unwrap();
    assert_eq!(
        p2.capabilities.get("telepathy"),
        Some(&serde_json::Value::Bool(true))
    );
}
