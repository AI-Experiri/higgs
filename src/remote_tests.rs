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
    // The pre-model_info row also decodes cleanly — `model_info` is
    // `serde(default)` so a pre-r-N node payload without the key stays
    // wire-compatible with a current hub (the whole point of Option<T>).
    assert!(full.model_info.is_none(), "no key present → None");
}

/// NL-VX successor: the full-parity `model_info` on an inventory row
/// round-trips (every static `HiggsModel` fact a LOCAL client sees flows
/// through), and the wire skip-attrs on `HiggsModel` (`path` is on the
/// wire, `chat_template` is `serde(skip)`) behave as documented.
#[test]
fn inventory_worker_carries_full_model_info_round_trip() {
    use crate::worker::models::{HiggsModel, HiggsModelSource, ModelDomain};
    let full = HiggsModel {
        id: "org/model".into(),
        path: "/tmp/models/org/model/f.gguf".into(),
        size_bytes: 1024,
        quant: Some("Q4_K_M".into()),
        source: HiggsModelSource::LmStudio,
        arch: Some("llama".into()),
        ctx_train: Some(4096),
        block_count: Some(32),
        head_count: Some(32),
        head_count_kv: Some(8),
        embedding_length: Some(4096),
        expert_count: None,
        has_chat_template: true,
        domain: ModelDomain::Llm,
        supports_tools: true,
        supports_reasoning: false,
        gguf_components: vec![],
        enrich_error: None,
        chat_template: Some("SECRET-HOST-ONLY".into()),
    };
    let row = InventoryWorker {
        worker_id: 9,
        model: "org/model".into(),
        served_id: String::new(),
        ctx_len: Some(2048),
        gpu_layers: None,
        domain: ModelDomain::Llm,
        threads: None,
        loaded_at_ms: Some(1),
        idle_ms: Some(0),
        in_flight: Some(0),
        model_info: Some(full.clone()),
    };
    let wire = serde_json::to_string(&row).expect("serialize");
    // `chat_template` is `serde(skip)` on HiggsModel → must not appear on the wire.
    assert!(
        !wire.contains("SECRET-HOST-ONLY"),
        "chat_template stays host-only: {wire}"
    );
    let back: InventoryWorker = serde_json::from_str(&wire).expect("decode");
    let info = back.model_info.expect("model_info round-trips");
    assert_eq!(info.id, full.id);
    assert_eq!(info.quant, full.quant);
    assert_eq!(info.arch, full.arch);
    assert_eq!(info.ctx_train, full.ctx_train);
    assert_eq!(info.block_count, full.block_count);
    assert_eq!(info.head_count_kv, full.head_count_kv);
    assert_eq!(info.embedding_length, full.embedding_length);
    assert!(info.has_chat_template);
    assert!(info.supports_tools);
    assert!(!info.supports_reasoning);
    // path DOES ride the wire (HiggsModel has no `serde(skip)` on it) —
    // documented in InventoryWorker's doc comment; the hub simply ignores
    // it as a node-local string.
    assert_eq!(info.path, full.path);
    // chat_template drops through the wire hop (per `serde(skip)`).
    assert!(
        info.chat_template.is_none(),
        "chat_template does NOT survive the wire"
    );
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
        downloads: vec![],
    }
}

#[test]
fn hello_announces_in_flight_downloads_and_they_roundtrip() {
    // "Hello, I am downloading …": the announcement rides HELLO so a
    // reconnecting hub continues a surviving transfer instead of re-issuing.
    let mut p = sample_params();
    p.downloads = vec![HelloDownload {
        repo: "acme/m".into(),
        file: "m-Q4.gguf".into(),
        downloaded: 123_456,
        total: Some(999_999),
        cancellable: true,
    }];
    let v = serde_json::to_value(&p).unwrap();
    assert_eq!(v["downloads"][0]["repo"], "acme/m");
    let back: HelloParams = serde_json::from_value(v).unwrap();
    assert_eq!(back.downloads, p.downloads);
    // An EMPTY list is omitted from the wire (additive-field hygiene), and a
    // legacy HELLO without the field parses to empty.
    let empty = serde_json::to_value(sample_params()).unwrap();
    assert!(empty.get("downloads").is_none());
    let legacy: HelloParams = serde_json::from_value(empty).unwrap();
    assert!(legacy.downloads.is_empty());
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

#[test]
fn node_capabilities_advertise_node_logs() {
    let caps = node_capabilities(false);
    assert_eq!(
        caps.get("node_logs"),
        Some(&serde_json::Value::Bool(true)),
        "current builds serve their daemon log on demand"
    );
}

#[test]
fn announced_downloads_are_validated_exact_or_dropped_never_truncated() {
    // `(repo, file)` is the download's addressable IDENTITY — the exact key
    // the node's cancel registry holds and the key an operator acts on from
    // the fleet view — so a VALID identity longer than any display cap must
    // survive VERBATIM. Truncating it would leave a visible row no
    // cancel/continue can ever address.
    let long = HelloDownload {
        repo: "acme/models".into(),
        file: format!("{}.gguf", "q".repeat(150)),
        downloaded: 7,
        total: Some(9),
        cancellable: true,
    };
    // Entries that could NEVER have registered on a well-behaved node (the
    // node validates via `dest_path` before registering a pull) are DROPPED
    // whole — never rewritten into a different, equally unaddressable string.
    let traversal = HelloDownload {
        repo: "../etc".into(),
        file: "pw.gguf".into(),
        downloaded: 0,
        total: None,
        cancellable: true,
    };
    let ansi = HelloDownload {
        repo: "acme/m".into(),
        file: "\u{1b}[2Jwipe.gguf".into(),
        downloaded: 0,
        total: None,
        cancellable: true,
    };
    let not_gguf = HelloDownload {
        repo: "acme/m".into(),
        file: "notes.txt".into(),
        downloaded: 0,
        total: None,
        cancellable: true,
    };
    // Longer than NAME_MAX (255 bytes): no such file can exist on disk, so no
    // real pull carries it — dropped, not displayed.
    let overlong = HelloDownload {
        repo: "acme/m".into(),
        file: format!("{}.gguf", "q".repeat(300)),
        downloaded: 0,
        total: None,
        cancellable: true,
    };
    let out = accept_announced_downloads(&[long.clone(), traversal, ansi, not_gguf, overlong]);
    assert_eq!(
        out,
        vec![long],
        "valid identity kept exact; unregistrable entries dropped whole"
    );
}

#[test]
fn announced_downloads_list_is_capped_at_16() {
    // The producer bounds its list (HELLO frame protection); the hub enforces
    // the same cap on the untrusted side.
    let raw: Vec<HelloDownload> = (0..40)
        .map(|i| HelloDownload {
            repo: "acme/m".into(),
            file: format!("f{i}.gguf"),
            downloaded: 0,
            total: None,
            cancellable: true,
        })
        .collect();
    assert_eq!(accept_announced_downloads(&raw).len(), 16);
}

#[test]
fn announced_download_counters_are_normalized_at_the_trust_boundary() {
    // The identity is validated, but the COUNTERS are also node-supplied: an
    // impossible pair (`total == Some(0)`, or `downloaded > total`) would feed
    // a UI percent/bar a divide-by-zero or >100%. The entry itself stays (the
    // transfer is real — dropping it would hide a live download); only the
    // inconsistent `total` degrades to None ("length unknown"), which every
    // consumer already renders.
    let mk = |downloaded: u64, total: Option<u64>| HelloDownload {
        repo: "acme/m".into(),
        file: "m.gguf".into(),
        downloaded,
        total,
        cancellable: true,
    };
    let out = accept_announced_downloads(&[mk(7, Some(0))]);
    assert_eq!(out[0].total, None, "zero total is not a divisor");
    assert_eq!(out[0].downloaded, 7, "byte count kept");
    let out = accept_announced_downloads(&[mk(u64::MAX, Some(1))]);
    assert_eq!(
        out[0].total, None,
        "downloaded > total is impossible — total degrades"
    );
    // Consistent pairs pass verbatim, including the complete boundary — but
    // dedup by (repo, file) keeps first-only, so vary the file per entry.
    let mk_f = |file: &str, downloaded, total| HelloDownload {
        repo: "acme/m".into(),
        file: file.into(),
        downloaded,
        total,
        cancellable: true,
    };
    let out = accept_announced_downloads(&[
        mk_f("a.gguf", 5, Some(5)),
        mk_f("b.gguf", 1, Some(5)),
        mk_f("c.gguf", 3, None),
    ]);
    assert_eq!(out[0].total, Some(5));
    assert_eq!(out[1].total, Some(5));
    assert_eq!(out[2].total, None);
}

#[test]
fn announced_downloads_are_deduped_by_repo_file_before_the_cap() {
    // A faulty/hostile node could fill the 16 slots with duplicates of one
    // key — hiding real entries and breaking the addressability invariant
    // ("one operator action key → one UI row"). First occurrence wins.
    let mut raw = Vec::new();
    for i in 0..17 {
        raw.push(HelloDownload {
            repo: "acme/m".into(),
            file: "same.gguf".into(),
            downloaded: i,
            total: None,
            cancellable: true,
        });
    }
    raw.push(HelloDownload {
        repo: "acme/m".into(),
        file: "different.gguf".into(),
        downloaded: 99,
        total: None,
        cancellable: true,
    });
    let out = accept_announced_downloads(&raw);
    assert_eq!(
        out.len(),
        2,
        "duplicates collapse, second key survives: {out:?}"
    );
    assert_eq!(out[0].downloaded, 0, "first-of-duplicates wins");
    assert_eq!(out[1].file, "different.gguf");
}

/// r87 pin: the dedup key is CASE-FOLDED, matching the machine
/// download-lock's identity fold — case-variant names are one on-disk file
/// / one lock slot on default case-insensitive APFS, so they must collapse
/// to one UI row (first occurrence, kept verbatim). Reverting the seen-set
/// key to the exact-case tuple keeps both variants and fails this pin.
#[test]
fn case_variant_announced_downloads_collapse_to_one_folded_key() {
    let mk = |repo: &str, file: &str, downloaded| HelloDownload {
        repo: repo.into(),
        file: file.into(),
        downloaded,
        total: None,
        cancellable: true,
    };
    let out = accept_announced_downloads(&[
        mk("acme/m", "same.gguf", 1),
        mk("ACME/m", "SAME.GGUF", 2),
        mk("acme/M", "Same.Gguf", 3),
    ]);
    assert_eq!(out.len(), 1, "case variants are one folded key: {out:?}");
    assert_eq!(out[0].repo, "acme/m", "first occurrence kept VERBATIM");
    assert_eq!(out[0].file, "same.gguf");
    assert_eq!(out[0].downloaded, 1);
}

#[test]
fn hello_download_cancellable_defaults_false_for_legacy_wires() {
    // The `cancellable` bit is additive: a hub decoding a NEWER node's
    // payload sees the field; a hub decoding an OLDER node's payload
    // (no field) must NOT falsely offer a cancel that can't take. Serde
    // default is `false`.
    let legacy = serde_json::json!({
        "repo": "acme/m", "file": "m.gguf",
        "downloaded": 5, "total": 10
    });
    let d: HelloDownload = serde_json::from_value(legacy).expect("legacy row parses");
    assert!(!d.cancellable, "no field on wire → observe-only by default");
    // A NEWER wire with the field set true parses through.
    let modern = serde_json::json!({
        "repo": "acme/m", "file": "m.gguf",
        "downloaded": 5, "total": 10, "cancellable": true
    });
    let d2: HelloDownload = serde_json::from_value(modern).expect("modern row parses");
    assert!(d2.cancellable, "explicit true is honored");
}
