use super::*;

#[test]
fn display_carries_code() {
    let e = HiggsError::ModelNotFound {
        id: "google/gemma-4-12b".into(),
    };
    assert!(e.to_string().starts_with("[HG002]"));
    assert!(e.to_string().contains("google/gemma-4-12b"));
}

#[test]
fn hg003_not_loaded_remediation_names_jit() {
    // HG003 fires when JIT auto-load is off (or a transient race); the remediation
    // must name JIT, not the stale "no JIT in v1" / "load it (or enable JIT auto-load) and retry".
    let m = HiggsError::ModelNotLoaded { id: "org/m".into() }.to_string();
    assert!(m.starts_with("[HG003]"), "{m}");
    assert!(m.contains("JIT"), "remediation names JIT auto-load: {m}");
}

#[test]
fn new_variants_carry_their_codes() {
    assert!(HiggsError::InvalidSamplingParam {
        param: "temperature".into(),
        detail: "x".into(),
    }
    .to_string()
    .starts_with("[HG013]"));
    assert!(HiggsError::ServerBusy {
        in_flight: 8,
        max: 8
    }
    .to_string()
    .starts_with("[HG014]"));
    assert!(HiggsError::InvalidModelId {
        id: "../x".into(),
        reason: "y".into(),
    }
    .to_string()
    .starts_with("[HG015]"));
    assert!(HiggsError::ChatTimeout {
        elapsed: std::time::Duration::from_secs(600),
    }
    .to_string()
    .starts_with("[HG016]"));
    assert!(HiggsError::InsufficientMemory {
        id: "org/model".into(),
        needed_bytes: 8_000_000_000,
        available_bytes: 4_000_000_000,
        headroom_fraction: 0.8,
    }
    .to_string()
    .starts_with("[HG017]"));
    assert!(HiggsError::ServingDisabled
        .to_string()
        .starts_with("[HG019]"));
    // HG020 is RETIRED but RESERVED (append-only policy) — the code still
    // formats even though nothing emits it anymore.
    assert!(HiggsError::ProbeWorkerFailed {
        context: "/models/x.gguf".into(),
    }
    .to_string()
    .starts_with("[HG020]"));
}

#[test]
fn fatal_variants_have_error_severity() {
    use miette::Diagnostic;
    let e = HiggsError::WorkerSpawnFailed {
        source: std::io::Error::new(std::io::ErrorKind::NotFound, "no exe"),
    };
    assert_eq!(e.severity(), Some(miette::Severity::Error));
}

#[test]
fn remote_gate_codes_render() {
    assert!(HiggsError::PairingTokenInvalid {
        detail: "expired".into()
    }
    .to_string()
    .starts_with("[HG022]"));
    assert!(HiggsError::NotAllowlisted {
        endpoint_id: "z32".into()
    }
    .to_string()
    .starts_with("[HG024]"));
    assert!(HiggsError::HandshakeStalled {
        endpoint_id: "z32".into(),
        window: 5
    }
    .to_string()
    .starts_with("[HG028]"));
    let unreachable = HiggsError::NodeUnreachable {
        endpoint_id: "z32".into(),
        detail: "closed".into(),
    }
    .to_string();
    assert!(unreachable.starts_with("[HG027]"));
    // HG027 must NOT claim the node was "retired" (retire is a separate, explicit
    // removal). It also must not UNCONDITIONALLY promise reconnect-recovery / "routes
    // kept" — the same code fires for an unknown/never-connected id (no routes), so the
    // remediation is conditional on it being a paired node.
    assert!(
        !unreachable.contains("retired"),
        "HG027 must not over-claim retirement: {unreachable}"
    );
    assert!(
        unreachable.contains("if it is a paired node") && unreachable.contains("reconnect"),
        "HG027 reconnect remediation must be conditional on being a paired node: {unreachable}"
    );
}

#[test]
fn hub_failure_codes_render() {
    let r = || "bartowski/Q-GGUF".to_string();
    assert!(HiggsError::HubAuthFailed {
        repo: r(),
        detail: "401".into()
    }
    .to_string()
    .starts_with("[HG029]"));
    assert!(HiggsError::HubResourceNotFound {
        repo: r(),
        resource: "file".into(),
        detail: "404".into()
    }
    .to_string()
    .starts_with("[HG030]"));
    assert!(HiggsError::HubRateLimited {
        repo: r(),
        detail: "429".into()
    }
    .to_string()
    .starts_with("[HG031]"));
    assert!(HiggsError::HubHttpStatus {
        repo: r(),
        status: 503,
        detail: "x".into()
    }
    .to_string()
    .starts_with("[HG032]"));
    assert!(HiggsError::HubTransport {
        repo: r(),
        detail: "timeout".into()
    }
    .to_string()
    .starts_with("[HG033]"));
    assert!(HiggsError::HubFileWrite {
        repo: r(),
        file: "m.gguf".into(),
        detail: "ENOSPC".into()
    }
    .to_string()
    .starts_with("[HG034]"));
    assert!(HiggsError::HubClient {
        repo: r(),
        detail: "bad json".into()
    }
    .to_string()
    .starts_with("[HG035]"));
    // The terminal both-paths-failed error preserves BOTH diagnoses.
    let exhausted = HiggsError::HubFetchExhausted {
        repo: r(),
        file: "m.gguf".into(),
        primary: "[HG029] auth".into(),
        fallback: "connection refused".into(),
    };
    let s = exhausted.to_string();
    assert!(s.starts_with("[HG036]"));
    assert!(s.contains("[HG029]") && s.contains("connection refused"));
}

#[test]
fn subsystem_fault_codes_render_with_remediation() {
    // Each new code renders its [HGxxx] tag AND carries a remediation clause
    // (the "—" separator introduces the "what to do"), per the diagnostics bar.
    let cases: Vec<(HiggsError, &str)> = vec![
        (
            HiggsError::RpcMethodNotFound {
                endpoint: "node".into(),
                method: "higgs/bogus".into(),
            },
            "[HG037]",
        ),
        (
            HiggsError::ProtocolViolation {
                peer_role: "node".into(),
                detail: "inventory decode failed".into(),
            },
            "[HG038]",
        ),
        (
            HiggsError::HubRequestRejected {
                stage: "hello".into(),
                detail: "not allowlisted".into(),
            },
            "[HG039]",
        ),
        (
            HiggsError::PersistenceFailed {
                store: "config".into(),
                path: "/x/config.json".into(),
                source: std::io::Error::other("disk full"),
            },
            "[HG040]",
        ),
        (
            HiggsError::StoreCorrupted {
                store: "pairings".into(),
                path: "/x/pairings.json".into(),
                detail: "expected value".into(),
            },
            "[HG041]",
        ),
        (
            HiggsError::InternalFault {
                context: "system info gather".into(),
                detail: "join error".into(),
            },
            "[HG042]",
        ),
        (
            HiggsError::HubControlFailed {
                op: "retire".into(),
                detail: "node gone".into(),
            },
            "[HG043]",
        ),
        (
            HiggsError::ChatTaskFailed {
                detail: "panicked".into(),
            },
            "[HG044]",
        ),
        (
            HiggsError::ControlSurfaceDown {
                reason: "serve task panicked".into(),
            },
            "[HG045]",
        ),
        (
            HiggsError::TemplateRenderFailed {
                reason: "unknown filter foo".into(),
            },
            "[HG050]",
        ),
    ];
    for (err, code) in cases {
        let s = err.to_string();
        assert!(s.starts_with(code), "code prefix: {s}");
        assert!(
            s.contains(" — "),
            "every diagnostic must carry a remediation clause (em-dash): {s}"
        );
    }
    // The two high-severity faults are fatal-severity.
    use miette::Diagnostic;
    assert_eq!(
        HiggsError::InternalFault {
            context: "x".into(),
            detail: "y".into()
        }
        .severity(),
        Some(miette::Severity::Error)
    );
    assert_eq!(
        HiggsError::ChatTaskFailed { detail: "y".into() }.severity(),
        Some(miette::Severity::Error)
    );
}

/// The node-chat-test ladder codes render their `[HGxxx]` tags AND `code()`
/// returns the bare string jigglebot's status mapping keys on. The `code()`
/// assertion is the load-bearing one for [HG077]: it is race-only (never
/// dispatch-drivable end to end), so this attribute is the ONLY link between
/// `ChatTestTargetMoved` and jigglebot's 409 — a typo'd or dropped
/// `#[diagnostic(code(HG077))]` would silently demote the race to the 500
/// default with every other test green.
#[test]
fn chat_test_ladder_codes_render_and_key() {
    use miette::Diagnostic;
    let cases: Vec<(HiggsError, &str)> = vec![
        (
            HiggsError::NodeNothingServed {
                endpoint_id: "abc".into(),
            },
            "HG074",
        ),
        (
            HiggsError::UnknownNode {
                endpoint_id: "abc".into(),
            },
            "HG075",
        ),
        (
            HiggsError::InvalidChatTestTarget { detail: "x".into() },
            "HG076",
        ),
        (
            HiggsError::ChatTestTargetMoved { detail: "x".into() },
            "HG077",
        ),
        (
            HiggsError::NodeTooOldForParams {
                endpoint_id: "abc".into(),
                agreed: 1,
            },
            "HG078",
        ),
    ];
    for (err, code) in cases {
        let s = err.to_string();
        assert!(s.starts_with(&format!("[{code}]")), "display prefix: {s}");
        assert!(
            !s.contains("  "),
            "no doubled spaces from a missing line-continuation: {s}"
        );
        // HG076 is detail-only like HG072: its remediation lives in the
        // call-site detail — all THREE producing arms are wording-pinned in the
        // embed tests (local sentinel: "this machine"+"directly"+em-dash;
        // mismatch: "omit `served`"; unrouted: em-dash + "refresh the fleet
        // view") — so the fixed display carries no em-dash of its own.
        if code != "HG076" {
            assert!(
                s.contains(" — "),
                "every diagnostic must carry a remediation clause (em-dash): {s}"
            );
        }
        assert_eq!(
            err.code().expect("diagnostic code").to_string(),
            code,
            "code() must render the bare HGxxx string the status mapping keys on"
        );
    }
}

#[test]
fn version_mismatch_is_fatal() {
    use miette::Diagnostic;
    let e = HiggsError::VersionMismatch {
        peer: vec![2],
        ours: vec![1],
    };
    assert!(e.to_string().starts_with("[HG023]"));
    assert_eq!(e.severity(), Some(miette::Severity::Error));
}
