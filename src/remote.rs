//! The remote wire vocabulary: ALPN, the `higgs/node/*` HELLO method, its serde
//! payloads, and version negotiation. Additive over the existing `rpc.rs` wire
//! (DESIGN-remote.md §4.1).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// An open capability map: feature name → JSON value. A peer NEVER hard-fails on an
/// unknown key — it ignores it. This (not the version vectors) is what lets a newer
/// hub talk to an older node and vice-versa, and is what makes M_UPDATE additive
/// (DESIGN-remote.md §4.1, §9). `Value` (not `bool`) so future non-bool caps still parse.
pub type Capabilities = BTreeMap<String, serde_json::Value>;

/// QUIC ALPN for the higgs remote protocol.
pub const ALPN: &[u8] = b"higgs/remote/1";

/// HELLO — the first control-stream frame (node → hub).
pub const M_HELLO: &str = "higgs/node/hello";

/// The wire-protocol majors this build speaks.
pub const PROTOCOL_VERSIONS: &[u32] = &[1];
/// The lowest major this build still accepts.
pub const MIN_SUPPORTED: u32 = 1;

/// node → hub HELLO request params.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HelloParams {
    /// "node" | "hub" — self-declared, cross-checked against the allowlist.
    pub role: String,
    /// Self EndpointId (canonical string); MUST equal the QUIC peer id.
    pub node_id: String,
    /// Only on first join; omitted once paired.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pairing_token: Option<String>,
    /// Every wire-protocol major this peer can speak.
    pub protocol_versions: Vec<u32>,
    /// Lowest major it will still accept.
    pub min_supported: u32,
    /// higgs build (semver) — informational + future M_UPDATE gating.
    pub software_version: String,
    /// Open capability map (e.g. `chat`, `download`, `log_stream`, `update`).
    /// Defaults to empty so a peer that omits it still parses.
    #[serde(default)]
    pub capabilities: Capabilities,
}

/// The capabilities a node advertises in its HELLO. (P1 sends the set; decisions
/// keyed on these arrive in later phases.)
pub fn node_capabilities() -> Capabilities {
    [
        ("chat", true),
        ("download", true),
        ("log_stream", true),
        ("update", false),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), serde_json::Value::Bool(v)))
    .collect()
}

/// The capabilities a hub advertises in its HELLO result.
pub fn hub_capabilities() -> Capabilities {
    [("update_push", true), ("log_aggregate", true)]
        .into_iter()
        .map(|(k, v)| (k.to_string(), serde_json::Value::Bool(v)))
        .collect()
}

/// hub → node HELLO result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HelloResult {
    pub role: String,
    pub node_id: String,
    /// The single major both sides pin for this session.
    pub agreed_version: u32,
    pub software_version: String,
    /// The hub's human label for this node (UI + LogSource).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub assigned_label: Option<String>,
    /// Open capability map (e.g. `update_push`, `log_aggregate`).
    #[serde(default)]
    pub capabilities: Capabilities,
}

/// No agreed protocol version (maps to HG023, a fatal typed close).
#[derive(Debug, PartialEq, Eq)]
pub struct VersionMismatch {
    pub peer: Vec<u32>,
    pub ours: Vec<u32>,
}

/// Agree the single major both sides pin: the max of the intersection, provided it
/// is ≥ both sides' `min_supported`. Open `capabilities` maps never gate this — only
/// the version vectors do, which is what lets a newer peer talk to an older one.
pub fn negotiate_version(
    peer_versions: &[u32],
    peer_min: u32,
    our_versions: &[u32],
    our_min: u32,
) -> Result<u32, VersionMismatch> {
    let agreed = peer_versions
        .iter()
        .filter(|v| our_versions.contains(v))
        .copied()
        .max();
    match agreed {
        Some(v) if v >= peer_min && v >= our_min => Ok(v),
        _ => Err(VersionMismatch {
            peer: peer_versions.to_vec(),
            ours: our_versions.to_vec(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
            Err(VersionMismatch { peer: vec![2], ours: vec![1] })
        );
    }

    #[test]
    fn negotiate_fails_when_overlap_below_a_min() {
        // agreed would be 1, but the peer refuses anything below 2.
        assert_eq!(
            negotiate_version(&[1, 2], 2, &[1], 1),
            Err(VersionMismatch { peer: vec![1, 2], ours: vec![1] })
        );
    }

    fn sample_params() -> HelloParams {
        HelloParams {
            role: "node".into(),
            node_id: "z32id".into(),
            pairing_token: Some("htk_abc".into()),
            protocol_versions: vec![1],
            min_supported: 1,
            software_version: "0.4.2".into(),
            capabilities: node_capabilities(),
        }
    }

    #[test]
    fn hello_params_roundtrip_json() {
        let p = sample_params();
        let s = serde_json::to_string(&p).unwrap();
        let back: HelloParams = serde_json::from_str(&s).unwrap();
        assert_eq!(back.node_id, "z32id");
        assert_eq!(back.pairing_token.as_deref(), Some("htk_abc"));
        assert_eq!(back.capabilities.get("chat"), Some(&serde_json::Value::Bool(true)));
    }

    #[test]
    fn hello_params_omits_token_when_absent() {
        let mut p = sample_params();
        p.pairing_token = None;
        let s = serde_json::to_string(&p).unwrap();
        assert!(!s.contains("pairing_token"), "absent token must not serialize");
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
        assert_eq!(p2.capabilities.get("telepathy"), Some(&serde_json::Value::Bool(true)));
    }
}
