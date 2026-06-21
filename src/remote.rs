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

/// `N_LOG_LINE` — a node → hub notification carrying ONE remote worker stderr line,
/// pushed on a dedicated uni stream the node opens after HELLO. The hub files it into its
/// `LogBus` under `LogSource::RemoteWorker { node, worker }` so the operator sees a remote
/// node's worker output in the same Developer-Logs console as local output
/// (DESIGN-remote.md §4.2, P4). Params: `{ "worker_id": <u32>, "line": <string> }`.
pub const N_LOG_LINE: &str = "higgs/node/log_line";

/// Control-plane methods (hub → node), all namespaced `higgs/node/*` so a reader never
/// confuses a hub→node op with a node→worker `higgs/*` op (DESIGN-remote.md §4.2, flag #1).
pub const M_NODE_LOAD: &str = "higgs/node/load";
pub const M_NODE_UNLOAD: &str = "higgs/node/unload";
pub const M_NODE_KILL: &str = "higgs/node/kill";
pub const M_NODE_SCAN: &str = "higgs/node/scan";
pub const M_NODE_SYSINFO: &str = "higgs/node/sysinfo";
pub const M_NODE_STATUS: &str = "higgs/node/status";
/// `higgs/node/inventory` — a node's full self-description in one call: host identity
/// (hostname/os), every resident worker (`worker_id` → model), and the hardware/runtime
/// snapshot. The hub calls it after admit (and on refresh) to populate its fleet view
/// (DESIGN-remote.md §4.2.1, P4). Takes `{}`.
pub const M_NODE_INVENTORY: &str = "higgs/node/inventory";
/// `higgs/node/update` — RESERVED handshake for a future signature-verified self-update
/// (DESIGN-remote.md §9, #18). This build only recognizes the method and refuses it with a
/// typed `HG026`; the real updater (minisign-verified download + swap) is a later task. The
/// `update` capability is advertised `false` so a well-behaved peer never sends it.
pub const M_NODE_UPDATE: &str = "higgs/node/update";

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
        // `download` (M_PULL) is P4b; `log_stream` (N_LOG_LINE relay) is P4.
        ("download", false),
        ("log_stream", false),
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

/// `higgs/node/load` params — spawn a NEW worker for model `id` (host-resolved).
///
/// `deny_unknown_fields`: a control RPC must reject params the node can't honor rather
/// than silently drop them (e.g. a hub sending `idle_ttl_minutes` to a node with no idle
/// reaper). Forward-compat for *optional* peer features rides the HELLO capabilities map,
/// not silently-ignored load fields.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NodeLoadParams {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ctx_len: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_layers: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threads: Option<u32>,
    // NOTE: no `idle_ttl_minutes` yet. A per-load idle TTL requires a node-side idle
    // reaper (the local TTL lives in the host `Higgs` reaper, not the worker). The wire
    // field and its enforcement land together in a later phase, so the node never accepts
    // a TTL it would silently fail to honor.
}

/// `{ "worker_id": <u32> }` — the target selector for `unload`/`kill`/`status`.
/// (`sysinfo` is node-level and takes `{}`; `load`'s result is [`NodeLoadResult`].)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerRef {
    pub worker_id: u32,
}

/// `higgs/node/load` result: the assigned `worker_id` plus the worker's load result
/// (`loaded`) passed through verbatim (`{id, ...}` today; richer `LoadedInfo` later) —
/// no new shape invented, matching DESIGN-remote.md §4.2.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeLoadResult {
    pub worker_id: u32,
    pub loaded: serde_json::Value,
}

/// One resident worker in a node's [`NodeInventory`]: its node-local id and the model it
/// currently serves.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct InventoryWorker {
    pub worker_id: u32,
    pub model: String,
}

/// A node's `M_NODE_INVENTORY` reply: host identity + resident workers + hardware/runtime.
/// The hub folds this into its per-node `NodeView` (§4.2.1). Hardware/runtime reuse the same
/// shapes as `M_NODE_SYSINFO` (they gained `Deserialize` for this).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeInventory {
    /// Node hostname (best-effort; empty if unavailable).
    pub hostname: String,
    /// Node OS, e.g. `"macos"` / `"linux"`.
    pub os: String,
    /// Every worker resident on the node right now.
    pub workers: Vec<InventoryWorker>,
    pub hardware: crate::system::HardwareInfo,
    pub runtime: crate::system::RuntimeInfo,
}

/// Hub → node `M_CHAT` (`higgs/chat`) params on a DATA stream: the worker selector +
/// the hub's `request_id` (echoed in `N_CHAT_CHUNK`) + the verbatim worker chat fields.
/// Not `deny_unknown_fields` — chat is a passthrough that may gain optional fields.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeChatParams {
    pub worker_id: u32,
    pub request_id: u64,
    pub model: String,
    pub messages_json: String,
    /// Omitted → the worker's default (1024), applied by the relay. (Plain `default`
    /// would forward 0 = zero-token generation.)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<usize>,
    /// Omitted → the worker's default (0.7), applied by the relay.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Serialized OpenAI `tools` array (a JSON string), matching the worker M_CHAT wire
    /// key `tools` (the value `Supervisor::chat` forwards as `tools_json`). Renamed so a
    /// hub sending the worker-compatible `tools` field is not silently dropped.
    #[serde(rename = "tools", default, skip_serializing_if = "Option::is_none")]
    pub tools_json: Option<String>,
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
        let r = NodeLoadResult { worker_id: 3, loaded: serde_json::json!({"id":"m"}) };
        let back: NodeLoadResult =
            serde_json::from_str(&serde_json::to_string(&r).unwrap()).unwrap();
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
        assert_eq!(p2.capabilities.get("telepathy"), Some(&serde_json::Value::Bool(true)));
    }
}
