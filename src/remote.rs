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
/// `higgs/node/pull` — DATA-plane request to download a GGUF from HuggingFace into the node's
/// own `~/.higgs/models/` (P4b). Streams [`N_PROGRESS`] then a final `{ path }`. `HG025` on
/// failure. A subsequent `M_NODE_SCAN`/`M_NODE_LOAD` then sees the pulled model.
pub const M_NODE_PULL: &str = "higgs/node/pull";
/// `N_PROGRESS` — node → hub download-progress notification on the pull stream:
/// `{ request_id, downloaded, total? }` (`total` omitted when the server sends no length).
pub const N_PROGRESS: &str = "higgs/node/progress";

/// `higgs/node/leave` — the one NODE→hub control op: the node asks the hub to retire IT
/// (`higgs node leave`). The hub authenticates by the connection's TLS `remote_id` and IGNORES
/// any payload, so a node can only ever remove ITSELF. The hub removes it from the allowlist +
/// fleet and replies `{ "status": "left" }`. Takes `{}`.
pub const M_NODE_LEAVE: &str = "higgs/node/leave";

/// The wire-protocol majors this build speaks. Major 2 (T8) is where the hub
/// STARTED SENDING the optional load params below on `M_NODE_LOAD`. Some
/// major-1 builds parse those fields; older ones hard-reject them
/// (`deny_unknown_fields` predating the rich `params` / typed `gpu_layers`) —
/// the hub cannot distinguish, so 2 is the capability statement that lets it
/// refuse a params-load against ANY major-1 node honestly.
pub const PROTOCOL_VERSIONS: &[u32] = &[1, 2];
/// The lowest major this build still accepts.
pub const MIN_SUPPORTED: u32 = 1;

/// Pairing-token lifetime. The token is a **single-use** bootstrap that only ever gates the
/// FIRST enrollment — once admitted, a node reconnects via its keypair + the hub allowlist (no
/// token), and the pairing persists until an explicit retire (hub-side or node self-`leave`),
/// NOT a clock. So the token is **effectively non-expiring** (≈100 years): single-use is the
/// real control, and retire is the revocation. This is what lets a node that was paired but
/// killed BEFORE it could store its hub still pair on the next run with the same token. Single
/// home for the TTL used by every mint site (the production hub `mint_pairing` + `higgs link
/// pair`). `validate` uses a saturating deadline so this never overflows.
pub const PAIRING_TOKEN_TTL_MS: u64 = 100 * 365 * 24 * 60 * 60 * 1000;

/// node → hub HELLO request params.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HelloParams {
    /// "node" | "hub" — self-declared, cross-checked against the allowlist.
    pub role: String,
    /// Self EndpointId (canonical string); MUST equal the QUIC peer id.
    pub node_id: String,
    /// The node's friendly name (`node-<eid8>(<host>)`), shown in the hub fleet view and
    /// stored as its allowlist label on first join. `#[serde(default)]` so an older node that
    /// omits it still parses (the hub then falls back to its own `label_for_new`).
    #[serde(default)]
    pub name: String,
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
        // `download` (M_PULL, P4b) and `log_stream` (N_LOG_LINE relay, P4) are now implemented;
        // `update` (M_UPDATE) is still only a stub (#18), so it stays advertised false.
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
    /// The hub's friendly name (`hub-<eid8>(<host>)`). The node saves this as the `label` of
    /// the hub in its `config.json` (Unit B) so `higgs --node --list` shows a human name, not a
    /// raw EndpointId. `#[serde(default)]` so an older hub that omits it still parses (empty).
    #[serde(default)]
    pub hub_name: String,
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
/// `deny_unknown_fields`: a control RPC must reject TOP-LEVEL params the node can't
/// honor rather than silently drop them (e.g. a hub sending `idle_ttl_minutes` to a node
/// with no idle reaper). The rich `params` OBJECT inside is deliberately looser
/// (`LlamaCppParams` tolerates unknown fields — engine-versioned) and its copies of the
/// base trio are ignored (base fields are authoritative at the top level only — pinned
/// in the runtime tests). Forward-compat for *optional* peer features rides the HELLO capabilities map,
/// not silently-ignored load fields.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NodeLoadParams {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ctx_len: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_layers: Option<crate::worker::engine::GpuLayers>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threads: Option<u32>,
    /// The FULL engine load params (the rich llama.cpp override set — `use_mmap`,
    /// `type_k`, `flash_attn`, `cpu_moe`, `n_seq_max`, …) the worker applies. The
    /// base fields above stay authoritative for the node's ctx-cap / resolve logic;
    /// `do_load` merges the OPTIONAL fields of this into the worker's `M_LOAD` json.
    /// `None` (omitted on the wire) ⇒ a bare load with only the base fields.
    ///
    /// Set only when there's something to apply (`LlamaCppParams::has_overrides`), so
    /// a plain/default load carries no payload. Exercised on the in-process LOCAL
    /// path (`Higgs::load` → `NodeRuntime::load`) since major 1, and on the hub's
    /// REMOTE `M_NODE_LOAD` (`HubFleet::load`) since major 2 (T8) — a params-load
    /// against a node that only negotiated major 1 is refused with [HG078]
    /// (some major-1 builds would parse the fields, older ones hard-reject;
    /// honoring them is a version-2 statement either way).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub params: Option<crate::worker::engine::llamacpp::params::LlamaCppParams>,
    // NOTE: no `idle_ttl_minutes` on the wire yet. The idle reaper lives in the node's
    // `NodeRuntime` and applies ONE per-node TTL to every worker; per-WORKER (per-load)
    // override enforcement is a deferred follow-up. The wire field and its enforcement
    // land together in that later phase — gated behind a protocol-version bump, since an
    // older `deny_unknown_fields` node would reject an unknown field — so the node never
    // accepts a TTL it would silently fail to honor. (The host `HiggsLoadRequest` accepts
    // `idle_ttl_minutes` for forward-compat but currently ignores it; see `serve::wire`.)
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

higgs_ts! {
/// One resident worker in a node's [`NodeInventory`]: its node-local id and the model it
/// currently serves, plus the hub-assigned `/v1` served id.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct InventoryWorker {
    pub worker_id: u32,
    pub model: String,
    /// The collision-free `/v1` served instance id clients call to reach THIS worker
    /// (`org/model`, `org/model-1`, …). This is a HUB concept derived from the routing
    /// table — the node does not know it, so `M_NODE_INVENTORY` payloads omit it
    /// (`serde(default)` → empty) and the hub fills it in [`crate::node::fleet`]'s
    /// `nodes_view`. Empty for a resident worker the hub holds no route for.
    #[serde(default)]
    pub served_id: String,
}
}

higgs_ts! {
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
}

/// Hub → node `M_NODE_PULL` params on a DATA stream: the file to download + the hub's
/// `request_id` (echoed in every [`N_PROGRESS`]). `revision` defaults to `"main"`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodePullParams {
    pub request_id: u64,
    pub repo: String,
    pub file: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,
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
    /// Per-request chat-template kwargs (JSON-object string), forwarded to the
    /// worker's template apply. Additive optional — absent from old hubs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chat_template_kwargs: Option<String>,
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
#[path = "remote_tests.rs"]
mod tests;
