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

/// `N_FLEET_EVENT` — a node → hub notification pushed on a DEDICATED uni stream the node
/// opens after HELLO (separate from the [`N_LOG_LINE`] stream; one stream ⇒ QUIC preserves
/// event order). Each event marks a node-side worker-state change (chat start/end,
/// worker load/unload) and CARRIES the authoritative post-change worker snapshot
/// ([`NodeFleetEvent`]), so the hub can update its cached inventory without a pull —
/// ordered by the same node-actor `snapshot_seq` as `M_NODE_INVENTORY` replies.
/// Additive (T10): the `fleet_events` capability advertises it; a hub that doesn't
/// know the method skips the frames (its notification reader filters by method).
pub const N_FLEET_EVENT: &str = "higgs/node/fleet_event";

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

/// Sanitize a peer-supplied DISPLAY NAME for a terminal (T14 r22/r23): names
/// (`hub-<eid8>(<host>)`, friendly labels) legitimately contain spaces,
/// punctuation, and non-ASCII hostnames, so unlike [`sanitize_version`] this
/// strips only what can SPOOF a terminal — control characters (Cc: ANSI
/// escapes, CR/LF) AND the bidi-override/format characters `is_control`
/// misses (Cf: RLO/LRO/isolates/marks, which visually reorder a printed
/// line) — and caps the length. Normal names pass intact.
pub fn sanitize_display(raw: &str) -> String {
    fn is_invisible_format(c: char) -> bool {
        matches!(
            c,
            '\u{200B}'..='\u{200F}'                          // ZWSP/ZWNJ/ZWJ/LRM/RLM
            | '\u{061C}'                                      // ALM
            | '\u{202A}'..='\u{202E}'                        // LRE/RLE/PDF/LRO/RLO
            | '\u{2060}'..='\u{2069}'                        // WJ/invisibles/isolates
            | '\u{FEFF}'                                      // BOM/ZWNBSP
            | '\u{E0000}'..='\u{E007F}'                      // tag characters
        )
    }
    raw.chars()
        .filter(|c| !c.is_control() && !is_invisible_format(*c))
        .take(128)
        .collect()
}

/// Sanitize a peer-supplied VERSION string for DISPLAY (T14 r10): `Hello.
/// software_version` is required on the wire but its content is entirely
/// peer-controlled — raw newlines/ANSI escapes would let an admitted (or
/// token-bearing) node spoof or erase pairing-terminal output. Keep only the
/// characters a semver can contain (alphanumerics and `.+-_`), capped at 64
/// — a normal version passes through unchanged; anything else degrades
/// visibly rather than executing in someone's terminal. An all-filtered
/// input yields the empty string, which callers treat as ABSENT.
pub fn sanitize_version(raw: &str) -> String {
    raw.chars()
        .filter(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '+' | '-' | '_'))
        .take(64)
        .collect()
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
        // `fleet_events` (T10): this node pushes N_FLEET_EVENT worker-state changes.
        // The hub keys its chat-end debounced re-pull fallback on the ABSENCE of this.
        ("fleet_events", true),
        ("update", false),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), serde_json::Value::Bool(v)))
    .collect()
}

/// The capabilities a hub advertises in its HELLO result.
pub fn hub_capabilities() -> Capabilities {
    [
        ("update_push", true),
        ("log_aggregate", true),
        // `fleet_events` (T10): this hub consumes N_FLEET_EVENT pushes. Informational —
        // a node pushes regardless; an older hub's notification reader skips the method.
        ("fleet_events", true),
    ]
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
    /// The EFFECTIVE context window the worker was loaded with — the crate's
    /// OPERATIVE window: the value the load pinned (explicit param, else
    /// trained-cap default, else the worker's own fallback; `node/runtime.rs`
    /// `effective_ctx`) and the one every fit-check enforces. llama.cpp may
    /// pad its internal allocation upward (`n_ctx` to a 256 multiple) — the
    /// pad is unobservable through any higgs surface. From the node's
    /// load-time cache: no RPC to a possibly-busy worker. Absent ONLY from
    /// pre-stats nodes (`serde(default)` — additive, no protocol bump); a
    /// current node always knows it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[ts(optional)]
    pub ctx_len: Option<u32>,
    /// GPU offload the load requested (absent = the worker default, all).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[ts(optional)]
    pub gpu_layers: Option<crate::worker::engine::GpuLayers>,
    /// Generation threads the load requested (absent = the worker default).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[ts(optional)]
    pub threads: Option<u32>,
    /// Wall-clock ms (Unix epoch) when this worker id FIRST loaded its model.
    /// A crash-respawn (the Supervisor restart FSM replaying the load) keeps
    /// the original stamp — "when did this worker come up", not "when did the
    /// child process last restart". `ts(type = "number")` like every other
    /// wire u64 (`system.rs`): the JSON value is a number, and epoch ms sit
    /// far below 2^53.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[ts(optional, type = "number")]
    pub loaded_at_ms: Option<u64>,
    /// Milliseconds since the worker's last chat activity, measured at
    /// snapshot time (the idle reaper's own clock). Freshness is
    /// event-driven: the hub re-pulls inventory on connect and after
    /// lifecycle ops, so this ages between pulls. `ts(type = "number")` as
    /// above.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[ts(optional, type = "number")]
    pub idle_ms: Option<u64>,
    /// Chats in flight on this worker at snapshot time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[ts(optional)]
    pub in_flight: Option<u32>,
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
    /// Node-side MONOTONIC snapshot sequence (T14 r17): incremented on the
    /// node ACTOR for every inventory snapshot, so it is true DATA order —
    /// hub-side pull stamps are not (concurrent QUIC streams can be served
    /// out of order, letting an earlier-stamped pull carry NEWER data). The
    /// hub's commit guard prefers this when both sides carry one; absent from
    /// pre-r17 nodes (`serde(default)` — additive, no protocol bump).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[ts(optional, type = "number")]
    pub snapshot_seq: Option<u64>,
}
}

higgs_const_enum! {
    /// What changed on a node, carried by every [`NodeFleetEvent`] and re-broadcast
    /// hub-side (with hub-local kinds) as a [`crate::node::fleet::FleetEvent`] for
    /// live UIs. Wire values are `snake_case`. Extensible: a reader ignores an
    /// event whose kind it can't decode (additive, no protocol bump).
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum FleetEventKind {
        /// A chat began on a worker (`in_flight` rose).
        ChatStart,
        /// A chat finished on a worker (`in_flight` fell, idle clock restarted).
        ChatEnd,
        /// A worker finished loading its model and is resident.
        WorkerLoaded,
        /// A worker was unloaded/killed/idle-reaped and is gone.
        WorkerUnloaded,
        /// A fresh whole-state snapshot re-sync, NOT a specific transition (T10
        /// r26): the node's relay emits this when recovering a possibly-lost
        /// delivery after a stream failure — reusing the lost event's kind would
        /// let a later, idle snapshot masquerade as e.g. a final ChatStart.
        Resync,
        /// Hub-local (never on the node wire): the hub's fleet networking was
        /// enabled or disabled (the kill switch, T10 r11). Carried with an EMPTY
        /// `endpoint_id` — a whole-fleet invalidation, not a per-node one.
        HubStateChanged,
        /// Hub-local (never on the node wire): the node's connection was admitted.
        /// Fires BEFORE the connect-time inventory pull — the cache still shows the
        /// pre-connect state (or nothing); [`FleetEventKind::InventorySynced`] follows
        /// once the pull commits.
        NodeConnected,
        /// Hub-local (never on the node wire): the hub's own view of this node
        /// changed — an inventory pull COMMITTED (connect-time or lifecycle; the
        /// cache now shows the node's current state, T10 r1/r3), or the hub's
        /// routing table for the node changed (a served id appeared/disappeared,
        /// r10). Subscribers re-read the fleet view on receipt.
        InventorySynced,
        /// Hub-local (never on the node wire): the node retired or its connection dropped.
        NodeDropped,
    }
}

/// Params of one [`N_FLEET_EVENT`] notification: the state-change kind plus the
/// FULL post-change worker snapshot, sequenced by the node actor's `snapshot_seq`
/// (same counter as [`NodeInventory::snapshot_seq`], bumped in the same actor turn
/// as the snapshot — mailbox order IS data order, across pushes AND pulls). Carrying
/// the whole (small) worker list instead of a delta means the hub's cache merge is a
/// guarded replace — there is no per-kind patch logic to drift, and a lost/lagged
/// event self-heals on the next one.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeFleetEvent {
    pub kind: FleetEventKind,
    pub snapshot_seq: u64,
    pub workers: Vec<InventoryWorker>,
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
