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

/// `M_NODE_LOGS` — hub → node: the node's OWN DAEMON log (the `LogSource::Serve` lines
/// that also land in node.log — connects, updates, scans, serving events), NEVER worker/
/// model output (that is `N_LOG_LINE`'s job). Params: `{ "n": <u64>, "follow": <bool> }`.
/// Reply on the SAME bi-stream: one [`N_NODE_LOG`] notification per line (the last-`n`
/// snapshot first, then — only when `follow` — live lines), closed by a final Response.
/// A `follow` stream ends when the HUB stops reading (the node sees `send.stopped()` and
/// tears down — no bytes cross iroh once the watcher is gone) or the node shuts down.
/// Gated on the `node_logs` HELLO capability.
pub const M_NODE_LOGS: &str = "higgs/node/logs";

/// `higgs/node/log_level` — CONTROL request (hub → node): change the node
/// daemon's `LogBus` filter LIVE. Params: [`NodeLogControlParams`]
/// (`{verbose?, sections?}`, both fields optional — omit to keep the current
/// value); reply: [`NodeLogControlReply`] with the EFFECTIVE post-apply state.
/// The change is DAEMON-GLOBAL (NL-V decision A): one setting per node, shared
/// by every `M_NODE_LOGS` subscriber. Gated on the `node_log_control` HELLO
/// capability — a legacy node refuses with `method_not_found` and the hub
/// friendly-errors before opening the RPC (mirrors the `pull_status` refusal).
pub const M_NODE_LOG_LEVEL: &str = "higgs/node/log_level";

/// One daemon-log line on an [`M_NODE_LOGS`] reply stream. Params: `{ "line": <string> }`,
/// plus `{ "lagged": <u64> }` marker frames when the node dropped lines to protect the
/// connection (the stream is lossy-by-design under log floods; the marker says so).
pub const N_NODE_LOG: &str = "higgs/node/node_log";

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
/// `higgs/node/update` — the hub PUSHES a signature-verified self-update to this node
/// (DESIGN-remote.md §9, REL-P4e). This build SHIPS the receive+apply handler, so the `update`
/// capability is advertised `true` ([`node_capabilities`]). The node is the authority: on receipt
/// it re-verifies the CI-signed manifest against its COMPILED-IN pubkeys ([`crate::update`],
/// `HIGGS_UPDATE_PUBKEYS`), re-hashes the artifact (sha256), enforces eligibility (target/variant
/// match, UPGRADE-ONLY — never a downgrade), strict-SSRF-vets the artifact URL, then stages +
/// trial-flips + re-execs with boot-guard rollback. It is dispatched in
/// [`crate::node`]'s node-stream handler (`accept_node_update` → `DeferredUpdate::spawn` →
/// `apply_pushed_update`), NOT the control-op table; the node ACKs the PUSH synchronously and
/// verifies/applies in a detached job. A LEGACY node that predates this handler refuses with a
/// typed `HG026`; a node NOT running from a managed `bin/v<ver>/current` install layout accepts
/// then fails `HG087` (cannot locate its install dir, checked BEFORE any key check); a MANAGED dev
/// build with no compiled-in pubkeys fails closed `HG081`.
pub const M_NODE_UPDATE: &str = "higgs/node/update";
/// `higgs/node/update_version` — the hub tells this node to update itself to an EXACT release
/// VERSION; the node does its OWN download. Params: [`NodeUpdateVersionParams`] (`{version}`,
/// plain semver, no `v` prefix). The node derives the manifest/`.minisig`/artifact URLs from
/// its OWN configured `release_url` (`config.json`, default = this repo's GitHub releases) and
/// the CI `higgs-v<ver>-<suffix>` naming for its OWN compiled target/variant, fetches them
/// (https-only, redirect-tolerant — GitHub 302s release assets to storage; every hop must be
/// https and the fetch is size/time-bounded), then runs the SAME verify+apply pipeline as every
/// other update source: CI signature against COMPILED-IN pubkeys, sha256, eligibility
/// (target/variant match, UPGRADE-ONLY), stage, trial-flip, re-exec, boot-guard rollback. The
/// hub is a pure trigger — it supplies ONE semver string and no URL, so a compromised hub can
/// name a version but can never choose WHERE the node fetches from or forge WHAT it applies.
/// ACKed synchronously (`{status:"accepted"}`), applied detached; the outcome is the node's
/// next HELLO (version advance or `update_failed`). Advertised by the `update_by_version`
/// capability; a hub must not send it to a node that did not advertise it.
pub const M_NODE_UPDATE_VERSION: &str = "higgs/node/update_version";
/// `higgs/node/pull` — DATA-plane request to download a GGUF from HuggingFace into the node's
/// own `~/.higgs/models/` (P4b). Streams [`N_PROGRESS`] then a final `{ path }`. `HG025` on
/// failure; `HG090` when the SAME (repo, file) is already transferring on the node (one copy
/// per file — wait or cancel it); `HG089` when the transfer was cancelled mid-flight (nothing
/// landed; temp cleanup is the transfer's own drop guard, best-effort — failures are
/// tracing-only). A subsequent `M_NODE_SCAN`/`M_NODE_LOAD` then sees the pulled model.
pub const M_NODE_PULL: &str = "higgs/node/pull";
/// `higgs/node/pull_status` — CONTROL request: every download currently in flight ON the node,
/// with live progress. Reply rows are [`HelloDownload`]:
/// `{ downloads: [{repo, file, downloaded, total?, cancellable}] }` — `cancellable` says
/// whether THIS node's process owns the transfer (its cancel registry can stop it); ledger-only
/// rows from sibling processes are observe-only (`false`). The hub asks this on (re)connect so
/// a transfer that survived a disconnect is CONTINUED (progress shown, duplicate never
/// attempted) instead of silently colliding with [HG090] on a blind re-issue. Advertised by
/// the `pull_status` capability.
pub const M_NODE_PULL_STATUS: &str = "higgs/node/pull_status";
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
    /// higgs build (semver) — informational + M_UPDATE gating: it is how the hub observes a
    /// pushed update's outcome (the next HELLO's version advanced, or `update_failed` below).
    pub software_version: String,
    /// The node's LAST self-update FAILURE, RE-REPORTED on every HELLO until it genuinely resolves
    /// (the node does not clear it on the reply — a valid reply does not prove the hub stored it).
    /// A boot-guard rollback runs at BOOT, before any hub connection exists, so the reconnect
    /// HELLO is the first chance to tell the hub WHY a pushed update did not take (the hub
    /// otherwise only infers failure from `software_version` not advancing). Absent (`None`)
    /// when the last update succeeded, the node advanced off the failed build, or none was
    /// attempted. Additive: an older node omits it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub update_failed: Option<UpdateFailed>,
    /// The node's build TARGET triple (`BuildIdentity::current().target`, sourced from the
    /// compiled-in `HIGGS_BUILD_TARGET`, e.g. `aarch64-apple-darwin`). The hub needs it to pick
    /// the matching release asset when it pushes a self-update: `Higgs::fleet_update` derives the
    /// per-node manifest name `higgs-v<ver>-<suffix>` from `(target, variant)` (the release.yml
    /// naming). Additive: an older node omits it, and the hub then cannot choose an asset for it
    /// (`fleet_update` reports it skipped rather than guessing). Sanitized at the trust boundary.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target: Option<String>,
    /// The node's acceleration VARIANT (`metal`/`cpu`/`cuda`, `BuildIdentity::current().variant`,
    /// in the LOWERCASE manifest/`install.sh` spelling) — the second half of the release-asset
    /// selector. A CUDA artifact on a CPU node fails to load, so the hub must match this before
    /// pushing. Additive: an older node omits it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub variant: Option<String>,
    /// Open capability map (e.g. `chat`, `download`, `log_stream`, `update`).
    /// Defaults to empty so a peer that omits it still parses.
    #[serde(default)]
    pub capabilities: Capabilities,
    /// Downloads currently IN FLIGHT on this node, with live progress — the
    /// node's opening announcement ("hello, I am downloading X, N bytes in")
    /// so a hub reconnecting mid-transfer CONTINUES the download it already
    /// started (shows progress, never re-issues into an [HG090] refusal).
    /// Refreshable while connected via `M_NODE_PULL_STATUS`. Additive: an
    /// older node omits it, an older hub ignores it.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub downloads: Vec<HelloDownload>,
}

higgs_ts! {
/// One in-flight download announced in [`HelloParams::downloads`] (and echoed
/// by `M_NODE_PULL_STATUS`): identity + live byte progress. Surfaced on
/// [`crate::node::fleet::NodeView::downloads`] so the Fleet UI shows a
/// transfer that survived a hub disconnect and the operator continues it.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HelloDownload {
    pub repo: String,
    pub file: String,
    #[ts(type = "number")]
    pub downloaded: u64,
    /// Absent when the server sent no content length.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[ts(optional, type = "number")]
    pub total: Option<u64>,
    /// `true` when the node's OWN process registry holds this transfer
    /// (`node_registry().in_flight()`) and can cancel it via the future
    /// M_NODE_PULL_CANCEL dispatch. `false` when the row is announced from
    /// the machine LEDGER only — another process on the node box
    /// (`higgs download` CLI, embedded hub) is doing the transfer and this
    /// node has no cancel channel into it. The UI hides the cancel button
    /// on non-cancellable rows; a future cancel dispatch refuses them with
    /// a clear "not this process — cancel it where it was started"
    /// message. Additive: default `false` so an older hub decoding a
    /// newer node's payload treats every row as observe-only rather than
    /// falsely offering a cancel that can't take.
    #[serde(default)]
    pub cancellable: bool,
}
}

/// Accept a node-announced in-flight download list at the hub trust boundary
/// (the HELLO `downloads` field and `M_NODE_PULL_STATUS` replies):
/// VALIDATE-or-DROP, never rewrite. `(repo, file)` is the download's
/// addressable IDENTITY — the exact key the node's cancel registry holds and
/// the key an operator acts on from the fleet view — so a lossy display
/// transform (e.g. [`sanitize_display`]'s 128-char cap) would leave a visible
/// row no continue/cancel can ever address. Instead each entry must pass the
/// SAME rules the node itself enforced before registering the pull —
/// [`crate::download::dest_path`] verbatim: `<org>/<model>` + single
/// `*.gguf`, all `[A-Za-z0-9._-]` segments, each within the filesystem's
/// NAME_MAX (one shared predicate on both sides, so a registrable pull is
/// always announceable and vice versa — no drift). A passing identity is
/// pure safe-charset ASCII, hence inherently display-safe VERBATIM; a
/// failing entry cannot correspond to a registered pull on a well-behaved
/// node and is dropped whole. The list is capped at 16 mirroring the
/// producer's bound (HELLO frame protection: 16 × ≤767-byte identities sits
/// far under the 64 KiB frame caps). Progress counters are normalized too:
/// an impossible `total` (zero, or less than `downloaded`) degrades to None
/// rather than reaching a UI percent computation.
pub fn accept_announced_downloads(raw: &[HelloDownload]) -> Vec<HelloDownload> {
    // DEDUP by CASE-FOLDED (repo, file) BEFORE the cap. A faulty/hostile
    // node could fill the 16 slots with duplicates of one key, hiding real
    // entries and breaking the "one operator action key → one UI row"
    // addressability invariant. The fold matches the machine
    // download-lock's key identity (case-variant names are one on-disk
    // file / one lock slot on default case-insensitive APFS); entries are
    // kept VERBATIM — the fold is only the dedup key. First occurrence
    // wins (arrival order).
    let mut seen = std::collections::HashSet::new();
    raw.iter()
        .filter(|d| crate::download::dest_path(std::path::Path::new("."), &d.repo, &d.file).is_ok())
        .filter(|d| seen.insert((d.repo.to_ascii_lowercase(), d.file.to_ascii_lowercase())))
        .take(16)
        .cloned()
        .map(|mut d| {
            // The COUNTERS are node-supplied too: an impossible pair
            // (`total == 0`, or `downloaded > total`) would feed a UI
            // percent/bar a divide-by-zero or >100%. The entry stays — the
            // transfer is real, dropping it would hide a live download — but
            // the inconsistent `total` degrades to None ("length unknown"),
            // which every consumer already renders.
            d.total = d.total.filter(|&t| t > 0 && d.downloaded <= t);
            d
        })
        .collect()
}

/// Sanitize a peer-supplied DISPLAY NAME for a terminal (T14 r22/r23): names
/// (`hub-<eid8>(<host>)`, friendly labels) legitimately contain spaces,
/// punctuation, and non-ASCII hostnames, so unlike [`sanitize_version`] this
/// strips only what can SPOOF a terminal or a JSON/HTML renderer — control
/// characters (Cc: ANSI escapes, CR/LF), the LINE/PARAGRAPH separators (Zl/Zp:
/// U+2028/U+2029, which a JS/JSON renderer treats as a line break even though
/// JSON allows them raw), AND the bidi-override/format characters `is_control`
/// misses (Cf: RLO/LRO/isolates/marks/soft-hyphen, which visually reorder or
/// hide printed text) — and caps the length. Normal names pass intact.
pub fn sanitize_display(raw: &str) -> String {
    // The COMPLETE Unicode 15.1 Format (Cf) category plus the Line/Paragraph separators (Zl/Zp)
    // — the set `char::is_control` (Cc only) misses. Enumerated in full rather than piecemeal so
    // the sanitizer can't be spoofed through an omitted format char (a renderer that honours these
    // reorders/hides/line-breaks the display).
    fn is_invisible_format(c: char) -> bool {
        matches!(
            c,
            '\u{2028}' | '\u{2029}'                          // Zl / Zp — line / paragraph separators
            | '\u{00AD}'                                      // SOFT HYPHEN
            | '\u{0600}'..='\u{0605}'                        // Arabic number/format signs
            | '\u{061C}'                                      // ARABIC LETTER MARK
            | '\u{06DD}'                                      // ARABIC END OF AYAH
            | '\u{070F}'                                      // SYRIAC ABBREVIATION MARK
            | '\u{0890}'..='\u{0891}'                        // Arabic pound / piastre marks
            | '\u{08E2}'                                      // ARABIC DISPUTED END OF AYAH
            | '\u{180E}'                                      // MONGOLIAN VOWEL SEPARATOR
            | '\u{200B}'..='\u{200F}'                        // ZWSP/ZWNJ/ZWJ/LRM/RLM
            | '\u{202A}'..='\u{202E}'                        // LRE/RLE/PDF/LRO/RLO
            | '\u{2060}'..='\u{2064}'                        // WJ + invisible operators
            | '\u{2066}'..='\u{206F}'                        // bidi isolates + deprecated format
            | '\u{FEFF}'                                      // BOM/ZWNBSP
            | '\u{FFF9}'..='\u{FFFB}'                        // interlinear annotation
            | '\u{110BD}' | '\u{110CD}'                      // KAITHI number signs
            | '\u{13430}'..='\u{1343F}'                      // Egyptian hieroglyph format controls
            | '\u{1BCA0}'..='\u{1BCA3}'                      // Shorthand format controls
            | '\u{1D173}'..='\u{1D17A}'                      // Musical symbol format controls
            | '\u{E0001}'                                     // LANGUAGE TAG
            | '\u{E0020}'..='\u{E007F}'                      // tag characters
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

/// The capabilities a node advertises in its HELLO. Decisions keyed on these:
/// `update` → the hub pushes or skips a self-update (`HubFleet`'s `update_capable`);
/// `update_reporting` → the hub treats a `None` `update_failed` as authoritative;
/// `fleet_events` → the hub keys its debounced re-pull fallback on its ABSENCE.
pub fn node_capabilities(reports_update_failures: bool) -> Capabilities {
    let mut caps: Capabilities = [
        ("chat", true),
        // `download` (M_PULL, P4b) and `log_stream` (N_LOG_LINE relay, P4) are implemented.
        ("download", true),
        ("log_stream", true),
        // `fleet_events` (T10): this node pushes N_FLEET_EVENT worker-state changes.
        // The hub keys its chat-end debounced re-pull fallback on the ABSENCE of this.
        ("fleet_events", true),
        // `update` (M_UPDATE, §9): this build ships the signature-verified self-update PUSH
        // handler, so a hub may push a signed update (a dev build still fails closed HG081).
        ("update", true),
        // `update_by_version` (M_NODE_UPDATE_VERSION): this build can be told a bare release
        // VERSION and will fetch + verify + apply it from its OWN configured release_url.
        ("update_by_version", true),
        // `pull_status` (M_NODE_PULL_STATUS): this build reports its in-flight
        // downloads + live progress to a (re)connecting hub.
        ("pull_status", true),
        // `node_logs` (M_NODE_LOGS): this build serves its own DAEMON log (snapshot +
        // follow) to the hub on demand — nothing streams unless a watcher asks.
        ("node_logs", true),
        // `node_log_control` (M_NODE_LOG_LEVEL, NL-V): this build accepts hub-
        // driven runtime changes to its LogBus verbosity + target-prefix section
        // filter. A pre-NL-V node would refuse the op with method_not_found; the
        // hub gates on this capability so the operator sees a friendly error.
        ("node_log_control", true),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), serde_json::Value::Bool(v)))
    .collect();
    // `update_reporting` (P4b (d)) is advertised ONLY when this launch can actually inspect its
    // `.update-lastfail` marker AND that read was CONCLUSIVE — a MANAGED install with a definitive
    // marker read (the caller passes `self_update_bin_dir().is_some() && marker_conclusive`).
    // The hub uses it to know a `None` report is AUTHORITATIVE (no failure) rather than one from a
    // dev/non-managed/legacy binary that simply CANNOT report — advertising it unconditionally would
    // let a copied binary at the same identity, which always reports `None`, erase a stored failure.
    if reports_update_failures {
        caps.insert("update_reporting".into(), serde_json::Value::Bool(true));
    }
    caps
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
/// A node's LAST self-update FAILURE — the node reports it in its next HELLO
/// ([`HelloParams::update_failed`]) and the hub surfaces it in the fleet view
/// ([`crate::node::fleet::NodeView::update_failed`]) so the operator learns WHY a pushed update
/// did not take (vs. inferring it from the version never advancing). Re-reported on EVERY HELLO
/// until the failure genuinely resolves (a successful apply, or the node advancing off the failed
/// version) — a valid HELLO reply does not prove the hub stored it, so the node never clears it on
/// the reply alone.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct UpdateFailed {
    /// The version the node is running NOW — the one it rolled back to, or stayed on when the
    /// apply failed before ever flipping `current`.
    pub from: String,
    /// The version the failed push TARGETED (from the manifest, or the push's `target_version`
    /// when the failure preceded manifest parse). Empty if it could not be determined.
    pub to: String,
    /// A short, sanitized reason — an HG code + phase, e.g. `HG084 artifact sha256 mismatch` or
    /// `crash-looped on boot — rolled back`. Never free-form peer text (see `sanitize_version`).
    pub reason: String,
}
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
    /// The worker's capability class, captured at LOAD time on the node
    /// (`LoadFacts.domain`). The hub reads it to keep non-generative remote
    /// workers out of the `/v1/models` chat union — the node's own `ChatHandle`
    /// [HG079] gate stays the enforcement. `serde(default)` = `Llm`: an OLDER
    /// node never reports it, and the hub deliberately stays permissive about
    /// what it cannot see (additive, no protocol bump).
    #[serde(default)]
    pub domain: crate::worker::models::ModelDomain,
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

/// Hub → node `M_NODE_UPDATE` params (DESIGN-remote §9): a signature-verified self-update
/// PUSH. The (tiny) manifest + its `.minisig` are carried INLINE (`manifest` = the manifest
/// JSON text, `manifest_sig` = the `.minisig` file text); only the (large) artifact is
/// fetched, from the DIRECT `artifact_url` (redirects are not followed, so the hub supplies
/// the final URL). `target_version` + `pinned_key_id` are informational — authenticity comes
/// from verifying `manifest_sig` against the pubkeys compiled into the node, and the artifact
/// from the manifest's `sha256`; a dev build pins no key and refuses (HG081). Not
/// `deny_unknown_fields`: a newer hub may add optional fields an older node should ignore.
/// Params of [`M_NODE_UPDATE_VERSION`] — the version-only update trigger. Not
/// `deny_unknown_fields`: a newer hub may add optional fields an older node should ignore.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeUpdateVersionParams {
    /// The exact release version to update to — plain semver, no `v` prefix (`0.1.0-beta.2`).
    /// The node validates the syntax, derives the asset names for its OWN target/variant, and
    /// enforces UPGRADE-ONLY eligibility after signature verification exactly like a pushed
    /// manifest — a hub can never downgrade a node by naming an old version.
    pub version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeUpdateParams {
    /// The CI-signed update manifest, verbatim JSON text.
    pub manifest: String,
    /// The manifest's detached minisign signature (`.minisig` file text).
    pub manifest_sig: String,
    /// A DIRECT URL to the artifact tarball named by the manifest (fetched + sha256-checked).
    pub artifact_url: String,
    /// Informational: the version the hub believes it is pushing (the manifest is authoritative).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_version: Option<String>,
    /// Informational: which pinned key the hub expects to verify it (the node tries all pinned
    /// keys regardless).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pinned_key_id: Option<String>,
    // NOTE: there is deliberately NO `allow_downgrade` on the wire. A hub-pushed self-update is
    // UPGRADE-ONLY — the node's `apply_pushed_update` always applies with downgrade REFUSED, so a
    // compromised paired hub can never replay an OLD signed release to downgrade a node. Rolling
    // back to a prior version is the node's own LOCAL job (`higgs node self-update --rollback`,
    // or the boot-guard auto-rollback on a crash-loop), never a hub push (DESIGN-remote.md §9).
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

higgs_ts! {
/// [`M_NODE_LOG_LEVEL`] request params (hub → node): flip the node daemon's
/// `LogBus.verbose` gate LIVE. `None` = keep the current value; a bool
/// value overrides it (`true` admits DEBUG/TRACE into the `LogSource::Serve`
/// stream, `false` restores the INFO+ gate). This is the only knob — the
/// section badge that appears at the start of each log line is set at write
/// time by the tracing target, not by the wire.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct NodeLogControlParams {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[ts(optional)]
    pub verbose: Option<bool>,
}
}

higgs_ts! {
/// [`M_NODE_LOG_LEVEL`] reply: the EFFECTIVE post-apply verbosity on the
/// node. Echoed so a caller stays in sync with the daemon's actual state.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct NodeLogControlReply {
    pub verbose: bool,
}
}

#[cfg(test)]
#[path = "remote_tests.rs"]
mod tests;
