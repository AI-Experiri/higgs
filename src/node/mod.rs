//! Node + hub iroh transport: bind, accept-loop gate, dial. Built out across P1–P3.
//!
//! P1 scope: the HELLO handshake. The hub gates one accepted connection
//! (`gate_connection`): read HELLO within a deadline, negotiate a version, admit by
//! allowlist or a one-time pairing token, persist the pairing, reply. The node dials
//! and sends HELLO first (`dial_and_hello`). Chat/data streams arrive in P2/P3.

pub mod cli;
pub mod control;
pub mod data;
pub mod fleet;
pub mod hub;
pub mod identity;
pub mod node_id;
pub mod release_courier;
pub mod runtime;
pub mod self_update;
pub mod served;
pub mod service;
pub mod transport;
pub mod worker_id;

#[cfg(test)]
mod e2e_tests;
#[cfg(test)]
pub(crate) mod test_support;

use std::time::Duration;

use iroh::endpoint::Connection;
use iroh::{Endpoint, EndpointAddr};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};

use crate::auth::{Allowlist, PairingTokens, TokenError};
use crate::diagnostic::HiggsError;
use crate::remote::{
    hub_capabilities, negotiate_version, node_capabilities, HelloParams, HelloResult, ALPN,
    MIN_SUPPORTED, M_HELLO, M_NODE_LEAVE, PROTOCOL_VERSIONS,
};
use crate::rpc::{self, RpcError, RpcFrame, RpcRequest, RpcResponse};

/// Max time from `accept_bi()` to a complete HELLO before the conn is dropped (HG028).
pub const HELLO_DEADLINE: Duration = Duration::from_secs(5);

/// Max bytes read while waiting for the (pre-auth) HELLO line. Bounds memory a peer
/// can force the hub to buffer before any allowlist/token check. A HELLO is well under
/// this; an over-long frame is treated as a malformed handshake (HG028).
const MAX_HELLO_BYTES: u64 = 64 * 1024;

/// The hub's self-identity sent back to a node in the HELLO result: its canonical `EndpointId`
/// (which the node dials) and its friendly name (`hub-<eid8>(<host>)`, from `config.json`). One
/// struct so the two strings can't be swapped at a call site, and so adding a future field
/// (e.g. a hub display version) doesn't churn every gate signature.
#[derive(Debug, Clone)]
pub struct HubIdentity {
    pub id: String,
    pub name: String,
}

impl HubIdentity {
    /// A hub identity with no friendly name (the back-compat default used by tests and the
    /// pre-naming `link pair` path — the node then sees an empty `hub_name`).
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: String::new(),
        }
    }
}

/// Outcome of gating one inbound connection.
#[derive(Debug, PartialEq, Eq)]
pub enum GateOutcome {
    /// Admitted: an already-allowlisted peer, or a valid pairing token (now burned
    /// and the peer added to the allowlist). `software_version` is the node's
    /// self-reported semver from its HELLO (T14: shown in the Fleet view; empty
    /// only if a peer sent an empty string — the field is required on the wire).
    Admitted {
        agreed_version: u32,
        software_version: String,
        /// Whether the node's HELLO advertised the `fleet_events` capability (T10):
        /// it will push `N_FLEET_EVENT` worker-state changes on a uni stream, so the
        /// hub can skip its chat-end debounced inventory re-pull for it.
        fleet_events: bool,
        /// Whether the node's HELLO advertised the `update_reporting` capability (P4b (d)):
        /// it reports its last self-update failure in `update_failed`. When set, a `None`
        /// report is AUTHORITATIVE (no failure) and clears a stored one; when absent (a
        /// legacy/dev/non-managed binary that cannot report), a `None` PRESERVES the stored
        /// failure so a non-reporting reconnect cannot erase one the operator has not seen.
        reports_update_failures: bool,
        /// The node's last self-update FAILURE from its HELLO, re-sanitized at this trust
        /// boundary. Surfaced in the fleet view so the operator learns WHY a pushed update did
        /// not take (vs. inferring it from `software_version` never advancing). `None` = the last
        /// update succeeded or none was attempted.
        update_failed: Option<crate::remote::UpdateFailed>,
        /// The node's build TARGET triple from its HELLO (sanitized), for release-asset selection
        /// when the hub pushes a self-update. `None` = a legacy node that did not report it (the
        /// hub then cannot pick an asset for it and `fleet_update` skips it).
        target: Option<String>,
        /// The node's acceleration VARIANT from its HELLO (sanitized) — the other half of the
        /// asset selector. `None` = a legacy node that did not report it.
        variant: Option<String>,
        /// Whether the node's HELLO advertised the `update` capability — it ships the
        /// signature-verified self-update PUSH handler, so the hub may push it a signed update.
        /// `fleet_update` pre-filters on this (a node without it is skipped, not push-and-failed).
        update_capable: bool,
    },
    /// Rejected; `code` is the HG diagnostic that explains why (logged at origin).
    Rejected { code: &'static str },
}

/// The diagnostic code to attach to a JSON-RPC error's `data` for a `HiggsError`,
/// preferring the worker's ORIGIN code carried in a `WorkerRpc` (e.g. HG003/HG005/HG018)
/// over the generic boundary code — so the hub maps the true status, exactly as the
/// local boundary does. Shared by the control dispatch and the chat relay.
pub(crate) fn worker_origin_code_data(e: &HiggsError) -> Option<serde_json::Value> {
    use miette::Diagnostic;
    if let HiggsError::WorkerRpc {
        worker_code: Some(code),
        ..
    } = e
    {
        return Some(serde_json::json!({ "code": code }));
    }
    e.code()
        .map(|c| serde_json::json!({ "code": c.to_string() }))
}

/// Maximum size of ONE inbound control/data frame (request line) the node will buffer from a
/// hub-opened stream. A hub is a PAIRED, authenticated peer, but a compromised/buggy one must
/// not be able to OOM a small (2–4 GiB Pi/Jetson) node by streaming gigabytes on a single line
/// before [`rpc::decode`] runs — the same DoS the self-update size caps bound for the inline
/// manifest/artifact.
///
/// This is the FLEET ingress bound, DELIBERATELY DECOUPLED from the HTTP `/v1` body limit
/// ([`crate::serve::MAX_BODY_BYTES`]). It is NOT a function of that limit: an `M_CHAT` frame is
/// not the HTTP body verbatim — `NodeTransport::chat` passes the messages as a pre-serialised
/// `messages_json` STRING (plus `tools_json`/`chat_template_kwargs`) embedded as a JSON
/// `Value::String`, so the body is PARSED and RE-SERIALISED (which can canonicalise/expand values,
/// e.g. `1e15` → `1000000000000000.0`) AND then escaped a second time. The blow-up factor of body
/// → frame is therefore NOT bounded by any small constant, so there is no honest "covers every
/// valid body" invariant to assert. Instead we pick a fixed 64 MiB ceiling purely as an
/// OOM bound: it is far above any REALISTIC chat frame (a 64 MiB frame is a multi-million-token
/// transcript) yet well under the 256 MiB the update path (`MAX_ARTIFACT_BYTES`) already buffers,
/// so a compromised hub cannot OOM a small (2–4 GiB) node with one frame. A chat whose serialised
/// `M_CHAT` frame nonetheless exceeds this — a pathologically-crafted or genuinely enormous
/// request — is rejected at the fleet ingress (its bidi stream is dropped; the node keeps
/// serving). That is intentional: the fleet ingress is a distinct boundary from HTTP, not a
/// mirror of it. (FOLLOW-UP: embedding chat JSON as a `Value` instead of a re-escaped STRING would
/// remove the double-encode and shrink the frame — a transport-efficiency change, out of scope.)
/// Shared read for ALL hub→node methods (chat, pull, control); an `M_NODE_UPDATE` manifest is far
/// smaller (capped at 64 KiB) and its artifact is fetched OUT of band.
///
/// SCOPE: this bounds ONE frame. It does NOT bound the AGGREGATE across many concurrent streams
/// ([`serve_node`] spawns a handler per accepted bidi stream), so a compromised hub opening N
/// streams each buffering near the cap costs ~N × 64 MiB. That aggregate is one facet of the
/// broader "a compromised PAIRED hub directs the node's workload and can exhaust its resources
/// many ways" surface (it can equally flood chats or model loads); a general per-connection
/// admission-control / resource-quota is a separate `serve_node` hardening (FOLLOW-UP), not the
/// per-frame concern here. Note this per-frame cap already IMPROVED the aggregate — before it,
/// each stream's `.lines()` read was itself unbounded.
const MAX_CONTROL_FRAME_BYTES: usize = 64 * 1024 * 1024;

/// Write one RPC frame as an NDJSON line to a stream.
pub(crate) async fn write_frame(
    send: &mut iroh::endpoint::SendStream,
    frame: &RpcFrame,
) -> std::io::Result<()> {
    send.write_all(format!("{}\n", rpc::encode(frame)).as_bytes())
        .await?;
    send.flush().await
}

/// Read the first frame as a HELLO request. Returns `(id, params)` or `None` if the
/// stream ended / the frame was malformed / it was not a HELLO. The caller bounds
/// this (together with `accept_bi`) by the handshake deadline.
async fn read_hello(recv: iroh::endpoint::RecvStream) -> Option<(u64, HelloParams)> {
    // Cap the bytes buffered for the pre-auth HELLO so a peer can't force a large
    // allocation before any allowlist/token check.
    let mut lines = BufReader::new(recv.take(MAX_HELLO_BYTES)).lines();
    let line = match lines.next_line().await {
        Ok(Some(line)) => line,
        _ => return None, // EOF or io error
    };
    match rpc::decode(&line) {
        Ok(RpcFrame::Request(req)) if req.method == M_HELLO => {
            serde_json::from_value::<HelloParams>(req.params)
                .ok()
                .map(|p| (req.id, p))
        }
        _ => None,
    }
}

/// A validated pre-auth handshake from [`gate_read_hello`]: the peer passed the deadline,
/// the identity/role check, and version negotiation. Carries the open send half so the admit
/// step can reply. Held only briefly — the admit decision consumes it.
pub(crate) struct Handshake {
    send: iroh::endpoint::SendStream,
    id: u64,
    hello: HelloParams,
    agreed_version: u32,
}

/// Write a TYPED rejection frame on an open gated bi-stream so the node decodes the specific
/// `HGxxx` code (via [`hub_rejection`]) instead of a bare transport EOF, then finish the
/// stream. FAST and lock-safe: it only buffers the small frame + queues the FIN — the same
/// kind of post-auth write the admit path already does under the lock — and never waits on
/// the peer. Pair with [`close_after_reject`], which does the slow grace-close OUTSIDE any
/// held lock. Used by the post-HELLO gate rejections that reply with a typed frame: version
/// (HG023), bad token (HG022), not-allowlisted (HG024), persist failure (HG040). (Handshake-
/// stalled HG028 and the identity/role-mismatch HG024 close directly, with no frame.)
async fn send_reject_frame(
    send: &mut iroh::endpoint::SendStream,
    id: u64,
    code: &'static str,
    message: String,
) -> GateOutcome {
    let resp = RpcResponse {
        jsonrpc: "2.0".into(),
        id,
        result: None,
        error: Some(RpcError {
            code: -32000,
            message,
            data: Some(serde_json::json!({ "code": code })),
        }),
    };
    let _ = write_frame(send, &RpcFrame::Response(resp)).await;
    let _ = send.finish();
    GateOutcome::Rejected { code }
}

/// Grace-close a rejected connection: wait briefly for the peer to read the rejection frame
/// ([`send_reject_frame`]) — a well-behaved node closes once it has — before forcing the
/// close with `code` as the QUIC reason. MUST run with NO pairing lock held: a slow/malicious
/// peer can stall it the full timeout, so holding `allow`/`tokens` across it would serialize
/// every other admission and token mint/burn behind one rejected join.
pub(crate) async fn close_after_reject(conn: &Connection, code: &'static str) {
    let _ = tokio::time::timeout(Duration::from_secs(2), conn.closed()).await;
    conn.close(0u32.into(), code.as_bytes());
}

/// Hub side, phase 1 (NO allowlist/token locks): accept the peer's HELLO within the deadline
/// and run the lock-free checks — anti-spoof identity and version negotiation. Returns the
/// validated [`Handshake`] or a `Rejected` outcome (connection already closed / replied).
///
/// Kept lock-free so a peer that completes QUIC but stalls HELLO cannot hold the pairing
/// locks for the whole deadline and starve other joins (or `POST /api/higgs/pair`). The
/// lock-needing decision is [`gate_admit`].
pub(crate) async fn gate_read_hello(
    conn: &Connection,
    hello_deadline: Duration,
) -> Result<Handshake, GateOutcome> {
    let peer = conn.remote_id().to_string();

    // Bound the WHOLE pre-HELLO window by the deadline: iroh defers stream creation
    // until the opener writes, so `accept_bi()` itself blocks until the node sends —
    // a silent peer must be caught here, not only at the read (§3.2.1).
    let handshake = tokio::time::timeout(hello_deadline, async {
        let (send, recv) = conn.accept_bi().await.ok()?;
        let (id, hello) = read_hello(recv).await?;
        Some((send, id, hello))
    })
    .await;

    let Ok(Some((mut send, id, hello))) = handshake else {
        let e = HiggsError::HandshakeStalled {
            endpoint_id: peer.clone(),
            window: hello_deadline.as_secs(),
        };
        tracing::warn!(error = %e, "higgs: dropping handshake-stalled peer");
        conn.close(0u32.into(), b"HG028");
        return Err(GateOutcome::Rejected { code: "HG028" });
    };

    // 1. identity: the self-declared node_id MUST equal the TLS-authenticated peer id,
    //    and the role must be "node" — otherwise a peer is spoofing another identity (§4.1).
    if hello.role != "node" || hello.node_id != peer {
        tracing::warn!(
            peer,
            claimed = %hello.node_id,
            role = %hello.role,
            "higgs: rejecting HELLO with mismatched identity/role"
        );
        conn.close(0u32.into(), b"HG024");
        return Err(GateOutcome::Rejected { code: "HG024" });
    }

    // 2. version negotiation (HG023 — fatal). Write a typed RpcError BEFORE closing so the
    //    node sees "you must update", not a bare transport EOF.
    let agreed_version = match negotiate_version(
        &hello.protocol_versions,
        hello.min_supported,
        PROTOCOL_VERSIONS,
        MIN_SUPPORTED,
    ) {
        Ok(v) => v,
        Err(mismatch) => {
            let e = HiggsError::VersionMismatch {
                peer: mismatch.peer,
                ours: mismatch.ours,
            };
            tracing::warn!(error = %e, "higgs: rejecting version-mismatched peer");
            // Lock-free here (phase 1), so the grace-close can run inline.
            let out = send_reject_frame(&mut send, id, "HG023", e.to_string()).await;
            close_after_reject(conn, "HG023").await;
            return Err(out);
        }
    };

    Ok(Handshake {
        send,
        id,
        hello,
        agreed_version,
    })
}

/// Hub side, phase 2 (allowlist/token locks held by the caller): admit `handshake` by the
/// allowlist or a one-time pairing token (persisting + burning), then reply `HelloResult`.
/// Synchronous in-memory decision + a small post-auth reply — the only part that needs the
/// pairing locks serialized (§3.2.1). Pair with [`gate_read_hello`].
pub(crate) async fn gate_admit(
    conn: &Connection,
    handshake: Handshake,
    allow: &mut Allowlist,
    tokens: &mut PairingTokens,
    now_ms: u64,
    hub: &HubIdentity,
    label_for_new: Option<String>,
) -> GateOutcome {
    let Handshake {
        mut send,
        id,
        hello,
        agreed_version,
    } = handshake;
    let peer = conn.remote_id().to_string();

    // The label for a FIRST-time pairing is the node's own friendly name (so the fleet view
    // shows a human name immediately), falling back to the caller's `label_for_new` when an
    // older node sends no name.
    let new_label = if hello.name.is_empty() {
        label_for_new
    } else {
        Some(hello.name.clone())
    };

    // 3. allowlist OR a valid one-time pairing token (the only path that admits a
    //    not-yet-allowlisted id). `assigned_label` is the persisted label for an existing
    //    pairing, or the new label on first join.
    let assigned_label = if allow.contains(&peer) {
        allow.label(&peer)
    } else {
        match hello.pairing_token.as_deref() {
            Some(tok) => match tokens.validate(tok, now_ms) {
                Ok(()) => {
                    // Persist FIRST, then burn — a failed save must leave the token usable.
                    if let Err(e) = allow.add(peer.clone(), new_label.clone()) {
                        // A failure HERE is a STORAGE problem (disk full, permissions, a
                        // missing parent in a custom path), NOT an auth one — the token was
                        // VALID. Surface the pairings-store WRITE code (HG040) so the operator
                        // gets the disk/permissions remediation, not a misleading "not
                        // allowlisted" (HG024). The token is NOT burned (burn is below), so a
                        // retry after fixing storage still admits the node.
                        tracing::error!(error = %e, "higgs: failed to persist new pairing");
                        // Locks are HELD here — write the frame only (fast); the caller
                        // grace-closes via `close_after_reject` AFTER releasing the locks.
                        return send_reject_frame(&mut send, id, "HG040", e.to_string()).await;
                    }
                    tokens.burn(tok);
                    new_label
                }
                Err(TokenError::Expired) | Err(TokenError::UnknownOrUsed) => {
                    let e = HiggsError::PairingTokenInvalid {
                        detail: "expired/used/unknown".into(),
                    };
                    tracing::warn!(error = %e, peer, "higgs: rejecting bad pairing token");
                    return send_reject_frame(&mut send, id, "HG022", e.to_string()).await;
                }
            },
            None => {
                let e = HiggsError::NotAllowlisted {
                    endpoint_id: peer.clone(),
                };
                tracing::warn!(error = %e, "higgs: rejecting unknown peer");
                return send_reject_frame(&mut send, id, "HG024", e.to_string()).await;
            }
        }
    };

    // 4. admitted — reply HelloResult (carrying the hub's friendly name so the node can save it).
    let result = HelloResult {
        role: "hub".into(),
        node_id: hub.id.clone(),
        hub_name: hub.name.clone(),
        agreed_version,
        software_version: env!("CARGO_PKG_VERSION").into(),
        assigned_label,
        capabilities: hub_capabilities(),
    };
    let resp = RpcResponse {
        jsonrpc: "2.0".into(),
        id,
        result: Some(serde_json::to_value(result).expect("HelloResult serializes")),
        error: None,
    };
    if let Err(e) = write_frame(&mut send, &RpcFrame::Response(resp)).await {
        tracing::warn!(error = %e, "higgs: failed to send HELLO result");
        return GateOutcome::Rejected { code: "HG027" };
    }
    GateOutcome::Admitted {
        agreed_version,
        // Sanitized at the trust boundary: the string is peer-controlled and
        // flows to terminals (CLI pairing line) and the fleet view.
        software_version: crate::remote::sanitize_version(&hello.software_version),
        // Capability map rule: only an explicit boolean `true` advertises the
        // feature — absent/other shapes read as "doesn't push" (legacy node).
        fleet_events: hello.capabilities.get("fleet_events")
            == Some(&serde_json::Value::Bool(true)),
        // Same rule: an explicit `true` means the node reports its update failures (so a `None` is
        // authoritative); absent/other → a non-reporting node whose `None` must not erase a stored
        // failure.
        reports_update_failures: hello.capabilities.get("update_reporting")
            == Some(&serde_json::Value::Bool(true)),
        // Re-sanitize the peer-supplied failure at the trust boundary (same as software_version):
        // both versions are semver-safe, the reason is display-safe.
        update_failed: hello
            .update_failed
            .as_ref()
            .map(|f| crate::remote::UpdateFailed {
                from: crate::remote::sanitize_version(&f.from),
                to: crate::remote::sanitize_version(&f.to),
                reason: crate::remote::sanitize_display(&f.reason),
            }),
        // Sanitize the build identity at the trust boundary: both flow into a release-asset
        // FILENAME (`higgs-v<ver>-<target>.manifest`) the hub fetches, so `sanitize_version`'s
        // charset (alphanumerics + `.+-_`, cap 64) both preserves a real target triple / variant
        // AND strips any `/`, whitespace, or path metacharacter a hostile node might inject. An
        // input that sanitizes away entirely is treated as ABSENT (the hub then skips the node).
        target: hello
            .target
            .as_deref()
            .map(crate::remote::sanitize_version)
            .filter(|s| !s.is_empty()),
        variant: hello
            .variant
            .as_deref()
            .map(crate::remote::sanitize_version)
            .filter(|s| !s.is_empty()),
        // Capability map rule (same as fleet_events): only an explicit boolean `true` advertises
        // the self-update push handler.
        update_capable: hello.capabilities.get("update") == Some(&serde_json::Value::Bool(true)),
    }
}

/// Hub side: gate one accepted connection (read HELLO + admit). `now_ms`, `hub_id`, and
/// `hello_deadline` are injected so the function is testable; `label_for_new` labels a new
/// pairing. Production passes [`HELLO_DEADLINE`]. This convenience wrapper holds `allow`/
/// `tokens` across the whole call; the production hub accept loop instead uses
/// [`gate_read_hello`] (lock-free) + [`gate_admit`] (locked) so a stalled peer can't starve
/// other joins.
pub async fn gate_connection(
    conn: &Connection,
    allow: &mut Allowlist,
    tokens: &mut PairingTokens,
    now_ms: u64,
    hub: &HubIdentity,
    label_for_new: Option<String>,
    hello_deadline: Duration,
) -> GateOutcome {
    match gate_read_hello(conn, hello_deadline).await {
        Ok(handshake) => {
            // gate_admit (phase 2) writes any rejection FRAME but does not grace-close (it runs
            // under the pairing locks in production). This convenience wrapper holds those locks
            // itself, so it does the grace-close here once gate_admit has returned.
            let outcome =
                gate_admit(conn, handshake, allow, tokens, now_ms, hub, label_for_new).await;
            // Borrow (not move) `outcome` so it can be returned below; the bound `code`
            // (`&&'static str`) auto-derefs to the `&'static str` the helper takes.
            if let GateOutcome::Rejected { code } = &outcome {
                close_after_reject(conn, code).await;
            }
            outcome
        }
        Err(rejected) => rejected,
    }
}

/// Node side: dial `target`, complete HELLO, and return the result — the connection is
/// dropped (one-shot, e.g. `node connect`). For a persistent node use [`connect_node`].
pub async fn dial_and_hello(
    endpoint: &Endpoint,
    target: impl Into<EndpointAddr>,
    self_id: String,
    name: String,
    pairing_token: Option<String>,
) -> std::io::Result<HelloResult> {
    let (_conn, result) = connect_node(endpoint, target, self_id, name, pairing_token).await?;
    Ok(result)
}

/// Surface a hub's error reply to a node request (HELLO / leave) as an `io::Error`.
/// The hub almost always classified its OWN rejection (`[HG023]` version → "update",
/// `[HG024]` not-allowlisted, `[HG022]` token invalid, `[HG040]` persistence →
/// "check disk", …) — that code+remediation is MORE specific than a generic "hub
/// rejected", so it is preserved verbatim. Only a bare/uncoded rejection falls back
/// to `HG039 HubRequestRejected` for `stage`, so the node always shows SOME code.
fn hub_rejection(stage: &str, err: crate::rpc::RpcError) -> std::io::Error {
    if err.message.contains("[HG") {
        std::io::Error::other(err.message)
    } else {
        std::io::Error::other(HiggsError::HubRequestRejected {
            stage: stage.to_owned(),
            detail: err.message,
        })
    }
}

/// Node side: dial `target`, open the control bi-stream, send HELLO first (satisfying
/// iroh's "opener writes first" rule), await the hub's HELLO result, and return the LIVE
/// connection so a persistent node can then [`serve_node`] the hub's control RPCs.
pub async fn connect_node(
    endpoint: &Endpoint,
    target: impl Into<EndpointAddr>,
    self_id: String,
    name: String,
    pairing_token: Option<String>,
) -> std::io::Result<(Connection, HelloResult)> {
    use std::io::Error;
    let conn = endpoint.connect(target, ALPN).await.map_err(Error::other)?;
    let (mut send, recv) = conn.open_bi().await.map_err(Error::other)?;

    // Report the last self-update FAILURE so the hub learns WHY a pushed update did not take — a
    // boot-guard rollback ran at boot, before this connection existed. This is re-reported on EVERY
    // HELLO (report-until-resolved): a valid HELLO reply does NOT prove the hub stored it (a
    // stale-generation admission stores nothing; a one-shot pairing hub discards the field), so
    // clearing on the reply would silently lose the only copy. `reportable_update_failure` is a PURE
    // READ that reports the marker only while its `from` matches the running build and filters a
    // stale one (never deleting it — that would race a detached apply). See its docs.
    let update_bin = crate::node::cli::self_update_bin_dir();
    // `(report, conclusive)`: `conclusive` is false for a non-managed launch (no marker to read) or a
    // managed launch whose marker read was inconclusive (a transient read error).
    let (update_failed, marker_conclusive) = match update_bin.as_deref() {
        Some(bin) => crate::node::self_update::reportable_update_failure(bin),
        None => (None, false),
    };
    // The compiled-in build identity the hub uses to pick a matching release asset when it
    // pushes a self-update (`fleet_update`). All three fields are compile-time constants — nothing
    // is read off disk, so a tampered install dir cannot lie about what is running.
    let build = crate::node::self_update::BuildIdentity::current();
    let params = HelloParams {
        role: "node".into(),
        node_id: self_id,
        name,
        pairing_token,
        protocol_versions: PROTOCOL_VERSIONS.to_vec(),
        min_supported: MIN_SUPPORTED,
        software_version: env!("CARGO_PKG_VERSION").into(),
        update_failed,
        target: Some(build.target),
        variant: Some(build.variant),
        // Advertise `update_reporting` ONLY when this is a managed install AND the marker read was
        // CONCLUSIVE — so the hub trusts a `None` as authoritative only when the node could actually
        // confirm it. A non-managed/dev launch (can't inspect the marker) or an inconclusive read
        // (transient error) omits the capability, so its `None` PRESERVES a stored failure.
        capabilities: node_capabilities(update_bin.is_some() && marker_conclusive),
    };
    let req = RpcRequest {
        jsonrpc: "2.0".into(),
        id: 1,
        method: M_HELLO.into(),
        params: serde_json::to_value(params).map_err(Error::other)?,
    };
    write_frame(&mut send, &RpcFrame::Request(req)).await?;

    // Bound the reply wait: a hub that accepts the stream but never answers must not
    // hang the node forever during startup/pairing.
    let mut lines = BufReader::new(recv).lines();
    let line = match tokio::time::timeout(HELLO_DEADLINE, lines.next_line()).await {
        Ok(Ok(Some(line))) => line,
        Ok(Ok(None)) => {
            return Err(Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "hub closed the stream before replying to HELLO",
            ))
        }
        Ok(Err(e)) => return Err(e),
        Err(_) => {
            return Err(Error::new(
                std::io::ErrorKind::TimedOut,
                "hub did not reply to HELLO within the deadline",
            ))
        }
    };
    let result = match rpc::decode(&line).map_err(Error::other)? {
        RpcFrame::Response(resp) => {
            if let Some(err) = resp.error {
                // Preserve the hub's specific code (version/allowlist/persistence);
                // only an uncoded rejection becomes the generic HG039.
                return Err(hub_rejection("hello", err));
            }
            serde_json::from_value::<HelloResult>(resp.result.unwrap_or_default())
                .map_err(Error::other)?
        }
        // The hub sent something other than a response to our HELLO (HG038).
        other => {
            return Err(Error::other(HiggsError::ProtocolViolation {
                peer_role: "hub".into(),
                detail: format!("unexpected reply frame to HELLO: {other:?}"),
            }))
        }
    };
    // Validate the hub's HELLO before trusting it (see [`validate_hub_hello`]).
    validate_hub_hello(&result, &conn.remote_id().to_string())?;
    // NOTE: the update-failure marker is deliberately NOT cleared here. A valid HELLO reply proves
    // the hub answered, not that it STORED the failure (a stale-generation admission stores nothing;
    // a one-shot pairing hub discards the field), so the node re-reports it on every HELLO until it
    // genuinely resolves — a successful apply clears it, or it becomes stale (the node advanced onto
    // a different build) and `reportable_update_failure` filters it out of the report.
    Ok((conn, result))
}

/// Validate a hub's HELLO reply against the TLS-authenticated peer — SYMMETRIC to the
/// hub-side gate ([`gate_connection`], where a node's self-declared id must equal the TLS
/// peer). `cli.rs` persists `result.node_id` as the saved hub id and re-dials it on every
/// reconnect, so a malformed / incompatible reply must not poison saved state or let the
/// session continue under a protocol we don't speak. Requires: the peer is actually a hub
/// (`role == "hub"`), its self-declared id equals the authenticated EndpointId, and the
/// hub-chosen `agreed_version` is one WE support.
fn validate_hub_hello(result: &HelloResult, peer: &str) -> std::io::Result<()> {
    use std::io::Error;
    if result.role != "hub" || result.node_id != peer {
        return Err(Error::other(HiggsError::ProtocolViolation {
            peer_role: "hub".into(),
            detail: format!(
                "hub HELLO identity mismatch: role={:?}, node_id={:?}, authenticated peer={peer}",
                result.role, result.node_id
            ),
        }));
    }
    if !PROTOCOL_VERSIONS.contains(&result.agreed_version) {
        return Err(Error::other(HiggsError::VersionMismatch {
            peer: vec![result.agreed_version],
            ours: PROTOCOL_VERSIONS.to_vec(),
        }));
    }
    Ok(())
}

/// Node side: ask the hub (on an established post-HELLO connection) to retire THIS node. Opens a
/// bi stream, sends `M_NODE_LEAVE`, and awaits the hub's ack. The hub authenticates by the
/// connection's TLS id, so no node id is sent. `Ok(())` once the hub confirms the retire; an
/// error reply, a closed stream, or a timeout is an `Err`. The caller then forgets the saved hub.
pub async fn send_leave(conn: &Connection) -> std::io::Result<()> {
    use std::io::Error;
    let (mut send, recv) = conn.open_bi().await.map_err(Error::other)?;
    let req = RpcRequest {
        jsonrpc: "2.0".into(),
        id: 1,
        method: M_NODE_LEAVE.into(),
        params: serde_json::json!({}),
    };
    write_frame(&mut send, &RpcFrame::Request(req)).await?;
    let _ = send.finish();

    let mut lines = BufReader::new(recv).lines();
    let line = match tokio::time::timeout(HELLO_DEADLINE, lines.next_line()).await {
        Ok(Ok(Some(line))) => line,
        Ok(Ok(None)) => {
            return Err(Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "hub closed the stream before acking leave",
            ))
        }
        Ok(Err(e)) => return Err(e),
        Err(_) => {
            return Err(Error::new(
                std::io::ErrorKind::TimedOut,
                "hub did not ack leave within the deadline",
            ))
        }
    };
    match rpc::decode(&line).map_err(Error::other)? {
        RpcFrame::Response(resp) => match resp.error {
            // Preserve the hub's specific code (e.g. HG040 persistence → "check disk");
            // only an uncoded rejection becomes the generic HG039.
            Some(err) => Err(hub_rejection("leave", err)),
            None => Ok(()),
        },
        // The hub sent something other than a response to our leave (HG038).
        other => Err(Error::other(HiggsError::ProtocolViolation {
            peer_role: "hub".into(),
            detail: format!("unexpected reply to leave: {other:?}"),
        })),
    }
}

/// Node side (persistent): serve the hub's control RPCs. After HELLO the hub opens a bi
/// stream per request batch; the node accepts each and dispatches frames on it until the
/// stream or connection closes. Control (`higgs/node/*`) routes to the `NodeRuntime`; the
/// chat data relay (`higgs/chat`) lands in P3. Returns when the connection closes.
pub async fn serve_node(conn: Connection, rt: std::sync::Arc<crate::node::runtime::NodeRuntime>) {
    // Relay resident workers' stderr to the hub on a dedicated uni stream for THIS
    // connection. Runs until the connection closes (a reconnect starts a fresh relay).
    /// Abort a spawned relay on EVERY exit of `serve_node` — including the
    /// future being CANCELLED (T10 r23): a trailing explicit `abort()` runs only
    /// on the normal accept-loop exit, so an embedder dropping `serve_node`
    /// mid-flight would leak the relay task, its uni stream, and its broadcast
    /// receiver — still pushing events on a session whose request loop is gone.
    struct AbortOnDrop(tokio::task::JoinHandle<()>);
    impl Drop for AbortOnDrop {
        fn drop(&mut self) {
            self.0.abort();
        }
    }
    let _relay = AbortOnDrop(tokio::spawn(relay_worker_logs(conn.clone(), rt.clone())));
    // Fleet events (T10) get their OWN uni stream: one stream keeps the events in
    // order; separate from the log stream so a log burst can't delay a state change.
    // Subscribe HERE, before the accept loop below runs (T10 r2 #3): the actor drops
    // events with no receiver, and every hub-induced state change arrives through
    // streams this loop accepts — subscribing inside the spawned relay would leave a
    // scheduling window where the first op's event (e.g. an immediate post-reconnect
    // ChatStart) is emitted before the relay task is ever polled, and lost.
    let fleet_events = rt.subscribe_fleet_events();
    let _fleet_relay = AbortOnDrop(tokio::spawn(relay_fleet_events(
        conn.clone(),
        rt.clone(),
        fleet_events,
    )));
    // Each iteration accepts a hub-opened stream; the loop ends when the connection
    // closes (caller decides whether to reconnect).
    while let Ok((send, recv)) = conn.accept_bi().await {
        let rt = rt.clone();
        let conn = conn.clone();
        // Count the handler for the update-restart drain (P4b (c)): a chat's `in_flight`
        // drops when the GENERATION ends, but this detached handler may still be flushing
        // buffered chunks + the final response frame — the drain waits for the guard too,
        // so a re-exec cannot truncate a completed generation's tail. The guard drops when
        // the handler finishes (or is cancelled); `handle_node_stream` awaits the peer's ACK
        // of all buffered data (finish + stopped) before returning, so guard release means
        // DELIVERED, not merely written.
        let guard = rt.stream_guard();
        tokio::spawn(async move {
            let _guard = guard;
            handle_node_stream(rt, conn, send, recv).await;
        });
    }
    // Both relays abort via their drop guards — on this normal exit AND on a
    // cancelled `serve_node` future alike. The per-stream handlers spawned above
    // stay DETACHED (pre-T10 design): a cancel during a long chat/pull leaves
    // that handler holding the connection until it finishes, with the relays
    // gone (r24 #1) — bounded self-heal: the hub's next control op on the
    // acceptor-less connection times out and drops the transport (NodeDropped),
    // restoring pull-model behavior; the daemon never cancels serve_node while
    // keeping the connection open.
}

/// Node side: drain the runtime's per-worker log relay onto a uni stream to the hub as
/// `N_LOG_LINE` notifications. Returns when the connection drops (the uni write fails) or
/// the runtime's relay sender goes away. Best-effort: a lagged hub drops the gap.
async fn relay_worker_logs(
    conn: Connection,
    rt: std::sync::Arc<crate::node::runtime::NodeRuntime>,
) {
    let Ok(mut send) = conn.open_uni().await else {
        return;
    };
    let mut logs = rt.subscribe_logs();
    loop {
        let (worker_id, line) = match logs.recv().await {
            Ok(entry) => entry,
            Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
            Err(tokio::sync::broadcast::error::RecvError::Closed) => return,
        };
        let note = crate::rpc::RpcNotification {
            jsonrpc: "2.0".into(),
            method: crate::remote::N_LOG_LINE.into(),
            params: serde_json::json!({ "worker_id": worker_id.0, "line": line }),
        };
        if write_frame(&mut send, &RpcFrame::Notification(note))
            .await
            .is_err()
        {
            return; // connection gone
        }
    }
}

/// Node side: drain the runtime's fleet events onto a dedicated uni stream to the hub as
/// `N_FLEET_EVENT` notifications (T10). Returns when the connection drops (the uni write
/// fails) or the runtime goes away. Best-effort: a lagged subscriber SKIPS the gap — each
/// event carries the full current worker snapshot, so the next one self-heals, and the
/// hub's pull paths remain as fallback.
/// The relay must outlive a STREAM-level failure (T10 r1 #2): the hub skips its
/// chat-end re-pull for an event-advertising node, so if this relay died on a
/// reset uni stream while the QUIC connection stayed healthy, later state
/// changes would reach the hub by NEITHER push NOR pull — a permanently stale
/// cache. So a stream failure reopens a fresh uni stream and RESENDS the event
/// in flight (every event carries the full worker list, so one resend fully
/// re-syncs); only a failed OPEN — the connection itself is gone, the caller's
/// accept loop is ending too — returns. The failure is detected two ways: a
/// write error, and `send.stopped()` observed WHILE IDLE between events (T10
/// r2 #2) — a reset arriving after the FINAL event was locally buffered but
/// before the hub decoded it would otherwise never surface (no later write to
/// fail), leaving the hub pinned on the pre-reset state (e.g. `in_flight: 1`
/// forever after a lost ChatEnd); on that idle reset the LAST-sent event is
/// resent on the fresh stream, restoring the hub to the current snapshot.
/// `events` is subscribed by the CALLER, before any hub op can run (r2 #3).
async fn relay_fleet_events(
    conn: Connection,
    rt: std::sync::Arc<crate::node::runtime::NodeRuntime>,
    mut events: tokio::sync::broadcast::Receiver<crate::remote::NodeFleetEvent>,
) {
    // The kind owed a FRESH re-snapshot on the next stream (T10 r25): a write
    // failed, or an idle reset may have eaten the last delivery. The stale event
    // itself is NOT resent — its idle_ms froze at capture, and the hub stamps at
    // receipt, so re-delivering it after a stall would read falsely fresh (the
    // one error direction the T14 freshness design forbids). A fresh actor
    // snapshot carries current idle_ms and the next seq.
    let mut resnap = false;
    // Whether anything was written to the CURRENT stream (nothing to recover
    // otherwise).
    let mut sent_any = false;
    'stream: loop {
        let Ok(mut send) = conn.open_uni().await else {
            return; // connection gone
        };
        if std::mem::take(&mut resnap) {
            rt.request_fleet_resnapshot(); // a Resync arrives via the broadcast below
        }
        loop {
            let ev = {
                tokio::select! {
                    got = events.recv() => match got {
                        Ok(ev) => ev,
                        Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                        Err(tokio::sync::broadcast::error::RecvError::Closed) => return,
                    },
                    // Idle reset: the hub stopped THIS stream while no event was
                    // flowing — the last write may not have been decoded. Reopen
                    // and re-snapshot (nothing to do if nothing was ever sent).
                    _ = send.stopped() => {
                        resnap = std::mem::take(&mut sent_any);
                        tokio::time::sleep(Duration::from_millis(200)).await;
                        continue 'stream;
                    }
                }
            };
            let Ok(params) = serde_json::to_value(&ev) else {
                continue;
            };
            let note = crate::rpc::RpcNotification {
                jsonrpc: "2.0".into(),
                method: crate::remote::N_FLEET_EVENT.into(),
                params,
            };
            if write_frame(&mut send, &RpcFrame::Notification(note))
                .await
                .is_err()
            {
                // Stream failed with the connection possibly alive: reopen and
                // re-snapshot. The brief pause keeps a hub that STOP_SENDINGs
                // every stream from turning this into a hot loop; if the
                // connection is actually dead, the reopen fails and we exit
                // above.
                resnap = true;
                sent_any = false;
                tokio::time::sleep(Duration::from_millis(200)).await;
                continue 'stream;
            }
            sent_any = true;
        }
    }
}

/// Dispatch every frame on one hub-opened stream until it ends.
async fn handle_node_stream(
    rt: std::sync::Arc<crate::node::runtime::NodeRuntime>,
    conn: Connection,
    mut send: iroh::endpoint::SendStream,
    recv: iroh::endpoint::RecvStream,
) {
    // Bounded frame read (NOT `.lines()`, which grows one line UNBOUNDED): a huge single frame
    // from a compromised hub returns an error here → the `while let Ok(Some(..))` exits → this
    // stream is dropped, but the node survives (see `MAX_CONTROL_FRAME_BYTES`).
    let mut reader = BufReader::new(recv);
    while let Ok(Some(line)) = rpc::read_bounded_frame(&mut reader, MAX_CONTROL_FRAME_BYTES).await {
        let req = match rpc::decode(&line) {
            Ok(RpcFrame::Request(r)) => r,
            _ => continue, // ignore non-request frames on this direction
        };
        if req.method == crate::worker::M_CHAT {
            // DATA plane: relay the chat to the worker's Supervisor and stream chunks +
            // final back. `relay_chat` owns the writes and its own cancellation.
            crate::node::data::relay_chat(&rt, &conn, &mut send, req).await;
            continue;
        }
        if req.method == crate::remote::M_NODE_PULL {
            // DATA plane: download a GGUF into ~/.higgs/models/, streaming N_PROGRESS.
            crate::node::data::relay_pull(&conn, &mut send, req).await;
            continue;
        }
        if req.method == crate::remote::M_NODE_UPDATE {
            // Hub-PUSHED self-update: write the `"accepted"` reply FIRST, then start applying — the
            // apply is a DEFERRED side effect that must not precede its own reply. A successful
            // `write_frame` means the reply was BUFFERED to QUIC, NOT that the hub RECEIVED it
            // (Quinn's `flush` is a no-op; we do not await a transport ACK), so this is NOT a
            // delivery guarantee — it only skips the apply when the write ERRORS (a clearly-dead
            // stream). The AUTHORITATIVE outcome is EVENTUAL: the node's next HELLO
            // `software_version` (which the fleet tracks) reflects whether the update took. A hub
            // that never receives the reply treats it as OUTCOME-UNKNOWN and retries; the apply is
            // IDEMPOTENT — `apply_pushed_update` refuses an equal/older version and refuses stacking
            // on an unconfirmed trial — so a retry after the node already applied is safely refused.
            let (resp, deferred) = crate::node::control::accept_node_update(&req);
            let reply_buffered = write_frame(&mut send, &RpcFrame::Response(resp))
                .await
                .is_ok();
            if reply_buffered {
                if let Some(update) = deferred {
                    update.spawn();
                }
            }
            continue;
        }
        let resp = if req.method.starts_with("higgs/node/") {
            // CONTROL plane. Tie the dispatch to BOTH connection and this stream's
            // liveness: if the hub drops the connection, or resets/abandons just this
            // stream, cancel the in-flight request so a partial `load` triggers
            // StopOnDrop cleanup instead of orphaning a worker whose id never reached
            // the hub.
            tokio::select! {
                r = crate::node::control::dispatch_node_control(&rt, req) => r,
                _ = conn.closed() => return,
                _ = send.stopped() => return,
            }
        } else {
            // Unknown method = a protocol skew (HG037, → 501): the shared helper keeps
            // -32601 and rides HG037 in data.code.
            RpcResponse {
                jsonrpc: "2.0".into(),
                id: req.id,
                result: None,
                error: Some(crate::rpc::method_not_found("node", &req.method)),
            }
        };
        if write_frame(&mut send, &RpcFrame::Response(resp))
            .await
            .is_err()
        {
            break; // stream gone
        }
    }
    // The update-restart drain (P4b (c)) waits on this handler's stream guard so a re-exec cannot
    // truncate a completed generation's tail. But a dropped `SendStream` only keeps retransmitting
    // buffered data UNTIL the connection closes (quinn `SendStream` docs) — and the re-exec closes
    // it. `write_frame`'s `flush` is a no-op and awaits no transport ACK, so at this point the tail
    // may be buffered-but-unacknowledged. FINISH the stream and await the peer's acknowledgement of
    // ALL data before returning — the guard drops only after delivery, so the drain (and the
    // re-exec it precedes) waits for the tail to actually land. On a severed/reset stream `finish`
    // and `stopped` return promptly with an error we ignore — nothing is left to deliver. The wait
    // is bounded by the caller's drain deadline (the handler is detached; an unresponsive peer just
    // holds `active_streams` until the 60s deadline, then the pre-drain truncation applies as ever).
    let _ = send.finish();
    let _ = send.stopped().await;
}

#[cfg(test)]
mod hub_hello_tests {
    use super::*;

    fn hub_hello(role: &str, node_id: &str, agreed_version: u32) -> HelloResult {
        HelloResult {
            role: role.into(),
            node_id: node_id.into(),
            hub_name: "hub-test".into(),
            agreed_version,
            software_version: "0.0.0".into(),
            assigned_label: None,
            capabilities: crate::remote::Capabilities::default(),
        }
    }

    #[test]
    fn accepts_a_well_formed_hub_hello() {
        let r = hub_hello("hub", "peer-eid", PROTOCOL_VERSIONS[0]);
        assert!(validate_hub_hello(&r, "peer-eid").is_ok());
    }

    #[test]
    fn rejects_wrong_role_or_mismatched_id() {
        // role is not "hub" → ProtocolViolation (HG038).
        let bad_role = hub_hello("node", "peer-eid", PROTOCOL_VERSIONS[0]);
        let e = validate_hub_hello(&bad_role, "peer-eid").unwrap_err();
        assert!(e.to_string().contains("[HG038]"), "{e}");
        // node_id ≠ the authenticated peer → ProtocolViolation (would poison the saved hub id).
        let bad_id = hub_hello("hub", "claimed-other", PROTOCOL_VERSIONS[0]);
        let e = validate_hub_hello(&bad_id, "peer-eid").unwrap_err();
        assert!(e.to_string().contains("[HG038]"), "{e}");
    }

    #[test]
    fn rejects_unsupported_agreed_version() {
        let bad_ver = hub_hello("hub", "peer-eid", 999);
        let e = validate_hub_hello(&bad_ver, "peer-eid").unwrap_err();
        assert!(e.to_string().contains("[HG023]"), "{e}");
    }
}
