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
pub mod runtime;
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
    MIN_SUPPORTED, M_HELLO, PROTOCOL_VERSIONS,
};
use crate::rpc::{self, RpcError, RpcFrame, RpcRequest, RpcResponse};

/// Max time from `accept_bi()` to a complete HELLO before the conn is dropped (HG028).
pub const HELLO_DEADLINE: Duration = Duration::from_secs(5);

/// Max bytes read while waiting for the (pre-auth) HELLO line. Bounds memory a peer
/// can force the hub to buffer before any allowlist/token check. A HELLO is well under
/// this; an over-long frame is treated as a malformed handshake (HG028).
const MAX_HELLO_BYTES: u64 = 64 * 1024;

/// Outcome of gating one inbound connection.
#[derive(Debug, PartialEq, Eq)]
pub enum GateOutcome {
    /// Admitted: an already-allowlisted peer, or a valid pairing token (now burned
    /// and the peer added to the allowlist).
    Admitted { agreed_version: u32 },
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
            let resp = RpcResponse {
                jsonrpc: "2.0".into(),
                id,
                result: None,
                error: Some(RpcError {
                    code: -32000,
                    message: e.to_string(),
                    data: Some(serde_json::json!({ "code": "HG023" })),
                }),
            };
            let _ = write_frame(&mut send, &RpcFrame::Response(resp)).await;
            let _ = send.finish();
            let _ = tokio::time::timeout(Duration::from_secs(2), conn.closed()).await;
            conn.close(0u32.into(), b"HG023");
            return Err(GateOutcome::Rejected { code: "HG023" });
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
    hub_id: String,
    label_for_new: Option<String>,
) -> GateOutcome {
    let Handshake {
        mut send,
        id,
        hello,
        agreed_version,
    } = handshake;
    let peer = conn.remote_id().to_string();

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
                    if let Err(e) = allow.add(peer.clone(), label_for_new.clone()) {
                        tracing::error!(error = %e, "higgs: failed to persist new pairing");
                        conn.close(0u32.into(), b"HG024");
                        return GateOutcome::Rejected { code: "HG024" };
                    }
                    tokens.burn(tok);
                    label_for_new
                }
                Err(TokenError::Expired) | Err(TokenError::UnknownOrUsed) => {
                    let e = HiggsError::PairingTokenInvalid {
                        detail: "expired/used/unknown".into(),
                    };
                    tracing::warn!(error = %e, peer, "higgs: rejecting bad pairing token");
                    conn.close(0u32.into(), b"HG022");
                    return GateOutcome::Rejected { code: "HG022" };
                }
            },
            None => {
                let e = HiggsError::NotAllowlisted {
                    endpoint_id: peer.clone(),
                };
                tracing::warn!(error = %e, "higgs: rejecting unknown peer");
                conn.close(0u32.into(), b"HG024");
                return GateOutcome::Rejected { code: "HG024" };
            }
        }
    };

    // 4. admitted — reply HelloResult.
    let result = HelloResult {
        role: "hub".into(),
        node_id: hub_id,
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
    GateOutcome::Admitted { agreed_version }
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
    hub_id: String,
    label_for_new: Option<String>,
    hello_deadline: Duration,
) -> GateOutcome {
    match gate_read_hello(conn, hello_deadline).await {
        Ok(handshake) => {
            gate_admit(
                conn,
                handshake,
                allow,
                tokens,
                now_ms,
                hub_id,
                label_for_new,
            )
            .await
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
    pairing_token: Option<String>,
) -> std::io::Result<HelloResult> {
    let (_conn, result) = connect_node(endpoint, target, self_id, pairing_token).await?;
    Ok(result)
}

/// Node side: dial `target`, open the control bi-stream, send HELLO first (satisfying
/// iroh's "opener writes first" rule), await the hub's HELLO result, and return the LIVE
/// connection so a persistent node can then [`serve_node`] the hub's control RPCs.
pub async fn connect_node(
    endpoint: &Endpoint,
    target: impl Into<EndpointAddr>,
    self_id: String,
    pairing_token: Option<String>,
) -> std::io::Result<(Connection, HelloResult)> {
    use std::io::Error;
    let conn = endpoint.connect(target, ALPN).await.map_err(Error::other)?;
    let (mut send, recv) = conn.open_bi().await.map_err(Error::other)?;

    let params = HelloParams {
        role: "node".into(),
        node_id: self_id,
        pairing_token,
        protocol_versions: PROTOCOL_VERSIONS.to_vec(),
        min_supported: MIN_SUPPORTED,
        software_version: env!("CARGO_PKG_VERSION").into(),
        capabilities: node_capabilities(),
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
                return Err(Error::other(format!("hub rejected HELLO: {}", err.message)));
            }
            serde_json::from_value::<HelloResult>(resp.result.unwrap_or_default())
                .map_err(Error::other)?
        }
        other => {
            return Err(Error::other(format!(
                "unexpected reply frame to HELLO: {other:?}"
            )))
        }
    };
    Ok((conn, result))
}

/// Node side (persistent): serve the hub's control RPCs. After HELLO the hub opens a bi
/// stream per request batch; the node accepts each and dispatches frames on it until the
/// stream or connection closes. Control (`higgs/node/*`) routes to the `NodeRuntime`; the
/// chat data relay (`higgs/chat`) lands in P3. Returns when the connection closes.
pub async fn serve_node(conn: Connection, rt: std::sync::Arc<crate::node::runtime::NodeRuntime>) {
    // Relay resident workers' stderr to the hub on a dedicated uni stream for THIS
    // connection. Runs until the connection closes (a reconnect starts a fresh relay).
    let relay = tokio::spawn(relay_worker_logs(conn.clone(), rt.clone()));
    // Each iteration accepts a hub-opened stream; the loop ends when the connection
    // closes (caller decides whether to reconnect).
    while let Ok((send, recv)) = conn.accept_bi().await {
        let rt = rt.clone();
        let conn = conn.clone();
        tokio::spawn(handle_node_stream(rt, conn, send, recv));
    }
    relay.abort(); // connection closed — stop relaying logs for it
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

/// Dispatch every frame on one hub-opened stream until it ends.
async fn handle_node_stream(
    rt: std::sync::Arc<crate::node::runtime::NodeRuntime>,
    conn: Connection,
    mut send: iroh::endpoint::SendStream,
    recv: iroh::endpoint::RecvStream,
) {
    let mut lines = BufReader::new(recv).lines();
    while let Ok(Some(line)) = lines.next_line().await {
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
            RpcResponse {
                jsonrpc: "2.0".into(),
                id: req.id,
                result: None,
                error: Some(RpcError {
                    code: -32601,
                    message: format!("unknown method {}", req.method),
                    data: None,
                }),
            }
        };
        if write_frame(&mut send, &RpcFrame::Response(resp))
            .await
            .is_err()
        {
            break; // stream gone
        }
    }
}
