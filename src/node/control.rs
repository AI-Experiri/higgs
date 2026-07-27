//! Node-side CONTROL dispatch: map an inbound `higgs/node/*` RPC request to a
//! `NodeRuntime` op and build the reply (DESIGN-remote.md §5.4a). These never reach a
//! worker's `WorkerState` — they operate on the registry one layer up. The data plane
//! (`M_CHAT`) is dispatched separately (§5.4b, P2 Task 5).

use serde_json::{json, Value};

use crate::diagnostic::HiggsError;
use crate::node::runtime::NodeRuntime;
use crate::node::worker_id::WorkerId;
use crate::remote::{
    NodeLoadParams, NodeLoadResult, WorkerRef, M_NODE_INVENTORY, M_NODE_KILL, M_NODE_LOAD,
    M_NODE_SCAN, M_NODE_STATUS, M_NODE_SYSINFO, M_NODE_UNLOAD,
};
use crate::rpc::{RpcError, RpcRequest, RpcResponse};

/// JSON-RPC error codes used by the control plane (mirrors the worker wire).
/// `-32601` (method-not-found) is now produced by the shared `rpc::method_not_found`
/// helper (which also rides the HG037 code); the named constant remains only for the
/// test that asserts the wire code.
#[cfg(test)]
const METHOD_NOT_FOUND: i64 = -32601;
const INVALID_PARAMS: i64 = -32602;
const INTERNAL_ERROR: i64 = -32000;

/// Dispatch one `higgs/node/*` control request against the node runtime.
pub async fn dispatch_node_control(rt: &NodeRuntime, req: RpcRequest) -> RpcResponse {
    let id = req.id;
    match req.method.as_str() {
        M_NODE_LOAD => match parse::<NodeLoadParams>(&req) {
            Ok(params) => match rt.load(params).await {
                Ok((worker_id, loaded)) => ok(
                    id,
                    NodeLoadResult {
                        worker_id: worker_id.0,
                        loaded,
                    },
                ),
                Err(e) => err_from(id, &e),
            },
            Err(resp) => resp(id),
        },
        M_NODE_UNLOAD => match parse::<WorkerRef>(&req) {
            Ok(w) => match rt.unload(WorkerId(w.worker_id)).await {
                Ok(()) => ok_value(id, json!({})),
                Err(e) => err_from(id, &e),
            },
            Err(resp) => resp(id),
        },
        M_NODE_KILL => match parse::<WorkerRef>(&req) {
            Ok(w) => match rt.kill(WorkerId(w.worker_id)).await {
                Ok(()) => ok_value(id, json!({})),
                Err(e) => err_from(id, &e),
            },
            Err(resp) => resp(id),
        },
        M_NODE_STATUS => match parse::<WorkerRef>(&req) {
            Ok(w) => match rt.status(WorkerId(w.worker_id)).await {
                Ok(v) => ok_value(id, v),
                Err(e) => err_from(id, &e),
            },
            Err(resp) => resp(id),
        },
        M_NODE_SYSINFO => match rt.sysinfo().await {
            Ok(v) => ok_value(id, v),
            Err(e) => err_from(id, &e),
        },
        M_NODE_SCAN => match rt.scan().await {
            Ok(v) => ok_value(id, v),
            Err(e) => err_from(id, &e),
        },
        M_NODE_INVENTORY => match rt.inventory().await {
            Ok(v) => ok_value(id, v),
            Err(e) => err_from(id, &e),
        },
        // NOTE: the hub-PUSHED self-update (`M_NODE_UPDATE`) is NOT handled here. It is the one
        // control op with a DEFERRED side effect (start applying an update), which is started ONLY
        // after its `"accepted"` reply is WRITTEN — so a clearly-dead stream (write error) applies
        // nothing. So `handle_node_stream` special-cases it BEFORE this dispatch (like `M_CHAT`/
        // `M_NODE_PULL`), via [`accept_node_update`] + [`DeferredUpdate::spawn`]. Reaching this arm
        // would be a routing bug; it falls through to method-not-found.

        // Unknown method = a protocol skew (HG037, → 501); reuse the shared helper's
        // message + data.code, keeping the -32601 numeric code.
        other => {
            let mnf = crate::rpc::method_not_found("node", other);
            err(id, mnf.code, mnf.message, mnf.data)
        }
    }
}

/// A verified-shaped hub-push (`M_NODE_UPDATE`) whose apply must run ONLY after the `"accepted"`
/// reply is WRITTEN — see [`accept_node_update`]. Held by `handle_node_stream` across the reply
/// write; on a successful (BUFFERED) write it calls [`Self::spawn`], on a WRITE ERROR (dead stream)
/// it is dropped and nothing is applied. Buffering is not delivery, so a reply LOST in flight can
/// still leave the node applying — that is reconciled by the eventually-consistent, IDEMPOTENT
/// contract on [`accept_node_update`], not by this write, not a synchronous handshake.
pub struct DeferredUpdate {
    params: crate::remote::NodeUpdateParams,
}

impl DeferredUpdate {
    /// Spawn the DETACHED apply (fetch + verify + stage + atomic flip) on a blocking thread; on
    /// success, restart to activate the staged binary. The apply can take MINUTES (network fetch +
    /// smoke), far longer than the hub's control-request timeout — which is exactly why it runs
    /// detached AFTER the immediate `"accepted"` reply, not inline. The single-update lock
    /// (acquired inside `apply_pushed_update` BEFORE the download) makes a CONCURRENT push fail
    /// fast (HG087) instead of each buffering the artifact. The bin dir is this daemon's own
    /// install (`self_update_bin_dir`; not in that layout → HG087); a dev build pins no key → HG081.
    /// The outcome is LOGGED; the hub observes SUCCESS via the node's next HELLO `software_version`.
    pub fn spawn(self) {
        let params = self.params;
        tokio::spawn(async move {
            let applied = tokio::task::spawn_blocking(move || {
                let bin = crate::node::cli::self_update_bin_dir().ok_or_else(|| {
                    HiggsError::UpdateApplyFailed {
                        detail: "cannot locate this node's install dir (needs the \
                                 bin/v<ver>/current layout from install.sh)"
                            .into(),
                    }
                })?;
                // apply_pushed_update records a post-lock failure itself (from = the installed
                // version, to = the AUTHENTICATED manifest version once verified, else this hint)
                // and clears the marker on success, so the connected apply-failure is reported to
                // the hub on the next HELLO with the most accurate target it can prove.
                crate::node::self_update::apply_pushed_update(
                    &bin,
                    &params.manifest,
                    &params.manifest_sig,
                    &params.artifact_url,
                    // A hub-pushed self-update is UPGRADE-ONLY (REL-P4e): downgrade is ALWAYS
                    // refused here, so a compromised paired hub cannot replay an old signed release
                    // to downgrade this node. Rollback is the node's own LOCAL job.
                    false,
                    params.target_version.as_deref().unwrap_or(""),
                )
            })
            .await;
            match applied {
                Ok(Ok((from, to))) => {
                    tracing::info!(
                        from = %from, to = %to,
                        "self-update: staged a hub-pushed update — restarting to activate it \
                         (the boot-guard then confirms it or rolls back)"
                    );
                    // ACTIVATE: the flip only changed `current`; the running process still executes
                    // the OLD binary. Trigger a graceful update restart (see `request_self_restart`)
                    // — the daemon DRAINS in-flight generations (bounded) then re-execs through
                    // `current`; without this the push is a no-op.
                    crate::node::self_update::request_self_restart();
                }
                Ok(Err(e)) => tracing::error!(error = %e, "self-update: hub-pushed update failed"),
                Err(join) => {
                    tracing::error!(join = %join, "self-update: hub-pushed update task did not run")
                }
            }
        });
    }
}

/// Accept a hub-PUSHED self-update (`M_NODE_UPDATE`, §9 P4): parse `NodeUpdateParams` and cap the
/// INLINE manifest+sig SYNCHRONOUSLY (a paired-but-compromised hub could otherwise send a huge
/// string to OOM the node before any copy). Returns the reply to send plus, on a WELL-FORMED push,
/// a [`DeferredUpdate`] the caller runs AFTER the reply is written (bad params / oversized manifest
/// → an error reply and `None`, so nothing is applied). Split from the dispatch table because the
/// apply is a deferred side effect that must not precede its own reply.
///
/// The `"accepted"` reply is BEST-EFFORT — buffering it to QUIC is not a hub-receipt guarantee — so
/// the contract is EVENTUALLY CONSISTENT, not a synchronous handshake: the AUTHORITATIVE outcome is
/// the node's next HELLO `software_version`, and a hub that never receives the reply retries. That
/// retry converges safely because [`crate::node::self_update::apply_pushed_update`] refuses re-work
/// in each post-attempt state: an equal/older version (already applied / restarted onto it), a
/// version staged on an unconfirmed trial (not yet restarted), and a version poisoned by a prior
/// rollback (crash-looped). The rollback poison is itself BEST-EFFORT (recorded AFTER the flip, so
/// it never jeopardises the critical rollback on a full/inode-exhausted disk or mid-crash), so the
/// rolled-back case is EVENTUALLY consistent too: if the poison was lost (a crash in the
/// flip→poison window, or a write failure), a re-push re-applies and SELF-HEALS via the SAME bounded
/// rollback (which retries the poison) rather than looping unboundedly — it does not permanently
/// re-crash, and each cycle rests the node on the known-good previous version.
pub fn accept_node_update(req: &RpcRequest) -> (RpcResponse, Option<DeferredUpdate>) {
    let params = match parse::<crate::remote::NodeUpdateParams>(req) {
        Ok(p) => p,
        Err(resp) => return (resp(req.id), None),
    };
    if let Err(e) =
        crate::node::self_update::check_pushed_sizes(&params.manifest, &params.manifest_sig)
    {
        return (err_from(req.id, &e), None);
    }
    let target = params.target_version.clone().unwrap_or_default();
    let resp = ok_value(
        req.id,
        json!({ "status": "accepted", "target_version": target }),
    );
    (resp, Some(DeferredUpdate { params }))
}

/// Parse params into `T`; on failure return a closure that builds the INVALID_PARAMS
/// response for a given request id (so the message names the parse error).
fn parse<T: serde::de::DeserializeOwned>(
    req: &RpcRequest,
) -> Result<T, impl FnOnce(u64) -> RpcResponse> {
    serde_json::from_value::<T>(req.params.clone()).map_err(|e| {
        let msg = format!("invalid params: {e}");
        move |id: u64| err(id, INVALID_PARAMS, msg, None)
    })
}

fn ok<T: serde::Serialize>(id: u64, value: T) -> RpcResponse {
    ok_value(
        id,
        serde_json::to_value(value).expect("control results serialize"),
    )
}

fn ok_value(id: u64, result: Value) -> RpcResponse {
    RpcResponse {
        jsonrpc: "2.0".into(),
        id,
        result: Some(result),
        error: None,
    }
}

fn err(id: u64, code: i64, message: String, data: Option<Value>) -> RpcResponse {
    RpcResponse {
        jsonrpc: "2.0".into(),
        id,
        result: None,
        error: Some(RpcError {
            code,
            message,
            data,
        }),
    }
}

/// Map a `HiggsError` to an RpcError carrying its HG code in `data` (mirrors the
/// supervisor's worker-error mapping, so the hub can recover the origin status — the
/// worker's own code when the failure came from the worker).
fn err_from(id: u64, e: &HiggsError) -> RpcResponse {
    err(
        id,
        INTERNAL_ERROR,
        e.to_string(),
        crate::node::worker_origin_code_data(e),
    )
}

#[cfg(test)]
#[path = "control_tests.rs"]
mod tests;
