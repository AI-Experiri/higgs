//! HTTP surfaces for higgs, split by concern:
//!
//! - [`v1`]      — strict OpenAI `/v1` (chat + models), `async-openai` types verbatim
//! - [`control`] — higgs's own `/api/higgs/*` surface (load/unload/status/…)
//! - [`wire`]    — the control request/response structs (ts-rs exported)
//! - [`stream`]  — SSE assembly for streaming chat
//!
//! This module owns only the cross-cutting pieces: the [`router`], graceful
//! [`serve_with_shutdown`], CORS, and the shared [`http_status`] mapping.
//!
//! ## Adding a control endpoint
//!
//! 1. Add the response struct to `wire.rs` wrapped in `higgs_ts! { … }` (the
//!    macro injects the `ts_rs::TS` derive + export into
//!    `frontend/src/lib/generated/higgs/`) and re-export it from
//!    `frontend/src/lib/types.ts` via `./generated/higgs/<Name>`.
//! 2. Add `async fn control_<name>` to `control.rs`.
//! 3. Register it in [`router`] under `/api/higgs/<name>`.

mod control;
// `pub` (not `pub(crate)`) because `ModelReadiness` is a field type on the
// publicly re-exported `wire::HiggsModelEntry` (`pub use wire::*`), matching the
// other wire field types' modules (`pub mod engine` / `pub mod models`). Keeps
// the public `serve` surface self-consistent — every wire field type is nameable.
pub mod readiness;
mod stream;
#[cfg(test)]
pub(crate) mod test_support;
mod v1;
mod v1_wire;
pub(crate) mod wire;

use std::sync::Arc;
use std::time::Duration;

use axum::extract::{DefaultBodyLimit, Request};
use axum::http::{HeaderMap, HeaderValue, Method, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::Router;
use tower_http::catch_panic::CatchPanicLayer;
use tower_http::cors::{AllowOrigin, CorsLayer};
use tower_http::timeout::TimeoutLayer;

use crate::api::Higgs;
use crate::diagnostic::HiggsError;

// ── Serve-layer limits ──────────────────────────────────────────────────────
//
// Documented `const`s (not config yet). A later phase lifts the user-facing
// ones (body limit, timeouts, bind) into `HiggsConfig` + the Server Settings UI;
// they are grouped here so that lift is a mechanical move.

/// Maximum accepted request-body size, in bytes. Caps `/v1/chat/completions`
/// (large `messages` arrays) and every control body. 32 MB is generous for
/// image-less chat while bounding memory per request — oversized bodies get a
/// `413 Payload Too Large` from [`DefaultBodyLimit`]. (vllm's default is ~4 MB;
/// ours is larger because a long conversation transcript is legitimately big.)
pub const MAX_BODY_BYTES: usize = 32 * 1024 * 1024;

/// Whole-request timeout for the **non-streaming control surface**
/// (`/api/higgs/*`). Generous (a cold model load can take seconds), but bounded
/// so a wedged control request can't pin a connection forever. Deliberately
/// **not** applied to `/v1/chat/completions`: a long SSE stream must outlive any
/// per-request timeout (its duration is bounded separately by the worker
/// chat-RPC timeout, a different layer).
pub const CONTROL_TIMEOUT: Duration = Duration::from_secs(120);

/// Whole-request timeout for the LONG model-op routes (`POST
/// /api/higgs/models/load` and `/api/higgs/models/tune`), exempt from the tighter
/// [`CONTROL_TIMEOUT`]. A **load** can walk the G5 OOM degrade-retry ladder (the
/// initial load plus several rungs, each a worker load bounded by the control-RPC
/// timeout, with a `SETTLE_BEFORE_RETRY` pause between); a Turbotune **benchmark**
/// loads + measures several candidate configs back-to-back. On a normal-size model
/// both easily exceed two minutes — under `CONTROL_TIMEOUT` the request would be
/// cancelled (408) before the ladder returns [HG060] / a profile is saved. This
/// generous cap is still a backstop: each underlying load/decode is independently
/// bounded by the worker load/RPC timeouts, so a wedged op can't run forever. The
/// fast analytical Suggest mode and a fits-first-time load complete well within it.
pub const LONG_OP_TIMEOUT: Duration = Duration::from_secs(20 * 60);

/// Upper cap on a chat request's generation budget (`max_tokens` /
/// `max_completion_tokens`). A request asking for more is rejected with
/// `400 [HG013]` BEFORE dispatch. Bounds per-request work on the no-auth server
/// (a single request can't pin the worker generating millions of tokens) and
/// keeps the prompt+budget fit-check meaningful. vllm/ollama bound output via
/// `max_tokens` against the model context window rather than a fixed ceiling;
/// higgs additionally enforces this absolute cap so the bound holds regardless
/// of the loaded model. 32768 matches [`DEFAULT_CTX_CAP`] — no single response
/// should exceed the default context window. Documented `const`, grouped for a
/// later config lift.
///
/// [`DEFAULT_CTX_CAP`]: crate::api::DEFAULT_CTX_CAP
pub const MAX_OUTPUT_TOKENS: u32 = 32_768;

/// Conservative bytes-per-token ratio for the serve-layer prompt-length
/// pre-check. The serve layer has no tokenizer (real tokenization needs the
/// loaded model and happens worker-side, where the authoritative `[HG005]`
/// context-overflow check runs — see `worker/engine/llamacpp.rs`). This ratio
/// gives a cheap LOWER bound on token count from the prompt byte length so an
/// obviously-too-large prompt is rejected early with `400 [HG005]` instead of
/// being shipped to the worker. 4 bytes/token is the standard conservative
/// English-text estimate (OpenAI's published rule of thumb); using it as a
/// divisor under-counts tokens, so the early reject only fires when the prompt
/// is unambiguously over the window — the worker remains the precise backstop.
pub const PROMPT_BYTES_PER_TOKEN: usize = 4;

// The control wire structs are part of this module's public surface (and the
// ts-rs export path), so re-export them at `crate::serve::*`.
pub use wire::*;

/// Map a `HiggsError` to its HTTP status — the single status table shared by
/// both surfaces and the SSE path: HG002/HG003 → 404, HG005/HG013/HG015 → 400,
/// HG006/HG007/HG014/HG017/HG018/HG019 → 503, HG016 → 504, HG037 → 501,
/// HG038/HG039 → 502, and the internal/persistence/admin/chat-task faults
/// (HG040–HG044) → 500 (the default).
pub(crate) fn http_status(err: &HiggsError) -> StatusCode {
    match err {
        HiggsError::ModelNotFound { .. } | HiggsError::ModelNotLoaded { .. } => {
            StatusCode::NOT_FOUND
        }
        // Missing/insufficient API key.
        HiggsError::Unauthorized => StatusCode::UNAUTHORIZED,
        // Refused state transition (last-key revoke on a LAN bind): the
        // request is well-formed but conflicts with the server's live
        // exposure state — mint a replacement first.
        HiggsError::LastKeyOnLan { .. } => StatusCode::CONFLICT,
        // Revoking the last Admin key while other keys remain would lock out the
        // key-management surface ([HG066]) — a conflict with the current key set.
        HiggsError::LastAdminKey { .. } => StatusCode::CONFLICT,
        // Benchmark refused because the model is loaded ([HG067]) — a conflict the
        // client resolves by unloading first.
        HiggsError::BenchModelLoaded { .. } => StatusCode::CONFLICT,
        // Load/chat refused because a benchmark owns the model ([HG068]) — a
        // transient capacity condition; retry after the benchmark finishes.
        HiggsError::BenchInProgress { .. } => StatusCode::SERVICE_UNAVAILABLE,
        // A Turbotune benchmark cancelled by a concurrent load/unload/stop — a
        // transient conflict; re-run when idle.
        HiggsError::BenchCancelled => StatusCode::CONFLICT,
        // Client-side request errors caught at the boundary before dispatch.
        HiggsError::ContextOverflow { .. }
        | HiggsError::InvalidSamplingParam { .. }
        | HiggsError::InvalidModelId { .. }
        | HiggsError::InvalidRequest { .. }
        // A model that isn't Prepared (or whose profile is stale) is a client
        // precondition failure — the caller must Prepare/Re-tune first.
        | HiggsError::NotPrepared { .. }
        | HiggsError::ProfileStale { .. }
        // HG050: the prompt could not be BUILT for this request+model (e.g.
        // the template rejects `tools`); an identical retry fails identically,
        // but a changed request (drop tools) may succeed → client error.
        | HiggsError::TemplateRenderFailed { .. } => StatusCode::BAD_REQUEST,
        // Capacity / infrastructure-down — retryable.
        HiggsError::WorkerSpawnFailed { .. }
        | HiggsError::WorkerDead { .. }
        | HiggsError::ServerBusy { .. }
        | HiggsError::InsufficientMemory { .. }
        // A load that OOM'd through the whole degrade ladder — the machine is
        // out of memory right now; freeing VRAM or a smaller footprint may fix
        // it, so it's a retryable capacity signal, not a client error.
        | HiggsError::LoadOomExhausted { .. }
        // A Turbotune benchmark that found no fitting config — same retryable
        // capacity signal (free VRAM / lower the budget and re-run).
        | HiggsError::BenchExhausted { .. }
        // A retired/unreachable remote node is infrastructure-down — retryable.
        | HiggsError::NodeUnreachable { .. }
        | HiggsError::ServingDisabled => StatusCode::SERVICE_UNAVAILABLE,
        // A wedged-but-alive worker that didn't finish generation in time.
        HiggsError::ChatTimeout { .. } => StatusCode::GATEWAY_TIMEOUT,
        // An unimplemented RPC method = a protocol this server doesn't speak → 501.
        HiggsError::RpcMethodNotFound { .. } => StatusCode::NOT_IMPLEMENTED,
        // An upstream PEER (a node/worker we relay to) misbehaved or rejected us —
        // we are a faulty gateway to it, not the origin of the fault → 502.
        HiggsError::ProtocolViolation { .. } | HiggsError::HubRequestRejected { .. } => {
            StatusCode::BAD_GATEWAY
        }
        // A worker-reported error carries the worker's origin diagnostic code;
        // map by it so a worker-side code (local OR propagated from a remote node)
        // reaches the client as its true status instead of a generic 500.
        HiggsError::WorkerRpc { worker_code, .. } => match worker_code.as_deref() {
            Some("HG002") | Some("HG003") => StatusCode::NOT_FOUND,
            // HG050: template render failed in the worker — same client-error
            // reasoning as the direct arm above.
            Some("HG005") | Some("HG050") => StatusCode::BAD_REQUEST,
            // HG006/HG007: worker down. HG017: remote node couldn't fit the model.
            // HG018: the resolved model was swapped out by a concurrent JIT load
            // (transient) — the client should retry, which re-JITs the model.
            // HG060: a REMOTE node's load exhausted the OOM ladder — the same
            // retryable capacity signal as the direct `LoadOomExhausted` arm,
            // propagated back as a worker code (codex r19).
            Some("HG006") | Some("HG007") | Some("HG017") | Some("HG018") | Some("HG060") => {
                StatusCode::SERVICE_UNAVAILABLE
            }
            // HG016: a remote chat that timed out, propagated as a worker code.
            Some("HG016") => StatusCode::GATEWAY_TIMEOUT,
            // Worker/node→hub→HTTP propagated codes keep their direct-arm status
            // (explicit, not the 500 default, for parity).
            Some("HG037") => StatusCode::NOT_IMPLEMENTED,
            Some("HG038") => StatusCode::BAD_GATEWAY,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        },
        // Internal faults (HG042), local persistence (HG040/HG041), fleet-admin
        // (HG043), and an aborted chat task (HG044) are all server-side 500s.
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

/// Cheap readiness probe: returns `200 OK` as soon as the HTTP server is up,
/// without any worker RPC. "Server is reachable" — not "a model is loaded".
/// Mirrors vllm's `/health`. Served at both `/health` and `/api/higgs/health`.
async fn health() -> StatusCode {
    StatusCode::OK
}

/// Build the higgs router: OpenAI-compatible `/v1` + `/api/higgs/*` control.
///
/// The host serves this on its own listener; all state flows through the shared
/// [`Higgs`] facade. Hardening (established ollama/vllm practice) layered here:
///
/// - **DNS-rebinding host guard** ([`host_guard`]) on every request — rejects a
///   `Host` header that isn't loopback, so a malicious page can't rebind a
///   hostname to `127.0.0.1` and reach this no-auth server.
/// - **Body-size limit** ([`MAX_BODY_BYTES`]) → `413` on oversized bodies.
/// - **Panic recovery** ([`CatchPanicLayer`]) → a handler panic becomes a `500`
///   instead of dropping the connection.
/// - **Control timeout** ([`CONTROL_TIMEOUT`]) on `/api/higgs/*` only — the
///   streaming `/v1/chat/completions` is left un-timed at the HTTP layer so a
///   long SSE stream is never aborted mid-flight.
pub fn router(higgs: Arc<Higgs>) -> Router {
    // Embedded hosts bind loopback, so the DNS-rebinding Host guard is on.
    router_with_host_policy(higgs, true, CONTROL_TIMEOUT, LONG_OP_TIMEOUT)
}

/// [`router`] with the Host-guard policy made explicit. `enforce_loopback_host`
/// is true for a loopback bind (the no-auth surface needs the DNS-rebinding
/// defense) and false for a DELIBERATE non-loopback bind — which [HG058]
/// guarantees is key-gated, so a rebound page gains nothing (401) and LAN
/// clients' natural `Host: <lan-ip>:<port>` must not 403 ([HG012]) before auth.
///
/// `control_timeout` / `tune_timeout` are injected (not read straight from the
/// consts) so a test can build the REAL router with tiny timeouts and prove the
/// tune route is exempt from the tighter control bound. Production callers pass
/// [`CONTROL_TIMEOUT`] / [`LONG_OP_TIMEOUT`].
///
/// `pub(crate)` — NOT `pub` (codex r5): the relaxed (`false`) host policy must be
/// reachable ONLY through [`serve_with_shutdown`] (which runs the [HG058] keyless-LAN
/// and [HG069] no-Admin refusals and records `lan_exposed` BEFORE building it) or the
/// loopback-guarded [`router`]. A `pub` constructor would let an out-of-crate embedder
/// merge a relaxed-host router into its own Axum app with an empty keystore and no
/// exposure state — serving unauthenticated on a LAN with the Host guard off.
/// `pub(crate)` keeps it crate-internal (embedders can't reach it), exposing it only to
/// the in-crate timeout test.
pub(crate) fn router_with_host_policy(
    higgs: Arc<Higgs>,
    enforce_loopback_host: bool,
    control_timeout: Duration,
    long_op_timeout: Duration,
) -> Router {
    // Streaming surface: NO whole-request timeout (an SSE stream must outlive
    // any per-request bound). The chat duration is bounded separately by the
    // worker chat-RPC timeout; the live log stream is unbounded by design.
    let streaming = Router::new()
        .route("/v1/chat/completions", post(v1::v1_chat_completions))
        .route("/api/higgs/logs/stream", get(control::control_logs_stream))
        .route("/api/higgs/events", get(control::control_events_stream));

    // Control + non-streaming surface: a generous whole-request timeout is safe
    // and prevents a wedged request pinning a connection.
    let control = Router::new()
        .route("/v1/models", get(v1::v1_models))
        .route("/api/higgs/models", get(control::control_models))
        .route(
            "/api/higgs/models/estimate",
            post(control::control_estimate),
        )
        .route("/api/higgs/models/unload", post(control::control_unload))
        .route("/api/higgs/models/{*id}", get(control::control_model_by_id))
        .route("/api/higgs/status", get(control::control_status))
        .route("/api/higgs/system", get(control::control_system))
        .route("/api/higgs/nodes", get(control::control_nodes))
        .route("/api/higgs/nodes/load", post(control::control_nodes_load))
        .route(
            "/api/higgs/nodes/unload",
            post(control::control_nodes_unload),
        )
        .route(
            "/api/higgs/nodes/retire",
            post(control::control_nodes_retire),
        )
        .route("/api/higgs/nodes/label", post(control::control_nodes_label))
        .route(
            "/api/higgs/nodes/{node}/models",
            get(control::control_node_models),
        )
        .route("/api/higgs/pair", post(control::control_pair))
        .route("/api/higgs/hub", get(control::control_hub))
        .route("/api/higgs/hub/enable", post(control::control_hub_enable))
        .route("/api/higgs/hub/disable", post(control::control_hub_disable))
        .route("/api/higgs/logs", get(control::control_logs))
        .route(
            "/api/higgs/logs/settings",
            get(control::control_logs_settings).put(control::control_set_logs_settings),
        )
        .route(
            "/api/higgs/settings",
            get(control::control_settings).put(control::control_set_settings),
        )
        .route(
            "/api/higgs/keys",
            get(control::control_keys_list).post(control::control_keys_mint),
        )
        .route(
            "/api/higgs/keys/{label}",
            axum::routing::delete(control::control_keys_revoke),
        )
        .route("/api/higgs/worker/stop", post(control::control_worker_stop))
        .route("/api/higgs/version", get(control::control_version))
        .layer(TimeoutLayer::with_status_code(
            StatusCode::REQUEST_TIMEOUT,
            control_timeout,
        ));

    // Long model-op surface: `/api/higgs/models/load` and `/api/higgs/models/tune`
    // get their OWN, much longer timeout ([`LONG_OP_TIMEOUT`]). A load walks the G5
    // OOM degrade-retry ladder (several worker loads + settle sleeps) and a Turbotune
    // benchmark loads + measures several candidate configs — both routinely outlive
    // the tighter control bound, so under `control_timeout` a big model would be
    // cancelled (408) before the ladder finishes / a profile is saved. Split into a
    // separate group so only these two routes are exempted; the rest of the control
    // surface keeps the tight cap.
    let long_ops = Router::new()
        .route("/api/higgs/models/load", post(control::control_load))
        .route("/api/higgs/models/tune", post(control::control_tune))
        .layer(TimeoutLayer::with_status_code(
            StatusCode::REQUEST_TIMEOUT,
            long_op_timeout,
        ));

    streaming
        .merge(control)
        .merge(long_ops)
        .route("/health", get(health))
        .route("/api/higgs/health", get(health))
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
        .layer(CatchPanicLayer::new())
        // Auth runs AFTER the host guard (outer layers run first): reject a non-loopback
        // Host before doing any key work. `from_fn_with_state` gives the guard the keystore.
        .layer(middleware::from_fn_with_state(higgs.clone(), auth_guard))
        .layer(middleware::from_fn(move |req, next| {
            host_guard(enforce_loopback_host, req, next)
        }))
        .layer(local_cors(higgs.extra_cors_origins()))
        .with_state(higgs)
}

/// DNS-rebinding defense: reject any request whose `Host` header is not a
/// trusted loopback host. The #1 protection for a no-auth loopback server
/// (ollama's `allowedHostsMiddleware`) — without it, a malicious web page can
/// DNS-rebind its own hostname to `127.0.0.1` and reach this server from the
/// victim's browser. A missing `Host` header is rejected (ollama behavior). On
/// rejection: `403` carrying the `[HG012]` diagnostic.
async fn host_guard(enforce_loopback_host: bool, req: Request, next: Next) -> Response {
    if !enforce_loopback_host {
        // Keyed non-loopback bind: every data route is bearer-gated, so DNS
        // rebinding gains nothing — and LAN clients legitimately send their
        // server's LAN ip/hostname in `Host`.
        return next.run(req).await;
    }
    match host_allowed(req.headers()) {
        Ok(()) => next.run(req).await,
        Err(host) => {
            let err = HiggsError::ForbiddenHost { host };
            tracing::warn!(%err, "rejected request with non-loopback Host header");
            (StatusCode::FORBIDDEN, err.to_string()).into_response()
        }
    }
}

/// API-key auth (P5): when keys are configured, require a `Authorization: Bearer hgk_…`
/// with the scope the route needs. An empty keystore disables auth (embedded host). Health
/// checks are always open. On failure: `401` with an OpenAI-style error envelope +
/// `WWW-Authenticate: Bearer`.
async fn auth_guard(
    axum::extract::State(higgs): axum::extract::State<Arc<Higgs>>,
    req: Request,
    next: Next,
) -> Response {
    let keys = higgs.api_keys();
    if keys.is_empty() {
        return next.run(req).await; // auth disabled
    }
    let Some(required) = required_scope(req.method(), req.uri().path()) else {
        return next.run(req).await; // health / unmatched → open (routing handles it)
    };
    match bearer_token(req.headers()) {
        Some(tok) if keys.authorizes(&tok, required) => next.run(req).await,
        _ => unauthorized(),
    }
}

/// The scope a request needs, or `None` for an always-open path (health). Reads: chat →
/// `Chat`, model listing → `Models`, everything else under `/v1` or `/api/higgs` → `Admin`.
fn required_scope(method: &Method, path: &str) -> Option<crate::keys::Scope> {
    use crate::keys::Scope;
    if path == "/health" || path == "/api/higgs/health" {
        return None;
    }
    if path == "/v1/chat/completions" {
        return Some(Scope::Chat);
    }
    if path == "/v1/models" || (method == Method::GET && path.starts_with("/api/higgs/models")) {
        return Some(Scope::Models);
    }
    if path.starts_with("/v1/") || path.starts_with("/api/higgs/") {
        return Some(Scope::Admin);
    }
    None
}

/// Extract the bearer token from an `Authorization: Bearer <token>` header.
fn bearer_token(headers: &HeaderMap) -> Option<String> {
    let v = headers
        .get(axum::http::header::AUTHORIZATION)?
        .to_str()
        .ok()?;
    v.strip_prefix("Bearer ")
        .or_else(|| v.strip_prefix("bearer "))
        .map(|t| t.trim().to_string())
}

/// The `401` envelope (OpenAI-style `error` object) for a missing/insufficient key.
/// The message is the coded [`HiggsError::Unauthorized`] display ("[HG048] …"),
/// so the `401` carries a diagnostic code + resolution like every other reply,
/// while the OpenAI `code: "unauthorized"` stays for client compatibility.
pub(super) fn unauthorized() -> Response {
    let body = axum::Json(serde_json::json!({
        "error": {
            "message": HiggsError::Unauthorized.to_string(),
            "type": "invalid_request_error",
            "code": "unauthorized",
        }
    }));
    let mut resp = (StatusCode::UNAUTHORIZED, body).into_response();
    resp.headers_mut().insert(
        axum::http::header::WWW_AUTHENTICATE,
        HeaderValue::from_static("Bearer"),
    );
    resp
}

/// Validate the `Host` header: the host portion (sans `:port`) must be a
/// loopback host. Returns `Err(host_string)` with the offending value (or
/// `"<missing>"` when absent) on rejection.
fn host_allowed(headers: &HeaderMap) -> Result<(), String> {
    let Some(value) = headers.get(axum::http::header::HOST) else {
        return Err("<missing>".to_string());
    };
    let Ok(s) = value.to_str() else {
        return Err("<non-utf8>".to_string());
    };
    if is_loopback_host(s) {
        Ok(())
    } else {
        Err(s.to_string())
    }
}

/// Whether a `Host`-header value (`host` or `host:port`) names a loopback host:
/// `localhost`, any IPv4 in `127.0.0.0/8`, or IPv6 `::1` / `[::1]`. The host
/// portion is the part before the final `:port`, with IPv6 brackets handled
/// (`[::1]:11434`). The `127.0.0.0/8` acceptance matches `is_loopback_bind` in
/// the `higgs` binary, so any loopback bind the binary permits without a
/// warning is also reachable through the Host guard.
fn is_loopback_host(host_port: &str) -> bool {
    // Bracketed IPv6: `[::1]` or `[::1]:port`.
    let host = if let Some(rest) = host_port.strip_prefix('[') {
        match rest.split_once(']') {
            Some((inner, _)) => inner,
            None => return false,
        }
    } else if host_port.matches(':').count() >= 2 {
        // Bare (unbracketed) IPv6 literal like `::1` — no port can be appended
        // without brackets, so take the whole thing.
        host_port
    } else {
        // host or host:port (IPv4 / DNS name).
        host_port.split(':').next().unwrap_or(host_port)
    };
    if host == "localhost" {
        return true;
    }
    // Any loopback IP literal: 127.0.0.0/8 (IPv4) or ::1 (IPv6).
    if let Ok(ip) = host.parse::<std::net::IpAddr>() {
        return ip.is_loopback();
    }
    false
}

/// Serve the higgs router on `listener`, shutting down **gracefully** when
/// `shutdown` resolves: in-flight requests drain, then the worker is stopped
/// via [`Higgs::stop`]. The single graceful-shutdown entry point for any host —
/// the `higgs` binary wires it to SIGTERM/Ctrl-C; tests pass their own
/// future. Returns the serve I/O error, if any.
pub async fn serve_with_shutdown(
    higgs: Arc<Higgs>,
    listener: tokio::net::TcpListener,
    shutdown: impl std::future::Future<Output = ()> + Send + 'static,
) -> std::io::Result<()> {
    // The Host-guard policy follows the REAL bound address: loopback keeps the
    // DNS-rebinding defense; a deliberate non-loopback bind (key-gated per
    // [HG058]) must accept LAN `Host` values or the mode is unusable.
    let enforce_loopback_host = listener
        .local_addr()
        .map(|a| a.ip().is_loopback())
        .unwrap_or(true); // unknowable ⇒ fail CLOSED (strictest policy)
                          // [HG058] AT THE ENFORCEMENT POINT: a non-loopback listener with zero keys
                          // must never serve — auth would be off AND the Host guard below would be
                          // relaxed, exposing the whole surface. `run_standalone` refuses earlier
                          // (before even binding — nicer failure), but THIS is the last line for
                          // library/embedded callers who hand us their own listener.
    if !enforce_loopback_host && higgs.api_keys().is_empty() {
        let bind = listener
            .local_addr()
            .map(|a| a.to_string())
            .unwrap_or_else(|_| "<unknown>".into());
        // Refusing to serve, but an embedded caller may have already
        // `start()`ed this facade (spawning a worker). Tear it down before the
        // early return so the rejected serve attempt doesn't LEAK the worker /
        // runtime (codex r11) — same cleanup the normal exit path runs.
        higgs.stop().await;
        return Err(std::io::Error::other(
            HiggsError::LanBindWithoutKeys { bind }.to_string(),
        ));
    }
    // [HG069] AT THE ENFORCEMENT POINT: a non-loopback listener whose keys are ALL
    // non-Admin locks out the Admin-only key-management API (mint/revoke) — mirror
    // `run_standalone`'s guard so a library/embedded caller handing us its own
    // listener can't bypass it. Zero-keys is already refused above, so at least one
    // key exists here; refuse if none is Admin-capable.
    if !enforce_loopback_host
        && !higgs
            .api_keys()
            .iter()
            .any(|k| k.scopes.contains(&crate::keys::Scope::Admin))
    {
        let bind = listener
            .local_addr()
            .map(|a| a.to_string())
            .unwrap_or_else(|_| "<unknown>".into());
        // Same worker-teardown-before-refuse as the [HG058] path (no leak).
        higgs.stop().await;
        return Err(std::io::Error::other(
            HiggsError::LanBindWithoutAdminKey { bind }.to_string(),
        ));
    }
    // Record the exposure on the facade: key management refuses to revoke the
    // LAST key while LAN-exposed ([HG059]) — revoke-to-empty would reopen the
    // whole surface at runtime, bypassing the [HG058] startup guarantee.
    higgs.set_lan_exposed(!enforce_loopback_host);
    let app = router_with_host_policy(
        Arc::clone(&higgs),
        enforce_loopback_host,
        CONTROL_TIMEOUT,
        LONG_OP_TIMEOUT,
    );
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown)
        .await?;
    higgs.stop().await;
    Ok(())
}

/// CORS for higgs's standalone listener: the frontend calls higgs's own port
/// cross-origin, so its serving origin must be allowed. The frontend is served
/// from three different origins depending on mode — the Tauri webview
/// (`tauri://localhost`), the Vite dev server (`http://localhost:5173`), and
/// the gateway-embedded SPA (`http://{host}:{port}`, a dynamic port not known
/// here). Rather than enumerate ports, allow any localhost/127.0.0.1 origin
/// plus the tauri webview origins. higgs is localhost/webview only — not a
/// public surface — so trusting any loopback origin matches its threat model.
fn local_cors(extra_origins: Vec<String>) -> CorsLayer {
    CorsLayer::new()
        .allow_origin(AllowOrigin::predicate(move |origin: &HeaderValue, _req| {
            is_local_origin(origin)
                || origin
                    .to_str()
                    .is_ok_and(|o| extra_origins.iter().any(|e| e == o))
        }))
        .allow_methods([Method::GET, Method::POST, Method::PUT, Method::DELETE])
        .allow_headers(tower_http::cors::Any)
}

/// Whether `origin` is a trusted local origin: the Tauri webview
/// (`tauri://localhost`, `http://tauri.localhost`) or any loopback HTTP origin
/// (`http://localhost[:port]`, `http://127.0.0.1[:port]`). Non-UTF-8 or
/// non-loopback origins are rejected.
fn is_local_origin(origin: &HeaderValue) -> bool {
    let Ok(s) = origin.to_str() else {
        return false;
    };
    if s == "tauri://localhost" || s == "http://tauri.localhost" {
        return true;
    }
    // Strip the scheme, then match the host with an optional `:port` suffix.
    let Some(host_port) = s.strip_prefix("http://") else {
        return false;
    };
    let host = host_port.split(':').next().unwrap_or(host_port);
    host == "localhost" || host == "127.0.0.1"
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;
