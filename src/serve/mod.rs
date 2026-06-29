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
mod stream;
#[cfg(test)]
pub(crate) mod test_support;
mod v1;
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
        // Client-side request errors caught at the boundary before dispatch.
        HiggsError::ContextOverflow { .. }
        | HiggsError::InvalidSamplingParam { .. }
        | HiggsError::InvalidModelId { .. } => StatusCode::BAD_REQUEST,
        // Capacity / infrastructure-down — retryable.
        HiggsError::WorkerSpawnFailed { .. }
        | HiggsError::WorkerDead { .. }
        | HiggsError::ServerBusy { .. }
        | HiggsError::InsufficientMemory { .. }
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
            Some("HG005") => StatusCode::BAD_REQUEST,
            // HG006/HG007: worker down. HG017: remote node couldn't fit the model.
            // HG018: the resolved model was swapped out by a concurrent JIT load
            // (transient) — the client should retry, which re-JITs the model.
            Some("HG006") | Some("HG007") | Some("HG017") | Some("HG018") => {
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
    // Streaming surface: NO whole-request timeout (an SSE stream must outlive
    // any per-request bound). The chat duration is bounded separately by the
    // worker chat-RPC timeout; the live log stream is unbounded by design.
    let streaming = Router::new()
        .route("/v1/chat/completions", post(v1::v1_chat_completions))
        .route("/api/higgs/logs/stream", get(control::control_logs_stream));

    // Control + non-streaming surface: a generous whole-request timeout is safe
    // and prevents a wedged request pinning a connection.
    let control = Router::new()
        .route("/v1/models", get(v1::v1_models))
        .route("/api/higgs/models", get(control::control_models))
        .route("/api/higgs/models/load", post(control::control_load))
        .route("/api/higgs/models/tune", post(control::control_tune))
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
        .route("/api/higgs/worker/stop", post(control::control_worker_stop))
        .route("/api/higgs/version", get(control::control_version))
        .layer(TimeoutLayer::with_status_code(
            StatusCode::REQUEST_TIMEOUT,
            CONTROL_TIMEOUT,
        ));

    streaming
        .merge(control)
        .route("/health", get(health))
        .route("/api/higgs/health", get(health))
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
        .layer(CatchPanicLayer::new())
        // Auth runs AFTER the host guard (outer layers run first): reject a non-loopback
        // Host before doing any key work. `from_fn_with_state` gives the guard the keystore.
        .layer(middleware::from_fn_with_state(higgs.clone(), auth_guard))
        .layer(middleware::from_fn(host_guard))
        .layer(local_cors())
        .with_state(higgs)
}

/// DNS-rebinding defense: reject any request whose `Host` header is not a
/// trusted loopback host. The #1 protection for a no-auth loopback server
/// (ollama's `allowedHostsMiddleware`) — without it, a malicious web page can
/// DNS-rebind its own hostname to `127.0.0.1` and reach this server from the
/// victim's browser. A missing `Host` header is rejected (ollama behavior). On
/// rejection: `403` carrying the `[HG012]` diagnostic.
async fn host_guard(req: Request, next: Next) -> Response {
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
fn unauthorized() -> Response {
    let body = axum::Json(serde_json::json!({
        "error": {
            "message": "missing or insufficient API key",
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
    let app = router(Arc::clone(&higgs));
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
fn local_cors() -> CorsLayer {
    CorsLayer::new()
        .allow_origin(AllowOrigin::predicate(|origin: &HeaderValue, _req| {
            is_local_origin(origin)
        }))
        .allow_methods([Method::GET, Method::POST, Method::PUT])
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
