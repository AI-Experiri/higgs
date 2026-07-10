//! The higgs HTTP surface. higgs is library-first, so the ONLY thing served over
//! a socket is the strict OpenAI `/v1` (chat + models):
//!
//! - [`v1`] — strict OpenAI `/v1` (chat + models), `async-openai` types verbatim.
//! - [`wire`] — higgs's own ts-rs-exported control request/response structs (still
//!   the crate-API return shapes the facade builds).
//! - [`stream`] — SSE assembly for streaming chat.
//! - [`control`] — the SHARED control helpers the [`Higgs`] facade delegates to
//!   (row formatting, mint/revoke decisions, hub-status snapshot). There is no
//!   longer a `/api/higgs/*` HTTP surface — control runs in-process via the crate API.
//!
//! This module owns the cross-cutting pieces: the `/v1` [`v1_router`], graceful
//! [`serve_v1`], CORS, and the shared [`http_status`] mapping.

pub(crate) mod control;
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

/// Whole-request timeout for the **non-streaming `/v1` routes** (models list
/// and friends — named "control" from when it bounded the deleted
/// `/api/higgs/*` surface; also reported in `system` info as
/// `control_timeout_secs`). Generous (a cold model load can take seconds), but
/// bounded so a wedged request can't pin a connection forever. Deliberately
/// **not** applied to `/v1/chat/completions`: a long SSE stream must outlive any
/// per-request timeout (its duration is bounded separately by the worker
/// chat-RPC timeout, a different layer).
pub const CONTROL_TIMEOUT: Duration = Duration::from_secs(120);

/// Whole-request timeout for the LONG model ops (load and tune — formerly the
/// `/api/higgs/models/load` and `/models/tune` routes), exempt from the tighter
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

/// Map a `HiggsError` to its HTTP status — the single status table shared by the
/// `/v1` surface and the SSE path: HG002/HG003 → 404, HG005/HG013/HG015 → 400,
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
        // HG072: a keystore request (mint/revoke) that failed validation — same
        // 400 class as HG049, split only so the message doesn't hand a token
        // mint advice about the chat schema.
        | HiggsError::InvalidKeyRequest { .. }
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
/// Mirrors vllm's `/health`. Served at `/health`.
async fn health() -> StatusCode {
    StatusCode::OK
}

/// Build the `/v1`-ONLY router: the OpenAI-compatible chat + models surface,
/// with NO `/api/higgs/*` control routes at all. This is the surface an external
/// app (a non-embedding client) reaches over HTTP; control lives on the
/// in-process crate API, not on any socket.
///
/// Routes: `POST /v1/chat/completions` (streaming — no whole-request timeout),
/// `GET /v1/models`, and the cheap `GET /health` probe. There is deliberately no
/// `/api/higgs/health` here — this router never carried the `/api/higgs/*`
/// namespace, so every such path 404s by construction (not by deletion).
///
/// The layer stack is `DefaultBodyLimit` → `CatchPanicLayer` → [`auth_guard`] →
/// [`host_guard`] → [`local_cors`] → `with_state`. Embedded hosts bind loopback,
/// so the DNS-rebinding Host guard is on.
pub fn v1_router(higgs: Arc<Higgs>) -> Router {
    // NOTE: building a router does NOT record `applied_cors_origins` — only
    // [`serve_v1`] (the entrypoint that actually puts a listener behind the
    // layer) captures what went live, so a built-but-never-served router (or a
    // second router built while another serves) can't clobber the running
    // listener's applied list and flip `restart_required` to a false `false`.
    // An embedder that mounts this router itself gets pre-serve semantics from
    // `Higgs::cors_settings` (`applied_origins` empty, `restart_required`
    // false): higgs cannot know when a mounted router goes live.
    let extra_origins = higgs.extra_cors_origins();
    v1_router_with_host_policy(higgs, true, extra_origins)
}

/// [`v1_router`] with the Host-guard policy made explicit. `enforce_loopback_host`
/// is `true` for a loopback bind (keep the DNS-rebinding defense) and `false` for
/// a deliberate non-loopback bind (which [HG058] guarantees is key-gated, so LAN
/// clients' natural `Host: <lan-ip>:<port>` must not 403 before auth).
///
/// `pub(crate)` — NOT `pub`: the relaxed (`false`) policy must be reachable ONLY
/// through [`serve_v1`], which runs the [HG058] keyless-LAN and [HG069] no-Admin
/// refusals and arms `lan_exposed` BEFORE building it. A `pub` relaxed
/// constructor would let an out-of-crate embedder serve unauthenticated on a LAN
/// with the Host guard off.
/// `extra_origins` is the extra CORS allowlist the layer is built with — passed
/// in (not read here) so the caller that records it as APPLIED ([`serve_v1`])
/// hands the layer the exact same snapshot it recorded (no double-read skew).
pub(crate) fn v1_router_with_host_policy(
    higgs: Arc<Higgs>,
    enforce_loopback_host: bool,
    extra_origins: Vec<String>,
) -> Router {
    // Streaming surface: NO whole-request timeout (an SSE stream must outlive any
    // per-request bound). The chat duration is bounded separately by the worker
    // chat-RPC timeout.
    let streaming = Router::new().route("/v1/chat/completions", post(v1::v1_chat_completions));

    // `/v1/models` is a cheap non-streaming call; a generous whole-request
    // timeout ([`CONTROL_TIMEOUT`]) keeps a wedged request from pinning a
    // connection.
    let models = Router::new().route("/v1/models", get(v1::v1_models)).layer(
        TimeoutLayer::with_status_code(StatusCode::REQUEST_TIMEOUT, CONTROL_TIMEOUT),
    );

    streaming
        .merge(models)
        .route("/health", get(health))
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
        .layer(CatchPanicLayer::new())
        // Auth runs AFTER the host guard (outer layers run first): reject a non-loopback
        // Host before doing any key work. `from_fn_with_state` gives the guard the keystore.
        .layer(middleware::from_fn_with_state(higgs.clone(), auth_guard))
        .layer(middleware::from_fn(move |req, next| {
            host_guard(enforce_loopback_host, req, next)
        }))
        .layer(local_cors(extra_origins))
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
    let Some(required) = required_scope(req.uri().path()) else {
        return next.run(req).await; // health / unmatched → open (routing handles it)
    };
    let keys = higgs.api_keys();
    if keys.is_empty() {
        // No keys configured ⇒ auth OFF (keyless-loopback-open). This is the
        // local-dev default: an empty keystore means the surface is open, which is
        // safe only because a keyless bind is loopback-only (the non-loopback [HG058]
        // refusal in `serve_v1` guarantees it). An embedder that wants auth mints
        // real keys (`Higgs::mint_key` / `register_internal_token`); once ANY key
        // exists the store is non-empty and every request — embedder and external
        // app alike — goes through the bearer check below on equal footing.
        return next.run(req).await;
    }
    match bearer_token(req.headers()).and_then(|tok| keys.authorizing_sha(&tok, required)) {
        Some(sha) => {
            // Record last-used on the matched key (throttled, best-effort).
            higgs.touch_api_key(&sha);
            next.run(req).await
        }
        None => unauthorized(),
    }
}

/// The scope a request needs, or `None` for an always-open path (health). Reads:
/// chat → `Chat`, model listing → `Models`, everything else under `/v1` → `Admin`.
/// Only the `/v1` surface is served now, so there are no `/api/higgs/*` arms (and
/// the method no longer disambiguates a scope, so it is not read).
fn required_scope(path: &str) -> Option<crate::keys::Scope> {
    use crate::keys::Scope;
    if path == "/health" {
        return None;
    }
    if path == "/v1/chat/completions" {
        return Some(Scope::Chat);
    }
    if path == "/v1/models" {
        return Some(Scope::Models);
    }
    if path.starts_with("/v1/") {
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
/// (`[::1]:11434`). The `127.0.0.0/8` acceptance matches the loopback-bind rule
/// [`serve_v1`] enforces on the REAL bound address, so any loopback listener the
/// server permits is also reachable through the Host guard.
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

/// Serve the `/v1`-ONLY router ([`v1_router`]) on `listener`, shutting down
/// **gracefully** when `shutdown` resolves: in-flight requests drain, then the
/// worker is stopped via [`Higgs::stop`]. The single graceful-serve entry point —
/// the only HTTP surface higgs exposes. Control no longer lives on any socket; an
/// external app reaches chat + models here, and an embedder drives control through
/// the in-process crate API.
///
/// The [HG058] keyless-LAN and [HG069] no-Admin refusals below are the surface's
/// last line of defense: a non-loopback listener with an empty keystore, or one
/// whose keys are all non-Admin, is refused before it can serve — because on such
/// a bind auth would be off AND the Host guard relaxed, exposing the whole surface
/// to the network. This runs on the REAL bound address, so a library/embedded
/// caller handing us its own listener can't bypass it.
pub async fn serve_v1(
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
                          // Register BEFORE the startup guards, UNARMED, under the serve-lifecycle lock.
                          //
                          // BEFORE the guards: a refusal must decide "was I the ONLY serve on this facade?"
                          // to know whether tearing it down would strand a sibling, and
                          // `ServeGuard::release()` answers that atomically under the registry lock.
                          //
                          // Under the LIFECYCLE lock: a concurrent last-listener exit does `release()` then
                          // `stop()` as two steps, and `stop()` is TERMINAL (`shutting_down` never resets).
                          // Registering in that gap would leave this listener serving a facade the departing
                          // one is about to drain for good.
    let serve_guard = {
        let _lifecycle = higgs.serve_lifecycle().await;
        higgs.register_serve(false, listener.local_addr().ok(), Vec::new())
    };
    // Refusing to serve, but an embedded caller may have already `start()`ed this
    // facade (spawning a worker). Tear it down before the early return so the
    // rejected serve doesn't LEAK the worker/runtime (codex r11) — but ONLY when
    // this was the sole serve; a sibling owns the facade and its workers. The
    // lifecycle lock makes release-then-stop atomic against a registration.
    macro_rules! refuse {
        ($err:expr) => {{
            let _lifecycle = higgs.serve_lifecycle().await;
            if serve_guard.release() {
                higgs.stop().await;
            }
            return Err(std::io::Error::other($err.to_string()));
        }};
    }
    // THE non-loopback enforcement point. [HG058] (a LAN listener needs at least one
    // key — else auth is off AND the Host guard below is relaxed, exposing the whole
    // surface) and [HG069] (…at least one ADMIN key, else the key-management API is
    // locked out) are checked, and this listener's LAN exposure is ARMED, together
    // under the keystore lock. They MUST be atomic with a key revoke: otherwise a
    // concurrent `revoke_key` reads `lan_exposed() == false`, this serve passes its
    // key check against the not-yet-published store, and the revoke then empties it —
    // a KEYLESS listener on a LAN. Serialized, one always loses. This is the last
    // line for library/embedded callers who hand us their own listener.
    if !enforce_loopback_host {
        let bind = listener
            .local_addr()
            .map(|a| a.to_string())
            .unwrap_or_else(|_| "<unknown>".into());
        if let Err(e) = higgs.arm_lan_serve(&serve_guard, &bind) {
            refuse!(e);
        }
    }
    // The EXACT extra CORS origins THIS listener's layer is built with — read once
    // and handed BOTH to the layer and to the registration, so the disclosed
    // "applied" list is byte-identical to what is actually enforced (no double-read
    // skew). Captured HERE (not in the router builder) so only a router that really
    // goes live behind a listener claims "applied"; see the note on [`v1_router`].
    // Residual race (accepted, self-correcting): a `set_cors_origins` landing
    // between this read and the record below can get a response computed against
    // pre-record state; the next `cors_settings` read is correct.
    let extra_origins = higgs.extra_cors_origins();
    // Record the CORS list this listener enforces (the LAN flag was armed at
    // registration — see there). Per-listener state, so a sibling serve starting or
    // exiting can't rewrite this one's disclosures. `serve_guard` deregisters on
    // DROP — covering task CANCELLATION (an aborted future runs destructors but no
    // code past its await) — and on the explicit `release()` below.
    serve_guard.set_cors_origins(extra_origins.clone());
    let app = v1_router_with_host_policy(Arc::clone(&higgs), enforce_loopback_host, extra_origins);
    let served = axum::serve(listener, app)
        .with_graceful_shutdown(shutdown)
        .await;
    // Deregister IMMEDIATELY — before the terminal worker drain below, which can take
    // seconds. A listener that has stopped accepting must not keep disclosing itself
    // (`server_config`/`cors_settings` reporting a dead `ip:port` or a stale applied
    // snapshot) nor keep forcing the [HG059] last-key-revoke refusal while workers
    // shut down. (On CANCELLATION this line never runs and the guard's `Drop` does it.)
    //
    // Only the LAST listener owns the facade teardown. `serve_v1` is public and an
    // embedder may run several listeners on one `Arc<Higgs>` (e.g. loopback + LAN);
    // draining the shared local node when just ONE of them stops would strand the
    // siblings still accepting requests on a facade whose workers are gone and whose
    // next load hits the node-runtime shutdown path.
    //
    // The lifecycle lock spans release-and-stop so an incoming `serve_v1` cannot
    // register in between and inherit the terminal `stop()` a departing last
    // listener is about to run. Teardown runs on BOTH exits (graceful shutdown and
    // a fatal `axum::serve` error) — the listener is equally gone either way, so
    // `served?` must not skip the drain and leak workers.
    {
        let _lifecycle = higgs.serve_lifecycle().await;
        if serve_guard.release() {
            higgs.stop().await;
        }
    }
    served?;
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
