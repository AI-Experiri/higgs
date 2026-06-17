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
mod wire;

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
/// HG006/HG007/HG014/HG017/HG018 → 503, HG016 → 504, else 500.
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
        | HiggsError::InsufficientMemory { .. } => StatusCode::SERVICE_UNAVAILABLE,
        // A wedged-but-alive worker that didn't finish generation in time.
        HiggsError::ChatTimeout { .. } => StatusCode::GATEWAY_TIMEOUT,
        // A worker-reported error carries the worker's origin diagnostic code;
        // map by it so a worker-side HG002/HG003/HG005/HG006/HG007 reaches the
        // client as its true status instead of a generic 500.
        HiggsError::WorkerRpc { worker_code, .. } => match worker_code.as_deref() {
            Some("HG002") | Some("HG003") => StatusCode::NOT_FOUND,
            Some("HG005") => StatusCode::BAD_REQUEST,
            // HG006/HG007: worker down. HG018: the resolved model was swapped
            // out by a concurrent JIT load before generation (transient) — the
            // client should retry, which re-JITs the requested model.
            Some("HG006") | Some("HG007") | Some("HG018") => StatusCode::SERVICE_UNAVAILABLE,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        },
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
        .route("/api/higgs/models/unload", post(control::control_unload))
        .route("/api/higgs/models/{*id}", get(control::control_model_by_id))
        .route("/api/higgs/status", get(control::control_status))
        .route("/api/higgs/system", get(control::control_system))
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
/// the `higgs-server` binary, so any loopback bind the binary permits without a
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
/// the `higgs-server` binary wires it to SIGTERM/Ctrl-C; tests pass their own
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
mod tests {
    use super::{host_allowed, http_status, is_local_origin, is_loopback_host};
    use axum::http::{HeaderMap, HeaderValue, StatusCode};

    use crate::diagnostic::HiggsError;

    #[test]
    fn loopback_host_matcher() {
        for h in [
            "127.0.0.1",
            "127.0.0.1:11434",
            "127.0.0.5", // 127.0.0.0/8 — matches is_loopback_bind acceptance
            "127.0.0.5:11434",
            "localhost",
            "localhost:5173",
            "::1",
            "[::1]",
            "[::1]:11434",
        ] {
            assert!(is_loopback_host(h), "should allow Host {h}");
        }
        for h in [
            "evil.example.com",
            "evil.example.com:11434",
            "10.0.0.5:11434",
            "0.0.0.0",
            "169.254.169.254",
            "[2001:db8::1]",
        ] {
            assert!(!is_loopback_host(h), "should reject Host {h}");
        }
    }

    #[test]
    fn host_guard_allows_loopback_and_rejects_others() {
        let mut ok = HeaderMap::new();
        ok.insert("host", HeaderValue::from_static("127.0.0.1:11434"));
        assert!(host_allowed(&ok).is_ok());

        let mut bad = HeaderMap::new();
        bad.insert("host", HeaderValue::from_static("evil.example.com"));
        assert_eq!(host_allowed(&bad), Err("evil.example.com".to_string()));

        // Missing Host is rejected (ollama behavior).
        let missing = HeaderMap::new();
        assert_eq!(host_allowed(&missing), Err("<missing>".to_string()));
    }

    /// End-to-end through the real router: a foreign `Host` → 403, an oversized
    /// body → 413, and a loopback `Host` with a small body passes the guard.
    #[tokio::test]
    async fn router_host_guard_and_body_limit() {
        use super::test_support::{make_app, make_supervisor};
        use axum::body::Body;
        use axum::http::Request;
        use tower::ServiceExt;

        let (sup, _w, _r, _ring) = make_supervisor();
        let app = make_app(sup);

        // Foreign Host → 403 (HG012), before any handler runs.
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/higgs/status")
                    .header("host", "evil.example.com")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);

        // Oversized body (loopback Host) → 413.
        let big = vec![b'x'; super::MAX_BODY_BYTES + 1];
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/higgs/models/load")
                    .header("host", "127.0.0.1:11434")
                    .header("content-type", "application/json")
                    .body(Body::from(big))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE);
    }

    /// `/health` returns 200 without a worker RPC.
    #[tokio::test]
    async fn health_is_cheap_200() {
        use super::test_support::{make_app, make_supervisor};
        use axum::body::Body;
        use axum::http::Request;
        use tower::ServiceExt;

        let (sup, _w, _r, _ring) = make_supervisor();
        let app = make_app(sup);
        for uri in ["/health", "/api/higgs/health"] {
            let resp = app
                .clone()
                .oneshot(
                    Request::builder()
                        .uri(uri)
                        .header("host", "127.0.0.1")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(resp.status(), StatusCode::OK, "{uri}");
        }
    }

    #[test]
    fn local_origin_matcher() {
        let yes = [
            "tauri://localhost",
            "http://tauri.localhost",
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:8081", // gateway-embedded SPA on an arbitrary port
            "http://127.0.0.1",      // no port
        ];
        let no = [
            "https://localhost:5173", // non-loopback scheme (TLS not served locally)
            "http://example.com",
            "http://evil.localhost.attacker.com",
            "http://10.0.0.5:5173",
        ];
        for o in yes {
            assert!(
                is_local_origin(&HeaderValue::from_static(o)),
                "should allow {o}"
            );
        }
        for o in no {
            assert!(
                !is_local_origin(&HeaderValue::from_static(o)),
                "should reject {o}"
            );
        }
    }

    #[test]
    fn http_status_mapping_table() {
        assert_eq!(
            http_status(&HiggsError::ModelNotFound { id: "x".into() }),
            StatusCode::NOT_FOUND
        );
        assert_eq!(
            http_status(&HiggsError::ModelNotLoaded { id: "x".into() }),
            StatusCode::NOT_FOUND
        );
        assert_eq!(
            http_status(&HiggsError::ContextOverflow {
                prompt_tokens: 10,
                max_gen: 5,
                n_ctx: 8,
            }),
            StatusCode::BAD_REQUEST
        );
        assert_eq!(
            http_status(&HiggsError::WorkerDead {
                context: "gone".into(),
            }),
            StatusCode::SERVICE_UNAVAILABLE
        );
        assert_eq!(
            http_status(&HiggsError::WorkerSpawnFailed {
                source: std::io::Error::other("boom"),
            }),
            StatusCode::SERVICE_UNAVAILABLE
        );
        // Phase A2 hardening codes.
        assert_eq!(
            http_status(&HiggsError::InvalidSamplingParam {
                param: "top_p".into(),
                detail: "must be in (0, 1]".into(),
            }),
            StatusCode::BAD_REQUEST
        );
        assert_eq!(
            http_status(&HiggsError::InvalidModelId {
                id: "../x".into(),
                reason: "traversal".into(),
            }),
            StatusCode::BAD_REQUEST
        );
        assert_eq!(
            http_status(&HiggsError::ServerBusy {
                in_flight: 8,
                max: 8,
            }),
            StatusCode::SERVICE_UNAVAILABLE
        );
        assert_eq!(
            http_status(&HiggsError::ChatTimeout {
                elapsed: std::time::Duration::from_secs(600),
            }),
            StatusCode::GATEWAY_TIMEOUT
        );
        // Phase B: insufficient-memory load refusal is a retryable capacity 503.
        assert_eq!(
            http_status(&HiggsError::InsufficientMemory {
                id: "org/model".into(),
                needed_bytes: 8_000_000_000,
                available_bytes: 4_000_000_000,
                headroom_fraction: 0.8,
            }),
            StatusCode::SERVICE_UNAVAILABLE
        );
    }

    #[test]
    fn worker_rpc_maps_by_worker_code() {
        // A worker-reported error maps to its origin code's status, not 500.
        let rpc = |code: Option<&str>| HiggsError::WorkerRpc {
            method: "higgs/chat".into(),
            message: "x".into(),
            worker_code: code.map(ToOwned::to_owned),
        };
        assert_eq!(http_status(&rpc(Some("HG002"))), StatusCode::NOT_FOUND);
        assert_eq!(http_status(&rpc(Some("HG003"))), StatusCode::NOT_FOUND);
        assert_eq!(http_status(&rpc(Some("HG005"))), StatusCode::BAD_REQUEST);
        assert_eq!(
            http_status(&rpc(Some("HG007"))),
            StatusCode::SERVICE_UNAVAILABLE
        );
        // HG018: resident-model swap (worker refused) → retryable 503.
        assert_eq!(
            http_status(&rpc(Some("HG018"))),
            StatusCode::SERVICE_UNAVAILABLE
        );
        // Unknown / absent code falls back to 500.
        assert_eq!(
            http_status(&rpc(Some("HG011"))),
            StatusCode::INTERNAL_SERVER_ERROR
        );
        assert_eq!(http_status(&rpc(None)), StatusCode::INTERNAL_SERVER_ERROR);
    }
}
