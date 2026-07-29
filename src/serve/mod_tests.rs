use super::{
    bearer_token, host_allowed, http_status, is_local_origin, is_loopback_host, required_scope,
    serve_v1, v1_router,
};
use axum::http::{HeaderMap, HeaderValue, StatusCode};

use std::sync::Arc;

use crate::diagnostic::HiggsError;
use crate::keys::Scope;

#[test]
fn required_scope_maps_routes() {
    // Only the `/v1` surface is served now: health is open, chat → Chat, models
    // listing → Models, and any other `/v1/*` path is Admin (fail-closed default).
    assert_eq!(required_scope("/health"), None);
    assert_eq!(required_scope("/v1/chat/completions"), Some(Scope::Chat));
    assert_eq!(required_scope("/v1/models"), Some(Scope::Models));
    // Any other `/v1/*` path → Admin (fail-closed).
    assert_eq!(required_scope("/v1/anything-else"), Some(Scope::Admin));
    // unknown path → open (routing 404s it)
    assert_eq!(required_scope("/random"), None);
    // The `/api/higgs/*` control surface is gone — such a path matches no `/v1`
    // arm, so it falls through to `None` (routing then 404s it).
    assert_eq!(required_scope("/api/higgs/status"), None);
}

#[test]
fn bearer_token_parses_authorization_header() {
    let mut h = HeaderMap::new();
    assert_eq!(bearer_token(&h), None);
    h.insert(
        axum::http::header::AUTHORIZATION,
        HeaderValue::from_static("Bearer hgk_abc"),
    );
    assert_eq!(bearer_token(&h).as_deref(), Some("hgk_abc"));
    h.insert(
        axum::http::header::AUTHORIZATION,
        HeaderValue::from_static("bearer hgk_xyz"),
    );
    assert_eq!(bearer_token(&h).as_deref(), Some("hgk_xyz"));
    h.insert(
        axum::http::header::AUTHORIZATION,
        HeaderValue::from_static("Basic zzz"),
    );
    assert_eq!(bearer_token(&h), None);
}

#[test]
fn loopback_host_matcher() {
    for h in [
        "127.0.0.1",
        "127.0.0.1:11434",
        "127.0.0.5", // 127.0.0.0/8 acceptance
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

/// The bracketed-IPv6 form with a missing closing `]` (`[::1` — no `]`) takes
/// the `None => return false` branch of `is_loopback_host`: a malformed
/// bracketed literal is rejected rather than parsed.
#[test]
fn loopback_host_rejects_unterminated_bracket() {
    assert!(!is_loopback_host("[::1"), "unterminated bracket rejected");
    assert!(
        !is_loopback_host("[127.0.0.1"),
        "unterminated bracket rejected even for an IPv4 inside"
    );
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

/// A non-UTF-8 `Host` header value takes the `to_str()` error branch of
/// `host_allowed`, which reports `"<non-utf8>"` (not the raw bytes).
#[test]
fn host_allowed_rejects_non_utf8_value() {
    let mut h = HeaderMap::new();
    // 0xFF is not valid UTF-8, so `HeaderValue::to_str()` fails.
    h.insert(
        "host",
        HeaderValue::from_bytes(&[0xff, 0xfe]).expect("bytes form a valid header value"),
    );
    assert_eq!(host_allowed(&h), Err("<non-utf8>".to_string()));
}

/// End-to-end through the real `/v1` router: a foreign `Host` → 403 (the guard
/// runs before routing), and an oversized body → 413. Both hardening layers are
/// shared by `v1_router`, so a `/v1` path exercises them just as the combined
/// router once did.
#[tokio::test]
async fn router_host_guard_and_body_limit() {
    use super::test_support::make_app;
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    let app = make_app();

    // Foreign Host → 403 (HG012), before any handler runs.
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/models")
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
                .uri("/v1/chat/completions")
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
    use super::test_support::make_app;
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    let app = make_app();
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/health")
                .header("host", "127.0.0.1")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
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

/// A non-UTF-8 `Origin` header value takes the `to_str()` error branch of
/// `is_local_origin`, which rejects it (returns `false`).
#[test]
fn local_origin_rejects_non_utf8_value() {
    let origin = HeaderValue::from_bytes(&[0xff, 0xff]).expect("bytes form a valid header value");
    assert!(!is_local_origin(&origin), "non-UTF-8 origin rejected");
}

/// CORS preflight through the real router: an `OPTIONS` request carrying a
/// trusted loopback `Origin` is reflected by [`local_cors`] (the
/// `AllowOrigin::predicate` closure → `is_local_origin`), so the preflight
/// succeeds and echoes `access-control-allow-origin`. A non-loopback `Origin`
/// is NOT reflected (no allow-origin header), exercising both predicate arms.
#[tokio::test]
async fn router_cors_preflight_reflects_loopback_origin_only() {
    use super::test_support::make_app;
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    let app = make_app();

    // Trusted loopback Origin → reflected.
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("OPTIONS")
                .uri("/v1/chat/completions")
                .header("host", "127.0.0.1")
                .header("origin", "http://localhost:5173")
                .header("access-control-request-method", "POST")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        resp.headers()
            .get("access-control-allow-origin")
            .and_then(|v| v.to_str().ok()),
        Some("http://localhost:5173"),
        "loopback origin is reflected by the CORS predicate"
    );

    // Foreign Origin → predicate returns false → no allow-origin header.
    let resp = app
        .oneshot(
            Request::builder()
                .method("OPTIONS")
                .uri("/v1/chat/completions")
                .header("host", "127.0.0.1")
                .header("origin", "http://evil.example.com")
                .header("access-control-request-method", "POST")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert!(
        resp.headers().get("access-control-allow-origin").is_none(),
        "a non-loopback origin is not reflected"
    );
}

/// With a non-empty keystore, an unauthenticated control request (no
/// `Authorization` header) is rejected by `auth_guard` with `401` carrying the
/// OpenAI-style envelope and a `WWW-Authenticate: Bearer` challenge — driving
/// `auth_guard`'s key path and `unauthorized()`.
#[tokio::test]
async fn auth_guard_rejects_missing_key_with_401_envelope() {
    use super::test_support::make_higgs;
    use crate::keys::{ApiKeys, Scope};
    use http_body_util::BodyExt;
    use std::sync::Arc;
    use tower::ServiceExt;

    let higgs = make_higgs();
    let mut keys = ApiKeys::default();
    keys.add("hgk_secret", "test".into(), vec![Scope::Models]);
    higgs.set_api_keys(Arc::new(keys));
    let app = v1_router(higgs);

    let resp = app
        .oneshot(super::test_support::get("/v1/models"))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    // The 401 carries the Bearer challenge header.
    assert_eq!(
        resp.headers()
            .get(axum::http::header::WWW_AUTHENTICATE)
            .and_then(|v| v.to_str().ok()),
        Some("Bearer")
    );
    // OpenAI-style error envelope body.
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(v["error"]["code"], "unauthorized");
    assert_eq!(v["error"]["type"], "invalid_request_error");
}

/// With a non-empty keystore, a request bearing a key that GRANTS the required
/// scope passes `auth_guard` and reaches the handler (200). This drives the
/// `keys.authorizes(...) => next.run(req)` success arm of `auth_guard`.
#[tokio::test]
async fn auth_guard_admits_authorized_key() {
    use super::test_support::make_higgs;
    use crate::keys::{ApiKeys, Scope};
    use std::sync::Arc;
    use tower::ServiceExt;

    let higgs = make_higgs();
    let mut keys = ApiKeys::default();
    keys.add("hgk_models", "models".into(), vec![Scope::Models]);
    higgs.set_api_keys(Arc::new(keys));
    let app = v1_router(higgs);

    // `/v1/models` needs Models; the Models key authorizes it.
    let resp = app
        .oneshot(
            axum::http::Request::builder()
                .uri("/v1/models")
                .header("host", "127.0.0.1")
                .header(axum::http::header::AUTHORIZATION, "Bearer hgk_models")
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

/// With a non-empty keystore, a request bearing a key WITHOUT the required
/// scope (a Chat-only key hitting an Admin route) is rejected `401` — the
/// `_ => unauthorized()` arm when `authorizes` returns false.
#[tokio::test]
async fn auth_guard_rejects_insufficient_scope() {
    use super::test_support::make_higgs;
    use crate::keys::{ApiKeys, Scope};
    use std::sync::Arc;
    use tower::ServiceExt;

    let higgs = make_higgs();
    let mut keys = ApiKeys::default();
    keys.add("hgk_chat", "chat".into(), vec![Scope::Chat]);
    higgs.set_api_keys(Arc::new(keys));
    let app = v1_router(higgs);

    // `/v1/models` needs Models; a Chat-only key does NOT authorize it.
    let resp = app
        .oneshot(
            axum::http::Request::builder()
                .uri("/v1/models")
                .header("host", "127.0.0.1")
                .header(axum::http::header::AUTHORIZATION, "Bearer hgk_chat")
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

/// With a non-empty keystore, `/health` is always open (the `required_scope`
/// `None` branch in `auth_guard`: no scope required → `next.run`), so it returns
/// 200 with no Authorization header even when auth is enabled.
#[tokio::test]
async fn auth_guard_leaves_health_open() {
    use super::test_support::make_higgs;
    use crate::keys::{ApiKeys, Scope};
    use std::sync::Arc;
    use tower::ServiceExt;

    let higgs = make_higgs();
    let mut keys = ApiKeys::default();
    keys.add("hgk_admin", "admin".into(), vec![Scope::Admin]);
    higgs.set_api_keys(Arc::new(keys));
    let app = v1_router(higgs);

    let resp = app
        .oneshot(super::test_support::get("/health"))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[test]
fn http_status_mapping_table() {
    assert_eq!(
        http_status(&HiggsError::ModelNotFound { id: "x".into() }),
        StatusCode::NOT_FOUND
    );
    // [HG079] is 400 on BOTH arrival paths: the direct variant (the facade's own
    // gates) and the worker_code relay (a node's ChatHandle refusal riding a
    // WorkerRpc). Losing the relay arm demotes a permanent refusal to the 500
    // default — a retryable class OpenAI clients retry-loop (Fable r8).
    assert_eq!(
        http_status(&HiggsError::ModelNotChatCapable {
            id: "x".into(),
            arch: "bert".into(),
            domain: crate::worker::models::ModelDomain::Embedding,
        }),
        StatusCode::BAD_REQUEST
    );
    assert_eq!(
        http_status(&HiggsError::WorkerRpc {
            method: "higgs/chat".into(),
            message: "[HG079] refused".into(),
            worker_code: Some("HG079".into()),
        }),
        StatusCode::BAD_REQUEST
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
        http_status(&HiggsError::TemplateRenderFailed {
            reason: "unknown filter".into(),
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
    // G5: an OOM-exhausted load is a retryable capacity 503 — BOTH the direct
    // (local) form AND a REMOTE node's, propagated as a worker code (codex r19).
    // Fail-on-revert: dropping HG060 from the worker-code arm makes the remote
    // case fall through to 500.
    assert_eq!(
        http_status(&HiggsError::LoadOomExhausted {
            attempts: 4,
            last: "out of memory".into(),
        }),
        StatusCode::SERVICE_UNAVAILABLE,
        "local OOM-exhausted load → 503"
    );
    assert_eq!(
        http_status(&HiggsError::WorkerRpc {
            method: "higgs/load".into(),
            message: "[HG060] load ran out of memory".into(),
            worker_code: Some("HG060".into()),
        }),
        StatusCode::SERVICE_UNAVAILABLE,
        "remote OOM-exhausted load (worker code HG060) → 503, not 500"
    );
    // G6 Turbotune: a benchmark that found no fitting config is a retryable
    // capacity 503; a benchmark cancelled by a concurrent op is a 409 conflict.
    assert_eq!(
        http_status(&HiggsError::BenchExhausted {
            detail: "no fit".into()
        }),
        StatusCode::SERVICE_UNAVAILABLE
    );
    assert_eq!(
        http_status(&HiggsError::BenchCancelled),
        StatusCode::CONFLICT
    );
    // Serving disabled is a retryable capacity 503 (re-enable then retry).
    assert_eq!(
        http_status(&HiggsError::ServingDisabled),
        StatusCode::SERVICE_UNAVAILABLE
    );
    // Subsystem faults: unimplemented method → 501; a misbehaving/rejecting
    // upstream peer → 502; internal/persistence/admin/chat-task → 500.
    assert_eq!(
        http_status(&HiggsError::RpcMethodNotFound {
            endpoint: "node".into(),
            method: "higgs/bogus".into()
        }),
        StatusCode::NOT_IMPLEMENTED
    );
    assert_eq!(
        http_status(&HiggsError::ProtocolViolation {
            peer_role: "node".into(),
            detail: "x".into()
        }),
        StatusCode::BAD_GATEWAY
    );
    assert_eq!(
        http_status(&HiggsError::HubRequestRejected {
            stage: "hello".into(),
            detail: "x".into()
        }),
        StatusCode::BAD_GATEWAY
    );
    for e in [
        HiggsError::PersistenceFailed {
            store: "config".into(),
            path: "/x".into(),
            source: std::io::Error::other("x"),
        },
        HiggsError::StoreCorrupted {
            store: "config".into(),
            path: "/x".into(),
            detail: "x".into(),
        },
        HiggsError::InternalFault {
            context: "x".into(),
            detail: "y".into(),
        },
        HiggsError::HubControlFailed {
            op: "retire".into(),
            detail: "x".into(),
        },
        HiggsError::ChatTaskFailed { detail: "x".into() },
    ] {
        assert_eq!(http_status(&e), StatusCode::INTERNAL_SERVER_ERROR, "{e}");
    }
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
    // HG050: template render failed in the worker → client error (a changed
    // request — e.g. dropping tools — may succeed; identical retry won't).
    assert_eq!(http_status(&rpc(Some("HG050"))), StatusCode::BAD_REQUEST);
    assert_eq!(
        http_status(&rpc(Some("HG007"))),
        StatusCode::SERVICE_UNAVAILABLE
    );
    // HG018: resident-model swap (worker refused) → retryable 503.
    assert_eq!(
        http_status(&rpc(Some("HG018"))),
        StatusCode::SERVICE_UNAVAILABLE
    );
    // Propagated subsystem codes keep their status through the worker_code hop.
    assert_eq!(
        http_status(&rpc(Some("HG037"))),
        StatusCode::NOT_IMPLEMENTED
    );
    assert_eq!(http_status(&rpc(Some("HG038"))), StatusCode::BAD_GATEWAY);
    // Unknown / absent code falls back to 500.
    assert_eq!(
        http_status(&rpc(Some("HG011"))),
        StatusCode::INTERNAL_SERVER_ERROR
    );
    assert_eq!(http_status(&rpc(None)), StatusCode::INTERNAL_SERVER_ERROR);
}

/// [HG069] is enforced at the PUBLIC serving entrypoint ([`serve_v1`]): a
/// non-loopback listener whose keystore holds only chat/models keys (NO Admin)
/// locks out the Admin-only key-management API, so `serve_v1` must refuse it — a
/// library caller handing us its own listener can't bypass it. Fail-on-revert:
/// dropping the entrypoint admin-key check serves happily and the Err assertion
/// fails. (The [HG058] keyless-LAN twin is `serve_v1_refuses_keyless_lan_listener`.)
#[tokio::test]
async fn serve_v1_refuses_lan_listener_without_admin_key() {
    let dir = tempfile::TempDir::new().unwrap();
    super::test_support::write_gguf_fixture(dir.path(), "org/model");
    let higgs = super::test_support::make_higgs_with_lmstudio(dir.path().to_path_buf());
    higgs.load("org/model", None).await.expect("load");
    // Keystore has a chat/models key but NO Admin scope (the `keys add` default).
    let mut keys = crate::keys::ApiKeys::default();
    keys.add(
        &crate::keys::mint_token([5u8; 16]),
        "lan".into(),
        vec![crate::keys::Scope::Chat, crate::keys::Scope::Models],
    );
    higgs.set_api_keys(Arc::new(keys));

    let listener = tokio::net::TcpListener::bind("0.0.0.0:0").await.unwrap();
    let err = serve_v1(Arc::clone(&higgs), listener, async {})
        .await
        .expect_err("non-loopback listener without an Admin key must be refused");
    assert!(
        err.to_string().contains("[HG069]"),
        "refusal carries the HG069 code: {err}"
    );
    // Same no-leak guarantee as the [HG058] path: the resident worker was torn down.
    assert!(
        higgs.status().await.unwrap().loaded.is_none(),
        "the refused serve tore down the resident worker (no leak)"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal embedder token: jigglebot registers a hidden in-memory token via
// `register_internal_token`; it authenticates over the NORMAL bearer path (no
// special branch) and is invisible to the key-management list. Fail-on-revert:
// reverting the internal-token wiring makes these fail.
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn internal_token_authenticates_like_any_bearer() {
    use super::test_support::{app_for, get, make_higgs, with_bearer};
    use tower::ServiceExt;

    let higgs = make_higgs();
    higgs.register_internal_token("hgk_embedder", vec![crate::keys::Scope::Models]);

    // With the internal token → a gated route (models list = Models scope) is allowed.
    let app = app_for(higgs.clone());
    let ok = app
        .oneshot(with_bearer(get("/v1/models"), "hgk_embedder"))
        .await
        .unwrap();
    assert_eq!(ok.status(), axum::http::StatusCode::OK);

    // A caller WITHOUT the token is refused now that auth is on.
    let app = app_for(higgs);
    let denied = app.oneshot(get("/v1/models")).await.unwrap();
    assert_eq!(denied.status(), axum::http::StatusCode::UNAUTHORIZED);
}

/// The hidden internal token authenticates (auth is on) but is INVISIBLE to the
/// key list — the key-management view (now a facade concern, `Higgs::api_keys()`)
/// counts only `visible()` keys, so the embedder's in-memory token never appears.
/// Fail-on-revert: reverting the internal-token `visible()` filter surfaces it in
/// the count. (This was `GET /api/higgs/keys` before control moved in-process.)
#[tokio::test]
async fn internal_token_is_hidden_from_the_key_list() {
    use super::test_support::make_higgs;

    let higgs = make_higgs();
    higgs.register_internal_token("hgk_embedder", vec![crate::keys::Scope::Admin]);

    let keys = higgs.api_keys();
    // Auth is ON (the store is non-empty), yet no VISIBLE key is listed.
    assert!(!keys.is_empty(), "the internal token enables auth");
    assert_eq!(
        keys.visible().count(),
        0,
        "the hidden internal token must not appear in the visible key list"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// `/v1`-only router + serve entry (Task A2.1, ADDITIVE): `v1_router`/`serve_v1`
// carry the OpenAI chat+models surface with NO `/api/higgs/*` control routes.
// The control surface is ABSENT by construction (this router never had it), not
// by deletion — and the HG058/HG069 keyless-LAN refusals survive the extraction.
// ─────────────────────────────────────────────────────────────────────────────

/// `v1_router` serves `/v1/chat/completions` + `/v1/models` (401 without a
/// bearer when the keystore is non-empty), and `/api/higgs/status` is genuinely
/// ABSENT: even a valid Admin bearer 404s it (the route never existed on this
/// router — it was never registered, not deleted). Fail-on-revert: if
/// `v1_router` ever merged the `/api/higgs/*` group, the Admin-bearer status
/// request would 200 instead of 404.
#[tokio::test]
async fn v1_router_serves_v1_and_omits_control_surface() {
    use super::test_support::{get, make_higgs, post_json, with_bearer};
    use crate::keys::ApiKeys;
    use tower::ServiceExt;

    let higgs = make_higgs();
    let mut keys = ApiKeys::default();
    keys.add(
        "hgk_admin",
        "admin".into(),
        vec![Scope::Admin, Scope::Chat, Scope::Models],
    );
    higgs.set_api_keys(Arc::new(keys));
    let app = v1_router(higgs);

    // `/v1/models` needs Models; no bearer + non-empty keystore → 401.
    let resp = app.clone().oneshot(get("/v1/models")).await.unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::UNAUTHORIZED,
        "/v1/models no-bearer"
    );

    // `/v1/chat/completions` needs Chat; no bearer → 401 (auth runs before the
    // handler, so the body need not be a valid chat request).
    let resp = app
        .clone()
        .oneshot(post_json(
            "/v1/chat/completions",
            &serde_json::json!({"model": "org/model", "messages": []}),
        ))
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::UNAUTHORIZED,
        "/v1/chat/completions no-bearer"
    );

    // `/api/higgs/status` is ABSENT from `v1_router`: a VALID Admin bearer passes
    // auth, then routing 404s it — proving the control surface isn't here.
    let resp = app
        .oneshot(with_bearer(get("/api/higgs/status"), "hgk_admin"))
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::NOT_FOUND,
        "/api/higgs/status must 404 on v1_router even for an authorized admin"
    );
}

/// [HG058] is enforced at the `serve_v1` entrypoint: a non-loopback (`0.0.0.0`)
/// bind with an EMPTY keystore must be refused before serving (auth would be off
/// AND the Host guard relaxed). Fail-on-revert: dropping the HG058 check from
/// `serve_v1` serves happily and the `expect_err` fails.
#[tokio::test]
async fn serve_v1_refuses_keyless_lan_listener() {
    let dir = tempfile::TempDir::new().unwrap();
    super::test_support::write_gguf_fixture(dir.path(), "org/model");
    let higgs = super::test_support::make_higgs_with_lmstudio(dir.path().to_path_buf());
    higgs.load("org/model", None).await.expect("load");
    assert!(
        higgs.status().await.unwrap().loaded.is_some(),
        "model resident before the refused serve"
    );

    let listener = tokio::net::TcpListener::bind("0.0.0.0:0").await.unwrap();
    let err = serve_v1(Arc::clone(&higgs), listener, async {})
        .await
        .expect_err("keyless non-loopback listener must be refused");
    assert!(
        err.to_string().contains("[HG058]"),
        "refusal carries the code: {err}"
    );
    // The refused serve tore the resident worker down (codex r11) rather than
    // leaking it.
    assert!(
        higgs.status().await.unwrap().loaded.is_none(),
        "the refused serve tore down the resident worker (no leak)"
    );
}
