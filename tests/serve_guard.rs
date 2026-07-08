//! Black-box integration tests for higgs's serve-layer request guards
//! (`src/serve/mod.rs`), migrated to the library-first surface.
//!
//! higgs is now library-first: control lives on the in-process `Higgs` crate API
//! and the ONLY socket surface is the strict OpenAI `/v1` router. These tests
//! drive the real `serve_v1` router (via [`serve_v1_local`], an ephemeral loopback
//! bind) to exercise:
//! - the DNS-rebinding `Host` guard (`[HG012]`, 403 on a non-loopback Host;
//!   loopback Hosts pass),
//! - the cheap `/health` 200 probe (no worker RPC),
//! - the CORS preflight (`OPTIONS`) behavior for a trusted local origin,
//! - the `MAX_BODY_BYTES` body-size limit (413 on an oversized body),
//! - the `[HG058]` keyless-LAN refusal (`serve_v1` on a non-loopback listener with
//!   an empty keystore returns `Err`).
//!
//! The build-info endpoint (`GET /api/higgs/version`) moved off HTTP to the
//! `Higgs::version()` crate method, so that test now calls the facade directly.
//!
//! Each test skips cleanly when the tiny GGUF is absent (the harness serializes
//! in-process instances via a process-global lock).

mod common;

use common::{higgs_local, serve_v1_local, TINY_MODEL_ID};

/// A `Host` header that is not a loopback host must be rejected with `403`
/// (`[HG012]` DNS-rebinding guard), while loopback Hosts (`127.0.0.1`,
/// `localhost`) are allowed through to the handler.
///
/// reqwest derives the `Host` header from the request URL; we override it with
/// an explicit `.header("host", ...)` to simulate a rebinding attack. The real
/// `/v1` router (loopback bind → Host guard ON) is served via `serve_v1_local`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn forbidden_host_is_rejected_loopback_allowed() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP forbidden_host_is_rejected_loopback_allowed: tiny gguf not found (set HIGGS_TEST_GGUF)");
        return;
    };
    let (base, guard) = serve_v1_local(higgs.handle()).await;
    let c = reqwest::Client::new();

    // A forged, non-loopback Host → 403 before any handler runs (HG012).
    let resp = c
        .get(format!("{base}/v1/models"))
        .header("host", "evil.example")
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        reqwest::StatusCode::FORBIDDEN,
        "non-loopback Host must be 403 (HG012)"
    );
    let body = resp.text().await.unwrap();
    assert!(
        body.contains("HG012"),
        "403 body carries the HG012 diagnostic, got: {body}"
    );

    // Loopback Hosts pass the guard and reach the real handler (200).
    for host in ["127.0.0.1", "localhost"] {
        let resp = c
            .get(format!("{base}/v1/models"))
            .header("host", host)
            .send()
            .await
            .unwrap();
        assert_eq!(
            resp.status(),
            reqwest::StatusCode::OK,
            "loopback Host {host} must pass the guard"
        );
        // Drain the body so the connection is freed (no SSE here, plain JSON).
        let _ = resp.text().await.unwrap();
    }

    guard.shutdown().await;
    higgs.shutdown().await;
}

/// `/health` is a cheap 200 probe that answers without any worker RPC (no model
/// is loaded in this test). The old `/api/higgs/health` twin was removed with the
/// `/api/higgs/*` surface; `/health` on the `/v1` router is the single probe now.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn health_endpoints_return_200() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP health_endpoints_return_200: tiny gguf not found (set HIGGS_TEST_GGUF)");
        return;
    };
    let (base, guard) = serve_v1_local(higgs.handle()).await;
    let c = reqwest::Client::new();

    let resp = c.get(format!("{base}/health")).send().await.unwrap();
    assert_eq!(
        resp.status(),
        reqwest::StatusCode::OK,
        "/health must be a cheap 200"
    );
    // Health body is empty (StatusCode only); drain to free the connection.
    let _ = resp.text().await.unwrap();

    guard.shutdown().await;
    higgs.shutdown().await;
}

/// `Higgs::version()` returns the crate version (`CARGO_PKG_VERSION`) plus engine
/// info: `engine == "llama.cpp"`, a non-empty `engine_version`, a `binding`, and
/// `supported_formats` listing `gguf`. Formerly `GET /api/higgs/version`; the
/// build-info now lives on the in-process crate method (the version strings are
/// crate-internal and unreadable across the crate boundary otherwise).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn version_endpoint_reports_crate_version() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP version_endpoint_reports_crate_version: tiny gguf not found (set HIGGS_TEST_GGUF)");
        return;
    };

    let v = higgs.version();

    // The crate version comes from this very crate's Cargo.toml at build time,
    // so it must match the integration-test binary's own CARGO_PKG_VERSION.
    assert_eq!(
        v.higgs,
        env!("CARGO_PKG_VERSION"),
        "version reports the higgs crate version"
    );
    assert_eq!(v.engine, "llama.cpp", "engine is llama.cpp");
    assert!(
        !v.engine_version.is_empty(),
        "engine_version present and non-empty: {v:?}"
    );
    assert!(
        !v.binding.is_empty(),
        "binding crate version present: {v:?}"
    );
    assert!(
        v.supported_formats.iter().any(|f| f == "gguf"),
        "supported_formats lists gguf: {v:?}"
    );

    higgs.shutdown().await;
}

/// A CORS preflight (`OPTIONS`) from a trusted local origin
/// (`http://localhost:5173`, the Vite dev origin) is answered with that origin
/// reflected in `access-control-allow-origin` and the configured allowed
/// methods (`local_cors` allows GET/POST/PUT/DELETE) — proving the CORS layer is
/// wired on the `/v1` router. Targets `/v1/chat/completions` (a POST route).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cors_preflight_allows_local_origin() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!(
            "SKIP cors_preflight_allows_local_origin: tiny gguf not found (set HIGGS_TEST_GGUF)"
        );
        return;
    };
    let (base, guard) = serve_v1_local(higgs.handle()).await;
    let c = reqwest::Client::new();

    let resp = c
        .request(
            reqwest::Method::OPTIONS,
            format!("{base}/v1/chat/completions"),
        )
        // A loopback Host so the rebinding guard lets the preflight through.
        .header("host", "127.0.0.1")
        .header("origin", "http://localhost:5173")
        .header("access-control-request-method", "POST")
        .send()
        .await
        .unwrap();

    // tower-http's CorsLayer answers a valid preflight with a 2xx and reflects
    // the (trusted) origin back. The exact status is 200; assert success + the
    // allow-origin echo, which is what the browser actually checks.
    assert!(
        resp.status().is_success(),
        "CORS preflight from a local origin must succeed, got {}",
        resp.status()
    );
    let allow_origin = resp
        .headers()
        .get("access-control-allow-origin")
        .and_then(|v| v.to_str().ok())
        .map(ToOwned::to_owned);
    assert_eq!(
        allow_origin.as_deref(),
        Some("http://localhost:5173"),
        "preflight reflects the trusted local origin"
    );
    let allow_methods = resp
        .headers()
        .get("access-control-allow-methods")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_ascii_uppercase();
    assert!(
        allow_methods.contains("POST"),
        "preflight advertises the allowed methods (POST among them): {allow_methods:?}"
    );
    let _ = resp.text().await.unwrap();

    guard.shutdown().await;
    higgs.shutdown().await;
}

/// An oversized request body (> `MAX_BODY_BYTES`, 32 MiB) is rejected with
/// `413 Payload Too Large` by the `DefaultBodyLimit` layer — before the handler
/// ever deserializes it. Sent with a loopback Host so the body-limit layer
/// (not the rebinding guard) is what does the rejecting. Targets the POST
/// `/v1/chat/completions` route (the largest legitimate body on the `/v1`
/// surface).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn oversized_body_is_rejected_413() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP oversized_body_is_rejected_413: tiny gguf not found (set HIGGS_TEST_GGUF)");
        return;
    };
    let (base, guard) = serve_v1_local(higgs.handle()).await;
    let c = reqwest::Client::new();

    // 32 MiB + 1 byte — one over the documented MAX_BODY_BYTES cap.
    let too_big = vec![b'x'; 32 * 1024 * 1024 + 1];
    let resp = c
        .post(format!("{base}/v1/chat/completions"))
        .header("host", "127.0.0.1")
        .header("content-type", "application/json")
        .body(too_big)
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        reqwest::StatusCode::PAYLOAD_TOO_LARGE,
        "an oversized body must be 413 (MAX_BODY_BYTES)"
    );
    let _ = resp.text().await.unwrap();

    guard.shutdown().await;
    higgs.shutdown().await;
}

/// `[HG058]` is enforced at the public serving entrypoint: `serve_v1` on a
/// NON-loopback (`0.0.0.0`) listener with an EMPTY keystore must be refused
/// before it can serve — on such a bind auth would be OFF and the Host guard
/// relaxed, exposing the whole surface to the network. The refusal is an
/// `io::Error` carrying `[HG058]`, and it tears the resident worker down (no
/// leak) rather than serving.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn serve_v1_refuses_keyless_lan_listener() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!(
            "SKIP serve_v1_refuses_keyless_lan_listener: tiny gguf not found (set HIGGS_TEST_GGUF)"
        );
        return;
    };

    // Bring a worker up so we can prove the refused serve tears it down.
    higgs
        .load(TINY_MODEL_ID, None)
        .await
        .expect("load tiny model");
    assert!(
        higgs.status().await.unwrap().loaded.is_some(),
        "model resident before the refused serve"
    );

    let listener = tokio::net::TcpListener::bind("0.0.0.0:0")
        .await
        .expect("bind 0.0.0.0");
    let err = higgs::serve::serve_v1(higgs.handle(), listener, async {})
        .await
        .expect_err("keyless non-loopback listener must be refused");
    assert!(
        err.to_string().contains("[HG058]"),
        "refusal carries the HG058 code: {err}"
    );
    // The refused serve tore the resident worker down (codex r11) rather than
    // leaking it.
    assert!(
        higgs.status().await.unwrap().loaded.is_none(),
        "the refused serve tore down the resident worker (no leak)"
    );

    higgs.shutdown().await;
}
