//! Black-box integration tests for higgs's serve-layer request guards and the
//! cross-cutting misc endpoints (`src/serve/mod.rs`).
//!
//! Spawns the real `higgs` binary and drives it over HTTP, exercising:
//! - the DNS-rebinding `Host` guard (`[HG012]`, 403 on a non-loopback Host;
//!   loopback Hosts pass),
//! - the cheap `/health` + `/api/higgs/health` 200 probes,
//! - the `/api/higgs/version` build-info endpoint,
//! - the CORS preflight (`OPTIONS`) behavior for a trusted local origin,
//! - the `MAX_BODY_BYTES` body-size limit (413 on an oversized body).
//!
//! Each test picks a unique port from the 12500 base and skips cleanly when the
//! tiny GGUF is absent.

mod common;

use common::{spawn_with_tiny_model, tiny_gguf_path};

/// A `Host` header that is not a loopback host must be rejected with `403`
/// (`[HG012]` DNS-rebinding guard), while loopback Hosts (`127.0.0.1`,
/// `localhost`) are allowed through to the handler.
///
/// reqwest derives the `Host` header from the request URL; we override it with
/// an explicit `.header("host", ...)` to simulate a rebinding attack.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn forbidden_host_is_rejected_loopback_allowed() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("SKIP forbidden_host_is_rejected_loopback_allowed: tiny gguf not found (set HIGGS_TEST_GGUF)");
        return;
    };
    let srv = spawn_with_tiny_model(12500, &gguf).await;
    let c = reqwest::Client::new();

    // A forged, non-loopback Host → 403 before any handler runs (HG012).
    let resp = c
        .get(format!("{}/api/higgs/status", srv.base))
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
            .get(format!("{}/api/higgs/status", srv.base))
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
}

/// Both `/health` and `/api/higgs/health` are cheap 200 probes that answer
/// without any worker RPC (no model is loaded in this test).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn health_endpoints_return_200() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("SKIP health_endpoints_return_200: tiny gguf not found (set HIGGS_TEST_GGUF)");
        return;
    };
    let srv = spawn_with_tiny_model(12501, &gguf).await;
    let c = reqwest::Client::new();

    for path in ["/health", "/api/higgs/health"] {
        let resp = c.get(format!("{}{path}", srv.base)).send().await.unwrap();
        assert_eq!(
            resp.status(),
            reqwest::StatusCode::OK,
            "{path} must be a cheap 200"
        );
        // Health body is empty (StatusCode only); drain to free the connection.
        let _ = resp.text().await.unwrap();
    }
}

/// `GET /api/higgs/version` returns the crate version (`CARGO_PKG_VERSION`)
/// plus engine info: `engine == "llama.cpp"`, a non-empty `engine_version`, a
/// `binding`, and `supported_formats` listing `gguf`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn version_endpoint_reports_crate_version() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("SKIP version_endpoint_reports_crate_version: tiny gguf not found (set HIGGS_TEST_GGUF)");
        return;
    };
    let srv = spawn_with_tiny_model(12502, &gguf).await;
    let c = reqwest::Client::new();

    let resp = c
        .get(format!("{}/api/higgs/version", srv.base))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let v: serde_json::Value = resp.json().await.unwrap();

    // The crate version comes from this very crate's Cargo.toml at build time,
    // so it must match the integration-test binary's own CARGO_PKG_VERSION.
    assert_eq!(
        v["higgs"].as_str(),
        Some(env!("CARGO_PKG_VERSION")),
        "version endpoint reports the higgs crate version"
    );
    assert_eq!(v["engine"], "llama.cpp", "engine is llama.cpp");
    assert!(
        v["engine_version"].as_str().is_some_and(|s| !s.is_empty()),
        "engine_version present and non-empty: {v}"
    );
    assert!(
        v["binding"].as_str().is_some_and(|s| !s.is_empty()),
        "binding crate version present: {v}"
    );
    let formats = v["supported_formats"].as_array().unwrap();
    assert!(
        formats.iter().any(|f| f == "gguf"),
        "supported_formats lists gguf: {v}"
    );
}

/// A CORS preflight (`OPTIONS`) from a trusted local origin
/// (`http://localhost:5173`, the Vite dev origin) is answered with that origin
/// reflected in `access-control-allow-origin` and the configured allowed
/// methods (`local_cors` allows GET/POST/PUT) — proving the CORS layer is wired.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cors_preflight_allows_local_origin() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!(
            "SKIP cors_preflight_allows_local_origin: tiny gguf not found (set HIGGS_TEST_GGUF)"
        );
        return;
    };
    let srv = spawn_with_tiny_model(12503, &gguf).await;
    let c = reqwest::Client::new();

    let resp = c
        .request(
            reqwest::Method::OPTIONS,
            format!("{}/api/higgs/models/load", srv.base),
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
}

/// An oversized request body (> `MAX_BODY_BYTES`, 32 MiB) is rejected with
/// `413 Payload Too Large` by the `DefaultBodyLimit` layer — before the handler
/// ever deserializes it. Sent with a loopback Host so the body-limit layer
/// (not the rebinding guard) is what does the rejecting.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn oversized_body_is_rejected_413() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("SKIP oversized_body_is_rejected_413: tiny gguf not found (set HIGGS_TEST_GGUF)");
        return;
    };
    let srv = spawn_with_tiny_model(12504, &gguf).await;
    let c = reqwest::Client::new();

    // 32 MiB + 1 byte — one over the documented MAX_BODY_BYTES cap.
    let too_big = vec![b'x'; 32 * 1024 * 1024 + 1];
    let resp = c
        .post(format!("{}/api/higgs/models/load", srv.base))
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
}
