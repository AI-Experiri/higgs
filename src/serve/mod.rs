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
mod v1;
mod wire;

use std::sync::Arc;

use axum::http::{HeaderValue, Method, StatusCode};
use axum::routing::{get, post};
use axum::Router;
use tower_http::cors::{AllowOrigin, CorsLayer};

use crate::api::Higgs;
use crate::diagnostic::HiggsError;

// The control wire structs are part of this module's public surface (and the
// ts-rs export path), so re-export them at `crate::serve::*`.
pub use wire::*;

/// Map a `HiggsError` to its HTTP status — the single status table shared by
/// both surfaces and the SSE path: HG002/HG003 → 404, HG005 → 400,
/// HG006/HG007 → 503, else 500.
pub(crate) fn http_status(err: &HiggsError) -> StatusCode {
    match err {
        HiggsError::ModelNotFound { .. } | HiggsError::ModelNotLoaded { .. } => {
            StatusCode::NOT_FOUND
        }
        HiggsError::ContextOverflow { .. } => StatusCode::BAD_REQUEST,
        HiggsError::WorkerSpawnFailed { .. } | HiggsError::WorkerDead { .. } => {
            StatusCode::SERVICE_UNAVAILABLE
        }
        // A worker-reported error carries the worker's origin diagnostic code;
        // map by it so a worker-side HG002/HG003/HG005/HG006/HG007 reaches the
        // client as its true status instead of a generic 500.
        HiggsError::WorkerRpc { worker_code, .. } => match worker_code.as_deref() {
            Some("HG002") | Some("HG003") => StatusCode::NOT_FOUND,
            Some("HG005") => StatusCode::BAD_REQUEST,
            Some("HG006") | Some("HG007") => StatusCode::SERVICE_UNAVAILABLE,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        },
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

/// Build the higgs router: OpenAI-compatible `/v1` + `/api/higgs/*` control.
///
/// The host serves this on its own listener; all state flows through the shared
/// [`Higgs`] facade.
pub fn router(higgs: Arc<Higgs>) -> Router {
    Router::new()
        .route("/v1/models", get(v1::v1_models))
        .route("/v1/chat/completions", post(v1::v1_chat_completions))
        .route("/api/higgs/models", get(control::control_models))
        .route("/api/higgs/models/load", post(control::control_load))
        .route("/api/higgs/models/unload", post(control::control_unload))
        .route("/api/higgs/models/{*id}", get(control::control_model_by_id))
        .route("/api/higgs/status", get(control::control_status))
        .route("/api/higgs/system", get(control::control_system))
        .route("/api/higgs/logs", get(control::control_logs))
        .route(
            "/api/higgs/worker/start",
            post(control::control_worker_start),
        )
        .route("/api/higgs/worker/stop", post(control::control_worker_stop))
        .route("/api/higgs/version", get(control::control_version))
        .layer(local_cors())
        .with_state(higgs)
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
        .allow_methods([Method::GET, Method::POST])
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

// ── Shared test harness ─────────────────────────────────────────────────────
//
// Spins up a `Supervisor` over duplex pipes with a mock worker, wraps it in a
// `Higgs` facade, and builds the router — so the `v1` and `control` handler
// tests drive the real router without a real worker process. Shared by both
// surfaces' test modules via `super::super::test_support::*`.
#[cfg(test)]
pub(crate) mod test_support {
    use std::collections::VecDeque;
    use std::sync::Arc;

    use axum::body::Body;
    use axum::http::Request;
    use axum::response::Response;
    use axum::Router;
    use parking_lot::Mutex;
    use serde_json::json;
    use tokio::io::AsyncWriteExt;

    use super::router;
    use crate::api::{Higgs, HiggsConfig};
    use crate::diagnostic::HiggsError;
    use crate::rpc::{encode, RpcFrame, RpcResponse};
    use crate::supervisor::{Supervisor, WorkerHalves};

    /// Build a `Supervisor` plus duplex test handles and its captured log ring.
    pub(crate) fn make_supervisor() -> (
        Supervisor,
        tokio::io::DuplexStream, // test_write: write responses → supervisor reads
        tokio::io::DuplexStream, // test_read:  supervisor writes requests → test reads
        Arc<Mutex<VecDeque<String>>>, // stderr ring (push lines for logs tests)
    ) {
        let (sup_write, test_read) = tokio::io::duplex(64 * 1024);
        let (test_write, sup_read) = tokio::io::duplex(64 * 1024);

        let sup_write_cell = Arc::new(Mutex::new(Some(sup_write)));
        let sup_read_cell = Arc::new(Mutex::new(Some(sup_read)));
        let ring_cell: Arc<Mutex<Option<Arc<Mutex<VecDeque<String>>>>>> =
            Arc::new(Mutex::new(None));
        let ring_capture = Arc::clone(&ring_cell);

        let sup = Supervisor::with_factory(Box::new(move |ring| {
            *ring_capture.lock() = Some(ring);
            let write =
                sup_write_cell
                    .lock()
                    .take()
                    .ok_or_else(|| HiggsError::WorkerSpawnFailed {
                        source: std::io::Error::other("mock: no more write halves"),
                    })?;
            let read =
                sup_read_cell
                    .lock()
                    .take()
                    .ok_or_else(|| HiggsError::WorkerSpawnFailed {
                        source: std::io::Error::other("mock: no more read halves"),
                    })?;
            Ok(WorkerHalves {
                write: Box::new(write),
                read: Box::new(read),
                proc: None,
            })
        }));

        sup.start().expect("mock start");
        let ring = ring_cell.lock().take().expect("factory ran on start");
        (sup, test_write, test_read, ring)
    }

    /// Write a JSON-RPC success response to the supervisor's read side.
    pub(crate) async fn write_response(
        stream: &mut tokio::io::DuplexStream,
        id: u64,
        result: serde_json::Value,
    ) {
        let line = encode(&RpcFrame::Response(RpcResponse {
            jsonrpc: "2.0".into(),
            id,
            result: Some(result),
            error: None,
        }));
        stream
            .write_all(format!("{line}\n").as_bytes())
            .await
            .unwrap();
        stream.flush().await.unwrap();
    }

    /// Wrap a mock supervisor in a `Higgs` facade and build the router.
    pub(crate) fn make_app(sup: Supervisor) -> Router {
        router(Arc::new(Higgs::with_supervisor(
            Arc::new(sup),
            HiggsConfig::default(),
        )))
    }

    /// A `GET` request to `uri`.
    pub(crate) fn get(uri: &str) -> Request<Body> {
        Request::builder().uri(uri).body(Body::empty()).unwrap()
    }

    /// A `POST` request to `uri` with a JSON body.
    pub(crate) fn post_json(uri: &str, body: &serde_json::Value) -> Request<Body> {
        Request::builder()
            .method("POST")
            .uri(uri)
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap()
    }

    /// Collect a response body into bytes.
    pub(crate) async fn body_bytes(resp: Response) -> Vec<u8> {
        use http_body_util::BodyExt;
        resp.into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes()
            .to_vec()
    }

    /// A canonical `higgs/status` response with one loaded model.
    pub(crate) fn loaded_status_json() -> serde_json::Value {
        json!({
            "loaded": { "id": "org/model", "ctx_len": 4096, "gpu_layers": 99, "threads": 4 },
            "models_scanned": 1,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{http_status, is_local_origin};
    use axum::http::{HeaderValue, StatusCode};

    use crate::diagnostic::HiggsError;

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
        // Unknown / absent code falls back to 500.
        assert_eq!(
            http_status(&rpc(Some("HG011"))),
            StatusCode::INTERNAL_SERVER_ERROR
        );
        assert_eq!(http_status(&rpc(None)), StatusCode::INTERNAL_SERVER_ERROR);
    }
}
