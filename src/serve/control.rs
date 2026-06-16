//! The `/api/higgs/*` control surface — higgs's OWN shapes (see `wire`): scan,
//! load/unload, status, system, logs, version, and worker lifecycle. Distinct
//! from the strict-OpenAI `/v1` surface in `v1`.

use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Deserialize;

use super::http_status;
use super::wire::{
    HiggsErrorResponse, HiggsLoadRequest, HiggsLoadResponse, HiggsLogsResponse, HiggsModelEntry,
    HiggsModelsResponse, HiggsOk, HiggsVersionResponse,
};
use crate::api::Higgs;
use crate::diagnostic::HiggsError;
use crate::system::SystemInfo;
use crate::worker::engine::LoadParams;
use crate::LLAMA_CPP_2_VERSION;

/// Query parameters for `GET /api/higgs/logs` (`?n=200`).
#[derive(Debug, Deserialize)]
pub(super) struct LogsQuery {
    /// Maximum number of tail lines to return (default 200).
    n: Option<usize>,
}

/// Control-route error response: mapped status + `{"error":"<display>"}` body.
fn control_error(err: &HiggsError) -> (StatusCode, Json<HiggsErrorResponse>) {
    (
        http_status(err),
        Json(HiggsErrorResponse {
            error: err.to_string(),
        }),
    )
}

/// `GET /api/higgs/models` — live scan of all configured directories, plus
/// the currently loaded model id.
// v1: two RPC round-trips (scan + status) — catalog reads are UI-paced, latency acceptable; single-RPC catalog is a v2 item
pub(super) async fn control_models(State(higgs): State<Arc<Higgs>>) -> Response {
    tracing::info!("higgs: GET /api/higgs/models");
    let models = match higgs.scan().await {
        Ok(m) => m,
        Err(err) => {
            tracing::warn!(error = %err, "higgs: scan failed");
            return control_error(&err).into_response();
        }
    };
    let loaded_id = match higgs.status().await {
        Ok(s) => s.loaded.map(|l| l.id),
        Err(err) => {
            tracing::warn!(error = %err, "higgs: status failed");
            return control_error(&err).into_response();
        }
    };
    let entries: Vec<HiggsModelEntry> = models
        .into_iter()
        .map(|m| {
            let is_loaded = loaded_id.as_deref() == Some(m.id.as_str());
            HiggsModelEntry {
                state: if is_loaded {
                    "loaded".to_owned()
                } else {
                    "not-loaded".to_owned()
                },
                format: "gguf".to_owned(),
                model: m,
            }
        })
        .collect();
    Json(HiggsModelsResponse {
        models: entries,
        loaded_id,
    })
    .into_response()
}

/// `GET /api/higgs/models/{*id}` — single enriched model by HuggingFace repo id.
///
/// The wildcard captures the full remaining path so slashed HF repo ids
/// (`org/model`, `lmstudio-community/Foo-GGUF`) round-trip correctly.
/// Returns [`HiggsModelEntry`] on success, or 404 [`HiggsErrorResponse`] when the
/// id is absent from the scanned catalog.
pub(super) async fn control_model_by_id(
    State(higgs): State<Arc<Higgs>>,
    Path(id): Path<String>,
) -> Response {
    tracing::info!(id = %id, "higgs: GET /api/higgs/models/{{*id}}");
    let models = match higgs.scan().await {
        Ok(m) => m,
        Err(err) => {
            tracing::warn!(id = %id, error = %err, "higgs: scan failed");
            return control_error(&err).into_response();
        }
    };
    let loaded_id = match higgs.status().await {
        Ok(s) => s.loaded.map(|l| l.id),
        Err(err) => {
            tracing::warn!(id = %id, error = %err, "higgs: status failed");
            return control_error(&err).into_response();
        }
    };
    match models.into_iter().find(|m| m.id == id) {
        Some(model) => {
            let is_loaded = loaded_id.as_deref() == Some(model.id.as_str());
            let entry = HiggsModelEntry {
                state: if is_loaded {
                    "loaded".to_owned()
                } else {
                    "not-loaded".to_owned()
                },
                format: "gguf".to_owned(),
                model,
            };
            Json(entry).into_response()
        }
        None => {
            let err = HiggsError::ModelNotFound { id };
            tracing::warn!(error = %err, "higgs: model not found");
            control_error(&err).into_response()
        }
    }
}

/// `POST /api/higgs/models/load` — load a model by id; absent parameters
/// fall back to the host-configured defaults.
pub(super) async fn control_load(
    State(higgs): State<Arc<Higgs>>,
    Json(req): Json<HiggsLoadRequest>,
) -> Response {
    tracing::warn!(id = %req.id, "higgs: loading model");
    let params = if req.ctx_len.is_none() && req.gpu_layers.is_none() && req.threads.is_none() {
        None // fully default load — Higgs::load applies default_load itself
    } else {
        let base = higgs.default_load();
        Some(LoadParams {
            ctx_len: req.ctx_len.unwrap_or(base.ctx_len),
            gpu_layers: req.gpu_layers.unwrap_or(base.gpu_layers),
            threads: req.threads.unwrap_or(base.threads),
        })
    };
    match higgs.load(&req.id, params).await {
        Ok(()) => Json(HiggsLoadResponse {
            status: HiggsOk::new(),
            id: req.id,
        })
        .into_response(),
        Err(err) => {
            tracing::warn!(id = %req.id, error = %err, "higgs: load failed");
            control_error(&err).into_response()
        }
    }
}

/// `POST /api/higgs/models/unload` — unload the current model.
pub(super) async fn control_unload(State(higgs): State<Arc<Higgs>>) -> Response {
    tracing::warn!("higgs: unloading model");
    match higgs.unload().await {
        Ok(()) => Json(HiggsOk::new()).into_response(),
        Err(err) => {
            tracing::warn!(error = %err, "higgs: unload failed");
            control_error(&err).into_response()
        }
    }
}

/// `GET /api/higgs/status` — live engine + model status snapshot.
pub(super) async fn control_status(State(higgs): State<Arc<Higgs>>) -> Response {
    tracing::info!("higgs: GET /api/higgs/status");
    match higgs.status().await {
        Ok(status) => Json(status).into_response(),
        Err(err) => {
            tracing::warn!(error = %err, "higgs: status failed");
            control_error(&err).into_response()
        }
    }
}

/// `GET /api/higgs/logs?n=200` — worker stderr tail, oldest first.
pub(super) async fn control_logs(
    State(higgs): State<Arc<Higgs>>,
    Query(q): Query<LogsQuery>,
) -> Json<HiggsLogsResponse> {
    tracing::info!(n = q.n.unwrap_or(200), "higgs: GET /api/higgs/logs");
    Json(HiggsLogsResponse {
        lines: higgs.logs(q.n.unwrap_or(200)),
    })
}

/// `POST /api/higgs/worker/stop` — gracefully shut down the worker.
pub(super) async fn control_worker_stop(State(higgs): State<Arc<Higgs>>) -> Json<HiggsOk> {
    tracing::warn!("higgs: stopping worker");
    higgs.stop().await;
    Json(HiggsOk::new())
}

/// `GET /api/higgs/version` — higgs build version and engine info.
pub(super) async fn control_version() -> Json<HiggsVersionResponse> {
    tracing::info!("higgs: GET /api/higgs/version");
    Json(HiggsVersionResponse {
        higgs: env!("CARGO_PKG_VERSION").to_owned(),
        engine: "llama.cpp".to_owned(),
        engine_version: crate::worker::engine::llamacpp::engine_version(),
        binding: LLAMA_CPP_2_VERSION.to_owned(),
        supported_formats: vec!["gguf".to_owned()],
    })
}

/// `GET /api/higgs/system` — host hardware (CPU/RAM/load) + inference runtime.
///
/// Gathering samples CPU load over a short interval, so it runs on a blocking
/// thread to avoid stalling the async executor.
pub(super) async fn control_system(State(higgs): State<Arc<Higgs>>) -> Json<SystemInfo> {
    tracing::info!("higgs: GET /api/higgs/system");
    // Snapshot the read-only server config (cheap lock, no I/O) on the async
    // thread, then move it into the blocking gather (which samples CPU load).
    let config = higgs.server_config();
    let info = tokio::task::spawn_blocking(move || SystemInfo::gather(config))
        .await
        .expect("system info gather task");
    Json(info)
}

#[cfg(test)]
mod tests {
    use super::super::test_support::*;
    use axum::http::StatusCode;
    use serde_json::json;
    use std::time::Duration;
    use tower::ServiceExt;

    // ── Test 6: control load + unload roundtrip ──────────────────────────────

    #[tokio::test]
    async fn control_load_unload_roundtrip() {
        let (sup, mut test_write, _test_read, _ring) = make_supervisor();
        // `load` resolves the GGUF path host-side, so the id must be discoverable.
        let dir = tempfile::TempDir::new().unwrap();
        write_gguf_fixture(dir.path(), "org/model");
        let app = make_app_with_lmstudio(sup, dir.path().to_path_buf());

        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(30)).await;
            write_response(&mut test_write, 1, json!({"id": "org/model"})).await; // higgs/load
            tokio::time::sleep(Duration::from_millis(50)).await;
            write_response(&mut test_write, 2, loaded_status_json()).await; // status (unload id capture)
            tokio::time::sleep(Duration::from_millis(30)).await;
            write_response(&mut test_write, 3, serde_json::Value::Null).await; // higgs/unload
        });

        let resp = app
            .clone()
            .oneshot(post_json(
                "/api/higgs/models/load",
                &json!({"id": "org/model"}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let v: serde_json::Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
        assert_eq!(v["status"], "ok");
        assert_eq!(v["id"], "org/model");

        let resp = app
            .oneshot(post_json("/api/higgs/models/unload", &json!({})))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let v: serde_json::Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
        assert_eq!(v["status"], "ok");
    }

    // ── Test 7: control_model_by_id with a slashed HF repo id ───────────────
    //
    // This is the regression test for the wildcard-route bug: with the old
    // single-segment `{id}` route, a request to `/api/higgs/models/org/model`
    // (literal slash in the path, as real curl sends) never matched — axum
    // treated `org` and `model` as separate segments. The test previously used
    // `org%2Fmodel` (percent-encoded) which happened to work against the broken
    // route because `%2F` is a single segment. Using a literal slash here
    // ensures the wildcard `{*id}` route is exercised as real callers do.

    #[tokio::test]
    async fn control_model_by_id_found_slashed() {
        let (sup, mut test_write, _test_read, _ring) = make_supervisor();
        // Scan runs host-side: discover the slashed id from a real GGUF fixture.
        let dir = tempfile::TempDir::new().unwrap();
        write_gguf_fixture(
            dir.path(),
            "lmstudio-community/NVIDIA-Nemotron-3-Nano-4B-GGUF",
        );
        let app = make_app_with_lmstudio(sup, dir.path().to_path_buf());

        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(30)).await;
            // Only the worker RPC remains: status (nothing loaded).
            write_response(
                &mut test_write,
                1,
                serde_json::json!({"loaded": null, "models_scanned": 1}),
            )
            .await;
        });

        // Literal slash in the URL — this is what real curl sends and what
        // the old `{id}` route could never match.
        let resp = app
            .oneshot(get(
                "/api/higgs/models/lmstudio-community/NVIDIA-Nemotron-3-Nano-4B-GGUF",
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let v: serde_json::Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
        assert_eq!(v["id"], "lmstudio-community/NVIDIA-Nemotron-3-Nano-4B-GGUF");
        assert_eq!(v["state"], "not-loaded");
        assert_eq!(v["format"], "gguf");
        assert_eq!(v["arch"], "llama");
    }

    // ── Test 8: control_model_by_id not found (slashed id) ───────────────────

    #[tokio::test]
    async fn control_model_by_id_not_found() {
        let (sup, mut test_write, _test_read, _ring) = make_supervisor();
        // Empty temp dir → host-side scan finds nothing → the id is absent.
        let dir = tempfile::TempDir::new().unwrap();
        let app = make_app_with_lmstudio(sup, dir.path().to_path_buf());

        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(30)).await;
            // Only the worker RPC remains: status (nothing loaded).
            write_response(
                &mut test_write,
                1,
                serde_json::json!({"loaded": null, "models_scanned": 0}),
            )
            .await;
        });

        // Slashed id that does not exist in the catalog → 404 HG002.
        let resp = app
            .oneshot(get("/api/higgs/models/org/nope"))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        let v: serde_json::Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
        assert!(v["error"].as_str().unwrap().contains("[HG002]"));
    }

    // ── Test 9: version endpoint ──────────────────────────────────────────────

    #[tokio::test]
    async fn version_endpoint() {
        let (sup, _test_write, _test_read, _ring) = make_supervisor();
        let app = make_app(sup);

        let resp = app.oneshot(get("/api/higgs/version")).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let v: serde_json::Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
        assert!(v["higgs"].as_str().is_some(), "higgs version present");
        assert_eq!(v["engine"], "llama.cpp");
        // engine_version is the real engine (ggml) version from ggml_version();
        // binding is the llama-cpp-2 wrapper version — distinct fields.
        assert!(v["engine_version"].as_str().is_some_and(|s| !s.is_empty()));
        assert!(v["binding"].as_str().is_some_and(|s| !s.is_empty()));
        let fmts = v["supported_formats"].as_array().expect("array");
        assert!(fmts.contains(&serde_json::Value::String("gguf".to_owned())));
    }

    // ── (original Test 7, now test 10): logs endpoint shape and tail semantics ─

    #[tokio::test]
    async fn logs_endpoint_shapes() {
        let (sup, _test_write, _test_read, ring) = make_supervisor();
        {
            let mut r = ring.lock();
            r.push_back("line one".to_owned());
            r.push_back("line two".to_owned());
            r.push_back("line three".to_owned());
        }
        let app = make_app(sup);

        let resp = app.oneshot(get("/api/higgs/logs?n=2")).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let v: serde_json::Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
        assert_eq!(
            v["lines"],
            json!(["line two", "line three"]),
            "tail of n, oldest first"
        );
    }

    // ── control_models: scan + status enrichment ─────────────────────────────

    #[tokio::test]
    async fn control_models_lists_with_loaded_flag() {
        let (sup, mut test_write, _test_read, _ring) = make_supervisor();
        // Host-side scan discovers `org/model`; the worker reports it loaded.
        let dir = tempfile::TempDir::new().unwrap();
        write_gguf_fixture(dir.path(), "org/model");
        let app = make_app_with_lmstudio(sup, dir.path().to_path_buf());

        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(30)).await;
            write_response(&mut test_write, 1, loaded_status_json()).await; // status
        });

        let resp = app.oneshot(get("/api/higgs/models")).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let v: serde_json::Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
        assert_eq!(v["loaded_id"], "org/model");
        let models = v["models"].as_array().expect("models array");
        assert_eq!(models.len(), 1);
        assert_eq!(models[0]["state"], "loaded", "loaded id is flagged");
        assert_eq!(models[0]["format"], "gguf");
        assert_eq!(models[0]["id"], "org/model");
    }

    // ── control_status passthrough ───────────────────────────────────────────

    #[tokio::test]
    async fn control_status_returns_snapshot() {
        let (sup, mut test_write, _test_read, _ring) = make_supervisor();
        let app = make_app(sup);

        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(30)).await;
            write_response(&mut test_write, 1, loaded_status_json()).await;
        });

        let resp = app.oneshot(get("/api/higgs/status")).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let v: serde_json::Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
        assert_eq!(v["loaded"]["id"], "org/model");
    }

    // ── control_load with explicit params (non-default branch) ───────────────

    #[tokio::test]
    async fn control_load_with_explicit_params() {
        let (sup, mut test_write, _test_read, _ring) = make_supervisor();
        // `load` resolves the GGUF path host-side, so the id must be discoverable.
        let dir = tempfile::TempDir::new().unwrap();
        write_gguf_fixture(dir.path(), "org/model");
        let app = make_app_with_lmstudio(sup, dir.path().to_path_buf());

        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(30)).await;
            write_response(&mut test_write, 1, json!({"id": "org/model"})).await;
            // higgs/load
        });

        // Providing ctx_len takes the param-merge branch (Some(LoadParams)).
        let resp = app
            .oneshot(post_json(
                "/api/higgs/models/load",
                &json!({"id": "org/model", "ctx_len": 2048}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let v: serde_json::Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
        assert_eq!(v["status"], "ok");
        assert_eq!(v["id"], "org/model");
    }

    // ── control_system: real host snapshot ───────────────────────────────────

    #[tokio::test]
    async fn control_system_returns_host_info() {
        let (sup, _test_write, _test_read, _ring) = make_supervisor();
        let app = make_app(sup);

        let resp = app.oneshot(get("/api/higgs/system")).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let v: serde_json::Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
        // SystemInfo always reports a positive total RAM on a real host.
        assert!(
            v.get("ram").is_some() || v.get("cpu").is_some() || v.is_object(),
            "system info is a populated object: {v}"
        );
    }

    // ── control_worker_stop: graceful, always ok ─────────────────────────────

    #[tokio::test]
    async fn control_worker_stop_ok() {
        let (sup, _test_write, _test_read, _ring) = make_supervisor();
        let app = make_app(sup);

        let resp = app
            .oneshot(post_json("/api/higgs/worker/stop", &json!({})))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let v: serde_json::Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
        assert_eq!(v["status"], "ok");
    }
}
