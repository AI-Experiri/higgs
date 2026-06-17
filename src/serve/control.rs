//! The `/api/higgs/*` control surface — higgs's OWN shapes (see `wire`): scan,
//! load/unload, status, system, logs, version, and worker lifecycle. Distinct
//! from the strict-OpenAI `/v1` surface in `v1`.

use std::convert::Infallible;
use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures::Stream;
use serde::Deserialize;
use tokio::sync::{broadcast, mpsc};

use super::http_status;
use super::wire::{
    HiggsErrorResponse, HiggsLoadRequest, HiggsLoadResponse, HiggsLogsResponse, HiggsModelEntry,
    HiggsModelsResponse, HiggsOk, HiggsVersionResponse, LogSettings,
};
use crate::api::Higgs;
use crate::diagnostic::HiggsError;
use crate::system::SystemInfo;
use crate::worker::engine::LoadParams;
use crate::worker::models::HiggsModel;
use crate::LLAMA_CPP_2_VERSION;

/// Query parameters for `GET /api/higgs/logs` and `/api/higgs/logs/stream`
/// (`?n=200`).
#[derive(Debug, Deserialize)]
pub(super) struct LogsQuery {
    /// Maximum number of history lines to return / replay (default 200).
    n: Option<usize>,
}

/// Default history depth for the logs snapshot and the SSE replay prefix.
const DEFAULT_LOG_LINES: usize = 200;

/// Control-route error response: mapped status + `{"error":"<display>"}` body.
fn control_error(err: &HiggsError) -> (StatusCode, Json<HiggsErrorResponse>) {
    (
        http_status(err),
        Json(HiggsErrorResponse {
            error: err.to_string(),
        }),
    )
}

/// Host-side scan of all configured directories plus the currently loaded model
/// id. The scan is pure Rust (no worker); only the `status()` call is a worker
/// RPC. `Err(response)` carries the mapped control error on either failure.
async fn scan_with_loaded(
    higgs: &Arc<Higgs>,
) -> Result<(Vec<HiggsModel>, Option<String>), Response> {
    let models = higgs.scan().await.map_err(|err| {
        tracing::warn!(error = %err, "higgs: scan failed");
        control_error(&err).into_response()
    })?;
    let loaded_id = higgs
        .status()
        .await
        .map(|s| s.loaded.map(|l| l.id))
        .map_err(|err| {
            tracing::warn!(error = %err, "higgs: status failed");
            control_error(&err).into_response()
        })?;
    Ok((models, loaded_id))
}

/// Build the per-model control entry: the canonical [`HiggsModel`] enriched with
/// its load `state` (flagged against `loaded_id`) and `format`.
fn model_entry(model: HiggsModel, loaded_id: Option<&str>) -> HiggsModelEntry {
    let is_loaded = loaded_id == Some(model.id.as_str());
    HiggsModelEntry {
        state: if is_loaded {
            "loaded".to_owned()
        } else {
            "not-loaded".to_owned()
        },
        format: "gguf".to_owned(),
        model,
    }
}

/// `GET /api/higgs/models` — live scan of all configured directories, plus
/// the currently loaded model id.
pub(super) async fn control_models(State(higgs): State<Arc<Higgs>>) -> Response {
    tracing::info!("higgs: GET /api/higgs/models");
    let (models, loaded_id) = match scan_with_loaded(&higgs).await {
        Ok(pair) => pair,
        Err(resp) => return resp,
    };
    let entries: Vec<HiggsModelEntry> = models
        .into_iter()
        .map(|m| model_entry(m, loaded_id.as_deref()))
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
    let (models, loaded_id) = match scan_with_loaded(&higgs).await {
        Ok(pair) => pair,
        Err(resp) => return resp,
    };
    match models.into_iter().find(|m| m.id == id) {
        Some(model) => Json(model_entry(model, loaded_id.as_deref())).into_response(),
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
    // A request with NO load parameter at all is a fully-default load —
    // `Higgs::load` applies `default_load` itself (and ctx_len auto-cap). The
    // moment ANY field is pinned, build a LoadParams: the three base fields fall
    // back to `default_load`, every optional override passes through verbatim
    // (absent = `None` = engine default).
    let any_pinned = req.ctx_len.is_some()
        || req.gpu_layers.is_some()
        || req.threads.is_some()
        || req.use_mmap.is_some()
        || req.use_mlock.is_some()
        || req.n_batch.is_some()
        || req.n_ubatch.is_some()
        || req.offload_kqv.is_some()
        || req.rope_freq_base.is_some()
        || req.rope_freq_scale.is_some()
        || req.flash_attn.is_some()
        || req.type_k.is_some()
        || req.type_v.is_some()
        || req.seed.is_some();
    let params = if !any_pinned {
        None
    } else {
        let base = higgs.default_load();
        Some(LoadParams {
            ctx_len: req.ctx_len.unwrap_or(base.ctx_len),
            gpu_layers: req.gpu_layers.unwrap_or(base.gpu_layers),
            threads: req.threads.unwrap_or(base.threads),
            use_mmap: req.use_mmap,
            use_mlock: req.use_mlock,
            n_batch: req.n_batch,
            n_ubatch: req.n_ubatch,
            offload_kqv: req.offload_kqv,
            rope_freq_base: req.rope_freq_base,
            rope_freq_scale: req.rope_freq_scale,
            flash_attn: req.flash_attn,
            type_k: req.type_k,
            type_v: req.type_v,
            seed: req.seed,
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

/// `GET /api/higgs/logs?n=200` — Developer-Log tail (worker stderr + captured
/// serve-layer request events), oldest first. Point-in-time snapshot; for a
/// live feed use `GET /api/higgs/logs/stream`.
pub(super) async fn control_logs(
    State(higgs): State<Arc<Higgs>>,
    Query(q): Query<LogsQuery>,
) -> Json<HiggsLogsResponse> {
    tracing::info!(
        n = q.n.unwrap_or(DEFAULT_LOG_LINES),
        "higgs: GET /api/higgs/logs"
    );
    Json(HiggsLogsResponse {
        lines: higgs.logs(q.n.unwrap_or(DEFAULT_LOG_LINES)),
    })
}

/// `GET /api/higgs/logs/settings` — current Developer-Log toggle state
/// ("Verbose Logging" and "Log Incoming Tokens").
pub(super) async fn control_logs_settings(State(higgs): State<Arc<Higgs>>) -> Json<LogSettings> {
    tracing::info!("higgs: GET /api/higgs/logs/settings");
    Json(LogSettings {
        verbose: higgs.verbose(),
        log_incoming_tokens: higgs.log_incoming_tokens(),
    })
}

/// `PUT /api/higgs/logs/settings` — set both Developer-Log toggles ("Verbose
/// Logging" and "Log Incoming Tokens"). A state-changing control op, so it logs
/// at `warn` with the new values.
pub(super) async fn control_set_logs_settings(
    State(higgs): State<Arc<Higgs>>,
    Json(body): Json<LogSettings>,
) -> Json<HiggsOk> {
    tracing::warn!(
        verbose = body.verbose,
        log_incoming_tokens = body.log_incoming_tokens,
        "higgs: set developer-log toggles"
    );
    higgs.set_verbose(body.verbose);
    higgs.set_log_incoming_tokens(body.log_incoming_tokens);
    Json(HiggsOk::new())
}

/// `GET /api/higgs/logs/stream?n=200` — LIVE Developer Logs over SSE.
///
/// Replays the last `n` history lines first (so a fresh subscriber sees recent
/// context), then streams every new line as it is produced. Each line — worker
/// stderr or a captured serve-layer request event — arrives as one SSE `data:`
/// frame. A slow client that overflows the broadcast buffer gets a `Lagged`
/// from the receiver: the handler logs a warning, emits a gap marker, and keeps
/// streaming rather than dropping the connection. Stream ends on client
/// disconnect or when the bus sender is permanently closed.
pub(super) async fn control_logs_stream(
    State(higgs): State<Arc<Higgs>>,
    Query(q): Query<LogsQuery>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let n = q.n.unwrap_or(DEFAULT_LOG_LINES);
    tracing::info!(n, "higgs: GET /api/higgs/logs/stream");

    // Subscribe BEFORE snapshotting so no line slips between replay and live —
    // a line that lands in this window is duplicated at worst, never lost.
    let mut rx = higgs.subscribe_logs();
    let replay = higgs.logs(n);

    // Pump replay-then-live lines into a channel, then `unfold` the receiver —
    // the same Sse construction shape as the chat SSE path (`serve::stream`).
    let (tx, out) = mpsc::unbounded_channel::<String>();
    tokio::spawn(async move {
        for line in replay {
            if tx.send(line).is_err() {
                return; // client gone before live phase
            }
        }
        loop {
            match rx.recv().await {
                Ok(line) => {
                    if tx.send(line).is_err() {
                        return; // client disconnected
                    }
                }
                Err(broadcast::error::RecvError::Lagged(skipped)) => {
                    tracing::warn!(skipped, "higgs: log stream subscriber lagged");
                    let marker = format!("[log stream lagged — dropped {skipped} lines]");
                    if tx.send(marker).is_err() {
                        return;
                    }
                }
                // Sender dropped (worker+bus gone) — end the stream cleanly.
                Err(broadcast::error::RecvError::Closed) => return,
            }
        }
    });

    let stream = futures::stream::unfold(out, |mut out| async move {
        out.recv()
            .await
            .map(|line| (Ok::<_, Infallible>(Event::default().data(line)), out))
    });

    Sse::new(stream).keep_alive(KeepAlive::default())
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
        let (sup, _test_write, _test_read, bus) = make_supervisor();
        bus.push("line one".to_owned());
        bus.push("line two".to_owned());
        bus.push("line three".to_owned());
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

    // ── logs SSE stream: replay-then-live ordering ───────────────────────────

    #[tokio::test]
    async fn logs_stream_replays_then_streams_live() {
        use super::{control_logs_stream, LogsQuery};
        use crate::api::{Higgs, HiggsConfig};
        use axum::extract::{Query, State};
        use axum::response::IntoResponse;
        use futures::StreamExt;
        use std::sync::Arc;
        use std::time::Duration;

        let (sup, _test_write, _test_read, bus) = make_supervisor();
        // Seed history BEFORE the request — this is the replay prefix.
        bus.push("hist-1".to_owned());
        bus.push("hist-2".to_owned());

        let higgs = Arc::new(Higgs::with_supervisor(
            Arc::new(sup),
            HiggsConfig::default(),
        ));
        let resp = control_logs_stream(State(higgs.clone()), Query(LogsQuery { n: Some(10) }))
            .await
            .into_response();
        assert_eq!(
            resp.headers()
                .get(axum::http::header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok()),
            Some("text/event-stream"),
            "SSE content-type"
        );

        let mut body = resp.into_body().into_data_stream();

        // The replay prefix arrives first; collect frames until both history
        // lines have been seen.
        let mut seen = String::new();
        while !(seen.contains("hist-1") && seen.contains("hist-2")) {
            let frame = tokio::time::timeout(Duration::from_secs(2), body.next())
                .await
                .expect("replay frame within timeout")
                .expect("body not ended")
                .expect("frame ok");
            seen.push_str(&String::from_utf8_lossy(&frame));
        }
        assert!(
            seen.contains("hist-1") && seen.contains("hist-2"),
            "replay: {seen}"
        );

        // After replay the stream is parked on the live receiver. Push a new line
        // and it must arrive as a frame — proving live delivery, not closure.
        bus.push("live-1".to_owned());
        let mut live_seen = String::new();
        while !live_seen.contains("live-1") {
            let frame = tokio::time::timeout(Duration::from_secs(2), body.next())
                .await
                .expect("live frame within timeout")
                .expect("body not ended")
                .expect("frame ok");
            live_seen.push_str(&String::from_utf8_lossy(&frame));
        }
        assert!(live_seen.contains("live-1"), "live: {live_seen}");

        // The replay source itself is ordered oldest-first ahead of the live line.
        assert_eq!(
            higgs.logs(10),
            vec![
                "hist-1".to_owned(),
                "hist-2".to_owned(),
                "live-1".to_owned()
            ]
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

    // ── logs settings: GET reflects default; PUT toggles verbose ─────────────

    #[tokio::test]
    async fn logs_settings_get_default_and_put_toggles() {
        let (sup, _test_write, _test_read, _ring) = make_supervisor();
        let app = make_app(sup);

        // GET defaults to verbose:false.
        let resp = app
            .clone()
            .oneshot(get("/api/higgs/logs/settings"))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let v: serde_json::Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
        assert_eq!(v["verbose"], false, "verbose defaults to false");
        assert_eq!(
            v["log_incoming_tokens"], false,
            "log_incoming_tokens defaults to false"
        );

        // PUT both flags true returns {"status":"ok"}.
        let resp = app
            .clone()
            .oneshot(put_json(
                "/api/higgs/logs/settings",
                &json!({"verbose": true, "log_incoming_tokens": true}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let v: serde_json::Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
        assert_eq!(v["status"], "ok");

        // GET now reflects the new state for both flags.
        let resp = app.oneshot(get("/api/higgs/logs/settings")).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
        assert_eq!(v["verbose"], true, "PUT toggled verbose on");
        assert_eq!(
            v["log_incoming_tokens"], true,
            "PUT toggled log_incoming_tokens on"
        );
    }

    // ── verbose gate: served line appears only when verbose is on ─────────────

    #[tokio::test]
    async fn verbose_gate_round_trips_through_handlers() {
        use super::{control_logs_settings, control_set_logs_settings};
        use crate::api::Higgs;
        use axum::extract::State;
        use std::sync::Arc;

        let (sup, _test_write, _test_read, _ring) = make_supervisor();
        let higgs = Arc::new(Higgs::with_supervisor(
            Arc::new(sup),
            crate::api::HiggsConfig::default(),
        ));

        // GET handler reflects the default-off state.
        assert!(
            !control_logs_settings(State(higgs.clone())).await.0.verbose,
            "verbose off by default"
        );

        // PUT handler flips it on; the chat path's gate (`higgs.verbose()`) now
        // returns true, so the served line would be emitted (format asserted in
        // v1's `served_message_format`).
        let ok = control_set_logs_settings(
            State(higgs.clone()),
            axum::Json(crate::serve::LogSettings {
                verbose: true,
                log_incoming_tokens: true,
            }),
        )
        .await;
        assert_eq!(ok.0.status, "ok");
        assert!(higgs.verbose(), "PUT enabled the chat verbose gate");
        assert!(
            higgs.log_incoming_tokens(),
            "PUT enabled the incoming-tokens gate"
        );
        let got = control_logs_settings(State(higgs.clone())).await.0;
        assert!(got.verbose, "GET reflects verbose on");
        assert!(
            got.log_incoming_tokens,
            "GET reflects log_incoming_tokens on"
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
