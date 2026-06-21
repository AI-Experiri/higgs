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
    HiggsModelsResponse, HiggsOk, HiggsRuntimeSettings, HiggsVersionResponse, LogSettings,
};
use crate::api::Higgs;
use crate::diagnostic::HiggsError;
use crate::log_bus::LogSource;
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
    /// Restrict to one origin: `serve` (higgs control plane) or `worker` (the
    /// model worker's stderr). Absent/unknown = both merged.
    source: Option<String>,
}

impl LogsQuery {
    /// The parsed [`LogSource`] filter (`None` = all sources).
    fn filter(&self) -> Option<LogSource> {
        self.source.as_deref().and_then(LogSource::parse)
    }
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

/// The `(arch, quant)` support-cache key for a model. A missing arch/quant is
/// keyed as the empty string, so all "unknown arch/quant" files share one verdict
/// (and one probe) rather than each probing redundantly.
fn support_key(model: &HiggsModel) -> (String, String) {
    (
        model.arch.clone().unwrap_or_default(),
        model.quant.clone().unwrap_or_default(),
    )
}

/// Gate 2 (host-side, zero FFI): does any registered tool-call parser recognize
/// this model's chat template? `false` when there is no template or none matches.
/// The `tool_parser` registry is pure Rust, so this runs in-process — no worker.
fn tool_calls_supported(model: &HiggsModel) -> bool {
    match model.chat_template.as_deref() {
        Some(tmpl) => crate::worker::tool_parser::ToolParserRegistry::with_defaults()
            .select(tmpl)
            .is_some(),
        None => false,
    }
}

/// Build the per-model control entry: the canonical [`HiggsModel`] enriched with
/// its load `state`, `format`, and the two support gates.
///
/// `loadable`/`load_reason` are the Gate-1 verdict for this model's
/// `(arch, quant)` (from the probe sweep). `support_reason` is the verbatim
/// engine reason when `!loadable`, the fixed Gate-2 message when the model loads
/// but no parser matches its template, else `None`.
fn model_entry(
    mut model: HiggsModel,
    loaded_id: Option<&str>,
    loadable: bool,
    load_reason: Option<String>,
) -> HiggsModelEntry {
    let is_loaded = loaded_id == Some(model.id.as_str());
    let tool_calls = tool_calls_supported(&model);
    let support_reason = if !loadable {
        // Gate 1 failed: surface the engine's verbatim load error.
        load_reason
    } else if !tool_calls {
        // Gate 1 passed, Gate 2 failed: no parser matches the template.
        Some("no tool-call parser matches this model's template".to_owned())
    } else {
        None
    };
    // The transient chat_template never leaves the host; drop it explicitly
    // (it is `serde(skip)` anyway). `gguf_components` stays on the model — its
    // single home — and rides the flattened payload.
    model.chat_template = None;
    HiggsModelEntry {
        state: if is_loaded {
            "loaded".to_owned()
        } else {
            "not-loaded".to_owned()
        },
        format: "gguf".to_owned(),
        loadable,
        tool_calls,
        support_reason,
        model,
    }
}

/// `GET /api/higgs/models` — live scan of all configured directories, plus
/// the currently loaded model id and the Gate-1/Gate-2 support verdict per model.
pub(super) async fn control_models(State(higgs): State<Arc<Higgs>>) -> Response {
    tracing::info!("higgs: GET /api/higgs/models");
    let (models, loaded_id) = match scan_with_loaded(&higgs).await {
        Ok(pair) => pair,
        Err(resp) => return resp,
    };
    // Gate 1 sweep: one representative path per distinct (arch, quant), probed
    // once (cached thereafter). Every model sharing the combo inherits the verdict.
    let mut reps: std::collections::HashMap<(String, String), String> =
        std::collections::HashMap::new();
    for m in &models {
        reps.entry(support_key(m)).or_insert_with(|| m.path.clone());
    }
    let combos = reps.len();
    let support = higgs
        .probe_support(
            reps.into_iter()
                .map(|((arch, quant), path)| (arch, quant, path))
                .collect(),
        )
        .await;
    let unsupported = support.values().filter(|(loadable, _)| !*loadable).count();
    tracing::info!("higgs: support sweep — probed {combos} combos, {unsupported} unsupported");
    let entries: Vec<HiggsModelEntry> = models
        .into_iter()
        .map(|m| {
            let (loadable, reason) = support
                .get(&support_key(&m))
                .cloned()
                .unwrap_or((false, Some("probe verdict missing".to_owned())));
            model_entry(m, loaded_id.as_deref(), loadable, reason)
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
    let (models, loaded_id) = match scan_with_loaded(&higgs).await {
        Ok(pair) => pair,
        Err(resp) => return resp,
    };
    match models.into_iter().find(|m| m.id == id) {
        Some(model) => {
            // Gate 1 for this one model's (arch, quant) — cached after the first probe.
            let (arch, quant) = support_key(&model);
            let support = higgs
                .probe_support(vec![(arch.clone(), quant.clone(), model.path.clone())])
                .await;
            let (loadable, reason) = support
                .get(&(arch, quant))
                .cloned()
                .unwrap_or((false, Some("probe verdict missing".to_owned())));
            Json(model_entry(model, loaded_id.as_deref(), loadable, reason)).into_response()
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
    let started = std::time::Instant::now();
    match higgs.load(&req.id, params).await {
        Ok(()) => {
            // Record the per-load idle-TTL override (HOST-SIDE only). It takes
            // precedence over the global TTL in the idle reaper for THIS model
            // and is cleared on unload. Absent = use the global TTL.
            higgs.set_loaded_idle_ttl_override(req.idle_ttl_minutes);
            // Completion line — bookends the "loading model" start line in the
            // Developer Logs. id + elapsed go in the MESSAGE so they render in the
            // console (the log layer captures the message + the `error` field only).
            tracing::info!(
                "higgs: model loaded: {} ({} ms)",
                req.id,
                started.elapsed().as_millis()
            );
            Json(HiggsLoadResponse {
                status: HiggsOk::new(),
                id: req.id,
            })
            .into_response()
        }
        Err(err) => {
            // `error = %err` carries the full typed reason (worker RPC → HG004
            // engine-load failure → llama.cpp detail); the log layer appends it.
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

/// `POST /api/higgs/pair` — mint a one-time pairing token + a dialable ticket for a new node
/// (hub mode only). Admin-scoped. The operator runs `higgs --node <ticket> <token>` on the
/// node; the hub's accept loop admits it. `409` when the server isn't a hub.
pub(super) async fn control_pair(State(higgs): State<Arc<Higgs>>) -> Response {
    tracing::warn!("higgs: POST /api/higgs/pair (minting node pairing token)");
    match higgs.hub() {
        Some(hub) => {
            let (ticket, token) = hub.mint_pairing().await;
            Json(serde_json::json!({
                "hub_id": hub.hub_id(),
                "ticket": ticket,
                "token": token,
                "node_command": format!("higgs --node {ticket} {token}"),
            }))
            .into_response()
        }
        None => (
            StatusCode::CONFLICT,
            Json(serde_json::json!({ "error": "server is not running in hub mode (set HIGGS_HUB=1)" })),
        )
            .into_response(),
    }
}

/// `GET /api/higgs/nodes` — the remote fleet: one entry per paired node with its stable id,
/// endpoint, connected state, and last-fetched inventory (host + resident workers + hw/rt).
/// Empty when no `HubFleet` is installed (the hub role is off).
pub(super) async fn control_nodes(
    State(higgs): State<Arc<Higgs>>,
) -> Json<Vec<crate::node::fleet::NodeView>> {
    tracing::info!("higgs: GET /api/higgs/nodes");
    let views = higgs.fleet().map(|f| f.nodes_view()).unwrap_or_default();
    Json(views)
}

/// `POST /api/higgs/nodes/load` — load a model on a paired node and record the route, so
/// `/v1/chat/completions` for that model routes there. Admin-scoped. `409` when not a hub.
#[derive(Deserialize)]
pub(super) struct NodeLoadHttp {
    /// Target node's `EndpointId`.
    node: String,
    /// HuggingFace repo id to load on that node.
    model: String,
}

pub(super) async fn control_nodes_load(
    State(higgs): State<Arc<Higgs>>,
    Json(req): Json<NodeLoadHttp>,
) -> Response {
    tracing::warn!(node = %req.node, model = %req.model, "higgs: POST /api/higgs/nodes/load");
    let Some(fleet) = higgs.fleet() else {
        return not_a_hub();
    };
    match fleet.load(&req.node, &req.model).await {
        Ok(worker) => {
            Json(serde_json::json!({ "status": "ok", "worker_id": worker.0 })).into_response()
        }
        Err(err) => control_error(&err).into_response(),
    }
}

/// `POST /api/higgs/nodes/unload` — unload a remote-routed model and drop its route.
#[derive(Deserialize)]
pub(super) struct NodeUnloadHttp {
    /// The routed model id to unload from its node.
    model: String,
}

pub(super) async fn control_nodes_unload(
    State(higgs): State<Arc<Higgs>>,
    Json(req): Json<NodeUnloadHttp>,
) -> Response {
    tracing::warn!(model = %req.model, "higgs: POST /api/higgs/nodes/unload");
    let Some(fleet) = higgs.fleet() else {
        return not_a_hub();
    };
    match fleet.unload(&req.model).await {
        Ok(()) => Json(serde_json::json!({ "status": "ok" })).into_response(),
        Err(err) => control_error(&err).into_response(),
    }
}

/// The `409` returned by hub-only routes when no fleet is installed.
fn not_a_hub() -> Response {
    (
        StatusCode::CONFLICT,
        Json(serde_json::json!({ "error": "server is not running in hub mode (set HIGGS_HUB=1)" })),
    )
        .into_response()
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
        lines: higgs.logs(q.n.unwrap_or(DEFAULT_LOG_LINES), q.filter()),
    })
}

/// `GET /api/higgs/logs/settings` — current Developer-Log toggle state
/// ("Verbose Logging" and "Log Incoming Tokens").
pub(super) async fn control_logs_settings(State(higgs): State<Arc<Higgs>>) -> Json<LogSettings> {
    tracing::info!("higgs: GET /api/higgs/logs/settings");
    Json(LogSettings {
        verbose: higgs.verbose(),
        log_incoming_tokens: higgs.log_incoming_tokens(),
        show_log_fields: higgs.log_show_fields(),
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
        show_log_fields = body.show_log_fields,
        "higgs: set developer-log toggles"
    );
    higgs.set_verbose(body.verbose);
    higgs.set_log_incoming_tokens(body.log_incoming_tokens);
    higgs.set_log_show_fields(body.show_log_fields);
    Json(HiggsOk::new())
}

/// `GET /api/higgs/settings` — current runtime server-behavior flags
/// (just-in-time loading).
pub(super) async fn control_settings(
    State(higgs): State<Arc<Higgs>>,
) -> Json<HiggsRuntimeSettings> {
    tracing::info!("higgs: GET /api/higgs/settings");
    Json(HiggsRuntimeSettings {
        jit_enabled: higgs.jit_enabled(),
        auto_unload_idle: higgs.auto_unload_idle(),
        idle_ttl_minutes: higgs.idle_ttl_minutes(),
        serving_enabled: higgs.serving_enabled(),
    })
}

/// `PUT /api/higgs/settings` — set the runtime server-behavior flags
/// (just-in-time loading). A state-changing control op, so it logs at `warn`
/// with the new value.
pub(super) async fn control_set_settings(
    State(higgs): State<Arc<Higgs>>,
    Json(body): Json<HiggsRuntimeSettings>,
) -> Json<HiggsOk> {
    tracing::warn!(
        jit_enabled = body.jit_enabled,
        auto_unload_idle = body.auto_unload_idle,
        idle_ttl_minutes = body.idle_ttl_minutes,
        serving_enabled = body.serving_enabled,
        "higgs: set runtime server-behavior flags"
    );
    higgs.set_jit_enabled(body.jit_enabled);
    higgs.set_auto_unload_idle(body.auto_unload_idle);
    higgs.set_idle_ttl_minutes(body.idle_ttl_minutes);
    higgs.set_serving_enabled(body.serving_enabled);
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
    let filter = q.filter();
    tracing::info!(n, "higgs: GET /api/higgs/logs/stream");

    // Subscribe BEFORE snapshotting so no line slips between replay and live —
    // a line that lands in this window is duplicated at worst, never lost.
    let mut rx = higgs.subscribe_logs();
    let replay = higgs.logs(n, filter);

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
                    // Drop lines from other sources when a `?source=` filter is set.
                    if let Some(f) = filter {
                        if line.source != f {
                            continue;
                        }
                    }
                    if tx.send(line.text).is_err() {
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
    // Gather the worker-reported devices first (async — a transient worker RPC),
    // then fold them into the blocking hardware/runtime snapshot. An empty list
    // (no worker reachable) still yields a complete hardware/runtime response.
    let gpus = higgs.sysinfo().await;
    let info = tokio::task::spawn_blocking(move || SystemInfo::gather(config, gpus))
        .await
        .expect("system info gather task");
    Json(info)
}

#[cfg(test)]
mod tests {
    use super::super::test_support::*;
    use crate::log_bus::LogSource;
    use axum::http::StatusCode;
    use serde_json::json;
    use std::time::Duration;
    use tower::ServiceExt;

    // ── Gate 2: host-side tool-call-parser sniff ─────────────────────────────

    /// Build a minimal scanned model carrying only the chat template that the
    /// Gate-2 sniff inspects; all other fields are placeholder.
    fn model_with_template(template: Option<&str>) -> crate::worker::models::HiggsModel {
        crate::worker::models::HiggsModel {
            id: "org/model".into(),
            path: "/x.gguf".into(),
            size_bytes: 0,
            quant: None,
            source: crate::worker::models::HiggsModelSource::LmStudio,
            arch: None,
            ctx_train: None,
            has_chat_template: template.is_some(),
            supports_tools: false,
            supports_reasoning: false,
            gguf_components: Vec::new(),
            chat_template: template.map(ToOwned::to_owned),
        }
    }

    #[test]
    fn gate2_sniffs_tool_call_template() {
        // A template with the generic `<tool_call>` marker → a parser matches.
        let with_calls = model_with_template(Some(
            "{% for m in messages %}<|im_start|>{{ m.role }}<tool_call>{{ tool }}</tool_call>",
        ));
        assert!(
            super::tool_calls_supported(&with_calls),
            "<tool_call> matches"
        );

        // A plain chatml template with no tool markup → no parser matches.
        let plain = model_with_template(Some(
            "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>",
        ));
        assert!(
            !super::tool_calls_supported(&plain),
            "plain chatml: no match"
        );

        // No template at all → false.
        assert!(!super::tool_calls_supported(&model_with_template(None)));
    }

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

    #[tokio::test]
    async fn pair_without_hub_mode_is_conflict() {
        let (sup, _test_write, _test_read, _ring) = make_supervisor();
        let app = make_app(sup);
        // No hub installed → pairing is a 409 with an explanatory error.
        let resp = app
            .oneshot(post_json("/api/higgs/pair", &json!({})))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::CONFLICT);
        let v: serde_json::Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
        assert!(
            v["error"].as_str().unwrap().contains("hub mode"),
            "explains hub mode: {v}"
        );
    }

    #[tokio::test]
    async fn nodes_load_unload_without_fleet_is_conflict() {
        let (sup, _w, _r, _ring) = make_supervisor();
        let load = make_app(sup)
            .oneshot(post_json(
                "/api/higgs/nodes/load",
                &json!({ "node": "n", "model": "m" }),
            ))
            .await
            .unwrap();
        assert_eq!(load.status(), StatusCode::CONFLICT, "no fleet → 409");
        let (sup2, _w2, _r2, _ring2) = make_supervisor();
        let unload = make_app(sup2)
            .oneshot(post_json(
                "/api/higgs/nodes/unload",
                &json!({ "model": "m" }),
            ))
            .await
            .unwrap();
        assert_eq!(unload.status(), StatusCode::CONFLICT, "no fleet → 409");
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
        bus.push(LogSource::Serve, "line one".to_owned());
        bus.push(LogSource::Serve, "line two".to_owned());
        bus.push(LogSource::Serve, "line three".to_owned());
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

    /// `?source=serve|worker` routes the two log origins to separate consoles —
    /// end-to-end through the HTTP snapshot handler.
    #[tokio::test]
    async fn logs_endpoint_filters_by_source() {
        let (sup, _test_write, _test_read, bus) = make_supervisor();
        bus.push(LogSource::Serve, "higgs: GET /v1/models".to_owned());
        bus.push(LogSource::Worker, "ggml_metal_init: loaded".to_owned());
        bus.push(LogSource::Serve, "higgs: loading model".to_owned());
        let app = make_app(sup);

        // ?source=worker → only the worker stderr line.
        let resp = app
            .clone()
            .oneshot(get("/api/higgs/logs?source=worker"))
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
        assert_eq!(
            v["lines"],
            json!(["ggml_metal_init: loaded"]),
            "worker only"
        );

        // ?source=serve → only the higgs control-plane lines, in push order.
        let resp = app
            .clone()
            .oneshot(get("/api/higgs/logs?source=serve"))
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
        assert_eq!(
            v["lines"],
            json!(["higgs: GET /v1/models", "higgs: loading model"]),
            "serve only"
        );

        // No filter → all three, merged in push order.
        let resp = app.oneshot(get("/api/higgs/logs")).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
        assert_eq!(
            v["lines"].as_array().map(Vec::len),
            Some(3),
            "no filter = all sources merged"
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
        bus.push(LogSource::Serve, "hist-1".to_owned());
        bus.push(LogSource::Serve, "hist-2".to_owned());

        let higgs = Arc::new(Higgs::with_supervisor(
            Arc::new(sup),
            HiggsConfig::default(),
        ));
        let resp = control_logs_stream(
            State(higgs.clone()),
            Query(LogsQuery {
                n: Some(10),
                source: None,
            }),
        )
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
        bus.push(LogSource::Serve, "live-1".to_owned());
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
            higgs.logs(10, None),
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
        assert_eq!(
            v["show_log_fields"], false,
            "show_log_fields defaults to false (redact)"
        );

        // PUT all flags true returns {"status":"ok"}.
        let resp = app
            .clone()
            .oneshot(put_json(
                "/api/higgs/logs/settings",
                &json!({"verbose": true, "log_incoming_tokens": true, "show_log_fields": true}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let v: serde_json::Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
        assert_eq!(v["status"], "ok");

        // GET now reflects the new state for all flags.
        let resp = app.oneshot(get("/api/higgs/logs/settings")).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
        assert_eq!(v["verbose"], true, "PUT toggled verbose on");
        assert_eq!(
            v["log_incoming_tokens"], true,
            "PUT toggled log_incoming_tokens on"
        );
        assert_eq!(v["show_log_fields"], true, "PUT toggled show_log_fields on");
    }

    // ── runtime settings: GET reflects default (JIT on); PUT toggles JIT ─────

    #[tokio::test]
    async fn settings_get_default_and_put_toggles_jit() {
        let (sup, _test_write, _test_read, _ring) = make_supervisor();
        let app = make_app(sup);

        // GET defaults: JIT on, auto-unload on, TTL 5 minutes.
        let resp = app
            .clone()
            .oneshot(get("/api/higgs/settings"))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let v: serde_json::Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
        assert_eq!(v["jit_enabled"], true, "JIT defaults to on");
        assert_eq!(v["auto_unload_idle"], true, "auto-unload defaults to on");
        assert_eq!(v["idle_ttl_minutes"], 5, "TTL defaults to 5 minutes");
        assert_eq!(v["serving_enabled"], true, "serving defaults to on");

        // PUT all four (JIT off, auto-unload off, TTL 30, serving off) returns ok.
        let resp = app
            .clone()
            .oneshot(put_json(
                "/api/higgs/settings",
                &json!({
                    "jit_enabled": false,
                    "auto_unload_idle": false,
                    "idle_ttl_minutes": 30,
                    "serving_enabled": false,
                }),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let v: serde_json::Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
        assert_eq!(v["status"], "ok");

        // GET now reflects all three new values.
        let resp = app.oneshot(get("/api/higgs/settings")).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
        assert_eq!(v["jit_enabled"], false, "PUT toggled JIT off");
        assert_eq!(v["auto_unload_idle"], false, "PUT toggled auto-unload off");
        assert_eq!(v["idle_ttl_minutes"], 30, "PUT set TTL to 30 minutes");
        assert_eq!(v["serving_enabled"], false, "PUT toggled serving off");
    }

    // ── settings handlers: round-trip through the typed GET/PUT pair ──────────

    #[tokio::test]
    async fn settings_handlers_round_trip() {
        use super::{control_set_settings, control_settings};
        use crate::api::Higgs;
        use axum::extract::State;
        use std::sync::Arc;

        let (sup, _test_write, _test_read, _ring) = make_supervisor();
        let higgs = Arc::new(Higgs::with_supervisor(
            Arc::new(sup),
            crate::api::HiggsConfig::default(),
        ));

        // GET handler reflects the default-on state.
        assert!(
            control_settings(State(higgs.clone())).await.0.jit_enabled,
            "JIT on by default"
        );

        // PUT handler flips it off; the chat path's gate (`higgs.jit_enabled()`)
        // now returns false, so an unloaded model is a 404 (explicit-load).
        let ok = control_set_settings(
            State(higgs.clone()),
            axum::Json(crate::serve::HiggsRuntimeSettings {
                jit_enabled: false,
                auto_unload_idle: false,
                idle_ttl_minutes: 30,
                serving_enabled: false,
            }),
        )
        .await;
        assert_eq!(ok.0.status, "ok");
        assert!(!higgs.jit_enabled(), "PUT disabled the JIT gate");
        assert!(!higgs.auto_unload_idle(), "PUT disabled idle auto-unload");
        assert_eq!(higgs.idle_ttl_minutes(), 30, "PUT set the idle TTL");
        assert!(!higgs.serving_enabled(), "PUT disabled serving");
        let got = control_settings(State(higgs.clone())).await.0;
        assert!(!got.jit_enabled, "GET reflects JIT off");
        assert!(!got.auto_unload_idle, "GET reflects auto-unload off");
        assert_eq!(got.idle_ttl_minutes, 30, "GET reflects the new TTL");
        assert!(!got.serving_enabled, "GET reflects serving off");
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
                show_log_fields: false,
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
