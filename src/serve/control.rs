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
    HiggsErrorResponse, HiggsHubStatus, HiggsLoadRequest, HiggsLoadResponse, HiggsLogsResponse,
    HiggsModelEntry, HiggsModelsResponse, HiggsOk, HiggsRuntimeSettings, HiggsVersionResponse,
    LogSettings,
};
use crate::api::Higgs;
use crate::diagnostic::HiggsError;
use crate::log_bus::LogSource;
use crate::system::SystemInfo;
use crate::worker::engine::llamacpp::params::LlamaCppParams;
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

/// Host-side scan of all configured directories plus the set of currently-resident
/// model ids and the PRIMARY one. The local node is multi-model, so the per-model
/// "loaded" flag must reflect EVERY resident model (`loaded_set`), not just the
/// primary; the singular legacy `loaded_id` (primary, lowest worker) is kept for the
/// `HiggsModelsResponse.loaded_id` field. At the facade served ids == raw repo ids
/// (one worker per model), so `local_served_ids` IS the set of loaded raw ids. The
/// scan is pure Rust (no worker); `Err(response)` carries the mapped control error.
async fn scan_with_loaded(
    higgs: &Arc<Higgs>,
) -> Result<(Vec<HiggsModel>, Vec<String>, Option<String>), Response> {
    let models = higgs.scan().await.map_err(|err| {
        tracing::warn!(error = %err, "higgs: scan failed");
        control_error(&err).into_response()
    })?;
    let loaded_set = higgs.local_served_ids().await;
    let primary = higgs
        .status()
        .await
        .map(|s| s.loaded.map(|l| l.id))
        .map_err(|err| {
            tracing::warn!(error = %err, "higgs: status failed");
            control_error(&err).into_response()
        })?;
    Ok((models, loaded_set, primary))
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
/// its load `state`, `format`, the Gate-2 tool-call support verdict, and the
/// `last_load` params persisted on the last successful load (if any).
///
/// There is NO load-to-test probe at scan time — engine loadability is learned
/// only when the model is actually loaded (the load error is surfaced then).
/// `support_reason` carries the fixed Gate-2 message when no tool-call parser
/// matches the model's template, else `None`.
fn model_entry(
    mut model: HiggsModel,
    loaded_ids: &[String],
    last_load: Option<LoadParams>,
    readiness: crate::serve::readiness::ModelReadiness,
    fit: Option<crate::serve::wire::ModelFit>,
) -> HiggsModelEntry {
    // Multi-model: this model is "loaded" if it is among the resident ids, not only
    // when it is the primary.
    let is_loaded = loaded_ids.iter().any(|id| id == &model.id);
    let tool_calls = tool_calls_supported(&model);
    // Gate 2 (pure host-side template sniff): no parser matches the template.
    let support_reason =
        (!tool_calls).then(|| "no tool-call parser matches this model's template".to_owned());
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
        tool_calls,
        support_reason,
        last_load,
        readiness,
        fit,
        model,
    }
}

/// `GET /api/higgs/models` — live scan of all configured directories, plus the
/// currently loaded model id and the Gate-2 tool-call support verdict per model.
/// Pure host-side: the scan never loads a model (no probe), so it is fast even
/// across a large catalog — loadability is learned only at actual load time.
pub(super) async fn control_models(State(higgs): State<Arc<Higgs>>) -> Response {
    tracing::info!("higgs: GET /api/higgs/models");
    let (models, loaded_set, primary) = match scan_with_loaded(&higgs).await {
        Ok(triple) => triple,
        Err(resp) => return resp,
    };
    // One read of the per-model load records; each entry attaches its own (if any).
    // Non-consuming lookup — the scan can list multiple rows for one id (e.g. several
    // HF cache revisions), and they should all carry the same persisted `last_load`.
    let records = higgs.model_records();
    // Load the per-model config records AND tuning profiles ONCE for the whole
    // list; readiness then does in-memory lookups (no per-model store reopen). A
    // genuinely unreadable store surfaces HG040 rather than badging every prepared
    // model `discovered` (the JIT path surfaces the same fault as HG040).
    let tuning = match higgs.tuning_records() {
        Ok(t) => t,
        Err(err) => return control_error(&err).into_response(),
    };
    // One hardware snapshot for the whole list — readiness (staleness + fit) is
    // computed against it per model.
    let hw = higgs.hardware().await;
    let mut entries: Vec<HiggsModelEntry> = Vec::with_capacity(models.len());
    for m in models {
        let last_load = records.get(&m.id).and_then(|r| r.load.clone());
        let (readiness, fit) = higgs.model_readiness(&m, &loaded_set, &hw, &tuning);
        entries.push(model_entry(m, &loaded_set, last_load, readiness, fit));
    }
    Json(HiggsModelsResponse {
        models: entries,
        loaded_id: primary,
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
    let (models, loaded_set, _primary) = match scan_with_loaded(&higgs).await {
        Ok(triple) => triple,
        Err(resp) => return resp,
    };
    match models.into_iter().find(|m| m.id == id) {
        Some(model) => {
            let last_load = higgs.model_records().remove(&model.id).and_then(|r| r.load);
            let hw = higgs.hardware().await;
            let tuning = match higgs.tuning_records() {
                Ok(t) => t,
                Err(err) => return control_error(&err).into_response(),
            };
            let (readiness, fit) = higgs.model_readiness(&model, &loaded_set, &hw, &tuning);
            Json(model_entry(model, &loaded_set, last_load, readiness, fit)).into_response()
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
    let params = if let Some(full) = req.params.clone() {
        // The full engine-tagged params (an accepted [Tune params] suggestion, or
        // a complete manual edit) supersede the flat fields and are used as-is.
        Some(full)
    } else if !any_pinned {
        None
    } else {
        let base = higgs.default_load();
        let base = base.as_llamacpp();
        Some(LoadParams::llamacpp(LlamaCppParams {
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
            ..Default::default()
        }))
    };
    let started = std::time::Instant::now();
    match higgs.load(&req.id, params).await {
        Ok(()) => {
            // NOTE: `req.idle_ttl_minutes` is accepted for forward-compat but NOT yet
            // enforced — per-load idle-TTL enforcement is a deferred follow-up (the node
            // reaper applies one per-node TTL; see the `NodeLoadParams` deferral note).
            // It is deliberately NOT recorded/surfaced so the API never advertises an
            // override the reaper would silently ignore. The global TTL applies today.
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

/// `POST /api/higgs/models/tune` — compute suggested load + sampling params for a
/// model (the model id is in the BODY, matching `/models/load`; higgs ids are
/// slashed/colon'd so a path id is infeasible). FILLS the editable fields; it
/// never loads. Persists the suggestion as the saved profile so the next plain
/// load reuses it.
pub(super) async fn control_tune(
    State(higgs): State<Arc<Higgs>>,
    Json(req): Json<crate::tune::TuneRequest>,
) -> Response {
    tracing::info!(id = %req.id, "higgs: POST /api/higgs/models/tune");
    match higgs.tune(req).await {
        Ok(suggestion) => Json(suggestion).into_response(),
        Err(err) => {
            tracing::warn!(error = %err, "higgs: tune failed");
            control_error(&err).into_response()
        }
    }
}

/// `POST /api/higgs/models/estimate` — the VRAM/RAM footprint of a CANDIDATE load
/// (context / KV types / GPU offload in the BODY), reusing the suggester's own
/// estimators. Never loads; pure + cheap so the load-params UI can call it on every
/// edit to show the live "≈ X GiB VRAM · verdict". higgs owns the formula.
pub(super) async fn control_estimate(
    State(higgs): State<Arc<Higgs>>,
    Json(req): Json<crate::tune::EstimateRequest>,
) -> Response {
    match higgs.estimate(req).await {
        Ok(report) => Json(report).into_response(),
        Err(err) => {
            tracing::warn!(error = %err, "higgs: estimate failed");
            control_error(&err).into_response()
        }
    }
}

/// `POST /api/higgs/models/unload` — unload one model by `{id}`, or ALL when no id.
///
/// The body is read as raw bytes and inspected as a `serde_json::Value` (NOT via the
/// `Json` extractor or a typed struct) so that ONLY a truly absent id is the
/// destructive drain-all. A typed `Option<String>` is unusable here: serde collapses
/// both an omitted key AND an explicit `{"id":null}` to `None`, so a buggy per-card
/// request with a null id would silently drain every model. Explicit inspection keeps
/// "key present" distinct from "key absent":
/// - empty body OR `{}` (no `id` key) → drain ALL local models (back-compat; reading
///   raw bytes also accepts an empty body carrying `Content-Type: application/json`,
///   which the `Json` extractor would 400);
/// - `{"id":"org/model"}` → unload just that served instance;
/// - a PRESENT but `null`/empty/non-string id → `400` — a per-model request with no
///   usable id must NOT fall through to draining every model;
/// - an unknown field (`{"model":…}`), a non-object, or malformed JSON → `400`.
pub(super) async fn control_unload(
    State(higgs): State<Arc<Higgs>>,
    body: axum::body::Bytes,
) -> Response {
    fn bad(msg: impl std::fmt::Display) -> Response {
        (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": msg.to_string() })),
        )
            .into_response()
    }
    let id: Option<String> = if body.is_empty() {
        None
    } else {
        let value: serde_json::Value = match serde_json::from_slice(&body) {
            Ok(v) => v,
            Err(e) => return bad(format!("invalid unload body: {e}")),
        };
        let Some(obj) = value.as_object() else {
            return bad("unload body must be a JSON object");
        };
        if let Some(unknown) = obj.keys().find(|k| k.as_str() != "id") {
            return bad(format!("unknown field `{unknown}` in unload body"));
        }
        match obj.get("id") {
            None => None, // `{}` → no id key → drain all
            Some(serde_json::Value::String(s)) if !s.is_empty() => Some(s.clone()),
            Some(_) => {
                // null / "" / number / etc. — a per-model request with no usable id.
                return bad(
                    "unload `id` must be a non-empty model id string (omit it entirely to unload all models)",
                );
            }
        }
    };
    let result = match &id {
        Some(id) => {
            tracing::warn!(%id, "higgs: unloading model");
            higgs.unload_one(id).await
        }
        None => {
            tracing::warn!("higgs: unloading all models");
            higgs.unload().await
        }
    };
    match result {
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
    // Serialize against the kill switch (enable/disable): without this, a /pair concurrent with a
    // disable could clone the still-published hub and mint a ticket/token from a closing endpoint,
    // returning a 200 with an unusable command. Holding the lock means /pair runs either fully
    // before a disable (valid at mint time) or fully after (sees hub() None → 409).
    let _lifecycle = higgs.hub_lifecycle().lock().await;
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

/// Build the current hub-mode status. `enabled` = the hub network is up (accepting dials);
/// `node_count` is the fleet size, which persists across a disable (nodes then show disconnected
/// until the hub is re-enabled and they reconnect).
async fn hub_status(higgs: &Higgs) -> HiggsHubStatus {
    let node_count = match higgs.fleet() {
        Some(fleet) => u32::try_from(fleet.nodes_view().await.len()).unwrap_or(u32::MAX),
        None => 0,
    };
    match higgs.hub() {
        Some(hub) => HiggsHubStatus {
            enabled: true,
            hub_id: Some(hub.hub_id().to_string()),
            node_count,
        },
        None => HiggsHubStatus {
            enabled: false,
            hub_id: None,
            node_count,
        },
    }
}

/// `GET /api/higgs/hub` — hub-mode status: whether this server is a fleet hub, and if so its
/// stable id and how many nodes it has admitted. Lets the Fleet tab show an explicit
/// "hub mode off" state instead of inferring it from a `/pair` 409.
pub(super) async fn control_hub(State(higgs): State<Arc<Higgs>>) -> Json<HiggsHubStatus> {
    Json(hub_status(&higgs).await)
}

/// `POST /api/higgs/hub/enable` — turn the hub network ON (the kill switch). Binds the iroh
/// endpoint + spawns the accept loop against the EXISTING fleet (routes preserved), so a node
/// that was disconnected by a prior disable reconnects with its routes intact. Idempotent —
/// a no-op returning the current status when already enabled.
pub(super) async fn control_hub_enable(State(higgs): State<Arc<Higgs>>) -> Response {
    // Serialize the whole check→start→publish against any concurrent enable/disable: without
    // this, two enables could both pass the `is_none` check, bind two endpoints + accept loops,
    // and orphan the loser (which would keep accepting after a later disable).
    let _lifecycle = higgs.hub_lifecycle().lock().await;
    if higgs.hub().is_some() {
        return Json(hub_status(&higgs).await).into_response();
    }
    // start_hub bumps the fleet's admission generation for its fresh accept loop, so a fleet that
    // a prior disable invalidated admits reconnecting nodes again under the new generation.
    match crate::node::hub::start_hub(higgs.log_bus(), higgs.fleet()).await {
        Ok(hub) => {
            let hub = Arc::new(hub);
            higgs.set_fleet(hub.fleet.clone());
            higgs.set_hub(hub);
            tracing::warn!("higgs: hub ENABLED via /api/higgs/hub/enable");
            Json(hub_status(&higgs).await).into_response()
        }
        Err(e) => {
            let err = HiggsError::HubControlFailed {
                op: "hub enable".into(),
                detail: e.to_string(),
            };
            tracing::error!(error = %err, "higgs: hub enable failed");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": err.to_string() })),
            )
                .into_response()
        }
    }
}

/// `POST /api/higgs/hub/disable` — turn the hub network OFF (the kill switch). Closes the iroh
/// endpoint (ends the accept loop + drops all relay connections) and every node transport, but
/// KEEPS the fleet route table so re-enabling is a pure reconnect (no re-pair). Idempotent.
pub(super) async fn control_hub_disable(State(higgs): State<Arc<Higgs>>) -> Json<HiggsHubStatus> {
    // Serialized against enable (and other disables) so the close→disconnect→clear sequence is
    // atomic w.r.t. a concurrent (re-)enable — see `control_hub_enable`.
    let _lifecycle = higgs.hub_lifecycle().lock().await;
    if let Some(hub) = higgs.hub() {
        // Publish "disabled" FIRST (synchronous, before any await): a /pair that is also waiting
        // on the lifecycle lock will, once it runs, see hub() None → 409 (never mints from the
        // closing endpoint). Then tear down the network.
        higgs.clear_hub();
        hub.shutdown().await;
        if let Some(fleet) = higgs.fleet() {
            fleet.disconnect_all().await;
        }
        tracing::warn!("higgs: hub DISABLED via /api/higgs/hub/disable (network off; routes kept)");
    }
    Json(hub_status(&higgs).await)
}

/// Node labels keyed by `EndpointId`, from the live hub allowlist when the hub is enabled, else
/// read straight from the persisted `pairings.json`. The kill switch (`clear_hub`) drops the
/// `Hub` but KEEPS the fleet, so without this disk fallback every remote node would lose its
/// operator label for the whole disabled window — the disk read keeps labels stable across
/// enable/disable. A missing/unreadable file yields no labels (callers fall back to hostname).
async fn node_labels(higgs: &Higgs) -> std::collections::HashMap<String, Option<String>> {
    if let Some(hub) = higgs.hub() {
        return hub.labels().await;
    }
    let path = crate::home::higgs_home().join("pairings.json");
    crate::auth::Allowlist::load(&path)
        .map(|allow| allow.labels())
        .unwrap_or_default()
}

/// `GET /api/higgs/nodes` — every node in one shape: the LOCAL machine FIRST (`is_local`,
/// always present, even when the hub role is off), then each paired remote node with its stable
/// id, endpoint, connected state, label, and last-fetched inventory. Remote labels are merged
/// from the hub allowlist (the operator-editable source of truth) so the view reflects renames.
pub(super) async fn control_nodes(
    State(higgs): State<Arc<Higgs>>,
) -> Json<Vec<crate::node::fleet::NodeView>> {
    tracing::info!("higgs: GET /api/higgs/nodes");

    // The local node always appears first, labelled with this instance's config.json name.
    let local_label = higgs
        .instance_name()
        .unwrap_or_else(|| "this machine".to_string());
    let mut out = vec![higgs.local_node_view(local_label).await];

    // Then the remote fleet, with each node's label filled from the allowlist (falling back to
    // its reported hostname, else a short endpoint id) so the UI shows a human name. Each node
    // keeps the served ids IT reports — the Fleet view shows EVERY resident model on EVERY node,
    // so the same raw model loaded both locally and on a remote node legitimately appears on
    // both (chat resolves local-first, but the view's job is full visibility, not routing).
    if let Some(fleet) = higgs.fleet() {
        let labels = node_labels(&higgs).await;
        let mut remotes = fleet.nodes_view().await;
        for v in &mut remotes {
            v.label = labels
                .get(&v.endpoint_id)
                .cloned()
                .flatten()
                .or_else(|| {
                    v.inventory
                        .as_ref()
                        .map(|i| i.hostname.clone())
                        .filter(|h| !h.is_empty())
                })
                .unwrap_or_else(|| v.endpoint_id.chars().take(8).collect());
        }
        out.extend(remotes);
    }
    Json(out)
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

/// `GET /api/higgs/nodes/{node}/models` — a paired node's on-disk model catalog
/// (`M_NODE_SCAN`), for the remote Load picker + the fleet's not-loaded list. `409` when not
/// a hub; the node's `{ "models": [HiggsModel, …] }` reply is returned verbatim; HG027 when
/// the node is disconnected.
pub(super) async fn control_node_models(
    State(higgs): State<Arc<Higgs>>,
    Path(node): Path<String>,
) -> Response {
    tracing::info!(node = %node, "higgs: GET /api/higgs/nodes/{{node}}/models");
    let Some(fleet) = higgs.fleet() else {
        return not_a_hub();
    };
    match fleet.scan_node(&node).await {
        Ok(catalog) => Json(catalog).into_response(),
        Err(err) => control_error(&err).into_response(),
    }
}

/// `POST /api/higgs/nodes/retire` — retire a node for good: remove it from the persistent
/// allowlist (so it can't silently re-admit) AND drop it from the fleet. Admin-scoped. `409`
/// when not a hub.
#[derive(Deserialize)]
pub(super) struct NodeRetireHttp {
    /// The node's `EndpointId` to retire.
    node: String,
}

pub(super) async fn control_nodes_retire(
    State(higgs): State<Arc<Higgs>>,
    Json(req): Json<NodeRetireHttp>,
) -> Response {
    tracing::warn!(node = %req.node, "higgs: POST /api/higgs/nodes/retire");
    let Some(hub) = higgs.hub() else {
        return not_a_hub();
    };
    match hub.retire(&req.node).await {
        Ok(()) => Json(serde_json::json!({ "status": "ok" })).into_response(),
        Err(e) => {
            let err = HiggsError::HubControlFailed {
                op: "retire".into(),
                detail: e.to_string(),
            };
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": err.to_string() })),
            )
                .into_response()
        }
    }
}

/// `POST /api/higgs/nodes/label` — rename a node. `node:"local"` edits this instance's
/// `config.json` name; any other id renames that paired node's allowlist label (operator
/// rename — surfaced in `GET /api/higgs/nodes`). An empty label clears a remote node's label
/// (it then falls back to hostname/short id in the view). Admin-scoped.
#[derive(Deserialize)]
pub(super) struct NodeLabelHttp {
    /// `"local"` for this machine, else the node's `EndpointId`.
    node: String,
    /// The new label (empty clears a remote node's label).
    label: String,
}

pub(super) async fn control_nodes_label(
    State(higgs): State<Arc<Higgs>>,
    Json(req): Json<NodeLabelHttp>,
) -> Response {
    tracing::warn!(node = %req.node, label = %req.label, "higgs: POST /api/higgs/nodes/label");
    if req.node == "local" {
        // Rename the LOCAL instance under the shared config-io lock (so it can't clobber a
        // concurrent load-record write). The next `GET /api/higgs/nodes` shows the new name, and
        // the hub accept loop re-reads it per admission so paired nodes learn the new `hub_name`.
        return match higgs.with_config_mut(|c| c.name = req.label.clone()) {
            Ok(()) => Json(serde_json::json!({ "status": "ok" })).into_response(),
            // The rename PERSISTS to config.json; a write failure is HG040 (the
            // remediation — check disk/permissions — is the local operator's).
            Err(e) => {
                let err = HiggsError::PersistenceFailed {
                    store: "config".into(),
                    path: "config.json".into(),
                    source: e,
                };
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({ "error": err.to_string() })),
                )
                    .into_response()
            }
        };
    }
    // Renaming a REMOTE node mutates the hub-owned allowlist, so it requires the hub enabled
    // (409 otherwise, like the other node-mutation routes). Deliberately NO disabled-hub
    // disk-write fallback: `clear_hub` publishes `hub() == None` before the old accept tasks are
    // necessarily drained, so a direct `pairings.json` rewrite from a stale snapshot could clobber
    // a concurrent admit/retire. Labels stay VIEWABLE while disabled (`node_labels` reads the file
    // read-only) — just not editable until re-enabled.
    //
    // Serialize against the kill switch (enable/disable) with the SAME lock as `/pair`: otherwise
    // a relabel concurrent with a disable→enable could clone the still-published OLD hub and write
    // `pairings.json` after the NEW hub loaded its allowlist — making the relabel invisible or
    // later clobbered. Holding the lock means this runs fully before a disable (writes the live
    // allowlist) or fully after (sees `hub()` None → 409).
    let _lifecycle = higgs.hub_lifecycle().lock().await;
    let Some(hub) = higgs.hub() else {
        return not_a_hub();
    };
    let label = (!req.label.is_empty()).then(|| req.label.clone());
    match hub.set_label(&req.node, label).await {
        Ok(true) => Json(serde_json::json!({ "status": "ok" })).into_response(),
        // Unknown node KEEPS its 404 (the one HG043 site that isn't a 500), but now
        // carries the code so the body is classifiable like the rest.
        Ok(false) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": HiggsError::HubControlFailed {
                op: "label".into(),
                detail: format!("unknown node {}", req.node),
            }.to_string() })),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": HiggsError::HubControlFailed {
                op: "label".into(),
                detail: e.to_string(),
            }.to_string() })),
        )
            .into_response(),
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

/// `POST /api/higgs/worker/stop` — unload every resident worker, freeing their memory.
/// The server STAYS UP and usable: a subsequent load (or a JIT chat) spawns a fresh
/// worker. This is a NON-terminal bulk unload ([`Higgs::unload`]), deliberately NOT
/// [`Higgs::stop`] — `stop` runs the node's TERMINAL `shutdown_all` drain (it marks the
/// runtime shutting-down, so every later load is rejected until the process restarts),
/// which is reserved for actual server shutdown ([`crate::serve`] bring-down).
pub(super) async fn control_worker_stop(State(higgs): State<Arc<Higgs>>) -> Json<HiggsOk> {
    tracing::warn!("higgs: stopping (unloading) all workers");
    // `unload` is best-effort (it ignores a worker a concurrent idle-reap already took)
    // and always returns Ok; its post-condition is "no worker resident".
    let _ = higgs.unload().await;
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
pub(super) async fn control_system(State(higgs): State<Arc<Higgs>>) -> Response {
    tracing::info!("higgs: GET /api/higgs/system");
    // Snapshot the read-only server config (cheap lock, no I/O) on the async
    // thread, then move it into the blocking gather (which samples CPU load).
    let config = higgs.server_config();
    // Gather the worker-reported devices first (async — a transient worker RPC),
    // then fold them into the blocking hardware/runtime snapshot. An empty list
    // (no worker reachable) still yields a complete hardware/runtime response.
    let gpus = higgs.sysinfo().await;
    // A panic INSIDE the blocking gather surfaces here as a JoinError. Rather than
    // re-panic (`.expect`, aborting the request task), return a typed HG042 500 —
    // a recoverable internal fault the caller can see and report.
    match tokio::task::spawn_blocking(move || SystemInfo::gather(config, gpus)).await {
        Ok(info) => Json(info).into_response(),
        Err(e) => control_error(&HiggsError::InternalFault {
            context: "system info gather".into(),
            detail: e.to_string(),
        })
        .into_response(),
    }
}

#[cfg(test)]
#[path = "control_tests.rs"]
mod tests;
