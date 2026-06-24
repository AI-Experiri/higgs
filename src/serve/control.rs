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
/// its load `state`, `format`, and the Gate-2 tool-call support verdict.
///
/// There is NO load-to-test probe at scan time — engine loadability is learned
/// only when the model is actually loaded (the load error is surfaced then).
/// `support_reason` carries the fixed Gate-2 message when no tool-call parser
/// matches the model's template, else `None`.
fn model_entry(mut model: HiggsModel, loaded_ids: &[String]) -> HiggsModelEntry {
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
    let entries: Vec<HiggsModelEntry> = models
        .into_iter()
        .map(|m| model_entry(m, &loaded_set))
        .collect();
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
        Some(model) => Json(model_entry(model, &loaded_set)).into_response(),
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
            tracing::error!(error = %e, "higgs: hub enable failed");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": format!("hub enable failed: {e}") })),
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
    let local_label = crate::config::config_path()
        .ok()
        .and_then(|p| crate::config::InstanceConfig::load(&p).ok())
        .map(|c| c.name)
        .filter(|n| !n.is_empty())
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
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": format!("retire failed: {e}") })),
        )
            .into_response(),
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
        return match rename_local(&req.label) {
            Ok(()) => Json(serde_json::json!({ "status": "ok" })).into_response(),
            Err(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": format!("rename failed: {e}") })),
            )
                .into_response(),
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
        Ok(false) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": format!("unknown node {}", req.node) })),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": format!("rename failed: {e}") })),
        )
            .into_response(),
    }
}

/// Rename the LOCAL instance: set `config.json`'s `name`. The next `GET /api/higgs/nodes` shows
/// it, and the hub accept loop re-reads it per admission so paired nodes learn the new `hub_name`.
fn rename_local(label: &str) -> std::io::Result<()> {
    let path = crate::config::config_path()?;
    let mut cfg = crate::config::InstanceConfig::load(&path)?;
    cfg.name = label.to_string();
    cfg.save(&path)
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
        // `load` resolves the GGUF path host-side, so the id must be discoverable.
        // The stateful fake worker auto-responds to M_LOAD/M_STATUS/M_UNLOAD, so
        // the load → unload round-trip runs through the real node path.
        let dir = tempfile::TempDir::new().unwrap();
        write_gguf_fixture(dir.path(), "org/model");
        let app = make_app_with_lmstudio(dir.path().to_path_buf());

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
        // Scan runs host-side: discover the slashed id from a real GGUF fixture.
        // Nothing is loaded (no worker), so the status enrichment reports
        // `not-loaded` — the fake worker need not be driven.
        let dir = tempfile::TempDir::new().unwrap();
        write_gguf_fixture(
            dir.path(),
            "lmstudio-community/NVIDIA-Nemotron-3-Nano-4B-GGUF",
        );
        let app = make_app_with_lmstudio(dir.path().to_path_buf());

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
        // Empty temp dir → host-side scan finds nothing → the id is absent.
        let dir = tempfile::TempDir::new().unwrap();
        let app = make_app_with_lmstudio(dir.path().to_path_buf());

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
        let app = make_app();
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
        let load = make_app()
            .oneshot(post_json(
                "/api/higgs/nodes/load",
                &json!({ "node": "n", "model": "m" }),
            ))
            .await
            .unwrap();
        assert_eq!(load.status(), StatusCode::CONFLICT, "no fleet → 409");
        let unload = make_app()
            .oneshot(post_json(
                "/api/higgs/nodes/unload",
                &json!({ "model": "m" }),
            ))
            .await
            .unwrap();
        assert_eq!(unload.status(), StatusCode::CONFLICT, "no fleet → 409");
    }

    #[tokio::test]
    async fn relabel_remote_without_hub_is_conflict() {
        // Renaming a REMOTE node requires the hub enabled (it owns the allowlist) → 409 when off,
        // like the other node-mutation routes. (The local rename + remote-success + unknown-id 404
        // paths run in the hub_server e2e under a temp HIGGS_HOME, so this doesn't touch ~/.higgs.)
        let resp = make_app()
            .oneshot(post_json(
                "/api/higgs/nodes/label",
                &json!({ "node": "some-remote-endpoint-id", "label": "x" }),
            ))
            .await
            .unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::CONFLICT,
            "remote relabel needs a hub → 409"
        );
    }

    #[tokio::test]
    async fn nodes_lists_the_local_node_first_even_without_a_fleet() {
        // Even with the hub role off (no fleet), GET /api/higgs/nodes returns the LOCAL machine
        // as a first-class node, so the Fleet view always shows "this machine".
        let resp = make_app().oneshot(get("/api/higgs/nodes")).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let v: Vec<serde_json::Value> = serde_json::from_slice(&body_bytes(resp).await).unwrap();
        assert_eq!(v.len(), 1, "only the local node when no fleet: {v:?}");
        assert_eq!(v[0]["endpoint_id"], "local", "local sentinel id");
        assert_eq!(v[0]["is_local"], true, "flagged local");
        assert_eq!(v[0]["connected"], true, "local node is always connected");
        assert!(
            v[0]["label"].as_str().is_some_and(|s| !s.is_empty()),
            "local node has a label: {v:?}"
        );
        assert!(
            v[0]["inventory"].is_object(),
            "local inventory present: {v:?}"
        );
    }

    #[tokio::test]
    async fn hub_status_without_hub_reports_disabled() {
        // No hub installed → GET /api/higgs/hub answers 200 with enabled:false, no id, 0 nodes.
        let resp = make_app().oneshot(get("/api/higgs/hub")).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let v: serde_json::Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
        assert_eq!(v["enabled"], false, "no hub installed → disabled: {v}");
        assert_eq!(v["node_count"], 0, "no nodes when disabled: {v}");
        assert!(
            v.get("hub_id").is_none(),
            "hub_id omitted when disabled: {v}"
        );
    }

    #[tokio::test]
    async fn hub_disable_without_hub_is_a_noop() {
        // The kill switch is idempotent: disabling when no hub is installed just reports disabled.
        let resp = make_app()
            .oneshot(post_json("/api/higgs/hub/disable", &json!({})))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let v: serde_json::Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
        assert_eq!(v["enabled"], false, "still disabled: {v}");
    }

    // ── Test 9: version endpoint ──────────────────────────────────────────────

    #[tokio::test]
    async fn version_endpoint() {
        let app = make_app();

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
        let (higgs, bus) = make_higgs_with_bus();
        bus.push(LogSource::Serve, "line one".to_owned());
        bus.push(LogSource::Serve, "line two".to_owned());
        bus.push(LogSource::Serve, "line three".to_owned());
        let app = app_for(higgs);

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
        let (higgs, bus) = make_higgs_with_bus();
        bus.push(LogSource::Serve, "higgs: GET /v1/models".to_owned());
        bus.push(LogSource::Worker, "ggml_metal_init: loaded".to_owned());
        bus.push(LogSource::Serve, "higgs: loading model".to_owned());
        let app = app_for(higgs);

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
        use axum::extract::{Query, State};
        use axum::response::IntoResponse;
        use futures::StreamExt;
        use std::time::Duration;

        let (higgs, bus) = make_higgs_with_bus();
        // Seed history BEFORE the request — this is the replay prefix.
        bus.push(LogSource::Serve, "hist-1".to_owned());
        bus.push(LogSource::Serve, "hist-2".to_owned());

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
        // Multi-model: load TWO distinct models — the models list must flag BOTH as
        // `loaded` (not just the primary), and `loaded_id` reports the primary.
        let dir = tempfile::TempDir::new().unwrap();
        write_gguf_fixture(dir.path(), "org/model");
        write_gguf_fixture(dir.path(), "org/other");
        let higgs = make_higgs_with_lmstudio(dir.path().to_path_buf());
        higgs.load("org/model", None).await.expect("load model");
        higgs.load("org/other", None).await.expect("load other");
        let app = app_for(higgs);

        let resp = app.oneshot(get("/api/higgs/models")).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let v: serde_json::Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
        // Primary (lowest worker = first loaded) is org/model.
        assert_eq!(v["loaded_id"], "org/model");
        let models = v["models"].as_array().expect("models array");
        assert_eq!(models.len(), 2, "both models listed");
        // BOTH resident models are flagged loaded — the multi-model fix.
        for m in models {
            assert_eq!(
                m["state"], "loaded",
                "every resident model is flagged loaded: {m}"
            );
            assert_eq!(m["format"], "gguf");
        }
    }

    // ── control_status passthrough ───────────────────────────────────────────

    #[tokio::test]
    async fn control_status_returns_snapshot() {
        // Load a fixture model so the status snapshot reports it resident (the
        // stateful fake worker echoes the loaded model in M_STATUS).
        let dir = tempfile::TempDir::new().unwrap();
        write_gguf_fixture(dir.path(), "org/model");
        let higgs = make_higgs_with_lmstudio(dir.path().to_path_buf());
        higgs.load("org/model", None).await.expect("load");
        let app = app_for(higgs);

        let resp = app.oneshot(get("/api/higgs/status")).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let v: serde_json::Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
        assert_eq!(v["loaded"]["id"], "org/model");
    }

    // ── control_load with explicit params (non-default branch) ───────────────

    #[tokio::test]
    async fn control_load_with_explicit_params() {
        // `load` resolves the GGUF path host-side, so the id must be discoverable.
        let dir = tempfile::TempDir::new().unwrap();
        write_gguf_fixture(dir.path(), "org/model");
        let app = make_app_with_lmstudio(dir.path().to_path_buf());

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
        let app = make_app();

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
        let app = make_app();

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
        let app = make_app();

        // GET defaults: JIT on, auto-unload on, TTL 60 minutes (the node default).
        let resp = app
            .clone()
            .oneshot(get("/api/higgs/settings"))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let v: serde_json::Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
        assert_eq!(v["jit_enabled"], true, "JIT defaults to on");
        assert_eq!(v["auto_unload_idle"], true, "auto-unload defaults to on");
        assert_eq!(v["idle_ttl_minutes"], 60, "TTL defaults to 60 minutes");
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
        use axum::extract::State;

        let higgs = make_higgs();

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
        use axum::extract::State;

        let higgs = make_higgs();

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
        let app = make_app();

        let resp = app
            .oneshot(post_json("/api/higgs/worker/stop", &json!({})))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let v: serde_json::Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
        assert_eq!(v["status"], "ok");
    }

    // ── nodes/{node}/models + nodes/retire: hub-only routes ──────────────────

    /// Both new fleet routes require hub mode: with no fleet/hub installed they
    /// answer `409 CONFLICT` (the `not_a_hub` guard), exercising the handler
    /// wiring + route registration without a live iroh hub.
    #[tokio::test]
    async fn node_models_and_retire_require_hub_mode() {
        let app = make_app();

        let resp = app
            .clone()
            .oneshot(get("/api/higgs/nodes/somenode/models"))
            .await
            .unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::CONFLICT,
            "node catalog needs a hub"
        );

        let resp = app
            .oneshot(post_json(
                "/api/higgs/nodes/retire",
                &json!({ "node": "somenode" }),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::CONFLICT, "retire needs a hub");
    }
}
