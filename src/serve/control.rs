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
    HiggsErrorResponse, HiggsHubStatus, HiggsKeyEntry, HiggsKeyRemoved, HiggsKeysList,
    HiggsLoadRequest, HiggsLoadResponse, HiggsLogsResponse, HiggsMintKeyRequest,
    HiggsMintKeyResponse, HiggsModelEntry, HiggsModelsResponse, HiggsOk, HiggsRuntimeSettings,
    HiggsVersionResponse, LogSettings,
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

/// Gate 2 (host-side, zero FFI): does this model's chat template declare tool
/// handling? llama.cpp's auto-parser derives the actual parser from the
/// template at load time, so the scan-time signal is the template mentioning
/// tools/functions (same heuristic as the scan's `supports_tools`); `false`
/// when there is no template (the legacy route renders no tool grammar).
fn tool_calls_supported(model: &HiggsModel) -> bool {
    match model.chat_template.as_deref() {
        Some(tmpl) => {
            let tl = tmpl.to_lowercase();
            tl.contains("tool") || tl.contains("function")
        }
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
    tune: TuneProfileViews<'_>,
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
        tuned_load: tune.analytical.map(|t| t.profile.clone()),
        benched_load: tune.bench.map(|t| t.profile.clone()),
        tune_provenance: tune.active.map(|t| t.provenance),
        bench_tps: tune.bench.and_then(|t| t.bench_tps),
        model,
    }
}

/// The per-model tune views the entry serves: the ACTIVE record (JIT default,
/// its provenance labels the default set) plus the analytical/bench history
/// slots. `from_triple` grandfathers a pre-dual-slot store — one whose BOTH
/// history slots are empty (an old `models.json` predating the slots) — by
/// serving its lone active record under its own provenance's label. Once
/// EITHER slot is populated the store is post-dual (`put_tuning` fills a slot
/// on every real tune/turbotune), so the active record is NOT borrowed: a later
/// bare load can demote it to a `Heuristic` that was never an analytical tune,
/// and borrowing that would fabricate a "Tuned" set. (Residual A: on a store
/// whose only write was a bare load, that lone record is indistinguishable from
/// a pre-dual analytical tune and is served as Tuned — a real tune supersedes it.
/// Residual B: on a pre-dual store whose FIRST post-upgrade action is a benchmark,
/// the bench slot populates and the gate turns off, so the prior analytical tune
/// stops being grandfathered as "Tuned" until the user re-runs Tune — accepted
/// because that analytical set is deterministic and re-derivable in one click,
/// whereas `put_tuning` backfilling the active record to keep it would reintroduce
/// the bare-load masquerade this gate exists to prevent.)
#[derive(Clone, Copy, Default)]
struct TuneProfileViews<'a> {
    active: Option<&'a crate::tune::store::TuneRecord>,
    analytical: Option<&'a crate::tune::store::TuneRecord>,
    bench: Option<&'a crate::tune::store::TuneRecord>,
}

type TuneTriple = (
    Option<crate::tune::store::TuneRecord>,
    Option<crate::tune::store::TuneRecord>,
    Option<crate::tune::store::TuneRecord>,
);

/// The ACTIVE ("latest") tuning record per model, extracted from the
/// `(active, analytical, bench)` triples the models list ALSO reads for its
/// dual-profile wire fields. Readiness reads THIS map, so readiness and the
/// `tuned_load`/`benched_load` fields both come from ONE `models.json` snapshot —
/// a concurrent tune between two separate store reads can no longer make a row's
/// readiness disagree with its profile fields.
fn active_records(
    profiles: &std::collections::BTreeMap<String, TuneTriple>,
) -> std::collections::BTreeMap<String, crate::tune::store::TuneRecord> {
    profiles
        .iter()
        .filter_map(|(id, (active, _, _))| active.clone().map(|a| (id.clone(), a)))
        .collect()
}

impl<'a> TuneProfileViews<'a> {
    fn from_triple(triple: Option<&'a TuneTriple>) -> Self {
        let Some((active, analytical, bench)) = triple else {
            return Self::default();
        };
        let active = active.as_ref();
        let mut analytical = analytical.as_ref();
        let mut bench = bench.as_ref();
        // Grandfather ONLY a true pre-dual store (both history slots empty). A
        // post-dual store always has a slot filled by `put_tuning`, so borrowing
        // the active record there would surface a bare-load `Heuristic` — a manual
        // reload that `set_profile` demoted after a turbotune — as the analytical
        // "Tuned" set it never was.
        if analytical.is_none() && bench.is_none() {
            if let Some(a) = active {
                if a.provenance == crate::tune::TuneProvenance::Bench {
                    bench = Some(a);
                } else {
                    analytical = Some(a);
                }
            }
        }
        Self {
            active,
            analytical,
            bench,
        }
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
    // Load the per-model dual-profile views (active/analytical/bench) ONCE for the
    // whole list; readiness then does in-memory lookups (no per-model store reopen).
    // A genuinely unreadable store surfaces HG040 rather than badging every prepared
    // model `discovered` (the JIT path surfaces the same fault as HG040).
    let profiles = match higgs.tuning_profiles() {
        Ok(p) => p,
        Err(err) => return control_error(&err).into_response(),
    };
    // The ACTIVE records readiness reads are EXTRACTED from those same triples —
    // NOT a second `models.json` read. A separate `tuning_records()` pass could
    // observe a different snapshot if a concurrent tune landed between the two
    // reads, making a row's readiness disagree with its `tuned_load`/`benched_load`
    // wire fields; deriving both from one snapshot removes that TOCTOU.
    let tuning = active_records(&profiles);
    // One hardware snapshot for the whole list — readiness (staleness + fit) is
    // computed against it per model.
    let hw = higgs.hardware().await;
    let mut entries: Vec<HiggsModelEntry> = Vec::with_capacity(models.len());
    for m in models {
        let last_load = records.get(&m.id).and_then(|r| r.load.clone());
        let (readiness, fit) = higgs.model_readiness(&m, &loaded_set, &hw, &tuning);
        let tune = TuneProfileViews::from_triple(profiles.get(&m.id));
        entries.push(model_entry(m, &loaded_set, last_load, readiness, fit, tune));
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
            // One store snapshot: readiness's ACTIVE records and the dual-profile
            // wire fields are both derived from `profiles` (see `control_models`).
            let profiles = match higgs.tuning_profiles() {
                Ok(p) => p,
                Err(err) => return control_error(&err).into_response(),
            };
            let tuning = active_records(&profiles);
            let (readiness, fit) = higgs.model_readiness(&model, &loaded_set, &hw, &tuning);
            let tune = TuneProfileViews::from_triple(profiles.get(&model.id));
            Json(model_entry(
                model,
                &loaded_set,
                last_load,
                readiness,
                fit,
                tune,
            ))
            .into_response()
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
                    // Drop lines from other sources when a `?source=` filter is set
                    // (`worker` is a union filter matching every local worker's lines).
                    if let Some(f) = filter {
                        if !line.source.matches_filter(f) {
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

/// `GET /api/higgs/events` — LIVE model-load lifecycle events over SSE.
///
/// Each [`ModelLoadEvent`](crate::api::types::ModelLoadEvent) (a phase transition of
/// an in-flight load) arrives as one SSE `data:` frame carrying the event JSON. The
/// UI subscribes once and drives its loading indicator from these PUSH events —
/// showing the bar on the first non-terminal phase, updating the label per phase, and
/// hiding it on the terminal `ready`/`failed` — so NO `status` polling is needed to
/// watch a load. There is no replay ring (unlike the log stream): the bar is
/// transient, and a subscriber that joins mid-load still gets every remaining phase
/// plus the terminal event. A slow client that overflows the buffer gets `Lagged`;
/// the handler skips the gap and keeps streaming (the terminal event still clears the
/// bar). Stream ends on client disconnect or when the sender is permanently closed.
pub(super) async fn control_events_stream(
    State(higgs): State<Arc<Higgs>>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    tracing::info!("higgs: GET /api/higgs/events");
    // Unfold DIRECTLY over the broadcast receiver — no intermediate forwarding
    // task. axum polls this stream from the connection, so when the client
    // disconnects the stream is dropped and the receiver with it: no idle task or
    // subscriber lingers during quiet periods (the load-event channel is silent
    // between loads). KeepAlive comments detect a dead connection without traffic.
    let rx = higgs.subscribe_load_events();
    let stream = futures::stream::unfold(rx, |mut rx| async move {
        loop {
            match rx.recv().await {
                // Serialize the typed event to JSON for the SSE `data:` frame. A
                // serialization error is impossible for this plain struct; skip that
                // frame rather than crash if it ever occurred.
                Ok(ev) => match serde_json::to_string(&ev) {
                    Ok(json) => {
                        return Some((Ok::<_, Infallible>(Event::default().data(json)), rx))
                    }
                    Err(_) => continue,
                },
                Err(broadcast::error::RecvError::Lagged(skipped)) => {
                    // RESIDUAL (bounded): a skipped event is not resent. The channel
                    // caps at 256 and a load emits ~5 events, so lag would need ~50
                    // loads interleaved faster than the connection drains — not
                    // reachable in practice; and a dropped mid-phase only coarsens the
                    // bar. Skip the gap and keep streaming.
                    tracing::warn!(skipped, "higgs: load-event stream subscriber lagged");
                    continue;
                }
                // Sender dropped (facade gone) — end the stream cleanly.
                Err(broadcast::error::RecvError::Closed) => return None,
            }
        }
    });
    Sse::new(stream).keep_alive(KeepAlive::default())
}

/// `POST /api/higgs/worker/stop` — unload every resident worker, freeing their memory.
/// The server STAYS UP and usable: a subsequent load (or a JIT chat) spawns a fresh
/// worker. This is a NON-terminal bulk unload ([`Higgs::unload`]), deliberately NOT
/// [`Higgs::stop`] — `stop` runs the node's TERMINAL `shutdown_all` drain (it marks the
/// runtime shutting-down, so every later load is rejected until the process restarts),
/// which is reserved for actual server shutdown ([`crate::serve`] bring-down).
pub(super) async fn control_worker_stop(State(higgs): State<Arc<Higgs>>) -> Response {
    tracing::warn!("higgs: stopping (unloading) all workers");
    // `unload` ignores a worker a concurrent idle-reap already took, but it is NOT
    // infallible: it refuses ([HG068] → 503) while a benchmark owns a model, since a
    // drain would evict the bench candidate mid-measurement. Surface that refusal
    // instead of reporting a false success — same as `control_unload`.
    match higgs.unload().await {
        Ok(()) => Json(HiggsOk::new()).into_response(),
        Err(err) => {
            tracing::warn!(error = %err, "higgs: worker stop refused");
            control_error(&err).into_response()
        }
    }
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

// ── API-key management (G4): mint / list / revoke over the control surface ──
//
// All three routes carry `Admin` scope via `required_scope`'s fail-closed
// default for `/api/higgs/*`. Mutations go through `Higgs::mutate_api_keys`
// (file read-modify-write under a lock + in-memory HOT-SWAP), so a minted or
// revoked key gates the very next request — no restart, unlike the CLI.

/// `GET /api/higgs/keys` — list configured keys (labels/scopes/digest prefix).
/// Never returns plaintext tokens; those exist only in the mint response.
pub(super) async fn control_keys_list(State(higgs): State<Arc<Higgs>>) -> Json<HiggsKeysList> {
    let keys = higgs.api_keys();
    Json(HiggsKeysList {
        auth_enabled: !keys.is_empty(),
        keys: keys
            .iter()
            .map(|k| HiggsKeyEntry {
                label: k.label.clone(),
                scopes: k.scopes.clone(),
                sha256_prefix: k.sha256_prefix().to_owned(),
                created_at_ms: k.created_at_ms,
                last_used_ms: k.last_used_ms,
            })
            .collect(),
    })
}

/// `POST /api/higgs/keys` — mint a key: random `hgk_` token, digest persisted,
/// PLAINTEXT returned this once (never logged, never stored). Minting the
/// FIRST key flips auth ON for every subsequent request — the caller must
/// store the token before doing anything else, or it locks itself out.
/// The outcome of a mint decision, computed against the LOCKED keystore.
enum Mint {
    /// Mint with these effective scopes.
    Ok(Vec<crate::keys::Scope>),
    /// A key with this label already exists.
    Duplicate,
    /// Non-bootstrap mint with no valid Admin bearer for the current store.
    Unauthorized,
    /// A BOOTSTRAP mint (first key) whose EXPLICIT scopes omit `admin`: it would
    /// flip auth on yet be unable to reach the Admin-scoped key-management API,
    /// with no HTTP path left to recover ([codex r10]).
    BootstrapNeedsAdmin,
}

/// Decide a mint against the CURRENT (locked) keystore `ks`. Pure — derives
/// `bootstrap` from `ks` itself, so the empty-store window can't be raced: a
/// second unauthenticated mint that reaches the lock after the first key
/// landed sees a non-empty store, fails the bearer recheck, and is refused.
/// The bootstrap mint (empty store) is allowed unauthenticated and MUST grant
/// `Admin` — it defaults to `[admin]` when scopes are omitted, and REJECTS
/// explicit scopes that omit `admin` ([`Mint::BootstrapNeedsAdmin`]), since a
/// non-admin first key would flip auth on and lock the HTTP management surface
/// out of itself with no recovery path. Later omitted-scopes mints default to
/// `[chat, models]`. Explicit `requested` scopes otherwise win.
fn decide_mint(
    ks: &crate::keys::ApiKeys,
    bearer: Option<&str>,
    requested: Option<Vec<crate::keys::Scope>>,
    label: &str,
) -> Mint {
    use crate::keys::Scope;
    let bootstrap = ks.is_empty();
    if !bootstrap && !bearer.is_some_and(|t| ks.authorizes(t, Scope::Admin)) {
        return Mint::Unauthorized;
    }
    if ks.iter().any(|k| k.label == label) {
        return Mint::Duplicate;
    }
    // The FIRST key must be able to manage keys — reject explicit non-admin
    // bootstrap scopes rather than mint a self-locking key.
    if bootstrap {
        if let Some(scopes) = &requested {
            if !scopes.contains(&Scope::Admin) {
                return Mint::BootstrapNeedsAdmin;
            }
        }
    }
    let scopes = requested.unwrap_or_else(|| {
        if bootstrap {
            vec![Scope::Admin]
        } else {
            vec![Scope::Chat, Scope::Models]
        }
    });
    Mint::Ok(scopes)
}

pub(super) async fn control_keys_mint(
    State(higgs): State<Arc<Higgs>>,
    headers: axum::http::HeaderMap,
    Json(body): Json<HiggsMintKeyRequest>,
) -> Response {
    let label = body.label.trim().to_owned();
    // Labels are addressed by `DELETE /api/higgs/keys/{label}` — a SINGLE path
    // segment — so anything that can't round-trip through that route (slashes,
    // whitespace, control chars) must be rejected at mint or the key becomes
    // unrevokable via the API. Same charset the CLI examples use.
    // `.` / `..` pass the charset but URL parsers normalize dot-segments away
    // before Axum captures the label, so such a key could be minted yet never
    // revoked through `DELETE /api/higgs/keys/{label}` — reject them at mint.
    let label_ok = !label.is_empty()
        && label.len() <= 64
        && label != "."
        && label != ".."
        && label
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'));
    if !label_ok {
        return control_error(&HiggsError::InvalidRequest {
            detail: format!(
                "invalid key label {label:?}: 1-64 chars from [A-Za-z0-9._-] (it must fit in DELETE /api/higgs/keys/{{label}})"
            ),
        })
        .into_response();
    }
    // An EXPLICIT empty scope list is a client error regardless of store state.
    if body.scopes.as_ref().is_some_and(Vec::is_empty) {
        return control_error(&HiggsError::InvalidRequest {
            detail: "at least one scope required (chat, models, admin)".into(),
        })
        .into_response();
    }
    let token = crate::keys::mint_token(rand::random());
    let requested = body.scopes.clone();
    // The bearer is re-checked against the LOCKED live store below, not trusted
    // from auth_guard's earlier (possibly pre-bootstrap) pass.
    let bearer = super::bearer_token(&headers);

    // The whole decision + mutation happens INSIDE the keystore lock, closing
    // the bootstrap RACE: two unauthenticated mints can both pass `auth_guard`
    // while the store is empty, but `decide_mint` derives `bootstrap` from the
    // LOCKED store, so only the one that finds it still empty may bootstrap; the
    // loser re-checks its bearer against the now-non-empty store and is refused.
    let outcome = match higgs.mutate_api_keys(|ks| {
        let decision = decide_mint(ks, bearer.as_deref(), requested.clone(), &label);
        if let Mint::Ok(scopes) = &decision {
            ks.add(&token, label.clone(), scopes.clone());
        }
        decision
    }) {
        Ok(o) => o,
        Err(e) => return control_error(&keystore_io_error(e)).into_response(),
    };
    let scopes = match outcome {
        Mint::Ok(scopes) => scopes,
        Mint::Duplicate => {
            return control_error(&HiggsError::InvalidRequest {
                detail: format!("a key labeled {label:?} already exists — revoke it first"),
            })
            .into_response();
        }
        Mint::Unauthorized => return super::unauthorized(),
        Mint::BootstrapNeedsAdmin => {
            return control_error(&HiggsError::InvalidRequest {
                detail: "the first API key must include the `admin` scope (it is the only key able to manage keys) — pass scopes: [\"admin\"], or omit scopes to default to admin".into(),
            })
            .into_response();
        }
    };
    // Deliberately NOT logging the token — the label + scopes are the audit line.
    tracing::warn!(%label, ?scopes, "higgs: API key minted; auth now gates the surface");
    Json(HiggsMintKeyResponse {
        label,
        scopes,
        token,
    })
    .into_response()
}

/// The outcome of a revoke decision, computed against the LOCKED keystore.
enum Revoke {
    /// Remove keys with the label; carries the count that will be removed.
    Removed(usize),
    /// The store became non-empty (a bootstrap mint won a race) and the request
    /// carries no Admin bearer — refuse (codex r7, mirrors the mint recheck).
    Unauthorized,
    /// Would empty the keystore while LAN-exposed ([HG059]).
    LastKeyOnLan,
    /// Would remove the LAST Admin-capable key while OTHER (non-Admin) keys remain
    /// — auth stays on but the Admin-only management surface becomes unreachable, a
    /// lockout ([HG066]). Emptying the store entirely (turning auth off) is allowed;
    /// stranding non-Admin keys is not.
    LastAdminKey,
}

/// Decide a revoke against the CURRENT (locked) keystore `ks`. Pure. A non-empty
/// store REQUIRES a live Admin bearer: `auth_guard` may have admitted this DELETE
/// while the store was still empty (auth off), but a concurrent bootstrap mint
/// can commit before the lock is taken — so authorization is re-derived from the
/// locked store, not trusted from the earlier pass. `lan_exposed` gates the
/// last-key [HG059] refusal.
fn decide_revoke(
    ks: &crate::keys::ApiKeys,
    bearer: Option<&str>,
    label: &str,
    lan_exposed: bool,
) -> Revoke {
    if !ks.is_empty() && !bearer.is_some_and(|t| ks.authorizes(t, crate::keys::Scope::Admin)) {
        return Revoke::Unauthorized;
    }
    let total = ks.iter().count();
    let matching = ks.iter().filter(|k| k.label == label).count();
    if matching > 0 && lan_exposed && matching == total {
        return Revoke::LastKeyOnLan;
    }
    // Revoking SOME (not all) keys must leave an Admin-capable key behind, else the
    // Admin-only key-management surface is locked out while auth stays enabled.
    // Emptying the store entirely (matching == total → auth OFF) is a separate,
    // allowed operation, gated only by the LAN check above.
    if matching > 0 && matching < total {
        let admin_remains = ks
            .iter()
            .filter(|k| k.label != label)
            .any(|k| k.scopes.contains(&crate::keys::Scope::Admin));
        if !admin_remains {
            return Revoke::LastAdminKey;
        }
    }
    Revoke::Removed(matching)
}

/// `DELETE /api/higgs/keys/{label}` — revoke every key with `label`; takes
/// effect on the next request. Revoking the last key turns auth OFF (loopback
/// only — a LAN bind refuses, [HG059]).
pub(super) async fn control_keys_revoke(
    State(higgs): State<Arc<Higgs>>,
    Path(label): Path<String>,
    headers: axum::http::HeaderMap,
) -> Response {
    // Re-checked against the LOCKED live store below, NOT trusted from
    // auth_guard's earlier (possibly pre-bootstrap) pass.
    let bearer = super::bearer_token(&headers);

    // Auth recheck + last-key check + removal all INSIDE the keystore lock —
    // atomic with the store state, no TOCTOU with a racing bootstrap mint. The
    // decision is the pure `decide_revoke` (unit-tested with a fail-on-revert
    // seam); the closure only applies its verdict.
    let lan = higgs.lan_exposed();
    let outcome = match higgs.mutate_api_keys(|ks| {
        let decision = decide_revoke(ks, bearer.as_deref(), &label, lan);
        // Match by REFERENCE — `decision` is returned below unmoved.
        if let Revoke::Removed(_) = &decision {
            let _ = ks.remove_label(&label);
        }
        decision
    }) {
        Ok(o) => o,
        Err(e) => return control_error(&keystore_io_error(e)).into_response(),
    };
    let removed = match outcome {
        Revoke::Removed(n) => n,
        Revoke::Unauthorized => return super::unauthorized(),
        Revoke::LastKeyOnLan => {
            return control_error(&HiggsError::LastKeyOnLan { label }).into_response();
        }
        Revoke::LastAdminKey => {
            return control_error(&HiggsError::LastAdminKey { label }).into_response();
        }
    };
    if removed == 0 {
        return control_error(&HiggsError::InvalidRequest {
            detail: format!("no key labeled {label:?}"),
        })
        .into_response();
    }
    let auth_enabled = !higgs.api_keys().is_empty();
    tracing::warn!(%label, removed, auth_enabled, "higgs: API key(s) revoked");
    Json(HiggsKeyRemoved {
        removed: removed as u64,
        auth_enabled,
    })
    .into_response()
}

/// Map a keystore file I/O failure onto the coded store error ([HG040]).
fn keystore_io_error(e: std::io::Error) -> HiggsError {
    HiggsError::PersistenceFailed {
        store: "api_keys".into(),
        path: crate::keys::keys_path()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|_| "api_keys.json".into()),
        source: e,
    }
}

#[cfg(test)]
#[path = "control_tests.rs"]
mod tests;
