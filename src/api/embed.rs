//! The control-plane behavior on the [`Higgs`] facade as typed, in-process
//! methods so an embedder needs no HTTP — higgs's ONLY control surface now that
//! the `/api/higgs/*` HTTP handlers are gone (only `/v1` is served, by `serve::v1`).
//!
//! Each method returns `Result<_, HiggsError>` (never an HTTP `Response`); the
//! embedder maps the error however it serves. The pure sub-primitives these
//! methods share (row formatting, the mint/revoke decision, the hub-status
//! snapshot) live in `serve::control` and are reached by their `pub(crate)` path.

use std::collections::HashSet;

use serde_json::Value;

use crate::diagnostic::HiggsError;
use crate::keys::Scope;
use crate::node::worker_id::WorkerId;
use crate::serve::wire::{
    HiggsCorsSettings, HiggsHubStatus, HiggsKeyRemoved, HiggsLoadRequest, HiggsMintKeyResponse,
    HiggsModelEntry, HiggsVersionResponse, LogSettings,
};
use crate::worker::engine::llamacpp::params::LlamaCppParams;
use crate::worker::engine::{CtxLen, LoadParams};

use super::{Higgs, LoadedInfo, PairInfo, PreparedChat, ProfileState};

impl Higgs {
    // ── A1.0: prepare_chat — the full `/v1` chat gate ──────────────────────

    /// The shared pre-dispatch chat gate, lifted from `serve::v1::gate_and_validate`:
    /// JIT-load the requested model, resolve the now-resident served id, and clamp
    /// the generation budget to what fits the loaded context window.
    ///
    /// `max_tokens` is the client's requested budget (`None` = unspecified → infer
    /// the remaining window on a fixed context, else the 1024 default). It is
    /// `Option<usize>` rather than a bare `usize` so the `/v1` "fill the window when
    /// `max_tokens` is omitted" behavior is preserved without a magic-int sentinel.
    /// `messages_json` is the verbatim OpenAI `messages` array (a JSON string); its
    /// textual content bytes give the conservative lower-bound prompt estimate.
    ///
    /// Returns [`PreparedChat`] `{ resolved_model, max_gen }`. The resolved id binds
    /// dispatch so the worker rejects (`[HG018]`) a concurrent JIT swap; the clamped
    /// budget truncates an over-budget request instead of erroring, surfacing
    /// `context_length_exceeded` only when the prompt ALONE overflows the window.
    pub async fn prepare_chat(
        &self,
        model: &str,
        max_tokens: Option<usize>,
        messages_json: &str,
    ) -> Result<PreparedChat, HiggsError> {
        let loaded = self.resolve_loaded(model).await?;
        let prompt_bytes = estimate_prompt_bytes(messages_json);
        let max_gen = fit_generation_budget(max_tokens, loaded.ctx_len, prompt_bytes)?;
        Ok(PreparedChat {
            resolved_model: loaded.id,
            max_gen,
        })
    }

    /// Resolve the model that will serve this chat, loading it on demand when JIT is
    /// on — the [`serve::v1::ensure_loaded`] logic, returning the underlying
    /// [`HiggsError`] instead of a mapped HTTP `Response`.
    ///
    /// - Already served locally → its [`LoadedInfo`] (no load), UNLESS it is a
    ///   benchmark candidate (fall through to the remote check, then `[HG068]`).
    /// - Remote-resident → a permissive placeholder [`LoadedInfo`] (the fleet routes
    ///   it; the remote worker's `[HG005]` is the prompt-fit backstop).
    /// - Benchmark-owned and not remote → `[HG068]` `BenchInProgress`.
    /// - Not loaded, JIT OFF → `[HG003]` `ModelNotLoaded`.
    /// - Not loaded, JIT ON → must be a scanned id (`[HG002]` else), Prepared
    ///   (`[HG046]`/`[HG047]` else), then [`Higgs::load_inner`] loads the VALIDATED
    ///   profile; a load failure surfaces its own mapped error.
    async fn resolve_loaded(&self, model: &str) -> Result<LoadedInfo, HiggsError> {
        // Already locally served — serve it, no load. Benchmark exclusivity is
        // re-checked AFTER the awaited resident lookup (a bench can make its
        // candidate transiently resident during the await), mirroring `ensure_loaded`.
        let local_bench_candidate = match self.local_loaded_info(model).await {
            Some(loaded) if !self.is_benchmarking(model) => return Ok(loaded),
            Some(_) => true,
            None => false,
        };

        // Remote-resident model: the fleet routes it, so skip the local scan/JIT
        // gate and report a permissive (unknown-window) `LoadedInfo`.
        let is_remote = match self.fleet() {
            Some(f) => f.is_remote(model).await,
            None => false,
        };
        if is_remote {
            return Ok(LoadedInfo {
                id: model.to_owned(),
                worker_id: 0,
                ctx_len: None,
                gpu_layers: None,
                threads: None,
                arch: None,
                quant: None,
                max_context_length: None,
                size_bytes: None,
                has_chat_template: None,
                idle_ttl_minutes: None,
            });
        }

        // A benchmark owns this model and no remote node serves it → refuse [HG068].
        // The raw-flag re-check also covers the between-candidates phase.
        if local_bench_candidate || self.is_benchmarking(model) {
            return Err(HiggsError::BenchInProgress {
                id: model.to_owned(),
            });
        }

        // Not loaded, JIT off → explicit-load 404 [HG003].
        if !self.jit_enabled() {
            return Err(HiggsError::ModelNotLoaded {
                id: model.to_owned(),
            });
        }

        // JIT path: the id must be a scanned model (never load an unknown id).
        let scanned = self.scan().await?;
        if !scanned.iter().any(|m| m.id == model) {
            return Err(HiggsError::ModelNotFound {
                id: model.to_owned(),
            });
        }

        // Readiness gate: JIT only loads a Prepared (fresh-profile) model. Capture
        // the VALIDATED profile so the load uses exactly what was gated.
        let profile = match self.profile_state(model).await? {
            ProfileState::Ready(p) => p,
            ProfileState::Missing => {
                return Err(HiggsError::NotPrepared {
                    id: model.to_owned(),
                })
            }
            ProfileState::Stale => {
                return Err(HiggsError::ProfileStale {
                    id: model.to_owned(),
                })
            }
        };

        tracing::info!("higgs: JIT loading {model}");
        self.load_inner(model, Some(profile), false).await?;

        // Re-resolve: the requested model must now be resident.
        self.local_loaded_info(model).await.ok_or_else(|| {
            tracing::warn!(model, "higgs: JIT load succeeded but model not resident");
            HiggsError::ModelNotLoaded {
                id: model.to_owned(),
            }
        })
    }

    // ── A1.1 / A1.7: models-list assembly ──────────────────────────────────

    /// The enriched per-model rows the models view shows (formerly `GET
    /// /api/higgs/models`). Each row is a
    /// scanned [`HiggsModel`] enriched with load state, tool-call verdict, last
    /// load params, readiness/fit, and the dual tune profiles, all derived from ONE
    /// scan + ONE `models.json` snapshot + ONE hardware sample (so a concurrent tune
    /// can't make a row's readiness disagree with its profile fields).
    pub async fn model_entries(&self) -> Result<Vec<HiggsModelEntry>, HiggsError> {
        let models = self.scan().await?;
        let loaded_set = self.local_served_ids().await;
        let records = self.model_records();
        // One store snapshot: readiness's ACTIVE records and the dual-profile wire
        // fields are both derived from `profiles`, closing the two-read TOCTOU.
        let profiles = self.tuning_profiles()?;
        let tuning = crate::serve::control::active_records(&profiles);
        let hw = self.hardware().await;
        let mut entries = Vec::with_capacity(models.len());
        for m in models {
            let last_load = records.get(&m.id).and_then(|r| r.load.clone());
            let (readiness, fit) = self.model_readiness(&m, &loaded_set, &hw, &tuning);
            let tune = crate::serve::control::TuneProfileViews::from_triple(profiles.get(&m.id));
            entries.push(crate::serve::control::model_entry(
                m,
                &loaded_set,
                last_load,
                readiness,
                fit,
                tune,
            ));
        }
        Ok(entries)
    }

    /// A single enriched model row by HuggingFace repo id — the `GET
    /// /api/higgs/models/{*id}` behavior. `ModelNotFound` ([HG002] → 404) when the
    /// id is absent from the scanned catalog.
    pub async fn model_by_id(&self, id: &str) -> Result<HiggsModelEntry, HiggsError> {
        self.model_entries()
            .await?
            .into_iter()
            .find(|e| e.model.id == id)
            .ok_or_else(|| HiggsError::ModelNotFound { id: id.to_owned() })
    }

    // ── A1.2: hub lifecycle + node ops ─────────────────────────────────────

    /// Turn the hub network ON (the kill switch) — the `POST /api/higgs/hub/enable`
    /// orchestration. Binds the iroh endpoint + spawns the accept loop against the
    /// EXISTING fleet (routes preserved). Idempotent: a no-op returning the current
    /// status when already enabled. The whole check→start→publish runs under the
    /// hub-lifecycle lock so two enables can't orphan a loser endpoint.
    pub async fn hub_enable(&self) -> Result<HiggsHubStatus, HiggsError> {
        let _lifecycle = self.hub_lifecycle().lock().await;
        if self.hub().is_some() {
            return Ok(crate::serve::control::hub_status(self).await);
        }
        match crate::node::hub::start_hub(self.log_bus(), self.fleet()).await {
            Ok(hub) => {
                let hub = std::sync::Arc::new(hub);
                self.set_fleet(hub.fleet.clone());
                self.set_hub(hub);
                tracing::warn!("higgs: hub ENABLED");
                Ok(crate::serve::control::hub_status(self).await)
            }
            Err(e) => Err(HiggsError::HubControlFailed {
                op: "hub enable".into(),
                detail: e.to_string(),
            }),
        }
    }

    /// Turn the hub network OFF (the kill switch) — the `POST /api/higgs/hub/disable`
    /// orchestration. Closes the iroh endpoint + every node transport but KEEPS the
    /// fleet route table so re-enabling is a pure reconnect. Idempotent.
    pub async fn hub_disable(&self) -> HiggsHubStatus {
        let _lifecycle = self.hub_lifecycle().lock().await;
        if let Some(hub) = self.hub() {
            // Publish "disabled" FIRST (before any await) so a /pair waiting on the
            // lifecycle lock sees hub() None → not-a-hub once it runs.
            self.clear_hub();
            hub.shutdown().await;
            if let Some(fleet) = self.fleet() {
                fleet.disconnect_all().await;
            }
            tracing::warn!("higgs: hub DISABLED (network off; routes kept)");
        }
        crate::serve::control::hub_status(self).await
    }

    /// Mint a one-time node-pairing credential — the `POST /api/higgs/pair`
    /// behavior. Serialized against the kill switch so a mint runs either fully
    /// before a disable (valid) or fully after (sees no hub). Errors only when the
    /// server is not a hub (the caller maps that to a 409).
    pub async fn pair(&self) -> Result<PairInfo, HiggsError> {
        let _lifecycle = self.hub_lifecycle().lock().await;
        match self.hub() {
            Some(hub) => {
                let (ticket, token) = hub.mint_pairing().await;
                Ok(PairInfo {
                    hub_id: hub.hub_id().to_string(),
                    node_command: format!("higgs --node {ticket} {token}"),
                    ticket,
                    token,
                })
            }
            None => Err(HiggsError::HubControlFailed {
                op: "pair".into(),
                detail: "server is not running in hub mode (set HIGGS_HUB=1)".into(),
            }),
        }
    }

    /// Load a model on a paired node and record the route — `POST
    /// /api/higgs/nodes/load`. A thin wrapper over the fleet so the embedder never
    /// touches [`HubFleet`](crate::node::fleet::HubFleet) directly. Errors when the
    /// server is not a hub (no fleet installed).
    pub async fn node_load(&self, node: &str, model: &str) -> Result<WorkerId, HiggsError> {
        match self.fleet() {
            Some(fleet) => fleet.load(node, model).await,
            None => Err(not_a_hub_error("nodes/load")),
        }
    }

    /// Unload a remote-routed model and drop its route — `POST
    /// /api/higgs/nodes/unload`.
    pub async fn node_unload(&self, model: &str) -> Result<(), HiggsError> {
        match self.fleet() {
            Some(fleet) => fleet.unload(model).await,
            None => Err(not_a_hub_error("nodes/unload")),
        }
    }

    /// Retire a paired node for good (drop from the allowlist + the fleet) — `POST
    /// /api/higgs/nodes/retire`.
    pub async fn node_retire(&self, node: &str) -> Result<(), HiggsError> {
        match self.hub() {
            Some(hub) => hub
                .retire(node)
                .await
                .map_err(|e| HiggsError::HubControlFailed {
                    op: "retire".into(),
                    detail: e.to_string(),
                }),
            None => Err(not_a_hub_error("nodes/retire")),
        }
    }

    /// Rename a node — `POST /api/higgs/nodes/label`. `node == "local"` renames this
    /// instance's `config.json`; any other id renames that paired node's allowlist
    /// label (empty clears it). `Ok(true)` = renamed, `Ok(false)` = unknown remote
    /// node (the caller maps that to a 404); the remote path needs the hub enabled.
    pub async fn node_label(&self, node: &str, label: &str) -> Result<bool, HiggsError> {
        if node == "local" {
            return self
                .with_config_mut(|c| c.name = label.to_owned())
                .map(|()| true)
                .map_err(|e| HiggsError::PersistenceFailed {
                    store: "config".into(),
                    path: "config.json".into(),
                    source: e,
                });
        }
        // Serialize against the kill switch with the SAME lock as `/pair`.
        let _lifecycle = self.hub_lifecycle().lock().await;
        let hub = self.hub().ok_or_else(|| not_a_hub_error("label"))?;
        let label = (!label.is_empty()).then(|| label.to_owned());
        hub.set_label(node, label)
            .await
            .map_err(|e| HiggsError::HubControlFailed {
                op: "label".into(),
                detail: e.to_string(),
            })
    }

    /// A paired node's on-disk model catalog — `GET /api/higgs/nodes/{node}/models`.
    /// Wraps `fleet().scan_node`; the node's `{ "models": [...] }` reply is returned
    /// verbatim.
    pub async fn node_scan(&self, node: &str) -> Result<Value, HiggsError> {
        match self.fleet() {
            Some(fleet) => fleet.scan_node(node).await,
            None => Err(not_a_hub_error("nodes/{node}/models")),
        }
    }

    /// The unified fleet view — the `GET /api/higgs/nodes` behavior lifted onto the
    /// facade. The LOCAL machine comes FIRST (`is_local`, always present, even with
    /// the hub role off), labelled with this instance's `config.json` name (via
    /// [`Higgs::instance_name`], falling back to `"this machine"`). Then each paired
    /// remote node, with its label filled from the hub allowlist — the
    /// operator-editable source of truth so the view reflects renames — falling back
    /// to the node's reported hostname, else a short endpoint id. The Fleet view shows
    /// EVERY resident model on EVERY node, so the same raw model loaded both locally
    /// and on a remote node legitimately appears on both (chat resolves local-first;
    /// the view's job is full visibility, not routing).
    pub async fn nodes(&self) -> Vec<crate::node::fleet::NodeView> {
        // The local node always appears first, labelled with this instance's config.json name.
        let local_label = self
            .instance_name()
            .unwrap_or_else(|| "this machine".to_string());
        let mut out = vec![self.local_node_view(local_label).await];

        // Then the remote fleet, with each node's label filled from the allowlist so
        // the UI shows a human name.
        if let Some(fleet) = self.fleet() {
            let labels = self.node_labels().await;
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
        out
    }

    /// Node labels keyed by `EndpointId`, from the live hub allowlist when the hub is
    /// enabled, else read straight from the persisted `pairings.json`. The kill switch
    /// (`clear_hub`) drops the `Hub` but KEEPS the fleet, so this disk fallback keeps a
    /// remote node's operator label stable across an enable/disable window. A
    /// missing/unreadable file yields no labels (callers fall back to hostname).
    async fn node_labels(&self) -> std::collections::HashMap<String, Option<String>> {
        if let Some(hub) = self.hub() {
            return hub.labels().await;
        }
        let path = crate::home::higgs_home().join("pairings.json");
        crate::auth::Allowlist::load(&path)
            .map(|allow| allow.labels())
            .unwrap_or_default()
    }

    // ── A1.3: load_flat / unload_spec ──────────────────────────────────────

    /// Load a model from the flat `POST /api/higgs/models/load` request shape,
    /// building the [`LoadParams`] the handler used to build inline. A request with
    /// NO pinned field is a fully-default load (`None`); a full `params` supersedes
    /// the flat fields; otherwise the three base fields fall back to `default_load`
    /// and every optional override passes through verbatim.
    pub async fn load_flat(&self, req: &HiggsLoadRequest) -> Result<(), HiggsError> {
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
            Some(full)
        } else if !any_pinned {
            None
        } else {
            let base = self.default_load();
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
        self.load(&req.id, params).await
    }

    /// Unload one model by served id, or ALL when `None` — the drain logic behind
    /// `POST /api/higgs/models/unload` (the raw-body "id present vs absent" parsing
    /// stays at the HTTP boundary, which alone must 400 a malformed body).
    pub async fn unload_spec(&self, id: Option<&str>) -> Result<(), HiggsError> {
        match id {
            Some(id) => self.unload_one(id).await,
            None => self.unload().await,
        }
    }

    // ── A1.4: trusted mint / revoke (bearer-free, invariants intact) ────────

    /// Mint an API key IN-PROCESS (trusted): skips ONLY the bearer authentication,
    /// KEEPING every structural invariant the HTTP handler enforces — label
    /// validation, explicit-empty-scopes rejection, `Duplicate`, and
    /// `BootstrapNeedsAdmin` (a non-admin first key would self-lock the surface).
    /// The decision + mutation run inside the keystore lock (the same race guard the
    /// handler uses) via the shared [`decide_mint`](crate::serve::control::decide_mint).
    pub fn mint_key(
        &self,
        label: &str,
        scopes: Option<Vec<Scope>>,
    ) -> Result<HiggsMintKeyResponse, HiggsError> {
        use crate::serve::control::{decide_mint, keystore_io_error, validate_key_label, Mint};
        let label = label.trim().to_owned();
        validate_key_label(&label)?;
        // An EXPLICIT empty scope list is a client error regardless of store state.
        if scopes.as_ref().is_some_and(Vec::is_empty) {
            return Err(HiggsError::InvalidKeyRequest {
                detail: "at least one scope required (chat, models, admin)".into(),
            });
        }
        let token = crate::keys::mint_token(rand::random());
        let outcome = self
            .mutate_api_keys(|ks| {
                let decision = decide_mint(ks, true, None, scopes, &label);
                if let Mint::Ok(scopes) = &decision {
                    ks.add(&token, label.clone(), scopes.clone());
                }
                decision
            })
            .map_err(keystore_io_error)?;
        match outcome {
            Mint::Ok(scopes) => Ok(HiggsMintKeyResponse {
                label,
                scopes,
                token,
            }),
            Mint::Duplicate => Err(HiggsError::InvalidKeyRequest {
                detail: format!("a key labeled {label:?} already exists — revoke it first"),
            }),
            Mint::BootstrapNeedsAdmin => Err(HiggsError::InvalidKeyRequest {
                detail: "the first API key must include the `admin` scope (it is the only key able to manage keys) — pass scopes: [\"admin\"], or omit scopes to default to admin".into(),
            }),
            // trusted=true skips the Unauthorized branch, so this is unreachable.
            Mint::Unauthorized => Err(HiggsError::InternalFault {
                context: "mint_key".into(),
                detail: "trusted mint returned Unauthorized".into(),
            }),
        }
    }

    /// Revoke every key with `label` IN-PROCESS (trusted): skips ONLY the bearer
    /// authentication, KEEPING the last-admin lockout ([HG066]) and last-key-on-LAN
    /// ([HG059]) invariants, accounted over `visible()` keys, via the shared
    /// [`decide_revoke`](crate::serve::control::decide_revoke).
    pub fn revoke_key(&self, label: &str) -> Result<HiggsKeyRemoved, HiggsError> {
        use crate::serve::control::{decide_revoke, keystore_io_error, Revoke};
        let outcome = self
            .mutate_api_keys(|ks| {
                // Read the LAN exposure INSIDE the keystore critical section, not as
                // a snapshot taken before it. `mutate_api_keys` holds `keys_io`
                // across decide-and-commit; a `lan_exposed()` sampled outside that
                // window can go stale between the sample and the commit, and this
                // one degrades UNSAFELY: a `serve_v1` arming a LAN listener in that
                // gap would pass its own [HG058] key check (the key is still there),
                // while this revoke commits against `lan = false` and empties the
                // store — leaving a KEYLESS LAN surface. Reading here makes the
                // exposure test and the removal atomic with respect to each other,
                // so one of the two operations always loses: either this revoke sees
                // the armed listener and refuses ([HG059]), or the listener's key
                // check sees the emptied store and refuses ([HG058]).
                //
                // Lock order is `keys_io` → `serves` (via `lan_exposed`); nothing
                // takes them the other way round, so this cannot deadlock.
                //
                // NOT PINNED BY A TEST — same class as `arm_lan_serve`'s scope note:
                // hoisting this read back out of the closure breaks no test, because
                // every test sets the exposure statically with no concurrent arm. The
                // property is a lock SCOPE, and proving it would need a test-only
                // injection point in production code, which this crate forbids. It is
                // held by construction: this read and the commit share one `keys_io`
                // critical section, and `Higgs::arm_lan_serve` arms under the same lock.
                let lan = self.lan_exposed();
                let decision = decide_revoke(ks, true, None, label, lan);
                if let Revoke::Removed(_) = &decision {
                    let _ = ks.remove_label(label);
                }
                decision
            })
            .map_err(keystore_io_error)?;
        let removed = match outcome {
            Revoke::Removed(n) => n,
            Revoke::LastKeyOnLan => {
                return Err(HiggsError::LastKeyOnLan {
                    label: label.to_owned(),
                })
            }
            Revoke::LastAdminKey => {
                return Err(HiggsError::LastAdminKey {
                    label: label.to_owned(),
                })
            }
            // trusted=true skips the Unauthorized branch, so this is unreachable.
            Revoke::Unauthorized => {
                return Err(HiggsError::InternalFault {
                    context: "revoke_key".into(),
                    detail: "trusted revoke returned Unauthorized".into(),
                })
            }
        };
        if removed == 0 {
            return Err(HiggsError::InvalidKeyRequest {
                detail: format!(
                    "no key labeled {label:?} — nothing was revoked; list the keys for the current labels"
                ),
            });
        }
        Ok(HiggsKeyRemoved {
            removed: removed as u64,
            auth_enabled: !self.api_keys().is_empty(),
        })
    }

    // ── A1.7: remaining thin control wrappers ──────────────────────────────

    /// The current Developer-Log toggles — `GET /api/higgs/logs/settings`.
    pub fn logs_settings(&self) -> LogSettings {
        LogSettings {
            verbose: self.verbose(),
            log_incoming_tokens: self.log_incoming_tokens(),
            show_log_fields: self.log_show_fields(),
        }
    }

    /// Set both Developer-Log toggles — `PUT /api/higgs/logs/settings`.
    pub fn set_logs_settings(&self, settings: &LogSettings) {
        self.set_verbose(settings.verbose);
        self.set_log_incoming_tokens(settings.log_incoming_tokens);
        self.set_log_show_fields(settings.show_log_fields);
    }

    /// The extra CORS-origins allowlist state: what's persisted in `config.json`
    /// now (`origins`), what the RUNNING server booted with (`applied_origins`), and
    /// whether they differ (`restart_required`).
    ///
    /// The extra origins are read ONCE at serve start when the CORS layer is built,
    /// so a persisted change applies only on the next restart — `restart_required`
    /// surfaces that honestly (a live rebind of the running layer is a separate,
    /// deferred feature). CORS only protects BROWSER clients; non-browser access is
    /// gated by API keys, not this list.
    pub fn cors_settings(&self) -> HiggsCorsSettings {
        let origins = self.extra_cors_origins();
        // `applied` is `None` until a CORS layer has actually been built (pre-serve).
        // Pre-serve there is nothing to diverge from — the first serve start applies
        // the persisted list — so `restart_required` is `false`, NOT a comparison
        // against a flattened-empty applied list (which would falsely flag a restart
        // for origins set before the server ever served).
        let applied = self.applied_cors_origins();
        // ANY live listener running a different allowlist means a restart is needed
        // — not just the primary one whose list `applied_origins` discloses. The
        // comparison is by SET (the allowlist is exact-match membership, so order is
        // meaningless to the layer): a reordered save of the same origins must not
        // claim a restart. `false` when nothing is live (the first serve applies the
        // persisted list, so nothing is pending).
        let restart_required = self.any_live_serve_cors_differs(&origins);
        HiggsCorsSettings {
            origins,
            applied_origins: applied.unwrap_or_default(),
            restart_required,
        }
    }

    /// Replace the persisted extra CORS-origins allowlist. Each entry must be a
    /// bare `http(s)://host[:port]` origin (no userinfo/path/query/fragment) — an
    /// invalid entry is rejected with `[HG071]` BEFORE anything is persisted. Every
    /// accepted entry is CANONICALIZED to the exact string a browser sends in the
    /// `Origin` header (lowercased host, default port stripped) so the exact-match
    /// allowlist can match; the STORED value is that canonical form, not the raw
    /// input. Repeated origins (after canonicalization) are deduped (first-seen
    /// order preserved) rather than erroring. The normalized list is written to
    /// `config.json` via [`Higgs::with_config_mut`]; the returned
    /// [`HiggsCorsSettings`] reflects the new persisted state (so `restart_required`
    /// flips `true` when it diverges from the running server's boot-applied list).
    pub fn set_cors_origins(&self, origins: Vec<String>) -> Result<HiggsCorsSettings, HiggsError> {
        let normalized = validate_and_dedup_cors_origins(origins)?;
        self.with_config_mut(|c| c.cors_origins = normalized.clone())
            .map_err(|e| HiggsError::PersistenceFailed {
                store: "config".into(),
                path: "config.json".into(),
                source: e,
            })?;
        Ok(self.cors_settings())
    }

    /// Unload every resident worker, freeing their memory — `POST
    /// /api/higgs/worker/stop`. The server stays up; a later load (or JIT chat)
    /// spawns a fresh worker. A NON-terminal bulk unload (not the terminal `stop`).
    pub async fn worker_stop(&self) -> Result<(), HiggsError> {
        self.unload().await
    }

    // ── A1.8: the `/v1/models` union ───────────────────────────────────────

    /// The exact set of model ids chat can REACH — the `serve::v1::v1_models` union
    /// lifted onto the facade: `local_served_ids()` (always) ∪ `servable_model_ids()`
    /// (ONLY when `jit_enabled()`) ∪ `fleet().routed_models()` (remote-routed),
    /// deduped, minus any model whose only local candidate is transiently benchmarking
    /// (unless a remote node also serves it). Using `servable_model_ids()` alone would
    /// omit fleet-routed remotes and (JIT-off) local-served ids, silently shrinking
    /// the picker — hence this method, not that call.
    pub async fn chat_model_ids(&self) -> Vec<String> {
        let mut ids = self.local_served_ids().await;
        // Advertise servable (JIT-loadable) catalog ids too, but ONLY while JIT is
        // on — with JIT off an unloaded servable model is not reachable.
        if self.jit_enabled() {
            for id in self.servable_model_ids().await {
                if !ids.contains(&id) {
                    ids.push(id);
                }
            }
        }
        // Remote-resident ids routed through the fleet (skip any already local).
        let remote_ids: HashSet<String> = match self.fleet() {
            Some(fleet) => fleet.routed_models().await.into_iter().collect(),
            None => HashSet::new(),
        };
        for id in &remote_ids {
            if !ids.contains(id) {
                ids.push(id.clone());
            }
        }
        // A locally-benchmarking model can't serve from its transient candidate
        // ([HG068]) unless a remote node ALSO serves the id (dispatch routes there).
        ids.retain(|id| !self.is_benchmarking(id) || remote_ids.contains(id));
        ids
    }

    /// Build + engine version info (formerly `GET /api/higgs/version`). The version
    /// strings (`CARGO_PKG_VERSION`, `LLAMA_CPP_2_VERSION`) are crate-internal to
    /// higgs, so an embedder needs this method to surface them (it cannot read them
    /// across the crate boundary).
    pub fn version(&self) -> HiggsVersionResponse {
        HiggsVersionResponse {
            higgs: env!("CARGO_PKG_VERSION").to_owned(),
            engine: "llama.cpp".to_owned(),
            engine_version: crate::worker::engine::llamacpp::engine_version(),
            binding: crate::LLAMA_CPP_2_VERSION.to_owned(),
            supported_formats: vec!["gguf".to_owned()],
        }
    }
}

/// The `HubControlFailed` a node op returns when the server is not a hub. The HTTP
/// handlers keep their own pre-check so the socket still answers the original 409;
/// this is the error an in-process embedder sees when it drives a node op with no
/// fleet/hub installed.
fn not_a_hub_error(op: &str) -> HiggsError {
    HiggsError::HubControlFailed {
        op: op.to_owned(),
        detail: "server is not running in hub mode (set HIGGS_HUB=1)".into(),
    }
}

/// Validate + CANONICALIZE every extra CORS origin, then DEDUP repeated entries
/// preserving first-seen order. Each entry is normalized by [`validate_cors_origin`]
/// to the exact string a browser sends in the `Origin` header; the first invalid
/// entry short-circuits with its `[HG071]` error (nothing is persisted). Dedup
/// operates on the CANONICAL forms, so two inputs that browsers serialize
/// identically (`https://EXAMPLE.com` and `https://example.com`) collapse to one.
fn validate_and_dedup_cors_origins(origins: Vec<String>) -> Result<Vec<String>, HiggsError> {
    let mut seen = HashSet::with_capacity(origins.len());
    let mut out = Vec::with_capacity(origins.len());
    for origin in origins {
        let canonical = validate_cors_origin(&origin)?;
        if seen.insert(canonical.clone()) {
            out.push(canonical);
        }
    }
    Ok(out)
}

/// Validate + CANONICALIZE ONE extra CORS origin, returning the exact ASCII string
/// a browser puts in the `Origin` header (which the allowlist matches verbatim).
/// The input is parsed with the WHATWG `url` crate — this is normalization, not
/// mere validation: the returned value is `url.origin().ascii_serialization()`,
/// i.e. lowercased host and default port (80/443) stripped, so `https://EXAMPLE.com`
/// becomes `https://example.com` and `http://example.com:80` becomes
/// `http://example.com`. `Url::parse` itself rejects the shapes a browser never
/// emits (unbracketed IPv6, unbalanced brackets, out-of-range ports). Anything that
/// is not a bare `http`/`https` origin — a non-http scheme, any userinfo
/// (username/password), a path beyond a single trailing `/` (checked on the RAW
/// input too, since the parser normalizes dot-segments like `/app/..` away), any
/// query or fragment, or a missing host — is `[HG071] InvalidCorsOrigin` with the
/// specific reason.
pub(crate) fn validate_cors_origin(origin: &str) -> Result<String, HiggsError> {
    let reject = |reason: &str| HiggsError::InvalidCorsOrigin {
        origin: origin.to_owned(),
        reason: reason.to_owned(),
    };
    let url = url::Url::parse(origin).map_err(|e| reject(&e.to_string()))?;
    match url.scheme() {
        "http" | "https" => {}
        _ => return Err(reject("must start with http:// or https://")),
    }
    // The parser NORMALIZES dot-segments and backslashes (`/app/..`, `/%2e%2e`,
    // `\app` all collapse into the parsed path — possibly to a bare `/`) BEFORE
    // the `url.path()` check below runs, so a pasted URL that VISIBLY carries a
    // path could otherwise slip through as its origin. Enforce "bare origin" on
    // the RAW input too: after the `://`, the only separator allowed is a single
    // trailing `/`. (A scheme-relative shorthand like `https:example.com` — which
    // the parser tolerates but a browser never emits — has no `://` and is
    // rejected here.)
    let rest = origin
        .find("://")
        .map(|sep| &origin[sep + 3..])
        .ok_or_else(|| reject("must start with http:// or https://"))?;
    if let Some(i) = rest.find(['/', '\\']) {
        if !(i == rest.len() - 1 && rest.as_bytes()[i] == b'/') {
            return Err(reject(
                "must not contain a path (a single trailing '/' is allowed)",
            ));
        }
    }
    if !url.username().is_empty() || url.password().is_some() {
        return Err(reject("must not contain a username or password"));
    }
    if !matches!(url.path(), "" | "/") {
        return Err(reject(
            "must not contain a path (a single trailing '/' is allowed)",
        ));
    }
    if url.query().is_some() {
        return Err(reject("must not contain a query string"));
    }
    if url.fragment().is_some() {
        return Err(reject("must not contain a fragment"));
    }
    if url.host().is_none() {
        return Err(reject("missing host"));
    }
    Ok(url.origin().ascii_serialization())
}

/// Conservative lower-bound prompt-token pre-check: sum the byte length of each
/// message's TEXTUAL content (array text parts joined with `\n`, the shimmy
/// convention), matching `serve::v1::messages_to_pairs`'s content extraction so the
/// facade estimate agrees with the `/v1` handler's. A message carrying any non-text
/// content part (image/audio/file/refusal) makes the whole estimate `0` — mirroring
/// `messages_to_pairs`'s `Err → 0` — since such a request is rejected by the
/// handler's text-only check regardless. The worker's tokenizer-exact `[HG005]` is
/// the authoritative backstop.
fn estimate_prompt_bytes(messages_json: &str) -> usize {
    let Ok(Value::Array(messages)) = serde_json::from_str::<Value>(messages_json) else {
        return 0;
    };
    let mut total = 0usize;
    for m in &messages {
        match message_text_len(m) {
            Some(n) => total += n,
            None => return 0,
        }
    }
    total
}

/// The textual byte length of one OpenAI message's content, or `None` if it carries
/// a non-text content part. `content` absent/null → 0 (assistant `None` /
/// function null); a string → its byte length; an array → the `\n`-joined text
/// parts' byte length (`Σ len + (parts − 1)` separators).
fn message_text_len(m: &Value) -> Option<usize> {
    match m.get("content") {
        None | Some(Value::Null) => Some(0),
        Some(Value::String(s)) => Some(s.len()),
        Some(Value::Array(parts)) => {
            let mut lens = Vec::with_capacity(parts.len());
            for p in parts {
                if p.get("type").and_then(Value::as_str) != Some("text") {
                    return None;
                }
                lens.push(p.get("text").and_then(Value::as_str).unwrap_or("").len());
            }
            let sum: usize = lens.iter().sum();
            Some(sum + lens.len().saturating_sub(1))
        }
        // A non-string, non-array content is not a shape `messages_to_pairs`
        // produces text from — treat as empty.
        Some(_) => Some(0),
    }
}

/// Resolve the GENERATION budget for a request against the loaded context window —
/// the `serve::v1::fit_generation_budget` logic lifted onto the facade. `requested`
/// is the client's `max_tokens` (`None` = infer). Returns `Err(ContextOverflow)`
/// (→ `context_length_exceeded`) ONLY when the prompt ALONE can't fit; otherwise the
/// budget is CLAMPED to `min(requested or available, available, MAX_OUTPUT_TOKENS)`
/// so an oversized request truncates instead of failing. An AUTO/unknown window
/// can't be bounded here, so the requested budget (or the 1024 default) stands,
/// capped at the absolute limit; the worker's `[HG005]` is the backstop.
pub(crate) fn fit_generation_budget(
    requested: Option<usize>,
    ctx_len: Option<CtxLen>,
    prompt_bytes: usize,
) -> Result<usize, HiggsError> {
    let prompt_tokens_est = prompt_bytes / crate::serve::PROMPT_BYTES_PER_TOKEN;
    let max_out = crate::serve::MAX_OUTPUT_TOKENS as usize;
    match ctx_len {
        None | Some(CtxLen::Auto) => Ok(requested.unwrap_or(1024).min(max_out)),
        Some(CtxLen::Fixed { n }) => {
            let n_ctx = n as usize;
            if prompt_tokens_est >= n_ctx {
                return Err(HiggsError::ContextOverflow {
                    prompt_tokens: prompt_tokens_est,
                    max_gen: requested.unwrap_or(0),
                    n_ctx,
                });
            }
            let available = n_ctx - prompt_tokens_est;
            let budget = requested.unwrap_or(available);
            Ok(budget.min(available).min(max_out).max(1))
        }
    }
}

#[cfg(test)]
#[path = "embed_tests.rs"]
mod tests;
