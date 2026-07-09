//! `Higgs` public facade and `HiggsConfig` — the host-facing API.
//!
//! One `Higgs` instance per host app. Typed facade over a co-located LOCAL
//! [`NodeRuntime`](crate::node::runtime::NodeRuntime) — the same multi-worker
//! engine remote nodes run (P4b). `Higgs` owns the facade-level state (config,
//! the load lifecycle mutex, the inference admission gate, the serve-layer
//! toggles) and delegates worker spawn/load/unload/status/chat, idle
//! auto-unload, and the Developer-Log bus to the node. A remote-resident model
//! routes through the `fleet` instead. The host maps its own config table onto
//! [`HiggsConfig`].

use std::path::PathBuf;
use std::sync::Arc;

use tokio::sync::broadcast;

use crate::diagnostic::HiggsError;
use crate::log_bus::{LogBus, LogLine, LogSource};
use crate::node::runtime::{NodeConfig, NodeRuntime, DEFAULT_IDLE_TTL};
use crate::node::worker_id::WorkerId;
use crate::remote::{InventoryWorker, NodeInventory, NodeLoadParams};
use crate::supervisor::HiggsEvent;
use crate::system::HardwareInfo;
use crate::tune::store::{JsonModelStore, TuneRecord};
use crate::tune::{
    BenchResult, EstimateReport, EstimateRequest, ModelMeta, RamEstimator, Suggester, TuneMode,
    TuneRequest, TuneSuggestion, VramEstimator,
};
use crate::worker::engine::LoadParams;
use crate::worker::models::HiggsModel;

// Submodules split out of this file (see api/README.md, api/DESIGN.md). `pub use`
// keeps every existing `crate::api::*` path resolving unchanged.
mod embed;
mod guards;
mod types;

#[cfg(test)]
use guards::fits_in_memory;
use guards::validate_repo_id;
pub(crate) use guards::{guard_memory_headroom, path_within_roots};
pub use types::*;

/// Freshness verdict for a saved tuning profile, consumed by the JIT gate.
#[derive(Debug, Clone)]
pub(crate) enum ProfileState {
    /// No profile — the model was never Prepared.
    Missing,
    /// Profile exists but the hardware or model file changed since Prepare.
    Stale,
    /// Profile exists and matches the current hardware + file. Carries the
    /// VALIDATED `LoadParams` so the JIT path loads exactly what was checked,
    /// without a second `models.json` read (closing the check-then-load race).
    Ready(LoadParams),
}

/// Is the saved profile stale for the given on-disk path + current hardware?
/// Stale when either anchor differs (an empty stored anchor — e.g. a bare load,
/// not a Prepare — never matches, so it reads as stale → must Re-tune).
fn profile_stale(rec: &TuneRecord, path: &str, hw: &crate::system::HardwareInfo) -> bool {
    rec.is_stale(&hw.fingerprint(), &file_sig(path))
}

/// Cheap model-file identity (`"{len}:{mtime_ms}"`) for staleness checks on a
/// tuning profile. Empty string when the file can't be stat'd — an empty sig
/// never matches a real one, so the profile reads as stale (needs Re-tune),
/// which fails safe.
pub(crate) fn file_sig(path: &str) -> String {
    let Ok(meta) = std::fs::metadata(path) else {
        return String::new();
    };
    let mtime_ms = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_millis())
        .unwrap_or(0);
    format!("{}:{}", meta.len(), mtime_ms)
}

/// The in-process handle to the higgs runtime. One instance per host app.
///
/// The host-facing facade over the co-located LOCAL [`NodeRuntime`]. `Higgs` owns
/// the facade-level state — the live [`HiggsConfig`], the `lifecycle` mutex that
/// serializes concurrent loads, the `inference_gate` admission semaphore, and the
/// serve-layer toggles — while the node owns the worker processes, RPC
/// correlation, idle auto-unload, and the load-replay state. Constructing `Higgs`
/// does not start a worker; workers are spawned lazily, one per [`load`](Self::load).
///
/// Freshness window for the live-estimate per-id metadata memo
/// ([`Higgs::estimate`]). Long enough that a burst of slider edits reuses one scan,
/// short enough that a model deleted/replaced underneath the UI self-heals quickly.
const ESTIMATE_META_TTL: std::time::Duration = std::time::Duration::from_secs(2);

pub struct Higgs {
    /// The co-located LOCAL node (P4b): an in-process multi-worker `NodeRuntime`, the same
    /// engine remote nodes run. Local inference/load/unload/status route through it; a remote
    /// model routes through the `fleet`. (Replaces the old single direct `Supervisor`.)
    local: Arc<NodeRuntime>,
    config: parking_lot::Mutex<HiggsConfig>,
    /// Serializes concurrent loads so two racing JIT loads of the same id can't
    /// each spawn a worker (the node load is additive — it never dedups).
    lifecycle: tokio::sync::Mutex<()>,
    /// Inference admission gate — at most [`MAX_CONCURRENT_INFERENCE`] chat
    /// requests in flight. A `try_acquire_owned` failure on the chat path is a
    /// [`HiggsError::ServerBusy`] (HTTP 503); the owned permit rides the spawned
    /// generation task and releases on its completion (success/error/timeout).
    inference_gate: Arc<tokio::sync::Semaphore>,
    /// Admission gate for REMOTE (fleet-routed) chats — capped at
    /// [`MAX_CONCURRENT_INFERENCE`] like the local gate, but SEPARATE so a flood of remote
    /// requests can't grow hub/node tasks + streams unbounded, AND so it doesn't entangle the
    /// idle reaper's "acquire all LOCAL permits to unload" logic (remote traffic must not keep
    /// a local model resident — see `chat_stream`). `None` until a fleet is installed is not
    /// needed: the gate exists always; it's only used on the remote branch.
    remote_gate: Arc<tokio::sync::Semaphore>,
    /// Runtime "Log Incoming Tokens" toggle for the serve layer. When `true`, the
    /// chat path emits an extra INFO `higgs:`-target line per request carrying the
    /// (capped) flattened incoming prompt CONTENT so the Developer Logs show the
    /// actual prompt. This is the explicit OPT-IN that overrides the redact-by-
    /// default policy (no prompt content at info); default `false`, so the
    /// redaction policy is unchanged unless the user turns this on. A plain atomic
    /// — set/read in isolation, no critical section, never across `.await`.
    log_incoming_tokens: std::sync::atomic::AtomicBool,
    /// Runtime "Just-in-Time loading" toggle for the serve layer. When `true`
    /// (the default), a chat request for a scanned-but-unloaded model triggers an
    /// on-demand additive [`load`](Self::load) (a fresh worker alongside any
    /// others — the local node is multi-model) instead of the `[HG003]` 404. When
    /// `false`, the chat path keeps the explicit-load behavior: an unloaded model
    /// is a 404. A plain atomic — set/read in isolation, no critical section,
    /// never across `.await`. Defaults to `true` (JIT on).
    jit_enabled: std::sync::atomic::AtomicBool,
    /// Runtime "Auto-unload idle models" toggle, mirrored to the node's live
    /// [`IdleConfig`](crate::node::runtime::IdleConfig) by
    /// [`set_auto_unload_idle`](Self::set_auto_unload_idle). When `true` (the
    /// default), the node's per-worker idle reaper unloads a worker after the idle
    /// TTL of no inference; when `false` it never auto-unloads. This atomic is the
    /// facade-side mirror the Server-Settings getter reads. A plain atomic —
    /// set/read in isolation, never across `.await`.
    auto_unload_idle: std::sync::atomic::AtomicBool,
    /// Runtime idle auto-unload TTL, in minutes, mirrored to the node's live
    /// [`IdleConfig`](crate::node::runtime::IdleConfig) by
    /// [`set_idle_ttl_minutes`](Self::set_idle_ttl_minutes). Seeded from the node's
    /// [`DEFAULT_IDLE_TTL`] (60 minutes). This atomic is the facade-side mirror the
    /// Server-Settings getter reads. A plain atomic — set/read in isolation, never
    /// across `.await`.
    idle_ttl_minutes: std::sync::atomic::AtomicU64,
    /// Runtime "serving on/off" gate for the `/v1` inference surface. When `false`,
    /// the `/v1` inference endpoints return `[HG019]` → 503 while the
    /// in-process control surface stays reachable so the user can re-enable.
    /// Defaults to `true` (serving on). A plain atomic — set/read in isolation,
    /// never across `.await`.
    serving_enabled: std::sync::atomic::AtomicBool,
    /// Cached host device list gathered once via the node's transient sysinfo
    /// worker ([`NodeRuntime::gpus`](crate::node::runtime::NodeRuntime::gpus)).
    /// Hardware is static-ish, so it is gathered on first request and reused; `None` until
    /// the first successful gather. A failed gather leaves it `None` so a later
    /// request retries. A plain `parking_lot::Mutex` — held only for the
    /// read/insert, never across `.await`.
    device_cache: parking_lot::Mutex<Option<Vec<crate::system::GpuDevice>>>,
    /// Per-id GGUF-metadata memo for the LIVE estimate path (`estimate`). The
    /// load-params UI calls `/models/estimate` on every edit; a full `scan()` walks
    /// every model dir + re-mmaps each GGUF, so without this each keystroke would
    /// re-scan. Caches the LAST resolved `(id, ModelMeta, fetched_at)` — repeated
    /// estimates for the same model within [`ESTIMATE_META_TTL`] reuse it; a different
    /// id or an expired entry re-scans + replaces. The short TTL bounds staleness so a
    /// model deleted/replaced underneath the UI self-heals (→ a fresh scan → 404 or
    /// current metadata) within a couple of seconds. `load`/`tune` always scan fresh.
    /// A plain `parking_lot::Mutex` — held only for the read/insert, never across `.await`.
    estimate_meta_cache: parking_lot::Mutex<Option<(String, ModelMeta, std::time::Instant)>>,
    /// The remote fleet (paired nodes + model routing), installed when the hub runs an
    /// iroh listener. `None` for a pure-local higgs. `/v1` chat for a remote-resident model
    /// routes through this instead of the local `Supervisor` (the two correlation domains
    /// stay separate, DESIGN-remote.md §2.3).
    fleet: parking_lot::Mutex<Option<Arc<crate::node::fleet::HubFleet>>>,
    /// API-key store gating the HTTP surface (P5). Default is empty = auth OFF (the embedded
    /// in-process host wants no gate); the standalone binary loads `api_keys.json` at startup.
    /// Swapped wholesale, so a reload is one atomic store.
    api_keys: parking_lot::Mutex<Arc<crate::keys::ApiKeys>>,
    /// The running hub (P3), when the server is in hub mode — used by the pairing API to mint
    /// node-join tokens. `None` for a pure-local / non-hub server.
    hub: parking_lot::Mutex<Option<Arc<crate::node::hub::Hub>>>,
    /// Serializes the hub kill-switch lifecycle ops (`enable`/`disable`). Held across the WHOLE
    /// check→start→publish (enable) / shutdown→disconnect→clear (disable) sequence so two
    /// concurrent enables can't both pass the `hub().is_none()` check, bind two endpoints + accept
    /// loops, and orphan the loser (whose accept loop would keep admitting nodes after a later
    /// disable, defeating the kill switch). SEPARATE from `lifecycle` (local model load/unload) so
    /// the two never cross-couple. An async `Mutex` because the guard is held across `.await`.
    hub_lifecycle: tokio::sync::Mutex<()>,
    /// Serializes read-modify-write of `config.json` (load records, instance rename, autoload
    /// toggles). The save is atomic (temp+rename) so the LAST writer wins — but two concurrent
    /// RMWs would each load the file, mutate a DIFFERENT field, and the second save would clobber
    /// the first's change. Holding this for the whole load→mutate→save makes each RMW atomic w.r.t.
    /// the others. A plain `parking_lot::Mutex` — held only across synchronous file I/O, never an
    /// `.await`. Pure reads of `config.json` (e.g. the node view's label) need no lock: the atomic
    /// rename means a reader always sees a complete prior-or-next version.
    config_io: parking_lot::Mutex<()>,
    /// Serializes read-modify-write of the per-node `models.json` (tuning records).
    /// `models.json` is rewritten WHOLESALE on flush, so two concurrent `tune`
    /// calls that each opened their own snapshot would have the last flush clobber
    /// the other's new `TuneRecord`. Holding this across a fresh re-open → mutate →
    /// atomic-save makes each write re-read the latest file first. A plain
    /// `parking_lot::Mutex` — held only across synchronous file I/O, never an
    /// `.await`. Mirrors [`Self::config_io`].
    models_io: parking_lot::Mutex<()>,
    /// The model load CURRENTLY in flight on the local node, or `None`. Set around
    /// the blocking [`local.load`](crate::node::runtime::NodeRuntime::load) so a
    /// concurrent `status` request can surface a "loading…" progress indicator to
    /// the UI. A plain `parking_lot::Mutex` — set/cleared in isolation, never held
    /// across the load `.await`.
    loading: parking_lot::Mutex<Option<crate::api::types::ModelLoading>>,
    /// Live fan-out of model-load lifecycle events ([`ModelLoadEvent`]) to
    /// `GET /api/higgs/events` SSE subscribers. Pushed at each real phase seam in
    /// [`load_inner`](Self::load_inner) so the UI drives its loading indicator from
    /// PUSH events instead of polling `status`. A `send` with no subscribers is a
    /// harmless no-op; late subscribers simply miss already-past phases (the bar is
    /// transient — a subscriber that joins mid-load still gets the remaining phases +
    /// the terminal event).
    load_events: tokio::sync::broadcast::Sender<crate::api::types::ModelLoadEvent>,
    /// Override for the `config.json` path. `None` in production → the real
    /// [`crate::config::config_path`] (`~/.higgs` or `$HIGGS_HOME`). Set to a unique temp
    /// path for in-crate unit tests (see [`default_config_path_override`]) so `load`'s
    /// per-model-record PERSISTENCE never writes the developer/CI `~/.higgs/config.json`
    /// and parallel tests don't share state. A `Mutex` only to satisfy `Sync` cheaply; it is
    /// set once at construction and read thereafter.
    config_path: parking_lot::Mutex<Option<std::path::PathBuf>>,
    /// Serializes runtime API-key mutations (mint/revoke). Held across the whole
    /// clone-live → mutate → save → re-install so two concurrent mutations can't
    /// each derive from the same live snapshot and have the last save clobber the
    /// other's change. A plain `parking_lot::Mutex` — held only across synchronous
    /// file I/O, never an `.await`. Mirrors [`Self::models_io`]/[`Self::config_io`].
    keys_io: parking_lot::Mutex<()>,
    /// In-memory last-touch stamp per key digest (unix-ms) — the throttle
    /// authority for [`Self::touch_api_key`], so the hot auth path does at most
    /// one live-store update per key per minute instead of one on every
    /// request. Bounded by the number of distinct key digests seen this process
    /// (small — a single-user server). A plain `parking_lot::Mutex` — the
    /// throttle check-and-set is atomic under it (so a concurrent burst at
    /// window expiry dedupes to ONE update), never held across an `.await`.
    key_touch_throttle: parking_lot::Mutex<std::collections::HashMap<String, u64>>,
    /// Every CURRENTLY-LIVE `/v1` listener, in registration order. `serve_v1` is
    /// public and an embedder may run SEVERAL listeners on one facade, so this is
    /// a per-listener registry, not a set of flat slots that overlapping serves
    /// would clobber: each entry owns the address it bound and the exact extra CORS
    /// origins ITS layer was built with.
    ///
    /// Entries are added by [`Higgs::register_serve`] and removed when the returned
    /// [`ServeGuard`] drops — which happens on graceful shutdown, on a serve error,
    /// AND on task cancellation (an aborted serve future), so an aborted listener
    /// can never strand its registration.
    ///
    /// Reads: [`Self::lan_exposed`] (any live LAN listener → the [HG059] revoke
    /// gate), [`Self::bound_addr`] and [`Self::applied_cors_origins`] (the PRIMARY —
    /// first-registered — live listener's disclosures), and
    /// [`Self::any_live_serve_cors_differs`] (does ANY live listener run a CORS list
    /// other than the persisted one → `restart_required`).
    serves: parking_lot::Mutex<Vec<ServeSlot>>,
    /// Monotonic id source for [`ServeSlot`] — a slot is removed by id, never by
    /// index, so a sibling deregistration can't shift another's identity.
    next_serve_id: std::sync::atomic::AtomicU64,
    /// Serializes a listener's REGISTRATION against another listener's
    /// DEREGISTRATION-and-teardown. `ServeGuard::release` and the `Higgs::stop`
    /// that may follow it are two steps: without this an incoming `serve_v1` could
    /// register in the gap, and the departing "last" listener would then drain the
    /// shared node under it — `stop()` sets a TERMINAL `shutting_down` flag, so the
    /// newcomer would serve a permanently dead facade. An ASYNC mutex: a
    /// `parking_lot` one cannot be held across `stop().await`.
    serve_lifecycle: tokio::sync::Mutex<()>,
    /// A manual LAN-exposure override for an embedder that serves `/v1` through
    /// its OWN stack rather than [`crate::serve::serve_v1`] (which registers its
    /// listeners above). ORed into [`Self::lan_exposed`].
    lan_override: std::sync::atomic::AtomicBool,
    /// Model ids currently being BENCHMARKED (Turbotune). A benchmark refuses to
    /// start if the model is loaded ([HG067]); while it runs, its id sits here so a
    /// concurrent public load/JIT-chat is refused ([HG068]) rather than racing the
    /// bench for the model. Set/cleared by a [`BenchmarkingGuard`]; the bench's OWN
    /// candidate loads use the PRIVATE `local.load` so they bypass this gate.
    benchmarking: Arc<parking_lot::Mutex<std::collections::HashSet<String>>>,
    /// Set once by [`stop`](Self::stop) (terminal shutdown), read by the Turbotune
    /// benchmark's per-candidate cancel hook so a shutdown that races an in-flight
    /// benchmark aborts it cleanly with [HG064] instead of letting it iterate
    /// candidates against a tearing-down node (surfacing confusing HG063/worker
    /// errors). In the normal graceful path the benchmark request DRAINS before
    /// `stop` runs, so this is defense-in-depth (e.g. a future shutdown drain
    /// deadline on a long benchmark) — never a hot path.
    shutting_down: std::sync::atomic::AtomicBool,
}

/// Resolve the effective per-request sampling for a local chat: overlay the
/// `request` sampler set onto the model's `stored` tuned/card-recommended base.
///
/// This is the "HF-card recommended sampling actually applies" seam. A `tune`
/// persists the recommendation (temp/top_k/top_p/min_p/penalties) under the raw
/// model id; a plain OpenAI chat usually pins only `temperature`, so the rest of
/// the recommendation flows through ([`LlamaCppSamplingParams::overlaid_with`]:
/// request fields win, the base's other samplers survive). With no stored profile
/// the request stands alone (the worker still applies its 0.7 temperature default
/// for an unset field). Engine-tagged throughout — the umbrella variant matches.
fn overlay_sampling(
    stored: Option<crate::worker::engine::SamplingParams>,
    request: crate::worker::engine::SamplingParams,
) -> crate::worker::engine::SamplingParams {
    match stored {
        Some(base) => crate::worker::engine::SamplingParams::llamacpp(
            base.as_llamacpp().overlaid_with(request.as_llamacpp()),
        ),
        None => request,
    }
}

/// One CURRENTLY-LIVE `/v1` listener's serve-scoped state (see [`Higgs::serves`]).
/// Per-listener, not a flat facade slot: overlapping `serve_v1` calls each own their
/// bound address and the exact extra CORS origins THEIR layer was built with, so one
/// serve's start or exit can never rewrite another's disclosures.
pub(crate) struct ServeSlot {
    /// Identity for removal — a sibling deregistration must not shift it.
    id: u64,
    /// This listener is bound BEYOND loopback (LAN/`0.0.0.0`) → it contributes the
    /// [HG059] last-key-revoke refusal for as long as it lives.
    lan: bool,
    /// The address this listener actually bound (`None` if unknowable).
    addr: Option<std::net::SocketAddr>,
    /// The extra CORS origins THIS listener's layer was built with.
    cors_origins: Vec<String>,
}

/// RAII registration for a live `/v1` listener, from [`Higgs::register_serve`].
///
/// Dropping it deregisters the listener. Drop — not an explicit call after the
/// serve `.await` — because a serve future may be CANCELLED (an embedder aborting
/// the task): cancelled futures never run code past their await point, but they DO
/// run destructors. An explicit deregistration would leak the registration on abort,
/// stranding `lan_exposed` at `true` and permanently refusing a legitimate last-key
/// revoke with [HG059].
pub(crate) struct ServeGuard {
    higgs: Arc<Higgs>,
    id: u64,
    /// Already deregistered by [`Self::release`] — `Drop` must not do it twice.
    released: bool,
}

impl ServeGuard {
    /// Arm this listener's LAN exposure (see [`Higgs::arm_lan_serve`], the only
    /// caller — it does so under the keystore lock).
    fn set_lan(&self) {
        if let Some(slot) = self
            .higgs
            .serves
            .lock()
            .iter_mut()
            .find(|s| s.id == self.id)
        {
            slot.lan = true;
        }
    }

    /// Record the extra CORS origins this listener's layer is being built with,
    /// once `serve_v1`'s startup guards have passed and the list has been read.
    pub(crate) fn set_cors_origins(&self, cors_origins: Vec<String>) {
        if let Some(slot) = self
            .higgs
            .serves
            .lock()
            .iter_mut()
            .find(|s| s.id == self.id)
        {
            slot.cors_origins = cors_origins;
        }
    }

    /// Deregister this listener NOW, returning whether it was the LAST live one.
    ///
    /// `serve_v1` calls this the moment its listener stops accepting — BEFORE the
    /// terminal worker drain — so a dead listener never keeps disclosing itself
    /// (`bind_host`, applied CORS) or forcing the [HG059] revoke refusal while
    /// workers take seconds to shut down. The returned flag is what tells `serve_v1`
    /// whether it owns the facade teardown: with SIBLING listeners still live,
    /// draining the shared node would strand them serving a stopped facade.
    ///
    /// Computed under the registry lock, so exactly one of several concurrently
    /// exiting listeners can observe itself as last.
    pub(crate) fn release(mut self) -> bool {
        self.released = true;
        self.higgs.deregister_serve(self.id)
    }
}

impl Drop for ServeGuard {
    fn drop(&mut self) {
        // The CANCELLATION path (an aborted serve future): `release` never ran, so
        // the registration must still be dropped here.
        if !self.released {
            self.higgs.deregister_serve(self.id);
        }
    }
}

/// RAII guard marking a model as BENCHMARKING for its scope, WITH cancellation-safe
/// candidate cleanup. `id` sits in [`Higgs::benchmarking`] while alive; `live` holds
/// the CURRENT candidate's [`WorkerId`] (Some only between a candidate's load and its
/// post-measure unload).
///
/// Drop semantics (codex r15 + r16):
/// - Normal exit (slot `None`): the [HG068] flag is cleared synchronously — no worker
///   is resident, nothing to reclaim.
/// - CANCEL-drop with a live candidate (long-op route timeout / client disconnect):
///   the flag must OUTLIVE the reclaim — clearing it first would open a window where
///   the doomed candidate is still registered but no longer HG068-gated, so a public
///   chat/load could adopt it just before the unload kills it (codex r16). The drop
///   spawns ONE task that unloads exactly that worker and THEN clears the flag.
///   Worker-id-scoped, so it can never stomp a worker a user loads later.
pub(crate) struct BenchmarkingGuard {
    benchmarking: Arc<parking_lot::Mutex<std::collections::HashSet<String>>>,
    local: Arc<NodeRuntime>,
    id: String,
    live: Arc<parking_lot::Mutex<Option<WorkerId>>>,
}

impl Drop for BenchmarkingGuard {
    fn drop(&mut self) {
        if let Some(worker) = self.live.lock().take() {
            let benchmarking = Arc::clone(&self.benchmarking);
            let local = Arc::clone(&self.local);
            let id = std::mem::take(&mut self.id);
            tokio::spawn(async move {
                let _ = local.unload(worker).await;
                benchmarking.lock().remove(&id);
            });
        } else {
            self.benchmarking.lock().remove(&self.id);
        }
    }
}

/// Drive a Turbotune (G6) benchmark over an ORDERED candidate set, measuring
/// each with an injected async `measure` (the ONLY dependency on the real
/// load/generate path — production loads+times+unloads; tests pass a fake).
/// `cancelled` is polled between candidates so a terminal shutdown (`stop`)
/// aborts the run ([HG064]) rather than iterating candidates against a draining
/// node. Returns the winning candidate + its measurement, or an aggregate
/// `[HG063]` when every candidate fails.
///
/// This is the orchestration seam kept separate from `Higgs` so the
/// loop/cancel/benchmarked/aggregate logic is unit-tested without a real FFI load.
async fn run_benchmark<M, Fut>(
    candidates: Vec<crate::tune::bench::Candidate>,
    mut measure: M,
    mut cancelled: impl FnMut() -> bool,
) -> Result<(crate::tune::bench::Candidate, crate::tune::BenchResult), HiggsError>
where
    M: FnMut(&crate::tune::bench::Candidate) -> Fut,
    Fut: std::future::Future<Output = Result<crate::tune::BenchResult, HiggsError>>,
{
    use crate::tune::bench::{aggregate_failure, pick_benchmarked};
    let mut results: Vec<(crate::tune::bench::Candidate, crate::tune::BenchResult)> = Vec::new();
    let mut failures: Vec<(&'static str, String)> = Vec::new();

    for cand in candidates {
        // A terminal shutdown aborts the bench rather than loading the next
        // candidate against a node that is being torn down.
        if cancelled() {
            return Err(HiggsError::BenchCancelled);
        }
        match measure(&cand).await {
            // A 0 tok/s "success" (immediate EOS / empty decode window) measured
            // NOTHING — treating it as a result would let `pick_benchmarked` crown the
            // first candidate by ORDERING and persist a Bench profile that claims a
            // measurement (Fable r8). It is a candidate failure like any other.
            Ok(bench) if bench.gen_tps > 0.0 => results.push((cand, bench)),
            Ok(_) => {
                tracing::warn!(
                    candidate = cand.label,
                    "[HG065] benchmark candidate produced no measurable decode (0 tok/s) — trying the next config"
                );
                failures.push((cand.label, "no measurable decode (0 tok/s)".to_owned()));
            }
            Err(HiggsError::BenchCancelled) => return Err(HiggsError::BenchCancelled),
            Err(e) => {
                tracing::warn!(
                    candidate = cand.label,
                    error = %e,
                    "[HG065] benchmark candidate failed — trying the next config"
                );
                failures.push((cand.label, e.to_string()));
            }
        }
    }

    match pick_benchmarked(&results) {
        Some(i) => Ok(results.swap_remove(i)),
        None => Err(HiggsError::BenchExhausted {
            detail: aggregate_failure(&failures),
        }),
    }
}

/// Walk the G5 OOM degrade-retry ladder over an injected `load` (the ONLY
/// dependency; production passes `self.local.load`, tests pass a fake). Attempt
/// 0 uses the caller's `np` unchanged — the fits-first-time path is one call,
/// no ladder built. ONLY an out-of-memory failure
/// ([`is_oom_reason`](crate::load_robustness::is_oom_reason)) walks the ladder;
/// each rung is announced `[HG061]`. A non-OOM failure returns immediately; a
/// non-OOM failure surfacing on a degraded rung stops the ladder and returns.
/// Exhausting every rung yields the aggregate `[HG060]`. `settle` is slept
/// before each degraded retry so VRAM from the failed attempt drains first
/// (tests inject `Duration::ZERO`); `layer_count` lets the ladder turn a default
/// `GpuLayers::All` OOM into a concrete half-offload rung.
async fn run_oom_ladder<F, Fut>(
    id: &str,
    np: crate::remote::NodeLoadParams,
    layer_count: Option<u32>,
    settle: std::time::Duration,
    mut load: F,
) -> Result<crate::remote::NodeLoadParams, HiggsError>
where
    F: FnMut(crate::remote::NodeLoadParams) -> Fut,
    Fut: std::future::Future<Output = Result<(), HiggsError>>,
{
    use crate::load_robustness::{is_oom_reason, oom_ladder};

    // Extract an OOM reason from a load error, or `None` if it isn't OOM
    // (corrupt GGUF, unsupported arch — a degraded retry can't help).
    fn oom_reason(e: &HiggsError) -> Option<String> {
        match e {
            HiggsError::EngineLoadFailed { reason, .. } if is_oom_reason(reason) => {
                Some(reason.clone())
            }
            HiggsError::WorkerRpc {
                message,
                worker_code: Some(c),
                ..
            } if c == "HG004" && is_oom_reason(message) => Some(message.clone()),
            _ => None,
        }
    }

    // Attempt 0: the caller's params, unchanged.
    let mut last = match load(np.clone()).await {
        Ok(()) => return Ok(np), // loaded as-requested — persist these params
        Err(e) => match oom_reason(&e) {
            Some(r) => r,
            None => return Err(e), // non-OOM: no retry
        },
    };

    // Degrade ladder: plain retry → KV to system memory → fewer GPU layers.
    let rungs = oom_ladder(&np, layer_count);
    let total = rungs.len();
    for (i, rung) in rungs.into_iter().enumerate() {
        tracing::warn!(
            id,
            attempt = i + 2, // attempt 0 + this rung (1-based for humans)
            degrade = if rung.what.is_empty() {
                "plain retry"
            } else {
                rung.what
            },
            "[HG061] load OOM — retrying with a cheaper config"
        );
        // Let VRAM from the failed attempt drain before retrying (mechanism #6).
        tokio::time::sleep(settle).await;
        match load(rung.params.clone()).await {
            // This degraded rung loaded — return ITS params so the caller persists
            // the config that actually runs, not the seed that already OOMed.
            Ok(()) => return Ok(rung.params),
            Err(e) => match oom_reason(&e) {
                Some(r) => last = r,   // still OOM — next rung
                None => return Err(e), // a different fault appeared — surface it
            },
        }
    }

    // Every rung OOMed — the aggregate [HG060] with the attempt count + last reason.
    Err(HiggsError::LoadOomExhausted {
        attempts: total + 1,
        last,
    })
}

/// Map a host [`LoadParams`] to the node's `NodeLoadParams` for an EXACT load —
/// the `Some(params)` case of [`Higgs::load_inner`](Higgs::load_inner_impl). The
/// Turbotune bench loads candidates through this + `self.local.load` directly, so a
/// candidate is measured VERBATIM: no OOM-ladder degrade (the candidate set is
/// itself the degrade ladder) and no `config.json` persist (the bench saves only
/// its winning profile, once, at the end).
fn node_params_for(id: &str, p: &LoadParams) -> crate::remote::NodeLoadParams {
    crate::remote::NodeLoadParams {
        id: id.to_owned(),
        ctx_len: p.ctx_len().fixed_n(),
        gpu_layers: Some(p.gpu_layers()),
        threads: Some(p.threads()),
        params: {
            let lc = p.as_llamacpp();
            lc.has_overrides().then(|| lc.clone())
        },
    }
}

/// Wall-clock milliseconds since the Unix epoch (`0` if the clock is before it). Used to stamp
/// per-model load records in `config.json`.
fn now_unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// The default `config.json` path override for a freshly-built [`Higgs`].
///
/// Production (and integration tests, which run the real binary under an isolated `HIGGS_HOME`)
/// get `None` → the real [`crate::config::config_path`]. In-crate unit tests (`cfg(test)`) get a
/// UNIQUE temp path per instance, so `Higgs::load`'s config persistence is hermetic (never
/// touches the real `~/.higgs/config.json`) and parallel-safe (no two instances share a file).
#[cfg(not(test))]
fn default_config_path_override() -> Option<std::path::PathBuf> {
    None
}
#[cfg(test)]
fn default_config_path_override() -> Option<std::path::PathBuf> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    // A unique per-instance SUBDIR (not just a unique file in the shared temp dir):
    // `models_store()` derives its home from this path's PARENT, so `config.json`
    // AND `models.json` must each live in an isolated dir or parallel unit tests
    // would clobber each other's per-node model store.
    Some(
        std::env::temp_dir()
            .join(format!("higgs-unit-{}-{n}", std::process::id()))
            .join("config.json"),
    )
}

impl Higgs {
    /// Construct the facade WITHOUT spawning the llama.cpp worker, owning a fresh
    /// [`LogBus`]. The serve-layer [`HiggsLogLayer`](crate::log_bus::HiggsLogLayer)
    /// is NOT wired to this internal bus — request events won't appear in the
    /// Developer Logs. Use [`with_log_bus`](Self::with_log_bus) when the caller
    /// installs the tracing layer.
    ///
    /// Call [`start`](Self::start) when the host is ready.
    ///
    /// RUNTIME: must be called from WITHIN a Tokio runtime — see
    /// [`with_log_bus`](Self::with_log_bus).
    pub fn new(config: HiggsConfig) -> Self {
        Self::with_log_bus(config, Arc::new(LogBus::new()))
    }

    /// Construct the facade WITHOUT spawning the llama.cpp worker, sharing `bus` with
    /// the caller's serve-layer [`HiggsLogLayer`](crate::log_bus::HiggsLogLayer).
    ///
    /// Worker stderr and captured serve-layer request events then flow into the
    /// same Developer-Log history+stream. The caller is responsible for
    /// installing `HiggsLogLayer::new(bus.clone())` on its tracing subscriber.
    ///
    /// RUNTIME: this is a synchronous constructor but it MUST be called from within a
    /// Tokio runtime. It builds the co-located local [`NodeRuntime`], which spawns its
    /// actor task + idle reaper immediately (via `tokio::spawn`) so the facade is usable
    /// the moment it returns — `load`/`status`/chat need the actor live. "Without
    /// spawning the worker" refers ONLY to the heavy llama.cpp subprocess (deferred to
    /// [`load`](Self::load)); the lightweight node tasks DO start here, so constructing
    /// outside a runtime panics with Tokio's "no reactor running". Every real caller
    /// (the standalone server, the embedded host) already constructs inside its async
    /// bring-up, so this is a contract note, not a behavior change.
    pub fn with_log_bus(config: HiggsConfig, bus: Arc<LogBus>) -> Self {
        // Build the co-located LOCAL node from the same config + bus. The node owns the
        // workers; it shares `bus` so local worker stderr lands in the Developer Logs.
        let node_config = NodeConfig {
            bus,
            lmstudio_dirs: config.lmstudio_dirs.clone(),
            hf_dirs: config.hf_dirs.clone(),
            ollama_dirs: config.ollama_dirs.clone(),
            idle_ttl: DEFAULT_IDLE_TTL,
        };
        // Thread the worker-exe DI seam ([`HiggsConfig::worker_exe`]) into the node's
        // Supervisor spawner. `None` ⇒ `Arc::new(Supervisor::spawn)` — EXACTLY what
        // `NodeRuntime::new` installs, so the production path is byte-identical when the
        // seam is unused. `Some(exe)` ⇒ each worker (re)spawns from `exe` instead of
        // `current_exe()`, for a host whose own binary can't answer `--higgs-worker`.
        let spawner: crate::node::runtime::SupervisorSpawner = match config.worker_exe.clone() {
            Some(exe) => {
                Arc::new(move |bus| crate::supervisor::Supervisor::spawn_with_exe(bus, exe.clone()))
            }
            None => Arc::new(crate::supervisor::Supervisor::spawn),
        };
        Self::with_local(
            Arc::new(NodeRuntime::with_spawner(node_config, spawner)),
            config,
        )
    }

    /// Construct the facade around an already-built local [`NodeRuntime`] — the single home for
    /// the facade's default field initialization. Production goes through [`with_log_bus`];
    /// tests inject a fake-worker-backed `NodeRuntime` here.
    pub(crate) fn with_local(local: Arc<NodeRuntime>, config: HiggsConfig) -> Self {
        Self {
            local,
            config: parking_lot::Mutex::new(config),
            lifecycle: tokio::sync::Mutex::new(()),
            inference_gate: Arc::new(tokio::sync::Semaphore::new(MAX_CONCURRENT_INFERENCE)),
            remote_gate: Arc::new(tokio::sync::Semaphore::new(MAX_CONCURRENT_INFERENCE)),
            log_incoming_tokens: std::sync::atomic::AtomicBool::new(false),
            jit_enabled: std::sync::atomic::AtomicBool::new(true),
            // Mirror the node's idle defaults (auto-unload on, 60-min TTL) so the
            // Server-Settings getters read the same values the node reaper enforces.
            auto_unload_idle: std::sync::atomic::AtomicBool::new(true),
            idle_ttl_minutes: std::sync::atomic::AtomicU64::new(DEFAULT_IDLE_TTL.as_secs() / 60),
            serving_enabled: std::sync::atomic::AtomicBool::new(true),
            device_cache: parking_lot::Mutex::new(None),
            estimate_meta_cache: parking_lot::Mutex::new(None),
            fleet: parking_lot::Mutex::new(None),
            api_keys: parking_lot::Mutex::new(std::sync::Arc::new(crate::keys::ApiKeys::default())),
            hub: parking_lot::Mutex::new(None),
            hub_lifecycle: tokio::sync::Mutex::new(()),
            config_io: parking_lot::Mutex::new(()),
            models_io: parking_lot::Mutex::new(()),
            loading: parking_lot::Mutex::new(None),
            // Capacity mirrors the log bus. Load events are low-rate (~5 per load),
            // so 256 makes a lagging subscriber effectively impossible: it would take
            // ~50 loads interleaved faster than a subscriber polls. A lagged SSE
            // client skips the gap and keeps streaming.
            load_events: tokio::sync::broadcast::channel(256).0,
            config_path: parking_lot::Mutex::new(default_config_path_override()),
            keys_io: parking_lot::Mutex::new(()),
            key_touch_throttle: parking_lot::Mutex::new(std::collections::HashMap::new()),
            serves: parking_lot::Mutex::new(Vec::new()),
            next_serve_id: std::sync::atomic::AtomicU64::new(0),
            serve_lifecycle: tokio::sync::Mutex::new(()),
            lan_override: std::sync::atomic::AtomicBool::new(false),
            benchmarking: Arc::new(parking_lot::Mutex::new(std::collections::HashSet::new())),
            shutting_down: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// The `config.json` path this instance reads/writes: the per-instance override when set
    /// (unit tests), else the real [`crate::config::config_path`].
    fn config_file_path(&self) -> std::io::Result<std::path::PathBuf> {
        match self.config_path.lock().clone() {
            Some(p) => Ok(p),
            None => crate::config::config_path(),
        }
    }

    /// This instance's `config.json` `name` — the local node's operator label, shown first in
    /// `GET /api/higgs/nodes`. Read through the SAME per-instance seam ([`Self::config_file_path`])
    /// that [`Self::with_config_mut`] writes, so a `POST /api/higgs/nodes/label {node:"local"}`
    /// rename round-trips. (In production the seam resolves to the global `~/.higgs/config.json`;
    /// only in unit tests is it a per-instance override — reading the global path directly there
    /// meant the view never saw the write.) Empty/absent → `None`.
    pub(crate) fn instance_name(&self) -> Option<String> {
        self.config_file_path()
            .ok()
            .and_then(|p| crate::config::InstanceConfig::load(&p).ok())
            .map(|c| c.name)
            .filter(|n| !n.is_empty())
    }

    /// Serialize a read-modify-write of `config.json` under [`Self::config_io`]: load the current
    /// config, apply `f`, then persist atomically — so concurrent load-record / rename / autoload
    /// writes can't clobber each other. The closure runs while the lock is held; it must not block
    /// or `.await` (it's a quick field mutation). A missing file loads as the default config.
    pub(crate) fn with_config_mut<R>(
        &self,
        f: impl FnOnce(&mut crate::config::InstanceConfig) -> R,
    ) -> std::io::Result<R> {
        let _guard = self.config_io.lock();
        let path = self.config_file_path()?;
        let mut cfg = crate::config::InstanceConfig::load(&path)?;
        let r = f(&mut cfg);
        cfg.save(&path)?;
        Ok(r)
    }

    /// The per-model load records persisted in this instance's `config.json` (read-only — no lock
    /// needed; the atomic save means a reader always sees a complete version). Empty on any error
    /// or absent file. Honors the per-instance config-path override so reads match writes.
    pub(crate) fn model_records(
        &self,
    ) -> std::collections::BTreeMap<String, crate::config::ModelRecord> {
        self.config_file_path()
            .ok()
            .and_then(|p| crate::config::InstanceConfig::load(&p).ok())
            .map(|c| c.models)
            .unwrap_or_default()
    }

    /// Every saved tuning record (from `models.json`), keyed by model id. Loaded
    /// ONCE per `/api/higgs/models` pass and threaded into `model_readiness` so
    /// the listing doesn't reopen + reparse the store per model.
    ///
    /// A store-OPEN failure (models.json exists but is unreadable — a directory,
    /// bad perms) is surfaced as `PersistenceFailed`, NOT collapsed to "no
    /// profiles": otherwise the listing would badge every prepared model
    /// `discovered` exactly in the persistence-failure case, contradicting the JIT
    /// path's HG040 and telling the user to Prepare again (which would also fail).
    /// A MISSING store is not an error — `models_store()` returns an empty store,
    /// so a fresh node lists fine.
    pub(crate) fn tuning_records(
        &self,
    ) -> Result<std::collections::BTreeMap<String, crate::tune::store::TuneRecord>, HiggsError>
    {
        let store = self
            .models_store()
            .map_err(|e| HiggsError::PersistenceFailed {
                store: "models".into(),
                path: "models.json".into(),
                source: e,
            })?;
        Ok(store.all_tuning())
    }

    /// Every model's `(active, analytical, bench)` tuning triple — the models
    /// list fills the dual "Tuned"/"Benchmarked" wire fields from this. Same
    /// error posture as [`Self::tuning_records`].
    #[allow(clippy::type_complexity)]
    pub(crate) fn tuning_profiles(
        &self,
    ) -> Result<
        std::collections::BTreeMap<
            String,
            (
                Option<crate::tune::store::TuneRecord>,
                Option<crate::tune::store::TuneRecord>,
                Option<crate::tune::store::TuneRecord>,
            ),
        >,
        HiggsError,
    > {
        let store = self
            .models_store()
            .map_err(|e| HiggsError::PersistenceFailed {
                store: "models".into(),
                path: "models.json".into(),
                source: e,
            })?;
        Ok(store.all_tuning_profiles())
    }

    /// Install the remote fleet (hub mode) — `/v1` chat for a remote-resident model then
    /// routes through it. Idempotent; replaces any prior fleet.
    pub fn set_fleet(&self, fleet: Arc<crate::node::fleet::HubFleet>) {
        *self.fleet.lock() = Some(fleet);
    }

    /// The installed remote fleet, if any (hub mode).
    pub fn fleet(&self) -> Option<Arc<crate::node::fleet::HubFleet>> {
        self.fleet.lock().clone()
    }

    /// Install/replace the API-key store gating the HTTP surface (P5). An empty store
    /// disables auth. The standalone binary calls this with the loaded `api_keys.json`.
    pub fn set_api_keys(&self, keys: Arc<crate::keys::ApiKeys>) {
        *self.api_keys.lock() = keys;
    }

    /// A snapshot handle to the current API-key store (cheap `Arc` clone) — read by the
    /// serve-layer auth middleware on each request.
    pub fn api_keys(&self) -> Arc<crate::keys::ApiKeys> {
        self.api_keys.lock().clone()
    }

    /// Atomically read-modify-write the API-key store under [`Self::keys_io`], then
    /// PERSIST to `api_keys.json` and re-install the live store. `f` sees a mutable
    /// clone of the LIVE keys and returns any decision value; the whole thing is
    /// serialized so a concurrent mint/revoke can't lose an update.
    ///
    /// EMBEDDED-HOST BOUNDARY (jigglebot): the embed never loads
    /// `api_keys.json` at boot (its keystore starts empty — "embedded host
    /// wants no gate"), so keys minted here are live for THIS process and
    /// persisted to disk, but do not re-arm auth on the next embedded start.
    /// That is deliberate: the jigglebot frontend does not attach bearers yet
    /// (G4b), and a silently re-armed keystore would 401 its own UI. The
    /// standalone binary loads the file at startup as before. Consistently,
    /// mutations act on the LIVE store (cloned below), never on the file —
    /// stale on-disk keys are never resurrected mid-session; the file is
    /// overwritten with the live-derived state.
    pub fn mutate_api_keys<T>(
        &self,
        f: impl FnOnce(&mut crate::keys::ApiKeys) -> T,
    ) -> std::io::Result<T> {
        let _guard = self.keys_io.lock();
        // The LIVE keystore is authoritative — never reload the file here. An
        // embedded host deliberately starts with an empty live store even when
        // a stale `api_keys.json` exists (standalone/CLI leftovers); loading
        // the file would RESURRECT those keys on any mutation (even a failed
        // duplicate mint) and lock out a UI that holds no bearer. Mutations
        // act on what `GET /api/higgs/keys` shows, and the file is synced TO
        // the live state.
        let before = self.api_keys();
        let mut keys = (*before).clone();
        let out = f(&mut keys);
        // Persist + re-install ONLY when `f` actually changed the store. A REJECTED
        // request (unauthorized mint, duplicate label, unknown revoke, last-key-on-LAN
        // / last-Admin conflict) leaves the clone identical to the live store: persisting
        // it would do a pointless disk write AND, on an unwritable keystore, turn the
        // intended 401/400/409 into an HG040 500 (codex r8). The decision `out` — which
        // carries the rejection the caller maps to its status — is returned regardless.
        if keys != *before {
            keys.save(&crate::keys::keys_path()?)?;
            self.set_api_keys(std::sync::Arc::new(keys));
        }
        Ok(out)
    }

    /// Record a successful authorization on the key with `sha256` — updates its
    /// `last_used_ms` on the LIVE store (in memory only), THROTTLED to once per
    /// key per minute so the hot auth path doesn't turn into a per-request
    /// update.
    ///
    /// The throttle authority is the in-memory [`Self::key_touch_throttle`] map:
    /// the check-and-set is atomic under that map's lock, so a concurrent burst
    /// at window expiry dedupes to a single update (only the first request past
    /// the atomic check proceeds). [`crate::keys::ApiKeys::touch`] is monotonic,
    /// so even a reordered stale `now` can't move `last_used_ms` backward.
    ///
    /// The update is IN-MEMORY ONLY — it never rewrites `api_keys.json`. A
    /// touch-driven full-file rewrite would clobber keys added out-of-band: the
    /// `higgs keys add` CLI writes the keystore directly and blesses a deferred
    /// restart, so passive request traffic rewriting the file from the live
    /// store (which does not yet know those keys) would silently DESTROY a
    /// just-minted CLI key before the operator's restart. `GET /api/higgs/keys`
    /// reads the live store, so the fresh `last_used_ms` is visible this
    /// session; a touch itself never writes the file. The durable record is
    /// key IDENTITY; usage stamps reach disk only as a side-effect of the next
    /// explicit mint/revoke (which persists the live store wholesale) — so a
    /// restart shows usage as of that last mutation, else "never".
    pub fn touch_api_key(&self, sha256: &str) {
        const THROTTLE_MS: u64 = 60_000;
        let now = now_unix_ms();
        // Atomic throttle check-and-set: within the window → nothing to do; else
        // claim this window BEFORE releasing the lock so a racing request sees
        // the fresh stamp and returns.
        {
            let mut throttle = self.key_touch_throttle.lock();
            if throttle
                .get(sha256)
                .is_some_and(|t| now.saturating_sub(*t) < THROTTLE_MS)
            {
                return;
            }
            throttle.insert(sha256.to_owned(), now);
        }
        // Update last-used on the LIVE store in memory only (no file write).
        // Serialized against mint/revoke by `keys_io` so a concurrent mutation
        // can't lose this update or vice-versa. A plain lock, no `.await`.
        let _guard = self.keys_io.lock();
        let before = self.api_keys();
        let mut keys = (*before).clone();
        if keys.touch(sha256, now) {
            self.set_api_keys(std::sync::Arc::new(keys));
        }
    }

    /// Force the LAN-exposure override (see the `lan_override` field docs) — for an
    /// embedder that serves `/v1` through its OWN stack rather than
    /// [`crate::serve::serve_v1`], which registers its listeners itself. ORed with
    /// the live-listener registry, so it can only ADD exposure, never mask a live
    /// LAN listener's [HG059] protection.
    pub fn set_lan_exposed(&self, exposed: bool) {
        self.lan_override
            .store(exposed, std::sync::atomic::Ordering::Relaxed);
    }

    /// Whether ANY live listener is bound beyond loopback (or the override is set)
    /// — the [HG059] revoke gate.
    pub(crate) fn lan_exposed(&self) -> bool {
        self.lan_override.load(std::sync::atomic::Ordering::Relaxed)
            || self.serves.lock().iter().any(|s| s.lan)
    }

    /// Register a `/v1` listener, UNARMED — called by [`crate::serve::serve_v1`]
    /// BEFORE its [HG058]/[HG069] guards run, so a refusal can decide "was I the sole
    /// serve on this facade?" atomically (via [`ServeGuard::release`], under the
    /// registry lock) instead of racing a sibling registering in the gap.
    ///
    /// Only `addr` is final here. `lan` is armed afterwards by [`Higgs::arm_lan_serve`]
    /// — which does it under the KEYSTORE lock, together with those guards, so a
    /// concurrent revoke can never interleave into a keyless LAN surface — and the
    /// enforced CORS list is recorded by [`ServeGuard::set_cors_origins`] once it has
    /// been read. A caller that knows both up front (a test) may pass them directly.
    ///
    /// The returned [`ServeGuard`] deregisters on drop, so the listener's state is
    /// released on graceful shutdown, on a serve error, and on task CANCELLATION
    /// (an aborted serve future never runs code after its `.await`, but it does run
    /// destructors). Once no LAN listener remains, the [HG059] last-key-revoke
    /// refusal lifts; safety rests on the START-side guards, which re-check the
    /// keystore on every serve ([HG058] refuses a keyless non-loopback bind, [HG069]
    /// one whose keys are all non-Admin).
    pub(crate) fn register_serve(
        self: &std::sync::Arc<Self>,
        lan: bool,
        addr: Option<std::net::SocketAddr>,
        cors_origins: Vec<String>,
    ) -> ServeGuard {
        let id = self
            .next_serve_id
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.serves.lock().push(ServeSlot {
            id,
            lan,
            addr,
            cors_origins,
        });
        ServeGuard {
            higgs: std::sync::Arc::clone(self),
            id,
            released: false,
        }
    }

    /// Serialize a listener registration against another listener's
    /// deregistration-and-teardown (see the `serve_lifecycle` field docs). Held by
    /// `serve_v1` across `register_serve`, and across `release()` + `stop()`.
    pub(crate) async fn serve_lifecycle(&self) -> tokio::sync::MutexGuard<'_, ()> {
        self.serve_lifecycle.lock().await
    }

    /// Run the non-loopback startup key checks and ARM the listener's LAN exposure
    /// **atomically**, under the same `keys_io` lock that guards a key revoke.
    ///
    /// The two must not interleave. `revoke_key` decides [HG059] (refuse emptying
    /// the store while LAN-exposed) and commits the removal under `keys_io`; this
    /// checks [HG058] (a non-loopback listener needs at least one key) and [HG069]
    /// (…at least one Admin key) and then arms the exposure. Without a shared lock a
    /// revoke could observe `lan_exposed() == false`, a serve could pass its key
    /// check against the not-yet-published store, and the revoke would then empty it
    /// — leaving a listener KEYLESS on a LAN. Serialized, one always loses: either
    /// the revoke sees the armed listener and refuses, or this sees the emptied
    /// store and refuses.
    ///
    /// Lock order is `keys_io` → `serves`; nothing takes them the other way round.
    pub(crate) fn arm_lan_serve(&self, guard: &ServeGuard, bind: &str) -> Result<(), HiggsError> {
        let _keys = self.keys_io.lock();
        let keys = self.api_keys();
        // [HG058]: zero keys ⇒ auth is off AND the Host guard is relaxed for this
        // bind — the whole surface would be exposed.
        if keys.is_empty() {
            return Err(HiggsError::LanBindWithoutKeys {
                bind: bind.to_owned(),
            });
        }
        // [HG069]: keys exist but none is Admin ⇒ the Admin-only key-management API
        // (mint/revoke) is locked out on the exposed surface.
        if !keys
            .iter()
            .any(|k| k.scopes.contains(&crate::keys::Scope::Admin))
        {
            return Err(HiggsError::LanBindWithoutAdminKey {
                bind: bind.to_owned(),
            });
        }
        guard.set_lan();
        Ok(())
    }

    /// Drop a listener's registration BY ID (never by index — a sibling
    /// deregistration must not shift another's identity).
    fn deregister_serve(&self, id: u64) -> bool {
        let mut serves = self.serves.lock();
        serves.retain(|s| s.id != id);
        serves.is_empty()
    }

    /// The PRIMARY (first-registered) live listener's bound address, or `None`
    /// pre-serve. With several listeners the primary is disclosed — the single
    /// `bind_host` wire field can't describe more, and the common case is one.
    pub(crate) fn bound_addr(&self) -> Option<std::net::SocketAddr> {
        self.serves.lock().first().and_then(|s| s.addr)
    }

    /// The PRIMARY live listener's applied extra CORS origins, or `None` pre-serve
    /// (which [`Higgs::cors_settings`] distinguishes from an applied-but-empty list).
    pub(crate) fn applied_cors_origins(&self) -> Option<Vec<String>> {
        self.serves.lock().first().map(|s| s.cors_origins.clone())
    }

    /// Does ANY live listener run a CORS allowlist other than `persisted`? That —
    /// not just the primary's — is what "a restart is required to apply the
    /// persisted list" means when several listeners are up. Compared as SETS (the
    /// allowlist is exact-match membership; order is meaningless to the layer).
    /// `false` when nothing is live: the first serve start applies `persisted`.
    pub(crate) fn any_live_serve_cors_differs(&self, persisted: &[String]) -> bool {
        use std::collections::HashSet;
        let want: HashSet<&String> = persisted.iter().collect();
        self.serves
            .lock()
            .iter()
            .any(|s| s.cors_origins.iter().collect::<HashSet<_>>() != want)
    }

    /// Register an INTERNAL, in-memory-only bearer `token` for an in-process
    /// embedder (jigglebot) so it can authenticate ITSELF to the embedded higgs
    /// over the normal bearer path — no special auth branch.
    ///
    /// The embedder owns the token value (a fixed dev token shared with the
    /// browser proxy, or a fresh random one in production) and calls this at boot.
    /// The key is [`crate::keys::ApiKey::hidden`]: added to the LIVE keystore but
    /// NEVER persisted to `api_keys.json` and NEVER shown in the key-management
    /// list. Because it lands in the live store, auth is now ON for the embedded
    /// surface — the embedder presents this token, external apps present their own
    /// user tokens, and an unauthenticated caller is refused. In-memory only —
    /// deliberately does NOT persist (unlike [`Self::mutate_api_keys`]).
    pub fn register_internal_token(&self, token: &str, scopes: Vec<crate::keys::Scope>) {
        let _guard = self.keys_io.lock();
        let before = self.api_keys();
        let mut keys = (*before).clone();
        keys.add_internal(token, "jigglebot (internal)".to_string(), scopes);
        self.set_api_keys(std::sync::Arc::new(keys));
    }

    /// Whether `id` is currently being benchmarked — a public load / JIT chat must
    /// refuse it ([HG068]) rather than race the benchmark for the model. Read by
    /// `load_inner_impl` (the load gate) AND by the serve layer's `ensure_loaded`
    /// (the chat gate), so it is `pub(crate)`.
    pub(crate) fn is_benchmarking(&self, id: &str) -> bool {
        self.benchmarking.lock().contains(id)
    }

    /// Mark `id` as benchmarking and return the RAII guard that clears it on drop.
    /// The bench's own candidate loads go through the PRIVATE `local.load`, so they
    /// are not gated by this flag; only PUBLIC loads/JIT-chats are. `pub(crate)` so
    /// the serve-layer benchmarking-gate test can hold the guard deterministically.
    pub(crate) fn begin_benchmark(&self, id: &str) -> BenchmarkingGuard {
        self.benchmarking.lock().insert(id.to_owned());
        BenchmarkingGuard {
            benchmarking: Arc::clone(&self.benchmarking),
            local: Arc::clone(&self.local),
            id: id.to_owned(),
            live: Arc::new(parking_lot::Mutex::new(None)),
        }
    }

    /// Extra CORS origins from `config.json` (`cors_origins`), matched exactly
    /// against the request `Origin` in addition to the built-in loopback/tauri
    /// set. Read once at serve start — a change needs a restart (G7 owns live
    /// rebind). Errors read as empty (the built-in set still applies).
    ///
    /// Every entry is CANONICALIZED on read (and an invalid one dropped with a
    /// warning), exactly as [`Higgs::set_cors_origins`] canonicalizes on write.
    /// `config.json` is a plain file an operator can hand-edit, and it predates
    /// that validation — a legacy `https://EXAMPLE.com:443` entry would otherwise
    /// be built into the layer verbatim and never match a browser's
    /// `Origin: https://example.com`, while being disclosed as applied-and-in-sync.
    /// Normalizing here keeps what we ENFORCE, what we DISCLOSE, and what a browser
    /// SENDS the same string.
    pub(crate) fn extra_cors_origins(&self) -> Vec<String> {
        let raw: Vec<String> = self
            .config_file_path()
            .ok()
            .and_then(|p| crate::config::InstanceConfig::load(&p).ok())
            .map(|c| c.cors_origins)
            .unwrap_or_default();
        let mut seen = std::collections::HashSet::with_capacity(raw.len());
        let mut out = Vec::with_capacity(raw.len());
        for origin in raw {
            match crate::api::embed::validate_cors_origin(&origin) {
                Ok(canonical) => {
                    if seen.insert(canonical.clone()) {
                        out.push(canonical);
                    }
                }
                Err(e) => {
                    tracing::warn!("higgs: ignoring invalid cors_origins entry in config.json: {e}")
                }
            }
        }
        out
    }

    /// Run the Turbotune (G6) benchmark: generate an ordered, fit-and-headroom
    /// filtered candidate set from the analytical `suggestion`, LOAD + time each
    /// candidate through the real chat path, and return the winning params + its
    /// [`BenchResult`]. Each measure loads the candidate (via the PRIVATE node
    /// `local.load`, bypassing the [HG068] gate), runs a short real decode to time
    /// generation, and tears the worker down. The benchmark is EXCLUSIVE — concurrent
    /// public load/chat/unload/worker-stop are refused ([HG068]) — and a terminal
    /// `stop` (shutdown) aborts it ([HG064]); every candidate failing yields [HG063].
    async fn turbotune_bench(
        &self,
        id: &str,
        meta: &ModelMeta,
        hw: &crate::system::HardwareInfo,
        budget: &crate::tune::ResourceBudget,
        suggestion: &TuneSuggestion,
        pins: &crate::tune::TunePins,
    ) -> Result<
        (
            crate::worker::engine::llamacpp::params::LlamaCppParams,
            BenchResult,
        ),
        HiggsError,
    > {
        // Phase 1 (cheap): apply the user's pins onto the analytical seed AND
        // build the pin-aware, fit-and-headroom-filtered candidate set (no
        // loads) — one seam so the pins wiring is unit-testable. Pins overwrite
        // the seed (every candidate derives from it) and suppress the rungs that
        // would search a pinned dimension. The filter respects BOTH estimators:
        // candidates like the half-GPU-layers rung SHIFT memory into system RAM,
        // so a VRAM-only check would load, measure, and persist a config the RAM
        // estimator calls Overflow under an explicit RAM cap (codex r20). A
        // RAM-overflowing candidate returns the RAM report (verdict Overflow →
        // filtered by `passes_headroom`); otherwise the VRAM report drives the
        // fit + absolute-headroom gate as before.
        let candidates = crate::tune::bench::pinned_bench_candidates(
            suggestion.load.as_llamacpp(),
            meta.block_count,
            pins,
            // `bench_fit` normalizes an `Auto` context to the node's load-time cap
            // BEFORE estimating (RAM-overflow wins over VRAM), so a pinned/seed
            // `Auto` on a long-context model is judged against the window that
            // actually loads — not the full `ctx_train` — and a loadable candidate
            // is not falsely dropped as Overflow.
            |lc| crate::tune::vram::bench_fit(lc, meta, hw, budget),
        );

        // EXCLUSIVE benchmark contract: refuse if the model is LOADED ([HG067]) — a
        // bench loads/unloads candidate configs and must not disrupt a live worker —
        // and mark it benchmarking so concurrent public loads / JIT chats are refused
        // ([HG068]). The loaded-check AND the flag-set happen UNDER `lifecycle` (which
        // `load_inner_impl` also holds when it reads the flag), so a racing load can't
        // slip between them. The guard clears the flag on every exit (incl. drop).
        let _bench_guard = {
            let _lifecycle = self.lifecycle.lock().await;
            // A benchmark is already running for this id: refuse the second one ([HG068])
            // rather than let two benches interleave (between candidates `instances()` is
            // momentarily empty, so the loaded-check below can't catch a concurrent bench,
            // and their `bench_unload_id` calls would evict each other's candidate workers).
            if self.is_benchmarking(id) {
                return Err(HiggsError::BenchInProgress { id: id.to_owned() });
            }
            if self.local.instances().await.iter().any(|(_, m)| m == id) {
                return Err(HiggsError::BenchModelLoaded { id: id.to_owned() });
            }
            self.begin_benchmark(id)
        };

        // Phase 2 (slow): load + measure each candidate. The `_bench_guard` above
        // makes the model EXCLUSIVE to this benchmark — public loads / JIT chats for
        // it are refused ([HG068]) — so no concurrent op can touch it and the loop
        // needs NO cancellation or teardown-race handling. The load is a DIRECT node
        // load (not `load_inner`): no OOM-ladder degrade, no config persist — a
        // candidate is measured EXACTLY as specified (the candidate set is itself the
        // degrade ladder; an OOMing candidate just fails and the next is tried).
        //
        // CANCELLATION SAFETY (codex r15/r16): if this future is dropped
        // mid-candidate (long-op route timeout / client disconnect), the guard's drop
        // spawns an unload of exactly the live candidate worker and clears the [HG068]
        // flag only AFTER that unload commits — so the doomed candidate is never
        // registered-but-ungated. On every normal exit the slot is `None` (each
        // candidate is unloaded right after its measurement), so the drop no-ops.
        let live_candidate = Arc::clone(&_bench_guard.live);
        let benchmarked = run_benchmark(
            candidates,
            |cand| {
                let load = cand.load.clone();
                let live = Arc::clone(&live_candidate);
                async move {
                    // GPU settle before the load (mechanism #6, shared const).
                    tokio::time::sleep(crate::load_robustness::SETTLE_BEFORE_RETRY).await;
                    // Tear down this model's prior candidate worker, then load THIS
                    // candidate verbatim, measure it, and unload it. The live-candidate
                    // slot is Some ONLY between the load and the post-measure unload —
                    // exactly the window a dropped future would leak the worker.
                    self.bench_unload_id(id).await;
                    let (worker, _) = self
                        .local
                        .load(node_params_for(
                            id,
                            &crate::worker::engine::LoadParams::llamacpp(load),
                        ))
                        .await?;
                    *live.lock() = Some(worker);
                    let bench = self.measure_gen_tps(id).await;
                    self.bench_unload_id(id).await;
                    *live.lock() = None;
                    bench
                }
            },
            // The benchmarking gate keeps the model EXCLUSIVE against concurrent public
            // ops (load/chat/unload/worker-stop are refused [HG068]). The one op that is
            // NOT refused is a terminal `stop` (shutdown) — so the ONLY cancel signal is
            // the shutdown flag: abort with [HG064] rather than load more candidates
            // against a draining node. (Normally the benchmark request drains before
            // `stop` runs; this is the defense-in-depth path.)
            || {
                self.shutting_down
                    .load(std::sync::atomic::Ordering::Relaxed)
            },
        )
        .await?;
        Ok((benchmarked.0.load, benchmarked.1))
    }

    /// Unload every worker serving `id` (the bench's own model) via the PRIVATE node
    /// `local.unload` — the bench's teardown between candidates. SCOPED to `id` so a
    /// benchmark never evicts OTHER resident/serving models on a multi-model node;
    /// distinct from the public [`unload`](Self::unload), which is REFUSED ([HG068])
    /// while a benchmark owns a model.
    async fn bench_unload_id(&self, id: &str) {
        for (worker, model) in self.local.instances().await {
            if model == id {
                let _ = self.local.unload(worker).await;
            }
        }
    }

    /// Measure generation throughput for the currently-resident `id`: run a short
    /// real decode and time it. `gen_tps` is the DECODE-ONLY rate (prefill excluded
    /// via [`bench::bench_gen_tps`](crate::tune::bench::bench_gen_tps)); `ttft_ms` =
    /// time to the first delta (the prefill window). An honest measurement of the
    /// loaded config on this hardware.
    async fn measure_gen_tps(&self, id: &str) -> Result<BenchResult, HiggsError> {
        const BENCH_MAX_TOKENS: usize = 64;
        let messages = r#"[{"role":"user","content":"Write a detailed paragraph about the ocean and its tides."}]"#;
        let sampling = crate::worker::engine::SamplingParams::llamacpp(
            crate::worker::engine::llamacpp::params::LlamaCppSamplingParams {
                temperature: Some(0.0),
                ..Default::default()
            },
        );
        let start = std::time::Instant::now();
        // `serve_public = false`: this is the bench's OWN measurement, so it MUST reach
        // its (benchmark-owned) candidate worker — the public benchmark-skip is bypassed.
        let (mut deltas, handle) = self
            .chat_stream_inner(
                id.to_owned(),
                messages.to_owned(),
                BENCH_MAX_TOKENS,
                sampling,
                None,
                None,
                false,
            )
            .await?;
        // Time to the first token = the prefill window. Captured from the same
        // `start` as `total` below, so `total - ttft` is EXACTLY the decode window.
        let mut ttft: Option<std::time::Duration> = None;
        while let Some(_delta) = deltas.recv().await {
            if ttft.is_none() {
                ttft = Some(start.elapsed());
            }
        }
        let outcome = handle.await.map_err(|e| HiggsError::ChatTaskFailed {
            detail: e.to_string(),
        })??;
        let total = start.elapsed();
        // No token ever arrived ⇒ ttft = total ⇒ empty decode window ⇒ gen_tps 0.
        let ttft = ttft.unwrap_or(total);
        let ttft_ms = ttft.as_secs_f32() * 1000.0;
        // Generation throughput EXCLUDES prompt prefill (see `bench::bench_gen_tps`),
        // so `pick_benchmarked` ranks candidates by real decode rate, not end-to-end
        // latency skewed by differing prompt-processing costs.
        let gen_tps = crate::tune::bench::bench_gen_tps(outcome.completion_tokens, ttft, total);
        // Prompt throughput approximated from prompt tokens over TTFT (the
        // prefill window); exact prefill timing is a worker-side refinement.
        let prompt_tps = if ttft_ms > 0.0 {
            outcome.prompt_tokens as f32 / (ttft_ms / 1000.0)
        } else {
            0.0
        };
        Ok(BenchResult {
            gen_tps,
            prompt_tps,
            ttft_ms,
        })
    }

    /// Install the running hub (P3 hub mode) so the pairing API can mint node-join tokens.
    pub fn set_hub(&self, hub: Arc<crate::node::hub::Hub>) {
        *self.hub.lock() = Some(hub);
    }

    /// The running hub, if the server is in hub mode.
    pub fn hub(&self) -> Option<Arc<crate::node::hub::Hub>> {
        self.hub.lock().clone()
    }

    /// Tear down the running-hub handle (the kill switch's network-off step). The caller first
    /// closes the endpoint (`Hub::shutdown`) and the node transports
    /// (`HubFleet::disconnect_all`); this then drops the hub so `hub()` is `None` (pairing 409s,
    /// status reports disabled). The `fleet` is deliberately KEPT installed so the route table
    /// survives and re-enabling is a pure reconnect.
    pub fn clear_hub(&self) {
        *self.hub.lock() = None;
    }

    /// The process [`LogBus`] this facade was built with (the one its serve layer reads). The
    /// hub kill switch hands it to `start_hub` when (re-)enabling so relayed remote-worker logs
    /// land in the same Developer-Log console.
    pub fn log_bus(&self) -> Arc<LogBus> {
        self.local.bus().clone()
    }

    /// The lock serializing hub enable/disable (the kill switch). Callers hold it across the
    /// whole lifecycle op (see the `hub_lifecycle` field doc) so concurrent toggles can't race.
    pub fn hub_lifecycle(&self) -> &tokio::sync::Mutex<()> {
        &self.hub_lifecycle
    }

    /// Whether "Verbose Logging" is on. Single home is the [`LogBus`] (read by
    /// the serve completion line AND the worker drain); delegated via the supervisor.
    pub fn verbose(&self) -> bool {
        self.local.bus().verbose()
    }

    /// Turn "Verbose Logging" on or off at runtime on the local node bus (the serve layer +
    /// log layer pick it up instantly; newly-spawned local workers inherit it at load time).
    /// NOTE: with multiple local workers the level is no longer pushed live into
    /// already-running workers' llama.cpp filters — it takes effect on the next load.
    pub fn set_verbose(&self, v: bool) {
        self.local.bus().set_verbose(v);
    }

    /// Whether serve-layer "Log Incoming Tokens" is on (the incoming-prompt line).
    pub fn log_incoming_tokens(&self) -> bool {
        self.log_incoming_tokens
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Turn serve-layer "Log Incoming Tokens" on or off at runtime. Enabling it
    /// opts into logging prompt CONTENT, overriding the redact-by-default policy.
    pub fn set_log_incoming_tokens(&self, v: bool) {
        self.log_incoming_tokens
            .store(v, std::sync::atomic::Ordering::Relaxed);
    }

    /// Whether the Developer Logs are in un-redacted DEBUG mode — the log layer
    /// then emits non-message structured fields (incl. prompt content). Off by
    /// default. Lives on the [`LogBus`] (the layer's only handle); delegated here.
    pub fn log_show_fields(&self) -> bool {
        self.local.bus().show_fields()
    }

    /// Toggle the un-redacted DEBUG log mode at runtime. Enabling it surfaces ALL
    /// structured log fields — including prompt CONTENT — for debugging.
    pub fn set_log_show_fields(&self, v: bool) {
        self.local.bus().set_show_fields(v);
    }

    /// Whether serve-layer "Just-in-Time loading" is on (default `true`). When
    /// on, a chat for a scanned-but-unloaded model loads it on demand instead of
    /// returning `[HG003]`.
    pub fn jit_enabled(&self) -> bool {
        self.jit_enabled.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Turn serve-layer "Just-in-Time loading" on or off at runtime.
    pub fn set_jit_enabled(&self, v: bool) {
        self.jit_enabled
            .store(v, std::sync::atomic::Ordering::Relaxed);
    }

    /// Whether the idle reaper auto-unloads the loaded model after the idle TTL
    /// (default `true`). When `false`, the reaper never unloads — a loaded model
    /// stays resident until an explicit unload regardless of idle time.
    pub fn auto_unload_idle(&self) -> bool {
        self.auto_unload_idle
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Turn idle auto-unload on or off at runtime. Read by the idle reaper each
    /// tick, so a change takes effect without a restart.
    pub fn set_auto_unload_idle(&self, v: bool) {
        self.auto_unload_idle
            .store(v, std::sync::atomic::Ordering::Relaxed);
        self.local.idle().set_enabled(v); // drive the local node reaper live
    }

    /// Idle minutes after which the loaded model is auto-unloaded (default seeded
    /// from the node's [`DEFAULT_IDLE_TTL`], 60 min). Mirrors the node reaper's live
    /// TTL; [`set_idle_ttl_minutes`](Self::set_idle_ttl_minutes) keeps them in sync.
    pub fn idle_ttl_minutes(&self) -> u64 {
        self.idle_ttl_minutes
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Set the idle auto-unload TTL in minutes at runtime. Read by the idle
    /// reaper each tick, so a change takes effect without a restart.
    pub fn set_idle_ttl_minutes(&self, minutes: u64) {
        // CLAMP the request-supplied value (an unbounded `u64` straight off
        // `PUT /api/higgs/settings`) before the `× 60` so a huge minutes count can't
        // overflow the seconds conversion — a debug-build panic, a release wrap. The
        // STORED value is the clamped one, so `idle_ttl_minutes()` and `server_config()`
        // (which both `× 60`) stay overflow-free AND mutually consistent.
        let minutes = minutes.min(MAX_IDLE_TTL_MINUTES);
        self.idle_ttl_minutes
            .store(minutes, std::sync::atomic::Ordering::Relaxed);
        self.local
            .idle()
            .set_ttl(std::time::Duration::from_secs(minutes * 60));
    }

    /// Whether the `/v1` inference surface is currently serving (default `true`).
    /// When `false`, the `/v1` inference endpoints return `[HG019]` → 503 while
    /// the in-process control surface stays reachable.
    pub fn serving_enabled(&self) -> bool {
        self.serving_enabled
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Turn `/v1` inference serving on or off at runtime. Read by the chat
    /// boundary on each request, so a change takes effect immediately without a
    /// restart; the control surface is unaffected.
    pub fn set_serving_enabled(&self, v: bool) {
        self.serving_enabled
            .store(v, std::sync::atomic::Ordering::Relaxed);
    }

    /// Bring up control only — does NOT spawn a worker.
    ///
    /// A worker is spawned lazily by [`load`](Self::load) (spawn-on-load,
    /// LM-Studio model). `scan` runs host-side and needs no worker. The serve
    /// layer holds `Arc<Higgs>` for control regardless of worker liveness.
    ///
    /// Idle auto-unload now lives IN the local `NodeRuntime` (per-worker), so the engine no
    /// longer runs its own reaper. This spawns a relay draining the local node's per-worker
    /// stderr into the shared Developer-Log bus (the node tags lines on per-worker buses,
    /// otherwise separate from the engine bus).
    pub async fn start(self: &Arc<Self>) -> Result<(), HiggsError> {
        let mut logs = self.local.subscribe_logs();
        let bus = self.local.bus().clone();
        tokio::spawn(async move {
            loop {
                match logs.recv().await {
                    // Keyed per worker so each loaded model gets its own log console;
                    // the legacy `?source=worker` filter still matches these (union).
                    Ok((worker, line)) => bus.push(LogSource::LocalWorker { worker }, line),
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {}
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                }
            }
        });
        Ok(())
    }

    /// Gracefully shut down EVERY local worker (the node's `shutdown_all` drain).
    ///
    /// Holds the `lifecycle` mutex for the whole body so a deliberate stop never
    /// interleaves with a concurrent `load` (which would let a fresh worker slip
    /// past the drain). The node's drain reaps each worker's process and clears its
    /// load-replay state, so no worker survives to be auto-restarted.
    pub async fn stop(&self) {
        // Signal a terminal shutdown BEFORE taking the lock: an in-flight Turbotune
        // benchmark (which does NOT hold `lifecycle` during its candidate loop) polls
        // this between candidates and aborts cleanly with [HG064] rather than loading
        // more candidates against the node we're about to drain.
        self.shutting_down
            .store(true, std::sync::atomic::Ordering::Relaxed);
        let _lifecycle = self.lifecycle.lock().await;
        // Drain every local worker (graceful shutdown of the local node's registry).
        self.local.shutdown_all().await;
    }

    /// Scan all configured model directories and return the discovered models.
    ///
    /// Runs host-side: model scanning is pure Rust (`ggus` + `memmap2` + `std::fs`,
    /// no llama.cpp FFI) so it needs no worker. Returns `Err` [HG001] if a
    /// configured root exists but cannot be read.
    pub async fn scan(&self) -> Result<Vec<HiggsModel>, HiggsError> {
        let (lmstudio, hf, ollama) = {
            let cfg = self.config.lock();
            (
                cfg.lmstudio_dirs.clone(),
                cfg.hf_dirs.clone(),
                cfg.ollama_dirs.clone(),
            )
        };
        // Scanning does blocking I/O (`std::fs::read_dir`, file open, `memmap2`
        // mmap). `status` polls `scan` and `load` calls it, so running inline
        // would block a tokio runtime thread — offload to a blocking thread.
        // A `JoinError` here means the scan task itself panicked, which is an
        // unrecoverable bug rather than a host-facing condition.
        tokio::task::spawn_blocking(move || {
            let mut store = crate::worker::models::ModelStore::default();
            store.scan(&lmstudio, &hf, &ollama).map(<[_]>::to_vec)
        })
        .await
        .expect("higgs model scan task panicked")
    }

    /// Load a model by HuggingFace repo id.
    ///
    /// `params` overrides `default_load` when supplied. On success, records the
    /// load params for post-restart replay and emits [`HiggsEvent::ModelLoaded`],
    /// and PERSISTS the effective requested params to `config.json` (per-model
    /// record) so the UI can display what the model was loaded with and a future
    /// autoload can reload it the same way. Persistence is best-effort: a config
    /// write failure is logged, never failing an otherwise-successful load.
    pub async fn load(&self, id: &str, params: Option<LoadParams>) -> Result<(), HiggsError> {
        let from_request = params.is_some();
        self.load_inner(id, params, from_request).await
    }

    /// `load` core, parameterized by `from_request` — whether to PERSIST `params`
    /// as the saved profile after a successful load. The public [`load`] derives it
    /// from `params.is_some()`. The JIT path passes a VALIDATED profile with
    /// `from_request = false`: a no-resync REUSE that loads exactly the profile the
    /// readiness gate just checked, with NO second `models.json` read — closing the
    /// check-then-load race where the profile could vanish and fall back to dumb
    /// defaults.
    pub(crate) async fn load_inner(
        &self,
        id: &str,
        params: Option<LoadParams>,
        from_request: bool,
    ) -> Result<(), HiggsError> {
        use crate::api::types::ModelLoadPhase;
        // PUSH the load lifecycle to `GET /api/higgs/events` subscribers. `Queued`
        // fires FIRST — before the (possibly contended) lifecycle lock — so the UI's
        // bar appears the instant a load is requested, even while it waits behind
        // another in-flight load. The inner impl emits the mid-load phases
        // (`Preparing`/`LoadingWeights`/`Finalizing`) at their real seams; here we
        // bracket it with the terminal `Ready`/`Failed` so the bar is ALWAYS closed
        // out (including the idempotent resident no-op, which emits Queued→Ready).
        self.emit_load_phase(id, ModelLoadPhase::Queued, None);
        // A DROP GUARD so the bar is closed even on CANCELLATION: if the caller's
        // future is dropped mid-load (e.g. an HTTP client disconnects during
        // `local.load().await`), the `.await` below never returns and the explicit
        // terminal never fires — so this guard emits a terminal `Failed` on drop.
        // Disarmed once we reach the explicit terminal, so the normal path emits
        // exactly one terminal. (Mirrors the `LoadingGuard` that clears `loading`.)
        struct TerminalGuard<'a> {
            higgs: &'a Higgs,
            id: &'a str,
            armed: bool,
        }
        impl Drop for TerminalGuard<'_> {
            fn drop(&mut self) {
                if self.armed {
                    self.higgs.emit_load_phase(
                        self.id,
                        crate::api::types::ModelLoadPhase::Failed,
                        Some("cancelled".to_owned()),
                    );
                }
            }
        }
        let mut term = TerminalGuard {
            higgs: self,
            id,
            armed: true,
        };
        let result = self.load_inner_impl(id, params, from_request).await;
        term.armed = false; // reached a real terminal — the guard must not also fire
        match &result {
            Ok(()) => self.emit_load_phase(id, ModelLoadPhase::Ready, None),
            Err(e) => {
                let code = miette::Diagnostic::code(e).map(|c| c.to_string());
                self.emit_load_phase(id, ModelLoadPhase::Failed, code);
            }
        }
        result
    }

    /// The load body — see [`load_inner`](Self::load_inner), which brackets this with
    /// the `Queued`/`Ready`/`Failed` load-event phases. This impl emits the mid-load
    /// phases at their real seams.
    async fn load_inner_impl(
        &self,
        id: &str,
        params: Option<LoadParams>,
        from_request: bool,
    ) -> Result<(), HiggsError> {
        use crate::api::types::ModelLoadPhase;
        // Serialize concurrent loads at the facade so two racing JIT loads of the
        // same id can't each spawn a worker (the node load is additive — it never
        // dedups). Held for the whole method body.
        let _lifecycle = self.lifecycle.lock().await;
        // A benchmark owns this model while it measures candidate configs — refuse a
        // public/JIT load rather than race it ([HG068]). Checked UNDER `lifecycle`,
        // which `turbotune_bench` also holds when it sets the flag, so the flag and
        // this check can't interleave. The bench's OWN candidate loads use the
        // private `local.load` and are not gated here.
        if self.is_benchmarking(id) {
            return Err(HiggsError::BenchInProgress { id: id.to_owned() });
        }
        // 1. Charset guard ([HG015]) — reject traversal/escape ids before any FS
        //    use. The node does resolve ([HG002]), the path-traversal guard
        //    ([HG015]), and the RAM headroom guard ([HG017]); this charset check
        //    is the one guard that stays host-side (it precedes any FS access).
        validate_repo_id(id)?;
        // Idempotent per model: if a worker for this raw id is already resident,
        // this is a no-op success. The node load is ADDITIVE (it never dedups), so
        // without this a second explicit load — or two racing JIT loads of the same
        // unloaded model — would spawn a duplicate worker. The `lifecycle` mutex
        // serializes concurrent loads, so the common check-then-load is race-free.
        // (To run N instances of one model, the node API is additive; the facade
        // deliberately presents one-worker-per-model.)
        //
        // RESIDUAL (cancellation): the mutex is released if a load future is
        // CANCELLED mid-flight (after its `NodeMsg::Load` is sent but before the
        // reply). A second same-id load can then acquire the mutex, observe no
        // COMMITTED worker (the first's commit is still pending), and spawn a
        // duplicate. This self-heals: the cancelled load's `LoadCommit` reaps its
        // worker (the caller is gone, so the keep-insert is skipped), and the RAM
        // headroom guard bounds the transient overlap — end state is one worker. A
        // fully cancellation-safe dedup would require the node actor to track and
        // dedup IN-FLIGHT loads, changing its additive contract (used by the remote
        // path); deferred as not worth that for a self-healing, memory-bounded race.
        if self.local.instances().await.iter().any(|(_, m)| m == id) {
            // Idempotent no-op: the model stays loaded with its CURRENT params — the
            // request params are NOT applied to the resident worker here. So we also do
            // NOT sync them to the saved profile: persisting params that were never
            // validated by a real load would let a later plain reload reuse a profile
            // that fails (the success-before-persist invariant the full path keeps —
            // it syncs only AFTER `self.local.load` succeeds). To change a resident
            // model's saved profile, unload then load.
            return Ok(());
        }
        // Committed to a real load now (past the resident no-op): announce the
        // pre-load work — id/profile validation, param resolution, anchor capture.
        self.emit_load_phase(id, ModelLoadPhase::Preparing, None);
        // Load-seam precedence (DESIGN-autotune §3.1): explicit request params win;
        // else a saved per-model tuning profile (a prior `tune` the user kept, in
        // `models.json`) is reused — "tune once, loads that way every time"; else
        // (`None`) the node's default_load / ctx-cap path. (The `autotune_on_load`
        // suggester branch slots between saved-profile and default_load — P1.5.)
        // Track explicit-vs-reused (`from_request`, a parameter): an EXPLICIT load
        // (request carried params — a fresh suggestion or an accepted edit) updates
        // the saved profile below; a load that REUSED a profile leaves it unchanged.
        let params = match params {
            Some(p) => Some(p),
            // Reusing the saved profile: a STALE one hard-blocks (same contract as
            // the JIT gate) so an explicit `load(id, None)` can't reuse a profile
            // whose hardware/model file changed since Prepare. Re-tune, or load with
            // explicit params, to recover. A Missing profile (no record) falls
            // through to `None` → the node's default_load.
            //
            // The store open is BEST-EFFORT here (`.ok()`): explicit load is
            // user-initiated and persistence-resilient — an unreadable store falls
            // through to a default load rather than failing the load. (The JIT gate
            // and the readiness LISTING surface the same store fault as HG040, where
            // the contract is "don't silently mislead"; an explicit load is not that.)
            None => match self.models_store().ok().and_then(|s| s.tuning(id)) {
                Some(rec) => {
                    // Validate staleness on THE SAME record we're about to reuse —
                    // NOT a fresh `profile_state()` re-read, which a concurrent tune
                    // could swap underneath us (validate one record, load another). A
                    // model absent from the scan isn't stale; let the load surface the
                    // real not-found instead of a misleading Re-tune.
                    if let Some(path) = self
                        .scan()
                        .await
                        .ok()
                        .and_then(|ms| ms.into_iter().find(|m| m.id == id).map(|m| m.path))
                    {
                        let hw = self.hardware().await;
                        if profile_stale(&rec, &path, &hw) {
                            return Err(HiggsError::ProfileStale { id: id.to_owned() });
                        }
                    }
                    Some(rec.profile)
                }
                None => None,
            },
        };
        // A REUSE load of a SAVED profile (JIT / a plain reload passes the profile with
        // `from_request = false`). If the OOM ladder has to DEGRADE such a load, the
        // fitting config it discovers must be written back to the saved profile — else
        // every reload re-reads the original OOMing profile and re-walks the ladder
        // (codex r10). A default load (`params = None`) is NOT a profile reuse.
        let reused_profile = !from_request && params.is_some();
        // Map the host `LoadParams` onto the node's `NodeLoadParams`. The base
        // fields (id/ctx_len/gpu_layers/threads) drive the node's resolve/ctx-cap;
        // the FULL engine override set rides `params` and is applied by the worker
        // (the prior drop-at-the-node gap is now closed). When the caller pins no
        // params, ctx_len is left `None` so the node defaults it to the model's
        // trained context (capped at DEFAULT_CTX_CAP).
        let np = match params {
            Some(p) => NodeLoadParams {
                id: id.to_owned(),
                // `ctx_len == 0` means AUTO → leave it `None` so the node picks the
                // model's trained context (capped), exactly like a no-params load.
                // This makes a replayed `last_load` (which records auto as `0`)
                // reproduce the SAME auto load instead of pinning a literal 0 (which
                // the worker would coerce to its 4096 fallback). A pinned 0 context is
                // nonsensical anyway, so nothing meaningful is lost.
                // CtxLen::Auto → None (the node defaults it to the trained context);
                // Fixed { n } → Some(n). Same intent as the old `(ctx_len > 0).then_some`.
                ctx_len: p.ctx_len().fixed_n(),
                gpu_layers: Some(p.gpu_layers()),
                threads: Some(p.threads()),
                // Forward the rich engine override set (type_k/flash_attn/cpu_moe/…)
                // so the worker applies it — but ONLY when there's something beyond the
                // base 3 to apply, so a base-only load carries no payload.
                params: {
                    let lc = p.as_llamacpp();
                    lc.has_overrides().then(|| lc.clone())
                },
            },
            None => {
                let d = self.config.lock().default_load.clone();
                NodeLoadParams {
                    id: id.to_owned(),
                    // A no-params load uses AUTO context (`None` → the model's TRAINED
                    // context, capped at DEFAULT_CTX_CAP by the node, runtime.rs:842),
                    // NOT `default_load.ctx_len`. This asymmetry with gpu_layers/threads
                    // below is DELIBERATE: `gpu_layers = all` and a thread count are
                    // sensible UNIVERSAL defaults, but a fixed context (the base
                    // placeholder 4096) is NOT — the node uses a PINNED ctx_len verbatim
                    // (runtime.rs:842 does NOT cap it at the trained context), so pinning
                    // 4096 would rope-extend a 2048-trained model and degrade it. Auto
                    // sizes per-model correctly. (See the fn doc + the Some(p) branch.)
                    ctx_len: None,
                    gpu_layers: Some(d.gpu_layers()),
                    threads: Some(d.threads()),
                    params: {
                        let lc = d.as_llamacpp();
                        lc.has_overrides().then(|| lc.clone())
                    },
                }
            }
        };
        // For an ACCEPTED explicit load we'll anchor the saved profile to the file
        // it was validated against — capture the hardware fingerprint + file
        // signature BEFORE the (slow) load, so a GGUF swapped mid-load can't mark
        // the profile fresh for a file it wasn't loaded against (the same race
        // `tune` avoids). Captured for an explicit load AND a reuse load (the latter so
        // a DEGRADED reuse can re-anchor the fitting config it discovered); a pure
        // default load never syncs, so it captures nothing.
        let anchors = if from_request || reused_profile {
            let hw_fp = self.hardware().await.fingerprint();
            let sig = self
                .scan()
                .await
                .ok()
                .and_then(|ms| ms.into_iter().find(|m| m.id == id).map(|m| m.path))
                .map(|p| file_sig(&p))
                .unwrap_or_default();
            Some((hw_fp, sig))
        } else {
            None
        };
        // Additive load on the local node: spawns a fresh worker for `id` (the
        // node emits `ModelLoaded` on commit). resolve / headroom / path-traversal
        // failures surface as their mapped HiggsError.
        //
        // Publish the in-flight load so a concurrent `status` can show a progress
        // indicator during the (multi-second) load. A DROP GUARD clears it on ANY
        // exit from the load — success, error (`?`), or CANCELLATION (the load
        // future dropped mid-`.await`, e.g. the caller disconnects) — so `status`
        // can't report a phantom load forever. The lock is never held across the
        // `.await` (only in the guard's `drop`).
        struct LoadingGuard<'a>(&'a parking_lot::Mutex<Option<crate::api::types::ModelLoading>>);
        impl Drop for LoadingGuard<'_> {
            fn drop(&mut self) {
                *self.0.lock() = None;
            }
        }
        *self.loading.lock() = Some(crate::api::types::ModelLoading {
            id: id.to_owned(),
            started_ms: now_unix_ms(),
        });
        let _loading_guard = LoadingGuard(&self.loading);
        // The multi-second worker load starts now (mmap → GPU → KV alloc).
        self.emit_load_phase(id, ModelLoadPhase::LoadingWeights, None);
        // G5: walk the OOM degrade-retry ladder around the real node load. The model's
        // transformer block count lets the ladder turn a default `GpuLayers::All` OOM
        // into a concrete half-offload rung; a scan miss leaves it `None` (the ladder
        // then falls back to all-CPU). A fits-first-time load is one call, no ladder.
        let layer_count = self
            .scan()
            .await
            .ok()
            .and_then(|ms| ms.into_iter().find(|m| m.id == id))
            .and_then(|m| m.block_count);
        let requested_np = np.clone();
        let loaded_np = run_oom_ladder(
            id,
            np,
            layer_count,
            crate::load_robustness::SETTLE_BEFORE_RETRY,
            |p| async move { self.local.load(p).await.map(|_| ()) },
        )
        .await?;
        // Whether the OOM ladder had to DEGRADE — the loaded rung differs from what was
        // requested. A degraded REUSE load must write the fitting config back to the
        // saved profile (below), or every reload re-walks the ladder (codex r10).
        let degraded = loaded_np != requested_np;
        // Weights are resident; the remaining work is fast bookkeeping.
        self.emit_load_phase(id, ModelLoadPhase::Finalizing, None);
        // The params we PERSIST are the ones that ACTUALLY loaded — `run_oom_ladder`
        // returns the successful rung, so after an OOM degrade this is the cheaper
        // (KV-off / fewer-layers) config, NOT the seed that OOMed. The record + a
        // future autoload therefore reproduce the config that fits. Derived from the
        // node params so it reflects EXACTLY what crossed the wire; `ctx_len` of 0
        // means AUTO (node called with `ctx_len: None` → trained context, capped).
        let effective = match &loaded_np.params {
            Some(lc) => {
                let mut lc = lc.clone();
                lc.ctx_len = loaded_np
                    .ctx_len
                    .map(crate::worker::engine::CtxLen::fixed)
                    .unwrap_or(crate::worker::engine::CtxLen::Auto);
                lc.gpu_layers = loaded_np.gpu_layers.unwrap_or_default();
                lc.threads = loaded_np.threads.unwrap_or_default();
                LoadParams::llamacpp(lc)
            }
            None => LoadParams::base(
                loaded_np
                    .ctx_len
                    .map(crate::worker::engine::CtxLen::fixed)
                    .unwrap_or(crate::worker::engine::CtxLen::Auto),
                loaded_np.gpu_layers.unwrap_or_default(),
                loaded_np.threads.unwrap_or_default(),
            ),
        };
        // Persist the per-model load record (best-effort — never fail a good load).
        let now = now_unix_ms();
        if let Err(e) = self.with_config_mut(|c| c.record_load(id, effective.clone(), now)) {
            tracing::warn!(id, error = %e, "higgs: failed to persist load record to config.json");
        }
        // Keep the saved tuning profile in sync with the config that ACTUALLY loaded:
        // - an ACCEPTED EXPLICIT load persists the accepted/edited params (as before), so
        //   a plain reload reuses them, not a stale tune suggestion; and
        // - a REUSE load that the OOM ladder had to DEGRADE writes the fitting config back
        //   (codex r10) so the next reload uses it instead of re-walking the ladder.
        // A reuse load that loaded AS-IS is left unchanged (no drift; `degraded` is false).
        // (The resident-model early return above does the same sync.) `sync_saved_profile`
        // → `set_profile` drops the record's stale `provenance`/`bench_tps` whenever the
        // written params DIFFER from the saved profile — an OOM-degraded fallback (r11) or
        // an explicitly edited reload (r12); an as-requested reload of the tuned params
        // keeps them.
        if let Some((hw_fp, sig)) = anchors {
            if from_request || degraded {
                self.sync_saved_profile(id, &effective, &hw_fp, &sig).await;
            }
        }
        Ok(())
    }

    /// Persist `profile` as the saved per-model tuning profile for `id` — the LAST
    /// accepted explicit load — so a later plain reload reuses it. Serialized through
    /// `models_io` with a fresh re-read (the whole `models.json` is rewritten on
    /// flush). Best-effort: a write failure is logged, never fails the load.
    /// Persist `profile` as the saved tuning profile for `id`, anchored to the
    /// `hw_fp` + `file_sig` the CALLER captured BEFORE the (slow) load — NOT
    /// re-sampled here. Sampling after the load would let a GGUF swapped mid-load
    /// anchor the new file to a profile validated against the old (the same race
    /// `tune` avoids by capturing the signature up front).
    async fn sync_saved_profile(&self, id: &str, profile: &LoadParams, hw_fp: &str, sig: &str) {
        let _guard = self.models_io.lock();
        if let Ok(store) = self.models_store() {
            // `set_profile` itself drops stale bench metrics if these params differ from
            // the saved profile (codex r11/r12), so the caller need not decide.
            store.set_profile(id, profile.clone(), hw_fp, sig, now_unix_ms());
            if let Err(e) = store.flush() {
                // Best-effort (a load already succeeded), but log the coded HG040 so a
                // recurring persistence problem is diagnosable from the warning.
                let pe = HiggsError::PersistenceFailed {
                    store: "models".into(),
                    path: "models.json".into(),
                    source: e,
                };
                tracing::warn!(id, error = %pe, "higgs: failed to persist accepted load profile");
            }
        }
    }

    /// Unload ALL locally-resident models.
    ///
    /// The control surface is single-button ("unload"), so this drains every local
    /// worker. Each [`NodeRuntime::unload`](crate::node::runtime::NodeRuntime::unload)
    /// removes the worker from the registry synchronously, then AWAITS its
    /// `Supervisor::stop()` (the node fires the reply only after the process is
    /// reaped) and emits one [`HiggsEvent::ModelUnloaded`].
    ///
    /// `shutdown_all` is NOT used here: it is terminal (it permanently rejects
    /// future loads), whereas unload must leave the node ready to load again.
    ///
    /// CONCURRENCY: the `lifecycle` mutex serializes this against `load` (no fresh
    /// worker can slip past the drain — `load` is the only thing that ADDS a worker).
    /// The one other concurrent mutator is the node's idle reaper, which only
    /// REMOVES workers. If it grabs a snapshotted worker between this loop's
    /// `instances()` snapshot and that worker's `unload`, our `unload` returns
    /// `no_worker` (ignored): that worker was already removed AND its stop is reaped
    /// by the node on a tracked teardown task (no leak/orphan — the actor holds
    /// itself alive until every in-flight stop completes). So the post-condition
    /// "no worker resident" always holds on return; the only residual is that a
    /// reaper-grabbed worker's process exit may finish microseconds after this
    /// returns, which is harmless (a following additive `load` is gated by the RAM
    /// headroom guard, and multi-model workers coexist by design anyway).
    pub async fn unload(&self) -> Result<(), HiggsError> {
        let _lifecycle = self.lifecycle.lock().await;
        // A benchmark owns its model exclusively while it measures candidate configs.
        // A drain-all here would evict the bench's transiently-resident candidate worker
        // mid-measurement (contaminating the numbers / failing the run), so refuse while
        // ANY benchmark is in flight ([HG068]) — retry once it finishes. The bench's OWN
        // teardown uses the private `local.unload`, not this facade drain, so it is not
        // blocked. (Checked under `lifecycle`, the same order the bench sets the flag.)
        if let Some(id) = self.benchmarking.lock().iter().next().cloned() {
            return Err(HiggsError::BenchInProgress { id });
        }
        for (worker, _model) in self.local.instances().await {
            // Ignore `no_worker`: a concurrent idle-reap already removed+reaped it
            // (see the CONCURRENCY note above). The reaper never ADDS, so after this
            // single pass the registry is empty.
            let _ = self.local.unload(worker).await;
        }
        Ok(())
    }

    /// Unload ONE resident model by its served id, leaving every other local
    /// worker resident — the per-model counterpart to [`unload`](Self::unload)'s
    /// drain-all. The local node is multi-model, so a UI Eject / Fleet Unload of a
    /// single card must target only that instance. Resolves the served id (`org/model`,
    /// `org/model-1`, …) to its worker and unloads it; an unknown/already-gone id is an
    /// idempotent no-op (the post-condition "that id is not resident" holds either way).
    /// Shares the `lifecycle` mutex with `load`/`unload` so it can't race a concurrent
    /// load that ADDS a worker.
    ///
    /// KNOWN LIMITATION (served-id stability — explicitly deferred): a served id is
    /// resolved live against the current worker set, the SAME contract as every other
    /// control path ([`local_loaded_info`](Self::local_loaded_info), `chat_stream`, and
    /// the remote `require_served`). For DISTINCT models (the common multi-model case)
    /// ids are unique and stable, so a per-card unload is exact. The unstable window is
    /// DUPLICATE instances of one raw model: suffixed ids (`org/model-1`, …) renumber
    /// when a sibling is idle-reaped/unloaded, so a request built from a STALE snapshot
    /// can resolve to a different instance — and, in the pathological case where a raw
    /// model id literally ends in `-N` and collides with a generated suffix, even to a
    /// DIFFERENT model. This is a property of higgs's served-id SCHEME (shared by chat
    /// and the remote unload), not of this method; the proper fix is a stable
    /// worker-id/generation selector threaded through the whole eject/unload chain
    /// (UI card + Fleet → wire → here), DEFERRED as disproportionate to a pathological
    /// trigger. Until then a destructive unload trusts the served id the caller saw.
    pub async fn unload_one(&self, served: &str) -> Result<(), HiggsError> {
        let _lifecycle = self.lifecycle.lock().await;
        // BETWEEN candidate loads the benchmarked model has no served worker, so the
        // resolution below misses it — without this bare-flag check a per-model unload
        // would return a FALSE success while the benchmark keeps running (codex r20).
        // (Benchmarks target RAW ids; a suffixed served id is never a bench target.)
        if self.is_benchmarking(served) {
            return Err(HiggsError::BenchInProgress {
                id: served.to_owned(),
            });
        }
        if let Some((worker, raw)) = self.local_served().await.remove(served) {
            // The resolved model is being benchmarked: refuse to eject its candidate
            // worker ([HG068]) — evicting it mid-measurement would contaminate/fail the
            // run. The bench's own teardown uses the private `local.unload`, not this.
            if self.is_benchmarking(&raw) {
                return Err(HiggsError::BenchInProgress { id: raw });
            }
            // Ignore `no_worker`: the idle reaper may have removed+reaped it between
            // the resolve and here — the end state (id not resident) is the same.
            let _ = self.local.unload(worker).await;
        }
        Ok(())
    }

    /// Return a live status snapshot of the PRIMARY local instance.
    ///
    /// The local node may host several models (additive load); the single-model
    /// control surface reports the PRIMARY — the lowest worker id. `worker_alive`
    /// is `true` iff a worker is resident and its `M_STATUS` round-trip succeeded.
    /// `loaded` is independently best-effort: no resident worker yields
    /// `worker_alive:false`/`loaded:None`; a malformed `loaded` shape yields
    /// `worker_alive:true`/`loaded:None`. `/v1/models` lists ALL served instances
    /// via [`local_served_ids`](Self::local_served_ids), not just this primary.
    pub async fn status(&self) -> Result<HiggsStatus, HiggsError> {
        // Scan host-side (pure Rust, no worker RPC): model metadata + on-disk
        // count both come from ONE FS walk, reused for enrichment below.
        let scan = self.scan().await.unwrap_or_default();
        let models_on_disk = scan.len() as u32;

        // Primary = lowest worker id. No resident worker → idle state.
        let mut instances = self.local.instances().await;
        instances.sort_by_key(|(w, _)| *w);
        let Some(primary) = instances.first().map(|(w, _)| *w) else {
            return Ok(HiggsStatus {
                worker_alive: false,
                loading: self.loading.lock().clone(),
                loaded: None,
                loaded_all: Vec::new(),
                models_on_disk,
            });
        };

        // The local node is multi-model (one worker per resident model). Report EVERY worker —
        // each tagged with its `worker_id` and its `/v1` served id — so the UI can show a card
        // (and a per-worker log pane) per worker, not just the primary. `served` maps the served
        // id back per worker (local is deduped one-worker-per-model, so served == raw id).
        let served = self.local_served().await;
        let worker_to_served: std::collections::HashMap<WorkerId, String> =
            served.into_iter().map(|(s, (w, _raw))| (w, s)).collect();
        // Live load params come from ONE probe of the PRIMARY — EXACTLY as before this
        // multi-model change (a busy primary's `M_STATUS` already queued once per poll; we add NO
        // new per-worker RPCs). SECONDARY workers are listed from CACHED state (the instance set +
        // the host-side scan), so a busy/wedged secondary never adds an RPC, never stalls the
        // snapshot, and never queues an orphaned status request behind a long generation. (Their
        // live ctx/gpu/threads come as a stub for now; per-worker live stats are the T9 follow-up.)
        let primary_status = self.local.status(primary).await;
        let worker_alive = primary_status.is_ok();
        let primary_info = primary_status
            .ok()
            .and_then(|v| self.loaded_info_from(&v, &scan));

        // `loaded` (back-compat) = the primary's SUCCESSFUL probe ONLY — `None` when it failed or
        // was malformed, preserving the legacy `/api/higgs/status.loaded` contract (a stub must
        // never masquerade as a real loaded model for provider seeding). The `loaded_all` list MAY
        // carry a stub for the primary so it still appears, but `loaded` does not.
        let loaded = primary_info.clone().map(|mut i| {
            if let Some(served_id) = worker_to_served.get(&primary) {
                i.id = served_id.clone();
            }
            i.worker_id = primary.0;
            i
        });

        // Persisted load records, read ONCE per status pass — the stub's fallback
        // for ctx/gpu/threads on workers that aren't live-probed.
        let records = self.model_records();
        let mut loaded_all = Vec::with_capacity(instances.len());
        for (worker, raw) in &instances {
            let served_id = worker_to_served.get(worker).cloned().unwrap_or(raw.clone());
            let info = if *worker == primary {
                // Primary in the LIST: full live params from the probe, or a stub if it failed —
                // so a busy/wedged primary still shows a card (its live params just default).
                primary_info
                    .clone()
                    .map(|mut i| {
                        i.id = served_id.clone();
                        i.worker_id = worker.0;
                        i
                    })
                    .unwrap_or_else(|| {
                        self.loaded_info_stub(served_id, worker.0, raw, &scan, &records)
                    })
            } else {
                self.loaded_info_stub(served_id, worker.0, raw, &scan, &records)
            };
            loaded_all.push(info);
        }

        Ok(HiggsStatus {
            worker_alive,
            loading: self.loading.lock().clone(),
            loaded,
            loaded_all,
            models_on_disk,
        })
    }

    /// A [`LoadedInfo`] built WITHOUT a worker RPC: `id`/`worker_id` + host-side scan metadata
    /// (`arch`/`quant`/`size_bytes`/`max_context_length`). The load params
    /// (`ctx_len`/`gpu_layers`/`threads`) are filled from the model's PERSISTED load record
    /// (what it was last successfully loaded with — which is what the resident worker runs),
    /// `None` if no record survives; [`status`](Self::status) still overrides with a bounded
    /// live probe when the worker answers in time. Lets EVERY resident worker appear in
    /// `loaded_all` even while its worker is busy generating (and so can't answer `M_STATUS`).
    fn loaded_info_stub(
        &self,
        served_id: String,
        worker_id: u32,
        raw_model: &str,
        scan: &[HiggsModel],
        records: &std::collections::BTreeMap<String, crate::config::ModelRecord>,
    ) -> LoadedInfo {
        let scanned = scan.iter().find(|m| m.id == raw_model);
        // Live params aren't probed (no worker RPC) — fall back to the persisted
        // load record: what this model was last successfully loaded with, which is
        // exactly what the resident worker is running (a successful load always
        // records). Last-known from the record, not a live probe.
        let (ctx_len, gpu_layers, threads) = match records
            .get(raw_model)
            .and_then(|r| r.load.as_ref())
        {
            Some(LoadParams::LlamaCpp(p)) => {
                // A recorded `Auto` was resolved BY THE NODE at load time to the trained
                // context capped at DEFAULT_CTX_CAP (runtime.rs `load_worker`) — mirror
                // that resolution so the reported window is the real one, not the
                // unresolved sentinel. No trained context known → `None` (unknown).
                let ctx = match p.ctx_len {
                    crate::worker::engine::CtxLen::Auto => {
                        scanned.and_then(|m| m.ctx_train).map(|t| {
                            crate::worker::engine::CtxLen::fixed((t as u32).min(DEFAULT_CTX_CAP))
                        })
                    }
                    fixed => Some(fixed),
                };
                (ctx, Some(p.gpu_layers), Some(p.threads))
            }
            None => (None, None, None),
        };
        LoadedInfo {
            id: served_id,
            worker_id,
            ctx_len,
            gpu_layers,
            threads,
            arch: scanned.and_then(|m| m.arch.clone()),
            quant: scanned.and_then(|m| m.quant.clone()),
            max_context_length: scanned.and_then(|m| m.ctx_train),
            size_bytes: scanned.map(|m| m.size_bytes),
            has_chat_template: scanned.map(|m| m.has_chat_template),
            idle_ttl_minutes: None,
        }
    }

    /// Enrich a worker's raw `M_STATUS` value into a [`LoadedInfo`] using the
    /// host-side `scan`. The worker reports `id`/`ctx_len`/`gpu_layers`/`threads`
    /// from the live model, but `arch`/`quant`/`size_bytes`/`max_context_length`/
    /// `has_chat_template` come back null (its store is empty) — those are filled
    /// from the matching scanned [`HiggsModel`]. `None` when nothing is loaded or
    /// the shape is malformed. Shared by [`status`](Self::status) (primary) and
    /// [`local_loaded_info`](Self::local_loaded_info) (a specific served id).
    fn loaded_info_from(&self, v: &serde_json::Value, scan: &[HiggsModel]) -> Option<LoadedInfo> {
        let l = v.get("loaded")?;
        if l.is_null() {
            return None;
        }
        let id = l.get("id")?.as_str()?.to_owned();
        let scanned = scan.iter().find(|m| m.id == id);
        Some(LoadedInfo {
            // None by default; the caller (status/local_loaded_info) sets the real worker id.
            worker_id: 0,
            // Probed live values (the worker answered M_STATUS) → Some.
            ctx_len: Some(serde_json::from_value(l.get("ctx_len")?.clone()).ok()?),
            gpu_layers: Some(serde_json::from_value(l.get("gpu_layers")?.clone()).ok()?),
            threads: Some(l.get("threads")?.as_u64()? as u32),
            arch: scanned.and_then(|m| m.arch.clone()),
            quant: scanned.and_then(|m| m.quant.clone()),
            max_context_length: scanned.and_then(|m| m.ctx_train),
            size_bytes: scanned.map(|m| m.size_bytes),
            has_chat_template: scanned.map(|m| m.has_chat_template),
            // Always `None` (= "uses the global idle TTL"): per-load idle-TTL
            // ENFORCEMENT is a deferred follow-up (the node reaper applies one
            // per-node TTL to every worker — runtime.rs `ReapIdle`), so surfacing a
            // per-load value here would claim an override the reaper does not honor.
            // See `LoadedInfo::idle_ttl_minutes` + the `NodeLoadParams` deferral note.
            idle_ttl_minutes: None,
            id,
        })
    }

    /// Stream a chat completion.
    ///
    /// `model` is the id the serve layer resolved this chat against (via
    /// [`ensure_loaded`](crate::serve)). It is carried to the worker, which
    /// refuses to generate (`[HG018]` → 503) if a concurrent JIT load swapped the
    /// resident model out between resolution and dispatch — binding each chat to
    /// its resolved model so a swap errors instead of serving the wrong model.
    ///
    /// Returns `(receiver, join_handle)`:
    /// - `receiver` carries streaming deltas — each item is one content chunk
    ///   from the worker; this is the canonical output for SSE / streaming consumers.
    /// - `join_handle` resolves with the final [`ChatOutcome`] when generation is
    ///   complete (or `Err` if the worker fails); `ChatOutcome::content` is the
    ///   full concatenated text and is the canonical output for non-streaming
    ///   consumers (`/v1` with `stream: false`).  Both are retained on purpose —
    ///   callers choose which representation they need.
    ///
    /// Concurrent callers are each accepted and routed their own deltas via a
    /// per-request keyed channel; the worker executes requests serially (single-
    /// threaded stdin loop) so throughput is sequential but callers never clobber
    /// each other's streams.
    /// PUBLIC chat entry (the serve layer's dispatch). Delegates to
    /// [`chat_stream_inner`](Self::chat_stream_inner) with `serve_public = true`, so a
    /// benchmark-owned local candidate worker is SKIPPED rather than served to public
    /// traffic (the Turbotune isolation the serve gate enforces, closed against the
    /// gate→dispatch re-resolution race).
    pub async fn chat_stream(
        &self,
        model: String,
        messages_json: String,
        max_tokens: usize,
        sampling: crate::worker::engine::SamplingParams,
        tools_json: Option<String>,
        chat_template_kwargs: Option<String>,
    ) -> Result<
        (
            crate::delta_queue::DeltaReceiver,
            tokio::task::JoinHandle<Result<ChatOutcome, HiggsError>>,
        ),
        HiggsError,
    > {
        self.chat_stream_inner(
            model,
            messages_json,
            max_tokens,
            sampling,
            tools_json,
            chat_template_kwargs,
            true,
        )
        .await
    }

    /// The chat dispatch body. `serve_public` gates benchmark isolation: `true` (public
    /// serve path) SKIPS a locally-resident worker whose raw model is being benchmarked
    /// — a TRANSIENT Turbotune candidate the serve gate did not route to (it chose
    /// remote/JIT because the id was not publicly resident), which the bench unloads
    /// between candidates. `false` is the bench's OWN measurement
    /// ([`measure_gen_tps`](Self::measure_gen_tps)), which MUST reach its candidate.
    #[allow(clippy::too_many_arguments)]
    async fn chat_stream_inner(
        &self,
        model: String,
        messages_json: String,
        max_tokens: usize,
        sampling: crate::worker::engine::SamplingParams,
        tools_json: Option<String>,
        chat_template_kwargs: Option<String>,
        serve_public: bool,
    ) -> Result<
        (
            crate::delta_queue::DeltaReceiver,
            tokio::task::JoinHandle<Result<ChatOutcome, HiggsError>>,
        ),
        HiggsError,
    > {
        // LOCAL routing FIRST: a locally-resident served id wins — it is faster and
        // keeps routing CONSISTENT with `/v1/models` (which lists local served ids
        // first) and `ensure_loaded` (local-first), so a listed local instance is
        // always reachable even if a remote node happens to expose the same served
        // string. `model` is a SERVED id (`org/model`, `org/model-1`); the worker
        // matches on the RAW model, so the raw id rides the wire.
        //
        // Served suffix ids are EPHEMERAL (a pure function of the live instance set —
        // see `node::served`), so in the narrow window between the serve-layer gate's
        // resolution and this one, an unload/idle-reap can renumber suffixes. This
        // resolves at DISPATCH time (the authoritative current meaning) and the worker
        // runs the exact tokenizer [HG005] check, so a stale gate estimate degrades to
        // the worker's own backstop rather than a wrong answer — never a panic.
        // PUBLIC traffic SKIPS a benchmark-owned local worker: it is a TRANSIENT
        // Turbotune candidate (the serve gate routed this request elsewhere — remote/JIT
        // — because the id was not publicly resident when it resolved), and the bench
        // unloads it between candidates, so streaming from it would serve the wrong
        // config or die mid-stream. Closes the gate→dispatch re-resolution race (codex
        // r9). The bench's OWN measurement (`serve_public == false`) is exempt so it can
        // time its candidate.
        let resolved = self
            .local_served()
            .await
            .remove(&model)
            .filter(|(_, raw_model)| !(serve_public && self.is_benchmarking(raw_model)));
        if let Some((worker, raw_model)) = resolved {
            // Admission gate: bound concurrent in-flight PUBLIC inference so the
            // no-auth server can't be flooded. A full gate is a capacity signal
            // (HTTP 503), not a failure — the client may retry. The owned permit is
            // moved into the spawned generation task below so it is held for the
            // WHOLE request (queue wait + generation) and released on any outcome.
            // The bench's OWN measurement (`serve_public == false`) does NOT take a
            // permit: it must not compete with public chats on OTHER models for the
            // slots — a full gate would misread as an HG065 "candidate failure" and
            // skew `pick_benchmarked` (Fable r1). It needs no bound of its own: the
            // benchmark is serialized per model ([HG067]/[HG068]) and measures ONE
            // candidate at a time.
            let permit = if serve_public {
                Some(
                    Arc::clone(&self.inference_gate)
                        .try_acquire_owned()
                        .map_err(|_| HiggsError::ServerBusy {
                            in_flight: MAX_CONCURRENT_INFERENCE,
                            max: MAX_CONCURRENT_INFERENCE,
                        })?,
                )
            } else {
                None
            };
            // Lease the worker's Supervisor for the whole generation. The lease stamps
            // the worker's last-activity on acquire and (on drop) re-stamps + drops the
            // in-flight reference, so the node's idle reaper never unloads a worker
            // mid-chat. A dead/unloaded worker here is a mapped error.
            let lease = self.local.chat_handle(worker).await?;
            // Apply the model's tuned/card-recommended sampling as the BASE, then
            // overlay the per-request fields the client sent. This is where "HF-card
            // recommended sampling actually applies": a `tune` persisted the
            // recommendation under the RAW model id in `models.json`, and a plain
            // OpenAI chat (usually only `temperature`) inherits the rest (top_k/min_p/
            // penalties/…). No stored profile ⇒ the request stands alone. Keyed by the
            // RAW model (the store's key), not the ephemeral served suffix id.
            let stored = self
                .models_store()
                .ok()
                .and_then(|s| s.tuning(&raw_model))
                .map(|t| t.sampling);
            let merged = overlay_sampling(stored, sampling);
            // Mint the request, register its keyed sink, and obtain a future that
            // drives the M_CHAT RPC to completion and removes the sink on any outcome —
            // all of it lives in `Supervisor::chat` (reached via the lease's `Deref`).
            // `rx` is returned to the caller now; `call` (and the lease that keeps the
            // worker alive) ride the spawned generation task with the admission permit.
            let (rx, call) = lease.chat(
                raw_model,
                messages_json,
                max_tokens,
                merged,
                tools_json,
                chat_template_kwargs,
            );
            let handle = tokio::spawn(async move {
                // Hold the admission permit AND the lease for the whole generation;
                // dropping them here (on any return path) releases the gate slot and
                // ends the worker's in-flight hold. Bound to names so neither drops early.
                let _permit = permit;
                let _lease = lease;
                let result = call.await;

                Ok(chat_outcome_from_value(&result?))
            });
            return Ok((rx, handle));
        }

        // REMOTE next: not local, so if a fleet is installed and this model lives on a
        // remote node, relay the chat over that node's transport. A remote model uses
        // NEITHER a local worker NOR the local node's idle timer, and uses a SEPARATE
        // admission gate (so a flood of remote traffic can't grow hub/node tasks
        // unbounded, and never blocks the local idle reaper). Bind the clone to a `let`
        // so the parking_lot guard drops HERE, not held across the `.await` (an `if let`
        // scrutinee temporary would make this !Send).
        let fleet = self.fleet.lock().clone();
        if let Some(fleet) = fleet {
            if fleet.is_remote(&model).await {
                let permit = Arc::clone(&self.remote_gate)
                    .try_acquire_owned()
                    .map_err(|_| HiggsError::ServerBusy {
                        in_flight: MAX_CONCURRENT_INFERENCE,
                        max: MAX_CONCURRENT_INFERENCE,
                    })?;
                // Remote sampling forwarding is DEFERRED (like remote load-param
                // forwarding): the node relay carries only temperature today. Extract
                // it from the umbrella; the rest of the sampler set applies locally.
                let temperature = sampling.as_llamacpp().temperature.unwrap_or(0.7);
                let (rx, fut) = fleet
                    .chat(
                        &model,
                        messages_json,
                        max_tokens,
                        temperature,
                        tools_json,
                        chat_template_kwargs,
                    )
                    .await?;
                let handle = tokio::spawn(async move {
                    let _permit = permit; // held for the whole remote generation
                    Ok(chat_outcome_from_value(&fut.await?))
                });
                return Ok((rx, handle));
            }
        }

        // Neither local nor remote serves this id (the serve layer JIT-loads a
        // scanned-but-unloaded model BEFORE calling this, so reaching here means the
        // model was unloaded out from under the request).
        Err(HiggsError::ModelNotFound { id: model })
    }

    /// Resolve the live LOCAL served-id → `(worker, raw model)` map (P4b). N local
    /// workers serving the same raw model coexist as N served ids
    /// (`org/model`, `org/model-1`, …) via the shared
    /// [`served_ids`](crate::node::served::served_ids) algorithm; a chat for a
    /// served id leases its worker and sends the RAW model on the wire.
    async fn local_served(&self) -> std::collections::HashMap<String, (WorkerId, String)> {
        let instances = self.local.instances().await;
        // All local instances share one location marker `()` — collisions are
        // resolved across workers exactly as the remote fleet resolves across nodes.
        let located: Vec<((), WorkerId, String)> =
            instances.iter().map(|(w, m)| ((), *w, m.clone())).collect();
        let by_worker: std::collections::HashMap<WorkerId, String> =
            instances.into_iter().collect();
        crate::node::served::served_ids(&located)
            .into_iter()
            .filter_map(|(served, ((), worker))| {
                by_worker
                    .get(&worker)
                    .map(|raw| (served, (worker, raw.clone())))
            })
            .collect()
    }

    /// Every LOCAL served model id, sorted — the input to `/v1/models` (joined with
    /// the fleet's remote ids by the serve layer).
    pub async fn local_served_ids(&self) -> Vec<String> {
        let mut ids: Vec<String> = self.local_served().await.into_keys().collect();
        ids.sort();
        ids
    }

    /// The LOCAL machine as a first-class [`NodeView`] (`endpoint_id = "local"`, `is_local`,
    /// always connected), so the serve layer can list it alongside the remote fleet in one
    /// shape. Each resident worker is tagged with its `/v1` served id, exactly as the fleet tags
    /// remote workers. `label` is the instance's `config.json` name, passed in by the caller.
    ///
    /// Hardware is sampled fresh (CPU/RAM, ~one update interval) but GPUs are deliberately NOT
    /// enumerated here — that would spawn a transient worker on every poll of this endpoint. The
    /// local machine's full hardware (incl. GPUs) is already served by `GET /api/higgs/system`.
    pub async fn local_node_view(&self, label: String) -> crate::node::fleet::NodeView {
        // ONE instance snapshot drives BOTH the served-id map and the worker list, so a
        // concurrent local load/unload can't make `/api/higgs/nodes` disagree with `/v1/models`
        // (a worker shown with a stale/empty served id). Same `served_ids` algorithm as
        // `local_served`/`local_served_ids`, applied to this single snapshot — no worker spawn.
        let instances = self.local.instances().await;
        let located: Vec<((), WorkerId, String)> =
            instances.iter().map(|(w, m)| ((), *w, m.clone())).collect();
        let by_worker: std::collections::HashMap<u32, String> =
            crate::node::served::served_ids(&located)
                .into_iter()
                .map(|(served, ((), worker))| (worker.0, served))
                .collect();
        let workers: Vec<InventoryWorker> = instances
            .into_iter()
            .map(|(worker, model)| InventoryWorker {
                worker_id: worker.0,
                model,
                served_id: by_worker.get(&worker.0).cloned().unwrap_or_default(),
            })
            .collect();
        // CPU/RAM/engine snapshot with NO GPU enumeration (empty `gpus` → no transient worker).
        let (hardware, runtime) = tokio::task::spawn_blocking(|| {
            crate::system::SystemInfo::gather_hardware_runtime(vec![])
        })
        .await
        .expect("hardware sample task");
        let inventory = NodeInventory {
            hostname: crate::system::hostname(),
            os: std::env::consts::OS.to_string(),
            workers,
            hardware,
            runtime,
        };
        crate::node::fleet::NodeView {
            node_id: 0, // remote NodeIds start at 1; 0 is the local sentinel
            endpoint_id: "local".to_string(),
            connected: true,
            is_local: true,
            label,
            inventory: Some(inventory),
        }
    }

    /// [`LoadedInfo`] for a LOCAL served id — resolves served → worker, reads that
    /// worker's status, and enriches it from the host scan. The reported `id` is the
    /// SERVED id (not the raw model) so a follow-up [`chat_stream`](Self::chat_stream)
    /// resolves the SAME worker. `None` when the id is not locally served. Used by
    /// the serve layer's loaded-model gate ([`ensure_loaded`](crate::serve)).
    ///
    /// CRITICAL: a served id that RESOLVES is resident, so a failed/timed-out status
    /// probe must NOT be reported as "not served" — that would mislead the serve gate
    /// into a not-loaded `[HG003]` (or a JIT no-op then HG003) for a model that is
    /// merely BUSY. A single worker serialises generation and cannot answer `M_STATUS`
    /// mid-generation, so the probe times out under concurrent load. The three cases:
    ///   * status OK + a loaded model → live params.
    ///   * status OK but `loaded` is null → the worker was unloaded out from under us
    ///     (a race vs idle-reap/unload) → genuinely not resident → `None` (the gate's
    ///     JIT path then reloads it).
    ///   * status ERR (busy / briefly wedged) → a permissive metadata stub (resident,
    ///     `ctx_len = u32::MAX` so the host prompt-fit gate doesn't reject) so the chat
    ///     QUEUES behind the generation; the worker's tokenizer-exact `[HG005]` stays
    ///     the authoritative prompt-fit backstop.
    pub async fn local_loaded_info(&self, served: &str) -> Option<LoadedInfo> {
        let (worker, raw) = self.local_served().await.remove(served)?;
        let scan = self.scan().await.unwrap_or_default();
        match self.local.status(worker).await {
            Ok(v) => {
                let mut info = self.loaded_info_from(&v, &scan)?;
                info.id = served.to_owned();
                info.worker_id = worker.0;
                Some(info)
            }
            Err(_) => {
                // Resident but the status probe failed (busy mid-generation): the stub's
                // params come from the PERSISTED load record (what the worker was actually
                // loaded with), so the prompt-fit gate sees the real context window; with
                // no surviving record they're `None`, which the gate treats as permissive —
                // the chat QUEUES behind the generation and the worker's [HG005] stays
                // authoritative. `None` replaced the old `ctx_len = u32::MAX` sentinel.
                Some(self.loaded_info_stub(
                    served.to_owned(),
                    worker.0,
                    &raw,
                    &scan,
                    &self.model_records(),
                ))
            }
        }
    }

    /// Subscribe to worker lifecycle events (ModelLoaded/ModelUnloaded) from the
    /// local node's event fan-out.
    pub fn events(&self) -> broadcast::Receiver<HiggsEvent> {
        self.local.events()
    }

    /// Return up to `n` recent Developer-Log lines (oldest first), optionally
    /// restricted to one [`LogSource`] (`None` = worker stderr + serve events).
    pub fn logs(&self, n: usize, filter: Option<LogSource>) -> Vec<String> {
        self.local.bus().snapshot(n, filter)
    }

    /// Subscribe to live Developer-Log lines pushed after this call. The SSE
    /// log-stream handler pairs this with [`logs`](Self::logs) for
    /// replay-then-live delivery; filter each by [`LogLine::source`].
    pub fn subscribe_logs(&self) -> tokio::sync::broadcast::Receiver<LogLine> {
        self.local.bus().subscribe()
    }

    /// Subscribe to live model-load lifecycle events ([`ModelLoadEvent`]) pushed
    /// AFTER this call — the source for the `GET /api/higgs/events` SSE endpoint.
    /// Unlike the log bus there is no replay ring: the loading bar is transient, so
    /// a subscriber that joins mid-load still receives every REMAINING phase plus the
    /// terminal `Ready`/`Failed` that closes it out.
    pub fn subscribe_load_events(
        &self,
    ) -> tokio::sync::broadcast::Receiver<crate::api::types::ModelLoadEvent> {
        self.load_events.subscribe()
    }

    /// Push one model-load lifecycle event to live SSE subscribers. A `send` with no
    /// subscribers is a harmless no-op (the load proceeds regardless). `code` is set
    /// only for [`ModelLoadPhase::Failed`].
    fn emit_load_phase(
        &self,
        id: &str,
        phase: crate::api::types::ModelLoadPhase,
        code: Option<String>,
    ) {
        let _ = self.load_events.send(crate::api::types::ModelLoadEvent {
            id: id.to_owned(),
            phase,
            at_ms: now_unix_ms(),
            code,
        });
    }

    /// Snapshot of the configured default load parameters.
    ///
    /// The serve router uses this to fill fields absent from a partial
    /// load request — config stays the single home for the defaults.
    pub(crate) fn default_load(&self) -> LoadParams {
        self.config.lock().default_load.clone()
    }

    /// Read-only snapshot of the effective server config for the `system`
    /// control-op. Clones the scan dirs and load defaults from the live
    /// [`HiggsConfig`]; `bind_host` is the RECORDED live listener address
    /// (`ip:port`, captured by `serve_v1`) when serving, else the built-in
    /// loopback default [`BIND_HOST`]. Pure read — no worker RPC, no mutation.
    pub fn server_config(&self) -> HiggsServerConfig {
        let cfg = self.config.lock();
        let to_strings = |dirs: &[PathBuf]| dirs.iter().map(|p| p.display().to_string()).collect();
        HiggsServerConfig {
            // The embedder owns the listener (loopback or 0.0.0.0 for LAN), so
            // disclose what's ACTUALLY bound once serving — not a loopback claim.
            bind_host: self
                .bound_addr()
                .map(|a| a.to_string())
                .unwrap_or_else(|| BIND_HOST.to_owned()),
            lmstudio_dirs: to_strings(&cfg.lmstudio_dirs),
            hf_dirs: to_strings(&cfg.hf_dirs),
            ollama_dirs: to_strings(&cfg.ollama_dirs),
            default_load: cfg.default_load.clone(),
            default_ctx_cap: DEFAULT_CTX_CAP,
            limits: HiggsLimits {
                max_body_bytes: crate::serve::MAX_BODY_BYTES as u64,
                control_timeout_secs: crate::serve::CONTROL_TIMEOUT.as_secs(),
                chat_timeout_secs: crate::supervisor::CHAT_RPC_TIMEOUT.as_secs(),
                max_output_tokens: crate::serve::MAX_OUTPUT_TOKENS,
                max_concurrent_inference: MAX_CONCURRENT_INFERENCE as u32,
                memory_headroom_fraction: MEMORY_HEADROOM_FRACTION,
                // Report the EFFECTIVE (live, runtime-mutable) idle TTL — the same
                // value `/api/higgs/settings` returns and the node reaper enforces —
                // not a fixed constant. Defaults to `DEFAULT_IDLE_TTL` (60 min) and
                // tracks any runtime change, so `/api/higgs/system` can never disagree
                // with `/api/higgs/settings` (a prior stale 5-min constant did). The
                // stored minutes are clamped to MAX_IDLE_TTL_MINUTES at the setter, so
                // `× 60` is overflow-safe; `saturating_mul` is a belt-and-suspenders.
                idle_unload_ttl_secs: self.idle_ttl_minutes().saturating_mul(60),
            },
        }
    }

    /// Host compute devices (CPU/GPU/accel), gathered once via a transient
    /// sysinfo worker and cached. Hardware is static-ish, so a cache hit returns
    /// immediately; a miss spawns a crash-isolated transient worker, runs
    /// M_SYSINFO, caches the result, and returns it. A gather failure returns an
    /// empty `Vec` and leaves the cache empty so a later call retries — the
    /// `system` control-op still returns hardware/runtime without devices.
    ///
    /// This is the canonical home for the gathered device list; the
    /// `SystemInfo::gather` path takes it as input rather than reading FFI itself
    /// (the FFI lives in the worker, not the host).
    pub async fn sysinfo(&self) -> Vec<crate::system::GpuDevice> {
        if let Some(cached) = self.device_cache.lock().clone() {
            return cached;
        }
        let gpus = self.local.gpus().await;
        // Cache only a non-empty result: an empty list usually means the gather
        // failed (spawn/EOF/timeout), so leave the cache empty to retry later.
        if !gpus.is_empty() {
            *self.device_cache.lock() = Some(gpus.clone());
        }
        gpus
    }

    /// Full host hardware snapshot (CPU cores/RAM + the cached GPU list) for the
    /// autotune suggester. The GPU list rides [`Self::sysinfo`] (cached worker
    /// round-trip); the CPU/RAM facts come from the `sysinfo` crate locally.
    /// Blocking (a short CPU-usage sample) — offloaded to a blocking thread.
    pub async fn hardware(&self) -> HardwareInfo {
        let gpus = self.sysinfo().await;
        tokio::task::spawn_blocking(move || {
            crate::system::SystemInfo::gather_hardware_runtime(gpus).0
        })
        .await
        .expect("higgs hardware gather task panicked")
    }

    /// Open this instance's per-node `models.json` store (tuning + perf + meta
    /// cache). Lives in the same home as `config.json` — so the per-instance
    /// config-path override (unit tests) isolates the store too; else `~/.higgs`.
    pub(crate) fn models_store(&self) -> std::io::Result<JsonModelStore> {
        let home = match self.config_path.lock().clone() {
            Some(p) => p
                .parent()
                .map(std::path::Path::to_path_buf)
                .unwrap_or_else(crate::home::higgs_home),
            None => crate::home::higgs_home(),
        };
        JsonModelStore::open(&home)
    }

    /// Run the autotune suggester for a model: look up its GGUF metadata + the host
    /// hardware, derive a nominal load + sampling parameter set within the budget,
    /// and persist it as the saved profile (so the next plain load reuses it — "tune
    /// once"). It NEVER loads the model — the caller fills the editable fields with
    /// the result and loads separately. `Suggest` mode is pure + a best-effort HF
    /// card fetch; `Benchmark` (P2) falls back to `Suggest` for now.
    pub async fn tune(&self, req: TuneRequest) -> Result<TuneSuggestion, HiggsError> {
        let id = req.id.clone();
        // 1. Typed GGUF metadata from the scan.
        let model = self
            .scan()
            .await?
            .into_iter()
            .find(|m| m.id == id)
            .ok_or_else(|| HiggsError::ModelNotFound { id: id.clone() })?;
        let meta = ModelMeta::from_model(&model);
        // Capture the file signature ALONGSIDE the metadata the profile is derived
        // from — NOT at persist time below. The bounded HF-card fetch can take ~10s;
        // a GGUF swapped in that window would otherwise have its NEW signature
        // anchored to a profile tuned for the OLD file, so the JIT gate would later
        // admit a mismatched profile as fresh instead of `NeedsRetune`.
        let model_file_sig = file_sig(&model.path);

        // 2. Host hardware + budget.
        let hw = self.hardware().await;
        let budget = req.budget.clone().unwrap_or_default();

        // Pins only steer the Benchmark search (they hold a dimension fixed while
        // Turbotune measures the rest); the analytical Suggest derives freely and
        // the user edits the result in the UI. A Suggest request that carries pins
        // is contradictory — honor it as Suggest but make the ignore VISIBLE in the
        // log rather than silently dropping the caller's intent.
        if req.mode.unwrap_or_default() != TuneMode::Benchmark {
            if let Some(pins) = &req.pins {
                if *pins != crate::tune::TunePins::default() {
                    tracing::warn!(
                        id = %id,
                        "higgs: tune pins ignored in Suggest mode (pins steer the Benchmark search only)"
                    );
                }
            }
        }

        // 3. Best-effort HF-card sampling (bounded; fail-open). Skipped for ollama ids.
        let card = fetch_card_bounded(&id).await;

        // 4. Suggest — always a FRESH derive within the budget (a prior saved profile
        //    is reused at the LOAD seam, not here, so a re-tune honors a new budget).
        //    (Benchmark is P2 — treated as Suggest here.)
        let mut suggestion = match card {
            Some(sampling) => Suggester {
                derive: crate::tune::derive::HeuristicStrategy,
                vram: crate::tune::vram::StaticVramEstimator,
                ram: crate::tune::vram::StaticRamEstimator,
                sampling: crate::tune::card_sampling::StaticSamplingSource(sampling),
            }
            .suggest(&meta, &hw, &budget),
            None => Suggester::static_default().suggest(&meta, &hw, &budget),
        };
        // 5. Benchmark (Turbotune, G6): actually LOAD + MEASURE the candidate configs
        //    and keep the FASTEST, upgrading the analytical suggestion to a MEASURED
        //    `Bench` profile. The benchmark is EXCLUSIVE: it refuses to start if the
        //    model is loaded ([HG067]) or already benchmarking ([HG068]), and while it
        //    runs every public load/chat/unload for the model is refused ([HG068]), so
        //    no concurrent op can contaminate the measurement. Every candidate failing
        //    surfaces [HG063]. Suggest mode skips this entirely.
        let mut bench_tps = None;
        if req.mode.unwrap_or_default() == TuneMode::Benchmark {
            let pins = req.pins.clone().unwrap_or_default();
            let (benchmarked_load, bench) = self
                .turbotune_bench(&id, &meta, &hw, &budget, &suggestion, &pins)
                .await?;
            // The measured benchmarked config may differ from the analytical seed on SEVERAL
            // dimensions (a pinned context/KV type, the winning KV-quant rung, a
            // half-offload rung), so recompute the fit AND replace the rationale —
            // otherwise the response's fit numbers and its narrated context/fit
            // describe the seed, not the config actually being saved.
            // `benchmarked_fit_reports` normalizes an `Auto` context to the node cap
            // (as the Phase-1 `bench_fit` filter does), so a ctx=Auto pin on a
            // long-context model does not report a false Overflow for a benchmarked
            // that loaded fine. The persisted `load` keeps the benchmarked config verbatim.
            let sampling_refined = suggestion.provenance == crate::tune::TuneProvenance::Card;
            let (vram_fit, ram_fit) =
                crate::tune::vram::benchmarked_fit_reports(&benchmarked_load, &meta, &hw, &budget);
            suggestion.vram_fit = vram_fit;
            suggestion.ram_fit = ram_fit;
            suggestion.rationale = crate::tune::benchmarked_rationale(
                &benchmarked_load,
                meta.ctx_train,
                vram_fit,
                ram_fit,
                bench.gen_tps,
                sampling_refined,
            );
            suggestion.load = LoadParams::llamacpp(benchmarked_load);
            suggestion.provenance = crate::tune::TuneProvenance::Bench;
            bench_tps = Some(bench.gen_tps);
        }

        // 6. Persist as the saved profile so the next plain load reuses it. Serialize
        //    through `models_io` with a FRESH re-read so two concurrent tunes of
        //    different models don't clobber each other's record (whole-file rewrite).
        {
            let _guard = self.models_io.lock();
            // The readiness gate makes a PERSISTED profile a serving precondition,
            // so a Prepare that can't save must FAIL loudly — otherwise the client
            // sees a "successful" tune followed by a refused `model_not_prepared`
            // chat (unwritable HIGGS_HOME, disk full, …). BOTH the store-open and
            // the flush are surfaced as errors, not silently skipped.
            let persist = |source: std::io::Error| HiggsError::PersistenceFailed {
                store: "models".into(),
                path: "models.json".into(),
                source,
            };
            let store = self.models_store().map_err(persist)?;
            store.put_tuning(
                &id,
                TuneRecord {
                    profile: suggestion.load.clone(),
                    sampling: suggestion.sampling.clone(),
                    budget: suggestion.budget.clone(),
                    provenance: suggestion.provenance,
                    bench_tps,
                    tuned_at_ms: now_unix_ms(),
                    // Staleness anchors: the hardware + model file this profile
                    // was derived against.
                    hw_fingerprint: hw.fingerprint(),
                    model_file_sig: model_file_sig.clone(),
                },
            );
            if let Err(e) = store.flush() {
                let pe = persist(e);
                tracing::warn!(id, error = %pe, "higgs: failed to persist tuning");
                return Err(pe);
            }
        }
        Ok(suggestion)
    }

    /// Live readiness for one scanned model on this node. Gathers profile
    /// presence + staleness + residency + serving, and evaluates the (heavier)
    /// Catalog models that are SERVABLE right now — prepared (fresh profile),
    /// fits free resources, serving on, not already resident. These are valid
    /// JIT chat targets, so `/v1/models` advertises them alongside resident
    /// models (an OpenAI client may pick any listed id and chat will serve it).
    /// Best-effort: any scan/store/hardware failure yields an empty list rather
    /// than failing the listing (resident models are still returned by the caller).
    pub async fn servable_model_ids(&self) -> Vec<String> {
        let Ok(models) = self.scan().await else {
            return Vec::new();
        };
        let Ok(tuning) = self.tuning_records() else {
            return Vec::new();
        };
        let loaded_set: Vec<String> = self
            .local
            .instances()
            .await
            .into_iter()
            .map(|(_, m)| m)
            .collect();
        let hw = self.hardware().await;
        models
            .into_iter()
            .filter(|m| {
                let (readiness, _) = self.model_readiness(m, &loaded_set, &hw, &tuning);
                readiness == crate::serve::readiness::ModelReadiness::Servable
            })
            .map(|m| m.id)
            .collect()
    }

    /// resource fit ONLY when the derived state would depend on it — a profiled,
    /// fresh, non-resident model with serving on. See [`crate::serve::readiness`].
    /// Sync + cheap per row: it reuses the caller's `hw` snapshot and the
    /// already-scanned `model`, so listing N models costs ONE scan + ONE hw sample.
    pub(crate) fn model_readiness(
        &self,
        model: &HiggsModel,
        loaded_set: &[String],
        hw: &crate::system::HardwareInfo,
        tuning: &std::collections::BTreeMap<String, crate::tune::store::TuneRecord>,
    ) -> (
        crate::serve::readiness::ModelReadiness,
        Option<crate::serve::wire::ModelFit>,
    ) {
        use crate::serve::readiness::{derive_readiness, ReadinessInputs};
        let loaded = loaded_set.iter().any(|id| id == &model.id);
        // The whole tuning set is loaded ONCE by the caller — an in-memory lookup
        // here, no per-model `models.json` reopen.
        let rec = tuning.get(&model.id);
        let profiled = rec.is_some();
        let stale = rec.is_some_and(|r| profile_stale(r, &model.path, hw));
        let serving = self.serving_enabled();
        // The fit is computed (and surfaced) only when a profile is actually
        // evaluated for serving — i.e. the `servable`/`unservable` branch.
        let (fits, fit) = match rec {
            Some(rec) if !stale && !loaded && serving => {
                let (fits, detail) = self.profile_fit(model, rec, hw);
                (fits, Some(detail))
            }
            _ => (false, None),
        };
        let readiness = derive_readiness(&ReadinessInputs {
            on_disk: true,
            profiled,
            stale,
            loaded,
            fits,
            serving,
        });
        (readiness, fit)
    }

    /// Does the saved profile for `model` fit the resources free **right now**?
    /// Computes the footprint from the already-scanned metadata + the caller's
    /// `hw` snapshot (NO re-scan, NO hardware re-sample), then compares each
    /// pool's needed bytes against CURRENT free VRAM/RAM (`vram_free_bytes` per
    /// GPU; RAM total minus used) — not totals or the tune-time budget — so
    /// `Servable` reflects what can actually load given other resident models.
    /// Returns `(fits, detail)`: the verdict plus the needed-vs-free numbers the
    /// UI shows as the `Servable`/`Unservable` gap.
    fn profile_fit(
        &self,
        model: &HiggsModel,
        rec: &crate::tune::store::TuneRecord,
        hw: &crate::system::HardwareInfo,
    ) -> (bool, crate::serve::wire::ModelFit) {
        let lc = rec.profile.as_llamacpp();
        let req = EstimateRequest {
            id: model.id.clone(),
            ctx_len: lc.ctx_len,
            gpu_layers: Some(lc.gpu_layers),
            type_k: lc.type_k,
            type_v: lc.type_v,
            offload_kqv: lc.offload_kqv,
            cpu_moe: lc.cpu_moe,
            budget: Some(rec.budget.clone()),
        };
        let report = estimate_footprint(&ModelMeta::from_model(model), hw, &req);
        // GPU-only free (a CPU/accel device reports system memory as its "vram").
        //
        // RESIDUAL (assessed, deferred): `free_vram` comes from the cached device
        // snapshot (`Higgs::sysinfo`), whose `vram_free_bytes` can lag a recent VRAM
        // change. Refreshing it spawns a TRANSIENT sysinfo worker per call — too
        // costly for the 5s `/api/higgs/models` poll — so the snapshot is reused.
        // `Servable`/`Unservable` is therefore an ADVISORY badge: it degrades
        // safely because the load path is gated independently (`profile_state` does
        // not use free VRAM), so a genuinely-too-tight load fails with a real
        // engine error rather than being wrongly admitted by a stale badge.
        let free_vram = hw.free_vram_bytes();
        let free_ram = hw.ram_total_bytes.saturating_sub(hw.ram_used_bytes);
        let fits = crate::serve::readiness::footprint_fits_free(
            report.vram.needed_bytes,
            report.ram.needed_bytes,
            free_vram,
            free_ram,
            hw.is_unified_memory(),
        );
        (
            fits,
            crate::serve::wire::ModelFit {
                needed_vram_bytes: report.vram.needed_bytes,
                needed_ram_bytes: report.ram.needed_bytes,
                free_vram_bytes: free_vram,
                free_ram_bytes: free_ram,
            },
        )
    }

    /// Classify the saved profile for `id` for the JIT gate: `Missing` (never
    /// Prepared), `Stale` (hardware or model file changed since Prepare), or
    /// `Ready`. The gate refuses `Missing`/`Stale` rather than load dumb defaults.
    pub(crate) async fn profile_state(&self, id: &str) -> Result<ProfileState, HiggsError> {
        // A store-OPEN failure (models.json unreadable — a directory, bad perms)
        // is a persistence fault, NOT an absent profile: surface it as HG040 so the
        // JIT gate doesn't mislead the user into re-Preparing (which would also fail
        // to persist). An absent *record* in a readable store is genuinely Missing.
        let store = self
            .models_store()
            .map_err(|e| HiggsError::PersistenceFailed {
                store: "models".into(),
                path: "models.json".into(),
                source: e,
            })?;
        let Some(rec) = store.tuning(id) else {
            return Ok(ProfileState::Missing);
        };
        // Resolve the on-disk path. If the model is NOT in the scan (its GGUF was
        // removed) or the scan failed, do NOT classify the profile as stale —
        // staleness is only meaningful for a present file, and re-tuning can't fix
        // a missing model. Return `Ready` so the actual load surfaces the real
        // not-found/scan error instead of a misleading `HG047` "Re-tune". (The JIT
        // path already rejects unknown ids with `ModelNotFound` before this.)
        let Some(path) = self
            .scan()
            .await
            .ok()
            .and_then(|ms| ms.into_iter().find(|m| m.id == id).map(|m| m.path))
        else {
            return Ok(ProfileState::Ready(rec.profile));
        };
        let hw = self.hardware().await;
        let stale = profile_stale(&rec, &path, &hw);
        Ok(if stale {
            ProfileState::Stale
        } else {
            ProfileState::Ready(rec.profile)
        })
    }

    /// Estimate the VRAM/RAM footprint of a CANDIDATE load (no model load). Reuses
    /// the suggester's own VRAM/RAM estimators — the single source of truth for the
    /// formula — so the UI shows the live cost ("≈ X GiB VRAM · verdict") as the user
    /// edits context / KV types / GPU offload. Pure + cheap: a GGUF-metadata lookup +
    /// the cached hardware read, no worker round-trip beyond that.
    pub async fn estimate(&self, req: EstimateRequest) -> Result<EstimateReport, HiggsError> {
        // Resolve the GGUF metadata, memoized per-id: the UI hits this on every edit,
        // so re-scanning all model dirs each keystroke would be wasteful on large
        // libraries. A fresh cache hit for the same model skips the scan; an expired
        // entry (TTL) re-scans so a deleted/replaced model self-heals to 404/current.
        let now = std::time::Instant::now();
        let meta = {
            let hit = self.estimate_meta_cache.lock().clone();
            match hit {
                Some((id, meta, at))
                    if id == req.id && now.duration_since(at) < ESTIMATE_META_TTL =>
                {
                    meta
                }
                _ => {
                    let model = self
                        .scan()
                        .await?
                        .into_iter()
                        .find(|m| m.id == req.id)
                        .ok_or_else(|| HiggsError::ModelNotFound { id: req.id.clone() })?;
                    let meta = ModelMeta::from_model(&model);
                    *self.estimate_meta_cache.lock() = Some((req.id.clone(), meta.clone(), now));
                    meta
                }
            }
        };
        let hw = self.hardware().await;
        Ok(estimate_footprint(&meta, &hw, &req))
    }
}

/// Pure footprint computation shared by [`Higgs::estimate`] (live UI) and the
/// readiness fit check ([`Higgs::profile_fits`]): given already-resolved GGUF
/// metadata + a hardware snapshot + a candidate request, return the VRAM/RAM
/// fit. No I/O and no hardware sampling — the caller supplies `meta` and `hw`,
/// so listing many models costs ONE scan + ONE hardware sample total (not one
/// per model). Reuses the suggester's own estimators (the single formula home).
fn estimate_footprint(
    meta: &ModelMeta,
    hw: &crate::system::HardwareInfo,
    req: &EstimateRequest,
) -> EstimateReport {
    // Resolve `Auto` the SAME way the load path does (the node caps an auto/trained
    // context at `DEFAULT_CTX_CAP`) so the estimate matches what would actually load.
    let ctx_len = crate::tune::vram::resolve_estimate_ctx(req.ctx_len, meta.ctx_train);
    let load = crate::worker::engine::llamacpp::params::LlamaCppParams {
        ctx_len,
        gpu_layers: req
            .gpu_layers
            .unwrap_or(crate::worker::engine::GpuLayers::All),
        type_k: Some(
            req.type_k
                .unwrap_or(crate::worker::engine::KvCacheKind::F16),
        ),
        type_v: Some(
            req.type_v
                .unwrap_or(crate::worker::engine::KvCacheKind::F16),
        ),
        // The remaining memory-affecting params so the verdict matches the real
        // load (KV-on-CPU and MoE-experts-on-CPU both shift bytes VRAM↔RAM).
        offload_kqv: req.offload_kqv,
        cpu_moe: req.cpu_moe,
        ..Default::default()
    };
    // Verdict against the caller's budget when supplied (so the live estimate
    // agrees with the budget-aware tune), else the detected machine.
    let budget = req.budget.clone().unwrap_or_default();
    EstimateReport {
        vram: crate::tune::vram::StaticVramEstimator.estimate(&load, meta, hw, &budget),
        ram: crate::tune::vram::StaticRamEstimator.estimate(&load, meta, hw, &budget),
    }
}

/// Best-effort, time-bounded HF-card sampling fetch (fail-open → `None`). The
/// fetch goes through the hub client (`src/hub.rs`) which prefers the structured
/// `generation_config.json` and falls back to README prose (and to a direct
/// `reqwest` GET if the hub client itself fails). The whole thing — up to two
/// files across two transports — is bounded here so a slow huggingface.co can
/// never stall a tune; on timeout we proceed with no recommendation.
async fn fetch_card_bounded(
    id: &str,
) -> Option<crate::worker::engine::llamacpp::params::LlamaCppSamplingParams> {
    let fetch = crate::tune::card_sampling::fetch_card_sampling(id);
    match tokio::time::timeout(std::time::Duration::from_secs(10), fetch).await {
        Ok(res) => res,
        Err(_) => {
            tracing::warn!(
                id,
                "higgs: HF-card sampling fetch timed out; proceeding without it"
            );
            None
        }
    }
}

// ── Tests (see api/tests.rs) ──
#[cfg(test)]
mod tests;
