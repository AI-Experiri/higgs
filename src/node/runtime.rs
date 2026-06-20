//! `NodeRuntime` — the net-new multi-worker orchestrator (DESIGN-remote.md §5.4a).
//!
//! Owns a `HashMap<WorkerId, Arc<Supervisor>>` of real child workers (one Supervisor =
//! one child, reused unchanged). Control ops run against the registry; the registry lock
//! is held only for insert/get/remove, never across `.await` (the `Arc<Supervisor>` is
//! cloned out first). The per-worker unit is the existing `Supervisor` — this layer adds
//! only the registry + lifecycle, never routing multi-worker through `Higgs`.

use std::path::PathBuf;
use std::sync::Arc;

use parking_lot::Mutex;
use serde_json::{json, Value};

use crate::diagnostic::HiggsError;
use crate::log_bus::LogBus;
use crate::node::worker_id::{WorkerId, WorkerRegistry};
use crate::remote::NodeLoadParams;
use crate::supervisor::Supervisor;
use crate::worker::models::ModelStore;
use crate::worker::{M_LOAD, M_STATUS};

/// How a node spawns a fresh Supervisor. Production: `Supervisor::spawn`; tests inject a
/// fake-worker-backed Supervisor so the registry/ops can be exercised without llama.cpp.
pub(crate) type SupervisorSpawner = Box<dyn Fn(Arc<LogBus>) -> Supervisor + Send + Sync>;

/// Node configuration: the model roots the node scans (it owns its own disk) plus the
/// shared log bus.
pub struct NodeConfig {
    pub bus: Arc<LogBus>,
    pub lmstudio_dirs: Vec<PathBuf>,
    pub hf_dirs: Vec<PathBuf>,
    pub ollama_dirs: Vec<PathBuf>,
}

/// Cancellation-safety guard: a dropped `Supervisor` does NOT reap its child, so if a
/// lifecycle future is cancelled mid-await the worker would orphan (RAM/VRAM held, no id
/// to stop it). This guard fires a detached `stop()` on drop unless `commit()` defuses it.
struct StopOnDrop(Option<Arc<Supervisor>>);

impl StopOnDrop {
    fn new(sup: Arc<Supervisor>) -> Self {
        Self(Some(sup))
    }
    /// Defuse: the worker was handed off (inserted) or already stopped — don't re-stop.
    fn commit(mut self) {
        self.0 = None;
    }
}

impl Drop for StopOnDrop {
    fn drop(&mut self) {
        if let Some(sup) = self.0.take() {
            // Best-effort, detached: we can't await in Drop. `stop()` is idempotent.
            tokio::spawn(async move { sup.stop().await });
        }
    }
}

/// The node orchestrator: N concurrent Supervisors, one child each.
pub struct NodeRuntime {
    registry: Mutex<WorkerRegistry<Arc<Supervisor>>>,
    spawner: SupervisorSpawner,
    config: NodeConfig,
}

impl NodeRuntime {
    /// Production runtime (spawns real child workers).
    pub fn new(config: NodeConfig) -> Self {
        Self::with_spawner(config, Box::new(Supervisor::spawn))
    }

    /// Runtime with an injected supervisor spawner (tests).
    pub(crate) fn with_spawner(config: NodeConfig, spawner: SupervisorSpawner) -> Self {
        Self { registry: Mutex::new(WorkerRegistry::new()), spawner, config }
    }

    /// Live worker ids, ascending.
    pub fn worker_ids(&self) -> Vec<WorkerId> {
        self.registry.lock().ids()
    }

    fn get(&self, id: WorkerId) -> Result<Arc<Supervisor>, HiggsError> {
        self.registry
            .lock()
            .get(id)
            .cloned()
            .ok_or_else(|| HiggsError::WorkerDead { context: format!("no worker {id}") })
    }

    fn take(&self, id: WorkerId) -> Result<Arc<Supervisor>, HiggsError> {
        self.registry
            .lock()
            .remove(id)
            .ok_or_else(|| HiggsError::WorkerDead { context: format!("no worker {id}") })
    }

    /// Resolve `id` to its on-disk GGUF `(path, size_bytes, ctx_train)` by scanning the
    /// node's model roots. The blocking FS scan + canonicalization run on a blocking
    /// thread (like the local path) so they never stall the control-plane executor. `get`
    /// only returns cataloged models found UNDER the roots, and the resolved path must
    /// canonicalize to within a root (symlink-escape guard, [HG015]). `[HG002]` if absent.
    async fn resolve_model(&self, id: &str) -> Result<(String, u64, Option<u64>), HiggsError> {
        let id = id.to_string();
        let lmstudio = self.config.lmstudio_dirs.clone();
        let hf = self.config.hf_dirs.clone();
        let ollama = self.config.ollama_dirs.clone();
        tokio::task::spawn_blocking(move || {
            let mut store = ModelStore::default();
            store.scan(&lmstudio, &hf, &ollama)?;
            let (path, size_bytes, ctx_train) = store
                .get(&id)
                .map(|m| (m.path.clone(), m.size_bytes, m.ctx_train))
                .ok_or_else(|| HiggsError::ModelNotFound { id: id.clone() })?;
            let roots: Vec<PathBuf> =
                lmstudio.into_iter().chain(hf).chain(ollama).collect();
            if !crate::api::path_within_roots(&path, &roots) {
                return Err(HiggsError::InvalidModelId {
                    id,
                    reason: format!("resolved path {path} is outside every configured scan directory"),
                });
            }
            Ok((path, size_bytes, ctx_train))
        })
        .await
        .map_err(|e| HiggsError::WorkerDead { context: format!("model scan task failed: {e}") })?
    }

    /// Spawn a NEW worker for `params.id` and load the model (net-new multi-worker). The
    /// node may already host other workers; this does NOT replace them. The node resolves
    /// the GGUF path from its own disk (the hub only sends the id) and runs the same
    /// pre-spawn RAM headroom guard as the local path ([HG017]) before bringing a worker
    /// up. Returns the new `WorkerId` and the worker's `M_LOAD` result (`loaded`).
    ///
    /// NOTE: the cross-worker VRAM fit-check (§4.2b, summing resident workers) is wired in
    /// P2 Task 6 alongside real device info; the RAM headroom guard here is the existing
    /// local capacity check, reused.
    pub async fn load(&self, params: NodeLoadParams) -> Result<(WorkerId, Value), HiggsError> {
        let (path, size_bytes, ctx_train) = self.resolve_model(&params.id).await?;
        // Reject before spawning a worker if the model can't fit (mirrors Higgs::load).
        crate::api::guard_memory_headroom(&params.id, size_bytes)?;
        self.spawn_and_load(&path, ctx_train, &params).await
    }

    /// Spawn a fresh Supervisor, start its child, and send `M_LOAD` with the host-resolved
    /// `path`. On load failure the worker is torn down (never leaked, never orphaned in
    /// the registry); on success it is inserted and its id returned.
    async fn spawn_and_load(
        &self,
        path: &str,
        ctx_train: Option<u64>,
        params: &NodeLoadParams,
    ) -> Result<(WorkerId, Value), HiggsError> {
        let sup = Arc::new((self.spawner)(self.config.bus.clone()));
        // Until the worker is committed to the registry, any early return — error OR
        // cancellation (hub disconnect/timeout while M_LOAD awaits) — must reap it.
        let guard = StopOnDrop::new(sup.clone());
        sup.start_for(&params.id)?;
        // When the caller omits ctx_len, default to the model's trained context capped at
        // DEFAULT_CTX_CAP (mirrors Higgs::load) rather than the worker's hardcoded 4096,
        // so large-context models loaded via a node accept long prompts.
        let ctx_len = params
            .ctx_len
            .or_else(|| ctx_train.map(|t| (t as u32).min(crate::api::DEFAULT_CTX_CAP)));
        // No `idle_ttl_minutes` here: the worker ignores it (the local path doesn't send
        // it to the worker either — idle TTL is a host/node reaper concern, pending).
        let load_params = json!({
            "id": params.id,
            "path": path,
            "ctx_len": ctx_len,
            "gpu_layers": params.gpu_layers,
            "threads": params.threads,
        });
        let loaded = sup.request(M_LOAD, load_params.clone()).await?; // err/cancel → guard reaps
        // Record the load so the Supervisor's restart FSM replays it on an unexpected
        // respawn — otherwise the replacement child would come back model-less (mirrors
        // the local `Higgs::load` path).
        sup.record_last_load(load_params);
        let id = self.registry.lock().insert(sup);
        guard.commit(); // handed off to the registry — don't reap
        Ok((id, loaded))
    }

    /// Graceful unload: stop the worker, free the id. Cancellation-safe — once removed
    /// from the registry, the worker is reaped even if this future is cancelled mid-stop.
    pub async fn unload(&self, id: WorkerId) -> Result<(), HiggsError> {
        let sup = self.take(id)?;
        let guard = StopOnDrop::new(sup.clone());
        sup.stop().await;
        guard.commit(); // stopped cleanly; no detached re-stop
        Ok(())
    }

    /// Force-kill ONE worker (at this layer the same as unload — `stop()` reaps the
    /// child; the OS-level distinction is the Supervisor's concern).
    pub async fn kill(&self, id: WorkerId) -> Result<(), HiggsError> {
        let sup = self.take(id)?;
        let guard = StopOnDrop::new(sup.clone());
        sup.stop().await;
        guard.commit();
        Ok(())
    }

    /// Per-worker status (forwards `M_STATUS` to that worker's Supervisor).
    pub async fn status(&self, id: WorkerId) -> Result<Value, HiggsError> {
        self.get(id)?.request(M_STATUS, Value::Null).await
    }

    /// Node-level model catalog (`{ "models": [HiggsModel, …] }`) from a fresh scan of the
    /// node's roots. Read-only; the blocking scan runs off the executor.
    pub async fn scan(&self) -> Result<Value, HiggsError> {
        let lmstudio = self.config.lmstudio_dirs.clone();
        let hf = self.config.hf_dirs.clone();
        let ollama = self.config.ollama_dirs.clone();
        let models = tokio::task::spawn_blocking(move || {
            let mut store = ModelStore::default();
            store.scan(&lmstudio, &hf, &ollama)?;
            Ok::<Value, HiggsError>(json!({ "models": store.models() }))
        })
        .await
        .map_err(|e| HiggsError::WorkerDead { context: format!("scan task failed: {e}") })??;
        Ok(models)
    }

    /// Node-level system info: `{ "hardware": HardwareInfo, "runtime": RuntimeInfo }`.
    /// The GPU device list is enumerated by a transient worker, then folded with sampled
    /// CPU/RAM/load into the full hardware snapshot (cpu_name, cores, RAM total/used,
    /// cpu_usage, gpus, vram) — the params the hub fleet view extracts (§4.2).
    pub async fn sysinfo(&self) -> Result<Value, HiggsError> {
        let sup = (self.spawner)(self.config.bus.clone());
        let gpus = sup.sysinfo().await;
        let (hardware, runtime) =
            tokio::task::spawn_blocking(move || crate::system::SystemInfo::gather_hardware_runtime(gpus))
                .await
                .map_err(|e| HiggsError::WorkerDead { context: format!("sysinfo task failed: {e}") })?;
        Ok(json!({ "hardware": hardware, "runtime": runtime }))
    }

    /// Look up a worker's Supervisor for the data-plane chat relay (P2 Task 5).
    #[allow(dead_code)] // consumed by the data relay in P2 Task 5
    pub(crate) fn chat_handle(&self, id: WorkerId) -> Result<Arc<Supervisor>, HiggsError> {
        self.get(id)
    }

    /// Graceful drain: stop every resident worker and empty the registry. The node daemon
    /// calls this on shutdown so committed workers are reaped (a dropped `Supervisor` does
    /// not reap its child; `Drop` below is only a best-effort backstop).
    pub async fn shutdown_all(&self) {
        let sups: Vec<Arc<Supervisor>> = {
            let mut reg = self.registry.lock();
            reg.ids().into_iter().filter_map(|id| reg.remove(id)).collect()
        };
        // Guard EVERY drained worker up front: if this future is cancelled mid-drain, the
        // still-uncommitted guards reap their children on drop (a dropped Vec<Arc> would
        // not). `stop()` is idempotent, so committing after a clean stop is safe.
        let guards: Vec<StopOnDrop> = sups.iter().cloned().map(StopOnDrop::new).collect();
        for sup in &sups {
            sup.stop().await;
        }
        for guard in guards {
            guard.commit();
        }
    }
}

impl Drop for NodeRuntime {
    fn drop(&mut self) {
        // Backstop for committed workers if `shutdown_all` wasn't called: detached,
        // best-effort (can't await in Drop, and a runtime may not exist at process exit).
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            let mut reg = self.registry.lock();
            for id in reg.ids() {
                if let Some(sup) = reg.remove(id) {
                    handle.spawn(async move { sup.stop().await });
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::test_support::fake_runtime as fake_runtime_with_dirs;

    fn fake_runtime() -> NodeRuntime {
        fake_runtime_with_dirs(vec![])
    }

    fn load_params(id: &str) -> NodeLoadParams {
        NodeLoadParams { id: id.into(), ctx_len: None, gpu_layers: None, threads: None}
    }

    // Spawn-and-load with a dummy path (the fake worker accepts any path), exercising the
    // multi-worker spawn/registry path without an on-disk GGUF.
    async fn fake_load(rt: &NodeRuntime, id: &str) -> (WorkerId, Value) {
        rt.spawn_and_load("/dev/null/fake.gguf", None, &load_params(id)).await.unwrap()
    }

    #[tokio::test]
    async fn load_assigns_ids_and_kill_frees_them() {
        let rt = fake_runtime();
        let (a, _) = fake_load(&rt, "m-a").await;
        let (b, _) = fake_load(&rt, "m-b").await;
        assert_ne!(a, b);
        assert_eq!(rt.worker_ids().len(), 2, "two concurrent workers");
        rt.kill(a).await.unwrap();
        assert_eq!(rt.worker_ids().len(), 1);
        assert!(rt.kill(a).await.is_err(), "killing a freed id errors");
    }

    #[tokio::test]
    async fn load_returns_worker_load_result() {
        let rt = fake_runtime();
        let (_, loaded) = fake_load(&rt, "org/model").await;
        assert_eq!(loaded["id"], "org/model");
    }

    #[tokio::test]
    async fn load_resolves_path_and_errors_when_model_absent() {
        // No model roots configured → resolution yields HG002 ModelNotFound, and no
        // worker is spawned.
        let rt = fake_runtime();
        let err = rt.load(load_params("missing/model")).await.unwrap_err();
        assert!(err.to_string().starts_with("[HG002]"), "got {err}");
        assert!(rt.worker_ids().is_empty(), "no worker spawned on resolve failure");
    }

    #[tokio::test]
    async fn status_forwards_to_the_worker() {
        let rt = fake_runtime();
        let (id, _) = fake_load(&rt, "m").await;
        let status = rt.status(id).await.unwrap();
        assert!(status.get("loaded").is_some());
        assert!(rt.status(WorkerId(999)).await.is_err(), "unknown worker errors");
    }

    #[tokio::test]
    async fn unload_stops_and_frees() {
        let rt = fake_runtime();
        let (id, _) = fake_load(&rt, "m").await;
        rt.unload(id).await.unwrap();
        assert!(rt.worker_ids().is_empty());
    }

    #[tokio::test]
    async fn shutdown_all_drains_every_worker() {
        let rt = fake_runtime();
        fake_load(&rt, "a").await;
        fake_load(&rt, "b").await;
        assert_eq!(rt.worker_ids().len(), 2);
        rt.shutdown_all().await;
        assert!(rt.worker_ids().is_empty(), "all workers drained");
    }
}
