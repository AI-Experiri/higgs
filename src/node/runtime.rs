//! `NodeRuntime` — the net-new multi-worker orchestrator (DESIGN-remote.md §5.4a),
//! an **actor** (P1 of `docs/superpowers/specs/2026-06-22-actor-runtime-design.md`).
//!
//! The worker registry is **private actor state** (no mutex): one owning task processes
//! a typed mailbox, so concurrent ops can't interleave across `.await` points. Per
//! `CLAUDE.md`, the actor's `handle` does only fast synchronous state work; slow
//! downstream RPCs (`resolve` + spawn + `M_LOAD`, `stop`, `M_STATUS`, sysinfo, scan,
//! inventory) run in a spawned task. `load` — the one op that mutates state AFTER a slow
//! RPC — uses the spawn-and-commit pattern: it reserves an id synchronously, runs the
//! slow load detached, then applies the registry insert via a `LoadCommit` message. So a
//! slow load never head-of-line-blocks an unload/retire.
//!
//! `NodeRuntime` is a thin handle wrapping the mailbox; its methods send a message and
//! await a oneshot reply.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::{json, Value};
use tokio::sync::{broadcast, oneshot};

use crate::actor::{spawn_actor_with, Actor, Handle, WeakHandle};
use crate::diagnostic::HiggsError;
use crate::log_bus::{LogBus, LogSource};
use crate::node::worker_id::{WorkerId, WorkerRegistry};
use crate::remote::NodeLoadParams;
use crate::supervisor::{HiggsEvent, Supervisor};
use crate::worker::models::ModelStore;
use crate::worker::{M_LOAD, M_STATUS};

/// How a node spawns a fresh Supervisor. `Arc` (not `Box`) so the spawn-and-commit load
/// task can clone it off the actor thread. Production: `Supervisor::spawn`; tests inject a
/// fake-worker-backed Supervisor so the registry/ops can be exercised without llama.cpp.
pub(crate) type SupervisorSpawner = Arc<dyn Fn(Arc<LogBus>) -> Supervisor + Send + Sync>;

/// Default idle auto-unload TTL: a worker with no chat activity for this long is reaped by
/// the node's idle reaper. 60 minutes (the uniform local+remote default); configurable per
/// node via [`NodeConfig::idle_ttl`].
pub const DEFAULT_IDLE_TTL: Duration = Duration::from_secs(60 * 60);

/// How often the idle reaper wakes, derived from the TTL: a quarter of it, clamped to
/// `[50ms, 60s]`. So a 60-min TTL polls once a minute (cheap); a tiny test TTL polls fast.
fn reap_interval(ttl: Duration) -> Duration {
    (ttl / 4).clamp(Duration::from_millis(50), Duration::from_secs(60))
}

/// Runtime-mutable idle auto-unload policy, shared (via `Arc`) between the wrapper's setters
/// and the actor's reaper, which reads it on EVERY tick — so the engine's Server-Settings
/// toggles (auto-unload on/off, TTL) take effect live without a restart, exactly as the old
/// engine-level reaper did. Plain atomics, set/read in isolation.
pub struct IdleConfig {
    /// TTL in MILLISECONDS (not seconds — sub-second TTLs are used in tests and truncating to
    /// seconds would make a 120ms TTL read back as 0 and reap instantly).
    ttl_millis: AtomicU64,
    enabled: AtomicBool,
    /// Notified on any change so the reaper can interrupt an in-progress sleep and re-evaluate
    /// at the new cadence immediately — a lowered TTL takes effect now, not after the old sleep.
    changed: tokio::sync::Notify,
}

impl IdleConfig {
    fn new(ttl: Duration) -> Self {
        Self {
            ttl_millis: AtomicU64::new(ttl.as_millis() as u64),
            enabled: AtomicBool::new(true),
            changed: tokio::sync::Notify::new(),
        }
    }
    /// Current idle TTL.
    pub fn ttl(&self) -> Duration {
        Duration::from_millis(self.ttl_millis.load(Ordering::Relaxed))
    }
    /// Set the idle TTL (live; the reaper re-evaluates immediately).
    pub fn set_ttl(&self, ttl: Duration) {
        self.ttl_millis
            .store(ttl.as_millis() as u64, Ordering::Relaxed);
        self.changed.notify_one();
    }
    /// Whether auto-unload is on.
    pub fn enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }
    /// Turn auto-unload on/off (live; the reaper re-evaluates immediately).
    pub fn set_enabled(&self, on: bool) {
        self.enabled.store(on, Ordering::Relaxed);
        self.changed.notify_one();
    }
}

/// Node configuration: the model roots the node scans (it owns its own disk), the shared log
/// bus, and the idle auto-unload TTL.
pub struct NodeConfig {
    pub bus: Arc<LogBus>,
    pub lmstudio_dirs: Vec<PathBuf>,
    pub hf_dirs: Vec<PathBuf>,
    pub ollama_dirs: Vec<PathBuf>,
    /// Idle auto-unload TTL: a worker idle (no chat) this long is reaped. [`DEFAULT_IDLE_TTL`].
    pub idle_ttl: Duration,
}

impl NodeConfig {
    /// The three model roots `(lmstudio, hf, ollama)`, cloned for a `spawn_blocking` scan
    /// that must own them. Shared by `resolve_model` and `scan`.
    fn model_dirs(&self) -> (Vec<PathBuf>, Vec<PathBuf>, Vec<PathBuf>) {
        (
            self.lmstudio_dirs.clone(),
            self.hf_dirs.clone(),
            self.ollama_dirs.clone(),
        )
    }
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

/// Capacity of the node→hub log relay broadcast. A hub that falls this far behind drops
/// the gap (Lagged) and keeps streaming — log relay is best-effort, never back-pressures
/// a worker.
const LOG_RELAY_CAP: usize = 1024;

/// Capacity of the lifecycle-event broadcast (ModelLoaded/ModelUnloaded). Small — events are
/// discrete and consumers (UI/SSE) keep up; a lagging consumer drops the gap.
const EVENT_CAP: usize = 256;

/// Per-path model-support probe verdicts: `(path, (loadable, reason, engine_version))`.
/// Mirrors `Supervisor::probe_paths`' return shape.
pub type ProbeVerdicts = Vec<(String, (bool, Option<String>, String))>;

/// The actor's typed mailbox. Each variant carries a oneshot `reply` the wrapper awaits.
enum NodeMsg {
    /// Spawn a new worker + load the model. Reserves an id synchronously, runs the slow
    /// load detached, commits via `LoadCommit`.
    Load {
        params: NodeLoadParams,
        reply: oneshot::Sender<Result<(WorkerId, Value), HiggsError>>,
    },
    /// Follow-up from the detached load task: apply the registry insert (or reap on
    /// failure/cancellation) and answer the original caller.
    LoadCommit {
        id: WorkerId,
        result: Result<(Arc<Supervisor>, Value), LoadFailure>,
        reply: oneshot::Sender<Result<(WorkerId, Value), HiggsError>>,
    },
    /// Graceful unload (and `Kill`, which is identical at this layer): remove from the
    /// registry synchronously, then stop the worker on a tracked teardown task.
    Unload {
        id: WorkerId,
        reply: oneshot::Sender<Result<(), HiggsError>>,
    },
    Kill {
        id: WorkerId,
        reply: oneshot::Sender<Result<(), HiggsError>>,
    },
    Status {
        id: WorkerId,
        reply: oneshot::Sender<Result<Value, HiggsError>>,
    },
    Scan {
        reply: oneshot::Sender<Result<Value, HiggsError>>,
    },
    Sysinfo {
        reply: oneshot::Sender<Result<Value, HiggsError>>,
    },
    Inventory {
        reply: oneshot::Sender<Result<Value, HiggsError>>,
    },
    WorkerIds {
        reply: oneshot::Sender<Vec<WorkerId>>,
    },
    ChatHandle {
        id: WorkerId,
        reply: oneshot::Sender<Result<ChatLease, HiggsError>>,
    },
    /// Posted by a [`ChatLease`] on drop: the chat finished (or was abandoned). Decrements the
    /// worker's in-flight count and re-stamps its last-activity, so the idle reaper measures
    /// from the END of a generation and never reaps a worker that's still generating.
    ChatEnd {
        id: WorkerId,
    },
    /// Periodic tick from the idle reaper: unload every resident worker with no in-flight chat
    /// that has been idle past the TTL.
    ReapIdle,
    /// Snapshot the resident instances as `(worker, raw model)` — the engine's input to the
    /// global served-id derivation (P4b).
    Instances {
        reply: oneshot::Sender<Vec<(WorkerId, String)>>,
    },
    /// Probe GGUF support for `paths` via a transient Supervisor (control-plane; independent of
    /// resident workers). Mirrors `Supervisor::probe_paths`.
    Probe {
        paths: Vec<String>,
        reply: oneshot::Sender<ProbeVerdicts>,
    },
    /// Enumerate host GPUs via a transient Supervisor (control-plane). Mirrors
    /// `Supervisor::sysinfo`.
    Gpus {
        reply: oneshot::Sender<Vec<crate::system::GpuDevice>>,
    },
    ShutdownAll {
        reply: oneshot::Sender<()>,
    },
    /// Posted by a worker-stop task once `Supervisor::stop()` has finished. Decrements the
    /// in-flight-stop count (so `shutdown_all` can tell when every teardown is truly done)
    /// and fires the op's own reply if it had one (e.g. the `unload`/`kill` caller).
    ReapDone {
        done: Option<oneshot::Sender<Result<(), HiggsError>>>,
    },
}

/// A live handle to a worker's [`Supervisor`] for the duration of one chat. Deref-exposes the
/// `Supervisor` so the relay drives it directly; on drop it posts [`NodeMsg::ChatEnd`] back to
/// the actor (decrement in-flight + re-stamp activity). Holding the lease for the whole
/// generation keeps the idle reaper from unloading a worker mid-chat.
pub(crate) struct ChatLease {
    sup: Arc<Supervisor>,
    id: WorkerId,
    actor: WeakHandle<NodeMsg>,
}

impl std::ops::Deref for ChatLease {
    type Target = Supervisor;
    fn deref(&self) -> &Supervisor {
        &self.sup
    }
}

impl Drop for ChatLease {
    fn drop(&mut self) {
        // Best-effort, non-blocking: the actor may already be gone (shutdown).
        if let Some(h) = self.actor.upgrade() {
            let _ = h.send(NodeMsg::ChatEnd { id: self.id });
        }
    }
}

/// Private actor state: N concurrent Supervisors (one child each) behind a mailbox.
struct NodeActor {
    /// Owned outright — NOT behind a mutex. Only the actor task touches it.
    registry: WorkerRegistry<Arc<Supervisor>>,
    spawner: SupervisorSpawner,
    config: Arc<NodeConfig>,
    /// Fan-out of every resident worker's stderr line, tagged with its `WorkerId`, to the
    /// CURRENT hub connection's relay task. Persists across reconnects with the runtime; a
    /// line emitted while no hub is connected is dropped (no subscriber).
    log_tx: broadcast::Sender<(WorkerId, String)>,
    /// Self-reference for posting `LoadCommit`/`ReapDone` back. A `WeakHandle` so an idle
    /// actor still shuts down when the last external handle drops.
    self_handle: WeakHandle<NodeMsg>,
    /// A STRONG self-handle held ONLY while work is in flight (`inflight_loads +
    /// inflight_stops > 0`). It keeps the mailbox open across the load→commit→reap→done
    /// chain even if every external handle drops mid-op, so that chain drains in-actor
    /// (no orphan) before `on_stop` runs. Cleared the instant both counters hit 0, so an
    /// idle actor never stays alive. (`Drop`-then-immediate-`process::exit` without a
    /// `shutdown_all` is the sole residual — `Drop` can't be async; the daemon always calls
    /// `shutdown_all` first, which IS fully awaited.)
    keepalive: Option<Handle<NodeMsg>>,
    /// Set once `ShutdownAll` (or `on_stop`) begins draining. Makes shutdown terminal: a
    /// `Load` that is still in flight when shutdown starts must NOT commit a new worker
    /// afterwards (it would survive the drain), so once this is set `Load` is rejected and
    /// a late `LoadCommit` reaps its worker instead of inserting it.
    shutting_down: bool,
    /// Loads spawned but not yet committed/reaped. `ShutdownAll` waits for this to reach 0
    /// so a worker still being brought up can't outlive the drain (and orphan on process
    /// exit). Bumped when a `Load` is accepted, dropped in every `LoadCommit`.
    inflight_loads: usize,
    /// Worker-stop tasks spawned but not yet finished (`ReapDone` pending). EVERY stop —
    /// unload/kill, cancel/failure reap, and the shutdown drain — goes through `reap`, so
    /// this counts all teardown in flight. `shutdown_all` waits for this to reach 0, so it
    /// can't return while ANY worker (even one stopped by a racing unload) is still alive.
    inflight_stops: usize,
    /// `ShutdownAll` callers parked until every load AND every stop has drained.
    shutdown_waiters: Vec<oneshot::Sender<()>>,
    /// Per-worker last chat activity (stamped on load, and on chat start + end). The idle
    /// reaper unloads a worker whose entry is older than `idle_ttl` with no in-flight chat.
    last_activity: HashMap<WorkerId, Instant>,
    /// Per-worker count of in-flight chats (held via [`ChatLease`]). A worker with a non-zero
    /// count is never idle-reaped, so a generation longer than the TTL is not killed mid-chat.
    in_flight: HashMap<WorkerId, u32>,
    /// Runtime-mutable idle auto-unload policy (TTL + on/off), read by the reaper each tick.
    idle: Arc<IdleConfig>,
    /// Set once the drain has fully completed. A `ShutdownAll` arriving afterwards answers
    /// immediately (no in-flight work remains to release a fresh waiter).
    shutdown_done: bool,
    /// Lifecycle event fan-out (ModelLoaded/ModelUnloaded), so the engine's event stream works
    /// the same for a local NodeRuntime as it did for the single Supervisor (P4b).
    events_tx: broadcast::Sender<HiggsEvent>,
}

impl NodeActor {
    /// Best-effort lifecycle event emit (no subscribers ⇒ dropped).
    fn emit(&self, ev: HiggsEvent) {
        let _ = self.events_tx.send(ev);
    }

    /// Snapshot resident workers (id → model) synchronously off the registry — the fast
    /// state read that precedes the slow `hardware_runtime` fetch in `inventory`.
    fn snapshot_workers(&self) -> Vec<crate::remote::InventoryWorker> {
        self.registry
            .ids()
            .into_iter()
            .filter_map(|id| {
                self.registry
                    .get(id)
                    .map(|sup| crate::remote::InventoryWorker {
                        worker_id: id.0,
                        model: sup.loaded_model_id().unwrap_or_default(),
                    })
            })
            .collect()
    }
}

impl Actor for NodeActor {
    type Msg = NodeMsg;

    async fn handle(&mut self, msg: NodeMsg) {
        match msg {
            NodeMsg::Load { params, reply } => {
                if self.shutting_down {
                    let _ = reply.send(Err(shutting_down()));
                    return;
                }
                // Reserve the id NOW (sync), run the slow load detached, commit later.
                let id = self.registry.reserve();
                self.inflight_loads += 1;
                self.refresh_keepalive();
                let spawner = self.spawner.clone();
                let config = self.config.clone();
                let relay = self.log_tx.clone();
                let self_handle = self.self_handle.clone();
                tokio::spawn(async move {
                    let result = do_load(id, params, spawner, config, relay).await;
                    // The keepalive (held while any work is in flight) keeps the actor alive,
                    // so this upgrade succeeds and the commit is handled on-thread — UNLESS
                    // the runtime was dropped without shutdown_all (accepted Drop residual),
                    // in which case the None branch reaps best-effort.
                    match self_handle.upgrade() {
                        Some(h) => {
                            let _ = h.send(NodeMsg::LoadCommit { id, result, reply });
                        }
                        None => {
                            // Runtime gone (Drop without shutdown_all): best-effort detached
                            // reap so a built worker can't orphan.
                            let sup = match result {
                                Ok((sup, _)) => Some(sup),
                                Err(LoadFailure { sup, .. }) => sup,
                            };
                            if let Some(sup) = sup {
                                tokio::spawn(async move { sup.stop().await });
                            }
                        }
                    }
                });
            }
            NodeMsg::LoadCommit { id, result, reply } => {
                match result {
                    Ok((sup, loaded)) => {
                        if self.shutting_down {
                            // Shutdown started while this load was in flight — reap the worker
                            // (tracked, so shutdown_all awaits it) and refuse the load.
                            self.reap(sup, None);
                            let _ = reply.send(Err(shutting_down()));
                        } else {
                            // Deliver the result FIRST, then commit only if the caller is
                            // still listening. Both run synchronously in this handler (no
                            // await between), so a cancel can't wedge a committed-but-unacked
                            // worker, and any follow-up op still observes the insert. If the
                            // caller vanished mid-commit, reap (tracked) instead of keeping a
                            // worker nobody asked to keep.
                            match reply.send(Ok((id, loaded))) {
                                Ok(()) => {
                                    let model = sup.loaded_model_id().unwrap_or_default();
                                    self.registry.insert_reserved(id, sup);
                                    // Start the idle clock now (a loaded-but-unused model is
                                    // idle from load time).
                                    self.last_activity.insert(id, Instant::now());
                                    self.emit(HiggsEvent::ModelLoaded { id: model });
                                }
                                Err(_) => self.reap(sup, None),
                            }
                        }
                    }
                    Err(LoadFailure { sup, error }) => {
                        // Reserved id is a harmless gap. A worker spawned before the failure is
                        // reaped (tracked), so a racing shutdown_all still awaits its stop.
                        if let Some(sup) = sup {
                            self.reap(sup, None);
                        }
                        let _ = reply.send(Err(error));
                    }
                }
                // This load is resolved; a waiting shutdown may now be able to finish.
                self.inflight_loads -= 1;
                self.refresh_keepalive();
                self.maybe_finish_shutdown();
            }
            NodeMsg::Unload { id, reply } | NodeMsg::Kill { id, reply } => {
                match self.registry.remove(id) {
                    // Removed from the registry synchronously (already invisible); the stop is
                    // tracked so a later shutdown_all awaits it, and the caller's reply fires
                    // when the stop actually completes.
                    Some(sup) => {
                        let model = sup.loaded_model_id().unwrap_or_default();
                        self.forget_activity(id);
                        self.reap(sup, Some(reply));
                        self.emit(HiggsEvent::ModelUnloaded { id: model });
                    }
                    None => {
                        let _ = reply.send(Err(no_worker(id)));
                    }
                }
            }
            NodeMsg::Status { id, reply } => match self.registry.get(id).cloned() {
                Some(sup) => {
                    tokio::spawn(async move {
                        let _ = reply.send(sup.request(M_STATUS, Value::Null).await);
                    });
                }
                None => {
                    let _ = reply.send(Err(no_worker(id)));
                }
            },
            NodeMsg::Scan { reply } => {
                let config = self.config.clone();
                tokio::spawn(async move {
                    let _ = reply.send(do_scan(&config).await);
                });
            }
            NodeMsg::Sysinfo { reply } => {
                // The GPU probe spawns a TRANSIENT worker that `Supervisor::sysinfo` reaps
                // itself (closes stdin → waits/kills) before returning — see supervisor.rs.
                // It holds no model/VRAM and is gone by the time this task completes, so it
                // needs no shutdown-barrier tracking (unlike resident model workers).
                let spawner = self.spawner.clone();
                let bus = self.config.bus.clone();
                tokio::spawn(async move {
                    let _ = reply.send(do_sysinfo(&spawner, &bus).await);
                });
            }
            NodeMsg::Inventory { reply } => {
                // Same as Sysinfo: the hardware probe's transient worker self-reaps.
                let workers = self.snapshot_workers();
                let spawner = self.spawner.clone();
                let bus = self.config.bus.clone();
                tokio::spawn(async move {
                    let _ = reply.send(do_inventory(workers, &spawner, &bus).await);
                });
            }
            NodeMsg::WorkerIds { reply } => {
                let _ = reply.send(self.registry.ids());
            }
            NodeMsg::ChatHandle { id, reply } => {
                let lease = match self.registry.get(id).cloned() {
                    Some(sup) => {
                        // Stamp activity + take an in-flight reference for the whole chat.
                        self.last_activity.insert(id, Instant::now());
                        *self.in_flight.entry(id).or_insert(0) += 1;
                        Ok(ChatLease {
                            sup,
                            id,
                            actor: self.self_handle.clone(),
                        })
                    }
                    None => Err(no_worker(id)),
                };
                let _ = reply.send(lease);
            }
            NodeMsg::ChatEnd { id } => {
                if self.registry.get(id).is_some() {
                    if let Some(c) = self.in_flight.get_mut(&id) {
                        *c = c.saturating_sub(1);
                    }
                    // Measure idle from the END of the generation.
                    self.last_activity.insert(id, Instant::now());
                } else {
                    // Worker was unloaded mid-chat — drop its bookkeeping rather than leak it.
                    self.forget_activity(id);
                }
            }
            NodeMsg::ReapIdle => {
                // Skip while draining (shutdown reaps everything anyway) or when auto-unload
                // is turned off at runtime.
                if !self.shutting_down && self.idle.enabled() {
                    let ttl = self.idle.ttl();
                    let idle: Vec<WorkerId> = self
                        .registry
                        .ids()
                        .into_iter()
                        .filter(|id| {
                            self.in_flight.get(id).copied().unwrap_or(0) == 0
                                && self
                                    .last_activity
                                    .get(id)
                                    .is_some_and(|t| t.elapsed() >= ttl)
                        })
                        .collect();
                    for id in idle {
                        if let Some(sup) = self.registry.remove(id) {
                            let model = sup.loaded_model_id().unwrap_or_default();
                            self.forget_activity(id);
                            self.reap(sup, None);
                            self.emit(HiggsEvent::ModelUnloaded { id: model });
                        }
                    }
                }
            }
            NodeMsg::Instances { reply } => {
                let _ = reply.send(
                    self.registry
                        .ids()
                        .into_iter()
                        .filter_map(|id| {
                            self.registry
                                .get(id)
                                .map(|sup| (id, sup.loaded_model_id().unwrap_or_default()))
                        })
                        .collect(),
                );
            }
            NodeMsg::Probe { paths, reply } => {
                // Control-plane probe on a transient Supervisor — independent of resident
                // workers, so run it off the actor thread.
                let spawner = self.spawner.clone();
                let bus = self.config.bus.clone();
                tokio::spawn(async move {
                    let sup = (spawner)(bus);
                    let _ = reply.send(sup.probe_paths(paths).await);
                });
            }
            NodeMsg::Gpus { reply } => {
                // Transient Supervisor enumerates GPUs and self-reaps; off the actor thread.
                let spawner = self.spawner.clone();
                let bus = self.config.bus.clone();
                tokio::spawn(async move {
                    let sup = (spawner)(bus);
                    let _ = reply.send(sup.sysinfo().await);
                });
            }
            NodeMsg::ShutdownAll { reply } => {
                if self.shutdown_done {
                    let _ = reply.send(()); // already fully drained; nothing to wait for.
                    return;
                }
                self.shutdown_waiters.push(reply);
                if self.shutting_down {
                    return; // a drain is already running; this caller waits for the same one.
                }
                // Terminal: block any in-flight load from committing after this drain. Reap
                // every committed worker (tracked). The waiters fire once `inflight_loads`
                // AND `inflight_stops` both reach 0 — i.e. every load resolved and every stop
                // (these + any racing unload/kill + late in-flight loads) has finished.
                self.shutting_down = true;
                for sup in self.drain() {
                    self.reap(sup, None);
                }
                self.maybe_finish_shutdown(); // handle the no-workers / nothing-in-flight case
            }
            NodeMsg::ReapDone { done } => {
                self.inflight_stops -= 1;
                if let Some(done) = done {
                    let _ = done.send(Ok(()));
                }
                self.refresh_keepalive();
                self.maybe_finish_shutdown();
            }
        }
    }

    async fn on_stop(&mut self) {
        // Backstop for committed workers when the last handle is dropped without a
        // `shutdown_all`: a dropped `Supervisor` does not reap its child. Unlike `Drop` this
        // runs in async context, so we await the teardown directly. (Stops already spawned by
        // `reap` can't message back — their `WeakHandle` upgrade fails — but each still runs
        // its own `sup.stop()` to completion; this is the best-effort drop path. The awaited
        // guarantee is `shutdown_all`, which the daemon uses.)
        self.shutting_down = true;
        for sup in self.drain() {
            sup.stop().await;
        }
    }
}

impl NodeActor {
    /// Remove every resident worker from the registry, returning them for teardown.
    fn drain(&mut self) -> Vec<Arc<Supervisor>> {
        let out = self
            .registry
            .ids()
            .into_iter()
            .filter_map(|id| self.registry.remove(id))
            .collect();
        self.last_activity.clear();
        self.in_flight.clear();
        out
    }

    /// Drop a worker's idle bookkeeping (called whenever it leaves the registry).
    fn forget_activity(&mut self, id: WorkerId) {
        self.last_activity.remove(&id);
        self.in_flight.remove(&id);
    }

    /// Stop a worker on a tracked task. EVERY teardown funnels through here so `shutdown_all`
    /// can wait for all of them: bumps `inflight_stops`, runs `sup.stop()` off-thread, then
    /// posts `ReapDone` to decrement and (optionally) answer the op's caller. `done` carries
    /// the `unload`/`kill` reply so it fires only once the stop has actually completed.
    fn reap(
        &mut self,
        sup: Arc<Supervisor>,
        done: Option<oneshot::Sender<Result<(), HiggsError>>>,
    ) {
        self.inflight_stops += 1;
        self.refresh_keepalive();
        let self_handle = self.self_handle.clone();
        tokio::spawn(async move {
            sup.stop().await;
            // The keepalive keeps the actor alive until this completes, so the upgrade
            // succeeds and the count/reply are handled on-thread — UNLESS the runtime was
            // dropped without shutdown_all (accepted Drop residual), where the stop has
            // already run and there's nothing left to track.
            match self_handle.upgrade() {
                Some(h) => {
                    let _ = h.send(NodeMsg::ReapDone { done });
                }
                None => drop(done),
            }
        });
    }

    /// Keep a strong self-handle exactly while work is in flight: take one when the first
    /// load/stop starts, drop it the moment the last finishes. This keeps the mailbox open
    /// across an in-flight chain even if every external handle drops, without ever pinning an
    /// idle actor alive.
    ///
    /// `upgrade()` is `Some` whenever a strong handle still exists — which is the case for
    /// any op whose caller is still awaiting (it holds `&NodeRuntime`). It is `None` ONLY
    /// when the runtime was already fully dropped before this message ran (a cancelled op
    /// whose message was still queued, then `NodeRuntime` dropped without `shutdown_all`).
    /// That is the accepted Drop residual: there is nothing left to keep alive, and the
    /// task's own `None` branch reaps any worker detached/best-effort.
    fn refresh_keepalive(&mut self) {
        let active = self.inflight_loads + self.inflight_stops > 0;
        if active {
            if self.keepalive.is_none() {
                self.keepalive = self.self_handle.upgrade();
            }
        } else {
            self.keepalive = None;
        }
    }

    /// Release the parked `shutdown_all` callers once the drain is truly complete: shutting
    /// down, with no load still resolving and no worker still stopping.
    fn maybe_finish_shutdown(&mut self) {
        if self.shutting_down
            && !self.shutdown_done
            && self.inflight_loads == 0
            && self.inflight_stops == 0
        {
            self.shutdown_done = true;
            for waiter in self.shutdown_waiters.drain(..) {
                let _ = waiter.send(());
            }
        }
    }
}

/// Returned to a `load` that races a shutdown — the runtime is draining, so it refuses to
/// bring up a worker that would outlive the drain.
fn shutting_down() -> HiggsError {
    HiggsError::WorkerDead {
        context: "node runtime shutting down".into(),
    }
}

/// The "no such worker" error shared by the registry lookups.
fn no_worker(id: WorkerId) -> HiggsError {
    HiggsError::WorkerDead {
        context: format!("no worker {id}"),
    }
}

/// Resolve `id` to its on-disk GGUF `(path, size_bytes, ctx_train)` by scanning the
/// node's model roots. The blocking FS scan + canonicalization run on a blocking thread so
/// they never stall the control-plane executor. `get` only returns cataloged models found
/// UNDER the roots, and the resolved path must canonicalize to within a root (symlink-
/// escape guard, [HG015]). `[HG002]` if absent.
async fn resolve_model(
    config: &NodeConfig,
    id: &str,
) -> Result<(String, u64, Option<u64>), HiggsError> {
    let id = id.to_string();
    let (lmstudio, hf, ollama) = config.model_dirs();
    tokio::task::spawn_blocking(move || {
        let mut store = ModelStore::default();
        store.scan(&lmstudio, &hf, &ollama)?;
        let (path, size_bytes, ctx_train) = store
            .get(&id)
            .map(|m| (m.path.clone(), m.size_bytes, m.ctx_train))
            .ok_or_else(|| HiggsError::ModelNotFound { id: id.clone() })?;
        let roots: Vec<PathBuf> = lmstudio.into_iter().chain(hf).chain(ollama).collect();
        if !crate::api::path_within_roots(&path, &roots) {
            return Err(HiggsError::InvalidModelId {
                id,
                reason: format!("resolved path {path} is outside every configured scan directory"),
            });
        }
        Ok((path, size_bytes, ctx_train))
    })
    .await
    .map_err(|e| HiggsError::WorkerDead {
        context: format!("model scan task failed: {e}"),
    })?
}

/// A failed load. `sup` is `Some` only when the worker was already spawned before the
/// failure (start / `M_LOAD`) — the actor then reaps it through the teardown coordinator
/// during shutdown (so `shutdown_all` awaits its stop) or detached otherwise. `None` for
/// pre-spawn failures (resolve / headroom), where there is nothing to reap.
struct LoadFailure {
    sup: Option<Arc<Supervisor>>,
    error: HiggsError,
}

impl From<HiggsError> for LoadFailure {
    /// Pre-spawn failure: no worker exists yet.
    fn from(error: HiggsError) -> Self {
        LoadFailure { sup: None, error }
    }
}

/// The slow load: resolve the GGUF, run the pre-spawn RAM headroom guard ([HG017]), spawn
/// a fresh Supervisor, start its child, and send `M_LOAD`. On success the `Arc<Supervisor>`
/// is returned for the actor to commit under the reserved `id`. On a POST-spawn failure the
/// worker is returned in [`LoadFailure::sup`] so the ACTOR owns its teardown — the reap is
/// then awaited by `shutdown_all` (never a detached, un-awaited stop that could orphan on
/// process exit). Runs OFF the actor thread.
///
/// NOTE: the cross-worker VRAM fit-check (§4.2b) is pending; the RAM headroom guard here
/// is the existing local capacity check, reused.
async fn do_load(
    id: WorkerId,
    params: NodeLoadParams,
    spawner: SupervisorSpawner,
    config: Arc<NodeConfig>,
    relay: broadcast::Sender<(WorkerId, String)>,
) -> Result<(Arc<Supervisor>, Value), LoadFailure> {
    let (path, size_bytes, ctx_train) = resolve_model(&config, &params.id).await?;
    // Reject before spawning a worker if the model can't fit (mirrors Higgs::load).
    crate::api::guard_memory_headroom(&params.id, size_bytes)?;

    // Each worker gets its OWN LogBus so its stderr can be relayed tagged with the worker's
    // id (the node has no UI of its own; the shared bus can't distinguish workers).
    // Subscribe BEFORE start so load-time output is captured, then republish each line to
    // the node-level relay tagged with `id`.
    let wbus = Arc::new(LogBus::new());
    // Inherit the node's verbose setting so the relayed lines match a local worker's.
    wbus.set_verbose(config.bus.verbose());
    let mut worker_logs = wbus.subscribe();
    let sup = Arc::new((spawner)(wbus));
    // Start the relay BEFORE `M_LOAD` — the verbose load-time stderr burst can far exceed
    // the broadcast capacity, so a relay spawned only after load returns would `Lagged` and
    // drop most of the dump. Draining concurrently keeps it flowing to the hub.
    tokio::spawn(async move {
        loop {
            match worker_logs.recv().await {
                Ok(line) if matches!(line.source, LogSource::Worker) => {
                    let _ = relay.send((id, line.text));
                }
                Ok(_) => {} // non-worker line on a worker bus: ignore
                Err(broadcast::error::RecvError::Lagged(_)) => {} // dropped gap; keep going
                Err(broadcast::error::RecvError::Closed) => break, // worker gone
            }
        }
    });
    // Panic-safety net only: if this task is dropped between spawning the worker and an
    // explicit return below (e.g. a panic), reap the child. Every NORMAL exit defuses it —
    // success and post-spawn failure both hand `sup` back to the actor, which owns the reap.
    let guard = StopOnDrop::new(sup.clone());
    // Carry the worker out on a post-spawn failure so the actor (not a detached task) reaps
    // it — that keeps the reap awaitable by `shutdown_all`.
    let fail = |error: HiggsError| LoadFailure {
        sup: Some(sup.clone()),
        error,
    };
    if let Err(e) = sup.start_for(&params.id) {
        guard.commit();
        return Err(fail(e));
    }
    // When the caller omits ctx_len, default to the model's trained context capped at
    // DEFAULT_CTX_CAP (mirrors Higgs::load) rather than the worker's hardcoded 4096.
    let ctx_len = params
        .ctx_len
        .or_else(|| ctx_train.map(|t| (t as u32).min(crate::api::DEFAULT_CTX_CAP)));
    let load_params = json!({
        "id": params.id,
        "path": path,
        "ctx_len": ctx_len,
        "gpu_layers": params.gpu_layers,
        "threads": params.threads,
    });
    let loaded = match sup.request(M_LOAD, load_params.clone()).await {
        Ok(v) => v,
        Err(e) => {
            guard.commit();
            return Err(fail(e));
        }
    };
    // Record the load so the Supervisor's restart FSM replays it on an unexpected respawn —
    // otherwise the replacement child would come back model-less.
    sup.record_last_load(load_params);
    guard.commit(); // handed back to the actor — don't reap
    Ok((sup, loaded))
}

/// Node-level model catalog (`{ "models": [HiggsModel, …] }`) from a fresh scan. Read-only.
async fn do_scan(config: &NodeConfig) -> Result<Value, HiggsError> {
    let (lmstudio, hf, ollama) = config.model_dirs();
    tokio::task::spawn_blocking(move || {
        let mut store = ModelStore::default();
        store.scan(&lmstudio, &hf, &ollama)?;
        Ok::<Value, HiggsError>(json!({ "models": store.models() }))
    })
    .await
    .map_err(|e| HiggsError::WorkerDead {
        context: format!("scan task failed: {e}"),
    })?
}

/// Enumerate GPUs via a transient worker, then fold them with sampled CPU/RAM/load into
/// the `(hardware, runtime)` snapshot off the executor. Shared by `do_sysinfo` and
/// `do_inventory`; `context` names the caller for the task-join error message.
async fn hardware_runtime(
    spawner: &SupervisorSpawner,
    bus: &Arc<LogBus>,
    context: &str,
) -> Result<(crate::system::HardwareInfo, crate::system::RuntimeInfo), HiggsError> {
    let sup = (spawner)(bus.clone());
    let gpus = sup.sysinfo().await;
    tokio::task::spawn_blocking(move || crate::system::SystemInfo::gather_hardware_runtime(gpus))
        .await
        .map_err(|e| HiggsError::WorkerDead {
            context: format!("{context} task failed: {e}"),
        })
}

/// Node-level system info: `{ "hardware": HardwareInfo, "runtime": RuntimeInfo }`.
async fn do_sysinfo(spawner: &SupervisorSpawner, bus: &Arc<LogBus>) -> Result<Value, HiggsError> {
    let (hardware, runtime) = hardware_runtime(spawner, bus, "sysinfo").await?;
    Ok(json!({ "hardware": hardware, "runtime": runtime }))
}

/// Full node self-description for `M_NODE_INVENTORY`: host identity + every resident worker
/// (snapshot taken on the actor thread) + the hardware/runtime snapshot.
async fn do_inventory(
    workers: Vec<crate::remote::InventoryWorker>,
    spawner: &SupervisorSpawner,
    bus: &Arc<LogBus>,
) -> Result<Value, HiggsError> {
    let (hardware, runtime) = hardware_runtime(spawner, bus, "inventory").await?;
    let inventory = crate::remote::NodeInventory {
        hostname: sysinfo::System::host_name().unwrap_or_default(),
        os: std::env::consts::OS.to_string(),
        workers,
        hardware,
        runtime,
    };
    serde_json::to_value(inventory).map_err(|e| HiggsError::WorkerDead {
        context: format!("inventory serialize failed: {e}"),
    })
}

/// The node orchestrator handle: a thin wrapper over the actor's mailbox. Cloning the
/// underlying `Handle` keeps the actor alive; dropping the last one triggers `on_stop`.
pub struct NodeRuntime {
    handle: Handle<NodeMsg>,
    /// Kept on the wrapper too so `subscribe_logs` needs no mailbox round-trip.
    log_tx: broadcast::Sender<(WorkerId, String)>,
    /// Lifecycle-event fan-out, mirrored on the wrapper so `events()` needs no round-trip.
    events_tx: broadcast::Sender<HiggsEvent>,
    /// The node's shared Developer-Log bus (P4b: the local engine reads/configures it here).
    bus: Arc<LogBus>,
    /// Runtime-mutable idle policy, shared with the reaper.
    idle: Arc<IdleConfig>,
}

impl NodeRuntime {
    /// Production runtime (spawns real child workers).
    pub fn new(config: NodeConfig) -> Self {
        Self::with_spawner(config, Arc::new(Supervisor::spawn))
    }

    /// Runtime with an injected supervisor spawner (tests). Spawns the actor task — must be
    /// called from within a Tokio runtime.
    pub(crate) fn with_spawner(config: NodeConfig, spawner: SupervisorSpawner) -> Self {
        let (log_tx, _) = broadcast::channel(LOG_RELAY_CAP);
        let (events_tx, _) = broadcast::channel(EVENT_CAP);
        let config = Arc::new(config);
        let bus = config.bus.clone();
        let relay = log_tx.clone();
        let events_for_actor = events_tx.clone();
        let idle = Arc::new(IdleConfig::new(config.idle_ttl));
        let idle_for_actor = idle.clone();
        let idle_for_reaper = idle.clone();
        let handle = spawn_actor_with(move |h| NodeActor {
            registry: WorkerRegistry::new(),
            spawner,
            config,
            log_tx: relay,
            self_handle: h.downgrade(),
            keepalive: None,
            shutting_down: false,
            inflight_loads: 0,
            inflight_stops: 0,
            shutdown_waiters: Vec::new(),
            shutdown_done: false,
            last_activity: HashMap::new(),
            in_flight: HashMap::new(),
            idle: idle_for_actor,
            events_tx: events_for_actor,
        });
        // Idle reaper: a WeakHandle so it never keeps the actor alive — it exits the tick the
        // last real handle drops (upgrade fails). The cadence is ADAPTIVE — each iteration
        // sleeps `reap_interval(idle.ttl())`, re-reading the live TTL — so a runtime TTL change
        // (Server-Settings) is honored at the new cadence within one period, not frozen to the
        // startup TTL.
        let reaper = handle.downgrade();
        tokio::spawn(async move {
            loop {
                // Wake on the cadence OR immediately when settings change, so a lowered TTL /
                // re-enable takes effect without waiting out the prior sleep.
                tokio::select! {
                    _ = tokio::time::sleep(reap_interval(idle_for_reaper.ttl())) => {}
                    _ = idle_for_reaper.changed.notified() => {}
                }
                match reaper.upgrade() {
                    Some(h) => {
                        if h.send(NodeMsg::ReapIdle).is_err() {
                            break;
                        }
                    }
                    None => break,
                }
            }
        });
        Self {
            handle,
            log_tx,
            events_tx,
            bus,
            idle,
        }
    }

    /// The runtime-mutable idle auto-unload policy (TTL + on/off). The engine wires its
    /// Server-Settings toggles to this; the reaper reads it live.
    pub fn idle(&self) -> &Arc<IdleConfig> {
        &self.idle
    }

    /// Error returned when the actor mailbox is gone (shutting down / loop ended).
    fn actor_gone() -> HiggsError {
        HiggsError::WorkerDead {
            context: "node runtime stopped".into(),
        }
    }

    /// Send a message carrying a `Result` reply and flatten the channel + op errors.
    async fn call<T>(
        &self,
        make: impl FnOnce(oneshot::Sender<Result<T, HiggsError>>) -> NodeMsg,
    ) -> Result<T, HiggsError> {
        let (tx, rx) = oneshot::channel();
        self.handle.send(make(tx)).map_err(|_| Self::actor_gone())?;
        rx.await.map_err(|_| Self::actor_gone())?
    }

    /// Subscribe to the per-worker stderr relay (`(worker_id, line)`). The node's
    /// `serve_node` drains this onto a uni stream to the hub as `N_LOG_LINE`.
    pub fn subscribe_logs(&self) -> broadcast::Receiver<(WorkerId, String)> {
        self.log_tx.subscribe()
    }

    /// Subscribe to lifecycle events (ModelLoaded/ModelUnloaded) — the engine's event stream.
    pub fn events(&self) -> broadcast::Receiver<HiggsEvent> {
        self.events_tx.subscribe()
    }

    /// The node's shared Developer-Log bus (Developer-Logs history + live stream + verbosity).
    pub fn bus(&self) -> &Arc<LogBus> {
        &self.bus
    }

    /// Resident instances as `(worker, raw model)` — the engine's input to global served-id
    /// derivation. Empty if the actor has stopped.
    pub async fn instances(&self) -> Vec<(WorkerId, String)> {
        let (tx, rx) = oneshot::channel();
        if self.handle.send(NodeMsg::Instances { reply: tx }).is_err() {
            return Vec::new();
        }
        rx.await.unwrap_or_default()
    }

    /// Probe GGUF support for `paths` (control-plane; via a transient worker).
    pub async fn probe_paths(&self, paths: Vec<String>) -> ProbeVerdicts {
        let (tx, rx) = oneshot::channel();
        if self
            .handle
            .send(NodeMsg::Probe { paths, reply: tx })
            .is_err()
        {
            return Vec::new();
        }
        rx.await.unwrap_or_default()
    }

    /// Enumerate host GPUs (control-plane; via a transient worker). Empty on failure.
    pub async fn gpus(&self) -> Vec<crate::system::GpuDevice> {
        let (tx, rx) = oneshot::channel();
        if self.handle.send(NodeMsg::Gpus { reply: tx }).is_err() {
            return Vec::new();
        }
        rx.await.unwrap_or_default()
    }

    /// Live worker ids, ascending (empty if the actor has stopped).
    pub async fn worker_ids(&self) -> Vec<WorkerId> {
        let (tx, rx) = oneshot::channel();
        if self.handle.send(NodeMsg::WorkerIds { reply: tx }).is_err() {
            return Vec::new();
        }
        rx.await.unwrap_or_default()
    }

    /// Spawn a NEW worker for `params.id` and load the model (does NOT replace others). The
    /// node resolves the GGUF path from its own disk; returns the new `WorkerId` and the
    /// worker's `M_LOAD` result. A slow load runs concurrently and cannot block other ops.
    pub async fn load(&self, params: NodeLoadParams) -> Result<(WorkerId, Value), HiggsError> {
        self.call(|reply| NodeMsg::Load { params, reply }).await
    }

    /// Graceful unload: stop the worker, free the id.
    pub async fn unload(&self, id: WorkerId) -> Result<(), HiggsError> {
        self.call(|reply| NodeMsg::Unload { id, reply }).await
    }

    /// Force-kill ONE worker. Identical to [`unload`](Self::unload) at this layer.
    pub async fn kill(&self, id: WorkerId) -> Result<(), HiggsError> {
        self.call(|reply| NodeMsg::Kill { id, reply }).await
    }

    /// Per-worker status (forwards `M_STATUS` to that worker's Supervisor).
    pub async fn status(&self, id: WorkerId) -> Result<Value, HiggsError> {
        self.call(|reply| NodeMsg::Status { id, reply }).await
    }

    /// Node-level model catalog from a fresh scan of the node's roots.
    pub async fn scan(&self) -> Result<Value, HiggsError> {
        self.call(|reply| NodeMsg::Scan { reply }).await
    }

    /// Node-level system info: `{ "hardware": HardwareInfo, "runtime": RuntimeInfo }`.
    pub async fn sysinfo(&self) -> Result<Value, HiggsError> {
        self.call(|reply| NodeMsg::Sysinfo { reply }).await
    }

    /// Full node self-description for `M_NODE_INVENTORY`.
    pub async fn inventory(&self) -> Result<Value, HiggsError> {
        self.call(|reply| NodeMsg::Inventory { reply }).await
    }

    /// Lease a worker's Supervisor for one chat (idle-reaper-safe — see [`ChatLease`]). Hold
    /// the returned lease for the whole generation; dropping it ends the chat's in-flight hold.
    pub(crate) async fn chat_handle(&self, id: WorkerId) -> Result<ChatLease, HiggsError> {
        self.call(|reply| NodeMsg::ChatHandle { id, reply }).await
    }

    /// Graceful drain: stop every resident worker and empty the registry. The node daemon
    /// calls this on shutdown so committed workers are reaped.
    pub async fn shutdown_all(&self) {
        let (tx, rx) = oneshot::channel();
        if self.handle.send(NodeMsg::ShutdownAll { reply: tx }).is_ok() {
            let _ = rx.await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::test_support::{
        fake_runtime as fake_runtime_with_dirs, fake_runtime_load_fails,
    };
    use tempfile::TempDir;

    /// Like `fake_runtime_with_models` but the worker fails every M_LOAD (post-spawn
    /// failure) so the reap-the-spawned-worker path is exercised.
    fn load_failing_runtime_with_models(ids: &[&str]) -> (NodeRuntime, TempDir) {
        let dir = TempDir::new().expect("staging dir");
        for id in ids {
            let model_dir = dir.path().join(id);
            std::fs::create_dir_all(&model_dir).expect("model dir");
            std::fs::write(model_dir.join("m.gguf"), b"GGUF\x00 dummy").expect("write dummy gguf");
        }
        let rt = fake_runtime_load_fails(vec![dir.path().to_path_buf()]);
        (rt, dir)
    }

    /// A fake-backed runtime whose scan root contains a dummy GGUF for each `id`, so the
    /// real `load` path (resolve → spawn → M_LOAD) runs without llama.cpp. Keep the
    /// returned `TempDir` alive for the test's duration.
    fn fake_runtime_with_models(ids: &[&str]) -> (NodeRuntime, TempDir) {
        let dir = TempDir::new().expect("staging dir");
        for id in ids {
            let model_dir = dir.path().join(id);
            std::fs::create_dir_all(&model_dir).expect("model dir");
            std::fs::write(model_dir.join("m.gguf"), b"GGUF\x00 dummy").expect("write dummy gguf");
        }
        let rt = fake_runtime_with_dirs(vec![dir.path().to_path_buf()]);
        (rt, dir)
    }

    fn load_params(id: &str) -> NodeLoadParams {
        NodeLoadParams {
            id: id.into(),
            ctx_len: None,
            gpu_layers: None,
            threads: None,
        }
    }

    async fn load(rt: &NodeRuntime, id: &str) -> (WorkerId, Value) {
        rt.load(load_params(id)).await.unwrap()
    }

    #[tokio::test]
    async fn load_assigns_ids_and_kill_frees_them() {
        let (rt, _dir) = fake_runtime_with_models(&["org/m-a", "org/m-b"]);
        let (a, _) = load(&rt, "org/m-a").await;
        let (b, _) = load(&rt, "org/m-b").await;
        assert_ne!(a, b);
        assert_eq!(rt.worker_ids().await.len(), 2, "two concurrent workers");
        rt.kill(a).await.unwrap();
        assert_eq!(rt.worker_ids().await.len(), 1);
        assert!(rt.kill(a).await.is_err(), "killing a freed id errors");
    }

    #[tokio::test]
    async fn load_returns_worker_load_result() {
        let (rt, _dir) = fake_runtime_with_models(&["org/model"]);
        let (_, loaded) = load(&rt, "org/model").await;
        assert_eq!(loaded["id"], "org/model");
    }

    #[tokio::test]
    async fn load_resolves_path_and_errors_when_model_absent() {
        // No model staged → resolution yields HG002 ModelNotFound, no worker spawned.
        let (rt, _dir) = fake_runtime_with_models(&[]);
        let err = rt.load(load_params("missing/model")).await.unwrap_err();
        assert!(err.to_string().starts_with("[HG002]"), "got {err}");
        assert!(
            rt.worker_ids().await.is_empty(),
            "no worker spawned on resolve failure"
        );
    }

    #[tokio::test]
    async fn status_forwards_to_the_worker() {
        let (rt, _dir) = fake_runtime_with_models(&["org/m"]);
        let (id, _) = load(&rt, "org/m").await;
        let status = rt.status(id).await.unwrap();
        assert!(status.get("loaded").is_some());
        assert!(
            rt.status(WorkerId(999)).await.is_err(),
            "unknown worker errors"
        );
    }

    #[tokio::test]
    async fn unload_stops_and_frees() {
        let (rt, _dir) = fake_runtime_with_models(&["org/m"]);
        let (id, _) = load(&rt, "org/m").await;
        rt.unload(id).await.unwrap();
        assert!(rt.worker_ids().await.is_empty());
    }

    #[tokio::test]
    async fn inventory_lists_resident_workers_with_their_models() {
        let (rt, _dir) = fake_runtime_with_models(&["org/a", "org/b"]);
        let (wa, _) = load(&rt, "org/a").await;
        load(&rt, "org/b").await;
        let inv = rt.inventory().await.unwrap();
        assert!(
            inv["hardware"]["cpu_cores"].as_u64().unwrap() > 0,
            "real hw"
        );
        assert!(!inv["os"].as_str().unwrap().is_empty(), "os present");
        let workers = inv["workers"].as_array().unwrap();
        assert_eq!(workers.len(), 2, "both workers listed");
        let a = workers
            .iter()
            .find(|w| w["worker_id"].as_u64().unwrap() as u32 == wa.0)
            .unwrap();
        assert_eq!(a["model"], "org/a", "worker reports its model");
    }

    #[tokio::test]
    async fn shutdown_all_drains_every_worker() {
        let (rt, _dir) = fake_runtime_with_models(&["org/a", "org/b"]);
        load(&rt, "org/a").await;
        load(&rt, "org/b").await;
        assert_eq!(rt.worker_ids().await.len(), 2);
        rt.shutdown_all().await;
        assert!(rt.worker_ids().await.is_empty(), "all workers drained");
    }

    #[tokio::test]
    async fn scan_lists_staged_models() {
        let (rt, _dir) = fake_runtime_with_models(&["org/seen"]);
        let scanned = rt.scan().await.unwrap();
        let models = scanned["models"].as_array().expect("models array");
        assert!(
            models.iter().any(|m| m["id"] == "org/seen"),
            "scan lists the staged model: {scanned}"
        );
    }

    // `gpus()` enumerates host GPUs via a transient worker (the engine's `Higgs::sysinfo`
    // input post-P4b). The fake worker reports an empty GPU list, returned without hanging.
    #[tokio::test]
    async fn gpus_enumerates_via_transient_worker() {
        let (rt, _dir) = fake_runtime_with_models(&[]);
        let gpus = rt.gpus().await;
        assert!(gpus.is_empty(), "fake worker reports no GPUs");
    }

    #[tokio::test]
    async fn sysinfo_reports_real_hardware() {
        let (rt, _dir) = fake_runtime_with_models(&[]);
        let sys = rt.sysinfo().await.unwrap();
        assert!(sys["hardware"]["cpu_cores"].as_u64().unwrap() > 0);
        assert!(sys["runtime"].is_object(), "runtime block present");
    }

    // The spawn-and-commit property: two loads issued back-to-back run CONCURRENTLY (a slow
    // load does not head-of-line-block the next op), and the same model on two workers gets
    // two distinct ids — the per-node basis for duplicate served-id suffixes.
    #[tokio::test]
    async fn concurrent_loads_of_one_model_get_distinct_ids() {
        let (rt, _dir) = fake_runtime_with_models(&["org/dup"]);
        let (a, b) = tokio::join!(load(&rt, "org/dup"), load(&rt, "org/dup"));
        assert_ne!(a.0, b.0, "same model on two workers → two ids");
        assert_eq!(rt.worker_ids().await.len(), 2);
    }

    // A post-spawn load failure (worker spawned, M_LOAD errors) surfaces the error and
    // leaves no committed worker — the spawned child is reaped.
    #[tokio::test]
    async fn post_spawn_load_failure_reaps_and_errors() {
        let (rt, _dir) = load_failing_runtime_with_models(&["org/m"]);
        let err = rt.load(load_params("org/m")).await.unwrap_err();
        assert!(err.to_string().contains("HG017"), "got {err}");
        assert!(
            rt.worker_ids().await.is_empty(),
            "failed load committed no worker"
        );
    }

    // A post-spawn load failure that resolves DURING shutdown is reaped through the teardown
    // coordinator — shutdown_all completes cleanly with nothing left committed.
    #[tokio::test]
    async fn post_spawn_failure_during_shutdown_drains() {
        let (rt, _dir) = load_failing_runtime_with_models(&["org/a", "org/b"]);
        // Issue loads (which will fail) concurrently with shutdown so some resolve mid-drain.
        let (_l1, _l2, _s) = tokio::join!(
            rt.load(load_params("org/a")),
            rt.load(load_params("org/b")),
            rt.shutdown_all(),
        );
        assert!(rt.worker_ids().await.is_empty(), "drained after shutdown");
    }

    // shutdown_all is idempotent: a second call (after the coordinator already finished)
    // returns immediately instead of hanging on a never-fired completion.
    #[tokio::test]
    async fn shutdown_all_is_idempotent() {
        let (rt, _dir) = fake_runtime_with_models(&["org/m"]);
        load(&rt, "org/m").await;
        rt.shutdown_all().await;
        // Second call must not block (would hang if it parked for a gone coordinator).
        tokio::time::timeout(std::time::Duration::from_secs(5), rt.shutdown_all())
            .await
            .expect("second shutdown_all returns promptly");
    }

    // Two concurrent shutdown_all callers both complete: the second parks behind the same
    // coordinator and is released by the single ShutdownDone.
    #[tokio::test]
    async fn concurrent_shutdown_all_callers_all_complete() {
        let (rt, _dir) = fake_runtime_with_models(&["org/a", "org/b"]);
        load(&rt, "org/a").await;
        load(&rt, "org/b").await;
        let (a, b) = tokio::join!(rt.shutdown_all(), rt.shutdown_all());
        let _ = (a, b);
        assert!(rt.worker_ids().await.is_empty(), "all workers drained");
    }

    // A load issued AFTER shutdown_all is rejected (shutdown is terminal) — it must not
    // resurrect a worker that would survive the drain.
    #[tokio::test]
    async fn load_after_shutdown_is_rejected() {
        let (rt, _dir) = fake_runtime_with_models(&["org/m"]);
        rt.shutdown_all().await;
        let err = rt.load(load_params("org/m")).await.unwrap_err();
        assert!(
            err.to_string().contains("shutting down"),
            "got {err} — load must be refused after shutdown"
        );
        assert!(
            rt.worker_ids().await.is_empty(),
            "no worker after a post-shutdown load"
        );
    }

    /// Poll until the runtime has no resident workers (idle reaper did its job), up to ~5s.
    /// Robust against scheduling jitter under full-suite parallel load (vs a fixed sleep).
    async fn wait_reaped(rt: &NodeRuntime) -> bool {
        for _ in 0..100 {
            if rt.worker_ids().await.is_empty() {
                return true;
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        false
    }

    // `instances()` lists resident (worker, raw model) pairs, and lifecycle events fire on
    // load/unload — the surface the local engine consumes in P4b.
    #[tokio::test]
    async fn instances_and_events_track_load_unload() {
        let (rt, _dir) = fake_runtime_with_models(&["org/a", "org/b"]);
        let mut events = rt.events();
        let (wa, _) = load(&rt, "org/a").await;
        load(&rt, "org/b").await;
        let mut insts = rt.instances().await;
        insts.sort();
        assert_eq!(insts.len(), 2);
        assert!(insts.iter().any(|(w, m)| *w == wa && m == "org/a"));
        // A load emitted a ModelLoaded event.
        let ev = events.recv().await.unwrap();
        assert!(matches!(ev, HiggsEvent::ModelLoaded { .. }));
        // Unload removes the instance and emits ModelUnloaded.
        rt.unload(wa).await.unwrap();
        assert_eq!(rt.instances().await.len(), 1);
    }

    // The idle reaper auto-unloads a worker that has had no chat activity for longer than the
    // configured TTL.
    #[tokio::test]
    async fn idle_worker_is_auto_unloaded_after_ttl() {
        use crate::node::test_support::fake_runtime_with_idle_ttl;
        let dir = TempDir::new().expect("staging dir");
        let model_dir = dir.path().join("org/m");
        std::fs::create_dir_all(&model_dir).unwrap();
        std::fs::write(model_dir.join("m.gguf"), b"GGUF\x00 dummy").unwrap();
        let rt = fake_runtime_with_idle_ttl(
            vec![dir.path().to_path_buf()],
            std::time::Duration::from_millis(120),
        );
        load(&rt, "org/m").await;
        assert_eq!(rt.worker_ids().await.len(), 1);
        // Poll past the TTL + reap ticks (robust to scheduling jitter under load).
        assert!(
            wait_reaped(&rt).await,
            "idle worker auto-unloaded after the TTL"
        );
    }

    // Turning auto-unload OFF at runtime stops the reaper; turning it back on (with a tiny TTL)
    // lets it reap again — the live Server-Settings behavior.
    #[tokio::test]
    async fn runtime_idle_config_toggles_take_effect_live() {
        use crate::node::test_support::fake_runtime_with_idle_ttl;
        let dir = TempDir::new().expect("staging dir");
        let model_dir = dir.path().join("org/m");
        std::fs::create_dir_all(&model_dir).unwrap();
        std::fs::write(model_dir.join("m.gguf"), b"GGUF\x00 dummy").unwrap();
        // Start with a tiny TTL but auto-unload DISABLED — the worker must survive.
        let rt = fake_runtime_with_idle_ttl(
            vec![dir.path().to_path_buf()],
            std::time::Duration::from_millis(120),
        );
        rt.idle().set_enabled(false);
        load(&rt, "org/m").await;
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        assert_eq!(
            rt.worker_ids().await.len(),
            1,
            "auto-unload disabled keeps the worker resident"
        );
        // Re-enable: now the idle worker is reaped.
        rt.idle().set_enabled(true);
        assert!(
            wait_reaped(&rt).await,
            "re-enabling auto-unload reaps the idle worker"
        );
    }

    // A chat in flight (the relay holds a ChatLease) keeps a worker resident past the TTL; once
    // the lease drops, the idle clock restarts from the end of the chat and it is reaped.
    #[tokio::test]
    async fn in_flight_chat_prevents_idle_reap() {
        use crate::node::test_support::fake_runtime_with_idle_ttl;
        let dir = TempDir::new().expect("staging dir");
        let model_dir = dir.path().join("org/m");
        std::fs::create_dir_all(&model_dir).unwrap();
        std::fs::write(model_dir.join("m.gguf"), b"GGUF\x00 dummy").unwrap();
        let rt = fake_runtime_with_idle_ttl(
            vec![dir.path().to_path_buf()],
            std::time::Duration::from_millis(120),
        );
        let (id, _) = load(&rt, "org/m").await;
        // Hold a chat lease across the whole idle window — the worker must NOT be reaped.
        let lease = rt.chat_handle(id).await.expect("lease");
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        assert_eq!(
            rt.worker_ids().await.len(),
            1,
            "in-flight chat keeps the worker resident past the TTL"
        );
        // End the chat; now the idle clock runs and the worker is reaped.
        drop(lease);
        assert!(
            wait_reaped(&rt).await,
            "worker reaped once the chat ended and it went idle"
        );
    }

    // After shutdown_all, the registry is empty so per-worker ops report "no worker"
    // (the actor itself is still alive — rt holds the handle).
    #[tokio::test]
    async fn ops_after_shutdown_report_no_worker() {
        let (rt, _dir) = fake_runtime_with_models(&["org/m"]);
        let (id, _) = load(&rt, "org/m").await;
        rt.shutdown_all().await;
        assert!(rt.status(id).await.is_err(), "stopped worker is gone");
        assert!(rt.unload(id).await.is_err(), "nothing left to unload");
    }
}
