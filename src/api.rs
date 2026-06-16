//! `Higgs` public facade and `HiggsConfig` — the host-facing API.
//!
//! One `Higgs` instance per host app. Thin typed delegation over
//! [`Supervisor`](crate::supervisor::Supervisor); all state lives in the
//! supervisor. The host maps its own config table onto [`HiggsConfig`].

use std::path::PathBuf;
use std::sync::Arc;

use serde_json::json;
use tokio::sync::{broadcast, mpsc};

use crate::diagnostic::HiggsError;
use crate::supervisor::{HiggsEvent, Supervisor};
use crate::worker::engine::LoadParams;
use crate::worker::models::HiggsModel;
use crate::worker::{M_CHAT, M_LOAD, M_STATUS, M_UNLOAD};

// ── HiggsConfig ───────────────────────────────────────────────────────────────

higgs_ts! {
    /// Host-supplied configuration (the host maps its own config table onto this).
    #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
    pub struct HiggsConfig {
        /// LM Studio model directories to scan.
        pub lmstudio_dirs: Vec<PathBuf>,
        /// HuggingFace Hub cache directories to scan.
        ///
        /// Note: HuggingFace hardcodes `~/.cache/huggingface/hub` on ALL platforms —
        /// it does not follow XDG or macOS conventions. We use
        /// `dirs::home_dir().join(".cache/huggingface/hub")`, NOT `dirs::cache_dir()`.
        pub hf_dirs: Vec<PathBuf>,
        /// Ollama model store directories to scan.
        pub ollama_dirs: Vec<PathBuf>,
        /// Load parameters used when none are supplied by the caller.
        pub default_load: LoadParams,
    }
}

impl Default for HiggsConfig {
    fn default() -> Self {
        let home = dirs::home_dir();

        // Helper: build a path from home; return empty vec when home is unknown.
        let home_path = |segments: &[&str]| -> Vec<PathBuf> {
            match &home {
                Some(h) => {
                    let mut p = h.clone();
                    for s in segments {
                        p = p.join(s);
                    }
                    vec![p]
                }
                None => vec![],
            }
        };

        let lmstudio_dirs = {
            // LM Studio < 0.3 stores models in ~/.lmstudio/models.
            // LM Studio >= 0.3 uses ~/.cache/lm-studio/models.
            // Higgs scans both; the host can narrow via config.
            let mut dirs = Vec::new();
            if let Some(h) = &home {
                dirs.push(h.join(".lmstudio").join("models"));
                dirs.push(h.join(".cache").join("lm-studio").join("models"));
            }
            dirs
        };

        // HuggingFace hardcodes ~/.cache on ALL platforms — do NOT use dirs::cache_dir().
        let hf_dirs = home_path(&[".cache", "huggingface", "hub"]);

        let ollama_dirs = home_path(&[".ollama", "models"]);

        let threads = {
            let avail = std::thread::available_parallelism()
                .map(std::num::NonZero::get)
                .unwrap_or(1);
            avail.saturating_sub(2).max(1) as u32
        };

        Self {
            lmstudio_dirs,
            hf_dirs,
            ollama_dirs,
            default_load: LoadParams {
                ctx_len: 4096,
                gpu_layers: u32::MAX,
                threads,
            },
        }
    }
}

// ── Output types ──────────────────────────────────────────────────────────────

higgs_ts! {
    /// Info about the currently loaded model.
    #[derive(Debug, Clone, serde::Serialize)]
    pub struct LoadedInfo {
        /// HuggingFace repo id of the resident model.
        pub id: String,
        /// Context window size in tokens.
        #[ts(type = "number")]
        pub ctx_len: u32,
        /// GPU layers offloaded; u32::MAX means all.
        #[ts(type = "number")]
        pub gpu_layers: u32,
        /// Worker threads used during generation.
        #[ts(type = "number")]
        pub threads: u32,
        // Model metadata from the store — present when the worker has scanned the model.
        /// Model architecture read from GGUF header (e.g. `"llama"`, `"gemma3"`).
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        pub arch: Option<String>,
        /// Quantization tag (e.g. `Q4_K_M`), if present.
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        pub quant: Option<String>,
        /// Training context length from GGUF header (model's maximum). Distinct from
        /// `ctx_len` which is the actually loaded window size.
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(type = "number")]
        #[ts(optional)]
        pub max_context_length: Option<u64>,
        /// File size in bytes.
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(type = "number")]
        #[ts(optional)]
        pub size_bytes: Option<u64>,
        /// Whether `tokenizer.chat_template` is present in the GGUF header.
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        pub has_chat_template: Option<bool>,
    }
}

higgs_ts! {
    /// Live status snapshot returned by [`Higgs::status`].
    #[derive(Debug, Clone, serde::Serialize)]
    pub struct HiggsStatus {
        /// Whether the worker process is currently alive.
        pub worker_alive: bool,
        /// Info about the loaded model, if any.
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        pub loaded: Option<LoadedInfo>,
        /// Number of models discovered in the last scan.
        #[ts(type = "number")]
        pub models_on_disk: u32,
    }
}

/// Final outcome of a completed chat request.
#[derive(Debug, Clone)]
pub struct ChatOutcome {
    /// Assistant text after tool-call parsing (the OpenAI message `content`).
    pub content: String,
    /// OpenAI finish_reason ("stop" or "length"); the boundary upgrades this to
    /// "tool_calls" when [`tool_calls`](Self::tool_calls) is present.
    pub finish_reason: String,
    /// Parsed OpenAI `tool_calls` array, or `None` when the turn emitted none.
    pub tool_calls: Option<serde_json::Value>,
    /// Prompt token count from the engine (for OpenAI `usage.prompt_tokens`).
    pub prompt_tokens: u32,
    /// Completion token count from the engine (for OpenAI `usage.completion_tokens`).
    pub completion_tokens: u32,
}

// ── Higgs ─────────────────────────────────────────────────────────────────────

/// The in-process handle to the higgs runtime. One instance per host app.
///
/// Constructing `Higgs` does not start the worker; call [`start`](Self::start)
/// when the host is ready to serve requests.
/// Default context-window cap used when a load does not pin `ctx_len`: the
/// model's trained context is used but never exceeds this, so a huge-context
/// model doesn't allocate an enormous KV cache by default. A caller (the UI)
/// can still request the full trained window explicitly.
const DEFAULT_CTX_CAP: u32 = 32_768;

pub struct Higgs {
    sup: Arc<Supervisor>,
    config: parking_lot::Mutex<HiggsConfig>,
    /// Serializes load/unload so spawn-on-load and kill-on-unload never
    /// interleave (protects last_load and the supervisor proc handle).
    lifecycle: tokio::sync::Mutex<()>,
}

impl Higgs {
    /// Construct the facade WITHOUT spawning the worker.
    ///
    /// Call [`start`](Self::start) when the host is ready.
    pub fn new(config: HiggsConfig) -> Self {
        Self {
            sup: Arc::new(Supervisor::spawn()),
            config: parking_lot::Mutex::new(config),
            lifecycle: tokio::sync::Mutex::new(()),
        }
    }

    /// Bring up control only — does NOT spawn a worker.
    ///
    /// A worker is spawned lazily by [`load`](Self::load) (spawn-on-load,
    /// LM-Studio model). `scan` runs host-side and needs no worker. The serve
    /// layer holds `Arc<Higgs>` for control regardless of worker liveness.
    pub async fn start(&self) -> Result<(), HiggsError> {
        Ok(())
    }

    /// Gracefully shut down the worker (2 s timeout).
    ///
    /// Holds the `lifecycle` mutex for the whole body so a deliberate stop never
    /// interleaves with a concurrent `load`/`unload` (which would let a load
    /// spawn + M_LOAD + emit `ModelLoaded` race this kill). Also clears the
    /// load-replay state: a deliberate worker stop must not leave `last_load`
    /// behind for `attempt_restart` to resurrect the model.
    pub async fn stop(&self) {
        let _lifecycle = self.lifecycle.lock().await;
        self.sup.clear_last_load();
        self.sup.stop().await;
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
    /// load params for post-restart replay and emits [`HiggsEvent::ModelLoaded`].
    pub async fn load(&self, id: &str, params: Option<LoadParams>) -> Result<(), HiggsError> {
        // Serialize the whole load/unload lifecycle: spawn-on-load and
        // kill-on-unload must never interleave (protects last_load and the
        // supervisor proc handle). Held for the entire method body.
        let _lifecycle = self.lifecycle.lock().await;
        let explicit_params = params.is_some();
        let mut p = params.unwrap_or_else(|| self.config.lock().default_load.clone());
        // Scan moved host-side, so the worker's ModelStore is empty on a fresh
        // spawn-on-load worker: resolve the model HERE and carry the GGUF path in
        // the M_LOAD params. Without this the worker's `store.get(id)` returns
        // HG002 for every normal load. Take the first matching model.
        let model = self
            .scan()
            .await?
            .into_iter()
            .find(|m| m.id == id)
            .ok_or_else(|| HiggsError::ModelNotFound { id: id.to_owned() })?;
        // When the caller didn't pin ctx_len, default it to the model's trained
        // context (capped at DEFAULT_CTX_CAP) rather than the hardcoded 4096 —
        // otherwise an agent asking for a large max_tokens overflows n_ctx
        // ([HG005]). The UI can still request the full trained window explicitly.
        if !explicit_params {
            if let Some(train) = model.ctx_train {
                p.ctx_len = (train as u32).min(DEFAULT_CTX_CAP);
            }
        }
        let req_params = json!({
            "id": id,
            "path": model.path,
            "ctx_len": p.ctx_len,
            "gpu_layers": p.gpu_layers,
            "threads": p.threads,
        });
        // Spawn-on-load: if no worker is live, bring one up named `higgs(<id>)`
        // before sending M_LOAD. A redundant call while a worker is running is a
        // no-op (single-reader invariant in the supervisor).
        self.sup.start_for(id)?;
        // If M_LOAD fails (bad GGUF, OOM, …) the worker is alive but holds no
        // model — that contradicts kill-on-unload. Tear it down before
        // returning. Call `self.sup.stop()` DIRECTLY (not `self.stop()`): we
        // already hold the `lifecycle` mutex, and `Higgs::stop()` would re-take
        // it → deadlock. `record_last_load`/`ModelLoaded` stay on success only.
        if let Err(e) = self.sup.request(M_LOAD, req_params.clone()).await {
            self.sup.clear_last_load();
            self.sup.stop().await;
            return Err(e);
        }
        self.sup.record_last_load(req_params);
        self.sup.emit(HiggsEvent::ModelLoaded { id: id.to_owned() });
        Ok(())
    }

    /// Unload the current model.
    ///
    /// Emits [`HiggsEvent::ModelUnloaded`] with an empty id when no model id
    /// is available at the facade layer (v1 limitation; worker tracks it).
    pub async fn unload(&self) -> Result<(), HiggsError> {
        // Serialize the whole load/unload lifecycle (see `load`): held for the
        // entire method body so a concurrent load cannot re-set last_load after
        // the clear or race start_for against this stop.
        let _lifecycle = self.lifecycle.lock().await;
        // TODO(v2): single RPC — status+unload is TOCTOU if worker state changes between calls (v1: worker serializes, benign)
        // Capture id from status before unloading so the event carries it.
        let id = self.loaded_id().await.unwrap_or_default();
        // Drop the load-replay state BEFORE the unload/stop awaits: if a respawn
        // races the stop, there must be nothing left for it to replay. Clearing
        // after the awaits leaves a window where attempt_restart could reload the
        // model the user just unloaded.
        self.sup.clear_last_load();
        // Best-effort graceful in-worker unload, then KILL the worker process
        // (spawn-on-load / kill-on-unload). `stop()` sets the deliberate-stop flag
        // so the death triggers no respawn, drains stdin, and reaps the process.
        let _ = self.sup.request(M_UNLOAD, serde_json::Value::Null).await;
        self.sup.stop().await;
        self.sup.emit(HiggsEvent::ModelUnloaded { id });
        Ok(())
    }

    /// Return a live status snapshot.
    ///
    /// `worker_alive` is `true` iff the RPC round-trip succeeded. `loaded` is
    /// independently best-effort: an RPC failure yields `worker_alive:false` with
    /// `loaded:None`; a malformed `loaded` shape in an otherwise-OK response yields
    /// `worker_alive:true` with `loaded:None`.
    pub async fn status(&self) -> Result<HiggsStatus, HiggsError> {
        let result = self.sup.request(M_STATUS, serde_json::Value::Null).await;
        let worker_alive = result.is_ok();
        let v = result.unwrap_or(serde_json::Value::Null);

        // Scan moved host-side: the worker no longer scans (its `ModelStore` is
        // empty), so model metadata and the on-disk count both come from ONE
        // host-side FS walk (pure Rust, no worker RPC), reused below.
        let scan = self.scan().await.unwrap_or_default();
        let models_on_disk = scan.len() as u32;

        // The worker's M_STATUS reports `id`/`ctx_len`/`gpu_layers`/`threads`
        // from the live model, but `arch`/`quant`/`size_bytes`/
        // `max_context_length`/`has_chat_template` come back null (its store is
        // empty). Enrich those from the matching host-scanned `HiggsModel` while
        // keeping the worker-reported id/ctx_len verbatim.
        let loaded = v.get("loaded").and_then(|l| {
            if l.is_null() {
                return None;
            }
            let id = l.get("id")?.as_str()?.to_owned();
            let scanned = scan.iter().find(|m| m.id == id);
            Some(LoadedInfo {
                ctx_len: l.get("ctx_len")?.as_u64()? as u32,
                gpu_layers: l.get("gpu_layers")?.as_u64()? as u32,
                threads: l.get("threads")?.as_u64()? as u32,
                arch: scanned.and_then(|m| m.arch.clone()),
                quant: scanned.and_then(|m| m.quant.clone()),
                max_context_length: scanned.and_then(|m| m.ctx_train),
                size_bytes: scanned.map(|m| m.size_bytes),
                has_chat_template: scanned.map(|m| m.has_chat_template),
                id,
            })
        });

        Ok(HiggsStatus {
            worker_alive,
            loaded,
            models_on_disk,
        })
    }

    /// Stream a chat completion.
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
    pub async fn chat_stream(
        &self,
        messages_json: String,
        max_tokens: usize,
        temperature: f32,
        tools_json: Option<String>,
    ) -> Result<
        (
            mpsc::UnboundedReceiver<String>,
            tokio::task::JoinHandle<Result<ChatOutcome, HiggsError>>,
        ),
        HiggsError,
    > {
        // Allocate the request id first so the same id is used for both:
        //   (a) the M_CHAT RPC frame's `id` field (for response correlation), and
        //   (b) the `request_id` in the M_CHAT params (for N_CHAT_CHUNK routing).
        // The worker echoes params.request_id in every N_CHAT_CHUNK notification;
        // `route_notification` looks up this id in `chat_sinks` to deliver each
        // delta to the correct caller's receiver.
        let request_id = self.sup.alloc_request_id();
        let rx = self.sup.register_chat_sink(request_id);
        let sup = Arc::clone(&self.sup);

        let handle = tokio::spawn(async move {
            let result = sup
                .request_with_id(
                    request_id,
                    M_CHAT,
                    json!({
                        "request_id": request_id,
                        // Raw OpenAI messages array (serialized verbatim by the
                        // serve layer) carried as a JSON string so tool_calls /
                        // tool_call_id survive to the engine's chat template.
                        "messages_json": messages_json,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "tools": tools_json,
                    }),
                )
                .await;

            // Remove the sink on any outcome: on success the sender is dropped
            // (closing the receiver); on failure the receiver is also closed.
            sup.remove_chat_sink(request_id);

            let result = result?;

            let content = result
                .get("content")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_owned();
            let finish_reason = result
                .get("finish_reason")
                .and_then(|v| v.as_str())
                .unwrap_or("stop")
                .to_owned();
            let prompt_tokens = result
                .get("prompt_tokens")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0) as u32;
            let completion_tokens = result
                .get("completion_tokens")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0) as u32;
            let tool_calls = result.get("tool_calls").filter(|v| !v.is_null()).cloned();

            Ok(ChatOutcome {
                content,
                finish_reason,
                tool_calls,
                prompt_tokens,
                completion_tokens,
            })
        });

        Ok((rx, handle))
    }

    /// Subscribe to worker lifecycle events.
    pub fn events(&self) -> broadcast::Receiver<HiggsEvent> {
        self.sup.events()
    }

    /// Return up to `n` recent stderr log lines from the worker (oldest first).
    pub fn logs(&self, n: usize) -> Vec<String> {
        self.sup.logs(n)
    }

    /// Snapshot of the configured default load parameters.
    ///
    /// The serve router uses this to fill fields absent from a partial
    /// load request — config stays the single home for the defaults.
    pub(crate) fn default_load(&self) -> LoadParams {
        self.config.lock().default_load.clone()
    }

    /// Test-only: build a `Higgs` over a pre-built (mock) supervisor.
    ///
    /// Lets sibling modules (`serve`) reuse the duplex mock seam without
    /// access to this module's private fields.
    #[cfg(test)]
    pub(crate) fn with_supervisor(sup: Arc<Supervisor>, config: HiggsConfig) -> Self {
        Self {
            sup,
            config: parking_lot::Mutex::new(config),
            lifecycle: tokio::sync::Mutex::new(()),
        }
    }

    // ── private ───────────────────────────────────────────────────────────────

    /// Best-effort: ask the worker for the currently loaded model id.
    async fn loaded_id(&self) -> Option<String> {
        let v = self
            .sup
            .request(M_STATUS, serde_json::Value::Null)
            .await
            .ok()?;
        v.get("loaded")?.get("id")?.as_str().map(ToOwned::to_owned)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::supervisor::WorkerHalves;
    use crate::worker::N_CHAT_CHUNK;
    use parking_lot::Mutex;
    use serde_json::json;
    use tokio::io::AsyncWriteExt;

    // ── Test seam (mirrored from supervisor::tests::make_supervisor) ──────────

    /// Build a `Supervisor` plus duplex test handles.
    fn make_supervisor() -> (
        Supervisor,
        tokio::io::DuplexStream, // test_write: write responses → supervisor reads
        tokio::io::DuplexStream, // test_read:  supervisor writes requests → test reads
    ) {
        let (sup_write, test_read) = tokio::io::duplex(64 * 1024);
        let (test_write, sup_read) = tokio::io::duplex(64 * 1024);

        let sup_write_cell = Arc::new(Mutex::new(Some(sup_write)));
        let sup_read_cell = Arc::new(Mutex::new(Some(sup_read)));

        let sup = Supervisor::with_factory(Box::new(move |_ring, _model| {
            let write =
                sup_write_cell
                    .lock()
                    .take()
                    .ok_or_else(|| HiggsError::WorkerSpawnFailed {
                        source: std::io::Error::other("mock: no more write halves"),
                    })?;
            let read =
                sup_read_cell
                    .lock()
                    .take()
                    .ok_or_else(|| HiggsError::WorkerSpawnFailed {
                        source: std::io::Error::other("mock: no more read halves"),
                    })?;
            Ok(WorkerHalves {
                write: Box::new(write),
                read: Box::new(read),
                proc: None,
            })
        }));

        sup.start_for("test-model").expect("mock start");
        (sup, test_write, test_read)
    }

    async fn write_response(
        stream: &mut tokio::io::DuplexStream,
        id: u64,
        result: serde_json::Value,
    ) {
        use crate::rpc::{encode, RpcFrame, RpcResponse};
        let line = encode(&RpcFrame::Response(RpcResponse {
            jsonrpc: "2.0".into(),
            id,
            result: Some(result),
            error: None,
        }));
        stream
            .write_all(format!("{line}\n").as_bytes())
            .await
            .unwrap();
        stream.flush().await.unwrap();
    }

    // ── Test 1: default config paths ─────────────────────────────────────────

    #[test]
    fn default_config_paths() {
        let home = match dirs::home_dir() {
            Some(h) => h,
            None => return, // skip if no home dir
        };

        let cfg = HiggsConfig::default();

        let has_suffix = |dirs: &[PathBuf], suffix: &str| dirs.iter().any(|p| p.ends_with(suffix));

        assert!(
            has_suffix(&cfg.lmstudio_dirs, ".lmstudio/models")
                || cfg
                    .lmstudio_dirs
                    .iter()
                    .any(|p| p.ends_with("lm-studio/models")),
            "lmstudio_dirs should contain .lmstudio/models or lm-studio/models"
        );
        assert!(
            cfg.hf_dirs
                .iter()
                .any(|p| { p == &home.join(".cache").join("huggingface").join("hub") }),
            "hf_dirs must use ~/.cache/huggingface/hub (not XDG cache_dir)"
        );
        assert!(
            cfg.ollama_dirs
                .iter()
                .any(|p| p.ends_with(".ollama/models")),
            "ollama_dirs should contain .ollama/models"
        );
    }

    // ── Test 2: scan runs host-side with no worker ───────────────────────────

    /// `scan()` runs host-side (pure Rust, no worker RPC): with a fresh facade
    /// that never spawned a worker and empty config dirs, it returns `Ok(empty)`.
    #[tokio::test]
    async fn scan_runs_host_side_without_worker() {
        // Empty config dirs → nothing to scan → Ok(empty). The point is that no
        // worker is live (start() never called) yet scan succeeds.
        let higgs = Higgs::new(HiggsConfig {
            lmstudio_dirs: vec![],
            hf_dirs: vec![],
            ollama_dirs: vec![],
            default_load: HiggsConfig::default().default_load,
        });

        let models = higgs.scan().await.expect("host-side scan should succeed");
        assert!(models.is_empty(), "empty dirs yield no models");

        // No worker was ever spawned: status reports worker_alive=false.
        let st = higgs.status().await.expect("status");
        assert!(!st.worker_alive, "scan must not spawn a worker");
    }

    // ── Test 3: load then status maps ─────────────────────────────────────────

    #[tokio::test]
    async fn load_then_status_maps() {
        use tokio::io::{AsyncBufReadExt, BufReader};

        let (sup, mut test_write, test_read) = make_supervisor();
        // `load` resolves the GGUF path host-side, so point config at a fixture.
        let dir = tempfile::TempDir::new().unwrap();
        crate::serve::test_support::write_gguf_fixture(dir.path(), "org/model");
        let cfg = HiggsConfig {
            lmstudio_dirs: vec![dir.path().to_path_buf()],
            hf_dirs: vec![],
            ollama_dirs: vec![],
            default_load: HiggsConfig::default().default_load,
        };
        let higgs = Higgs {
            sup: Arc::new(sup),
            config: parking_lot::Mutex::new(cfg),
            lifecycle: tokio::sync::Mutex::new(()),
        };
        let mut events_rx = higgs.events();
        // `load`/`status` run a host-side scan (on a blocking thread) before each
        // RPC, so drive the operation future concurrently with the responder: the
        // responder reads the request line (proving the id is pending) and only
        // then writes the reply. A fixed pre-sleep + sequential write would race
        // the scan and drop the response.
        let mut lines = BufReader::new(test_read).lines();

        // Issue load — mock responds with ok.
        let load_fut = higgs.load("org/model", None);
        let (load_res, _) = tokio::join!(load_fut, async {
            lines.next_line().await.unwrap().expect("M_LOAD request");
            write_response(&mut test_write, 1, json!({"id": "org/model"})).await;
        });
        load_res.expect("load should succeed");

        // ModelLoaded event must arrive.
        let ev = tokio::time::timeout(std::time::Duration::from_millis(100), events_rx.recv())
            .await
            .expect("timeout")
            .expect("recv");
        assert!(matches!(ev, HiggsEvent::ModelLoaded { id } if id == "org/model"));

        // Issue status — mock responds with loaded info.
        let status_fut = higgs.status();
        let (st, _) = tokio::join!(status_fut, async {
            lines.next_line().await.unwrap().expect("M_STATUS request");
            write_response(
                &mut test_write,
                2,
                json!({
                    "loaded": { "id": "org/model", "ctx_len": 4096, "gpu_layers": 4294967295u64, "threads": 4 },
                    "models_scanned": 3,
                }),
            )
            .await;
        });
        let st = st.expect("status should succeed");
        assert!(st.worker_alive);
        // models_on_disk now comes from a host-side scan of the config dirs
        // (one GGUF fixture), not the worker's `models_scanned`.
        assert_eq!(st.models_on_disk, 1);
        let li = st.loaded.expect("loaded should be Some");
        assert_eq!(li.id, "org/model");
        assert_eq!(li.ctx_len, 4096);
        assert_eq!(li.gpu_layers, u32::MAX);
    }

    // ── Test 3b: status loaded info includes model metadata ──────────────────

    #[tokio::test]
    async fn status_loaded_info_includes_model_metadata() {
        use tokio::io::{AsyncBufReadExt, BufReader};

        let (sup, mut test_write, test_read) = make_supervisor();
        // Metadata now comes from the HOST scan, not the worker response: point
        // config at a GGUF fixture (arch=llama, ctx_train=4096, chat template)
        // so the host-scanned `HiggsModel` enriches the worker-reported `loaded`.
        let dir = tempfile::TempDir::new().unwrap();
        crate::serve::test_support::write_gguf_fixture(dir.path(), "org/model");
        let cfg = HiggsConfig {
            lmstudio_dirs: vec![dir.path().to_path_buf()],
            hf_dirs: vec![],
            ollama_dirs: vec![],
            default_load: HiggsConfig::default().default_load,
        };
        let higgs = Higgs {
            sup: Arc::new(sup),
            config: parking_lot::Mutex::new(cfg),
            lifecycle: tokio::sync::Mutex::new(()),
        };

        // `status` runs a host-side scan (on a blocking thread) before M_STATUS,
        // so drive the future concurrently with a responder that reads the
        // request line before replying — a fixed sleep would race the scan. The
        // worker reports only id/ctx_len/gpu_layers/threads; the metadata fields
        // are filled host-side from the fixture.
        let mut lines = BufReader::new(test_read).lines();
        let status_fut = higgs.status();
        let (st, _) = tokio::join!(status_fut, async {
            lines.next_line().await.unwrap().expect("M_STATUS request");
            write_response(
                &mut test_write,
                1,
                json!({
                    "loaded": {
                        "id": "org/model",
                        "ctx_len": 4096,
                        "gpu_layers": 99,
                        "threads": 4,
                    },
                    "models_scanned": 1,
                }),
            )
            .await;
        });
        let st = st.expect("status should succeed");
        let li = st.loaded.expect("loaded should be Some");
        assert_eq!(li.id, "org/model");
        assert_eq!(li.arch.as_deref(), Some("llama"));
        assert_eq!(li.quant.as_deref(), Some("Q4_K_M"));
        assert_eq!(li.max_context_length, Some(4096));
        assert!(li.size_bytes.is_some(), "size_bytes from fixture file");
        assert_eq!(li.has_chat_template, Some(true));
    }

    // ── Test 3c: host-resolved load carries the GGUF path (no worker scan) ────
    //
    // Regression: after scan moved host-side, the worker's ModelStore is empty,
    // so the worker can only resolve a path if the host puts it in M_LOAD params.
    // This asserts `load(id)` resolves the path host-side and includes it in the
    // M_LOAD request — proving the load works WITHOUT a prior worker scan. If the
    // path-passing were removed (worker fell back to its empty `store.get(id)`),
    // the params would carry no `path` and this test would fail.
    #[tokio::test]
    async fn load_carries_host_resolved_path() {
        use tokio::io::{AsyncBufReadExt, BufReader};

        let (sup, mut test_write, test_read) = make_supervisor();

        // Real GGUF fixture so the host-side scan discovers the id with a path.
        let dir = tempfile::TempDir::new().unwrap();
        crate::serve::test_support::write_gguf_fixture(dir.path(), "org/model");
        let cfg = HiggsConfig {
            lmstudio_dirs: vec![dir.path().to_path_buf()],
            hf_dirs: vec![],
            ollama_dirs: vec![],
            default_load: HiggsConfig::default().default_load,
        };
        let higgs = Higgs {
            sup: Arc::new(sup),
            config: parking_lot::Mutex::new(cfg),
            lifecycle: tokio::sync::Mutex::new(()),
        };

        // Drive the load. `load` first runs a host-side scan (on a blocking
        // thread) before sending M_LOAD, so drive the load future concurrently
        // with a responder that reads the request line (proving id=1 is pending)
        // before replying. A fixed pre-sleep would race the scan and drop the
        // response.
        let mut lines = BufReader::new(test_read).lines();
        let load_fut = higgs.load("org/model", None);
        let (load_res, line) = tokio::join!(load_fut, async {
            let line = lines.next_line().await.unwrap().expect("M_LOAD request");
            write_response(&mut test_write, 1, json!({"id": "org/model"})).await;
            line
        });
        load_res.expect("host-resolved load should succeed");

        // The M_LOAD request carries the fixture path resolved host-side.
        let v: serde_json::Value = serde_json::from_str(&line).unwrap();
        assert_eq!(v["method"], M_LOAD);
        let path = v["params"]["path"].as_str().expect("path in M_LOAD params");
        assert!(path.ends_with(".gguf"), "path was: {path}");
        assert!(path.contains("org/model"), "path was: {path}");
    }

    // ── Test 4: chat_stream delivers chunks and outcome ────────────────────────
    //
    // Verifies end-to-end: alloc_request_id allocates id=1; chat_stream registers
    // the sink under that id and sends M_CHAT with request_id=1; the test injects
    // N_CHAT_CHUNK notifications tagged request_id=1; route_notification delivers
    // them to rx; the final response for RPC id=1 resolves the outcome handle.

    #[tokio::test]
    async fn chat_stream_delivers() {
        let (sup, mut test_write, _test_read) = make_supervisor();
        let higgs = Higgs {
            sup: Arc::new(sup),
            config: parking_lot::Mutex::new(HiggsConfig::default()),
            lifecycle: tokio::sync::Mutex::new(()),
        };

        let (mut rx, handle) = higgs
            .chat_stream(
                r#"[{"role":"user","content":"hi"}]"#.to_owned(),
                256,
                0.7,
                None,
            )
            .await
            .expect("chat_stream should succeed");

        // Inject chunk notifications tagged with request_id=1 (the first allocated id).
        use crate::rpc::{encode, RpcFrame, RpcNotification};
        for delta in &["hel", "lo"] {
            let notif = encode(&RpcFrame::Notification(RpcNotification {
                jsonrpc: "2.0".into(),
                method: N_CHAT_CHUNK.into(),
                params: json!({ "request_id": 1u64, "delta": delta }),
            }));
            test_write
                .write_all(format!("{notif}\n").as_bytes())
                .await
                .unwrap();
        }
        test_write.flush().await.unwrap();

        tokio::time::sleep(std::time::Duration::from_millis(30)).await;

        // Final response for M_CHAT (RPC id=1) — includes token counts.
        write_response(
            &mut test_write,
            1,
            json!({"content": "hello", "finish_reason": "stop", "prompt_tokens": 10, "completion_tokens": 3}),
        )
        .await;

        let outcome = tokio::time::timeout(std::time::Duration::from_millis(500), handle)
            .await
            .expect("join timeout")
            .expect("join error")
            .expect("chat outcome error");

        assert_eq!(outcome.content, "hello");
        assert_eq!(outcome.finish_reason, "stop");
        assert_eq!(outcome.prompt_tokens, 10);
        assert_eq!(outcome.completion_tokens, 3);

        // Chunks must have arrived.
        let chunk1 = rx.try_recv().expect("chunk 1");
        let chunk2 = rx.try_recv().expect("chunk 2");
        assert_eq!(chunk1, "hel");
        assert_eq!(chunk2, "lo");
    }

    // ── Test 5: chat_stream against dead worker removes sink ─────────────────

    /// When the chat request fails (write_tx is None — worker not running), the
    /// spawned task removes the sink on the error path so the map stays clean.
    #[tokio::test]
    async fn chat_stream_dead_worker_removes_sink() {
        // Build a Supervisor with no worker halves — factory always fails.
        let sup = crate::supervisor::Supervisor::with_factory(Box::new(|_ring, _model| {
            Err(HiggsError::WorkerSpawnFailed {
                source: std::io::Error::other("mock: no worker"),
            })
        }));
        // Do NOT call start() — write_tx stays None (dead worker).

        let higgs = Higgs {
            sup: Arc::new(sup),
            config: parking_lot::Mutex::new(HiggsConfig::default()),
            lifecycle: tokio::sync::Mutex::new(()),
        };

        // chat_stream registers the sink then the spawned task encounters dead worker.
        let (_rx, handle) = higgs
            .chat_stream(
                r#"[{"role":"user","content":"hi"}]"#.to_owned(),
                8,
                0.0,
                None,
            )
            .await
            .expect("chat_stream itself should not fail");

        // The spawned task must return an Err (worker dead).
        let result = tokio::time::timeout(std::time::Duration::from_millis(200), handle)
            .await
            .expect("join timeout")
            .expect("join error");
        assert!(result.is_err(), "chat against dead worker must fail");

        // After the failed request, the sink map must be empty (remove_chat_sink was called).
        assert_eq!(
            higgs.sup.chat_sinks_count(),
            0,
            "chat_sinks must be empty after failed request"
        );
    }
}
