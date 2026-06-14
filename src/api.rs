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
use crate::worker::{M_CHAT, M_LOAD, M_SCAN, M_STATUS, M_UNLOAD};

// ── HiggsConfig ───────────────────────────────────────────────────────────────

/// Host-supplied configuration (the host maps its own config table onto this).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, ts_rs::TS)]
#[ts(export, export_to = "../../../frontend/src/lib/generated/")]
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

/// Info about the currently loaded model.
#[derive(Debug, Clone, serde::Serialize, ts_rs::TS)]
#[ts(export, export_to = "../../../frontend/src/lib/generated/")]
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
    #[ts(optional)]
    pub arch: Option<String>,
    /// Quantization tag (e.g. `Q4_K_M`), if present.
    #[ts(optional)]
    pub quant: Option<String>,
    /// Training context length from GGUF header (model's maximum). Distinct from
    /// `ctx_len` which is the actually loaded window size.
    #[ts(type = "number")]
    #[ts(optional)]
    pub max_context_length: Option<u64>,
    /// File size in bytes.
    #[ts(type = "number")]
    #[ts(optional)]
    pub size_bytes: Option<u64>,
    /// Whether `tokenizer.chat_template` is present in the GGUF header.
    #[ts(optional)]
    pub has_chat_template: Option<bool>,
}

/// Live status snapshot returned by [`Higgs::status`].
#[derive(Debug, Clone, serde::Serialize, ts_rs::TS)]
#[ts(export, export_to = "../../../frontend/src/lib/generated/")]
pub struct HiggsStatus {
    /// Whether the worker process is currently alive.
    pub worker_alive: bool,
    /// Info about the loaded model, if any.
    #[ts(optional)]
    pub loaded: Option<LoadedInfo>,
    /// Number of models discovered in the last scan.
    #[ts(type = "number")]
    pub models_on_disk: u32,
}

/// Final outcome of a completed chat request.
#[derive(Debug, Clone)]
pub struct ChatOutcome {
    /// Concatenated full text from all chunks.
    pub content: String,
    /// OpenAI finish_reason ("stop" or "length").
    pub finish_reason: String,
}

// ── Higgs ─────────────────────────────────────────────────────────────────────

/// The in-process handle to the higgs runtime. One instance per host app.
///
/// Constructing `Higgs` does not start the worker; call [`start`](Self::start)
/// when the host is ready to serve requests.
pub struct Higgs {
    sup: Arc<Supervisor>,
    config: parking_lot::Mutex<HiggsConfig>,
}

impl Higgs {
    /// Construct the facade WITHOUT spawning the worker.
    ///
    /// Call [`start`](Self::start) when the host is ready.
    pub fn new(config: HiggsConfig) -> Self {
        Self {
            sup: Arc::new(Supervisor::spawn()),
            config: parking_lot::Mutex::new(config),
        }
    }

    /// Spawn the worker and issue an initial scan.
    ///
    /// Returns `Err` [HG006] if the worker process cannot be started.
    pub async fn start(&self) -> Result<(), HiggsError> {
        self.sup.start()?;
        // Best-effort initial scan; failures are non-fatal at startup.
        let _ = self.scan().await;
        Ok(())
    }

    /// Gracefully shut down the worker (2 s timeout).
    pub async fn stop(&self) {
        self.sup.stop().await;
    }

    /// Scan all configured model directories and return the discovered models.
    ///
    /// Records the scan params for post-restart replay.
    pub async fn scan(&self) -> Result<Vec<HiggsModel>, HiggsError> {
        let (lmstudio, hf, ollama) = {
            let cfg = self.config.lock();
            let to_str_array = |dirs: &[PathBuf]| {
                dirs.iter()
                    .filter_map(|p| p.to_str().map(|s| serde_json::Value::String(s.to_owned())))
                    .collect::<Vec<_>>()
            };
            (
                to_str_array(&cfg.lmstudio_dirs),
                to_str_array(&cfg.hf_dirs),
                to_str_array(&cfg.ollama_dirs),
            )
        };
        let params = json!({ "lmstudio": lmstudio, "hf": hf, "ollama": ollama });
        let result = self.sup.request(M_SCAN, params.clone()).await?;
        let models: Vec<HiggsModel> =
            serde_json::from_value(result).map_err(|e| HiggsError::WorkerRpc {
                method: M_SCAN.into(),
                message: format!("response parse failed: {e}"),
            })?;
        self.sup.record_last_scan(params);
        Ok(models)
    }

    /// Load a model by HuggingFace repo id.
    ///
    /// `params` overrides `default_load` when supplied. On success, records the
    /// load params for post-restart replay and emits [`HiggsEvent::ModelLoaded`].
    pub async fn load(&self, id: &str, params: Option<LoadParams>) -> Result<(), HiggsError> {
        let p = params.unwrap_or_else(|| self.config.lock().default_load.clone());
        let req_params = json!({
            "id": id,
            "ctx_len": p.ctx_len,
            "gpu_layers": p.gpu_layers,
            "threads": p.threads,
        });
        self.sup.request(M_LOAD, req_params.clone()).await?;
        self.sup.record_last_load(req_params);
        self.sup.emit(HiggsEvent::ModelLoaded { id: id.to_owned() });
        Ok(())
    }

    /// Unload the current model.
    ///
    /// Emits [`HiggsEvent::ModelUnloaded`] with an empty id when no model id
    /// is available at the facade layer (v1 limitation; worker tracks it).
    pub async fn unload(&self) -> Result<(), HiggsError> {
        // TODO(v2): single RPC — status+unload is TOCTOU if worker state changes between calls (v1: worker serializes, benign)
        // Capture id from status before unloading so the event carries it.
        let id = self.loaded_id().await.unwrap_or_default();
        self.sup.request(M_UNLOAD, serde_json::Value::Null).await?;
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

        let loaded = v.get("loaded").and_then(|l| {
            if l.is_null() {
                return None;
            }
            Some(LoadedInfo {
                id: l.get("id")?.as_str()?.to_owned(),
                ctx_len: l.get("ctx_len")?.as_u64()? as u32,
                gpu_layers: l.get("gpu_layers")?.as_u64()? as u32,
                threads: l.get("threads")?.as_u64()? as u32,
                arch: l
                    .get("arch")
                    .and_then(|v| v.as_str())
                    .map(ToOwned::to_owned),
                quant: l
                    .get("quant")
                    .and_then(|v| v.as_str())
                    .map(ToOwned::to_owned),
                max_context_length: l
                    .get("max_context_length")
                    .and_then(serde_json::Value::as_u64),
                size_bytes: l.get("size_bytes").and_then(serde_json::Value::as_u64),
                has_chat_template: l
                    .get("has_chat_template")
                    .and_then(serde_json::Value::as_bool),
            })
        });

        let models_on_disk = v
            .get("models_scanned")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0) as u32;

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
        messages: Vec<(String, String)>,
        max_tokens: usize,
        temperature: f32,
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

        let msgs: Vec<serde_json::Value> = messages
            .into_iter()
            .map(|(role, content)| json!({ "role": role, "content": content }))
            .collect();

        let handle = tokio::spawn(async move {
            let result = sup
                .request_with_id(
                    request_id,
                    M_CHAT,
                    json!({
                        "request_id": request_id,
                        "messages": msgs,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
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

            Ok(ChatOutcome {
                content,
                finish_reason,
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

        let sup = Supervisor::with_factory(Box::new(move |_ring| {
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
            })
        }));

        sup.start().expect("mock start");
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

    // ── Test 2: scan records and returns ─────────────────────────────────────

    #[tokio::test]
    async fn scan_records_and_returns() {
        let (sup, mut test_write, _test_read) = make_supervisor();

        // Build Higgs wrapping this test supervisor.
        let higgs = Higgs {
            sup: Arc::new(sup),
            config: parking_lot::Mutex::new(HiggsConfig::default()),
        };

        // Mock worker responds to higgs/scan with one model.
        let model_json = json!([{
            "id": "org/model",
            "path": "/models/model.gguf",
            "size_bytes": 4000000000u64,
            "quant": "Q4_K_M",
            "source": "LmStudio",
            "arch": "llama",
            "ctx_train": 4096u64,
            "has_chat_template": true,
        }]);

        let scan_fut = higgs.scan();
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        write_response(&mut test_write, 1, model_json.clone()).await;

        let models = scan_fut.await.expect("scan should succeed");
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "org/model");

        // Verify supervisor recorded the params for replay.
        let recorded = higgs.sup.last_scan_params();
        assert!(recorded.is_some(), "last_scan should be recorded");
        let recorded = recorded.unwrap();
        assert!(recorded.get("lmstudio").is_some());
        assert!(recorded.get("hf").is_some());
        assert!(recorded.get("ollama").is_some());
    }

    // ── Test 3: load then status maps ─────────────────────────────────────────

    #[tokio::test]
    async fn load_then_status_maps() {
        let (sup, mut test_write, _test_read) = make_supervisor();
        let higgs = Higgs {
            sup: Arc::new(sup),
            config: parking_lot::Mutex::new(HiggsConfig::default()),
        };
        let mut events_rx = higgs.events();

        // Issue load — mock responds with ok.
        let load_fut = higgs.load("org/model", None);
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        write_response(&mut test_write, 1, json!({"id": "org/model"})).await;
        load_fut.await.expect("load should succeed");

        // ModelLoaded event must arrive.
        let ev = tokio::time::timeout(std::time::Duration::from_millis(100), events_rx.recv())
            .await
            .expect("timeout")
            .expect("recv");
        assert!(matches!(ev, HiggsEvent::ModelLoaded { id } if id == "org/model"));

        // Issue status — mock responds with loaded info.
        let status_fut = higgs.status();
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        write_response(
            &mut test_write,
            2,
            json!({
                "loaded": { "id": "org/model", "ctx_len": 4096, "gpu_layers": 4294967295u64, "threads": 4 },
                "models_scanned": 3,
            }),
        )
        .await;

        let st = status_fut.await.expect("status should succeed");
        assert!(st.worker_alive);
        assert_eq!(st.models_on_disk, 3);
        let li = st.loaded.expect("loaded should be Some");
        assert_eq!(li.id, "org/model");
        assert_eq!(li.ctx_len, 4096);
        assert_eq!(li.gpu_layers, u32::MAX);
    }

    // ── Test 3b: status loaded info includes model metadata ──────────────────

    #[tokio::test]
    async fn status_loaded_info_includes_model_metadata() {
        let (sup, mut test_write, _test_read) = make_supervisor();
        let higgs = Higgs {
            sup: Arc::new(sup),
            config: parking_lot::Mutex::new(HiggsConfig::default()),
        };

        let status_fut = higgs.status();
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        write_response(
            &mut test_write,
            1,
            json!({
                "loaded": {
                    "id": "org/model",
                    "ctx_len": 4096,
                    "gpu_layers": 99,
                    "threads": 4,
                    "arch": "llama",
                    "quant": "Q4_K_M",
                    "max_context_length": 8192u64,
                    "size_bytes": 4000000000u64,
                    "has_chat_template": true,
                },
                "models_scanned": 1,
            }),
        )
        .await;

        let st = status_fut.await.expect("status should succeed");
        let li = st.loaded.expect("loaded should be Some");
        assert_eq!(li.id, "org/model");
        assert_eq!(li.arch.as_deref(), Some("llama"));
        assert_eq!(li.quant.as_deref(), Some("Q4_K_M"));
        assert_eq!(li.max_context_length, Some(8192));
        assert_eq!(li.size_bytes, Some(4000000000));
        assert_eq!(li.has_chat_template, Some(true));
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
        };

        let (mut rx, handle) = higgs
            .chat_stream(vec![("user".into(), "hi".into())], 256, 0.7)
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

        // Final response for M_CHAT (RPC id=1).
        write_response(
            &mut test_write,
            1,
            json!({"content": "hello", "finish_reason": "stop"}),
        )
        .await;

        let outcome = tokio::time::timeout(std::time::Duration::from_millis(500), handle)
            .await
            .expect("join timeout")
            .expect("join error")
            .expect("chat outcome error");

        assert_eq!(outcome.content, "hello");
        assert_eq!(outcome.finish_reason, "stop");

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
        let sup = crate::supervisor::Supervisor::with_factory(Box::new(|_ring| {
            Err(HiggsError::WorkerSpawnFailed {
                source: std::io::Error::other("mock: no worker"),
            })
        }));
        // Do NOT call start() — write_tx stays None (dead worker).

        let higgs = Higgs {
            sup: Arc::new(sup),
            config: parking_lot::Mutex::new(HiggsConfig::default()),
        };

        // chat_stream registers the sink then the spawned task encounters dead worker.
        let (_rx, handle) = higgs
            .chat_stream(vec![("user".into(), "hi".into())], 8, 0.0)
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
