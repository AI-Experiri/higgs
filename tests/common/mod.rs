#![allow(dead_code)] // shared test helpers; each test binary uses a subset
//! Shared harness for higgs black-box integration tests.
//!
//! Spawns the real `higgs` binary on a localhost port, waits for the
//! worker to come up, and kills the process (and its worker) on drop. Tests
//! drive the server purely over HTTP — exactly how an external client would.
//!
//! To run against a real, tiny model in CI the harness stages a small GGUF into
//! a temp LM-Studio-style scan root and points the spawned binary at it via
//! `HIGGS_MODEL_DIR`. The model is `ggml-org`'s ~1MB `stories260K.gguf` (a real
//! llama-arch toy GGUF that loads and generates) — enough to exercise the full
//! load → chat → unload engine path. The default path is the on-disk HF-cache
//! copy; override with `HIGGS_TEST_GGUF`.

use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::sync::Arc;
use std::time::Duration;

use higgs::{Higgs, HiggsConfig};
use tempfile::TempDir;

/// HuggingFace repo id the staged tiny model is discovered under. The harness
/// writes the GGUF to `<scan-root>/higgs-test/stories260k/`, so the LM-Studio
/// scanner (`<root>/{org}/{model}/*.gguf`) lists it under this id.
pub const TINY_MODEL_ID: &str = "higgs-test/stories260k";

/// The on-disk tiny GGUF used when `HIGGS_TEST_GGUF` is unset: `ggml-org`'s
/// `stories260K.gguf`, ~1MB, a real llama-arch toy model that loads + generates.
const DEFAULT_TINY_GGUF: &str = "/Users/bicepjai/.cache/huggingface/hub/models--ggml-org--models/snapshots/499bc8821c6b12b4e53c5bffcb21ec206f212d81/tinyllamas/stories260K.gguf";

/// Resolve the tiny GGUF path: `HIGGS_TEST_GGUF` if set, else the default
/// on-disk copy. Returns `None` (so the test SKIPs) when the file is absent.
pub fn tiny_gguf_path() -> Option<PathBuf> {
    let p = std::env::var("HIGGS_TEST_GGUF").unwrap_or_else(|_| DEFAULT_TINY_GGUF.to_owned());
    let path = PathBuf::from(p);
    path.is_file().then_some(path)
}

/// A running `higgs` child process. Killed (with its worker) on drop.
///
/// Holds the staging `TempDir` so the scan root outlives the server.
pub struct ServerGuard {
    child: Child,
    /// Base URL, e.g. `http://127.0.0.1:11500`.
    pub base: String,
    /// Staging dir for the tiny model (kept alive for the server's lifetime).
    _model_dir: TempDir,
    /// Isolated `HIGGS_HOME` (kept alive for the server's lifetime) so the spawned server
    /// never picks up the developer's / CI's real `~/.higgs` — e.g. an `api_keys.json` that
    /// would enable auth and 401 these no-token tests.
    _home: TempDir,
}

impl Drop for ServerGuard {
    fn drop(&mut self) {
        // SIGTERM (not SIGKILL): higgs shuts down gracefully — stopping
        // its worker and running at-exit handlers — then we reap it. Under
        // coverage instrumentation the graceful exit is what flushes the spawned
        // process's profile; a hard kill() would discard it.
        // SAFETY: a plain kill(2) on our own child's pid.
        unsafe {
            libc::kill(self.child.id() as libc::pid_t, libc::SIGTERM);
        }
        // BOUNDED grace, then SIGKILL: a server wedged mid-load (e.g. stuck in
        // the llama.cpp FFI) ignores SIGTERM, and an unconditional `wait()`
        // here has twice hung an entire test binary for HOURS on one flaky
        // load. The happy path reaps within the first poll or two (profile
        // flushed as before); only a wedged child is hard-killed, losing that
        // one process's coverage profile — the right trade against a gate that
        // never finishes.
        for _ in 0..100 {
            match self.child.try_wait() {
                Ok(Some(_)) => return,
                Ok(None) => std::thread::sleep(Duration::from_millis(100)),
                Err(_) => break,
            }
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl ServerGuard {
    /// The spawned `higgs` server process pid (for tests that need to find its worker child).
    pub fn pid(&self) -> u32 {
        self.child.id()
    }

    /// Absolute path of the staged GGUF for `id` (the `stage_models` layout
    /// `<model_dir>/<id>/stories260K.gguf`). Lets staleness tests mutate the file
    /// so its `model_file_sig` changes and a saved profile reads back as stale.
    pub fn staged_gguf(&self, id: &str) -> PathBuf {
        self._model_dir.path().join(id).join("stories260K.gguf")
    }

    /// The server's isolated `HIGGS_HOME` — where `models.json` / `config.json`
    /// live. Lets a test induce a persistence failure (e.g. occupy `models.json`
    /// with a directory so the store flush can't write the file).
    pub fn home(&self) -> &std::path::Path {
        self._home.path()
    }
}

/// Stage `gguf` into a temp LM-Studio layout (`<tmp>/higgs-test/stories260k/`)
/// so the scanner discovers it under [`TINY_MODEL_ID`]. Returns the temp dir
/// (the scan root) — keep it alive for the server's lifetime. The GGUF is
/// copied (not symlinked) so it stays inside the scan root and passes higgs's
/// path-within-roots guard.
pub fn stage_tiny_model(gguf: &Path) -> TempDir {
    stage_models(gguf, &[TINY_MODEL_ID])
}

/// Stage `gguf` into a temp LM-Studio layout under EACH `id` (`<tmp>/{org}/{model}/`), so the
/// scanner discovers one model per id. Copies (not symlinks) so each stays inside the scan root
/// and passes higgs's path-within-roots guard. Use for multi-model tests (each distinct id →
/// its own worker on load).
pub fn stage_models(gguf: &Path, ids: &[&str]) -> TempDir {
    let dir = TempDir::new().expect("create staging dir");
    for id in ids {
        let model_dir = dir.path().join(id); // `org/model` nests via the path separator
        std::fs::create_dir_all(&model_dir).expect("create model dir");
        std::fs::copy(gguf, model_dir.join("stories260K.gguf")).expect("copy tiny gguf");
    }
    dir
}

/// Spawn `higgs` on `127.0.0.1:{port}` with the tiny model staged into a temp
/// scan root (passed via `HIGGS_MODEL_DIR`), and wait until `/api/higgs/status`
/// answers (listener bound). `gguf` is the source GGUF (see [`tiny_gguf_path`]).
/// A fresh higgs has NO worker yet — spawn-on-load brings one up on the first
/// `/api/higgs/models/load`.
//
// The spawned child is reaped by `ServerGuard::drop` (kill + wait), so the
// zombie-process lint is a false positive — clippy can't see the Drop impl.
#[allow(clippy::zombie_processes)]
pub async fn spawn_with_tiny_model(port: u16, gguf: &Path) -> ServerGuard {
    spawn_with_models(port, gguf, &[TINY_MODEL_ID]).await
}

/// Prepare (autotune) the tiny model so a subsequent JIT chat is allowed by the
/// readiness gate — JIT refuses an un-profiled model. Tests that intentionally
/// exercise the JIT happy path call this once after spawn; tests that explicitly
/// `POST /models/load` don't need it (an explicit load bypasses the gate).
pub async fn prepare_tiny(base: &str) {
    let r = reqwest::Client::new()
        .post(format!("{base}/api/higgs/models/tune"))
        .json(&serde_json::json!({ "id": TINY_MODEL_ID }))
        .send()
        .await
        .expect("send tune request");
    assert!(
        r.status().is_success(),
        "Prepare (tune) tiny model succeeded, got {}",
        r.status()
    );
}

/// The chat-template/parser test fleet: REAL small instruct models (downloaded
/// from HF via `scripts/fetch_test_fleet.sh`, test-only — never under the live
/// app's scan root). The dir is an LM-Studio layout
/// (`<root>/{org}/{model}/*.gguf`) so both `ModelStore::scan` and a spawned
/// server catalog it directly. Override with `HIGGS_TEST_FLEET`.
///
/// Default matches the fetch script: `$HOME/.cache/higgs-test-models` (higgs
/// targets Unix/macOS only, so `HOME` is always set).
fn default_fleet_dir() -> PathBuf {
    PathBuf::from(std::env::var("HOME").expect("HOME set on Unix/macOS"))
        .join(".cache/higgs-test-models")
}

/// `(fixture slug, catalog id, gguf file name)` for each fleet model.
pub const FLEET: [(&str, &str, &str); 4] = [
    ("qwen3-0.6b", "qwen/Qwen3-0.6B", "Qwen3-0.6B-Q8_0.gguf"),
    (
        "gemma-3-1b-it",
        "google/gemma-3-1b-it",
        "gemma-3-1b-it-Q4_K_M.gguf",
    ),
    (
        "llama-3.2-1b",
        "meta-llama/Llama-3.2-1B-Instruct",
        "Llama-3.2-1B-Instruct-Q4_0.gguf",
    ),
    (
        "deepseek-r1-1.5b",
        "deepseek/DeepSeek-R1-Distill-Qwen-1.5B",
        "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf",
    ),
];

/// Resolve the fleet scan root: `HIGGS_TEST_FLEET` if set, else the default
/// cache dir. Returns `None` (so the test SKIPs) unless EVERY fleet GGUF is
/// present — a partial fleet would make per-model tests pass/fail by accident
/// of which download finished.
pub fn fleet_dir() -> Option<PathBuf> {
    let root = std::env::var("HIGGS_TEST_FLEET")
        .map(PathBuf::from)
        .unwrap_or_else(|_| default_fleet_dir());
    FLEET
        .iter()
        .all(|(_, id, file)| root.join(id).join(file).is_file())
        .then_some(root)
}

/// Spawn `higgs` with `HIGGS_MODEL_DIR` pointed at an EXISTING scan root (the
/// fleet dir) instead of a staged temp copy — the fleet GGUFs are hundreds of
/// MB each, so copying them per test is not viable. The isolated `HIGGS_HOME`
/// still applies.
#[allow(clippy::zombie_processes)]
pub async fn spawn_with_model_root(port: u16, root: &Path) -> ServerGuard {
    // Dummy staging dir: ServerGuard owns TempDirs for lifetime symmetry.
    let staged = TempDir::new().expect("create dummy staging dir");
    let home = TempDir::new().expect("create temp HIGGS_HOME");
    let child = Command::new(env!("CARGO_BIN_EXE_higgs"))
        .env("HIGGS_BIND", "127.0.0.1")
        .env("HIGGS_PORT", port.to_string())
        .env("HIGGS_MODEL_DIR", root)
        .env("HIGGS_HOME", home.path())
        .env("HIGGS_HF_ENDPOINT", "http://127.0.0.1:1")
        .env("RUST_LOG", "warn")
        .spawn()
        .expect("spawn higgs");
    let base = format!("http://127.0.0.1:{port}");
    let client = reqwest::Client::new();
    for _ in 0..150 {
        if let Ok(r) = client.get(format!("{base}/api/higgs/status")).send().await {
            if r.status().is_success() {
                return ServerGuard {
                    child,
                    base,
                    _model_dir: staged,
                    _home: home,
                };
            }
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    panic!("higgs never became ready on {base}");
}

/// Like [`spawn_with_tiny_model`] but stages one model per id in `ids` (each loadable under that
/// id), for multi-model / multi-worker tests.
#[allow(clippy::zombie_processes)]
pub async fn spawn_with_models(port: u16, gguf: &Path, ids: &[&str]) -> ServerGuard {
    let staged = stage_models(gguf, ids);
    // Isolated home so the server never reads the machine's real ~/.higgs (a present
    // api_keys.json there would turn auth ON and 401 these no-token tests).
    let home = TempDir::new().expect("create temp HIGGS_HOME");
    let child = Command::new(env!("CARGO_BIN_EXE_higgs"))
        .env("HIGGS_BIND", "127.0.0.1")
        .env("HIGGS_PORT", port.to_string())
        .env("HIGGS_MODEL_DIR", staged.path())
        .env("HIGGS_HOME", home.path())
        // Point the HF hub at a dead local port so `Prepare`/tune's best-effort card
        // fetch FAILS FAST (connection refused) instead of paying the 10s bounded
        // timeout on offline/firewalled CI — `prepare_tiny` runs in many tests. The
        // tiny fixture is pre-staged on disk (never downloaded), so no test here
        // needs a live hub; the download tests set their OWN endpoint and don't use
        // this helper.
        .env("HIGGS_HF_ENDPOINT", "http://127.0.0.1:1")
        .env("RUST_LOG", "warn")
        .spawn()
        .expect("spawn higgs");
    let base = format!("http://127.0.0.1:{port}");
    let client = reqwest::Client::new();
    // Up to ~30s for the listener bind + host-side model scan.
    for _ in 0..150 {
        if let Ok(r) = client.get(format!("{base}/api/higgs/status")).send().await {
            if r.status().is_success() {
                return ServerGuard {
                    child,
                    base,
                    _model_dir: staged,
                    _home: home,
                };
            }
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    panic!("higgs never became ready on {base}");
}

/// Spawn `higgs` bound to 0.0.0.0 with a PRE-SEEDED admin key (a non-loopback
/// bind with zero keys refuses to start, [HG058]) — the keyed-LAN mode. Returns
/// the guard plus the admin bearer token. Requests still connect via 127.0.0.1;
/// LAN behavior is exercised through the `Host` header.
#[allow(clippy::zombie_processes)]
pub async fn spawn_lan_keyed(port: u16, gguf: &Path) -> (ServerGuard, String) {
    let staged = stage_models(gguf, &[TINY_MODEL_ID]);
    let home = TempDir::new().expect("create temp HIGGS_HOME");
    let token = higgs::keys::mint_token([42u8; 16]);
    let mut keys = higgs::keys::ApiKeys::default();
    keys.add(&token, "lan-admin".into(), vec![higgs::keys::Scope::Admin]);
    keys.save(&home.path().join("api_keys.json"))
        .expect("seed keystore");
    let child = Command::new(env!("CARGO_BIN_EXE_higgs"))
        .env("HIGGS_BIND", "0.0.0.0")
        .env("HIGGS_PORT", port.to_string())
        .env("HIGGS_MODEL_DIR", staged.path())
        .env("HIGGS_HOME", home.path())
        .env("HIGGS_HF_ENDPOINT", "http://127.0.0.1:1")
        .env("RUST_LOG", "warn")
        .spawn()
        .expect("spawn higgs");
    let base = format!("http://127.0.0.1:{port}");
    let client = reqwest::Client::new();
    for _ in 0..150 {
        if let Ok(r) = client
            .get(format!("{base}/api/higgs/status"))
            .bearer_auth(&token)
            .send()
            .await
        {
            if r.status().is_success() {
                return (
                    ServerGuard {
                        child,
                        base,
                        _model_dir: staged,
                        _home: home,
                    },
                    token,
                );
            }
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    panic!("keyed-LAN higgs never became ready on {base}");
}

// ── In-process harness (library-first) ───────────────────────────────────────
//
// higgs is now a LIBRARY: control + chat are the in-process `Higgs` crate API and
// the binary is node-only. These helpers drive higgs IN-PROCESS — no spawned
// server, no `/api/higgs/*` HTTP surface. A REAL local llama.cpp worker still runs
// via the `HiggsConfig::worker_exe` DI seam: workers spawn from the real, worker-
// capable `higgs` binary (`CARGO_BIN_EXE_higgs`) rather than the libtest binary
// (which ignores `--higgs-worker`), so `load`/`chat_stream` exercise the full
// engine path. The few tests that must hit the `/v1` HTTP surface use
// `serve_v1_local` to bind the real router on an ephemeral loopback port.

/// Process-global lock serializing every in-process harness test. `HIGGS_HOME`
/// (where `config.json` / `models.json` / `api_keys.json` live) and the model
/// scan root are threaded through PROCESS-GLOBAL env/config and read lazily by the
/// stores, so two harness tests must never run concurrently or they'd clobber each
/// other's isolated home. Each [`higgs_local`] holds this for its whole lifetime.
fn home_lock() -> Arc<tokio::sync::Mutex<()>> {
    static LOCK: std::sync::OnceLock<Arc<tokio::sync::Mutex<()>>> = std::sync::OnceLock::new();
    LOCK.get_or_init(|| Arc::new(tokio::sync::Mutex::new(())))
        .clone()
}

/// An in-process `Higgs` wired to a REAL local llama.cpp worker under an isolated
/// `HIGGS_HOME`. Owns the staging dirs + the process-global home lock for its
/// whole lifetime; drop restores `HIGGS_HOME`. Deref-exposes the `Arc<Higgs>` so a
/// test calls `local.status()`, `local.model_entries()`, `local.load(...)`, etc.
/// directly. Call [`shutdown`](Self::shutdown) (or drop) at the end.
pub struct LocalHiggs {
    higgs: Arc<Higgs>,
    _home: TempDir,
    _scan: TempDir,
    /// Held for the instance's lifetime: serializes `HIGGS_HOME` isolation.
    _lock: tokio::sync::OwnedMutexGuard<()>,
    /// The `HIGGS_HOME` value before this instance overrode it (restored on drop).
    prev_home: Option<std::ffi::OsString>,
    /// The `HIGGS_HF_ENDPOINT` value before this instance overrode it (restored on drop).
    prev_hf: Option<std::ffi::OsString>,
}

impl std::ops::Deref for LocalHiggs {
    type Target = Arc<Higgs>;
    fn deref(&self) -> &Arc<Higgs> {
        &self.higgs
    }
}

impl LocalHiggs {
    /// This instance's isolated `HIGGS_HOME` (config/models/keys store root). Lets a
    /// test induce a persistence failure (chmod `models.json`) or read the keystore.
    pub fn home(&self) -> &std::path::Path {
        self._home.path()
    }

    /// A clone of the facade handle — for `serve_v1_local`, which needs an owned Arc.
    pub fn handle(&self) -> Arc<Higgs> {
        self.higgs.clone()
    }

    /// Absolute path of the staged GGUF for `id` (the `stage_models` layout). Lets a
    /// staleness test mutate the file so its `file_sig` changes.
    pub fn staged_gguf(&self, id: &str) -> PathBuf {
        self._scan.path().join(id).join("stories260K.gguf")
    }

    /// Graceful teardown: drain every resident worker before the runtime unwinds.
    /// Prefer this whenever a worker was loaded so the child process is reaped
    /// deterministically (drop can only best-effort via the node's `on_stop`).
    pub async fn shutdown(self) {
        self.higgs.stop().await;
    }
}

impl Drop for LocalHiggs {
    fn drop(&mut self) {
        // Restore `HIGGS_HOME` + `HIGGS_HF_ENDPOINT` under the STILL-HELD `_lock` so
        // the next test starts clean. SAFETY: the held lock guarantees no other
        // harness thread reads/writes the process env concurrently.
        unsafe {
            match &self.prev_home {
                Some(v) => std::env::set_var("HIGGS_HOME", v),
                None => std::env::remove_var("HIGGS_HOME"),
            }
            match &self.prev_hf {
                Some(v) => std::env::set_var("HIGGS_HF_ENDPOINT", v),
                None => std::env::remove_var("HIGGS_HF_ENDPOINT"),
            }
        }
    }
}

/// Build an in-process `Higgs` with the tiny model staged under EACH id in
/// `models`, an isolated `HIGGS_HOME`, and the `worker_exe` seam pointed at the
/// real `higgs` binary — so `load` / `chat_stream` run REAL llama.cpp in-process.
/// Control tests call the facade directly (`status`/`model_entries`/`mint_key`/…);
/// real-chat tests call `load`+`chat_stream` (or drive `/v1` via `serve_v1_local`).
/// Returns `None` (the test SKIPs) when no tiny GGUF is present.
pub async fn higgs_local(models: &[&str]) -> Option<LocalHiggs> {
    let gguf = tiny_gguf_path()?;
    // Serialize: `HIGGS_HOME` + the staged scan root are process-global and read
    // lazily, so concurrent harness tests must not overlap. Held for the lifetime.
    let lock = home_lock().lock_owned().await;
    let home = TempDir::new().expect("create temp HIGGS_HOME");
    let scan = stage_models(&gguf, models);
    let prev_home = std::env::var_os("HIGGS_HOME");
    let prev_hf = std::env::var_os("HIGGS_HF_ENDPOINT");
    // SAFETY: serialized by the held `lock`; restored on drop.
    unsafe {
        std::env::set_var("HIGGS_HOME", home.path());
        // Point the HF hub at a dead loopback port so `tune`/prepare's best-effort
        // card fetch FAILS FAST (connection refused) instead of hitting the network
        // — the tiny fixture is pre-staged on disk and never downloaded.
        std::env::set_var("HIGGS_HF_ENDPOINT", "http://127.0.0.1:1");
    }

    let config = HiggsConfig {
        lmstudio_dirs: vec![scan.path().to_path_buf()],
        hf_dirs: vec![],
        ollama_dirs: vec![],
        default_load: HiggsConfig::default().default_load,
        // The DI seam: workers spawn from the REAL higgs binary, so an in-process
        // test (whose `current_exe` is libtest) still runs a genuine llama.cpp worker.
        worker_exe: Some(env!("CARGO_BIN_EXE_higgs").into()),
    };
    let higgs = Arc::new(Higgs::new(config));
    higgs.start().await.expect("higgs start");
    Some(LocalHiggs {
        higgs,
        _home: home,
        _scan: scan,
        _lock: lock,
        prev_home,
        prev_hf,
    })
}

/// A running `serve_v1` HTTP server for the FEW tests that must exercise the real
/// `/v1` surface (chat/models over HTTP, auth 401, host guard). Fires the graceful
/// shutdown on drop and (via [`shutdown`](Self::shutdown)) joins the server task —
/// which runs `Higgs::stop`, draining the worker. NEVER leave an SSE stream open
/// across this teardown (it hangs graceful shutdown — see CLAUDE.md).
pub struct ServeGuard {
    shutdown: Option<tokio::sync::oneshot::Sender<()>>,
    join: Option<tokio::task::JoinHandle<std::io::Result<()>>>,
}

impl ServeGuard {
    /// Graceful shutdown: signal `serve_v1` to drain (stopping the worker) and await
    /// the server task. Prefer this over drop whenever the test can await.
    pub async fn shutdown(mut self) {
        if let Some(tx) = self.shutdown.take() {
            let _ = tx.send(());
        }
        if let Some(join) = self.join.take() {
            let _ = join.await;
        }
    }
}

impl Drop for ServeGuard {
    fn drop(&mut self) {
        if let Some(tx) = self.shutdown.take() {
            let _ = tx.send(());
        }
        // Can't await in drop; abort as a fallback so the task never lingers.
        if let Some(join) = self.join.take() {
            join.abort();
        }
    }
}

/// Serve `higgs` on an ephemeral loopback port via the real [`higgs::serve::serve_v1`]
/// router, polling `GET /health` until it answers 200. Returns the base URL
/// (`http://127.0.0.1:<port>`) and a [`ServeGuard`] that shuts the server down on
/// drop. For tests that must hit the `/v1` HTTP surface itself.
pub async fn serve_v1_local(higgs: Arc<Higgs>) -> (String, ServeGuard) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral loopback");
    let port = listener.local_addr().expect("local_addr").port();
    let (tx, rx) = tokio::sync::oneshot::channel::<()>();
    let join = tokio::spawn(higgs::serve::serve_v1(higgs, listener, async move {
        let _ = rx.await;
    }));
    let base = format!("http://127.0.0.1:{port}");
    let client = reqwest::Client::new();
    for _ in 0..200 {
        if let Ok(r) = client.get(format!("{base}/health")).send().await {
            if r.status().is_success() {
                return (
                    base,
                    ServeGuard {
                        shutdown: Some(tx),
                        join: Some(join),
                    },
                );
            }
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("serve_v1 never became ready on {base}");
}
