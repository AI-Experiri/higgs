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
use std::time::Duration;

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
        let _ = self.child.wait();
    }
}

/// Stage `gguf` into a temp LM-Studio layout (`<tmp>/higgs-test/stories260k/`)
/// so the scanner discovers it under [`TINY_MODEL_ID`]. Returns the temp dir
/// (the scan root) — keep it alive for the server's lifetime. The GGUF is
/// copied (not symlinked) so it stays inside the scan root and passes higgs's
/// path-within-roots guard.
fn stage_tiny_model(gguf: &Path) -> TempDir {
    let dir = TempDir::new().expect("create staging dir");
    let model_dir = dir.path().join("higgs-test").join("stories260k");
    std::fs::create_dir_all(&model_dir).expect("create model dir");
    std::fs::copy(gguf, model_dir.join("stories260K.gguf")).expect("copy tiny gguf");
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
    let staged = stage_tiny_model(gguf);
    let child = Command::new(env!("CARGO_BIN_EXE_higgs"))
        .env("HIGGS_BIND", "127.0.0.1")
        .env("HIGGS_PORT", port.to_string())
        .env("HIGGS_MODEL_DIR", staged.path())
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
                };
            }
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    panic!("higgs never became ready on {base}");
}
