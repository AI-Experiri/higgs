//! Shared harness for higgs black-box integration tests.
//!
//! Spawns the real `higgs` binary on a localhost port, waits for the
//! worker to come up, and kills the process (and its worker) on drop. Tests
//! drive the server purely over HTTP — exactly how an external client would.

use std::process::{Child, Command};
use std::time::Duration;

/// A running `higgs` child process. Killed (with its worker) on drop.
pub struct ServerGuard {
    child: Child,
    /// Base URL, e.g. `http://127.0.0.1:11500`.
    pub base: String,
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

/// Spawn `higgs` on `127.0.0.1:{port}` and wait until `/api/higgs/status`
/// answers (worker spawned + listener bound). Panics if it never comes up.
// The spawned child is reaped by `ServerGuard::drop` (kill + wait), so the
// zombie-process lint is a false positive — clippy can't see the Drop impl.
#[allow(clippy::zombie_processes)]
pub async fn spawn(port: u16) -> ServerGuard {
    let child = Command::new(env!("CARGO_BIN_EXE_higgs"))
        .env("HIGGS_BIND", "127.0.0.1")
        .env("HIGGS_PORT", port.to_string())
        .env("RUST_LOG", "warn")
        .spawn()
        .expect("spawn higgs");
    let base = format!("http://127.0.0.1:{port}");
    let client = reqwest::Client::new();
    // Up to ~30s for the worker re-exec + model scan.
    for _ in 0..150 {
        if let Ok(r) = client.get(format!("{base}/api/higgs/status")).send().await {
            if r.status().is_success() {
                return ServerGuard { child, base };
            }
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    panic!("higgs never became ready on {base}");
}

/// Find an on-disk Nemotron model id in a `/api/higgs/models` body, if present.
/// Tests skip (return early) when no such model is installed.
pub fn nemotron_id(models: &serde_json::Value) -> Option<String> {
    models["models"].as_array()?.iter().find_map(|m| {
        let id = m["id"].as_str()?;
        id.to_lowercase()
            .contains("nemotron")
            .then(|| id.to_string())
    })
}
