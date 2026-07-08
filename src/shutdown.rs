//! The process-termination future shared by every long-running higgs entrypoint.
//!
//! `serve_v1` (the surviving HTTP surface) and the node CLI daemons
//! (`node/cli.rs`) all take a graceful-shutdown future; [`shutdown_signal`] is the
//! standard one — it resolves on SIGTERM (the supervisor/`kill` signal) or Ctrl-C.
//! Kept in its own tiny module (re-exported as `crate::shutdown_signal`) so it
//! outlives the now-deleted standalone-server module.

use tokio::signal;

/// Resolve when the process is asked to terminate — SIGTERM (the standard
/// supervisor/`kill` signal) or Ctrl-C. Graceful shutdown lets in-flight
/// requests finish and runs normal at-exit handlers (which, under coverage
/// instrumentation, flush this process's profile).
pub async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("install SIGTERM handler")
            .recv()
            .await;
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}

#[cfg(test)]
#[path = "shutdown_tests.rs"]
mod tests;
