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

/// Like [`shutdown_signal`], but the OS signal handlers are installed SYNCHRONOUSLY at
/// CALL time — the `signal(...)` constructors run in this function body, before the
/// returned future is ever polled. `shutdown_signal` is an `async fn`, so its handler
/// registration does not happen until the future is first polled; a caller that must not
/// miss a signal delivered in the window BEFORE that first poll (the node daemon, which
/// records a self-update boot attempt just before it starts awaiting the shutdown future
/// — a SIGTERM in between would otherwise terminate by default and falsely spend the
/// rollback budget) uses this instead. A signal delivered after construction but before
/// `.recv()` is buffered by the `Signal` stream, so it is not lost.
#[cfg(unix)]
pub fn shutdown_listener() -> impl std::future::Future<Output = ()> {
    use tokio::signal::unix::{signal, SignalKind};
    // Constructing the streams registers the kernel handlers NOW (synchronously).
    let mut term = signal(SignalKind::terminate()).expect("install SIGTERM handler");
    let mut intr = signal(SignalKind::interrupt()).expect("install SIGINT handler");
    async move {
        tokio::select! {
            _ = term.recv() => {},
            _ = intr.recv() => {},
        }
    }
}

#[cfg(not(unix))]
pub fn shutdown_listener() -> impl std::future::Future<Output = ()> {
    shutdown_signal()
}

#[cfg(test)]
#[path = "shutdown_tests.rs"]
mod tests;
