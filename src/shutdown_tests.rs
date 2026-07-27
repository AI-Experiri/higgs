use super::*;

/// `shutdown_signal()` installs the Ctrl-C + SIGTERM handlers and parks on a
/// `select!` until one fires. We race it against a short timeout: it must NOT
/// resolve on its own (no signal sent), proving it actually waits — and driving
/// the handler-install + select-setup lines. (Delivering a real SIGTERM to the
/// shared test binary is unsafe, so the terminal recv branch is left to the
/// integration harness's real subprocess.)
#[tokio::test]
async fn shutdown_signal_parks_until_signalled() {
    let timed_out = tokio::time::timeout(std::time::Duration::from_millis(150), shutdown_signal())
        .await
        .is_err();
    assert!(
        timed_out,
        "shutdown_signal must keep waiting when no signal is delivered"
    );
}
