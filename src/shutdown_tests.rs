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

/// `shutdown_listener()` registers its SIGTERM/SIGINT handlers SYNCHRONOUSLY in
/// the function body (its whole reason to exist vs the lazy `shutdown_signal`:
/// the node daemon must not miss a signal delivered before the first poll).
/// Constructing it must succeed inside a runtime, and the returned future must
/// PARK — not resolve — when no signal is delivered. (Actually delivering a
/// signal would race the sibling park test above, whose assertion is that
/// `shutdown_signal` does NOT resolve inside its window, so the terminal recv
/// branches stay with the integration harness's real SIGTERM'd subprocess.)
#[tokio::test]
async fn shutdown_listener_installs_handlers_eagerly_and_parks() {
    // Handler installation happens HERE (before any poll) — a panic in the
    // `expect("install …")` constructors would fail the test at this line.
    let fut = shutdown_listener();
    let timed_out = tokio::time::timeout(std::time::Duration::from_millis(150), fut)
        .await
        .is_err();
    assert!(
        timed_out,
        "shutdown_listener must keep waiting when no signal is delivered"
    );
}
