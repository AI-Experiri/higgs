//! Standalone-server run flow — the testable core of the `higgs` binary.
//!
//! The `higgs` binary's `main()` is a thin wrapper: it detects the
//! `--higgs-worker` re-exec role (which must happen BEFORE anything touches
//! stdout/tracing), installs the tracing subscriber, parses the bind/port env,
//! then hands off to [`run_standalone`]. Everything below the tracing install —
//! constructing [`Higgs`], starting it, binding the listener, and serving with
//! graceful shutdown — lives here so it can be driven in-process by a unit test
//! (the binary's serve path is otherwise only exercised as a spawned subprocess,
//! which coverage instrumentation cannot see).
//!
//! The caller owns the [`LogBus`] and the tracing layer: the bus is created
//! before the subscriber so the binary's [`HiggsLogLayer`](crate::HiggsLogLayer)
//! and the [`Higgs`] facade share one Developer-Log history. It is passed in via
//! [`StandaloneConfig::log_bus`].

use std::sync::Arc;

use tokio::signal;

use crate::api::{Higgs, HiggsConfig};
use crate::log_bus::LogBus;

/// Inputs for [`run_standalone`]: the bind address, the higgs runtime config,
/// and the shared [`LogBus`] the caller already installed its tracing layer on.
pub struct StandaloneConfig {
    /// Bind host (e.g. `127.0.0.1`, `0.0.0.0`). A non-loopback value triggers a
    /// security warning (no-auth surface beyond localhost).
    pub bind: String,
    /// Listen port (e.g. `11434`).
    pub port: u16,
    /// Runtime config for the [`Higgs`] facade (scan dirs, default load params).
    pub higgs: HiggsConfig,
    /// Shared Developer-Log bus the caller's [`HiggsLogLayer`] writes to. Passed
    /// into [`Higgs::with_log_bus`] so serve-layer events and worker stderr land
    /// in the same history.
    ///
    /// [`HiggsLogLayer`]: crate::HiggsLogLayer
    pub log_bus: Arc<LogBus>,
}

/// Construct + start [`Higgs`], bind the listener, and serve until `shutdown`
/// resolves — the standalone server's whole run flow below the tracing install.
///
/// `shutdown` is injected (not hardcoded to [`shutdown_signal`]) so a test can
/// drive the full bind→serve→drain cycle in-process with its own trigger. The
/// binary passes [`shutdown_signal`] (SIGTERM/Ctrl-C).
///
/// Returns `Err` if the worker fails to start, the address fails to bind, or
/// `serve` returns an I/O error. The error is rendered by the caller; this fn
/// does not call `std::process::exit` so it stays testable.
pub async fn run_standalone(
    config: StandaloneConfig,
    shutdown: impl std::future::Future<Output = ()> + Send + 'static,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let StandaloneConfig {
        bind,
        port,
        higgs: higgs_config,
        log_bus,
    } = config;
    let addr = format!("{bind}:{port}");

    // SECURITY WARNING on a non-loopback bind (vllm's startup warning). higgs
    // has NO auth — its control surface (`/api/higgs/*`) and OpenAI `/v1` are
    // open to anyone who can reach the port. The DNS-rebinding Host guard + CORS
    // only protect *browser* clients; a non-loopback bind exposes the surface to
    // any non-browser client on the network. The embedded path always binds
    // loopback — this only fires for the standalone bin.
    if !is_loopback_bind(&bind) {
        tracing::warn!(
            %bind,
            "SECURITY WARNING: higgs is binding to a NON-LOOPBACK address. The \
             no-auth control + /v1 surface is exposed beyond localhost; CORS and \
             the Host guard do not protect non-browser clients. Bind 127.0.0.1 \
             unless you intend a LAN-reachable server."
        );
    }

    let higgs = Arc::new(Higgs::with_log_bus(higgs_config, log_bus.clone()));
    // Load API keys (P5). A MISSING store leaves the surface open by design (embedded host);
    // but a present-yet-unreadable/malformed store, or an unusable home dir, FAILS CLOSED —
    // we abort startup rather than silently serve unauthenticated, since the file's presence
    // is exactly what enables protection.
    let keys_path = crate::keys::keys_path()?;
    let keys = crate::keys::ApiKeys::load(&keys_path)?;
    let n = keys.iter().count();
    if n > 0 {
        tracing::info!(keys = n, "higgs: API-key auth ENABLED");
    }
    higgs.set_api_keys(Arc::new(keys));

    // Hub mode (HIGGS_HUB=1, P3): bind the iroh endpoint, install the fleet, and accept node
    // dials. The facade owns the hub Arc (`set_hub`) for the server's lifetime, so the endpoint
    // stays bound; the kill switch (`POST /api/higgs/hub/{disable,enable}`) can later tear it
    // down + rebind it against the same fleet (routes survive).
    if std::env::var("HIGGS_HUB").is_ok_and(|v| v == "1" || v == "true") {
        match crate::node::hub::start_hub(log_bus.clone(), None).await {
            Ok(hub) => {
                let hub = Arc::new(hub);
                tracing::info!(
                    hub_id = hub.hub_id(),
                    "higgs: HUB mode — accepting node dials"
                );
                higgs.set_fleet(hub.fleet.clone());
                higgs.set_hub(hub);
            }
            Err(e) => return Err(Box::new(e)),
        }
    }

    higgs.start().await?;

    let listener = tokio::net::TcpListener::bind(&addr).await?;

    tracing::info!(%addr, "higgs listening — /v1 (OpenAI) + /api/higgs (control)");
    // Graceful shutdown on SIGTERM/Ctrl-C: drain requests, then stop the worker.
    // The crate owns the shutdown semantics (serve_with_shutdown).
    crate::serve::serve_with_shutdown(higgs, listener, shutdown).await?;
    Ok(())
}

/// Whether `bind` names a loopback address (no security warning needed).
/// Anything else (`0.0.0.0`, a LAN IP, a public IP, `::`) is non-loopback.
pub fn is_loopback_bind(bind: &str) -> bool {
    bind == "127.0.0.1"
        || bind == "localhost"
        || bind == "::1"
        || bind == "[::1]"
        || bind.starts_with("127.")
}

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
mod tests {
    use super::*;

    #[test]
    fn loopback_bind_detection() {
        for b in ["127.0.0.1", "localhost", "::1", "[::1]", "127.0.0.5"] {
            assert!(is_loopback_bind(b), "{b} is loopback");
        }
        for b in ["0.0.0.0", "10.0.0.5", "192.168.1.10", "::", "203.0.113.1"] {
            assert!(!is_loopback_bind(b), "{b} is non-loopback");
        }
    }

    /// Drive the full standalone run flow in-process — construct + start Higgs,
    /// bind an ephemeral port, serve, hit `/api/higgs/status`, then trigger the
    /// injected shutdown and confirm `run_standalone` returns `Ok`. No
    /// subprocess, so coverage instrumentation sees the serve path the binary's
    /// `main()` otherwise only runs as a spawned (un-instrumented) child.
    ///
    /// A non-loopback bind (`0.0.0.0`) is used so the security-warning branch is
    /// also exercised; binding `0.0.0.0:0` is fine for a localhost test client.
    // TEST_ENV_LOCK (a std Mutex) is intentionally held across awaits to serialize the whole
    // env-overridden run; this is a #[tokio::test] (current-thread) and the only other holder
    // is a sync test, so there's no deadlock/Send hazard.
    #[allow(clippy::await_holding_lock)]
    #[tokio::test]
    async fn run_standalone_serves_and_shuts_down() {
        // Isolate HIGGS_HOME so `run_standalone` never loads the dev/CI machine's real
        // `~/.higgs/api_keys.json` — a present keystore would turn auth ON and 401 the
        // no-token `/api/higgs/status` poll below. Serialize with other env-mutating tests
        // (shared lock) and RESTORE the prior value at the end so nothing leaks. The lock +
        // `home` TempDir are held for the whole test.
        let _env = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let home = tempfile::tempdir().unwrap();
        let prev_home = std::env::var_os("HIGGS_HOME");
        // SAFETY: serialized by TEST_ENV_LOCK; restored before the lock releases.
        unsafe { std::env::set_var("HIGGS_HOME", home.path()) };
        // Restore on every exit path (drops in reverse order: this runs, then `home`, then lock).
        struct RestoreHome(Option<std::ffi::OsString>);
        impl Drop for RestoreHome {
            fn drop(&mut self) {
                // SAFETY: still under TEST_ENV_LOCK.
                unsafe {
                    match self.0.take() {
                        Some(v) => std::env::set_var("HIGGS_HOME", v),
                        None => std::env::remove_var("HIGGS_HOME"),
                    }
                }
            }
        }
        let _restore = RestoreHome(prev_home);

        // Reserve an ephemeral port, read it back, then drop the listener so
        // run_standalone can bind the same addr. (run_standalone binds itself.)
        let probe = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = probe.local_addr().unwrap().port();
        drop(probe);

        let (tx, rx) = tokio::sync::oneshot::channel::<()>();
        let config = StandaloneConfig {
            // 0.0.0.0 exercises the non-loopback security-warning branch; the
            // client still connects via 127.0.0.1 on the same port.
            bind: "0.0.0.0".to_string(),
            port,
            higgs: HiggsConfig::default(),
            log_bus: Arc::new(LogBus::new()),
        };
        let server = tokio::spawn(async move {
            run_standalone(config, async {
                let _ = rx.await;
            })
            .await
        });

        // Poll the status endpoint until the server is accepting (bind races the
        // client). status() needs no worker, so it answers immediately.
        let url = format!("http://127.0.0.1:{port}/api/higgs/status");
        let client = reqwest::Client::new();
        let mut last_status = None;
        for _ in 0..50 {
            match client.get(&url).header("host", "127.0.0.1").send().await {
                Ok(resp) => {
                    last_status = Some(resp.status());
                    break;
                }
                Err(_) => tokio::time::sleep(std::time::Duration::from_millis(20)).await,
            }
        }
        assert_eq!(
            last_status,
            Some(reqwest::StatusCode::OK),
            "/api/higgs/status should answer 200 while serving"
        );

        // Trigger graceful shutdown and confirm the run flow returns Ok.
        tx.send(()).unwrap();
        let result = tokio::time::timeout(std::time::Duration::from_secs(5), server)
            .await
            .expect("server task should finish after shutdown")
            .expect("server task should not panic");
        assert!(result.is_ok(), "run_standalone returned {result:?}");
    }
}
