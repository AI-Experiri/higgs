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
    /// Listen port (default [`crate::DEFAULT_PORT`], 31415).
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

    let higgs = Arc::new(Higgs::with_log_bus(higgs_config, log_bus.clone()));
    // Load API keys (P5) BEFORE the bind-exposure check below needs them. A MISSING store
    // leaves the surface open by design (embedded host); but a present-yet-unreadable/
    // malformed store, or an unusable home dir, FAILS CLOSED — we abort startup rather than
    // silently serve unauthenticated, since the file's presence is exactly what enables
    // protection.
    let keys_path = crate::keys::keys_path()?;
    let keys = crate::keys::ApiKeys::load(&keys_path)?;
    let n = keys.iter().count();
    if n > 0 {
        tracing::info!(keys = n, "higgs: API-key auth ENABLED");
    }
    higgs.set_api_keys(Arc::new(keys));

    // Non-loopback bind exposure (G4). The Host guard + CORS only protect *browser*
    // clients; a non-loopback bind exposes the surface to any non-browser client on
    // the network. With ZERO keys that surface would be wide open — REFUSE to start
    // ([HG058], fail closed; lmstudio/ollama never gate this — higgs does). With keys
    // present the bind is intentional: auth gates every route, so only warn.
    //
    // Decide on the RESOLVED bind, not the raw string: a hostname / `/etc/hosts`
    // alias to `127.0.0.1` is loopback-only and must not be refused (codex r14).
    // Only refuse when the bind DEFINITELY resolves to a non-loopback address;
    // if resolution is inconclusive, `serve_with_shutdown` still enforces
    // [HG058] on the REAL bound address (`listener.local_addr()`) as the
    // authoritative backstop.
    if bind_resolves_non_loopback(&bind, port) {
        let keys = higgs.api_keys();
        if keys.is_empty() {
            return Err(Box::new(
                crate::diagnostic::HiggsError::LanBindWithoutKeys { bind },
            ));
        }
        // Keys present but NONE Admin-capable: auth is ON, yet every Admin-scoped
        // control/key route (mint/revoke) is rejected — the operator can't manage
        // keys over the API on this running LAN server (mint itself needs Admin).
        // Fail closed ([HG069]); recovery is to add an Admin key out-of-band and
        // restart. `higgs keys add <label>` defaults to `chat,models`, so a LAN
        // bootstrap easily lands here — refuse rather than serve a locked-out surface.
        if !keys
            .iter()
            .any(|k| k.scopes.contains(&crate::keys::Scope::Admin))
        {
            return Err(Box::new(
                crate::diagnostic::HiggsError::LanBindWithoutAdminKey { bind },
            ));
        }
        tracing::warn!(
            %bind,
            "higgs is binding beyond localhost; API-key auth is ENABLED and gates \
             every /v1 + /api/higgs route (health stays open)."
        );
    }

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

    // A bind failure is almost always "port already in use" (another higgs /
    // another app) or an address this machine doesn't own — say so, with the
    // exact knobs to fix it, instead of a bare os error. This is the message an
    // operator sees when standing higgs up on a fresh/remote machine.
    let listener = tokio::net::TcpListener::bind(&addr).await.map_err(|e| {
        format!(
            "cannot listen on {addr}: {e}
               → the port may already be in use — pick another with HIGGS_PORT=<port>
               → to serve other machines on the network, bind with HIGGS_BIND=0.0.0.0
               → current defaults: HIGGS_BIND=127.0.0.1 HIGGS_PORT={default}",
            default = crate::DEFAULT_PORT
        )
    })?;

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

/// Whether `bind:port` DEFINITELY resolves to a non-loopback address (the G4
/// [HG058] refusal condition). Resolves the actual socket addresses — so a
/// hostname / hosts-alias to `127.0.0.1` is correctly treated as loopback — and
/// returns `true` only when resolution SUCCEEDS and yields at least one
/// non-loopback address. A resolution failure returns `false` (inconclusive →
/// don't refuse here; `serve_with_shutdown` checks the real bound address).
fn bind_resolves_non_loopback(bind: &str, port: u16) -> bool {
    use std::net::ToSocketAddrs;
    match (bind, port).to_socket_addrs() {
        Ok(addrs) => {
            let mut any = false;
            for a in addrs {
                any = true;
                if !a.ip().is_loopback() {
                    return true; // definitely exposed
                }
            }
            // Resolved to only loopback addresses (or, if `any` is false, to
            // none) — not known-exposed.
            let _ = any;
            false
        }
        Err(_) => false, // inconclusive → defer to serve_with_shutdown
    }
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
#[path = "standalone_tests.rs"]
mod tests;
