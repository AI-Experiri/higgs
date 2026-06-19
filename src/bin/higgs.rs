//! `higgs` — the standalone higgs runtime as its own process.
//!
//! higgs is a self-contained local-model server (OpenAI `/v1/*` + its own
//! `/api/higgs/*` control surface). It owns its whole HTTP surface on its own
//! port; other apps (jigglebot included) consume it purely as an
//! OpenAI-compatible endpoint via HTTP — nothing imports higgs's internals.
//!
//! Crash isolation: the worker supervisor re-executes THIS binary with
//! `--higgs-worker` (Chromium model). That flag is detected before anything
//! touches stdout, because `worker_main()` owns stdin/stdout for NDJSON JSON-RPC.
//!
//! Configuration (env):
//!   HIGGS_BIND   bind address      (default `127.0.0.1` — localhost only)
//!   HIGGS_PORT   listen port       (default `11434`)
//!   RUST_LOG     tracing filter    (default `info`)
//!
//! ```text
//! higgs                       # 127.0.0.1:11434
//! HIGGS_BIND=0.0.0.0 HIGGS_PORT=1234 higgs   # LAN-reachable on :1234
//! ```

use std::sync::Arc;

use higgs::{Higgs, HiggsConfig};
use tokio::signal;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::Layer;

fn main() {
    // Worker role: detect BEFORE tracing/anything writes stdout — the worker
    // speaks NDJSON JSON-RPC over stdio and must own it exclusively.
    if std::env::args().skip(1).any(|a| a == "--higgs-worker") {
        higgs::worker::worker_main();
        return;
    }

    // Single home for Developer-Log lines: worker stderr + serve-layer events.
    // Created before the subscriber so the HiggsLogLayer and the Higgs facade
    // can share it.
    let log_bus = Arc::new(higgs::LogBus::new());
    // Per-layer filters so the higgs log layer can admit higgs DEBUG (verbose
    // mode) without flooding fmt; info-level filter applied to fmt individually.
    let env = || {
        tracing_subscriber::EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"))
    };
    tracing_subscriber::registry()
        .with(higgs::HiggsLogLayer::new(log_bus.clone()).with_filter(higgs::log_filter()))
        .with(tracing_subscriber::fmt::layer().with_filter(env()))
        .init();

    let bind = std::env::var("HIGGS_BIND").unwrap_or_else(|_| "127.0.0.1".to_string());
    // Parse HIGGS_PORT; an unset var is silently the default, but a SET-but-bad
    // value (non-numeric, out of u16 range) is a misconfiguration the operator
    // must see — warn naming the bad value and the fallback before using 11434.
    let port: u16 = match std::env::var("HIGGS_PORT") {
        Ok(raw) => raw.parse().unwrap_or_else(|e| {
            tracing::warn!(
                value = %raw,
                error = %e,
                fallback = 11434,
                "HIGGS_PORT is not a valid port — falling back to 11434"
            );
            11434
        }),
        Err(_) => 11434,
    };
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

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("build tokio runtime");

    rt.block_on(async move {
        let higgs = Arc::new(Higgs::with_log_bus(HiggsConfig::default(), log_bus));
        if let Err(e) = higgs.start().await {
            tracing::error!(error = %e, "higgs failed to start (worker spawn)");
            std::process::exit(1);
        }

        let listener = tokio::net::TcpListener::bind(&addr)
            .await
            .unwrap_or_else(|e| panic!("bind {addr}: {e}"));

        tracing::info!(%addr, "higgs listening — /v1 (OpenAI) + /api/higgs (control)");
        // Graceful shutdown on SIGTERM/Ctrl-C: drain requests, then stop the
        // worker. The crate owns the shutdown semantics (serve_with_shutdown).
        if let Err(e) = higgs::serve::serve_with_shutdown(higgs, listener, shutdown_signal()).await
        {
            tracing::error!(error = %e, "higgs serve failed");
            std::process::exit(1);
        }
    });
}

/// Whether `bind` names a loopback address (no security warning needed).
/// Anything else (`0.0.0.0`, a LAN IP, a public IP, `::`) is non-loopback.
fn is_loopback_bind(bind: &str) -> bool {
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
async fn shutdown_signal() {
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
    use super::is_loopback_bind;

    #[test]
    fn loopback_bind_detection() {
        for b in ["127.0.0.1", "localhost", "::1", "[::1]", "127.0.0.5"] {
            assert!(is_loopback_bind(b), "{b} is loopback");
        }
        for b in ["0.0.0.0", "10.0.0.5", "192.168.1.10", "::", "203.0.113.1"] {
            assert!(!is_loopback_bind(b), "{b} is non-loopback");
        }
    }
}
