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
//! This `main()` is a THIN wrapper: it handles the `--higgs-worker` re-exec
//! role, installs the tracing subscriber, parses the bind/port env, then hands
//! off to [`higgs::run_standalone`], which owns the construct→start→bind→serve
//! flow (and is unit-tested in-process — see `src/standalone.rs`).
//!
//! Configuration (env):
//!   HIGGS_BIND        bind address  (default `127.0.0.1` — localhost only)
//!   HIGGS_PORT        listen port   (default `11434`)
//!   HIGGS_MODEL_DIR   extra model scan root in LM-Studio layout
//!                     (`<dir>/{org}/{model}/*.gguf`), appended to the default
//!                     scan dirs. Lets an operator (or CI) point higgs at an
//!                     arbitrary model directory without editing config.
//!   RUST_LOG          tracing filter (default `info`)
//!
//! ```text
//! higgs                       # 127.0.0.1:11434
//! HIGGS_BIND=0.0.0.0 HIGGS_PORT=1234 higgs   # LAN-reachable on :1234
//! ```

use std::sync::Arc;

use higgs::{run_standalone, shutdown_signal, HiggsConfig, StandaloneConfig};
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

    // Remote-worker roles/subcommands (iroh). Each runs to completion and exits,
    // before the default server path below.
    let args: Vec<String> = std::env::args().skip(1).collect();
    let remote = match args.first().map(String::as_str) {
        Some("--node") => Some(higgs::node::cli::run_node_daemon(&args[1..])),
        Some("link") => Some(higgs::node::cli::run_link(&args[1..])),
        Some("node") => Some(higgs::node::cli::run_node(&args[1..])),
        Some("keys") => Some(higgs::keys::run_keys(&args[1..])),
        _ => None,
    };
    if let Some(result) = remote {
        if let Err(e) = result {
            eprintln!("higgs: {e}");
            std::process::exit(1);
        }
        return;
    }

    // Single home for Developer-Log lines: worker stderr + serve-layer events.
    // Created before the subscriber so the HiggsLogLayer and the Higgs facade
    // (built inside run_standalone) can share it.
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

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("build tokio runtime");

    // Optional extra scan root (operator/CI override). Appended as an LM-Studio
    // root so a single `<dir>/{org}/{model}/*.gguf` tree is discovered without
    // editing the default config dirs.
    let mut higgs_config = HiggsConfig::default();
    if let Ok(dir) = std::env::var("HIGGS_MODEL_DIR") {
        if !dir.is_empty() {
            tracing::info!(dir = %dir, "higgs: adding HIGGS_MODEL_DIR to LM-Studio scan roots");
            higgs_config
                .lmstudio_dirs
                .push(std::path::PathBuf::from(dir));
        }
    }

    rt.block_on(async move {
        let config = StandaloneConfig {
            bind,
            port,
            higgs: higgs_config,
            log_bus,
        };
        // Graceful shutdown on SIGTERM/Ctrl-C: drain requests, then stop the
        // worker. run_standalone owns construct→start→bind→serve; we render any
        // failure and exit non-zero (it never calls process::exit itself).
        if let Err(e) = run_standalone(config, shutdown_signal()).await {
            tracing::error!(error = %e, "higgs failed");
            std::process::exit(1);
        }
    });
}
