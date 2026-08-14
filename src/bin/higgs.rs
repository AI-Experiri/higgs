//! `higgs` — the NODE binary for the higgs runtime.
//!
//! higgs is library-first: control + chat run in-process via the `Higgs` facade,
//! and the only HTTP surface is OpenAI `/v1` (chat + models), served by
//! [`higgs::serve::serve_v1`]. An embedder (jigglebot) drives everything through
//! the crate API; there is NO standalone HTTP-control server anymore.
//!
//! So this binary is NODE-ONLY. It handles:
//!   - the `--higgs-worker` re-exec role (the crash-isolated llama.cpp worker),
//!     detected before anything touches stdout because `worker_main()` owns
//!     stdin/stdout for NDJSON JSON-RPC;
//!   - `--version` / `-V`;
//!   - the fleet subcommands: `--node` (persistent worker daemon), `node`
//!     (one-shot node ops), `link` (hub-side pairing), and `keys` (API-key mgmt).
//!
//! A bare invocation with no subcommand prints a usage note and exits non-zero —
//! there is no default server to fall back to.
//!
//! Configuration (env):
//!   HIGGS_MODEL_DIR   extra model scan root in LM-Studio layout
//!                     (`<dir>/{org}/{model}/*.gguf`), honored by `--node`.
//!   HIGGS_VERBOSE     `1` keeps the full llama.cpp per-load dump (node worker).
//!   RUST_LOG          tracing filter (default `info`)

use std::sync::Arc;

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

    let args: Vec<String> = std::env::args().skip(1).collect();
    // `higgs --version` / `-V`: report the crate version. The single source of
    // truth is Cargo.toml `[package] version`, surfaced here via CARGO_PKG_VERSION
    // so the CLI and the release tag/artifacts always agree. First-arg only, so a
    // `--version` buried in a subcommand's own args is left for that subcommand.
    if matches!(args.first().map(String::as_str), Some("--version" | "-V")) {
        println!("higgs {}", env!("CARGO_PKG_VERSION"));
        return;
    }

    // Self-update boot-guard ROLLBACK check, run for the `--node` daemon path BEFORE any
    // logging/tracing setup — the earliest stable point in a freshly-flipped binary — so a
    // trial that already SPENT its budget rolls back here before this start even tries again.
    // The paired boot-ATTEMPT record is taken at the top of `run_node_daemon_body`'s async
    // block (the first point with a tokio context for the SIGTERM handler), BEFORE the risky
    // bind/runtime/serve init, so an update that dies during that init accrues (see the
    // boot-guard comment there). Exits so the service manager re-execs the now-current (old)
    // binary on a rollback; a no-op with no pending self-update trial.
    if args.first().map(String::as_str) == Some("--node") && higgs::node::cli::node_boot_preflight()
    {
        return;
    }

    // Single home for Developer-Log lines (worker stderr + serve-layer events),
    // plus the terminal `fmt` layer. Installed ABOVE the subcommand dispatch so
    // the node/link/keys subcommands get higgs's own logging (previously the
    // subscriber sat after the dispatch and node subcommands ran with none).
    let log_bus = Arc::new(higgs::LogBus::new());
    // Per-layer filters so the higgs log layer can admit higgs DEBUG (verbose
    // mode) without flooding fmt; info-level filter applied to fmt individually.
    // Default filter demotes the iroh/QUIC transport's WARN firehose (relay retries, IPv6
    // no-route sends, multipath close chatter) to ERROR — operationally meaningless noise
    // that buried the useful lines during the first real-hardware install. RUST_LOG still
    // overrides everything (set RUST_LOG=info,iroh=warn to get the firehose back).
    let env = || {
        tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
            tracing_subscriber::EnvFilter::new(
                "info,iroh=error,noq_proto=error,noq_udp=error,netwatch=error,\
                 portmapper=error,hickory_resolver=error",
            )
        })
    };
    // Register the SAME bus as the process-global before it moves into the layer:
    // the node daemon's NodeRuntime picks it up so its own tracing (the Serve ring)
    // is what M_NODE_LOGS serves to the hub.
    higgs::LogBus::install_global(log_bus.clone());
    tracing_subscriber::registry()
        .with(higgs::HiggsLogLayer::new(log_bus).with_filter(higgs::log_filter()))
        .with(tracing_subscriber::fmt::layer().with_filter(env()))
        .init();

    // Fleet subcommands (iroh). Each runs to completion and exits.
    let remote = match args.first().map(String::as_str) {
        Some("--node") => Some(higgs::node::cli::run_node_daemon(&args[1..])),
        Some("link") => Some(higgs::node::cli::run_link(&args[1..])),
        Some("node") => Some(higgs::node::cli::run_node(&args[1..])),
        Some("model") => Some(higgs::catalog::cli::run_model(&args[1..])),
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

    // No subcommand: higgs is node-only — there is no standalone HTTP server to
    // start. Point the operator at the real entrypoints and exit non-zero.
    eprintln!(
        "higgs is node-only — it no longer runs a standalone HTTP server.\n\
         (control + chat run in-process via the `higgs` crate; serve OpenAI /v1 with `higgs::serve::serve_v1`.)\n\
         \n\
         usage:\n  \
         higgs --node [<ticket> [token]]   join a hub as a worker node (persistent daemon)\n  \
         higgs node <connect|leave> …       one-shot node-side ops against a hub\n  \
         higgs node install-service          node service — user-space, login-bound by default; --system = always-on\n  \
         higgs link <pair|status>           hub-side fleet pairing\n  \
         higgs model <search|show|download>  search the Hugging Face catalog / pull a GGUF\n  \
         higgs keys <…>                      manage API keys\n  \
         higgs --version                     print the version"
    );
    std::process::exit(2);
}
