//! Standalone OpenAI `/v1` server over the in-process `Higgs` facade.
//!
//! higgs is library-first and its CLI is node-only, so this EXAMPLE is the
//! way to get a bare `/v1` endpoint (chat + models) without an embedder —
//! used by external benchmarking (`AI-Experiri/engine-bench`) and manual
//! curl testing. Serves until Ctrl-C, then shuts the facade down gracefully.
//! Like `smoke.rs`, it hosts the `--higgs-worker` re-exec role itself so
//! real llama.cpp workers spawn from this binary.
//!
//! Run: `cargo run --release --example serve`
//!   env HIGGS_SERVE_ADDR=…   bind address (default `127.0.0.1:8311`)
//!   env HIGGS_MODEL_DIR=…    extra LM-Studio-style scan root
//!   env HIGGS_PARAMS_JSON=…  path to `{"model": "<id>", "params": <LoadParams>}`
//!                            — the model is EPHEMERALLY pre-loaded with exactly
//!                            those params (the store's saved profiles are never
//!                            touched), so a benchmark can pin its own config.
//!                            NOTE: unknown/misspelled param keys are silently
//!                            ignored (serde defaults apply) — generate the file
//!                            from a store profile rather than hand-writing it,
//!                            and verify the EFFECTIVE settings from the llama.cpp
//!                            dump this binary echoes
//!   env RUST_LOG=higgs=info  for load progress
//!
//! Worker stderr (including llama.cpp's effective-config load dump) is echoed
//! to THIS process's stdout via the log bus, so an external harness can verify
//! the settings a load actually ran with.

use std::sync::Arc;

use higgs::{Higgs, HiggsConfig, LoadParams};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer};

fn main() {
    // Worker role FIRST — before anything touches stdout (NDJSON JSON-RPC).
    if std::env::args().skip(1).any(|a| a == "--higgs-worker") {
        higgs::worker::worker_main();
        return;
    }

    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer().with_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("higgs=info,warn")),
        ))
        .init();

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("build tokio runtime");
    rt.block_on(run());
}

async fn run() {
    let mut config = HiggsConfig::default();
    if let Ok(dir) = std::env::var("HIGGS_MODEL_DIR") {
        if !dir.is_empty() {
            config.lmstudio_dirs.push(std::path::PathBuf::from(dir));
        }
    }
    let addr = std::env::var("HIGGS_SERVE_ADDR").unwrap_or_else(|_| "127.0.0.1:8311".into());

    // Route worker stderr (the llama.cpp load dump included) through the log
    // bus and echo it to stdout — the ONLY external window onto the effective
    // load settings. `set_verbose(true)` keeps the full per-load dump.
    let bus = Arc::new(higgs::LogBus::new());
    bus.set_verbose(true);
    let mut log_rx = bus.subscribe();
    tokio::spawn(async move {
        // Same pattern as the crate's own pumps: a broadcast `Lagged` (this
        // consumer fell >256 lines behind a llama.cpp dump burst) is a SKIP,
        // not an exit — `while let Ok` would silently kill the echo on the
        // first lag (Fable r3). Only a closed channel ends the loop.
        loop {
            match log_rx.recv().await {
                Ok(line) => println!("[{:?}] {}", line.source, line.text),
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            }
        }
    });
    let higgs = Arc::new(Higgs::with_log_bus(config, bus));
    if let Err(e) = higgs.start().await {
        eprintln!("FATAL: higgs.start() failed: {e}");
        return;
    }

    // Optional ephemeral pre-load: pin an exact LoadParams config for this
    // serve session without touching the persisted profiles.
    // `var_os`, not `var`: a SET-but-non-UTF-8 value must be FATAL like every
    // other malformed pin request — never silently serve unpinned.
    if let Some(path_os) = std::env::var_os("HIGGS_PARAMS_JSON") {
        let Some(path) = path_os.to_str().map(str::to_owned) else {
            eprintln!("FATAL: HIGGS_PARAMS_JSON is set but not valid UTF-8");
            higgs.stop().await;
            return;
        };
        // SET-but-EMPTY is a caller error (an unset shell variable expanded
        // into the env), not "no pin": serving unpinned when a pin was asked
        // for is the silent-wrong-config failure this contract forbids.
        if path.is_empty() {
            eprintln!("FATAL: HIGGS_PARAMS_JSON is set but empty");
            higgs.stop().await;
            return;
        }
        {
            #[derive(serde::Deserialize)]
            struct ParamsFile {
                model: String,
                params: LoadParams,
            }
            let parsed: Result<ParamsFile, String> = std::fs::read_to_string(&path)
                .map_err(|e| e.to_string())
                .and_then(|s| serde_json::from_str(&s).map_err(|e| e.to_string()));
            match parsed {
                Ok(pf) => {
                    // The idle reaper would eject the pinned worker after the
                    // default TTL, and the NEXT chat would JIT-reload with the
                    // SAVED profile — silently un-pinning the config this whole
                    // mechanism exists to pin (Fable r2). Ephemeral serve = no
                    // auto-unload for the server's lifetime.
                    higgs.set_auto_unload_idle(false);
                    println!("ephemeral pre-load: {} from {path}", pf.model);
                    if let Err(e) = higgs.load_ephemeral(&pf.model, pf.params).await {
                        eprintln!("FATAL: ephemeral load failed: {e}");
                        higgs.stop().await;
                        return;
                    }
                }
                Err(e) => {
                    eprintln!("FATAL: bad HIGGS_PARAMS_JSON ({path}): {e}");
                    higgs.stop().await;
                    return;
                }
            }
        }
    }

    let listener = match tokio::net::TcpListener::bind(&addr).await {
        Ok(l) => l,
        Err(e) => {
            eprintln!("FATAL: bind {addr}: {e}");
            higgs.stop().await;
            return;
        }
    };
    println!(
        "higgs /v1 serving on http://{} (Ctrl-C to stop)",
        listener.local_addr().expect("bound listener has an addr")
    );

    let shutdown = async {
        let _ = tokio::signal::ctrl_c().await;
        println!("shutting down…");
    };
    if let Err(e) = higgs::serve::serve_v1(higgs, listener, shutdown).await {
        eprintln!("serve_v1: {e}");
    }
}
