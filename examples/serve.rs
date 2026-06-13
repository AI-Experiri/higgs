//! Standalone higgs server — FOR TESTING ONLY (higgs ships as a library crate;
//! production wires it in via the `Higgs` API + `serve::router`, not this binary).
//!
//! Runs the full higgs stack on its own, with no jigglebot server or frontend:
//! constructs `Higgs`, starts the worker (re-execs THIS example binary with
//! `--higgs-worker`), and serves `/v1` + `/api/higgs/*` on a local port. Lets
//! the whole stack (worker dispatch, model scan, load, chat) be exercised with
//! plain curl, independent of the app.
//!
//! ```text
//! env -u LIBCLANG_PATH cargo run -p higgs --example serve            # default :11434
//! env -u LIBCLANG_PATH cargo run -p higgs --example serve -- 11500   # custom port
//!
//! curl localhost:11434/api/higgs/status                 # did the worker start?
//! curl localhost:11434/api/higgs/models                 # scanned GGUFs
//! curl -XPOST localhost:11434/api/higgs/models/load -d '{"id":"<id>"}'
//! curl -XPOST localhost:11434/v1/chat/completions -d '{"model":"<id>","messages":[...]}'
//! ```

use std::sync::Arc;

use higgs::{Higgs, HiggsConfig};

fn main() {
    // Worker role: the supervisor re-execs THIS binary with `--higgs-worker`.
    // Detect it before anything touches stdout — worker_main() owns stdin/stdout
    // for NDJSON JSON-RPC. (Same convention the jigglebot server `main` honors.)
    if std::env::args().skip(1).any(|a| a == "--higgs-worker") {
        higgs::worker::worker_main();
        return;
    }

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let port: u16 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(11434);

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("build tokio runtime");

    rt.block_on(async move {
        let higgs = Arc::new(Higgs::new(HiggsConfig::default()));
        if let Err(e) = higgs.start().await {
            eprintln!("higgs failed to start (worker spawn): {e}");
            std::process::exit(1);
        }

        let app = higgs::serve::router(Arc::clone(&higgs));
        let addr = format!("127.0.0.1:{port}");
        let listener = tokio::net::TcpListener::bind(&addr)
            .await
            .unwrap_or_else(|e| panic!("bind {addr}: {e}"));

        println!("higgs standalone serving on http://{addr}  (/v1 + /api/higgs)");
        println!("  curl http://{addr}/api/higgs/status");
        axum::serve(listener, app).await.expect("serve");
    });
}
