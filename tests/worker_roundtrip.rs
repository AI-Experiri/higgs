//! Worker re-exec round-trip integration test.
//!
//! Re-executes this test binary under `HIGGS_WORKER_TEST=1` so the child runs
//! `worker_main()` directly on its stdio — the same Chromium-style re-exec used
//! by the real supervisor.  The parent drives real pipes: scan → status →
//! shutdown, asserts each response, and waits for a clean exit.
//!
//! Engine methods (load/chat) are intentionally NOT exercised here; those paths
//! are covered by unit tests in `worker/mod.rs`.

use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::time::Duration;

use higgs::rpc::{decode, encode, RpcFrame, RpcRequest, RpcResponse};

/// Env-var sentinel that tells the re-exec'd child to run `worker_main()` and exit.
const SENTINEL: &str = "HIGGS_WORKER_TEST";

/// Build one NDJSON request line.
fn req_line(id: u64, method: &str, params: serde_json::Value) -> String {
    let r = RpcRequest {
        jsonrpc: "2.0".into(),
        id,
        method: method.into(),
        params,
    };
    format!("{}\n", encode(&RpcFrame::Request(r)))
}

/// Write `content` at `path`, creating parent directories as needed.
fn write_file(path: &Path, content: &[u8]) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(path, content).unwrap();
}

/// Parse one NDJSON line into an `RpcResponse`, panicking on anything else.
fn parse_response(line: &str) -> RpcResponse {
    match decode(line).expect("valid rpc frame") {
        RpcFrame::Response(r) => r,
        other => panic!("expected Response, got: {other:?}"),
    }
}

/// Read exactly one response line from `rx` with a 10-second deadline.
fn recv_line(rx: &mpsc::Receiver<String>, label: &str) -> String {
    rx.recv_timeout(Duration::from_secs(10))
        .unwrap_or_else(|e| panic!("timed out waiting for {label}: {e}"))
}

// ---------------------------------------------------------------------------
// The test
// ---------------------------------------------------------------------------

#[test]
fn worker_stdio_roundtrip() {
    // ---- child role ---------------------------------------------------------
    // When the re-exec'd process enters this test, HIGGS_WORKER_TEST is set.
    // `worker_main()` runs the serve loop on real stdio and exits 0 when
    // `higgs/shutdown` is received (or stdin closes).
    if std::env::var(SENTINEL).is_ok() {
        higgs::worker::worker_main();
        std::process::exit(0);
    }

    // ---- parent role --------------------------------------------------------

    // Build a minimal LM Studio fixture: root/google/gemma-4-12b/file-Q4_K_M.gguf
    let tmp = tempfile::TempDir::new().expect("tempdir");
    write_file(
        &tmp.path().join("google/gemma-4-12b/gemma-4-12b-Q4_K_M.gguf"),
        &[0u8; 16], // 16 dummy bytes — not a valid GGUF; scan tolerates it
    );
    let lmstudio_root = tmp.path().to_str().expect("utf-8 path").to_string();

    // Re-exec this test binary as the child worker.
    // `--exact worker_stdio_roundtrip --nocapture` makes the child jump straight
    // into this test function, where it sees HIGGS_WORKER_TEST and calls worker_main().
    let exe = std::env::current_exe().expect("current_exe");
    let mut child = Command::new(&exe)
        .args(["--exact", "worker_stdio_roundtrip", "--nocapture"])
        .env(SENTINEL, "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit()) // worker logs go to stderr; keep them visible
        .spawn()
        .expect("spawn child worker");

    let mut child_stdin = child.stdin.take().expect("child stdin");
    let child_stdout = child.stdout.take().expect("child stdout");

    // Drain child stdout on a background thread; send lines over a channel so
    // the main thread can recv with a timeout (never hang CI).
    // Filter to JSON-object lines only: the cargo test harness emits "running N
    // tests" / "test foo ... ok" banners on stdout before worker_main() takes
    // over; those never start with '{', so they are safely discarded here.
    let (tx, rx) = mpsc::channel::<String>();
    std::thread::spawn(move || {
        let reader = BufReader::new(child_stdout);
        for line in reader.lines() {
            match line {
                Ok(l) if l.trim_start().starts_with('{') => {
                    if tx.send(l).is_err() {
                        break;
                    }
                }
                Ok(_) => {} // test harness banner or blank — skip
                Err(_) => break,
            }
        }
    });

    // ---- request 1: higgs/scan ---------------------------------------------
    let scan_req = req_line(
        1,
        "higgs/scan",
        serde_json::json!({
            "lmstudio": [lmstudio_root],
            "hf": [],
            "ollama": []
        }),
    );
    child_stdin.write_all(scan_req.as_bytes()).expect("write scan");
    child_stdin.flush().expect("flush scan");

    let scan_line = recv_line(&rx, "scan response");
    let scan_resp = parse_response(&scan_line);
    assert_eq!(scan_resp.id, 1, "scan id");
    assert!(scan_resp.error.is_none(), "scan error: {:?}", scan_resp.error);
    let models = scan_resp
        .result
        .as_ref()
        .and_then(|v| v.as_array())
        .expect("scan result is array");
    assert_eq!(models.len(), 1, "expected exactly one model: {models:?}");
    assert_eq!(
        models[0]["id"], "google/gemma-4-12b",
        "model id mismatch: {models:?}"
    );

    // ---- request 2: higgs/status -------------------------------------------
    let status_req = req_line(2, "higgs/status", serde_json::Value::Null);
    child_stdin.write_all(status_req.as_bytes()).expect("write status");
    child_stdin.flush().expect("flush status");

    let status_line = recv_line(&rx, "status response");
    let status_resp = parse_response(&status_line);
    assert_eq!(status_resp.id, 2, "status id");
    assert!(status_resp.error.is_none(), "status error: {:?}", status_resp.error);
    let result = status_resp.result.as_ref().expect("status result");
    assert_eq!(result["models_scanned"], 1, "models_scanned after scan");
    assert!(
        result["loaded"].is_null(),
        "nothing should be loaded yet: {}",
        result["loaded"]
    );

    // ---- request 3: higgs/shutdown -----------------------------------------
    let shutdown_req = req_line(3, "higgs/shutdown", serde_json::Value::Null);
    child_stdin.write_all(shutdown_req.as_bytes()).expect("write shutdown");
    child_stdin.flush().expect("flush shutdown");

    let shutdown_line = recv_line(&rx, "shutdown response");
    let shutdown_resp = parse_response(&shutdown_line);
    assert_eq!(shutdown_resp.id, 3, "shutdown id");
    assert!(
        shutdown_resp.error.is_none(),
        "shutdown error: {:?}",
        shutdown_resp.error
    );

    // Close stdin so the child doesn't block if it somehow missed the shutdown.
    drop(child_stdin);

    // ---- wait for child exit (max 5 s) -------------------------------------
    let done = {
        let (exit_tx, exit_rx) = mpsc::channel::<std::process::ExitStatus>();
        std::thread::spawn(move || {
            let status = child.wait().expect("wait");
            let _ = exit_tx.send(status);
        });
        exit_rx.recv_timeout(Duration::from_secs(5))
    };
    match done {
        Ok(status) => assert!(status.success(), "child exited non-zero: {status}"),
        Err(_) => panic!("child did not exit within 5 s after shutdown"),
    }
}
