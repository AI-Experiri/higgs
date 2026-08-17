//! Black-box `higgs model download` / `higgs model downloads` CLI e2e: the DL2
//! machine-wide download slice driven through the REAL binary — flock authority,
//! ledger status recording, the three staleness sweeps, and the ledger renderer —
//! against a loopback fixture Hub (`HIGGS_HF_ENDPOINT`). No GGUF, no server, no
//! fleet: every child gets its own `HIGGS_HOME`, so no env locking is needed.

use std::future::IntoFuture;
use std::path::Path;
use std::time::Duration;

use axum::response::IntoResponse;
use axum::routing::get;

use higgs::catalog::wire::{DownloadLedgerEntry, DownloadLedgerStatus};

/// Loopback "Hugging Face": bytes for any resolve path, canned 404 for
/// `missing-*`, and a 3s stall for `slow-*` (keeps a transfer in flight long
/// enough for a second process to collide with the machine-wide flock).
async fn fixture_hub() -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = axum::Router::new().fallback(get(|uri: axum::http::Uri| async move {
        if uri.path().contains("missing-") {
            (axum::http::StatusCode::NOT_FOUND, "no such file").into_response()
        } else if uri.path().contains("slow-") {
            tokio::time::sleep(Duration::from_secs(3)).await;
            b"GGUF-cli-bytes".to_vec().into_response()
        } else {
            b"GGUF-cli-bytes".to_vec().into_response()
        }
    }));
    tokio::spawn(axum::serve(listener, app).into_future());
    format!("http://{addr}")
}

fn higgs_model_cmd(home: &Path, endpoint: &str, args: &[&str]) -> tokio::process::Command {
    let mut c = tokio::process::Command::new(env!("CARGO_BIN_EXE_higgs"));
    c.arg("model")
        .args(args)
        .env("HIGGS_HOME", home)
        .env("HIGGS_HF_ENDPOINT", endpoint);
    c
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cli_download_lands_the_file_and_the_ledger_renders_it_done() {
    let endpoint = fixture_hub().await;
    let home = tempfile::tempdir().unwrap();

    let out = higgs_model_cmd(
        home.path(),
        &endpoint,
        &["download", "acme/tiny", "tiny-Q4_K_M.gguf"],
    )
    .output()
    .await
    .expect("spawn higgs model download");
    assert!(
        out.status.success(),
        "download succeeds: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let expected = home.path().join("models/acme/tiny/tiny-Q4_K_M.gguf");
    assert!(
        String::from_utf8_lossy(&out.stdout).contains(&expected.display().to_string()),
        "the landed path is printed: {}",
        String::from_utf8_lossy(&out.stdout)
    );
    assert_eq!(std::fs::read(&expected).unwrap(), b"GGUF-cli-bytes");

    // The machine ledger recorded the terminal — `higgs model downloads`
    // renders the Done row with its final byte count.
    let out = higgs_model_cmd(home.path(), &endpoint, &["downloads"])
        .output()
        .await
        .expect("spawn higgs model downloads");
    let listing = String::from_utf8_lossy(&out.stdout).to_string();
    assert!(out.status.success(), "downloads listing succeeds");
    assert!(
        listing.contains("acme/tiny/tiny-Q4_K_M.gguf") && listing.contains("done"),
        "ledger renders the completed transfer: {listing}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cli_download_failure_is_a_coded_error_and_renders_failed() {
    let endpoint = fixture_hub().await;
    let home = tempfile::tempdir().unwrap();

    let out = higgs_model_cmd(
        home.path(),
        &endpoint,
        &["download", "acme/tiny", "missing-file.gguf"],
    )
    .output()
    .await
    .expect("spawn higgs model download");
    assert!(!out.status.success(), "404 on both fetchers fails the CLI");
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    assert!(
        stderr.contains("HG"),
        "a classified higgs code reaches the operator: {stderr}"
    );
    assert!(
        !home
            .path()
            .join("models/acme/tiny/missing-file.gguf")
            .exists(),
        "nothing landed"
    );

    let out = higgs_model_cmd(home.path(), &endpoint, &["downloads"])
        .output()
        .await
        .expect("spawn higgs model downloads");
    let listing = String::from_utf8_lossy(&out.stdout).to_string();
    assert!(
        listing.contains("acme/tiny/missing-file.gguf") && listing.contains("failed"),
        "ledger renders the failed transfer with its detail: {listing}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_concurrent_cli_download_of_the_same_key_refuses_machine_wide() {
    let endpoint = fixture_hub().await;
    let home = tempfile::tempdir().unwrap();

    // Child A: a slow transfer that holds the per-key flock for ~3s.
    let a = higgs_model_cmd(
        home.path(),
        &endpoint,
        &["download", "acme/tiny", "slow-dup.gguf"],
    )
    .stdout(std::process::Stdio::piped())
    .stderr(std::process::Stdio::piped())
    .spawn()
    .expect("spawn slow download");

    // Wait until A provably owns the machine-wide slot (its lock file exists —
    // created BEFORE any bytes move), bounded so a broken A can't hang us.
    let locks_dir = home.path().join("models/.download-locks");
    let mut armed = false;
    for _ in 0..100 {
        if locks_dir
            .read_dir()
            .map(|mut d| d.next().is_some())
            .unwrap_or(false)
        {
            armed = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert!(armed, "child A acquired the download lock");

    // Child B: same key while A is mid-transfer → HG090 refusal, fast, with
    // ZERO side effects (no second temp, no second ledger writer).
    let out = higgs_model_cmd(
        home.path(),
        &endpoint,
        &["download", "acme/tiny", "slow-dup.gguf"],
    )
    .output()
    .await
    .expect("spawn duplicate download");
    assert!(!out.status.success(), "duplicate is refused");
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    assert!(
        stderr.contains("HG090") || stderr.contains("already"),
        "the refusal is the HG090 contention class: {stderr}"
    );

    // A still completes — the refusal never disturbed the live transfer.
    let a_out = a.wait_with_output().await.expect("child A finishes");
    assert!(
        a_out.status.success(),
        "the original transfer completes: {}",
        String::from_utf8_lossy(&a_out.stderr)
    );
    let expected = home.path().join("models/acme/tiny/slow-dup.gguf");
    assert_eq!(std::fs::read(&expected).unwrap(), b"GGUF-cli-bytes");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_ledger_sweeps_retire_dead_stale_and_ancient_rows_on_read() {
    // Seed the three residue shapes the sweeps exist for, then let a REAL
    // process read the ledger (`higgs model downloads`):
    //   (a) dead-pid row            → swept to Failed (process exited)
    //   (b) live-pid row + lock     → swept to Failed (lock file exists but
    //       file UNHELD                nobody holds the flock — is_key_stale)
    //   (c) live-pid row, no lock   → swept to Failed (lockless legacy row
    //       file, 25h old             past the 24h grace window)
    let endpoint = fixture_hub().await;
    let home = tempfile::tempdir().unwrap();
    let models = home.path().join("models");
    std::fs::create_dir_all(&models).unwrap();

    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;
    let mk = |repo: &str, file: &str, pid: u32, started_at_ms: u64| DownloadLedgerEntry {
        repo: repo.into(),
        file: file.into(),
        pid,
        pid_started_at: None,
        started_at_ms,
        downloaded: 5,
        total: Some(10),
        status: DownloadLedgerStatus::Downloading,
        ended_at_ms: None,
        path: None,
        detail: None,
    };
    let rows = vec![
        mk("acme/dead", "dead.gguf", u32::MAX - 1, now_ms), // no such pid
        mk("acme/stale", "stale.gguf", 1, now_ms),          // pid 1 is alive
        mk("acme/old", "old.gguf", 1, now_ms - 25 * 60 * 60 * 1000),
    ];
    std::fs::write(
        models.join(".downloads.json"),
        serde_json::to_vec(&rows).unwrap(),
    )
    .unwrap();
    // Shape (b): the lock FILE exists but no process holds the flock.
    std::fs::create_dir_all(models.join(".download-locks")).unwrap();
    let stale_lock = higgs::catalog::download_lock::lock_path(&models, "acme/stale", "stale.gguf");
    std::fs::write(&stale_lock, b"").unwrap();

    let out = higgs_model_cmd(home.path(), &endpoint, &["downloads"])
        .output()
        .await
        .expect("spawn higgs model downloads");
    assert!(out.status.success());
    let listing = String::from_utf8_lossy(&out.stdout).to_string();
    for key in [
        "acme/dead/dead.gguf",
        "acme/stale/stale.gguf",
        "acme/old/old.gguf",
    ] {
        assert!(listing.contains(key), "row {key} still listed: {listing}");
    }
    assert!(
        !listing.contains("downloading"),
        "every residue shape was swept to a terminal — none reads as live: {listing}"
    );
    assert_eq!(
        listing.matches("failed").count(),
        3,
        "all three sweeps flip their rows to Failed: {listing}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_empty_ledger_and_a_bad_subcommand_render_their_own_messages() {
    let endpoint = fixture_hub().await;
    let home = tempfile::tempdir().unwrap();

    // No downloads ever → the renderer's empty arm.
    let out = higgs_model_cmd(home.path(), &endpoint, &["downloads"])
        .output()
        .await
        .expect("spawn higgs model downloads");
    assert!(out.status.success());
    assert!(
        String::from_utf8_lossy(&out.stdout).contains("no downloads recorded"),
        "empty-ledger message: {}",
        String::from_utf8_lossy(&out.stdout)
    );

    // Unknown subcommand → usage on stderr, nonzero exit.
    let out = higgs_model_cmd(home.path(), &endpoint, &["bogus"])
        .output()
        .await
        .expect("spawn higgs model bogus");
    assert!(!out.status.success(), "unknown subcommand fails");
    assert!(
        !out.stderr.is_empty(),
        "usage is printed for the unknown subcommand"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_live_foreign_transfer_renders_downloading_with_percent() {
    // THIS test process stands in for a sibling `higgs download`: it holds
    // the machine-wide flock and owns a live Downloading ledger row. The
    // child (`higgs model downloads`) must KEEP the row through every sweep
    // (live pid + held lock) and render the progress/percent arm.
    let endpoint = fixture_hub().await;
    let home = tempfile::tempdir().unwrap();
    let models = home.path().join("models");
    std::fs::create_dir_all(&models).unwrap();

    let _held =
        higgs::catalog::download_lock::DownloadLock::acquire(&models, "acme/live", "live.gguf")
            .expect("test process owns the machine-wide slot");
    let row = DownloadLedgerEntry {
        repo: "acme/live".into(),
        file: "live.gguf".into(),
        pid: std::process::id(),
        pid_started_at: None,
        started_at_ms: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64,
        downloaded: 5,
        total: Some(10),
        status: DownloadLedgerStatus::Downloading,
        ended_at_ms: None,
        path: None,
        detail: None,
    };
    std::fs::write(
        models.join(".downloads.json"),
        serde_json::to_vec(&vec![row]).unwrap(),
    )
    .unwrap();

    let out = higgs_model_cmd(home.path(), &endpoint, &["downloads"])
        .output()
        .await
        .expect("spawn higgs model downloads");
    assert!(out.status.success());
    let listing = String::from_utf8_lossy(&out.stdout).to_string();
    assert!(
        listing.contains("acme/live/live.gguf") && listing.contains("downloading"),
        "a genuinely live foreign transfer is NEVER swept: {listing}"
    );
    assert!(
        listing.contains("(50%)"),
        "the progress/percent arm renders from the live counters: {listing}"
    );
}
