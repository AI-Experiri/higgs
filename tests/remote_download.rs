//! Black-box REMOTE catalog download (`Higgs::model_download_on`, RD1): the
//! in-process hub triggers `M_NODE_PULL` on a REAL `higgs --node` child over
//! hermetic iroh; the NODE fetches the file from a loopback fixture Hub (its
//! own `HIGGS_HF_ENDPOINT`) into its own `HIGGS_HOME/models` — only progress
//! crosses the fleet, tagged onto the ONE `ModelDownloadEvent` stream. Skips
//! cleanly without the tiny GGUF (the shared `higgs_local` harness gate).

mod common;

use std::future::IntoFuture;
use std::process::{Child, Command, Stdio};
use std::time::Duration;

use axum::response::IntoResponse;
use axum::routing::get;

use common::higgs_local;
use higgs::catalog::ModelDownloadPhase;
use higgs::diagnostic::HiggsError;

struct Proc(Child);
impl Drop for Proc {
    fn drop(&mut self) {
        // SAFETY: a plain kill(2) on our own child's pid.
        unsafe { libc::kill(self.0.id() as libc::pid_t, libc::SIGTERM) };
        let _ = self.0.wait();
    }
}

/// Loopback "Hugging Face" for the NODE: serves GGUF bytes for any resolve
/// path except the one canned 404 (`missing-*.gguf`).
async fn fixture_hub() -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = axum::Router::new().fallback(get(|uri: axum::http::Uri| async move {
        if uri.path().contains("missing-") {
            (axum::http::StatusCode::NOT_FOUND, "no such file").into_response()
        } else if uri.path().contains("slow-") {
            // Keep a transfer IN FLIGHT long enough for a duplicate re-issue
            // to collide with it (the HG090 adopt-as-downloading section).
            tokio::time::sleep(Duration::from_secs(3)).await;
            b"GGUF-remote-bytes".to_vec().into_response()
        } else {
            b"GGUF-remote-bytes".to_vec().into_response()
        }
    }));
    tokio::spawn(axum::serve(listener, app).into_future());
    format!("http://{addr}")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn remote_download_lands_on_the_node_with_tagged_events() {
    let Some(higgs) = higgs_local(&[]).await else {
        eprintln!("skipping remote_download: no tiny GGUF (set HIGGS_TEST_GGUF)");
        return;
    };
    higgs.hub_enable().await.expect("hub enable");
    let pair = higgs.pair().await.expect("mint pairing credential");

    let endpoint = fixture_hub().await;
    let node_home = tempfile::tempdir().unwrap();
    let _node = Proc(
        Command::new(env!("CARGO_BIN_EXE_higgs"))
            .arg("--node")
            .arg(&pair.ticket)
            .arg(&pair.token)
            .env("HIGGS_HOME", node_home.path())
            .env("HIGGS_HF_ENDPOINT", &endpoint)
            .env("HIGGS_IROH_LOCAL", "1")
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("spawn higgs --node"),
    );

    let mut node_id = String::new();
    for _ in 0..150 {
        if let Some(n) = higgs
            .nodes()
            .await
            .into_iter()
            .find(|n| n.connected && !n.is_local)
        {
            node_id = n.endpoint_id.clone();
            break;
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    assert!(!node_id.is_empty(), "remote node paired + connected");

    // ── happy path: the file lands in the NODE's models dir ────────────────
    let mut rx = higgs.subscribe_download_events();
    let path = higgs
        .model_download_on(&node_id, "acme/tiny", "tiny-Q4_K_M.gguf")
        .await
        .expect("remote download");
    let expected = node_home.path().join("models/acme/tiny/tiny-Q4_K_M.gguf");
    assert_eq!(
        std::path::Path::new(&path),
        expected,
        "node-side path returned"
    );
    assert_eq!(std::fs::read(&expected).unwrap(), b"GGUF-remote-bytes");

    let mut phases = Vec::new();
    while let Ok(ev) = rx.try_recv() {
        assert_eq!(ev.node.as_deref(), Some(node_id.as_str()), "events tagged");
        assert_eq!(ev.repo, "acme/tiny");
        phases.push(ev.phase);
    }
    assert_eq!(phases.first(), Some(&ModelDownloadPhase::Starting));
    assert_eq!(phases.last(), Some(&ModelDownloadPhase::Done));

    // ── node-side failure: the Hub 404s → coded error + terminal Failed ────
    let mut rx = higgs.subscribe_download_events();
    let err = higgs
        .model_download_on(&node_id, "acme/tiny", "missing-file.gguf")
        .await
        .expect_err("404 on the node's Hub fetch");
    // The event must carry the node-ORIGIN code (what actually failed on the
    // node), never the HG009 relay envelope.
    let origin = match &err {
        HiggsError::WorkerRpc { worker_code, .. } => worker_code.clone(),
        other => miette::Diagnostic::code(other).map(|c| c.to_string()),
    };
    assert!(origin.is_some(), "coded error: {err}");
    let mut phases = Vec::new();
    while let Ok(ev) = rx.try_recv() {
        phases.push((ev.phase, ev.code));
    }
    assert_eq!(
        phases.first().map(|p| p.0),
        Some(ModelDownloadPhase::Starting)
    );
    let (last_phase, last_code) = phases.last().cloned().expect("terminal");
    assert_eq!(last_phase, ModelDownloadPhase::Failed);
    assert_eq!(last_code, origin, "the event carries the node-origin code");
    assert_ne!(
        last_code.as_deref(),
        Some("HG009"),
        "never the relay envelope"
    );

    // ── invalid file: refused after Starting, no stream ever opened ────────
    let mut rx = higgs.subscribe_download_events();
    let err = higgs
        .model_download_on(&node_id, "acme/tiny", "evil.txt")
        .await
        .expect_err("non-gguf refused");
    assert!(matches!(err, HiggsError::DownloadFailed { .. }));
    let mut phases = Vec::new();
    while let Ok(ev) = rx.try_recv() {
        phases.push(ev.phase);
    }
    assert_eq!(
        phases,
        vec![ModelDownloadPhase::Starting, ModelDownloadPhase::Failed],
        "slot-claimed contract holds for the validation refusal"
    );

    // ── duplicate while IN FLIGHT: request refused, THIS attempt cancelled ──
    // Start a slow transfer, then re-issue the SAME (repo, file) while it is
    // still running. The hub-facade duplicate gate refuses this second
    // attempt ([HG090]) before it ever reaches the node; the refusal emits a
    // `Cancelled` TERMINAL carrying code HG090 (this attempt is honestly
    // cancelled in favor of the live one — no ownerless non-terminal row,
    // no blocking live-check RPC). The ORIGINAL transfer's visibility lives
    // in `NodeView.downloads`, not this attempt's event stream, so the
    // original keeps flowing Downloading/Done as normal and the row is
    // NEVER painted Failed. Fail-on-revert: emit Failed{HG090} for the
    // refusal and the no-Failed assert below fails.
    let mut rx = higgs.subscribe_download_events();
    let slow = {
        let higgs = higgs.clone();
        let node_id = node_id.clone();
        tokio::spawn(async move {
            higgs
                .model_download_on(&node_id, "acme/tiny", "slow-dup.gguf")
                .await
        })
    };
    // Give the node time to receive the pull and register it (the fixture
    // then holds its transfer open for 3 s).
    tokio::time::sleep(Duration::from_millis(700)).await;
    let err = higgs
        .model_download_on(&node_id, "acme/tiny", "slow-dup.gguf")
        .await
        .expect_err("the duplicate REQUEST is refused while the original runs");
    let origin = match &err {
        HiggsError::WorkerRpc { worker_code, .. } => worker_code.clone(),
        other => miette::Diagnostic::code(other).map(|c| c.to_string()),
    };
    assert_eq!(origin.as_deref(), Some("HG090"), "coded refusal: {err}");
    slow.await
        .expect("join")
        .expect("the ORIGINAL transfer is undisturbed and completes");
    let mut phases = Vec::new();
    while let Ok(ev) = rx.try_recv() {
        phases.push((ev.phase, ev.code));
    }
    assert!(
        phases
            .iter()
            .any(|(p, c)| *p == ModelDownloadPhase::Cancelled && c.as_deref() == Some("HG090")),
        "the refused attempt terminalizes with Cancelled{{HG090}}: {phases:?}"
    );
    assert!(
        !phases.iter().any(|(p, _)| *p == ModelDownloadPhase::Failed),
        "a running download is never painted failed by a duplicate re-issue: {phases:?}"
    );
    assert_eq!(
        phases.last().map(|p| p.0),
        Some(ModelDownloadPhase::Done),
        "the original ends Done"
    );

    // ── ledger union: ANOTHER process's download on the node box shows ─────
    // A `higgs download` CLI running on the node machine is a separate
    // process the node daemon's in-memory registry cannot see — the machine
    // LEDGER is how it becomes fleet-visible. This test process stands in for
    // that CLI: it writes a live ledger entry under ITS OWN (alive) pid into
    // the node's models root; the node's pull_status must announce it.
    // Fail-on-revert: announce registry-only and the entry never appears.
    {
        use higgs::catalog::ledger;
        use higgs::catalog::wire::{DownloadLedgerEntry, DownloadLedgerStatus};
        let node_models = node_home.path().join("models");
        let alien = DownloadLedgerEntry {
            repo: "acme/cli".into(),
            file: "cli.gguf".into(),
            pid: std::process::id(),
            pid_started_at: None,
            // FRESH: a lockless row older than the 24h grace window is
            // reaped by the legacy sweep (r64) before it can be announced —
            // this row stands in for a LIVE CLI download, so it must sit
            // inside the bootstrap window.
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
            ledger::ledger_path(&node_models),
            serde_json::to_vec(&vec![alien]).unwrap(),
        )
        .unwrap();
        let listed = higgs.node_downloads(&node_id).await.expect("status poll");
        let cli = listed
            .iter()
            .find(|d| d.repo == "acme/cli" && d.file == "cli.gguf")
            .unwrap_or_else(|| panic!("the other process's download is announced: {listed:?}"));
        assert_eq!((cli.downloaded, cli.total), (5, Some(10)));
    }

    // ── unknown node: unreachable (HG027) ──────────────────────────────────
    let bad = "0000000000000000000000000000000000000000000000000000000000000000";
    let err = higgs
        .model_download_on(bad, "acme/tiny", "tiny-Q4_K_M.gguf")
        .await
        .expect_err("unknown node");
    assert!(matches!(err, HiggsError::NodeUnreachable { .. }), "{err:?}");

    higgs.shutdown().await;
}
