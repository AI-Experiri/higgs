use super::*;
use crate::node::test_support::local_endpoint;
use crate::remote::{ALPN, M_NODE_PULL, N_PROGRESS};
use crate::rpc;
use tokio::io::{AsyncBufReadExt, BufReader};

/// A no-network fetcher: one chunk + one progress tick.
struct FakeFetcher;
impl crate::download::Fetcher for FakeFetcher {
    async fn fetch(
        &self,
        _target: &crate::download::PullTarget,
        on_chunk: &mut (dyn FnMut(&[u8]) + Send),
        progress: &mut (dyn FnMut(u64, Option<u64>) + Send),
    ) -> Result<(), HiggsError> {
        on_chunk(b"hello");
        progress(5, Some(5));
        Ok(())
    }
}

/// A fetcher that always fails (transport) — for exercising the dual-path fallback.
struct FailFetcher;
impl crate::download::Fetcher for FailFetcher {
    async fn fetch(
        &self,
        target: &crate::download::PullTarget,
        _on_chunk: &mut (dyn FnMut(&[u8]) + Send),
        _progress: &mut (dyn FnMut(u64, Option<u64>) + Send),
    ) -> Result<(), HiggsError> {
        Err(HiggsError::HubTransport {
            repo: target.repo.clone(),
            detail: "fake primary down".into(),
        })
    }
}

/// Pin the refusal-log discriminator: only HG090 (a live foreign holder of
/// the machine-wide slot) reads as contention; an HG034 filesystem failure
/// from `DownloadLock::acquire` (locks dir uncreatable, lock file
/// unopenable) must NOT be logged as "already in flight" — reverting the
/// discriminated log branch back to one fixed message fails this pin.
#[test]
fn only_a_live_holder_refusal_classifies_as_contention() {
    let contention = HiggsError::DownloadInFlight {
        repo: "acme/m".into(),
        file: "m.gguf".into(),
    };
    assert!(acquire_refusal_is_contention(&contention));

    let io_fault = HiggsError::HubFileWrite {
        repo: "acme/m".into(),
        file: "m.gguf".into(),
        detail: "create download-locks dir /nope: permission denied".into(),
    };
    assert!(!acquire_refusal_is_contention(&io_fault));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pull_stream_streams_progress_then_final_path() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().to_path_buf();
    let node = local_endpoint().await;
    let hub = local_endpoint().await;
    let node_addr = node.addr();

    // Node side: accept the bi stream, read the M_NODE_PULL request, run pull_stream.
    tokio::spawn(async move {
        let conn = node.accept().await.unwrap().await.unwrap();
        let (mut send, recv) = conn.accept_bi().await.unwrap();
        let mut lines = BufReader::new(recv).lines();
        let line = lines.next_line().await.unwrap().unwrap();
        let RpcFrame::Request(req) = rpc::decode(&line).unwrap() else {
            panic!("want request")
        };
        // Primary FAILS, fallback succeeds — exercises the dual-path fallback end-to-end:
        // the streamed bytes + final path must still come from the fallback.
        pull_stream(
            &conn,
            &mut send,
            req,
            FailFetcher,
            FakeFetcher,
            root,
            // Fresh LEAKED registry per test — the process-global node_registry()
            // is shared state: a same-key sibling test (or an in-process
            // retry racing the guard drop) would be refused [HG090].
            Box::leak(Box::new(crate::catalog::cancel::PullCancelRegistry::new())),
        )
        .await;
        let _ = send.finish();
        // keep conn alive until the hub reads
        let _ = conn.closed().await;
    });

    let conn = hub.connect(node_addr, ALPN).await.unwrap();
    let (mut send, recv) = conn.open_bi().await.unwrap();
    let req = RpcRequest {
        jsonrpc: "2.0".into(),
        id: 1,
        method: M_NODE_PULL.into(),
        params: json!({ "request_id": 1, "repo": "org/m", "file": "x.gguf" }),
    };
    send.write_all(format!("{}\n", rpc::encode(&RpcFrame::Request(req))).as_bytes())
        .await
        .unwrap();
    send.finish().unwrap();

    let mut lines = BufReader::new(recv).lines();
    let mut progress = 0;
    let final_resp = loop {
        let line = lines.next_line().await.unwrap().expect("a frame");
        match rpc::decode(&line).unwrap() {
            RpcFrame::Notification(n) if n.method == N_PROGRESS => progress += 1,
            RpcFrame::Response(r) => break r,
            other => panic!("unexpected: {other:?}"),
        }
    };
    assert!(progress >= 1, "at least one N_PROGRESS");
    assert!(final_resp.error.is_none(), "pull ok: {final_resp:?}");
    let path = final_resp.result.unwrap()["path"]
        .as_str()
        .unwrap()
        .to_string();
    assert_eq!(
        std::fs::read(&path).unwrap(),
        b"hello",
        "downloaded bytes written"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pull_stream_rejects_bad_params() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().to_path_buf();
    let node = local_endpoint().await;
    let hub = local_endpoint().await;
    let node_addr = node.addr();
    tokio::spawn(async move {
        let conn = node.accept().await.unwrap().await.unwrap();
        let (mut send, recv) = conn.accept_bi().await.unwrap();
        let mut lines = BufReader::new(recv).lines();
        let line = lines.next_line().await.unwrap().unwrap();
        let RpcFrame::Request(req) = rpc::decode(&line).unwrap() else {
            panic!()
        };
        pull_stream(
            &conn,
            &mut send,
            req,
            FakeFetcher,
            FakeFetcher,
            root,
            // Fresh LEAKED registry per test — the process-global node_registry()
            // is shared state: a same-key sibling test (or an in-process
            // retry racing the guard drop) would be refused [HG090].
            Box::leak(Box::new(crate::catalog::cancel::PullCancelRegistry::new())),
        )
        .await;
        let _ = send.finish();
        let _ = conn.closed().await;
    });
    let conn = hub.connect(node_addr, ALPN).await.unwrap();
    let (mut send, recv) = conn.open_bi().await.unwrap();
    // Missing required fields → invalid params.
    let req = RpcRequest {
        jsonrpc: "2.0".into(),
        id: 9,
        method: M_NODE_PULL.into(),
        params: json!({}),
    };
    send.write_all(format!("{}\n", rpc::encode(&RpcFrame::Request(req))).as_bytes())
        .await
        .unwrap();
    send.finish().unwrap();
    let mut lines = BufReader::new(recv).lines();
    let line = lines.next_line().await.unwrap().unwrap();
    let RpcFrame::Response(r) = rpc::decode(&line).unwrap() else {
        panic!()
    };
    assert_eq!(r.error.unwrap().code, -32602, "invalid params");
}

#[test]
fn announced_downloads_validate_ledger_rows_at_the_producer() {
    // Ledger rows are FILE CONTENT: a semantically-bad row with a huge
    // repo/file would inflate this node's OWN HELLO/status frame past the
    // 64 KiB caps and cost it admission. The producer applies the same
    // validate-or-drop the hub does, so a bad status file only shrinks the
    // announcement. Runs under TEST_ENV_LOCK (models_dir derives from the
    // process-global HIGGS_HOME).
    let _env = crate::TEST_ENV_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let home = tempfile::tempdir().unwrap();
    let prev = std::env::var_os("HIGGS_HOME");
    // SAFETY: serialized by TEST_ENV_LOCK; restored below.
    unsafe { std::env::set_var("HIGGS_HOME", home.path()) };
    let models = crate::download::models_dir().expect("models dir");

    let mk = |file: String| crate::catalog::wire::DownloadLedgerEntry {
        repo: "acme/m".into(),
        file,
        pid: 1, // alive, never ours — a live "other process" row
        pid_started_at: None,
        started_at_ms: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64, // fresh: a lockless row older than the grace window is reaped
        downloaded: 5,
        total: Some(10),
        status: crate::catalog::wire::DownloadLedgerStatus::Downloading,
        ended_at_ms: None,
        path: None,
        detail: None,
    };
    let rows = vec![
        mk("ok.gguf".into()),
        mk(format!("{}.gguf", "q".repeat(60_000))),
    ];
    std::fs::write(
        crate::catalog::ledger::ledger_path(&models),
        serde_json::to_vec(&rows).unwrap(),
    )
    .unwrap();

    let out = announced_downloads();
    assert!(
        out.iter().any(|d| d.file == "ok.gguf"),
        "the valid other-process row is announced: {out:?}"
    );
    assert!(
        !out.iter().any(|d| d.file.len() > 300),
        "an oversized row is dropped at the producer, protecting our own frame"
    );

    // SAFETY: still under the lock.
    unsafe {
        match prev {
            Some(v) => std::env::set_var("HIGGS_HOME", v),
            None => std::env::remove_var("HIGGS_HOME"),
        }
    }
}

#[test]
fn announced_downloads_marks_registry_backed_cancellable_ledger_only_not() {
    // The `cancellable` bit on a HelloDownload row means "this node's OWN
    // process registry can cancel it". Registry entries → true. Ledger
    // rows from OTHER pids on the same machine (a `higgs download` CLI,
    // another higgs process) → false: we announce them so the fleet sees
    // machine truth, but this node has no cancel channel into that
    // process. Runs under TEST_ENV_LOCK (models_dir derives from the
    // process-global HIGGS_HOME).
    let _env = crate::TEST_ENV_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let home = tempfile::tempdir().unwrap();
    let prev = std::env::var_os("HIGGS_HOME");
    // SAFETY: serialized by TEST_ENV_LOCK; restored below.
    unsafe { std::env::set_var("HIGGS_HOME", home.path()) };
    let models = crate::download::models_dir().expect("models dir");
    // Seed a foreign-pid ledger row for one key.
    let foreign = crate::catalog::wire::DownloadLedgerEntry {
        repo: "acme/cli".into(),
        file: "cli.gguf".into(),
        pid: 1, // alive, never ours
        pid_started_at: None,
        started_at_ms: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64, // fresh: a lockless row older than the grace window is reaped
        downloaded: 3,
        total: Some(10),
        status: crate::catalog::wire::DownloadLedgerStatus::Downloading,
        ended_at_ms: None,
        path: None,
        detail: None,
    };
    std::fs::write(
        crate::catalog::ledger::ledger_path(&models),
        serde_json::to_vec(&vec![foreign]).unwrap(),
    )
    .unwrap();
    // Register a real in-process download so registry has an entry.
    let reg = crate::catalog::cancel::node_registry();
    let _guard = reg
        .register(None, "acme/reg", "reg.gguf")
        .expect("register");
    let out = announced_downloads();
    let reg_row = out
        .iter()
        .find(|d| d.repo == "acme/reg" && d.file == "reg.gguf")
        .expect("registry row announced");
    assert!(reg_row.cancellable, "registry-backed row is cancellable");
    let cli_row = out
        .iter()
        .find(|d| d.repo == "acme/cli" && d.file == "cli.gguf")
        .expect("ledger row announced");
    assert!(
        !cli_row.cancellable,
        "ledger-only row is observe-only (no cancel channel into that process)"
    );

    // SAFETY: still under the lock.
    unsafe {
        match prev {
            Some(v) => std::env::set_var("HIGGS_HOME", v),
            None => std::env::remove_var("HIGGS_HOME"),
        }
    }
}

/// r87 pin: the registry-vs-ledger union dedup is CASE-FOLDED, matching
/// the machine download-lock's key fold. A stale foreign ledger row whose
/// identity differs from a live registry entry only by ASCII case (one
/// on-disk file, one lock slot on default case-insensitive APFS) must be
/// SUPPRESSED — reverting the union compare to exact-case announces a
/// phantom frozen row beside the live one and fails this pin.
#[test]
fn a_case_variant_ledger_row_is_suppressed_by_the_live_registry_entry() {
    let _env = crate::TEST_ENV_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let home = tempfile::tempdir().unwrap();
    let prev = std::env::var_os("HIGGS_HOME");
    // SAFETY: serialized by TEST_ENV_LOCK; restored below.
    unsafe { std::env::set_var("HIGGS_HOME", home.path()) };
    let models = crate::download::models_dir().expect("models dir");
    // Foreign-pid CASE-VARIANT row of the registry key below (fresh, live
    // pid, no lock file → survives every sweep, exactly the residue shape).
    let foreign = crate::catalog::wire::DownloadLedgerEntry {
        repo: "acme/reg".into(),
        file: "REG.GGUF".into(),
        pid: 1, // alive, never ours
        pid_started_at: None,
        started_at_ms: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64,
        downloaded: 3,
        total: Some(10),
        status: crate::catalog::wire::DownloadLedgerStatus::Downloading,
        ended_at_ms: None,
        path: None,
        detail: None,
    };
    std::fs::write(
        crate::catalog::ledger::ledger_path(&models),
        serde_json::to_vec(&vec![foreign]).unwrap(),
    )
    .unwrap();
    let reg = crate::catalog::cancel::node_registry();
    let _guard = reg
        .register(None, "acme/reg", "reg.gguf")
        .expect("register");
    let out = announced_downloads();
    assert!(
        out.iter()
            .any(|d| d.repo == "acme/reg" && d.file == "reg.gguf"),
        "the live registry row is announced: {out:?}"
    );
    assert!(
        !out.iter().any(|d| d.file == "REG.GGUF"),
        "the case-variant ledger row is one folded key with the live \
         registry entry — it must be suppressed, not announced beside it: {out:?}"
    );

    // SAFETY: still under the lock.
    unsafe {
        match prev {
            Some(v) => std::env::set_var("HIGGS_HOME", v),
            None => std::env::remove_var("HIGGS_HOME"),
        }
    }
}
