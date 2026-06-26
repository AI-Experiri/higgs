
use super::*;

#[test]
fn loopback_bind_detection() {
    for b in ["127.0.0.1", "localhost", "::1", "[::1]", "127.0.0.5"] {
        assert!(is_loopback_bind(b), "{b} is loopback");
    }
    for b in ["0.0.0.0", "10.0.0.5", "192.168.1.10", "::", "203.0.113.1"] {
        assert!(!is_loopback_bind(b), "{b} is non-loopback");
    }
}

/// Drive the full standalone run flow in-process — construct + start Higgs,
/// bind an ephemeral port, serve, hit `/api/higgs/status`, then trigger the
/// injected shutdown and confirm `run_standalone` returns `Ok`. No
/// subprocess, so coverage instrumentation sees the serve path the binary's
/// `main()` otherwise only runs as a spawned (un-instrumented) child.
///
/// A non-loopback bind (`0.0.0.0`) is used so the security-warning branch is
/// also exercised; binding `0.0.0.0:0` is fine for a localhost test client.
// TEST_ENV_LOCK (a std Mutex) is intentionally held across awaits to serialize the whole
// env-overridden run; this is a #[tokio::test] (current-thread) and the only other holder
// is a sync test, so there's no deadlock/Send hazard.
#[allow(clippy::await_holding_lock)]
#[tokio::test]
async fn run_standalone_serves_and_shuts_down() {
    // Isolate HIGGS_HOME so `run_standalone` never loads the dev/CI machine's real
    // `~/.higgs/api_keys.json` — a present keystore would turn auth ON and 401 the
    // no-token `/api/higgs/status` poll below. Serialize with other env-mutating tests
    // (shared lock) and RESTORE the prior value at the end so nothing leaks. The lock +
    // `home` TempDir are held for the whole test.
    let _env = crate::TEST_ENV_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let home = tempfile::tempdir().unwrap();
    let prev_home = std::env::var_os("HIGGS_HOME");
    // SAFETY: serialized by TEST_ENV_LOCK; restored before the lock releases.
    unsafe { std::env::set_var("HIGGS_HOME", home.path()) };
    // Restore on every exit path (drops in reverse order: this runs, then `home`, then lock).
    struct RestoreHome(Option<std::ffi::OsString>);
    impl Drop for RestoreHome {
        fn drop(&mut self) {
            // SAFETY: still under TEST_ENV_LOCK.
            unsafe {
                match self.0.take() {
                    Some(v) => std::env::set_var("HIGGS_HOME", v),
                    None => std::env::remove_var("HIGGS_HOME"),
                }
            }
        }
    }
    let _restore = RestoreHome(prev_home);

    // Reserve an ephemeral port, read it back, then drop the listener so
    // run_standalone can bind the same addr. (run_standalone binds itself.)
    let probe = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = probe.local_addr().unwrap().port();
    drop(probe);

    let (tx, rx) = tokio::sync::oneshot::channel::<()>();
    let config = StandaloneConfig {
        // 0.0.0.0 exercises the non-loopback security-warning branch; the
        // client still connects via 127.0.0.1 on the same port.
        bind: "0.0.0.0".to_string(),
        port,
        higgs: HiggsConfig::default(),
        log_bus: Arc::new(LogBus::new()),
    };
    let server = tokio::spawn(async move {
        run_standalone(config, async {
            let _ = rx.await;
        })
        .await
    });

    // Poll the status endpoint until the server is accepting (bind races the
    // client). status() needs no worker, so it answers immediately.
    let url = format!("http://127.0.0.1:{port}/api/higgs/status");
    let client = reqwest::Client::new();
    let mut last_status = None;
    for _ in 0..50 {
        match client.get(&url).header("host", "127.0.0.1").send().await {
            Ok(resp) => {
                last_status = Some(resp.status());
                break;
            }
            Err(_) => tokio::time::sleep(std::time::Duration::from_millis(20)).await,
        }
    }
    assert_eq!(
        last_status,
        Some(reqwest::StatusCode::OK),
        "/api/higgs/status should answer 200 while serving"
    );

    // Trigger graceful shutdown and confirm the run flow returns Ok.
    tx.send(()).unwrap();
    let result = tokio::time::timeout(std::time::Duration::from_secs(5), server)
        .await
        .expect("server task should finish after shutdown")
        .expect("server task should not panic");
    assert!(result.is_ok(), "run_standalone returned {result:?}");
}
