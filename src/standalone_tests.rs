use super::*;

/// The resolve-based [HG058] exposure check (codex r14): decide on the RESOLVED
/// address, not the raw string. `0.0.0.0` is exposed; `127.0.0.1` and a name
/// that RESOLVES to loopback (here `LOCALHOST` — DNS is case-insensitive, but
/// the old string literal check was case-SENSITIVE) are loopback-only.
#[test]
fn bind_exposure_uses_resolved_address() {
    assert!(
        bind_resolves_non_loopback("0.0.0.0", 0),
        "0.0.0.0 is exposed"
    );
    assert!(
        !bind_resolves_non_loopback("127.0.0.1", 0),
        "127.0.0.1 loopback"
    );
    // Resolves to loopback despite not matching the string literals — the
    // false-positive the old `!is_loopback_bind` check produced.
    assert!(
        !bind_resolves_non_loopback("LOCALHOST", 0),
        "a name resolving to loopback is not exposed"
    );
}

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
/// Binds LOOPBACK (`127.0.0.1`) — the normal standalone path. (A non-loopback
/// bind with an empty keystore now REFUSES to start per [HG058]; that branch is
/// covered by `lan_bind_with_zero_keys_refuses_to_start` /
/// `lan_bind_with_non_admin_keys_refuses_to_start` below.)
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
        // Loopback: the normal standalone path. (A non-loopback bind with an
        // EMPTY keystore now REFUSES to start — [HG058], covered by
        // `lan_bind_with_zero_keys_refuses_to_start` below.)
        bind: "127.0.0.1".to_string(),
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

/// Restore an env var to its prior value (or removes it) on drop. Used by the env-mutating
/// run-flow tests below; all are serialized under TEST_ENV_LOCK so the set→restore is safe.
struct RestoreVar {
    key: &'static str,
    prev: Option<std::ffi::OsString>,
}

impl RestoreVar {
    /// Set `key` to `val`, remembering the prior value for restoration on drop.
    /// SAFETY: caller must hold TEST_ENV_LOCK for the lifetime of the returned guard.
    fn set(key: &'static str, val: &std::ffi::OsStr) -> Self {
        let prev = std::env::var_os(key);
        // SAFETY: serialized by TEST_ENV_LOCK; restored on drop.
        unsafe { std::env::set_var(key, val) };
        Self { key, prev }
    }
}

impl Drop for RestoreVar {
    fn drop(&mut self) {
        // SAFETY: still under TEST_ENV_LOCK.
        unsafe {
            match self.prev.take() {
                Some(v) => std::env::set_var(self.key, v),
                None => std::env::remove_var(self.key),
            }
        }
    }
}

/// A present, non-empty API-keys store makes `run_standalone` turn auth ON and log
/// "API-key auth ENABLED" (standalone.rs:88-91). The poll then sends the minted bearer so
/// the no-auth `/api/higgs/status` request is not 401'd, confirming the keystore was loaded
/// and applied to the live surface.
#[allow(clippy::await_holding_lock)]
#[tokio::test]
async fn run_standalone_enables_api_key_auth() {
    let _env = crate::TEST_ENV_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let home = tempfile::tempdir().unwrap();
    let _home = RestoreVar::set("HIGGS_HOME", home.path().as_os_str());

    // Seed a keystore with one admin key so the load() path counts n > 0 (auth enabled).
    let token = crate::keys::mint_token([7u8; 16]);
    let mut keys = crate::keys::ApiKeys::default();
    keys.add(&token, "test".into(), vec![crate::keys::Scope::Admin]);
    keys.save(&home.path().join("api_keys.json"))
        .expect("seed keystore");

    let probe = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = probe.local_addr().unwrap().port();
    drop(probe);

    let (tx, rx) = tokio::sync::oneshot::channel::<()>();
    let config = StandaloneConfig {
        bind: "127.0.0.1".to_string(),
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

    // With auth ON, the status poll must carry the bearer or it would 401.
    let url = format!("http://127.0.0.1:{port}/api/higgs/status");
    let client = reqwest::Client::new();
    let mut last_status = None;
    for _ in 0..50 {
        match client
            .get(&url)
            .header("host", "127.0.0.1")
            .bearer_auth(&token)
            .send()
            .await
        {
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
        "authorized /api/higgs/status should answer 200 with auth enabled"
    );

    // An UNauthorized request is rejected — proves the keystore was actually applied.
    let unauth = client
        .get(&url)
        .header("host", "127.0.0.1")
        .send()
        .await
        .expect("unauth request sends");
    assert_eq!(
        unauth.status(),
        reqwest::StatusCode::UNAUTHORIZED,
        "no-token request is 401 when auth is enabled"
    );

    tx.send(()).unwrap();
    let result = tokio::time::timeout(std::time::Duration::from_secs(5), server)
        .await
        .expect("server task should finish after shutdown")
        .expect("server task should not panic");
    assert!(result.is_ok(), "run_standalone returned {result:?}");
}

/// `HIGGS_HUB=1` makes `run_standalone` start a hub, install the fleet, and accept dials
/// (standalone.rs:97-107). `HIGGS_IROH_LOCAL=1` keeps it hermetic (relay disabled, no
/// online() wait). The hub serves a `/api/higgs/hub` status, which is only wired once
/// `set_hub` ran, so a 200 there proves the hub-start branch executed.
#[allow(clippy::await_holding_lock)]
#[tokio::test]
async fn run_standalone_hub_mode_starts_hub() {
    let _env = crate::TEST_ENV_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let home = tempfile::tempdir().unwrap();
    let _home = RestoreVar::set("HIGGS_HOME", home.path().as_os_str());
    let _hub = RestoreVar::set("HIGGS_HUB", std::ffi::OsStr::new("1"));
    let _local = RestoreVar::set("HIGGS_IROH_LOCAL", std::ffi::OsStr::new("1"));

    let probe = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = probe.local_addr().unwrap().port();
    drop(probe);

    let (tx, rx) = tokio::sync::oneshot::channel::<()>();
    let config = StandaloneConfig {
        bind: "127.0.0.1".to_string(),
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

    // The hub status endpoint reports enabled=true once the hub started.
    let url = format!("http://127.0.0.1:{port}/api/higgs/hub");
    let client = reqwest::Client::new();
    let mut body: Option<serde_json::Value> = None;
    for _ in 0..50 {
        match client.get(&url).header("host", "127.0.0.1").send().await {
            Ok(resp) if resp.status() == reqwest::StatusCode::OK => {
                body = resp.json().await.ok();
                break;
            }
            _ => tokio::time::sleep(std::time::Duration::from_millis(20)).await,
        }
    }
    let body = body.expect("hub status should answer while serving in hub mode");
    assert_eq!(
        body["enabled"], true,
        "hub mode reports an enabled hub: {body:?}"
    );
    let hub_id = body["hub_id"].as_str().unwrap_or_default();
    assert!(
        !hub_id.is_empty(),
        "hub status carries a non-empty hub_id: {body:?}"
    );

    tx.send(()).unwrap();
    let result = tokio::time::timeout(std::time::Duration::from_secs(10), server)
        .await
        .expect("server task should finish after shutdown")
        .expect("server task should not panic");
    assert!(result.is_ok(), "run_standalone returned {result:?}");
}

/// `shutdown_signal()` installs the Ctrl-C + SIGTERM handlers and parks on a `select!` until
/// one fires (standalone.rs:137-154). We race it against a short timeout: it must NOT resolve
/// on its own (no signal sent), proving it actually waits — and driving the handler-install +
/// select-setup lines. (Delivering a real SIGTERM to the shared test binary is unsafe, so the
/// terminal recv branch is left to the integration harness's real subprocess.)
#[tokio::test]
async fn shutdown_signal_parks_until_signalled() {
    let timed_out = tokio::time::timeout(
        std::time::Duration::from_millis(150),
        crate::shutdown_signal(),
    )
    .await
    .is_err();
    assert!(
        timed_out,
        "shutdown_signal must keep waiting when no signal is delivered"
    );
}

/// G4 fail-closed: a NON-LOOPBACK bind with ZERO configured API keys must
/// refuse to start with the coded [HG058] error — never serve the open
/// control + /v1 surface to the network. Fail-on-revert: restoring the old
/// warn-and-serve behavior makes `run_standalone` return Ok here.
#[tokio::test]
#[allow(clippy::await_holding_lock)] // TEST_ENV_LOCK spans the test (serializes HIGGS_HOME)
async fn lan_bind_with_zero_keys_refuses_to_start() {
    let _env = crate::TEST_ENV_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let home = tempfile::tempdir().unwrap();
    let _home = RestoreVar::set("HIGGS_HOME", home.path().as_os_str());
    // No api_keys.json in the fresh home ⇒ empty keystore.

    let config = StandaloneConfig {
        bind: "0.0.0.0".to_string(),
        port: 0, // refusal fires BEFORE any bind, so the port never matters
        higgs: HiggsConfig::default(),
        log_bus: Arc::new(LogBus::new()),
    };
    let err = run_standalone(config, async {})
        .await
        .expect_err("non-loopback bind with zero keys must refuse to start");
    assert!(
        err.to_string().contains("[HG058]"),
        "refusal carries the HG058 code: {err}"
    );
    // The remediation must mint a key that SATISFIES the follow-up [HG069] Admin
    // gate — `keys add <label>` alone defaults to chat,models and would trap the
    // operator in a second startup failure (codex r19).
    assert!(
        err.to_string().contains("keys add <label> admin"),
        "HG058 remediation points at an Admin-capable key: {err}"
    );
}

/// G4 fail-closed: a NON-LOOPBACK bind with keys present but NONE Admin-capable
/// must refuse to start with the coded [HG069] error — a keyed-but-Adminless LAN
/// server locks the operator out of the key-management API (every Admin-scoped
/// route rejected, and minting an Admin key itself needs Admin). `higgs keys add
/// <label>` defaults to `chat,models`, so this is the common LAN-bootstrap footgun.
/// Fail-on-revert: dropping the Admin-key check in `run_standalone` lets it START
/// (return Ok) with only a chat/models key, so `expect_err` fails.
#[tokio::test]
#[allow(clippy::await_holding_lock)] // TEST_ENV_LOCK spans the test (serializes HIGGS_HOME)
async fn lan_bind_with_non_admin_keys_refuses_to_start() {
    let _env = crate::TEST_ENV_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let home = tempfile::tempdir().unwrap();
    let _home = RestoreVar::set("HIGGS_HOME", home.path().as_os_str());
    // Seed a keystore with a chat/models key but NO Admin scope.
    let token = crate::keys::mint_token([7u8; 16]);
    let mut keys = crate::keys::ApiKeys::default();
    keys.add(
        &token,
        "lan".into(),
        vec![crate::keys::Scope::Chat, crate::keys::Scope::Models],
    );
    keys.save(&home.path().join("api_keys.json"))
        .expect("seed keystore");

    let config = StandaloneConfig {
        bind: "0.0.0.0".to_string(),
        port: 0, // refusal fires BEFORE any bind, so the port never matters
        higgs: HiggsConfig::default(),
        log_bus: Arc::new(LogBus::new()),
    };
    let err = run_standalone(config, async {})
        .await
        .expect_err("non-loopback bind with no Admin key must refuse to start");
    assert!(
        err.to_string().contains("[HG069]"),
        "refusal carries the HG069 code: {err}"
    );
}

/// The counterpart: with a key configured, a non-loopback bind STARTS (the
/// operator opted in and auth gates every route) — only the warning fires.
#[tokio::test]
#[allow(clippy::await_holding_lock)] // TEST_ENV_LOCK spans the test (serializes HIGGS_HOME)
async fn lan_bind_with_keys_starts() {
    let _env = crate::TEST_ENV_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let home = tempfile::tempdir().unwrap();
    let _home = RestoreVar::set("HIGGS_HOME", home.path().as_os_str());
    let token = crate::keys::mint_token([9u8; 16]);
    let mut keys = crate::keys::ApiKeys::default();
    keys.add(&token, "lan".into(), vec![crate::keys::Scope::Admin]);
    keys.save(&home.path().join("api_keys.json"))
        .expect("seed keystore");

    let probe = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = probe.local_addr().unwrap().port();
    drop(probe);

    let (tx, rx) = tokio::sync::oneshot::channel::<()>();
    let config = StandaloneConfig {
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
    // Reachable (with the bearer) — the keyed LAN bind serves.
    let url = format!("http://127.0.0.1:{port}/api/higgs/status");
    let client = reqwest::Client::new();
    let mut ok = false;
    for _ in 0..50 {
        match client
            .get(&url)
            .header("host", "127.0.0.1")
            .bearer_auth(&token)
            .send()
            .await
        {
            Ok(resp) if resp.status().is_success() => {
                ok = true;
                break;
            }
            _ => tokio::time::sleep(std::time::Duration::from_millis(20)).await,
        }
    }
    assert!(ok, "keyed non-loopback bind serves (auth-gated)");
    let _ = tx.send(());
    server.await.expect("join").expect("clean shutdown");
}

/// [HG058] fail-on-revert (codex r14): a bind that RESOLVES to loopback but is
/// not a string literal (`LOCALHOST`) with ZERO keys must START — it only
/// listens on loopback. Reverting to the raw-string `!is_loopback_bind` check
/// refuses it, so `run_standalone` returns [HG058] and this serve assertion fails.
#[tokio::test]
#[allow(clippy::await_holding_lock)] // TEST_ENV_LOCK spans the test (serializes HIGGS_HOME)
async fn loopback_alias_bind_with_zero_keys_starts() {
    let _env = crate::TEST_ENV_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let home = tempfile::tempdir().unwrap();
    let _home = RestoreVar::set("HIGGS_HOME", home.path().as_os_str());
    // No api_keys.json ⇒ empty keystore.

    let probe = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = probe.local_addr().unwrap().port();
    drop(probe);

    let (tx, rx) = tokio::sync::oneshot::channel::<()>();
    let config = StandaloneConfig {
        bind: "LOCALHOST".to_string(), // resolves to 127.0.0.1 (loopback-only)
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
    // A refusal returns Err almost immediately (before binding/serving); a
    // served instance runs until shutdown. Checking task liveness (not an HTTP
    // round-trip) sidesteps LOCALHOST's IPv4/IPv6 bind ambiguity. On revert to
    // the raw-string check, run_standalone refuses → the task finishes early →
    // this assertion fails.
    tokio::time::sleep(std::time::Duration::from_millis(400)).await;
    assert!(
        !server.is_finished(),
        "loopback-resolving alias bind with zero keys was NOT refused (still serving)"
    );
    let _ = tx.send(());
    server.await.expect("join").expect("clean shutdown");
}
