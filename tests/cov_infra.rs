//! Integration coverage for higgs's INFRASTRUCTURE modules — the pieces that sit
//! below the facade/serve layers and are reachable through the crate's public API
//! without a running llama.cpp worker:
//!
//! - `download.rs` — path/guard validation, the atomic-write `download`/`download_dual`
//!   engine (driven with in-test [`Fetcher`] fakes), and the `reqwest` `HttpFetcher`
//!   against a local HTTP server.
//! - `hub.rs` — the HuggingFace-Hub `fetch_bytes` primary/fallback path and its
//!   `HGxxx` error classification (dead endpoint, bad repo id, a live 200/429 server).
//! - `config.rs` — `InstanceConfig` load/save round-trips, the lenient tag-less
//!   `LoadParams` migration, saved-hub promotion, and the load/save error branches.
//! - `keys.rs` — `ApiKeys` mint/hidden/visible/authorize/touch semantics, digest-only
//!   persistence, and the `keys` CLI subcommands.
//! - `auth.rs` — the `Allowlist` store (+ its corruption/IO/rename-fail branches) and
//!   one-time `PairingTokens` (expiry / replay).
//! - `log_bus.rs` — `LogSource` parsing/filtering, per-source rings + eviction, and the
//!   `HiggsLogLayer` tracing capture (redaction + verbose gating).
//! - `delta_queue.rs` — merging, the `CAP_BYTES` overflow guard, drop semantics.
//! - `rpc.rs` — NDJSON frame encode/decode + `method_not_found`.
//! - `system.rs` — `SystemInfo::gather` composition.
//!
//! Every test runs regardless of the tiny-GGUF fixture (no worker is spawned), so none
//! skip. A single process-global async mutex ([`serial`]) serializes ALL tests: several
//! mutate process-global env (`HIGGS_HF_ENDPOINT`/`HIGGS_HOME`) while others read it
//! (via `reqwest`/`TempDir`), and concurrent `setenv`/`getenv` is a data race.

use std::sync::{Arc, OnceLock};

use higgs::HiggsError;
use tempfile::TempDir;
use tokio::sync::{Mutex, OwnedMutexGuard};

// ── Serialization + env helpers ──────────────────────────────────────────────

/// Process-global lock every test holds for its whole body. Serializes the
/// env-mutating tests against the env-reading (and `TempDir`-creating) ones so no
/// two threads touch the process environment concurrently.
async fn serial() -> OwnedMutexGuard<()> {
    static LOCK: OnceLock<Arc<Mutex<()>>> = OnceLock::new();
    LOCK.get_or_init(|| Arc::new(Mutex::new(())))
        .clone()
        .lock_owned()
        .await
}

/// Sets an env var for the current test and restores the prior value on drop.
/// Only ever constructed while holding [`serial`], so the process env is never
/// mutated concurrently with another thread's read.
struct EnvVarGuard {
    key: &'static str,
    prev: Option<std::ffi::OsString>,
}

impl EnvVarGuard {
    fn set(key: &'static str, val: &str) -> Self {
        let prev = std::env::var_os(key);
        // SAFETY: serialized by the `serial()` lock the caller holds — no other
        // thread reads or writes the process env for the duration.
        unsafe { std::env::set_var(key, val) };
        Self { key, prev }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        // SAFETY: still under the caller's `serial()` lock.
        unsafe {
            match &self.prev {
                Some(v) => std::env::set_var(self.key, v),
                None => std::env::remove_var(self.key),
            }
        }
    }
}

/// Spawn a local HTTP server on an ephemeral loopback port that answers EVERY path
/// with `status` + `body`. Returns its base URL (`http://127.0.0.1:PORT`). The
/// `HuggingFace`-hub client and the `reqwest` fallback both hit it via
/// `HIGGS_HF_ENDPOINT`, so no network is touched.
async fn serve_fixed(status: axum::http::StatusCode, body: &'static str) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let app = axum::Router::new().fallback(move || async move { (status, body) });
    tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    base
}

// ── download.rs: in-test Fetcher fakes ───────────────────────────────────────

use higgs::download::{dest_path, download_dual, Fetcher, HttpFetcher, PullTarget};

/// A [`Fetcher`] that emits `body` in one chunk and reports progress — the "network
/// succeeded" fake, so the atomic-write path in `download` runs end to end.
struct OkFetcher {
    body: Vec<u8>,
}

impl Fetcher for OkFetcher {
    async fn fetch(
        &self,
        _target: &PullTarget,
        on_chunk: &mut (dyn FnMut(&[u8]) + Send),
        progress: &mut (dyn FnMut(u64, Option<u64>) + Send),
    ) -> Result<(), HiggsError> {
        on_chunk(&self.body);
        progress(self.body.len() as u64, Some(self.body.len() as u64));
        Ok(())
    }
}

/// A [`Fetcher`] that always fails with a classified transport error — the "network
/// failed" fake, so the fetcher-error / fallback branches run.
struct ErrFetcher;

impl Fetcher for ErrFetcher {
    async fn fetch(
        &self,
        target: &PullTarget,
        _on_chunk: &mut (dyn FnMut(&[u8]) + Send),
        _progress: &mut (dyn FnMut(u64, Option<u64>) + Send),
    ) -> Result<(), HiggsError> {
        Err(HiggsError::HubTransport {
            repo: target.repo.clone(),
            detail: "fake transport failure".into(),
        })
    }
}

#[tokio::test]
async fn download_dest_path_and_pulltarget() {
    let _g = serial().await;
    let root = TempDir::new().unwrap();

    // Constructor defaults revision to "main".
    let t = PullTarget::new("acme/model", "weights.gguf");
    assert_eq!(t.revision, "main");

    // Valid `<org>/<model>/<file>.gguf` resolves under the models root.
    let p = dest_path(root.path(), "acme/model", "weights.gguf").unwrap();
    assert!(p.ends_with("acme/model/weights.gguf"), "layout: {p:?}");

    // A repo that is not exactly two safe segments is refused.
    assert!(dest_path(root.path(), "single", "f.gguf").is_err());
    assert!(dest_path(root.path(), "a/b/c", "f.gguf").is_err());
    assert!(
        dest_path(root.path(), "org/mo del", "f.gguf").is_err(),
        "reserved space char rejected"
    );
    assert!(dest_path(root.path(), "../etc/model", "f.gguf").is_err());

    // A non-gguf, subdir'd, or reserved-char file is refused.
    assert!(dest_path(root.path(), "acme/model", "weights.bin").is_err());
    assert!(dest_path(root.path(), "acme/model", "sub/weights.gguf").is_err());
    assert!(dest_path(root.path(), "acme/model", "..gguf..").is_err());
}

#[tokio::test]
async fn download_single_fetcher_paths() {
    let _g = serial().await;

    // Happy path: bytes stream to a `.part` temp and rename onto the final path.
    let root = TempDir::new().unwrap();
    let t = PullTarget::new("acme/model", "weights.gguf");
    let mut prog = |_dl: u64, _total: Option<u64>| {};
    let ok = OkFetcher {
        body: b"GGUF".to_vec(),
    };
    let path = download_dual(&t, root.path(), &ok, &ok, &mut prog)
        .await
        .expect("download succeeds");
    assert!(path.exists(), "final file written: {path:?}");
    assert_eq!(std::fs::read(&path).unwrap(), b"GGUF");

    // Bad revision short-circuits with a wire-validation error (HG025), never touching
    // the fetcher.
    let bad = PullTarget {
        repo: "acme/model".into(),
        file: "weights.gguf".into(),
        revision: "bad rev".into(),
    };
    let empty = OkFetcher { body: vec![] };
    let err = download_dual(&bad, root.path(), &empty, &empty, &mut prog)
        .await
        .unwrap_err();
    assert!(
        matches!(err, HiggsError::DownloadFailed { .. }),
        "invalid revision → DownloadFailed, got {err:?}"
    );

    // Rename failure: a directory already occupies the destination path, so the atomic
    // rename fails and the local filesystem error surfaces as HubFileWrite (HG034).
    let root2 = TempDir::new().unwrap();
    let occupied = root2.path().join("acme/model/weights.gguf");
    std::fs::create_dir_all(&occupied).unwrap();
    let onefetch = OkFetcher {
        body: b"x".to_vec(),
    };
    let err = download_dual(&t, root2.path(), &onefetch, &onefetch, &mut prog)
        .await
        .unwrap_err();
    assert!(
        matches!(err, HiggsError::HubFileWrite { .. }),
        "rename onto a directory → HubFileWrite, got {err:?}"
    );
}

#[tokio::test]
async fn download_dual_paths() {
    let _g = serial().await;

    // A LOCAL (deterministic) failure in the primary attempt is returned verbatim —
    // the fallback is NOT tried (it would fail identically).
    let root = TempDir::new().unwrap();
    let bad = PullTarget {
        repo: "acme/model".into(),
        file: "weights.gguf".into(),
        revision: "bad rev".into(),
    };
    let mut prog = |_dl: u64, _total: Option<u64>| {};
    let err = download_dual(
        &bad,
        root.path(),
        &ErrFetcher,
        &OkFetcher { body: vec![] },
        &mut prog,
    )
    .await
    .unwrap_err();
    assert!(
        matches!(err, HiggsError::DownloadFailed { .. }),
        "local primary error returned verbatim, got {err:?}"
    );

    // A FETCHER failure in the primary triggers the fallback; if the fallback succeeds,
    // the file lands.
    let good = PullTarget::new("acme/model", "weights.gguf");
    let path = download_dual(
        &good,
        root.path(),
        &ErrFetcher,
        &OkFetcher {
            body: b"fallback-data".to_vec(),
        },
        &mut prog,
    )
    .await
    .expect("fallback succeeds");
    assert_eq!(std::fs::read(&path).unwrap(), b"fallback-data");

    // BOTH transports exhausted → HubFetchExhausted (HG036) carrying both diagnoses.
    let root2 = TempDir::new().unwrap();
    let err = download_dual(&good, root2.path(), &ErrFetcher, &ErrFetcher, &mut prog)
        .await
        .unwrap_err();
    match err {
        HiggsError::HubFetchExhausted {
            primary, fallback, ..
        } => {
            assert!(primary.contains("HG033"), "primary diagnosis: {primary}");
            assert!(fallback.contains("HG033"), "fallback diagnosis: {fallback}");
        }
        other => panic!("both-exhausted → HubFetchExhausted, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn download_httpfetcher_live_server() {
    let _g = serial().await;

    // A 200 with a body: HttpFetcher streams it to disk and reports the content length.
    {
        let base = serve_fixed(axum::http::StatusCode::OK, "GGUF-bytes").await;
        let _env = EnvVarGuard::set("HIGGS_HF_ENDPOINT", &base);
        let root = TempDir::new().unwrap();
        let t = PullTarget::new("acme/model", "weights.gguf");
        // `Send` progress sink (the fetcher trait requires it): an Arc'd mutex the
        // closure moves in, recording the last `(downloaded, total)` report.
        let seen = Arc::new(std::sync::Mutex::new((0u64, Option::<u64>::None)));
        let seen2 = seen.clone();
        let mut prog = move |d: u64, tot: Option<u64>| {
            *seen2.lock().unwrap() = (d, tot);
        };
        let path = download_dual(&t, root.path(), &HttpFetcher, &HttpFetcher, &mut prog)
            .await
            .expect("HttpFetcher 200 succeeds");
        assert_eq!(std::fs::read(&path).unwrap(), b"GGUF-bytes");
        let (dl, total) = *seen.lock().unwrap();
        assert_eq!(dl, 10, "streamed the whole body");
        assert_eq!(total, Some(10), "content-length reported to progress");
    }

    // A 429: HttpFetcher classifies the status as HG031 (rate-limited). download_dual
    // retries the fallback on a fetcher failure, and here BOTH fetchers are HttpFetcher
    // against the same 429 server, so the result is the terminal HubFetchExhausted (HG036)
    // carrying two HG031 diagnoses — not a bare HubRateLimited.
    {
        let base = serve_fixed(axum::http::StatusCode::TOO_MANY_REQUESTS, "slow down").await;
        let _env = EnvVarGuard::set("HIGGS_HF_ENDPOINT", &base);
        let root = TempDir::new().unwrap();
        let t = PullTarget::new("acme/model", "weights.gguf");
        let mut prog = |_d: u64, _t: Option<u64>| {};
        let err = download_dual(&t, root.path(), &HttpFetcher, &HttpFetcher, &mut prog)
            .await
            .unwrap_err();
        match err {
            HiggsError::HubFetchExhausted {
                primary, fallback, ..
            } => {
                assert!(primary.contains("HG031"), "primary is HG031: {primary}");
                assert!(fallback.contains("HG031"), "fallback is HG031: {fallback}");
            }
            other => panic!("HTTP 429 on both paths → HubFetchExhausted, got {other:?}"),
        }
    }
}

// ── hub.rs ───────────────────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn hub_fetch_bytes_error_classification() {
    let _g = serial().await;

    // A dead endpoint fails on BOTH the hub-client primary and the reqwest fallback →
    // HubFetchExhausted (HG036).
    {
        let _env = EnvVarGuard::set("HIGGS_HF_ENDPOINT", "http://127.0.0.1:1");
        let err = higgs::hub::fetch_bytes("acme/model", "README.md")
            .await
            .unwrap_err();
        assert!(
            matches!(err, HiggsError::HubFetchExhausted { .. }),
            "dead endpoint → both transports exhausted, got {err:?}"
        );
    }

    // A malformed repo id (no `<org>/<model>`) fails the primary's split immediately;
    // the fallback then also fails against the dead endpoint → HubFetchExhausted.
    {
        let _env = EnvVarGuard::set("HIGGS_HF_ENDPOINT", "http://127.0.0.1:1");
        let err = higgs::hub::fetch_bytes("norepo", "README.md")
            .await
            .unwrap_err();
        assert!(
            matches!(err, HiggsError::HubFetchExhausted { .. }),
            "bad repo id → exhausted, got {err:?}"
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn hub_fetch_bytes_live_server_returns_bytes() {
    let _g = serial().await;

    // A live 200 server: whichever path succeeds (primary hub client OR reqwest
    // fallback), fetch_bytes returns the served bytes.
    let base = serve_fixed(axum::http::StatusCode::OK, "hello-card").await;
    let _env = EnvVarGuard::set("HIGGS_HF_ENDPOINT", &base);
    let bytes = higgs::hub::fetch_bytes("acme/model", "README.md")
        .await
        .expect("live 200 yields bytes");
    assert_eq!(bytes, b"hello-card");
}

// ── config.rs ─────────────────────────────────────────────────────────────────

use higgs::config::{friendly_name, InstanceConfig, Role, SavedHub};

#[tokio::test]
async fn config_roundtrip_lenient_and_friendly_name() {
    let _g = serial().await;
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("config.json");

    // A missing file is the default (empty) config.
    assert!(InstanceConfig::load(&path).unwrap().name.is_empty());

    // Save with a name, a load record, and two saved hubs; reload round-trips it all.
    let mut cfg = InstanceConfig {
        name: "node-a1b2c3d4(studio)".into(),
        ..Default::default()
    };
    cfg.record_load(
        "acme/model",
        higgs::HiggsConfig::default().default_load,
        12_345,
    );
    cfg.remember_hub(SavedHub {
        hub_id: "hub-1".into(),
        ticket: "t1".into(),
        label: "Alpha".into(),
        last_used_ms: 5,
    });
    cfg.remember_hub(SavedHub {
        hub_id: "hub-2".into(),
        ticket: "t2".into(),
        label: "Beta".into(),
        last_used_ms: 9,
    });
    cfg.save(&path).unwrap();

    let re = InstanceConfig::load(&path).unwrap();
    assert_eq!(re.name, "node-a1b2c3d4(studio)");
    assert_eq!(
        re.default_hub.as_deref(),
        Some("hub-2"),
        "latest hub is default"
    );
    let rec = re.model_record("acme/model").expect("record present");
    assert_eq!(rec.last_loaded_ms, 12_345);
    assert!(rec.load.is_some());
    assert!(re.model_record("nope").is_none());

    // Removing the DEFAULT hub promotes the most-recently-used remaining one.
    let mut re2 = re.clone();
    assert!(re2.remove_hub("hub-2"), "removed");
    assert_eq!(
        re2.default_hub.as_deref(),
        Some("hub-1"),
        "promoted survivor"
    );
    assert!(
        !re2.remove_hub("ghost"),
        "removing an unknown hub is a no-op"
    );

    // A pre-umbrella TAG-LESS flat LoadParams record still loads (lenient migration):
    // strip the `engine` tag off a real serialized LoadParams and confirm it parses back.
    let tagged = serde_json::to_value(higgs::HiggsConfig::default().default_load).unwrap();
    assert!(
        tagged.get("engine").is_some(),
        "LoadParams is engine-tagged"
    );
    let mut flat = tagged.clone();
    flat.as_object_mut().unwrap().remove("engine");
    let flat_json = serde_json::json!({
        "name": "node-x(y)",
        "models": { "acme/flat": { "load": flat, "last_loaded_ms": 42 } }
    });
    let flat_path = dir.path().join("flat.json");
    std::fs::write(&flat_path, serde_json::to_vec(&flat_json).unwrap()).unwrap();
    let flat_cfg = InstanceConfig::load(&flat_path).unwrap();
    let flat_rec = flat_cfg.model_record("acme/flat").expect("flat record");
    assert!(
        flat_rec.load.is_some(),
        "tag-less flat LoadParams parsed via the lenient fallback"
    );

    // friendly_name: with and without a hostname.
    assert_eq!(
        friendly_name(Role::Node, "abcdef1234567", "studio"),
        "node-abcdef12(studio)"
    );
    assert_eq!(
        friendly_name(Role::Hub, "abcdef1234567", ""),
        "hub-abcdef12"
    );
}

#[tokio::test]
async fn config_error_paths() {
    let _g = serial().await;
    let dir = TempDir::new().unwrap();

    // Present-but-unparseable JSON → corruption (InvalidData), NOT a silent reset.
    let corrupt = dir.path().join("corrupt.json");
    std::fs::write(&corrupt, b"{ this is not json").unwrap();
    let err = InstanceConfig::load(&corrupt).unwrap_err();
    assert_eq!(
        err.kind(),
        std::io::ErrorKind::InvalidData,
        "corruption: {err}"
    );

    // A real read I/O error (the path is a directory) surfaces, NOT as NotFound.
    let asdir = dir.path().join("cfgdir");
    std::fs::create_dir(&asdir).unwrap();
    let err = InstanceConfig::load(&asdir).unwrap_err();
    assert_ne!(err.kind(), std::io::ErrorKind::NotFound, "io error: {err}");

    // Save rename failure: the destination path is an existing directory, so the
    // atomic rename can't replace it.
    let occupied = dir.path().join("occupied");
    std::fs::create_dir(&occupied).unwrap();
    assert!(
        InstanceConfig::default().save(&occupied).is_err(),
        "saving onto a directory fails at rename"
    );

    // Save temp-create failure: the parent dir is read-only, so the `.part` temp can't
    // be created. Skipped under root (root ignores the mode bits).
    if unsafe { libc::geteuid() } != 0 {
        use std::os::unix::fs::PermissionsExt;
        let ro = dir.path().join("ro");
        std::fs::create_dir(&ro).unwrap();
        let mut perms = std::fs::metadata(&ro).unwrap().permissions();
        perms.set_mode(0o500);
        std::fs::set_permissions(&ro, perms).unwrap();
        assert!(
            InstanceConfig::default()
                .save(&ro.join("config.json"))
                .is_err(),
            "save into a read-only dir fails creating the temp"
        );
        // Restore write so TempDir cleanup can remove it.
        let mut perms = std::fs::metadata(&ro).unwrap().permissions();
        perms.set_mode(0o700);
        let _ = std::fs::set_permissions(&ro, perms);
    }
}

// ── keys.rs ───────────────────────────────────────────────────────────────────

use higgs::keys::{hash_token, mint_token, run_keys, ApiKeys, Scope};

#[tokio::test]
async fn keys_apikeys_operations() {
    let _g = serial().await;
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("api_keys.json");

    // A missing store is empty (auth OFF).
    let mut ks = ApiKeys::load(&path).unwrap();
    assert!(ks.is_empty());

    // Add one user key and one INTERNAL (hidden) key.
    let user_token = mint_token([7u8; 16]);
    ks.add(&user_token, "admin".into(), vec![Scope::Admin]);
    ks.add_internal(
        "secret-internal-token",
        "embedder (internal)".into(),
        vec![Scope::Chat],
    );

    // `iter()` sees ALL keys; `visible()` hides the internal one.
    assert_eq!(ks.iter().count(), 2);
    assert_eq!(ks.visible().count(), 1);

    // Debug is redacted: the label + a digest PREFIX, never the plaintext or full digest.
    let admin = ks.iter().find(|k| k.label == "admin").unwrap();
    let dbg = format!("{admin:?}");
    assert!(dbg.contains("ApiKey") && dbg.contains("admin") && dbg.contains('…'));
    assert!(!dbg.contains(&user_token), "plaintext not in Debug");

    // Admin authorizes any scope; the internal key authorizes its own scope; a wrong
    // token authorizes nothing.
    assert!(ks.authorizes(&user_token, Scope::Models));
    assert!(ks.authorizes("secret-internal-token", Scope::Chat));
    assert!(!ks.authorizes("hgk_wrong", Scope::Chat));

    // `authorizing_sha` returns the matched digest.
    let sha = ks.authorizing_sha(&user_token, Scope::Admin).unwrap();
    assert_eq!(sha, hash_token(&user_token));

    // `touch` is monotonic: advances forward, no-ops on a stale stamp or unknown digest.
    assert!(ks.touch(&sha, 1_000));
    assert!(!ks.touch(&sha, 500), "stale stamp is a no-op");
    assert!(!ks.touch("deadbeef", 2_000), "unknown digest is a no-op");

    // Persist: only the user key is written (hidden internal is never persisted, and no
    // plaintext ever hits disk).
    ks.save(&path).unwrap();
    let disk = std::fs::read_to_string(&path).unwrap();
    assert!(disk.contains("admin"), "user key persisted");
    assert!(
        !disk.contains("embedder (internal)"),
        "hidden key not persisted"
    );
    assert!(!disk.contains("secret-internal-token") && !disk.contains(&user_token));

    // Reload: the digest-only user key is back; nothing hidden round-tripped from disk.
    let reloaded = ApiKeys::load(&path).unwrap();
    assert_eq!(reloaded.iter().count(), 1);
    assert_eq!(reloaded.visible().count(), 1);

    // A non-NotFound load error (the path is a directory) surfaces.
    let asdir = dir.path().join("kdir");
    std::fs::create_dir(&asdir).unwrap();
    let err = ApiKeys::load(&asdir).unwrap_err();
    assert_ne!(err.kind(), std::io::ErrorKind::NotFound);
}

#[tokio::test]
async fn keys_run_keys_cli() {
    let _g = serial().await;
    let home = TempDir::new().unwrap();
    let _env = EnvVarGuard::set("HIGGS_HOME", home.path().to_str().unwrap());

    // list on an empty store is Ok ("no API keys configured — OPEN").
    run_keys(&["list".into()]).expect("list empty ok");

    // add with an empty scope list is rejected.
    assert!(
        run_keys(&["add".into(), "ci".into(), String::new()]).is_err(),
        "empty scopes"
    );
    // add with no label is rejected.
    assert!(run_keys(&["add".into()]).is_err(), "missing label");
    // remove with no label is rejected.
    assert!(
        run_keys(&["remove".into()]).is_err(),
        "missing remove label"
    );
    // an unknown subcommand is rejected.
    assert!(
        run_keys(&["frobnicate".into()]).is_err(),
        "unknown subcommand"
    );

    // A real add → list → remove round-trip through the store.
    run_keys(&["add".into(), "ci".into(), "chat,models".into()]).expect("add ok");
    run_keys(&["list".into()]).expect("list non-empty ok");
    run_keys(&["remove".into(), "ci".into()]).expect("remove ok");
}

// ── auth.rs ───────────────────────────────────────────────────────────────────

use higgs::auth::{Allowlist, PairingTokens, TokenError};

#[tokio::test]
async fn auth_allowlist_and_pairing_tokens() {
    let _g = serial().await;
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("pairings.json");

    // Missing file → empty allowlist; mutate + persist + reload round-trips.
    let mut al = Allowlist::load(&path).unwrap();
    assert!(al.is_empty());
    al.add("nodeA".into(), Some("Alpha".into())).unwrap();
    al.add("nodeB".into(), None).unwrap();
    assert_eq!(al.len(), 2, "two paired nodes");
    assert!(al.contains("nodeA") && !al.is_empty());
    assert_eq!(al.label("nodeA").as_deref(), Some("Alpha"));
    assert_eq!(al.ids(), vec!["nodeA".to_string(), "nodeB".to_string()]);
    assert_eq!(al.labels().get("nodeA"), Some(&Some("Alpha".to_string())));
    assert!(
        al.relabel("nodeA", Some("Alpha2".into())).unwrap(),
        "relabel present"
    );
    assert!(
        !al.relabel("ghost", None).unwrap(),
        "relabel unknown is a no-op"
    );
    al.remove("nodeB").unwrap();
    assert_eq!(al.path(), path);

    let al2 = Allowlist::load(&path).unwrap();
    assert!(al2.contains("nodeA") && !al2.contains("nodeB"));
    assert_eq!(al2.label("nodeA").as_deref(), Some("Alpha2"));

    // Corruption → InvalidData; a directory read → non-NotFound io error.
    // (`Allowlist` isn't `Debug`, so pull the error out via `.err()` — no `unwrap_err`.)
    let corrupt = dir.path().join("bad.json");
    std::fs::write(&corrupt, b"{ nope").unwrap();
    assert_eq!(
        Allowlist::load(&corrupt)
            .err()
            .expect("corruption errors")
            .kind(),
        std::io::ErrorKind::InvalidData
    );
    let asdir = dir.path().join("adir");
    std::fs::create_dir(&asdir).unwrap();
    assert_ne!(
        Allowlist::load(&asdir)
            .err()
            .expect("dir read errors")
            .kind(),
        std::io::ErrorKind::NotFound
    );

    // Save rename failure: the store path becomes a directory after load, so the
    // persisting `add` fails at rename and rolls back.
    let occ = dir.path().join("occ");
    let mut al3 = Allowlist::load(&occ).unwrap();
    std::fs::create_dir(&occ).unwrap();
    assert!(
        al3.add("x".into(), None).is_err(),
        "save onto a directory fails"
    );

    // Pairing tokens: fresh admits, expired + replayed reject.
    let mut pt = PairingTokens::new();
    let now = 1_000u64;
    let tok = pt.mint(now, 10_000);
    assert!(tok.starts_with("htk_"));
    assert!(pt.validate(&tok, now).is_ok());
    assert_eq!(pt.validate(&tok, 20_000), Err(TokenError::Expired));
    assert_eq!(
        pt.validate("htk_unknown", now),
        Err(TokenError::UnknownOrUsed)
    );
    // validate_and_burn admits once, then the replay is unknown/used.
    assert!(pt.validate_and_burn(&tok, now).is_ok());
    assert_eq!(pt.validate(&tok, now), Err(TokenError::UnknownOrUsed));
}

// ── log_bus.rs ────────────────────────────────────────────────────────────────

use higgs::log_bus::{log_filter, HiggsLogLayer, LogBus, LogSource};
use higgs::node::node_id::NodeId;
use higgs::node::worker_id::WorkerId;

#[tokio::test]
async fn log_bus_parse_and_matches() {
    let _g = serial().await;

    assert_eq!(LogSource::parse("serve"), Some(LogSource::Serve));
    assert_eq!(LogSource::parse("worker"), Some(LogSource::Worker));
    match LogSource::parse("worker:3") {
        Some(LogSource::LocalWorker { worker }) => assert_eq!(worker.0, 3),
        other => panic!("worker:3 → LocalWorker, got {other:?}"),
    }
    match LogSource::parse("node:1:2") {
        Some(LogSource::RemoteWorker { node, worker }) => {
            assert_eq!(node.0, 1);
            assert_eq!(worker.0, 2);
        }
        other => panic!("node:1:2 → RemoteWorker, got {other:?}"),
    }
    // `node:<id>` (no worker part) = that node's own DAEMON log (M_NODE_LOGS).
    match LogSource::parse("node:1") {
        Some(LogSource::RemoteNode { node }) => assert_eq!(node.0, 1),
        other => panic!("node:1 → RemoteNode, got {other:?}"),
    }
    // Unknown / malformed selectors → None (all sources).
    assert_eq!(LogSource::parse("bogus"), None);
    assert_eq!(LogSource::parse("worker:nan"), None);
    assert_eq!(LogSource::parse("node:x"), None);
    assert_eq!(LogSource::parse("node:x:2"), None);

    // The `worker` filter is the UNION selector: it matches keyed local workers too.
    assert!(LogSource::Serve.matches_filter(LogSource::Serve));
    assert!(LogSource::LocalWorker {
        worker: WorkerId(5)
    }
    .matches_filter(LogSource::Worker));
    assert!(!LogSource::Serve.matches_filter(LogSource::Worker));
}

#[tokio::test]
async fn log_bus_rings_snapshot_and_eviction() {
    let _g = serial().await;
    let bus = LogBus::default();

    bus.push(LogSource::Serve, "s1".into());
    bus.push(LogSource::Worker, "w1".into());
    bus.push(
        LogSource::LocalWorker {
            worker: WorkerId(1),
        },
        "lw1".into(),
    );
    bus.push(
        LogSource::RemoteWorker {
            node: NodeId(2),
            worker: WorkerId(3),
        },
        "rw1".into(),
    );

    // Per-source snapshots.
    assert_eq!(
        bus.snapshot(10, Some(LogSource::Serve)),
        vec!["s1".to_string()]
    );
    let worker_union = bus.snapshot(10, Some(LogSource::Worker));
    assert!(worker_union.contains(&"w1".to_string()) && worker_union.contains(&"lw1".to_string()));
    assert_eq!(
        bus.snapshot(
            10,
            Some(LogSource::LocalWorker {
                worker: WorkerId(1)
            })
        ),
        vec!["lw1".to_string()]
    );
    assert_eq!(
        bus.snapshot(
            10,
            Some(LogSource::RemoteWorker {
                node: NodeId(2),
                worker: WorkerId(3),
            })
        ),
        vec!["rw1".to_string()]
    );
    // Unfiltered snapshot re-interleaves every ring by arrival (seq) order.
    assert_eq!(bus.snapshot(10, None), vec!["s1", "w1", "lw1", "rw1"]);

    // Eviction: a fresh ring overflowed past RING_CAP keeps only the newest 2000 lines.
    let ev = LogBus::new();
    for i in 0..2_050 {
        ev.push(LogSource::Serve, format!("l{i}"));
    }
    let serve = ev.snapshot(10_000, Some(LogSource::Serve));
    assert_eq!(serve.len(), 2_000, "ring capped at RING_CAP");
    assert_eq!(serve.last().unwrap(), "l2049", "newest kept");
    assert!(!serve.contains(&"l0".to_string()), "oldest evicted");

    // Reclaiming a worker's ring drops its lines.
    bus.evict_local(WorkerId(1));
    assert!(bus
        .snapshot(
            10,
            Some(LogSource::LocalWorker {
                worker: WorkerId(1)
            })
        )
        .is_empty());
    bus.evict_node(NodeId(2));
    assert!(bus
        .snapshot(
            10,
            Some(LogSource::RemoteWorker {
                node: NodeId(2),
                worker: WorkerId(3),
            })
        )
        .is_empty());
}

#[tokio::test]
async fn log_bus_layer_captures_higgs_events() {
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::Layer;

    let _g = serial().await;

    // With verbose OFF, a higgs DEBUG event is admitted to the layer (via `log_filter`)
    // but dropped by the level gate — nothing lands in the ring.
    let bus_a = Arc::new(LogBus::new());
    {
        let sub = tracing_subscriber::registry()
            .with(HiggsLogLayer::new(bus_a.clone()).with_filter(log_filter()));
        tracing::subscriber::with_default(sub, || {
            tracing::debug!(target: "higgs::probe", "debug-dropped-when-not-verbose");
        });
    }
    assert!(
        bus_a.snapshot(50, Some(LogSource::Serve)).is_empty(),
        "a higgs DEBUG event is dropped while verbose is off"
    );

    // Unfiltered layer: a non-higgs target is ignored, an empty-message higgs event is
    // dropped, and a real higgs WARN is captured WITH its typed `error` field and (in
    // show-fields mode) its extra fields.
    let bus_b = Arc::new(LogBus::new());
    bus_b.set_verbose(true);
    bus_b.set_show_fields(true);
    {
        let sub = tracing_subscriber::registry().with(HiggsLogLayer::new(bus_b.clone()));
        tracing::subscriber::with_default(sub, || {
            tracing::info!(target: "other::mod", "ignored-non-higgs");
            tracing::info!(target: "higgs::probe", key = "v");
            tracing::warn!(target: "higgs::probe", error = "boom", extra = "xyz", "hello {}", 7);
        });
    }
    let lines = bus_b.snapshot(50, Some(LogSource::Serve));
    assert!(
        lines
            .iter()
            .any(|l| l.contains("hello 7") && l.contains("boom") && l.contains("extra=xyz")),
        "higgs WARN captured with error + extra fields: {lines:?}"
    );
    assert!(
        !lines
            .iter()
            .any(|l| l.contains("ignored-non-higgs") || l.contains("key=v")),
        "non-higgs + empty-message events are not captured: {lines:?}"
    );
}

// ── delta_queue.rs ───────────────────────────────────────────────────────────

use higgs::delta_queue::{delta_channel, CAP_BYTES};
use higgs::{ChatDelta, ChatDeltaKind};

#[tokio::test]
async fn delta_queue_merge_overflow_and_drop() {
    let _g = serial().await;

    // Consecutive same-kind text deltas MERGE into one entry; tool-calls never merge;
    // a different kind starts a new entry.
    let (tx, mut rx) = delta_channel();
    tx.send(ChatDelta {
        kind: ChatDeltaKind::Content,
        text: "ab".into(),
    });
    tx.send(ChatDelta {
        kind: ChatDeltaKind::Content,
        text: "cd".into(),
    });
    tx.send(ChatDelta {
        kind: ChatDeltaKind::Reasoning,
        text: "think".into(),
    });
    tx.send(ChatDelta {
        kind: ChatDeltaKind::ToolCall,
        text: "{a}".into(),
    });
    tx.send(ChatDelta {
        kind: ChatDeltaKind::ToolCall,
        text: "{b}".into(),
    });

    // Debug renders the queued-entry accounting.
    assert!(format!("{rx:?}").contains("entries"));

    let d0 = rx.recv().await.unwrap();
    assert_eq!(d0.kind, ChatDeltaKind::Content);
    assert_eq!(d0.text, "abcd", "same-kind content merged in order");
    assert_eq!(rx.try_recv().unwrap().text, "think");
    assert_eq!(rx.try_recv().unwrap().text, "{a}");
    assert_eq!(
        rx.try_recv().unwrap().text,
        "{b}",
        "tool-calls stay separate"
    );
    assert!(rx.try_recv().is_none());
    drop(tx);
    assert!(
        rx.recv().await.is_none(),
        "closed-and-drained ends the stream"
    );

    // The CAP_BYTES guard trips loudly: the pending backlog is reported, the stream
    // aborts, and later sends are no-ops.
    let (tx, mut rx) = delta_channel();
    tx.send(ChatDelta {
        kind: ChatDeltaKind::Content,
        text: "0123456789".into(),
    });
    tx.send(ChatDelta {
        kind: ChatDeltaKind::Content,
        text: "x".repeat(CAP_BYTES + 1),
    });
    assert!(
        rx.recv().await.is_none(),
        "overflow ends the stream as None"
    );
    assert!(rx.overflowed(), "overflow flag set");
    assert_eq!(
        rx.buffered_bytes(),
        10,
        "the pre-overflow backlog is reported"
    );
    tx.send(ChatDelta {
        kind: ChatDeltaKind::Content,
        text: "later".into(),
    });
    assert!(rx.overflowed(), "post-overflow sends are no-ops");
}

// ── rpc.rs ────────────────────────────────────────────────────────────────────

use higgs::rpc::{
    decode, encode, method_not_found, RpcFrame, RpcNotification, RpcRequest, RpcResponse,
};

#[tokio::test]
async fn rpc_encode_decode_and_method_not_found() {
    let _g = serial().await;

    // method_not_found carries the JSON-RPC -32601 code AND the HG037 origin in `data`.
    let e = method_not_found("worker", "M_BOGUS");
    assert_eq!(e.code, -32601);
    assert_eq!(
        e.data
            .as_ref()
            .and_then(|d| d.get("code"))
            .and_then(|c| c.as_str()),
        Some("HG037")
    );
    assert!(
        e.message.contains("HG037"),
        "message carries the code: {}",
        e.message
    );

    // Request / Response / Notification each round-trip through encode → decode.
    let req = RpcRequest {
        jsonrpc: "2.0".into(),
        id: 7,
        method: "M_PING".into(),
        params: serde_json::json!({ "a": 1 }),
    };
    match decode(&encode(&RpcFrame::Request(req.clone()))).unwrap() {
        RpcFrame::Request(r) => assert_eq!(r, req),
        other => panic!("expected Request, got {other:?}"),
    }
    let resp = RpcResponse {
        jsonrpc: "2.0".into(),
        id: 7,
        result: Some(serde_json::json!("ok")),
        error: None,
    };
    match decode(&encode(&RpcFrame::Response(resp.clone()))).unwrap() {
        RpcFrame::Response(r) => assert_eq!(r, resp),
        other => panic!("expected Response, got {other:?}"),
    }
    let note = RpcNotification {
        jsonrpc: "2.0".into(),
        method: "N_CHAT".into(),
        params: serde_json::json!({}),
    };
    match decode(&encode(&RpcFrame::Notification(note.clone()))).unwrap() {
        RpcFrame::Notification(n) => assert_eq!(n, note),
        other => panic!("expected Notification, got {other:?}"),
    }

    // A non-"2.0" version is validated SOFTLY: warned, but still decoded.
    match decode(r#"{"jsonrpc":"1.0","id":1,"method":"x"}"#).unwrap() {
        RpcFrame::Request(r) => assert_eq!(r.id, 1),
        other => panic!("soft version check still decodes, got {other:?}"),
    }

    // Non-JSON and a frame that is neither request nor notification are decode errors.
    assert!(matches!(
        decode("not json at all"),
        Err(HiggsError::RpcDecode { .. })
    ));
    assert!(matches!(
        decode(r#"{"jsonrpc":"2.0"}"#),
        Err(HiggsError::RpcDecode { .. })
    ));
}

// ── system.rs ─────────────────────────────────────────────────────────────────

use higgs::system::{DeviceKind, GpuDevice, SystemInfo};

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn system_gather_composes_snapshot() {
    let _g = serial().await;
    let home = TempDir::new().unwrap();
    let scan = TempDir::new().unwrap();
    let _env = EnvVarGuard::set("HIGGS_HOME", home.path().to_str().unwrap());

    // A bare facade (no worker) yields the read-only server config that `gather` folds in.
    let config = higgs::HiggsConfig {
        lmstudio_dirs: vec![scan.path().to_path_buf()],
        hf_dirs: vec![],
        ollama_dirs: vec![],
        default_load: higgs::HiggsConfig::default().default_load,
        worker_exe: None,
    };
    let higgs = Arc::new(higgs::Higgs::new(config));
    let server_config = higgs.server_config();

    // Gather with a synthetic device roster: only GPU-kind VRAM is summed (a CPU device
    // reports system memory and must NOT inflate the GPU total).
    let gpus = vec![
        GpuDevice {
            name: "Metal".into(),
            description: "Test GPU".into(),
            kind: DeviceKind::Gpu,
            vram_total_bytes: 8_000_000_000,
            vram_free_bytes: 4_000_000_000,
        },
        GpuDevice {
            name: "CPU".into(),
            description: "host".into(),
            kind: DeviceKind::Cpu,
            vram_total_bytes: 999,
            vram_free_bytes: 999,
        },
    ];
    let info = SystemInfo::gather(server_config, gpus);

    assert_eq!(info.runtime.engine, "llama.cpp");
    assert!(!info.runtime.version.is_empty(), "engine version reported");
    assert!(!info.hardware.cpu_name.is_empty(), "cpu brand sampled");
    assert!(info.hardware.cpu_cores >= 1, "at least one logical core");
    assert!(info.hardware.ram_total_bytes > 0, "total RAM sampled");
    assert_eq!(info.hardware.gpus.len(), 2, "roster folded in verbatim");
    assert_eq!(
        info.hardware.vram_total_bytes, 8_000_000_000,
        "only GPU-kind VRAM summed"
    );
    assert_eq!(
        info.config.lmstudio_dirs.len(),
        1,
        "server config folded into the snapshot"
    );

    higgs.stop().await;
}
