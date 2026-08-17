use super::*;

/// A no-network fetcher that emits canned chunks + progress. `report_total` toggles
/// whether it advertises a content length (`Some` vs `None`) so both progress arms run.
/// `fail_with` injects a CLASSIFIED failure (via a constructor closure, since
/// `HiggsError` isn't `Clone`), exercising the propagate-verbatim path.
struct FakeFetcher {
    chunks: Vec<Vec<u8>>,
    fail_with: Option<Box<dyn Fn() -> HiggsError + Send + Sync>>,
    report_total: bool,
}

impl FakeFetcher {
    fn new(chunks: Vec<Vec<u8>>) -> Self {
        Self {
            chunks,
            fail_with: None,
            report_total: true,
        }
    }

    /// A fetcher that always fails with the error produced by `mk`.
    fn failing(mk: impl Fn() -> HiggsError + Send + Sync + 'static) -> Self {
        Self {
            chunks: vec![],
            fail_with: Some(Box::new(mk)),
            report_total: true,
        }
    }
}

impl Fetcher for FakeFetcher {
    async fn fetch(
        &self,
        _target: &PullTarget,
        on_chunk: &mut (dyn FnMut(&[u8]) + Send),
        progress: &mut (dyn FnMut(u64, Option<u64>) + Send),
    ) -> Result<(), HiggsError> {
        if let Some(mk) = &self.fail_with {
            return Err(mk());
        }
        let total: u64 = self.chunks.iter().map(|c| c.len() as u64).sum();
        let mut done = 0u64;
        for c in &self.chunks {
            on_chunk(c);
            done += c.len() as u64;
            progress(done, self.report_total.then_some(total));
        }
        Ok(())
    }
}

#[tokio::test]
async fn download_handles_unknown_content_length() {
    let dir = tempfile::tempdir().unwrap();
    let f = FakeFetcher {
        chunks: vec![b"a".to_vec(), b"bc".to_vec()],
        fail_with: None,
        report_total: false,
    };
    let mut seen: Vec<(u64, Option<u64>)> = Vec::new();
    let p = download(
        &PullTarget::new("org/m", "x.gguf"),
        dir.path(),
        &f,
        &mut |d, t| seen.push((d, t)),
    )
    .await
    .unwrap();
    assert_eq!(std::fs::read(&p).unwrap(), b"abc");
    assert_eq!(
        seen,
        vec![(1, None), (3, None)],
        "progress with unknown total"
    );
}

#[test]
fn hf_url_resolve_endpoint_default_and_override() {
    // `hf_url` builds the `resolve` URL for the reqwest FALLBACK path. It must target the
    // SAME mirror/proxy as the hub-client primary (`hub::hf_client` reads the same var) —
    // else a user's `HIGGS_HF_ENDPOINT` mirror is silently bypassed on the fallback. BOTH
    // the default and the override are asserted under one `TEST_ENV_LOCK` hold (and the var
    // is restored), so neither can race a parallel test that reads `HIGGS_HF_ENDPOINT`.
    let _env = crate::TEST_ENV_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let prev = std::env::var_os("HIGGS_HF_ENDPOINT");

    // Unset → the public default (removed explicitly so ambient env can't perturb it).
    // SAFETY: serialized by TEST_ENV_LOCK; restored below.
    unsafe { std::env::remove_var("HIGGS_HF_ENDPOINT") };
    assert_eq!(
        hf_url("org/model", "main", "m.gguf"),
        "https://huggingface.co/org/model/resolve/main/m.gguf",
        "default endpoint"
    );

    // Override → the mirror, with the trailing slash trimmed.
    unsafe { std::env::set_var("HIGGS_HF_ENDPOINT", "https://mirror.example.test/") };
    assert_eq!(
        hf_url("org/model", "main", "m.gguf"),
        "https://mirror.example.test/org/model/resolve/main/m.gguf",
        "override honored, trailing slash trimmed"
    );

    // Empty value → ignored (treated as unset) → the public default.
    unsafe { std::env::set_var("HIGGS_HF_ENDPOINT", "") };
    assert_eq!(
        hf_url("org/model", "main", "m.gguf"),
        "https://huggingface.co/org/model/resolve/main/m.gguf",
        "empty override ignored"
    );

    // SAFETY: still under the lock.
    unsafe {
        match prev {
            Some(v) => std::env::set_var("HIGGS_HF_ENDPOINT", v),
            None => std::env::remove_var("HIGGS_HF_ENDPOINT"),
        }
    }
}

#[test]
fn dest_path_enforces_scanner_layout_and_rejects_escapes() {
    let root = Path::new("/models");
    // (repo, file, should_succeed, why) — layout is <org>/<model>/*.gguf, and any path
    // escape / URL-reserved char / empty segment is rejected.
    let cases: &[(&str, &str, bool, &str)] = &[
        ("org/m", "x.gguf", true, "canonical <org>/<model>/*.gguf"),
        ("gpt2", "x.gguf", false, "single-segment repo"),
        ("a/b/c", "x.gguf", false, "three-segment repo"),
        ("org/m", "sub/x.gguf", false, "subdir in file"),
        ("org/m", "x.bin", false, "non-gguf"),
        ("org/m", "X.GGUF", true, "uppercase .GGUF"),
        ("org\\m", "x.gguf", false, "backslash in repo"),
        ("../etc", "passwd", false, "parent-dir escape"),
        ("org", "/abs", false, "absolute file"),
        ("", "x.gguf", false, "empty repo"),
        ("../m", "x.gguf", false, "parent in org"),
        ("org/m", "x?.gguf", false, "'?' (query) in file"),
        ("org/m", "x#.gguf", false, "'#' (fragment) in file"),
        ("or#g/m", "x.gguf", false, "'#' in repo"),
        ("org/m", "a b.gguf", false, "space in file"),
        (
            "TheBloke/Llama-2.7B_GGUF",
            "model.q4_0.gguf",
            true,
            "normal HF charset",
        ),
    ];
    for (repo, file, ok, why) in cases {
        assert_eq!(
            dest_path(root, repo, file).is_ok(),
            *ok,
            "{why}: {repo:?}/{file:?}"
        );
    }
}

#[tokio::test]
async fn download_rejects_unsafe_revision() {
    let dir = tempfile::tempdir().unwrap();
    let mut t = PullTarget::new("org/m", "x.gguf");
    t.revision = "main#frag".into();
    let f = FakeFetcher::new(vec![b"x".to_vec()]);
    let err = download(&t, dir.path(), &f, &mut |_, _| {})
        .await
        .unwrap_err();
    assert!(
        err.to_string().starts_with("[HG025]"),
        "reserved char in revision → HG025: {err}"
    );
    // A branch-path revision with '/' is allowed.
    t.revision = "refs/pr/1".into();
    // (won't actually fetch over network in this unit test; the fake fetcher serves it)
    assert!(
        download(&t, dir.path(), &f, &mut |_, _| {}).await.is_ok(),
        "branch-path revision ok"
    );
}

#[tokio::test]
async fn download_writes_atomically_and_reports_progress() {
    let dir = tempfile::tempdir().unwrap();
    let fetcher = FakeFetcher::new(vec![b"he".to_vec(), b"llo".to_vec()]);
    let mut seen: Vec<(u64, Option<u64>)> = Vec::new();
    let target = PullTarget::new("higgs-test/m", "model.gguf");
    let path = download(&target, dir.path(), &fetcher, &mut |d, t| seen.push((d, t)))
        .await
        .unwrap();
    assert_eq!(std::fs::read(&path).unwrap(), b"hello", "bytes written");
    assert!(
        path.ends_with("higgs-test/m/model.gguf"),
        "lands under repo/file: {path:?}"
    );
    assert_eq!(seen.last(), Some(&(5, Some(5))), "final progress = total");
    // No leftover .part temp.
    assert!(!path.with_extension("part").exists(), "temp cleaned up");
}

#[test]
fn pull_target_new_defaults_to_main() {
    let t = PullTarget::new("org/m", "x.gguf");
    assert_eq!(t.revision, "main");
    assert_eq!((t.repo.as_str(), t.file.as_str()), ("org/m", "x.gguf"));
}

#[tokio::test]
async fn download_replaces_an_existing_model_in_place() {
    let dir = tempfile::tempdir().unwrap();
    let target = PullTarget::new("org/m", "x.gguf");
    let f1 = FakeFetcher::new(vec![b"v1".to_vec()]);
    let p1 = download(&target, dir.path(), &f1, &mut |_, _| {})
        .await
        .unwrap();
    assert_eq!(std::fs::read(&p1).unwrap(), b"v1");
    // Re-pull replaces the existing file in place (atomic rename over the old model).
    let f2 = FakeFetcher::new(vec![b"v2!".to_vec()]);
    let p2 = download(&target, dir.path(), &f2, &mut |_, _| {})
        .await
        .unwrap();
    assert_eq!(p2, p1, "same destination");
    assert_eq!(std::fs::read(&p2).unwrap(), b"v2!", "model replaced");
}

#[tokio::test]
async fn download_propagates_classified_fetch_error_and_leaves_no_file() {
    let dir = tempfile::tempdir().unwrap();
    // The fetcher's CLASSIFIED error is propagated VERBATIM (not re-wrapped in HG025).
    let fetcher = FakeFetcher::failing(|| HiggsError::HubTransport {
        repo: "org/m".into(),
        detail: "network down".into(),
    });
    let target = PullTarget::new("org/m", "m.gguf");
    let err = download(&target, dir.path(), &fetcher, &mut |_, _| {})
        .await
        .unwrap_err();
    assert!(
        err.to_string().starts_with("[HG033]"),
        "classified transport error propagates: {err}"
    );
    let dest = dest_path(dir.path(), "org/m", "m.gguf").unwrap();
    assert!(!dest.exists(), "no file on failure");
    assert!(!dest.with_extension("part").exists(), "no temp left behind");
}

#[tokio::test]
async fn download_dual_falls_back_to_second_fetcher() {
    let dir = tempfile::tempdir().unwrap();
    // Primary fails (auth) → fallback succeeds → bytes land, no error.
    let primary = FakeFetcher::failing(|| HiggsError::HubAuthFailed {
        repo: "org/m".into(),
        detail: "401".into(),
    });
    let fallback = FakeFetcher::new(vec![b"ok".to_vec()]);
    let target = PullTarget::new("org/m", "x.gguf");
    let path = download_dual(&target, dir.path(), &primary, &fallback, &mut |_, _| {})
        .await
        .expect("fallback succeeds");
    assert_eq!(
        std::fs::read(&path).unwrap(),
        b"ok",
        "fallback bytes written"
    );
}

#[tokio::test]
async fn download_dual_does_not_retry_local_validation_errors() {
    let dir = tempfile::tempdir().unwrap();
    // A bad revision fails wire-validation INSIDE download() (HG025), before any fetcher.
    // download_dual must surface that HG025 verbatim — NOT retry the fallback and wrap it
    // as HG036. The fallback here WOULD succeed if reached, so a non-HG025 result fails.
    let mut target = PullTarget::new("org/m", "x.gguf");
    target.revision = "main#frag".into();
    let primary = FakeFetcher::new(vec![b"a".to_vec()]);
    let fallback = FakeFetcher::new(vec![b"b".to_vec()]);
    let err = download_dual(&target, dir.path(), &primary, &fallback, &mut |_, _| {})
        .await
        .unwrap_err();
    assert!(
        err.to_string().starts_with("[HG025]"),
        "local validation error surfaced verbatim (not HG036): {err}"
    );
}

#[tokio::test]
async fn download_dual_retries_fallback_on_fetcher_sourced_hubfilewrite() {
    let dir = tempfile::tempdir().unwrap();
    // HG034 RETURNED BY THE FETCHER is the hub client's own cache I/O (HFError::Io) — a
    // fetcher-side failure the direct fallback (streaming to ~/.higgs/models) can recover
    // from. Source = Fetcher ⇒ the fallback runs and succeeds.
    let primary = FakeFetcher::failing(|| HiggsError::HubFileWrite {
        repo: "org/m".into(),
        file: "x.gguf".into(),
        detail: "crate cache write failed".into(),
    });
    let fallback = FakeFetcher::new(vec![b"recovered".to_vec()]);
    let target = PullTarget::new("org/m", "x.gguf");
    let path = download_dual(&target, dir.path(), &primary, &fallback, &mut |_, _| {})
        .await
        .expect("fallback recovers a fetcher-sourced HG034");
    assert_eq!(std::fs::read(&path).unwrap(), b"recovered");
}

#[tokio::test]
async fn download_dual_does_not_retry_downloads_own_fs_error() {
    let dir = tempfile::tempdir().unwrap();
    // Put a FILE where the `org` dir must be, so `download`'s OWN create_dir_all (for the
    // dest parent `org/m`) fails — a LOCAL HG034. The fallback would hit the same
    // un-creatable path, so it must NOT retry: surface HG034 verbatim, not HG036. The
    // fakes WOULD succeed if reached, so a non-HG034 result fails the test.
    std::fs::write(dir.path().join("org"), b"x").unwrap();
    let primary = FakeFetcher::new(vec![b"a".to_vec()]);
    let fallback = FakeFetcher::new(vec![b"b".to_vec()]);
    let target = PullTarget::new("org/m", "x.gguf");
    let err = download_dual(&target, dir.path(), &primary, &fallback, &mut |_, _| {})
        .await
        .unwrap_err();
    assert!(
        err.to_string().starts_with("[HG034]"),
        "download's own fs error surfaced verbatim (not HG036): {err}"
    );
}

#[tokio::test]
async fn download_maps_local_create_dir_failure_to_hg034() {
    // `download` (not `download_dual`): put a FILE where the `org` dir must be created, so the
    // dest-parent `create_dir_all` (download.rs:194-195) fails → mapped to a LOCAL HG034 and
    // surfaced verbatim through `download`'s `into_inner`. No fetcher is ever consulted.
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("org"), b"blocker").unwrap();
    let fetcher = FakeFetcher::new(vec![b"never".to_vec()]);
    let target = PullTarget::new("org/m", "x.gguf");
    let err = download(&target, dir.path(), &fetcher, &mut |_, _| {})
        .await
        .unwrap_err();
    assert!(
        err.to_string().starts_with("[HG034]"),
        "create_dir_all failure → HG034 (HubFileWrite): {err}"
    );
}

#[tokio::test]
async fn download_maps_rename_failure_to_hg034_and_cleans_temp() {
    // Force the atomic rename (download.rs:240-242) to fail: pre-create the final dest path as a
    // NON-EMPTY directory. `rename(2)` of a regular `.part` file onto a non-empty dir fails
    // (ENOTEMPTY/EISDIR) → mapped to a LOCAL HG034, and the `.part` temp is removed.
    let dir = tempfile::tempdir().unwrap();
    let dest = dest_path(dir.path(), "org/m", "x.gguf").unwrap();
    std::fs::create_dir_all(&dest).unwrap();
    // A child inside the dest dir makes the rename-over-dir definitively fail.
    std::fs::write(dest.join("occupied"), b"keep").unwrap();
    let fetcher = FakeFetcher::new(vec![b"payload".to_vec()]);
    let target = PullTarget::new("org/m", "x.gguf");
    let err = download(&target, dir.path(), &fetcher, &mut |_, _| {})
        .await
        .unwrap_err();
    assert!(
        err.to_string().starts_with("[HG034]"),
        "rename onto a non-empty dir → HG034: {err}"
    );
    // No `.part` temp lingers next to the dest after the failed rename.
    let leftover = std::fs::read_dir(dest.parent().unwrap())
        .unwrap()
        .filter_map(Result::ok)
        .any(|e| {
            e.file_name()
                .to_string_lossy()
                .to_string()
                .contains("part.")
        });
    assert!(!leftover, "the .part temp is removed after a failed rename");
}

#[tokio::test]
async fn download_dual_returns_primary_success_without_consulting_fallback() {
    // The primary succeeds on the first attempt → `download_dual` returns its path via the
    // `Ok(path) => Ok(path)` arm (download.rs:267); the fallback is never invoked. A fallback
    // that would PANIC if called proves it is not consulted.
    let dir = tempfile::tempdir().unwrap();
    let primary = FakeFetcher::new(vec![b"primary-bytes".to_vec()]);
    let fallback = FakeFetcher::failing(|| panic!("fallback must NOT run when primary succeeds"));
    let target = PullTarget::new("org/m", "x.gguf");
    let path = download_dual(&target, dir.path(), &primary, &fallback, &mut |_, _| {})
        .await
        .expect("primary succeeds");
    assert_eq!(std::fs::read(&path).unwrap(), b"primary-bytes");
}

#[test]
fn models_dir_creates_under_higgs_home() {
    // `models_dir()` resolves `<HIGGS_HOME>/models` and creates it. Drive it through an
    // isolated HIGGS_HOME (serialized by TEST_ENV_LOCK, restored after) so it neither reads
    // nor mutates the dev machine's real `~/.higgs`.
    let _env = crate::TEST_ENV_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let home = tempfile::tempdir().unwrap();
    let prev = std::env::var_os("HIGGS_HOME");
    // SAFETY: serialized by TEST_ENV_LOCK; restored below.
    unsafe { std::env::set_var("HIGGS_HOME", home.path()) };

    let dir = models_dir().expect("models_dir resolves + creates");
    assert_eq!(dir, home.path().join("models"), "lands at <HOME>/models");
    assert!(dir.is_dir(), "the models dir is created");

    // SAFETY: still under the lock.
    unsafe {
        match prev {
            Some(v) => std::env::set_var("HIGGS_HOME", v),
            None => std::env::remove_var("HIGGS_HOME"),
        }
    }
}

/// Spawn a local axum server on `127.0.0.1:0` serving `app`. Returns the base URL
/// (`http://127.0.0.1:PORT`) and the server task. The `HttpFetcher` tests point
/// `HIGGS_HF_ENDPOINT` at this so they exercise the real `reqwest` streaming path with NO
/// network.
async fn serve_once(app: axum::Router) -> (String, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let handle = tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    (base, handle)
}

#[allow(clippy::await_holding_lock)]
#[tokio::test]
async fn http_fetcher_streams_success_body_and_reports_progress() {
    // The reqwest FALLBACK fetcher streams a 200 `resolve` response: each chunk is handed to
    // `on_chunk` and progress is reported with the advertised content length
    // (download.rs:306-340). A LOCAL axum server stands in for huggingface.co via
    // HIGGS_HF_ENDPOINT — no network. The env mutation is serialized by TEST_ENV_LOCK; the URL
    // is built (under the lock) before the first await, so the override can't race.
    let body = b"hello-gguf-bytes".to_vec();
    let app = {
        let body = body.clone();
        axum::Router::new().fallback(move || {
            let body = body.clone();
            async move { body }
        })
    };
    let (base, server) = serve_once(app).await;

    let _env = crate::TEST_ENV_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let prev = std::env::var_os("HIGGS_HF_ENDPOINT");
    // SAFETY: serialized by TEST_ENV_LOCK; restored below.
    unsafe { std::env::set_var("HIGGS_HF_ENDPOINT", &base) };

    let mut got = Vec::new();
    let mut seen: Vec<(u64, Option<u64>)> = Vec::new();
    let res = HttpFetcher
        .fetch(
            &PullTarget::new("org/m", "x.gguf"),
            &mut |b: &[u8]| got.extend_from_slice(b),
            &mut |d, t| seen.push((d, t)),
        )
        .await;

    // SAFETY: still under the lock.
    unsafe {
        match prev {
            Some(v) => std::env::set_var("HIGGS_HF_ENDPOINT", v),
            None => std::env::remove_var("HIGGS_HF_ENDPOINT"),
        }
    }
    server.abort();

    res.expect("a 200 resolve streams successfully");
    assert_eq!(got, body, "all streamed bytes delivered to on_chunk");
    let total: u64 = body.len() as u64;
    assert_eq!(
        seen.last(),
        Some(&(total, Some(total))),
        "final progress = full body with the advertised content length"
    );
}

#[allow(clippy::await_holding_lock)]
#[tokio::test]
async fn http_fetcher_classifies_non_success_status() {
    // A non-success status routes through `hub::http_status_to_error`: 404 → HG030
    // (download.rs:321-329). The local server returns 404 for every path.
    let app =
        axum::Router::new().fallback(|| async { (axum::http::StatusCode::NOT_FOUND, "missing") });
    let (base, server) = serve_once(app).await;

    let _env = crate::TEST_ENV_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let prev = std::env::var_os("HIGGS_HF_ENDPOINT");
    // SAFETY: serialized by TEST_ENV_LOCK; restored below.
    unsafe { std::env::set_var("HIGGS_HF_ENDPOINT", &base) };

    let res = HttpFetcher
        .fetch(
            &PullTarget::new("org/m", "x.gguf"),
            &mut |_: &[u8]| {},
            &mut |_, _| {},
        )
        .await;

    // SAFETY: still under the lock.
    unsafe {
        match prev {
            Some(v) => std::env::set_var("HIGGS_HF_ENDPOINT", v),
            None => std::env::remove_var("HIGGS_HF_ENDPOINT"),
        }
    }
    server.abort();

    let err = res.expect_err("404 status must fail the fetch");
    assert!(
        err.to_string().starts_with("[HG030]"),
        "404 → HubResourceNotFound (HG030): {err}"
    );
}

#[allow(clippy::await_holding_lock)]
#[tokio::test]
async fn http_fetcher_maps_connection_refused_to_transport() {
    // A transport-level failure (the initial `reqwest::get`) maps to HG033 (HubTransport,
    // download.rs:318-320). Point the endpoint at a reserved closed port so the connect fails
    // locally with no network.
    let _env = crate::TEST_ENV_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let prev = std::env::var_os("HIGGS_HF_ENDPOINT");
    // 127.0.0.1:1 — privileged/closed; the connect is refused locally (no DNS, no network).
    // SAFETY: serialized by TEST_ENV_LOCK; restored below.
    unsafe { std::env::set_var("HIGGS_HF_ENDPOINT", "http://127.0.0.1:1") };

    let res = HttpFetcher
        .fetch(
            &PullTarget::new("org/m", "x.gguf"),
            &mut |_: &[u8]| {},
            &mut |_, _| {},
        )
        .await;

    // SAFETY: still under the lock.
    unsafe {
        match prev {
            Some(v) => std::env::set_var("HIGGS_HF_ENDPOINT", v),
            None => std::env::remove_var("HIGGS_HF_ENDPOINT"),
        }
    }

    let err = res.expect_err("a refused connection must fail the fetch");
    assert!(
        err.to_string().starts_with("[HG033]"),
        "connection refused → HubTransport (HG033): {err}"
    );
}

#[tokio::test]
async fn download_dual_exhausted_is_hg036_with_both_diagnoses() {
    let dir = tempfile::tempdir().unwrap();
    // BOTH fail → HG036 carrying the primary's code AND the fallback's detail.
    let primary = FakeFetcher::failing(|| HiggsError::HubResourceNotFound {
        repo: "org/m".into(),
        resource: "file x.gguf".into(),
        detail: "404".into(),
    });
    let fallback = FakeFetcher::failing(|| HiggsError::HubTransport {
        repo: "org/m".into(),
        detail: "dns failure".into(),
    });
    let target = PullTarget::new("org/m", "x.gguf");
    let err = download_dual(&target, dir.path(), &primary, &fallback, &mut |_, _| {})
        .await
        .unwrap_err();
    let s = err.to_string();
    assert!(s.starts_with("[HG036]"), "both failed → exhausted: {s}");
    assert!(s.contains("[HG030]"), "primary code preserved: {s}");
    assert!(s.contains("dns failure"), "fallback detail preserved: {s}");
    let dest = dest_path(dir.path(), "org/m", "x.gguf").unwrap();
    assert!(!dest.exists(), "no file when both paths fail");
}

#[test]
fn remove_partials_matches_the_temp_prefix_exactly() {
    let dir = tempfile::tempdir().unwrap();
    // Multi-dot filename: dest "a.b.gguf" → temps named "a.b.gguf.part.<pid>.<n>".
    let model_dir = dir.path().join("org/m");
    std::fs::create_dir_all(&model_dir).unwrap();
    let pid = std::process::id();
    let temp1 = model_dir.join(format!("a.b.gguf.part.{pid}.0"));
    let temp2 = model_dir.join(format!("a.b.gguf.part.{pid}.7"));
    // Another PROCESS's temp for the SAME file — its transfer is live; the
    // sweep only ever cleans temps this process minted.
    let foreign = model_dir.join("a.b.gguf.part.999999.0");
    let dest = model_dir.join("a.b.gguf");
    // Prefix near-misses that must survive: another file's temp, a model whose
    // NAME happens to start the same, a REAL model literally named
    // "<stem>.part.<x>.gguf" (a bare prefix match would delete it — data
    // loss), and that model's own in-flight temp.
    let other_temp = model_dir.join(format!("a.c.gguf.part.{pid}.0"));
    let lookalike = model_dir.join("a.b.partial.gguf");
    let prefix_model = model_dir.join("a.b.part.x.gguf");
    let prefix_model_temp = model_dir.join(format!("a.b.part.x.gguf.part.{pid}.0"));
    for p in [
        &temp1,
        &temp2,
        &foreign,
        &dest,
        &other_temp,
        &lookalike,
        &prefix_model,
        &prefix_model_temp,
    ] {
        std::fs::write(p, b"x").unwrap();
    }
    remove_partials(dir.path(), "org/m", "a.b.gguf").expect("sweep");
    assert!(!temp1.exists() && !temp2.exists(), "both OWN temps swept");
    assert!(
        foreign.exists(),
        "another process's live temp for the same file must survive"
    );
    assert!(dest.exists(), "final file untouched");
    assert!(other_temp.exists(), "another file's temp untouched");
    assert!(lookalike.exists(), "non-temp lookalike untouched");
    assert!(
        prefix_model.exists(),
        "a REAL model named <stem>.part.<x>.gguf must never be deleted"
    );
    assert!(
        prefix_model_temp.exists(),
        "a concurrent different-file temp sharing the prefix must survive"
    );
}

#[test]
fn remove_partials_rejects_the_same_bad_shapes_as_the_download() {
    let dir = tempfile::tempdir().unwrap();
    assert!(remove_partials(dir.path(), "../escape", "m.gguf").is_err());
    assert!(remove_partials(dir.path(), "org/m", "not-gguf.txt").is_err());
    // A missing model dir is fine — nothing to sweep.
    remove_partials(dir.path(), "org/m", "m.gguf").expect("no dir, no-op");
}

#[test]
fn dest_path_rejects_segments_the_filesystem_or_temp_naming_cannot_hold() {
    // A name the filesystem can't hold is refused UP FRONT with the typed
    // error — which also keeps every REGISTERED pull announceable (the hub's
    // announcement validator IS dest_path; a node-side accept the hub would
    // drop would leave an in-flight pull invisible to the fleet view).
    //
    // The FILE bound reserves room for the download's own temp: bytes go to
    // `<file>.part.<u32 pid>.<u64 seq>` first, so the file name must fit
    // NAME_MAX (255) MINUS that worst-case 37-byte suffix — a 250-char name
    // whose FINAL path is legal would still die ENAMETOOLONG at the temp
    // create, i.e. "accepted but uncreatable", exactly what this refusal
    // exists to prevent. Repo segments are directories (no temp suffix):
    // plain NAME_MAX.
    let root = std::path::Path::new("/models");
    let file_at_bound = format!("{}.gguf", "q".repeat(213)); // 218 bytes
    assert_eq!(file_at_bound.len(), 218);
    assert!(dest_path(root, "acme/m", &file_at_bound).is_ok());
    let file_over = format!("{}.gguf", "q".repeat(214)); // 219 bytes
    assert!(
        dest_path(root, "acme/m", &file_over).is_err(),
        "a file whose TEMP cannot be created is refused up front"
    );
    let org_at_bound = format!("{}/m", "q".repeat(255));
    assert!(dest_path(root, &org_at_bound, "m.gguf").is_ok());
    let org_over = format!("{}/m", "q".repeat(256));
    assert!(dest_path(root, &org_over, "m.gguf").is_err());
}

#[tokio::test]
async fn download_records_the_ledger_lifecycle_and_respects_a_live_claim() {
    // The ledger's ONE writer is this download path: a successful transfer
    // must leave a terminal `done` history entry (with the final path), and a
    // key some live process already claims must be refused [HG090] BEFORE any
    // bytes move — the machine-wide extension of the in-process rule.
    use crate::catalog::ledger;
    use crate::catalog::wire::DownloadLedgerStatus;
    let home = tempfile::tempdir().expect("home");
    let root = home.path().join("models");
    std::fs::create_dir_all(&root).expect("root");
    let target = PullTarget::new("acme/m", "m.gguf");
    let ok = FakeFetcher::new(vec![b"GGUF-bytes".to_vec()]);
    let path = download_dual(&target, &root, &ok, &ok, &mut |_, _| {})
        .await
        .expect("download lands");
    let all = ledger::read_all(&root);
    assert_eq!(all.len(), 1, "one history entry: {all:?}");
    assert_eq!(all[0].status, DownloadLedgerStatus::Done);
    assert_eq!(all[0].path.as_deref(), path.to_str());
    // The terminal carries the FINAL byte count — the throttled progress
    // mirror (8 MiB deltas) never fires for a small file, so without a final
    // flush a done entry would read "0 bytes".
    assert_eq!(
        all[0].downloaded,
        b"GGUF-bytes".len() as u64,
        "done entry records the real final size"
    );
    assert!(ledger::read_live(&root).is_empty(), "key freed");

    // Another live download on the same key refuses the duplicate up front
    // with [HG090] — the machine-wide flock in `catalog::download_lock`
    // (held here directly to simulate a concurrent process without a
    // second binary; the download-lock tests cover the flock semantics
    // themselves).
    let _held = crate::catalog::download_lock::DownloadLock::acquire(&root, "acme/m", "held.gguf")
        .expect("hold the key");
    let err = download_dual(
        &PullTarget::new("acme/m", "held.gguf"),
        &root,
        &ok,
        &ok,
        &mut |_, _| {},
    )
    .await
    .expect_err("held key refuses the duplicate");
    assert!(
        matches!(err, HiggsError::DownloadInFlight { .. }),
        "machine-wide [HG090]: {err}"
    );
    drop(_held);
    // Once released, the same key downloads normally.
    let _ok_after = download_dual(
        &PullTarget::new("acme/m", "held.gguf"),
        &root,
        &ok,
        &ok,
        &mut |_, _| {},
    )
    .await
    .expect("key downloads once released");

    // A FAILED transfer records a failed terminal (detail kept).
    let bad = FakeFetcher::failing(|| HiggsError::HubTransport {
        repo: "acme/m".into(),
        detail: "boom".into(),
    });
    let _ = download_dual(
        &PullTarget::new("acme/m", "gone.gguf"),
        &root,
        &bad,
        &bad,
        &mut |_, _| {},
    )
    .await
    .expect_err("both fetchers fail");
    let failed = ledger::read_all(&root)
        .into_iter()
        .find(|e| e.file == "gone.gguf")
        .expect("failed entry recorded");
    assert_eq!(failed.status, DownloadLedgerStatus::Failed);
    assert!(failed.detail.is_some());
}

#[tokio::test]
async fn an_invalid_identity_never_touches_the_ledger() {
    // The ledger documents its `(repo, file)` as dest_path-valid — so
    // validation must run BEFORE the claim, or a refused pull (HG025) would
    // persist an identity the announcement validator itself would drop.
    use crate::catalog::ledger;
    let home = tempfile::tempdir().expect("home");
    let root = home.path().join("models");
    std::fs::create_dir_all(&root).expect("root");
    let ok = FakeFetcher::new(vec![b"x".to_vec()]);
    let err = download_dual(
        &PullTarget::new("acme/m", "not-a-model.txt"),
        &root,
        &ok,
        &ok,
        &mut |_, _| {},
    )
    .await
    .expect_err("non-gguf refused");
    assert!(err.to_string().starts_with("[HG025]"), "{err}");
    assert!(
        ledger::read_all(&root).is_empty(),
        "a refused identity leaves NO ledger trace"
    );
}

#[tokio::test]
async fn fallback_restart_resets_the_ledger_progress_mirror() {
    // The primary can die 4 GiB in; the fallback restarts from ZERO. The
    // throttle's high-water mark must reset on that regression, or the
    // ledger keeps announcing near-complete progress while the fallback
    // re-downloads from the start. The probe fallback reads the ledger the
    // moment its own first progress tick lands.
    use crate::catalog::ledger;
    use std::sync::{Arc, Mutex};
    let home = tempfile::tempdir().expect("home");
    let root = home.path().join("models");
    std::fs::create_dir_all(&root).expect("root");

    // Primary: report a huge progress mark (forces a throttled ledger write),
    // then fail with a fetcher-class error so the fallback runs.
    struct BigThenFail;
    impl Fetcher for BigThenFail {
        async fn fetch(
            &self,
            target: &PullTarget,
            _on_chunk: &mut (dyn FnMut(&[u8]) + Send),
            progress: &mut (dyn FnMut(u64, Option<u64>) + Send),
        ) -> Result<(), HiggsError> {
            progress(100 * 1024 * 1024, Some(200 * 1024 * 1024));
            Err(HiggsError::HubTransport {
                repo: target.repo.clone(),
                detail: "primary died mid-flight".into(),
            })
        }
    }
    // Fallback: on its first progress tick, capture what the LEDGER says.
    // CRITICAL for the pin's revert-power: this fallback reports the SAME
    // total as the primary (200 MiB) — exactly what production does, since
    // both transports fetch the same file. With an identical total, the
    // throttle's `total_changed` branch CANNOT fire on the restart tick,
    // so ONLY the `regressed` branch can force the immediate mirror write.
    // (A differing total here would mask a `regressed` revert: the r73
    // mutation test proved the old Some(3) made the pin pass with the
    // regression branch deleted.)
    struct ProbeFallback {
        root: std::path::PathBuf,
        seen: Arc<Mutex<Option<u64>>>,
    }
    impl Fetcher for ProbeFallback {
        async fn fetch(
            &self,
            _t: &PullTarget,
            on_chunk: &mut (dyn FnMut(&[u8]) + Send),
            progress: &mut (dyn FnMut(u64, Option<u64>) + Send),
        ) -> Result<(), HiggsError> {
            on_chunk(b"x");
            progress(1, Some(200 * 1024 * 1024));
            let live = ledger::read_live(&self.root)
                .into_iter()
                .next()
                .map(|e| e.downloaded);
            *self.seen.lock().unwrap() = live;
            on_chunk(b"yz");
            progress(3, Some(200 * 1024 * 1024));
            Ok(())
        }
    }
    let seen = Arc::new(Mutex::new(None));
    let fallback = ProbeFallback {
        root: root.clone(),
        seen: seen.clone(),
    };
    download_dual(
        &PullTarget::new("acme/m", "m.gguf"),
        &root,
        &BigThenFail,
        &fallback,
        &mut |_, _| {},
    )
    .await
    .expect("fallback lands the file");
    assert_eq!(
        *seen.lock().unwrap(),
        Some(1),
        "the fallback's restart-from-zero is mirrored immediately — not the primary's stale high-water mark"
    );
}

#[tokio::test]
async fn a_dropped_download_future_cleans_its_own_temp_via_the_guard() {
    // The `TempGuard` inside `download_attempt` unlinks the transfer's
    // specific `.part.<pid>.<seq>` tmp on drop — that's what makes the
    // cancel path safe to STOP blanket-sweeping (which was clipping
    // concurrent same-pid callers' live tmps, r45 finding). Test:
    // spawn a download future that parks on its first await; abort the
    // task; assert the tmp is gone AND no other side-effect leaked.
    struct ParkFetcher {
        park: tokio::sync::Mutex<Option<tokio::sync::oneshot::Receiver<()>>>,
    }
    impl Fetcher for ParkFetcher {
        async fn fetch(
            &self,
            _t: &PullTarget,
            on_chunk: &mut (dyn FnMut(&[u8]) + Send),
            _p: &mut (dyn FnMut(u64, Option<u64>) + Send),
        ) -> Result<(), HiggsError> {
            on_chunk(b"partial");
            if let Some(rx) = self.park.lock().await.take() {
                let _ = rx.await;
            }
            Ok(())
        }
    }
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().to_path_buf();
    let (_hold, rx) = tokio::sync::oneshot::channel::<()>();
    let park = ParkFetcher {
        park: tokio::sync::Mutex::new(Some(rx)),
    };
    let target = PullTarget::new("acme/m", "m.gguf");
    let task = {
        let root = root.clone();
        tokio::spawn(async move {
            let mut prog = |_: u64, _: Option<u64>| {};
            download(&target, &root, &park, &mut prog).await
        })
    };
    // Let the fetcher park after writing a partial.
    tokio::task::yield_now().await;
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    task.abort();
    let _ = task.await;
    // The tmp guard's drop must have unlinked our specific temp.
    let file_dir = root.join("acme/m");
    let leftover: Vec<_> = std::fs::read_dir(&file_dir)
        .map(|it| {
            it.filter_map(Result::ok)
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .collect()
        })
        .unwrap_or_default();
    assert!(
        leftover.is_empty() || leftover.iter().all(|n| !n.contains(".part.")),
        "the aborted download's `.part` temp is gone: {leftover:?}"
    );
}

/// The public `download()` primitive must ALSO acquire the machine-wide
/// download lock, else `flock` is bypassable via a caller that never used
/// `download_dual`. Before the fix, an in-crate caller could race a locked
/// download and both writers' renames would land the same final path.
/// Pinned by holding the lock and asserting `download()` refuses HG090.
#[tokio::test]
async fn public_download_also_gates_on_the_machine_wide_lock() {
    let root = tempfile::tempdir().unwrap();
    let _held =
        crate::catalog::download_lock::DownloadLock::acquire(root.path(), "acme/m", "held.gguf")
            .expect("hold the key");
    let ok = FakeFetcher::new(vec![b"x".to_vec()]);
    let err = download(
        &PullTarget::new("acme/m", "held.gguf"),
        root.path(),
        &ok,
        &mut |_, _| {},
    )
    .await
    .expect_err("public download() must refuse when the flock is held");
    assert!(
        matches!(err, HiggsError::DownloadInFlight { .. }),
        "public download() shares the same HG090 gate: {err}"
    );
    drop(_held);
    // Once released, download() proceeds normally.
    let _ok = download(
        &PullTarget::new("acme/m", "held.gguf"),
        root.path(),
        &ok,
        &mut |_, _| {},
    )
    .await
    .expect("download() works once the lock is released");
}

/// `download_dual_locked` must refuse a caller that holds a `DownloadLock`
/// for the WRONG key. Without this identity check, a caller could acquire
/// a lock for key A and pass it in for key B — bypassing B's flock and
/// letting a concurrent transfer of B corrupt disk state.
#[tokio::test]
async fn download_dual_locked_refuses_a_lock_that_protects_a_different_key() {
    let root = tempfile::tempdir().unwrap();
    // Acquire lock for OTHER key.
    let wrong =
        crate::catalog::download_lock::DownloadLock::acquire(root.path(), "acme/m", "other.gguf")
            .expect("acquire wrong-key lock");
    let ok = FakeFetcher::new(vec![b"x".to_vec()]);
    // Try to use it for the ACTUAL target.
    let err = crate::download::download_dual_locked(
        &PullTarget::new("acme/m", "target.gguf"),
        root.path(),
        &ok,
        &ok,
        &mut |_, _| {},
        &wrong,
    )
    .await
    .expect_err("mismatched lock must be refused");
    let msg = err.to_string();
    assert!(
        msg.contains("DownloadLock was acquired for a different key"),
        "coded internal-error surfaces the mismatch: {msg}"
    );
}

/// A malformed REVISION must refuse (HG025) with ZERO side effects — no
/// download-lock file, no ledger row. Before the fix, the revision check
/// lived only inside `download_attempt`, so `download_dual` had already
/// acquired the flock and written a `Downloading`+`Failed` ledger pair by
/// the time the refusal fired.
#[tokio::test]
async fn a_malformed_revision_refuses_before_any_lock_or_ledger_side_effect() {
    let root = tempfile::tempdir().unwrap();
    let ok = FakeFetcher::new(vec![b"x".to_vec()]);
    let mut t = PullTarget::new("acme/m", "m.gguf");
    t.revision = "main#frag".into();
    let err = download_dual(&t, root.path(), &ok, &ok, &mut |_, _| {})
        .await
        .expect_err("malformed revision refused");
    assert!(
        err.to_string().starts_with("[HG025]"),
        "wire-validation refusal: {err}"
    );
    // ZERO side effects: no lock file for the key, no ledger entries at all.
    assert!(
        !crate::catalog::download_lock::lock_file_exists(root.path(), "acme/m", "m.gguf"),
        "no lock file created by a refused pull"
    );
    assert!(
        crate::catalog::ledger::read_all(root.path()).is_empty(),
        "no ledger row created by a refused pull"
    );
}

#[test]
fn remove_partials_surfaces_an_unreadable_model_dir() {
    // A readdir failure other than NotFound must ERROR — a silent Ok would
    // let [HG089] claim "partial swept" over a multi-GB `.part` we never
    // even saw. Simulate with an unreadable (0o000) model dir.
    use std::os::unix::fs::PermissionsExt;
    let root = tempfile::tempdir().unwrap();
    let dir = root.path().join("acme/m");
    std::fs::create_dir_all(&dir).unwrap();
    let mode = std::fs::metadata(&dir).unwrap().permissions().mode();
    let mut deny = std::fs::metadata(&dir).unwrap().permissions();
    deny.set_mode(0o000);
    std::fs::set_permissions(&dir, deny).unwrap();
    let err = remove_partials(root.path(), "acme/m", "m.gguf")
        .expect_err("unreadable dir must not read as swept");
    let mut back = std::fs::metadata(&dir).unwrap().permissions();
    back.set_mode(mode);
    std::fs::set_permissions(&dir, back).unwrap();
    assert!(
        err.to_string().contains("cannot read model dir"),
        "the failure names the readdir: {err}"
    );
}

#[test]
fn remove_partials_reports_a_failed_unlink_instead_of_claiming_swept() {
    // First unlink failure wins: a matching temp in a read-only dir cannot
    // be removed — the sweep must ERROR (caller reports partial_swept =
    // false) instead of silently leaving the temp under a "swept" claim.
    use std::os::unix::fs::PermissionsExt;
    let root = tempfile::tempdir().unwrap();
    let dir = root.path().join("acme/m");
    std::fs::create_dir_all(&dir).unwrap();
    let temp = dir.join(format!("m.gguf.part.{}.7", std::process::id()));
    std::fs::write(&temp, b"partial").unwrap();
    let mode = std::fs::metadata(&dir).unwrap().permissions().mode();
    let mut ro = std::fs::metadata(&dir).unwrap().permissions();
    ro.set_mode(0o500); // r-x: readable, unlink refused
    std::fs::set_permissions(&dir, ro).unwrap();
    let err =
        remove_partials(root.path(), "acme/m", "m.gguf").expect_err("failed unlink must surface");
    let mut back = std::fs::metadata(&dir).unwrap().permissions();
    back.set_mode(mode);
    std::fs::set_permissions(&dir, back).unwrap();
    assert!(
        err.to_string().contains("could not remove"),
        "the failure names the stuck temp: {err}"
    );
    assert!(temp.is_file(), "the temp really is still there");
}

#[tokio::test]
async fn an_uncreatable_dest_parent_is_a_local_hg034_not_a_fetcher_error() {
    // A FILE squatting the repo-dir path makes `create_dir_all` fail before
    // any fetch: HG034 local filesystem failure (never eligible for the
    // fetcher fallback, never HG090 contention).
    let root = tempfile::tempdir().unwrap();
    std::fs::write(root.path().join("acme"), b"squatter").unwrap();
    let ok = FakeFetcher::new(vec![b"x".to_vec()]);
    let err = download(
        &PullTarget::new("acme/m", "m.gguf"),
        root.path(),
        &ok,
        &mut |_, _| {},
    )
    .await
    .expect_err("uncreatable parent refuses");
    assert!(
        matches!(err, HiggsError::HubFileWrite { .. }),
        "local fs failure is HG034: {err}"
    );
}

#[tokio::test]
async fn a_single_fetcher_download_propagates_the_failure_and_cleans_its_temp() {
    // The single-fetcher `download` path (CLI/facade delegation): a fetcher
    // failure propagates VERBATIM (classified, no HG036 dual-wrap) and the
    // attempt's temp is removed — no `.part` residue.
    let root = tempfile::tempdir().unwrap();
    let failing = FakeFetcher::failing(|| HiggsError::HubTransport {
        repo: "acme/m".into(),
        detail: "boom".into(),
    });
    let err = download(
        &PullTarget::new("acme/m", "m.gguf"),
        root.path(),
        &failing,
        &mut |_, _| {},
    )
    .await
    .expect_err("fetcher failure propagates");
    assert!(
        matches!(err, HiggsError::HubTransport { .. }),
        "verbatim classified error, not a dual-exhaust wrap: {err}"
    );
    let dir = root.path().join("acme/m");
    if let Ok(entries) = std::fs::read_dir(&dir) {
        for e in entries.flatten() {
            assert!(
                !e.file_name().to_string_lossy().contains(".part."),
                "no temp residue after a failed attempt"
            );
        }
    }
}
