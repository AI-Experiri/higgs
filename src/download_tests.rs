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
