
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
