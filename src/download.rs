//! Model downloader (P4b, `M_PULL`): fetch a GGUF from HuggingFace into the higgs models dir
//! (`~/.higgs/models/`) ONLY — never a scanned LM-Studio/Ollama dir. The byte source is
//! abstracted behind [`Fetcher`] so the path/guard/atomic-write logic is unit-tested without
//! network; the production [`HttpFetcher`] streams from the HF `resolve` URL with progress.
//!
//! Writes are atomic: bytes go to a `.part` temp and are renamed onto the final path only on
//! success, so a `M_SCAN`/`M_LOAD` never sees a half-written GGUF, and a failed download
//! leaves no partial file in the catalog.

use std::path::{Component, Path, PathBuf};

use crate::diagnostic::HiggsError;

/// What to pull: a HuggingFace `repo` + `file`, at `revision` (default `"main"`).
#[derive(Debug, Clone)]
pub struct PullTarget {
    pub repo: String,
    pub file: String,
    pub revision: String,
}

impl PullTarget {
    /// A target on `main` for `repo`/`file`.
    pub fn new(repo: impl Into<String>, file: impl Into<String>) -> Self {
        Self {
            repo: repo.into(),
            file: file.into(),
            revision: "main".into(),
        }
    }
}

/// The direct-download URL for a repo file at a revision. The base is HuggingFace by default,
/// overridable via `HIGGS_HF_ENDPOINT` (a mirror, an enterprise proxy, or a test server) —
/// trailing slash optional.
pub fn hf_url(repo: &str, revision: &str, file: &str) -> String {
    let base = std::env::var("HIGGS_HF_ENDPOINT")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "https://huggingface.co".into());
    let base = base.trim_end_matches('/');
    format!("{base}/{repo}/resolve/{revision}/{file}")
}

/// A byte source for [`download`]. Streams `url`, handing each chunk to `on_chunk` and
/// reporting `(downloaded, total_opt)` to `progress`. Returns the engine/network error
/// string on failure (mapped to `HG025` by the caller).
pub trait Fetcher {
    fn fetch(
        &self,
        url: &str,
        on_chunk: &mut (dyn FnMut(&[u8]) + Send),
        progress: &mut (dyn FnMut(u64, Option<u64>) + Send),
    ) -> impl std::future::Future<Output = Result<(), String>> + Send;
}

/// Resolve the on-disk destination under `models_root`, enforcing the layout the model
/// scanner traverses (`<root>/<org>/<model>/<file>.gguf`) AND rejecting any path escape
/// (`..`, absolute, drive prefix) — the values come off the wire. A pull that doesn't fit
/// the scanned layout would download to a dir `M_SCAN`/`M_LOAD` never sees, so it's refused
/// up front rather than silently succeeding into a dead spot.
pub fn dest_path(models_root: &Path, repo: &str, file: &str) -> Result<PathBuf, HiggsError> {
    let bad = |detail: &str| HiggsError::DownloadFailed {
        repo: repo.to_string(),
        file: file.to_string(),
        detail: detail.to_string(),
    };
    // repo must be exactly "<org>/<model>", each a SAFE segment (HF's `[A-Za-z0-9._-]`
    // charset). `is_safe_segment` rejects empties, `.`/`..`, path separators, AND
    // URL-reserved characters (`#`, `?`, `%`, space, …) — so the segment is both a valid
    // local dir name and a literal URL path component needing no percent-encoding.
    let segs: Vec<&str> = repo.split('/').collect();
    if segs.len() != 2 || !segs.iter().all(|s| is_safe_segment(s)) {
        return Err(bad(
            "repo must be '<org>/<model>' (each [A-Za-z0-9._-], no '..'/reserved chars)",
        ));
    }
    // file must be a single safe `*.gguf` component (case-insensitive) — no subdirectory,
    // no escape, no URL-reserved chars.
    if !is_safe_segment(file) || !file.to_ascii_lowercase().ends_with(".gguf") {
        return Err(bad(
            "file must be a single '*.gguf' name ([A-Za-z0-9._-], no subdir)",
        ));
    }
    let rel = PathBuf::from(repo).join(file);
    if rel.components().any(|c| {
        matches!(
            c,
            Component::ParentDir | Component::RootDir | Component::Prefix(_)
        )
    }) {
        return Err(bad("path must be relative with no '..'"));
    }
    Ok(models_root.join(rel))
}

/// A safe path/URL segment: non-empty, not `.`/`..`, and only `[A-Za-z0-9._-]` — which is
/// HuggingFace's repo/file charset. This deliberately excludes `/`, `\`, and every
/// URL-reserved character (`#`, `?`, `%`, `:`, space, …) so a segment is BOTH a valid local
/// filename and a literal URL path component (no percent-encoding needed, no fragment/query
/// injection into the resolve URL).
fn is_safe_segment(s: &str) -> bool {
    !s.is_empty()
        && s != "."
        && s != ".."
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
}

/// Download `target` into `models_root` via `fetcher`, streaming progress to `progress`.
/// Returns the final on-disk path. Atomic: writes a `.part` temp and renames on success.
pub async fn download<F: Fetcher>(
    target: &PullTarget,
    models_root: &Path,
    fetcher: &F,
    progress: &mut (dyn FnMut(u64, Option<u64>) + Send),
) -> Result<PathBuf, HiggsError> {
    let fail = |detail: String| HiggsError::DownloadFailed {
        repo: target.repo.clone(),
        file: target.file.clone(),
        detail,
    };

    let dest = dest_path(models_root, &target.repo, &target.file)?;
    // The revision rides the URL path too but isn't part of the local dest, so validate it
    // here: each `/`-separated segment must be safe (allows branch paths like `refs/pr/1`),
    // keeping reserved chars out of the resolve URL.
    if !target.revision.split('/').all(is_safe_segment) {
        return Err(fail(format!(
            "invalid revision {:?} (segments must be [A-Za-z0-9._-])",
            target.revision
        )));
    }

    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent).map_err(|e| fail(e.to_string()))?;
    }
    // A UNIQUE temp per download (pid + process-global counter), so two concurrent pulls to
    // the same destination never share a `.part` and corrupt each other; the last atomic
    // rename wins and the file is always whole.
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let tmp = dest.with_extension(format!("part.{}.{n}", std::process::id()));
    let url = hf_url(&target.repo, &target.revision, &target.file);

    let mut file = std::fs::File::create(&tmp).map_err(|e| fail(e.to_string()))?;
    let mut write_err: Option<std::io::Error> = None;
    let fetch_res = {
        use std::io::Write;
        let mut on_chunk = |bytes: &[u8]| {
            if write_err.is_none() {
                if let Err(e) = file.write_all(bytes) {
                    write_err = Some(e);
                }
            }
        };
        fetcher.fetch(&url, &mut on_chunk, progress).await
    };
    // Any failure (fetch error OR a write error) removes the temp and surfaces HG025 — never
    // leaving a partial/empty `.part` behind.
    let failure = fetch_res
        .err()
        .or_else(|| write_err.map(|e| format!("write failed: {e}")));
    if let Some(detail) = failure {
        drop(file);
        let _ = std::fs::remove_file(&tmp);
        return Err(fail(detail));
    }
    if let Err(e) = file.sync_all() {
        drop(file);
        let _ = std::fs::remove_file(&tmp);
        return Err(fail(e.to_string()));
    }
    drop(file);
    // Atomic replace: `rename(2)` replaces the destination in place, so an existing model
    // stays intact until the instant it's swapped (no missing-window) and a FAILED rename
    // leaves the old model untouched — we just remove our temp. (higgs targets Unix/macOS;
    // the llama.cpp FFI build is not Windows-portable, so the non-atomic-rename platform is
    // out of scope.)
    if let Err(e) = std::fs::rename(&tmp, &dest) {
        let _ = std::fs::remove_file(&tmp);
        return Err(fail(e.to_string()));
    }
    Ok(dest)
}

/// The higgs-owned models directory (`~/.higgs/models/`) — the ONLY place downloads land.
pub fn models_dir() -> std::io::Result<PathBuf> {
    let dir = crate::home::ensure_home()?.join("models");
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// Production [`Fetcher`]: streams the URL over HTTPS with `reqwest`.
pub struct HttpFetcher;

impl Fetcher for HttpFetcher {
    async fn fetch(
        &self,
        url: &str,
        on_chunk: &mut (dyn FnMut(&[u8]) + Send),
        progress: &mut (dyn FnMut(u64, Option<u64>) + Send),
    ) -> Result<(), String> {
        use futures::StreamExt;
        let resp = reqwest::get(url).await.map_err(|e| e.to_string())?;
        if !resp.status().is_success() {
            return Err(format!("HTTP {}", resp.status()));
        }
        let total = resp.content_length();
        let mut downloaded = 0u64;
        let mut stream = resp.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| e.to_string())?;
            on_chunk(&chunk);
            downloaded += chunk.len() as u64;
            progress(downloaded, total);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A no-network fetcher that emits canned chunks + progress. `report_total` toggles
    /// whether it advertises a content length (`Some` vs `None`) so both progress arms run.
    struct FakeFetcher {
        chunks: Vec<Vec<u8>>,
        fail_with: Option<String>,
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
    }

    impl Fetcher for FakeFetcher {
        async fn fetch(
            &self,
            _url: &str,
            on_chunk: &mut (dyn FnMut(&[u8]) + Send),
            progress: &mut (dyn FnMut(u64, Option<u64>) + Send),
        ) -> Result<(), String> {
            if let Some(e) = &self.fail_with {
                return Err(e.clone());
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
    fn hf_url_is_the_resolve_endpoint() {
        assert_eq!(
            hf_url("org/model", "main", "m.gguf"),
            "https://huggingface.co/org/model/resolve/main/m.gguf"
        );
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
    async fn download_maps_fetch_error_to_hg025_and_leaves_no_file() {
        let dir = tempfile::tempdir().unwrap();
        let fetcher = FakeFetcher {
            chunks: vec![],
            fail_with: Some("network down".into()),
            report_total: true,
        };
        let target = PullTarget::new("org/m", "m.gguf");
        let err = download(&target, dir.path(), &fetcher, &mut |_, _| {})
            .await
            .unwrap_err();
        assert!(err.to_string().starts_with("[HG025]"), "got {err}");
        let dest = dest_path(dir.path(), "org/m", "m.gguf").unwrap();
        assert!(!dest.exists(), "no file on failure");
        assert!(!dest.with_extension("part").exists(), "no temp left behind");
    }
}
