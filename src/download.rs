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

/// A byte source for [`download`]. Streams the bytes of `target`, handing each chunk to
/// `on_chunk` and reporting `(downloaded, total_opt)` to `progress`. Returns a CLASSIFIED
/// [`HiggsError`] on failure (the specific `HG029`–`HG035` code), which [`download`]
/// propagates verbatim — so a 404 vs an auth refusal vs a network blip stays distinguishable
/// all the way up. Implementors resolve their own endpoint from `target` (the hub client from
/// the repo handle; [`HttpFetcher`] from the `resolve` URL).
pub trait Fetcher {
    fn fetch(
        &self,
        target: &PullTarget,
        on_chunk: &mut (dyn FnMut(&[u8]) + Send),
        progress: &mut (dyn FnMut(u64, Option<u64>) + Send),
    ) -> impl std::future::Future<Output = Result<(), HiggsError>> + Send;
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
pub(crate) fn is_safe_segment(s: &str) -> bool {
    !s.is_empty()
        && s != "."
        && s != ".."
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
}

/// Where a [`download`] attempt failed, by SOURCE — so [`download_dual`] can decide whether a
/// fallback retry could possibly help. The DISTINCTION IS THE SOURCE, NOT THE ERROR VARIANT:
/// the `HG034` (`HubFileWrite`) code is overloaded — `download`'s OWN write into
/// `~/.higgs/models` is `Local` (deterministic; a retry fails identically), but the hub
/// client's cache I/O surfaces the SAME code through the fetcher and is `Fetcher` (the direct
/// `reqwest` fallback bypasses that cache and may still succeed). A variant-only heuristic
/// can't tell those apart; this enum carries the source down from where it's known.
enum AttemptError {
    /// `download`'s own pre/around-fetch failure (wire-validation `HG025`, or a local
    /// filesystem `HG034`). Deterministic — `download_dual` returns it verbatim, no retry.
    Local(HiggsError),
    /// The fetcher failed (classified `HG029`–`HG035`, incl. the hub client's own I/O).
    /// Eligible for the dual-path fallback.
    Fetcher(HiggsError),
}

impl AttemptError {
    /// The underlying error, discarding the source tag (for the single-fetcher `download`).
    fn into_inner(self) -> HiggsError {
        match self {
            AttemptError::Local(e) | AttemptError::Fetcher(e) => e,
        }
    }
}

/// Download `target` into `models_root` via a SINGLE `fetcher`, streaming progress to
/// `progress`. Returns the final on-disk path. Atomic + self-contained: writes its OWN
/// unique `.part` temp and renames on success, removing the temp on ANY failure — so it is
/// safe to call repeatedly (the dual-path retry in [`download_dual`] just calls it twice).
///
/// Error mapping: wire-validation (bad repo/file/revision) stays `HG025`; the fetcher's
/// CLASSIFIED error (`HG029`–`HG035`) is propagated verbatim; a local filesystem error
/// (temp create / write / fsync / rename) is `HG034` (`HubFileWrite`).
pub async fn download<F: Fetcher>(
    target: &PullTarget,
    models_root: &Path,
    fetcher: &F,
    progress: &mut (dyn FnMut(u64, Option<u64>) + Send),
) -> Result<PathBuf, HiggsError> {
    download_attempt(target, models_root, fetcher, progress)
        .await
        .map_err(AttemptError::into_inner)
}

/// [`download`] but tagging each failure with its [`AttemptError`] source, for [`download_dual`].
async fn download_attempt<F: Fetcher>(
    target: &PullTarget,
    models_root: &Path,
    fetcher: &F,
    progress: &mut (dyn FnMut(u64, Option<u64>) + Send),
) -> Result<PathBuf, AttemptError> {
    // Wire-validation failures (bad repo/file/revision shape) — HG025, always LOCAL.
    let bad_request = |detail: String| {
        AttemptError::Local(HiggsError::DownloadFailed {
            repo: target.repo.clone(),
            file: target.file.clone(),
            detail,
        })
    };
    // OUR OWN filesystem failures (temp / write / fsync / rename into ~/.higgs/models) — HG034,
    // always LOCAL (a fallback's write to the same dir fails identically).
    let fs_fail = |detail: String| {
        AttemptError::Local(HiggsError::HubFileWrite {
            repo: target.repo.clone(),
            file: target.file.clone(),
            detail,
        })
    };

    let dest = dest_path(models_root, &target.repo, &target.file).map_err(AttemptError::Local)?;
    // The revision rides the URL path too but isn't part of the local dest, so validate it
    // here: each `/`-separated segment must be safe (allows branch paths like `refs/pr/1`),
    // keeping reserved chars out of the resolve URL.
    if !target.revision.split('/').all(is_safe_segment) {
        return Err(bad_request(format!(
            "invalid revision {:?} (segments must be [A-Za-z0-9._-])",
            target.revision
        )));
    }

    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent).map_err(|e| fs_fail(e.to_string()))?;
    }
    // A UNIQUE temp per download (pid + process-global counter), so two concurrent pulls to
    // the same destination never share a `.part` and corrupt each other; the last atomic
    // rename wins and the file is always whole.
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let tmp = dest.with_extension(format!("part.{}.{n}", std::process::id()));

    let mut file = std::fs::File::create(&tmp).map_err(|e| fs_fail(e.to_string()))?;
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
        fetcher.fetch(target, &mut on_chunk, progress).await
    };
    // A fetch failure (a classified HiggsError) is a FETCHER error — propagated verbatim and
    // eligible for the fallback. A local write failure is HG034 LOCAL. Either removes the temp
    // — never leaving a partial/empty `.part` behind.
    if let Err(e) = fetch_res {
        drop(file);
        let _ = std::fs::remove_file(&tmp);
        return Err(AttemptError::Fetcher(e));
    }
    if let Some(e) = write_err {
        drop(file);
        let _ = std::fs::remove_file(&tmp);
        return Err(fs_fail(format!("write failed: {e}")));
    }
    if let Err(e) = file.sync_all() {
        drop(file);
        let _ = std::fs::remove_file(&tmp);
        return Err(fs_fail(e.to_string()));
    }
    drop(file);
    // Atomic replace: `rename(2)` replaces the destination in place, so an existing model
    // stays intact until the instant it's swapped (no missing-window) and a FAILED rename
    // leaves the old model untouched — we just remove our temp. (higgs targets Unix/macOS;
    // the llama.cpp FFI build is not Windows-portable, so the non-atomic-rename platform is
    // out of scope.)
    if let Err(e) = std::fs::rename(&tmp, &dest) {
        let _ = std::fs::remove_file(&tmp);
        return Err(fs_fail(e.to_string()));
    }
    Ok(dest)
}

/// Download `target` trying `primary` first and `fallback` only if the primary fails with a
/// FETCHER error — the hub-client-primary / `reqwest`-fallback strategy. Each attempt is a
/// self-contained [`download_attempt`] (its own temp + atomic rename), so a failed primary
/// leaves NO partial file for the fallback to trip over.
///
/// A primary failure in `download`'s OWN code ([`AttemptError::Local`] — wire-validation
/// `HG025`, or a local filesystem `HG034` writing to `~/.higgs/models`) is deterministic: the
/// fallback would fail identically, so it is returned VERBATIM (preserving the actionable
/// `HG025`/`HG034` rather than a misleading `HG036`) and the fallback is NOT attempted — no
/// pointless re-download of a multi-GB model. A FETCHER failure ([`AttemptError::Fetcher`] —
/// network/auth/404/http, or the hub client's own cache I/O) DOES trigger the fallback; if
/// that also fails, returns [`HiggsError::HubFetchExhausted`] (`HG036`) carrying both diagnoses.
pub async fn download_dual<P: Fetcher, F: Fetcher>(
    target: &PullTarget,
    models_root: &Path,
    primary: &P,
    fallback: &F,
    progress: &mut (dyn FnMut(u64, Option<u64>) + Send),
) -> Result<PathBuf, HiggsError> {
    match download_attempt(target, models_root, primary, progress).await {
        Ok(path) => Ok(path),
        // Deterministic local failure (bad input / our own fs write) — surface it verbatim.
        Err(AttemptError::Local(e)) => Err(e),
        // A fetcher failure could be transport-specific; the other transport may succeed.
        Err(AttemptError::Fetcher(primary_err)) => {
            tracing::warn!(
                repo = %target.repo,
                file = %target.file,
                error = %primary_err,
                "higgs: hub download failed; trying reqwest fallback"
            );
            match download_attempt(target, models_root, fallback, progress).await {
                Ok(path) => Ok(path),
                Err(fallback_err) => Err(HiggsError::HubFetchExhausted {
                    repo: target.repo.clone(),
                    file: target.file.clone(),
                    primary: primary_err.to_string(),
                    fallback: fallback_err.into_inner().to_string(),
                }),
            }
        }
    }
}

/// The higgs-owned models directory (`~/.higgs/models/`) — the ONLY place downloads land.
pub fn models_dir() -> std::io::Result<PathBuf> {
    let dir = crate::home::ensure_home()?.join("models");
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// The FALLBACK [`Fetcher`]: streams the `resolve` URL over HTTPS with `reqwest` (the
/// hand-rolled path, used when the hub client primary fails). Classifies its failures into
/// the same distinct codes the hub path uses — a non-success status routes through
/// [`crate::hub::http_status_to_error`] (404→`HG030`, 401/403→`HG029`, 429→`HG031`, else
/// `HG032`); a transport error is `HG033` (`HubTransport`).
pub struct HttpFetcher;

impl Fetcher for HttpFetcher {
    async fn fetch(
        &self,
        target: &PullTarget,
        on_chunk: &mut (dyn FnMut(&[u8]) + Send),
        progress: &mut (dyn FnMut(u64, Option<u64>) + Send),
    ) -> Result<(), HiggsError> {
        use futures::StreamExt;
        let transport = |detail: String| HiggsError::HubTransport {
            repo: target.repo.clone(),
            detail,
        };
        let url = hf_url(&target.repo, &target.revision, &target.file);
        let resp = reqwest::get(&url)
            .await
            .map_err(|e| transport(e.to_string()))?;
        if !resp.status().is_success() {
            let status = resp.status();
            return Err(crate::hub::http_status_to_error(
                &target.repo,
                &target.file,
                status.as_u16(),
                format!("HTTP {status}"),
            ));
        }
        let total = resp.content_length();
        let mut downloaded = 0u64;
        let mut stream = resp.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| transport(e.to_string()))?;
            on_chunk(&chunk);
            downloaded += chunk.len() as u64;
            progress(downloaded, total);
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "download_tests.rs"]
mod tests;
