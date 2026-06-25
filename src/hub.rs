//! HuggingFace Hub client — the PRIMARY path for fetching repo files (model
//! GGUFs via [`HubFetcher`], and the card / `generation_config.json` via
//! [`fetch_bytes`]). The hand-rolled `reqwest` path in [`crate::download`] is the
//! fail-open FALLBACK: every primary failure is classified into a distinct
//! `HGxxx` code (see [`classify_hf`]), and a fetch that exhausts BOTH paths
//! surfaces [`HiggsError::HubFetchExhausted`] (`HG036`) carrying both diagnoses.
//!
//! The crate (`huggingface-hub`) is pre-1.0 and pulls its own `reqwest 0.13`; we
//! confine every mention of it to this module so the rest of higgs sees only
//! `HiggsError` + `Vec<u8>`.

use futures::StreamExt;
use huggingface_hub::{
    HFClient, HFError, RepoDownloadFileStreamParams, RepoDownloadFileToBytesParams, RepoType,
};

use crate::diagnostic::HiggsError;
use crate::download::{Fetcher, PullTarget};

/// Split an `org/model` repo id into `(owner, name)` for the hub `repo()` call.
/// Returns `None` for anything that isn't exactly two non-empty segments.
fn split_repo(repo_id: &str) -> Option<(&str, &str)> {
    let (owner, name) = repo_id.split_once('/')?;
    (!owner.is_empty() && !name.is_empty() && !name.contains('/')).then_some((owner, name))
}

/// Classify a `huggingface-hub` [`HFError`] into the matching distinct higgs
/// diagnostic code, so an operator can tell auth from not-found from a network
/// blip. `repo`/`file` provide context for the codes that carry them. The catch-all
/// (`HG035`) covers the crate's parse/config/internal variants (and any future
/// variant the pre-1.0 crate adds).
pub(crate) fn classify_hf(repo: &str, file: &str, err: &HFError) -> HiggsError {
    let detail = err.to_string();
    match err {
        HFError::AuthRequired | HFError::Forbidden => HiggsError::HubAuthFailed {
            repo: repo.to_owned(),
            detail,
        },
        HFError::RepoNotFound { .. } => HiggsError::HubResourceNotFound {
            repo: repo.to_owned(),
            resource: "repo".to_owned(),
            detail,
        },
        HFError::RevisionNotFound { revision, .. } => HiggsError::HubResourceNotFound {
            repo: repo.to_owned(),
            resource: format!("revision {revision}"),
            detail,
        },
        HFError::EntryNotFound { path, .. } | HFError::LocalEntryNotFound { path } => {
            HiggsError::HubResourceNotFound {
                repo: repo.to_owned(),
                resource: format!("file {path}"),
                detail,
            }
        }
        HFError::RateLimited => HiggsError::HubRateLimited {
            repo: repo.to_owned(),
            detail,
        },
        HFError::Http { status, .. } => http_status_to_error(repo, file, status.as_u16(), detail),
        // reqwest / middleware transport faults (DNS, connect, TLS, timeout).
        HFError::Request(_) | HFError::Middleware(_) => HiggsError::HubTransport {
            repo: repo.to_owned(),
            detail,
        },
        // Filesystem error writing into the hub cache / target dir.
        HFError::Io(_) => HiggsError::HubFileWrite {
            repo: repo.to_owned(),
            file: file.to_owned(),
            detail,
        },
        // Everything else (JSON/diff parse, bad URL/param, cache config, internal).
        _ => HiggsError::HubClient {
            repo: repo.to_owned(),
            detail,
        },
    }
}

/// Route an HTTP status code to its distinct higgs code: 401/403 → auth (`HG029`),
/// 404 → not-found (`HG030`), 429 → rate-limited (`HG031`), anything else →
/// `HG032`. Pure over the numeric status so it is unit-testable without
/// constructing the crate's (reqwest 0.13) `StatusCode`. Shared by the hub-error
/// classifier and the `reqwest` fallback (which sees a 0.12 status).
pub(crate) fn http_status_to_error(
    repo: &str,
    file: &str,
    status: u16,
    detail: String,
) -> HiggsError {
    match status {
        401 | 403 => HiggsError::HubAuthFailed {
            repo: repo.to_owned(),
            detail,
        },
        404 => HiggsError::HubResourceNotFound {
            repo: repo.to_owned(),
            resource: format!("file {file}"),
            detail,
        },
        429 => HiggsError::HubRateLimited {
            repo: repo.to_owned(),
            detail,
        },
        other => HiggsError::HubHttpStatus {
            repo: repo.to_owned(),
            status: other,
            detail,
        },
    }
}

/// Build an [`HFClient`] honoring higgs's `HIGGS_HF_ENDPOINT` override (a mirror,
/// an enterprise proxy, or a test server) — the SAME override [`crate::download::hf_url`]
/// applies to the fallback path, so primary and fallback always target the same host
/// (otherwise a user's mirror would be silently bypassed by the hub client). Unset ⇒ the
/// crate's default (`https://huggingface.co`), and its own `HF_ENDPOINT`/token resolution.
fn hf_client(repo_id: &str) -> Result<HFClient, HiggsError> {
    let mut builder = HFClient::builder();
    if let Ok(ep) = std::env::var("HIGGS_HF_ENDPOINT") {
        if !ep.is_empty() {
            builder = builder.endpoint(ep);
        }
    }
    builder.build().map_err(|e| classify_hf(repo_id, "", &e))
}

/// Construct a hub repo handle for `repo_id` (a model repo). Centralizes the
/// client build + `split_repo` + `repo()` dance and its error classification.
fn model_repo(repo_id: &str) -> Result<huggingface_hub::HFRepository, HiggsError> {
    let (owner, name) = split_repo(repo_id).ok_or_else(|| HiggsError::HubClient {
        repo: repo_id.to_owned(),
        detail: format!("repo id must be 'org/model', got {repo_id:?}"),
    })?;
    let client = hf_client(repo_id)?;
    Ok(client.repo(RepoType::Model, owner, name))
}

/// Fetch one file from a model repo into memory via the hub client (PRIMARY),
/// falling back to a direct `reqwest` GET of the resolve URL (FALLBACK). Used for
/// small card/config files (`README.md`, `generation_config.json`) — NOT for
/// multi-GB GGUFs (those stream to disk via [`HubFetcher`] + [`crate::download`]).
///
/// Both paths exhausted ⇒ [`HiggsError::HubFetchExhausted`] (`HG036`) carrying both
/// diagnoses. Callers that are best-effort (the card fetch) map any `Err` to "no
/// recommendation"; callers that require the bytes surface the coded error.
pub async fn fetch_bytes(repo_id: &str, file: &str) -> Result<Vec<u8>, HiggsError> {
    let primary = hub_fetch_bytes(repo_id, file).await;
    match primary {
        Ok(bytes) => Ok(bytes),
        Err(primary_err) => match reqwest_fetch_bytes(repo_id, file).await {
            Ok(bytes) => {
                tracing::warn!(
                    repo = repo_id,
                    file,
                    error = %primary_err,
                    "higgs: hub client fetch failed; reqwest fallback succeeded"
                );
                Ok(bytes)
            }
            Err(fallback_err) => Err(HiggsError::HubFetchExhausted {
                repo: repo_id.to_owned(),
                file: file.to_owned(),
                primary: primary_err.to_string(),
                fallback: fallback_err,
            }),
        },
    }
}

/// PRIMARY in-memory fetch via `download_file_to_bytes`.
async fn hub_fetch_bytes(repo_id: &str, file: &str) -> Result<Vec<u8>, HiggsError> {
    let repo = model_repo(repo_id)?;
    let bytes = repo
        .download_file_to_bytes(
            &RepoDownloadFileToBytesParams::builder()
                .filename(file)
                .build(),
        )
        .await
        .map_err(|e| classify_hf(repo_id, file, &e))?;
    Ok(bytes.to_vec())
}

/// FALLBACK in-memory fetch: a direct `reqwest` GET of the `resolve/main` URL.
/// Returns a plain detail string (the caller wraps it into `HubFetchExhausted`).
async fn reqwest_fetch_bytes(repo_id: &str, file: &str) -> Result<Vec<u8>, String> {
    let url = crate::download::hf_url(repo_id, "main", file);
    let resp = reqwest::get(&url).await.map_err(|e| e.to_string())?;
    if !resp.status().is_success() {
        return Err(format!("HTTP {}", resp.status()));
    }
    let bytes = resp.bytes().await.map_err(|e| e.to_string())?;
    Ok(bytes.to_vec())
}

/// Production PRIMARY [`Fetcher`]: streams a repo file via the hub client's
/// `download_file_stream`, handing each chunk to `on_chunk` and reporting
/// `(downloaded, total)` to `progress`. Errors are classified into distinct
/// `HGxxx` codes. Composed with the `reqwest` [`crate::download::HttpFetcher`]
/// fallback by [`crate::download::download_dual`].
pub struct HubFetcher;

impl Fetcher for HubFetcher {
    async fn fetch(
        &self,
        target: &PullTarget,
        on_chunk: &mut (dyn FnMut(&[u8]) + Send),
        progress: &mut (dyn FnMut(u64, Option<u64>) + Send),
    ) -> Result<(), HiggsError> {
        let repo = model_repo(&target.repo)?;
        let (total, mut stream) = repo
            .download_file_stream(
                &RepoDownloadFileStreamParams::builder()
                    .filename(target.file.clone())
                    .revision(target.revision.clone())
                    .build(),
            )
            .await
            .map_err(|e| classify_hf(&target.repo, &target.file, &e))?;
        let mut downloaded = 0u64;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| classify_hf(&target.repo, &target.file, &e))?;
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

    #[test]
    fn split_repo_requires_two_segments() {
        assert_eq!(split_repo("org/model"), Some(("org", "model")));
        assert_eq!(
            split_repo("bartowski/Qwen2.5-GGUF"),
            Some(("bartowski", "Qwen2.5-GGUF"))
        );
        assert_eq!(split_repo("gpt2"), None, "single segment");
        assert_eq!(split_repo("a/b/c"), None, "three segments");
        assert_eq!(split_repo("/model"), None, "empty owner");
        assert_eq!(split_repo("org/"), None, "empty name");
    }

    #[test]
    fn classify_maps_each_hf_error_to_its_code() {
        let c = |e: &HFError| classify_hf("org/m", "m.gguf", e).to_string();
        assert!(c(&HFError::AuthRequired).starts_with("[HG029]"));
        assert!(c(&HFError::Forbidden).starts_with("[HG029]"));
        assert!(c(&HFError::RepoNotFound {
            repo_id: "org/m".into()
        })
        .starts_with("[HG030]"));
        assert!(c(&HFError::RevisionNotFound {
            repo_id: "org/m".into(),
            revision: "main".into()
        })
        .starts_with("[HG030]"));
        assert!(c(&HFError::EntryNotFound {
            path: "m.gguf".into(),
            repo_id: "org/m".into()
        })
        .starts_with("[HG030]"));
        assert!(c(&HFError::RateLimited).starts_with("[HG031]"));
        assert!(c(&HFError::Other("boom".into())).starts_with("[HG035]"));
        assert!(c(&HFError::Io(std::io::Error::other("disk"))).starts_with("[HG034]"));
    }

    #[test]
    fn http_status_routes_to_distinct_codes() {
        // 401/403→auth(029), 404→not-found(030), 429→rate-limit(031), else→032.
        let h =
            |s: u16| http_status_to_error("org/m", "m.gguf", s, format!("HTTP {s}")).to_string();
        assert!(h(401).starts_with("[HG029]"));
        assert!(h(403).starts_with("[HG029]"));
        assert!(h(404).starts_with("[HG030]"));
        assert!(h(429).starts_with("[HG031]"));
        assert!(h(500).starts_with("[HG032]"));
        assert!(h(503).starts_with("[HG032]"));
    }
}
