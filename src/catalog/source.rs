//! The catalog's Hub transport — the [`CatalogSource`] seam and its ONE
//! production implementation, [`HfSource`], on the pinned `huggingface-hub`
//! crate (the same `HFClient` + `HIGGS_HF_ENDPOINT` override `crate::hub`
//! uses). The seam exists so [`crate::catalog::service`] is testable with an
//! injected fake; nothing else implements it in production. Failures classify
//! into the existing `HG029`–`HG035` codes via [`crate::hub::classify_hf`].

use std::collections::HashMap;

use futures::StreamExt;
use huggingface_hub::{
    ListModelsParams, ModelInfo, RepoDownloadFileStreamParams, RepoGetPathsInfoParams, RepoInfo,
    RepoInfoParams, RepoTreeEntry,
};

use crate::catalog::wire::{CatalogQuery, CatalogSort, DEFAULT_SEARCH_LIMIT, MAX_SEARCH_LIMIT};
use crate::diagnostic::HiggsError;
use crate::hub::classify_hf;

/// Byte cap on a fetched README — the stream is stopped once this much has
/// arrived, so a pathological repo cannot balloon a detail response.
pub const README_MAX_BYTES: usize = 1024 * 1024;

/// The Hub reads the catalog needs, as a seam: production is [`HfSource`],
/// tests inject canned data. Every method classifies its failure into the
/// existing `HGxxx` hub codes.
pub trait CatalogSource {
    /// Search model repos (`GET /api/models` semantics): free text and/or
    /// author, GGUF-only, sorted, capped.
    fn search(
        &self,
        q: &CatalogQuery,
    ) -> impl std::future::Future<Output = Result<Vec<ModelInfo>, HiggsError>> + Send;

    /// One repo's full info (`GET /api/models/{repo}` semantics), including
    /// siblings and the `gguf` block when the Hub reports them.
    fn info(
        &self,
        repo: &str,
    ) -> impl std::future::Future<Output = Result<ModelInfo, HiggsError>> + Send;

    /// Byte sizes for specific repo files (`paths-info` semantics). Missing
    /// paths are simply absent from the map.
    fn file_sizes(
        &self,
        repo: &str,
        files: &[String],
    ) -> impl std::future::Future<Output = Result<HashMap<String, u64>, HiggsError>> + Send;

    /// The repo README (markdown), bounded to [`README_MAX_BYTES`].
    /// `Ok(None)` when the repo has none — only a real failure is an `Err`.
    fn readme(
        &self,
        repo: &str,
    ) -> impl std::future::Future<Output = Result<Option<String>, HiggsError>> + Send;
}

/// The Hub API's sort key for a catalog sort order.
pub(crate) fn hub_sort(sort: CatalogSort) -> &'static str {
    match sort {
        CatalogSort::Downloads => "downloads",
        CatalogSort::Likes => "likes",
        CatalogSort::Updated => "lastModified",
        CatalogSort::Trending => "trendingScore",
    }
}

/// The row count actually requested: `0` means the default page, and nothing
/// exceeds [`MAX_SEARCH_LIMIT`].
pub(crate) fn effective_limit(limit: u32) -> usize {
    let limit = if limit == 0 {
        DEFAULT_SEARCH_LIMIT
    } else {
        limit
    };
    limit.min(MAX_SEARCH_LIMIT) as usize
}

/// Bytes → `String`, cut to at most `max` bytes on a char boundary (lossy on
/// invalid UTF-8). The README bound.
pub(crate) fn clip_utf8(bytes: &[u8], max: usize) -> String {
    let s = String::from_utf8_lossy(bytes);
    let mut end = max.min(s.len());
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    s[..end].to_string()
}

/// Flatten `paths-info` entries to `path → byte size`, preferring the LFS
/// real size over the entry size (for LFS files the entry size is the pointer
/// blob, not the payload). Directories carry no size.
pub(crate) fn sizes_from_entries(entries: &[RepoTreeEntry]) -> HashMap<String, u64> {
    entries
        .iter()
        .filter_map(|e| match e {
            RepoTreeEntry::File {
                path, size, lfs, ..
            } => {
                let real = lfs.as_ref().and_then(|l| l.size).unwrap_or(*size);
                Some((path.clone(), real))
            }
            RepoTreeEntry::Directory { .. } => None,
        })
        .collect()
}

/// Refuse a repo id that is not exactly two safe segments (`org/model`,
/// `[A-Za-z0-9._-]` each) BEFORE it reaches the Hub client — the crate
/// interpolates the id into the request PATH, so a `?`/`#`/space/`..` would
/// otherwise alter the URL the read actually hits. The same charset
/// [`crate::download::dest_path`] enforces on downloads, applied to reads.
fn validated_repo(repo: &str) -> Result<&str, HiggsError> {
    let ok = matches!(
        repo.split('/').collect::<Vec<_>>().as_slice(),
        [org, model] if crate::download::is_safe_segment(org)
            && crate::download::is_safe_segment(model)
    );
    if ok {
        Ok(repo)
    } else {
        Err(HiggsError::HubClient {
            repo: repo.to_owned(),
            detail: "repo id must be 'org/model' ([A-Za-z0-9._-] each)".to_owned(),
        })
    }
}

/// The production [`CatalogSource`]: the pinned `huggingface-hub` crate,
/// honoring `HIGGS_HF_ENDPOINT` exactly like every other higgs Hub touch.
pub struct HfSource;

impl CatalogSource for HfSource {
    async fn search(&self, q: &CatalogQuery) -> Result<Vec<ModelInfo>, HiggsError> {
        let context = q.author.as_deref().unwrap_or(&q.search);
        let client = crate::hub::hf_client(context)?;
        let params = ListModelsParams {
            search: (!q.search.is_empty()).then(|| q.search.clone()),
            author: q.author.clone(),
            // higgs serves GGUF only, so the catalog only ever asks for it.
            filter: Some("gguf".to_owned()),
            sort: Some(hub_sort(q.sort.unwrap_or_default()).to_owned()),
            pipeline_tag: None,
            // `full` adds lastModified + siblings to list rows.
            full: Some(true),
            card_data: None,
            fetch_config: None,
            limit: Some(effective_limit(q.limit.unwrap_or(0))),
        };
        let stream = client
            .list_models(&params)
            .map_err(|e| classify_hf(context, "", &e))?;
        let mut hits = Vec::new();
        let mut stream = std::pin::pin!(stream);
        while let Some(item) = stream.next().await {
            hits.push(item.map_err(|e| classify_hf(context, "", &e))?);
        }
        Ok(hits)
    }

    async fn info(&self, repo: &str) -> Result<ModelInfo, HiggsError> {
        let handle = crate::hub::model_repo(validated_repo(repo)?)?;
        let info = handle
            .info(&RepoInfoParams::builder().build())
            .await
            .map_err(|e| classify_hf(repo, "", &e))?;
        match info {
            RepoInfo::Model(m) => Ok(m),
            // Unreachable for a `RepoType::Model` handle; classified rather
            // than panicking in case the pre-1.0 crate changes shape.
            _ => Err(HiggsError::HubClient {
                repo: repo.to_owned(),
                detail: "hub returned non-model info for a model repo".to_owned(),
            }),
        }
    }

    async fn file_sizes(
        &self,
        repo: &str,
        files: &[String],
    ) -> Result<HashMap<String, u64>, HiggsError> {
        let repo = validated_repo(repo)?;
        if files.is_empty() {
            return Ok(HashMap::new());
        }
        let handle = crate::hub::model_repo(repo)?;
        let entries = handle
            .get_paths_info(
                &RepoGetPathsInfoParams::builder()
                    .paths(files.to_vec())
                    .build(),
            )
            .await
            .map_err(|e| classify_hf(repo, "", &e))?;
        Ok(sizes_from_entries(&entries))
    }

    async fn readme(&self, repo: &str) -> Result<Option<String>, HiggsError> {
        const README: &str = "README.md";
        let handle = crate::hub::model_repo(validated_repo(repo)?)?;
        let res = handle
            .download_file_stream(
                &RepoDownloadFileStreamParams::builder()
                    .filename(README)
                    .build(),
            )
            .await;
        let (_total, mut stream) = match res {
            Ok(s) => s,
            Err(e) => {
                return match classify_hf(repo, README, &e) {
                    // No README is a normal repo state, not a failure.
                    HiggsError::HubResourceNotFound { .. } => Ok(None),
                    other => Err(other),
                };
            }
        };
        let mut bytes: Vec<u8> = Vec::new();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| classify_hf(repo, README, &e))?;
            // Take only what fits — the bound holds on ALLOCATION, not just on
            // the returned string, so one oversized chunk can't overshoot it.
            let room = README_MAX_BYTES - bytes.len();
            bytes.extend_from_slice(&chunk[..chunk.len().min(room)]);
            if bytes.len() >= README_MAX_BYTES {
                // Dropping the stream aborts the transfer.
                break;
            }
        }
        Ok(Some(clip_utf8(&bytes, README_MAX_BYTES)))
    }
}

#[cfg(test)]
#[path = "source_tests.rs"]
mod tests;
