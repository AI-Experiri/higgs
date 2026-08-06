//! Catalog assembly — pure mapping from Hub responses ([`ModelInfo`]) to the
//! catalog wire types, plus the local-inventory facts the mapping needs.
//! No I/O here: the Hub is behind [`crate::catalog::source::CatalogSource`],
//! the local disk behind [`LocalInventory`].

use std::collections::{HashMap, HashSet};
use std::path::Path;

use huggingface_hub::ModelInfo;

use crate::catalog::source::CatalogSource;
use crate::catalog::wire::{
    CatalogGgufMeta, CatalogModelDetail, CatalogModelSummary, CatalogQuant, CatalogQuery,
    CatalogSearchResponse, CatalogSort,
};
use crate::diagnostic::HiggsError;

/// Cap on "more by this author" rows in a detail response.
pub const MORE_BY_AUTHOR_LIMIT: usize = 6;

/// How many per-row detail fetches run at once while enriching search rows
/// with their `gguf` blocks (the list endpoint omits them).
const SEARCH_ENRICH_CONCURRENCY: usize = 8;

/// Ceiling on the WHOLE search-row enrichment pass. Without it a page of
/// black-holed info fetches stacks per-row timeouts (a full page at
/// [`SEARCH_ENRICH_CONCURRENCY`] × [`ENRICHMENT_TIMEOUT`] is over a minute of
/// silence); past this deadline the remaining rows go out bare instead.
const SEARCH_ENRICH_TOTAL_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(15);

/// Ceiling on each best-effort detail enrichment (file sizes, README, author
/// rows). The pinned Hub client sets no request timeout, so a black-holed
/// connection would otherwise hold the whole detail pane — with this ceiling
/// the stalled part degrades exactly like a failed one.
pub(crate) const ENRICHMENT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

/// A repo sibling the download path would actually accept: a single safe
/// `*.gguf` file name (the scanned `<org>/<model>/<file>.gguf` layout). Rows
/// the picker offers must be rows `pull` can land — a subdirectory or
/// reserved-character sibling never becomes a quant row.
fn is_downloadable_gguf(name: &str) -> bool {
    crate::download::is_safe_segment(name) && name.to_ascii_lowercase().ends_with(".gguf")
}

/// What of the catalog is already on THIS machine: the repo ids and exact
/// files present under the higgs models layout (`<root>/<org>/<model>/*.gguf`).
/// Built once per catalog op so every row's `downloaded` flag comes from the
/// same snapshot.
#[derive(Debug, Default)]
pub struct LocalInventory {
    /// Repo ids (`org/model`) with at least one local GGUF.
    repos: HashSet<String>,
    /// Exact `(repo id, file name)` pairs on disk.
    files: HashSet<(String, String)>,
}

impl LocalInventory {
    /// An inventory with nothing downloaded.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Record one local file under `repo`.
    pub fn insert(&mut self, repo: &str, file: &str) {
        self.repos.insert(repo.to_owned());
        self.files.insert((repo.to_owned(), file.to_owned()));
    }

    /// Whether any file of `repo` is local.
    pub fn has_repo(&self, repo: &str) -> bool {
        self.repos.contains(repo)
    }

    /// Whether this exact `repo`/`file` is local.
    pub fn has_file(&self, repo: &str, file: &str) -> bool {
        self.files.contains(&(repo.to_owned(), file.to_owned()))
    }

    /// Walk the higgs models layout (`<root>/<org>/<model>/*.gguf`) into an
    /// inventory. Missing/unreadable directories read as empty — the catalog
    /// still works, rows just show as not downloaded.
    pub fn from_models_dir(root: &Path) -> Self {
        let mut inv = Self::empty();
        inv.add_models_dir(root);
        inv
    }

    /// [`from_models_dir`](Self::from_models_dir) accumulating into an
    /// existing inventory — for merging several LM-Studio-layout roots.
    pub fn add_models_dir(&mut self, root: &Path) {
        let orgs = match std::fs::read_dir(root) {
            Ok(rd) => rd,
            Err(_) => return,
        };
        for org in orgs.flatten() {
            let Ok(org_name) = org.file_name().into_string() else {
                continue;
            };
            let Ok(models) = std::fs::read_dir(org.path()) else {
                continue;
            };
            for model in models.flatten() {
                let Ok(model_name) = model.file_name().into_string() else {
                    continue;
                };
                let Ok(files) = std::fs::read_dir(model.path()) else {
                    continue;
                };
                let repo = format!("{org_name}/{model_name}");
                for f in files.flatten() {
                    let Ok(fname) = f.file_name().into_string() else {
                        continue;
                    };
                    if fname.to_ascii_lowercase().ends_with(".gguf") {
                        self.insert(&repo, &fname);
                    }
                }
            }
        }
    }
}

/// Map one Hub model row to a catalog summary. The author falls back to the
/// repo id's `org/` prefix when the Hub omitted it; `downloaded` reflects the
/// local inventory; `fit` is the advisory smallest-quant estimate against
/// `vram_total_bytes` (absent inputs → no verdict).
pub fn summary_from(
    info: &ModelInfo,
    inv: &LocalInventory,
    vram_total_bytes: Option<u64>,
) -> CatalogModelSummary {
    let author = info
        .author
        .clone()
        .or_else(|| info.id.split_once('/').map(|(org, _)| org.to_owned()));
    let fit = match (estimated_min_quant_bytes(info), vram_total_bytes) {
        (Some(bytes), Some(vram)) => Some(crate::system::fits_vram(
            bytes,
            vram,
            crate::api::MEMORY_HEADROOM_FRACTION,
        )),
        _ => None,
    };
    CatalogModelSummary {
        downloaded: inv.has_repo(&info.id),
        id: info.id.clone(),
        author,
        downloads: info.downloads,
        likes: info.likes,
        updated: info.last_modified.clone(),
        pipeline: info.pipeline_tag.clone(),
        gguf: gguf_meta(info),
        fit,
    }
}

/// Effective bytes-per-parameter of a GGUF quant FAMILY — whole-file size over
/// parameter count as llama.cpp produces them (the higher-precision embedding
/// and output tensors are why e.g. `Q4_K_M` lands near 0.66, not 4.5/8).
/// Advisory input to the pre-download row estimate only; the detail pane uses
/// real file sizes. Unknown labels get no number, hence no verdict.
///
/// I-quants (`IQ2_XXS`, …) get their OWN numbers: they are materially cheaper
/// than the K-quant family of the same digit, and because `compatible_only`
/// DROPS estimated misfits, an overestimate would hide models the machine can
/// actually run — the unsafe error direction for this table.
fn quant_bytes_per_param(label: &str) -> Option<f64> {
    let l = label.to_ascii_uppercase();
    if let Some(rest) = l.strip_prefix("IQ") {
        return match rest.chars().next()? {
            '1' => Some(0.2),
            '2' => Some(0.3),
            '3' => Some(0.45),
            '4' => Some(0.55),
            _ => None,
        };
    }
    Some(match l.split(['_', '-']).next()? {
        "Q1" => 0.25,
        "Q2" => 0.42,
        "Q3" => 0.52,
        "Q4" => 0.66,
        "Q5" => 0.78,
        "Q6" => 0.89,
        "Q8" => 1.12,
        "F16" | "BF16" | "FP16" => 2.1,
        "F32" | "FP32" => 4.2,
        _ => return None,
    })
}

/// Advisory estimate of the SMALLEST downloadable quant's file size: parameter
/// count × the cheapest labeled quant family the repo ships. `None` without a
/// parameter count or a recognized quant label — the caller shows no verdict.
fn estimated_min_quant_bytes(info: &ModelInfo) -> Option<u64> {
    let params = info.gguf.as_ref()?.get("total")?.as_u64()?;
    let min_bpp = info
        .siblings
        .as_deref()
        .unwrap_or_default()
        .iter()
        .filter(|s| is_downloadable_gguf(&s.rfilename))
        .filter_map(|s| crate::worker::models::quant_from_filename(&s.rfilename))
        .filter_map(|label| quant_bytes_per_param(&label))
        .min_by(f64::total_cmp)?;
    Some((params as f64 * min_bpp) as u64)
}

/// Reduce the Hub's `gguf` block to the badge fields. `None` when the block is
/// absent or carries none of them — the UI shows a badge or nothing, never an
/// empty shell.
pub fn gguf_meta(info: &ModelInfo) -> Option<CatalogGgufMeta> {
    let block = info.gguf.as_ref()?;
    let meta = CatalogGgufMeta {
        arch: block
            .get("architecture")
            .and_then(|v| v.as_str())
            .map(str::to_owned),
        params_total: block.get("total").and_then(serde_json::Value::as_u64),
        ctx_train: block
            .get("context_length")
            .and_then(serde_json::Value::as_u64),
    };
    (meta.arch.is_some() || meta.params_total.is_some() || meta.ctx_train.is_some()).then_some(meta)
}

/// Map a repo's `.gguf` siblings to quant-picker rows: size from the
/// paths-info map first (authoritative), then the sibling's own LFS/plain
/// size; quant label parsed from the file name; `downloaded` per exact file;
/// fit verdict only when BOTH the size and the VRAM total are known (an
/// unknown never gets a fake verdict). Rows are size-ascending, unknown sizes
/// last, ties by name — a stable picker order.
pub fn quants_from(
    info: &ModelInfo,
    sizes: &HashMap<String, u64>,
    inv: &LocalInventory,
    vram_total_bytes: Option<u64>,
) -> Vec<CatalogQuant> {
    let mut quants: Vec<CatalogQuant> = info
        .siblings
        .as_deref()
        .unwrap_or_default()
        .iter()
        .filter(|s| is_downloadable_gguf(&s.rfilename))
        .map(|s| {
            let size = sizes
                .get(&s.rfilename)
                .copied()
                .or_else(|| s.lfs.as_ref().and_then(|l| l.size))
                .or(s.size);
            let fit = match (size, vram_total_bytes) {
                (Some(bytes), Some(vram)) => Some(crate::system::fits_vram(
                    bytes,
                    vram,
                    crate::api::MEMORY_HEADROOM_FRACTION,
                )),
                _ => None,
            };
            CatalogQuant {
                file: s.rfilename.clone(),
                quant: crate::worker::models::quant_from_filename(&s.rfilename),
                size_bytes: size,
                downloaded: inv.has_file(&info.id, &s.rfilename),
                fit,
            }
        })
        .collect();
    quants.sort_by(|a, b| {
        let key = |q: &CatalogQuant| (q.size_bytes.is_none(), q.size_bytes, q.file.clone());
        key(a).cmp(&key(b))
    });
    quants
}

/// The quant to preselect/download when the caller names none: the community
/// default `Q4_K_M` if the repo ships one, else the LARGEST quant this machine
/// fits (best quality that loads), else the smallest by size (the safest bet
/// when nothing fits or no verdict exists).
pub fn default_quant(quants: &[CatalogQuant]) -> Option<&CatalogQuant> {
    // Never preselect ONE SHARD of a split model when a whole file exists —
    // a shard is smaller than any whole quant by construction, so the
    // "smallest" fallback would otherwise hand the picker an unloadable
    // fragment. A shards-only repo falls back to the full set (the
    // documented per-file limitation).
    let whole: Vec<&CatalogQuant> = quants.iter().filter(|q| !is_shard_part(&q.file)).collect();
    let pool: Vec<&CatalogQuant> = if whole.is_empty() {
        quants.iter().collect()
    } else {
        whole
    };
    if let Some(q) = pool.iter().find(|q| {
        q.quant
            .as_deref()
            .is_some_and(|l| l.eq_ignore_ascii_case("Q4_K_M"))
    }) {
        return Some(q);
    }
    if let Some(q) = pool
        .iter()
        .filter(|q| q.fit.as_ref().is_some_and(|f| f.fits))
        .max_by_key(|q| q.size_bytes)
    {
        return Some(q);
    }
    // Smallest known size (an unknown size never beats a known one) — computed
    // here rather than assuming the caller pre-sorted the rows.
    pool.into_iter()
        .min_by_key(|q| (q.size_bytes.is_none(), q.size_bytes))
}

/// One catalog search (an EMPTY `search` is the browse page — the Hub's full
/// GGUF listing in the requested order): ask the source, enrich each row with
/// its `gguf` block (the list endpoint omits it; each fetch is bounded and
/// degrades to the bare row), then map every hit against the SAME local
/// inventory snapshot. With `compatible_only`, rows whose estimate says
/// "does not fit" are dropped (the Hub is over-asked so the page stays full).
/// A source failure is the caller's to surface — there is no stale cache to
/// fall back to.
pub async fn search<S: CatalogSource + Sync>(
    src: &S,
    q: &CatalogQuery,
    inv: &LocalInventory,
    vram_total_bytes: Option<u64>,
) -> Result<CatalogSearchResponse, HiggsError> {
    use crate::catalog::source::effective_limit;
    use crate::catalog::wire::MAX_SEARCH_LIMIT;
    let want = effective_limit(q.limit.unwrap_or(0));
    let compatible_only = q.compatible_only.unwrap_or(false);
    // Filtering loses rows, so ask the Hub for the hard cap up front — but
    // only when a verdict is even possible: with no VRAM figure every row's
    // fit is `None` and nothing can ever be dropped.
    let over_ask = compatible_only && vram_total_bytes.is_some();
    let hub_q = CatalogQuery {
        limit: Some(if over_ask {
            MAX_SEARCH_LIMIT
        } else {
            want as u32
        }),
        ..q.clone()
    };
    let hits = enriched(src, src.search(&hub_q).await?, vram_total_bytes).await;
    let mut models: Vec<CatalogModelSummary> = hits
        .iter()
        .map(|h| {
            let mut s = summary_from(&h.info, inv, vram_total_bytes);
            // A REAL-size verdict (the metadata-less fallback) beats the
            // estimate path — which, for these rows, produced nothing.
            if h.real_fit.is_some() {
                s.fit = h.real_fit;
            }
            s
        })
        .collect();
    if compatible_only {
        // Unknown is kept: absence of an estimate is not evidence of misfit.
        models.retain(|m| m.fit.as_ref().is_none_or(|f| f.fits));
    }
    models.truncate(want);
    Ok(CatalogSearchResponse { models })
}

/// One enriched search row: the (possibly info-merged) Hub row, plus a
/// REAL-size fit verdict when the row's estimate had no inputs and actual
/// file sizes could be resolved in the background (`None` = the estimate
/// path applies).
struct EnrichedHit {
    info: ModelInfo,
    real_fit: Option<crate::system::FitAssessment>,
}

/// Fill each search row's `gguf` block (and full sibling list) via the repo
/// info endpoint, [`SEARCH_ENRICH_CONCURRENCY`] at a time, order preserved.
/// Best-effort per row: a failed or stalled ([`ENRICHMENT_TIMEOUT`]) fetch
/// leaves that row as the list endpoint returned it. Rows whose ESTIMATE has
/// no inputs (no parameter count / no recognized quant label) fall back to
/// REAL sizes — a known sibling size first, else one bounded `paths-info`
/// fetch — so "no metadata" degrades to an honest verdict, not a blind spot.
/// The fallback runs only when `vram_total_bytes` is known (else no verdict
/// is possible and the fetch would be pure cost).
async fn enriched<S: CatalogSource + Sync>(
    src: &S,
    hits: Vec<ModelInfo>,
    vram_total_bytes: Option<u64>,
) -> Vec<EnrichedHit> {
    use futures::StreamExt;
    // One deadline for the whole pass ([`SEARCH_ENRICH_TOTAL_TIMEOUT`]) so a
    // page of dead fetches cannot stack per-row ceilings serially.
    let deadline = tokio::time::Instant::now() + SEARCH_ENRICH_TOTAL_TIMEOUT;
    // UNORDERED completion, order restored by index afterward: an ordered
    // buffer would keep COMPLETED fetches occupying their slots until the
    // slowest earlier row yields, starving later rows of the shared deadline.
    let mut rows: Vec<(usize, EnrichedHit)> =
        futures::stream::iter(hits.into_iter().enumerate().map(|(idx, row)| async move {
            let row = if row.gguf.is_some() {
                row
            } else {
                let per_row = tokio::time::Instant::now() + ENRICHMENT_TIMEOUT;
                match tokio::time::timeout_at(deadline.min(per_row), src.info(&row.id)).await {
                    Ok(Ok(full)) => merge_enrichment(row, full),
                    _ => row,
                }
            };
            let real_fit = match vram_total_bytes {
                Some(vram) if estimated_min_quant_bytes(&row).is_none() => {
                    real_min_size(src, &row, deadline).await.map(|bytes| {
                        crate::system::fits_vram(bytes, vram, crate::api::MEMORY_HEADROOM_FRACTION)
                    })
                }
                _ => None,
            };
            (
                idx,
                EnrichedHit {
                    info: row,
                    real_fit,
                },
            )
        }))
        .buffer_unordered(SEARCH_ENRICH_CONCURRENCY)
        .collect()
        .await;
    rows.sort_unstable_by_key(|r| r.0);
    rows.into_iter().map(|(_, hit)| hit).collect()
}

/// Whether a GGUF file name is ONE SHARD of a split model
/// (`…-00001-of-00003.gguf`): its size is a fraction of the model's, so it
/// must never stand in for a repo-level fit verdict.
fn is_shard_part(name: &str) -> bool {
    let Some(stem) = name
        .strip_suffix(".gguf")
        .or_else(|| name.strip_suffix(".GGUF"))
    else {
        return false;
    };
    // …-<digits>-of-<digits>
    let mut parts = stem.rsplitn(3, '-');
    let (Some(last), Some(of), Some(first)) = (parts.next(), parts.next(), parts.next()) else {
        return false;
    };
    of == "of"
        && !last.is_empty()
        && last.bytes().all(|b| b.is_ascii_digit())
        && first
            .rsplit('-')
            .next()
            .is_some_and(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()))
}

/// The smallest REAL size of a row's downloadable quants: a known sibling
/// size when the Hub already reported one, else one `paths-info` fetch
/// bounded like every other enrichment. Shard parts of split models are
/// excluded — one shard's size is not a model's. `None` when nothing
/// resolves — the row stays an honest unknown.
async fn real_min_size<S: CatalogSource + Sync>(
    src: &S,
    row: &ModelInfo,
    deadline: tokio::time::Instant,
) -> Option<u64> {
    let siblings: Vec<_> = row
        .siblings
        .as_deref()
        .unwrap_or_default()
        .iter()
        .filter(|s| is_downloadable_gguf(&s.rfilename) && !is_shard_part(&s.rfilename))
        .collect();
    if siblings.is_empty() {
        return None;
    }
    if let Some(min) = siblings
        .iter()
        .filter_map(|s| s.lfs.as_ref().and_then(|l| l.size).or(s.size))
        .min()
    {
        return Some(min);
    }
    let files: Vec<String> = siblings.iter().map(|s| s.rfilename.clone()).collect();
    let per_fetch = tokio::time::Instant::now() + ENRICHMENT_TIMEOUT;
    match tokio::time::timeout_at(deadline.min(per_fetch), src.file_sizes(&row.id, &files)).await {
        Ok(Ok(sizes)) => sizes.values().copied().min(),
        _ => None,
    }
}

/// Merge a repo-info response into its LIST row: the info supplies what the
/// list omits (the `gguf` block, the full sibling set, tags) and fills fields
/// the list left empty — but the list row keeps its identity and its stats.
/// An info response for a DIFFERENT id is discarded outright: enrichment must
/// never rename a row, and a partial info must never blank data the list had.
fn merge_enrichment(mut row: ModelInfo, full: ModelInfo) -> ModelInfo {
    if full.id != row.id {
        return row;
    }
    row.gguf = full.gguf.or(row.gguf);
    row.siblings = full.siblings.or(row.siblings);
    row.tags = full.tags.or(row.tags);
    row.author = row.author.or(full.author);
    row.downloads = row.downloads.or(full.downloads);
    row.likes = row.likes.or(full.likes);
    row.last_modified = row.last_modified.or(full.last_modified);
    row.pipeline_tag = row.pipeline_tag.or(full.pipeline_tag);
    row
}

/// One repo's full catalog detail. Only the repo info itself is load-bearing;
/// the enrichments (file sizes, README, author rows) each degrade
/// independently on failure OR stall ([`ENRICHMENT_TIMEOUT`]) — a Hub hiccup
/// on one never blanks or holds the pane.
pub async fn detail<S: CatalogSource>(
    src: &S,
    repo: &str,
    inv: &LocalInventory,
    vram_total_bytes: Option<u64>,
) -> Result<CatalogModelDetail, HiggsError> {
    let info = src.info(repo).await?;
    let gguf_files: Vec<String> = info
        .siblings
        .as_deref()
        .unwrap_or_default()
        .iter()
        .filter(|s| is_downloadable_gguf(&s.rfilename))
        .map(|s| s.rfilename.clone())
        .collect();
    let mut summary = summary_from(&info, inv, vram_total_bytes);
    let (sizes, readme, more) = futures::join!(
        bounded(repo, "file sizes", src.file_sizes(repo, &gguf_files)),
        bounded(repo, "README", src.readme(repo)),
        bounded(
            repo,
            "author browse",
            more_by_author(src, summary.author.as_deref(), repo, inv, vram_total_bytes)
        ),
    );
    let sizes = sizes.unwrap_or_else(|e| {
        tracing::warn!(repo, error = %e, "higgs: catalog file sizes failed; using sibling sizes");
        HashMap::new()
    });
    let readme = readme.unwrap_or_else(|e| {
        tracing::warn!(repo, error = %e, "higgs: catalog README fetch failed");
        None
    });
    let more_by_author = more.unwrap_or_else(|e| {
        tracing::warn!(repo, error = %e, "higgs: catalog author browse failed");
        Vec::new()
    });
    let quants = quants_from(&info, &sizes, inv, vram_total_bytes);
    // In a DETAIL the repo-level verdict is real-size-only: the smallest
    // sized quant's own fit (rows are size-ascending, so the first verdict is
    // the smallest file's) — and when NO size is known it is absent, never
    // the search-side estimate the summary mapping computed.
    // Shard rows are excluded from the REPO-level verdict for the same reason
    // as the search-side fallback: one shard's size is not a model's.
    summary.fit = quants
        .iter()
        .filter(|q| !is_shard_part(&q.file))
        .find_map(|q| q.fit);
    Ok(CatalogModelDetail {
        default_file: default_quant(&quants).map(|q| q.file.clone()),
        quants,
        tags: info.tags.clone().unwrap_or_default(),
        summary,
        readme,
        more_by_author,
    })
}

/// Bound one best-effort enrichment to [`ENRICHMENT_TIMEOUT`]: a stall
/// becomes an ordinary error, which the caller's degrade path absorbs.
async fn bounded<T>(
    repo: &str,
    what: &'static str,
    fut: impl std::future::Future<Output = Result<T, HiggsError>>,
) -> Result<T, HiggsError> {
    match tokio::time::timeout(ENRICHMENT_TIMEOUT, fut).await {
        Ok(res) => res,
        Err(_) => Err(HiggsError::HubTransport {
            repo: repo.to_owned(),
            detail: format!("{what} timed out after {}s", ENRICHMENT_TIMEOUT.as_secs()),
        }),
    }
}

/// The "more by this publisher" rows: an author-only search, self excluded,
/// capped at [`MORE_BY_AUTHOR_LIMIT`]. No author → no rows, no Hub call.
async fn more_by_author<S: CatalogSource>(
    src: &S,
    author: Option<&str>,
    self_repo: &str,
    inv: &LocalInventory,
    vram_total_bytes: Option<u64>,
) -> Result<Vec<CatalogModelSummary>, HiggsError> {
    let Some(author) = author else {
        return Ok(Vec::new());
    };
    let q = CatalogQuery {
        search: String::new(),
        author: Some(author.to_owned()),
        sort: Some(CatalogSort::Downloads),
        // One extra row so the self-exclusion below still fills the cap.
        limit: Some((MORE_BY_AUTHOR_LIMIT + 1) as u32),
        compatible_only: None,
    };
    let hits = src.search(&q).await?;
    Ok(hits
        .iter()
        .filter(|m| m.id != self_repo)
        .take(MORE_BY_AUTHOR_LIMIT)
        .map(|m| summary_from(m, inv, vram_total_bytes))
        .collect())
}

#[cfg(test)]
#[path = "service_tests.rs"]
mod tests;
