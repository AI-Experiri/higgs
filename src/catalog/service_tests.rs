use super::*;

use std::collections::{HashMap, HashSet};

use huggingface_hub::ModelInfo;
use serde_json::json;

/// Build a crate `ModelInfo` exactly as the Hub API would serialize it.
fn info(v: serde_json::Value) -> ModelInfo {
    serde_json::from_value(v).expect("fixture ModelInfo")
}

fn inv_with(entries: &[(&str, &str)]) -> LocalInventory {
    let mut inv = LocalInventory::empty();
    for (repo, file) in entries {
        inv.insert(repo, file);
    }
    inv
}

// ── summary_from ───────────────────────────────────────────────────────────

#[test]
fn summary_maps_the_hub_fields_and_downloaded_state() {
    let m = info(json!({
        "id": "acme/tiny-llama-gguf",
        "author": "acme",
        "downloads": 1234u64,
        "likes": 56u64,
        "lastModified": "2026-07-01T10:00:00.000Z",
        "pipeline_tag": "text-generation",
    }));
    let s = summary_from(&m, &inv_with(&[("acme/tiny-llama-gguf", "a.gguf")]), None);
    assert_eq!(s.id, "acme/tiny-llama-gguf");
    assert_eq!(s.author.as_deref(), Some("acme"));
    assert_eq!(s.downloads, Some(1234));
    assert_eq!(s.likes, Some(56));
    assert_eq!(s.updated.as_deref(), Some("2026-07-01T10:00:00.000Z"));
    assert_eq!(s.pipeline.as_deref(), Some("text-generation"));
    assert!(s.downloaded);
}

#[test]
fn summary_falls_back_to_the_id_org_for_a_missing_author_and_marks_not_downloaded() {
    let m = info(json!({ "id": "acme/other-model" }));
    let s = summary_from(&m, &LocalInventory::empty(), None);
    assert_eq!(s.author.as_deref(), Some("acme"));
    assert!(!s.downloaded);
    assert_eq!(s.downloads, None);
    assert!(s.gguf.is_none());
}

// ── gguf_meta ──────────────────────────────────────────────────────────────

#[test]
fn gguf_meta_reduces_the_hub_gguf_block() {
    let m = info(json!({
        "id": "acme/m",
        "gguf": { "architecture": "llama", "total": 8_030_261_248u64, "context_length": 131_072u64 },
    }));
    let g = gguf_meta(&m).expect("gguf meta");
    assert_eq!(g.arch.as_deref(), Some("llama"));
    assert_eq!(g.params_total, Some(8_030_261_248));
    assert_eq!(g.ctx_train, Some(131_072));
}

#[test]
fn gguf_meta_is_none_without_the_block_or_with_an_empty_block() {
    assert!(gguf_meta(&info(json!({ "id": "a/b" }))).is_none());
    assert!(gguf_meta(&info(json!({ "id": "a/b", "gguf": {} }))).is_none());
}

// ── quants_from ────────────────────────────────────────────────────────────

/// Siblings fixture: one non-GGUF, three GGUFs with sizes resolved from the
/// paths-info map, an LFS-sibling fallback, and one unknown size.
fn quant_fixture() -> ModelInfo {
    info(json!({
        "id": "acme/m",
        "siblings": [
            { "rfilename": "README.md" },
            { "rfilename": "m-Q4_K_M.gguf" },
            { "rfilename": "m-Q8_0.gguf", "lfs": { "size": 8_000u64 } },
            { "rfilename": "m-F16.GGUF" },
        ],
    }))
}

#[test]
fn quants_keep_only_gguf_files_and_resolve_sizes_by_precedence() {
    let sizes = HashMap::from([("m-Q4_K_M.gguf".to_string(), 4_000u64)]);
    let quants = quants_from(&quant_fixture(), &sizes, &LocalInventory::empty(), None);
    let names: Vec<&str> = quants.iter().map(|q| q.file.as_str()).collect();
    // Size-ascending, unknown sizes last.
    assert_eq!(names, ["m-Q4_K_M.gguf", "m-Q8_0.gguf", "m-F16.GGUF"]);
    assert_eq!(quants[0].size_bytes, Some(4_000)); // paths-info wins
    assert_eq!(quants[1].size_bytes, Some(8_000)); // LFS sibling fallback
    assert_eq!(quants[2].size_bytes, None); // unknown stays unknown
    assert_eq!(quants[0].quant.as_deref(), Some("Q4_K_M"));
    assert_eq!(quants[2].quant.as_deref(), Some("F16"));
}

#[test]
fn quants_mark_downloaded_files_and_fit_only_when_size_and_vram_are_known() {
    let sizes = HashMap::from([("m-Q4_K_M.gguf".to_string(), 4_000u64)]);
    let inv = inv_with(&[("acme/m", "m-Q4_K_M.gguf")]);
    let quants = quants_from(&quant_fixture(), &sizes, &inv, Some(10_000));
    assert!(quants[0].downloaded);
    assert!(!quants[1].downloaded);
    let fit = quants[0].fit.as_ref().expect("fit for sized quant");
    assert!(fit.fits);
    assert_eq!(fit.needed_bytes, 4_000);
    // Unknown size → no verdict, never a fake one.
    assert!(quants[2].fit.is_none());
    // Known size but unknown hardware → no verdict either.
    let no_hw = quants_from(&quant_fixture(), &sizes, &inv, None);
    assert!(no_hw[0].fit.is_none());
}

#[test]
fn quants_exclude_files_the_downloader_would_refuse() {
    // `pull` only accepts a single safe `*.gguf` name (the scanned layout) —
    // a row the picker offers must be a row the download can land, so
    // subdirectory/unsafe siblings never become quant rows.
    let m = info(json!({
        "id": "acme/m",
        "siblings": [
            { "rfilename": "sub/dir-Q4_K_M.gguf" },
            { "rfilename": "we ird-Q4_K_M.gguf" },
            { "rfilename": "ok-Q4_K_M.gguf" },
        ],
    }));
    let quants = quants_from(&m, &HashMap::new(), &LocalInventory::empty(), None);
    let files: Vec<&str> = quants.iter().map(|q| q.file.as_str()).collect();
    assert_eq!(files, ["ok-Q4_K_M.gguf"]);
}

// ── default_quant ──────────────────────────────────────────────────────────

fn quant(file: &str, size: Option<u64>, fits: Option<bool>) -> CatalogQuant {
    CatalogQuant {
        file: file.to_string(),
        quant: crate::worker::models::quant_from_filename(file),
        size_bytes: size,
        downloaded: false,
        fit: fits.map(|f| crate::system::FitAssessment {
            fits: f,
            needed_bytes: size.unwrap_or(0),
            available_bytes: 0,
        }),
    }
}

#[test]
fn default_quant_never_picks_a_shard_when_a_whole_file_exists() {
    // A shard is smaller than any whole file by construction — the
    // "smallest" fallback must not preselect an unloadable fragment.
    let quants = vec![
        quant("m-00001-of-00003.gguf", Some(1_000), None),
        quant("m-Q6_K.gguf", Some(6_000), None),
    ];
    assert_eq!(default_quant(&quants).unwrap().file, "m-Q6_K.gguf");
}

#[tokio::test]
async fn detail_repo_verdict_and_default_skip_shard_rows() {
    // The smallest SIZED row is a shard; the repo-level verdict and the
    // preselected download must both come from the whole file instead.
    let src = FakeSource {
        info: Some(json!({
            "id": "acme/m",
            "siblings": [
                { "rfilename": "m-00001-of-00003.gguf" },
                { "rfilename": "m-Q4_K_M.gguf" },
            ],
        })),
        sizes: HashMap::from([
            ("m-00001-of-00003.gguf".to_string(), 1_000u64),
            ("m-Q4_K_M.gguf".to_string(), 4_000u64),
        ]),
        ..Default::default()
    };
    let d = detail(&src, "acme/m", &LocalInventory::empty(), Some(100_000))
        .await
        .expect("detail");
    assert_eq!(
        d.summary.fit.as_ref().expect("repo verdict").needed_bytes,
        4_000,
        "the verdict is the smallest WHOLE file, never a shard"
    );
    assert_eq!(d.default_file.as_deref(), Some("m-Q4_K_M.gguf"));
}

#[test]
fn default_quant_prefers_q4_k_m() {
    let quants = vec![
        quant("m-Q2_K.gguf", Some(2_000), Some(true)),
        quant("m-Q4_K_M.gguf", Some(4_000), Some(false)),
        quant("m-Q8_0.gguf", Some(8_000), Some(true)),
    ];
    assert_eq!(default_quant(&quants).unwrap().file, "m-Q4_K_M.gguf");
}

#[test]
fn default_quant_falls_back_to_the_largest_that_fits() {
    let quants = vec![
        quant("m-Q2_K.gguf", Some(2_000), Some(true)),
        quant("m-Q6_K.gguf", Some(6_000), Some(true)),
        quant("m-F16.gguf", Some(16_000), Some(false)),
    ];
    assert_eq!(default_quant(&quants).unwrap().file, "m-Q6_K.gguf");
}

#[test]
fn default_quant_without_any_fit_verdict_picks_the_smallest() {
    let quants = vec![
        quant("m-Q8_0.gguf", Some(8_000), None),
        quant("m-Q2_K.gguf", Some(2_000), None),
        quant("m-F16.gguf", None, None),
    ];
    assert_eq!(default_quant(&quants).unwrap().file, "m-Q2_K.gguf");
    assert!(default_quant(&[]).is_none());
}

// ── search / detail over a fake source ─────────────────────────────────────

/// Canned [`CatalogSource`]: fixed responses, per-method failure switches, and
/// a record of every search query so tests can assert the author browse.
#[derive(Default)]
struct FakeSource {
    hits: Vec<serde_json::Value>,
    author_hits: Vec<serde_json::Value>,
    info: Option<serde_json::Value>,
    /// Per-repo info responses for the search-row enrichment; consulted
    /// before the single `info` fixture.
    infos: HashMap<String, serde_json::Value>,
    sizes: HashMap<String, u64>,
    readme: Option<String>,
    fail_search: bool,
    fail_info: bool,
    /// Info endpoint black-holes: never resolves, never errors.
    hang_info: bool,
    /// Repo ids whose info fetch black-holes (per-row variant of `hang_info`).
    hang_ids: HashSet<String>,
    /// How many times `file_sizes` was called (asserts fallback skips).
    sizes_calls: std::sync::atomic::AtomicU32,
    /// Delay before every (non-hanging) info answer — for timing tests.
    info_delay: Option<std::time::Duration>,
    fail_sizes: bool,
    fail_readme: bool,
    fail_author_search: bool,
    hang_readme: bool,
    searches: std::sync::Mutex<Vec<crate::catalog::wire::CatalogQuery>>,
}

fn fake_err() -> crate::diagnostic::HiggsError {
    crate::diagnostic::HiggsError::HubTransport {
        repo: "fake".into(),
        detail: "injected".into(),
    }
}

impl crate::catalog::source::CatalogSource for FakeSource {
    async fn search(
        &self,
        q: &crate::catalog::wire::CatalogQuery,
    ) -> Result<Vec<ModelInfo>, crate::diagnostic::HiggsError> {
        let author_browse = q.author.is_some();
        self.searches.lock().unwrap().push(q.clone());
        if author_browse {
            if self.fail_author_search {
                return Err(fake_err());
            }
            return Ok(self.author_hits.iter().cloned().map(info).collect());
        }
        if self.fail_search {
            return Err(fake_err());
        }
        Ok(self.hits.iter().cloned().map(info).collect())
    }

    async fn info(&self, repo: &str) -> Result<ModelInfo, crate::diagnostic::HiggsError> {
        if self.hang_info || self.hang_ids.contains(repo) {
            std::future::pending::<()>().await;
        }
        if let Some(delay) = self.info_delay {
            tokio::time::sleep(delay).await;
        }
        if self.fail_info {
            return Err(fake_err());
        }
        if let Some(v) = self.infos.get(repo) {
            return Ok(info(v.clone()));
        }
        // No fixture = a failing info endpoint; the search enrichment must
        // degrade to the bare row rather than lose it.
        self.info.clone().map(info).ok_or_else(fake_err)
    }

    async fn file_sizes(
        &self,
        _repo: &str,
        _files: &[String],
    ) -> Result<HashMap<String, u64>, crate::diagnostic::HiggsError> {
        self.sizes_calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if self.fail_sizes {
            return Err(fake_err());
        }
        Ok(self.sizes.clone())
    }

    async fn readme(&self, _repo: &str) -> Result<Option<String>, crate::diagnostic::HiggsError> {
        if self.hang_readme {
            // A black-holed connection: never resolves, never errors.
            std::future::pending::<()>().await;
        }
        if self.fail_readme {
            return Err(fake_err());
        }
        Ok(self.readme.clone())
    }
}

fn query(search: &str) -> crate::catalog::wire::CatalogQuery {
    crate::catalog::wire::CatalogQuery {
        search: search.into(),
        author: None,
        sort: None,
        limit: None,
        compatible_only: None,
    }
}

#[tokio::test]
async fn search_maps_hits_with_the_downloaded_flag() {
    let src = FakeSource {
        hits: vec![json!({ "id": "acme/a" }), json!({ "id": "acme/b" })],
        ..Default::default()
    };
    let inv = inv_with(&[("acme/b", "b.gguf")]);
    let resp = search(&src, &query("a"), &inv, None).await.expect("search");
    assert_eq!(resp.models.len(), 2);
    assert!(!resp.models[0].downloaded);
    assert!(resp.models[1].downloaded);
}

#[tokio::test]
async fn search_propagates_a_source_failure() {
    let src = FakeSource {
        fail_search: true,
        ..Default::default()
    };
    assert!(search(&src, &query("a"), &LocalInventory::empty(), None)
        .await
        .is_err());
}

// ── search enrichment + compatibility gate ─────────────────────────────────

/// A ~1B-param repo whose smallest quant (Q2 family, 0.42 B/param) estimates
/// to ~420 MB.
fn small_repo_info() -> serde_json::Value {
    json!({
        "id": "acme/small",
        "gguf": { "architecture": "llama", "total": 1_000_000_000u64 },
        "siblings": [
            { "rfilename": "small-Q2_K.gguf" },
            { "rfilename": "small-Q4_K_M.gguf" },
        ],
    })
}

/// A ~70B-param repo shipping only a Q4 quant → ~46 GB estimate.
fn big_repo_info() -> serde_json::Value {
    json!({
        "id": "acme/big",
        "gguf": { "architecture": "llama", "total": 70_000_000_000u64 },
        "siblings": [ { "rfilename": "big-Q4_K_M.gguf" } ],
    })
}

#[tokio::test]
async fn search_enriches_rows_with_their_gguf_blocks() {
    let src = FakeSource {
        hits: vec![json!({ "id": "acme/small" })],
        infos: HashMap::from([("acme/small".to_string(), small_repo_info())]),
        ..Default::default()
    };
    let resp = search(&src, &query(""), &LocalInventory::empty(), Some(8 << 30))
        .await
        .expect("search");
    let row = &resp.models[0];
    assert_eq!(
        row.gguf.as_ref().expect("enriched gguf").arch.as_deref(),
        Some("llama")
    );
    let fit = row.fit.as_ref().expect("estimate against known VRAM");
    assert!(fit.fits, "a ~420MB smallest quant fits an 8GB budget");
}

#[tokio::test]
async fn enrichment_never_renames_a_row_or_blanks_its_list_stats() {
    let src = FakeSource {
        hits: vec![
            json!({ "id": "acme/a", "downloads": 9u64, "likes": 2u64 }),
            json!({ "id": "acme/b", "downloads": 5u64 }),
        ],
        infos: HashMap::from([
            // A response for a DIFFERENT repo (misrouted/renamed) — discarded.
            (
                "acme/a".to_string(),
                json!({ "id": "acme/zzz", "gguf": { "total": 1u64 } }),
            ),
            // Same id but PARTIAL: gguf present, stats absent — merged, the
            // list row's stats survive.
            (
                "acme/b".to_string(),
                json!({ "id": "acme/b", "gguf": { "architecture": "llama", "total": 7u64 } }),
            ),
        ]),
        ..Default::default()
    };
    let resp = search(&src, &query(""), &LocalInventory::empty(), None)
        .await
        .expect("search");
    let a = &resp.models[0];
    assert_eq!(a.id, "acme/a", "a mismatched info must not rename the row");
    assert_eq!(a.downloads, Some(9));
    assert!(
        a.gguf.is_none(),
        "the mismatched info is discarded wholesale"
    );
    let b = &resp.models[1];
    assert_eq!(b.id, "acme/b");
    assert_eq!(b.downloads, Some(5), "list stats survive a partial info");
    assert_eq!(
        b.gguf.as_ref().expect("merged gguf").arch.as_deref(),
        Some("llama")
    );
}

#[tokio::test]
async fn search_keeps_the_bare_row_when_enrichment_fails() {
    let src = FakeSource {
        hits: vec![json!({ "id": "acme/a", "downloads": 7u64 })],
        // No per-repo fixture and no `info` → the info endpoint errors.
        ..Default::default()
    };
    let resp = search(&src, &query("a"), &LocalInventory::empty(), Some(8 << 30))
        .await
        .expect("search");
    assert_eq!(
        resp.models.len(),
        1,
        "a failed enrichment never drops the row"
    );
    assert_eq!(resp.models[0].downloads, Some(7));
    assert!(resp.models[0].gguf.is_none());
    assert!(resp.models[0].fit.is_none());
}

#[tokio::test]
async fn compatible_only_drops_misfits_keeps_unknowns_and_over_asks_the_hub() {
    let src = FakeSource {
        hits: vec![
            json!({ "id": "acme/big" }),
            json!({ "id": "acme/small" }),
            json!({ "id": "acme/mystery" }), // enrichment fails → no verdict
        ],
        infos: HashMap::from([
            ("acme/big".to_string(), big_repo_info()),
            ("acme/small".to_string(), small_repo_info()),
        ]),
        ..Default::default()
    };
    let q = crate::catalog::wire::CatalogQuery {
        compatible_only: Some(true),
        ..query("")
    };
    let resp = search(&src, &q, &LocalInventory::empty(), Some(8 << 30))
        .await
        .expect("search");
    let ids: Vec<&str> = resp.models.iter().map(|m| m.id.as_str()).collect();
    assert_eq!(
        ids,
        ["acme/small", "acme/mystery"],
        "the proven misfit goes; the unknown is never treated as one"
    );
    // Filtering loses rows, so the Hub was over-asked up to the hard cap.
    let searches = src.searches.lock().unwrap();
    assert_eq!(
        searches[0].limit,
        Some(crate::catalog::wire::MAX_SEARCH_LIMIT)
    );
}

#[tokio::test]
async fn search_truncates_to_the_requested_limit_after_filtering() {
    let src = FakeSource {
        hits: vec![
            json!({ "id": "acme/a" }),
            json!({ "id": "acme/b" }),
            json!({ "id": "acme/c" }),
        ],
        ..Default::default()
    };
    let q = crate::catalog::wire::CatalogQuery {
        limit: Some(2),
        compatible_only: Some(true),
        ..query("")
    };
    let resp = search(&src, &q, &LocalInventory::empty(), None)
        .await
        .expect("search");
    assert_eq!(resp.models.len(), 2);
}

#[tokio::test(start_paused = true)]
async fn search_enrichment_is_bounded_as_a_whole_not_per_row() {
    // A full page of black-holed info fetches must not stack per-row
    // timeouts (50 rows / 8 at a time × 10s would be ~70s of silence); the
    // whole pass shares one deadline, after which rows go out bare. The
    // paused clock auto-advances, so virtual elapsed time IS the bound.
    let src = FakeSource {
        hits: (0..50)
            .map(|i| json!({ "id": format!("acme/m{i}") }))
            .collect(),
        hang_info: true,
        ..Default::default()
    };
    let start = tokio::time::Instant::now();
    let resp = search(&src, &query(""), &LocalInventory::empty(), None)
        .await
        .expect("search");
    assert_eq!(resp.models.len(), 25, "bare rows still map to the page");
    let elapsed = start.elapsed();
    assert!(
        elapsed <= SEARCH_ENRICH_TOTAL_TIMEOUT + std::time::Duration::from_secs(1),
        "enrichment must share one deadline, took {elapsed:?}"
    );
}

#[tokio::test(start_paused = true)]
async fn a_slow_head_row_does_not_starve_later_enrichments() {
    // Row 0 black-holes; every other row answers after 6 virtual seconds.
    // Slots must recycle as fetches COMPLETE, not as they yield in order —
    // with ordered buffering, rows past the first window would only start
    // after the hung head's 10s per-row timeout and then miss the 15s
    // whole-pass deadline (bare rows that a faster refresh would enrich).
    let infos: HashMap<String, serde_json::Value> = (1..50)
        .map(|i| {
            (
                format!("acme/m{i}"),
                json!({
                    "id": format!("acme/m{i}"),
                    "gguf": { "architecture": "llama", "total": 1u64 },
                }),
            )
        })
        .collect();
    let src = FakeSource {
        hits: (0..50)
            .map(|i| json!({ "id": format!("acme/m{i}") }))
            .collect(),
        infos,
        hang_ids: HashSet::from(["acme/m0".to_string()]),
        info_delay: Some(std::time::Duration::from_secs(6)),
        ..Default::default()
    };
    let q = crate::catalog::wire::CatalogQuery {
        limit: Some(50),
        ..query("")
    };
    let resp = search(&src, &q, &LocalInventory::empty(), None)
        .await
        .expect("search");
    assert!(resp.models[0].gguf.is_none(), "the hung head degrades bare");
    assert!(
        resp.models[8].gguf.is_some(),
        "a row past the first concurrency window must still enrich"
    );
    // Order survives the unordered completion.
    assert_eq!(resp.models[8].id, "acme/m8");
}

#[tokio::test]
async fn compatible_only_without_vram_does_not_over_ask_the_hub() {
    // With no VRAM figure NO row can ever get a verdict, so filtering can
    // never drop one — over-asking (and enriching) 50 rows would be pure
    // cost with zero effect on the result set.
    let src = FakeSource {
        hits: vec![json!({ "id": "acme/a" })],
        ..Default::default()
    };
    let q = crate::catalog::wire::CatalogQuery {
        limit: Some(5),
        compatible_only: Some(true),
        ..query("")
    };
    search(&src, &q, &LocalInventory::empty(), None)
        .await
        .expect("search");
    let searches = src.searches.lock().unwrap();
    assert_eq!(
        searches[0].limit,
        Some(5),
        "no verdict possible → no over-ask"
    );
}

#[tokio::test]
async fn an_iq_quant_repo_near_the_boundary_is_not_hidden_by_the_q_family_estimate() {
    // ~70B params shipping ONLY IQ2_XXS (~21 GB real-world). The VRAM budget
    // sits between the IQ2 estimate (0.30 B/param = 21 GB) and the old Q2
    // mapping (0.42 B/param = 29.4 GB): with the biased table the default
    // browse silently dropped a model this machine can actually run.
    let src = FakeSource {
        hits: vec![json!({ "id": "acme/iq" })],
        infos: HashMap::from([(
            "acme/iq".to_string(),
            json!({
                "id": "acme/iq",
                "gguf": { "architecture": "llama", "total": 70_000_000_000u64 },
                "siblings": [ { "rfilename": "iq-IQ2_XXS.gguf" } ],
            }),
        )]),
        ..Default::default()
    };
    let q = crate::catalog::wire::CatalogQuery {
        compatible_only: Some(true),
        ..query("")
    };
    // 32 GB VRAM: the canonical headroom fraction leaves a budget above
    // 21 GB (IQ2 estimate fits) but below 29.4 GB (the Q2 mapping did not).
    let resp = search(&src, &q, &LocalInventory::empty(), Some(32 << 30))
        .await
        .expect("search");
    assert_eq!(
        resp.models.len(),
        1,
        "a fitting IQ-quant repo must survive the compatible filter"
    );
    assert!(resp.models[0].fit.as_ref().expect("estimate").fits);
}

// ── the real-size fallback (no estimate inputs) ────────────────────────────

fn bare_row_info(size_hint: Option<u64>) -> serde_json::Value {
    // Siblings but NO gguf block → the estimate has no inputs. `size_hint`
    // puts an LFS size on the sibling (the fetch-free shortcut).
    match size_hint {
        Some(s) => json!({
            "id": "acme/bare",
            "siblings": [ { "rfilename": "w.gguf", "lfs": { "size": s } } ],
        }),
        None => json!({
            "id": "acme/bare",
            "siblings": [ { "rfilename": "w.gguf" } ],
        }),
    }
}

#[tokio::test]
async fn a_row_without_metadata_gets_a_real_size_verdict_from_paths_info() {
    let src = FakeSource {
        hits: vec![json!({ "id": "acme/bare" })],
        infos: HashMap::from([("acme/bare".to_string(), bare_row_info(None))]),
        sizes: HashMap::from([("w.gguf".to_string(), 1u64 << 30)]),
        ..Default::default()
    };
    let resp = search(&src, &query(""), &LocalInventory::empty(), Some(8 << 30))
        .await
        .expect("search");
    let fit = resp.models[0].fit.as_ref().expect("real-size verdict");
    assert!(fit.fits);
    assert_eq!(fit.needed_bytes, 1 << 30, "the verdict is the REAL size");
}

#[tokio::test]
async fn a_metadata_less_real_misfit_is_dropped_by_the_compatible_filter() {
    let src = FakeSource {
        hits: vec![json!({ "id": "acme/bare" })],
        infos: HashMap::from([("acme/bare".to_string(), bare_row_info(None))]),
        sizes: HashMap::from([("w.gguf".to_string(), 100u64 << 30)]),
        ..Default::default()
    };
    let q = crate::catalog::wire::CatalogQuery {
        compatible_only: Some(true),
        ..query("")
    };
    let resp = search(&src, &q, &LocalInventory::empty(), Some(8 << 30))
        .await
        .expect("search");
    assert!(
        resp.models.is_empty(),
        "a 100 GiB only-quant on an 8 GB budget is a PROVEN misfit"
    );
}

#[tokio::test]
async fn the_size_fallback_prefers_sibling_sizes_over_a_fetch() {
    let src = FakeSource {
        hits: vec![json!({ "id": "acme/bare" })],
        infos: HashMap::from([("acme/bare".to_string(), bare_row_info(Some(2 << 30)))]),
        ..Default::default()
    };
    let resp = search(&src, &query(""), &LocalInventory::empty(), Some(8 << 30))
        .await
        .expect("search");
    assert!(resp.models[0].fit.as_ref().expect("verdict").fits);
    assert_eq!(
        src.sizes_calls.load(std::sync::atomic::Ordering::Relaxed),
        0,
        "a known sibling size needs no paths-info round-trip"
    );
}

#[tokio::test]
async fn a_sharded_only_repo_gets_no_real_size_verdict() {
    // `m-00001-of-00003.gguf` is ONE SHARD of a model, not a runnable file —
    // min() over shard sizes would claim a fits=true verdict for a model
    // three times the size. Shard parts are excluded; a shards-only repo
    // stays an honest unknown (and is therefore never dropped).
    let src = FakeSource {
        hits: vec![json!({ "id": "acme/sharded" })],
        infos: HashMap::from([(
            "acme/sharded".to_string(),
            json!({
                "id": "acme/sharded",
                "siblings": [
                    { "rfilename": "m-00001-of-00003.gguf", "lfs": { "size": 1u64 << 30 } },
                    { "rfilename": "m-00002-of-00003.gguf", "lfs": { "size": 1u64 << 30 } },
                    { "rfilename": "m-00003-of-00003.gguf", "lfs": { "size": 1u64 << 30 } },
                ],
            }),
        )]),
        ..Default::default()
    };
    let resp = search(&src, &query(""), &LocalInventory::empty(), Some(8 << 30))
        .await
        .expect("search");
    assert!(
        resp.models[0].fit.is_none(),
        "one shard's size must never become the repo's verdict"
    );
}

#[tokio::test]
async fn the_size_fallback_is_skipped_when_no_verdict_is_possible() {
    let src = FakeSource {
        hits: vec![json!({ "id": "acme/bare" })],
        infos: HashMap::from([("acme/bare".to_string(), bare_row_info(None))]),
        sizes: HashMap::from([("w.gguf".to_string(), 1u64 << 30)]),
        ..Default::default()
    };
    let resp = search(&src, &query(""), &LocalInventory::empty(), None)
        .await
        .expect("search");
    assert!(resp.models[0].fit.is_none());
    assert_eq!(
        src.sizes_calls.load(std::sync::atomic::Ordering::Relaxed),
        0,
        "no VRAM figure → no verdict possible → no fetch"
    );
}

// ── the fit estimator ──────────────────────────────────────────────────────

#[test]
fn quant_bytes_per_param_covers_the_common_families_and_refuses_unknowns() {
    assert_eq!(quant_bytes_per_param("Q4_K_M"), Some(0.66));
    // I-quants are materially cheaper than their K-quant family — they get
    // their OWN numbers. Mapping IQ2 onto Q2's 0.42 overestimated by ~40%,
    // and under the default compatible_only filter an overestimate HIDES
    // runnable models (the unsafe error direction).
    assert_eq!(quant_bytes_per_param("IQ2_XS"), Some(0.3));
    assert_eq!(quant_bytes_per_param("IQ1_S"), Some(0.2));
    assert_eq!(quant_bytes_per_param("IQ4_XS"), Some(0.55));
    assert_eq!(
        quant_bytes_per_param("q8_0"),
        Some(1.12),
        "case-insensitive"
    );
    assert_eq!(quant_bytes_per_param("F16"), Some(2.1));
    assert_eq!(
        quant_bytes_per_param("MXFP4"),
        None,
        "unknown label → no verdict"
    );
}

#[test]
fn estimated_min_quant_bytes_uses_the_cheapest_shipped_family() {
    let m = info(small_repo_info());
    // Smallest family shipped is Q2 (0.42 B/param) — not the Q4 sibling.
    assert_eq!(estimated_min_quant_bytes(&m), Some(420_000_000));
    // No parameter count → no estimate.
    let bare = info(json!({ "id": "a/b", "siblings": [ { "rfilename": "x-Q4_K_M.gguf" } ] }));
    assert_eq!(estimated_min_quant_bytes(&bare), None);
    // Parameters but no recognizably-labeled quant → no estimate.
    let unlabeled = info(json!({
        "id": "a/b",
        "gguf": { "total": 1_000_000_000u64 },
        "siblings": [ { "rfilename": "weights.gguf" } ],
    }));
    assert_eq!(estimated_min_quant_bytes(&unlabeled), None);
}

fn detail_fixture() -> serde_json::Value {
    json!({
        "id": "acme/m",
        "author": "acme",
        "tags": ["gguf", "text-generation"],
        "siblings": [
            { "rfilename": "m-Q4_K_M.gguf" },
            { "rfilename": "m-Q8_0.gguf", "lfs": { "size": 8_000u64 } },
            { "rfilename": "LICENSE" },
        ],
    })
}

#[tokio::test]
async fn detail_assembles_quants_readme_tags_and_more_by_author() {
    let src = FakeSource {
        info: Some(detail_fixture()),
        sizes: HashMap::from([("m-Q4_K_M.gguf".to_string(), 4_000u64)]),
        readme: Some("# hello".into()),
        author_hits: vec![
            json!({ "id": "acme/m" }), // self — must be excluded
            json!({ "id": "acme/other" }),
        ],
        ..Default::default()
    };
    let d = detail(&src, "acme/m", &LocalInventory::empty(), Some(100_000))
        .await
        .expect("detail");
    assert_eq!(d.summary.id, "acme/m");
    assert_eq!(d.tags, vec!["gguf", "text-generation"]);
    assert_eq!(d.readme.as_deref(), Some("# hello"));
    let files: Vec<&str> = d.quants.iter().map(|q| q.file.as_str()).collect();
    assert_eq!(files, ["m-Q4_K_M.gguf", "m-Q8_0.gguf"]);
    assert_eq!(d.quants[0].size_bytes, Some(4_000));
    assert!(d.quants[0].fit.as_ref().expect("fit").fits);
    assert_eq!(
        d.default_file.as_deref(),
        Some("m-Q4_K_M.gguf"),
        "the picker preselect is higgs's default-quant policy"
    );
    // With real sizes known, the repo verdict is the smallest file's own fit.
    assert_eq!(
        d.summary.fit.as_ref().expect("repo fit").needed_bytes,
        4_000
    );
    let more: Vec<&str> = d.more_by_author.iter().map(|m| m.id.as_str()).collect();
    assert_eq!(more, ["acme/other"]);
    // The author browse went out as an author-only query.
    let searches = src.searches.lock().unwrap();
    assert_eq!(searches.len(), 1);
    assert_eq!(searches[0].author.as_deref(), Some("acme"));
    assert!(searches[0].search.is_empty());
}

#[tokio::test]
async fn detail_degrades_sizes_readme_and_author_rows_best_effort() {
    let src = FakeSource {
        info: Some(detail_fixture()),
        fail_sizes: true,
        fail_readme: true,
        fail_author_search: true,
        ..Default::default()
    };
    let d = detail(&src, "acme/m", &LocalInventory::empty(), None)
        .await
        .expect("detail still succeeds");
    // Sizes fall back to what the siblings themselves carried (sized rows
    // sort ahead of unknown-size rows).
    let by_file = |f: &str| d.quants.iter().find(|q| q.file == f).expect("row");
    assert_eq!(by_file("m-Q4_K_M.gguf").size_bytes, None);
    assert_eq!(by_file("m-Q8_0.gguf").size_bytes, Some(8_000));
    assert!(d.readme.is_none());
    assert!(d.more_by_author.is_empty());
}

#[tokio::test(start_paused = true)]
async fn detail_degrades_a_stalled_enrichment_instead_of_hanging() {
    let src = FakeSource {
        info: Some(detail_fixture()),
        hang_readme: true,
        ..Default::default()
    };
    // The paused clock auto-advances, so the enrichment ceiling fires
    // instantly; the outer timeout only catches a regression to "hangs".
    let d = tokio::time::timeout(
        std::time::Duration::from_secs(3600),
        detail(&src, "acme/m", &LocalInventory::empty(), None),
    )
    .await
    .expect("a stalled README must not hold the whole detail")
    .expect("detail");
    assert!(d.readme.is_none(), "the stalled part degrades to absent");
    assert_eq!(d.quants.len(), 2, "the rest of the pane still assembles");
}

#[tokio::test]
async fn detail_with_no_known_sizes_clears_the_search_estimate() {
    // The repo has params + a labeled quant, so a search-side ESTIMATE would
    // exist — but no file size is knowable here, and detail promises REAL
    // size verdicts only: its repo-level fit must be absent, not the
    // estimate.
    let src = FakeSource {
        info: Some(json!({
            "id": "acme/m",
            "gguf": { "total": 1_000_000_000u64 },
            "siblings": [ { "rfilename": "m-Q4_K_M.gguf" } ],
        })),
        fail_sizes: true,
        ..Default::default()
    };
    let d = detail(&src, "acme/m", &LocalInventory::empty(), Some(8 << 30))
        .await
        .expect("detail");
    assert!(d.quants[0].fit.is_none(), "no size → no quant verdict");
    assert!(
        d.summary.fit.is_none(),
        "detail must not fall back to the search estimate"
    );
}

#[tokio::test]
async fn detail_without_an_author_skips_the_author_browse() {
    let src = FakeSource {
        // A bare id with no `org/` prefix and no author field.
        info: Some(json!({ "id": "gpt2", "siblings": [] })),
        ..Default::default()
    };
    let d = detail(&src, "gpt2", &LocalInventory::empty(), None)
        .await
        .expect("detail");
    assert!(d.more_by_author.is_empty());
    assert!(src.searches.lock().unwrap().is_empty());
}

#[tokio::test]
async fn detail_propagates_an_info_failure() {
    let src = FakeSource {
        fail_info: true,
        ..Default::default()
    };
    assert!(detail(&src, "acme/m", &LocalInventory::empty(), None)
        .await
        .is_err());
}

// ── LocalInventory ─────────────────────────────────────────────────────────

#[test]
fn inventory_from_models_dir_reads_the_org_model_layout() {
    let dir = tempfile::tempdir().expect("tempdir");
    let repo = dir.path().join("acme").join("tiny");
    std::fs::create_dir_all(&repo).unwrap();
    std::fs::write(repo.join("tiny-Q4_K_M.gguf"), b"x").unwrap();
    std::fs::write(repo.join("notes.txt"), b"x").unwrap();

    let inv = LocalInventory::from_models_dir(dir.path());
    assert!(inv.has_repo("acme/tiny"));
    assert!(inv.has_file("acme/tiny", "tiny-Q4_K_M.gguf"));
    assert!(!inv.has_file("acme/tiny", "notes.txt"));
    assert!(!inv.has_repo("acme/absent"));

    // A missing root is an empty inventory, never an error.
    let empty = LocalInventory::from_models_dir(&dir.path().join("nope"));
    assert!(!empty.has_repo("acme/tiny"));
}
