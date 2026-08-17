//! Catalog + model-scan coverage: the catalog service assembly (fit
//! estimates, default-quant pick, enrichment degradation), `HfSource` over a
//! loopback fixture Hub (`HIGGS_HF_ENDPOINT`), and the `ModelStore` scan arms
//! (GGUF header enrichment, projector exclusion, unreadable-dir errors) — all
//! against temp dirs and 127.0.0.1 fixtures, never the real Hub or `~/.higgs`.

use std::collections::{HashMap, HashSet};
use std::future::IntoFuture;
use std::path::Path;
use std::sync::{Arc, Mutex};

use axum::extract::RawQuery;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use huggingface_hub::ModelInfo;
use serde_json::json;

use higgs::catalog::service::{self, LocalInventory};
use higgs::catalog::source::CatalogSource;
use higgs::catalog::wire::{CatalogQuant, CatalogQuery, CatalogSort, MAX_SEARCH_LIMIT};
use higgs::catalog::HfSource;
use higgs::system::FitAssessment;
use higgs::worker::models::ModelStore;
use higgs::{HiggsError, ModelDomain};

/// Serializes the tests that point the process-global `HIGGS_HF_ENDPOINT` at a
/// fixture Hub — they must not overlap each other. A tokio mutex so the guard
/// is await-safe. Tests that never read the env run unlocked.
static ENV_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// Build a crate `ModelInfo` exactly as the Hub API would serialize it.
fn info(v: serde_json::Value) -> ModelInfo {
    serde_json::from_value(v).expect("fixture ModelInfo")
}

fn query(search: &str) -> CatalogQuery {
    CatalogQuery {
        search: search.into(),
        author: None,
        sort: None,
        limit: None,
        compatible_only: None,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ModelDomain display
// ═══════════════════════════════════════════════════════════════════════════

/// The `[HG079]` diagnostics interpolate the domain as a lowercase noun —
/// operators read "this model is an embedding" straight from the message.
#[test]
fn model_domain_display_renders_the_diagnostic_nouns() {
    assert_eq!(ModelDomain::Llm.to_string(), "llm");
    assert_eq!(ModelDomain::Embedding.to_string(), "embedding");
    assert_eq!(ModelDomain::Reranker.to_string(), "reranker");
}

// ═══════════════════════════════════════════════════════════════════════════
// ModelStore scan — GGUF enrichment arms
// ═══════════════════════════════════════════════════════════════════════════

/// Write `content` at `path`, creating parent dirs.
fn write_file(path: &Path, content: &[u8]) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, content).unwrap();
}

/// Raw GGUF bytes (header + metadata KVs, no tensors) via the dev-only `ggus`
/// writer — lets a test stage a real parseable GGUF anywhere on disk.
fn gguf_bytes(kvs: &[(&str, ggus::GGufMetaDataValueType, Vec<u8>)]) -> Vec<u8> {
    use ggus::{GGufFileHeader, GGufFileWriter};
    use std::io::Cursor;
    let header = GGufFileHeader::new(3, 0, kvs.len() as u64);
    let mut buf = Cursor::new(Vec::<u8>::new());
    let mut writer = GGufFileWriter::new(&mut buf, header).unwrap();
    for (key, ty, bytes) in kvs {
        writer.write_meta_kv(key, *ty, bytes).unwrap();
    }
    writer.finish::<Vec<u8>>(false).finish().unwrap();
    buf.into_inner()
}

/// A GGUF metadata string value (u64 length prefix + bytes).
fn gguf_str(s: &str) -> Vec<u8> {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(8 + b.len());
    out.extend_from_slice(&(b.len() as u64).to_le_bytes());
    out.extend_from_slice(b);
    out
}

/// A DIRECTORY carrying a `.gguf` name inside a model dir: the walker treats
/// it as a model file, `open()` succeeds (Unix opens dirs read-only) and the
/// mmap fails (EINVAL) — the entry must stay cataloged with the `[HG070]`
/// mmap diagnostic rather than crash the scan or silently disappear.
#[test]
fn scan_stamps_a_mmap_enrich_error_when_the_gguf_path_is_a_directory() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(dir.path().join("acme/m/weights-Q4_0.gguf")).unwrap();

    let mut store = ModelStore::default();
    let models = store.scan(&[dir.path().to_path_buf()], &[], &[]).unwrap();
    assert_eq!(models.len(), 1);
    assert_eq!(models[0].id, "acme/m");
    let err = models[0].enrich_error.as_deref().expect("mmap diagnostic");
    assert!(err.contains("mmap failed"), "got: {err}");
    assert!(err.contains("HG070"), "coded diagnostic: {err}");
}

/// A full arch-scoped header fills every typed tuning field, the domain stays
/// `Llm` (causal, no pooling), and the template capabilities + curated
/// components come out of the same single parse.
#[test]
fn scan_reads_arch_scoped_tuning_fields_and_template_capabilities() {
    use ggus::GGufMetaDataValueType::{String as GS, U32};
    let dir = tempfile::tempdir().unwrap();
    let bytes = gguf_bytes(&[
        ("general.architecture", GS, gguf_str("llama")),
        ("general.file_type", U32, 15u32.to_le_bytes().to_vec()),
        ("llama.context_length", U32, 4096u32.to_le_bytes().to_vec()),
        ("llama.block_count", U32, 2u32.to_le_bytes().to_vec()),
        (
            "llama.attention.head_count",
            U32,
            4u32.to_le_bytes().to_vec(),
        ),
        (
            "llama.attention.head_count_kv",
            U32,
            2u32.to_le_bytes().to_vec(),
        ),
        ("llama.embedding_length", U32, 64u32.to_le_bytes().to_vec()),
        ("llama.expert_count", U32, 0u32.to_le_bytes().to_vec()),
        (
            "tokenizer.chat_template",
            GS,
            gguf_str("{% if tools %}call{% endif %}<think>"),
        ),
    ]);
    write_file(&dir.path().join("acme/chat/chat-Q4_K_M.gguf"), &bytes);

    let mut store = ModelStore::default();
    let models = store.scan(&[dir.path().to_path_buf()], &[], &[]).unwrap();
    assert_eq!(models.len(), 1);
    let m = &models[0];
    assert_eq!(m.arch.as_deref(), Some("llama"));
    assert_eq!(m.ctx_train, Some(4096));
    assert_eq!(m.block_count, Some(2));
    assert_eq!(m.head_count, Some(4));
    assert_eq!(m.head_count_kv, Some(2));
    assert_eq!(m.embedding_length, Some(64));
    assert_eq!(m.expert_count, Some(0));
    assert_eq!(m.domain, ModelDomain::Llm);
    assert!(m.has_chat_template);
    assert!(m.supports_tools, "template names tools");
    assert!(m.supports_reasoning, "template carries <think>");
    let component = |key: &str| {
        m.gguf_components
            .iter()
            .find(|c| c.key == key)
            .map(|c| c.value.clone())
    };
    assert_eq!(component("general.file_type").as_deref(), Some("Q4_K_M"));
    assert_eq!(component("llama.context_length").as_deref(), Some("4096"));
    assert_eq!(component("llama.block_count").as_deref(), Some("2"));
}

/// llama.cpp's `RANK` pooling (`pooling_type = 4`) declares a reranker — its
/// own domain, distinct from `Embedding`, so a future `/v1/embeddings` never
/// mistakes a scorer for a vector model.
#[test]
fn scan_classifies_rank_pooling_as_a_reranker() {
    use ggus::GGufMetaDataValueType::{String as GS, U32};
    let dir = tempfile::tempdir().unwrap();
    let bytes = gguf_bytes(&[
        ("general.architecture", GS, gguf_str("bert")),
        ("bert.pooling_type", U32, 4u32.to_le_bytes().to_vec()),
    ]);
    write_file(&dir.path().join("acme/rank/rank-Q8_0.gguf"), &bytes);

    let mut store = ModelStore::default();
    let models = store.scan(&[dir.path().to_path_buf()], &[], &[]).unwrap();
    assert_eq!(models.len(), 1);
    assert_eq!(models[0].domain, ModelDomain::Reranker);
}

/// Every `LLAMA_FTYPE` id renders its canonical quant name, and an unknown id
/// degrades to `ftype <n>` instead of hiding the field or failing the scan.
#[test]
fn scan_renders_every_llama_ftype_id_and_degrades_unknown_ids() {
    use ggus::GGufMetaDataValueType::U32;
    let cases: &[(u32, &str)] = &[
        (0, "F32"),
        (3, "Q4_1"),
        (8, "Q5_0"),
        (9, "Q5_1"),
        (10, "Q2_K"),
        (11, "Q3_K_S"),
        (12, "Q3_K_M"),
        (13, "Q3_K_L"),
        (14, "Q4_K_S"),
        (16, "Q5_K_S"),
        (17, "Q5_K_M"),
        (18, "Q6_K"),
        (19, "IQ2_XXS"),
        (20, "IQ2_XS"),
        (21, "Q2_K_S"),
        (22, "IQ3_XS"),
        (23, "IQ3_XXS"),
        (24, "IQ1_S"),
        (25, "IQ4_NL"),
        (26, "IQ3_S"),
        (27, "IQ3_M"),
        (28, "IQ2_S"),
        (29, "IQ2_M"),
        (30, "IQ4_XS"),
        (31, "IQ1_M"),
        (32, "BF16"),
        (36, "TQ1_0"),
        (37, "TQ2_0"),
        (38, "MXFP4_MOE"),
        // Unknown id: a new quant must degrade to an odd label, never break.
        (99, "ftype 99"),
    ];
    let dir = tempfile::tempdir().unwrap();
    for (i, (id, _)) in cases.iter().enumerate() {
        let bytes = gguf_bytes(&[("general.file_type", U32, id.to_le_bytes().to_vec())]);
        write_file(
            &dir.path().join(format!("ft/m{i:02}/model-Q2_K.gguf")),
            &bytes,
        );
    }

    let mut store = ModelStore::default();
    store.scan(&[dir.path().to_path_buf()], &[], &[]).unwrap();
    for (i, (id, label)) in cases.iter().enumerate() {
        let m = store
            .get(&format!("ft/m{i:02}"))
            .unwrap_or_else(|| panic!("model for ftype {id} cataloged"));
        let got = m
            .gguf_components
            .iter()
            .find(|c| c.key == "general.file_type")
            .map(|c| c.value.as_str());
        assert_eq!(got, Some(*label), "ftype {id}");
    }
}

/// A `clip`-arch GGUF is a multimodal projector sidecar, not a servable
/// model — every store layout (LM Studio, HF cache, Ollama) must drop it
/// after enrichment while keeping the real chat models alongside.
#[test]
fn scan_excludes_clip_arch_projectors_in_every_store_layout() {
    use ggus::GGufMetaDataValueType::String as GS;
    let clip = gguf_bytes(&[("general.architecture", GS, gguf_str("clip"))]);
    let chat = gguf_bytes(&[("general.architecture", GS, gguf_str("llama"))]);

    // LM Studio layout: <root>/{org}/{model}/*.gguf
    let lm = tempfile::tempdir().unwrap();
    write_file(&lm.path().join("acme/vis/enc-Q4_0.gguf"), &clip);
    write_file(&lm.path().join("acme/chat/chat-Q4_0.gguf"), &chat);

    // HF cache layout: <root>/models--{org}--{name}/snapshots/{rev}/*.gguf
    let hf = tempfile::tempdir().unwrap();
    write_file(
        &hf.path()
            .join("models--acme--vis2/snapshots/main/e-Q4_0.gguf"),
        &clip,
    );
    write_file(
        &hf.path()
            .join("models--acme--chat2/snapshots/main/c-Q4_0.gguf"),
        &chat,
    );

    // Ollama layout: manifests/**/{name}/{tag} JSON + content-hashed blobs.
    let ol = tempfile::tempdir().unwrap();
    let manifest = |digest: &str| {
        format!(
            r#"{{"layers":[{{"mediaType":"application/vnd.ollama.image.model","digest":"sha256:{digest}"}}]}}"#
        )
    };
    write_file(
        &ol.path().join("manifests/reg/lib/vis/latest"),
        manifest("aaa").as_bytes(),
    );
    write_file(&ol.path().join("blobs/sha256-aaa"), &clip);
    write_file(
        &ol.path().join("manifests/reg/lib/chat/latest"),
        manifest("bbb").as_bytes(),
    );
    write_file(&ol.path().join("blobs/sha256-bbb"), &chat);

    let mut store = ModelStore::default();
    let models = store
        .scan(
            &[lm.path().to_path_buf()],
            &[hf.path().to_path_buf()],
            &[ol.path().to_path_buf()],
        )
        .unwrap();
    let ids: Vec<&str> = models.iter().map(|m| m.id.as_str()).collect();
    assert_eq!(
        ids,
        ["acme/chat", "acme/chat2", "ollama/chat:latest"],
        "projector sidecars excluded in every layout"
    );
}

/// An unreadable directory inside a scan root (permission stripped) surfaces
/// as `[HG001] ModelDirUnreadable` naming the exact dir — for the LM Studio
/// org level, the HF `snapshots` level, and an HF revision dir. Unix-only
/// mechanics (chmod).
#[cfg(unix)]
#[test]
fn scan_errors_hg001_on_unreadable_lmstudio_org_and_hf_dirs() {
    use std::os::unix::fs::PermissionsExt;
    let unreadable = std::fs::Permissions::from_mode(0o000);
    let restore = std::fs::Permissions::from_mode(0o755);

    // LM Studio: the org dir itself is unreadable → model-level read_dir errs.
    let lm = tempfile::tempdir().unwrap();
    let org = lm.path().join("org");
    std::fs::create_dir_all(&org).unwrap();
    std::fs::set_permissions(&org, unreadable.clone()).unwrap();
    let mut store = ModelStore::default();
    // `.map(len)` detaches the Ok borrow so perms restore before asserting.
    let result = store
        .scan(&[lm.path().to_path_buf()], &[], &[])
        .map(<[_]>::len);
    std::fs::set_permissions(&org, restore.clone()).unwrap();
    let err = result.expect_err("unreadable org dir");
    assert!(
        matches!(err, HiggsError::ModelDirUnreadable { ref path, .. } if path.ends_with("org")),
        "got: {err}"
    );

    // HF cache: the snapshots dir is unreadable.
    let hf = tempfile::tempdir().unwrap();
    let snapshots = hf.path().join("models--acme--m/snapshots");
    std::fs::create_dir_all(&snapshots).unwrap();
    std::fs::set_permissions(&snapshots, unreadable.clone()).unwrap();
    let mut store = ModelStore::default();
    let result = store
        .scan(&[], &[hf.path().to_path_buf()], &[])
        .map(<[_]>::len);
    std::fs::set_permissions(&snapshots, restore.clone()).unwrap();
    let err = result.expect_err("unreadable snapshots dir");
    assert!(
        matches!(err, HiggsError::ModelDirUnreadable { ref path, .. }
            if path.ends_with("snapshots")),
        "got: {err}"
    );

    // HF cache: one revision dir is unreadable.
    let hf2 = tempfile::tempdir().unwrap();
    let rev = hf2.path().join("models--acme--m/snapshots/rev");
    std::fs::create_dir_all(&rev).unwrap();
    std::fs::set_permissions(&rev, unreadable).unwrap();
    let mut store = ModelStore::default();
    let result = store
        .scan(&[], &[hf2.path().to_path_buf()], &[])
        .map(<[_]>::len);
    std::fs::set_permissions(&rev, restore).unwrap();
    let err = result.expect_err("unreadable revision dir");
    assert!(
        matches!(err, HiggsError::ModelDirUnreadable { ref path, .. } if path.ends_with("rev")),
        "got: {err}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// LocalInventory
// ═══════════════════════════════════════════════════════════════════════════

/// The inventory walk mirrors the models layout: GGUFs register repo + exact
/// file, non-GGUFs don't, a stray file at the org level is skipped (its
/// read_dir fails), and a missing root reads as an empty inventory.
#[test]
fn inventory_walks_the_models_layout_and_tolerates_stray_files() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    write_file(&root.join("acme/tiny/tiny-Q4_K_M.gguf"), b"x");
    write_file(&root.join("acme/tiny/notes.txt"), b"x");
    // A plain file where an org dir is expected: read_dir on it fails and the
    // walk moves on instead of aborting the inventory.
    write_file(&root.join("stray.txt"), b"x");

    let inv = LocalInventory::from_models_dir(root);
    assert!(inv.has_repo("acme/tiny"));
    assert!(inv.has_file("acme/tiny", "tiny-Q4_K_M.gguf"));
    assert!(!inv.has_file("acme/tiny", "notes.txt"), "non-GGUF ignored");

    let empty = LocalInventory::from_models_dir(&root.join("does-not-exist"));
    assert!(!empty.has_repo("acme/tiny"), "missing root reads empty");
}

// ═══════════════════════════════════════════════════════════════════════════
// Pure service mapping — fit estimates, quant rows, default pick
// ═══════════════════════════════════════════════════════════════════════════

/// The pre-download row estimate prices EVERY recognized quant family
/// (I-quants get their own, cheaper, numbers) and takes the cheapest;
/// unrecognized labels contribute nothing. No VRAM figure → no verdict.
#[test]
fn summary_estimates_fit_from_the_cheapest_recognized_quant_family() {
    let m = info(json!({
        "id": "acme/m",
        "gguf": { "total": 1_000_000u64 },
        "siblings": [
            { "rfilename": "m-IQ1_S.gguf" },
            { "rfilename": "m-IQ2_XS.gguf" },
            { "rfilename": "m-IQ3_M.gguf" },
            { "rfilename": "m-IQ4_NL.gguf" },
            { "rfilename": "m-IQX.gguf" },  // unknown I-quant digit → no number
            { "rfilename": "m-F32.gguf" },
            { "rfilename": "m-Q9_0.gguf" }, // unknown family → no number
        ],
    }));
    let fit = service::summary_from(&m, &LocalInventory::empty(), Some(10_000_000))
        .fit
        .expect("estimate with params + labels + vram");
    assert_eq!(
        fit.needed_bytes, 200_000,
        "cheapest family is IQ1 at 0.2 B/param"
    );
    assert!(fit.fits);

    let tight = service::summary_from(&m, &LocalInventory::empty(), Some(100_000))
        .fit
        .expect("estimate against small vram");
    assert!(!tight.fits, "100kB VRAM cannot hold a 200kB quant");

    let no_hw = service::summary_from(&m, &LocalInventory::empty(), None);
    assert!(no_hw.fit.is_none(), "no VRAM figure → no verdict");
}

/// Quant-picker rows get a fit verdict only when BOTH the file size and the
/// VRAM total are known — an unknown never gets a fake verdict.
#[test]
fn quant_rows_get_a_verdict_only_when_size_and_vram_are_both_known() {
    let m = info(json!({
        "id": "acme/m",
        "siblings": [
            { "rfilename": "m-Q4_K_M.gguf" },
            { "rfilename": "m-F16.gguf" }, // no size anywhere → no verdict
        ],
    }));
    let sizes = HashMap::from([("m-Q4_K_M.gguf".to_string(), 4_000u64)]);
    let quants = service::quants_from(&m, &sizes, &LocalInventory::empty(), Some(1_000_000));
    assert_eq!(quants[0].file, "m-Q4_K_M.gguf");
    let fit = quants[0].fit.as_ref().expect("sized row gets a verdict");
    assert!(fit.fits);
    assert_eq!(fit.needed_bytes, 4_000);
    assert!(quants[1].fit.is_none(), "unsized row never gets a verdict");
}

/// A quant row assembled by hand — `default_quant` is a pub fn over rows.
fn cq(file: &str, size: Option<u64>, fits: Option<bool>) -> CatalogQuant {
    CatalogQuant {
        file: file.to_string(),
        quant: None,
        size_bytes: size,
        downloaded: false,
        fit: fits.map(|f| FitAssessment {
            fits: f,
            needed_bytes: size.unwrap_or(0),
            available_bytes: 0,
        }),
    }
}

/// Without a `Q4_K_M`, the default pick is the LARGEST quant this machine
/// fits — best quality that actually loads.
#[test]
fn default_pick_falls_back_to_the_largest_quant_that_fits() {
    let quants = vec![
        cq("m-Q2_K.gguf", Some(2_000), Some(true)),
        cq("m-Q6_K.gguf", Some(6_000), Some(true)),
        cq("m-F16.gguf", Some(16_000), Some(false)),
    ];
    assert_eq!(
        service::default_quant(&quants).expect("pick").file,
        "m-Q6_K.gguf"
    );
}

/// A shards-only repo has no whole file to prefer, so the pick falls back to
/// the full set and takes the smallest KNOWN size (the documented per-file
/// limitation).
#[test]
fn default_pick_of_a_shards_only_repo_uses_the_smallest_known_shard() {
    let quants = vec![
        cq("m-00001-of-00002.gguf", Some(100), None),
        cq("m-00002-of-00002.gguf", Some(90), None),
    ];
    assert_eq!(
        service::default_quant(&quants).expect("pick").file,
        "m-00002-of-00002.gguf"
    );
}

/// A row whose file name has no `.gguf` suffix can never be a shard part, so
/// it competes as a whole file in the smallest-known fallback.
#[test]
fn default_pick_treats_a_non_gguf_name_as_a_whole_file() {
    let quants = vec![
        cq("weird-noext", Some(5), None),
        cq("m-Q6_K.gguf", Some(100), None),
    ];
    assert_eq!(
        service::default_quant(&quants).expect("pick").file,
        "weird-noext"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// service::search / service::detail over an injected source
// ═══════════════════════════════════════════════════════════════════════════

/// Canned [`CatalogSource`]: fixed responses, per-repo failure switches, and
/// call logs so tests can assert exactly which Hub reads happened.
#[derive(Default)]
struct FakeSource {
    hits: Vec<serde_json::Value>,
    infos: HashMap<String, serde_json::Value>,
    fail_infos: HashSet<String>,
    sizes: HashMap<String, u64>,
    fail_sizes_for: HashSet<String>,
    readme: Option<String>,
    fail_readme: bool,
    /// README fetch black-holes: never resolves, never errors.
    hang_readme: bool,
    fail_author_search: bool,
    searches: Mutex<Vec<CatalogQuery>>,
    info_calls: Mutex<Vec<String>>,
}

fn injected_err() -> HiggsError {
    HiggsError::HubTransport {
        repo: "fake".into(),
        detail: "injected".into(),
    }
}

impl CatalogSource for FakeSource {
    async fn search(&self, q: &CatalogQuery) -> Result<Vec<ModelInfo>, HiggsError> {
        self.searches.lock().unwrap().push(q.clone());
        if q.author.is_some() && self.fail_author_search {
            return Err(injected_err());
        }
        Ok(self.hits.iter().cloned().map(info).collect())
    }

    async fn info(&self, repo: &str) -> Result<ModelInfo, HiggsError> {
        self.info_calls.lock().unwrap().push(repo.to_owned());
        if self.fail_infos.contains(repo) {
            return Err(injected_err());
        }
        self.infos
            .get(repo)
            .cloned()
            .map(info)
            .ok_or_else(injected_err)
    }

    async fn file_sizes(
        &self,
        repo: &str,
        _files: &[String],
    ) -> Result<HashMap<String, u64>, HiggsError> {
        if self.fail_sizes_for.contains(repo) {
            return Err(injected_err());
        }
        Ok(self.sizes.clone())
    }

    async fn readme(&self, _repo: &str) -> Result<Option<String>, HiggsError> {
        if self.hang_readme {
            std::future::pending::<()>().await;
        }
        if self.fail_readme {
            return Err(injected_err());
        }
        Ok(self.readme.clone())
    }
}

/// `compatible_only` over-asks the Hub (the hard cap, since filtering loses
/// rows), drops rows the ESTIMATE says don't fit, drops rows whose REAL
/// resolved size says don't fit (the metadata-less fallback), and keeps rows
/// with no verdict at all — absence of data is never evidence of misfit.
#[tokio::test]
async fn compatible_search_over_asks_drops_misfits_and_keeps_unknowns() {
    let src = FakeSource {
        hits: vec![
            // Estimate says fits (1000 params × 0.66 ≈ 660 B) — kept. The list
            // row already carries its gguf block, so no info fetch happens.
            json!({ "id": "acme/small", "gguf": { "total": 1_000u64 },
                    "siblings": [{ "rfilename": "s-Q4_K_M.gguf" }] }),
            // Estimate says misfit (10^12 params) after info enrichment — dropped.
            json!({ "id": "acme/big" }),
            // No estimate inputs, but a REAL sibling size (900 GB) — dropped
            // via the real-size fallback verdict.
            json!({ "id": "acme/realbig",
                    "siblings": [{ "rfilename": "r-Q4_0.gguf", "size": 900_000_000_000u64 }] }),
            // No metadata, no siblings → no verdict either way — kept.
            json!({ "id": "acme/unknown" }),
        ],
        infos: HashMap::from([(
            "acme/big".to_string(),
            json!({ "id": "acme/big", "gguf": { "total": 1_000_000_000_000u64 },
                    "siblings": [{ "rfilename": "b-Q4_K_M.gguf" }] }),
        )]),
        fail_infos: HashSet::from(["acme/realbig".to_string(), "acme/unknown".to_string()]),
        ..Default::default()
    };
    let q = CatalogQuery {
        compatible_only: Some(true),
        limit: Some(2),
        ..query("x")
    };
    let resp = service::search(&src, &q, &LocalInventory::empty(), Some(1_000_000))
        .await
        .expect("search");
    let ids: Vec<&str> = resp.models.iter().map(|m| m.id.as_str()).collect();
    assert_eq!(ids, ["acme/small", "acme/unknown"]);
    let sent = src.searches.lock().unwrap();
    assert_eq!(
        sent[0].limit,
        Some(MAX_SEARCH_LIMIT),
        "filtering loses rows, so the Hub is over-asked to the hard cap"
    );
}

/// Rows the list endpoint already enriched are NOT re-fetched, and a row
/// whose info fetch fails goes out exactly as the list returned it.
#[tokio::test]
async fn search_skips_info_for_pre_enriched_rows_and_keeps_bare_rows_on_failure() {
    let src = FakeSource {
        hits: vec![
            json!({ "id": "acme/pre",
                    "gguf": { "architecture": "llama", "total": 1_000u64 } }),
            json!({ "id": "acme/bare", "downloads": 7u64 }),
        ],
        fail_infos: HashSet::from(["acme/bare".to_string()]),
        ..Default::default()
    };
    let resp = service::search(&src, &query("x"), &LocalInventory::empty(), None)
        .await
        .expect("search");
    assert_eq!(resp.models.len(), 2);
    assert_eq!(
        resp.models[0].gguf.as_ref().expect("badge").arch.as_deref(),
        Some("llama")
    );
    assert_eq!(
        resp.models[1].downloads,
        Some(7),
        "bare row keeps its stats"
    );
    assert_eq!(
        *src.info_calls.lock().unwrap(),
        vec!["acme/bare".to_string()],
        "only the un-enriched row is fetched"
    );
}

/// When a metadata-less row's siblings carry no sizes, ONE bounded paths-info
/// fetch resolves them; a failed fetch leaves the row an honest unknown
/// (kept), never a fake verdict.
#[tokio::test]
async fn search_resolves_real_sizes_via_paths_info_and_degrades_on_failure() {
    let src = FakeSource {
        hits: vec![
            json!({ "id": "acme/fetched",
                    "siblings": [{ "rfilename": "f-Q4_0.gguf" }] }),
            json!({ "id": "acme/failfetch",
                    "siblings": [{ "rfilename": "g-Q4_0.gguf" }] }),
        ],
        fail_infos: HashSet::from(["acme/fetched".to_string(), "acme/failfetch".to_string()]),
        sizes: HashMap::from([("f-Q4_0.gguf".to_string(), 900_000_000u64)]),
        fail_sizes_for: HashSet::from(["acme/failfetch".to_string()]),
        ..Default::default()
    };
    let q = CatalogQuery {
        compatible_only: Some(true),
        ..query("x")
    };
    let resp = service::search(&src, &q, &LocalInventory::empty(), Some(1_000_000))
        .await
        .expect("search");
    let ids: Vec<&str> = resp.models.iter().map(|m| m.id.as_str()).collect();
    assert_eq!(
        ids,
        ["acme/failfetch"],
        "paths-info verdict drops the 900MB misfit; the failed fetch stays unknown and kept"
    );
}

/// Each detail enrichment (file sizes, README, author browse) degrades
/// independently on failure — the pane still assembles from what's left.
#[tokio::test]
async fn detail_degrades_each_failed_enrichment_independently() {
    let src = FakeSource {
        infos: HashMap::from([(
            "acme/m".to_string(),
            json!({ "id": "acme/m", "author": "acme",
                    "siblings": [{ "rfilename": "m-Q4_K_M.gguf", "lfs": { "size": 4_000u64 } }] }),
        )]),
        fail_sizes_for: HashSet::from(["acme/m".to_string()]),
        fail_readme: true,
        fail_author_search: true,
        ..Default::default()
    };
    let d = service::detail(&src, "acme/m", &LocalInventory::empty(), None)
        .await
        .expect("detail assembles despite three failed enrichments");
    assert_eq!(d.summary.id, "acme/m");
    assert_eq!(
        d.quants[0].size_bytes,
        Some(4_000),
        "sibling LFS size stands in for the failed paths-info"
    );
    assert!(d.readme.is_none(), "failed README degrades to none");
    assert!(
        d.more_by_author.is_empty(),
        "failed browse degrades to none"
    );
    assert_eq!(d.default_file.as_deref(), Some("m-Q4_K_M.gguf"));
}

/// A black-holed README fetch is cut off by the enrichment timeout (paused
/// clock — no real waiting) while the other enrichments land normally: a
/// stalled part degrades exactly like a failed one.
#[tokio::test(start_paused = true)]
async fn detail_times_out_a_stalled_readme_instead_of_holding_the_pane() {
    let src = FakeSource {
        infos: HashMap::from([(
            "acme/m".to_string(),
            json!({ "id": "acme/m", "author": "acme",
                    "siblings": [{ "rfilename": "m-Q4_K_M.gguf", "size": 4_000u64 }] }),
        )]),
        hits: vec![json!({ "id": "acme/other" })],
        readme: Some("never delivered".into()),
        hang_readme: true,
        ..Default::default()
    };
    let d = service::detail(&src, "acme/m", &LocalInventory::empty(), None)
        .await
        .expect("detail");
    assert!(d.readme.is_none(), "stalled README timed out to none");
    let more: Vec<&str> = d.more_by_author.iter().map(|m| m.id.as_str()).collect();
    assert_eq!(more, ["acme/other"], "author browse still landed");
}

/// A repo with no derivable author gets no author browse — and therefore no
/// Hub search call at all.
#[tokio::test]
async fn detail_without_an_author_makes_no_author_browse_call() {
    let src = FakeSource {
        // An id without an `org/` prefix leaves the author underivable.
        infos: HashMap::from([("solo".to_string(), json!({ "id": "solo" }))]),
        ..Default::default()
    };
    let d = service::detail(&src, "solo", &LocalInventory::empty(), None)
        .await
        .expect("detail");
    assert!(d.more_by_author.is_empty());
    assert!(
        src.searches.lock().unwrap().is_empty(),
        "no author → no search call"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// HfSource over a loopback fixture Hub (HIGGS_HF_ENDPOINT)
// ═══════════════════════════════════════════════════════════════════════════

/// Spawn a loopback "Hugging Face": the model list (with per-query failure
/// modes), paths-info (with a directory entry), and README resolve downloads
/// (missing / oversized / erroring variants). Returns the endpoint URL and
/// the log of list query strings.
async fn fixture_hub() -> (String, Arc<Mutex<Vec<String>>>) {
    let queries = Arc::new(Mutex::new(Vec::<String>::new()));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    let list_log = queries.clone();
    let app = axum::Router::new()
        .route(
            "/api/models",
            get(move |RawQuery(q): RawQuery| {
                let log = list_log.clone();
                async move {
                    let q = q.unwrap_or_default();
                    log.lock().unwrap().push(q.clone());
                    if q.contains("search=boom") {
                        return (axum::http::StatusCode::INTERNAL_SERVER_ERROR, "kaput")
                            .into_response();
                    }
                    if q.contains("search=garbage") {
                        return "{ this is not json".into_response();
                    }
                    axum::Json(json!([{ "id": "acme/tiny" }, { "id": "acme/big" }])).into_response()
                }
            }),
        )
        .route(
            "/api/models/{org}/{name}/paths-info/{rev}",
            post(|| async {
                axum::Json(json!([
                    // LFS file: the pointer-blob `size` must NOT win over the
                    // real LFS payload size.
                    { "type": "file", "oid": "a", "size": 134u64,
                      "path": "a-Q4_K_M.gguf", "lfs": { "size": 4_000u64 } },
                    { "type": "file", "oid": "b", "size": 9_000u64, "path": "b-F16.gguf" },
                    // Directories carry no size and must be skipped.
                    { "type": "directory", "oid": "d", "path": "sub" },
                ]))
            }),
        )
        .route(
            "/{org}/{name}/resolve/{rev}/{file}",
            get(
                |axum::extract::Path((_, name, _, file)): axum::extract::Path<(
                    String,
                    String,
                    String,
                    String,
                )>| async move {
                    match (name.as_str(), file.as_str()) {
                        // No README is a normal repo state → 404.
                        ("noreadme", "README.md") => {
                            (axum::http::StatusCode::NOT_FOUND, "missing").into_response()
                        }
                        // 1 MiB - 1 ASCII bytes then one multibyte char that
                        // straddles the byte cap — exercises the char-boundary
                        // clip on the bounded README read.
                        ("bigreadme", "README.md") => {
                            let mut body = vec![b'a'; 1024 * 1024 - 1];
                            body.extend_from_slice("€".as_bytes());
                            body.into_response()
                        }
                        ("errreadme", "README.md") => {
                            (axum::http::StatusCode::INTERNAL_SERVER_ERROR, "boom").into_response()
                        }
                        _ => "# tiny\nhello".into_response(),
                    }
                },
            ),
        );
    tokio::spawn(axum::serve(listener, app).into_future());
    (format!("http://{addr}"), queries)
}

/// Each catalog sort order maps to the Hub API's own sort key, and a Hub 500
/// / non-JSON body classifies into a coded error instead of a panic or hang.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn hf_search_maps_sort_keys_and_classifies_bad_hub_responses() {
    let _env = ENV_LOCK.lock().await;
    let (endpoint, queries) = fixture_hub().await;
    unsafe { std::env::set_var("HIGGS_HF_ENDPOINT", &endpoint) };

    let mut q = query("tiny");
    q.sort = Some(CatalogSort::Updated);
    HfSource.search(&q).await.expect("updated sort");
    q.sort = Some(CatalogSort::Trending);
    HfSource.search(&q).await.expect("trending sort");
    {
        let log = queries.lock().unwrap();
        assert!(log[0].contains("sort=lastModified"), "got: {}", log[0]);
        assert!(log[1].contains("sort=trendingScore"), "got: {}", log[1]);
    }

    let err = HfSource
        .search(&query("boom"))
        .await
        .expect_err("500 classified");
    assert!(
        matches!(err, HiggsError::HubHttpStatus { status: 500, .. }),
        "got: {err}"
    );

    let err = HfSource
        .search(&query("garbage"))
        .await
        .expect_err("garbage JSON classified");
    assert!(
        matches!(
            err,
            HiggsError::HubClient { .. } | HiggsError::HubTransport { .. }
        ),
        "got: {err}"
    );
}

/// A repo id that is not exactly two safe segments is refused BEFORE any
/// request (the id is interpolated into the URL path), and an empty
/// file-sizes ask short-circuits to an empty map without a Hub call.
#[tokio::test]
async fn hf_reads_refuse_a_malformed_repo_id_before_any_request() {
    // No endpoint set up: these calls must fail/return before any HTTP.
    let err = HfSource
        .info("not-a-repo-id")
        .await
        .expect_err("one segment refused");
    assert!(
        matches!(err, HiggsError::HubClient { ref detail, .. } if detail.contains("org/model")),
        "got: {err}"
    );

    let sizes = HfSource
        .file_sizes("acme/tiny", &[])
        .await
        .expect("empty ask");
    assert!(sizes.is_empty());
}

/// paths-info sizes prefer the LFS payload size over the pointer-blob entry
/// size, and directory entries are skipped — they carry no size at all.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn hf_file_sizes_prefer_lfs_and_skip_directory_entries() {
    let _env = ENV_LOCK.lock().await;
    let (endpoint, _) = fixture_hub().await;
    unsafe { std::env::set_var("HIGGS_HF_ENDPOINT", &endpoint) };

    let files = vec![
        "a-Q4_K_M.gguf".to_string(),
        "b-F16.gguf".to_string(),
        "sub".to_string(),
    ];
    let sizes = HfSource
        .file_sizes("acme/tiny", &files)
        .await
        .expect("sizes");
    assert_eq!(sizes.get("a-Q4_K_M.gguf"), Some(&4_000), "LFS real size");
    assert_eq!(sizes.get("b-F16.gguf"), Some(&9_000), "plain entry size");
    assert!(!sizes.contains_key("sub"), "directories carry no size");
}

/// The README read is byte-bounded ON ALLOCATION and clipped to a char
/// boundary; a missing README is `None` (a normal repo state), and a Hub 500
/// is a real coded error.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn hf_readme_is_bounded_none_on_404_and_coded_error_on_500() {
    let _env = ENV_LOCK.lock().await;
    let (endpoint, _) = fixture_hub().await;
    unsafe { std::env::set_var("HIGGS_HF_ENDPOINT", &endpoint) };

    // 1 MiB - 1 'a's + '€': the cap lands mid-char, so the clip must back up
    // to the last char boundary — never split a UTF-8 sequence.
    let big = HfSource
        .readme("acme/bigreadme")
        .await
        .expect("bounded read")
        .expect("some readme");
    assert_eq!(big.len(), 1024 * 1024 - 1);
    assert!(
        big.bytes().all(|b| b == b'a'),
        "no partial/replacement char"
    );

    let none = HfSource.readme("acme/noreadme").await.expect("404 is ok");
    assert!(none.is_none(), "no README is a normal repo state");

    let err = HfSource
        .readme("acme/errreadme")
        .await
        .expect_err("500 is a failure");
    assert!(
        matches!(err, HiggsError::HubHttpStatus { status: 500, .. }),
        "got: {err}"
    );
}
