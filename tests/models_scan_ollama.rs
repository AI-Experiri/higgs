//! Black-box integration tests for the host-side OLLAMA scan branches of
//! `src/worker/models.rs`, driven through the in-process `Higgs` crate API
//! (`model_entries()` / `model_by_id()` — the library-first replacement for the
//! deleted `GET /api/higgs/models` HTTP control surface).
//!
//! higgs is now a LIBRARY: control + discovery are the in-process `Higgs` facade,
//! not a spawned standalone server. The scan roots are no longer derived from a
//! crafted `$HOME`; they are passed EXPLICITLY on [`HiggsConfig`]
//! (`ollama_dirs` / `lmstudio_dirs`). Each test builds a hermetic model store on
//! disk under a temp dir and points a fresh in-process `Higgs` at it, so we can
//! still assert exact list membership / total counts for the Ollama ids (no stray
//! real store leaks in). An isolated `HIGGS_HOME` keeps the machine's real
//! `~/.higgs` (config / models.json / auth keys) out.
//!
//! The Ollama store is built by hand
//! (`<root>/manifests/.../{name}/{tag}` + `blobs/sha256-<hex>`) and asserts
//! `model_entries()` + `model_by_id()` reflect:
//!   * a VALID manifest whose model-layer blob is the real tiny GGUF — discovered
//!     under `ollama/{name}:{tag}` with `source: Ollama` and the GGUF-header
//!     metadata (arch / ctx_train) read off the blob;
//!   * MULTIPLE `.gguf` variants in one LM-Studio model dir collapsing to a single
//!     id with both paths cataloged, `model_by_id()` returning the lexically-first;
//!   * non-GGUF / garbage manifests (no `layers`, embedding-only layer, binary
//!     `.DS_Store`) silently skipped while the valid model is still returned;
//!   * the HG010 `OllamaManifestInvalid` abort (a model-layer digest with no
//!     `sha256:` prefix, or a layer missing the digest field) surfacing as the
//!     typed `HiggsError::OllamaManifestInvalid` from the scan.

mod common;

use std::path::Path;
use std::sync::Arc;

use common::tiny_gguf_path;
use higgs::worker::models::HiggsModelSource;
use higgs::{Higgs, HiggsConfig, HiggsError};
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// In-process scan harness. Points a fresh `Higgs` at an explicit, hermetic model
// store (the `ollama_dirs` / `lmstudio_dirs` on `HiggsConfig`) and an isolated
// `HIGGS_HOME`, so `model_entries()` scans exactly the crafted store and nothing
// of the developer's real `~/.ollama` / `~/.higgs`. No worker is spawned (scan
// only), so `worker_exe` is left unset.
// ---------------------------------------------------------------------------

/// A running in-process `Higgs` whose entire model-scan surface lives under a
/// crafted store dir. Holds the isolated `HIGGS_HOME` temp dir alive for the
/// instance's lifetime.
struct ScanHiggs {
    higgs: Arc<Higgs>,
    _higgs_home: TempDir,
}

impl std::ops::Deref for ScanHiggs {
    type Target = Arc<Higgs>;
    fn deref(&self) -> &Arc<Higgs> {
        &self.higgs
    }
}

impl ScanHiggs {
    async fn shutdown(self) {
        self.higgs.stop().await;
    }
}

/// Build an in-process `Higgs` scanning EXACTLY the Ollama + LM-Studio roots under
/// `store` (the crafted `<store>/.ollama/models` and `<store>/.lmstudio/models`),
/// under an isolated `HIGGS_HOME`. `HIGGS_HOME` is process-global; this binary runs
/// `--test-threads=1`, so the tests never overlap on it.
async fn scan_higgs(store: &Path) -> ScanHiggs {
    let higgs_home = TempDir::new().expect("create temp HIGGS_HOME");
    // SAFETY: this test binary runs single-threaded (`--test-threads=1`), so no
    // other thread reads/writes the process env concurrently.
    unsafe {
        std::env::set_var("HIGGS_HOME", higgs_home.path());
    }
    let config = HiggsConfig {
        lmstudio_dirs: vec![store.join(".lmstudio").join("models")],
        hf_dirs: vec![],
        ollama_dirs: vec![store.join(".ollama").join("models")],
        default_load: HiggsConfig::default().default_load,
        worker_exe: None,
    };
    let higgs = Arc::new(Higgs::new(config));
    higgs.start().await.expect("higgs start");
    ScanHiggs {
        higgs,
        _higgs_home: higgs_home,
    }
}

// ---------------------------------------------------------------------------
// Ollama-store builders. The Ollama layout is
//   <store>/.ollama/models/manifests/<registry>/<library>/<name>/<tag>   (JSON)
//   <store>/.ollama/models/blobs/sha256-<hex>                            (blob)
// ---------------------------------------------------------------------------

/// Path to the Ollama models root inside a crafted store dir.
fn ollama_root(store: &Path) -> std::path::PathBuf {
    store.join(".ollama").join("models")
}

/// Write a blob at `<ollama>/blobs/sha256-<hex>` from `src` (copied, so the real
/// tiny GGUF magic + header survive verbatim).
fn write_blob_copy(store: &Path, hex: &str, src: &Path) {
    let blobs = ollama_root(store).join("blobs");
    std::fs::create_dir_all(&blobs).expect("create blobs dir");
    std::fs::copy(src, blobs.join(format!("sha256-{hex}"))).expect("copy blob");
}

/// Write a manifest file at `manifests/registry.ollama.ai/library/<name>/<tag>`.
fn write_manifest(store: &Path, name: &str, tag: &str, body: &str) {
    let dir = ollama_root(store)
        .join("manifests")
        .join("registry.ollama.ai")
        .join("library")
        .join(name);
    std::fs::create_dir_all(&dir).expect("create manifest dir");
    std::fs::write(dir.join(tag), body).expect("write manifest");
}

/// A standard Ollama model manifest whose model layer points at `digest`.
fn model_manifest(digest: &str) -> String {
    format!(
        r#"{{"layers":[{{"mediaType":"application/vnd.ollama.image.model","digest":"{digest}"}}]}}"#
    )
}

/// The model ids present in a `model_entries()` result.
fn ids_of(entries: &[higgs::HiggsModelEntry]) -> Vec<String> {
    entries.iter().map(|e| e.model.id.clone()).collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// A VALID Ollama manifest whose model-layer digest resolves to the real tiny
/// GGUF blob is discovered under `ollama/{name}:{tag}` with `source: Ollama`,
/// and its GGUF-header metadata (arch / ctx_train) is read off the content-hashed
/// blob (the Ollama `enrich_gguf_metadata` path — blobs have no filename, so this
/// is the only way arch is learned). `model_by_id()` returns the same enriched
/// entry, and the path ends with `sha256-<hex>`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ollama_valid_manifest_resolves_and_enriches() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("SKIP ollama_valid_manifest_resolves_and_enriches: tiny gguf not found");
        return;
    };
    let store = TempDir::new().unwrap();
    // 64 hex chars is a realistic sha256, but any hex string works — the digest in
    // the manifest just has to match the blob filename.
    let hex = "1111111111111111111111111111111111111111111111111111111111111111";
    write_blob_copy(store.path(), hex, &gguf);
    write_manifest(
        store.path(),
        "tinystories",
        "latest",
        &model_manifest(&format!("sha256:{hex}")),
    );

    let higgs = scan_higgs(store.path()).await;

    let entries = higgs.model_entries().await.expect("scan succeeds");
    let id = "ollama/tinystories:latest";
    let entry = entries
        .iter()
        .find(|e| e.model.id == id)
        .unwrap_or_else(|| panic!("scan lists `{id}`: {:?}", ids_of(&entries)));

    assert_eq!(
        entry.model.source,
        HiggsModelSource::Ollama,
        "blob came from the Ollama store"
    );
    assert_eq!(entry.format, "gguf", "discovered model is gguf");
    assert_eq!(entry.state, "not-loaded", "nothing loaded");
    assert!(
        entry.model.path.ends_with(&format!("sha256-{hex}")),
        "path resolves to the content-hashed blob: {}",
        entry.model.path
    );
    assert!(entry.model.size_bytes > 0, "blob size read from disk");
    // stories260K is a real llama-arch GGUF, so enrichment reads the header off the
    // blob even though the blob has no filename to parse.
    assert_eq!(
        entry.model.arch.as_deref(),
        Some("llama"),
        "general.architecture read off the Ollama blob"
    );
    assert!(
        entry.model.ctx_train.is_some_and(|n| n > 0),
        "llama.context_length read off the blob"
    );

    // The by-id route returns the same enriched entry.
    let detail = higgs
        .model_by_id(id)
        .await
        .expect("by-id for the ollama model resolves");
    assert_eq!(detail.model.id, id, "by-id returns the ollama entry");
    assert_eq!(
        detail.model.source,
        HiggsModelSource::Ollama,
        "by-id source is Ollama"
    );
    assert_eq!(
        detail.model.arch.as_deref(),
        Some("llama"),
        "by-id carries the enriched arch"
    );

    higgs.shutdown().await;
}

/// MULTIPLE `.gguf` files in ONE LM-Studio model dir (two quant variants) collapse
/// to a single id with BOTH path variants cataloged. The list contains exactly one
/// id (`org/multi`) across two rows, `model_by_id()` resolves it (the route picks
/// one), and the two distinct quant tags are both surfaced across the rows.
/// Exercised under the crafted `<store>/.lmstudio/models` root.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn lmstudio_multiple_gguf_in_one_model_dir() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("SKIP lmstudio_multiple_gguf_in_one_model_dir: tiny gguf not found");
        return;
    };
    let store = TempDir::new().unwrap();
    // LM-Studio root is `<store>/.lmstudio/models`. Put `org/multi` there with two
    // quant variants of the SAME real GGUF.
    let model_dir = store.path().join(".lmstudio/models/org/multi");
    std::fs::create_dir_all(&model_dir).unwrap();
    std::fs::copy(&gguf, model_dir.join("multi-Q4_K_M.gguf")).unwrap();
    std::fs::copy(&gguf, model_dir.join("multi-Q8_0.gguf")).unwrap();

    let higgs = scan_higgs(store.path()).await;

    let entries = higgs.model_entries().await.expect("scan succeeds");
    let ids = ids_of(&entries);
    // Exactly one id, but two rows (the two quant variants share the id).
    assert_eq!(
        ids.iter().filter(|id| *id == "org/multi").count(),
        2,
        "both quant variants are cataloged under the same id: {ids:?}"
    );

    // Both quant tags appear across the two rows.
    let quants: Vec<String> = entries
        .iter()
        .filter(|e| e.model.id == "org/multi")
        .filter_map(|e| e.model.quant.clone())
        .collect();
    assert!(
        quants.contains(&"Q4_K_M".to_string()),
        "Q4_K_M variant present: {quants:?}"
    );
    assert!(
        quants.contains(&"Q8_0".to_string()),
        "Q8_0 variant present: {quants:?}"
    );

    // The by-id route resolves the multi-variant id (the route's `.find` returns the
    // first match — the lexically-first path, Q4_K_M).
    let detail = higgs
        .model_by_id("org/multi")
        .await
        .expect("by-id resolves a multi-variant id");
    assert_eq!(detail.model.id, "org/multi", "by-id returns the shared id");
    assert_eq!(
        detail.model.source,
        HiggsModelSource::LmStudio,
        "found under <store>/.lmstudio"
    );
    assert!(
        detail.model.path.contains("Q4_K_M"),
        "by-id returns the lexically-first (Q4_K_M) variant: {}",
        detail.model.path
    );

    higgs.shutdown().await;
}

/// Non-GGUF / garbage manifests must NOT abort the Ollama scan: a manifest with no
/// `layers`, an embedding-only manifest (a layer that is not the model mediaType),
/// a binary `.DS_Store`, and a manifest whose blob is absent are all skipped
/// silently while the ONE valid model is still returned. Drives the silent-skip
/// branches of `scan_ollama` end-to-end (the scan must succeed, not error).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ollama_garbage_manifests_skipped_valid_returned() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("SKIP ollama_garbage_manifests_skipped_valid_returned: tiny gguf not found");
        return;
    };
    let store = TempDir::new().unwrap();

    // One valid model.
    let good_hex = "2222222222222222222222222222222222222222222222222222222222222222";
    write_blob_copy(store.path(), good_hex, &gguf);
    write_manifest(
        store.path(),
        "good",
        "latest",
        &model_manifest(&format!("sha256:{good_hex}")),
    );

    // Vision-style manifest: no `layers` key at all.
    write_manifest(
        store.path(),
        "novision",
        "latest",
        r#"{"config":{"mediaType":"application/vnd.ollama.image"}}"#,
    );

    // Embedding-style manifest: has `layers`, but none with the model mediaType.
    write_manifest(
        store.path(),
        "embed",
        "latest",
        r#"{"layers":[{"mediaType":"application/vnd.ollama.image.params"}]}"#,
    );

    // Manifest whose blob is absent (partial pull) — model layer points nowhere.
    write_manifest(
        store.path(),
        "partialpull",
        "latest",
        &model_manifest("sha256:deadbeefabsent"),
    );

    // A binary file in the manifests tree (.DS_Store) that is not UTF-8 / JSON.
    let ds = ollama_root(store.path())
        .join("manifests")
        .join("registry.ollama.ai")
        .join("library");
    std::fs::create_dir_all(&ds).unwrap();
    std::fs::write(ds.join(".DS_Store"), [0x00u8, 0xFF, 0xFE, 0xFD, 0x42, 0x50]).unwrap();

    let higgs = scan_higgs(store.path()).await;

    // The scan MUST succeed (the garbage does not abort it).
    let entries = higgs
        .model_entries()
        .await
        .expect("garbage manifests must not error the scan");
    let ids = ids_of(&entries);

    // Exactly the one valid model is present; none of the junk produced an entry.
    assert!(
        ids.iter().any(|id| id == "ollama/good:latest"),
        "the valid GGUF manifest is discovered: {ids:?}"
    );
    let ollama_ids: Vec<&String> = ids.iter().filter(|id| id.starts_with("ollama/")).collect();
    assert_eq!(
        ollama_ids,
        vec![&"ollama/good:latest".to_string()],
        "only the valid model yields an ollama id; junk skipped: {ollama_ids:?}"
    );

    higgs.shutdown().await;
}

/// HG010: an Ollama model layer whose `digest` lacks the `sha256:` prefix is an
/// invalid manifest — `scan_ollama` aborts with `OllamaManifestInvalid`, surfaced
/// as the typed `HiggsError::OllamaManifestInvalid` from `model_entries()`. Reaches
/// the HG010 path through the live scan.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ollama_invalid_digest_surfaces_hg010() {
    // No GGUF needed — the digest-format check fires before any blob lookup.
    let store = TempDir::new().unwrap();
    write_manifest(
        store.path(),
        "badigest",
        "latest",
        // `md5:` is not the expected `sha256:` prefix → HG010.
        &model_manifest("md5:zzz"),
    );

    let higgs = scan_higgs(store.path()).await;

    let err = higgs
        .model_entries()
        .await
        .expect_err("an invalid ollama manifest aborts the scan");
    assert!(
        matches!(err, HiggsError::OllamaManifestInvalid { .. }),
        "the invalid manifest surfaces as the typed HG010 error: {err:?}"
    );

    higgs.shutdown().await;
}

/// HG010 variant: a model layer that is entirely MISSING the `digest` field also
/// aborts the scan as `OllamaManifestInvalid` (the `digest.as_str()` None arm). A
/// distinct manifest shape from the bad-prefix case above.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ollama_missing_digest_surfaces_hg010() {
    let store = TempDir::new().unwrap();
    write_manifest(
        store.path(),
        "nodigest",
        "latest",
        // model-mediaType layer with NO digest field at all.
        r#"{"layers":[{"mediaType":"application/vnd.ollama.image.model"}]}"#,
    );

    let higgs = scan_higgs(store.path()).await;

    let err = higgs
        .model_entries()
        .await
        .expect_err("a model layer with no digest aborts the scan");
    assert!(
        matches!(err, HiggsError::OllamaManifestInvalid { .. }),
        "the missing-digest manifest surfaces as the typed HG010 error: {err:?}"
    );

    higgs.shutdown().await;
}
