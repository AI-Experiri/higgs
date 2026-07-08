//! Black-box integration tests for the host-side model SCAN edges
//! (`src/worker/models.rs`), driven through the in-process `Higgs` crate API
//! (`model_entries` / `model_by_id` / `status` / `load` / `unload`).
//!
//! The in-process harness (`common::higgs_local`) stages the tiny
//! `stories260K.gguf` into an isolated LM-Studio-layout scan root
//! (`<root>/{org}/{model}/*.gguf`) and wires it as the ONLY configured scan dir
//! (no HF-cache / Ollama dirs), so these tests see EXACTLY the freshly-staged ids
//! — never the developer's real model dirs. They exercise:
//!   * multi-model discovery — every staged id appears in `model_entries()`;
//!   * graceful tolerance of a NON-gguf file and an EMPTY model dir in a scan root;
//!   * `status()` reflecting EXACTLY the one model that was loaded out of several
//!     staged;
//!   * `model_by_id()` returning the on-disk metadata (arch / quant / ctx_train /
//!     size / source) for a scanned-but-unloaded model, and `ModelNotFound` for an
//!     unknown id.
//!
//! `Higgs::scan` runs a FRESH `ModelStore::default().scan(...)` on every
//! `model_entries()` call, so the pollute-the-root-then-list edge test (junk files
//! written after staging) is observed on the next `model_entries()` — no restart
//! needed.
//!
//! NOTE on Ollama HG010 (`OllamaManifestInvalid`): the shared harness only stages
//! an LM-Studio root; an Ollama manifests root is not reachable here. That case is
//! covered by the in-crate unit tests `ollama_bad_digest_format_errors` /
//! `ollama_missing_digest_errors` in `src/worker/models.rs`.

mod common;

use common::{higgs_local, TINY_MODEL_ID};
use higgs::worker::models::HiggsModelSource;
use higgs::HiggsError;

/// `model_entries()` lists EVERY staged model id (multi-model scan).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn scan_lists_all_staged_models() {
    let ids = ["alpha-org/m-one", "beta-org/m-two", "gamma-org/m-three"];
    let Some(higgs) = higgs_local(&ids).await else {
        eprintln!("SKIP scan_lists_all_staged_models: tiny gguf not found (set HIGGS_TEST_GGUF)");
        return;
    };

    let models = higgs.model_entries().await.expect("model_entries");

    // Each staged id is discovered by the LM-Studio walker, with the on-disk facts
    // the tiny GGUF carries (a real llama-arch header).
    for id in ids {
        let entry = models
            .iter()
            .find(|m| m.model.id == id)
            .unwrap_or_else(|| panic!("scan lists `{id}`"));
        assert_eq!(entry.model.id, id, "id round-trips");
        assert_eq!(entry.format, "gguf", "discovered model is gguf format");
        assert_eq!(
            entry.model.source,
            HiggsModelSource::LmStudio,
            "staged under the LM-Studio scan root → LmStudio source"
        );
        assert_eq!(
            entry.state, "not-loaded",
            "nothing loaded yet, every staged model is not-loaded"
        );
        assert_eq!(
            entry.model.arch.as_deref(),
            Some("llama"),
            "stories260K is a real llama-arch GGUF, so enrichment reads arch"
        );
        assert!(
            entry.model.size_bytes > 0,
            "scanned file size is read from disk"
        );
    }

    higgs.shutdown().await;
}

/// A scan root that also contains a NON-gguf file (at the org level and inside a
/// model dir) and an EMPTY model directory must be handled gracefully: the junk is
/// ignored and the one real model is still discovered. Exercises the LM-Studio
/// walker's `is_dir` / `.gguf`-suffix skip branches. The junk is written into the
/// harness's isolated scan root AFTER staging; the next `model_entries()` re-scans
/// and must tolerate it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn scan_ignores_non_gguf_and_empty_dirs() {
    let Some(higgs) = higgs_local(&["realorg/realmodel"]).await else {
        eprintln!("SKIP scan_ignores_non_gguf_and_empty_dirs: tiny gguf not found");
        return;
    };

    // The staged GGUF lives at `<root>/realorg/realmodel/stories260K.gguf`; walk up
    // to the scan root so we can pollute it with junk the walker must skip.
    let staged = higgs.staged_gguf("realorg/realmodel");
    let root = staged
        .parent() // .../realorg/realmodel
        .and_then(|p| p.parent()) // .../realorg
        .and_then(|p| p.parent()) // scan root
        .expect("scan root")
        .to_path_buf();

    // Pollute the SAME scan root with junk the walker must skip without erroring: a
    // stray top-level file, a non-gguf file inside a model dir, an empty model dir,
    // and a loose file where a model dir is expected.
    std::fs::write(root.join("stray-top-level.txt"), b"not an org dir").unwrap();
    std::fs::write(
        root.join("realorg/realmodel/README.md"),
        b"docs, not a gguf",
    )
    .unwrap();
    std::fs::write(root.join("realorg/realmodel/config.json"), b"{}").unwrap();
    std::fs::create_dir_all(root.join("emptyorg/emptymodel")).unwrap();
    // A file where a model dir is expected — the model-entry `is_dir` guard must skip it.
    std::fs::write(root.join("realorg/loose-file"), b"x").unwrap();

    let models = higgs.model_entries().await.expect("model_entries");

    // The real model is discovered despite the surrounding junk.
    let entry = models
        .iter()
        .find(|m| m.model.id == "realorg/realmodel")
        .expect("real model still found");
    assert_eq!(entry.format, "gguf", "real model still found");
    assert_eq!(
        entry.model.arch.as_deref(),
        Some("llama"),
        "metadata still read"
    );

    // The junk produced no catalog entries: no id derived from the stray names.
    let ids: Vec<&str> = models.iter().map(|m| m.model.id.as_str()).collect();
    assert!(
        !ids.contains(&"emptyorg/emptymodel"),
        "an empty model dir yields no entry: {ids:?}"
    );
    assert!(
        !ids.iter().any(|id| id.starts_with("realorg/loose-file")
            || *id == "realorg/README.md"
            || id.contains("stray-top-level")),
        "non-gguf / loose files yield no entries: {ids:?}"
    );

    higgs.shutdown().await;
}

/// With several models staged, loading exactly ONE makes `status()` report that
/// single model as loaded — and the others stay not-loaded in the models list.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn status_reflects_only_the_loaded_one() {
    let ids = ["load-org/picked", "skip-org/other"];
    let Some(higgs) = higgs_local(&ids).await else {
        eprintln!("SKIP status_reflects_only_the_loaded_one: tiny gguf not found");
        return;
    };

    // Nothing loaded at boot (spawn-on-load).
    let status = higgs.status().await.expect("status");
    assert!(status.loaded.is_none(), "nothing loaded at start");

    // Load just ONE of the staged ids.
    higgs
        .load("load-org/picked", None)
        .await
        .expect("load picked model ok");

    // Status reports exactly the picked model.
    let status = higgs.status().await.expect("status");
    assert_eq!(
        status.loaded.as_ref().map(|l| l.id.as_str()),
        Some("load-org/picked"),
        "status shows the one loaded model"
    );

    // The models list marks the picked one loaded and the other not-loaded.
    let models = higgs.model_entries().await.expect("model_entries");
    let find = |id: &str| {
        models
            .iter()
            .find(|m| m.model.id == id)
            .unwrap_or_else(|| panic!("scan lists `{id}`"))
    };
    assert_eq!(find("load-org/picked").state, "loaded", "picked is loaded");
    assert_eq!(
        find("skip-org/other").state,
        "not-loaded",
        "the unloaded sibling stays not-loaded"
    );

    // Clean unload so graceful teardown is tidy (no model resident).
    higgs.unload().await.expect("unload ok");
    higgs.shutdown().await;
}

/// `model_by_id()` returns the on-disk GGUF metadata (arch / quant / ctx_train /
/// size / source) for a SCANNED-but-unloaded model — the by-id branch that reads
/// the catalog entry without a load probe — and `ModelNotFound` for an unknown id.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn model_by_id_returns_scanned_metadata() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP model_by_id_returns_scanned_metadata: tiny gguf not found");
        return;
    };

    let detail = higgs
        .model_by_id(TINY_MODEL_ID)
        .await
        .expect("by-id for a scanned model succeeds");

    // Identity + the on-disk facts the scanner records.
    assert_eq!(detail.model.id, TINY_MODEL_ID, "by-id returns the entry");
    assert_eq!(detail.format, "gguf", "format is gguf");
    assert_eq!(
        detail.model.source,
        HiggsModelSource::LmStudio,
        "staged under the LM-Studio scan root → LmStudio"
    );
    assert_eq!(detail.state, "not-loaded", "never loaded → not-loaded");
    assert!(detail.model.size_bytes > 0, "size read from disk");

    // GGUF-header-derived enrichment fields (stories260K is a real llama GGUF).
    assert_eq!(
        detail.model.arch.as_deref(),
        Some("llama"),
        "general.architecture extracted from the header"
    );
    assert!(
        detail.model.ctx_train.is_some_and(|n| n > 0),
        "llama.context_length extracted from the header"
    );
    // The harness names the staged file `stories260K.gguf` (no quant suffix that
    // looks like a quant tag), so `quant` is absent (the no-quant filename branch).
    assert!(
        detail.model.quant.is_none(),
        "stories260K.gguf carries no quant tag → quant omitted"
    );

    // A genuinely unknown id is `ModelNotFound` (the not-found by-id branch).
    let err = higgs
        .model_by_id("no-org/no-model")
        .await
        .expect_err("unknown id → ModelNotFound");
    assert!(
        matches!(err, HiggsError::ModelNotFound { .. }),
        "got {err:?}"
    );

    higgs.shutdown().await;
}

/// A CORRUPT `.gguf` (valid magic, truncated body — a real mid-download shape)
/// must still be CATALOGED, but with a coded `[HG070]` enrichment diagnostic on
/// its entry, so the UI can explain the blank header fields rather than showing
/// it as a genuinely sparse model. The VALID staged model in the same scan
/// carries NO `enrich_error` (no false positives).
///
/// Fail-on-revert: drop the `model.enrich_error = …` assignments in
/// `worker/models.rs::enrich_gguf_metadata` / `enrich_from_gguf` and the HG070
/// assertion below fails (the corrupt entry's `enrich_error` is `None`).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn scan_surfaces_corrupt_gguf_as_hg070() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP scan_surfaces_corrupt_gguf_as_hg070: tiny gguf not found");
        return;
    };

    // Walk from the staged VALID model up to the scan root, then plant a corrupt
    // `.gguf` under a fresh `{org}/{model}` dir. `GGuf::new` rejects the malformed
    // header (an out-of-range value read yields `Err`, not a panic here) → the
    // enrichment stamps [HG070] on the entry.
    let staged = higgs.staged_gguf(TINY_MODEL_ID);
    let root = staged
        .parent()
        .and_then(|p| p.parent())
        .and_then(|p| p.parent())
        .expect("scan root")
        .to_path_buf();
    let corrupt_dir = root.join("corrupt-org/corrupt-model");
    std::fs::create_dir_all(&corrupt_dir).expect("mk corrupt dir");
    std::fs::write(
        corrupt_dir.join("broken.gguf"),
        b"GGUF\x03\x00\x00\x00\x01\x02\x03\x04",
    )
    .expect("write corrupt gguf");

    // `Higgs::scan` runs a fresh scan on every `model_entries()`, so the corrupt
    // file is picked up now.
    let models = higgs.model_entries().await.expect("model_entries");

    // Still cataloged, with the coded HG070 diagnostic explaining the failure.
    let corrupt = models
        .iter()
        .find(|m| m.model.id == "corrupt-org/corrupt-model")
        .expect("corrupt gguf is still cataloged despite the parse failure");
    let err = corrupt
        .model
        .enrich_error
        .as_deref()
        .expect("corrupt gguf carries a coded enrich_error");
    assert!(
        err.contains("[HG070]"),
        "coded HG070 diagnostic, got: {err}"
    );

    // The VALID staged model enriched fine → no false-positive diagnostic.
    let valid = models
        .iter()
        .find(|m| m.model.id == TINY_MODEL_ID)
        .expect("valid model listed");
    assert!(
        valid.model.enrich_error.is_none(),
        "valid model carries no enrich_error: {:?}",
        valid.model.enrich_error
    );

    higgs.shutdown().await;
}
