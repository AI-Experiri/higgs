//! Black-box integration tests for the host-side model SCAN edges
//! (`src/worker/models.rs`), driven entirely over HTTP against a real spawned
//! `higgs`.
//!
//! The harness stages the tiny `stories260K.gguf` into an LM-Studio-layout scan
//! root (`<root>/{org}/{model}/*.gguf`) and points the server at it via
//! `HIGGS_MODEL_DIR`. These tests exercise:
//!   * multi-model discovery — every staged id appears in `GET /api/higgs/models`;
//!   * graceful tolerance of a NON-gguf file and an EMPTY model dir in a scan root;
//!   * `GET /api/higgs/status` reflecting EXACTLY the one model that was loaded out
//!     of several staged;
//!   * `GET /api/higgs/models/{id}` returning the on-disk metadata (arch / quant /
//!     size / source) for a scanned-but-unloaded model.
//!
//! NOTE on Ollama HG010 (`OllamaManifestInvalid`): the shared harness only stages
//! an LM-Studio root via `HIGGS_MODEL_DIR`; the standalone binary exposes no env
//! override for an Ollama manifests root (only `HiggsConfig::default()`'s real
//! `~/.ollama/models`, which a test must not touch). So a black-box Ollama
//! invalid-manifest scan is NOT reachable from this surface — it is covered by the
//! in-crate unit tests `ollama_bad_digest_format_errors` / `ollama_missing_digest_errors`
//! in `src/worker/models.rs`. See the structured-result `notes` for the rationale.
//!
//! The default `HiggsConfig` ALSO scans the developer's real `~/.cache/huggingface`,
//! `~/.ollama`, and LM-Studio dirs, so these tests assert only on the PRESENCE of
//! their freshly-staged ids — never on total counts or exact list membership.

mod common;

use std::path::Path;
use std::process::{Child, Command};
use std::time::Duration;

use common::{spawn_with_models, stage_models, tiny_gguf_path, TINY_MODEL_ID};
use tempfile::TempDir;

/// Find the scanned entry for `id` in a `GET /api/higgs/models` response, or fail.
fn find_entry<'a>(models: &'a serde_json::Value, id: &str) -> &'a serde_json::Value {
    models["models"]
        .as_array()
        .expect("models is an array")
        .iter()
        .find(|m| m["id"] == serde_json::json!(id))
        .unwrap_or_else(|| panic!("scan lists `{id}`: {models}"))
}

/// `GET /api/higgs/models` lists EVERY staged model id (multi-model scan).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn scan_lists_all_staged_models() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("SKIP scan_lists_all_staged_models: tiny gguf not found (set HIGGS_TEST_GGUF)");
        return;
    };
    let ids = ["alpha-org/m-one", "beta-org/m-two", "gamma-org/m-three"];
    let srv = spawn_with_models(12400, &gguf, &ids).await;
    let c = reqwest::Client::new();

    let models: serde_json::Value = c
        .get(format!("{}/api/higgs/models", srv.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();

    // Each staged id is discovered by the LM-Studio walker, with the on-disk facts
    // the tiny GGUF carries (a real llama-arch header).
    for id in ids {
        let entry = find_entry(&models, id);
        assert_eq!(entry["id"], serde_json::json!(id), "id round-trips");
        assert_eq!(
            entry["format"], "gguf",
            "discovered model is gguf format: {entry}"
        );
        assert_eq!(
            entry["source"], "LmStudio",
            "staged under HIGGS_MODEL_DIR → LmStudio source: {entry}"
        );
        assert_eq!(
            entry["state"], "not-loaded",
            "nothing loaded yet, every staged model is not-loaded: {entry}"
        );
        assert_eq!(
            entry["arch"], "llama",
            "stories260K is a real llama-arch GGUF, so enrichment reads arch: {entry}"
        );
        assert!(
            entry["size_bytes"].as_u64().unwrap() > 0,
            "scanned file size is read from disk: {entry}"
        );
    }
}

/// A scan root that also contains a NON-gguf file (at the org level and inside a
/// model dir) and an EMPTY model directory must be handled gracefully: the junk is
/// ignored and the one real model is still discovered. Exercises the LM-Studio
/// walker's `is_dir` / `.gguf`-suffix skip branches.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn scan_ignores_non_gguf_and_empty_dirs() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("SKIP scan_ignores_non_gguf_and_empty_dirs: tiny gguf not found");
        return;
    };

    // Stage one real model, then pollute the SAME scan root with junk the walker
    // must skip without erroring: a stray top-level file, a non-gguf file inside a
    // model dir, and an empty model dir with no files at all.
    let staged = stage_models(&gguf, &["realorg/realmodel"]);
    let root = staged.path();
    std::fs::write(root.join("stray-top-level.txt"), b"not an org dir").unwrap();
    std::fs::write(
        root.join("realorg/realmodel/README.md"),
        b"docs, not a gguf",
    )
    .unwrap();
    std::fs::write(root.join("realorg/realmodel/config.json"), b"{}").unwrap();
    std::fs::create_dir_all(root.join("emptyorg/emptymodel")).unwrap();
    // An org dir that is itself a file at the model level (file where a model dir
    // is expected) — the model-entry `is_dir` guard must skip it.
    std::fs::write(root.join("realorg/loose-file"), b"x").unwrap();

    let srv = spawn_at_root(12401, root).await;
    let c = reqwest::Client::new();

    let models: serde_json::Value = c
        .get(format!("{}/api/higgs/models", srv.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();

    // The real model is discovered despite the surrounding junk.
    let entry = find_entry(&models, "realorg/realmodel");
    assert_eq!(entry["format"], "gguf", "real model still found: {entry}");
    assert_eq!(entry["arch"], "llama", "metadata still read: {entry}");

    // The junk produced no catalog entries: no id derived from the stray names.
    let ids: Vec<String> = models["models"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["id"].as_str().unwrap_or_default().to_string())
        .collect();
    assert!(
        !ids.iter().any(|id| id == "emptyorg/emptymodel"),
        "an empty model dir yields no entry: {ids:?}"
    );
    assert!(
        !ids.iter().any(|id| id.starts_with("realorg/loose-file")
            || id == "realorg/README.md"
            || id.contains("stray-top-level")),
        "non-gguf / loose files yield no entries: {ids:?}"
    );

    // Keep the staging dir alive until the server is dropped (it reads from `root`).
    drop(srv);
    drop(staged);
}

/// With several models staged, loading exactly ONE makes `GET /api/higgs/status`
/// report that single model as loaded — and the others stay not-loaded in the
/// models list.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn status_reflects_only_the_loaded_one() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("SKIP status_reflects_only_the_loaded_one: tiny gguf not found");
        return;
    };
    let ids = ["load-org/picked", "skip-org/other"];
    let srv = spawn_with_models(12402, &gguf, &ids).await;
    let c = reqwest::Client::new();

    // Nothing loaded at boot (spawn-on-load).
    let status: serde_json::Value = c
        .get(format!("{}/api/higgs/status", srv.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(
        status["loaded"].is_null(),
        "nothing loaded at start: {status}"
    );

    // Load just ONE of the staged ids.
    let load = c
        .post(format!("{}/api/higgs/models/load", srv.base))
        .json(&serde_json::json!({ "id": "load-org/picked" }))
        .send()
        .await
        .unwrap();
    assert!(load.status().is_success(), "load picked model ok");

    // Status reports exactly the picked model.
    let status: serde_json::Value = c
        .get(format!("{}/api/higgs/status", srv.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        status["loaded"]["id"], "load-org/picked",
        "status shows the one loaded model: {status}"
    );

    // The models list marks the picked one loaded and the other not-loaded.
    let models: serde_json::Value = c
        .get(format!("{}/api/higgs/models", srv.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        find_entry(&models, "load-org/picked")["state"],
        "loaded",
        "picked entry is loaded"
    );
    assert_eq!(
        find_entry(&models, "skip-org/other")["state"],
        "not-loaded",
        "the unloaded sibling stays not-loaded"
    );

    // Clean unload so graceful shutdown is tidy (no model resident, no SSE open).
    let _ = c
        .post(format!("{}/api/higgs/models/unload", srv.base))
        .json(&serde_json::json!({}))
        .send()
        .await
        .unwrap();
}

/// `GET /api/higgs/models/{id}` returns the on-disk GGUF metadata (arch / quant /
/// ctx_train / size / source) for a SCANNED-but-unloaded model — the by-id branch
/// that reads the catalog entry without a load probe.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn model_by_id_returns_scanned_metadata() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("SKIP model_by_id_returns_scanned_metadata: tiny gguf not found");
        return;
    };
    let srv = spawn_with_models(12403, &gguf, &[TINY_MODEL_ID]).await;
    let c = reqwest::Client::new();

    let by_id = c
        .get(format!("{}/api/higgs/models/{}", srv.base, TINY_MODEL_ID))
        .send()
        .await
        .unwrap();
    assert!(
        by_id.status().is_success(),
        "by-id for a scanned model is 200"
    );
    let detail: serde_json::Value = by_id.json().await.unwrap();

    // Identity + the on-disk facts the scanner records.
    assert_eq!(
        detail["id"], TINY_MODEL_ID,
        "by-id returns the entry: {detail}"
    );
    assert_eq!(detail["format"], "gguf", "format is gguf: {detail}");
    assert_eq!(
        detail["source"], "LmStudio",
        "staged under HIGGS_MODEL_DIR → LmStudio: {detail}"
    );
    assert_eq!(
        detail["state"], "not-loaded",
        "never loaded → not-loaded: {detail}"
    );
    assert!(
        detail["size_bytes"].as_u64().unwrap() > 0,
        "size read from disk: {detail}"
    );

    // GGUF-header-derived enrichment fields (stories260K is a real llama GGUF).
    assert_eq!(
        detail["arch"], "llama",
        "general.architecture extracted from the header: {detail}"
    );
    assert!(
        detail["ctx_train"].as_u64().is_some_and(|n| n > 0),
        "llama.context_length extracted from the header: {detail}"
    );
    // The harness names the staged file `stories260K.gguf` (no quant suffix that
    // looks like a quant tag), so `quant` is absent (the no-quant filename branch).
    assert!(
        detail["quant"].is_null(),
        "stories260K.gguf carries no quant tag → quant omitted: {detail}"
    );

    // A genuinely unknown id is a 404 (the not-found branch of the by-id route).
    let missing = c
        .get(format!("{}/api/higgs/models/no-org/no-model", srv.base))
        .send()
        .await
        .unwrap();
    assert_eq!(missing.status(), 404, "unknown id → 404");
}

// ---------------------------------------------------------------------------
// Local spawn helper: like `common::spawn_with_models`, but points the server at
// a scan root the TEST already built (so we can pollute it with junk before the
// scan). The shared harness only stages its OWN dir, so this is the one seam that
// lets us drive the walker's skip branches end-to-end. NOT a production knob — it
// reuses the same `HIGGS_*` env the operator/CI uses.
// ---------------------------------------------------------------------------

/// A running `higgs` child pointed at `root` as its LM-Studio scan dir. SIGTERMs on
/// drop (coverage-flushing graceful exit), holding an isolated `HIGGS_HOME` so the
/// machine's real `~/.higgs` (auth keys) never leaks in.
struct RootServer {
    child: Child,
    base: String,
    _home: TempDir,
}

impl Drop for RootServer {
    fn drop(&mut self) {
        // SIGTERM for a graceful, coverage-flushing shutdown, then reap.
        // SAFETY: a plain kill(2) on our own child's pid.
        unsafe {
            libc::kill(self.child.id() as libc::pid_t, libc::SIGTERM);
        }
        let _ = self.child.wait();
    }
}

/// Spawn `higgs` on `127.0.0.1:{port}` with `root` as the LM-Studio scan dir and
/// wait until `/api/higgs/status` answers. Mirrors `common::spawn_with_models`'s
/// env wiring (isolated `HIGGS_HOME`, quiet logs).
#[allow(clippy::zombie_processes)] // reaped by RootServer::drop
async fn spawn_at_root(port: u16, root: &Path) -> RootServer {
    let home = TempDir::new().expect("create temp HIGGS_HOME");
    let child = Command::new(env!("CARGO_BIN_EXE_higgs"))
        .env("HIGGS_BIND", "127.0.0.1")
        .env("HIGGS_PORT", port.to_string())
        .env("HIGGS_MODEL_DIR", root)
        .env("HIGGS_HOME", home.path())
        .env("RUST_LOG", "warn")
        .spawn()
        .expect("spawn higgs");
    let base = format!("http://127.0.0.1:{port}");
    let client = reqwest::Client::new();
    for _ in 0..150 {
        if let Ok(r) = client.get(format!("{base}/api/higgs/status")).send().await {
            if r.status().is_success() {
                return RootServer {
                    child,
                    base,
                    _home: home,
                };
            }
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    panic!("higgs never became ready on {base}");
}
