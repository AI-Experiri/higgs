//! Black-box integration tests for the host-side OLLAMA scan branches of
//! `src/worker/models.rs`, driven entirely over HTTP against a real spawned
//! `higgs`.
//!
//! The shared harness (`common::spawn_with_models`) only stages an LM-Studio
//! root via `HIGGS_MODEL_DIR`; the standalone binary exposes NO env override for
//! the Ollama manifests root — it derives `ollama_dirs` from
//! `HiggsConfig::default()`, which builds `~/.ollama/models` off `dirs::home_dir()`.
//! `dirs 5.0.1` resolves the home dir from `$HOME` first (verified in
//! `dirs-sys-0.4.1/src/lib.rs:34`), so pointing the spawned process's `HOME` at a
//! crafted temp dir redirects ALL of higgs's default scan roots
//! (`$HOME/.ollama/models`, `$HOME/.cache/huggingface/hub`,
//! `$HOME/.lmstudio/models`) into that hermetic dir — never touching the
//! developer's real `~/.ollama`. This is the legitimate operator-grade seam the
//! `scan_edges.rs` note said was missing for the Ollama surface; it is NOT a
//! test-only knob (`HOME` is just the shell's own env).
//!
//! These tests build a real Ollama store by hand
//! (`<home>/.ollama/models/manifests/.../{name}/{tag}` + `blobs/sha256-<hex>`)
//! and assert `GET /api/higgs/models` + `GET /api/higgs/models/{id}` reflect:
//!   * a VALID manifest whose model-layer blob is the real tiny GGUF — discovered
//!     under `ollama/{name}:{tag}` with `source: "Ollama"` and the GGUF-header
//!     metadata (arch / ctx_train) read off the blob;
//!   * MULTIPLE `.gguf` variants in one LM-Studio model dir collapsing to a single
//!     id with both paths cataloged, `get()` returning the lexically-first;
//!   * non-GGUF / garbage manifests (no `layers`, embedding-only layer, binary
//!     `.DS_Store`) silently skipped while the valid model is still returned;
//!   * the HG010 `OllamaManifestInvalid` abort (a model-layer digest with no
//!     `sha256:` prefix) surfacing as the mapped 500 control error.
//!
//! Because the crafted `HOME` is hermetic, these tests CAN assert exact list
//! membership / total counts for the Ollama ids (no stray real store leaks in).

mod common;

use std::path::Path;
use std::process::{Child, Command};
use std::time::Duration;

use common::tiny_gguf_path;
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// Hermetic-HOME spawn helper. Mirrors `common::spawn_with_models`'s env wiring
// but ALSO sets `HOME` so `HiggsConfig::default()`'s Ollama / HF / LM-Studio
// roots all resolve under a temp dir the test fully controls. No
// `HIGGS_MODEL_DIR` is set, so the LM-Studio roots are exactly `$HOME/.lmstudio`
// + `$HOME/.cache/lm-studio` (both empty unless the test populates them) and the
// Ollama root is the crafted `$HOME/.ollama/models`. NOT a production knob — it
// reuses the same `HOME`/`HIGGS_HOME` env an operator's shell sets.
// ---------------------------------------------------------------------------

/// A running `higgs` child whose entire model-scan surface lives under a crafted
/// `HOME`. SIGTERMs on drop (coverage-flushing graceful exit). Holds the `HOME`
/// and `HIGGS_HOME` temp dirs alive for the server's lifetime.
struct HomeServer {
    child: Child,
    base: String,
    _home: TempDir,
    _higgs_home: TempDir,
}

impl Drop for HomeServer {
    fn drop(&mut self) {
        // SIGTERM for a graceful, coverage-flushing shutdown, then reap.
        // SAFETY: a plain kill(2) on our own child's pid.
        unsafe {
            libc::kill(self.child.id() as libc::pid_t, libc::SIGTERM);
        }
        let _ = self.child.wait();
    }
}

/// Spawn `higgs` on `127.0.0.1:{port}` with `home_dir` as `$HOME` (so the default
/// Ollama / HF / LM-Studio scan roots resolve under it) and wait until
/// `/api/higgs/status` answers. Takes the `home_dir` `TempDir` BY VALUE (kept
/// alive in the returned guard) so the caller can build the store under it first,
/// then hand it over without a path-borrow conflict. An isolated `HIGGS_HOME`
/// keeps the machine's real `~/.higgs` (auth keys) out, mirroring the shared harness.
#[allow(clippy::zombie_processes)] // reaped by HomeServer::drop
async fn spawn_with_home(port: u16, home_dir: TempDir) -> HomeServer {
    let higgs_home = TempDir::new().expect("create temp HIGGS_HOME");
    let child = Command::new(env!("CARGO_BIN_EXE_higgs"))
        .env("HIGGS_BIND", "127.0.0.1")
        .env("HIGGS_PORT", port.to_string())
        .env("HOME", home_dir.path())
        .env("HIGGS_HOME", higgs_home.path())
        .env("RUST_LOG", "warn")
        .spawn()
        .expect("spawn higgs");
    let base = format!("http://127.0.0.1:{port}");
    let client = reqwest::Client::new();
    for _ in 0..150 {
        if let Ok(r) = client.get(format!("{base}/api/higgs/status")).send().await {
            if r.status().is_success() {
                return HomeServer {
                    child,
                    base,
                    _home: home_dir,
                    _higgs_home: higgs_home,
                };
            }
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    panic!("higgs never became ready on {base}");
}

// ---------------------------------------------------------------------------
// Ollama-store builders. The Ollama layout is
//   <home>/.ollama/models/manifests/<registry>/<library>/<name>/<tag>   (JSON)
//   <home>/.ollama/models/blobs/sha256-<hex>                            (blob)
// ---------------------------------------------------------------------------

/// Path to the Ollama models root inside a crafted HOME.
fn ollama_root(home: &Path) -> std::path::PathBuf {
    home.join(".ollama").join("models")
}

/// Write a blob at `<ollama>/blobs/sha256-<hex>` from `src` (copied, so the real
/// tiny GGUF magic + header survive verbatim).
fn write_blob_copy(home: &Path, hex: &str, src: &Path) {
    let blobs = ollama_root(home).join("blobs");
    std::fs::create_dir_all(&blobs).expect("create blobs dir");
    std::fs::copy(src, blobs.join(format!("sha256-{hex}"))).expect("copy blob");
}

/// Write a manifest file at `manifests/registry.ollama.ai/library/<name>/<tag>`.
fn write_manifest(home: &Path, name: &str, tag: &str, body: &str) {
    let dir = ollama_root(home)
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

/// Fetch `GET /api/higgs/models` as JSON.
async fn get_models(c: &reqwest::Client, base: &str) -> serde_json::Value {
    c.get(format!("{base}/api/higgs/models"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap()
}

/// The model ids present in a `GET /api/higgs/models` response.
fn ids_of(models: &serde_json::Value) -> Vec<String> {
    models["models"]
        .as_array()
        .expect("models is an array")
        .iter()
        .map(|m| m["id"].as_str().unwrap_or_default().to_string())
        .collect()
}

/// Find the scanned entry for `id`, or fail with the full body.
fn find_entry<'a>(models: &'a serde_json::Value, id: &str) -> &'a serde_json::Value {
    models["models"]
        .as_array()
        .expect("models is an array")
        .iter()
        .find(|m| m["id"] == serde_json::json!(id))
        .unwrap_or_else(|| panic!("scan lists `{id}`: {models}"))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// A VALID Ollama manifest whose model-layer digest resolves to the real tiny
/// GGUF blob is discovered under `ollama/{name}:{tag}` with `source: "Ollama"`,
/// and its GGUF-header metadata (arch / ctx_train) is read off the content-hashed
/// blob (the Ollama `enrich_gguf_metadata` path — blobs have no filename, so this
/// is the only way arch is learned). `GET /api/higgs/models/{id}` returns the same
/// enriched entry, and the path ends with `sha256-<hex>`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ollama_valid_manifest_resolves_and_enriches() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("SKIP ollama_valid_manifest_resolves_and_enriches: tiny gguf not found");
        return;
    };
    let home = TempDir::new().unwrap();
    // 64 hex chars is a realistic sha256, but any hex string works — the digest in
    // the manifest just has to match the blob filename.
    let hex = "1111111111111111111111111111111111111111111111111111111111111111";
    write_blob_copy(home.path(), hex, &gguf);
    write_manifest(
        home.path(),
        "tinystories",
        "latest",
        &model_manifest(&format!("sha256:{hex}")),
    );

    let srv = spawn_with_home(12700, home).await;
    let c = reqwest::Client::new();

    let models = get_models(&c, &srv.base).await;
    let id = "ollama/tinystories:latest";
    let entry = find_entry(&models, id);

    assert_eq!(
        entry["source"], "Ollama",
        "blob came from the Ollama store: {entry}"
    );
    assert_eq!(entry["format"], "gguf", "discovered model is gguf: {entry}");
    assert_eq!(entry["state"], "not-loaded", "nothing loaded: {entry}");
    assert!(
        entry["path"]
            .as_str()
            .unwrap()
            .ends_with(&format!("sha256-{hex}")),
        "path resolves to the content-hashed blob: {entry}"
    );
    assert!(
        entry["size_bytes"].as_u64().unwrap() > 0,
        "blob size read from disk: {entry}"
    );
    // stories260K is a real llama-arch GGUF, so enrichment reads the header off the
    // blob even though the blob has no filename to parse.
    assert_eq!(
        entry["arch"], "llama",
        "general.architecture read off the Ollama blob: {entry}"
    );
    assert!(
        entry["ctx_train"].as_u64().is_some_and(|n| n > 0),
        "llama.context_length read off the blob: {entry}"
    );

    // The by-id route returns the same enriched entry.
    let by_id = c
        .get(format!("{}/api/higgs/models/{id}", srv.base))
        .send()
        .await
        .unwrap();
    assert!(
        by_id.status().is_success(),
        "by-id for the ollama model is 200"
    );
    let detail: serde_json::Value = by_id.json().await.unwrap();
    assert_eq!(detail["id"], id, "by-id returns the ollama entry: {detail}");
    assert_eq!(
        detail["source"], "Ollama",
        "by-id source is Ollama: {detail}"
    );
    assert_eq!(
        detail["arch"], "llama",
        "by-id carries the enriched arch: {detail}"
    );
}

/// MULTIPLE `.gguf` files in ONE LM-Studio model dir (two quant variants) collapse
/// to a single id with BOTH path variants cataloged. The list contains exactly one
/// id (`org/multi`), `GET /api/higgs/models/{id}` resolves it (the by-id route picks
/// one), and the two distinct quant tags are both surfaced across the rows.
/// Exercised under the crafted `$HOME/.lmstudio/models` root (NOT `HIGGS_MODEL_DIR`).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn lmstudio_multiple_gguf_in_one_model_dir() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("SKIP lmstudio_multiple_gguf_in_one_model_dir: tiny gguf not found");
        return;
    };
    let home = TempDir::new().unwrap();
    // Default LM-Studio root #1 is `$HOME/.lmstudio/models`. Put `org/multi` there
    // with two quant variants of the SAME real GGUF.
    let model_dir = home.path().join(".lmstudio/models/org/multi");
    std::fs::create_dir_all(&model_dir).unwrap();
    std::fs::copy(&gguf, model_dir.join("multi-Q4_K_M.gguf")).unwrap();
    std::fs::copy(&gguf, model_dir.join("multi-Q8_0.gguf")).unwrap();

    let srv = spawn_with_home(12701, home).await;
    let c = reqwest::Client::new();

    let models = get_models(&c, &srv.base).await;
    let ids = ids_of(&models);
    // Exactly one id, but two rows (the two quant variants share the id).
    assert_eq!(
        ids.iter().filter(|id| *id == "org/multi").count(),
        2,
        "both quant variants are cataloged under the same id: {ids:?}"
    );

    // Both quant tags appear across the two rows.
    let quants: Vec<String> = models["models"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|m| m["id"] == serde_json::json!("org/multi"))
        .filter_map(|m| m["quant"].as_str().map(str::to_string))
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
    let by_id = c
        .get(format!("{}/api/higgs/models/org/multi", srv.base))
        .send()
        .await
        .unwrap();
    assert!(
        by_id.status().is_success(),
        "by-id resolves a multi-variant id"
    );
    let detail: serde_json::Value = by_id.json().await.unwrap();
    assert_eq!(
        detail["id"], "org/multi",
        "by-id returns the shared id: {detail}"
    );
    assert_eq!(
        detail["source"], "LmStudio",
        "found under $HOME/.lmstudio: {detail}"
    );
    assert!(
        detail["path"].as_str().unwrap().contains("Q4_K_M"),
        "by-id returns the lexically-first (Q4_K_M) variant: {detail}"
    );
}

/// Non-GGUF / garbage manifests must NOT abort the Ollama scan: a manifest with no
/// `layers`, an embedding-only manifest (a layer that is not the model mediaType),
/// a binary `.DS_Store`, and a manifest whose blob is absent are all skipped
/// silently while the ONE valid model is still returned. Drives the silent-skip
/// branches of `scan_ollama` end-to-end (the scan must return 200, not error).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ollama_garbage_manifests_skipped_valid_returned() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("SKIP ollama_garbage_manifests_skipped_valid_returned: tiny gguf not found");
        return;
    };
    let home = TempDir::new().unwrap();

    // One valid model.
    let good_hex = "2222222222222222222222222222222222222222222222222222222222222222";
    write_blob_copy(home.path(), good_hex, &gguf);
    write_manifest(
        home.path(),
        "good",
        "latest",
        &model_manifest(&format!("sha256:{good_hex}")),
    );

    // Vision-style manifest: no `layers` key at all.
    write_manifest(
        home.path(),
        "novision",
        "latest",
        r#"{"config":{"mediaType":"application/vnd.ollama.image"}}"#,
    );

    // Embedding-style manifest: has `layers`, but none with the model mediaType.
    write_manifest(
        home.path(),
        "embed",
        "latest",
        r#"{"layers":[{"mediaType":"application/vnd.ollama.image.params"}]}"#,
    );

    // Manifest whose blob is absent (partial pull) — model layer points nowhere.
    write_manifest(
        home.path(),
        "partialpull",
        "latest",
        &model_manifest("sha256:deadbeefabsent"),
    );

    // A binary file in the manifests tree (.DS_Store) that is not UTF-8 / JSON.
    let ds = ollama_root(home.path())
        .join("manifests")
        .join("registry.ollama.ai")
        .join("library");
    std::fs::create_dir_all(&ds).unwrap();
    std::fs::write(ds.join(".DS_Store"), [0x00u8, 0xFF, 0xFE, 0xFD, 0x42, 0x50]).unwrap();

    let srv = spawn_with_home(12702, home).await;
    let c = reqwest::Client::new();

    // The scan MUST succeed (the garbage does not abort it).
    let resp = c
        .get(format!("{}/api/higgs/models", srv.base))
        .send()
        .await
        .unwrap();
    assert!(
        resp.status().is_success(),
        "garbage manifests must not error the scan: {}",
        resp.status()
    );
    let models: serde_json::Value = resp.json().await.unwrap();
    let ids = ids_of(&models);

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
}

/// HG010: an Ollama model layer whose `digest` lacks the `sha256:` prefix is an
/// invalid manifest — `scan_ollama` aborts with `OllamaManifestInvalid`, which the
/// control surface maps to a 500 carrying the `[HG010]` diagnostic. Reaches the
/// HG010 path through the live HTTP scan (the branch `scan_edges.rs` documents as
/// otherwise unreachable from this surface).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ollama_invalid_digest_surfaces_hg010() {
    // No GGUF needed — the digest-format check fires before any blob lookup.
    let home = TempDir::new().unwrap();
    write_manifest(
        home.path(),
        "badigest",
        "latest",
        // `md5:` is not the expected `sha256:` prefix → HG010.
        &model_manifest("md5:zzz"),
    );

    let srv = spawn_with_home(12703, home).await;
    let c = reqwest::Client::new();

    let resp = c
        .get(format!("{}/api/higgs/models", srv.base))
        .send()
        .await
        .unwrap();
    // OllamaManifestInvalid → the `_ => INTERNAL_SERVER_ERROR` http_status arm.
    assert_eq!(
        resp.status(),
        500,
        "an invalid ollama manifest aborts the scan as a 500 control error"
    );
    let body: serde_json::Value = resp.json().await.unwrap();
    let msg = body["error"].as_str().unwrap_or_default();
    assert!(
        msg.contains("HG010"),
        "the 500 body carries the [HG010] OllamaManifestInvalid diagnostic: {body}"
    );
}

/// HG010 variant: a model layer that is entirely MISSING the `digest` field also
/// aborts the scan as `OllamaManifestInvalid` (the `digest.as_str()` None arm),
/// surfacing as the same 500 `[HG010]` control error. A distinct manifest shape
/// from the bad-prefix case above.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ollama_missing_digest_surfaces_hg010() {
    let home = TempDir::new().unwrap();
    write_manifest(
        home.path(),
        "nodigest",
        "latest",
        // model-mediaType layer with NO digest field at all.
        r#"{"layers":[{"mediaType":"application/vnd.ollama.image.model"}]}"#,
    );

    let srv = spawn_with_home(12704, home).await;
    let c = reqwest::Client::new();

    let resp = c
        .get(format!("{}/api/higgs/models", srv.base))
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        500,
        "a model layer with no digest aborts the scan as a 500"
    );
    let body: serde_json::Value = resp.json().await.unwrap();
    assert!(
        body["error"].as_str().unwrap_or_default().contains("HG010"),
        "the 500 body carries the [HG010] diagnostic: {body}"
    );
}
