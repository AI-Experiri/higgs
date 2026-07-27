//! Black-box integration tests for the worker subsystem's still-uncovered paths:
//!   * `src/worker/models.rs` — the HF-cache scan walker, the `HG001
//!     ModelDirUnreadable` error arms across all three stores, the Ollama
//!     non-JSON-manifest / non-GGUF-blob skips, and the GGUF-enrichment
//!     tolerance edges (missing file, empty file, garbage bytes, truncated
//!     header) that must degrade gracefully instead of panicking the scan;
//!   * `src/worker/engine/mod.rs` — the engine registry selection
//!     (`build_engine` / `default_engine_name` / `engine_names`) and the
//!     `LoadParams` / `SamplingParams` / `GpuLayers` / `CtxLen` / `EngineDelta`
//!     umbrellas' public constructors + accessors;
//!   * `src/worker/engine/llamacpp/logging.rs` — the engine-diagnostic buffer's
//!     public drain/clear/verbose surface, plus a real corrupt-GGUF load that
//!     drives the worker's load-failure diagnostic path end-to-end;
//!   * `src/worker/mod.rs` — a real loaded tiny worker exercised over the full
//!     `M_LOAD` → `M_STATUS` → `M_SYSINFO` → `M_CHAT` → unload round-trip.
//!
//! Discovery/scan is HOST-SIDE, so the scan tests build their OWN `HiggsConfig`
//! (pointing `lmstudio_dirs` / `hf_dirs` / `ollama_dirs` at crafted, hermetic
//! store dirs) under an isolated `HIGGS_HOME` — the same construction
//! `common::higgs_local` uses, inlined here so we can aim the scan at custom
//! roots it doesn't expose. The real-worker tests wire the `worker_exe` DI seam
//! to the real `higgs` binary so `load`/`chat` run genuine llama.cpp.
//!
//! Every test that touches the PROCESS-GLOBAL `HIGGS_HOME` / `HIGGS_HF_ENDPOINT`
//! env holds a single binary-wide async lock for its whole lifetime, so parallel
//! test threads never clobber each other's isolated home.

mod common;

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use common::{serve_v1_local, stage_models, tiny_gguf_path, TINY_MODEL_ID};
use higgs::worker::engine::llamacpp::logging::{
    clear_engine_diagnostics, set_engine_verbose, take_engine_diagnostics,
};
use higgs::worker::engine::{
    build_engine, default_engine_name, engine_names, ChatDeltaKind, CtxLen, EngineDelta, GpuLayers,
    LoadParams, SamplingParams,
};
use higgs::worker::models::HiggsModelSource;
use higgs::{Higgs, HiggsConfig, HiggsError};
use serde_json::{json, Value};
use tempfile::TempDir;
use tokio::sync::{Mutex, OwnedMutexGuard};

// ───────────────────────────── in-file harness ──────────────────────────────

/// Binary-wide lock serializing every test that mutates the process-global
/// `HIGGS_HOME` / `HIGGS_HF_ENDPOINT` env. Held for a test's whole lifetime so
/// two tests never race on the isolated home (env is per-process, so this only
/// needs to guard within THIS test binary).
fn env_lock() -> Arc<Mutex<()>> {
    static LOCK: OnceLock<Arc<Mutex<()>>> = OnceLock::new();
    LOCK.get_or_init(|| Arc::new(Mutex::new(()))).clone()
}

/// An in-process `Higgs` under an isolated `HIGGS_HOME`, holding its staging
/// TempDirs + the env lock alive for its lifetime; drop restores the env.
struct Local {
    higgs: Arc<Higgs>,
    _home: TempDir,
    _keep: Vec<TempDir>,
    _lock: OwnedMutexGuard<()>,
    prev_home: Option<OsString>,
    prev_hf: Option<OsString>,
}

impl std::ops::Deref for Local {
    type Target = Arc<Higgs>;
    fn deref(&self) -> &Arc<Higgs> {
        &self.higgs
    }
}

impl Local {
    /// A clone of the facade handle (for `serve_v1_local`, which needs an owned Arc).
    fn handle(&self) -> Arc<Higgs> {
        self.higgs.clone()
    }

    /// Graceful teardown: stop the facade (draining any resident worker).
    async fn shutdown(self) {
        self.higgs.stop().await;
    }
}

impl Drop for Local {
    fn drop(&mut self) {
        // SAFETY: the still-held `_lock` guarantees no other harness thread reads
        // or writes the process env concurrently.
        unsafe {
            match &self.prev_home {
                Some(v) => std::env::set_var("HIGGS_HOME", v),
                None => std::env::remove_var("HIGGS_HOME"),
            }
            match &self.prev_hf {
                Some(v) => std::env::set_var("HIGGS_HF_ENDPOINT", v),
                None => std::env::remove_var("HIGGS_HF_ENDPOINT"),
            }
        }
    }
}

/// Start an in-process `Higgs` scanning exactly the given roots, under an
/// isolated `HIGGS_HOME` (so the machine's real `~/.higgs` config/models never
/// leak in) and a dead-port `HIGGS_HF_ENDPOINT`. `worker` wires the real `higgs`
/// binary as the worker-spawn seam. `keep` holds crafted store TempDirs alive.
async fn build(
    lmstudio: Vec<PathBuf>,
    hf: Vec<PathBuf>,
    ollama: Vec<PathBuf>,
    worker: bool,
    keep: Vec<TempDir>,
) -> Local {
    let lock = env_lock().lock_owned().await;
    let home = TempDir::new().expect("create temp HIGGS_HOME");
    let prev_home = std::env::var_os("HIGGS_HOME");
    let prev_hf = std::env::var_os("HIGGS_HF_ENDPOINT");
    // SAFETY: serialized by the held `lock`; restored on drop.
    unsafe {
        std::env::set_var("HIGGS_HOME", home.path());
        std::env::set_var("HIGGS_HF_ENDPOINT", "http://127.0.0.1:1");
    }
    let config = HiggsConfig {
        lmstudio_dirs: lmstudio,
        hf_dirs: hf,
        ollama_dirs: ollama,
        default_load: HiggsConfig::default().default_load,
        worker_exe: worker.then(|| env!("CARGO_BIN_EXE_higgs").into()),
    };
    let higgs = Arc::new(Higgs::new(config));
    higgs.start().await.expect("higgs start");
    Local {
        higgs,
        _home: home,
        _keep: keep,
        _lock: lock,
        prev_home,
        prev_hf,
    }
}

/// Scan-only `Higgs` (no worker) pointed at crafted store roots.
async fn scan_higgs(
    lmstudio: Vec<PathBuf>,
    hf: Vec<PathBuf>,
    ollama: Vec<PathBuf>,
    keep: Vec<TempDir>,
) -> Local {
    build(lmstudio, hf, ollama, false, keep).await
}

/// Worker-enabled `Higgs` with the real tiny GGUF staged under each id in
/// `models` (an LM-Studio root). Returns the instance plus the scan-root path
/// so a test can mutate a staged GGUF. `None` (test SKIPs) when no tiny GGUF.
async fn worker_higgs(models: &[&str]) -> Option<(Local, PathBuf)> {
    let gguf = tiny_gguf_path()?;
    let scan = stage_models(&gguf, models);
    let root = scan.path().to_path_buf();
    let local = build(vec![root.clone()], vec![], vec![], true, vec![scan]).await;
    Some((local, root))
}

/// The model ids present in a `model_entries()` result.
fn ids_of(entries: &[higgs::HiggsModelEntry]) -> Vec<String> {
    entries.iter().map(|e| e.model.id.clone()).collect()
}

// ── HF-cache store builder (`<root>/models--{org}--{name}/snapshots/{rev}/`) ──

/// Copy `gguf` to `<hf>/models--{org}--{name}/snapshots/{rev}/{file}`.
fn hf_put(hf: &Path, org: &str, name: &str, rev: &str, file: &str, gguf: &Path) {
    let dir = hf
        .join(format!("models--{org}--{name}"))
        .join("snapshots")
        .join(rev);
    std::fs::create_dir_all(&dir).expect("create hf snapshot dir");
    std::fs::copy(gguf, dir.join(file)).expect("copy hf gguf");
}

// ── Ollama store builders (`<root>/manifests/.../{name}/{tag}` + `blobs/…`) ──

fn ollama_manifest_dir(root: &Path, name: &str) -> PathBuf {
    root.join("manifests")
        .join("registry.ollama.ai")
        .join("library")
        .join(name)
}

fn ollama_write_manifest(root: &Path, name: &str, tag: &str, body: &str) {
    let dir = ollama_manifest_dir(root, name);
    std::fs::create_dir_all(&dir).expect("create manifest dir");
    std::fs::write(dir.join(tag), body).expect("write manifest");
}

fn ollama_write_blob(root: &Path, hex: &str, bytes: &[u8]) {
    let blobs = root.join("blobs");
    std::fs::create_dir_all(&blobs).expect("create blobs dir");
    std::fs::write(blobs.join(format!("sha256-{hex}")), bytes).expect("write blob");
}

fn ollama_model_manifest(digest: &str) -> String {
    format!(
        r#"{{"layers":[{{"mediaType":"application/vnd.ollama.image.model","digest":"{digest}"}}]}}"#
    )
}

// ───────────────────────────── models.rs: HF scan ───────────────────────────

/// The HF-cache walker (`scan_hf_cache`): a valid `models--{org}--{name}/
/// snapshots/{rev}/*.gguf` is discovered under `org/name` with `source: HfCache`
/// and enriched arch/quant, WHILE the walker's skip branches all hold — a repo
/// dir with no `--` separator, a repo with no `snapshots` subdir, a loose file
/// where a revision dir is expected, and an `mmproj-*` projector sidecar are all
/// omitted, and the surrounding non-gguf file yields no entry.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn hf_cache_valid_and_edges() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("SKIP hf_cache_valid_and_edges: tiny gguf not found (set HIGGS_TEST_GGUF)");
        return;
    };
    let hf = TempDir::new().unwrap();
    let root = hf.path();

    // A valid HF-cache model.
    hf_put(root, "hforg", "hfmodel", "rev1", "model-Q4_K_M.gguf", &gguf);
    // A projector sidecar alongside a real model in the SAME rev dir: the model
    // is discovered, the `mmproj-*` sidecar is excluded.
    hf_put(root, "porg", "pmodel", "rev1", "real-Q8_0.gguf", &gguf);
    hf_put(root, "porg", "pmodel", "rev1", "mmproj-vision.gguf", &gguf);
    // A non-gguf file in a rev dir → no entry (non-`.gguf` skip).
    let notes = root
        .join("models--rorg--rmodel")
        .join("snapshots")
        .join("rev1");
    std::fs::create_dir_all(&notes).unwrap();
    std::fs::write(notes.join("notes.txt"), b"not a gguf").unwrap();

    // Skip-branch fixtures:
    //  * a repo dir whose name has no `--` after `models--`  → skipped.
    std::fs::create_dir_all(root.join("models--onlyorg")).unwrap();
    //  * a repo dir with NO `snapshots` subdir              → skipped.
    std::fs::create_dir_all(root.join("models--eorg--emodel")).unwrap();
    //  * a snapshots dir holding a loose FILE (not a rev dir) → skipped.
    let snaps = root.join("models--forg--fmodel").join("snapshots");
    std::fs::create_dir_all(&snaps).unwrap();
    std::fs::write(snaps.join("loose-not-a-rev"), b"x").unwrap();
    //  * a dir lacking the `models--` prefix entirely        → skipped.
    std::fs::create_dir_all(root.join("some-other-cache-dir")).unwrap();

    let higgs = scan_higgs(vec![], vec![root.to_path_buf()], vec![], vec![hf]).await;
    let entries = higgs.model_entries().await.expect("hf scan succeeds");
    let ids = ids_of(&entries);

    // The valid model: HfCache source, enriched arch, quant from the filename.
    let valid = entries
        .iter()
        .find(|e| e.model.id == "hforg/hfmodel")
        .unwrap_or_else(|| panic!("hf scan lists `hforg/hfmodel`: {ids:?}"));
    assert_eq!(
        valid.model.source,
        HiggsModelSource::HfCache,
        "HfCache source"
    );
    assert_eq!(valid.format, "gguf", "discovered as gguf");
    assert_eq!(
        valid.model.arch.as_deref(),
        Some("llama"),
        "arch read off the HF-cache GGUF header"
    );
    assert_eq!(
        valid.model.quant.as_deref(),
        Some("Q4_K_M"),
        "quant parsed from the filename"
    );
    assert!(valid.model.size_bytes > 0, "size read from disk");

    // The projector sidecar is excluded; the real sibling is present exactly once.
    let pmodel_rows = ids.iter().filter(|id| *id == "porg/pmodel").count();
    assert_eq!(
        pmodel_rows, 1,
        "the mmproj sidecar is excluded; only the real model remains: {ids:?}"
    );
    let pmodel = entries
        .iter()
        .find(|e| e.model.id == "porg/pmodel")
        .expect("porg/pmodel present");
    assert_eq!(
        pmodel.model.quant.as_deref(),
        Some("Q8_0"),
        "real variant kept"
    );
    assert!(
        !pmodel.model.path.contains("mmproj"),
        "the surviving path is the real model, not the projector: {}",
        pmodel.model.path
    );

    // None of the skip-branch fixtures produced a catalog id.
    for junk in [
        "onlyorg",
        "onlyorg/",
        "eorg/emodel",
        "forg/fmodel",
        "rorg/rmodel",
        "some-other-cache-dir",
    ] {
        assert!(
            !ids.iter().any(|id| id == junk || id.starts_with(junk)),
            "skip-branch fixture `{junk}` yields no entry: {ids:?}"
        );
    }

    higgs.shutdown().await;
}

/// `HG001 ModelDirUnreadable` from the HF walker: an `hf_dirs` root that exists
/// but is a regular FILE (not a directory) makes `read_dir` fail with `ENOTDIR`
/// (not `NotFound`), which `scan_hf_cache` surfaces as the typed error rather
/// than swallowing.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn hf_cache_root_is_file_is_hg001() {
    let tmp = TempDir::new().unwrap();
    let file = tmp.path().join("i-am-a-file-not-a-dir");
    std::fs::write(&file, b"x").unwrap();

    let higgs = scan_higgs(vec![], vec![file], vec![], vec![tmp]).await;
    let err = higgs
        .model_entries()
        .await
        .expect_err("an hf root that is a file aborts the scan");
    assert!(
        matches!(err, HiggsError::ModelDirUnreadable { .. }),
        "unreadable hf root surfaces as HG001 ModelDirUnreadable: {err:?}"
    );
    higgs.shutdown().await;
}

// ─────────────────────── models.rs: LM-Studio + Ollama ──────────────────────

/// `HG001 ModelDirUnreadable` from the LM-Studio walker: an `lmstudio_dirs` root
/// that is a regular FILE surfaces the typed error (the `read_dir` ENOTDIR arm).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn lmstudio_root_is_file_is_hg001() {
    let tmp = TempDir::new().unwrap();
    let file = tmp.path().join("lmstudio-root-file");
    std::fs::write(&file, b"x").unwrap();

    let higgs = scan_higgs(vec![file], vec![], vec![], vec![tmp]).await;
    let err = higgs
        .model_entries()
        .await
        .expect_err("an lmstudio root that is a file aborts the scan");
    assert!(
        matches!(err, HiggsError::ModelDirUnreadable { .. }),
        "unreadable lmstudio root surfaces as HG001: {err:?}"
    );
    higgs.shutdown().await;
}

/// `HG001 ModelDirUnreadable` from the Ollama walker's manifest recursion: when
/// `<root>/manifests` exists but is a regular FILE, `collect_manifest_files`
/// hits the ENOTDIR read_dir arm and surfaces the typed error.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ollama_manifests_is_file_is_hg001() {
    let tmp = TempDir::new().unwrap();
    let ollama_root = tmp.path().join("models");
    std::fs::create_dir_all(&ollama_root).unwrap();
    // `<root>/manifests` is a FILE, so the recursive walk's read_dir fails ENOTDIR.
    std::fs::write(ollama_root.join("manifests"), b"not a dir").unwrap();

    let higgs = scan_higgs(vec![], vec![], vec![ollama_root], vec![tmp]).await;
    let err = higgs
        .model_entries()
        .await
        .expect_err("an ollama manifests path that is a file aborts the scan");
    assert!(
        matches!(err, HiggsError::ModelDirUnreadable { .. }),
        "unreadable ollama manifests dir surfaces as HG001: {err:?}"
    );
    higgs.shutdown().await;
}

/// Ollama per-file tolerance: a manifest that is valid UTF-8 but NOT JSON, and a
/// manifest whose model-layer blob EXISTS yet lacks the `GGUF` magic, are both
/// skipped silently while the one valid GGUF-backed manifest is still returned —
/// the scan does not abort.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ollama_non_json_and_bad_magic_blob_skipped() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("SKIP ollama_non_json_and_bad_magic_blob_skipped: tiny gguf not found");
        return;
    };
    let tmp = TempDir::new().unwrap();
    let root = tmp.path().join("models");
    std::fs::create_dir_all(&root).unwrap();

    // Valid model backed by the real tiny GGUF blob.
    let good_hex = "aaaa000000000000000000000000000000000000000000000000000000000000";
    ollama_write_blob(&root, good_hex, &std::fs::read(&gguf).unwrap());
    ollama_write_manifest(
        &root,
        "good",
        "latest",
        &ollama_model_manifest(&format!("sha256:{good_hex}")),
    );

    // Valid UTF-8 but NOT JSON → serde parse fails → skipped.
    ollama_write_manifest(&root, "notjson", "latest", "this is plainly not json {{{");

    // Model layer whose blob EXISTS but does not start with the GGUF magic → skipped.
    let bad_hex = "bbbb000000000000000000000000000000000000000000000000000000000000";
    ollama_write_blob(
        &root,
        bad_hex,
        b"NOPE not a gguf blob at all, no magic here",
    );
    ollama_write_manifest(
        &root,
        "badmagic",
        "latest",
        &ollama_model_manifest(&format!("sha256:{bad_hex}")),
    );

    let higgs = scan_higgs(vec![], vec![], vec![root], vec![tmp]).await;
    let entries = higgs
        .model_entries()
        .await
        .expect("garbage ollama manifests must not abort the scan");
    let ids = ids_of(&entries);

    assert!(
        ids.iter().any(|id| id == "ollama/good:latest"),
        "the valid GGUF-backed manifest is discovered: {ids:?}"
    );
    assert!(
        !ids.iter().any(|id| id == "ollama/notjson:latest"),
        "the non-JSON manifest is skipped: {ids:?}"
    );
    assert!(
        !ids.iter().any(|id| id == "ollama/badmagic:latest"),
        "the non-GGUF-magic blob is skipped: {ids:?}"
    );
    higgs.shutdown().await;
}

// ───────────────────────── models.rs: enrichment edges ──────────────────────

/// GGUF enrichment MUST degrade gracefully — never panic the scan — on a range
/// of broken files: a dangling symlink (open fails), an empty file (mmap of a
/// zero-length file fails), garbage bytes (`GGuf::new` returns Err), and a
/// truncated header (`GGuf::new` panics on an out-of-range slice, caught by the
/// enrichment's `catch_unwind`). All four stay cataloged with `arch == None`,
/// and a real sibling model is still enriched normally.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn enrich_tolerates_broken_gguf_files() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("SKIP enrich_tolerates_broken_gguf_files: tiny gguf not found");
        return;
    };
    let store = TempDir::new().unwrap();
    let root = store.path();

    // One model dir full of broken `.gguf` files (all share id `badorg/badmodel`).
    let bad = root.join("badorg").join("badmodel");
    std::fs::create_dir_all(&bad).unwrap();
    // (a) garbage bytes with no GGUF magic → GGuf::new returns Err.
    std::fs::write(bad.join("garbage.gguf"), b"this is not a gguf file at all").unwrap();
    // (b) real header truncated mid-first-KV → GGuf::new panics (caught).
    let real = std::fs::read(&gguf).unwrap();
    std::fs::write(bad.join("truncated.gguf"), &real[..40]).unwrap();
    // (c) empty file → mmap of zero length fails.
    std::fs::write(bad.join("empty.gguf"), b"").unwrap();
    // (d) dangling symlink → File::open fails.
    std::os::unix::fs::symlink(root.join("nonexistent-target"), bad.join("dead.gguf")).unwrap();

    // A genuinely valid sibling model.
    let good = root.join("goodorg").join("goodmodel");
    std::fs::create_dir_all(&good).unwrap();
    std::fs::copy(&gguf, good.join("stories260K.gguf")).unwrap();

    let higgs = scan_higgs(vec![root.to_path_buf()], vec![], vec![], vec![store]).await;
    // The whole scan SUCCEEDS despite the broken files (no panic escapes).
    let entries = higgs
        .model_entries()
        .await
        .expect("broken gguf files must not abort or panic the scan");

    // The valid sibling is enriched normally.
    let good_entry = entries
        .iter()
        .find(|e| e.model.id == "goodorg/goodmodel")
        .expect("the valid sibling is discovered");
    assert_eq!(
        good_entry.model.arch.as_deref(),
        Some("llama"),
        "the valid model is enriched normally"
    );

    // Every broken file is cataloged with EMPTY enrichment (arch stays None).
    let broken: Vec<_> = entries
        .iter()
        .filter(|e| e.model.id == "badorg/badmodel")
        .collect();
    assert!(
        broken.len() >= 4,
        "all four broken files are cataloged (partial enrichment), got {}: {:?}",
        broken.len(),
        broken
            .iter()
            .map(|e| e.model.path.clone())
            .collect::<Vec<_>>()
    );
    for e in &broken {
        assert!(
            e.model.arch.is_none(),
            "a broken gguf yields no arch (enrichment failed gracefully): {}",
            e.model.path
        );
        assert!(
            !e.model.has_chat_template,
            "a broken gguf has no chat template: {}",
            e.model.path
        );
    }

    higgs.shutdown().await;
}

// ─────────────────────────── engine/mod.rs: pub API ─────────────────────────

/// The engine registry + parameter umbrellas' public surface: `build_engine`
/// selects case-insensitively and falls back to the default for an
/// unknown/empty/absent name (the warn branch), and the `LoadParams` /
/// `SamplingParams` / `GpuLayers` / `CtxLen` / `EngineDelta` constructors +
/// accessors + lenient (legacy bare-int) deserialization return the documented
/// values.
#[test]
fn engine_registry_and_param_umbrellas() {
    // ── registry selection ──
    assert_eq!(default_engine_name(), "llamacpp", "default engine");
    assert!(
        engine_names().contains(&"llamacpp"),
        "llamacpp is registered: {:?}",
        engine_names()
    );
    assert_eq!(
        build_engine(Some("LlamaCpp")).1,
        "llamacpp",
        "selects case-insensitively"
    );
    assert_eq!(build_engine(None).1, "llamacpp", "absent name → default");
    assert_eq!(build_engine(Some("")).1, "llamacpp", "empty name → default");
    assert_eq!(
        build_engine(Some("no-such-engine")).1,
        "llamacpp",
        "unknown name → default (warn branch)"
    );

    // ── LoadParams umbrella ──
    assert!(
        matches!(LoadParams::default(), LoadParams::LlamaCpp(_)),
        "default LoadParams is the llamacpp variant"
    );
    let lp = LoadParams::base(CtxLen::fixed(2048), GpuLayers::all(), 6);
    assert_eq!(
        lp.ctx_len(),
        CtxLen::Fixed { n: 2048 },
        "base ctx_len accessor"
    );
    assert_eq!(lp.gpu_layers(), GpuLayers::All, "base gpu_layers accessor");
    assert_eq!(lp.threads(), 6, "base threads accessor");
    assert_eq!(
        LoadParams::llamacpp(lp.as_llamacpp().clone()).ctx_len(),
        CtxLen::Fixed { n: 2048 },
        "llamacpp() round-trips the payload"
    );

    // ── GpuLayers ──
    assert!(GpuLayers::all().is_all(), "all() is_all");
    assert!(!GpuLayers::all().is_cpu_only(), "all() is not cpu-only");
    assert!(GpuLayers::count(0).is_cpu_only(), "count(0) is cpu-only");
    assert!(!GpuLayers::count(8).is_all(), "count(8) is not all");
    assert_eq!(GpuLayers::count(8).to_n_gpu_layers(), 8, "count → raw n");
    assert_eq!(
        GpuLayers::all().to_n_gpu_layers(),
        u32::MAX,
        "all → the u32::MAX FFI sentinel"
    );
    // Lenient deserialize: a legacy bare int that is not u32::MAX → Count.
    assert_eq!(
        serde_json::from_value::<GpuLayers>(json!(8)).unwrap(),
        GpuLayers::Count { n: 8 },
        "bare int → Count"
    );
    assert_eq!(
        serde_json::from_value::<GpuLayers>(json!(u32::MAX)).unwrap(),
        GpuLayers::All,
        "bare u32::MAX → All"
    );

    // ── CtxLen ──
    assert_eq!(
        CtxLen::fixed(0),
        CtxLen::Auto,
        "fixed(0) normalizes to Auto"
    );
    assert!(CtxLen::Auto.is_auto(), "Auto is_auto");
    assert_eq!(CtxLen::Auto.fixed_n(), None, "Auto has no fixed n");
    assert_eq!(
        CtxLen::fixed(2048).fixed_n(),
        Some(2048),
        "Fixed exposes its n"
    );
    assert_eq!(CtxLen::Auto.to_n_ctx(), 0, "Auto → 0 FFI sentinel");
    assert_eq!(CtxLen::Fixed { n: 2048 }.to_n_ctx(), 2048, "Fixed → raw n");
    assert_eq!(
        serde_json::from_value::<CtxLen>(json!(4096)).unwrap(),
        CtxLen::Fixed { n: 4096 },
        "bare int → Fixed"
    );
    assert_eq!(
        serde_json::from_value::<CtxLen>(json!(0)).unwrap(),
        CtxLen::Auto,
        "bare 0 → Auto"
    );

    // ── SamplingParams umbrella ──
    assert!(
        matches!(SamplingParams::default(), SamplingParams::LlamaCpp(_)),
        "default SamplingParams is the llamacpp variant"
    );
    let sp = SamplingParams::default();
    let sp_rt = SamplingParams::llamacpp(sp.as_llamacpp().clone());
    assert!(
        matches!(sp_rt, SamplingParams::LlamaCpp(_)),
        "as_llamacpp + llamacpp() round-trip the payload"
    );

    // ── EngineDelta borrowed-delta kind/text (incl. the tool-call arm) ──
    assert_eq!(EngineDelta::Content("hi").kind(), ChatDeltaKind::Content);
    assert_eq!(EngineDelta::Content("hi").text(), "hi");
    assert_eq!(EngineDelta::Reasoning("t").kind(), ChatDeltaKind::Reasoning);
    assert_eq!(
        EngineDelta::ToolCall("{}").kind(),
        ChatDeltaKind::ToolCall,
        "tool-call delta reports the ToolCall kind"
    );
    assert_eq!(EngineDelta::ToolCall("{}").text(), "{}");
}

// ──────────────────────── logging.rs: diagnostics API ───────────────────────

/// The engine-diagnostic buffer's public surface: `clear` then `take` yields an
/// empty drain (nothing was captured in this host process — the buffer only
/// fills inside a worker's engine-error tap), and `set_engine_verbose` is a safe
/// no-op when the worker logging subscriber was never installed in this process.
#[test]
fn logging_diagnostics_pub_api() {
    clear_engine_diagnostics();
    assert!(
        take_engine_diagnostics().is_empty(),
        "no engine diagnostics are captured in the host process"
    );
    // No-op when logging was never installed here — must not panic, must not
    // spuriously populate the buffer.
    set_engine_verbose(true);
    set_engine_verbose(false);
    assert!(
        take_engine_diagnostics().is_empty(),
        "verbose toggling records no diagnostics"
    );
}

// ─────────────────────── worker/mod.rs: real round-trip ─────────────────────

/// POST a non-streaming chat and return the parsed JSON (dropping the response
/// so no stream is left open across teardown).
async fn chat_json(base: &str, body: Value) -> Value {
    reqwest::Client::new()
        .post(format!("{base}/v1/chat/completions"))
        .json(&body)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap()
}

/// A real loaded tiny worker exercised over the full RPC vocabulary: an explicit
/// `load` (M_LOAD), a `status` that reports the LIVE worker-probed params
/// (M_STATUS → id/worker_id/arch/ctx_len/threads), a `sysinfo` device
/// enumeration (M_SYSINFO), a short `/v1` chat (M_CHAT → well-formed completion
/// with non-zero prompt tokens), and an `unload` that clears the resident model.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn real_worker_load_status_sysinfo_chat_unload() {
    let Some((higgs, _root)) = worker_higgs(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP real_worker_load_status_sysinfo_chat_unload: tiny gguf not found");
        return;
    };

    // Explicit load (bypasses the JIT readiness gate) → M_LOAD to a real worker.
    higgs
        .load(TINY_MODEL_ID, None)
        .await
        .expect("explicit load");

    // M_STATUS round-trip: the worker reports the live model + probed params.
    let status = higgs.status().await.expect("status");
    assert!(status.worker_alive, "a worker is alive after load");
    let loaded = status.loaded.as_ref().expect("a model is resident");
    assert_eq!(loaded.id, TINY_MODEL_ID, "status reports the loaded id");
    assert_eq!(
        loaded.arch.as_deref(),
        Some("llama"),
        "status carries the host-enriched arch"
    );
    assert!(
        loaded.ctx_len.is_some(),
        "the worker probe returned a live context window"
    );
    assert!(
        loaded.threads.is_some(),
        "the worker probe returned the live thread count"
    );
    // `loaded_all` is the multi-model view; its primary entry matches `loaded`.
    let primary = status
        .loaded_all
        .iter()
        .find(|i| i.worker_id == loaded.worker_id)
        .expect("loaded_all carries the primary worker");
    assert_eq!(primary.id, TINY_MODEL_ID, "loaded_all keys by worker id");

    // M_SYSINFO round-trip: the engine enumerates at least one compute device.
    let devices = higgs.sysinfo().await;
    assert!(!devices.is_empty(), "sysinfo enumerates ≥1 device");
    assert!(
        !devices[0].name.is_empty(),
        "the enumerated device has a name: {:?}",
        devices[0]
    );

    // M_CHAT round-trip over the real `/v1` surface.
    let served = higgs.local_served_ids().await;
    let model_id = served
        .iter()
        .find(|s| s.starts_with("higgs-test"))
        .cloned()
        .unwrap_or_else(|| TINY_MODEL_ID.to_string());
    let (base, guard) = serve_v1_local(higgs.handle()).await;
    let resp = chat_json(
        &base,
        json!({
            "model": model_id, "stream": false, "max_tokens": 8,
            "messages": [{ "role": "user", "content": "Say hello." }]
        }),
    )
    .await;
    assert!(
        resp["choices"][0]["message"]["content"].is_string(),
        "chat returns string content: {resp:?}"
    );
    assert!(
        matches!(
            resp["choices"][0]["finish_reason"].as_str(),
            Some("stop") | Some("length") | Some("tool_calls")
        ),
        "chat returns a known finish_reason: {resp:?}"
    );
    assert!(
        resp["usage"]["prompt_tokens"].as_u64().unwrap_or(0) > 0,
        "the rendered prompt tokenized to a non-zero count: {resp:?}"
    );
    guard.shutdown().await;

    // M_UNLOAD path via the facade → the resident model clears.
    higgs.unload().await.expect("unload");
    let after = higgs.status().await.expect("status after unload");
    assert!(after.loaded.is_none(), "nothing resident after unload");

    higgs.shutdown().await;
}

/// A corrupt (non-GGUF) model file makes the real worker's engine `load` fail:
/// the facade surfaces a coded, non-empty error (the worker's load-failure
/// diagnostic path), NOT a silent success or a hang. A valid sibling still
/// loads afterward, proving the failure was isolated to the bad model.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn corrupt_gguf_load_surfaces_error() {
    let Some((higgs, root)) = worker_higgs(&["cov/good", "cov/bad"]).await else {
        eprintln!("SKIP corrupt_gguf_load_surfaces_error: tiny gguf not found");
        return;
    };
    // Overwrite the staged GGUF for `cov/bad` with garbage (no GGUF magic), so
    // the worker's llama.cpp load rejects it cleanly (a non-OOM hard failure,
    // returned immediately — no degrade-retry ladder).
    let bad_gguf = root.join("cov/bad").join("stories260K.gguf");
    std::fs::write(&bad_gguf, b"this file is not a valid gguf model").unwrap();

    let err = higgs
        .load("cov/bad", None)
        .await
        .expect_err("loading a corrupt gguf must fail");
    let msg = err.to_string();
    assert!(
        !msg.is_empty(),
        "the load failure surfaces a diagnostic message"
    );
    assert!(
        msg.contains("HG004") || msg.contains("HG009") || msg.to_lowercase().contains("load"),
        "the error identifies a load/worker failure: {msg}"
    );

    // The failure was isolated: the valid sibling still loads.
    higgs
        .load("cov/good", None)
        .await
        .expect("the valid sibling still loads after the corrupt one failed");
    let status = higgs.status().await.expect("status");
    assert_eq!(
        status.loaded.as_ref().map(|l| l.id.as_str()),
        Some("cov/good"),
        "the good model is resident after its load"
    );

    higgs.unload().await.expect("unload");
    higgs.shutdown().await;
}
