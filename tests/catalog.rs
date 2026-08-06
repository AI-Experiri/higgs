//! Black-box model-search catalog: the facade ops (`model_search` /
//! `model_detail` / `model_download` + download events) and the `higgs model`
//! CLI, all against a LOCAL fixture Hub served over `HIGGS_HF_ENDPOINT` — no
//! network, no real Hugging Face.

use std::future::IntoFuture;
use std::sync::{Arc, Mutex};

use axum::extract::RawQuery;
use axum::routing::{get, post};
use serde_json::json;

use higgs::api::{Higgs, HiggsConfig};
use higgs::catalog::{CatalogQuery, ModelDownloadPhase};

/// The canned repo the fixture Hub serves.
const REPO: &str = "acme/tiny";

/// Serializes the IN-PROCESS facade tests: each mutates the process-global
/// `HIGGS_HF_ENDPOINT`/`HIGGS_HOME`, so they must not overlap. (The CLI tests
/// pass env to their CHILD explicitly and need no lock.) A tokio mutex so the
/// guard is await-safe.
static ENV_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// Spawn a loopback "Hugging Face": model list, model info, paths-info, and
/// resolve downloads. Returns the endpoint URL and the log of list queries.
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
                    // An author browse (the detail's more-by-author call)
                    // gets the same two repos — the self-exclusion is the
                    // caller's job.
                    axum::Json(json!([
                        {
                            "id": REPO,
                            "author": "acme",
                            "downloads": 42u64,
                            "likes": 7u64,
                            "lastModified": "2026-07-30T00:00:00.000Z",
                            "pipeline_tag": "text-generation",
                        },
                        { "id": "acme/big" },
                    ]))
                }
            }),
        )
        .route(
            "/api/models/{org}/{name}",
            get(|| async {
                axum::Json(json!({
                    "id": REPO,
                    "author": "acme",
                    "downloads": 42u64,
                    "likes": 7u64,
                    "lastModified": "2026-07-30T00:00:00.000Z",
                    "pipeline_tag": "text-generation",
                    "tags": ["gguf", "text-generation"],
                    "gguf": {
                        "architecture": "llama",
                        "total": 1_000_000u64,
                        "context_length": 4096u64,
                    },
                    "siblings": [
                        { "rfilename": "README.md" },
                        { "rfilename": "tiny-Q4_K_M.gguf" },
                        { "rfilename": "tiny-F16.gguf" },
                    ],
                }))
            }),
        )
        .route(
            "/api/models/{org}/{name}/paths-info/{rev}",
            post(|| async {
                axum::Json(json!([
                    { "type": "file", "oid": "a", "size": 134u64,
                      "path": "tiny-Q4_K_M.gguf", "lfs": { "size": 4_000u64 } },
                    { "type": "file", "oid": "b", "size": 9_000u64, "path": "tiny-F16.gguf" },
                ]))
            }),
        )
        .route(
            "/{org}/{name}/resolve/{rev}/{file}",
            get(
                |axum::extract::Path((_, _, _, file)): axum::extract::Path<(
                    String,
                    String,
                    String,
                    String,
                )>| async move {
                    if file == "README.md" {
                        "# tiny\nhello from the fixture".into_response()
                    } else {
                        b"GGUF-fixture-bytes".to_vec().into_response()
                    }
                },
            ),
        );
    tokio::spawn(axum::serve(listener, app).into_future());
    (format!("http://{addr}"), queries)
}

use axum::response::IntoResponse;

/// End-to-end over the embed facade: search rows (with the Hub query shaped
/// GGUF-only), detail assembly, a real download landing in `$HIGGS_HOME/models`
/// with Starting→Done events, and the downloaded flags flipping on re-query.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn facade_search_detail_download_end_to_end() {
    let _env = ENV_LOCK.lock().await;
    let (endpoint, queries) = fixture_hub().await;
    let home = tempfile::tempdir().expect("home");
    unsafe {
        std::env::set_var("HIGGS_HF_ENDPOINT", &endpoint);
        std::env::set_var("HIGGS_HOME", home.path());
    }

    let higgs = Higgs::new(HiggsConfig {
        lmstudio_dirs: vec![home.path().join("models")],
        hf_dirs: vec![],
        ollama_dirs: vec![],
        default_load: HiggsConfig::default().default_load,
        worker_exe: None,
    });

    // ── search ─────────────────────────────────────────────────────────────
    let q = CatalogQuery {
        search: "tiny".into(),
        author: None,
        sort: None,
        limit: None,
        compatible_only: None,
    };
    let resp = higgs.model_search(&q).await.expect("search");
    assert_eq!(resp.models.len(), 2);
    let row = &resp.models[0];
    assert_eq!(row.id, REPO);
    assert_eq!(row.author.as_deref(), Some("acme"));
    assert_eq!(row.downloads, Some(42));
    assert_eq!(row.likes, Some(7));
    assert!(!row.downloaded);
    {
        let log = queries.lock().unwrap();
        let listed = log.first().expect("one list call");
        assert!(
            listed.contains("search=tiny"),
            "free text forwarded: {listed}"
        );
        assert!(listed.contains("filter=gguf"), "GGUF-only filter: {listed}");
        assert!(listed.contains("sort=downloads"), "default sort: {listed}");
        assert!(listed.contains("limit=25"), "default page size: {listed}");
    }

    // ── detail ─────────────────────────────────────────────────────────────
    let d = higgs.model_detail(REPO).await.expect("detail");
    assert_eq!(d.summary.id, REPO);
    let g = d.summary.gguf.as_ref().expect("gguf badge");
    assert_eq!(g.arch.as_deref(), Some("llama"));
    assert_eq!(g.ctx_train, Some(4096));
    assert_eq!(d.tags, vec!["gguf", "text-generation"]);
    assert_eq!(
        d.readme.as_deref(),
        Some("# tiny\nhello from the fixture"),
        "README fetched through the resolve URL"
    );
    let files: Vec<&str> = d.quants.iter().map(|q| q.file.as_str()).collect();
    assert_eq!(
        files,
        ["tiny-Q4_K_M.gguf", "tiny-F16.gguf"],
        "size-ascending"
    );
    assert_eq!(
        d.quants[0].size_bytes,
        Some(4_000),
        "LFS size via paths-info"
    );
    assert_eq!(d.quants[1].size_bytes, Some(9_000));
    assert_eq!(d.quants[0].quant.as_deref(), Some("Q4_K_M"));
    assert!(!d.quants[0].downloaded);
    let more: Vec<&str> = d.more_by_author.iter().map(|m| m.id.as_str()).collect();
    assert_eq!(more, ["acme/big"], "author browse excludes the repo itself");

    // ── download + events ──────────────────────────────────────────────────
    let mut rx = higgs.subscribe_download_events();
    let path = higgs
        .model_download(REPO, "tiny-Q4_K_M.gguf")
        .await
        .expect("download");
    assert_eq!(path, home.path().join("models/acme/tiny/tiny-Q4_K_M.gguf"));
    assert_eq!(std::fs::read(&path).unwrap(), b"GGUF-fixture-bytes");
    let mut phases = Vec::new();
    while let Ok(ev) = rx.try_recv() {
        assert_eq!(ev.repo, REPO);
        phases.push(ev.phase);
    }
    assert_eq!(phases.first(), Some(&ModelDownloadPhase::Starting));
    assert_eq!(phases.last(), Some(&ModelDownloadPhase::Done));

    // ── downloaded flags close the loop ────────────────────────────────────
    let d2 = higgs
        .model_detail(REPO)
        .await
        .expect("detail after download");
    assert!(d2.summary.downloaded, "repo now marked downloaded");
    assert!(d2.quants[0].downloaded, "the exact quant is marked");
    assert!(!d2.quants[1].downloaded, "the sibling quant is not");
    let resp2 = higgs.model_search(&q).await.expect("search after download");
    assert!(resp2.models[0].downloaded);
}

/// A download whose file the fixture cannot serve as a valid pull (the repo
/// refuses it before any fetch) emits `Starting` then `Failed` with the
/// diagnostic code — the facade's error event path, end to end.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn facade_download_failure_emits_failed_with_a_code() {
    let _env = ENV_LOCK.lock().await;
    let (endpoint, _) = fixture_hub().await;
    let home = tempfile::tempdir().expect("home");
    unsafe {
        std::env::set_var("HIGGS_HF_ENDPOINT", &endpoint);
        std::env::set_var("HIGGS_HOME", home.path());
    }
    let higgs = Higgs::new(HiggsConfig {
        lmstudio_dirs: vec![home.path().join("models")],
        hf_dirs: vec![],
        ollama_dirs: vec![],
        default_load: HiggsConfig::default().default_load,
        worker_exe: None,
    });
    let mut rx = higgs.subscribe_download_events();
    // Not a `.gguf` → refused by wire validation before any fetch.
    let err = higgs
        .model_download(REPO, "not-a-model.txt")
        .await
        .expect_err("refused");
    assert!(matches!(err, higgs::HiggsError::DownloadFailed { .. }));
    let mut phases = Vec::new();
    while let Ok(ev) = rx.try_recv() {
        if ev.phase == ModelDownloadPhase::Failed {
            assert_eq!(ev.code.as_deref(), Some("HG025"));
        }
        phases.push(ev.phase);
    }
    assert_eq!(phases.first(), Some(&ModelDownloadPhase::Starting));
    assert_eq!(phases.last(), Some(&ModelDownloadPhase::Failed));
}

/// The CLI end-to-end in a child process (its own env — no interplay with the
/// in-process test): search renders rows, download with no file picks the
/// default quant and lands it, and a bad subcommand exits non-zero with usage.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cli_search_and_default_pick_download() {
    let (endpoint, _) = fixture_hub().await;
    let home = tempfile::tempdir().expect("home");
    let exe = env!("CARGO_BIN_EXE_higgs");

    let search = tokio::process::Command::new(exe)
        .args(["model", "search", "tiny"])
        .env("HIGGS_HF_ENDPOINT", &endpoint)
        .env("HIGGS_HOME", home.path())
        .output()
        .await
        .expect("run search");
    assert!(
        search.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&search.stderr)
    );
    let out = String::from_utf8_lossy(&search.stdout);
    assert!(out.contains("MODEL"), "header: {out}");
    assert!(out.contains(REPO));
    assert!(out.contains("acme/big"));

    let download = tokio::process::Command::new(exe)
        .args(["model", "download", REPO])
        .env("HIGGS_HF_ENDPOINT", &endpoint)
        .env("HIGGS_HOME", home.path())
        .output()
        .await
        .expect("run download");
    assert!(
        download.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&download.stderr)
    );
    // No file named → the default pick is announced and Q4_K_M lands on disk.
    let err = String::from_utf8_lossy(&download.stderr);
    assert!(err.contains("picking tiny-Q4_K_M.gguf"), "stderr: {err}");
    let landed = home.path().join("models/acme/tiny/tiny-Q4_K_M.gguf");
    assert!(landed.is_file());
    let stdout = String::from_utf8_lossy(&download.stdout);
    assert!(
        stdout.trim_end().ends_with("tiny-Q4_K_M.gguf"),
        "prints the path: {stdout}"
    );

    let bad = tokio::process::Command::new(exe)
        .args(["model", "frobnicate"])
        .env("HIGGS_HOME", home.path())
        .output()
        .await
        .expect("run bad");
    assert!(!bad.status.success());
    assert!(String::from_utf8_lossy(&bad.stderr).contains("usage:"));
}

/// `higgs model show` renders the detail pane (meta line, quant table with
/// sizes, more-by-author, README) and search honors the `--limit`/`--sort`
/// flags — all in a child process against the fixture Hub.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cli_show_renders_the_detail_and_flags_parse() {
    let (endpoint, queries) = fixture_hub().await;
    let home = tempfile::tempdir().expect("home");
    let exe = env!("CARGO_BIN_EXE_higgs");

    let show = tokio::process::Command::new(exe)
        .args(["model", "show", REPO])
        .env("HIGGS_HF_ENDPOINT", &endpoint)
        .env("HIGGS_HOME", home.path())
        .output()
        .await
        .expect("run show");
    assert!(
        show.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&show.stderr)
    );
    let out = String::from_utf8_lossy(&show.stdout);
    assert!(out.contains(REPO));
    assert!(out.contains("arch llama"), "gguf meta line: {out}");
    assert!(out.contains("tiny-Q4_K_M.gguf"));
    assert!(out.contains("3.9 KiB"), "paths-info size rendered: {out}");
    assert!(out.contains("acme/big"), "more-by-author: {out}");
    assert!(out.contains("hello from the fixture"), "README: {out}");

    let flagged = tokio::process::Command::new(exe)
        .args(["model", "search", "tiny", "--limit", "5", "--sort", "likes"])
        .env("HIGGS_HF_ENDPOINT", &endpoint)
        .env("HIGGS_HOME", home.path())
        .output()
        .await
        .expect("run flagged search");
    assert!(flagged.status.success());
    let sent = queries.lock().unwrap().last().cloned().expect("query");
    assert!(sent.contains("limit=5"), "{sent}");
    assert!(sent.contains("sort=likes"), "{sent}");

    // Named-file download (no default pick involved).
    let named = tokio::process::Command::new(exe)
        .args(["model", "download", REPO, "tiny-F16.gguf"])
        .env("HIGGS_HF_ENDPOINT", &endpoint)
        .env("HIGGS_HOME", home.path())
        .output()
        .await
        .expect("run named download");
    assert!(named.status.success());
    assert!(home.path().join("models/acme/tiny/tiny-F16.gguf").is_file());

    // Parse-error arms: each exits non-zero with the usage line.
    for bad in [
        vec!["model", "search"],
        vec!["model", "search", "q", "--limit", "x"],
        vec!["model", "search", "q", "--sort", "weird"],
        vec!["model", "show"],
        vec!["model", "download"],
    ] {
        let out = tokio::process::Command::new(exe)
            .args(&bad)
            .env("HIGGS_HOME", home.path())
            .output()
            .await
            .expect("run parse-error case");
        assert!(!out.status.success(), "{bad:?} must fail");
        assert!(
            String::from_utf8_lossy(&out.stderr).contains("usage:"),
            "{bad:?} prints usage"
        );
    }
}
