use super::*;

use crate::catalog::wire::{
    CatalogModelDetail, CatalogModelSummary, CatalogQuant, CatalogSearchResponse, CatalogSort,
};
use crate::system::FitAssessment;

fn args(list: &[&str]) -> Vec<String> {
    list.iter().map(|s| (*s).to_string()).collect()
}

// ── parse_model_cmd ────────────────────────────────────────────────────────

#[test]
fn parse_search_joins_query_words_and_reads_flags() {
    let cmd = parse_model_cmd(&args(&[
        "search", "tiny", "llama", "--limit", "5", "--sort", "likes",
    ]))
    .expect("parse");
    match cmd {
        ModelCmd::Search { query, limit, sort } => {
            assert_eq!(query, "tiny llama");
            assert_eq!(limit, 5);
            assert_eq!(sort, CatalogSort::Likes);
        }
        other => panic!("expected Search, got {other:?}"),
    }
}

#[test]
fn parse_search_defaults_limit_and_sort() {
    match parse_model_cmd(&args(&["search", "qwen"])).expect("parse") {
        ModelCmd::Search { query, limit, sort } => {
            assert_eq!(query, "qwen");
            assert_eq!(limit, 0, "0 = service default page size");
            assert_eq!(sort, CatalogSort::Downloads);
        }
        other => panic!("expected Search, got {other:?}"),
    }
}

#[test]
fn parse_show_and_download_take_a_repo_and_optional_file() {
    match parse_model_cmd(&args(&["show", "acme/m"])).expect("parse") {
        ModelCmd::Show { repo } => assert_eq!(repo, "acme/m"),
        other => panic!("expected Show, got {other:?}"),
    }
    match parse_model_cmd(&args(&["download", "acme/m"])).expect("parse") {
        ModelCmd::Download { repo, file } => {
            assert_eq!(repo, "acme/m");
            assert!(file.is_none());
        }
        other => panic!("expected Download, got {other:?}"),
    }
    match parse_model_cmd(&args(&["download", "acme/m", "m-Q4_K_M.gguf"])).expect("parse") {
        ModelCmd::Download { repo, file } => {
            assert_eq!(repo, "acme/m");
            assert_eq!(file.as_deref(), Some("m-Q4_K_M.gguf"));
        }
        other => panic!("expected Download, got {other:?}"),
    }
}

#[test]
fn parse_rejects_bad_input_with_a_usage_message() {
    for bad in [
        &args(&[]) as &[String],
        &args(&["search"]),                         // no query
        &args(&["search", "q", "--limit", "x"]),    // non-numeric limit
        &args(&["search", "q", "--sort", "weird"]), // unknown sort
        &args(&["show"]),                           // no repo
        &args(&["download"]),                       // no repo
        &args(&["frobnicate"]),                     // unknown subcommand
    ] {
        let err = parse_model_cmd(bad).expect_err("must reject");
        assert!(err.contains("usage:"), "usage in {err:?}");
    }
}

// ── the driver over the loopback fixture Hub ───────────────────────────────

/// `run_cmd` all three arms + `run_model` end-to-end against the fixture Hub
/// — one sync test (own runtime) so `TEST_ENV_LOCK` never crosses an await in
/// an async fn.
#[test]
fn driver_search_show_and_download_over_a_loopback_hub() {
    let _env = crate::TEST_ENV_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let home = tempfile::tempdir().expect("home");
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let endpoint = rt.block_on(async {
        let hub = crate::catalog::test_support::fixture_hub().await;
        hub.endpoint
    });
    let _redirect = crate::catalog::test_support::EnvRedirect::set(&endpoint, Some(home.path()));

    rt.block_on(async {
        run_cmd(ModelCmd::Search {
            query: "tiny".into(),
            limit: 0,
            sort: CatalogSort::Downloads,
        })
        .await
        .expect("search");
        run_cmd(ModelCmd::Show {
            repo: "acme/tiny".into(),
        })
        .await
        .expect("show");
        // Named file.
        run_cmd(ModelCmd::Download {
            repo: "acme/tiny".into(),
            file: Some("tiny-F16.gguf".into()),
        })
        .await
        .expect("named download");
        assert!(home.path().join("models/acme/tiny/tiny-F16.gguf").is_file());
        // No file → the default pick (Q4_K_M) is chosen and lands.
        run_cmd(ModelCmd::Download {
            repo: "acme/tiny".into(),
            file: None,
        })
        .await
        .expect("default-pick download");
        assert!(home
            .path()
            .join("models/acme/tiny/tiny-Q4_K_M.gguf")
            .is_file());
    });

    // `run_model` builds its OWN runtime — call it outside any async context
    // (`rt` stays alive so the fixture Hub keeps serving).
    run_model(&args(&["search", "tiny"])).expect("run_model search");
    let err = run_model(&args(&["frobnicate"])).expect_err("bad subcommand");
    assert!(err.to_string().contains("unknown model subcommand"));
}

// ── renderers ──────────────────────────────────────────────────────────────

#[test]
fn fmt_bytes_picks_a_human_unit() {
    assert_eq!(fmt_bytes(512), "512 B");
    assert_eq!(fmt_bytes(4_096), "4.0 KiB");
    assert_eq!(fmt_bytes(5 * 1024 * 1024), "5.0 MiB");
    assert_eq!(fmt_bytes(4_600_000_000), "4.3 GiB");
}

fn summary(id: &str, downloaded: bool) -> CatalogModelSummary {
    CatalogModelSummary {
        id: id.to_string(),
        author: None,
        downloads: Some(1_234),
        likes: Some(56),
        updated: Some("2026-07-01T10:00:00.000Z".into()),
        pipeline: None,
        gguf: None,
        fit: None,
        downloaded,
    }
}

#[test]
fn render_search_lists_rows_and_marks_downloaded() {
    let out = render_search(&CatalogSearchResponse {
        models: vec![summary("acme/a", false), summary("acme/b", true)],
    });
    assert!(out.contains("acme/a"));
    assert!(out.contains("acme/b"));
    // The downloaded row is marked; the other is not.
    let b_line = out.lines().find(|l| l.contains("acme/b")).unwrap();
    let a_line = out.lines().find(|l| l.contains("acme/a")).unwrap();
    assert!(b_line.contains("downloaded"));
    assert!(!a_line.contains("downloaded"));
    // The date column shows the day, not the full timestamp.
    assert!(out.contains("2026-07-01"));
    assert!(!out.contains("10:00:00"));
}

#[test]
fn render_search_says_so_when_nothing_matched() {
    let out = render_search(&CatalogSearchResponse { models: vec![] });
    assert!(out.contains("no models"));
}

#[test]
fn render_collapses_layout_chars_in_hub_derived_table_fields() {
    // A hostile Hub row must not fabricate table lines/columns: layout chars
    // inside structured fields collapse to spaces (the README keeps layout —
    // only cells are collapsed).
    let hostile = summary("acme/a\nacme/fake-row    999", false);
    let out = render_search(&CatalogSearchResponse {
        models: vec![hostile.clone()],
    });
    assert!(!out.contains("\nacme/fake-row"), "no injected row: {out:?}");
    assert!(
        out.contains("acme/a acme/fake-row"),
        "collapsed inline: {out:?}"
    );

    let d = CatalogModelDetail {
        summary: hostile,
        tags: vec!["gg\nuf".into()],
        readme: None,
        default_file: None,
        quants: vec![CatalogQuant {
            file: "m\n.gguf".into(),
            quant: Some("Q4\t_K".into()),
            size_bytes: None,
            downloaded: false,
            fit: None,
        }],
        more_by_author: vec![],
    };
    let out = render_detail(&d);
    assert!(!out.contains("gg\nuf"));
    assert!(!out.contains("m\n.gguf"));
    assert!(!out.contains("Q4\t_K"));
}

#[test]
fn cli_errors_are_sanitized_and_single_line_before_printing() {
    // A Hub/mirror-derived error detail must not carry ANSI escapes or
    // fabricated extra lines onto stderr.
    let e = HiggsError::HubTransport {
        repo: "acme/m".into(),
        detail: "boom\x1b[2J\nhiggs: fake second error".into(),
    };
    let printed = sanitized_error(&e).to_string();
    assert!(!printed.contains('\x1b'), "{printed:?}");
    assert!(!printed.contains('\n'), "{printed:?}");
    assert!(
        printed.contains("fake second error"),
        "text kept: {printed:?}"
    );
}

#[test]
fn sanitize_terminal_strips_ansi_and_control_bytes_but_keeps_layout() {
    // A Hub README is arbitrary bytes — ANSI escapes and control chars must
    // never reach the operator's terminal; newlines/tabs are layout and stay.
    let hostile = "safe\n\x1b[2Jcleared\twide\r\x07bell";
    assert_eq!(sanitize_terminal(hostile), "safe\n[2Jcleared\twide\nbell");
}

#[test]
fn render_detail_shows_quants_sizes_and_fit() {
    let d = CatalogModelDetail {
        summary: summary("acme/m", false),
        tags: vec!["gguf".into()],
        readme: Some("# Title\nbody".into()),
        default_file: None,
        quants: vec![
            CatalogQuant {
                file: "m-Q4_K_M.gguf".into(),
                quant: Some("Q4_K_M".into()),
                size_bytes: Some(4_600_000_000),
                downloaded: true,
                fit: Some(FitAssessment {
                    fits: true,
                    needed_bytes: 4_600_000_000,
                    available_bytes: 20_000_000_000,
                }),
            },
            CatalogQuant {
                file: "m-F16.gguf".into(),
                quant: Some("F16".into()),
                size_bytes: None,
                downloaded: false,
                fit: None,
            },
        ],
        more_by_author: vec![summary("acme/other", false)],
    };
    let out = render_detail(&d);
    assert!(out.contains("acme/m"));
    assert!(out.contains("m-Q4_K_M.gguf"));
    assert!(out.contains("4.3 GiB"));
    let q4 = out.lines().find(|l| l.contains("Q4_K_M.gguf")).unwrap();
    assert!(q4.contains("fits"));
    assert!(q4.contains("downloaded"));
    let f16 = out.lines().find(|l| l.contains("m-F16.gguf")).unwrap();
    assert!(f16.contains('-'), "unknown size/fit shown as placeholders");
    assert!(out.contains("acme/other"));
    assert!(out.contains("# Title"));
}
