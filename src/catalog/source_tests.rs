use super::*;

use crate::catalog::wire::{CatalogSort, DEFAULT_SEARCH_LIMIT, MAX_SEARCH_LIMIT};

#[test]
fn hub_sort_maps_every_variant_to_the_hub_key() {
    assert_eq!(hub_sort(CatalogSort::Downloads), "downloads");
    assert_eq!(hub_sort(CatalogSort::Likes), "likes");
    assert_eq!(hub_sort(CatalogSort::Updated), "lastModified");
    assert_eq!(hub_sort(CatalogSort::Trending), "trendingScore");
}

#[test]
fn effective_limit_defaults_zero_and_clamps_to_the_cap() {
    assert_eq!(effective_limit(0), DEFAULT_SEARCH_LIMIT as usize);
    assert_eq!(effective_limit(10), 10);
    assert_eq!(
        effective_limit(MAX_SEARCH_LIMIT + 500),
        MAX_SEARCH_LIMIT as usize
    );
}

#[test]
fn clip_utf8_cuts_on_a_char_boundary_and_keeps_short_input_whole() {
    assert_eq!(clip_utf8(b"hello", 100), "hello");
    // "héllo" — cutting at byte 2 lands inside the two-byte 'é'.
    let s = "héllo".as_bytes();
    assert_eq!(clip_utf8(s, 2), "h");
    assert_eq!(clip_utf8(s, 3), "hé");
    assert_eq!(clip_utf8(s, 0), "");
}

#[tokio::test]
async fn hub_reads_reject_a_repo_id_that_could_alter_the_request_url() {
    // `?`/`#`/space/`..` in a repo id would land in the request PATH (the
    // crate interpolates it) — every Hub read validates the id first, before
    // any network touch, so no fixture is needed here.
    for bad in ["a?b/c", "a/../x", "a b/c", "a#f/c", "/c", "a/", "a"] {
        assert!(
            matches!(
                HfSource.info(bad).await,
                Err(crate::diagnostic::HiggsError::HubClient { .. })
            ),
            "info({bad:?}) must be refused"
        );
        assert!(HfSource
            .file_sizes(bad, &["f.gguf".to_string()])
            .await
            .is_err());
        assert!(HfSource.readme(bad).await.is_err());
    }
}

/// The production `HfSource` over the crate's REAL HTTP paths against the
/// loopback fixture Hub — one sync test (own runtime) so `TEST_ENV_LOCK` is
/// never held across an await in an async fn.
#[test]
fn hf_source_end_to_end_over_a_loopback_hub() {
    let _env = crate::TEST_ENV_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(async {
        let hub = crate::catalog::test_support::fixture_hub().await;
        let _redirect = crate::catalog::test_support::EnvRedirect::set(&hub.endpoint, None);

        // search: rows come back and the wire query carries the GGUF filter,
        // sort key, and clamped limit.
        let q = crate::catalog::wire::CatalogQuery {
            search: "tiny".into(),
            author: None,
            sort: Some(CatalogSort::Likes),
            limit: Some(7),
            compatible_only: None,
        };
        let hits = HfSource.search(&q).await.expect("search");
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].id, "acme/tiny");
        let sent = hub
            .list_queries
            .lock()
            .unwrap()
            .first()
            .cloned()
            .expect("query");
        assert!(sent.contains("filter=gguf"), "{sent}");
        assert!(sent.contains("sort=likes"), "{sent}");
        assert!(sent.contains("limit=7"), "{sent}");

        // info: the gguf block and siblings arrive.
        let info = HfSource.info("acme/tiny").await.expect("info");
        assert_eq!(info.id, "acme/tiny");
        assert!(info.gguf.is_some());
        assert_eq!(info.siblings.as_deref().map(<[_]>::len), Some(3));

        // file_sizes: paths-info flattened with the LFS size preferred.
        let sizes = HfSource
            .file_sizes(
                "acme/tiny",
                &["tiny-Q4_K_M.gguf".to_string(), "tiny-F16.gguf".to_string()],
            )
            .await
            .expect("sizes");
        assert_eq!(sizes.get("tiny-Q4_K_M.gguf"), Some(&4_000));
        assert_eq!(sizes.get("tiny-F16.gguf"), Some(&9_000));
        // No files → no request, empty map.
        assert!(HfSource
            .file_sizes("acme/tiny", &[])
            .await
            .unwrap()
            .is_empty());

        // readme: present, absent (404 → None), and byte-bounded.
        let readme = HfSource.readme("acme/tiny").await.expect("readme");
        assert_eq!(readme.as_deref(), Some("# tiny\nhello"));
        assert!(HfSource
            .readme("acme/noreadme")
            .await
            .expect("no readme")
            .is_none());
        let big = HfSource.readme("acme/bigreadme").await.expect("big readme");
        assert_eq!(big.expect("clipped").len(), README_MAX_BYTES);

        // A dead endpoint classifies as a hub failure instead of hanging.
        let closed = {
            let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            let addr = l.local_addr().unwrap();
            drop(l);
            format!("http://{addr}")
        };
        let _redirect2 = crate::catalog::test_support::EnvRedirect::set(&closed, None);
        assert!(HfSource.search(&q).await.is_err());
    });
}

#[test]
fn sizes_from_entries_prefers_lfs_size_and_skips_directories() {
    let entries: Vec<huggingface_hub::RepoTreeEntry> = serde_json::from_value(serde_json::json!([
        { "type": "file", "oid": "a", "size": 134u64, "path": "m-Q4_K_M.gguf",
          "lfs": { "size": 4_000_000u64 } },
        { "type": "file", "oid": "b", "size": 512u64, "path": "config.json" },
        { "type": "directory", "oid": "c", "path": "sub" },
    ]))
    .expect("fixture entries");
    let sizes = sizes_from_entries(&entries);
    assert_eq!(sizes.get("m-Q4_K_M.gguf"), Some(&4_000_000)); // LFS real size wins
    assert_eq!(sizes.get("config.json"), Some(&512)); // plain size otherwise
    assert_eq!(sizes.len(), 2);
}
