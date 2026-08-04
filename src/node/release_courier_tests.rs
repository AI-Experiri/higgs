//! Unit tests for the hub release courier's PURE resolution + the no-network SSRF/scheme vet, plus
//! the async fetch/assemble driven against a LOOPBACK server (deterministic, no external network,
//! no GGUF). The whole-fleet/one-node PUSH over real iroh is the integration test's job
//! (`tests/remote_update_push.rs`).

use std::sync::Arc;

use super::*;

fn url(s: &str) -> Url {
    Url::parse(s).expect("valid test URL")
}

/// Spin a loopback "release server" serving a valid manifest + sig at `/d/higgs.manifest`, plus an
/// OVER-CAP manifest at `/d/big.manifest`. Returns its port. The `.tar.gz` is never fetched by the
/// courier (the node does that), so it is not served.
async fn spawn_release_server() -> u16 {
    let manifest = serde_json::json!({
        "schema": 1,
        "version": "1.2.3",
        "commit": "abc123",
        "file": "higgs-art.tar.gz",
        "target": "aarch64-apple-darwin",
        "variant": "metal",
        "sha256": "00",
    })
    .to_string();
    // A body just over MAX_MANIFEST_BYTES (64 KiB) — the cap must reject it.
    let big = "x".repeat(70 * 1024);
    let app = axum::Router::new()
        .route(
            "/d/higgs.manifest",
            axum::routing::get(move || {
                let m = manifest.clone();
                async move { m }
            }),
        )
        .route(
            "/d/higgs.manifest.minisig",
            axum::routing::get(|| async { "untrusted comment\nRWSig==\n".to_string() }),
        )
        .route(
            "/d/big.manifest",
            axum::routing::get(move || {
                let b = big.clone();
                async move { b }
            }),
        )
        .route(
            "/d/big.manifest.minisig",
            axum::routing::get(|| async { "sig".to_string() }),
        )
        // A well-formed-JSON but INVALID `UpdateManifest` (schema is a string, not u32) whose value
        // carries a capability token — the parse error must NOT echo it (codex REL-P4e r5 #1).
        .route(
            "/d/evil.manifest",
            axum::routing::get(|| async {
                "{\"schema\":\"CAP-TOKEN-LEAK\",\"version\":\"1.2.3\",\"commit\":\"c\",\
                 \"file\":\"a.tar.gz\",\"target\":\"t\",\"variant\":\"metal\",\"sha256\":\"00\"}"
                    .to_string()
            }),
        )
        .route(
            "/d/evil.manifest.minisig",
            axum::routing::get(|| async { "untrusted comment\nRWSig==\n".to_string() }),
        );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    port
}

fn empty_fleet() -> Arc<crate::node::fleet::HubFleet> {
    Arc::new(crate::node::fleet::HubFleet::new(Arc::new(
        crate::log_bus::LogBus::new(),
    )))
}

// ── asset_suffix + manifest_filename — the release.yml naming ───────────────

#[test]
fn asset_suffix_matches_release_yaml_table() {
    // metal (macOS) + cpu (Linux) are the sole/default variant → bare triple.
    assert_eq!(
        asset_suffix("aarch64-apple-darwin", "metal"),
        "aarch64-apple-darwin"
    );
    assert_eq!(
        asset_suffix("x86_64-unknown-linux-gnu", "cpu"),
        "x86_64-unknown-linux-gnu"
    );
    // cuda ships a SEPARATE artifact for the same triple → `-cuda` suffix.
    assert_eq!(
        asset_suffix("x86_64-unknown-linux-gnu", "cuda"),
        "x86_64-unknown-linux-gnu-cuda"
    );
    // An unknown variant gets a DISTINCT name (404s, never silently the wrong artifact).
    assert_eq!(asset_suffix("some-triple", "rocm"), "some-triple-rocm");
}

#[test]
fn manifest_filename_shape() {
    assert_eq!(
        manifest_filename("1.2.3", "aarch64-apple-darwin", "metal"),
        "higgs-v1.2.3-aarch64-apple-darwin.manifest"
    );
    assert_eq!(
        manifest_filename("0.5.0", "x86_64-unknown-linux-gnu", "cuda"),
        "higgs-v0.5.0-x86_64-unknown-linux-gnu-cuda.manifest"
    );
}

// ── release_version_from_base ───────────────────────────────────────────────

#[test]
fn version_from_base_reads_the_tag_segment() {
    assert_eq!(
        release_version_from_base(&url("https://ex.com/org/repo/releases/download/v1.2.3"))
            .unwrap(),
        "1.2.3"
    );
    // A trailing slash is tolerated (the empty last segment is skipped).
    assert_eq!(
        release_version_from_base(&url("https://ex.com/releases/download/v1.2.3/")).unwrap(),
        "1.2.3"
    );
    // A prerelease semver is fine.
    assert_eq!(
        release_version_from_base(&url("https://ex.com/download/v1.2.3-beta.1")).unwrap(),
        "1.2.3-beta.1"
    );
}

#[test]
fn version_from_base_rejects_non_tag_segments() {
    // Missing the `v` prefix.
    assert!(release_version_from_base(&url("https://ex.com/download/1.2.3")).is_err());
    // A `v`-prefixed but non-semver segment.
    assert!(release_version_from_base(&url("https://ex.com/download/vlatest")).is_err());
    // No path at all.
    assert!(release_version_from_base(&url("https://ex.com")).is_err());
    // Bare `v`.
    assert!(release_version_from_base(&url("https://ex.com/download/v")).is_err());
}

// ── per_node_manifest_url + sig_url_for + artifact_url_from_manifest ─────────

#[test]
fn per_node_manifest_url_appends_under_the_tag_dir() {
    // No trailing slash on the base — join must APPEND, not replace `v1.2.3`.
    let m = per_node_manifest_url(
        &url("https://ex.com/org/repo/releases/download/v1.2.3"),
        "1.2.3",
        "aarch64-apple-darwin",
        "metal",
    )
    .unwrap();
    assert_eq!(
        m.as_str(),
        "https://ex.com/org/repo/releases/download/v1.2.3/higgs-v1.2.3-aarch64-apple-darwin.manifest"
    );
    // With a trailing slash — same result.
    let m2 = per_node_manifest_url(
        &url("https://ex.com/releases/download/v1.2.3/"),
        "1.2.3",
        "x86_64-unknown-linux-gnu",
        "cuda",
    )
    .unwrap();
    assert_eq!(
        m2.as_str(),
        "https://ex.com/releases/download/v1.2.3/higgs-v1.2.3-x86_64-unknown-linux-gnu-cuda.manifest"
    );
}

#[test]
fn sig_url_is_the_manifest_plus_minisig() {
    let m = url("https://ex.com/d/v1.2.3/higgs-v1.2.3-aarch64-apple-darwin.manifest");
    assert_eq!(
        sig_url_for(&m).unwrap().as_str(),
        "https://ex.com/d/v1.2.3/higgs-v1.2.3-aarch64-apple-darwin.manifest.minisig"
    );
}

#[test]
fn artifact_url_is_the_bare_sibling() {
    let m = url("https://ex.com/d/v1.2.3/higgs-v1.2.3-aarch64-apple-darwin.manifest");
    let a = artifact_url_from_manifest(&m, "higgs-v1.2.3-aarch64-apple-darwin.tar.gz").unwrap();
    assert_eq!(
        a.as_str(),
        "https://ex.com/d/v1.2.3/higgs-v1.2.3-aarch64-apple-darwin.tar.gz"
    );
}

#[test]
fn artifact_url_rejects_non_bare_file_names() {
    let m = url("https://ex.com/d/v1.2.3/higgs.manifest");
    for bad in [
        "",                    // empty
        "sub/higgs.tar.gz",    // has a slash
        "..",                  // traversal
        "../higgs.tar.gz",     // traversal
        ".hidden.tar.gz",      // leading dot
        "a\\b.tar.gz",         // backslash
        "/etc/passwd",         // absolute
        "%2e%2e",              // percent-encoded traversal
        "%2e%2e/higgs.tar.gz", // percent-encoded traversal
        "https:evil.com/x",    // a `scheme:` that Url::join would use to re-root off-origin
        "//evil.com/x.tar.gz", // protocol-relative — an off-host result
        "higgs.tar.gz?token",  // a query would decorate the derived artifact URL
        "higgs.tar.gz#frag",   // a fragment likewise
    ] {
        assert!(
            artifact_url_from_manifest(&m, bad).is_err(),
            "expected {bad:?} to be refused as a non-bare filename"
        );
    }
    // The happy path stays a same-origin sibling (the same-origin guard does not reject it).
    assert_eq!(
        artifact_url_from_manifest(&m, "higgs.tar.gz")
            .unwrap()
            .as_str(),
        "https://ex.com/d/v1.2.3/higgs.tar.gz"
    );
}

#[test]
fn artifact_url_rejection_never_echoes_the_unverified_file() {
    // The manifest is UNVERIFIED at the courier (the node checks the signature), so a hostile
    // release response could stuff a capability-bearing string into `file`. The rejection error is
    // copied verbatim into the per-node fleet report, so it must NEVER reflect `file`. Guards codex
    // REL-P4e r3 #4. (Both the bare-filename filter AND the same-origin guard are covered: `:`
    // trips the filter; a value that slips it would trip the same-origin assert — both must redact.)
    let m = url("https://ex.com/d/v1.2.3/higgs.manifest");
    for evil in [
        "cap-SECRET-TOKEN:payload",
        "sub/SECRET-TOKEN.tar.gz",
        "higgs.tar.gz?sig=SECRET-TOKEN", // a query token must not leak into the report either
    ] {
        let e = artifact_url_from_manifest(&m, evil).unwrap_err();
        assert!(
            !e.to_string().contains("SECRET-TOKEN"),
            "unverified manifest `file` leaked into the error: {e}"
        );
    }
}

// ── parse_courier_url — the pure scheme/host-shape vet ──────────────────────

#[test]
fn parse_url_accepts_https_any_host_and_http_literal_loopback_ip() {
    assert!(parse_courier_url("https://github.com/o/r/releases/download/v1.2.3").is_ok());
    // Loopback affordances for on-box mirrors / test servers — LITERAL loopback IPs only.
    assert!(parse_courier_url("http://127.0.0.1:8080/x.manifest").is_ok());
    assert!(parse_courier_url("http://[::1]:8080/x.manifest").is_ok());
    // https to a literal loopback IP is fine too.
    assert!(parse_courier_url("https://127.0.0.1/x.manifest").is_ok());
}

#[test]
fn parse_url_rejects_http_to_the_name_localhost() {
    // The NAME `localhost` is NOT a loopback shortcut for the courier: NSS/`/etc/hosts`/DNS can
    // remap it off-box, so a plaintext `http://localhost` must be refused at the pure vet (only a
    // literal 127.0.0.0/8 or ::1 IP takes the on-box branch). Guards codex REL-P4e r3 #3.
    let e = parse_courier_url("http://localhost:8080/x.manifest").unwrap_err();
    assert!(matches!(e, HiggsError::UpdateFetchFailed { .. }));
    // `https://localhost` parses (any-host https), but the async client then resolves it and the
    // SSRF vet refuses the loopback address — so the name never reaches an unvetted on-box fetch.
    assert!(parse_courier_url("https://localhost/x.manifest").is_ok());
}

#[test]
fn parse_url_rejects_plaintext_http_to_a_public_host() {
    let e = parse_courier_url("http://example.com/x.manifest").unwrap_err();
    assert!(matches!(e, HiggsError::UpdateFetchFailed { .. }));
    // The error must not leak the host.
    assert!(!e.to_string().contains("example.com"), "URL leaked: {e}");
}

#[test]
fn parse_url_rejects_userinfo_query_fragment_and_bad_scheme() {
    assert!(parse_courier_url("https://user:pass@ex.com/x.manifest").is_err());
    assert!(parse_courier_url("https://ex.com/x.manifest?sig=abc").is_err());
    assert!(parse_courier_url("https://ex.com/x.manifest#frag").is_err());
    assert!(parse_courier_url("ftp://ex.com/x.manifest").is_err());
    // A made-up capability scheme is refused without echoing the scheme.
    let e = parse_courier_url("cap-secret://ex.com/x.manifest").unwrap_err();
    assert!(!e.to_string().contains("cap-secret"), "scheme leaked: {e}");
}

// ── build_manifest_client — the async SSRF resolve/vet (NO server needed) ────

#[tokio::test]
async fn client_refuses_a_private_literal_ip_host() {
    // A literal private/reserved IP is vetted directly — no DNS, no connect.
    for host in [
        "https://10.0.0.5/x.manifest",
        "https://192.168.1.10/x.manifest",
        "https://169.254.1.1/x.manifest", // link-local
        "https://[fd00::1]/x.manifest",   // ULA v6
        "https://[fe80::1]/x.manifest",   // link-local v6
    ] {
        let u = url(host);
        let e = build_manifest_client(&u)
            .await
            .expect_err("private literal must be refused");
        assert!(
            matches!(e, HiggsError::UpdateFetchFailed { .. }),
            "{host}: {e:?}"
        );
    }
}

#[tokio::test]
async fn client_builds_for_a_loopback_host() {
    // Loopback is allowed (on-box / test) and builds a client without any SSRF resolution.
    assert!(build_manifest_client(&url("http://127.0.0.1:9/x.manifest"))
        .await
        .is_ok());
    assert!(
        build_manifest_client(&url("https://127.0.0.1:9/x.manifest"))
            .await
            .is_ok()
    );
}

#[tokio::test]
async fn client_refuses_the_name_localhost_by_resolving_it() {
    // The NAME `localhost` is NOT taken as an on-box shortcut: `build_manifest_client` resolves it
    // (localhost is in /etc/hosts → 127.0.0.1 / ::1, so this stays offline) and the SSRF vet then
    // refuses the loopback address. A remapped `localhost` therefore cannot cause an unvetted
    // off-box fetch. Guards codex REL-P4e r3 #3 on the async side (the pure side is
    // `parse_url_rejects_http_to_the_name_localhost`).
    let e = build_manifest_client(&url("https://localhost/x.manifest"))
        .await
        .expect_err("the name localhost must be resolved + refused as loopback, not trusted");
    assert!(matches!(e, HiggsError::UpdateFetchFailed { .. }), "{e:?}");
}

#[tokio::test]
async fn resolve_push_params_refuses_a_private_manifest_host() {
    // The full node_update entry-point still fails closed on an SSRF-prone host, before any fetch.
    // A LITERAL private IP is vetted with no DNS, so this stays network-free.
    let e = resolve_push_params("https://10.0.0.1/x.manifest", None)
        .await
        .expect_err("a private host must fail");
    assert!(matches!(e, HiggsError::UpdateFetchFailed { .. }), "{e:?}");

    let e2 = resolve_push_params("http://example.com/x.manifest", None)
        .await
        .expect_err("plaintext http to a public host must fail");
    assert!(matches!(e2, HiggsError::UpdateFetchFailed { .. }), "{e2:?}");
}

// ── async fetch + assemble against a loopback server (deterministic, no external network) ────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn resolve_push_params_fetches_and_assembles() {
    let port = spawn_release_server().await;
    let manifest_url = format!("http://127.0.0.1:{port}/d/higgs.manifest");
    // Bind to the matching version — the happy path.
    let params = resolve_push_params(&manifest_url, Some("1.2.3"))
        .await
        .expect("valid manifest + sig fetched and assembled");
    // The manifest text is carried VERBATIM for the node to verify over.
    assert!(
        params.manifest.contains("\"version\":\"1.2.3\""),
        "{}",
        params.manifest
    );
    assert!(
        params.manifest_sig.contains("RWSig"),
        "{}",
        params.manifest_sig
    );
    // The artifact URL is the sibling named by the manifest's bare `file`.
    assert_eq!(
        params.artifact_url,
        format!("http://127.0.0.1:{port}/d/higgs-art.tar.gz")
    );
    assert_eq!(params.target_version.as_deref(), Some("1.2.3"));
    assert!(params.pinned_key_id.is_none());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn resolve_push_params_rejects_a_version_mismatch() {
    // Fix 2: an origin serving a validly-shaped manifest of a DIFFERENT version at the requested
    // path is refused when the caller binds the expected version (the fleet_update case).
    let port = spawn_release_server().await; // serves version "1.2.3"
    let manifest_url = format!("http://127.0.0.1:{port}/d/higgs.manifest");
    let e = resolve_push_params(&manifest_url, Some("2.0.0"))
        .await
        .expect_err("a manifest whose version != the requested version must be refused");
    assert!(
        matches!(e, HiggsError::UpdateManifestInvalid { .. }),
        "{e:?}"
    );
    // Fix 3 (redaction): the mismatch error must NOT echo the requested version — a base-URL path
    // segment can carry a capability token that this message is copied into the fleet report.
    let e2 = resolve_push_params(&manifest_url, Some("2.0.0+SecretToken"))
        .await
        .expect_err("mismatch");
    assert!(
        !e2.to_string().contains("SecretToken") && !e2.to_string().contains("2.0.0"),
        "the version-mismatch error leaked the requested version: {e2}"
    );
    // `None` (node_update's exact-URL path) does NOT bind, so the same fetch succeeds.
    assert!(resolve_push_params(&manifest_url, None).await.is_ok());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn resolve_push_params_maps_a_404_manifest() {
    let port = spawn_release_server().await;
    let e = resolve_push_params(&format!("http://127.0.0.1:{port}/d/missing.manifest"), None)
        .await
        .expect_err("a 404 manifest must be a fetch error");
    assert!(matches!(e, HiggsError::UpdateFetchFailed { .. }), "{e:?}");
    // The redacted URL never leaks the host/path.
    assert!(!e.to_string().contains("127.0.0.1"), "URL leaked: {e}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn resolve_push_params_never_echoes_the_unverified_manifest_body() {
    // A well-formed-JSON but invalid manifest whose value carries a capability token: the parse
    // error is copied verbatim into the per-node fleet report, so it must NEVER reflect ANY
    // server-controlled value. Guards codex REL-P4e r5 #1 (the field value) + r6 #3 (the serde
    // line/column, which a hostile origin can POSITION to encode a numeric token) — the message is
    // fully constant.
    let port = spawn_release_server().await;
    let e = resolve_push_params(&format!("http://127.0.0.1:{port}/d/evil.manifest"), None)
        .await
        .expect_err("an invalid manifest must be a manifest error");
    assert!(
        matches!(e, HiggsError::UpdateManifestInvalid { .. }),
        "{e:?}"
    );
    let msg = e.to_string();
    assert!(
        !msg.contains("CAP-TOKEN-LEAK"),
        "the unverified manifest body leaked into the error: {e}"
    );
    assert!(
        !msg.contains("line") && !msg.contains("column"),
        "serde parse coordinates leaked into the error: {e}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn resolve_push_params_rejects_an_over_cap_manifest() {
    let port = spawn_release_server().await;
    let e = resolve_push_params(&format!("http://127.0.0.1:{port}/d/big.manifest"), None)
        .await
        .expect_err("a >64KiB manifest must exceed the cap");
    assert!(matches!(e, HiggsError::UpdateFetchFailed { .. }), "{e:?}");
    assert!(e.to_string().contains("cap"), "{e}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fleet_update_with_no_connected_nodes_reports_empty() {
    let port = spawn_release_server().await;
    let fleet = empty_fleet();
    // A well-formed base URL — no nodes are connected, so the results list is empty (nothing to
    // fetch/push), but the call succeeds (a systemic-error-free run).
    let report = fleet_update(
        &fleet,
        &format!("http://127.0.0.1:{port}/releases/download/v1.2.3"),
    )
    .await
    .expect("fleet_update over an empty fleet succeeds");
    assert_eq!(report["results"].as_array().unwrap().len(), 0, "{report}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fleet_update_rejects_a_base_url_without_a_version_tag() {
    let fleet = empty_fleet();
    // No `v<semver>` tag in the path → a single systemic error, before any per-node work.
    let e = fleet_update(&fleet, "https://example.com/releases/latest")
        .await
        .expect_err("a base URL without a v<version> tag is refused");
    assert!(matches!(e, HiggsError::UpdateFetchFailed { .. }), "{e:?}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn node_update_resolves_then_fails_on_an_absent_node() {
    let port = spawn_release_server().await;
    let fleet = empty_fleet();
    // The manifest resolves fine (loopback server), but the node isn't connected → HG027.
    let e = node_update(
        &fleet,
        "some-endpoint-id",
        &format!("http://127.0.0.1:{port}/d/higgs.manifest"),
    )
    .await
    .expect_err("push to an unconnected node is unreachable");
    assert!(matches!(e, HiggsError::NodeUnreachable { .. }), "{e:?}");
}

// ── UPx: version-choosing flow ────────────────────────────────────────────────

#[test]
fn releases_api_url_accepts_github_repo_shape_only() {
    let api = releases_api_url("https://github.com/AI-Experiri/higgs/releases").unwrap();
    assert_eq!(
        api.as_str(),
        "https://api.github.com/repos/AI-Experiri/higgs/releases?per_page=100"
    );
    // Trailing slash tolerated (empty segments are filtered).
    releases_api_url("https://github.com/o/r/releases/").unwrap();
    // Non-GitHub mirror: listing unsupported (exact-URL push is the path there).
    assert!(releases_api_url("https://mirror.example/higgs/releases").is_err());
    // Wrong path shape on github.com.
    assert!(releases_api_url("https://github.com/AI-Experiri/higgs").is_err());
    assert!(releases_api_url("https://github.com/o/r/releases/extra").is_err());
}

#[test]
fn newer_releases_filters_and_sorts() {
    let target = "aarch64-apple-darwin";
    let variant = "metal";
    let entry = |tag: &str, draft: bool, with_asset: bool| {
        let assets = if with_asset {
            let ver = tag.trim_start_matches('v');
            let base = format!("higgs-v{ver}-{}", asset_suffix(target, variant));
            serde_json::json!([
                { "name": format!("{base}.manifest") },
                { "name": format!("{base}.manifest.minisig") },
                { "name": format!("{base}.tar.gz") },
            ])
        } else {
            serde_json::json!([])
        };
        serde_json::json!({ "tag_name": tag, "draft": draft, "assets": assets })
    };
    let index = serde_json::json!([
        entry("v0.1.0-beta.3", false, true), // newer, has asset → in
        entry("v0.1.0-beta.2", false, true), // newer, has asset → in
        entry("v0.1.0-beta.1", false, true), // == current → out
        entry("v0.0.9", false, true),        // older → out
        entry("v0.2.0", true, true),         // draft → out
        entry("v0.3.0", false, false),       // newer, NO asset for this build → out
        serde_json::json!({ "tag_name": "not-a-tag", "draft": false, "assets": [] }),
    ]);
    // A manifest-only (incomplete) release must NOT be offered.
    let partial = serde_json::json!([{
        "tag_name": "v0.4.0", "draft": false,
        "assets": [{ "name": manifest_filename("0.4.0", target, variant) }]
    }]);
    assert!(
        newer_releases_for(&partial, "0.1.0", target, variant).is_empty(),
        "incomplete asset trio is not offered"
    );
    // A +build-metadata tag is never offered (no semver precedence; CI never
    // publishes one).
    let meta = serde_json::json!([entry("v0.9.0+hotpatch", false, true)]);
    assert!(newer_releases_for(&meta, "0.1.0", target, variant).is_empty());
    let got = newer_releases_for(&index, "0.1.0-beta.1", target, variant);
    assert_eq!(got, vec!["0.1.0-beta.3", "0.1.0-beta.2"], "newest first");
    // Unknown current version → never guess an upgrade.
    assert!(newer_releases_for(&index, "", target, variant).is_empty());
    assert!(newer_releases_for(&index, "garbage", target, variant).is_empty());
    // Non-array index → empty, not a panic.
    assert!(newer_releases_for(&serde_json::json!({}), "0.1.0", target, variant).is_empty());
}

// ── version-flow ops against a REAL admitted fleet (loopback iroh) ──────────

/// Admit ONE node with a chosen identity so the version ops see a realistic
/// `NodeUpdateTarget` — mirrors `fleet_tests::fleet_with_one_node`, with the
/// identity fields parameterized.
async fn fleet_with_identity(
    software_version: Option<&str>,
    target: Option<&str>,
    variant: Option<&str>,
    version_capable: bool,
) -> (Arc<crate::node::fleet::HubFleet>, String, tempfile::TempDir) {
    use crate::node::test_support::{fake_runtime, local_endpoint, stage_dummy_model};
    let (root, _model_id) = stage_dummy_model("higgs-test/m");
    let hub = local_endpoint().await;
    let node = local_endpoint().await;
    let hub_addr = hub.addr();
    let node_key = node.id().to_string();
    let rt = Arc::new(fake_runtime(vec![root.path().to_path_buf()]));
    tokio::spawn(async move {
        let node_conn = node
            .connect(hub_addr, crate::remote::ALPN)
            .await
            .expect("connect");
        crate::node::serve_node(node_conn, rt).await;
    });
    let conn = hub.accept().await.expect("incoming").await.expect("conn");
    std::mem::forget(hub);
    let fleet = Arc::new(crate::node::fleet::HubFleet::new(Arc::new(
        crate::log_bus::LogBus::new(),
    )));
    fleet
        .add_node_with_identity(
            node_key.clone(),
            Arc::new(crate::node::transport::NodeTransport::new(conn)),
            None,
            None,
            software_version.map(str::to_string),
            false,
            None,
            true,
            target.map(str::to_string),
            variant.map(str::to_string),
            true,
            version_capable,
        )
        .await;
    (fleet, node_key, root)
}

#[tokio::test]
async fn node_releases_unknown_node_is_unreachable() {
    let err = node_releases(&empty_fleet(), "nope", "https://github.com/o/r/releases")
        .await
        .unwrap_err();
    assert!(
        matches!(err, HiggsError::NodeUnreachable { .. }),
        "got {err:?}"
    );
}

#[tokio::test]
async fn node_releases_empty_listing_for_an_unknown_build_without_network() {
    // Version present but no target/variant → the network is never touched (the
    // release_url here would fail `releases_api_url` if it were) and the listing
    // is empty with the capability faithfully reported.
    let (fleet, node, _root) = fleet_with_identity(Some("0.1.0"), None, None, true).await;
    let v = node_releases(&fleet, &node, "https://not-github.example/whatever")
        .await
        .expect("no-network listing");
    assert_eq!(v["node"], node);
    assert_eq!(v["current"], "0.1.0");
    assert_eq!(v["available"], json!([]));
    assert_eq!(v["update_by_version"], true);
}

#[tokio::test]
async fn node_releases_skips_the_fetch_for_an_unparseable_version() {
    let (fleet, node, _root) = fleet_with_identity(
        Some("not-semver"),
        Some("aarch64-apple-darwin"),
        Some("metal"),
        false,
    )
    .await;
    let v = node_releases(&fleet, &node, "https://not-github.example/whatever")
        .await
        .expect("no-network listing");
    assert_eq!(v["available"], json!([]));
    assert_eq!(v["update_by_version"], false);
}

#[tokio::test]
async fn node_update_version_rejects_non_semver_before_any_lookup() {
    let err = node_update_version(&empty_fleet(), "any", "v1.2.3")
        .await
        .unwrap_err();
    assert!(format!("{err}").contains("semver"), "got {err}");
}

#[tokio::test]
async fn node_update_version_unknown_node_is_unreachable() {
    let err = node_update_version(&empty_fleet(), "nope", "1.2.3")
        .await
        .unwrap_err();
    assert!(
        matches!(err, HiggsError::NodeUnreachable { .. }),
        "got {err:?}"
    );
}

#[tokio::test]
async fn node_update_version_refuses_a_node_without_the_capability() {
    let (fleet, node, _root) = fleet_with_identity(
        Some("0.1.0"),
        Some("aarch64-apple-darwin"),
        Some("metal"),
        false,
    )
    .await;
    let err = node_update_version(&fleet, &node, "9.9.9")
        .await
        .unwrap_err();
    assert!(
        format!("{err}").contains("does not support version-triggered updates"),
        "got {err}"
    );
}

#[tokio::test]
async fn fleet_update_version_rejects_non_semver() {
    let err = fleet_update_version(&empty_fleet(), "v1.2.3", "https://github.com/o/r/releases")
        .await
        .unwrap_err();
    assert!(format!("{err}").contains("semver"), "got {err}");
}

#[tokio::test]
async fn fleet_update_version_reports_local_skips_without_touching_the_network() {
    // Each of these rows is decided hub-side; none is pushable, so the release
    // index is NEVER fetched (release_url would fail `releases_api_url` if it were).
    for (sv, target, variant, capable, expect) in [
        (
            Some("0.1.0"),
            Some("aarch64-apple-darwin"),
            Some("metal"),
            false,
            "does not support version-triggered updates",
        ),
        (
            Some("not-semver"),
            Some("aarch64-apple-darwin"),
            Some("metal"),
            true,
            "parseable version",
        ),
        (
            Some("9.9.9"),
            Some("aarch64-apple-darwin"),
            Some("metal"),
            true,
            "already at or above",
        ),
        (Some("0.1.0"), None, None, true, "build target/variant"),
    ] {
        let (fleet, node, _root) = fleet_with_identity(sv, target, variant, capable).await;
        let v = fleet_update_version(&fleet, "1.0.0", "https://not-github.example/x")
            .await
            .expect("local skip report");
        let rows = v["results"].as_array().expect("rows");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["node"], node);
        assert_eq!(rows[0]["status"], "skipped");
        let reason = rows[0]["reason"].as_str().unwrap_or_default();
        assert!(reason.contains(expect), "reason {reason:?} vs {expect:?}");
    }
}

#[tokio::test]
async fn fleet_update_skips_a_node_without_a_reported_build() {
    // Capable of updates but never reported target/variant → the courier cannot
    // pick an asset; the row is SKIPPED, never pushed-and-failed. No manifest is
    // fetched for a skip, so the (bound, silent) release origin is never asked.
    let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!(
        "http://127.0.0.1:{}/higgs/v9.9.9",
        l.local_addr().unwrap().port()
    );
    let (fleet, node, _root) = fleet_with_identity(Some("0.1.0"), None, None, true).await;
    let v = fleet_update(&fleet, &base).await.expect("report");
    let rows = v["results"].as_array().unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["node"], node);
    assert_eq!(rows[0]["status"], "skipped");
    assert!(
        rows[0]["reason"]
            .as_str()
            .unwrap()
            .contains("build target/variant"),
        "{:?}",
        rows[0]
    );
}

#[tokio::test]
async fn fleet_update_reports_a_manifest_fetch_failure_per_node() {
    // Full build identity → the courier derives the per-node manifest URL and
    // fetches it; a 404 origin folds into an ERROR row for that node (never
    // fatal to the fleet report).
    let port = spawn_release_server().await;
    let base = format!("http://127.0.0.1:{port}/wrong-path/v9.9.9");
    let (fleet, node, _root) = fleet_with_identity(
        Some("0.1.0"),
        Some("aarch64-apple-darwin"),
        Some("metal"),
        true,
    )
    .await;
    let v = fleet_update(&fleet, &base).await.expect("report");
    let rows = v["results"].as_array().unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["node"], node);
    assert_eq!(rows[0]["status"], "error", "{:?}", rows[0]);
}
