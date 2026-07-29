//! Integration coverage for the HUB release-courier's async fetch / SSRF-vet / assemble EDGE paths
//! (REL-P4e). The happy-path whole-fleet + single-node PUSH over real iroh lives in
//! `remote_update_push.rs`; this file drives the courier's PUBLIC resolve entry points
//! (`resolve_push_params` and the pure resolvers) against a LOCAL HTTP origin + literal private IPs,
//! so every rejection branch — 404, over-cap, malformed/UTF-8, version-mismatch bind, SSRF refusal,
//! redirect refusal, artifact-derivation guard — is exercised black-box, with no GGUF and no node.
//!
//! These paths are also `--lib`-unit-covered, but the integration gate measures the black-box
//! surface separately, and the courier is a hub-side outbound-fetch component whose failure modes
//! are worth an integration-level assertion through its real `pub` API.

use higgs::diagnostic::HiggsError;
use higgs::node::release_courier::{
    artifact_url_from_manifest, asset_suffix, manifest_filename, parse_courier_url,
    per_node_manifest_url, release_version_from_base, resolve_push_params, sig_url_for,
};

// ── a local "release origin" serving the CI-shaped trio + deliberately-broken variants ──────────

const OK_MANIFEST: &str = r#"{"schema":1,"version":"1.2.3","commit":"abc","file":"higgs-art.tar.gz","target":"aarch64-apple-darwin","variant":"metal","sha256":"00"}"#;
// Valid shape but a DIFFERENT version — the fleet version-bind must refuse it.
const V990_MANIFEST: &str = r#"{"schema":1,"version":"9.9.0","commit":"abc","file":"higgs-art.tar.gz","target":"aarch64-apple-darwin","variant":"metal","sha256":"00"}"#;
const SIG_BODY: &str = "untrusted comment\nRWSig==\n";

async fn spawn_origin() -> u16 {
    let big = "x".repeat(70 * 1024); // > MAX_MANIFEST_BYTES (64 KiB)
    let app = axum::Router::new()
        .route(
            "/d/ok.manifest",
            axum::routing::get(|| async { OK_MANIFEST }),
        )
        .route(
            "/d/ok.manifest.minisig",
            axum::routing::get(|| async { SIG_BODY }),
        )
        .route(
            "/d/v990.manifest",
            axum::routing::get(|| async { V990_MANIFEST }),
        )
        .route(
            "/d/v990.manifest.minisig",
            axum::routing::get(|| async { SIG_BODY }),
        )
        // Well-formed JSON that is NOT a valid UpdateManifest (schema is a string).
        .route(
            "/d/bad.manifest",
            axum::routing::get(|| async { r#"{"schema":"NOPE"}"# }),
        )
        .route(
            "/d/bad.manifest.minisig",
            axum::routing::get(|| async { SIG_BODY }),
        )
        // Non-UTF-8 manifest bytes.
        .route(
            "/d/nonutf8.manifest",
            axum::routing::get(|| async { axum::body::Bytes::from_static(&[0xff, 0xfe, 0x00]) }),
        )
        .route(
            "/d/nonutf8.manifest.minisig",
            axum::routing::get(|| async { SIG_BODY }),
        )
        // Over-cap manifest body.
        .route(
            "/d/big.manifest",
            axum::routing::get(move || {
                let b = big.clone();
                async move { b }
            }),
        )
        .route(
            "/d/big.manifest.minisig",
            axum::routing::get(|| async { SIG_BODY }),
        )
        // A manifest that is fine, but whose SIG is over-cap (> MAX_SIG_BYTES 4 KiB).
        .route(
            "/d/bigsig.manifest",
            axum::routing::get(|| async { OK_MANIFEST }),
        )
        .route(
            "/d/bigsig.manifest.minisig",
            axum::routing::get(|| async { "s".repeat(5 * 1024) }),
        )
        // A 302 redirect — the courier follows none.
        .route(
            "/d/redir.manifest",
            axum::routing::get(|| async { axum::response::Redirect::temporary("/d/ok.manifest") }),
        );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    port
}

fn is_fetch(e: &HiggsError) -> bool {
    matches!(e, HiggsError::UpdateFetchFailed { .. })
}
fn is_manifest(e: &HiggsError) -> bool {
    matches!(e, HiggsError::UpdateManifestInvalid { .. })
}

// ── the async fetch/assemble path via the real pub entry point ──────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn resolve_fetches_and_assembles_from_a_local_origin() {
    let port = spawn_origin().await;
    let url = format!("http://127.0.0.1:{port}/d/ok.manifest");
    let params = resolve_push_params(&url, Some("1.2.3"))
        .await
        .expect("a valid manifest + sig assemble");
    // The artifact URL is the DIRECT same-origin sibling named by the manifest's `file`.
    let json = serde_json::to_value(&params).unwrap();
    assert_eq!(
        json["artifact_url"],
        format!("http://127.0.0.1:{port}/d/higgs-art.tar.gz")
    );
    assert_eq!(json["target_version"], "1.2.3");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn resolve_maps_every_origin_failure() {
    let port = spawn_origin().await;
    let at = |p: &str| format!("http://127.0.0.1:{port}{p}");

    // 404 (route not registered).
    let e = resolve_push_params(&at("/d/missing.manifest"), None)
        .await
        .unwrap_err();
    assert!(
        is_fetch(&e) && !e.to_string().contains("127.0.0.1"),
        "{e:?}"
    );

    // Over-cap manifest body.
    let e = resolve_push_params(&at("/d/big.manifest"), None)
        .await
        .unwrap_err();
    assert!(is_fetch(&e), "{e:?}");

    // Over-cap signature body.
    let e = resolve_push_params(&at("/d/bigsig.manifest"), None)
        .await
        .unwrap_err();
    assert!(is_fetch(&e), "{e:?}");

    // A redirect is refused (no redirects followed).
    let e = resolve_push_params(&at("/d/redir.manifest"), None)
        .await
        .unwrap_err();
    assert!(is_fetch(&e), "{e:?}");

    // Malformed manifest → a CONSTANT manifest error (no bytes echoed).
    let e = resolve_push_params(&at("/d/bad.manifest"), None)
        .await
        .unwrap_err();
    assert!(is_manifest(&e) && !e.to_string().contains("NOPE"), "{e:?}");

    // Non-UTF-8 manifest → a constant manifest error.
    let e = resolve_push_params(&at("/d/nonutf8.manifest"), None)
        .await
        .unwrap_err();
    assert!(is_manifest(&e), "{e:?}");

    // Valid signed manifest but a DIFFERENT version than requested → the fleet version-bind refuses.
    let e = resolve_push_params(&at("/d/v990.manifest"), Some("1.2.3"))
        .await
        .unwrap_err();
    assert!(is_manifest(&e), "{e:?}");
    // …and with no expected version (the single-node path), the SAME manifest resolves fine.
    assert!(resolve_push_params(&at("/d/v990.manifest"), None)
        .await
        .is_ok());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn resolve_refuses_ssrf_and_bad_shapes_before_any_fetch() {
    // Literal private / reserved IPs are vetted with NO DNS and refused.
    for bad in [
        "https://10.0.0.1/x.manifest",
        "https://169.254.1.1/x.manifest",
        "https://[fd00::1]/x.manifest",
    ] {
        let e = resolve_push_params(bad, None).await.unwrap_err();
        assert!(is_fetch(&e), "{bad}: {e:?}");
    }
    // The NAME `localhost` is resolved (→ loopback) and refused, not trusted as an on-box shortcut.
    assert!(resolve_push_params("https://localhost/x.manifest", None)
        .await
        .is_err());
    // Plaintext http to a non-loopback-IP host, userinfo, query, fragment, and a bad scheme are all
    // refused at the pure vet (before any network).
    for bad in [
        "http://example.com/x.manifest",
        "https://user:pass@ex.com/x.manifest",
        "https://ex.com/x.manifest?sig=1",
        "https://ex.com/x.manifest#f",
        "ftp://ex.com/x.manifest",
    ] {
        assert!(resolve_push_params(bad, None).await.is_err(), "{bad}");
    }
}

// ── the PURE resolvers (asset naming, URL derivation, guards) through the pub API ────────────────

#[test]
fn pure_resolvers_cover_naming_and_url_derivation() {
    // asset suffix + manifest filename mirror release.yml.
    assert_eq!(
        asset_suffix("aarch64-apple-darwin", "metal"),
        "aarch64-apple-darwin"
    );
    assert_eq!(
        asset_suffix("x86_64-unknown-linux-gnu", "cuda"),
        "x86_64-unknown-linux-gnu-cuda"
    );
    assert_eq!(
        manifest_filename("1.2.3", "aarch64-apple-darwin", "metal"),
        "higgs-v1.2.3-aarch64-apple-darwin.manifest"
    );

    // release version from a base tag (+ its rejections).
    let base = parse_courier_url("https://mirror.example/higgs/v1.2.3").unwrap();
    assert_eq!(release_version_from_base(&base).unwrap(), "1.2.3");
    assert!(release_version_from_base(
        &parse_courier_url("https://mirror.example/higgs/latest").unwrap()
    )
    .is_err());

    // per-node manifest URL is a child under the tag dir.
    let m = per_node_manifest_url(&base, "1.2.3", "aarch64-apple-darwin", "metal").unwrap();
    assert_eq!(
        m.as_str(),
        "https://mirror.example/higgs/v1.2.3/higgs-v1.2.3-aarch64-apple-darwin.manifest"
    );
    // its .minisig sibling.
    assert_eq!(
        sig_url_for(&m).unwrap().as_str(),
        "https://mirror.example/higgs/v1.2.3/higgs-v1.2.3-aarch64-apple-darwin.manifest.minisig"
    );

    // artifact-from-manifest: the happy bare sibling, and the traversal/re-root/decoration guards.
    assert_eq!(
        artifact_url_from_manifest(&m, "higgs.tar.gz")
            .unwrap()
            .as_str(),
        "https://mirror.example/higgs/v1.2.3/higgs.tar.gz"
    );
    for bad in [
        "../x",
        "a/b",
        "https:evil/x",
        "x?y",
        "x#y",
        "%2e%2e",
        ".hidden",
    ] {
        assert!(artifact_url_from_manifest(&m, bad).is_err(), "{bad}");
    }
}
