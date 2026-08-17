//! Black-box integration coverage for the self-update TRUST ANCHOR (`src/update.rs`) and the
//! hub-side release courier (`src/node/release_courier.rs`) — the branches the existing suites
//! (`courier_edges.rs` fetch edges, `remote_update_push.rs` happy-path push) leave unexercised:
//!
//!  - `update.rs` via its REAL pub API against the SHIPPED pin table (`HIGGS_UPDATE_PUBKEYS`):
//!    an unpinned key id fails closed (HG081), a malformed signature and a valid-format FOREIGN
//!    signature are both rejected (HG082) by name-lookup and try-all verification, and the
//!    artifact sha256 bind accepts case-insensitively / refuses a mismatch (HG084). A signature
//!    that VERIFIES under the compiled-in key cannot be minted by a test (that is the security
//!    property), so the post-verification parse paths stay unit-covered via the private seam.
//!  - `release_courier.rs` PURE resolvers: version-from-base rejections (no path, non-semver
//!    tag, a cannot-be-a-base URL), the underivable-artifact guard, the github.com-only
//!    releases-API shape, and the `newer_releases_for` filter (draft / incomplete-assets /
//!    build-metadata / older / duplicate entries).
//!  - `release_courier.rs` fleet flows over an IN-PROCESS hermetic iroh link (no spawned
//!    binary, no GGUF): unknown-node refusals, the version-only fleet trigger's per-node skip
//!    reasons, and `fleet_update`'s per-node error/skip report against a loopback origin —
//!    including the redaction contract (reports never echo the release host).
//!  - the chunked (no Content-Length) over-cap manifest body refusal in `fetch_bounded`.
//!
//! Everything is loopback/in-process; nothing here touches DNS, GitHub, or any real host.

use std::collections::HashMap;
use std::io::Cursor;
use std::sync::Arc;

use iroh::endpoint::Connection;
use iroh::Endpoint;
use serde_json::{json, Value};

use higgs::diagnostic::HiggsError;
use higgs::log_bus::LogBus;
use higgs::node::fleet::HubFleet;
use higgs::node::release_courier::{
    artifact_url_from_manifest, fleet_update, fleet_update_version, newer_releases_for,
    node_releases, node_update_version, parse_courier_url, release_version_from_base,
    releases_api_url, resolve_push_params,
};
use higgs::node::transport::NodeTransport;
use higgs::remote::ALPN;
use higgs::update::{
    verify_artifact_sha256, verify_artifact_sha256_hex, verify_manifest, verify_manifest_any,
    UpdateManifest, HIGGS_UPDATE_PUBKEYS, UPDATE_MANIFEST_SCHEMA,
};

/// A CI-shaped manifest body. The signature checks reject before parsing, so the content only
/// needs to be realistic, not signed.
const MANIFEST: &str = r#"{"schema":1,"version":"1.2.3","commit":"abc","file":"higgs-art.tar.gz","target":"aarch64-apple-darwin","variant":"metal","sha256":"00"}"#;

/// The well-known sha256 of `b"abc"` — an INDEPENDENT pin so the digest test does not trust the
/// code under test to hash correctly.
const SHA256_ABC: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";

fn sign_with_throwaway_key(bytes: &[u8]) -> String {
    // A freshly minted key produces a STRUCTURALLY VALID .minisig that can never verify under
    // the shipped pin — exactly the "foreign but well-formed" courier-tampering case.
    let minisign::KeyPair { pk: _, sk } =
        minisign::KeyPair::generate_unencrypted_keypair().expect("keygen");
    minisign::sign(None, &sk, Cursor::new(bytes), Some("test manifest"), None)
        .expect("sign")
        .into_string()
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// update.rs — the pub verification API against the SHIPPED pin table
// ─────────────────────────────────────────────────────────────────────────────────────────────

#[test]
fn shipped_pins_fail_closed_for_unknown_ids_and_foreign_signatures() {
    // The shipped table pins the real release key; that is WHY the success paths below are
    // untestable here (no test holds the CI secret key) and why every arm must refuse.
    assert!(
        HIGGS_UPDATE_PUBKEYS
            .iter()
            .any(|(id, _)| *id == "higgs-release-1"),
        "the release key must be pinned for these fail-closed checks to be meaningful"
    );

    let bytes = MANIFEST.as_bytes();

    // An id that is not pinned fails closed BEFORE any crypto (HG081).
    let e = verify_manifest(bytes, "irrelevant", "no-such-key").unwrap_err();
    assert!(
        matches!(&e, HiggsError::UpdateKeyUnknown { key_id } if key_id == "no-such-key"),
        "want HG081, got: {e}"
    );

    // Garbage signature text under the pinned id: rejected at signature decode (HG082).
    let e = verify_manifest(bytes, "not a minisig file at all", "higgs-release-1").unwrap_err();
    assert!(
        matches!(e, HiggsError::UpdateSignatureInvalid { .. }),
        "want HG082 for malformed signature text, got: {e}"
    );

    // A STRUCTURALLY VALID signature from a foreign key: decodes fine, then fails the actual
    // cryptographic verify under the pinned key (HG082) — the courier-swapped-the-sig case.
    let foreign_sig = sign_with_throwaway_key(bytes);
    let e = verify_manifest(bytes, &foreign_sig, "higgs-release-1").unwrap_err();
    assert!(
        matches!(e, HiggsError::UpdateSignatureInvalid { .. }),
        "want HG082 for a foreign signature, got: {e}"
    );

    // Try-all (the direct-download shape, no caller-named key): every pin is tried, none
    // verifies, and the aggregate error names how many pins refused — never which keys exist.
    let e = verify_manifest_any(bytes, &foreign_sig).unwrap_err();
    assert!(
        matches!(&e, HiggsError::UpdateSignatureInvalid { detail }
            if detail.contains("pinned release key")),
        "want the aggregate HG082, got: {e}"
    );
}

#[test]
fn artifact_sha256_bind_is_case_insensitive_and_reports_mismatches() {
    let mut manifest: UpdateManifest = serde_json::from_str(MANIFEST).expect("fixture parses");
    assert_eq!(manifest.schema, UPDATE_MANIFEST_SCHEMA);

    // Mismatch: HG084 carries the manifest's file + expected AND the ACTUAL computed digest —
    // that `got` value doubles as proof the production hasher/hex are correct (pinned against
    // the well-known sha256 of b"abc").
    let e = verify_artifact_sha256(&manifest, b"abc").unwrap_err();
    let HiggsError::UpdateArtifactMismatch {
        file,
        expected,
        got,
    } = e
    else {
        panic!("want HG084, got: {e}");
    };
    assert_eq!(file, "higgs-art.tar.gz");
    assert_eq!(expected, "00");
    assert_eq!(
        got, SHA256_ABC,
        "production sha256+hex must match the known digest"
    );

    // A matching pin verifies — and an UPPERCASE pin too (shasum emits lowercase, but a
    // hand-typed manifest pin may not; the compare is ascii-case-insensitive by contract).
    manifest.sha256 = SHA256_ABC.to_string();
    verify_artifact_sha256(&manifest, b"abc").expect("matching digest verifies");
    manifest.sha256 = SHA256_ABC.to_uppercase();
    verify_artifact_sha256(&manifest, b"abc").expect("uppercase pin verifies");

    // The streamed-download shape: the caller hashed incrementally and hands only the digest.
    verify_artifact_sha256_hex(&manifest, SHA256_ABC).expect("hex form verifies");
    let e = verify_artifact_sha256_hex(&manifest, "beef").unwrap_err();
    assert!(matches!(e, HiggsError::UpdateArtifactMismatch { .. }));
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// release_courier.rs — PURE resolvers (no network, no fleet)
// ─────────────────────────────────────────────────────────────────────────────────────────────

#[test]
fn release_version_from_base_rejects_bases_without_a_semver_tag() {
    // A cannot-be-a-base URL has NO path segments at all — the "no release dir" refusal.
    let no_path = reqwest::Url::parse("mailto:owner@example.com").expect("parses");
    let e = release_version_from_base(&no_path).unwrap_err();
    assert!(matches!(e, HiggsError::UpdateFetchFailed { .. }), "{e}");

    // A host-only base (every segment empty) is the same refusal.
    let root = parse_courier_url("https://mirror.example/").expect("vets");
    assert!(release_version_from_base(&root).is_err());

    // `v` with an INVALID semver after it: refused so a typo can't derive a wrong asset name.
    let bad_semver = parse_courier_url("https://mirror.example/higgs/v1.2.3.4").expect("vets");
    let e = release_version_from_base(&bad_semver).unwrap_err();
    assert!(
        matches!(&e, HiggsError::UpdateFetchFailed { detail } if detail.contains("semver")),
        "{e}"
    );

    // A bare `v` (empty version) is refused too.
    let bare_v = parse_courier_url("https://mirror.example/higgs/v").expect("vets");
    assert!(release_version_from_base(&bare_v).is_err());

    // A trailing slash after the tag dir still resolves (the last NON-EMPTY segment wins).
    let trailing = parse_courier_url("https://mirror.example/higgs/v1.2.3/").expect("vets");
    assert_eq!(release_version_from_base(&trailing).unwrap(), "1.2.3");
}

#[test]
fn artifact_derivation_refuses_a_manifest_url_it_cannot_join() {
    // A cannot-be-a-base manifest URL cannot take a sibling join — the guard must refuse with
    // the CONSTANT error (never echoing the unverified `file`).
    let weird = reqwest::Url::parse("mailto:owner@example.com").expect("parses");
    let e = artifact_url_from_manifest(&weird, "x.tar.gz").unwrap_err();
    assert!(
        matches!(&e, HiggsError::UpdateManifestInvalid { detail }
            if !detail.contains("x.tar.gz")),
        "constant error text, no reflected filename: {e}"
    );
}

#[test]
fn releases_api_url_accepts_only_the_github_releases_shape() {
    // The one accepted shape maps to the REST index with a bounded page size.
    let api = releases_api_url("https://github.com/AI-Experiri/higgs/releases").expect("derives");
    assert_eq!(
        api.as_str(),
        "https://api.github.com/repos/AI-Experiri/higgs/releases?per_page=100"
    );

    // A custom mirror has no index contract — the error tells the operator to push by URL.
    let e = releases_api_url("https://mirror.example/higgs/releases").unwrap_err();
    assert!(e.to_string().contains("github.com"), "{e}");

    // github.com but NOT <owner>/<repo>/releases — refused rather than guessed.
    for bad in [
        "https://github.com/AI-Experiri/higgs",
        "https://github.com/AI-Experiri/higgs/releases/download/v1.2.3",
    ] {
        assert!(releases_api_url(bad).is_err(), "{bad}");
    }

    // The courier URL vet runs FIRST: plaintext http to a public host never derives anything.
    assert!(releases_api_url("http://github.com/o/r/releases").is_err());
}

#[test]
fn newer_releases_listing_offers_only_complete_newer_matching_builds() {
    let full_assets = |ver: &str| {
        json!([
            { "name": format!("higgs-v{ver}-aarch64-apple-darwin.manifest") },
            { "name": format!("higgs-v{ver}-aarch64-apple-darwin.manifest.minisig") },
            { "name": format!("higgs-v{ver}-aarch64-apple-darwin.tar.gz") },
        ])
    };
    let index = json!([
        // Offerable: newer, non-draft, complete trio.
        { "tag_name": "v2.0.0", "draft": false, "assets": full_assets("2.0.0") },
        // Draft releases are never offered, however complete.
        { "tag_name": "v3.0.0", "draft": true, "assets": full_assets("3.0.0") },
        // Incomplete upload (no tarball): offering it would fail at fetch — refused up front.
        { "tag_name": "v1.5.0", "assets": [
            { "name": "higgs-v1.5.0-aarch64-apple-darwin.manifest" },
            { "name": "higgs-v1.5.0-aarch64-apple-darwin.manifest.minisig" },
        ] },
        // Build metadata has no semver precedence — "newer" is ill-defined, so refused.
        { "tag_name": "v2.5.0+nightly", "assets": full_assets("2.5.0+nightly") },
        // Older than current: an upgrade listing never offers a downgrade.
        { "tag_name": "v0.5.0", "assets": full_assets("0.5.0") },
        // Not a v<semver> tag at all.
        { "tag_name": "some-tag", "assets": full_assets("9.9.9") },
        // An entry with no assets key is skipped, not a panic.
        { "tag_name": "v2.2.0" },
        // A second offerable release, plus a duplicate of the first (deduped).
        { "tag_name": "v2.1.0", "assets": full_assets("2.1.0") },
        { "tag_name": "v2.0.0", "assets": full_assets("2.0.0") },
    ]);

    // Newest-first, deduped, only the complete matching builds.
    assert_eq!(
        newer_releases_for(&index, "1.0.0", "aarch64-apple-darwin", "metal"),
        vec!["2.1.0".to_string(), "2.0.0".to_string()]
    );

    // An unknown/unparseable CURRENT version never guesses an upgrade.
    assert!(
        newer_releases_for(&index, "not-a-version", "aarch64-apple-darwin", "metal").is_empty()
    );

    // A non-array index (e.g. a GitHub error object) yields the empty listing, not a panic.
    assert!(newer_releases_for(&json!({"message": "rate limited"}), "1.0.0", "t", "v").is_empty());

    // A different (target, variant) sees none of these assets.
    assert!(newer_releases_for(&index, "1.0.0", "x86_64-unknown-linux-gnu", "cuda").is_empty());
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// fleet flows over an in-process hermetic iroh link (no spawned binary, no GGUF)
// ─────────────────────────────────────────────────────────────────────────────────────────────

async fn minimal_ep() -> Endpoint {
    Endpoint::builder(iroh::endpoint::presets::Minimal)
        .alpns(vec![ALPN.to_vec()])
        .relay_mode(iroh::RelayMode::Disabled)
        .bind()
        .await
        .expect("bind endpoint")
}

/// One raw ALPN connection between two hermetic endpoints. The fleet paths under test never
/// reach a push RPC (they skip/error first), so the "node" end never has to answer — but the
/// endpoints + dialer connection must stay alive or the fleet's close-watcher evicts the nodes.
async fn connect_pair() -> (Connection, Connection, (Endpoint, Endpoint)) {
    let a = minimal_ep().await;
    let b = minimal_ep().await;
    let b_addr = b.addr();
    let accept = tokio::spawn(async move {
        let incoming = b.accept().await.expect("incoming");
        let conn = incoming.await.expect("accept conn");
        (conn, b)
    });
    let dialer = a.connect(b_addr, ALPN).await.expect("dial");
    let (acceptor, b_ep) = accept.await.expect("accept join");
    (dialer, acceptor, (a, b_ep))
}

/// Admit one fake node with a chosen identity shape — the same direct-admission seam
/// `remote_update_push.rs` uses (`admit_gen: None` bypasses the kill-switch gate by design
/// for direct/test callers).
#[allow(clippy::too_many_arguments)]
async fn admit(
    fleet: &Arc<HubFleet>,
    conn: &Connection,
    key: &str,
    target: Option<&str>,
    variant: Option<&str>,
    update_capable: bool,
    version_capable: bool,
    software_version: Option<&str>,
) {
    fleet
        .add_node_with_identity(
            key.to_string(),
            Arc::new(NodeTransport::new(conn.clone())),
            None,
            Some(2),
            software_version.map(str::to_string),
            false,
            None,
            false,
            target.map(str::to_string),
            variant.map(str::to_string),
            update_capable,
            version_capable,
            false,
            false,
            Vec::new(),
        )
        .await;
}

fn results_by_node(report: &Value) -> HashMap<String, Value> {
    report["results"]
        .as_array()
        .unwrap_or_else(|| panic!("results array in {report}"))
        .iter()
        .map(|r| (r["node"].as_str().expect("node key").to_string(), r.clone()))
        .collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn courier_ops_refuse_unknown_nodes_and_bad_versions_before_any_network() {
    let bus = Arc::new(LogBus::new());
    let fleet = Arc::new(HubFleet::new(bus));

    // A node that is not connected is a reachability error, for both flows.
    let e = node_releases(&fleet, "ghost", "https://github.com/o/r/releases")
        .await
        .unwrap_err();
    assert!(matches!(e, HiggsError::NodeUnreachable { .. }), "{e}");
    let e = node_update_version(&fleet, "ghost", "1.2.3")
        .await
        .unwrap_err();
    assert!(matches!(e, HiggsError::NodeUnreachable { .. }), "{e}");

    // A non-semver fleet trigger is refused hub-side before anything else.
    let e = fleet_update_version(&fleet, "not.semver.x", "https://github.com/o/r/releases")
        .await
        .unwrap_err();
    assert!(e.to_string().contains("semver"), "{e}");

    // An EMPTY fleet yields the empty report — and, with no pushable target, the release index
    // is never fetched (this call would otherwise need github.com and fail offline).
    let report = fleet_update_version(&fleet, "1.2.3", "https://github.com/o/r/releases")
        .await
        .expect("empty fleet reports empty");
    assert_eq!(report["results"], json!([]));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fleet_version_trigger_reports_each_unpushable_node_without_network() {
    let (_dialer, hub_conn, _eps) = connect_pair().await;
    let bus = Arc::new(LogBus::new());
    let fleet = Arc::new(HubFleet::new(bus));

    // Four shapes, NONE of which is pushable — so `fleet_update_version` must skip each with
    // its own reason and never fetch the release index (github.com is unreachable in tests;
    // reaching for it would fail the run, which is itself part of the assertion).
    let t = Some("aarch64-apple-darwin");
    let v = Some("metal");
    admit(
        &fleet,
        &hub_conn,
        "cap-high",
        t,
        v,
        true,
        true,
        Some("99.0.0"),
    )
    .await;
    admit(
        &fleet,
        &hub_conn,
        "cap-nover",
        t,
        v,
        true,
        true,
        Some("a-dev-build"),
    )
    .await;
    admit(
        &fleet,
        &hub_conn,
        "cap-nobuild",
        None,
        None,
        true,
        true,
        Some("0.0.1"),
    )
    .await;
    // A legacy admission (no build identity at all → not version-capable).
    fleet
        .add_node(
            "legacy".to_string(),
            Arc::new(NodeTransport::new(hub_conn.clone())),
            None,
            Some(2),
            Some("0.0.9".to_string()),
            false,
            None,
            false,
        )
        .await;

    let report = fleet_update_version(&fleet, "9.9.9", "https://github.com/o/r/releases")
        .await
        .expect("per-node skips never fail the fleet");
    let rows = results_by_node(&report);
    assert_eq!(rows.len(), 4, "{report}");
    for row in rows.values() {
        assert_eq!(row["status"], "skipped", "{row}");
    }
    // Each skip names ITS reason — the report is what the operator debugs from.
    assert!(
        rows["cap-high"]["reason"]
            .as_str()
            .unwrap()
            .contains("already at or above"),
        "{report}"
    );
    assert!(
        rows["cap-nover"]["reason"]
            .as_str()
            .unwrap()
            .contains("parseable version"),
        "{report}"
    );
    assert!(
        rows["cap-nobuild"]["reason"]
            .as_str()
            .unwrap()
            .contains("target/variant"),
        "{report}"
    );
    assert!(
        rows["legacy"]["reason"]
            .as_str()
            .unwrap()
            .contains("version-triggered"),
        "{report}"
    );

    // The LISTING flow for a node with an unknown build (or unparseable version) short-circuits
    // to the empty offering WITHOUT touching the network — the UI shows bootstrap state, not a
    // spurious fetch error.
    let listing = node_releases(&fleet, "cap-nobuild", "https://github.com/o/r/releases")
        .await
        .expect("unknown build lists empty offline");
    assert_eq!(listing["node"], "cap-nobuild");
    assert_eq!(listing["current"], "0.0.1");
    assert_eq!(listing["available"], json!([]));
    assert_eq!(listing["update_by_version"], true);
    let listing = node_releases(&fleet, "cap-nover", "https://github.com/o/r/releases")
        .await
        .expect("unparseable current lists empty offline");
    assert_eq!(listing["available"], json!([]));

    // A node WITH a known build + parseable version does try to list — and a non-github
    // release_url is refused by the pure shape check BEFORE any fetch.
    let e = node_releases(&fleet, "cap-high", "http://127.0.0.1:9/releases")
        .await
        .unwrap_err();
    assert!(e.to_string().contains("github.com"), "{e}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fleet_manifest_update_reports_fetch_failures_and_missing_builds_per_node() {
    // A loopback "release origin" that 404s everything: the per-node manifest fetch must fail
    // and be REPORTED for that node — never fail the whole fleet.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move { axum::serve(listener, axum::Router::new()).await.unwrap() });

    let (_dialer, hub_conn, _eps) = connect_pair().await;
    let bus = Arc::new(LogBus::new());
    let fleet = Arc::new(HubFleet::new(bus));

    let t = Some("aarch64-apple-darwin");
    let v = Some("metal");
    admit(&fleet, &hub_conn, "full", t, v, true, true, Some("0.0.1")).await;
    admit(
        &fleet,
        &hub_conn,
        "nobuild",
        None,
        None,
        true,
        true,
        Some("0.0.1"),
    )
    .await;
    fleet
        .add_node(
            "legacy".to_string(),
            Arc::new(NodeTransport::new(hub_conn.clone())),
            None,
            Some(2),
            Some("0.0.9".to_string()),
            false,
            None,
            false,
        )
        .await;

    let base = format!("http://127.0.0.1:{port}/rel/v9.9.9");
    let report = fleet_update(&fleet, &base)
        .await
        .expect("per-node failures never fail the fleet");
    let rows = results_by_node(&report);
    assert_eq!(rows.len(), 3, "{report}");

    // The identified node's fetch 404s → an ERROR row naming the numeric status but — the
    // redaction contract — NEVER the release host (a URL component can carry a capability and
    // the report may be read by a less-privileged viewer).
    assert_eq!(rows["full"]["status"], "error", "{report}");
    let err_text = rows["full"]["error"].as_str().unwrap();
    assert!(err_text.contains("HTTP 404"), "{err_text}");
    assert!(
        !err_text.contains("127.0.0.1") && !err_text.contains(&port.to_string()),
        "release host must be redacted from the per-node report: {err_text}"
    );

    // No build identity → the courier cannot pick an asset → skipped, not pushed-and-failed.
    assert_eq!(rows["nobuild"]["status"], "skipped");
    assert!(
        rows["nobuild"]["reason"]
            .as_str()
            .unwrap()
            .contains("target/variant"),
        "{report}"
    );
    // No `update` capability → skipped.
    assert_eq!(rows["legacy"]["status"], "skipped");
    assert!(
        rows["legacy"]["reason"]
            .as_str()
            .unwrap()
            .contains("update"),
        "{report}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// fetch_bounded: the CHUNKED (no Content-Length) over-cap refusal
// ─────────────────────────────────────────────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_chunked_manifest_body_over_the_cap_is_refused_mid_stream() {
    // 5 × 16 KiB chunks = 80 KiB, streamed with NO Content-Length: the header pre-check cannot
    // catch it, so the bounded chunk loop must refuse the moment the cap would be exceeded —
    // the "server lied / omitted the header" half of the OOM bound.
    fn drip() -> axum::response::Response {
        let chunks =
            (0..5).map(|_| Ok::<_, std::io::Error>(axum::body::Bytes::from(vec![b'x'; 16 * 1024])));
        axum::response::Response::builder()
            .body(axum::body::Body::from_stream(futures::stream::iter(chunks)))
            .expect("response builds")
    }

    let app = axum::Router::new()
        .route("/d/drip.manifest", axum::routing::get(|| async { drip() }))
        // A small well-formed manifest + sig on the SAME origin: the sized happy fetch (its
        // Content-Length is under the cap) pins that the cap refusal above is about SIZE, not
        // some origin-wide breakage.
        .route("/d/ok.manifest", axum::routing::get(|| async { MANIFEST }))
        .route(
            "/d/ok.manifest.minisig",
            axum::routing::get(|| async { "untrusted comment\nRWSig==\n" }),
        );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let e = resolve_push_params(&format!("http://127.0.0.1:{port}/d/drip.manifest"), None)
        .await
        .unwrap_err();
    assert!(
        matches!(&e, HiggsError::UpdateFetchFailed { detail }
            if detail.contains("exceeded") && detail.contains("65536")),
        "want the mid-stream cap refusal naming our own cap, got: {e}"
    );

    let params = resolve_push_params(&format!("http://127.0.0.1:{port}/d/ok.manifest"), None)
        .await
        .expect("the small sibling manifest still assembles");
    let json = serde_json::to_value(&params).unwrap();
    assert_eq!(
        json["artifact_url"],
        format!("http://127.0.0.1:{port}/d/higgs-art.tar.gz")
    );
}
