//! Hub-side release courier (DESIGN-remote.md §9, REL-P4e) — the SENDER half of the
//! signature-verified self-update push.
//!
//! The node's RECEIVE + apply side ([`crate::node::self_update`], `M_NODE_UPDATE`) is the
//! authority: it re-verifies the CI-signed manifest against the pubkeys COMPILED INTO the node,
//! re-hashes the artifact, and enforces eligibility (target/variant match, no silent downgrade).
//! This module is ONLY a courier — it cannot forge a binary. What it does:
//!
//! 1. resolve a per-node manifest URL (the CI `higgs-v<ver>-<suffix>` naming from `release.yml`),
//! 2. async-fetch the (tiny) manifest + its sibling `.minisig` — size-capped, time-bounded, NO
//!    redirects, SSRF-vetted (the hub now makes an OUTBOUND fetch),
//! 3. derive the DIRECT artifact URL as the sibling named by the manifest's `file`,
//! 4. assemble [`NodeUpdateParams`] and push them via [`HubFleet::push_update`].
//!
//! The courier PARSES the manifest only to read `file` + `version` (to name the artifact and set
//! the informational `target_version`); it does NOT verify the signature — it pins no key, and a
//! forged manifest is caught by the node's own verify. A hub push is UPGRADE-ONLY: there is no
//! downgrade knob on the wire and the node always refuses a downgrade, so a courier replaying an old
//! signed release can never downgrade a node. `fleet_update` additionally BINDS each fetched
//! manifest's `version` to the base URL's requested `v<ver>`, so a swapped-but-signed older manifest
//! at the expected path is refused rather than pushed.
//!
//! ## Release-source contract (IMPORTANT — not the GitHub release-download URL)
//! The operator must point the courier at a DIRECT STATIC HTTPS ORIGIN that serves the `.manifest`,
//! its `.minisig`, and the artifact `.tar.gz` as static SIBLING files with NO redirect and NO query
//! string. The base URL's last path segment is the `v<version>` release dir
//! (`https://mirror.example/higgs/v1.2.3/`). The GitHub `…/releases/download/v<ver>/…` URL does NOT
//! work: GitHub 302-REDIRECTS a release asset to storage (the courier follows no redirects — an SSRF
//! defence — and rejects a 3xx as HG088), and the final storage URL carries a QUERY (rejected by
//! [`parse_courier_url`], since the `.minisig`/artifact siblings are derived by editing the PATH).
//! Mirror the signed release assets to a static origin (or a CDN dir) and pass that.
//!
//! ## Trust asymmetry (deliberate, mirrors [`crate::node::self_update`])
//! - The MANIFEST url is OPERATOR-supplied to the hub (the hub admin typed it / a base URL), so it
//!   is trusted at the same level as the node's CLI `--url` path: `https` to any host, OR
//!   `http`/`https` to a LOOPBACK host. A non-loopback host is additionally RESOLVED and refused if
//!   any address is non-global ([`is_ssrf_prone_ip`]), with the client PINNED to the vetted
//!   addresses (DNS-rebind defence).
//! - The ARTIFACT url is what the hub hands the UNTRUSTED node, which re-vets it with the STRICTER
//!   https-only, global-only policy before it fetches — so the courier only has to derive it.
//! - LOOPBACK is a TEST affordance (the integration test serves the manifest from a `127.0.0.1`
//!   server, without a `cfg(test)` production knob). A loopback release URL is NOT applyable on a
//!   MANAGED node: the derived artifact URL is a same-origin loopback sibling, and the node re-vets
//!   it as https+global, so a managed node replies `accepted` then records HG088 — a VISIBLE failure
//!   surfaced on its next HELLO `update_failed`, not a silent one. A production operator passes their
//!   mirror's public https URL; a loopback push is operator error that fails loudly.
//!
//! ## SSRF residual — network-specific NAT64 (codex REL-P4e r4, accepted)
//! [`is_ssrf_prone_ip`] recognises the WELL-KNOWN NAT64 prefix (`64:ff9b::/96`) but not a
//! NETWORK-SPECIFIC Pref64 (RFC 7050): on an IPv6-only network whose NAT64 uses a locally-chosen
//! globally-routed prefix, a hostname that resolves ONLY to a synthetic global IPv6 mapping a private
//! IPv4 passes the global-only vet, so the courier's MANIFEST fetch could be steered at an on-net
//! private host. This is a property of the SHARED classifier (already a documented residual on the
//! node's `--url`/artifact path), and the courier's exposure is STRICTLY WEAKER: (1) the fetch is a
//! blind probe of a `.manifest`/`.minisig` only, size-capped + no-redirect; (2) the bytes are
//! UNTRUSTED — the node re-verifies the CI signature against compiled-in keys, so nothing fetched
//! here is trusted; (3) an `https` probe must complete TLS for the ATTACKER'S hostname, which a
//! random internal service won't. Detecting an arbitrary network's Pref64 needs RFC 7050
//! (`ipv4only.arpa`) discovery or independent A/AAAA inspection — a change to the shared
//! `is_ssrf_prone_ip`, tracked as a follow-up there, not courier-local churn.
//!
//! ## Testability
//! PURE resolution ([`asset_suffix`], [`manifest_filename`], [`release_version_from_base`],
//! [`per_node_manifest_url`], [`sig_url_for`], [`artifact_url_from_manifest`], [`parse_courier_url`])
//! is split from the async I/O so it unit-tests with NO network — the ≥90% unit gate covers it.
//! The fleet/facade push paths are covered by the `tests/remote_update_push.rs` integration test.

use reqwest::Url;
use serde_json::{json, Value};

use crate::diagnostic::HiggsError;
use crate::node::fleet::{HubFleet, NodeUpdateTarget, PinnedPush};
use crate::node::self_update::is_ssrf_prone_ip;
use crate::remote::NodeUpdateParams;
use crate::update::UpdateManifest;

/// A manifest JSON is tiny — cap the download so a hostile/broken server cannot stream unbounded
/// bytes before we parse it. Mirrors the node's `self_update::MAX_MANIFEST_BYTES` (kept a courier
/// -local const, not a widened `pub`, since it bounds a DISTINCT hub-side fetch).
const MAX_MANIFEST_BYTES: u64 = 64 * 1024;
/// A minisign signature is a couple of short base64 lines.
const MAX_SIG_BYTES: u64 = 4 * 1024;
const HTTP_CONNECT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(15);
/// Whole-request ceiling (async reqwest applies `timeout` to the entire request incl. body). The
/// manifest + sig are tiny, so a generous-but-bounded window catches a stalled/slow-drip server.
const HTTP_FETCH_DEADLINE: std::time::Duration = std::time::Duration::from_secs(60);
/// How many nodes `fleet_update` fetches + pushes to at once — bounded so a big fleet does not open
/// hundreds of concurrent outbound fetches / control streams. The per-node work is independent.
const FLEET_UPDATE_CONCURRENCY: usize = 4;

fn fetch_err(detail: String) -> HiggsError {
    HiggsError::UpdateFetchFailed { detail }
}
fn manifest_invalid(detail: String) -> HiggsError {
    HiggsError::UpdateManifestInvalid { detail }
}

/// A URL rendered SAFE for an error message: a CONSTANT placeholder naming no component. Every
/// component of an operator/release URL can carry a capability (a token in the path, a userinfo,
/// a subdomain), and a `fleet_update` report may be read by a less-privileged viewer — the
/// operator knows the URL they passed, so naming nothing loses no actionable context.
fn redact(_u: &Url) -> &'static str {
    "the release URL"
}

// ---------------------------------------------------------------------------
// PURE resolution — no network (the home for exhaustive unit tests)
// ---------------------------------------------------------------------------

/// The release-artifact suffix for a `(target, variant)`, mirroring the `table` in
/// `.github/workflows/release.yml`: the artifact name is `higgs-v<ver>-<suffix>`, where the suffix
/// is the Rust target triple for the SOLE/default variant of that triple (`metal` on macOS, `cpu`
/// on Linux), and `<triple>-<variant>` for a variant that ships a SEPARATE artifact for the same
/// triple (`cuda` → `x86_64-unknown-linux-gnu-cuda`). An unknown variant gets a distinct
/// `<triple>-<variant>` name too, so it 404s rather than silently selecting the wrong artifact.
pub fn asset_suffix(target: &str, variant: &str) -> String {
    match variant {
        "metal" | "cpu" => target.to_string(),
        other => format!("{target}-{other}"),
    }
}

/// The CI manifest file name for a release + build: `higgs-v<version>-<suffix>.manifest`.
pub fn manifest_filename(version: &str, target: &str, variant: &str) -> String {
    format!(
        "higgs-v{version}-{}.manifest",
        asset_suffix(target, variant)
    )
}

/// Extract the release VERSION from a `fleet_update` base URL, whose last non-empty path segment is
/// the `v<semver>` release dir (`https://mirror.example/higgs/v1.2.3`). Returns the plain semver
/// (`1.2.3`) — the form the asset name embeds (`higgs-v1.2.3-…`). Errors if the last segment is not
/// a `v<valid-semver>` tag, so a mistyped base can't silently produce a wrong asset name.
pub fn release_version_from_base(base: &Url) -> Result<String, HiggsError> {
    let segs: Vec<&str> = match base.path_segments() {
        Some(s) => s.collect(),
        None => Vec::new(),
    };
    let seg = segs.iter().rev().find(|s| !s.is_empty()).ok_or_else(|| {
        fetch_err(
            "the release base URL has no path segment — expected a …/v<version> release dir".into(),
        )
    })?;
    // Do NOT echo the segment/version: a path segment can carry a capability token and
    // this text is surfaced in the fleet report (see the `redact` note above).
    let ver = seg
        .strip_prefix('v')
        .filter(|v| !v.is_empty())
        .ok_or_else(|| {
            fetch_err(
                "the release base URL's last path segment is not a v<version> release tag".into(),
            )
        })?;
    semver::Version::parse(ver).map_err(|_| {
        fetch_err(
            "the release base URL's last path segment is not a valid v<semver> release tag".into(),
        )
    })?;
    Ok(ver.to_string())
}

/// Ensure `base` ends with `/` (so [`Url::join`] APPENDS `child` rather than replacing the last
/// path segment), then join the bare `child` file name onto it.
fn join_child(base: &Url, child: &str) -> Result<Url, HiggsError> {
    let mut b = base.clone();
    let path = b.path().to_string();
    if !path.ends_with('/') {
        b.set_path(&format!("{path}/"));
    }
    b.join(child)
        .map_err(|e| fetch_err(format!("cannot derive a release URL: {e}")))
}

/// The per-node manifest URL under `base` (`…/download/v<ver>/higgs-v<ver>-<suffix>.manifest`),
/// PURE over the already-vetted `base`. The async fetch re-vets the derived URL's host.
pub fn per_node_manifest_url(
    base: &Url,
    version: &str,
    target: &str,
    variant: &str,
) -> Result<Url, HiggsError> {
    join_child(base, &manifest_filename(version, target, variant))
}

/// The `.minisig` sibling of a manifest URL — appended to the PATH (never the whole URL string), so
/// it stays correct even if the URL grows a component. Mirrors the node's `HttpSource::manifest`.
pub fn sig_url_for(manifest_url: &Url) -> Result<Url, HiggsError> {
    let mut u = manifest_url.clone();
    u.set_path(&format!("{}.minisig", manifest_url.path()));
    Ok(u)
}

/// The parent-directory portion of a URL path — everything up to and INCLUDING the last `/`. Two
/// sibling files share this; used by [`artifact_url_from_manifest`] to prove the derived artifact
/// is a sibling of the manifest, not somewhere else on (or off) the origin.
fn parent_path(u: &Url) -> String {
    let p = u.path();
    match p.rfind('/') {
        Some(i) => p[..=i].to_string(),
        None => "/".to_string(),
    }
}

/// Derive the artifact URL as a SIBLING of the manifest URL named by the manifest's `file` field.
/// `file` must be a BARE filename — no `/`, `\`, `..`, `%` (percent-encoded traversal like `%2e%2e`),
/// `:` (a `scheme:`/`//host` re-root that `Url::join` would honour), `?`/`#` (a query/fragment that
/// would attach to the derived artifact URL), and no leading `.` — so it can only name a file in the
/// manifest's own directory, never traverse, re-root, or decorate the URL. Mirrors the node's
/// `artifact_url_from` (defence in depth: the node re-checks + re-vets this before fetching). After
/// `join`, the derived URL is additionally asserted to share the manifest's scheme + host + port and
/// its parent directory AND to carry no query/fragment — belt-and-braces against any input that slips
/// the character filter.
pub fn artifact_url_from_manifest(manifest_url: &Url, file: &str) -> Result<Url, HiggsError> {
    if file.is_empty()
        || file.contains('/')
        || file.contains('\\')
        || file.contains("..")
        || file.contains('%')
        || file.contains(':')
        || file.contains('?')
        || file.contains('#')
        || file.starts_with('.')
    {
        // Do NOT echo `file` — the manifest is UNVERIFIED here (the node checks the signature), so a
        // hostile release response could stuff a capability-bearing string into `file` that would
        // otherwise be reflected into the per-node fleet report a less-privileged reader can see.
        return Err(manifest_invalid(
            "manifest `file` is not a bare sibling filename — refusing to derive an artifact URL"
                .into(),
        ));
    }
    // `Url::join(bare)` replaces the manifest URL's last path segment with `file`. The error is
    // CONSTANT — never echo the unverified `file` (same redaction reason as the rejections above).
    let artifact = manifest_url.join(file).map_err(|_| {
        manifest_invalid("manifest `file` does not derive a valid artifact URL".into())
    })?;
    // Defence in depth: the derived URL MUST be a same-origin sibling (same scheme/host/port + the
    // same parent directory) with NO query/fragment. Anything else means `file` re-rooted or
    // decorated the URL despite the filter.
    if artifact.scheme() != manifest_url.scheme()
        || artifact.host_str() != manifest_url.host_str()
        || artifact.port_or_known_default() != manifest_url.port_or_known_default()
        || parent_path(&artifact) != parent_path(manifest_url)
        || artifact.query().is_some()
        || artifact.fragment().is_some()
    {
        // Constant text (see above): never reflect the unverified `file` into an error/report.
        return Err(manifest_invalid(
            "manifest `file` does not resolve to a same-origin sibling of the manifest — refusing \
             to derive an artifact URL"
                .into(),
        ));
    }
    Ok(artifact)
}

/// True iff `u`'s host is a LITERAL loopback IP (`127.0.0.0/8` or `::1`) — deliberately NOT the
/// NAME `localhost`. This is the courier's ONLY on-box/test affordance: a numeric loopback address
/// is trusted to stay on the box without a resolve step. The node's own `--url` path treats the
/// name `localhost` as loopback too, but the courier must NOT — NSS / `/etc/hosts` / DNS can map
/// `localhost` off-box, so a name (even `localhost`) goes through the resolver + [`is_ssrf_prone_ip`]
/// vet like any other domain, where a loopback (or any non-global) resolution is then refused. That
/// closes the "an allowed `http://localhost` leaves the machine because localhost was remapped" hole.
fn is_literal_loopback_url(u: &Url) -> bool {
    match u.host() {
        Some(url::Host::Ipv4(ip)) => ip.is_loopback(),
        Some(url::Host::Ipv6(ip)) => ip.is_loopback(),
        _ => false,
    }
}

/// PURE parse + SCHEME/host-shape vet of an operator-supplied release URL (manifest or base). No
/// DNS — the resolved-IP SSRF check is the async [`build_manifest_client`]. Policy: `https` to any
/// host, or `http` to a LITERAL loopback IP (on-box mirror / test server; the NAME `localhost` is
/// NOT a loopback shortcut — it must resolve + pass the SSRF vet); REFUSE a userinfo
/// (`user:pass@` — could leak on a redirect) or a query/fragment (the sibling `.minisig`/artifact
/// URLs are derived by editing the PATH, which a query/fragment would corrupt; a presigned URL does
/// not fit the sibling model). Error text carries only the [`redact`]ed placeholder.
pub fn parse_courier_url(raw: &str) -> Result<Url, HiggsError> {
    // A `url::ParseError` does not echo the input, so it is safe; `raw` itself is NOT included (it
    // may be a mistyped secret-bearing URL).
    let u = Url::parse(raw).map_err(|e| fetch_err(format!("invalid release URL: {e}")))?;
    if !u.username().is_empty() || u.password().is_some() {
        return Err(fetch_err(format!(
            "the release URL {} must not embed credentials (they could leak on a redirect)",
            redact(&u)
        )));
    }
    if u.query().is_some() || u.fragment().is_some() {
        return Err(fetch_err(format!(
            "the release URL {} must not have a query or fragment — the .minisig + artifact URLs \
             are derived from its path",
            redact(&u)
        )));
    }
    match u.scheme() {
        "https" => Ok(u),
        "http" if is_literal_loopback_url(&u) => Ok(u),
        "http" => Err(fetch_err(format!(
            "refusing a plaintext http:// release URL to a non-loopback-IP host ({}) — use https \
             (the name `localhost` is not a loopback shortcut here; use a literal 127.0.0.1/::1)",
            redact(&u)
        ))),
        _ => Err(fetch_err(format!(
            "the release URL {} has an unsupported scheme — only https (or http to a loopback \
             host) is allowed",
            redact(&u)
        ))),
    }
}

// ---------------------------------------------------------------------------
// Async I/O — SSRF-vetted fetch + assemble + push
// ---------------------------------------------------------------------------

/// Build the async client for fetching `url` (already scheme-vetted by [`parse_courier_url`]).
/// Loopback → a plain no-proxy client (an on-box/test fetch never leaves the machine). A
/// non-loopback host is RESOLVED and refused if any address is non-global ([`is_ssrf_prone_ip`]);
/// a DOMAIN host is then PINNED to the vetted addresses (`resolve_to_addrs`) so the connect cannot
/// rebind to a private IP between this check and the fetch. `no_proxy` so a configured proxy can't
/// bypass the pin. No redirects — a 3xx surfaces as a clear error.
async fn build_manifest_client(url: &Url) -> Result<reqwest::Client, HiggsError> {
    let builder = reqwest::Client::builder()
        .connect_timeout(HTTP_CONNECT_TIMEOUT)
        .timeout(HTTP_FETCH_DEADLINE)
        .redirect(reqwest::redirect::Policy::none())
        // GitHub's REST API rejects requests without a User-Agent (403) — name
        // ourselves for every courier fetch (harmless to static mirrors).
        .user_agent(concat!("higgs/", env!("CARGO_PKG_VERSION")))
        .no_proxy();
    if is_literal_loopback_url(url) {
        return builder
            .build()
            .map_err(|e| fetch_err(format!("cannot build the HTTP client: {e}")));
    }
    // Use the TYPED host: a literal IP connects directly (no DNS, so only VET it); only a DOMAIN
    // goes through the resolver + gets pinned.
    let vet = |ip: std::net::IpAddr| -> Result<(), HiggsError> {
        if is_ssrf_prone_ip(&ip) {
            return Err(fetch_err(format!(
                "the release URL ({}) resolves to a private/loopback/link-local/reserved address \
                 — refusing (SSRF guard)",
                redact(url)
            )));
        }
        Ok(())
    };
    let builder = match url.host() {
        Some(url::Host::Ipv4(ip)) => {
            vet(ip.into())?;
            builder
        }
        Some(url::Host::Ipv6(ip)) => {
            vet(ip.into())?;
            builder
        }
        Some(url::Host::Domain(d)) => {
            let port = url.port_or_known_default().unwrap_or(443);
            // Bound the resolver explicitly: it runs BEFORE the client exists, so the client's
            // connect/request timeouts don't cover it — a wedged DNS server would otherwise pin a
            // fleet-update slot indefinitely.
            let addrs: Vec<std::net::SocketAddr> =
                tokio::time::timeout(HTTP_CONNECT_TIMEOUT, tokio::net::lookup_host((d, port)))
                    .await
                    .map_err(|_| fetch_err("resolving the release host timed out".into()))?
                    .map_err(|e| fetch_err(format!("cannot resolve the release host: {e}")))?
                    .collect();
            if addrs.is_empty() {
                return Err(fetch_err(
                    "the release host did not resolve to any address".into(),
                ));
            }
            for a in &addrs {
                vet(a.ip())?;
            }
            builder.resolve_to_addrs(d, &addrs)
        }
        None => {
            return Err(fetch_err(format!(
                "the release URL ({}) has no host",
                redact(url)
            )))
        }
    };
    builder
        .build()
        .map_err(|e| fetch_err(format!("cannot build the HTTP client: {e}")))
}

/// GET `url` with `client`, erroring on a non-2xx status, an over-cap `Content-Length`, or a body
/// that exceeds `cap` bytes even if the header lied/was absent (read in bounded chunks so nothing
/// over `cap` is ever buffered). The whole-request deadline is the client's `timeout`.
async fn fetch_bounded(
    client: &reqwest::Client,
    url: &Url,
    cap: u64,
    what: &str,
) -> Result<Vec<u8>, HiggsError> {
    let loc = redact(url);
    // reqwest embeds the URL in its own errors — strip it with `.without_url()` before formatting.
    let mut resp = client
        .get(url.clone())
        .send()
        .await
        .map_err(|e| fetch_err(format!("GET {what} {loc}: {}", e.without_url())))?;
    let status = resp.status();
    if !status.is_success() {
        if status.is_redirection() {
            return Err(fetch_err(format!(
                "{what} {loc} redirected (HTTP {}); redirects are not followed — pass the final, \
                 direct release URL",
                status.as_u16()
            )));
        }
        // Format ONLY the numeric status — the reason phrase is server-controlled.
        return Err(fetch_err(format!(
            "GET {what} {loc}: HTTP {}",
            status.as_u16()
        )));
    }
    if let Some(len) = resp.content_length() {
        if len > cap {
            // Do NOT echo `len` (server-controlled); only our own `cap` is named.
            return Err(fetch_err(format!(
                "{what} at {loc} exceeds the {cap}-byte cap"
            )));
        }
    }
    let mut buf: Vec<u8> = Vec::new();
    while let Some(chunk) = resp
        .chunk()
        .await
        .map_err(|e| fetch_err(format!("reading {what} {loc}: {}", e.without_url())))?
    {
        if buf.len() as u64 + chunk.len() as u64 > cap {
            return Err(fetch_err(format!(
                "{what} at {loc} exceeded the {cap}-byte cap"
            )));
        }
        buf.extend_from_slice(&chunk);
    }
    Ok(buf)
}

/// Fetch and assemble the [`NodeUpdateParams`] for one manifest URL: vet the URL, fetch the
/// manifest and its `.minisig` (both size-capped, no redirects, SSRF-vetted), PARSE the manifest
/// (untrusted — only to read `file` and `version`; the node re-verifies the signature), and derive
/// the DIRECT artifact URL as the sibling named by `file`.
///
/// `expected_version` BINDS the fetched manifest to the version the operator requested: `fleet_update`
/// derives the manifest URL from a base whose tag says `v<ver>`, and passes `Some("<ver>")` so a
/// compromised/misconfigured release origin serving a validly-signed OLDER/DIFFERENT manifest at the
/// expected path is REFUSED here rather than pushed. `node_update` (an EXACT operator URL) passes
/// `None` — there is no separate requested version to bind against. The node's own signature +
/// eligibility checks are the authority regardless; this is a hub-side courier sanity bind.
pub async fn resolve_push_params(
    manifest_url: &str,
    expected_version: Option<&str>,
) -> Result<NodeUpdateParams, HiggsError> {
    let url = parse_courier_url(manifest_url)?;
    let client = build_manifest_client(&url).await?;
    fetch_and_assemble(&client, &url, expected_version).await
}

/// Fetch (with an ALREADY-built, SSRF-vetted + origin-pinned `client`) + assemble the params for one
/// manifest `url`. Split from [`resolve_push_params`] so `fleet_update` builds the client — which
/// RESOLVES + vets + pins the origin — ONCE for the whole fleet and reuses it across every per-node
/// fetch: all per-node manifests are siblings under the SAME host, so a per-node client would
/// re-resolve it N times — wasteful, and (because `tokio`'s `getaddrinfo` runs on a NON-cancellable
/// blocking thread that the 15s timeout cannot kill) a way for a wedged resolver to leak one detached
/// lookup per node and saturate the blocking pool on a big fleet. `url` MUST be a shape-vetted
/// sibling of the `client`'s pinned origin (same host) — the callers pass a `parse_courier_url`-vetted
/// URL, and the pin means only that origin is reachable regardless.
async fn fetch_and_assemble(
    client: &reqwest::Client,
    url: &Url,
    expected_version: Option<&str>,
) -> Result<NodeUpdateParams, HiggsError> {
    let manifest_bytes = fetch_bounded(client, url, MAX_MANIFEST_BYTES, "manifest").await?;
    let sig_url = sig_url_for(url)?;
    let sig_bytes = fetch_bounded(client, &sig_url, MAX_SIG_BYTES, "signature").await?;

    // Keep the manifest bytes VERBATIM as the wire string — the node verifies the signature over
    // these exact bytes, so any re-serialization would break the check. `from_utf8` is lossless.
    // CONSTANT error text throughout: the manifest is UNVERIFIED (the node checks the signature), so
    // a hostile release response could stuff a capability-bearing string into a field, and the
    // per-node fleet report a less-privileged reader sees is built from these errors. Never reflect
    // ANY server-controlled value — not the bytes, and not the serde error's line/column either (a
    // hostile origin can position invalid JSON at chosen coordinates to encode a numeric token in
    // `(line, column)`), so the parse error is fully CONSTANT.
    let manifest_text = String::from_utf8(manifest_bytes)
        .map_err(|_| manifest_invalid("the fetched manifest is not UTF-8".into()))?;
    let sig_text = String::from_utf8(sig_bytes)
        .map_err(|_| manifest_invalid("the fetched signature is not UTF-8".into()))?;
    // Parse a COPY only to read `file` + `version`; authenticity is the node's job (it pins the
    // key). A malformed manifest is caught here so `fleet_update` reports it per node.
    let m: UpdateManifest = serde_json::from_str(&manifest_text).map_err(|_| {
        manifest_invalid("the fetched manifest is not a valid update manifest".into())
    })?;
    // Bind the fetched manifest to the operator-requested version, when there is one (fleet_update).
    // A signed-but-DIFFERENT manifest at the expected path (an origin serving an older release) is
    // refused before it can be pushed.
    if let Some(expected) = expected_version {
        if m.version != expected {
            // Do NOT echo either version: a base-URL path segment can carry a capability
            // (`v1.2.3+SecretToken`) and this message is copied into the fleet report a
            // less-privileged viewer may read. The operator knows the URL they passed.
            return Err(manifest_invalid(
                "the fetched manifest's version does not match the requested release — \
                 refusing to push a mismatched release"
                    .to_owned(),
            ));
        }
    }
    let artifact_url = artifact_url_from_manifest(url, &m.file)?;

    Ok(NodeUpdateParams {
        manifest: manifest_text,
        manifest_sig: sig_text,
        artifact_url: artifact_url.to_string(),
        // Informational (the manifest the node re-verifies is authoritative).
        target_version: Some(m.version),
        // The courier names no key; the node tries all its pinned keys.
        pinned_key_id: None,
    })
}

// ---------------------------------------------------------------------------
// Version-choosing flow (UPx): list newer releases + version-only pushes
// ---------------------------------------------------------------------------

/// A releases-index JSON is bounded (100 entries of metadata) — cap it so a hostile/broken
/// origin cannot stream unbounded bytes.
const MAX_RELEASES_INDEX_BYTES: u64 = 2 * 1024 * 1024;

/// Derive the GitHub RELEASES-API index URL from a configured `release_url` of the shape
/// `https://github.com/<owner>/<repo>/releases`. Listing is a GitHub-shape feature: a custom
/// static mirror has no index contract, so a non-GitHub `release_url` gets a clear error
/// telling the operator to use the exact-URL push instead. PURE (unit-testable).
pub fn releases_api_url(release_url: &str) -> Result<Url, HiggsError> {
    let u = parse_courier_url(release_url)?;
    if u.host_str() != Some("github.com") {
        return Err(fetch_err(
            "listing releases is only supported for a github.com release_url — for a custom \
             mirror, push with an exact manifest/base URL instead"
                .into(),
        ));
    }
    let segs: Vec<&str> = u
        .path_segments()
        .map(|s| s.filter(|p| !p.is_empty()).collect())
        .unwrap_or_default();
    match segs.as_slice() {
        [owner, repo, "releases"] => Url::parse(&format!(
            "https://api.github.com/repos/{owner}/{repo}/releases?per_page=100"
        ))
        .map_err(|e| fetch_err(format!("cannot derive the releases API URL: {e}"))),
        _ => Err(fetch_err(
            "the release_url is not of the form https://github.com/<owner>/<repo>/releases — \
             cannot list releases from it"
                .into(),
        )),
    }
}

/// PURE filter over a fetched releases index: every non-draft release whose tag is
/// `v<semver>` STRICTLY NEWER than `current` and whose assets include the manifest for
/// `(target, variant)` — i.e. versions this exact node can actually be updated to.
/// Sorted newest-first. `current` unparseable/absent → empty (never guess an upgrade for a
/// node whose version is unknown). The index is UNTRUSTED text: only syntactically-valid
/// semver tags and exact asset-name matches survive, and nothing else is echoed anywhere.
pub fn newer_releases_for(
    index: &Value,
    current: &str,
    target: &str,
    variant: &str,
) -> Vec<String> {
    let Ok(cur) = semver::Version::parse(current) else {
        return Vec::new();
    };
    let Some(entries) = index.as_array() else {
        return Vec::new();
    };
    let mut out: Vec<semver::Version> = entries
        .iter()
        .filter(|e| e.get("draft").and_then(Value::as_bool) != Some(true))
        .filter_map(|e| {
            let tag = e.get("tag_name")?.as_str()?;
            let ver = semver::Version::parse(tag.strip_prefix('v')?).ok()?;
            // Build metadata has NO precedence under semver, so "newer" is
            // ill-defined across it — and CI never publishes +metadata tags.
            // Refuse to offer them rather than risk an offered-but-refused push.
            if !ver.build.is_empty() || ver <= cur {
                return None;
            }
            // The node fetches manifest + .minisig + tarball — offer only releases
            // that published the COMPLETE trio for this build (a partial upload
            // would be offered and then fail at fetch).
            let base = format!("higgs-v{}-{}", ver, asset_suffix(target, variant));
            let names: Vec<&str> = e
                .get("assets")?
                .as_array()?
                .iter()
                .filter_map(|a| a.get("name").and_then(Value::as_str))
                .collect();
            let complete = [
                format!("{base}.manifest"),
                format!("{base}.manifest.minisig"),
                format!("{base}.tar.gz"),
            ]
            .iter()
            .all(|w| names.contains(&w.as_str()));
            complete.then_some(ver)
        })
        .collect();
    out.sort();
    out.dedup();
    out.reverse();
    out.into_iter().map(|v| v.to_string()).collect()
}

/// The version-listing flow behind the UI's Update button — ON DEMAND ONLY (this is the
/// only network touch, and it happens when the operator clicks, never on a timer): fetch
/// the releases index from the hub's configured `release_url` and return, for `node`,
/// `{ current, available: [newer-first…], update_by_version }`. `available` is empty when
/// the node is already newest (the UI says so), when its version/target/variant are
/// unknown, or when nothing matching its build exists.
pub async fn node_releases(
    fleet: &HubFleet,
    node: &str,
    release_url: &str,
) -> Result<Value, HiggsError> {
    let t = fleet
        .update_targets()
        .await
        .into_iter()
        .find(|t| t.node == node)
        .ok_or_else(|| HiggsError::NodeUnreachable {
            endpoint_id: node.to_string(),
            detail: "node is not connected".into(),
        })?;
    let current = t.software_version.clone().unwrap_or_default();
    // A node whose version/build is unknown can never be offered anything —
    // return the empty listing WITHOUT touching the network (the UI then shows
    // its bootstrap/up-to-date state instead of a spurious fetch error).
    let available = match (t.target.as_deref(), t.variant.as_deref()) {
        // Parseable-version gate matches newer_releases_for's own rule — an
        // unparseable current would filter to empty anyway, so never fetch for it.
        (Some(target), Some(variant)) if semver::Version::parse(&current).is_ok() => {
            let api = releases_api_url(release_url)?;
            let client = build_manifest_client(&api).await?;
            let body =
                fetch_bounded(&client, &api, MAX_RELEASES_INDEX_BYTES, "releases index").await?;
            // CONSTANT error text: the index is an UNTRUSTED body (courier-wide
            // redaction rules).
            let index: Value = serde_json::from_slice(&body)
                .map_err(|_| fetch_err("the releases index is not valid JSON".into()))?;
            newer_releases_for(&index, &current, target, variant)
        }
        _ => Vec::new(),
    };
    // NOTE deliberately NO bootstrap URL here: the listing only succeeds from the
    // GitHub releases API, and a GitHub `…/releases/download/…` asset URL 302s to
    // storage — which the manifest-push path refuses (no-redirect SSRF defence) —
    // so a prefilled URL would predictably fail. The UI instead tells a legacy
    // node's operator the honest bootstrap: re-run the installer on that machine
    // (or paste a DIRECT static-mirror manifest URL).
    Ok(json!({
        "node": node,
        "current": current,
        "available": available,
        "update_by_version": t.version_capable,
    }))
}

/// Push the version-only update trigger to ONE node: the hub sends `version` (validated
/// semver) and NOTHING else — the node self-fetches from its OWN configured `release_url`
/// and re-verifies everything. Refused client-side for a node that did not advertise
/// `update_by_version` (it must be updated once by installer/manifest-push first).
pub async fn node_update_version(
    fleet: &HubFleet,
    node: &str,
    version: &str,
) -> Result<Value, HiggsError> {
    semver::Version::parse(version).map_err(|_| {
        fetch_err("the requested update version is not a plain semver string".into())
    })?;
    let t = fleet
        .update_targets()
        .await
        .into_iter()
        .find(|t| t.node == node)
        .ok_or_else(|| HiggsError::NodeUnreachable {
            endpoint_id: node.to_string(),
            detail: "node is not connected".into(),
        })?;
    if !t.version_capable {
        return Err(fetch_err(
            "node does not support version-triggered updates — update it once via the \
             installer (or an exact manifest-URL push); every later update is one click"
                .into(),
        ));
    }
    match fleet
        .push_update_version_pinned(node, &t.transport, version)
        .await?
    {
        PinnedPush::Accepted(reply) => Ok(reply),
        // Same classification as a mid-op transport loss elsewhere (HG027): the
        // node moved between the snapshot and the send — a retryable reachability
        // condition, not a fetch failure.
        PinnedPush::Reconnected => Err(HiggsError::NodeUnreachable {
            endpoint_id: node.to_string(),
            detail: "node reconnected during the update — retry".into(),
        }),
    }
}

/// Push the version-only trigger to EVERY connected node that can take it, collecting the
/// same per-node `{node, status, …}` report `fleet_update` produces. A node without the
/// capability (or without a reported version) is SKIPPED with the reason; one node's
/// failure never fails the fleet. A node already AT or ABOVE `version` is skipped
/// hub-side (the node would refuse the non-upgrade anyway — skipping keeps the report
/// honest instead of "accepted-then-failed").
pub async fn fleet_update_version(
    fleet: &HubFleet,
    version: &str,
    release_url: &str,
) -> Result<Value, HiggsError> {
    use futures::stream::{self, StreamExt};
    let want = semver::Version::parse(version).map_err(|_| {
        fetch_err("the requested update version is not a plain semver string".into())
    })?;
    let targets = fleet.update_targets().await;
    // Fetch the release index ONCE and verify per-node asset availability BEFORE
    // pushing: a node whose build has no complete asset trio for `version` is
    // SKIPPED here rather than ACKed-then-failed (the fleet report stays honest).
    // Fetched only when a capable target exists — with none, every row is a
    // capability-skip and no network is touched.
    let index: Value = if targets.iter().any(|t| {
        // Fetch only when some node could actually be PUSHED: capable, with a
        // known build, and currently BELOW the requested version — otherwise
        // every row is a local skip and an unreachable index must not fail them.
        t.version_capable
            && t.target.is_some()
            && t.variant.is_some()
            && t.software_version
                .as_deref()
                .and_then(|v| semver::Version::parse(v).ok())
                .is_some_and(|c| c < want)
    }) {
        let api = releases_api_url(release_url)?;
        let client = build_manifest_client(&api).await?;
        let body = fetch_bounded(&client, &api, MAX_RELEASES_INDEX_BYTES, "releases index").await?;
        serde_json::from_slice(&body)
            .map_err(|_| fetch_err("the releases index is not valid JSON".into()))?
    } else {
        Value::Null
    };
    let results: Vec<Value> = stream::iter(targets)
        .map(|t| {
            let want = want.clone();
            let index = &index;
            async move {
                if !t.version_capable {
                    return skipped(&t.node, "node does not support version-triggered updates");
                }
                let Some(cur) = t
                    .software_version
                    .as_deref()
                    .and_then(|v| semver::Version::parse(v).ok())
                else {
                    return skipped(&t.node, "node did not report a parseable version");
                };
                if cur >= want {
                    return skipped(&t.node, "node is already at or above the requested version");
                }
                match (t.target.as_deref(), t.variant.as_deref()) {
                    (Some(target), Some(variant)) => {
                        // Availability check via the same PURE filter the listing uses:
                        // `version` must be among this build's offerable releases.
                        let offered = newer_releases_for(index, &cur.to_string(), target, variant);
                        if !offered.iter().any(|v| v == &want.to_string()) {
                            return skipped(
                                &t.node,
                                "the requested release ships no complete asset set for this \
                                 node's build",
                            );
                        }
                    }
                    _ => return skipped(&t.node, "node did not report its build target/variant"),
                }
                match fleet
                    .push_update_version_pinned(&t.node, &t.transport, &want.to_string())
                    .await
                {
                    Ok(PinnedPush::Accepted(reply)) => {
                        json!({ "node": t.node, "status": "accepted", "reply": reply })
                    }
                    Ok(PinnedPush::Reconnected) => {
                        skipped(&t.node, "node reconnected during the update")
                    }
                    Err(e) => errored(&t.node, &e),
                }
            }
        })
        .buffer_unordered(FLEET_UPDATE_CONCURRENCY)
        .collect()
        .await;
    Ok(json!({ "results": results }))
}

// ---------------------------------------------------------------------------
// Facade entry points (thin — `Higgs::node_update` / `Higgs::fleet_update` delegate here)
// ---------------------------------------------------------------------------

/// Push a self-update to ONE node from an EXACT operator-supplied manifest URL: resolve the params
/// (fetch + assemble) then [`HubFleet::push_update`]. Returns the node's `{status, target_version}`
/// reply (a receipt of the PUSH; the update's real outcome is the node's next HELLO). The push is
/// UPGRADE-ONLY (the node always refuses a downgrade); rollback is the node's own local job.
pub async fn node_update(
    fleet: &HubFleet,
    node: &str,
    manifest_url: &str,
) -> Result<Value, HiggsError> {
    // Exact operator URL — no separate requested version to bind against (`None`).
    let params = resolve_push_params(manifest_url, None).await?;
    fleet.push_update(node, params).await
}

/// Push a self-update to EVERY connected, update-capable node, each from its OWN per-(target,
/// variant) release asset under `release_base_url` (the CI `higgs-v<ver>-<suffix>` naming). The
/// slow per-node fetch + RPC run OFF the FleetActor with bounded concurrency; a per-node failure is
/// REPORTED, never fatal to the fleet. Returns `{ "results": [ {node, status, …}, … ] }`.
pub async fn fleet_update(fleet: &HubFleet, release_base_url: &str) -> Result<Value, HiggsError> {
    use futures::stream::{self, StreamExt};

    // Vet the base + derive the release version ONCE, up front — a bad base is a single systemic
    // error, not a per-node one.
    let base = parse_courier_url(release_base_url)?;
    let version = release_version_from_base(&base)?;
    // Resolve + vet + PIN the release ORIGIN ONCE and share the client across every per-node fetch:
    // every per-node manifest is a sibling under this same host, so a per-node client would re-resolve
    // it N times (wasteful) and — since `getaddrinfo` is non-cancellable — a wedged resolver would
    // leak one detached lookup per node. A base host that fails to resolve/vet is a single systemic
    // error (every node would fail it identically), so it aborts the whole run rather than N times.
    let client = build_manifest_client(&base).await?;
    let targets = fleet.update_targets().await;

    let results: Vec<Value> = stream::iter(targets)
        .map(|t| {
            let base = &base;
            let version = version.as_str();
            let client = &client;
            async move { per_node_update(fleet, client, base, version, t).await }
        })
        .buffer_unordered(FLEET_UPDATE_CONCURRENCY)
        .collect()
        .await;
    Ok(json!({ "results": results }))
}

/// Resolve + push for ONE fleet node, folding every outcome into a per-node report object. A node
/// that did not advertise `update`, or that never reported its build target/variant, is SKIPPED
/// (the courier cannot pick an asset for it) rather than pushed-and-failed.
async fn per_node_update(
    fleet: &HubFleet,
    client: &reqwest::Client,
    base: &Url,
    version: &str,
    t: NodeUpdateTarget,
) -> Value {
    if !t.update_capable {
        return skipped(&t.node, "node did not advertise the `update` capability");
    }
    let (Some(target), Some(variant)) = (t.target.as_deref(), t.variant.as_deref()) else {
        return skipped(&t.node, "node did not report its build target/variant");
    };
    let manifest_url = match per_node_manifest_url(base, version, target, variant) {
        Ok(u) => u,
        Err(e) => return errored(&t.node, &e),
    };
    // Re-vet the derived URL's SHAPE (pure, no DNS): the node's target/variant are ingestion-
    // sanitized (they can carry no `/`,`?`,`#`), but re-checking scheme/no-query/no-fragment keeps
    // the exact belt-and-braces `node_update` has. The origin is already pinned in `client`.
    let vetted = match parse_courier_url(manifest_url.as_str()) {
        Ok(u) => u,
        Err(e) => return errored(&t.node, &e),
    };
    // Bind the fetched manifest to the requested release version — a mismatched (older/different)
    // signed manifest at the expected path is refused, not pushed. Reuses the shared `client`.
    let params = match fetch_and_assemble(client, &vetted, Some(version)).await {
        Ok(p) => p,
        Err(e) => return errored(&t.node, &e),
    };
    // Send ONLY if the node's CURRENT transport is still the one whose (target, variant) we
    // selected the asset from (`t.transport`): a reconnect between the snapshot and now could have
    // changed the node's build, making this asset the WRONG one (the node would ACK it, then fail
    // eligibility) — so report that node skipped rather than claim a bogus "accepted".
    match fleet
        .push_update_pinned(&t.node, &t.transport, params)
        .await
    {
        Ok(PinnedPush::Accepted(reply)) => {
            json!({ "node": t.node, "status": "accepted", "reply": reply })
        }
        Ok(PinnedPush::Reconnected) => skipped(&t.node, "node reconnected during the update"),
        Err(e) => errored(&t.node, &e),
    }
}

fn skipped(node: &str, reason: &str) -> Value {
    json!({ "node": node, "status": "skipped", "reason": reason })
}

fn errored(node: &str, e: &HiggsError) -> Value {
    let mut obj = json!({ "node": node, "status": "error", "error": e.to_string() });
    // Prefer the ORIGIN HG code (a legacy node's HG026 refusal, or a worker's HGxxx carried in a
    // `WorkerRpc`) over the generic boundary code — the same rule the control dispatch + chat relay
    // use (`worker_origin_code_data`), so the report names the true failure.
    if let Some(code) = crate::node::worker_origin_code_data(e).and_then(|d| d.get("code").cloned())
    {
        obj["code"] = code;
    }
    obj
}

#[cfg(test)]
#[path = "release_courier_tests.rs"]
mod tests;
