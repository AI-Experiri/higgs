//! Binary self-update verification — pinned release keys + signed manifests.
//!
//! This module is the TRUST ANCHOR for the node self-update path
//! ([`crate::node::self_update`]; DESIGN-remote.md §9, `M_NODE_UPDATE`): release CI signs a small JSON
//! *manifest* per artifact with the project's minisign secret key, and every
//! `higgs` binary carries the matching public keys in
//! [`HIGGS_UPDATE_PUBKEYS`]. A node verifies the manifest signature against a
//! pinned key BEFORE trusting anything the manifest says (version, artifact
//! sha256) — no TOFU, and the hub (or GitHub itself) is only ever a courier,
//! never an authority.
//!
//! Why the signature covers a MANIFEST and not the artifact bytes directly: a
//! raw detached signature over the tarball proves the bytes came from CI, but
//! binds no version and no target — a courier could replay an older signed
//! artifact (downgrade) or hand a CUDA build to a CPU node. The manifest binds
//! `(version, target, variant, sha256)` under ONE signature; the artifact is
//! then authenticated transitively by its sha256 ([`verify_artifact_sha256`]).
//!
//! Nothing here performs a download or a swap — those are
//! [`crate::node::self_update`]'s job. This module only answers "is this manifest really from release CI, and
//! does this artifact match it?" — AUTHENTICITY, never ELIGIBILITY. Whether a
//! verified manifest's `version` is actually newer than the running binary,
//! and whether its `target`/`variant` match this installation, are separate
//! policy checks the updater (DESIGN-remote.md §9 P3: semver compare,
//! downgrade refusal, install-recorded variant binding) must enforce before
//! applying anything — a genuine old manifest replayed by a courier still
//! verifies here.
//!
//! What signing can and cannot promise: the signature attests "this is what
//! release CI built and published for this version/target" — post-build
//! tampering (courier, storage, transport) and cross-version/target replay of
//! artifacts are refused. It does NOT attest the dependency tree is honest:
//! builds run `--locked`, so that trust boundary is the COMMITTED Cargo.lock
//! (an attacker inside it owns the build output before any signature exists —
//! no release-signing scheme can close that; reviewing lockfile changes is
//! the control).

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::diagnostic::HiggsError;

/// Raw content of `.github/release-pubkeys.txt` — THE single source of truth
/// for the release keys, embedded at compile time. Release CI reads the same
/// file, so the pins a binary verifies against and the keys CI will sign with
/// can never drift (there is no second copy to forget).
const PUBKEYS_FILE: &str = include_str!("../.github/release-pubkeys.txt");

/// The release public keys this binary trusts, as `(key_id, minisign public
/// key base64)` pairs parsed from [`PUBKEYS_FILE`] (`<key_id> <base64>` per
/// non-comment line — the base64 payload line of a `minisign.pub` file).
///
/// The `key_id` is OUR label (e.g. `"higgs-release-1"`), named by update
/// requests as `pinned_key_id` (DESIGN-remote.md §9); it is independent of
/// minisign's internal 8-byte key id. EMPTY-CAPABLE by design: with no pinned
/// key every verification fails closed ([HG081]) and self-update is simply
/// impossible — never "unverified". Lines that do not parse are SKIPPED
/// (fewer pins = more refusal, the fail-closed direction); a unit test fails
/// the quality gate on any malformed or duplicate-id line, so a bad edit
/// cannot reach a release quietly.
///
/// Populating it (operator step, one time): mint a keypair with
/// `minisign -G -W -p minisign.pub -s minisign.key`, put the SECRET key's file
/// content into the `MINISIGN_SECRET_KEY` environment secret (release.yml
/// signs with it), and add a `higgs-release-1 <base64 line of minisign.pub>`
/// line to `.github/release-pubkeys.txt` — the ONLY place keys live.
/// Rotation: mint the new key, ship a BRIDGE release whose binary pins BOTH
/// keys (old + new) and is still SIGNED WITH THE OLD key — deployed binaries
/// pin only the old key, so a bridge signed with the new one would be
/// rejected by every node in the field. Only releases AFTER the bridge switch
/// the signing secret to the new key; nodes pick up the new pin by updating
/// through the bridge.
///
/// Release CI ENFORCES the pin: the release job derives the public key from
/// its signing secret and refuses to sign unless that key appears in the same
/// file at the commit being released — so while the file lists no keys,
/// releases are blocked until the one-time pinning step is done (a release
/// signed by an unpinned key would be silently un-verifiable by every shipped
/// binary).
pub static HIGGS_UPDATE_PUBKEYS: std::sync::LazyLock<Vec<(&'static str, &'static str)>> =
    std::sync::LazyLock::new(|| parse_pubkeys_file(PUBKEYS_FILE));

/// Parses `<key_id> <base64>` lines, skipping `#` comments, blanks, and (in
/// the fail-closed direction) anything malformed. Split out so the unit tests
/// can pin the format rules directly.
fn parse_pubkeys_file(content: &'static str) -> Vec<(&'static str, &'static str)> {
    content
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .filter_map(|l| {
            let mut parts = l.split_whitespace();
            match (parts.next(), parts.next(), parts.next()) {
                (Some(id), Some(pk), None) => Some((id, pk)),
                _ => None,
            }
        })
        .collect()
}

/// The manifest schema this binary understands. CI stamps it into every
/// manifest; a binary refuses schemas it does not know ([HG083]) instead of
/// guessing — the remedy is one manual update to a binary that knows the new
/// schema. Additive fields do NOT bump this (unknown fields are ignored).
pub const UPDATE_MANIFEST_SCHEMA: u32 = 1;

/// One release artifact's signed self-description, minted AND signed by the
/// RELEASE job of `.github/workflows/release.yml` (never by the build jobs,
/// which execute arbitrary dependency build scripts) as
/// `higgs-v<ver>-<suffix>.manifest` plus a sibling `.manifest.minisig`.
/// Verify with [`verify_manifest`] / [`verify_manifest_any`] BEFORE reading
/// any field.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UpdateManifest {
    /// Manifest schema — see [`UPDATE_MANIFEST_SCHEMA`].
    pub schema: u32,
    /// The release semver, exactly as in the tag `v<version>`.
    pub version: String,
    /// Full SHA of the commit release CI built — binds the artifact to its
    /// exact sources independently of the tag (a moved tag cannot re-home a
    /// signed artifact's provenance).
    pub commit: String,
    /// The artifact file name (`higgs-v<ver>-<suffix>.tar.gz`).
    pub file: String,
    /// Rust target triple the binary was built for (`aarch64-apple-darwin`, …).
    pub target: String,
    /// Acceleration variant: `metal`, `cpu`, or `cuda`. A CUDA binary on a
    /// CPU-only box fails to even load, so the updater must match this against
    /// the variant recorded at install — the triple alone cannot tell them apart.
    pub variant: String,
    /// Hex SHA-256 of the artifact tarball; check with [`verify_artifact_sha256`].
    pub sha256: String,
}

/// Verifies `signature_text` (the full text of a `.minisig` file) over
/// `manifest_bytes` against the COMPILED-IN key named `pinned_key_id`
/// ([`HIGGS_UPDATE_PUBKEYS`] — the table is deliberately not a parameter, so
/// no caller can be talked into verifying against courier-supplied keys),
/// then parses the manifest. The signature is checked BEFORE the JSON is
/// parsed — this function never interprets unauthenticated input.
///
/// AUTHENTICITY ONLY. A verified manifest proves "release CI said this about
/// artifact X" — it does NOT decide eligibility: whether `version` is newer
/// than the running binary, or `target`/`variant` match this installation, is
/// the updater's policy layer ([`crate::node::self_update`], which verifies via
/// [`verify_manifest_any`] and then enforces these checks in
/// `evaluate_eligibility`) and MUST also be enforced before any download is
/// applied.
///
/// Errors: [HG081] unknown/unpinned key id, [HG082] signature rejected (or a
/// malformed pinned key / signature text), [HG083] authenticated but
/// unparseable or wrong-schema manifest.
pub fn verify_manifest(
    manifest_bytes: &[u8],
    signature_text: &str,
    pinned_key_id: &str,
) -> Result<UpdateManifest, HiggsError> {
    verify_manifest_with(
        manifest_bytes,
        signature_text,
        pinned_key_id,
        &HIGGS_UPDATE_PUBKEYS,
    )
}

/// [`verify_manifest`] with an injected key table — the unit-test seam
/// (throwaway keypairs), kept PRIVATE so production code can only ever verify
/// against the compiled-in pins.
fn verify_manifest_with(
    manifest_bytes: &[u8],
    signature_text: &str,
    pinned_key_id: &str,
    pubkeys: &[(&str, &str)],
) -> Result<UpdateManifest, HiggsError> {
    let (_, pk_b64) = pubkeys
        .iter()
        .find(|(id, _)| *id == pinned_key_id)
        .ok_or_else(|| HiggsError::UpdateKeyUnknown {
            key_id: pinned_key_id.to_string(),
        })?;
    verify_with_key(manifest_bytes, signature_text, pinned_key_id, pk_b64)
}

/// Verifies against ONE concrete key tuple. Both callers route here:
/// name-lookup ([`verify_manifest_with`]) after resolving the label, and
/// try-all ([`verify_manifest_any_with`]) iterating tuples DIRECTLY — never
/// re-resolving by label, so a duplicated `key_id` cannot shadow a later
/// tuple out of reach.
fn verify_with_key(
    manifest_bytes: &[u8],
    signature_text: &str,
    key_id: &str,
    pk_b64: &str,
) -> Result<UpdateManifest, HiggsError> {
    let pk = minisign_verify::PublicKey::from_base64(pk_b64).map_err(|e| {
        HiggsError::UpdateSignatureInvalid {
            detail: format!("pinned public key {key_id:?} is malformed: {e}"),
        }
    })?;
    let sig = minisign_verify::Signature::decode(signature_text).map_err(|e| {
        HiggsError::UpdateSignatureInvalid {
            detail: format!("signature is malformed: {e}"),
        }
    })?;
    pk.verify(manifest_bytes, &sig, false)
        .map_err(|e| HiggsError::UpdateSignatureInvalid {
            detail: format!("signature does not verify under key {key_id:?}: {e}"),
        })?;
    parse_verified_manifest(manifest_bytes)
}

/// Like [`verify_manifest`], but with no caller-named key: tries EVERY
/// compiled-in pin ([`HIGGS_UPDATE_PUBKEYS`]) and returns the `key_id` that
/// verified alongside the manifest. This is the direct-download path's shape
/// (a GitHub release carries no `pinned_key_id`; the table is tiny). An empty
/// table fails closed with [HG081]; a table none of whose keys verify fails
/// with [HG082]. Authenticity only — see [`verify_manifest`] on eligibility.
pub fn verify_manifest_any(
    manifest_bytes: &[u8],
    signature_text: &str,
) -> Result<(String, UpdateManifest), HiggsError> {
    verify_manifest_any_with(manifest_bytes, signature_text, &HIGGS_UPDATE_PUBKEYS)
}

/// [`verify_manifest_any`] with an injected key table — PRIVATE test seam,
/// same rationale as [`verify_manifest_with`].
fn verify_manifest_any_with(
    manifest_bytes: &[u8],
    signature_text: &str,
    pubkeys: &[(&str, &str)],
) -> Result<(String, UpdateManifest), HiggsError> {
    if pubkeys.is_empty() {
        return Err(HiggsError::UpdateKeyUnknown {
            key_id: "<none pinned>".to_string(),
        });
    }
    for (key_id, pk_b64) in pubkeys {
        match verify_with_key(manifest_bytes, signature_text, key_id, pk_b64) {
            Ok(m) => return Ok((key_id.to_string(), m)),
            // The signature VERIFIED under this key — the manifest itself is
            // the problem (bad JSON / unknown schema). Trying other keys can't
            // change that, and reporting HG082 would send the operator toward
            // key repair when the remedy is a manual binary update (rotation /
            // schema skew). Surface the manifest error as-is.
            Err(e @ HiggsError::UpdateManifestInvalid { .. }) => return Err(e),
            // Wrong key / malformed signature — try the next pin.
            Err(_) => {}
        }
    }
    Err(HiggsError::UpdateSignatureInvalid {
        detail: format!(
            "signature does not verify under any of the {} pinned release key(s)",
            pubkeys.len()
        ),
    })
}

/// Checks that `artifact` (the tarball bytes) matches the VERIFIED manifest's
/// sha256 — the step that extends the manifest signature's authenticity to the
/// artifact itself. Hex compare is case-insensitive. [HG084] on mismatch.
pub fn verify_artifact_sha256(
    manifest: &UpdateManifest,
    artifact: &[u8],
) -> Result<(), HiggsError> {
    let got = hex(&Sha256::digest(artifact));
    if got.eq_ignore_ascii_case(&manifest.sha256) {
        Ok(())
    } else {
        Err(HiggsError::UpdateArtifactMismatch {
            file: manifest.file.clone(),
            expected: manifest.sha256.clone(),
            got,
        })
    }
}

/// Parses manifest JSON that has ALREADY passed signature verification and
/// enforces the schema pin. Split out so both verify fns share one parse rule.
fn parse_verified_manifest(manifest_bytes: &[u8]) -> Result<UpdateManifest, HiggsError> {
    let manifest: UpdateManifest =
        serde_json::from_slice(manifest_bytes).map_err(|e| HiggsError::UpdateManifestInvalid {
            detail: format!("manifest is not valid JSON: {e}"),
        })?;
    if manifest.schema != UPDATE_MANIFEST_SCHEMA {
        return Err(HiggsError::UpdateManifestInvalid {
            detail: format!(
                "unsupported manifest schema {} (this binary understands {})",
                manifest.schema, UPDATE_MANIFEST_SCHEMA
            ),
        });
    }
    Ok(manifest)
}

/// Lowercase hex of a digest. `sha2` has no hex helper and pulling `hex`/
/// `data-encoding` for 3 lines isn't worth a dependency.
fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    bytes
        .iter()
        .fold(String::with_capacity(bytes.len() * 2), |mut s, b| {
            let _ = write!(s, "{b:02x}");
            s
        })
}

#[cfg(test)]
#[path = "update_tests.rs"]
mod tests;
