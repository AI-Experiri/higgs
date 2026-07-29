use std::io::Cursor;

use super::*;

/// A freshly minted throwaway signing identity: the verify-side table entry
/// plus the secret key to sign test manifests with.
struct TestKey {
    key_id: &'static str,
    pk_b64: String,
    sk: minisign::SecretKey,
}

fn mint_key(key_id: &'static str) -> TestKey {
    let minisign::KeyPair { pk, sk } =
        minisign::KeyPair::generate_unencrypted_keypair().expect("keygen");
    TestKey {
        key_id,
        pk_b64: pk.to_base64(),
        sk,
    }
}

impl TestKey {
    fn table(&self) -> Vec<(&str, &str)> {
        vec![(self.key_id, self.pk_b64.as_str())]
    }

    /// Signs `bytes` exactly the way release CI does (prehashed minisign over
    /// the manifest file) and returns the `.minisig` text.
    fn sign(&self, bytes: &[u8]) -> String {
        minisign::sign(
            None,
            &self.sk,
            Cursor::new(bytes),
            Some("test manifest"),
            None,
        )
        .expect("sign")
        .into_string()
    }
}

fn manifest_json(version: &str) -> Vec<u8> {
    serde_json::to_vec(&serde_json::json!({
        "schema": UPDATE_MANIFEST_SCHEMA,
        "version": version,
        "commit": "c0ffee0000000000000000000000000000000000",
        "file": format!("higgs-v{version}-aarch64-apple-darwin.tar.gz"),
        "target": "aarch64-apple-darwin",
        "variant": "metal",
        "sha256": "aa".repeat(32),
    }))
    .expect("manifest json")
}

#[test]
fn verifies_a_ci_shaped_manifest_and_parses_every_field() {
    let key = mint_key("higgs-release-1");
    let bytes = manifest_json("0.9.9");
    let sig = key.sign(&bytes);

    let m = verify_manifest_with(&bytes, &sig, "higgs-release-1", &key.table()).expect("verify");
    assert_eq!(m.schema, UPDATE_MANIFEST_SCHEMA);
    assert_eq!(m.version, "0.9.9");
    assert_eq!(m.commit, "c0ffee0000000000000000000000000000000000");
    assert_eq!(m.file, "higgs-v0.9.9-aarch64-apple-darwin.tar.gz");
    assert_eq!(m.target, "aarch64-apple-darwin");
    assert_eq!(m.variant, "metal");
    assert_eq!(m.sha256, "aa".repeat(32));
}

#[test]
fn every_pubkeys_file_line_parses_and_looks_like_a_key() {
    // The parser SKIPS malformed lines (fail-closed: fewer pins = more
    // refusal) — this test is what turns a silent skip into a loud quality-
    // gate failure: every non-comment line must parse into exactly
    // (key_id, base64), and every base64 must be minisign-pubkey shaped.
    let content = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/.github/release-pubkeys.txt"
    ))
    .expect("read .github/release-pubkeys.txt");
    // PRINTABLE ASCII ONLY (plus tab/LF/CR), enforced: Rust's `trim`
    // understands Unicode whitespace where the release workflow's
    // byte-oriented awk does not, and a NUL survives into the embedded pin
    // here while shell command substitution silently drops it there. On this
    // byte set the two parsers are provably equivalent, and the release job
    // refuses the file with the identical rule.
    assert!(
        content
            .chars()
            .all(|c| matches!(c, '\t' | '\n' | '\r' | ' '..='~')),
        "release-pubkeys.txt must be printable ASCII (+tab/LF/CR) — NULs, control bytes, or Unicode would desync the CI and binary parsers"
    );
    let candidate_lines: Vec<&str> = content
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .collect();
    assert_eq!(
        candidate_lines.len(),
        HIGGS_UPDATE_PUBKEYS.len(),
        "a non-comment line in release-pubkeys.txt failed to parse as `<key_id> <base64>`"
    );
    for (id, pk) in HIGGS_UPDATE_PUBKEYS.iter() {
        assert!(!id.is_empty());
        assert!(
            pk.starts_with("RW") && pk.len() == 56,
            "pinned key {id:?} does not look like a minisign public key: {pk:?}"
        );
    }
}

#[test]
fn the_pubkeys_parser_pins_the_format_rules() {
    let parsed = parse_pubkeys_file(
        "# comment

  key-a RWQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
malformed-only-one-field
key-b RWQBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB extra-field
  # RWQCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
key-d RWQDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD\r
",
    );
    // key-a parses; the one-field and three-field lines are skipped (and the
    // file-shape test above is what makes such lines a gate failure in the
    // REAL file). The INDENTED comment carrying key-shaped material parses as
    // a comment — the release workflow's awk mirrors this trim-then-'#' order
    // exactly, so CI can never accept a key the binary treats as commented
    // out.
    // key-d's CRLF line parses with the \r TRIMMED — CI's awk strips
    // trailing [[:space:]] the same way, so a CRLF-committed pin can never
    // pass the binary while blocking the release gate.
    assert_eq!(
        parsed,
        vec![
            (
                "key-a",
                "RWQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
            ),
            (
                "key-d",
                "RWQDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD"
            )
        ]
    );
}

#[test]
fn unknown_key_id_fails_closed_before_any_crypto() {
    let key = mint_key("higgs-release-1");
    let bytes = manifest_json("0.9.9");
    let sig = key.sign(&bytes);

    let err = verify_manifest_with(&bytes, &sig, "higgs-release-2", &key.table()).unwrap_err();
    assert!(
        matches!(&err, HiggsError::UpdateKeyUnknown { key_id } if key_id == "higgs-release-2"),
        "want HG081, got: {err}"
    );
}

#[test]
fn the_shipped_pubkey_table_rejects_a_foreign_signature() {
    // The PRODUCTION table now pins the real release key, so verification through
    // the PUBLIC wrappers (hardwired to HIGGS_UPDATE_PUBKEYS) must fail CLOSED for
    // anything not signed by that key: a foreign key's signature under the pinned
    // id is HG082 (does not verify), and an unpinned id is HG081.
    assert!(
        HIGGS_UPDATE_PUBKEYS
            .iter()
            .any(|(id, _)| *id == "higgs-release-1"),
        "the release key id must be pinned in .github/release-pubkeys.txt"
    );
    let foreign = mint_key("higgs-release-1"); // a throwaway, NOT the real pinned key
    let bytes = manifest_json("0.9.9");
    let sig = foreign.sign(&bytes);

    // Named lookup under the pinned id: the real key is resolved, the foreign
    // signature does not verify under it → HG082.
    let named = verify_manifest(&bytes, &sig, "higgs-release-1");
    assert!(
        matches!(
            named.unwrap_err(),
            HiggsError::UpdateSignatureInvalid { .. }
        ),
        "a foreign signature under the pinned id fails closed with HG082"
    );
    // An id that is not pinned → HG081, before any crypto.
    let unknown = verify_manifest(&bytes, &sig, "higgs-release-99");
    assert!(
        matches!(unknown.unwrap_err(), HiggsError::UpdateKeyUnknown { .. }),
        "an unpinned id fails closed with HG081"
    );
    // Try-all against the shipped table also rejects the foreign signature.
    let any = verify_manifest_any(&bytes, &sig);
    assert!(
        matches!(any.unwrap_err(), HiggsError::UpdateSignatureInvalid { .. }),
        "try-all against the shipped table rejects a foreign signature (HG082)"
    );
}

#[test]
fn an_empty_pubkey_table_fails_closed() {
    // The empty-table invariant (no pins → nothing verifies, HG081) is no longer
    // reachable through the shipped table now that a real key is pinned, so pin it
    // via the private injection seam instead.
    let key = mint_key("higgs-release-1");
    let bytes = manifest_json("0.9.9");
    let sig = key.sign(&bytes);

    let named = verify_manifest_with(&bytes, &sig, "higgs-release-1", &[]);
    assert!(matches!(
        named.unwrap_err(),
        HiggsError::UpdateKeyUnknown { .. }
    ));
    let any = verify_manifest_any_with(&bytes, &sig, &[]);
    assert!(matches!(
        any.unwrap_err(),
        HiggsError::UpdateKeyUnknown { .. }
    ));
}

#[test]
fn a_tampered_manifest_is_rejected() {
    let key = mint_key("higgs-release-1");
    let bytes = manifest_json("0.9.9");
    let sig = key.sign(&bytes);
    let tampered = String::from_utf8(bytes)
        .expect("utf8")
        .replace("0.9.9", "9.9.9");

    let err = verify_manifest_with(tampered.as_bytes(), &sig, "higgs-release-1", &key.table())
        .unwrap_err();
    assert!(
        matches!(err, HiggsError::UpdateSignatureInvalid { .. }),
        "want HG082, got: {err}"
    );
}

#[test]
fn a_signature_from_a_different_key_under_the_claimed_id_is_rejected() {
    let pinned = mint_key("higgs-release-1");
    let imposter = mint_key("higgs-release-1");
    let bytes = manifest_json("0.9.9");
    let sig = imposter.sign(&bytes);

    let err = verify_manifest_with(&bytes, &sig, "higgs-release-1", &pinned.table()).unwrap_err();
    assert!(
        matches!(err, HiggsError::UpdateSignatureInvalid { .. }),
        "want HG082, got: {err}"
    );
}

#[test]
fn garbage_signature_text_is_rejected_not_panicked_on() {
    let key = mint_key("higgs-release-1");
    let bytes = manifest_json("0.9.9");

    let err = verify_manifest_with(
        &bytes,
        "not a minisig file",
        "higgs-release-1",
        &key.table(),
    )
    .unwrap_err();
    assert!(
        matches!(err, HiggsError::UpdateSignatureInvalid { .. }),
        "want HG082, got: {err}"
    );
}

#[test]
fn a_malformed_pinned_pubkey_is_a_signature_error_not_a_panic() {
    let key = mint_key("higgs-release-1");
    let bytes = manifest_json("0.9.9");
    let sig = key.sign(&bytes);
    let bad_table = [("higgs-release-1", "definitely not base64 key material")];

    let err = verify_manifest_with(&bytes, &sig, "higgs-release-1", &bad_table).unwrap_err();
    assert!(
        matches!(err, HiggsError::UpdateSignatureInvalid { .. }),
        "want HG082, got: {err}"
    );
}

#[test]
fn an_authentic_signature_over_non_manifest_json_is_a_manifest_error() {
    let key = mint_key("higgs-release-1");
    let bytes = b"this is signed but it is not JSON".to_vec();
    let sig = key.sign(&bytes);

    let err = verify_manifest_with(&bytes, &sig, "higgs-release-1", &key.table()).unwrap_err();
    assert!(
        matches!(err, HiggsError::UpdateManifestInvalid { .. }),
        "want HG083, got: {err}"
    );
}

#[test]
fn an_unknown_schema_is_refused_even_when_authentic() {
    let key = mint_key("higgs-release-1");
    let bytes = String::from_utf8(manifest_json("0.9.9"))
        .expect("utf8")
        .replace("\"schema\":1", "\"schema\":2")
        .into_bytes();
    let sig = key.sign(&bytes);

    let err = verify_manifest_with(&bytes, &sig, "higgs-release-1", &key.table()).unwrap_err();
    assert!(
        matches!(&err, HiggsError::UpdateManifestInvalid { detail } if detail.contains("schema 2")),
        "want HG083 naming schema 2, got: {err}"
    );
}

#[test]
fn unknown_manifest_fields_are_ignored_for_forward_compat() {
    // Additive CI fields must not brick older verifiers — only a schema bump may.
    let key = mint_key("higgs-release-1");
    let mut value: serde_json::Value =
        serde_json::from_slice(&manifest_json("0.9.9")).expect("json");
    value["a_future_field"] = serde_json::json!("ignored");
    let bytes = serde_json::to_vec(&value).expect("json");
    let sig = key.sign(&bytes);

    let m = verify_manifest_with(&bytes, &sig, "higgs-release-1", &key.table()).expect("verify");
    assert_eq!(m.version, "0.9.9");
}

#[test]
fn verify_any_reports_which_pinned_key_matched() {
    let old = mint_key("higgs-release-1");
    let new = mint_key("higgs-release-2");
    let table = vec![
        (old.key_id, old.pk_b64.as_str()),
        (new.key_id, new.pk_b64.as_str()),
    ];
    let bytes = manifest_json("0.9.9");
    let sig = new.sign(&bytes);

    let (key_id, m) = verify_manifest_any_with(&bytes, &sig, &table).expect("verify");
    assert_eq!(key_id, "higgs-release-2");
    assert_eq!(m.version, "0.9.9");
}

#[test]
fn verify_any_surfaces_a_manifest_error_once_a_key_authenticates() {
    // Rotation/skew scenario: the signature is GENUINE under a pinned key but
    // the manifest schema is newer than this binary. Reporting HG082 ("no key
    // verifies") would send the operator toward key repair; the honest error
    // is HG083 — the remedy is one manual binary update.
    let key = mint_key("higgs-release-1");
    let bytes = String::from_utf8(manifest_json("0.9.9"))
        .expect("utf8")
        .replace("\"schema\":1", "\"schema\":2")
        .into_bytes();
    let sig = key.sign(&bytes);

    let err = verify_manifest_any_with(&bytes, &sig, &key.table()).unwrap_err();
    assert!(
        matches!(&err, HiggsError::UpdateManifestInvalid { detail } if detail.contains("schema 2")),
        "want HG083 naming schema 2, got: {err}"
    );
}

#[test]
fn verify_any_reaches_a_tuple_shadowed_by_a_duplicate_label() {
    // A mis-edited table can carry two tuples under ONE label. The name-lookup
    // path can only ever see the first; the try-all path must still verify
    // against the SECOND tuple's key (it iterates tuples, not labels) — a
    // signature valid under the shadowed key would otherwise be rejected.
    let first = mint_key("higgs-release-1");
    let shadowed = mint_key("higgs-release-1");
    let table = vec![
        (first.key_id, first.pk_b64.as_str()),
        (shadowed.key_id, shadowed.pk_b64.as_str()),
    ];
    let bytes = manifest_json("0.9.9");
    let sig = shadowed.sign(&bytes);

    let (key_id, m) = verify_manifest_any_with(&bytes, &sig, &table).expect("verify");
    assert_eq!(key_id, "higgs-release-1");
    assert_eq!(m.version, "0.9.9");
}

#[test]
fn the_shipped_pubkey_table_has_unique_labels() {
    // Duplicate labels shadow tuples on the name-lookup path (first match
    // wins) — refuse them at the source. Trivially true while the table is
    // empty; guards every future pin/rotation edit.
    let mut seen = std::collections::HashSet::new();
    for (id, _) in HIGGS_UPDATE_PUBKEYS.iter() {
        assert!(
            seen.insert(id),
            "duplicate key_id {id:?} in HIGGS_UPDATE_PUBKEYS"
        );
    }
}

#[test]
fn verify_any_rejects_when_no_pinned_key_matches() {
    let pinned = mint_key("higgs-release-1");
    let stranger = mint_key("elsewhere");
    let bytes = manifest_json("0.9.9");
    let sig = stranger.sign(&bytes);

    let err = verify_manifest_any_with(&bytes, &sig, &pinned.table()).unwrap_err();
    assert!(
        matches!(err, HiggsError::UpdateSignatureInvalid { .. }),
        "want HG082, got: {err}"
    );
}

#[test]
fn artifact_sha256_binds_the_bytes_to_the_manifest() {
    let artifact = b"pretend tarball bytes";
    let digest = {
        use sha2::Digest;
        let mut hex = String::new();
        for b in sha2::Sha256::digest(artifact) {
            use std::fmt::Write;
            let _ = write!(hex, "{b:02x}");
        }
        hex
    };
    let mut manifest: UpdateManifest =
        serde_json::from_slice(&manifest_json("0.9.9")).expect("json");
    manifest.sha256 = digest.clone();

    verify_artifact_sha256(&manifest, artifact).expect("matching bytes verify");

    // Case-insensitive: shasum emits lowercase but a hand-typed pin may not.
    manifest.sha256 = digest.to_uppercase();
    verify_artifact_sha256(&manifest, artifact).expect("uppercase pin verifies");

    manifest.sha256 = digest;
    let err = verify_artifact_sha256(&manifest, b"different bytes").unwrap_err();
    assert!(
        matches!(&err, HiggsError::UpdateArtifactMismatch { file, .. } if file == &manifest.file),
        "want HG084, got: {err}"
    );
}
