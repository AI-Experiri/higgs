use super::*;

#[test]
fn empty_store_is_open() {
    let ks = ApiKeys::default();
    assert!(ks.is_empty());
    assert!(
        ks.authorizes("anything", Scope::Admin),
        "no keys = auth disabled"
    );
}

#[test]
fn scoped_key_authorizes_its_scope_only() {
    let mut ks = ApiKeys::default();
    ks.add("hgk_chat", "c".into(), vec![Scope::Chat]);
    assert!(ks.authorizes("hgk_chat", Scope::Chat));
    assert!(
        !ks.authorizes("hgk_chat", Scope::Models),
        "chat key can't list models"
    );
    assert!(
        !ks.authorizes("hgk_chat", Scope::Admin),
        "chat key isn't admin"
    );
    assert!(
        !ks.authorizes("wrong", Scope::Chat),
        "unknown token rejected"
    );
}

#[test]
fn admin_is_a_superset() {
    let mut ks = ApiKeys::default();
    ks.add("hgk_admin", "a".into(), vec![Scope::Admin]);
    assert!(ks.authorizes("hgk_admin", Scope::Chat));
    assert!(ks.authorizes("hgk_admin", Scope::Models));
    assert!(ks.authorizes("hgk_admin", Scope::Admin));
}

#[test]
fn add_replaces_same_token_and_remove_by_label() {
    let mut ks = ApiKeys::default();
    ks.add("hgk_x", "first".into(), vec![Scope::Chat]);
    ks.add("hgk_x", "second".into(), vec![Scope::Admin]); // same token, new label/scope
    assert_eq!(
        ks.iter().count(),
        1,
        "identical token replaced, not duplicated"
    );
    assert!(ks.authorizes("hgk_x", Scope::Admin));
    assert_eq!(ks.remove_label("second"), 1);
    assert!(ks.is_empty());
}

#[test]
fn hash_is_stable_hex_and_token_minting_is_prefixed() {
    assert_eq!(hash_token("a").len(), 64, "sha256 hex is 64 chars");
    assert_eq!(hash_token("a"), hash_token("a"));
    assert_ne!(hash_token("a"), hash_token("b"));
    assert!(mint_token([0u8; 16]).starts_with("hgk_"));
}

#[test]
fn scope_parse_roundtrips_known_and_rejects_unknown() {
    assert_eq!(Scope::parse("Chat"), Some(Scope::Chat));
    assert_eq!(Scope::parse(" admin "), Some(Scope::Admin));
    assert_eq!(Scope::parse("models"), Some(Scope::Models));
    assert_eq!(Scope::parse("root"), None);
}

#[test]
fn parse_scopes_handles_lists_and_rejects_unknown() {
    assert_eq!(
        parse_scopes("chat, models").unwrap(),
        vec![Scope::Chat, Scope::Models]
    );
    assert!(parse_scopes("").unwrap().is_empty());
    assert!(parse_scopes("chat,bogus").is_err());
}

#[test]
fn run_keys_add_list_remove_against_a_path() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("api_keys.json");
    let s = |v: &[&str]| v.iter().map(ToString::to_string).collect::<Vec<_>>();

    // add → store has the key with the requested scopes.
    run_keys_at(&path, &s(&["add", "ci", "chat,models"])).unwrap();
    let ks = ApiKeys::load(&path).unwrap();
    let k = ks.iter().find(|k| k.label == "ci").expect("key added");
    assert_eq!(k.scopes, vec![Scope::Chat, Scope::Models]);

    // list + remove are exercised; remove drops it.
    run_keys_at(&path, &s(&["list"])).unwrap();
    run_keys_at(&path, &s(&["remove", "ci"])).unwrap();
    assert!(ApiKeys::load(&path).unwrap().is_empty(), "key removed");

    // usage errors: missing label, bad scope, unknown subcommand.
    assert!(run_keys_at(&path, &s(&["add"])).is_err());
    assert!(run_keys_at(&path, &s(&["add", "x", "nope"])).is_err());
    assert!(run_keys_at(&path, &s(&["remove"])).is_err());
    assert!(run_keys_at(&path, &s(&["bogus"])).is_err());
    // list on an empty/missing store is fine.
    run_keys_at(&path, &s(&["list"])).unwrap();
}

#[test]
fn load_io_error_other_than_not_found_propagates() {
    // A non-NotFound read error (here: the path is a directory) propagates verbatim,
    // rather than being swallowed as an empty/open store like a missing file is.
    let dir = tempfile::tempdir().unwrap();
    let as_dir = dir.path().join("api_keys-is-a-dir");
    std::fs::create_dir(&as_dir).unwrap();
    let err = ApiKeys::load(&as_dir).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::IsADirectory);
}

#[test]
fn save_rename_failure_is_surfaced() {
    // The atomic rename fails when the destination path is itself a directory,
    // surfacing the error rather than silently dropping the write.
    let dir = tempfile::tempdir().unwrap();
    let target = dir.path().join("keys-target");
    std::fs::create_dir(&target).unwrap();
    let mut ks = ApiKeys::default();
    ks.add("hgk_t", "t".into(), vec![Scope::Chat]);
    let err = ks.save(&target).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::IsADirectory);
}

#[test]
fn save_write_failure_is_surfaced() {
    // Writing the temp file into a READ-ONLY directory fails and the error propagates
    // (the `?` on the write). Skipped when running as root (perms bypassed).
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().unwrap();
    let ro = dir.path().join("readonly");
    std::fs::create_dir(&ro).unwrap();
    std::fs::set_permissions(&ro, std::fs::Permissions::from_mode(0o555)).unwrap();
    let mut ks = ApiKeys::default();
    ks.add("hgk_w", "w".into(), vec![Scope::Chat]);
    let result = ks.save(&ro.join("api_keys.json"));
    let _ = std::fs::set_permissions(&ro, std::fs::Permissions::from_mode(0o755));
    // Root bypasses the read-only bit; only assert when the write was actually denied.
    if let Err(e) = result {
        assert_eq!(e.kind(), std::io::ErrorKind::PermissionDenied);
    }
}

#[test]
fn run_keys_add_empty_scope_is_rejected() {
    // An explicit empty scope string parses to zero scopes, which `add` rejects with
    // "at least one scope required" (distinct from the unknown-scope parse error).
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("api_keys.json");
    let s = |v: &[&str]| v.iter().map(ToString::to_string).collect::<Vec<_>>();
    let err = run_keys_at(&path, &s(&["add", "lbl", ""])).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    assert!(
        err.to_string().contains("at least one scope"),
        "empty scope list is rejected: {err}"
    );
    // No store was written (the add never reached save).
    assert!(ApiKeys::load(&path).unwrap().is_empty());
}

#[test]
fn run_keys_uses_higgs_home_keystore() {
    // `run_keys` (the public entry) resolves the keystore via `keys_path()` (HIGGS_HOME);
    // a `list` against a fresh home is a clean no-op. Serialized with other HIGGS_HOME tests.
    let _env = crate::TEST_ENV_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let prev = std::env::var_os("HIGGS_HOME");
    let tmp = std::env::temp_dir().join("higgs-run-keys-home-test");
    let _ = std::fs::remove_dir_all(&tmp);
    // SAFETY: serialized by TEST_ENV_LOCK; restored below.
    unsafe { std::env::set_var("HIGGS_HOME", &tmp) };

    // list on an empty home keystore is Ok (the surface is OPEN).
    run_keys(&["list".to_string()]).unwrap();
    // add via the public entry writes into the HIGGS_HOME keystore.
    run_keys(&["add".to_string(), "home-key".to_string()]).unwrap();
    let ks = ApiKeys::load(&keys_path().unwrap()).unwrap();
    assert!(
        ks.iter().any(|k| k.label == "home-key"),
        "run_keys wrote into the HIGGS_HOME keystore"
    );

    // SAFETY: still under the lock.
    unsafe {
        match prev {
            Some(v) => std::env::set_var("HIGGS_HOME", v),
            None => std::env::remove_var("HIGGS_HOME"),
        }
    }
    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn save_then_load_roundtrips() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("api_keys.json");
    let mut ks = ApiKeys::default();
    ks.add("hgk_z", "z".into(), vec![Scope::Chat, Scope::Models]);
    ks.save(&path).unwrap();
    let back = ApiKeys::load(&path).unwrap();
    assert!(back.authorizes("hgk_z", Scope::Models));
    // Missing file → empty/open store.
    assert!(ApiKeys::load(&dir.path().join("nope.json"))
        .unwrap()
        .is_empty());
}

/// G4 redaction: `{:?}` on a stored key must show only the short digest
/// prefix — never the full digest (a stable key identifier that has no
/// business in logs or panic messages).
#[test]
fn debug_output_redacts_the_digest() {
    let mut keys = ApiKeys::default();
    keys.add("hgk_deadbeef", "ci".into(), vec![Scope::Chat]);
    let key = keys.iter().next().unwrap();
    let dbg = format!("{key:?}");
    assert!(dbg.contains("ci"), "label shown: {dbg}");
    assert!(
        !dbg.contains(&key.sha256),
        "full digest must not appear in Debug: {dbg}"
    );
    assert!(
        dbg.contains(key.sha256_prefix()),
        "digest prefix identifies the key: {dbg}"
    );
}

// ── Key timestamps + last-used tracking (G4b tokens UI) ──────────────────

/// Mint stamps `created_at_ms` (wall-clock, non-zero) and starts with no
/// last-used; `touch` records a use and reports whether anything changed.
#[test]
fn mint_stamps_created_at_and_touch_records_last_used() {
    let mut ks = ApiKeys::default();
    ks.add("hgk_t1", "laptop".into(), vec![Scope::Chat]);
    let k = ks.iter().next().unwrap();
    assert!(
        k.created_at_ms.is_some(),
        "created_at_ms must be stamped at mint"
    );
    assert_eq!(k.last_used_ms, None, "never used at mint");
    let sha = k.sha256.clone();

    assert!(ks.touch(&sha, 1234), "first touch changes the key");
    assert_eq!(ks.iter().next().unwrap().last_used_ms, Some(1234));
    assert!(!ks.touch(&sha, 1234), "same-timestamp touch is a no-op");
    assert!(!ks.touch("no-such-digest", 99), "unknown digest is a no-op");
}

/// `touch` is MONOTONIC: a stale/reordered timestamp older than the stored
/// last-used is a no-op — the stamp never moves backward (round-1 finding #2).
#[test]
fn touch_never_moves_last_used_backward() {
    let mut ks = ApiKeys::default();
    ks.add("hgk_m", "m".into(), vec![Scope::Chat]);
    let sha = ks.iter().next().unwrap().sha256.clone();
    assert!(ks.touch(&sha, 5000));
    assert!(!ks.touch(&sha, 1000), "an older timestamp must be a no-op");
    assert_eq!(
        ks.iter().next().unwrap().last_used_ms,
        Some(5000),
        "last_used_ms must not regress"
    );
    assert!(ks.touch(&sha, 5001), "a newer timestamp still advances");
    assert_eq!(ks.iter().next().unwrap().last_used_ms, Some(5001));
}

/// `authorizing_sha` returns the matched key's digest for an authorized
/// bearer and `None` for a wrong scope — the seam the auth middleware uses
/// to record last-used without a second lookup.
#[test]
fn authorizing_sha_identifies_the_matching_key() {
    let mut ks = ApiKeys::default();
    ks.add("hgk_chat", "c".into(), vec![Scope::Chat]);
    let sha = ks.iter().next().unwrap().sha256.clone();
    assert_eq!(ks.authorizing_sha("hgk_chat", Scope::Chat), Some(sha));
    assert_eq!(ks.authorizing_sha("hgk_chat", Scope::Admin), None);
    assert_eq!(ks.authorizing_sha("hgk_wrong", Scope::Chat), None);
}

/// A pre-timestamp store (no created_at/last_used fields) still loads — the
/// serde defaults grandfather old `api_keys.json` files.
#[test]
fn pre_timestamp_store_loads_with_defaults() {
    let json = r#"{"keys":[{"sha256":"abc123","label":"old","scopes":["chat"]}]}"#;
    let ks: ApiKeys = serde_json::from_str(json).unwrap();
    let k = ks.iter().next().unwrap();
    assert_eq!(k.created_at_ms, None);
    assert_eq!(k.last_used_ms, None);
}

/// A tampered/hand-edited store carrying `"hidden": true` must NOT yield a
/// stealth backdoor: `hidden` is an in-memory-only flag (set solely by
/// `add_internal`) and is `skip_deserializing`, so a disk value is ignored. The
/// Admin key loads as a VISIBLE, manageable key — it shows in the key list and
/// is revocable — not a credential that `authorizes()` accepts while `visible()`
/// hides it and `remove_label()` refuses to delete.
///
/// Fail-on-revert: drop `skip_deserializing` from `ApiKey::hidden` and the key
/// deserializes as hidden → `visible().count()` is 0 here.
#[test]
fn a_hidden_flag_on_disk_is_ignored_no_stealth_backdoor() {
    let json =
        r#"{"keys":[{"sha256":"deadbeef","label":"backdoor","scopes":["admin"],"hidden":true}]}"#;
    let mut ks: ApiKeys = serde_json::from_str(json).unwrap();
    assert_eq!(ks.iter().count(), 1, "the key still loads");
    assert_eq!(
        ks.visible().count(),
        1,
        "a disk 'hidden:true' must not make the key invisible to management"
    );
    assert!(
        ks.iter().all(|k| !k.hidden),
        "no key loaded from disk may be flagged hidden"
    );
    // It is on the visible/removable surface — not a stealth key.
    assert_eq!(ks.remove_label("backdoor"), 1, "and it is revocable");
}

// ── Internal, in-memory-only embedder token (hidden keys) ───────────────────

#[test]
fn add_internal_registers_a_working_hidden_token() {
    let mut ks = ApiKeys::default();
    ks.add_internal(
        "hgk_internal",
        "jigglebot (internal)".into(),
        vec![Scope::Chat],
    );
    // The provided plaintext authorizes its scope like any bearer…
    assert!(ks.authorizes("hgk_internal", Scope::Chat));
    assert!(
        !ks.authorizes("hgk_internal", Scope::Admin),
        "scoped to Chat only"
    );
    // …and the store is no longer "open" — auth is now ON.
    assert!(!ks.is_empty());
    // The key is flagged hidden.
    assert!(
        ks.iter().all(|k| k.hidden),
        "the only key is the hidden one"
    );
}

#[test]
fn add_internal_rotation_revokes_the_previous_token() {
    // Re-registering the internal token under the same label must REVOKE the old
    // bearer — otherwise the stale hidden key lives forever (never persisted,
    // never listed, not removable via remove_label).
    let mut ks = ApiKeys::default();
    ks.add_internal("hgk_old", "jigglebot (internal)".into(), vec![Scope::Admin]);
    ks.add_internal("hgk_new", "jigglebot (internal)".into(), vec![Scope::Admin]);
    assert!(
        !ks.authorizes("hgk_old", Scope::Chat),
        "the old internal token must not authorize after rotation"
    );
    assert!(ks.authorizes("hgk_new", Scope::Admin));
    assert_eq!(
        ks.iter().count(),
        1,
        "only the current internal token remains"
    );
}

#[test]
fn add_internal_does_not_drop_a_visible_key_sharing_the_label() {
    // The label-drop is guarded by `k.hidden`, so a user's VISIBLE key that happens
    // to share the internal label survives an internal-token registration.
    let mut ks = ApiKeys::default();
    ks.add("hgk_user", "jigglebot (internal)".into(), vec![Scope::Chat]);
    ks.add_internal("hgk_int", "jigglebot (internal)".into(), vec![Scope::Admin]);
    assert!(
        ks.authorizes("hgk_user", Scope::Chat),
        "a visible key sharing the label must survive"
    );
    assert!(ks.authorizes("hgk_int", Scope::Admin));
    assert_eq!(ks.visible().count(), 1);
    assert_eq!(ks.iter().count(), 2);
}

#[test]
fn visible_excludes_hidden_internal_keys() {
    let mut ks = ApiKeys::default();
    ks.add("hgk_user", "laptop".into(), vec![Scope::Chat]);
    ks.add_internal(
        "hgk_internal",
        "jigglebot (internal)".into(),
        vec![Scope::Chat],
    );
    let visible: Vec<_> = ks.visible().map(|k| k.label.clone()).collect();
    assert_eq!(
        visible,
        vec!["laptop".to_string()],
        "only the user key is visible"
    );
    // But auth still sees BOTH (iter is unfiltered).
    assert_eq!(ks.iter().count(), 2);
}

/// A revoke that names the hidden internal key's label must be a NO-OP — the
/// embedder's in-memory token is immune to the public key-management surface, so
/// `DELETE /api/higgs/keys/jigglebot (internal)` can't strand the embedder's own
/// auth. Fail-on-revert: drop the `k.hidden ||` guard in `remove_label` and this
/// removes 1 (the hidden key), after which `authorizes` returns false.
#[test]
fn remove_label_never_removes_a_hidden_internal_key() {
    let mut ks = ApiKeys::default();
    ks.add("hgk_user", "laptop".into(), vec![Scope::Chat]);
    ks.add_internal(
        "hgk_internal",
        "jigglebot (internal)".into(),
        vec![Scope::Chat, Scope::Models, Scope::Admin],
    );

    let removed = ks.remove_label("jigglebot (internal)");
    assert_eq!(
        removed, 0,
        "the hidden internal key must not be revocable by label"
    );
    assert!(
        ks.authorizes("hgk_internal", Scope::Admin),
        "the hidden internal token still authorizes after a same-label revoke attempt"
    );
    // A visible label still revokes normally.
    assert_eq!(
        ks.remove_label("laptop"),
        1,
        "visible keys revoke by label as before"
    );
}

#[test]
fn save_never_persists_a_hidden_internal_key() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("api_keys.json");
    let mut ks = ApiKeys::default();
    ks.add("hgk_user", "laptop".into(), vec![Scope::Chat]);
    let internal = "hgk_internal";
    ks.add_internal(internal, "jigglebot (internal)".into(), vec![Scope::Chat]);
    ks.save(&path).unwrap();

    // Reload from disk: the user key survives, the internal token is GONE.
    let reloaded = ApiKeys::load(&path).unwrap();
    assert_eq!(reloaded.iter().count(), 1, "only the user key persisted");
    assert!(reloaded.authorizes("hgk_user", Scope::Chat));
    assert!(
        !reloaded.authorizes(internal, Scope::Chat),
        "the hidden internal token must never reach disk"
    );
    // And the raw file must not contain the internal digest.
    let raw = std::fs::read_to_string(&path).unwrap();
    assert!(
        !raw.contains(&hash_token(internal)),
        "internal digest leaked to disk"
    );
    assert!(
        !raw.contains("hidden"),
        "hidden flag elided from persisted keys"
    );
}

/// [codex r1] `add_internal` must REFUSE an empty/whitespace token so it never becomes a
/// credential. An empty registration would push a hidden key whose digest is
/// `hash_token("")`; with auth on, `authorizes("")` would then match it — a trivially-open
/// admin bypass for a caller (embedder) that fumbles its token value. Not reachable over the
/// `/v1` bearer path (header normalization keeps a client from presenting an empty bearer),
/// so this is proved at the keys layer.
///
/// Fail-on-revert: remove the empty-token guard in `add_internal` and the empty key is
/// pushed, so `authorizes("", Admin)` returns true and the count is 2 — both asserts flip.
#[test]
fn empty_internal_token_is_refused_not_a_credential() {
    // A real Admin key arms auth (store non-empty ⇒ the bearer path is enforced).
    let mut ks = ApiKeys::default();
    ks.add("hgk_real_admin", "admin".into(), vec![Scope::Admin]);
    assert!(!ks.is_empty(), "a real key armed auth");

    // An EMPTY internal-token registration is a no-op (no credential armed).
    ks.add_internal("", "jigglebot (internal)".into(), vec![Scope::Admin]);
    assert!(
        ks.authorizes("hgk_real_admin", Scope::Admin),
        "the real admin token still authorizes"
    );
    assert!(
        !ks.authorizes("", Scope::Admin),
        "an empty token must NOT authorize — no empty-digest credential was armed"
    );
    assert_eq!(
        ks.iter().count(),
        1,
        "the empty internal token was not added"
    );

    // Whitespace-only is likewise refused.
    ks.add_internal("   ", "ws".into(), vec![Scope::Admin]);
    assert!(
        !ks.authorizes("   ", Scope::Admin),
        "a whitespace token is not a credential"
    );
    assert_eq!(
        ks.iter().count(),
        1,
        "the whitespace internal token was not added"
    );

    // A real internal token IS still registered (the guard only rejects empty/whitespace).
    ks.add_internal(
        "secret-internal-token",
        "jigglebot (internal)".into(),
        vec![Scope::Admin],
    );
    assert!(
        ks.authorizes("secret-internal-token", Scope::Admin),
        "a non-empty internal token authorizes"
    );
    assert_eq!(
        ks.iter().count(),
        2,
        "the non-empty internal token was added"
    );
}
