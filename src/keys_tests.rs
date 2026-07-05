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
