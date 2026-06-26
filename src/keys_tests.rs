
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
