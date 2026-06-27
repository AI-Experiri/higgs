use super::*;

fn id_str() -> String {
    "aa".repeat(32)
}

#[test]
fn add_contains_remove_roundtrip() {
    let dir = std::env::temp_dir().join("higgs-allow-roundtrip-test");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("pairings.json");
    let _ = std::fs::remove_file(&path);

    let mut allow = Allowlist::load(&path).unwrap();
    assert!(!allow.contains(&id_str()));
    allow.add(id_str(), Some("studio-mac".into())).unwrap();
    assert!(allow.contains(&id_str()));
    assert_eq!(allow.len(), 1);

    // persisted: a fresh load sees it
    let reloaded = Allowlist::load(&path).unwrap();
    assert!(reloaded.contains(&id_str()));

    // ids() enumerates paired ids (for fleet seeding).
    allow.add(id_str(), None).unwrap();
    assert_eq!(allow.ids(), vec![id_str()], "ids lists the paired node");

    allow.remove(&id_str()).unwrap();
    assert!(!allow.contains(&id_str()));
    assert!(allow.is_empty());
    assert!(allow.ids().is_empty(), "ids empty after removal");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn load_corrupt_pairings_is_hg041_with_remediation() {
    // A present-but-unparseable pairings.json is corruption (HG041) — fail loudly
    // with the code + "delete to reset" rather than silently emptying the allowlist.
    let dir = std::env::temp_dir().join("higgs-allow-corrupt-test");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("pairings.json");
    std::fs::write(&path, b"not json at all").unwrap();
    // `Allowlist` isn't `Debug`, so destructure rather than `unwrap_err()`.
    let Err(err) = Allowlist::load(&path) else {
        panic!("corrupt pairings must fail to load");
    };
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    let msg = err.to_string();
    assert!(msg.contains("[HG041]"), "corruption carries HG041: {msg}");
    assert!(
        msg.contains("delete the file to reset"),
        "carries the remediation: {msg}"
    );
    let _ = std::fs::remove_file(&path);
}

#[test]
fn relabel_updates_only_existing_ids() {
    let dir = std::env::temp_dir().join("higgs-allow-relabel-test");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("pairings.json");
    let _ = std::fs::remove_file(&path);

    let mut allow = Allowlist::load(&path).unwrap();
    allow.add(id_str(), Some("old".into())).unwrap();

    // Rename an existing node → true, persisted.
    assert!(allow.relabel(&id_str(), Some("new".into())).unwrap());
    assert_eq!(
        Allowlist::load(&path).unwrap().label(&id_str()).as_deref(),
        Some("new")
    );
    // Clearing the label (None) is allowed.
    assert!(allow.relabel(&id_str(), None).unwrap());
    assert_eq!(allow.label(&id_str()), None);
    // Relabeling an UNKNOWN id is a no-op false — never an insert.
    assert!(!allow
        .relabel("bb".repeat(32).as_str(), Some("x".into()))
        .unwrap());
    assert!(!allow.contains(&"bb".repeat(32)));
    let _ = std::fs::remove_file(&path);
}

#[test]
fn labels_maps_ids_to_their_labels() {
    let dir = std::env::temp_dir().join("higgs-allow-labels-test");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("pairings.json");
    let _ = std::fs::remove_file(&path);

    let mut allow = Allowlist::load(&path).unwrap();
    allow
        .add("aa".repeat(32), Some("node-a(box)".into()))
        .unwrap();
    allow.add("bb".repeat(32), None).unwrap();

    // The serve layer reads labels straight off disk when the hub is disabled.
    let labels = Allowlist::load(&path).unwrap().labels();
    assert_eq!(
        labels.get(&"aa".repeat(32)),
        Some(&Some("node-a(box)".into()))
    );
    assert_eq!(labels.get(&"bb".repeat(32)), Some(&None));
    let _ = std::fs::remove_file(&path);
}

#[test]
fn load_missing_file_is_empty() {
    let path = std::env::temp_dir().join("higgs-allow-absent-xyz.json");
    let _ = std::fs::remove_file(&path);
    let allow = Allowlist::load(&path).unwrap();
    assert!(allow.is_empty());
}

#[test]
fn failed_save_rolls_back_in_memory() {
    // Parent dir does not exist → load sees NotFound (empty), but save's write
    // fails, so add() must return Err AND leave the allowlist unchanged.
    let path = std::env::temp_dir()
        .join("higgs-allow-rollback-missing-dir")
        .join("pairings.json");
    let _ = std::fs::remove_dir_all(path.parent().unwrap());
    let mut allow = Allowlist::load(&path).unwrap();
    let err = allow.add(id_str(), None).unwrap_err();
    assert!(err.kind() == std::io::ErrorKind::NotFound || err.raw_os_error().is_some());
    assert!(
        !allow.contains(&id_str()),
        "rolled back: not paired in memory"
    );
    assert!(allow.is_empty());
}

#[test]
fn token_mint_validate_burn() {
    let mut tokens = PairingTokens::new();
    let tok = tokens.mint(now_ms(), 600_000);
    assert!(tok.starts_with("htk_"));
    assert!(tokens.validate_and_burn(&tok, now_ms()).is_ok());
    // single-use: second validation fails
    assert_eq!(
        tokens.validate_and_burn(&tok, now_ms()),
        Err(TokenError::UnknownOrUsed)
    );
}

#[test]
fn validate_does_not_consume_until_burned() {
    let mut tokens = PairingTokens::new();
    let tok = tokens.mint(now_ms(), 600_000);
    // validate is idempotent — the token survives repeated peeks (e.g. a failed save)
    assert!(tokens.validate(&tok, now_ms()).is_ok());
    assert!(tokens.validate(&tok, now_ms()).is_ok());
    tokens.burn(&tok);
    // now consumed
    assert_eq!(
        tokens.validate(&tok, now_ms()),
        Err(TokenError::UnknownOrUsed)
    );
}

#[test]
fn token_expires() {
    let mut tokens = PairingTokens::new();
    let minted_at = 1_000_000u64;
    let tok = tokens.mint(minted_at, 600_000);
    assert_eq!(
        tokens.validate_and_burn(&tok, minted_at + 660_000),
        Err(TokenError::Expired)
    );
}

#[test]
fn unknown_token_is_rejected() {
    let mut tokens = PairingTokens::new();
    assert_eq!(
        tokens.validate_and_burn("htk_nope", now_ms()),
        Err(TokenError::UnknownOrUsed)
    );
}

fn now_ms() -> u64 {
    1_750_000_000_000
}

#[test]
fn load_io_error_is_hg040_with_kind_preserved() {
    // A read error that is NOT NotFound (here: the path is a directory, so
    // `std::fs::read` returns IsADirectory) is an HG040 persistence fault — the
    // code rides INSIDE the io::Error and the original kind is preserved for any
    // caller that matches on it. Distinct from the corruption (HG041) path.
    let dir = tempfile::tempdir().unwrap();
    let as_dir = dir.path().join("pairings-is-a-dir");
    std::fs::create_dir(&as_dir).unwrap();
    let Err(err) = Allowlist::load(&as_dir) else {
        panic!("reading a directory as pairings.json must fail");
    };
    assert_eq!(
        err.kind(),
        std::io::ErrorKind::IsADirectory,
        "kind preserved"
    );
    let msg = err.to_string();
    assert!(msg.contains("[HG040]"), "I/O fault carries HG040: {msg}");
    assert!(
        msg.contains("check free disk space"),
        "carries the HG040 remediation: {msg}"
    );
}

#[test]
fn path_returns_the_backing_file() {
    // The allowlist exposes its on-disk path so an HG040 names the real file.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("pairings.json");
    let allow = Allowlist::load(&path).unwrap();
    assert_eq!(allow.path(), path.as_path());
}

#[test]
fn failed_save_on_rename_is_hg040_and_rolls_back() {
    // write_tmp SUCCEEDS (the parent dir exists) but the final atomic rename FAILS
    // because the destination is itself a directory — exercising save's rename
    // map_err closure (remove the tmp + wrap as HG040). The in-memory map must roll
    // back, so authorization never diverges from disk.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("pairings-target");
    // Load while it's absent → empty allowlist that will persist to `path`.
    let mut allow = Allowlist::load(&path).unwrap();
    // Now make `path` a directory so the rename-over-it fails.
    std::fs::create_dir(&path).unwrap();

    let Err(err) = allow.add(id_str(), Some("x".into())) else {
        panic!("add must fail when the rename target is a directory");
    };
    assert_eq!(err.kind(), std::io::ErrorKind::IsADirectory);
    assert!(
        err.to_string().contains("[HG040]"),
        "rename failure carries HG040: {err}"
    );
    assert!(!allow.contains(&id_str()), "rolled back after failed save");
    assert!(allow.is_empty());
    // The temp file was cleaned up — only the directory remains in the parent.
    let leftovers: Vec<_> = std::fs::read_dir(dir.path())
        .unwrap()
        .filter_map(Result::ok)
        .filter(|e| {
            e.file_name()
                .to_string_lossy()
                .starts_with(".pairings.json.tmp")
        })
        .collect();
    assert!(leftovers.is_empty(), "the temp file was removed on failure");
}
