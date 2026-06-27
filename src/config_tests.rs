use super::*;
use crate::worker::engine::GpuLayers;

#[test]
fn friendly_name_uses_role_eid8_and_host() {
    assert_eq!(
        friendly_name(Role::Node, "a1b2c3d4e5f6", "studio-mac"),
        "node-a1b2c3d4(studio-mac)"
    );
    assert_eq!(
        friendly_name(Role::Hub, "3f9a2b1c0000", "studio-mac"),
        "hub-3f9a2b1c(studio-mac)"
    );
}

#[test]
fn friendly_name_drops_parens_when_host_empty() {
    assert_eq!(friendly_name(Role::Node, "a1b2c3d4e5", ""), "node-a1b2c3d4");
}

#[test]
fn friendly_name_uses_short_id_whole() {
    // An id shorter than 8 chars is taken as-is (never panics on a short slice).
    assert_eq!(friendly_name(Role::Hub, "abc", "m"), "hub-abc(m)");
}

#[test]
fn load_missing_file_is_default() {
    let path = std::env::temp_dir().join("higgs-config-absent-xyz.json");
    let _ = std::fs::remove_file(&path);
    let cfg = InstanceConfig::load(&path).unwrap();
    assert!(cfg.name.is_empty());
}

#[test]
fn load_corrupt_config_is_hg041_with_remediation() {
    // A present-but-unparseable config.json is corruption (HG041), NOT a missing
    // file — so it fails loudly with the code + the "delete to reset" remediation.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("config.json");
    std::fs::write(&path, b"{ this is not json").unwrap();
    let err = InstanceConfig::load(&path).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    let msg = err.to_string();
    assert!(msg.contains("[HG041]"), "corruption carries HG041: {msg}");
    assert!(
        msg.contains("delete the file to reset"),
        "carries the remediation: {msg}"
    );
}

fn hub(id: &str, ticket: &str, label: &str, ts: u64) -> SavedHub {
    SavedHub {
        hub_id: id.into(),
        ticket: ticket.into(),
        label: label.into(),
        last_used_ms: ts,
    }
}

#[test]
fn remember_hub_dedups_by_id_and_sets_default() {
    let mut cfg = InstanceConfig::default();
    cfg.remember_hub(hub("aaa111", "tkt-old", "hub-a(srv)", 1));
    assert_eq!(cfg.default_hub.as_deref(), Some("aaa111"));
    assert_eq!(cfg.hubs.len(), 1);

    // Re-remembering the SAME hub_id replaces (newer ticket/label/ts win), no duplicate.
    cfg.remember_hub(hub("aaa111", "tkt-new", "hub-a2(srv)", 5));
    assert_eq!(cfg.hubs.len(), 1, "deduped by hub_id");
    assert_eq!(cfg.hubs[0].ticket, "tkt-new");
    assert_eq!(cfg.hubs[0].label, "hub-a2(srv)");

    // A different hub is added and becomes the new default.
    cfg.remember_hub(hub("bbb222", "tkt-b", "hub-b(box)", 9));
    assert_eq!(cfg.hubs.len(), 2);
    assert_eq!(cfg.default_hub.as_deref(), Some("bbb222"));
    assert_eq!(cfg.default_saved_hub().unwrap().ticket, "tkt-b");
}

#[test]
fn find_hub_by_label_then_id_prefix() {
    let mut cfg = InstanceConfig::default();
    cfg.remember_hub(hub("a1b2c3d4ffff", "tkt", "hub-a1b2c3d4(srv)", 1));
    // Exact label.
    assert_eq!(
        cfg.find_hub("hub-a1b2c3d4(srv)").map(|h| h.hub_id.as_str()),
        Some("a1b2c3d4ffff")
    );
    // Short id prefix.
    assert_eq!(
        cfg.find_hub("a1b2c3d4").map(|h| h.hub_id.as_str()),
        Some("a1b2c3d4ffff")
    );
    // No match.
    assert!(cfg.find_hub("nope").is_none());
}

#[test]
fn remove_hub_drops_and_repoints_default() {
    let mut cfg = InstanceConfig::default();
    cfg.remember_hub(hub("aaa", "tkt-a", "hub-a", 1));
    cfg.remember_hub(hub("bbb", "tkt-b", "hub-b", 9)); // default = bbb (last remembered)
    assert_eq!(cfg.default_hub.as_deref(), Some("bbb"));

    // Removing the default promotes the most-recently-used remaining hub (aaa).
    assert!(cfg.remove_hub("bbb"));
    assert_eq!(cfg.hubs.len(), 1);
    assert_eq!(cfg.default_hub.as_deref(), Some("aaa"), "default repointed");

    // Removing the last hub clears the default; removing an unknown id is a no-op.
    assert!(!cfg.remove_hub("ghost"));
    assert!(cfg.remove_hub("aaa"));
    assert!(cfg.hubs.is_empty());
    assert!(
        cfg.default_hub.is_none(),
        "default cleared with no hubs left"
    );
}

#[test]
fn default_saved_hub_none_when_id_dangles() {
    // A default_hub id with no matching entry resolves to None (defensive against a hand-
    // edited config), rather than panicking.
    let cfg = InstanceConfig {
        name: "node-x(box)".into(),
        hubs: vec![],
        default_hub: Some("ghost".into()),
        ..Default::default()
    };
    assert!(cfg.default_saved_hub().is_none());
}

#[test]
fn config_roundtrips_hubs_and_default() {
    let dir = std::env::temp_dir().join("higgs-config-hubs-roundtrip-test");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("config.json");
    let _ = std::fs::remove_file(&path);

    let mut cfg = InstanceConfig {
        name: "node-aa(box)".into(),
        ..Default::default()
    };
    cfg.remember_hub(hub("h1", "tkt1", "hub-1(srv)", 7));
    cfg.save(&path).unwrap();

    let back = InstanceConfig::load(&path).unwrap();
    assert_eq!(back.name, "node-aa(box)");
    assert_eq!(back.hubs, cfg.hubs);
    assert_eq!(back.default_hub.as_deref(), Some("h1"));
    let _ = std::fs::remove_file(&path);
}

#[test]
fn record_load_stores_and_replaces_per_model() {
    let mut cfg = InstanceConfig::default();
    assert!(cfg.model_record("org/m").is_none());

    let p1 = LoadParams::base(4096, GpuLayers::Count { n: 99 }, 0);
    cfg.record_load("org/m", p1.clone(), 1000);
    let rec = cfg.model_record("org/m").expect("record present");
    assert_eq!(rec.load.as_ref().unwrap().ctx_len(), 4096);
    assert_eq!(rec.last_loaded_ms, 1000);

    // A second load of the SAME id replaces the load info (latest wins), no duplicate key.
    let p2 = LoadParams::base(8192, GpuLayers::Count { n: 0 }, 0);
    cfg.record_load("org/m", p2, 2000);
    assert_eq!(cfg.models.len(), 1, "keyed by id, no duplicate");
    let rec = cfg.model_record("org/m").unwrap();
    assert_eq!(rec.load.as_ref().unwrap().ctx_len(), 8192);
    assert_eq!(rec.last_loaded_ms, 2000);
}

/// Back-compat: a `config.json` written before `LoadParams` became an engine
/// umbrella stored a tag-less flat `load` object. It must still deserialize
/// (mapped to the `LlamaCpp` variant), and a corrupt `load` value must degrade
/// to `None` rather than failing the whole config load.
#[test]
fn model_record_load_accepts_legacy_flat_and_degrades_garbage() {
    let legacy = r#"{"name":"node-x","models":{"org/m":{"load":{"ctx_len":2048,"gpu_layers":12,"threads":4},"last_loaded_ms":7}}}"#;
    let cfg: InstanceConfig = serde_json::from_str(legacy).unwrap();
    let rec = cfg.model_record("org/m").expect("legacy record present");
    assert_eq!(rec.load.as_ref().unwrap().ctx_len(), 2048);
    assert_eq!(
        rec.load.as_ref().unwrap().gpu_layers(),
        GpuLayers::Count { n: 12 }
    );
    assert_eq!(rec.last_loaded_ms, 7);

    // A non-object `load` value degrades to None (whole config still loads).
    let garbage = r#"{"name":"n","models":{"org/m":{"load":42,"last_loaded_ms":1}}}"#;
    let cfg2: InstanceConfig = serde_json::from_str(garbage).unwrap();
    assert!(cfg2.model_record("org/m").unwrap().load.is_none());
}

#[test]
fn config_roundtrips_model_records() {
    let dir = std::env::temp_dir().join("higgs-config-models-roundtrip-test");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("config.json");
    let _ = std::fs::remove_file(&path);

    let mut cfg = InstanceConfig {
        name: "node-aa(box)".into(),
        ..Default::default()
    };
    cfg.record_load(
        "org/m",
        LoadParams::base(2048, GpuLayers::Count { n: 12 }, 0),
        42,
    );
    cfg.save(&path).unwrap();

    let back = InstanceConfig::load(&path).unwrap();
    assert_eq!(back.models, cfg.models);
    assert_eq!(back.model_record("org/m").unwrap().last_loaded_ms, 42);
    assert_eq!(
        back.model_record("org/m")
            .unwrap()
            .load
            .as_ref()
            .unwrap()
            .ctx_len(),
        2048
    );
    let _ = std::fs::remove_file(&path);
}

#[test]
fn load_tolerates_unknown_future_fields() {
    // A config.json written by a NEWER build (with extra keys) must still load on this
    // build — unknown keys are ignored, so an upgrade/downgrade never wipes the name. A pure
    // hub's minimal `{"name":…}` (no hubs/default_hub) also defaults in cleanly.
    let dir = std::env::temp_dir().join("higgs-config-future-test");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("config.json");
    std::fs::write(
        &path,
        br#"{"name":"hub-3f9a2b1c(srv)","some_future_field":true}"#,
    )
    .unwrap();
    let cfg = InstanceConfig::load(&path).unwrap();
    assert_eq!(cfg.name, "hub-3f9a2b1c(srv)");
    assert!(cfg.hubs.is_empty());
    assert!(cfg.default_hub.is_none());
    let _ = std::fs::remove_file(&path);
}

#[test]
fn load_io_error_is_hg040_with_kind_preserved() {
    // A read error that is NOT NotFound (here: the path is a directory, so
    // `std::fs::read` returns IsADirectory) is an HG040 persistence fault, distinct
    // from the corruption (HG041) path. The original io kind is preserved.
    let dir = tempfile::tempdir().unwrap();
    let as_dir = dir.path().join("config-is-a-dir");
    std::fs::create_dir(&as_dir).unwrap();
    let err = InstanceConfig::load(&as_dir).unwrap_err();
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
fn save_rename_failure_cleans_up_temp() {
    // write_tmp SUCCEEDS (the parent dir exists) but the final atomic rename FAILS
    // because the destination is itself a directory — exercising save's rename
    // inspect_err closure (remove the stray temp file, surface the error).
    let dir = tempfile::tempdir().unwrap();
    let target = dir.path().join("config-target");
    std::fs::create_dir(&target).unwrap(); // rename(tmp, target) fails: target is a dir
    let cfg = InstanceConfig {
        name: "node-x(box)".into(),
        ..Default::default()
    };
    let err = cfg.save(&target).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::IsADirectory);
    // The temp file (.config.json.tmp.*) was removed; the parent holds only the dir.
    let strays: Vec<_> = std::fs::read_dir(dir.path())
        .unwrap()
        .filter_map(Result::ok)
        .filter(|e| {
            e.file_name()
                .to_string_lossy()
                .starts_with(".config.json.tmp")
        })
        .collect();
    assert!(strays.is_empty(), "the temp file was cleaned up on failure");
}

#[test]
fn save_write_tmp_failure_cleans_up_and_errors() {
    // `create_dir_all(dir)` succeeds on an already-present directory, but writing the
    // temp file inside a READ-ONLY directory fails — exercising save's write_tmp
    // failure arm (remove the partial temp, return the raw error). Skipped if the
    // process is root (perms are bypassed), so the assertion stays meaningful.
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().unwrap();
    let ro = dir.path().join("readonly");
    std::fs::create_dir(&ro).unwrap();
    std::fs::set_permissions(&ro, std::fs::Permissions::from_mode(0o555)).unwrap();
    let cfg = InstanceConfig {
        name: "node-y(box)".into(),
        ..Default::default()
    };
    let result = cfg.save(&ro.join("config.json"));
    // Restore perms so the TempDir can clean up regardless of the outcome.
    let _ = std::fs::set_permissions(&ro, std::fs::Permissions::from_mode(0o755));
    // Running as root bypasses the read-only bit; only assert when actually denied.
    if let Err(e) = result {
        assert_eq!(e.kind(), std::io::ErrorKind::PermissionDenied);
    }
}

#[test]
fn name_or_init_generates_then_reuses() {
    // Serialize with other HIGGS_HOME-mutating tests; restore the prior value after.
    let _env = crate::TEST_ENV_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let prev = std::env::var_os("HIGGS_HOME");
    let tmp = std::env::temp_dir().join("higgs-name-or-init-test");
    let _ = std::fs::remove_dir_all(&tmp);
    // SAFETY: serialized by TEST_ENV_LOCK; restored below.
    unsafe { std::env::set_var("HIGGS_HOME", &tmp) };

    // First call generates + persists.
    let first = name_or_init(Role::Node, "a1b2c3d4e5f6", "box").unwrap();
    assert_eq!(first, "node-a1b2c3d4(box)");
    // The name landed in config.json.
    let on_disk = InstanceConfig::load(&tmp.join("config.json")).unwrap();
    assert_eq!(on_disk.name, "node-a1b2c3d4(box)");

    // Second call reuses it verbatim — even with a DIFFERENT role/id/host (idempotent).
    let second = name_or_init(Role::Hub, "zzzzzzzz", "other").unwrap();
    assert_eq!(
        second, "node-a1b2c3d4(box)",
        "existing name is never overwritten"
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
