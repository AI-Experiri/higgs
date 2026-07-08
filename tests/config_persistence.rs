//! In-process coverage for the per-instance `config.json` store
//! ([`higgs::config::InstanceConfig`]) and the runtime settings flags, reached
//! through the library-first `Higgs` crate API (the `/api/higgs/*` HTTP control
//! surface is deleted).
//!
//! The hub-mode local/remote relabel is already exercised by
//! `hub_server`; this file targets the SAME `config.rs` save/load/merge code on a
//! bare in-process instance (no fleet installed), plus the settings read-back
//! paths that drive the in-memory runtime flags. Each test renames the local
//! instance and/or flips a setting and reads it straight back — proving the value
//! round-tripped through `InstanceConfig::{load,save}` (the rename) or the runtime
//! setters (settings).
//!
//! Each test builds an in-process `Higgs` with the tiny model staged into an
//! isolated `HIGGS_HOME` (so the config.json under test is the harness's, never
//! the developer's `~/.higgs`) via `higgs_local`.

mod common;

use common::higgs_local;

/// The local node's label in `Higgs::nodes()`, loaded from `config.json`.
async fn local_label(higgs: &higgs::Higgs) -> String {
    let nodes = higgs.nodes().await;
    nodes
        .iter()
        .find(|n| n.is_local)
        .map(|n| n.label.clone())
        .expect("local node has a label")
}

/// Renaming the LOCAL instance on a PLAIN (no-hub) instance persists to `config.json`
/// (`node_label("local")` → `with_config_mut` → `InstanceConfig::save`) and the next
/// `nodes()` reads it back (→ `InstanceConfig::load`). This is the non-hub branch of the
/// local-label path — the hub e2e covers it only with a fleet up.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn local_rename_persists_and_round_trips_without_a_hub() {
    let Some(higgs) = higgs_local(&[common::TINY_MODEL_ID]).await else {
        eprintln!("SKIP local_rename_persists_and_round_trips_without_a_hub: tiny gguf not found");
        return;
    };

    // A fresh in-process instance has no config.json name yet, so the nodes view
    // falls back to the non-empty "this machine" sentinel.
    let booted = local_label(&higgs).await;
    assert!(!booted.is_empty(), "boot label non-empty: {booted:?}");

    // Rename "local" — drives with_config_mut → InstanceConfig::{load,save}.
    assert!(
        higgs.node_label("local", "studio-alpha").await.unwrap(),
        "local relabel returns Ok(true)"
    );

    // The next nodes read loads config.json fresh → reflects the new name.
    assert_eq!(
        local_label(&higgs).await,
        "studio-alpha",
        "renamed local instance persisted + read back from config.json"
    );

    // A SECOND rename overwrites the first (load-modify-save replaces `name`), not
    // appends — the round-trip is idempotent and last-write-wins.
    assert!(
        higgs.node_label("local", "studio-beta").await.unwrap(),
        "second relabel ok"
    );
    assert_eq!(
        local_label(&higgs).await,
        "studio-beta",
        "second rename replaced the first in config.json"
    );

    // An EMPTY local label is written verbatim (the empty-label clear is a REMOTE-only
    // behavior; for "local" it just sets `name = ""`). The view then falls back to the
    // "this machine" sentinel because `instance_name` filters out an empty config name.
    assert!(
        higgs.node_label("local", "").await.unwrap(),
        "empty local label accepted"
    );
    assert_eq!(
        local_label(&higgs).await,
        "this machine",
        "empty config name → 'this machine' fallback in the nodes view"
    );

    higgs.shutdown().await;
}

/// `logs_settings`/`set_logs_settings` round-trips ALL THREE Developer-Log toggles —
/// crucially `show_log_fields` (the `#[serde(default)]` field the existing log-settings
/// test never flips) and `log_incoming_tokens` — so a set of every flag is read back.
/// Exercises the full LogSettings get/set path on a plain instance.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn log_settings_round_trip_every_toggle() {
    let Some(higgs) = higgs_local(&[common::TINY_MODEL_ID]).await else {
        eprintln!("SKIP log_settings_round_trip_every_toggle: tiny gguf not found");
        return;
    };

    let before = higgs.logs_settings();

    // Flip EVERY toggle to the negation of its current value, then set.
    let flipped = higgs::LogSettings {
        verbose: !before.verbose,
        log_incoming_tokens: !before.log_incoming_tokens,
        show_log_fields: !before.show_log_fields,
    };
    higgs.set_logs_settings(&flipped);

    // Read reflects every flipped toggle (the setters took effect).
    let after = higgs.logs_settings();
    assert_eq!(after.verbose, flipped.verbose, "verbose toggle persisted");
    assert_eq!(
        after.log_incoming_tokens, flipped.log_incoming_tokens,
        "log_incoming_tokens toggle persisted"
    );
    assert_eq!(
        after.show_log_fields, flipped.show_log_fields,
        "show_log_fields toggle persisted"
    );

    // Setting show_log_fields back to false clears that flag (the original test drove this
    // via a PUT body that OMITTED the #[serde(default)] field; the wire default is gone, so
    // the typed equivalent is an explicit false — same observable end state).
    let cleared = higgs::LogSettings {
        verbose: false,
        log_incoming_tokens: false,
        show_log_fields: false,
    };
    higgs.set_logs_settings(&cleared);
    assert!(
        !higgs.logs_settings().show_log_fields,
        "show_log_fields cleared to false"
    );

    higgs.shutdown().await;
}

/// The runtime server-behavior flags (jit/auto-unload/serving) round-trip — flipping the
/// booleans and confirming each is read back. (The idle-TTL number path is covered
/// elsewhere; this nails the three boolean setters and that the TTL stays untouched.)
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn runtime_settings_boolean_flags_round_trip() {
    let Some(higgs) = higgs_local(&[common::TINY_MODEL_ID]).await else {
        eprintln!("SKIP runtime_settings_boolean_flags_round_trip: tiny gguf not found");
        return;
    };

    let jit0 = higgs.jit_enabled();
    let auto0 = higgs.auto_unload_idle();
    let serve0 = higgs.serving_enabled();
    let ttl0 = higgs.idle_ttl_minutes();

    // Negate every boolean flag (keep the TTL number unchanged), set, read back.
    higgs.set_jit_enabled(!jit0);
    higgs.set_auto_unload_idle(!auto0);
    higgs.set_serving_enabled(!serve0);

    assert_eq!(higgs.jit_enabled(), !jit0, "jit_enabled flag persisted");
    assert_eq!(
        higgs.auto_unload_idle(),
        !auto0,
        "auto_unload_idle flag persisted"
    );
    assert_eq!(
        higgs.serving_enabled(),
        !serve0,
        "serving_enabled flag persisted"
    );
    assert_eq!(higgs.idle_ttl_minutes(), ttl0, "untouched TTL unchanged");

    // Restore serving_enabled to true so teardown stays clean and no /v1 surface is left
    // disabled (defensive).
    higgs.set_serving_enabled(true);

    higgs.shutdown().await;
}
