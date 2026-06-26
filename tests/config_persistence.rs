//! Black-box integration coverage for the per-instance `config.json` store
//! ([`higgs::config::InstanceConfig`]) reached through the HTTP control surface
//! of a PLAIN (non-hub) server.
//!
//! The hub-mode local/remote relabel is already exercised by
//! `hub_server::hub_server_relabel_local_and_remote`; this file targets the
//! SAME `config.rs` save/load/merge code on a bare server (no fleet installed),
//! plus the settings read-back paths that drive the in-memory runtime flags.
//! Each test renames the local instance and/or flips a setting and reads it
//! straight back over HTTP — proving the value round-tripped through
//! `InstanceConfig::{load,save}` (the rename) or the runtime setters (settings).
//!
//! Ports: 13000 base. Each test spawns the real `higgs` binary with a temp
//! `HIGGS_HOME` (so the config.json under test is the harness's, never the
//! developer's `~/.higgs`) and SIGTERMs it on drop. No SSE stream is opened.

mod common;

use common::{spawn_with_tiny_model, tiny_gguf_path};

/// The local node's label in `GET /api/higgs/nodes`, loaded from `config.json`.
async fn local_label(c: &reqwest::Client, base: &str) -> String {
    let nodes: serde_json::Value = c
        .get(format!("{base}/api/higgs/nodes"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    nodes
        .as_array()
        .and_then(|a| a.iter().find(|n| n["is_local"] == true))
        .and_then(|n| n["label"].as_str())
        .expect("local node has a label")
        .to_string()
}

/// Renaming the LOCAL instance on a PLAIN (no-hub) server persists to `config.json`
/// (`control_nodes_label("local")` → `with_config_mut` → `InstanceConfig::save`) and
/// the next `GET /api/higgs/nodes` reads it back (→ `InstanceConfig::load`). This is the
/// non-hub branch of the local-label path — the hub e2e covers it only with a fleet up.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn local_rename_persists_and_round_trips_without_a_hub() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("SKIP local_rename_persists_and_round_trips_without_a_hub: tiny gguf not found");
        return;
    };
    let srv = spawn_with_tiny_model(13000, &gguf).await;
    let c = reqwest::Client::new();

    // The server generated a non-empty friendly name at boot (`name_or_init` →
    // `config.json`). It always begins with the role prefix.
    let booted = local_label(&c, &srv.base).await;
    assert!(!booted.is_empty(), "boot generated a name: {booted:?}");

    // Rename "local" — drives with_config_mut → InstanceConfig::{load,save}.
    let resp = c
        .post(format!("{}/api/higgs/nodes/label", srv.base))
        .json(&serde_json::json!({ "node": "local", "label": "studio-alpha" }))
        .send()
        .await
        .unwrap();
    assert!(
        resp.status().is_success(),
        "local relabel ok: {}",
        resp.status()
    );
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["status"], "ok", "relabel returns ok: {body}");

    // The next nodes read loads config.json fresh → reflects the new name.
    assert_eq!(
        local_label(&c, &srv.base).await,
        "studio-alpha",
        "renamed local instance persisted + read back from config.json"
    );

    // A SECOND rename overwrites the first (load-modify-save replaces `name`), not
    // appends — the round-trip is idempotent and last-write-wins.
    let resp = c
        .post(format!("{}/api/higgs/nodes/label", srv.base))
        .json(&serde_json::json!({ "node": "local", "label": "studio-beta" }))
        .send()
        .await
        .unwrap();
    assert!(resp.status().is_success(), "second relabel ok");
    assert_eq!(
        local_label(&c, &srv.base).await,
        "studio-beta",
        "second rename replaced the first in config.json"
    );

    // An EMPTY local label is written verbatim (the empty-label clear is a REMOTE-only
    // behavior; for "local" it just sets `name = ""`). The view then falls back to the
    // "this machine" sentinel because `control_nodes` filters out an empty config name.
    let resp = c
        .post(format!("{}/api/higgs/nodes/label", srv.base))
        .json(&serde_json::json!({ "node": "local", "label": "" }))
        .send()
        .await
        .unwrap();
    assert!(resp.status().is_success(), "empty local label accepted");
    assert_eq!(
        local_label(&c, &srv.base).await,
        "this machine",
        "empty config name → 'this machine' fallback in the nodes view"
    );
}

/// `logs/settings` round-trips ALL THREE Developer-Log toggles — crucially
/// `show_log_fields` (the `#[serde(default)]` field the existing log-settings test
/// never flips) and `log_incoming_tokens` — so a PUT that sets every flag is read back
/// by a following GET. Exercises the full LogSettings get/set wire path on a plain server.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn log_settings_round_trip_every_toggle() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("SKIP log_settings_round_trip_every_toggle: tiny gguf not found");
        return;
    };
    let srv = spawn_with_tiny_model(13001, &gguf).await;
    let c = reqwest::Client::new();

    let before: serde_json::Value = c
        .get(format!("{}/api/higgs/logs/settings", srv.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    // All three toggle fields are present in the GET shape.
    for k in ["verbose", "log_incoming_tokens", "show_log_fields"] {
        assert!(before[k].is_boolean(), "{k} present as a bool: {before}");
    }

    // Flip EVERY toggle to the negation of its current value, then PUT.
    let mut put_body = before.clone();
    for k in ["verbose", "log_incoming_tokens", "show_log_fields"] {
        put_body[k] = serde_json::Value::Bool(!before[k].as_bool().unwrap());
    }
    let put = c
        .put(format!("{}/api/higgs/logs/settings", srv.base))
        .json(&put_body)
        .send()
        .await
        .unwrap();
    assert!(put.status().is_success(), "PUT logs/settings ok");

    // GET reflects every flipped toggle (the setters took effect).
    let after: serde_json::Value = c
        .get(format!("{}/api/higgs/logs/settings", srv.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    for k in ["verbose", "log_incoming_tokens", "show_log_fields"] {
        assert_eq!(
            after[k], put_body[k],
            "{k} toggle persisted across GET: {after}"
        );
    }

    // A PUT body that OMITS show_log_fields still deserializes (it's #[serde(default)])
    // and is accepted, clearing that flag back to false.
    let put = c
        .put(format!("{}/api/higgs/logs/settings", srv.base))
        .json(&serde_json::json!({ "verbose": false, "log_incoming_tokens": false }))
        .send()
        .await
        .unwrap();
    assert!(
        put.status().is_success(),
        "PUT omitting show_log_fields defaults it: {}",
        put.status()
    );
    let after: serde_json::Value = c
        .get(format!("{}/api/higgs/logs/settings", srv.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        after["show_log_fields"], false,
        "omitted show_log_fields defaulted to false: {after}"
    );
}

/// `GET`/`PUT /api/higgs/settings` round-trips the runtime server-behavior flags
/// (jit/auto-unload/serving) — flipping the booleans and confirming each is read back.
/// (The idle-TTL number path is covered elsewhere; this nails the three boolean setters.)
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn runtime_settings_boolean_flags_round_trip() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("SKIP runtime_settings_boolean_flags_round_trip: tiny gguf not found");
        return;
    };
    let srv = spawn_with_tiny_model(13002, &gguf).await;
    let c = reqwest::Client::new();

    let before: serde_json::Value = c
        .get(format!("{}/api/higgs/settings", srv.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    for k in ["jit_enabled", "auto_unload_idle", "serving_enabled"] {
        assert!(before[k].is_boolean(), "{k} present as a bool: {before}");
    }
    assert!(
        before["idle_ttl_minutes"].is_number(),
        "idle_ttl_minutes present: {before}"
    );

    // Negate every boolean flag (keep the TTL number unchanged), PUT, read back.
    let mut put_body = before.clone();
    for k in ["jit_enabled", "auto_unload_idle", "serving_enabled"] {
        put_body[k] = serde_json::Value::Bool(!before[k].as_bool().unwrap());
    }
    let put = c
        .put(format!("{}/api/higgs/settings", srv.base))
        .json(&put_body)
        .send()
        .await
        .unwrap();
    assert!(put.status().is_success(), "PUT settings ok");

    let after: serde_json::Value = c
        .get(format!("{}/api/higgs/settings", srv.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    for k in ["jit_enabled", "auto_unload_idle", "serving_enabled"] {
        assert_eq!(after[k], put_body[k], "{k} flag persisted: {after}");
    }
    assert_eq!(
        after["idle_ttl_minutes"], before["idle_ttl_minutes"],
        "untouched TTL unchanged: {after}"
    );

    // Restore serving_enabled to true so graceful shutdown stays clean and no /v1
    // surface is left disabled (defensive; no SSE stream is open here).
    let mut restore = after.clone();
    restore["serving_enabled"] = serde_json::Value::Bool(true);
    let _ = c
        .put(format!("{}/api/higgs/settings", srv.base))
        .json(&restore)
        .send()
        .await
        .unwrap();
}
