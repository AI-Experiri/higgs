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

/// `set_cors_origins` validates + CANONICALIZES + dedups + persists the extra
/// CORS-origins allowlist to `config.json` (via `with_config_mut` →
/// `InstanceConfig::save`), `cors_settings` reads it back. This helper does NOT
/// serve `/v1`, so no CORS layer is ever built and `applied` stays `None` — hence
/// `restart_required` is `false` (the first serve start would apply the persisted
/// list; nothing is pending). An invalid entry is rejected with the `[HG071]`
/// diagnostic and NOTHING is persisted. Fail-on-revert: neutering the
/// `with_config_mut` write in `set_cors_origins` (so nothing lands in config.json)
/// fails the "config.json contains them" assertion; reverting the canonicalization
/// (storing the raw input) fails the `https://EXAMPLE.com → https://example.com`
/// assertion.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cors_origins_persist_and_flag_restart_required() {
    let Some(higgs) = higgs_local(&[common::TINY_MODEL_ID]).await else {
        eprintln!("SKIP cors_origins_persist_and_flag_restart_required: tiny gguf not found");
        return;
    };

    // Fresh instance: no extra origins persisted, none applied → no restart pending.
    let initial = higgs.cors_settings();
    assert!(initial.origins.is_empty(), "no extra origins at boot");
    assert!(
        initial.applied_origins.is_empty(),
        "nothing applied pre-serve"
    );
    assert!(!initial.restart_required, "nothing pending");

    // Persist a list carrying a duplicate and a NON-canonical entry
    // (`https://EXAMPLE.com`, uppercased host) — it is canonicalized to the exact
    // string a browser sends (`https://example.com`) and deduped (first-seen order
    // kept).
    let updated = higgs
        .set_cors_origins(vec![
            "https://EXAMPLE.com".to_string(),
            "http://localhost:5173".to_string(),
            "https://example.com".to_string(),
        ])
        .expect("valid origins persist");
    assert_eq!(
        updated.origins,
        vec![
            "https://example.com".to_string(),
            "http://localhost:5173".to_string()
        ],
        "host canonicalized (lowercased); canonically-equal duplicate dropped, order preserved"
    );
    assert!(
        !updated.restart_required,
        "no CORS layer built (helper never serves /v1) → nothing applied → no restart pending"
    );

    // The persisted config.json ACTUALLY contains the origins (the with_config_mut
    // write reached disk). Read the raw file — the proof the store round-tripped.
    let config_json = std::fs::read_to_string(higgs.home().join("config.json"))
        .expect("config.json written to HIGGS_HOME");
    let parsed: serde_json::Value =
        serde_json::from_str(&config_json).expect("config.json is valid JSON");
    let persisted: Vec<String> = parsed["cors_origins"]
        .as_array()
        .expect("cors_origins array present")
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    assert_eq!(
        persisted,
        vec![
            "https://example.com".to_string(),
            "http://localhost:5173".to_string()
        ],
        "config.json on disk carries the CANONICALIZED, deduped origins"
    );

    // A fresh read reflects the persisted state.
    assert_eq!(higgs.cors_settings().origins, updated.origins);

    // An invalid origin is rejected with [HG071] and NOTHING new is persisted.
    let err = higgs
        .set_cors_origins(vec![
            "https://ok.example".to_string(),
            "notaurl".to_string(),
        ])
        .expect_err("invalid entry rejected");
    let msg = err.to_string();
    assert!(
        msg.contains("HG071"),
        "rejection carries the HG071 code: {msg}"
    );
    assert_eq!(
        higgs.cors_settings().origins,
        updated.origins,
        "rejected write left the previously-persisted allowlist untouched"
    );

    higgs.shutdown().await;
}

/// The CORS disclosures follow the LIVE list (G7): an API write applies to the
/// running layer immediately (`restart_required` stays false, `applied`
/// equals what was written), building ANOTHER router for the same `Higgs`
/// (the pub embedder constructor `serve::v1_router`) perturbs nothing, and
/// once the last listener exits the disclosures return to honest pre-serve
/// semantics rather than claiming a dead listener's state.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn second_router_build_does_not_clobber_applied_cors() {
    let Some(higgs) = higgs_local(&[common::TINY_MODEL_ID]).await else {
        eprintln!("SKIP second_router_build_does_not_clobber_applied_cors: tiny gguf not found");
        return;
    };

    // Go live on an ephemeral loopback port — serve_v1 captures the boot-applied
    // extra origins (empty here) as it puts the listener behind the layer.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind loopback");
    let addr = listener.local_addr().expect("listener addr");
    let (stop_tx, stop_rx) = tokio::sync::oneshot::channel::<()>();
    let serve_higgs = std::sync::Arc::clone(&higgs);
    let server = tokio::spawn(async move {
        higgs::serve::serve_v1(serve_higgs, listener, async {
            let _ = stop_rx.await;
        })
        .await
    });
    // The applied capture happens BEFORE the listener starts accepting, so the
    // first successful /health proves it has been recorded.
    let health = format!("http://{addr}/health");
    let mut live = false;
    for _ in 0..250 {
        if reqwest::get(&health)
            .await
            .is_ok_and(|r| r.status().is_success())
        {
            live = true;
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    assert!(live, "serve_v1 came up on {addr}");

    // While live, the disclosed `bind_host` is the RECORDED actual listener
    // address (ip + the ephemeral port — provably not the hardcoded loopback
    // default, which carries no port). The embedder owns the listener, so a
    // hardcoded claim would misreport a LAN (0.0.0.0) bind.
    assert_eq!(
        higgs.server_config().bind_host,
        addr.to_string(),
        "server_config discloses the live bound address"
    );

    // Persist a NEW allowlist while live: G7 publishes it to the live list the
    // running layer reads per request — applied immediately, nothing pending.
    let updated = higgs
        .set_cors_origins(vec!["https://tools.example".to_string()])
        .expect("valid origin persists");
    assert!(
        !updated.restart_required,
        "the write applied to the running layer — no restart pending (G7)"
    );
    assert_eq!(
        updated.applied_origins,
        vec!["https://tools.example".to_string()],
        "the applied disclosure equals what was just written"
    );

    // Build a second router for the same instance WITHOUT serving it — the pub
    // embedder constructor. It must not perturb the live disclosures.
    let _second = higgs::serve::v1_router(std::sync::Arc::clone(&higgs));
    let after = higgs.cors_settings();
    assert!(
        !after.restart_required,
        "a built-but-never-served router changes nothing"
    );
    assert_eq!(
        after.applied_origins,
        vec!["https://tools.example".to_string()],
        "the live disclosure survives a router build"
    );

    let _ = stop_tx.send(());
    server
        .await
        .expect("serve task join")
        .expect("serve_v1 result");

    // After the listener is gone the serve-scoped disclosures are CLEARED:
    // `bind_host` returns to the loopback default (no dead ip:port claim) and
    // the CORS settings return to pre-serve semantics (no stale applied
    // snapshot claiming a restart the next serve start would make moot).
    let post = higgs.cors_settings();
    assert!(
        !higgs.server_config().bind_host.contains(':'),
        "post-shutdown bind_host is the host-only loopback default, not a dead ip:port"
    );
    assert!(
        post.applied_origins.is_empty() && !post.restart_required,
        "post-shutdown CORS settings report pre-serve semantics"
    );

    higgs.shutdown().await;
}

/// `lan_exposed` is SERVE-SCOPED: it exists so a revoke that would EMPTY the
/// keystore is refused ([HG059]) while a LAN listener is live (revoke-to-empty
/// would reopen an exposed surface). Once the listener is gone there is nothing
/// to reopen, so a facade that outlives its serve must let that revoke through —
/// otherwise an embedder that stops `serve_v1` can never delete its last key.
/// The START-side guards ([HG058] keyless-LAN, [HG069] no-Admin-LAN) re-check the
/// keystore on every serve, so clearing this on teardown loses no safety.
///
/// Fail-on-revert: make `ServeGuard`'s deregistration keep the slot's `lan` flag
/// (e.g. skip removing the slot in `Higgs::deregister_serve`) and the final revoke
/// fails with `[HG059]`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn lan_exposure_clears_when_the_listener_goes_away() {
    let Some(higgs) = higgs_local(&[common::TINY_MODEL_ID]).await else {
        eprintln!("SKIP lan_exposure_clears_when_the_listener_goes_away: tiny gguf not found");
        return;
    };

    // A non-loopback bind needs an Admin key ([HG058]/[HG069] refuse otherwise).
    // This is the SOLE key, so revoking it later empties the store — exactly the
    // operation [HG059] refuses while LAN-exposed.
    higgs
        .mint_key("lan-admin", Some(vec![higgs::keys::Scope::Admin]))
        .expect("admin key mints");

    // Serve on a NON-loopback address (ephemeral port) → `lan_exposed` goes true.
    let listener = tokio::net::TcpListener::bind("0.0.0.0:0")
        .await
        .expect("bind 0.0.0.0");
    let addr = listener.local_addr().expect("listener addr");
    let (stop_tx, stop_rx) = tokio::sync::oneshot::channel::<()>();
    let serve_higgs = std::sync::Arc::clone(&higgs);
    let server = tokio::spawn(async move {
        higgs::serve::serve_v1(serve_higgs, listener, async {
            let _ = stop_rx.await;
        })
        .await
    });
    let health = format!("http://127.0.0.1:{}/health", addr.port());
    let mut live = false;
    for _ in 0..250 {
        if reqwest::get(&health)
            .await
            .is_ok_and(|r| r.status().is_success())
        {
            live = true;
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    assert!(live, "serve_v1 came up on {addr}");

    // While the LAN listener is LIVE, emptying the keystore is refused [HG059].
    let err = higgs
        .revoke_key("lan-admin")
        .expect_err("last-key revoke refused while LAN-exposed");
    assert!(
        err.to_string().contains("HG059"),
        "live LAN revoke carries HG059: {err}"
    );

    // Tear the listener down: the exposure is over.
    let _ = stop_tx.send(());
    server
        .await
        .expect("serve task join")
        .expect("serve_v1 result");

    // The SAME revoke now succeeds — no phantom LAN gate outlives the listener.
    let removed = higgs
        .revoke_key("lan-admin")
        .expect("last-key revoke allowed once the listener is gone");
    assert_eq!(removed.removed, 1, "the sole key was removed");

    higgs.shutdown().await;
}

/// `serve_v1` is public and an embedder may run SEVERAL listeners on one facade.
/// Shutting ONE down must not tear the shared facade down under its siblings: only
/// the LAST listener to leave owns the teardown (`higgs.stop()`). Before the fix,
/// every `serve_v1` exit called `stop()`, draining the local node's workers while a
/// sibling listener was still accepting requests.
///
/// Fail-on-revert: make `serve_v1` call `higgs.stop()` unconditionally (instead of
/// only when `serve_guard.release()` reports it was the last listener) and the
/// resident-model assertion below fails — the sibling's workers are drained.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn stopping_one_listener_leaves_its_sibling_serving() {
    let Some(higgs) = higgs_local(&[common::TINY_MODEL_ID]).await else {
        eprintln!("SKIP stopping_one_listener_leaves_its_sibling_serving: tiny gguf not found");
        return;
    };

    // Two loopback listeners on the SAME facade.
    let mut ports = Vec::new();
    let mut stops = Vec::new();
    let mut tasks = Vec::new();
    for _ in 0..2 {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind loopback");
        ports.push(listener.local_addr().expect("addr").port());
        let (stop_tx, stop_rx) = tokio::sync::oneshot::channel::<()>();
        stops.push(stop_tx);
        let serve_higgs = std::sync::Arc::clone(&higgs);
        tasks.push(tokio::spawn(async move {
            higgs::serve::serve_v1(serve_higgs, listener, async {
                let _ = stop_rx.await;
            })
            .await
        }));
    }

    let healthy = |port: u16| async move {
        let url = format!("http://127.0.0.1:{port}/health");
        reqwest::get(&url)
            .await
            .is_ok_and(|r| r.status().is_success())
    };
    for port in &ports {
        let mut live = false;
        for _ in 0..250 {
            if healthy(*port).await {
                live = true;
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        assert!(live, "listener on {port} came up");
    }

    // Load a model: `Higgs::stop()` unloads every resident worker, so a still-loaded
    // model after the sibling's shutdown proves the facade survived.
    higgs
        .load(common::TINY_MODEL_ID, None)
        .await
        .expect("tiny model loads");
    assert_eq!(
        higgs.local_served_ids().await,
        vec![common::TINY_MODEL_ID.to_string()],
        "the model is resident before any listener stops"
    );

    // Stop the FIRST listener only. Its `serve_v1` must NOT stop the facade.
    let first_stop = stops.remove(0);
    let _ = first_stop.send(());
    tasks
        .remove(0)
        .await
        .expect("first serve join")
        .expect("first serve result");

    // The sibling is still serving...
    assert!(
        healthy(ports[1]).await,
        "the surviving listener still answers after its sibling stopped"
    );
    // ...on a facade that was NOT torn down. `Higgs::stop()` drains the local node,
    // unloading every resident worker — so the model loaded before the shutdown must
    // STILL be resident. This is the assertion that actually detects the defect;
    // /health alone does not (axum's other listener keeps answering regardless).
    assert_eq!(
        higgs.local_served_ids().await,
        vec![common::TINY_MODEL_ID.to_string()],
        "a sibling listener's shutdown must not drain the shared facade's workers"
    );

    // Now stop the last one — it owns the teardown.
    let _ = stops.remove(0).send(());
    tasks
        .remove(0)
        .await
        .expect("second serve join")
        .expect("second serve result");

    higgs.shutdown().await;
}

/// The [HG058]/[HG069] LAN startup refusals tear the facade down before returning
/// so a rejected serve doesn't LEAK a worker the embedder already started — but
/// only when no listener is already live. With a SIBLING serving, that drain would
/// strand it on a stopped node.
///
/// Fail-on-revert: make the [HG058] branch call `higgs.stop()` unconditionally
/// (instead of only when `serve_guard.release()` reports it was the last listener)
/// and the resident-model assertion below fails.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_refused_lan_serve_does_not_drain_a_live_sibling() {
    let Some(higgs) = higgs_local(&[common::TINY_MODEL_ID]).await else {
        eprintln!("SKIP a_refused_lan_serve_does_not_drain_a_live_sibling: tiny gguf not found");
        return;
    };

    // A loopback listener is live, with a model resident on the shared facade.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind loopback");
    let port = listener.local_addr().expect("addr").port();
    let (stop_tx, stop_rx) = tokio::sync::oneshot::channel::<()>();
    let serve_higgs = std::sync::Arc::clone(&higgs);
    let server = tokio::spawn(async move {
        higgs::serve::serve_v1(serve_higgs, listener, async {
            let _ = stop_rx.await;
        })
        .await
    });
    let health = format!("http://127.0.0.1:{port}/health");
    let mut live = false;
    for _ in 0..250 {
        if reqwest::get(&health)
            .await
            .is_ok_and(|r| r.status().is_success())
        {
            live = true;
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    assert!(live, "the loopback listener came up");
    higgs
        .load(common::TINY_MODEL_ID, None)
        .await
        .expect("tiny model loads");

    // Now attempt a NON-loopback serve with an EMPTY keystore → refused [HG058].
    let lan = tokio::net::TcpListener::bind("0.0.0.0:0")
        .await
        .expect("bind 0.0.0.0");
    let err = higgs::serve::serve_v1(
        std::sync::Arc::clone(&higgs),
        lan,
        std::future::pending::<()>(),
    )
    .await
    .expect_err("keyless LAN bind is refused");
    assert!(
        err.to_string().contains("HG058"),
        "refusal carries the HG058 code: {err}"
    );

    // The refusal must NOT have drained the facade the loopback listener owns.
    assert_eq!(
        higgs.local_served_ids().await,
        vec![common::TINY_MODEL_ID.to_string()],
        "a refused LAN serve must not drain a live sibling's workers"
    );
    assert!(
        reqwest::get(&health)
            .await
            .is_ok_and(|r| r.status().is_success()),
        "the loopback listener still serves after the refusal"
    );

    let _ = stop_tx.send(());
    server.await.expect("serve join").expect("serve result");
    higgs.shutdown().await;
}

/// `config.json` is a plain file an operator can hand-edit, and `cors_origins`
/// predates the [HG071] validation. A legacy/manual entry like
/// `https://EXAMPLE.com:443` would be built into the CORS layer verbatim and never
/// match a browser's `Origin: https://example.com` — while being disclosed as
/// applied-and-in-sync. Reads CANONICALIZE (and drop invalid entries), so what we
/// enforce, what we disclose, and what a browser sends are the same string.
///
/// Fail-on-revert: make `Higgs::extra_cors_origins` return the raw `cors_origins`
/// and the canonical-form assertion below fails.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn hand_edited_cors_origins_are_canonicalized_on_read() {
    let Some(higgs) = higgs_local(&[common::TINY_MODEL_ID]).await else {
        eprintln!("SKIP hand_edited_cors_origins_are_canonicalized_on_read: tiny gguf not found");
        return;
    };

    // Seed config.json the way a hand-edit / older higgs would: a non-canonical
    // origin, a duplicate of it in canonical form, and an outright invalid entry.
    let path = higgs.home().join("config.json");
    let mut cfg: serde_json::Value = std::fs::read_to_string(&path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_else(|| serde_json::json!({}));
    cfg["cors_origins"] = serde_json::json!([
        "https://EXAMPLE.com:443", // canonicalizes to https://example.com
        "https://example.com",     // the same origin, already canonical → deduped
        "not a url",               // invalid → dropped with a warning, not fatal
    ]);
    std::fs::write(&path, serde_json::to_string_pretty(&cfg).unwrap()).expect("seed config.json");

    // The disclosed list is the canonical, deduped one the layer would enforce.
    let settings = higgs.cors_settings();
    assert_eq!(
        settings.origins,
        vec!["https://example.com".to_string()],
        "hand-edited origins are canonicalized + deduped, invalid entries dropped"
    );

    higgs.shutdown().await;
}
