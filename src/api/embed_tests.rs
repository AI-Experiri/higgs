//! Unit tests for the Phase A1 embed-API facade methods (`api/embed.rs`).
//!
//! The pure helpers (`fit_generation_budget`, `estimate_prompt_bytes`) are tested
//! directly; the facade methods are driven over a stateful-fake-worker LOCAL node
//! (no llama.cpp), the same seam the sibling `api/tests.rs` uses.

use std::path::PathBuf;
use std::sync::Arc;

use serde_json::json;

use crate::api::{Higgs, HiggsConfig};
use crate::diagnostic::HiggsError;
use crate::keys::Scope;
use crate::worker::engine::CtxLen;

// ── Test seam ──────────────────────────────────────────────────────────────

/// A `Higgs` facade over a STATEFUL-fake-worker LOCAL node scanning `dirs`.
fn fake_higgs(dirs: Vec<PathBuf>) -> Higgs {
    let node = crate::node::test_support::fake_runtime_stateful(dirs.clone());
    let cfg = HiggsConfig {
        lmstudio_dirs: dirs,
        hf_dirs: vec![],
        ollama_dirs: vec![],
        default_load: HiggsConfig::default().default_load,
        worker_exe: None,
    };
    Higgs::with_local(Arc::new(node), cfg)
}

/// A `Higgs` scanning a fresh temp dir carrying one `org/model` GGUF fixture
/// (arch=llama, ctx_train=4096, chat template), plus the `TempDir` guard.
fn fake_higgs_with_fixture() -> (Higgs, tempfile::TempDir) {
    let dir = tempfile::TempDir::new().unwrap();
    crate::serve::test_support::write_gguf_fixture(dir.path(), "org/model");
    (fake_higgs(vec![dir.path().to_path_buf()]), dir)
}

/// A one-message user prompt as the verbatim OpenAI `messages` JSON string.
fn user_msg(text: &str) -> String {
    json!([{"role": "user", "content": text}]).to_string()
}

// ── fit_generation_budget (relocated from serve::v1's fit tests) ────────────

#[test]
fn fit_budget_rejects_prompt_that_alone_overflows() {
    // ~2000 estimated tokens (8000 bytes / 4) into a 128-token window: the prompt
    // ALONE overflows → no room to generate → genuine overflow.
    let err = super::fit_generation_budget(Some(16), Some(CtxLen::Fixed { n: 128 }), 8000)
        .expect_err("a prompt larger than the window overflows");
    assert!(matches!(err, HiggsError::ContextOverflow { .. }));
    assert!(err.to_string().starts_with("[HG005]"));
}

#[test]
fn fit_budget_honors_a_request_that_fits() {
    let budget = super::fit_generation_budget(Some(16), Some(CtxLen::Fixed { n: 4096 }), 5)
        .expect("a fitting request is honored");
    assert_eq!(budget, 16, "a request that fits is honored unchanged");
}

#[test]
fn fit_budget_clamps_oversized_max_tokens_instead_of_rejecting() {
    // Prompt fits, but max_tokens + prompt > n_ctx → CLAMP to what fits, not reject.
    let budget = super::fit_generation_budget(Some(16384), Some(CtxLen::Fixed { n: 8192 }), 2)
        .expect("an oversized max_tokens is clamped, not rejected");
    assert!(
        budget > 0 && budget <= 8192,
        "clamped to the space after the prompt (< the requested 16384): {budget}"
    );
}

#[test]
fn fit_budget_infers_full_window_when_max_tokens_omitted() {
    // No max_tokens (None) → infer the remaining window (n_ctx − prompt), NOT 1024.
    let budget = super::fit_generation_budget(None, Some(CtxLen::Fixed { n: 8192 }), 2)
        .expect("inferred budget");
    assert!(
        budget > 1024 && budget <= 8192,
        "inferred ~n_ctx, not the flat 1024 default: {budget}"
    );
}

#[test]
fn fit_budget_auto_window_honors_request_capped() {
    // An AUTO/unknown window can't be bounded here → honor the request (worker
    // [HG005] backstops), capped at MAX_OUTPUT_TOKENS.
    let budget = super::fit_generation_budget(Some(500), Some(CtxLen::Auto), 2).expect("auto");
    assert_eq!(budget, 500);
}

// ── estimate_prompt_bytes (matches serve::v1::messages_to_pairs) ────────────

#[test]
fn estimate_prompt_bytes_sums_string_content() {
    let json = json!([
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"}
    ])
    .to_string();
    assert_eq!(super::estimate_prompt_bytes(&json), 3 + 5);
}

#[test]
fn estimate_prompt_bytes_joins_text_parts_with_newline() {
    // Array text parts join with "\n" (the shimmy convention): "ab\ncd" = 5 bytes.
    let json = json!([
        {"role": "user", "content": [
            {"type": "text", "text": "ab"},
            {"type": "text", "text": "cd"}
        ]}
    ])
    .to_string();
    assert_eq!(super::estimate_prompt_bytes(&json), 5);
}

#[test]
fn estimate_prompt_bytes_non_text_part_is_zero() {
    // A non-text content part → the whole estimate is 0 (mirrors messages_to_pairs
    // `Err → 0`); such a request is rejected by the handler's text-only check anyway.
    let json = json!([
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}
    ])
    .to_string();
    assert_eq!(super::estimate_prompt_bytes(&json), 0);
}

#[test]
fn estimate_prompt_bytes_null_and_absent_content_are_zero() {
    let json = json!([
        {"role": "assistant"},
        {"role": "assistant", "content": null}
    ])
    .to_string();
    assert_eq!(super::estimate_prompt_bytes(&json), 0);
    // Malformed JSON degrades to 0, never a panic.
    assert_eq!(super::estimate_prompt_bytes("not json"), 0);
}

// ── prepare_chat: the shared /v1 chat gate ──────────────────────────────────

#[tokio::test]
async fn prepare_chat_already_loaded_returns_resolved_id() {
    let (higgs, _dir) = fake_higgs_with_fixture();
    higgs.load("org/model", None).await.expect("load");
    let prepared = higgs
        .prepare_chat("org/model", Some(16), &user_msg("hi"))
        .await
        .expect("a loaded model prepares cheaply");
    assert_eq!(prepared.resolved_model, "org/model");
    assert_eq!(
        prepared.max_gen, 16,
        "a fitting request is honored unchanged"
    );
}

#[tokio::test]
async fn prepare_chat_clamps_oversized_max_tokens() {
    // Loaded at the trained window (4096). A huge max_tokens is CLAMPED to what fits.
    let (higgs, _dir) = fake_higgs_with_fixture();
    higgs.load("org/model", None).await.expect("load");
    let prepared = higgs
        .prepare_chat("org/model", Some(1_000_000), &user_msg("hi"))
        .await
        .expect("prepare");
    assert!(
        (1..=4096).contains(&prepared.max_gen),
        "clamped to the loaded window (< the requested 1_000_000): {}",
        prepared.max_gen
    );
}

#[tokio::test]
async fn prepare_chat_jit_off_unloaded_refuses() {
    // With JIT off, an unloaded model is the explicit-load 404 [HG003].
    let (higgs, _dir) = fake_higgs_with_fixture();
    higgs.set_jit_enabled(false);
    let err = higgs
        .prepare_chat("org/model", Some(16), &user_msg("hi"))
        .await
        .expect_err("JIT off + unloaded refuses");
    assert!(
        matches!(err, HiggsError::ModelNotLoaded { .. }),
        "expected ModelNotLoaded, got {err}"
    );
}

#[tokio::test]
async fn prepare_chat_jit_on_unprepared_refuses() {
    // JIT on reaches the readiness gate: a scanned-but-un-Prepared model is refused
    // ([HG046]) rather than silently loaded with dumb defaults.
    let (higgs, _dir) = fake_higgs_with_fixture();
    assert!(higgs.jit_enabled(), "JIT is on by default");
    let err = higgs
        .prepare_chat("org/model", Some(16), &user_msg("hi"))
        .await
        .expect_err("JIT on + un-prepared refuses");
    assert!(
        matches!(err, HiggsError::NotPrepared { .. }),
        "expected NotPrepared, got {err}"
    );
}

#[tokio::test]
async fn prepare_chat_unknown_model_jit_on_not_found() {
    let (higgs, _dir) = fake_higgs_with_fixture();
    let err = higgs
        .prepare_chat("org/does-not-exist", Some(16), &user_msg("hi"))
        .await
        .expect_err("an unknown id is not found");
    assert!(matches!(err, HiggsError::ModelNotFound { .. }), "got {err}");
}

// ── model_entries / model_by_id ─────────────────────────────────────────────

#[tokio::test]
async fn model_entries_lists_scanned_models_with_load_state() {
    let (higgs, _dir) = fake_higgs_with_fixture();
    let entries = higgs.model_entries().await.expect("entries");
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].model.id, "org/model");
    assert_eq!(entries[0].state, "not-loaded");
    assert_eq!(entries[0].format, "gguf");

    higgs.load("org/model", None).await.expect("load");
    let entries = higgs.model_entries().await.expect("entries");
    assert_eq!(entries[0].state, "loaded", "a resident model reads loaded");
}

#[tokio::test]
async fn model_by_id_found_and_not_found() {
    let (higgs, _dir) = fake_higgs_with_fixture();
    let entry = higgs.model_by_id("org/model").await.expect("found");
    assert_eq!(entry.model.id, "org/model");
    let err = higgs
        .model_by_id("org/nope")
        .await
        .expect_err("absent id is 404");
    assert!(matches!(err, HiggsError::ModelNotFound { .. }), "got {err}");
}

// ── chat_model_ids: the /v1/models union ────────────────────────────────────

#[tokio::test]
async fn chat_model_ids_includes_local_served_even_with_jit_off() {
    // The A1.8 point: local-served ids are ALWAYS in the union, even JIT-off — a
    // bare `servable_model_ids()` would drop them and shrink the picker.
    let (higgs, _dir) = fake_higgs_with_fixture();
    higgs.load("org/model", None).await.expect("load");

    let with_jit = higgs.chat_model_ids().await;
    assert!(with_jit.contains(&"org/model".to_string()));

    higgs.set_jit_enabled(false);
    let jit_off = higgs.chat_model_ids().await;
    assert!(
        jit_off.contains(&"org/model".to_string()),
        "JIT off must still list a locally-served model"
    );
}

// ── load_flat / unload_spec ─────────────────────────────────────────────────

#[tokio::test]
async fn load_flat_default_and_pinned() {
    let (higgs, _dir) = fake_higgs_with_fixture();
    // No pinned field → a fully-default load.
    let req = serde_json::from_value(json!({"id": "org/model"})).unwrap();
    higgs.load_flat(&req).await.expect("default load");
    assert!(higgs.status().await.unwrap().loaded.is_some());

    higgs.unload_spec(None).await.expect("drain all");
    assert!(higgs.status().await.unwrap().loaded.is_none());

    // A pinned flat field (threads) takes the build-LoadParams branch.
    let req = serde_json::from_value(json!({"id": "org/model", "threads": 3})).unwrap();
    higgs.load_flat(&req).await.expect("pinned load");
    let li = higgs.status().await.unwrap().loaded.expect("loaded");
    assert_eq!(li.threads, Some(3), "the pinned threads value took effect");
}

#[tokio::test]
async fn unload_spec_one_vs_all() {
    let (higgs, _dir) = fake_higgs_with_fixture();
    higgs.load("org/model", None).await.expect("load");
    higgs
        .unload_spec(Some("org/model"))
        .await
        .expect("unload one");
    assert!(higgs.status().await.unwrap().loaded.is_none());
    // Draining an already-empty node is a no-op success.
    higgs.unload_spec(None).await.expect("drain all no-op");
}

// ── node ops / hub: the not-a-hub error for an embedder ─────────────────────

#[tokio::test]
async fn node_ops_without_a_hub_error() {
    let higgs = fake_higgs(vec![]);
    assert!(higgs.pair().await.is_err(), "no hub → pair errors");
    assert!(higgs.node_load("n", "m").await.is_err());
    assert!(higgs.node_unload("m").await.is_err());
    assert!(higgs.node_retire("n").await.is_err());
    assert!(higgs.node_scan("n").await.is_err());
    // hub_disable is a no-op returning a disabled status when no hub is installed.
    let status = higgs.hub_disable().await;
    assert!(!status.enabled);
}

// ── nodes(): the unified fleet view carries the local node's operator label ──

/// Parity with the deleted `control_nodes` handler (facade gap 2): `Higgs::nodes()`
/// lists the LOCAL machine first and labels it with this instance's `config.json`
/// name (read via `instance_name()`). Under `cfg(test)` the config path is a
/// hermetic per-instance temp file, so the rename round-trips. Fail-on-revert:
/// drop the `instance_name()` label wiring from `nodes()` and the local node falls
/// back to `"this machine"`, so the renamed label no longer appears.
#[tokio::test]
async fn nodes_view_labels_the_local_node_from_instance_name() {
    let higgs = fake_higgs(vec![]);

    // Default (no config name set) → the local node uses the "this machine" fallback.
    let default_view = higgs.nodes().await;
    assert_eq!(default_view.len(), 1, "no fleet → just the local node");
    assert!(
        default_view[0].is_local,
        "the sole node is the local machine"
    );
    assert_eq!(
        default_view[0].label, "this machine",
        "an unnamed instance falls back to 'this machine'"
    );

    // Rename this instance (writes the hermetic per-instance config.json), then the
    // nodes view must carry that label on the local node.
    higgs
        .node_label("local", "my-workstation")
        .await
        .expect("local rename persists");
    let view = higgs.nodes().await;
    assert_eq!(
        view[0].label, "my-workstation",
        "the local node's label comes from instance_name(): {view:?}"
    );
}

// ── logs settings round-trip / worker_stop ──────────────────────────────────

#[tokio::test]
async fn logs_settings_round_trips() {
    let higgs = fake_higgs(vec![]);
    let before = higgs.logs_settings();
    assert!(!before.verbose);
    higgs.set_logs_settings(&crate::serve::wire::LogSettings {
        verbose: true,
        log_incoming_tokens: true,
        show_log_fields: true,
    });
    let after = higgs.logs_settings();
    assert!(after.verbose && after.log_incoming_tokens && after.show_log_fields);
}

#[tokio::test]
async fn worker_stop_drains_workers() {
    let (higgs, _dir) = fake_higgs_with_fixture();
    higgs.load("org/model", None).await.expect("load");
    higgs.worker_stop().await.expect("stop unloads");
    assert!(higgs.status().await.unwrap().loaded.is_none());
}

// ── mint_key / revoke_key: trusted skips the bearer, keeps the invariants ───

/// The whole key test runs under `TEST_ENV_LOCK` with an isolated `HIGGS_HOME`
/// so the trusted mint/revoke persistence never touches the real keystore.
#[tokio::test]
#[allow(clippy::await_holding_lock)] // TEST_ENV_LOCK serializes HIGGS_HOME for the test
async fn mint_and_revoke_key_trusted_keep_invariants() {
    let _env = crate::TEST_ENV_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let home = tempfile::tempdir().unwrap();
    let prev = std::env::var_os("HIGGS_HOME");
    // SAFETY: serialized by TEST_ENV_LOCK; restored below.
    unsafe { std::env::set_var("HIGGS_HOME", home.path()) };

    let higgs = fake_higgs(vec![]);

    // Bootstrap: the first (empty-store) key must include admin.
    assert!(
        higgs.mint_key("laptop", Some(vec![Scope::Chat])).is_err(),
        "a non-admin bootstrap key is refused (BootstrapNeedsAdmin)"
    );
    let admin = higgs
        .mint_key("admin", Some(vec![Scope::Admin]))
        .expect("bootstrap admin mint");
    assert_eq!(admin.scopes, vec![Scope::Admin]);
    assert!(!admin.token.is_empty());

    // Trusted mint on a NON-empty store bypasses the bearer Unauthorized branch.
    let laptop = higgs
        .mint_key("laptop", Some(vec![Scope::Chat]))
        .expect("trusted mint bypasses the bearer requirement");
    assert_eq!(laptop.scopes, vec![Scope::Chat]);

    // Duplicate label is still rejected.
    assert!(
        higgs
            .mint_key("laptop", Some(vec![Scope::Chat]))
            .is_err_and(|e| e.to_string().contains("already exists")),
        "a duplicate label is still Duplicate-rejected"
    );
    // Invalid label is still rejected.
    assert!(higgs.mint_key("a/b", Some(vec![Scope::Chat])).is_err());
    // An explicit empty scope list is still rejected.
    assert!(higgs.mint_key("x", Some(vec![])).is_err());

    // Revoking the LAST admin while a non-admin key remains is still HG066.
    assert!(
        matches!(
            higgs.revoke_key("admin"),
            Err(HiggsError::LastAdminKey { .. })
        ),
        "trusted revoke still refuses the last-admin lockout"
    );
    // Revoking the non-admin key is fine.
    let removed = higgs.revoke_key("laptop").expect("revoke non-admin");
    assert_eq!(removed.removed, 1);

    // Revoking the last key while LAN-exposed is still HG059.
    higgs.set_lan_exposed(true);
    assert!(
        matches!(
            higgs.revoke_key("admin"),
            Err(HiggsError::LastKeyOnLan { .. })
        ),
        "trusted revoke still refuses last-key-on-LAN"
    );

    // SAFETY: serialized by TEST_ENV_LOCK.
    unsafe {
        match prev {
            Some(v) => std::env::set_var("HIGGS_HOME", v),
            None => std::env::remove_var("HIGGS_HOME"),
        }
    }
}
