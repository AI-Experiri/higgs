//! In-process integration tests for the OpenAI `/v1` ERROR + edge paths.
//!
//! higgs is a library now: control (load / tune) runs through the in-process
//! `Higgs` facade and the REAL `/v1` HTTP surface is served by `serve_v1_local`
//! on an ephemeral loopback port. These tests drive the request-validation and
//! error branches in `src/serve/v1.rs` and the SSE assembly in `src/serve/stream.rs`:
//!   * out-of-range sampling params  → 400 `[HG013]` invalid_request_error
//!   * unknown model (JIT on)        → 404 `model_not_found` `[HG002]`
//!   * malformed / empty JSON body   → 4xx (no 5xx, no panic)
//!   * missing required `model`      → 4xx
//!   * `stream: true` happy path     → SSE deltas drained to `[DONE]`
//!   * `stream: false` happy path    → full completion envelope
//!   * `GET /v1/models`              → OpenAI shape, lists the loaded model
//!
//! A few readiness/persistence branches (stale profile, removed model, a Prepare
//! that can't persist) are pure control-plane errors with no `/v1` surface, so
//! they assert on the typed [`HiggsError`] the facade returns directly.
//!
//! The tiny model is `ggml-org`'s ~1MB `stories260K.gguf` (see `common`). A REAL
//! local llama.cpp worker runs via the `worker_exe` DI seam. SSE streams are read
//! to completion then dropped — never left open (an open stream blocks graceful
//! shutdown).

mod common;

use common::{higgs_local, serve_v1_local, TINY_MODEL_ID};
use higgs::worker::engine::llamacpp::params::LlamaCppParams;
use higgs::worker::engine::CtxLen;
use higgs::{HiggsError, LoadParams, TuneRequest};
use serde_json::{json, Value};

/// Explicitly load the tiny model via the control facade so the subsequent chat
/// request's failing gate is unambiguously the one under test (not a JIT load).
/// An explicit load bypasses the readiness gate. Idempotent when already resident.
async fn load_tiny(h: &higgs::Higgs) {
    h.load(TINY_MODEL_ID, None)
        .await
        .expect("tiny model load succeeded");
}

/// Load the tiny model with an EXPLICIT small context window so the overflow /
/// clamp behavior is deterministic regardless of the model's trained window.
async fn load_tiny_with_ctx(h: &higgs::Higgs, n: u32) {
    h.load(
        TINY_MODEL_ID,
        Some(LoadParams::llamacpp(LlamaCppParams {
            ctx_len: CtxLen::fixed(n),
            ..Default::default()
        })),
    )
    .await
    .unwrap_or_else(|e| panic!("load with ctx {n} succeeded, got {e}"));
}

/// Prepare (autotune) the tiny model so a JIT chat / explicit reload is allowed
/// by the readiness gate and a saved profile with file/hardware anchors exists.
async fn prepare_tiny(h: &higgs::Higgs) {
    h.tune(TuneRequest {
        id: TINY_MODEL_ID.to_owned(),
        mode: None,
        budget: None,
        pins: None,
    })
    .await
    .expect("Prepare (tune) tiny model succeeded");
}

// ── Un-prepared model chat (JIT) → 400 model_not_prepared [HG046] ────────────
//
// A scanned-but-never-Prepared model must NOT silently JIT-load with default
// params: the readiness gate refuses it with a coded error. This is the fix for
// the "loads with the wrong context" class of bug. Fails-on-revert: drop the
// gate in `resolve_loaded` and the chat 200s (the model loads) instead.
#[tokio::test]
async fn unprepared_model_chat_is_model_not_prepared() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP unprepared_model_chat_is_model_not_prepared: tiny gguf not found");
        return;
    };
    let (base, guard) = serve_v1_local(higgs.handle()).await;
    let c = reqwest::Client::new();
    // Deliberately NO load / tune — the model is scanned but un-Prepared, so the
    // chat hits the JIT path and must be refused.
    let resp = c
        .post(format!("{base}/v1/chat/completions"))
        .json(&json!({
            "model": TINY_MODEL_ID, "stream": false,
            "messages": [{ "role": "user", "content": "hi" }]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        400,
        "un-prepared JIT chat is a 400, got {}",
        resp.status()
    );
    let env: Value = resp.json().await.unwrap();
    assert_eq!(
        env["error"]["code"], "model_not_prepared",
        "coded model_not_prepared: {env:?}"
    );
    // And it never became resident — no silent default load. With JIT on, an
    // un-prepared model is not "servable", so `/v1/models` is empty.
    let models: Value = c
        .get(format!("{base}/v1/models"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let resident = models["data"].as_array().map(Vec::len).unwrap_or(0);
    assert_eq!(
        resident, 0,
        "no model resident after a refused JIT: {models:?}"
    );

    guard.shutdown().await;
    higgs.shutdown().await;
}

// ── Explicit load of a STALE profile is refused [HG047] ──────────────────────
//
// The JIT gate hard-blocks a stale profile; an explicit `load(id, None)` (which
// reuses the saved profile) must too. Prepare records the file/hardware anchors;
// mutating the GGUF changes its file signature so the profile reads back stale
// and the reuse is refused. Fails-on-revert: drop the stale guard in
// `Higgs::load_inner_impl` and the explicit load no longer returns `ProfileStale`.
#[tokio::test]
async fn explicit_load_of_stale_profile_is_refused() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP explicit_load_of_stale_profile_is_refused: tiny gguf not found");
        return;
    };
    // Prepare → a saved profile with anchors for the current hardware + file.
    prepare_tiny(&higgs).await;
    // Make the on-disk model differ from the profile's recorded file signature by
    // growing it one byte (the GGUF header still scans fine).
    {
        use std::io::Write as _;
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(higgs.staged_gguf(TINY_MODEL_ID))
            .expect("open staged gguf");
        f.write_all(b"\0").expect("append to staged gguf");
    }
    // Explicit load reusing the now-stale saved profile → ProfileStale [HG047].
    let err = higgs
        .load(TINY_MODEL_ID, None)
        .await
        .expect_err("stale profile reuse is refused");
    assert!(
        matches!(err, HiggsError::ProfileStale { .. }),
        "carries HG047 stale: {err}"
    );

    higgs.shutdown().await;
}

// ── Removed model is NOT reported stale (preserve not-found) ─────────────────
//
// A previously-Prepared model whose GGUF is then removed must surface the real
// not-found/load failure, NOT a misleading [HG047] "Re-tune" — re-tuning can't
// fix a missing model. Fails-on-revert: restore the removed-model stale short
// circuit and the load reports `ProfileStale` instead.
#[tokio::test]
async fn explicit_load_of_removed_model_is_not_reported_stale() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP explicit_load_of_removed_model_is_not_reported_stale: tiny gguf not found");
        return;
    };
    // Prepare → a saved profile with anchors, then delete the model file.
    prepare_tiny(&higgs).await;
    std::fs::remove_file(higgs.staged_gguf(TINY_MODEL_ID)).expect("remove staged gguf");
    // It must fail (the model is gone) but NOT as a stale/Re-tune.
    let err = higgs
        .load(TINY_MODEL_ID, None)
        .await
        .expect_err("load of a removed model fails");
    assert!(
        !matches!(err, HiggsError::ProfileStale { .. }),
        "a removed model is not reported stale: {err}"
    );

    higgs.shutdown().await;
}

// ── Prepare that can't persist FAILS (not a silent success) ──────────────────
//
// The readiness gate makes a PERSISTED profile a serving precondition, so a tune
// that can't write `models.json` must surface the error — otherwise a client sees
// a "successful" Prepare followed by a refused `model_not_prepared` chat. We force
// the persistence failure by occupying `models.json` with a directory so the
// store's temp→rename flush can't create the file. Fails-on-revert: make `tune`
// log-and-`Ok` again and the request succeeds despite saving nothing.
#[tokio::test]
async fn prepare_that_cannot_persist_fails_loudly() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP prepare_that_cannot_persist_fails_loudly: tiny gguf not found");
        return;
    };
    // Occupy models.json with a NON-EMPTY directory → the flush rename can't write.
    let mj = higgs.home().join("models.json");
    let _ = std::fs::remove_file(&mj);
    std::fs::create_dir_all(mj.join("blocker")).expect("occupy models.json with a dir");
    let res = higgs
        .tune(TuneRequest {
            id: TINY_MODEL_ID.to_owned(),
            mode: None,
            budget: None,
            pins: None,
        })
        .await;
    assert!(
        res.is_err(),
        "Prepare that can't persist must fail, got {res:?}"
    );

    higgs.shutdown().await;
}

// ── Out-of-range sampling param → 400 [HG013] invalid_request_error ──────────
//
// `top_p` must be in (0, 1]; 2.0 is out of range. The model is loaded first so
// the sampling-validation gate (which runs AFTER the loaded-model gate in
// `gate_and_validate`) is the unambiguous point of failure.
#[tokio::test]
async fn invalid_sampling_param_top_p_400_hg013() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP invalid_sampling_param_top_p_400_hg013: tiny gguf not found");
        return;
    };
    load_tiny(&higgs).await;
    let (base, guard) = serve_v1_local(higgs.handle()).await;
    let c = reqwest::Client::new();

    let resp = c
        .post(format!("{base}/v1/chat/completions"))
        .json(&json!({
            "model": TINY_MODEL_ID, "stream": false,
            "top_p": 2.0,
            "messages": [{ "role": "user", "content": "hi" }]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        400,
        "top_p out of (0,1] is a 400 invalid_request_error"
    );
    let env: Value = resp.json().await.unwrap();
    assert_eq!(
        env["error"]["type"], "invalid_request_error",
        "400 envelope type is invalid_request_error: {env:?}"
    );
    let msg = env["error"]["message"].as_str().unwrap_or_default();
    assert!(msg.contains("[HG013]"), "message carries HG013: {env:?}");
    assert!(
        msg.contains("top_p"),
        "message names the offending param: {env:?}"
    );
    // On a 4xx the OpenAI envelope `code` is reserved for model_not_found (404).
    assert!(
        env["error"]["code"].is_null(),
        "HG013 4xx carries no `code`: {env:?}"
    );

    guard.shutdown().await;
    higgs.shutdown().await;
}

// ── Unreadable store on a Prepared model → HG040, not model_not_prepared ──────
//
// A store-READ failure (models.json exists but can't be opened) must NOT be
// masked as an absent profile: that would tell a user who DID Prepare to Prepare
// again (which also can't persist), hiding the real persistence fault. Prepare
// writes the profile, then we make models.json unreadable; the JIT readiness gate
// must surface [HG040], not `model_not_prepared`. Fails-on-revert: restore the
// `.ok()` swallow in `profile_state` and the chat returns `model_not_prepared`.
#[tokio::test]
async fn jit_unreadable_store_surfaces_persistence_error() {
    use std::os::unix::fs::PermissionsExt;
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP jit_unreadable_store_surfaces_persistence_error: tiny gguf not found");
        return;
    };
    prepare_tiny(&higgs).await; // writes models.json with the profile
                                // Make models.json exist-but-unreadable (owner read denied) so the store open
                                // fails when the JIT gate reads the profile.
    let mj = higgs.home().join("models.json");
    std::fs::set_permissions(&mj, std::fs::Permissions::from_mode(0o000))
        .expect("chmod models.json unreadable");

    let (base, guard) = serve_v1_local(higgs.handle()).await;
    let c = reqwest::Client::new();
    let resp = c
        .post(format!("{base}/v1/chat/completions"))
        .json(&json!({
            "model": TINY_MODEL_ID, "stream": false,
            "messages": [{ "role": "user", "content": "hi" }]
        }))
        .send()
        .await
        .unwrap();
    let body = resp.text().await.unwrap();
    assert!(
        body.contains("[HG040]"),
        "unreadable store surfaces the persistence error: {body}"
    );
    assert!(
        !body.contains("model_not_prepared"),
        "a read failure is NOT masked as unprepared: {body}"
    );

    // Restore perms so the temp HIGGS_HOME cleans up predictably.
    let _ = std::fs::set_permissions(&mj, std::fs::Permissions::from_mode(0o600));
    guard.shutdown().await;
    higgs.shutdown().await;
}

// ── Malformed request body (text-only rejects image content) → 400 [HG049] ───
//
// `/v1` is text-only, so an `image_url` content part is rejected by the request
// validator (`messages_to_pairs`), which returns the otherwise-bare 400 through
// the coded `HG049` path. Proves every non-success reply — including a malformed
// body — carries a diagnostic code. Fails-on-revert: route `v1_bad_request` back
// to a plain message and the body no longer mentions HG049.
#[tokio::test]
async fn malformed_image_content_400_hg049() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP malformed_image_content_400_hg049: tiny gguf not found");
        return;
    };
    load_tiny(&higgs).await;
    let (base, guard) = serve_v1_local(higgs.handle()).await;
    let c = reqwest::Client::new();

    let resp = c
        .post(format!("{base}/v1/chat/completions"))
        .json(&json!({
            "model": TINY_MODEL_ID, "stream": false,
            "messages": [{
                "role": "user",
                "content": [{ "type": "image_url", "image_url": { "url": "http://example/x.png" } }]
            }]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        400,
        "image content in text-only /v1 is a 400"
    );
    let env: Value = resp.json().await.unwrap();
    let msg = env["error"]["message"].as_str().unwrap_or_default();
    assert!(
        msg.contains("[HG049]"),
        "malformed-body 400 carries the HG049 code: {env:?}"
    );

    guard.shutdown().await;
    higgs.shutdown().await;
}

// ── A second out-of-range param: presence_penalty must be in [-2, 2] ─────────
//
// Distinct sampling-validation branch from top_p, asserting the same 400 [HG013]
// envelope shape so the param-range table in `validate_sampling` is exercised.
#[tokio::test]
async fn invalid_sampling_param_presence_penalty_400_hg013() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP invalid_sampling_param_presence_penalty_400_hg013: tiny gguf not found");
        return;
    };
    load_tiny(&higgs).await;
    let (base, guard) = serve_v1_local(higgs.handle()).await;
    let c = reqwest::Client::new();

    let resp = c
        .post(format!("{base}/v1/chat/completions"))
        .json(&json!({
            "model": TINY_MODEL_ID, "stream": false,
            "presence_penalty": 5.0,
            "messages": [{ "role": "user", "content": "hi" }]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        400,
        "presence_penalty out of [-2,2] is a 400"
    );
    let env: Value = resp.json().await.unwrap();
    assert_eq!(env["error"]["type"], "invalid_request_error");
    let msg = env["error"]["message"].as_str().unwrap_or_default();
    assert!(msg.contains("[HG013]"), "message carries HG013: {env:?}");
    assert!(
        msg.contains("presence_penalty"),
        "message names the offending param: {env:?}"
    );

    guard.shutdown().await;
    higgs.shutdown().await;
}

// ── n != 1 is rejected (higgs serves a single choice) → 400 [HG013] ──────────
#[tokio::test]
async fn invalid_sampling_param_n_not_one_400_hg013() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP invalid_sampling_param_n_not_one_400_hg013: tiny gguf not found");
        return;
    };
    load_tiny(&higgs).await;
    let (base, guard) = serve_v1_local(higgs.handle()).await;
    let c = reqwest::Client::new();

    let resp = c
        .post(format!("{base}/v1/chat/completions"))
        .json(&json!({
            "model": TINY_MODEL_ID, "stream": false,
            "n": 2,
            "messages": [{ "role": "user", "content": "hi" }]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400, "n>1 is unsupported → 400");
    let env: Value = resp.json().await.unwrap();
    assert_eq!(env["error"]["type"], "invalid_request_error");
    let msg = env["error"]["message"].as_str().unwrap_or_default();
    assert!(msg.contains("[HG013]"), "message carries HG013: {env:?}");
    assert!(msg.contains('n'), "message references n: {env:?}");

    guard.shutdown().await;
    higgs.shutdown().await;
}

// ── max_tokens above the cap → 400 [HG013] ───────────────────────────────────
//
// `max_tokens` must be <= MAX_OUTPUT_TOKENS (32_768); above it is a 400. This
// validation runs only AFTER the loaded gate, so the model is loaded first.
#[tokio::test]
async fn max_tokens_over_cap_400_hg013() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP max_tokens_over_cap_400_hg013: tiny gguf not found");
        return;
    };
    load_tiny(&higgs).await;
    let (base, guard) = serve_v1_local(higgs.handle()).await;
    let c = reqwest::Client::new();

    let resp = c
        .post(format!("{base}/v1/chat/completions"))
        .json(&json!({
            "model": TINY_MODEL_ID, "stream": false,
            "max_tokens": 1_000_000,
            "messages": [{ "role": "user", "content": "hi" }]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400, "max_tokens over the cap is a 400");
    let env: Value = resp.json().await.unwrap();
    assert_eq!(env["error"]["type"], "invalid_request_error");
    assert!(
        env["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("[HG013]"),
        "message carries HG013: {env:?}"
    );

    guard.shutdown().await;
    higgs.shutdown().await;
}

// ── Unknown model (JIT on default) → 404 model_not_found [HG002] ─────────────
//
// JIT auto-load is on by default, but JIT only loads a SCANNED id; an id absent
// from the scan is a `model_not_found` 404, NEVER a load attempt. The envelope
// codes `model_not_found` (the only coded case on this surface) and the message
// carries [HG002] (distinct from the explicit-load [HG003]).
#[tokio::test]
async fn unknown_model_404_model_not_found() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP unknown_model_404_model_not_found: tiny gguf not found");
        return;
    };
    let (base, guard) = serve_v1_local(higgs.handle()).await;
    let c = reqwest::Client::new();

    let resp = c
        .post(format!("{base}/v1/chat/completions"))
        .json(&json!({
            "model": "no-such-org/no-such-model", "stream": false,
            "messages": [{ "role": "user", "content": "hi" }]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404, "unknown model is a 404");
    let env: Value = resp.json().await.unwrap();
    assert_eq!(
        env["error"]["code"], "model_not_found",
        "404 envelope codes model_not_found: {env:?}"
    );
    assert_eq!(
        env["error"]["type"], "invalid_request_error",
        "404 is an invalid_request_error: {env:?}"
    );
    assert!(
        env["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("[HG002]"),
        "JIT unknown-id message carries HG002 (not HG003): {env:?}"
    );

    guard.shutdown().await;
    higgs.shutdown().await;
}

// ── Malformed body branches: empty body, invalid JSON, missing `model` ───────
//
// All three must be a 4xx caught at the boundary (axum JSON extractor or the
// handler's own validation) — never a 5xx or a panic. Exercises the
// request-rejection edges around `v1_chat_completions`.
#[tokio::test]
async fn malformed_and_missing_field_bodies_are_4xx() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP malformed_and_missing_field_bodies_are_4xx: tiny gguf not found");
        return;
    };
    let (base, guard) = serve_v1_local(higgs.handle()).await;
    let c = reqwest::Client::new();
    let url = format!("{base}/v1/chat/completions");

    // Empty body with a JSON content-type → the extractor rejects it 4xx.
    let empty = c
        .post(&url)
        .header("content-type", "application/json")
        .body("")
        .send()
        .await
        .unwrap();
    assert!(
        empty.status().is_client_error(),
        "empty body is a 4xx, got {}",
        empty.status()
    );

    // Syntactically invalid JSON → 4xx (no 5xx, no panic).
    let invalid = c
        .post(&url)
        .header("content-type", "application/json")
        .body("{not valid json")
        .send()
        .await
        .unwrap();
    assert!(
        invalid.status().is_client_error(),
        "invalid JSON is a 4xx, got {}",
        invalid.status()
    );

    // Well-formed JSON but missing the required `model` field → 4xx.
    let missing_model = c
        .post(&url)
        .json(&json!({ "messages": [{ "role": "user", "content": "hi" }] }))
        .send()
        .await
        .unwrap();
    assert!(
        missing_model.status().is_client_error(),
        "missing `model` is a 4xx, got {}",
        missing_model.status()
    );

    // Well-formed JSON but missing the required `messages` field → 4xx.
    let missing_messages = c
        .post(&url)
        .json(&json!({ "model": TINY_MODEL_ID }))
        .send()
        .await
        .unwrap();
    assert!(
        missing_messages.status().is_client_error(),
        "missing `messages` is a 4xx, got {}",
        missing_messages.status()
    );

    guard.shutdown().await;
    higgs.shutdown().await;
}

// ── stream: true happy path — drain the SSE to [DONE], then drop ─────────────
//
// Reads the FULL SSE body to a String (which closes the stream) so it's never
// left open, then asserts the assistant-role chunk, content deltas, finish, and
// the trailing [DONE] — the framing produced by `stream::chat_sse` / `assemble`.
#[tokio::test]
async fn stream_true_happy_path_drains_to_done() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP stream_true_happy_path_drains_to_done: tiny gguf not found");
        return;
    };
    load_tiny(&higgs).await;
    let (base, guard) = serve_v1_local(higgs.handle()).await;
    let c = reqwest::Client::new();

    // `.text()` reads the body to completion and drops the response — the stream
    // is never left open (which would block graceful shutdown).
    let body = c
        .post(format!("{base}/v1/chat/completions"))
        .json(&json!({
            "model": TINY_MODEL_ID, "stream": true, "max_tokens": 8,
            "messages": [{ "role": "user", "content": "Count to three." }]
        }))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();

    assert!(body.contains("data:"), "stream has data lines: {body}");
    assert!(body.contains("[DONE]"), "stream terminates with [DONE]");
    assert!(
        body.contains("chat.completion.chunk"),
        "chunks carry the chunk object type"
    );

    // The first data chunk announces the assistant role; at least one carries a
    // content delta. Parse the framed `data:` payloads (skipping [DONE]).
    let chunks: Vec<Value> = body
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter(|p| *p != "[DONE]")
        .filter_map(|p| serde_json::from_str::<Value>(p).ok())
        .collect();
    assert!(!chunks.is_empty(), "at least one parseable chunk: {body}");
    assert_eq!(
        chunks[0]["choices"][0]["delta"]["role"], "assistant",
        "first chunk announces the assistant role: {body}"
    );
    assert!(
        chunks
            .iter()
            .any(|ch| ch["choices"][0]["delta"]["content"].is_string()),
        "at least one content delta: {body}"
    );
    assert!(
        chunks.iter().any(|ch| matches!(
            ch["choices"][0]["finish_reason"].as_str(),
            Some("stop") | Some("length")
        )),
        "a finish chunk carries a known reason: {body}"
    );

    guard.shutdown().await;
    higgs.shutdown().await;
}

// ── stream: true streaming AND stream: false non-streaming, same server ──────
//
// Drives both the SSE path and the buffered non-streaming path against one
// loaded model so the two `req.stream` branches of `v1_chat_completions` are
// exercised side by side. The stream is read to completion (closed) before the
// non-streaming request.
#[tokio::test]
async fn stream_and_nonstream_both_succeed() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP stream_and_nonstream_both_succeed: tiny gguf not found");
        return;
    };
    load_tiny(&higgs).await;
    let (base, guard) = serve_v1_local(higgs.handle()).await;
    let c = reqwest::Client::new();

    // Streaming branch — drained to [DONE], stream closed by `.text()`.
    let stream_body = c
        .post(format!("{base}/v1/chat/completions"))
        .json(&json!({
            "model": TINY_MODEL_ID, "stream": true, "max_tokens": 8,
            "messages": [{ "role": "user", "content": "Say hi." }]
        }))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(
        stream_body.contains("[DONE]"),
        "streaming branch terminates with [DONE]: {stream_body}"
    );

    // Non-streaming branch — a full chat.completion envelope.
    let resp: Value = c
        .post(format!("{base}/v1/chat/completions"))
        .json(&json!({
            "model": TINY_MODEL_ID, "stream": false, "max_tokens": 8,
            "messages": [{ "role": "user", "content": "Say hi." }]
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        resp["object"], "chat.completion",
        "non-streaming envelope object type: {resp:?}"
    );
    assert!(
        resp["id"]
            .as_str()
            .is_some_and(|s| s.starts_with("chatcmpl-")),
        "non-streaming id is a chatcmpl id: {resp:?}"
    );
    assert!(
        resp["choices"][0]["message"]["content"].is_string(),
        "non-streaming returns content: {resp:?}"
    );
    assert!(
        matches!(
            resp["choices"][0]["finish_reason"].as_str(),
            Some("stop") | Some("length")
        ),
        "non-streaming finish_reason is stop|length: {resp:?}"
    );
    assert!(
        resp["usage"]["total_tokens"].as_u64().unwrap_or(0) > 0,
        "non-streaming reports real token usage: {resp:?}"
    );

    guard.shutdown().await;
    higgs.shutdown().await;
}

// ── GET /v1/models returns the loaded model in OpenAI shape ──────────────────
//
// Empty before any load (the correct OpenAI answer for "nothing can serve chat
// now" — an un-prepared model is not servable), then after an explicit load the
// model appears with the OpenAI `Model` shape: id, object:"model",
// owned_by:"higgs", numeric created.
#[tokio::test]
async fn v1_models_openai_shape_after_load() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP v1_models_openai_shape_after_load: tiny gguf not found");
        return;
    };
    let (base, guard) = serve_v1_local(higgs.handle()).await;
    let c = reqwest::Client::new();

    // Empty list before any load (the un-prepared tiny model is not servable).
    let empty: Value = c
        .get(format!("{base}/v1/models"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(empty["object"], "list", "/v1/models is a list");
    assert!(
        empty["data"].as_array().unwrap().is_empty(),
        "/v1/models is empty with nothing loaded: {empty:?}"
    );

    load_tiny(&higgs).await;

    let listed: Value = c
        .get(format!("{base}/v1/models"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(listed["object"], "list");
    let data = listed["data"].as_array().unwrap();
    let m = data
        .iter()
        .find(|m| m["id"] == json!(TINY_MODEL_ID))
        .unwrap_or_else(|| panic!("/v1/models lists the loaded model: {listed:?}"));
    assert_eq!(m["object"], "model", "OpenAI model object type: {m:?}");
    assert_eq!(m["owned_by"], "higgs", "owned_by is higgs: {m:?}");
    assert!(
        m["created"].as_u64().is_some(),
        "created is a numeric unix timestamp: {m:?}"
    );

    guard.shutdown().await;
    higgs.shutdown().await;
}

// ── Step 5: prompt that ALONE overflows → 400 `context_length_exceeded` ───────────
//
// A prompt larger than the window can't generate anything → the genuine overflow,
// surfaced with the OpenAI-standard `context_length_exceeded` code (was `null`).
#[tokio::test]
async fn prompt_overflow_400_context_length_exceeded() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP prompt_overflow: tiny gguf not found");
        return;
    };
    load_tiny_with_ctx(&higgs, 256).await;
    let (base, guard) = serve_v1_local(higgs.handle()).await;
    let c = reqwest::Client::new();

    // ~5000 estimated tokens (20000 chars / 4 bytes) ≫ the 256-token window.
    let resp = c
        .post(format!("{base}/v1/chat/completions"))
        .json(&json!({
            "model": TINY_MODEL_ID, "stream": false,
            "messages": [{ "role": "user", "content": "x".repeat(20000) }]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400, "an over-window prompt is a 400");
    let env: Value = resp.json().await.unwrap();
    assert_eq!(
        env["error"]["code"], "context_length_exceeded",
        "the standard OpenAI code, not null: {env:?}"
    );

    guard.shutdown().await;
    higgs.shutdown().await;
}

// ── Step 5: oversized max_tokens is CLAMPED, not rejected ─────────────────────────
//
// A small prompt with a `max_tokens` far larger than the window used to 400 [HG005].
// Now the budget is clamped (serve layer AND the worker's tokenizer-exact backstop)
// so the request succeeds and generates, truncating at the window.
#[tokio::test]
async fn oversized_max_tokens_is_clamped_not_rejected() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP clamp: tiny gguf not found");
        return;
    };
    load_tiny_with_ctx(&higgs, 256).await;
    let (base, guard) = serve_v1_local(higgs.handle()).await;
    let c = reqwest::Client::new();

    let resp = c
        .post(format!("{base}/v1/chat/completions"))
        .json(&json!({
            "model": TINY_MODEL_ID, "stream": false,
            "max_tokens": 16384, // ≫ the 256 window: OLD → 400; NEW → clamp → 200
            "messages": [{ "role": "user", "content": "hi" }]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        200,
        "an oversized max_tokens is clamped to fit, not rejected"
    );
    let body: Value = resp.json().await.unwrap();
    assert!(
        body["choices"][0]["message"].is_object(),
        "a completion was generated: {body:?}"
    );
    // It truncated at the window (didn't run the full 16384) → finish_reason length.
    assert_eq!(
        body["choices"][0]["finish_reason"], "length",
        "generation truncated at the window: {body:?}"
    );

    guard.shutdown().await;
    higgs.shutdown().await;
}
