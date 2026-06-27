//! Autotune end-to-end: `POST /api/higgs/models/tune` (Suggest) → load with the
//! suggestion → serve → a plain reload reuses the saved profile.
//!
//! Spawns a real `higgs` against the tiny `stories260K` GGUF (skips when absent —
//! see `tiny_gguf_path`). SIGTERM teardown via `ServerGuard::drop`; the only chat
//! is non-streaming (no SSE stream left open, per CLAUDE.md).

mod common;

use common::{spawn_with_models, spawn_with_tiny_model, tiny_gguf_path, TINY_MODEL_ID};

/// `POST …/tune` (Suggest) returns suggested params honoring the CPU-thread cap,
/// a fit verdict for VRAM + RAM, and a non-empty rationale.
#[tokio::test]
async fn tune_suggest_returns_params_fit_and_rationale() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("SKIP tune_suggest: no tiny GGUF (set HIGGS_TEST_GGUF)");
        return;
    };
    let srv = spawn_with_tiny_model(11540, &gguf).await;
    let c = reqwest::Client::new();

    let resp = c
        .post(format!("{}/api/higgs/models/tune", srv.base))
        .json(&serde_json::json!({
            "id": TINY_MODEL_ID,
            "mode": "suggest",
            "budget": { "max_cpu_threads": 2 }
        }))
        .send()
        .await
        .expect("tune request");
    assert_eq!(
        resp.status(),
        200,
        "the tune route exists and the body-id is honored"
    );
    let s: serde_json::Value = resp.json().await.unwrap();

    // The umbrella is engine-tagged + flattened.
    assert_eq!(
        s["load"]["engine"], "LlamaCpp",
        "load is the LlamaCpp variant: {s}"
    );
    // The CPU-thread cap is honored (threads = min(floor(cores/2), 2) ≤ 2).
    assert!(
        s["load"]["threads"].as_u64().unwrap() <= 2,
        "cpu thread cap honored: {}",
        s["load"]
    );
    assert!(
        s["load"]["ctx_len"]["n"].as_u64().unwrap() > 0,
        "a concrete ctx_len"
    );
    // Derived defaults: flash attention on.
    assert_eq!(s["load"]["flash_attn"], "on");
    // Both fit verdicts present and a rationale.
    assert!(s["vram_fit"]["verdict"].is_string(), "vram verdict: {s}");
    assert!(s["ram_fit"]["verdict"].is_string(), "ram verdict: {s}");
    assert!(
        !s["rationale"].as_array().unwrap().is_empty(),
        "rationale explains the choices: {s}"
    );
}

/// Tune → load WITH the suggestion → non-streaming chat serves; then a PLAIN
/// reload (no params) reuses the saved profile (the persisted `last_load` carries
/// the tuned optionals, distinguishing it from the bare `default_load`).
#[tokio::test]
async fn tune_then_load_and_plain_load_reuses_profile() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("SKIP tune_then_load: no tiny GGUF (set HIGGS_TEST_GGUF)");
        return;
    };
    let srv = spawn_with_tiny_model(11541, &gguf).await;
    let c = reqwest::Client::new();

    // 1. Tune (Suggest) — persists the profile to the node's models.json.
    let s: serde_json::Value = c
        .post(format!("{}/api/higgs/models/tune", srv.base))
        .json(&serde_json::json!({ "id": TINY_MODEL_ID }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let suggested_load = s["load"].clone();
    let tuned_ctx = suggested_load["ctx_len"]["n"].as_u64().unwrap();

    // 2. Load WITH the accepted suggestion (the `engine` tag is ignored by the
    //    flat LlamaCppParams deserializer).
    let r = c
        .post(format!("{}/api/higgs/models/load", srv.base))
        .json(&serde_json::json!({ "id": TINY_MODEL_ID, "params": suggested_load }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200, "load with tuned params");

    // 3. The resident model loaded with the tuned context window.
    let st: serde_json::Value = c
        .get(format!("{}/api/higgs/status", srv.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        st["loaded"]["ctx_len"]["n"].as_u64().unwrap(),
        tuned_ctx,
        "loaded with the tuned ctx_len: {st}"
    );

    // 4. It serves a (non-streaming) completion with the tuned params applied.
    let chat = c
        .post(format!("{}/v1/chat/completions", srv.base))
        .json(&serde_json::json!({
            "model": TINY_MODEL_ID,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 4,
            "stream": false
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(chat.status(), 200, "served with tuned params");
    let _: serde_json::Value = chat.json().await.unwrap();

    // 5. Unload, then a PLAIN reload (no params) reuses the saved tuning profile.
    c.post(format!("{}/api/higgs/models/unload", srv.base))
        .json(&serde_json::json!({}))
        .send()
        .await
        .unwrap();
    let r2 = c
        .post(format!("{}/api/higgs/models/load", srv.base))
        .json(&serde_json::json!({ "id": TINY_MODEL_ID }))
        .send()
        .await
        .unwrap();
    assert_eq!(r2.status(), 200, "plain reload");

    // 6. The persisted `last_load` carries the tuned optionals (flash_attn on) —
    //    proving the saved profile was reused, not the bare default_load (which
    //    leaves flash_attn unset).
    let models: serde_json::Value = c
        .get(format!("{}/api/higgs/models", srv.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let entry = models["models"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["id"] == TINY_MODEL_ID)
        .expect("model present in catalog");
    assert_eq!(
        entry["last_load"]["flash_attn"], "on",
        "plain load reused the saved tuned profile: {}",
        entry["last_load"]
    );
}

/// An ACCEPTED edit to the suggested params must survive an unload/reload — a
/// plain reload reuses the last accepted load, not the stale tune suggestion. (The
/// successful explicit load syncs the saved profile in `models.json`.)
#[tokio::test]
async fn edited_load_params_survive_reload() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("SKIP edited_params: no tiny GGUF (set HIGGS_TEST_GGUF)");
        return;
    };
    let srv = spawn_with_tiny_model(11543, &gguf).await;
    let c = reqwest::Client::new();

    // Tune, then EDIT the suggestion (threads = 1 is distinctive + always valid).
    let s: serde_json::Value = c
        .post(format!("{}/api/higgs/models/tune", srv.base))
        .json(&serde_json::json!({ "id": TINY_MODEL_ID }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let mut edited = s["load"].clone();
    edited["threads"] = serde_json::json!(1);

    // Load with the edited params.
    let r = c
        .post(format!("{}/api/higgs/models/load", srv.base))
        .json(&serde_json::json!({ "id": TINY_MODEL_ID, "params": edited }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200, "load with edited params");
    let st: serde_json::Value = c
        .get(format!("{}/api/higgs/status", srv.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(st["loaded"]["threads"].as_u64().unwrap(), 1, "edit applied");

    // Unload, then a PLAIN reload must reuse the EDITED threads (1), not the tuned value.
    c.post(format!("{}/api/higgs/models/unload", srv.base))
        .json(&serde_json::json!({}))
        .send()
        .await
        .unwrap();
    c.post(format!("{}/api/higgs/models/load", srv.base))
        .json(&serde_json::json!({ "id": TINY_MODEL_ID }))
        .send()
        .await
        .unwrap();
    let st2: serde_json::Value = c
        .get(format!("{}/api/higgs/status", srv.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        st2["loaded"]["threads"].as_u64().unwrap(),
        1,
        "plain reload reused the accepted EDIT (threads=1), not the stale tune"
    );

    // The model is now RESIDENT. Loading again with a new edit (threads=2) is an
    // idempotent no-op — the request params are NOT applied to the resident worker,
    // and (success-before-persist) are NOT saved to the profile either. So a later
    // plain reload still reuses the PREVIOUS accepted edit (threads=1), not threads=2.
    let mut edited2 = s["load"].clone();
    edited2["threads"] = serde_json::json!(2);
    let r2 = c
        .post(format!("{}/api/higgs/models/load", srv.base))
        .json(&serde_json::json!({ "id": TINY_MODEL_ID, "params": edited2 }))
        .send()
        .await
        .unwrap();
    assert_eq!(r2.status(), 200, "idempotent load of a resident model");
    c.post(format!("{}/api/higgs/models/unload", srv.base))
        .json(&serde_json::json!({}))
        .send()
        .await
        .unwrap();
    c.post(format!("{}/api/higgs/models/load", srv.base))
        .json(&serde_json::json!({ "id": TINY_MODEL_ID }))
        .send()
        .await
        .unwrap();
    let st3: serde_json::Value = c
        .get(format!("{}/api/higgs/status", srv.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        st3["loaded"]["threads"].as_u64().unwrap(),
        1,
        "resident-load edit was a no-op + unsaved (success-before-persist); profile unchanged"
    );
}

/// Two concurrent tunes of DIFFERENT models must BOTH persist their profile — the
/// `models.json` write is serialized + re-read, so neither flush clobbers the
/// other (regression guard for the whole-file-rewrite race). Verified indirectly:
/// a plain reload of each model reuses ITS saved profile (`flash_attn: "on"`),
/// which only survives if its `TuneRecord` was not dropped.
#[tokio::test]
async fn concurrent_tunes_persist_both_profiles() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("SKIP concurrent_tunes: no tiny GGUF (set HIGGS_TEST_GGUF)");
        return;
    };
    let ids = ["zz/alpha", "zz/beta"];
    let srv = spawn_with_models(11542, &gguf, &ids).await;
    let c = reqwest::Client::new();

    // Fire both tunes concurrently — the card-fetch wait keeps them overlapping.
    let tune = |id: &str| {
        let c = c.clone();
        let base = srv.base.clone();
        let id = id.to_string();
        async move {
            c.post(format!("{base}/api/higgs/models/tune"))
                .json(&serde_json::json!({ "id": id }))
                .send()
                .await
                .unwrap()
                .status()
        }
    };
    let (sa, sb) = tokio::join!(tune(ids[0]), tune(ids[1]));
    assert_eq!(sa, 200, "tune {} ok", ids[0]);
    assert_eq!(sb, 200, "tune {} ok", ids[1]);

    // Each model's saved profile survived: a PLAIN load reuses it (flash_attn on).
    for id in ids {
        let r = c
            .post(format!("{}/api/higgs/models/load", srv.base))
            .json(&serde_json::json!({ "id": id }))
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 200, "plain load {id}");
        let models: serde_json::Value = c
            .get(format!("{}/api/higgs/models", srv.base))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        let entry = models["models"]
            .as_array()
            .unwrap()
            .iter()
            .find(|m| m["id"] == id)
            .unwrap_or_else(|| panic!("model {id} present"));
        assert_eq!(
            entry["last_load"]["flash_attn"], "on",
            "concurrent tune for {id} persisted (profile reused on plain load): {}",
            entry["last_load"]
        );
        // Unload so the next model's load isn't gated by a resident worker cap.
        c.post(format!("{}/api/higgs/models/unload", srv.base))
            .json(&serde_json::json!({}))
            .send()
            .await
            .unwrap();
    }
}
