//! Embedding-only models must never be served as chat models.
//!
//! A GGUF converted for embedding (a pooling head and/or non-causal attention) cannot
//! generate text — but llama.cpp does not refuse to sample from one. It happily returns
//! a fluent-looking sequence of unrelated tokens with `finish_reason: "length"` and a
//! 200. Silently-wrong output is the worst failure mode there is, so higgs classifies
//! the model at scan time ([`higgs`'s `ModelDomain`]) and refuses chat against it with
//! `[HG079]`.
//!
//! This drives the REAL `/v1` router over HTTP (the same surface an OpenAI client hits)
//! against an embedding GGUF staged on disk, and asserts the two user-visible contracts:
//! chat is REFUSED (not answered with nonsense), and the model is not advertised as a
//! chat target.

mod common;

use common::{higgs_local, serve_v1_local, stage_embedding_model, TINY_MODEL_ID};
use higgs::TuneRequest;
use serde_json::json;

/// The id the synthetic embedding GGUF is staged under.
const EMBED_ID: &str = "higgs-test/bge-tiny";

#[tokio::test]
async fn an_embedding_model_is_refused_for_chat_and_never_advertised() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP embedding gate: no tiny GGUF (set HIGGS_TEST_GGUF)");
        return;
    };
    // Stage the embedding model beside the (generative) tiny one, in the same scan root.
    stage_embedding_model(higgs.scan_root(), EMBED_ID);

    // Tune it, so that EVERY other gate says "serve this": it now has a fresh profile
    // that fits free memory with serving on — i.e. `Servable`, and advertised on
    // `/v1/models`, for any reason OTHER than its domain. Without this the model would
    // be withheld merely for being untuned and the assertions below would pass
    // vacuously. The domain must be the ONLY thing holding it back.
    higgs
        .tune(TuneRequest {
            id: EMBED_ID.to_owned(),
            mode: None,
            budget: None,
            pins: None,
        })
        .await
        .expect("tune (prepare) the embedding model");

    let (base, _guard) = serve_v1_local(higgs.clone()).await;
    let client = reqwest::Client::new();

    // ── The bug: chat against it used to JIT-load it and return sampled garbage. ──
    let resp = client
        .post(format!("{base}/v1/chat/completions"))
        .json(&json!({
            "model": EMBED_ID,
            "messages": [{ "role": "user", "content": "hi" }],
            "max_tokens": 16,
        }))
        .send()
        .await
        .expect("chat request");

    let status = resp.status();
    let body: serde_json::Value = resp.json().await.expect("error envelope");
    let message = body["error"]["message"].as_str().unwrap_or_default();

    assert_eq!(
        status, 400,
        "chat against an embedding model must be REFUSED, not answered; body: {body}"
    );
    assert!(
        message.contains("[HG079]"),
        "the refusal must carry the embedding-model code so a client can act on it; got: {message}"
    );

    // ── `stream: true` refuses the same way, BEFORE any SSE commits: the gate
    // runs pre-stream, so a streaming client gets the same HTTP 400, not a 200
    // whose error arrives as an in-stream event. ──
    let resp = client
        .post(format!("{base}/v1/chat/completions"))
        .json(&json!({
            "model": EMBED_ID,
            "stream": true,
            "messages": [ { "role": "user", "content": "hi" } ],
            "max_tokens": 16,
        }))
        .send()
        .await
        .expect("streaming chat request");
    assert_eq!(
        resp.status(),
        400,
        "a streaming chat against an embedding model must 400 before SSE commits"
    );

    // ── …and it must not be advertised to OpenAI clients as a chat model. ──
    let listed: serde_json::Value = client
        .get(format!("{base}/v1/models"))
        .send()
        .await
        .expect("models request")
        .json()
        .await
        .expect("models body");
    let ids: Vec<&str> = listed["data"]
        .as_array()
        .expect("data array")
        .iter()
        .filter_map(|m| m["id"].as_str())
        .collect();
    assert!(
        !ids.contains(&EMBED_ID),
        "/v1/models must not advertise an embedding model as a chat target; got {ids:?}"
    );

    // ── …and the catalog badges it for what it is. ──
    // It IS discovered — this test is about how it is CLASSIFIED, not about hiding it
    // from the catalog. A user must still see it on disk (and, later, embed with it).
    let entries = higgs.model_entries().await.expect("model entries");
    let entry = entries
        .iter()
        .find(|e| e.model.id == EMBED_ID)
        .expect("the embedding model is scanned and listed in the catalog");
    assert_eq!(
        entry.readiness,
        higgs::ModelReadiness::Embedding,
        "the catalog must badge it Embedding, not offer it as a chat target"
    );
}
