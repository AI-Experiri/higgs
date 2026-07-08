//! Black-box auth (P5): mint API keys via the in-process facade, then drive the REAL `/v1`
//! HTTP surface (`serve_v1_local`) to verify the bearer middleware — unauthenticated requests
//! are 401 (with a `WWW-Authenticate` challenge + the HG048 diagnostic), scoped keys reach only
//! their routes, admin reaches everything, and `/health` is always open.

mod common;

use higgs::keys::Scope;

use common::{higgs_local, serve_v1_local, TINY_MODEL_ID};

#[tokio::test]
async fn api_keys_gate_the_http_surface() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP api_keys_gate_the_http_surface: tiny gguf not found");
        return;
    };

    // Mint keys through the trusted in-process facade. The FIRST key must include `admin`
    // (bootstrap invariant), so mint the admin key first, then a chat-only key. Minting makes
    // the keystore non-empty, which turns the bearer middleware ON for the served `/v1` surface.
    let admin_token = higgs
        .mint_key("admin", Some(vec![Scope::Admin]))
        .expect("mint admin key")
        .token;
    let chat_token = higgs
        .mint_key("chat", Some(vec![Scope::Chat]))
        .expect("mint chat key")
        .token;

    let (base, guard) = serve_v1_local(higgs.handle()).await;
    let c = reqwest::Client::new();

    // Health is always open (no key).
    assert!(c
        .get(format!("{base}/health"))
        .send()
        .await
        .unwrap()
        .status()
        .is_success());

    // No key → 401 on a gated route, with WWW-Authenticate.
    let no_key = c.get(format!("{base}/v1/models")).send().await.unwrap();
    assert_eq!(no_key.status(), 401, "no key is 401");
    assert!(
        no_key.headers().contains_key("www-authenticate"),
        "challenges with WWW-Authenticate"
    );
    // The 401 body carries the HG048 diagnostic code + resolution, so an auth
    // failure is as diagnosable as every other reply (fails-on-revert: drop the
    // coded message in `unauthorized()` and the body no longer mentions HG048).
    let no_key_body = no_key.text().await.unwrap();
    assert!(
        no_key_body.contains("[HG048]") && no_key_body.contains("Authorization: Bearer"),
        "401 body carries the HG048 code + resolution, got: {no_key_body}"
    );

    // Chat-scoped key can't list models (needs Models) → 401.
    let chat_on_models = c
        .get(format!("{base}/v1/models"))
        .bearer_auth(&chat_token)
        .send()
        .await
        .unwrap();
    assert_eq!(chat_on_models.status(), 401, "chat key lacks models scope");

    // Admin key can list models → 200.
    let admin_models = c
        .get(format!("{base}/v1/models"))
        .bearer_auth(&admin_token)
        .send()
        .await
        .unwrap();
    assert!(
        admin_models.status().is_success(),
        "admin lists models: {}",
        admin_models.status()
    );

    // The gated chat route also requires a key: no key → 401 (auth runs before the handler, so
    // this is a pure auth check independent of any model being loaded). This preserves the
    // original "a gated route with no key is 401" intent — the old assertion pointed at the
    // now-deleted `/api/higgs/models/load` management route; management moved to the crate API
    // and is no longer part of the HTTP surface, so we re-target the check at the served `/v1`
    // chat route.
    let chat_no_key = c
        .post(format!("{base}/v1/chat/completions"))
        .json(&serde_json::json!({ "model": TINY_MODEL_ID, "messages": [] }))
        .send()
        .await
        .unwrap();
    assert_eq!(chat_no_key.status(), 401, "gated chat route requires a key");

    // A bogus token is rejected too.
    let bogus = c
        .get(format!("{base}/v1/models"))
        .bearer_auth("hgk_not_a_real_key")
        .send()
        .await
        .unwrap();
    assert_eq!(bogus.status(), 401, "unknown token is 401");

    guard.shutdown().await;
    higgs.shutdown().await;
}
