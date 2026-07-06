//! Black-box integration tests for the G4 key-management surface
//! (`/api/higgs/keys`): mint → hot-swap auth ON → scope enforcement → list
//! (no plaintext) → revoke → auth OFF, plus the CORS-preflight-before-auth
//! ordering guarantee.
//!
//! Fail-on-revert: removing the routes 404s the lifecycle; removing the
//! hot-swap (`set_api_keys` inside `mutate_api_keys`) breaks the
//! "unkeyed request 401s IMMEDIATELY after mint" assertion (the old
//! CLI-only flow needed a restart).

mod common;

use common::{spawn_lan_keyed, spawn_with_tiny_model, tiny_gguf_path};

#[tokio::test]
async fn keys_lifecycle_mint_scope_list_revoke() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("SKIP keys_lifecycle: tiny gguf not found (set HIGGS_TEST_GGUF)");
        return;
    };
    let srv = spawn_with_tiny_model(13450, &gguf).await;
    let c = reqwest::Client::new();
    let status_url = format!("{}/api/higgs/status", srv.base);
    let keys_url = format!("{}/api/higgs/keys", srv.base);

    // Fresh home ⇒ empty keystore ⇒ auth OFF: the control surface is open.
    let r = c.get(&status_url).send().await.expect("status");
    assert!(r.status().is_success(), "auth off: unkeyed status serves");

    // The FIRST key must be able to manage keys: an explicit non-admin bootstrap
    // mint is rejected (codex r10 — it would self-lock the management API).
    let r = c
        .post(&keys_url)
        .json(&serde_json::json!({ "label": "chatonly", "scopes": ["chat"] }))
        .send()
        .await
        .expect("bootstrap chat-only mint");
    assert_eq!(r.status(), 400, "non-admin first key refused");
    // Still empty (nothing minted) — auth stays off.
    assert!(
        c.get(&status_url)
            .send()
            .await
            .expect("status")
            .status()
            .is_success(),
        "refused bootstrap mint left the store empty"
    );

    // Mint an ADMIN key over HTTP. The response carries the plaintext ONCE.
    let r = c
        .post(&keys_url)
        .json(&serde_json::json!({ "label": "admin", "scopes": ["admin"] }))
        .send()
        .await
        .expect("mint admin");
    assert!(r.status().is_success(), "mint: {}", r.status());
    let minted: serde_json::Value = r.json().await.expect("mint json");
    let admin_token = minted["token"].as_str().expect("token").to_owned();
    assert!(
        admin_token.starts_with("hgk_"),
        "token prefix: {admin_token}"
    );

    // HOT-SWAP: the very NEXT unkeyed request must 401 — no restart involved.
    let r = c.get(&status_url).send().await.expect("status unkeyed");
    assert_eq!(r.status(), 401, "auth flipped on by the mint");
    assert!(
        r.headers().get("www-authenticate").is_some(),
        "401 carries WWW-Authenticate"
    );
    let r = c
        .get(&status_url)
        .bearer_auth(&admin_token)
        .send()
        .await
        .expect("status keyed");
    assert!(r.status().is_success(), "admin key reaches Admin routes");

    // CORS preflight is answered BEFORE the auth guard: an OPTIONS from a
    // browser origin needs NO key even with auth on (the layer order
    // CORS → host → auth is a wire contract, not an accident).
    let r = c
        .request(
            reqwest::Method::OPTIONS,
            format!("{}/v1/chat/completions", srv.base),
        )
        .header("Origin", "http://localhost:5173")
        .header("Access-Control-Request-Method", "POST")
        .send()
        .await
        .expect("preflight");
    assert!(
        r.status().is_success(),
        "unkeyed preflight passes with auth on: {}",
        r.status()
    );

    // Mint a MODELS-scoped key (using the admin bearer).
    let r = c
        .post(&keys_url)
        .bearer_auth(&admin_token)
        .json(&serde_json::json!({ "label": "reader", "scopes": ["models"] }))
        .send()
        .await
        .expect("mint reader");
    assert!(r.status().is_success());
    let reader_token = r.json::<serde_json::Value>().await.expect("json")["token"]
        .as_str()
        .expect("token")
        .to_owned();

    // Scope enforcement: reader lists models but cannot reach Admin routes.
    let r = c
        .get(format!("{}/v1/models", srv.base))
        .bearer_auth(&reader_token)
        .send()
        .await
        .expect("models w/ reader");
    assert!(r.status().is_success(), "models scope covers /v1/models");
    let r = c
        .get(&status_url)
        .bearer_auth(&reader_token)
        .send()
        .await
        .expect("status w/ reader");
    assert_eq!(r.status(), 401, "models scope must NOT reach Admin");

    // List: both labels, auth flagged on, and NO plaintext token anywhere.
    let r = c
        .get(&keys_url)
        .bearer_auth(&admin_token)
        .send()
        .await
        .expect("list");
    assert!(r.status().is_success());
    let body = r.text().await.expect("list body");
    assert!(body.contains("\"admin\"") && body.contains("\"reader\""));
    assert!(body.contains("\"auth_enabled\":true"));
    assert!(
        !body.contains(&admin_token) && !body.contains(&reader_token),
        "plaintext must never appear in the list: {body}"
    );

    // Unroutable label (path separator) → coded 400 at mint, never persisted.
    let r = c
        .post(&keys_url)
        .bearer_auth(&admin_token)
        .json(&serde_json::json!({ "label": "bad/label" }))
        .send()
        .await
        .expect("mint slash label");
    assert_eq!(
        r.status(),
        400,
        "slash labels are unrevokable — rejected at mint"
    );

    // Duplicate label → coded 400 (HG049), nothing minted.
    let r = c
        .post(&keys_url)
        .bearer_auth(&admin_token)
        .json(&serde_json::json!({ "label": "reader" }))
        .send()
        .await
        .expect("dup mint");
    assert_eq!(r.status(), 400);
    assert!(r.text().await.expect("body").contains("[HG049]"));

    // The persisted store holds digests only — never a plaintext token.
    let persisted =
        std::fs::read_to_string(srv.home().join("api_keys.json")).expect("api_keys.json");
    assert!(
        !persisted.contains(&admin_token) && !persisted.contains(&reader_token),
        "persisted store is digest-only"
    );

    // Revoke reader: it stops working on the NEXT request.
    let r = c
        .delete(format!("{keys_url}/reader"))
        .bearer_auth(&admin_token)
        .send()
        .await
        .expect("revoke reader");
    assert!(r.status().is_success());
    let r = c
        .get(format!("{}/v1/models", srv.base))
        .bearer_auth(&reader_token)
        .send()
        .await
        .expect("models w/ revoked reader");
    assert_eq!(r.status(), 401, "revoked key is dead immediately");

    // Revoking an unknown label is a coded 400.
    let r = c
        .delete(format!("{keys_url}/ghost"))
        .bearer_auth(&admin_token)
        .send()
        .await
        .expect("revoke ghost");
    assert_eq!(r.status(), 400);

    // Revoke the LAST key: auth turns OFF, the surface reopens (loopback bind).
    let r = c
        .delete(format!("{keys_url}/admin"))
        .bearer_auth(&admin_token)
        .send()
        .await
        .expect("revoke admin");
    assert!(r.status().is_success());
    let removed: serde_json::Value = r.json().await.expect("json");
    assert_eq!(removed["auth_enabled"], false);
    let r = c.get(&status_url).send().await.expect("status open again");
    assert!(
        r.status().is_success(),
        "zero keys ⇒ auth off ⇒ unkeyed serves again"
    );
}

/// G4 keyed-LAN mode is USABLE: on a non-loopback bind, a client whose `Host`
/// header carries the server's LAN address (what every real LAN client sends)
/// must reach the surface with a valid key — the DNS-rebinding Host guard is
/// bind-aware, not unconditional. Fail-on-revert: an unconditional loopback
/// Host guard 403s ([HG012]) before auth and this test fails.
#[tokio::test]
async fn keyed_lan_bind_accepts_lan_host_headers() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("SKIP keyed_lan: tiny gguf not found (set HIGGS_TEST_GGUF)");
        return;
    };
    let (srv, token) = spawn_lan_keyed(13460, &gguf).await;
    let c = reqwest::Client::new();

    // A LAN-style Host value + valid bearer serves.
    let r = c
        .get(format!("{}/api/higgs/status", srv.base))
        .header("host", "192.168.1.50:13460")
        .bearer_auth(&token)
        .send()
        .await
        .expect("status w/ LAN host");
    assert!(
        r.status().is_success(),
        "keyed LAN client with LAN Host must serve, got {}",
        r.status()
    );

    // Auth still gates everything: same LAN Host, no key → 401 (not 403).
    let r = c
        .get(format!("{}/api/higgs/status", srv.base))
        .header("host", "192.168.1.50:13460")
        .send()
        .await
        .expect("status unkeyed");
    assert_eq!(r.status(), 401, "keys, not Host filtering, gate the LAN");

    // [HG059]: the LAST key cannot be revoked while LAN-exposed — revoking it
    // would reopen the whole surface at runtime (auth off + relaxed Host
    // guard), bypassing the [HG058] startup protection.
    let r = c
        .delete(format!("{}/api/higgs/keys/lan-admin", srv.base))
        .bearer_auth(&token)
        .send()
        .await
        .expect("revoke last on LAN");
    assert_eq!(r.status(), 409, "last-key revoke refused on a LAN bind");
    assert!(r.text().await.expect("body").contains("[HG059]"));

    // With a replacement minted, the original CAN be revoked (not last).
    let r = c
        .post(format!("{}/api/higgs/keys", srv.base))
        .bearer_auth(&token)
        .json(&serde_json::json!({ "label": "rotated", "scopes": ["admin"] }))
        .send()
        .await
        .expect("mint replacement");
    assert!(r.status().is_success());
    let r = c
        .delete(format!("{}/api/higgs/keys/lan-admin", srv.base))
        .bearer_auth(&token)
        .send()
        .await
        .expect("revoke rotated-out key");
    assert!(r.status().is_success(), "not-last revoke proceeds on LAN");
}

/// The auth middleware stamps `last_used_ms` on the key that authorized a
/// request, and mint stamps `created_at_ms`. Fail-on-revert for the
/// `auth_guard` → `touch_api_key` wiring: dropping the touch call leaves
/// `last_used_ms` null forever, even after many authorized requests.
///
/// (Every `/api/higgs/keys` GET is itself an authorized request that stamps
/// last-used before its handler reads the store, so a "never used" state
/// isn't observable through an authed endpoint — we assert the stamp is
/// PRESENT after auth, which the wiring provides and reverting it removes.)
#[tokio::test]
async fn authorized_request_stamps_created_and_last_used() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("SKIP last_used: tiny gguf not found (set HIGGS_TEST_GGUF)");
        return;
    };
    let (srv, token) = spawn_lan_keyed(13470, &gguf).await;
    let c = reqwest::Client::new();
    let keys_url = format!("{}/api/higgs/keys", srv.base);

    // A couple of authorized requests to exercise the touch path.
    for _ in 0..2 {
        let r = c
            .get(format!("{}/api/higgs/status", srv.base))
            .bearer_auth(&token)
            .send()
            .await
            .expect("authorized status");
        assert!(r.status().is_success());
    }

    let list: serde_json::Value = c
        .get(&keys_url)
        .bearer_auth(&token)
        .send()
        .await
        .expect("list")
        .json()
        .await
        .expect("json");
    let entry = &list["keys"][0];
    assert!(
        entry["created_at_ms"].as_u64().unwrap_or(0) > 0,
        "mint must stamp created_at_ms, got {entry}"
    );
    assert!(
        entry["last_used_ms"].as_u64().unwrap_or(0) > 0,
        "an authorized request must stamp last_used_ms, got {entry}"
    );
}
