//! In-process integration tests for higgs's G4 key-management surface — now the
//! crate API (`Higgs::mint_key` / `revoke_key` / `api_keys`).
//!
//! The old `/api/higgs/keys` HTTP surface is GONE; key management runs in-process
//! through the facade. These tests preserve the load-bearing INVARIANTS the shared
//! decision cores enforce: bootstrap-needs-admin, label validation, the last-admin
//! lockout ([HG066]), the last-key-on-LAN refusal ([HG059]), digest-only
//! persistence, and the created/last-used stamping. The single behavior that is
//! genuinely HTTP — the `auth_guard → touch_api_key` wiring — is exercised over the
//! real `/v1` surface via `serve_v1_local`.

mod common;

use common::{higgs_local, serve_v1_local, TINY_MODEL_ID};
use higgs::{HiggsError, Scope};

#[tokio::test]
async fn keys_lifecycle_mint_scope_list_revoke() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP keys_lifecycle: tiny gguf not found (set HIGGS_TEST_GGUF)");
        return;
    };

    // Fresh home ⇒ empty keystore ⇒ auth OFF.
    assert!(
        higgs.api_keys().is_empty(),
        "fresh home starts with no keys"
    );

    // The FIRST key must be able to manage keys: an explicit non-admin bootstrap
    // mint is rejected (it would self-lock the management API).
    let boot = higgs.mint_key("chatonly", Some(vec![Scope::Chat]));
    assert!(
        matches!(boot, Err(HiggsError::InvalidRequest { .. })),
        "non-admin first key refused: {boot:?}"
    );
    assert!(
        higgs.api_keys().is_empty(),
        "refused bootstrap mint left the store empty"
    );

    // Mint an ADMIN key. The response carries the plaintext ONCE.
    let admin = higgs
        .mint_key("admin", Some(vec![Scope::Admin]))
        .expect("mint admin");
    assert!(
        admin.token.starts_with("hgk_"),
        "token prefix: {}",
        admin.token
    );
    let admin_token = admin.token.clone();
    // The mint flipped auth ON (the live store is now non-empty).
    assert!(!higgs.api_keys().is_empty(), "auth flipped on by the mint");

    // Mint a MODELS-scoped key.
    let reader = higgs
        .mint_key("reader", Some(vec![Scope::Models]))
        .expect("mint reader");
    let reader_token = reader.token.clone();

    // List (`api_keys`): both labels present, no plaintext token stored anywhere.
    let keys = higgs.api_keys();
    let labels: Vec<&str> = keys.iter().map(|k| k.label.as_str()).collect();
    assert!(
        labels.contains(&"admin") && labels.contains(&"reader"),
        "both labels listed: {labels:?}"
    );
    // The store holds sha digests + labels only — an ApiKey has no plaintext field.
    // The persisted file must also be digest-only (never a plaintext token).
    let persisted =
        std::fs::read_to_string(higgs.home().join("api_keys.json")).expect("api_keys.json");
    assert!(
        !persisted.contains(&admin_token) && !persisted.contains(&reader_token),
        "persisted store is digest-only"
    );

    // Unroutable label (path separator) → rejected at mint, never persisted.
    let slash = higgs.mint_key("bad/label", None);
    assert!(
        matches!(slash, Err(HiggsError::InvalidRequest { .. })),
        "slash labels are unrevokable — rejected at mint: {slash:?}"
    );

    // Duplicate label → rejected, nothing minted.
    let dup = higgs.mint_key("reader", None);
    assert!(
        matches!(&dup, Err(HiggsError::InvalidRequest { detail }) if detail.contains("already exists")),
        "duplicate label refused: {dup:?}"
    );

    // ── [HG066] last-admin lockout: revoking the LAST Admin-capable key while
    // OTHER (non-Admin) keys remain is refused — it would lock out key management.
    let last_admin = higgs.revoke_key("admin");
    assert!(
        matches!(last_admin, Err(HiggsError::LastAdminKey { .. })),
        "revoking the last admin while a non-admin key remains is refused [HG066]: {last_admin:?}"
    );

    // Revoking a NON-admin key is allowed (it can't strand the management surface).
    let removed = higgs.revoke_key("reader").expect("revoke reader");
    assert_eq!(removed.removed, 1, "one key removed");
    assert!(removed.auth_enabled, "admin remains → auth still on");

    // Revoking an unknown label is an error.
    let ghost = higgs.revoke_key("ghost");
    assert!(
        matches!(ghost, Err(HiggsError::InvalidRequest { .. })),
        "revoking an unknown label errors: {ghost:?}"
    );

    // Now admin is the ONLY key: revoking it empties the store (auth OFF) — allowed.
    let removed = higgs.revoke_key("admin").expect("revoke admin");
    assert!(
        !removed.auth_enabled,
        "revoking the last key turns auth off"
    );
    assert!(
        higgs.api_keys().is_empty(),
        "zero keys ⇒ auth off ⇒ store empty again"
    );

    higgs.shutdown().await;
}

/// [HG059] the LAST key cannot be revoked while LAN-exposed — revoking it would
/// reopen the whole surface at runtime (auth off), bypassing the startup guard. Once
/// a replacement is minted, the original CAN be revoked (not last).
#[tokio::test]
async fn last_key_revoke_refused_while_lan_exposed() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP last_key_lan: tiny gguf not found (set HIGGS_TEST_GGUF)");
        return;
    };

    let _admin = higgs
        .mint_key("lan-admin", Some(vec![Scope::Admin]))
        .expect("mint admin");
    // Mark the surface as bound beyond loopback (what `serve_v1` records on a LAN bind).
    higgs.set_lan_exposed(true);

    // The last key cannot be revoked while LAN-exposed.
    let last = higgs.revoke_key("lan-admin");
    assert!(
        matches!(last, Err(HiggsError::LastKeyOnLan { .. })),
        "last-key revoke refused on a LAN bind [HG059]: {last:?}"
    );

    // With a replacement minted, the original CAN be revoked (not last).
    higgs
        .mint_key("rotated", Some(vec![Scope::Admin]))
        .expect("mint replacement");
    higgs
        .revoke_key("lan-admin")
        .expect("not-last revoke proceeds on LAN");

    higgs.shutdown().await;
}

/// Mint stamps `created_at_ms`, and an AUTHORIZED request over the real `/v1`
/// surface stamps `last_used_ms` on the key that authorized it (the
/// `auth_guard → touch_api_key` wiring). Fail-on-revert: dropping the touch call in
/// `auth_guard` leaves `last_used_ms` null even after authorized requests.
#[tokio::test]
async fn authorized_request_stamps_created_and_last_used() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP last_used: tiny gguf not found (set HIGGS_TEST_GGUF)");
        return;
    };

    // Mint an Admin key (Admin satisfies any scope, incl. /v1/models).
    let admin = higgs
        .mint_key("admin", Some(vec![Scope::Admin]))
        .expect("mint admin");
    let token = admin.token.clone();

    // Serve `/v1` on loopback; the non-empty keystore means auth is ON.
    let (base, guard) = serve_v1_local(higgs.handle()).await;
    let client = reqwest::Client::new();

    // A couple of AUTHORIZED requests to exercise the touch path.
    for _ in 0..2 {
        let r = client
            .get(format!("{base}/v1/models"))
            .bearer_auth(&token)
            .send()
            .await
            .expect("authorized /v1/models");
        assert!(
            r.status().is_success(),
            "admin bearer reaches /v1/models: {}",
            r.status()
        );
    }

    // The key now carries a mint `created_at_ms` and an auth `last_used_ms`.
    let keys = higgs.api_keys();
    let entry = keys
        .iter()
        .find(|k| k.label == "admin")
        .expect("admin key present");
    assert!(
        entry.created_at_ms.unwrap_or(0) > 0,
        "mint stamps created_at_ms, got {entry:?}"
    );
    assert!(
        entry.last_used_ms.unwrap_or(0) > 0,
        "an authorized request stamps last_used_ms, got {entry:?}"
    );

    guard.shutdown().await;
    higgs.shutdown().await;
}
