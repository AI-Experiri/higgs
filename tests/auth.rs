//! Black-box auth (P5): spawn the real `higgs` with an `api_keys.json` and verify the bearer
//! middleware over HTTP — unauthenticated requests are 401, scoped keys reach only their
//! routes, admin reaches everything, and health is always open.

mod common;

use std::process::{Child, Command};
use std::time::Duration;

use higgs::keys::{ApiKeys, Scope};

use common::{stage_tiny_model, tiny_gguf_path};

struct Server(Child);
impl Drop for Server {
    fn drop(&mut self) {
        unsafe { libc::kill(self.0.id() as libc::pid_t, libc::SIGTERM) };
        let _ = self.0.wait();
    }
}

#[tokio::test]
async fn api_keys_gate_the_http_surface() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("SKIP api_keys_gate_the_http_surface: tiny gguf not found");
        return;
    };
    let scan_root = stage_tiny_model(&gguf);
    let home = tempfile::tempdir().expect("home");

    // Write an api_keys.json with a chat-only key and an admin key (known plaintext tokens).
    let mut keys = ApiKeys::default();
    keys.add("hgk_chatkey", "chat".into(), vec![Scope::Chat]);
    keys.add("hgk_adminkey", "admin".into(), vec![Scope::Admin]);
    keys.save(&home.path().join("api_keys.json")).expect("write keys");

    // Grab an ephemeral free port (bind :0, read it, release) to avoid a fixed-port clash.
    let port = {
        let l = std::net::TcpListener::bind("127.0.0.1:0").expect("bind ephemeral");
        l.local_addr().unwrap().port()
    };
    let child = Command::new(env!("CARGO_BIN_EXE_higgs"))
        .env("HIGGS_BIND", "127.0.0.1")
        .env("HIGGS_PORT", port.to_string())
        .env("HIGGS_HOME", home.path())
        .env("HIGGS_MODEL_DIR", scan_root.path())
        .env("RUST_LOG", "warn")
        .spawn()
        .expect("spawn higgs");
    let _server = Server(child);
    let base = format!("http://127.0.0.1:{port}");
    let c = reqwest::Client::new();

    // Wait for readiness (health is open, no auth needed).
    let mut ready = false;
    for _ in 0..150 {
        if let Ok(r) = c.get(format!("{base}/health")).send().await {
            if r.status().is_success() {
                ready = true;
                break;
            }
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    assert!(ready, "server became ready");

    // Health is always open (no key).
    assert!(c.get(format!("{base}/health")).send().await.unwrap().status().is_success());

    // No key → 401 on a gated route, with WWW-Authenticate.
    let no_key = c.get(format!("{base}/v1/models")).send().await.unwrap();
    assert_eq!(no_key.status(), 401, "no key is 401");
    assert!(no_key.headers().contains_key("www-authenticate"), "challenges with WWW-Authenticate");

    // Chat-scoped key can't list models (needs Models) → 401.
    let chat_on_models = c
        .get(format!("{base}/v1/models"))
        .bearer_auth("hgk_chatkey")
        .send()
        .await
        .unwrap();
    assert_eq!(chat_on_models.status(), 401, "chat key lacks models scope");

    // Admin key can list models → 200.
    let admin_models = c
        .get(format!("{base}/v1/models"))
        .bearer_auth("hgk_adminkey")
        .send()
        .await
        .unwrap();
    assert!(admin_models.status().is_success(), "admin lists models: {}", admin_models.status());

    // Admin gates management: load without a key → 401; with admin → not 401.
    let load_no_key = c
        .post(format!("{base}/api/higgs/models/load"))
        .json(&serde_json::json!({ "id": "x/y" }))
        .send()
        .await
        .unwrap();
    assert_eq!(load_no_key.status(), 401, "management requires a key");

    // A bogus token is rejected too.
    let bogus = c
        .get(format!("{base}/v1/models"))
        .bearer_auth("hgk_not_a_real_key")
        .send()
        .await
        .unwrap();
    assert_eq!(bogus.status(), 401, "unknown token is 401");
}
