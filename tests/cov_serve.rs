//! In-process integration coverage for the `/v1` HTTP surface + serve layer that
//! the other `tests/*.rs` files leave uncovered: SSE/router edges in
//! `src/serve/{v1,stream,mod,control,readiness}.rs`.
//!
//! higgs is library-first: control (load / tune / mint-key / model-entries) runs
//! through the in-process `Higgs` facade, and the REAL `/v1` HTTP surface is
//! served by `serve_v1_local` (loopback) — or, for the keyed-LAN paths, by
//! `serve_v1` bound to `0.0.0.0` here. A REAL local llama.cpp worker runs via the
//! `worker_exe` DI seam, so `load` / `chat` exercise the full engine path.
//!
//! Every test that opens an SSE / HTTP body drains it to completion before
//! `guard.shutdown()` (an open stream blocks graceful shutdown), then
//! `higgs.shutdown()`. Each skips cleanly when the tiny GGUF is absent.

mod common;

use std::sync::Arc;
use std::time::Duration;

use common::{higgs_local, serve_v1_local, TINY_MODEL_ID};
use higgs::serve::readiness::ModelReadiness;
use higgs::worker::engine::llamacpp::params::LlamaCppParams;
use higgs::worker::engine::CtxLen;
use higgs::{Higgs, HiggsError, LoadParams, Scope, TuneRequest};
use serde_json::{json, Value};

// ── small helpers (inlined — the shared harness is off-limits) ────────────────

/// Load the tiny model with an explicit fixed context window (deterministic
/// prompt-fit behavior regardless of the model's trained window).
async fn load_tiny_ctx(h: &Higgs, n: u32) {
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

/// Prepare (analytical tune) the tiny model → a saved profile with anchors.
async fn tune_tiny(h: &Higgs) {
    h.tune(TuneRequest {
        id: TINY_MODEL_ID.to_owned(),
        mode: None,
        budget: None,
        pins: None,
    })
    .await
    .expect("prepare (tune) tiny model succeeded");
}

/// Serve `handle` on `bind` (e.g. `"0.0.0.0:0"`) via the REAL `serve_v1`, polling
/// `/health` (over loopback) until it answers. Returns the base URL, the graceful
/// shutdown sender, and the server join handle. For the keyed-LAN paths that a
/// loopback-only `serve_v1_local` can't reach.
async fn serve_v1_on(
    handle: Arc<Higgs>,
    bind: &str,
) -> (
    String,
    tokio::sync::oneshot::Sender<()>,
    tokio::task::JoinHandle<std::io::Result<()>>,
) {
    let listener = tokio::net::TcpListener::bind(bind)
        .await
        .unwrap_or_else(|e| panic!("bind {bind}: {e}"));
    let port = listener.local_addr().expect("local_addr").port();
    let (tx, rx) = tokio::sync::oneshot::channel::<()>();
    let join = tokio::spawn(higgs::serve::serve_v1(handle, listener, async move {
        let _ = rx.await;
    }));
    let base = format!("http://127.0.0.1:{port}");
    let c = reqwest::Client::new();
    for _ in 0..200 {
        if let Ok(r) = c.get(format!("{base}/health")).send().await {
            if r.status().is_success() {
                return (base, tx, join);
            }
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("serve_v1 never became ready on {base}");
}

// ── serving toggle: chat refused with 503 [HG019] ─────────────────────────────
//
// The `/v1` inference surface refuses when serving is toggled OFF, BEFORE any
// gate/JIT/worker RPC. Covers `v1_chat_completions`'s serving-gate short-circuit
// and the `ServingDisabled → 503` status arm. Fail-on-revert: drop the
// `!serving_enabled()` guard and the request 404s/loads instead of 503.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn serving_disabled_chat_is_503_hg019() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP serving_disabled_chat_is_503_hg019: tiny gguf not found");
        return;
    };
    higgs.set_serving_enabled(false);
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
    assert_eq!(
        resp.status(),
        503,
        "serving-off chat is a 503 (ServingDisabled)"
    );
    let body = resp.text().await.unwrap();
    assert!(
        body.contains("[HG019]"),
        "503 body carries the HG019 code: {body}"
    );

    guard.shutdown().await;
    higgs.shutdown().await;
}

// ── sampling-param + text-only-content edges → 400 ────────────────────────────
//
// One loaded model, several requests, each hitting a distinct branch of
// `validate_sampling` / `messages_to_pairs` that the other test files don't:
//   * temperature < 0            → 400 [HG013]
//   * frequency_penalty out of [-2,2] → 400 [HG013]
//   * n != 1                     → 400 [HG013]
//   * assistant refusal part     → 400 (v1 is text-only)
//   * chat_template_kwargs that is NOT a JSON object → ignored, request 200s
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chat_param_and_content_edges() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP chat_param_and_content_edges: tiny gguf not found");
        return;
    };
    load_tiny_ctx(&higgs, 512).await;
    let (base, guard) = serve_v1_local(higgs.handle()).await;
    let c = reqwest::Client::new();
    let url = format!("{base}/v1/chat/completions");

    // temperature < 0 → 400 naming the offending param.
    let resp = c
        .post(&url)
        .json(&json!({
            "model": TINY_MODEL_ID, "stream": false, "temperature": -0.5,
            "messages": [{ "role": "user", "content": "hi" }]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400, "negative temperature is a 400");
    let env: Value = resp.json().await.unwrap();
    let msg = env["error"]["message"].as_str().unwrap_or_default();
    assert!(msg.contains("[HG013]"), "HG013: {env:?}");
    assert!(msg.contains("temperature"), "names temperature: {env:?}");

    // frequency_penalty out of [-2, 2] → 400.
    let resp = c
        .post(&url)
        .json(&json!({
            "model": TINY_MODEL_ID, "stream": false, "frequency_penalty": 5.0,
            "messages": [{ "role": "user", "content": "hi" }]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        400,
        "frequency_penalty out of range is a 400"
    );
    let env: Value = resp.json().await.unwrap();
    let msg = env["error"]["message"].as_str().unwrap_or_default();
    assert!(msg.contains("[HG013]"), "HG013: {env:?}");
    assert!(
        msg.contains("frequency_penalty"),
        "names frequency_penalty: {env:?}"
    );

    // n != 1 (higgs serves a single choice) → 400.
    let resp = c
        .post(&url)
        .json(&json!({
            "model": TINY_MODEL_ID, "stream": false, "n": 3,
            "messages": [{ "role": "user", "content": "hi" }]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400, "n>1 is a 400");
    let env: Value = resp.json().await.unwrap();
    assert!(
        env["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("[HG013]"),
        "HG013: {env:?}"
    );

    // An assistant refusal content part → 400 (v1 is text-only).
    let resp = c
        .post(&url)
        .json(&json!({
            "model": TINY_MODEL_ID, "stream": false,
            "messages": [
                { "role": "user", "content": "hi" },
                { "role": "assistant", "content": [{ "type": "refusal", "refusal": "no" }] }
            ]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400, "assistant refusal part is a 400");
    let body = resp.text().await.unwrap();
    assert!(
        body.contains("refusal"),
        "the 400 explains the refusal rejection: {body}"
    );

    // chat_template_kwargs that is NOT an object is ignored (warn + None), and the
    // request still serves → 200.
    let resp = c
        .post(&url)
        .json(&json!({
            "model": TINY_MODEL_ID, "stream": false, "max_tokens": 8,
            "chat_template_kwargs": "not-an-object",
            "messages": [{ "role": "user", "content": "hi" }]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        200,
        "a non-object chat_template_kwargs is ignored, not rejected"
    );
    let body: Value = resp.json().await.unwrap();
    assert!(
        body["choices"][0]["message"]["content"].is_string(),
        "a completion was still generated: {body:?}"
    );

    guard.shutdown().await;
    higgs.shutdown().await;
}

// ── per-role text-array content flattening passes the v1 content gate ──────────
//
// `messages_to_pairs` (the v1 text-only validator) flattens developer / assistant
// / tool / function messages whose content is a TEXT array. Those array arms are
// otherwise unexercised. The text arrays must be ACCEPTED (no "text-only"
// rejection) — the contrast with the refusal rejection above proves the policy.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn message_role_text_arrays_pass_content_gate() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP message_role_text_arrays_pass_content_gate: tiny gguf not found");
        return;
    };
    load_tiny_ctx(&higgs, 512).await;
    let (base, guard) = serve_v1_local(higgs.handle()).await;
    let c = reqwest::Client::new();
    let url = format!("{base}/v1/chat/completions");

    // developer + system + assistant with TEXT-ARRAY content, plus tool + function
    // messages (also text). All are valid text → the content gate must not reject
    // any as a non-text part.
    let resp = c
        .post(&url)
        .json(&json!({
            "model": TINY_MODEL_ID, "stream": false, "max_tokens": 8,
            "messages": [
                { "role": "system", "content": [{ "type": "text", "text": "be terse" }] },
                { "role": "developer", "content": [{ "type": "text", "text": "dev note" }] },
                { "role": "user", "content": "hi" },
                { "role": "assistant", "content": [{ "type": "text", "text": "hello" }] },
                { "role": "tool", "tool_call_id": "call_1",
                  "content": [{ "type": "text", "text": "tool result" }] },
                { "role": "function", "name": "f", "content": "fn result" },
                { "role": "user", "content": "and now?" }
            ]
        }))
        .send()
        .await
        .unwrap();
    let status = resp.status();
    let body = resp.text().await.unwrap();
    // The v1 text-only content gate accepted every text array (whatever the model
    // then does downstream): the response is NOT the messages_to_pairs rejection.
    assert!(
        !body.contains("text-only"),
        "text-array content of every role passes the v1 content gate, got {status}: {body}"
    );

    guard.shutdown().await;
    higgs.shutdown().await;
}

// ── verbose serving line + incoming-prompt logging paths ──────────────────────
//
// With verbose + "log incoming tokens" ON, a completed non-streaming chat emits
// the `log_served` line and the `log_incoming` preview. A short prompt exercises
// the un-truncated preview branch; a > 800-char prompt exercises the truncation
// branch. Fail-on-revert: gate these behind the toggles (as production does) — a
// non-verbose run leaves both functions uncovered.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn verbose_and_incoming_logging_paths() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP verbose_and_incoming_logging_paths: tiny gguf not found");
        return;
    };
    higgs.set_verbose(true);
    higgs.set_log_incoming_tokens(true);
    load_tiny_ctx(&higgs, 1024).await;
    let (base, guard) = serve_v1_local(higgs.handle()).await;
    let c = reqwest::Client::new();
    let url = format!("{base}/v1/chat/completions");

    // Short prompt → un-truncated incoming preview + served line.
    let short: Value = c
        .post(&url)
        .json(&json!({
            "model": TINY_MODEL_ID, "stream": false, "max_tokens": 8,
            "messages": [{ "role": "user", "content": "hi there" }]
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        short["object"], "chat.completion",
        "short chat served: {short:?}"
    );

    // Long prompt (> 800 chars, but few tokens) → truncated incoming preview.
    let long_prompt = "hello ".repeat(150); // 900 chars
    let long: Value = c
        .post(&url)
        .json(&json!({
            "model": TINY_MODEL_ID, "stream": false, "max_tokens": 8,
            "messages": [{ "role": "user", "content": long_prompt }]
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        long["object"], "chat.completion",
        "long chat served: {long:?}"
    );
    assert!(
        long["choices"][0]["message"]["content"].is_string(),
        "long-prompt chat returned content: {long:?}"
    );

    guard.shutdown().await;
    higgs.shutdown().await;
}

// ── Host guard: IPv6 loopback variants + malformed bracket ────────────────────
//
// `is_loopback_host` handles bracketed IPv6 (`[::1]:port`), bare IPv6 (`::1`),
// and a malformed bracket (`[::1`). Loopback IPv6 passes the DNS-rebinding guard;
// the malformed bracket is rejected 403. The other files only cover IPv4 /
// localhost / a non-loopback DNS name.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn host_guard_ipv6_variants() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP host_guard_ipv6_variants: tiny gguf not found");
        return;
    };
    let (base, guard) = serve_v1_local(higgs.handle()).await;
    let c = reqwest::Client::new();

    // Bracketed + bare IPv6 loopback → allowed through to the handler (200).
    for host in ["[::1]:8080", "::1"] {
        let resp = c
            .get(format!("{base}/v1/models"))
            .header("host", host)
            .send()
            .await
            .unwrap();
        assert_eq!(
            resp.status(),
            200,
            "loopback IPv6 Host {host:?} passes the guard"
        );
        let _ = resp.text().await.unwrap();
    }

    // Malformed bracketed IPv6 (no closing `]`) → not loopback → 403 [HG012].
    let resp = c
        .get(format!("{base}/v1/models"))
        .header("host", "[::1")
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        403,
        "a malformed bracketed Host is rejected 403"
    );
    let body = resp.text().await.unwrap();
    assert!(body.contains("HG012"), "403 carries HG012: {body}");

    guard.shutdown().await;
    higgs.shutdown().await;
}

// ── CORS: reflected tauri origin, rejected https + non-local origins ───────────
//
// `is_local_origin` / `local_cors`: the tauri webview origin is reflected; an
// `https://` origin (scheme mismatch) and a non-loopback `http://` origin are
// NOT reflected (no allow-origin header). Covers the tauri arm, the scheme-strip
// reject, and the extra-origins predicate branch that the other files skip.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cors_origin_variants() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP cors_origin_variants: tiny gguf not found");
        return;
    };
    let (base, guard) = serve_v1_local(higgs.handle()).await;
    let c = reqwest::Client::new();
    let url = format!("{base}/v1/chat/completions");

    let allow_origin = |origin: &'static str| {
        let c = c.clone();
        let url = url.clone();
        async move {
            let resp = c
                .request(reqwest::Method::OPTIONS, &url)
                .header("host", "127.0.0.1")
                .header("origin", origin)
                .header("access-control-request-method", "POST")
                .send()
                .await
                .unwrap();
            let acao = resp
                .headers()
                .get("access-control-allow-origin")
                .and_then(|v| v.to_str().ok())
                .map(ToOwned::to_owned);
            let _ = resp.text().await.unwrap();
            acao
        }
    };

    // The tauri webview origin is a trusted local origin → reflected.
    assert_eq!(
        allow_origin("tauri://localhost").await.as_deref(),
        Some("tauri://localhost"),
        "the tauri webview origin is reflected"
    );
    // https scheme (not http://) → not a local origin → not reflected.
    assert_eq!(
        allow_origin("https://localhost").await,
        None,
        "an https origin is not a trusted local origin"
    );
    // A non-loopback http origin → not reflected (exercises the extra-origins
    // predicate, which is empty by default).
    assert_eq!(
        allow_origin("http://evil.example").await,
        None,
        "a non-loopback origin is not reflected"
    );

    guard.shutdown().await;
    higgs.shutdown().await;
}

// ── unknown routes / methods: 404 + 405, exercising required_scope arms ────────
//
// The auth middleware's `required_scope` runs on EVERY path before routing: an
// unknown `/v1/*` path resolves to the Admin scope then 404s at routing; a
// non-`/v1` path resolves to no scope (open) then 404s; a wrong method on a known
// route is a 405. Covers the `starts_with("/v1/")` + `None` scope arms and the
// router's 404/405 behavior.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unknown_routes_and_methods() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP unknown_routes_and_methods: tiny gguf not found");
        return;
    };
    let (base, guard) = serve_v1_local(higgs.handle()).await;
    let c = reqwest::Client::new();

    // Unknown /v1/* path → required_scope = Admin, then 404 (no such route).
    let resp = c
        .get(format!("{base}/v1/does-not-exist"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404, "unknown /v1/* path is a 404");
    let _ = resp.text().await.unwrap();

    // Unknown non-/v1 path → required_scope = None (open), then 404.
    let resp = c.get(format!("{base}/nope")).send().await.unwrap();
    assert_eq!(resp.status(), 404, "unknown non-/v1 path is a 404");
    let _ = resp.text().await.unwrap();

    // Wrong method on a known route (POST on GET-only /v1/models) → 405.
    let resp = c.post(format!("{base}/v1/models")).send().await.unwrap();
    assert_eq!(
        resp.status(),
        405,
        "POST on the GET-only /v1/models is a 405"
    );
    let _ = resp.text().await.unwrap();

    guard.shutdown().await;
    higgs.shutdown().await;
}

// ── the pub `v1_router` constructor builds (loopback host policy) ──────────────
//
// `serve_v1` builds its router through the policy-explicit constructor; the thin
// pub `v1_router` wrapper (host guard ON) is the one an out-of-crate embedder
// would call. Construct it to prove the public constructor builds without
// panicking (its served behavior is covered by every serve_v1_local test).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn v1_router_pub_constructor_builds() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP v1_router_pub_constructor_builds: tiny gguf not found");
        return;
    };
    // Building the router runs the pub wrapper (→ v1_router_with_host_policy(_, true)).
    let router = higgs::serve::v1_router(higgs.handle());
    // A Router has no cheap observable state; dropping it after construction is the
    // assertion that the public constructor is callable and does not panic.
    drop(router);
    higgs.shutdown().await;
}

// ── mint: bootstrap → admin default, second → chat+models, duplicate rejected ─
//
// The in-process (trusted) mint keeps every structural invariant. Covers the
// bootstrap Admin default, the later chat+models default, the duplicate-label
// rejection, and a successful revoke.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mint_scope_defaults_duplicate_and_revoke() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP mint_scope_defaults_duplicate_and_revoke: tiny gguf not found");
        return;
    };

    // Bootstrap mint (empty store), scopes omitted → defaults to [admin].
    let first = higgs.mint_key("admin-key", None).expect("bootstrap mint");
    assert!(
        first.scopes.contains(&Scope::Admin),
        "the bootstrap key defaults to admin: {:?}",
        first.scopes
    );
    assert!(first.token.starts_with("hgk_"), "token is a hgk_ token");

    // Non-bootstrap mint, scopes omitted → defaults to [chat, models].
    let second = higgs.mint_key("app-key", None).expect("second mint");
    assert!(
        second.scopes.contains(&Scope::Chat) && second.scopes.contains(&Scope::Models),
        "a later key defaults to chat+models: {:?}",
        second.scopes
    );
    assert!(
        !second.scopes.contains(&Scope::Admin),
        "the default non-bootstrap key is not admin: {:?}",
        second.scopes
    );

    // A duplicate label is rejected.
    let dup = higgs.mint_key("admin-key", None);
    assert!(
        matches!(&dup, Err(HiggsError::InvalidRequest { detail }) if detail.contains("already exists")),
        "duplicate label rejected: {dup:?}"
    );

    // Revoke the app key → removed 1, auth still enabled (admin remains).
    let removed = higgs.revoke_key("app-key").expect("revoke app key");
    assert_eq!(removed.removed, 1, "one key removed");
    assert!(removed.auth_enabled, "auth still on (admin key remains)");

    higgs.shutdown().await;
}

// ── mint: bootstrap key MUST be admin-capable ─────────────────────────────────
//
// A first key with explicit non-admin scopes would flip auth on yet be unable to
// reach the Admin-only key API — refused ([codex r10]).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mint_bootstrap_requires_admin() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP mint_bootstrap_requires_admin: tiny gguf not found");
        return;
    };
    let res = higgs.mint_key("chat-only", Some(vec![Scope::Chat]));
    assert!(
        matches!(&res, Err(HiggsError::InvalidRequest { detail }) if detail.contains("admin")),
        "bootstrap with non-admin scopes is refused: {res:?}"
    );
    higgs.shutdown().await;
}

// ── revoke: removing the last admin while other keys remain is refused [HG066] ─
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn revoke_last_admin_key_refused() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP revoke_last_admin_key_refused: tiny gguf not found");
        return;
    };
    higgs.mint_key("admin-key", None).expect("bootstrap admin");
    higgs
        .mint_key("chat-key", Some(vec![Scope::Chat]))
        .expect("second non-admin key");

    // Revoking the only admin while a non-admin key remains would lock out the
    // key-management surface.
    let res = higgs.revoke_key("admin-key");
    assert!(
        matches!(res, Err(HiggsError::LastAdminKey { .. })),
        "last-admin revoke is refused: {res:?}"
    );
    higgs.shutdown().await;
}

// ── revoke: emptying the keystore while LAN-exposed is refused [HG059] ─────────
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn revoke_last_key_on_lan_refused() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP revoke_last_key_on_lan_refused: tiny gguf not found");
        return;
    };
    higgs.mint_key("admin-key", None).expect("bootstrap admin");
    // Mark the server LAN-exposed (what serve_v1 records on a non-loopback bind).
    higgs.set_lan_exposed(true);

    // Revoking the LAST key while LAN-exposed would reopen the whole surface.
    let res = higgs.revoke_key("admin-key");
    assert!(
        matches!(res, Err(HiggsError::LastKeyOnLan { .. })),
        "last-key-on-LAN revoke is refused: {res:?}"
    );
    higgs.shutdown().await;
}

// ── mint: an unwritable keystore surfaces the coded persistence error [HG040] ─
//
// Occupy `api_keys.json` with a directory so the store's temp→rename flush can't
// write the file; the successful decision then fails to persist and the mint
// surfaces [HG040] rather than a silent success.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mint_key_persistence_failure_hg040() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP mint_key_persistence_failure_hg040: tiny gguf not found");
        return;
    };
    // Occupy api_keys.json with a directory → the flush rename can't create it.
    let ak = higgs.home().join("api_keys.json");
    let _ = std::fs::remove_file(&ak);
    std::fs::create_dir_all(&ak).expect("occupy api_keys.json with a dir");

    let res = higgs.mint_key("k", None);
    assert!(
        matches!(&res, Err(HiggsError::PersistenceFailed { .. })),
        "an unwritable keystore surfaces a persistence error: {res:?}"
    );
    assert!(
        res.unwrap_err().to_string().contains("[HG040]"),
        "the persistence error carries the HG040 code"
    );
    higgs.shutdown().await;
}

// ── serve_v1 refuses a non-loopback bind whose keys are all non-Admin [HG069] ─
//
// A LAN bind with keys but none Admin-capable would lock out the key-management
// API — refused before it can serve, tearing any worker down. (The keyless-LAN
// [HG058] twin is covered in serve_guard.rs.)
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn serve_v1_refuses_lan_without_admin_key() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP serve_v1_refuses_lan_without_admin_key: tiny gguf not found");
        return;
    };
    // Install a live keystore with a single NON-admin key (mint_key can't build
    // this — the first key must be admin — so seed the live store directly).
    let token = higgs::keys::mint_token([9u8; 16]);
    let mut ks = higgs::keys::ApiKeys::default();
    ks.add(&token, "chat-only".into(), vec![Scope::Chat]);
    higgs.set_api_keys(Arc::new(ks));

    let listener = tokio::net::TcpListener::bind("0.0.0.0:0")
        .await
        .expect("bind 0.0.0.0");
    let err = higgs::serve::serve_v1(higgs.handle(), listener, async {})
        .await
        .expect_err("a non-Admin-only LAN bind must be refused");
    assert!(
        err.to_string().contains("[HG069]"),
        "the refusal carries the HG069 code: {err}"
    );
    higgs.shutdown().await;
}

// ── keyed LAN: a non-loopback bind relaxes the Host guard + enforces auth ──────
//
// With an Admin key present, serve_v1 accepts a 0.0.0.0 bind and serves with the
// DNS-rebinding Host guard RELAXED (LAN clients send their own Host) but auth ON:
// no/other bearer → 401, the Admin bearer → 200. Covers the relaxed-host branch
// and the auth bearer accept/reject on a keyed bind.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn keyed_lan_bind_relaxes_host_guard_and_enforces_auth() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP keyed_lan_bind_relaxes_host_guard_and_enforces_auth: tiny gguf not found");
        return;
    };
    let admin = higgs
        .mint_key("lan-admin", Some(vec![Scope::Admin]))
        .expect("mint admin key");

    let (base, tx, join) = serve_v1_on(higgs.handle(), "0.0.0.0:0").await;
    let c = reqwest::Client::new();

    // No bearer → 401 (auth is on because the keystore is non-empty).
    let resp = c.get(format!("{base}/v1/models")).send().await.unwrap();
    assert_eq!(resp.status(), 401, "no bearer → 401 on a keyed bind");
    let _ = resp.text().await.unwrap();

    // A wrong bearer → 401.
    let resp = c
        .get(format!("{base}/v1/models"))
        .bearer_auth("hgk_deadbeef")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 401, "a bad bearer → 401");
    let _ = resp.text().await.unwrap();

    // The Admin bearer → 200 (Admin satisfies the Models scope), proving the LAN
    // Host was accepted (relaxed guard) AND the bearer authorized.
    let resp = c
        .get(format!("{base}/v1/models"))
        .bearer_auth(&admin.token)
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        200,
        "the admin bearer → 200 on the keyed LAN bind"
    );
    let listed: Value = resp.json().await.unwrap();
    assert_eq!(listed["object"], "list", "the models list is returned");

    // Graceful shutdown: signal, join (serve_v1 stops the worker), then belt-and-
    // braces facade stop.
    let _ = tx.send(());
    let _ = join.await;
    higgs.shutdown().await;
}

// ── readiness derivation via model_entries: Servable / Profiled / NeedsRetune ─
//
// `model_entries` derives per-model readiness. A prepared tiny model is Servable
// (serving on, fits free resources — the unified-memory fit path on Apple);
// serving OFF demotes it to Profiled; a mutated GGUF (stale profile) is
// NeedsRetune. Covers the stale / serving-off / fit branches of the pure
// derivation.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn readiness_states_via_model_entries() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP readiness_states_via_model_entries: tiny gguf not found");
        return;
    };
    // Prepare → profiled, fits, serving on → Servable (also computes the fit).
    tune_tiny(&higgs).await;
    let entry = tiny_entry(&higgs).await;
    assert_eq!(
        entry.readiness,
        ModelReadiness::Servable,
        "a prepared, fitting, serving model is Servable"
    );
    assert!(
        entry.fit.is_some(),
        "the servable branch surfaces the fit detail"
    );

    // Serving OFF → the same profiled model is Profiled (cannot serve now).
    higgs.set_serving_enabled(false);
    let entry = tiny_entry(&higgs).await;
    assert_eq!(
        entry.readiness,
        ModelReadiness::Profiled,
        "serving-off demotes a profiled model to Profiled"
    );
    higgs.set_serving_enabled(true);

    // Mutate the GGUF so the saved profile's file signature no longer matches →
    // NeedsRetune (hard-blocks load until re-tuned).
    {
        use std::io::Write as _;
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(higgs.staged_gguf(TINY_MODEL_ID))
            .expect("open staged gguf");
        f.write_all(b"\0").expect("append to staged gguf");
    }
    let entry = tiny_entry(&higgs).await;
    assert_eq!(
        entry.readiness,
        ModelReadiness::NeedsRetune,
        "a stale profile is NeedsRetune"
    );

    higgs.shutdown().await;
}

// ── an explicit load grandfathers the lone active record as the "Tuned" profile ─
//
// An explicit-params load (no prior tune) persists only an ACTIVE (Heuristic)
// record — both dual-profile history slots empty. `TuneProfileViews::from_triple`
// grandfathers that lone record into the analytical (`tuned_load`) slot so the
// pre-dual-store row still shows a tuned profile.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn explicit_load_grandfathers_active_as_tuned() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP explicit_load_grandfathers_active_as_tuned: tiny gguf not found");
        return;
    };
    // An explicit-params load (from_request) persists the active profile only —
    // set_profile writes the active record, put_tuning is never called, so both
    // history slots stay empty (a pre-dual store).
    load_tiny_ctx(&higgs, 512).await;
    let entry = tiny_entry(&higgs).await;
    assert!(
        entry.tuned_load.is_some(),
        "the lone active record is grandfathered into the tuned slot"
    );
    assert!(
        entry.tune_provenance.is_some(),
        "the active record's provenance labels the row"
    );

    higgs.shutdown().await;
}

/// The tiny model's enriched entry from `model_entries`.
async fn tiny_entry(h: &Higgs) -> higgs::HiggsModelEntry {
    h.model_entries()
        .await
        .expect("model_entries")
        .into_iter()
        .find(|e| e.model.id == TINY_MODEL_ID)
        .expect("tiny model entry present")
}
