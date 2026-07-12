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
    assert!(higgs.node_load("n", "m", None).await.is_err());
    assert!(higgs.node_unload("m").await.is_err());
    assert!(higgs.node_retire("n").await.is_err());
    assert!(higgs.node_scan("n").await.is_err());
    assert!(higgs.node_chat_test("n", None, None).await.is_err());
    // The "local" sentinel is refused as HG076 EVEN WITH THE HUB OFF (the arm
    // precedes the not-a-hub gate, like node_label's) — never the "enable the
    // hub first" runaround that would refuse again after being followed.
    assert!(
        matches!(
            higgs.node_chat_test("local", None, None).await,
            Err(HiggsError::InvalidChatTestTarget { .. })
        ),
        "hub-off local → HG076 chat-directly, not not-a-hub"
    );
    // hub_disable is a no-op returning a disabled status when no hub is installed.
    let status = higgs.hub_disable().await;
    assert!(!status.enabled);
}

/// The FACADE params path: `node_load(.., Some(p))` dispatches over a major-2
/// admission and an id-force is load-bearing — `p.id` deliberately names a
/// model the node does NOT have; only a forced `id = model` makes the load
/// resolve. This test fails only when BOTH forces (facade + fleet) are dropped
/// — the FLEET force in isolation is pinned by cov_fleet2's divergent-id mock
/// case, which bypasses the facade.
#[tokio::test]
async fn node_load_params_forces_the_wire_id_from_model() {
    let (higgs, node_key, model_id, _guards) = fake_higgs_with_remote_node().await;
    let params = crate::remote::NodeLoadParams {
        id: "divergent/id-the-node-lacks".into(),
        ctx_len: Some(256),
        gpu_layers: None,
        threads: None,
        params: None,
    };
    let worker = higgs
        .node_load(&node_key, &model_id, Some(params))
        .await
        .expect("params-load dispatches with the forced id");
    assert!(worker.0 >= 1);
}

// ── node_chat_test: the Fleet view's per-node iroh-link proof ────────────────

/// Spawn ONE in-process fake remote node serving `model_id` and register it on
/// `fleet` (real iroh via `test_support::local_endpoint`, fake worker halves —
/// the PLAIN fake, whose chat reply is "hello" with NO token counts). Returns
/// the node's endpoint key; the staged scan root rides in `roots`.
async fn add_fake_remote_node(
    fleet: &Arc<crate::node::fleet::HubFleet>,
    model_id: &str,
    roots: &mut Vec<tempfile::TempDir>,
    endpoints: &mut Vec<iroh::Endpoint>,
) -> String {
    use crate::node::test_support::{fake_runtime, local_endpoint, stage_dummy_model};

    let (root, _) = stage_dummy_model(model_id);
    let hub = local_endpoint().await;
    let node = local_endpoint().await;
    let hub_addr = hub.addr();
    let node_key = node.id().to_string();

    let rt = Arc::new(fake_runtime(vec![root.path().to_path_buf()]));
    tokio::spawn(async move {
        let node_conn = node
            .connect(hub_addr, crate::remote::ALPN)
            .await
            .expect("connect");
        crate::node::serve_node(node_conn, rt).await;
    });
    let conn = hub.accept().await.expect("incoming").await.expect("conn");
    // Keep the hub endpoint ALIVE via the caller's guard (dropping it would
    // close the accepted connection) — bounded by the test's scope instead of
    // the old `mem::forget` leak-per-node-per-test.
    endpoints.push(hub);

    fleet
        .add_node(
            node_key.clone(),
            Arc::new(crate::node::transport::NodeTransport::new(conn)),
            None,
            // Current-build semantics: the fake node "negotiated" major 2, so
            // facade-level params-loads dispatch (the floor-1 refusal arm is
            // pinned in fleet_tests over an explicitly version-less admit).
            Some(2),
            Some(env!("CARGO_PKG_VERSION").to_string()),
            false,
        )
        .await;
    fleet
        .load(&node_key, model_id, None)
        .await
        .expect("remote load");
    roots.push(root);
    node_key
}

/// A `Higgs` whose fleet routes one fake-worker remote node (nothing scanned
/// locally), plus the node's key and its loaded model's served id — the same
/// seam `fleet_tests::fleet_with_one_node` uses, lifted onto the facade so
/// `node_chat_test`'s arms are drivable without llama.cpp.
async fn fake_higgs_with_remote_node() -> (Higgs, String, String, RemoteNodeGuards) {
    let fleet = Arc::new(crate::node::fleet::HubFleet::new(Arc::new(
        crate::log_bus::LogBus::new(),
    )));
    let mut guards = RemoteNodeGuards::default();
    let model_id = "higgs-test/m".to_string();
    let node_key =
        add_fake_remote_node(&fleet, &model_id, &mut guards.roots, &mut guards.endpoints).await;

    let higgs = fake_higgs(vec![]);
    higgs.set_fleet(fleet);
    (higgs, node_key, model_id, guards)
}

/// Keeps the fake-remote-node scaffolding alive for a test's duration: the
/// staged scan roots AND the hub-side iroh endpoints (dropping an endpoint
/// closes its accepted connection).
#[derive(Default)]
struct RemoteNodeGuards {
    roots: Vec<tempfile::TempDir>,
    endpoints: Vec<iroh::Endpoint>,
}

/// Happy path: the test prompt relays over the (in-process) iroh link to the
/// node's routed instance and the report carries the worker's actual reply.
/// (The always-remote property itself is pinned by
/// `node_chat_test_bypasses_the_local_first_dispatch` below, which sets up a
/// distinguishable LOCAL twin — this seam has no local model, so a local-first
/// revert would relay remotely here too and still pass.)
#[tokio::test]
async fn node_chat_test_relays_to_the_nodes_instance() {
    let (higgs, node_key, model_id, _guards) = fake_higgs_with_remote_node().await;

    // Implicit instance pick (served = None → the node's first served id).
    let report = higgs
        .node_chat_test(&node_key, None, None)
        .await
        .expect("chat test");
    assert_eq!(report.endpoint_id, node_key);
    assert_eq!(report.served_id, model_id);
    assert_eq!(report.content, "hello", "the fake remote worker's reply");
    assert_eq!(report.finish_reason, "stop");

    // Explicit instance pick lands on the same worker.
    let explicit = higgs
        .node_chat_test(&node_key, Some(&model_id), Some("ping?"))
        .await
        .expect("explicit chat test");
    assert_eq!(explicit.served_id, model_id);
    assert_eq!(explicit.content, "hello");
}

/// THE always-remote pin: the same id resident BOTH locally and on the remote
/// node, with distinguishable answers — the LOCAL stateful fake reports
/// `prompt_tokens: 10`, the REMOTE plain fake omits token counts (→ 0). The
/// generic dispatch (`chat_stream`) resolves LOCAL-first, so a revert of
/// `node_chat_test` onto it answers locally with `prompt_tokens == 10` and this
/// test fails; only the fleet relay produces the remote shape. (Fail-on-revert
/// verified by actually rerouting the dispatch and watching this fail.)
#[tokio::test]
async fn node_chat_test_bypasses_the_local_first_dispatch() {
    let fleet = Arc::new(crate::node::fleet::HubFleet::new(Arc::new(
        crate::log_bus::LogBus::new(),
    )));
    let mut guards = RemoteNodeGuards::default();
    let model_id = "higgs-test/m".to_string();
    let node_key =
        add_fake_remote_node(&fleet, &model_id, &mut guards.roots, &mut guards.endpoints).await;

    // The LOCAL engine scans the same staged root, so the SAME id is locally
    // loadable; load it on the local stateful fake engine (the "local twin").
    let higgs = fake_higgs(vec![guards.roots[0].path().to_path_buf()]);
    higgs.set_fleet(fleet);
    higgs.load(&model_id, None).await.expect("local twin loads");

    // SELF-GUARD for the distinguisher: prove the local twin really answers
    // with prompt_tokens == 10 through the generic dispatch. If a fake refactor
    // ever makes the two shapes identical, this fails loudly instead of letting
    // the probe assertion below pass against a local-first dispatch.
    let (rx, handle) = higgs
        .chat_stream(
            model_id.clone(),
            json!([{"role": "user", "content": "hi"}]).to_string(),
            8,
            crate::worker::engine::SamplingParams::default(),
            None,
            None,
        )
        .await
        .expect("generic dispatch reaches the local twin");
    drop(rx);
    let local_shape = handle.await.expect("join").expect("local twin outcome");
    assert_eq!(
        local_shape.prompt_tokens, 10,
        "the local twin's shape must stay distinguishable from the remote fake's \
         (prompt_tokens 10 vs absent→0), or this test can no longer detect a \
         local-first dispatch: {local_shape:?}"
    );

    let report = higgs
        .node_chat_test(&node_key, None, None)
        .await
        .expect("chat test with a local twin resident");
    assert_eq!(report.content, "hello");
    assert_eq!(
        report.prompt_tokens, 0,
        "the REMOTE plain fake omits token counts; the local twin would report 10 — \
         a local-first dispatch answered this: {report:?}"
    );
}

/// Two remote nodes serving the SAME model split it into `m` / `m-1` served ids.
/// Each node's implicit test must dispatch to ITS OWN instance, and pinning one
/// node's served id to the other node is [HG076] — exercised with REAL suffixed
/// routes, not stand-ins.
#[tokio::test]
async fn node_chat_test_pins_across_two_nodes_sharing_a_model() {
    let fleet = Arc::new(crate::node::fleet::HubFleet::new(Arc::new(
        crate::log_bus::LogBus::new(),
    )));
    let mut guards = RemoteNodeGuards::default();
    let node_a = add_fake_remote_node(
        &fleet,
        "higgs-test/m",
        &mut guards.roots,
        &mut guards.endpoints,
    )
    .await;
    let node_b = add_fake_remote_node(
        &fleet,
        "higgs-test/m",
        &mut guards.roots,
        &mut guards.endpoints,
    )
    .await;
    let higgs = fake_higgs(vec![]);
    higgs.set_fleet(fleet.clone());

    // The shared model splits into base + "-1" across the two nodes (assignment
    // order rides the endpoint-key sort — derive it, don't assume it).
    let served_a = fleet.served_on(&node_a).await;
    let served_b = fleet.served_on(&node_b).await;
    assert_eq!(served_a.len(), 1, "one instance on node A");
    assert_eq!(served_b.len(), 1, "one instance on node B");
    assert_ne!(served_a[0], served_b[0], "distinct served ids");
    let mut both = vec![served_a[0].clone(), served_b[0].clone()];
    both.sort();
    assert_eq!(both, vec!["higgs-test/m", "higgs-test/m-1"]);

    // Implicit pick tests each node's OWN instance.
    for (node, served) in [(&node_a, &served_a[0]), (&node_b, &served_b[0])] {
        let report = higgs.node_chat_test(node, None, None).await.expect("test");
        assert_eq!(&report.endpoint_id, node);
        assert_eq!(&report.served_id, served);
        assert_eq!(report.content, "hello");
    }

    // Cross-pin: node A's served id on node B → HG076 naming both nodes, from
    // the facade's PRE-CHECK (its wording carries the caller-mistake remedy
    // "omit `served`" — the dispatch-time [HG077] backstop's display is "not at
    // the pinned node at dispatch" with no such remedy; asserting the wording
    // pins the pre-check itself).
    let cross = higgs
        .node_chat_test(&node_b, Some(&served_a[0]), None)
        .await;
    match cross {
        Err(HiggsError::InvalidChatTestTarget { ref detail }) => {
            assert!(detail.contains(&node_a), "names the routed node: {detail}");
            assert!(
                detail.contains(&node_b),
                "names the requested node: {detail}"
            );
            assert!(
                detail.contains("omit `served`"),
                "the PRE-CHECK's caller-mistake remedy, not the HG077 race wording: {detail}"
            );
        }
        other => panic!("expected HG076 for a cross-node pin, got {other:?}"),
    }
}

/// After the hub kill switch (`disconnect_all` — the fleet and its durable
/// routes survive, the transports drop), a chat test fails with [HG027] whose
/// advice is enriched to say the node CANNOT reconnect until the hub network
/// is re-enabled — the plain "recovers once it reconnects" would be a
/// follow-it-and-wait-forever runaround. (The seam has no `Hub` installed, so
/// `hub().is_none()` — the same state a production kill switch leaves.)
#[tokio::test]
async fn node_chat_test_after_kill_switch_names_the_real_remedy() {
    let (higgs, node_key, _model_id, _guards) = fake_higgs_with_remote_node().await;
    higgs
        .fleet()
        .expect("fleet installed")
        .disconnect_all()
        .await;

    let err = higgs.node_chat_test(&node_key, None, None).await;
    match err {
        Err(HiggsError::NodeUnreachable { ref detail, .. }) => {
            assert!(
                detail.contains("hub network is currently disabled")
                    && detail.contains("hub_enable"),
                "the enriched advice names the real remedy: {detail}"
            );
        }
        other => panic!("expected enriched HG027, got {other:?}"),
    }
}

/// The refusal ladder, most-specific first: a never-paired endpoint id is
/// [HG075] (no "load first" advice for a nonexistent node); a KNOWN node with
/// nothing routed is [HG074]; an explicit served id that is unrouted, or routed
/// on a DIFFERENT node, is [HG076] (the report would claim a link the test
/// never exercised).
#[tokio::test]
async fn node_chat_test_refusal_arms() {
    let (higgs, node_key, model_id, _guards) = fake_higgs_with_remote_node().await;
    let fleet = higgs.fleet().expect("fleet installed");

    // The LOCAL machine's sentinel id → HG076 "chat it directly", never the
    // HG075 "pair the node first" advice it would otherwise fall through to.
    let local = higgs.node_chat_test("local", None, None).await;
    match local {
        Err(HiggsError::InvalidChatTestTarget { ref detail }) => {
            assert!(
                detail.contains("this machine") && detail.contains("directly"),
                "the local arm's remedy is a direct chat, not pairing: {detail}"
            );
            assert!(
                detail.contains(" — "),
                "the local arm's detail carries its own remediation: {detail}"
            );
        }
        other => panic!("expected HG076 for the local sentinel, got {other:?}"),
    }

    // A node the hub has NEVER paired → HG075 unknown-node, not load-first advice.
    let unknown = higgs.node_chat_test("unknown-node", None, None).await;
    assert!(
        matches!(unknown, Err(HiggsError::UnknownNode { ref endpoint_id }) if endpoint_id == "unknown-node"),
        "expected HG075, got {unknown:?}"
    );
    // …even when it names a REAL served id (the node check is most-specific-first).
    let unknown_with_served = higgs
        .node_chat_test("unknown-node", Some(&model_id), None)
        .await;
    assert!(
        matches!(unknown_with_served, Err(HiggsError::UnknownNode { .. })),
        "expected HG075, got {unknown_with_served:?}"
    );

    // A KNOWN (seeded) node with an empty route table → HG074 (load first).
    fleet.seed_node("bare-node").await;
    let bare = higgs.node_chat_test("bare-node", None, None).await;
    assert!(
        matches!(bare, Err(HiggsError::NodeNothingServed { ref endpoint_id }) if endpoint_id == "bare-node"),
        "expected HG074, got {bare:?}"
    );

    // An unrouted served id on a known node → HG076 naming the operand, not a
    // "not found on disk" (no disk was consulted).
    let unrouted = higgs
        .node_chat_test(&node_key, Some("nope/none"), None)
        .await;
    match unrouted {
        Err(HiggsError::InvalidChatTestTarget { ref detail }) => {
            assert!(detail.contains("nope/none"), "names the operand: {detail}");
            assert!(
                detail.contains("not routed"),
                "says what was actually checked: {detail}"
            );
            // HG076's display is detail-only (no fixed remediation tail), so the
            // ladder-codes test exempts it from the em-dash rule ON THE PROMISE
            // that every producing arm's detail carries its own remedy — this
            // pin is that promise for the unrouted arm (the mismatch arm's
            // "omit `served`" pins are below).
            assert!(
                detail.contains(" — ") && detail.contains("refresh the fleet view"),
                "the unrouted arm's detail must carry its own remediation: {detail}"
            );
        }
        other => panic!("expected HG076 for an unrouted served id, got {other:?}"),
    }

    // A REAL served id but the WRONG (known) node → HG076 naming both nodes,
    // from the facade's PRE-CHECK: the "omit `served`" remedy is unique to it —
    // the dispatch-time [HG077] backstop catches the same mismatch but its
    // "re-resolve against the fleet view" advice is wrong for a caller who
    // simply named the wrong node. Fail-on-revert for the pre-check itself.
    let mismatch = higgs
        .node_chat_test("bare-node", Some(&model_id), None)
        .await;
    match mismatch {
        Err(HiggsError::InvalidChatTestTarget { ref detail }) => {
            assert!(
                detail.contains(&node_key),
                "names the routed node: {detail}"
            );
            assert!(
                detail.contains("bare-node"),
                "names the requested node: {detail}"
            );
            assert!(
                detail.contains("omit `served`"),
                "the PRE-CHECK's caller-mistake remedy, not the HG077 race wording: {detail}"
            );
        }
        other => panic!("expected HG076 for a node mismatch, got {other:?}"),
    }
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

// ── CORS origins allowlist (validate + persist + applied/restart state) ──────

#[test]
fn validate_cors_origin_accepts_and_canonicalizes() {
    // (input, expected canonical form a browser sends in the Origin header)
    for (input, expected) in [
        ("http://tools.example", "http://tools.example"),
        ("https://tools.example", "https://tools.example"),
        ("http://localhost:5173", "http://localhost:5173"),
        (
            "https://app.example.com:8443",
            "https://app.example.com:8443",
        ),
        ("http://127.0.0.1:31415", "http://127.0.0.1:31415"),
        ("http://[::1]", "http://[::1]"),
        ("http://[::1]:8080", "http://[::1]:8080"),
        // Bracketed IPv6 with a port is accepted verbatim.
        ("https://[::1]:8080", "https://[::1]:8080"),
        // Browsers lowercase the host — canonicalize to match.
        ("https://EXAMPLE.com", "https://example.com"),
        // Browsers omit the default port — strip 80 (http) / 443 (https).
        ("http://example.com:80", "http://example.com"),
        ("https://example.com:443", "https://example.com"),
        // A lone trailing slash is the empty path — normalized away, not rejected.
        ("https://tools.example/", "https://tools.example"),
    ] {
        let canonical = super::validate_cors_origin(input)
            .unwrap_or_else(|e| panic!("expected {input:?} to validate, got {e:?}"));
        assert_eq!(
            canonical, expected,
            "expected {input:?} to canonicalize to {expected:?}"
        );
    }
}

#[test]
fn validate_cors_origin_rejects_malformed() {
    for bad in [
        "",                             // empty
        "tools.example",                // no scheme
        "ftp://tools.example",          // wrong scheme
        "https://tools.example/app",    // path
        "https://tools.example?x=1",    // query
        "https://tools.example#top",    // fragment
        "http://user:pass@example.com", // userinfo (username + password)
        "http://user@example.com",      // userinfo (username only)
        "http://",                      // missing host
        "http://:8080",                 // missing host, has port
        "http://host:abc",              // non-numeric port
        "http://::1",                   // unbracketed IPv6 — never a browser Origin
        "http://[::1",                  // unbalanced bracket
        "http://[::1]:8080:9090",       // malformed / double port
        "https://example.com:65536",    // port out of range (> 65535)
        // Dot-segment / escaped / backslash paths NORMALIZE to "/" inside the
        // parser, so only the raw-input check catches them — a pasted URL with a
        // visible path must not silently become its origin.
        "https://tools.example/app/..", // dot-segments collapse to "/"
        "https://tools.example/%2e%2e", // escaped dot-segments collapse to "/"
        "https://tools.example\\app",   // backslash is a path separator to the parser
        "https:tools.example",          // scheme-relative shorthand — no "://"
        // A BARE delimiter: `url` reports `query`/`fragment` as `Some("")`, which
        // is still not a bare origin (JS `URL` reads these back as `''`, so the
        // frontend mirror needs its own raw scan to agree with this).
        "https://tools.example?", // empty query delimiter
        "https://tools.example#", // empty fragment delimiter
    ] {
        assert!(
            matches!(
                super::validate_cors_origin(bad),
                Err(HiggsError::InvalidCorsOrigin { .. })
            ),
            "expected {bad:?} to be rejected as HG071"
        );
    }
}

#[test]
fn validate_and_dedup_preserves_first_seen_order() {
    let deduped = super::validate_and_dedup_cors_origins(vec![
        "https://a.example".to_string(),
        "https://b.example".to_string(),
        "https://a.example".to_string(),
    ])
    .expect("all valid");
    assert_eq!(
        deduped,
        vec![
            "https://a.example".to_string(),
            "https://b.example".to_string()
        ],
        "repeated origin dropped, first-seen order kept"
    );
}

#[test]
fn validate_and_dedup_collapses_canonically_equal_origins() {
    // Two inputs a browser serializes identically collapse to one canonical entry.
    let deduped = super::validate_and_dedup_cors_origins(vec![
        "https://EXAMPLE.com".to_string(),
        "https://example.com:443".to_string(),
        "http://b.example:80".to_string(),
    ])
    .expect("all valid");
    assert_eq!(
        deduped,
        vec![
            "https://example.com".to_string(),
            "http://b.example".to_string()
        ],
        "canonically-equal origins deduped; each stored in canonical form"
    );
}

#[tokio::test]
async fn cors_settings_persists_and_flags_restart_required() {
    let higgs = fake_higgs(vec![]);

    // Fresh instance: nothing persisted, nothing applied → no restart pending.
    let initial = higgs.cors_settings();
    assert!(initial.origins.is_empty());
    assert!(initial.applied_origins.is_empty());
    assert!(!initial.restart_required);

    // Persist a (deduped) list. No CORS layer has been built yet (no serve), so
    // there is nothing live to diverge from — the first serve start will apply
    // this list — hence NO restart is pending (applied stays `None`, not empty).
    let updated = higgs
        .set_cors_origins(vec![
            "https://tools.example".to_string(),
            "http://localhost:5173".to_string(),
            "https://tools.example".to_string(), // dup dropped
        ])
        .expect("valid origins persist");
    assert_eq!(
        updated.origins,
        vec![
            "https://tools.example".to_string(),
            "http://localhost:5173".to_string()
        ]
    );
    assert!(updated.applied_origins.is_empty());
    assert!(
        !updated.restart_required,
        "pre-serve set: no CORS layer built yet → no restart pending"
    );

    // A fresh read reflects the persisted config.json.
    let read_back = higgs.cors_settings();
    assert_eq!(read_back.origins, updated.origins);

    // With a listener live, the disclosures compare the persisted file against
    // the LIVE list (G7: layers read the shared live list per request). A
    // divergence can only come from a HAND-EDITED config.json — an API write
    // publishes live as it persists. Simulate the hand-edit with a direct
    // config write that bypasses the publish.
    let higgs = Arc::new(higgs);
    let _live = higgs.register_serve(false, None);
    higgs.publish_live_cors(vec!["https://old.example".to_string()]);
    assert!(
        higgs.cors_settings().restart_required,
        "persisted list differs from the LIVE list → restart required (hand-edit case)"
    );

    // An API write reconciles: it persists AND publishes, so nothing is pending
    // and the applied disclosure equals the persisted list.
    let after_write = higgs
        .set_cors_origins(read_back.origins.clone())
        .expect("valid origins persist");
    assert!(
        !after_write.restart_required,
        "an API write applies LIVE (G7) → no restart pending"
    );
    assert_eq!(
        after_write.applied_origins, read_back.origins,
        "the live disclosure equals what was just written"
    );
}

#[tokio::test]
async fn cors_settings_pre_serve_never_flags_restart() {
    // No listener is live (no CORS layer exists), so however many origins are
    // persisted before the first serve, `restart_required` stays false — the first
    // serve start applies them, so nothing is pending.
    let higgs = fake_higgs(vec![]);
    higgs
        .set_cors_origins(vec!["https://tools.example".to_string()])
        .expect("valid origin persists");
    let settings = higgs.cors_settings();
    assert_eq!(settings.origins, vec!["https://tools.example".to_string()]);
    assert!(
        settings.applied_origins.is_empty(),
        "nothing applied pre-serve"
    );
    assert!(
        !settings.restart_required,
        "pre-serve set must not report a restart"
    );
}

#[tokio::test]
async fn cors_restart_flag_ignores_origin_order() {
    // The allowlist is exact-match MEMBERSHIP — order is meaningless to the
    // running CORS layer, so a hand-edited config.json holding the SAME
    // origins reordered must not claim a restart. A genuinely different
    // hand-edited set still must. (API writes publish live and can never
    // diverge — the divergence is simulated with a direct config write.)
    let higgs = Arc::new(fake_higgs(vec![]));
    let _live = higgs.register_serve(false, None);
    higgs.publish_live_cors(vec![
        "https://a.example".to_string(),
        "https://b.example".to_string(),
    ]);

    higgs
        .with_config_mut(|c| {
            c.cors_origins = vec![
                "https://b.example".to_string(),
                "https://a.example".to_string(),
            ]
        })
        .expect("direct config write");
    assert!(
        !higgs.cors_settings().restart_required,
        "same origins, different order → everything is already live"
    );

    higgs
        .with_config_mut(|c| c.cors_origins = vec!["https://b.example".to_string()])
        .expect("direct config write");
    assert!(
        higgs.cors_settings().restart_required,
        "a genuinely different hand-edited set requires a restart (or any API write)"
    );

    // The API write reconciles the divergence live.
    let reconciled = higgs
        .set_cors_origins(vec!["https://b.example".to_string()])
        .expect("valid origin persists");
    assert!(
        !reconciled.restart_required,
        "an API write publishes live → divergence gone"
    );
}

#[tokio::test]
async fn set_cors_origins_canonicalizes_persisted_value() {
    // A mixed-case host with an explicit default port is stored as the exact
    // string a browser sends in the Origin header.
    let higgs = fake_higgs(vec![]);
    let settings = higgs
        .set_cors_origins(vec!["https://EXAMPLE.com:443".to_string()])
        .expect("valid origin persists");
    assert_eq!(
        settings.origins,
        vec!["https://example.com".to_string()],
        "host lowercased + default port stripped on persist"
    );
}

#[tokio::test]
async fn set_cors_origins_rejects_invalid_without_persisting() {
    let higgs = fake_higgs(vec![]);
    let err = higgs
        .set_cors_origins(vec![
            "https://ok.example".to_string(),
            "notaurl".to_string(),
        ])
        .expect_err("invalid entry rejected");
    assert!(
        matches!(err, HiggsError::InvalidCorsOrigin { .. }),
        "HG071 InvalidCorsOrigin, got {err:?}"
    );
    // Nothing was persisted — the config.json still lists no extra origins.
    assert!(
        higgs.cors_settings().origins.is_empty(),
        "rejected write leaves the allowlist untouched"
    );
}

/// `serve_v1` is public and an embedder may run SEVERAL listeners on one facade,
/// so LAN exposure comes from the live-listener REGISTRY, not a single bool: a
/// loopback serve starting — or any sibling serve exiting — must not clear the
/// exposure a still-live LAN listener depends on for its [HG059] last-key-revoke
/// refusal. The exposure lifts only when the LAST LAN listener leaves.
#[tokio::test]
async fn lan_exposure_tracks_every_live_serve() {
    let higgs = Arc::new(fake_higgs(vec![]));
    assert!(!higgs.lan_exposed(), "no listener → not exposed");

    // A LAN listener goes live, then a LOOPBACK one joins it. The loopback serve
    // must NOT clear the LAN exposure (the old bool `set_lan_exposed(!loopback)`
    // wrote `false` here and dropped the guard out from under a live LAN listener).
    let lan = higgs.register_serve(true, None);
    let loopback = higgs.register_serve(false, None);
    assert!(higgs.lan_exposed(), "LAN listener still live");

    // The loopback listener exits: a SIBLING serve ending must not lift exposure.
    drop(loopback);
    assert!(
        higgs.lan_exposed(),
        "sibling exit must not lift LAN exposure"
    );

    // The LAN listener exits — nothing is exposed any more.
    drop(lan);
    assert!(!higgs.lan_exposed(), "last LAN listener gone → not exposed");

    // Two overlapping LAN listeners: exposure survives until BOTH are gone.
    let a = higgs.register_serve(true, None);
    let b = higgs.register_serve(true, None);
    drop(a);
    assert!(higgs.lan_exposed(), "one LAN listener remains");
    drop(b);
    assert!(!higgs.lan_exposed(), "both gone → not exposed");

    // The manual override ORs in — it can ADD exposure for an embedder serving
    // through its own stack, but can never mask a live LAN listener.
    higgs.set_lan_exposed(true);
    assert!(higgs.lan_exposed(), "override exposes");
    higgs.set_lan_exposed(false);
    assert!(!higgs.lan_exposed(), "override lifts");
    let live_lan = higgs.register_serve(true, None);
    higgs.set_lan_exposed(false);
    assert!(
        higgs.lan_exposed(),
        "clearing the override must not mask a LIVE LAN listener"
    );
    drop(live_lan);
}

/// `arm_lan_serve` is the non-loopback startup gate: it runs the [HG058] (needs a
/// key) / [HG069] (needs an Admin key) checks and arms `lan_exposed`, both under the
/// SAME `keys_io` lock a revoke commits under, so a revoke and a LAN serve start can
/// never interleave into a keyless LAN surface — one of them always loses.
///
/// SCOPE OF THIS TEST — read before trusting it. It pins the OBSERVABLE gate
/// behavior (a refused serve arms nothing; a passing serve arms, and thereby makes
/// the store-emptying revoke fail with [HG059]). It does NOT pin the lock SCOPE:
/// moving `guard.set_lan()` outside the `keys_io` critical section still passes
/// here, because the interleaving that opens is a nanosecond race between two
/// threads. That property is argued from the code — `Higgs::arm_lan_serve` holds
/// `keys_io` across check-and-arm, and `revoke_key` reads `lan_exposed()` INSIDE
/// `mutate_api_keys`, which holds the same lock — plus the documented lock order
/// `keys_io` → `serves`. Pinning it in a test would need a test-only injection
/// point in production code, which this crate forbids.
///
/// ([HG069]'s own refusal is covered end-to-end by
/// `tests/cov_serve.rs::serve_v1_refuses_lan_without_admin_key`; a keys-but-no-Admin
/// store isn't reachable through `mint_key`, which enforces the
/// first-key-must-be-Admin bootstrap rule.)
#[tokio::test]
#[allow(clippy::await_holding_lock)] // TEST_ENV_LOCK serializes HIGGS_HOME for the test
async fn arm_lan_serve_gates_and_arms_atomically() {
    // Minting persists `api_keys.json` under `HIGGS_HOME`; isolate it so this never
    // touches the real keystore (nor races a sibling test's temp home away).
    let _env = crate::TEST_ENV_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let home = tempfile::tempdir().unwrap();
    let prev = std::env::var_os("HIGGS_HOME");
    // SAFETY: serialized by TEST_ENV_LOCK; restored below.
    unsafe { std::env::set_var("HIGGS_HOME", home.path()) };

    let higgs = Arc::new(fake_higgs(vec![]));
    let guard = higgs.register_serve(false, None);
    assert!(!higgs.lan_exposed(), "registration alone does not arm");

    // Empty keystore → [HG058], and NOTHING is armed: a refused serve must leave no
    // phantom exposure that would strand a later revoke.
    let err = higgs
        .arm_lan_serve(&guard, "0.0.0.0:1")
        .expect_err("keyless LAN bind refused");
    assert!(
        matches!(err, HiggsError::LanBindWithoutKeys { .. }),
        "{err}"
    );
    assert!(!higgs.lan_exposed(), "a refused LAN serve arms nothing");

    // With an Admin key the gate passes AND arms in the same critical section.
    higgs
        .mint_key("admin", Some(vec![Scope::Admin]))
        .expect("first key is admin (bootstrap rule)");
    higgs.arm_lan_serve(&guard, "0.0.0.0:1").expect("armed");
    assert!(higgs.lan_exposed(), "a passing LAN serve is armed");

    // Armed ⇒ the revoke that would EMPTY the store is refused ([HG059]) — the exact
    // invariant the shared-lock atomicity exists to protect.
    assert!(
        matches!(
            higgs.revoke_key("admin"),
            Err(HiggsError::LastKeyOnLan { .. })
        ),
        "emptying the store while a LAN listener is armed is refused"
    );

    // Deregistering lifts the exposure, and the same revoke then succeeds.
    assert!(guard.release(), "sole listener");
    assert!(!higgs.lan_exposed(), "no phantom exposure");
    higgs
        .revoke_key("admin")
        .expect("revoke once nothing is live");

    // SAFETY: serialized by TEST_ENV_LOCK.
    unsafe {
        match prev {
            Some(v) => std::env::set_var("HIGGS_HOME", v),
            None => std::env::remove_var("HIGGS_HOME"),
        }
    }
}

/// A CANCELLED serve future (an embedder aborting the task) never runs code past
/// its await point — but it DOES run destructors. The registration must therefore
/// be released by the guard's `Drop`, or an aborted LAN serve would strand
/// `lan_exposed` at true and refuse a legitimate last-key revoke forever.
#[tokio::test]
async fn aborting_a_serve_future_releases_its_registration() {
    let higgs = Arc::new(fake_higgs(vec![]));
    let serving = Arc::clone(&higgs);
    // A task that registers a LAN listener and then parks forever, exactly as a
    // real `serve_v1` parks on `axum::serve(..).await`.
    let (ready_tx, ready_rx) = tokio::sync::oneshot::channel::<()>();
    let task = tokio::spawn(async move {
        let _guard = serving.register_serve(true, None);
        let _ = ready_tx.send(());
        std::future::pending::<()>().await;
    });
    ready_rx.await.expect("the serve task registered");
    assert!(higgs.lan_exposed(), "the LAN listener is live");

    // Abort it — no code after the await runs, only the guard's Drop.
    task.abort();
    let _ = task.await;
    assert!(
        !higgs.lan_exposed(),
        "an aborted serve must not strand its LAN registration"
    );
}

/// The `bound_addr` disclosure is PER-LISTENER: it comes from the primary
/// (first-registered) live listener, and a sibling serve starting or exiting
/// must never rewrite or erase it. It returns to pre-serve semantics only
/// once every listener is gone. (The CORS disclosure is deliberately NOT
/// per-listener after G7 — see `applied_cors_disclosure_follows_listener_liveness`.)
#[tokio::test]
async fn serve_disclosures_are_per_listener() {
    let higgs = Arc::new(fake_higgs(vec![]));
    let a_addr: std::net::SocketAddr = "127.0.0.1:31415".parse().unwrap();
    let b_addr: std::net::SocketAddr = "127.0.0.1:31416".parse().unwrap();

    let a = higgs.register_serve(false, Some(a_addr));
    // A SECOND listener starts with a different address. It must not overwrite
    // the primary's disclosure (the old flat slots did exactly that).
    let b = higgs.register_serve(false, Some(b_addr));
    assert_eq!(
        higgs.bound_addr(),
        Some(a_addr),
        "primary keeps its address"
    );

    // B exits first: A is still live, so nothing is cleared and nothing shifts.
    drop(b);
    assert_eq!(
        higgs.bound_addr(),
        Some(a_addr),
        "live serve keeps its address"
    );

    // The last one exits: back to honest pre-serve semantics.
    drop(a);
    assert_eq!(higgs.bound_addr(), None, "no listener → no bound address");
}

/// `ServeGuard::release` reports whether the releasing listener was the LAST one —
/// that flag is what makes `serve_v1` own the facade teardown only when nothing else
/// is serving. With a sibling still live, draining the shared node would strand it.
/// It also deregisters immediately (before the caller's terminal worker drain), so a
/// stopped listener never keeps disclosing itself or forcing [HG059].
#[tokio::test]
async fn only_the_last_listener_reports_itself_as_last() {
    let higgs = Arc::new(fake_higgs(vec![]));
    let a = higgs.register_serve(true, None);
    let b = higgs.register_serve(false, None);

    // A sibling is still live → NOT last → its `serve_v1` must not stop the facade.
    assert!(
        !a.release(),
        "a listener with a live sibling is not the last"
    );
    // Releasing also deregistered it: the LAN exposure it contributed is gone at
    // once, without waiting for any worker drain.
    assert!(
        !higgs.lan_exposed(),
        "released LAN listener stops contributing exposure immediately"
    );

    // The remaining one IS last → its `serve_v1` owns the teardown.
    assert!(b.release(), "the final listener reports itself as last");
    assert_eq!(higgs.bound_addr(), None, "registry is empty");
}

/// `release` and `Drop` must not double-deregister: a released guard's drop is a
/// no-op, so it can never remove a LATER listener that reused nothing (ids are
/// monotonic) or mis-report emptiness.
#[tokio::test]
async fn releasing_a_guard_makes_its_drop_a_no_op() {
    let higgs = Arc::new(fake_higgs(vec![]));
    let a = higgs.register_serve(false, None);
    assert!(a.release(), "sole listener is last"); // `a` consumed + dropped here

    // A fresh listener registers after the released guard was dropped.
    let _b = higgs.register_serve(true, None);
    assert!(
        higgs.lan_exposed(),
        "the new listener is live (the released guard's drop removed nothing)"
    );
}

/// The applied-CORS disclosure keys off "any listener live" — after G7 the
/// live list is SHARED across listeners (their layers all read it per
/// request), so per-listener staleness cannot exist: a sibling exit changes
/// nothing, and only the LAST exit returns to pre-serve semantics.
#[tokio::test]
async fn applied_cors_disclosure_follows_listener_liveness() {
    let higgs = Arc::new(fake_higgs(vec![]));
    higgs
        .set_cors_origins(vec!["https://tools.example".to_string()])
        .expect("valid origin persists");
    assert!(
        higgs.cors_settings().applied_origins.is_empty(),
        "pre-serve: nothing is applied"
    );

    let a = higgs.register_serve(false, None);
    let b = higgs.register_serve(false, None);
    assert_eq!(
        higgs.cors_settings().applied_origins,
        vec!["https://tools.example".to_string()],
        "live: the shared list is the applied disclosure for every listener"
    );

    drop(b);
    assert_eq!(
        higgs.cors_settings().applied_origins,
        vec!["https://tools.example".to_string()],
        "a sibling exit changes nothing"
    );

    drop(a);
    assert!(
        higgs.cors_settings().applied_origins.is_empty(),
        "the last exit returns to pre-serve semantics"
    );
}
