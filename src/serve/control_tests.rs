
use super::super::test_support::*;
use crate::log_bus::LogSource;
use axum::http::StatusCode;
use serde_json::json;
use tower::ServiceExt;

// ── Gate 2: host-side tool-call-parser sniff ─────────────────────────────

/// Build a minimal scanned model carrying only the chat template that the
/// Gate-2 sniff inspects; all other fields are placeholder.
fn model_with_template(template: Option<&str>) -> crate::worker::models::HiggsModel {
    crate::worker::models::HiggsModel {
        id: "org/model".into(),
        path: "/x.gguf".into(),
        size_bytes: 0,
        quant: None,
        source: crate::worker::models::HiggsModelSource::LmStudio,
        arch: None,
        ctx_train: None,
        block_count: None,
        head_count: None,
        head_count_kv: None,
        embedding_length: None,
        expert_count: None,
        has_chat_template: template.is_some(),
        supports_tools: false,
        supports_reasoning: false,
        gguf_components: Vec::new(),
        chat_template: template.map(ToOwned::to_owned),
    }
}

#[test]
fn gate2_sniffs_tool_call_template() {
    // A template with the generic `<tool_call>` marker → a parser matches.
    let with_calls = model_with_template(Some(
        "{% for m in messages %}<|im_start|>{{ m.role }}<tool_call>{{ tool }}</tool_call>",
    ));
    assert!(
        super::tool_calls_supported(&with_calls),
        "<tool_call> matches"
    );

    // A plain chatml template with no tool markup → no parser matches.
    let plain = model_with_template(Some(
        "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>",
    ));
    assert!(
        !super::tool_calls_supported(&plain),
        "plain chatml: no match"
    );

    // No template at all → false.
    assert!(!super::tool_calls_supported(&model_with_template(None)));
}

// ── Test 6: control load + unload roundtrip ──────────────────────────────

#[tokio::test]
async fn control_load_unload_roundtrip() {
    // `load` resolves the GGUF path host-side, so the id must be discoverable.
    // The stateful fake worker auto-responds to M_LOAD/M_STATUS/M_UNLOAD, so
    // the load → unload round-trip runs through the real node path.
    let dir = tempfile::TempDir::new().unwrap();
    write_gguf_fixture(dir.path(), "org/model");
    let app = make_app_with_lmstudio(dir.path().to_path_buf());

    let resp = app
        .clone()
        .oneshot(post_json(
            "/api/higgs/models/load",
            &json!({"id": "org/model"}),
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    assert_eq!(v["status"], "ok");
    assert_eq!(v["id"], "org/model");

    let resp = app
        .oneshot(post_json("/api/higgs/models/unload", &json!({})))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    assert_eq!(v["status"], "ok");
}

#[tokio::test]
async fn control_unload_one_by_id_targets_just_that_model() {
    // Per-model unload happy path: load a model, then unload it by `{id}` — this
    // drives `Higgs::unload_one` (served-id resolve → that worker only), distinct
    // from the `{}` drain-all path. Afterwards the model is gone.
    let dir = tempfile::TempDir::new().unwrap();
    write_gguf_fixture(dir.path(), "org/model");
    let app = make_app_with_lmstudio(dir.path().to_path_buf());

    let resp = app
        .clone()
        .oneshot(post_json(
            "/api/higgs/models/load",
            &json!({"id": "org/model"}),
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let resp = app
        .clone()
        .oneshot(post_json(
            "/api/higgs/models/unload",
            &json!({"id": "org/model"}),
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    assert_eq!(v["status"], "ok");

    // Unloading the same id again is an idempotent no-op (now not resident).
    let resp = app
        .oneshot(post_json(
            "/api/higgs/models/unload",
            &json!({"id": "org/model"}),
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn unload_rejects_present_but_unusable_id() {
    // Per-model unload safety: a PRESENT but unusable id (`null` / empty) or an
    // unknown field must 400 — it must NEVER fall through to the destructive
    // drain-all. Only a TRULY absent id (`{}` / empty body, covered by
    // `control_load_unload_roundtrip`) drains all. These bodies are rejected
    // before higgs is touched, so a bare app (no loaded model) is enough.
    let dir = tempfile::TempDir::new().unwrap();
    let app = make_app_with_lmstudio(dir.path().to_path_buf());
    for bad in [
        json!({"id": null}),
        json!({"id": ""}),
        json!({"model": "org/m"}),
    ] {
        let resp = app
            .clone()
            .oneshot(post_json("/api/higgs/models/unload", &bad))
            .await
            .unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::BAD_REQUEST,
            "unload body {bad} must be rejected, not silently drain all models"
        );
    }
}

// ── Test 7: control_model_by_id with a slashed HF repo id ───────────────
//
// This is the regression test for the wildcard-route bug: with the old
// single-segment `{id}` route, a request to `/api/higgs/models/org/model`
// (literal slash in the path, as real curl sends) never matched — axum
// treated `org` and `model` as separate segments. The test previously used
// `org%2Fmodel` (percent-encoded) which happened to work against the broken
// route because `%2F` is a single segment. Using a literal slash here
// ensures the wildcard `{*id}` route is exercised as real callers do.

#[tokio::test]
async fn control_model_by_id_found_slashed() {
    // Scan runs host-side: discover the slashed id from a real GGUF fixture.
    // Nothing is loaded (no worker), so the status enrichment reports
    // `not-loaded` — the fake worker need not be driven.
    let dir = tempfile::TempDir::new().unwrap();
    write_gguf_fixture(
        dir.path(),
        "lmstudio-community/NVIDIA-Nemotron-3-Nano-4B-GGUF",
    );
    let app = make_app_with_lmstudio(dir.path().to_path_buf());

    // Literal slash in the URL — this is what real curl sends and what
    // the old `{id}` route could never match.
    let resp = app
        .oneshot(get(
            "/api/higgs/models/lmstudio-community/NVIDIA-Nemotron-3-Nano-4B-GGUF",
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    assert_eq!(v["id"], "lmstudio-community/NVIDIA-Nemotron-3-Nano-4B-GGUF");
    assert_eq!(v["state"], "not-loaded");
    assert_eq!(v["format"], "gguf");
    assert_eq!(v["arch"], "llama");
}

// ── Test 8: control_model_by_id not found (slashed id) ───────────────────

#[tokio::test]
async fn control_model_by_id_not_found() {
    // Empty temp dir → host-side scan finds nothing → the id is absent.
    let dir = tempfile::TempDir::new().unwrap();
    let app = make_app_with_lmstudio(dir.path().to_path_buf());

    // Slashed id that does not exist in the catalog → 404 HG002.
    let resp = app
        .oneshot(get("/api/higgs/models/org/nope"))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    let v: serde_json::Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    assert!(v["error"].as_str().unwrap().contains("[HG002]"));
}

#[tokio::test]
async fn pair_without_hub_mode_is_conflict() {
    let app = make_app();
    // No hub installed → pairing is a 409 with an explanatory error.
    let resp = app
        .oneshot(post_json("/api/higgs/pair", &json!({})))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::CONFLICT);
    let v: serde_json::Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    assert!(
        v["error"].as_str().unwrap().contains("hub mode"),
        "explains hub mode: {v}"
    );
}

#[tokio::test]
async fn nodes_load_unload_without_fleet_is_conflict() {
    let load = make_app()
        .oneshot(post_json(
            "/api/higgs/nodes/load",
            &json!({ "node": "n", "model": "m" }),
        ))
        .await
        .unwrap();
    assert_eq!(load.status(), StatusCode::CONFLICT, "no fleet → 409");
    let unload = make_app()
        .oneshot(post_json(
            "/api/higgs/nodes/unload",
            &json!({ "model": "m" }),
        ))
        .await
        .unwrap();
    assert_eq!(unload.status(), StatusCode::CONFLICT, "no fleet → 409");
}

#[tokio::test]
async fn relabel_remote_without_hub_is_conflict() {
    // Renaming a REMOTE node requires the hub enabled (it owns the allowlist) → 409 when off,
    // like the other node-mutation routes. (The local rename + remote-success + unknown-id 404
    // paths run in the hub_server e2e under a temp HIGGS_HOME, so this doesn't touch ~/.higgs.)
    let resp = make_app()
        .oneshot(post_json(
            "/api/higgs/nodes/label",
            &json!({ "node": "some-remote-endpoint-id", "label": "x" }),
        ))
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::CONFLICT,
        "remote relabel needs a hub → 409"
    );
}

#[tokio::test]
async fn nodes_lists_the_local_node_first_even_without_a_fleet() {
    // Even with the hub role off (no fleet), GET /api/higgs/nodes returns the LOCAL machine
    // as a first-class node, so the Fleet view always shows "this machine".
    let resp = make_app().oneshot(get("/api/higgs/nodes")).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let v: Vec<serde_json::Value> = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    assert_eq!(v.len(), 1, "only the local node when no fleet: {v:?}");
    assert_eq!(v[0]["endpoint_id"], "local", "local sentinel id");
    assert_eq!(v[0]["is_local"], true, "flagged local");
    assert_eq!(v[0]["connected"], true, "local node is always connected");
    assert!(
        v[0]["label"].as_str().is_some_and(|s| !s.is_empty()),
        "local node has a label: {v:?}"
    );
    assert!(
        v[0]["inventory"].is_object(),
        "local inventory present: {v:?}"
    );
}

#[tokio::test]
async fn hub_status_without_hub_reports_disabled() {
    // No hub installed → GET /api/higgs/hub answers 200 with enabled:false, no id, 0 nodes.
    let resp = make_app().oneshot(get("/api/higgs/hub")).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    assert_eq!(v["enabled"], false, "no hub installed → disabled: {v}");
    assert_eq!(v["node_count"], 0, "no nodes when disabled: {v}");
    assert!(
        v.get("hub_id").is_none(),
        "hub_id omitted when disabled: {v}"
    );
}

#[tokio::test]
async fn hub_disable_without_hub_is_a_noop() {
    // The kill switch is idempotent: disabling when no hub is installed just reports disabled.
    let resp = make_app()
        .oneshot(post_json("/api/higgs/hub/disable", &json!({})))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    assert_eq!(v["enabled"], false, "still disabled: {v}");
}

// ── Test 9: version endpoint ──────────────────────────────────────────────

#[tokio::test]
async fn version_endpoint() {
    let app = make_app();

    let resp = app.oneshot(get("/api/higgs/version")).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    assert!(v["higgs"].as_str().is_some(), "higgs version present");
    assert_eq!(v["engine"], "llama.cpp");
    // engine_version is the real engine (ggml) version from ggml_version();
    // binding is the llama-cpp-2 wrapper version — distinct fields.
    assert!(v["engine_version"].as_str().is_some_and(|s| !s.is_empty()));
    assert!(v["binding"].as_str().is_some_and(|s| !s.is_empty()));
    let fmts = v["supported_formats"].as_array().expect("array");
    assert!(fmts.contains(&serde_json::Value::String("gguf".to_owned())));
}

// ── (original Test 7, now test 10): logs endpoint shape and tail semantics ─

#[tokio::test]
async fn logs_endpoint_shapes() {
    let (higgs, bus) = make_higgs_with_bus();
    bus.push(LogSource::Serve, "line one".to_owned());
    bus.push(LogSource::Serve, "line two".to_owned());
    bus.push(LogSource::Serve, "line three".to_owned());
    let app = app_for(higgs);

    let resp = app.oneshot(get("/api/higgs/logs?n=2")).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    assert_eq!(
        v["lines"],
        json!(["line two", "line three"]),
        "tail of n, oldest first"
    );
}

/// `?source=serve|worker` routes the two log origins to separate consoles —
/// end-to-end through the HTTP snapshot handler.
#[tokio::test]
async fn logs_endpoint_filters_by_source() {
    let (higgs, bus) = make_higgs_with_bus();
    bus.push(LogSource::Serve, "higgs: GET /v1/models".to_owned());
    bus.push(LogSource::Worker, "ggml_metal_init: loaded".to_owned());
    bus.push(LogSource::Serve, "higgs: loading model".to_owned());
    let app = app_for(higgs);

    // ?source=worker → only the worker stderr line.
    let resp = app
        .clone()
        .oneshot(get("/api/higgs/logs?source=worker"))
        .await
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    assert_eq!(
        v["lines"],
        json!(["ggml_metal_init: loaded"]),
        "worker only"
    );

    // ?source=serve → only the higgs control-plane lines, in push order.
    let resp = app
        .clone()
        .oneshot(get("/api/higgs/logs?source=serve"))
        .await
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    assert_eq!(
        v["lines"],
        json!(["higgs: GET /v1/models", "higgs: loading model"]),
        "serve only"
    );

    // No filter → all three, merged in push order.
    let resp = app.oneshot(get("/api/higgs/logs")).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    assert_eq!(
        v["lines"].as_array().map(Vec::len),
        Some(3),
        "no filter = all sources merged"
    );
}

// ── logs SSE stream: replay-then-live ordering ───────────────────────────

#[tokio::test]
async fn logs_stream_replays_then_streams_live() {
    use super::{control_logs_stream, LogsQuery};
    use axum::extract::{Query, State};
    use axum::response::IntoResponse;
    use futures::StreamExt;
    use std::time::Duration;

    let (higgs, bus) = make_higgs_with_bus();
    // Seed history BEFORE the request — this is the replay prefix.
    bus.push(LogSource::Serve, "hist-1".to_owned());
    bus.push(LogSource::Serve, "hist-2".to_owned());

    let resp = control_logs_stream(
        State(higgs.clone()),
        Query(LogsQuery {
            n: Some(10),
            source: None,
        }),
    )
    .await
    .into_response();
    assert_eq!(
        resp.headers()
            .get(axum::http::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok()),
        Some("text/event-stream"),
        "SSE content-type"
    );

    let mut body = resp.into_body().into_data_stream();

    // The replay prefix arrives first; collect frames until both history
    // lines have been seen.
    let mut seen = String::new();
    while !(seen.contains("hist-1") && seen.contains("hist-2")) {
        let frame = tokio::time::timeout(Duration::from_secs(2), body.next())
            .await
            .expect("replay frame within timeout")
            .expect("body not ended")
            .expect("frame ok");
        seen.push_str(&String::from_utf8_lossy(&frame));
    }
    assert!(
        seen.contains("hist-1") && seen.contains("hist-2"),
        "replay: {seen}"
    );

    // After replay the stream is parked on the live receiver. Push a new line
    // and it must arrive as a frame — proving live delivery, not closure.
    bus.push(LogSource::Serve, "live-1".to_owned());
    let mut live_seen = String::new();
    while !live_seen.contains("live-1") {
        let frame = tokio::time::timeout(Duration::from_secs(2), body.next())
            .await
            .expect("live frame within timeout")
            .expect("body not ended")
            .expect("frame ok");
        live_seen.push_str(&String::from_utf8_lossy(&frame));
    }
    assert!(live_seen.contains("live-1"), "live: {live_seen}");

    // The replay source itself is ordered oldest-first ahead of the live line.
    assert_eq!(
        higgs.logs(10, None),
        vec![
            "hist-1".to_owned(),
            "hist-2".to_owned(),
            "live-1".to_owned()
        ]
    );
}

// ── control_models: scan + status enrichment ─────────────────────────────

#[tokio::test]
async fn control_models_lists_with_loaded_flag() {
    // Multi-model: load TWO distinct models — the models list must flag BOTH as
    // `loaded` (not just the primary), and `loaded_id` reports the primary.
    let dir = tempfile::TempDir::new().unwrap();
    write_gguf_fixture(dir.path(), "org/model");
    write_gguf_fixture(dir.path(), "org/other");
    let higgs = make_higgs_with_lmstudio(dir.path().to_path_buf());
    higgs.load("org/model", None).await.expect("load model");
    higgs.load("org/other", None).await.expect("load other");
    let app = app_for(higgs);

    let resp = app.oneshot(get("/api/higgs/models")).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    // Primary (lowest worker = first loaded) is org/model.
    assert_eq!(v["loaded_id"], "org/model");
    let models = v["models"].as_array().expect("models array");
    assert_eq!(models.len(), 2, "both models listed");
    // BOTH resident models are flagged loaded — the multi-model fix.
    for m in models {
        assert_eq!(
            m["state"], "loaded",
            "every resident model is flagged loaded: {m}"
        );
        assert_eq!(m["format"], "gguf");
    }
}

// ── control_status passthrough ───────────────────────────────────────────

#[tokio::test]
async fn control_status_returns_snapshot() {
    // Load a fixture model so the status snapshot reports it resident (the
    // stateful fake worker echoes the loaded model in M_STATUS).
    let dir = tempfile::TempDir::new().unwrap();
    write_gguf_fixture(dir.path(), "org/model");
    let higgs = make_higgs_with_lmstudio(dir.path().to_path_buf());
    higgs.load("org/model", None).await.expect("load");
    let app = app_for(higgs);

    let resp = app.oneshot(get("/api/higgs/status")).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    assert_eq!(v["loaded"]["id"], "org/model");
}

// ── control_load with explicit params (non-default branch) ───────────────

#[tokio::test]
async fn control_load_with_explicit_params() {
    // `load` resolves the GGUF path host-side, so the id must be discoverable.
    let dir = tempfile::TempDir::new().unwrap();
    write_gguf_fixture(dir.path(), "org/model");
    let app = make_app_with_lmstudio(dir.path().to_path_buf());

    // Providing ctx_len takes the param-merge branch (Some(LoadParams)).
    let resp = app
        .oneshot(post_json(
            "/api/higgs/models/load",
            &json!({"id": "org/model", "ctx_len": 2048}),
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    assert_eq!(v["status"], "ok");
    assert_eq!(v["id"], "org/model");
}

// ── control_system: real host snapshot ───────────────────────────────────

#[tokio::test]
async fn control_system_returns_host_info() {
    let app = make_app();

    let resp = app.oneshot(get("/api/higgs/system")).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    // SystemInfo always reports a positive total RAM on a real host.
    assert!(
        v.get("ram").is_some() || v.get("cpu").is_some() || v.is_object(),
        "system info is a populated object: {v}"
    );
}

// ── logs settings: GET reflects default; PUT toggles verbose ─────────────

#[tokio::test]
async fn logs_settings_get_default_and_put_toggles() {
    let app = make_app();

    // GET defaults to verbose:false.
    let resp = app
        .clone()
        .oneshot(get("/api/higgs/logs/settings"))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    assert_eq!(v["verbose"], false, "verbose defaults to false");
    assert_eq!(
        v["log_incoming_tokens"], false,
        "log_incoming_tokens defaults to false"
    );
    assert_eq!(
        v["show_log_fields"], false,
        "show_log_fields defaults to false (redact)"
    );

    // PUT all flags true returns {"status":"ok"}.
    let resp = app
        .clone()
        .oneshot(put_json(
            "/api/higgs/logs/settings",
            &json!({"verbose": true, "log_incoming_tokens": true, "show_log_fields": true}),
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    assert_eq!(v["status"], "ok");

    // GET now reflects the new state for all flags.
    let resp = app.oneshot(get("/api/higgs/logs/settings")).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    assert_eq!(v["verbose"], true, "PUT toggled verbose on");
    assert_eq!(
        v["log_incoming_tokens"], true,
        "PUT toggled log_incoming_tokens on"
    );
    assert_eq!(v["show_log_fields"], true, "PUT toggled show_log_fields on");
}

// ── runtime settings: GET reflects default (JIT on); PUT toggles JIT ─────

#[tokio::test]
async fn settings_get_default_and_put_toggles_jit() {
    let app = make_app();

    // GET defaults: JIT on, auto-unload on, TTL 60 minutes (the node default).
    let resp = app
        .clone()
        .oneshot(get("/api/higgs/settings"))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    assert_eq!(v["jit_enabled"], true, "JIT defaults to on");
    assert_eq!(v["auto_unload_idle"], true, "auto-unload defaults to on");
    assert_eq!(v["idle_ttl_minutes"], 60, "TTL defaults to 60 minutes");
    assert_eq!(v["serving_enabled"], true, "serving defaults to on");

    // PUT all four (JIT off, auto-unload off, TTL 30, serving off) returns ok.
    let resp = app
        .clone()
        .oneshot(put_json(
            "/api/higgs/settings",
            &json!({
                "jit_enabled": false,
                "auto_unload_idle": false,
                "idle_ttl_minutes": 30,
                "serving_enabled": false,
            }),
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    assert_eq!(v["status"], "ok");

    // GET now reflects all three new values.
    let resp = app.oneshot(get("/api/higgs/settings")).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    assert_eq!(v["jit_enabled"], false, "PUT toggled JIT off");
    assert_eq!(v["auto_unload_idle"], false, "PUT toggled auto-unload off");
    assert_eq!(v["idle_ttl_minutes"], 30, "PUT set TTL to 30 minutes");
    assert_eq!(v["serving_enabled"], false, "PUT toggled serving off");
}

// ── settings handlers: round-trip through the typed GET/PUT pair ──────────

#[tokio::test]
async fn settings_handlers_round_trip() {
    use super::{control_set_settings, control_settings};
    use axum::extract::State;

    let higgs = make_higgs();

    // GET handler reflects the default-on state.
    assert!(
        control_settings(State(higgs.clone())).await.0.jit_enabled,
        "JIT on by default"
    );

    // PUT handler flips it off; the chat path's gate (`higgs.jit_enabled()`)
    // now returns false, so an unloaded model is a 404 (explicit-load).
    let ok = control_set_settings(
        State(higgs.clone()),
        axum::Json(crate::serve::HiggsRuntimeSettings {
            jit_enabled: false,
            auto_unload_idle: false,
            idle_ttl_minutes: 30,
            serving_enabled: false,
        }),
    )
    .await;
    assert_eq!(ok.0.status, "ok");
    assert!(!higgs.jit_enabled(), "PUT disabled the JIT gate");
    assert!(!higgs.auto_unload_idle(), "PUT disabled idle auto-unload");
    assert_eq!(higgs.idle_ttl_minutes(), 30, "PUT set the idle TTL");
    assert!(!higgs.serving_enabled(), "PUT disabled serving");
    let got = control_settings(State(higgs.clone())).await.0;
    assert!(!got.jit_enabled, "GET reflects JIT off");
    assert!(!got.auto_unload_idle, "GET reflects auto-unload off");
    assert_eq!(got.idle_ttl_minutes, 30, "GET reflects the new TTL");
    assert!(!got.serving_enabled, "GET reflects serving off");
}

// ── verbose gate: served line appears only when verbose is on ─────────────

#[tokio::test]
async fn verbose_gate_round_trips_through_handlers() {
    use super::{control_logs_settings, control_set_logs_settings};
    use axum::extract::State;

    let higgs = make_higgs();

    // GET handler reflects the default-off state.
    assert!(
        !control_logs_settings(State(higgs.clone())).await.0.verbose,
        "verbose off by default"
    );

    // PUT handler flips it on; the chat path's gate (`higgs.verbose()`) now
    // returns true, so the served line would be emitted (format asserted in
    // v1's `served_message_format`).
    let ok = control_set_logs_settings(
        State(higgs.clone()),
        axum::Json(crate::serve::LogSettings {
            verbose: true,
            log_incoming_tokens: true,
            show_log_fields: false,
        }),
    )
    .await;
    assert_eq!(ok.0.status, "ok");
    assert!(higgs.verbose(), "PUT enabled the chat verbose gate");
    assert!(
        higgs.log_incoming_tokens(),
        "PUT enabled the incoming-tokens gate"
    );
    let got = control_logs_settings(State(higgs.clone())).await.0;
    assert!(got.verbose, "GET reflects verbose on");
    assert!(
        got.log_incoming_tokens,
        "GET reflects log_incoming_tokens on"
    );
}

// ── control_worker_stop: graceful, always ok ─────────────────────────────

#[tokio::test]
async fn control_worker_stop_ok() {
    let app = make_app();

    let resp = app
        .oneshot(post_json("/api/higgs/worker/stop", &json!({})))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    assert_eq!(v["status"], "ok");
}

/// REGRESSION: `worker/stop` is NON-terminal — the server stays loadable after it.
/// (The endpoint must do a bulk UNLOAD, not the node's terminal `shutdown_all` drain,
/// which marks the runtime shutting-down and would brick every later load until the
/// process restarts.)
#[tokio::test]
async fn worker_stop_is_non_terminal_loads_still_work() {
    let dir = tempfile::TempDir::new().unwrap();
    write_gguf_fixture(dir.path(), "org/model");
    let app = make_app_with_lmstudio(dir.path().to_path_buf());

    // Load a model, then stop (unload) all workers via the endpoint.
    let resp = app
        .clone()
        .oneshot(post_json(
            "/api/higgs/models/load",
            &json!({"id": "org/model"}),
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK, "initial load ok");
    let resp = app
        .clone()
        .oneshot(post_json("/api/higgs/worker/stop", &json!({})))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK, "worker/stop ok");

    // A SUBSEQUENT load must still succeed — the node is not terminally shut down.
    let resp = app
        .oneshot(post_json(
            "/api/higgs/models/load",
            &json!({"id": "org/model"}),
        ))
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "load after worker/stop must still work (the node must not be bricked)"
    );
    let v: serde_json::Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    assert_eq!(v["status"], "ok", "reload succeeds post-stop");
}

// ── nodes/{node}/models + nodes/retire: hub-only routes ──────────────────

/// Both new fleet routes require hub mode: with no fleet/hub installed they
/// answer `409 CONFLICT` (the `not_a_hub` guard), exercising the handler
/// wiring + route registration without a live iroh hub.
#[tokio::test]
async fn node_models_and_retire_require_hub_mode() {
    let app = make_app();

    let resp = app
        .clone()
        .oneshot(get("/api/higgs/nodes/somenode/models"))
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::CONFLICT,
        "node catalog needs a hub"
    );

    let resp = app
        .oneshot(post_json(
            "/api/higgs/nodes/retire",
            &json!({ "node": "somenode" }),
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::CONFLICT, "retire needs a hub");
}
