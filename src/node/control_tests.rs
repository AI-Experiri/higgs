use super::*;
use crate::node::test_support::fake_runtime as fake_runtime_with_dirs;
use tempfile::TempDir;

fn fake_runtime() -> NodeRuntime {
    fake_runtime_with_dirs(vec![])
}

/// A fake-backed runtime whose scan root holds a dummy GGUF for each `id`, so the real
/// `load` path (resolve → spawn → M_LOAD) runs without llama.cpp. Mirrors the runtime
/// tests' helper. Keep the returned `TempDir` alive for the test's duration.
fn fake_runtime_with_models(ids: &[&str]) -> (NodeRuntime, TempDir) {
    let dir = TempDir::new().expect("staging dir");
    for id in ids {
        let model_dir = dir.path().join(id);
        std::fs::create_dir_all(&model_dir).expect("model dir");
        std::fs::write(model_dir.join("m.gguf"), b"GGUF\x00 dummy").expect("write dummy gguf");
    }
    let rt = fake_runtime_with_dirs(vec![dir.path().to_path_buf()]);
    (rt, dir)
}

/// Load model `id` through the runtime and return its assigned worker id (the fake
/// worker answers `M_LOAD` so no llama.cpp is involved).
async fn load_worker(rt: &NodeRuntime, id: &str) -> u32 {
    let resp = dispatch_node_control(rt, req(1, M_NODE_LOAD, json!({ "id": id }))).await;
    assert!(resp.error.is_none(), "load ok: {resp:?}");
    resp.result.expect("load result")["worker_id"]
        .as_u64()
        .expect("worker_id is a number") as u32
}

fn req(id: u64, method: &str, params: Value) -> RpcRequest {
    RpcRequest {
        jsonrpc: "2.0".into(),
        id,
        method: method.into(),
        params,
    }
}

#[tokio::test]
async fn unknown_method_is_method_not_found() {
    let rt = fake_runtime();
    let resp = dispatch_node_control(&rt, req(1, "higgs/node/bogus", json!({}))).await;
    assert_eq!(resp.error.unwrap().code, METHOD_NOT_FOUND);
}

#[tokio::test]
async fn load_with_unhonorable_param_is_invalid_params() {
    let rt = fake_runtime();
    // idle_ttl_minutes is rejected by deny_unknown_fields → -32602.
    let resp = dispatch_node_control(
        &rt,
        req(1, M_NODE_LOAD, json!({ "id": "m", "idle_ttl_minutes": 5 })),
    )
    .await;
    assert_eq!(resp.error.unwrap().code, INVALID_PARAMS);
}

#[tokio::test]
async fn load_missing_model_maps_hg002() {
    let rt = fake_runtime(); // no model roots → ModelNotFound
    let resp = dispatch_node_control(&rt, req(1, M_NODE_LOAD, json!({ "id": "missing" }))).await;
    let e = resp.error.unwrap();
    assert_eq!(e.code, INTERNAL_ERROR);
    assert_eq!(e.data.unwrap()["code"], "HG002");
}

#[tokio::test]
async fn update_rejects_empty_params_but_accepts_well_formed_ones() {
    // `accept_node_update` builds the reply + a DEFERRED apply; `handle_node_stream` runs the
    // deferred ONLY after the reply is written (so a lost reply never applies). No deferred here
    // is exercised — we assert the reply shape + that a well-formed push yields a deferred apply.
    // Empty params: needs manifest + manifest_sig + artifact_url → INVALID_PARAMS, NO deferred.
    let (resp, deferred) = accept_node_update(&req(1, "higgs/node/update", json!({})));
    let e = resp.error.expect("empty params refused");
    assert_eq!(e.code, INVALID_PARAMS);
    assert!(deferred.is_none(), "a rejected push has nothing to apply");
    // Well-formed params reply "accepted" + carry a deferred apply (the caller spawns it after
    // the reply is on the wire; the apply itself fails closed HG081/HG087 async).
    let (resp, deferred) = accept_node_update(&req(
        2,
        "higgs/node/update",
        json!({
            "manifest": "{}",
            "manifest_sig": "sig",
            "artifact_url": "https://example.com/higgs.tar.gz"
        }),
    ));
    assert!(
        resp.error.is_none(),
        "well-formed push is accepted: {resp:?}"
    );
    assert_eq!(resp.result.unwrap()["status"], "accepted");
    assert!(
        deferred.is_some(),
        "a well-formed push carries a deferred apply"
    );
    // Drop the deferred WITHOUT spawning — this test must not launch the detached apply.
    drop(deferred);
}

#[tokio::test]
async fn update_rejects_an_oversized_inline_manifest_synchronously() {
    // A paired-but-compromised hub could push a huge inline manifest to OOM the node. The
    // handler caps it SYNCHRONOUSLY (before building any deferred apply) and refuses in the
    // reply — the push is NOT "accepted" and there is NO deferred. Dropping the precheck lets
    // this reply "accepted".
    let big = "x".repeat(70 * 1024); // > MAX_MANIFEST_BYTES (64 KiB)
    let (resp, deferred) = accept_node_update(&req(
        1,
        "higgs/node/update",
        json!({
            "manifest": big,
            "manifest_sig": "sig",
            "artifact_url": "https://example.com/higgs.tar.gz"
        }),
    ));
    assert!(
        resp.result.is_none(),
        "an oversized push is never accepted: {resp:?}"
    );
    assert!(deferred.is_none(), "an oversized push has nothing to apply");
    let e = resp.error.expect("oversized push refused in the reply");
    assert_eq!(e.code, INTERNAL_ERROR, "size cap maps through err_from");
}

#[tokio::test]
async fn inventory_dispatch_reports_host_and_no_workers() {
    let rt = fake_runtime(); // empty registry
    let resp = dispatch_node_control(&rt, req(1, M_NODE_INVENTORY, json!({}))).await;
    assert!(resp.error.is_none(), "inventory ok: {resp:?}");
    let inv = resp.result.unwrap();
    assert!(
        inv["hardware"]["cpu_cores"].as_u64().unwrap() > 0,
        "real hw"
    );
    assert!(!inv["os"].as_str().unwrap().is_empty(), "os present");
    assert!(
        inv["workers"].as_array().unwrap().is_empty(),
        "no workers loaded"
    );
}

#[tokio::test]
async fn sysinfo_and_status_dispatch_ok() {
    let rt = fake_runtime();
    // sysinfo is node-level (no worker needed).
    let sysinfo = dispatch_node_control(&rt, req(1, M_NODE_SYSINFO, json!({}))).await;
    assert!(sysinfo.error.is_none());
    assert!(sysinfo.result.unwrap().get("hardware").is_some());

    // status for an unknown worker errors cleanly.
    let status =
        dispatch_node_control(&rt, req(2, M_NODE_STATUS, json!({ "worker_id": 999 }))).await;
    assert!(status.error.is_some());
}

/// `M_NODE_SCAN` returns the node's catalog (`{ "models": [...] }`) with no worker — the
/// scan-dispatch Ok arm (control.rs:68-70). The staged dummy model must appear.
#[tokio::test]
async fn scan_dispatch_lists_staged_model() {
    let (rt, _dir) = fake_runtime_with_models(&["org/scanme"]);
    let resp = dispatch_node_control(&rt, req(1, M_NODE_SCAN, json!({}))).await;
    assert!(resp.error.is_none(), "scan ok: {resp:?}");
    let models = resp.result.expect("scan result");
    let arr = models["models"].as_array().expect("models array");
    assert!(
        arr.iter().any(|m| m["id"] == "org/scanme"),
        "staged model is cataloged: {arr:?}"
    );
}

/// Load → status → unload → kill drives the Ok arms of `M_NODE_STATUS` (control.rs:60),
/// `M_NODE_UNLOAD` (control.rs:46 — `ok_value(id, {})`), and `M_NODE_KILL` (control.rs:53),
/// all via the dispatcher with a real (fake-backed) worker.
#[tokio::test]
async fn load_status_unload_kill_dispatch_ok() {
    let (rt, _dir) = fake_runtime_with_models(&["org/a", "org/b"]);

    // STATUS Ok arm: a resident worker answers (fake worker → `{ "loaded": null }`).
    let wa = load_worker(&rt, "org/a").await;
    let status =
        dispatch_node_control(&rt, req(2, M_NODE_STATUS, json!({ "worker_id": wa }))).await;
    assert!(status.error.is_none(), "status ok: {status:?}");
    assert!(
        status.result.unwrap().get("loaded").is_some(),
        "status carries loaded field"
    );

    // UNLOAD Ok arm → empty object result, error absent.
    let unload =
        dispatch_node_control(&rt, req(3, M_NODE_UNLOAD, json!({ "worker_id": wa }))).await;
    assert!(unload.error.is_none(), "unload ok: {unload:?}");
    assert_eq!(unload.result.expect("unload result"), json!({}));
    assert_eq!(unload.id, 3, "reply echoes the request id");

    // KILL Ok arm for a fresh worker → empty object result.
    let wb = load_worker(&rt, "org/b").await;
    let kill = dispatch_node_control(&rt, req(4, M_NODE_KILL, json!({ "worker_id": wb }))).await;
    assert!(kill.error.is_none(), "kill ok: {kill:?}");
    assert_eq!(kill.result.expect("kill result"), json!({}));
}

/// Unload/kill of an UNKNOWN worker hit the runtime's `no_worker` error → `err_from`
/// (control.rs:47/54). The HG code travels in `data`.
#[tokio::test]
async fn unload_and_kill_unknown_worker_error() {
    let rt = fake_runtime();
    for (id, method) in [(1u64, M_NODE_UNLOAD), (2, M_NODE_KILL)] {
        let resp = dispatch_node_control(&rt, req(id, method, json!({ "worker_id": 7 }))).await;
        let e = resp.error.expect("unknown worker errors");
        assert_eq!(e.code, INTERNAL_ERROR, "{method} maps to internal error");
    }
}

/// `WorkerRef`-shaped methods (unload/kill/status) take the INVALID_PARAMS parse-failure
/// closure arm (control.rs:48, 55, 62) when `worker_id` is missing/mistyped — the `Err(resp)
/// => resp(id)` branch. Table-driven across all three so one test covers every parse arm.
#[tokio::test]
async fn worker_ref_methods_reject_bad_params() {
    let rt = fake_runtime();
    for (rid, method) in [
        (10u64, M_NODE_UNLOAD),
        (11, M_NODE_KILL),
        (12, M_NODE_STATUS),
    ] {
        // worker_id is required + must be a u32; a string fails deserialization.
        let resp =
            dispatch_node_control(&rt, req(rid, method, json!({ "worker_id": "nope" }))).await;
        let e = resp.error.expect("bad params errors");
        assert_eq!(e.code, INVALID_PARAMS, "{method} → -32602");
        assert!(
            e.message.starts_with("invalid params:"),
            "{method} message names the parse error: {}",
            e.message
        );
        assert_eq!(resp.id, rid, "the parse closure preserves the request id");
    }
}

// ── accept_node_update_version (M_NODE_UPDATE_VERSION) ──────────────────────

#[test]
fn update_version_rejects_missing_and_non_semver_versions() {
    // Missing `version` → INVALID_PARAMS, nothing deferred.
    let (resp, deferred) =
        accept_node_update_version(&req(1, crate::remote::M_NODE_UPDATE_VERSION, json!({})));
    let e = resp.error.expect("empty params refused");
    assert_eq!(e.code, INVALID_PARAMS);
    assert!(
        deferred.is_none(),
        "a rejected trigger has nothing to apply"
    );
    // A tag-shaped string is NOT plain semver — the syntax gate fails fast in the
    // reply instead of detached.
    let (resp, deferred) = accept_node_update_version(&req(
        2,
        crate::remote::M_NODE_UPDATE_VERSION,
        json!({ "version": "v1.2.3" }),
    ));
    let e = resp.error.expect("tag-shaped version refused");
    assert_eq!(e.code, INVALID_PARAMS);
    assert!(
        e.message.contains("semver"),
        "names the rule: {}",
        e.message
    );
    assert!(deferred.is_none());
}

#[test]
fn update_version_accepts_plain_semver_and_defers_the_apply() {
    let (resp, deferred) = accept_node_update_version(&req(
        3,
        crate::remote::M_NODE_UPDATE_VERSION,
        json!({ "version": "1.2.3" }),
    ));
    assert!(resp.error.is_none(), "plain semver accepted: {resp:?}");
    let result = resp.result.expect("accepted reply");
    assert_eq!(result["status"], "accepted");
    assert_eq!(result["target_version"], "1.2.3");
    // Deferred apply present; drop WITHOUT spawning (same contract as the manifest push).
    assert!(
        deferred.is_some(),
        "a valid trigger carries a deferred apply"
    );
    drop(deferred);
}

#[tokio::test]
async fn update_version_spawn_outside_an_install_layout_fails_without_applying() {
    // The detached apply resolves this daemon's install dir FIRST; a test binary
    // is not in the bin/v<ver>/current layout, so the apply fails fast (logged,
    // nothing staged) — the reply was already "accepted", per the deferred
    // contract. This drives the REAL spawn path end-to-end minus the apply.
    let (resp, deferred) = accept_node_update_version(&req(
        7,
        crate::remote::M_NODE_UPDATE_VERSION,
        json!({ "version": "1.2.3" }),
    ));
    assert!(resp.error.is_none());
    deferred.expect("deferred").spawn();
    // Give the detached task a moment to run to its logged failure.
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
}

/// M_NODE_PULL_STATUS's reply must serialize `HelloDownload` VERBATIM
/// (r49 finding: a hand-built JSON subset lost the r48 `cancellable`
/// field and made the live refresh downgrade registry-backed rows to
/// observe-only). Test the announce → serialize → decode round-trip that
/// the dispatch performs: any field on `HelloDownload` survives.
#[test]
fn m_node_pull_status_reply_serializes_the_full_hello_download() {
    use crate::remote::HelloDownload;
    // Register an in-process download so the announcement has a
    // registry-backed row (cancellable=true).
    let reg = crate::catalog::cancel::node_registry();
    let _guard = reg
        .register(None, "acme/reg-pull-status", "cancellable-check.gguf")
        .expect("register");
    // Mirror the r49 dispatch path: announce → serde_json::to_value → decode.
    let announced = crate::node::data::announced_downloads();
    let wire = serde_json::to_value(&announced).expect("full wire serialize");
    let decoded: Vec<HelloDownload> = serde_json::from_value(wire).expect("round-trip");
    let row = decoded
        .iter()
        .find(|d| d.repo == "acme/reg-pull-status" && d.file == "cancellable-check.gguf")
        .expect("our registry row is in the announcement");
    assert!(
        row.cancellable,
        "registry-backed row is cancellable (dispatch must serialize the full wire type, not a subset): {decoded:?}"
    );
}
