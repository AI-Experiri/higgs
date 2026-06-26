
use super::*;
use crate::node::test_support::fake_runtime as fake_runtime_with_dirs;

fn fake_runtime() -> NodeRuntime {
    fake_runtime_with_dirs(vec![])
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
async fn update_is_recognized_but_refused_hg026() {
    let rt = fake_runtime();
    let resp = dispatch_node_control(&rt, req(1, M_NODE_UPDATE, json!({}))).await;
    let e = resp.error.expect("update refused");
    assert_eq!(e.code, INTERNAL_ERROR);
    assert_eq!(e.data.unwrap()["code"], "HG026", "typed update-unsupported");
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
