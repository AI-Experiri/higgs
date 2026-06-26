
use super::HiggsOk;

/// `HiggsOk::default()` (the `Default` impl) yields the same `{"status":"ok"}`
/// body as `new()`, and both serialize to the canonical wire shape.
#[test]
fn higgs_ok_default_matches_new_and_serializes() {
    let from_default = HiggsOk::default();
    let from_new = HiggsOk::new();
    assert_eq!(from_default.status, "ok");
    assert_eq!(from_default.status, from_new.status);
    assert_eq!(
        serde_json::to_value(&from_default).unwrap(),
        serde_json::json!({ "status": "ok" }),
    );
}
