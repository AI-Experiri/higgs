
use super::*;

#[test]
fn assign_is_stable_per_endpoint_and_monotonic() {
    let mut a = NodeIdAllocator::new();
    let n1 = a.assign("endpointA");
    let n2 = a.assign("endpointB");
    assert_ne!(n1, n2);
    assert!(n2.0 > n1.0, "ids are monotonic");
    // Same endpoint → same id (stable across reconnect).
    assert_eq!(a.assign("endpointA"), n1);
    assert_eq!(a.get("endpointA"), Some(n1));
    assert_eq!(a.get("unknown"), None);
}

#[test]
fn node_id_renders_as_n_prefix() {
    assert_eq!(NodeId(1).to_string(), "n-1");
}

#[test]
fn all_lists_assigned_endpoints_ascending() {
    let mut a = NodeIdAllocator::new();
    let n1 = a.assign("a");
    let n2 = a.assign("b");
    assert_eq!(a.all(), vec![("a".to_string(), n1), ("b".to_string(), n2)]);
}

#[test]
fn remove_drops_the_slot_and_ids_never_reuse() {
    let mut a = NodeIdAllocator::new();
    let n1 = a.assign("a");
    a.assign("b");
    a.remove("a");
    assert!(a.get("a").is_none(), "removed endpoint has no id");
    assert!(a.all().iter().all(|(e, _)| e != "a"), "gone from all()");
    // Re-adding gets a FRESH id (monotonic, never reused), not n1.
    let n1b = a.assign("a");
    assert_ne!(n1, n1b, "re-added node gets a new id");
    // Removing an unknown endpoint is a no-op.
    a.remove("nope");
}
