
use super::*;

#[test]
fn collision_free_and_deterministic_across_locations() {
    // Two `org/model` instances (on two locations) want `org/model` + `org/model-1`; a
    // literal model `org/model-1` also wants `org/model-1`. All four must get unique,
    // reachable served ids, deterministically.
    let instances = vec![
        ("locA".to_string(), WorkerId(2), "org/model".to_string()),
        ("locB".to_string(), WorkerId(1), "org/model".to_string()),
        ("locA".to_string(), WorkerId(3), "org/model-1".to_string()),
        ("locA".to_string(), WorkerId(1), "solo/x".to_string()),
    ];
    let served = served_ids(&instances);
    assert_eq!(served.len(), 4, "every instance reachable: {served:?}");
    let workers: HashSet<_> = served.values().cloned().collect();
    assert_eq!(workers.len(), 4, "no two served ids share an instance");
    assert!(served.contains_key("solo/x"));
    assert!(served.contains_key("org/model"));
    assert!(served.contains_key("org/model-1"));
    // Deterministic.
    assert_eq!(served_ids(&instances), served);
}
