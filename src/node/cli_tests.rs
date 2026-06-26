
use super::*;

#[test]
fn link_rejects_unknown_subcommand() {
    assert!(run_link(&["bogus".into()]).is_err());
    assert!(run_link(&[]).is_err());
}

#[test]
fn node_rejects_unknown_subcommand() {
    assert!(run_node(&["bogus".into()]).is_err());
    assert!(run_node(&[]).is_err());
}

#[test]
fn node_connect_requires_a_ticket() {
    // No ticket arg → usage error, before any runtime/bind.
    assert!(run_node(&["connect".into()]).is_err());
}

#[test]
fn node_connect_rejects_malformed_ticket() {
    // A malformed ticket fails at parse, before any runtime/bind/network.
    let err = run_node(&["connect".into(), "not-a-ticket".into()]).unwrap_err();
    assert!(!err.to_string().is_empty());
}
