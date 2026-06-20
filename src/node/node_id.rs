//! NodeId — the hub's per-paired-node key (u32, Copy). Assigned by the hub the first time
//! a node is admitted and STABLE across reconnects, so `LogSource::RemoteWorker` (which
//! embeds it) stays `Copy` and a node's log ring survives a dropped connection.
//!
//! Distinct from the node's `EndpointId` (a long public-key string, the wire/allowlist
//! identity): `NodeId` is a small local handle the hub mints so logs and the UI can refer
//! to a node compactly (`n-1`). The `EndpointId ↔ NodeId` mapping lives in the fleet.

use std::collections::HashMap;
use std::fmt;

/// Hub-local node key. `u32` + `Copy`. Rendered "n-1" in the UI / `?source=node:1:…`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(pub u32);

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "n-{}", self.0)
    }
}

/// Mints a stable [`NodeId`] per `EndpointId` string: the same endpoint always maps to the
/// same id (so reconnects keep their log ring and routes), and ids are never reused for a
/// different endpoint.
#[derive(Default)]
pub struct NodeIdAllocator {
    next: u32,
    by_endpoint: HashMap<String, NodeId>,
}

impl NodeIdAllocator {
    pub fn new() -> Self {
        Self { next: 1, by_endpoint: HashMap::new() }
    }

    /// The id for `endpoint_id`, assigning a fresh one on first sight.
    pub fn assign(&mut self, endpoint_id: &str) -> NodeId {
        if let Some(id) = self.by_endpoint.get(endpoint_id) {
            return *id;
        }
        let id = NodeId(self.next.max(1));
        self.next = id.0 + 1;
        self.by_endpoint.insert(endpoint_id.to_string(), id);
        id
    }

    /// The id already assigned to `endpoint_id`, if any (no assignment).
    pub fn get(&self, endpoint_id: &str) -> Option<NodeId> {
        self.by_endpoint.get(endpoint_id).copied()
    }
}

#[cfg(test)]
mod tests {
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
}
