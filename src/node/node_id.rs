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
        Self {
            next: 1,
            by_endpoint: HashMap::new(),
        }
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

    /// Forget a node's id assignment (operator retire/remove) so it no longer appears in
    /// [`all`](Self::all). `next` is NOT rewound — ids stay monotonic and never reused, so a
    /// stale `(node, worker)` reference can't alias a re-added node. No-op if unknown.
    pub fn remove(&mut self, endpoint_id: &str) {
        self.by_endpoint.remove(endpoint_id);
    }

    /// Every `(endpoint_id, NodeId)` assigned so far, ascending by id — the set of nodes the
    /// hub has ever admitted (connected or not), for the fleet view.
    pub fn all(&self) -> Vec<(String, NodeId)> {
        let mut v: Vec<_> = self
            .by_endpoint
            .iter()
            .map(|(e, id)| (e.clone(), *id))
            .collect();
        v.sort_by_key(|(_, id)| id.0);
        v
    }
}

#[cfg(test)]
#[path = "node_id_tests.rs"]
mod tests;
