//! WorkerId — the per-node worker key (u32, Copy). Owned by the NodeRuntime registry;
//! assigned on load, freed on unload/kill (DESIGN-remote.md §5.4a). `Copy` so
//! `LogSource::RemoteWorker` stays `Copy` in P4.

use std::collections::HashMap;
use std::fmt;

/// Per-node worker key. `u32` + `Copy`. The wire carries it as a number
/// (`"worker_id": 1`); the UI may render it "w-1".
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct WorkerId(pub u32);

impl fmt::Display for WorkerId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "w-{}", self.0)
    }
}

/// A monotonic-id registry of live workers. Generic over the stored value
/// (`Arc<Supervisor>` in production; a fake in tests). Ids are never reused, so a stale
/// `(node, worker)` reference can't later alias a different worker.
pub struct WorkerRegistry<T> {
    next: u32,
    map: HashMap<WorkerId, T>,
}

impl<T> WorkerRegistry<T> {
    pub fn new() -> Self {
        Self {
            next: 1,
            map: HashMap::new(),
        }
    }

    /// Assign the next id and store `value`; returns the new id.
    pub fn insert(&mut self, value: T) -> WorkerId {
        let id = self.reserve();
        self.map.insert(id, value);
        id
    }

    /// Reserve the next id WITHOUT storing a value, so a caller can wire up id-dependent
    /// work (e.g. start a log relay tagged with the id) before the value is ready. Commit the
    /// value later with [`insert_reserved`](Self::insert_reserved). Ids are never reused, so an
    /// abandoned reservation (e.g. a failed load) just leaves a harmless gap.
    pub fn reserve(&mut self) -> WorkerId {
        let id = WorkerId(self.next);
        self.next += 1;
        id
    }

    /// Store `value` under a previously [`reserve`](Self::reserve)d id.
    pub fn insert_reserved(&mut self, id: WorkerId, value: T) {
        self.map.insert(id, value);
    }

    pub fn get(&self, id: WorkerId) -> Option<&T> {
        self.map.get(&id)
    }

    pub fn remove(&mut self, id: WorkerId) -> Option<T> {
        self.map.remove(&id)
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Live worker ids, ascending.
    pub fn ids(&self) -> Vec<WorkerId> {
        let mut v: Vec<_> = self.map.keys().copied().collect();
        v.sort();
        v
    }
}

impl<T> Default for WorkerRegistry<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[path = "worker_id_tests.rs"]
mod tests;
