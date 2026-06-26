//! Served-instance-id derivation, shared by the remote fleet and (P4b) the local engine.
//!
//! N workers serving the same raw model coexist as N instances; each gets a deterministic,
//! collision-free SERVED id: `org/model`, `org/model-1`, … . The mapping is a pure function
//! of the live instance set (never persisted), generic over the instance LOCATION `L` (a
//! remote `NodeKey`, or a local marker) so local and remote share one algorithm.

use std::collections::{HashMap, HashSet};

use crate::node::worker_id::WorkerId;

/// Map every instance `(location, worker, raw_model)` to a unique served id.
///
/// Instances are assigned in sorted `(model, location, worker)` order against a global
/// taken-set; a candidate served id already taken (e.g. a model literally named
/// `org/model-1` clashing with the suffix of a second `org/model` instance) bumps to the next
/// free suffix. So EVERY instance gets a unique, reachable served id — none is ever left
/// unaddressable — and the result is identical every time for a given live set.
pub(crate) fn served_ids<L: Ord + Clone>(
    instances: &[(L, WorkerId, String)],
) -> HashMap<String, (L, WorkerId)> {
    // Sort by (model, location, worker) so the mapping is a stable function of the live set.
    let mut entries: Vec<(&str, &L, WorkerId)> = instances
        .iter()
        .map(|(loc, worker, model)| (model.as_str(), loc, *worker))
        .collect();
    entries.sort_unstable_by(|a, b| a.0.cmp(b.0).then(a.1.cmp(b.1)).then(a.2.cmp(&b.2)));

    let mut taken: HashSet<String> = HashSet::new();
    let mut next_suffix: HashMap<&str, usize> = HashMap::new();
    let mut out = HashMap::new();
    for (model, loc, worker) in entries {
        let i = next_suffix.entry(model).or_insert(0);
        let served = loop {
            let candidate = if *i == 0 {
                model.to_string()
            } else {
                format!("{model}-{i}")
            };
            *i += 1;
            if taken.insert(candidate.clone()) {
                break candidate;
            }
        };
        out.insert(served, (loc.clone(), worker));
    }
    out
}

#[cfg(test)]
#[path = "served_tests.rs"]
mod tests;
