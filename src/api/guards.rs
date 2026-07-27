//! Host-side guards for `Higgs::load`: model-id charset validation, path-traversal
//! containment, and the pre-load RAM headroom check. Split out of `api.rs`.

use std::path::PathBuf;

use tracing::warn;

use crate::diagnostic::HiggsError;

use super::MEMORY_HEADROOM_FRACTION;

/// Validate a model id's charset before it is used to resolve a filesystem path.
///
/// Identity is the HuggingFace repo id (`org/model`); Ollama-sourced ids keep their
/// `ollama/{name}:{tag}` form (see [`HiggsModel::id`](crate::worker::models::HiggsModel)). The
/// accepted charset mirrors ollama's byte-level name validation (`types/model/name.go`): ASCII
/// alphanumerics plus `_ - . / :`. The structural separators `/` (org/model) and `:` (ollama
/// tag) are permitted, but a path component of exactly `..` is rejected outright — that is the
/// traversal vector — as are empty ids and absolute paths. `Err(InvalidModelId)` (→ 400) on
/// any violation.
pub(super) fn validate_repo_id(id: &str) -> Result<(), HiggsError> {
    let reject = |reason: &str| {
        Err(HiggsError::InvalidModelId {
            id: id.to_owned(),
            reason: reason.to_owned(),
        })
    };
    if id.is_empty() {
        return reject("id is empty");
    }
    if id.starts_with('/') || id.starts_with('\\') {
        return reject("id must not be an absolute path");
    }
    if id.contains('\0') {
        return reject("id contains a NUL byte");
    }
    for ch in id.chars() {
        let ok = ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.' | '/' | ':');
        if !ok {
            return reject(&format!("id contains an illegal character {ch:?}"));
        }
    }
    // A `..` path component is the traversal vector — reject it even though its
    // bytes are individually legal. Split on both separators a path may use.
    if id.split(['/', '\\']).any(|seg| seg == "..") {
        return reject("id contains a `..` path component");
    }
    Ok(())
}

/// Whether `path` canonicalizes to a location inside one of `roots`.
///
/// Both sides are canonicalized (resolving symlinks and `..`) before the prefix
/// comparison, so a symlink or `..` that escapes a root is caught. A root that
/// does not exist on disk is skipped (a missing scan dir is legitimate — see
/// `HiggsConfig`). Returns `false` when `path` itself cannot be canonicalized
/// (e.g. it does not exist) — a non-existent resolved model path is not a valid
/// load target.
pub(crate) fn path_within_roots(path: &str, roots: &[PathBuf]) -> bool {
    let Ok(canon_path) = std::fs::canonicalize(path) else {
        return false;
    };
    roots.iter().any(|root| {
        std::fs::canonicalize(root)
            .map(|canon_root| canon_path.starts_with(&canon_root))
            .unwrap_or(false)
    })
}

/// Currently-available system RAM in bytes, via the same `sysinfo` path that
/// backs `GET /api/higgs/system`. "Available" (not merely "free") is what an
/// allocation can realistically claim, so it is the right basis for the load
/// headroom guard. A fresh `System` is sampled per call (loads are infrequent,
/// so the sampling cost is irrelevant) — no shared state to keep coherent.
fn available_system_memory() -> u64 {
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    sys.available_memory()
}

/// Whether a load needing `needed_bytes` fits within the safe RAM headroom over
/// `available_bytes` — i.e. `needed <= available * MEMORY_HEADROOM_FRACTION`
/// (ollama's `predictedForLoad <= freeMemory*80/100` placement rule). Pure
/// arithmetic, factored out of [`Higgs::load`] so the threshold is unit-testable
/// without provisioning multi-gigabyte fixtures.
pub(super) fn fits_in_memory(needed_bytes: u64, available_bytes: u64) -> bool {
    #[allow(clippy::cast_precision_loss)]
    let safe = (available_bytes as f64) * MEMORY_HEADROOM_FRACTION;
    (needed_bytes as f64) <= safe
}

/// Pre-load RAM headroom guard: refuse a load whose estimated memory need
/// (`needed_bytes`, the GGUF file size on disk — a lower-bound proxy for
/// resident weights) exceeds [`MEMORY_HEADROOM_FRACTION`] of currently-available
/// system RAM. Checked before spawning a worker so an oversized load fails fast
/// with `Err` [HG017] `InsufficientMemory` → 503 (retryable) instead of
/// OOM-killing the worker (an opaque [HG004]/[HG006]). The same `sysinfo` path
/// that backs `GET /api/higgs/system` reads available memory.
pub(crate) fn guard_memory_headroom(id: &str, needed_bytes: u64) -> Result<(), HiggsError> {
    let available = available_system_memory();
    if fits_in_memory(needed_bytes, available) {
        return Ok(());
    }
    warn!(
        id,
        needed_bytes,
        available_bytes = available,
        "higgs: refusing load — insufficient memory headroom"
    );
    Err(HiggsError::InsufficientMemory {
        id: id.to_owned(),
        needed_bytes,
        available_bytes: available,
        headroom_fraction: MEMORY_HEADROOM_FRACTION,
    })
}
