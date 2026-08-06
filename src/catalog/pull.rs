//! One catalog download — the SINGLE entry both the facade op and the CLI go
//! through. Everything hard lives in [`crate::download`] already (wire
//! validation, the org/model layout, atomic `.part` writes, the hub-primary /
//! reqwest-fallback strategy); this module just binds it to the higgs models
//! dir and exposes an injectable seam for tests.

use std::path::{Path, PathBuf};

use crate::diagnostic::HiggsError;
use crate::download::{download_dual, Fetcher, PullTarget};

/// Download `repo`/`file` from the Hub into `~/.higgs/models/`, streaming
/// `(downloaded, total)` to `progress`. Returns the final on-disk path.
pub async fn pull(
    repo: &str,
    file: &str,
    progress: &mut (dyn FnMut(u64, Option<u64>) + Send),
) -> Result<PathBuf, HiggsError> {
    let root = crate::download::models_dir().map_err(|e| HiggsError::HubFileWrite {
        repo: repo.to_owned(),
        file: file.to_owned(),
        detail: format!("models dir: {e}"),
    })?;
    pull_with(
        repo,
        file,
        &root,
        &crate::hub::HubFetcher,
        &crate::download::HttpFetcher,
        progress,
    )
    .await
}

/// [`pull`] with the destination root and both fetchers injected — the test
/// seam (production always passes the hub-client primary + `reqwest`
/// fallback into the real models dir).
pub(crate) async fn pull_with<P: Fetcher, F: Fetcher>(
    repo: &str,
    file: &str,
    root: &Path,
    primary: &P,
    fallback: &F,
    progress: &mut (dyn FnMut(u64, Option<u64>) + Send),
) -> Result<PathBuf, HiggsError> {
    let target = PullTarget::new(repo, file);
    download_dual(&target, root, primary, fallback, progress).await
}

/// Minimum spacing between `Downloading` progress events — frequent enough
/// for a live bar, sparse enough that a fast local mirror can't flood the
/// broadcast channel.
pub(crate) const DOWNLOAD_EVENT_INTERVAL: std::time::Duration =
    std::time::Duration::from_millis(250);

/// Paces progress-event emission: the first report emits, then at most one
/// per `min_interval` (terminal events bypass the gate entirely). Takes `now`
/// as a parameter so the pacing is testable without sleeping.
pub(crate) struct ProgressGate {
    last_emit: Option<std::time::Instant>,
    min_interval: std::time::Duration,
}

impl ProgressGate {
    /// A gate emitting at most once per `min_interval`.
    pub(crate) fn new(min_interval: std::time::Duration) -> Self {
        Self {
            last_emit: None,
            min_interval,
        }
    }

    /// Whether a report at `now` should emit; records the emission when yes.
    pub(crate) fn should_emit(&mut self, now: std::time::Instant) -> bool {
        let due = self
            .last_emit
            .is_none_or(|last| now.duration_since(last) >= self.min_interval);
        if due {
            self.last_emit = Some(now);
        }
        due
    }
}

#[cfg(test)]
#[path = "pull_tests.rs"]
mod tests;
