//! The engine-level idle auto-unload reaper. Split out of `api.rs`.

use std::sync::Arc;
use tracing::warn;

use super::{Higgs, IDLE_REAP_INTERVAL, MAX_CONCURRENT_INFERENCE};

/// Idle reaper: every [`IDLE_REAP_INTERVAL`], unload the loaded model once the
/// time since the last chat exceeds the runtime idle TTL (ollama `keep_alive`).
///
/// The TTL and the on/off switch are read from the live atoms
/// ([`Higgs::auto_unload_idle`], [`Higgs::idle_ttl_minutes`]) on EVERY tick, so a
/// Server-Settings change takes effect without a restart: when auto-unload is
/// off the reaper skips entirely, and the TTL is `idle_ttl_minutes` minutes
/// (seeded from [`IDLE_UNLOAD_TTL`]). A per-load override
/// ([`Higgs::loaded_idle_ttl_override`], set at load time and cleared on unload)
/// takes precedence over the global TTL for the currently-loaded model.
///
/// Holds a `Weak<Higgs>` so it terminates when the host drops its `Arc<Higgs>`.
/// It never unloads mid-generation and never races a just-admitted chat: the
/// reaper atomically acquires ALL [`MAX_CONCURRENT_INFERENCE`] inference permits
/// before unloading. Holding every permit proves zero in-flight requests AND
/// blocks any new `chat_stream` admission until the unload finishes and the
/// permits drop. An in-flight request also re-stamps `last_activity`, so a long
/// generation keeps the model resident regardless.
/// The idle [`Instant`] is copied out from under the `parking_lot` guard before
/// any `.await`, honoring the never-hold-a-lock-across-await rule; the unload
/// itself runs through the existing [`Higgs::unload`] path (which takes the
/// lifecycle mutex), so it serializes correctly against a concurrent load.
pub(super) async fn idle_reaper(weak: std::sync::Weak<Higgs>) {
    loop {
        tokio::time::sleep(IDLE_REAP_INTERVAL).await;
        // Upgrade per tick: a failed upgrade means the host dropped Higgs — exit.
        let Some(higgs) = weak.upgrade() else {
            return;
        };
        // Auto-unload disabled at runtime → never reap. Read each tick so a live
        // toggle takes effect without a restart.
        if !higgs.auto_unload_idle() {
            continue;
        }
        // Read the effective TTL each tick (minutes → Duration). A per-load
        // override (set at load time, cleared on unload) wins over the global
        // runtime TTL for the currently-loaded model; otherwise the global TTL
        // (seeded from IDLE_UNLOAD_TTL) applies. Read each tick so a live change
        // to either takes effect immediately.
        let mins = higgs
            .loaded_idle_ttl_override()
            .unwrap_or_else(|| higgs.idle_ttl_minutes());
        let ttl = std::time::Duration::from_secs(mins * 60);
        // Copy the idle instant out under the lock, then drop the guard before
        // any await (never hold a parking_lot lock across .await).
        let idle_for = {
            let last = *higgs.last_activity.lock();
            last.elapsed()
        };
        if idle_for < ttl {
            continue;
        }
        // Don't unload while any inference is in flight, and don't let a new
        // request slip in mid-unload (TOCTOU). Acquiring ALL permits is atomic
        // proof of both: success means zero in-flight AND — because the reaper
        // now holds every permit — no `chat_stream` can `try_acquire_owned`
        // until we drop them after the unload completes. A failure means a
        // generation is running (it holds a permit); skip this tick. (A running
        // request also re-stamps `last_activity`, so a long generation keeps the
        // model resident regardless.)
        let Ok(_all_permits) = Arc::clone(&higgs.inference_gate)
            .try_acquire_many_owned(MAX_CONCURRENT_INFERENCE as u32)
        else {
            continue;
        };
        // Only unload if a model is actually loaded — otherwise the unload path
        // would needlessly kill an idle (modelless) worker or no-op. A dead/
        // empty worker reports `loaded: None`. The held permits gate out any new
        // chat for the duration of this status+unload.
        match higgs.status().await {
            Ok(st) if st.loaded.is_some() => {
                warn!(
                    idle_secs = idle_for.as_secs(),
                    "higgs: auto-unloading idle model (keep_alive TTL exceeded)"
                );
                if let Err(e) = higgs.unload().await {
                    warn!(error = %e, "higgs: idle auto-unload failed");
                }
            }
            _ => { /* nothing loaded, or status unavailable — nothing to reap */ }
        }
        // `_all_permits` drops here, reopening the gate for new chats.
        // Drop the strong ref before sleeping so we never pin Higgs alive.
        drop(higgs);
    }
}
