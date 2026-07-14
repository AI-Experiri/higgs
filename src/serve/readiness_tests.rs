use super::*;

/// Build inputs for an on-disk, CHAT-CAPABLE model with the given flags — the
/// ladder every case below exercises. Embedding-only models take the dedicated
/// `chat_capable: false` cases at the bottom of this file.
fn inputs(profiled: bool, stale: bool, loaded: bool, fits: bool, serving: bool) -> ReadinessInputs {
    ReadinessInputs {
        on_disk: true,
        profiled,
        stale,
        loaded,
        fits,
        serving,
        domain: crate::worker::models::ModelDomain::Llm,
    }
}

#[test]
fn loaded_wins_over_everything() {
    assert_eq!(
        derive_readiness(&inputs(true, false, true, true, true)),
        ModelReadiness::Loaded
    );
    // resident is resident even if the profile went stale or no longer fits free space
    assert_eq!(
        derive_readiness(&inputs(true, true, true, false, true)),
        ModelReadiness::Loaded
    );
    assert_eq!(
        derive_readiness(&inputs(true, false, true, false, false)),
        ModelReadiness::Loaded
    );
}

#[test]
fn unprofiled_on_disk_is_discovered() {
    assert_eq!(
        derive_readiness(&inputs(false, false, false, true, true)),
        ModelReadiness::Discovered
    );
}

#[test]
fn stale_profile_needs_retune() {
    assert_eq!(
        derive_readiness(&inputs(true, true, false, true, true)),
        ModelReadiness::NeedsRetune
    );
}

#[test]
fn profiled_fits_serving_is_servable() {
    assert_eq!(
        derive_readiness(&inputs(true, false, false, true, true)),
        ModelReadiness::Servable
    );
}

#[test]
fn profiled_no_room_is_unservable() {
    assert_eq!(
        derive_readiness(&inputs(true, false, false, false, true)),
        ModelReadiness::Unservable
    );
}

#[test]
fn profiled_serving_off_is_profiled_only() {
    // serving disabled outranks fit: you can't serve it, so it's just Profiled
    assert_eq!(
        derive_readiness(&inputs(true, false, false, true, false)),
        ModelReadiness::Profiled
    );
    assert_eq!(
        derive_readiness(&inputs(true, false, false, false, false)),
        ModelReadiness::Profiled
    );
}

#[test]
fn footprint_fits_only_against_current_free() {
    // 8 GB VRAM + 4 GB RAM footprint. Discrete-memory host (unified = false).
    let (nv, nr) = (8u64 << 30, 4u64 << 30);
    // Plenty free → fits.
    assert!(footprint_fits_free(nv, nr, 24 << 30, 32 << 30, false));
    // VRAM occupied by another resident model (only 4 GB free) → does NOT fit,
    // even though the GPU TOTAL is large. This is the codex-flagged case.
    assert!(!footprint_fits_free(nv, nr, 4 << 30, 32 << 30, false));
    // RAM pressure → does not fit.
    assert!(!footprint_fits_free(nv, nr, 24 << 30, 2 << 30, false));
    // Exact fit (needed == free) is allowed.
    assert!(footprint_fits_free(nv, nr, nv, nr, false));
    // CPU-only model (no VRAM needed) fits a 0-VRAM host.
    assert!(footprint_fits_free(0, nr, 0, 8 << 30, false));
}

#[test]
fn footprint_unified_memory_sums_pools_against_one_free() {
    // 20 GB VRAM + 20 GB RAM footprint on a 32 GB-free unified host (Apple Metal).
    let (nv, nr) = (20u64 << 30, 20u64 << 30);
    let free = 32u64 << 30;
    // Discrete check would WRONGLY pass (each pool independently <= 32 GB)…
    assert!(footprint_fits_free(nv, nr, free, free, false));
    // …but the SAME physical pool can't hold 40 GB → unified must reject it.
    assert!(
        !footprint_fits_free(nv, nr, free, free, true),
        "unified: combined 40 GB does not fit 32 GB shared free"
    );
    // A combined footprint that DOES fit the shared pool is servable.
    assert!(footprint_fits_free(10 << 30, 10 << 30, free, free, true));
    // Unified uses the CONSERVATIVE (smaller) free figure as the shared pool.
    assert!(!footprint_fits_free(
        10 << 30,
        10 << 30,
        64 << 30,
        16 << 30,
        true
    ));
}

#[test]
fn missing_on_disk_is_discovered_fallback() {
    let i = ReadinessInputs {
        on_disk: false,
        ..inputs(true, false, false, true, true)
    };
    assert_eq!(derive_readiness(&i), ModelReadiness::Discovered);
}

/// An embedding-only model is `Embedding` no matter where it sits on the chat ladder —
/// including while RESIDENT. Reporting a loaded embedding model as `Loaded` would put it
/// straight back into the model picker as a chat target, which is exactly the bug this
/// state closes; `servable_model_ids` (readiness == `Servable`) drops it either way.
#[test]
fn an_embedding_model_never_climbs_the_chat_ladder() {
    let embedding = |profiled, stale, loaded, fits, serving| ReadinessInputs {
        domain: crate::worker::models::ModelDomain::Embedding,
        ..inputs(profiled, stale, loaded, fits, serving)
    };
    // Profiled, fresh, fits, serving on — would be `Servable` if it could chat.
    assert_eq!(
        derive_readiness(&embedding(true, false, false, true, true)),
        ModelReadiness::Embedding
    );
    // Resident — `Embedding` outranks `Loaded`.
    assert_eq!(
        derive_readiness(&embedding(true, false, true, true, true)),
        ModelReadiness::Embedding
    );
    // Untuned — still `Embedding`, not `Discovered`: tuning it changes nothing.
    assert_eq!(
        derive_readiness(&embedding(false, false, false, false, true)),
        ModelReadiness::Embedding
    );
    // A reranker keeps ITS OWN terminal state — never collapsed into `Embedding`
    // (that would misreport what the model is), never `Loaded`.
    let reranker = ReadinessInputs {
        domain: crate::worker::models::ModelDomain::Reranker,
        ..inputs(true, false, true, true, true)
    };
    assert_eq!(derive_readiness(&reranker), ModelReadiness::Reranker);
    // The same flags on a chat-capable model DO reach `Servable` — proving the
    // demotion above comes from the domain and nothing else.
    assert_eq!(
        derive_readiness(&inputs(true, false, false, true, true)),
        ModelReadiness::Servable
    );
}
