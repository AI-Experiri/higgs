use super::*;

/// Build inputs for an on-disk model with the given flags.
fn inputs(profiled: bool, stale: bool, loaded: bool, fits: bool, serving: bool) -> ReadinessInputs {
    ReadinessInputs {
        on_disk: true,
        profiled,
        stale,
        loaded,
        fits,
        serving,
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
