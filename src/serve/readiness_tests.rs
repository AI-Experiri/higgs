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
fn missing_on_disk_is_discovered_fallback() {
    let i = ReadinessInputs {
        on_disk: false,
        ..inputs(true, false, false, true, true)
    };
    assert_eq!(derive_readiness(&i), ModelReadiness::Discovered);
}
