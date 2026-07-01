//! Per-model readiness state + its pure derivation.
//!
//! `ModelReadiness` is the contract both the UI and (future) autonomous agents
//! read. It is derived from facts higgs already owns: whether a tuning profile
//! exists (`Profiled`), whether that profile is stale, whether the model is
//! resident, whether it fits free resources right now, and whether serving is
//! enabled. `Configured` (a provider points at it) is a jigglebot overlay and
//! deliberately not modelled here — higgs does not know about providers.

higgs_const_enum! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum ModelReadiness {
        /// On disk, no canonical profile yet — must Prepare.
        Discovered,
        /// Profile exists but serving is off — cannot serve right now.
        Profiled,
        /// Profiled, fits free resources, serving on — JIT may load now.
        Servable,
        /// Profiled but won't fit current free VRAM/RAM — evict, then load.
        Unservable,
        /// Profile stale (hardware or model file changed) — must Re-tune. Hard-blocks load.
        NeedsRetune,
        /// Resident in the worker, actively serving.
        Loaded,
    }
}

/// Facts the derivation consumes — gathered at the call site (control handler /
/// fleet inventory), kept here as a plain struct so the logic stays unit-pure.
#[derive(Debug, Clone, Copy)]
pub struct ReadinessInputs {
    /// The model file is present on disk (passed a scan).
    pub on_disk: bool,
    /// A canonical tuning profile (`TuneRecord`) exists for it.
    pub profiled: bool,
    /// The profile's hardware fingerprint or model-file signature no longer matches.
    pub stale: bool,
    /// The model is currently resident in a worker.
    pub loaded: bool,
    /// The profile fits current free VRAM/RAM (estimate verdict != Overflow).
    pub fits: bool,
    /// Serving is enabled on this node.
    pub serving: bool,
}

/// Does an estimated footprint fit the model into the resources free **right
/// now**? Compares the needed bytes against current free VRAM/RAM — NOT totals
/// or the tune-time budget — so `Servable` reflects what can actually load given
/// other resident models / memory pressure. Pure for unit testing.
///
/// `unified` = the host shares ONE physical pool for GPU + CPU memory (Apple
/// Metal). There, VRAM and RAM free figures are views of the SAME memory, so a
/// per-pool check would double-count it (a 20 GB-VRAM + 20 GB-RAM model would
/// "fit" 32 GB of unified free). On unified we instead require the COMBINED
/// footprint to fit the conservative shared free (`min(free_vram, free_ram)`).
pub fn footprint_fits_free(
    needed_vram: u64,
    needed_ram: u64,
    free_vram: u64,
    free_ram: u64,
    unified: bool,
) -> bool {
    if unified {
        needed_vram.saturating_add(needed_ram) <= free_vram.min(free_ram)
    } else {
        needed_vram <= free_vram && needed_ram <= free_ram
    }
}

/// Collapse the facts into one state.
///
/// Precedence: `Loaded` (resident — true regardless of staleness/fit) > not on
/// disk or not profiled → `Discovered` > stale → `NeedsRetune` > serving off →
/// `Profiled` > fits → `Servable` else `Unservable`.
pub fn derive_readiness(i: &ReadinessInputs) -> ModelReadiness {
    if i.loaded {
        return ModelReadiness::Loaded;
    }
    if !i.on_disk || !i.profiled {
        return ModelReadiness::Discovered;
    }
    if i.stale {
        return ModelReadiness::NeedsRetune;
    }
    if !i.serving {
        return ModelReadiness::Profiled;
    }
    if i.fits {
        ModelReadiness::Servable
    } else {
        ModelReadiness::Unservable
    }
}

#[cfg(test)]
#[path = "readiness_tests.rs"]
mod tests;
