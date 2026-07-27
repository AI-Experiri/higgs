//! Load robustness (G5): OOM classification + a degrade-and-retry ladder, and
//! a bounded VRAM-recovery wait — the pure decision logic wired into the load
//! path in [`api`](crate::api).
//!
//! A model load can fail for reasons a *retry with the same params* will never
//! fix (a corrupt GGUF) and reasons a *degraded retry* often will (the machine
//! is momentarily VRAM-tight: another process holds memory, or the just-unloaded
//! model's VRAM hasn't been reclaimed yet). higgs must not hard-fail the second
//! kind when a cheaper load would succeed — but it must ALSO not silently serve
//! a worse configuration without saying so. So each rung emits a coded event and
//! the final failure carries an aggregate diagnosis.
//!
//! Everything here is a PURE function of its inputs (an error string, the load
//! params, a VRAM reading) so it is exhaustively unit-tested; the `api` wiring
//! only supplies the real error/reader and applies the returned params.
//!
//! # Scope note (stall-based load timeout — DEFERRED)
//!
//! G5's third goal, a *stall-based* load timeout (reset a deadline whenever the
//! load makes progress), is DEFERRED with evidence: the worker emits NO progress
//! during a load — `M_LOAD` is one blocking RPC (`node/runtime.rs` `sup.request(
//! M_LOAD, …)`), and `src/worker/` sends nothing mid-load. A true progress-reset
//! timeout therefore requires a llama.cpp load *progress callback* surfaced from
//! the worker over a new notification — an FFI + worker-protocol change (fork /
//! G3 territory), not a backend-only change. Two mitigations already exist, so
//! the residual risk is small: a **dead** worker fast-fails immediately via the
//! reply demux (`supervisor.rs` — the `Ok(Err(_))` "worker died before response"
//! arm, no 120s wait), and a load that OOMs is rescued by the ladder here. Only
//! a *wedged-but-alive* worker still waits out the fixed `CONTROL_RPC_TIMEOUT`;
//! bumping that constant for large models is an untestable hot-path change (the
//! ~1 MB test GGUF loads instantly, so no fail-on-revert test is possible) and
//! is intentionally not shipped without one.

use crate::remote::NodeLoadParams;
use crate::worker::engine::GpuLayers;

/// Does this `[HG004]` failure reason look like an out-of-memory / allocation
/// failure (as opposed to a corrupt-GGUF / unsupported-arch failure)? The reason
/// is the joined llama.cpp ENGINE ERROR lines (see
/// `llamacpp::logging::take_engine_diagnostics`), so we match the allocator
/// signatures every backend emits. Case-insensitive; ANY match ⇒ OOM.
///
/// Only an OOM-classified failure is eligible for the degrade ladder — a
/// non-OOM load error is returned to the caller immediately (a retry would fail
/// identically and just waste seconds).
pub fn is_oom_reason(reason: &str) -> bool {
    // Lowercase once; match backend-agnostic allocator failure phrases.
    let r = reason.to_ascii_lowercase();
    const SIGNATURES: &[&str] = &[
        "out of memory",
        "failed to allocate",
        "unable to allocate",
        "cannot allocate",
        "insufficient memory",
        "not enough memory",
        "cudamalloc failed", // CUDA
        "cuda error: out of memory",
        "mtlbuffer",          // Metal buffer alloc failure line
        "ggml_backend_alloc", // ctx-tensor / buffer alloc helpers
        "buffer allocation failed",
        "alloc_tensor_range",
    ];
    SIGNATURES.iter().any(|s| r.contains(s))
}

/// One rung of the OOM degrade ladder — a cheaper load to try next, and the
/// human phrase that names the degradation in the coded `[HG060]` event.
#[derive(Debug, Clone, PartialEq)]
pub struct LoadRung {
    /// The load params for this attempt (a degraded clone of the original).
    pub params: NodeLoadParams,
    /// What was degraded, for the event/log line (empty for the plain re-try).
    pub what: &'static str,
}

/// Build the OOM degrade ladder for `base`: the ordered list of *retry* params
/// to try after the initial attempt OOMs, cheapest-relief-first. Bounded and
/// deterministic (no unbounded gpu-layer halving); an empty ladder (or exhausting
/// it) means "give up with an aggregate diagnosis".
///
/// Rungs, in order:
/// 1. **settle + same params** — a transient allocation (the just-unloaded
///    model, a peer process) may have cleared; a plain retry after the VRAM
///    settle costs one load and often wins. (`what == ""`.)
/// 2. **KV cache to system memory** (`offload_kqv = false`) — moves the KV
///    cache off the GPU, the single biggest VRAM lever that keeps all layers
///    resident.
/// 3. **KV-off AND fewer GPU layers** — rung 2's KV-off PLUS moving transformer
///    layers off the GPU. For a fixed `Count{n}` this halves to `n/2`; for the
///    DEFAULT `All` it halves the model's known layer total (`layer_count`), or
///    drops to ALL-CPU (`Count{0}`) as the last-resort maximal VRAM relief when
///    the total is unknown — so a full-GPU load still gets a layer-reduction
///    rung (codex r16). Skipped only when already CPU-only / a 0-1 count.
///
/// `layer_count` is the model's transformer block count (GGUF header), used to
/// turn `All` into a concrete half-offload when known.
pub fn oom_ladder(base: &NodeLoadParams, layer_count: Option<u32>) -> Vec<LoadRung> {
    let mut rungs = Vec::new();

    // Rung 1: plain retry (the settle wait happens in the caller before each rung).
    rungs.push(LoadRung {
        params: base.clone(),
        what: "",
    });

    // Rung 2: KV cache off the GPU — but ONLY when that actually changes anything:
    // - a previously-degraded persisted profile already carries `offload_kqv=false`
    //   (byte-identical → skip), and
    // - a CPU-ONLY load (`gpu_layers == Count{0}`) keeps its KV cache in system
    //   memory regardless, so `offload_kqv=false` changes bytes but not semantics
    //   (Fable r6). Either way the rung would be a guaranteed-OOM duplicate of the
    //   plain retry (plus a settle sleep) under an [HG061] line claiming the KV
    //   cache moved when nothing changed — and, on a lucky success, a
    //   `degraded=true` persist of a functionally unchanged config.
    let cpu_only = matches!(base.gpu_layers, Some(GpuLayers::Count { n: 0 }));
    let kv_off = with_kv_off(base);
    let kv_moved = !cpu_only && kv_off != *base;
    if kv_moved {
        rungs.push(LoadRung {
            params: kv_off.clone(),
            what: "moved the KV cache to system memory (offload_kqv=false)",
        });
    }

    // Rung 3: KEEP rung 2's KV-off and ALSO reduce GPU layers. Degradations are
    // CUMULATIVE, so this is built from `kv_off` (not `base`) — otherwise the
    // strongest combined config would never be tried and a load needing both
    // would fail HG060 avoidably (codex r4).
    if let Some((combined, what)) = reduce_gpu_layers(&kv_off, layer_count, kv_moved) {
        rungs.push(LoadRung {
            params: combined,
            what,
        });
    }

    rungs
}

/// A clone of `p` with the KV cache forced off the GPU (`offload_kqv = false`),
/// preserving every other override.
fn with_kv_off(p: &NodeLoadParams) -> NodeLoadParams {
    let mut out = p.clone();
    out.params = Some({
        let mut lc = out.params.clone().unwrap_or_default();
        lc.offload_kqv = Some(false);
        lc
    });
    out
}

/// A `NodeLoadParams` with fewer GPU layers + the phrase naming the change, or
/// `None` when there is nothing meaningful to reduce (already CPU-only, or a
/// 0-1 fixed count). `All` uses the model's `layer_count` to halve when known,
/// else drops to ALL-CPU (`Count{0}`) — the maximal relief that always applies.
fn reduce_gpu_layers(
    base: &NodeLoadParams,
    layer_count: Option<u32>,
    kv_moved: bool,
) -> Option<(NodeLoadParams, &'static str)> {
    let current = base
        .gpu_layers
        .or_else(|| base.params.as_ref().map(|p| p.gpu_layers));
    // The phrase must name what THIS walk actually degraded: when the base config
    // already carried `offload_kqv=false` (a reloaded previously-degraded profile),
    // the KV cache did not move on this walk — claiming it did makes the coded
    // [HG061] event lie, the same defect class as the rung-2 skip above.
    let halved = if kv_moved {
        "moved the KV cache to system memory AND halved the GPU-offloaded layers"
    } else {
        "halved the GPU-offloaded layers (KV cache already in system memory)"
    };
    let all_cpu = if kv_moved {
        "moved the KV cache to system memory AND moved all layers to the CPU"
    } else {
        "moved all layers to the CPU (KV cache already in system memory)"
    };
    let (target, what) = match current {
        Some(GpuLayers::Count { n }) if n >= 2 => (GpuLayers::Count { n: n / 2 }, halved),
        Some(GpuLayers::All) => match layer_count {
            Some(total) if total >= 2 => (GpuLayers::Count { n: total / 2 }, halved),
            // Unknown (or tiny) total: all-CPU is the deterministic last resort.
            _ => (GpuLayers::Count { n: 0 }, all_cpu),
        },
        // A 0-1 fixed count: already minimal.
        _ => return None,
    };
    let mut out = base.clone();
    out.gpu_layers = Some(target);
    Some((out, what))
}

/// A bounded pause before each OOM-retry rung: a GPU driver / allocator often
/// needs a beat to reclaim the just-failed (or just-unloaded) allocation before
/// the next attempt can fit. This is the practical "VRAM-recovery wait" — a
/// TIME settle, cheap and worker-free. (A poll-until-free variant was considered
/// and DEFERRED: a fresh VRAM reading spawns a transient sysinfo worker per poll
/// — see `Higgs::sysinfo` — and the common unified-memory path frees lazily, so
/// polling would add real load-path latency for little gain. The `api` wiring
/// passes this as a parameter so tests inject `Duration::ZERO`.)
pub const SETTLE_BEFORE_RETRY: std::time::Duration = std::time::Duration::from_millis(750);

#[cfg(test)]
#[path = "load_robustness_tests.rs"]
mod tests;
