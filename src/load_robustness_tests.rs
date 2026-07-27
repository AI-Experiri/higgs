use super::*;
use crate::worker::engine::llamacpp::params::LlamaCppParams;
use crate::worker::engine::GpuLayers;

fn np(gpu_layers: Option<GpuLayers>, params: Option<LlamaCppParams>) -> NodeLoadParams {
    NodeLoadParams {
        id: "org/model".into(),
        ctx_len: None,
        gpu_layers,
        threads: None,
        params,
    }
}

/// Real llama.cpp allocator-failure lines (across backends) all classify as OOM.
#[test]
fn oom_signatures_are_recognized() {
    for line in [
        "ggml_metal_graph_compute: command buffer 0 failed with status 5: out of memory",
        "llama_model_load: failed to allocate buffer",
        "CUDA error: out of memory",
        "cudaMalloc failed: out of memory",
        "ggml_backend_alloc_ctx_tensors_from_buft: failed to allocate buffer",
        "unable to allocate backend buffer",
        "MTLBuffer allocation of 12884901888 bytes failed",
        "insufficient memory to load model",
    ] {
        assert!(is_oom_reason(line), "should classify as OOM: {line:?}");
    }
}

/// Non-OOM load failures must NOT be retried (a degraded retry fails identically
/// and wastes seconds) — corrupt GGUF, unsupported arch, missing tensors.
#[test]
fn non_oom_failures_are_not_classified_oom() {
    for line in [
        "llama_model_load: error loading model: done_getting_tensors: wrong number of tensors",
        "unknown model architecture: 'frobnicate'",
        "invalid magic characters in gguf header",
        "error loading model vocab",
    ] {
        assert!(!is_oom_reason(line), "should NOT be OOM: {line:?}");
    }
}

/// Classification is case-insensitive (backends vary the casing).
#[test]
fn classification_is_case_insensitive() {
    assert!(is_oom_reason("OUT OF MEMORY"));
    assert!(is_oom_reason("Failed To Allocate Buffer"));
}

/// The ladder always offers a plain retry first (the settle wait may have freed
/// memory), then the KV-off relief. Rung 1 is byte-identical to the base.
#[test]
fn ladder_starts_with_plain_retry_then_kv_off() {
    let base = np(Some(GpuLayers::All), None);
    let ladder = oom_ladder(&base, None);
    assert_eq!(ladder[0].params, base, "rung 1 is an unmodified retry");
    assert_eq!(ladder[0].what, "");
    // Rung 2 moves the KV cache off the GPU.
    assert_eq!(
        ladder[1].params.params.as_ref().unwrap().offload_kqv,
        Some(false),
        "rung 2 sets offload_kqv=false"
    );
    assert!(ladder[1].what.contains("KV cache"));
}

/// KV-off preserves the caller's other overrides (it only flips offload_kqv).
#[test]
fn kv_off_rung_preserves_other_overrides() {
    let lc = LlamaCppParams {
        gpu_layers: GpuLayers::Count { n: 20 },
        type_k: Some(crate::worker::engine::KvCacheKind::Q8_0),
        ..Default::default()
    };
    let base = np(Some(GpuLayers::Count { n: 20 }), Some(lc));
    let ladder = oom_ladder(&base, None);
    let kv = ladder[1].params.params.as_ref().unwrap();
    assert_eq!(kv.offload_kqv, Some(false));
    assert_eq!(kv.type_k, Some(crate::worker::engine::KvCacheKind::Q8_0));
    assert_eq!(kv.gpu_layers, GpuLayers::Count { n: 20 });
}

/// A fixed GPU-layer count adds a CUMULATIVE third rung: rung 2's KV-off is
/// PRESERVED and the layers are ALSO halved (codex r4 — building rung 3 from
/// `base` would drop KV-off and never try the strongest combined config). This
/// is the fail-on-revert for the cumulative fix.
#[test]
fn count_layers_add_a_cumulative_kv_off_plus_halve_rung() {
    let base = np(Some(GpuLayers::Count { n: 40 }), None);
    let ladder = oom_ladder(&base, None);
    assert_eq!(ladder.len(), 3, "retry + kv-off + (kv-off & halve)");
    // Layers halved AND KV-off retained from rung 2.
    assert_eq!(
        ladder[2].params.gpu_layers,
        Some(GpuLayers::Count { n: 20 }),
        "layers halved"
    );
    assert_eq!(
        ladder[2].params.params.as_ref().unwrap().offload_kqv,
        Some(false),
        "rung 3 KEEPS rung 2's KV-off (cumulative)"
    );
    assert!(ladder[2].what.contains("halved") && ladder[2].what.contains("KV cache"));
}

/// The DEFAULT `All` config gets a layer-reduction rung (codex r16): with a
/// KNOWN layer total it halves; with an UNKNOWN total it drops to all-CPU
/// (`Count{0}`) — the deterministic last-resort relief. Fail-on-revert: an
/// `All` arm that returns no rung leaves the ladder at length 2.
#[test]
fn all_gets_a_layer_reduction_rung() {
    // Known total → halve.
    let ladder = oom_ladder(&np(Some(GpuLayers::All), None), Some(32));
    assert_eq!(ladder.len(), 3, "All + known count → KV-off + halve");
    assert_eq!(
        ladder[2].params.gpu_layers,
        Some(GpuLayers::Count { n: 16 })
    );
    assert_eq!(
        ladder[2].params.params.as_ref().unwrap().offload_kqv,
        Some(false),
        "rung 3 keeps KV-off (cumulative)"
    );
    // Unknown total → all-CPU last resort.
    let ladder = oom_ladder(&np(Some(GpuLayers::All), None), None);
    assert_eq!(ladder.len(), 3, "All + unknown count → KV-off + all-CPU");
    assert_eq!(ladder[2].params.gpu_layers, Some(GpuLayers::Count { n: 0 }));
    assert!(ladder[2].what.contains("all layers to the CPU"));
}

/// CPU-only / tiny fixed counts still have NO further layer rung (already minimal).
#[test]
fn cpu_only_and_tiny_counts_have_no_layer_rung() {
    // CPU-only: the KV cache is ALREADY in system memory, so the KV rung would be a
    // semantic no-op duplicate of the plain retry (Fable r6) — the ladder is JUST
    // the plain retry. Fail-on-revert: guard rung 2 by byte-equality alone and the
    // no-op KV rung reappears (len 2 with a false "moved the KV cache" label).
    let cpu_only = oom_ladder(&np(Some(GpuLayers::Count { n: 0 }), None), None);
    assert_eq!(
        cpu_only.len(),
        1,
        "CPU-only: plain retry only: {cpu_only:#?}"
    );
    // A single GPU layer: the KV rung is REAL (KV moves off the GPU), but there is
    // no meaningful layer-reduction rung.
    let one = oom_ladder(&np(Some(GpuLayers::Count { n: 1 }), None), None);
    assert_eq!(one.len(), 2, "n=1: plain retry + KV rung: {one:#?}");
    assert!(one[1].what.contains("(offload_kqv=false)"));
}

/// A base that ALREADY carries `offload_kqv = false` (a previously-degraded persisted
/// profile) gets NO separate KV rung — it would be byte-identical to the plain retry:
/// a guaranteed-OOM duplicate attempt under a lying [HG061] "moved the KV cache" line.
/// Fail-on-revert: push rung 2 unconditionally and the duplicate reappears.
#[test]
fn ladder_skips_the_kv_rung_when_already_kv_off() {
    let base = np(
        Some(GpuLayers::Count { n: 8 }),
        Some(LlamaCppParams {
            offload_kqv: Some(false),
            ..Default::default()
        }),
    );
    let ladder = oom_ladder(&base, None);
    // Plain retry stays; the would-be KV rung (== base) must NOT be duplicated.
    assert_eq!(ladder[0].params, base, "rung 1 is the plain retry");
    assert!(
        ladder.iter().skip(1).all(|r| r.params != base),
        "no later rung repeats the base params: {ladder:#?}"
    );
    assert!(
        ladder
            .iter()
            .all(|r| !r.what.contains("(offload_kqv=false)")),
        "the standalone KV rung is absent (its params would equal the base): {ladder:#?}"
    );
    // The cumulative layer-halving rung is still built (from the kv-off state).
    assert!(
        ladder.iter().any(|r| r.what.contains("layers")),
        "layer-reduction rung survives: {ladder:#?}"
    );
    // …and its label must not claim a KV move that did not happen on THIS walk
    // (the base already carried offload_kqv=false) — the [HG061] event stays honest.
    assert!(
        ladder
            .iter()
            .all(|r| !r.what.contains("moved the KV cache")),
        "no rung claims a KV move on the already-kv-off walk: {ladder:#?}"
    );
    // Sanity: a base WITHOUT kv-off still gets the KV rung.
    let fresh = oom_ladder(&np(Some(GpuLayers::Count { n: 8 }), None), None);
    assert!(
        fresh.iter().any(|r| r.what.contains("KV cache")),
        "fresh base keeps the KV rung"
    );
}
