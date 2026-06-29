use super::*;
use crate::system::{DeviceKind, GpuDevice, HardwareInfo};
use crate::tune::ModelMeta;
use crate::worker::engine::llamacpp::params::LlamaCppParams;
use crate::worker::engine::{CtxLen, GpuLayers, KvCacheKind};

fn hw(vram: u64, ram: u64) -> HardwareInfo {
    HardwareInfo {
        cpu_name: "t".into(),
        arch: "aarch64".into(),
        cpu_cores: 16,
        ram_total_bytes: ram,
        ram_used_bytes: ram / 8,
        cpu_usage_percent: 1.0,
        gpus: if vram > 0 {
            vec![GpuDevice {
                name: "Metal".into(),
                description: "g".into(),
                kind: DeviceKind::Gpu,
                vram_total_bytes: vram,
                vram_free_bytes: vram,
            }]
        } else {
            vec![]
        },
        vram_total_bytes: vram,
    }
}

/// 8 GiB dense model: 32 layers, GQA 8 KV heads, head_dim 128, trained to 128k.
fn dense_meta() -> ModelMeta {
    ModelMeta {
        id: "org/m".into(),
        size_bytes: 8u64 << 30,
        block_count: Some(32),
        head_count: Some(32),
        head_count_kv: Some(8),
        embedding_length: Some(4096),
        expert_count: Some(0),
        ctx_train: Some(131_072),
        ..Default::default()
    }
}

fn load_with(gpu_layers: GpuLayers) -> LlamaCppParams {
    LlamaCppParams {
        ctx_len: CtxLen::Fixed { n: 1 }, // placeholder — the inversion drives it
        gpu_layers,
        type_k: Some(KvCacheKind::F16),
        type_v: Some(KvCacheKind::F16),
        ..Default::default()
    }
}

#[test]
fn all_gpu_load_is_vram_bound_not_ram_bound() {
    let m = dense_meta();
    let load = load_with(GpuLayers::All);
    // Modest VRAM, generous RAM → VRAM binds. Growing RAM must not change the answer.
    let a = Analytical.max_ctx_for_budget(
        &load,
        &m,
        &hw(24u64 << 30, 64u64 << 30),
        24u64 << 30,
        64u64 << 30,
    );
    let b = Analytical.max_ctx_for_budget(
        &load,
        &m,
        &hw(24u64 << 30, 256u64 << 30),
        24u64 << 30,
        256u64 << 30,
    );
    assert!(a > 0, "a concrete context");
    assert_eq!(
        a, b,
        "an all-GPU context is VRAM-bound; RAM size is irrelevant"
    );
}

#[test]
fn cpu_only_load_is_ram_bound_not_vram_bound() {
    let m = dense_meta();
    let load = load_with(GpuLayers::Count { n: 0 });
    // A CPU-only load uses NO VRAM, so the VRAM budget must not constrain it; RAM binds.
    let small_vram = Analytical.max_ctx_for_budget(
        &load,
        &m,
        &hw(1u64 << 30, 128u64 << 30),
        1u64 << 30,
        128u64 << 30,
    );
    let big_vram = Analytical.max_ctx_for_budget(
        &load,
        &m,
        &hw(48u64 << 30, 128u64 << 30),
        48u64 << 30,
        128u64 << 30,
    );
    assert!(small_vram > 0, "RAM affords a context even with ~no VRAM");
    assert_eq!(
        small_vram, big_vram,
        "a CPU-only context is RAM-bound; VRAM size is irrelevant"
    );
}

#[test]
fn partial_offload_affords_more_than_all_gpu_under_tight_vram() {
    // A VRAM budget too small to hold all-GPU weights → all-GPU derives ~0 context,
    // but a partial offload (fewer GPU layers, the rest of the KV in ample RAM) fits a
    // much larger window. This is exactly the partial-offload case the binary
    // VRAM/RAM split got wrong.
    let m = dense_meta();
    let h = hw(8u64 << 30, 256u64 << 30);
    let all =
        Analytical.max_ctx_for_budget(&load_with(GpuLayers::All), &m, &h, 8u64 << 30, 256u64 << 30);
    let partial = Analytical.max_ctx_for_budget(
        &load_with(GpuLayers::Count { n: 8 }),
        &m,
        &h,
        8u64 << 30,
        256u64 << 30,
    );
    assert!(
        partial > all,
        "partial offload (less GPU-resident weight/KV) affords more context: partial {partial} vs all {all}"
    );
}

#[test]
fn bigger_budget_affords_more_context() {
    let m = dense_meta();
    let load = load_with(GpuLayers::All);
    assert!(
        Analytical.max_ctx_for_budget(
            &load,
            &m,
            &hw(48u64 << 30, 256u64 << 30),
            48u64 << 30,
            256u64 << 30
        ) > Analytical.max_ctx_for_budget(
            &load,
            &m,
            &hw(16u64 << 30, 256u64 << 30),
            16u64 << 30,
            256u64 << 30
        ),
        "a larger VRAM budget derives a larger context"
    );
}

#[test]
fn zero_slope_pool_overflowing_its_budget_blocks_context() {
    // An all-GPU load charges NO per-token RAM (RAM slope 0) but DOES charge a fixed
    // RAM overhead. A RAM budget below that overhead must yield 0 context — not be
    // ignored in favour of the (large) VRAM-derived context. Ample VRAM here, so only
    // the overflowing zero-slope RAM pool can force the result to 0.
    let m = dense_meta();
    let load = load_with(GpuLayers::All);
    let n = Analytical.max_ctx_for_budget(
        &load,
        &m,
        &hw(48u64 << 30, 64u64 << 30),
        48u64 << 30,
        1u64 << 20, // RAM budget far below the fixed overhead
    );
    assert_eq!(
        n, 0,
        "a RAM budget below the fixed overhead blocks all context, got {n}"
    );
}

#[test]
fn tiny_budget_yields_zero_context() {
    // Below the fixed weights+overhead on BOTH pools → no context fits (caller clamps).
    let m = dense_meta();
    let load = load_with(GpuLayers::All);
    assert_eq!(
        Analytical.max_ctx_for_budget(
            &load,
            &m,
            &hw(1u64 << 30, 1u64 << 30),
            1u64 << 20,
            1u64 << 20
        ),
        0
    );
}
