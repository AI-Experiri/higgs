
use super::*;
use crate::system::{DeviceKind, GpuDevice};
use crate::tune::vram::{StaticRamEstimator, StaticVramEstimator};
use crate::tune::{FitVerdict, Suggester};

fn hw(vram: u64, ram: u64, cores: u32) -> HardwareInfo {
    HardwareInfo {
        cpu_name: "t".into(),
        arch: "aarch64".into(),
        cpu_cores: cores,
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

fn dense_meta() -> ModelMeta {
    ModelMeta {
        id: "org/m".into(),
        size_bytes: 4u64 << 30,
        block_count: Some(32),
        head_count: Some(32),
        head_count_kv: Some(8),
        embedding_length: Some(4096),
        ctx_train: Some(32768),
        expert_count: Some(0),
        ..Default::default()
    }
}

#[test]
fn derive_defaults_all_gpu_flash_on_and_threads() {
    let d = derive_default(
        &dense_meta(),
        &hw(24u64 << 30, 64u64 << 30, 16),
        &ResourceBudget::default(),
    );
    assert_eq!(d.gpu_layers, u32::MAX, "all layers on GPU");
    assert_eq!(d.flash_attn, Some(FlashAttn::On));
    assert_eq!(d.threads, 8, "floor(16/2)");
    assert_eq!(d.ctx_len, 8192, "ctx_train 32768 capped to 8192");
    assert_eq!(d.type_k, Some(KvCacheKind::F16));
    assert_eq!(d.n_seq_max, Some(1));
}

#[test]
fn cpu_thread_cap_clamps() {
    let capped = ResourceBudget {
        max_cpu_threads: Some(4),
        ..Default::default()
    };
    let d = derive_default(&dense_meta(), &hw(24u64 << 30, 64u64 << 30, 16), &capped);
    assert_eq!(d.threads, 4);
    assert_eq!(d.n_threads_batch, Some(4));
}

#[test]
fn no_gpu_derives_cpu_only() {
    let d = derive_default(
        &dense_meta(),
        &hw(0, 32u64 << 30, 8),
        &ResourceBudget::default(),
    );
    assert_eq!(d.gpu_layers, 0, "no GPU → CPU-only");
    assert_eq!(d.threads, 4, "floor(8/2)");
}

/// MoE back-off: an MoE model that overflows VRAM gets `cpu_moe = true` when
/// RAM still fits (exercised through the full `Suggester::suggest`).
#[test]
fn moe_backoff_sets_cpu_moe_when_ram_fits() {
    // A 20 GiB MoE that overflows an 8 GiB GPU but fits 128 GiB RAM.
    let meta = ModelMeta {
        id: "org/moe".into(),
        size_bytes: 20u64 << 30,
        block_count: Some(48),
        head_count: Some(32),
        head_count_kv: Some(8),
        embedding_length: Some(4096),
        ctx_train: Some(8192),
        expert_count: Some(8),
        ..Default::default()
    };
    let s = Suggester {
        derive: HeuristicStrategy,
        vram: StaticVramEstimator,
        ram: StaticRamEstimator,
        sampling: crate::tune::card_sampling::EmptySamplingSource,
    };
    let sugg = s.suggest(
        &meta,
        &hw(8u64 << 30, 128u64 << 30, 16),
        &ResourceBudget::default(),
    );
    assert_eq!(
        sugg.load.as_llamacpp().cpu_moe,
        Some(true),
        "MoE overflow + RAM fits → cpu_moe; rationale: {:?}",
        sugg.rationale
    );
    // And the RAM verdict is not overflow after the back-off.
    assert_ne!(sugg.ram_fit.verdict, FitVerdict::Overflow);
}
