
use super::*;
use crate::system::{DeviceKind, GpuDevice};

/// A hardware snapshot with `vram` bytes of GPU VRAM, 64 GiB RAM, 16 cores.
fn hw_gpu(vram: u64) -> HardwareInfo {
    HardwareInfo {
        cpu_name: "test".into(),
        arch: "aarch64".into(),
        cpu_cores: 16,
        ram_total_bytes: 64u64 << 30,
        ram_used_bytes: 8u64 << 30,
        cpu_usage_percent: 5.0,
        gpus: vec![GpuDevice {
            name: "Metal".into(),
            description: "test gpu".into(),
            kind: DeviceKind::Gpu,
            vram_total_bytes: vram,
            vram_free_bytes: vram,
        }],
        vram_total_bytes: vram,
    }
}

fn dense_meta_8gb() -> ModelMeta {
    ModelMeta {
        id: "org/m".into(),
        size_bytes: 8u64 << 30,
        block_count: Some(32),
        head_count: Some(32),
        head_count_kv: Some(8),
        embedding_length: Some(4096),
        expert_count: Some(0),
        ..Default::default()
    }
}

fn all_gpu_load() -> LlamaCppParams {
    LlamaCppParams {
        ctx_len: 8192,
        gpu_layers: u32::MAX,
        type_k: Some(KvCacheKind::F16),
        type_v: Some(KvCacheKind::F16),
        ..Default::default()
    }
}

#[test]
fn vram_estimate_uses_gqa_kv_and_tiers() {
    let meta = dense_meta_8gb();
    let load = all_gpu_load();
    let hw = hw_gpu(24u64 << 30);
    let v = StaticVramEstimator.estimate(&load, &meta, &hw, &ResourceBudget::default());
    // weights (8 GiB) + GQA KV (~1 GiB) + overhead — between 8 and 24 GiB.
    assert!(v.needed_bytes > (8u64 << 30), "needed {} ", v.needed_bytes);
    assert!(v.needed_bytes < (24u64 << 30));
    assert_eq!(v.verdict, FitVerdict::Fits);

    // A 6 GiB VRAM cap overflows the same model.
    let capped = ResourceBudget {
        max_vram_bytes: Some(6u64 << 30),
        ..Default::default()
    };
    assert_eq!(
        StaticVramEstimator
            .estimate(&load, &meta, &hw, &capped)
            .verdict,
        FitVerdict::Overflow
    );
}

#[test]
fn kv_estimate_is_gqa_correct_not_query_heads() {
    // Same model, but if we (wrongly) used the query head count (32) the KV term
    // would be 4× larger. Pin that head_count_kv (8) drives the estimate.
    let hw = hw_gpu(24u64 << 30);
    let load = all_gpu_load();
    let gqa =
        StaticVramEstimator.estimate(&load, &dense_meta_8gb(), &hw, &ResourceBudget::default());

    let mut no_gqa = dense_meta_8gb();
    no_gqa.head_count_kv = Some(32); // pretend MHA
    let mha = StaticVramEstimator.estimate(&load, &no_gqa, &hw, &ResourceBudget::default());

    // The MHA KV is ~4× the GQA KV, so the MHA need is meaningfully larger.
    assert!(
        mha.needed_bytes > gqa.needed_bytes + (2u64 << 30),
        "GQA {} vs MHA {}",
        gqa.needed_bytes,
        mha.needed_bytes
    );
}

#[test]
fn gpu_layers_within_budget_backs_off() {
    let meta = dense_meta_8gb();
    let hw = hw_gpu(24u64 << 30);
    let load = all_gpu_load();
    // A 0 cap → CPU-only (0 layers).
    let zero = ResourceBudget {
        max_vram_bytes: Some(0),
        ..Default::default()
    };
    assert_eq!(gpu_layers_within_budget(&load, &meta, &hw, &zero), 0);
    // A huge cap → all layers.
    let big = ResourceBudget {
        max_vram_bytes: Some(100u64 << 30),
        ..Default::default()
    };
    assert_eq!(gpu_layers_within_budget(&load, &meta, &hw, &big), u32::MAX);
    // A small (4 GiB) cap can't hold the 8 GiB weights → a PARTIAL offload.
    let small = ResourceBudget {
        max_vram_bytes: Some(4u64 << 30),
        ..Default::default()
    };
    let n = gpu_layers_within_budget(&load, &meta, &hw, &small);
    assert!(n > 0 && n < 32, "partial offload within the cap: {n}");
}

#[test]
fn cpu_only_host_vram_fits_not_overflow() {
    // No GPU + gpu_layers 0 → nothing lives in VRAM (needed == 0), so the VRAM
    // verdict must read Fits, NOT a misleading Overflow (basis is also 0).
    let cpu_load = LlamaCppParams {
        ctx_len: 8192,
        gpu_layers: 0,
        ..Default::default()
    };
    let v = StaticVramEstimator.estimate(
        &cpu_load,
        &dense_meta_8gb(),
        &hw_gpu(0),
        &ResourceBudget::default(),
    );
    assert_eq!(v.needed_bytes, 0, "CPU-only load uses no VRAM");
    assert_eq!(v.verdict, FitVerdict::Fits);
}

#[test]
fn ram_estimate_charges_cpu_when_not_offloaded() {
    let meta = dense_meta_8gb();
    let hw = hw_gpu(24u64 << 30);
    // All on GPU → CPU RAM need is just overhead-ish (well under 64 GiB).
    let on_gpu =
        StaticRamEstimator.estimate(&all_gpu_load(), &meta, &hw, &ResourceBudget::default());
    assert_eq!(on_gpu.verdict, FitVerdict::Fits);
    assert!(on_gpu.needed_bytes < (2u64 << 30));

    // gpu_layers = 0 (CPU-only) → all 8 GiB of weights land in RAM.
    let cpu_load = LlamaCppParams {
        ctx_len: 8192,
        gpu_layers: 0,
        ..Default::default()
    };
    let on_cpu = StaticRamEstimator.estimate(&cpu_load, &meta, &hw, &ResourceBudget::default());
    assert!(
        on_cpu.needed_bytes > (8u64 << 30),
        "needed {}",
        on_cpu.needed_bytes
    );
}

#[test]
fn kv_cache_bytes_saturates_on_absurd_metadata() {
    // A corrupt/hostile GGUF header with absurd dims must not overflow the u64
    // product (a debug panic / release garbage) — the estimate saturates instead.
    let meta = ModelMeta {
        id: "org/evil".into(),
        block_count: Some(u32::MAX),
        head_count: Some(u32::MAX),
        head_count_kv: Some(u32::MAX),
        embedding_length: Some(u32::MAX),
        ctx_train: Some(u64::MAX),
        ..Default::default()
    };
    let load = LlamaCppParams {
        ctx_len: 0, // → ctx from ctx_train (u64::MAX)
        ..Default::default()
    };
    // Must NOT panic; saturates to a huge (finite) estimate.
    assert!(kv_cache_bytes(&load, &meta) > 0);
}
