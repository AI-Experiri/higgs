use super::*;
use crate::system::{DeviceKind, GpuDevice, HardwareInfo};
use crate::tune::context::ContextEstimator;
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
        gpus: vec![GpuDevice {
            name: "Metal".into(),
            description: "g".into(),
            kind: DeviceKind::Gpu,
            vram_total_bytes: vram,
            vram_free_bytes: vram,
        }],
        vram_total_bytes: vram,
    }
}

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

fn all_gpu() -> LlamaCppParams {
    LlamaCppParams {
        ctx_len: CtxLen::Fixed { n: 1 },
        gpu_layers: GpuLayers::All,
        type_k: Some(KvCacheKind::F16),
        type_v: Some(KvCacheKind::F16),
        ..Default::default()
    }
}

/// A fixed-output estimator, for asserting the averaging arithmetic in isolation.
struct Fixed(u32);
impl ContextEstimator for Fixed {
    fn max_ctx_for_budget(
        &self,
        _load: &LlamaCppParams,
        _meta: &ModelMeta,
        _hw: &HardwareInfo,
        _vram_budget: u64,
        _ram_budget: u64,
    ) -> u32 {
        self.0
    }
}

#[test]
fn analytical_only_matches_the_single_estimator() {
    let m = dense_meta();
    let h = hw(24u64 << 30, 64u64 << 30);
    let load = all_gpu();
    let avg =
        AverageStrategy::analytical_only().derive_ctx(&load, &m, &h, 24u64 << 30, 64u64 << 30);
    let direct = Analytical.max_ctx_for_budget(&load, &m, &h, 24u64 << 30, 64u64 << 30);
    assert_eq!(avg.ctx, direct);
    assert_eq!(avg.min, direct);
    assert_eq!(avg.max, direct);
    assert_eq!(avg.methods, 1);
}

#[test]
fn averages_multiple_estimators_and_reports_spread() {
    let m = dense_meta();
    let h = hw(24u64 << 30, 64u64 << 30);
    let strat = AverageStrategy::new(vec![Box::new(Fixed(1000)), Box::new(Fixed(3000))]);
    let d = strat.derive_ctx(&all_gpu(), &m, &h, 0, 0);
    assert_eq!(d.ctx, 2000, "mean of 1000 and 3000");
    assert_eq!(d.min, 1000);
    assert_eq!(d.max, 3000);
    assert_eq!(d.methods, 2);
}

#[test]
fn averaging_a_u32_max_method_does_not_overflow() {
    let m = dense_meta();
    let h = hw(24u64 << 30, 64u64 << 30);
    let strat = AverageStrategy::new(vec![Box::new(Fixed(u32::MAX)), Box::new(Fixed(u32::MAX))]);
    let d = strat.derive_ctx(&all_gpu(), &m, &h, 0, 0);
    assert_eq!(
        d.ctx,
        u32::MAX,
        "mean of two u32::MAX is u32::MAX, not a wrap"
    );
}

#[test]
fn empty_ensemble_is_zero() {
    let m = dense_meta();
    let h = hw(24u64 << 30, 64u64 << 30);
    let d = AverageStrategy::new(vec![]).derive_ctx(&all_gpu(), &m, &h, 1u64 << 30, 1u64 << 30);
    assert_eq!(
        d,
        CtxDerivation {
            ctx: 0,
            min: 0,
            max: 0,
            methods: 0
        }
    );
}
