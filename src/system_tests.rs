use super::*;

/// A scripted CPU device — the shape `FakeEngine::devices` and these tests
/// use so the VRAM-sum and fit logic run without real FFI.
fn cpu_device() -> GpuDevice {
    GpuDevice {
        name: "CPU".into(),
        description: "test cpu".into(),
        kind: DeviceKind::Cpu,
        vram_total_bytes: 0,
        vram_free_bytes: 0,
    }
}

// `Higgs::new` constructs the co-located node (which spawns its actor task), so
// these need a Tokio runtime even though the bodies are otherwise synchronous.
#[tokio::test]
async fn vram_total_sums_only_gpu_devices() {
    let cfg = crate::api::Higgs::new(crate::HiggsConfig::default()).server_config();
    let gpus = vec![
        cpu_device(),
        GpuDevice {
            name: "Metal".into(),
            description: "Apple GPU".into(),
            kind: DeviceKind::Gpu,
            vram_total_bytes: 16_000_000_000,
            vram_free_bytes: 8_000_000_000,
        },
    ];
    let info = SystemInfo::gather(cfg, gpus);
    // CPU device's memory must NOT contribute to the VRAM total.
    assert_eq!(info.hardware.vram_total_bytes, 16_000_000_000);
    assert_eq!(info.hardware.gpus.len(), 2);
}

#[test]
fn fits_vram_fits_wont_fit_and_no_gpu() {
    // Fits: 8 GB model under 0.8 * 16 GB = 12.8 GB budget.
    let a = fits_vram(8_000_000_000, 16_000_000_000, 0.8);
    assert!(a.fits);
    assert_eq!(a.needed_bytes, 8_000_000_000);
    assert_eq!(a.available_bytes, 12_800_000_000);

    // Won't fit: 14 GB model over the 12.8 GB budget.
    assert!(!fits_vram(14_000_000_000, 16_000_000_000, 0.8).fits);

    // No GPU (vram_total 0): budget is 0, so nothing fits — caller falls
    // back to the system-RAM headroom guard.
    let none = fits_vram(1, 0, 0.8);
    assert!(!none.fits);
    assert_eq!(none.available_bytes, 0);
}

#[tokio::test]
async fn gather_reports_plausible_hardware() {
    let cfg = crate::api::Higgs::new(crate::HiggsConfig::default()).server_config();
    let info = SystemInfo::gather(cfg, vec![cpu_device()]);
    assert!(!info.hardware.cpu_name.is_empty(), "cpu name present");
    assert!(info.hardware.cpu_cores >= 1, "at least one core");
    assert!(info.hardware.ram_total_bytes > 0, "ram total > 0");
    assert!(
        info.hardware.ram_used_bytes <= info.hardware.ram_total_bytes,
        "used ram does not exceed total"
    );
    assert!(
        (0.0..=100.0).contains(&info.hardware.cpu_usage_percent),
        "cpu usage in 0..=100, got {}",
        info.hardware.cpu_usage_percent
    );
    assert_eq!(info.runtime.engine, "llama.cpp");
    assert!(!info.runtime.backend.is_empty());
    // Read-only server config is folded in: bind host is the fixed loopback
    // invariant and the auto ctx cap is the DEFAULT_CTX_CAP const.
    assert_eq!(info.config.bind_host, crate::api::BIND_HOST);
    assert_eq!(info.config.default_ctx_cap, crate::api::DEFAULT_CTX_CAP);
    // The gathered devices are carried through verbatim; a CPU-only list
    // contributes nothing to the VRAM total.
    assert_eq!(info.hardware.gpus.len(), 1);
    assert_eq!(info.hardware.vram_total_bytes, 0);
}
