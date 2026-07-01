use super::*;

/// Minimal `HardwareInfo` for fingerprint tests — no FFI, no sampling.
fn fp_hw(vram: u64, ram: u64, gpus: Vec<GpuDevice>) -> HardwareInfo {
    HardwareInfo {
        cpu_name: "test".into(),
        arch: "test".into(),
        cpu_cores: 8,
        ram_total_bytes: ram,
        ram_used_bytes: 0,
        cpu_usage_percent: 0.0,
        gpus,
        vram_total_bytes: vram,
    }
}

#[test]
fn fingerprint_is_stable_and_hardware_sensitive() {
    let hw = fp_hw(24 << 30, 64 << 30, vec![]);
    let a = hw.fingerprint();
    assert_eq!(a, hw.fingerprint(), "same hardware → same fingerprint");

    // VRAM total change → new fingerprint.
    assert_ne!(a, fp_hw(25 << 30, 64 << 30, vec![]).fingerprint());
    // RAM total change → new fingerprint.
    assert_ne!(a, fp_hw(24 << 30, 32 << 30, vec![]).fingerprint());
    // GPU roster change → new fingerprint.
    assert_ne!(
        a,
        fp_hw(24 << 30, 64 << 30, vec![cpu_device()]).fingerprint()
    );
    // CPU core count change → new fingerprint: the tuner derives `threads` /
    // `n_threads_batch` from `cpu_cores`, so a different CPU must re-tune.
    let mut more_cores = fp_hw(24 << 30, 64 << 30, vec![]);
    more_cores.cpu_cores = 16;
    assert_ne!(
        a,
        more_cores.fingerprint(),
        "cpu core count change → new fingerprint (re-tune)"
    );
}

#[test]
fn fingerprint_ignores_volatile_fields() {
    // usage % and free bytes (RuntimeInfo / per-device free) are not part of the
    // signature, so they must not flip it. Only used_memory differs here.
    let mut a = fp_hw(24 << 30, 64 << 30, vec![]);
    let base = a.fingerprint();
    a.ram_used_bytes = 10 << 30;
    a.cpu_usage_percent = 88.0;
    assert_eq!(base, a.fingerprint());
}

#[test]
fn free_vram_counts_only_gpu_devices() {
    let gpu = GpuDevice {
        name: "Metal".into(),
        description: "GPU".into(),
        kind: DeviceKind::Gpu,
        vram_total_bytes: 16 << 30,
        vram_free_bytes: 8 << 30,
    };
    let cpu = GpuDevice {
        name: "CPU".into(),
        description: "cpu".into(),
        kind: DeviceKind::Cpu,
        vram_total_bytes: 0,
        // A CPU/accel device reports SYSTEM memory here — must be excluded from
        // GPU headroom, else readiness would over-report `Servable`.
        vram_free_bytes: 64 << 30,
    };
    let hw = fp_hw(16 << 30, 64 << 30, vec![gpu, cpu]);
    assert_eq!(
        hw.free_vram_bytes(),
        8 << 30,
        "only the GPU device's free VRAM counts toward the fit check"
    );
}

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

#[test]
fn is_unified_memory_needs_apple_silicon_and_metal() {
    let metal = GpuDevice {
        name: "Metal".into(),
        description: "Apple M3 Max".into(),
        kind: DeviceKind::Gpu,
        vram_total_bytes: 64 << 30,
        vram_free_bytes: 40 << 30,
    };
    // Apple Silicon: aarch64 + Metal → unified (the fit check must sum VRAM + RAM).
    let mut apple = fp_hw(64 << 30, 64 << 30, vec![metal.clone()]);
    apple.arch = "aarch64".into();
    assert!(apple.is_unified_memory(), "apple silicon + metal → unified");
    // Intel Mac: x86_64 + Metal but a DISCRETE AMD GPU → NOT unified.
    let mut intel = fp_hw(64 << 30, 64 << 30, vec![metal]);
    intel.arch = "x86_64".into();
    assert!(
        !intel.is_unified_memory(),
        "intel mac Metal is a discrete GPU"
    );
    // aarch64 with a discrete CUDA GPU (not Metal) → not unified.
    let cuda = GpuDevice {
        name: "CUDA0".into(),
        description: "NVIDIA".into(),
        kind: DeviceKind::Gpu,
        vram_total_bytes: 24 << 30,
        vram_free_bytes: 20 << 30,
    };
    let mut arm_cuda = fp_hw(24 << 30, 64 << 30, vec![cuda]);
    arm_cuda.arch = "aarch64".into();
    assert!(!arm_cuda.is_unified_memory(), "aarch64 + CUDA is discrete");
    // No GPU at all → not unified.
    let mut nogpu = fp_hw(0, 64 << 30, vec![]);
    nogpu.arch = "aarch64".into();
    assert!(!nogpu.is_unified_memory());
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
