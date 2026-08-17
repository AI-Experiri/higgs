use super::*;

#[test]
fn gpu_layers_constructors_and_predicates_agree() {
    // The typed replacement for the old magic-int sentinels: `All` is the
    // old `u32::MAX`, `Count{0}` the old `0` (CPU-only). Predicates and
    // constructors must agree with each other and with the FFI mapping.
    let all = GpuLayers::all();
    assert!(all.is_all());
    assert!(!all.is_cpu_only());
    assert_eq!(all.to_n_gpu_layers(), u32::MAX, "FFI keeps the sentinel");

    let cpu = GpuLayers::count(0);
    assert!(cpu.is_cpu_only());
    assert!(!cpu.is_all());
    assert_eq!(cpu.to_n_gpu_layers(), 0);

    let some = GpuLayers::count(33);
    assert!(!some.is_all());
    assert!(!some.is_cpu_only());
    assert_eq!(some.to_n_gpu_layers(), 33);

    assert_eq!(
        GpuLayers::default(),
        GpuLayers::Count { n: 0 },
        "default is CPU-only"
    );
}
