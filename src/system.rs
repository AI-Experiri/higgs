//! Host hardware + inference-runtime info for the `system` control-op (formerly
//! `GET /api/higgs/system`).
//!
//! Mirrors the panels LM Studio shows (Hardware, Runtime): CPU/RAM/usage from
//! the `sysinfo` crate, and the engine/backend higgs runs on. All values are
//! observed at request time — nothing is curated or hardcoded except the engine
//! identity (which is fixed: higgs runs llama.cpp via `llama-cpp-2`).

use serde::Serialize;
use sysinfo::System;

use crate::api::HiggsServerConfig;
use crate::LLAMA_CPP_2_VERSION;

higgs_const_enum! {
    /// Kind of compute device, mirroring ggml's `ggml_backend_dev_type`
    /// (`GGML_BACKEND_DEVICE_TYPE_{CPU,GPU,ACCEL}`). The worker maps the raw
    /// FFI enum to this engine-agnostic shape; the mapping lives only in
    /// `llamacpp.rs` (the sole file allowed to name the FFI).
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, serde::Deserialize)]
    pub enum DeviceKind {
        /// A CPU backend device.
        Cpu,
        /// A GPU backend device (Metal on macOS, CUDA/Vulkan elsewhere).
        Gpu,
        /// A non-GPU accelerator backend device.
        Accel,
    }
}

higgs_ts! {
    /// One compute device enumerated by the worker via ggml's backend-device FFI.
    ///
    /// Gathered by the WORKER process (the one linking llama.cpp/Metal) for the
    /// host it runs on, so the values are engine-native and ready for future
    /// remote workers. `vram_total_bytes`/`vram_free_bytes` are ggml's reported
    /// device memory; for a unified-memory Apple GPU these track system RAM.
    #[derive(Debug, Clone, Serialize, serde::Deserialize)]
    pub struct GpuDevice {
        /// Backend device name (e.g. `"Metal"`, `"CUDA0"`).
        pub name: String,
        /// Human-readable device description (e.g. `"Apple M3 Max"`).
        pub description: String,
        /// Device kind (CPU / GPU / accelerator).
        pub kind: DeviceKind,
        /// Total device memory in bytes, as ggml reports it.
        #[ts(type = "number")]
        pub vram_total_bytes: u64,
        /// Free device memory in bytes at enumeration time.
        #[ts(type = "number")]
        pub vram_free_bytes: u64,
    }
}

higgs_ts! {
    /// Host hardware snapshot.
    #[derive(Debug, Clone, Serialize, serde::Deserialize)]
    pub struct HardwareInfo {
        /// CPU brand string (e.g. `"Apple M3 Max"`).
        pub cpu_name: String,
        /// Process architecture (e.g. `"aarch64"`).
        pub arch: String,
        /// Logical CPU count.
        #[ts(type = "number")]
        pub cpu_cores: u32,
        /// Total physical RAM in bytes.
        #[ts(type = "number")]
        pub ram_total_bytes: u64,
        /// RAM currently in use, in bytes.
        #[ts(type = "number")]
        pub ram_used_bytes: u64,
        /// Global CPU load at request time, 0–100.
        pub cpu_usage_percent: f32,
        /// Compute devices the worker enumerated via ggml's backend-device FFI.
        /// Empty when no worker could be reached or no devices were reported.
        pub gpus: Vec<GpuDevice>,
        /// Sum of every enumerated GPU device's `vram_total_bytes`; `0` when no
        /// GPU device is present. The headroom basis for [`fits_vram`].
        #[ts(type = "number")]
        pub vram_total_bytes: u64,
    }
}

higgs_ts! {
    /// Inference engine/runtime identity.
    #[derive(Debug, Clone, Serialize, serde::Deserialize)]
    pub struct RuntimeInfo {
        /// Engine name — always `"llama.cpp"` in v1.
        pub engine: String,
        /// Compute backend: `"Metal"` on macOS, else `"CPU"`.
        pub backend: String,
        /// Engine version reported at runtime by `ggml_version()` (e.g. `"0.9.7"`) —
        /// the actual vendored ggml/llama.cpp engine version.
        pub version: String,
        /// `llama-cpp-2` Rust binding crate version (e.g. `"0.1.139"`) — the wrapper
        /// the engine is driven through, distinct from the engine version above.
        pub binding: String,
    }
}

higgs_ts! {
    /// The `system` control-op response (formerly `GET /api/higgs/system`):
    /// hardware + runtime + server config.
    #[derive(Debug, Clone, Serialize)]
    pub struct SystemInfo {
        /// Host hardware (CPU, RAM, live load).
        pub hardware: HardwareInfo,
        /// Inference engine + backend.
        pub runtime: RuntimeInfo,
        /// Read-only effective server config (scan dirs, load defaults, bind host).
        pub config: HiggsServerConfig,
    }
}

/// Best-effort host name (empty string when unavailable). Single source for both the friendly
/// instance name ([`crate::config::name_or_init`]) and a node's `M_NODE_INVENTORY` hostname.
pub fn hostname() -> String {
    System::host_name().unwrap_or_default()
}

impl HardwareInfo {
    /// Stable signature over the facts that change a tuned profile: total VRAM,
    /// total RAM, CPU core count (the tuner derives `threads`/`n_threads_batch`
    /// from it — see `tune::derive`), and the GPU roster (name + per-device VRAM).
    /// Excludes volatile fields (usage %, free bytes) so it flips only on a real
    /// hardware change — used to detect a stale tuning profile
    /// (`ModelReadiness::NeedsRetune`).
    pub fn fingerprint(&self) -> String {
        use std::fmt::Write as _;
        let mut s = String::new();
        let _ = write!(
            s,
            "v{}r{}c{}n{}",
            self.vram_total_bytes,
            self.ram_total_bytes,
            self.cpu_cores,
            self.gpus.len()
        );
        for g in &self.gpus {
            let _ = write!(s, "|{}:{}", g.name, g.vram_total_bytes);
        }
        s
    }

    /// Free VRAM across GPU devices ONLY. A CPU/accel device reports *system*
    /// memory as its `vram_free_bytes`, so summing every device would overstate
    /// GPU headroom — the same reason `vram_total_bytes` is GPU-filtered in
    /// `gather_hardware_runtime`. Used by the readiness fit check so a profiled
    /// model isn't marked `Servable` when actual GPU free memory is insufficient.
    pub fn free_vram_bytes(&self) -> u64 {
        self.gpus
            .iter()
            .filter(|g| g.kind == DeviceKind::Gpu)
            .map(|g| g.vram_free_bytes)
            .sum()
    }

    /// True on a UNIFIED-memory system (Apple Silicon), where the GPU's "VRAM" is
    /// the same physical pool as system RAM. The readiness fit check must then SUM
    /// the VRAM + RAM footprints against one free pool rather than checking them
    /// independently (which would double-count the shared memory and over-report
    /// `Servable`).
    ///
    /// Signalled by a Metal GPU on an `aarch64` host: Intel Macs ALSO run Metal
    /// but with a DISCRETE AMD GPU (separate VRAM), so a bare "Metal" name isn't
    /// enough — gate on the arm64 arch too.
    pub fn is_unified_memory(&self) -> bool {
        self.arch == "aarch64"
            && self
                .gpus
                .iter()
                .any(|g| g.kind == DeviceKind::Gpu && g.name == "Metal")
    }
}

impl SystemInfo {
    /// Gather a fresh snapshot. Blocking (samples CPU usage over a short
    /// interval) — call from a blocking context, not directly in an async task.
    ///
    /// `config` is the read-only server-config snapshot from
    /// [`Higgs::server_config`](crate::api::Higgs::server_config); it is folded
    /// in verbatim (no I/O) so the response carries the live scan dirs and load
    /// defaults alongside the sampled hardware/runtime.
    ///
    /// `gpus` is the worker-gathered device list (see [`Higgs::sysinfo`]); it is
    /// folded into the hardware snapshot and its GPU totals summed into
    /// `vram_total_bytes`. Pass an empty vec when no worker could be reached.
    pub fn gather(config: HiggsServerConfig, gpus: Vec<GpuDevice>) -> Self {
        let (hardware, runtime) = Self::gather_hardware_runtime(gpus);
        SystemInfo {
            hardware,
            runtime,
            config,
        }
    }

    /// Sample just the host hardware + runtime (no server config) — used by a node, which
    /// has no `HiggsServerConfig` but still reports cpu/ram/gpu over `M_NODE_SYSINFO`.
    /// Blocking (samples CPU over a short interval); call from a blocking context.
    pub fn gather_hardware_runtime(gpus: Vec<GpuDevice>) -> (HardwareInfo, RuntimeInfo) {
        let mut sys = System::new();
        sys.refresh_memory();
        // CPU usage needs two samples spaced by the platform minimum interval.
        sys.refresh_cpu_usage();
        std::thread::sleep(sysinfo::MINIMUM_CPU_UPDATE_INTERVAL);
        sys.refresh_cpu_usage();

        let cpu_name = sys
            .cpus()
            .first()
            .map(|c| c.brand().trim().to_string())
            .unwrap_or_else(|| "unknown".to_string());

        // Sum GPU-device VRAM only — CPU/accel devices report system memory and
        // would double-count RAM. `0` when no GPU device was enumerated.
        let vram_total_bytes = gpus
            .iter()
            .filter(|d| d.kind == DeviceKind::Gpu)
            .map(|d| d.vram_total_bytes)
            .sum();

        let hardware = HardwareInfo {
            cpu_name,
            arch: std::env::consts::ARCH.to_string(),
            cpu_cores: sys.cpus().len() as u32,
            ram_total_bytes: sys.total_memory(),
            ram_used_bytes: sys.used_memory(),
            cpu_usage_percent: sys.global_cpu_usage(),
            gpus,
            vram_total_bytes,
        };
        let runtime = RuntimeInfo {
            engine: "llama.cpp".to_string(),
            // Report the backend llama.cpp was actually COMPILED with. macOS links
            // Metal; a Linux release built `--features cuda` (→ `llama-cpp-2/cuda`)
            // links CUDA; everything else is a CPU build. A plain `target_os` check
            // mislabelled CUDA builds as "CPU", misleading the `system` control-op + node
            // inventory (and any remote scheduling/diagnostics keyed on this field).
            backend: if cfg!(target_os = "macos") {
                "Metal".to_string()
            } else if cfg!(feature = "cuda") {
                "CUDA".to_string()
            } else {
                "CPU".to_string()
            },
            version: crate::worker::engine::llamacpp::engine_version(),
            binding: LLAMA_CPP_2_VERSION.to_string(),
        };
        (hardware, runtime)
    }
}

higgs_ts! {
    /// Outcome of a VRAM fit decision: whether a model of a given size fits the
    /// safe headroom over reported VRAM, plus the two byte figures the verdict
    /// was computed from. Produced by [`fits_vram`].
    #[derive(Debug, Clone, Copy, PartialEq, Serialize)]
    pub struct FitAssessment {
        /// `true` when `needed_bytes <= available_bytes`.
        pub fits: bool,
        /// The model's estimated memory need in bytes (the GGUF file size on
        /// disk — a lower-bound proxy for resident weights).
        #[ts(type = "number")]
        pub needed_bytes: u64,
        /// The safe VRAM budget: `vram_total_bytes` scaled by the headroom
        /// fraction. `0` when no VRAM was reported (no GPU), so the verdict is
        /// "does not fit" and the caller falls back to the RAM headroom guard.
        #[ts(type = "number")]
        pub available_bytes: u64,
    }
}

/// Decide whether a model of `model_size_bytes` fits the safe headroom over
/// reported VRAM: `needed <= vram_total_bytes * headroom_fraction`.
///
/// This is the host-side decision primitive higgs makes from worker-reported
/// VRAM. `headroom_fraction` is the EXISTING
/// [`MEMORY_HEADROOM_FRACTION`](crate::api::MEMORY_HEADROOM_FRACTION) — no new
/// knob is invented. When `vram_total_bytes` is `0` (no GPU enumerated) the
/// budget is `0`, so the verdict is `false`: the caller falls back to the
/// system-RAM headroom guard that already gates loads.
pub fn fits_vram(
    model_size_bytes: u64,
    vram_total_bytes: u64,
    headroom_fraction: f64,
) -> FitAssessment {
    #[allow(clippy::cast_precision_loss, clippy::cast_sign_loss)]
    let available_bytes = ((vram_total_bytes as f64) * headroom_fraction) as u64;
    FitAssessment {
        fits: model_size_bytes <= available_bytes,
        needed_bytes: model_size_bytes,
        available_bytes,
    }
}

#[cfg(test)]
#[path = "system_tests.rs"]
mod tests;
