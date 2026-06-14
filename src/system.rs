//! Host hardware + inference-runtime info for `GET /api/higgs/system`.
//!
//! Mirrors the panels LM Studio shows (Hardware, Runtime): CPU/RAM/usage from
//! the `sysinfo` crate, and the engine/backend higgs runs on. All values are
//! observed at request time — nothing is curated or hardcoded except the engine
//! identity (which is fixed: higgs runs llama.cpp via `llama-cpp-2`).

use serde::Serialize;
use sysinfo::System;

use crate::LLAMA_CPP_2_VERSION;

/// Host hardware snapshot.
#[derive(Debug, Clone, Serialize, ts_rs::TS)]
#[ts(export, export_to = "../../../frontend/src/lib/generated/")]
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
}

/// Inference engine/runtime identity.
#[derive(Debug, Clone, Serialize, ts_rs::TS)]
#[ts(export, export_to = "../../../frontend/src/lib/generated/")]
pub struct RuntimeInfo {
    /// Engine name — always `"llama.cpp"` in v1.
    pub engine: String,
    /// Compute backend: `"Metal"` on macOS, else `"CPU"`.
    pub backend: String,
    /// Binding crate version.
    pub version: String,
}

/// Response for `GET /api/higgs/system`: hardware + runtime panels.
#[derive(Debug, Clone, Serialize, ts_rs::TS)]
#[ts(export, export_to = "../../../frontend/src/lib/generated/")]
pub struct SystemInfo {
    /// Host hardware (CPU, RAM, live load).
    pub hardware: HardwareInfo,
    /// Inference engine + backend.
    pub runtime: RuntimeInfo,
}

impl SystemInfo {
    /// Gather a fresh snapshot. Blocking (samples CPU usage over a short
    /// interval) — call from a blocking context, not directly in an async task.
    pub fn gather() -> Self {
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

        SystemInfo {
            hardware: HardwareInfo {
                cpu_name,
                arch: std::env::consts::ARCH.to_string(),
                cpu_cores: sys.cpus().len() as u32,
                ram_total_bytes: sys.total_memory(),
                ram_used_bytes: sys.used_memory(),
                cpu_usage_percent: sys.global_cpu_usage(),
            },
            runtime: RuntimeInfo {
                engine: "llama.cpp".to_string(),
                backend: if cfg!(target_os = "macos") {
                    "Metal".to_string()
                } else {
                    "CPU".to_string()
                },
                version: LLAMA_CPP_2_VERSION.to_string(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gather_reports_plausible_hardware() {
        let info = SystemInfo::gather();
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
    }
}
