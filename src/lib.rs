//! higgs — jigglebot's in-app local model runtime.
//!
//! Standalone crate: imports nothing from any jigglebot crate. Hosts an
//! OpenAI-compatible `/v1` surface over a llama.cpp engine. The FFI runs in a
//! worker process created by re-executing the current executable with
//! `--higgs-worker` (Chromium model), speaking newline-delimited JSON-RPC 2.0
//! over stdio (MCP wire). Spec: docs/superpowers/specs/2026-06-12-higgs-runtime-design.md

// Declared first with `#[macro_use]` so `higgs_ts!` is in scope for every
// module below — it owns the single ts-rs export path for all higgs types.
#[macro_use]
mod ts_export;

pub mod actor;
pub mod api;
pub mod diagnostic;
pub mod home;
pub mod node;
pub mod remote;
pub mod log_bus;
pub mod rpc;
pub mod serve;
pub mod standalone;
pub mod supervisor;
pub mod system;
pub mod worker;

pub use api::{Higgs, HiggsConfig};
pub use diagnostic::HiggsError;
pub use log_bus::{log_filter, HiggsLogLayer, LogBus};
pub use standalone::{run_standalone, shutdown_signal, StandaloneConfig};
pub use supervisor::HiggsEvent;

/// The `llama-cpp-2` Rust binding crate version bundled with this build.
///
/// This is the BINDING version, not the engine version — the underlying engine
/// version is reported at runtime by
/// [`engine_version`](crate::worker::engine::llamacpp::engine_version)
/// (`ggml_version()`). The binding crate exposes no runtime version constant of
/// its own, so it is baked from the lock file as a compile-time const. Single
/// home — both the `/api/higgs/version` and `/api/higgs/system` responses read
/// the `binding` field from here.
pub(crate) const LLAMA_CPP_2_VERSION: &str = "0.1.139";
