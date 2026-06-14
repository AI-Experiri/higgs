//! higgs — jigglebot's in-app local model runtime.
//!
//! Standalone crate: imports nothing from any jigglebot crate. Hosts an
//! OpenAI-compatible `/v1` surface over a llama.cpp engine. The FFI runs in a
//! worker process created by re-executing the current executable with
//! `--higgs-worker` (Chromium model), speaking newline-delimited JSON-RPC 2.0
//! over stdio (MCP wire). Spec: docs/superpowers/specs/2026-06-12-higgs-runtime-design.md

pub mod api;
pub mod diagnostic;
pub mod rpc;
pub mod serve;
pub mod supervisor;
pub mod system;
pub mod worker;

pub use api::{Higgs, HiggsConfig};
pub use diagnostic::HiggsError;
pub use supervisor::HiggsEvent;

/// The `llama-cpp-2` crate version bundled with this build.
///
/// llama-cpp-2 exposes no runtime build constant, so the dependency version is
/// baked from the lock file as a compile-time const. Single home — both the
/// `/api/higgs/version` and `/api/higgs/system` responses read it from here.
pub(crate) const LLAMA_CPP_2_VERSION: &str = "0.1.139";
