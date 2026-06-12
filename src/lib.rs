//! higgs — jigglebot's in-app local model runtime.
//!
//! Standalone crate: imports nothing from any jigglebot crate. Hosts an
//! OpenAI-compatible `/v1` surface over a llama.cpp engine. The FFI runs in a
//! worker process created by re-executing the current executable with
//! `--higgs-worker` (Chromium model), speaking newline-delimited JSON-RPC 2.0
//! over stdio (MCP wire). Spec: docs/superpowers/specs/2026-06-12-higgs-runtime-design.md

pub mod diagnostic;
pub mod rpc;

pub use diagnostic::HiggsError;
