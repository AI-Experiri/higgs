//! Higgs worker process modules.
//!
//! The worker is the crash-isolated process (re-executed with `--higgs-worker`)
//! that runs llama.cpp FFI. This module contains its constituent parts.

pub mod models;
