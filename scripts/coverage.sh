#!/usr/bin/env bash
# Coverage gate for higgs. Fails if line coverage drops below 90%.
#
# Runs the full test suite (lib unit tests + the black-box integration tests
# in tests/) under llvm instrumentation and enforces the threshold. The
# integration tests load a real ~1MB tiny GGUF (ggml-org's stories260K) so the
# load → chat → unload engine path is actually exercised; without it those
# tests SKIP and coverage falls below the gate.
#
#   HIGGS_TEST_GGUF   path to the tiny GGUF (default: the on-disk HF-cache copy,
#                     see tests/common/mod.rs::tiny_gguf_path)
#
# Requires cargo-llvm-cov (`cargo install cargo-llvm-cov`). The FFI build env
# (SDKROOT / BINDGEN_EXTRA_CLANG_ARGS) is supplied by ./.cargo/config.toml.
#
# Usage: scripts/coverage.sh            # gate only (pass/fail)
#        scripts/coverage.sh --html     # also write an HTML report under target/
set -euo pipefail
cd "$(dirname "$0")/.."

exec cargo llvm-cov --fail-under-lines 90 "$@"
