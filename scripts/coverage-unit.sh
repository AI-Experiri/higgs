#!/usr/bin/env bash
# Unit-coverage gate for higgs. Fails if LINE coverage of the lib unit tests
# (the in-crate `#[cfg(test)] mod tests` blocks) drops below 90%.
#
# This measures ONLY `cargo test --lib` — the fast, branch-heavy logic tests.
# The end-to-end / spawn-the-process tests in tests/ are gated separately by
# scripts/coverage-integration.sh (a lower threshold, since they exercise paths,
# not every branch). scripts/coverage.sh runs BOTH gates.
#
# EXCLUSIONS: two files are inherently un-unit-testable and excluded from THIS
# gate (they are the integration gate's responsibility):
#   * node/cli.rs              — the `--node` daemon `main`/run-loop, only
#                                reachable by spawning the real process.
#   * engine/llamacpp/mod.rs   — the llama.cpp FFI engine, only exercised by
#                                loading a real GGUF in a real worker process.
# Everything else (pure logic + wiring) must hit 90% from unit tests alone.
#
# Requires cargo-llvm-cov (`cargo install cargo-llvm-cov`). The FFI build env
# (SDKROOT / BINDGEN_EXTRA_CLANG_ARGS) is supplied by ./.cargo/config.toml.
#
# Usage: scripts/coverage-unit.sh            # gate only (pass/fail)
#        scripts/coverage-unit.sh --html     # also write an HTML report under target/
set -euo pipefail
cd "$(dirname "$0")/.."

exec cargo llvm-cov --lib \
  --ignore-filename-regex 'node/cli\.rs|engine/llamacpp/mod\.rs' \
  --fail-under-lines 90 "$@"
