#!/usr/bin/env bash
# Coverage gate for higgs — runs BOTH independent gates and fails if EITHER does:
#
#   * unit        (cargo test --lib)        line coverage >= 90%   (coverage-unit.sh)
#   * integration (the tests/ targets only) line coverage >= 75%   (coverage-integration.sh)
#
# The two suites are gated separately on purpose: unit tests carry exhaustive
# branch coverage of the in-crate logic, while the integration tests in tests/
# spawn `higgs` as a real OS process and drive it over HTTP + iroh, loading a
# real ~1MB tiny GGUF — they cover end-to-end PATHS, so a lower bar fits them.
# Each gate excludes the files that are the OTHER suite's responsibility (the
# unit gate drops the daemon + FFI; the integration gate drops the pure-logic
# tool parsers) — see each sub-script's header for the rationale.
#
#   HIGGS_TEST_GGUF   path to the tiny GGUF (default: the on-disk HF-cache copy,
#                     see tests/common/mod.rs::tiny_gguf_path)
#
# Requires cargo-llvm-cov (`cargo install cargo-llvm-cov`). The FFI build env
# (SDKROOT / BINDGEN_EXTRA_CLANG_ARGS) is supplied by ./.cargo/config.toml.
#
# Usage: scripts/coverage.sh            # run both gates (pass/fail)
#        scripts/coverage.sh --html     # also write HTML reports under target/
set -euo pipefail
cd "$(dirname "$0")"

echo "===== UNIT gate (cargo test --lib, >= 90%) ====="
./coverage-unit.sh "$@"

echo
echo "===== INTEGRATION gate (tests/ only, >= 75%) ====="
./coverage-integration.sh "$@"

echo
echo "Both coverage gates passed."
