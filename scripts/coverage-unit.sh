#!/usr/bin/env bash
# Unit-coverage gate for higgs. Fails if LINE coverage of the lib unit tests
# (the in-crate `#[cfg(test)] mod tests` blocks) drops below 90%.
#
# This measures ONLY `cargo test --lib` — the fast, branch-heavy logic tests.
# The end-to-end / spawn-the-process tests in tests/ are gated separately by
# scripts/coverage-integration.sh (a lower threshold, since they exercise paths,
# not every branch). scripts/coverage.sh runs BOTH gates.
#
# EXCLUSIONS: files that are inherently un-unit-testable (real process / FFI / live iroh
# transport) and are the INTEGRATION gate's responsibility instead. Their lib coverage is
# either zero (only a spawned process runs them) or nondeterministic (async spawned tasks —
# close-watchers, accept loops, relay readers — race the test's end, so the % flickers ±0.25
# run-to-run). The integration gate exercises them deterministically via real processes:
#   * node/cli.rs            — the `--node` daemon main/run-loop (spawned process only).
#   * engine/llamacpp/mod.rs — the llama.cpp FFI engine (needs a real GGUF in a worker).
#   * node/mod.rs            — iroh bind/accept-gate/dial/serve loop (live connections).
#   * node/{hub,fleet,transport,data}.rs — the hub accept loop, fleet routing + background
#                              watchers, per-node transport, and chat/pull relays — all driven
#                              by spawned tasks over real iroh streams.
# Everything else (pure logic + sync wiring) must hit 90% from unit tests alone.
#
# Requires cargo-llvm-cov (`cargo install cargo-llvm-cov`). The FFI build env
# (SDKROOT / BINDGEN_EXTRA_CLANG_ARGS) is supplied by ./.cargo/config.toml.
#
# Usage: scripts/coverage-unit.sh            # gate only (pass/fail)
#        scripts/coverage-unit.sh --html     # also write an HTML report under target/
set -euo pipefail
cd "$(dirname "$0")/.."

exec cargo llvm-cov --lib \
  --ignore-filename-regex 'node/cli\.rs|engine/llamacpp/mod\.rs|node/mod\.rs|node/hub\.rs|node/fleet\.rs|node/transport\.rs|node/data\.rs' \
  --fail-under-lines 90 "$@"
