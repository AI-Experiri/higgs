#!/usr/bin/env bash
# Integration-coverage gate for higgs. Fails if LINE coverage of the black-box
# tests in tests/ — ALONE, with the lib unit tests EXCLUDED — drops below 75%.
#
# These tests spawn `higgs` as a real OS process and/or drive it over real HTTP
# + iroh, loading a real ~1MB tiny GGUF (ggml-org's stories260K) so the
# spawn → pair → load → chat → unload paths are actually exercised. Without the
# GGUF those tests SKIP and coverage collapses well below the gate.
#
# The threshold is 75% (not 90%): integration tests cover end-to-end PATHS, not
# every branch — exhaustive branch coverage is the unit gate's job
# (scripts/coverage-unit.sh, 90%). scripts/coverage.sh runs BOTH gates.
#
#   HIGGS_TEST_GGUF   path to the tiny GGUF (default: the on-disk HF-cache copy,
#                     see tests/common/mod.rs::tiny_gguf_path)
#
# Requires cargo-llvm-cov (`cargo install cargo-llvm-cov`). The FFI build env
# (SDKROOT / BINDGEN_EXTRA_CLANG_ARGS) is supplied by ./.cargo/config.toml.
#
# Usage: scripts/coverage-integration.sh            # gate only (pass/fail)
#        scripts/coverage-integration.sh --html     # also write an HTML report
set -euo pipefail
cd "$(dirname "$0")/.."

# EXCLUSION: the tool_parser/* AND tune/* subtrees are pure logic — tool-call
# dialect parsing and the autotune suggester (derive/vram/card/store/merge). Both
# are exhaustively covered by the UNIT gate; integration can only reach them
# indirectly (a real model emitting a dialect; the spawned process's suggester),
# which the tiny test GGUF never fully exercises. Excluded here so the 75%
# reflects integration-owned code (wiring, serving, process, remote, the tune
# ROUTE + apply path) rather than pure-module branches.
#
# Enumerate every integration target explicitly (NOT --tests, which would also
# pull in the lib unit tests and inflate the number). Keep in sync with tests/.
exec cargo llvm-cov \
  --test auth \
  --test autotune \
  --test control_api \
  --test hub_server \
  --test inference \
  --test pull \
  --test remote_cli \
  --test remote_hub_e2e \
  --test remote_node_e2e \
  --test remote_pairing \
  --test worker_roundtrip \
  --ignore-filename-regex 'tool_parser|/tune/' \
  --fail-under-lines 75 "$@"
