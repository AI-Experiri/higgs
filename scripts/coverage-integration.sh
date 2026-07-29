#!/usr/bin/env bash
# Integration-coverage gate for higgs. Fails if LINE coverage of the black-box
# tests in tests/ — ALONE, with the lib unit tests EXCLUDED — drops below 80%.
#
# These tests spawn `higgs` as a real OS process and/or drive it over real HTTP
# + iroh, loading a real ~1MB tiny GGUF (ggml-org's stories260K) so the
# spawn → pair → load → chat → unload paths are actually exercised. Without the
# GGUF those tests SKIP and coverage collapses well below the gate. The
# fleet-backed targets (chat_fleet, engine_variety, reasoning_v1,
# load_params_variety) likewise SKIP without the real small-model fleet
# (scripts/fetch_test_fleet.sh / HIGGS_TEST_FLEET) — the 80% bar assumes BOTH
# fixture sets are present, as they are on the gate machine.
#
# The threshold is 80% (not 90%): integration tests cover end-to-end PATHS, not
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

# EXCLUSION: the tune/* subtree is pure logic — autotune
# dialect parsing and the autotune suggester (derive/vram/card/store/merge). Both
# are exhaustively covered by the UNIT gate; integration can only reach them
# indirectly (a real model emitting a dialect; the spawned process's suggester),
# which the tiny test GGUF never fully exercises. Excluded here so the 80%
# reflects integration-owned code (wiring, serving, process, remote, the tune
# ROUTE + apply path) rather than pure-module branches.
# (node/service.rs is NOT excluded: `tests/install_surface.rs` spawns the real
# binary and its `--dry-run` renders the plist/systemd unit + plans, so the
# SPAWNED process's coverage of service.rs IS attributed here — it measures
# ~66% lines, well contributing to the gate, no exclusion needed.)
#
# EXCLUSION: node/self_update.rs — the self-update APPLY/boot/lock internals
# (stage_and_flip, the four boot-guard hooks, UpdateLock, perform_trial_rollback,
# the verify pipeline) are STRUCTURALLY unreachable from integration: a dev build
# pins NO release key, so every `higgs node self-update` fails CLOSED at signature
# verification (HG081) before `stage_and_flip` runs, and the boot hooks only fire
# when the daemon runs from a real `bin/v<semver>/current` install layout (the
# test binary is `target/debug/higgs`, so `self_update_bin_dir` returns None and
# the record/rollback block is skipped). Wiring in a signing key or a fake install
# layout would be exactly the test-only production knob CLAUDE.md forbids. This
# code is instead exhaustively covered by the UNIT gate (~92% lines, 71 tests).
# The REACHABLE surface is NOT excluded: `run_node_self_update` + the boot
# preflight live in node/cli.rs (measured here — `tests/self_update_surface.rs`
# spawns the real binary and drives arg parsing, the HG081 fail-closed path, and
# --rollback/--prune dry-runs). So the 80% reflects integration-owned code, not
# an apply path that cannot execute without a signed release.
#
# Enumerate every integration target explicitly (NOT --tests, which would also
# pull in the lib unit tests and inflate the number). Keep in sync with tests/.
exec cargo llvm-cov \
  --test auth \
  --test autotune \
  --test chat_fleet \
  --test config_persistence \
  --test control_api \
  --test control_errors \
  --test control_fleet_routes \
  --test cors_live \
  --test cov_facade \
  --test cov_fleet2 \
  --test cov_infra \
  --test cov_nodecli \
  --test cov_paths2 \
  --test cov_remote \
  --test cov_serve \
  --test cov_worker \
  --test courier_edges \
  --test facade_hub_edges \
  --test download_errors \
  --test engine_variety \
  --test higgs_events \
  --test hub_server \
  --test inference \
  --test install_surface \
  --test keys_api \
  --test load_params_variety \
  --test models_scan_ollama \
  --test node_chat_test \
  --test pull \
  --test reasoning_v1 \
  --test rebind \
  --test remote_cli \
  --test remote_hub_e2e \
  --test remote_load_params \
  --test remote_node_e2e \
  --test remote_pairing \
  --test remote_update_push \
  --test scan_edges \
  --test self_update_surface \
  --test serve_guard \
  --test stream_remote_chat \
  --test supervisor_lifecycle \
  --test turbotune \
  --test v1_errors \
  --test v1_models_servable \
  --test worker_exe_seam \
  --test worker_logs \
  --test worker_roundtrip \
  --ignore-filename-regex '/tune/|/node/self_update\.rs$' \
  --fail-under-lines 80 "$@"
