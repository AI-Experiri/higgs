#!/usr/bin/env bash
set -euo pipefail

# Quality gate for higgs (Rust-only — this crate has no frontend). Runs the fast
# checks every change should pass before commit:
#
#   * cargo fmt   (apply, then verify clean)
#   * cargo clippy --all-targets -D warnings
#   * cargo test  (full suite; also REGENERATES the ts-rs bindings)
#   * bindings sync (the committed bindings/higgs/*.ts match the Rust types)
#
# The ts-rs TypeScript files in bindings/higgs/ are emitted by ts-rs's own
# derive-generated test functions during `cargo test` (the `higgs_ts!` macro in
# src/ts_export.rs just injects `#[ts(export, export_to = "higgs/")]`). So the
# normal test pass writes them; the sync step then fails if a Rust wire-type
# change wasn't accompanied by a regenerated + committed .ts.
#
# This is the FAST gate. Line-coverage gates are separate and heavier — see
# scripts/coverage.sh (unit >= 90%, integration >= 75%).
#
# The FFI build env (SDKROOT / BINDGEN_EXTRA_CLANG_ARGS) is supplied by
# ./.cargo/config.toml `[env]`, so it applies to every cargo entry point —
# nothing to set here.
#
# Integration tests in tests/ spawn a real `higgs` process and load a tiny GGUF;
# they SKIP when it's absent (override the path with HIGGS_TEST_GGUF). They are
# not required for this gate to pass, but run them when you can.

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

FAILED=0

log()  { echo -e "${GREEN}[quality]${NC} $1"; }
warn() { echo -e "${YELLOW}[quality]${NC} $1"; }
err()  { echo -e "${RED}[quality]${NC} $1"; }
step() { echo -e "\n${CYAN}━━━ $1 ━━━${NC}"; }

run_check() {
    local label="$1"
    shift
    if "$@"; then
        log "✓ $label"
    else
        err "✗ $label"
        FAILED=1
    fi
}

# ── Format ───────────────────────────────────────────────

# Apply rustfmt first (auto-fix), then verify the tree is clean — so callers
# don't have to run `cargo fmt --all` manually before re-running this gate.
step "cargo fmt --all (apply) + --check (verify)"
cargo fmt --all
run_check "rustfmt" cargo fmt --all -- --check

# ── Lint ─────────────────────────────────────────────────

step "cargo clippy --all-targets -D warnings"
run_check "clippy" cargo clippy --all-targets -- -D warnings

# ── Test ─────────────────────────────────────────────────

# Full suite. This also regenerates bindings/higgs/*.ts via ts-rs's derive tests.
step "cargo test"
run_check "tests" cargo test

# ── Const-enum bindings (TsConstEnum) ────────────────────

# Unit-variant enums emit a const-OBJECT (`higgs_const_enum!` → TsConstEnum) instead
# of a ts-rs `"a"|"b"` union, so the frontend can use them as VALUES (consistent with
# jigglebot). Their writers are #[ignore]d `macro_run_*` tests run in a SEPARATE pass
# AFTER `cargo test` — ts-rs's transitive export (above) writes a union for each as a
# side effect of exporting a dependent struct, so this pass must run LAST to win.
step "cargo test macro_run (regenerate const-object enum bindings)"
run_check "ts-const-enum bindings" cargo test macro_run -- --ignored

# ── Bindings sync ────────────────────────────────────────

# After the test pass regenerated them, the committed ts-rs bindings must be
# unchanged — a drift means a Rust wire type changed without committing the
# regenerated TypeScript. Scope strictly to bindings/ so unrelated working-tree
# changes don't trip this.
step "ts-rs bindings sync (bindings/ unchanged after cargo test + macro_run)"
if git diff --quiet -- bindings/; then
    log "✓ bindings in sync with Rust types"
else
    err "✗ bindings/ drifted — the regen produced different .ts than committed"
    git --no-pager diff --stat -- bindings/
    echo "  Commit the regenerated bindings: git add bindings/ && commit."
    FAILED=1
fi

# ── Summary ──────────────────────────────────────────────

echo ""
if [ "$FAILED" -eq 0 ]; then
    log "All checks passed."
else
    err "Some checks failed."
    exit 1
fi
