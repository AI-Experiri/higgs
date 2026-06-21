#!/usr/bin/env bash
# Coverage gates for higgs — runs the unit and integration gates SEPARATELY and
# fails if any selected gate does:
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
# Usage:
#   scripts/coverage.sh                      # run BOTH gates (default)
#   scripts/coverage.sh -u | --unit          # run only the unit gate
#   scripts/coverage.sh -i | --integration   # run only the integration gate
#
# When both gates run, BOTH are executed even if the first fails (no early
# abort), and a combined pass/fail + line-% summary is printed at the end; the
# script exits non-zero if any selected gate failed.
#   scripts/coverage.sh --html               # also write HTML report(s) under target/
#   scripts/coverage.sh -u --open            # unit gate + open its HTML report
#   scripts/coverage.sh --summary-only       # terse per-file summary
#   scripts/coverage.sh -h | --help
#
# Any flag that isn't a selector below is forwarded verbatim to `cargo llvm-cov`
# (e.g. --html, --open, --json, --summary-only, --output-dir DIR). With BOTH
# gates selected, --html/--open run twice and the second overwrites the first's
# report dir — select a single gate (or pass --output-dir) when you want to keep
# an HTML report.
set -euo pipefail
self="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$self")"

usage() { sed -n '/^# Usage:/,/^# report dir/p' "$self" | sed 's/^# \{0,1\}//'; }

select_unit=false
select_integration=false
any_select=false
passthrough=()

while [ $# -gt 0 ]; do
  case "$1" in
    -u|--unit)        select_unit=true;        any_select=true ;;
    -i|--integration) select_integration=true; any_select=true ;;
    -h|--help)        usage; exit 0 ;;
    *)                passthrough+=("$1") ;;
  esac
  shift
done

# No selector given → run both (the gate's default).
if ! $any_select; then
  select_unit=true
  select_integration=true
fi

# Safe expansion of a possibly-empty array under `set -u` (macOS bash 3.2).
args=("${passthrough[@]+${passthrough[@]}}")

# Run a gate WITHOUT aborting on failure, streaming its output while capturing
# it so we can report the line-% and pass/fail in the end-of-run summary. When
# both gates run, this lets the second gate run even if the first failed — you
# always see both results. Records into the parallel summary_* arrays.
tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT
summary_names=()
summary_pcts=()
summary_status=()
overall=0

run_gate() {
  local name="$1" threshold="$2" script="$3"; shift 3
  local out="$tmpdir/$name.out" rc pct upper
  upper="$(printf '%s' "$name" | tr '[:lower:]' '[:upper:]')"
  echo "===== ${upper} gate (>= ${threshold}%) ====="
  set +e
  "$script" "$@" 2>&1 | tee "$out"
  rc=${PIPESTATUS[0]}
  set -e
  # The LAST % column on llvm-cov's TOTAL row is LINES (the gated metric).
  pct="$(awk '/^TOTAL/ {print $(NF-3)}' "$out" | tail -1)"
  [ -n "$pct" ] || pct="n/a"
  summary_names+=("$name")
  summary_pcts+=("$pct")
  if [ "$rc" -eq 0 ]; then
    summary_status+=("PASS")
  else
    summary_status+=("FAIL")
    overall=1
  fi
}

if $select_unit; then
  run_gate unit 90 ./coverage-unit.sh "${args[@]+${args[@]}}"
fi
if $select_integration; then
  $select_unit && echo
  run_gate integration 75 ./coverage-integration.sh "${args[@]+${args[@]}}"
fi

# End-of-run summary: both (or the one selected) gate results together.
echo
echo "===== COVERAGE SUMMARY ====="
i=0
while [ "$i" -lt "${#summary_names[@]}" ]; do
  printf '  %-12s %-8s lines   %s\n' \
    "${summary_names[$i]}" "${summary_pcts[$i]}" "${summary_status[$i]}"
  i=$((i + 1))
done
echo
if [ "$overall" -eq 0 ]; then
  echo "All selected coverage gates passed."
else
  echo "One or more coverage gates FAILED."
fi
exit "$overall"
