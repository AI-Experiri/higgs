#!/usr/bin/env bash
# check-release.sh — validate that a PR which WOULD cut a release is well-formed.
#
# Mirrors release.yml's gate and shifts it LEFT onto the PR: run on every pull request
# to main so a malformed release can't merge. A PR "would cut a release" iff the
# v<Cargo.toml version> tag does NOT already exist (the same condition release.yml uses
# on push). When it would, this enforces the release requirements; otherwise it is a
# no-op pass. Content-only — no secrets, no build — so it is safe on fork PRs.
#
# Usage:  scripts/release/check-release.sh
# Exit:   0 = ok (release-ready, or not a release PR); 1 = a requirement failed.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

fail=0
ok()  { printf '  [ok]   %s\n' "$*"; }
bad() { printf '  [FAIL] %s\n' "$*"; fail=1; }

SEMVER_RE='^[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.-]+)?(\+[0-9A-Za-z.-]+)?$'
ver="$(grep -m1 '^version = ' Cargo.toml | sed -E 's/^version = "([^"]*)".*$/\1/')"
[ -n "$ver" ] || { echo "error: no [package] version in Cargo.toml"; exit 1; }

echo "higgs release check — Cargo.toml [package] version: $ver"

# 1. valid semver (release.yml refuses anything else)
if printf '%s' "$ver" | grep -qE "$SEMVER_RE"; then
  ok "version is valid semver"
else
  bad "version '$ver' is not a plain semver string (release.yml would refuse it)"
fi

# Is this version already released? A matching tag means merging this PR will NOT cut a
# release, so there is nothing further to enforce.
if git ls-remote --tags --exit-code origin "refs/tags/v$ver" >/dev/null 2>&1; then
  echo
  echo "Tag v$ver already exists → merging this PR does NOT cut a release. Nothing to enforce."
  exit "$fail"
fi

echo "  (v$ver is untagged → merging this PR CUTS a release; enforcing requirements)"

# 2. CHANGELOG.md has a dated section for this version (release notes come from it)
esc="${ver//./\\.}"; esc="${esc//+/\\+}"
if grep -qE "^## \[$esc\] - [0-9]{4}-[0-9]{2}-[0-9]{2}" CHANGELOG.md; then
  ok "CHANGELOG.md has a dated [$ver] section"
else
  bad "CHANGELOG.md is missing '## [$ver] - YYYY-MM-DD' — add release notes (scripts/release/cut-release.sh does this)"
fi

# 3. Cargo.lock's higgs entry is in sync with the bumped version
lock_ver="$(awk '/^name = "higgs"$/{f=1;next} f&&/^version = /{gsub(/version = "|"/,"");print;exit}' Cargo.lock)"
if [ "$lock_ver" = "$ver" ]; then
  ok "Cargo.lock higgs entry == $ver"
else
  bad "Cargo.lock higgs version is '${lock_ver:-<none>}', expected '$ver' — run: cargo update -p higgs"
fi

# 4. a signing key is pinned, else release.yml would refuse to sign this release
pins="$(grep -vE '^[[:space:]]*(#|$)' .github/release-pubkeys.txt 2>/dev/null | awk 'NF==2{c++} END{print c+0}' || true)"
if [ "${pins:-0}" -ge 1 ]; then
  ok "$pins signing key(s) pinned in .github/release-pubkeys.txt"
else
  bad "no signing key pinned — release.yml would fail at signing. Run scripts/keys/mint-keys.sh (RELEASING.md Part A)"
fi

echo
if [ "$fail" -eq 0 ]; then
  echo "PASS — v$ver is release-ready."
else
  echo "FAIL — fix the [FAIL] items above before merging (they would break the release)."
fi
exit "$fail"
