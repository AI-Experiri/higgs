#!/usr/bin/env bash
# cut-release.sh — cut a new higgs release the ONLY way `main` changes: branch off
# `main`, merge `develop` in, bump the version there, and open the PR that (on merge)
# triggers CI to build + sign + publish. You merge the PR — the script never merges.
#
# The branch flow (see RELEASING.md "Branch flow"):
#   feature ─▶ develop        (you merge your feature branch first — Step 1, manual)
#   main ─▶ release/vX ◀─ merge develop ─▶ bump version+CHANGELOG ─▶ PR ─▶ main
# The version + CHANGELOG bump rides IN the PR to main (on the release branch).
#
# Usage:
#   scripts/release/cut-release.sh <x.y.z> [--from <branch>] [--no-verify] [--no-pr] [--dry-run]
#
#   <x.y.z>        New version. Must pass CI's exact semver allowlist:
#                  ^[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.-]+)?(\+[0-9A-Za-z.-]+)?$
#   --from <b>     Integration branch to merge into the release (default: develop).
#   --no-verify    Skip scripts/quality.sh (fmt/clippy/test/bindings-sync).
#   --no-pr        Prepare the release branch locally but do not push / open a PR.
#   --dry-run      Show what it would do and touch nothing (no writes, no git, no gh).
#
# PRECONDITION: your feature is already merged into `develop` (Step 1) and the tree
# is clean. Run from anywhere in the repo — the script checks out `main` itself.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

# CI's version regex (release.yml gate) — keep byte-identical.
SEMVER_RE='^[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.-]+)?(\+[0-9A-Za-z.-]+)?$'

version=""
from_branch="develop"
verify=1
open_pr=1
dry_run=0
while [ $# -gt 0 ]; do
  case "$1" in
    --from)      from_branch="${2:?--from needs a branch}"; shift 2 ;;
    --no-verify) verify=0; shift ;;
    --no-pr)     open_pr=0; shift ;;
    --dry-run)   dry_run=1; shift ;;
    -h|--help)   sed -n '2,29p' "$0"; exit 0 ;;
    -*)          echo "error: unknown flag: $1" >&2; exit 2 ;;
    *)           [ -z "$version" ] || { echo "error: version given twice" >&2; exit 2; }
                 version="$1"; shift ;;
  esac
done
[ -n "$version" ] || { echo "error: missing <x.y.z>. See --help." >&2; exit 2; }

printf '%s' "$version" | grep -qE "$SEMVER_RE" || {
  echo "error: '$version' is not a plain semver string (CI would refuse it)." >&2; exit 1
}

command -v cargo >/dev/null 2>&1 || { echo "error: cargo not found." >&2; exit 1; }
command -v git   >/dev/null 2>&1 || { echo "error: git not found." >&2; exit 1; }

if git rev-parse "refs/tags/v$version" >/dev/null 2>&1; then
  echo "error: tag v$version already exists — that version is already released." >&2; exit 1
fi

# Clean tree required (we checkout, branch, merge, commit). Skipped for a dry run.
if [ "$dry_run" -eq 0 ] && [ -n "$(git status --porcelain)" ]; then
  echo "error: working tree is dirty. Commit or stash first, then re-run." >&2
  echo "       (or use --dry-run to preview)" >&2
  exit 1
fi

# The integration branch must exist (locally or on origin) — it carries the code
# we are releasing (your feature merged into it in Step 1).
if ! git rev-parse --verify --quiet "$from_branch" >/dev/null \
   && ! git rev-parse --verify --quiet "origin/$from_branch" >/dev/null; then
  echo "error: integration branch '$from_branch' not found (locally or on origin)." >&2
  echo "       Merge your feature into it first (Step 1), then re-run." >&2
  exit 1
fi

branch="release/v$version"
today="$(date +%F)"

# --- dry run: describe the plan and stop (no git, no writes) -------------------
if [ "$dry_run" -eq 1 ]; then
  cur="$(grep -m1 '^version = ' Cargo.toml | sed -E 's/^version = "([^"]*)".*$/\1/')"
  cat <<EOF
── dry run: would cut v$version ──
  1. git switch main; git pull --ff-only
  2. git switch -c $branch
  3. git merge --no-ff $from_branch
  4. bump Cargo.toml version ${cur:-?} → $version, refresh Cargo.lock,
     roll CHANGELOG.md [Unreleased] → [$version] - $today
  5. ${verify:+run scripts/quality.sh; }preview scripts/release/check-release.sh
  6. commit "release: v$version"${open_pr:+; push + gh pr create --base main}
EOF
  exit 0
fi

# --- 1. baseline main ---------------------------------------------------------
echo "→ checking out a clean main …"
git fetch origin --quiet 2>/dev/null || true
git switch main
git pull --ff-only 2>/dev/null || echo "warn: could not fast-forward main from origin (continuing on local main)."

# --- 2 + 3. release branch off main, merge the integration branch in ----------
echo "→ branching $branch off main and merging $from_branch …"
git switch -c "$branch"
git merge --no-ff --no-edit "$from_branch" \
  || { echo "error: merge of '$from_branch' into $branch hit conflicts — resolve them, commit, then re-run the bump manually." >&2; exit 1; }

# --- 4a. Cargo.toml: bump the FIRST version line (the [package] version) -------
cur="$(grep -m1 '^version = ' Cargo.toml | sed -E 's/^version = "([^"]*)".*$/\1/')"
[ -n "$cur" ] || { echo "error: could not read [package] version from Cargo.toml." >&2; exit 1; }
echo "→ version: $cur → $version"
if [ "$cur" = "$version" ]; then
  echo "error: Cargo.toml is already at $version — bump to a new version." >&2; exit 1
fi
awk -v v="$version" 'BEGIN{d=0} !d && /^version = "/ {sub(/"[^"]*"/,"\"" v "\""); d=1} {print}' \
  Cargo.toml > Cargo.toml.new
mv Cargo.toml.new Cargo.toml

# --- 4b. CHANGELOG.md: insert a dated section right under [Unreleased] ---------
[ -f CHANGELOG.md ] || { echo "error: CHANGELOG.md missing." >&2; exit 1; }
grep -qE '^## \[Unreleased\]' CHANGELOG.md || {
  echo "error: CHANGELOG.md has no '## [Unreleased]' section to roll." >&2; exit 1
}
awk -v v="$version" -v d="$today" '
  BEGIN{done=0}
  /^## \[Unreleased\]/ && !done { print; print ""; print "## [" v "] - " d; done=1; next }
  {print}
' CHANGELOG.md > CHANGELOG.md.new
mv CHANGELOG.md.new CHANGELOG.md

# --- 4c. refresh Cargo.lock's higgs entry -------------------------------------
if ! cargo update -p "higgs@$cur" --precise "$version" 2>/dev/null; then
  cargo update -p higgs 2>/dev/null || {
    echo "warn: could not auto-refresh Cargo.lock; run 'cargo build' before merging." >&2
  }
fi

# --- 5. quality gate ----------------------------------------------------------
if [ "$verify" -eq 1 ]; then
  echo "→ running scripts/quality.sh (skip with --no-verify) …"
  ./scripts/quality.sh
fi

# --- 5b. preview the PR's release-check (non-fatal; the PR is where you fix it) -
echo "→ previewing the release-check (what the PR's CI gate will assert) …"
if ! ./scripts/release/check-release.sh; then
  echo "⚠ the PR's release-check will FAIL as above — fix before merging (e.g. pin a signing key)."
fi

# --- 6. commit + PR -----------------------------------------------------------
git add Cargo.toml Cargo.lock CHANGELOG.md
git commit -m "release: v$version"

if [ "$open_pr" -eq 0 ]; then
  echo "✅ Prepared $branch locally (no PR, --no-pr). Push + open the PR when ready:"
  echo "   git push -u origin $branch && gh pr create --base main --head $branch"
  exit 0
fi

command -v gh >/dev/null 2>&1 || {
  echo "✅ Committed on $branch. gh not found — push and open the PR manually:" >&2
  echo "   git push -u origin $branch && gh pr create --base main --head $branch" >&2
  exit 0
}
git push -u origin "$branch"
gh pr create --base main --head "$branch" \
  --title "release: v$version" \
  --body "$(printf 'Cut v%s (release/%s off main, %s merged in).\n\nOn merge to main, release.yml builds, signs, and publishes the v%s GitHub Release (macOS metal + Linux cpu/cuda). Release notes come from the CHANGELOG [%s] section.\n\nMerge is the release trigger.' "$version" "v$version" "$from_branch" "$version" "$version")"

cat <<EOF

✅ Opened the release PR for v$version.
   Next: review + MERGE it → CI cuts the release automatically.
   Then: scripts/release/mirror-assets.sh $version <dest>   (for remote self-update)
EOF
