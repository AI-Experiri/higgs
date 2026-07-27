#!/usr/bin/env bash
# mirror-assets.sh — download a signed higgs release's assets into a v<version>/
# directory laid out for a STATIC HTTPS origin, so `higgs node self-update --url …`
# and the hub courier can fetch them.
#
# Why this exists: GitHub's `…/releases/download/…` URLs are NOT usable by the
# self-update fetcher — GitHub 302-redirects release assets to storage and the
# storage URL carries a query string, both of which the SSRF-hardened fetcher
# rejects (HG088). The updater needs .manifest / .minisig / .tar.gz as sibling
# files on a direct origin with no redirect and no query.
#
# Usage:
#   scripts/release/mirror-assets.sh <x.y.z> [dest] [--repo <owner/name>] [--verify]
#
#   <x.y.z>          Release version (the tag is v<x.y.z>).
#   dest             Output root (default: ./mirror). Assets land in <dest>/v<x.y.z>/.
#   --repo o/n       GitHub repo (default: AI-Experiri/higgs).
#   --verify         Re-check each .tar.gz against its .sha256 sidecar after download.
#
# Requires: gh (authenticated).  gh auth login
set -euo pipefail

repo="AI-Experiri/higgs"
verify=0
version=""
dest="./mirror"
positional=0
while [ $# -gt 0 ]; do
  case "$1" in
    --repo)   repo="${2:?--repo needs a value}"; shift 2 ;;
    --verify) verify=1; shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    -*)       echo "error: unknown flag: $1" >&2; exit 2 ;;
    *)        if [ "$positional" -eq 0 ]; then version="$1"; positional=1
              else dest="$1"; positional=2; fi
              shift ;;
  esac
done
[ -n "$version" ] || { echo "error: missing <x.y.z>. See --help." >&2; exit 2; }
printf '%s' "$version" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+([-+][0-9A-Za-z.-]+)*$' || {
  echo "error: '$version' is not a plain semver string." >&2; exit 1
}
command -v gh >/dev/null 2>&1 || { echo "error: gh not found (gh auth login)." >&2; exit 1; }

tag="v$version"
out="$dest/$tag"
mkdir -p "$out"

echo "→ downloading $repo release $tag → $out/"
# All four asset kinds per platform: .tar.gz, .tar.gz.sha256, .manifest, .manifest.minisig
gh release download "$tag" --repo "$repo" --dir "$out" --clobber \
  --pattern 'higgs-*.tar.gz' \
  --pattern 'higgs-*.tar.gz.sha256' \
  --pattern 'higgs-*.manifest' \
  --pattern 'higgs-*.manifest.minisig'

# --- sanity: every .manifest must have siblings the updater derives -----------
missing=0
shopt -s nullglob
manifests=("$out"/higgs-*.manifest)
[ "${#manifests[@]}" -gt 0 ] || { echo "error: no .manifest assets downloaded for $tag." >&2; exit 1; }
for m in "${manifests[@]}"; do
  base="${m%.manifest}"                       # …/higgs-v<ver>-<suffix>
  for sib in "$m.minisig" "$base.tar.gz" "$base.tar.gz.sha256"; do
    [ -f "$sib" ] || { echo "  MISSING sibling: $(basename "$sib")" >&2; missing=1; }
  done
done
[ "$missing" -eq 0 ] || { echo "error: incomplete asset set — the release may still be finalizing." >&2; exit 1; }

# --- optional: verify tarball hashes against their sidecars -------------------
if [ "$verify" -eq 1 ]; then
  echo "→ verifying tarball sha256 sidecars …"
  if command -v sha256sum >/dev/null 2>&1; then hasher="sha256sum"; else hasher="shasum -a 256"; fi
  for t in "$out"/higgs-*.tar.gz; do
    want="$(awk '{print $1}' "$t.sha256")"
    got="$($hasher "$t" | awk '{print $1}')"
    [ "$want" = "$got" ] || { echo "  HASH MISMATCH: $(basename "$t")" >&2; exit 1; }
    echo "  ok: $(basename "$t")"
  done
fi

cat <<EOF

✅ Mirrored $tag to: $out/
   Files (per platform): higgs-$tag-<suffix>.{tar.gz,tar.gz.sha256,manifest,manifest.minisig}

Serve <dest> from a DIRECT static HTTPS origin (no redirects, no query strings) so this
resolves as a plain file:

   https://<your-origin>/higgs/$tag/higgs-$tag-<suffix>.manifest

Then a node updates with:
   higgs node self-update --url https://<your-origin>/higgs/$tag/higgs-$tag-<suffix>.manifest

<suffix> is aarch64-apple-darwin (macOS metal), x86_64-unknown-linux-gnu (Linux cpu),
or x86_64-unknown-linux-gnu-cuda (Linux CUDA).
EOF
