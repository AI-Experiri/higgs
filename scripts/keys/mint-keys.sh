#!/usr/bin/env bash
# mint-keys.sh — mint the higgs release-signing minisign key and pin its public
# half in .github/release-pubkeys.txt.
#
# The PRIVATE key never leaves your machine and is NEVER committed: this script
# writes it OUTSIDE the repo (~/.higgs/release-keys/minisign.key by default, or
# $HIGGS_RELEASE_KEYS_DIR) and tells you to paste its contents into the GitHub
# `MINISIGN_SECRET_KEY` environment secret, then store it in a password manager.
# Only the PUBLIC pin line (key_id + base64) is added to the repo.
#
# Usage:
#   scripts/keys/mint-keys.sh [--key-id <id>] [--out <dir>]
#   scripts/keys/mint-keys.sh --rotate [--key-id <id>] [--out <dir>]
#
#   --key-id <id>   Label for the pin (default: higgs-release-1, or
#                   higgs-release-<n+1> for --rotate). This is OUR label, not
#                   minisign's internal key id.
#   --out <dir>     Where to write minisign.{key,pub}, MUST be outside the repo
#                   (default: $HIGGS_RELEASE_KEYS_DIR or ~/.higgs/release-keys).
#   --rotate        Mint a NEW key and pin it ALONGSIDE existing keys (for the
#                   bridge-release rotation in RELEASING.md Part E). Without this,
#                   the pin file must have no keys yet.
#
# Requires: minisign (https://jedisct1.github.io/minisign/).  brew install minisign
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
pin_file="$repo_root/.github/release-pubkeys.txt"

key_id=""
out_dir="${HIGGS_RELEASE_KEYS_DIR:-${HOME:?HOME must be set}/.higgs/release-keys}"
rotate=0

while [ $# -gt 0 ]; do
  case "$1" in
    --key-id) key_id="${2:?--key-id needs a value}"; shift 2 ;;
    --out)    out_dir="${2:?--out needs a value}";    shift 2 ;;
    --rotate) rotate=1; shift ;;
    -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
    *) echo "error: unknown argument: $1" >&2; exit 2 ;;
  esac
done

# The private key must live OUTSIDE the repo so it can never be staged or committed.
case "$out_dir" in
  /*) abs_out="$out_dir" ;;
  *)  abs_out="$(pwd)/$out_dir" ;;
esac
case "$abs_out/" in
  "$repo_root"/*)
    echo "error: --out must be OUTSIDE the repo so the private key can never be committed." >&2
    echo "       repo: $repo_root" >&2
    echo "       got:  $out_dir" >&2
    exit 2 ;;
esac

command -v minisign >/dev/null 2>&1 || {
  echo "error: minisign not found. Install it (macOS: brew install minisign)." >&2
  exit 1
}
[ -f "$pin_file" ] || { echo "error: $pin_file not found (run from the higgs repo)." >&2; exit 1; }

# --- pick a key_id, defaulting sensibly and refusing duplicates ---------------
existing_ids="$(grep -vE '^\s*(#|$)' "$pin_file" 2>/dev/null | awk '{print $1}' || true)"
existing_count="$(printf '%s' "$existing_ids" | grep -c . || true)"

if [ "$rotate" -eq 0 ] && [ "$existing_count" -ne 0 ]; then
  echo "error: $pin_file already pins key(s):" >&2
  while IFS= read -r _id; do echo "  $_id" >&2; done <<<"$existing_ids"
  echo "Use --rotate to add a new key alongside them (see RELEASING.md Part E)." >&2
  exit 1
fi

if [ -z "$key_id" ]; then
  if [ "$rotate" -eq 1 ]; then
    key_id="higgs-release-$((existing_count + 1))"
  else
    key_id="higgs-release-1"
  fi
fi
# key_id must be a single clean token (the parser splits on whitespace; keep it ASCII)
printf '%s' "$key_id" | grep -qE '^[0-9A-Za-z._-]+$' || {
  echo "error: --key-id must match [0-9A-Za-z._-]+ (got: $key_id)" >&2; exit 1
}
if printf '%s\n' "$existing_ids" | grep -qxF "$key_id"; then
  echo "error: key id '$key_id' is already pinned in $pin_file." >&2; exit 1
fi

# --- generate the keypair (‑W = no password, required for unattended CI) -------
mkdir -p "$out_dir"
sec="$out_dir/minisign.key"
pub="$out_dir/minisign.pub"
if [ -e "$sec" ] || [ -e "$pub" ]; then
  echo "error: $sec or $pub already exists — refusing to overwrite an existing key." >&2
  echo "Move them aside (or pass --out <fresh-dir>) and re-run." >&2
  exit 1
fi
old_umask="$(umask)"; umask 077          # secret key readable only by you
minisign -G -W -p "$pub" -s "$sec"
umask "$old_umask"
chmod 600 "$sec" 2>/dev/null || true

# minisign.pub is 2 lines: an untrusted comment, then the base64 public key.
b64="$(sed -n '2p' "$pub" | tr -d '[:space:]')"
[ -n "$b64" ] || { echo "error: could not read the public key base64 from $pub" >&2; exit 1; }
# mirror the Rust/CI parser's fail-closed rules: printable ASCII, single token.
printf '%s' "$b64" | LC_ALL=C grep -qE '^[!-~]+$' || {
  echo "error: public key contains non-printable-ASCII bytes — refusing to pin." >&2; exit 1
}

# --- append the pin line ------------------------------------------------------
# Ensure the file ends with a newline (read fully) BEFORE appending, so the read
# and the writes are separate operations, not one read-and-write pipeline.
if [ -s "$pin_file" ] && [ -n "$(tail -c1 "$pin_file")" ]; then
  printf '\n' >> "$pin_file"
fi
printf '%s %s\n' "$key_id" "$b64" >> "$pin_file"

cat <<EOF

✅ Minted key '$key_id' and pinned its public half in:
     .github/release-pubkeys.txt      (COMMIT this — it is the public trust root)

   Private key written to:
     $sec   (mode 600 — NEVER commit this)

Now do the TWO manual steps only you can do:

  1. Paste the CONTENTS of the private key into the GitHub environment secret:
       gh secret set MINISIGN_SECRET_KEY --env release < "$sec"
     (or: Settings ▸ Environments ▸ release ▸ Environment secrets ▸ MINISIGN_SECRET_KEY)

  2. Store $sec in your password manager, then delete the local copy:
       rm -f "$sec"

Then commit the pin line and (with the rest of the feature) open a PR to main:
     git add .github/release-pubkeys.txt
     git commit -m "release: pin $key_id signing key"

EOF

if [ "$rotate" -eq 1 ]; then
  cat <<'EOF'
🔁 Rotation reminder (RELEASING.md Part E): the FIRST release after this pin must be a
   BRIDGE release that is still SIGNED WITH THE OLD KEY (keep MINISIGN_SECRET_KEY = old
   key for it). Only switch the secret to the new key for releases AFTER the fleet has
   updated through the bridge.

EOF
fi
