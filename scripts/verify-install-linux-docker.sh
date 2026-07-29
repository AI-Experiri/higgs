#!/usr/bin/env bash
# Local LINUX verification (no CI needed) for the platform-specific install.sh
# behavior a macOS dev box cannot exercise:
#
#   On Linux, `--system` is a NON-root user unit + linger (no sudo), so install.sh
#   must NOT withhold it for a group-writable prefix — unlike macOS, where `--system`
#   runs via sudo (root) and a group-writable exec path IS a root-escalation threat.
#
# Mirrors tests/install_surface.rs::
#   install_sh_warns_when_a_symlinked_bin_hides_a_group_writable_lexical_prefix
# but runs the REAL install.sh inside ubuntu:22.04 via Docker, using a STUB `higgs`
# (a shell script) — so it needs NO llama.cpp/FFI build and finishes in seconds.
#
# Requires Docker. Exit 0 = Linux behavior is correct (fix verified).
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
command -v docker >/dev/null 2>&1 || { echo "error: docker is required"; exit 2; }

inner="$(mktemp)"
trap 'rm -f "$inner"' EXIT
cat > "$inner" <<'INNER'
set -euo pipefail
w="$HOME/w"; mkdir -p "$w"; cd "$w"
# STUB artifact, exactly as tests/install_surface.rs::stage_artifact builds it.
mkdir pkg
printf '#!/bin/sh\n[ "$1" = --version ] && echo "higgs 0.0.1"\nexit 0\n' > pkg/higgs
chmod 755 pkg/higgs
name="higgs-v0.0.1-x86_64-unknown-linux-gnu"
tar -C pkg -czf "$name.tar.gz" higgs
sha256sum "$name.tar.gz" > "$name.tar.gz.sha256"
# The scenario: a CLEAN 0755 real bin the symlink points at, while the LEXICAL
# prefix that CONTAINS the `bin` symlink is GROUP-writable (0775).
mkdir clean-bin; chmod 755 clean-bin
mkdir prefix; ln -s "$w/clean-bin" prefix/bin; chmod 775 prefix
out="$(bash /install.sh --tarball "$w/$name.tar.gz" --prefix "$w/prefix" 2>&1 || true)"
echo "----- install.sh 'Next steps' on Linux -----"
echo "$out" | grep -A6 "Next steps" || echo "$out" | tail -25
echo "----- assertion -----"
if echo "$out" | grep -q "always-on: UNAVAILABLE"; then
  echo "FAIL: --system was withheld on Linux (that is macOS-only, sudo/root behavior)"; exit 1
fi
if echo "$out" | grep -q -- "--system"; then
  echo "PASS: Linux keeps --system available for a group-writable prefix (not a root escalation)"
else
  echo "FAIL: the --system next-step line is missing"; echo "$out"; exit 1
fi
INNER

echo "→ running install.sh in ubuntu:22.04 x86_64 (stub artifact, no FFI build) …"
# --platform linux/amd64: match CI's x86_64 runner (install.sh only ships Linux
# x86_64 + macOS arm64), emulated via qemu on Apple Silicon.
docker run --rm --platform linux/amd64 \
  -v "$repo/install.sh:/install.sh:ro" \
  -v "$inner:/inner.sh:ro" \
  ubuntu:22.04 \
  bash -c 'useradd -m op && cp /inner.sh /home/op/inner.sh && chown op:op /home/op/inner.sh && su - op -c "bash /home/op/inner.sh"'
