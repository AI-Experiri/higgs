#!/usr/bin/env bash
# Download the chat-template/parser TEST FLEET — the real small instruct models
# `tests/chat_fleet.rs` runs against (see tests/common/mod.rs FLEET).
#
# The fleet lives OUTSIDE the live app's scan roots (test-only; higgs never
# lists these), in an LM-Studio layout (`<root>/{org}/{model}/*.gguf`) so both
# `ModelStore::scan` and a spawned test server catalog it directly.
#
# Idempotent + resumable: re-running resumes partial files (`curl -C -`) and
# skips complete ones. Sequential on purpose — parallel streams contend on a
# throttled CDN path and end up slower.
#
# Usage:
#   scripts/fetch_test_fleet.sh          # download into $HIGGS_TEST_FLEET or the default root
#   HIGGS_TEST_FLEET=/tmp/fleet scripts/fetch_test_fleet.sh
#
# After downloading, run the fleet suite (parser goldens + E2E):
#   cargo test --test chat_fleet
# Re-bless the parser goldens after an intentional parser change with:
#   UPDATE_GOLDENS=1 cargo test --test chat_fleet parser_golden

set -euo pipefail

ROOT="${HIGGS_TEST_FLEET:-$HOME/.cache/higgs-test-models}"

# id|file|url — keep in sync with tests/common/mod.rs FLEET.
FLEET=(
  "qwen/Qwen3-0.6B|Qwen3-0.6B-Q8_0.gguf|https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf"
  "google/gemma-3-1b-it|gemma-3-1b-it-Q4_K_M.gguf|https://huggingface.co/ggml-org/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q4_K_M.gguf"
  "meta-llama/Llama-3.2-1B-Instruct|Llama-3.2-1B-Instruct-Q4_0.gguf|https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_0.gguf"
  "deepseek/DeepSeek-R1-Distill-Qwen-1.5B|DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf|https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf"
)

echo "==> Fleet root: $ROOT"
mkdir -p "$ROOT"

for entry in "${FLEET[@]}"; do
  IFS='|' read -r id file url <<<"$entry"
  dir="$ROOT/$id"
  dest="$dir/$file"
  mkdir -p "$dir"

  # Ask the CDN for the expected size so complete files are skipped cheaply.
  expected=$(curl -sIL "$url" | awk 'tolower($1)=="content-length:" {gsub(/\r/,""); n=$2} END {print n}')
  have=$([ -f "$dest" ] && stat -f%z "$dest" 2>/dev/null || stat -c%s "$dest" 2>/dev/null || echo 0)

  if [ -n "$expected" ] && [ "$have" = "$expected" ]; then
    echo "==> OK (complete)  $id/$file ($have bytes)"
    continue
  fi

  echo "==> Downloading    $id/$file ($have/${expected:-?} bytes)"
  curl -L -C - --retry 5 --retry-all-errors --progress-bar -o "$dest" "$url"
done

echo "==> Fleet complete:"
find "$ROOT" -name "*.gguf" -exec ls -lh {} \; | awk '{print "    " $5, $9}'
echo "==> Next: cargo test --test chat_fleet"
