# Changelog

All notable changes to **higgs** are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
higgs adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). The
version in `Cargo.toml` is the single source of truth — the CLI (`higgs
--version`), the git tag (`v<version>`), and the release artifacts all derive
from it.

## Release process

1. On a branch, bump `version` in `Cargo.toml` and move the items below from
   `[Unreleased]` into a new `## [x.y.z] - YYYY-MM-DD` section.
2. Open a PR to `main`. On merge, `.github/workflows/release.yml` reads the
   `Cargo.toml` version, and — if no `v<version>` tag exists yet — tags the merge
   commit, builds the binaries (macOS arm64 + Linux x86_64 CUDA), and publishes a
   GitHub Release whose body is this file's section for that version.

## [Unreleased]

## [0.1.0-beta.5] - 2026-08-08

### Fixed

- **gpt-oss / MXFP4 model metadata**: the model scanner's GGUF header reader
  (`ggus`) panicked on MXFP4 tensors — gpt-oss models cataloged with partial
  metadata (degraded autotune/fit estimates) and the node log filled with a
  repeating panic warning on every rescan. The scanner now uses `gguf-rs-lib`
  and reads only the header + metadata section, so models in any current or
  future quantization enrich fully and a single bad file can never spam the
  log or crash the scan.
- `install.sh` with `--pubkey` now fails on a missing `minisign` CLI BEFORE
  downloading the artifact, not after.

### Changed

- `higgs node install-service` output is human-readable: colored step results
  and an aligned quick-reference block (logs/state/status/stop) with
  advisories as separate marked paragraphs, instead of a wall of text.

## [0.1.0-beta.4] - 2026-08-06

### Added

- **Node connect diagnostics**: each failed hub dial now logs the FULL error
  cause chain plus an attempt counter, and roughly once a minute a "still
  unreachable — check:" block — on macOS including the Local Network
  permission recovery steps (the permission is per-binary, so a self-update
  can silently lose it). The hub serves the same platform-specific steps to
  the UI via a new `NodeView.offline_help` field, rendered on offline Fleet
  cards — nothing platform-specific is hardcoded client-side.

## [0.1.0-beta.3] - 2026-08-05

### Added

- **HF model-search catalog** (`catalog` module + jigglebot Model Search UI):
  browse mode (empty query → most-downloaded compatible GGUF repos), sort
  (downloads/likes/updated/trending), "fits this machine" filter with a
  per-quant-family footprint estimate (I-quant-aware), background real-size
  fallback when the Hub omits sizes, shard-aware default-quant preselect, and
  local + remote (fleet-node) downloads with full lifecycle logging on both
  the hub and node (`pull requested/starting/progress/done|FAILED` with
  repo/file/bytes/elapsed). Model-load phases are logged too, and the Servers
  tab shows every loaded model fleet-wide (remote worker pills + eject).

## [0.1.0-beta.2] - 2026-08-03

### Added

- **One-click fleet updates**: the jigglebot Fleet "Update" button now lists, on
  click only (never on a timer), every release newer than what the node runs
  (`node_releases` via the GitHub releases API — complete asset trio, upgrade
  only, newest first). Picking one sends the node a BARE version string
  (`M_NODE_UPDATE_VERSION`, new `update_by_version` HELLO capability); the node
  downloads manifest+signature+artifact itself from its own configured
  `release_url` (new `config.json` field, default = this repo's GitHub
  releases), re-verifies the CI minisign signature against its compiled-in
  keys, binds the authenticated manifest version to the requested one BEFORE
  the artifact download, and applies through the same verify → stage → flip →
  restart pipeline as every other update. `fleet_update_version` pushes to
  every capable node with honest per-node skip reasons. A pre-capability node
  is told precisely: on the latest → "nothing to update"; newer exists → re-run
  the installer (or a direct static-mirror manifest URL — GitHub links redirect
  and are refused by those builds).

- **Pairing preflight** (`higgs --node <ticket> <token>`): colored, gated
  self-diagnosis before dialing — per-nameserver DNS probes that name the exact
  dead resolver, ticket relay/direct-path analysis, and macOS Local Network
  guidance (with an SSH caveat: the permission prompt cannot appear over SSH).
  Pairing hard-stops only when no path to the hub can exist; every failure
  prints the specific user action that fixes it.
- **One-shot pairing with verified service handoff**: pairing saves the hub,
  best-effort restarts the installed service, and exits only after the hub
  demonstrably supersedes the pairing connection with the service's own dial
  (duplicate-identity close) — no Ctrl-C, no manual `launchctl kickstart`, never
  two node processes fighting, never zero. If no service takes over, pairing
  keeps serving in the foreground so the node stays online.
- An installed-but-unpaired node service now **waits quietly for pairing**
  (polls `config.json` every 3s; one hint line, then a reminder every ~5
  minutes) instead of exit/respawn log spam — and the wait doubles as the
  seamless handoff for a fresh install.
- Colored, user-consumable output for `install.sh` and the pairing flow
  (tty-gated ANSI; `NO_COLOR` disables).

### Changed

- Default log filter demotes iroh/transport/hickory-resolver internals to
  `error` (a `RUST_LOG` override still shows everything).
- A late service takeover while a paired foreground node is serving now exits
  that foreground process cleanly (it recognizes the hub's supersede close)
  instead of redialing into a duplicate-identity fight.

### Fixed

- Repeated pairing/preflight failures (environment problems: unreachable hub,
  dead DNS, corrupt config, malformed saved ticket) no longer spend the
  self-update rollback budget — only real boot crashes do.
- Ctrl-C/SIGTERM during pairing cancels immediately (never persists a hub or
  hands off afterwards) and exits nonzero so scripts cannot mistake a cancel
  for a successful pairing.

## [0.1.0-beta.1] - 2026-06-25

First public beta. A semver pre-release (`0.1.0-beta.1` < `0.1.0`); the release is
published as a GitHub pre-release.

### Added

- In-app local model runtime: OpenAI-compatible `/v1` serving over llama.cpp with
  a crash-isolated re-exec worker (`--higgs-worker`).
- Multi-model `NodeRuntime` (additive loads, one worker per model) with per-worker
  idle auto-unload.
- iroh QUIC fleet: hub/node pairing (`higgs link` / `higgs --node`), saved hubs,
  node self-retire, unified local-first node view, and per-node labels.
- `higgs --version` reports the crate version.
- CUDA build feature (`--features cuda`) for the Linux release.
