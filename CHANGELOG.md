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

### Added

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
