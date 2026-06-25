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
