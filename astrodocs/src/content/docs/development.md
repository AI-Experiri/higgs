---
title: Development
description: Local quality gate (fmt + clippy + test + ts-rs bindings sync), formatting policy, and the unit/integration coverage gates with their options.
---

This page covers the local workflow for contributing to higgs: the fast quality
gate, the formatting policy, and the coverage gates. higgs is a standalone Rust
crate — there is no frontend to build here (the TypeScript bindings it emits are
consumed by embedders).

## Quality gate

`scripts/quality.sh` is the fast pre-commit gate. Run it before every commit:

```sh
scripts/quality.sh
```

It runs these steps, accumulating failures and reporting them all at the end
(exit non-zero if any failed):

1. **Format** — `cargo fmt --all` (apply, then `--check` to verify). It auto-fixes
   trivial drift, so you rarely need to run `cargo fmt` yourself.
2. **Lint** — `cargo clippy --all-targets -- -D warnings` (warnings are errors).
3. **Test** — `cargo test`, the full suite. This pass also **regenerates** the
   ts-rs TypeScript bindings under `bindings/higgs/`.
4. **Bindings sync** — fails if `bindings/` changed after the test pass. A drift
   means a Rust wire type changed without committing the regenerated `.ts`.

## Formatting policy

The style is pinned in `rustfmt.toml` (`edition = 2021`, `max_width = 100`).
There is no separate hand-formatting convention — whatever `cargo fmt` produces
is the canonical form. Keep changes formatted (`cargo fmt --all`) and the gate
stays green.

## ts-rs bindings

Types that cross to a TypeScript consumer are wrapped with the `higgs_ts!` macro
(`src/ts_export.rs`), which injects ts-rs's `#[derive(TS)]` +
`#[ts(export, export_to = "higgs/")]`. ts-rs emits each file via a hidden,
derive-generated test, so **`cargo test` is what writes** `bindings/higgs/*.ts`.

The quality gate's bindings-sync step enforces that the committed `.ts` match the
Rust types. If you change a wire type, run `cargo test` (or `scripts/quality.sh`)
and commit the regenerated bindings together with the Rust change.

## Coverage

Two independent line-coverage gates, run separately by `scripts/coverage.sh`
(requires `cargo-llvm-cov`: `cargo install cargo-llvm-cov`):

| Gate | Command | Threshold | Scope |
|------|---------|-----------|-------|
| **Unit** | `coverage-unit.sh` (`cargo test --lib`) | **≥ 90% lines** | in-crate logic; excludes the daemon `main` + FFI (spawned-process only) |
| **Integration** | `coverage-integration.sh` (the `tests/` targets) | **≥ 75% lines** | end-to-end paths; excludes the pure-logic `tool_parser` subtree the unit gate owns |

```sh
scripts/coverage.sh              # both gates, then a combined summary
scripts/coverage.sh -u           # unit gate only          (--unit)
scripts/coverage.sh -i           # integration gate only   (--integration)
scripts/coverage.sh -u --open    # unit gate + open its HTML report
scripts/coverage.sh --html       # write HTML report(s) under target/
scripts/coverage.sh -h           # full usage
```

Any flag that isn't a selector (`-u`/`-i`) is forwarded verbatim to
`cargo llvm-cov` — e.g. `--html`, `--open`, `--json`, `--summary-only`,
`--output-dir DIR`.

When **both** gates run, both execute **even if the first fails** (no early
abort), and a combined summary prints at the end; the script exits non-zero if
any gate failed:

```
===== COVERAGE SUMMARY =====
  unit         90.99%   lines   PASS
  integration  76.72%   lines   PASS

All selected coverage gates passed.
```

## The test GGUF

Integration tests spawn the real `higgs` binary and drive it over HTTP + iroh
against a real ~1MB GGUF (ggml-org's `stories260K`), exercising
spawn → pair → load → chat → unload end to end. Override the path with
`HIGGS_TEST_GGUF`; when the file is absent these tests **skip** (and integration
coverage collapses below its gate, so set it before measuring coverage):

```sh
HIGGS_TEST_GGUF=/path/to/model.gguf scripts/coverage.sh -i
```
