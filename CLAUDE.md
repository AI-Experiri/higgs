# CLAUDE.md — higgs working agreement

Conventions an agent MUST follow when changing this crate. These are hard
requirements, not suggestions.

## Codex review convergence (required before any change is "done")

Every non-trivial change is reviewed with `codex` in a loop until it **converges**.

**The loop:**
1. Run a codex review of the change.
2. **Validate every finding yourself by reading the code.** Codex is fallible —
   confirm the claim is real before acting. Fix real findings; **dismiss false
   positives with a one-line evidence note** (cite the file:line that disproves
   it). Never apply a change you can't independently justify.
3. Re-run. Repeat.

**Converged =** results are **stable across 3 consecutive rounds** — i.e. either
three clean rounds, or three rounds that surface only the same already-assessed
items (known false positives / explicitly-deferred wont-fixes). A round that
finds a new real bug resets the count: fix it and keep going.

**An integration test must validate the fix.** Every real fix ships a `tests/`
integration test (spawn the real `higgs` over HTTP, or the in-process iroh gate)
that you've **proven fails if the fix is reverted** — verify by actually reverting,
running, seeing it fail, restoring; not by reasoning. **Commit the test as soon as
it's verified**, then converge. Write *testable code, not code for tests*: use
seams (dependency injection, pub fns, real HTTP) — never a test-only env/config/
`cfg(test)` knob in production. If a fix is genuinely not reachable from the
HTTP/iroh surface, say so with file:line evidence and cover it with a unit test via
an injected seam — do not fake an integration test.

**Tooling:** `codex review --base <ref>` / `--commit <sha>` do NOT accept a custom
prompt. For scoped, prompted reviews use `codex exec --skip-git-repo-check '<prompt>'`
(tightly scope it and tell it to CONCLUDE quickly, or it exhausts budget). Reviews
print the verdict twice; the block after `^codex` is the final one.

**Known codex false positives (do not "fix"):**
- "missing `AsyncWriteExt` import" for `SendStream::write_all`/`finish`/`stopped` —
  these are **inherent** iroh methods, not trait methods.
- Windows-rename / non-atomic-replace concerns — higgs targets **Unix/macOS** only
  (the llama.cpp FFI build isn't Windows-portable).
- `route.1 .0` "odd spacing" — required syntax: `route.1.0` lexes `1.0` as a float.

**Concurrency reviews:** the fleet/hub paths are concurrent; codex will keep
surfacing interleavings. Validate each against the actual lock scopes. A fix that
narrows a race window from "across an `.await`" to "a couple of instructions" and
degrades safely is acceptable when a full lock refactor isn't warranted —
document the residual.

## Test layout (where tests live)

- **Unit tests live in a SEPARATE sibling file**, never inline in the production
  file. For `src/<path>/<name>.rs`, the tests go in `src/<path>/<name>_tests.rs`,
  wired from the production file as a child module so private-item access is kept:

  ```rust
  // bottom of src/<path>/<name>.rs — the ONLY test line in a prod file:
  #[cfg(test)]
  #[path = "<name>_tests.rs"]
  mod tests;
  ```

  The `_tests.rs` file starts with `use super::*;`. The `#[path]` is REQUIRED —
  without it `mod tests;` would look for a `<name>/tests.rs` subdir. Keeping it a
  child `mod tests` (not a top-level module) is what preserves access to the
  module's private items. Do NOT write inline `#[cfg(test)] mod tests { … }`.
- **Integration tests live in `tests/` only** — one file per end-to-end area
  (spawn the real `higgs` over HTTP / the in-process iroh gate).
- **`mod.rs` files carry no logic or tests** — they are export barrels
  (`pub mod …;` / `pub use …;` + module docs); a module's own code lives in a
  named sibling file (e.g. `node/gate.rs`), whose tests are `node/gate_tests.rs`.
- `cargo llvm-cov` does NOT count `_tests.rs` files, so the unit gate below
  measures PRODUCTION lines only (moving tests out of the prod file is why the
  prod-only number is the true coverage, not an inline-test-inflated one).

## Coverage requirements (both gates must stay green)

Two independent line-coverage gates, run by `scripts/coverage.sh`
(`-u` unit only, `-i` integration only, no flag = both + combined summary):

| Gate | Command | Threshold |
|------|---------|-----------|
| **Unit** | `coverage-unit.sh` (`cargo test --lib`) | **≥ 90% lines** |
| **Integration** | `coverage-integration.sh` (the `tests/` targets) | **≥ 75% lines** |

- The gated metric is the **LAST `%` column** in `cargo llvm-cov` output (LINES),
  not regions.
- **Unit gate excludes** files that are inherently un-unit-testable and are the
  integration gate's job: `node/cli.rs`, `engine/llamacpp/mod.rs`, and the
  live-iroh async files `node/{mod,hub,fleet,transport,data}.rs` (their unit % is
  flaky from spawned tasks). New logic in those files is covered by integration
  tests, not the unit gate.
- **Integration tests spawn a real `higgs` process** and need a tiny GGUF
  (`HIGGS_TEST_GGUF`, else the on-disk default); they **skip** when it's absent, so
  integration coverage collapses — set the GGUF before measuring. Tear processes
  down with SIGTERM (coverage flush) and **never leave an SSE stream open** (it
  hangs graceful shutdown).

**New code must ship with tests that keep both gates green** — unit tests for
in-crate logic, an integration test in `tests/` for any new end-to-end path
(spawn → HTTP/iroh → assert). Run `scripts/coverage.sh` before calling a change done.

## Quality gate

`scripts/quality.sh` is the fast pre-commit gate and must pass:
`cargo fmt --all` (style pinned in `rustfmt.toml`) → `cargo clippy --all-targets -D
warnings` → `cargo test` → **ts-rs bindings sync** (the test pass regenerates
`bindings/higgs/*.ts`; the committed copies must match — a Rust wire-type change
must commit the regenerated TS).

## Commit discipline

- One logical unit per commit; conventional-commit subject.
- End commit messages with the `Co-Authored-By` trailer.
- Commit/push only when asked; branch off the default branch first if needed.
