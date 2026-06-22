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
- **Integration gate excludes** `tool_parser/*` (pure logic, unit's job).
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
