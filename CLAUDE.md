# CLAUDE.md — higgs working agreement

Conventions an agent MUST follow when changing this crate. These are hard
requirements, not suggestions.

## No bespoke code when a popular crate exists (HARD RULE, ZERO TOLERANCE)

**If a well-adopted Rust crate does the thing, USE IT. Do not write your own.**
Not "consider it." Not "check first." USE THE CRATE. This is not negotiable.

Before implementing ANY mechanism (locking, dedup, atomicity, sync, cache,
retry, backoff, timeouts, file I/O primitives, concurrent maps, atomics,
observability, HTTP, TLS, JSON, checksums, path handling, temp files, glob,
crypto, hashing, base64, uuid, time, subprocess, watchers, config, CLI parsing,
templating, compression, semver, url, mime, etc.) check in this order and
STOP at the first hit:

1. `std` / `libc` / `tokio` / `parking_lot` — is the primitive already there?
2. A well-adopted crate (>100k downloads, >1 yr old) — the ecosystem has one
   for basically everything: `fs2`/`fd-lock` (flock), `dashmap` (concurrent
   map), `arc-swap` (RCU), `once_cell` (lazy statics), `bytes`, `blake3`,
   `sha2`, `hex`, `base64`, `uuid`, `tempfile`, `walkdir`, `notify`,
   `serde_yaml`, `toml`, `humantime`, `chrono`, `time`, `backoff`,
   `retry-policies`, `reqwest`, `hyper`, `tower`, `governor`, `indicatif`,
   `crossbeam`, `flume`, `regex`, `tracing`, `metrics`, etc.
3. A pattern already used elsewhere in this repo — but ONLY as a tiebreaker
   between two equally-good crate wrappers; a bespoke pattern already in the
   repo is not license to keep adding to it (it may itself be tech debt).

**Only invent custom if NO CRATE APPLIES and the commit message must justify
why.** "It's just a few lines" is not a justification. Bespoke wrappers with
`unsafe`, errno branching, and safety comments are exactly the boilerplate a
1-line crate call replaces.

**File locking = `fs2`. No exceptions.** Every file-lock in this crate goes
through `fs2::FileExt` (`try_lock_exclusive`, `lock_exclusive`,
`try_lock_shared`, `unlock`). NO raw `libc::flock` in new code, and existing
raw-flock sites (`ledger.rs`, `self_update.rs`) are TECH DEBT to migrate to
`fs2` when touched. If you find yourself writing `unsafe { libc::flock(...) }`
you are violating this rule.

**Also NEVER invent bespoke schemes when a primitive would make the problem
disappear.** Repeated pattern that WASTES CODEX ROUNDS: bespoke
residue-detection predicates / mtime heuristics / drain queues get built when
a per-key flock (via `fs2::try_lock_exclusive`) makes the whole
"detect residue" question unaskable. Codex then hammers the bespoke heuristic
across rounds and the loop converges on nothing. When codex clusters findings
on your custom mechanism, ask "what primitive would make this question
unaskable?" BEFORE iterating another heuristic.

## Agent concurrency (HARD RULE)

**No more than 2 subagents run at a time — ever.** Any Workflow or Agent
fan-out (reviews, test-writing, anything) executes as SEQUENTIAL PAIRS:
`parallel(chunk-of-2)` in a loop, never a wider fan-out. Parallel subagents
burn the session quota; a 6-wide fan-out can exhaust it mid-round and waste
everything in flight.

## Communication style (HARD RULE)

**No walls of text. Complex topics get a diagram, not paragraphs.** When
explaining a race, a control-flow decision, a two-process interaction, or any
"case A vs case B" comparison, lead with an ASCII diagram / table / boxed
scenario — prose is at most a couple of tight sentences framing it. Bullet
lists of scenarios stacked on top of each other are a wall of text in
disguise; convert them to a side-by-side box comparison. If the reader has to
scroll to see the shape of the thing, you failed. This applies inside
convergence-loop status messages too — a redesign-vs-residual question shows
the two shapes visually, not in bullet paragraphs.

## Convergence audit (HARD RULE)

For every finding a reviewer surfaces, before touching code answer four
questions in writing:

  1. Who calls this in production?
  2. Does that caller produce the input shape the finding needs?
     No → hypothetical → DISMISS, note it, next.
  3. If a hostile caller could produce it, does exploiting the finding
     grant them capability they don't already have on the same
     trust boundary?
     No → trust-model dismiss.
  4. Have I already dismissed the same CLASS this round?
     Yes → refer to the earlier dismissal, don't re-litigate.

Only if the answers survive 1–4 do I open an editor. Ratifying a
reviewer's finding without running this audit turns the convergence
loop from safety net into over-engineering amplifier.

## Match feature surface to user ask (HARD RULE)

Restate the user's ask in one sentence at the start of any new-feature
spec. Mark every field / knob / test I'm about to add as:

    NEEDED   the ask directly requires it
    NICE     arguable — flag it and ask before adding
    NOISE    I invented it — do not add

If it is not NEEDED, ask before it lands in code. "It might be useful"
is not NEEDED.

## Codex review convergence (required before any change is "done")

Every non-trivial change is reviewed with `codex` in a loop until it **converges**.

**The loop:**
1. Run a codex review of the change.
2. **Validate every finding yourself by reading the code.** Codex is fallible —
   confirm the claim is real before acting. Fix real findings; **dismiss false
   positives with a one-line evidence note** (cite the file:line that disproves
   it). Never apply a change you can't independently justify.
3. Re-run. Repeat.

**Reviews MUST be neutral — never bias the reviewer.** The codex prompt gives it
**facts only**: what the code does and the design/security model it operates in.
It **NEVER** pre-loads the reviewer to agree with you. Do **not** write "do not
re-flag X", "already fixed", "ASSESSED — NOT A BUG", "known false positive", or
any phrasing that tells codex what conclusion to reach or shields a judgment call
of yours from scrutiny. If anything, **explicitly invite codex to CHALLENGE the
exact decisions you are least sure of** (e.g. "argue whether this bootstrap
default is a lockout bug or correct", "is this Host allowlist a meaningful control
or security theater?"). A review told what not to find is not a review — it
launders your own opinion back to you. The ONLY steering allowed is scoping (which
files/change to look at) and the mechanical `Do NOT flag:` list of **proven**
tooling false positives below (AsyncWriteExt, Windows, `route.1.0`) — those are
facts about the language/platform, not verdicts about this change. You do the
validation and dismissal in step 2, from the code, on your own authority — the
reviewer's job is to find, not to ratify.

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
| **Integration** | `coverage-integration.sh` (the `tests/` targets) | **≥ 80% lines** |

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

- **Release branches receive ZERO commits — ever.** EVERYTHING lands on
  `develop` first: fixes, features, AND the version bump + changelog move. A
  `release/v*` branch is purely a cut vehicle — branch off `main`, merge
  `develop` in, PR to `main` — it is never committed to directly. A commit
  born on a release branch merges nowhere and silently misses every future
  release (this exact failure stranded the HG088 stream-to-disk fix on the
  beta.6 branch; beta.7/beta.8 shipped without it). Before cutting, verify the
  fixes the release claims are ancestors of `develop` (`git branch --contains
  <sha>`); after the release merges, sync `develop` from `main`.
- One logical unit per commit; conventional-commit subject.
- End commit messages with the `Co-Authored-By` trailer.
- Commit/push only when asked; branch off the default branch first if needed.

## Cutting a release (HARD RULE — use `scripts/release/cut-release.sh`, no other path)

The ONLY way `main` changes is a release PR. The ONLY way that PR gets built
is `scripts/release/cut-release.sh <x.y.z>`. Never hand-craft a release
branch, never edit `Cargo.toml`'s version by hand for a release, never
commit to `release/v*`.

**Prerequisite (must be true before you run the script):**

  1. Every feature this release ships is MERGED INTO `develop` (a commit
     stranded on a feature branch will NOT ship — that's how the HG088 fix
     missed beta.7 and beta.8).
  2. `develop`'s `git status` is clean (nothing uncommitted).
  3. `develop` is IN SYNC with `origin/develop` (pushed).
  4. `scripts/quality.sh` is green on `develop` — fmt + clippy + tests +
     ts-rs bindings match committed copies.

**Sequence:**

  1. On `develop`, add the release's changes to `CHANGELOG.md` under
     `## [Unreleased]` (Added / Changed / Fixed sections). Commit + push.
     The script rolls `[Unreleased]` into `## [x.y.z] - YYYY-MM-DD` at
     cut time — do not do that yourself.
  2. From ANYWHERE in the repo, run:
         scripts/release/cut-release.sh <x.y.z>
     (plain semver, no `v` prefix — e.g. `0.1.0-beta.11`). It checks out
     `main`, branches `release/v<x.y.z>` off it, merges `develop` in,
     bumps `Cargo.toml`'s version, rolls the CHANGELOG, pushes the
     release branch, opens the PR to `main`. `--dry-run` shows the plan
     without touching anything; `--no-verify` skips `scripts/quality.sh`
     (do NOT use unless the user explicitly asks).
  3. Review + merge the PR (you or the user — the script does not merge).
     `.github/workflows/release.yml` reads `Cargo.toml`'s version on the
     merged commit, tags `v<x.y.z>`, builds macOS arm64 + Linux x86_64 +
     Linux CUDA, signs manifests with the CI minisign key, and publishes
     the GitHub Release with 12 assets (tar.gz + sha256 + manifest +
     minisig per target). If any tag `v<x.y.z>` already exists the workflow
     no-ops — DO NOT try to reuse a version number.
  4. Watch for the release to appear (`gh release view v<x.y.z>`). Then
     SYNC `develop` FROM `main` and push:
         git checkout develop
         git fetch origin
         git merge origin/main -m "Merge main (v<x.y.z>) back into develop"
         git push origin develop
     Without this sync, the next release cut off `main` will re-merge the
     release bump commit as a change instead of the parent commit —
     eventually causing "hey where is my fix" surprises.

**Never:**

  - Commit directly to `release/v*` (a fix born there ships nowhere).
  - Push a release tag by hand (CI is the source of truth).
  - Bump `Cargo.toml`'s version outside `cut-release.sh`.
  - Rename or delete an already-published release (users' installers pin the
    tag; a rename breaks their update path).
  - Skip step 4 (the develop-from-main sync).
