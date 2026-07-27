# Releasing higgs

The clean path for shipping higgs: cutting a version, building **signed** artifacts,
installing on a remote box, and updating a running fleet.

higgs ships as **signed binary artifacts via GitHub Releases** — GitHub Actions builds,
minisign-signs, and publishes per-platform tarballs; nodes install with `install.sh` and
update themselves (`higgs node self-update`) or are pushed to from a hub. **This is the
only release mechanism we use.**

Publishing the library to **crates.io is out of scope** — not planned. The mechanism is
recorded for reference in [Appendix: crates.io (parked)](#appendix--cratesio-parked), but
it is not on the roadmap (and is blocked by git dependencies regardless).

Every step below is tagged by **who does it**:

| Tag | Meaning |
|-----|---------|
| 🔧 **manual** | Only you can/should do it (key material, secrets, PR merges, running on a box). |
| ⚙️ **script** | A script does it; you run it. Two sets: [`scripts/keys/`](scripts/keys/) (signing-key management) and [`scripts/release/`](scripts/release/) (cut + distribute a release). |
| 🤖 **CI** | GitHub Actions does it automatically; no human action. |

---

## At a glance

| # | Phase | Who | Command / action |
|---|-------|-----|------------------|
| **A1** | Mint + pin signing keys | ⚙️ + 🔧 | `scripts/keys/mint-keys.sh` → then paste the secret |
| **A2** | Enable releases | 🔧 | Configure the `release` environment (GitHub settings) |
| **B1** | Cut a version | ⚙️ | `scripts/release/cut-release.sh <x.y.z>` → opens a PR |
| **B2** | Build + sign + publish | 🤖 | *(merge to `main` triggers CI)* |
| **B3** | Mirror artifacts | ⚙️ | `scripts/release/mirror-assets.sh <x.y.z> <dest>` → serve it |
| **C1** | Install + service on a node | 🔧 | `./install.sh …` then `higgs node install-service …` |
| **C2** | Pair the node to a hub | 🔧 | mint a token, then `higgs --node <ticket> <token>` |
| **D1/D2** | Update nodes | 🔧 / UI | `higgs node self-update --url …` or the Fleet-tab push |
| **E** | Rotate keys (rare) | ⚙️ + 🔧 | `scripts/keys/mint-keys.sh --rotate` → bridge release |

> **Ordering is load-bearing.** A1 must ship a binary that **pins the public key**
> before any node can verify an update. While `.github/release-pubkeys.txt` is empty,
> CI **refuses to sign** — so releases are blocked until A1 is done. Today the pin file
> is empty and the whole feature branch is unmerged: start at A1/A2.

---

## Part A — One-time setup

### A1. Mint and pin the signing key ⚙️🔧

The release is signed with a [minisign](https://jedisct1.github.io/minisign/) key.
Its **private** half lives only in a GitHub secret; its **public** half is compiled
into every higgs binary (`.github/release-pubkeys.txt` → `src/update.rs`), so a node
trusts exactly the key it was built with.

```bash
scripts/keys/mint-keys.sh          # generates ~/.higgs/release-keys/minisign.{key,pub}
                                       # and appends the pin line to .github/release-pubkeys.txt
```

The script does the automatable, error-prone parts (keygen with `-W` / no password,
extracting the base64, formatting + validating the pin line). The private key is written
**outside the repo** (`~/.higgs/release-keys/` by default, or `$HIGGS_RELEASE_KEYS_DIR`;
the script refuses any `--out` inside the repo), so it can never be committed. It then
prints the **two manual steps only you can do**:

1. 🔧 Paste the **contents** of `~/.higgs/release-keys/minisign.key` into
   **GitHub → Settings → Environments → `release` → Environment secrets →
   `MINISIGN_SECRET_KEY`**.
2. 🔧 Store `minisign.key` in your password manager, then delete the local copy.
   **Never commit it** — only the public pin line in `.github/release-pubkeys.txt` is
   committed.

The pin line (`higgs-release-1 <base64>`) in `.github/release-pubkeys.txt` **is**
committed — it is public and is the trust root every binary ships with. CI cross-checks
that the pinned key equals the public half of `MINISIGN_SECRET_KEY` **at the released
commit** and refuses to sign otherwise, so the two can never drift.

### A2. Enable releases 🔧

**Configure the `release` environment** (GitHub → Settings → Environments → `release`):
- Add a **required reviewer** (this is the human gate on releasing; below GitHub
  Enterprise there is no reviewer enforcement, so treat write access as equivalent
  to holding the signing key).
- Set a **deployment branch/tag policy** allowing **both `main` and `v*` tags**
  (tag-dispatched retries need the `v*` allowance).

That is the only GitHub-settings step. The A1 pin line and all the feature work reach
`main` through the **[branch flow](#branch-flow)** in Part B — there is no separate
"push the setup" PR: the first `cut-release.sh` PR carries the pin, the feature (via
`develop`), and the version bump together, and merging it cuts the first release.

The repository can stay **private** — `install.sh` fetches releases with a fine-grained
PAT (`HIGGS_GITHUB_TOKEN`, Contents:read). Making it public is optional.

---

## Part B — Cutting a release

<a id="branch-flow"></a>
### Branch flow — the only way `main` changes

`main` is never edited directly. Every change reaches it as a **PR from a release branch
that was cut off `main` with `develop` merged into it**:

```
feature branch (cut from develop)         ← day-to-day work happens here
      │   Step 1 (manual): integrate onto develop
      │     git switch develop
      │     git merge --no-ff <feature>  &&  git push origin develop
      ▼
   develop
      │   Steps 2–6: cut-release.sh does all of this
      │     git switch main               ← clean baseline (never edited in place)
      │     git switch -c release/vX      ← branch OFF main
      │     git merge --no-ff develop     ← bring develop's code onto the branch
      │     bump Cargo.toml + Cargo.lock + CHANGELOG   ← rides IN the PR to main
      ▼
 release/vX ──PR──▶ main ──(you merge)──▶ 🤖 release.yml builds / signs / tags / publishes
```

- **Step 1 is yours** and deliberate: merge your feature branch into `develop` and push.
- **`cut-release.sh` automates Steps 2–6** — clean `main`, branch `release/v<version>`,
  merge `develop`, bump the version, open the PR. The version + CHANGELOG live on the
  release branch, i.e. **in the PR to `main`** (they can equally sit on `develop` before
  the cut — either is fine).
- **You merge the PR.** That push to `main` is the release trigger.

### B1. Cut the version ⚙️

With your feature already merged into `develop` (Step 1) and a clean working tree:

```bash
scripts/release/cut-release.sh 0.1.0-beta.2       # branch off main, merge develop, bump, open the PR
scripts/release/cut-release.sh 0.1.0 --dry-run    # show the plan, touch nothing
scripts/release/cut-release.sh 0.1.0 --from staging   # integrate a branch other than develop
```

`cut-release.sh` runs the whole branch flow plus the standard Rust crate-release hygiene:

1. Validates the version against CI's **exact** semver allowlist
   (`^[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.-]+)?(\+[0-9A-Za-z.-]+)?$`) and refuses if the
   `v<version>` tag already exists.
2. Checks out a clean `main`, branches `release/v<version>` off it, and merges `develop`
   in (`--no-ff`).
3. Bumps `[package] version` in `Cargo.toml` and refreshes `Cargo.lock`.
4. Rolls `CHANGELOG.md`: `## [Unreleased]` → a new `## [<version>] - <YYYY-MM-DD>` section
   (this becomes the GitHub Release notes).
5. Runs the quality gate (`scripts/quality.sh`, skip with `--no-verify`) and previews the
   release-check.
6. Commits, pushes `release/v<version>`, and opens the PR to `main` (skip with `--no-pr`).

> **You merge the PR.** The merge is the deliberate release trigger — the script never
> auto-merges.

### B1.5 The PR requirements check 🤖

Every PR to `main` runs `.github/workflows/release-check.yml`
(`scripts/release/check-release.sh`), which **mirrors `release.yml`'s gate and shifts it
left**: if the PR's version is untagged (so merging would cut a release), it enforces
valid semver, a dated `CHANGELOG` section, `Cargo.lock` in sync, and **a pinned signing
key** — otherwise it is a no-op pass. Make it a **required** status check in branch
protection so a release that would fail to build/sign can't merge. `cut-release.sh`
previews the same check before opening the PR.

### B2. Build, sign, publish 🤖

Merging the version bump to `main` triggers `.github/workflows/release.yml` — **the GitHub
Release action**, the one release mechanism we use:

- A **gate** job reads the `Cargo.toml` version and proceeds only if the `v<version>`
  tag does not yet exist.
- Three **build** legs (no cache, so only freshly-compiled bytes get signed):

  | OS | Target | Variant | Notes |
  |----|--------|---------|-------|
  | macOS 14 (arm64) | `aarch64-apple-darwin` | `metal` | system Metal, no runtime dep |
  | Ubuntu 22.04 | `x86_64-unknown-linux-gnu` | `cpu` | the universal build |
  | Ubuntu 22.04 | `x86_64-unknown-linux-gnu` | `cuda` | needs a CUDA runtime to run |

- The **release** job mints a canonical JSON manifest per artifact and minisign-signs it
  (signature covers the *manifest*, which binds `version/target/variant/sha256`), then
  publishes a **GitHub Release** under tag `v<version>` and **creates the tag itself**.
  Per platform, 4 assets (12 total):
  `higgs-v<ver>-<suffix>.tar.gz`, `.tar.gz.sha256`, `.manifest`, `.manifest.minisig`.
- Pre-releases (a version containing `-`, e.g. `0.1.0-beta.2`) publish as GitHub
  pre-releases.

**Retry** (only recovers a run that failed *before* uploading — published assets are
immutable and a partial upload burns the version, so bump the patch instead):

```bash
gh workflow run release.yml                 # first run / pre-upload retry (on main)
gh workflow run release.yml --ref v0.1.0    # retry an already-tagged version (on the tag)
```

### B3. Mirror artifacts for remote update ⚙️

GitHub's `…/releases/download/…` URLs **do not work** for `higgs node self-update` or the
hub courier: GitHub 302-redirects release assets to storage and the storage URL carries a
query string, both of which the SSRF-hardened fetcher rejects (→ HG088). Mirror the signed
assets to a **direct static HTTPS origin** where `.manifest`, `.minisig`, and `.tar.gz` are
sibling files:

```bash
scripts/release/mirror-assets.sh 0.1.0 ./mirror     # downloads → ./mirror/v0.1.0/
# then serve ./mirror so this resolves with NO redirect and NO query string:
#   https://<your-origin>/higgs/v0.1.0/higgs-v0.1.0-aarch64-apple-darwin.manifest
```

The last path segment of the URL you hand to nodes must be the `v<version>` directory.

---

## Part C — Install & run on a node

> Full detail (all flags, root rules, headless caveats) lives in `src/node/README.md`
> and `docs/DESIGN-remote.md`. This is the clean path.

### C1. Install the binary, then the service 🔧

Run **as the operator user, never `sudo`**, and invoke `./install.sh` directly (not
`bash install.sh`, so its `-p` shebang takes effect):

```bash
# from a private-repo release (verifies sha256; --pubkey also verifies the signature):
HIGGS_GITHUB_TOKEN=github_pat_… ./install.sh --pubkey <base64>     # add --version <x.y.z> to pin; --cuda on Linux
# or from a scp'd artifact (needs the .tar.gz.sha256 sidecar beside it):
./install.sh --tarball higgs-v0.1.0-aarch64-apple-darwin.tar.gz
```

This lands the binary at `~/.higgs/bin/v<ver>/higgs`, flips `~/.higgs/bin/current`, and
**prints the exact `install-service` command** to run next. Then enable the service:

```bash
# login-bound (starts at login, no sudo — both OSes):
~/.higgs/bin/current/higgs node install-service --prefix ~/.higgs --higgs-home ~/.higgs
# always-on:  macOS →  sudo … --system      Linux →  … --system   (adds enable-linger)
```

Add `--dry-run` to preview the unit/plist without touching anything. Root rules are
inverted by scope: login-bound and Linux `--system` **refuse** root; only macOS
`--system` **requires** `sudo`.

### C2. Pair the node to a hub 🔧

The service runs bare `higgs --node`, which only *reconnects* to an already-saved hub —
so pair once first:

1. On the **hub** (your embedded higgs / jigglebot): enable hub mode (Fleet tab
   "enable hub", i.e. `Higgs::hub_enable()`), then mint a single-use token
   (`Higgs::pair()` → `{ticket, token, node_command}`). Standalone alternative:
   `higgs link pair` does both and prints the exact node command.
2. On the **node**, run the printed command once:
   ```bash
   higgs --node <ticket> <token>          # first join; saves the hub to config.json
   ```
   After the first admission the node reconnects by its stable identity with no token —
   so the **service** takes over from here. Confirm on the hub (Fleet view / `hub_status`);
   the node prints `paired with hub <name> …`.
3. Load a model onto it from the Fleet UI (`Higgs::node_load`). A `/v1/chat/completions`
   to that served id then routes hub → node → worker automatically.

---

### C3. Uninstall (clean teardown) 🔧

To remove the node service with no leftovers **except** `~/.higgs` (identity, saved hubs,
models, logs) — e.g. to reset a test box:

```bash
./uninstall.sh              # stop + unload the launchd job (or systemd unit) + remove its plist; keeps ~/.higgs
./uninstall.sh --dry-run    # preview exactly what would be removed
./uninstall.sh --purge      # ALSO delete ~/.higgs (destroys endpoint.key / pairings / models)
```

macOS removes the `com.higgs.node` LaunchAgent (and LaunchDaemon if `--system` was used;
that step needs `sudo` and is printed for you). Linux removes the `higgs-node.service` user
unit and any enable-linger. To also de-register the node from its hub first, run
`higgs node leave` before uninstalling.

## Part D — Updating nodes

A node can only be updated once it runs a **release build with a pinned key** (a dev build
pins nothing and fails closed at HG081). Updates are always **upgrade-only**; rolling back
is the node's own job.

### D1. Node self-update (per node) 🔧

```bash
higgs node self-update --url https://<origin>/higgs/v0.1.0/higgs-v0.1.0-aarch64-apple-darwin.manifest
# then restart the node service; or, if you curl the triple yourself:
higgs node self-update --tarball higgs-….tar.gz --manifest higgs-….manifest --manifest-sig higgs-….manifest.minisig
```

Flow: verify signature vs the compiled-in key → eligibility (target/variant match,
strictly newer) → sha256 → stage → smoke-test the staged `--version` → flip `current` →
you restart → boot-guard auto-rolls-back if the new binary crash-loops (3 boots).

### D2. Hub push (whole fleet) — Fleet UI

From the jigglebot Fleet tab (crate API `Higgs::node_update(node, url)` /
`Higgs::fleet_update(base_url)` — there is no `higgs` binary subcommand for this). The
`base_url`'s last path segment must be the `v<version>` mirror directory. Each node
**re-verifies against its own pinned key**, applies detached, and re-execs. The push reply
is only a delivery receipt — the authoritative outcome is the node's **next HELLO**
(software version advanced = success; `update_failed{from,to,reason}` surfaced in the Fleet
view = failure).

### D3. Rollback 🔧

```bash
higgs node self-update --rollback     # repoint current → the recorded previous version
higgs node self-update --prune        # drop old version dirs, keep current + rollback target
```

The boot-guard also rolls back automatically after 3 failed boots and **poisons** the
crash-looping version so a re-push of that exact version is refused until a fresh
`install.sh` clears it.

---

## Part E — Key rotation (rare)

Rotation must not lock out deployed nodes, which pin only the **old** key:

```bash
scripts/keys/mint-keys.sh --rotate     # mints a new key, pins OLD + NEW in release-pubkeys.txt
```

1. Ship a **bridge release** whose binary pins **both** keys, still **signed with the OLD
   key** (deployed nodes only trust the old key, so a bridge signed with the new key would
   be rejected everywhere). `MINISIGN_SECRET_KEY` stays the old key for this release.
2. Nodes update *through* the bridge and pick up the new pin.
3. **Only after** the fleet is on the bridge, switch `MINISIGN_SECRET_KEY` to the new key.
   Later releases sign with the new key; a subsequent release may drop the old pin.

---

## Appendix — crates.io (parked)

We do **not** publish higgs to crates.io — the git-release mechanism above is the only one
we use, and there are no plans to change that. This section is kept only as a record of the
mechanism. It is also not currently possible: `cargo publish` refuses because **crates.io
forbids `git` and `path` dependencies**, and higgs has three git deps:

- `llama-cpp-2` and `llama-cpp-sys-2` — the AI-Experiri fork (restores the oaicompat chat
  API). Blocker: the fork must be published to crates.io, or its changes upstreamed and
  higgs moved back to the upstream crate version.
- `huggingface-hub` — a pre-1.0 git dependency ("not on crates.io"). Blocker: wait for a
  crates.io release, or drop it in favour of the `reqwest` fallback path.

A disabled workflow scaffold already exists: `.github/workflows/crates-io.yml` runs only on
manual dispatch and its publish job is guarded by the repo variable `CRATES_IO_PUBLISH`
(unset → skipped). Nothing publishes until that variable is set to `true` and the git-dep
blocker is resolved.

When the deps are unblocked, the standard crate-publish path is:

1. Publish `higgs-macros` first (it is a `path` workspace member; a dependent can't publish
   before its path deps are on crates.io). Give it real metadata and `cargo publish` it.
2. Complete higgs `[package]` metadata for a good listing / docs.rs:
   `repository`, `readme = "README.md"`, `keywords`, `categories`, `rust-version` (declare
   the MSRV), and add a `LICENSE` file (the manifest already declares `license = "MIT"`).
3. Consider `publish = false` on any member you do **not** want on crates.io, and an
   `[package.metadata.docs.rs]` block if the FFI build needs docs.rs feature flags.
4. Dry-run, then publish:
   ```bash
   cargo publish -p higgs-macros --dry-run   &&   cargo publish -p higgs-macros
   cargo publish --dry-run                    &&   cargo publish
   ```
5. Recovery: `cargo yank --version <x.y.z>` pulls a bad release from the index (it does not
   delete it). A crates.io API token (`cargo login`) is required.

Note: the binary/artifact channel (Parts A–E) is independent of crates.io and stays the
delivery mechanism for running nodes regardless.

---

## Appendix — GitHub Actions workflows

| Workflow | Trigger | Does |
|----------|---------|------|
| `ci.yml` | PR + push `main` | fmt / clippy `-D warnings` / test / ts-rs bindings sync. |
| `release-check.yml` | PR to `main` | Runs `check-release.sh` — release requirements gate (see B1.5). |
| `release.yml` | push `main` + manual | **GitHub Release** — build (3 legs) → sign → publish → tag. |
| `crates-io.yml` | manual only | **crates.io publish — DISABLED.** A parked scaffold; the job is guarded by the repo variable `CRATES_IO_PUBLISH` (unset → skipped) and blocked by git deps. Enable per the [crates.io appendix](#appendix--cratesio-parked). |

## Appendix — standard Rust crate-release checklist

`cut-release.sh` automates the ones marked ⚙️; the rest are inherent to the flow above.

- ⚙️ Bump `[package] version` (semver).
- ⚙️ Refresh `Cargo.lock`.
- ⚙️ Update `CHANGELOG.md` (`[Unreleased]` → `[x.y.z] - date`).
- ⚙️ Run the quality gate (`fmt` / `clippy -D warnings` / `test` / ts-rs bindings sync).
- 🤖 Build release binaries + sign + create the `v<version>` tag + GitHub Release.
- 🔧 Mirror signed assets to a static origin (⚙️ `mirror-assets.sh` does the fetch).
- ⏸️ crates.io publish — **not planned** (see the [crates.io appendix](#appendix--cratesio-parked)).

## Appendix — update error codes

`HG081` unknown/unpinned key · `HG082` bad signature · `HG083` bad/unknown-schema manifest ·
`HG084` artifact sha256 mismatch · `HG085` not newer (downgrade/same refused) ·
`HG086` target/variant mismatch · `HG087` apply/stage/smoke/unmanaged-install failure ·
`HG088` fetch failure (non-https, redirect, query, timeout, or size cap).
