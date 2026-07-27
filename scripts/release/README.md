# scripts/release — cut and distribute a release

The recurring per-version release workflow. Signing-key management lives separately in
[`../keys/`](../keys/).

| Script | Does |
|--------|------|
| `cut-release.sh <x.y.z>` | Cut a release the only way `main` changes: check out clean `main`, branch `release/v<x.y.z>` off it, merge `develop` in (`--from` overrides the integration branch), bump `Cargo.toml` + `Cargo.lock`, roll `CHANGELOG.md` (`[Unreleased]` → `[x.y.z] - date`), run the quality gate, preview the release-check, then commit + push + open the PR to `main`. Merging the PR triggers CI to build, sign, and publish. **Requires your feature already merged to `develop`.** `--dry-run` / `--no-verify` / `--no-pr` / `--from <branch>`. |
| `check-release.sh` | Validate that a PR which would cut a release is well-formed — valid semver, dated `CHANGELOG` section, `Cargo.lock` in sync, a pinned signing key. Mirrors `release.yml`'s gate; run by `.github/workflows/release-check.yml` on every PR (and previewed by `cut-release.sh`). No-op pass when the version is already tagged. |
| `mirror-assets.sh <x.y.z> [dest]` | Download the signed release assets into a `v<x.y.z>/` layout for a static HTTPS origin, so `higgs node self-update` and the hub courier can fetch them (GitHub's redirecting download URLs don't work for that). `--verify` re-checks hashes. |

Full context: [`../../RELEASING.md`](../../RELEASING.md) Part B.
