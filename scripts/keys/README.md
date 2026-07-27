# scripts/keys — release signing-key management

Manages the **minisign key** that signs higgs release manifests. Rare, sensitive,
one-time-per-key operations — kept apart from the recurring release workflow in
[`../release/`](../release/).

| Script | Does |
|--------|------|
| `mint-keys.sh` | Mint the signing keypair (`minisign -G -W`), validate + append the public pin line to `.github/release-pubkeys.txt`, and print the manual steps (install the private key as the `MINISIGN_SECRET_KEY` secret; store it in a password manager). `--rotate` mints a new key alongside the old for a bridge release. |

The **private key never leaves your machine and is never committed** — only the public
pin line goes in the repo. Full context: [`../../RELEASING.md`](../../RELEASING.md)
Parts A and E.
