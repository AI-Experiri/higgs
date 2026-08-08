#!/bin/bash -p
# install.sh — install a higgs release on this machine (macOS arm64 / Linux x86_64).
# Shebang is the ABSOLUTE /bin/bash (present on both macOS and Linux), NOT
# `/usr/bin/env bash`: the root-safety guard below must not be preceded by a
# PATH-resolved interpreter a poisoned PATH could hijack under sudo. `-p`
# (PRIVILEGED mode) is critical for the PAT: it makes bash IGNORE `$BASH_ENV`/
# `$ENV` (arbitrary code that would run before the root guard) and NOT import
# shell FUNCTIONS from the environment (a planted `uname`/`curl` function would
# run IN this shell and could read the non-exported `_gh_token`), and ignore
# `SHELLOPTS`/`BASHOPTS`. This is NOT applied when invoked as `bash install.sh`
# (only `./install.sh` / a direct exec honors the shebang) — the integration
# tests exec it directly to exercise it.
#
# Two ways to get the artifact:
#   1) A tarball you already have (scp'd from a release, or a local build):
#        ./install.sh --tarball higgs-v0.2.0-aarch64-apple-darwin.tar.gz
#   2) Straight from the private GitHub repo over REST, using a FINE-GRAINED
#      personal access token with ONLY "Contents: read" on AI-Experiri/higgs:
#        HIGGS_GITHUB_TOKEN=github_pat_… ./install.sh            # latest release
#        HIGGS_GITHUB_TOKEN=github_pat_… ./install.sh --version 0.2.0
#      The token is read from the environment, sent only to api.github.com, and
#      never written to disk; it is moved into a non-exported shell var and its
#      env var UNSET before the FIRST subprocess, so no child ever inherits it.
#      Do NOT
#      `gh auth login` on a worker node — that persists a broad credential; a
#      scoped PAT in the environment does not. The REST path needs `jq` (a
#      hand-rolled JSON scrape would break on a minified/reordered response).
#
# What it does:
#   - picks the artifact for THIS machine (uname → target suffix; --cuda opts
#     the Linux box into the CUDA build),
#   - verifies the .sha256 (and, with --pubkey, the minisign-signed update
#     manifest AND that its version/file/target/variant match what was asked
#     for — a signed OLD manifest cannot be replayed under a new name),
#   - unpacks in a STAGING dir, then renames into <prefix>/bin/v<version>/
#     (a reinstall never truncates a running binary). install.sh NEVER RUNS
#     the artifact — it only verifies + places; the binary executes only when
#     the operator installs the service. Use --pubkey for untrusted sources.
#   - ATOMICALLY flips the <prefix>/bin/current symlink to the new version dir
#     (create-temp + rename(2); `ln -sfn` is unlink+create — a crash between
#     the two leaves NO current at all, so it is not used),
#   - keeps every previously installed version dir untouched (rollback =
#     re-run with --version <old>, which just re-flips `current`).
#
# `current` is the single authority: put <prefix>/bin/current on PATH and point
# service units at <prefix>/bin/current/higgs — an update or rollback then
# never touches PATH or the unit files.
#
# Flags:
#   --tarball <path>    install from a local tarball (skips GitHub entirely)
#   --version <semver>  release version to fetch (default: latest)
#   --prefix <dir>      install root (default: $HOME/.higgs)
#   --cuda              on Linux, pick the CUDA artifact instead of CPU
#   --pubkey <base64>   minisign public key (the base64 line of minisign.pub);
#                       when set AND the minisign CLI is installed, the signed
#                       update manifest is verified — a hard failure on mismatch
#   --repo <owner/name> GitHub repo (default: AI-Experiri/higgs)
#   -h | --help         this text

set -euo pipefail

# =============================================================================
# FUNCTION MAP  (maintainer reference — `--help`/`usage` prints only the file
# header above this line, so this overview lives here, out of the user help.)
#
#   die                       Print "install.sh: ERROR: <msg>" to stderr, exit 1.
#   note                      Print an informational "install.sh: <msg>" to stderr.
#   usage                     Print the file header above as the --help text.
#   need_val                  Require a value-taking flag to be followed by a
#                             value (not another flag or end-of-args).
#   _has_write_acl_for_peer   macOS: true if a directory's ACL grants write to a
#                             user other than the operator (catches grants the
#                             mode bits miss). Shared by the TMPDIR + exec walks.
#   _refuse_renamable         Walk a $TMPDIR ancestry and die if any level is
#                             owned/writable by a peer who could rename the
#                             integrity workdir and swap the unverified download.
#   cleanup                   EXIT trap: remove the workdir, stage dir, temp link.
#   gh_api                    curl the GitHub REST API with the PAT fed via a pipe
#                             (the token never appears in argv or on disk).
#   fetch_asset               Download one named release asset into the workdir.
#   sha256                    SHA-256 of a file via a trusted absolute tool
#                             (/usr/bin/sha256sum or /usr/bin/shasum).
#   mfield                    Extract one "key":"value" field from the signed
#                             update manifest.
#   _refuse_ww                Die if a directory on the install path is world-
#                             writable (trusted-sticky ancestors are exempt).
#   abspath                   Resolve a possibly-relative dir to an absolute path
#                             against the physical cwd.
#   reject_dotdot             Die if a resolved path contains a '..' component.
#   _gw_check_dir             Set _exec_gw if one directory is group-writable or
#                             carries a peer-write ACL.
#   _gw_chain                 Run _gw_check_dir up one ancestry chain to '/'.
#   _gw_walk                  Check the lexical AND physical ancestry chains;
#                             gates the printed --system service advice.
# =============================================================================

# Disable `allexport` (a caller's `set -a`, possibly inherited via SHELLOPTS):
# otherwise every plain assignment below — including the PAT capture — would be
# auto-exported into the environment of subprocesses.
set +a
# Scrub env vars that our TRUSTED helpers themselves honor for CODE EXECUTION or
# secret leakage — `-p` neutralizes bash's own (BASH_ENV/ENV/SHELLOPTS) but not
# these tool-specific ones, so a poisoned environment could otherwise run code
# after verification or exfiltrate the PAT:
#  - SSLKEYLOGFILE: curl (OpenSSL/GnuTLS) writes TLS SESSION SECRETS to it, from
#    which the PAT's request could be decrypted off a packet capture.
#  - TAR_OPTIONS: GNU tar PREPENDS it to argv, and `--checkpoint-action=exec=…`
#    runs an arbitrary command during extraction of the (already sha-verified)
#    tarball.
#  - PERL5OPT / PERL5LIB / PERL5DB / PERLLIB: perl (used for the atomic
#    publish rename AND as `shasum`) loads modules / runs code from them
#    (`PERL5OPT=-Mevil`) — code execution right at the verify/publish step.
#  - CURL_CA_BUNDLE / SSL_CERT_FILE / SSL_CERT_DIR: REPLACE curl's trust anchor. A
#    poisoned env pointing these at an ATTACKER CA lets a MITM proxy present a
#    forged api.github.com cert that VALIDATES — stealing the PAT and (on a
#    no-`--pubkey` install) serving a tarball+sha the check accepts. Certificate
#    validation does NOT protect the token once the trust root is attacker-chosen.
#  - OPENSSL_CONF / OPENSSL_MODULES: OpenSSL loads an arbitrary provider/engine DSO
#    named there — code execution inside the very curl process that holds the PAT.
# (Plain PROXY vars — HTTPS_PROXY/HTTP_PROXY/ALL_PROXY/NO_PROXY — are LEFT intact:
# corporate fleet nodes legitimately need them, and with the REAL CA a plain proxy
# still can't forge a valid cert, so the PAT stays protected. Only the trust-anchor
# and code-load overrides above are the MITM/code-exec vectors, so only they are
# scrubbed.)
unset SSLKEYLOGFILE TAR_OPTIONS PERL5OPT PERL5LIB PERL5DB PERLLIB \
      CURL_CA_BUNDLE SSL_CERT_FILE SSL_CERT_DIR OPENSSL_CONF OPENSSL_MODULES

# PREPEND the trusted, root-owned system dirs to PATH before ANY subprocess
# runs. `-p` does not reset PATH, so a caller's `PATH=~/evil:...` could otherwise
# resolve a bare helper to a PLANTED binary. That matters beyond code-exec:
# `unset HIGGS_GITHUB_TOKEN` below removes the PAT from bash's LIVE environment
# (so no child inherits it), but on Linux the token still sits in this process's
# INITIAL environment, readable by a same-user child via `/proc/$PPID/environ`
# (that file is the execve snapshot and does NOT reflect a later unset). With the
# system dirs FIRST, every helper that HAS a system copy — `mkdir`/`uname`/
# `mktemp`/`shasum`/`sha256sum`/`tar`/`perl`/`od`/`rm` — resolves to the trusted
# one, so no planted copy of them ever runs. We PREPEND rather than REPLACE
# because `jq` (REST path) and `minisign` (--pubkey) are commonly installed under
# `/opt/homebrew/bin`/`/usr/local/bin`, which a system-only PATH would hide.
# Residual: a planted `jq`/`minisign` earlier than its real install could still
# run; that narrower window (two tools, admin-owned dirs) is the accepted
# trade-off for tool discovery. (curl is separately pinned to /usr/bin/curl.)
PATH=/usr/sbin:/usr/bin:/sbin:/bin:${PATH}
export PATH

# Pin a bounded umask so EVERY dir/file this script creates — the prefix, `bin/`,
# the version dir, `logs/`, the staged files — is at most owner-writable
# (dirs 0755, files 0644), regardless of a permissive caller umask. Under
# `umask 000` the install tree would otherwise be WORLD-WRITABLE, and on a
# multi-user machine any local user could replace `bin/current`, the binary, or
# the linked unit and have code run as the operator at the next restart/login.
# The explicit per-file 0644 chmods do NOT protect entries inside a
# world-writable parent directory.
umask 022

# install.sh is an OPERATOR-context tool: it installs into the operator's own
# ~/.higgs and must NOT run as root. Running it via sudo would create
# root-owned files in the operator's tree AND open a symlink-follow privesc
# (staging lives under the operator-writable bin/, and a pre-created
# `v<version> -> /etc` would be written through as root). Only `higgs node
# install-service` needs elevation (a macOS LaunchDaemon), and it drops back
# to the operator for anything under the home.
# Use bash's builtin `$EUID` (no subprocess) — NOT `$(id -u)`: a planted `id`
# on a preserved sudo PATH could report a non-root uid and slip past the guard.
[ "$EUID" -ne 0 ] \
  || { echo "install.sh: ERROR: run as the operator, not root (do NOT use sudo) — it installs into your own ~/.higgs; only 'higgs node install-service' needs sudo, on macOS" >&2; exit 1; }

# Refuse a broken HIGGS_HOME / HIGGS_MODEL_DIR (matching install-service's own
# refusals so the Next-steps command it prints is always runnable):
#  - SET-but-EMPTY: the higgs runtime resolves "" as a RELATIVE (cwd-dependent)
#    path; silently substituting `$HOME/.higgs` (via `${VAR:-…}`) would pin the
#    service to a DIFFERENT store than the node paired against → restart loop.
#    `${VAR+set}` is non-empty only when VAR is SET (even to ""), distinguishing
#    set-but-empty from unset (unset is fine — defaults apply).
#  - a `..` COMPONENT: install-service's dir flags REJECT `..` (systemd keeps it
#    verbatim and silently misdirects `append:` logs), so a `..`-bearing value
#    would print a Next-steps command that FAILS the CLI's own validation. Refuse
#    it here too, early and loud. (`case` matches a leading, embedded, or sole
#    `..` path component.)
for _v in HIGGS_HOME HIGGS_MODEL_DIR; do
  # The SET-but-EMPTY refusal applies ONLY to HIGGS_HOME: the runtime resolves an
  # empty HOME as a broken relative store path (a restart-loop), so it is
  # ambiguous and refused. HIGGS_MODEL_DIR is an OPTIONAL extra scan root — the
  # runtime AND install-service (cli.rs `.filter(|v| !v.is_empty())`) both treat
  # empty as UNSET (there is no default to conflict with), so an empty value is
  # ignored here too, matching them (a provisioning env may export it blank).
  if [ "$_v" = HIGGS_HOME ] && [ -n "${!_v+set}" ] && [ -z "${!_v}" ]; then
    echo "install.sh: ERROR: HIGGS_HOME is set but EMPTY — unset it, or export a real directory (the higgs runtime would resolve an empty value as a broken relative path)" >&2
    exit 1
  fi
  case "/${!_v:-}/" in
    */../*)
      echo "install.sh: ERROR: ${_v} contains a '..' component — export a resolved absolute path (a '..' would make the printed 'install-service' command fail its own validation and could misdirect the daemon's logs)" >&2
      exit 1 ;;
  esac
done
unset _v
# Validate the state-dir FALLBACK too, and do it HERE — BEFORE any download,
# publish, or `current` flip. The env-var loop above misses the `$HOME/.higgs`
# fallback when HIGGS_HOME is unset and `$HOME` itself carries a `..`; catching
# it only at Next-steps (after publication) would flip `current` to the new
# binary and THEN abort, leaving the "rejected" version live. Refuse up front so
# a broken state dir touches nothing. (Detection needs only the raw value — a
# relative dir is absolutized later against a clean cwd, adding no `..`.)
_sd="${HIGGS_HOME:-${HOME:+${HOME}/.higgs}}"
case "/${_sd}/" in
  */../*)
    echo "install.sh: ERROR: the state dir '${_sd}' contains a '..' component (from \$HOME) — export a resolved absolute HIGGS_HOME, or fix \$HOME (a '..' would make the printed 'install-service' command fail its own validation)" >&2
    exit 1 ;;
esac
unset _sd

# Capture the PAT into a NON-EXPORTED shell var and DROP the environment
# variable NOW — before the very first child process. An exported env var is
# inherited by EVERY subprocess (uname, mktemp, jq, sed, …); a binary planted
# on the PATH could read and exfiltrate it. A plain shell var is not inherited
# by children — only the `printf | curl -K -` pipe below (run by this shell)
# ever expands it. xtrace is silenced around the value so `bash -x` can't print
# it (the `2>/dev/null` also hides the trace of the `set +x` itself).
{ _xtrace_on=0; case "$-" in *x*) _xtrace_on=1;; esac; set +x; } 2>/dev/null
_gh_token="${HIGGS_GITHUB_TOKEN-}"
# Explicitly clear the export attribute: bash PRESERVES it if `_gh_token` was
# already exported in the inherited environment (so the assignment above would
# otherwise re-export the PAT). Belt-and-suspenders with `set +a` above.
export -n _gh_token 2>/dev/null || true
unset HIGGS_GITHUB_TOKEN
[ "$_xtrace_on" = 1 ] && set -x

REPO="AI-Experiri/higgs"
# Empty SENTINEL, not `${HOME}/.higgs`: expanding `$HOME` here would abort under
# `set -u` when HOME is unset (a scrubbed provisioning env) even for `--help` or
# an explicit `--prefix` that never needs it. The default is resolved AFTER arg
# parsing (below), and only then does the default-prefix case require HOME.
PREFIX=""
TARBALL=""
VERSION=""
CUDA=0
PUBKEY="${HIGGS_MINISIGN_PUBKEY:-}"

# fn die() — print an error to stderr and exit 1.
die() { echo "install.sh: ERROR: $*" >&2; exit 1; }
# fn note() — print an informational message to stderr.
note() { echo "install.sh: $*" >&2; }

# ── Color (visual clarity only) ────────────────────────────────────────────────
# Enabled ONLY when BOTH stdout and stderr are a terminal, and NO_COLOR is unset:
# piped/captured output (tests, provisioning logs, `> file`) stays byte-plain, so
# nothing that greps this script's output ever sees an escape code.
if [ -t 1 ] && [ -t 2 ] && [ -z "${NO_COLOR+set}" ]; then
  C_BOLD=$'\033[1m'; C_GREEN=$'\033[32m'; C_CYAN=$'\033[36m'; C_YELLOW=$'\033[33m'; C_OFF=$'\033[0m'
else
  C_BOLD=""; C_GREEN=""; C_CYAN=""; C_YELLOW=""; C_OFF=""
fi
# A verified/completed milestone — green with a check mark.
ok() { echo "install.sh: ${C_GREEN}✓ $*${C_OFF}" >&2; }

# fn usage() — print the file header (top of file) as the --help text.
usage() { sed -n '2,/^set -euo/p' "$0" | sed 's/^# \{0,1\}//' | sed '$d'; }

# fn need_val() — require a value-taking flag to be followed by a value.
# A value-taking flag must be followed by a VALUE, never another flag —
# `--prefix --cuda` (and `--version -h`) must error, not swallow the flag.
# Rejects anything starting with `-` (both `--long` and `-h`); a real path
# value that must start with `-` can be written `./-foo`.
need_val() { # $1 = flag name, $2 = the next arg (may be unset)
  case "${2-}" in
    "") die "$1 needs a value" ;;
    -*) die "$1 expected a value but got the flag '$2' — a value is missing" ;;
    *)  printf '%s' "$2" ;;
  esac
}

while [ $# -gt 0 ]; do
  case "$1" in
    --tarball) TARBALL="$(need_val --tarball "${2-}")"; shift 2 ;;
    --version) VERSION="$(need_val --version "${2-}")"; shift 2 ;;
    --prefix)  PREFIX="$(need_val --prefix "${2-}")"; shift 2 ;;
    --cuda)    CUDA=1; shift ;;
    --pubkey)  PUBKEY="$(need_val --pubkey "${2-}")"; shift 2 ;;
    --repo)    REPO="$(need_val --repo "${2-}")"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown flag: $1 (see --help)" ;;
  esac
done

# PREFLIGHT: fail on a missing verification tool NOW, before any network or
# download work — discovering it only at verify time wastes the whole (large)
# artifact download. The same check at the verify site stays as the authoritative
# guard (this early one is UX; PATH could in principle change in between).
if [ -n "$PUBKEY" ]; then
  command -v minisign >/dev/null \
    || die "--pubkey given but the minisign CLI is not installed — install it first (apt install minisign / brew install minisign)"
fi

# Resolve the DEFAULT prefix now that flags are parsed (--help already exited).
# ONLY the default needs HOME, so a scrubbed env with HOME unset can still
# `--prefix /opt/higgs` — and gets a clear error, not a raw `set -u` abort, if
# it relies on the default without HOME.
if [ -z "$PREFIX" ]; then
  [ -n "${HOME:-}" ] || die "HOME is unset — pass an explicit --prefix <dir> (the default prefix is \$HOME/.higgs)"
  PREFIX="${HOME}/.higgs"
fi

# Resolve PREFIX to an ABSOLUTE path NOW, before any `cd` — a relative prefix
# would otherwise be interpreted against the temp workdir below and get deleted
# by the cleanup trap. mkdir first so `cd … && pwd` can canonicalize it.
mkdir -p "$PREFIX" || die "cannot create prefix $PREFIX"
# `CDPATH=` disables CDPATH for THIS cd: with a caller's `CDPATH=.:$HOME/projects`,
# `cd node` could otherwise select a DIFFERENT dir AND make `cd` print that path,
# so the command substitution would capture a two-line PREFIX. `cd --` stops a
# `-`-leading name being read as an option. (Belt-and-suspenders with `-p`, which
# already ignores an inherited CDPATH.)
# `-P` (physical): resolve the prefix to its CANONICAL path — symlinks and any
# `..` collapsed the way the kernel/getcwd sees it — so the service unit is
# pinned to the real directory. Logical `cd`/`pwd` would record a symlink or
# `..` spelling that publishes files one place while reporting another, and that
# repointing/removing the symlink later would break. (The prefix exists here —
# `mkdir -p` ran above — so `cd -P` can fully canonicalize it.)
# shellcheck disable=SC1007  # `CDPATH= cd` is an intentional env-prefix, not a typo'd assignment
PREFIX="$(CDPATH= cd -P -- "$PREFIX" && pwd -P)" || die "cannot resolve prefix $PREFIX"

# ---- pick the artifact suffix + (target, variant) for this machine ----------
os="$(uname -s)"; arch="$(uname -m)"
case "${os}/${arch}" in
  Darwin/arm64) suffix="aarch64-apple-darwin"; target="aarch64-apple-darwin"; variant="metal"
                [ "$CUDA" = 1 ] && die "--cuda is a Linux option (macOS uses Metal)" ;;
  Linux/x86_64) target="x86_64-unknown-linux-gnu"
                if [ "$CUDA" = 1 ]; then suffix="${target}-cuda"; variant="cuda"
                else suffix="$target"; variant="cpu"; fi ;;
  *) die "unsupported platform ${os}/${arch} (releases cover macOS arm64 + Linux x86_64)" ;;
esac

# The integrity workdir (the downloaded tarball + its .sha256 sidecar live here
# until they are verified and published) must sit under a TRUSTED base. `mktemp
# -d` creates it 0700, but RENAMING it needs write on its PARENT — so if ANY dir
# on $TMPDIR's PHYSICAL ancestry is writable by another user (GROUP- or OTHER-
# writable, without a trusted-owner sticky bit), that user can rename the 0700
# workdir out of the way, recreate the pathname, and substitute BOTH the tarball
# and its sha sidecar — a no-`--pubkey` install would then verify attacker content
# against attacker's sha and publish attacker code. The default /tmp is sticky and
# root-owned (safe); refuse a hostile custom $TMPDIR up front. Stricter than the
# install-path check below (which ALLOWS group-write for the credential-group
# prefix): the workdir holds UNVERIFIED downloads, so no untrusted writer may
# reach its parent at all.
# fn _has_write_acl_for_peer() — macOS: true if a dir's ACL grants write to a
# non-operator principal (used by both the TMPDIR and exec-path ancestry walks).
# Whether ONE directory carries a macOS ACL granting WRITE to a principal OTHER than
# the operator (a peer) — independent of the 0755 mode bits. `ls -lde` lists each ACL
# entry; parse from the RIGHT (perms = last field, allow/deny = second-to-last; an
# `inherited` flag sits BEFORE allow/deny) so a spaced Directory-Service principal and
# an inherited ACE are both handled. Self-exclusion uses the FULL principal span
# (never a flag-stripped name), so a `<self> inherited` peer is not mistaken for the
# operator. Shared by the exec-path walk AND the TMPDIR walk. TRI-STATE, NOT
# "ls-failed == no ACL": a DEFINITELY-absent path has no ACL (exit 1, no match), but a
# path that EXISTS whose ACL we cannot READ is an INSPECTION FAILURE — a root-owned
# ancestor can DENY the operator `readsecurity` (so `ls -lde` fails EACCES) while
# GRANTING a peer write; reading that as safe would let the peer swap the binary. On
# macOS (`ls -e` is a macOS extension) treat a failure over a PRESENT path as a
# peer-write ACL (exit 0, UNSAFE). On Linux `ls -lde` errors on `-e` for EVERY dir, so
# a failure there is NOT unsafe (no `ls`-visible ACLs) — the mode/owner checks govern.
_has_write_acl_for_peer() {
  { [ -e "$1" ] || [ -L "$1" ]; } || return 1
  _acl="$(ls -lde "$1" 2>/dev/null)" || { [ "$(uname)" = Darwin ] && return 0; return 1; }
  printf '%s\n' "$_acl" | awk -v me="$(id -un)" '
      /^[[:space:]]*[0-9]+:/ && NF >= 4 {
        if ($(NF-1) == "allow") {
          principal = ""
          for (i = 2; i <= NF-2; i++) principal = (principal == "" ? $i : principal " " $i)
          if (principal != ("user:" me) && $NF ~ /(write|add_file|add_subdirectory|append|delete|delete_child|writeattr|writeextattr|chown|writesecurity)/) found = 1
        }
      }
      END { exit(found ? 0 : 1) }'
}
# fn _refuse_renamable() — die if any level of a $TMPDIR ancestry is owned or
# writable by a peer (who could rename the workdir and swap the download).
_refuse_renamable() {
  # DIE on a resolution failure — never fall back to the lexical path (a peer who
  # makes this `cd -P` fail with a dangling link, then restores a safe target for the
  # walk and repoints the link before `mktemp`, would land the workdir in their dir).
  _r="$(CDPATH= cd -P -- "$1" 2>/dev/null && pwd -P)" \
    || die "TMPDIR path $1 could not be resolved (a symlink whose target vanished mid-check) — refusing rather than validate only its lexical spelling; export TMPDIR to a private dir and re-run"
  while :; do
    # UNTRUSTED OWNER: a dir owned by another user is renamable/replaceable by that
    # owner regardless of its mode bits — reject it (e.g. $TMPDIR beneath an
    # attacker-owned 0755 /srv/drop).
    if [ -z "$(find "$_r" -maxdepth 0 \( -uid 0 -o -uid "$EUID" \) 2>/dev/null)" ]; then
      die "TMPDIR path $_r is owned by an untrusted user — its owner can rename it and swap the integrity workdir; export TMPDIR to a dir whose every parent is owned by you or root and re-run"
    fi
    if [ -n "$(find "$_r" -maxdepth 0 \( -perm -0002 -o -perm -0020 \) 2>/dev/null)" ] \
       && [ -z "$(find "$_r" -maxdepth 0 -perm -1000 \( -uid 0 -o -uid "$EUID" \) 2>/dev/null)" ]; then
      die "TMPDIR path $_r is group/other-writable without a trusted sticky owner — a peer could rename the integrity workdir and swap the downloaded artifact; export TMPDIR to a private dir (e.g. \"\$(mktemp -d)\") and re-run"
    fi
    # A macOS ACL can grant a peer add_subdirectory/delete_child while the mode stays
    # 0755 — the mode-bit check above misses it. The workdir holds UNVERIFIED
    # downloads, so ANY peer-write ACL on its ancestry lets that peer rename the
    # workdir and swap the tarball + its .sha256 (a no-`--pubkey` install then
    # publishes attacker code). Refuse it.
    if _has_write_acl_for_peer "$_r"; then
      die "TMPDIR path $_r carries a macOS ACL granting write to another user — a peer could rename the integrity workdir and swap the downloaded artifact; remove it (chmod -N $_r) or export TMPDIR to a private ACL-free dir and re-run"
    fi
    [ "$_r" = "/" ] && break
    _r="$(dirname "$_r")"
  done
}
# Canonicalize $TMPDIR ONCE and use that physical path for BOTH the check and the
# mktemp: validating the resolved path but then handing the LEXICAL path to mktemp
# would be a TOCTOU — with `TMPDIR=/srv/shared/link` (link → a safe dir), the walk
# passes, then a peer repoints `link` at an attacker dir before mktemp lands the
# workdir there. Using the resolved path throughout removes the symlink a peer
# could repoint. DIE on a resolution failure — a lexical fallback would let a peer
# make this `cd -P` fail (a dangling `TMPDIR=/shared/link`), pass `_refuse_renamable`
# against a restored safe target, then repoint the link before `mktemp` lands the
# workdir in their dir.
_tmpbase="$(CDPATH= cd -P -- "${TMPDIR:-/tmp}" 2>/dev/null && pwd -P)" \
  || die "TMPDIR (${TMPDIR:-/tmp}) could not be resolved — refusing rather than fall back to its lexical spelling (a repoint race on the integrity workdir); export TMPDIR to a private dir and re-run"
_refuse_renamable "$_tmpbase"
workdir="$(mktemp -d "${_tmpbase}/higgs-install.XXXXXX")"
stage=""
tmplink=""
# fn cleanup() — EXIT trap: remove the workdir, stage dir, and temp symlink.
cleanup() { rm -rf "$workdir" ${stage:+"$stage"} ${tmplink:+"$tmplink"}; }
trap cleanup EXIT

# ---- obtain tarball + sidecars ----------------------------------------------
# The PAT is piped (via -K -) into curl's stdin, so curl is the ONE subprocess
# that ever receives the token. Resolve it to the TRUSTED absolute system path
# (/usr/bin/curl, root-owned on both macOS and Linux) — a `curl` planted earlier
# on the operator's PATH could otherwise read the config off stdin and exfiltrate
# the credential. (jq/sed/etc. never see the token, so only curl needs this.)
CURL_BIN=/usr/bin/curl

# fn gh_api() — curl the GitHub REST API with the PAT fed via a config pipe.
# curl auth goes through a config read on a PIPE (-K -) so the token never
# appears in argv (world-readable) NOR on disk — a here-doc would be a temp
# file under bash 3.2 (stock macOS).
gh_api() { # $1 = URL, $2 = Accept header, $3 = output file
  # (checked here, not at top level, so a --tarball install never needs curl.)
  [ -x "$CURL_BIN" ] || die "trusted curl not found at ${CURL_BIN} — install curl, or use --tarball"
  # -q FIRST: ignore ~/.curlrc — an operator's persistent `trace-ascii`,
  # extra `url`, or `location-trusted` there could otherwise write/leak the
  # Authorization header. Everything else is set explicitly here.
  # TOKENLESS when no PAT is set (the repo is public): the config pipe then
  # carries no Authorization header at all — an empty Bearer would be a 401.
  if [ -n "$_gh_token" ]; then
    printf 'header = "Authorization: Bearer %s"\n' "$_gh_token" \
      | "$CURL_BIN" -q -fsSL -K - -H "Accept: $2" -H "X-GitHub-Api-Version: 2022-11-28" \
             -o "$3" "$1"
  else
    "$CURL_BIN" -q -fsSL -H "Accept: $2" -H "X-GitHub-Api-Version: 2022-11-28" \
           -o "$3" "$1"
  fi
}

# fn fetch_asset() — download one named release asset into the workdir.
# Downloads the release asset named $1 into $workdir, if the release has it.
# Returns 1 when absent — callers decide whether that is fatal. Asset id lookup
# is via jq (a line-oriented scrape breaks on minified/reordered JSON).
fetch_asset() { # $1 = asset name
  local id
  id="$(jq -r --arg n "$1" '.assets[] | select(.name == $n) | .id' \
          "$workdir/release.json")"
  [ -n "$id" ] && [ "$id" != "null" ] || return 1
  gh_api "https://api.github.com/repos/${REPO}/releases/assets/${id}" \
         "application/octet-stream" "$workdir/$1"
}

if [ -n "$TARBALL" ]; then
  [ -f "$TARBALL" ] || die "no such tarball: $TARBALL"
  name="$(basename "$TARBALL" .tar.gz)"
  case "$name" in
    higgs-v*-"$suffix") : ;;
    *) die "tarball $name does not look like a higgs-v<ver>-${suffix} artifact for this machine" ;;
  esac
  _tarver="${name#higgs-v}"; _tarver="${_tarver%-"$suffix"}"
  # If the caller ALSO pinned --version, it must MATCH the tarball. A tarball is a
  # complete, self-naming artifact, so we take its version as authoritative — but
  # SILENTLY overriding an explicit --version would let automation that pins
  # `--version 0.5.0` be handed a validly-signed OLDER `higgs-v0.4.0-…` tarball
  # and perform a DOWNGRADE (the 0.4.0 manifest verifies fine against 0.4.0).
  # Refuse the mismatch loudly instead. (VERSION="" = --version not given.)
  if [ -n "$VERSION" ] && [ "$VERSION" != "$_tarver" ]; then
    die "--version $VERSION conflicts with tarball version $_tarver ($name) — they must match (drop --version to use the tarball's own version, or supply the matching tarball)"
  fi
  VERSION="$_tarver"
  cp "$TARBALL" "$workdir/${name}.tar.gz"
  # Sidecars travel next to the tarball when present (scp the whole set).
  for side in .tar.gz.sha256 .manifest .manifest.minisig; do
    [ -f "$(dirname "$TARBALL")/${name}${side}" ] \
      && cp "$(dirname "$TARBALL")/${name}${side}" "$workdir/"
  done
else
  # Keep the PAT out of an xtrace for the whole token-handling fetch: `bash -x`
  # or an inherited SHELLOPTS=xtrace would otherwise expand `$_gh_token`
  # into the trace (the `-K -` pipe only hides it from argv/ps, not from
  # `set -x`) — leaking it into a CI/debug log. Silence tracing here (the
  # `2>/dev/null` also hides the trace of the `set +x` line itself), then
  # restore the caller's setting. gh_api is only ever called from within this
  # block, so its `printf` of the token is covered too.
  _xtrace_on=0; case "$-" in *x*) _xtrace_on=1;; esac
  { set +x; } 2>/dev/null
  # No hard token gate: the repo is public, so an unauthenticated fetch is the
  # default; a fetch failure below names the PAT as the fix for a private repo.
  command -v jq >/dev/null \
    || die "the GitHub download path needs jq — install jq, or use --tarball with a scp'd artifact"
  if [ -n "$VERSION" ]; then
    gh_api "https://api.github.com/repos/${REPO}/releases/tags/v${VERSION}" \
           "application/vnd.github+json" "$workdir/release.json" \
      || die "could not fetch release 'v${VERSION}' from ${REPO} — no such version, or a private repo (set HIGGS_GITHUB_TOKEN, a fine-grained PAT with Contents:read)"
  else
    # "Newest release" via the LIST, not GitHub's /releases/latest — that endpoint
    # excludes PRERELEASES entirely (every higgs beta 404s there). Newest = the
    # first non-draft entry (the list is newest-first).
    gh_api "https://api.github.com/repos/${REPO}/releases?per_page=20" \
           "application/vnd.github+json" "$workdir/releases.json" \
      || die "could not list releases from ${REPO} — no releases yet, or a private repo (set HIGGS_GITHUB_TOKEN, a fine-grained PAT with Contents:read)"
    jq '[.[] | select(.draft | not)][0] // empty' "$workdir/releases.json" > "$workdir/release.json"
    [ -s "$workdir/release.json" ] || die "${REPO} has no published releases"
    VERSION="$(jq -r '.tag_name' "$workdir/release.json" | sed 's/^v//')"
    [ -n "$VERSION" ] && [ "$VERSION" != "null" ] || die "release JSON carried no tag_name"
  fi
  name="higgs-v${VERSION}-${suffix}"
  note "fetching ${name}.tar.gz from ${REPO} v${VERSION}"
  fetch_asset "${name}.tar.gz"        || die "release v${VERSION} has no asset ${name}.tar.gz"
  fetch_asset "${name}.tar.gz.sha256" || die "release v${VERSION} has no asset ${name}.tar.gz.sha256"
  fetch_asset "${name}.manifest"         || note "release has no update manifest (pre-signing release)"
  fetch_asset "${name}.manifest.minisig" || true
  if [ "$_xtrace_on" = 1 ]; then set -x; fi
fi

# The token was captured into `_gh_token` and its env var dropped at the top,
# before any subprocess. `_gh_token` (a non-exported shell var) has now done its
# one job; clear it too so nothing below can reference the credential.
unset _gh_token

# ---- verify ------------------------------------------------------------------
# fn sha256() — hash a file with the trusted absolute SHA-256 tool (defined
# once, per platform, from whichever trusted binary exists).
# Pick a SHA-256 tool by TRUSTED ABSOLUTE PATH — never a bare `command -v`. This
# tool is the ARTIFACT INTEGRITY check (its output is compared to the .sha256
# sidecar and the signed manifest): a PLANTED `sha256sum`/`shasum` that emits a
# genuine manifest's hash for a MALICIOUS tarball would defeat both. macOS has
# no `/usr/bin/sha256sum` (only `shasum`), so `command -v sha256sum` there would
# fall through the PATH tail to a plant. GNU `sha256sum` (coreutils) lives at
# `/usr/bin/sha256sum` on Linux; `shasum` (Perl) at `/usr/bin/shasum` on both.
# Both print "<hex>  <file>", so the `cut -d' ' -f1` below is format-compatible.
if [ -x /usr/bin/sha256sum ]; then
  sha256() { /usr/bin/sha256sum "$1"; }
elif [ -x /usr/bin/shasum ]; then
  sha256() { /usr/bin/shasum -a 256 "$1"; }
else
  die "need '/usr/bin/sha256sum' or '/usr/bin/shasum' to verify the download — install coreutils"
fi
if [ -f "$workdir/${name}.tar.gz.sha256" ]; then
  # The .sha256 is "<hex>  <file>" output; recompute and compare the hex.
  want="$(cut -d' ' -f1 < "$workdir/${name}.tar.gz.sha256")"
  got="$(sha256 "$workdir/${name}.tar.gz" | cut -d' ' -f1)"
  [ "$want" = "$got" ] || die "sha256 MISMATCH for ${name}.tar.gz: manifest says $want, file is $got"
  ok "sha256 verified"
else
  die "no .sha256 next to the tarball — refusing an unverifiable install"
fi

if [ -n "$PUBKEY" ]; then
  command -v minisign >/dev/null || die "--pubkey given but the minisign CLI is not installed"
  [ -f "$workdir/${name}.manifest" ] && [ -f "$workdir/${name}.manifest.minisig" ] \
    || die "--pubkey given but this release ships no signed manifest"
  minisign -V -P "$PUBKEY" -m "$workdir/${name}.manifest" -x "$workdir/${name}.manifest.minisig" >/dev/null \
    || die "minisign verification FAILED for ${name}.manifest — do not install this artifact"
  # Schema FIRST: a bump (schema 2) explicitly means an old verifier like this
  # one must not interpret the fields — refuse rather than trust a scrape.
  mschema="$(sed -n 's/.*"schema":\([0-9]*\).*/\1/p' "$workdir/${name}.manifest")"
  [ "$mschema" = "1" ] \
    || die "signed manifest schema is ${mschema:-<none>}, this installer understands 1 — update install.sh"
  # Every field must match what we ASKED FOR — otherwise a signed OLD manifest +
  # OLD tarball (with a regenerated unsigned .sha256) could be replayed under a
  # new release's filenames and silently install the old binary.
  # fn mfield() — extract one "key":"value" field from the signed manifest.
  mfield() { sed -n "s/.*\"$1\":\"\\([^\"]*\\)\".*/\\1/p" "$workdir/${name}.manifest"; }
  mver="$(mfield version)"; mfile="$(mfield file)"; mtarget="$(mfield target)"; mvariant="$(mfield variant)"
  msha="$(printf '%s' "$(mfield sha256)" | tr '[:upper:]' '[:lower:]')"
  [ "$mver" = "$VERSION" ]           || die "signed manifest is for version $mver, expected $VERSION"
  [ "$mfile" = "${name}.tar.gz" ]    || die "signed manifest is for file $mfile, expected ${name}.tar.gz"
  [ "$mtarget" = "$target" ]         || die "signed manifest is for target $mtarget, expected $target"
  [ "$mvariant" = "$variant" ]       || die "signed manifest is for variant $mvariant, expected $variant"
  [ "$msha" = "$got" ]               || die "signed manifest sha256 ($msha) does not match the tarball ($got)"
  ok "minisign-signed manifest verified (version/file/target/variant/sha256)"
elif [ -f "$workdir/${name}.manifest.minisig" ]; then
  note "signed manifest present but no --pubkey given — skipping signature check (sha256 only)"
fi

# ---- unpack in a STAGING dir, then rename into place -------------------------
bindir="${PREFIX}/bin"
verdir="${bindir}/v${VERSION}"
mkdir -p "$bindir" "${PREFIX}/logs"

# Refuse a WORLD-writable dir ANYWHERE on the install path BEFORE publishing into
# it. `umask 022` above bounds dirs WE create, but a PRE-EXISTING `0777` (from an
# earlier permissive-umask run, or a hand-made dir) would let any local user
# replace `bin/current`/the binary — or, for an ANCESTOR of the prefix, rename
# the whole subtree — and have the service run it as the operator. Checked: the
# `bin/` dir and every ANCESTOR up to `/`, the `v<version>` dir (a REINSTALL
# preserves it, so an old world-writable version dir stays a swap target), and
# `logs/` (a peer could pre-plant node.log as a symlink into an operator file).
# The STICKY bit (`-perm -1000`, as on `/tmp`) exempts a dir ONLY when it is owned
# by root or the operator ($EUID): sticky lets the dir's OWNER — not just an
# entry's owner — rename/delete ANY entry, so an ATTACKER-owned sticky dir is
# still a rename/swap vector. GROUP-write is allowed (the credential-group prefix
# feature); only OTHER-write WITHOUT a trusted-owner sticky is the hazard.
# `find -maxdepth 0 -perm -NNNN` is portable (BSD + GNU).
# $1 = dir; $2 = "strict" (refuse ANY other-write — a MANAGED dir where we create
# a predictable entry) or "sticky_ok" (refuse other-write only WITHOUT a
# trusted-owner sticky — an ANCESTOR above the prefix, where sticky blocks the
# only attack, a rename).
# fn _refuse_ww() — die if a directory on the install path is world-writable
# (trusted-sticky ancestors exempt via the "sticky_ok" mode). $1 = dir, $2 = mode.
_refuse_ww() {
  # Existence: -e FOLLOWS symlinks, so a DANGLING symlink (target hidden mid-check)
  # reads false — do NOT skip it as "absent"; a symlink at a managed exec path is a
  # redirect/hide vector. Test -e OR -L.
  { [ -e "$1" ] || [ -L "$1" ]; } || return 0
  # Check the PHYSICAL directory: `find <symlink>` inspects the SYMLINK's own mode
  # (0755), so a `bin -> /var/tmp/shared` whose TARGET is 0777 would slip through
  # while the install still publishes `current`/`v<ver>` into that shared dir. `cd
  # -P` follows the link to the real dir we actually write to, and checks THAT. A
  # symlink/dir that EXISTS but won't resolve (its target hidden mid-check) is a
  # TOCTOU/hide signal — DIE rather than fall back to the unchecked lexical path.
  _p="$1"
  if [ -L "$1" ] || [ -d "$1" ]; then
    _p="$(CDPATH= cd -P -- "$1" 2>/dev/null && pwd -P)" \
      || die "$1 could not be resolved (a symlink to a vanished target, or an unreadable dir) — refusing rather than validating only its lexical spelling; re-run"
  fi
  # UNTRUSTED OWNER: a dir owned by another user is renamable/replaceable by its
  # owner regardless of its mode — reject it (the exec path must be owned entirely
  # by you or root, or a peer who owns an ancestor swaps the service binary).
  if [ -z "$(find "$_p" -maxdepth 0 \( -uid 0 -o -uid "$EUID" \) 2>/dev/null)" ]; then
    die "$1 is owned by an untrusted user — its owner can rename or replace it; use a path whose every parent is owned by you or root, and re-run"
  fi
  [ -n "$(find "$_p" -maxdepth 0 -perm -0002 2>/dev/null)" ] || return 0
  # Sticky exemption only when sticky AND owned by root or the operator.
  if [ "$2" = sticky_ok ] \
     && [ -n "$(find "$_p" -maxdepth 0 -perm -1000 \( -uid 0 -o -uid "$EUID" \) 2>/dev/null)" ]; then
    return 0
  fi
  die "$1 is world-writable (without a trusted sticky owner) — a local user could replace the installed binary, plant its log, or rename this path; run 'chmod o-w $1' and re-run"
}
# Managed dirs (prefix and everything we create in it) are STRICT — even a sticky
# world-writable one lets a peer plant the predictable `current`/`v<ver>`/node.log.
# Ancestors ABOVE the prefix only risk a rename, which sticky blocks.
_d="$bindir"
while :; do
  case "$_d" in
    "$PREFIX"|"$PREFIX"/*) _refuse_ww "$_d" strict ;;   # prefix + below: strict
    *) _refuse_ww "$_d" sticky_ok ;;                     # ancestor: sticky ok
  esac
  [ "$_d" = "/" ] && break
  _d="$(dirname "$_d")"
done
# The loop above is LEXICAL. If `bin` is a SYMLINK (e.g. bin -> /srv/drop/alice-bin,
# a shared publish dir), it walks the $PREFIX chain but NEVER the symlink's real
# parent (/srv/drop). A world-writable /srv/drop then lets a peer replace the
# published binary AFTER install, and the printed `<prefix>/bin/current/higgs node
# install-service` command execs attacker code as the operator. So ALSO walk the
# PHYSICAL bindir chain: the real publish dir (strict) + its physical ancestors
# (sticky_ok, trusted-owner). Skipped when `bin` is a plain dir (already covered).
# DIE on a resolution failure — a `bin` symlink whose target is hidden mid-check
# (a peer renaming `/srv/drop/alice-bin` away during this `cd -P`) must NOT fall open
# to an empty `_binphys` that SKIPS the physical-ancestry walk (which would then miss
# a world-writable real publish dir). A managed path that won't resolve is hostile.
_binphys="$(CDPATH= cd -P -- "$bindir" 2>/dev/null && pwd -P)" \
  || die "${bindir} could not be resolved (a symlink whose target vanished mid-check) — refusing rather than skip its physical-ancestry validation; re-run"
if [ "$_binphys" != "$bindir" ]; then
  _d="$_binphys"
  while :; do
    case "$_d" in
      "$_binphys"|"$_binphys"/*) _refuse_ww "$_d" strict ;;  # real publish dir + below
      *) _refuse_ww "$_d" sticky_ok ;;                        # its physical ancestors
    esac
    [ "$_d" = "/" ] && break
    _d="$(dirname "$_d")"
  done
fi
_refuse_ww "$verdir" strict
_refuse_ww "${PREFIX}/logs" strict

# Stage under bindir (same filesystem → the final move is a rename(2)). Unpack
# HERE so a reinstall of the currently-running version can never truncate its
# live binary. install.sh NEVER EXECUTES THE ARTIFACT: it verifies (sha256 +,
# with --pubkey, the signature) and places files. The binary runs only when
# the operator explicitly installs the service — so a compromised tarball
# cannot run code as the operator merely by being install.sh's argument, and
# an unauthenticated source is verified-for-integrity but trusted-for-code
# only at the operator's explicit `install-service`/service-start step. Use
# --pubkey for any artifact whose source you do not fully control.
stage="$(mktemp -d "${bindir}/.stage.XXXXXX")"
tar -xzf "$workdir/${name}.tar.gz" -C "$stage"
# A REGULAR executable file — `-x` alone also passes a directory named `higgs`,
# which would then publish as v<ver>/higgs/ (a dir) and make current/higgs
# unexecutable. `-h` excludes a symlink.
{ [ -f "${stage}/higgs" ] && [ ! -h "${stage}/higgs" ] && [ -x "${stage}/higgs" ]; } \
  || die "tarball did not contain a regular executable 'higgs'"

# macOS quarantines downloaded executables; a LaunchDaemon can't answer the GUI
# prompt, so strip the xattr. Gated to macOS and the TRUSTED ABSOLUTE binary:
# `xattr` is a macOS-only tool absent from Linux's trusted dirs, so a bare
# `command -v xattr` on a Linux worker would fall through the PATH tail to a
# PLANTED `xattr` — which (running as a child while the PAT is still in this
# process's initial environment) could read it from `/proc/$PPID/environ`. On
# Linux there is no quarantine, so skip it entirely.
if [ "$os" = "Darwin" ] && [ -x /usr/bin/xattr ]; then
  /usr/bin/xattr -d com.apple.quarantine "${stage}/higgs" 2>/dev/null || true
fi

# Refuse a symlinked version dir: publishing into `v<ver> -> /somewhere` would
# write through the link. (We already refuse root, so this is the operator's
# own tree either way — but never silently follow.)
[ -L "$verdir" ] && die "${verdir} is a symlink — refusing; remove it and re-run"

# Publish into v<ver>/ by renaming each staged FILE over its target. verdir
# always exists throughout (mkdir -p is idempotent), so `current` never points
# at a missing dir even for a moment; and a per-file rename never truncates a
# running binary (the live process keeps its old inode) — so reinstalling the
# very version `current` names is safe, with no move-aside window to crash in.
# rename(2) via perl, NOT `mv`: BSD `mv` DESCENDS into a directory (or symlink-
# to-dir) destination, silently nesting the file as v<ver>/higgs/higgs and
# leaving current/higgs unexecutable; rename(2) replaces a regular-file or
# symlink target atomically and FAILS on a real directory instead of nesting.
#
# A MANUAL install supersedes any pending node self-update TRIAL — cleared HERE,
# BEFORE anything goes live. `higgs node self-update` writes these boot-guard markers;
# a leftover `{to:v<this>, prev:v<old>}` trial whose failure budget is already spent
# would otherwise make the next node boot roll the freshly-installed binary back. This
# must precede the PUBLISH, not just the flip: on a SAME-version repair `current`
# already points at v<ver>/, so the per-file rename below makes the repaired bytes live
# IMMEDIATELY (no flip) — the trial has to be gone before that instant. (A DIFFERENT-
# version install additionally relies on the boot-guard's own `trial.to != current`
# staleness check once `current` flips.) Clearing first is crash-safe: killed at any
# point, current ends on a binary with no stale spent trial. install.sh is the
# authoritative "known-good, start fresh" action — it ALSO clears `.update-failed` (the
# rolled-back-version poison list), so a deliberate re-install can re-apply a version that
# had crash-looped (e.g. a fixed rebuild), and a hub push of it is no longer refused.
# -rf (not -f): recover even if a marker path is a stray DIRECTORY — `rm -f` fails on a dir, which
# would otherwise wedge the boot-guard (a `.update-failed` directory blocks the poison write, so a
# crash-looping version could never be poisoned and would loop; a re-install must be able to reset).
rm -rf "${bindir}/.update-trial" "${bindir}/.update-bootfails" "${bindir}/.update-failed"

# Flush the STAGED files' DATA to disk BEFORE publishing. On a same-version
# reinstall `current` already points at v<ver>/, so the per-file rename below
# makes the new inode live IMMEDIATELY — a sync only AFTER the rename would be
# too late. rename(2) is atomic against readers but NOT crash-durable, so a
# power loss between the rename and a flush could leave current/higgs on an
# unflushed/corrupt inode. Syncing the staged data first means the inode is
# already durable the instant it goes live.
sync
mkdir -p "$verdir"
for f in "$stage"/*; do
  dest="${verdir}/$(basename "$f")"
  perl -e 'rename $ARGV[0], $ARGV[1] or die "rename: $!\n"' "$f" "$dest" \
    || die "publishing ${dest} failed — is a directory in the way? remove ${verdir} and re-run"
done
rmdir "$stage" 2>/dev/null || rm -rf "$stage"
stage=""

# Record the acceleration VARIANT this artifact was built for into the version dir, so
# the node self-updater (`higgs node self-update`, DESIGN-remote §9 P3) can read the
# INSTALLED variant of `current` and refuse an update that would silently switch it
# (e.g. a CPU artifact replacing a CUDA install). Written as a plain one-line marker
# next to the binary; a fresh temp + rename keeps it atomic. self-update writes the
# same marker on its own publishes, so both install paths agree.
vmarker_tmp="${verdir}/.variant.tmp.$(od -An -N8 -tx1 /dev/urandom | tr -d ' \n')"
printf '%s' "$variant" > "$vmarker_tmp" \
  && perl -e 'rename $ARGV[0], $ARGV[1] or die "rename: $!\n"' "$vmarker_tmp" "${verdir}/.variant" \
  || die "recording the variant marker at ${verdir}/.variant failed"

# Now flush the rename/version-dir entries before the flip makes `current`
# point at them, so a post-crash `current` never resolves to a missing entry.
# (`sync` is fully durable on Linux — the fleet-node target; best-effort on
# macOS, whose only hard guarantee is F_FULLFSYNC, out of reach from a script.)
sync

# Atomic flip: build the new symlink under a temp name in the SAME directory,
# then rename(2) over `current`. rename never follows the destination symlink
# and either fully succeeds or changes nothing — there is no window where
# `current` is missing or dangling. perl is used for the rename because BSD
# `mv` on a symlink-to-a-directory descends INTO the directory instead.
# The temp name is RANDOM (urandom hex), never the predictable PID: a stale
# `.current.tmp.<pid>` symlink left by an interrupted install could be REUSED —
# `ln -s` into an existing symlink-to-a-directory descends INTO it (creating
# `vOld/vNew`) and the rename below would then flip `current` onto the STALE
# target while reporting success. cleanup() also removes the staged link on any
# exit, so an interrupted flip leaves no litter behind.
tmplink="${bindir}/.current.tmp.$(od -An -N8 -tx1 /dev/urandom | tr -d ' \n')"
ln -s "v${VERSION}" "$tmplink" || die "cannot stage the current flip at ${tmplink}"
# Prove the staged link points at the NEW version before it replaces `current`:
# if anything already sat at the temp name (so ln descended or was a no-op),
# readlink exposes it — refuse rather than flip current onto the wrong target.
[ "$(readlink "$tmplink")" = "v${VERSION}" ] \
  || die "staged flip link ${tmplink} does not point at v${VERSION} — refusing to flip"
perl -e 'rename $ARGV[0], $ARGV[1] or die "rename: $!\n"' "$tmplink" "${bindir}/current" \
  || die "atomic flip of ${bindir}/current failed"
tmplink=""
# Flush the flipped `current` symlink itself, so the completed install survives
# a crash the moment this script returns.
sync

ok "installed higgs v${VERSION} → ${verdir}"
ok "current → $(readlink "${bindir}/current")"

# UPDATE-IN-PLACE UX: if a higgs node service is ALREADY installed, the running
# process still executes the OLD binary — restart it now so the flip takes
# effect immediately (the operator should never need to know launchctl/systemctl
# incantations). Best-effort: a restart failure is NOTED, never fatal — the
# install itself succeeded, and the next service (re)start picks up `current`.
if [ "$os" = "Darwin" ]; then
  if [ -n "${HOME-}" ] && [ -f "${HOME}/Library/LaunchAgents/com.higgs.node.plist" ]; then
    if launchctl kickstart -k "gui/$(id -u)/com.higgs.node" 2>/dev/null; then
      ok "restarted the higgs node service onto v${VERSION}"
      # The binaries are not (yet) Developer-ID signed, so macOS keys the Local
      # Network grant to the EXACT binary — every new version re-prompts, and
      # the prompt only appears in a GUI session (never over SSH; there is no
      # command-line grant). Without the click, the node times out connecting.
      note "macOS will show a 'higgs wants to find devices on local networks' popup — click ALLOW"
      note "(GUI only — it cannot appear over SSH, and there is no command-line way to grant it;"
      note " if the node shows 'connect failed: timed out', check System Settings → Privacy &"
      note " Security → Local Network and enable higgs, then rerun the launchctl restart above)"
    else
      note "could not restart com.higgs.node — restart it (or log out/in) to run v${VERSION}"
    fi
  fi
else
  if command -v systemctl >/dev/null && systemctl --user cat higgs-node.service >/dev/null 2>&1; then
    if systemctl --user restart higgs-node.service 2>/dev/null; then
      ok "restarted the higgs node service onto v${VERSION}"
    else
      note "could not restart higgs-node.service — restart it to run v${VERSION}"
    fi
  fi
fi

# The service command ALWAYS carries the resolved --prefix: `install-service`
# defaults to the PASSWD home's ~/.higgs, which can differ from where we just
# installed (our default used $HOME) — so omitting it could point the service
# at a different dir. macOS needs sudo for the LaunchDaemon; Linux must NOT.
# Everything printed as a runnable command is %q-quoted so a prefix with
# spaces (`/opt/Higgs Fleet`) stays a single copy-pasteable argument.
qbin="$(printf '%q' "${bindir}/current/higgs")"
qprefix="$(printf '%q' "$PREFIX")"
# USER-SPACE BY DEFAULT: the default service scope (macOS LaunchAgent / Linux
# user unit without linger) needs NO sudo on either OS. Only the --system
# always-on variant needs sudo — and only on macOS (LaunchDaemon); on Linux
# --system just adds enable-linger to the same user unit.
sys_sudo=""
[ "$os" = "Darwin" ] && sys_sudo="sudo "
# The printed service command ALWAYS pins the state dir as the --higgs-home
# ARGV flag: sudo strips env vars (and rejects `sudo VAR=value` under
# command-specific/NOSETENV sudoers policies), so a copy-pasted macOS command
# would otherwise fall back to install-service's passwd-home default — which
# can differ from where THIS shell's higgs actually keeps its state
# ($HIGGS_HOME else $HOME/.higgs, the runtime's own derivation; e.g. an
# overridden HOME=/srv/node pairs into /srv/node/.higgs). Omitted only if both
# are unset/empty (nothing trustworthy to pin).
# Resolve a possibly-RELATIVE dir to absolute against THIS shell's cwd — the
# pairing context, where `higgs --node` resolved the same value. The printed
# command is meant to be copy-pasted, possibly from a DIFFERENT directory, and
# install-service re-absolutizes a relative value against ITS cwd — so a bare
# relative `--higgs-home state` would silently pin a different store. Lexical
# `$PWD/…` (the dir need not exist yet) matches install-service's own
# std::path::absolute; an already-absolute value is left untouched.
# Resolve against the PHYSICAL cwd (`pwd -P`), NOT bash's logical `$PWD`: the
# higgs runtime and install-service resolve a relative dir via getcwd (physical,
# symlinks collapsed), so from a symlinked working dir the logical `$PWD`
# spelling would pin a path that repointing/removing the symlink later breaks.
# (Purely a resolver — the `..` refusal below runs in the MAIN shell, because a
# `die` here would only exit the `$(…)` command-substitution subshell.)
# fn abspath() — resolve a possibly-relative dir to absolute against the physical cwd.
abspath() { case "$1" in /*) printf '%s' "$1" ;; *) printf '%s' "$(pwd -P)/$1" ;; esac; }
# Refuse a `..` in the RESOLVED value, whatever the source. The env-var guard
# near the top covers HIGGS_HOME/HIGGS_MODEL_DIR when SET; this also catches the
# `$HOME/.higgs` FALLBACK when `$HOME` itself contains `..`. install-service
# rejects `..`, so a printed Next-steps command carrying one would fail its own
# validation. Runs in the main shell (not inside abspath) so `die` aborts.
# fn reject_dotdot() — die if a resolved path ($1) contains a '..' component ($2 = label).
reject_dotdot() { case "/$1/" in */../*) die "$2 resolves to '$1' which contains a '..' component — use a resolved absolute path (install-service rejects '..')" ;; esac; }
svc_home=""
state_dir="${HIGGS_HOME:-${HOME:+${HOME}/.higgs}}"
if [ -n "$state_dir" ]; then
  state_dir="$(abspath "$state_dir")"
  reject_dotdot "$state_dir" "state dir"
  svc_home=" --higgs-home $(printf '%q' "$state_dir")"
fi
# Same handoff for the extra model scan root (README's documented pairing knob):
# a node paired with HIGGS_MODEL_DIR=/models must not reboot into a service
# that scans only the defaults and advertises no models. Absolutized + omitted
# when unset.
if [ -n "${HIGGS_MODEL_DIR:-}" ]; then
  _md="$(abspath "$HIGGS_MODEL_DIR")"
  reject_dotdot "$_md" "HIGGS_MODEL_DIR"
  svc_home="${svc_home} --model-dir $(printf '%q' "$_md")"
  unset _md
fi
# And the preserved DEPLOYMENT-CONFIG vars (allowlist mirrors service.rs's
# PRESERVED_ENV): a node paired against an enterprise HF mirror or a specific
# engine must reboot with the same config — sudo strips these, so carry each set
# one as a sudo-proof `--env KEY=VALUE` flag. Debug knobs (HIGGS_VERBOSE,
# RUST_LOG) are deliberately NOT here — they must not bake into a service.
for _k in HIGGS_HF_ENDPOINT HIGGS_ENGINE; do
  # ${!_k} is the value of the var named by $_k; only carry it when non-empty.
  if [ -n "${!_k:-}" ]; then
    svc_home="${svc_home} --env $(printf '%q' "${_k}=${!_k}")"
  fi
done
unset _k
echo
echo "${C_BOLD}Next steps:${C_OFF}"
# Default line: user-space, login-bound, no sudo. Second line: the --system
# always-on opt-in (survives logout, starts at boot).
echo "  ${C_CYAN}${C_BOLD}PATH:${C_OFF}    add $(printf '%q' "${bindir}/current") to PATH (or symlink ${qbin} into ~/.local/bin)"
if [ -z "$state_dir" ]; then
  # HOME and HIGGS_HOME are BOTH unset (a scrubbed provisioning env), so no state dir
  # can be derived and $svc_home carries no --higgs-home. `install-service` now
  # REFUSES an unset HOME (rather than silently pin the passwd default and restart-
  # loop), so even the login-bound command would be rejected — emit the fix instead of
  # an unusable command. (Same handling as the always-on line below.)
  echo "  ${C_CYAN}${C_BOLD}service:${C_OFF} ${C_YELLOW}set HIGGS_HOME (or run with HOME set) first — install-service needs an explicit --higgs-home state dir, which can't be derived with HOME unset.${C_OFF}"
else
  echo "  ${C_CYAN}${C_BOLD}service:${C_OFF} ${qbin} node install-service --prefix ${qprefix}${svc_home}   # user service (login-bound, no sudo)"
fi
# The always-on (--system) line runs `sudo <prefix>/bin/current/higgs` — the PREFIX
# BINARY as ROOT (macOS LaunchDaemon). If the exec path is GROUP-writable (a
# credential-group prefix), a group peer could swap the binary and have `sudo …`
# run their code as root — a check INSIDE install-service is too late (the binary
# already runs as root). Group-write is fine for the login-bound (no-sudo) service
# above, but NOT for --system: detect it and warn instead of advising the command.
# Gated on ${sys_sudo} (non-empty only on macOS, where --system uses sudo; Linux
# --system is a user unit + linger, no sudo).
# Walk the FULL ANCESTRY (not just the prefix/bin/verdir themselves): a
# group-writable ANCESTOR above the prefix (e.g. `/srv/fleet` root:team 0770, with
# a 0755 prefix beneath it) lets a group member RENAME the whole prefix subtree and
# substitute the binary — the printed `sudo …/higgs … --system` then runs their
# payload as root, even though the operator-owned dirs below are 0755. Check BOTH
# chains, as install-service (cli.rs `exec_ancestry`) does: the LEXICAL chain (the
# path AS the sudo command will name it — a group-writable dir reached only THROUGH
# a symlinked component, e.g. a `bin` symlink over a group-writable `${PREFIX}`,
# sits on THIS chain, not the physical one) AND the PHYSICAL chain (`cd -P`, so a
# group-writable REAL ancestor above a symlink target is covered too). A prior
# `cd -P`-first walk collapsed symlinks up front and saw ONLY the physical chain.
_exec_gw=""
# Group-write / write-ACL check for ONE directory (sets _exec_gw on a hit). SKIPS a
# symlink NODE: its own bits/ACL are meaningless (the OS governs by the target,
# covered on the physical chain), and a symlink's mode is 0777 on Linux — checking
# it would false-positive. Mirrors cli.rs `refuse_writable_mode` skipping a symlink.
# fn _gw_check_dir() — set _exec_gw if one dir is group-writable or has a peer-write ACL.
_gw_check_dir() {
  [ -L "$1" ] && return
  [ -n "$(find "$1" -maxdepth 0 -perm -0020 2>/dev/null)" ] && { _exec_gw=1; return; }
  # macOS ACL: an ACL can grant a peer write/add_file/delete_child while the mode
  # stays 0755 — the mode-bit check above misses it. Same peer-write-ACL detection as
  # the TMPDIR walk (`_has_write_acl_for_peer`): a `deny`, a read-only `allow`, or an
  # allow-to-self grants a peer nothing (so the default `~/.higgs` --system is not
  # falsely marked UNAVAILABLE); off macOS it is a no-op.
  if _has_write_acl_for_peer "$1"; then
    _exec_gw=1
  fi
  return 0
}
# Walk one chain from $1 up to `/`, checking every component. Always returns 0 —
# a hit is signalled via $_exec_gw, NEVER the exit status, so a "clean" walk under
# `set -e` (this script) never aborts the install.
# fn _gw_chain() — run _gw_check_dir up one ancestry chain from $1 to '/'.
_gw_chain() {
  _g="$1"
  while :; do
    _gw_check_dir "$_g"
    [ -n "$_exec_gw" ] && return 0
    [ "$_g" = "/" ] && break
    _g="$(dirname "$_g")"
  done
  return 0
}
# LEXICAL chain first (as typed), then the PHYSICAL chain if it differs. Explicit
# `return 0`: a trailing `&& [ "$_p" != "$1" ]` that is FALSE in the common case (a
# non-symlink prefix, where the physical path equals the lexical one) would
# otherwise make this function return non-zero and, under `set -e`, ABORT the whole
# install before Next-steps. The result is carried by $_exec_gw, not the exit code.
# fn _gw_walk() — check the lexical AND physical ancestry chains of $1
# (gates the printed --system service advice via _exec_gw).
_gw_walk() {
  _gw_chain "$1"
  [ -n "$_exec_gw" ] && return 0
  # If the physical resolution FAILS (a peer hid the symlink target mid-check), do
  # NOT fall back to validating only the lexical chain — mark the exec path UNSAFE so
  # the --system advice is WITHHELD (this runs after publish, so we suppress the
  # sudo command rather than abort). A clean lexical spelling over a hidden hostile
  # resolved tree is exactly the TOCTOU this guards.
  _p="$(CDPATH= cd -P -- "$1" 2>/dev/null && pwd -P)" || { _exec_gw=1; return 0; }
  [ "$_p" != "$1" ] && _gw_chain "$_p"
  return 0
}
_gw_walk "$bindir"
_gw_walk "$verdir"
if [ -n "$sys_sudo" ] && [ -n "$_exec_gw" ]; then
  echo "  ${C_CYAN}${C_BOLD}always-on:${C_OFF} ${C_YELLOW}UNAVAILABLE — the exec path (${PREFIX} or an ancestor) is GROUP-writable or carries an ACL, so a --system daemon (installed via sudo, exec'd as root) could let a peer run code as ROOT. Use a clean, non-group-writable, ACL-free prefix for --system ('chmod -RN ${PREFIX}; chmod -R g-w ${PREFIX}'), or keep the login-bound service above.${C_OFF}"
elif [ -z "$state_dir" ]; then
  # With both HOME and HIGGS_HOME unset, no state dir can be derived, so the
  # `install-service … --system` command would carry no --higgs-home — and since r54
  # the CLI REFUSES an unset HOME (rather than silently pin the passwd default). This
  # applies on BOTH platforms — macOS --system is elevated (sudo strips HOME) and
  # Linux --system is a non-elevated user unit that STILL refuses an unset HOME — so
  # the guard must NOT be gated on `$sys_sudo`. Emit the fix instead of an unusable
  # command. (The login-bound line above is guarded the same way.)
  echo "  ${C_CYAN}${C_BOLD}always-on:${C_OFF} ${C_YELLOW}set HIGGS_HOME (or run with HOME set) first — install-service needs an explicit --higgs-home state dir, which can't be derived with HOME unset.${C_OFF}"
else
  echo "  ${C_CYAN}${C_BOLD}always-on:${C_OFF} ${sys_sudo}${qbin} node install-service --prefix ${qprefix}${svc_home} --system   # survives logout, starts at boot"
fi
unset _exec_gw _g _p
