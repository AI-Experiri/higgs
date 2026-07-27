#!/usr/bin/env bash
# uninstall.sh — cleanly remove the higgs node service, leaving NO leftovers except the
# install prefix (~/.higgs by default: binaries, config, keys, pairings, models, logs).
#
# Removes exactly what `higgs node install-service` created:
#   macOS  — the LaunchAgent (gui/<uid>) and/or LaunchDaemon (/Library/LaunchDaemons)
#            labelled com.higgs.node, plus its plist.
#   Linux  — the systemd USER unit higgs-node.service, plus any enable-linger.
# The install PREFIX is kept by default (that's your identity, saved hubs, and models);
# pass --purge to delete it too.
#
# Usage:
#   ./uninstall.sh [--prefix <dir>] [--purge] [--dry-run]
#     --prefix <dir>  install root to look under (default: $HOME/.higgs)
#     --purge         ALSO delete the prefix (destroys endpoint.key/pairings/models — loud, opt-in)
#     --dry-run       print what would be done, touch nothing
#
# Run as the OPERATOR (not sudo) for the default login-bound service; a system daemon
# (installed via `--system`) needs root and is torn down with the sudo line this prints.
set -euo pipefail
export PATH="/usr/sbin:/usr/bin:/sbin:/bin${PATH:+:$PATH}"   # pin: no planted launchctl/systemctl/rm

SERVICE_NAME="com.higgs.node"     # macOS launchd label
UNIT_NAME="higgs-node.service"    # Linux systemd user unit

prefix="${HOME:-}/.higgs"
purge=0
dry=0
while [ $# -gt 0 ]; do
  case "$1" in
    --prefix)  prefix="${2:?--prefix needs a value}"; shift 2 ;;
    --purge)   purge=1; shift ;;
    --dry-run) dry=1; shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "error: unknown argument: $1" >&2; exit 2 ;;
  esac
done

run() { # echo + (unless dry-run) execute
  echo "  + $*"
  [ "$dry" -eq 1 ] || "$@"
}
removed_any=0

case "$(uname -s)" in
  Darwin)
    uid="$(id -u)"
    agent_plist="${HOME:-}/Library/LaunchAgents/${SERVICE_NAME}.plist"
    daemon_plist="/Library/LaunchDaemons/${SERVICE_NAME}.plist"

    # LaunchAgent (login-bound default) — no sudo; must run as the operator.
    if [ -e "$agent_plist" ] || launchctl print "gui/${uid}/${SERVICE_NAME}" >/dev/null 2>&1; then
      if [ "$uid" -eq 0 ]; then
        echo "note: an agent is bound to a user session — run uninstall.sh as that OPERATOR (not root) to remove it."
      else
        echo "macOS LaunchAgent ${SERVICE_NAME}:"
        run launchctl bootout "gui/${uid}/${SERVICE_NAME}" || true   # not-loaded → non-fatal
        [ -e "$agent_plist" ] && run rm -f "$agent_plist"
        removed_any=1
      fi
    fi

    # LaunchDaemon (--system) — needs root.
    if [ -e "$daemon_plist" ]; then
      if [ "$uid" -eq 0 ]; then
        echo "macOS LaunchDaemon ${SERVICE_NAME}:"
        run launchctl bootout "system/${SERVICE_NAME}" || true
        run rm -f "$daemon_plist"
        removed_any=1
      else
        echo "A system LaunchDaemon is installed — remove it as root:"
        echo "    sudo launchctl bootout system/${SERVICE_NAME} ; sudo rm -f ${daemon_plist}"
      fi
    fi
    ;;

  Linux)
    unit_path="${prefix}/${UNIT_NAME}"
    if systemctl --user list-unit-files "$UNIT_NAME" >/dev/null 2>&1 || [ -e "$unit_path" ]; then
      echo "Linux systemd user unit ${UNIT_NAME}:"
      run systemctl --user disable --now "$UNIT_NAME" || true   # stop + un-enable; non-fatal if absent
      [ -e "$unit_path" ] && run rm -f "$unit_path"
      run systemctl --user daemon-reload || true
      run loginctl disable-linger "$(id -un)" || true           # best-effort (only set by --system)
      removed_any=1
    fi
    ;;

  *) echo "error: unsupported platform $(uname -s) (macOS / Linux only)" >&2; exit 1 ;;
esac

[ "$removed_any" -eq 1 ] || echo "no higgs node service found (nothing to remove)."

# The prefix: kept by default, per the clean-uninstall contract.
if [ "$purge" -eq 1 ]; then
  echo
  echo "PURGE: deleting the install prefix and ALL higgs state — identity (endpoint.key),"
  echo "       saved hubs/pairings, downloaded models, and logs — under: ${prefix}"
  run rm -rf "$prefix"
else
  echo
  echo "Kept the install prefix (identity, saved hubs, models, logs): ${prefix}"
  echo "  (pass --purge to remove it too — destroys keys/pairings/models)"
fi

[ "$dry" -eq 1 ] && echo "(dry run — nothing was changed)"
echo "done."
