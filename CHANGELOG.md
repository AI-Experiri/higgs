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

### Added

- **Per-node runtime toggles for `log_incoming_tokens` and `log_show_fields`
  (NL-VX).** `NodeLogControlParams` / `NodeLogControlReply` gained the two
  remaining Log Terminal controls, so flipping "Log Incoming Tokens" or
  "Show log fields" for a remotely-loaded worker now behaves identically to
  the local `/v1` path. Under the hood, `log_incoming_tokens` moved from a
  struct-local atomic on `Higgs` to `LogBus` (mirroring `show_fields`) so
  the node-side dispatcher can flip it via `LogBus::global()`. Facade
  callers see no change — `Higgs::log_incoming_tokens()` /
  `set_log_incoming_tokens()` still work exactly the same, they just
  delegate to the bus now.
- **`scripts/check_diag_codes.py`** — a fast quality-gate step that scans
  every `#[diagnostic(code(X))]` under `src/` and fails on duplicate HG
  codes, so a copy-pasted code can never silently claim a second meaning.

### Fixed

- **`log_incoming_tokens` was a silent no-op on iroh-relayed chats.** The
  only reader was `serve/v1.rs`, which never sees hub → iroh chats — so a
  user who flipped the toggle for a remotely-loaded worker got the reply
  saying it applied, but no line ever landed in the Log Terminal.
  `node/data.rs::relay_chat` now emits the same `higgs: incoming <model> —
  N chars: <preview>` line the /v1 handler does, gated on the same bus flag.

### Improved

- **Remediation text on 27 HG codes** that previously said only what
  happened, without saying what to do about it. HG001, HG002, HG004,
  HG006-HG010, HG012-HG017, HG020-HG025, HG028-HG036, HG070, HG091 now
  read like the ones that always had inline guidance (HG005, HG011,
  HG038, …) — the error itself tells the operator the next step.

## [0.1.0-beta.13] - 2026-08-19

### Fixed

- **Node daemon lifecycle events stream live in the per-node log pane.**
  During a hub-pushed self-update, the operator watching the jigglebot
  per-node log terminal previously saw only two lines ("node daemon
  starting" + "connected to hub"); the ~8 interesting events between
  (drain start, worker unloads, drain complete, re-exec, boot-guard,
  hub reconnect) were emitted via `println!`/`eprintln!` from the
  daemon serve loop in `cli.rs` — bypassing LogBus and `M_NODE_LOGS`
  entirely. They only reached `node.log` via systemd/launchd's stdout
  capture, invisible to any operator watching remotely. Two changes fix
  the routing: the tracing `fmt::layer` now writes to stderr (was
  stdout by default), and every daemon-runtime print is now
  `tracing::info!/warn!/error!` with `target: "higgs::node"` and a
  stable `event=` field. Interactive one-shot CLI paths (`higgs link
  pair`, `higgs node connect`, `higgs --version`, the enrollment
  wizard) intentionally keep `println!`/`eprintln!` — their stdout IS
  the shell contract for scripts and pipes.

## [0.1.0-beta.12] - 2026-08-19

### Added

- **Passive network stats + inferred label (NQ).** New crate-level
  `Higgs::network_stats(node) -> Option<NetworkStats>` samples iroh's
  currently-selected path ON DEMAND (no probe traffic, no cache, no
  subscription) and returns the path kind (`Direct` / `Relay`), RTT,
  per-path counters (`lost_packets` / `sent_datagrams` / `bytes_tx` /
  `bytes_rx`), the current connection's `uptime_ms`, and an inferred
  `LinkState` (`Healthy` for Direct, `Degraded` for Relay or mid-migration,
  `Disconnected` when unpaired / dropped). Sample and uptime are read
  atomically on the fleet actor thread from ONE `Path::stats()` snapshot,
  so no cross-snapshot torn reads exist. Wire additions: `LinkPath`,
  `LinkState`, and `NetworkStats` in `crate::remote` (with matching
  const-object TS bindings). `None` from the wrapper only when the hub
  is off or the actor is unreachable.

## [0.1.0-beta.11] - 2026-08-17

### Added

- **Node log control (NL-V).** New `M_NODE_LOG_LEVEL` control op + wire
  types `NodeLogControlParams { verbose: Option<bool> }` and
  `NodeLogControlReply { verbose: bool }` let a hub toggle a paired node's
  log verbosity at runtime. `verbose=true` (the new default) admits
  `debug!`/`trace!` in addition to `info!`/`warn!`/`error!`; `false` clamps
  to `info!` and above. Every admitted line now carries a `[section]`
  badge derived from the tracing target's 2nd segment (`higgs::node::…` →
  `[node]`, `higgs::worker::…` → `[worker]`), so log consumers can tell
  where a line originated without inventing per-message tags. Facade:
  `Higgs::set_node_log_level(endpoint, params)`. A pre-NL-V node without
  the capability surfaces as `NodeUnreachable` with a "update to a
  node_log_control-capable release" hint.

- **Node daemon lifecycle logs.** Happy-path `tracing::info!` calls on
  node startup, hub-connect / hub-reconnect, worker load / unload / kill,
  and download start / finish, with targets that feed the section badge
  above. Fixes an empty `M_NODE_LOGS` stream on a healthy idle node — the
  stream worked but had nothing to carry.

- **Per-QUIC-stream priority (SP1).** Every send-stream open/accept site
  in the hub↔node iroh transport now tags Quinn/noq's
  `SendStream::set_priority` so the local scheduler orders traffic
  control > interactive > diagnostic. Three named constants
  (`CONTROL_STREAM_PRIORITY = +100`, `INTERACTIVE_STREAM_PRIORITY = 0`,
  `DIAGNOSTIC_STREAM_PRIORITY = -100`) map opcode → tier via
  `priority_for(method)`; unknown opcodes safely default to CONTROL. No
  wire-format change, no public API change, no bindings regen — peers see
  identical protocol; only local scheduler behavior shifts. Companion to
  NL-V's default-on verbose: LogBus subscriber-side was already lossy,
  SP1 closes the loop on the wire so a debug firehose on one connection
  cannot backpressure the chat stream sharing it.

### Changed

- **`LogBus::verbose` defaults to `true`.** Previously `false` (silent);
  now every daemon lifecycle line is admitted to the ring by default so
  `M_NODE_LOGS` streams useful content the moment a hub subscribes. Tests
  that asserted the old default were updated accordingly.

## [0.1.0-beta.10] - 2026-08-16

### Added

- **Machine-wide download deduplication.** Every download entry point (hub
  facade, node pull, `higgs model download` CLI) now claims a per-key kernel
  `flock` (`<models>/.download-locks/`, via `fs2`) held for the transfer's
  whole life — the kernel releases it on ANY exit, including SIGKILL and
  power loss. A second start of the same `(repo, file)` anywhere on the
  machine refuses with `[HG090]` and ADOPTS the live transfer instead of
  corrupting it: the UI row repaints with the original's real progress.
- **Machine downloads ledger + `higgs model downloads`.** Every transfer on
  the box records status/history (live progress, done/failed/cancelled) in
  `~/.higgs/models/.downloads.json`; the new CLI subcommand renders it, and
  the node announces the ledger UNION with its own in-flight registry over
  HELLO/`pull_status`, so a sibling process's CLI download is visible in the
  fleet view. Three staleness sweeps (dead pid, unheld lock file, 24 h
  lockless aging) retire residue rows on every read.
- **Honest `cancelled` download terminals.** Download event streams gained a
  `Cancelled` phase: `[HG089]` = this transfer stopped (nothing landed,
  partial cleaned), `[HG090]` = this *attempt* yielded to an already-running
  transfer that continues elsewhere. A duplicate click can no longer paint a
  running download as failed or leave an ownerless row.

### Changed

- Download identity is case-folded end to end (lock key, adopt lookup,
  announcement dedups) to match the default case-insensitive APFS — case
  variants of one on-disk file can no longer race two writers or show
  phantom duplicate rows.
- The node's pull-refusal daemon log now discriminates lock contention
  (`[HG090]` "already in flight") from filesystem faults (`[HG034]`, with
  the error attached) — a perms/disk problem no longer reads as a phantom
  duplicate transfer.
- Coverage gates: integration coverage raised to 84% (new black-box suites
  for the CLI surface, fleet wire, update/courier fixtures, catalog API, and
  serve edges); unit coverage held at ≥90%.

## [0.1.0-beta.9] - 2026-08-14

### Fixed

- **The [HG088] streamed-artifact fix actually ships.** The stream-to-disk
  update downloader (documented under beta.8 below) was stranded on the
  beta.6 release branch by a merge race — beta.7 and beta.8 binaries were
  built WITHOUT it, so hub-pushed updates to CUDA nodes still failed on the
  256 MiB in-memory cap. It is now merged through develop and in this build.
  Nodes still running beta.8 or older must take THIS update via `install.sh`
  once (their running downloader is the capped one); updates stream from
  beta.9 onward.

### Added

- `NodeLogWatch.created` (hub-side library API): tells the embedding consumer
  whether its watch spawned the node-log stream (its `rx` carries the full
  snapshot) or joined an existing one (ring replay needed) — fixes duplicated
  history in per-node log terminals.

## [0.1.0-beta.8] - 2026-08-13

### Added

- **Per-node daemon-log streaming**: the hub can view each fleet node's own
  higgs daemon log live (the node.log lines — never model/worker output).
  Off by default and watcher-driven: nothing crosses the network until the
  UI asks, the stream tears down the moment the last viewer leaves, and a
  log flood degrades to an explicit "lines dropped" marker instead of
  loading the connection. Groundwork for per-node log terminals in the
  jigglebot Fleet tab.

### Fixed

- **Fleet updates to CUDA nodes** [HG088]: the update artifact was downloaded
  into memory under a 256 MiB cap, refusing the ~650 MiB Linux CUDA tarball —
  every hub-pushed update to a CUDA node failed. Artifacts now stream to disk
  with the sha256 computed as bytes land (no memory spike at any size), are
  unpacked from the very file handle the hash covered, and slow links are
  tolerated (throughput-based stall guard instead of a fixed 10-minute clock).
  (This fix was authored for beta.7 but was stranded on the beta.6 release
  branch by a merge race — beta.7/beta.8 shipped without it.)

## [0.1.0-beta.7] - 2026-08-10

## [0.1.0-beta.6] - 2026-08-10

### Added

- **Incompatible-binary diagnosis**: self-update's smoke gate and `install.sh`
  (which now smoke-runs the staged binary BEFORE flipping `current`) report
  loader deaths in plain language — "requires a NEWER macOS" / "requires a
  NEWER glibc" with the loader's own line quoted — instead of a bare exit
  status. macOS release binaries now declare their supported floor
  (macOS 14.0), so an older Mac refuses launch cleanly instead of crashing.

## [0.1.0-beta.5] - 2026-08-08

### Fixed

- **gpt-oss / MXFP4 model metadata**: the model scanner's GGUF header reader
  (`ggus`) panicked on MXFP4 tensors — gpt-oss models cataloged with partial
  metadata (degraded autotune/fit estimates) and the node log filled with a
  repeating panic warning on every rescan. The scanner now uses `gguf-rs-lib`
  and reads only the header + metadata section, so models in any current or
  future quantization enrich fully and a single bad file can never spam the
  log or crash the scan.
- `install.sh` with `--pubkey` now fails on a missing `minisign` CLI BEFORE
  downloading the artifact, not after.

### Changed

- `higgs node install-service` output is human-readable: colored step results
  and an aligned quick-reference block (logs/state/status/stop) with
  advisories as separate marked paragraphs, instead of a wall of text.

## [0.1.0-beta.4] - 2026-08-06

### Added

- **Node connect diagnostics**: each failed hub dial now logs the FULL error
  cause chain plus an attempt counter, and roughly once a minute a "still
  unreachable — check:" block — on macOS including the Local Network
  permission recovery steps (the permission is per-binary, so a self-update
  can silently lose it). The hub serves the same platform-specific steps to
  the UI via a new `NodeView.offline_help` field, rendered on offline Fleet
  cards — nothing platform-specific is hardcoded client-side.

## [0.1.0-beta.3] - 2026-08-05

### Added

- **HF model-search catalog** (`catalog` module + jigglebot Model Search UI):
  browse mode (empty query → most-downloaded compatible GGUF repos), sort
  (downloads/likes/updated/trending), "fits this machine" filter with a
  per-quant-family footprint estimate (I-quant-aware), background real-size
  fallback when the Hub omits sizes, shard-aware default-quant preselect, and
  local + remote (fleet-node) downloads with full lifecycle logging on both
  the hub and node (`pull requested/starting/progress/done|FAILED` with
  repo/file/bytes/elapsed). Model-load phases are logged too, and the Servers
  tab shows every loaded model fleet-wide (remote worker pills + eject).

## [0.1.0-beta.2] - 2026-08-03

### Added

- **One-click fleet updates**: the jigglebot Fleet "Update" button now lists, on
  click only (never on a timer), every release newer than what the node runs
  (`node_releases` via the GitHub releases API — complete asset trio, upgrade
  only, newest first). Picking one sends the node a BARE version string
  (`M_NODE_UPDATE_VERSION`, new `update_by_version` HELLO capability); the node
  downloads manifest+signature+artifact itself from its own configured
  `release_url` (new `config.json` field, default = this repo's GitHub
  releases), re-verifies the CI minisign signature against its compiled-in
  keys, binds the authenticated manifest version to the requested one BEFORE
  the artifact download, and applies through the same verify → stage → flip →
  restart pipeline as every other update. `fleet_update_version` pushes to
  every capable node with honest per-node skip reasons. A pre-capability node
  is told precisely: on the latest → "nothing to update"; newer exists → re-run
  the installer (or a direct static-mirror manifest URL — GitHub links redirect
  and are refused by those builds).

- **Pairing preflight** (`higgs --node <ticket> <token>`): colored, gated
  self-diagnosis before dialing — per-nameserver DNS probes that name the exact
  dead resolver, ticket relay/direct-path analysis, and macOS Local Network
  guidance (with an SSH caveat: the permission prompt cannot appear over SSH).
  Pairing hard-stops only when no path to the hub can exist; every failure
  prints the specific user action that fixes it.
- **One-shot pairing with verified service handoff**: pairing saves the hub,
  best-effort restarts the installed service, and exits only after the hub
  demonstrably supersedes the pairing connection with the service's own dial
  (duplicate-identity close) — no Ctrl-C, no manual `launchctl kickstart`, never
  two node processes fighting, never zero. If no service takes over, pairing
  keeps serving in the foreground so the node stays online.
- An installed-but-unpaired node service now **waits quietly for pairing**
  (polls `config.json` every 3s; one hint line, then a reminder every ~5
  minutes) instead of exit/respawn log spam — and the wait doubles as the
  seamless handoff for a fresh install.
- Colored, user-consumable output for `install.sh` and the pairing flow
  (tty-gated ANSI; `NO_COLOR` disables).

### Changed

- Default log filter demotes iroh/transport/hickory-resolver internals to
  `error` (a `RUST_LOG` override still shows everything).
- A late service takeover while a paired foreground node is serving now exits
  that foreground process cleanly (it recognizes the hub's supersede close)
  instead of redialing into a duplicate-identity fight.

### Fixed

- Repeated pairing/preflight failures (environment problems: unreachable hub,
  dead DNS, corrupt config, malformed saved ticket) no longer spend the
  self-update rollback budget — only real boot crashes do.
- Ctrl-C/SIGTERM during pairing cancels immediately (never persists a hub or
  hands off afterwards) and exits nonzero so scripts cannot mistake a cancel
  for a successful pairing.

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
