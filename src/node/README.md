# `src/node/` — remote-worker fleet over iroh

Pair two `higgs` instances over iroh QUIC so a **hub** can borrow the GPUs of remote
**nodes**. A hub is the `higgs` server clients hit on `/v1`; a node is a `higgs` that dials
out and runs real llama.cpp child workers on the hub's behalf. The hub never talks to a
remote worker directly — **two hops, never collapsed**:

```
hub ──iroh──▶ node ──stdio──▶ worker (llama.cpp child)
```

See `DESIGN.md` (this folder) for the why/invariants and `../../docs/DESIGN-remote.md` for the
full spec.

## Files

| File | Responsibility |
|---|---|
| `mod.rs` | **Not a barrel** — it declares the child modules AND owns the live-iroh handshake surface: `HELLO_DEADLINE`, `HubIdentity`, `GateOutcome`, the two-phase hub gate (`gate_read_hello` lock-free + `gate_admit` locked, wrapped by `gate_connection`), the node dialers (`connect_node`, `dial_and_hello`, `send_leave`), the persistent node serve loop (`serve_node` → `handle_node_stream` + the `AbortOnDrop`-guarded `relay_worker_logs` / `relay_fleet_events` uni-stream relays), and the shared frame helpers (`write_frame`, `worker_origin_code_data`, `close_after_reject`). |
| `runtime.rs` | `NodeRuntime` — the net-new multi-worker orchestrator, an **actor** (private `WorkerRegistry<Arc<Supervisor>>`, no mutex). Lifecycle (`load`/`unload`/`kill`/`status`/`scan`/`sysinfo`/`inventory`), idle auto-unload reaper (`IdleConfig`), lease-based chat (`ChatLease`), cancellation-safe teardown (`StopOnDrop` + tracked `reap` + `shutdown_all`), log + lifecycle-event fan-out, and the T10 fleet-event fan-out: every worker-state change (chat start/end, load, unload, reap) emits a `NodeFleetEvent` carrying the full worker snapshot, sequenced by the actor's `snapshot_seq` (shared with `Inventory` pulls, so mailbox order IS data order); `FleetResnapshot` re-emits a fresh `Resync` snapshot for the relay's stream-failure recovery. |
| `hub.rs` | Production hub listener: `start_hub` binds the endpoint, seeds/reuses the `HubFleet`, spawns `spawn_accept_loop` (gates each dial, registers admitted nodes), and returns the live `Hub` (mint pairings, `retire`/`set_label`/`labels`, `shutdown`). Also `serve_node_requests` (hub side of node self-`leave`). |
| `fleet.rs` | `HubFleet` — the hub's fleet read-model + `model→(node,worker)` routing table, an **actor** (was 7 mutexes). Node admission/retire, durable instance routes, per-node epochs, served-id derivation, the atomic `nodes_view`, the remote ops `scan_node`/`load`/`unload`/`kill`/`chat`/`refresh_inventory`, and the T10 event side: `read_node_notifications` folds node-pushed `N_FLEET_EVENT`s into the inventory cache (seq/transport/capability guarded, with pending-push retention + one bounded fallback pull) and re-broadcasts `FleetEvent`s — every emit happens INSIDE the actor handler that performs the state change, so event order equals state order. `NodeView`/`FleetEvent` are the ts-rs UI wire types. |
| `transport.rs` | `NodeTransport` — hub-side per-node client over one live iroh `Connection`: `request()` (one `higgs/node/*` control RPC per bidi stream) and `chat()` (relay `M_CHAT`, stream `N_CHAT_CHUNK` + final). One stream per call is the demux. |
| `data.rs` | Node-side DATA relay: `relay_chat` bridges a hub chat stream to `Supervisor::chat()` (via a `ChatLease`), `relay_pull` downloads a GGUF (`M_NODE_PULL`) streaming `N_PROGRESS`. Single writer per stream; cancelled on connection/stream drop. |
| `control.rs` | Node-side CONTROL dispatch: `dispatch_node_control` maps a `higgs/node/*` request to a `NodeRuntime` op and builds the JSON-RPC reply (carrying the origin HG code in `data`). Also owns `accept_node_update` + `DeferredUpdate` — the `M_NODE_UPDATE` receive path, deliberately NOT a `NodeRuntime` op: `handle_node_stream` special-cases it, replying `accepted` first and only then spawning the detached apply. |
| `self_update.rs` | Node self-update receiver (REL-P4, §9): verify the pushed manifest (`verify_manifest_any`, every compiled-in pin) → eligibility (`evaluate_eligibility`; a hub push is upgrade-only) → SSRF-vetted artifact fetch → sha256 check → stage → smoke (`--version`) → atomic flip of `bin/current` → boot-guard auto-rollback on a crash-loop, plus the manual `--rollback` and version-dir `--prune` paths (`apply_pushed_update` is the detached hub-push apply). |
| `release_courier.rs` | Hub-side update courier (`Higgs::node_update`/`fleet_update`): fetches the (tiny) signed manifest + `.minisig` from a static HTTPS origin (SSRF-vetted, no redirects, size/time-capped), derives the direct `artifact_url`, and hands `HubFleet::push_update`/`push_update_pinned` the `M_NODE_UPDATE` payload. The hub is a courier only — the node re-verifies signature + sha256 against its own pins. |
| `served.rs` | `served_ids` — pure, deterministic served-instance-id derivation (`org/model`, `org/model-1`, …), generic over instance location so the remote fleet and the local engine share one algorithm. |
| `identity.rs` | Persisted ed25519 `SecretKey` → stable `EndpointId` (`load_or_create_secret`, atomic temp+hard_link publish, `0600`); `bind_endpoint` (N0 relays by default, `HIGGS_IROH_LOCAL` → relay-disabled LAN mode for tests). |
| `node_id.rs` | `NodeId(u32)` (Copy) + `NodeIdAllocator` — the hub's stable per-paired-node handle (`n-1`), distinct from the long `EndpointId`; used for `LogSource::RemoteWorker` and the UI. |
| `worker_id.rs` | `WorkerId(u32)` (Copy) + `WorkerRegistry<T>` — the node's monotonic, never-reused worker ids (`reserve`/`insert_reserved` for the load spawn-and-commit). |
| `cli.rs` | Hand-drive CLI: `higgs link pair/status` (hub), `higgs node connect/leave/install-service/self-update` (one-shot), `higgs --node [<ticket> [token]] | --list | --hub <sel>` (persistent daemon: dial → HELLO → `serve_node` → reconnect with backoff, saved hubs in `config.json`). `higgs node self-update [--url <manifest-url> | --tarball <f> --manifest <f> --manifest-sig <f>] [--allow-downgrade] [--dry-run] [--rollback] [--prune]` drives the same verified atomic swap by hand, and the persistent daemon runs `self_update`'s boot-guard at start (records the boot attempt, auto-rolls-back a crash-looping just-updated binary, confirms alive once serving). `install-service` resolves the OPERATOR from the passwd database (SUDO_USER-aware; a nested `SUDO_USER=root` is refused), then writes+activates `service.rs`'s plan; `--dry-run` prints it without touching anything. The DEFAULT (LoginBound) refuses root on BOTH OSes (macOS agent = your gui domain; Linux user unit = your manager); only `--system` on macOS REQUIRES root (LaunchDaemon dir), while Linux `--system` still refuses it (linger is a user-level flag). The systemd unit file is written to a STABLE absolute path in the operator's own tree (`~/.higgs/higgs-node.service`) and installed by ENABLING IT BY ABSOLUTE PATH (`systemctl --user enable <abspath>`, which links-if-outside-a-search-dir and enables, and just enables when the prefix already falls inside one) — so systemd decides where it lives (no `$XDG_CONFIG_HOME`/`UnitPath` guessing, and no `systemctl` runs during discovery). The `~/.higgs/logs` dir + `node.log` are CREATED AS THE OPERATOR (uid/gid-dropped `mkdir`, a DIR-writability probe that create+unlinks a temp so a stale root-owned `logs/` — which `mkdir -p` no-ops over — can't block node.log recreation after log rotation, and a READ/WRITE non-truncating open `sh -c 'exec 3<> …'` of node.log), never chowned by root — so a planted ancestor symlink resolves under the operator's own permissions (no privesc), and a stale root-owned/mode-restricted log is caught by the open failing (a bare `touch` would pass a mode-0400 file the daemon still can't write). The probe is R/W by DELIBERATE safe-failure policy: sources disagree on launchd's exact open mode for a shared StandardOut/Error path, and R/W can only refuse a pathological mode-0200 log with a clear error rather than let the daemon silently fail to open stderr. Every privileged subprocess (`systemctl`/`launchctl`/`loginctl`/`sh`/`mkdir`) is spawned with a pinned root-owned `PATH`, so a bare name can never resolve to a planted binary under sudo. The unit file is written atomically (temp + rename). |
| `service.rs` | Node service (REL-P2): pure renderers + `plan_install` → `ServicePlan` (file to write, argv commands, notes), driven by ONE cross-platform dial `ServiceScope { LoginBound (default), SurvivesLogout (--system) }` (+ a commented FUTURE `KeepAwake` sleep-inhibit variant). USER-SPACE BY DEFAULT: LoginBound = macOS LaunchAgent in `~/Library/LaunchAgents` (gui/<uid> domain, no sudo, stops at logout — a locked screen is fine) / Linux systemd USER unit with NO linger (zero prompts). `--system` (SurvivesLogout) = macOS LaunchDaemon pinned via `UserName` (sudo; tears down a leftover agent — gui bootout + as-operator plist rm — so a scope switch never runs two nodes; the reverse switch REFUSES while the daemon plist exists, with the exact sudo cleanup) / Linux the SAME user unit + best-effort `loginctl enable-linger` (never auto-disabled on downgrade; a note surfaces it). `Restart=always`/`RestartSec=5`/`StartLimitIntervalSec=0` (a crash-looping binary must never brick the unit). ExecStart escapes `\`/`"`/`%` and is quoted; `append:` log paths are `%`-escaped and UNquoted (that directive takes the line verbatim). All variants exec through `<prefix>/bin/current/higgs` — the atomic symlink `install.sh` flips (stage+rename, never truncating a live binary), so updates/rollbacks never edit the service files. |

`*_tests.rs`, `e2e_tests.rs`, and `test_support.rs` are the test sidecars (see the layout
rule in `../../CLAUDE.md`).

## Public surface (what the rest of the crate uses)

- **`hub::{start_hub, Hub}`** — there is **no `/api/higgs/*` HTTP control surface**; the embedder's
  `Higgs` facade (`../api/embed.rs`) drives the hub via the crate API. `Higgs::hub_enable()` calls
  `start_hub(bus, existing_fleet)` and holds the `Hub` alive; the rest of `Hub`'s methods back
  facade calls, not routes: `mint_pairing()`/`hub_id()` ← `Higgs::pair()`, `retire()` ←
  `node_retire()`, `set_label()` ← `node_label()`, `labels()` ← `nodes()`, `shutdown()` ←
  `hub_disable()` (the kill switch), and `serve_node_requests` accepts a node's self-`leave` on its
  own connection.
- **`fleet::{HubFleet, NodeView, NodeKey}`** — the `Higgs` facade routes `/v1` chat through the
  fleet (`api.rs`): `is_remote(served)` (remote-vs-local decision) + `resolve` / `chat(...)`
  (relay); `routed_models()` is folded into `Higgs::chat_model_ids()` for `GET /v1/models`;
  `nodes_view()` ← `Higgs::nodes()` (Fleet view, merged with allowlist labels + the local node);
  `load`/`unload` ← `node_load`/`node_unload` (`kill` is the force-unload variant);
  `served_on(node)` + `resolve` + `chat_pinned` ← `node_chat_test` (the Fleet view's
  per-node link proof — always relayed, never local: the `"local"` sentinel is refused
  outright [HG076], and the pin rides the same
  resolution that picks the transport so a concurrently re-homed id is refused
  [HG077], never mis-attested); `disconnect_all()` ← `hub_disable()` (kill
  switch); `subscribe_fleet_events()` ← `Higgs::subscribe_fleet_events()` (the live fleet-event
  broadcast the embedder's UI lane forwards — see "Live fleet events" below);
  `push_update`/`push_update_pinned`/`update_targets` ← `Higgs::node_update`/`Higgs::fleet_update`
  (via `release_courier`, the hub-side self-update push). `NodeView` and
  `FleetEvent` derive ts-rs bindings.
- **`runtime::{NodeRuntime, NodeConfig, IdleConfig, DEFAULT_IDLE_TTL}`** — the node daemon owns a
  `NodeRuntime`; the local single-machine engine also uses it as its own multi-worker orchestrator
  (`instances()` feeds served-id derivation, `events()`/`subscribe_logs()`/`bus()` feed the SSE +
  Developer-Log surfaces, `idle()` wires Server-Settings auto-unload).
- **`served::served_ids`** — reused by both the fleet and the local engine (P4b).
- **`{connect_node, dial_and_hello, send_leave, serve_node, gate_connection, HubIdentity,
  GateOutcome, HELLO_DEADLINE}`** — the transport handshake, used by `cli.rs` and `hub.rs`.
- **`identity::{load_or_create_secret, bind_endpoint}`**, **`node_id::{NodeId, NodeIdAllocator}`**,
  **`worker_id::{WorkerId, WorkerRegistry}`** — the identity/id primitives.

## Live fleet events (T10)

A node whose HELLO advertises the `fleet_events` capability PUSHES its worker-state changes
instead of waiting to be polled: each change emits an `N_FLEET_EVENT` notification (kind +
full worker snapshot + actor `snapshot_seq`) on a dedicated uni stream. The hub merges the
snapshot into its inventory cache under the same seq ordering as pulls and re-broadcasts a
`FleetEvent {endpoint_id, kind}` — a pure invalidation signal (subscribers re-read
`nodes_view`). Hub-local kinds (`NodeConnected`/`NodeDropped`/`InventorySynced`/
`HubStateChanged`) announce admissions, drops, committed pulls, route changes, and the kill
switch; the chat-end debounced re-pull survives only as the legacy fallback for admissions
that did not declare the capability. See `DESIGN.md` § "Live fleet events" for the guard
stack and residuals.

## Auth (two surfaces)

- **Surface A (machines):** `EndpointId` allowlist (`~/.higgs/pairings.json`) + one-time pairing
  tokens (`../auth.rs`). The dialer's TLS `remote_id()` proves it holds the key; the self-declared
  HELLO `node_id` must equal it (anti-spoof).
- **Surface B (callers):** Bearer API keys on `/v1` — outside this folder.

## Trying it (two terminals)

```sh
higgs link pair                 # hub: mint a pairing ticket + token, listen
higgs --node <ticket> <token>   # node: dial the hub, serve it, persist it for next time
```

## Tests

- Unit tests: the `_tests.rs` sidecars (registry, dispatch, gate, served-ids, fleet actor).
- End-to-end over a **real spawned `higgs` process**: `../../tests/remote_node_e2e.rs`,
  `../../tests/remote_pairing.rs`, `../../tests/remote_hub_e2e.rs` (route survives reconnect,
  kill switch), plus the in-process `e2e_tests.rs`.
</content>
</invoke>
