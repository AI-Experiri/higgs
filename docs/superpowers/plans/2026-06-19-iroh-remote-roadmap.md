# iroh Remote-Worker — Phase Roadmap

> **Source spec:** `DESIGN-remote.md` (Rev 2, committed). This roadmap is the spine
> for the whole feature. **Each phase gets its own detailed plan file**, written
> *just-in-time* against the real post-refactor code (so file:line refs stay accurate),
> executed task-by-task with a codex review loop until it converges, then the next
> phase is planned.

**Why phase-by-phase planning, not one mega-plan:** The spec's line anchors
(`supervisor.rs:148`, the 7 `proc` reap sites, etc.) shift the moment P0 extracts
`actor.rs`. P3's seams depend on what P0/P2 produce. Writing detailed P5 code today
would be fiction. So we plan each phase right before building it.

**Quality gate for every phase:** `scripts/coverage.sh` (cargo llvm-cov, `--fail-under-lines 90`).
No phase is "done" until the gate is green AND the codex review loop has converged.

---

## Phase ledger

| Phase | Task # | Plan file | Goal (exit criteria) | New deps | Status |
|---|---|---|---|---|---|
| **P0** | — | `2026-06-19-P0-actor-runtime.md` | `src/actor.rs`: `trait Actor`+`spawn_actor`+shared `ReplyDemux`, factored out of `Supervisor` and adopted by it. **No behaviour change, all existing tests green.** (Worker async port deferred to P2 — no consumer + current-thread-runtime hazard; validated by codex.) | none (tokio only) | **DONE** |
| **P1** | #12 | `2026-06-19-P1-pairing-handshake.md` | iroh `Endpoint`+persisted `SecretKey` (`~/.higgs/endpoint.key`); `src/auth.rs` allowlist + `pairings.json`; pairing-token mint/burn; HELLO frame + version negotiation; post-HELLO gate + HELLO-stalled timer; `link pair/status` + `node connect` CLI. HG022/023/024/028. **Exit (met):** two binaries pair, HELLO agrees, stranger rejected (HG024), silent peer dropped (HG028), version mismatch typed-closed (HG023); 5 pairing integration tests + unit tests; gate 90.41%. | `iroh`, `iroh-base`, `iroh-tickets` 1.0 | **DONE** |
| **P2** | #13 | `2026-06-19-P2-node-runtime.md` | `NodeRuntime{ HashMap<WorkerId,Arc<Supervisor>> }` + `WorkerId(u32)`; `higgs/node/*` CONTROL dispatch; `--node` persistent daemon (dial → HELLO → serve control, graceful drain, reconnect); `M_LOAD` spawns 2nd+ Supervisor (path resolve + RAM guard + ctx_train default + restart replay + cancellation-safe); `M_KILL`/`M_UNLOAD` free `WorkerId`. Chat DATA relay → **P3** (pairs with hub side). **Exit (met):** e2e test — node hosts 2 concurrent workers, `M_SYSINFO`+`M_STATUS` over iroh; gate lines 90.14% (⚠ thin — `node/cli.rs` daemon ~22%, needs black-box coverage before/with P3). | none | **DONE** |
| **P3** | #14 | done | node + hub chat relay (`data::relay_chat`, `NodeTransport`, `HubFleet` routing, `/v1`→remote, HG027) + **production hub listener** (`node/hub.rs`: `start_hub` accept loop, `HIGGS_HUB=1`, `POST /api/higgs/pair`, fleet seeded from pairings) + e2e (`remote_node_e2e`, `remote_hub_e2e`, `hub_server`). | none | **DONE** |
| **P4** | #15 | done | `M_NODE_INVENTORY`/`NodeView`/`GET /api/higgs/nodes` (epoch-guarded cache; HW/RT `Deserialize`); `LogSource::RemoteWorker{node,worker}` (`Copy`) + keyed rings + eviction + `?source=node:<id>:<worker>`; `N_LOG_LINE` relay. e2e: relay assertion + inventory in `remote_hub_e2e`. | none | **DONE** |
| **P4b** | #15b | done | `src/download.rs` HF downloader (injectable `Fetcher` + `HttpFetcher`, `HIGGS_HF_ENDPOINT`) → `~/.higgs/models/` ONLY, scanner-layout enforced, atomic; `M_NODE_PULL` data relay + `N_PROGRESS`; HG025. e2e `tests/pull.rs` (local HTTP server, offline). | `reqwest` | **DONE** |
| **P5** | #16 | done | `src/keys.rs` `api_keys.json` (SHA-256 + constant-time, scopes chat\|models\|admin); serve bearer middleware + 401 envelope; `keys` CLI; standalone fails-closed. e2e `tests/auth.rs`. | `sha2`, `subtle` | **DONE** |
| **P6** | #17 | backend done | `GET /api/higgs/nodes` fleet view + `POST /api/higgs/pair` shipped (P3/P4). The Svelte UI panel + QR + ts-rs exports live in the **jigglebot** repo (this standalone crate exposes the API contract). | `qrcode` (UI repo) | **backend DONE** |
| **#18** | #18 | done | `M_NODE_UPDATE` SHIPPED (REL-P1..P4e): CI-signed manifests + pinned keys, node receive+verify+stage+trial-flip+re-exec + boot-guard rollback, hub courier, drain, `update_failed`-over-HELLO, jigglebot UI. `update` capability `true`; a legacy node refuses `HG026`. | — | **DONE** |

---

## Execution protocol (per task, every phase)

1. Implement the task (TDD: failing test → minimal impl → green).
2. Run the task's tests; then the full gate `scripts/coverage.sh` when the task closes a unit.
3. **Codex review loop:** submit the diff to codex review; address findings; re-review;
   repeat **until it converges** (no further actionable findings).
4. Commit. Move to the next task.
5. When a phase's tasks are all done + gate green + review converged, mark it in the
   ledger and write the next phase's plan against the now-current code.

## Architecture correction (P2 planning, codex-validated 2026-06-19)

The spec's §5.3 "bridge iroh into `serve_state` via `SyncIoBridge`" on the node, and the
§2.5 "Worker rides spawn_actor / minimal tokio runtime," do **not** survive the real
two-hop topology. The node drives real child workers through `Supervisor`, which already
bridges to the child's sync stdio. So:

- **Node data path = relay through `Supervisor.chat()`/`request()`** (async), not
  `SyncIoBridge` into `serve_state`. Strictly simpler and faithful to the process model.
- **The worker child stays pure-sync stdio forever.** Its parent is always a real process
  piping stdio (local `Higgs` or a node `Supervisor`); the iroh boundary lives at the
  hub's Supervisor transport (P3) and the node's relay (P2), never inside the worker.
  The "P0-deferred worker tokio port" is therefore **eliminated, not deferred** — it was
  never needed.
- **Scope (P2 Task 5, decided during impl):** the node-side **chat data relay** moves to
  **P3**, where the hub chat side exists to drive it end-to-end (P2 alone can't e2e-test a
  relay). P2 delivers the control-plane daemon (dial → HELLO → serve `higgs/node/*`), which
  is exactly what the P2 exit (sysinfo+status over iroh) needs.
- Relay risks to handle (codex), for P3: one writer task per stream (chunks then final, no race);
  hub-stream `request_id` vs Supervisor-local id are separate namespaces; on stream close,
  drop the chunk receiver (cancel in-flight where possible); on worker error, emit chunks
  then a final error frame.

## Cross-phase invariants (from the spec — never violate)

- **Two hops, never one:** hub→node→worker. The hub never speaks to a remote worker directly.
- **Two correlation domains, never merged:** hub's local-worker `Supervisor` vs hub's per-node iroh transport (§2.3).
- **Two dispatchers, never one:** `NodeRuntime` (control `higgs/node/*`) vs `serve_state` (data `M_CHAT`/`N_CHAT_CHUNK`).
- **`WorkerId` LOCKED to `u32`** (Copy) ⇒ `LogSource` stays `Copy`.
- **Invent nothing where a standard exists;** reuse `RpcFrame`/`HalvesFactory`/`Supervisor`-as-per-worker-unit verbatim. `M_PULL` is the one honestly net-new subsystem.
- **One-way crate dependency:** `auth.rs`/`node.rs` use the crate's own serde; no `common`/`engine`/jigglebot import.
