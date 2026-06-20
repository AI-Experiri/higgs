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
| **P2** | #13 | `2026-06-19-P2-node-runtime.md` | `NodeRuntime{ HashMap<WorkerId,Arc<Supervisor>> }` + `WorkerId(u32)`; node-side CONTROL dispatch (`higgs/node/*` registry ops over the control stream); node DATA relay (iroh `M_CHAT` → `Supervisor.chat()` → stream back — **NOT** `SyncIoBridge`/`serve_state`, see correction below); `--node` persistent daemon (dial → HELLO → control loop + data accept loop); `M_LOAD` spawns 2nd+ Supervisor w/ VRAM fit-check; `M_KILL`/`M_UNLOAD` free `WorkerId`. **Exit:** node hosts 2 concurrent workers, `M_SYSINFO`+`M_STATUS` over iroh. | `toml` (only if config file needed) | **planned** |
| **P3** | #14 | _(plan after P2)_ | `remote_factory`+`spawn_remote` ctor; `WorkerProc` trait replacing `proc:Option<Child>` across the 7 reap sites; hub per-node iroh transport (own pending/correlation, keyed by `NodeId`, route per `(NodeId,WorkerId)` — distinct from local-worker `Supervisor`); hub chat relay (request_id translation table); wedged-worker reap. HG027. **Exit:** `/v1` chat to remote-resident worker streams back; wedged worker escalates `M_KILL`→redial→retire (HG027). | none | pending |
| **P4** | #15 | _(plan after P3)_ | `M_INVENTORY{boot\|refresh}`/`M_SCAN`; `HashMap<NodeId,NodeView>` (+hostname/os/IP §4.2.1; HW/RT gain `Deserialize`); `LogSource::RemoteWorker{node,worker}` (stays `Copy`) + keyed remote ring map + eviction on unload/kill/retire; `LogSource::parse` node arm; `N_LOG_LINE` relay; `?source=node:<id>:<worker>` selector. **Exit:** 2 nodes' logs interleave+filter by (node,worker); killed worker's ring reclaimed. | none | pending |
| **P4b** | #15b | _(plan after P4)_ | NEW HF downloader: `hf-hub`→`~/.higgs/models/` ONLY; `N_PROGRESS` on data plane; HG025. Never writes scanned dirs. **Exit:** `M_PULL` downloads a GGUF, progress streams, subsequent `M_SCAN`/`M_LOAD` sees it. | `hf-hub` (pulls `reqwest`) | pending |
| **P5** | #16 | _(plan after P4b)_ | `api_keys.json` + SHA-256 middleware on `/v1`+`/api/higgs/*`; scopes (chat\|models\|admin); `keys` CLI. 401 envelope. **Exit:** scoped key allows chat, admin gates node mgmt. | `sha2`, `subtle` | pending |
| **P6** | #17 | _(plan after P5)_ | `/api/higgs/nodes` panel (fleet view incl. `observed_addr`+`path`); pairing QR flow; per-node+per-worker load/unload/kill; keys pane; ts-rs exports (`NodeView`/`NodeInventory`/`NodePath`/`HelloResult`/`LogSource`). **Exit (Playwright):** pair a node, load 2 workers, chat, see logs. | `qrcode` (optional) | pending |
| **#18** | #18 | _(reserved)_ | `M_UPDATE` const + capability + pubkey table + HG026 stub only. Real updater = later separate task. | `minisign-verify` (deferred) | reserved |

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
- Relay risks to handle (codex): one writer task per stream (chunks then final, no race);
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
