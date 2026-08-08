# `worker/` — design notes

## Why a separate process

llama.cpp is C/C++ FFI: a bad GGUF, an OOM, or a tensor-shape mismatch can segfault or abort.
Running the engine in-process would take the whole server down. So higgs re-execs itself as a
`worker` subprocess (`worker_main()`) and talks to it over stdio. A crash kills only the worker;
the host (`supervisor.rs`) sees the non-zero exit, reports it, and can spawn a fresh worker and
replay the last load. This crash-isolation split is **non-negotiable** (see the repo's
out-of-scope list) and is why the engine never runs in the host.

## The picture — host-side catalog, worker-side engine

```
  HOST process                                          WORKER process  (higgs --higgs-worker)
  ============                                          =====================================

  Higgs facade  (api/embed.rs, api.rs)
    scan()          ─▶ ModelStore::scan (models.rs)  ── pure Rust: gguf-rs-lib + memmap2 + std::fs,
        LM Studio / HF cache / Ollama roots              NO FFI, so the catalog is HOST-side
        ─▶ Vec<HiggsModel>  (+ GGUF enrichment)
    model_entries() ─▶ HiggsModelEntry rows
         │ resolved GGUF path
         ▼
  supervisor.rs  (spawn · restart · RPC correlation)
         │
         │   M_LOAD {id, path, ctx_len, gpu_layers…}       serve_state() run loop
   stdin │   M_CHAT · M_STATUS · M_UNLOAD · M_SYSINFO ─▶      one line in ─▶ one response out
         │                                                    (single-threaded, no locks)
  stdout │ ◀── {result}  +  N_CHAT_CHUNK deltas                    │
  stderr │ ◀── engine logs                                         ▼
         ▼                                                    WorkerState
   (host matches each response/chunk to the                    ├─ engine:  Box<dyn HiggsEngine>
    in-flight request by JSON-RPC `id`)                         └─ loaded:  Option<(id, LoadParams)>
                                                                     │
                       NDJSON JSON-RPC over stdio                    ▼  llama.cpp FFI (engine/llamacpp)
```

The worker holds **no** model catalog: the host scans (`models.rs`, pure Rust) and passes the
resolved GGUF `path` in `M_LOAD`. There is no `/api/higgs/*` HTTP surface anywhere in this
picture — control is the in-process `Higgs` crate API; the supervisor is the worker's *only*
client.

## The run loop is strictly sequential

`serve_state()` reads stdin line-by-line and processes one frame at a time:

- **One request → one response**, with the matching `id`. An `M_CHAT` request additionally
  emits zero-or-more `N_CHAT_CHUNK` notifications first, then its single final response.
- **Single-threaded at the RPC boundary.** The loop serializes every request, so there is no
  in-worker concurrency and no race between, say, a chat and a model swap. (llama.cpp may use
  threads internally for decode; that is below the RPC boundary.)
- **No buffering games.** Each frame is one NDJSON line, flushed immediately (`writeln!`), so the
  host's reader gets tokens as they are produced. A broken response/chunk pipe warns rather than
  crashing — the supervisor is gone, and stdin EOF will end the loop shortly.
- **EOF ends the loop** cleanly (the host closed stdin = deliberate stop).

Because the host drives one worker per model and the loop is sequential, the worker needs no
locks of its own.

## State the worker keeps (and doesn't)

`WorkerState` holds the engine (`Box<dyn HiggsEngine>`) and `Option<(model_id, LoadParams)>` — the
resident model id and its load-time params. It deliberately holds **no model catalog**: scanning
is **pure Rust with no FFI** (`models.rs`: `gguf-rs-lib` + `memmap2` + `std::fs`), so the host scans and
resolves the GGUF path, then passes it in `M_LOAD`. A fresh worker's store would be empty, so
without host-side resolution every load would `[HG002]`. (`models.rs` lives under `worker/`
because it owns the `HiggsModel` types, but it runs host-side.)

`handle_load` clears the tracked id **before** calling `engine.load`: the engine drops the
resident model first, so if the new load then fails the `?` returns before `loaded = Some(...)`,
leaving status reporting "nothing loaded" (matching the empty engine) instead of lying about the
old id.

## Model scanning (`models.rs`) — pure Rust, host-side

`ModelStore::scan` walks three store layouts and merges the results (sort + dedup by `(id, path)`):

- **LM Studio** — `<root>/{org}/{model}/*.gguf`, id `org/model`.
- **HF cache** — `<root>/models--{org}--{name}/snapshots/{rev}/*.gguf`; revision symlinks are
  `canonicalize`d so two revisions pointing at the same blob collapse via dedup. Id `org/name`.
- **Ollama** — `<root>/manifests/**` JSON manifests → the `vnd.ollama.image.model` layer's blob
  under `blobs/sha256-<hex>`; GGUF magic is validated. Id `ollama/{name}:{tag}` (no HF id
  fabricated). Per-file read/JSON errors skip the file (debug log), not the whole scan.

Each candidate is enriched from the GGUF header (`enrich_gguf_metadata` → `enrich_from_gguf`):
`arch`, `ctx_train`, the autotune KV-VRAM inputs (`block_count`, `head_count`, `head_count_kv`,
`embedding_length`, `expert_count`), chat-template presence + tool/reasoning heuristics, and the
curated `gguf_components` list for the UI. The chat template is captured transiently
(`serde(skip)`, host-only) for the Gate-2 tool-parser sniff.

### Scan invariants / guards

- **Enrichment is error-first, and still panic-caught.** The `gguf-rs-lib` reader returns
  errors on malformed/truncated input and knows the modern tensor types (MXFP4/NVFP4 —
  the previous `ggus` dependency hit a literal `todo!()` on gpt-oss's MXFP4 and panicked
  on every rescan). The whole enrichment still runs inside `catch_unwind(AssertUnwindSafe(...))`
  as defense-in-depth: a misbehaving file stays cataloged with whatever fields were set —
  one bad file never crashes the scan.
- **A corrupt/unreadable header is not an error** — the enrichment fields stay `None`/`false`, the
  model stays in the catalog. No new error code is raised there.
- **Projector sidecars are excluded.** `is_projector_sidecar` drops `mmproj-*.gguf` /
  `general.architecture == "clip"` files (filename check before the mmap; arch check after enrich)
  — they share the parent repo's id and are not standalone servable models.
- **Higgs never writes** into another app's model storage; scanning is strictly read-only.

## Notable RPC guards (`mod.rs`)

- **`M_CHAT` model bind (HG018).** The host resolves a model, then dispatches; a concurrent load
  could swap the resident model in between. `M_CHAT` carries the resolved `model` and the worker
  rejects with `[HG018] ResidentModelMismatch` (→ retryable 503) if it no longer matches the
  resident id — a swap errors instead of silently serving the wrong model. An absent/empty `model`
  means "no check" (back-compat); the serve layer always sends it now. (Post-P4b the local node
  runs one worker per model for its lifetime, so this rarely fires locally, but it remains the
  authoritative cross-process check.)
- **Chat before load (HG003).** `handle_chat` with nothing resident returns `[HG003]
  ModelNotLoaded` without touching the engine.
- **`ctx_len = 0` coercion.** `handle_load` coerces a 0 context to the default (4096), so a model
  can't load into an unusable 0-sized window that fails every fit-check.
- **Malformed overrides degrade, don't fail.** Bad `LlamaCppParams` in `M_LOAD` and a
  bad/absent `sampling` object in `M_CHAT` fall back to defaults (with a warn / legacy top-level
  `temperature`) rather than 500ing an older or garbled caller.
- **Error codes cross the boundary.** `to_rpc_error` encodes any engine `HG*` code into the
  JSON-RPC error's `data.code` (app errors as `-32000`); a decode failure rides `HG008` and an
  unknown method rides `HG037` (`method_not_found`) — so the host's `http_status` table maps each
  to its true status (404 / 400 / 503 / 501) instead of a blanket 500.

## Error codes this module owns

| Code | Raised by | Meaning |
|------|-----------|---------|
| `HG001` | `models.rs` scanners (`ModelDirUnreadable`) | a model root exists but a directory read failed (NotFound is skipped silently). |
| `HG010` | `models.rs` `scan_ollama` (`OllamaManifestInvalid`) | an Ollama model layer has a missing/mis-formatted `digest`. |
| `HG018` | `mod.rs` `handle_chat` (`ResidentModelMismatch`) | requested model no longer resident (concurrent swap) → retryable 503. |
| `HG003` | `mod.rs` `handle_chat` (`ModelNotLoaded`) | chat with nothing loaded. |

`HG037` (unknown method) and `HG008` (decode failure) are surfaced from `mod.rs` via shared
`rpc` helpers; engine errors (HG004/HG005/HG011/…) are relayed with their codes intact.

## Pluggability

The engine is chosen at startup by `HIGGS_ENGINE` (default `llamacpp`) via the `engine/`
registry (`build_engine`). A new backend (e.g. MLX) is one registry line + a new
`engine/<name>/` submodule implementing `HiggsEngine`; nothing else in higgs changes — the trait
is the boundary.

## Boundaries / what does NOT belong here

- Worker process lifecycle, spawn, RPC correlation, restart FSM → `../supervisor.rs` (the host).
- Multi-worker orchestration + remote routing → `../node`.
- The HTTP surface → `../serve`.
- The engine internals + the FFI → `engine/` (its own README/DESIGN).
