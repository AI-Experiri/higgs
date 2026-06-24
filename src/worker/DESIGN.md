# `worker/` — design notes

## Why a separate process

llama.cpp is C/C++ FFI: a bad GGUF, an OOM, or a tensor-shape mismatch can segfault or abort.
Running the engine in-process would take the whole server down. So higgs re-execs itself as a
`worker` subprocess and talks to it over stdio. A crash kills only the worker; the host
(`supervisor.rs`) sees the non-zero exit, reports it (`[HG006]`-class), and can spawn a fresh
worker and replay the last load. This crash-isolation split is **non-negotiable** (see the
repo's out-of-scope list) and is why the engine never runs in the host.

## The run loop is strictly sequential

`serve_state()` reads stdin line-by-line and processes one frame at a time:

- **One request → one response**, with the matching `id`. An `M_CHAT` request additionally
  emits zero-or-more `N_CHAT_CHUNK` notifications first, then its single final response.
- **Single-threaded at the RPC boundary.** The loop serializes every request, so there is no
  in-worker concurrency and no race between, say, a chat and a model swap. (llama.cpp may use
  threads internally for decode; that is below the RPC boundary.)
- **No buffering games.** Each frame is one NDJSON line, flushed immediately, so the host's
  reader gets tokens as they are produced.
- **EOF ends the loop** cleanly (the host closed stdin = deliberate stop).

Because the host drives one worker per model and the loop is sequential, the worker needs no
locks of its own.

## State the worker keeps (and doesn't)

`WorkerState` holds the engine and `Option<(model_id, LoadParams)>` — the resident model id and
its load-time params. It deliberately holds **no model catalog**: scanning is pure Rust
(`models.rs`: `ggus` + `memmap2` + `std::fs`, no FFI), so the host scans and resolves the GGUF
path, then passes it in `M_LOAD`. A fresh worker's store would be empty, so without host-side
resolution every load would `[HG002]`. (`models.rs` lives under `worker/` because it owns the
`HiggsModel` types, but it runs host-side.)

## Notable guards

- **`M_CHAT` model bind (HG018).** The host resolves a model, then dispatches; a concurrent
  load could swap the resident model in between. `M_CHAT` carries the resolved `model` and the
  worker rejects with `[HG018]` (→ retryable 503) if it no longer matches the resident id —
  so a swap errors instead of silently serving the wrong model. (Post-P4b the local node runs
  one worker per model for its lifetime, so this guard rarely fires locally, but it remains the
  authoritative cross-process check.)
- **`ctx_len = 0` coercion.** `M_LOAD` coerces a 0 context to the default, so a model can't load
  into an unusable 0-sized window that fails every fit-check.
- **Error codes cross the boundary.** Engine errors carrying an `HG*` code are encoded in the
  JSON-RPC error's `data.code` so the host's `http_status` table maps them correctly.

## Pluggability

The engine is chosen at startup by `HIGGS_ENGINE` (default `llamacpp`) via the `engine/`
registry. A new backend (e.g. MLX) is one registry line + a new `engine/<name>/` submodule
implementing `HiggsEngine`; nothing else in higgs changes — the trait is the boundary.

## Boundaries / what does NOT belong here

- Worker process lifecycle, spawn, RPC correlation, restart FSM → `../supervisor.rs` (the host).
- Multi-worker orchestration + remote routing → `../node`.
- The HTTP surface → `../serve`.
