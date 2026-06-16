# higgs

## Table of Contents
- [What It Is](#what-it-is)
- [Files](#files)

## What It Is

higgs is jigglebot's in-app local model runtime. It is a standalone Rust crate — it imports nothing from any other jigglebot crate. Point it at a directory of GGUF files and it runs OpenAI-compatible inference in the same process, accessible over a regular HTTP server.

In production, higgs runs **embedded in-process** inside the jigglebot server: the `backend/server/src/higgs/` launcher owns the `Arc<Higgs>` and serves higgs's own router on an ephemeral `127.0.0.1` port. The standalone `higgs-server` binary still exists, but only for dev / standalone use. Either way, the control surface (router + facade) is pure Rust and cannot crash the host — only the worker is a separate process.

The inference engine (llama.cpp) runs inside a **worker process** created by re-executing the host binary with `--higgs-worker`. The worker speaks newline-delimited JSON-RPC 2.0 on stdio. The worker is **spawned on load and killed on unload**: with nothing loaded there is zero worker process (zero idle RAM); loading a model spawns exactly one worker named `higgs(<model>)`. If a live worker crashes mid-use, the supervisor restarts it once and replays the last load, so the model is available again without host intervention. Model scanning runs host-side (pure Rust, no worker), so the model list is always available even with no worker.

Because higgs carries its own HTTP router, any Axum host can mount it with one call and get a `/v1/chat/completions` endpoint backed by a local GGUF model.

## Files

| Path | What it does |
|------|-------------|
| `src/lib.rs` | Crate root — re-exports `Higgs`, `HiggsConfig`, `HiggsError`, `HiggsEvent` |
| `src/api.rs` | `Higgs` facade and `HiggsConfig` — the host-facing API; `HiggsStatus`, `LoadedInfo`, `ChatOutcome`, `HiggsServerConfig` (read-only effective config surfaced by `Higgs::server_config()`) |
| `src/diagnostic.rs` | `HiggsError` enum with diagnostic codes HG001–HG011 |
| `src/rpc.rs` | NDJSON JSON-RPC 2.0 encode/decode — the supervisor↔worker wire protocol |
| `src/supervisor.rs` | Worker process supervisor: spawn, restart, request correlation, chat-chunk routing |
| `src/serve/mod.rs` | Axum router: `/v1/models`, `/v1/chat/completions`, `/api/higgs/*` control routes |
| `src/serve/stream.rs` | SSE assembly for streaming chat completions (OpenAI chunk protocol) |
| `src/worker/mod.rs` | Worker entry point — JSON-RPC dispatch loop, `worker_main()` |
| `src/worker/models.rs` | Model discovery across LM Studio, HuggingFace cache, and Ollama stores; `HiggsModel`, `ModelStore` |
| `src/worker/engine/mod.rs` | `HiggsEngine` trait: `load`, `unload`, `is_loaded`, `chat` |
| `src/worker/engine/llamacpp.rs` | llama.cpp engine impl: GGUF-embedded chat templates, context fit-check, token streaming |
