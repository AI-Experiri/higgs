# higgs

## Table of Contents
- [What It Is](#what-it-is)
- [Files](#files)

## What It Is

higgs is jigglebot's in-app local model runtime. It is a standalone Rust crate — it imports nothing from any other jigglebot crate. Point it at a directory of GGUF files and it runs OpenAI-compatible inference in the same process, accessible over a regular HTTP server.

The inference engine (llama.cpp) runs inside a **worker process** created by re-executing the host binary with `--higgs-worker`. The worker speaks newline-delimited JSON-RPC 2.0 on stdio. If the worker crashes, the supervisor restarts it once and replays the last scan and load, so the model is available again without host intervention.

Because higgs carries its own HTTP router, any Axum host can mount it with one call and get a `/v1/chat/completions` endpoint backed by a local GGUF model.

## Files

| Path | What it does |
|------|-------------|
| `src/lib.rs` | Crate root — re-exports `Higgs`, `HiggsConfig`, `HiggsError`, `HiggsEvent` |
| `src/api.rs` | `Higgs` facade and `HiggsConfig` — the host-facing API; `HiggsStatus`, `LoadedInfo`, `ChatOutcome` |
| `src/diagnostic.rs` | `HiggsError` enum with diagnostic codes HG001–HG011 |
| `src/rpc.rs` | NDJSON JSON-RPC 2.0 encode/decode — the supervisor↔worker wire protocol |
| `src/supervisor.rs` | Worker process supervisor: spawn, restart, request correlation, chat-chunk routing |
| `src/serve/mod.rs` | Axum router: `/v1/models`, `/v1/chat/completions`, `/api/higgs/*` control routes |
| `src/serve/stream.rs` | SSE assembly for streaming chat completions (OpenAI chunk protocol) |
| `src/worker/mod.rs` | Worker entry point — JSON-RPC dispatch loop, `worker_main()` |
| `src/worker/models.rs` | Model discovery across LM Studio, HuggingFace cache, and Ollama stores; `HiggsModel`, `ModelStore` |
| `src/worker/engine/mod.rs` | `HiggsEngine` trait: `load`, `unload`, `is_loaded`, `chat` |
| `src/worker/engine/llamacpp.rs` | llama.cpp engine impl: GGUF-embedded chat templates, context fit-check, token streaming |
