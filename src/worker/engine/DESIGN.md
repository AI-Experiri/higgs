# `worker/engine` — design notes

## Why a trait, and why here

higgs must support more than one inference backend over time (llama.cpp today; MLX, vLLM-style
runtimes, or a remote API shim later) **without** rippling change through the rest of the
system. The design answer is a single narrow trait, `HiggsEngine`, placed at the lowest layer
that still hides every backend detail:

- **Below the trait**: the backend crate, its FFI, its model handle, its sampler, its template
  renderer. All of it lives in one submodule (`llamacpp/`) and is named nowhere else.
- **Above the trait**: `Box<dyn HiggsEngine>` only. The worker dispatch loop, the supervisor,
  the node runtime, and the serve layer never learn which engine they're driving.

This is what makes "add an engine" a local change: a new submodule + one registry line.

## The boundary is OpenAI JSON, not rendered prompts

`chat()` takes `messages_json` — the OpenAI `messages` array (with `tools`, `tool_calls`,
`tool` results) serialized verbatim — **not** a pre-rendered prompt string. Reason: chat
templating is engine-specific (llama.cpp ships `common_chat`; another engine has its own
jinja). If higgs rendered the prompt above the trait, it would bake in one engine's template
mechanism and re-introduce the coupling the trait exists to remove. So the trait hands each
engine the raw conversation and lets it render by its own means. higgs invents **no** chat
template and **no** tool-call grammar of its own.

## Selection: a `const` registry, not a plugin system

Engines are compiled in and listed in `REGISTRY` (a `const [EngineEntry]`). We deliberately do
NOT use dynamic loading / `dlopen` plugins:

- Compiled-in engines are type-checked, vendored, and reproducible — no ABI drift, no runtime
  discovery failures, no supply-chain surface.
- The registry is a `const` table, so it's usable from tests and tooling with zero cost.
- Selection is a single string (`HIGGS_ENGINE`); unknown values fall back to the default with
  a warning rather than failing to start. Per-model selection can layer on top later without
  changing the trait.

## Probe vs. load (two gates)

`probe()` exists separately from `load()` because model→engine compatibility is a real
question ("can our llama.cpp build actually load this GGUF architecture?"). `probe()` loads
into a throwaway handle and drops it, taking `&self` so it never disturbs a model already being
served. Its `(false, Some(reason))` carries the engine's verbatim error (e.g.
`"unknown model architecture: 'gemma4'"`) — that exact string is what the UI shows.

## Object safety & threading

`HiggsEngine: Send` and every method is dyn-compatible, so the worker can hold a
`Box<dyn HiggsEngine>` and move it across the (single-threaded) worker's request loop. The
worker process is the isolation boundary; the engine itself need not be `Sync`.

## Invariants

- The trait is the **only** higgs↔backend contract. A backend crate name (`llama_cpp_2`, …)
  must appear in exactly one submodule.
- `chat()` receives OpenAI `messages_json`; rendering happens below the trait.
- `REGISTRY[0]` is the default and the registry is never empty (enforced at build time).
- `probe()` is side-effect-free w.r.t. the resident model (`&self`).
