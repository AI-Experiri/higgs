# higgs Configuration

## Table of Contents
- [config.toml — \[higgs\] Section](#configtoml--higgs-section)
- [HiggsConfig Fields](#higgsconfig-fields)
- [Model Directory Layouts](#model-directory-layouts)
- [Build Note: LIBCLANG_PATH](#build-note-libclang_path)
- [Non-Configurable Defaults](#non-configurable-defaults)
- [Links](#links)

---

## config.toml — [higgs] Section

higgs is configured through the `[higgs]` table in `~/.jigglebot/config.toml`.
The server reads it at boot and maps it onto `HiggsConfig` before passing `Arc<Higgs>`
into `AppState`. All fields have serde defaults via `HiggsConfig::default()` so the
table may be omitted entirely.

```toml
[higgs]
# Each field below is optional — the Default impl fills in the right value
# when the key is absent.

# LM Studio model directories (both pre-0.3 and post-0.3 paths by default)
lmstudio_dirs = [
  "~/.lmstudio/models",
  "~/.cache/lm-studio/models",
]

# HuggingFace Hub cache (always ~/.cache/huggingface/hub — NOT dirs::cache_dir())
hf_dirs = ["~/.cache/huggingface/hub"]

# Ollama model store
ollama_dirs = ["~/.ollama/models"]

# Default load parameters (used when no params are supplied to /api/higgs/load)
[higgs.default_load]
ctx_len    = 4096     # context window tokens
gpu_layers = 4294967295  # u32::MAX → all layers on GPU
threads    = 0        # 0 = auto (available_cpus - 2, min 1)
```

---

## HiggsConfig Fields

The host maps its own config table onto `HiggsConfig`. There is no standalone config file — higgs receives its configuration through the Rust API.

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `lmstudio_dirs` | `Vec<PathBuf>` | `[~/.lmstudio/models, ~/.cache/lm-studio/models]` | Both LM Studio < 0.3 and >= 0.3 paths are included by default; the host can narrow the list |
| `hf_dirs` | `Vec<PathBuf>` | `[~/.cache/huggingface/hub]` | HuggingFace hardcodes `~/.cache` on ALL platforms — it does not follow XDG or macOS conventions. higgs resolves this as `dirs::home_dir().join(".cache/huggingface/hub")`, NOT `dirs::cache_dir()` |
| `ollama_dirs` | `Vec<PathBuf>` | `[~/.ollama/models]` | |
| `default_load.ctx_len` | `u32` | `4096` | Context window tokens for new loads when the caller does not supply params |
| `default_load.gpu_layers` | `u32` | `u32::MAX` | `u32::MAX` means all layers offloaded (LM Studio "max" semantics) |
| `default_load.threads` | `u32` | `available_cpus - 2` (min 1) | Worker threads during generation; computed from `std::thread::available_parallelism()` |

---

## Model Directory Layouts

higgs reads the following directory trees during a scan. **It never writes into any of these stores.**

```
~/.lmstudio/models/          (LM Studio < 0.3)
  org/
    model-name/
      *.gguf

~/.cache/lm-studio/models/   (LM Studio >= 0.3)
  org/
    model-name/
      *.gguf

~/.cache/huggingface/hub/    (HuggingFace Hub)
  models--org--model-name/
    snapshots/
      <commit-hash>/
        *.gguf

~/.ollama/models/            (Ollama)
  manifests/registry.ollama.ai/<name>/<tag>   (JSON manifest)
  blobs/sha256-<hash>                          (GGUF blob)
```

All roots are optional — a missing directory is silently skipped. An existing but unreadable root produces `[HG001] ModelDirUnreadable`.

---

## Build Note: LIBCLANG_PATH

The `llama-cpp-2` crate requires `libclang` at compile time (bindgen dependency). On machines where libclang is not on the default path, set:

```sh
export LIBCLANG_PATH=/path/to/libclang/lib
```

The project prefix `env -u LIBCLANG_PATH cargo …` unsets any stale value when the correct path is already on `PATH`. When building for the first time on a new machine, set `LIBCLANG_PATH` explicitly before running `cargo build`.

---

## Non-Configurable Defaults

| Item | Value | Source |
|------|-------|--------|
| stderr ring buffer cap | 2000 lines | `supervisor.rs` — hardcoded |
| event broadcast channel cap | 64 | `supervisor.rs` — hardcoded |
| respawn backoff | 1 second | `supervisor.rs` — hardcoded |
| respawn attempts per death | 1 | supervisor restarts once; factory failure is terminal |
| graceful stop timeout | 2 seconds | `Higgs::stop()` — hardcoded |
| `/api/higgs/logs` default tail | 200 lines | `LogsQuery.n` default |

---

## Links

- Root configuration reference: [CONFIG.md](../../CONFIG.md)
- Full field reference (gateway / metrics / agent): [backend/server/src/config/CONFIG.md](../server/src/config/CONFIG.md)
