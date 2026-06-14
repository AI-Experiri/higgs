---
title: How to Use
description: Scan, load, and chat walkthrough — via the Higgs API and via HTTP. HIGGS_TEST_GGUF smoke test. Troubleshooting table.
---

## Scan → Load → Chat walkthrough

### Via the Higgs Rust API

```rust
use higgs::{Higgs, HiggsConfig};
use std::sync::Arc;

// 1. Configure
let config = HiggsConfig::default(); // uses ~/.lmstudio, ~/.cache/huggingface/hub, ~/.ollama

// 2. Construct + start (spawns worker, runs initial scan)
let h = Arc::new(Higgs::new(config));
h.start().await?;

// 3. Scan to see what's available
let models = h.scan().await?;
for m in &models {
    println!("{} ({}) — {} bytes", m.id, m.source, m.size_bytes);
}

// 4. Load a model
h.load("org/model-name", None).await?;

// 5. Chat — streaming
let (mut rx, outcome) = h.chat_stream(
    vec![
        ("system".into(), "You are a helpful assistant.".into()),
        ("user".into(), "What is 2 + 2?".into()),
    ],
    256,   // max_tokens
    0.7,   // temperature
).await?;

while let Some(delta) = rx.recv().await {
    print!("{delta}");
}
let result = outcome.await??;
println!("\n[{}]", result.finish_reason);

// 6. Unload when done
h.unload().await?;
h.stop().await;
```

### Via HTTP

The HTTP surface is the OpenAI wire protocol — any OpenAI-compatible client works.

**Step 1: scan**

```sh
curl http://localhost:8081/api/higgs/models | jq '.models[].id'
```

**Step 2: load**

```sh
curl -X POST -H "Content-Type: application/json" \
  -d '{"id":"org/model-name"}' \
  http://localhost:8081/api/higgs/models/load
# → {"status":"ok","id":"org/model-name"}
```

**Step 3: chat**

```sh
curl -N -H "Content-Type: application/json" \
  -d '{
    "model": "org/model-name",
    "messages": [{"role":"user","content":"What is 2 + 2?"}],
    "stream": true,
    "max_completion_tokens": 64
  }' \
  http://localhost:8081/v1/chat/completions
```

**Step 4: unload**

```sh
curl -X POST http://localhost:8081/api/higgs/models/unload
```

---

## Tool calling

Send an OpenAI `tools` array; when the model decides to call a tool, the
response carries spec-shaped `tool_calls` with `finish_reason: "tool_calls"`.
Works the same against any loaded model — the tool-call format is read from the
model's own GGUF chat template, not configured per request.

```sh
curl -H "Content-Type: application/json" \
  -d '{
    "model": "org/model-name",
    "messages": [{"role":"user","content":"What is the weather in Paris?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
          "type": "object",
          "properties": {"city": {"type": "string"}},
          "required": ["city"]
        }
      }
    }]
  }' \
  http://localhost:8081/v1/chat/completions | jq '.choices[0]'
```

```json
{
  "index": 0,
  "message": {
    "role": "assistant",
    "content": "",
    "tool_calls": [
      {
        "id": "call_abc123",
        "type": "function",
        "function": { "name": "get_weather", "arguments": "{\"city\":\"Paris\"}" }
      }
    ]
  },
  "finish_reason": "tool_calls"
}
```

Add `"stream": true` to receive the call as a single SSE delta followed by a
finish chunk with `finish_reason: "tool_calls"` (see the Endpoints reference for
the exact framing). The tool-call envelope never leaks into the content deltas.

---

## HIGGS_TEST_GGUF Smoke Test

The worker and engine have integration tests that run against a real GGUF file
when the environment variable `HIGGS_TEST_GGUF` is set:

```sh
HIGGS_TEST_GGUF=/path/to/model.gguf cargo test -p higgs -- --nocapture
```

Without `HIGGS_TEST_GGUF` set, the llama.cpp tests are skipped automatically.
The supervisor round-trip tests run with in-memory duplex streams and do not need a real model.

To run only the worker stdio round-trip test (no GGUF needed):

```sh
cargo test -p higgs worker_roundtrip -- --nocapture
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `[HG001] model dir unreadable` | Configured scan root exists but cannot be read | Check directory permissions |
| `[HG002] model not found on disk` | Requested id absent from last scan | Run scan again; verify the id matches exactly |
| `[HG003] model not loaded` | Chat issued without a prior load | Call `POST /api/higgs/models/load` first |
| `[HG004] engine failed to load` | llama.cpp rejected the GGUF file | Check `GET /api/higgs/logs` for the llama.cpp error; the file may be corrupt or incompatible |
| `[HG005] context overflow` | prompt tokens + max_gen exceed n_ctx | Reduce `max_completion_tokens` or shorten the prompt; or reload with a larger `ctx_len` |
| `[HG006] worker spawn failed` | The host binary could not be re-exec'd | Check that the binary path is accessible; the worker entry point must be wired in `main()` |
| `[HG007] worker unavailable` | Worker process died mid-request | Check `GET /api/higgs/logs` for the crash reason; supervisor will attempt one restart |
| `[HG008] rpc decode failed` | Malformed NDJSON from the worker | Indicates a worker binary version mismatch or corruption on the stdio pipe |
| `[HG009] worker error on <method>` | Worker returned a JSON-RPC error | The method's handler failed; check logs for the worker-side cause |
| `[HG010] ollama manifest invalid` | Ollama manifest JSON does not resolve to a GGUF blob | Verify the Ollama model was fully pulled (`ollama pull <name>`) |
| `[HG011] generation failed at <stage>` | llama.cpp sampling loop error | Check logs; may indicate an out-of-memory condition or model incompatibility |
| Build fails: `libclang not found` | `LIBCLANG_PATH` not set | `export LIBCLANG_PATH=/path/to/llvm/lib` before building |
| Model loads but generation is slow | Too few GPU layers or threads | Reload with `gpu_layers: 4294967295` (all) and increase `threads` |
| Model-too-big error in logs | VRAM exceeded | Reduce `gpu_layers` to offload fewer layers to GPU |
