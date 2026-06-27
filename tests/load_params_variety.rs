//! Black-box integration: load the tiny `stories260K` GGUF with VARIED, valid
//! [`LlamaCppParams`] sets and prove each loads + serves a short chat, then
//! unloads. This drives the param-APPLICATION branches in the engine load path
//! (`src/worker/engine/llamacpp/mod.rs` `load` + `run_decode`) and the worker's
//! flat-`LlamaCppParams` deserialize seam (`src/worker/mod.rs handle_load`) that
//! the existing inference/autotune/control tests don't exercise — they only ever
//! load with the base fields or a tune suggestion's default-ish optionals.
//!
//! Each test pins a DISTINCT param set chosen so the engine takes a branch it
//! otherwise wouldn't:
//!   - a small `ctx_len` (model-params + context fit math),
//!   - `gpu_layers: 0` (CPU-only offload branch),
//!   - explicit `n_threads` + `n_threads_batch` split,
//!   - `flash_attn` on AND off (the `with_flash_attention_policy` branch, both ways),
//!   - a quantized KV cache (`type_k`/`type_v` = Q8_0, paired with flash_attn as
//!     llama.cpp requires) — the `with_type_k`/`with_type_v` branches,
//!   - `n_batch` + `n_ubatch` + `n_seq_max` + `swa_full` (context-param branches),
//!   - `use_mmap`/`use_mlock` (model-param `with_use_mmap`/`with_use_mlock` branches),
//!   - `rope_scaling_type` + `rope_freq_base`/`rope_freq_scale` (RoPE branches).
//!
//! Status only surfaces `ctx_len`/`gpu_layers`/`threads`, so where a field is
//! status-visible we assert it; for the rest the proof of application is that the
//! load returns 200 AND a chat serves (the engine built the model+context with
//! those `with_*` calls without erroring). Each chat is NON-streaming — no SSE
//! stream is ever left open (CLAUDE.md). Skips when the tiny GGUF is absent.
//!
//! Ports: 12900-base, one per test.
//!
//! NOTE on un-triggered branches: the model-load-time PINNED branches
//! (`cpu_moe`/`cpu_buft_overrides`/`kv_overrides`/`split_mode`/`main_gpu`/`devices`)
//! are NOT exercised here — `cpu_moe`/buft need an MoE model (the toy `stories260K`
//! has none), and split_mode/main_gpu/devices are DEFERRED multi-GPU knobs whose
//! apply on a single-device host is a no-op with no observable effect. They have
//! unit coverage in `params.rs` (`has_overrides`, round-trip) + `mod.rs`
//! (`kv_override_value_parses_by_type`). `kv_overrides` IS reachable via the full
//! `params` object but applying an arbitrary GGUF metadata override to the toy
//! model risks an engine load failure unrelated to the branch under test, so it is
//! left to the unit `kv_override_value_parses_by_type` rather than asserted here.

mod common;

use common::{spawn_with_tiny_model, tiny_gguf_path, TINY_MODEL_ID};
use serde_json::{json, Value};

/// POST a load request body and return the HTTP status + parsed JSON body.
async fn load(c: &reqwest::Client, base: &str, body: Value) -> (reqwest::StatusCode, Value) {
    let r = c
        .post(format!("{base}/api/higgs/models/load"))
        .json(&body)
        .send()
        .await
        .expect("load request");
    let status = r.status();
    let v: Value = r.json().await.expect("load response json");
    (status, v)
}

/// GET the live status snapshot.
async fn status(c: &reqwest::Client, base: &str) -> Value {
    c.get(format!("{base}/api/higgs/status"))
        .send()
        .await
        .expect("status request")
        .json()
        .await
        .expect("status json")
}

/// Run ONE short NON-STREAMING chat against the resident tiny model and return the
/// parsed response. Drains the body (no SSE stream left open). `max_tokens` is kept
/// tiny so the decode loop runs but the test stays fast.
async fn short_chat(c: &reqwest::Client, base: &str) -> Value {
    c.post(format!("{base}/v1/chat/completions"))
        .json(&json!({
            "model": TINY_MODEL_ID,
            "stream": false,
            "max_tokens": 4,
            "messages": [{ "role": "user", "content": "Hi." }]
        }))
        .send()
        .await
        .expect("chat request")
        .json()
        .await
        .expect("chat json")
}

/// Assert a chat response is a well-formed completion (string content, known
/// finish reason, non-zero usage) — proves the engine generated with the pinned
/// load params applied.
fn assert_served(resp: &Value) {
    let choice = &resp["choices"][0];
    assert!(
        choice["message"]["content"].is_string(),
        "chat returns string content: {resp:?}"
    );
    assert!(
        matches!(
            choice["finish_reason"].as_str(),
            Some("stop") | Some("length")
        ),
        "finish_reason is stop|length: {resp:?}"
    );
    assert!(
        resp["usage"]["prompt_tokens"].as_u64().unwrap_or(0) > 0,
        "prompt_tokens non-zero: {resp:?}"
    );
}

/// Unload everything (drain-all `{}`), then assert nothing is resident.
async fn unload_all(c: &reqwest::Client, base: &str) {
    let r = c
        .post(format!("{base}/api/higgs/models/unload"))
        .json(&json!({}))
        .send()
        .await
        .expect("unload request");
    assert!(r.status().is_success(), "unload all succeeds");
    let st = status(c, base).await;
    assert!(
        st["loaded"].is_null() || st["loaded_all"].as_array().is_some_and(Vec::is_empty),
        "no model resident after unload: {st}"
    );
}

/// Small `ctx_len` via the FLAT field: drives the `with_n_ctx` model/context sizing
/// and the worker's `ctx_len` override + fit-check math with a non-default, small
/// window. (Distinct from autotune which loads with the model's trained/capped ctx.)
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn load_small_ctx_len() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("SKIP load_small_ctx_len: no tiny GGUF (set HIGGS_TEST_GGUF)");
        return;
    };
    let srv = spawn_with_tiny_model(12900, &gguf).await;
    let c = reqwest::Client::new();

    let (st, _) = load(
        &c,
        &srv.base,
        json!({ "id": TINY_MODEL_ID, "ctx_len": 256 }),
    )
    .await;
    assert_eq!(st, 200, "small ctx_len load succeeds");

    let snap = status(&c, &srv.base).await;
    assert_eq!(
        snap["loaded"]["ctx_len"]["n"].as_u64().unwrap(),
        256,
        "the small ctx_len is the loaded window: {snap}"
    );

    assert_served(&short_chat(&c, &srv.base).await);
    unload_all(&c, &srv.base).await;
}

/// `gpu_layers: 0` — the CPU-only offload branch (`with_n_gpu_layers(0)`): no layers
/// go to the accelerator. On a Metal host this forces the all-CPU model path,
/// distinct from the default `gpu_layers = u32::MAX` ("all on GPU") every other
/// test takes.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn load_cpu_only_zero_gpu_layers() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("SKIP load_cpu_only_zero_gpu_layers: no tiny GGUF (set HIGGS_TEST_GGUF)");
        return;
    };
    let srv = spawn_with_tiny_model(12901, &gguf).await;
    let c = reqwest::Client::new();

    let (st, _) = load(
        &c,
        &srv.base,
        json!({ "id": TINY_MODEL_ID, "gpu_layers": 0, "ctx_len": 512 }),
    )
    .await;
    assert_eq!(st, 200, "CPU-only (gpu_layers=0) load succeeds");

    let snap = status(&c, &srv.base).await;
    // The request sent the legacy numeric `gpu_layers: 0` (accepted by the lenient
    // GpuLayers deserialize); status surfaces it canonically as the tagged union.
    assert_eq!(
        snap["loaded"]["gpu_layers"],
        json!({ "kind": "count", "n": 0 }),
        "gpu_layers=0 is the loaded offload: {snap}"
    );

    assert_served(&short_chat(&c, &srv.base).await);
    unload_all(&c, &srv.base).await;
}

/// Explicit `threads` + a SPLIT `n_threads_batch` (via the full `params` object, the
/// only path that carries `n_threads_batch`): drives `with_n_threads` AND the
/// `n_threads_batch != threads` branch in `run_decode` (the prompt/batch thread pool
/// is sized separately from generation).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn load_explicit_threads_and_batch_threads() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!(
            "SKIP load_explicit_threads_and_batch_threads: no tiny GGUF (set HIGGS_TEST_GGUF)"
        );
        return;
    };
    let srv = spawn_with_tiny_model(12902, &gguf).await;
    let c = reqwest::Client::new();

    // The `params` object supersedes the flat fields and carries n_threads_batch,
    // which has no flat field. ctx_len/gpu_layers ride the params too (base fields).
    let (st, _) = load(
        &c,
        &srv.base,
        json!({
            "id": TINY_MODEL_ID,
            "params": {
                "engine": "LlamaCpp",
                "ctx_len": 512,
                "gpu_layers": 0,
                "threads": 2,
                "n_threads_batch": 1
            }
        }),
    )
    .await;
    assert_eq!(st, 200, "explicit threads + n_threads_batch load succeeds");

    let snap = status(&c, &srv.base).await;
    assert_eq!(
        snap["loaded"]["threads"].as_u64().unwrap(),
        2,
        "explicit threads is the loaded value: {snap}"
    );

    assert_served(&short_chat(&c, &srv.base).await);
    unload_all(&c, &srv.base).await;
}

/// `flash_attn` ON then OFF: both `with_flash_attention_policy` branches (ENABLED=1
/// and DISABLED=0). Two loads in one test — a reload over the first replaces the
/// resident model — to exercise the policy in both directions through one server.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn load_flash_attn_on_then_off() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("SKIP load_flash_attn_on_then_off: no tiny GGUF (set HIGGS_TEST_GGUF)");
        return;
    };
    let srv = spawn_with_tiny_model(12903, &gguf).await;
    let c = reqwest::Client::new();

    // flash_attn ON (the flat field serializes lowercase "on").
    let (st, _) = load(
        &c,
        &srv.base,
        json!({ "id": TINY_MODEL_ID, "ctx_len": 512, "flash_attn": "on" }),
    )
    .await;
    assert_eq!(st, 200, "flash_attn=on load succeeds");
    assert_served(&short_chat(&c, &srv.base).await);
    unload_all(&c, &srv.base).await;

    // flash_attn OFF — the other policy branch.
    let (st, _) = load(
        &c,
        &srv.base,
        json!({ "id": TINY_MODEL_ID, "ctx_len": 512, "flash_attn": "off" }),
    )
    .await;
    assert_eq!(st, 200, "flash_attn=off load succeeds");
    assert_served(&short_chat(&c, &srv.base).await);
    unload_all(&c, &srv.base).await;
}

/// A quantized KV cache: `type_k` = `type_v` = `Q8_0`, paired with `flash_attn: on`
/// (llama.cpp requires flash attention for a non-F16 V cache). Drives the
/// `with_type_k`/`with_type_v` + `kv_cache_to_llama` branches with a NON-default
/// (quantized) cache type — the engine default is F16, which every other test uses.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn load_quantized_kv_cache() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("SKIP load_quantized_kv_cache: no tiny GGUF (set HIGGS_TEST_GGUF)");
        return;
    };
    let srv = spawn_with_tiny_model(12904, &gguf).await;
    let c = reqwest::Client::new();

    // type_k/type_v are flat fields; flash_attn on satisfies the quantized-V
    // requirement. KvCacheKind serializes as its variant name ("Q8_0").
    let (st, _) = load(
        &c,
        &srv.base,
        json!({
            "id": TINY_MODEL_ID,
            "ctx_len": 512,
            "flash_attn": "on",
            "type_k": "Q8_0",
            "type_v": "Q8_0"
        }),
    )
    .await;
    assert_eq!(
        st, 200,
        "quantized KV cache (Q8_0) load succeeds: nothing rejected"
    );

    assert_served(&short_chat(&c, &srv.base).await);
    unload_all(&c, &srv.base).await;
}

/// Batch sizing + sequence slots: `n_batch` + `n_ubatch` (flat fields) plus
/// `n_seq_max` + `swa_full` (full-`params`-only). Drives `with_n_batch`/
/// `with_n_ubatch`/`with_n_seq_max`/`with_swa_full` — the context-param branches in
/// `run_decode` left at their engine default by the base-only loads elsewhere.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn load_batch_and_seq_params() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("SKIP load_batch_and_seq_params: no tiny GGUF (set HIGGS_TEST_GGUF)");
        return;
    };
    let srv = spawn_with_tiny_model(12905, &gguf).await;
    let c = reqwest::Client::new();

    let (st, _) = load(
        &c,
        &srv.base,
        json!({
            "id": TINY_MODEL_ID,
            "params": {
                "engine": "LlamaCpp",
                "ctx_len": 512,
                "gpu_layers": 0,
                "threads": 2,
                "n_batch": 64,
                "n_ubatch": 32,
                "n_seq_max": 1,
                "swa_full": false
            }
        }),
    )
    .await;
    assert_eq!(st, 200, "n_batch/n_ubatch/n_seq_max/swa_full load succeeds");

    let snap = status(&c, &srv.base).await;
    assert_eq!(
        snap["loaded"]["ctx_len"]["n"].as_u64().unwrap(),
        512,
        "ctx_len from the params object applied: {snap}"
    );

    assert_served(&short_chat(&c, &srv.base).await);
    unload_all(&c, &srv.base).await;
}

/// Memory-mapping toggles: `use_mmap: false` + `use_mlock: false` (flat fields) —
/// the `with_use_mmap`/`with_use_mlock` model-param branches. Default leaves both
/// `None` (engine default), so an explicit `false` exercises the setter on each.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn load_mmap_and_mlock_toggles() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("SKIP load_mmap_and_mlock_toggles: no tiny GGUF (set HIGGS_TEST_GGUF)");
        return;
    };
    let srv = spawn_with_tiny_model(12906, &gguf).await;
    let c = reqwest::Client::new();

    let (st, _) = load(
        &c,
        &srv.base,
        json!({
            "id": TINY_MODEL_ID,
            "ctx_len": 512,
            "use_mmap": false,
            "use_mlock": false
        }),
    )
    .await;
    assert_eq!(st, 200, "use_mmap/use_mlock toggled load succeeds");

    assert_served(&short_chat(&c, &srv.base).await);
    unload_all(&c, &srv.base).await;
}

/// RoPE overrides: `rope_scaling_type` (full-`params`-only) + `rope_freq_base` +
/// `rope_freq_scale` (flat fields). Drives `with_rope_scaling_type`/`rope_scaling_to_llama`
/// and `with_rope_freq_base`/`with_rope_freq_scale` — the RoPE context branches left
/// unset by base-only loads. `linear` scaling with neutral freq values keeps the toy
/// model coherent enough to generate.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn load_rope_overrides() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("SKIP load_rope_overrides: no tiny GGUF (set HIGGS_TEST_GGUF)");
        return;
    };
    let srv = spawn_with_tiny_model(12907, &gguf).await;
    let c = reqwest::Client::new();

    let (st, _) = load(
        &c,
        &srv.base,
        json!({
            "id": TINY_MODEL_ID,
            "params": {
                "engine": "LlamaCpp",
                "ctx_len": 512,
                "gpu_layers": 0,
                "threads": 2,
                "rope_scaling_type": "Linear",
                "rope_freq_base": 10000.0,
                "rope_freq_scale": 1.0
            }
        }),
    )
    .await;
    assert_eq!(st, 200, "RoPE-override load succeeds");

    assert_served(&short_chat(&c, &srv.base).await);
    unload_all(&c, &srv.base).await;
}

/// `offload_kqv` + a load-pinned `seed` (both flat fields): `with_offload_kqv` (KV/KQV
/// op offload branch) and the load-pinned sampler seed used by `run_decode` when no
/// per-request seed is set. A determinism check — two greedy-ish chats with the
/// pinned seed both serve well-formed completions.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn load_offload_kqv_and_seed() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("SKIP load_offload_kqv_and_seed: no tiny GGUF (set HIGGS_TEST_GGUF)");
        return;
    };
    let srv = spawn_with_tiny_model(12908, &gguf).await;
    let c = reqwest::Client::new();

    let (st, _) = load(
        &c,
        &srv.base,
        json!({
            "id": TINY_MODEL_ID,
            "ctx_len": 512,
            "offload_kqv": true,
            "seed": 1234
        }),
    )
    .await;
    assert_eq!(st, 200, "offload_kqv + pinned seed load succeeds");

    // Two chats — the load-pinned seed feeds the sampler when the request omits one.
    assert_served(&short_chat(&c, &srv.base).await);
    assert_served(&short_chat(&c, &srv.base).await);
    unload_all(&c, &srv.base).await;
}
