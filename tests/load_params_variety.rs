//! In-process integration: load the tiny `stories260K` GGUF with VARIED, valid
//! [`LlamaCppParams`] sets and prove each loads + serves a short chat, then
//! unloads. This drives the param-APPLICATION branches in the engine load path
//! (`src/worker/engine/llamacpp/mod.rs` `load` + `run_decode`) via the LIBRARY
//! facade (`higgs.load(id, Some(LoadParams::llamacpp(LlamaCppParams { … })))`) that
//! the existing inference/autotune/control tests don't exercise — they only ever
//! load with the base fields or a tune suggestion's default-ish optionals.
//!
//! Each test pins a DISTINCT param set chosen so the engine takes a branch it
//! otherwise wouldn't:
//!   - a small `ctx_len` (model-params + context fit math),
//!   - `gpu_layers: 0` (CPU-only offload branch),
//!   - explicit `n_threads` + `n_threads_batch` split,
//!   - `flash_attn` on AND off (the `with_flash_attention_policy` branch, both ways),
//!   - a quantized KV cache (`type_k` = Q8_0; K-only + no FA — the one combo the
//!     toy model supports, see the test doc) — the `with_type_k`/`with_type_v`
//!     branches,
//!   - `n_batch` + `n_ubatch` + `n_seq_max` + `swa_full` (context-param branches),
//!   - `use_mmap`/`use_mlock` (model-param `with_use_mmap`/`with_use_mlock` branches),
//!   - `rope_scaling_type` + `rope_freq_base`/`rope_freq_scale` (RoPE branches).
//!
//! `HiggsStatus::loaded` only surfaces `ctx_len`/`gpu_layers`/`threads`, so where a
//! field is status-visible we assert it; for the rest the proof of application is
//! that `load` returns `Ok` AND a chat serves (the engine built the model+context
//! with those `with_*` calls without erroring). Each chat is NON-streaming, over the
//! real `/v1` HTTP surface via `serve_v1_local` — no SSE stream is ever left open
//! (CLAUDE.md). Skips when the tiny GGUF is absent.
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

use common::{higgs_local, serve_v1_local, TINY_MODEL_ID};
use higgs::worker::engine::llamacpp::params::{LlamaCppParams, RopeScalingType};
use higgs::worker::engine::{CtxLen, FlashAttn, GpuLayers, KvCacheKind};
use higgs::LoadParams;
use serde_json::{json, Value};

/// The three base fields (`ctx_len`/`gpu_layers`/`threads`) from the node's live
/// `default_load` — the fallback the deleted flat `POST /models/load` used for any
/// base field a request left unpinned. Tests that pin only an *optional* override
/// (e.g. `ctx_len` or `flash_attn`) ride these so they mirror the original flat-load
/// behavior (only the pinned field differs from the node default).
fn base_defaults(higgs: &higgs::Higgs) -> LlamaCppParams {
    higgs.server_config().default_load.as_llamacpp().clone()
}

/// Run ONE short NON-STREAMING chat against the resident tiny model over the real
/// `/v1` surface and return the parsed response. Drains the body (no SSE stream left
/// open). `max_tokens` is kept tiny so the decode loop runs but the test stays fast.
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

/// Unload everything (drain-all), then assert nothing is resident (typed status).
async fn unload_all(higgs: &higgs::Higgs) {
    higgs.unload().await.expect("unload all succeeds");
    let st = higgs.status().await.expect("status");
    assert!(
        st.loaded.is_none() && st.loaded_all.is_empty(),
        "no model resident after unload: {st:?}"
    );
}

/// Small `ctx_len`: drives the `with_n_ctx` model/context sizing and the fit-check
/// math with a non-default, small window. (Distinct from autotune which loads with
/// the model's trained/capped ctx.)
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn load_small_ctx_len() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP load_small_ctx_len: no tiny GGUF (set HIGGS_TEST_GGUF)");
        return;
    };
    let (base, guard) = serve_v1_local(higgs.handle()).await;
    let c = reqwest::Client::new();

    let params = LoadParams::llamacpp(LlamaCppParams {
        ctx_len: CtxLen::Fixed { n: 256 },
        ..base_defaults(&higgs)
    });
    higgs
        .load(TINY_MODEL_ID, Some(params))
        .await
        .expect("small ctx_len load succeeds");

    let snap = higgs.status().await.expect("status");
    assert_eq!(
        snap.loaded.as_ref().and_then(|l| l.ctx_len),
        Some(CtxLen::Fixed { n: 256 }),
        "the small ctx_len is the loaded window: {snap:?}"
    );

    assert_served(&short_chat(&c, &base).await);
    unload_all(&higgs).await;

    guard.shutdown().await;
    higgs.shutdown().await;
}

/// `gpu_layers: 0` — the CPU-only offload branch (`with_n_gpu_layers(0)`): no layers
/// go to the accelerator. On a Metal host this forces the all-CPU model path,
/// distinct from the default `GpuLayers::All` ("all on GPU") every other test takes.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn load_cpu_only_zero_gpu_layers() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP load_cpu_only_zero_gpu_layers: no tiny GGUF (set HIGGS_TEST_GGUF)");
        return;
    };
    let (base, guard) = serve_v1_local(higgs.handle()).await;
    let c = reqwest::Client::new();

    let params = LoadParams::llamacpp(LlamaCppParams {
        ctx_len: CtxLen::Fixed { n: 512 },
        gpu_layers: GpuLayers::Count { n: 0 },
        ..base_defaults(&higgs)
    });
    higgs
        .load(TINY_MODEL_ID, Some(params))
        .await
        .expect("CPU-only (gpu_layers=0) load succeeds");

    let snap = higgs.status().await.expect("status");
    assert_eq!(
        snap.loaded.as_ref().and_then(|l| l.gpu_layers),
        Some(GpuLayers::Count { n: 0 }),
        "gpu_layers=0 is the loaded offload: {snap:?}"
    );

    assert_served(&short_chat(&c, &base).await);
    unload_all(&higgs).await;

    guard.shutdown().await;
    higgs.shutdown().await;
}

/// Explicit `threads` + a SPLIT `n_threads_batch`: drives `with_n_threads` AND the
/// `n_threads_batch != threads` branch in `run_decode` (the prompt/batch thread pool
/// is sized separately from generation).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn load_explicit_threads_and_batch_threads() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!(
            "SKIP load_explicit_threads_and_batch_threads: no tiny GGUF (set HIGGS_TEST_GGUF)"
        );
        return;
    };
    let (base, guard) = serve_v1_local(higgs.handle()).await;
    let c = reqwest::Client::new();

    let params = LoadParams::llamacpp(LlamaCppParams {
        ctx_len: CtxLen::Fixed { n: 512 },
        gpu_layers: GpuLayers::Count { n: 0 },
        threads: 2,
        n_threads_batch: Some(1),
        ..Default::default()
    });
    higgs
        .load(TINY_MODEL_ID, Some(params))
        .await
        .expect("explicit threads + n_threads_batch load succeeds");

    let snap = higgs.status().await.expect("status");
    assert_eq!(
        snap.loaded.as_ref().and_then(|l| l.threads),
        Some(2),
        "explicit threads is the loaded value: {snap:?}"
    );

    assert_served(&short_chat(&c, &base).await);
    unload_all(&higgs).await;

    guard.shutdown().await;
    higgs.shutdown().await;
}

/// `flash_attn` ON then OFF: both `with_flash_attention_policy` branches (ENABLED=1
/// and DISABLED=0). Two loads in one test — a reload over the first replaces the
/// resident model — to exercise the policy in both directions through one instance.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn load_flash_attn_on_then_off() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP load_flash_attn_on_then_off: no tiny GGUF (set HIGGS_TEST_GGUF)");
        return;
    };
    let (base, guard) = serve_v1_local(higgs.handle()).await;
    let c = reqwest::Client::new();

    // flash_attn ON.
    let params = LoadParams::llamacpp(LlamaCppParams {
        ctx_len: CtxLen::Fixed { n: 512 },
        flash_attn: Some(FlashAttn::On),
        ..base_defaults(&higgs)
    });
    higgs
        .load(TINY_MODEL_ID, Some(params))
        .await
        .expect("flash_attn=on load succeeds");
    assert_served(&short_chat(&c, &base).await);
    unload_all(&higgs).await;

    // flash_attn OFF — the other policy branch.
    let params = LoadParams::llamacpp(LlamaCppParams {
        ctx_len: CtxLen::Fixed { n: 512 },
        flash_attn: Some(FlashAttn::Off),
        ..base_defaults(&higgs)
    });
    higgs
        .load(TINY_MODEL_ID, Some(params))
        .await
        .expect("flash_attn=off load succeeds");
    assert_served(&short_chat(&c, &base).await);
    unload_all(&higgs).await;

    guard.shutdown().await;
    higgs.shutdown().await;
}

/// A quantized KV cache: `type_k` = `Q8_0` (with `type_v` = default F16 and
/// `flash_attn: off`). Drives the `with_type_k`/`with_type_v` +
/// `kv_cache_to_llama` branches with a NON-default (quantized) cache type —
/// the engine default is F16, which every other test uses.
///
/// K-only, no FA, because of the llama.cpp shipped with llama-cpp-2 ≥ 0.1.150:
/// a FORCED `flash_attn: on` now fails context creation when no FA kernel
/// exists for the model's head size (the toy `stories260K` has none — the old
/// engine silently fell back), and a quantized V cache requires FA — so
/// K-quant-without-FA is the only quantized-KV combination the toy can serve.
/// Real target models (head size 128) take Q8_0 K+V with FA fine.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn load_quantized_kv_cache() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP load_quantized_kv_cache: no tiny GGUF (set HIGGS_TEST_GGUF)");
        return;
    };
    let (base, guard) = serve_v1_local(higgs.handle()).await;
    let c = reqwest::Client::new();

    let params = LoadParams::llamacpp(LlamaCppParams {
        ctx_len: CtxLen::Fixed { n: 512 },
        flash_attn: Some(FlashAttn::Off),
        type_k: Some(KvCacheKind::Q8_0),
        type_v: Some(KvCacheKind::F16),
        ..base_defaults(&higgs)
    });
    higgs
        .load(TINY_MODEL_ID, Some(params))
        .await
        .expect("quantized KV cache (Q8_0) load succeeds: nothing rejected");

    assert_served(&short_chat(&c, &base).await);
    unload_all(&higgs).await;

    guard.shutdown().await;
    higgs.shutdown().await;
}

/// Batch sizing + sequence slots: `n_batch` + `n_ubatch` + `n_seq_max` + `swa_full`.
/// Drives `with_n_batch`/`with_n_ubatch`/`with_n_seq_max`/`with_swa_full` — the
/// context-param branches in `run_decode` left at their engine default by the
/// base-only loads elsewhere.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn load_batch_and_seq_params() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP load_batch_and_seq_params: no tiny GGUF (set HIGGS_TEST_GGUF)");
        return;
    };
    let (base, guard) = serve_v1_local(higgs.handle()).await;
    let c = reqwest::Client::new();

    let params = LoadParams::llamacpp(LlamaCppParams {
        ctx_len: CtxLen::Fixed { n: 512 },
        gpu_layers: GpuLayers::Count { n: 0 },
        threads: 2,
        n_batch: Some(64),
        n_ubatch: Some(32),
        n_seq_max: Some(1),
        swa_full: Some(false),
        ..Default::default()
    });
    higgs
        .load(TINY_MODEL_ID, Some(params))
        .await
        .expect("n_batch/n_ubatch/n_seq_max/swa_full load succeeds");

    let snap = higgs.status().await.expect("status");
    assert_eq!(
        snap.loaded.as_ref().and_then(|l| l.ctx_len),
        Some(CtxLen::Fixed { n: 512 }),
        "ctx_len from the params applied: {snap:?}"
    );

    assert_served(&short_chat(&c, &base).await);
    unload_all(&higgs).await;

    guard.shutdown().await;
    higgs.shutdown().await;
}

/// Memory-mapping toggles: `use_mmap: false` + `use_mlock: false` — the
/// `with_use_mmap`/`with_use_mlock` model-param branches. Default leaves both
/// `None` (engine default), so an explicit `false` exercises the setter on each.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn load_mmap_and_mlock_toggles() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP load_mmap_and_mlock_toggles: no tiny GGUF (set HIGGS_TEST_GGUF)");
        return;
    };
    let (base, guard) = serve_v1_local(higgs.handle()).await;
    let c = reqwest::Client::new();

    let params = LoadParams::llamacpp(LlamaCppParams {
        ctx_len: CtxLen::Fixed { n: 512 },
        use_mmap: Some(false),
        use_mlock: Some(false),
        ..base_defaults(&higgs)
    });
    higgs
        .load(TINY_MODEL_ID, Some(params))
        .await
        .expect("use_mmap/use_mlock toggled load succeeds");

    assert_served(&short_chat(&c, &base).await);
    unload_all(&higgs).await;

    guard.shutdown().await;
    higgs.shutdown().await;
}

/// RoPE overrides: `rope_scaling_type` + `rope_freq_base` + `rope_freq_scale`. Drives
/// `with_rope_scaling_type`/`rope_scaling_to_llama` and `with_rope_freq_base`/
/// `with_rope_freq_scale` — the RoPE context branches left unset by base-only loads.
/// `Linear` scaling with neutral freq values keeps the toy model coherent enough to
/// generate.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn load_rope_overrides() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP load_rope_overrides: no tiny GGUF (set HIGGS_TEST_GGUF)");
        return;
    };
    let (base, guard) = serve_v1_local(higgs.handle()).await;
    let c = reqwest::Client::new();

    let params = LoadParams::llamacpp(LlamaCppParams {
        ctx_len: CtxLen::Fixed { n: 512 },
        gpu_layers: GpuLayers::Count { n: 0 },
        threads: 2,
        rope_scaling_type: Some(RopeScalingType::Linear),
        rope_freq_base: Some(10000.0),
        rope_freq_scale: Some(1.0),
        ..Default::default()
    });
    higgs
        .load(TINY_MODEL_ID, Some(params))
        .await
        .expect("RoPE-override load succeeds");

    assert_served(&short_chat(&c, &base).await);
    unload_all(&higgs).await;

    guard.shutdown().await;
    higgs.shutdown().await;
}

/// `offload_kqv` + a load-pinned `seed`: `with_offload_kqv` (KV/KQV op offload branch)
/// and the load-pinned sampler seed used by `run_decode` when no per-request seed is
/// set. A determinism check — two greedy-ish chats with the pinned seed both serve
/// well-formed completions.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn load_offload_kqv_and_seed() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP load_offload_kqv_and_seed: no tiny GGUF (set HIGGS_TEST_GGUF)");
        return;
    };
    let (base, guard) = serve_v1_local(higgs.handle()).await;
    let c = reqwest::Client::new();

    let params = LoadParams::llamacpp(LlamaCppParams {
        ctx_len: CtxLen::Fixed { n: 512 },
        offload_kqv: Some(true),
        seed: Some(1234),
        ..base_defaults(&higgs)
    });
    higgs
        .load(TINY_MODEL_ID, Some(params))
        .await
        .expect("offload_kqv + pinned seed load succeeds");

    // Two chats — the load-pinned seed feeds the sampler when the request omits one.
    assert_served(&short_chat(&c, &base).await);
    assert_served(&short_chat(&c, &base).await);
    unload_all(&higgs).await;

    guard.shutdown().await;
    higgs.shutdown().await;
}
