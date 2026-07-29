//! Autotune end-to-end: the in-process `Higgs::tune` (Suggest) → `load` with the
//! suggestion → serve over the real `/v1` → a plain reload reuses the saved profile.
//!
//! Drives higgs IN-PROCESS (library-first) against the tiny `stories260K` GGUF via
//! the `higgs_local` harness (a REAL local llama.cpp worker). Skips when no tiny
//! GGUF is present. The only chat is non-streaming over `serve_v1_local` (no SSE
//! stream left open, per CLAUDE.md).

mod common;

use common::{higgs_local, serve_v1_local, TINY_MODEL_ID};

use higgs::system::DeviceKind;
use higgs::tune::{FitVerdict, ResourceBudget, TuneMode};
use higgs::worker::engine::{FlashAttn, GpuLayers, LoadParams};
use higgs::TuneRequest;

/// `tune` (Suggest) returns suggested params honoring the CPU-thread cap, a fit
/// verdict for VRAM + RAM, and a non-empty rationale.
#[tokio::test]
async fn tune_suggest_returns_params_fit_and_rationale() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP tune_suggest: no tiny GGUF (set HIGGS_TEST_GGUF)");
        return;
    };

    let s = higgs
        .tune(TuneRequest {
            id: TINY_MODEL_ID.to_string(),
            mode: Some(TuneMode::Suggest),
            budget: Some(ResourceBudget {
                max_cpu_threads: Some(2),
                ..Default::default()
            }),
            pins: None,
        })
        .await
        .expect("tune request");

    // The umbrella is the LlamaCpp variant.
    let llama = s.load.as_llamacpp();
    // The CPU-thread cap is honored (threads = min(floor(cores/2), 2) ≤ 2).
    assert!(
        s.load.threads() <= 2,
        "cpu thread cap honored: {}",
        s.load.threads()
    );
    // A concrete (fixed, non-zero) ctx_len.
    assert!(
        s.load.ctx_len().fixed_n().is_some_and(|n| n > 0),
        "a concrete ctx_len: {:?}",
        s.load.ctx_len()
    );
    // Derived defaults: flash attention on.
    assert_eq!(llama.flash_attn, Some(FlashAttn::On));
    // Both fit verdicts present (a valid tri-state variant) and a rationale.
    assert!(
        matches!(
            s.vram_fit.verdict,
            FitVerdict::Fits | FitVerdict::Tight | FitVerdict::Overflow
        ),
        "vram verdict present"
    );
    assert!(
        matches!(
            s.ram_fit.verdict,
            FitVerdict::Fits | FitVerdict::Tight | FitVerdict::Overflow
        ),
        "ram verdict present"
    );
    assert!(
        !s.rationale.is_empty(),
        "rationale explains the choices: {:?}",
        s.rationale
    );
    // The context is DERIVED from the budget (not a flat cap), and the rationale
    // names the method. This assertion holds only once derive_ctx is wired — a
    // flat-8192 derive never pushes it (fail-on-revert).
    assert!(
        s.rationale
            .iter()
            .any(|t| t.contains("context") && t.contains("analytical")),
        "rationale explains the budget-derived context (analytical): {:?}",
        s.rationale
    );
    assert!(
        s.load.ctx_len().fixed_n().is_some_and(|n| n > 0),
        "a concrete derived context: {:?}",
        s.load.ctx_len()
    );

    higgs.shutdown().await;
}

/// An explicit VRAM budget that the all-GPU load overflows makes the suggester
/// back off GPU offload to fit the cap (the `gpu_layers_within_budget` path, dead
/// until a budget is actually supplied). A 1 MiB cap can't hold the model+overhead,
/// so the suggestion drops to CPU-only (`gpu_layers` count 0) with a rationale note.
#[tokio::test]
async fn tune_vram_budget_backs_off_gpu_offload() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP tune_vram_budget: no tiny GGUF (set HIGGS_TEST_GGUF)");
        return;
    };

    let s = higgs
        .tune(TuneRequest {
            id: TINY_MODEL_ID.to_string(),
            mode: Some(TuneMode::Suggest),
            // 1 MiB — far below the ~800 MiB compute overhead, so an all-GPU load
            // overflows it and the suggester must reduce GPU offload to fit.
            budget: Some(ResourceBudget {
                max_vram_bytes: Some(1u64 << 20),
                ..Default::default()
            }),
            pins: None,
        })
        .await
        .expect("tune request");

    // CPU-only is `Count { n: 0 }`; the all-GPU default `All` would mean the budget
    // was ignored. This holds on a GPU host (backed off to fit the 1 MiB cap) AND on
    // a CPU-only host (CPU-only from the start) — env-robust.
    assert_eq!(
        s.load.gpu_layers(),
        GpuLayers::Count { n: 0 },
        "1 MiB VRAM budget → CPU-only offload: {:?}",
        s.load.gpu_layers()
    );

    // The "VRAM budget" rationale only appears when there IS a GPU to back off from.
    // On a CPU-only host the heuristic derives CPU-only without ever entering the
    // budget-backoff path, so gate that assertion on detected GPU presence (else the
    // test would falsely fail on CPU-only CI/dev hosts).
    let hw = higgs.hardware().await;
    let has_gpu = hw.gpus.iter().any(|g| g.kind == DeviceKind::Gpu);
    if has_gpu {
        assert!(
            s.rationale
                .iter()
                .any(|t| t.to_lowercase().contains("vram budget")),
            "rationale notes the VRAM-budget backoff: {:?}",
            s.rationale
        );
    }

    higgs.shutdown().await;
}

/// Tune → load WITH the suggestion → non-streaming chat serves; then a PLAIN
/// reload (no params) reuses the saved profile (the persisted `last_load` carries
/// the tuned optionals, distinguishing it from the bare `default_load`).
#[tokio::test]
async fn tune_then_load_and_plain_load_reuses_profile() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP tune_then_load: no tiny GGUF (set HIGGS_TEST_GGUF)");
        return;
    };

    // 1. Tune (Suggest) — persists the profile to the node's models.json.
    let s = higgs
        .tune(TuneRequest {
            id: TINY_MODEL_ID.to_string(),
            mode: None,
            budget: None,
            pins: None,
        })
        .await
        .unwrap();
    let suggested_load = s.load.clone();
    let tuned_ctx = suggested_load.ctx_len();

    // 2. Load WITH the accepted suggestion.
    higgs
        .load(TINY_MODEL_ID, Some(suggested_load))
        .await
        .expect("load with tuned params");

    // 3. The resident model loaded with the tuned context window.
    let st = higgs.status().await.unwrap();
    let loaded = st.loaded.expect("a resident model");
    assert_eq!(
        loaded.ctx_len,
        Some(tuned_ctx),
        "loaded with the tuned ctx_len: {loaded:?}"
    );

    // 4. It serves a (non-streaming) completion with the tuned params applied — via
    //    the REAL /v1 HTTP surface. Keep the serve guard alive across steps 5-6 (its
    //    shutdown drains the worker) and tear it down last.
    let (base, guard) = serve_v1_local(higgs.handle()).await;
    let chat = reqwest::Client::new()
        .post(format!("{base}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": TINY_MODEL_ID,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 4,
            "stream": false
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(chat.status(), 200, "served with tuned params");
    let _: serde_json::Value = chat.json().await.unwrap();

    // 5. Unload, then a PLAIN reload (no params) reuses the saved tuning profile.
    higgs.unload().await.unwrap();
    higgs.load(TINY_MODEL_ID, None).await.expect("plain reload");

    // 6. The persisted `last_load` carries the tuned optionals (flash_attn on) —
    //    proving the saved profile was reused, not the bare default_load (which
    //    leaves flash_attn unset).
    let entry = higgs.model_by_id(TINY_MODEL_ID).await.unwrap();
    let last = entry.last_load.expect("last_load persisted");
    assert_eq!(
        last.as_llamacpp().flash_attn,
        Some(FlashAttn::On),
        "plain load reused the saved tuned profile: {last:?}"
    );

    guard.shutdown().await;
    higgs.shutdown().await;
}

/// An ACCEPTED edit to the suggested params must survive an unload/reload — a
/// plain reload reuses the last accepted load, not the stale tune suggestion. (The
/// successful explicit load syncs the saved profile in `models.json`.)
#[tokio::test]
async fn edited_load_params_survive_reload() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP edited_params: no tiny GGUF (set HIGGS_TEST_GGUF)");
        return;
    };

    // Tune, then EDIT the suggestion (threads = 1 is distinctive + always valid).
    let s = higgs
        .tune(TuneRequest {
            id: TINY_MODEL_ID.to_string(),
            mode: None,
            budget: None,
            pins: None,
        })
        .await
        .unwrap();
    let edited = edit_threads(&s.load, 1);

    // Load with the edited params.
    higgs
        .load(TINY_MODEL_ID, Some(edited))
        .await
        .expect("load with edited params");
    let st = higgs.status().await.unwrap();
    assert_eq!(st.loaded.and_then(|l| l.threads), Some(1), "edit applied");

    // Unload, then a PLAIN reload must reuse the EDITED threads (1), not the tuned value.
    higgs.unload().await.unwrap();
    higgs.load(TINY_MODEL_ID, None).await.unwrap();
    let st2 = higgs.status().await.unwrap();
    assert_eq!(
        st2.loaded.and_then(|l| l.threads),
        Some(1),
        "plain reload reused the accepted EDIT (threads=1), not the stale tune"
    );

    // The model is now RESIDENT. Loading again with a new edit (threads=2) is an
    // idempotent no-op — the request params are NOT applied to the resident worker,
    // and (success-before-persist) are NOT saved to the profile either. So a later
    // plain reload still reuses the PREVIOUS accepted edit (threads=1), not threads=2.
    let edited2 = edit_threads(&s.load, 2);
    higgs
        .load(TINY_MODEL_ID, Some(edited2))
        .await
        .expect("idempotent load of a resident model");
    higgs.unload().await.unwrap();
    higgs.load(TINY_MODEL_ID, None).await.unwrap();
    let st3 = higgs.status().await.unwrap();
    assert_eq!(
        st3.loaded.and_then(|l| l.threads),
        Some(1),
        "resident-load edit was a no-op + unsaved (success-before-persist); profile unchanged"
    );

    higgs.shutdown().await;
}

/// Two concurrent tunes of DIFFERENT models must BOTH persist their profile — the
/// `models.json` write is serialized + re-read, so neither flush clobbers the
/// other (regression guard for the whole-file-rewrite race). Verified indirectly:
/// a plain reload of each model reuses ITS saved profile (`flash_attn: On`), which
/// only survives if its `TuneRecord` was not dropped.
#[tokio::test]
async fn concurrent_tunes_persist_both_profiles() {
    let ids = ["zz/alpha", "zz/beta"];
    let Some(higgs) = higgs_local(&ids).await else {
        eprintln!("SKIP concurrent_tunes: no tiny GGUF (set HIGGS_TEST_GGUF)");
        return;
    };

    // Fire both tunes concurrently — the card-fetch wait keeps them overlapping.
    let tune = |id: &str| {
        let h = higgs.handle();
        let id = id.to_string();
        async move {
            h.tune(TuneRequest {
                id,
                mode: None,
                budget: None,
                pins: None,
            })
            .await
        }
    };
    let (sa, sb) = tokio::join!(tune(ids[0]), tune(ids[1]));
    assert!(sa.is_ok(), "tune {} ok: {sa:?}", ids[0]);
    assert!(sb.is_ok(), "tune {} ok: {sb:?}", ids[1]);

    // Each model's saved profile survived: a PLAIN load reuses it (flash_attn on).
    for id in ids {
        higgs.load(id, None).await.expect("plain load");
        let entry = higgs.model_by_id(id).await.unwrap();
        let last = entry
            .last_load
            .unwrap_or_else(|| panic!("last_load persisted for {id}"));
        assert_eq!(
            last.as_llamacpp().flash_attn,
            Some(FlashAttn::On),
            "concurrent tune for {id} persisted (profile reused on plain load): {last:?}"
        );
        // Unload so the next model's load isn't gated by a resident worker cap.
        higgs.unload().await.unwrap();
    }

    higgs.shutdown().await;
}

/// Clone the umbrella and override the generation-thread count (the accepted-edit
/// shape the tests exercise).
fn edit_threads(load: &LoadParams, threads: u32) -> LoadParams {
    let mut p = load.as_llamacpp().clone();
    p.threads = threads;
    LoadParams::llamacpp(p)
}
