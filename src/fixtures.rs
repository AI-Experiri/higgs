//! Test fixtures — real GGUF files and a prepared profile — for tests that need
//! a higgs with actual, scannable models.
//!
//! Compiled only under `cfg(test)` or the `test-support` feature, so none of this
//! reaches a production build. The feature is what lets an EMBEDDER's tests build
//! a non-trivial higgs: jigglebot's control-op tests need a facade whose
//! `chat_model_ids` is non-empty, and a test asserting a list is "forwarded
//! verbatim" is vacuous while that list is empty (a hardcoded `Vec::new()` in the
//! op passes it) — Fable r12 proved exactly that mutant survives.
//!
//! ```toml
//! # the embedder's Cargo.toml
//! [dev-dependencies]
//! higgs = { path = "…", features = ["test-support"] }
//! ```
//!
//! These are the ONE in-crate source of the fixture bytes: `serve::test_support`
//! re-exports them, so the [HG079] domain fixtures cannot drift between the unit
//! and embedder suites. (`tests/common/mod.rs` still carries its own copy — an
//! integration test compiles against the lib as an external crate and would need
//! the feature enabled to see this module.)

use std::path::Path;

use gguf_rs_lib::builder::GGUFBuilder;
use gguf_rs_lib::format::metadata::MetadataValue as V;

use crate::api::Higgs;

/// Serialize `kvs` as a header-only GGUF (no tensors) and write it to
/// `<root>/<id>/<filename>`, creating the model dir.
fn write_header_only(root: &Path, id: &str, filename: &str, kvs: Vec<(&str, V)>) {
    let mut b = GGUFBuilder::new();
    for (k, v) in kvs {
        b = b.add_metadata(k, v);
    }
    let (bytes, _) = b.build_to_bytes().unwrap();

    let path = root.join(id).join(filename);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, bytes).unwrap();
}

/// A minimal valid GENERATIVE GGUF (arch=llama, ctx=4096, chat template) at
/// `<root>/<id>/model-Q4_K_M.gguf`, so a scan discovers `id` with enriched
/// metadata and [`crate::worker::models::ModelDomain::Llm`].
pub fn write_gguf_fixture(root: &Path, id: &str) {
    write_header_only(
        root,
        id,
        "model-Q4_K_M.gguf",
        vec![
            ("general.architecture", V::String("llama".into())),
            ("llama.context_length", V::U32(4096)),
            (
                "tokenizer.chat_template",
                V::String("{% for m in messages %}{{ m.content }}{% endfor %}".into()),
            ),
        ],
    );
}

/// A header-only EMBEDDING GGUF — the arch-scoped keys declare a pooling head and
/// non-causal attention, the two header shapes `read_domain` classifies as
/// [`crate::worker::models::ModelDomain::Embedding`] (bge-small's actual
/// declarations). For [HG079] chat-vs-embedding gate tests that need a scannable
/// embedding model without a real weights file.
pub fn write_embedding_gguf_fixture(root: &Path, id: &str) {
    write_embedding_gguf_fixture_named(root, id, "model-f16.gguf");
}

/// [`write_embedding_gguf_fixture`] with a caller-chosen FILENAME — scan order
/// inside one model dir is path-lexical, so a test that needs the embedding
/// variant to sort before/after a sibling controls the order through the name.
pub fn write_embedding_gguf_fixture_named(root: &Path, id: &str, filename: &str) {
    write_header_only(
        root,
        id,
        filename,
        vec![
            ("general.architecture", V::String("bert".into())),
            ("bert.context_length", V::U32(512)),
            ("bert.pooling_type", V::U32(2)),
            ("bert.attention.causal", V::Bool(false)),
        ],
    );
}

/// A header-only RERANKER GGUF (llama.cpp `RANK` pooling, `pooling_type` = 4,
/// causal attention — the bge-reranker shape). The [HG079] gates refuse ANY
/// non-Llm domain; tests use this to pin that the comparisons are `!= Llm`, not
/// `== Embedding` (a mutation the embedding fixtures alone cannot catch).
pub fn write_reranker_gguf_fixture(root: &Path, id: &str) {
    write_header_only(
        root,
        id,
        "model-Q8_0.gguf",
        vec![
            ("general.architecture", V::String("bert".into())),
            ("bert.context_length", V::U32(512)),
            ("bert.pooling_type", V::U32(4)),
        ],
    );
}

/// Seed a FRESH tuning profile for `id` directly into the models store, skipping
/// the real Prepare (which runs a bounded HF card fetch that stalls ~10s when
/// offline or firewalled). Anchored to the CURRENT hardware and model file, so
/// readiness reads it as prepared — `Servable` when serving is on, which is what
/// puts the id in `chat_model_ids`' JIT leg.
///
/// **Set `HIGGS_HOME` to a temp dir before calling this.** It writes a `TuneRecord`
/// into the models store rooted at `HIGGS_HOME` — with the default home that is the
/// developer's real `~/.higgs/models.json`. A test that forgets to redirect it would
/// clobber that file. (This is compiled only under `test`/`test-support`, so a
/// production build can never reach it — but an embedder's test can.)
///
/// Panics if `id` is not scannable — seeding a profile for a model that does not
/// exist on disk is always a test bug.
pub async fn seed_prepared_profile(higgs: &Higgs, id: &str) {
    seed_profile_with_ctx(higgs, id, crate::worker::engine::CtxLen::Auto).await;
}

/// [`seed_prepared_profile`] with a FIXED context window — the tuned-metrics
/// shape whose concrete window surfaces as `HiggsModelEntry::tuned_max_tokens`.
/// `put_tuning` routes the Heuristic record into the analytical history slot, so
/// the entry reads as genuinely TUNED (Servable AND a concrete window), where
/// the Auto-window [`seed_prepared_profile`] is merely prepared (Servable but no
/// tuned metrics — an embedder's tuned-models-only provider gate excludes it).
/// The same **`HIGGS_HOME` warning** applies — redirect it to a temp dir first.
pub async fn seed_tuned_profile(higgs: &Higgs, id: &str, ctx: u32) {
    // `CtxLen::fixed(0)` normalizes to Auto — that would silently seed the
    // merely-prepared shape this fixture exists to contrast with. A test that
    // wants Auto calls `seed_prepared_profile`.
    assert_ne!(ctx, 0, "seed_tuned_profile needs a non-zero window");
    seed_profile_with_ctx(higgs, id, crate::worker::engine::CtxLen::fixed(ctx)).await;
}

async fn seed_profile_with_ctx(higgs: &Higgs, id: &str, ctx: crate::worker::engine::CtxLen) {
    use crate::worker::engine::{GpuLayers, LoadParams};
    let hw = higgs.hardware().await;
    let path = higgs
        .scan()
        .await
        .ok()
        .and_then(|ms| ms.into_iter().find(|m| m.id == id).map(|m| m.path))
        .expect("fixture model is scannable");
    let store = higgs.models_store().expect("open models store");
    store.put_tuning(
        id,
        crate::tune::store::TuneRecord {
            profile: LoadParams::base(ctx, GpuLayers::All, 8),
            sampling: crate::worker::engine::SamplingParams::default(),
            budget: crate::tune::ResourceBudget::default(),
            provenance: crate::tune::TuneProvenance::Heuristic,
            bench_tps: None,
            tuned_at_ms: 0,
            hw_fingerprint: hw.fingerprint(),
            model_file_sig: crate::api::file_sig(&path),
        },
    );
    store.flush().expect("persist seeded profile");
}
