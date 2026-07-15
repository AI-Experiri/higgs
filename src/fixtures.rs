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

use std::io::Cursor;
use std::path::Path;

use ggus::{GGufFileHeader, GGufFileWriter, GGufMetaDataValueType as T};

use crate::api::Higgs;

/// A GGUF metadata string value: `u64` length prefix + raw bytes.
fn gguf_string(s: &str) -> Vec<u8> {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(8 + bytes.len());
    out.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(bytes);
    out
}

/// Serialize `kvs` as a header-only GGUF (no tensors) and write it to
/// `<root>/<id>/<filename>`, creating the model dir.
fn write_header_only(root: &Path, id: &str, filename: &str, kvs: &[(&str, T, Vec<u8>)]) {
    let mut buf = Cursor::new(Vec::<u8>::new());
    let mut w = GGufFileWriter::new(&mut buf, GGufFileHeader::new(3, 0, kvs.len() as u64)).unwrap();
    for (k, t, v) in kvs {
        w.write_meta_kv(k, *t, v).unwrap();
    }
    w.finish::<Vec<u8>>(false).finish().unwrap();

    let path = root.join(id).join(filename);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, buf.into_inner()).unwrap();
}

/// A minimal valid GENERATIVE GGUF (arch=llama, ctx=4096, chat template) at
/// `<root>/<id>/model-Q4_K_M.gguf`, so a scan discovers `id` with enriched
/// metadata and [`crate::worker::models::ModelDomain::Llm`].
pub fn write_gguf_fixture(root: &Path, id: &str) {
    write_header_only(
        root,
        id,
        "model-Q4_K_M.gguf",
        &[
            ("general.architecture", T::String, gguf_string("llama")),
            (
                "llama.context_length",
                T::U32,
                4096u32.to_le_bytes().to_vec(),
            ),
            (
                "tokenizer.chat_template",
                T::String,
                gguf_string("{% for m in messages %}{{ m.content }}{% endfor %}"),
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
        &[
            ("general.architecture", T::String, gguf_string("bert")),
            ("bert.context_length", T::U32, 512u32.to_le_bytes().to_vec()),
            ("bert.pooling_type", T::U32, 2u32.to_le_bytes().to_vec()),
            ("bert.attention.causal", T::Bool, vec![0u8]),
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
        &[
            ("general.architecture", T::String, gguf_string("bert")),
            ("bert.context_length", T::U32, 512u32.to_le_bytes().to_vec()),
            ("bert.pooling_type", T::U32, 4u32.to_le_bytes().to_vec()),
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
    use crate::worker::engine::{CtxLen, GpuLayers, LoadParams};
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
            profile: LoadParams::base(CtxLen::Auto, GpuLayers::All, 8),
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
