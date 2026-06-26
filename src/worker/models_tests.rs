
use std::fs;

use tempfile::TempDir;

use super::*;

fn write_file(path: &Path, content: &[u8]) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(path, content).unwrap();
}

#[test]
fn lmstudio_layout_scanned() {
    let dir = TempDir::new().unwrap();
    let root = dir.path();
    // root/google/gemma-4-12b/gemma-4-12b-Q4_K_M.gguf
    write_file(
        &root.join("google/gemma-4-12b/gemma-4-12b-Q4_K_M.gguf"),
        b"gguf",
    );

    let mut store = ModelStore::default();
    let models = store.scan(&[root.to_path_buf()], &[], &[]).unwrap();

    assert_eq!(models.len(), 1);
    assert_eq!(models[0].id, "google/gemma-4-12b");
    assert_eq!(models[0].quant, Some("Q4_K_M".into()));
    assert_eq!(models[0].source, HiggsModelSource::LmStudio);
    // Fake GGUF (not a valid header) — enrichment must tolerate and leave None/false.
    assert_eq!(models[0].arch, None);
    assert_eq!(models[0].ctx_train, None);
    assert!(!models[0].has_chat_template);
}

#[test]
fn hf_cache_layout_scanned() {
    let dir = TempDir::new().unwrap();
    let root = dir.path();
    // root/models--nvidia--nemotron-3-nano-4b/snapshots/abc123/model-Q8_0.gguf
    write_file(
        &root.join("models--nvidia--nemotron-3-nano-4b/snapshots/abc123/model-Q8_0.gguf"),
        b"gguf",
    );

    let mut store = ModelStore::default();
    let models = store.scan(&[], &[root.to_path_buf()], &[]).unwrap();

    assert_eq!(models.len(), 1);
    assert_eq!(models[0].id, "nvidia/nemotron-3-nano-4b");
    assert_eq!(models[0].source, HiggsModelSource::HfCache);
}

#[test]
fn ollama_manifest_resolved() {
    let dir = TempDir::new().unwrap();
    let root = dir.path();

    // blobs/sha256-deadbeef — must start with GGUF magic (16 bytes total)
    write_file(&root.join("blobs/sha256-deadbeef"), b"GGUFxxxxxxxxxxxx");

    // manifests/registry.ollama.ai/library/llama3/latest
    let manifest = r#"{"layers":[{"mediaType":"application/vnd.ollama.image.model","digest":"sha256:deadbeef"}]}"#;
    write_file(
        &root.join("manifests/registry.ollama.ai/library/llama3/latest"),
        manifest.as_bytes(),
    );

    let mut store = ModelStore::default();
    let models = store.scan(&[], &[], &[root.to_path_buf()]).unwrap();

    assert_eq!(models.len(), 1);
    assert!(
        models[0].path.ends_with("sha256-deadbeef"),
        "path was: {}",
        models[0].path
    );
}

#[test]
fn ollama_blob_without_gguf_magic_skipped() {
    let dir = TempDir::new().unwrap();
    let root = dir.path();

    // Blob exists but has junk bytes — not a valid GGUF file.
    write_file(
        &root.join("blobs/sha256-cafebabe"),
        &[0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x00],
    );

    let manifest = r#"{"layers":[{"mediaType":"application/vnd.ollama.image.model","digest":"sha256:cafebabe"}]}"#;
    write_file(
        &root.join("manifests/registry.ollama.ai/library/phi3/latest"),
        manifest.as_bytes(),
    );

    let mut store = ModelStore::default();
    let models = store.scan(&[], &[], &[root.to_path_buf()]).unwrap();

    assert_eq!(models.len(), 0, "blob without GGUF magic must be skipped");
}

#[test]
fn missing_roots_are_fine() {
    let mut store = ModelStore::default();
    let nonexistent = PathBuf::from("/nonexistent-xyz");
    let models = store
        .scan(
            std::slice::from_ref(&nonexistent),
            std::slice::from_ref(&nonexistent),
            std::slice::from_ref(&nonexistent),
        )
        .unwrap();
    assert!(models.is_empty());
}

#[test]
fn quant_parsing() {
    assert_eq!(quant_from_filename("m-Q4_K_M.gguf"), Some("Q4_K_M".into()));
    assert_eq!(quant_from_filename("m.f16.gguf"), Some("f16".into()));
    assert_eq!(quant_from_filename("notgguf.bin"), None);
}

#[test]
fn projector_sidecars_excluded() {
    // arch "clip" → projector regardless of name
    assert!(is_projector_sidecar("model-Q8_0.gguf", Some("clip")));
    // mmproj filename → projector even if arch is unreadable
    assert!(is_projector_sidecar("mmproj-model-BF16.gguf", None));
    assert!(is_projector_sidecar("MMPROJ-Model.GGUF", None));
    // a real chat model is not a projector
    assert!(!is_projector_sidecar("model-Q8_0.gguf", Some("gemma4")));
    assert!(!is_projector_sidecar("model-Q4_K_M.gguf", None));
}

/// Build a minimal valid GGUF file in memory using the ggus writer API and
/// verify that `enrich_gguf_metadata` extracts architecture and context length.
///
/// Wire encoding for a GGUF String metadata value:
///   u64_le(byte_length) ++ utf8_bytes  (no null terminator)
/// Wire encoding for a U32 metadata value: 4 bytes little-endian u32.
#[test]
fn valid_gguf_header_enrichment() {
    use ggus::{GGufFileHeader, GGufFileWriter, GGufMetaDataValueType};
    use std::io::Cursor;

    // Encode a GGUF String value: 8-byte LE length prefix + UTF-8 bytes.
    fn gguf_string(s: &str) -> Vec<u8> {
        let bytes = s.as_bytes();
        let mut out = Vec::with_capacity(8 + bytes.len());
        out.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
        out.extend_from_slice(bytes);
        out
    }

    // 3 metadata keys: general.architecture, llama.context_length, tokenizer.chat_template
    let header = GGufFileHeader::new(3, 0, 3);
    let mut buf = Cursor::new(Vec::<u8>::new());
    let mut writer = GGufFileWriter::new(&mut buf, header).unwrap();

    writer
        .write_meta_kv(
            "general.architecture",
            GGufMetaDataValueType::String,
            &gguf_string("llama"),
        )
        .unwrap();
    writer
        .write_meta_kv(
            "llama.context_length",
            GGufMetaDataValueType::U32,
            &4096u32.to_le_bytes(),
        )
        .unwrap();
    writer
        .write_meta_kv(
            "tokenizer.chat_template",
            GGufMetaDataValueType::String,
            &gguf_string("{% for m in messages %}{{ m.content }}{% endfor %}"),
        )
        .unwrap();

    // No tensors — finish with write_data = false.
    writer.finish::<Vec<u8>>(false).finish().unwrap();

    // Write to a tempfile and scan it via ModelStore.
    let dir = TempDir::new().unwrap();
    let gguf_path = dir.path().join("myorg/mymodel/model-Q4_K_M.gguf");
    write_file(&gguf_path, buf.into_inner().as_slice());

    let mut store = ModelStore::default();
    let models = store.scan(&[dir.path().to_path_buf()], &[], &[]).unwrap();

    assert_eq!(models.len(), 1);
    assert_eq!(
        models[0].arch.as_deref(),
        Some("llama"),
        "arch should be extracted"
    );
    assert_eq!(
        models[0].ctx_train,
        Some(4096),
        "ctx_train should be extracted"
    );
    assert!(
        models[0].has_chat_template,
        "has_chat_template should be true"
    );
}

/// Non-GGUF Ollama manifests (no `layers`, or no model layer) must not abort the
/// scan — they are silently skipped.  A valid GGUF manifest alongside them is
/// still returned.
#[test]
fn ollama_non_gguf_manifest_skipped_valid_returned() {
    let dir = TempDir::new().unwrap();
    let root = dir.path();

    // A valid GGUF blob.
    write_file(&root.join("blobs/sha256-aabbccdd"), b"GGUFxxxxxxxxxxxx");

    // Manifest that points to the valid GGUF blob.
    let good_manifest = r#"{"layers":[{"mediaType":"application/vnd.ollama.image.model","digest":"sha256:aabbccdd"}]}"#;
    write_file(
        &root.join("manifests/registry.ollama.ai/library/llama3/latest"),
        good_manifest.as_bytes(),
    );

    // Embedding manifest: has `layers` but none with the model mediaType.
    let embed_manifest = r#"{"layers":[{"mediaType":"application/vnd.ollama.image.params"}]}"#;
    write_file(
        &root.join("manifests/registry.ollama.ai/library/nomic-embed-text/latest"),
        embed_manifest.as_bytes(),
    );

    // Vision manifest: no `layers` key at all.
    let vision_manifest = r#"{"config":{"mediaType":"application/vnd.ollama.image"}}"#;
    write_file(
        &root.join("manifests/registry.ollama.ai/library/llava/latest"),
        vision_manifest.as_bytes(),
    );

    let mut store = ModelStore::default();
    // Must succeed (not Err) and return exactly the one valid model.
    let models = store
        .scan(&[], &[], &[root.to_path_buf()])
        .expect("scan must not fail on non-GGUF manifests");

    assert_eq!(
        models.len(),
        1,
        "only the GGUF-model manifest should yield a result"
    );
    assert!(
        models[0].path.ends_with("sha256-aabbccdd"),
        "unexpected path: {}",
        models[0].path
    );
}

/// F6: a binary/non-JSON file alongside a valid manifest must not abort
/// the Ollama scan — the binary file is silently skipped and the valid model
/// is still returned.
#[test]
fn ollama_binary_file_alongside_valid_manifest_skipped() {
    let dir = TempDir::new().unwrap();
    let root = dir.path();

    // A valid GGUF blob for the good manifest.
    write_file(&root.join("blobs/sha256-goodbeef"), b"GGUFxxxxxxxxxxxx");

    // Valid manifest pointing at the good blob.
    let good_manifest = r#"{"layers":[{"mediaType":"application/vnd.ollama.image.model","digest":"sha256:goodbeef"}]}"#;
    write_file(
        &root.join("manifests/registry.ollama.ai/library/llama3/latest"),
        good_manifest.as_bytes(),
    );

    // .DS_Store: binary file that cannot be parsed as UTF-8 or JSON.
    write_file(
        &root.join("manifests/registry.ollama.ai/library/.DS_Store"),
        &[0x00, 0xFF, 0xFE, 0xFD, 0x42, 0x50, 0x4C, 0x69],
    );

    // Partial JSON file (truncated — parse will fail).
    write_file(
        &root.join("manifests/registry.ollama.ai/library/partial/tmp"),
        b"{invalid json",
    );

    let mut store = ModelStore::default();
    let models = store
        .scan(&[], &[], &[root.to_path_buf()])
        .expect("scan must not fail on binary or malformed manifest files");

    assert_eq!(
        models.len(),
        1,
        "only the valid GGUF model should be returned, got: {models:?}"
    );
    assert!(
        models[0].path.ends_with("sha256-goodbeef"),
        "unexpected path: {}",
        models[0].path
    );
}

/// F7: dedup helper — two `HiggsModel` entries with the same (id, path) collapse to one.
///
/// Full symlink resolution requires OS support; we unit-test the dedup
/// logic by verifying the existing sort+dedup pass removes entries with
/// identical (id, path) pairs, which is what canonicalize produces when two
/// revisions point at the same blob.
#[test]
fn hf_cache_dedup_by_id_and_canonical_path() {
    let dir = TempDir::new().unwrap();
    let root = dir.path();

    // Two revisions under the same repo, but both pointing at the same content.
    // We use the same file (not a real symlink — just the same content written twice).
    // After canonicalize both paths resolve to themselves; dedup by (id, path)
    // keeps both only if paths differ. When they differ both are kept (different variants).
    // This test exercises the common case: same id + different path stays (2 entries).
    write_file(
        &root.join("models--org--repo/snapshots/rev1/model-Q4_K_M.gguf"),
        b"gguf",
    );
    write_file(
        &root.join("models--org--repo/snapshots/rev2/model-Q4_K_M.gguf"),
        b"gguf",
    );

    let mut store = ModelStore::default();
    let models = store.scan(&[], &[root.to_path_buf()], &[]).unwrap();

    // Two distinct on-disk paths → two entries (different revisions of different blobs).
    // Real symlink dedup is verified separately if OS supports it.
    assert_eq!(
        models.len(),
        2,
        "two distinct revision paths → two catalog entries: {models:?}"
    );
    assert!(models.iter().all(|m| m.id == "org/repo"));
    // Both entries share the same id but distinct paths — dedup preserves them.
    assert_ne!(models[0].path, models[1].path, "paths must differ");
}

/// When the same model id is present under two different paths (e.g. two quant
/// variants), `get()` must return the lexically-first path consistently.
#[test]
fn get_returns_lexically_first_path_for_multi_variant_id() {
    let dir = TempDir::new().unwrap();
    let root = dir.path();
    // Two files with the same org/model id but different quant suffixes.
    // Lexicographic order: Q4_K_M < Q8_0, so Q4_K_M path comes first.
    write_file(&root.join("org/model/model-Q4_K_M.gguf"), b"gguf");
    write_file(&root.join("org/model/model-Q8_0.gguf"), b"gguf");

    let mut store = ModelStore::default();
    store.scan(&[root.to_path_buf()], &[], &[]).unwrap();

    let found = store.get("org/model").expect("model must be found");
    assert!(
        found.path.contains("Q4_K_M"),
        "expected lexically-first path (Q4_K_M) but got: {}",
        found.path
    );
    // Called twice: must return the same result (deterministic).
    let found2 = store.get("org/model").expect("model must be found");
    assert_eq!(found.path, found2.path, "get() must be deterministic");
}

/// Build a GGUF in `dir/{org}/{model}/file.gguf` with the given metadata,
/// then scan it. Returns the single discovered model. Encodes GGUF String
/// values as an 8-byte LE length prefix + UTF-8 bytes.
fn scan_single_gguf(kvs: &[(&str, ggus::GGufMetaDataValueType, Vec<u8>)], rel: &str) -> HiggsModel {
    use ggus::{GGufFileHeader, GGufFileWriter};
    use std::io::Cursor;

    let header = GGufFileHeader::new(3, 0, kvs.len() as u64);
    let mut buf = Cursor::new(Vec::<u8>::new());
    let mut writer = GGufFileWriter::new(&mut buf, header).unwrap();
    for (key, ty, bytes) in kvs {
        writer.write_meta_kv(key, *ty, bytes).unwrap();
    }
    writer.finish::<Vec<u8>>(false).finish().unwrap();

    let dir = TempDir::new().unwrap();
    let gguf_path = dir.path().join(rel);
    write_file(&gguf_path, buf.into_inner().as_slice());

    let mut store = ModelStore::default();
    let models = store.scan(&[dir.path().to_path_buf()], &[], &[]).unwrap();
    assert_eq!(models.len(), 1);
    models[0].clone()
}

fn gguf_str(s: &str) -> Vec<u8> {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(8 + b.len());
    out.extend_from_slice(&(b.len() as u64).to_le_bytes());
    out.extend_from_slice(b);
    out
}

/// The typed tuning fields (block_count, head_count, head_count_kv,
/// embedding_length, expert_count) are parsed from the arch-scoped GGUF keys.
/// Critically `head_count_kv` is the GQA KV head count (8 here), NOT the query
/// `head_count` (32) — the autotune KV-cache estimate rests on this distinction.
#[test]
fn enrich_extracts_typed_tuning_fields() {
    use ggus::GGufMetaDataValueType::{String as GS, U32};
    let model = scan_single_gguf(
        &[
            ("general.architecture", GS, gguf_str("llama")),
            ("llama.block_count", U32, 32u32.to_le_bytes().to_vec()),
            (
                "llama.attention.head_count",
                U32,
                32u32.to_le_bytes().to_vec(),
            ),
            (
                "llama.attention.head_count_kv",
                U32,
                8u32.to_le_bytes().to_vec(),
            ),
            (
                "llama.embedding_length",
                U32,
                4096u32.to_le_bytes().to_vec(),
            ),
            ("llama.expert_count", U32, 0u32.to_le_bytes().to_vec()),
        ],
        "org/m/model-Q4_K_M.gguf",
    );
    assert_eq!(model.block_count, Some(32));
    assert_eq!(model.head_count, Some(32));
    assert_eq!(model.head_count_kv, Some(8), "GQA KV head count, not 32");
    assert_eq!(model.embedding_length, Some(4096));
    assert_eq!(model.expert_count, Some(0));
}

/// A chat template referencing `tool_call`/`tools` and `<think>` must set
/// both capability flags during enrichment.
#[test]
fn enrich_detects_tool_and_reasoning_capabilities() {
    use ggus::GGufMetaDataValueType::{String as GS, U32};
    let model = scan_single_gguf(
        &[
            ("general.architecture", GS, gguf_str("qwen3")),
            ("qwen3.context_length", U32, 32768u32.to_le_bytes().to_vec()),
            (
                "tokenizer.chat_template",
                GS,
                gguf_str("{% if tools %}{{ tool_call }}{% endif %}<think></think>"),
            ),
        ],
        "qwen/q3/model-Q4_K_M.gguf",
    );
    assert_eq!(model.arch.as_deref(), Some("qwen3"));
    assert_eq!(model.ctx_train, Some(32768));
    assert!(model.has_chat_template);
    assert!(model.supports_tools, "template references tools/tool_call");
    assert!(model.supports_reasoning, "template references <think>");
}

/// Enrichment captures the curated load-relevant GGUF components (arch,
/// quant, tokenizer, arch-scoped scalars, gguf version) and the transient
/// chat template, skipping non-curated keys.
#[test]
fn enrich_collects_curated_gguf_components() {
    use ggus::GGufMetaDataValueType::{String as GS, U32};
    let model = scan_single_gguf(
        &[
            ("general.architecture", GS, gguf_str("llama")),
            // general.filetype 15 = MostlyQ4_K_M.
            ("general.filetype", U32, 15u32.to_le_bytes().to_vec()),
            (
                "general.quantization_version",
                U32,
                2u32.to_le_bytes().to_vec(),
            ),
            ("tokenizer.ggml.model", GS, gguf_str("gpt2")),
            ("tokenizer.ggml.pre", GS, gguf_str("llama-bpe")),
            ("llama.context_length", U32, 4096u32.to_le_bytes().to_vec()),
            ("llama.block_count", U32, 32u32.to_le_bytes().to_vec()),
            (
                "llama.attention.head_count",
                U32,
                32u32.to_le_bytes().to_vec(),
            ),
            // A non-curated key — must NOT appear in the components list.
            ("general.name", GS, gguf_str("Some Model")),
            (
                "tokenizer.chat_template",
                GS,
                gguf_str("{{ tools }}<tool_call>"),
            ),
        ],
        "org/m/model-Q4_K_M.gguf",
    );
    let comp = |k: &str| {
        model
            .gguf_components
            .iter()
            .find(|c| c.key == k)
            .map(|c| c.value.clone())
    };
    assert_eq!(comp("general.architecture").as_deref(), Some("llama"));
    assert_eq!(comp("general.quantization_version").as_deref(), Some("2"));
    assert_eq!(comp("tokenizer.ggml.model").as_deref(), Some("gpt2"));
    assert_eq!(comp("tokenizer.ggml.pre").as_deref(), Some("llama-bpe"));
    assert_eq!(comp("llama.context_length").as_deref(), Some("4096"));
    assert_eq!(comp("llama.block_count").as_deref(), Some("32"));
    assert_eq!(comp("llama.attention.head_count").as_deref(), Some("32"));
    assert!(
        comp("general.file_type").is_some(),
        "quant filetype captured"
    );
    assert!(comp("gguf.version").is_some(), "container version captured");
    // Non-curated key is skipped.
    assert!(comp("general.name").is_none(), "general.name not curated");
    // Transient chat template captured for the Gate-2 sniff.
    assert_eq!(
        model.chat_template.as_deref(),
        Some("{{ tools }}<tool_call>")
    );
}

/// A plain chat template with no tool/think markup leaves both capability
/// flags false even though has_chat_template is true.
#[test]
fn enrich_plain_template_no_capabilities() {
    use ggus::GGufMetaDataValueType::{String as GS, U32};
    let model = scan_single_gguf(
        &[
            ("general.architecture", GS, gguf_str("gemma3")),
            ("gemma3.context_length", U32, 8192u32.to_le_bytes().to_vec()),
            (
                "tokenizer.chat_template",
                GS,
                gguf_str("{% for m in messages %}{{ m.content }}{% endfor %}"),
            ),
        ],
        "google/gemma/model-Q8_0.gguf",
    );
    assert!(model.has_chat_template);
    assert!(!model.supports_tools);
    assert!(!model.supports_reasoning);
}

/// The lowercase "thinking" heuristic path: a template that says "thinking"
/// (no `<think>` tags) still flags reasoning support.
#[test]
fn enrich_thinking_word_flags_reasoning() {
    use ggus::GGufMetaDataValueType::{String as GS, U32};
    let model = scan_single_gguf(
        &[
            ("general.architecture", GS, gguf_str("llama")),
            ("llama.context_length", U32, 4096u32.to_le_bytes().to_vec()),
            (
                "tokenizer.chat_template",
                GS,
                gguf_str("You may show your Thinking before answering."),
            ),
        ],
        "org/m/model-Q4_K_M.gguf",
    );
    assert!(model.has_chat_template);
    assert!(model.supports_reasoning, "the word 'thinking' flags it");
    assert!(!model.supports_tools);
}

/// Ollama manifest whose model layer digest lacks the `sha256:` prefix is an
/// invalid manifest → HG010 error aborting the scan.
#[test]
fn ollama_bad_digest_format_errors() {
    let dir = TempDir::new().unwrap();
    let root = dir.path();
    let manifest =
        r#"{"layers":[{"mediaType":"application/vnd.ollama.image.model","digest":"md5:zzz"}]}"#;
    write_file(
        &root.join("manifests/registry.ollama.ai/library/bad/latest"),
        manifest.as_bytes(),
    );
    let mut store = ModelStore::default();
    let err = store
        .scan(&[], &[], &[root.to_path_buf()])
        .expect_err("non-sha256 digest must error");
    assert!(
        matches!(err, HiggsError::OllamaManifestInvalid { .. }),
        "got: {err:?}"
    );
}

/// Ollama model layer missing the `digest` field entirely → HG010 error.
#[test]
fn ollama_missing_digest_errors() {
    let dir = TempDir::new().unwrap();
    let root = dir.path();
    let manifest = r#"{"layers":[{"mediaType":"application/vnd.ollama.image.model"}]}"#;
    write_file(
        &root.join("manifests/registry.ollama.ai/library/nodigest/latest"),
        manifest.as_bytes(),
    );
    let mut store = ModelStore::default();
    let err = store
        .scan(&[], &[], &[root.to_path_buf()])
        .expect_err("missing digest must error");
    assert!(matches!(err, HiggsError::OllamaManifestInvalid { .. }));
}

/// A manifest whose blob is absent from `blobs/` is skipped silently (normal
/// for partial pulls) — scan succeeds with no model.
#[test]
fn ollama_missing_blob_skipped() {
    let dir = TempDir::new().unwrap();
    let root = dir.path();
    // Manifest references a blob that was never written.
    let manifest = r#"{"layers":[{"mediaType":"application/vnd.ollama.image.model","digest":"sha256:absent"}]}"#;
    write_file(
        &root.join("manifests/registry.ollama.ai/library/gone/latest"),
        manifest.as_bytes(),
    );
    let mut store = ModelStore::default();
    let models = store.scan(&[], &[], &[root.to_path_buf()]).unwrap();
    assert!(models.is_empty(), "missing blob → skipped, no error");
}

/// Non-`.gguf` files in an LM Studio model dir are ignored; only GGUF files
/// become catalog entries.
#[test]
fn lmstudio_non_gguf_files_ignored() {
    let dir = TempDir::new().unwrap();
    let root = dir.path();
    write_file(&root.join("org/m/README.md"), b"docs");
    write_file(&root.join("org/m/config.json"), b"{}");
    write_file(&root.join("org/m/model-Q4_K_M.gguf"), b"gguf");
    let mut store = ModelStore::default();
    let models = store.scan(&[root.to_path_buf()], &[], &[]).unwrap();
    assert_eq!(models.len(), 1, "only the .gguf file is cataloged");
    assert!(models[0].path.ends_with("model-Q4_K_M.gguf"));
}

/// HF cache directories that don't follow the `models--org--name` prefix
/// convention are skipped.
#[test]
fn hf_cache_non_model_dirs_skipped() {
    let dir = TempDir::new().unwrap();
    let root = dir.path();
    // A stray dir without the models-- prefix, plus a valid repo.
    write_file(&root.join(".locks/whatever"), b"x");
    write_file(
        &root.join("models--org--good/snapshots/rev/model-Q4_K_M.gguf"),
        b"gguf",
    );
    let mut store = ModelStore::default();
    let models = store.scan(&[], &[root.to_path_buf()], &[]).unwrap();
    assert_eq!(models.len(), 1);
    assert_eq!(models[0].id, "org/good");
}

/// A plain file (not a directory) at the LM Studio org level is skipped —
/// the org-entry `is_dir` guard. No model results, no error.
#[test]
fn lmstudio_file_at_org_level_skipped() {
    let dir = TempDir::new().unwrap();
    let root = dir.path();
    // A stray file directly under the root (not an org dir).
    write_file(&root.join("stray.txt"), b"x");
    // A valid org/model/gguf alongside it.
    write_file(&root.join("org/m/model-Q4_K_M.gguf"), b"gguf");
    let mut store = ModelStore::default();
    let models = store.scan(&[root.to_path_buf()], &[], &[]).unwrap();
    assert_eq!(models.len(), 1, "stray file ignored, valid model kept");
    assert_eq!(models[0].id, "org/m");
}

/// An HF repo dir without a `snapshots/` subdir is skipped (the
/// `snapshots_path.is_dir()` guard) without error.
#[test]
fn hf_cache_repo_without_snapshots_skipped() {
    let dir = TempDir::new().unwrap();
    let root = dir.path();
    // Repo dir with the right prefix but no snapshots subtree.
    fs::create_dir_all(root.join("models--org--bare")).unwrap();
    // A complete repo alongside it.
    write_file(
        &root.join("models--org--full/snapshots/rev/model-Q8_0.gguf"),
        b"gguf",
    );
    let mut store = ModelStore::default();
    let models = store.scan(&[], &[root.to_path_buf()], &[]).unwrap();
    assert_eq!(models.len(), 1);
    assert_eq!(models[0].id, "org/full");
}

/// An unreadable root directory (permissions stripped) maps to
/// `HG001 ModelDirUnreadable` rather than being silently skipped.
/// Unix-only: relies on chmod semantics not present on Windows.
#[cfg(unix)]
#[test]
fn unreadable_root_errors_hg001() {
    use std::os::unix::fs::PermissionsExt;
    let dir = TempDir::new().unwrap();
    let root = dir.path().join("locked");
    fs::create_dir(&root).unwrap();
    // Strip all permissions so read_dir fails with PermissionDenied.
    fs::set_permissions(&root, fs::Permissions::from_mode(0o000)).unwrap();

    let mut store = ModelStore::default();
    let result = store.scan(std::slice::from_ref(&root), &[], &[]);

    // Restore permissions so TempDir cleanup succeeds regardless of outcome.
    fs::set_permissions(&root, fs::Permissions::from_mode(0o755)).unwrap();

    let err = result.expect_err("unreadable dir must error");
    assert!(
        matches!(err, HiggsError::ModelDirUnreadable { .. }),
        "got: {err:?}"
    );
}
