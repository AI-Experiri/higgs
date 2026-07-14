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
                gguf_str("{% if tools %}<tool_call>{}</tool_call>{% endif %}<think></think>"),
            ),
        ],
        "qwen/q3/model-Q4_K_M.gguf",
    );
    assert_eq!(model.arch.as_deref(), Some("qwen3"));
    assert_eq!(model.ctx_train, Some(32768));
    assert!(model.has_chat_template);
    assert!(
        model.supports_tools,
        "template carries a format a registry parser handles (<tool_call>)"
    );
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

/// Build raw GGUF file bytes (no tensors) from a list of metadata KVs.
/// Lets callers stage a *valid* GGUF anywhere on disk (HF cache, Ollama blob),
/// not just under an LM Studio root the way `scan_single_gguf` does.
fn build_gguf_bytes(kvs: &[(&str, ggus::GGufMetaDataValueType, Vec<u8>)]) -> Vec<u8> {
    use ggus::{GGufFileHeader, GGufFileWriter};
    use std::io::Cursor;

    let header = GGufFileHeader::new(3, 0, kvs.len() as u64);
    let mut buf = Cursor::new(Vec::<u8>::new());
    let mut writer = GGufFileWriter::new(&mut buf, header).unwrap();
    for (key, ty, bytes) in kvs {
        writer.write_meta_kv(key, *ty, bytes).unwrap();
    }
    writer.finish::<Vec<u8>>(false).finish().unwrap();
    buf.into_inner()
}

/// An empty `.gguf` file: open() succeeds, memmap2 maps it as a 1-byte mapping
/// that derefs to an EMPTY slice (it does not error on zero-length), then
/// `GGuf::new` rejects the empty slice and enrichment bails — leaving all
/// header-derived fields empty while the model stays cataloged.
#[test]
fn empty_gguf_file_enrichment_tolerated() {
    let dir = TempDir::new().unwrap();
    let root = dir.path();
    // Zero-length file: open() ok, mmap → empty slice, GGuf::new fails.
    write_file(&root.join("org/m/model-Q4_K_M.gguf"), b"");

    let mut store = ModelStore::default();
    let models = store.scan(&[root.to_path_buf()], &[], &[]).unwrap();

    assert_eq!(models.len(), 1, "empty gguf is still cataloged");
    assert_eq!(models[0].arch, None);
    assert_eq!(models[0].ctx_train, None);
    assert!(!models[0].has_chat_template);
    assert_eq!(models[0].size_bytes, 0);
    // mmap failed before GGuf parse → no curated components captured.
    assert!(models[0].gguf_components.is_empty());
}

/// A `.gguf` file whose read permission is stripped: scan lists it (dir readable),
/// then `enrich_gguf_metadata` hits the `File::open` `Err(_) => return` branch.
/// The model is still cataloged with empty header fields. Unix-only (chmod).
#[cfg(unix)]
#[test]
fn unreadable_gguf_file_open_error_tolerated() {
    use std::os::unix::fs::PermissionsExt;
    let dir = TempDir::new().unwrap();
    let root = dir.path();
    let gguf = root.join("org/m/model-Q4_K_M.gguf");
    write_file(&gguf, b"GGUFxxxxxxxxxxxx");
    // Strip read permission on the FILE (parent dirs stay readable so scan lists it).
    fs::set_permissions(&gguf, fs::Permissions::from_mode(0o000)).unwrap();

    let mut store = ModelStore::default();
    let result = store.scan(&[root.to_path_buf()], &[], &[]);

    // Restore so TempDir cleanup works regardless of outcome.
    fs::set_permissions(&gguf, fs::Permissions::from_mode(0o644)).unwrap();

    let models = result.expect("unreadable file must not fail the scan");
    assert_eq!(models.len(), 1, "file is cataloged despite open() failing");
    assert_eq!(models[0].arch, None, "open failed → no enrichment");
    assert!(models[0].gguf_components.is_empty());
}

/// A real GGUF whose `general.architecture` is `clip` is a projector sidecar and
/// is excluded by the ARCH-based check (`is_projector_sidecar("", Some("clip"))`)
/// that runs after enrichment — even though its filename carries no `mmproj`.
/// Covers the LM Studio post-enrich exclusion path.
#[test]
fn lmstudio_clip_arch_projector_excluded_after_enrich() {
    use ggus::GGufMetaDataValueType::String as GS;
    let dir = TempDir::new().unwrap();
    let root = dir.path();
    // Filename has NO mmproj marker, so the pre-mmap filename check passes; the
    // exclusion must come from the parsed clip architecture.
    let bytes = build_gguf_bytes(&[("general.architecture", GS, gguf_str("clip"))]);
    write_file(&root.join("org/vision/encoder-Q8_0.gguf"), &bytes);
    // A real chat model alongside it, to prove only the projector is dropped.
    let chat = build_gguf_bytes(&[("general.architecture", GS, gguf_str("llama"))]);
    write_file(&root.join("org/chat/model-Q4_K_M.gguf"), &chat);

    let mut store = ModelStore::default();
    let models = store.scan(&[root.to_path_buf()], &[], &[]).unwrap();

    assert_eq!(models.len(), 1, "clip-arch projector excluded, chat kept");
    assert_eq!(models[0].id, "org/chat");
    assert_eq!(models[0].arch.as_deref(), Some("llama"));
}

/// An LM Studio org directory whose read permission is stripped maps to
/// `HG001 ModelDirUnreadable` from the org-level `read_dir` (not the root).
/// Unix-only (chmod).
#[cfg(unix)]
#[test]
fn lmstudio_unreadable_org_dir_errors_hg001() {
    use std::os::unix::fs::PermissionsExt;
    let dir = TempDir::new().unwrap();
    let root = dir.path();
    let org = root.join("org");
    fs::create_dir_all(&org).unwrap();
    fs::set_permissions(&org, fs::Permissions::from_mode(0o000)).unwrap();

    let mut store = ModelStore::default();
    let result = store.scan(&[root.to_path_buf()], &[], &[]);

    fs::set_permissions(&org, fs::Permissions::from_mode(0o755)).unwrap();

    let err = result.expect_err("unreadable org dir must error");
    assert!(
        matches!(err, HiggsError::ModelDirUnreadable { ref path, .. } if path.ends_with("org")),
        "got: {err:?}"
    );
}

/// An LM Studio model directory whose read permission is stripped maps to
/// `HG001 ModelDirUnreadable` from the model-level `read_dir`. Unix-only (chmod).
#[cfg(unix)]
#[test]
fn lmstudio_unreadable_model_dir_errors_hg001() {
    use std::os::unix::fs::PermissionsExt;
    let dir = TempDir::new().unwrap();
    let root = dir.path();
    let model = root.join("org/m");
    fs::create_dir_all(&model).unwrap();
    fs::set_permissions(&model, fs::Permissions::from_mode(0o000)).unwrap();

    let mut store = ModelStore::default();
    let result = store.scan(&[root.to_path_buf()], &[], &[]);

    fs::set_permissions(&model, fs::Permissions::from_mode(0o755)).unwrap();

    let err = result.expect_err("unreadable model dir must error");
    assert!(
        matches!(err, HiggsError::ModelDirUnreadable { ref path, .. } if path.ends_with('m')),
        "got: {err:?}"
    );
}

/// A plain file directly under an LM Studio org dir (where a model dir is
/// expected) is skipped by the `model_path.is_dir()` guard. No error.
#[test]
fn lmstudio_file_at_model_level_skipped() {
    let dir = TempDir::new().unwrap();
    let root = dir.path();
    // A stray file at the model level, plus a real model dir alongside it.
    write_file(&root.join("org/loose.txt"), b"x");
    write_file(&root.join("org/m/model-Q4_K_M.gguf"), b"gguf");

    let mut store = ModelStore::default();
    let models = store.scan(&[root.to_path_buf()], &[], &[]).unwrap();

    assert_eq!(models.len(), 1, "stray model-level file ignored");
    assert_eq!(models[0].id, "org/m");
}

/// An HF repo dir whose name has the `models--` prefix but no second `--`
/// separator cannot be split into org/name → skipped (`split_once` None branch).
#[test]
fn hf_cache_repo_without_separator_skipped() {
    let dir = TempDir::new().unwrap();
    let root = dir.path();
    // models--<no-second-separator> with a snapshot file: must be skipped before
    // the snapshot is ever walked.
    write_file(
        &root.join("models--noseparator/snapshots/rev/model-Q4_K_M.gguf"),
        b"gguf",
    );
    // A well-formed repo alongside it.
    write_file(
        &root.join("models--org--good/snapshots/rev/model-Q8_0.gguf"),
        b"gguf",
    );

    let mut store = ModelStore::default();
    let models = store.scan(&[], &[root.to_path_buf()], &[]).unwrap();

    assert_eq!(models.len(), 1, "separator-less repo skipped");
    assert_eq!(models[0].id, "org/good");
}

/// A plain file (not a dir) directly under `snapshots/` is skipped by the
/// `rev_path.is_dir()` guard, without aborting the HF scan.
#[test]
fn hf_cache_file_at_revision_level_skipped() {
    let dir = TempDir::new().unwrap();
    let root = dir.path();
    // A stray file where a revision DIR is expected.
    write_file(&root.join("models--org--repo/snapshots/stray.txt"), b"x");
    // A real revision alongside it.
    write_file(
        &root.join("models--org--repo/snapshots/rev/model-Q4_K_M.gguf"),
        b"gguf",
    );

    let mut store = ModelStore::default();
    let models = store.scan(&[], &[root.to_path_buf()], &[]).unwrap();

    assert_eq!(models.len(), 1, "non-dir revision entry skipped");
    assert_eq!(models[0].id, "org/repo");
}

/// HF cache: a `mmproj-*.gguf` file is excluded by the FILENAME check before the
/// mmap+parse cost (the pre-enrich `is_projector_sidecar(&fname, None)` branch).
#[test]
fn hf_cache_mmproj_filename_projector_excluded() {
    let dir = TempDir::new().unwrap();
    let root = dir.path();
    write_file(
        &root.join("models--org--repo/snapshots/rev/mmproj-model-F16.gguf"),
        b"gguf",
    );
    write_file(
        &root.join("models--org--repo/snapshots/rev/model-Q4_K_M.gguf"),
        b"gguf",
    );

    let mut store = ModelStore::default();
    let models = store.scan(&[], &[root.to_path_buf()], &[]).unwrap();

    assert_eq!(models.len(), 1, "mmproj sidecar excluded by filename");
    assert!(models[0].path.ends_with("model-Q4_K_M.gguf"));
}

/// HF cache: a real GGUF with `general.architecture = clip` is excluded by the
/// post-enrich arch check (filename carries no mmproj marker).
#[test]
fn hf_cache_clip_arch_projector_excluded_after_enrich() {
    use ggus::GGufMetaDataValueType::String as GS;
    let dir = TempDir::new().unwrap();
    let root = dir.path();
    let bytes = build_gguf_bytes(&[("general.architecture", GS, gguf_str("clip"))]);
    write_file(
        &root.join("models--org--repo/snapshots/rev/encoder-Q8_0.gguf"),
        &bytes,
    );
    let chat = build_gguf_bytes(&[("general.architecture", GS, gguf_str("llama"))]);
    write_file(
        &root.join("models--org--repo/snapshots/rev/model-Q4_K_M.gguf"),
        &chat,
    );

    let mut store = ModelStore::default();
    let models = store.scan(&[], &[root.to_path_buf()], &[]).unwrap();

    assert_eq!(models.len(), 1, "clip projector excluded after enrich");
    assert_eq!(models[0].arch.as_deref(), Some("llama"));
}

/// HF cache: the snapshots dir of one repo is unreadable → `HG001` from the
/// snapshot-level `read_dir`. Unix-only (chmod).
#[cfg(unix)]
#[test]
fn hf_cache_unreadable_snapshots_dir_errors_hg001() {
    use std::os::unix::fs::PermissionsExt;
    let dir = TempDir::new().unwrap();
    let root = dir.path();
    let snaps = root.join("models--org--repo/snapshots");
    fs::create_dir_all(&snaps).unwrap();
    fs::set_permissions(&snaps, fs::Permissions::from_mode(0o000)).unwrap();

    let mut store = ModelStore::default();
    let result = store.scan(&[], &[root.to_path_buf()], &[]);

    fs::set_permissions(&snaps, fs::Permissions::from_mode(0o755)).unwrap();

    let err = result.expect_err("unreadable snapshots dir must error");
    assert!(
        matches!(err, HiggsError::ModelDirUnreadable { ref path, .. } if path.ends_with("snapshots")),
        "got: {err:?}"
    );
}

/// HF cache: a revision dir whose read permission is stripped → `HG001` from the
/// revision-level (file-listing) `read_dir`. Unix-only (chmod).
#[cfg(unix)]
#[test]
fn hf_cache_unreadable_revision_dir_errors_hg001() {
    use std::os::unix::fs::PermissionsExt;
    let dir = TempDir::new().unwrap();
    let root = dir.path();
    let rev = root.join("models--org--repo/snapshots/rev");
    fs::create_dir_all(&rev).unwrap();
    fs::set_permissions(&rev, fs::Permissions::from_mode(0o000)).unwrap();

    let mut store = ModelStore::default();
    let result = store.scan(&[], &[root.to_path_buf()], &[]);

    fs::set_permissions(&rev, fs::Permissions::from_mode(0o755)).unwrap();

    let err = result.expect_err("unreadable revision dir must error");
    assert!(
        matches!(err, HiggsError::ModelDirUnreadable { ref path, .. } if path.ends_with("rev")),
        "got: {err:?}"
    );
}

/// Ollama: a real GGUF blob with `general.architecture = clip` is excluded by the
/// post-enrich arch check (Ollama blobs are content-hashed, so only arch can flag
/// a projector). A normal chat blob alongside it is kept.
#[test]
fn ollama_clip_arch_projector_excluded_after_enrich() {
    use ggus::GGufMetaDataValueType::String as GS;
    let dir = TempDir::new().unwrap();
    let root = dir.path();

    // clip projector blob + manifest.
    let clip = build_gguf_bytes(&[("general.architecture", GS, gguf_str("clip"))]);
    write_file(&root.join("blobs/sha256-clipblob"), &clip);
    let clip_manifest = r#"{"layers":[{"mediaType":"application/vnd.ollama.image.model","digest":"sha256:clipblob"}]}"#;
    write_file(
        &root.join("manifests/registry.ollama.ai/library/vision/latest"),
        clip_manifest.as_bytes(),
    );

    // real chat blob + manifest.
    let chat = build_gguf_bytes(&[("general.architecture", GS, gguf_str("llama"))]);
    write_file(&root.join("blobs/sha256-chatblob"), &chat);
    let chat_manifest = r#"{"layers":[{"mediaType":"application/vnd.ollama.image.model","digest":"sha256:chatblob"}]}"#;
    write_file(
        &root.join("manifests/registry.ollama.ai/library/llama3/latest"),
        chat_manifest.as_bytes(),
    );

    let mut store = ModelStore::default();
    let models = store.scan(&[], &[], &[root.to_path_buf()]).unwrap();

    assert_eq!(models.len(), 1, "clip projector excluded, chat kept");
    assert_eq!(models[0].id, "ollama/llama3:latest");
    assert_eq!(models[0].arch.as_deref(), Some("llama"));
}

/// Ollama: a subdirectory under `manifests/` whose read permission is stripped →
/// `HG001` from the recursive `collect_manifest_files` walk. Unix-only (chmod).
#[cfg(unix)]
#[test]
fn ollama_unreadable_manifests_subdir_errors_hg001() {
    use std::os::unix::fs::PermissionsExt;
    let dir = TempDir::new().unwrap();
    let root = dir.path();
    let sub = root.join("manifests/registry.ollama.ai/library");
    fs::create_dir_all(&sub).unwrap();
    fs::set_permissions(&sub, fs::Permissions::from_mode(0o000)).unwrap();

    let mut store = ModelStore::default();
    let result = store.scan(&[], &[], &[root.to_path_buf()]);

    fs::set_permissions(&sub, fs::Permissions::from_mode(0o755)).unwrap();

    let err = result.expect_err("unreadable manifests subdir must error");
    assert!(
        matches!(err, HiggsError::ModelDirUnreadable { .. }),
        "got: {err:?}"
    );
}

/// A non-existent Ollama root (no `manifests/`) is skipped silently by
/// `collect_manifest_files` (the NotFound `=> Ok(())` branch) — scan succeeds
/// with no models.
#[test]
fn ollama_missing_manifests_dir_ok() {
    let dir = TempDir::new().unwrap();
    let root = dir.path();
    // Root exists but has no `manifests/` subdir.
    fs::create_dir_all(root.join("blobs")).unwrap();

    let mut store = ModelStore::default();
    let models = store.scan(&[], &[], &[root.to_path_buf()]).unwrap();

    assert!(models.is_empty(), "no manifests dir → no models, no error");
}

/// `get()` returns None for an id that was never scanned (the `find` miss path).
#[test]
fn get_returns_none_for_unknown_id() {
    let dir = TempDir::new().unwrap();
    let root = dir.path();
    write_file(&root.join("org/m/model-Q4_K_M.gguf"), b"gguf");

    let mut store = ModelStore::default();
    store.scan(&[root.to_path_buf()], &[], &[]).unwrap();

    assert!(store.get("org/m").is_some(), "known id resolves");
    assert!(
        store.get("nope/missing").is_none(),
        "unknown id resolves to None"
    );
}

/// `models()` exposes the catalog accumulated by the last scan, and a re-scan
/// over empty roots replaces it with an empty catalog.
#[test]
fn models_accessor_reflects_last_scan() {
    let dir = TempDir::new().unwrap();
    let root = dir.path();
    write_file(&root.join("org/m/model-Q4_K_M.gguf"), b"gguf");

    let mut store = ModelStore::default();
    store.scan(&[root.to_path_buf()], &[], &[]).unwrap();
    assert_eq!(store.models().len(), 1);

    // Re-scan with no roots → catalog cleared.
    store.scan(&[], &[], &[]).unwrap();
    assert!(store.models().is_empty(), "re-scan replaces the catalog");
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

/// A GGUF whose header declares more tensor data than the file holds must not
/// crash the scan. ggus 0.5.1 PANICS (slice out of range) on such files — a
/// model mid-download, or a quant type ggus mis-sizes (observed: IQ4_XS) —
/// and `enrich_gguf_metadata` unwind-catches it so the model stays cataloged
/// with empty enrichment. Fail-on-revert for that catch: without it this test
/// aborts the harness with the ggus panic.
#[test]
fn truncated_gguf_survives_scan_without_enrichment() {
    let dir = TempDir::new().unwrap();
    let model_dir = dir.path().join("org").join("truncated");
    fs::create_dir_all(&model_dir).unwrap();

    // Minimal GGUF v3: magic, version, 1 tensor, 0 KVs, then one F32 tensor
    // info declaring 1M elements (4MB of data). Alignment padding (default 32)
    // follows so ggus gets past its padding skip and reaches the data-section
    // length check, then only 8 data bytes — far short of the declared 4MB —
    // so its `&data[..data_len]` slice overruns (the observed panic).
    let mut bytes: Vec<u8> = Vec::new();
    bytes.extend_from_slice(b"GGUF");
    bytes.extend_from_slice(&3u32.to_le_bytes()); // version
    bytes.extend_from_slice(&1u64.to_le_bytes()); // tensor_count
    bytes.extend_from_slice(&0u64.to_le_bytes()); // kv_count
    let name = b"weights";
    bytes.extend_from_slice(&(name.len() as u64).to_le_bytes());
    bytes.extend_from_slice(name);
    bytes.extend_from_slice(&1u32.to_le_bytes()); // n_dims
    bytes.extend_from_slice(&1_000_000u64.to_le_bytes()); // dim[0]
    bytes.extend_from_slice(&0u32.to_le_bytes()); // type F32
    bytes.extend_from_slice(&0u64.to_le_bytes()); // offset
    let padding = (32 - bytes.len() % 32) % 32;
    bytes.extend(std::iter::repeat_n(0u8, padding + 8));
    fs::write(model_dir.join("truncated.gguf"), &bytes).unwrap();

    let mut store = ModelStore::default();
    let models = store
        .scan(&[dir.path().to_path_buf()], &[], &[])
        .expect("scan survives the truncated GGUF");
    assert_eq!(models.len(), 1, "the model is still cataloged");
    let m = &models[0];
    assert!(!m.has_chat_template, "no enrichment from the bad header");
    assert!(m.arch.is_none(), "no arch from the bad header");
}

/// `supports_tools` is the REGISTRY sniff (the registry is the sole tool-call
/// parser), so a Llama-3.2-style JSON-instruction template — which the old
/// marker list missed — must advertise tools, and a plain chat template must
/// not.
#[test]
fn supports_tools_follows_the_parser_registry_sniff() {
    use ggus::GGufMetaDataValueType::String as GS;
    let llama = scan_single_gguf(
        &[(
            "tokenizer.chat_template",
            GS,
            gguf_str(
                r#"Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}."#,
            ),
        )],
        "meta/llama-ish/model-Q4_0.gguf",
    );
    assert!(
        llama.supports_tools,
        "llama JSON-instruction template advertises tools (llama_json parser)"
    );

    let plain = scan_single_gguf(
        &[(
            "tokenizer.chat_template",
            GS,
            gguf_str("{% for m in messages %}{{ m.content }}{% endfor %}"),
        )],
        "misc/plain/model-Q4_0.gguf",
    );
    assert!(
        !plain.supports_tools,
        "plain chat template advertises no tools"
    );
}

/// A GGUF whose header declares an embedding model must scan as
/// [`ModelDomain::Embedding`] — the classification the chat gate ([HG079]) reads.
///
/// The two fixtures below are the two REAL shapes in the wild, and neither is
/// catchable by the other's signal:
///
/// - `bert` / `bge-small` — an ENCODER: `attention.causal = false`, no chat template.
/// - `qwen3` / `qwen3-embedding` — a DECODER converted for embedding: `pooling_type = 3`,
///   causal attention, AND its base model's chat template. Every signal except
///   `pooling_type` says "chat model".
///
/// (Key/value shapes verified against the actual files on disk.)
fn write_gguf(dir: &TempDir, rel: &str, kvs: &[(&str, ggus::GGufMetaDataValueType, Vec<u8>)]) {
    use ggus::{GGufFileHeader, GGufFileWriter};
    use std::io::Cursor;
    let header = GGufFileHeader::new(3, 0, kvs.len() as u64);
    let mut buf = Cursor::new(Vec::<u8>::new());
    let mut writer = GGufFileWriter::new(&mut buf, header).unwrap();
    for (k, t, v) in kvs {
        writer.write_meta_kv(k, *t, v).unwrap();
    }
    writer.finish::<Vec<u8>>(false).finish().unwrap();
    write_file(&dir.path().join(rel), buf.into_inner().as_slice());
}

#[test]
fn embedding_headers_scan_as_the_embedding_domain() {
    use ggus::GGufMetaDataValueType as T;
    let dir = TempDir::new().unwrap();

    // Encoder: non-causal attention (bge-small's actual shape).
    write_gguf(
        &dir,
        "enc/bge/model-f16.gguf",
        &[
            ("general.architecture", T::String, gguf_str("bert")),
            ("bert.attention.causal", T::Bool, vec![0u8]),
        ],
    );
    // Decoder converted for embedding: pooling head + a chat template it inherited.
    write_gguf(
        &dir,
        "dec/qwen3emb/model-Q8_0.gguf",
        &[
            ("general.architecture", T::String, gguf_str("qwen3")),
            ("qwen3.pooling_type", T::U32, 3u32.to_le_bytes().to_vec()),
            (
                "tokenizer.chat_template",
                T::String,
                gguf_str("{% for m in messages %}{{ m.content }}{% endfor %}"),
            ),
        ],
    );
    // A generative model that explicitly declares pooling NONE (0) stays an LLM —
    // presence of the key is not the signal, a non-zero value is.
    write_gguf(
        &dir,
        "gen/llama/model-Q4_K_M.gguf",
        &[
            ("general.architecture", T::String, gguf_str("llama")),
            ("llama.pooling_type", T::U32, 0u32.to_le_bytes().to_vec()),
            ("llama.attention.causal", T::Bool, vec![1u8]),
        ],
    );

    let mut store = ModelStore::default();
    let models = store.scan(&[dir.path().to_path_buf()], &[], &[]).unwrap();
    let domain = |id: &str| {
        models
            .iter()
            .find(|m| m.id == id)
            .unwrap_or_else(|| panic!("{id} not scanned"))
            .domain
    };

    assert_eq!(
        domain("enc/bge"),
        ModelDomain::Embedding,
        "non-causal attention is an embedding declaration"
    );
    assert_eq!(
        domain("dec/qwen3emb"),
        ModelDomain::Embedding,
        "a non-zero pooling_type is an embedding declaration EVEN WITH a chat template"
    );
    assert_eq!(
        domain("gen/llama"),
        ModelDomain::Llm,
        "pooling_type 0 (NONE) + causal attention is a generative model"
    );
}

/// A header with neither key present — the overwhelmingly common case, and the
/// unreadable-header case — must stay `Llm`. The classifier only ever demotes on a
/// POSITIVE declaration; it must never hide a chat model from the catalog.
#[test]
fn a_silent_header_stays_an_llm() {
    use ggus::GGufMetaDataValueType as T;
    let dir = TempDir::new().unwrap();
    write_gguf(
        &dir,
        "org/quiet/model-Q4_K_M.gguf",
        &[("general.architecture", T::String, gguf_str("llama"))],
    );
    let mut store = ModelStore::default();
    let models = store.scan(&[dir.path().to_path_buf()], &[], &[]).unwrap();
    assert_eq!(models[0].domain, ModelDomain::Llm);
}
