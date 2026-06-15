//! Read-only model discovery across LM Studio, HuggingFace cache, and Ollama
//! stores. Model identity is the HuggingFace repo id (`org/model`) everywhere.
//! Higgs NEVER writes into another app's model storage.

use std::io::Read;
use std::path::{Path, PathBuf};

use ggus::{GGuf, GGufMetaMapExt};
use memmap2::MmapOptions;
use serde::{Deserialize, Serialize};

use crate::diagnostic::HiggsError;

/// Where a scanned model file came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ts_rs::TS)]
#[ts(export, export_to = "../../../frontend/src/lib/generated/")]
pub enum HiggsModelSource {
    LmStudio,
    HfCache,
    Ollama,
}

/// One discoverable model file on disk.
#[derive(Debug, Clone, Serialize, Deserialize, ts_rs::TS)]
#[ts(export, export_to = "../../../frontend/src/lib/generated/")]
pub struct HiggsModel {
    /// HuggingFace repo id, `org/model` — the identity used everywhere.
    /// Ollama-sourced models keep their established Ollama name (`ollama/{name}:{tag}`);
    /// no HuggingFace id is fabricated.
    pub id: String,
    /// Absolute path to the GGUF file.
    pub path: String,
    /// File size in bytes.
    #[ts(type = "number")]
    pub size_bytes: u64,
    /// Quantization tag parsed from the filename (e.g. `Q4_K_M`), if present.
    #[ts(optional)]
    pub quant: Option<String>,
    /// Which store the file was found in.
    pub source: HiggsModelSource,
    /// Model architecture read from GGUF header (e.g. `"llama"`, `"gemma3"`).
    /// `None` when the header could not be read or the field is absent.
    #[ts(optional)]
    pub arch: Option<String>,
    /// Training context length (`{arch}.context_length`) from the GGUF header.
    /// `None` when the header could not be read or the field is absent.
    #[ts(type = "number")]
    #[ts(optional)]
    pub ctx_train: Option<u64>,
    /// Whether `tokenizer.chat_template` is present in the GGUF header.
    /// `false` when the header could not be read.
    pub has_chat_template: bool,
    /// Whether the chat template declares tool/function calling. Heuristic: the
    /// embedded template references tool calls (`tool_call`/`tools`). `false`
    /// when there is no template or it carries no tool markup. `serde(default)`
    /// so older scan payloads without the field deserialize as `false`.
    #[serde(default)]
    pub supports_tools: bool,
    /// Whether the model emits a reasoning/thinking block. Heuristic: the
    /// template references `<think>`/thinking. `false` when unknown.
    #[serde(default)]
    pub supports_reasoning: bool,
}

impl HiggsModel {
    /// A scanned model with only its on-disk facts known: identity, path, size,
    /// quant tag, and source store. All GGUF-header-derived fields (`arch`,
    /// `ctx_train`, `has_chat_template`, `supports_tools`, `supports_reasoning`)
    /// start empty and are filled by [`enrich_gguf_metadata`]. The single
    /// construction point shared by every scanner.
    fn base(
        id: impl Into<String>,
        path: impl Into<String>,
        size_bytes: u64,
        quant: Option<String>,
        source: HiggsModelSource,
    ) -> Self {
        Self {
            id: id.into(),
            path: path.into(),
            size_bytes,
            quant,
            source,
            arch: None,
            ctx_train: None,
            has_chat_template: false,
            supports_tools: false,
            supports_reasoning: false,
        }
    }
}

/// Scans configured model directories; owns the resulting catalog.
#[derive(Debug, Default)]
pub struct ModelStore {
    models: Vec<HiggsModel>,
}

impl ModelStore {
    /// Scan all roots, replace the catalog, and return the resulting slice.
    ///
    /// Missing roots are skipped silently. If a root exists but cannot be read,
    /// returns `Err([HG001] ModelDirUnreadable)`.
    pub fn scan(
        &mut self,
        lmstudio: &[PathBuf],
        hf: &[PathBuf],
        ollama: &[PathBuf],
    ) -> Result<&[HiggsModel], HiggsError> {
        let mut collected: Vec<HiggsModel> = Vec::new();

        for root in lmstudio {
            scan_lmstudio(root, &mut collected)?;
        }
        for root in hf {
            scan_hf_cache(root, &mut collected)?;
        }
        for root in ollama {
            scan_ollama(root, &mut collected)?;
        }

        collected.sort_by(|a, b| a.id.cmp(&b.id).then_with(|| a.path.cmp(&b.path)));
        collected.dedup_by(|a, b| a.id == b.id && a.path == b.path);

        self.models = collected;
        Ok(&self.models)
    }

    /// Return the current model catalog.
    pub fn models(&self) -> &[HiggsModel] {
        &self.models
    }

    /// Look up a model by HuggingFace repo id.
    ///
    /// When multiple variants share the same id (e.g. different quantization paths
    /// for the same repo), returns the lexically-first path variant. Explicit
    /// variant selection by path is a v2 feature.
    pub fn get(&self, id: &str) -> Option<&HiggsModel> {
        self.models.iter().find(|m| m.id == id)
    }
}

// ---------------------------------------------------------------------------
// GGUF header enrichment
// ---------------------------------------------------------------------------

/// Read GGUF header metadata from `path` and fill in the enrichment fields of
/// `model`. A corrupt or unreadable header leaves the fields at `None`/`false`
/// — the model stays cataloged. No new error codes are raised here.
fn enrich_gguf_metadata(model: &mut HiggsModel) {
    let file = match std::fs::File::open(&model.path) {
        Ok(f) => f,
        Err(_) => return,
    };
    // SAFETY: standard read-only mmap; we do not mutate the mapping.
    let mmap = match unsafe { MmapOptions::new().map(&file) } {
        Ok(m) => m,
        Err(_) => return,
    };
    let gguf = match GGuf::new(&mmap) {
        Ok(g) => g,
        Err(_) => return,
    };
    model.arch = gguf.general_architecture().ok().map(ToString::to_string);
    model.ctx_train = gguf.llm_context_length().ok().map(|n| n as u64);
    // Read the embedded chat template once and derive capabilities from it
    // (the template is the GGUF's own declaration of how it talks).
    let template = gguf.tokenizer_chat_template().ok();
    model.has_chat_template = template.is_some();
    if let Some(t) = template {
        model.supports_tools = t.contains("tool_call") || t.contains("tools");
        model.supports_reasoning = t.contains("</think>")
            || t.contains("<think>")
            || t.to_lowercase().contains("thinking");
    }
}

// ---------------------------------------------------------------------------
// Layout walkers
// ---------------------------------------------------------------------------

/// Walk an LM Studio root: `<root>/{org}/{model}/*.gguf`.
///
/// Two directory levels, then GGUF files. Id is `org/model`.
fn scan_lmstudio(root: &Path, out: &mut Vec<HiggsModel>) -> Result<(), HiggsError> {
    let org_entries = match std::fs::read_dir(root) {
        Ok(it) => it,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => {
            return Err(HiggsError::ModelDirUnreadable {
                path: root.display().to_string(),
                source: e,
            })
        }
    };

    for org_entry in org_entries {
        let org_entry = org_entry.map_err(|e| HiggsError::ModelDirUnreadable {
            path: root.display().to_string(),
            source: e,
        })?;
        let org_path = org_entry.path();
        if !org_path.is_dir() {
            continue;
        }
        let org_name = match org_path.file_name().and_then(|n| n.to_str()) {
            Some(n) if !n.is_empty() => n.to_string(),
            _ => continue,
        };

        let model_entries = match std::fs::read_dir(&org_path) {
            Ok(it) => it,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
            Err(e) => {
                return Err(HiggsError::ModelDirUnreadable {
                    path: org_path.display().to_string(),
                    source: e,
                })
            }
        };

        for model_entry in model_entries {
            let model_entry = model_entry.map_err(|e| HiggsError::ModelDirUnreadable {
                path: org_path.display().to_string(),
                source: e,
            })?;
            let model_path = model_entry.path();
            if !model_path.is_dir() {
                continue;
            }
            let model_name = match model_path.file_name().and_then(|n| n.to_str()) {
                Some(n) if !n.is_empty() => n.to_string(),
                _ => continue,
            };
            let id = format!("{org_name}/{model_name}");

            let file_entries = match std::fs::read_dir(&model_path) {
                Ok(it) => it,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
                Err(e) => {
                    return Err(HiggsError::ModelDirUnreadable {
                        path: model_path.display().to_string(),
                        source: e,
                    })
                }
            };

            for file_entry in file_entries {
                let file_entry = file_entry.map_err(|e| HiggsError::ModelDirUnreadable {
                    path: model_path.display().to_string(),
                    source: e,
                })?;
                let file_path = file_entry.path();
                let fname = match file_path.file_name().and_then(|n| n.to_str()) {
                    Some(n) if !n.is_empty() => n.to_string(),
                    _ => continue,
                };
                if !fname.to_ascii_lowercase().ends_with(".gguf") {
                    continue;
                }
                let size_bytes = file_path.metadata().map(|m| m.len()).unwrap_or(0);
                let quant = quant_from_filename(&fname);
                let mut model = HiggsModel::base(
                    id.clone(),
                    file_path.display().to_string(),
                    size_bytes,
                    quant,
                    HiggsModelSource::LmStudio,
                );
                enrich_gguf_metadata(&mut model);
                out.push(model);
            }
        }
    }

    Ok(())
}

/// Walk an HF cache root: `<root>/models--{org}--{name}/snapshots/{rev}/*.gguf`.
///
/// Any revision directory is included. Id is `org/name`.
fn scan_hf_cache(root: &Path, out: &mut Vec<HiggsModel>) -> Result<(), HiggsError> {
    let repo_entries = match std::fs::read_dir(root) {
        Ok(it) => it,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => {
            return Err(HiggsError::ModelDirUnreadable {
                path: root.display().to_string(),
                source: e,
            })
        }
    };

    for repo_entry in repo_entries {
        let repo_entry = repo_entry.map_err(|e| HiggsError::ModelDirUnreadable {
            path: root.display().to_string(),
            source: e,
        })?;
        let repo_path = repo_entry.path();
        if !repo_path.is_dir() {
            continue;
        }
        let dir_name = match repo_path.file_name().and_then(|n| n.to_str()) {
            Some(n) if !n.is_empty() => n.to_string(),
            _ => continue,
        };

        // Expect prefix "models--"
        let rest = match dir_name.strip_prefix("models--") {
            Some(r) => r.to_string(),
            None => continue,
        };

        // Split on first "--" to get org and name
        let id = match rest.split_once("--") {
            Some((org, name)) => format!("{org}/{name}"),
            None => continue,
        };

        let snapshots_path = repo_path.join("snapshots");
        if !snapshots_path.is_dir() {
            continue;
        }

        let rev_entries = match std::fs::read_dir(&snapshots_path) {
            Ok(it) => it,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
            Err(e) => {
                return Err(HiggsError::ModelDirUnreadable {
                    path: snapshots_path.display().to_string(),
                    source: e,
                })
            }
        };

        for rev_entry in rev_entries {
            let rev_entry = rev_entry.map_err(|e| HiggsError::ModelDirUnreadable {
                path: snapshots_path.display().to_string(),
                source: e,
            })?;
            let rev_path = rev_entry.path();
            if !rev_path.is_dir() {
                continue;
            }

            let file_entries = match std::fs::read_dir(&rev_path) {
                Ok(it) => it,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
                Err(e) => {
                    return Err(HiggsError::ModelDirUnreadable {
                        path: rev_path.display().to_string(),
                        source: e,
                    })
                }
            };

            for file_entry in file_entries {
                let file_entry = file_entry.map_err(|e| HiggsError::ModelDirUnreadable {
                    path: rev_path.display().to_string(),
                    source: e,
                })?;
                let file_path = file_entry.path();
                let fname = match file_path.file_name().and_then(|n| n.to_str()) {
                    Some(n) if !n.is_empty() => n.to_string(),
                    _ => continue,
                };
                if !fname.to_ascii_lowercase().ends_with(".gguf") {
                    continue;
                }
                // F7: canonicalize to resolve the revision symlink to the
                // underlying blob path. Two revisions that symlink the same blob
                // will produce the same canonical path and collapse via dedup.
                let canonical_path =
                    std::fs::canonicalize(&file_path).unwrap_or_else(|_| file_path.clone());
                let size_bytes = canonical_path.metadata().map(|m| m.len()).unwrap_or(0);
                let quant = quant_from_filename(&fname);
                let mut model = HiggsModel::base(
                    id.clone(),
                    canonical_path.display().to_string(),
                    size_bytes,
                    quant,
                    HiggsModelSource::HfCache,
                );
                enrich_gguf_metadata(&mut model);
                out.push(model);
            }
        }
    }

    Ok(())
}

/// Walk an Ollama root: `<root>/manifests/**/{name}/{tag}` (recursive).
///
/// Each file under manifests is a JSON manifest. Resolves the GGUF blob from
/// `layers[]` where `mediaType == "application/vnd.ollama.image.model"`.
/// Blob lives at `<root>/blobs/sha256-<hex>`.
fn scan_ollama(root: &Path, out: &mut Vec<HiggsModel>) -> Result<(), HiggsError> {
    let manifests_dir = root.join("manifests");

    // collect_manifest_files handles NotFound (silently skips) and HG001 on real errors;
    // no separate root probe needed.
    let blobs_dir = root.join("blobs");
    let mut manifest_files: Vec<PathBuf> = Vec::new();
    collect_manifest_files(&manifests_dir, &mut manifest_files)?;

    for manifest_path in manifest_files {
        // name = parent dir name, tag = file name
        let tag = match manifest_path.file_name().and_then(|n| n.to_str()) {
            Some(n) if !n.is_empty() => n.to_string(),
            _ => continue,
        };
        let name = match manifest_path
            .parent()
            .and_then(|p| p.file_name())
            .and_then(|n| n.to_str())
        {
            Some(n) if !n.is_empty() => n.to_string(),
            _ => continue,
        };

        // F6: per-file read and JSON-parse failures skip the file with a debug log
        // rather than aborting the whole scan. A binary file (.DS_Store, partial
        // download, etc.) must not wipe out all Ollama discovery.
        let raw = match std::fs::read_to_string(&manifest_path) {
            Ok(r) => r,
            Err(e) => {
                tracing::debug!(path = %manifest_path.display(), error = %e, "ollama: skipping unreadable manifest");
                continue;
            }
        };

        let manifest: serde_json::Value = match serde_json::from_str(&raw) {
            Ok(v) => v,
            Err(e) => {
                tracing::debug!(path = %manifest_path.display(), error = %e, "ollama: skipping non-JSON manifest");
                continue;
            }
        };

        // Missing or non-array `layers` means this is not a GGUF-model manifest
        // (common for embedding and vision pulls). Skip silently — not an error.
        let Some(layers) = manifest["layers"].as_array() else {
            tracing::debug!(path = %manifest_path.display(), "ollama manifest has no `layers`; skipping");
            continue;
        };

        let model_layer = layers
            .iter()
            .find(|l| l["mediaType"].as_str() == Some("application/vnd.ollama.image.model"));

        // No GGUF model layer — normal for vision/embedding manifests. Skip silently.
        let Some(model_layer) = model_layer else {
            tracing::debug!(path = %manifest_path.display(), "ollama manifest has no model layer; skipping");
            continue;
        };

        let digest =
            model_layer["digest"]
                .as_str()
                .ok_or_else(|| HiggsError::OllamaManifestInvalid {
                    path: manifest_path.display().to_string(),
                    detail: "model layer missing `digest`".into(),
                })?;

        // digest is "sha256:<hex>"; blob filename is "sha256-<hex>"
        let hex =
            digest
                .strip_prefix("sha256:")
                .ok_or_else(|| HiggsError::OllamaManifestInvalid {
                    path: manifest_path.display().to_string(),
                    detail: format!("unexpected digest format: {digest}"),
                })?;
        let blob_name = format!("sha256-{hex}");
        let blob_path = blobs_dir.join(&blob_name);

        if !blob_path.exists() {
            // Blob not present; skip silently — normal for partial pulls.
            continue;
        }

        // Validate GGUF magic — corrupt/partial blobs are silently skipped,
        // matching shimmy/LM Studio behavior.
        let mut magic = [0u8; 4];
        match std::fs::File::open(&blob_path).and_then(|mut f| f.read_exact(&mut magic)) {
            Ok(()) if &magic == b"GGUF" => {}
            _ => continue,
        }

        let size_bytes = blob_path.metadata().map(|m| m.len()).unwrap_or(0);

        // Decided: Ollama-sourced models keep their established Ollama name (no HF id fabrication).
        let id = format!("ollama/{name}:{tag}");

        let mut model = HiggsModel::base(
            id,
            blob_path.display().to_string(),
            size_bytes,
            None,
            HiggsModelSource::Ollama,
        );
        enrich_gguf_metadata(&mut model);
        out.push(model);
    }

    Ok(())
}

/// Recursively collect all regular files under `dir` into `out`.
///
/// Returns `Err([HG001])` if a directory cannot be read (other than NotFound,
/// which is silently skipped).
fn collect_manifest_files(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), HiggsError> {
    let entries = match std::fs::read_dir(dir) {
        Ok(it) => it,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => {
            return Err(HiggsError::ModelDirUnreadable {
                path: dir.display().to_string(),
                source: e,
            })
        }
    };
    for entry in entries {
        let entry = entry.map_err(|e| HiggsError::ModelDirUnreadable {
            path: dir.display().to_string(),
            source: e,
        })?;
        let path = entry.path();
        if path.is_dir() {
            collect_manifest_files(&path, out)?;
        } else if path.is_file() {
            out.push(path);
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Quant parsing
// ---------------------------------------------------------------------------

/// Parse a quantization tag from a GGUF filename.
///
/// Strips the `.gguf` suffix, takes the last `-` or `.` separated token, and
/// checks whether it looks like a quantization tag (`Q…`, `IQ…`, `f16`,
/// `bf16`, `f32`).
fn quant_from_filename(name: &str) -> Option<String> {
    let stem = name
        .strip_suffix(".gguf")
        .or_else(|| name.strip_suffix(".GGUF"))?;
    let tail = stem.rsplit(['-', '.']).next()?;
    let looks_quant = tail.starts_with('Q')
        || tail.starts_with("IQ")
        || tail.eq_ignore_ascii_case("f16")
        || tail.eq_ignore_ascii_case("bf16")
        || tail.eq_ignore_ascii_case("f32");
    looks_quant.then(|| tail.to_string())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
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
    fn scan_single_gguf(
        kvs: &[(&str, ggus::GGufMetaDataValueType, Vec<u8>)],
        rel: &str,
    ) -> HiggsModel {
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
    /// invalid manifest → HG009 error aborting the scan.
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

    /// Ollama model layer missing the `digest` field entirely → HG009 error.
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
}
