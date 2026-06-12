//! Read-only model discovery across LM Studio, HuggingFace cache, and Ollama
//! stores. Model identity is the HuggingFace repo id (`org/model`) everywhere.
//! Higgs NEVER writes into another app's model storage.

use std::path::{Path, PathBuf};

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
    /// Ollama-sourced models currently use `ollama/{name}:{tag}` (PARKED: pending user decision).
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

        collected.sort_by(|a, b| a.id.cmp(&b.id));
        collected.dedup_by(|a, b| a.id == b.id && a.path == b.path);

        self.models = collected;
        Ok(&self.models)
    }

    /// Return the current model catalog.
    pub fn models(&self) -> &[HiggsModel] {
        &self.models
    }

    /// Look up a model by HuggingFace repo id.
    pub fn get(&self, id: &str) -> Option<&HiggsModel> {
        self.models.iter().find(|m| m.id == id)
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
                out.push(HiggsModel {
                    id: id.clone(),
                    path: file_path.display().to_string(),
                    size_bytes,
                    quant,
                    source: HiggsModelSource::LmStudio,
                });
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
                let size_bytes = file_path.metadata().map(|m| m.len()).unwrap_or(0);
                let quant = quant_from_filename(&fname);
                out.push(HiggsModel {
                    id: id.clone(),
                    path: file_path.display().to_string(),
                    size_bytes,
                    quant,
                    source: HiggsModelSource::HfCache,
                });
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

    // Probe manifests dir directly — NotFound = silently skip, other errors = HG001.
    match std::fs::read_dir(root) {
        Ok(_) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => {
            return Err(HiggsError::ModelDirUnreadable {
                path: root.display().to_string(),
                source: e,
            })
        }
    }

    let blobs_dir = root.join("blobs");
    let mut manifest_files: Vec<PathBuf> = Vec::new();
    collect_manifest_files(&manifests_dir, &mut manifest_files)?;

    for manifest_path in manifest_files {
        // name = parent dir name, tag = file name
        let tag = match manifest_path.file_name().and_then(|n| n.to_str()) {
            Some(n) if !n.is_empty() => n.to_string(),
            _ => continue,
        };
        let name = match manifest_path.parent().and_then(|p| p.file_name()).and_then(|n| n.to_str())
        {
            Some(n) if !n.is_empty() => n.to_string(),
            _ => continue,
        };

        let raw = std::fs::read_to_string(&manifest_path).map_err(|e| {
            HiggsError::OllamaManifestInvalid {
                path: manifest_path.display().to_string(),
                detail: e.to_string(),
            }
        })?;

        let manifest: serde_json::Value =
            serde_json::from_str(&raw).map_err(|e| HiggsError::OllamaManifestInvalid {
                path: manifest_path.display().to_string(),
                detail: format!("json parse: {e}"),
            })?;

        let layers =
            manifest["layers"].as_array().ok_or_else(|| HiggsError::OllamaManifestInvalid {
                path: manifest_path.display().to_string(),
                detail: "missing or non-array `layers`".into(),
            })?;

        let model_layer = layers
            .iter()
            .find(|l| l["mediaType"].as_str() == Some("application/vnd.ollama.image.model"));

        let model_layer =
            model_layer.ok_or_else(|| HiggsError::OllamaManifestInvalid {
                path: manifest_path.display().to_string(),
                detail: "no layer with mediaType application/vnd.ollama.image.model".into(),
            })?;

        let digest =
            model_layer["digest"].as_str().ok_or_else(|| HiggsError::OllamaManifestInvalid {
                path: manifest_path.display().to_string(),
                detail: "model layer missing `digest`".into(),
            })?;

        // digest is "sha256:<hex>"; blob filename is "sha256-<hex>"
        let hex = digest.strip_prefix("sha256:").ok_or_else(|| {
            HiggsError::OllamaManifestInvalid {
                path: manifest_path.display().to_string(),
                detail: format!("unexpected digest format: {digest}"),
            }
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
        match std::fs::File::open(&blob_path).and_then(|mut f| {
            use std::io::Read;
            f.read_exact(&mut magic)
        }) {
            Ok(()) if &magic == b"GGUF" => {}
            _ => continue,
        }

        let size_bytes = blob_path.metadata().map(|m| m.len()).unwrap_or(0);

        // PARKED: identity form pending user decision (spec: HF repo id everywhere)
        let id = format!("ollama/{name}:{tag}");

        out.push(HiggsModel {
            id,
            path: blob_path.display().to_string(),
            size_bytes,
            quant: None,
            source: HiggsModelSource::Ollama,
        });
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
    let stem = name.strip_suffix(".gguf").or_else(|| name.strip_suffix(".GGUF"))?;
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
        write_file(&root.join("blobs/sha256-cafebabe"), &[0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x00]);

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
}
