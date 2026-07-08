//! Read-only model discovery across LM Studio, HuggingFace cache, and Ollama
//! stores. Model identity is the HuggingFace repo id (`org/model`) everywhere.
//! Higgs NEVER writes into another app's model storage.

use std::io::Read;
use std::path::{Path, PathBuf};

use ggus::{GGuf, GGufMetaMapExt};
use memmap2::MmapOptions;
use serde::{Deserialize, Serialize};

use crate::diagnostic::HiggsError;
use crate::serve::wire::GgufComponent;

higgs_const_enum! {
    /// Where a scanned model file came from.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    pub enum HiggsModelSource {
        LmStudio,
        HfCache,
        Ollama,
    }
}

higgs_ts! {
    /// One discoverable model file on disk.
    #[derive(Debug, Clone, Serialize, Deserialize)]
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
    #[serde(skip_serializing_if = "Option::is_none")]
    #[ts(optional)]
    pub quant: Option<String>,
    /// Which store the file was found in.
    pub source: HiggsModelSource,
    /// Model architecture read from GGUF header (e.g. `"llama"`, `"gemma3"`).
    /// `None` when the header could not be read or the field is absent.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[ts(optional)]
    pub arch: Option<String>,
    /// Training context length (`{arch}.context_length`) from the GGUF header.
    /// `None` when the header could not be read or the field is absent.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[ts(type = "number")]
    #[ts(optional)]
    pub ctx_train: Option<u64>,
    /// Transformer block count (`{arch}.block_count`) from the GGUF header — the
    /// number of layers, used by the autotune KV-cache VRAM estimate. `None` when
    /// the header could not be read or the field is absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[ts(type = "number")]
    #[ts(optional)]
    pub block_count: Option<u32>,
    /// Attention query-head count (`{arch}.attention.head_count`). Used with
    /// `embedding_length` to derive `head_dim` for the KV estimate. `None` when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[ts(type = "number")]
    #[ts(optional)]
    pub head_count: Option<u32>,
    /// Attention KV/GQA head count (`{arch}.attention.head_count_kv`) — the
    /// grouped-query KV head count, which is what sizes the KV cache (NOT the query
    /// `head_count`, which over-estimates KV by the GQA factor). `None` when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[ts(type = "number")]
    #[ts(optional)]
    pub head_count_kv: Option<u32>,
    /// Embedding/hidden size (`{arch}.embedding_length`). With `head_count` gives
    /// `head_dim = embedding_length / head_count`. `None` when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[ts(type = "number")]
    #[ts(optional)]
    pub embedding_length: Option<u32>,
    /// Number of MoE experts (`{arch}.expert_count`); `0`/absent for dense models.
    /// Drives the autotune `cpu_moe` back-off decision. `None` when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[ts(type = "number")]
    #[ts(optional)]
    pub expert_count: Option<u32>,
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
    /// Curated load-relevant GGUF header fields (architecture, quant, tokenizer,
    /// block/head counts, gguf version) for the UI to pin a support mismatch to a
    /// specific component. Empty when the header could not be read. `serde(default)`
    /// so older scan payloads without the field deserialize as empty.
    #[serde(default)]
    pub gguf_components: Vec<GgufComponent>,
    /// A coded diagnostic set when GGUF-header enrichment FAILED — the file was
    /// unreadable/un-mmappable, its header was malformed, or the `ggus` parse
    /// panicked (a truncated file mid-download, an unsupported quant, or a header
    /// missing `general.architecture`). Carries the rendered `[HG070]` message so
    /// the UI can explain why this model's header fields are blank instead of
    /// treating it as a genuinely sparse model. `None` when enrichment succeeded;
    /// `serde(default)` so older scan payloads without the field deserialize as
    /// `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[ts(optional)]
    pub enrich_error: Option<String>,
    /// The embedded chat-template string, captured transiently for the host-side
    /// Gate-2 tool-call-parser sniff. NOT on the wire (`serde(skip)`): it never
    /// leaves the host, and stamping it into every model payload would bloat the
    /// scan response with multi-KB templates the frontend never reads.
    #[serde(skip)]
    pub chat_template: Option<String>,
    }
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
            block_count: None,
            head_count: None,
            head_count_kv: None,
            embedding_length: None,
            expert_count: None,
            has_chat_template: false,
            supports_tools: false,
            supports_reasoning: false,
            gguf_components: Vec::new(),
            enrich_error: None,
            chat_template: None,
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

/// Render the `[HG070]` enrichment diagnostic stamped onto a model whose GGUF
/// header could not be fully read (see [`HiggsError::GgufEnrichFailed`]).
fn enrich_err(path: &str, reason: impl Into<String>) -> String {
    HiggsError::GgufEnrichFailed {
        path: path.to_string(),
        reason: reason.into(),
    }
    .to_string()
}

/// Read GGUF header metadata from `path` and fill in the enrichment fields of
/// `model`. A corrupt or unreadable header leaves the fields at `None`/`false`
/// — the model stays cataloged, with a coded `[HG070]` diagnostic on
/// `model.enrich_error` explaining the failure (see [`enrich_err`]).
fn enrich_gguf_metadata(model: &mut HiggsModel) {
    let file = match std::fs::File::open(&model.path) {
        Ok(f) => f,
        Err(e) => {
            model.enrich_error = Some(enrich_err(&model.path, format!("unreadable: {e}")));
            return;
        }
    };
    // SAFETY: standard read-only mmap; we do not mutate the mapping.
    let mmap = match unsafe { MmapOptions::new().map(&file) } {
        Ok(m) => m,
        Err(e) => {
            model.enrich_error = Some(enrich_err(&model.path, format!("mmap failed: {e}")));
            return;
        }
    };
    // ggus 0.5.1 PANICS (not errors) on inputs it dislikes: `GGuf::new` slices
    // out of range on a TRUNCATED file (a model mid-download in a watched
    // LM-Studio dir) or a quant type whose block size it mis-sizes (observed:
    // bartowski IQ4_XS), and its GETTERS unwrap internally (e.g.
    // `llm_context_length()` unwraps `general_architecture()` — a GGUF without
    // `general.architecture` panics). A single such file must not crash the
    // whole scan, so the ENTIRE enrichment is unwind-caught; the model stays
    // cataloged with whatever fields were set before the panic. ggus types are
    // not UnwindSafe (interior refs), which AssertUnwindSafe waives — sound
    // here because the closure only writes plain field values into `model`.
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        enrich_from_gguf(model, &mmap);
    }));
    if outcome.is_err() {
        model.enrich_error = Some(enrich_err(
            &model.path,
            "header parse panicked (truncated file, unsupported quant, or missing keys)",
        ));
        tracing::warn!(
            path = %model.path,
            "GGUF header parse panicked (truncated file, unsupported quant, or missing keys); model cataloged with partial enrichment"
        );
    }
}

/// The panicky part of [`enrich_gguf_metadata`]: every `ggus` call lives here,
/// inside the caller's `catch_unwind`.
fn enrich_from_gguf(model: &mut HiggsModel, mmap: &memmap2::Mmap) {
    let Ok(gguf) = GGuf::new(mmap) else {
        model.enrich_error = Some(enrich_err(&model.path, "malformed GGUF header"));
        return;
    };
    let arch = gguf.general_architecture().ok().map(ToString::to_string);
    model.arch = arch.clone();
    // NOT ggus's `llm_context_length()` — it UNWRAPS `general_architecture()`
    // internally, panicking on an arch-less GGUF and aborting the rest of the
    // enrichment (the catch_unwind above turns that into partial fields).
    // Read the arch-scoped key directly instead.
    model.ctx_train = arch.as_deref().and_then(|a| {
        gguf.get_usize(&format!("{a}.context_length"))
            .ok()
            .map(|n| n as u64)
    });
    // Typed tuning fields (arch-scoped). Read the GQA KV head count
    // (`attention.head_count_kv`) — the KV-cache size driver — NOT the query
    // `head_count`, which over-estimates KV by the GQA factor.
    if let Some(a) = arch.as_deref() {
        let read_u32 = |suffix: &str| {
            gguf.get_usize(&format!("{a}.{suffix}"))
                .ok()
                .map(|n| n as u32)
        };
        model.block_count = read_u32("block_count");
        model.head_count = read_u32("attention.head_count");
        model.head_count_kv = read_u32("attention.head_count_kv");
        model.embedding_length = read_u32("embedding_length");
        model.expert_count = read_u32("expert_count");
    }
    // Read the embedded chat template once and derive capabilities from it
    // (the template is the GGUF's own declaration of how it talks).
    let template = gguf.tokenizer_chat_template().ok();
    model.has_chat_template = template.is_some();
    if let Some(t) = template {
        // `supports_tools` means "higgs can serve tool calls for this model".
        // llama.cpp's PEG auto-parser derives a tool parser from ANY jinja
        // template, so the capability signal is the template itself declaring
        // tool/function handling — the same heuristic class as
        // `supports_reasoning` below, no curated parser list.
        let tl = t.to_lowercase();
        model.supports_tools = tl.contains("tool") || tl.contains("function");
        model.supports_reasoning = t.contains("</think>")
            || t.contains("<think>")
            || t.to_lowercase().contains("thinking");
        // Capture the template for the host-side Gate-2 parser sniff (not on the wire).
        model.chat_template = Some(t.to_string());
    }
    model.gguf_components = curated_components(&gguf, arch.as_deref());
}

/// Build the curated list of LOAD-RELEVANT GGUF header fields for the UI.
///
/// Only keys that bear on whether the engine can load and serve the model are
/// surfaced — giant arrays (token lists, merges) are deliberately skipped so the
/// scan payload stays small. Missing keys are simply omitted. `arch` (when known)
/// prefixes the architecture-scoped keys (`{arch}.context_length`, etc.).
fn curated_components(gguf: &GGuf, arch: Option<&str>) -> Vec<GgufComponent> {
    let mut out: Vec<GgufComponent> = Vec::new();
    let mut push = |key: &str, value: String| {
        out.push(GgufComponent {
            key: key.into(),
            value,
        })
    };

    // gguf container version (from the file header — not a metadata KV).
    push("gguf.version", gguf.header.version.to_string());

    // general.* — architecture and quantization shape.
    if let Ok(v) = gguf.general_architecture() {
        push("general.architecture", v.to_string());
    }
    // general.file_type / general.filetype is the quant enum; render its name.
    if let Ok(ft) = gguf.general_filetype() {
        // GGufFileType has no name(); its Debug form is the canonical quant label
        // (e.g. `MostlyQ4_K_M`).
        push("general.file_type", format!("{ft:?}"));
    }
    if let Ok(v) = gguf.general_quantization_version() {
        push("general.quantization_version", v.to_string());
    }

    // tokenizer.* — load-relevant tokenizer identity (scalars only).
    if let Ok(v) = gguf.get_str("tokenizer.ggml.model") {
        push("tokenizer.ggml.model", v.to_string());
    }
    if let Ok(v) = gguf.get_str("tokenizer.ggml.pre") {
        push("tokenizer.ggml.pre", v.to_string());
    }

    // {arch}.* — architecture-scoped scalars that shape the load.
    if let Some(a) = arch {
        if let Ok(v) = gguf.get_usize(&format!("{a}.context_length")) {
            push(&format!("{a}.context_length"), v.to_string());
        }
        if let Ok(v) = gguf.get_usize(&format!("{a}.block_count")) {
            push(&format!("{a}.block_count"), v.to_string());
        }
        if let Ok(v) = gguf.get_usize(&format!("{a}.attention.head_count")) {
            push(&format!("{a}.attention.head_count"), v.to_string());
        }
    }

    out
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
                // Filename-based projector exclusion BEFORE the mmap+parse cost.
                if is_projector_sidecar(&fname, None) {
                    continue;
                }
                enrich_gguf_metadata(&mut model);
                // Arch-based exclusion needs the parsed `general.architecture`.
                if is_projector_sidecar("", model.arch.as_deref()) {
                    continue;
                }
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
                // Filename-based projector exclusion BEFORE the mmap+parse cost.
                if is_projector_sidecar(&fname, None) {
                    continue;
                }
                enrich_gguf_metadata(&mut model);
                // Arch-based exclusion needs the parsed `general.architecture`.
                if is_projector_sidecar("", model.arch.as_deref()) {
                    continue;
                }
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
        // Ollama blobs are content-hashed (no filename) — detect projectors by arch.
        if is_projector_sidecar("", model.arch.as_deref()) {
            continue;
        }
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

/// True when a scanned GGUF is a multimodal **projector sidecar** (a vision/audio
/// encoder shipped alongside a chat model), not a loadable chat model.
///
/// llama.cpp names these `mmproj-*.gguf` and their GGUF `general.architecture` is
/// `"clip"`. They share their parent repo's id (`{org}/{model}`), so listing them
/// as standalone models both collides ids and offers a non-servable row. They are
/// excluded from scan results; the projector is consumed implicitly when the
/// parent multimodal model is loaded.
fn is_projector_sidecar(fname: &str, arch: Option<&str>) -> bool {
    arch == Some("clip") || fname.to_ascii_lowercase().contains("mmproj")
}

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
#[path = "models_tests.rs"]
mod tests;
