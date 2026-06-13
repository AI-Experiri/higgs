//! Higgs diagnostics. Standalone snafu+miette infra (no jigglebot imports);
//! follows the project's four-pillar conventions by style: code baked into
//! Display, origin-only logging, severity for fatality, append-only codes.

use snafu::Snafu;

/// All higgs failures. Codes are append-only; never renumber.
#[derive(Debug, Snafu, miette::Diagnostic)]
#[non_exhaustive]
pub enum HiggsError {
    /// A configured model directory could not be read during scan.
    #[snafu(display("[HG001] model dir unreadable: {path}: {source}"))]
    #[diagnostic(code(HG001))]
    ModelDirUnreadable { path: String, source: std::io::Error },

    /// Requested model id is not present in any scanned source.
    #[snafu(display("[HG002] model not found on disk: {id}"))]
    #[diagnostic(code(HG002))]
    ModelNotFound { id: String },

    /// Chat requested for a model that is not loaded (no JIT in v1).
    #[snafu(display("[HG003] model not loaded: {id} — load it explicitly first"))]
    #[diagnostic(code(HG003))]
    ModelNotLoaded { id: String },

    /// llama.cpp failed to load the model file.
    #[snafu(display("[HG004] engine failed to load {id}: {reason}"))]
    #[diagnostic(code(HG004), severity(Error))]
    EngineLoadFailed { id: String, reason: String },

    /// Prompt + max generation tokens exceed the context window.
    #[snafu(display("[HG005] context overflow: prompt {prompt_tokens} + max_gen {max_gen} > n_ctx {n_ctx}"))]
    #[diagnostic(code(HG005))]
    ContextOverflow { prompt_tokens: usize, max_gen: usize, n_ctx: usize },

    /// The worker process could not be spawned.
    #[snafu(display("[HG006] worker spawn failed: {source}"))]
    #[diagnostic(code(HG006), severity(Error))]
    WorkerSpawnFailed { source: std::io::Error },

    /// The worker died or its stdio closed mid-request.
    #[snafu(display("[HG007] worker unavailable: {context}"))]
    #[diagnostic(code(HG007))]
    WorkerDead { context: String },

    /// A JSON-RPC frame failed to encode/decode.
    #[snafu(display("[HG008] rpc decode failed: {detail}"))]
    #[diagnostic(code(HG008))]
    RpcDecode { detail: String },

    /// The worker returned a JSON-RPC error for a request.
    #[snafu(display("[HG009] worker error on {method}: {message}"))]
    #[diagnostic(code(HG009))]
    WorkerRpc { method: String, message: String },

    /// An Ollama manifest existed but could not be resolved to a GGUF blob.
    #[snafu(display("[HG010] ollama manifest invalid: {path}: {detail}"))]
    #[diagnostic(code(HG010))]
    OllamaManifestInvalid { path: String, detail: String },

    /// A generation-time failure: context creation, prompt decode, sampling, detokenize, or loop decode.
    #[snafu(display("[HG011] generation failed at {stage}: {reason}"))]
    #[diagnostic(code(HG011))]
    GenerationFailed { stage: String, reason: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_carries_code() {
        let e = HiggsError::ModelNotFound { id: "google/gemma-4-12b".into() };
        assert!(e.to_string().starts_with("[HG002]"));
        assert!(e.to_string().contains("google/gemma-4-12b"));
    }

    #[test]
    fn fatal_variants_have_error_severity() {
        use miette::Diagnostic;
        let e = HiggsError::WorkerSpawnFailed {
            source: std::io::Error::new(std::io::ErrorKind::NotFound, "no exe"),
        };
        assert_eq!(e.severity(), Some(miette::Severity::Error));
    }
}
