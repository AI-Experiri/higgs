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
    ModelDirUnreadable {
        path: String,
        source: std::io::Error,
    },

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
    #[snafu(display(
        "[HG005] context overflow: prompt {prompt_tokens} + max_gen {max_gen} > n_ctx {n_ctx}"
    ))]
    #[diagnostic(code(HG005))]
    ContextOverflow {
        prompt_tokens: usize,
        max_gen: usize,
        n_ctx: usize,
    },

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

    /// The worker returned a JSON-RPC error for a request. `worker_code` carries
    /// the worker's own origin diagnostic code (e.g. `"HG005"`) when present, so
    /// the HTTP boundary can map to the true status; `None` falls back to 500.
    #[snafu(display("[HG009] worker error on {method}: {message}"))]
    #[diagnostic(code(HG009))]
    WorkerRpc {
        method: String,
        message: String,
        worker_code: Option<String>,
    },

    /// An Ollama manifest existed but could not be resolved to a GGUF blob.
    #[snafu(display("[HG010] ollama manifest invalid: {path}: {detail}"))]
    #[diagnostic(code(HG010))]
    OllamaManifestInvalid { path: String, detail: String },

    /// A generation-time failure: context creation, prompt decode, sampling, detokenize, or loop decode.
    #[snafu(display("[HG011] generation failed at {stage}: {reason}"))]
    #[diagnostic(code(HG011))]
    GenerationFailed { stage: String, reason: String },

    /// The request's `Host` header is not a trusted loopback host. Rejected at
    /// the serve layer (HTTP 403) as a DNS-rebinding defense — a no-auth
    /// loopback server must not honor requests addressed to an arbitrary
    /// hostname a malicious page may have rebound to `127.0.0.1`.
    #[snafu(display("[HG012] forbidden host: {host}"))]
    #[diagnostic(code(HG012))]
    ForbiddenHost { host: String },

    /// A chat sampling parameter is outside its accepted range. Rejected at the
    /// `/v1/chat/completions` boundary (HTTP 400) BEFORE dispatching to the
    /// worker, so a malformed request never reaches generation. `param` names
    /// the offending field; `detail` states the accepted range. Ranges mirror
    /// vllm `SamplingParams._verify_args`.
    #[snafu(display("[HG013] invalid sampling parameter {param}: {detail}"))]
    #[diagnostic(code(HG013))]
    InvalidSamplingParam { param: String, detail: String },

    /// The inference admission gate is full: too many chat requests are already
    /// in flight. Rejected at the chat boundary (HTTP 503) so the no-auth
    /// loopback server can't be flooded. A capacity signal (vllm/ollama queue
    /// limit), not a permanent failure — the client may retry.
    #[snafu(display(
        "[HG014] server busy: {in_flight} concurrent inference requests in flight (max {max})"
    ))]
    #[diagnostic(code(HG014))]
    ServerBusy { in_flight: usize, max: usize },

    /// A model id failed charset validation or resolved to a path outside every
    /// scanned source directory. Rejected on the load path (HTTP 400) as a
    /// path-traversal guard — a `..`/absolute id must never escape the
    /// read-only scan roots. `id` is the rejected value; `reason` states why.
    #[snafu(display("[HG015] invalid model id {id}: {reason}"))]
    #[diagnostic(code(HG015))]
    InvalidModelId { id: String, reason: String },

    /// A chat/inference RPC exceeded its bounded duration. The worker is alive
    /// but generation did not complete within [`CHAT_RPC_TIMEOUT`]. Surfaced at
    /// the chat boundary as HTTP 504 — the layer that bounds streaming chat
    /// duration (the HTTP layer deliberately does not time the SSE stream).
    ///
    /// [`CHAT_RPC_TIMEOUT`]: crate::supervisor::CHAT_RPC_TIMEOUT
    #[snafu(display("[HG016] chat RPC timed out after {elapsed:?}"))]
    #[diagnostic(code(HG016))]
    ChatTimeout { elapsed: std::time::Duration },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_carries_code() {
        let e = HiggsError::ModelNotFound {
            id: "google/gemma-4-12b".into(),
        };
        assert!(e.to_string().starts_with("[HG002]"));
        assert!(e.to_string().contains("google/gemma-4-12b"));
    }

    #[test]
    fn new_variants_carry_their_codes() {
        assert!(HiggsError::InvalidSamplingParam {
            param: "temperature".into(),
            detail: "x".into(),
        }
        .to_string()
        .starts_with("[HG013]"));
        assert!(HiggsError::ServerBusy {
            in_flight: 8,
            max: 8
        }
        .to_string()
        .starts_with("[HG014]"));
        assert!(HiggsError::InvalidModelId {
            id: "../x".into(),
            reason: "y".into(),
        }
        .to_string()
        .starts_with("[HG015]"));
        assert!(HiggsError::ChatTimeout {
            elapsed: std::time::Duration::from_secs(600),
        }
        .to_string()
        .starts_with("[HG016]"));
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
