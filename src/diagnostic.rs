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

    /// A load was refused because the model's estimated memory need exceeds the
    /// safe headroom over currently-available system RAM. Rejected on the load
    /// path BEFORE spawning a worker (HTTP 503 — capacity, retryable: the user
    /// may free memory or unload another model and retry). The estimate uses the
    /// GGUF file size on disk as a lower-bound proxy for resident weights; the
    /// safe threshold is [`MEMORY_HEADROOM_FRACTION`] of available RAM (ollama's
    /// `freeMemory*80/100` placement rule). `needed_bytes` is the model file
    /// size; `available_bytes` is the system's available RAM at request time.
    ///
    /// [`MEMORY_HEADROOM_FRACTION`]: crate::api::MEMORY_HEADROOM_FRACTION
    #[snafu(display(
        "[HG017] insufficient memory to load {id}: need ~{needed_bytes} bytes, only {available_bytes} available (headroom {headroom_fraction})"
    ))]
    #[diagnostic(code(HG017))]
    InsufficientMemory {
        id: String,
        needed_bytes: u64,
        available_bytes: u64,
        headroom_fraction: f64,
    },

    /// The chat's resolved model is no longer the worker's resident model — a
    /// concurrent JIT load swapped it out between resolution and dispatch (higgs
    /// serves one model at a time, only-keep-last). Detected worker-side, where
    /// the resident id is known: the worker refuses to generate rather than serve
    /// the WRONG model. Surfaced at the chat boundary as HTTP 503 — transient and
    /// retryable (the client's retry re-JITs the requested model). `requested` is
    /// the model the chat resolved against; `resident` is what is loaded now.
    #[snafu(display(
        "[HG018] requested model '{requested}' is no longer resident (now '{resident}') — retry"
    ))]
    #[diagnostic(code(HG018))]
    ResidentModelMismatch { requested: String, resident: String },

    /// The `/v1` inference surface is disabled (server "serving" toggled off).
    /// Rejected at the chat boundary (HTTP 503) before the loaded-model gate, so
    /// no inference runs while serving is off. The `/api/higgs/*` control surface
    /// stays reachable so the user can re-enable serving. Non-fatal and
    /// retryable: a retry after re-enabling succeeds.
    #[snafu(display("[HG019] serving is disabled — enable the server to accept requests"))]
    #[diagnostic(code(HG019))]
    ServingDisabled,

    /// A model-support probe could not run: the transient probe worker failed to
    /// spawn, its stdio closed before replying, or the probe RPC timed out. This
    /// is a probe-infrastructure failure, distinct from a model that loaded and
    /// returned a verbatim engine reason — those are reported as a verdict, not
    /// an error. Surfaced as the probe verdict `(false, Some("<context>"))` so a
    /// failed probe never hangs or panics the support sweep; `context` names the
    /// path or stage that failed.
    #[snafu(display("[HG020] probe worker failed: {context}"))]
    #[diagnostic(code(HG020))]
    ProbeWorkerFailed { context: String },

    /// The transient sysinfo worker could not enumerate devices: it failed to
    /// spawn, its stdio closed before replying, or the M_SYSINFO RPC timed out.
    /// A device-enumeration infrastructure failure — surfaced as an empty device
    /// list so `GET /api/higgs/system` still returns hardware/runtime rather than
    /// failing; `context` names the stage that failed.
    #[snafu(display("[HG021] sysinfo worker failed: {context}"))]
    #[diagnostic(code(HG021))]
    SysinfoWorkerFailed { context: String },

    /// A presented pairing token was expired, already used, or unknown. Non-fatal:
    /// the dialing node is turned away but the hub keeps listening (§7.1).
    #[snafu(display("[HG022] pairing token invalid (expired, used, or unknown): {detail}"))]
    #[diagnostic(code(HG022))]
    PairingTokenInvalid { detail: String },

    /// No protocol version both peers accept. Fatal for this connection: the hub
    /// returns this then closes the stream, telling the node's UI "you must update"
    /// rather than "network broke" (§4.1).
    #[snafu(display("[HG023] no agreed protocol version: peer speaks {peer:?}, we accept {ours:?}"))]
    #[diagnostic(code(HG023), severity(Error))]
    VersionMismatch { peer: Vec<u32>, ours: Vec<u32> },

    /// The peer is not in the allowlist and presented no valid pairing token. The
    /// one path that admits a new id is a valid token; otherwise this. Non-fatal (§7.1).
    #[snafu(display(
        "[HG024] peer {endpoint_id} is not in the allowlist and presented no valid pairing token"
    ))]
    #[diagnostic(code(HG024))]
    NotAllowlisted { endpoint_id: String },

    /// QUIC/TLS completed but the peer sent no HELLO within the deadline — a
    /// pre-auth-DoS guard that bounds half-open admitted connections (§3.2.1). Non-fatal.
    #[snafu(display(
        "[HG028] peer {endpoint_id} completed QUIC but sent no HELLO within {window}s; dropped"
    ))]
    #[diagnostic(code(HG028))]
    HandshakeStalled { endpoint_id: String, window: u64 },

    /// A paired node became unreachable (connection closed, dial failed, or a wedged
    /// worker escalation exhausted) and was retired from the fleet. Non-fatal,
    /// best-effort (§3.4, §3.4.1).
    #[snafu(display("[HG027] node {endpoint_id} unreachable; retired from fleet: {detail}"))]
    #[diagnostic(code(HG027))]
    NodeUnreachable { endpoint_id: String, detail: String },

    /// A peer requested `M_UPDATE` but this build ships only the update *handshake* stub —
    /// no real updater yet (signature-verified self-update is a later task, §9). The
    /// capability is advertised as `false`, so a well-behaved peer never sends it; this is
    /// the typed refusal for one that does anyway.
    #[snafu(display("[HG026] software update not supported by this build: {detail}"))]
    #[diagnostic(code(HG026))]
    UpdateUnsupported { detail: String },

    /// A model download (`M_PULL`) failed — network error, bad repo/file, HTTP status, or a
    /// filesystem error writing into `~/.higgs/models/` (§4 P4b). Never partially exposes a
    /// file: the download writes a temp and renames only on success.
    #[snafu(display("[HG025] model download failed for {repo}/{file}: {detail}"))]
    #[diagnostic(code(HG025))]
    DownloadFailed { repo: String, file: String, detail: String },
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
        assert!(HiggsError::InsufficientMemory {
            id: "org/model".into(),
            needed_bytes: 8_000_000_000,
            available_bytes: 4_000_000_000,
            headroom_fraction: 0.8,
        }
        .to_string()
        .starts_with("[HG017]"));
        assert!(HiggsError::ServingDisabled
            .to_string()
            .starts_with("[HG019]"));
        assert!(HiggsError::ProbeWorkerFailed {
            context: "/models/x.gguf".into(),
        }
        .to_string()
        .starts_with("[HG020]"));
    }

    #[test]
    fn fatal_variants_have_error_severity() {
        use miette::Diagnostic;
        let e = HiggsError::WorkerSpawnFailed {
            source: std::io::Error::new(std::io::ErrorKind::NotFound, "no exe"),
        };
        assert_eq!(e.severity(), Some(miette::Severity::Error));
    }

    #[test]
    fn remote_gate_codes_render() {
        assert!(HiggsError::PairingTokenInvalid { detail: "expired".into() }
            .to_string()
            .starts_with("[HG022]"));
        assert!(HiggsError::NotAllowlisted { endpoint_id: "z32".into() }
            .to_string()
            .starts_with("[HG024]"));
        assert!(HiggsError::HandshakeStalled { endpoint_id: "z32".into(), window: 5 }
            .to_string()
            .starts_with("[HG028]"));
        assert!(HiggsError::NodeUnreachable { endpoint_id: "z32".into(), detail: "closed".into() }
            .to_string()
            .starts_with("[HG027]"));
    }

    #[test]
    fn version_mismatch_is_fatal() {
        use miette::Diagnostic;
        let e = HiggsError::VersionMismatch { peer: vec![2], ours: vec![1] };
        assert!(e.to_string().starts_with("[HG023]"));
        assert_eq!(e.severity(), Some(miette::Severity::Error));
    }
}
