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

    /// Prompt + max generation tokens exceed the loaded context window. The lever is
    /// `ctx_len` (the window the model was loaded with); raising it helps up to the
    /// model's TRAINED context, beyond which the prompt itself must shrink. When
    /// `autotune` is on, a re-tune derives the largest fitting context automatically.
    #[snafu(display(
        "[HG005] context overflow: prompt {prompt_tokens} + max_gen {max_gen} tokens exceed the loaded context window n_ctx={n_ctx} — reload the model with a larger ctx_len (up to its trained context), or lower max_tokens / shorten the prompt; if autotune is enabled, re-tune to derive the largest fitting context"
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

    /// A line could not be decoded as a JSON-RPC 2.0 frame. `rpc::decode` is shared
    /// by BOTH the supervisor↔worker pipe and the hub↔node control plane, so this
    /// stays transport-NEUTRAL: it means the peer (a worker, or a remote node/hub)
    /// sent a malformed/partial line — a crash mid-write or a version-mismatched
    /// peer. Retry (the supervisor auto-restarts a dead worker); a persistent
    /// failure means a faulty/mismatched peer — upgrade both higgs peers.
    #[snafu(display("[HG008] rpc decode failed: {detail} — the peer sent a malformed JSON-RPC frame (a crash mid-write or a version-mismatched peer); retry, and if it persists upgrade both higgs peers to the same version"))]
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

    /// A generation-time failure inside the engine: context creation, prompt decode,
    /// sampling, detokenize, or the decode loop. Usually transient (resource pressure)
    /// or a too-aggressive load — a reload with a smaller `ctx_len`/`gpu_layers` (less
    /// VRAM/RAM pressure) is the lever; a persistent failure points at a corrupt GGUF.
    #[snafu(display("[HG011] generation failed at {stage}: {reason} — retry; if it recurs, reload the model with a smaller ctx_len/gpu_layers (relieves VRAM/RAM pressure), or re-verify the GGUF is not corrupt"))]
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

    /// RETIRED (reserved). Formerly raised when the Gate-1 loadability probe
    /// worker failed to spawn/reply. The scan-time probe was removed — loadability
    /// is now learned only at actual load — so this is never emitted. The variant
    /// and its `HG020` code are KEPT (not deleted/renumbered) to honor the
    /// append-only code policy: downstream consumers that matched on `HG020` keep
    /// a stable, documented meaning rather than seeing the code silently reused.
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
    #[snafu(display(
        "[HG023] no agreed protocol version: peer speaks {peer:?}, we accept {ours:?}"
    ))]
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
    /// file: the download writes a temp and renames only on success. This is the GENERIC
    /// download umbrella; the HuggingFace-hub client path (`src/hub.rs`) classifies failures
    /// into the specific `HG029`–`HG035` codes below — `HG025` remains the fallback umbrella.
    #[snafu(display("[HG025] model download failed for {repo}/{file}: {detail}"))]
    #[diagnostic(code(HG025))]
    DownloadFailed {
        repo: String,
        file: String,
        detail: String,
    },

    // ── HuggingFace hub-client failure taxonomy (src/hub.rs) ─────────────────
    // The hub client is the PRIMARY fetch path (model GGUFs + card/config); the
    // hand-rolled `reqwest` path is the fail-open FALLBACK. Each distinct failure
    // mode gets its own code so an operator can tell auth from not-found from a
    // network blip — and so a terminal error names which path(s) were exhausted.
    /// Auth/permission was refused for a repo — it is gated or private and no
    /// valid `HF_TOKEN` was presented (HTTP 401/403). Actionable: set `HF_TOKEN`
    /// (or `~/.cache/huggingface/token`) and accept the model's license.
    #[snafu(display("[HG029] HuggingFace auth failed for {repo}: {detail}"))]
    #[diagnostic(code(HG029))]
    HubAuthFailed { repo: String, detail: String },

    /// A repo, revision, or file does not exist on the hub (HTTP 404). `resource`
    /// names what was missing (the repo id, a revision, or a file path).
    #[snafu(display("[HG030] HuggingFace resource not found ({resource}) in {repo}: {detail}"))]
    #[diagnostic(code(HG030))]
    HubResourceNotFound {
        repo: String,
        resource: String,
        detail: String,
    },

    /// The hub rate-limited the request (HTTP 429). Transient and retryable —
    /// back off and retry; not a permanent failure.
    #[snafu(display("[HG031] HuggingFace rate-limited for {repo}: {detail}"))]
    #[diagnostic(code(HG031))]
    HubRateLimited { repo: String, detail: String },

    /// The hub returned an unexpected HTTP status (e.g. 5xx) that is neither
    /// auth, not-found, nor rate-limit. `status` is the numeric code.
    #[snafu(display("[HG032] HuggingFace HTTP {status} for {repo}: {detail}"))]
    #[diagnostic(code(HG032))]
    HubHttpStatus {
        repo: String,
        status: u16,
        detail: String,
    },

    /// A network/transport error reaching huggingface.co — DNS, connection
    /// refused, TLS, or a timeout. Transient and retryable.
    #[snafu(display("[HG033] HuggingFace transport error for {repo}: {detail}"))]
    #[diagnostic(code(HG033))]
    HubTransport { repo: String, detail: String },

    /// A filesystem error writing a downloaded file (temp create, write, fsync,
    /// or rename) into the hub cache / target dir. Actionable: check disk space
    /// and permissions on `~/.cache/huggingface` / `~/.higgs/models`.
    #[snafu(display("[HG034] HuggingFace file write failed for {repo}/{file}: {detail}"))]
    #[diagnostic(code(HG034))]
    HubFileWrite {
        repo: String,
        file: String,
        detail: String,
    },

    /// Any other hub-client failure not covered above — a JSON/diff parse error,
    /// a malformed URL/parameter, a cache-config error, or an internal client
    /// error. Carries the underlying detail verbatim.
    #[snafu(display("[HG035] HuggingFace client error for {repo}: {detail}"))]
    #[diagnostic(code(HG035))]
    HubClient { repo: String, detail: String },

    /// BOTH the hub client (primary) AND the `reqwest` fallback failed for a
    /// fetch — the terminal, non-retryable outcome of the dual-path strategy.
    /// `primary` carries the hub failure's own `[HGxxx]` code+message;
    /// `fallback` carries the fallback's detail, so neither path's diagnosis is lost.
    #[snafu(display(
        "[HG036] HuggingFace fetch exhausted for {repo}/{file} — primary: {primary}; fallback: {fallback}"
    ))]
    #[diagnostic(code(HG036))]
    HubFetchExhausted {
        repo: String,
        file: String,
        primary: String,
        fallback: String,
    },

    // ── Control-plane RPC + peer-protocol faults (supervisor↔worker, hub↔node) ──
    /// A peer sent an RPC method this endpoint does not implement. On the
    /// internal wire this means a PROTOCOL SKEW — the two higgs binaries are
    /// different versions (a method one side speaks, the other doesn't).
    /// `endpoint` is who received it (`worker`/`node`/`hub`).
    #[snafu(display(
        "[HG037] {endpoint} received unknown RPC method '{method}' — protocol skew; rebuild/upgrade so both higgs peers run the same version"
    ))]
    #[diagnostic(code(HG037))]
    RpcMethodNotFound { endpoint: String, method: String },

    /// A control-plane peer sent a structurally-invalid or undecodable message —
    /// a malformed frame, a reply that won't deserialize, or an out-of-range id.
    /// The peer is version-mismatched or faulty; this is NOT a local fault, so the
    /// action is to update the peer and inspect ITS logs. `peer_role` is the
    /// misbehaving side (`hub`/`node`/`worker`).
    #[snafu(display(
        "[HG038] {peer_role} sent a malformed control message: {detail} — the peer is version-mismatched or faulty; upgrade both higgs peers to the same version and check the {peer_role}'s logs"
    ))]
    #[diagnostic(code(HG038))]
    ProtocolViolation { peer_role: String, detail: String },

    /// The hub refused a node's request (HELLO admission, `M_NODE_LEAVE`, or a
    /// self-retire). The usual cause is the hub no longer recognizing this node or
    /// a spent/expired pairing token. `stage` names the rejected request.
    #[snafu(display(
        "[HG039] hub rejected this node's {stage} request: {detail} — re-pair with a fresh token (`higgs node --hub <ticket> <token>`) if the hub no longer recognizes this node"
    ))]
    #[diagnostic(code(HG039))]
    HubRequestRejected { stage: String, detail: String },

    // ── Local on-disk store faults (config / pairings / keystore / models.json) ──
    /// A read/write/rename against a higgs on-disk store failed (an OS I/O error).
    /// `store` names which store (`config`/`pairings`/`keystore`/`models`); `path`
    /// is the file. The lever is the filesystem — free space and the file/directory
    /// permissions. Covers both reads and writes (the message stays neutral).
    #[snafu(display(
        "[HG040] I/O error on the {store} store at {path}: {source} — check free disk space and the file/directory permissions"
    ))]
    #[diagnostic(code(HG040))]
    PersistenceFailed {
        store: String,
        path: String,
        source: std::io::Error,
    },

    /// A higgs store file exists but its JSON will not deserialize — it is
    /// corrupted (hand-edited, truncated, or written by an incompatible version).
    /// Distinct from [HG040] (an I/O error) so the fix is unambiguous: reset the
    /// file. higgs recreates a valid default on next start.
    #[snafu(display(
        "[HG041] the {store} store at {path} is corrupted: {detail} — back up and delete the file to reset it to defaults, or repair its JSON by hand"
    ))]
    #[diagnostic(code(HG041))]
    StoreCorrupted {
        store: String,
        path: String,
        detail: String,
    },

    /// An internal invariant failed on a path that COULD recover — e.g. a
    /// `spawn_blocking` task that panicked (a `JoinError`). Converted to a typed
    /// error so the boundary returns a clean 500 instead of the process aborting.
    /// There is no user lever: this is a higgs bug. `context` names the operation.
    #[snafu(display(
        "[HG042] internal fault in {context}: {detail} — this is a higgs bug, not a configuration problem; capture this message and the surrounding logs and report it"
    ))]
    #[diagnostic(code(HG042), severity(Error))]
    InternalFault { context: String, detail: String },

    // ── Fleet/hub admin (HTTP) + background chat task ───────────────────────────
    /// A `/api/higgs/hub/*` or `/api/higgs/nodes/*` admin mutation failed (enable
    /// the hub, retire/relabel a node). `op` names the operation; `detail` carries
    /// the cause. Usually the hub is disabled or the target node id is wrong/gone.
    #[snafu(display(
        "[HG043] fleet admin operation '{op}' failed: {detail} — verify the hub is enabled and the target node id is current (`GET /api/higgs/nodes`)"
    ))]
    #[diagnostic(code(HG043))]
    HubControlFailed { op: String, detail: String },

    /// The background chat generation task aborted — it panicked or was cancelled
    /// (a tokio `JoinError`), as opposed to returning a typed engine error. The
    /// request produced no result; a retry usually succeeds. A recurring abort is
    /// a bug worth reporting with the worker logs.
    #[snafu(display(
        "[HG044] the chat generation task aborted unexpectedly: {detail} — retry the request; if it recurs, capture this message and the worker logs and report it"
    ))]
    #[diagnostic(code(HG044), severity(Error))]
    ChatTaskFailed { detail: String },

    /// The higgs control surface (`/api/higgs` + `/v1`) serve task EXITED while the
    /// host process is still running, so those endpoints are now unreachable behind
    /// a live gateway — the symptom is a connection-refused / 500 from the UI with
    /// no other cause. The serve task is meant to run for the whole process
    /// lifetime, so ANY exit is a fault. `reason` says HOW it ended (clean exit /
    /// serve error / panic / abort). The lever is a process restart; a panic is a
    /// bug — capture the backtrace and report it. Emitted by the embedding host's
    /// serve-task supervisor, NOT returned from a request handler.
    #[snafu(display(
        "[HG045] higgs control surface is DOWN ({reason}) — /api/higgs + /v1 are unreachable while the gateway is still up; restart the server (a panic is a bug: capture the backtrace and report it)"
    ))]
    #[diagnostic(code(HG045), severity(Error))]
    ControlSurfaceDown { reason: String },
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
        // HG020 is RETIRED but RESERVED (append-only policy) — the code still
        // formats even though nothing emits it anymore.
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
        assert!(HiggsError::PairingTokenInvalid {
            detail: "expired".into()
        }
        .to_string()
        .starts_with("[HG022]"));
        assert!(HiggsError::NotAllowlisted {
            endpoint_id: "z32".into()
        }
        .to_string()
        .starts_with("[HG024]"));
        assert!(HiggsError::HandshakeStalled {
            endpoint_id: "z32".into(),
            window: 5
        }
        .to_string()
        .starts_with("[HG028]"));
        assert!(HiggsError::NodeUnreachable {
            endpoint_id: "z32".into(),
            detail: "closed".into()
        }
        .to_string()
        .starts_with("[HG027]"));
    }

    #[test]
    fn hub_failure_codes_render() {
        let r = || "bartowski/Q-GGUF".to_string();
        assert!(HiggsError::HubAuthFailed {
            repo: r(),
            detail: "401".into()
        }
        .to_string()
        .starts_with("[HG029]"));
        assert!(HiggsError::HubResourceNotFound {
            repo: r(),
            resource: "file".into(),
            detail: "404".into()
        }
        .to_string()
        .starts_with("[HG030]"));
        assert!(HiggsError::HubRateLimited {
            repo: r(),
            detail: "429".into()
        }
        .to_string()
        .starts_with("[HG031]"));
        assert!(HiggsError::HubHttpStatus {
            repo: r(),
            status: 503,
            detail: "x".into()
        }
        .to_string()
        .starts_with("[HG032]"));
        assert!(HiggsError::HubTransport {
            repo: r(),
            detail: "timeout".into()
        }
        .to_string()
        .starts_with("[HG033]"));
        assert!(HiggsError::HubFileWrite {
            repo: r(),
            file: "m.gguf".into(),
            detail: "ENOSPC".into()
        }
        .to_string()
        .starts_with("[HG034]"));
        assert!(HiggsError::HubClient {
            repo: r(),
            detail: "bad json".into()
        }
        .to_string()
        .starts_with("[HG035]"));
        // The terminal both-paths-failed error preserves BOTH diagnoses.
        let exhausted = HiggsError::HubFetchExhausted {
            repo: r(),
            file: "m.gguf".into(),
            primary: "[HG029] auth".into(),
            fallback: "connection refused".into(),
        };
        let s = exhausted.to_string();
        assert!(s.starts_with("[HG036]"));
        assert!(s.contains("[HG029]") && s.contains("connection refused"));
    }

    #[test]
    fn subsystem_fault_codes_render_with_remediation() {
        // Each new code renders its [HGxxx] tag AND carries a remediation clause
        // (the "—" separator introduces the "what to do"), per the diagnostics bar.
        let cases: Vec<(HiggsError, &str)> = vec![
            (
                HiggsError::RpcMethodNotFound {
                    endpoint: "node".into(),
                    method: "higgs/bogus".into(),
                },
                "[HG037]",
            ),
            (
                HiggsError::ProtocolViolation {
                    peer_role: "node".into(),
                    detail: "inventory decode failed".into(),
                },
                "[HG038]",
            ),
            (
                HiggsError::HubRequestRejected {
                    stage: "hello".into(),
                    detail: "not allowlisted".into(),
                },
                "[HG039]",
            ),
            (
                HiggsError::PersistenceFailed {
                    store: "config".into(),
                    path: "/x/config.json".into(),
                    source: std::io::Error::other("disk full"),
                },
                "[HG040]",
            ),
            (
                HiggsError::StoreCorrupted {
                    store: "pairings".into(),
                    path: "/x/pairings.json".into(),
                    detail: "expected value".into(),
                },
                "[HG041]",
            ),
            (
                HiggsError::InternalFault {
                    context: "system info gather".into(),
                    detail: "join error".into(),
                },
                "[HG042]",
            ),
            (
                HiggsError::HubControlFailed {
                    op: "retire".into(),
                    detail: "node gone".into(),
                },
                "[HG043]",
            ),
            (
                HiggsError::ChatTaskFailed {
                    detail: "panicked".into(),
                },
                "[HG044]",
            ),
            (
                HiggsError::ControlSurfaceDown {
                    reason: "serve task panicked".into(),
                },
                "[HG045]",
            ),
        ];
        for (err, code) in cases {
            let s = err.to_string();
            assert!(s.starts_with(code), "code prefix: {s}");
            assert!(
                s.contains(" — "),
                "every diagnostic must carry a remediation clause (em-dash): {s}"
            );
        }
        // The two high-severity faults are fatal-severity.
        use miette::Diagnostic;
        assert_eq!(
            HiggsError::InternalFault {
                context: "x".into(),
                detail: "y".into()
            }
            .severity(),
            Some(miette::Severity::Error)
        );
        assert_eq!(
            HiggsError::ChatTaskFailed { detail: "y".into() }.severity(),
            Some(miette::Severity::Error)
        );
    }

    #[test]
    fn version_mismatch_is_fatal() {
        use miette::Diagnostic;
        let e = HiggsError::VersionMismatch {
            peer: vec![2],
            ours: vec![1],
        };
        assert!(e.to_string().starts_with("[HG023]"));
        assert_eq!(e.severity(), Some(miette::Severity::Error));
    }
}
