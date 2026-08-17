//! Higgs diagnostics. Standalone snafu+miette infra (no jigglebot imports);
//! follows the project's four-pillar conventions by style: code baked into
//! Display, origin-only logging, severity for fatality, append-only codes.
//!
//! # Log-only diagnostic codes (no `HiggsError` variant)
//!
//! Degrade-don't-fail events carry an `[HGxxx]` token in their `tracing` line
//! instead of an error variant — the request still succeeds, but a debugging
//! agent can grep the Developer Logs for the code. Same append-only registry
//! as the variants above; NEVER reuse a number:
//!
//! | code  | site | meaning |
//! |-------|------|---------|
//! | HG051 | supervisor / hub transport | `N_CHAT_CHUNK` payload undecodable — chunk dropped (wire bug or skewed peer) |
//! | HG052 | engine `run_decode` | incremental chat parse failed mid-generation — raw content deltas for the remainder |
//! | HG053 | engine `parse_output` | crate parse rejected the full generation — raw text returned as content |
//! | HG054 | engine `parse_output` | crate parse returned non-JSON (internal bug) — raw text returned |
//! | HG055 | serve `/v1` | `chat_template_kwargs` present but not a JSON object — ignored |
//! | HG056 | serve stream | streamed tool-call fragment malformed — dropped; terminal buffered chunk covers the call |
//! | HG061 | api `load_inner_impl` | OOM degrade-retry rung taken — a load OOM'd and is being retried with a cheaper config (settle/KV-to-RAM/fewer layers) |
//! | HG062 | api `load_inner_impl` | VRAM-recovery wait timed out — the just-unloaded model's VRAM did not free before the deadline; the load proceeds anyway (may OOM → HG060 ladder) |
//! | HG065 | api Turbotune `run_benchmark` | a benchmark candidate was rejected/failed (didn't fit, OOM'd, or a log-fault watchdog match) — moving to the next candidate |

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

    /// Chat requested for a model whose GGUF header declares it non-generative — an
    /// embedder or a reranker (a pooling head and/or non-causal attention — see
    /// [`crate::worker::models::ModelDomain`]). The engine would NOT reliably fail on
    /// it (an embedder samples its pooling head and returns fluent nonsense), so this
    /// gate exists to turn silently-wrong output into a refusal the client can act on.
    #[snafu(display(
        "[HG079] {id} is a non-generative model (domain: {domain}, arch: {arch}) — it \
         cannot serve chat; pick a generative model"
    ))]
    #[diagnostic(code(HG079), severity(Error))]
    ModelNotChatCapable {
        id: String,
        arch: String,
        domain: crate::worker::models::ModelDomain,
    },

    /// Chat requested for a model that is not loaded — JIT auto-load is off, or a
    /// transient race (a restart/reap window) left it unloaded.
    #[snafu(display(
        "[HG003] model not loaded: {id} — load it (or enable JIT auto-load) and retry"
    ))]
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
    /// VRAM/RAM pressure) is the lever; `create context` failures can also be a
    /// rejected param combo (llama.cpp hard-fails a FORCED `flash_attn: on` when the
    /// model has no FA kernel, and a quantized `type_v` KV cache REQUIRES flash
    /// attention); a persistent failure points at a corrupt GGUF.
    #[snafu(display("[HG011] generation failed at {stage}: {reason} — retry; if it recurs, reload with a smaller ctx_len/gpu_layers (relieves VRAM/RAM pressure); for `create context` also try flash_attn=auto/off and an F16 KV cache (a forced flash_attn fails on models without an FA kernel, and a quantized type_v REQUIRES flash attention); or re-verify the GGUF is not corrupt"))]
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
    /// no inference runs while serving is off. The in-process control surface
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
    /// list so the `system` control-op still returns hardware/runtime rather than
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

    /// A node is not currently reachable: a paired node whose connection closed / dial
    /// failed / wedged-worker transport was dropped (routes kept, recovers on reconnect),
    /// OR an unknown/never-connected endpoint id (no routes — e.g. a `scan_node` for a ghost
    /// id). Either way it is NOT "retired from the fleet" — retire is a separate, explicit
    /// removal that drops the node slot. The remediation is therefore phrased conditionally,
    /// so it doesn't promise reconnect-recovery for an id the fleet never knew (§3.4, §3.4.1).
    #[snafu(display(
        "[HG027] node {endpoint_id} unreachable: {detail} — if it is a paired node, it recovers once it reconnects"
    ))]
    #[diagnostic(code(HG027))]
    NodeUnreachable { endpoint_id: String, detail: String },

    /// A node that does NOT ship the signature-verified self-update receiver (a LEGACY build that
    /// predates REL-P4, or a build compiled without it) was sent an `M_UPDATE` push and refuses it
    /// with this typed error. The CURRENT build ships the full receiver+applier ([`crate::node::
    /// self_update`]) and advertises the `update` capability `true`, so it never emits this for its
    /// own `M_NODE_UPDATE` handling — a bad manifest/artifact fails later with HG081-HG088 instead.
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
        "[HG039] hub rejected this node's {stage} request: {detail} — re-pair with a fresh token (`higgs --node <ticket> <token>`, or one-shot `higgs node connect <ticket> <token>`) if the hub no longer recognizes this node"
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

    // ── Fleet/hub admin (control ops) + background chat task ─────────────────────
    /// A fleet admin mutation failed (enable the hub, pair, or a node op like
    /// load/unload/retire/relabel). `op` names the operation; `detail` carries the
    /// cause. The remediation tail DEFERS to that cause rather than prescribing —
    /// the raisers span "hub disabled", "pairing failed", "unknown node" and
    /// wrapped persistence failures ([HG040] inside the detail), so any concrete
    /// one-size advice is wrong for some of them (a node-id hint is nonsense for
    /// `pair`; "check the fleet list" is nonsense when the real lever is disk
    /// space). The `hub`/`nodes` pointer is framed as where to INSPECT state, not
    /// as the fix. (The old tail named `GET /api/higgs/nodes`, an HTTP surface
    /// that no longer exists.)
    #[snafu(display(
        "[HG043] fleet admin operation '{op}' failed: {detail} — address the cause above and retry (the `hub`/`nodes` control ops report the current fleet state)"
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
        "[HG045] higgs serve task is DOWN ({reason}) — the `/v1` surface is unreachable while the embedding host is still up; restart the server (a panic is a bug: capture the backtrace and report it)"
    ))]
    #[diagnostic(code(HG045), severity(Error))]
    ControlSurfaceDown { reason: String },

    /// A chat/load targeted a model with no canonical load profile. JIT will not
    /// load with silent defaults — the model must be Prepared (autotuned) first
    /// so it loads with the right context/offload for this hardware.
    #[snafu(display(
        "[HG046] model not prepared: {id} — run autotune (Prepare) to pin a load profile before serving"
    ))]
    #[diagnostic(code(HG046))]
    NotPrepared { id: String },

    /// A chat/load targeted a model whose profile is stale — the hardware or the
    /// model file changed since it was Prepared, so the profile may no longer fit.
    /// Re-tune before loading (a stale profile hard-blocks, by design).
    #[snafu(display(
        "[HG047] profile stale for {id}: hardware or model file changed since Prepare — Re-tune before loading"
    ))]
    #[diagnostic(code(HG047))]
    ProfileStale { id: String },

    /// A `/v1` request presented no API key, or one the node's keystore does not
    /// hold. Carries the code so a `401` is as diagnosable as any other reply.
    /// The message names no file: auth is armed by the LIVE keystore, which an
    /// embedder can populate with in-memory internal tokens
    /// (`register_internal_token`) that `api_keys.json` never contains — advice
    /// to edit or delete that file would be wrong, and possibly ineffective,
    /// on such a node.
    #[snafu(display(
        "[HG048] unauthorized: missing or insufficient API key — send `Authorization: Bearer <key>` with a key minted on this node (its operator manages keys via mint/revoke)"
    ))]
    #[diagnostic(code(HG048))]
    Unauthorized,

    /// A `/v1` request body failed validation before reaching the engine (a
    /// malformed message/tool list, or a rejected field). `detail` is the
    /// specific reason; codes the otherwise-bare `400` so it states the fix.
    #[snafu(display(
        "[HG049] {detail} — check the request body against the OpenAI chat schema and retry"
    ))]
    #[diagnostic(code(HG049))]
    InvalidRequest { detail: String },

    /// The model's GGUF-embedded Jinja chat template failed to render over
    /// this request (llama.cpp's `common_chat` template apply, via the
    /// AI-Experiri llama-cpp-rs fork). Distinct from [HG011] so an agent can
    /// tell "the prompt could not even be built" from an engine failure:
    /// generation never started, and retrying the identical request will fail
    /// identically.
    #[snafu(display("[HG050] chat template render failed: {reason} — the request cannot build a prompt for this model; if the request carried `tools`, retry WITHOUT tools (the template may not support them); if it recurs on plain chat, the GGUF's embedded template is broken or uses unsupported Jinja — re-download the model or pick a different one"))]
    #[diagnostic(code(HG050))]
    TemplateRenderFailed { reason: String },

    /// A streaming client stalled long enough that its undelivered chat deltas
    /// exceeded the per-request buffer cap ([`delta_queue::CAP_BYTES`]), so the
    /// buffered stream was dropped rather than growing without bound. The
    /// generation itself completed on the worker; only this client's incremental
    /// view was truncated, and the request errors LOUDLY instead of silently
    /// omitting text. Reissue the request (non-streaming, or with a client that
    /// keeps reading).
    ///
    /// [`delta_queue::CAP_BYTES`]: crate::delta_queue::CAP_BYTES
    #[snafu(display(
        "[HG057] chat stream buffer overflow: the client stopped reading and {buffered_bytes} buffered delta bytes exceeded the cap — the stream was aborted (the generation finished server-side); retry non-streaming or keep the connection drained"
    ))]
    #[diagnostic(code(HG057))]
    ChatStreamOverflow { buffered_bytes: usize },

    /// Startup refused: a non-loopback bind (`HIGGS_BIND` beyond `127.0.0.1`)
    /// with ZERO configured API keys would expose the open control + `/v1`
    /// surface to the whole network — the Host guard and CORS only protect
    /// browser clients. Fail closed at startup instead of serving wide open:
    /// mint a key first (`higgs keys add` for a standalone server — a REAL,
    /// wired subcommand, `bin/higgs.rs` → `keys::run_keys` — or
    /// `Higgs::mint_key` from an embedder), or bind loopback. The deleted
    /// `/api/higgs/keys` HTTP surface is deliberately not mentioned.
    #[snafu(display(
        "[HG058] refusing to bind {bind}: no API keys are configured, so the `/v1` surface would be OPEN to the network — mint an Admin-capable key first (`higgs keys add <label> admin` under the server's HIGGS_HOME, then restart; or `Higgs::mint_key` from an embedder — a LAN bind requires Admin, [HG069]), or bind 127.0.0.1"
    ))]
    #[diagnostic(code(HG058), severity(Error))]
    LanBindWithoutKeys { bind: String },

    /// A model load exhausted the OOM degrade-retry ladder (G5): every rung —
    /// the plain retry, KV-cache-to-system-memory, and reduced GPU layers — hit
    /// an out-of-memory failure. The machine genuinely can't fit this model as
    /// configured right now. `attempts` is how many loads were tried; `last`
    /// is the final engine reason. The lever is a smaller footprint (lower
    /// `ctx_len`/`gpu_layers`, a smaller quant) or freeing VRAM (close other GPU
    /// apps); a re-tune (Prepare) derives a fitting profile for this hardware.
    #[snafu(display(
        "[HG060] model load ran out of memory after {attempts} attempts (plain retry → KV to system memory → fewer GPU layers): {last} — free VRAM (close other GPU apps), lower ctx_len/gpu_layers, pick a smaller quant, or re-tune (Prepare) for this hardware"
    ))]
    #[diagnostic(code(HG060), severity(Error))]
    LoadOomExhausted { attempts: usize, last: String },

    /// A Turbotune (measured) benchmark run (`TuneMode::Benchmark`) produced no
    /// usable measurement: every candidate config either did not fit the budget
    /// or failed to load/measure. `detail` names each candidate and why. The
    /// analytical `Suggest` tune still works — the machine just can't be
    /// measured for this model right now (free VRAM, or lower the budget).
    #[snafu(display(
        "[HG063] Turbotune benchmark found no working config: {detail} — free VRAM or lower the resource budget, or use the analytical tune (Suggest) instead"
    ))]
    #[diagnostic(code(HG063), severity(Error))]
    BenchExhausted { detail: String },

    /// A Turbotune benchmark run was CANCELLED because a load / unload / stop
    /// arrived for the same facade while it was benching (a bench holds the
    /// worker to measure it, so a concurrent model op takes precedence). Re-run
    /// the benchmark when the machine is idle.
    #[snafu(display(
        "[HG064] Turbotune benchmark cancelled: a model load/unload/stop arrived mid-run — re-run the benchmark when idle"
    ))]
    #[diagnostic(code(HG064))]
    BenchCancelled,

    /// Refused to revoke the LAST API key while the server is bound beyond
    /// loopback: an empty keystore turns auth OFF, and on a LAN bind (whose
    /// Host guard is deliberately relaxed for keyed clients) that would leave
    /// the entire control + `/v1` surface open to the network at runtime —
    /// silently bypassing the [HG058] startup protection. Mint a replacement
    /// key first, or restart bound to `127.0.0.1` to manage keys freely.
    #[snafu(display(
        "[HG059] refusing to revoke the last API key {label:?} while bound beyond loopback — the surface would go fully OPEN to the network; mint a replacement key first, or restart bound to 127.0.0.1"
    ))]
    #[diagnostic(code(HG059), severity(Error))]
    LastKeyOnLan { label: String },

    /// Refusing to revoke the last Admin-capable key while other (non-Admin) keys
    /// remain: auth would stay ON but the Admin-only key-management surface
    /// (`/api/higgs/keys`) becomes unreachable — an operator lockout recoverable
    /// only by editing the keystore out-of-band. Mint a replacement Admin key
    /// first, or revoke every key (which turns auth off) instead.
    #[snafu(display(
        "[HG066] refusing to revoke {label:?}: it holds the last Admin-capable key, and other keys remain — the key-management API would lock out; mint a replacement Admin key first, or revoke all keys to disable auth"
    ))]
    #[diagnostic(code(HG066), severity(Error))]
    LastAdminKey { label: String },

    /// A benchmark was requested for a model that is currently LOADED. Benchmarking
    /// measures candidate configs by loading/unloading the model, so it needs the
    /// model unloaded first — otherwise it would disrupt the live worker and the
    /// numbers would be contaminated by the resident model. Unload it, then bench.
    #[snafu(display(
        "[HG067] model {id:?} is loaded — unload it first to benchmark (benchmarking loads/unloads candidate configs and needs the model offline)"
    ))]
    #[diagnostic(code(HG067), severity(Error))]
    BenchModelLoaded { id: String },

    /// A load/chat was requested for a model that is currently being BENCHMARKED. The
    /// benchmark owns the model while it measures candidate configs; the request is
    /// refused rather than racing it. Retry once the benchmark finishes (~5 min).
    #[snafu(display(
        "[HG068] model {id:?} is being benchmarked — retry in ~5 min (a benchmark owns the model while it measures candidate configs)"
    ))]
    #[diagnostic(code(HG068), severity(Error))]
    BenchInProgress { id: String },

    /// An EPHEMERAL load (`load_ephemeral`) targeted an already-resident model.
    /// The resident worker keeps its CURRENT params, so silently "succeeding"
    /// would hand the caller a config it did not pin — the exact failure an
    /// ephemeral load exists to prevent. Eject the model first.
    #[snafu(display(
        "[HG080] {id} is already loaded — an ephemeral load applies EXACTLY the requested params, so eject the resident model first"
    ))]
    #[diagnostic(code(HG080), severity(Error))]
    EphemeralResident { id: String },

    /// An update manifest named a `pinned_key_id` this binary does not pin in
    /// [`crate::update::HIGGS_UPDATE_PUBKEYS`] (or the table is empty). Fails
    /// CLOSED: an unpinned key can never authorize an update. Two honest
    /// causes: this binary predates a key rotation (the remedy is one manual
    /// update to a release that pins the new key), or nobody has populated the
    /// table yet (self-update is simply not enabled for this build).
    #[snafu(display(
        "[HG081] update refused: release key {key_id:?} is not pinned in this binary — update manually once to a build that pins it (`crate::update::HIGGS_UPDATE_PUBKEYS`)"
    ))]
    #[diagnostic(code(HG081), severity(Error))]
    UpdateKeyUnknown { key_id: String },

    /// An update manifest's minisign signature did not verify against the
    /// pinned release key (or the key/signature text itself was malformed).
    /// The manifest is untrusted input until this check passes, so nothing in
    /// it was read; `detail` states which step rejected it.
    #[snafu(display("[HG082] update refused: {detail}"))]
    #[diagnostic(code(HG082), severity(Error))]
    UpdateSignatureInvalid { detail: String },

    /// An update manifest passed signature verification but is not a manifest
    /// this binary can use: unparseable JSON, or a schema it does not know
    /// ([`crate::update::UPDATE_MANIFEST_SCHEMA`]). A schema mismatch means
    /// release CI moved ahead of this binary — update manually once.
    #[snafu(display("[HG083] update refused: {detail}"))]
    #[diagnostic(code(HG083), severity(Error))]
    UpdateManifestInvalid { detail: String },

    /// A downloaded update artifact does not match the sha256 its VERIFIED
    /// manifest promises — a truncated/corrupted download or a courier serving
    /// the wrong bytes. The artifact must be discarded, never unpacked.
    #[snafu(display(
        "[HG084] update refused: artifact {file} hashes to {got} but its signed manifest promises {expected} — discard the download and re-fetch"
    ))]
    #[diagnostic(code(HG084), severity(Error))]
    UpdateArtifactMismatch {
        file: String,
        expected: String,
        got: String,
    },

    /// Self-update ELIGIBILITY refused a manifest that AUTHENTICATED fine (§9 P3,
    /// `src/node/self_update.rs`): its `version` is not newer than the running
    /// binary's. Authenticity (`src/update.rs`) proves "release CI built this",
    /// never "this is an upgrade" — a genuine OLD release, replayed by a courier,
    /// still verifies, so the updater refuses a downgrade unless `--allow-downgrade`
    /// is passed. `from`/`to` are both authenticated version strings.
    #[snafu(display(
        "[HG085] self-update refused: manifest version {to} is not newer than the running {from} — a STRICTLY older version needs --allow-downgrade; the SAME version cannot be self-updated (reinstall via install.sh to repair it)"
    ))]
    #[diagnostic(code(HG085), severity(Error))]
    UpdateNotNewer { from: String, to: String },

    /// Self-update eligibility refused a manifest whose `target` triple or
    /// acceleration `variant` does not match the running binary (§9 P3). A CUDA
    /// build on a CPU-only box (or an `x86_64` artifact on `aarch64`) fails to even
    /// load, so applying it would brick the node — refused before any download is
    /// swapped in. Both fields are authenticated; `field` names which mismatched.
    #[snafu(display(
        "[HG086] self-update refused: manifest {field} {manifest:?} does not match this binary's {running:?} — this artifact is for a different build and would not run here"
    ))]
    #[diagnostic(code(HG086), severity(Error))]
    UpdateTargetMismatch {
        field: String,
        manifest: String,
        running: String,
    },

    /// Self-update failed at an APPLY step AFTER authentication+eligibility passed
    /// (§9 P3): unpack, the post-stage `--version` smoke test, the atomic `current`
    /// flip, a filesystem error under `<prefix>/bin`, a rollback/prune op, or the
    /// single-update lock being held by a concurrent run. Nothing partial is left
    /// live: staging happens off to the side and `current` is flipped atomically, so
    /// a failure here leaves the previously-installed binary running. `detail` names
    /// the step.
    #[snafu(display("[HG087] self-update failed: {detail}"))]
    #[diagnostic(code(HG087), severity(Error))]
    UpdateApplyFailed { detail: String },

    /// Self-update could not FETCH the manifest/signature/artifact over the network
    /// (§9 P4): a malformed/insecure URL (non-`https` to a non-loopback host), an HTTP
    /// error, a connect/read timeout, or a response exceeding the size cap. This is a
    /// TRANSPORT failure BEFORE any authentication — the fetched bytes are still verified
    /// against the pinned key (HG081-084) before anything is applied, so a fetch never
    /// bypasses the signature. `detail` names the URL/step.
    #[snafu(display("[HG088] self-update fetch failed: {detail}"))]
    #[diagnostic(code(HG088), severity(Error))]
    UpdateFetchFailed { detail: String },

    /// An in-flight model download was CANCELLED — today that means the
    /// caller dropped/aborted the op future (a local drop kills the
    /// transfer; an unattested remote drop may have never reached the
    /// node); once the FUTURE cancel-dispatch slice ships, an explicit
    /// operator cancel produces the same code. Terminal for that transfer: the per-transfer temp-guard
    /// unlinks the `.part.<pid>.<seq>` file on future-drop (best-effort — a
    /// unlink failure is only visible in tracing) and nothing lands in
    /// `~/.higgs/models/`. Distinct from [HG025] (a FAILURE) so the UI can
    /// render "cancelled" instead of an error.
    #[snafu(display(
        "[HG089] model download cancelled for {repo}/{file} — partial temp cleanup attempted (best-effort; check the node log if disk usage grows)"
    ))]
    #[diagnostic(code(HG089), severity(Warning))]
    DownloadCancelled {
        repo: String,
        file: String,
        /// Historical field: since r46 the per-transfer temp guard in
        /// [`crate::download::download`] performs cleanup asynchronously on
        /// its own drop; the outcome (rare unlink failure on
        /// permissions/I/O) is only visible in tracing, not on this wire
        /// field. Kept for on-wire back-compat and always set to `true` by
        /// the cancel path — do NOT read as "cleanup verified". The
        /// diagnostic message reflects the best-effort reality.
        partial_swept: bool,
    },

    /// A download request was refused because the SAME (node, repo, file) is
    /// already transferring — one copy per key; wait for it or cancel it.
    /// Its own code (not the [HG025] failure umbrella) so the hub classifies
    /// "already downloading" purely by code — a wait/info state in the UI,
    /// never an error toast.
    #[snafu(display(
        "[HG090] download already in flight for {repo}/{file} — wait for it or cancel it first"
    ))]
    #[diagnostic(code(HG090), severity(Warning))]
    DownloadInFlight { repo: String, file: String },

    /// A cancel signal was accepted but the download finished before it could
    /// be observed — the file IS on disk. Surfaced (as an info-severity signal,
    /// not an error) so the UI can distinguish "cancel honored, nothing landed"
    /// (`[HG089]`) from "cancel outraced by completion, file landed" — both
    /// have a legitimate meaning and neither is a bug. The download's terminal
    /// event stream still fires `Done` (the file is real).
    ///
    /// BEST-EFFORT emission: HG091 fires when `cancellable_pull` observes the
    /// cancel signal set at the moment it returns Ok. Because cancel signals
    /// and completion resolve on separate async task-wake paths, a cancel
    /// accepted in the microscopic window AFTER the outrace-check reads
    /// `false` but BEFORE the receiver is dropped may not fire HG091 — the
    /// caller sees `cancel() → Ok` and the event stream sees `Done`, with
    /// no warning between them. That residual is the known, documented
    /// limit of async cancel semantics (see [`cancellable_pull`]'s two-regime
    /// contract); the truth on disk is not affected.
    #[snafu(display(
        "[HG091] cancel requested but download for {repo}/{file} completed first — the file is on disk"
    ))]
    #[diagnostic(code(HG091), severity(Warning))]
    CancelLostToCompletion { repo: String, file: String },

    /// Startup refused: a non-loopback bind with keys present but NONE holding the
    /// `Admin` scope. Auth would be ON, but every Admin-scoped operation
    /// (mint/revoke) is then rejected — the operator can't manage keys on the
    /// running LAN server and must fix the keystore out-of-band and restart.
    /// A mint without explicit scopes defaults to `chat,models` (both the
    /// `higgs keys add` CLI and `Higgs::mint_key`), so this is the easy footgun.
    /// Fail closed at startup: mint an Admin key and restart, or bind loopback.
    /// Symmetric to [HG058]. The message names the CLI (real: `keys::run_keys`)
    /// and the crate API — never `keys_mint`, which is an EMBEDDER's wire tag
    /// (jigglebot's control-op name), not a higgs identifier.
    #[snafu(display(
        "[HG069] refusing to bind {bind}: API keys are configured but NONE is Admin-capable, so key management (mint/revoke) would be locked out on this LAN bind — mint an Admin-capable key (`higgs keys add <label> admin` under the server's HIGGS_HOME, or `Higgs::mint_key` from an embedder) and restart, or bind 127.0.0.1"
    ))]
    #[diagnostic(code(HG069), severity(Error))]
    LanBindWithoutAdminKey { bind: String },

    /// GGUF header enrichment FAILED for a scanned model — the file could not be
    /// opened/mmapped, or its header failed to parse (a truncated file
    /// mid-download, a malformed metadata table). The model is still cataloged with whatever
    /// partial fields were read before the failure; this code is stamped onto the
    /// model entry (`HiggsModel::enrich_error`) so the UI can explain the blank
    /// header fields instead of showing the model as genuinely sparse. Non-fatal —
    /// a corrupt or mid-download GGUF never aborts the scan.
    #[snafu(display("[HG070] GGUF enrichment failed for {path}: {reason}"))]
    #[diagnostic(code(HG070))]
    GgufEnrichFailed { path: String, reason: String },

    /// A CORS origin submitted to [`crate::Higgs::set_cors_origins`] is not a bare
    /// origin. The extra-origins allowlist stores exact-match `Origin` header
    /// values — a browser's `Origin` is always `scheme://host[:port]` with an
    /// `http`/`https` scheme and NO userinfo, path (beyond a single trailing `/`,
    /// which is normalized away), query, or fragment. The offending entry is
    /// echoed with the specific reason so the operator can fix the one bad value.
    /// Rejected BEFORE persistence, so a bad entry never reaches `config.json` or
    /// the running CORS allowlist.
    #[snafu(display(
        "[HG071] invalid CORS origin {origin:?}: {reason} (expected an exact origin like \"https://tools.example\" — scheme://host[:port], no path/query/fragment)"
    ))]
    #[diagnostic(code(HG071), severity(Error))]
    InvalidCorsOrigin { origin: String, reason: String },

    /// A KEYSTORE request failed validation: an empty scope list, a duplicate
    /// label, a first key that omits `admin`, or a revoke naming a key that does
    /// not exist. Split from [HG049], which is the `/v1` request-body error and
    /// signs off with "check the request body against the OpenAI chat schema" —
    /// advice that is nonsense for a token mint, and which the Manage-Tokens UI
    /// renders verbatim. `detail` states the specific rule and its remedy, so no
    /// generic tail is appended here.
    #[snafu(display("[HG072] {detail}"))]
    #[diagnostic(code(HG072), severity(Error))]
    InvalidKeyRequest { detail: String },

    /// A listener rebind was requested on a facade whose terminal `stop()` has
    /// already run — there is nothing left to serve. Raised by
    /// `Higgs::reserve_rebind` before any listener is touched, so the caller's
    /// old listener (if it still runs) keeps serving untouched.
    #[snafu(display(
        "[HG073] cannot rebind the /v1 listener: this higgs instance is shutting down — \
         restart the embedding application to serve again"
    ))]
    #[diagnostic(code(HG073), severity(Error))]
    RebindAfterShutdown,

    /// A node-directed chat test ([`crate::Higgs::node_chat_test`]) found NO served
    /// instance ROUTED on the target node, so the hub has nothing to relay the test
    /// prompt to. The route table is the hub's in-memory dispatch truth: it survives
    /// a node DISCONNECT (that case surfaces as [HG027] from the chat itself, not
    /// here) but NOT a hub process restart — after one this fires even though the
    /// node's card shows a resident model. The remedy is a load on the node, which
    /// routes a NEW instance; loads are strictly additive, so a worker parked by a
    /// hub restart is NOT re-attached — it stays resident and route-less (no hub
    /// unload lever) until the node's own idle reaper reclaims it. The node itself
    /// is guaranteed known — an unknown endpoint id is refused earlier as [HG075].
    #[snafu(display(
        "[HG074] no served model instance is routed on node {endpoint_id} — load a model \
         there (`Higgs::node_load` / the Fleet view's Load) to route a NEW instance (a \
         worker parked by a hub restart is not re-attached; the node's idle reaper reclaims it)"
    ))]
    #[diagnostic(code(HG074), severity(Error))]
    NodeNothingServed { endpoint_id: String },

    /// The node chat test ([`crate::Higgs::node_chat_test`], its only producer
    /// today) named an endpoint id this hub has NEVER paired (or has since
    /// retired) — distinct from [HG074] (node known, route table empty → load
    /// first) and from [HG027] (node known but offline → reconnect). Raised
    /// before any route/served resolution, so "load a model on it first" advice
    /// is never issued for a node that does not exist. Sibling node ops
    /// (`node_load`/`node_scan`) predate this gate and still surface an unknown
    /// id as HG027 — aligning them is a candidate follow-up, not a property
    /// this code already guarantees surface-wide. The hub's OWN endpoint id
    /// lands here too when unpaired — deliberately: pairing the hub machine to
    /// itself as a node is a real topology with a real iroh hop, so "pair the
    /// node first" is honest advice there (unlike the `"local"` sentinel,
    /// which is refused earlier as [HG076]).
    #[snafu(display(
        "[HG075] unknown node {endpoint_id} — it is not paired with this hub; check the id \
         (the fleet view / `nodes` op lists every paired node), or pair the node first"
    ))]
    #[diagnostic(code(HG075), severity(Error))]
    UnknownNode { endpoint_id: String },

    /// A node chat test ([`crate::Higgs::node_chat_test`]) refused its target
    /// at the pre-dispatch check — three shapes: the `"local"` sentinel (this
    /// machine has no iroh hop to prove; runs before even the not-a-hub gate),
    /// an explicit `served` operand that is not routed anywhere, or one that
    /// resolves to a DIFFERENT node than the test names (a reply would attest
    /// a link the test never exercised). A caller-input refusal against the
    /// state as it stands when the call arrives; the CONCURRENT-change refusal
    /// (target moved between the pre-check and the pinned dispatch) is
    /// [HG077], not this. `detail` names the specific conflict and its remedy.
    #[snafu(display("[HG076] invalid chat-test target: {detail}"))]
    #[diagnostic(code(HG076), severity(Error))]
    InvalidChatTestTarget { detail: String },

    /// The pinned chat dispatch
    /// ([`chat_pinned`](crate::node::fleet::HubFleet::chat_pinned)) found the
    /// served id NOT at the pinned node: it resolved to a different node, or to
    /// nothing at all. Via the facade's chat test this means the target moved
    /// CONCURRENTLY between pick and dispatch (served ids renumber over a
    /// model's whole instance set, so a single unload or additive load re-homes
    /// them) and a retry succeeds; a DIRECT `chat_pinned` caller can also land
    /// here on a first call with an id that never resolved — the detail states
    /// only what was checked, fabricating no history either way. Refused rather
    /// than silently exercising (and then reporting) the wrong node.
    #[snafu(display(
        "[HG077] chat-test target not at the pinned node at dispatch: {detail} — served ids \
         renumber as instance sets change; re-resolve against the fleet view"
    ))]
    #[diagnostic(code(HG077), severity(Error))]
    ChatTestTargetMoved { detail: String },

    /// A remote load carried explicit params, but the target node negotiated
    /// only protocol major `agreed` (< 2, the major where the hub started
    /// sending per-load params). Some major-1 builds would PARSE the fields;
    /// older ones (pre-rich-params / pre-typed-gpu_layers) would hard-reject
    /// them — the hub cannot distinguish, and silently loading with the node's
    /// defaults when the operator asked for specific params would be a lie
    /// either way — refuse instead. Param-less loads still work against any
    /// node.
    #[snafu(display(
        "[HG078] node {endpoint_id} is on protocol {agreed} (or predates version reporting), \
         before per-load params (major 2) — update higgs on the node and reconnect, or load \
         without params"
    ))]
    #[diagnostic(code(HG078), severity(Error))]
    NodeTooOldForParams { endpoint_id: String, agreed: u32 },
}

#[cfg(test)]
#[path = "diagnostic_tests.rs"]
mod tests;
