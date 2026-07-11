//! Black-box integration coverage for the STREAMING SSE assembly (`serve/stream.rs`)
//! and the REMOTE-routed streaming chat (`serve/v1.rs` `ensure_loaded` remote branch +
//! `node/data.rs` `relay_chat`), driven over the REAL `/v1` HTTP surface — now served by
//! `serve_v1_local` on an in-process, library-first `Higgs` (the standalone `higgs`
//! server + `/api/higgs/*` control plane are gone).
//!
//! What this file targets that the existing suite does NOT:
//!   (a) Incrementally reading the SSE byte stream (not buffered `.text()`) and parsing
//!       the FINISH chunk's `finish_reason` field precisely — plus the verbose
//!       `log_served` line on the SSE branch (`stream::assemble` Ok(Ok) arm with
//!       `verbose: true`), which `inference.rs` never turns on, and the
//!       `include_usage` terminal chunk ORDER relative to the finish chunk.
//!   (b) A streaming client that DROPS the response mid-stream (cancellation): the SSE
//!       body's spawned `assemble` keeps running to its outcome while its `tx.send`s
//!       discard silently (`stream.rs` `send` closure); the surface must not hang.
//!   (c) A full `/v1/chat/completions` POST on a real HUB `Higgs` (with a `HubFleet`
//!       installed) that routes the chat to a paired remote `higgs --node`, STREAMED —
//!       exercising v1's remote-resident `ensure_loaded` branch + the node-side
//!       `relay_chat` chunk relay end-to-end over `/v1` HTTP + iroh.
//!
//! Every SSE stream is fully drained or explicitly cancelled; tests skip when no tiny
//! GGUF is available.
//!
//! NOTE ON `finish_reason`: the tiny `stories260K` toy model does not reach an EOG token
//! within its small context budget (verified empirically — even a 2000-token budget
//! finishes on `length`), and stop STRINGS are a latent/unwired engine feature
//! (`engine/llamacpp/mod.rs` §"KNOWN LIMITATION"). So the SSE-extracted finish reason
//! reachable end-to-end over the real engine is `"length"`; the `"stop"` mapping branch
//! (`stream::finish_reason_from` → `Stop`) is exercised by the unit tests in `stream.rs`
//! and is NOT reachable from the HTTP surface with this model — asserted faithfully here.

mod common;

use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use futures::StreamExt;
use iroh_tickets::endpoint::EndpointTicket;
use serde_json::{json, Value};

use higgs::auth::{Allowlist, PairingTokens};
use higgs::log_bus::LogBus;
use higgs::node::fleet::HubFleet;
use higgs::node::transport::NodeTransport;
use higgs::node::{gate_connection, GateOutcome, HubIdentity, HELLO_DEADLINE};
use higgs::remote::ALPN;
use higgs::{Higgs, HiggsConfig};

use common::{higgs_local, serve_v1_local, stage_tiny_model, tiny_gguf_path, TINY_MODEL_ID};

/// Collect the `data:`-payload strings from a streamed SSE response, reading the body
/// INCREMENTALLY (byte chunks) rather than buffering with `.text()`, so the real
/// `Sse` body-unfold + per-event framing path runs. Stops after `[DONE]`.
async fn collect_sse_payloads(resp: reqwest::Response) -> Vec<String> {
    assert!(
        resp.headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .is_some_and(|ct| ct.starts_with("text/event-stream")),
        "streaming response is an event-stream"
    );
    let mut buf = String::new();
    let mut stream = resp.bytes_stream();
    let mut payloads = Vec::new();
    'outer: while let Some(chunk) = stream.next().await {
        let bytes = chunk.expect("read SSE chunk");
        buf.push_str(&String::from_utf8_lossy(&bytes));
        // SSE events are separated by a blank line; each `data:` line is one payload.
        while let Some(nl) = buf.find('\n') {
            let line = buf[..nl].trim_end_matches('\r').to_string();
            buf.drain(..=nl);
            if let Some(p) = line.strip_prefix("data: ") {
                payloads.push(p.to_string());
                if p == "[DONE]" {
                    break 'outer;
                }
            }
        }
    }
    payloads
}

/// The finish chunk's `finish_reason` (the last non-`[DONE]`, non-usage chunk that
/// carries a non-null `finish_reason` on choice 0).
fn finish_reason_of(payloads: &[String]) -> Option<String> {
    payloads
        .iter()
        .filter(|p| *p != "[DONE]")
        .filter_map(|p| serde_json::from_str::<Value>(p).ok())
        .filter_map(|v| v["choices"][0]["finish_reason"].as_str().map(str::to_owned))
        .next_back()
}

/// (a) Stream `/v1/chat/completions` with a tiny `max_tokens` and assert the SSE finish
/// chunk's `finish_reason` is exactly `"length"` (the engine breaks on the token budget).
/// Also turns VERBOSE on first (via the in-process `set_logs_settings` facade), so the
/// streaming `log_served` line fires inside the SSE assembly (`stream::assemble` verbose
/// arm) — a path `inference.rs` never enables. Then a second stream with
/// `stream_options.include_usage` asserts the terminal usage chunk comes AFTER the finish
/// chunk and carries real token counts (chunk ORDER, not just presence).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stream_finish_reason_length_verbose_and_usage_order() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP stream_finish_reason_length_verbose_and_usage_order: no tiny GGUF");
        return;
    };

    // Turn VERBOSE on so the streamed-completion `log_served` line fires inside
    // `stream::assemble` (the verbose Ok(Ok) arm) — the same effect the old
    // `PUT /api/higgs/logs/settings` had, now via the in-process facade.
    let mut settings = higgs.logs_settings();
    settings.verbose = true;
    higgs.set_logs_settings(&settings);

    // Explicit load resolves the model resident (bypassing the JIT readiness gate),
    // so the streamed `/v1` chat serves it locally.
    higgs
        .load(TINY_MODEL_ID, None)
        .await
        .expect("load tiny model");

    let (base, guard) = serve_v1_local(higgs.handle()).await;
    let c = reqwest::Client::new();

    // A tiny budget guarantees the generation breaks on `n_generated >= max_tokens` →
    // finish_reason "length" (EOG is never reached by this toy model in 4 tokens).
    let resp = c
        .post(format!("{base}/v1/chat/completions"))
        .json(&json!({
            "model": TINY_MODEL_ID, "stream": true, "max_tokens": 4, "temperature": 0.0,
            "messages": [{ "role": "user", "content": "Once upon a time" }]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200, "streaming chat opens 200");
    let payloads = collect_sse_payloads(resp).await;

    assert_eq!(
        payloads.last().map(String::as_str),
        Some("[DONE]"),
        "stream terminates with [DONE]: {payloads:?}"
    );
    // The first chunk is the assistant-role chunk (no content, role set).
    let role: Value = serde_json::from_str(&payloads[0]).unwrap();
    assert_eq!(
        role["choices"][0]["delta"]["role"], "assistant",
        "first chunk carries the assistant role: {}",
        payloads[0]
    );
    assert_eq!(role["object"], "chat.completion.chunk");
    assert_eq!(
        finish_reason_of(&payloads).as_deref(),
        Some("length"),
        "tiny max_tokens → SSE finish chunk reason is exactly \"length\": {payloads:?}"
    );

    // ── include_usage: the terminal usage chunk must come AFTER the finish chunk and
    // before [DONE], carrying real token counts with no choices (OpenAI convention).
    let resp = c
        .post(format!("{base}/v1/chat/completions"))
        .json(&json!({
            "model": TINY_MODEL_ID, "stream": true, "max_tokens": 4, "temperature": 0.0,
            "messages": [{ "role": "user", "content": "Hello there" }],
            "stream_options": { "include_usage": true }
        }))
        .send()
        .await
        .unwrap();
    let payloads = collect_sse_payloads(resp).await;

    // Locate the finish chunk index and the usage chunk index.
    let finish_idx = payloads
        .iter()
        .position(|p| {
            serde_json::from_str::<Value>(p)
                .ok()
                .is_some_and(|v| v["choices"][0]["finish_reason"].is_string())
        })
        .expect("a finish chunk is present");
    let usage_idx = payloads
        .iter()
        .position(|p| {
            serde_json::from_str::<Value>(p)
                .ok()
                .is_some_and(|v| v["usage"].is_object())
        })
        .expect("a non-null usage chunk is present");
    assert!(
        usage_idx > finish_idx,
        "usage chunk follows the finish chunk: finish={finish_idx} usage={usage_idx} {payloads:?}"
    );
    let usage: Value = serde_json::from_str(&payloads[usage_idx]).unwrap();
    assert!(
        usage["choices"].as_array().is_some_and(Vec::is_empty),
        "usage chunk carries no choices: {usage}"
    );
    assert!(
        usage["usage"]["prompt_tokens"].as_u64().is_some()
            && usage["usage"]["completion_tokens"].as_u64().is_some(),
        "usage chunk has real token counts: {usage}"
    );
    assert_eq!(
        payloads.last().map(String::as_str),
        Some("[DONE]"),
        "usage stream still ends with [DONE]"
    );

    guard.shutdown().await;
    higgs.shutdown().await;
}

/// (b) Client DROPS the stream mid-flight (cancellation). After reading a couple of byte
/// chunks we drop the response, closing the receiver side; the SSE body's spawned
/// `assemble` keeps running to its outcome while its `tx.send`s discard silently
/// (`stream.rs` `send` closure: "client disconnected — sends discard"). The surface must
/// not hang, and the SAME server must keep serving a subsequent full stream afterward —
/// proving the abandoned request released cleanly (worker not pinned).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stream_client_drops_midstream_then_server_keeps_serving() {
    let Some(higgs) = higgs_local(&[TINY_MODEL_ID]).await else {
        eprintln!("SKIP stream_client_drops_midstream_then_server_keeps_serving: no tiny GGUF");
        return;
    };
    higgs
        .load(TINY_MODEL_ID, None)
        .await
        .expect("load tiny model");
    let (base, guard) = serve_v1_local(higgs.handle()).await;
    let c = reqwest::Client::new();

    // Open a streaming chat with a generous budget so there is plenty of stream left to
    // abandon, then drop the response after the first byte chunks (early cancellation).
    let resp = c
        .post(format!("{base}/v1/chat/completions"))
        .json(&json!({
            "model": TINY_MODEL_ID, "stream": true, "max_tokens": 256, "temperature": 0.0,
            "messages": [{ "role": "user", "content": "Tell a long story please." }]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200, "streaming chat opens 200");
    {
        let mut stream = resp.bytes_stream();
        // Pull a couple of chunks so the stream is genuinely mid-flight, then drop it.
        let _ = tokio::time::timeout(Duration::from_secs(10), stream.next()).await;
        let _ = tokio::time::timeout(Duration::from_secs(10), stream.next()).await;
        drop(stream); // closes the SSE receiver — assemble's sends now discard
    }

    // Give the abandoned request a moment to wind down server-side, then prove the server
    // still serves a fresh, fully-drained stream (worker not pinned by the dropped one).
    tokio::time::sleep(Duration::from_millis(300)).await;
    let resp = c
        .post(format!("{base}/v1/chat/completions"))
        .json(&json!({
            "model": TINY_MODEL_ID, "stream": true, "max_tokens": 4, "temperature": 0.0,
            "messages": [{ "role": "user", "content": "Hi again" }]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        200,
        "server still serves after a dropped stream"
    );
    let payloads = collect_sse_payloads(resp).await;
    assert_eq!(
        payloads.last().map(String::as_str),
        Some("[DONE]"),
        "the follow-up stream completes cleanly: {payloads:?}"
    );
    assert_eq!(
        finish_reason_of(&payloads).as_deref(),
        Some("length"),
        "follow-up stream still reports a finish reason: {payloads:?}"
    );

    guard.shutdown().await;
    higgs.shutdown().await;
}

// ── (c) Hub HTTP streaming chat routed to a remote node ──────────────────────────────

/// A spawned `higgs --node` child, SIGTERM'd (graceful) on drop so its coverage profile
/// flushes.
struct NodeProc(Child);
impl Drop for NodeProc {
    fn drop(&mut self) {
        unsafe { libc::kill(self.0.id() as libc::pid_t, libc::SIGTERM) };
        let _ = self.0.wait();
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

async fn hub_endpoint() -> iroh::Endpoint {
    iroh::Endpoint::builder(iroh::endpoint::presets::Minimal)
        .alpns(vec![ALPN.to_vec()])
        .relay_mode(iroh::RelayMode::Disabled)
        .bind()
        .await
        .expect("bind hub endpoint")
}

/// (c) A `/v1/chat/completions` POST on a REAL hub `Higgs` (a `HubFleet` installed), with
/// `stream: true`, routed to a paired remote `higgs --node` that has the tiny model
/// loaded — driven over the REAL `/v1` HTTP surface via `serve_v1_local`. This drives the
/// full remote streaming path over HTTP + iroh: v1's `ensure_loaded` remote-resident
/// branch (returns the permissive `LoadedInfo`, no local worker), `chat_stream` fleet
/// routing, the node-side `relay_chat` chunk relay (`N_CHAT_CHUNK` → hub delta sink →
/// SSE), and the SSE assembly's finish chunk. Asserts the streamed SSE is well-formed,
/// carries a content delta, and ends with `[DONE]`. Built in-process via the proven
/// `remote_hub_e2e` fleet pattern (hermetic iroh, `RelayMode::Disabled`).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn hub_http_streaming_chat_routes_to_remote_node() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("SKIP hub_http_streaming_chat_routes_to_remote_node: no tiny GGUF");
        return;
    };
    let scan_root = stage_tiny_model(&gguf);
    let node_home = tempfile::tempdir().expect("node home");

    // Hub: a local Higgs (no local models) with a HubFleet installed — so any model it can
    // serve is necessarily remote-routed through the fleet.
    let bus = Arc::new(LogBus::new());
    let higgs = Arc::new(Higgs::with_log_bus(HiggsConfig::default(), bus.clone()));
    let fleet = Arc::new(HubFleet::new(bus.clone()));
    higgs.set_fleet(fleet.clone());

    // Hub iroh endpoint + a pairing ticket/token.
    let hub = hub_endpoint().await;
    let hub_id = hub.id().to_string();
    let ticket = EndpointTicket::new(hub.addr()).to_string();
    let mut allow = Allowlist::load(&node_home.path().join("hub-pairings.json")).unwrap();
    let mut tokens = PairingTokens::new();
    let token = tokens.mint(now_ms(), 600_000);

    // Spawn the real node process (hermetic iroh, real model dir).
    let child = Command::new(env!("CARGO_BIN_EXE_higgs"))
        .arg("--node")
        .arg(&ticket)
        .arg(&token)
        .env("HIGGS_HOME", node_home.path())
        .env("HIGGS_MODEL_DIR", scan_root.path())
        .env("HIGGS_IROH_LOCAL", "1")
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("spawn higgs --node");
    let _node = NodeProc(child);

    // Accept the node's dial, gate it, and register it in the fleet.
    let incoming = tokio::time::timeout(Duration::from_secs(30), hub.accept())
        .await
        .expect("node dialed within 30s")
        .expect("incoming");
    let conn = incoming.await.expect("connection");
    let peer = conn.remote_id().to_string();
    let outcome = gate_connection(
        &conn,
        &mut allow,
        &mut tokens,
        now_ms(),
        &HubIdentity::new(hub_id),
        Some("test".into()),
        HELLO_DEADLINE,
    )
    .await;
    assert!(
        matches!(outcome, GateOutcome::Admitted { .. }),
        "node admitted: {outcome:?}"
    );
    fleet
        .add_node(peer.clone(), Arc::new(NodeTransport::new(conn)), None, None)
        .await;

    // Load the tiny model on the remote node via the fleet → records the route.
    fleet
        .load(&peer, TINY_MODEL_ID, None)
        .await
        .expect("remote load");
    assert!(
        fleet.is_remote(TINY_MODEL_ID).await,
        "model is now remote-routable"
    );

    // Serve the REAL `/v1` HTTP surface on the hub Higgs.
    let (base, guard) = serve_v1_local(higgs.clone()).await;
    let c = reqwest::Client::new();

    // It is advertised as a remote-routable model in /v1/models.
    let models: Value = c
        .get(format!("{base}/v1/models"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(
        models["data"]
            .as_array()
            .is_some_and(|d| d.iter().any(|m| m["id"] == TINY_MODEL_ID)),
        "remote model routable in /v1/models: {models}"
    );

    // ── The hub's /v1 STREAMING chat, routed through the fleet to the node. ──
    let resp = c
        .post(format!("{base}/v1/chat/completions"))
        .json(&json!({
            "model": TINY_MODEL_ID, "stream": true, "max_tokens": 8, "temperature": 0.0,
            "messages": [{ "role": "user", "content": "Once upon a time" }]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        200,
        "hub streaming chat to a remote model opens 200"
    );
    let payloads = collect_sse_payloads(resp).await;

    assert_eq!(
        payloads.last().map(String::as_str),
        Some("[DONE]"),
        "remote-routed stream ends with [DONE]: {payloads:?}"
    );
    let role: Value = serde_json::from_str(&payloads[0]).unwrap();
    assert_eq!(
        role["choices"][0]["delta"]["role"], "assistant",
        "remote stream opens with the assistant-role chunk: {}",
        payloads[0]
    );
    assert_eq!(role["object"], "chat.completion.chunk");
    // The remote relay produced a finish chunk with a known reason, and the streamed
    // chunks decode as the chunk object type (full remote relay path exercised).
    assert!(
        matches!(
            finish_reason_of(&payloads).as_deref(),
            Some("stop") | Some("length")
        ),
        "remote-routed finish reason is stop|length: {payloads:?}"
    );
    let had_content = payloads
        .iter()
        .filter(|p| *p != "[DONE]")
        .filter_map(|p| serde_json::from_str::<Value>(p).ok())
        .any(|v| v["choices"][0]["delta"]["content"].is_string());
    assert!(
        had_content,
        "remote relay streamed at least one content delta: {payloads:?}"
    );

    // Tear the /v1 server down BEFORE the fleet/node so no SSE stream is open across
    // shutdown (all streams above are fully drained).
    guard.shutdown().await;
}
