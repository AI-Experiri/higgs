//! Tool-call chat routed through the IN-PROCESS hub fleet to a REAL remote node.
//!
//! higgs is now library-first: there is no standalone server and no
//! `/api/higgs/*` control surface. The old multi-model download fleet
//! (`HIGGS_TEST_FLEET`) is retired; this test now builds the fleet IN-PROCESS —
//! a hub `Higgs` with a `HubFleet`, one real `higgs --node` child paired over
//! hermetic iroh, and the tiny GGUF loaded on that node. A tool-bearing chat is
//! then driven over the REAL `/v1` HTTP surface (`serve_v1_local`), so the
//! request routes hub → fleet → node → worker and the OpenAI response shape is
//! asserted end-to-end.
//!
//! The chat pipeline (template apply + tool-call/reasoning parse) is llama.cpp's
//! own `common_chat` machinery via the AI-Experiri llama-cpp-rs fork — higgs
//! ships no parser of its own. The tiny `stories260K` model doesn't emit
//! structured calls deterministically, so the well-formed-response assertion is
//! unconditional and the OpenAI-tool-call-shape assertion is opportunistic
//! (exactly as the original fleet E2E treated its small instruct models).
//!
//! Skips when no tiny GGUF is available.

mod common;

use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use iroh_tickets::endpoint::EndpointTicket;
use serde_json::{json, Value};

use higgs::auth::{Allowlist, PairingTokens};
use higgs::log_bus::LogBus;
use higgs::node::fleet::HubFleet;
use higgs::node::transport::NodeTransport;
use higgs::node::{gate_connection, GateOutcome, HubIdentity, HELLO_DEADLINE};
use higgs::remote::ALPN;
use higgs::{Higgs, HiggsConfig};

use common::{serve_v1_local, stage_tiny_model, tiny_gguf_path, TINY_MODEL_ID};

/// The get_weather tool used by the E2E tool chat.
const TOOLS: &str = r#"[
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"}
                },
                "required": ["city"]
            }
        }
    }
]"#;

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// A spawned `higgs --node` child, SIGTERM'd + reaped on drop (a graceful stop
/// flushes its llvm-cov profile; a hard kill drops it).
struct NodeProc(Child);
impl Drop for NodeProc {
    fn drop(&mut self) {
        // SAFETY: a plain kill(2) on our own child's pid.
        unsafe {
            libc::kill(self.0.id() as libc::pid_t, libc::SIGTERM);
        }
        let _ = self.0.wait();
    }
}

/// A hermetic (relay-disabled) hub iroh endpoint on the higgs remote ALPN.
async fn hub_endpoint() -> iroh::Endpoint {
    iroh::Endpoint::builder(iroh::endpoint::presets::Minimal)
        .alpns(vec![ALPN.to_vec()])
        .relay_mode(iroh::RelayMode::Disabled)
        .bind()
        .await
        .expect("bind hub endpoint")
}

/// Accept + gate the next inbound node connection, returning `(conn, peer_id)`.
async fn admit_node(
    hub: &iroh::Endpoint,
    allow: &mut Allowlist,
    tokens: &mut PairingTokens,
    hub_id: &str,
) -> (iroh::endpoint::Connection, String) {
    let incoming = tokio::time::timeout(Duration::from_secs(30), hub.accept())
        .await
        .expect("node dialed within 30s")
        .expect("incoming");
    let conn = incoming.await.expect("connection");
    let peer = conn.remote_id().to_string();
    let outcome = gate_connection(
        &conn,
        allow,
        tokens,
        now_ms(),
        &HubIdentity::new(hub_id),
        Some("test".into()),
        HELLO_DEADLINE,
    )
    .await;
    assert!(
        matches!(outcome, GateOutcome::Admitted { .. }),
        "admitted: {outcome:?}"
    );
    (conn, peer)
}

/// Live fleet E2E: pair one real node, load the tiny model on it via the
/// `HubFleet`, then run ONE tool-call chat over the hub's `/v1` HTTP surface —
/// which routes hub → fleet → node → worker. Assert a well-formed OpenAI
/// response; when structured `tool_calls` come back, assert the OpenAI shape and
/// that no call markup leaked into `content`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fleet_e2e_tool_chat() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("SKIP fleet_e2e_tool_chat: no tiny GGUF (set HIGGS_TEST_GGUF)");
        return;
    };
    let scan_root = stage_tiny_model(&gguf);
    let node_home = tempfile::tempdir().expect("node home");
    let hub_home = tempfile::tempdir().expect("hub home");

    // Isolate the hub's HIGGS_HOME so `serve_v1` reads an EMPTY keystore (auth off)
    // — otherwise a developer's real `~/.jigglebot/higgs/api_keys.json` would turn
    // auth ON and 401 this no-token chat. Read lazily by the facade at construction,
    // so set it before building the hub. This binary runs single-threaded
    // (`--test-threads=1`) with one test, so a bare `set_var` needs no restore.
    // SAFETY: no other thread touches the process env in this single-test binary.
    unsafe {
        std::env::set_var("HIGGS_HOME", hub_home.path());
    }

    // Hub: a local Higgs (no local models) with a HubFleet installed.
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

    // Admit the node's dial, gate it, and register it in the fleet.
    let (conn, peer) = admit_node(&hub, &mut allow, &mut tokens, &hub_id).await;
    fleet
        .add_node(peer.clone(), Arc::new(NodeTransport::new(conn)), None, None)
        .await;

    // Load the tiny model on the node via the fleet → records the remote route.
    fleet
        .load(&peer, TINY_MODEL_ID, None)
        .await
        .expect("remote load");
    assert!(
        fleet.is_remote(TINY_MODEL_ID).await,
        "model is now remote-routable"
    );

    // Drive a tool-call chat over the REAL /v1 HTTP surface: the request routes
    // hub → fleet → node → worker and comes back in the OpenAI shape.
    let (base, guard) = serve_v1_local(higgs.clone()).await;
    let c = reqwest::Client::new();
    let tools: Value = serde_json::from_str(TOOLS).expect("tools parse");

    let resp: Value = c
        .post(format!("{base}/v1/chat/completions"))
        .json(&json!({
            "model": TINY_MODEL_ID,
            "stream": false,
            "max_tokens": 256,
            "messages": [
                { "role": "system", "content": "You are a helpful assistant." },
                { "role": "user", "content": "What's the weather in Paris? Use the get_weather tool." }
            ],
            "tools": tools
        }))
        .send()
        .await
        .expect("chat request")
        .json()
        .await
        .expect("chat json");

    let msg = &resp["choices"][0]["message"];
    assert!(
        msg["content"].is_string() || msg["tool_calls"].is_array(),
        "response has content or tool_calls: {resp}"
    );
    if let Some(calls) = msg["tool_calls"].as_array() {
        assert!(!calls.is_empty(), "tool_calls non-empty when present");
        let f = &calls[0]["function"];
        assert_eq!(f["name"], "get_weather", "correct tool: {resp}");
        assert!(
            f["arguments"].is_string(),
            "OpenAI arguments are a JSON STRING: {resp}"
        );
        let content = msg["content"].as_str().unwrap_or("");
        assert!(
            !content.contains("<tool_call>") && !content.contains("<function"),
            "no call markup in content: {resp}"
        );
        eprintln!("E2E fleet: structured tool call parsed");
    } else {
        eprintln!("E2E fleet: no structured call this run (content path)");
    }

    // Teardown: retire the node so its routes + transport are dropped (the
    // drain-clean shutdown the original "unload all" step guaranteed), then stop
    // the /v1 server.
    fleet.retire(&peer).await;
    assert!(
        fleet.routed_models().await.is_empty(),
        "retire clears the fleet routes"
    );
    guard.shutdown().await;
}
