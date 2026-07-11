//! Remote NodeRuntime smoke: an in-process hub + a REAL spawned `higgs --node`
//! process, with a real LLM loaded ON THE NODE and a chat routed through the
//! `HubFleet` via `Higgs::chat_stream`. No HTTP. Proves the remote fleet path
//! end-to-end with a real model and coherent output.
//!
//! Run: `cargo build --bin higgs && cargo run --example smoke_remote`
//!   env SMOKE_REMOTE_MODEL=<id>  (default: the Nemotron LM-Studio id)
//!   env HIGGS_MODEL_DIR=<dir>    node's scan root (default: ~/.cache/lm-studio/models)
//!
//! Uses the SAME recipe as tests/remote_hub_e2e.rs (hermetic iroh, gate_connection,
//! HubFleet), but points the node at the real model dir instead of a staged copy.

use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use iroh_tickets::endpoint::EndpointTicket;

use higgs::auth::{Allowlist, PairingTokens};
use higgs::log_bus::LogBus;
use higgs::node::fleet::HubFleet;
use higgs::node::transport::NodeTransport;
use higgs::node::{gate_connection, GateOutcome, HubIdentity, HELLO_DEADLINE};
use higgs::remote::ALPN;
use higgs::{ChatDeltaKind, Higgs, HiggsConfig, SamplingParams};

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// SIGTERM the node on drop so it gracefully stops its llama.cpp worker.
struct NodeProc(Child);
impl Drop for NodeProc {
    fn drop(&mut self) {
        unsafe {
            libc::kill(self.0.id() as libc::pid_t, libc::SIGTERM);
        }
        let _ = self.0.wait();
    }
}

fn main() {
    // Belt-and-suspenders: this hub-side binary never loads the model locally
    // (it routes remote), but honor the worker role in case anything re-execs us.
    if std::env::args().skip(1).any(|a| a == "--higgs-worker") {
        higgs::worker::worker_main();
        return;
    }
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");
    rt.block_on(run());
}

async fn run() {
    let model = std::env::var("SMOKE_REMOTE_MODEL")
        .unwrap_or_else(|_| "lmstudio-community/NVIDIA-Nemotron-3-Nano-4B-GGUF".to_string());
    let model_dir = std::env::var("HIGGS_MODEL_DIR").unwrap_or_else(|_| {
        format!(
            "{}/.cache/lm-studio/models",
            std::env::var("HOME").unwrap_or_default()
        )
    });
    // Examples don't get CARGO_BIN_EXE_higgs — derive the built binary from our
    // own path: target/debug/examples/smoke_remote → target/debug/higgs.
    let higgs_bin = std::env::current_exe()
        .expect("current_exe")
        .parent()
        .and_then(|p| p.parent())
        .expect("target dir")
        .join("higgs");
    if !higgs_bin.exists() {
        eprintln!("higgs binary not found at {higgs_bin:?} — run `cargo build --bin higgs` first");
        return;
    }

    // Hub: a local Higgs with NO local models + a HubFleet installed.
    let bus = Arc::new(LogBus::new());
    let higgs = Arc::new(Higgs::with_log_bus(HiggsConfig::default(), bus.clone()));
    let fleet = Arc::new(HubFleet::new(bus.clone()));
    higgs.set_fleet(fleet.clone());

    // Hub iroh endpoint (hermetic: relay disabled) + a one-time pairing ticket/token.
    let hub = iroh::Endpoint::builder(iroh::endpoint::presets::Minimal)
        .alpns(vec![ALPN.to_vec()])
        .relay_mode(iroh::RelayMode::Disabled)
        .bind()
        .await
        .expect("bind hub endpoint");
    let hub_id = hub.id().to_string();
    let ticket = EndpointTicket::new(hub.addr()).to_string();
    let node_home = tempfile::tempdir().expect("node home");
    let mut allow = Allowlist::load(&node_home.path().join("hub-pairings.json")).unwrap();
    let mut tokens = PairingTokens::new();
    let token = tokens.mint(now_ms(), 600_000);

    println!("══════════════ REMOTE NodeRuntime SMOKE ══════════════");
    println!("spawning `higgs --node` (scan dir: {model_dir}) …");
    let child = Command::new(&higgs_bin)
        .arg("--node")
        .arg(&ticket)
        .arg(&token)
        .env("HIGGS_HOME", node_home.path())
        .env("HIGGS_MODEL_DIR", &model_dir)
        .env("HIGGS_IROH_LOCAL", "1") // must match hub RelayMode::Disabled
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn higgs --node");
    let _node = NodeProc(child);

    // Accept the node's dial, gate it, register it in the fleet.
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
        Some("smoke".into()),
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
    println!("node admitted: {peer}");

    // Load the real model ON THE NODE via the fleet → records the remote route.
    println!("loading {model} on the remote node …");
    let t_load = Instant::now();
    if let Err(e) = fleet.load(&peer, &model, None).await {
        println!("❌ remote load failed: {e}");
        return;
    }
    println!("remote load ok in {:.1}s", t_load.elapsed().as_secs_f64());

    // Confirm it's remote and NOT locally resident (so chat_stream MUST route remote).
    let local = higgs.local_served_ids().await;
    let is_remote = fleet.is_remote(&model).await;
    println!(
        "routing check → is_remote={is_remote}, locally_resident={}",
        local.contains(&model)
    );
    assert!(
        is_remote && !local.contains(&model),
        "model routes remote, not local"
    );

    // The hub's chat path: chat_stream routes the remote-resident model through the
    // fleet to the node and streams real tokens back.
    let msgs = serde_json::json!([
        {"role": "user", "content": "Reply with a short one-sentence friendly greeting."}
    ])
    .to_string();
    let t_gen = Instant::now();
    let (mut rx, handle) = match higgs
        .chat_stream(
            model.clone(),
            msgs,
            512,
            SamplingParams::default(),
            None,
            None,
        )
        .await
    {
        Ok(pair) => pair,
        Err(e) => {
            println!("❌ chat_stream: {e}");
            return;
        }
    };

    let mut content = String::new();
    let mut reasoning = String::new();
    while let Some(d) = rx.recv().await {
        match d.kind {
            ChatDeltaKind::Content => content.push_str(&d.text),
            ChatDeltaKind::Reasoning => reasoning.push_str(&d.text),
            ChatDeltaKind::ToolCall => {}
        }
    }
    match handle.await {
        Ok(Ok(o)) => {
            let secs = t_gen.elapsed().as_secs_f64();
            let tps = if secs > 0.0 {
                o.completion_tokens as f64 / secs
            } else {
                0.0
            };
            let ans = if o.content.trim().is_empty() {
                content.trim()
            } else {
                o.content.trim()
            };
            if !reasoning.trim().is_empty() {
                let snip: String = reasoning.trim().chars().take(160).collect();
                println!("  💭 thought: {snip}…");
            }
            println!("  💬 answer (via REMOTE node): {ans}");
            println!(
                "  ✅ PASS  {} tokens in {:.1}s ({:.1} tok/s), finish={}",
                o.completion_tokens, secs, tps, o.finish_reason
            );
        }
        Ok(Err(e)) => println!("  ❌ generation error: {e}"),
        Err(e) => println!("  ❌ chat task join error: {e}"),
    }

    higgs.stop().await;
    println!("Done (node shutting down).");
}
