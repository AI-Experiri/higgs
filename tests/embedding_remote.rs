//! A remote embedding worker is refused over the REAL wire — end to end.
//!
//! A REAL `higgs --node` child loads a REAL embedding GGUF (bge-small — an
//! actual encoder with `bert.pooling_type=2` / `attention.causal=false`, which
//! real llama.cpp loads happily), then a chat against it is driven over the
//! hub's real `/v1` HTTP surface and must come back `400` carrying `[HG079]` —
//! not a `200` of sampled nonsense, which is what the ORIGINAL bug produced
//! over this exact wire.
//!
//! What a green run proves (r5 doc truth-up): since r3, `HubFleet::load`
//! refreshes the inventory, so the FIRST defense to fire here is the hub's own
//! pre-dispatch refusal (`resolve_loaded`'s remote arm reading the
//! node-reported inventory domain) — reverting the node's ChatHandle gate ALONE
//! does not fail this test. The node gate — the enforcement of last resort for
//! stale/absent hub knowledge — is pinned separately by the in-process fleet
//! test (`a_non_generative_remote_worker_is_unadvertised_but_still_refused_on_
//! the_wire`, which dispatches `fleet.chat` directly past the facade). This
//! test pins the STACK end to end against a real engine and real iroh.
//!
//! Skips when the real embedding GGUF is absent (`HIGGS_TEST_EMBED_GGUF`, else
//! the HF-cache bge-small path).

mod common;

use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use iroh_tickets::endpoint::EndpointTicket;

use higgs::auth::{Allowlist, PairingTokens};
use higgs::log_bus::LogBus;
use higgs::node::fleet::HubFleet;
use higgs::node::transport::NodeTransport;
use higgs::node::{gate_connection, GateOutcome, HubIdentity, HELLO_DEADLINE};
use higgs::remote::ALPN;
use higgs::{Higgs, HiggsConfig};

use common::{serve_v1_local, stage_models};

/// The id the real embedding GGUF is staged under on the NODE.
const EMBED_ID: &str = "higgs-test/bge-remote";

/// A real, loadable embedding GGUF: `HIGGS_TEST_EMBED_GGUF` when set, else the
/// HF-cache bge-small snapshot (the snapshot hash varies, so walk the dir).
fn real_embedding_gguf() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("HIGGS_TEST_EMBED_GGUF") {
        let p = PathBuf::from(p);
        return p.is_file().then_some(p);
    }
    let home = std::env::var("HOME").ok()?;
    let snapshots = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--unsloth--bge-small-en-v1.5-GGUF/snapshots");
    for snap in std::fs::read_dir(snapshots).ok()?.flatten() {
        for f in std::fs::read_dir(snap.path()).ok()?.flatten() {
            let p = f.path();
            if p.extension().is_some_and(|e| e == "gguf") && p.is_file() {
                return Some(p);
            }
        }
    }
    None
}

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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_remote_embedding_worker_refuses_relayed_chat() {
    let Some(gguf) = real_embedding_gguf() else {
        eprintln!(
            "SKIP a_remote_embedding_worker_refuses_relayed_chat: no real embedding GGUF \
             (set HIGGS_TEST_EMBED_GGUF)"
        );
        return;
    };
    let scan_root = stage_models(&gguf, &[EMBED_ID]);
    let node_home = tempfile::tempdir().expect("node home");
    let hub_home = tempfile::tempdir().expect("hub home");

    // Isolate the hub's HIGGS_HOME so `serve_v1` reads an EMPTY keystore (auth
    // off). Single-test binary — a bare `set_var` needs no restore.
    // SAFETY: no other thread touches the process env in this single-test binary.
    unsafe {
        std::env::set_var("HIGGS_HOME", hub_home.path());
    }

    // Hub: a local Higgs (no local models) with a HubFleet installed.
    let bus = Arc::new(LogBus::new());
    let higgs = Arc::new(Higgs::with_log_bus(HiggsConfig::default(), bus.clone()));
    let fleet = Arc::new(HubFleet::new(bus.clone()));
    higgs.set_fleet(fleet.clone());

    // Hub iroh endpoint + a pairing ticket/token (hermetic, relay-disabled).
    let hub = iroh::Endpoint::builder(iroh::endpoint::presets::Minimal)
        .alpns(vec![ALPN.to_vec()])
        .relay_mode(iroh::RelayMode::Disabled)
        .bind()
        .await
        .expect("bind hub endpoint");
    let hub_id = hub.id().to_string();
    let ticket = EndpointTicket::new(hub.addr()).to_string();
    let mut allow = Allowlist::load(&node_home.path().join("hub-pairings.json")).unwrap();
    let mut tokens = PairingTokens::new();
    let token = tokens.mint(now_ms(), 600_000);

    // The real node process, scanning the staged embedding model.
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

    // Admit + register the node.
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
        &HubIdentity::new(&hub_id),
        Some("test".into()),
        HELLO_DEADLINE,
    )
    .await;
    assert!(
        matches!(outcome, GateOutcome::Admitted { .. }),
        "admitted: {outcome:?}"
    );
    fleet
        .add_node(
            peer.clone(),
            Arc::new(NodeTransport::new(conn)),
            None,
            None,
            None,
            false,
            None,
            true,
        )
        .await;

    // Load the embedding model ON THE NODE (real llama.cpp; loads are allowed —
    // only chat is refused). This is what makes the test non-vacuous: the hub now
    // routes EMBED_ID remotely, and its own resolve_loaded goes permissive.
    fleet
        .load(&peer, EMBED_ID, None)
        .await
        .expect("remote load of the embedding model succeeds");
    assert!(
        fleet.is_remote(EMBED_ID).await,
        "the embedding model is remote-routable"
    );

    // Chat against it over the hub's real /v1 surface. The first defense to fire
    // is the hub's inventory-informed pre-dispatch refusal (see the module doc);
    // either layer of the stack must map to the same 400 [HG079].
    let (base, _guard) = serve_v1_local(higgs.clone()).await;
    let resp = reqwest::Client::new()
        .post(format!("{base}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": EMBED_ID,
            "stream": false,
            "max_tokens": 16,
            "messages": [ { "role": "user", "content": "hi" } ],
        }))
        .send()
        .await
        .expect("chat request");
    let status = resp.status();
    let body = resp.text().await.expect("body");

    assert_eq!(
        status, 400,
        "a relayed chat against a remote embedding worker must be refused, \
         not answered with sampled nonsense; body: {body}"
    );
    assert!(
        body.contains("[HG079]"),
        "the refusal carries the node's [HG079] code; body: {body}"
    );
}
