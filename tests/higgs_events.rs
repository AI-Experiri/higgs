//! Black-box integration test for the `GET /api/higgs/events` SSE stream.
//!
//! Spawns the real `higgs` binary against the tiny on-disk model, opens the
//! model-load lifecycle SSE stream, then triggers a real load over HTTP and
//! asserts the pushed phases arrive in order: `queued` → `preparing` →
//! `loading_weights` → `finalizing` → `ready`. This proves the loading indicator
//! is driven by PUSH events (no polling) end-to-end.
//!
//! Fail-on-revert: if the phase emits are removed from `load_inner` (or the route
//! is unregistered) the stream yields no phases and the ordered-sequence assert
//! fails.

mod common;

use common::{spawn_with_tiny_model, tiny_gguf_path, TINY_MODEL_ID};
use futures::StreamExt;
use std::time::Duration;

/// Read `data:` frames off the SSE byte stream, decode each as a load event, and
/// return the ordered `phase` strings until `ready`/`failed` (or a timeout).
async fn collect_phases(resp: reqwest::Response) -> Vec<String> {
    let mut phases = Vec::new();
    let mut buf = String::new();
    let mut stream = resp.bytes_stream();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
    loop {
        let chunk = match tokio::time::timeout_at(deadline, stream.next()).await {
            Ok(Some(Ok(chunk))) => chunk, // a byte chunk
            _ => break,                   // timeout, end of stream, or stream error
        };
        buf.push_str(&String::from_utf8_lossy(&chunk));
        // SSE frames are separated by a blank line; each `data:` line is one JSON event.
        while let Some(nl) = buf.find('\n') {
            let line = buf[..nl].trim().to_owned();
            buf.drain(..=nl);
            let Some(json) = line.strip_prefix("data:") else {
                continue;
            };
            let ev: serde_json::Value = match serde_json::from_str(json.trim()) {
                Ok(v) => v,
                Err(_) => continue,
            };
            if let Some(phase) = ev["phase"].as_str() {
                phases.push(phase.to_owned());
                if phase == "ready" || phase == "failed" {
                    return phases;
                }
            }
        }
    }
    phases
}

#[tokio::test]
async fn events_stream_pushes_load_phases_in_order() {
    let Some(gguf) = tiny_gguf_path() else {
        eprintln!("SKIP events_stream: tiny gguf not found (set HIGGS_TEST_GGUF)");
        return;
    };
    let srv = spawn_with_tiny_model(11540, &gguf).await;
    let c = reqwest::Client::new();

    // Open the SSE stream BEFORE the load so no phase is missed (there is no replay).
    let resp = c
        .get(format!("{}/api/higgs/events", srv.base))
        .send()
        .await
        .expect("open /api/higgs/events");
    assert!(resp.status().is_success(), "events stream opens");

    // Trigger a real load in the background while we read the stream.
    let base = srv.base.clone();
    let id = TINY_MODEL_ID;
    let loader = tokio::spawn(async move {
        reqwest::Client::new()
            .post(format!("{base}/api/higgs/models/load"))
            // Belt-and-braces above the server's own 120s control timeout: a
            // fully wedged server must fail this test in minutes, never hang it.
            .timeout(Duration::from_secs(150))
            .json(&serde_json::json!({ "id": id }))
            .send()
            .await
            .expect("load request")
            .json::<serde_json::Value>()
            .await
            .expect("load response json")
    });

    let phases = collect_phases(resp).await;
    let load = loader.await.expect("loader task");
    assert_eq!(load["status"], "ok", "load succeeded: {load}");

    // Every real progress phase must appear, terminated by `ready`.
    for want in [
        "queued",
        "preparing",
        "loading_weights",
        "finalizing",
        "ready",
    ] {
        assert!(
            phases.iter().any(|p| p == want),
            "missing phase {want:?} in {phases:?}"
        );
    }
    // …and in order (subsequence check: each appears after the previous).
    let order = [
        "queued",
        "preparing",
        "loading_weights",
        "finalizing",
        "ready",
    ];
    let mut idx = 0usize;
    for p in &phases {
        if idx < order.len() && p == order[idx] {
            idx += 1;
        }
    }
    assert_eq!(idx, order.len(), "phases not in expected order: {phases:?}");
}
