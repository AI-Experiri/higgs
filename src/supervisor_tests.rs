
use super::*;
use crate::worker::M_CHAT;
use serde_json::json;

// ── Test seam ─────────────────────────────────────────────────────────────
//
// `tokio::io::duplex(N)` yields a bidirectional pair.  We need two
// independent pairs:
//   sup_write → test_read   (supervisor writes requests; test reads them)
//   test_write → sup_read   (test writes responses; supervisor reads them)
//
// The factory returns `WorkerHalves { write: sup_write, read: sup_read }`.
// The test controls `test_write` (inject responses) and `test_read` (observe requests).

/// Build a supervisor plus test control handles.
///
/// Returns `(supervisor, test_write, test_read)`.
/// - `test_write`: write responses/notifications here (supervisor reads them).
/// - `test_read`:  read requests the supervisor sent to the "worker".
fn make_supervisor() -> (
    Supervisor,
    tokio::io::DuplexStream, // test writes → supervisor reads
    tokio::io::DuplexStream, // supervisor writes → test reads
) {
    // Pair A: supervisor write half ←→ test_read
    let (sup_write, test_read) = tokio::io::duplex(64 * 1024);
    // Pair B: test_write ←→ supervisor read half
    let (test_write, sup_read) = tokio::io::duplex(64 * 1024);

    // Wrap both halves in Arc<Mutex<Option<…>>> so the factory closure
    // (Fn, not FnOnce) can hand them out exactly once.
    let sup_write_cell = Arc::new(Mutex::new(Some(sup_write)));
    let sup_read_cell = Arc::new(Mutex::new(Some(sup_read)));

    let sup = Supervisor::with_factory(Box::new(move |_ring, _model| {
        let write = sup_write_cell
            .lock()
            .take()
            .ok_or_else(|| HiggsError::WorkerSpawnFailed {
                source: std::io::Error::other("mock: no more write halves"),
            })?;
        let read = sup_read_cell
            .lock()
            .take()
            .ok_or_else(|| HiggsError::WorkerSpawnFailed {
                source: std::io::Error::other("mock: no more read halves"),
            })?;
        Ok(WorkerHalves {
            write: Box::new(write),
            read: Box::new(read),
            proc: None,
        })
    }));

    sup.start_for("test-model").expect("mock start failed");

    (sup, test_write, test_read)
}

fn ok_response(id: u64, result: Value) -> String {
    rpc::encode(&RpcFrame::Response(rpc::RpcResponse {
        jsonrpc: "2.0".into(),
        id,
        result: Some(result),
        error: None,
    }))
}

fn err_response(id: u64, code: i64, message: &str) -> String {
    rpc::encode(&RpcFrame::Response(rpc::RpcResponse {
        jsonrpc: "2.0".into(),
        id,
        result: None,
        error: Some(rpc::RpcError {
            code,
            message: message.into(),
            data: None,
        }),
    }))
}

fn chunk_notif(request_id: u64, delta: &str) -> String {
    rpc::encode(&RpcFrame::Notification(rpc::RpcNotification {
        jsonrpc: "2.0".into(),
        method: N_CHAT_CHUNK.into(),
        params: json!({"request_id": request_id, "delta": delta}),
    }))
}

/// Write a line into `stream`, appending `\n`, and flush.
async fn write_line(stream: &mut tokio::io::DuplexStream, line: &str) {
    use tokio::io::AsyncWriteExt;
    stream
        .write_all(format!("{line}\n").as_bytes())
        .await
        .expect("test write_line");
    stream.flush().await.expect("test flush");
}

// ─── Test 1: out-of-order response correlation ───────────────────────────

#[tokio::test]
async fn request_response_correlation() {
    let (sup, mut test_write, _test_read) = make_supervisor();

    // Issue two requests concurrently. The supervisor assigns ids 1 and 2.
    let fut1 = sup.request("higgs/ping", json!({"n": 1}));
    let fut2 = sup.request("higgs/ping", json!({"n": 2}));

    // Give both requests time to register their pending entries.
    tokio::time::sleep(std::time::Duration::from_millis(20)).await;

    // Respond to id 2 first, then id 1.
    write_line(&mut test_write, &ok_response(2, json!({"n": 2}))).await;
    write_line(&mut test_write, &ok_response(1, json!({"n": 1}))).await;

    let (r1, r2) = tokio::join!(fut1, fut2);
    assert_eq!(r1.unwrap(), json!({"n": 1}));
    assert_eq!(r2.unwrap(), json!({"n": 2}));
}

// ─── Test 1-a: spawn-on-start_for then kill-on-stop lifecycle ────────────
//
// No worker is live until start_for; a request then correlates. After stop()
// the write_tx is cleared so a subsequent request fails WorkerDead — proving
// stop() tears the worker down (kill-on-unload at the supervisor layer).

#[tokio::test]
async fn start_for_then_stop_lifecycle() {
    let (sup_write, _test_read) = tokio::io::duplex(64 * 1024);
    let (mut test_write, sup_read) = tokio::io::duplex(64 * 1024);
    let sup_write_cell = Arc::new(Mutex::new(Some(sup_write)));
    let sup_read_cell = Arc::new(Mutex::new(Some(sup_read)));

    let sup = Supervisor::with_factory(Box::new(move |_ring, _model| {
        Ok(WorkerHalves {
            write: Box::new(sup_write_cell.lock().take().expect("one spawn")),
            read: Box::new(sup_read_cell.lock().take().expect("one spawn")),
            proc: None,
        })
    }));

    // Before start_for: no write_tx → request fails immediately.
    assert!(
        sup.request("higgs/ping", json!({})).await.is_err(),
        "no worker before start_for"
    );

    // start_for spawns a worker; a request now correlates. The pre-spawn
    // request already consumed id 1, so this one is id 2.
    sup.start_for("org/model").expect("spawn");
    let fut = sup.request("higgs/ping", json!({}));
    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    write_line(&mut test_write, &ok_response(2, json!({"ok": true}))).await;
    assert_eq!(fut.await.unwrap(), json!({"ok": true}));

    // stop() kills the worker; a later request fails WorkerDead.
    sup.stop().await;
    let err = sup.request("higgs/ping", json!({})).await.unwrap_err();
    assert!(err.to_string().contains("[HG007]"), "display: {err}");
}

// A spawn bumps the worker-lifetime generation (the tag the stale-reader guard compares
// against), and a redundant start() while already running does NOT bump it.
#[tokio::test]
async fn spawn_bumps_generation_redundant_start_does_not() {
    // make_supervisor already called start_for once, so the spawn bumped gen 0 → 1.
    let (sup, _tw, _tr) = make_supervisor();
    assert_eq!(
        sup.inner.generation.load(Ordering::Acquire),
        1,
        "spawn bumped gen to 1"
    );
    // Redundant start() while running is an idempotent no-op (running.swap guard) — it
    // must NOT bump the generation or the live reader would be wrongly marked stale.
    sup.start_for("org/model").expect("redundant start ok");
    assert_eq!(
        sup.inner.generation.load(Ordering::Acquire),
        1,
        "redundant start does not bump generation"
    );
    sup.stop().await;
}

// A crash-replay must SKIP if an explicit load/unload superseded the captured model
// (load_epoch bumped) — otherwise a stale replay would resurrect the old model over the
// newer one. Drive replay_load directly, bump the epoch before the spawned task runs
// (current-thread runtime: it can't run until we await), and assert NO M_LOAD is sent.
#[tokio::test]
async fn replay_load_skips_when_epoch_superseded() {
    let (sup, _tw, test_read) = make_supervisor(); // worker running (gen 1)
    sup.record_last_load(json!({ "id": "org/a", "path": "/x" }));
    // replay_load captures the current epoch synchronously, then spawns the replay task.
    replay_load(&sup.inner);
    // Supersede before the task runs: a newer explicit load/unload bumped the epoch.
    sup.inner.load_epoch.fetch_add(1, Ordering::AcqRel);
    // The replay task now observes a changed epoch and returns without sending M_LOAD.
    let mut lines = BufReader::new(test_read).lines();
    let got = tokio::time::timeout(std::time::Duration::from_millis(200), lines.next_line()).await;
    match got {
        // Timed out / EOF / error → nothing was sent → replay correctly skipped.
        Err(_) | Ok(Ok(None)) | Ok(Err(_)) => {}
        // A line was sent: it must NOT be an M_LOAD (that would be the stale replay).
        Ok(Ok(Some(line))) => {
            assert!(
                !line.contains("higgs/load"),
                "superseded replay sent M_LOAD: {line}"
            );
        }
    }
    sup.stop().await;
}

// ─── Test 1-b: redundant start() is an idempotent no-op ──────────────────
//
// The mock factory hands out exactly one set of duplex halves, so a second
// real spawn would fail (`mock: no more write halves`). The `running` guard
// means a `start()` on a live worker is never a second spawn — it returns
// Ok without touching the factory, preserving the single reader and the
// original transport. Guards the supervisor.rs:225 "start while running"
// race that would otherwise let an old reader clear the new write_tx.

#[tokio::test]
async fn redundant_start_is_noop() {
    let (sup, mut test_write, _test_read) = make_supervisor();

    // Already started by make_supervisor; a second start must not spawn.
    sup.start_for("test-model")
        .expect("redundant start is a no-op, not a second spawn");

    // The original worker transport is intact: a request still correlates.
    let fut = sup.request("higgs/ping", json!({}));
    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    write_line(&mut test_write, &ok_response(1, json!({"ok": true}))).await;
    assert_eq!(fut.await.unwrap(), json!({"ok": true}));
}

// ─── Test 2: chat-chunk routing (keyed) ──────────────────────────────────

#[tokio::test]
async fn chat_chunks_routed() {
    let (sup, mut test_write, _test_read) = make_supervisor();

    // Register a keyed sink; request_id=42 matches the notification.
    let mut rx = sup.register_chat_sink(42);

    let deltas = ["hello", " world", "!"];
    for d in &deltas {
        write_line(&mut test_write, &chunk_notif(42, d)).await;
    }
    tokio::time::sleep(std::time::Duration::from_millis(30)).await;

    for expected in &deltas {
        let got = rx.try_recv().expect("delta expected");
        assert_eq!(got, *expected);
    }

    sup.remove_chat_sink(42);
}

// ─── Test 2-b: two keyed sinks route independently ───────────────────────
//
// Registers sinks for request_id 1 and 2; feeds N_CHAT_CHUNK notifications
// for each; asserts each receiver gets ONLY its own deltas in order.

#[tokio::test]
async fn two_keyed_sinks_route_independently() {
    let (sup, mut test_write, _test_read) = make_supervisor();

    let mut rx1 = sup.register_chat_sink(1);
    let mut rx2 = sup.register_chat_sink(2);

    // Feed deltas for request_id 2 first, then request_id 1.
    write_line(&mut test_write, &chunk_notif(2, "alpha")).await;
    write_line(&mut test_write, &chunk_notif(1, "beta")).await;
    write_line(&mut test_write, &chunk_notif(2, "gamma")).await;
    write_line(&mut test_write, &chunk_notif(1, "delta")).await;

    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    // rx1 must only see deltas tagged request_id=1.
    let r1_a = rx1.try_recv().expect("rx1 first chunk");
    let r1_b = rx1.try_recv().expect("rx1 second chunk");
    assert_eq!(r1_a, "beta");
    assert_eq!(r1_b, "delta");
    assert!(rx1.try_recv().is_err(), "rx1 must have no more chunks");

    // rx2 must only see deltas tagged request_id=2.
    let r2_a = rx2.try_recv().expect("rx2 first chunk");
    let r2_b = rx2.try_recv().expect("rx2 second chunk");
    assert_eq!(r2_a, "alpha");
    assert_eq!(r2_b, "gamma");
    assert!(rx2.try_recv().is_err(), "rx2 must have no more chunks");

    sup.remove_chat_sink(1);
    sup.remove_chat_sink(2);
}

// ─── Test 3: EOF fails pending + emits WorkerDied ────────────────────────

#[tokio::test]
async fn eof_fails_pending_and_emits_died() {
    // Factory: first call succeeds (using duplex), second call fails (no more halves).
    let (sup_write_1, test_read_1) = tokio::io::duplex(64 * 1024);
    let (test_write_1, sup_read_1) = tokio::io::duplex(64 * 1024);

    let sup_write_cell = Arc::new(Mutex::new(Some(sup_write_1)));
    let sup_read_cell = Arc::new(Mutex::new(Some(sup_read_1)));

    let sup = Supervisor::with_factory(Box::new(move |_ring, _model| {
        let write = sup_write_cell
            .lock()
            .take()
            .ok_or_else(|| HiggsError::WorkerSpawnFailed {
                source: std::io::Error::other("mock: no more halves"),
            })?;
        let read = sup_read_cell
            .lock()
            .take()
            .ok_or_else(|| HiggsError::WorkerSpawnFailed {
                source: std::io::Error::other("mock: no more halves"),
            })?;
        Ok(WorkerHalves {
            write: Box::new(write),
            read: Box::new(read),
            proc: None,
        })
    }));
    sup.start_for("test-model").expect("start");

    let mut events = sup.events();

    // Register a pending request directly (bypass the network send so
    // the pending entry exists before EOF arrives).
    let pending_id = sup.inner.next_id.fetch_add(1, Ordering::Relaxed);
    let reply_rx = sup.inner.demux.register_pending(pending_id);

    // Give the reader task time to start.
    tokio::time::sleep(std::time::Duration::from_millis(20)).await;

    // Trigger EOF by dropping the test write end.
    drop(test_write_1);
    drop(test_read_1);

    // The pending oneshot should be dropped (channel closed → Err).
    let result = tokio::time::timeout(std::time::Duration::from_millis(500), reply_rx).await;
    assert!(
        matches!(result, Ok(Err(_))),
        "pending request should fail on EOF"
    );

    // First WorkerDied must arrive (worker EOF).
    let first_died = tokio::time::timeout(std::time::Duration::from_millis(2000), async {
        loop {
            match events.recv().await {
                Ok(HiggsEvent::WorkerDied) => return true,
                Ok(_) => continue,
                Err(_) => return false,
            }
        }
    })
    .await;
    assert!(
        matches!(first_died, Ok(true)),
        "first WorkerDied event expected (EOF)"
    );

    // After the 1s respawn backoff, the factory fails (no more halves) →
    // a second terminal WorkerDied must be broadcast before the reader task exits.
    let second_died = tokio::time::timeout(std::time::Duration::from_millis(2000), async {
        loop {
            match events.recv().await {
                Ok(HiggsEvent::WorkerDied) => return true,
                Ok(_) => continue,
                Err(_) => return false,
            }
        }
    })
    .await;
    assert!(
        matches!(second_died, Ok(true)),
        "second WorkerDied event expected (factory failure)"
    );
}

// ─── Test 3-b: chat RPC times out → HG016 ChatTimeout ────────────────────
//
// Uses a paused clock so the 600 s CHAT_RPC_TIMEOUT elapses instantly. The
// worker never responds; request_with_id must remove its pending entry and
// return ChatTimeout (→ 504), not hang.

#[tokio::test(start_paused = true)]
async fn chat_rpc_times_out() {
    let (sup, _test_write, _test_read) = make_supervisor();
    // Pre-allocate the id the way the chat path does.
    let id = sup.alloc_request_id();
    let fut = sup.request_with_id(id, M_CHAT, json!({"request_id": id}));
    // Advance past the chat-RPC timeout with no response written.
    tokio::time::advance(CHAT_RPC_TIMEOUT + std::time::Duration::from_secs(1)).await;
    let err = fut.await.expect_err("must time out");
    assert!(matches!(err, HiggsError::ChatTimeout { .. }), "got {err}");
    assert!(err.to_string().starts_with("[HG016]"));
    // The orphaned pending entry was removed (no leak).
    assert_eq!(sup.inner.demux.pending_count(), 0);
}

// ─── Test 4: worker RPC error maps to HG009 ──────────────────────────────

#[tokio::test]
async fn worker_error_maps_to_hg009() {
    let (sup, mut test_write, _test_read) = make_supervisor();

    let fut = sup.request(M_LOAD, json!({"id": "org/bad"}));
    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    write_line(
        &mut test_write,
        &err_response(1, -32000, "model file corrupt"),
    )
    .await;

    let err = fut.await.expect_err("should be an error");
    let display = err.to_string();
    assert!(display.contains("[HG009]"), "display: {display}");
}

// ─── Test 5: stderr ring caps at 2000 ────────────────────────────────────

#[tokio::test]
async fn logs_ring_caps() {
    // The supervisor's log history caps at the LogBus ring capacity (2000);
    // the production path fills it via the stderr drain task. Drive the same
    // bus the supervisor exposes and confirm the cap + oldest-first tail.
    let (sup, _tw, _tr) = make_supervisor();
    for i in 0..2100usize {
        sup.inner.bus.push(LogSource::Worker, format!("line-{i}"));
    }
    let snap = sup.logs(usize::MAX, None);
    assert_eq!(snap.len(), 2000);
    // 2100 pushed, 100 dropped → oldest remaining is line-100.
    assert_eq!(snap.first().unwrap(), "line-100");
}

// ─── Test 6: restart replays the load (scan is host-side) + emits ModelLoaded ─
//
// The mock factory hands out two transport pairs:
//   pair-1: first worker lifetime (killed by dropping test_write_1)
//   pair-2: second worker lifetime (receives replayed RPCs; the test drives
//           it by reading the replayed higgs/load and writing back an OK)
//
// Duplex wiring — each pair (A, B) means writing to A comes out of B:
//   sup_write_1  ↔ _obs_rx_1  : supervisor writes requests; test ignores them
//   test_write_1 ↔ sup_read_1 : drop triggers EOF on supervisor's read half
//   sup_write_2  ↔ obs_rx_2   : supervisor writes replayed msgs; test reads here
//   test_tx_2    ↔ sup_read_2 : test writes mock OK responses back to supervisor

#[tokio::test]
async fn restart_replays_load() {
    let (sup_write_1, _obs_rx_1) = tokio::io::duplex(64 * 1024);
    let (test_write_1, sup_read_1) = tokio::io::duplex(64 * 1024);

    let (sup_write_2, mut obs_rx_2) = tokio::io::duplex(64 * 1024);
    let (mut test_tx_2, sup_read_2) = tokio::io::duplex(64 * 1024);

    let cell_sup_write_1 = Arc::new(Mutex::new(Some(sup_write_1)));
    let cell_sup_read_1 = Arc::new(Mutex::new(Some(sup_read_1)));
    let cell_sup_write_2 = Arc::new(Mutex::new(Some(sup_write_2)));
    let cell_sup_read_2 = Arc::new(Mutex::new(Some(sup_read_2)));

    let call_count = Arc::new(std::sync::atomic::AtomicU32::new(0));
    let call_count2 = Arc::clone(&call_count);

    let sup =
        Supervisor::with_factory(Box::new(move |_ring, _model| {
            let n = call_count2.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if n == 0 {
                let write = cell_sup_write_1.lock().take().unwrap();
                let read = cell_sup_read_1.lock().take().unwrap();
                Ok(WorkerHalves {
                    write: Box::new(write),
                    read: Box::new(read),
                    proc: None,
                })
            } else {
                let write = cell_sup_write_2.lock().take().ok_or_else(|| {
                    HiggsError::WorkerSpawnFailed {
                        source: std::io::Error::other(
                            "mock: second factory call after cells exhausted",
                        ),
                    }
                })?;
                let read =
                    cell_sup_read_2
                        .lock()
                        .take()
                        .ok_or_else(|| HiggsError::WorkerSpawnFailed {
                            source: std::io::Error::other(
                                "mock: second factory call after cells exhausted",
                            ),
                        })?;
                Ok(WorkerHalves {
                    write: Box::new(write),
                    read: Box::new(read),
                    proc: None,
                })
            }
        }));
    sup.start_for("test-model").expect("start");

    let mut events = sup.events();

    // Record load so replay has something to send. Scan is host-side now —
    // no scan replay; only the load is re-driven after restart.
    sup.record_last_load(json!({"id": "org/model"}));

    // Wait for the reader task to settle, then trigger EOF on first transport.
    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    drop(test_write_1);

    // After 1s backoff + restart, the second transport receives the replayed
    // higgs/load (awaited). obs_rx_2 is the duplex peer of sup_write_2 — data
    // written by the supervisor's writer task comes out here. The test reads
    // the load RPC then replies OK so ModelLoaded is emitted.
    let deadline = std::time::Duration::from_millis(3000);

    use tokio::io::AsyncBufReadExt;
    let load_line = tokio::time::timeout(deadline, async {
        let mut lines = BufReader::new(&mut obs_rx_2).lines();
        lines
            .next_line()
            .await
            .unwrap()
            .expect("load line expected")
    })
    .await
    .expect("timeout waiting for replayed load message");

    let load: serde_json::Value = serde_json::from_str(&load_line).expect("valid json");
    assert_eq!(load["method"], M_LOAD, "replayed method must be higgs/load");
    assert_eq!(load["params"], json!({"id": "org/model"}));

    // Reply to the replayed higgs/load so ModelLoaded is emitted.
    let load_id = load["id"].as_u64().expect("load request must carry an id");
    let ok_line = ok_response(load_id, json!({"id": "org/model"}));
    write_line(&mut test_tx_2, &ok_line).await;

    // ModelLoaded must arrive on the event channel.
    let got_loaded = tokio::time::timeout(std::time::Duration::from_millis(500), async {
        loop {
            match events.recv().await {
                Ok(HiggsEvent::ModelLoaded { id }) => return Some(id),
                Ok(_) => continue,
                Err(_) => return None,
            }
        }
    })
    .await
    .expect("timeout waiting for ModelLoaded");

    assert_eq!(
        got_loaded.as_deref(),
        Some("org/model"),
        "ModelLoaded must carry the replayed model id"
    );
}

/// Residual-TOCTOU close (#1): when `stopped` flips between the restart's
/// install guard and the async replay, `replay_load` must abort — no M_LOAD
/// frame may reach the (respawned) worker, since the user deliberately
/// stopped/unloaded. Drives `replay_load` directly with `stopped` already
/// set and asserts nothing is written to the worker transport.
#[tokio::test]
async fn replay_aborts_when_stopped_flips() {
    let (sup, _test_write, test_read) = make_supervisor();

    // A load is recorded (as after a normal load) so replay HAS something to
    // send — proving the abort is driven by `stopped`, not an empty replay.
    sup.record_last_load(json!({"id": "org/model"}));
    // Deliberate stop/unload flipped the flag in the TOCTOU window.
    sup.inner.stopped.store(true, Ordering::Release);

    // Fire the replay: the spawned task re-checks `stopped` before the M_LOAD.
    replay_load(&sup.inner);

    // Give the spawned task time to run (and to NOT send anything).
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    // No frame must have been written to the worker. Read with a short
    // timeout: a clean abort means the read times out (nothing arrived).
    use tokio::io::AsyncBufReadExt;
    let mut lines = BufReader::new(test_read).lines();
    let got = tokio::time::timeout(std::time::Duration::from_millis(100), lines.next_line()).await;
    assert!(
        got.is_err(),
        "replay must send NO M_LOAD when stopped flipped, but a frame arrived"
    );
}

/// `logs(n)` returns the tail of the stderr ring, oldest first, and clamps
/// to the ring length when fewer than `n` lines are present.
#[tokio::test]
async fn logs_tail_and_clamp() {
    let (sup, _tw, _tr) = make_supervisor();
    sup.inner.bus.push(LogSource::Worker, "a".to_owned());
    sup.inner.bus.push(LogSource::Worker, "b".to_owned());
    sup.inner.bus.push(LogSource::Worker, "c".to_owned());
    // Tail of 2 → last two, oldest first.
    assert_eq!(sup.logs(2, None), vec!["b".to_owned(), "c".to_owned()]);
    // n larger than the ring → all lines, no panic.
    assert_eq!(
        sup.logs(100, None),
        vec!["a".to_owned(), "b".to_owned(), "c".to_owned()]
    );
    // n == 0 → empty.
    assert!(sup.logs(0, None).is_empty());
}

/// `record_last_load` persists replay params, and `alloc_request_id` hands
/// out strictly increasing ids.
#[tokio::test]
async fn record_params_and_alloc_ids() {
    let (sup, _tw, _tr) = make_supervisor();
    assert!(sup.last_load_params().is_none(), "no load recorded yet");

    sup.record_last_load(json!({"id": "org/model"}));
    assert_eq!(sup.last_load_params(), Some(json!({"id": "org/model"})));

    let a = sup.alloc_request_id();
    let b = sup.alloc_request_id();
    assert!(b > a, "ids strictly increase: {a} then {b}");
}

/// `register_chat_sink` then `remove_chat_sink` adds and releases the keyed
/// map entry; the dropped sender closes the receiver (end-of-stream).
#[tokio::test]
async fn chat_sink_register_and_remove() {
    let (sup, _tw, _tr) = make_supervisor();
    assert_eq!(sup.chat_sinks_count(), 0);

    let mut rx = sup.register_chat_sink(42);
    assert_eq!(sup.chat_sinks_count(), 1);

    sup.remove_chat_sink(42);
    assert_eq!(sup.chat_sinks_count(), 0);
    // The dropped sender closes the receiver → recv yields None.
    assert!(rx.recv().await.is_none(), "removed sink closes the stream");
}

/// A `route_notification` with an unrelated method, a missing `request_id`,
/// or a missing `delta` is a silent no-op (no sink delivery, no panic).
#[tokio::test]
async fn route_notification_ignores_malformed() {
    let (sup, _tw, _tr) = make_supervisor();
    let mut rx = sup.register_chat_sink(7);

    // Wrong method.
    route_notification(
        &sup.inner,
        &rpc::RpcNotification {
            jsonrpc: "2.0".into(),
            method: "higgs/other".into(),
            params: json!({"request_id": 7, "delta": "x"}),
        },
    );
    // Right method but no request_id.
    route_notification(
        &sup.inner,
        &rpc::RpcNotification {
            jsonrpc: "2.0".into(),
            method: N_CHAT_CHUNK.into(),
            params: json!({"delta": "x"}),
        },
    );
    // Right method, request_id present, but no delta.
    route_notification(
        &sup.inner,
        &rpc::RpcNotification {
            jsonrpc: "2.0".into(),
            method: N_CHAT_CHUNK.into(),
            params: json!({"request_id": 7}),
        },
    );

    // None of the above delivered anything to the sink.
    assert!(
        rx.try_recv().is_err(),
        "malformed notifications deliver nothing"
    );

    // A well-formed one delivers.
    route_notification(
        &sup.inner,
        &rpc::RpcNotification {
            jsonrpc: "2.0".into(),
            method: N_CHAT_CHUNK.into(),
            params: json!({"request_id": 7, "delta": "hi"}),
        },
    );
    assert_eq!(rx.try_recv().unwrap(), "hi");
}

// ─── Transient-worker factory helper (sysinfo) ───────────────────────────
//
// `sysinfo` calls the factory directly for a SEPARATE, transient worker (no
// `start_for`, no persistent reader/writer). This builds a supervisor whose
// factory hands out exactly one duplex pair and returns the test control
// handles for that transient worker.
//
//   sup_write ↔ test_read  : supervisor writes the M_SYSINFO request
//   test_write ↔ sup_read  : test writes the worker's response back
fn transient_supervisor() -> (
    Supervisor,
    tokio::io::DuplexStream, // test writes → supervisor reads
    tokio::io::DuplexStream, // supervisor writes → test reads
) {
    let (sup_write, test_read) = tokio::io::duplex(64 * 1024);
    let (test_write, sup_read) = tokio::io::duplex(64 * 1024);
    let sup_write_cell = Arc::new(Mutex::new(Some(sup_write)));
    let sup_read_cell = Arc::new(Mutex::new(Some(sup_read)));
    let sup = Supervisor::with_factory(Box::new(move |_ring, _model| {
        Ok(WorkerHalves {
            write: Box::new(sup_write_cell.lock().take().ok_or_else(|| {
                HiggsError::WorkerSpawnFailed {
                    source: std::io::Error::other("mock: no more write halves"),
                }
            })?),
            read: Box::new(sup_read_cell.lock().take().ok_or_else(|| {
                HiggsError::WorkerSpawnFailed {
                    source: std::io::Error::other("mock: no more read halves"),
                }
            })?),
            proc: None,
        })
    }));
    (sup, test_write, test_read)
}

/// Build a supervisor whose factory always fails — used to drive the
/// spawn-failure verdict paths in `sysinfo` and `start_for`/`do_spawn`.
fn failing_supervisor() -> Supervisor {
    Supervisor::with_factory(Box::new(|_ring, _model| {
        Err(HiggsError::WorkerSpawnFailed {
            source: std::io::Error::other("mock: spawn always fails"),
        })
    }))
}

/// Read one request line the supervisor wrote to the transient worker and
/// return its decoded JSON-RPC request id.
async fn read_request_id(obs: &mut tokio::io::DuplexStream) -> u64 {
    use tokio::io::AsyncBufReadExt;
    let mut lines = BufReader::new(obs).lines();
    let line = lines
        .next_line()
        .await
        .expect("read request line")
        .expect("request line present");
    let v: Value = serde_json::from_str(&line).expect("valid json frame");
    v["id"].as_u64().expect("request carries an id")
}

// ─── sysinfo: success → typed GpuDevice list deserialized from worker ─────

#[tokio::test]
async fn sysinfo_returns_devices() {
    let (sup, mut test_write, mut test_read) = transient_supervisor();
    let task = tokio::spawn(async move { sup.sysinfo().await });

    let id = read_request_id(&mut test_read).await;
    write_line(
        &mut test_write,
        &ok_response(
            id,
            json!({"gpus": [{
                "name": "Metal",
                "description": "Apple M3 Max",
                "kind": "Gpu",
                "vram_total_bytes": 42,
                "vram_free_bytes": 21
            }]}),
        ),
    )
    .await;

    let gpus = task.await.expect("sysinfo task");
    assert_eq!(gpus.len(), 1);
    assert_eq!(gpus[0].name, "Metal");
    assert_eq!(gpus[0].vram_total_bytes, 42);
}

// ─── sysinfo_one: worker error reply → empty device list ─────────────────

#[tokio::test]
async fn sysinfo_error_reply_is_empty() {
    let (sup, mut test_write, mut test_read) = transient_supervisor();
    let task = tokio::spawn(async move { sup.sysinfo().await });

    let id = read_request_id(&mut test_read).await;
    write_line(&mut test_write, &err_response(id, -32000, "no devices")).await;

    assert!(task.await.expect("sysinfo task").is_empty());
}

// ─── sysinfo_one: stray frame skipped, then EOF → empty (covers Ok(_) arm) ─

#[tokio::test]
async fn sysinfo_skips_stray_then_eof() {
    let (sup, mut test_write, mut test_read) = transient_supervisor();
    let task = tokio::spawn(async move { sup.sysinfo().await });

    let _ = read_request_id(&mut test_read).await;
    // Blank + stray notification (skipped), then EOF before a real response.
    write_line(&mut test_write, "").await;
    write_line(&mut test_write, &chunk_notif(1, "noise")).await;
    drop(test_write);

    assert!(task.await.expect("sysinfo task").is_empty());
}

// ─── sysinfo: factory spawn failure → empty device list ──────────────────

#[tokio::test]
async fn sysinfo_spawn_failure_is_empty() {
    let sup = failing_supervisor();
    assert!(sup.sysinfo().await.is_empty());
}

// ─── sysinfo_one: pipe broken on write → empty ───────────────────────────

#[tokio::test]
async fn sysinfo_pipe_broken_on_write() {
    let (sup, _test_write, test_read) = transient_supervisor();
    drop(test_read);
    assert!(sup.sysinfo().await.is_empty());
}

// ─── request(): control RPC times out → WorkerDead, pending drained ──────
//
// Paused clock so the 120 s CONTROL_RPC_TIMEOUT elapses instantly. No
// response is ever written; `request` must remove its orphaned pending entry
// and return WorkerDead("… timed out …") rather than hang (lines 359-366).

#[tokio::test(start_paused = true)]
async fn control_rpc_times_out() {
    let (sup, _test_write, _test_read) = make_supervisor();
    let fut = sup.request("higgs/status", json!({}));
    tokio::time::advance(CONTROL_RPC_TIMEOUT + std::time::Duration::from_secs(1)).await;
    let err = fut.await.expect_err("must time out");
    assert!(matches!(err, HiggsError::WorkerDead { .. }), "got {err}");
    assert!(err.to_string().contains("timed out"), "display: {err}");
    // Orphaned pending entry removed — no leak.
    assert_eq!(sup.inner.demux.pending_count(), 0);
}

// ─── send_request(): worker dies before responding → WorkerDead ──────────
//
// Registers a request, then drops the supervisor's pending oneshot sender
// (simulating on_worker_death draining it) so `rx.await` yields Err and the
// "worker died before response" branch (lines 746-748) is taken.

#[tokio::test]
async fn send_request_worker_dies_before_response() {
    let (sup, _test_write, _test_read) = make_supervisor();
    // Spawn the request on its own task so it polls (registers its pending
    // entry + sends its frame) independently of this test's timeline.
    let inner = Arc::clone(&sup.inner);
    let task = tokio::spawn(async move {
        Supervisor { inner }
            .request("higgs/status", json!({}))
            .await
    });
    // Let the request register its pending entry + send its frame.
    tokio::time::sleep(std::time::Duration::from_millis(30)).await;
    // Fail the pending request (what on_worker_death does on death) →
    // dropping its sender closes the oneshot, so rx.await returns Err.
    assert_eq!(
        sup.inner.demux.pending_count(),
        1,
        "exactly one pending request"
    );
    sup.inner.demux.fail_all_pending();
    let err = task
        .await
        .expect("task")
        .expect_err("worker died before response");
    assert!(
        err.to_string().contains("worker died before response"),
        "display: {err}"
    );
}

// ─── send_request(): no worker running → WorkerDead immediately ──────────

#[tokio::test]
async fn send_request_no_worker_is_dead() {
    // No start_for → write_tx is None → send fails before any await.
    let sup = failing_supervisor();
    let err = sup.request("higgs/status", json!({})).await.unwrap_err();
    assert!(err.to_string().contains("[HG007]"), "display: {err}");
    assert_eq!(
        sup.inner.demux.pending_count(),
        0,
        "pending must be cleaned"
    );
}

// ─── do_spawn(): factory failure surfaces the error, running flag released ─

#[tokio::test]
async fn start_for_spawn_failure_releases_running() {
    let sup = failing_supervisor();
    let err = sup.start_for("org/model").expect_err("spawn must fail");
    assert!(
        matches!(err, HiggsError::WorkerSpawnFailed { .. }),
        "got {err}"
    );
    // running was released, so a later start can retry (and also fail).
    assert!(!sup.inner.running.load(Ordering::Relaxed));
    assert!(sup.start_for("org/model").is_err(), "retry also fails");
}

// ─── set_worker_verbose(): writes a frame when a worker is live ──────────
//
// Fire-and-forget M_LOG_LEVEL: no pending entry, no await. With a worker
// live the frame is written to the worker transport; with none it is a
// silent no-op (write_tx is None).

#[tokio::test]
async fn set_worker_verbose_writes_when_live() {
    let (sup, _test_write, mut test_read) = make_supervisor();
    sup.set_worker_verbose(true);

    use tokio::io::AsyncBufReadExt;
    let line = tokio::time::timeout(std::time::Duration::from_millis(500), async {
        let mut lines = BufReader::new(&mut test_read).lines();
        lines.next_line().await.unwrap().expect("frame written")
    })
    .await
    .expect("frame must arrive");
    let v: Value = serde_json::from_str(&line).expect("valid json");
    assert_eq!(v["method"], M_LOG_LEVEL);
    assert_eq!(v["params"]["verbose"], true);
}

#[tokio::test]
async fn set_worker_verbose_noop_without_worker() {
    // No worker live → write_tx None → no-op, no panic.
    let sup = failing_supervisor();
    sup.set_worker_verbose(false); // must not panic
    sup.set_worker_verbose(true);
}

// ─── dispatch(): worker Request and malformed line are no-ops ─────────────
//
// Workers never send requests; a Request frame and an undecodable line both
// hit the silent / warn arms of `dispatch` (lines 1114-1115) without panic.

#[tokio::test]
async fn dispatch_ignores_request_and_malformed() {
    let (sup, _tw, _tr) = make_supervisor();
    // Worker-originated Request frame — silently ignored.
    let req = rpc::encode(&RpcFrame::Request(RpcRequest {
        jsonrpc: "2.0".into(),
        id: 1,
        method: "higgs/whatever".into(),
        params: json!({}),
    }));
    dispatch(&sup.inner, &req);
    // Undecodable line — warn-and-drop.
    dispatch(&sup.inner, "{not valid json");
    dispatch(&sup.inner, "");
    // Pending map untouched, no panic.
    assert_eq!(sup.inner.demux.pending_count(), 0);
}

// ─── on_worker_death(): deliberate stop is silent; drains pending+sinks ───

#[tokio::test]
async fn on_worker_death_deliberate_is_silent() {
    let (sup, _tw, _tr) = make_supervisor();
    // Seed a pending entry and a chat sink so the clear branches execute.
    let rx = sup.inner.demux.register_pending(77);
    let mut sink_rx = sup.register_chat_sink(88);
    assert_eq!(sup.chat_sinks_count(), 1);

    on_worker_death(&sup.inner, None, true);

    // pending failed (oneshot closed), sinks cleared, write_tx None.
    assert_eq!(sup.inner.demux.pending_count(), 0);
    assert_eq!(sup.chat_sinks_count(), 0);
    assert!(sup.inner.write_tx.lock().is_none());
    assert!(rx.await.is_err(), "pending oneshot closed");
    assert!(sink_rx.recv().await.is_none(), "sink closed");
}

// ─── on_worker_death(): unexpected death with no reason → EOF warn arm ────

#[tokio::test]
async fn on_worker_death_unexpected_no_reason_emits() {
    let (sup, _tw, _tr) = make_supervisor();
    let mut events = sup.events();
    // reason = None exercises the "exited unexpectedly (EOF)" warn arm.
    on_worker_death(&sup.inner, None, false);
    let got = tokio::time::timeout(std::time::Duration::from_millis(200), events.recv())
        .await
        .expect("event within timeout")
        .expect("WorkerDied broadcast");
    assert!(matches!(got, HiggsEvent::WorkerDied));
}

// ─── on_worker_death(): unexpected death WITH a reason → warn(detail) arm ─

#[tokio::test]
async fn on_worker_death_unexpected_with_reason_emits() {
    let (sup, _tw, _tr) = make_supervisor();
    let mut events = sup.events();
    // reason = Some(..) exercises the `warn!(detail = %r, ...)` arm.
    on_worker_death(&sup.inner, Some("read error: broken pipe".into()), false);
    let got = tokio::time::timeout(std::time::Duration::from_millis(200), events.recv())
        .await
        .expect("event within timeout")
        .expect("WorkerDied broadcast");
    assert!(matches!(got, HiggsEvent::WorkerDied));
}

// ─── replay_load(): no recorded load → early return, nothing sent ────────

#[tokio::test]
async fn replay_load_empty_is_noop() {
    let (sup, _test_write, test_read) = make_supervisor();
    // No record_last_load → the `let Some(...) else { return }` returns.
    replay_load(&sup.inner);
    tokio::time::sleep(std::time::Duration::from_millis(30)).await;
    use tokio::io::AsyncBufReadExt;
    let mut lines = BufReader::new(test_read).lines();
    let got = tokio::time::timeout(std::time::Duration::from_millis(80), lines.next_line()).await;
    assert!(got.is_err(), "no replay frame when nothing recorded");
}

// ─── replay_load(): recorded load + worker error reply → logged-and-dropped ─
//
// Drives the Err arm of replay_load's spawned task (lines 1219-1220): the
// replayed M_LOAD gets a JSON-RPC error back, so NO ModelLoaded is emitted.

#[tokio::test]
async fn replay_load_error_emits_no_model_loaded() {
    let (sup, mut test_write, mut test_read) = make_supervisor();
    let mut events = sup.events();
    sup.record_last_load(json!({"id": "org/model"}));

    replay_load(&sup.inner);

    // The replayed M_LOAD is written to the (live) worker transport; reply
    // with an error so replay_load_await returns Err.
    let load_id = read_request_id(&mut test_read).await;
    write_line(
        &mut test_write,
        &err_response(load_id, -32000, "load failed"),
    )
    .await;

    // No ModelLoaded must arrive within a short window.
    let got = tokio::time::timeout(std::time::Duration::from_millis(150), async {
        loop {
            match events.recv().await {
                Ok(HiggsEvent::ModelLoaded { .. }) => return true,
                Ok(_) => continue,
                Err(_) => return false,
            }
        }
    })
    .await;
    assert!(got.is_err(), "error reply must NOT emit ModelLoaded");
}

// ─── replay_rpc_await(): no worker running → WorkerDead, pending cleaned ──

#[tokio::test]
async fn replay_rpc_await_no_worker() {
    // Build inner with no live worker (write_tx None).
    let sup = failing_supervisor();
    let err = replay_rpc_await(&sup.inner, M_LOAD, json!({"id": "x"}))
        .await
        .expect_err("no worker → WorkerDead");
    assert!(err.to_string().contains("no worker running"), "got {err}");
    assert_eq!(sup.inner.demux.pending_count(), 0);
}

// ─── replay_rpc_await(): worker dies before response → WorkerDead ─────────

#[tokio::test]
async fn replay_rpc_await_worker_dies() {
    let (sup, _test_write, _test_read) = make_supervisor();
    let inner = Arc::clone(&sup.inner);
    let fut = tokio::spawn(async move { replay_rpc_await(&inner, M_LOAD, json!({})).await });
    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    // Fail the pending request so rx resolves Err (worker death simulation).
    assert_eq!(sup.inner.demux.pending_count(), 1);
    sup.inner.demux.fail_all_pending();
    let err = fut.await.expect("task").expect_err("worker died");
    assert!(
        err.to_string().contains("worker died before response"),
        "got {err}"
    );
}

// ─── replay_rpc_await(): worker accepts stdin but never replies → timeout ─
//
// Paused clock so CONTROL_RPC_TIMEOUT elapses instantly; the orphaned
// pending entry must be removed (lines 1287-1292).

#[tokio::test(start_paused = true)]
async fn replay_rpc_await_times_out() {
    let (sup, _test_write, _test_read) = make_supervisor();
    let inner = Arc::clone(&sup.inner);
    let fut = tokio::spawn(async move { replay_rpc_await(&inner, M_LOAD, json!({})).await });
    // Let the request register + send before advancing the clock.
    tokio::task::yield_now().await;
    tokio::time::sleep(std::time::Duration::from_millis(1)).await;
    tokio::time::advance(CONTROL_RPC_TIMEOUT + std::time::Duration::from_secs(1)).await;
    let err = fut.await.expect("task").expect_err("must time out");
    assert!(err.to_string().contains("timed out"), "got {err}");
    assert_eq!(sup.inner.demux.pending_count(), 0, "orphan removed");
}

// ─── replay_rpc_await(): worker RPC error → WorkerRpc with worker_code ────
//
// Reply carries a `data.code` so the worker's origin diagnostic is recovered
// (lines 1269-1278). Covers the success-path encode/insert too.

#[tokio::test]
async fn replay_rpc_await_worker_rpc_error() {
    let (sup, mut test_write, mut test_read) = make_supervisor();
    let inner = Arc::clone(&sup.inner);
    let fut = tokio::spawn(async move { replay_rpc_await(&inner, M_LOAD, json!({})).await });

    let id = read_request_id(&mut test_read).await;
    // Error reply carrying a worker diagnostic code in `data`.
    let frame = rpc::encode(&RpcFrame::Response(rpc::RpcResponse {
        jsonrpc: "2.0".into(),
        id,
        result: None,
        error: Some(rpc::RpcError {
            code: -32000,
            message: "load bad".into(),
            data: Some(json!({"code": "HG005"})),
        }),
    }));
    write_line(&mut test_write, &frame).await;

    let err = fut.await.expect("task").expect_err("worker rpc error");
    match err {
        HiggsError::WorkerRpc {
            worker_code,
            message,
            ..
        } => {
            assert_eq!(worker_code.as_deref(), Some("HG005"));
            assert_eq!(message, "load bad");
        }
        other => panic!("expected WorkerRpc, got {other}"),
    }
}

// ─── writer_task(): write failure on closed read end ends the task ───────
//
// The writer drains the channel into the worker's stdin. Dropping the read
// end of the duplex makes `write_all` fail, breaking the loop (lines 938-944)
// — silent, the reader drives the death path.

#[tokio::test]
async fn writer_task_exits_on_broken_pipe() {
    let (sup_write, test_read) = tokio::io::duplex(64);
    drop(test_read); // close the peer so writes fail
    let (tx, rx) = mpsc::unbounded_channel::<String>();
    let handle = tokio::spawn(writer_task(Box::new(sup_write), rx));
    // Send a line; the write fails and the task returns.
    tx.send("hello".to_string()).expect("send queued");
    // Task must complete (not hang) once the broken write breaks the loop.
    tokio::time::timeout(std::time::Duration::from_millis(500), handle)
        .await
        .expect("writer task should exit on broken pipe")
        .expect("task join");
}

// ─── reader_task: deliberate stop racing a death → break, no respawn ─────
//
// `stopped` is set before the EOF is observed, so on_worker_death is silent
// and the reader breaks immediately (line 973) without the 1s backoff or a
// respawn — exercising the `if deliberate { break; }` EOF arm.

#[tokio::test]
async fn reader_task_breaks_on_deliberate_stop_eof() {
    let (sup_write, _obs) = tokio::io::duplex(64 * 1024);
    let (test_write, sup_read) = tokio::io::duplex(64 * 1024);
    let cell_w = Arc::new(Mutex::new(Some(sup_write)));
    let cell_r = Arc::new(Mutex::new(Some(sup_read)));
    let sup = Supervisor::with_factory(Box::new(move |_ring, _model| {
        Ok(WorkerHalves {
            write: Box::new(cell_w.lock().take().expect("one spawn")),
            read: Box::new(cell_r.lock().take().expect("one spawn")),
            proc: None,
        })
    }));
    sup.start_for("org/model").expect("start");
    let mut events = sup.events();

    // Mark deliberate stop, THEN trigger EOF. The reader must break without
    // emitting WorkerDied and without attempting a respawn.
    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    sup.inner.stopped.store(true, Ordering::Release);
    drop(test_write);

    // No WorkerDied event must arrive (deliberate stop is silent).
    let got = tokio::time::timeout(std::time::Duration::from_millis(300), events.recv()).await;
    assert!(got.is_err(), "deliberate EOF must not broadcast WorkerDied");
    // running released by the post-loop clear.
    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    assert!(!sup.inner.running.load(Ordering::Relaxed));
}

// ─── attempt_restart(): stopped flips during respawn → reap + abandon ────
//
// The reader's pre-call check passes, but `stopped` flips while the factory
// spawns the replacement. attempt_restart must abandon WITHOUT installing
// the new worker or replaying (lines 1033-1036). Driven directly so the race
// window is deterministic: stopped is already set, but last_load is present
// and the factory succeeds, so the only reason to abandon is the guard.

#[tokio::test]
async fn attempt_restart_abandons_when_stopped() {
    let (sup_write, _obs) = tokio::io::duplex(64 * 1024);
    let (_test_write, sup_read) = tokio::io::duplex(64 * 1024);
    let cell_w = Arc::new(Mutex::new(Some(sup_write)));
    let cell_r = Arc::new(Mutex::new(Some(sup_read)));
    let sup =
        Supervisor::with_factory(Box::new(move |_ring, _model| {
            Ok(WorkerHalves {
                write: Box::new(cell_w.lock().take().ok_or_else(|| {
                    HiggsError::WorkerSpawnFailed {
                        source: std::io::Error::other("one spawn"),
                    }
                })?),
                read: Box::new(cell_r.lock().take().ok_or_else(|| {
                    HiggsError::WorkerSpawnFailed {
                        source: std::io::Error::other("one spawn"),
                    }
                })?),
                proc: None,
            })
        }));
    // Stopped already set → the post-factory guard fires.
    sup.inner.stopped.store(true, Ordering::Release);
    let got = attempt_restart(&sup.inner).await;
    assert!(got.is_none(), "must abandon restart when stopped flipped");
    // No write_tx installed (the new worker was abandoned, not wired in).
    assert!(sup.inner.write_tx.lock().is_none());
}

// ─── attempt_restart(): factory failure → reap old + terminal WorkerDied ─
//
// spawn_replacement fails, so attempt_restart logs, reaps the (None) old
// child, broadcasts a terminal WorkerDied, and returns None (lines 1020-1027).

#[tokio::test]
async fn attempt_restart_factory_failure_is_terminal() {
    let sup = failing_supervisor();
    sup.record_last_load(json!({"id": "org/model"}));
    let mut events = sup.events();
    let got = attempt_restart(&sup.inner).await;
    assert!(got.is_none(), "factory failure → give up");
    let died = tokio::time::timeout(std::time::Duration::from_millis(200), events.recv())
        .await
        .expect("event")
        .expect("WorkerDied");
    assert!(matches!(died, HiggsEvent::WorkerDied));
}

// ─── spawn_replacement(): stamps the recorded model id into argv0 ────────
//
// The replacement factory call carries the last-load `id`. Drive it through
// a factory that records the model arg it was handed.

#[tokio::test]
async fn spawn_replacement_passes_recorded_model() {
    let seen = Arc::new(Mutex::new(Vec::<String>::new()));
    let seen2 = Arc::clone(&seen);
    let sup = Supervisor::with_factory(Box::new(move |_ring, model| {
        seen2.lock().push(model.to_string());
        Err(HiggsError::WorkerSpawnFailed {
            source: std::io::Error::other("capture only"),
        })
    }));
    sup.record_last_load(json!({"id": "org/cool-model"}));
    let _ = spawn_replacement(&sup.inner);
    assert_eq!(seen.lock().as_slice(), &["org/cool-model".to_string()]);
}

// ─── reap_child(None) / install_writer: test-factory no-op reap paths ─────

#[tokio::test]
async fn reap_child_none_is_noop() {
    // None proc (test factory) → reap_child is a no-op, no panic.
    reap_child(None).await;
}

// ─── proc-reap helpers with a real (trivial) Child ───────────────────────
//
// The proc-reap branches in `stop()` / `sysinfo` and the
// `reap_child` / `reap_old_child` / `install_child` helpers all need an
// actual `tokio::process::Child` to `wait()`/`start_kill()`. We use a
// SHORT-LIVED system command (`true` — exits immediately) purely as a Child
// handle. This is NOT the higgs worker (no `--higgs-worker`, no FFI, no
// model) — it is a deterministic, instantly-exiting dummy that exercises the
// OS-process reap path without any real-worker plumbing.

/// Spawn a trivial, instantly-exiting child purely to drive reap logic.
fn dummy_child() -> tokio::process::Child {
    Command::new("true")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn `true`")
}

/// `reap_child(Some(child))` force-kills and reaps a real child without hang.
#[tokio::test]
async fn reap_child_some_kills_and_reaps() {
    reap_child(Some(dummy_child())).await;
}

/// `reap_old_child` + `install_child` reap the OLD child then stash the NEW.
#[tokio::test]
async fn install_and_reap_old_child() {
    let (sup, _tw, _tr) = make_supervisor();
    // Stash an old child, then reap it.
    *sup.inner.proc.lock().await = Some(dummy_child());
    reap_old_child(&sup.inner).await;
    assert!(sup.inner.proc.lock().await.is_none(), "old child reaped");

    // install_child reaps any old (None here) then stores the new one.
    install_child(&sup.inner, Some(dummy_child())).await;
    assert!(sup.inner.proc.lock().await.is_some(), "new child stashed");
    // Reap the one we just installed so the test leaves no live child.
    reap_old_child(&sup.inner).await;
}

/// `stop()` reaps the live child via the proc branch (waits for self-exit).
#[tokio::test]
async fn stop_reaps_live_child() {
    let (sup, _tw, _tr) = make_supervisor();
    // Populate the live-worker proc handle with a trivial child so stop()'s
    // proc-reap branch (child.wait()) executes against a real process.
    *sup.inner.proc.lock().await = Some(dummy_child());
    // Give `true` a moment to exit so wait() returns via the clean arm.
    tokio::time::sleep(std::time::Duration::from_millis(30)).await;
    sup.stop().await;
    assert!(
        sup.inner.proc.lock().await.is_none(),
        "child reaped by stop"
    );
    assert!(!sup.inner.running.load(Ordering::Relaxed));
}

// install_child's stop()-race close: when `stopped` is set (a stop() won the proc lock
// first), install_child must NOT install the new worker — it reaps it and returns false
// so attempt_restart gives up instead of resurrecting a worker after stop. When not
// stopped, it installs and returns true.
#[tokio::test]
async fn install_child_aborts_and_reaps_when_stopped() {
    let (sup, _tw, _tr) = make_supervisor();

    // Not stopped → installs the new child and returns true.
    let installed = install_child(&sup.inner, Some(dummy_child())).await;
    assert!(installed, "installs when not stopped");
    assert!(
        sup.inner.proc.lock().await.is_some(),
        "new child stored on install"
    );

    // Now simulate a concurrent stop() having flipped `stopped` and reaped: clear proc,
    // set stopped. A subsequent restart's install_child must abandon + reap the child it
    // just spawned (return false), leaving proc empty — no resurrection.
    *sup.inner.proc.lock().await = None;
    sup.inner.stopped.store(true, Ordering::Release);
    let installed = install_child(&sup.inner, Some(dummy_child())).await;
    assert!(!installed, "aborts when stopped");
    assert!(
        sup.inner.proc.lock().await.is_none(),
        "aborted restart installs no worker after stop"
    );
}
