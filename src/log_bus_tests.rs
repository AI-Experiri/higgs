use super::*;
use std::sync::Arc;

#[test]
fn push_writes_ring_and_broadcast() {
    let bus = Arc::new(LogBus::new());
    let mut rx = bus.subscribe();
    bus.push(LogSource::Serve, "line-1".to_owned());
    // Ring captured it (snapshot history).
    assert_eq!(bus.snapshot(10, None), vec!["line-1".to_owned()]);
    // Broadcast delivered the same line to the live subscriber.
    assert_eq!(rx.try_recv().unwrap().text, "line-1");
}

#[test]
fn snapshot_returns_last_n_oldest_first() {
    let bus = LogBus::new();
    for i in 0..5 {
        bus.push(LogSource::Serve, format!("l{i}"));
    }
    assert_eq!(
        bus.snapshot(2, None),
        vec!["l3".to_owned(), "l4".to_owned()]
    );
    assert_eq!(bus.snapshot(0, None), Vec::<String>::new());
}

#[test]
fn snapshot_filters_by_source() {
    let bus = LogBus::new();
    bus.push(LogSource::Serve, "serve-1".to_owned());
    bus.push(LogSource::Worker, "worker-1".to_owned());
    bus.push(LogSource::Serve, "serve-2".to_owned());
    assert_eq!(bus.snapshot(10, None).len(), 3, "no filter = all sources");
    assert_eq!(
        bus.snapshot(10, Some(LogSource::Worker)),
        vec!["worker-1".to_owned()]
    );
    assert_eq!(
        bus.snapshot(10, Some(LogSource::Serve)),
        vec!["serve-1".to_owned(), "serve-2".to_owned()]
    );
}

#[test]
fn worker_flood_does_not_evict_serve_history() {
    // The bug: a single shared ring let worker model-load spam evict every
    // serve line, leaving the Developer Logs console empty. Per-source rings
    // keep each console's history independent.
    let bus = LogBus::new();
    bus.push(LogSource::Serve, "higgs: GET /v1/models".to_owned());
    bus.push(LogSource::Serve, "higgs: loading model".to_owned());
    for i in 0..(RING_CAP + 500) {
        bus.push(LogSource::Worker, format!("ggml line {i}"));
    }
    // Serve history survives the worker flood intact.
    assert_eq!(
        bus.snapshot(10, Some(LogSource::Serve)),
        vec![
            "higgs: GET /v1/models".to_owned(),
            "higgs: loading model".to_owned()
        ],
        "serve lines must not be evicted by worker output"
    );
    // The worker ring is still bounded (capped, not unbounded).
    assert_eq!(
        bus.snapshot(usize::MAX, Some(LogSource::Worker)).len(),
        RING_CAP
    );
}

#[test]
fn concurrent_pushes_keep_ring_seq_monotonic() {
    // Per-ring invariant: a ring's seq stamps must increase with insertion order, even
    // under heavy concurrent pushes to ONE source. `snapshot(None)` sorts by seq while
    // `snapshot(source)` returns insertion order, so an out-of-order seq makes the two
    // views disagree on near-simultaneous lines. Guards the "stamp seq UNDER the ring
    // lock" fix; the prior fetch_add-before-lock could stamp seq=N then insert it after
    // seq=N+1 when the N thread stalled between the fetch_add and the lock.
    const THREADS: u64 = 8;
    // THREADS * PER == RING_CAP → the whole window is retained (no eviction to muddy it).
    const PER: u64 = (RING_CAP as u64) / THREADS;
    let bus = std::sync::Arc::new(LogBus::new());
    let handles: Vec<_> = (0..THREADS)
        .map(|t| {
            let bus = std::sync::Arc::clone(&bus);
            std::thread::spawn(move || {
                for i in 0..PER {
                    bus.push(LogSource::Serve, format!("t{t}-{i}"));
                }
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
    let seqs: Vec<u64> = bus.serve.lock().iter().map(|(q, _)| *q).collect();
    assert_eq!(seqs.len() as u64, THREADS * PER, "no eviction at RING_CAP");
    assert!(
        seqs.windows(2).all(|w| w[0] < w[1]),
        "ring seqs must strictly increase with insertion order — an inversion makes \
             snapshot(None) reorder concurrently-pushed lines vs the per-source view",
    );
}

#[test]
fn ring_evicts_oldest_at_capacity() {
    let bus = LogBus::new();
    for i in 0..(RING_CAP + 5) {
        bus.push(LogSource::Worker, format!("l{i}"));
    }
    let snap = bus.snapshot(RING_CAP + 100, None);
    assert_eq!(snap.len(), RING_CAP);
    // Oldest 5 were evicted; first surviving line is l5.
    assert_eq!(snap[0], "l5");
}

#[test]
fn subscriber_only_sees_lines_after_subscribe() {
    let bus = LogBus::new();
    bus.push(LogSource::Serve, "before".to_owned());
    let mut rx = bus.subscribe();
    bus.push(LogSource::Serve, "after".to_owned());
    // "before" is NOT delivered live (it predates the subscription); it is
    // only available via the snapshot replay.
    assert_eq!(rx.try_recv().unwrap().text, "after");
    assert!(rx.try_recv().is_err());
    assert_eq!(
        bus.snapshot(10, None),
        vec!["before".to_owned(), "after".to_owned()]
    );
}

#[tokio::test]
async fn lagged_subscriber_reports_lagged_then_recovers() {
    let bus = LogBus::new();
    let mut rx = bus.subscribe();
    // Overflow the broadcast channel without draining — the slow subscriber
    // falls behind and the next recv reports Lagged rather than crashing.
    for i in 0..(BROADCAST_CAP + 10) {
        bus.push(LogSource::Worker, format!("l{i}"));
    }
    match rx.recv().await {
        Err(broadcast::error::RecvError::Lagged(skipped)) => {
            assert!(skipped > 0, "expected a positive lagged count");
        }
        other => panic!("expected Lagged, got {other:?}"),
    }
    // After Lagged the receiver recovers and yields the still-buffered tail.
    let next = rx.recv().await.expect("recovers after lag");
    assert!(next.text.starts_with('l'));
}

#[test]
fn layer_captures_error_field_and_redacts_other_fields() {
    use tracing_subscriber::layer::SubscriberExt;
    let bus = Arc::new(LogBus::new());
    let subscriber = tracing_subscriber::registry().with(HiggsLogLayer::new(bus.clone()));
    tracing::subscriber::with_default(subscriber, || {
        // A higgs-target load failure: the typed reason rides the `error` field.
        tracing::warn!(
            target: "higgs::test",
            error = "[HG004] engine failed to load m.gguf: out of memory",
            prompt = "secret user prompt content",
            "higgs: load failed"
        );
        // Wrong target — must be dropped entirely.
        tracing::info!(target: "not_higgs", "should be ignored");
    });
    let snap = bus.snapshot(10, None);
    assert_eq!(snap.len(), 1, "only the higgs-target event is captured");
    let line = &snap[0];
    assert!(
        line.contains("higgs: load failed"),
        "message present: {line}"
    );
    assert!(
        line.contains("[HG004] engine failed to load m.gguf: out of memory"),
        "error field appended so failures are debuggable: {line}"
    );
    assert!(
        !line.contains("secret user prompt content"),
        "non-error fields stay redacted by default (no prompt content): {line}"
    );
}

#[test]
fn layer_show_fields_unredacts_for_debug() {
    use tracing_subscriber::layer::SubscriberExt;
    let bus = Arc::new(LogBus::new());
    bus.set_show_fields(true); // DEBUG mode on
    let subscriber = tracing_subscriber::registry().with(HiggsLogLayer::new(bus.clone()));
    tracing::subscriber::with_default(subscriber, || {
        tracing::info!(
            target: "higgs::test",
            prompt = "the actual user prompt",
            "higgs: chat"
        );
    });
    let snap = bus.snapshot(10, None);
    assert_eq!(snap.len(), 1);
    assert!(
        snap[0].contains("prompt=the actual user prompt"),
        "show mode appends other fields incl. prompt content: {}",
        snap[0]
    );
}

#[test]
fn parses_remote_node_source_selector() {
    assert_eq!(
        LogSource::parse("node:1:2"),
        Some(LogSource::RemoteWorker {
            node: NodeId(1),
            worker: WorkerId(2)
        })
    );
    // `node:<id>` without a worker part is the node's own DAEMON log (M_NODE_LOGS).
    assert_eq!(
        LogSource::parse("node:1"),
        Some(LogSource::RemoteNode { node: NodeId(1) })
    );
    // Malformed selectors fall back to "all sources" (None).
    assert_eq!(LogSource::parse("node:x:2"), None);
    assert_eq!(LogSource::parse("node:x"), None);
    assert_eq!(LogSource::parse("bogus"), None);
}

#[test]
fn remote_worker_lines_are_keyed_and_separable() {
    let bus = LogBus::new();
    let a = LogSource::RemoteWorker {
        node: NodeId(1),
        worker: WorkerId(1),
    };
    let b = LogSource::RemoteWorker {
        node: NodeId(2),
        worker: WorkerId(1),
    };
    bus.push(a, "a-line".to_owned());
    bus.push(b, "b-line".to_owned());
    bus.push(a, "a-line-2".to_owned());
    assert_eq!(
        bus.snapshot(10, Some(a)),
        vec!["a-line".to_owned(), "a-line-2".to_owned()]
    );
    assert_eq!(bus.snapshot(10, Some(b)), vec!["b-line".to_owned()]);
    // Unfiltered interleaves both remote workers in arrival order.
    assert_eq!(
        bus.snapshot(10, None),
        vec![
            "a-line".to_owned(),
            "b-line".to_owned(),
            "a-line-2".to_owned()
        ]
    );
}

#[test]
fn evict_remote_reclaims_a_dead_workers_ring() {
    let bus = LogBus::new();
    let a = LogSource::RemoteWorker {
        node: NodeId(1),
        worker: WorkerId(7),
    };
    bus.push(a, "x".to_owned());
    assert_eq!(bus.snapshot(10, Some(a)).len(), 1);
    bus.evict_remote(NodeId(1), WorkerId(7));
    assert!(bus.snapshot(10, Some(a)).is_empty(), "ring reclaimed");
    // Evicting an unknown worker is a harmless no-op.
    bus.evict_remote(NodeId(9), WorkerId(9));
}

#[test]
fn evict_node_reclaims_all_of_a_nodes_rings() {
    let bus = LogBus::new();
    let w1 = LogSource::RemoteWorker {
        node: NodeId(1),
        worker: WorkerId(1),
    };
    let w2 = LogSource::RemoteWorker {
        node: NodeId(1),
        worker: WorkerId(2),
    };
    let other = LogSource::RemoteWorker {
        node: NodeId(2),
        worker: WorkerId(1),
    };
    bus.push(w1, "a".to_owned());
    bus.push(w2, "b".to_owned()); // a displaced worker's ring, off any current route
    bus.push(other, "c".to_owned());
    bus.evict_node(NodeId(1));
    assert!(
        bus.snapshot(10, Some(w1)).is_empty(),
        "node 1 worker 1 reclaimed"
    );
    assert!(
        bus.snapshot(10, Some(w2)).is_empty(),
        "node 1 worker 2 reclaimed too"
    );
    assert_eq!(
        bus.snapshot(10, Some(other)),
        vec!["c".to_owned()],
        "other node untouched"
    );
}

#[test]
fn remote_ring_is_capacity_bounded() {
    let bus = LogBus::new();
    let a = LogSource::RemoteWorker {
        node: NodeId(1),
        worker: WorkerId(1),
    };
    for i in 0..(RING_CAP + 10) {
        bus.push(a, format!("l{i}"));
    }
    assert_eq!(bus.snapshot(usize::MAX, Some(a)).len(), RING_CAP);
}

#[test]
fn default_constructs_an_empty_bus() {
    // `Default` defers to `new()` — a fresh bus with empty rings.
    let bus = LogBus::default();
    assert!(bus.snapshot(10, None).is_empty());
    assert!(!bus.show_fields(), "DEBUG mode off by default");
    assert!(bus.verbose(), "verbose ON by default (NL-V design)");
}

#[test]
fn layer_captures_debug_typed_error_and_fields() {
    // Non-string fields (recorded via the `?` Debug sigil) route through the visitor's
    // `record_debug` path, distinct from the string `record_str` path. The error field
    // is always appended; the extra is appended only in DEBUG (show_fields) mode.
    use tracing_subscriber::layer::SubscriberExt;
    let bus = Arc::new(LogBus::new());
    bus.set_show_fields(true);
    let subscriber = tracing_subscriber::registry().with(HiggsLogLayer::new(bus.clone()));
    tracing::subscriber::with_default(subscriber, || {
        // `?value` (the Debug sigil) routes the field through the visitor's
        // `record_debug`, distinct from the `record_str` path the string tests hit.
        let err_val = vec![1u8, 2, 3];
        let extra_val = Some(9u32);
        tracing::warn!(
            target: "higgs::test",
            error = ?err_val,
            extra_dbg = ?extra_val,
            "higgs: debug-typed"
        );
    });
    let snap = bus.snapshot(10, None);
    assert_eq!(snap.len(), 1);
    let line = &snap[0];
    assert!(
        line.contains("higgs: debug-typed"),
        "message present: {line}"
    );
    assert!(
        line.contains("[1, 2, 3]"),
        "debug-typed error field appended: {line}"
    );
    assert!(
        line.contains("extra_dbg=Some(9)"),
        "debug-typed extra field shown in DEBUG mode: {line}"
    );
}

#[test]
fn layer_drops_debug_level_when_not_verbose() {
    // With verbose OFF, a higgs-target DEBUG/TRACE event is dropped at the level gate
    // (`return`) — only INFO+ reaches the Developer-Logs ring. Fresh bus defaults
    // verbose=true per NL-V, so this test explicitly flips it off.
    use tracing_subscriber::layer::SubscriberExt;
    let bus = Arc::new(LogBus::new());
    bus.set_verbose(false);
    assert!(!bus.verbose());
    let subscriber = tracing_subscriber::registry().with(HiggsLogLayer::new(bus.clone()));
    tracing::subscriber::with_default(subscriber, || {
        tracing::debug!(target: "higgs::test", "higgs: a debug line");
        tracing::info!(target: "higgs::test", "higgs: an info line");
    });
    let snap = bus.snapshot(10, None);
    assert_eq!(snap.len(), 1, "only the INFO line survives the level gate");
    assert!(
        snap[0].contains("higgs: an info line"),
        "the DEBUG line was dropped, the INFO line kept: {snap:?}"
    );
    assert!(
        !snap[0].contains("a debug line"),
        "the DEBUG line must not appear: {snap:?}"
    );
}

#[test]
fn layer_drops_event_with_no_message() {
    // An event carrying no `message` (only structured fields) is dropped — the bus
    // only buffers human lines, never a bare field dump.
    use tracing_subscriber::layer::SubscriberExt;
    let bus = Arc::new(LogBus::new());
    let subscriber = tracing_subscriber::registry().with(HiggsLogLayer::new(bus.clone()));
    tracing::subscriber::with_default(subscriber, || {
        tracing::info!(target: "higgs::test", only_field = "x");
    });
    assert!(
        bus.snapshot(10, None).is_empty(),
        "a message-less event is not buffered"
    );
}

#[test]
fn log_filter_admits_higgs_debug() {
    // The per-layer filter scopes DEBUG admission to the `higgs` target only, so a
    // higgs DEBUG event reaches the layer (its own gate then drops it unless verbose),
    // while DEBUG from any other target is never generated for this layer.
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::Layer;
    let bus = Arc::new(LogBus::new());
    bus.set_verbose(true); // so admitted DEBUG survives the layer's own gate
    let subscriber = tracing_subscriber::registry()
        .with(HiggsLogLayer::new(bus.clone()).with_filter(log_filter()));
    tracing::subscriber::with_default(subscriber, || {
        tracing::debug!(target: "higgs::test", "higgs: filtered debug");
        tracing::debug!(target: "other::test", "should be filtered out");
    });
    let snap = bus.snapshot(10, None);
    assert_eq!(snap.len(), 1, "only the higgs-target DEBUG is admitted");
    assert!(snap[0].contains("higgs: filtered debug"), "{:?}", snap);
}

#[test]
fn timestamp_has_expected_shape() {
    let ts = timestamp();
    // "YYYY-MM-DD HH:MM:SS" — 19 chars.
    assert_eq!(ts.len(), 19, "got {ts:?}");
    assert_eq!(&ts[4..5], "-");
    assert_eq!(&ts[10..11], " ");
    assert_eq!(&ts[13..14], ":");
}

#[test]
fn parses_local_worker_source_selector() {
    assert_eq!(
        LogSource::parse("worker:3"),
        Some(LogSource::LocalWorker {
            worker: WorkerId(3)
        })
    );
    assert_eq!(LogSource::parse("worker:"), None);
    assert_eq!(LogSource::parse("worker:x"), None);
    // The bare legacy selector still parses to the union filter.
    assert_eq!(LogSource::parse("worker"), Some(LogSource::Worker));
}

#[test]
fn worker_filter_is_a_union_over_local_workers() {
    let lw = LogSource::LocalWorker {
        worker: WorkerId(1),
    };
    assert!(lw.matches_filter(LogSource::Worker), "union rule");
    assert!(lw.matches_filter(lw), "exact match");
    assert!(!lw.matches_filter(LogSource::LocalWorker {
        worker: WorkerId(2)
    }));
    assert!(!lw.matches_filter(LogSource::Serve));
    // The union rule is one-directional: a legacy unkeyed line does NOT match
    // a per-worker filter.
    assert!(!LogSource::Worker.matches_filter(lw));
}

#[test]
fn local_worker_lines_are_keyed_and_separable() {
    let bus = LogBus::new();
    let w1 = LogSource::LocalWorker {
        worker: WorkerId(1),
    };
    let w2 = LogSource::LocalWorker {
        worker: WorkerId(2),
    };
    bus.push(w1, "one-a".into());
    bus.push(w2, "two-a".into());
    bus.push(w1, "one-b".into());
    assert_eq!(bus.snapshot(10, Some(w1)), vec!["one-a", "one-b"]);
    assert_eq!(bus.snapshot(10, Some(w2)), vec!["two-a"]);
}

#[test]
fn worker_snapshot_unions_legacy_and_local_rings_in_arrival_order() {
    let bus = LogBus::new();
    bus.push(LogSource::Worker, "legacy".into());
    bus.push(
        LogSource::LocalWorker {
            worker: WorkerId(1),
        },
        "w1".into(),
    );
    bus.push(
        LogSource::LocalWorker {
            worker: WorkerId(2),
        },
        "w2".into(),
    );
    bus.push(LogSource::Serve, "serve".into());
    assert_eq!(
        bus.snapshot(10, Some(LogSource::Worker)),
        vec!["legacy", "w1", "w2"],
        "union of the legacy ring and every local worker ring, no serve lines"
    );
    // The unfiltered snapshot interleaves all four ring kinds by arrival.
    assert_eq!(bus.snapshot(10, None), vec!["legacy", "w1", "w2", "serve"]);
}

#[test]
fn evict_local_reclaims_a_dead_workers_ring() {
    let bus = LogBus::new();
    let w = LogSource::LocalWorker {
        worker: WorkerId(7),
    };
    bus.push(w, "line".into());
    assert_eq!(bus.snapshot(10, Some(w)).len(), 1);
    bus.evict_local(WorkerId(7));
    assert!(bus.snapshot(10, Some(w)).is_empty());
    // Idempotent on a never-logged / already-evicted worker.
    bus.evict_local(WorkerId(7));
}

#[test]
fn remote_node_source_parses_snapshots_and_evicts() {
    use crate::node::node_id::NodeId;
    // `node:<id>` (no worker part) selects a node's own DAEMON log.
    assert_eq!(
        LogSource::parse("node:7"),
        Some(LogSource::RemoteNode { node: NodeId(7) })
    );
    // The worker form still parses as before.
    assert!(matches!(
        LogSource::parse("node:7:3"),
        Some(LogSource::RemoteWorker { .. })
    ));
    let bus = LogBus::new();
    let n7 = LogSource::RemoteNode { node: NodeId(7) };
    let n8 = LogSource::RemoteNode { node: NodeId(8) };
    bus.push(n7, "seven says hi".into());
    bus.push(n8, "eight says hi".into());
    assert_eq!(bus.snapshot(10, Some(n7)), vec!["seven says hi"]);
    // Unfiltered snapshot interleaves the daemon rings too.
    assert_eq!(bus.snapshot(10, None).len(), 2);
    // Retire eviction clears the node's daemon ring alongside its worker rings.
    bus.evict_node(NodeId(7));
    assert!(bus.snapshot(10, Some(n7)).is_empty());
    assert_eq!(bus.snapshot(10, Some(n8)), vec!["eight says hi"]);
}

/// A WORKER token flood must never lag (or drop lines from) the SERVE-only
/// subscription — the daemon-log stream's isolation guarantee. (On the shared
/// broadcast this fails: worker pushes advance every subscriber's cursor.)
#[test]
fn worker_floods_do_not_lag_the_serve_subscription() {
    let bus = LogBus::new();
    let mut serve_rx = bus.subscribe_serve();
    // Overrun the shared channel capacity many times over with worker lines.
    for i in 0..2048 {
        bus.push(LogSource::Worker, format!("token {i}"));
    }
    bus.push(LogSource::Serve, "daemon: still crisp".into());
    match serve_rx.try_recv() {
        Ok(line) => assert_eq!(line, "daemon: still crisp"),
        other => panic!("serve subscriber must see its line un-lagged, got {other:?}"),
    }
}

/// The process-global bus registration is FIRST-CALL-WINS: the bus bound at
/// startup stays the one `global()` hands the node daemon for its whole life —
/// a later (buggy/duplicate) install must be a no-op, never a silent swap that
/// would split daemon tracing across two buses mid-run. This test is the ONLY
/// unit test allowed to touch the global slot (all others build private buses),
/// so both installs and the reads are deterministic within it.
#[test]
fn install_global_is_first_call_wins_and_read_back_by_global() {
    let first = Arc::new(LogBus::new());
    LogBus::install_global(first.clone());
    let got = LogBus::global().expect("global() returns the installed bus");
    assert!(
        Arc::ptr_eq(&got, &first),
        "global() hands back the very bus that was installed"
    );

    // A second install is a no-op — the first registration survives.
    let second = Arc::new(LogBus::new());
    LogBus::install_global(second.clone());
    let still = LogBus::global().expect("global() still set");
    assert!(
        Arc::ptr_eq(&still, &first),
        "later installs must not replace the startup bus"
    );
}
