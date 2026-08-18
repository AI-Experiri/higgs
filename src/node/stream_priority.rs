//! Per-QUIC-stream scheduler priority (SP1). Higher wins airtime under wire
//! contention; iroh/noq schedules locally-buffered bytes across a connection's
//! send streams by this value (default 0). Applied through
//! `iroh::endpoint::SendStream::set_priority` — no bespoke shaper, no wire
//! change, no public API surface change. Peers see the same protocol; only the
//! local scheduler behavior shifts.
//!
//! Set once at open time (or right after `accept_bi` reads the request method
//! for the reply side), so both ends of the same stream apply the same tier —
//! `set_priority` is a per-side hint, so the CHAT priority set by the hub on
//! its `open_bi` send only affects the hub's outbound; the node's outbound on
//! the same stream (where the chat tokens actually flow) must be re-tagged on
//! the accept side.

use iroh::endpoint::SendStream;

/// Control-plane RPCs and small state deltas (hello, leave, load/unload/kill,
/// scan, sysinfo, status, inventory, log-level, update, pull, pull_status,
/// fleet_event, progress). Must land first — commands and state pushes drive
/// the fleet UI, and starving them behind a log firehose would freeze user
/// perception of the system.
pub(crate) const CONTROL_STREAM_PRIORITY: i32 = 100;

/// User-visible payload: chat tokens (`M_CHAT` reply chunks + final). Neutral
/// baseline (QUIC's default) — kept ABOVE the diagnostic tier so verbose logs
/// cannot steal wire time from tokens the user is watching arrive.
pub(crate) const INTERACTIVE_STREAM_PRIORITY: i32 = 0;

/// Diagnostic firehose: node daemon log streams (`M_NODE_LOGS`, `N_NODE_LOG`)
/// and worker-log push (`N_LOG_LINE`). Best-effort — never steals from chat or
/// control. The LogBus broadcast side is already lossy on the subscriber; this
/// closes the loop on the wire so a debug flood on one connection cannot
/// backpressure the chat stream sharing it.
pub(crate) const DIAGNOSTIC_STREAM_PRIORITY: i32 = -100;

/// Map a method/notification opcode to its scheduler tier. Unknown opcodes
/// (future control RPCs added to `remote.rs` without touching this file)
/// safely default to CONTROL — a new state RPC is more likely control-like
/// than a firehose, and an accidental over-classification only means it
/// competes fairly with existing control traffic.
pub(crate) fn priority_for(method: &str) -> i32 {
    use crate::remote::{M_NODE_LOGS, N_LOG_LINE, N_NODE_LOG};
    use crate::worker::M_CHAT;
    match method {
        M_CHAT => INTERACTIVE_STREAM_PRIORITY,
        M_NODE_LOGS | N_NODE_LOG | N_LOG_LINE => DIAGNOSTIC_STREAM_PRIORITY,
        _ => CONTROL_STREAM_PRIORITY,
    }
}

/// Apply [`priority_for`] to `send`. The only failure mode Quinn/noq surfaces
/// is `ClosedStream` — a freshly opened or freshly accepted stream cannot yet
/// be closed by the peer, so any error here means the connection is already
/// torn down and the first write is about to surface the real error. Nothing
/// better we can do at the priority-set site; drop the result.
pub(crate) fn apply_for(send: &SendStream, method: &str) {
    let _ = send.set_priority(priority_for(method));
}

#[cfg(test)]
#[path = "stream_priority_tests.rs"]
mod tests;
