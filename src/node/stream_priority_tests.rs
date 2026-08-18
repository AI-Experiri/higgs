use super::*;
use crate::remote::{
    M_HELLO, M_NODE_INVENTORY, M_NODE_KILL, M_NODE_LEAVE, M_NODE_LOAD, M_NODE_LOGS,
    M_NODE_LOG_LEVEL, M_NODE_PULL, M_NODE_PULL_STATUS, M_NODE_SCAN, M_NODE_STATUS, M_NODE_SYSINFO,
    M_NODE_UNLOAD, M_NODE_UPDATE, M_NODE_UPDATE_VERSION, N_FLEET_EVENT, N_LOG_LINE, N_NODE_LOG,
    N_PROGRESS,
};
use crate::worker::M_CHAT;

#[test]
fn tiers_are_totally_ordered_high_to_low() {
    // The whole point of the slice: control outranks interactive outranks
    // diagnostic. A future edit that flips two consts would silently invert
    // the wire policy — this pins the ordering itself.
    const _: () = assert!(CONTROL_STREAM_PRIORITY > INTERACTIVE_STREAM_PRIORITY);
    const _: () = assert!(INTERACTIVE_STREAM_PRIORITY > DIAGNOSTIC_STREAM_PRIORITY);
}

#[test]
fn control_ops_get_control_priority() {
    for op in [
        M_HELLO,
        M_NODE_LEAVE,
        M_NODE_LOAD,
        M_NODE_UNLOAD,
        M_NODE_KILL,
        M_NODE_SCAN,
        M_NODE_SYSINFO,
        M_NODE_STATUS,
        M_NODE_INVENTORY,
        M_NODE_LOG_LEVEL,
        M_NODE_UPDATE,
        M_NODE_UPDATE_VERSION,
        M_NODE_PULL,
        M_NODE_PULL_STATUS,
        N_FLEET_EVENT,
        N_PROGRESS,
    ] {
        assert_eq!(
            priority_for(op),
            CONTROL_STREAM_PRIORITY,
            "opcode {op} should be CONTROL",
        );
    }
}

#[test]
fn chat_gets_interactive_priority() {
    assert_eq!(priority_for(M_CHAT), INTERACTIVE_STREAM_PRIORITY);
}

#[test]
fn log_ops_get_diagnostic_priority() {
    for op in [M_NODE_LOGS, N_NODE_LOG, N_LOG_LINE] {
        assert_eq!(
            priority_for(op),
            DIAGNOSTIC_STREAM_PRIORITY,
            "opcode {op} should be DIAGNOSTIC",
        );
    }
}

#[test]
fn unknown_opcode_falls_back_to_control() {
    // A future `higgs/node/<future_op>` that lands without a stream_priority
    // update must NOT accidentally be scheduled below chat. Catch-all is
    // CONTROL by design; if a new op is a firehose it needs an explicit arm.
    assert_eq!(
        priority_for("higgs/node/future_op"),
        CONTROL_STREAM_PRIORITY,
    );
    assert_eq!(priority_for(""), CONTROL_STREAM_PRIORITY);
}
