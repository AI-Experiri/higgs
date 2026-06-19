//! The remote wire vocabulary (ALPN, HELLO method + payloads, version negotiation).
//! Additive over the existing `rpc.rs` wire. Filled out in P1 Task 4; Task 1 needs
//! only the ALPN so the Endpoint can bind.

/// QUIC ALPN for the higgs remote protocol.
pub const ALPN: &[u8] = b"higgs/remote/1";
