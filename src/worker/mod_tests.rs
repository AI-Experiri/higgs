//! Sibling unit tests for the worker serve-loop edges the inline `mod tests`
//! does not reach: skipped input frames, a broken response pipe, the runtime
//! log-level flip, the ctx_len=0 coercion, and malformed load-override
//! degradation. Kept in a sibling file per the repo test-layout rule; the
//! inline module predates it and stays frozen.

use std::io::Cursor;

use serde_json::{json, Value};

use super::{
    engine, serve, serve_state, WorkerState, DEFAULT_WORKER_CTX, M_CHAT, M_LOAD, M_LOG_LEVEL,
    M_STATUS,
};
use crate::diagnostic::HiggsError;
use crate::rpc::{decode, encode, RpcFrame, RpcRequest, RpcResponse};

// ---------------------------------------------------------------------------
// Test helpers (private clones of the inline module's helpers — that module is
// private, so its FakeEngine/req_line cannot be reused from here)
// ---------------------------------------------------------------------------

/// Shared record of engine calls, inspectable after `serve_state` consumes
/// the [`FakeEngine`].
type CallLog = std::sync::Arc<parking_lot::Mutex<Vec<String>>>;

/// Scripted [`engine::HiggsEngine`]: records load/unload/chat calls; chat
/// streams one "hi" delta and returns ("hi", "stop").
struct FakeEngine {
    calls: CallLog,
    loaded: bool,
}

impl FakeEngine {
    fn new(calls: CallLog) -> Self {
        Self {
            calls,
            loaded: false,
        }
    }
}

impl engine::HiggsEngine for FakeEngine {
    fn load(&mut self, path: &str, _params: &engine::LoadParams) -> Result<(), HiggsError> {
        self.calls.lock().push(format!("load {path}"));
        self.loaded = true;
        Ok(())
    }

    fn unload(&mut self) {
        self.calls.lock().push("unload".into());
        self.loaded = false;
    }

    fn is_loaded(&self) -> bool {
        self.loaded
    }

    fn devices(&self) -> Vec<crate::system::GpuDevice> {
        Vec::new()
    }

    fn chat(
        &mut self,
        _messages_json: &str,
        _params: &engine::GenParams,
        sink: &mut dyn FnMut(engine::EngineDelta<'_>),
    ) -> Result<engine::ChatResult, HiggsError> {
        self.calls.lock().push("chat".into());
        sink(engine::EngineDelta::Content("hi"));
        Ok(engine::ChatResult {
            content: "hi".into(),
            finish_reason: "stop",
            tool_calls: None,
            prompt_tokens: 1,
            completion_tokens: 1,
            reasoning_content: None,
        })
    }
}

/// Build a single NDJSON request line.
fn req_line(id: u64, method: &str, params: serde_json::Value) -> String {
    let r = RpcRequest {
        jsonrpc: "2.0".into(),
        id,
        method: method.into(),
        params,
    };
    format!("{}\n", encode(&RpcFrame::Request(r)))
}

/// An M_LOAD request line for `id` carrying the host-resolved GGUF `path`,
/// plus any extra params merged in.
fn load_line(req_id: u64, id: &str, path: &str, extra: serde_json::Value) -> String {
    let mut params = json!({ "id": id, "path": path });
    if let Value::Object(map) = extra {
        for (k, v) in map {
            params[k] = v;
        }
    }
    req_line(req_id, M_LOAD, params)
}

/// Parse all non-empty output lines as RpcFrames.
fn parse_responses(buf: &[u8]) -> Vec<RpcFrame> {
    std::str::from_utf8(buf)
        .unwrap()
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| decode(l).expect("valid frame"))
        .collect()
}

/// Run `input` through a worker whose engine is a [`FakeEngine`]; returns
/// (output frames, engine call log).
fn serve_with_fake(input: &str) -> (Vec<RpcFrame>, CallLog) {
    let calls = CallLog::default();
    let state = WorkerState::with_engine(Box::new(FakeEngine::new(calls.clone())));
    let mut out: Vec<u8> = Vec::new();
    serve_state(state, Cursor::new(input.as_bytes()), &mut out);
    (parse_responses(&out), calls)
}

/// A writer whose pipe is gone: every write fails with `BrokenPipe`, the way
/// stdout behaves once the supervisor process has died.
struct BrokenPipe;

impl std::io::Write for BrokenPipe {
    fn write(&mut self, _buf: &[u8]) -> std::io::Result<usize> {
        Err(std::io::Error::new(
            std::io::ErrorKind::BrokenPipe,
            "supervisor pipe gone",
        ))
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn blank_lines_and_incoming_response_frames_are_skipped() {
    // A blank stdin line is not a frame, and a Response frame is something the
    // worker (a pure server) never answers — both must be skipped WITHOUT
    // emitting anything, and the next real request must still be served. A
    // worker that replied to stray responses would corrupt the NDJSON protocol
    // (the supervisor matches replies by id).
    let stray = RpcResponse {
        jsonrpc: "2.0".into(),
        id: 9,
        result: Some(json!({})),
        error: None,
    };
    let mut input = String::from("\n   \n");
    input.push_str(&format!("{}\n", encode(&RpcFrame::Response(stray))));
    input.push_str(&req_line(4, M_STATUS, json!(null)));

    let mut out: Vec<u8> = Vec::new();
    serve(Cursor::new(input.as_bytes()), &mut out);

    let frames = parse_responses(&out);
    assert_eq!(
        frames.len(),
        1,
        "only the real request is answered: {frames:?}"
    );
    let RpcFrame::Response(resp) = &frames[0] else {
        panic!("expected response")
    };
    assert_eq!(resp.id, 4);
    assert!(resp.error.is_none(), "status succeeds: {:?}", resp.error);
}

#[test]
fn log_level_flip_is_acknowledged_with_an_empty_reply() {
    // Both directions of the "Verbose Logging" flip reply `{}` (an ack, no
    // payload) and never touch the engine's model state — the toggle must be
    // safe on a worker that has nothing loaded. (In-process the engine-log
    // hook was never installed, so the flip is the documented no-op.)
    let mut input = req_line(2, M_LOG_LEVEL, json!({"verbose": true}));
    input.push_str(&req_line(3, M_LOG_LEVEL, json!({"verbose": false})));

    let (frames, calls) = serve_with_fake(&input);
    assert_eq!(frames.len(), 2, "one ack per flip: {frames:?}");
    for f in &frames {
        let RpcFrame::Response(resp) = f else {
            panic!("expected response")
        };
        assert!(resp.error.is_none(), "flip succeeds: {:?}", resp.error);
        assert_eq!(
            resp.result.as_ref().unwrap(),
            &json!({}),
            "the ack carries no payload"
        );
    }
    assert!(calls.lock().is_empty(), "the engine is never touched");
}

#[test]
fn load_with_ctx_len_zero_is_coerced_to_the_default_window() {
    // ctx_len=0 would load (llama would use the model default) but the STORED
    // 0 would fail the chat fit-check for every request — a loaded-yet-unusable
    // model. The worker must coerce 0 to its default so the stored window
    // matches the real one, and status must report the coerced value.
    let path = "/models/google/gemma-4-12b/gemma-4-12b-Q4_K_M.gguf";
    let mut input = load_line(2, "google/gemma-4-12b", path, json!({"ctx_len": 0}));
    input.push_str(&req_line(3, M_STATUS, json!(null)));

    let (frames, _calls) = serve_with_fake(&input);
    let RpcFrame::Response(status) = &frames[1] else {
        panic!("expected status response")
    };
    let loaded = &status.result.as_ref().unwrap()["loaded"];
    assert_eq!(
        loaded["ctx_len"],
        json!({"kind": "fixed", "n": DEFAULT_WORKER_CTX}),
        "the stored window is the coerced default, never 0"
    );
}

#[test]
fn malformed_load_overrides_degrade_to_defaults_not_failure() {
    // Engine-specific overrides that fail to deserialize (here a string where
    // `use_mmap` wants a bool) must NOT fail the load — an older/garbled host
    // still gets its model, with default overrides. The base fields keep their
    // own defaults (threads=4) proving the whole param set fell back cleanly.
    let path = "/models/google/gemma-4-12b/gemma-4-12b-Q4_K_M.gguf";
    let mut input = load_line(
        2,
        "google/gemma-4-12b",
        path,
        json!({"use_mmap": "definitely-not-a-bool"}),
    );
    input.push_str(&req_line(3, M_STATUS, json!(null)));

    let (frames, calls) = serve_with_fake(&input);
    let RpcFrame::Response(load) = &frames[0] else {
        panic!("expected load response")
    };
    assert!(
        load.error.is_none(),
        "malformed overrides never fail the load: {:?}",
        load.error
    );
    assert_eq!(load.result.as_ref().unwrap()["id"], "google/gemma-4-12b");
    let RpcFrame::Response(status) = &frames[1] else {
        panic!("expected status response")
    };
    let loaded = &status.result.as_ref().unwrap()["loaded"];
    assert_eq!(loaded["threads"], 4, "base defaults survive the fallback");
    assert_eq!(
        calls.lock().first().map(String::as_str),
        Some(format!("load {path}").as_str()),
        "the engine load ran with the degraded params"
    );
}

#[test]
fn broken_response_pipe_never_crashes_the_serve_loop() {
    // Once the supervisor dies, every response/chunk write hits a broken pipe.
    // The loop must warn and keep dispatching to stdin EOF (stdin EOF is what
    // ends it) instead of panicking — the engine work still runs. This drives
    // all three broken-pipe sites: the response writer, the mid-chat chunk
    // sink, and the post-chat warn.
    let calls = CallLog::default();
    let state = WorkerState::with_engine(Box::new(FakeEngine::new(calls.clone())));
    let path = "/models/google/gemma-4-12b/gemma-4-12b-Q4_K_M.gguf";
    let mut input = load_line(2, "google/gemma-4-12b", path, json!({}));
    input.push_str(&req_line(
        3,
        M_CHAT,
        json!({
            "request_id": 7,
            "model": "google/gemma-4-12b",
            "messages_json": "[{\"role\":\"user\",\"content\":\"hi\"}]",
        }),
    ));
    input.push_str(&req_line(4, M_STATUS, json!(null)));

    // Must return (loop survives every failed write to EOF), not panic.
    serve_state(state, Cursor::new(input.as_bytes()), &mut BrokenPipe);

    let calls = calls.lock();
    assert_eq!(
        *calls,
        vec![format!("load {path}"), "chat".to_string()],
        "the engine ran the full load+chat despite the dead pipe"
    );
}
