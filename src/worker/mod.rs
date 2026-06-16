//! Worker role: runs inside the re-exec'd process. Owns the engine. Speaks
//! NDJSON JSON-RPC on stdin/stdout; logs to stderr. The supervisor is the ONLY
//! client. Model scanning is host-side — the worker holds no catalog; the host
//! resolves the GGUF path and passes it in the M_LOAD params.

pub mod engine;
pub mod models;
pub mod tool_parser;

use std::io::{BufRead, Write};

use miette::Diagnostic;
use serde_json::{json, Value};

use crate::diagnostic::HiggsError;
use crate::rpc::{decode, encode, RpcError, RpcFrame, RpcNotification, RpcRequest, RpcResponse};

/// Method names — the only vocabulary on the supervisor↔worker wire.
pub const M_LOAD: &str = "higgs/load";
pub const M_UNLOAD: &str = "higgs/unload";
pub const M_STATUS: &str = "higgs/status";
pub const M_CHAT: &str = "higgs/chat";
pub const M_SHUTDOWN: &str = "higgs/shutdown";
/// Streaming notification carrying one content delta for an in-flight chat.
pub const N_CHAT_CHUNK: &str = "higgs/chat/chunk";

/// Entry point for the `--higgs-worker` role: serve JSON-RPC on stdio until
/// `higgs/shutdown` or stdin EOF. Called by the HOST binary's main().
pub fn worker_main() {
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    serve(stdin.lock(), stdout.lock());
}

/// IO-generic server loop (unit-testable with in-memory buffers).
fn serve(reader: impl BufRead, writer: impl Write) {
    serve_state(WorkerState::new(), reader, writer);
}

/// Server loop over caller-supplied state — the test seam for engine injection.
fn serve_state(mut state: WorkerState, reader: impl BufRead, mut writer: impl Write) {
    for line in reader.lines() {
        let Ok(line) = line else { break };
        if line.trim().is_empty() {
            continue;
        }
        match decode(&line) {
            Ok(RpcFrame::Request(req)) => {
                if req.method == M_SHUTDOWN {
                    respond(&mut writer, req.id, Ok(json!({})));
                    break;
                }
                let out = state.dispatch(&req, &mut writer);
                respond(&mut writer, req.id, out);
            }
            Ok(_) => {} // worker never receives responses/notifications
            Err(e) => {
                // Decode failure has no id: JSON-RPC null-id convention, we use 0.
                respond(
                    &mut writer,
                    0,
                    Err(RpcError {
                        code: -32700,
                        message: e.to_string(),
                        data: None,
                    }),
                );
            }
        }
    }
}

fn respond(writer: &mut impl Write, id: u64, out: Result<Value, RpcError>) {
    let resp = match out {
        Ok(result) => RpcResponse {
            jsonrpc: "2.0".into(),
            id,
            result: Some(result),
            error: None,
        },
        Err(error) => RpcResponse {
            jsonrpc: "2.0".into(),
            id,
            result: None,
            error: Some(error),
        },
    };
    // The supervisor is the sole reader; a broken response pipe means it is
    // gone. Warn rather than crash the loop — stdin EOF will end it shortly.
    if writeln!(writer, "{}", encode(&RpcFrame::Response(resp))).is_err() {
        tracing::warn!("response write failed; supervisor pipe broken");
    }
}

/// Worker-held state: engine and load bookkeeping. No model catalog — scanning
/// is host-side and the GGUF path arrives in the M_LOAD params.
struct WorkerState {
    engine: Box<dyn engine::HiggsEngine>,
    /// (model id, load params) of the resident model — reported by M_STATUS.
    loaded: Option<(String, engine::LoadParams)>,
}

impl WorkerState {
    /// Production state: llama.cpp engine, nothing loaded.
    fn new() -> Self {
        Self {
            engine: Box::new(engine::llamacpp::LlamaCppEngine::default()),
            loaded: None,
        }
    }

    /// Test seam: same state shape with an injected engine.
    #[cfg(test)]
    fn with_engine(engine: Box<dyn engine::HiggsEngine>) -> Self {
        Self {
            engine,
            loaded: None,
        }
    }

    fn dispatch(
        &mut self,
        req: &RpcRequest,
        // M_CHAT streams N_CHAT_CHUNK notifications through this writer mid-request
        writer: &mut impl Write,
    ) -> Result<Value, RpcError> {
        match req.method.as_str() {
            M_STATUS => Ok(self.handle_status()),
            M_LOAD => self.handle_load(req),
            M_UNLOAD => {
                self.engine.unload();
                self.loaded = None;
                Ok(json!({}))
            }
            M_CHAT => self.handle_chat(req, writer),
            other => Err(RpcError {
                code: -32601,
                message: format!("unknown method {other}"),
                data: None,
            }),
        }
    }

    /// Report the resident model's id and load params (or `null` when nothing is
    /// loaded). Model metadata (arch/quant/size/ctx) is enriched host-side from
    /// the scan; the worker holds no catalog, so it reports only what it knows.
    fn handle_status(&self) -> Value {
        let loaded = self.loaded.as_ref().map(|(id, p)| {
            json!({
                "id": id,
                "ctx_len": p.ctx_len,
                "gpu_layers": p.gpu_layers,
                "threads": p.threads,
            })
        });
        json!({ "loaded": loaded })
    }

    /// Load the GGUF at the host-resolved `path` (the host scans and carries the
    /// path in M_LOAD params). Coerces `ctx_len == 0` to a usable default.
    fn handle_load(&mut self, req: &RpcRequest) -> Result<Value, RpcError> {
        let id = req
            .params
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or_default();
        // ctx_len 0 would load (NonZeroU32::new(0)=None → llama uses the
        // model default) but the stored 0 makes the chat fit-check
        // (tokens + max_tokens > n_ctx) fail for every request, so the
        // model loads yet is unusable. Coerce 0 → default so the stored
        // window matches the real one.
        let mut ctx_len = u32_param(&req.params, "ctx_len", 4096);
        if ctx_len == 0 {
            tracing::warn!("higgs: ctx_len=0 requested; using default 4096");
            ctx_len = 4096;
        }
        let params = engine::LoadParams {
            ctx_len,
            gpu_layers: u32_param(&req.params, "gpu_layers", u32::MAX),
            threads: u32_param(&req.params, "threads", 4),
        };
        // Scan moved host-side: the host resolves the GGUF path and passes it in
        // `params.path`. The worker holds no catalog of its own.
        let path = req
            .params
            .get("path")
            .and_then(Value::as_str)
            .ok_or_else(|| RpcError {
                code: -32602,
                data: None,
                message: "missing path".to_string(),
            })?
            .to_string();
        // The engine drops any resident model before loading the new one,
        // so the previous model is already gone the moment we attempt this
        // load. Clear our tracked id up front: if the load then fails, the
        // `?` returns before the `Some(...)` below runs, leaving `loaded`
        // as `None` — so status reports nothing loaded, matching the empty
        // engine, instead of lying about the old id.
        self.loaded = None;
        self.engine
            .load(&path, &params)
            .map_err(|e| to_rpc_error(&e))?;
        self.loaded = Some((id.to_string(), params));
        Ok(json!({ "id": id }))
    }

    /// Run one chat completion, streaming each content delta as an
    /// N_CHAT_CHUNK notification through `writer`, then return the final result.
    fn handle_chat(
        &mut self,
        req: &RpcRequest,
        writer: &mut impl Write,
    ) -> Result<Value, RpcError> {
        if self.loaded.is_none() {
            return Err(to_rpc_error(&HiggsError::ModelNotLoaded {
                id: "unloaded".into(),
            }));
        }
        let request_id = req.params.get("request_id").cloned().unwrap_or(Value::Null);
        // The serve layer serialized the request's OpenAI `messages`
        // array verbatim (preserving tool_calls / tool_call_id); the
        // engine feeds it straight to the chat template. Carried as a
        // JSON string so it round-trips unparsed.
        let messages_json = req
            .params
            .get("messages_json")
            .and_then(Value::as_str)
            .ok_or_else(|| RpcError {
                code: -32602,
                data: None,
                message: "missing messages_json".to_string(),
            })?;
        let gen = engine::GenParams {
            max_tokens: req
                .params
                .get("max_tokens")
                .and_then(Value::as_u64)
                .map_or(1024, |v| usize::try_from(v).unwrap_or(usize::MAX)),
            temperature: req
                .params
                .get("temperature")
                .and_then(Value::as_f64)
                .map_or(0.7, |v| v as f32),
            // OpenAI `tools` array, already serialized to a JSON string by
            // the serve layer; passed verbatim to the chat template.
            tools_json: req
                .params
                .get("tools")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned),
        };
        let mut chunk_write_failed = false;
        let mut sink = |delta: &str| {
            let note = RpcNotification {
                jsonrpc: "2.0".into(),
                method: N_CHAT_CHUNK.into(),
                params: json!({"request_id": request_id.clone(), "delta": delta}),
            };
            if writeln!(writer, "{}", encode(&RpcFrame::Notification(note))).is_err() {
                chunk_write_failed = true;
            }
        };
        let result = self
            .engine
            .chat(messages_json, &gen, &mut sink)
            .map_err(|e| to_rpc_error(&e))?;
        if chunk_write_failed {
            tracing::warn!("chat chunk write failed; supervisor pipe broken");
        }
        Ok(json!({
            "content": result.content,
            "finish_reason": result.finish_reason,
            "tool_calls": result.tool_calls,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
        }))
    }
}

/// Read an optional u32 field from request params, falling back to `default`.
fn u32_param(params: &Value, key: &str, default: u32) -> u32 {
    params
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|v| u32::try_from(v).ok())
        .unwrap_or(default)
}

/// Serialize a `HiggsError` into a JSON-RPC error, carrying its origin
/// diagnostic code (HG002/HG005/…) in `data.code`. The supervisor reconstructs
/// the code so the HTTP boundary maps the worker error to its true status (404
/// for not-found/not-loaded, 400 for context overflow, 503 for worker-down)
/// instead of collapsing every worker failure to a 500.
fn to_rpc_error(e: &crate::diagnostic::HiggsError) -> RpcError {
    let data = e
        .code()
        .map(|c| serde_json::json!({ "code": c.to_string() }));
    RpcError {
        code: -32000,
        message: e.to_string(),
        data,
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use serde_json::json;

    use super::*;
    use crate::rpc::{decode, RpcFrame};

    // ---------------------------------------------------------------------------
    // Test helpers
    // ---------------------------------------------------------------------------

    /// Shared record of engine calls, inspectable after `serve_state` consumes
    /// the [`FakeEngine`].
    type CallLog = std::sync::Arc<parking_lot::Mutex<Vec<String>>>;

    /// Scripted [`engine::HiggsEngine`]: records load/unload/chat calls; chat
    /// streams "he" then "llo" and returns ("hello", "stop").
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

        fn chat(
            &mut self,
            _messages_json: &str,
            _params: &engine::GenParams,
            sink: &mut dyn FnMut(&str),
        ) -> Result<engine::ChatResult, HiggsError> {
            self.calls.lock().push("chat".into());
            sink("he");
            sink("llo");
            Ok(engine::ChatResult {
                content: "hello".into(),
                finish_reason: "stop",
                tool_calls: None,
                // Scripted counts: 5 prompt tokens, 2 completion tokens.
                prompt_tokens: 5,
                completion_tokens: 2,
            })
        }
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

    /// Build a single NDJSON line from a request value.
    fn req_line(id: u64, method: &str, params: serde_json::Value) -> String {
        let r = RpcRequest {
            jsonrpc: "2.0".into(),
            id,
            method: method.into(),
            params,
        };
        format!("{}\n", encode(&RpcFrame::Request(r)))
    }

    /// Parse all non-empty lines in `buf` as RpcFrames and return them.
    fn parse_responses(buf: &[u8]) -> Vec<RpcFrame> {
        std::str::from_utf8(buf)
            .unwrap()
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| decode(l).expect("valid frame"))
            .collect()
    }

    // ---------------------------------------------------------------------------
    // Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn unknown_method_is_32601() {
        let input = req_line(2, "higgs/nope", json!(null));
        let mut out: Vec<u8> = Vec::new();
        serve(Cursor::new(input.as_bytes()), &mut out);

        let frames = parse_responses(&out);
        assert_eq!(frames.len(), 1);
        let RpcFrame::Response(resp) = &frames[0] else {
            panic!("expected response")
        };
        assert_eq!(resp.id, 2);
        let err = resp.error.as_ref().expect("expected error");
        assert_eq!(err.code, -32601);
        assert!(
            err.message.contains("unknown method"),
            "message was: {}",
            err.message
        );
    }

    #[test]
    fn load_then_chat_streams() {
        let path = "/models/google/gemma-4-12b/gemma-4-12b-Q4_K_M.gguf";
        let mut input = load_line(2, "google/gemma-4-12b", path, json!({}));
        input.push_str(&req_line(
            3,
            M_CHAT,
            json!({"request_id": 7, "messages_json": "[{\"role\":\"user\",\"content\":\"hi\"}]"}),
        ));

        let (frames, calls) = serve_with_fake(&input);
        assert_eq!(
            frames.len(),
            4,
            "load response, 2 chunks, chat response: {frames:?}"
        );

        let RpcFrame::Response(load) = &frames[0] else {
            panic!("expected load response")
        };
        assert_eq!(load.id, 2);
        assert_eq!(load.result.as_ref().unwrap()["id"], "google/gemma-4-12b");

        // The two chunk notifications arrive in order, BEFORE the final response.
        for (idx, delta) in [(1, "he"), (2, "llo")] {
            let RpcFrame::Notification(note) = &frames[idx] else {
                panic!("frame {idx} should be a notification: {:?}", frames[idx])
            };
            assert_eq!(note.method, N_CHAT_CHUNK);
            assert_eq!(note.params["request_id"], 7);
            assert_eq!(note.params["delta"], delta);
        }

        let RpcFrame::Response(chat) = &frames[3] else {
            panic!("expected chat response")
        };
        assert_eq!(chat.id, 3);
        let result = chat.result.as_ref().unwrap();
        assert_eq!(result["content"], "hello");
        assert_eq!(result["finish_reason"], "stop");

        let calls = calls.lock();
        assert_eq!(calls.len(), 2, "calls: {calls:?}");
        assert_eq!(calls[0], format!("load {path}"), "calls: {calls:?}");
        assert_eq!(calls[1], "chat");
    }

    #[test]
    fn chat_before_load_is_hg003() {
        let input = req_line(2, M_CHAT, json!({"request_id": 1, "messages_json": "[]"}));
        let (frames, calls) = serve_with_fake(&input);

        assert_eq!(frames.len(), 1);
        let RpcFrame::Response(resp) = &frames[0] else {
            panic!("expected response")
        };
        assert_eq!(resp.id, 2);
        let err = resp.error.as_ref().expect("expected error");
        assert!(
            err.message.contains("[HG003]"),
            "message was: {}",
            err.message
        );
        assert!(calls.lock().is_empty(), "engine must not be touched");
    }

    #[test]
    fn unload_then_status() {
        let path = "/models/google/gemma-4-12b/gemma-4-12b-Q4_K_M.gguf";
        let mut input = load_line(2, "google/gemma-4-12b", path, json!({"ctx_len": 2048}));
        input.push_str(&req_line(3, M_STATUS, json!(null)));
        input.push_str(&req_line(4, M_UNLOAD, json!(null)));
        input.push_str(&req_line(5, M_STATUS, json!(null)));

        let (frames, calls) = serve_with_fake(&input);
        assert_eq!(frames.len(), 4);

        let RpcFrame::Response(loaded_status) = &frames[1] else {
            panic!("expected response")
        };
        let loaded = &loaded_status.result.as_ref().unwrap()["loaded"];
        assert_eq!(loaded["id"], "google/gemma-4-12b");
        assert_eq!(loaded["ctx_len"], 2048);
        assert_eq!(loaded["threads"], 4, "default threads");

        let RpcFrame::Response(final_status) = &frames[3] else {
            panic!("expected response")
        };
        assert_eq!(final_status.result.as_ref().unwrap()["loaded"], Value::Null);
        assert!(calls.lock().contains(&"unload".to_string()));
    }

    #[test]
    fn load_without_path_is_invalid_params() {
        // Scan is host-side: M_LOAD must carry the resolved GGUF path. A request
        // missing `path` is a malformed call (-32602), and the engine is never
        // touched.
        let input = req_line(2, M_LOAD, json!({"id": "nope/nope"}));

        let (frames, calls) = serve_with_fake(&input);
        assert_eq!(frames.len(), 1);
        let RpcFrame::Response(resp) = &frames[0] else {
            panic!("expected response")
        };
        let err = resp.error.as_ref().expect("expected error");
        assert_eq!(err.code, -32602);
        assert!(
            err.message.contains("missing path"),
            "message was: {}",
            err.message
        );
        assert!(calls.lock().is_empty(), "engine must not load anything");
    }

    #[test]
    fn shutdown_ends_loop() {
        let mut input = String::new();
        input.push_str(&req_line(4, M_STATUS, json!(null)));
        input.push_str(&req_line(5, M_SHUTDOWN, json!(null)));
        input.push_str(&req_line(6, M_STATUS, json!(null))); // must NOT appear in output

        let mut out: Vec<u8> = Vec::new();
        serve(Cursor::new(input.as_bytes()), &mut out);

        let frames = parse_responses(&out);
        // Only responses for id=4 and id=5; loop stopped before processing id=6.
        assert_eq!(
            frames.len(),
            2,
            "expected exactly 2 responses, got: {frames:?}"
        );

        let ids: Vec<u64> = frames
            .iter()
            .map(|f| match f {
                RpcFrame::Response(r) => r.id,
                _ => panic!("expected response frame"),
            })
            .collect();
        assert!(ids.contains(&4));
        assert!(ids.contains(&5));
        assert!(
            !ids.contains(&6),
            "response for id=6 must not appear after shutdown"
        );
    }

    #[test]
    fn garbage_line_yields_parse_error() {
        let mut input = String::new();
        input.push_str("not json\n");
        input.push_str(&req_line(7, M_STATUS, json!(null)));

        let mut out: Vec<u8> = Vec::new();
        serve(Cursor::new(input.as_bytes()), &mut out);

        let frames = parse_responses(&out);
        assert_eq!(frames.len(), 2);

        // First response: id=0, code=-32700 (parse error).
        let RpcFrame::Response(first) = &frames[0] else {
            panic!("expected response")
        };
        assert_eq!(first.id, 0);
        let err = first
            .error
            .as_ref()
            .expect("expected error on garbage line");
        assert_eq!(err.code, -32700);

        // Second response: id=7, success.
        let RpcFrame::Response(second) = &frames[1] else {
            panic!("expected response")
        };
        assert_eq!(second.id, 7);
        assert!(
            second.error.is_none(),
            "id=7 should succeed, got: {:?}",
            second.error
        );
    }
}
