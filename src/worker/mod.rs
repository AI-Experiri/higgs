//! Worker role: runs inside the re-exec'd process. Owns the ModelStore and
//! (from Task 6) the engine. Speaks NDJSON JSON-RPC on stdin/stdout; logs to
//! stderr. The supervisor is the ONLY client.

pub mod models;

use std::io::{BufRead, Write};

use serde_json::{json, Value};

use crate::rpc::{decode, encode, RpcError, RpcFrame, RpcRequest, RpcResponse};
use models::ModelStore;

/// Method names — the only vocabulary on the supervisor↔worker wire.
pub const M_SCAN: &str = "higgs/scan";
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
fn serve(reader: impl BufRead, mut writer: impl Write) {
    let mut state = WorkerState::default();
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
                respond(&mut writer, 0, Err(RpcError { code: -32700, message: e.to_string() }));
            }
        }
    }
}

fn respond(writer: &mut impl Write, id: u64, out: Result<Value, RpcError>) {
    let resp = match out {
        Ok(result) => RpcResponse { jsonrpc: "2.0".into(), id, result: Some(result), error: None },
        Err(error) => RpcResponse { jsonrpc: "2.0".into(), id, result: None, error: Some(error) },
    };
    let _ = writeln!(writer, "{}", encode(&RpcFrame::Response(resp)));
}

/// Worker-held state: catalog + (Task 6) the loaded engine.
#[derive(Default)]
struct WorkerState {
    store: ModelStore,
}

impl WorkerState {
    fn dispatch(&mut self, req: &RpcRequest, _writer: &mut impl Write) -> Result<Value, RpcError> {
        match req.method.as_str() {
            M_SCAN => {
                let dirs = |k: &str| {
                    req.params
                        .get(k)
                        .and_then(Value::as_array)
                        .map(|a| {
                            a.iter()
                                .filter_map(Value::as_str)
                                .map(Into::into)
                                .collect::<Vec<std::path::PathBuf>>()
                        })
                        .unwrap_or_default()
                };
                self.store
                    .scan(&dirs("lmstudio"), &dirs("hf"), &dirs("ollama"))
                    .map(|models| serde_json::to_value(models).expect("serializable"))
                    .map_err(|e| to_rpc_error(&e))
            }
            M_STATUS => {
                Ok(json!({"loaded": Value::Null, "models_scanned": self.store.models().len()}))
            }
            M_LOAD | M_UNLOAD | M_CHAT => {
                Err(RpcError { code: -32601, message: "engine lands in Task 6".into() })
            }
            other => {
                Err(RpcError { code: -32601, message: format!("unknown method {other}") })
            }
        }
    }
}

fn to_rpc_error(e: &crate::diagnostic::HiggsError) -> RpcError {
    RpcError { code: -32000, message: e.to_string() }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::io::Cursor;
    use std::path::Path;

    use serde_json::json;
    use tempfile::TempDir;

    use super::*;
    use crate::rpc::{decode, RpcFrame};

    // ---------------------------------------------------------------------------
    // Test helpers
    // ---------------------------------------------------------------------------

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

    fn write_file(path: &Path, content: &[u8]) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, content).unwrap();
    }

    // ---------------------------------------------------------------------------
    // Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn scan_over_fixture() {
        let dir = TempDir::new().unwrap();
        let root = dir.path();
        // LM Studio layout: root/google/gemma-4-12b/gemma-4-12b-Q4_K_M.gguf
        write_file(
            &root.join("google/gemma-4-12b/gemma-4-12b-Q4_K_M.gguf"),
            b"dummy",
        );

        let root_str = root.to_str().unwrap();
        let input = req_line(1, M_SCAN, json!({"lmstudio": [root_str], "hf": [], "ollama": []}));

        let mut out: Vec<u8> = Vec::new();
        serve(Cursor::new(input.as_bytes()), &mut out);

        let frames = parse_responses(&out);
        assert_eq!(frames.len(), 1);
        let RpcFrame::Response(resp) = &frames[0] else { panic!("expected response") };
        assert_eq!(resp.id, 1);
        assert!(resp.error.is_none(), "unexpected error: {:?}", resp.error);

        let result = resp.result.as_ref().unwrap();
        let arr = result.as_array().expect("result should be array");
        assert_eq!(arr.len(), 1, "expected exactly one model");
        assert_eq!(arr[0]["id"], "google/gemma-4-12b");
    }

    #[test]
    fn unknown_method_is_32601() {
        let input = req_line(2, "higgs/nope", json!(null));
        let mut out: Vec<u8> = Vec::new();
        serve(Cursor::new(input.as_bytes()), &mut out);

        let frames = parse_responses(&out);
        assert_eq!(frames.len(), 1);
        let RpcFrame::Response(resp) = &frames[0] else { panic!("expected response") };
        assert_eq!(resp.id, 2);
        let err = resp.error.as_ref().expect("expected error");
        assert_eq!(err.code, -32601);
        assert!(err.message.contains("unknown method"), "message was: {}", err.message);
    }

    #[test]
    fn engine_methods_not_ready() {
        let input = req_line(3, M_LOAD, json!({"id": "google/gemma-4-12b"}));
        let mut out: Vec<u8> = Vec::new();
        serve(Cursor::new(input.as_bytes()), &mut out);

        let frames = parse_responses(&out);
        assert_eq!(frames.len(), 1);
        let RpcFrame::Response(resp) = &frames[0] else { panic!("expected response") };
        assert_eq!(resp.id, 3);
        let err = resp.error.as_ref().expect("expected error");
        assert_eq!(err.code, -32601);
        assert!(
            err.message.contains("engine lands in Task 6"),
            "message was: {}",
            err.message,
        );
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
        assert_eq!(frames.len(), 2, "expected exactly 2 responses, got: {frames:?}");

        let ids: Vec<u64> = frames
            .iter()
            .map(|f| match f {
                RpcFrame::Response(r) => r.id,
                _ => panic!("expected response frame"),
            })
            .collect();
        assert!(ids.contains(&4));
        assert!(ids.contains(&5));
        assert!(!ids.contains(&6), "response for id=6 must not appear after shutdown");
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
        let RpcFrame::Response(first) = &frames[0] else { panic!("expected response") };
        assert_eq!(first.id, 0);
        let err = first.error.as_ref().expect("expected error on garbage line");
        assert_eq!(err.code, -32700);

        // Second response: id=7, success.
        let RpcFrame::Response(second) = &frames[1] else { panic!("expected response") };
        assert_eq!(second.id, 7);
        assert!(second.error.is_none(), "id=7 should succeed, got: {:?}", second.error);
    }
}
