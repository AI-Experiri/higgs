//! Shared test harness for the serve-layer handler tests.
//!
//! Spins up a [`Supervisor`] over duplex pipes with a mock worker, wraps it in a
//! [`Higgs`] facade, and builds the [`router`] — so the `v1` and `control`
//! handler tests drive the real router without a real worker process. Shared by
//! both surfaces' test modules via `super::test_support::*`.

use std::collections::VecDeque;
use std::io::Cursor;
use std::sync::Arc;

use axum::body::Body;
use axum::http::Request;
use axum::response::Response;
use axum::Router;
use ggus::{GGufFileHeader, GGufFileWriter, GGufMetaDataValueType};
use http_body_util::BodyExt;
use parking_lot::Mutex;
use serde_json::json;
use tokio::io::AsyncWriteExt;

use super::router;
use crate::api::{Higgs, HiggsConfig};
use crate::diagnostic::HiggsError;
use crate::rpc::{encode, RpcFrame, RpcResponse};
use crate::supervisor::{Supervisor, WorkerHalves};

/// Build a `Supervisor` plus duplex test handles and its captured log ring.
pub(crate) fn make_supervisor() -> (
    Supervisor,
    tokio::io::DuplexStream, // test_write: write responses → supervisor reads
    tokio::io::DuplexStream, // test_read:  supervisor writes requests → test reads
    Arc<Mutex<VecDeque<String>>>, // stderr ring (push lines for logs tests)
) {
    let (sup_write, test_read) = tokio::io::duplex(64 * 1024);
    let (test_write, sup_read) = tokio::io::duplex(64 * 1024);

    let sup_write_cell = Arc::new(Mutex::new(Some(sup_write)));
    let sup_read_cell = Arc::new(Mutex::new(Some(sup_read)));
    let ring_cell: Arc<Mutex<Option<Arc<Mutex<VecDeque<String>>>>>> = Arc::new(Mutex::new(None));
    let ring_capture = Arc::clone(&ring_cell);

    let sup = Supervisor::with_factory(Box::new(move |ring, _model| {
        *ring_capture.lock() = Some(ring);
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

    sup.start_for("test-model").expect("mock start");
    let ring = ring_cell.lock().take().expect("factory ran on start");
    (sup, test_write, test_read, ring)
}

/// Build a `Supervisor` that has NEVER spawned a worker — its factory is never
/// invoked, so `status()` reports `worker_alive:false` with `loaded:None`. This
/// reproduces higgs's normal idle state (spawn-on-load: nothing loaded ⇒ no
/// worker) for the `/v1` idle-behavior tests.
pub(crate) fn make_idle_supervisor() -> Supervisor {
    Supervisor::with_factory(Box::new(|_ring, _model| {
        Err(HiggsError::WorkerSpawnFailed {
            source: std::io::Error::other("mock: idle supervisor never spawns"),
        })
    }))
}

/// Write a JSON-RPC success response to the supervisor's read side.
pub(crate) async fn write_response(
    stream: &mut tokio::io::DuplexStream,
    id: u64,
    result: serde_json::Value,
) {
    let line = encode(&RpcFrame::Response(RpcResponse {
        jsonrpc: "2.0".into(),
        id,
        result: Some(result),
        error: None,
    }));
    stream
        .write_all(format!("{line}\n").as_bytes())
        .await
        .unwrap();
    stream.flush().await.unwrap();
}

/// Wrap a mock supervisor in a `Higgs` facade and build the router.
pub(crate) fn make_app(sup: Supervisor) -> Router {
    router(Arc::new(Higgs::with_supervisor(
        Arc::new(sup),
        HiggsConfig::default(),
    )))
}

/// Build an app whose host-side `scan()` reads `lmstudio_dirs`.
///
/// Scan runs host-side now, so control tests that need a discoverable model
/// point the config at a temp LM Studio fixture dir (see [`write_gguf_fixture`])
/// instead of injecting models through the worker.
pub(crate) fn make_app_with_lmstudio(sup: Supervisor, dir: std::path::PathBuf) -> Router {
    let cfg = HiggsConfig {
        lmstudio_dirs: vec![dir],
        hf_dirs: vec![],
        ollama_dirs: vec![],
        default_load: HiggsConfig::default().default_load,
    };
    router(Arc::new(Higgs::with_supervisor(Arc::new(sup), cfg)))
}

/// Write a minimal valid GGUF file (arch=llama, ctx=4096, chat template) at
/// `<root>/<id>/model-Q4_K_M.gguf` so a host-side scan discovers `id` with
/// enriched metadata. Returns nothing; the caller owns the temp dir.
pub(crate) fn write_gguf_fixture(root: &std::path::Path, id: &str) {
    fn gguf_string(s: &str) -> Vec<u8> {
        let bytes = s.as_bytes();
        let mut out = Vec::with_capacity(8 + bytes.len());
        out.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
        out.extend_from_slice(bytes);
        out
    }

    let header = GGufFileHeader::new(3, 0, 3);
    let mut buf = Cursor::new(Vec::<u8>::new());
    let mut writer = GGufFileWriter::new(&mut buf, header).unwrap();
    writer
        .write_meta_kv(
            "general.architecture",
            GGufMetaDataValueType::String,
            &gguf_string("llama"),
        )
        .unwrap();
    writer
        .write_meta_kv(
            "llama.context_length",
            GGufMetaDataValueType::U32,
            &4096u32.to_le_bytes(),
        )
        .unwrap();
    writer
        .write_meta_kv(
            "tokenizer.chat_template",
            GGufMetaDataValueType::String,
            &gguf_string("{% for m in messages %}{{ m.content }}{% endfor %}"),
        )
        .unwrap();
    writer.finish::<Vec<u8>>(false).finish().unwrap();

    let path = root.join(id).join("model-Q4_K_M.gguf");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, buf.into_inner()).unwrap();
}

/// A `GET` request to `uri`. Carries a loopback `Host` so it passes the
/// serve-layer DNS-rebinding guard (`host_guard`).
pub(crate) fn get(uri: &str) -> Request<Body> {
    Request::builder()
        .uri(uri)
        .header("host", "127.0.0.1")
        .body(Body::empty())
        .unwrap()
}

/// A `POST` request to `uri` with a JSON body. Carries a loopback `Host` so
/// it passes the serve-layer DNS-rebinding guard (`host_guard`).
pub(crate) fn post_json(uri: &str, body: &serde_json::Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(uri)
        .header("host", "127.0.0.1")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

/// Collect a response body into bytes.
pub(crate) async fn body_bytes(resp: Response) -> Vec<u8> {
    resp.into_body()
        .collect()
        .await
        .unwrap()
        .to_bytes()
        .to_vec()
}

/// A canonical `higgs/status` response with one loaded model.
pub(crate) fn loaded_status_json() -> serde_json::Value {
    json!({
        "loaded": { "id": "org/model", "ctx_len": 4096, "gpu_layers": 99, "threads": 4 },
        "models_scanned": 1,
    })
}
