//! Shared test harness for the serve-layer handler tests.
//!
//! Builds a [`Higgs`] facade over a co-located LOCAL [`NodeRuntime`] whose workers
//! are STATEFUL fakes ([`fake_worker_factory_stateful`]) — they auto-respond to
//! load/status/chat without llama.cpp. So the handler tests drive the REAL
//! load/chat path (load a fixture model, then assert) instead of hand-driving
//! worker stdio. "Nothing loaded" is the natural idle state (no resident worker),
//! so there is no separate idle-supervisor seam anymore. Shared by both surfaces'
//! test modules via `super::test_support::*`.

use std::io::Cursor;
use std::path::PathBuf;
use std::sync::Arc;

use axum::body::Body;
use axum::http::Request;
use axum::response::Response;
use axum::Router;
use ggus::{GGufFileHeader, GGufFileWriter, GGufMetaDataValueType};
use http_body_util::BodyExt;

use super::router;
use crate::api::{Higgs, HiggsConfig};
use crate::log_bus::LogBus;
use crate::node::runtime::{NodeConfig, NodeRuntime, DEFAULT_IDLE_TTL};
use crate::node::test_support::fake_worker_factory_stateful;
use crate::supervisor::Supervisor;

/// Build a fake-worker-backed LOCAL [`NodeRuntime`] scanning `dirs`, sharing `bus`
/// so logs tests can seed the same Developer-Log history `higgs.logs()` reads.
fn make_node(dirs: Vec<PathBuf>, bus: Arc<LogBus>) -> NodeRuntime {
    NodeRuntime::with_spawner(
        NodeConfig {
            bus,
            lmstudio_dirs: dirs,
            hf_dirs: vec![],
            ollama_dirs: vec![],
            idle_ttl: DEFAULT_IDLE_TTL,
        },
        Arc::new(|_bus| Supervisor::with_factory(fake_worker_factory_stateful())),
    )
}

/// A `Higgs` facade over a stateful-fake-worker node scanning `dirs`, plus the
/// node's shared [`LogBus`]. Both the node and the facade config see `dirs`.
fn node_higgs(dirs: Vec<PathBuf>) -> (Arc<Higgs>, Arc<LogBus>) {
    let bus = Arc::new(LogBus::new());
    let node = make_node(dirs.clone(), bus.clone());
    let cfg = HiggsConfig {
        lmstudio_dirs: dirs,
        hf_dirs: vec![],
        ollama_dirs: vec![],
        default_load: HiggsConfig::default().default_load,
    };
    (Arc::new(Higgs::with_local(Arc::new(node), cfg)), bus)
}

/// A `Higgs` facade scanning nothing (the common idle starting point).
pub(crate) fn make_higgs() -> Arc<Higgs> {
    node_higgs(vec![]).0
}

/// A `Higgs` facade whose host-side `scan()` + node both read `dir` (an LM Studio
/// fixture root). Tests `higgs.load("org/model", None)` against a fixture there.
pub(crate) fn make_higgs_with_lmstudio(dir: PathBuf) -> Arc<Higgs> {
    node_higgs(vec![dir]).0
}

/// A `Higgs` facade plus its node's Developer-Log bus — for the logs tests, which
/// seed history before hitting the endpoint.
pub(crate) fn make_higgs_with_bus() -> (Arc<Higgs>, Arc<LogBus>) {
    node_higgs(vec![])
}

/// Wrap a `Higgs` (typically after a `load`) in the serve router.
pub(crate) fn app_for(higgs: Arc<Higgs>) -> Router {
    router(higgs)
}

/// The serve router over a fresh idle facade (nothing loaded, JIT on, serving on).
pub(crate) fn make_app() -> Router {
    router(make_higgs())
}

/// The serve router over a facade whose `scan()` reads `dir` (LM Studio fixture).
pub(crate) fn make_app_with_lmstudio(dir: PathBuf) -> Router {
    router(make_higgs_with_lmstudio(dir))
}

/// Like [`make_app_with_lmstudio`] but Prepares (autotunes) `id` first, so the
/// JIT readiness gate admits it. Use for tests that exercise the JIT load/serve
/// or post-load validation paths — an un-prepared model is refused by the gate
/// before those paths run, so they need a fresh, matching profile in place.
pub(crate) async fn make_app_with_lmstudio_prepared(dir: PathBuf, id: &str) -> Router {
    let higgs = make_higgs_with_lmstudio(dir);
    seed_prepared_profile(&higgs, id).await;
    app_for(higgs)
}

/// Seed a fresh tuning profile DIRECTLY into the store — NOT `Higgs::tune`, which
/// runs a bounded HF card fetch (`fetch_card_bounded`) that stalls ~10s on offline
/// or firewalled CI. Anchored to the CURRENT hardware + model file so
/// `profile_state` reads it as `Ready` (the JIT gate admits it), exactly like a
/// real Prepare. The serve tests use FAKE workers, so the profile's params don't
/// need to load llama.cpp — only the staleness anchors matter. Keeps these unit
/// tests hermetic and fast.
async fn seed_prepared_profile(higgs: &Higgs, id: &str) {
    use crate::worker::engine::{CtxLen, GpuLayers, LoadParams};
    let hw = higgs.hardware().await;
    let path = higgs
        .scan()
        .await
        .ok()
        .and_then(|ms| ms.into_iter().find(|m| m.id == id).map(|m| m.path))
        .expect("fixture model is scannable");
    let store = higgs.models_store().expect("open models store");
    store.put_tuning(
        id,
        crate::tune::store::TuneRecord {
            profile: LoadParams::base(CtxLen::Auto, GpuLayers::All, 8),
            sampling: crate::worker::engine::SamplingParams::default(),
            budget: crate::tune::ResourceBudget::default(),
            provenance: crate::tune::TuneProvenance::Heuristic,
            bench_tps: None,
            tuned_at_ms: 0,
            hw_fingerprint: hw.fingerprint(),
            model_file_sig: crate::api::file_sig(&path),
        },
    );
    store.flush().expect("persist seeded profile");
}

/// The serve router with JIT turned OFF — for the `/v1` tests that assert the
/// explicit-load HG003 404 path (chat against an unloaded model).
pub(crate) fn make_app_jit_off() -> Router {
    let higgs = make_higgs();
    higgs.set_jit_enabled(false);
    router(higgs)
}

/// The serve router with serving turned OFF — for the test that asserts the
/// serving-disabled HG019 503 path (`/v1` refuses while serving is off).
pub(crate) fn make_app_serving_off() -> Router {
    let higgs = make_higgs();
    higgs.set_serving_enabled(false);
    router(higgs)
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

/// A `PUT` request to `uri` with a JSON body. Carries a loopback `Host` so it
/// passes the serve-layer DNS-rebinding guard (`host_guard`).
pub(crate) fn put_json(uri: &str, body: &serde_json::Value) -> Request<Body> {
    Request::builder()
        .method("PUT")
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
