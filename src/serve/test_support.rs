//! Shared test harness for the serve-layer handler tests.
//!
//! Builds a [`Higgs`] facade over a co-located LOCAL [`NodeRuntime`] whose workers
//! are STATEFUL fakes ([`fake_worker_factory_stateful`]) — they auto-respond to
//! load/status/chat without llama.cpp. So the handler tests drive the REAL
//! load/chat path (load a fixture model, then assert) instead of hand-driving
//! worker stdio. "Nothing loaded" is the natural idle state (no resident worker),
//! so there is no separate idle-supervisor seam anymore. Shared by both surfaces'
//! test modules via `super::test_support::*`.

use std::path::PathBuf;
use std::sync::Arc;

use axum::body::Body;
use axum::http::Request;
use axum::response::Response;
use axum::Router;
use http_body_util::BodyExt;

use super::v1_router;
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
        worker_exe: None,
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

/// Wrap a `Higgs` (typically after a `load`) in the serve router.
pub(crate) fn app_for(higgs: Arc<Higgs>) -> Router {
    v1_router(higgs)
}

/// The serve router over a fresh idle facade (nothing loaded, JIT on, serving on).
pub(crate) fn make_app() -> Router {
    v1_router(make_higgs())
}

/// The serve router over a facade whose `scan()` reads `dir` (LM Studio fixture).
pub(crate) fn make_app_with_lmstudio(dir: PathBuf) -> Router {
    v1_router(make_higgs_with_lmstudio(dir))
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

/// The serve router with JIT turned OFF — for the `/v1` tests that assert the
/// explicit-load HG003 404 path (chat against an unloaded model).
pub(crate) fn make_app_jit_off() -> Router {
    let higgs = make_higgs();
    higgs.set_jit_enabled(false);
    v1_router(higgs)
}

/// The serve router with serving turned OFF — for the test that asserts the
/// serving-disabled HG019 503 path (`/v1` refuses while serving is off).
pub(crate) fn make_app_serving_off() -> Router {
    let higgs = make_higgs();
    higgs.set_serving_enabled(false);
    v1_router(higgs)
}

// The GGUF fixture bytes live in ONE place — `crate::fixtures` (also reachable by
// an embedder's tests through the `test-support` feature) — so the [HG079] domain
// fixtures cannot drift between the two suites. Re-exported here under the names
// the serve/api tests already use.
pub(crate) use crate::fixtures::{
    seed_prepared_profile, write_embedding_gguf_fixture, write_embedding_gguf_fixture_named,
    write_gguf_fixture, write_reranker_gguf_fixture,
};

/// A `GET` request to `uri`. Carries a loopback `Host` so it passes the
/// serve-layer DNS-rebinding guard (`host_guard`).
pub(crate) fn get(uri: &str) -> Request<Body> {
    Request::builder()
        .uri(uri)
        .header("host", "127.0.0.1")
        .body(Body::empty())
        .unwrap()
}

pub(crate) fn post_json(uri: &str, body: &serde_json::Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(uri)
        .header("host", "127.0.0.1")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

/// Attach a bearer token to a built request (G4 keys tests).
pub(crate) fn with_bearer(mut req: Request<Body>, token: &str) -> Request<Body> {
    req.headers_mut().insert(
        axum::http::header::AUTHORIZATION,
        format!("Bearer {token}").parse().expect("bearer header"),
    );
    req
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
