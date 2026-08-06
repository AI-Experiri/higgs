//! Test-only loopback "Hugging Face" for catalog unit tests: serves the model
//! list, model info, paths-info, and resolve downloads for one canned repo
//! (`acme/tiny`), so `HfSource` and everything above it is exercised over the
//! crate's REAL HTTP paths with `HIGGS_HF_ENDPOINT` pointed here. The env
//! mutation itself is the callers' job (under `TEST_ENV_LOCK`).

use std::future::IntoFuture;
use std::sync::{Arc, Mutex};

use axum::extract::{Path as AxumPath, RawQuery};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use serde_json::json;

/// A running fixture Hub: its `http://…` endpoint and the log of `/api/models`
/// query strings (for asserting the search parameters actually sent).
pub(crate) struct FixtureHub {
    pub(crate) endpoint: String,
    pub(crate) list_queries: Arc<Mutex<Vec<String>>>,
}

/// Spawn the fixture Hub on a loopback port (serves until the runtime drops).
pub(crate) async fn fixture_hub() -> FixtureHub {
    let list_queries = Arc::new(Mutex::new(Vec::<String>::new()));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    let log = list_queries.clone();
    let app =
        axum::Router::new()
            .route(
                "/api/models",
                get(move |RawQuery(q): RawQuery| {
                    let log = log.clone();
                    async move {
                        log.lock().unwrap().push(q.unwrap_or_default());
                        axum::Json(json!([
                            {
                                "id": "acme/tiny",
                                "author": "acme",
                                "downloads": 42u64,
                                "likes": 7u64,
                                "lastModified": "2026-07-30T00:00:00.000Z",
                                "pipeline_tag": "text-generation",
                            },
                            { "id": "acme/big" },
                        ]))
                    }
                }),
            )
            .route(
                "/api/models/{org}/{name}",
                get(|| async {
                    axum::Json(json!({
                        "id": "acme/tiny",
                        "author": "acme",
                        "downloads": 42u64,
                        "likes": 7u64,
                        "tags": ["gguf"],
                        "gguf": { "architecture": "llama", "total": 1_000_000u64,
                                  "context_length": 4096u64 },
                        "siblings": [
                            { "rfilename": "README.md" },
                            { "rfilename": "tiny-Q4_K_M.gguf" },
                            { "rfilename": "tiny-F16.gguf" },
                        ],
                    }))
                }),
            )
            .route(
                "/api/models/{org}/{name}/paths-info/{rev}",
                post(|| async {
                    axum::Json(json!([
                        { "type": "file", "oid": "a", "size": 134u64,
                          "path": "tiny-Q4_K_M.gguf", "lfs": { "size": 4_000u64 } },
                        { "type": "file", "oid": "b", "size": 9_000u64, "path": "tiny-F16.gguf" },
                    ]))
                }),
            )
            .route(
                "/{org}/{name}/resolve/{rev}/{file}",
                get(
                    |AxumPath((_org, name, _rev, file)): AxumPath<(
                        String,
                        String,
                        String,
                        String,
                    )>| async move {
                        match (name.as_str(), file.as_str()) {
                            // A repo with no README — the not-found → `None` path.
                            ("noreadme", "README.md") => {
                                (axum::http::StatusCode::NOT_FOUND, "missing").into_response()
                            }
                            // An oversized README — the byte-bound path.
                            ("bigreadme", "README.md") => {
                                "a".repeat(2 * 1024 * 1024).into_response()
                            }
                            (_, "README.md") => "# tiny\nhello".into_response(),
                            _ => b"GGUF-fixture-bytes".to_vec().into_response(),
                        }
                    },
                ),
            );
    tokio::spawn(axum::serve(listener, app).into_future());
    FixtureHub {
        endpoint: format!("http://{addr}"),
        list_queries,
    }
}

/// Point `HIGGS_HF_ENDPOINT` (and optionally `HIGGS_HOME`) somewhere for the
/// duration of a test, restoring the previous values on drop. The caller MUST
/// hold `crate::TEST_ENV_LOCK` first — this only does the set/restore.
pub(crate) struct EnvRedirect {
    prev_endpoint: Option<std::ffi::OsString>,
    prev_home: Option<std::ffi::OsString>,
    set_home: bool,
}

impl EnvRedirect {
    pub(crate) fn set(endpoint: &str, home: Option<&std::path::Path>) -> Self {
        let prev_endpoint = std::env::var_os("HIGGS_HF_ENDPOINT");
        let prev_home = std::env::var_os("HIGGS_HOME");
        // SAFETY: caller holds TEST_ENV_LOCK; restored on drop.
        unsafe { std::env::set_var("HIGGS_HF_ENDPOINT", endpoint) };
        if let Some(home) = home {
            unsafe { std::env::set_var("HIGGS_HOME", home) };
        }
        Self {
            prev_endpoint,
            prev_home,
            set_home: home.is_some(),
        }
    }
}

impl Drop for EnvRedirect {
    fn drop(&mut self) {
        // SAFETY: caller still holds TEST_ENV_LOCK.
        unsafe {
            match &self.prev_endpoint {
                Some(v) => std::env::set_var("HIGGS_HF_ENDPOINT", v),
                None => std::env::remove_var("HIGGS_HF_ENDPOINT"),
            }
            if self.set_home {
                match &self.prev_home {
                    Some(v) => std::env::set_var("HIGGS_HOME", v),
                    None => std::env::remove_var("HIGGS_HOME"),
                }
            }
        }
    }
}
