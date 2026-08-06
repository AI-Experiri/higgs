use super::*;

use crate::download::Fetcher;

/// Minimal fake fetcher (mirrors `download_tests::FakeFetcher`): either
/// streams `chunks` or fails with a constructed classified error.
struct FakeFetcher {
    chunks: Vec<Vec<u8>>,
    fail_with: Option<Box<dyn Fn() -> HiggsError + Send + Sync>>,
}

impl Fetcher for FakeFetcher {
    async fn fetch(
        &self,
        _target: &crate::download::PullTarget,
        on_chunk: &mut (dyn FnMut(&[u8]) + Send),
        progress: &mut (dyn FnMut(u64, Option<u64>) + Send),
    ) -> Result<(), HiggsError> {
        if let Some(mk) = &self.fail_with {
            return Err(mk());
        }
        let total: u64 = self.chunks.iter().map(|c| c.len() as u64).sum();
        let mut sent = 0u64;
        for c in &self.chunks {
            on_chunk(c);
            sent += c.len() as u64;
            progress(sent, Some(total));
        }
        Ok(())
    }
}

fn transport_err() -> HiggsError {
    HiggsError::HubTransport {
        repo: "acme/m".into(),
        detail: "injected".into(),
    }
}

#[tokio::test]
async fn pull_lands_the_file_in_the_org_model_layout_with_progress() {
    let root = tempfile::tempdir().expect("root");
    let ok = FakeFetcher {
        chunks: vec![b"gg".to_vec(), b"uf".to_vec()],
        fail_with: None,
    };
    let never = FakeFetcher {
        chunks: vec![],
        fail_with: Some(Box::new(|| panic!("fallback must not run"))),
    };
    let mut seen: Vec<(u64, Option<u64>)> = Vec::new();
    let path = pull_with(
        "acme/m",
        "m-Q4_K_M.gguf",
        root.path(),
        &ok,
        &never,
        &mut |d, t| seen.push((d, t)),
    )
    .await
    .expect("pull");
    assert_eq!(path, root.path().join("acme/m/m-Q4_K_M.gguf"));
    assert_eq!(std::fs::read(&path).unwrap(), b"gguf");
    assert_eq!(seen, vec![(2, Some(4)), (4, Some(4))]);
}

#[tokio::test]
async fn pull_falls_back_when_the_primary_fetcher_fails() {
    let root = tempfile::tempdir().expect("root");
    let bad = FakeFetcher {
        chunks: vec![],
        fail_with: Some(Box::new(transport_err)),
    };
    let good = FakeFetcher {
        chunks: vec![b"ok".to_vec()],
        fail_with: None,
    };
    let path = pull_with("acme/m", "m.gguf", root.path(), &bad, &good, &mut |_, _| {})
        .await
        .expect("fallback succeeds");
    assert_eq!(std::fs::read(&path).unwrap(), b"ok");
}

/// `pull` (the production entry: models_dir + real hub/reqwest fetchers)
/// against the loopback fixture Hub, landing under `$HIGGS_HOME/models`.
#[test]
fn pull_downloads_via_the_real_fetchers_into_higgs_home() {
    let _env = crate::TEST_ENV_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(async {
        let hub = crate::catalog::test_support::fixture_hub().await;
        let home = tempfile::tempdir().expect("home");
        let _redirect =
            crate::catalog::test_support::EnvRedirect::set(&hub.endpoint, Some(home.path()));
        let mut reported = 0u64;
        let path = pull("acme/tiny", "tiny-Q4_K_M.gguf", &mut |d, _t| reported = d)
            .await
            .expect("pull");
        assert_eq!(path, home.path().join("models/acme/tiny/tiny-Q4_K_M.gguf"));
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(bytes, b"GGUF-fixture-bytes");
        assert_eq!(reported, bytes.len() as u64);
    });
}

#[test]
fn progress_gate_emits_first_then_throttles_by_interval() {
    use std::time::{Duration, Instant};
    let t0 = Instant::now();
    let mut gate = ProgressGate::new(Duration::from_millis(250));
    assert!(gate.should_emit(t0), "first report always emits");
    assert!(!gate.should_emit(t0 + Duration::from_millis(100)));
    assert!(gate.should_emit(t0 + Duration::from_millis(300)));
    assert!(
        !gate.should_emit(t0 + Duration::from_millis(400)),
        "interval restarts from the last EMITTED report"
    );
}

#[tokio::test]
async fn pull_refuses_a_non_gguf_file_before_any_fetch() {
    let root = tempfile::tempdir().expect("root");
    let never = FakeFetcher {
        chunks: vec![],
        fail_with: Some(Box::new(|| panic!("fetcher must not run"))),
    };
    let err = pull_with(
        "acme/m",
        "evil.txt",
        root.path(),
        &never,
        &never,
        &mut |_, _| {},
    )
    .await
    .expect_err("refused");
    assert!(matches!(err, HiggsError::DownloadFailed { .. }));
}
