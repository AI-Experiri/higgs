
use super::*;

fn tune_record() -> TuneRecord {
    TuneRecord {
        profile: LoadParams::base(8192, u32::MAX, 8),
        sampling: SamplingParams::default(),
        budget: ResourceBudget::default(),
        provenance: TuneProvenance::Heuristic,
        bench_tps: None,
        tuned_at_ms: 100,
    }
}

fn perf_sample(gen: f32) -> BenchResult {
    BenchResult {
        gen_tps: gen,
        prompt_tps: gen * 4.0,
        ttft_ms: 50.0,
    }
}

fn meta_cache(path: &str, size: u64, mtime: u64) -> ModelMetaCache {
    ModelMetaCache {
        meta: ModelMeta {
            id: "org/m".into(),
            size_bytes: size,
            block_count: Some(32),
            ..Default::default()
        },
        key: MetaCacheKey {
            path: path.into(),
            size_bytes: size,
            mtime_ms: mtime,
        },
    }
}

#[test]
fn store_roundtrips_tuning_and_perf_and_invalidates_meta() {
    let home = tempfile::tempdir().unwrap();
    let s = JsonModelStore::open(home.path()).unwrap();
    s.put_tuning("org/m", tune_record());
    s.record_perf("org/m", perf_sample(42.0), 1000);
    s.flush().unwrap();

    // Reopen: tuning + perf persisted durably.
    let s2 = JsonModelStore::open(home.path()).unwrap();
    assert!(s2.tuning("org/m").is_some());
    assert!((s2.perf("org/m").unwrap().avg_gen_tps - 42.0).abs() < 0.1);
    assert_eq!(s2.perf("org/m").unwrap().samples, 1);

    // Meta cache invalidates when {size,mtime} changes; tuning survives.
    s2.put_meta("org/m", meta_cache("/x.gguf", 100, 1));
    assert!(s2.meta_if_fresh("org/m", "/x.gguf", 100, 1).is_some());
    assert!(
        s2.meta_if_fresh("org/m", "/x.gguf", 100, 2).is_none(),
        "mtime changed → stale"
    );
    assert!(
        s2.meta_if_fresh("org/m", "/other.gguf", 100, 1).is_none(),
        "path changed → stale"
    );
    assert!(
        s2.tuning("org/m").is_some(),
        "tuning retained across meta change"
    );
}

#[test]
fn set_profile_updates_profile_preserving_other_fields() {
    let home = tempfile::tempdir().unwrap();
    let s = JsonModelStore::open(home.path()).unwrap();
    // With an existing record, set_profile changes ONLY the profile + timestamp.
    let mut rec = tune_record();
    rec.provenance = TuneProvenance::Bench;
    rec.bench_tps = Some(9.0);
    s.put_tuning("org/m", rec);
    s.set_profile("org/m", LoadParams::base(1234, 0, 2), 99);
    let got = s.tuning("org/m").unwrap();
    assert_eq!(got.profile.ctx_len(), 1234, "profile updated");
    assert_eq!(got.profile.gpu_layers(), 0, "CPU-only accepted profile");
    assert_eq!(
        got.provenance,
        TuneProvenance::Bench,
        "provenance preserved"
    );
    assert_eq!(got.bench_tps, Some(9.0), "bench_tps preserved");
    assert_eq!(got.tuned_at_ms, 99);
    // With NO prior record, set_profile creates one with defaults.
    s.set_profile("org/new", LoadParams::base(2048, u32::MAX, 8), 50);
    let fresh = s.tuning("org/new").unwrap();
    assert_eq!(fresh.profile.ctx_len(), 2048);
    assert_eq!(fresh.provenance, TuneProvenance::Heuristic);
}

#[test]
fn record_perf_rolls_running_average() {
    let home = tempfile::tempdir().unwrap();
    let s = JsonModelStore::open(home.path()).unwrap();
    s.record_perf("org/m", perf_sample(30.0), 1);
    s.record_perf("org/m", perf_sample(50.0), 2);
    let p = s.perf("org/m").unwrap();
    assert_eq!(p.samples, 2);
    assert!(
        (p.avg_gen_tps - 40.0).abs() < 0.01,
        "avg of 30 and 50 is 40"
    );
    assert_eq!(p.last_gen_tps, 50.0, "last is the most recent");
}

#[test]
fn missing_file_is_empty_store() {
    let home = tempfile::tempdir().unwrap();
    let s = JsonModelStore::open(home.path()).unwrap();
    assert!(s.tuning("anything").is_none());
    assert!(s.perf("anything").is_none());
}
