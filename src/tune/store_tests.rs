use super::*;
use crate::worker::engine::{CtxLen, GpuLayers};

fn tune_record() -> TuneRecord {
    TuneRecord {
        profile: LoadParams::base(CtxLen::Fixed { n: 8192 }, GpuLayers::All, 8),
        sampling: SamplingParams::default(),
        budget: ResourceBudget::default(),
        provenance: TuneProvenance::Heuristic,
        bench_tps: None,
        tuned_at_ms: 100,
        hw_fingerprint: "v0r0n0".into(),
        model_file_sig: "123:456".into(),
    }
}

#[test]
fn legacy_profile_without_anchors_is_grandfathered_not_stale() {
    // A pre-existing models.json profile (from before staleness tracking) loads
    // with empty anchors via serde defaults; it must NOT be marked stale, or
    // every upgraded user's saved profiles would flip to NeedsRetune.
    let mut rec = tune_record();
    rec.hw_fingerprint = String::new();
    rec.model_file_sig = String::new();
    assert!(
        !rec.is_stale("v1r1n1", "999:999"),
        "empty anchors are grandfathered → never stale"
    );
}

#[test]
fn present_anchor_mismatch_marks_stale() {
    let rec = tune_record(); // anchors: hw "v0r0n0", file "123:456"
    assert!(
        !rec.is_stale("v0r0n0", "123:456"),
        "matching anchors → fresh"
    );
    assert!(
        rec.is_stale("vXrXnX", "123:456"),
        "hardware changed → stale"
    );
    assert!(
        rec.is_stale("v0r0n0", "999:999"),
        "model file changed → stale"
    );
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
fn set_profile_clears_bench_only_when_params_change() {
    let home = tempfile::tempdir().unwrap();
    let s = JsonModelStore::open(home.path()).unwrap();
    let p1 = LoadParams::base(CtxLen::Fixed { n: 1234 }, GpuLayers::Count { n: 0 }, 2);
    let mut rec = tune_record();
    rec.profile = p1.clone();
    rec.provenance = TuneProvenance::Bench;
    rec.bench_tps = Some(9.0);
    s.put_tuning("org/m", rec);

    // Re-anchoring the SAME params keeps the measured metrics (the benchmarked config is
    // just re-validated on current hardware) — only the anchors/timestamp move.
    s.set_profile("org/m", p1.clone(), "vNew", "sigNew", 99);
    let got = s.tuning("org/m").unwrap();
    assert_eq!(got.tuned_at_ms, 99, "timestamp refreshed");
    assert_eq!(got.hw_fingerprint, "vNew", "anchor refreshed");
    assert_eq!(
        got.provenance,
        TuneProvenance::Bench,
        "same params → provenance preserved"
    );
    assert_eq!(
        got.bench_tps,
        Some(9.0),
        "same params → bench_tps preserved"
    );

    // A CHANGED profile (an OOM-degraded fallback OR an edited explicit reload) DROPS the
    // stale metrics — they described the OLD config (codex r11/r12).
    let p2 = LoadParams::base(CtxLen::Fixed { n: 2048 }, GpuLayers::All, 8);
    s.set_profile("org/m", p2, "vNew", "sigNew", 100);
    let got = s.tuning("org/m").unwrap();
    assert_eq!(
        got.profile.ctx_len(),
        CtxLen::Fixed { n: 2048 },
        "profile updated"
    );
    assert_eq!(
        got.provenance,
        TuneProvenance::Heuristic,
        "changed params → provenance cleared"
    );
    assert_eq!(got.bench_tps, None, "changed params → bench_tps cleared");

    // With NO prior record, set_profile creates one with defaults.
    s.set_profile(
        "org/new",
        LoadParams::base(CtxLen::Fixed { n: 2048 }, GpuLayers::All, 8),
        "vNew",
        "sigNew",
        50,
    );
    let fresh = s.tuning("org/new").unwrap();
    assert_eq!(fresh.profile.ctx_len(), CtxLen::Fixed { n: 2048 });
    assert_eq!(fresh.provenance, TuneProvenance::Heuristic);
}

#[test]
fn put_tuning_never_fabricates_an_analytical_slot_from_a_bare_load_demotion() {
    // A `Heuristic` active record is ambiguous: it is either a real analytical
    // tune OR a bare-load reload `set_profile` demoted (identical field-for-field).
    // `put_tuning` must NEVER backfill that active record into the analytical slot,
    // or the bare-load params would surface as the analytical "Tuned" set — the
    // exact masquerade `control.rs::from_triple`'s both-slots-empty grandfather
    // gate exists to prevent (codex r9 P8). Sequence: Benchmark → bare-load edit →
    // Benchmark. The analytical slot must stay empty throughout.
    let home = tempfile::tempdir().unwrap();
    let s = JsonModelStore::open(home.path()).unwrap();

    // 1) Benchmark: fills the bench slot only.
    let mut benched = tune_record();
    benched.profile = LoadParams::base(CtxLen::Fixed { n: 2222 }, GpuLayers::All, 8);
    benched.provenance = TuneProvenance::Bench;
    benched.bench_tps = Some(33.0);
    s.put_tuning("org/m", benched);

    // 2) Bare-load with EDITED params: set_profile demotes the active record to
    //    Heuristic and touches no history slot.
    s.set_profile(
        "org/m",
        LoadParams::base(CtxLen::Fixed { n: 3333 }, GpuLayers::All, 4),
        "vEdit",
        "333:333",
        20,
    );
    let (active, _, _) = s.tuning_profiles("org/m");
    assert_eq!(
        active.as_ref().unwrap().provenance,
        TuneProvenance::Heuristic
    );
    assert_eq!(
        active.unwrap().profile.ctx_len(),
        CtxLen::Fixed { n: 3333 },
        "active now carries the bare-load edited params",
    );

    // 3) Benchmark again.
    let mut benched2 = tune_record();
    benched2.profile = LoadParams::base(CtxLen::Fixed { n: 4444 }, GpuLayers::All, 8);
    benched2.provenance = TuneProvenance::Bench;
    benched2.bench_tps = Some(44.0);
    s.put_tuning("org/m", benched2);

    let (active, analytical, bench) = s.tuning_profiles("org/m");
    // Fail-on-revert: re-adding the backfill migration would populate the
    // analytical slot with the ctx-3333 bare-load params.
    assert!(
        analytical.is_none(),
        "analytical slot must stay empty — a bare-load demotion is NOT a Tuned set",
    );
    assert_eq!(
        bench.unwrap().profile.ctx_len(),
        CtxLen::Fixed { n: 4444 },
        "bench slot holds the latest benchmark",
    );
    assert_eq!(active.unwrap().provenance, TuneProvenance::Bench);
}

#[test]
fn set_profile_refreshes_anchors_to_current_preserving_future_detection() {
    let home = tempfile::tempdir().unwrap();
    let s = JsonModelStore::open(home.path()).unwrap();
    // A Prepared profile carrying anchors for the ORIGINAL hardware/file.
    s.put_tuning("org/m", tune_record()); // hw "v0r0n0", file "123:456"
                                          // An accepted load anchors the profile to the CURRENT hardware/file.
    s.set_profile(
        "org/m",
        LoadParams::base(CtxLen::Fixed { n: 1024 }, GpuLayers::All, 4),
        "vCURrCUR",
        "999:999",
        123,
    );
    let rec = s.tuning("org/m").unwrap();
    // Anchors are REFRESHED to current (not cleared) — so the accepted load reads
    // back fresh, AND a later hardware/file change is still detected as stale.
    assert_eq!(
        rec.hw_fingerprint, "vCURrCUR",
        "hw anchor refreshed to current"
    );
    assert_eq!(
        rec.model_file_sig, "999:999",
        "file anchor refreshed to current"
    );
    assert!(
        !rec.is_stale("vCURrCUR", "999:999"),
        "matches current → fresh"
    );
    assert!(
        rec.is_stale("vLATER", "999:999"),
        "a LATER hardware change is still detected (detection preserved)"
    );
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

#[test]
fn corrupt_file_starts_empty_not_error() {
    // A garbage models.json is logged and treated as the empty store rather than
    // failing open() — the GGUFs stay the source of truth, tuning/perf re-earned.
    let home = tempfile::tempdir().unwrap();
    std::fs::write(home.path().join("models.json"), b"{ this is not json ]")
        .expect("seed corrupt file");
    let s = JsonModelStore::open(home.path()).expect("corrupt file is not an open() error");
    assert!(s.tuning("org/m").is_none());
    assert!(s.perf("org/m").is_none());
    assert!(s.entry("org/m").is_none());
    // The recovered store is fully usable: writes + flush + reopen round-trip.
    s.put_tuning("org/m", tune_record());
    s.flush().unwrap();
    let s2 = JsonModelStore::open(home.path()).unwrap();
    assert!(s2.tuning("org/m").is_some());
}

#[test]
fn entry_snapshots_whole_record() {
    let home = tempfile::tempdir().unwrap();
    let s = JsonModelStore::open(home.path()).unwrap();
    // No record yet → None.
    assert!(s.entry("org/m").is_none());
    // Populate all three parts; entry() returns the coherent snapshot.
    s.put_tuning("org/m", tune_record());
    s.record_perf("org/m", perf_sample(12.0), 7);
    s.put_meta("org/m", meta_cache("/m.gguf", 200, 9));
    let e = s.entry("org/m").expect("entry present");
    assert!(e.tuning.is_some(), "tuning part present");
    assert!(e.perf.is_some(), "perf part present");
    let cache = e.meta.expect("meta part present");
    assert_eq!(cache.key.path, "/m.gguf");
    assert_eq!(cache.key.size_bytes, 200);
    assert_eq!(cache.key.mtime_ms, 9);
}

#[test]
fn profile_store_trait_delegates_to_inherent() {
    // The ProfileStore impl forwards to the inherent tuning/put_tuning so the
    // suggester can drive the store through the trait object seam.
    let home = tempfile::tempdir().unwrap();
    let s = JsonModelStore::open(home.path()).unwrap();
    let store: &dyn ProfileStore = &s;
    assert!(store.tuning("org/m").is_none(), "empty via trait");
    store.put_tuning("org/m", tune_record());
    let got = store.tuning("org/m").expect("written via trait");
    assert_eq!(got.profile.ctx_len(), CtxLen::Fixed { n: 8192 });
    assert_eq!(got.provenance, TuneProvenance::Heuristic);
    // And the inherent reader sees the trait-written record (same backing store).
    assert!(JsonModelStore::tuning(&s, "org/m").is_some());
}

#[test]
fn open_home_uses_higgs_home_override() {
    // open_home() resolves the per-node home via $HIGGS_HOME (the on-disk seam the
    // default store uses); point it at a TempDir so the test never touches ~/.higgs.
    // Serialize with other env-mutating tests and restore the prior value (cargo runs
    // lib tests in parallel threads of one process).
    let _env = crate::TEST_ENV_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let home = tempfile::tempdir().unwrap();
    // Seed a record through the explicit-path opener, flush, then reopen via the
    // home-resolving opener and confirm it reads the same file.
    let seed = JsonModelStore::open(home.path()).unwrap();
    seed.put_tuning("org/m", tune_record());
    seed.flush().unwrap();

    let prev = std::env::var_os("HIGGS_HOME");
    // SAFETY: serialized by TEST_ENV_LOCK; restored below.
    unsafe { std::env::set_var("HIGGS_HOME", home.path()) };
    let opened = JsonModelStore::open_home();
    // SAFETY: still under the lock.
    unsafe {
        match prev {
            Some(v) => std::env::set_var("HIGGS_HOME", v),
            None => std::env::remove_var("HIGGS_HOME"),
        }
    }
    let opened = opened.expect("open_home under HIGGS_HOME");
    assert!(
        opened.tuning("org/m").is_some(),
        "open_home read the seeded models.json under $HIGGS_HOME"
    );
}
