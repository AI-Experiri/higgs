// Unit tests for the surviving control HELPERS — the pure sub-primitives the
// `Higgs` facade delegates to (`api/embed.rs`). The `/api/higgs/*` HTTP handlers
// these once accompanied were deleted with the standalone control surface; their
// end-to-end behavior is now covered at the facade level (`api/embed_tests.rs`,
// `api/tests.rs`).

// ── Gate 2: host-side tool-call-parser sniff ─────────────────────────────

/// Build a minimal scanned model carrying only the chat template that the
/// Gate-2 sniff inspects; all other fields are placeholder.
fn model_with_template(template: Option<&str>) -> crate::worker::models::HiggsModel {
    crate::worker::models::HiggsModel {
        id: "org/model".into(),
        path: "/x.gguf".into(),
        size_bytes: 0,
        quant: None,
        source: crate::worker::models::HiggsModelSource::LmStudio,
        arch: None,
        ctx_train: None,
        block_count: None,
        head_count: None,
        head_count_kv: None,
        embedding_length: None,
        expert_count: None,
        has_chat_template: template.is_some(),
        domain: crate::worker::models::ModelDomain::Llm,
        supports_tools: false,
        supports_reasoning: false,
        gguf_components: Vec::new(),
        enrich_error: None,
        chat_template: template.map(ToOwned::to_owned),
    }
}

#[test]
fn gate2_sniffs_tool_call_template() {
    // A template with the generic `<tool_call>` marker → a parser matches.
    let with_calls = model_with_template(Some(
        "{% for m in messages %}<|im_start|>{{ m.role }}<tool_call>{{ tool }}</tool_call>",
    ));
    assert!(
        super::tool_calls_supported(&with_calls),
        "<tool_call> matches"
    );

    // A plain chatml template with no tool markup → no parser matches.
    let plain = model_with_template(Some(
        "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>",
    ));
    assert!(
        !super::tool_calls_supported(&plain),
        "plain chatml: no match"
    );

    // No template at all → false.
    assert!(!super::tool_calls_supported(&model_with_template(None)));
}

// ── model_entry carries the tune provenance + measured tok/s ─────────────
/// `put_tuning` routes each result into its provenance slot while keeping the
/// ACTIVE record, and `TuneProfileViews::from_triple` grandfathers a
/// pre-dual-slot store (lone active record serves its own provenance's set).
/// Fail-on-revert: dropping the history-slot routing in `put_tuning` loses the
/// "Tuned" set the moment a turbotune becomes active.
#[test]
fn dual_profiles_offer_both_tuned_and_benchmarked_sets() {
    use crate::tune::store::{JsonModelStore, TuneRecord};
    use crate::worker::engine::CtxLen;
    let dir = tempfile::tempdir().unwrap();
    let store = JsonModelStore::open(dir.path()).unwrap();
    let rec = |ctx: u32, prov: crate::tune::TuneProvenance| TuneRecord {
        profile: crate::worker::engine::LoadParams::llamacpp(
            crate::worker::engine::llamacpp::params::LlamaCppParams {
                ctx_len: CtxLen::Fixed { n: ctx },
                ..Default::default()
            },
        ),
        sampling: Default::default(),
        budget: Default::default(),
        provenance: prov,
        bench_tps: (prov == crate::tune::TuneProvenance::Bench).then_some(33.0),
        tuned_at_ms: 1,
        hw_fingerprint: String::new(),
        model_file_sig: String::new(),
    };
    // Analytical first, then a turbotune becomes ACTIVE — both sets must remain.
    store.put_tuning("m", rec(1111, crate::tune::TuneProvenance::Heuristic));
    store.put_tuning("m", rec(2222, crate::tune::TuneProvenance::Bench));
    let (active, analytical, bench) = store.tuning_profiles("m");
    assert_eq!(
        active.as_ref().map(|r| r.provenance),
        Some(crate::tune::TuneProvenance::Bench),
        "the last tune is the active profile"
    );
    assert_eq!(
        analytical.map(|r| r.profile.as_llamacpp().ctx_len),
        Some(CtxLen::Fixed { n: 1111 }),
        "the analytical set survives a later turbotune"
    );
    assert_eq!(
        bench.map(|r| r.profile.as_llamacpp().ctx_len),
        Some(CtxLen::Fixed { n: 2222 })
    );

    // Pre-dual-store fallback: a lone ACTIVE analytical record serves the
    // Tuned set (and not the Benchmarked one).
    let lone = (
        Some(rec(3333, crate::tune::TuneProvenance::Heuristic)),
        None,
        None,
    );
    let views = super::TuneProfileViews::from_triple(Some(&lone));
    assert!(views.analytical.is_some() && views.bench.is_none());
}

/// Regression (dual-profile finding): after a turbotune, a later BARE LOAD with
/// edited params demotes the ACTIVE record to a bare-load `Heuristic` via
/// `set_profile` — but that manual record was never an analytical tune. The
/// `tuned_load` ("last ANALYTICAL tune") set must stay ABSENT, not be fabricated
/// from the demoted active record; grandfathering runs only for a true pre-dual
/// store (BOTH history slots empty).
/// Fail-on-revert: restoring the unconditional `analytical.or(Some(active))`
/// grandfather surfaces the manual bare-load params as the Tuned set.
#[test]
fn bare_load_after_turbotune_does_not_fabricate_a_tuned_set() {
    use crate::tune::store::{JsonModelStore, TuneRecord};
    use crate::worker::engine::llamacpp::params::LlamaCppParams;
    use crate::worker::engine::{CtxLen, LoadParams};
    let dir = tempfile::tempdir().unwrap();
    let store = JsonModelStore::open(dir.path()).unwrap();
    // 1) Turbotune wins → active=Bench, bench slot filled, analytical slot empty.
    store.put_tuning(
        "m",
        TuneRecord {
            profile: LoadParams::llamacpp(LlamaCppParams {
                ctx_len: CtxLen::Fixed { n: 2222 },
                ..Default::default()
            }),
            sampling: Default::default(),
            budget: Default::default(),
            provenance: crate::tune::TuneProvenance::Bench,
            bench_tps: Some(41.0),
            tuned_at_ms: 1,
            hw_fingerprint: String::new(),
            model_file_sig: String::new(),
        },
    );
    // 2) A manual reload with DIFFERENT params → set_profile demotes the active
    //    record to a bare-load Heuristic (provenance + bench_tps dropped); the
    //    bench history slot is untouched.
    store.set_profile(
        "m",
        LoadParams::llamacpp(LlamaCppParams {
            ctx_len: CtxLen::Fixed { n: 9999 },
            ..Default::default()
        }),
        "",
        "",
        2,
    );

    let triple = store.tuning_profiles("m");
    let views = super::TuneProfileViews::from_triple(Some(&triple));
    // The Benchmarked set survives; the Tuned set is ABSENT (no analytical tune
    // ever ran — the demoted active record must NOT masquerade as one).
    assert!(
        views.bench.is_some(),
        "the benchmarked set survives the bare load"
    );
    assert!(
        views.analytical.is_none(),
        "a bare-load-demoted active record must not be served as the Tuned set"
    );
    // Guard the scenario's premise: the active record really did become a
    // bare-load Heuristic carrying the manual params.
    assert_eq!(
        views.active.map(|r| r.provenance),
        Some(crate::tune::TuneProvenance::Heuristic)
    );
    assert_eq!(
        views.active.map(|r| r.profile.as_llamacpp().ctx_len),
        Some(CtxLen::Fixed { n: 9999 })
    );
}

/// The models list derives the ACTIVE-record map (readiness input) from the SAME
/// `tuning_profiles()` triples that fill the wire's `tuned_load`/`benched_load` —
/// one `models.json` snapshot, so readiness can't disagree with the profile
/// fields under a concurrent tune. `active_records` must extract exactly each
/// triple's active (`.0`) record and skip entries whose active slot is empty.
/// Fail-on-revert: extracting a history slot (or not skipping a None active)
/// makes readiness read a record the wire fields never expose.
#[test]
fn active_records_extracts_only_the_active_slot_per_model() {
    use crate::tune::store::TuneRecord;
    use crate::worker::engine::llamacpp::params::LlamaCppParams;
    use crate::worker::engine::{CtxLen, LoadParams};
    let rec = |ctx: u32, prov: crate::tune::TuneProvenance| TuneRecord {
        profile: LoadParams::llamacpp(LlamaCppParams {
            ctx_len: CtxLen::Fixed { n: ctx },
            ..Default::default()
        }),
        sampling: Default::default(),
        budget: Default::default(),
        provenance: prov,
        bench_tps: None,
        tuned_at_ms: 1,
        hw_fingerprint: String::new(),
        model_file_sig: String::new(),
    };
    let mut profiles = std::collections::BTreeMap::new();
    // "a": active Bench + a bench history slot → active_records takes the ACTIVE.
    profiles.insert(
        "a".to_string(),
        (
            Some(rec(2222, crate::tune::TuneProvenance::Bench)),
            None,
            Some(rec(2222, crate::tune::TuneProvenance::Bench)),
        ),
    );
    // "b": NO active record (only a history slot) → must be skipped entirely.
    profiles.insert(
        "b".to_string(),
        (
            None,
            Some(rec(1111, crate::tune::TuneProvenance::Heuristic)),
            None,
        ),
    );
    // "c": lone active Heuristic → taken.
    profiles.insert(
        "c".to_string(),
        (
            Some(rec(3333, crate::tune::TuneProvenance::Heuristic)),
            None,
            None,
        ),
    );

    let active = super::active_records(&profiles);
    assert_eq!(active.len(), 2, "only models with an active record appear");
    assert_eq!(
        active.get("a").map(|r| r.profile.as_llamacpp().ctx_len),
        Some(CtxLen::Fixed { n: 2222 }),
        "the ACTIVE record is taken, not a history slot"
    );
    assert!(!active.contains_key("b"), "a None active slot is skipped");
    assert_eq!(
        active.get("c").map(|r| r.profile.as_llamacpp().ctx_len),
        Some(CtxLen::Fixed { n: 3333 })
    );
}

/// The models list is the frontend's ONE source for the Tuned/Benchmarked
/// badge: `model_entry` must copy `provenance` + `bench_tps` from the tune
/// record, and leave both absent when the model has no record.
#[test]
fn model_entry_carries_tune_provenance_and_bench_tps() {
    let model = model_with_template(None);
    let rec = crate::tune::store::TuneRecord {
        profile: Default::default(),
        sampling: Default::default(),
        budget: Default::default(),
        provenance: crate::tune::TuneProvenance::Bench,
        bench_tps: Some(42.5),
        tuned_at_ms: 1,
        hw_fingerprint: String::new(),
        model_file_sig: String::new(),
    };
    // Active = the bench record; no separate analytical slot (pre-dual store) —
    // the views borrow the active record for the bench side.
    let triple = (Some(rec.clone()), None, Some(rec.clone()));
    let entry = super::model_entry(
        model.clone(),
        &[],
        None,
        crate::serve::readiness::ModelReadiness::Discovered,
        None,
        super::TuneProfileViews::from_triple(Some(&triple)),
    );
    assert_eq!(
        entry.tune_provenance,
        Some(crate::tune::TuneProvenance::Bench)
    );
    assert_eq!(entry.bench_tps, Some(42.5));
    assert_eq!(
        entry.benched_load.as_ref().map(|p| p.as_llamacpp().clone()),
        Some(rec.profile.as_llamacpp().clone()),
        "a Bench record serves the Benchmarked set (both panes seed from the wire)"
    );
    assert!(
        entry.tuned_load.is_none(),
        "no analytical record → no Tuned set"
    );

    // No record → all tune fields absent (and absent from the JSON wire).
    let entry = super::model_entry(
        model,
        &[],
        None,
        crate::serve::readiness::ModelReadiness::Discovered,
        None,
        super::TuneProfileViews::from_triple(None),
    );
    assert_eq!(entry.tune_provenance, None);
    assert_eq!(entry.bench_tps, None);
    assert!(entry.tuned_load.is_none());
    assert!(entry.benched_load.is_none());
    let json = serde_json::to_value(&entry).unwrap();
    assert!(json.get("tune_provenance").is_none());
    assert!(json.get("bench_tps").is_none());
}

/// `tuned_max_tokens` is the first CONCRETE tuned window — bench slot preferred,
/// Auto falls through, min'd with the ACTIVE record's fixed window (what a JIT
/// load actually pins — Fable mt-r1), capped at `MAX_OUTPUT_TOKENS`, and GATED
/// on the history slots only (a post-dual demoted Heuristic active never grants).
/// Fail-on-revert, per branch: hardcoding `None` fails the bench case; preferring
/// the analytical slot fails the bench-wins assert (1111 ≠ 2222); dropping the
/// Auto fall-through fails the auto-bench case; GATING on `active` instead of
/// the slots fails the demoted-Heuristic-no-slots case; dropping the active-min
/// fails the demoted-smaller-window case (8192 ≠ 2048); dropping the cap fails
/// the over-cap case.
#[test]
fn tuned_max_tokens_is_the_first_concrete_tuned_window() {
    use crate::worker::engine::CtxLen;
    let model = model_with_template(None);
    let rec = |ctx: CtxLen, prov: crate::tune::TuneProvenance| crate::tune::store::TuneRecord {
        profile: crate::worker::engine::LoadParams::llamacpp(
            crate::worker::engine::llamacpp::params::LlamaCppParams {
                ctx_len: ctx,
                ..Default::default()
            },
        ),
        sampling: Default::default(),
        budget: Default::default(),
        provenance: prov,
        bench_tps: None,
        tuned_at_ms: 1,
        hw_fingerprint: String::new(),
        model_file_sig: String::new(),
    };
    let entry_for = |triple: Option<&(_, _, _)>| {
        super::model_entry(
            model.clone(),
            &[],
            None,
            crate::serve::readiness::ModelReadiness::Discovered,
            None,
            super::TuneProfileViews::from_triple(triple),
        )
    };
    let heur = crate::tune::TuneProvenance::Heuristic;
    let bench = crate::tune::TuneProvenance::Bench;

    // Bench (measured) wins over analytical.
    let both = (
        Some(rec(CtxLen::Fixed { n: 2222 }, bench)),
        Some(rec(CtxLen::Fixed { n: 1111 }, heur)),
        Some(rec(CtxLen::Fixed { n: 2222 }, bench)),
    );
    assert_eq!(entry_for(Some(&both)).tuned_max_tokens, Some(2222));

    // Analytical only.
    let analytical_only = (
        Some(rec(CtxLen::Fixed { n: 1111 }, heur)),
        Some(rec(CtxLen::Fixed { n: 1111 }, heur)),
        None,
    );
    assert_eq!(
        entry_for(Some(&analytical_only)).tuned_max_tokens,
        Some(1111)
    );

    // An Auto-pinned bench window falls through to the analytical one rather
    // than hiding it.
    let auto_bench = (
        Some(rec(CtxLen::Auto, bench)),
        Some(rec(CtxLen::Fixed { n: 1111 }, heur)),
        Some(rec(CtxLen::Auto, bench)),
    );
    assert_eq!(entry_for(Some(&auto_bench)).tuned_max_tokens, Some(1111));

    // A bare load demoted the active record to a BIGGER fixed window than the
    // tune: the value stays the TUNED slot's (the min is a min, not
    // active-wins — "per its TUNED metrics" caps at the tune even when the
    // serving window is roomier; Fable mt-r4 killed the `active-fixed-wins`
    // mutant with this case).
    let demoted_bigger = (
        Some(rec(CtxLen::Fixed { n: 16_384 }, heur)),
        None,
        Some(rec(CtxLen::Fixed { n: 8192 }, bench)),
    );
    assert_eq!(
        entry_for(Some(&demoted_bigger)).tuned_max_tokens,
        Some(8192)
    );

    // Post-dual store, bare load demoted the active record to a Heuristic and
    // NO history slot holds a real tune → not tuned (the active is never read).
    let demoted_rec = rec(CtxLen::Fixed { n: 9999 }, heur);
    let demoted = super::TuneProfileViews {
        active: Some(&demoted_rec),
        analytical: None,
        bench: None,
    };
    let entry = super::model_entry(
        model.clone(),
        &[],
        None,
        crate::serve::readiness::ModelReadiness::Discovered,
        None,
        demoted,
    );
    assert_eq!(entry.tuned_max_tokens, None);

    // A bare load DEMOTED the active record to a SMALLER fixed window than the
    // bench tune: the value follows the active window (what a JIT load actually
    // pins — advertising the bench 8192 would promise output a 2048-window
    // serving can't hold), while the SLOTS still gate (the model stays tuned).
    let demoted_smaller = (
        Some(rec(CtxLen::Fixed { n: 2048 }, heur)),
        Some(rec(CtxLen::Fixed { n: 1111 }, heur)),
        Some(rec(CtxLen::Fixed { n: 8192 }, bench)),
    );
    assert_eq!(
        entry_for(Some(&demoted_smaller)).tuned_max_tokens,
        Some(2048)
    );

    // Untuned → None, and absent from the JSON wire.
    let entry = entry_for(None);
    assert_eq!(entry.tuned_max_tokens, None);
    let json = serde_json::to_value(&entry).unwrap();
    assert!(json.get("tuned_max_tokens").is_none());

    // A tuned window above the absolute output cap is capped — `/v1` rejects a
    // request asking for more ([HG013]) and the in-process path clamps to it,
    // so advertising more would promise output no request can deliver.
    let huge = (
        Some(rec(CtxLen::Fixed { n: 40_000 }, heur)),
        Some(rec(CtxLen::Fixed { n: 40_000 }, heur)),
        None,
    );
    assert_eq!(
        entry_for(Some(&huge)).tuned_max_tokens,
        Some(crate::serve::MAX_OUTPUT_TOKENS)
    );
}

/// G4 bootstrap-race defense at the SEAM ([codex r9]): `decide_mint` derives
/// `bootstrap` from the LOCKED store it is handed, so the empty-store window
/// cannot be raced — a second unauthenticated mint that reaches the lock after
/// the first key landed sees a NON-empty store and is refused. This is the
/// deterministic core the HTTP handler runs inside `keys_io`. Fail-on-revert:
/// deciding `bootstrap` from a pre-lock snapshot (the bug) makes the
/// "non-empty store + no bearer" case mint instead of refuse.
#[test]
fn decide_mint_derives_bootstrap_from_the_locked_store() {
    use crate::keys::{ApiKeys, Scope};

    // Empty store: the bootstrap mint is allowed UNAUTHENTICATED and defaults
    // to Admin (so it isn't locked out of key management it just enabled).
    let empty = ApiKeys::default();
    match super::decide_mint(&empty, false, None, None, "first") {
        super::Mint::Ok(scopes) => {
            assert_eq!(scopes, vec![Scope::Admin], "bootstrap defaults admin")
        }
        _ => panic!("empty store must allow the unauthenticated bootstrap mint"),
    }
    // Explicit ADMIN-inclusive bootstrap scopes are honored.
    match super::decide_mint(
        &empty,
        false,
        None,
        Some(vec![Scope::Admin, Scope::Chat]),
        "first",
    ) {
        super::Mint::Ok(scopes) => assert_eq!(scopes, vec![Scope::Admin, Scope::Chat]),
        _ => panic!("explicit admin-inclusive bootstrap scopes honored"),
    }
    // Explicit NON-admin bootstrap scopes are REJECTED (codex r10): a chat-only
    // first key would flip auth on and lock the Admin-scoped key-management API
    // out of itself. Fail-on-revert: honoring them makes this Mint::Ok.
    assert!(
        matches!(
            super::decide_mint(&empty, false, None, Some(vec![Scope::Chat]), "first"),
            super::Mint::BootstrapNeedsAdmin
        ),
        "a non-admin first key is refused"
    );

    // Non-empty store: the race loser (no bearer) is REFUSED — this is the
    // decision that a pre-lock bootstrap snapshot would get wrong.
    let mut seeded = ApiKeys::default();
    let admin_tok = crate::keys::mint_token([3u8; 16]);
    seeded.add(&admin_tok, "boss".into(), vec![Scope::Admin]);
    assert!(
        matches!(
            super::decide_mint(&seeded, false, None, None, "second"),
            super::Mint::Unauthorized
        ),
        "non-empty store + no bearer must be refused"
    );
    // A non-admin bearer is also refused for a management mint.
    let mut chat_only = ApiKeys::default();
    let chat_tok = crate::keys::mint_token([4u8; 16]);
    chat_only.add(&chat_tok, "reader".into(), vec![Scope::Chat]);
    assert!(
        matches!(
            super::decide_mint(&chat_only, false, Some(&chat_tok), None, "x"),
            super::Mint::Unauthorized
        ),
        "a non-admin bearer cannot mint"
    );
    // A valid Admin bearer mints, defaulting to [chat, models] (non-bootstrap).
    match super::decide_mint(&seeded, false, Some(&admin_tok), None, "svc") {
        super::Mint::Ok(scopes) => assert_eq!(scopes, vec![Scope::Chat, Scope::Models]),
        _ => panic!("admin bearer mints with the non-bootstrap default"),
    }
    // Duplicate label is caught regardless.
    assert!(
        matches!(
            super::decide_mint(&seeded, false, Some(&admin_tok), None, "boss"),
            super::Mint::Duplicate
        ),
        "duplicate label rejected"
    );
}

/// G4/G5 revoke bootstrap-race defense at the SEAM ([codex r7]): `decide_revoke`
/// derives authorization from the LOCKED store, so a DELETE admitted while the
/// store was empty (auth off) is refused if a bootstrap mint made the store
/// non-empty before the lock — no unauthenticated deletion of a freshly minted
/// key. Fail-on-revert: dropping the locked-store auth check makes the
/// "non-empty store + no bearer" case Remove instead of Unauthorized.
#[test]
fn decide_revoke_rechecks_auth_against_the_locked_store() {
    use crate::keys::{ApiKeys, Scope};

    // Empty store: nothing to remove, but not an auth error (auth is off).
    let empty = ApiKeys::default();
    assert!(
        matches!(
            super::decide_revoke(&empty, false, None, "x", false),
            super::Revoke::Removed(0)
        ),
        "empty store: 0 removed, not an auth failure"
    );

    // Non-empty store, NO bearer → refused (the race loser).
    let mut seeded = ApiKeys::default();
    let admin = crate::keys::mint_token([5u8; 16]);
    seeded.add(&admin, "boss".into(), vec![Scope::Admin]);
    assert!(
        matches!(
            super::decide_revoke(&seeded, false, None, "boss", false),
            super::Revoke::Unauthorized
        ),
        "non-empty store + no bearer must be refused"
    );
    // Non-admin bearer also refused.
    let mut chat = ApiKeys::default();
    let ct = crate::keys::mint_token([6u8; 16]);
    chat.add(&ct, "reader".into(), vec![Scope::Chat]);
    assert!(
        matches!(
            super::decide_revoke(&chat, false, Some(&ct), "reader", false),
            super::Revoke::Unauthorized
        ),
        "non-admin bearer cannot revoke"
    );
    // Valid admin bearer → removes (1 match).
    assert!(
        matches!(
            super::decide_revoke(&seeded, false, Some(&admin), "boss", false),
            super::Revoke::Removed(1)
        ),
        "admin bearer revokes"
    );
    // Last key on a LAN-exposed server → refused ([HG059]) even WITH the bearer.
    assert!(
        matches!(
            super::decide_revoke(&seeded, false, Some(&admin), "boss", true),
            super::Revoke::LastKeyOnLan
        ),
        "last-key revoke refused while LAN-exposed"
    );
}

/// Revoking the LAST Admin-capable key while other (non-Admin) keys remain is
/// refused ([HG066]): it would leave auth ON but the Admin-only management surface
/// unreachable — an operator lockout. Emptying the whole store (auth OFF) is still
/// allowed. Fail-on-revert: drop the admin-remains check in `decide_revoke` and the
/// first assertion returns `Removed(1)`, stranding the operator.
#[test]
fn decide_revoke_preserves_the_last_admin_key() {
    use crate::keys::{ApiKeys, Scope};
    let mut ks = ApiKeys::default();
    let admin = crate::keys::mint_token([7u8; 16]);
    let reader = crate::keys::mint_token([8u8; 16]);
    ks.add(&admin, "boss".into(), vec![Scope::Admin]);
    ks.add(&reader, "reader".into(), vec![Scope::Chat, Scope::Models]);

    assert!(
        matches!(
            super::decide_revoke(&ks, false, Some(&admin), "boss", false),
            super::Revoke::LastAdminKey
        ),
        "revoking the last Admin key while a non-Admin key remains must be refused"
    );
    assert!(
        matches!(
            super::decide_revoke(&ks, false, Some(&admin), "reader", false),
            super::Revoke::Removed(1)
        ),
        "revoking a non-Admin key leaves Admin access intact"
    );

    // An admin-only store: revoking it empties the store (auth OFF) — allowed.
    let mut solo = ApiKeys::default();
    let only = crate::keys::mint_token([9u8; 16]);
    solo.add(&only, "boss".into(), vec![Scope::Admin]);
    assert!(
        matches!(
            super::decide_revoke(&solo, false, Some(&only), "boss", false),
            super::Revoke::Removed(1)
        ),
        "revoking the entire store (turns auth off) is allowed, not a lockout"
    );
}

/// A HIDDEN internal admin key (the embedder's in-memory token) must NOT satisfy
/// the last-visible-admin guard: revoking the last VISIBLE admin WHILE OTHER
/// VISIBLE keys remain would strand the persisted management surface (only the
/// unlistable hidden admin would be left). Fail-on-revert: change the
/// `admin_remains` scan in `decide_revoke` back to `iter()` (counts the hidden
/// admin) and this returns `Removed(1)` instead of `LastAdminKey`.
#[test]
fn decide_revoke_ignores_a_hidden_internal_admin() {
    use crate::keys::{ApiKeys, Scope};
    let mut ks = ApiKeys::default();
    let admin = crate::keys::mint_token([11u8; 16]);
    let reader = crate::keys::mint_token([12u8; 16]);
    ks.add(&admin, "boss".into(), vec![Scope::Admin]);
    ks.add(&reader, "reader".into(), vec![Scope::Chat, Scope::Models]);
    // The embedder's hidden internal admin (jigglebot's proxy token): Admin-scoped,
    // but hidden — never written to disk, never listed to the key-management UI.
    ks.add_internal(
        "jigglebot-internal-token",
        "jigglebot (internal)".into(),
        vec![Scope::Chat, Scope::Models, Scope::Admin],
    );

    // Revoking the last VISIBLE admin while a visible reader remains is a lockout:
    // the hidden admin can't manage keys for an external/LAN operator.
    assert!(
        matches!(
            super::decide_revoke(&ks, false, Some(&admin), "boss", false),
            super::Revoke::LastAdminKey
        ),
        "a hidden internal admin must NOT count toward the last-visible-admin guard"
    );
    // Sanity: the hidden admin still AUTHORIZES the request (auth is on) — the
    // guard refuses on management grounds, not on a missing bearer.
    assert!(
        ks.authorizes("jigglebot-internal-token", Scope::Admin),
        "the hidden internal token authorizes Admin even though it can't be the last visible admin"
    );
}

/// Deleting the LAST VISIBLE key when it is NON-admin (a hidden admin is present)
/// must be ALLOWED on loopback — it empties the VISIBLE store (auth stays on via
/// the hidden token), it is not a `[HG066]` last-admin lockout. Fail-on-revert:
/// with `total`/`matching` computed over `iter()` (counting the hidden key),
/// `matching (1) < total (2)` enters the admin guard and wrongly returns
/// `LastAdminKey`.
#[test]
fn decide_revoke_allows_deleting_the_last_visible_nonadmin_key() {
    use crate::keys::{ApiKeys, Scope};
    let mut ks = ApiKeys::default();
    let reader = crate::keys::mint_token([13u8; 16]);
    ks.add(&reader, "reader".into(), vec![Scope::Chat, Scope::Models]);
    ks.add_internal(
        "jigglebot-internal-token",
        "jigglebot (internal)".into(),
        vec![Scope::Chat, Scope::Models, Scope::Admin],
    );

    // Bearer is the hidden admin (authorizes the DELETE); the only visible key is
    // a non-admin reader. Loopback (not LAN) → removing it is allowed.
    assert!(
        matches!(
            super::decide_revoke(
                &ks,
                false,
                Some("jigglebot-internal-token"),
                "reader",
                false
            ),
            super::Revoke::Removed(1)
        ),
        "deleting the last visible non-admin key (hidden admin present) must be allowed, not HG066"
    );
}

/// Naming the HIDDEN internal key's label in a revoke decision matches NOTHING
/// (visible-only counting) → `Removed(0)`, a no-op. Combined with
/// `ApiKeys::remove_label` refusing to drop hidden keys, the embedder's token is
/// immune to the public key-management surface. Fail-on-revert: revert the
/// visible() counting and `matching` becomes 1 → `Removed(1)`.
#[test]
fn decide_revoke_treats_the_hidden_label_as_absent() {
    use crate::keys::{ApiKeys, Scope};
    let mut ks = ApiKeys::default();
    let admin = crate::keys::mint_token([14u8; 16]);
    ks.add(&admin, "boss".into(), vec![Scope::Admin]);
    ks.add_internal(
        "jigglebot-internal-token",
        "jigglebot (internal)".into(),
        vec![Scope::Chat, Scope::Models, Scope::Admin],
    );

    assert!(
        matches!(
            super::decide_revoke(&ks, false, Some(&admin), "jigglebot (internal)", false),
            super::Revoke::Removed(0)
        ),
        "revoking the hidden internal label must be a no-op (Removed(0)), not touch it"
    );
}

/// A store of ONLY VISIBLE NON-admin keys (plus a hidden admin) has no visible
/// Admin to strand, so revoking one such key must be ALLOWED even though another
/// visible key remains. This state is reachable when an embedder mints scoped
/// (non-admin) keys under its hidden Admin. Fail-on-revert: drop the
/// `removing_an_admin` gate and `admin_remains` is false → this wrongly returns
/// `LastAdminKey`.
#[test]
fn decide_revoke_allows_revoking_a_nonadmin_when_no_visible_admin_exists() {
    use crate::keys::{ApiKeys, Scope};
    let mut ks = ApiKeys::default();
    let r1 = crate::keys::mint_token([15u8; 16]);
    let r2 = crate::keys::mint_token([16u8; 16]);
    ks.add(&r1, "reader1".into(), vec![Scope::Chat, Scope::Models]);
    ks.add(&r2, "reader2".into(), vec![Scope::Chat, Scope::Models]);
    ks.add_internal(
        "jigglebot-internal-token",
        "jigglebot (internal)".into(),
        vec![Scope::Admin],
    );

    assert!(
        matches!(
            super::decide_revoke(
                &ks,
                false,
                Some("jigglebot-internal-token"),
                "reader1",
                false
            ),
            super::Revoke::Removed(1)
        ),
        "revoking a non-admin key must be allowed when there is no visible admin to strand"
    );
}

/// Dot-segment labels (`.` / `..`) pass the charset check but URL parsers normalize
/// them out of a single path segment, so a key so labelled could be minted but never
/// revoked (codex r16). `validate_key_label` rejects them at the source — the shared
/// helper the facade's `mint_key` calls. Fail-on-revert: drop the dot-label rejection
/// and both `validate_key_label` calls return `Ok`.
#[test]
fn validate_key_label_rejects_dot_segments() {
    for label in [".", ".."] {
        assert!(
            super::validate_key_label(label).is_err(),
            "label {label:?} must be rejected"
        );
    }
    // A normal label passes.
    assert!(super::validate_key_label("laptop").is_ok());
}
