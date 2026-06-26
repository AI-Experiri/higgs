
use super::*;

/// A lean JSON object carrying only a couple of base fields deserializes into
/// the full struct with every other field at its `None`/empty default — the
/// back-compat / quick-load shape.
#[test]
fn lean_json_deserializes_into_full_params() {
    let p: LlamaCppParams =
        serde_json::from_value(serde_json::json!({"ctx_len": 4096, "gpu_layers": 32})).unwrap();
    assert_eq!(p.ctx_len, 4096);
    assert_eq!(p.gpu_layers, 32);
    assert!(p.type_k.is_none() && p.cpu_moe.is_none() && p.n_seq_max.is_none());
    assert!(p.cpu_buft_overrides.is_empty() && p.kv_overrides.is_empty());
}

/// `Option` overrides are omitted when `None`; the always-serialized vec
/// fields appear as `[]` (so their required-array bindings stay honest). A
/// quick-load thus carries only the base fields + the empty advanced vecs.
#[test]
fn absent_optionals_do_not_serialize() {
    let bare = LlamaCppParams::base(4096, u32::MAX, 4);
    let v = serde_json::to_value(&bare).unwrap();
    let obj = v.as_object().unwrap();
    assert!(
        obj.contains_key("ctx_len")
            && obj.contains_key("gpu_layers")
            && obj.contains_key("threads"),
        "base fields present: {obj:?}"
    );
    // Vec fields always serialize (possibly empty) — matches the required-array binding.
    assert_eq!(obj["cpu_buft_overrides"], serde_json::json!([]));
    assert_eq!(obj["kv_overrides"], serde_json::json!([]));
    // Every Option override is omitted when None.
    assert!(
        !obj.contains_key("type_k")
            && !obj.contains_key("flash_attn")
            && !obj.contains_key("cpu_moe"),
        "None options omitted: {obj:?}"
    );
}

/// A fully-populated round-trip preserves the new fields (cpu_moe, n_seq_max,
/// rope_scaling_type, kv_overrides) — the expanded coverage survives the wire.
#[test]
fn full_params_round_trip() {
    let full = LlamaCppParams {
        ctx_len: 8192,
        gpu_layers: u32::MAX,
        threads: 8,
        cpu_moe: Some(true),
        n_seq_max: Some(4),
        n_threads_batch: Some(6),
        swa_full: Some(false),
        type_k: Some(KvCacheKind::Q8_0),
        rope_scaling_type: Some(RopeScalingType::Yarn),
        kv_overrides: vec![KvOverride {
            key: "llama.context_length".into(),
            value: "8192".into(),
        }],
        ..Default::default()
    };
    let json = serde_json::to_string(&full).unwrap();
    let back: LlamaCppParams = serde_json::from_str(&json).unwrap();
    assert_eq!(back, full);
}

/// Overlaying a request set onto a card base: the request's set fields win,
/// the base's other samplers survive. This is the chat-time merge that lets a
/// tuned/card recommendation (top_k/min_p/…) apply while a per-request
/// temperature/top_p/penalty still overrides.
#[test]
fn overlaid_with_request_overrides_base_keeps_rest() {
    let base = LlamaCppSamplingParams {
        temperature: Some(0.6),
        top_k: Some(40),
        min_p: Some(0.05),
        penalty_repeat: Some(1.1),
        logit_bias: vec![LogitBias {
            token: 7,
            bias: 1.0,
        }],
        ..Default::default()
    };
    // A request that sets only temperature + top_p (the common OpenAI subset).
    let req = LlamaCppSamplingParams {
        temperature: Some(1.2),
        top_p: Some(0.9),
        ..Default::default()
    };
    let merged = base.overlaid_with(&req);
    assert_eq!(merged.temperature, Some(1.2), "request temp wins");
    assert_eq!(merged.top_p, Some(0.9), "request top_p applied");
    assert_eq!(merged.top_k, Some(40), "base top_k survives");
    assert_eq!(merged.min_p, Some(0.05), "base min_p survives");
    assert_eq!(merged.penalty_repeat, Some(1.1), "base penalty survives");
    assert_eq!(merged.logit_bias.len(), 1, "base logit_bias survives");
    // An all-empty request leaves the base untouched.
    let untouched = base.overlaid_with(&LlamaCppSamplingParams::default());
    assert_eq!(untouched, base);
    // A request logit_bias replaces the base's (last-writer-wins on the vec).
    let with_bias = base.overlaid_with(&LlamaCppSamplingParams {
        logit_bias: vec![LogitBias {
            token: 9,
            bias: -2.0,
        }],
        ..Default::default()
    });
    assert_eq!(with_bias.logit_bias.len(), 1);
    assert_eq!(with_bias.logit_bias[0].token, 9, "request bias wins");
}

/// `unsupported_sampler` reports `None` for a set of only-supported samplers and
/// the precise field name for each not-yet-applied advanced sampler — the worker's
/// fail-loud guard (HG013) against silently dropping a grammar/logit_bias/dry/mirostat.
#[test]
fn unsupported_sampler_flags_only_unapplied_advanced_samplers() {
    // Supported-only set (temperature/top_p/penalties) ⇒ nothing flagged.
    let ok = LlamaCppSamplingParams {
        temperature: Some(0.7),
        top_p: Some(0.9),
        penalty_repeat: Some(1.1),
        ..Default::default()
    };
    assert_eq!(ok.unsupported_sampler(), None, "supported samplers pass");
    // Each advanced sampler is flagged by name.
    assert_eq!(
        LlamaCppSamplingParams {
            grammar: Some(GrammarParams {
                gbnf: "root ::= \"x\"".to_string(),
                root: "root".to_string(),
            }),
            ..Default::default()
        }
        .unsupported_sampler(),
        Some("grammar"),
    );
    assert_eq!(
        LlamaCppSamplingParams {
            logit_bias: vec![LogitBias {
                token: 1,
                bias: 2.0,
            }],
            ..Default::default()
        }
        .unsupported_sampler(),
        Some("logit_bias"),
    );
    assert_eq!(
        LlamaCppSamplingParams {
            mirostat: Some(MirostatParams {
                version: 2,
                tau: 5.0,
                eta: 0.1,
            }),
            ..Default::default()
        }
        .unsupported_sampler(),
        Some("mirostat"),
    );
}

/// `has_overrides` is false for a base-only load and true once any engine
/// override (an optional field or a non-empty advanced vec) is set.
#[test]
fn has_overrides_detects_engine_overrides() {
    assert!(
        !LlamaCppParams::base(4096, u32::MAX, 8).has_overrides(),
        "base-only load carries no overrides"
    );
    let mut p = LlamaCppParams::base(4096, u32::MAX, 8);
    p.flash_attn = Some(FlashAttn::On);
    assert!(p.has_overrides(), "an optional override counts");
    let mut p2 = LlamaCppParams::base(4096, u32::MAX, 8);
    p2.kv_overrides.push(KvOverride {
        key: "k".into(),
        value: "v".into(),
    });
    assert!(p2.has_overrides(), "a non-empty advanced vec counts");
}

/// The sampling type covers the full sampler surface; a lean card-derived
/// subset (temp/top_k/top_p/min_p/penalties) round-trips with the rest `None`.
#[test]
fn sampling_params_lean_round_trip() {
    let s = LlamaCppSamplingParams {
        temperature: Some(0.6),
        top_k: Some(20),
        top_p: Some(0.95),
        min_p: Some(0.0),
        penalty_repeat: Some(1.1),
        ..Default::default()
    };
    let json = serde_json::to_string(&s).unwrap();
    let back: LlamaCppSamplingParams = serde_json::from_str(&json).unwrap();
    assert_eq!(back, s);
    assert!(back.mirostat.is_none() && back.grammar.is_none() && back.dry.is_none());
}
