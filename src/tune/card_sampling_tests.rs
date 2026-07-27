use super::*;

#[test]
fn parses_recommended_sampling_drop_not_clamp() {
    let s = parse_card_sampling("Use temperature 0.6, top_p 0.95, top_k 20, min_p 0.0.");
    assert_eq!(s.temperature, Some(0.6));
    assert_eq!(s.top_p, Some(0.95));
    assert_eq!(s.top_k, Some(20));
    assert_eq!(s.min_p, Some(0.0));

    // Out-of-range temperature is DROPPED, not clamped.
    assert!(parse_card_sampling("temperature 9.0").temperature.is_none());
    // No numbers → nothing recommended.
    assert!(!has_any_sampling(&parse_card_sampling("no numbers here")));
}

#[test]
fn ignores_values_inside_code_fences() {
    let card = "Recommended: temperature 0.7.\n\n```\n--temp 1.9 --top-k 100\n```\n";
    let s = parse_card_sampling(card);
    assert_eq!(s.temperature, Some(0.7), "prose value wins");
    // The code-fence top-k 100 is stripped, so no top_k is mined.
    assert!(s.top_k.is_none(), "code-fence values are ignored");
}

#[test]
fn key_in_prose_without_number_yields_nothing() {
    // "temperature is important" — a letter follows the key, so no number.
    let s = parse_card_sampling("The temperature is important for this model.");
    assert!(s.temperature.is_none());
}

#[test]
fn key_without_value_does_not_steal_a_later_keys_number() {
    // "temperature and top_p 0.95" — temperature has no value before the next key
    // (top_p), so it must stay UNSET; top_p gets 0.95 (not stolen by temperature).
    let s = parse_card_sampling("temperature and top_p 0.95");
    assert!(
        s.temperature.is_none(),
        "temperature must not steal top_p's value: {:?}",
        s.temperature
    );
    assert_eq!(s.top_p, Some(0.95));
}

#[test]
fn hyphen_and_underscore_spellings_both_parse() {
    let s = parse_card_sampling("set top-p to 0.9 and top_k to 40");
    assert_eq!(s.top_p, Some(0.9));
    assert_eq!(s.top_k, Some(40));
}

#[test]
fn generation_config_json_parses_structured_fields() {
    // The structured source: real JSON fields, in-range values kept, no scraping.
    let s = parse_generation_config(
        br#"{"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "do_sample": true}"#,
    )
    .unwrap();
    assert_eq!(s.temperature, Some(0.6));
    assert_eq!(s.top_p, Some(0.95));
    assert_eq!(s.top_k, Some(20));
    assert_eq!(s.min_p, Some(0.0));
    // Drop-not-clamp: out-of-range temperature is discarded, not coerced.
    let s = parse_generation_config(br#"{"temperature": 9.0, "top_p": 0.9}"#).unwrap();
    assert!(s.temperature.is_none(), "out-of-range temp dropped");
    assert_eq!(s.top_p, Some(0.9), "in-range top_p kept");
    // Invalid JSON → None (fail-open at the caller).
    assert!(parse_generation_config(b"not json").is_none());
    // No recognized fields → an all-default (empty) set, not None.
    assert!(!has_any_sampling(
        &parse_generation_config(br#"{"x": 1}"#).unwrap()
    ));
}

#[test]
fn key_as_substring_of_a_word_is_not_matched() {
    // `temp` lives inside `template`/`attempt`/`contemplate` — a nearby number in
    // such prose must NOT be recorded as a temperature recommendation (it would
    // then persist into models.json and skew every later chat). The real key must
    // appear as a whole token.
    for prose in [
        "the prompt template 0.8 is provided below",
        "our first attempt 0.9 produced gibberish",
        "we contemplate 0.7 different strategies",
    ] {
        let s = parse_card_sampling(prose);
        assert!(
            s.temperature.is_none(),
            "substring `temp` in prose must not set temperature ({prose:?}): {:?}",
            s.temperature
        );
    }
    // A real whole-token `temp` abbreviation still parses.
    assert_eq!(parse_card_sampling("temp 0.6").temperature, Some(0.6));
    // And a word containing a key before a real key doesn't block the real one.
    let s = parse_card_sampling("see the template, then set temperature 0.5");
    assert_eq!(s.temperature, Some(0.5));
}

#[test]
fn generation_config_parses_float_top_k() {
    // Some configs write top_k as a JSON float (`20.0`); an integer-valued float must
    // parse like the other (as_f64) params, not be silently dropped by an int-only
    // `as_i64`.
    let s = parse_generation_config(br#"{"top_k": 20.0}"#).expect("parsed");
    assert_eq!(s.top_k, Some(20));
}

#[test]
fn generation_config_drops_fractional_top_k() {
    // top_k is a discrete count: a fractional float (`20.5`) is DROPPED, not
    // truncated to 20 — coercion would persist a value the config never stated.
    let s = parse_generation_config(br#"{"top_k": 20.5}"#).expect("parsed");
    assert_eq!(s.top_k, None);
}

#[test]
fn empty_and_static_sources_recommend_expected() {
    // EmptySamplingSource always recommends nothing (the offline default).
    let empty = EmptySamplingSource.recommend(&ModelMeta::default());
    assert_eq!(empty, LlamaCppSamplingParams::default());
    assert!(!has_any_sampling(&empty));

    // StaticSamplingSource echoes whatever it was built with, regardless of meta.
    let pinned = LlamaCppSamplingParams {
        temperature: Some(0.8),
        top_p: Some(0.92),
        ..Default::default()
    };
    let src = StaticSamplingSource(pinned.clone());
    let got = src.recommend(&ModelMeta {
        id: "org/whatever".into(),
        ..Default::default()
    });
    assert_eq!(got, pinned);
    assert!(has_any_sampling(&got));
}

#[test]
fn typical_p_is_mined_from_prose() {
    // typical_p (and the hyphen spelling) round out the prose key set.
    let s = parse_card_sampling("we set typical_p 0.85 for this run");
    assert_eq!(s.typical_p, Some(0.85));
    let s = parse_card_sampling("set typical-p to 0.5");
    assert_eq!(s.typical_p, Some(0.5));
    // Out-of-range typical_p is DROPPED, not clamped.
    assert!(parse_card_sampling("typical_p 1.5").typical_p.is_none());
}

#[test]
fn typical_p_is_mined_from_generation_config() {
    let s = parse_generation_config(br#"{"typical_p": 0.9}"#).expect("parsed");
    assert_eq!(s.typical_p, Some(0.9));
    // Out-of-range typical_p in structured config is dropped, not clamped.
    let s = parse_generation_config(br#"{"typical_p": 2.0}"#).expect("parsed");
    assert!(s.typical_p.is_none());
}

#[test]
fn negative_number_token_is_parsed() {
    // A leading `-` starts a numeric token (e.g. a negative penalty window written
    // next to a recognized key); first_number_in must consume the sign. top_k's
    // range starts at 0, so a negative top_k is parsed THEN dropped as out-of-range.
    let s = parse_card_sampling("top_k -5 was tried");
    assert!(
        s.top_k.is_none(),
        "negative top_k parses then drops (range 0..=1000): {:?}",
        s.top_k
    );
    // A negative value still parses for a key whose lower bound is 0.0: temperature
    // -0.5 is in-window, parsed, and dropped (out of 0.0..=2.0).
    let s = parse_card_sampling("temperature -0.5 reported");
    assert!(s.temperature.is_none(), "negative temp dropped: {:?}", s);
}

#[test]
fn trailing_sentence_period_is_stripped_from_number() {
    // "0.0." — the trailing period is a sentence stop, not a second decimal point;
    // the value parses as 0.0, not as an invalid "0.0.".
    let s = parse_card_sampling("Recommended min_p 0.0. Then generate.");
    assert_eq!(s.min_p, Some(0.0));
    // A whole number followed by a period likewise strips the period (top_k 20.).
    let s = parse_card_sampling("top_k 20. is plenty");
    assert_eq!(s.top_k, Some(20));
}

#[tokio::test]
async fn fetch_card_sampling_skips_non_hf_ids() {
    // Fail-open, no network: ollama ids and ids without exactly one slash return
    // None before any fetch is attempted (the early-return guard, no I/O).
    assert!(fetch_card_sampling("ollama/llama3:8b").await.is_none());
    assert!(fetch_card_sampling("bareword").await.is_none());
    assert!(
        fetch_card_sampling("a/b/c").await.is_none(),
        "two slashes is not org/model"
    );
}
