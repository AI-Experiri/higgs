//! HuggingFace-card recommended sampling (`SamplingSource` impls + the parser).
//!
//! Deterministic, drop-not-clamp parse of recommended sampling values out of a
//! model card's prose. The network fetch is **async** and best-effort
//! (fail-open) — it lives in [`fetch_card_sampling`], called from the async tune
//! handler, which then injects a [`StaticSamplingSource`] into the (sync)
//! suggester. The default suggester uses [`EmptySamplingSource`] (no network).

use crate::worker::engine::llamacpp::params::LlamaCppSamplingParams;

use super::{ModelMeta, SamplingSource};

/// A sampling source that recommends nothing — the static / offline default.
pub struct EmptySamplingSource;

impl SamplingSource for EmptySamplingSource {
    fn recommend(&self, _meta: &ModelMeta) -> LlamaCppSamplingParams {
        LlamaCppSamplingParams::default()
    }
}

/// A sampling source wrapping a pre-fetched/parsed recommendation (the button
/// path fetches the card asynchronously, then injects this).
pub struct StaticSamplingSource(pub LlamaCppSamplingParams);

impl SamplingSource for StaticSamplingSource {
    fn recommend(&self, _meta: &ModelMeta) -> LlamaCppSamplingParams {
        self.0.clone()
    }
}

/// True when any sampling field is set.
pub fn has_any_sampling(s: &LlamaCppSamplingParams) -> bool {
    s != &LlamaCppSamplingParams::default()
}

/// Strip fenced code blocks (```…```) so example CLI flags inside code don't get
/// mined as "recommended" values.
fn strip_code_fences(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut in_fence = false;
    for line in text.lines() {
        if line.trim_start().starts_with("```") {
            in_fence = !in_fence;
            continue;
        }
        if !in_fence {
            out.push_str(line);
            out.push('\n');
        }
    }
    out
}

/// Number-search window after a key (chars). Bounds how far past the key a value
/// may sit, so a far-away clause's number isn't mistaken for this key's value.
const NUMBER_WINDOW: usize = 28;

/// Every recognized sampling key spelling — used as **boundaries** so a value that
/// belongs to a LATER key (`temperature and top_p 0.95`) isn't mis-assigned to an
/// earlier one: the search window is truncated at the next key it meets.
const BOUNDARY_KEYS: &[&str] = &[
    "temperature",
    "temp",
    "top_p",
    "top-p",
    "min_p",
    "min-p",
    "top_k",
    "top-k",
    "typical_p",
    "typical-p",
];

/// A byte that, if adjacent to a key match, means the key is really a substring of
/// a larger identifier/word (so `temp` inside `template`/`attempt` is NOT a key).
/// ASCII alphanumerics + `_` (keys like `top_p` carry their own underscore, so an
/// underscore on the OUTSIDE marks a longer token).
fn is_word_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// Find `needle` in `hay` at or after `from`, but only as a **whole token** — the
/// bytes immediately before/after the match must not be word bytes. This rejects
/// `temp` inside `template`/`attempt`/`contemplate` (and likewise any key that is a
/// substring of a longer word), so an unrelated number in card prose can't be
/// mis-recorded as a sampling recommendation. ASCII keys only.
fn find_token(hay: &str, needle: &str, from: usize) -> Option<usize> {
    let hb = hay.as_bytes();
    let mut search = from;
    while let Some(rel) = hay.get(search..)?.find(needle) {
        let start = search + rel;
        let end = start + needle.len();
        let before_ok = start == 0 || !is_word_byte(hb[start - 1]);
        let after_ok = end >= hb.len() || !is_word_byte(hb[end]);
        if before_ok && after_ok {
            return Some(start);
        }
        // Overlapping matches are possible (`temp` in `temperature`); step one byte.
        search = start + 1;
    }
    None
}

/// Parse the first numeric token within [`NUMBER_WINDOW`] chars after `key` in
/// `hay`, **stopping at the next recognized key**. Connector words ("to", "is",
/// "=") between the key and the number are tolerated; a key mentioned with no
/// nearby number before the next key ("temperature is important", "temperature and
/// top_p 0.95") yields `None` rather than grabbing a far/unrelated value. The key
/// must appear as a WHOLE token (see [`find_token`]) — `temp` inside `template`
/// never matches.
fn number_after(hay: &str, key: &str) -> Option<f64> {
    let mut from = 0;
    while let Some(start) = find_token(hay, key, from) {
        let after = start + key.len();
        let end = (after + NUMBER_WINDOW).min(hay.len());
        // Respect char boundaries for the window slice (all keys/numbers are ASCII).
        let window = hay.get(after..end).unwrap_or("");
        // Truncate at the EARLIEST following whole-token key so we don't read its
        // value (token-aware so a word like `template` doesn't truncate the window).
        let scan_end = BOUNDARY_KEYS
            .iter()
            .filter_map(|bk| find_token(window, bk, 0))
            .min()
            .unwrap_or(window.len());
        if let Some(n) = first_number_in(&window[..scan_end]) {
            return Some(n);
        }
        from = after;
    }
    None
}

/// First numeric token anywhere in `s` (a leading `-`, digits, at most one `.`),
/// with a trailing sentence period stripped (`"0.0."` → `0.0`).
fn first_number_in(s: &str) -> Option<f64> {
    let b = s.as_bytes();
    let mut i = 0;
    while i < b.len() {
        let starts_number = b[i].is_ascii_digit()
            || ((b[i] == b'-' || b[i] == b'.') && i + 1 < b.len() && b[i + 1].is_ascii_digit());
        if starts_number {
            let start = i;
            if b[i] == b'-' {
                i += 1;
            }
            let mut seen_dot = false;
            while i < b.len() {
                match b[i] {
                    c if c.is_ascii_digit() => i += 1,
                    b'.' if !seen_dot => {
                        seen_dot = true;
                        i += 1;
                    }
                    _ => break,
                }
            }
            // Strip a trailing '.' (sentence period, not a decimal point).
            let mut end = i;
            if end > start && b[end - 1] == b'.' {
                end -= 1;
            }
            return s.get(start..end)?.parse::<f64>().ok();
        }
        i += 1;
    }
    None
}

/// Try several spellings of a key, returning the first number found.
fn number_after_any(hay: &str, keys: &[&str]) -> Option<f64> {
    keys.iter().find_map(|k| number_after(hay, k))
}

/// Deterministically parse recommended sampling out of card prose. Out-of-range
/// values are **dropped**, not clamped (a wrong number is worse than no number).
pub fn parse_card_sampling(text: &str) -> LlamaCppSamplingParams {
    let lower = strip_code_fences(text).to_lowercase();
    let mut s = LlamaCppSamplingParams::default();

    if let Some(t) = number_after_any(&lower, &["temperature", "temp"]) {
        if (0.0..=2.0).contains(&t) {
            s.temperature = Some(t as f32);
        }
    }
    if let Some(p) = number_after_any(&lower, &["top_p", "top-p"]) {
        if (0.0..=1.0).contains(&p) {
            s.top_p = Some(p as f32);
        }
    }
    if let Some(p) = number_after_any(&lower, &["min_p", "min-p"]) {
        if (0.0..=1.0).contains(&p) {
            s.min_p = Some(p as f32);
        }
    }
    if let Some(k) = number_after_any(&lower, &["top_k", "top-k"]) {
        if (0.0..=1000.0).contains(&k) {
            s.top_k = Some(k as i32);
        }
    }
    if let Some(p) = number_after_any(&lower, &["typical_p", "typical-p"]) {
        if (0.0..=1.0).contains(&p) {
            s.typical_p = Some(p as f32);
        }
    }
    s
}

/// Parse recommended sampling out of a STRUCTURED `generation_config.json` — real
/// JSON fields (`temperature`/`top_p`/`top_k`/`min_p`/`typical_p`), not scraped
/// prose. This is the PREFERRED source (the original model repo ships it);
/// drop-not-clamp out-of-range values, same as the prose parser. Returns `None`
/// for invalid JSON.
pub fn parse_generation_config(bytes: &[u8]) -> Option<LlamaCppSamplingParams> {
    let v: serde_json::Value = serde_json::from_slice(bytes).ok()?;
    let mut s = LlamaCppSamplingParams::default();
    if let Some(t) = v.get("temperature").and_then(serde_json::Value::as_f64) {
        if (0.0..=2.0).contains(&t) {
            s.temperature = Some(t as f32);
        }
    }
    if let Some(p) = v.get("top_p").and_then(serde_json::Value::as_f64) {
        if (0.0..=1.0).contains(&p) {
            s.top_p = Some(p as f32);
        }
    }
    if let Some(k) = v.get("top_k").and_then(serde_json::Value::as_i64) {
        if (0..=1000).contains(&k) {
            s.top_k = Some(k as i32);
        }
    }
    if let Some(p) = v.get("min_p").and_then(serde_json::Value::as_f64) {
        if (0.0..=1.0).contains(&p) {
            s.min_p = Some(p as f32);
        }
    }
    if let Some(p) = v.get("typical_p").and_then(serde_json::Value::as_f64) {
        if (0.0..=1.0).contains(&p) {
            s.typical_p = Some(p as f32);
        }
    }
    Some(s)
}

/// Best-effort fetch of a model's recommended sampling via the HuggingFace hub
/// client (`src/hub.rs`, with the `reqwest` fallback). Fail-open: a non-HF id, a
/// network/auth/not-found failure, or no recommendation all return `None`, leaving
/// sampling untouched. Prefers the STRUCTURED `generation_config.json`; falls back
/// to scraping the `README.md` prose (which is all GGUF quant repos usually ship).
/// The caller bounds it with a timeout.
pub async fn fetch_card_sampling(repo_id: &str) -> Option<LlamaCppSamplingParams> {
    // Only HuggingFace ids (`org/model`) have a card; ollama ids are skipped.
    if repo_id.starts_with("ollama/") || repo_id.matches('/').count() != 1 {
        return None;
    }
    // 1. Structured generation_config.json (preferred — real fields, no scraping).
    if let Ok(bytes) = crate::hub::fetch_bytes(repo_id, "generation_config.json").await {
        if let Some(s) = parse_generation_config(&bytes) {
            if has_any_sampling(&s) {
                return Some(s);
            }
        }
    }
    // 2. README.md prose fallback (GGUF quant repos rarely ship generation_config.json).
    if let Ok(bytes) = crate::hub::fetch_bytes(repo_id, "README.md").await {
        let text = String::from_utf8_lossy(&bytes);
        let s = parse_card_sampling(&text);
        if has_any_sampling(&s) {
            return Some(s);
        }
    }
    None
}

#[cfg(test)]
mod tests {
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
}
