use super::*;

/// The generated TS quotes every property key, so a serde wire name that is not
/// a valid TS identifier (e.g. `#[serde(rename = "max-tokens")]`) still yields
/// parseable TypeScript rather than `max-tokens: "…"` (a subtraction expression).
/// Fails if the key-quoting fix is reverted to a bare `{k}:` emit.
#[test]
fn render_help_ts_quotes_non_identifier_keys() {
    let pairs = vec![
        ("ctx_len".to_string(), "context window size".to_string()),
        ("max-tokens".to_string(), "cap".to_string()),
    ];
    let ts = render_help_ts("Foo", &pairs);
    assert!(ts.contains("export const FooHelp = {"), "header:\n{ts}");
    // Both keys are quoted string literals.
    assert!(
        ts.contains("\"ctx_len\": \"context window size\","),
        "identifier key not quoted:\n{ts}"
    );
    assert!(
        ts.contains("\"max-tokens\": \"cap\","),
        "non-identifier key not quoted:\n{ts}"
    );
    // No BARE (unquoted) hyphenated key — that would be invalid TS.
    assert!(
        !ts.contains("    max-tokens:"),
        "non-identifier key emitted bare (invalid TS):\n{ts}"
    );
    assert!(ts.contains("} as const;"), "closing:\n{ts}");
    assert!(
        ts.contains("export type FooHelpKey = keyof typeof FooHelp;"),
        "key type:\n{ts}"
    );
}

/// Quotes and backslashes inside a help string are escaped (backslash first)
/// so the emitted string literal stays valid.
#[test]
fn render_help_ts_escapes_quotes_and_backslashes() {
    let pairs = vec![("k".to_string(), "a \"b\" \\ c".to_string())];
    let ts = render_help_ts("Foo", &pairs);
    assert!(
        ts.contains("\"k\": \"a \\\"b\\\" \\\\ c\","),
        "value not escaped:\n{ts}"
    );
}

/// `escape_ts_string` escapes the backslash before the quote so a literal
/// backslash-then-quote round-trips correctly.
#[test]
fn escape_ts_string_orders_backslash_before_quote() {
    assert_eq!(escape_ts_string("a\\b"), "a\\\\b");
    assert_eq!(escape_ts_string("a\"b"), "a\\\"b");
    assert_eq!(escape_ts_string("\\\""), "\\\\\\\"");
}

/// Line breaks, tabs, and other control chars are escaped so a multi-line or
/// control-char-bearing help string still yields a valid single-line TS literal
/// (a raw `\n` inside `"…"` would be a syntax error). Fails if the escaper only
/// handles backslash/quote.
#[test]
fn escape_ts_string_escapes_control_chars_and_line_separators() {
    assert_eq!(escape_ts_string("a\nb"), "a\\nb");
    assert_eq!(escape_ts_string("a\rb"), "a\\rb");
    assert_eq!(escape_ts_string("a\tb"), "a\\tb");
    // A bell (U+0007) has no short escape → .
    assert_eq!(escape_ts_string("a\u{0007}b"), "a\\u0007b");
    // JS line/paragraph separators are illegal raw in a TS string literal.
    assert_eq!(escape_ts_string("a\u{2028}b"), "a\\u2028b");
    assert_eq!(escape_ts_string("a\u{2029}b"), "a\\u2029b");
    // No raw newline survives in the output.
    assert!(!escape_ts_string("x\ny").contains('\n'));
}

/// Container `#[serde(rename_all)]` maps a snake_case field to the same wire name
/// serde/ts-rs serialize, so `TsParamHelp` keys stay in lockstep. Mirrors serde's
/// `RenameRule::apply_to_field` (snake_case source — NOT the variant rule).
#[test]
fn apply_rename_all_field_matches_serde_field_rules() {
    // No rule / snake_case / lowercase → unchanged (already snake_case).
    assert_eq!(apply_rename_all_field("ctx_len", None), "ctx_len");
    assert_eq!(
        apply_rename_all_field("ctx_len", Some("snake_case")),
        "ctx_len"
    );
    assert_eq!(
        apply_rename_all_field("ctx_len", Some("lowercase")),
        "ctx_len"
    );
    // camelCase / PascalCase split on underscores (the field-name boundary).
    assert_eq!(
        apply_rename_all_field("ctx_len", Some("camelCase")),
        "ctxLen"
    );
    assert_eq!(
        apply_rename_all_field("rope_freq_base", Some("camelCase")),
        "ropeFreqBase"
    );
    assert_eq!(
        apply_rename_all_field("ctx_len", Some("PascalCase")),
        "CtxLen"
    );
    // UPPER / SCREAMING_SNAKE / kebab / SCREAMING-KEBAB.
    assert_eq!(
        apply_rename_all_field("ctx_len", Some("UPPERCASE")),
        "CTX_LEN"
    );
    assert_eq!(
        apply_rename_all_field("ctx_len", Some("SCREAMING_SNAKE_CASE")),
        "CTX_LEN"
    );
    assert_eq!(
        apply_rename_all_field("ctx_len", Some("kebab-case")),
        "ctx-len"
    );
    assert_eq!(
        apply_rename_all_field("ctx_len", Some("SCREAMING-KEBAB-CASE")),
        "CTX-LEN"
    );
    // Single-word field stays correct under camelCase (no leading capital).
    assert_eq!(
        apply_rename_all_field("threads", Some("camelCase")),
        "threads"
    );
}
