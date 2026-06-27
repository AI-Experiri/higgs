use super::*;

fn p() -> Gemma4Parser {
    Gemma4Parser
}

#[test]
fn handles_gemma4_template_only() {
    // Template with the Gemma-4 envelope + string delimiter markers.
    assert!(p().handles("… '<|tool_call>call:' ~ name … value '<|\"|>' …"));
    // Generic XML `<tool_call>` (no pipe) must NOT match.
    assert!(!p().handles("…{{- '<tool_call>\\n<function=' ~ name }}…"));
    assert!(!p().handles("<|im_start|>{{ role }} plain chatml, no markers"));
}

#[test]
fn parses_single_call() {
    let text = r#"<|tool_call>call:get_weather{location:<|"|>Paris<|"|>}<tool_call|>"#;
    let calls = p().parse(text, "abc").expect("one call");
    assert_eq!(calls.len(), 1);
    let f = &calls[0]["function"];
    assert_eq!(f["name"], "get_weather");
    let args: Value = serde_json::from_str(f["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(args["location"], "Paris");
    assert_eq!(calls[0]["type"], "function");
    assert_eq!(calls[0]["id"], "call_abc_0");
}

#[test]
fn parses_multiple_args_with_typed_values() {
    // String, number, bool, and a nested object (ground truth from the
    // Ollama gemma4 test files).
    let text = r#"<|tool_call>call:process{name:<|"|>test<|"|>,value:42,enabled:true,config:{enabled:true,name:<|"|>x<|"|>}}<tool_call|>"#;
    let calls = p().parse(text, "z").unwrap();
    let args: Value =
        serde_json::from_str(calls[0]["function"]["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(args["name"], "test");
    assert_eq!(args["value"], 42); // bare number stays numeric
    assert_eq!(args["enabled"], true); // bare bool stays bool
    assert_eq!(args["config"]["enabled"], true); // nested object
    assert_eq!(args["config"]["name"], "x");
}

#[test]
fn parses_string_with_embedded_quotes() {
    // Gemma string spans may contain literal `"` unescaped.
    let text = r#"<|tool_call>call:search{query:<|"|>say "hi"<|"|>}<tool_call|>"#;
    let calls = p().parse(text, "q").unwrap();
    let args: Value =
        serde_json::from_str(calls[0]["function"]["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(args["query"], r#"say "hi""#);
}

#[test]
fn parses_two_tool_calls() {
    let text = r#"<|tool_call>call:a{x:1}<tool_call|><|tool_call>call:b{y:2}<tool_call|>"#;
    let calls = p().parse(text, "s").unwrap();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0]["function"]["name"], "a");
    assert_eq!(calls[1]["function"]["name"], "b");
    assert_eq!(calls[1]["id"], "call_s_1");
}

#[test]
fn no_tool_call_returns_none() {
    assert!(p().parse("just a normal answer", "x").is_none());
    // Unterminated envelope (no close tag).
    assert!(p().parse(r#"<|tool_call>call:x{a:1}"#, "x").is_none());
}

#[test]
fn content_strips_think_and_tool_call() {
    let with_preamble =
        "Sure, let me check.\n<|tool_call>call:get_weather{location:<|\"|>Paris<|\"|>}<tool_call|>";
    assert_eq!(p().content(with_preamble), "Sure, let me check.");

    // Forced-open think block: only a trailing </think>, no opening tag.
    let forced =
        "I should look this up.\n</think>\nHere we go.\n<|tool_call>call:x{a:1}<tool_call|>";
    assert_eq!(p().content(forced), "Here we go.");
}

#[test]
fn content_preserves_text_around_and_between_calls() {
    // The content contract: text BEFORE ("A"), BETWEEN ("B"), and AFTER ("C") complete
    // `<|tool_call>`..`<tool_call|>` spans all survive — content() removes only the call
    // markup, it does not stop at the first call (the regression that bit mistral_bracket).
    let text = "A<|tool_call>call:a{}<tool_call|>B<|tool_call>call:b{}<tool_call|>C";
    assert_eq!(p().content(text), "ABC");
}

// --- Regression tests for the three ported-parser bugs ---------------

#[test]
fn bug1_json_quoted_value_with_embedded_comma_and_colon() {
    // Bug 1: an ordinary JSON double-quoted value containing structural
    // bytes (`,` `:`) must be copied verbatim. Before the fix the `,`/`:`
    // inside the string were reinterpreted as structure, serde_json failed,
    // and ALL args were silently dropped.
    let text = r#"<|tool_call>call:search{query:"a,b:c",limit:5}<tool_call|>"#;
    let calls = p().parse(text, "j").unwrap();
    let args: Value =
        serde_json::from_str(calls[0]["function"]["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(args["query"], "a,b:c"); // string survives intact
    assert_eq!(args["limit"], 5); // sibling arg still parsed
}

#[test]
fn bug1_json_quoted_value_with_escaped_quote() {
    // The verbatim copy must honor backslash escaping: an escaped `\"`
    // does not terminate the JSON string, and a trailing `,` after it still
    // separates the next key.
    let text = r#"<|tool_call>call:note{msg:"say \"hi\", ok",n:1}<tool_call|>"#;
    let calls = p().parse(text, "e").unwrap();
    let args: Value =
        serde_json::from_str(calls[0]["function"]["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(args["msg"], r#"say "hi", ok"#);
    assert_eq!(args["n"], 1);
}

#[test]
fn bug2_multibyte_utf8_value_survives() {
    // Bug 2: bytes 0x80-0xFF must not be reinterpreted as a single char.
    // A multibyte UTF-8 value (and a non-ASCII bare key) must round-trip.
    let text = "<|tool_call>call:greet{naïve:<|\"|>café — 日本語<|\"|>}<tool_call|>";
    let calls = p().parse(text, "u").unwrap();
    let args: Value =
        serde_json::from_str(calls[0]["function"]["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(args["naïve"], "café — 日本語");
}

#[test]
fn bug2_multibyte_in_bare_json_value_survives() {
    // A multibyte value inside a JSON double-quoted bare literal must also
    // survive byte-for-byte through the verbatim copy path.
    let text = "<|tool_call>call:echo{text:\"héllo 世界\"}<tool_call|>";
    let calls = p().parse(text, "w").unwrap();
    let args: Value =
        serde_json::from_str(calls[0]["function"]["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(args["text"], "héllo 世界");
}

#[test]
fn bug3_content_strips_gemma_channel_thinking_block() {
    // Bug 3: Gemma-4 reasoning uses `<|channel>thought\n…<channel|>`, not
    // `<think>`. content() must drop the channel block (markup included).
    // Layout matches the Ollama `thinking_then_content` test.
    let input = "<|channel>thought\nLet me think about this...<channel|>The answer is 42.";
    assert_eq!(p().content(input), "The answer is 42.");
}

#[test]
fn bug3_content_strips_channel_block_before_tool_call() {
    // Channel reasoning followed by content and then a tool call: the
    // channel markup must not leak, and the tool-call envelope is removed.
    let input = "<|channel>thought\nplanning<channel|>On it.\n<|tool_call>call:x{a:1}<tool_call|>";
    assert_eq!(p().content(input), "On it.");
}

#[test]
fn bare_multibyte_args_do_not_panic() {
    // A model can emit malformed BARE (unquoted) multibyte args; the byte-walk
    // must advance whole runes — a single-byte step lands `i` mid-codepoint and
    // panics the next `s[i..]` slice. The call still parses; invalid bare args
    // degrade to empty.
    for text in [
        "<|tool_call>call:f{a:naïve}<tool_call|>",
        "<|tool_call>call:f{🎉:1}<tool_call|>",
        "<|tool_call>call:f{a:café,b:naïve}<tool_call|>",
    ] {
        let calls = p().parse(text, "w").expect("tool call still parses");
        assert_eq!(calls[0]["function"]["name"], "f", "name parsed: {text}");
        assert_eq!(
            calls[0]["function"]["arguments"], "{}",
            "invalid bare args degrade to empty, no panic: {text}"
        );
    }
}

#[test]
fn id_and_open_markers() {
    assert_eq!(p().id(), "gemma4");
    assert_eq!(p().open_markers(), &[OPEN_TAG]);
}

#[test]
fn empty_name_call_is_dropped() {
    // `call:{…}` — the name before `{` is empty, so parse_one returns None and
    // the whole parse yields no calls.
    assert!(p()
        .parse("<|tool_call>call:{a:1}<tool_call|>", "x")
        .is_none());
}

#[test]
fn unterminated_string_delimiter_degrades_to_empty() {
    // A `<|"|>` span that never closes: gemma4_args_to_json copies the rest
    // verbatim and stops, so the JSON does not parse → empty args. The call
    // still emits (parse_one only requires the name + brace).
    let text = "<|tool_call>call:f{a:<|\"|>oops}<tool_call|>";
    let calls = p().parse(text, "u").expect("tool call still parses");
    assert_eq!(calls[0]["function"]["name"], "f");
    assert_eq!(calls[0]["function"]["arguments"], "{}");
}

#[test]
fn unterminated_json_quote_degrades_to_empty() {
    // A bare JSON double-quoted value with no closing quote: json_quoted_string_end
    // returns None, so the verbatim-copy path is skipped and the bytes fall through
    // to the rune copy. The malformed JSON degrades to empty args without panic.
    let text = "<|tool_call>call:f{a:\"oops}<tool_call|>";
    let calls = p().parse(text, "q").expect("tool call still parses");
    assert_eq!(calls[0]["function"]["name"], "f");
    assert_eq!(calls[0]["function"]["arguments"], "{}");
}

#[test]
fn whitespace_after_brace_before_key_is_skipped() {
    // skip_space advances past the space between `{`/`,` and the bare key, so the
    // key is still recognized and quoted into valid JSON.
    let text = "<|tool_call>call:f{ a:1, b:2}<tool_call|>";
    let calls = p().parse(text, "s").expect("one call");
    let args: Value =
        serde_json::from_str(calls[0]["function"]["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(args["a"], 1);
    assert_eq!(args["b"], 2);
}
