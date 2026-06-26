
use super::*;

fn p() -> MistralBracketParser {
    MistralBracketParser
}

#[test]
fn handles_mistral_template_only() {
    // A template that renders both bracket tokens.
    assert!(p().handles("… '[TOOL_CALLS]' ~ name ~ '[ARGS]' ~ args …"));
    // XML family — no bracket tokens.
    assert!(!p().handles("… '<function=' ~ name … '<parameter=' ~ k …"));
    // Has [TOOL_CALLS] but no [ARGS] — not this dialect.
    assert!(!p().handles("plain [TOOL_CALLS] only, no args tag"));
}

#[test]
fn parses_single_call() {
    // Ground truth from ministral_test.go line 46.
    let text = r#"[TOOL_CALLS]get_weather[ARGS]{"location": "San Francisco"}"#;
    let calls = p().parse(text, "abc").expect("one call");
    assert_eq!(calls.len(), 1);
    let f = &calls[0]["function"];
    assert_eq!(f["name"], "get_weather");
    let args: Value = serde_json::from_str(f["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(args["location"], "San Francisco");
    assert_eq!(calls[0]["type"], "function");
    assert_eq!(calls[0]["id"], "call_abc_0");
}

#[test]
fn parses_multiple_args_with_nested_json_value() {
    // From ministral_test.go line 64 — nested object argument.
    let text = r#"[TOOL_CALLS]update_config[ARGS]{"settings": {"user": {"name": "John", "age": 30}}, "theme": "dark"}"#;
    let calls = p().parse(text, "z").unwrap();
    let args: Value =
        serde_json::from_str(calls[0]["function"]["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(args["theme"], "dark");
    assert_eq!(args["settings"]["user"]["name"], "John");
    assert_eq!(args["settings"]["user"]["age"], 30); // numeric stays typed
}

#[test]
fn parses_two_tool_calls() {
    // From ministral_test.go line 147 — concatenated calls.
    let text = r#"[TOOL_CALLS]get_weather[ARGS]{"location": "NYC"}[TOOL_CALLS]get_time[ARGS]{"timezone": "EST"}"#;
    let calls = p().parse(text, "s").unwrap();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0]["function"]["name"], "get_weather");
    assert_eq!(calls[1]["function"]["name"], "get_time");
    assert_eq!(calls[1]["id"], "call_s_1");
}

#[test]
fn no_tool_call_returns_none() {
    assert!(p().parse("just a normal answer", "x").is_none());
    // Unterminated JSON — find_json_end never closes.
    assert!(p().parse("[TOOL_CALLS]test[ARGS]{", "x").is_none());
    // Missing [ARGS] separator.
    assert!(p().parse("[TOOL_CALLS]test{}", "x").is_none());
}

#[test]
fn content_strips_think_and_tool_call() {
    // Preamble before the first [TOOL_CALLS] is the content.
    let with_preamble = r#"Sure, let me check.
[TOOL_CALLS]get_weather[ARGS]{"location": "NYC"}"#;
    assert_eq!(p().content(with_preamble), "Sure, let me check.");

    // Explicit [THINK]…[/THINK] block is stripped.
    let with_think = r#"[THINK]reasoning here[/THINK]done[TOOL_CALLS]x[ARGS]{}"#;
    assert_eq!(p().content(with_think), "done");
}

#[test]
fn content_strips_forced_open_think_without_opening_tag() {
    // Template forced the think block open → only a trailing [/THINK].
    // (ministral_test.go line 337 exercises this lead-in shape.)
    let forced = "let me think[/THINK][TOOL_CALLS]test[ARGS]{}";
    assert_eq!(p().content(forced), "");
}

#[test]
fn content_preserves_text_after_and_between_calls() {
    // Contract: content is the text with call markup REMOVED — preamble, gaps BETWEEN
    // calls, and trailing text all survive (this dialect has no close marker, so the span
    // end is the balanced [ARGS] JSON, mirroring parse()). The old `content()` kept only
    // the text before the first [TOOL_CALLS], dropping "mid" and "post".
    let text = r#"pre[TOOL_CALLS]get_weather[ARGS]{"location":"NYC"}mid[TOOL_CALLS]get_time[ARGS]{"timezone":"EST"}post"#;
    assert_eq!(p().content(text), "premidpost");
}
