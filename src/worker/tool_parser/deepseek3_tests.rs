use super::*;

/// Single call, exact byte layout from the Ollama Go `tool_call_simple` test.
const SINGLE: &str = "I'll check the weather.<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{\"location\":\"Paris\"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>";

fn p() -> DeepSeek3Parser {
    DeepSeek3Parser
}

#[test]
fn handles_deepseek_template_only() {
    assert!(p().handles(&format!("…{CALLS_BEGIN}…{TOOL_SEP}…")));
    // Generic XML tool-call template has none of the ▁ unicode tags.
    assert!(!p().handles("<tool_call>\n<function= … <parameter= …"));
    assert!(!p().handles("<|im_start|>{{ role }} plain chatml"));
}

#[test]
fn parses_single_call() {
    let calls = p().parse(SINGLE, "abc").expect("one call");
    assert_eq!(calls.len(), 1);
    let f = &calls[0]["function"];
    assert_eq!(f["name"], "get_weather");
    let args: Value = serde_json::from_str(f["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(args["location"], "Paris");
    assert_eq!(calls[0]["type"], "function");
    assert_eq!(calls[0]["id"], "call_abc_0");
}

#[test]
fn parses_multiple_calls() {
    let text = "Getting weather for both cities.<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{\"location\":\"Paris\"}<｜tool▁call▁end｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{\"location\":\"London\"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>";
    let calls = p().parse(text, "s").unwrap();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0]["function"]["name"], "get_weather");
    assert_eq!(calls[1]["id"], "call_s_1");
    let a1: Value =
        serde_json::from_str(calls[1]["function"]["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(a1["location"], "London");
}

#[test]
fn parses_complex_typed_arguments() {
    // From the Go `complex_tool_arguments` test: arrays, nested objects,
    // bools and numbers must keep their JSON types.
    let text = "Processing data.<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>process_data<｜tool▁sep｜>{\"items\":[\"item1\",\"item2\"],\"config\":{\"enabled\":true,\"threshold\":0.95}}<｜tool▁call▁end｜><｜tool▁calls▁end｜>";
    let calls = p().parse(text, "z").unwrap();
    let args: Value =
        serde_json::from_str(calls[0]["function"]["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(args["items"][0], "item1");
    assert_eq!(args["config"]["enabled"], true);
    assert_eq!(args["config"]["threshold"], 0.95);
}

#[test]
fn no_tool_call_returns_none() {
    assert!(p().parse("Hello, how are you?", "x").is_none());
    // calls-begin present but no well-formed call body.
    assert!(p()
        .parse("text<｜tool▁calls▁begin｜><｜tool▁calls▁end｜>", "x")
        .is_none());
}

#[test]
fn content_strips_think_and_calls() {
    assert_eq!(p().content(SINGLE), "I'll check the weather.");
    // Explicit reasoning block before the answer.
    let with_think = "Let me check...</think>I'll get that.<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{}<｜tool▁call▁end｜><｜tool▁calls▁end｜>";
    assert_eq!(p().content(with_think), "I'll get that.");
}

#[test]
fn content_strips_forced_open_think_without_opening_tag() {
    let forced = "Reasoning about the weather request.\n</think>\nDone.";
    assert_eq!(p().content(forced), "Done.");
}

#[test]
fn content_preserves_text_around_and_between_calls() {
    // The content contract: text BEFORE ("A"), BETWEEN ("B"), and AFTER ("C") complete
    // call envelopes all survive — content() removes only the markup, it does not stop
    // at the first envelope (the regression that bit mistral_bracket).
    let env = "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>f<｜tool▁sep｜>{}<｜tool▁call▁end｜><｜tool▁calls▁end｜>";
    let text = format!("A{env}B{env}C");
    assert_eq!(p().content(&text), "ABC");
}

#[test]
fn id_and_open_markers() {
    // Trait identity + the streaming-filter open marker list.
    assert_eq!(p().id(), "deepseek3");
    assert_eq!(p().open_markers(), &[CALLS_BEGIN]);
}

#[test]
fn unterminated_call_begin_breaks() {
    // calls-begin present, a call-begin present, but the matching call-end is
    // missing — the parse loop hits the `break` (no CALL_END) and yields no calls.
    let text =
        "before<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{\"a\":1}";
    assert!(p().parse(text, "x").is_none());
}

#[test]
fn region_unbounded_when_calls_end_missing() {
    // No CALLS_END at all: `region` is the whole remainder (the `None => region`
    // branch), yet a complete inner call still parses.
    let text = "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>f<｜tool▁sep｜>{\"k\":1}<｜tool▁call▁end｜>trailing";
    let calls = p().parse(text, "y").expect("one call");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0]["function"]["name"], "f");
}

#[test]
fn empty_name_call_is_dropped() {
    // A `<｜tool▁sep｜>` with an empty (whitespace-only) name before it is rejected
    // by parse_one's empty-name guard, so no call is emitted.
    let text =
        "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>   <｜tool▁sep｜>{\"a\":1}<｜tool▁call▁end｜><｜tool▁calls▁end｜>";
    assert!(p().parse(text, "x").is_none());
}

#[test]
fn non_object_or_invalid_args_dropped() {
    // Args that are not a JSON object (a bare array, and outright malformed JSON)
    // both hit parse_one's `_ => return None`.
    let arr = "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>f<｜tool▁sep｜>[1,2,3]<｜tool▁call▁end｜><｜tool▁calls▁end｜>";
    assert!(p().parse(arr, "x").is_none());
    let bad = "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>f<｜tool▁sep｜>{not json}<｜tool▁call▁end｜><｜tool▁calls▁end｜>";
    assert!(p().parse(bad, "x").is_none());
}

// --- R1 dialect (`function<｜tool▁sep｜>NAME` + ```json fence) — what the ---
// --- R1-distill GGUF templates render (fleet golden caught the gap). ------

#[test]
fn r1_dialect_fenced_json_call_parses() {
    let text = "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n{\"city\": \"Paris\"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>";
    let calls = DeepSeek3Parser
        .parse(text, "seed")
        .expect("R1 dialect parses");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0]["function"]["name"], "get_weather");
    assert_eq!(calls[0]["function"]["arguments"], "{\"city\":\"Paris\"}");
}

#[test]
fn r1_dialect_without_fence_is_dropped() {
    // `function<sep>` promises the R1 shape; a missing ```json fence is malformed.
    let text = "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n{\"city\": \"Paris\"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>";
    assert!(DeepSeek3Parser.parse(text, "seed").is_none());
}

#[test]
fn v3_dialect_still_parses_after_r1_support() {
    let text = "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{\"city\": \"Paris\"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>";
    let calls = DeepSeek3Parser
        .parse(text, "seed")
        .expect("V3 dialect parses");
    assert_eq!(calls[0]["function"]["name"], "get_weather");
}

#[test]
fn v3_tool_literally_named_function_needs_the_fence_shape() {
    // A V3 tool genuinely named "function" is ambiguous with the R1 type
    // keyword; the R1 interpretation wins and raw-JSON args are rejected.
    // Documented residual: no real tool is named "function".
    let text = "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>{\"x\": 1}<｜tool▁call▁end｜><｜tool▁calls▁end｜>";
    assert!(DeepSeek3Parser.parse(text, "seed").is_none());
}
