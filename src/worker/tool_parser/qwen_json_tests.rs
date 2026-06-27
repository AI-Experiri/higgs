use super::*;

/// The exact byte layout Qwen3 emits: a `<think>` block then a JSON object
/// inside `<tool_call>` tags.
const QWEN3: &str = "<think>\nThe user wants weather.\n</think>\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Paris\"}}\n</tool_call>\n";

fn p() -> QwenJsonParser {
    QwenJsonParser
}

#[test]
fn handles_qwen_template_only() {
    // Plain ChatML JSON template with <tool_call> and no XML markers.
    assert!(p().handles("…{{ '<tool_call>\\n' }}{{ tc | tojson }}{{ '\\n</tool_call>' }}…"));
    // Must NOT steal the XML-function family.
    assert!(!p().handles("… '<tool_call>' … '<function=' ~ name … '<parameter=' ~ k …"));
    // Must NOT steal the GLM <arg_key> dialect.
    assert!(!p().handles("… '<tool_call>' … '<arg_key>' … '<arg_value>' …"));
    // No <tool_call> at all → not this format.
    assert!(!p().handles("<|im_start|>{{ role }} plain chatml, no tool markers"));
}

#[test]
fn parses_qwen3_single_call() {
    let calls = p().parse(QWEN3, "abc").expect("one call");
    assert_eq!(calls.len(), 1);
    let f = &calls[0]["function"];
    assert_eq!(f["name"], "get_weather");
    let args: Value = serde_json::from_str(f["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(args["location"], "Paris");
    assert_eq!(calls[0]["type"], "function");
    assert_eq!(calls[0]["id"], "call_abc_0");
}

#[test]
fn parses_multiple_args_incl_json_typed_value() {
    let text = "<tool_call>\n{\"name\": \"search\", \"arguments\": {\"query\": \"rust\", \"limit\": 5, \"filters\": {\"lang\": \"en\"}}}\n</tool_call>";
    let calls = p().parse(text, "z").unwrap();
    let args: Value =
        serde_json::from_str(calls[0]["function"]["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(args["query"], "rust");
    assert_eq!(args["limit"], 5); // numeric stays typed
    assert_eq!(args["filters"]["lang"], "en"); // nested object stays typed
}

#[test]
fn parses_two_tool_calls() {
    let text = "<tool_call>\n{\"name\": \"a\", \"arguments\": {}}\n</tool_call>\n<tool_call>\n{\"name\": \"b\", \"arguments\": {\"x\": 1}}\n</tool_call>";
    let calls = p().parse(text, "s").unwrap();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0]["function"]["name"], "a");
    assert_eq!(calls[1]["function"]["name"], "b");
    assert_eq!(calls[1]["id"], "call_s_1");
}

#[test]
fn no_tool_call_returns_none() {
    assert!(p().parse("just a normal answer", "x").is_none());
    // Unterminated block — no closing tag.
    assert!(p()
        .parse("<tool_call>\n{\"name\": \"x\", \"arguments\": {}}", "x")
        .is_none());
    // Empty function name is rejected.
    assert!(p()
        .parse(
            "<tool_call>\n{\"name\": \"\", \"arguments\": {}}\n</tool_call>",
            "x"
        )
        .is_none());
}

#[test]
fn content_strips_think_and_tool_call() {
    assert_eq!(p().content(QWEN3), "");
    let with_preamble = "Sure.\n<tool_call>\n{\"name\": \"x\", \"arguments\": {}}\n</tool_call>";
    assert_eq!(p().content(with_preamble), "Sure.");
}

#[test]
fn content_strips_forced_open_think_without_opening_tag() {
    // Template forced the think block open → only a trailing </think>.
    let forced = "Reasoning about the request.\n</think>\n<tool_call>\n{\"name\": \"x\", \"arguments\": {}}\n</tool_call>";
    assert_eq!(p().content(forced), "");
}

#[test]
fn content_preserves_text_after_a_tool_call() {
    // Contract: content is the text with tool-call markup REMOVED — text after
    // (and between) calls must survive, not be dropped at the first marker.
    let text = "Calling.<tool_call>{\"name\":\"f\",\"arguments\":{}}</tool_call>Done.";
    assert_eq!(p().content(text), "Calling.Done.");
}

#[test]
fn id_and_open_markers() {
    assert_eq!(p().id(), "qwen-json");
    assert_eq!(p().open_markers(), &["<tool_call>"]);
}

#[test]
fn content_drops_unterminated_open_marker() {
    // An <tool_call> that never closes: the shared content_outside_calls hits its
    // `None => return out` arm — the preamble survives, the dangling markup is
    // dropped as incomplete (no trailing content leaks).
    let text = "Preamble here.<tool_call>\n{\"name\": \"x\"";
    assert_eq!(p().content(text), "Preamble here.");
}

#[test]
fn parse_drops_call_with_invalid_json_body() {
    // A <tool_call> whose body is not valid JSON is dropped by parse_one (its
    // `serde_json::from_str(...).ok()?`), yielding no calls.
    assert!(p()
        .parse("<tool_call>\nnot json at all\n</tool_call>", "x")
        .is_none());
}

#[test]
fn parse_defaults_missing_arguments_to_empty_object() {
    // A call object with `name` but no `arguments` key defaults to `{}`.
    let calls = p()
        .parse("<tool_call>\n{\"name\": \"ping\"}\n</tool_call>", "d")
        .expect("one call");
    assert_eq!(calls[0]["function"]["name"], "ping");
    assert_eq!(calls[0]["function"]["arguments"], "{}");
}
