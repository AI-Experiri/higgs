
use super::*;

/// Byte layout taken verbatim from Ollama's `glm46_test.go` ground truth.
const SINGLE: &str =
        "<think>Let me check.</think><tool_call>get_weather\n<arg_key>location</arg_key>\n<arg_value>Paris</arg_value>\n</tool_call>";

fn p() -> GlmXmlParser {
    GlmXmlParser
}

#[test]
fn handles_glm_template_only() {
    assert!(p().handles("… <arg_key>{{ k }}</arg_key>\n<arg_value>{{ v }}</arg_value> …"));
    // Qwen JSON family also uses <tool_call> but no arg_key/arg_value.
    assert!(!p().handles("<tool_call>\n{\"name\": \"x\", \"arguments\": {}}\n</tool_call>"));
    assert!(!p().handles("<|im_start|>{{ role }} plain chatml, no markers"));
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
fn parses_multiple_args_and_json_value() {
    let text = "<tool_call>search\n<arg_key>query</arg_key>\n<arg_value>rust</arg_value>\n<arg_key>limit</arg_key>\n<arg_value>5</arg_value>\n<arg_key>filters</arg_key>\n<arg_value>{\"lang\":\"en\"}</arg_value>\n</tool_call>";
    let calls = p().parse(text, "z").unwrap();
    let args: Value =
        serde_json::from_str(calls[0]["function"]["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(args["query"], "rust");
    assert_eq!(args["limit"], 5); // numeric, re-parsed as JSON
    assert_eq!(args["filters"]["lang"], "en"); // nested object
}

#[test]
fn parses_two_tool_calls() {
    let text = "<tool_call>a\n<arg_key>k</arg_key>\n<arg_value>1</arg_value>\n</tool_call>between<tool_call>b</tool_call>";
    let calls = p().parse(text, "s").unwrap();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0]["function"]["name"], "a");
    assert_eq!(calls[1]["function"]["name"], "b");
    assert_eq!(calls[1]["id"], "call_s_1");
}

#[test]
fn no_tool_call_returns_none() {
    assert!(p().parse("just a normal answer", "x").is_none());
    assert!(p()
        .parse("<tool_call>x\n<arg_key>k</arg_key>", "x")
        .is_none()); // unterminated
}

#[test]
fn content_strips_think_and_tool_call() {
    assert_eq!(p().content(SINGLE), "");
    // Only the call MARKUP is stripped; text before AND after the call survives.
    let around = "Here's the answer:<tool_call>test</tool_call>done";
    assert_eq!(p().content(around), "Here's the answer:done");
}

#[test]
fn content_strips_forced_open_think_without_opening_tag() {
    // Template forced the think block open → only a trailing </think>.
    let forced = "Let me check the weather.\n</think>\n<tool_call>get_weather\n<arg_key>location</arg_key>\n<arg_value>Paris</arg_value>\n</tool_call>";
    assert_eq!(p().content(forced), "");
}
