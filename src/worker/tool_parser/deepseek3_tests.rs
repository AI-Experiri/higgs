
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
