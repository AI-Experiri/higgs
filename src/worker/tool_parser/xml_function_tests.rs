use super::*;

/// The exact format the Nemotron GGUF template documents.
const NEMOTRON: &str = "<think>\nThe user wants weather.\n</think>\n<tool_call>\n<function=get_weather>\n<parameter=location>\nParis\n</parameter>\n</function>\n</tool_call>\n";

fn p() -> XmlFunctionParser {
    XmlFunctionParser
}

#[test]
fn handles_xml_template_only() {
    assert!(p().handles("… '<function=' ~ name … '<parameter=' ~ k …"));
    assert!(!p().handles("<|im_start|>{{ role }} plain chatml, no markers"));
}

#[test]
fn parses_nemotron_single_call() {
    let calls = p().parse(NEMOTRON, "abc").expect("one call");
    assert_eq!(calls.len(), 1);
    let f = &calls[0]["function"];
    assert_eq!(f["name"], "get_weather");
    let args: Value = serde_json::from_str(f["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(args["location"], "Paris");
    assert_eq!(calls[0]["type"], "function");
    assert_eq!(calls[0]["id"], "call_abc_0");
}

#[test]
fn content_strips_think_and_tool_call() {
    assert_eq!(p().content(NEMOTRON), "");
    let with_preamble = "Sure.\n<tool_call>\n<function=x>\n</function>\n</tool_call>";
    assert_eq!(p().content(with_preamble), "Sure.");
}

#[test]
fn content_strips_forced_open_think_without_opening_tag() {
    // Template forced the think block open → only a trailing </think>.
    let forced = "The user wants weather. Let me do that.\n</think>\n<tool_call>\n<function=get_weather>\n</function>\n</tool_call>";
    assert_eq!(p().content(forced), "");
}

#[test]
fn content_preserves_text_around_and_between_calls() {
    // The content contract: text BEFORE ("A"), BETWEEN ("B"), and AFTER ("C") complete
    // `<tool_call>` spans all survive — content() removes only the call markup, it does
    // not stop at the first call (the regression that bit mistral_bracket).
    let text = "A<tool_call><function=a></function></tool_call>B<tool_call><function=b></function></tool_call>C";
    assert_eq!(p().content(text), "ABC");
}

#[test]
fn parses_multiple_params_and_json_values() {
    let text = "<tool_call>\n<function=search>\n<parameter=query>\nrust\n</parameter>\n<parameter=limit>\n5\n</parameter>\n<parameter=filters>\n{\"lang\":\"en\"}\n</parameter>\n</function>\n</tool_call>";
    let calls = p().parse(text, "z").unwrap();
    let args: Value =
        serde_json::from_str(calls[0]["function"]["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(args["query"], "rust");
    assert_eq!(args["limit"], 5); // numeric, via tojson
    assert_eq!(args["filters"]["lang"], "en"); // nested object
}

#[test]
fn parses_two_tool_calls() {
    let text = "<tool_call>\n<function=a>\n</function>\n</tool_call>\n<tool_call>\n<function=b>\n</function>\n</tool_call>";
    let calls = p().parse(text, "s").unwrap();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0]["function"]["name"], "a");
    assert_eq!(calls[1]["function"]["name"], "b");
    assert_eq!(calls[1]["id"], "call_s_1");
}

#[test]
fn no_tool_call_returns_none() {
    assert!(p().parse("just a normal answer", "x").is_none());
    assert!(p().parse("<tool_call>\n<function=x>", "x").is_none()); // unterminated
}

#[test]
fn id_and_open_markers() {
    assert_eq!(p().id(), "xml-function");
    assert_eq!(p().open_markers(), &["<tool_call>"]);
}

#[test]
fn empty_function_name_is_dropped() {
    // `<function=>` with an empty name → parse_one returns None → no calls.
    let text = "<tool_call><function=></function></tool_call>";
    assert!(p().parse(text, "x").is_none());
}

#[test]
fn missing_parameter_close_breaks_with_empty_args() {
    // A <parameter=…> opens but never closes with </parameter>: the param loop
    // breaks, and the named call is still emitted with no args parsed.
    let text = "<tool_call><function=f><parameter=k>val</function></tool_call>";
    let calls = p().parse(text, "b").expect("one call");
    assert_eq!(calls[0]["function"]["name"], "f");
    assert_eq!(calls[0]["function"]["arguments"], "{}");
}

#[test]
fn parameter_without_value_close_angle_breaks() {
    // `<parameter=k` with no closing `>` hits the `find('>') else break` arm; the
    // named call still emits with empty args.
    let text = "<tool_call><function=f><parameter=k</function></tool_call>";
    let calls = p().parse(text, "c").expect("one call");
    assert_eq!(calls[0]["function"]["name"], "f");
    assert_eq!(calls[0]["function"]["arguments"], "{}");
}
