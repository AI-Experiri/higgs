use serde_json::json;

use super::*;

/// Byte layout matching Ollama's `functiongemma_test.go` ground truth.
const SINGLE: &str =
    "<start_function_call>call:get_weather{city:<escape>Paris<escape>}<end_function_call>";

fn p() -> FunctionGemmaParser {
    FunctionGemmaParser
}

fn args_of(call: &Value) -> Value {
    serde_json::from_str(call["function"]["arguments"].as_str().unwrap()).unwrap()
}

#[test]
fn handles_gemma_template_only() {
    assert!(p().handles("… '<start_function_call>call:' ~ name … '<end_function_call>' …"));
    assert!(!p().handles("<tool_call>\n<function= … no gemma markers"));
}

#[test]
fn parses_single_call() {
    let calls = p().parse(SINGLE, "abc").expect("one call");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0]["function"]["name"], "get_weather");
    assert_eq!(calls[0]["type"], "function");
    assert_eq!(calls[0]["id"], "call_abc_0");
    assert_eq!(args_of(&calls[0])["city"], "Paris");
}

#[test]
fn parses_multiple_args_with_typed_values() {
    // numbers, bools, and a JSON-object value alongside an escaped string.
    let text = "<start_function_call>call:configure{a:1,ratio:0.5,enabled:true,name:<escape>x<escape>,opts:{deep:true}}<end_function_call>";
    let calls = p().parse(text, "z").unwrap();
    let a = args_of(&calls[0]);
    assert_eq!(a["a"], 1);
    assert_eq!(a["ratio"], 0.5);
    assert_eq!(a["enabled"], true);
    assert_eq!(a["name"], "x");
    assert_eq!(a["opts"]["deep"], true); // nested object, top-level comma respected
}

#[test]
fn parses_array_argument() {
    let text = "<start_function_call>call:process{items:[<escape>a<escape>,<escape>b<escape>,<escape>c<escape>]}<end_function_call>";
    let a = args_of(&p().parse(text, "q").unwrap()[0]);
    assert_eq!(a["items"], json!(["a", "b", "c"]));
}

#[test]
fn parses_two_tool_calls() {
    let text = "<start_function_call>call:get_weather{city:<escape>Paris<escape>}<end_function_call><start_function_call>call:get_weather{city:<escape>London<escape>}<end_function_call>";
    let calls = p().parse(text, "s").unwrap();
    assert_eq!(calls.len(), 2);
    assert_eq!(args_of(&calls[0])["city"], "Paris");
    assert_eq!(args_of(&calls[1])["city"], "London");
    assert_eq!(calls[1]["id"], "call_s_1");
}

#[test]
fn no_tool_call_returns_none() {
    assert!(p().parse("Hello, world!", "x").is_none());
    // unterminated: open marker but no close.
    assert!(p().parse("<start_function_call>call:x{", "x").is_none());
}

#[test]
fn content_strips_preamble_and_think() {
    // Content before the first call is reported.
    let t = "Let me check.<start_function_call>call:get_weather{city:<escape>Paris<escape>}<end_function_call>";
    assert_eq!(p().content(t), "Let me check.");
    // Leading reasoning block stripped (forced-open: trailing </think> only).
    let forced = "thinking out loud\n</think>\nLet me check.<start_function_call>call:x{}<end_function_call>";
    assert_eq!(p().content(forced), "Let me check.");
}

// --- Regression tests for the ported bugs ---------------------------------

/// Bug 1: a multibyte UTF-8 value (`日本語`) inside an escaped string must
/// not panic. The old byte-stepping `split_top_level` sliced mid-codepoint.
#[test]
fn parses_multibyte_utf8_value() {
    let text =
        "<start_function_call>call:translate{text:<escape>日本語<escape>}<end_function_call>";
    let calls = p().parse(text, "u").expect("one call");
    assert_eq!(args_of(&calls[0])["text"], "日本語");
}

/// Bug 1 (split path): a top-level comma after a multibyte escaped value
/// still splits correctly and on a char boundary.
#[test]
fn splits_after_multibyte_value() {
    let text = "<start_function_call>call:f{a:<escape>café<escape>,b:1}<end_function_call>";
    let a = args_of(&p().parse(text, "u").unwrap()[0]);
    assert_eq!(a["a"], "café");
    assert_eq!(a["b"], 1);
}

/// JSON-quoted value with an embedded comma stays inside the escaped span —
/// the comma must not split the pair. The literal includes the quotes since
/// only `<escape>` strips, not JSON quoting.
#[test]
fn escaped_value_with_embedded_comma_does_not_split() {
    let text =
        "<start_function_call>call:f{msg:<escape>hello, world<escape>,n:2}<end_function_call>";
    let a = args_of(&p().parse(text, "u").unwrap()[0]);
    assert_eq!(a["msg"], "hello, world");
    assert_eq!(a["n"], 2);
}

/// Bug 2: content AFTER a tool call is reported (Go `content_after_tool_call`).
#[test]
fn content_after_tool_call() {
    let t = "<start_function_call>call:test{}<end_function_call>Done!";
    let calls = p().parse(t, "a").unwrap();
    assert_eq!(calls[0]["function"]["name"], "test");
    assert_eq!(p().content(t), "Done!");
}

/// Bug 2: content BETWEEN two tool calls is reported
/// (Go `content_between_tool_calls`).
#[test]
fn content_between_tool_calls() {
    let t = "<start_function_call>call:first{}<end_function_call>Some text here<start_function_call>call:second{}<end_function_call>";
    let calls = p().parse(t, "b").unwrap();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0]["function"]["name"], "first");
    assert_eq!(calls[1]["function"]["name"], "second");
    assert_eq!(p().content(t), "Some text here");
}

/// Bug 4: an empty function name (`call:{…}`) is skipped, not emitted.
#[test]
fn empty_name_call_is_skipped() {
    assert!(p()
        .parse("<start_function_call>call:{a:1}<end_function_call>", "x")
        .is_none());
}

#[test]
fn id_and_open_markers() {
    assert_eq!(p().id(), "function-gemma");
    assert_eq!(p().open_markers(), &[OPEN]);
}

/// The `call:` prefix may be preceded by whitespace/newlines inside the block —
/// the `or_else(trim_start)` fallback strips it.
#[test]
fn leading_whitespace_before_call_prefix() {
    let text =
        "<start_function_call>\n  call:get_weather{city:<escape>Paris<escape>}<end_function_call>";
    let calls = p().parse(text, "w").expect("one call");
    assert_eq!(calls[0]["function"]["name"], "get_weather");
    assert_eq!(args_of(&calls[0])["city"], "Paris");
}

/// A block whose args do NOT end in `}` (trailing text after the args object)
/// exercises the `rfind('}')` fallback: args are bounded by the LAST brace.
#[test]
fn args_with_trailing_text_after_brace() {
    let text = "<start_function_call>call:f{a:1}trailing<end_function_call>";
    let calls = p().parse(text, "t").expect("one call");
    let a = args_of(&calls[0]);
    assert_eq!(a["a"], 1); // args bounded at the last `}`, trailing ignored
}

/// A block with NO closing brace at all takes the `None => args_body` arm of
/// the suffix fallback (the whole remainder is the args body).
#[test]
fn args_without_closing_brace() {
    let text = "<start_function_call>call:f{a:1<end_function_call>";
    let calls = p().parse(text, "n").expect("one call");
    assert_eq!(args_of(&calls[0])["a"], 1);
}

/// A segment with no `:` is skipped (the `continue` in parse_object).
#[test]
fn segment_without_colon_is_skipped() {
    let text = "<start_function_call>call:f{noColonHere,b:2}<end_function_call>";
    let a = args_of(&p().parse(text, "c").expect("one call")[0]);
    assert!(a.get("noColonHere").is_none());
    assert_eq!(a["b"], 2);
}

/// `false` coerces to a JSON bool (the false arm of parse_value), an integer
/// stays an integer, a float stays a float, and a bare unrecognized token
/// degrades to a string (the final fall-through).
#[test]
fn typed_value_coercions() {
    let text =
        "<start_function_call>call:f{flag:false,n:7,ratio:1.5,bare:hello}<end_function_call>";
    let a = args_of(&p().parse(text, "v").expect("one call")[0]);
    assert_eq!(a["flag"], false);
    assert_eq!(a["n"], 7);
    assert_eq!(a["ratio"], 1.5);
    assert_eq!(a["bare"], "hello"); // bare non-typed token stays a string
}

/// A nested `{…}` object value goes through the object arm of parse_value.
#[test]
fn nested_object_value() {
    let text = "<start_function_call>call:f{opts:{inner:5}}<end_function_call>";
    let a = args_of(&p().parse(text, "o").expect("one call")[0]);
    assert_eq!(a["opts"]["inner"], 5);
}
