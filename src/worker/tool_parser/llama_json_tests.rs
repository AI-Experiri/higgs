use super::*;

const LLAMA32_TEMPLATE_SNIFF: &str = r#"Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}."#;

#[test]
fn handles_python_tag_and_llama32_json_instruction() {
    assert!(LlamaJsonParser.handles("… <|python_tag|> …"));
    assert!(LlamaJsonParser.handles(LLAMA32_TEMPLATE_SNIFF));
    assert!(!LlamaJsonParser.handles("<tool_call>{json}</tool_call> qwen style"));
}

#[test]
fn bare_json_call_parses() {
    let text = r#"{"name": "get_weather", "parameters": {"city": "Paris"}}"#;
    let calls = LlamaJsonParser
        .parse(text, "seed")
        .expect("bare JSON parses");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0]["function"]["name"], "get_weather");
    assert_eq!(calls[0]["function"]["arguments"], "{\"city\":\"Paris\"}");
    assert_eq!(calls[0]["id"], "call_seed_0");
}

#[test]
fn python_tag_prefixed_call_parses() {
    let text = "<|python_tag|>{\"name\": \"get_weather\", \"parameters\": {\"city\": \"Paris\"}}";
    let calls = LlamaJsonParser.parse(text, "seed").expect("tagged parses");
    assert_eq!(calls[0]["function"]["name"], "get_weather");
}

#[test]
fn parallel_semicolon_calls_parse_in_order() {
    let text = r#"{"name": "a", "parameters": {}};{"name": "b", "parameters": {"x": 1}}"#;
    let calls = LlamaJsonParser
        .parse(text, "seed")
        .expect("parallel parses");
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0]["function"]["name"], "a");
    assert_eq!(calls[1]["function"]["name"], "b");
    assert_eq!(calls[1]["id"], "call_seed_1");
}

#[test]
fn arguments_key_accepted_for_lenience() {
    let text = r#"{"name": "f", "arguments": {"k": "v"}}"#;
    let calls = LlamaJsonParser
        .parse(text, "seed")
        .expect("arguments key parses");
    assert_eq!(calls[0]["function"]["arguments"], "{\"k\":\"v\"}");
}

#[test]
fn prose_and_non_call_json_return_none() {
    assert!(LlamaJsonParser.parse("The weather is nice.", "s").is_none());
    // JSON but not a call shape (no name/parameters).
    assert!(LlamaJsonParser.parse(r#"{"city": "Paris"}"#, "s").is_none());
    // JSON embedded in prose is not a call turn.
    assert!(LlamaJsonParser
        .parse(r#"Sure: {"name": "f", "parameters": {}}"#, "s")
        .is_none());
}

#[test]
fn reasoning_block_before_call_is_stripped() {
    let text = "<think>which tool?</think>\n{\"name\": \"f\", \"parameters\": {}}";
    let calls = LlamaJsonParser
        .parse(text, "seed")
        .expect("post-think call parses");
    assert_eq!(calls[0]["function"]["name"], "f");
}

#[test]
fn content_is_empty_for_call_turns() {
    assert_eq!(
        LlamaJsonParser.content(r#"{"name": "f", "parameters": {}}"#),
        ""
    );
}

#[test]
fn malformed_json_segment_drops_the_turn() {
    // One bad segment poisons the parse (matching strict single-turn intent).
    assert!(LlamaJsonParser
        .parse(r#"{"name": "a", "parameters": {}};{broken"#, "s")
        .is_none());
}

#[test]
fn semicolon_inside_string_arguments_does_not_split() {
    // A `;` inside a JSON string is data, not a call separator.
    let text = r#"{"name": "search", "parameters": {"query": "a;b"}}"#;
    let calls = LlamaJsonParser
        .parse(text, "seed")
        .expect("semicolon-in-args parses");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0]["function"]["arguments"], "{\"query\":\"a;b\"}");
}

#[test]
fn parallel_calls_with_semicolons_in_args_parse() {
    let text = r#"{"name": "a", "parameters": {"q": "x;y"}};{"name": "b", "parameters": {}}"#;
    let calls = LlamaJsonParser.parse(text, "seed").expect("both parse");
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0]["function"]["arguments"], "{\"q\":\"x;y\"}");
    assert_eq!(calls[1]["function"]["name"], "b");
}

#[test]
fn open_markers_cover_the_bare_json_call_prefix() {
    // The stream filter suppresses on `{"name"` so a bare-JSON streamed call
    // does not leak as content deltas.
    assert!(LlamaJsonParser.open_markers().contains(&"{\"name\""));
}

#[test]
fn trailing_garbage_after_calls_drops_the_turn() {
    // Non-separator text between/after objects means the turn is not a pure
    // call — reject rather than half-parse.
    assert!(LlamaJsonParser
        .parse(r#"{"name": "a", "parameters": {}} and more prose"#, "s")
        .is_none());
}

#[test]
fn adjacent_objects_without_separator_drop_the_turn() {
    // Parallel calls are `;`-separated; butted-together objects are malformed.
    assert!(LlamaJsonParser
        .parse(
            r#"{"name": "a", "parameters": {}}{"name": "b", "parameters": {}}"#,
            "s"
        )
        .is_none());
}
