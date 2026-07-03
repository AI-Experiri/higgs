use super::*;

const M: &[&str] = &["<tool_call>"];

/// Collect everything the filter emits across the given pieces + finish.
fn run(pieces: &[&str]) -> String {
    let mut f = ToolCallStreamFilter::new(M);
    let mut out = String::new();
    for p in pieces {
        f.push(p, &mut |s| out.push_str(s));
    }
    f.finish(&mut |s| out.push_str(s));
    out
}

#[test]
fn plain_text_passes_through() {
    assert_eq!(run(&["Hello", " ", "world"]), "Hello world");
}

#[test]
fn suppresses_from_marker_onward() {
    assert_eq!(
        run(&[
            "Sure!",
            "<tool_call>",
            "<function=x></function></tool_call>"
        ]),
        "Sure!"
    );
}

#[test]
fn marker_split_across_pieces_never_leaks() {
    // The marker arrives one char at a time; none of it must be emitted.
    let pieces = ["Hi ", "<tool", "_cal", "l>", "junk"];
    assert_eq!(run(&pieces), "Hi ");
}

#[test]
fn partial_marker_that_is_not_a_marker_is_flushed() {
    // "<too" looks like it could start the marker but the turn ends; it must
    // be flushed at finish, not swallowed.
    assert_eq!(run(&["abc", "<too"]), "abc<too");
}

#[test]
fn lone_lt_is_not_held_forever() {
    assert_eq!(run(&["a < b"]), "a < b");
}

#[test]
fn multibyte_content_not_split() {
    // CJK before a marker; bytes must stay char-aligned.
    assert_eq!(run(&["天气", "<tool_call>x"]), "天气");
}

#[test]
fn emits_text_preceding_marker_in_same_piece() {
    // Content and the opening marker arrive in ONE piece: the text before the
    // marker (pos > 0) is emitted, then everything from the marker on is
    // suppressed. Exercises the `if pos > 0 { emit(held[..pos]) }` branch.
    assert_eq!(
        run(&["Sure!<tool_call><function=x></function></tool_call>"]),
        "Sure!"
    );
}

#[test]
#[should_panic(expected = "tool-call open markers must be non-empty")]
fn new_rejects_empty_marker() {
    // The debug_assert guards against an empty marker (which would underflow
    // `partial_tail_len`'s `m.len() - 1`). Constructing with one must panic.
    const EMPTY: &[&str] = &[""];
    ToolCallStreamFilter::new(EMPTY);
}

#[test]
fn suppressed_text_is_buffered_and_takeable() {
    let mut f = ToolCallStreamFilter::new(&["<tool_call>"]);
    let mut out = String::new();
    let mut emit = |s: &str| out.push_str(s);
    f.push("hello <tool_call>{\"x\":1}", &mut emit);
    f.push("</tool_call>", &mut emit);
    f.finish(&mut emit);
    assert_eq!(out, "hello ", "prefix streamed, call withheld");
    assert_eq!(
        f.take_suppressed().as_deref(),
        Some("<tool_call>{\"x\":1}</tool_call>"),
        "withheld text (marker included) is retrievable for the false-positive flush"
    );
    assert!(f.take_suppressed().is_none(), "take drains once");
}

#[test]
fn no_suppression_means_no_suppressed_text() {
    let mut f = ToolCallStreamFilter::new(&["<tool_call>"]);
    let mut out = String::new();
    let mut emit = |s: &str| out.push_str(s);
    f.push("plain text only", &mut emit);
    f.finish(&mut emit);
    assert_eq!(out, "plain text only");
    assert!(f.take_suppressed().is_none());
}
