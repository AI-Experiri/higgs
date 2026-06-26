
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
