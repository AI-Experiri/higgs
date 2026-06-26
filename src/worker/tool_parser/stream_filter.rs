//! Streaming content filter — suppresses tool-call markup from the streamed
//! `content` deltas so a model's `<tool_call>…` envelope never leaks to the
//! client as assistant text.
//!
//! The structured `tool_calls` are emitted separately, parsed from the full
//! generation at end of stream (see `serve::stream`). This filter only decides,
//! incrementally, which generated text is safe to stream as content NOW. It is
//! engine-agnostic — it operates on `&str` and the open markers a
//! [`ToolCallParser`](super::ToolCallParser) declares.

/// Incrementally filters generated text, emitting only content that is
/// definitely not part of a tool call.
///
/// Holds back a short tail that could still grow into an opening marker (e.g.
/// a piece ending in `"<tool_"`), so a marker split across pieces is never
/// streamed. Once a full opening marker is seen, every subsequent piece is
/// withheld for the rest of the turn.
///
/// KNOWN LIMITATION: this latches `suppressing` permanently at the first open
/// marker — it does not track OPEN..CLOSE spans. So text emitted AFTER or
/// BETWEEN tool calls is dropped from the stream, whereas the non-streaming
/// parser's `content()` preserves it. Streaming and non-streaming `content`
/// therefore diverge for that (uncommon) case; the structured `tool_calls`
/// themselves are identical in both modes (both parsed from the full text).
/// Fixing this requires resuming emission after each close marker.
pub struct ToolCallStreamFilter {
    markers: &'static [&'static str],
    held: String,
    suppressing: bool,
}

impl ToolCallStreamFilter {
    /// Filter for a parser whose tool calls open with any of `markers`.
    ///
    /// INVARIANT: every marker is non-empty. `partial_tail_len` computes
    /// `m.len() - 1`, which underflows on an empty marker. All built-in
    /// [`ToolCallParser::open_markers`](super::ToolCallParser::open_markers) are
    /// non-empty literals, so this holds; the `debug_assert` catches a future
    /// empty marker in tests.
    pub fn new(markers: &'static [&'static str]) -> Self {
        debug_assert!(
            markers.iter().all(|m| !m.is_empty()),
            "tool-call open markers must be non-empty"
        );
        Self {
            markers,
            held: String::new(),
            suppressing: false,
        }
    }

    /// Feed one generated piece; `emit` receives the content safe to stream now.
    pub fn push(&mut self, piece: &str, emit: &mut dyn FnMut(&str)) {
        if self.suppressing {
            return;
        }
        self.held.push_str(piece);

        // A full opening marker present → emit everything before it, then
        // suppress the marker and all that follows for the rest of the turn.
        if let Some(pos) = self.earliest_marker() {
            if pos > 0 {
                let safe = self.held[..pos].to_string();
                emit(&safe);
            }
            self.held.clear();
            self.suppressing = true;
            return;
        }

        // No full marker: emit all but a tail that could still become one.
        let keep = self.partial_tail_len();
        if self.held.len() > keep {
            let cut = self.held.len() - keep;
            let chunk = self.held[..cut].to_string();
            self.held.drain(..cut);
            emit(&chunk);
        }
    }

    /// End of generation: flush any held text that never became a marker.
    pub fn finish(&mut self, emit: &mut dyn FnMut(&str)) {
        if !self.suppressing && !self.held.is_empty() {
            let rest = std::mem::take(&mut self.held);
            emit(&rest);
        }
    }

    /// Earliest byte index where any marker fully occurs in `held`.
    fn earliest_marker(&self) -> Option<usize> {
        self.markers.iter().filter_map(|m| self.held.find(m)).min()
    }

    /// Longest suffix of `held` that is a proper prefix of some marker — held
    /// back in case the remainder of the marker arrives in the next piece.
    fn partial_tail_len(&self) -> usize {
        let mut best = 0;
        let len = self.held.len();
        for m in self.markers {
            let max = (m.len() - 1).min(len);
            for k in (1..=max).rev() {
                if self.held.is_char_boundary(len - k) && m.starts_with(&self.held[len - k..]) {
                    best = best.max(k);
                    break;
                }
            }
        }
        best
    }
}

#[cfg(test)]
#[path = "stream_filter_tests.rs"]
mod tests;
