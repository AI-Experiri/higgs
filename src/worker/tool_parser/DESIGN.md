# tool_parser — Design

## Table of Contents
- [The Trait](#the-trait)
- [Registry Selection by Template Sniff](#registry-selection-by-template-sniff)
- [Primary / Fallback / Raw Pipeline](#primary--fallback--raw-pipeline)
- [Streaming Filter State Machine](#streaming-filter-state-machine)
- [Ordering Rationale](#ordering-rationale)
- [Layering](#layering)

## The Trait

```
trait ToolCallParser : Send + Sync
  │
  ├── id()           -> &'static str        stable name for logs ("xml-function")
  ├── handles(tmpl)  -> bool                does this model's GGUF chat template
  │                                         declare my format markers?
  ├── open_markers() -> &'static [&str]     literal strings that OPEN a call
  │                                         (fed to the stream filter)
  ├── parse(text,    -> Option<Vec<Value>>  full generated text -> OpenAI
  │        id_seed)                         tool_calls, or None (no call)
  └── content(text)  -> String              assistant content with call markup +
                                            leading reasoning block stripped

  Pure text transform. Holds no per-request state.
  One instance serves every request and every engine.
```

## Registry Selection by Template Sniff

```
ToolParserRegistry { parsers: Vec<Box<dyn ToolCallParser>> }

  select(chat_template) ─┐
                         │  parsers.iter().find(|p| p.handles(chat_template))
                         ▼
        ┌────────────────────────────────────────────────┐
        │  GGUF-embedded chat template (a &str)           │
        └────────────────────────────────────────────────┘
                         │  contains which markers?
      ┌──────────────────┼───────────────────┬───────────────────┐
      ▼                  ▼                    ▼                   ▼
  "<function="       "<arg_key>" +        "[TOOL_CALLS]"   "<tool_call>" only
  + "<parameter="    "<arg_value>"            │            (no XML markers)
      │                  │                    │                   │
      ▼                  ▼                    ▼                   ▼
  xml-function        glm-xml           mistral-bracket       qwen-json
   (first match wins; None when no parser recognizes the format)
```

Selection happens **before** generation (the streaming path needs the open
markers up front), the same approach mlx-lm / omlx use.

## Primary / Fallback / Raw Pipeline

```
full generated text (engine decode loop)
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│ PRIMARY:  crate parse_response_oaicompat(full)             │
│           llama.cpp vendored common_chat — covers the      │
│           families derived from the GGUF template          │
│           (Qwen, Mistral, Llama-3, Hermes, …)              │
└───────────────────────────────────────────────────────────┘
        │ Ok(msg_json)                 │ Err (crate declined)
        ▼                              ▼
   content + tool_calls       ┌─────────────────────────────────────┐
   from the crate             │ FALLBACK: registry parser already    │
   (content null on a         │   selected for this model by         │
   pure-tool turn falls       │   template sniff                     │
   back to raw text)          └─────────────────────────────────────┘
                                 │ Some(parser)         │ None
                                 ▼                      ▼
                          parser.parse(full, seed)   RAW: return full
                            │ Some(calls)              text verbatim,
                            │   content()+calls        no tool_calls
                            │ None                     (warn logged)
                            ▼   no call: raw text, no tool_calls
```

Both PRIMARY and FALLBACK read the GGUF template; neither curates per-model.
RAW guarantees text is never silently dropped.

## Streaming Filter State Machine

```
ToolCallStreamFilter::new(open_markers)   { markers, held="", suppressing=false }

push(piece, emit):
  suppressing? ──yes──▶ drop piece (rest of turn withheld)
        │ no
        ▼
   held += piece
        │
        ├── full marker found at pos? ──yes──▶ emit held[..pos]
        │                                       held.clear()
        │                                       suppressing = true
        │ no
        ▼
   keep = longest suffix of held that is a proper prefix of a marker
   emit held[.. len-keep]   (hold back the tail that could still
   drain emitted bytes       grow into a marker; never split a marker
                             across pieces, char-boundary safe)

finish(emit):
  !suppressing && held nonempty ──▶ flush held
  (a "<too" that never became "<tool_call>" is emitted, not swallowed)
```

```
States:                     content deltas streamed:
  PASS  (suppressing=false)    everything safe, minus partial-marker tail
  HOLD  (transient tail)       tail withheld until next piece resolves it
  SUPPRESS (suppressing=true)  nothing — envelope + rest of turn withheld

  PASS ──(full marker seen)──▶ SUPPRESS   (one-way, for the turn)

  Structured tool_calls are NOT produced here — they are parsed from the
  full generation at end of stream (Primary/Fallback pipeline above) and
  emitted by serve::stream as a final delta + finish_reason "tool_calls".
```

## Ordering Rationale

```
with_defaults() — most-specific → most-generic (first handles() match wins):

  1. xml-function     <function=…><parameter=…>   ┐ all share the
  2. glm-xml          <tool_call>+<arg_key/value> ┘ <tool_call> open marker;
  3. deepseek3        <｜tool▁calls▁begin｜>        the discriminating ones
  4. mistral-bracket  [TOOL_CALLS]…[ARGS]          must precede the generic
  5. gemma4           <|tool_call>…<tool_call|>
  6. function-gemma   <start_function_call>…
  7. qwen-json        <tool_call>{json}   ◀── generic catch-all, last

  qwen-json's handles() already excludes <function=/<arg_key> markers,
  so position 7 is belt-and-suspenders, not load-bearing.
```

## Layering

```
        serve / engine boundary (OpenAI tool_calls on the wire)
                          ▲
                          │ Vec<Value> / String
        ┌─────────────────────────────────────────┐
        │  tool_parser  (this module)             │  pure &str -> tool_calls
        │  ToolCallParser / ToolParserRegistry    │  no engine types imported
        │  ToolCallStreamFilter                   │
        └─────────────────────────────────────────┘
                          ▲
            reused unchanged by every engine
                          │
        ┌──────────────┬──────────────┬───────────┐
        │  llama.cpp   │   MLX (fut.) │ CUDA (fut.)│
        └──────────────┴──────────────┴───────────┘
```
