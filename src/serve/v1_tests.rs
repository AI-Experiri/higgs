use super::*;
use crate::worker::engine::CtxLen;
use async_openai::types::chat::{
    CreateChatCompletionRequest, CreateChatCompletionResponse, CreateChatCompletionStreamResponse,
};
use serde_json::json;
use tower::ServiceExt;

use super::super::test_support::*;

fn parse_messages(v: serde_json::Value) -> Vec<ChatCompletionRequestMessage> {
    serde_json::from_value(v).expect("messages deserialize")
}

// ── served_message: verbose serving line wording ────────────────────────

#[test]
fn served_message_format() {
    // The numbers must land in the message text (the log layer captures
    // only `message`), so they're greppable in the Developer Logs.
    assert_eq!(
        served_message("org/model", "length", 12, 1234),
        "higgs: served org/model — 12 tok, finish=length, 1234ms"
    );
    assert_eq!(
        served_message("ollama/llama3:8b", "stop", 0, 7),
        "higgs: served ollama/llama3:8b — 0 tok, finish=stop, 7ms"
    );
}

// ── incoming_message: log-incoming-tokens line wording + cap ─────────────

#[test]
fn incoming_message_format_and_cap() {
    // The flattened prompt content lands in the message text (the log layer
    // captures only `message`), so it's greppable in the Developer Logs.
    let msgs = parse_messages(json!([
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hello there"}
    ]));
    assert_eq!(
        incoming_message("org/model", &msgs),
        "higgs: incoming org/model — 20 chars: be brief hello there"
    );

    // A prompt longer than the cap is truncated to INCOMING_PREVIEW_CHARS
    // with a `…` suffix, while `chars` reports the full pre-cap length so one
    // request can't flood the log ring.
    let big = "x".repeat(INCOMING_PREVIEW_CHARS + 50);
    let msgs = parse_messages(json!([{"role": "user", "content": big}]));
    let line = incoming_message("org/model", &msgs);
    assert!(
        line.starts_with(&format!(
            "higgs: incoming org/model — {} chars: ",
            INCOMING_PREVIEW_CHARS + 50
        )),
        "full char count reported: {line}"
    );
    assert!(line.ends_with('…'), "truncation marker present: {line}");
    let preview = line.rsplit("chars: ").next().unwrap();
    // Capped preview: INCOMING_PREVIEW_CHARS chars + the `…` suffix.
    assert_eq!(
        preview.chars().count(),
        INCOMING_PREVIEW_CHARS + 1,
        "preview capped to INCOMING_PREVIEW_CHARS + ellipsis"
    );
}

// ── redact_paths: client-facing /v1 error sanitization ───────────────────

#[test]
fn redact_paths_strips_paths_and_addresses() {
    // Unix absolute path is redacted; the diagnostic code and prose survive.
    let r = redact_paths("[HG001] model dir unreadable: /Users/me/.lmstudio/models: oops");
    assert!(!r.contains("/Users/me"), "host path leaked: {r}");
    assert!(r.contains("[HG001]"), "code preserved: {r}");
    assert!(r.contains("<redacted>"), "redaction marker present: {r}");

    // host:port bind address is redacted.
    let r = redact_paths("bind failed on 127.0.0.1:8081");
    assert!(!r.contains("127.0.0.1:8081"), "bind address leaked: {r}");

    // Windows drive path is redacted.
    let r = redact_paths(r"engine load failed C:\Users\m\model.gguf bad");
    assert!(!r.contains(r"C:\Users"), "windows path leaked: {r}");
}

#[test]
fn redact_paths_preserves_ids_and_prose() {
    // A model id (no leading slash) and ordinary words are NOT redacted.
    let r = redact_paths("[HG003] model not loaded: org/model — load it first");
    assert_eq!(
        r, "[HG003] model not loaded: org/model — load it first",
        "ids and prose must pass through unchanged"
    );
}

#[test]
fn v1_envelope_redacts_message() {
    // End-to-end: a path-carrying message is sanitized in the built envelope.
    let env = v1_envelope(
        StatusCode::INTERNAL_SERVER_ERROR,
        "[HG004] engine failed to load org/model: /opt/models/x.gguf mmap error",
    );
    assert!(
        !env.error.message.contains("/opt/models"),
        "envelope leaked host path: {}",
        env.error.message
    );
    assert!(env.error.message.contains("[HG004]"));
}

// ── Test 1: /v1/models empty when nothing is loaded ──────────────────────

#[tokio::test]
async fn v1_models_empty_when_unloaded() {
    // Nothing loaded (no resident worker) → empty served set → empty list.
    let app = make_app();
    let resp = app.oneshot(get("/v1/models")).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let list: ListModelResponse = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    assert_eq!(list.object, "list");
    assert!(list.data.is_empty());
}

// ── Test 2: /v1/models lists the loaded model ─────────────────────────────

#[tokio::test]
async fn v1_models_lists_loaded() {
    // Load a fixture model so it becomes a served instance, then list it.
    let dir = tempfile::TempDir::new().unwrap();
    write_gguf_fixture(dir.path(), "org/model");
    let higgs = make_higgs_with_lmstudio(dir.path().to_path_buf());
    higgs.load("org/model", None).await.expect("load");
    let app = app_for(higgs);

    let resp = app.oneshot(get("/v1/models")).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let list: ListModelResponse = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    assert_eq!(list.data.len(), 1);
    assert_eq!(list.data[0].id, "org/model");
    assert_eq!(list.data[0].object, "model");
    assert_eq!(list.data[0].owned_by, "higgs");
}

// ── Test 3: JIT OFF + chat against an unloaded model → 404 HG003 envelope ─

#[tokio::test]
async fn chat_unloaded_404_hg003() {
    // JIT off: an unloaded model is the explicit-load HG003 404 (not a JIT load).
    let app = make_app_jit_off();

    let req = post_json(
        "/v1/chat/completions",
        &json!({"model": "org/missing", "messages": [{"role": "user", "content": "hi"}]}),
    );
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);

    let body = String::from_utf8(body_bytes(resp).await).unwrap();
    assert!(
        body.contains("[HG003]"),
        "body carries the diagnostic code: {body}"
    );
    assert!(
        body.contains("model_not_found"),
        "envelope code present: {body}"
    );
    assert!(
        body.contains("invalid_request_error"),
        "envelope type present: {body}"
    );
}

// ── Serving disabled: /v1 chat → 503 HG019 (control surface stays up) ────
//
// With serving toggled off, a chat request is refused at the boundary with
// HG019 → 503 BEFORE any loaded-model gate or worker RPC. Mirrors the
// unloaded-model boundary test but asserts the serving gate fires first (no
// M_STATUS is sent — the idle supervisor would fail it, but the gate returns
// before reaching it).

#[tokio::test]
async fn chat_serving_disabled_503_hg019() {
    let app = make_app_serving_off();
    let req = post_json(
        "/v1/chat/completions",
        &json!({"model": "org/model", "messages": [{"role": "user", "content": "hi"}]}),
    );
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::SERVICE_UNAVAILABLE,
        "serving off → 503, not 404/500"
    );
    let body = String::from_utf8(body_bytes(resp).await).unwrap();
    assert!(body.contains("[HG019]"), "carries HG019: {body}");
    assert!(
        body.contains("server_error"),
        "503 envelope type is server_error: {body}"
    );
}

// ── Benchmark exclusivity: /v1 chat for a benchmarking model → 503 HG068 ──
//
// A Turbotune benchmark owns its model while it measures candidate configs.
// Even when a candidate is TRANSIENTLY resident, an external /v1 chat for that
// model must be refused ([HG068]) rather than served from the candidate worker
// (which would contaminate the measurement and get torn down mid-stream). The
// gate lives at the TOP of `ensure_loaded`, BEFORE the resident-serve shortcut.
//
// Fail-on-revert: drop the `is_benchmarking` check in `ensure_loaded` and the
// resident model is served (fake worker streams "hello" → 200), failing the 503.
#[tokio::test]
async fn chat_refuses_while_model_is_benchmarking() {
    let dir = tempfile::TempDir::new().unwrap();
    write_gguf_fixture(dir.path(), "org/model");
    let higgs = make_higgs_with_lmstudio(dir.path().to_path_buf());
    // The model is resident (a benchmark candidate is loaded to be measured)…
    higgs.load("org/model", None).await.expect("load candidate");
    // …and marked benchmarking: the guard holds the exclusivity flag for the test.
    let _bench = higgs.begin_benchmark("org/model");

    let app = app_for(higgs.clone());
    let req = post_json(
        "/v1/chat/completions",
        &json!({"model": "org/model", "messages": [{"role": "user", "content": "hi"}]}),
    );
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::SERVICE_UNAVAILABLE,
        "benchmarking model → 503, not served from the transient candidate worker"
    );
    let body = String::from_utf8(body_bytes(resp).await).unwrap();
    assert!(body.contains("[HG068]"), "carries HG068: {body}");
}

/// `POST /api/higgs/worker/stop` REFUSES (503 HG068) while a model is being
/// benchmarked — the drain would evict the bench candidate, so the route must
/// surface the refusal, not report a false success. Fail-on-revert: restore the
/// `let _ = higgs.unload().await` swallow and the route 200s while the model stays.
#[tokio::test]
async fn worker_stop_refuses_while_model_is_benchmarking() {
    let dir = tempfile::TempDir::new().unwrap();
    write_gguf_fixture(dir.path(), "org/model");
    let higgs = make_higgs_with_lmstudio(dir.path().to_path_buf());
    higgs.load("org/model", None).await.expect("load candidate");
    let _bench = higgs.begin_benchmark("org/model");

    let app = app_for(higgs.clone());
    let resp = app
        .oneshot(post_json("/api/higgs/worker/stop", &json!({})))
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::SERVICE_UNAVAILABLE,
        "worker/stop must refuse (503) while a benchmark owns a model"
    );
    let body = String::from_utf8(body_bytes(resp).await).unwrap();
    assert!(body.contains("[HG068]"), "carries HG068: {body}");
}

/// `GET /v1/models` HIDES a model that is being benchmarked — the listing must
/// only advertise what chat can actually reach, and chat refuses a benchmarking
/// model ([HG068]). Fail-on-revert: drop the `is_benchmarking` retain filter and
/// the resident candidate is advertised, so a client discovers a model that 503s.
#[tokio::test]
async fn v1_models_hides_a_benchmarking_model() {
    let dir = tempfile::TempDir::new().unwrap();
    write_gguf_fixture(dir.path(), "org/model");
    let higgs = make_higgs_with_lmstudio(dir.path().to_path_buf());
    higgs.load("org/model", None).await.expect("load candidate");

    // Listed while resident and NOT benchmarking.
    let listed = app_for(higgs.clone())
        .oneshot(get("/v1/models"))
        .await
        .unwrap();
    let listed_body = String::from_utf8(body_bytes(listed).await).unwrap();
    assert!(
        listed_body.contains("org/model"),
        "sanity: resident model is listed when not benchmarking: {listed_body}"
    );

    // Hidden once a benchmark owns it.
    let _bench = higgs.begin_benchmark("org/model");
    let resp = app_for(higgs.clone())
        .oneshot(get("/v1/models"))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = String::from_utf8(body_bytes(resp).await).unwrap();
    assert!(
        !body.contains("org/model"),
        "a benchmarking model must NOT appear in /v1/models: {body}"
    );
}

// ── C1: idle (no worker) — /v1/models 200 empty, /v1/chat 404 ────────────
//
// higgs is spawn-on-load: the normal idle state has NO worker, so
// status().worker_alive is false. /v1/models must still answer 200 with an
// empty list (the correct OpenAI answer for "nothing loaded"), and chat for
// any model must be a 404 HG003 — NOT a 503. The idle supervisor never
// spawns, so its M_STATUS RPC fails and status() reports worker_alive:false
// with loaded:None, no worker I/O involved.

#[tokio::test]
async fn v1_models_idle_no_worker_200_empty() {
    let app = make_app();
    let resp = app.oneshot(get("/v1/models")).await.unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "idle /v1/models is 200, not 503"
    );
    let list: ListModelResponse = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    assert_eq!(list.object, "list");
    assert!(list.data.is_empty(), "idle list is empty");
}

#[tokio::test]
async fn v1_chat_idle_no_worker_404_hg003() {
    // JIT off: idle chat is the explicit-load HG003 404, not a JIT attempt.
    let app = make_app_jit_off();
    let req = post_json(
        "/v1/chat/completions",
        &json!({"model": "org/model", "messages": [{"role": "user", "content": "hi"}]}),
    );
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::NOT_FOUND,
        "idle chat is 404 model-not-loaded, not 503"
    );
    let body = String::from_utf8(body_bytes(resp).await).unwrap();
    assert!(body.contains("[HG003]"), "carries HG003: {body}");
    assert!(body.contains("model_not_found"), "envelope code: {body}");
}

// ── JIT ON + unknown model → 404 HG002 (no load attempt) ─────────────────
//
// JIT is on by default. A chat for an id absent from the scan must NOT
// attempt a load — it is a ModelNotFound 404 [HG002]. The idle supervisor
// never spawns and the default config dirs hold no fixture, so the scan is
// empty and the id is unknown.

#[tokio::test]
async fn v1_chat_jit_on_unknown_model_404_hg002() {
    // Default app → JIT on. Empty temp config so the scan finds nothing.
    let dir = tempfile::TempDir::new().unwrap();
    let app = make_app_with_lmstudio(dir.path().to_path_buf());
    let req = post_json(
        "/v1/chat/completions",
        &json!({"model": "org/unknown", "messages": [{"role": "user", "content": "hi"}]}),
    );
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::NOT_FOUND,
        "JIT + unknown id is 404 model-not-found"
    );
    let body = String::from_utf8(body_bytes(resp).await).unwrap();
    assert!(
        body.contains("[HG002]"),
        "carries HG002 (not HG003): {body}"
    );
    assert!(body.contains("model_not_found"), "envelope code: {body}");
}

// ── JIT ON + scanned-but-unloaded model → loads on demand, then serves ───
//
// JIT is on by default. A chat for a scanned model that isn't currently
// loaded must trigger a load (M_LOAD) and then serve. The mock RPC sequence
// proves the JIT load path is taken: M_STATUS (nothing loaded) → M_LOAD (the
// JIT swap) → M_STATUS (now resident) → M_CHAT. With only-keep-last, after
// the M_LOAD the requested model is the resident one.

#[tokio::test]
async fn v1_chat_jit_on_scanned_loads_then_serves() {
    // JIT on (default). Host-side scan discovers `org/model` but nothing is
    // loaded, so the chat triggers a JIT load (the stateful fake worker spawns
    // and records it), then serves. The model is NOT pre-loaded — reaching 200
    // with the completion proves the JIT load path ran.
    let dir = tempfile::TempDir::new().unwrap();
    write_gguf_fixture(dir.path(), "org/model");
    let app = make_app_with_lmstudio_prepared(dir.path().to_path_buf(), "org/model").await;

    let req = post_json(
        "/v1/chat/completions",
        &json!({"model": "org/model", "messages": [{"role": "user", "content": "hi"}]}),
    );
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "JIT load then serve returns 200"
    );
    let chat: CreateChatCompletionResponse =
        serde_json::from_slice(&body_bytes(resp).await).unwrap();
    assert_eq!(chat.model, "org/model");
    assert_eq!(chat.choices[0].message.content.as_deref(), Some("hello"));
}

// ── Test 4: non-streaming chat returns ChatOutcome.content ───────────────

#[tokio::test]
async fn chat_nonstream_returns_content() {
    // Pre-load a fixture model, then chat it. The stateful fake worker returns
    // `content:"hello", finish:"stop", prompt_tokens:10, completion_tokens:3`.
    let dir = tempfile::TempDir::new().unwrap();
    write_gguf_fixture(dir.path(), "org/model");
    let higgs = make_higgs_with_lmstudio(dir.path().to_path_buf());
    higgs.load("org/model", None).await.expect("load");
    let app = app_for(higgs);

    let req = post_json(
        "/v1/chat/completions",
        &json!({"model": "org/model", "messages": [{"role": "user", "content": "hi"}]}),
    );
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let chat: CreateChatCompletionResponse =
        serde_json::from_slice(&body_bytes(resp).await).unwrap();
    assert_eq!(chat.object, "chat.completion");
    assert_eq!(chat.model, "org/model");
    assert!(chat.id.starts_with("chatcmpl-"));
    assert_eq!(chat.choices[0].message.content.as_deref(), Some("hello"));
    assert_eq!(chat.choices[0].finish_reason, Some(FinishReason::Stop));
    let usage = chat.usage.expect("usage must be present");
    assert_eq!(usage.prompt_tokens, 10);
    assert_eq!(usage.completion_tokens, 3);
    assert_eq!(usage.total_tokens, 13);
}

// ── Test 5: streaming chat SSE framing ───────────────────────────────────

#[tokio::test]
async fn chat_stream_sse_framing() {
    // Pre-load a fixture model, then stream a chat. The stateful fake worker
    // streams two deltas `he`/`llo` then a final `hello`/stop response, so the
    // SSE assembly emits: role + 2 deltas + finish + [DONE].
    let dir = tempfile::TempDir::new().unwrap();
    write_gguf_fixture(dir.path(), "org/model");
    let higgs = make_higgs_with_lmstudio(dir.path().to_path_buf());
    higgs.load("org/model", None).await.expect("load");
    let app = app_for(higgs);

    let req = post_json(
        "/v1/chat/completions",
        &json!({
            "model": "org/model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true,
        }),
    );
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let content_type = resp
        .headers()
        .get("content-type")
        .unwrap()
        .to_str()
        .unwrap();
    assert!(
        content_type.starts_with("text/event-stream"),
        "got: {content_type}"
    );

    let body = String::from_utf8(body_bytes(resp).await).unwrap();
    let datas: Vec<&str> = body
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .collect();
    // The merging delta queue makes the content PARTITION timing-dependent
    // (1 or 2 chunks depending on when assembly drains), so the framing
    // contract is: role first, ≥1 content delta whose concatenation is the
    // full text, then finish, then [DONE].
    assert!(
        (4..=5).contains(&datas.len()),
        "role + content delta(s) + finish + [DONE]: {body}"
    );

    let parse =
        |s: &str| -> CreateChatCompletionStreamResponse { serde_json::from_str(s).unwrap() };
    let role = parse(datas[0]);
    assert_eq!(role.choices[0].delta.role, Some(Role::Assistant));
    assert_eq!(role.object, "chat.completion.chunk");
    let content: String = datas[1..datas.len() - 2]
        .iter()
        .filter_map(|d| parse(d).choices[0].delta.content.clone())
        .collect();
    assert_eq!(content, "hello");
    let finish = parse(datas[datas.len() - 2]);
    assert_eq!(finish.choices[0].finish_reason, Some(FinishReason::Stop));
    assert_eq!(datas[datas.len() - 1], "[DONE]");
}

// ── messages_to_pairs: pure role-flattening ──────────────────────────────
//
// Parse OpenAI request messages from JSON (their canonical wire form) so the
// async-openai untagged content enums deserialize exactly as a real request
// would, then assert the flattened (role, text) pairs.

#[test]
fn messages_to_pairs_role_interleaving_and_multiturn() {
    let msgs = parse_messages(json!([
        {"role": "system", "content": "be terse"},
        {"role": "developer", "content": "dev note"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "answer one"},
        {"role": "user", "content": "second"},
        {"role": "tool", "content": "tool out", "tool_call_id": "call_1"}
    ]));
    let pairs = messages_to_pairs(&msgs).expect("all text → Ok");
    assert_eq!(
        pairs,
        vec![
            ("system".to_owned(), "be terse".to_owned()),
            ("developer".to_owned(), "dev note".to_owned()),
            ("user".to_owned(), "first".to_owned()),
            ("assistant".to_owned(), "answer one".to_owned()),
            ("user".to_owned(), "second".to_owned()),
            ("tool".to_owned(), "tool out".to_owned()),
        ],
        "order and roles preserved across the multi-turn conversation"
    );
}

#[test]
fn messages_to_pairs_text_part_arrays_joined_with_newline() {
    // Every role whose content can be a text-part array: system, developer,
    // user, assistant, tool. Parts join with `\n` (shimmy convention).
    let msgs = parse_messages(json!([
        {"role": "system", "content": [
            {"type": "text", "text": "a"}, {"type": "text", "text": "b"}
        ]},
        {"role": "developer", "content": [
            {"type": "text", "text": "c"}, {"type": "text", "text": "d"}
        ]},
        {"role": "user", "content": [
            {"type": "text", "text": "e"}, {"type": "text", "text": "f"}
        ]},
        {"role": "assistant", "content": [
            {"type": "text", "text": "g"}, {"type": "text", "text": "h"}
        ]},
        {"role": "tool", "tool_call_id": "t1", "content": [
            {"type": "text", "text": "i"}, {"type": "text", "text": "j"}
        ]}
    ]));
    let pairs = messages_to_pairs(&msgs).expect("text parts → Ok");
    assert_eq!(pairs[0], ("system".to_owned(), "a\nb".to_owned()));
    assert_eq!(pairs[1], ("developer".to_owned(), "c\nd".to_owned()));
    assert_eq!(pairs[2], ("user".to_owned(), "e\nf".to_owned()));
    assert_eq!(pairs[3], ("assistant".to_owned(), "g\nh".to_owned()));
    assert_eq!(pairs[4], ("tool".to_owned(), "i\nj".to_owned()));
}

#[test]
fn messages_to_pairs_assistant_none_content_is_empty_string() {
    // An assistant turn with only tool_calls (no content) flattens to "".
    let msgs = parse_messages(json!([
        {"role": "assistant", "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "f", "arguments": "{}"}}
        ]}
    ]));
    let pairs = messages_to_pairs(&msgs).expect("None content → Ok");
    assert_eq!(pairs, vec![("assistant".to_owned(), String::new())]);
}

#[test]
fn messages_to_pairs_rejects_user_image_part() {
    let msgs = parse_messages(json!([
        {"role": "user", "content": [
            {"type": "text", "text": "look"},
            {"type": "image_url", "image_url": {"url": "http://x/y.png"}}
        ]}
    ]));
    let err = messages_to_pairs(&msgs).expect_err("image part → Err");
    assert!(
        err.contains("non-text content part"),
        "rejection message: {err}"
    );
}

#[test]
fn messages_to_pairs_rejects_assistant_refusal_part() {
    let msgs = parse_messages(json!([
        {"role": "assistant", "content": [
            {"type": "refusal", "refusal": "I cannot help with that"}
        ]}
    ]));
    let err = messages_to_pairs(&msgs).expect_err("refusal part → Err");
    assert!(err.contains("refusal content part"), "message: {err}");
}

// ── validate_sampling / check_prompt_fits: pure request validation ────────

fn req(v: serde_json::Value) -> CreateChatCompletionRequest {
    serde_json::from_value(v).expect("request deserialize")
}

#[test]
fn build_sampling_maps_openai_subset_and_omits_absent() {
    // Absent sampling → an all-None set (the model base / engine default stands).
    let bare = build_sampling(&req(json!({
        "model": "m", "messages": [{"role": "user", "content": "hi"}]
    })));
    let s = bare.as_llamacpp();
    assert!(
        s.temperature.is_none()
            && s.top_p.is_none()
            && s.penalty_present.is_none()
            && s.penalty_freq.is_none(),
        "absent OpenAI fields → None: {s:?}"
    );
    // Present fields map across (presence→present, frequency→freq).
    let full = build_sampling(&req(json!({
        "model": "m", "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.3, "top_p": 0.9,
        "presence_penalty": 1.0, "frequency_penalty": -0.5,
    })));
    let s = full.as_llamacpp();
    assert_eq!(s.temperature, Some(0.3));
    assert_eq!(s.top_p, Some(0.9));
    assert_eq!(s.penalty_present, Some(1.0), "presence_penalty → present");
    assert_eq!(s.penalty_freq, Some(-0.5), "frequency_penalty → freq");
    // Fields outside the OpenAI subset are never invented.
    assert!(s.top_k.is_none() && s.min_p.is_none());
}

#[test]
fn validate_sampling_accepts_in_range_and_absent() {
    // All absent → defaults are valid.
    validate_sampling(&req(json!({
        "model": "m", "messages": [{"role": "user", "content": "hi"}]
    })))
    .expect("absent params valid");
    // In-range values valid (temperature 0 and 2, top_p 1, penalties at bounds).
    validate_sampling(&req(json!({
        "model": "m", "messages": [{"role": "user", "content": "hi"}],
        "temperature": 2.0, "top_p": 1.0, "n": 1,
        "presence_penalty": -2.0, "frequency_penalty": 2.0,
        "max_tokens": 256
    })))
    .expect("in-range params valid");
}

#[test]
fn validate_sampling_rejects_each_out_of_range() {
    let base = |extra: serde_json::Value| {
        let mut m = json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]});
        let obj = m.as_object_mut().unwrap();
        for (k, v) in extra.as_object().unwrap() {
            obj.insert(k.clone(), v.clone());
        }
        m
    };
    for (extra, who) in [
        (json!({"temperature": -0.1}), "temperature"),
        (json!({"top_p": 0.0}), "top_p"),
        (json!({"top_p": 1.5}), "top_p"),
        (json!({"n": 0}), "n"),
        // C2: higgs serves a single choice, so n>1 is rejected (not honored
        // as one choice). Both n==0 and n==2 flag the `n` param.
        (json!({"n": 2}), "n"),
        (json!({"presence_penalty": 2.1}), "presence_penalty"),
        (json!({"frequency_penalty": -2.1}), "frequency_penalty"),
        (json!({"max_tokens": 0}), "max_tokens"),
        (json!({"max_tokens": MAX_OUTPUT_TOKENS + 1}), "max_tokens"),
    ] {
        let err = validate_sampling(&req(base(extra))).expect_err("must reject");
        let HiggsError::InvalidSamplingParam { param, .. } = &err else {
            panic!("expected InvalidSamplingParam, got {err}");
        };
        assert_eq!(param, who, "wrong param flagged for {who}");
        assert!(err.to_string().starts_with("[HG013]"));
    }
}

#[test]
fn effective_max_tokens_precedence() {
    // max_completion_tokens wins over max_tokens; default 1024 when absent.
    assert_eq!(
        effective_max_tokens(&req(json!({
            "model": "m", "messages": [], "max_tokens": 10, "max_completion_tokens": 20
        }))),
        20
    );
    assert_eq!(
        effective_max_tokens(&req(json!({
            "model": "m", "messages": [], "max_tokens": 10
        }))),
        10
    );
    assert_eq!(
        effective_max_tokens(&req(json!({"model": "m", "messages": []}))),
        1024
    );
}

#[test]
fn fit_budget_rejects_prompt_that_alone_overflows() {
    // A tiny window with a long prompt: the prompt ALONE exceeds n_ctx → genuine
    // overflow (no room to generate even one token).
    let long = "x".repeat(8000); // ~2000 estimated tokens at 4 bytes/token.
    let err = fit_generation_budget(
        &req(json!({
            "model": "m",
            "messages": [{"role": "user", "content": long}],
            "max_tokens": 16
        })),
        Some(CtxLen::Fixed { n: 128 }),
    )
    .expect_err("a prompt larger than the window overflows");
    assert!(matches!(err, HiggsError::ContextOverflow { .. }));
    assert!(err.to_string().starts_with("[HG005]"));
}

#[test]
fn fit_budget_honors_a_request_that_fits() {
    // Small prompt, large window, modest request → the request is honored as-is.
    let budget = fit_generation_budget(
        &req(json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 16
        })),
        Some(CtxLen::Fixed { n: 4096 }),
    )
    .expect("a fitting request is honored");
    assert_eq!(budget, 16, "a request that fits is honored unchanged");
}

#[test]
fn fit_budget_clamps_oversized_max_tokens_instead_of_rejecting() {
    // The exact case the user hit: prompt fits, but max_tokens + prompt > n_ctx. The
    // OLD behavior rejected (400 [HG005]); the new behavior CLAMPS to what fits so the
    // request proceeds and truncates (finish_reason "length").
    let budget = fit_generation_budget(
        &req(json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 16384
        })),
        Some(CtxLen::Fixed { n: 8192 }),
    )
    .expect("an oversized max_tokens is clamped, not rejected");
    assert!(
        budget > 0 && budget <= 8192,
        "clamped to the space after the prompt (< the requested 16384): {budget}"
    );
}

#[test]
fn fit_budget_infers_full_window_when_max_tokens_omitted() {
    // No max_tokens → infer the remaining window (n_ctx − prompt), NOT the 1024 default.
    let budget = fit_generation_budget(
        &req(json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}]
        })),
        Some(CtxLen::Fixed { n: 8192 }),
    )
    .expect("inferred budget");
    assert!(
        budget > 1024 && budget <= 8192,
        "inferred ~n_ctx, not the flat 1024 default: {budget}"
    );
}

#[test]
fn fit_budget_auto_window_honors_request_capped() {
    // An AUTO/unknown window can't be bounded here → honor the request (worker
    // [HG005] backstops), capped at the absolute MAX_OUTPUT_TOKENS limit.
    let budget = fit_generation_budget(
        &req(json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 500
        })),
        Some(CtxLen::Auto),
    )
    .expect("auto window");
    assert_eq!(budget, 500);
}

#[test]
fn v1_sse_error_carries_context_length_exceeded() {
    // The streaming error path must surface the same code as the non-streaming one.
    let json = v1_sse_error(&HiggsError::ContextOverflow {
        prompt_tokens: 9000,
        max_gen: 0,
        n_ctx: 8192,
    });
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(
        v["error"]["code"], "context_length_exceeded",
        "SSE error envelope carries the OpenAI code: {v}"
    );
}

#[test]
fn v1_error_code_maps_context_overflow() {
    // A context overflow surfaces the OpenAI-standard code (was `null`).
    let overflow = HiggsError::ContextOverflow {
        prompt_tokens: 9000,
        max_gen: 0,
        n_ctx: 8192,
    };
    assert_eq!(v1_error_code(&overflow), Some("context_length_exceeded"));
    // An unrelated error gets no special code.
    assert_eq!(
        v1_error_code(&HiggsError::InvalidSamplingParam {
            param: "top_p".into(),
            detail: "x".into(),
        }),
        None
    );
}

// ── chat_response: pure ChatOutcome → response mapping ────────────────────

#[test]
fn chat_response_without_tool_calls() {
    let out = ChatOutcome {
        content: "hi there".into(),
        finish_reason: "stop".into(),
        tool_calls: None,
        reasoning_content: None,
        prompt_tokens: 10,
        completion_tokens: 7,
    };
    let resp = chat_response("org/model".into(), &out);
    assert_eq!(resp.model, "org/model");
    assert_eq!(resp.object, "chat.completion");
    let choice = &resp.choices[0];
    assert_eq!(choice.message.content.as_deref(), Some("hi there"));
    assert!(choice.message.tool_calls.is_none());
    assert_eq!(choice.finish_reason, Some(FinishReason::Stop));
    assert_eq!(choice.message.role, Role::Assistant);
    let usage = resp.usage.expect("usage present");
    assert_eq!(usage.prompt_tokens, 10);
    assert_eq!(usage.completion_tokens, 7);
    assert_eq!(usage.total_tokens, 17);
}

#[test]
fn chat_response_length_finish_reason_passthrough() {
    let out = ChatOutcome {
        content: "truncated".into(),
        finish_reason: "length".into(),
        tool_calls: None,
        reasoning_content: None,
        prompt_tokens: 3,
        completion_tokens: 1024,
    };
    let resp = chat_response("org/model".into(), &out);
    assert_eq!(
        resp.choices[0].finish_reason,
        Some(FinishReason::Length),
        "engine length signal passes through when no tool calls"
    );
}

#[test]
fn chat_response_with_tool_calls_forces_finish_reason() {
    let out = ChatOutcome {
        content: String::new(),
        // Engine says "stop" but tool_calls must force "tool_calls".
        finish_reason: "stop".into(),
        tool_calls: Some(json!([{
            "id": "call_99",
            "type": "function",
            "function": {"name": "search", "arguments": "{\"q\":\"rust\"}"}
        }])),
        reasoning_content: None,
        prompt_tokens: 20,
        completion_tokens: 4,
    };
    let resp = chat_response("org/model".into(), &out);
    let choice = &resp.choices[0];
    assert_eq!(
        choice.finish_reason,
        Some(FinishReason::ToolCalls),
        "tool_calls overrides the engine stop signal"
    );
    let calls = choice
        .message
        .tool_calls
        .as_ref()
        .expect("tool_calls populated");
    assert_eq!(calls.len(), 1);
    let ChatCompletionMessageToolCalls::Function(call) = &calls[0] else {
        panic!("expected a function tool call, got {:?}", calls[0]);
    };
    assert_eq!(call.id, "call_99");
    assert_eq!(call.function.name, "search");
    assert_eq!(call.function.arguments, "{\"q\":\"rust\"}");
    // OpenAI: a pure tool-call turn (empty content) carries content: null.
    assert!(
        choice.message.content.is_none(),
        "tool-call turn with empty content must emit null content, not empty string"
    );
    let usage = resp.usage.expect("usage present");
    assert_eq!(usage.total_tokens, 24);
}

#[test]
fn chat_response_malformed_tool_calls_degrade_to_none() {
    // tool_calls that don't deserialize into the async-openai wire type must
    // degrade to no tool calls (logged internally) rather than panic; the
    // finish reason then comes from the engine signal.
    let out = ChatOutcome {
        content: "text".into(),
        finish_reason: "stop".into(),
        // A bare string is not a tool_calls array.
        tool_calls: Some(json!("not a tool call array")),
        reasoning_content: None,
        prompt_tokens: 1,
        completion_tokens: 1,
    };
    let resp = chat_response("org/model".into(), &out);
    assert!(
        resp.choices[0].message.tool_calls.is_none(),
        "malformed tool_calls degrade to None"
    );
    assert_eq!(resp.choices[0].finish_reason, Some(FinishReason::Stop));
}

// ── redact_paths: empty-token short circuit (is_sensitive) ────────────────

#[test]
fn redact_paths_ignores_punctuation_only_tokens() {
    // A token that trims to empty after stripping wrapping punctuation is NOT
    // sensitive — `is_sensitive` returns false (the empty-trim guard). Such
    // tokens (a lone `.` / `()` / `","`) pass through verbatim, never redacted.
    let r = redact_paths("done . here ( ) more , words");
    assert_eq!(
        r, "done . here ( ) more , words",
        "punctuation-only tokens are left alone, not redacted"
    );
    assert!(!r.contains("<redacted>"), "no redaction occurred: {r}");
}

// ── messages_to_pairs: function-role message (deprecated arm) ─────────────

#[test]
fn messages_to_pairs_function_role_with_and_without_content() {
    // A `function`-role turn flattens via the Msg::Function arm: `content`
    // present → its text; `content` absent → the empty string (unwrap_or_default).
    let with_content = parse_messages(json!([
        {"role": "function", "name": "get_weather", "content": "sunny"}
    ]));
    let pairs = messages_to_pairs(&with_content).expect("function content → Ok");
    assert_eq!(pairs, vec![("function".to_owned(), "sunny".to_owned())]);

    let no_content = parse_messages(json!([
        {"role": "function", "name": "get_weather", "content": null}
    ]));
    let pairs = messages_to_pairs(&no_content).expect("function null content → Ok");
    assert_eq!(
        pairs,
        vec![("function".to_owned(), String::new())],
        "absent function content degrades to empty string"
    );
}

// ── Facade over a custom NodeRuntime (failing factories) ──────────────────
//
// Mirrors the serve `test_support::node_higgs` builder but lets a test pick the
// backing NodeRuntime (e.g. a load-failing fake) so the `/v1` chat handler's
// JIT-load-failure branch runs end-to-end. The facade's host-side `scan()` reads
// `dirs`, the same roots the runtime scans, so a staged model is discoverable.
fn facade_over(
    node: crate::node::runtime::NodeRuntime,
    dirs: Vec<std::path::PathBuf>,
) -> Arc<Higgs> {
    use crate::api::HiggsConfig;
    let cfg = HiggsConfig {
        lmstudio_dirs: dirs,
        hf_dirs: vec![],
        ollama_dirs: vec![],
        default_load: HiggsConfig::default().default_load,
    };
    Arc::new(Higgs::with_local(Arc::new(node), cfg))
}

// ── JIT ON + scanned model whose load FAILS → mapped error, not 404 ───────
//
// JIT is on. A chat for a scanned model triggers a JIT load; when the load
// itself fails (the load-failing fake worker errors every M_LOAD with HG017),
// `ensure_loaded` surfaces the MAPPED load error (503), NOT a spurious 404.
// Exercises the `if let Err(err) = higgs.load(...)` branch in `ensure_loaded`.

#[tokio::test]
async fn v1_chat_jit_load_failure_surfaces_mapped_error() {
    let dir = tempfile::TempDir::new().unwrap();
    // A valid GGUF (not a dummy) so Prepare can read its metadata; the node still
    // fails the actual LOAD regardless of the file.
    write_gguf_fixture(dir.path(), "org/model");
    let node = crate::node::test_support::fake_runtime_load_fails(vec![dir.path().to_path_buf()]);
    let higgs = facade_over(node, vec![dir.path().to_path_buf()]);
    // Prepare so the readiness gate admits the model; the load then fails (the
    // path under test) rather than being refused as un-prepared.
    higgs
        .tune(crate::tune::TuneRequest {
            id: "org/model".into(),
            mode: None,
            budget: None,
        })
        .await
        .expect("Prepare the test fixture model");
    let app = app_for(higgs);

    let req = post_json(
        "/v1/chat/completions",
        &json!({"model": "org/model", "messages": [{"role": "user", "content": "hi"}]}),
    );
    let resp = app.oneshot(req).await.unwrap();
    // The load failed with HG017 (insufficient memory) → 503, NOT a 404.
    assert_eq!(
        resp.status(),
        StatusCode::SERVICE_UNAVAILABLE,
        "JIT load failure surfaces the mapped error, not a 404"
    );
    let body = String::from_utf8(body_bytes(resp).await).unwrap();
    assert!(
        body.contains("[HG017]"),
        "carries the load failure code: {body}"
    );
    assert!(
        body.contains("server_error"),
        "503 envelope type is server_error: {body}"
    );
}

// ── gate_and_validate Err branches via the live chat handler ──────────────
//
// With JIT on + a scanned fixture, `ensure_loaded` JIT-loads the model, THEN
// the request validation runs. Each of these drives the handler so the gate's
// own Err-return arms run (not just the pure validators tested above):
//   - bad sampling  → validate_sampling Err  → 400 [HG013]
//   - huge prompt   → check_prompt_fits Err   → 400 [HG005]
//   - image part    → messages_to_pairs Err   → 400 (v1_bad_request envelope)

#[tokio::test]
async fn v1_chat_gate_rejects_bad_sampling_400() {
    let dir = tempfile::TempDir::new().unwrap();
    write_gguf_fixture(dir.path(), "org/model");
    let app = make_app_with_lmstudio_prepared(dir.path().to_path_buf(), "org/model").await;

    let req = post_json(
        "/v1/chat/completions",
        &json!({
            "model": "org/model",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": -1.0,
        }),
    );
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::BAD_REQUEST,
        "out-of-range sampling is a 400 after the model is JIT-resolved"
    );
    let body = String::from_utf8(body_bytes(resp).await).unwrap();
    assert!(body.contains("[HG013]"), "carries HG013: {body}");
    assert!(
        body.contains("invalid_request_error"),
        "400 envelope type: {body}"
    );
}

#[tokio::test]
async fn v1_chat_gate_rejects_oversized_prompt_400() {
    let dir = tempfile::TempDir::new().unwrap();
    write_gguf_fixture(dir.path(), "org/model");
    let app = make_app_with_lmstudio_prepared(dir.path().to_path_buf(), "org/model").await;

    // The fixture loads with the GGUF's 4096-token window. A ~40k-char prompt is
    // ~10k estimated tokens, well over the window → ContextOverflow 400.
    let huge = "x".repeat(40_000);
    let req = post_json(
        "/v1/chat/completions",
        &json!({
            "model": "org/model",
            "messages": [{"role": "user", "content": huge}],
        }),
    );
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::BAD_REQUEST,
        "an over-window prompt is rejected at the gate as a 400"
    );
    let body = String::from_utf8(body_bytes(resp).await).unwrap();
    assert!(body.contains("[HG005]"), "carries HG005: {body}");
}

#[tokio::test]
async fn v1_chat_gate_rejects_image_part_400() {
    let dir = tempfile::TempDir::new().unwrap();
    write_gguf_fixture(dir.path(), "org/model");
    let app = make_app_with_lmstudio_prepared(dir.path().to_path_buf(), "org/model").await;

    let req = post_json(
        "/v1/chat/completions",
        &json!({
            "model": "org/model",
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": "http://x/y.png"}}
            ]}],
        }),
    );
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::BAD_REQUEST,
        "a non-text content part is rejected at the gate as a 400"
    );
    let body = String::from_utf8(body_bytes(resp).await).unwrap();
    assert!(
        body.contains("non-text content part"),
        "v1_bad_request carries the rejection detail: {body}"
    );
    assert!(
        body.contains("invalid_request_error"),
        "400 envelope type: {body}"
    );
}

// ── verbose serving line + incoming-prompt line on the live chat path ─────
//
// With both "Verbose Logging" and "Log Incoming Tokens" on, a successful
// non-streaming chat runs `log_incoming` (the incoming-prompt line) before
// dispatch and `log_served` (the completion line) after — the wrappers around
// the pure `incoming_message`/`served_message` builders. Reaching 200 proves
// both wrappers ran on the request path (their content is asserted purely above).

#[tokio::test]
async fn v1_chat_verbose_and_incoming_logging_on_success() {
    let dir = tempfile::TempDir::new().unwrap();
    write_gguf_fixture(dir.path(), "org/model");
    let higgs = make_higgs_with_lmstudio(dir.path().to_path_buf());
    higgs.load("org/model", None).await.expect("load");
    // Turn ON both verbose serving + incoming-prompt logging so the chat handler
    // takes the log_incoming + log_served wrapper branches.
    higgs.set_verbose(true);
    higgs.set_log_incoming_tokens(true);
    let app = app_for(higgs);

    let req = post_json(
        "/v1/chat/completions",
        &json!({
            "model": "org/model",
            "messages": [{"role": "system", "content": "be brief"},
                         {"role": "user", "content": "hi"}],
        }),
    );
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "verbose + incoming logging do not change the response"
    );
    let chat: CreateChatCompletionResponse =
        serde_json::from_slice(&body_bytes(resp).await).unwrap();
    assert_eq!(chat.choices[0].message.content.as_deref(), Some("hello"));
}

// ── verbose streaming path: SSE happy path with verbose on + include_usage ─
//
// The streaming branch reads `higgs.verbose()` (threaded into the SSE assembly)
// and honors `stream_options.include_usage`. With verbose on and include_usage
// requested, the stream still frames correctly and ends in [DONE]; the terminal
// usage chunk is emitted before it.

#[tokio::test]
async fn v1_chat_stream_verbose_with_usage_option() {
    let dir = tempfile::TempDir::new().unwrap();
    write_gguf_fixture(dir.path(), "org/model");
    let higgs = make_higgs_with_lmstudio(dir.path().to_path_buf());
    higgs.load("org/model", None).await.expect("load");
    higgs.set_verbose(true);
    let app = app_for(higgs);

    let req = post_json(
        "/v1/chat/completions",
        &json!({
            "model": "org/model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true,
            "stream_options": {"include_usage": true},
        }),
    );
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = String::from_utf8(body_bytes(resp).await).unwrap();
    let datas: Vec<&str> = body
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .collect();
    // role + 2 deltas + finish + usage + [DONE].
    assert_eq!(
        datas.last().copied(),
        Some("[DONE]"),
        "stream terminates with [DONE]: {body}"
    );
    // include_usage adds a terminal usage chunk carrying token counts.
    assert!(
        datas.iter().any(|d| d.contains("\"usage\"")),
        "include_usage emits a usage chunk: {body}"
    );
}

// ── /v1/models with a fleet attached (empty) exercises the fleet branch ───
//
// With a HubFleet installed (no remote nodes), `v1_models` takes the
// `if let Some(fleet)` arm and awaits `routed_models()` (empty), and chat's
// `ensure_loaded` consults `fleet.is_remote()` (false) — the with-fleet code
// paths run without a live remote route. The list still reflects only local
// served ids (none here).

#[tokio::test]
async fn v1_models_with_empty_fleet_lists_local_only() {
    let higgs = make_higgs();
    let fleet = Arc::new(crate::node::fleet::HubFleet::new(Arc::new(
        crate::log_bus::LogBus::new(),
    )));
    higgs.set_fleet(fleet);
    let app = app_for(higgs);

    let resp = app.oneshot(get("/v1/models")).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let list: ListModelResponse = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    assert_eq!(list.object, "list");
    assert!(
        list.data.is_empty(),
        "an empty fleet adds no models to the local (empty) set"
    );
}

#[tokio::test]
async fn v1_chat_jit_off_with_fleet_consults_is_remote_then_404() {
    // JIT off + an empty fleet: `ensure_loaded` checks `fleet.is_remote()`
    // (false for an unknown id), so it falls through to the explicit-load HG003
    // 404 — proving the with-fleet `is_remote` branch ran on the chat path.
    let higgs = make_higgs();
    higgs.set_jit_enabled(false);
    let fleet = Arc::new(crate::node::fleet::HubFleet::new(Arc::new(
        crate::log_bus::LogBus::new(),
    )));
    higgs.set_fleet(fleet);
    let app = app_for(higgs);

    let req = post_json(
        "/v1/chat/completions",
        &json!({"model": "org/missing", "messages": [{"role": "user", "content": "hi"}]}),
    );
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::NOT_FOUND,
        "unknown id, not remote → HG003 404"
    );
    let body = String::from_utf8(body_bytes(resp).await).unwrap();
    assert!(body.contains("[HG003]"), "carries HG003: {body}");
}

// ── Raw messages passthrough (reasoning echo-back survives) ───────────────────

/// The raw `messages` array must reach the template VERBATIM: assistant
/// `reasoning_content` echo-back (DeepSeek/Kimi multi-turn convention — genai
/// sends it) is not modeled by async-openai's typed structs, so re-serializing
/// the typed request would silently DROP it. Fails on reverting
/// `raw_messages_json` back to a typed re-serialization.
#[test]
fn raw_messages_json_preserves_reasoning_echo_back() {
    let raw = serde_json::json!({
        "model": "m",
        "messages": [
            { "role": "user", "content": "hi" },
            { "role": "assistant", "content": "4", "reasoning_content": "2+2 is 4" },
            { "role": "user", "content": "why?" }
        ]
    });
    let out = raw_messages_json(&raw);
    assert!(
        out.contains("\"reasoning_content\":\"2+2 is 4\""),
        "echo-back field survives to the template: {out}"
    );
    // Sanity: the typed round-trip WOULD drop it (the reason this fn exists).
    let typed: CreateChatCompletionRequest = serde_json::from_value(raw).unwrap();
    let retyped = serde_json::to_string(&typed.messages).unwrap();
    assert!(
        !retyped.contains("reasoning_content"),
        "typed re-serialization drops the field (proves the raw path is load-bearing)"
    );
}

/// BETWEEN candidate loads (benchmark flag set, no worker resident) with JIT OFF,
/// a chat for the benchmarked model must still refuse with 503 [HG068] — not flap
/// to the HG003 404 not-loaded branch depending on benchmark phase (codex r18).
/// Fail-on-revert: gate only on `local_bench_candidate` (resident case) and this
/// returns 404.
#[tokio::test]
async fn chat_refuses_between_candidates_with_jit_off() {
    let higgs = make_higgs();
    higgs.set_jit_enabled(false);
    // Between candidates: benchmarking, but NOTHING resident.
    let _bench = higgs.begin_benchmark("org/model");
    let resp = app_for(higgs)
        .oneshot(post_json(
            "/v1/chat/completions",
            &json!({"model": "org/model", "messages": [{"role": "user", "content": "hi"}]}),
        ))
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::SERVICE_UNAVAILABLE,
        "between-candidates + JIT off → still 503, not a phase-dependent 404"
    );
    let body = String::from_utf8(body_bytes(resp).await).unwrap();
    assert!(body.contains("[HG068]"), "carries HG068: {body}");
}
