//! Local mirrors of the two async-openai `/v1/chat/completions` response
//! shapes higgs must EXTEND with `reasoning_content` — the crate's structs
//! (0.41) have no such field and no extension hook, so these are the only
//! `/v1` bodies not serialized from async-openai types verbatim.
//!
//! Serde parity is load-bearing: each mirror reproduces the crate struct's
//! field set and skip/null semantics byte-for-byte for the fields higgs sets
//! (crate fields that are BOTH `skip_serializing_if = None` and never set by
//! higgs are omitted entirely — absent either way). The `_tests` prove parity
//! by comparing serialized `Value`s against the crate types.

use async_openai::types::chat::{
    ChatCompletionMessageToolCallChunk, ChatCompletionMessageToolCalls, CompletionUsage,
    FinishReason, FunctionCallStream, Role, ServiceTier,
};
use serde::Serialize;

// ── Non-streaming final response ───────────────────────────────────────────────

/// Mirror of `ChatCompletionResponseMessage` + `reasoning_content`.
/// (Omitted crate fields — `annotations`, `function_call`, `audio` — are
/// skip-if-none and never set by higgs.)
#[derive(Debug, Serialize)]
pub(crate) struct ReasoningResponseMessage {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refusal: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ChatCompletionMessageToolCalls>>,
    pub role: Role,
    /// Model thinking. ABSENT when the model emitted none (skip-if-none, the
    /// llama.cpp-server convention — never `null`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

/// Mirror of `ChatChoice` (omitted: `logprobs`, skip-if-none + never set).
#[derive(Debug, Serialize)]
pub(crate) struct ReasoningChatChoice {
    pub index: u32,
    pub message: ReasoningResponseMessage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
}

/// Mirror of `CreateChatCompletionResponse` (omitted: `service_tier`,
/// `system_fingerprint` — skip-if-none + never set).
#[derive(Debug, Serialize)]
pub(crate) struct ReasoningChatResponse {
    pub id: String,
    pub choices: Vec<ReasoningChatChoice>,
    pub created: u32,
    pub model: String,
    /// Always `"chat.completion"`.
    pub object: String,
    pub usage: Option<CompletionUsage>,
}

// ── Streaming chunks ────────────────────────────────────────────────────────────

/// Mirror of `ChatCompletionStreamResponseDelta` + `reasoning_content`.
/// The crate serializes ALL five base fields (as `null` when `None`) — no
/// skip attributes — so this mirror does too; only the new field is
/// skip-if-none (absent on non-reasoning chunks).
#[derive(Debug, Default, Serialize)]
pub(crate) struct ReasoningStreamDelta {
    pub content: Option<String>,
    pub function_call: Option<FunctionCallStream>,
    pub tool_calls: Option<Vec<ChatCompletionMessageToolCallChunk>>,
    pub role: Option<Role>,
    pub refusal: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

/// Mirror of `ChatChoiceStream` (omitted: `logprobs`, skip-if-none + never set).
#[derive(Debug, Serialize)]
pub(crate) struct ReasoningChoiceStream {
    pub index: u32,
    pub delta: ReasoningStreamDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
}

/// Mirror of `CreateChatCompletionStreamResponse` (all fields — the crate
/// serializes `service_tier`/`system_fingerprint`/`usage` as `null` here,
/// unlike the non-streaming response).
#[derive(Debug, Serialize)]
pub(crate) struct ReasoningStreamChunk {
    pub id: String,
    pub choices: Vec<ReasoningChoiceStream>,
    pub created: u32,
    pub model: String,
    pub service_tier: Option<ServiceTier>,
    pub system_fingerprint: Option<String>,
    /// Always `"chat.completion.chunk"`.
    pub object: String,
    pub usage: Option<CompletionUsage>,
}

#[cfg(test)]
#[path = "v1_wire_tests.rs"]
mod tests;
