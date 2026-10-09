// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::HashMap;
use std::slice;

use llm_multimodal::ImageDetail;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;
use vllm_chat::ChatTokenUsage;
use vllm_engine_core_client::protocol::stats::SpecDecodeMetrics;

// ============================================================================
// Constants
// ============================================================================

// ============================================================================
// Default value helpers
// ============================================================================

/// Helper function for serde default value (returns true).
pub fn default_true() -> bool {
    true
}

/// Deserialize an OpenAI request `top_k` while preserving explicit disable.
///
/// Null remains `None` so model generation defaults apply. Explicit `-1` and
/// `0` become `Some(0)` so the request overrides those defaults and disables
/// top-k sampling. Positive limits are preserved.
pub fn deserialize_request_top_k<'de, D>(deserializer: D) -> Result<Option<u32>, D::Error>
where
    D: Deserializer<'de>,
{
    match Option::<i64>::deserialize(deserializer)? {
        None => Ok(None),
        Some(value) => vllm_text::normalize_top_k(value)
            .map(|value| Some(value.unwrap_or(0)))
            .map_err(serde::de::Error::custom),
    }
}

// ============================================================================
// String/Array Utilities
// ============================================================================

/// A type that can be either a single string or an array of strings.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(untagged)]
pub enum StringOrArray {
    String(String),
    Array(Vec<String>),
}

impl StringOrArray {
    pub fn as_slice(&self) -> &[String] {
        match self {
            StringOrArray::String(s) => slice::from_ref(s),
            StringOrArray::Array(arr) => arr,
        }
    }

    #[allow(unused)]
    pub fn into_vec(self) -> Vec<String> {
        match self {
            StringOrArray::String(s) => vec![s],
            StringOrArray::Array(arr) => arr,
        }
    }
}

fn max_stop_strings() -> usize {
    std::env::var("VLLM_MAX_STOP_STRINGS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(4)
}

/// Validates stop sequences (at most the configured number of non-empty strings).
pub fn validate_stop(stop: &StringOrArray) -> Result<(), validator::ValidationError> {
    let stop = stop.as_slice();
    let max_stop_strings = max_stop_strings();
    if stop.len() > max_stop_strings {
        let mut error = validator::ValidationError::new("too_many_stop_strings");
        error.code = format!("stop strings must contain at most {max_stop_strings} items").into();
        return Err(error);
    }
    if stop.iter().any(|s| s.is_empty()) {
        return Err(validator::ValidationError::new(
            "stop strings cannot be empty",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{StringOrArray, validate_stop};

    #[test]
    fn validate_stop_accepts_four_stop_strings() {
        let stop = StringOrArray::Array(
            ["one", "two", "three", "four"].into_iter().map(str::to_string).collect(),
        );

        validate_stop(&stop).expect("four stop strings should be accepted");
    }

    #[test]
    fn validate_stop_rejects_more_than_four_stop_strings() {
        let stop = StringOrArray::Array(
            ["one", "two", "three", "four", "five"]
                .into_iter()
                .map(str::to_string)
                .collect(),
        );

        let error = validate_stop(&stop).expect_err("five stop strings should be rejected");

        assert_eq!(
            error.code.as_ref(),
            "stop strings must contain at most 4 items"
        );
    }
}

// ============================================================================
// Validation helpers
// ============================================================================

/// Validates top_p: 0.0 < top_p <= 1.0.
pub fn validate_top_p_value(top_p: f32) -> Result<(), validator::ValidationError> {
    if !(top_p > 0.0 && top_p <= 1.0) {
        return Err(validator::ValidationError::new(
            "top_p must be in (0, 1] - greater than 0.0 and at most 1.0",
        ));
    }
    Ok(())
}

// ============================================================================
// Content Parts (for multimodal messages)
// ============================================================================

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type")]
pub enum ContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl {
        image_url: ImageUrl,
        #[serde(skip_serializing_if = "Option::is_none")]
        uuid: Option<String>,
    },
    #[serde(rename = "video_url")]
    VideoUrl {
        video_url: VideoUrl,
        #[serde(skip_serializing_if = "Option::is_none")]
        uuid: Option<String>,
    },
    #[serde(rename = "audio_url")]
    AudioUrl {
        audio_url: AudioUrl,
        #[serde(skip_serializing_if = "Option::is_none")]
        uuid: Option<String>,
    },
    #[serde(rename = "input_audio")]
    InputAudio {
        input_audio: InputAudio,
        #[serde(skip_serializing_if = "Option::is_none")]
        uuid: Option<String>,
    },
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct ImageUrl {
    pub url: String,
    pub detail: Option<ImageDetail>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct VideoUrl {
    pub url: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct AudioUrl {
    pub url: String,
}

/// Base64-encoded audio bytes in OpenAI `input_audio` form.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct InputAudio {
    pub data: String,
    pub format: Option<String>,
}

// ============================================================================
// Streaming
// ============================================================================

#[derive(Debug, Serialize)]
pub(crate) struct StreamResponseEnvelope {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
}

impl StreamResponseEnvelope {
    pub(crate) fn new(id: String, object: &'static str, created: u64, model: String) -> Self {
        Self {
            id,
            object,
            created,
            model,
        }
    }

    pub(crate) fn model(&self) -> &str {
        &self.model
    }
}

/// Mirrors the Python vLLM `StreamOptions` class.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct StreamOptions {
    pub include_usage: Option<bool>,
    pub continuous_usage_stats: Option<bool>,
}

// ============================================================================
// Tools and Function Calling
// ============================================================================

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Tool {
    #[serde(rename = "type")]
    #[serde(default = "default_function_tool_type")]
    pub tool_type: String,
    pub function: Function,
}

fn default_function_tool_type() -> String {
    "function".to_string()
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Function {
    pub name: String,
    pub description: Option<String>,
    #[serde(default)]
    pub parameters: Value,
    /// Whether to enable strict schema adherence (OpenAI structured outputs).
    pub strict: Option<bool>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: FunctionCallResponse,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FunctionCallResponse {
    pub name: String,
    #[serde(default)]
    pub arguments: Option<String>,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ToolCallDelta {
    pub index: u32,
    pub id: Option<String>,
    #[serde(rename = "type")]
    pub tool_type: Option<String>,
    pub function: Option<FunctionCallDelta>,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FunctionCallDelta {
    pub name: Option<String>,
    pub arguments: Option<String>,
}

/// Tool choice value for simple string options.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoiceValue {
    Auto,
    Required,
    None,
}

/// Tool choice for the Chat Completion API.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum ToolChoice {
    Value(ToolChoiceValue),
    Function {
        #[serde(rename = "type")]
        tool_type: String,
        function: FunctionChoice,
    },
    AllowedTools {
        #[serde(rename = "type")]
        tool_type: String,
        mode: String,
        tools: Vec<ToolReference>,
    },
}

impl Default for ToolChoice {
    fn default() -> Self {
        Self::Value(ToolChoiceValue::Auto)
    }
}

/// Function choice specification for `ToolChoice::Function`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FunctionChoice {
    pub name: String,
}

/// Tool reference for `ToolChoice::AllowedTools`.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum ToolReference {
    #[serde(rename = "function")]
    Function { name: String },
    #[serde(rename = "mcp")]
    Mcp {
        server_label: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    #[serde(rename = "file_search")]
    FileSearch,
    #[serde(rename = "web_search_preview")]
    WebSearchPreview,
    #[serde(rename = "computer_use_preview")]
    ComputerUsePreview,
    #[serde(rename = "code_interpreter")]
    CodeInterpreter,
    #[serde(rename = "image_generation")]
    ImageGeneration,
}

impl ToolReference {
    /// Get a unique identifier for this tool reference.
    pub fn identifier(&self) -> String {
        match self {
            ToolReference::Function { name } => format!("function:{name}"),
            ToolReference::Mcp {
                server_label,
                name: Some(n),
            } => format!("mcp:{server_label}:{n}"),
            ToolReference::Mcp {
                server_label,
                name: _,
            } => format!("mcp:{server_label}"),
            ToolReference::FileSearch => "file_search".to_string(),
            ToolReference::WebSearchPreview => "web_search_preview".to_string(),
            ToolReference::ComputerUsePreview => "computer_use_preview".to_string(),
            ToolReference::CodeInterpreter => "code_interpreter".to_string(),
            ToolReference::ImageGeneration => "image_generation".to_string(),
        }
    }
}

// ============================================================================
// Chat Messages
// ============================================================================

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "role")]
pub enum ChatMessage {
    #[serde(rename = "system")]
    System {
        content: MessageContent,
        name: Option<String>,
    },
    #[serde(rename = "user")]
    User {
        content: MessageContent,
        name: Option<String>,
    },
    #[serde(rename = "assistant")]
    Assistant {
        content: Option<MessageContent>,
        name: Option<String>,
        tool_calls: Option<Vec<ToolCall>>,
        /// Reasoning content for reasoning-capable models.
        #[serde(alias = "reasoning_content")]
        #[serde(alias = "thinking")]
        reasoning: Option<String>,
    },
    #[serde(rename = "tool")]
    Tool {
        content: MessageContent,
        tool_call_id: String,
    },
    #[serde(rename = "function")]
    Function { content: String, name: String },
    #[serde(rename = "developer")]
    Developer {
        content: MessageContent,
        tools: Option<Vec<Tool>>,
        name: Option<String>,
    },
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

// ============================================================================
// Usage and Logging
// ============================================================================

/// Mirrors the Python vLLM `UsageInfo` class.
///
/// Do not skip serializing `None` fields here: non-streaming response types
/// should serialize `None` as explicit `null`.
#[derive(Debug, Clone, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub total_tokens: usize,
    pub completion_tokens: Option<usize>,
    pub prompt_tokens_details: Option<PromptTokenUsageInfo>,
    /// Reasoning-token breakdown.
    /// Always present, 0 when the configured parser has no reasoning channel.
    pub completion_tokens_details: CompletionTokenUsageInfo,
}

impl Usage {
    /// Create a Usage with prompt-token cache and reasoning-token details.
    pub fn from_counts(
        prompt_tokens: usize,
        completion_tokens: usize,
        cached_tokens: Option<usize>,
        reasoning_tokens: usize,
    ) -> Self {
        Self {
            prompt_tokens,
            total_tokens: prompt_tokens + completion_tokens,
            completion_tokens: Some(completion_tokens),
            prompt_tokens_details: cached_tokens
                .filter(|&c| c > 0)
                .map(|c| PromptTokenUsageInfo { cached_tokens: c }),
            completion_tokens_details: CompletionTokenUsageInfo { reasoning_tokens },
        }
    }

    pub fn from_token_usage(
        usage: impl Into<ChatTokenUsage>,
        enable_prompt_tokens_details: bool,
    ) -> Self {
        let usage = usage.into();
        Self::from_counts(
            usage.prompt_token_count,
            usage.output_token_count,
            enable_prompt_tokens_details.then_some(usage.cached_token_count),
            usage.reasoning_tokens,
        )
    }
}

/// Mirrors the Python vLLM `PromptTokenUsageInfo` class.
#[derive(Debug, Clone, Serialize)]
pub struct PromptTokenUsageInfo {
    pub cached_tokens: usize,
}

/// Mirrors the Python vLLM `CompletionTokenUsageInfo` class.
#[derive(Debug, Clone, Default, Serialize)]
pub struct CompletionTokenUsageInfo {
    pub reasoning_tokens: usize,
}

// ============================================================================
// Per-request metrics
// ============================================================================

/// Per-request speculative-decoding acceptance metrics.
///
/// Mirrors the Python `SpeculativeDecodingMetrics` response model and the
/// `RequestSpecDecodeMetrics.to_dict` conversion that feeds it: the acceptance
/// histogram is dense, indexed by the number of accepted draft tokens `j`, and
/// `mean_acceptance_length` includes the always-accepted bonus token.
///
/// Original Python definitions: `vllm/entrypoints/generate/base/protocol.py`
/// and `vllm/v1/metrics/stats.py`.
///
/// Do not skip serializing `None` fields here: non-streaming response types
/// should serialize `None` as explicit `null`, matching the Python frontend.
#[derive(Debug, Clone, Serialize)]
pub struct SpeculativeDecodingMetrics {
    pub mean_acceptance_length: f64,
    pub draft_acceptance_rate: f64,
    pub acceptance_histogram: Vec<u64>,
    pub num_spec_steps: u64,
    pub num_accepted_draft_tokens: u64,
    pub num_draft_tokens: u64,
    pub num_spec_tokens: u32,
    /// Ordered per-verify-step arrays, populated at the `detailed` level only.
    pub per_step_accepted: Option<Vec<u64>>,
    pub per_step_drafted: Option<Vec<u64>>,
}

impl SpeculativeDecodingMetrics {
    /// Derive the response payload from the engine-reported wire metrics.
    pub fn from_wire(wire: &SpecDecodeMetrics) -> Self {
        let num_spec_steps: u64 = wire.histogram.iter().sum();
        let num_accepted: u64 = wire
            .histogram
            .iter()
            .enumerate()
            .map(|(accepted, count)| accepted as u64 * count)
            .sum();
        let mean_acceptance_length = if num_spec_steps > 0 {
            1.0 + num_accepted as f64 / num_spec_steps as f64
        } else {
            1.0
        };
        let draft_acceptance_rate = if wire.num_draft_tokens > 0 {
            num_accepted as f64 / wire.num_draft_tokens as f64
        } else {
            0.0
        };

        Self {
            mean_acceptance_length,
            draft_acceptance_rate,
            acceptance_histogram: wire.histogram.clone(),
            num_spec_steps,
            num_accepted_draft_tokens: num_accepted,
            num_draft_tokens: wire.num_draft_tokens,
            num_spec_tokens: wire.num_spec_tokens,
            per_step_accepted: (!wire.per_step_accepted.is_empty())
                .then(|| wire.per_step_accepted.clone()),
            per_step_drafted: (!wire.per_step_drafted.is_empty())
                .then(|| wire.per_step_drafted.clone()),
        }
    }
}

/// Per-request timing and speculative-decoding metrics.
///
/// Mirrors the Python `PerRequestMetrics` response model. The Rust frontend
/// does not compute per-request timings yet, so those fields serialize as
/// `null`; the Python frontend emits the same `null`s whenever per-request
/// timing metrics are unavailable. The speculative-decoding block is populated
/// only when the engine reported it.
///
/// Original Python definition: `vllm/entrypoints/generate/base/protocol.py`.
#[derive(Debug, Clone, Default, Serialize)]
pub struct PerRequestMetrics {
    pub time_to_first_token_ms: Option<f64>,
    pub generation_time_ms: Option<f64>,
    pub queue_time_ms: Option<f64>,
    pub mean_itl_ms: Option<f64>,
    pub tokens_per_second: Option<f64>,
    pub speculative_decoding: Option<SpeculativeDecodingMetrics>,
}

impl PerRequestMetrics {
    /// Build the response `metrics` block from one request's engine-reported
    /// spec-decode metrics, or `None` when the engine reported none.
    pub fn from_spec_decode_metrics(wire: Option<&SpecDecodeMetrics>) -> Option<Self> {
        wire.map(|wire| Self {
            speculative_decoding: Some(SpeculativeDecodingMetrics::from_wire(wire)),
            ..Default::default()
        })
    }

    /// Serialize this block for a streaming chunk, dropping `null` fields the
    /// way the Python frontend's streaming responses do
    /// (`model_dump_json(exclude_none=True)`). Non-streaming responses keep the
    /// explicit `null`s instead.
    pub fn to_stream_json(&self) -> serde_json::Value {
        let mut value = serde_json::to_value(self).expect("per-request metrics must serialize");
        if let Some(object) = value.as_object_mut() {
            object.retain(|_, field| !field.is_null());
        }
        if let Some(spec) = value.get_mut("speculative_decoding")
            && let Some(object) = spec.as_object_mut()
        {
            object.retain(|_, field| !field.is_null());
        }
        value
    }
}

/// Serialize an optional `metrics` block for a streaming chunk, omitting it
/// entirely when absent and dropping its `null` fields when present.
pub fn serialize_stream_metrics<S>(
    metrics: &Option<PerRequestMetrics>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    match metrics {
        None => serializer.serialize_none(),
        Some(metrics) => metrics.to_stream_json().serialize(serializer),
    }
}

#[cfg(test)]
mod usage_tests {
    use vllm_chat::ChatTokenUsage;
    use vllm_llm::TokenUsage;

    use super::Usage;

    #[test]
    fn token_usage_hides_prompt_token_details_by_default() {
        let usage = Usage::from_token_usage(
            TokenUsage {
                prompt_token_count: 5,
                output_token_count: 2,
                cached_token_count: 3,
            },
            false,
        );

        assert_eq!(usage.prompt_tokens, 5);
        assert_eq!(usage.completion_tokens, Some(2));
        assert!(usage.prompt_tokens_details.is_none());
    }

    #[test]
    fn token_usage_includes_prompt_token_details_when_enabled() {
        let usage = Usage::from_token_usage(
            TokenUsage {
                prompt_token_count: 5,
                output_token_count: 2,
                cached_token_count: 3,
            },
            true,
        );

        assert_eq!(
            usage.prompt_tokens_details.as_ref().map(|details| details.cached_tokens),
            Some(3)
        );
    }

    #[test]
    fn token_usage_includes_reasoning_token_details() {
        let usage = Usage::from_token_usage(
            ChatTokenUsage {
                engine: TokenUsage {
                    prompt_token_count: 5,
                    output_token_count: 4,
                    cached_token_count: 0,
                },
                reasoning_tokens: 3,
            },
            false,
        );

        assert_eq!(usage.completion_tokens_details.reasoning_tokens, 3);
        let json = serde_json::to_value(&usage).expect("usage serializes");
        assert_eq!(json["completion_tokens_details"]["reasoning_tokens"], 3);
    }

    #[test]
    fn token_usage_serializes_zero_reasoning_tokens() {
        let usage = Usage::from_token_usage(
            ChatTokenUsage {
                engine: TokenUsage {
                    prompt_token_count: 5,
                    output_token_count: 0,
                    cached_token_count: 0,
                },
                reasoning_tokens: 0,
            },
            false,
        );

        let json = serde_json::to_value(&usage).expect("usage serializes");
        assert_eq!(json["completion_tokens_details"]["reasoning_tokens"], 0);
    }

    #[test]
    fn token_usage_reports_zero_reasoning_tokens_without_reasoning_parser() {
        let usage = Usage::from_token_usage(
            TokenUsage {
                prompt_token_count: 5,
                output_token_count: 2,
                cached_token_count: 0,
            },
            false,
        );

        assert_eq!(usage.completion_tokens_details.reasoning_tokens, 0);
        let json = serde_json::to_value(&usage).expect("usage serializes");
        assert_eq!(json["completion_tokens_details"]["reasoning_tokens"], 0);
    }
}

/// OpenAI completions-style logprobs.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LogProbs {
    pub tokens: Vec<String>,
    pub token_logprobs: Vec<Option<f32>>,
    pub top_logprobs: Vec<Option<HashMap<String, f32>>>,
    pub text_offset: Vec<u32>,
}

/// vLLM prompt-logprob metadata keyed by vocabulary token ID.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PromptLogprob {
    pub logprob: f32,
    pub rank: u32,
    pub decoded_token: String,
}

pub type PromptLogprobs = Vec<Option<HashMap<u32, PromptLogprob>>>;

/// Mirrors the Python vLLM `ChatCompletionLogProbs` class.
#[derive(Debug, Clone, Serialize)]
pub struct ChatLogProbs {
    pub content: Option<Vec<ChatLogProbsContent>>,
}

/// Mirrors the Python vLLM `ChatCompletionLogProbsContent` class.
#[derive(Debug, Clone, Serialize)]
pub struct ChatLogProbsContent {
    pub token: String,
    pub logprob: f32,
    pub bytes: Option<Vec<u8>>,
    pub top_logprobs: Vec<TopLogProb>,
}

/// Mirrors the Python vLLM `ChatCompletionLogProb` class.
#[derive(Debug, Clone, Serialize)]
pub struct TopLogProb {
    pub token: String,
    pub logprob: f32,
    pub bytes: Option<Vec<u8>>,
}

// ============================================================================
// Error Types
// ============================================================================

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ErrorDetail {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    pub param: Option<String>,
    pub code: Option<String>,
}

// ============================================================================
// Model types
// ============================================================================

/// A single model entry in the `/v1/models` response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelObject {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub owned_by: String,
    /// Backend model path (base cards) or adapter path (LoRA cards).
    pub root: Option<String>,
    /// Base model a LoRA adapter derives from; `null` for base models.
    pub parent: Option<String>,
    /// Maximum context length; `null` for LoRA adapter cards.
    pub max_model_len: Option<u32>,
}

/// Response body for `GET /v1/models`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListModelsResponse {
    pub object: String,
    pub data: Vec<ModelObject>,
}

// ============================================================================
// Shared validation helpers
// ============================================================================

/// Validates a messages array is non-empty and has valid user-message content.
///
/// Used by both `POST /v1/chat/completions` and `POST /tokenize` (chat form)
/// so validation behaviour stays in lockstep.
pub(crate) fn validate_messages(
    messages: &[ChatMessage],
) -> Result<(), validator::ValidationError> {
    if messages.is_empty() {
        return Err(validator::ValidationError::new("messages cannot be empty"));
    }

    for msg in messages {
        if let ChatMessage::User { content, .. } = msg {
            match content {
                MessageContent::Text(text) if text.is_empty() => {
                    return Err(validator::ValidationError::new(
                        "message content cannot be empty",
                    ));
                }
                MessageContent::Parts(parts) if parts.is_empty() => {
                    return Err(validator::ValidationError::new(
                        "message content parts cannot be empty",
                    ));
                }
                _ => {}
            }
        }
    }
    Ok(())
}

// ============================================================================
// Normalizable trait
// ============================================================================

/// Trait for request types that need post-deserialization normalization.
pub trait Normalizable {
    /// Normalize the request by applying defaults and transformations.
    fn normalize(&mut self) {
        // Default: no-op
    }
}

#[cfg(test)]
mod metrics_tests {
    use vllm_engine_core_client::protocol::stats::SpecDecodeMetrics;

    use super::{PerRequestMetrics, SpeculativeDecodingMetrics};

    /// Reference values measured from the Python frontend on the same MTP route
    /// (`per_step_accepted=[1,3,1,4,3]`, `per_step_drafted=[4,4,4,4,4]`,
    /// `mean_acceptance_length=3.4`, `draft_acceptance_rate=0.6`).
    fn detailed_wire_metrics() -> SpecDecodeMetrics {
        SpecDecodeMetrics {
            num_spec_tokens: 4,
            histogram: vec![0, 2, 0, 2, 1],
            num_draft_tokens: 20,
            per_step_accepted: vec![1, 3, 1, 4, 3],
            per_step_drafted: vec![4, 4, 4, 4, 4],
        }
    }

    #[test]
    fn spec_decode_metrics_match_python_to_dict() {
        let metrics = SpeculativeDecodingMetrics::from_wire(&detailed_wire_metrics());

        assert_eq!(metrics.num_spec_steps, 5);
        assert_eq!(metrics.num_accepted_draft_tokens, 12);
        assert_eq!(metrics.num_draft_tokens, 20);
        assert_eq!(metrics.num_spec_tokens, 4);
        assert_eq!(metrics.acceptance_histogram, vec![0, 2, 0, 2, 1]);
        assert!((metrics.mean_acceptance_length - 3.4).abs() < 1e-12);
        assert!((metrics.draft_acceptance_rate - 0.6).abs() < 1e-12);
        assert_eq!(metrics.per_step_accepted, Some(vec![1, 3, 1, 4, 3]));
        assert_eq!(metrics.per_step_drafted, Some(vec![4, 4, 4, 4, 4]));
    }

    #[test]
    fn summary_level_omits_per_step_arrays() {
        let wire = SpecDecodeMetrics {
            per_step_accepted: Vec::new(),
            per_step_drafted: Vec::new(),
            ..detailed_wire_metrics()
        };

        let metrics = SpeculativeDecodingMetrics::from_wire(&wire);

        assert!(metrics.per_step_accepted.is_none());
        assert!(metrics.per_step_drafted.is_none());
    }

    #[test]
    fn empty_histogram_reports_neutral_acceptance() {
        let wire = SpecDecodeMetrics {
            num_spec_tokens: 4,
            histogram: Vec::new(),
            num_draft_tokens: 0,
            ..Default::default()
        };

        let metrics = SpeculativeDecodingMetrics::from_wire(&wire);

        assert_eq!(metrics.num_spec_steps, 0);
        assert_eq!(metrics.num_accepted_draft_tokens, 0);
        assert_eq!(metrics.mean_acceptance_length, 1.0);
        assert_eq!(metrics.draft_acceptance_rate, 0.0);
    }

    #[test]
    fn absent_wire_metrics_produce_no_block() {
        assert!(PerRequestMetrics::from_spec_decode_metrics(None).is_none());
    }

    #[test]
    fn response_block_serializes_null_timings_and_spec_block() {
        let block = PerRequestMetrics::from_spec_decode_metrics(Some(&detailed_wire_metrics()))
            .expect("spec metrics should produce a block");
        let json = serde_json::to_value(&block).expect("metrics must serialize");

        assert!(json["time_to_first_token_ms"].is_null());
        assert!(json["generation_time_ms"].is_null());
        assert!(json["queue_time_ms"].is_null());
        assert!(json["mean_itl_ms"].is_null());
        assert!(json["tokens_per_second"].is_null());
        assert_eq!(json["speculative_decoding"]["num_spec_steps"], 5);
        assert_eq!(
            json["speculative_decoding"]["mean_acceptance_length"],
            serde_json::json!(3.4)
        );
    }

    /// Streaming chunks drop `null` fields, like Python's
    /// `model_dump_json(exclude_none=True)`; non-streaming responses keep them.
    #[test]
    fn stream_form_drops_null_fields_only() {
        let block = PerRequestMetrics::from_spec_decode_metrics(Some(&detailed_wire_metrics()))
            .expect("spec metrics should produce a block");
        let json = block.to_stream_json();
        let object = json.as_object().expect("stream metrics are an object");

        assert_eq!(object.len(), 1);
        assert!(object.contains_key("speculative_decoding"));
        assert!(!object.contains_key("time_to_first_token_ms"));
        assert_eq!(json["speculative_decoding"]["num_draft_tokens"], 20);

        let summary = SpecDecodeMetrics {
            per_step_accepted: Vec::new(),
            per_step_drafted: Vec::new(),
            ..detailed_wire_metrics()
        };
        let block = PerRequestMetrics::from_spec_decode_metrics(Some(&summary))
            .expect("spec metrics should produce a block");
        let json = block.to_stream_json();
        let spec = json["speculative_decoding"].as_object().expect("spec block is an object");
        assert!(!spec.contains_key("per_step_accepted"));
        assert!(!spec.contains_key("per_step_drafted"));
    }
}
