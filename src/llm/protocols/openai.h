// SPDX-License-Identifier: Apache-2.0
//
// OpenAI-compatible request/response types with NVIDIA extensions — Dynamo's
// protocols::openai (chat_completions, completions, nvext) with the subset of
// async-openai types the Rust crate actually uses, hand-modelled on
// nlohmann_json with tolerant parsing (missing/null optional fields accepted,
// unknown fields ignored).
//
// Includes the DeltaGenerator (BackendOutput -> streaming response chunk) and
// DeltaAggregator (stream of chunks -> final unary response) for both the
// chat and legacy completions endpoints.

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "llm/protocols/common.h"
#include "llm/protocols/llm_backend.h"
#include "pipeline/annotated.h"

namespace dynamo::llm::openai {

// Allowed ranges for OpenAI sampling options.
inline constexpr std::pair<float, float> kTemperatureRange{0.0f, 2.0f};
inline constexpr std::pair<float, float> kTopPRange{0.0f, 1.0f};
inline constexpr std::pair<float, float> kFrequencyPenaltyRange{-2.0f, 2.0f};
inline constexpr std::pair<float, float> kPresencePenaltyRange{-2.0f, 2.0f};

// ---------------------------------------------------------------------------
// NvExt — NVIDIA LLM extensions to the OpenAI API

struct NvExt {
  /// Ignore EOS and generate until max_tokens.
  std::optional<bool> ignore_eos;
  /// -1 (all) or >= 1.
  std::optional<int64_t> top_k;
  /// In (0, 2]; 1 means no penalty.
  std::optional<double> repetition_penalty;
  /// Force greedy sampling.
  std::optional<bool> greed_sampling;
  /// Bypass the prompt template and tokenize the prompt directly.
  std::optional<bool> use_raw_prompt;
  /// Annotation triggers echoed back out-of-band on the SSE `event:` field.
  std::optional<std::vector<std::string>> annotations;

  /// Throws std::invalid_argument on out-of-range values.
  void validate() const;
};

void to_json(nlohmann::json& j, const NvExt& e);
void from_json(const nlohmann::json& j, NvExt& e);

// ---------------------------------------------------------------------------
// Usage

struct CompletionUsage {
  int32_t prompt_tokens = 0;
  int32_t completion_tokens = 0;
  int32_t total_tokens = 0;
};

void to_json(nlohmann::json& j, const CompletionUsage& u);
void from_json(const nlohmann::json& j, CompletionUsage& u);

// ---------------------------------------------------------------------------
// Chat completions

/// A chat request message. `content` is flattened to text: string content is
/// taken as-is; array-of-parts content concatenates its "text" parts joined
/// with newlines.
struct ChatMessage {
  std::string role;
  std::optional<std::string> content;
  std::optional<std::string> name;
  /// Assistant tool-call turns, kept as raw JSON for template rendering;
  /// null when absent.
  nlohmann::json tool_calls = nullptr;
  /// Set on role:"tool" result turns.
  std::optional<std::string> tool_call_id;
};

void to_json(nlohmann::json& j, const ChatMessage& m);
void from_json(const nlohmann::json& j, ChatMessage& m);

struct NvCreateChatCompletionRequest {
  std::string model;
  std::vector<ChatMessage> messages;
  std::optional<bool> stream;
  std::optional<float> temperature;
  std::optional<float> top_p;
  std::optional<float> frequency_penalty;
  std::optional<float> presence_penalty;
  std::optional<uint32_t> max_tokens;             // deprecated upstream
  std::optional<uint32_t> max_completion_tokens;  // preferred
  /// Normalized from string-or-array JSON.
  std::optional<std::vector<std::string>> stop;
  std::optional<int32_t> n;
  std::optional<bool> logprobs;
  std::optional<int32_t> top_logprobs;
  std::optional<int64_t> seed;
  std::optional<std::string> user;
  /// Tool definitions / selection, kept as raw JSON (validated only as
  /// "array" / "string-or-object"); consumed by the chat template. Null when
  /// absent.
  nlohmann::json tools = nullptr;
  nlohmann::json tool_choice = nullptr;
  std::optional<NvExt> nvext;

  std::optional<std::vector<std::string>> annotations() const;
  bool has_annotation(const std::string& annotation) const;
};

void to_json(nlohmann::json& j, const NvCreateChatCompletionRequest& r);
void from_json(const nlohmann::json& j, NvCreateChatCompletionRequest& r);

struct ChatStreamDelta {
  std::optional<std::string> role;
  std::optional<std::string> content;
};

void to_json(nlohmann::json& j, const ChatStreamDelta& d);
void from_json(const nlohmann::json& j, ChatStreamDelta& d);

struct ChatChoiceStream {
  uint32_t index = 0;
  ChatStreamDelta delta;
  std::optional<std::string> finish_reason;
  nlohmann::json logprobs;  // pass-through; null when absent
};

void to_json(nlohmann::json& j, const ChatChoiceStream& c);
void from_json(const nlohmann::json& j, ChatChoiceStream& c);

struct NvCreateChatCompletionStreamResponse {
  std::string id;
  std::vector<ChatChoiceStream> choices;
  uint64_t created = 0;
  std::string model;
  std::string object = "chat.completion.chunk";
  std::optional<CompletionUsage> usage;
  std::optional<std::string> system_fingerprint;
  std::optional<std::string> service_tier;
};

void to_json(nlohmann::json& j, const NvCreateChatCompletionStreamResponse& r);
void from_json(const nlohmann::json& j, NvCreateChatCompletionStreamResponse& r);

struct ChatResponseMessage {
  std::string role;
  std::optional<std::string> content;
};

void to_json(nlohmann::json& j, const ChatResponseMessage& m);
void from_json(const nlohmann::json& j, ChatResponseMessage& m);

struct ChatChoice {
  uint32_t index = 0;
  ChatResponseMessage message;
  std::optional<std::string> finish_reason;
  nlohmann::json logprobs;
};

void to_json(nlohmann::json& j, const ChatChoice& c);
void from_json(const nlohmann::json& j, ChatChoice& c);

struct NvCreateChatCompletionResponse {
  std::string id;
  std::vector<ChatChoice> choices;
  uint64_t created = 0;
  std::string model;
  std::string object = "chat.completion";
  std::optional<CompletionUsage> usage;
  std::optional<std::string> system_fingerprint;
  std::optional<std::string> service_tier;
};

void to_json(nlohmann::json& j, const NvCreateChatCompletionResponse& r);
void from_json(const nlohmann::json& j, NvCreateChatCompletionResponse& r);

// ---------------------------------------------------------------------------
// Legacy completions

struct NvCreateCompletionRequest {
  std::string model;
  /// string | string[] | int[] | int[][] — kept raw; see prompt_to_string().
  nlohmann::json prompt;
  std::optional<uint32_t> max_tokens;
  std::optional<float> temperature;
  std::optional<float> top_p;
  std::optional<float> frequency_penalty;
  std::optional<float> presence_penalty;
  std::optional<int32_t> n;
  std::optional<int32_t> best_of;
  std::optional<int32_t> logprobs;
  std::optional<bool> stream;
  std::optional<bool> echo;
  std::optional<std::vector<std::string>> stop;
  std::optional<std::string> suffix;
  std::optional<std::string> user;
  std::optional<NvExt> nvext;

  /// Flattens the prompt to a single string (arrays joined with spaces,
  /// nested int arrays separated by " | "), as in the Rust prompt_to_string.
  std::string prompt_to_string() const;

  /// The raw prompt when nvext.use_raw_prompt is set.
  std::optional<std::string> raw_prompt() const;

  std::optional<std::vector<std::string>> annotations() const;
  bool has_annotation(const std::string& annotation) const;
};

void to_json(nlohmann::json& j, const NvCreateCompletionRequest& r);
void from_json(const nlohmann::json& j, NvCreateCompletionRequest& r);

struct CompletionChoice {
  std::string text;
  uint64_t index = 0;
  std::optional<std::string> finish_reason;
  nlohmann::json logprobs;
};

void to_json(nlohmann::json& j, const CompletionChoice& c);
void from_json(const nlohmann::json& j, CompletionChoice& c);

/// Streamed and unary completion responses share this shape.
struct NvCreateCompletionResponse {
  std::string id;
  std::vector<CompletionChoice> choices;
  uint64_t created = 0;
  std::string model;
  std::string object = "text_completion";
  std::optional<CompletionUsage> usage;
  std::optional<std::string> system_fingerprint;
};

void to_json(nlohmann::json& j, const NvCreateCompletionResponse& r);
void from_json(const nlohmann::json& j, NvCreateCompletionResponse& r);

// ---------------------------------------------------------------------------
// Request -> engine option extraction

/// Validates ranges and maps OpenAI sampling fields to engine SamplingOptions;
/// nvext.greed_sampling clears temperature/top_p. Throws std::invalid_argument.
SamplingOptions extract_sampling_options(const NvCreateChatCompletionRequest& request);
SamplingOptions extract_sampling_options(const NvCreateCompletionRequest& request);

/// Maps stop/max_tokens (and nvext.ignore_eos) to engine StopConditions.
/// At most 4 stop sequences are allowed. Throws std::invalid_argument.
StopConditions extract_stop_conditions(const NvCreateChatCompletionRequest& request);
StopConditions extract_stop_conditions(const NvCreateCompletionRequest& request);

// ---------------------------------------------------------------------------
// Delta generators: BackendOutput -> response chunk

struct DeltaGeneratorOptions {
  bool enable_usage = true;
  bool enable_logprobs = false;
};

/// Streaming chat response factory; one per request.
class ChatDeltaGenerator {
 public:
  ChatDeltaGenerator(std::string model, DeltaGeneratorOptions options);

  /// Records the input (prompt) sequence length for usage reporting.
  void update_isl(uint32_t input_tokens);

  NvCreateChatCompletionStreamResponse create_choice(
      uint32_t index, std::optional<std::string> text,
      std::optional<std::string> finish_reason);

  /// Maps a BackendOutput delta to a chunk; throws std::runtime_error when
  /// the backend reported FinishReason::error.
  NvCreateChatCompletionStreamResponse choice_from_postprocessor(const BackendOutput& delta);

  const std::string& id() const { return id_; }

 private:
  std::string id_;
  uint64_t created_ = 0;
  std::string model_;
  CompletionUsage usage_;
  DeltaGeneratorOptions options_;
};

/// Factory matching Rust: options from the request (usage on; logprobs when
/// request.logprobs is true).
ChatDeltaGenerator response_generator(const NvCreateChatCompletionRequest& request);

/// Streaming legacy-completions response factory; one per request.
class CompletionDeltaGenerator {
 public:
  CompletionDeltaGenerator(std::string model, DeltaGeneratorOptions options);

  void update_isl(int32_t input_tokens);

  NvCreateCompletionResponse create_choice(uint64_t index, std::optional<std::string> text,
                                           std::optional<std::string> finish_reason);

  NvCreateCompletionResponse choice_from_postprocessor(const BackendOutput& delta);

  const std::string& id() const { return id_; }

 private:
  std::string id_;
  uint64_t created_ = 0;
  std::string model_;
  CompletionUsage usage_;
  DeltaGeneratorOptions options_;
};

CompletionDeltaGenerator response_generator(const NvCreateCompletionRequest& request);

// ---------------------------------------------------------------------------
// Delta aggregators: stream of chunks -> final unary response

/// Folds annotated chat chunks into a unary chat response. push() all deltas,
/// then finalize(); finalize throws std::runtime_error if the stream carried
/// an error annotation (or a choice never received a role).
class ChatDeltaAggregator {
 public:
  void push(const pipeline::Annotated<NvCreateChatCompletionStreamResponse>& delta);
  NvCreateChatCompletionResponse finalize() &&;

 private:
  struct DeltaChoice {
    uint32_t index = 0;
    std::string text;
    std::optional<std::string> role;
    std::optional<std::string> finish_reason;
    nlohmann::json logprobs;
  };

  std::string id_;
  std::string model_;
  uint64_t created_ = 0;
  std::optional<CompletionUsage> usage_;
  std::optional<std::string> system_fingerprint_;
  std::optional<std::string> service_tier_;
  std::unordered_map<uint32_t, DeltaChoice> choices_;
  std::optional<std::string> error_;
};

/// Folds annotated completion chunks into a unary completion response.
class CompletionDeltaAggregator {
 public:
  void push(const pipeline::Annotated<NvCreateCompletionResponse>& delta);
  NvCreateCompletionResponse finalize() &&;

 private:
  struct DeltaChoice {
    uint64_t index = 0;
    std::string text;
    std::optional<FinishReason> finish_reason;
    nlohmann::json logprobs;
  };

  std::string id_;
  std::string model_;
  uint64_t created_ = 0;
  std::optional<CompletionUsage> usage_;
  std::optional<std::string> system_fingerprint_;
  std::unordered_map<uint64_t, DeltaChoice> choices_;
  std::optional<std::string> error_;
};

}  // namespace dynamo::llm::openai
