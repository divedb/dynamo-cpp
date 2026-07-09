// SPDX-License-Identifier: Apache-2.0

#include "llm/protocols/openai.h"

#include <algorithm>
#include <chrono>
#include <random>
#include <stdexcept>

#include <fmt/format.h>

namespace dynamo::llm::openai {

namespace {

uint64_t unix_now_seconds() {
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::seconds>(
                                   std::chrono::system_clock::now().time_since_epoch())
                                   .count());
}

std::string uuid4() {
  thread_local std::mt19937_64 rng{std::random_device{}()};
  uint64_t hi = rng();
  uint64_t lo = rng();
  // RFC 4122 version/variant bits.
  hi = (hi & 0xffffffffffff0fffULL) | 0x0000000000004000ULL;
  lo = (lo & 0x3fffffffffffffffULL) | 0x8000000000000000ULL;
  return fmt::format("{:08x}-{:04x}-{:04x}-{:04x}-{:012x}", static_cast<uint32_t>(hi >> 32),
                     static_cast<uint16_t>(hi >> 16), static_cast<uint16_t>(hi),
                     static_cast<uint16_t>(lo >> 48), lo & 0xffffffffffffULL);
}

std::optional<float> validate_range(std::optional<float> value,
                                    const std::pair<float, float>& range, const char* what) {
  if (!value.has_value()) return std::nullopt;
  if (*value < range.first || *value > range.second) {
    throw std::invalid_argument(fmt::format("Error validating {}: Value {} is out of range [{}, {}]",
                                            what, *value, range.first, range.second));
  }
  return value;
}

const NvExt* nvext_of(const std::optional<NvExt>& e) { return e.has_value() ? &*e : nullptr; }

SamplingOptions extract_sampling_impl(std::optional<float> temperature, std::optional<float> top_p,
                                      std::optional<float> frequency_penalty,
                                      std::optional<float> presence_penalty, const NvExt* nvext) {
  SamplingOptions options;
  options.temperature = validate_range(temperature, kTemperatureRange, "temperature");
  options.top_p = validate_range(top_p, kTopPRange, "top_p");
  options.frequency_penalty =
      validate_range(frequency_penalty, kFrequencyPenaltyRange, "frequency_penalty");
  options.presence_penalty =
      validate_range(presence_penalty, kPresencePenaltyRange, "presence_penalty");

  if (nvext != nullptr && nvext->greed_sampling.value_or(false)) {
    options.top_p.reset();
    options.temperature.reset();
  }
  return options;
}

StopConditions extract_stop_impl(std::optional<uint32_t> max_tokens,
                                 std::optional<std::vector<std::string>> stop,
                                 const NvExt* nvext) {
  if (stop.has_value() && stop->size() > 4) {
    throw std::invalid_argument("stop conditions must be less than 4");
  }
  StopConditions conditions;
  conditions.max_tokens = max_tokens;
  conditions.stop = std::move(stop);
  if (nvext != nullptr) conditions.ignore_eos = nvext->ignore_eos;
  return conditions;
}

std::optional<std::vector<std::string>> parse_stop_field(const nlohmann::json& j) {
  if (auto it = j.find("stop"); it != j.end() && !it->is_null()) {
    if (it->is_string()) return std::vector<std::string>{it->get<std::string>()};
    return it->get<std::vector<std::string>>();
  }
  return std::nullopt;
}

/// Maps a backend finish reason to the OpenAI wire string; throws on error
/// reasons. `cancelled_as` differs between chat ("stop") and completions
/// ("cancelled"), matching the Rust delta generators.
std::optional<std::string> map_finish_reason(const std::optional<FinishReason>& reason,
                                             const char* cancelled_as) {
  if (!reason.has_value()) return std::nullopt;
  switch (reason->kind) {
    case FinishReason::Kind::eos:
    case FinishReason::Kind::stop: return "stop";
    case FinishReason::Kind::length: return "length";
    case FinishReason::Kind::cancelled: return cancelled_as;
    case FinishReason::Kind::error: throw std::runtime_error(reason->error_message);
  }
  return std::nullopt;
}

std::string annotated_error_message(const std::optional<std::vector<std::string>>& comment) {
  if (comment.has_value() && !comment->empty()) return comment->front();
  return "unknown error";
}

}  // namespace

// ---------------------------------------------------------------------------
// NvExt

void NvExt::validate() const {
  if (top_k.has_value() && !(*top_k == -1 || *top_k >= 1)) {
    throw std::invalid_argument("top_k must be -1 or greater than or equal to 1");
  }
  if (repetition_penalty.has_value() &&
      !(*repetition_penalty > 0.0 && *repetition_penalty <= 2.0)) {
    throw std::invalid_argument("repetition_penalty must be in (0, 2]");
  }
}

void to_json(nlohmann::json& j, const NvExt& e) {
  j = nlohmann::json::object();
  set_opt(j, "ignore_eos", e.ignore_eos);
  set_opt(j, "top_k", e.top_k);
  set_opt(j, "repetition_penalty", e.repetition_penalty);
  set_opt(j, "greed_sampling", e.greed_sampling);
  set_opt(j, "use_raw_prompt", e.use_raw_prompt);
  set_opt(j, "annotations", e.annotations);
}

void from_json(const nlohmann::json& j, NvExt& e) {
  get_opt(j, "ignore_eos", e.ignore_eos);
  get_opt(j, "top_k", e.top_k);
  get_opt(j, "repetition_penalty", e.repetition_penalty);
  get_opt(j, "greed_sampling", e.greed_sampling);
  get_opt(j, "use_raw_prompt", e.use_raw_prompt);
  get_opt(j, "annotations", e.annotations);
}

// ---------------------------------------------------------------------------
// Usage

void to_json(nlohmann::json& j, const CompletionUsage& u) {
  j = nlohmann::json{{"prompt_tokens", u.prompt_tokens},
                     {"completion_tokens", u.completion_tokens},
                     {"total_tokens", u.total_tokens}};
}

void from_json(const nlohmann::json& j, CompletionUsage& u) {
  get_or(j, "prompt_tokens", u.prompt_tokens, 0);
  get_or(j, "completion_tokens", u.completion_tokens, 0);
  get_or(j, "total_tokens", u.total_tokens, 0);
}

// ---------------------------------------------------------------------------
// Chat messages

void to_json(nlohmann::json& j, const ChatMessage& m) {
  j = nlohmann::json{{"role", m.role}};
  // OpenAI requires the content key for most roles; emit null when absent.
  j["content"] = m.content.has_value() ? nlohmann::json(*m.content) : nlohmann::json(nullptr);
  set_opt(j, "name", m.name);
  if (!m.tool_calls.is_null()) j["tool_calls"] = m.tool_calls;
  set_opt(j, "tool_call_id", m.tool_call_id);
}

void from_json(const nlohmann::json& j, ChatMessage& m) {
  get_or(j, "role", m.role, {});
  get_opt(j, "name", m.name);
  m.tool_calls = j.value("tool_calls", nlohmann::json(nullptr));
  get_opt(j, "tool_call_id", m.tool_call_id);
  m.content.reset();
  if (auto it = j.find("content"); it != j.end() && !it->is_null()) {
    if (it->is_string()) {
      m.content = it->get<std::string>();
    } else if (it->is_array()) {
      // Multi-part content: concatenate the text parts.
      std::string text;
      for (const auto& part : *it) {
        if (part.is_object() && part.value("type", "") == "text") {
          if (!text.empty()) text += '\n';
          text += part.value("text", "");
        }
      }
      m.content = std::move(text);
    }
  }
}

// ---------------------------------------------------------------------------
// Chat request

std::optional<std::vector<std::string>> NvCreateChatCompletionRequest::annotations() const {
  if (nvext.has_value()) return nvext->annotations;
  return std::nullopt;
}

bool NvCreateChatCompletionRequest::has_annotation(const std::string& annotation) const {
  if (!nvext.has_value() || !nvext->annotations.has_value()) return false;
  const auto& list = *nvext->annotations;
  return std::find(list.begin(), list.end(), annotation) != list.end();
}

void to_json(nlohmann::json& j, const NvCreateChatCompletionRequest& r) {
  j = nlohmann::json{{"model", r.model}, {"messages", r.messages}};
  set_opt(j, "stream", r.stream);
  set_opt(j, "temperature", r.temperature);
  set_opt(j, "top_p", r.top_p);
  set_opt(j, "frequency_penalty", r.frequency_penalty);
  set_opt(j, "presence_penalty", r.presence_penalty);
  set_opt(j, "max_tokens", r.max_tokens);
  set_opt(j, "max_completion_tokens", r.max_completion_tokens);
  set_opt(j, "stop", r.stop);
  set_opt(j, "n", r.n);
  set_opt(j, "logprobs", r.logprobs);
  set_opt(j, "top_logprobs", r.top_logprobs);
  set_opt(j, "seed", r.seed);
  set_opt(j, "user", r.user);
  if (!r.tools.is_null()) j["tools"] = r.tools;
  if (!r.tool_choice.is_null()) j["tool_choice"] = r.tool_choice;
  set_opt(j, "nvext", r.nvext);
}

void from_json(const nlohmann::json& j, NvCreateChatCompletionRequest& r) {
  get_or(j, "model", r.model, {});
  get_or(j, "messages", r.messages, {});
  get_opt(j, "stream", r.stream);
  get_opt(j, "temperature", r.temperature);
  get_opt(j, "top_p", r.top_p);
  get_opt(j, "frequency_penalty", r.frequency_penalty);
  get_opt(j, "presence_penalty", r.presence_penalty);
  get_opt(j, "max_tokens", r.max_tokens);
  get_opt(j, "max_completion_tokens", r.max_completion_tokens);
  r.stop = parse_stop_field(j);
  get_opt(j, "n", r.n);
  get_opt(j, "logprobs", r.logprobs);
  get_opt(j, "top_logprobs", r.top_logprobs);
  get_opt(j, "seed", r.seed);
  get_opt(j, "user", r.user);
  // Tolerant like the rest, but shape-checked: a present-but-mistyped field
  // is rejected rather than fed to the chat template.
  r.tools = j.value("tools", nlohmann::json(nullptr));
  if (!r.tools.is_null() && !r.tools.is_array()) {
    throw std::invalid_argument("tools must be an array");
  }
  r.tool_choice = j.value("tool_choice", nlohmann::json(nullptr));
  if (!r.tool_choice.is_null() && !r.tool_choice.is_string() && !r.tool_choice.is_object()) {
    throw std::invalid_argument("tool_choice must be a string or an object");
  }
  get_opt(j, "nvext", r.nvext);
}

// ---------------------------------------------------------------------------
// Chat responses

void to_json(nlohmann::json& j, const ChatStreamDelta& d) {
  j = nlohmann::json::object();
  set_opt(j, "role", d.role);
  set_opt(j, "content", d.content);
}

void from_json(const nlohmann::json& j, ChatStreamDelta& d) {
  get_opt(j, "role", d.role);
  get_opt(j, "content", d.content);
}

void to_json(nlohmann::json& j, const ChatChoiceStream& c) {
  j = nlohmann::json{{"index", c.index}, {"delta", c.delta}};
  j["finish_reason"] =
      c.finish_reason.has_value() ? nlohmann::json(*c.finish_reason) : nlohmann::json(nullptr);
  j["logprobs"] = c.logprobs;
}

void from_json(const nlohmann::json& j, ChatChoiceStream& c) {
  get_or(j, "index", c.index, 0u);
  get_or(j, "delta", c.delta, {});
  get_opt(j, "finish_reason", c.finish_reason);
  c.logprobs = j.value("logprobs", nlohmann::json(nullptr));
}

void to_json(nlohmann::json& j, const NvCreateChatCompletionStreamResponse& r) {
  j = nlohmann::json{{"id", r.id},
                     {"choices", r.choices},
                     {"created", r.created},
                     {"model", r.model},
                     {"object", r.object}};
  set_opt(j, "usage", r.usage);
  set_opt(j, "system_fingerprint", r.system_fingerprint);
  set_opt(j, "service_tier", r.service_tier);
}

void from_json(const nlohmann::json& j, NvCreateChatCompletionStreamResponse& r) {
  get_or(j, "id", r.id, {});
  get_or(j, "choices", r.choices, {});
  get_or(j, "created", r.created, {});
  get_or(j, "model", r.model, {});
  get_or(j, "object", r.object, std::string("chat.completion.chunk"));
  get_opt(j, "usage", r.usage);
  get_opt(j, "system_fingerprint", r.system_fingerprint);
  get_opt(j, "service_tier", r.service_tier);
}

void to_json(nlohmann::json& j, const ChatResponseMessage& m) {
  j = nlohmann::json{{"role", m.role}};
  j["content"] = m.content.has_value() ? nlohmann::json(*m.content) : nlohmann::json(nullptr);
}

void from_json(const nlohmann::json& j, ChatResponseMessage& m) {
  get_or(j, "role", m.role, {});
  get_opt(j, "content", m.content);
}

void to_json(nlohmann::json& j, const ChatChoice& c) {
  j = nlohmann::json{{"index", c.index}, {"message", c.message}};
  j["finish_reason"] =
      c.finish_reason.has_value() ? nlohmann::json(*c.finish_reason) : nlohmann::json(nullptr);
  j["logprobs"] = c.logprobs;
}

void from_json(const nlohmann::json& j, ChatChoice& c) {
  get_or(j, "index", c.index, 0u);
  get_or(j, "message", c.message, {});
  get_opt(j, "finish_reason", c.finish_reason);
  c.logprobs = j.value("logprobs", nlohmann::json(nullptr));
}

void to_json(nlohmann::json& j, const NvCreateChatCompletionResponse& r) {
  j = nlohmann::json{{"id", r.id},
                     {"choices", r.choices},
                     {"created", r.created},
                     {"model", r.model},
                     {"object", r.object}};
  set_opt(j, "usage", r.usage);
  set_opt(j, "system_fingerprint", r.system_fingerprint);
  set_opt(j, "service_tier", r.service_tier);
}

void from_json(const nlohmann::json& j, NvCreateChatCompletionResponse& r) {
  get_or(j, "id", r.id, {});
  get_or(j, "choices", r.choices, {});
  get_or(j, "created", r.created, {});
  get_or(j, "model", r.model, {});
  get_or(j, "object", r.object, std::string("chat.completion"));
  get_opt(j, "usage", r.usage);
  get_opt(j, "system_fingerprint", r.system_fingerprint);
  get_opt(j, "service_tier", r.service_tier);
}

// ---------------------------------------------------------------------------
// Completions

std::string NvCreateCompletionRequest::prompt_to_string() const {
  if (prompt.is_string()) return prompt.get<std::string>();
  if (prompt.is_array()) {
    std::string out;
    bool nested = !prompt.empty() && prompt.front().is_array();
    const char* sep = nested ? " | " : " ";
    for (const auto& item : prompt) {
      if (!out.empty()) out += sep;
      if (item.is_string()) {
        out += item.get<std::string>();
      } else if (item.is_number()) {
        out += item.dump();
      } else if (item.is_array()) {
        std::string inner;
        for (const auto& num : item) {
          if (!inner.empty()) inner += ' ';
          inner += num.dump();
        }
        out += inner;
      }
    }
    return out;
  }
  return {};
}

std::optional<std::string> NvCreateCompletionRequest::raw_prompt() const {
  if (nvext.has_value() && nvext->use_raw_prompt.value_or(false)) return prompt_to_string();
  return std::nullopt;
}

std::optional<std::vector<std::string>> NvCreateCompletionRequest::annotations() const {
  if (nvext.has_value()) return nvext->annotations;
  return std::nullopt;
}

bool NvCreateCompletionRequest::has_annotation(const std::string& annotation) const {
  if (!nvext.has_value() || !nvext->annotations.has_value()) return false;
  const auto& list = *nvext->annotations;
  return std::find(list.begin(), list.end(), annotation) != list.end();
}

void to_json(nlohmann::json& j, const NvCreateCompletionRequest& r) {
  j = nlohmann::json{{"model", r.model}, {"prompt", r.prompt}};
  set_opt(j, "max_tokens", r.max_tokens);
  set_opt(j, "temperature", r.temperature);
  set_opt(j, "top_p", r.top_p);
  set_opt(j, "frequency_penalty", r.frequency_penalty);
  set_opt(j, "presence_penalty", r.presence_penalty);
  set_opt(j, "n", r.n);
  set_opt(j, "best_of", r.best_of);
  set_opt(j, "logprobs", r.logprobs);
  set_opt(j, "stream", r.stream);
  set_opt(j, "echo", r.echo);
  set_opt(j, "stop", r.stop);
  set_opt(j, "suffix", r.suffix);
  set_opt(j, "user", r.user);
  set_opt(j, "nvext", r.nvext);
}

void from_json(const nlohmann::json& j, NvCreateCompletionRequest& r) {
  get_or(j, "model", r.model, {});
  r.prompt = j.value("prompt", nlohmann::json(nullptr));
  get_opt(j, "max_tokens", r.max_tokens);
  get_opt(j, "temperature", r.temperature);
  get_opt(j, "top_p", r.top_p);
  get_opt(j, "frequency_penalty", r.frequency_penalty);
  get_opt(j, "presence_penalty", r.presence_penalty);
  get_opt(j, "n", r.n);
  get_opt(j, "best_of", r.best_of);
  get_opt(j, "logprobs", r.logprobs);
  get_opt(j, "stream", r.stream);
  get_opt(j, "echo", r.echo);
  r.stop = parse_stop_field(j);
  get_opt(j, "suffix", r.suffix);
  get_opt(j, "user", r.user);
  get_opt(j, "nvext", r.nvext);
}

void to_json(nlohmann::json& j, const CompletionChoice& c) {
  j = nlohmann::json{{"text", c.text}, {"index", c.index}};
  j["finish_reason"] =
      c.finish_reason.has_value() ? nlohmann::json(*c.finish_reason) : nlohmann::json(nullptr);
  if (!c.logprobs.is_null()) j["logprobs"] = c.logprobs;
}

void from_json(const nlohmann::json& j, CompletionChoice& c) {
  get_or(j, "text", c.text, {});
  get_or(j, "index", c.index, {});
  get_opt(j, "finish_reason", c.finish_reason);
  c.logprobs = j.value("logprobs", nlohmann::json(nullptr));
}

void to_json(nlohmann::json& j, const NvCreateCompletionResponse& r) {
  j = nlohmann::json{{"id", r.id},
                     {"choices", r.choices},
                     {"created", r.created},
                     {"model", r.model},
                     {"object", r.object}};
  set_opt(j, "usage", r.usage);
  set_opt(j, "system_fingerprint", r.system_fingerprint);
}

void from_json(const nlohmann::json& j, NvCreateCompletionResponse& r) {
  get_or(j, "id", r.id, {});
  get_or(j, "choices", r.choices, {});
  get_or(j, "created", r.created, {});
  get_or(j, "model", r.model, {});
  get_or(j, "object", r.object, std::string("text_completion"));
  get_opt(j, "usage", r.usage);
  get_opt(j, "system_fingerprint", r.system_fingerprint);
}

// ---------------------------------------------------------------------------
// Option extraction

SamplingOptions extract_sampling_options(const NvCreateChatCompletionRequest& request) {
  return extract_sampling_impl(request.temperature, request.top_p, request.frequency_penalty,
                               request.presence_penalty, nvext_of(request.nvext));
}

SamplingOptions extract_sampling_options(const NvCreateCompletionRequest& request) {
  return extract_sampling_impl(request.temperature, request.top_p, request.frequency_penalty,
                               request.presence_penalty, nvext_of(request.nvext));
}

StopConditions extract_stop_conditions(const NvCreateChatCompletionRequest& request) {
  // max_completion_tokens is preferred over the deprecated max_tokens.
  auto max_tokens = request.max_completion_tokens.has_value() ? request.max_completion_tokens
                                                              : request.max_tokens;
  return extract_stop_impl(max_tokens, request.stop, nvext_of(request.nvext));
}

StopConditions extract_stop_conditions(const NvCreateCompletionRequest& request) {
  // Deviation from Rust v0.1.0, which drops `stop` for legacy completions
  // (its get_stop() is hardwired to None); we honor the request field.
  return extract_stop_impl(request.max_tokens, request.stop, nvext_of(request.nvext));
}

// ---------------------------------------------------------------------------
// ChatDeltaGenerator

ChatDeltaGenerator::ChatDeltaGenerator(std::string model, DeltaGeneratorOptions options)
    : id_("chatcmpl-" + uuid4()),
      created_(unix_now_seconds()),
      model_(std::move(model)),
      options_(options) {}

void ChatDeltaGenerator::update_isl(uint32_t input_tokens) {
  usage_.prompt_tokens = static_cast<int32_t>(input_tokens);
}

NvCreateChatCompletionStreamResponse ChatDeltaGenerator::create_choice(
    uint32_t index, std::optional<std::string> text,
    std::optional<std::string> finish_reason) {
  ChatChoiceStream choice;
  choice.index = index;
  // Rust v0.1.0 stamps the assistant role on every chunk (its msg_counter is
  // never incremented); we match that.
  choice.delta.role = "assistant";
  choice.delta.content = std::move(text);
  choice.finish_reason = std::move(finish_reason);

  NvCreateChatCompletionStreamResponse response;
  response.id = id_;
  response.created = created_;
  response.model = model_;
  response.choices.push_back(std::move(choice));
  if (options_.enable_usage) {
    CompletionUsage usage = usage_;
    // Rust leaves total_tokens at 0; computing it here is the obvious intent.
    usage.total_tokens = usage.prompt_tokens + usage.completion_tokens;
    response.usage = usage;
  }
  return response;
}

NvCreateChatCompletionStreamResponse ChatDeltaGenerator::choice_from_postprocessor(
    const BackendOutput& delta) {
  if (options_.enable_usage) {
    usage_.completion_tokens += static_cast<int32_t>(delta.token_ids.size());
  }
  auto finish_reason = map_finish_reason(delta.finish_reason, /*cancelled_as=*/"stop");
  return create_choice(0, delta.text, std::move(finish_reason));
}

ChatDeltaGenerator response_generator(const NvCreateChatCompletionRequest& request) {
  DeltaGeneratorOptions options;
  options.enable_usage = true;
  options.enable_logprobs = request.logprobs.value_or(false);
  return ChatDeltaGenerator(request.model, options);
}

// ---------------------------------------------------------------------------
// CompletionDeltaGenerator

CompletionDeltaGenerator::CompletionDeltaGenerator(std::string model,
                                                   DeltaGeneratorOptions options)
    : id_("cmpl-" + uuid4()),
      created_(unix_now_seconds()),
      model_(std::move(model)),
      options_(options) {}

void CompletionDeltaGenerator::update_isl(int32_t input_tokens) {
  usage_.prompt_tokens = input_tokens;
}

NvCreateCompletionResponse CompletionDeltaGenerator::create_choice(
    uint64_t index, std::optional<std::string> text,
    std::optional<std::string> finish_reason) {
  CompletionChoice choice;
  choice.text = text.value_or("");
  choice.index = index;
  choice.finish_reason = std::move(finish_reason);

  NvCreateCompletionResponse response;
  response.id = id_;
  response.created = created_;
  response.model = model_;
  response.choices.push_back(std::move(choice));
  if (options_.enable_usage) {
    CompletionUsage usage = usage_;
    usage.total_tokens = usage.prompt_tokens + usage.completion_tokens;
    response.usage = usage;
  }
  return response;
}

NvCreateCompletionResponse CompletionDeltaGenerator::choice_from_postprocessor(
    const BackendOutput& delta) {
  if (options_.enable_usage) {
    usage_.completion_tokens += static_cast<int32_t>(delta.token_ids.size());
  }
  auto finish_reason = map_finish_reason(delta.finish_reason, /*cancelled_as=*/"cancelled");
  return create_choice(0, delta.text, std::move(finish_reason));
}

CompletionDeltaGenerator response_generator(const NvCreateCompletionRequest& request) {
  DeltaGeneratorOptions options;
  options.enable_usage = true;
  options.enable_logprobs = false;
  return CompletionDeltaGenerator(request.model, options);
}

// ---------------------------------------------------------------------------
// ChatDeltaAggregator

void ChatDeltaAggregator::push(
    const pipeline::Annotated<NvCreateChatCompletionStreamResponse>& delta) {
  if (delta.is_error()) {
    error_ = annotated_error_message(delta.comment);
    return;
  }
  if (error_.has_value() || !delta.data.has_value()) return;

  const auto& chunk = *delta.data;
  id_ = chunk.id;
  model_ = chunk.model;
  created_ = chunk.created;
  if (chunk.service_tier.has_value()) service_tier_ = chunk.service_tier;
  if (chunk.usage.has_value()) usage_ = chunk.usage;
  if (chunk.system_fingerprint.has_value()) system_fingerprint_ = chunk.system_fingerprint;

  for (const auto& choice : chunk.choices) {
    auto [it, inserted] = choices_.try_emplace(choice.index);
    DeltaChoice& state = it->second;
    if (inserted) {
      state.index = choice.index;
      state.role = choice.delta.role;
      state.logprobs = choice.logprobs;
    }
    if (choice.delta.content.has_value()) state.text += *choice.delta.content;
    if (choice.finish_reason.has_value()) state.finish_reason = choice.finish_reason;
  }
}

NvCreateChatCompletionResponse ChatDeltaAggregator::finalize() && {
  if (error_.has_value()) throw std::runtime_error(*error_);

  NvCreateChatCompletionResponse response;
  response.id = std::move(id_);
  response.model = std::move(model_);
  response.created = created_;
  response.usage = usage_;
  response.system_fingerprint = std::move(system_fingerprint_);
  response.service_tier = std::move(service_tier_);

  for (auto& [index, delta] : choices_) {
    if (!delta.role.has_value()) {
      throw std::runtime_error("chat choice " + std::to_string(index) + " never received a role");
    }
    ChatChoice choice;
    choice.index = delta.index;
    choice.message.role = std::move(*delta.role);
    choice.message.content = std::move(delta.text);
    choice.finish_reason = std::move(delta.finish_reason);
    choice.logprobs = std::move(delta.logprobs);
    response.choices.push_back(std::move(choice));
  }
  std::sort(response.choices.begin(), response.choices.end(),
            [](const ChatChoice& a, const ChatChoice& b) { return a.index < b.index; });
  return response;
}

// ---------------------------------------------------------------------------
// CompletionDeltaAggregator

void CompletionDeltaAggregator::push(
    const pipeline::Annotated<NvCreateCompletionResponse>& delta) {
  if (delta.is_error()) {
    error_ = annotated_error_message(delta.comment);
    return;
  }
  if (error_.has_value() || !delta.data.has_value()) return;

  const auto& chunk = *delta.data;
  id_ = chunk.id;
  model_ = chunk.model;
  created_ = chunk.created;
  if (chunk.usage.has_value()) usage_ = chunk.usage;
  if (chunk.system_fingerprint.has_value()) system_fingerprint_ = chunk.system_fingerprint;

  for (const auto& choice : chunk.choices) {
    auto [it, inserted] = choices_.try_emplace(choice.index);
    DeltaChoice& state = it->second;
    if (inserted) {
      state.index = choice.index;
      state.logprobs = choice.logprobs;
    }
    state.text += choice.text;
    if (choice.finish_reason.has_value()) {
      // Tolerant parse, as in Rust: unknown reason strings become "none".
      try {
        state.finish_reason = FinishReason::from_string(*choice.finish_reason);
      } catch (const std::invalid_argument&) {
        state.finish_reason.reset();
      }
    }
  }
}

NvCreateCompletionResponse CompletionDeltaAggregator::finalize() && {
  if (error_.has_value()) throw std::runtime_error(*error_);

  NvCreateCompletionResponse response;
  response.id = std::move(id_);
  response.model = std::move(model_);
  response.created = created_;
  response.usage = usage_;
  response.system_fingerprint = std::move(system_fingerprint_);

  for (auto& [index, delta] : choices_) {
    (void)index;
    CompletionChoice choice;
    choice.index = delta.index;
    choice.text = std::move(delta.text);
    if (delta.finish_reason.has_value()) choice.finish_reason = delta.finish_reason->to_string();
    choice.logprobs = std::move(delta.logprobs);
    response.choices.push_back(std::move(choice));
  }
  std::sort(response.choices.begin(), response.choices.end(),
            [](const CompletionChoice& a, const CompletionChoice& b) { return a.index < b.index; });
  return response;
}

}  // namespace dynamo::llm::openai
