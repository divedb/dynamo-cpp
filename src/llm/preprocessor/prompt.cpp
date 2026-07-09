// SPDX-License-Identifier: Apache-2.0

#include "llm/preprocessor/prompt.h"

#include <ctime>
#include <stdexcept>

#include <fmt/format.h>
#include <minja/chat-template.hpp>

namespace dynamo::llm {

namespace {

/// Rust's llama3_datetime: chrono "%d, %B, %Y", e.g. "09, July, 2026".
/// Month names are spelled out so the result is locale-independent.
std::string llama3_datetime_now() {
  static constexpr const char* kMonths[] = {"January",   "February", "March",    "April",
                                            "May",       "June",     "July",     "August",
                                            "September", "October",  "November", "December"};
  std::time_t now = std::time(nullptr);
  std::tm tm{};
  gmtime_r(&now, &tm);
  return fmt::format("{:02d}, {}, {}", tm.tm_mday, kMonths[tm.tm_mon], 1900 + tm.tm_year);
}

nlohmann::ordered_json message_json(const openai::ChatMessage& message) {
  nlohmann::ordered_json m;
  m["role"] = message.role;
  m["content"] = message.content.has_value() ? nlohmann::ordered_json(*message.content)
                                             : nlohmann::ordered_json(nullptr);
  if (message.name.has_value()) m["name"] = *message.name;
  if (!message.tool_calls.is_null()) m["tool_calls"] = message.tool_calls;
  if (message.tool_call_id.has_value()) m["tool_call_id"] = *message.tool_call_id;
  return m;
}

}  // namespace

ChatTemplateInput chat_template_input(const openai::NvCreateChatCompletionRequest& request) {
  ChatTemplateInput input;
  for (const auto& message : request.messages) {
    input.messages.push_back(message_json(message));
  }
  input.tools = request.tools;
  // Rust: add the generation prompt iff the last message is a user turn
  // (or the message list is empty).
  input.add_generation_prompt =
      request.messages.empty() || request.messages.back().role == "user";
  return input;
}

ChatTemplateInput chat_template_input(const openai::NvCreateCompletionRequest& request) {
  ChatTemplateInput input;
  openai::ChatMessage message;
  message.role = "user";
  message.content = request.prompt_to_string();
  input.messages.push_back(message_json(message));
  input.add_generation_prompt = true;
  return input;
}

struct HfChatTemplateFormatter::Impl {
  Impl(const std::string& source, const std::string& bos, const std::string& eos)
      : tmpl(source, bos, eos) {}
  minja::chat_template tmpl;
  // Distinct "tool_use" named template, when the config provides one.
  std::optional<minja::chat_template> tool_tmpl;
  // llama3_datetime mixin: inject `datetime` at render time.
  bool llama3_datetime = false;
};

HfChatTemplateFormatter::HfChatTemplateFormatter(const std::string& chat_template,
                                                 const std::string& bos_token,
                                                 const std::string& eos_token,
                                                 const std::optional<std::string>& tool_use_template,
                                                 const std::vector<std::string>& prompt_context) {
  if (chat_template.empty()) {
    throw std::runtime_error("chat_template must not be empty");
  }
  try {
    impl_ = std::make_unique<Impl>(chat_template, bos_token, eos_token);
    if (tool_use_template.has_value() && !tool_use_template->empty()) {
      impl_->tool_tmpl.emplace(*tool_use_template, bos_token, eos_token);
    }
  } catch (const std::exception& e) {
    throw std::runtime_error(std::string("failed to parse chat template: ") + e.what());
  }
  for (const auto& mixin : prompt_context) {
    if (mixin == "llama3_datetime") {
      impl_->llama3_datetime = true;
    } else if (mixin != "oai_chat") {  // oai_chat: recognized no-op, as in Rust
      throw std::runtime_error("unknown prompt_context mixin: " + mixin);
    }
  }
  // Rust requires every named template to contain the key; with at most two
  // templates here, "all of them" is the same check applied to each.
  supports_add_generation_prompt_ =
      chat_template.find("add_generation_prompt") != std::string::npos &&
      (!impl_->tool_tmpl.has_value() ||
       tool_use_template->find("add_generation_prompt") != std::string::npos);
}

HfChatTemplateFormatter::~HfChatTemplateFormatter() = default;

std::string HfChatTemplateFormatter::render(const ChatTemplateInput& input) const {
  minja::chat_template_inputs inputs;
  inputs.messages = input.messages;
  inputs.tools = input.tools;
  inputs.add_generation_prompt = input.add_generation_prompt;
  if (impl_->llama3_datetime) {
    // Computed per render, matching Rust (Utc::now() inside the mixin).
    inputs.extra_context["datetime"] = llama3_datetime_now();
  }
  // Rust renders the "tool_use" named template whenever tools are present
  // (for a plain string chat_template both names hold the same source).
  // Deviation: with a named-list template that lacks "tool_use", Rust errors;
  // we fall back to "default" with tools in the context, where minja's
  // tool-call polyfills apply.
  const bool has_tools = !input.tools.is_null();
  const minja::chat_template& tmpl =
      (has_tools && impl_->tool_tmpl.has_value()) ? *impl_->tool_tmpl : impl_->tmpl;
  try {
    return tmpl.apply(inputs);
  } catch (const std::exception& e) {
    throw std::runtime_error(std::string("failed to render chat template: ") + e.what());
  }
}

}  // namespace dynamo::llm
