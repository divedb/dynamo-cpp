// SPDX-License-Identifier: Apache-2.0
//
// Prompt formatting — Dynamo's preprocessor/prompt: renders OpenAI-shaped
// requests through the model's HF chat template. The Jinja engine is
// vendored google/minja (built for tokenizer_config.json chat templates),
// standing in for Rust's minijinja+pycompat.

#pragma once

#include <memory>
#include <string>

#include <nlohmann/json.hpp>

#include "llm/protocols/openai.h"

namespace dynamo::llm {

/// What a chat-like request contributes to template rendering
/// (Rust's OAIChatLikeRequest).
struct ChatTemplateInput {
  nlohmann::ordered_json messages = nlohmann::ordered_json::array();
  nlohmann::ordered_json tools;  // null when absent
  bool add_generation_prompt = true;
};

ChatTemplateInput chat_template_input(const openai::NvCreateChatCompletionRequest& request);
ChatTemplateInput chat_template_input(const openai::NvCreateCompletionRequest& request);

/// Renders a chat-like request to the model's prompt string
/// (Rust's OAIPromptFormatter).
class PromptFormatter {
 public:
  virtual ~PromptFormatter() = default;
  virtual bool supports_add_generation_prompt() const = 0;
  /// Throws std::runtime_error on template errors.
  virtual std::string render(const ChatTemplateInput& input) const = 0;
};

/// HF chat-template formatter (Rust's HfTokenizerConfigJsonFormatter): built
/// from the `chat_template` + bos/eos token strings of a
/// tokenizer_config.json.
class HfChatTemplateFormatter final : public PromptFormatter {
 public:
  /// Throws std::runtime_error if a template does not parse.
  /// `tool_use_template` is the optional distinct "tool_use" entry of a
  /// named-template list; it is rendered instead of `chat_template` when the
  /// request carries tools (Rust's env["tool_use"] selection).
  /// `prompt_context` holds the MDC's context-mixin names (Rust's
  /// PromptContextMixin): "oai_chat" is a recognized no-op (as in Rust, where
  /// only the datetime mixin is implemented) and "llama3_datetime" injects
  /// `datetime` — the current UTC date, Llama-3 formatted — into the template
  /// context at render time. Unrecognized names throw (Rust rejects them at
  /// MDC deserialization).
  HfChatTemplateFormatter(const std::string& chat_template, const std::string& bos_token,
                          const std::string& eos_token,
                          const std::optional<std::string>& tool_use_template = std::nullopt,
                          const std::vector<std::string>& prompt_context = {});
  ~HfChatTemplateFormatter() override;

  bool supports_add_generation_prompt() const override { return supports_add_generation_prompt_; }
  std::string render(const ChatTemplateInput& input) const override;

 private:
  struct Impl;  // keeps minja out of this header
  std::unique_ptr<Impl> impl_;
  bool supports_add_generation_prompt_ = false;
};

}  // namespace dynamo::llm
