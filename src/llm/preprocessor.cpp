// SPDX-License-Identifier: Apache-2.0

#include "llm/preprocessor.h"

#include <algorithm>

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include "llm/model_card.h"

namespace dynamo::llm {

TokenizerPtr tokenizer_from_mdc(const ModelDeploymentCard& mdc) {
  if (mdc.tokenizer.kind == kTokenizerByteLevel) {
    return std::make_shared<ByteLevelTokenizer>();
  }
  throw std::runtime_error(
      "unsupported tokenizer kind '" + mdc.tokenizer.kind +
      "': the HuggingFace tokenizer backend is not vendored yet (see TODO.md M2)");
}

std::shared_ptr<const PromptFormatter> formatter_from_mdc(const ModelDeploymentCard& mdc) {
  if (!mdc.prompt_formatter.has_value()) {
    throw std::runtime_error("model card has no prompt_formatter artifact");
  }
  if (mdc.prompt_formatter->kind != kPromptFormatterHfTokenizerConfigJson) {
    throw std::runtime_error("unsupported prompt_formatter kind: " + mdc.prompt_formatter->kind);
  }
  auto config = HfTokenizerConfig::from_file(mdc.prompt_formatter->path);
  if (!config.chat_template.has_value()) {
    throw std::runtime_error(
        "chat_template field is required in the tokenizer_config.json file");
  }
  return std::make_shared<HfChatTemplateFormatter>(
      *config.chat_template, config.bos_token, config.eos_token, config.tool_use_chat_template,
      mdc.prompt_context.value_or(std::vector<std::string>{}));
}

std::shared_ptr<OpenAIPreprocessor> OpenAIPreprocessor::from_mdc(const ModelDeploymentCard& mdc) {
  return std::make_shared<OpenAIPreprocessor>(formatter_from_mdc(mdc), tokenizer_from_mdc(mdc),
                                              mdc.load_model_info().eos_token_ids, mdc.mdcsum());
}

namespace {

/// nvext.use_raw_prompt handling: only the legacy completions request can
/// supply a raw prompt (as in Rust, where chat raw_prompt() is None).
std::optional<std::string> raw_prompt_of(const openai::NvCreateChatCompletionRequest&) {
  return std::nullopt;
}
std::optional<std::string> raw_prompt_of(const openai::NvCreateCompletionRequest& request) {
  return request.raw_prompt();
}

}  // namespace

OpenAIPreprocessor::OpenAIPreprocessor(std::shared_ptr<const PromptFormatter> formatter,
                                       TokenizerPtr tokenizer,
                                       std::vector<TokenIdType> eos_token_ids, std::string mdcsum)
    : formatter_(std::move(formatter)),
      tokenizer_(std::move(tokenizer)),
      eos_token_ids_(std::move(eos_token_ids)),
      mdcsum_(std::move(mdcsum)) {}

template <typename Request>
std::pair<BackendInput, std::map<std::string, std::string>> OpenAIPreprocessor::preprocess_impl(
    const Request& request) const {
  std::map<std::string, std::string> annotations;
  BackendInput input;

  bool use_raw_prompt =
      request.nvext.has_value() && request.nvext->use_raw_prompt.value_or(false);

  std::string formatted_prompt;
  if (use_raw_prompt) {
    if (auto raw = raw_prompt_of(request)) {
      formatted_prompt = std::move(*raw);
    } else {
      spdlog::warn("Raw prompt requested but not available");
      formatted_prompt = formatter_->render(chat_template_input(request));
    }
  } else {
    formatted_prompt = formatter_->render(chat_template_input(request));
  }

  Encoding encoding = tokenizer_->encode(formatted_prompt);

  if (request.has_annotation(kAnnotationFormattedPrompt)) {
    annotations[kAnnotationFormattedPrompt] = formatted_prompt;
  }
  if (request.has_annotation(kAnnotationTokenIds)) {
    annotations[kAnnotationTokenIds] = nlohmann::json(encoding.token_ids).dump();
  }

  StopConditions stop_conditions = openai::extract_stop_conditions(request);
  if (stop_conditions.stop_token_ids_hidden.has_value()) {
    for (TokenIdType eos : eos_token_ids_) {
      auto& hidden = *stop_conditions.stop_token_ids_hidden;
      if (std::find(hidden.begin(), hidden.end(), eos) == hidden.end()) hidden.push_back(eos);
    }
  } else {
    stop_conditions.stop_token_ids_hidden = eos_token_ids_;
  }
  stop_conditions.apply_ignore_eos();

  if (!stop_conditions.ignore_eos.value_or(false)) {
    input.eos_token_ids = eos_token_ids_;
  }

  input.token_ids = std::move(encoding.token_ids);
  input.sampling_options = openai::extract_sampling_options(request);
  input.stop_conditions = std::move(stop_conditions);
  input.annotations = request.annotations().value_or(std::vector<std::string>{});
  input.mdc_sum = mdcsum_;

  return {std::move(input), std::move(annotations)};
}

std::pair<BackendInput, std::map<std::string, std::string>> OpenAIPreprocessor::preprocess(
    const openai::NvCreateChatCompletionRequest& request) const {
  return preprocess_impl(request);
}

std::pair<BackendInput, std::map<std::string, std::string>> OpenAIPreprocessor::preprocess(
    const openai::NvCreateCompletionRequest& request) const {
  return preprocess_impl(request);
}

namespace {

template <typename Resp>
std::vector<pipeline::Annotated<Resp>> annotation_prologue(
    const std::map<std::string, std::string>& annotations) {
  std::vector<pipeline::Annotated<Resp>> prologue;
  prologue.reserve(annotations.size());
  for (const auto& [name, value] : annotations) {
    prologue.push_back(pipeline::Annotated<Resp>::from_annotation(name, nlohmann::json(value)));
  }
  return prologue;
}

}  // namespace

coro::Task<pipeline::ManyOut<pipeline::Annotated<openai::NvCreateChatCompletionStreamResponse>>>
OpenAIPreprocessor::generate(
    pipeline::SingleIn<openai::NvCreateChatCompletionRequest> request,
    pipeline::EnginePtr<BackendInput, pipeline::Annotated<BackendOutput>> next) {
  using Resp = openai::NvCreateChatCompletionStreamResponse;

  auto generator = openai::response_generator(request.payload());
  auto [input, annotations] = preprocess(request.payload());
  generator.update_isl(static_cast<uint32_t>(input.token_ids.size()));

  auto downstream =
      std::move(request).map([&input](openai::NvCreateChatCompletionRequest&&) mutable {
        return std::move(input);
      });
  auto out = co_await next->generate(std::move(downstream));
  pipeline::ContextPtr context = out.context();

  co_return pipeline::ManyOut<pipeline::Annotated<Resp>>(
      transform_postprocessor_stream<Resp>(std::move(out), std::move(generator),
                                           annotation_prologue<Resp>(annotations), context),
      context);
}

coro::Task<pipeline::ManyOut<pipeline::Annotated<openai::NvCreateCompletionResponse>>>
OpenAIPreprocessor::generate(
    pipeline::SingleIn<openai::NvCreateCompletionRequest> request,
    pipeline::EnginePtr<BackendInput, pipeline::Annotated<BackendOutput>> next) {
  using Resp = openai::NvCreateCompletionResponse;

  auto generator = openai::response_generator(request.payload());
  auto [input, annotations] = preprocess(request.payload());
  generator.update_isl(static_cast<int32_t>(input.token_ids.size()));

  auto downstream = std::move(request).map([&input](openai::NvCreateCompletionRequest&&) mutable {
    return std::move(input);
  });
  auto out = co_await next->generate(std::move(downstream));
  pipeline::ContextPtr context = out.context();

  co_return pipeline::ManyOut<pipeline::Annotated<Resp>>(
      transform_postprocessor_stream<Resp>(std::move(out), std::move(generator),
                                           annotation_prologue<Resp>(annotations), context),
      context);
}

}  // namespace dynamo::llm
