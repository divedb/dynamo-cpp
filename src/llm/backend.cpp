// SPDX-License-Identifier: Apache-2.0

#include "llm/backend.h"

#include <spdlog/spdlog.h>

#include "llm/model_card.h"
#include "llm/preprocessor.h"

namespace dynamo::llm {

std::shared_ptr<Backend> Backend::from_mdc(const ModelDeploymentCard& mdc) {
  return std::make_shared<Backend>(tokenizer_from_mdc(mdc), mdc.mdcsum());
}

Decoder::Decoder(TokenizerPtr tokenizer, const StopConditions& stop_conditions)
    : sequence_(std::move(tokenizer)) {
  if (stop_conditions.stop_token_ids_hidden.has_value()) {
    hidden_stop_ids_.insert(stop_conditions.stop_token_ids_hidden->begin(),
                            stop_conditions.stop_token_ids_hidden->end());
  }
  if (stop_conditions.stop.has_value()) {
    hidden_stop_sequences_ = *stop_conditions.stop;
  }
  min_tokens_ = stop_conditions.min_tokens.value_or(0);
}

std::string Decoder::flush() {
  std::string out = std::move(pending_);
  pending_.clear();
  return out;
}

StepResult Decoder::step(TokenIdType token_id) {
  ++generated_tokens_;

  std::string decoded = sequence_.append_token_id(token_id);

  // Stop conditions do not apply until min_tokens have been generated.
  if (generated_tokens_ < min_tokens_) {
    std::string out = flush() + decoded;
    return {out.empty() ? std::nullopt : std::optional<std::string>(std::move(out)),
            std::nullopt};
  }

  if (hidden_stop_ids_.count(token_id) != 0) {
    // The stop token's own text is hidden; jailed text turned out legitimate.
    std::string out = flush();
    return {out.empty() ? std::nullopt : std::optional<std::string>(std::move(out)),
            StopTrigger{StopTrigger::Kind::hidden_stop_token, token_id, {}}};
  }

  if (hidden_stop_sequences_.empty()) {
    return {decoded.empty() ? std::nullopt : std::optional<std::string>(std::move(decoded)),
            std::nullopt};
  }

  // candidate = previously jailed text + this step's text. A full stop match
  // cuts the stream (text before the match is released); otherwise the
  // longest candidate suffix that could still begin a stop sequence stays
  // jailed and everything ahead of it is released.
  std::string candidate = std::move(pending_) + decoded;
  pending_.clear();

  for (const auto& seq : hidden_stop_sequences_) {
    size_t offset = candidate.find(seq);
    if (offset == std::string::npos) continue;
    std::string released = candidate.substr(0, offset);
    return {released.empty() ? std::nullopt : std::optional<std::string>(std::move(released)),
            StopTrigger{StopTrigger::Kind::hidden_stop_sequence, 0, seq}};
  }

  size_t hold = 0;
  size_t max_hold = 0;
  for (const auto& seq : hidden_stop_sequences_) {
    max_hold = std::max(max_hold, seq.size() - 1);
  }
  size_t scan = std::min(candidate.size(), max_hold);
  for (size_t len = scan; len > 0 && hold == 0; --len) {
    std::string_view suffix(candidate.data() + candidate.size() - len, len);
    for (const auto& seq : hidden_stop_sequences_) {
      if (seq.size() > len && std::string_view(seq).substr(0, len) == suffix) {
        hold = len;
        break;
      }
    }
  }

  std::string released = candidate.substr(0, candidate.size() - hold);
  pending_ = candidate.substr(candidate.size() - hold);
  return {released.empty() ? std::nullopt : std::optional<std::string>(std::move(released)),
          std::nullopt};
}

SeqResult Decoder::process_token_ids(const std::vector<TokenIdType>& token_ids) {
  SeqResult result;
  for (TokenIdType token_id : token_ids) {
    StepResult step_result = step(token_id);

    if (step_result.token.has_value()) {
      if (!result.text.has_value()) result.text = std::string();
      *result.text += *step_result.token;
    }
    result.tokens.push_back(std::move(step_result.token));

    if (step_result.stop_trigger.has_value()) {
      result.stop_trigger = std::move(step_result.stop_trigger);
      return result;
    }
  }
  return result;
}

namespace {

BackendOutput to_backend_output(LLMEngineOutput data, const std::string& mdcsum) {
  BackendOutput out;
  out.token_ids = std::move(data.token_ids);
  out.tokens = data.tokens.has_value() ? std::move(*data.tokens) : std::vector<TokenType>{};
  out.text = std::move(data.text);
  out.cum_log_probs = data.cum_log_probs;
  out.log_probs = std::move(data.log_probs);
  out.finish_reason = std::move(data.finish_reason);
  out.mdcsum = mdcsum;
  return out;
}

coro::AsyncGenerator<pipeline::Annotated<BackendOutput>> decode_stream(
    pipeline::ManyOut<pipeline::Annotated<LLMEngineOutput>> in, Decoder decoder,
    std::string mdcsum, bool validate_engine_decode, pipeline::ContextPtr context) {
  bool finished = false;
  while (auto item = co_await in.next()) {
    pipeline::Annotated<BackendOutput> out;
    out.id = std::move(item->id);
    out.event = std::move(item->event);
    out.comment = std::move(item->comment);

    // Events and data-less items pass through.
    if (!item->data.has_value() || out.event.has_value()) {
      if (item->data.has_value()) {
        out.data = to_backend_output(std::move(*item->data), mdcsum);
      }
      co_yield out;
      continue;
    }

    LLMEngineOutput data = std::move(*item->data);

    // Engines that decode their own text pass through untouched.
    if (data.text.has_value() && !validate_engine_decode) {
      out.data = to_backend_output(std::move(data), mdcsum);
      co_yield out;
      continue;
    }

    SeqResult result = decoder.process_token_ids(data.token_ids);

    std::optional<FinishReason> decoder_reason;
    if (result.stop_trigger.has_value()) {
      decoder_reason = result.stop_trigger->finish_reason();
    }

    if (!data.finish_reason.has_value() && decoder_reason.has_value()) {
      spdlog::debug(
          "upstream did not provide a finish reason; issuing stop_generating to free resources");
      context->stop_generating();
    }

    if (validate_engine_decode) {
      if (data.finish_reason != decoder_reason) {
        spdlog::warn("finish reason mismatch between engine and decoder");
      }
      if (data.text.has_value() && data.text != result.text) {
        spdlog::warn("text mismatch: engine '{}' vs decoder '{}'", data.text.value_or(""),
                     result.text.value_or(""));
      }
    }

    // Deviation from Rust v0.1.0, which overwrites the engine's finish reason
    // with the decoder's (dropping e.g. an engine-issued Length on token-only
    // deltas): the engine's own reason wins when the decoder found none.
    if (decoder_reason.has_value()) data.finish_reason = decoder_reason;

    // A finishing delta releases any jailed text (it was not a stop after
    // all) — unless a stop-sequence trigger hid it on purpose.
    if (data.finish_reason.has_value() && !result.stop_trigger.has_value()) {
      std::string flushed = decoder.flush();
      if (!flushed.empty()) {
        if (!result.text.has_value()) result.text = std::string();
        *result.text += flushed;
      }
    }

    data.text = std::move(result.text);
    data.tokens = std::move(result.tokens);

    bool stop_now = result.stop_trigger.has_value();
    if (data.finish_reason.has_value()) finished = true;
    out.data = to_backend_output(std::move(data), mdcsum);
    co_yield out;

    if (stop_now) break;  // the stop trigger ends the logical stream
  }

  // Stream ended without any finishing delta: release jailed text.
  if (!finished) {
    std::string flushed = decoder.flush();
    if (!flushed.empty()) {
      BackendOutput tail;
      tail.text = std::move(flushed);
      tail.mdcsum = mdcsum;
      auto out = pipeline::Annotated<BackendOutput>::from_data(std::move(tail));
      co_yield out;
    }
  }
}

}  // namespace

coro::Task<pipeline::ManyOut<pipeline::Annotated<BackendOutput>>> Backend::generate(
    pipeline::SingleIn<BackendInput> request, ExecutionContext next) {
  StopConditions stop_conditions = request.payload().stop_conditions;
  auto out = co_await next->generate(std::move(request));
  pipeline::ContextPtr context = out.context();

  Decoder decoder(tokenizer_, stop_conditions);
  co_return pipeline::ManyOut<pipeline::Annotated<BackendOutput>>(
      decode_stream(std::move(out), std::move(decoder), mdcsum_, validate_engine_decode_,
                    context),
      context);
}

}  // namespace dynamo::llm
