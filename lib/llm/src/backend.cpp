#include <dynamo/llm/backend.h>

#include <algorithm>
#include <spdlog/spdlog.h>

namespace dynamo::llm {

// ---------------------------------------------------------------------------
// Decoder implementation
// ---------------------------------------------------------------------------

Decoder::Decoder(DecoderConfig config)
    : config_(std::move(config))
    , min_tokens_remaining_(0) {}

bool Decoder::step(int32_t token_id, double logprob) {
    output_ids_.push_back(token_id);

    // Track min_tokens requirement
    if (min_tokens_remaining_ > 0) {
        --min_tokens_remaining_;
    }

    // Only check stop conditions after min_tokens are satisfied
    bool min_met = (config_.min_tokens <= 0 || min_tokens_remaining_ <= 0);

    if (min_met) {
        // Check EOS tokens
        auto is_eos = std::find(config_.eos_token_ids.begin(),
                                config_.eos_token_ids.end(),
                                token_id) != config_.eos_token_ids.end();

        // Check stop tokens
        auto is_stop = std::find(config_.stop_token_ids.begin(),
                                 config_.stop_token_ids.end(),
                                 token_id) != config_.stop_token_ids.end();

        // Check stop sequences
        bool seq_stop = check_stop_sequences(token_id);

        if (is_eos || is_stop || seq_stop) {
            finished_ = true;
            if (is_eos) finish_reason_ = "eos";
            else if (is_stop) finish_reason_ = "stop";
            else finish_reason_ = "stop_sequence";

            // Remove the stop token from output if it's EOS
            if (is_eos) {
                output_ids_.pop_back();
            }
            return true;
        }
    }

    return false;
}

void Decoder::reset() {
    output_ids_.clear();
    finished_ = false;
    finish_reason_.clear();
    jail_.clear();
    jail_len_ = 0;
    min_tokens_remaining_ = config_.min_tokens;
}

bool Decoder::check_stop_sequences(int32_t token_id) {
    if (config_.stop_sequences.empty()) return false;

    jail_.push_back(token_id);
    jail_len_++;

    for (const auto& seq : config_.stop_sequences) {
        if (seq.empty()) continue;
        // Check if jail ends with this stop sequence
        if (jail_len_ >= static_cast<int>(seq.size())) {
            bool match = true;
            for (int i = 0; i < static_cast<int>(seq.size()); ++i) {
                if (jail_[jail_len_ - seq.size() + i] != seq[i]) {
                    match = false;
                    break;
                }
            }
            if (match) return true;
        }
    }
    return false;
}

} // namespace dynamo::llm
