#pragma once

#include <dynamo/engine.h>
#include <dynamo/pipeline.h>
#include <dynamo/llm/protocols.h>

#include <memory>
#include <functional>
#include <folly/coro/Task.h>

namespace dynamo::llm {

// ---------------------------------------------------------------------------
// BackendInput / BackendOutput
// ---------------------------------------------------------------------------

struct BackendInput {
    std::vector<int32_t> input_ids;
    GenerationParams params;
};

struct BackendOutput {
    std::vector<int32_t> output_ids;
    std::vector<double> logprobs;
    bool finished = false;
    std::string finish_reason;
};

// ---------------------------------------------------------------------------
// Backend — the final pipeline stage for LLM execution
// ---------------------------------------------------------------------------

class Backend {
public:
    virtual ~Backend() = default;

    virtual folly::coro::Task<void> generate(
        BackendInput input,
        std::shared_ptr<ResponseStream<BackendOutput>> stream) = 0;

    virtual std::string_view model_name() const noexcept = 0;
};

// ---------------------------------------------------------------------------
// Decoder — token-by-token decoding with stop condition detection
// ---------------------------------------------------------------------------

struct DecoderConfig {
    std::vector<int32_t> eos_token_ids{2};   // default </s>
    std::vector<int32_t> stop_token_ids;
    std::vector<std::vector<int32_t>> stop_sequences;
    int32_t pad_token_id = 0;
    bool skip_special_tokens = true;
};

class Decoder {
public:
    explicit Decoder(DecoderConfig config = {});

    // Process a single token, check stop conditions
    // Returns true if generation should stop
    bool step(int32_t token_id, double logprob);

    // Access decoded state
    const std::vector<int32_t>& output_ids() const noexcept {
        return output_ids_;
    }

    bool finished() const noexcept { return finished_; }
    std::string finish_reason() const noexcept { return finish_reason_; }
    int num_generated_tokens() const noexcept {
        return static_cast<int>(output_ids_.size());
    }

    void reset();

private:
    DecoderConfig config_;
    std::vector<int32_t> output_ids_;
    bool finished_ = false;
    std::string finish_reason_;
    int min_tokens_remaining_ = 0;

    // Partial stop sequence matching (jail mechanism)
    std::vector<int32_t> jail_;
    int jail_len_ = 0;

    bool check_stop_sequences(int32_t token_id);
};

} // namespace dynamo::llm
