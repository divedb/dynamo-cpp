#pragma once

#include <string>
#include <vector>
#include <memory>
#include <functional>

namespace dynamo::llm {

class Tokenizer {
public:
    virtual ~Tokenizer() = default;

    virtual std::vector<int32_t> encode(
        std::string_view text, bool add_special_tokens = true) = 0;

    virtual std::string decode(
        const std::vector<int32_t>& ids, bool skip_special_tokens = true) = 0;

    virtual int32_t vocab_size() const noexcept = 0;
    virtual int32_t bos_token_id() const noexcept = 0;
    virtual int32_t eos_token_id() const noexcept = 0;
    virtual int32_t pad_token_id() const noexcept = 0;

    virtual std::string_view name() const noexcept = 0;
};

// Simple BPE tokenizer implementation for testing/demo
class SimpleTokenizer : public Tokenizer {
public:
    SimpleTokenizer(std::string name, int vocab_size = 32000);

    std::vector<int32_t> encode(
        std::string_view text, bool add_special_tokens = true) override;

    std::string decode(
        const std::vector<int32_t>& ids,
        bool skip_special_tokens = true) override;

    int32_t vocab_size() const noexcept override { return vocab_size_; }
    int32_t bos_token_id() const noexcept override { return 1; }
    int32_t eos_token_id() const noexcept override { return 2; }
    int32_t pad_token_id() const noexcept override { return 0; }

    std::string_view name() const noexcept override { return name_; }

private:
    std::string name_;
    int vocab_size_;
};

} // namespace dynamo::llm
