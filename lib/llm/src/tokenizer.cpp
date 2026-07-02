#include <dynamo/llm/tokenizer.h>

#include <algorithm>
#include <numeric>
#include <spdlog/spdlog.h>

namespace dynamo::llm {

SimpleTokenizer::SimpleTokenizer(std::string name, int vocab_size)
    : name_(std::move(name)), vocab_size_(vocab_size) {
    spdlog::debug("SimpleTokenizer '{}' created with vocab_size={}",
                  name_, vocab_size_);
}

std::vector<int32_t> SimpleTokenizer::encode(
    std::string_view text, bool add_special_tokens) {
    // Simple character-level tokenization for demo purposes
    std::vector<int32_t> ids;
    if (add_special_tokens) {
        ids.push_back(bos_token_id());
    }
    // Map each character to a token ID (offset from special tokens)
    for (char c : text) {
        // Simple hash-like mapping for demonstration
        int32_t id = 3 + (static_cast<int32_t>(c) % (vocab_size_ - 3));
        ids.push_back(id);
    }
    return ids;
}

std::string SimpleTokenizer::decode(
    const std::vector<int32_t>& ids, bool skip_special_tokens) {
    std::string result;
    for (int32_t id : ids) {
        if (skip_special_tokens && id < 3) {
            continue; // skip BOS=1, EOS=2, PAD=0
        }
        // Reverse the character encoding
        char c = static_cast<char>((id - 3) % 128);
        result.push_back(c);
    }
    return result;
}

} // namespace dynamo::llm
