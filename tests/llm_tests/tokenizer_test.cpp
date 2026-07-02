#include <catch2/catch_test_macros.hpp>
#include <dynamo/llm/tokenizer.h>

using namespace dynamo::llm;

TEST_CASE("SimpleTokenizer encodes and decodes", "[llm][tokenizer]") {
    SimpleTokenizer tokenizer("test", 32000);
    auto text = std::string("Hello!");

    auto ids = tokenizer.encode(text);
    CHECK_FALSE(ids.empty());
    CHECK(ids[0] == tokenizer.bos_token_id());  // BOS added

    auto decoded = tokenizer.decode(ids);
    CHECK(decoded == "Hello!");
}

TEST_CASE("SimpleTokenizer skips special tokens", "[llm][tokenizer]") {
    SimpleTokenizer tokenizer("test", 32000);

    std::vector<int32_t> ids = {0, 1, 2, 42, 73};
    auto decoded = tokenizer.decode(ids, true);
    // 0, 1, 2 are special (PAD, BOS, EOS) and should be skipped
    CHECK(decoded.size() == 2);
}

TEST_CASE("Tokenizer properties", "[llm][tokenizer]") {
    SimpleTokenizer tokenizer("my-tokenizer", 50000);

    CHECK(tokenizer.name() == "my-tokenizer");
    CHECK(tokenizer.vocab_size() == 50000);
    CHECK(tokenizer.bos_token_id() == 1);
    CHECK(tokenizer.eos_token_id() == 2);
    CHECK(tokenizer.pad_token_id() == 0);
}
