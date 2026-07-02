#include <catch2/catch_test_macros.hpp>
#include <dynamo/llm/backend.h>

using namespace dynamo::llm;

TEST_CASE("Decoder stops on EOS token", "[llm][backend]") {
    DecoderConfig cfg;
    cfg.eos_token_ids = {2};
    Decoder decoder(cfg);

    CHECK_FALSE(decoder.step(42, 0.0));
    CHECK_FALSE(decoder.finished());

    CHECK_FALSE(decoder.step(73, -0.5));
    CHECK_FALSE(decoder.finished());

    CHECK(decoder.step(2, 0.0));  // EOS
    CHECK(decoder.finished());
    CHECK(decoder.finish_reason() == "eos");
}

TEST_CASE("Decoder stops on stop token", "[llm][backend]") {
    DecoderConfig cfg;
    cfg.stop_token_ids = {99};
    Decoder decoder(cfg);

    CHECK(decoder.step(99, 0.0));
    CHECK(decoder.finished());
    CHECK(decoder.finish_reason() == "stop");
}

TEST_CASE("Decoder tracks output tokens", "[llm][backend]") {
    DecoderConfig cfg;
    Decoder decoder(cfg);

    decoder.step(10, -0.1);
    decoder.step(20, -0.2);
    decoder.step(30, -0.3);

    CHECK(decoder.num_generated_tokens() == 3);
    CHECK(decoder.output_ids().size() == 3);
    CHECK(decoder.output_ids()[0] == 10);
    CHECK(decoder.output_ids()[2] == 30);
}

TEST_CASE("Decoder resets state", "[llm][backend]") {
    DecoderConfig cfg;
    Decoder decoder(cfg);

    decoder.step(1, 0.0);
    CHECK(decoder.num_generated_tokens() == 1);

    decoder.reset();
    CHECK(decoder.num_generated_tokens() == 0);
    CHECK_FALSE(decoder.finished());
}

TEST_CASE("Decoder handles min_tokens", "[llm][backend]") {
    DecoderConfig cfg;
    cfg.eos_token_ids = {2};
    cfg.min_tokens = 3;
    Decoder decoder(cfg);

    CHECK_FALSE(decoder.step(10, 0.0));  // token 1
    CHECK_FALSE(decoder.step(20, 0.0));  // token 2
    CHECK_FALSE(decoder.step(2, 0.0));   // EOS but before min_tokens
    CHECK_FALSE(decoder.finished());     // should NOT finish because min_tokens

    CHECK_FALSE(decoder.step(30, 0.0));  // token 3 = min_tokens reached
    CHECK(decoder.step(2, 0.0));         // EOS now respected
    CHECK(decoder.finished());
}
