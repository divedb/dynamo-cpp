#include <catch2/catch_test_macros.hpp>
#include <dynamo/llm/protocols.h>
#include <nlohmann/json.hpp>

using namespace dynamo::llm;

TEST_CASE("GenerationParams serializes to JSON", "[llm][protocols]") {
    GenerationParams p;
    p.max_tokens = 512;
    p.temperature = 0.8;
    p.top_p = 0.95;
    p.top_k = 40;
    p.stream = true;
    p.seed = 42;
    p.stop_token_ids = {2, 3};

    nlohmann::json j = p;
    CHECK(j["max_tokens"] == 512);
    CHECK(j["temperature"] == 0.8);
    CHECK(j["top_p"] == 0.95);
    CHECK(j["top_k"] == 40);
    CHECK(j["stream"] == true);
    CHECK(j["seed"] == 42);
    CHECK(j["stop_token_ids"].size() == 2);
}

TEST_CASE("GenerationParams deserializes from JSON", "[llm][protocols]") {
    nlohmann::json j = {
        {"max_tokens", 128},
        {"temperature", 0.5},
        {"top_p", 0.9},
        {"top_k", 50},
        {"stream", false},
    };

    auto p = j.get<GenerationParams>();
    CHECK(p.max_tokens == 128);
    CHECK(p.temperature == 0.5);
    CHECK(p.top_p == 0.9);
    CHECK(p.top_k == 50);
    CHECK(p.stream == false);
}

TEST_CASE("Annotated data envelope", "[llm][protocols]") {
    auto ann = Annotated<std::string>::make_data("hello");
    CHECK(ann.has_data());
    CHECK_FALSE(ann.is_event());
    CHECK(*ann.data == "hello");
}

TEST_CASE("Annotated event envelope", "[llm][protocols]") {
    auto ann = Annotated<int>::make_event("done");
    CHECK_FALSE(ann.has_data());
    CHECK(ann.is_event());
    CHECK(*ann.event == "done");
}

TEST_CASE("CompletionRequest round-trips JSON", "[llm][protocols]") {
    CompletionRequest req;
    req.model = "test-model";
    req.prompt = "Hello world";
    req.params.max_tokens = 100;

    nlohmann::json j = req;
    auto req2 = j.get<CompletionRequest>();
    CHECK(req2.model == "test-model");
    CHECK(req2.prompt == "Hello world");
    CHECK(req2.params.max_tokens == 100);
}
