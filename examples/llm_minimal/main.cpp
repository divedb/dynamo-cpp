#include <dynamo/runtime.h>
#include <dynamo/llm/protocols.h>
#include <dynamo/llm/tokenizer.h>
#include <dynamo/llm/backend.h>
#include <dynamo/llm/router.h>
#include <spdlog/spdlog.h>

using namespace dynamo;
using namespace dynamo::llm;

int main() {
    spdlog::set_level(spdlog::level::info);

    auto runtime = Runtime::create(RuntimeConfig::load());

    // ---- Tokenizer ----
    auto tokenizer = std::make_shared<SimpleTokenizer>("demo", 32000);
    auto input_text = "Hello, Dynamo!";
    auto input_ids = tokenizer->encode(input_text);
    spdlog::info("Tokenized '{}' -> {} tokens", input_text, input_ids.size());

    // ---- Decoder ----
    DecoderConfig dec_cfg;
    dec_cfg.eos_token_ids = {2};
    dec_cfg.stop_token_ids = {};
    Decoder decoder(dec_cfg);

    // Simulate generation
    std::vector<int32_t> fake_output = {42, 73, 15, 88, 2};  // ends with EOS
    for (auto tok : fake_output) {
        bool stop = decoder.step(tok, -0.5);
        spdlog::info("Token: {} (finished={}, reason='{}')",
                     tok, decoder.finished(), decoder.finish_reason());
        if (stop) break;
    }

    // Decode output
    auto decoded = tokenizer->decode(decoder.output_ids());
    spdlog::info("Decoded output: '{}'", decoded);

    // ---- KV Router ----
    auto indexer = std::make_shared<KvIndexer>();
    KvRouter router(indexer);

    // Add sequences to the index
    indexer->add_sequence(0, {1, 42, 73, 15});
    indexer->add_sequence(1, {1, 99, 88, 77});

    // Schedule
    std::vector<WorkerMetadata> workers = {
        {"0", "10.0.0.1", 50051, 0.3, 50},
        {"1", "10.0.0.2", 50051, 0.1, 30},
    };
    auto chosen = router.schedule({1, 42, 73}, workers).get();
    spdlog::info("Router chose worker {}", chosen);

    // ---- Disaggregated Router ----
    DisaggregatedRouter disagg_router;
    bool remote = disagg_router.prefill_remote(8192, 100, 3);
    spdlog::info("Disagg: prefill_remote({}, {}, {}) = {}",
                 8192, 100, 3, remote);

    // ---- Completion Request/Response ----
    CompletionRequest req;
    req.model = "test-model";
    req.prompt = "Hello";
    req.params.max_tokens = 100;

    CompletionResponse resp;
    resp.request_id = "test-123";
    resp.token_id = 42;
    resp.logprob = -0.5;
    resp.finished = false;

    spdlog::info("LLM minimal example completed successfully.");

    runtime->shutdown();
    return 0;
}
