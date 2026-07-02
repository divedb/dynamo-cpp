#include <dynamo/runtime.h>
#include <dynamo/engine.h>
#include <dynamo/pipeline.h>
#include <spdlog/spdlog.h>

#include <folly/coro/Task.h>
#include <folly/coro/BlockingWait.h>
#include <folly/futures/Future.h>

using namespace dynamo;

// A simple engine that echoes the input
class EchoEngine : public AsyncEngine<std::string, std::string> {
public:
    folly::coro::Task<void> generate(
        std::string request,
        std::shared_ptr<ResponseStream<std::string>> response_stream) override {
        spdlog::info("EchoEngine received: '{}'", request);
        auto echoed = "echo: " + request;
        co_await response_stream->push(std::move(echoed));
        response_stream->complete();
    }
};

int main() {
    spdlog::set_level(spdlog::level::info);

    auto runtime = Runtime::create(RuntimeConfig::load());
    auto engine = std::make_shared<EchoEngine>();

    // Create a pipeline node
    auto node = PipelineNode<std::string, std::string>(
        [](std::string input) -> folly::coro::Task<std::string> {
            co_return "pipeline: " + std::move(input);
        });

    spdlog::info("Running echo pipeline...");

    // Run a request through the engine
    auto ctx = std::make_shared<AsyncEngineContext>();
    auto stream = std::make_shared<ResponseStream<std::string>>(
        ctx,
        [](std::string resp) -> folly::coro::Task<void> {
            spdlog::info("Response: {}", resp);
            co_return;
        });

    // Execute
    auto result = engine->generate("hello dynamo", stream);
    std::move(result).scheduleOn(&runtime->primary()).start();

    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    // Test pipeline node
    auto node_task = node.generate(
        Context<std::string>("pipeline test"),
        stream);
    std::move(node_task).scheduleOn(&runtime->primary()).start();

    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    runtime->shutdown();
    spdlog::info("Echo pipeline done.");
    return 0;
}
