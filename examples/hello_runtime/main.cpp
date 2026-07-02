#include <dynamo/runtime.h>
#include <dynamo/config.h>
#include <spdlog/spdlog.h>
#include <folly/coro/Task.h>
#include <folly/futures/Future.h>

using namespace dynamo;

folly::coro::Task<void> worker_task(std::shared_ptr<Runtime> rt) {
    spdlog::info("Worker task running on runtime {}", rt->id());
    co_return;
}

int main() {
    spdlog::set_level(spdlog::level::info);

    // Load config with defaults (env vars override)
    auto cfg = RuntimeConfig::load();
    cfg.num_worker_threads = 4;

    // Create and start the runtime
    auto runtime = Runtime::create(cfg);
    spdlog::info("Hello Dynamo Runtime!");

    // Demonstrate running a coroutine on the primary executor
    auto task = worker_task(runtime);
    std::move(task).scheduleOn(&runtime->primary()).start();

    // Give it time to run
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Graceful shutdown
    runtime->shutdown();
    spdlog::info("Done.");

    return 0;
}
