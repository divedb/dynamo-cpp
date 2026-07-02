#include <dynamo/runtime.h>

#include <random>
#include <spdlog/spdlog.h>

namespace dynamo {

static std::string generate_id() {
    static const char chars[] = "abcdef0123456789";
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, sizeof(chars) - 2);
    std::string id(16, ' ');
    for (auto& c : id) {
        c = chars[dis(gen)];
    }
    return id;
}

Runtime::Runtime(const RuntimeConfig& cfg)
    : id_(generate_id())
    , config_(cfg)
    , cancellation_token_(CancellationToken::create_source())
    , io_(cfg.num_worker_threads)
{
    primary_ = std::make_unique<folly::CPUThreadPoolExecutor>(
        cfg.num_worker_threads,
        std::make_shared<folly::NamedThreadFactory>("dyn-primary"));

    secondary_ = std::make_unique<folly::CPUThreadPoolExecutor>(
        std::max(2, cfg.num_worker_threads / 4),
        std::make_shared<folly::NamedThreadFactory>("dyn-secondary"));

    spdlog::info("Runtime {} started: {} primary threads, {} io threads",
                 id_, cfg.num_worker_threads, cfg.num_worker_threads);
}

Runtime::~Runtime() {
    shutdown();
}

void Runtime::shutdown() {
    if (cancellation_token_) {
        cancellation_token_.cancel();
    }
    if (primary_) {
        primary_->join();
        primary_.reset();
    }
    if (secondary_) {
        secondary_->join();
        secondary_.reset();
    }
    spdlog::info("Runtime {} shut down", id_);
}

std::shared_ptr<Runtime> Runtime::create(const RuntimeConfig& cfg) {
    return std::make_shared<Runtime>(cfg);
}

} // namespace dynamo
