#pragma once

#include <dynamo/config.h>
#include <dynamo/cancellation.h>

#include <memory>
#include <string>
#include <functional>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/executors/IOThreadPoolExecutor.h>
#include <folly/futures/Future.h>
#include <folly/coro/Task.h>
#include <folly/coro/Sleep.h>
#include <folly/ScopeGuard.h>

namespace dynamo {

class Runtime {
public:
    explicit Runtime(const RuntimeConfig& cfg = RuntimeConfig::load());
    ~Runtime();

    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;
    Runtime(Runtime&&) = delete;
    Runtime& operator=(Runtime&&) = delete;

    folly::CPUThreadPoolExecutor& primary() noexcept { return *primary_; }
    folly::IOThreadPoolExecutor& io() noexcept { return io_; }
    folly::CPUThreadPoolExecutor& secondary() noexcept { return *secondary_; }

    const CancellationToken& cancellation_token() const noexcept {
        return cancellation_token_;
    }
    CancellationToken& cancellation_token() noexcept {
        return cancellation_token_;
    }

    void shutdown();

    const std::string& id() const noexcept { return id_; }
    const RuntimeConfig& config() const noexcept { return config_; }

    static std::shared_ptr<Runtime> create(
        const RuntimeConfig& cfg = RuntimeConfig::load());

private:
    std::string id_;
    RuntimeConfig config_;
    CancellationToken cancellation_token_;
    std::unique_ptr<folly::CPUThreadPoolExecutor> primary_;
    folly::IOThreadPoolExecutor io_;
    std::unique_ptr<folly::CPUThreadPoolExecutor> secondary_;
};

} // namespace dynamo
