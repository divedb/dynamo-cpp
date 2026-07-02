#pragma once

#include <atomic>
#include <memory>
#include <functional>

namespace dynamo {

class CancellationToken {
public:
    CancellationToken() = default;

    CancellationToken(const CancellationToken& other) noexcept
        : state_(other.state_) {}

    CancellationToken& operator=(const CancellationToken& other) noexcept {
        if (this != &other) {
            state_ = other.state_;
        }
        return *this;
    }

    bool is_cancelled() const noexcept {
        return state_ && state_->cancelled.load(std::memory_order_acquire);
    }

    void cancel() noexcept {
        if (state_) {
            state_->cancelled.store(true, std::memory_order_release);
            auto cb = std::move(state_->callback);
            if (cb) {
                cb();
            }
        }
    }

    void on_cancel(std::function<void()> callback) {
        if (state_) {
            state_->callback = std::move(callback);
        }
    }

    static CancellationToken never() {
        auto tok = CancellationToken{};
        return tok;
    }

    static CancellationToken create_source() {
        auto tok = CancellationToken{};
        tok.state_ = std::make_shared<State>();
        return tok;
    }

    explicit operator bool() const noexcept {
        return state_ != nullptr;
    }

private:
    struct State {
        std::atomic<bool> cancelled{false};
        std::function<void()> callback;
    };
    std::shared_ptr<State> state_;
};

} // namespace dynamo
