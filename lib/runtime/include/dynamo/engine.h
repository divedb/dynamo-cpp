#pragma once

#include <dynamo/cancellation.h>

#include <memory>
#include <functional>
#include <exception>
#include <folly/coro/Task.h>
#include <folly/io/async/EventBase.h>

namespace dynamo {

class AsyncEngineContext {
public:
    AsyncEngineContext(CancellationToken token = CancellationToken::never())
        : token_(std::move(token)) {}

    bool is_stopped() const noexcept { return stopped_; }
    bool is_killed() const noexcept { return killed_; }

    void stop_generating() noexcept { stopped_ = true; }
    void kill() noexcept { killed_ = true; stop_generating(); }

    bool is_cancelled() const noexcept {
        return killed_ || token_.is_cancelled();
    }

    const CancellationToken& cancellation_token() const noexcept {
        return token_;
    }

private:
    bool stopped_ = false;
    bool killed_ = false;
    CancellationToken token_;
};

template <typename T>
class ResponseStream {
public:
    using Callback = std::function<folly::coro::Task<void>(T)>;
    using DoneCallback = std::function<void()>;
    using ErrorCallback = std::function<void(std::exception_ptr)>;

    ResponseStream(
        std::shared_ptr<AsyncEngineContext> ctx,
        Callback on_data,
        DoneCallback on_done = nullptr,
        ErrorCallback on_error = nullptr)
        : ctx_(std::move(ctx))
        , on_data_(std::move(on_data))
        , on_done_(std::move(on_done))
        , on_error_(std::move(on_error)) {}

    folly::coro::Task<void> push(T value) {
        if (ctx_->is_cancelled()) {
            co_return;
        }
        if (on_data_) {
            co_await on_data_(std::move(value));
        }
    }

    void complete() {
        if (on_done_) {
            on_done_();
        }
    }

    void error(std::exception_ptr e) {
        if (on_error_) {
            on_error_(std::move(e));
        }
    }

    bool is_active() const noexcept {
        return ctx_ && !ctx_->is_cancelled();
    }

    std::shared_ptr<AsyncEngineContext> context() const noexcept {
        return ctx_;
    }

private:
    std::shared_ptr<AsyncEngineContext> ctx_;
    Callback on_data_;
    DoneCallback on_done_;
    ErrorCallback on_error_;
};

template <typename Req, typename Resp, typename E = std::exception_ptr>
class AsyncEngine {
public:
    virtual ~AsyncEngine() = default;

    virtual folly::coro::Task<void> generate(
        Req request,
        std::shared_ptr<ResponseStream<Resp>> response_stream) = 0;
};

} // namespace dynamo
