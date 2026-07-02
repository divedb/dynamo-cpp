#pragma once

#include <dynamo/engine.h>
#include <dynamo/cancellation.h>

#include <memory>
#include <any>
#include <string>
#include <unordered_map>
#include <functional>
#include <folly/coro/Task.h>

namespace dynamo {

// ---------------------------------------------------------------------------
// Context<T> — input value with metadata registry + lifecycle controller
// ---------------------------------------------------------------------------

class ContextRegistry {
public:
    template <typename T>
    void insert(std::string key, T value) {
        store_[std::move(key)] = std::make_shared<AnyWrapper<T>>(std::move(value));
    }

    template <typename T>
    T* get(const std::string& key) const {
        auto it = store_.find(key);
        if (it == store_.end()) return nullptr;
        auto* wrapper = dynamic_cast<AnyWrapper<T>*>(it->second.get());
        return wrapper ? &wrapper->value : nullptr;
    }

private:
    struct WrapperBase {
        virtual ~WrapperBase() = default;
    };
    template <typename T>
    struct AnyWrapper : WrapperBase {
        explicit AnyWrapper(T v) : value(std::move(v)) {}
        T value;
    };
    std::unordered_map<std::string, std::shared_ptr<WrapperBase>> store_;
};

template <typename T>
class Context {
public:
    explicit Context(T value)
        : value_(std::move(value))
        , registry_(std::make_shared<ContextRegistry>()) {}

    Context(T value, std::shared_ptr<AsyncEngineContext> ctx)
        : value_(std::move(value))
        , ctx_(std::move(ctx))
        , registry_(std::make_shared<ContextRegistry>()) {}

    T& value() & noexcept { return value_; }
    const T& value() const& noexcept { return value_; }
    T&& take_value() && noexcept { return std::move(value_); }

    AsyncEngineContext& context() noexcept { return *ctx_; }
    const AsyncEngineContext& context() const noexcept { return *ctx_; }

    std::shared_ptr<AsyncEngineContext> context_ptr() const noexcept {
        return ctx_;
    }

    ContextRegistry& registry() noexcept { return *registry_; }
    const ContextRegistry& registry() const noexcept { return *registry_; }

    template <typename U>
    Context<U> map(std::function<U(const T&)> fn) const {
        auto new_ctx = Context<U>(fn(value_), ctx_);
        new_ctx.registry_ = registry_;
        return new_ctx;
    }

    template <typename U>
    Context<U> transfer(U new_value) const {
        auto new_ctx = Context<U>(std::move(new_value), ctx_);
        new_ctx.registry_ = registry_;
        return new_ctx;
    }

    bool is_cancelled() const noexcept {
        return ctx_ && ctx_->is_cancelled();
    }

private:
    T value_;
    std::shared_ptr<AsyncEngineContext> ctx_;
    std::shared_ptr<ContextRegistry> registry_;
};

// ---------------------------------------------------------------------------
// Pipeline type aliases  (matching Dynamo's SingleIn/ManyIn, SingleOut/ManyOut)
// ---------------------------------------------------------------------------

template <typename T>
using SingleIn = Context<T>;

template <typename T>
using ManyIn = Context<folly::coro::AsyncGenerator<T>>;

template <typename T>
using SingleOut = folly::coro::Task<T>;

template <typename T>
using ManyOut = std::shared_ptr<ResponseStream<T>>;

// Engine convenience aliases
template <typename T, typename U>
using UnaryEngine = AsyncEngine<SingleIn<T>, SingleOut<U>>;

template <typename T, typename U>
using ServerStreamingEngine = AsyncEngine<SingleIn<T>, ManyOut<U>>;

template <typename T, typename U>
using ClientStreamingEngine = AsyncEngine<ManyIn<T>, SingleOut<U>>;

template <typename T, typename U>
using BidirectionalStreamingEngine = AsyncEngine<ManyIn<T>, ManyOut<U>>;

// ---------------------------------------------------------------------------
// Pipeline nodes: Source, Sink, Edge, Operator
// ---------------------------------------------------------------------------

template <typename T>
class Source {
public:
    using OutputFn = std::function<folly::coro::Task<void>(T)>;

    explicit Source(OutputFn fn) : output_(std::move(fn)) {}
    virtual ~Source() = default;

    folly::coro::Task<void> emit(T value) {
        if (output_) {
            co_await output_(std::move(value));
        }
    }

protected:
    OutputFn output_;
};

template <typename T>
class Sink {
public:
    virtual ~Sink() = default;
    virtual folly::coro::Task<void> on_data(T value) = 0;
};

template <typename T>
class Edge {
public:
    Edge(std::shared_ptr<Source<T>> source, std::shared_ptr<Sink<T>> sink)
        : source_(std::move(source)), sink_(std::move(sink)) {}

    std::shared_ptr<Source<T>> source() const noexcept { return source_; }
    std::shared_ptr<Sink<T>> sink() const noexcept { return sink_; }

    folly::coro::Task<void> connect() {
        source_->output_ = [sink = sink_](T value) -> folly::coro::Task<void> {
            co_await sink->on_data(std::move(value));
        };
        co_return;
    }

private:
    std::shared_ptr<Source<T>> source_;
    std::shared_ptr<Sink<T>> sink_;
};

template <typename UpIn, typename UpOut, typename DownIn, typename DownOut>
class Operator {
public:
    virtual ~Operator() = default;

    // Forward transform (request path)
    virtual folly::coro::Task<DownIn> forward(UpIn input) = 0;

    // Backward transform (response path)
    virtual folly::coro::Task<UpOut> backward(DownOut output) = 0;
};

template <typename In, typename Out>
class PipelineNode : public AsyncEngine<In, Out> {
public:
    using TransformFn = std::function<folly::coro::Task<Out>(In)>;

    explicit PipelineNode(TransformFn fn) : transform_(std::move(fn)) {}

    folly::coro::Task<void> generate(
        In request,
        std::shared_ptr<ResponseStream<Out>> response_stream) override {
        try {
            auto result = co_await transform_(std::move(request));
            co_await response_stream->push(std::move(result));
            response_stream->complete();
        } catch (...) {
            response_stream->error(std::current_exception());
        }
    }

private:
    TransformFn transform_;
};

} // namespace dynamo
