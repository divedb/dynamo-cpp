// SPDX-License-Identifier: Apache-2.0
//
// The execution model: AsyncEngine (generate-style interface), per-request
// Context with a remotely-controllable Controller (stop/kill), and streaming
// responses as AsyncGenerator + context.

#pragma once

#include <atomic>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <string>
#include <typeindex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "runtime/coro/async_generator.h"
#include "runtime/coro/event.h"
#include "runtime/coro/task.h"

namespace dynamo::pipeline {

/// Control surface carried by every in-flight request/response stream.
/// `stop` asks the engine to stop producing (drain allowed); `kill` asks for
/// immediate termination without draining.
class AsyncEngineContext {
 public:
  virtual ~AsyncEngineContext() = default;

  virtual const std::string& id() const = 0;
  virtual bool is_stopped() const = 0;
  virtual bool is_killed() const = 0;

  /// Awaitables resolving when stop/kill is requested.
  virtual coro::AsyncEvent& stopped_event() = 0;
  virtual coro::AsyncEvent& killed_event() = 0;

  virtual void stop_generating() = 0;
  void stop() { stop_generating(); }
  virtual void kill() = 0;
};

using ContextPtr = std::shared_ptr<AsyncEngineContext>;

/// Default AsyncEngineContext implementation. Hooks allow transports to relay
/// stop/kill signals across the network.
class Controller final : public AsyncEngineContext {
 public:
  Controller() : id_(generate_id()) {}
  explicit Controller(std::string id) : id_(std::move(id)) {}

  const std::string& id() const override { return id_; }
  bool is_stopped() const override { return stopped_.load(std::memory_order_acquire); }
  bool is_killed() const override { return killed_.load(std::memory_order_acquire); }
  coro::AsyncEvent& stopped_event() override { return stopped_event_; }
  coro::AsyncEvent& killed_event() override { return killed_event_; }

  void stop_generating() override {
    if (stopped_.exchange(true, std::memory_order_acq_rel)) return;
    stopped_event_.set();
    run_hooks(stop_hooks_);
  }

  void kill() override {
    bool was_killed = killed_.exchange(true, std::memory_order_acq_rel);
    stop_generating();
    if (was_killed) return;
    killed_event_.set();
    run_hooks(kill_hooks_);
  }

  /// Registers a hook fired once on stop/kill (fires immediately if already
  /// signalled). Used by transports to forward control frames.
  void on_stop(std::function<void()> fn) { add_hook(stop_hooks_, std::move(fn), is_stopped()); }
  void on_kill(std::function<void()> fn) { add_hook(kill_hooks_, std::move(fn), is_killed()); }

  /// Routes stop/kill continuations (event waiters AND registered hooks)
  /// through `post` instead of running them inline on the signalling thread.
  /// Transports set this so control frames arriving on I/O threads never run
  /// engine continuations there.
  using PostFn = std::function<void(std::function<void()>)>;
  void set_signal_executor(PostFn post) {
    stopped_event_.set_resume_hook(
        [post](std::coroutine_handle<> h) { post([h] { h.resume(); }); });
    killed_event_.set_resume_hook(
        [post](std::coroutine_handle<> h) { post([h] { h.resume(); }); });
    std::lock_guard lock(mutex_);
    signal_post_ = std::move(post);
  }

  static std::string generate_id() {
    std::random_device rd;
    std::mt19937_64 gen((static_cast<uint64_t>(rd()) << 32) ^ rd());
    char buf[33];
    snprintf(buf, sizeof(buf), "%016llx%016llx", static_cast<unsigned long long>(gen()),
             static_cast<unsigned long long>(gen()));
    return std::string(buf);
  }

 private:
  void add_hook(std::list<std::function<void()>>& hooks, std::function<void()> fn, bool fired) {
    {
      std::lock_guard lock(mutex_);
      if (!fired) {
        hooks.push_back(std::move(fn));
        return;
      }
    }
    fn();
  }

  void run_hooks(std::list<std::function<void()>>& hooks) {
    std::list<std::function<void()>> to_run;
    PostFn post;
    {
      std::lock_guard lock(mutex_);
      to_run.swap(hooks);
      post = signal_post_;
    }
    for (auto& fn : to_run) {
      if (post) {
        post(std::move(fn));
      } else {
        fn();
      }
    }
  }

  std::string id_;
  std::atomic<bool> stopped_{false};
  std::atomic<bool> killed_{false};
  coro::AsyncEvent stopped_event_;
  coro::AsyncEvent killed_event_;
  std::mutex mutex_;
  PostFn signal_post_;
  std::list<std::function<void()>> stop_hooks_;
  std::list<std::function<void()>> kill_hooks_;
};

/// Type-erased key/value store carried by a request context across pipeline
/// stages (Dynamo's pipeline::registry). Shared entries are read-many;
/// unique entries can be taken (moved out) exactly once.
class ContextRegistry {
 public:
  template <typename U>
  void insert_shared(const std::string& key, U value) {
    std::lock_guard lock(mutex_);
    entries_[key] = Entry{std::make_shared<U>(std::move(value)), typeid(U), false};
  }

  template <typename U>
  void insert_unique(const std::string& key, U value) {
    std::lock_guard lock(mutex_);
    entries_[key] = Entry{std::make_shared<U>(std::move(value)), typeid(U), true};
  }

  /// Shared lookup; nullptr if missing or the type does not match.
  template <typename V>
  std::shared_ptr<V> get(const std::string& key) const {
    std::lock_guard lock(mutex_);
    auto it = entries_.find(key);
    if (it == entries_.end() || it->second.type != typeid(V)) return nullptr;
    return std::static_pointer_cast<V>(it->second.value);
  }

  /// Removes and returns a unique entry; nullopt if missing/mismatched.
  template <typename V>
  std::optional<V> take_unique(const std::string& key) {
    std::lock_guard lock(mutex_);
    auto it = entries_.find(key);
    if (it == entries_.end() || !it->second.unique || it->second.type != typeid(V)) {
      return std::nullopt;
    }
    auto value = std::static_pointer_cast<V>(it->second.value);
    entries_.erase(it);
    return std::move(*value);
  }

 private:
  struct Entry {
    std::shared_ptr<void> value;
    std::type_index type;
    bool unique = false;
    Entry() : type(typeid(void)) {}
    Entry(std::shared_ptr<void> v, std::type_index t, bool u)
        : value(std::move(v)), type(t), unique(u) {}
  };
  mutable std::mutex mutex_;
  std::unordered_map<std::string, Entry> entries_;
};

/// Request payload + its engine context. Moves through pipeline stages via
/// transfer()/map(), preserving the controller, request id, registry and
/// stage trail.
template <typename T>
class Context {
 public:
  explicit Context(T payload)
      : payload_(std::move(payload)), controller_(std::make_shared<Controller>()) {}
  Context(T payload, std::string id)
      : payload_(std::move(payload)), controller_(std::make_shared<Controller>(std::move(id))) {}
  Context(T payload, std::shared_ptr<Controller> controller)
      : payload_(std::move(payload)), controller_(std::move(controller)) {}

  Context(Context&&) noexcept = default;
  Context& operator=(Context&&) noexcept = default;

  const std::string& id() const { return controller_->id(); }
  const T& payload() const { return payload_; }
  T& payload() { return payload_; }

  std::shared_ptr<Controller> controller() const { return controller_; }
  ContextPtr context() const { return controller_; }

  /// Typed metadata carried alongside the payload (created lazily).
  ContextRegistry& registry() {
    if (!registry_) registry_ = std::make_shared<ContextRegistry>();
    return *registry_;
  }

  /// Names of the pipeline stages this request has passed through.
  const std::vector<std::string>& stages() const { return stages_; }
  void add_stage(std::string name) { stages_.push_back(std::move(name)); }

  /// Moves the context onto a new payload, returning the old payload.
  template <typename U>
  std::pair<T, Context<U>> transfer(U new_payload) && {
    Context<U> next(std::move(new_payload), std::move(controller_));
    next.registry_ = std::move(registry_);
    next.stages_ = std::move(stages_);
    return {std::move(payload_), std::move(next)};
  }

  /// Maps the payload, keeping the context (Dynamo's Context::map).
  template <typename F>
  auto map(F&& f) && -> Context<std::invoke_result_t<F, T>> {
    using U = std::invoke_result_t<F, T>;
    Context<U> next(std::forward<F>(f)(std::move(payload_)), std::move(controller_));
    next.registry_ = std::move(registry_);
    next.stages_ = std::move(stages_);
    return next;
  }

  std::pair<T, std::shared_ptr<Controller>> into_parts() && {
    return {std::move(payload_), std::move(controller_)};
  }

 private:
  template <typename U>
  friend class Context;

  T payload_;
  std::shared_ptr<Controller> controller_;
  std::shared_ptr<ContextRegistry> registry_;
  std::vector<std::string> stages_;
};

/// Streaming response: an async generator paired with the request context.
template <typename U>
class ResponseStream {
 public:
  ResponseStream() = default;
  ResponseStream(coro::AsyncGenerator<U> stream, ContextPtr ctx)
      : stream_(std::move(stream)), context_(std::move(ctx)) {}

  auto next() { return stream_.next(); }
  const ContextPtr& context() const { return context_; }
  coro::AsyncGenerator<U>& generator() { return stream_; }

 private:
  coro::AsyncGenerator<U> stream_;
  ContextPtr context_;
};

template <typename T>
using SingleIn = Context<T>;
template <typename U>
using ManyOut = ResponseStream<U>;

/// The generate-style engine interface (Dynamo's AsyncEngine).
template <typename Req, typename Resp>
class AsyncEngine {
 public:
  virtual ~AsyncEngine() = default;
  virtual coro::Task<ManyOut<Resp>> generate(SingleIn<Req> request) = 0;
};

template <typename Req, typename Resp>
using EnginePtr = std::shared_ptr<AsyncEngine<Req, Resp>>;

// --- unary (SingleIn → SingleOut) -------------------------------------------
//
// The wire protocol carries response_type = "single_out" | "many_out"; a
// unary call is a single-item stream on the transport. make_unary_engine
// adapts a value-returning handler so it can be served like any engine, and
// Client::unary() consumes the single item.

/// Handler shape for unary endpoints.
template <typename Req, typename Resp>
using UnaryFn = std::function<coro::Task<Resp>(SingleIn<Req>)>;

namespace detail {

template <typename Resp>
coro::AsyncGenerator<Resp> one_item_stream(Resp value) {
  co_yield value;
}

template <typename Req, typename Resp>
class UnaryEngineAdapter final : public AsyncEngine<Req, Resp> {
 public:
  explicit UnaryEngineAdapter(UnaryFn<Req, Resp> fn) : fn_(std::move(fn)) {}

  coro::Task<ManyOut<Resp>> generate(SingleIn<Req> request) override {
    ContextPtr ctx = request.context();
    Resp value = co_await fn_(std::move(request));
    co_return ManyOut<Resp>(one_item_stream<Resp>(std::move(value)), std::move(ctx));
  }

 private:
  UnaryFn<Req, Resp> fn_;
};

}  // namespace detail

/// Wraps a value-returning handler into an engine producing a one-item stream.
template <typename Req, typename Resp>
EnginePtr<Req, Resp> make_unary_engine(UnaryFn<Req, Resp> fn) {
  return std::make_shared<detail::UnaryEngineAdapter<Req, Resp>>(std::move(fn));
}

}  // namespace dynamo::pipeline
