// SPDX-License-Identifier: Apache-2.0
//
// Typed endpoint client: watches the live instance set through the shared
// InstanceSource and routes requests — random (default), round-robin, or
// direct to a chosen instance id. One call = one unary request dispatched on
// the control plane, answered by a response stream on the data plane.

#pragma once

#include <atomic>
#include <random>

#include <spdlog/spdlog.h>

#include "component/distributed_runtime.h"
#include "runtime/timer.h"

namespace dynamo::component {

namespace detail {

/// Kills the request context if the consumer abandons the stream mid-flight,
/// so the worker stops producing (Dynamo's dropped-stream behavior).
struct AbandonGuard {
  pipeline::ContextPtr ctx;
  bool completed = false;
  ~AbandonGuard() {
    if (ctx && !completed) ctx->kill();
  }
};

template <typename Resp>
coro::AsyncGenerator<Resp> typed_stream(coro::Receiver<std::string> rx, pipeline::ContextPtr ctx) {
  AbandonGuard guard{std::move(ctx)};
  while (auto bytes = co_await rx.recv()) {
    Resp item;
    try {
      item = nlohmann::json::parse(*bytes).template get<Resp>();
    } catch (const std::exception& e) {
      spdlog::warn("client: dropping undecodable response item: {}", e.what());
      continue;
    }
    co_yield item;
  }
  guard.completed = true;
}

}  // namespace detail

template <typename Req, typename Resp>
class Client {
 public:
  static coro::Task<Client> create(Endpoint endpoint, CallOptions options = {}) {
    Client client(std::move(endpoint), options);
    client.source_ = co_await client.endpoint_.drt().instance_source(client.endpoint_);
    co_return client;
  }

  /// Blocks until at least one live instance exists.
  coro::Task<void> wait_for_instances() { co_await source_->wait_nonempty(); }

  std::vector<EndpointInfo> instances() const { return source_->snapshot(); }

  std::vector<int64_t> instance_ids() const {
    std::vector<int64_t> ids;
    for (auto& info : source_->snapshot()) ids.push_back(info.instance_id);
    return ids;
  }

  /// Routes to a random live instance.
  coro::Task<pipeline::ManyOut<Resp>> random(pipeline::SingleIn<Req> request) {
    auto live = source_->snapshot();
    if (live.empty()) throw no_instances();
    thread_local std::mt19937_64 rng{std::random_device{}()};
    co_return co_await call(std::move(request), live[rng() % live.size()]);
  }

  /// Routes round-robin over the live set.
  coro::Task<pipeline::ManyOut<Resp>> round_robin(pipeline::SingleIn<Req> request) {
    auto live = source_->snapshot();
    if (live.empty()) throw no_instances();
    uint64_t n = counter_->fetch_add(1);
    co_return co_await call(std::move(request), live[n % live.size()]);
  }

  /// Routes to a specific instance; fails if it is not live.
  coro::Task<pipeline::ManyOut<Resp>> direct(pipeline::SingleIn<Req> request,
                                             int64_t instance_id) {
    for (auto& info : source_->snapshot()) {
      if (info.instance_id == instance_id) {
        co_return co_await call(std::move(request), info);
      }
    }
    throw std::runtime_error("instance " + hex_id(instance_id) + " not found for endpoint " +
                             endpoint_.key_prefix());
  }

  /// Default policy (random), matching Dynamo's Client::generate.
  coro::Task<pipeline::ManyOut<Resp>> generate(pipeline::SingleIn<Req> request) {
    co_return co_await random(std::move(request));
  }

  /// Unary call (SingleIn → SingleOut): dispatches with
  /// response_type=single_out to a random live instance and resolves to the
  /// single response value. Throws if the worker streams nothing back.
  coro::Task<Resp> unary(pipeline::SingleIn<Req> request) {
    auto live = source_->snapshot();
    if (live.empty()) throw no_instances();
    thread_local std::mt19937_64 rng{std::random_device{}()};
    auto stream =
        co_await call(std::move(request), live[rng() % live.size()], "single_out");
    auto item = co_await stream.next();
    if (!item) throw std::runtime_error("unary call produced no response");
    while (co_await stream.next()) {
    }  // drain to the sentinel for a clean close
    co_return std::move(*item);
  }

  /// Scrapes the stats endpoint of one live instance.
  coro::Task<nlohmann::json> scrape_stats(int64_t instance_id) {
    for (auto& info : source_->snapshot()) {
      if (info.instance_id == instance_id) {
        co_return nlohmann::json::parse(
            transports::dispatch_query(info.address, info.subject + "/stats", "{}"));
      }
    }
    throw std::runtime_error("instance " + hex_id(instance_id) + " not found");
  }

  /// Type-erased engine view of this client (random routing): the caller
  /// side of a distributed pipeline segment, composable with
  /// pipeline::link()/operators.
  pipeline::EnginePtr<Req, Resp> as_engine() const {
    struct Adapter final : pipeline::AsyncEngine<Req, Resp> {
      explicit Adapter(Client client) : client(std::move(client)) {}
      coro::Task<pipeline::ManyOut<Resp>> generate(pipeline::SingleIn<Req> request) override {
        co_return co_await client.generate(std::move(request));
      }
      Client client;
    };
    return std::make_shared<Adapter>(*this);
  }

 private:
  explicit Client(Endpoint endpoint, CallOptions options = {})
      : endpoint_(std::move(endpoint)),
        options_(options),
        counter_(std::make_shared<std::atomic<uint64_t>>(0)) {}

  std::runtime_error no_instances() const {
    return std::runtime_error("no live instances for endpoint " + endpoint_.key_prefix());
  }

  coro::Task<pipeline::ManyOut<Resp>> call(pipeline::SingleIn<Req> request, EndpointInfo target,
                                           std::string response_type = "many_out") {
    auto [payload, controller] = std::move(request).into_parts();
    pipeline::ContextPtr ctx = controller;

    auto data_plane = endpoint_.drt().data_plane();

    // Register the return path first, then dispatch with its coordinates.
    auto registered =
        data_plane->register_response_stream(ctx, options_.recv_buffer_count);
    std::string subject = registered.info.subject;

    transports::DispatchHeader header;
    header.subject = target.subject;
    header.control.id = ctx->id();
    header.control.response_type = std::move(response_type);
    header.control.connection_info = std::move(registered.info);

    transports::dispatch_request(target.address, header, nlohmann::json(payload).dump(),
                                 options_.dispatch_timeout);

    // Bound the wait for the worker's call-home: a worker that acked but died
    // before connecting must not park us forever.
    uint64_t timer_id = TimerQueue::instance().schedule_at(
        TimerQueue::Clock::now() + options_.arrival_timeout,
        [tx = registered.arrival_tx]() mutable {
          tx.send(transports::StreamArrival{
              std::string("timed out waiting for the worker's call-home"), std::nullopt});
        });

    // generate() resolves only when the worker reports its outcome (prologue).
    auto arrival = co_await registered.arrival.recv();
    TimerQueue::instance().cancel(timer_id);
    if (!arrival) throw std::runtime_error("data plane shut down while awaiting response stream");
    if (arrival->error) {
      data_plane->deregister(subject);  // no-op if the worker already connected
      throw std::runtime_error("remote generate failed: " + *arrival->error);
    }

    co_return pipeline::ManyOut<Resp>(
        detail::typed_stream<Resp>(std::move(arrival->stream->data), ctx), ctx);
  }

  Endpoint endpoint_;
  CallOptions options_;
  std::shared_ptr<InstanceSource> source_;
  std::shared_ptr<std::atomic<uint64_t>> counter_;
};

template <typename Req, typename Resp>
coro::Task<Client<Req, Resp>> Endpoint::client(CallOptions options) const {
  // Not a coroutine: *this is copied into create()'s frame eagerly, so
  // endpoint temporaries are safe (see serve()).
  return Client<Req, Resp>::create(*this, options);
}

}  // namespace dynamo::component
