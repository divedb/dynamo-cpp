// SPDX-License-Identifier: Apache-2.0
//
// Endpoint serving: EngineIngress adapts a typed AsyncEngine to control-plane
// push handling — decode, rebuild the request context (propagating the id),
// call home, prologue, pump the response stream, honour stop/kill.

#pragma once

#include <atomic>

#include <spdlog/spdlog.h>

#include "component/distributed_runtime.h"

namespace dynamo::component {

template <typename Req, typename Resp>
class EngineIngress final : public transports::PushHandler {
 public:
  struct Stats {
    std::atomic<uint64_t> requests{0};
    std::atomic<uint64_t> errors{0};
    std::atomic<uint64_t> inflight{0};
  };

  EngineIngress(pipeline::EnginePtr<Req, Resp> engine, Runtime runtime)
      : engine_(std::move(engine)), runtime_(std::move(runtime)) {}

  coro::Task<void> handle(transports::RequestControlMessage control,
                          std::string payload) override {
    stats_->requests.fetch_add(1);
    stats_->inflight.fetch_add(1);
    struct InflightGuard {
      std::shared_ptr<Stats> stats;
      ~InflightGuard() { stats->inflight.fetch_sub(1); }
    } guard{stats_};


    // Rebuild the request context with the propagated id; the requester can
    // control it remotely via data-plane control frames. Stop/kill
    // continuations run on the primary pool, never on transport threads.
    auto controller = std::make_shared<pipeline::Controller>(control.id);
    controller->set_signal_executor(
        [&pool = runtime_.primary()](std::function<void()> fn) { pool.post(std::move(fn)); });

    auto sender = transports::StreamSender::connect(controller, control.connection_info,
                                                    &runtime_.primary());
    if (!sender) {
      spdlog::warn("ingress: cannot reach requester for {}", control.id);
      stats_->errors.fetch_add(1);
      co_return;
    }

    if (control.request_type != "single_in") {
      // ManyIn request streams: declared but unimplemented in Dynamo too
      // (process_request_stream is a stub). Reject explicitly via the
      // prologue so the caller fails fast instead of timing out.
      stats_->errors.fetch_add(1);
      sender->send_prologue("request_type '" + control.request_type +
                            "' is not supported (single_in only)");
      sender->finish();
      co_return;
    }

    Req request_payload;
    try {
      request_payload = nlohmann::json::parse(payload).template get<Req>();
    } catch (const std::exception& e) {
      stats_->errors.fetch_add(1);
      sender->send_prologue(std::string("invalid request payload: ") + e.what());
      sender->finish();
      co_return;
    }

    pipeline::ManyOut<Resp> stream;
    try {
      stream = co_await engine_->generate(
          pipeline::SingleIn<Req>(std::move(request_payload), controller));
    } catch (const std::exception& e) {
      stats_->errors.fetch_add(1);
      sender->send_prologue(std::string(e.what()));
      sender->finish();
      co_return;
    }

    sender->send_prologue(std::nullopt);

    // A unary requester consumes exactly one item; anything further the
    // engine would produce is cut off (stream teardown ends the producer).
    const bool unary = control.response_type == "single_out";

    while (auto item = co_await stream.next()) {
      if (controller->is_killed()) break;
      std::string bytes = nlohmann::json(*item).dump();
      if (!sender->send(std::move(bytes))) {
        // Requester is gone: ask the engine to stop producing.
        controller->stop_generating();
        break;
      }
      if (unary) {
        controller->stop_generating();
        break;
      }
    }
    sender->finish();
  }

  nlohmann::json stats_json() const {
    return {{"requests", stats_->requests.load()},
            {"errors", stats_->errors.load()},
            {"inflight", stats_->inflight.load()}};
  }

 private:
  pipeline::EnginePtr<Req, Resp> engine_;
  Runtime runtime_;
  std::shared_ptr<Stats> stats_ = std::make_shared<Stats>();
};

namespace detail {

/// Runs one queued work item; owns the ingress for the handler's lifetime
/// (same pattern as the control plane's run_handler).
template <typename Req, typename Resp>
coro::Task<void> run_queued_item(std::shared_ptr<EngineIngress<Req, Resp>> ingress,
                                 transports::RequestControlMessage control,
                                 std::string payload) {
  co_await ingress->handle(std::move(control), std::move(payload));
}

/// Queue-group pump: receives broker-balanced work items from the endpoint's
/// queue subject and feeds them to the same ingress that serves
/// instance-addressed dispatches. Ends when the subscription channel closes
/// (lease death via the caller's registration, or discovery shutdown).
template <typename Req, typename Resp>
coro::Task<void> queue_pump(coro::Receiver<discovery::Event> events,
                            std::shared_ptr<EngineIngress<Req, Resp>> ingress,
                            Runtime runtime) {
  while (auto event = co_await events.recv()) {
    transports::RequestControlMessage control;
    std::string payload;
    try {
      auto envelope = nlohmann::json::parse(event->payload);
      control = envelope.at("control").template get<transports::RequestControlMessage>();
      payload = envelope.at("payload").template get<std::string>();
    } catch (const std::exception& e) {
      spdlog::warn("queue pump: dropping malformed work item: {}", e.what());
      continue;
    }
    // Concurrent like control-plane dispatches; each item runs detached.
    runtime.spawn(run_queued_item(ingress, std::move(control), std::move(payload)));
  }
}

/// Free coroutine taking the endpoint BY VALUE: member coroutines on value
/// handles capture `this`, which dangles when called on a temporary
/// (endpoint("x").serve(...) is legal and must stay safe).
template <typename Req, typename Resp>
coro::Task<void> serve_impl(Endpoint endpoint, pipeline::EnginePtr<Req, Resp> engine,
                            ServeOptions options) {
  auto drt = endpoint.drt();
  auto lease = options.lease ? *options.lease : co_await drt.primary_lease();
  auto control_plane = drt.control_plane();
  std::string subject = endpoint.subject_for(lease.id);
  std::string stats_subject = subject + "/stats";

  auto ingress =
      std::make_shared<EngineIngress<Req, Resp>>(std::move(engine), drt.runtime());
  control_plane->register_handler(subject, ingress);
  control_plane->register_query(
      stats_subject, [ingress, stats_handler = options.stats_handler](
                         const std::string& payload) {
        auto stats = ingress->stats_json();
        if (stats_handler) {
          // Merge as JSON when the handler returns JSON; raw string otherwise.
          std::string custom = stats_handler(payload);
          try {
            stats["custom"] = nlohmann::json::parse(custom);
          } catch (const std::exception&) {
            stats["custom"] = std::move(custom);
          }
        }
        return stats.dump();
      });

  // Queue-group membership: subscribe to the endpoint's shared queue subject
  // and pump broker-balanced items into the same ingress. The lease token
  // closes the subscription channel, ending the pump with the instance.
  std::optional<CancellationRegistration> queue_close;
  coro::Receiver<discovery::Event> queue_events;
  if (options.queue_group) {
    auto stream = co_await drt.discovery()->subscribe(endpoint.queue_subject());
    queue_events = stream.events;  // kept for teardown on registration failure
    queue_close.emplace(lease.token.register_callback(
        [events = stream.events]() mutable { events.close(); }));
    drt.runtime().spawn_background(
        detail::queue_pump<Req, Resp>(std::move(stream.events), ingress, drt.runtime()));
  }

  EndpointInfo info;
  info.namespace_name = endpoint.component().ns().name();
  info.component = endpoint.component().name();
  info.endpoint = endpoint.name();
  info.instance_id = lease.id;
  info.address = control_plane->address();
  info.subject = subject;
  info.description =
      options.description.empty()
          ? "dynamo-cpp endpoint " + endpoint.component().path() + "." + endpoint.name()
          : options.description;
  info.version = options.version;

  try {
    // Atomic create bound to the lease: the instance key vanishes with it.
    co_await drt.discovery()->kv_create(endpoint.instance_key(lease.id),
                                        nlohmann::json(info).dump(), lease.id);
  } catch (...) {
    control_plane->unregister_handler(subject);
    control_plane->unregister_query(stats_subject);
    queue_events.close();  // ends the pump; no-op when not in a queue group
    throw;
  }

  spdlog::info("serving {} (instance {})", subject, hex_id(lease.id));
  co_await lease.token.cancelled();

  spdlog::info("stopping {} (instance {})", subject, hex_id(lease.id));
  control_plane->unregister_handler(subject);
  control_plane->unregister_query(stats_subject);
}

}  // namespace detail

template <typename Req, typename Resp>
coro::Task<void> Endpoint::serve(pipeline::EnginePtr<Req, Resp> engine,
                                 ServeOptions options) const {
  // Not a coroutine: *this is copied into the impl frame before returning,
  // so serving on a temporary Endpoint is safe.
  return detail::serve_impl<Req, Resp>(*this, std::move(engine), std::move(options));
}

}  // namespace dynamo::component
