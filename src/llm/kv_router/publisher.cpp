// SPDX-License-Identifier: Apache-2.0

#include "llm/kv_router/publisher.h"

namespace dynamo::llm::kv {

namespace {

/// Unary metrics endpoint: replies with the current snapshot.
class KvLoadEndpointHandler final
    : public pipeline::AsyncEngine<nlohmann::json,
                                   pipeline::Annotated<ForwardPassMetrics>> {
 public:
  explicit KvLoadEndpointHandler(std::shared_ptr<const KvMetricsPublisher> publisher)
      : publisher_(std::move(publisher)) {}

  coro::Task<pipeline::ManyOut<pipeline::Annotated<ForwardPassMetrics>>> generate(
      pipeline::SingleIn<nlohmann::json> request) override {
    auto [ignored, controller] = std::move(request).into_parts();
    (void)ignored;
    pipeline::ContextPtr context = controller;
    co_return pipeline::ManyOut<pipeline::Annotated<ForwardPassMetrics>>(
        one(publisher_->current()), std::move(context));
  }

 private:
  static coro::AsyncGenerator<pipeline::Annotated<ForwardPassMetrics>> one(
      ForwardPassMetrics metrics) {
    co_yield pipeline::Annotated<ForwardPassMetrics>::from_data(std::move(metrics));
  }

  std::shared_ptr<const KvMetricsPublisher> publisher_;
};

/// Free coroutine so the frame owns the publisher handle (member coroutines
/// on shared_ptr targets dangle; see docs/architecture.md).
coro::Task<void> serve_metrics_endpoint(std::shared_ptr<const KvMetricsPublisher> self,
                                        component::Component component,
                                        std::optional<discovery::Lease> lease) {
  auto handler = std::make_shared<KvLoadEndpointHandler>(self);
  component::ServeOptions options;
  options.lease = std::move(lease);
  options.stats_handler = [self](const std::string&) {
    return nlohmann::json(self->current()).dump();
  };
  co_await component.endpoint(kKvMetricsEndpoint)
      .serve<nlohmann::json, pipeline::Annotated<ForwardPassMetrics>>(handler, options);
}

}  // namespace

coro::Task<void> KvMetricsPublisher::create_endpoint(
    component::Component component, std::optional<discovery::Lease> lease) const {
  return serve_metrics_endpoint(shared_from_this(), std::move(component), std::move(lease));
}

}  // namespace dynamo::llm::kv
