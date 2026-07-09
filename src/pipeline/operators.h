// SPDX-License-Identifier: Apache-2.0
//
// Composable pipeline graph (Dynamo's pipeline/nodes): an Operator sits
// between a frontend and a downstream engine, transforming requests on the
// way down (forward edge) and the response stream on the way up (backward
// edge). link() produces a plain engine, so operators compose:
//
//   auto engine = link(op_a, link(op_b, backend));   // local pipeline
//   auto remote = link(op_a, client.as_engine());    // distributed segment
//
// A distributed pipeline's segment boundary is exactly Dynamo's
// SegmentSink/SegmentSource pair: on the caller side a Client-as-engine, on
// the worker side Endpoint::serve() of the linked engine.

#pragma once

#include "pipeline/engine.h"

namespace dynamo::pipeline {

/// Bidirectional transform node. `generate` receives the request and the
/// downstream engine; implementations transform the request, delegate, and
/// transform the resulting stream.
template <typename ReqIn, typename RespOut, typename ReqOut, typename RespIn>
class Operator {
 public:
  virtual ~Operator() = default;
  virtual coro::Task<ManyOut<RespOut>> generate(SingleIn<ReqIn> request,
                                                EnginePtr<ReqOut, RespIn> next) = 0;
};

template <typename ReqIn, typename RespOut, typename ReqOut, typename RespIn>
using OperatorPtr = std::shared_ptr<Operator<ReqIn, RespOut, ReqOut, RespIn>>;

namespace detail {

template <typename ReqIn, typename RespOut, typename ReqOut, typename RespIn>
class LinkedEngine final : public AsyncEngine<ReqIn, RespOut> {
 public:
  LinkedEngine(OperatorPtr<ReqIn, RespOut, ReqOut, RespIn> op, EnginePtr<ReqOut, RespIn> next)
      : op_(std::move(op)), next_(std::move(next)) {}

  coro::Task<ManyOut<RespOut>> generate(SingleIn<ReqIn> request) override {
    co_return co_await op_->generate(std::move(request), next_);
  }

 private:
  OperatorPtr<ReqIn, RespOut, ReqOut, RespIn> op_;
  EnginePtr<ReqOut, RespIn> next_;
};

template <typename RespIn, typename RespOut>
coro::AsyncGenerator<RespOut> map_stream(ManyOut<RespIn> in,
                                         std::function<RespOut(RespIn)> map_fn) {
  while (auto item = co_await in.next()) {
    RespOut mapped = map_fn(std::move(*item));
    co_yield mapped;
  }
}

template <typename ReqIn, typename RespOut, typename ReqOut, typename RespIn>
class MapOperator final : public Operator<ReqIn, RespOut, ReqOut, RespIn> {
 public:
  MapOperator(std::string stage, std::function<ReqOut(ReqIn)> fwd,
              std::function<RespOut(RespIn)> bwd)
      : stage_(std::move(stage)), fwd_(std::move(fwd)), bwd_(std::move(bwd)) {}

  coro::Task<ManyOut<RespOut>> generate(SingleIn<ReqIn> request,
                                        EnginePtr<ReqOut, RespIn> next) override {
    if (!stage_.empty()) request.add_stage(stage_);
    auto downstream_request = std::move(request).map(fwd_);
    auto out = co_await next->generate(std::move(downstream_request));
    ContextPtr ctx = out.context();
    co_return ManyOut<RespOut>(map_stream<RespIn, RespOut>(std::move(out), bwd_),
                               std::move(ctx));
  }

 private:
  std::string stage_;
  std::function<ReqOut(ReqIn)> fwd_;
  std::function<RespOut(RespIn)> bwd_;
};

}  // namespace detail

/// Composes an operator with its downstream engine into a new engine.
template <typename ReqIn, typename RespOut, typename ReqOut, typename RespIn>
EnginePtr<ReqIn, RespOut> link(OperatorPtr<ReqIn, RespOut, ReqOut, RespIn> op,
                               EnginePtr<ReqOut, RespIn> next) {
  return std::make_shared<detail::LinkedEngine<ReqIn, RespOut, ReqOut, RespIn>>(std::move(op),
                                                                                std::move(next));
}

/// Functional operator: per-request forward map + per-item backward map
/// (Dynamo's PipelineNode role). `stage` is recorded on the context trail.
template <typename ReqIn, typename RespOut, typename ReqOut, typename RespIn>
OperatorPtr<ReqIn, RespOut, ReqOut, RespIn> make_map_operator(
    std::string stage, std::function<ReqOut(ReqIn)> forward,
    std::function<RespOut(RespIn)> backward) {
  return std::make_shared<detail::MapOperator<ReqIn, RespOut, ReqOut, RespIn>>(
      std::move(stage), std::move(forward), std::move(backward));
}

}  // namespace dynamo::pipeline
