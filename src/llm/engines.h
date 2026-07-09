// SPDX-License-Identifier: Apache-2.0
//
// Engine seam and test engines — Dynamo's engines.rs shape plus dynamo-run's
// echo engines. Real engine adapters (vllm/sglang/trtllm/...) plug in behind
// ExecutionContext (see backend.h); the echo engine is what integration
// tests and demos run against.

#pragma once

#include <chrono>

#include "llm/backend.h"

namespace dynamo::llm {

struct EchoEngineOptions {
  /// Demo pacing between token deltas (blocks the emitting pool thread; keep
  /// at zero in tests). dynamo-run's TOKEN_ECHO_DELAY equivalent.
  std::chrono::milliseconds token_delay{0};
};

/// Engine that echoes the pre-processed request's token ids back one delta
/// at a time (token-only: the Backend operator detokenizes), then finishes
/// with a stop delta. Rust: dynamo-run output/echo_core.rs — extended here to
/// honor stop_conditions.max_tokens (length finish) and context stop.
ExecutionContext make_echo_engine_core(EchoEngineOptions options = {});

}  // namespace dynamo::llm
