// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace dynamo::logging {

/// Initializes spdlog from the DYN_LOG env var (trace|debug|info|warn|error).
/// Safe to call more than once.
void init();

}  // namespace dynamo::logging
