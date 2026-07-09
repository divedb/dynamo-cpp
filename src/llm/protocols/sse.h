// SPDX-License-Identifier: Apache-2.0
//
// Server-Sent Events codec — Dynamo's protocols::codec (SseLineCodec).
// Incremental decoder: feed byte chunks, get complete Messages out; the
// encoder side (for the HTTP frontend) renders a Message back to SSE text.
//
// Behavior mirrors the Rust implementation: line-based parsing, multi-line
// `data:` accumulation joined with '\n', `:` comments (trimmed), `id` ignored
// when it contains NUL, `retry` and unknown fields ignored, and the literal
// `data: [DONE]` sentinel dropped (it does not surface as a message).

#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "pipeline/annotated.h"

namespace dynamo::llm {

inline constexpr std::string_view kSseDoneSentinel = "[DONE]";

/// A parsed SSE event.
struct SseMessage {
  std::optional<std::string> id;
  std::optional<std::string> event;
  std::optional<std::string> data;
  std::optional<std::vector<std::string>> comments;

  bool empty() const { return !id && !event && !data && !comments; }

  /// Parses the data field as JSON into T; throws std::runtime_error if the
  /// message has no data or the data fails to parse/convert.
  template <typename T>
  T decode_data() const {
    if (!data.has_value()) {
      throw std::runtime_error("no data: message to decode");
    }
    try {
      return nlohmann::json::parse(*data).get<T>();
    } catch (const std::exception& e) {
      throw std::runtime_error(std::string("failed to deserialize data: ") + e.what());
    }
  }
};

/// Incremental SSE decoder. Not thread-safe; one instance per stream.
class SseDecoder {
 public:
  /// Consumes a chunk of bytes and returns all events completed by it.
  std::vector<SseMessage> feed(std::string_view chunk);

  /// Signals end-of-stream: processes any partial trailing line and flushes
  /// a final event if fields are pending.
  std::optional<SseMessage> finish();

 private:
  void process_line(std::string_view line, std::vector<SseMessage>& out);
  std::optional<SseMessage> take_pending();

  std::string line_buffer_;
  std::string data_;
  std::string event_;
  std::string id_;
  std::vector<std::string> comments_;
};

/// Renders a message as SSE wire text (multi-line data is split into
/// consecutive `data:` lines), terminated by the blank line.
std::string encode_sse(const SseMessage& message);

/// The `data: [DONE]\n\n` terminator used by OpenAI-style streams.
std::string encode_sse_done();

/// Converts a parsed SSE message into an Annotated<R> envelope, mirroring
/// Rust's TryFrom<Message> + convert_sse_stream: an `event: error` message
/// or an undecodable data payload becomes an error annotation.
template <typename R>
pipeline::Annotated<R> annotated_from_sse(const SseMessage& message) {
  if (message.event.has_value() && *message.event == "error") {
    std::string text = "`event: error` detected, but no error message found";
    if (message.comments.has_value() && !message.comments->empty()) {
      text.clear();
      for (size_t i = 0; i < message.comments->size(); ++i) {
        if (i > 0) text += '\n';
        text += (*message.comments)[i];
      }
    }
    return pipeline::Annotated<R>::from_error(std::move(text));
  }

  pipeline::Annotated<R> annotated;
  if (message.data.has_value()) {
    try {
      annotated.data = message.decode_data<R>();
    } catch (const std::exception& e) {
      return pipeline::Annotated<R>::from_error(e.what());
    }
  }
  annotated.id = message.id;
  annotated.event = message.event;
  annotated.comment = message.comments;
  return annotated;
}

}  // namespace dynamo::llm
