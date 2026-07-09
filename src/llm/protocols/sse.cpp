// SPDX-License-Identifier: Apache-2.0

#include "llm/protocols/sse.h"

namespace dynamo::llm {

namespace {

std::string_view trim(std::string_view s) {
  while (!s.empty() && (s.front() == ' ' || s.front() == '\t')) s.remove_prefix(1);
  while (!s.empty() && (s.back() == ' ' || s.back() == '\t')) s.remove_suffix(1);
  return s;
}

}  // namespace

std::optional<SseMessage> SseDecoder::take_pending() {
  if (data_.empty() && event_.empty() && id_.empty() && comments_.empty()) {
    return std::nullopt;
  }
  SseMessage message;
  if (!data_.empty()) {
    // Multi-line data accumulates a trailing '\n'; the last one is stripped.
    if (data_.back() == '\n') data_.pop_back();
    if (!data_.empty()) message.data = std::move(data_);
  }
  if (!id_.empty()) message.id = std::move(id_);
  if (!event_.empty()) message.event = std::move(event_);
  if (!comments_.empty()) message.comments = std::move(comments_);
  data_.clear();
  id_.clear();
  event_.clear();
  comments_.clear();
  if (message.empty()) return std::nullopt;
  return message;
}

void SseDecoder::process_line(std::string_view line, std::vector<SseMessage>& out) {
  // Tolerate \r\n line endings.
  while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.remove_suffix(1);

  if (line.empty()) {
    if (auto message = take_pending()) out.push_back(std::move(*message));
    return;
  }

  if (line.front() == ':') {
    comments_.emplace_back(trim(line.substr(1)));
    return;
  }

  std::string_view field_name = line;
  std::string_view field_value;
  if (auto idx = line.find(':'); idx != std::string_view::npos) {
    field_name = line.substr(0, idx);
    field_value = line.substr(idx + 1);
    while (!field_value.empty() && field_value.front() == ' ') field_value.remove_prefix(1);
  }

  if (field_name == "event") {
    event_ = std::string(field_value);
  } else if (field_name == "data") {
    if (field_value != kSseDoneSentinel) {
      if (!data_.empty()) data_ += '\n';
      data_ += field_value;
    }
  } else if (field_name == "id") {
    if (field_value.find('\0') == std::string_view::npos) {
      id_ = std::string(field_value);
    }
  }
  // "retry" and unknown fields are ignored.
}

std::vector<SseMessage> SseDecoder::feed(std::string_view chunk) {
  std::vector<SseMessage> out;
  size_t start = 0;
  while (true) {
    size_t nl = chunk.find('\n', start);
    if (nl == std::string_view::npos) {
      line_buffer_.append(chunk.substr(start));
      break;
    }
    if (!line_buffer_.empty()) {
      line_buffer_.append(chunk.substr(start, nl - start));
      process_line(line_buffer_, out);
      line_buffer_.clear();
    } else {
      process_line(chunk.substr(start, nl - start), out);
    }
    start = nl + 1;
  }
  return out;
}

std::optional<SseMessage> SseDecoder::finish() {
  std::vector<SseMessage> out;
  if (!line_buffer_.empty()) {
    process_line(line_buffer_, out);
    line_buffer_.clear();
  }
  // A trailing non-blank line can never complete an event by itself.
  if (!out.empty()) return std::move(out.front());
  return take_pending();
}

std::string encode_sse(const SseMessage& message) {
  std::string out;
  if (message.comments.has_value()) {
    for (const auto& comment : *message.comments) {
      out += ": ";
      out += comment;
      out += '\n';
    }
  }
  if (message.id.has_value()) {
    out += "id: ";
    out += *message.id;
    out += '\n';
  }
  if (message.event.has_value()) {
    out += "event: ";
    out += *message.event;
    out += '\n';
  }
  if (message.data.has_value()) {
    size_t start = 0;
    while (true) {
      size_t nl = message.data->find('\n', start);
      out += "data: ";
      out += message.data->substr(start, nl == std::string::npos ? nl : nl - start);
      out += '\n';
      if (nl == std::string::npos) break;
      start = nl + 1;
    }
  }
  out += '\n';
  return out;
}

std::string encode_sse_done() {
  return "data: [DONE]\n\n";
}

}  // namespace dynamo::llm
