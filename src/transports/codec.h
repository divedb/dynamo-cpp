// SPDX-License-Identifier: Apache-2.0
//
// Two-part message framing: | u64 header_len | u64 body_len | u64 checksum |
// header bytes | body bytes |. Header carries control/JSON metadata, body the
// payload. Checksum (FNV-1a 64) covers header+body.

#pragma once

#include <cstdint>
#include <optional>
#include <string>

namespace dynamo::transports {

struct TwoPartMessage {
  std::string header;
  std::string data;

  static TwoPartMessage from_header(std::string h) { return {std::move(h), {}}; }
  static TwoPartMessage from_data(std::string d) { return {{}, std::move(d)}; }
  static TwoPartMessage from_parts(std::string h, std::string d) {
    return {std::move(h), std::move(d)};
  }

  bool has_header() const { return !header.empty(); }
  bool has_data() const { return !data.empty(); }
};

class TwoPartCodec {
 public:
  static constexpr size_t kPreludeSize = 24;

  explicit TwoPartCodec(std::optional<size_t> max_message_size = std::nullopt)
      : max_message_size_(max_message_size) {}

  /// Serializes prelude+parts. Throws std::length_error if over the limit.
  std::string encode(const TwoPartMessage& msg) const;

  /// Parses one full message. Throws std::runtime_error on checksum/format errors.
  TwoPartMessage decode(std::string_view buffer) const;

  static uint64_t checksum(std::string_view header, std::string_view data);

 private:
  std::optional<size_t> max_message_size_;
};

}  // namespace dynamo::transports
