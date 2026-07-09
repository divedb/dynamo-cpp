// SPDX-License-Identifier: Apache-2.0

#include "transports/codec.h"

#include <cstring>
#include <stdexcept>

namespace dynamo::transports {

namespace {

void put_u64(std::string& out, uint64_t v) {
  for (int i = 7; i >= 0; --i) out.push_back(static_cast<char>((v >> (i * 8)) & 0xff));
}

uint64_t get_u64(const unsigned char* p) {
  uint64_t v = 0;
  for (int i = 0; i < 8; ++i) v = (v << 8) | p[i];
  return v;
}

}  // namespace

uint64_t TwoPartCodec::checksum(std::string_view header, std::string_view data) {
  // FNV-1a 64
  uint64_t h = 0xcbf29ce484222325ull;
  auto mix = [&h](std::string_view s) {
    for (unsigned char c : s) {
      h ^= c;
      h *= 0x100000001b3ull;
    }
  };
  mix(header);
  mix(data);
  return h;
}

std::string TwoPartCodec::encode(const TwoPartMessage& msg) const {
  size_t total = kPreludeSize + msg.header.size() + msg.data.size();
  if (max_message_size_ && total > *max_message_size_) {
    throw std::length_error("two-part message exceeds max size");
  }
  std::string out;
  out.reserve(total);
  put_u64(out, msg.header.size());
  put_u64(out, msg.data.size());
  put_u64(out, checksum(msg.header, msg.data));
  out.append(msg.header);
  out.append(msg.data);
  return out;
}

TwoPartMessage TwoPartCodec::decode(std::string_view buffer) const {
  if (buffer.size() < kPreludeSize) throw std::runtime_error("two-part message truncated prelude");
  auto* p = reinterpret_cast<const unsigned char*>(buffer.data());
  uint64_t header_len = get_u64(p);
  uint64_t body_len = get_u64(p + 8);
  uint64_t expected = get_u64(p + 16);
  if (buffer.size() != kPreludeSize + header_len + body_len) {
    throw std::runtime_error("two-part message length mismatch");
  }
  TwoPartMessage msg;
  msg.header.assign(buffer.substr(kPreludeSize, header_len));
  msg.data.assign(buffer.substr(kPreludeSize + header_len, body_len));
  if (checksum(msg.header, msg.data) != expected) {
    throw std::runtime_error("two-part message checksum mismatch");
  }
  return msg;
}

}  // namespace dynamo::transports
