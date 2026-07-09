// SPDX-License-Identifier: Apache-2.0
//
// Two-part codec hardening: roundtrips plus malformed/corrupt/oversize input.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>

#include <string>

#include "transports/codec.h"

using namespace dynamo::transports;

TEST_CASE("codec roundtrips all message shapes", "[codec]") {
  TwoPartCodec codec;
  for (auto& msg : {TwoPartMessage::from_parts("header", "data"),
                    TwoPartMessage::from_header("only-header"),
                    TwoPartMessage::from_data("only-data"), TwoPartMessage{}}) {
    auto decoded = codec.decode(codec.encode(msg));
    REQUIRE(decoded.header == msg.header);
    REQUIRE(decoded.data == msg.data);
  }

  // Binary payloads (embedded NULs) survive.
  std::string binary("\x00\x01\xff\x00zzz", 7);
  auto decoded = codec.decode(codec.encode(TwoPartMessage::from_data(binary)));
  REQUIRE(decoded.data == binary);
}

TEST_CASE("codec rejects malformed input", "[codec]") {
  TwoPartCodec codec;
  auto valid = codec.encode(TwoPartMessage::from_parts("hdr", "body"));

  SECTION("truncated prelude") {
    REQUIRE_THROWS(codec.decode(std::string_view(valid).substr(0, 10)));
  }
  SECTION("truncated body") {
    REQUIRE_THROWS(codec.decode(std::string_view(valid).substr(0, valid.size() - 2)));
  }
  SECTION("trailing garbage") {
    REQUIRE_THROWS(codec.decode(valid + "extra"));
  }
  SECTION("corrupt payload byte fails the checksum") {
    auto corrupted = valid;
    corrupted[TwoPartCodec::kPreludeSize + 1] ^= 0x40;
    REQUIRE_THROWS_WITH(codec.decode(corrupted),
                        "two-part message checksum mismatch");
  }
  SECTION("corrupt length field") {
    auto corrupted = valid;
    corrupted[7] = 0x7f;  // absurd header_len
    REQUIRE_THROWS(codec.decode(corrupted));
  }
  SECTION("declared size over the configured limit") {
    TwoPartCodec bounded(64);
    REQUIRE_THROWS(bounded.encode(
        TwoPartMessage::from_data(std::string(128, 'x'))));
  }
}
