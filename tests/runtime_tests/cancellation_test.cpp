#include <catch2/catch_test_macros.hpp>
#include <dynamo/cancellation.h>

using namespace dynamo;

TEST_CASE("CancellationToken defaults to not cancelled", "[cancellation]") {
    CancellationToken tok;
    CHECK_FALSE(tok.is_cancelled());
}

TEST_CASE("CancellationToken can be cancelled", "[cancellation]") {
    auto tok = CancellationToken::create_source();
    CHECK_FALSE(tok.is_cancelled());
    tok.cancel();
    CHECK(tok.is_cancelled());
}

TEST_CASE("CancellationToken never() never cancels", "[cancellation]") {
    auto tok = CancellationToken::never();
    CHECK_FALSE(tok.is_cancelled());
}

TEST_CASE("CancellationToken copy shares state", "[cancellation]") {
    auto tok1 = CancellationToken::create_source();
    auto tok2 = tok1;  // copy
    CHECK_FALSE(tok1.is_cancelled());
    CHECK_FALSE(tok2.is_cancelled());
    tok1.cancel();
    CHECK(tok1.is_cancelled());
    CHECK(tok2.is_cancelled());
}

TEST_CASE("CancellationToken calls callback on cancel", "[cancellation]") {
    auto tok = CancellationToken::create_source();
    bool called = false;
    tok.on_cancel([&]() { called = true; });
    CHECK_FALSE(called);
    tok.cancel();
    CHECK(called);
}
