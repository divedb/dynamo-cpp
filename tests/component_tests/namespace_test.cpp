#include <catch2/catch_test_macros.hpp>
#include <dynamo/runtime.h>
#include <dynamo/component.h>

using namespace dynamo;

TEST_CASE("Namespace creates with name and etcd prefix", "[component][namespace]") {
    auto runtime = Runtime::create(RuntimeConfig{});
    auto ns = Namespace(runtime, "test-ns");

    CHECK(ns.name() == "test-ns");
    CHECK(ns.etcd_prefix() == "/dynamo/namespaces/test-ns");

    runtime->shutdown();
}

TEST_CASE("Namespace creates component", "[component][namespace]") {
    auto runtime = Runtime::create(RuntimeConfig{});
    auto ns = Namespace(runtime, "ns");
    auto comp = ns.component("my-component");

    CHECK(comp.name() == "my-component");
    CHECK(comp.ns().name() == "ns");

    runtime->shutdown();
}
