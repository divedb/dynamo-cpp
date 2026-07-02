#include <catch2/catch_test_macros.hpp>
#include <dynamo/runtime.h>
#include <dynamo/component.h>

using namespace dynamo;

TEST_CASE("Component creates endpoints and paths", "[component]") {
    auto runtime = Runtime::create(RuntimeConfig{});
    auto ns = Namespace(runtime, "prod");
    auto comp = ns.component("llm-server");

    CHECK(comp.name() == "llm-server");
    CHECK(comp.etcd_path() == "/dynamo/namespaces/prod/components/llm-server");

    auto ep = comp.endpoint("generate");
    CHECK(ep.name() == "generate");
    CHECK(ep.etcd_path() ==
          "/dynamo/namespaces/prod/components/llm-server/endpoints/generate");

    runtime->shutdown();
}
