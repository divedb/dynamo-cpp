#include <catch2/catch_test_macros.hpp>
#include <dynamo/runtime.h>
#include <dynamo/component.h>

using namespace dynamo;

TEST_CASE("Endpoint generates subject and path", "[component][endpoint]") {
    auto runtime = Runtime::create(RuntimeConfig{});
    auto ns = Namespace(runtime, "test");
    auto comp = ns.component("my-service");
    auto ep = comp.endpoint("my-endpoint");

    auto subject = ep.subject();
    CHECK_FALSE(subject.empty());
    CHECK(ep.etcd_path().find("my-endpoint") != std::string::npos);

    runtime->shutdown();
}

TEST_CASE("Endpoint can register service", "[component][endpoint]") {
    auto runtime = Runtime::create(RuntimeConfig{});
    auto ns = Namespace(runtime, "test");
    auto comp = ns.component("svc");
    auto ep = comp.endpoint("ep");

    auto result = ep.register_service(50051).get();
    CHECK(result.isReady());

    runtime->shutdown();
}
