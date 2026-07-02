#include <catch2/catch_test_macros.hpp>
#include <dynamo/runtime.h>
#include <dynamo/discovery.h>

using namespace dynamo;

TEST_CASE("DiscoveryClient creates lease", "[component][discovery]") {
    auto runtime = Runtime::create(RuntimeConfig{});
    DiscoveryClient discovery(runtime);

    auto lease = discovery.create_lease(30).get();
    CHECK(lease > 0);

    discovery.shutdown();
    runtime->shutdown();
}

TEST_CASE("DiscoveryClient registers and discovers endpoints", "[component][discovery]") {
    auto runtime = Runtime::create(RuntimeConfig{});
    DiscoveryClient discovery(runtime);

    EndpointInfo info;
    info.namespace_name = "test";
    info.component_name = "comp";
    info.endpoint_name = "ep";
    info.addr = "127.0.0.1";
    info.port = 50051;

    auto lease = discovery.create_lease(30).get();
    discovery.register_endpoint(info, lease).get();

    auto eps = discovery.discover_endpoints("test", "comp", "ep").get();
    CHECK(eps.size() == 1);
    CHECK(eps[0].addr == "127.0.0.1");
    CHECK(eps[0].port == 50051);

    discovery.shutdown();
    runtime->shutdown();
}

TEST_CASE("DiscoveryClient unregisters endpoints", "[component][discovery]") {
    auto runtime = Runtime::create(RuntimeConfig{});
    DiscoveryClient discovery(runtime);

    EndpointInfo info;
    info.namespace_name = "test";
    info.component_name = "comp";
    info.endpoint_name = "ep";
    info.addr = "127.0.0.1";
    info.port = 50051;

    auto lease = discovery.create_lease(30).get();
    discovery.register_endpoint(info, lease).get();
    discovery.unregister_endpoint(info).get();

    auto eps = discovery.discover_endpoints("test", "comp", "ep").get();
    CHECK(eps.empty());

    discovery.shutdown();
    runtime->shutdown();
}
