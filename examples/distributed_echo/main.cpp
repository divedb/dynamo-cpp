#include <dynamo/runtime.h>
#include <dynamo/component.h>
#include <dynamo/discovery.h>
#include <dynamo/transport/grpc.h>
#include <spdlog/spdlog.h>

using namespace dynamo;

int main() {
    spdlog::set_level(spdlog::level::info);

    auto runtime = Runtime::create(RuntimeConfig::load());

    // Create component hierarchy
    auto ns = Namespace(runtime, "example");
    auto component = ns.component("echo_server");
    auto endpoint = component.endpoint("echo");

    spdlog::info("Namespace: {}", ns.name());
    spdlog::info("Component: {}", component.name());
    spdlog::info("Endpoint subject: {}", endpoint.subject());
    spdlog::info("Etcd path: {}", endpoint.etcd_path());

    // Demonstrate service discovery
    DiscoveryClient discovery(runtime);

    // Register an endpoint
    EndpointInfo info{
        .namespace_name = "example",
        .component_name = "echo_server",
        .endpoint_name = "echo",
        .addr = "127.0.0.1",
        .port = 50051,
    };

    auto lease_fut = discovery.create_lease(30);
    auto lease = std::move(lease_fut).get();
    spdlog::info("Created lease: {}", lease);

    auto reg = discovery.register_endpoint(info, lease);
    std::move(reg).get();

    // Discover endpoints
    auto eps = discovery.discover_endpoints("example", "echo_server", "echo").get();
    spdlog::info("Discovered {} endpoint(s)", eps.size());
    for (const auto& ep : eps) {
        spdlog::info("  -> {}:{}", ep.addr, ep.port);
    }

    // gRPC server
    transport::GrpcServer grpc_server(runtime, "0.0.0.0:50051");
    grpc_server.start();

    spdlog::info("Distributed echo example running. Press Ctrl+C to stop...");

    // Keep running
    std::this_thread::sleep_for(std::chrono::seconds(1));

    grpc_server.shutdown();
    discovery.shutdown();
    runtime->shutdown();
    spdlog::info("Distributed echo done.");

    return 0;
}
