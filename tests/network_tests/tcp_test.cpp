#include <catch2/catch_test_macros.hpp>
#include <dynamo/runtime.h>
#include <dynamo/transport/tcp.h>

#include <thread>

using namespace dynamo;
using namespace dynamo::transport;

TEST_CASE("TcpServer starts and stops", "[network][tcp]") {
    auto runtime = Runtime::create(RuntimeConfig{});
    TcpServer server(runtime, 0);

    server.start();
    CHECK(server.is_running());
    CHECK(server.port() > 0);

    server.shutdown();
    CHECK_FALSE(server.is_running());

    runtime->shutdown();
}

TEST_CASE("TcpClient connect to server", "[network][tcp]") {
    auto runtime = Runtime::create(RuntimeConfig{});
    TcpServer server(runtime, 0);
    server.start();

    TcpClient client(runtime);
    bool connected = client.connect("127.0.0.1", server.port());
    CHECK(connected);

    client.disconnect();
    server.shutdown();
    runtime->shutdown();
}
