#include <catch2/catch_test_macros.hpp>
#include <dynamo/transport/etcd.h>

using namespace dynamo::transport;

TEST_CASE("EtcdClient put and get", "[network][etcd]") {
    EtcdClient client;

    auto ok = client.put("/test/key", "value").get();
    CHECK(ok);

    auto val = client.get("/test/key").get();
    CHECK(val == "value");
}

TEST_CASE("EtcdClient delete", "[network][etcd]") {
    EtcdClient client;
    client.put("/test/todelete", "will be deleted").get();

    auto deleted = client.del("/test/todelete").get();
    CHECK(deleted);

    auto val = client.get("/test/todelete").get();
    CHECK(val.empty());
}

TEST_CASE("EtcdClient prefix query", "[network][etcd]") {
    EtcdClient client;
    client.put("/test/prefix/a", "1").get();
    client.put("/test/prefix/b", "2").get();
    client.put("/test/prefix/c", "3").get();

    auto results = client.get_prefix("/test/prefix/").get();
    CHECK(results.size() == 3);
}

TEST_CASE("EtcdClient lease management", "[network][etcd]") {
    EtcdClient client;

    auto lease = client.grant_lease(60).get();
    CHECK(lease > 0);

    auto alive = client.keep_alive_lease(lease).get();
    CHECK(alive);

    auto revoked = client.revoke_lease(lease).get();
    CHECK(revoked);
}

TEST_CASE("EtcdClient put with lease", "[network][etcd]") {
    EtcdClient client;

    auto lease = client.grant_lease(30).get();
    auto ok = client.put_with_lease("/test/lease-key", "leased value", lease).get();
    CHECK(ok);

    auto val = client.get("/test/lease-key").get();
    CHECK(val == "leased value");
}
