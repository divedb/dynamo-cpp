add_executable(network_tests
    etcd_client_test.cpp
    tcp_test.cpp
)
target_link_libraries(network_tests PRIVATE dynamo_runtime Catch2::Catch2WithMain)
