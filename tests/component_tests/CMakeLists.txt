add_executable(component_tests
    namespace_test.cpp
    component_test.cpp
    endpoint_test.cpp
    discovery_test.cpp
)
target_link_libraries(component_tests PRIVATE dynamo_runtime Catch2::Catch2WithMain)
