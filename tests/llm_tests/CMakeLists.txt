add_executable(llm_tests
    protocols_test.cpp
    backend_test.cpp
    tokenizer_test.cpp
    router_test.cpp
)
target_link_libraries(llm_tests PRIVATE dynamo_llm Catch2::Catch2WithMain)
