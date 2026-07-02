add_executable(llm_minimal main.cpp)
target_link_libraries(llm_minimal PRIVATE dynamo_llm dynamo_runtime)
